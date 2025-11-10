"""
Triad Engines with offline LLM + RAG integration.

CognitiveEngine:
 - loads an open medical LLM (BioMistral example) locally via HuggingFace transformers
 - performs RAG: local FAISS retrieval using sentence-transformers embeddings
 - produces answers with soft confidence estimate from token scores

EmpathicEngine:
 - language detection, simple sentiment proxy (can be replaced by stronger detectors)

EthicalEngine:
 - loads rules.json, validates responses by pattern match + confidence threshold,
   and sends escalations to utils.add_escalation when needed.

Requires: transformers, accelerate, bitsandbytes (optional quantization), sentence-transformers, faiss-cpu
"""

import os
import json
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
from sentence_transformers import SentenceTransformer
import faiss
import utils

# ---------------------------
# Cognitive Engine
# ---------------------------
class CognitiveEngine:
    def __init__(self,
                 model_name="BioMistral/BioMistral-7B",
                 embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # try to load with low memory footprint; if GPU is available prefer float16
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        except Exception:
            # fallback to cpu
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)

        # embeddings + FAISS index
        self.embedder = SentenceTransformer(embed_model_name)
        self.index = None
        self.doc_texts = []

    def build_knowledge_base(self, doc_texts):
        """
        Build FAISS index from a list of document texts.
        doc_texts: list[str]
        """
        if not doc_texts:
            return
        self.doc_texts = doc_texts
        embeddings = self.embedder.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype('float32'))

    def retrieve_context(self, query, top_k=3):
        if self.index is None or len(self.doc_texts) == 0:
            return ""
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb.astype('float32'), top_k)
        hits = []
        for i in I[0]:
            if i < len(self.doc_texts):
                hits.append(self.doc_texts[i])
        return "\n\n---\n\n".join(hits)

    def _generate_with_scores(self, prompt, max_new_tokens=256, temperature=0.2):
        """
        Uses generate(..., return_dict_in_generate=True, output_scores=True) to obtain scores
        which we use to approximate token-level uncertainty/entropy for a confidence proxy.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
        )
        out = self.model.generate(**gen_kwargs)
        sequences = out.sequences  # tensor of token ids
        scores = out.scores  # list of logits for each generated step
        # decode
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        # compute approximate entropy over generated tokens
        # each element of scores is logits for that step
        entropies = []
        for step_logits in scores:
            probs = torch.nn.functional.softmax(step_logits, dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()
            entropies.append(ent)
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        return decoded, avg_entropy

    def medical_rag_query(self, user_query, top_k=3):
        """
        Retrieve context, build a prompt, and generate an answer along with a confidence proxy.
        """
        context = self.retrieve_context(user_query, top_k=top_k)
        system = ("You are a helpful, cautious clinical assistant. Do not provide definitive diagnoses"
                  " or prescriptive medication dosages. When unsure, say you are unsure and recommend"
                  " a clinician evaluation. Cite or reference supporting context when available.\n\n")
        prompt = f"{system}Context:\n{context}\n\nUser question:\n{user_query}\n\nAnswer (concise, explain reasoning):"
        answer, avg_entropy = self._generate_with_scores(prompt)
        # normalize "confidence" from entropy (higher entropy -> lower confidence)
        # map entropy to confidence in [0,1] using a simple exponential transform
        confidence = float(np.exp(-avg_entropy))
        return {"answer": answer, "confidence": confidence, "context": context}

# ---------------------------
# Empathic Engine
# ---------------------------
class EmpathicEngine:
    def __init__(self):
        pass

    def detect_language(self, text):
        try:
            # lazy language detection (fast)
            import langdetect
            return langdetect.detect(text)
        except Exception:
            return "en"

    def detect_sentiment(self, text):
        """
        Lightweight sentiment proxy: returns 'neutral'/'anxious'/'distressed'
        This is a placeholder — replace with a stronger classifier if needed.
        """
        text_l = text.lower()
        if any(w in text_l for w in ["suicid", "kill myself", "end my life", "i can't"]):
            return "distressed"
        if any(w in text_l for w in ["worried", "anxious", "panic", "nervous", "concerned"]):
            return "anxious"
        return "neutral"

# ---------------------------
# Ethical Engine
# ---------------------------
class EthicalEngine:
    def __init__(self, rules_path="rules.json", confidence_threshold=0.4):
        self.rules_path = rules_path
        self.confidence_threshold = confidence_threshold
        self._load_rules()

    def _load_rules(self):
        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r") as f:
                try:
                    self.rules = json.load(f)
                except Exception:
                    self.rules = []
        else:
            self.rules = []

    def match_rule(self, text):
        text_l = text.lower()
        for r in self.rules:
            patt = r.get("pattern", "")
            try:
                if patt and patt in text_l:
                    return r
            except Exception:
                continue
        return None

    def validate_and_maybe_escalate(self, user_query, draft_response, context="", conversation_history=None, empathic_flag=None):
        """
        Validate response: returns dict with keys:
         - action: 'release'|'escalate'|'block'
         - message: text to show to user (either draft_response or canned escalation)
         - reason: why escalated
        """
        # reload rules for every decision (rules.json can change)
        self._load_rules()
        # check rules against user query and draft_response
        rule = self.match_rule(user_query) or self.match_rule(draft_response)
        # simple confidence check: if draft_response includes "I am not sure" or similar, treat as low confidence
        low_confidence_phrases = ["i am not sure", "i may be wrong", "as an ai", "cannot"]
        if any(p in draft_response.lower() for p in low_confidence_phrases):
            confidence_flag = True
        else:
            confidence_flag = False

        # combine signals
        if empathic_flag in ("distressed",):
            # highest priority: user distress -> escalate
            esc_message = self._escalation_message_for("distress")
            # log to DB
            case_id = utils.add_escalation(user_query, draft_response, "user_distress", conversation_history)
            return {"action": "escalate", "message": esc_message, "reason": "user_distress", "case_id": case_id}

        if rule:
            act = rule.get("action", "escalate")
            if act == "escalate":
                case_id = utils.add_escalation(user_query, draft_response, f"rule:{rule.get('id')}", conversation_history)
                return {"action": "escalate", "message": rule.get("message", "Your request has been escalated."), "reason": f"rule:{rule.get('id')}", "case_id": case_id}
            else:
                return {"action": "block", "message": rule.get("message", "This content is not allowed."), "reason": f"rule:{rule.get('id')}"}

        # confidence numeric check (if cognitive engine attaches confidence in metadata)
        conf = None
        try:
            conf = float(draft_response.get("confidence", 1.0)) if isinstance(draft_response, dict) and "confidence" in draft_response else None
        except Exception:
            conf = None

        if conf is not None:
            if conf < self.confidence_threshold or confidence_flag:
                case_id = utils.add_escalation(user_query, draft_response if isinstance(draft_response, str) else draft_response.get("answer",""), "low_confidence", conversation_history)
                return {"action": "escalate", "message": "I am not confident in this result and have asked a human to review.", "reason": "low_confidence", "case_id": case_id}

        # default: release
        final_text = draft_response["answer"] if isinstance(draft_response, dict) else draft_response
        return {"action": "release", "message": final_text, "reason": "ok"}
