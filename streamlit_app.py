"""
Streamlit front-end integrating the Triad engines and using local LLM + RAG.

Place in repo root. It assumes utils.py, agent_architecture.py, pages/1_Supervisor_Dashboard.py exist.
"""

import streamlit as st
import json
from agent_architecture import CognitiveEngine, EmpathicEngine, EthicalEngine
import utils
import os

st.set_page_config(page_title="Level4 HITL Triad Agent", layout="wide")

# initialize
if "cognitive" not in st.session_state:
    with st.spinner("Loading models (this may take a while first run)..."):
        st.session_state.cognitive = CognitiveEngine()  # loads BioMistral (or configured model)
        st.session_state.empathic = EmpathicEngine()
        st.session_state.ethical = EthicalEngine()

cognitive = st.session_state.cognitive
empathic = st.session_state.empathic
ethical = st.session_state.ethical

st.title("Level-4 HITL Triad — Offline LLM + RAG")

# sidebar: knowledge base upload / build index
st.sidebar.header("Knowledge base")
uploaded = st.sidebar.file_uploader("Upload a plain-text file (txt) or JSONL of docs to add to KB", type=["txt","json","jsonl"])
if st.sidebar.button("Add & Build KB") and uploaded is not None:
    if uploaded.type == "text/plain":
        raw = uploaded.getvalue().decode("utf-8")
        docs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    else:
        # try parse JSON lines or JSON array
        data = uploaded.getvalue().decode("utf-8")
        try:
            lines = [json.loads(l) for l in data.splitlines() if l.strip()]
            docs = [l.get("text") or l.get("content") or json.dumps(l) for l in lines]
        except Exception:
            try:
                arr = json.loads(data)
                docs = [a.get("text") or a.get("content") or json.dumps(a) for a in arr]
            except Exception:
                docs = [data]
    st.session_state.kb_docs = docs
    cognitive.build_knowledge_base(docs)
    st.sidebar.success(f"KB built with {len(docs)} docs")

# chat UI
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("chatform", clear_on_submit=False):
    user_text = st.text_area("Your question", height=120)
    submitted = st.form_submit_button("Ask")
    if submitted and user_text.strip():
        # add to history
        st.session_state.history.append({"role":"user","text":user_text})
        # empathic signals
        lang = empathic.detect_language(user_text)
        senti = empathic.detect_sentiment(user_text)
        # cognitive: RAG
        rag_out = cognitive.medical_rag_query(user_text)
        draft_response = rag_out  # dict with answer, confidence, context
        # ethical validation
        result = ethical.validate_and_maybe_escalate(user_text, draft_response, context=rag_out.get("context",""), conversation_history=st.session_state.history, empathic_flag=senti)
        if result["action"] == "release":
            st.session_state.history.append({"role":"assistant","text":result["message"]})
        elif result["action"] == "block":
            st.session_state.history.append({"role":"assistant","text":result["message"]})
        else:
            # escalate: show canned message, supervisor will handle in dashboard
            st.session_state.history.append({"role":"assistant","text":result["message"]})
        st.experimental_rerun()

# display chat history
st.subheader("Conversation")
for msg in st.session_state.history[::-1]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Agent:** {msg['text']}")

st.markdown("---")
st.caption("Notes: This demo runs an open LLM locally. For production, use secure deployments and a managed DB for escalations.")
