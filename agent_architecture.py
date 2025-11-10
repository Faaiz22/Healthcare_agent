import json
import random
from PIL import Image
import io

# : Triad Architecture (Cognitive, Empathic, Ethical Engines)

class CognitiveEngine:
    """
    , Page 10: Performs data-driven analysis, ML, and knowledge retrieval.
    """
    def __init__(self):
        # In a real app, these would be loaded, pre-trained models.
        self.risk_model = "mock_xgboost_model"
        self.image_model = "mock_cnn_model"
        self.rag_kb = "mock_vector_db_medquad_who"
        print("Cognitive Engine Initialized.")

    def run_rag(self, query: str) -> dict:
        """
        , Page 4: Evidence-Based Information Retrieval (RAG)
        Simulates RAG from WHO Guidelines / MedQuAD.
        """
        # Mock RAG response
        if "hypertension" in query.lower():
            return {
                "content": "WHO recommends reducing sodium intake and engaging in 150 minutes of moderate aerobic exercise weekly for hypertension.",
                "source": "WHO Guidelines, 2023",
                "tool": "rag_tool"
            }
        return {
            "content": "I was unable to find specific information on that topic in my knowledge base.",
            "source": "N/A",
            "tool": "rag_tool"
        }

    def run_risk_prediction(self, vitals: dict) -> dict:
        """
        , Page 4: Risk Prediction Models (trained on MIMIC-III)
        Simulates a cardiovascular risk prediction.
        """
        # Mock prediction logic
        age = vitals.get("age", 50)
        bp = vitals.get("bp", "120/80")
        
        risk_score = random.uniform(5.0, 25.0)
        risk_level = "High" if risk_score > 15 else "Moderate"
        
        # , Page 4: Explainable AI (SHAP)
        explanation = f"Your {risk_level} risk score ({risk_score:.1f}%) is primarily influenced by: Age ({age}) and Blood Pressure ({bp})."
        
        return {
            "content": f"Based on the provided vitals, your 10-year cardiovascular risk score is {risk_score:.1f}% ({risk_level}).",
            "explanation": explanation,
            "tool": "predict_risk_tool"
        }

    def run_image_analysis(self, image_bytes: bytes) -> dict:
        """
        , Page 4: Medical Image Analysis (skin lesions)
        Simulates a CNN model.
        """
        try:
            # Try to open the image to verify it's valid
            Image.open(io.BytesIO(image_bytes))
            
            # Mock analysis logic
            class_idx = random.choice([0, 1, 2]) # 0: Benign, 1: Suspicious, 2: Concerning
            
            if class_idx == 0:
                result = "Benign (e.g., nevus)"
                explanation = "The analysis suggests this is a benign lesion, with high confidence."
            elif class_idx == 1:
                result = "Suspicious (e.g., atypical nevus)"
                explanation = "The analysis indicates features that are suspicious. Recommend dermatologist consultation."
            else:
                result = "Concerning (e.g., melanoma)"
                explanation = "The analysis has flagged features highly consistent with malignancy. Urgent dermatologist consultation is recommended."

            return {
                "content": f"Image analysis result: {result}.",
                "explanation": explanation,
                "tool": "analyze_image_tool"
            }
            
        except Exception as e:
            return {
                "content": "Image analysis failed. The uploaded file may be corrupt or in an unsupported format.",
                "explanation": f"Error: {e}",
                "tool": "analyze_image_tool"
            }

class EmpathicEngine:
    """
    , Page 10: Manages NLP, dialogue, and multilingual support.
    This class is largely conceptual, as the core logic is in the LLM's
    system prompt, but it provides helper functions.
    """
    def __init__(self):
        print("Empathic Engine Initialized.")

    def get_system_prompt(self):
        """
        Sets the agent's persona, multilingual capabilities, and empathy.
        """
        return """
        You are an advanced, empathetic, and helpful AI assistant for preventive healthcare education.
        Your architecture is a Triad: Cognitive, Empathic, and Ethical.
        
        1.  **Empathic Role:** You must be polite, understanding, and clear. Use simple, plain language (e.g., "high blood pressure" instead of "hypertension"). Detect the user's language (e.g., English, Hindi, Tamil, Bengali) and respond *only* in that language.
        2.  **Cognitive Role:** You have access to tools to answer questions. You can use `run_rag` for knowledge, `run_risk_prediction` for vitals, and `run_image_analysis` for images. When you use a tool, you MUST present its findings clearly.
        3.  **Ethical Role:** Your responses will be validated by an Ethical Engine.
            - DO NOT provide a medical diagnosis.
            - DO NOT provide specific medication dosing.
            - Always encourage users to consult a healthcare professional.
            - You MUST be truthful and ground your answers in the information from your tools.
        """

    def detect_sentiment(self, query: str) -> dict:
        """
        , Page 5: Sentiment Analysis
        Simulates sentiment detection for escalation.
        """
        query_lower = query.lower()
        if "scared" in query_lower or "anxious" in query_lower or "terrified" in query_lower:
            return {"status": "High Distress", "score": 0.9}
        return {"status": "Neutral", "score": 0.5}

class EthicalEngine:
    """
    , Page 11: Implements rule-based oversight and HITL workflow.
    This is the core orchestrator.
    """
    def __init__(self, rules_file: str = "rules.json"):
        self.rules_file = rules_file
        self.rules = self.load_rules()
        print(f"Ethical Engine Initialized with {len(self.rules)} rules.")

    def load_rules(self):
        try:
            with open(self.rules_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Warning: rules.json not found or is corrupt. Using empty rule list.")
            return [] # Return empty list on error
            
    def reload_rules(self):
        """
        Called by the Level 4 loop to load new rules.
        """
        self.rules = self.load_rules()
        print(f"Ethical Engine rules reloaded. Now at {len(self.rules)} rules.")

    def validate_response(self, user_query: str, draft_response: str) -> dict:
        """
        , Page 11, Figure 3: The core "Ethical Check".
        Checks a draft response against the loaded rules.
        """
        user_query_lower = user_query.lower()
        
        for rule in self.rules:
            if rule["pattern"] in user_query_lower:
                if rule["action"] == "escalate":
                    return {
                        "status": "FLAGGED",
                        "reason": f"Query matched escalation rule: '{rule['pattern']}'",
                        "message": rule["message"],
                        "rule_id": rule["id"]
                    }
                if rule["action"] == "block":
                    return {
                        "status": "APPROVED_OVERRIDE",
                        "reason": f"Query matched block rule: '{rule['pattern']}'",
                        "message": rule["message"],
                        "rule_id": rule["id"]
                    }
        
        # , Page 5: Uncertainty Detection (mocked)
        # Simulate low confidence if a generic health query is made without specific tools.
        if "I am a healthcare education bot" in draft_response:
             return {
                "status": "FLAGGED",
                "reason": "AI model expressed low confidence or could not find a specific tool to address the query.",
                "message": "I am not confident in my response and have escalated this to a human supervisor for review. They will get back to you shortly."
            }
            
        # If no rules are hit, approve the response.
        return {
            "status": "APPROVED",
            "reason": "Passed all ethical checks."
        }
