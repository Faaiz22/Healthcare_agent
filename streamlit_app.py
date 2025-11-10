import streamlit as st
from PIL import Image
from agent_architecture import CognitiveEngine, EmpathicEngine, EthicalEngine
import utils
import time
import os

# --- Page Config ---
st.set_page_config(
    page_title="Healthcare Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Ensure the database is initialized (important for fresh deployments)
if not os.path.exists(utils.DB_NAME):
    utils.init_db()
if not os.path.exists(utils.RULES_FILE):
    # Recreate default rules if file is missing (e.g., in a new deployment)
    default_rules = [
        {"id": "rule_001", "pattern": "diagnose", "action": "block", "message": "I cannot provide a medical diagnosis. Please consult a qualified healthcare professional.", "source": "initial_policy"},
        {"id": "rule_002", "pattern": "medication dosage", "action": "block", "message": "I cannot recommend specific medication dosages. Always follow your doctor's instructions or pharmacist's advice.", "source": "initial_policy"},
        {"id": "rule_003", "pattern": "severe chest pain", "action": "escalate", "message": "This requires urgent medical attention. Please call emergency services immediately or go to the nearest emergency room. A healthcare professional has been notified.", "source": "initial_policy"},
        {"id": "rule_004", "pattern": "suicidal ideation", "action": "escalate", "message": "It sounds like you're going through a difficult time. Please reach out for immediate help. You can contact a crisis hotline or emergency services. Your well-being is important.", "source": "initial_policy"}
    ]
    with open(utils.RULES_FILE, 'w') as f:
        json.dump(default_rules, f, indent=2)


# --- Agent Initialization ---
# : Initialize the Triad Engine
@st.cache_resource
def load_agent():
    return {
        "cognitive": CognitiveEngine(),
        "empathic": EmpathicEngine(),
        "ethical": EthicalEngine(rules_file="rules.json")
    }

agent = load_agent()
cognitive_engine = agent["cognitive"]
empathic_engine = agent["empathic"]
ethical_engine = agent["ethical"]

# --- Session State ---
# , Page 5: Maintain dialogue state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vitals" not in st.session_state:
    st.session_state.vitals = {}
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

# --- UI Components ---
st.title("🤖 Human-AI Coordinated Healthcare Chatbot")
st.markdown("This AI chatbot provides preventive healthcare education. It is *not* a substitute for professional medical advice.")

# , Page 5: Multi-page architecture for data input
with st.sidebar:
    st.header("👤 Your Health Profile")
    st.caption("This (simulated) data helps the AI personalize its risk assessments.")
    
    st.session_state.vitals["age"] = st.number_input("Age", min_value=18, max_value=100, value=50, key="age_input")
    st.session_state.vitals["bp"] = st.text_input("Blood Pressure (e.g., 120/80)", value="145/92", key="bp_input")
    
    st.header("🔬 Image Analysis")
    uploaded_image = st.file_uploader("Upload a medical image (e.g., skin lesion)", type=["jpg", "png", "jpeg"], key="image_uploader")
    if uploaded_image:
        st.session_state.image_bytes = uploaded_image.getvalue()
        st.image(st.session_state.image_bytes, caption="Image ready for analysis.", use_column_width=True)

# --- Chat Interface ---
# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a health-related question..."):
    # 1. Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare for agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        
        # --- (SIMULATED) LLM "THINK" STEP ---
        # This simulates the LLM call to get a draft response.
        # In a real app, this would be a call to Gemini,
        # using empathic_engine.get_system_prompt() and tools.
        
        # Mocking the tool-use logic based on query
        time.sleep(0.5) # Simulate network latency
        
        draft_response = ""
        
        if st.session_state.image_bytes is not None:
            # User uploaded an image, assume they want it analyzed
            tool_result = cognitive_engine.run_image_analysis(st.session_state.image_bytes)
            draft_response = f"I have analyzed the image you uploaded. {tool_result['content']} \n\n**Explanation:** {tool_result['explanation']}"
            st.session_state.image_bytes = None # Clear image after use
        
        elif "risk" in prompt.lower() or "vitals" in prompt.lower() or "score" in prompt.lower():
            tool_result = cognitive_engine.run_risk_prediction(st.session_state.vitals)
            draft_response = f"{tool_result['content']} \n\n**Explanation:** {tool_result['explanation']}"
        
        elif "hypertension" in prompt.lower() or "blood pressure" in prompt.lower() or "who recommends" in prompt.lower():
            tool_result = cognitive_engine.run_rag(prompt)
            draft_response = f"According to my knowledge base: {tool_result['content']} (Source: {tool_result['source']})"
        
        else:
            # Fallback response for general queries
            draft_response = "I am a healthcare education bot. I can provide information on preventive health, analyze (mock) health data you provide, and analyze (mock) medical images you upload. Please ask about general health topics, risk assessments, or upload an image for analysis."
        
        # --- 3. ETHICAL ENGINE VALIDATION  ---
        validation = ethical_engine.validate_response(
            user_query=prompt,
            draft_response=draft_response
        )
        
        # --- 4. ACT & OBSERVE LOOP  ---
        if validation["status"] == "APPROVED":
            # Case 1: Response is safe. Deliver it.
            final_message = draft_response
            full_response_content = final_message
            
        elif validation["status"] == "APPROVED_OVERRIDE":
            # Case 2: Rule-based block. Deliver the rule's canned message.
            final_message = validation["message"]
            full_response_content = final_message
            
        elif validation["status"] == "FLAGGED":
            # Case 3: Escalation. Deliver canned message AND log for HITL.
            final_message = validation["message"]
            full_response_content = final_message
            
            # Log for supervisor review
            utils.add_escalation(
                user_query=prompt,
                flagged_ai_response=draft_response,
                flag_reason=validation["reason"],
                conversation_history=st.session_state.messages # Log current full history
            )
        
        # Simulate streaming response
        for chunk in full_response_content.split():
            message_placeholder.markdown(full_response_content + "▌") # Show typing indicator
            full_response_content = full_response_content.replace(chunk + " ", "", 1)
            time.sleep(0.05)
        message_placeholder.markdown(final_message) # Display full message
        
    # 5. Add final agent message to state
    st.session_state.messages.append({"role": "assistant", "content": final_message})
