import streamlit as st
import utils
import json
import os # Import os to check for file existence

st.set_page_config(
    page_title="Supervisor Dashboard",
    page_icon="📋",
    layout="wide"
)

st.title("📋 HITL Supervisor Dashboard")
st.markdown("Review and resolve AI-flagged conversations. This feedback drives the agent's self-evolution.")

# , Page 18: This dashboard is the source of feedback for Level 4 evolution.

# Initialize EthicalEngine if not already in session state.
# This ensures that even if a user navigates directly to this page,
# the ethical engine and its rules are correctly loaded.
if 'ethical_engine_dashboard' not in st.session_state:
    from agent_architecture import EthicalEngine
    st.session_state.ethical_engine_dashboard = EthicalEngine(rules_file=utils.RULES_FILE)
ethical_engine_dashboard = st.session_state.ethical_engine_dashboard

# Ensure the database is initialized (important for fresh deployments)
if not os.path.exists(utils.DB_NAME):
    utils.init_db()


# --- Dashboard UI ---
tab1, tab2 = st.tabs(["Pending Escalations", "Resolved Cases"])

with tab1:
    st.header("Pending Escalations")
    pending_cases = utils.get_escalations(status="PENDING")

    if not pending_cases:
        st.success("No pending cases in the review queue.")

    for case in pending_cases:
        with st.expander(f"**Case ID:** {case['id'][:8]} | **Time:** {case['timestamp'].split('T')[0]} | **Reason:** {case['flag_reason']}"):
            st.subheader("Conversation History")
            history = json.loads(case['conversation_history'])
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            st.subheader("Flagged Interaction Details")
            with st.chat_message("user"):
                st.markdown(f"**User Query:** {case['user_query']}")

            st.error(f"**AI Draft Response (BLOCKED):** {case['flagged_ai_response']}")
            st.warning(f"**Reason for Flag:** {case['flag_reason']}")

            st.subheader("Supervisor Action")
            st.markdown("Provide the correct, safe response to send to the user. This feedback will be used by the agent to learn and evolve its rules.")

            supervisor_response = st.text_area("Your Corrective Response:", key=f"supervisor_response_{case['id']}")

            if st.button("Resolve and Submit Feedback", key=f"resolve_button_{case['id']}"):
                if supervisor_response:
                    # 1. Resolve the case in the DB
                    utils.resolve_escalation(case['id'], supervisor_response)

                    # 2. Trigger the Level 4 Evolution Loop
                    # , Page 31: "Cherish Human Feedback"
                    # The agent now consumes this feedback to self-evolve.
                    case['supervisor_response'] = supervisor_response # Add for loop
                    utils.level_4_evolution_loop(case)

                    # 3. Reload rules in the EthicalEngine instance specific to the dashboard
                    # This ensures the dashboard's EthicalEngine is aware of new rules for consistency.
                    ethical_engine_dashboard.reload_rules()

                    st.success(f"Case {case['id'][:8]} resolved and Level 4 learning loop triggered.")
                    st.rerun() # Rerun to refresh the list of pending cases
                else:
                    st.error("Please provide a corrective response.")

with tab2:
    st.header("Resolved Cases")
    resolved_cases = utils.get_escalations(status="RESOLVED")

    if not resolved_cases:
        st.info("No resolved cases yet.")

    for case in resolved_cases:
        with st.expander(f"**Case ID:** {case['id'][:8]} | **Time:** {case['timestamp'].split('T')[0]} | **Reason:** {case['flag_reason']}"):
            st.json(case, expanded=False)
