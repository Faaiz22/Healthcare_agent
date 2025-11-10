Level 4 HITL Triad Agent

This is a Level 4 Self-Evolving Agentic Chatbot built with Streamlit, based on the architectures described in Google's "Introduction to Agents" and the "Human-AI Coordinated Healthcare" proposal.

Architecture
Triad Engine (agent_architecture.py):
Cognitive Engine: Handles RAG, ML risk-prediction, and image analysis.
Empathic Engine: Manages multilingual, stateful chat.
Ethical Engine: Provides rule-based validation and HITL escalation.

Multi-Page HITL (streamlit_app.py, pages/1_Supervisor_Dashboard.py):
Healthcare Chatbot: The main, user-facing application.
Supervisor Dashboard: A separate page for human supervisors to review and act on flagged conversations.

Level 4 Self-Evolution (utils.py):
When a supervisor resolves a case, the agent's level_4_evolution_loop runs.
It analyzes the supervisor's correction and autonomously generates a new safety rule, which it writes back to rules.json.

Deployment
Upload this entire directory to a public GitHub repository.
Log in to Streamlit Community Cloud (share.streamlit.io).
Click "New App" and deploy from your GitHub repository.
Streamlit will automatically detect requirements.txt and packages.txt to build the environment.
