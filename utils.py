import sqlite3, json, uuid, datetime, os
from typing import List, Dict, Any

DB_NAME = "escalations.db"
RULES_FILE = "rules.json"

def init_db(db_path: str = DB_NAME):
    """Initialize the SQLite escalation DB and return connection path."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        user_query TEXT,
        flagged_ai_response TEXT,
        flag_reason TEXT,
        conversation_history TEXT,
        status TEXT,
        supervisor_response TEXT
    )
    """)
    conn.commit()
    conn.close()
    return db_path

def add_escalation(user_query: str, flagged_ai_response: str, flag_reason: str, conversation_history: Any, db_path: str = DB_NAME) -> str:
    """Add a pending escalation to the DB. Returns the new case id."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    case_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    history_json = conversation_history if isinstance(conversation_history, str) else json.dumps(conversation_history)
    c.execute("""INSERT INTO escalations (id, timestamp, user_query, flagged_ai_response, flag_reason, conversation_history, status) VALUES (?,?,?,?,?,?,?)""",
              (case_id, timestamp, user_query, flagged_ai_response, flag_reason, history_json, "PENDING"))
    conn.commit()
    conn.close()
    return case_id

def get_escalations(status: str = "PENDING", db_path: str = DB_NAME) -> List[Dict[str, Any]]:
    """Retrieve escalations by status. status can be 'PENDING' or 'RESOLVED'."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, user_query, flagged_ai_response, flag_reason, conversation_history, status, supervisor_response FROM escalations WHERE status = ?", (status,))
    rows = c.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "timestamp": r[1],
            "user_query": r[2],
            "flagged_ai_response": r[3],
            "flag_reason": r[4],
            "conversation_history": r[5],
            "status": r[6],
            "supervisor_response": r[7]
        })
    return results

def resolve_escalation(case_id: str, supervisor_response: str, db_path: str = DB_NAME):
    """Mark an escalation resolved and store the supervisor response."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE escalations SET status = ?, supervisor_response = ? WHERE id = ?", ("RESOLVED", supervisor_response, case_id))
    conn.commit()
    conn.close()

def level_4_evolution_loop(case: Dict[str, Any], rules_file_path: str = RULES_FILE):
    """
    Simple Level 4 evolution prototype:
    - Analyze the flagged AI response and supervisor corrective response
    - Generate a generalized rule (heuristic) and append to rules.json
    Note: This is a simple, transparent rule-generation example. In production,
    this step should be human-supervised and use robust validation before deployment.
    """
    # Load current rules
    if not os.path.exists(rules_file_path):
        with open(rules_file_path, "w") as f:
            json.dump([], f, indent=2)
    with open(rules_file_path, "r") as f:
        try:
            rules = json.load(f)
        except Exception:
            rules = []
    # Heuristic rule creation: pick a short key phrase from the user_query or flag_reason
    user_query = case.get("user_query", "") if isinstance(case, dict) else ""
    flag_reason = case.get("flag_reason", "") if isinstance(case, dict) else ""
    supervisor_response = case.get("supervisor_response", "") if isinstance(case, dict) else ""
    # Choose candidate patterns
    candidate = None
    for token in ["diagnose", "medication", "dose", "suicid", "chest pain", "emergency"]:
        if token in user_query.lower() or token in flag_reason.lower():
            candidate = token
            break
    if not candidate and user_query:
        # fallback: take first two words as pattern (safe)
        candidate = " ".join(user_query.lower().split()[:2])
    # Compose new rule entry
    new_rule = {
        "id": f"rule_autogen_{str(uuid.uuid4())[:8]}",
        "pattern": candidate,
        "action": "escalate" if "emerg" in (candidate or "") or "suicid" in (candidate or "") else "block",
        "message": supervisor_response or "This content has been flagged and requires human review.",
        "source": "autogen_level4_loop"
    }
    # Append if not duplicate pattern
    patterns = [r.get("pattern") for r in rules]
    if new_rule["pattern"] not in patterns:
        rules.append(new_rule)
        with open(rules_file_path, "w") as f:
            json.dump(rules, f, indent=2)
    # Return the new rule for visibility
    return new_rule