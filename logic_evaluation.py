import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "Content-Type": "application/json",
}

def evaluate_decision(clause: str, context: dict) -> dict:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "sk-or-v1-92f...202":
        return mock_evaluate_decision(clause, context)

    prompt = f"""
Given the clause: "{clause}"
And the context: {json.dumps(context)}
Is this compliant? Provide rationale.
"""

    payload = {
        "model": "mistral-7b-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        res_json = response.json()
        text = res_json["choices"][0]["message"]["content"]
        compliant = "yes" in text.lower()
        rationale = text.strip()
        return {
            "compliant": compliant,
            "rationale": rationale
        }
    except Exception as e:
        print(f"OpenRouter API error or evaluation failed: {e}")
        return mock_evaluate_decision(clause, context)

def mock_evaluate_decision(clause: str, context: dict) -> dict:
    clause_lower = clause.lower()
    compliant = False
    rationale = "Clause does not meet compliance criteria."

    if "limit" in clause_lower or "shall" in clause_lower or "terminate" in clause_lower:
        compliant = True
        rationale = "Clause provides standard compliance features (limitation or termination)."

    return {
        "compliant": compliant,
        "rationale": rationale
    }
