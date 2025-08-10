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

def parse_query_to_structured(query: str) -> dict:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "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":
        # Fallback mock parser
        return mock_parse_query(query)

    prompt = f"""
Extract intent and filters from the query:
Query: {query}
Output JSON with keys: intent, clause_type, filters
"""

    payload = {
        "model": "mistral-7b-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        res_json = response.json()
        text = res_json["choices"][0]["message"]["content"]
        return json.loads(text)
    except Exception as e:
        print(f"OpenRouter API error or parsing failed: {e}")
        return mock_parse_query(query)


def mock_parse_query(query: str) -> dict:
    # Simple keyword parser as fallback
    intent = "find_clause"
    clause_type = None
    filters = {}

    q = query.lower()
    if "liability" in q:
        clause_type = "liability"
    elif "termination" in q:
        clause_type = "termination"
    elif "compliance" in q:
        clause_type = "compliance"

    if "after" in q:
        after_part = q.split("after")[-1].strip()
        date = after_part.split()[0]
        filters["effective_date"] = date

    for state in ["california", "new york", "texas"]:
        if state in q:
            filters["jurisdiction"] = state.title()

    return {
        "intent": intent,
        "clause_type": clause_type,
        "filters": filters
    }
