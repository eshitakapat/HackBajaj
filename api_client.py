import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "Content-Type": "application/json",
}

def generate_answer(question, context_chunks):
    prompt = f"""Answer the question concisely based on the context below:

Context:
{context_chunks}

Question: {question}
Answer:"""

    payload = {
        "model": "mistral-7b-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 150
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        res_json = response.json()
        text = res_json["choices"][0]["message"]["content"].strip()
        return text
    except Exception as e:
        print(f"LLM API error: {e}")
        return "Answer not available."
