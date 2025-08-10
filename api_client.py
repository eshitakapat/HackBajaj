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
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=HEADERS,
            json=payload
        )
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM API error: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response content: {e.response.text}")
        return "Answer not available."
