import os
import httpx
from dotenv import load_dotenv

load_dotenv(".env")
api_key = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
body = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Ping"}],
    "max_tokens": 1
}

print("Checking Groq Rate Limits...")
try:
    resp = httpx.post(url, json=body, headers=headers, timeout=10)
    print(f"Status: {resp.status_code}")
    
    # Print relevant headers
    print("--- Rate Limit Headers ---")
    for k, v in resp.headers.items():
        if "ratelimit" in k.lower():
            print(f"{k}: {v}")
            
    if resp.status_code != 200:
        print("\nError Body:", resp.text)

except Exception as e:
    print("Connection Error:", e)
