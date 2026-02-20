import os
import sys
import httpx
from dotenv import load_dotenv

# Load .env
load_dotenv(".env")

api_key = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY present: {bool(api_key)}")
if api_key:
    # masking
    print(f"Key preview: {api_key[:4]}...{api_key[-4:]}")

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
body = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}

print("\nTesting Groq API Connection...")
try:
    resp = httpx.post(url, json=body, headers=headers, timeout=10)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("Success! Response:", resp.json())
    else:
        print("Failed! Response:", resp.text)
except Exception as e:
    print(f"Connection Error: {e}")
