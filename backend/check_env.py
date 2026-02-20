import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

key = os.getenv("DEEPGRAM_API_KEY", "")
print(f"DEEPGRAM_API_KEY length: {len(key)}")
if len(key) > 5:
    print(f"Starts with: {key[:5]}...")
else:
    print("Key is empty or missing!")
try:
    import httpx
    print("httpx is installed")
except ImportError:
    print("httpx is NOT installed")
