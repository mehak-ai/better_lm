import httpx

url = "http://localhost:8000/api/voice/token/"
print(f"Testing GET {url}...")
try:
    r = httpx.get(url, timeout=5.0)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")

url_no_slash = "http://localhost:8000/api/voice/token"
print(f"\nTesting GET {url_no_slash}...")
try:
    r = httpx.get(url_no_slash, timeout=5.0)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")
