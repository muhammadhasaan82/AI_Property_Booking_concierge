from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

response = client.get("/debug/config")
print(f"Status Code: {response.status_code}")
data = response.json()
print("Keys returned:", list(data.keys()))
print("Legacy mode:", data.get("legacy_mode"))
print("Intents:", list(data.get("intent_catalog", {}).get("intents", {}).keys()))
