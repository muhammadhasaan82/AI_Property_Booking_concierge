import asyncio
from app.services.graph import node_triage

state1 = {"user_text": "9", "filters": {"last_results": [{"id": "p1"}]}}
print("State 1 (with last_results):", node_triage(state1)["intent"])

state2 = {"user_text": "9", "filters": {}}
print("State 2 (no last_results):", node_triage(state2)["intent"])

