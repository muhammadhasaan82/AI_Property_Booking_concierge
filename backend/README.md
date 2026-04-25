# Hotel Booking Backend
## Run locally
```bash
# from project root
docker compose up redis -d
cd backend
uvicorn app.main:app --reload --port 8000