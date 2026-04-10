FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt chainlit
COPY . .
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8501"]
