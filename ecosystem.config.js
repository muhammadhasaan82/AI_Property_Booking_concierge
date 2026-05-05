module.exports = {
  apps: [
    {
      name: "concierge-backend",
      cwd: "./backend",
      script: ".venv/bin/uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8002",
      interpreter: "none",
      env: {
        NODE_ENV: "production",
      },
      autorestart: true,
      max_restarts: 5,
      restart_delay: 3000,
    },
    {
      name: "concierge-frontend",
      cwd: ".",
      script: "backend/.venv/bin/chainlit",
      args: "run frontend/chainlit_app.py --host 0.0.0.0 --port 8501",
      interpreter: "none",
      env: {
        NODE_ENV: "production",
      },
      autorestart: true,
      max_restarts: 5,
      restart_delay: 3000,
    },
  ],
};
