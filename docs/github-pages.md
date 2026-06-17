# GitHub Pages Frontend

GitHub Pages is static hosting. It does not run FastAPI, Chainlit, Redis,
Supabase/Postgres, the Rust gateway, Python workers, or model-provider calls.

This repository therefore uses GitHub Pages only for a static browser frontend.
The live backend must be deployed separately as a public HTTPS API.

## Deployment Model

- GitHub Pages serves `frontend_static/`.
- The existing Chainlit UI in `frontend/chainlit_app.py` remains available for
  local, GCP, PM2, or server deployments.
- The FastAPI backend in `backend/app/main.py` remains the API runtime.
- The Pages frontend calls the backend through `API_BASE_URL`.

Expected Pages URL:

```text
https://muhammadhasaan82.github.io/AI_Property_Booking_concierge/
```

## Repository Settings

In the GitHub repository:

1. Open **Settings**.
2. Open **Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

## Backend API URL

Set a repository variable named `PUBLIC_API_BASE_URL`:

```text
https://your-backend-host.example.com
```

Do not put secrets in this variable. Any frontend configuration is visible to
users in the browser.

If `PUBLIC_API_BASE_URL` is not set, the Pages site still deploys and clearly
shows that the backend API is not configured. Users can enter a backend URL in
the page for their current browser session.

## Backend CORS

The backend must allow the GitHub Pages origin:

```bash
BACKEND_CORS_ORIGINS=http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501,https://muhammadhasaan82.github.io
```

CORS origins must not include a path. Use:

```text
https://muhammadhasaan82.github.io
```

not:

```text
https://muhammadhasaan82.github.io/AI_Property_Booking_concierge/
```

## Workflows

- `.github/workflows/ci.yml` runs backend tests, Rust gateway checks, workflow
  YAML validation, and the v2 AI Report Card eval.
- `.github/workflows/pages.yml` deploys the static frontend to GitHub Pages on
  pushes to `main` and on manual dispatch.

## Live Chat Status

The Pages frontend can call `/health`, `/docs`, and `/chat/message` only when a
public backend API is configured and reachable. Without that hosted backend, the
Pages site is a static demo and status console.
