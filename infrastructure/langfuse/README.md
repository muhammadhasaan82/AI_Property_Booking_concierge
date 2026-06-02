# Langfuse Self-Hosted Deployment

This directory contains scripts and documentation for deploying a self-hosted Langfuse instance via Docker Compose for observability, traces, evals, and prompt/version tracking.

## 1. Requirements

- `git`
- `docker`
- `docker compose`
- **Recommended VM size**: 4 cores, 16 GiB RAM, and at least 100 GiB of disk space.

## 2. Deployment

Do not vendor the full Langfuse repository into this project. Clone it into a dedicated directory on your server (e.g., `/root/langfuse-selfhosted`).

```bash
# 1. Clone the official repository
cd /root
git clone https://github.com/langfuse/langfuse.git langfuse-selfhosted
cd langfuse-selfhosted

# 2. Copy the example environment file and update all #CHANGEME secrets
cp .env.example .env
# Edit .env and replace all #CHANGEME values with secure secrets

# 3. Start the services
docker compose up -d
```

## 3. Access

Once deployed, the Langfuse web UI is accessible at:
```
http://<server-ip>:3000
```
Log in with the default credentials defined in your `.env` file (e.g., `admin@langfuse.com` / `password`), and change them immediately.

## 4. Firewall & Security

- **Web UI**: Only expose Langfuse web port `3000` publicly if strictly necessary. It is highly recommended to restrict access via IP whitelisting or a reverse proxy with authentication.
- **Internal Ports**: Keep internal database (PostgreSQL), Redis, and ClickHouse ports private. Do not expose them to the public internet.
- **Media Uploads**: If MinIO/media upload is required, handle port `9090` carefully and restrict access.

## 5. Upgrade

To upgrade to the latest Langfuse version:

```bash
cd /root/langfuse-selfhosted
docker compose down
docker compose pull
docker compose up -d
```

## 6. Shutdown

To gracefully shut down the Langfuse instance:

```bash
cd /root/langfuse-selfhosted
docker compose down
```

**WARNING**: Do **NOT** use `docker compose down -v` unless you intentionally want to delete all data volumes (PostgreSQL, ClickHouse, MinIO). This action is irreversible and will result in total data loss.
