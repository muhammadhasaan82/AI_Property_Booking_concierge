from app.services import db_client


def test_to_psycopg_conninfo_converts_sqlalchemy_psycopg_url():
    raw = "postgresql+psycopg://user:pass@127.0.0.1:54322/postgres"

    assert (
        db_client._to_psycopg_conninfo(raw)
        == "postgresql://user:pass@127.0.0.1:54322/postgres"
    )


def test_to_psycopg_conninfo_preserves_libpq_dsn():
    raw = "host=127.0.0.1 port=54322 dbname=postgres user=supabase_admin password=secret"

    assert db_client._to_psycopg_conninfo(raw) == raw


def test_build_conninfo_prefers_psycopg_database_url(monkeypatch):
    monkeypatch.setenv("PSYCOPG_DATABASE_URL", "postgresql://direct:pass@127.0.0.1:54322/postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://sqlalchemy:pass@127.0.0.1:54322/postgres")

    assert db_client._build_conninfo() == "postgresql://direct:pass@127.0.0.1:54322/postgres"


def test_build_conninfo_converts_database_url_for_psycopg_pool(monkeypatch):
    monkeypatch.delenv("PSYCOPG_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@127.0.0.1:54322/postgres")

    assert db_client._build_conninfo() == "postgresql://user:pass@127.0.0.1:54322/postgres"