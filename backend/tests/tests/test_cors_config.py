from app.main import DEFAULT_CORS_ORIGINS, _get_cors_origins, _origin_only


def test_default_cors_includes_github_pages_origin(monkeypatch):
    monkeypatch.delenv("BACKEND_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    origins = _get_cors_origins()

    assert "https://muhammadhasaan82.github.io" in origins
    assert "http://localhost:8501" in origins
    assert origins == list(DEFAULT_CORS_ORIGINS)


def test_cors_origin_normalization_removes_paths():
    assert (
        _origin_only("https://muhammadhasaan82.github.io/AI_Property_Booking_concierge/")
        == "https://muhammadhasaan82.github.io"
    )


def test_env_cors_origins_are_comma_separated_and_normalized(monkeypatch):
    monkeypatch.setenv(
        "BACKEND_CORS_ORIGINS",
        " http://localhost:3000, https://example.com/app/path , ",
    )

    assert _get_cors_origins() == ["http://localhost:3000", "https://example.com"]
