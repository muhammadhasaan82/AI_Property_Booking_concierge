// database/rust_db_gateway/src/main.rs
// ─────────────────────────────────────────────────────────────────────────────
// Rust DB Gateway — lightweight async HTTP microservice that wraps all
// Supabase PostgreSQL operations (users + bookings CRUD).
//
// Endpoints:
//   GET  /health
//   POST /users/upsert          { name, email, phone? }         → { ok, user_id }
//   POST /bookings/create       { user_id, property_id, check_in, check_out, guests?, phone? }
//   GET  /bookings/:id/status   → { ok, status, check_in, check_out }
//   PATCH /bookings/:id/status  { status }                      → { ok }
// ─────────────────────────────────────────────────────────────────────────────

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    routing::{get, patch, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{Pool, Postgres, postgres::PgPoolOptions};
use std::{env, sync::Arc};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

// ─── Shared state ─────────────────────────────────────────────────────────────
struct AppState {
    db: Pool<Postgres>,
}

// ─── Request / Response models ────────────────────────────────────────────────
#[derive(Deserialize)]
struct UpsertUserReq {
    name:  Option<String>,
    email: String,
    phone: Option<String>,
}

#[derive(Deserialize)]
struct CreateBookingReq {
    user_id:     String,
    property_id: String,
    check_in:    String,
    check_out:   String,
    guests:      Option<i32>,
    phone:       Option<String>,
    payment_url: Option<String>,
}

#[derive(Deserialize)]
struct UpdateStatusReq {
    status: String,
}

#[derive(Serialize)]
struct OkId {
    ok: bool,
    id: String,
}

// ─── Health ───────────────────────────────────────────────────────────────────
async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    match sqlx::query_scalar::<_, i64>("SELECT 1")
        .fetch_one(&state.db)
        .await
    {
        Ok(_) => Json(json!({"ok": true, "service": "rust_db_gateway", "db": "connected"})),
        Err(e) => Json(json!({"ok": false, "service": "rust_db_gateway", "db": "error", "error": e.to_string()})),
    }
}

// ─── Users upsert ─────────────────────────────────────────────────────────────
async fn upsert_user(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpsertUserReq>,
) -> (StatusCode, Json<Value>) {
    let sql = r#"
        INSERT INTO public.users (name, email, phone)
        VALUES ($1, $2, $3)
        ON CONFLICT (email) DO UPDATE
            SET name  = COALESCE(EXCLUDED.name,  public.users.name),
                phone = COALESCE(EXCLUDED.phone, public.users.phone)
        RETURNING id::text
    "#;

    match sqlx::query_scalar::<_, String>(sql)
        .bind(req.name.as_deref().unwrap_or(""))
        .bind(&req.email)
        .bind(req.phone.as_deref())
        .fetch_one(&state.db)
        .await
    {
        Ok(id) => (StatusCode::OK, Json(json!({"ok": true, "user_id": id}))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": e.to_string()})),
        ),
    }
}

// ─── Bookings create ──────────────────────────────────────────────────────────
async fn create_booking(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateBookingReq>,
) -> (StatusCode, Json<Value>) {
    let user_id = match Uuid::parse_str(&req.user_id) {
        Ok(u) => u,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"ok": false, "error": "invalid user_id UUID"})),
            )
        }
    };

    let sql = r#"
        INSERT INTO public.bookings
            (user_id, property_id, check_in, check_out, guests, phone, status, payment_url)
        VALUES ($1, $2, $3::date, $4::date, $5, $6, 'pending', $7)
        RETURNING id::text, status
    "#;

    match sqlx::query_as::<_, (String, String)>(sql)
        .bind(user_id)
        .bind(&req.property_id)
        .bind(&req.check_in)
        .bind(&req.check_out)
        .bind(req.guests.unwrap_or(1))
        .bind(req.phone.as_deref())
        .bind(req.payment_url.as_deref())
        .fetch_one(&state.db)
        .await
    {
        Ok((booking_id, status)) => (
            StatusCode::CREATED,
            Json(json!({"ok": true, "booking_id": booking_id, "status": status})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": e.to_string()})),
        ),
    }
}

// ─── Booking status GET ───────────────────────────────────────────────────────
async fn get_booking_status(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> (StatusCode, Json<Value>) {
    let sql = r#"
        SELECT status, check_in::text, check_out::text
        FROM public.bookings
        WHERE id = $1::uuid
    "#;

    match sqlx::query_as::<_, (String, Option<String>, Option<String>)>(sql)
        .bind(&id)
        .fetch_optional(&state.db)
        .await
    {
        Ok(Some((status, check_in, check_out))) => (
            StatusCode::OK,
            Json(json!({
                "ok": true,
                "status": status,
                "check_in": check_in,
                "check_out": check_out,
            })),
        ),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "error": "booking not found"})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": e.to_string()})),
        ),
    }
}

// ─── Booking status PATCH ─────────────────────────────────────────────────────
async fn update_booking_status(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateStatusReq>,
) -> (StatusCode, Json<Value>) {
    let sql = "UPDATE public.bookings SET status = $1 WHERE id = $2::uuid";

    match sqlx::query(sql)
        .bind(&req.status)
        .bind(&id)
        .execute(&state.db)
        .await
    {
        Ok(r) if r.rows_affected() > 0 => (StatusCode::OK, Json(json!({"ok": true}))),
        Ok(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "error": "booking not found"})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok": false, "error": e.to_string()})),
        ),
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() {
    // Load .env from project root or services/
    let _ = dotenvy::from_path("../../services/.env").map(|_| ())
        .or_else(|_| dotenvy::from_path("../services/.env").map(|_| ()))
        .or_else(|_| dotenvy::dotenv().map(|_| ()));

    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    // Build Supabase Postgres URL from env
    // Supports both DATABASE_URL directly or individual Supabase vars
    let database_url = env::var("DATABASE_URL").unwrap_or_else(|_| {
        let host     = env::var("SUPABASE_DB_HOST").unwrap_or_else(|_| "db.mdehkfdilygmscdbowbh.supabase.co".to_string());
        let port     = env::var("SUPABASE_DB_PORT").unwrap_or_else(|_| "5432".to_string());
        let db       = env::var("SUPABASE_DB_NAME").unwrap_or_else(|_| "postgres".to_string());
        let user     = env::var("SUPABASE_DB_USER").unwrap_or_else(|_| "postgres".to_string());
        let password = env::var("SUPABASE_DB_PASSWORD").expect(
            "Set DATABASE_URL or SUPABASE_DB_PASSWORD environment variable"
        );
        format!("postgres://{}:{}@{}:{}/{}", user, password, host, port, db)
    });

    tracing::info!("Connecting to Supabase PostgreSQL...");

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Failed to connect to database");

    tracing::info!("Database connection pool established.");

    let state = Arc::new(AppState { db: pool });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health",               get(health))
        .route("/users/upsert",         post(upsert_user))
        .route("/bookings/create",      post(create_booking))
        .route("/bookings/:id/status",  get(get_booking_status))
        .route("/bookings/:id/status",  patch(update_booking_status))
        .layer(DefaultBodyLimit::disable())
        .layer(cors)
        .with_state(state);

    let port = env::var("DB_GATEWAY_PORT").unwrap_or_else(|_| "3002".to_string());
    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Rust DB Gateway listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
