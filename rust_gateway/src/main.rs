mod gateway;
mod tools;
mod cache;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

// ---------------------------------------------------------------------------
// Application state shared across handlers
// ---------------------------------------------------------------------------
struct AppState {
    registry: tools::ToolRegistry,
    cache: cache::Cache,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Health check.
async fn health() -> Json<Value> {
    Json(json!({"ok": true, "service": "rust_gateway", "version": "0.1.0"}))
}

/// Schema-agnostic autonomous gateway.
/// Accepts arbitrary JSON, infers intent, routes to the best tool.
async fn execute(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let data = body.get("data").cloned().unwrap_or(body.clone());
    let context = body.get("context").cloned().unwrap_or(json!({}));

    // Check cache first
    let cache_key = cache::cache_key("execute", &data);
    if let Some(cached) = state.cache.get(&cache_key) {
        let mut result = cached;
        if let Some(obj) = result.as_object_mut() {
            obj.insert("cached".to_string(), json!(true));
        }
        return (StatusCode::OK, Json(result));
    }

    // Process through gateway
    let result = gateway::process_request(&data, &context, &state.registry);

    // Cache successful results
    if result.get("ok") == Some(&json!(true)) {
        let intent = result.get("intent").and_then(|v| v.as_str()).unwrap_or("unknown");
        let ttl = match intent {
            "search" => cache::ttl::PROPERTY_SEARCH,
            "faq" => cache::ttl::FAQ_ANSWER,
            _ => cache::ttl::PRICING,
        };
        state.cache.set(cache_key, result.clone(), ttl);
    }

    (StatusCode::OK, Json(result))
}

/// Direct property search tool endpoint.
async fn tool_search(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = state.registry.select(&body);
    match tool {
        Some(t) if t.name() == "property_search" => Json(t.execute(&body)),
        _ => {
            // Force use the search tool
            let search_tool = tools::search::PropertySearchTool;
            Json(search_tool.execute(&body))
        }
    }
}

/// Direct booking validation tool endpoint.
async fn tool_validate_booking(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::booking_validator::BookingValidatorTool;
    Json(tool.execute(&body))
}

/// Direct pricing tool endpoint.
async fn tool_pricing(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::pricing::PricingTool;
    Json(tool.execute(&body))
}

/// Direct sentiment analysis tool endpoint.
async fn tool_sentiment(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::sentiment::SentimentTool;
    Json(tool.execute(&body))
}

/// Direct fraud check tool endpoint.
async fn tool_fraud(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::fraud_check::FraudCheckTool;
    Json(tool.execute(&body))
}

/// List all registered tools.
async fn list_tools(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "tools": state.registry.list_tools()
    }))
}

/// Cache stats.
async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(state.cache.stats())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    tracing::info!("Initializing Rust Gateway...");

    // Build tool registry
    let registry = tools::build_default_registry();
    tracing::info!("Registered {} tools", registry.list_tools().len());

    // Build cache
    let cache = cache::Cache::new(10_000);

    let state = Arc::new(AppState { registry, cache });

    // CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Router
    let app = Router::new()
        // Core
        .route("/health", get(health))
        .route("/execute", post(execute))
        .route("/tools", get(list_tools))
        .route("/cache/stats", get(cache_stats))
        // Direct tool endpoints
        .route("/tools/search", post(tool_search))
        .route("/tools/validate-booking", post(tool_validate_booking))
        .route("/tools/pricing", post(tool_pricing))
        .route("/tools/sentiment", post(tool_sentiment))
        .route("/tools/fraud", post(tool_fraud))
        .layer(cors)
        .with_state(state);

    let port = std::env::var("RUST_PORT").unwrap_or_else(|_| "3001".to_string());
    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Rust Gateway listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
