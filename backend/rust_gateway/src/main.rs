mod gateway;
mod tools;
mod cache;
mod toon;
mod config;
mod cag;

use crate::tools::Tool;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{StatusCode, HeaderMap},
    routing::{get, post},
    Json, Router,
};
use axum::body::Body;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

// ---------------------------------------------------------------------------
// Application state shared across handlers
// ---------------------------------------------------------------------------
struct AppState {
    registry: tools::ToolRegistry,
    cache: cache::Cache,
    cag_store: cag::CagStore,
    thresholds: config::ThresholdsConfig,
    vader_lexicon: config::VaderLexiconConfig,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Health check.
async fn health() -> Json<Value> {
    Json(json!({"ok": true, "service": "rust_gateway", "version": "0.1.0"}))
}

/// Schema-agnostic autonomous gateway.
/// Accepts JSON or TOON payloads, infers intent, routes to the best tool.
/// Responds in TOON if Accept header requests it.
async fn execute(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Parse body based on Content-Type
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json");

    let parsed: Value = if content_type.contains("toon") {
        match toon::decode(&body) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({"ok": false, "error": format!("TOON parse error: {}", e)});
                return make_response(&headers, &err);
            }
        }
    } else {
        match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({"ok": false, "error": format!("JSON parse error: {}", e)});
                return make_response(&headers, &err);
            }
        }
    };

    let data = parsed.get("data").cloned().unwrap_or(parsed.clone());
    let context = parsed.get("context").cloned().unwrap_or(json!({}));

    // ── CAG Zero-Latency Intercept ──────────────────────────────────
    // Check if this is a FAQ/policy query that can be answered from RAM.
    if let Some(query) = data.get("question").and_then(|v| v.as_str()) {
        if let Some(hit) = state.cag_store.try_intercept(query) {
            tracing::info!(
                "[CAG] Cache Hit: Intercepted query '{}' (policy={}, match={}, score={:.3})",
                query, hit.policy_id, hit.match_type, hit.score
            );
            let cag_response = json!({
                "ok": true,
                "intent": "faq",
                "cached": true,
                "cag": true,
                "policy_id": hit.policy_id,
                "match_type": hit.match_type,
                "match_score": hit.score,
                "answer": hit.answer,
            });
            return make_response(&headers, &cag_response);
        } else {
            tracing::info!("[CAG] Cache Miss: Passing to database for query '{}'", query);
        }
    }

    // ── Generic result cache ────────────────────────────────────────
    let cache_key = cache::cache_key("execute", &data);
    if let Some(cached) = state.cache.get(&cache_key) {
        let mut result = cached;
        if let Some(obj) = result.as_object_mut() {
            obj.insert("cached".to_string(), json!(true));
        }
        return make_response(&headers, &result);
    }

    // Process through gateway
    let result = gateway::process_request(&data, &context, &state.registry);

    // Cache successful results
    if result.get("ok") == Some(&json!(true)) {
        let intent = result.get("intent").and_then(|v| v.as_str()).unwrap_or("unknown");
        let ttl = match intent {
            "search" => cache::ttl::PROPERTY_SEARCH,
            "booking" | "status" => cache::ttl::SESSION_STATE,
            "faq" => cache::ttl::FAQ_ANSWER,
            _ => cache::ttl::PRICING,
        };
        state.cache.set(cache_key, result.clone(), ttl);
    }

    make_response(&headers, &result)
}

/// Build a response in JSON or TOON based on the Accept header.
fn make_response(headers: &HeaderMap, value: &Value) -> Response {
    let accept = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json");

    if accept.contains("toon") {
        let toon_body = toon::encode(value);
        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", toon::CONTENT_TYPE)
            .body(Body::from(toon_body))
            .unwrap_or_else(|_| {
                Json(json!({"ok": false, "error": "Response build error"})).into_response()
            })
    } else {
        Json(value.clone()).into_response()
    }
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
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::pricing::PricingTool::new(state.thresholds.pricing.clone());
    Json(tool.execute(&body))
}

/// Direct sentiment analysis tool endpoint.
async fn tool_sentiment(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::sentiment::SentimentTool::new(state.vader_lexicon.words.clone());
    Json(tool.execute(&body))
}

/// Direct fraud check tool endpoint.
async fn tool_fraud(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let tool = tools::fraud_check::FraudCheckTool::new(state.thresholds.fraud.clone());
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

/// CAG policy stats.
async fn cag_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(state.cag_store.stats())
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

    let thresholds = config::load_thresholds_config();
    let vader_lexicon = config::load_vader_lexicon();

    // Build tool registry
    let registry = tools::build_default_registry(&thresholds, &vader_lexicon);
    tracing::info!("Registered {} tools", registry.list_tools().len());

    // Build CAG store
    let cag_config = config::load_cag_config();
    let cag_store = cag::CagStore::new(cag_config);
    tracing::info!("[CAG] Initialized with {} policies", cag_store.policy_count());

    // Build cache
    let cache = cache::Cache::new(10_000);

    let state = Arc::new(AppState {
        registry,
        cache,
        cag_store,
        thresholds,
        vader_lexicon,
    });

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
        .route("/cag/stats", get(cag_stats))
        // Direct tool endpoints
        .route("/tools/search", post(tool_search))
        .route("/tools/validate-booking", post(tool_validate_booking))
        .route("/tools/pricing", post(tool_pricing))
        .route("/tools/sentiment", post(tool_sentiment))
        .route("/tools/fraud", post(tool_fraud))
        .layer(DefaultBodyLimit::disable())
        .layer(cors)
        .with_state(state);

    let port = std::env::var("RUST_PORT").unwrap_or_else(|_| "3001".to_string());
    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Rust Gateway listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
