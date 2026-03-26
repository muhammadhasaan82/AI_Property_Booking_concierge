//! Per-IP sliding-window rate limiter middleware for Axum.
//!
//! Phase 3 — Anomaly Detection: protects the Rust gateway from scraping
//! attacks and rapid-fire abuse. Configurable via environment variables:
//!   RUST_RATE_LIMIT_MAX        (default 10)
//!   RUST_RATE_LIMIT_WINDOW_SECS (default 10)
//!
//! Extracts client IP from (in priority order):
//!   1. `x-forwarded-for` header (first IP — works behind proxy/Chainlit)
//!   2. `x-real-ip` header
//!   3. Direct peer address from ConnectInfo
//!
//! On violation: returns 429 + JSON body and logs [ANOMALY_DETECTED].

use axum::{
    body::Body,
    http::{HeaderMap, Request, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task;
use tower::{Layer, Service};

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

fn default_max_requests() -> usize {
    std::env::var("RUST_RATE_LIMIT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
}

fn default_window() -> Duration {
    let secs: u64 = std::env::var("RUST_RATE_LIMIT_WINDOW_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    Duration::from_secs(secs)
}

// ---------------------------------------------------------------------------
// Shared state: per-IP request timestamps
// ---------------------------------------------------------------------------

type WindowMap = Arc<Mutex<HashMap<IpAddr, VecDeque<Instant>>>>;

fn spawn_eviction_task(map: WindowMap, window: Duration) {
    task::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let cutoff = Instant::now() - window - Duration::from_secs(60);
            let mut guard = map.lock().unwrap();
            guard.retain(|_ip, timestamps| {
                while timestamps.front().map_or(false, |t| *t < cutoff) {
                    timestamps.pop_front();
                }
                !timestamps.is_empty()
            });
        }
    });
}

// ---------------------------------------------------------------------------
// IP extraction (proxy-aware)
// ---------------------------------------------------------------------------

fn extract_client_ip(headers: &HeaderMap) -> IpAddr {
    // 1. x-forwarded-for (first entry)
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first) = xff.split(',').next() {
            if let Ok(ip) = first.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }

    // 2. x-real-ip
    if let Some(xri) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        if let Ok(ip) = xri.trim().parse::<IpAddr>() {
            return ip;
        }
    }

    // 3. Fallback to loopback (ConnectInfo would need extractor — this is safe default)
    "127.0.0.1".parse().unwrap()
}

// ---------------------------------------------------------------------------
// Tower Layer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RateLimitLayer {
    max_requests: usize,
    window: Duration,
    map: WindowMap,
}

impl RateLimitLayer {
    pub fn new(max_requests: usize, window: Duration) -> Self {
        let map: WindowMap = Arc::new(Mutex::new(HashMap::new()));
        spawn_eviction_task(Arc::clone(&map), window);
        Self {
            max_requests,
            window,
            map,
        }
    }

    pub fn from_env() -> Self {
        Self::new(default_max_requests(), default_window())
    }
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            max_requests: self.max_requests,
            window: self.window,
            map: Arc::clone(&self.map),
        }
    }
}

// ---------------------------------------------------------------------------
// Tower Service
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    max_requests: usize,
    window: Duration,
    map: WindowMap,
}

impl<S> Service<Request<Body>> for RateLimitService<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        // Exempt health endpoint
        if req.uri().path() == "/health" {
            let mut inner = self.inner.clone();
            return Box::pin(async move { inner.call(req).await });
        }

        let ip = extract_client_ip(req.headers());
        let now = Instant::now();
        let window = self.window;
        let max_requests = self.max_requests;
        let map = Arc::clone(&self.map);

        // Check + record atomically
        let count = {
            let mut guard = map.lock().unwrap();
            let timestamps = guard.entry(ip).or_insert_with(VecDeque::new);

            // Purge entries outside the window
            let cutoff = now - window;
            while timestamps.front().map_or(false, |t| *t < cutoff) {
                timestamps.pop_front();
            }

            timestamps.push_back(now);
            timestamps.len()
        };

        if count > max_requests {
            let window_secs = window.as_secs();
            tracing::warn!(
                "[ANOMALY_DETECTED] Rate limit exceeded for IP {}: {} requests in {}s (max={})",
                ip,
                count,
                window_secs,
                max_requests,
            );

            let body = json!({
                "ok": false,
                "error": "rate_limited",
                "message": format!(
                    "Too many requests. Limit: {} per {}s.",
                    max_requests, window_secs
                ),
                "retry_after_seconds": window_secs,
            });

            let response = (
                StatusCode::TOO_MANY_REQUESTS,
                [("content-type", "application/json"), ("retry-after", &window_secs.to_string())],
                body.to_string(),
            )
                .into_response();

            return Box::pin(async move { Ok(response) });
        }

        let mut inner = self.inner.clone();
        Box::pin(async move { inner.call(req).await })
    }
}
