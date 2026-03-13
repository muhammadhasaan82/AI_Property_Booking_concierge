use serde_json::{json, Value};
use crate::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// Intent inference – heuristic detection from raw JSON keys
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub enum InferredIntent {
    Search,
    Booking,
    Status,
    Payment,
    Faq,
    Sentiment,
    FraudCheck,
    Unknown,
}

impl std::fmt::Display for InferredIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferredIntent::Search => write!(f, "search"),
            InferredIntent::Booking => write!(f, "booking"),
            InferredIntent::Status => write!(f, "status"),
            InferredIntent::Payment => write!(f, "payment"),
            InferredIntent::Faq => write!(f, "faq"),
            InferredIntent::Sentiment => write!(f, "sentiment"),
            InferredIntent::FraudCheck => write!(f, "fraud_check"),
            InferredIntent::Unknown => write!(f, "unknown"),
        }
    }
}

use serde::Deserialize;
use std::fs;
use std::sync::OnceLock;

#[derive(Debug, Deserialize)]
struct IntentFeatureConfig {
    keys: Option<Vec<String>>,
    keywords: Option<Vec<String>>,
    weight: f64,
    base_score: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct IntentsConfig {
    search: IntentFeatureConfig,
    booking: IntentFeatureConfig,
    status: IntentFeatureConfig,
    payment: IntentFeatureConfig,
    faq: IntentFeatureConfig,
}

#[derive(Debug, Deserialize)]
struct GatewayConfig {
    intents: IntentsConfig,
}

fn get_config() -> &'static GatewayConfig {
    static CONFIG: OnceLock<GatewayConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let config_path = format!("{}/config/intent_features.toml", manifest_dir);
        let config_str = fs::read_to_string(&config_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", config_path));
        toml::from_str(&config_str).expect("Failed to parse intent_features.toml")
    })
}

/// Infer intent from the keys present in raw JSON data.
pub fn infer_intent(data: &Value) -> InferredIntent {
    // Explicit intent override
    if let Some(intent) = data.get("intent").and_then(|v| v.as_str()) {
        return match intent.to_lowercase().as_str() {
            "search" | "property_search" => InferredIntent::Search,
            "booking" | "book" | "reserve" => InferredIntent::Booking,
            "status" | "check_status" => InferredIntent::Status,
            "payment" | "pay" => InferredIntent::Payment,
            "faq" | "question" | "policy" => InferredIntent::Faq,
            "sentiment" | "analyze" => InferredIntent::Sentiment,
            "fraud" | "fraud_check" => InferredIntent::FraudCheck,
            _ => InferredIntent::Unknown,
        };
    }

    // Fraud check (explicit)
    if data.get("check_fraud").is_some() {
        return InferredIntent::FraudCheck;
    }

    // Sentiment (explicit)
    if data.get("analyze_sentiment").is_some()
        || data.get("review").is_some()
        || data.get("feedback").is_some()
    {
        return InferredIntent::Sentiment;
    }

    let config = get_config();
    let mut scores: Vec<(InferredIntent, f64)> = Vec::new();

    // Helper to score based on key presence
    let score_keys = |cfg: &IntentFeatureConfig| -> f64 {
        if let Some(keys) = &cfg.keys {
            let count = keys.iter().filter(|k| data.get(*k).is_some()).count();
            (count as f64) * cfg.weight + cfg.base_score.unwrap_or(0.0)
        } else {
            0.0
        }
    };

    let search_score = score_keys(&config.intents.search);
    if search_score > 0.0 { scores.push((InferredIntent::Search, search_score)); }

    let booking_score = score_keys(&config.intents.booking);
    if booking_score > 0.0 { scores.push((InferredIntent::Booking, booking_score)); }

    let status_score = score_keys(&config.intents.status);
    if status_score > 0.0 { scores.push((InferredIntent::Status, status_score)); }

    let payment_score = score_keys(&config.intents.payment);
    if payment_score > 0.0 { scores.push((InferredIntent::Payment, payment_score)); }

    // FAQ signals
    if let Some(q) = data.get("question").and_then(|v| v.as_str()) {
        let q_lower = q.to_lowercase();
        if let Some(keywords) = &config.intents.faq.keywords {
            let count = keywords.iter().filter(|w| q_lower.contains(*w)).count();
            let faq_score = (count as f64) * config.intents.faq.weight + config.intents.faq.base_score.unwrap_or(0.0);
            scores.push((InferredIntent::Faq, faq_score));
        }
    }

    // Select highest-scoring intent
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.first().map(|(intent, _)| intent.clone()).unwrap_or(InferredIntent::Unknown)
}

// ---------------------------------------------------------------------------
// Gateway request processing
// ---------------------------------------------------------------------------

/// Process an arbitrary JSON request through the autonomous gateway.
pub fn process_request(data: &Value, context: &Value, registry: &ToolRegistry) -> Value {
    let intent = infer_intent(data);

    tracing::info!(
        intent = %intent,
        keys = ?data.as_object().map(|o| o.keys().collect::<Vec<_>>()),
        "Gateway processing request"
    );

    // Safety: never execute destructive actions without required data
    if intent == InferredIntent::Booking {
        let required = ["property_id", "check_in", "check_out"];
        let missing: Vec<&str> = required.iter()
            .filter(|k| data.get(**k).is_none())
            .copied()
            .collect();
        if !missing.is_empty() {
            return json!({
                "ok": false,
                "intent": intent.to_string(),
                "error": "insufficient_data",
                "message": format!("Booking requires: {}. Please provide these fields.", missing.join(", ")),
                "missing_fields": missing
            });
        }
    }

    if intent == InferredIntent::Payment {
        if data.get("amount").is_none() {
            return json!({
                "ok": false,
                "intent": intent.to_string(),
                "error": "insufficient_data",
                "message": "Payment requires an amount. Please provide the 'amount' field."
            });
        }
    }

    // Route to tool registry for execution
    let result = registry.auto_execute(data);

    // Enrich with metadata
    json!({
        "ok": result.get("ok").unwrap_or(&json!(true)),
        "intent": intent.to_string(),
        "tool_used": result.get("tool_used").unwrap_or(&json!(null)),
        "result": result.get("result").unwrap_or(&json!(null)),
        "context_received": !context.is_null() && context.as_object().map_or(false, |o| !o.is_empty()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_search_intent() {
        let data = json!({"location": "New York", "budget": 200});
        assert_eq!(infer_intent(&data), InferredIntent::Search);
    }

    #[test]
    fn test_infer_booking_intent() {
        let data = json!({"property_id": "p1", "check_in": "2027-01-01", "check_out": "2027-01-05", "user_id": "u1"});
        assert_eq!(infer_intent(&data), InferredIntent::Booking);
    }

    #[test]
    fn test_infer_status_intent() {
        let data = json!({"booking_id": "ABC123"});
        assert_eq!(infer_intent(&data), InferredIntent::Status);
    }

    #[test]
    fn test_explicit_intent() {
        let data = json!({"intent": "faq", "question": "What is the refund policy?"});
        assert_eq!(infer_intent(&data), InferredIntent::Faq);
    }

    #[test]
    fn test_unknown_intent() {
        let data = json!({"random_key": "hello"});
        assert_eq!(infer_intent(&data), InferredIntent::Unknown);
    }
}
