use serde_json::{json, Value};
use super::Tool;
use crate::config::FraudThresholds;

/// Fraud and sanity checks on booking/payment data.
pub struct FraudCheckTool {
    thresholds: FraudThresholds,
}

impl FraudCheckTool {
    pub fn new(thresholds: FraudThresholds) -> Self {
        Self { thresholds }
    }
}

impl Tool for FraudCheckTool {
    fn name(&self) -> &'static str {
        "fraud_check"
    }

    fn can_handle(&self, input: &Value) -> bool {
        // Only handle if explicitly requested or has fraud-related context
        input.get("check_fraud").is_some()
            || (input.get("email").is_some() && input.get("phone").is_some() && input.get("amount").is_some())
    }

    fn confidence(&self, input: &Value) -> f64 {
        if input.get("check_fraud").is_some() { return 0.95; }
        let mut score: f64 = 0.0;
        if input.get("email").is_some() { score += 0.1; }
        if input.get("phone").is_some() { score += 0.1; }
        if input.get("amount").is_some() { score += 0.1; }
        if input.get("guests").is_some() { score += 0.05; }
        score.min(0.4) // Stay below other tools unless explicitly requested
    }

    fn execute(&self, input: &Value) -> Value {
        let mut flags: Vec<String> = Vec::new();
        let mut risk_score: f64 = 0.0;

        // Email checks
        if let Some(email) = input.get("email").and_then(|v| v.as_str()) {
            if !email.contains('@') || !email.contains('.') {
                flags.push("invalid_email_format".to_string());
                risk_score += 30.0;
            }
            // Disposable email domains
            // TODO: Move this list to config/tool_registry.toml for operational updates without recompilation.
            let disposable = ["tempmail.com", "throwaway.email", "guerrillamail.com",
                            "mailinator.com", "10minutemail.com", "yopmail.com"];
            let domain = email.split('@').last().unwrap_or("");
            if disposable.iter().any(|d| domain == *d) {
                flags.push("disposable_email".to_string());
                risk_score += 25.0;
            }
        }

        // Phone checks
        if let Some(phone) = input.get("phone").and_then(|v| v.as_str()) {
            let digits: String = phone.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() < 7 {
                flags.push("phone_too_short".to_string());
                risk_score += 15.0;
            }
            if digits.len() > 15 {
                flags.push("phone_too_long".to_string());
                risk_score += 10.0;
            }
            // All same digits
            if digits.len() > 3 && digits.chars().all(|c| c == digits.chars().next().unwrap()) {
                flags.push("phone_all_same_digits".to_string());
                risk_score += 20.0;
            }
        }

        // Amount checks
        if let Some(amount) = input.get("amount").and_then(|v| v.as_f64()) {
            if amount > self.thresholds.amount_limit {
                flags.push("unusually_high_amount".to_string());
                risk_score += 20.0;
            }
            if amount <= 0.0 {
                flags.push("invalid_amount".to_string());
                risk_score += 30.0;
            }
        }

        // Guest count anomaly
        if let Some(guests) = input.get("guests").and_then(|v| v.as_i64()) {
            if guests > 10 {
                flags.push(format!("large_party_{}_guests", guests));
                risk_score += 10.0;
            }
            if guests > 20 {
                risk_score += 15.0;
            }
        }

        // Nights anomaly
        if let Some(nights) = input.get("nights").and_then(|v| v.as_i64()) {
            if nights > 30 {
                flags.push(format!("extended_stay_{}_nights", nights));
                risk_score += 10.0;
            }
            if nights > 90 {
                risk_score += 15.0;
            }
        }

        let risk_level = if risk_score >= self.thresholds.high_risk {
            "high"
        } else if risk_score >= self.thresholds.medium_risk {
            "medium"
        } else {
            "low"
        };

        json!({
            "risk_level": risk_level,
            "risk_score": risk_score,
            "flags": flags,
            "recommendation": match risk_level {
                "high" => "Block or require manual review before processing.",
                "medium" => "Proceed with caution. Consider additional verification.",
                _ => "Safe to proceed."
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_risk() {
        let tool = FraudCheckTool::new(FraudThresholds::default());
        let input = json!({
            "check_fraud": true,
            "email": "john@gmail.com",
            "phone": "+15551234567",
            "amount": 500.0,
            "guests": 2
        });
        let result = tool.execute(&input);
        assert_eq!(result["risk_level"], "low");
    }

    #[test]
    fn test_high_risk_disposable_email() {
        let tool = FraudCheckTool::new(FraudThresholds::default());
        let input = json!({
            "check_fraud": true,
            "email": "fake@mailinator.com",
            "phone": "1111111",
            "amount": 60000.0,
            "guests": 25
        });
        let result = tool.execute(&input);
        assert_eq!(result["risk_level"], "high");
        assert!(result["flags"].as_array().unwrap().len() > 0);
    }
}
