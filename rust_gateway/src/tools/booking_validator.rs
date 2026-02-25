use serde_json::{json, Value};
use chrono::NaiveDate;
use super::Tool;

/// Validates booking requests for completeness, date logic, and safety.
pub struct BookingValidatorTool;

impl Tool for BookingValidatorTool {
    fn name(&self) -> &'static str {
        "booking_validator"
    }

    fn can_handle(&self, input: &Value) -> bool {
        input.get("property_id").is_some()
            && (input.get("check_in").is_some() || input.get("check_out").is_some())
    }

    fn confidence(&self, input: &Value) -> f64 {
        let mut score = 0.0;
        let booking_keys = ["property_id", "check_in", "check_out", "user_id", "guests", "name", "email"];
        for key in &booking_keys {
            if input.get(*key).is_some() {
                score += 0.15;
            }
        }
        score.min(1.0)
    }

    fn execute(&self, input: &Value) -> Value {
        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // Required fields
        let property_id = input.get("property_id").and_then(|v| v.as_str()).unwrap_or("");
        if property_id.is_empty() {
            errors.push("property_id is required".to_string());
        }

        let check_in_str = input.get("check_in").and_then(|v| v.as_str()).unwrap_or("");
        let check_out_str = input.get("check_out").and_then(|v| v.as_str()).unwrap_or("");

        if check_in_str.is_empty() {
            errors.push("check_in date is required (YYYY-MM-DD)".to_string());
        }
        if check_out_str.is_empty() {
            errors.push("check_out date is required (YYYY-MM-DD)".to_string());
        }

        // Parse dates
        let check_in = NaiveDate::parse_from_str(check_in_str, "%Y-%m-%d").ok();
        let check_out = NaiveDate::parse_from_str(check_out_str, "%Y-%m-%d").ok();

        if !check_in_str.is_empty() && check_in.is_none() {
            errors.push(format!("Invalid check_in date format: '{}'. Use YYYY-MM-DD.", check_in_str));
        }
        if !check_out_str.is_empty() && check_out.is_none() {
            errors.push(format!("Invalid check_out date format: '{}'. Use YYYY-MM-DD.", check_out_str));
        }

        let mut nights: i64 = 0;
        if let (Some(ci), Some(co)) = (check_in, check_out) {
            let diff = co.signed_duration_since(ci).num_days();
            if diff <= 0 {
                errors.push("check_out must be after check_in".to_string());
            } else {
                nights = diff;
                if nights > 365 {
                    warnings.push("Booking is longer than 1 year — please verify.".to_string());
                }
            }

            // Past date check
            let today = chrono::Local::now().date_naive();
            if ci < today {
                errors.push("check_in date is in the past".to_string());
            }
        }

        // Guest validation
        let guests = input.get("guests").and_then(|v| v.as_i64()).unwrap_or(1);
        if guests < 1 {
            errors.push("guests must be at least 1".to_string());
        }
        if guests > 20 {
            warnings.push(format!("Large party ({} guests) — may require special arrangements.", guests));
        }

        // User info validation
        if let Some(email) = input.get("email").and_then(|v| v.as_str()) {
            if !email.contains('@') || !email.contains('.') {
                errors.push(format!("Invalid email format: '{}'", email));
            }
        }

        let valid = errors.is_empty();

        json!({
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "nights": nights,
            "guests": guests,
            "property_id": property_id,
            "check_in": check_in_str,
            "check_out": check_out_str
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_booking() {
        let tool = BookingValidatorTool;
        let input = json!({
            "property_id": "p1",
            "check_in": "2027-06-01",
            "check_out": "2027-06-05",
            "guests": 2,
            "email": "test@example.com"
        });
        let result = tool.execute(&input);
        assert_eq!(result["valid"], true);
        assert_eq!(result["nights"], 4);
        assert_eq!(result["errors"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_checkout_before_checkin() {
        let tool = BookingValidatorTool;
        let input = json!({
            "property_id": "p1",
            "check_in": "2027-06-05",
            "check_out": "2027-06-01",
            "guests": 1
        });
        let result = tool.execute(&input);
        assert_eq!(result["valid"], false);
        let errors: Vec<String> = result["errors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_str().unwrap().to_string())
            .collect();
        assert!(errors.iter().any(|e| e.contains("check_out must be after")));
    }

    #[test]
    fn test_missing_fields() {
        let tool = BookingValidatorTool;
        let input = json!({
            "property_id": "",
            "check_in": "",
            "check_out": ""
        });
        let result = tool.execute(&input);
        assert_eq!(result["valid"], false);
        assert!(result["errors"].as_array().unwrap().len() >= 3);
    }
}
