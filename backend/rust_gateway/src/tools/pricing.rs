use serde_json::{json, Value};
use super::Tool;
use crate::config::PricingThresholds;

/// Computes pricing: nights × rate, optional tax, seasonal multipliers.
pub struct PricingTool {
    thresholds: PricingThresholds,
}

impl PricingTool {
    pub fn new(thresholds: PricingThresholds) -> Self {
        Self { thresholds }
    }
}

impl Tool for PricingTool {
    fn name(&self) -> &'static str {
        "pricing"
    }

    fn can_handle(&self, input: &Value) -> bool {
        input.get("price_per_night").is_some()
            || (input.get("amount").is_some() && input.get("currency").is_some())
    }

    fn confidence(&self, input: &Value) -> f64 {
        let mut score: f64 = 0.0;
        if input.get("price_per_night").is_some() { score += 0.3; }
        if input.get("nights").is_some() { score += 0.2; }
        if input.get("check_in").is_some() && input.get("check_out").is_some() { score += 0.2; }
        if input.get("amount").is_some() { score += 0.2; }
        if input.get("currency").is_some() { score += 0.1; }
        // Lower priority than booking_validator if booking keys present
        if input.get("property_id").is_some() { score -= 0.1; }
        score.max(0.0).min(1.0)
    }

    fn execute(&self, input: &Value) -> Value {
        let price_per_night = input
            .get("price_per_night")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Calculate nights from dates or direct value
        let nights = if let Some(n) = input.get("nights").and_then(|v| v.as_i64()) {
            n
        } else {
            let ci = input.get("check_in").and_then(|v| v.as_str()).unwrap_or("");
            let co = input.get("check_out").and_then(|v| v.as_str()).unwrap_or("");
            if let (Ok(d1), Ok(d2)) = (
                chrono::NaiveDate::parse_from_str(ci, "%Y-%m-%d"),
                chrono::NaiveDate::parse_from_str(co, "%Y-%m-%d"),
            ) {
                let diff = d2.signed_duration_since(d1).num_days();
                if diff > 0 { diff } else { 1 }
            } else {
                1
            }
        };

        let guests = input.get("guests").and_then(|v| v.as_i64()).unwrap_or(1);

        // Seasonal multiplier (simple: Dec-Feb = 1.2, Jun-Aug = 1.15, else 1.0)
        let season_multiplier = if let Some(ci_str) = input.get("check_in").and_then(|v| v.as_str()) {
            if let Ok(d) = chrono::NaiveDate::parse_from_str(ci_str, "%Y-%m-%d") {
                match d.format("%m").to_string().parse::<u32>().unwrap_or(1) {
                    12 | 1 | 2 => self.thresholds.peak_multiplier,
                    6 | 7 | 8 => self.thresholds.shoulder_multiplier,
                    _ => 1.0,
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        let subtotal = price_per_night * nights as f64 * season_multiplier;
        let tax_rate = input.get("tax_rate").and_then(|v| v.as_f64()).unwrap_or(self.thresholds.tax_rate);
        let tax = subtotal * tax_rate;
        let total = subtotal + tax;

        let currency = input.get("currency").and_then(|v| v.as_str()).unwrap_or("USD");

        json!({
            "price_per_night": price_per_night,
            "nights": nights,
            "guests": guests,
            "season_multiplier": season_multiplier,
            "subtotal": (subtotal * 100.0).round() / 100.0,
            "tax_rate": tax_rate,
            "tax": (tax * 100.0).round() / 100.0,
            "total": (total * 100.0).round() / 100.0,
            "currency": currency
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pricing() {
        let tool = PricingTool::new(PricingThresholds::default());
        let input = json!({
            "price_per_night": 100.0,
            "nights": 3,
            "guests": 2
        });
        let result = tool.execute(&input);
        // No seasonal multiplier without dates, subtotal = 100 * 3 * 1.0 = 300
        assert_eq!(result["subtotal"], 300.0);
        assert_eq!(result["nights"], 3);
        // Default 10% tax
        assert_eq!(result["tax"], 30.0);
        assert_eq!(result["total"], 330.0);
    }

    #[test]
    fn test_pricing_with_dates() {
        let tool = PricingTool::new(PricingThresholds::default());
        let input = json!({
            "price_per_night": 200.0,
            "check_in": "2027-03-10",
            "check_out": "2027-03-15",
            "tax_rate": 0.08
        });
        let result = tool.execute(&input);
        assert_eq!(result["nights"], 5);
        // March = off-season (1.0x), subtotal = 200 * 5 = 1000
        assert_eq!(result["subtotal"], 1000.0);
    }
}
