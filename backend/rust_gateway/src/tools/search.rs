use serde_json::{json, Value};
use super::Tool;

/// Property search tool – filters and ranks properties from provided data.
pub struct PropertySearchTool;

impl PropertySearchTool {
    fn matches_location(row_city: &str, wanted: &str) -> bool {
        if wanted.is_empty() {
            return true;
        }
        let a = row_city.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let b = wanted.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        a == b
    }

    fn matches_amenities(row_amenities: &[String], wanted: &[String]) -> bool {
        if wanted.is_empty() {
            return true;
        }
        let row_set: std::collections::HashSet<String> =
            row_amenities.iter().map(|a| a.to_lowercase()).collect();
        wanted.iter().all(|w| row_set.contains(&w.to_lowercase()))
    }
}

impl Tool for PropertySearchTool {
    fn name(&self) -> &'static str {
        "property_search"
    }

    fn can_handle(&self, input: &Value) -> bool {
        // Triggers on location, city, budget, beds, amenities, property_type, or query
        input.get("location").is_some()
            || input.get("city").is_some()
            || input.get("budget").is_some()
            || input.get("beds").is_some()
            || input.get("amenities").is_some()
            || input.get("property_type").is_some()
            || input.get("query_text").is_some()
    }

    fn confidence(&self, input: &Value) -> f64 {
        let mut score: f64 = 0.0;
        let keys = ["location", "city", "budget", "beds", "amenities", "property_type", "query_text"];
        for key in &keys {
            if input.get(*key).is_some() {
                score += 0.15;
            }
        }
        // Boost if multiple search-related keys present
        if score > 0.3 { score += 0.1; }
        score.min(1.0)
    }

    fn execute(&self, input: &Value) -> Value {
        let location = input.get("location")
            .or_else(|| input.get("city"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let budget = input.get("budget").and_then(|v| v.as_f64());
        let beds = input.get("beds").and_then(|v| v.as_i64());
        let property_type = input.get("property_type").and_then(|v| v.as_str()).unwrap_or("");
        let query_text = input.get("query_text").and_then(|v| v.as_str()).unwrap_or("");
        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(5)
            .max(1);
        let summary_mode_threshold = input
            .get("summary_mode_threshold")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(12)
            .max(1);

        let wanted_amenities: Vec<String> = input
            .get("amenities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| a.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // If properties are provided in the input, filter them
        let properties = input.get("properties").and_then(|v| v.as_array());

        let mut results: Vec<Value> = Vec::new();

        if let Some(props) = properties {
            for p in props {
                let p_city = p.get("city").and_then(|v| v.as_str()).unwrap_or("");
                if !Self::matches_location(p_city, location) {
                    continue;
                }

                if let Some(max_budget) = budget {
                    if let Some(price) = p.get("price_per_night").and_then(|v| v.as_f64()) {
                        if price > max_budget {
                            continue;
                        }
                    }
                }

                if let Some(min_beds) = beds {
                    if let Some(b) = p.get("beds").and_then(|v| v.as_i64()) {
                        if b < min_beds {
                            continue;
                        }
                    }
                }

                if !property_type.is_empty() {
                    let p_type = p.get("property_type").and_then(|v| v.as_str()).unwrap_or("");
                    if !p_type.to_lowercase().contains(&property_type.to_lowercase()) {
                        continue;
                    }
                }

                let row_amenities: Vec<String> = p
                    .get("amenities")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|a| a.as_str().map(String::from)).collect())
                    .unwrap_or_default();

                if !Self::matches_amenities(&row_amenities, &wanted_amenities) {
                    continue;
                }

                // Text relevance check
                if !query_text.is_empty() && location.is_empty() && property_type.is_empty() {
                    let hay = format!(
                        "{} {} {} {}",
                        p.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                        p.get("property_type").and_then(|v| v.as_str()).unwrap_or(""),
                        p_city,
                        p.get("description").and_then(|v| v.as_str()).unwrap_or("")
                    )
                    .to_lowercase();
                    let has_match = query_text
                        .to_lowercase()
                        .split_whitespace()
                        .any(|tok| hay.contains(tok));
                    if !has_match {
                        continue;
                    }
                }

                results.push(p.clone());
            }

            // Sort: price ascending, then rating descending
            results.sort_by(|a, b| {
                let price_a = a.get("price_per_night").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
                let price_b = b.get("price_per_night").and_then(|v| v.as_f64()).unwrap_or(f64::MAX);
                let cmp = price_a.partial_cmp(&price_b).unwrap_or(std::cmp::Ordering::Equal);
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
                let rating_a = a.get("rating").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let rating_b = b.get("rating").and_then(|v| v.as_f64()).unwrap_or(0.0);
                rating_b.partial_cmp(&rating_a).unwrap_or(std::cmp::Ordering::Equal)
            });

        }

        let total_matches = results.len();
        if results.len() > max_results {
            results.truncate(max_results);
        }

        if total_matches > summary_mode_threshold {
            for entry in results.iter_mut() {
                if let Some(obj) = entry.as_object_mut() {
                    obj.remove("description");
                    obj.remove("amenities");
                }
            }
        }

        json!({
            "count": total_matches,
            "shown_count": results.len(),
            "max_results": max_results,
            "summary_mode": total_matches > summary_mode_threshold,
            "summary_mode_threshold": summary_mode_threshold,
            "filters_applied": {
                "location": location,
                "budget": budget,
                "beds": beds,
                "property_type": property_type,
                "amenities": wanted_amenities
            },
            "results": results
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_filters_by_budget() {
        let tool = PropertySearchTool;
        let input = json!({
            "budget": 150.0,
            "properties": [
                {"id": "p1", "title": "Cheap", "city": "NYC", "price_per_night": 100.0, "beds": 2},
                {"id": "p2", "title": "Expensive", "city": "NYC", "price_per_night": 300.0, "beds": 3}
            ]
        });
        let result = tool.execute(&input);
        assert_eq!(result["count"], 1);
        assert_eq!(result["results"][0]["id"], "p1");
    }

    #[test]
    fn test_search_filters_by_city() {
        let tool = PropertySearchTool;
        let input = json!({
            "location": "miami",
            "properties": [
                {"id": "p1", "city": "Miami", "price_per_night": 100.0},
                {"id": "p2", "city": "New York", "price_per_night": 100.0}
            ]
        });
        let result = tool.execute(&input);
        assert_eq!(result["count"], 1);
        assert_eq!(result["results"][0]["id"], "p1");
    }

    #[test]
    fn test_city_match_is_not_substring() {
        let tool = PropertySearchTool;
        let input = json!({
            "location": "York",
            "properties": [
                {"id": "p1", "city": "York", "price_per_night": 120.0},
                {"id": "p2", "city": "New York", "price_per_night": 140.0}
            ]
        });
        let result = tool.execute(&input);
        assert_eq!(result["count"], 1);
        assert_eq!(result["results"][0]["id"], "p1");
    }

    #[test]
    fn test_can_handle() {
        let tool = PropertySearchTool;
        assert!(tool.can_handle(&json!({"location": "NYC"})));
        assert!(tool.can_handle(&json!({"budget": 200})));
        assert!(!tool.can_handle(&json!({"booking_id": "abc"})));
    }
}
