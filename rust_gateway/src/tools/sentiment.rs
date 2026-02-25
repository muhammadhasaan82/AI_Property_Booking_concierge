use serde_json::{json, Value};
use std::collections::HashMap;
use super::Tool;

/// VADER-inspired sentiment analysis tool.
/// Uses a built-in lexicon for fast, deterministic scoring.
pub struct SentimentTool;

impl SentimentTool {
    fn build_lexicon() -> HashMap<&'static str, f64> {
        let mut lex = HashMap::new();
        // Positive words
        for (word, score) in [
            ("good", 1.9), ("great", 3.1), ("excellent", 3.4), ("amazing", 3.2),
            ("wonderful", 3.0), ("fantastic", 3.1), ("perfect", 3.0), ("beautiful", 2.7),
            ("love", 2.9), ("like", 1.5), ("happy", 2.7), ("best", 3.2),
            ("awesome", 3.1), ("nice", 1.8), ("lovely", 2.5), ("comfortable", 2.0),
            ("clean", 1.8), ("friendly", 2.2), ("helpful", 2.1), ("recommend", 2.3),
            ("spacious", 1.9), ("convenient", 1.8), ("pleasant", 2.0), ("enjoy", 2.3),
            ("satisfied", 2.0), ("impressed", 2.5), ("thank", 1.5), ("thanks", 1.5),
            ("please", 0.5), ("yes", 0.5), ("sure", 0.5),
        ] {
            lex.insert(word, score);
        }
        // Negative words
        for (word, score) in [
            ("bad", -2.5), ("terrible", -3.4), ("awful", -3.1), ("horrible", -3.2),
            ("poor", -2.3), ("worst", -3.5), ("dirty", -2.7), ("ugly", -2.2),
            ("hate", -3.0), ("dislike", -2.0), ("unhappy", -2.5), ("disappointed", -2.5),
            ("rude", -2.5), ("noisy", -2.0), ("uncomfortable", -2.2), ("broken", -2.3),
            ("slow", -1.5), ("expensive", -1.8), ("overpriced", -2.5), ("cold", -1.2),
            ("small", -1.0), ("cramped", -2.0), ("smelly", -2.8), ("annoying", -2.2),
            ("problem", -1.8), ("issue", -1.5), ("complaint", -2.0), ("angry", -2.7),
            ("no", -0.5), ("not", -0.5), ("never", -1.0), ("cancel", -1.0),
        ] {
            lex.insert(word, score);
        }
        // Boosters
        for (word, score) in [
            ("very", 0.0), ("really", 0.0), ("extremely", 0.0), ("absolutely", 0.0),
            ("quite", 0.0), ("so", 0.0),
        ] {
            lex.insert(word, score);
        }
        lex
    }
}

impl Tool for SentimentTool {
    fn name(&self) -> &'static str {
        "sentiment_analysis"
    }

    fn can_handle(&self, input: &Value) -> bool {
        input.get("text").is_some()
            || input.get("message").is_some()
            || input.get("review").is_some()
            || input.get("feedback").is_some()
    }

    fn confidence(&self, input: &Value) -> f64 {
        // Sentiment is a support tool — lower priority unless explicitly about text analysis
        if input.get("analyze_sentiment").is_some() { return 0.9; }
        if input.get("feedback").is_some() || input.get("review").is_some() { return 0.7; }
        if input.get("text").is_some() || input.get("message").is_some() {
            // Low priority to avoid stealing from other tools
            return 0.15;
        }
        0.0
    }

    fn execute(&self, input: &Value) -> Value {
        let text = input.get("text")
            .or_else(|| input.get("message"))
            .or_else(|| input.get("review"))
            .or_else(|| input.get("feedback"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let lexicon = Self::build_lexicon();
        let words: Vec<&str> = text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        let mut pos_sum = 0.0_f64;
        let mut neg_sum = 0.0_f64;
        let mut neu_count = 0_usize;
        let mut matched_words: Vec<(&str, f64)> = Vec::new();
        let boosters = ["very", "really", "extremely", "absolutely", "quite", "so"];

        for (i, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            if let Some(&score) = lexicon.get(lower.as_str()) {
                if score == 0.0 { continue; } // booster, skip direct scoring

                // Apply booster if previous word is an intensifier
                let mut final_score = score;
                if i > 0 {
                    let prev = words[i - 1].to_lowercase();
                    if boosters.contains(&prev.as_str()) {
                        final_score *= 1.5;
                    }
                }

                // Negation handling
                if i > 0 {
                    let prev = words[i - 1].to_lowercase();
                    if prev == "not" || prev == "no" || prev == "never" || prev.ends_with("n't") {
                        final_score *= -0.75;
                    }
                }

                matched_words.push((word, final_score));
                if final_score > 0.0 {
                    pos_sum += final_score;
                } else {
                    neg_sum += final_score.abs();
                }
            } else {
                neu_count += 1;
            }
        }

        let total = pos_sum + neg_sum + neu_count as f64;
        let (pos_ratio, neg_ratio, neu_ratio) = if total > 0.0 {
            (pos_sum / total, neg_sum / total, neu_count as f64 / total)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Compound score normalized to [-1, 1]
        let raw_compound = pos_sum - neg_sum;
        let compound = raw_compound / (raw_compound.abs() + 15.0).sqrt();

        let label = if compound >= 0.05 {
            "positive"
        } else if compound <= -0.05 {
            "negative"
        } else {
            "neutral"
        };

        json!({
            "label": label,
            "compound": (compound * 1000.0).round() / 1000.0,
            "scores": {
                "positive": (pos_ratio * 1000.0).round() / 1000.0,
                "negative": (neg_ratio * 1000.0).round() / 1000.0,
                "neutral": (neu_ratio * 1000.0).round() / 1000.0
            },
            "word_count": words.len(),
            "matched_words": matched_words.len()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let tool = SentimentTool;
        let input = json!({"text": "This hotel is amazing and the staff is very friendly"});
        let result = tool.execute(&input);
        assert_eq!(result["label"], "positive");
        assert!(result["compound"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn test_negative_sentiment() {
        let tool = SentimentTool;
        let input = json!({"text": "Terrible experience, the room was dirty and the service was awful"});
        let result = tool.execute(&input);
        assert_eq!(result["label"], "negative");
        assert!(result["compound"].as_f64().unwrap() < 0.0);
    }

    #[test]
    fn test_neutral_sentiment() {
        let tool = SentimentTool;
        let input = json!({"text": "I checked in at noon on Wednesday"});
        let result = tool.execute(&input);
        assert_eq!(result["label"], "neutral");
    }
}
