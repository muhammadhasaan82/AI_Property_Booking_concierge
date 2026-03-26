// rust_gateway/src/cag.rs
//! Cache-Augmented Generation (CAG) — Zero-Latency FAQ Intercept Layer.
//!
//! Pre-computed policy answers are loaded from `config/cag_policies.toml` at
//! startup and served from RAM. The matching engine uses a two-pass strategy:
//!   1. Keyword-threshold matching (fast, no allocations beyond normalization).
//!   2. Jaro-Winkler fuzzy matching against canonical question variants.
//!
//! If neither pass produces a hit, the request falls through to the standard
//! database/LLM pipeline without interruption.

use serde::Deserialize;
use serde_json::Value;
use strsim::jaro_winkler;

// ─────────────────────────────────────────────────────────────────────
// Config structs (deserialized from TOML)
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct CagSettings {
    pub keyword_threshold: f64,
    pub fuzzy_threshold: f64,
    pub ttl_seconds: u64,
}

impl Default for CagSettings {
    fn default() -> Self {
        Self {
            keyword_threshold: 0.6,
            fuzzy_threshold: 0.82,
            ttl_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CagPolicy {
    pub id: String,
    pub answer: String,
    pub keywords: Vec<String>,
    pub canonical_questions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CagConfig {
    pub settings: CagSettings,
    pub policies: Vec<CagPolicy>,
}

impl Default for CagConfig {
    fn default() -> Self {
        Self {
            settings: CagSettings::default(),
            policies: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// CAG Hit result
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CagHit {
    pub policy_id: String,
    pub answer: String,
    pub match_type: &'static str,
    pub score: f64,
}

// ─────────────────────────────────────────────────────────────────────
// CAG Store — the runtime intercept engine
// ─────────────────────────────────────────────────────────────────────

pub struct CagStore {
    policies: Vec<CagPolicy>,
    settings: CagSettings,
}

impl CagStore {
    /// Build a new store from a parsed config.
    pub fn new(config: CagConfig) -> Self {
        Self {
            policies: config.policies,
            settings: config.settings,
        }
    }

    /// Build an empty (no-op) store. All queries will miss.
    pub fn empty() -> Self {
        Self {
            policies: Vec::new(),
            settings: CagSettings::default(),
        }
    }

    /// Number of loaded policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    /// Try to intercept a query. Returns `Some(CagHit)` on match, `None` on miss.
    pub fn try_intercept(&self, query: &str) -> Option<CagHit> {
        if self.policies.is_empty() {
            return None;
        }

        let normalized = normalize(query);
        if normalized.is_empty() {
            return None;
        }

        // Pass 1: Keyword-threshold matching
        if let Some(hit) = self.keyword_match(&normalized) {
            return Some(hit);
        }

        // Pass 2: Fuzzy matching against canonical questions
        if let Some(hit) = self.fuzzy_match(&normalized) {
            return Some(hit);
        }

        None
    }

    /// Pass 1 — count keyword hits per policy, return the best above threshold.
    fn keyword_match(&self, normalized_query: &str) -> Option<CagHit> {
        let mut best: Option<(usize, f64)> = None; // (policy index, score)

        for (i, policy) in self.policies.iter().enumerate() {
            if policy.keywords.is_empty() {
                continue;
            }
            let matched = policy
                .keywords
                .iter()
                .filter(|kw| normalized_query.contains(kw.as_str()))
                .count();
            let score = matched as f64 / policy.keywords.len() as f64;

            if score >= self.settings.keyword_threshold {
                if best.is_none() || score > best.unwrap().1 {
                    best = Some((i, score));
                }
            }
        }

        best.map(|(idx, score)| {
            let policy = &self.policies[idx];
            CagHit {
                policy_id: policy.id.clone(),
                answer: policy.answer.clone(),
                match_type: "keyword",
                score,
            }
        })
    }

    /// Pass 2 — Jaro-Winkler similarity against each canonical question.
    fn fuzzy_match(&self, normalized_query: &str) -> Option<CagHit> {
        let mut best: Option<(usize, f64)> = None; // (policy index, best score)

        for (i, policy) in self.policies.iter().enumerate() {
            for canonical in &policy.canonical_questions {
                let canonical_norm = normalize(canonical);
                let sim = jaro_winkler(normalized_query, &canonical_norm);

                if sim >= self.settings.fuzzy_threshold {
                    if best.is_none() || sim > best.unwrap().1 {
                        best = Some((i, sim));
                    }
                }
            }
        }

        best.map(|(idx, score)| {
            let policy = &self.policies[idx];
            CagHit {
                policy_id: policy.id.clone(),
                answer: policy.answer.clone(),
                match_type: "fuzzy",
                score,
            }
        })
    }

    /// Stats for the `/cag/stats` endpoint.
    pub fn stats(&self) -> Value {
        serde_json::json!({
            "policy_count": self.policies.len(),
            "keyword_threshold": self.settings.keyword_threshold,
            "fuzzy_threshold": self.settings.fuzzy_threshold,
            "ttl_seconds": self.settings.ttl_seconds,
            "policy_ids": self.policies.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        })
    }
}

/// Normalize a query string for matching: lowercase, strip punctuation, collapse whitespace.
fn normalize(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> CagConfig {
        CagConfig {
            settings: CagSettings {
                keyword_threshold: 0.6,
                fuzzy_threshold: 0.82,
                ttl_seconds: 3600,
            },
            policies: vec![
                CagPolicy {
                    id: "check_in_time".to_string(),
                    answer: "Check-in is from 3:00 PM.".to_string(),
                    keywords: vec![
                        "check-in".to_string(),
                        "check".to_string(),
                        "time".to_string(),
                        "arrive".to_string(),
                        "arrival".to_string(),
                    ],
                    canonical_questions: vec![
                        "what time is check-in".to_string(),
                        "when can i check in".to_string(),
                    ],
                },
                CagPolicy {
                    id: "pet_policy".to_string(),
                    answer: "Pet policies vary by property.".to_string(),
                    keywords: vec![
                        "pet".to_string(),
                        "pets".to_string(),
                        "dog".to_string(),
                        "cat".to_string(),
                        "animal".to_string(),
                    ],
                    canonical_questions: vec![
                        "do you allow pets".to_string(),
                        "are pets allowed".to_string(),
                        "can i bring my dog".to_string(),
                    ],
                },
            ],
        }
    }

    #[test]
    fn test_keyword_hit() {
        let store = CagStore::new(sample_config());
        // "check-in time" matches 2/5 keywords for check_in_time => 0.4, but
        // "what time is check-in" matches 3/5 => 0.6, exactly at threshold
        let hit = store.try_intercept("What time is check-in?");
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.policy_id, "check_in_time");
        assert_eq!(hit.match_type, "keyword");
    }

    #[test]
    fn test_fuzzy_hit() {
        let store = CagStore::new(sample_config());
        // Close phrasing should trigger fuzzy match
        let hit = store.try_intercept("when can I check in please");
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.policy_id, "check_in_time");
    }

    #[test]
    fn test_miss() {
        let store = CagStore::new(sample_config());
        let hit = store.try_intercept("I want to book a property in New York");
        assert!(hit.is_none());
    }

    #[test]
    fn test_empty_store() {
        let store = CagStore::empty();
        let hit = store.try_intercept("what time is check-in");
        assert!(hit.is_none());
    }

    #[test]
    fn test_empty_query() {
        let store = CagStore::new(sample_config());
        let hit = store.try_intercept("");
        assert!(hit.is_none());
        let hit2 = store.try_intercept("   ");
        assert!(hit2.is_none());
    }

    #[test]
    fn test_pet_keyword_match() {
        let store = CagStore::new(sample_config());
        // "do you allow pets" -> matches pet, pets, allow? No "allow" in keywords,
        // but "pet" and "pets" = 2/5 = 0.4. Need fuzzy fallback.
        // Actually "do you allow pets" is a canonical question, so fuzzy should hit.
        let hit = store.try_intercept("do you allow pets");
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().policy_id, "pet_policy");
    }

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("  What TIME is Check-In??  "), "what time is check-in");
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("!!!"), "");
    }

    #[test]
    fn test_stats() {
        let store = CagStore::new(sample_config());
        let stats = store.stats();
        assert_eq!(stats["policy_count"], 2);
        assert_eq!(stats["keyword_threshold"], 0.6);
    }
}
