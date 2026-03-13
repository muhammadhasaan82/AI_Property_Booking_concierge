use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, Deserialize)]
pub struct FraudThresholds {
    pub high_risk: f64,
    pub medium_risk: f64,
    pub amount_limit: f64,
}

impl Default for FraudThresholds {
    fn default() -> Self {
        Self {
            high_risk: 50.0,
            medium_risk: 20.0,
            amount_limit: 50_000.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PricingThresholds {
    pub peak_multiplier: f64,
    pub shoulder_multiplier: f64,
    pub tax_rate: f64,
}

impl Default for PricingThresholds {
    fn default() -> Self {
        Self {
            peak_multiplier: 1.20,
            shoulder_multiplier: 1.15,
            tax_rate: 0.10,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ThresholdsConfig {
    pub fraud: FraudThresholds,
    pub pricing: PricingThresholds,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct VaderLexiconConfig {
    pub words: HashMap<String, f64>,
}

fn read_toml_file(path: &str) -> Option<String> {
    match fs::read_to_string(path) {
        Ok(contents) => Some(contents),
        Err(err) => {
            tracing::warn!("Failed to read {}: {}", path, err);
            None
        }
    }
}

pub fn load_thresholds_config() -> ThresholdsConfig {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = format!("{}/config/thresholds.toml", manifest_dir);
    if let Some(contents) = read_toml_file(&path) {
        match toml::from_str::<ThresholdsConfig>(&contents) {
            Ok(cfg) => return cfg,
            Err(err) => tracing::warn!("Failed to parse {}: {}", path, err),
        }
    }
    ThresholdsConfig::default()
}

pub fn load_vader_lexicon() -> VaderLexiconConfig {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = format!("{}/config/vader_lexicon.toml", manifest_dir);
    if let Some(contents) = read_toml_file(&path) {
        match toml::from_str::<VaderLexiconConfig>(&contents) {
            Ok(cfg) => return cfg,
            Err(err) => tracing::warn!("Failed to parse {}: {}", path, err),
        }
    }
    VaderLexiconConfig::default()
}
