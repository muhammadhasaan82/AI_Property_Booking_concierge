pub mod search;
pub mod booking_validator;
pub mod pricing;
pub mod sentiment;
pub mod fraud_check;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Tool trait – plugin-style extensibility
// ---------------------------------------------------------------------------
pub trait Tool: Send + Sync {
    /// Human-readable tool name.
    fn name(&self) -> &'static str;

    /// Return `true` if this tool can handle the given input.
    fn can_handle(&self, input: &Value) -> bool;

    /// Execute the tool and return a JSON result.
    fn execute(&self, input: &Value) -> Value;

    /// Confidence score (0.0–1.0) that this tool is the right match.
    /// Default implementation returns 0.5 if `can_handle` is true.
    fn confidence(&self, input: &Value) -> f64 {
        if self.can_handle(input) { 0.5 } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Tool Registry – dynamic collection of tools
// ---------------------------------------------------------------------------
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        tracing::info!("Registered tool: {}", tool.name());
        self.tools.push(tool);
    }

    /// Select the best-matching tool for the given input.
    pub fn select(&self, input: &Value) -> Option<&dyn Tool> {
        let mut best: Option<(&dyn Tool, f64)> = None;
        for tool in &self.tools {
            let score = tool.confidence(input);
            if score > 0.0 {
                if best.is_none() || score > best.unwrap().1 {
                    best = Some((tool.as_ref(), score));
                }
            }
        }
        best.map(|(t, _)| t)
    }

    /// Execute the best-matching tool automatically.
    pub fn auto_execute(&self, input: &Value) -> Value {
        match self.select(input) {
            Some(tool) => {
                tracing::info!("Auto-selected tool: {}", tool.name());
                let result = tool.execute(input);
                serde_json::json!({
                    "ok": true,
                    "tool_used": tool.name(),
                    "result": result
                })
            }
            None => {
                tracing::warn!("No tool matched input");
                serde_json::json!({
                    "ok": false,
                    "error": "no_matching_tool",
                    "message": "Could not infer intent from the provided data. Please include more context.",
                    "hint": "Include keys like 'location', 'property_id', 'booking_id', 'amount', or 'question'."
                })
            }
        }
    }

    pub fn list_tools(&self) -> Vec<&'static str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
}

/// Build the default registry with all built-in tools.
pub fn build_default_registry() -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(search::PropertySearchTool));
    reg.register(Box::new(booking_validator::BookingValidatorTool));
    reg.register(Box::new(pricing::PricingTool));
    reg.register(Box::new(sentiment::SentimentTool));
    reg.register(Box::new(fraud_check::FraudCheckTool));
    reg
}
