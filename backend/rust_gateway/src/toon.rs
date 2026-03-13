// rust_gateway/src/toon.rs
//! TOON (Token-Optimized Object Notation) serializer/deserializer.
//!
//! Provides lossless round-trip conversion between `serde_json::Value`
//! and TOON text format.  Handles edge cases: strings with colons,
//! newlines, booleans/null keywords, and numeric strings.

use serde_json::{Map, Value};

// ────────────────────── Encoder ──────────────────────

/// Encode a `serde_json::Value` to TOON text.
pub fn encode(value: &Value) -> String {
    match value {
        Value::Object(map) => encode_object(map, 0),
        _ => encode_value(value, 0),
    }
}

fn needs_quoting(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    // Contains structural characters
    if s.contains(':') || s.contains('\n') || s.contains('\r') {
        return true;
    }
    // Looks like a keyword
    let lower = s.trim().to_lowercase();
    if lower == "true" || lower == "false" || lower == "null" || lower == "none" {
        return true;
    }
    // Looks like a number
    if s.trim().parse::<f64>().is_ok() {
        return true;
    }
    false
}

fn quote_string(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("\"{}\"", escaped)
}

fn encode_value(value: &Value, indent: usize) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => format!("{}", n),
        Value::String(s) => {
            if needs_quoting(s) {
                quote_string(s)
            } else {
                s.clone()
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                return "[]".to_string();
            }
            let prefix = "  ".repeat(indent);
            let child_prefix = "  ".repeat(indent + 1);
            let mut lines = vec!["[]".to_string()];
            for item in arr {
                match item {
                    Value::Object(map) => {
                        lines.push(format!("{child_prefix}-"));
                        lines.push(encode_object(map, indent + 2));
                    }
                    _ => {
                        let encoded = encode_value(item, indent + 1);
                        lines.push(format!("{child_prefix}- {encoded}"));
                    }
                }
            }
            let _ = prefix; // used for context
            lines.join("\n")
        }
        Value::Object(map) => encode_object(map, indent),
    }
}

fn encode_object(map: &Map<String, Value>, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    let mut lines: Vec<String> = Vec::new();

    for (key, val) in map {
        let safe_key = if key.contains(':') {
            key.replace(':', "\\:")
        } else {
            key.clone()
        };

        match val {
            Value::Object(child) => {
                if child.is_empty() {
                    lines.push(format!("{prefix}{safe_key}: {{}}"));
                } else {
                    lines.push(format!("{prefix}{safe_key}:"));
                    lines.push(encode_object(child, indent + 1));
                }
            }
            Value::Array(_) => {
                let encoded = encode_value(val, indent);
                if encoded.contains('\n') {
                    let parts: Vec<&str> = encoded.splitn(2, '\n').collect();
                    lines.push(format!("{prefix}{safe_key}: {}", parts[0]));
                    if parts.len() > 1 {
                        lines.push(parts[1].to_string());
                    }
                } else {
                    lines.push(format!("{prefix}{safe_key}: {encoded}"));
                }
            }
            _ => {
                let encoded = encode_value(val, indent);
                lines.push(format!("{prefix}{safe_key}: {encoded}"));
            }
        }
    }

    lines.join("\n")
}

// ────────────────────── Decoder ──────────────────────

struct ToonDecoder {
    lines: Vec<String>,
    pos: usize,
}

impl ToonDecoder {
    fn new(text: &str) -> Self {
        Self {
            lines: text.lines().map(String::from).collect(),
            pos: 0,
        }
    }

    fn current_indent(&self, line: &str) -> usize {
        let stripped = line.trim_start_matches(' ');
        (line.len() - stripped.len()) / 2
    }

    fn skip_blank(&mut self) {
        while self.pos < self.lines.len() && self.lines[self.pos].trim().is_empty() {
            self.pos += 1;
        }
    }

    fn parse(&mut self) -> Value {
        self.skip_blank();
        if self.pos >= self.lines.len() {
            return Value::Object(Map::new());
        }
        let line = self.lines[self.pos].trim().to_string();
        if line.starts_with("- ") || line == "-" {
            let indent = self.current_indent(&self.lines[self.pos]);
            return Value::Array(self.parse_array_items(indent));
        }
        Value::Object(self.parse_object(0))
    }

    fn parse_value_str(&self, raw: &str) -> Value {
        let raw = raw.trim();
        if raw.is_empty() {
            return Value::String(String::new());
        }
        match raw {
            "null" => return Value::Null,
            "true" => return Value::Bool(true),
            "false" => return Value::Bool(false),
            "[]" => return Value::Array(vec![]),
            "{}" => return Value::Object(Map::new()),
            _ => {}
        }

        // Quoted string
        if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
            let inner = &raw[1..raw.len() - 1];
            let unescaped = inner
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\");
            return Value::String(unescaped);
        }

        // Try number
        if let Ok(n) = raw.parse::<i64>() {
            return Value::Number(n.into());
        }
        if let Ok(n) = raw.parse::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(n) {
                return Value::Number(num);
            }
        }

        // Bare string
        Value::String(raw.to_string())
    }

    fn find_key_colon(line: &str) -> Option<usize> {
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b':' {
                i += 2;
                continue;
            }
            if bytes[i] == b':' {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn parse_object(&mut self, expected_indent: usize) -> Map<String, Value> {
        let mut result = Map::new();

        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].clone();
            if line.trim().is_empty() {
                self.pos += 1;
                continue;
            }
            let indent = self.current_indent(&line);
            if indent < expected_indent {
                break;
            }
            if indent > expected_indent {
                break;
            }
            let stripped = line.trim();
            if stripped.starts_with("- ") || stripped == "-" {
                break;
            }

            if let Some(colon_idx) = Self::find_key_colon(stripped) {
                let key = stripped[..colon_idx].replace("\\:", ":");
                let rest = stripped[colon_idx + 1..].trim();
                self.pos += 1;

                let value = if rest.is_empty() {
                    // Nested object on next lines
                    Value::Object(self.parse_object(expected_indent + 1))
                } else if rest == "[]" {
                    // Array header — items follow
                    Value::Array(self.parse_array_items(expected_indent + 1))
                } else {
                    self.parse_value_str(rest)
                };

                result.insert(key, value);
            } else {
                self.pos += 1;
            }
        }

        result
    }

    fn parse_array_items(&mut self, expected_indent: usize) -> Vec<Value> {
        let mut items = Vec::new();

        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].clone();
            if line.trim().is_empty() {
                self.pos += 1;
                continue;
            }
            let indent = self.current_indent(&line);
            if indent < expected_indent {
                break;
            }
            let stripped = line.trim();
            if !stripped.starts_with('-') {
                break;
            }

            let rest = if stripped.len() > 1 {
                stripped[1..].trim()
            } else {
                ""
            };
            self.pos += 1;

            if rest.is_empty() {
                // Multi-line dict item
                items.push(Value::Object(self.parse_object(expected_indent + 1)));
            } else {
                items.push(self.parse_value_str(rest));
            }
        }

        items
    }
}

/// Parse TOON text back to `serde_json::Value`.
pub fn decode(input: &str) -> Result<Value, String> {
    if input.trim().is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    let mut decoder = ToonDecoder::new(input);
    Ok(decoder.parse())
}

/// TOON content type constant.
pub const CONTENT_TYPE: &str = "application/toon";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_encode_simple_object() {
        let val = json!({"name": "John", "age": 30, "active": true});
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["name"], "John");
        assert_eq!(decoded["age"], 30);
        assert_eq!(decoded["active"], true);
    }

    #[test]
    fn test_encode_nested_object() {
        let val = json!({
            "user": {
                "name": "Jane",
                "address": {
                    "city": "NYC"
                }
            }
        });
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["user"]["name"], "Jane");
        assert_eq!(decoded["user"]["address"]["city"], "NYC");
    }

    #[test]
    fn test_encode_array() {
        let val = json!({"items": [1, 2, 3]});
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["items"].as_array().unwrap().len(), 3);
        assert_eq!(decoded["items"][0], 1);
    }

    #[test]
    fn test_string_with_colon() {
        let val = json!({"time": "12:30:00", "normal": "hello"});
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["time"], "12:30:00");
        assert_eq!(decoded["normal"], "hello");
    }

    #[test]
    fn test_string_with_newline() {
        let val = json!({"message": "line1\nline2\nline3"});
        let toon = encode(&val);
        assert!(toon.contains("\\n"));
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["message"], "line1\nline2\nline3");
    }

    #[test]
    fn test_null_and_bool() {
        let val = json!({"a": null, "b": true, "c": false});
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert!(decoded["a"].is_null());
        assert_eq!(decoded["b"], true);
        assert_eq!(decoded["c"], false);
    }

    #[test]
    fn test_empty_object_and_array() {
        let val = json!({"empty_obj": {}, "empty_arr": []});
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert!(decoded["empty_obj"].as_object().unwrap().is_empty());
        assert!(decoded["empty_arr"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_numeric_string_preserved() {
        let val = json!({"phone": "12345", "count": 42});
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        // The phone was a string in JSON, should still be a string
        assert_eq!(decoded["phone"].as_str().unwrap(), "12345");
        // Count was a number, should still be a number
        assert_eq!(decoded["count"], 42);
    }

    #[test]
    fn test_roundtrip_complex() {
        let val = json!({
            "ok": true,
            "service": "rust_gateway",
            "result": {
                "count": 2,
                "items": [
                    {"id": "p1", "title": "Beach House", "price": 150.0},
                    {"id": "p2", "title": "City Loft", "price": 250.0}
                ]
            },
            "message": "Results found: 2 properties\nFiltered by budget"
        });
        let toon = encode(&val);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["ok"], true);
        assert_eq!(decoded["result"]["count"], 2);
        assert_eq!(decoded["result"]["items"][0]["id"], "p1");
        assert!(decoded["message"].as_str().unwrap().contains('\n'));
    }

    #[test]
    fn test_key_with_colon() {
        let val = json!({"time:stamp": "2024-01-01"});
        let toon = encode(&val);
        assert!(toon.contains("time\\:stamp"));
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded["time:stamp"], "2024-01-01");
    }
}
