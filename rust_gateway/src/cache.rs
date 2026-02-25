use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde_json::Value;

/// A single cache entry with TTL.
struct CacheEntry {
    value: Value,
    created_at: Instant,
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Thread-safe in-memory LRU-ish cache with TTL.
pub struct Cache {
    store: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_entries: usize,
}

impl Cache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
        }
    }

    /// Get a value from the cache. Returns None if missing or expired.
    pub fn get(&self, key: &str) -> Option<Value> {
        let store = self.store.read().ok()?;
        if let Some(entry) = store.get(key) {
            if !entry.is_expired() {
                tracing::debug!(key = key, "Cache HIT");
                return Some(entry.value.clone());
            }
            tracing::debug!(key = key, "Cache EXPIRED");
        }
        None
    }

    /// Set a value in the cache with a specific TTL.
    pub fn set(&self, key: String, value: Value, ttl: Duration) {
        let mut store = match self.store.write() {
            Ok(s) => s,
            Err(_) => return,
        };

        // Evict expired entries if we're at capacity
        if store.len() >= self.max_entries {
            let expired_keys: Vec<String> = store
                .iter()
                .filter(|(_, v)| v.is_expired())
                .map(|(k, _)| k.clone())
                .collect();
            for k in expired_keys {
                store.remove(&k);
            }
            // If still full, remove oldest entry
            if store.len() >= self.max_entries {
                if let Some(oldest) = store
                    .iter()
                    .min_by_key(|(_, v)| v.created_at)
                    .map(|(k, _)| k.clone())
                {
                    store.remove(&oldest);
                }
            }
        }

        tracing::debug!(key = %key, ttl_secs = ttl.as_secs(), "Cache SET");
        store.insert(key, CacheEntry {
            value,
            created_at: Instant::now(),
            ttl,
        });
    }

    /// Remove all expired entries.
    pub fn cleanup(&self) {
        if let Ok(mut store) = self.store.write() {
            let expired: Vec<String> = store
                .iter()
                .filter(|(_, v)| v.is_expired())
                .map(|(k, _)| k.clone())
                .collect();
            let count = expired.len();
            for k in expired {
                store.remove(&k);
            }
            if count > 0 {
                tracing::info!(evicted = count, "Cache cleanup");
            }
        }
    }

    /// Get cache stats.
    pub fn stats(&self) -> Value {
        let store = self.store.read().unwrap();
        let total = store.len();
        let expired = store.values().filter(|v| v.is_expired()).count();
        serde_json::json!({
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "max_entries": self.max_entries
        })
    }
}

/// Default TTLs for different cache categories.
pub mod ttl {
    use std::time::Duration;

    pub const PROPERTY_SEARCH: Duration = Duration::from_secs(300);   // 5 minutes
    pub const FAQ_ANSWER: Duration = Duration::from_secs(900);        // 15 minutes
    pub const SESSION_STATE: Duration = Duration::from_secs(1800);    // 30 minutes
    pub const PRICING: Duration = Duration::from_secs(60);            // 1 minute
}

/// Generate a cache key from request data.
pub fn cache_key(prefix: &str, data: &Value) -> String {
    // Simple hash: sort keys and serialize
    let canonical = if let Some(obj) = data.as_object() {
        let mut sorted: Vec<_> = obj.iter().collect();
        sorted.sort_by_key(|(k, _)| k.clone());
        let sorted_map: serde_json::Map<String, Value> = sorted.into_iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        serde_json::to_string(&Value::Object(sorted_map)).unwrap_or_default()
    } else {
        serde_json::to_string(data).unwrap_or_default()
    };

    // Simple hash (not crypto, just for cache dedup)
    let hash: u64 = canonical.bytes().enumerate().fold(0u64, |acc, (i, b)| {
        acc.wrapping_add((b as u64).wrapping_mul(31u64.wrapping_pow(i as u32)))
    });

    format!("{}:{:016x}", prefix, hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_cache_set_get() {
        let cache = Cache::new(100);
        cache.set("key1".to_string(), json!({"result": 42}), Duration::from_secs(60));
        let val = cache.get("key1");
        assert!(val.is_some());
        assert_eq!(val.unwrap()["result"], 42);
    }

    #[test]
    fn test_cache_miss() {
        let cache = Cache::new(100);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_expired() {
        let cache = Cache::new(100);
        cache.set("key1".to_string(), json!({"result": 1}), Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_cache_key_deterministic() {
        let data = json!({"location": "NYC", "budget": 200});
        let k1 = cache_key("search", &data);
        let k2 = cache_key("search", &data);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_eviction() {
        let cache = Cache::new(2);
        cache.set("k1".to_string(), json!(1), Duration::from_secs(60));
        cache.set("k2".to_string(), json!(2), Duration::from_secs(60));
        cache.set("k3".to_string(), json!(3), Duration::from_secs(60));
        // One entry should have been evicted
        let stats = cache.stats();
        assert!(stats["total_entries"].as_u64().unwrap() <= 2);
    }
}
