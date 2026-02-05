use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramConfig {
    pub vocab_size: usize,
    pub order: usize,
    pub zipf_mu: f64,
    pub zipf_sigma: f64,
    pub length_zipf_exponent: f64,
    pub min_length: usize,
    pub min_exponent: f64,
}

impl NgramConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 || self.vocab_size > 10_000 {
            return Err(format!(
                "vocab_size must be between 1 and 10,000, got {}",
                self.vocab_size
            ));
        }

        if self.order < 2 {
            return Err(format!(
                "order must be at least 2 for an n-gram model, got {}",
                self.order
            ));
        }

        if self.zipf_sigma <= 0.0 {
            return Err(format!(
                "zipf_sigma must be greater than 0, got {}",
                self.zipf_sigma
            ));
        }

        if self.length_zipf_exponent <= 0.0 {
            return Err(format!(
                "length_zipf_exponent must be greater than 0, got {}",
                self.length_zipf_exponent
            ));
        }

        if self.min_length == 0 {
            return Err("min_length must be greater than 0".to_string());
        }

        Ok(())
    }
}
