use crate::Token;
use crate::config::NgramConfig;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;


#[derive(Clone)]
struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<usize>,
}

impl AliasTable {
    fn new(weights: &[f64]) -> Self {
        let n = weights.len();
        let mut prob = vec![0.0; n];
        let mut alias = vec![0; n];
        let mut scaled: Vec<f64> = weights.iter().map(|&w| w * n as f64).collect();
        let mut small = Vec::new();
        let mut large = Vec::new();
        for (i, &val) in scaled.iter().enumerate() {
            if val < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }
        while !small.is_empty() && !large.is_empty() {
            let small_idx = small.pop().unwrap();
            let large_idx = large.pop().unwrap();
            prob[small_idx] = scaled[small_idx];
            alias[small_idx] = large_idx;
            scaled[large_idx] -= 1.0 - scaled[small_idx];
            if scaled[large_idx] < 1.0 {
                small.push(large_idx);
            } else {
                large.push(large_idx);
            }
        }
        for &idx in &large {
            prob[idx] = 1.0;
            alias[idx] = idx;
        }
        for &idx in &small {
            prob[idx] = 1.0;
            alias[idx] = idx;
        }
        AliasTable { prob, alias }
    }

    fn sample(&self, rng: &mut SmallRng) -> usize {
        let i = rng.random_range(0..self.prob.len());
        let u: f64 = rng.random_range(0.0..1.0);
        if u < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }

    fn pmf(&self) -> Vec<f64> {
        let n = self.prob.len();
        let inv_n = 1.0 / n as f64;
        let mut p = vec![0.0; n];

        for i in 0..n {
            p[i] += self.prob[i] * inv_n;
            let a = self.alias[i];
            p[a] += (1.0 - self.prob[i]) * inv_n;
        }

        let s: f64 = p.iter().sum();
        if (s - 1.0).abs() > 1e-12 {
            for x in &mut p {
                *x /= s;
            }
        }
        p
    }
}

#[derive(Clone)]
pub struct NgramTables {
    pub config: NgramConfig,
    pub vocab_size: usize,
    pub eos_token: Token,
    alias_tables: Vec<AliasTable>,
}

impl NgramTables {
    pub fn new(config: NgramConfig, seed: u64) -> Arc<Self> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let vocab_size = config.vocab_size;
        let eos_token = (config.vocab_size - 1) as Token;

        let num_contexts = config.vocab_size;
        let mut alias_tables = Vec::with_capacity(num_contexts);
        let normal = Normal::new(config.zipf_mu, config.zipf_sigma).unwrap();

        for _ in 0..num_contexts {
            let exponent = normal.sample(&mut rng).max(config.min_exponent);
            let mut weights = Vec::with_capacity(vocab_size);
            let mut sum = 0.0;
            for rank in 1..=vocab_size {
                let weight = 1.0 / (rank as f64).powf(exponent);
                weights.push(weight);
                sum += weight;
            }
            for w in &mut weights {
                *w /= sum;
            }
            alias_tables.push(AliasTable::new(&weights));
        }

        Arc::new(NgramTables {
            config,
            vocab_size,
            eos_token,
            alias_tables,
        })
    }

    pub fn get_context_pmf(&self, context: usize) -> Vec<f64> {
        self.alias_tables[context].pmf()
    }

    pub fn sample_from_context(&self, context: usize, rng: &mut SmallRng) -> usize {
        self.alias_tables[context].sample(rng)
    }
}

pub struct NgramSampler<'a> {
    tables: &'a NgramTables,
    rng: SmallRng,
    orders: Vec<usize>,
    order_selector: AliasTable,
}

impl<'a> NgramSampler<'a> {
    pub fn new(tables: &'a NgramTables, seed: u64) -> Self {
        let rng = SmallRng::seed_from_u64(seed);

        let orders = vec![1, 2, 4, 8, 16, 32, 64, 128];
        
        let mut order_weights = Vec::new();
        let mut sum = 0.0;
        for (i, _) in orders.iter().enumerate() {
            let weight = 1.0 / ((i + 1) as f64).powf(1.5); 
            order_weights.push(weight);
            sum += weight;
        }
        for w in &mut order_weights { *w /= sum; }
        
        let order_selector = AliasTable::new(&order_weights);

        NgramSampler {
            tables,
            rng,
            orders,
            order_selector,
        }
    }


    fn sample_length(&mut self) -> usize {
        self.tables.config.min_length + self.rng.random_range(0..100)
    }

    fn get_mixed_order_context(&mut self, history: &[Token]) -> usize {
        let order_idx = self.order_selector.sample(&mut self.rng);
        let n = self.orders[order_idx];

        let hist_len = history.len();
        let start = if hist_len > n { hist_len - n } else { 0 };
        let context_slice = &history[start..];

        let mut hasher = DefaultHasher::new();
        context_slice.hash(&mut hasher);
        let hash = hasher.finish();

        (hash as usize) % self.tables.vocab_size
    }

    pub fn sample_sentence(&mut self) -> Vec<Token> {
        let length = self.sample_length();
        let mut tokens = Vec::with_capacity(length + 1);
        
        let mut context_idx = self.get_mixed_order_context(&tokens);

        for _ in 0..length {
            loop {
                let token_idx = self.tables.alias_tables[context_idx].sample(&mut self.rng);
                
                if token_idx as Token == self.tables.eos_token {
                    continue;
                }
                
                tokens.push(token_idx as Token);
                break;
            }
            
            context_idx = self.get_mixed_order_context(&tokens);
        }
        
        tokens.push(self.tables.eos_token);
        tokens
    }
}


pub struct NgramLanguage {
    _config: NgramConfig,
    tables: Arc<NgramTables>,
    num_sentences: usize,
    seed: u64,
}

impl NgramLanguage {
    #[inline]
    pub fn new(config: NgramConfig, num_sentences: usize, seed: u64) -> Self {
        let tables = NgramTables::new(config.clone(), seed);
        NgramLanguage {
            _config: config,
            tables,
            num_sentences,
            seed,
        }
    }

    #[inline]
    pub fn generate_test_only(&mut self, test_split: f64) -> Vec<Vec<Token>> {
        let num_test = (self.num_sentences as f64 * test_split) as usize;

        (0..num_test)
            .into_par_iter()
            .map(|i| {
                let seed = self.seed.wrapping_add(i as u64);
                let mut sampler = NgramSampler::new(&self.tables, seed);
                sampler.sample_sentence()
            })
            .collect()
    }
}

impl NgramLanguage {
    #[inline]
    pub fn generate_samples(&mut self, n: usize) -> Vec<Vec<Token>> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let seed = self.seed.wrapping_add(i as u64);
                let mut sampler = NgramSampler::new(&self.tables, seed);
                sampler.sample_sentence()
            })
            .collect()
    }



    pub fn generate_dataset(
        &mut self,
        batch_size: usize,
    ) -> impl Iterator<Item = Vec<u16>> + '_ {
        let num_batches = (self.num_sentences + batch_size - 1) / batch_size;
        let mut current_batch = 0;

        std::iter::from_fn(move || {
            if current_batch >= num_batches {
                return None;
            }

            let sentences_in_this_batch = if current_batch == num_batches - 1 {
                self.num_sentences - (current_batch * batch_size)
            } else {
                batch_size
            };

            let batch: Vec<Vec<Token>> = (0..rayon::current_num_threads())
                .into_par_iter()
                .flat_map(|thread_id| {
                    let sentences_this_thread = sentences_in_this_batch / rayon::current_num_threads()
                        + if thread_id < sentences_in_this_batch % rayon::current_num_threads() {
                            1
                        } else {
                            0
                        };

                    let mut local_sentences = Vec::with_capacity(sentences_this_thread);
                    let batch_offset = current_batch * batch_size;
                    let base_seed = self.seed.wrapping_add(
                        batch_offset as u64 + thread_id as u64 * 1_000_000
                    );

                    let mut sampler = NgramSampler::new(&self.tables, base_seed);

                    for _ in 0..sentences_this_thread {
                        local_sentences.push(sampler.sample_sentence());
                    }

                    local_sentences
                })
                .collect();

            let flattened: Vec<u16> = batch.into_iter().flatten().collect();
            current_batch += 1;
            Some(flattened)
        })
    }
}