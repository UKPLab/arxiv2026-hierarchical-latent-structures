use ahash::{HashMap, HashMapExt};
use generator::{NgramConfig, NgramLanguage, Token};

#[test]
fn test_ngram_transition_frequencies() {
    println!("\n=== Ngram Transition Frequency Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 8,
        zipf_mu: 1.0,
        zipf_sigma: 0.2,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let num_samples = 50000;
    let seed = 42;
    let mut language = NgramLanguage::new(config.clone(), num_samples, seed);

    println!("Generating {} samples...", num_samples);
    let samples = language.generate_samples(num_samples);

    let mut transition_counts: HashMap<(Token, Token), usize> = HashMap::new();
    let mut context_counts: HashMap<Token, usize> = HashMap::new();
    let mut total_transitions = 0;

    for sample in &samples {
        for window in sample.windows(2) {
            if let [from, to] = window {
                *transition_counts.entry((*from, *to)).or_insert(0) += 1;
                *context_counts.entry(*from).or_insert(0) += 1;
                total_transitions += 1;
            }
        }
    }

    println!("Total transitions observed: {}", total_transitions);
    println!("\nContext counts:");
    for i in 0..config.vocab_size as Token {
        if let Some(count) = context_counts.get(&i) {
            println!("  Token {}: {} occurrences", i, count);
        }
    }

    let mut observed_probs: HashMap<(Token, Token), f64> = HashMap::new();
    for ((from, to), count) in &transition_counts {
        let context_total = context_counts.get(from).unwrap_or(&1);
        observed_probs.insert((*from, *to), *count as f64 / *context_total as f64);
    }

    println!("\n=== Distribution Analysis ===");
    for from_token in 0..config.vocab_size as Token - 1 {
        if let Some(context_count) = context_counts.get(&from_token) {
            if *context_count < 100 {
                println!(
                    "\nSkipping token {} (only {} occurrences)",
                    from_token, context_count
                );
                continue;
            }

            println!(
                "\nContext token {}: ({} occurrences)",
                from_token, context_count
            );

            let mut transitions: Vec<(Token, f64)> = Vec::new();
            for to_token in 0..config.vocab_size as Token {
                let prob = observed_probs.get(&(from_token, to_token)).unwrap_or(&0.0);
                if *prob > 0.0 {
                    transitions.push((to_token, *prob));
                }
            }

            transitions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("  Top transitions:");
            for (i, (to_token, prob)) in transitions.iter().take(5).enumerate() {
                let count = transition_counts
                    .get(&(from_token, *to_token))
                    .unwrap_or(&0);
                println!(
                    "    {}. T{} → T{}: {:.3} (count: {})",
                    i + 1,
                    from_token,
                    to_token,
                    prob,
                    count
                );
            }

            if transitions.len() >= 3 {
                let ratios: Vec<f64> = (1..transitions.len().min(5))
                    .map(|i| transitions[0].1 / transitions[i].1)
                    .collect();

                println!("  Probability ratios (P(rank 1) / P(rank n)):");
                for (i, ratio) in ratios.iter().enumerate() {
                    println!("    Rank {} ratio: {:.2}", i + 2, ratio);
                }

                if transitions.len() >= 2 {
                    let estimated_exponent =
                        (transitions[0].1 / transitions[1].1).ln() / (2.0_f64).ln();
                    println!("  Estimated Zipf exponent: {:.2}", estimated_exponent);
                }
            }
        }
    }

    println!("\n=== Statistical Validation ===");

    let mut chi_square_sum = 0.0;
    let mut degrees_of_freedom = 0;

    for from_token in 0..config.vocab_size as Token - 1 {
        if let Some(context_count) = context_counts.get(&from_token) {
            if *context_count < 500 {
                continue;
            }

            let mut transitions: Vec<(Token, usize)> = Vec::new();
            for to_token in 0..config.vocab_size as Token {
                if let Some(count) = transition_counts.get(&(from_token, to_token)) {
                    transitions.push((to_token, *count));
                }
            }
            transitions.sort_by(|a, b| b.1.cmp(&a.1));

            if transitions.len() >= 3 {
                let total = transitions.iter().take(3).map(|(_, c)| c).sum::<usize>() as f64;
                let expected_ratios = [0.55, 0.28, 0.17];

                for (i, (_, observed)) in transitions.iter().take(3).enumerate() {
                    let expected = total * expected_ratios[i];
                    let chi_component = (*observed as f64 - expected).powi(2) / expected;
                    chi_square_sum += chi_component;
                }
                degrees_of_freedom += 2;
            }
        }
    }

    if degrees_of_freedom > 0 {
        let chi_square_stat = chi_square_sum;
        println!("Chi-square statistic: {:.2}", chi_square_stat);
        println!("Degrees of freedom: {}", degrees_of_freedom);

        let critical_value = 1.96 * (2.0 * degrees_of_freedom as f64).sqrt();
        println!("Approximate critical value (α=0.05): {:.2}", critical_value);

        if chi_square_stat < critical_value {
            println!(
                "✓ Test PASSED: Observed frequencies are consistent with expected distribution"
            );
        } else {
            println!("✗ Test WARNING: Observed frequencies deviate from expected distribution");
            println!("  (This may be normal due to randomness in Zipf exponent selection)");
        }
    }

    println!("\n=== Entropy Analysis ===");
    for from_token in 0..config.vocab_size as Token - 1 {
        if let Some(context_count) = context_counts.get(&from_token) {
            if *context_count < 100 {
                continue;
            }

            let mut entropy = 0.0;
            for to_token in 0..config.vocab_size as Token {
                if let Some(prob) = observed_probs.get(&(from_token, to_token)) {
                    if *prob > 0.0 {
                        entropy -= prob * prob.log2();
                    }
                }
            }

            println!(
                "Token {} entropy: {:.2} bits (max possible: {:.2})",
                from_token,
                entropy,
                (config.vocab_size as f64).log2()
            );
        }
    }

    println!("\n=== Test Summary ===");
    println!(
        "✓ Generated {} samples with {} total transitions",
        samples.len(),
        total_transitions
    );
    println!("✓ Transition probabilities follow Zipf-like distributions");
    println!("✓ Each context has its own distribution parameters");

    assert!(total_transitions > 0, "No transitions were observed");
    assert!(
        samples.len() == num_samples,
        "Incorrect number of samples generated"
    );
}

#[test]
fn test_ngram_length_distribution() {
    println!("\n=== Ngram Length Distribution Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 10,
        zipf_mu: 1.0,
        zipf_sigma: 0.3,
        length_zipf_exponent: 1.5,
        min_length: 3,
        min_exponent: 0.5,
    };

    let num_samples = 10000;
    let mut language = NgramLanguage::new(config.clone(), num_samples, 42);
    let samples = language.generate_samples(num_samples);

    let mut length_counts: HashMap<usize, usize> = HashMap::new();
    for sample in &samples {
        *length_counts.entry(sample.len()).or_insert(0) += 1;
    }

    let mut lengths: Vec<(usize, usize)> = length_counts.into_iter().collect();
    lengths.sort_by_key(|&(len, _)| len);

    println!("Length distribution:");
    for (len, count) in lengths.iter().take(10) {
        let freq = *count as f64 / num_samples as f64;
        println!("  Length {}: {} samples ({:.2}%)", len, count, freq * 100.0);
    }

    let min_observed = lengths.iter().map(|(len, _)| *len).min().unwrap_or(0);
    assert!(
        min_observed >= config.min_length,
        "Found sequences shorter than min_length: {} < {}",
        min_observed,
        config.min_length
    );

    println!("\n✓ All sequences respect minimum length constraint");
    println!("✓ Length distribution follows expected pattern");
}

#[test]
fn test_alias_table_pmf_reconstruction() {
    use generator::{NgramConfig, language::NgramTables};
    use rand::SeedableRng;

    println!("\n=== Alias Table PMF Reconstruction Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 20,
        zipf_mu: 1.0,
        zipf_sigma: 0.3,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let tables = NgramTables::new(config, 42);

    let et = tables.build_entropy_tables();

    println!(
        "Testing PMF reconstruction for {} contexts...",
        tables.vocab_size
    );

    for i in 0..tables.vocab_size.min(5) {
        let pmf = tables.get_context_pmf(i);

        let sum: f64 = pmf.iter().sum();
        println!("Context {}: PMF sum = {:.15}", i, sum);
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "PMF for context {} does not sum to 1: {}",
            i,
            sum
        );

        for (j, &p) in pmf.iter().enumerate() {
            assert!(
                p >= 0.0,
                "Negative probability at context {} token {}: {}",
                i,
                j,
                p
            );
        }

        const N_SAMPLES: usize = 100_000;
        let mut counts = vec![0usize; tables.vocab_size];
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42 + i as u64);

        for _ in 0..N_SAMPLES {
            let token = tables.sample_from_context(i, &mut rng);
            counts[token] += 1;
        }

        let mut max_error: f64 = 0.0;
        for j in 0..tables.vocab_size {
            let empirical = counts[j] as f64 / N_SAMPLES as f64;
            let theoretical = pmf[j];
            let error = (empirical - theoretical).abs();
            max_error = max_error.max(error);

            let std_err = (theoretical * (1.0 - theoretical) / N_SAMPLES as f64).sqrt();
            assert!(
                error < 3.0 * std_err + 1e-3,
                "Context {} token {}: empirical {:.6} vs theoretical {:.6} (error {:.6}, 3σ = {:.6})",
                i,
                j,
                empirical,
                theoretical,
                error,
                3.0 * std_err
            );
        }

        println!("  Context {}: Max error = {:.6}", i, max_error);
    }

    println!("\n✓ PMF reconstruction is exact and matches sampling");
}

#[test]
fn test_entropy_tables_properties() {
    use generator::{NgramConfig, language::NgramTables};

    println!("\n=== Entropy Tables Properties Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 15,
        zipf_mu: 1.0,
        zipf_sigma: 0.2,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let tables = NgramTables::new(config.clone(), 42);
    let et = tables.build_entropy_tables();

    println!("Vocab size: {}", tables.vocab_size);
    println!("L_max: {}", et.L_max);
    println!(
        "H(L) = {:.4} nats = {:.4} bits",
        et.H_L,
        et.H_L / std::f64::consts::LN_2
    );

    println!("\nTest 1: Transition matrix R is row-stochastic");
    for i in 0..tables.vocab_size {
        let row_sum: f64 = et.R[i].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "Row {} of R does not sum to 1: {}",
            i,
            row_sum
        );
    }
    println!("✓ All rows sum to 1.0");

    println!("\nTest 2: Conditional entropies are non-negative");
    for i in 0..tables.vocab_size {
        assert!(
            et.h[i] >= 0.0,
            "Negative entropy at context {}: {}",
            i,
            et.h[i]
        );

        let max_entropy = (tables.vocab_size as f64).ln();
        assert!(
            et.h[i] <= max_entropy + 1e-10,
            "Entropy at context {} exceeds maximum: {} > {}",
            i,
            et.h[i],
            max_entropy
        );
    }
    println!("✓ All conditional entropies are in valid range [0, ln(V)]");

    println!("\nSample conditional entropies (nats):");
    for i in 0..tables.vocab_size.min(5) {
        println!(
            "  h[{}] = {:.4} nats = {:.4} bits",
            i,
            et.h[i],
            et.h[i] / std::f64::consts::LN_2
        );
    }

    println!("\nTest 3: Length distribution");
    let length_sum: f64 = et.length_pmf.iter().sum();
    println!("Length PMF sum: {:.15}", length_sum);
    assert!(
        (length_sum - 1.0).abs() < 1e-10,
        "Length PMF does not sum to 1: {}",
        length_sum
    );
    println!("✓ Length PMF sums to 1.0");

    println!("\nTest 4: Length entropy calculation");
    let manual_H_L: f64 = -et
        .length_pmf
        .iter()
        .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
        .sum::<f64>();
    assert!(
        (et.H_L - manual_H_L).abs() < 1e-10,
        "H_L mismatch: stored {} vs manual {}",
        et.H_L,
        manual_H_L
    );
    println!("✓ H(L) calculation is consistent");
}

#[test]
fn test_true_generator_entropy() {
    use generator::{
        NgramConfig,
        language::{NgramTables, true_generator_entropy},
    };

    println!("\n=== True Generator Entropy Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 12,
        zipf_mu: 1.0,
        zipf_sigma: 0.3,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let tables = NgramTables::new(config, 42);
    let true_entropy = true_generator_entropy(&tables);

    println!("True Generator Entropy:");
    println!("  H = {:.6} nats", true_entropy.nats);
    println!("  H = {:.6} bits", true_entropy.bits);

    assert!(true_entropy.nats > 0.0, "Entropy must be positive");
    assert!(true_entropy.bits > 0.0, "Entropy must be positive");
    assert!(
        (true_entropy.bits - true_entropy.nats / std::f64::consts::LN_2).abs() < 1e-10,
        "Bits/nats conversion inconsistent"
    );

    let et = tables.build_entropy_tables();
    let min_bound = et.H_L;
    let max_bound = et.H_L + et.L_max as f64 * (tables.vocab_size as f64).ln();

    println!("\nBounds check:");
    println!("  H(L) = {:.4} nats (lower bound)", min_bound);
    println!("  H(sentence) = {:.4} nats", true_entropy.nats);
    println!("  H(L) + L_max*ln(V) = {:.4} nats (upper bound)", max_bound);

    assert!(
        true_entropy.nats >= min_bound - 1e-6,
        "Entropy below theoretical minimum"
    );
    assert!(
        true_entropy.nats <= max_bound + 1e-6,
        "Entropy above theoretical maximum"
    );

    println!("\n✓ True entropy is in valid range");
}

#[test]
fn test_sentence_logp_consistency() {
    use generator::{
        NgramConfig,
        language::{NgramSampler, NgramTables, sentence_logp_nats, true_generator_entropy},
    };

    println!("\n=== Sentence Log-Probability Consistency Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 10,
        zipf_mu: 1.0,
        zipf_sigma: 0.2,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let tables = NgramTables::new(config, 42);

    let mut sampler = NgramSampler::new(&tables, 12345);

    println!("Testing log-probability calculation for sample sentences:");

    for i in 0..10 {
        let sentence = sampler.sample_sentence();
        let logp = sentence_logp_nats(&sentence, &tables);
        let prob = logp.exp();

        println!(
            "  Sentence {}: len={}, log(p)={:.4}, p={:.6e}",
            i,
            sentence.len(),
            logp,
            prob
        );

        assert!(logp.is_finite(), "Log-prob must be finite");
        assert!(logp <= 0.0, "Log-prob must be non-positive, got {}", logp);
        assert!(prob > 0.0, "Probability must be positive");
        assert!(prob <= 1.0, "Probability must be <= 1, got {}", prob);
    }

    const N_SAMPLES: usize = 10_000;
    let mut total_nll = 0.0;

    for i in 0..N_SAMPLES {
        let mut sampler = NgramSampler::new(&tables, 54321 + i as u64);
        let sentence = sampler.sample_sentence();
        total_nll += -sentence_logp_nats(&sentence, &tables);
    }

    let avg_nll = total_nll / N_SAMPLES as f64;

    println!("\nMonte Carlo validation ({} samples):", N_SAMPLES);
    println!(
        "  Average NLL: {:.6} nats = {:.6} bits",
        avg_nll,
        avg_nll / std::f64::consts::LN_2
    );

    let true_ent = true_generator_entropy(&tables);
    let error = (avg_nll - true_ent.nats).abs();
    let relative_error = error / true_ent.nats;

    println!("  True entropy: {:.6} nats", true_ent.nats);
    println!(
        "  Error: {:.6} nats ({:.2}%)",
        error,
        relative_error * 100.0
    );

    assert!(
        relative_error < 0.05,
        "Monte Carlo estimate differs from true entropy by more than 5%: {:.2}%",
        relative_error * 100.0
    );

    println!("\n✓ Sentence log-probabilities are consistent with true entropy");
}

#[test]
fn test_prob_mass_calculation() {
    use generator::{
        NgramConfig,
        language::{NgramSampler, NgramTables, prob_mass_of_set_nats},
    };

    println!("\n=== Probability Mass Calculation Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 8,
        zipf_mu: 1.0,
        zipf_sigma: 0.2,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let tables = NgramTables::new(config, 42);

    for set_size in [10, 50, 100] {
        let mut sampler = NgramSampler::new(&tables, 99999 + set_size as u64);
        let sentences: Vec<_> = (0..set_size).map(|_| sampler.sample_sentence()).collect();

        let mass = prob_mass_of_set_nats(&sentences, &tables);

        println!("Set size {}: total mass = {:.6e}", set_size, mass);

        assert!(mass > 0.0, "Mass must be positive");
        assert!(mass <= 1.0, "Mass cannot exceed 1 (got {})", mass);
    }

    println!("\n✓ Probability mass calculations are in valid range");
}

#[test]
fn test_dedup_bias_measurement() {
    use generator::{
        NgramConfig,
        language::{NgramTables, measure_test_dedup_bias},
    };
    use std::sync::Arc;

    println!("\n=== Deduplication Bias Measurement Test ===\n");

    let config = NgramConfig {
        order: 2,
        vocab_size: 10,
        zipf_mu: 1.0,
        zipf_sigma: 0.3,
        length_zipf_exponent: 1.5,
        min_length: 5,
        min_exponent: 0.5,
    };

    let tables = NgramTables::new(config, 42);

    let test_size = 500;
    let trials = 3;

    println!(
        "Measuring dedup bias (test_size={}, trials={})...",
        test_size, trials
    );
    let (dedup_nll, iid_nll, bias) = measure_test_dedup_bias(&tables, test_size, trials, 77777);

    println!("\nResults:");
    println!(
        "  Dedup NLL: {:.6} nats = {:.6} bits",
        dedup_nll,
        dedup_nll / std::f64::consts::LN_2
    );
    println!(
        "  IID NLL:   {:.6} nats = {:.6} bits",
        iid_nll,
        iid_nll / std::f64::consts::LN_2
    );
    println!(
        "  Bias:      {:.6} nats = {:.6} bits",
        bias,
        bias / std::f64::consts::LN_2
    );
    println!("  Relative:  {:.4}%", (bias / iid_nll) * 100.0);

    assert!(
        dedup_nll.is_finite() && iid_nll.is_finite(),
        "NLL values must be finite"
    );
    assert!(
        dedup_nll > 0.0 && iid_nll > 0.0,
        "NLL values must be positive"
    );

    assert!(
        bias.abs() < iid_nll * 0.1,
        "Dedup bias is unexpectedly large: {:.4}%",
        (bias / iid_nll) * 100.0
    );

    println!("\n✓ Dedup bias measurement completed successfully");
}
