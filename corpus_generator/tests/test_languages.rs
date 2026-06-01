use anyhow::Result;
use generator::{
    VocabConfig,
    tokenizer::{TokenizerType, generate_tokenizer_files},
};
use std::path::Path;
use tempfile::tempdir;
use tokenizers::Tokenizer;

#[cfg(test)]
mod combined_tests {
    use super::*;
    use generator::{NgramConfig, NgramLanguage};

    #[test]
    fn generate_and_roundtrip_first_10_sentences() -> Result<()> {
        let vocab_config = VocabConfig {
            vocab_size: 32,
            pct_subj: 0.3,
            pct_obj: 0.3,
            pct_verb: 0.3,
            pct_conn: 0.03,
            pct_intersection: 0.07,
        };
        let config = HierarchicalConfig::from_vocab_config(
            vocab_config.clone(),
            10,
            2,
            2,
            2,
            1.0,
            0.8,
            1,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        let mut lang = HierarchicalLanguage::new(config, 42);
        let (train_set, _test_set) = lang.generate_dataset(true);

        let dir = tempdir()?;
        let out = dir.path().to_str().unwrap();
        let vocab_size = 32;

        let grammar_info = vocab_config.to_grammar_info();
        generate_tokenizer_files(out, vocab_size, TokenizerType::Pcfg(grammar_info))?;

        let tok_path = Path::new(out).join("tokenizer").join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(tok_path.to_str().unwrap()).map_err(|e| anyhow::anyhow!(e))?;

        println!("{}", tokenizer.get_vocab_size(true));
        println!("Decoding first 10 tokenized sequences:");
        for (i, token_ids) in train_set.iter().take(10).enumerate() {
            let ids_u32: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
            let decoded = tokenizer
                .decode(&ids_u32, false)
                .map_err(|e| anyhow::anyhow!(e))?;
            println!(
                "{}: token_ids = {:?}\n   decoded = {}\n",
                i + 1,
                token_ids,
                decoded
            );
        }
        Ok(())
    }

    #[test]
    fn generate_and_roundtrip_first_10_ngram_sentences() -> Result<()> {
        let config = NgramConfig {
            vocab_size: 32,
            order: 2,
            zipf_mu: 2.0,
            zipf_sigma: 0.5,
            length_zipf_exponent: 2.0,
            min_length: 1,
            min_exponent: 0.1,
        };
        let mut lang = NgramLanguage::new(config.clone(), 100, 42);
        let (train_set, _test_set) = lang.generate_dataset(true);

        let dir = tempdir()?;
        let out = dir.path().to_str().unwrap();
        generate_tokenizer_files(out, config.vocab_size, TokenizerType::Ngram)?;

        let tok_path = Path::new(out).join("tokenizer").join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(tok_path.to_str().unwrap()).map_err(|e| anyhow::anyhow!(e))?;

        println!(
            "Ngram tokenizer vocab size: {}",
            tokenizer.get_vocab_size(true)
        );

        println!("Decoding first 10 tokenized ngram sequences:");
        for (i, token_ids) in train_set.iter().take(10).enumerate() {
            let ids_u32: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
            let decoded = tokenizer
                .decode(&ids_u32, false)
                .map_err(|e| anyhow::anyhow!(e))?;
            println!(
                "{}: token_ids = {:?}\n   decoded = {}\n",
                i + 1,
                token_ids,
                decoded
            );
        }

        Ok(())
    }

    #[test]
    fn test_no_test_sentences_in_train_pcfg() -> Result<()> {
        let vocab_config = VocabConfig {
            vocab_size: 50,
            pct_subj: 0.3,
            pct_obj: 0.3,
            pct_verb: 0.3,
            pct_conn: 0.05,
            pct_intersection: 0.05,
        };

        let config = HierarchicalConfig::from_vocab_config(
            vocab_config,
            20,
            5,
            3,
            3,
            1.5,
            0.7,
            1,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        let mut lang = HierarchicalLanguage::new(config, 42);
        let (train_set, test_set) = lang.generate_dataset(true);

        let train_sentences: std::collections::HashSet<Vec<u16>> = train_set.into_iter().collect();

        let mut overlap_count = 0;
        for test_sentence in &test_set {
            if train_sentences.contains(test_sentence) {
                overlap_count += 1;
                println!("Found overlapping sentence: {:?}", test_sentence);
            }
        }

        assert_eq!(
            overlap_count, 0,
            "Found {} test sentences in training data! Train/test sets should be disjoint.",
            overlap_count
        );

        println!(
            "✓ PCFG: Train set size: {}, Test set size: {}, No overlap found",
            train_sentences.len(),
            test_set.len()
        );

        Ok(())
    }

    #[test]
    fn test_no_test_sentences_in_train_ngram() -> Result<()> {
        let config = NgramConfig {
            vocab_size: 50,
            order: 2,
            zipf_mu: 2.0,
            zipf_sigma: 0.5,
            length_zipf_exponent: 2.0,
            min_length: 5,
            min_exponent: 0.1,
        };

        let mut lang = NgramLanguage::new(config, 1000, 42);
        let (train_set, test_set) = lang.generate_dataset(true);

        let train_sentences: std::collections::HashSet<Vec<u16>> = train_set.into_iter().collect();

        let mut overlap_count = 0;
        for test_sentence in &test_set {
            if train_sentences.contains(test_sentence) {
                overlap_count += 1;
                println!("Found overlapping sentence: {:?}", test_sentence);
            }
        }

        assert_eq!(
            overlap_count, 0,
            "Found {} test sentences in training data! Train/test sets should be disjoint.",
            overlap_count
        );

        println!(
            "✓ Ngram: Train set size: {}, Test set size: {}, No overlap found",
            train_sentences.len(),
            test_set.len()
        );

        Ok(())
    }
}
