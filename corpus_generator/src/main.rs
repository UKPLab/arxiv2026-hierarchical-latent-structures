use anyhow::Result;
use std::env;
use std::fs;
use std::path::PathBuf;

use generator::arrow_writer::save_u16_as_arrow;
use generator::config::NgramConfig;
use generator::language::NgramLanguage;
use generator::tokenizer::{TokenizerType, generate_tokenizer_files};
use indicatif::{ProgressBar, ProgressStyle};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dataset_output_path>", args[0]);
        eprintln!("Configuration should be provided via stdin as JSON");
        std::process::exit(1);
    }

    let dataset_path = PathBuf::from(&args[1]);

    use std::io::Read;
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let config: serde_json::Value = serde_json::from_str(&input)?;

    let language_type = config["language_type"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing language_type"))?;
    let vocab_size = config["vocab_size"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing vocab_size"))? as usize;
    let context_window = config["context_window"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing context_window"))?
        as usize;

    fs::create_dir_all(&dataset_path)?;

    println!(
        "Generating {} dataset at {}",
        language_type,
        dataset_path.display()
    );
    println!("  Vocab size: {}", vocab_size);
    println!("  Context window: {}", context_window);

    let train_path = dataset_path.join("train.arrow");
    let test_path = dataset_path.join("test.arrow");

    match language_type.to_lowercase().as_str() {
        "pcfg" => {
            generator::language::pcfg::generate_and_save_pcfg_datasets(
                &train_path.to_string_lossy(),
                &test_path.to_string_lossy(),
            );

            println!("Generated PCFG datasets");
        }
        "ngram" => {
            let min_length = config["language_config"]["min_length"]
                .as_u64()
                .unwrap_or(10) as usize;
            let num_sentences = config["language_config"]["num_sentences"]
                .as_u64()
                .unwrap_or(1000000) as usize;
            let zipf_mu = config["language_config"]["zipf_mu"].as_f64().unwrap_or(2.0);
            let zipf_sigma = config["language_config"]["zipf_sigma"]
                .as_f64()
                .unwrap_or(0.5);
            let length_zipf_exponent = config["language_config"]["length_zipf_exponent"]
                .as_f64()
                .unwrap_or(2.0);
            let min_exponent = config["language_config"]["min_exponent"]
                .as_f64()
                .unwrap_or(2.0);

            let ngram_config = NgramConfig {
                vocab_size,
                order: config["language_config"]["order"]
                    .as_u64()
                    .unwrap_or(2) as usize,
                min_length,
                zipf_mu,
                zipf_sigma,
                length_zipf_exponent,
                min_exponent,
            };

            let mut language = NgramLanguage::new(ngram_config.clone(), num_sentences, 0);

            let batch_size = 200_000;

            let progress = ProgressBar::new(num_sentences as u64);
            let style = ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("##-");
            progress.set_style(style);
            progress.set_message("Generating sentences");

            let mut streaming_iter = language.generate_dataset(batch_size);
            let mut sentences_generated = 0;

            let progress_iter = {
                let mut progress_clone = progress.clone();
                std::iter::from_fn(move || {
                    let batch = streaming_iter.next()?;
                    sentences_generated += batch_size;
                    progress_clone.inc(batch_size as u64);
                    Some(batch)
                })
            };

            let flattened_iter = progress_iter.flat_map(|batch| batch.into_iter());

            save_u16_as_arrow(&train_path.to_string_lossy(), flattened_iter)?;

            progress.finish_with_message("Generated all sentences");

            println!("Generated {} train sequences (streaming)", num_sentences);
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported language type: {}",
                language_type
            ));
        }
    }

    println!(
        "Saved datasets to {} and {}",
        train_path.display(),
        test_path.display()
    );

    let tokenizer_type = match language_type.to_lowercase().as_str() {
        "pcfg" => TokenizerType::Pcfg,
        "ngram" => TokenizerType::Ngram,
        _ => TokenizerType::Ngram,
    };

    generate_tokenizer_files(&dataset_path.to_string_lossy(), vocab_size, tokenizer_type)?;
    println!(
        "Generated tokenizer files in {}/tokenizer",
        dataset_path.display()
    );

    println!("Dataset generation completed successfully");
    Ok(())
}
