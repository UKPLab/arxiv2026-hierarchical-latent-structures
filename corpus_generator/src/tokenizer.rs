use ahash::{HashMap, HashMapExt};
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use std::fs;

#[derive(Serialize)]
pub struct TokenizerConfig {
    pub added_tokens_decoder: HashMap<String, AddedToken>,
    pub clean_up_tokenization_spaces: bool,
    pub eos_token: String,
    pub extra_special_tokens: HashMap<String, Value>,
    pub model_max_length: u64,
    pub pad_token: String,
    pub tokenizer_class: String,
    pub unk_token: String,
}

#[derive(Serialize)]
pub struct AddedToken {
    pub content: String,
    pub lstrip: bool,
    pub normalized: bool,
    pub rstrip: bool,
    pub single_word: bool,
    pub special: bool,
}

#[derive(Serialize)]
pub struct SpecialTokensMap {
    pub eos_token: String,
    pub pad_token: String,
    pub unk_token: String,
}

#[derive(Serialize)]
pub struct Tokenizer {
    pub version: String,
    pub truncation: Option<Value>,
    pub padding: Option<Value>,
    pub added_tokens: Vec<TokenizerAddedToken>,
    pub normalizer: Option<Value>,
    pub pre_tokenizer: PreTokenizer,
    pub post_processor: Option<Value>,
    pub decoder: Option<Value>,
    pub model: TokenizerModel,
}

#[derive(Serialize)]
pub struct TokenizerAddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Serialize)]
pub struct PreTokenizer {
    #[serde(rename = "type")]
    pub tokenizer_type: String,
}

#[derive(Serialize)]
pub struct TokenizerModel {
    #[serde(rename = "type")]
    pub model_type: String,
    pub vocab: HashMap<String, u32>,
    pub unk_token: String,
}

pub enum TokenizerType {
    Ngram,
    Pcfg,
    PcfgRec,
}

pub fn generate_tokenizer_files(
    output_dir: &str,
    vocab_size: usize,
    tokenizer_type: TokenizerType,
) -> Result<()> {
    let tokenizer_dir = format!("{output_dir}/tokenizer");
    fs::create_dir_all(&tokenizer_dir)?;

    let (num_regular_tokens, eos_token_id, unk_token_id, pad_token_id) = match tokenizer_type {
        TokenizerType::Pcfg | TokenizerType::PcfgRec => {
            (
                vocab_size,
                vocab_size as u32,
                (vocab_size + 1) as u32,
                (vocab_size + 2) as u32,
            )
        }
        TokenizerType::Ngram => {
            (
                vocab_size - 1,
                (vocab_size - 1) as u32,
                vocab_size as u32,
                (vocab_size + 1) as u32,
            )
        }
    };

    let mut added_tokens_decoder = HashMap::new();
    added_tokens_decoder.insert(
        eos_token_id.to_string(),
        AddedToken {
            content: "[EOS]".to_string(),
            lstrip: false,
            normalized: false,
            rstrip: false,
            single_word: false,
            special: true,
        },
    );
    added_tokens_decoder.insert(
        unk_token_id.to_string(),
        AddedToken {
            content: "[UNK]".to_string(),
            lstrip: false,
            normalized: false,
            rstrip: false,
            single_word: false,
            special: true,
        },
    );
    added_tokens_decoder.insert(
        pad_token_id.to_string(),
        AddedToken {
            content: "[PAD]".to_string(),
            lstrip: false,
            normalized: false,
            rstrip: false,
            single_word: false,
            special: true,
        },
    );

    let tokenizer_config = TokenizerConfig {
        added_tokens_decoder,
        clean_up_tokenization_spaces: false,
        eos_token: "[EOS]".to_string(),
        extra_special_tokens: HashMap::new(),
        model_max_length: u64::MAX,
        pad_token: "[PAD]".to_string(),
        tokenizer_class: "PreTrainedTokenizerFast".to_string(),
        unk_token: "[UNK]".to_string(),
    };

    let config_path = format!("{tokenizer_dir}/tokenizer_config.json");
    let config_json = serde_json::to_string_pretty(&tokenizer_config)?;
    fs::write(&config_path, config_json)?;

    let special_tokens_map = SpecialTokensMap {
        eos_token: "[EOS]".to_string(),
        pad_token: "[PAD]".to_string(),
        unk_token: "[UNK]".to_string(),
    };

    let special_path = format!("{tokenizer_dir}/special_tokens_map.json");
    let special_json = serde_json::to_string_pretty(&special_tokens_map)?;
    fs::write(&special_path, special_json)?;

    let mut vocab = HashMap::new();

    match tokenizer_type {
        TokenizerType::Ngram => {
            for i in 0..num_regular_tokens {
                vocab.insert(format!("w{}", i + 1), i as u32);
            }
        }
        TokenizerType::Pcfg => {
            for i in 0..num_regular_tokens {
                vocab.insert(format!("t{}", i), i as u32);
            }
        }
        TokenizerType::PcfgRec => {
            for i in 0..num_regular_tokens {
                vocab.insert(format!("t{}", i), i as u32);
            }
        }
    }

    vocab.insert("[EOS]".to_string(), eos_token_id);
    vocab.insert("[UNK]".to_string(), unk_token_id);
    vocab.insert("[PAD]".to_string(), pad_token_id);

    let added_tokens = vec![
        TokenizerAddedToken {
            id: eos_token_id,
            content: "[EOS]".to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        },
        TokenizerAddedToken {
            id: unk_token_id,
            content: "[UNK]".to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        },
        TokenizerAddedToken {
            id: pad_token_id,
            content: "[PAD]".to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        },
    ];

    let tokenizer = Tokenizer {
        version: "1.0".to_string(),
        truncation: None,
        padding: None,
        added_tokens,
        normalizer: None,
        pre_tokenizer: PreTokenizer {
            tokenizer_type: "WhitespaceSplit".to_string(),
        },
        post_processor: None,
        decoder: None,
        model: TokenizerModel {
            model_type: "WordLevel".to_string(),
            vocab,
            unk_token: "[UNK]".to_string(),
        },
    };

    let tokenizer_path = format!("{tokenizer_dir}/tokenizer.json");
    let tokenizer_json = serde_json::to_string_pretty(&tokenizer)?;
    fs::write(&tokenizer_path, tokenizer_json)?;

    Ok(())
}
