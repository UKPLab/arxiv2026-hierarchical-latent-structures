#![warn(clippy::disallowed_types)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub type Token = u16;

pub mod arrow_writer;
pub mod config;
pub mod language;
pub mod tokenizer;

pub use config::NgramConfig;
pub use language::NgramLanguage;
pub use tokenizer::TokenizerType;
