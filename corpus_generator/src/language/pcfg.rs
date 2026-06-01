use ahash::AHashSet;
use std::{ops::Range, sync::Arc, time::Instant};

use rand::distr::{Bernoulli, Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_distr::Zipf;
use rayon::prelude::*;

use crate::arrow_writer::save_u16_as_arrow;
const AMOUNT_TRAINING_DOCS: u32 = 60_500_000;
const AMOUNT_TEST_DOCS: u32 = 100_000;
const AMOUNT_OF_DOCUMENT_REPETITIONS: u8 = 10;
const AMOUNT_OF_SECTIONS: u8 = 2;
const AMOUNT_OF_PARAGRAPHS: u8 = 4;
const AMOUNT_OF_SENTENCES: u8 = 2;

const TRAIN_CHUNK_DOCS: u32 = 1_000_000;


const SUBJECTS: [u16; 300] = {
    let mut arr = [0u16; 300];
    let mut i = 0;
    while i < 300 {
        arr[i] = i as u16;
        i += 1;
    }
    arr
};
const VERBS: [u16; 300] = {
    let mut arr = [0u16; 300];
    let mut i = 0;
    while i < 300 {
        arr[i] = (300 + i) as u16;
        i += 1;
    }
    arr
};
const OBJECTS: [u16; 300] = {
    let mut arr = [0u16; 300];
    let mut i = 0;
    while i < 300 {
        arr[i] = (600 + i) as u16;
        i += 1;
    }
    arr
};
const CONNECTORS: [u16; 100] = {
    let mut arr = [0u16; 100];
    let mut i = 0;
    while i < 100 {
        arr[i] = (900 + i) as u16;
        i += 1;
    }
    arr
};
const EOS: u16 = 1000;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DocSpan {
    start: u64,
    len: u32,
}

#[inline]
fn seed32(doc_id: u32, domain_tag: u8) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[..4].copy_from_slice(&doc_id.to_le_bytes());
    seed[31] = domain_tag;
    seed
}

fn permute_for_doc(items: &[u16], doc_id: u32, domain_tag: u8) -> Vec<u16> {
    let mut rng = StdRng::from_seed(seed32(doc_id, domain_tag));
    let mut v = items.to_vec();
    v.shuffle(&mut rng);
    v
}

#[inline]
fn sample_svo<R: Rng + ?Sized>(
    zipf: &Zipf<f64>,
    trng: &mut R,
    subj_perm: &[u16],
    verb_perm: &[u16],
    obj_perm: &[u16],
) -> (u16, u16, u16) {
    let subj_index = zipf.sample(trng) as usize - 1;
    let verb_index = zipf.sample(trng) as usize - 1;
    let object_index = zipf.sample(trng) as usize - 1;
    (
        subj_perm[subj_index],
        verb_perm[verb_index],
        obj_perm[object_index],
    )
}

#[inline]
fn generate_sentence<R: Rng + ?Sized>(
    coin: &Bernoulli,
    zipf: &Zipf<f64>,
    die: &Uniform<u8>,
    trng: &mut R,
    subj_perm: &[u16],
    verb_perm: &[u16],
    obj_perm: &[u16],
) -> Vec<u16> {
    let (subject, verb, object) = sample_svo(zipf, trng, subj_perm, verb_perm, obj_perm);
    let mut sentence: Vec<u16> = Vec::with_capacity(16);

    if coin.sample(trng) {
        let amount_of_connectors = die.sample(trng);

        let push_svo = |out: &mut Vec<u16>, s: u16, v: u16, o: u16| {
            out.push(s);
            out.push(v);
            out.push(o);
        };

        push_svo(&mut sentence, subject, verb, object);

        match amount_of_connectors {
            0 => {}
            1 => {
                let idx = trng.random_range(0..CONNECTORS.len());
                let con = CONNECTORS[idx];
                sentence.push(con);

                let (subject2, verb2, object2) =
                    sample_svo(zipf, trng, subj_perm, verb_perm, obj_perm);

                push_svo(&mut sentence, subject2, verb2, object2);
            }
            _ => {
                let idx = trng.random_range(0..CONNECTORS.len());
                let con1 = CONNECTORS[idx];
                sentence.push(con1);

                let (subject2, verb2, object2) =
                    sample_svo(zipf, trng, subj_perm, verb_perm, obj_perm);

                push_svo(&mut sentence, subject2, verb2, object2);

                let idx = trng.random_range(0..CONNECTORS.len());
                let con2 = CONNECTORS[idx];
                sentence.push(con2);

                let (subject3, verb3, object3) =
                    sample_svo(zipf, trng, subj_perm, verb_perm, obj_perm);

                push_svo(&mut sentence, subject3, verb3, object3);
            }
        }
    } else {
        sentence.push(verb);
        sentence.push(subject);
        sentence.push(object);
    }

    sentence.push(EOS);
    sentence
}

#[inline]
fn encode_tuple_slice(tup: &[u16]) -> u128 {
    const BASE: u128 = 5001;
    let mut k: u128 = 0;
    let mut len: u8 = 0;
    for &v in tup {
        debug_assert!(v <= 5000);
        k = k * BASE + v as u128;
        len += 1;
    }
    (k << 4) | len as u128
}

#[inline]
fn doc_rng(doc_id: u32) -> StdRng {
    StdRng::from_seed(seed32(doc_id, 255))
}

fn build_test_data() -> (Arc<AHashSet<u128>>, Vec<u16>, Vec<DocSpan>, usize) {
    let coin = Bernoulli::new(0.5).unwrap();
    let zipf = Zipf::<f64>::new(300f64, 1.0).expect("n>=1 and s>0");
    let die = Uniform::<u8>::from(Uniform::try_from(0u8..3u8).unwrap());

    let mut test_sentences: AHashSet<u128> = AHashSet::new();
    let mut test_tokens: Vec<u16> = Vec::new();
    let mut test_spans: Vec<DocSpan> = Vec::with_capacity(AMOUNT_TEST_DOCS as usize);
    let mut total_tokens: usize = 0;

    for doc in 0..AMOUNT_TEST_DOCS {
        let subj_perm = permute_for_doc(&SUBJECTS, doc, 0);
        let verb_perm = permute_for_doc(&VERBS, doc, 1);
        let obj_perm = permute_for_doc(&OBJECTS, doc, 2);
        let mut rng = doc_rng(doc);

        let mut rep_idx: Vec<u8> = (0..AMOUNT_OF_DOCUMENT_REPETITIONS).collect();
        rep_idx.shuffle(&mut rng);

        for _rep in rep_idx {
            let doc_start = test_tokens.len() as u64;

            for _sec in 0..AMOUNT_OF_SECTIONS {
                for _para in 0..AMOUNT_OF_PARAGRAPHS {
                    for _sent in 0..AMOUNT_OF_SENTENCES {
                        let sentence = generate_sentence(
                            &coin, &zipf, &die, &mut rng, &subj_perm, &verb_perm, &obj_perm,
                        );
                        total_tokens += sentence.len();
                        let key = encode_tuple_slice(&sentence);
                        test_sentences.insert(key);
                        test_tokens.extend_from_slice(&sentence);
                    }
                }
            }

            let doc_len = (test_tokens.len() as u64 - doc_start) as u32;
            test_spans.push(DocSpan {
                start: doc_start,
                len: doc_len,
            });
        }
    }

    (
        Arc::new(test_sentences),
        test_tokens,
        test_spans,
        total_tokens,
    )
}

fn chunk_ranges(total: u32, chunk: u32) -> Vec<Range<u32>> {
    let mut v = Vec::with_capacity(((total + chunk - 1) / chunk) as usize);
    let mut start = 0;
    while start < total {
        let end = (start + chunk).min(total);
        v.push(start..end);
        start = end;
    }
    v
}

fn generate_chunk_tokens_and_spans(
    range: Range<u32>,
    test_keys: &AHashSet<u128>,
) -> (Vec<Vec<u16>>, usize) {
    let coin = Bernoulli::new(0.5).unwrap();
    let zipf = Zipf::<f64>::new(300f64, 1.0).expect("n>=1 and s>0");
    let die = Uniform::<u8>::from(Uniform::try_from(0u8..3u8).unwrap());

    let est_docs = (range.end - range.start) as usize;
    let mut docs: Vec<Vec<u16>> = Vec::with_capacity(est_docs);

    let mut total_tokens = 0usize;

    for off in range {
        let base_doc_id = AMOUNT_TEST_DOCS + (off / AMOUNT_OF_DOCUMENT_REPETITIONS as u32);
        let rep_tag = (off % AMOUNT_OF_DOCUMENT_REPETITIONS as u32) as u8;

        let subj_perm = permute_for_doc(&SUBJECTS, base_doc_id, 0);
        let verb_perm = permute_for_doc(&VERBS, base_doc_id, 1);
        let obj_perm = permute_for_doc(&OBJECTS, base_doc_id, 2);
        let mut rng = StdRng::from_seed(seed32(base_doc_id, rep_tag));

        let mut doc_tokens = Vec::with_capacity(
            (AMOUNT_OF_SECTIONS as usize)
                * (AMOUNT_OF_PARAGRAPHS as usize)
                * (AMOUNT_OF_SENTENCES as usize)
                * 10,
        );

        for _sec in 0..AMOUNT_OF_SECTIONS {
            for _para in 0..AMOUNT_OF_PARAGRAPHS {
                for _sent in 0..AMOUNT_OF_SENTENCES {
                    let sentence = loop {
                        let s = generate_sentence(
                            &coin, &zipf, &die, &mut rng, &subj_perm, &verb_perm, &obj_perm,
                        );
                        let key = encode_tuple_slice(&s);
                        if !test_keys.contains(&key) {
                            break s;
                        }
                    };
                    total_tokens += sentence.len();
                    doc_tokens.extend_from_slice(&sentence);
                }
            }
        }

        docs.push(doc_tokens);
    }

    (docs, total_tokens)
}

pub fn generate_and_save_pcfg_datasets(
    train_path: &str,
    test_path: &str,
) {
    println!("Generating {} test docs...", AMOUNT_TEST_DOCS);
    let _start = Instant::now();

    let (test_keys, test_tokens, _test_spans, mut _total_tokens) = build_test_data();

    save_u16_as_arrow(test_path, test_tokens.into_iter()).unwrap();
    println!("\nSaved test dataset to {}", test_path);

    println!(
        "\nGenerating {} training docs...",
        AMOUNT_TRAINING_DOCS * AMOUNT_OF_DOCUMENT_REPETITIONS as u32
    );
    let ranges = chunk_ranges(
        AMOUNT_TRAINING_DOCS * AMOUNT_OF_DOCUMENT_REPETITIONS as u32,
        TRAIN_CHUNK_DOCS,
    );

    let results: Vec<(Vec<Vec<u16>>, usize)> = ranges
        .into_par_iter()
        .map(|r| generate_chunk_tokens_and_spans(r, &test_keys))
        .collect();

    println!("Flattening chunk results...");
    let flatten_start = Instant::now();

    let total_train_docs: usize = results.iter().map(|(docs, _)| docs.len()).sum();
    let mut docs: Vec<Vec<u16>> = Vec::with_capacity(total_train_docs);

    for (mut chunk_docs, chunk_tok_count) in results {
        _total_tokens += chunk_tok_count;
        docs.append(&mut chunk_docs);
    }

    let flatten_dur = flatten_start.elapsed();
    println!(
        "Collected {} training docs in {:.2?}.",
        docs.len(),
        flatten_dur
    );

    println!("Shuffling document order...");
    let shuffle_start = Instant::now();
    {
        let mut rng = StdRng::from_seed(seed32(42, 77));
        docs.shuffle(&mut rng);
    }
    let shuffle_dur = shuffle_start.elapsed();
    println!("Shuffled {} documents in {:.2?}.", docs.len(), shuffle_dur);

    println!("Creating shuffled training data...");
    let write_start = Instant::now();

    let train_tokens: Vec<u16> = docs.into_iter().flatten().collect();

    save_u16_as_arrow(train_path, train_tokens.into_iter()).unwrap();

    let write_dur = write_start.elapsed();
    println!(
        "\nSaved training dataset to {} in {:.2?}",
        train_path, write_dur
    );
}
