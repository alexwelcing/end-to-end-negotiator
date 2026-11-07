//! Corpus loading, vocabularies, and padded mini-batches.

use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub const SPECIAL: [&str; 4] = ["<eos>", "<unk>", "<selection>", "<pad>"];
pub const STOP_TOKENS: [&str; 2] = ["<eos>", "<selection>"];
pub const UNK: &str = "<unk>";
pub const PAD: &str = "<pad>";

/// Returns the tokens between `<tag>` and `</tag>`.
pub fn get_tag<'a>(tokens: &'a [&'a str], tag: &str) -> Result<&'a [&'a str]> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = tokens.iter().position(|t| *t == open).with_context(|| format!("missing {open}"))?;
    let end = tokens.iter().position(|t| *t == close).with_context(|| format!("missing {close}"))?;
    ensure!(start < end, "{close} precedes {open}");
    Ok(&tokens[start + 1..end])
}

pub fn read_lines(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(text.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dictionary {
    idx2word: Vec<String>,
    #[serde(skip)]
    word2idx: HashMap<String, usize>,
}

impl Dictionary {
    pub fn new(with_special: bool) -> Self {
        let mut d = Self::default();
        if with_special {
            SPECIAL.iter().for_each(|w| {
                d.add_word(w);
            });
        }
        d
    }

    /// Rebuilds the reverse index after deserialisation.
    pub fn reindex(mut self) -> Self {
        self.word2idx = self.idx2word.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
        self
    }

    pub fn add_word(&mut self, word: &str) -> usize {
        if let Some(&i) = self.word2idx.get(word) {
            return i;
        }
        let i = self.idx2word.len();
        self.idx2word.push(word.to_string());
        self.word2idx.insert(word.to_string(), i);
        i
    }

    /// Looks a word up, falling back to `<unk>` when the dictionary has one.
    pub fn get_idx(&self, word: &str) -> Option<usize> {
        self.word2idx.get(word).or_else(|| self.word2idx.get(UNK)).copied()
    }

    pub fn idx(&self, word: &str) -> Result<usize> {
        self.get_idx(word).with_context(|| format!("{word:?} is not in the dictionary"))
    }

    pub fn contains(&self, word: &str) -> bool {
        self.word2idx.contains_key(word)
    }

    pub fn w2i<S: AsRef<str>>(&self, words: &[S]) -> Result<Vec<u32>> {
        words.iter().map(|w| self.idx(w.as_ref()).map(|i| i as u32)).collect()
    }

    pub fn word(&self, idx: usize) -> &str {
        &self.idx2word[idx]
    }

    pub fn i2w(&self, idxs: &[u32]) -> Vec<String> {
        idxs.iter().map(|&i| self.idx2word[i as usize].clone()).collect()
    }

    pub fn len(&self) -> usize {
        self.idx2word.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idx2word.is_empty()
    }

    /// Builds a dictionary from one tag of every line, most frequent first,
    /// keeping tokens seen more than `freq_cutoff` times.
    pub fn from_tag(lines: &[String], tag: &str, freq_cutoff: i64, with_special: bool) -> Result<Self> {
        let mut order: Vec<String> = Vec::new();
        let mut freqs: HashMap<String, i64> = HashMap::new();
        for line in lines {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            for &t in get_tag(&tokens, tag)? {
                let f = freqs.entry(t.to_string()).or_insert_with(|| {
                    order.push(t.to_string());
                    0
                });
                *f += 1;
            }
        }
        // Stable sort keeps first-seen order among ties, matching Python's sorted().
        order.sort_by_key(|t| std::cmp::Reverse(freqs[t]));
        let mut d = Self::new(with_special);
        order.iter().filter(|t| freqs[*t] > freq_cutoff).for_each(|t| {
            d.add_word(t);
        });
        Ok(d)
    }
}

/// One dialogue from a single agent's perspective, as vocabulary indices.
#[derive(Debug, Clone)]
pub struct Example {
    pub ctx: Vec<u32>,
    pub words: Vec<u32>,
    pub items: Vec<u32>,
}

/// A padded, time-major batch already resident on the compute device.
pub struct Batch {
    /// `(2 * n_items, bsz)` context tokens.
    pub ctx: Tensor,
    /// `(T - 1, bsz)` input words.
    pub inpt: Tensor,
    /// `((T - 1) * bsz,)` next-word targets.
    pub tgt: Tensor,
    /// `(sel_len * bsz,)` selection targets, selection-slot major.
    pub sel_tgt: Tensor,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStats {
    pub n: usize,
    pub nonpadn: usize,
}

pub struct Vocab {
    pub word: Dictionary,
    pub item: Dictionary,
    pub context: Dictionary,
}

pub struct WordCorpus {
    pub vocab: Vocab,
    pub train: Vec<Example>,
    pub valid: Vec<Example>,
    pub test: Vec<Example>,
    pub output_length: usize,
}

impl WordCorpus {
    /// Loads `train.txt`, `val.txt`, and `test.txt`; vocabularies come from train only.
    pub fn load(dir: &Path, freq_cutoff: i64, verbose: bool) -> Result<Self> {
        let train_lines = read_lines(&dir.join("train.txt"))?;
        let vocab = Vocab {
            word: Dictionary::from_tag(&train_lines, "dialogue", freq_cutoff, true)?,
            item: Dictionary::from_tag(&train_lines, "output", -1, false)?,
            context: Dictionary::from_tag(&train_lines, "input", -1, false)?,
        };
        let train = tokenize(&vocab, &train_lines, "train.txt", verbose)?;
        let valid = tokenize(&vocab, &read_lines(&dir.join("val.txt"))?, "val.txt", verbose)?;
        let test = tokenize(&vocab, &read_lines(&dir.join("test.txt"))?, "test.txt", verbose)?;
        let output_length = train.iter().map(|e| e.items.len()).max().context("empty train set")?;
        Ok(Self { vocab, train, valid, test, output_length })
    }

    /// Shuffles, length-sorts and pads `examples` into batches on `device`.
    pub fn batches<R: Rng>(
        &self,
        examples: &[Example],
        bsz: usize,
        rng: &mut R,
        device: &Device,
    ) -> Result<(Vec<Batch>, BatchStats)> {
        let pad = self.vocab.word.idx(PAD)? as u32;
        make_batches(examples, bsz, pad, rng, device)
    }
}

fn tokenize(vocab: &Vocab, lines: &[String], name: &str, verbose: bool) -> Result<Vec<Example>> {
    let unk = vocab.word.idx(UNK)? as u32;
    let (mut total, mut unks) = (0usize, 0usize);
    let examples = lines
        .iter()
        .map(|line| {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let e = Example {
                ctx: vocab.context.w2i(get_tag(&tokens, "input")?)?,
                words: vocab.word.w2i(get_tag(&tokens, "dialogue")?)?,
                items: vocab.item.w2i(get_tag(&tokens, "output")?)?,
            };
            total += e.ctx.len() + e.words.len() + e.items.len();
            unks += e.words.iter().filter(|&&w| w == unk).count();
            Ok(e)
        })
        .collect::<Result<Vec<_>>>()?;
    if verbose {
        println!(
            "dataset {name}, total {total}, unks {unks}, ratio {:.2}%",
            100.0 * unks as f64 / total.max(1) as f64
        );
    }
    Ok(examples)
}

fn make_batches<R: Rng>(
    examples: &[Example],
    bsz: usize,
    pad: u32,
    rng: &mut R,
    device: &Device,
) -> Result<(Vec<Batch>, BatchStats)> {
    ensure!(bsz > 0, "batch size must be positive");
    let mut order: Vec<&Example> = examples.iter().collect();
    order.shuffle(rng);
    order.sort_by_key(|e| e.words.len());

    let mut stats = BatchStats::default();
    let mut batches = Vec::with_capacity(order.len().div_ceil(bsz));
    for chunk in order.chunks(bsz) {
        let b = chunk.len();
        let max_len = chunk.iter().map(|e| e.words.len()).max().unwrap_or(0);
        ensure!(max_len >= 2, "dialogue too short to model");
        stats.n += max_len * b;
        stats.nonpadn += chunk.iter().map(|e| e.words.len()).sum::<usize>();

        let ctx = time_major(chunk.iter().map(|e| e.ctx.as_slice()), chunk[0].ctx.len(), pad)?;
        let words = time_major(chunk.iter().map(|e| e.words.as_slice()), max_len, pad)?;
        let items = time_major(chunk.iter().map(|e| e.items.as_slice()), chunk[0].items.len(), pad)?;

        let ctx = Tensor::from_vec(ctx, (chunk[0].ctx.len(), b), device)?;
        let words = Tensor::from_vec(words, (max_len, b), device)?;
        let sel_tgt = Tensor::from_vec(items, chunk[0].items.len() * b, device)?;
        let inpt = words.narrow(0, 0, max_len - 1)?;
        let tgt = words.narrow(0, 1, max_len - 1)?.flatten_all()?;
        batches.push(Batch { ctx, inpt, tgt, sel_tgt });
    }
    batches.shuffle(rng);
    Ok((batches, stats))
}

/// Lays sequences out time-major (`[t * b + i]`), right-padding with `pad`.
fn time_major<'a>(seqs: impl Iterator<Item = &'a [u32]> + Clone, len: usize, pad: u32) -> Result<Vec<u32>> {
    let b = seqs.clone().count();
    let mut out = vec![pad; len * b];
    for (i, s) in seqs.enumerate() {
        ensure!(s.len() <= len, "sequence longer than its batch width");
        for (t, &w) in s.iter().enumerate() {
            out[t * b + i] = w;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const LINE: &str = "<input> 1 4 4 1 1 2 </input> <dialogue> THEM: hi <eos> YOU: deal <eos> THEM: <selection> </dialogue> <output> item0=1 item1=0 item2=1 item0=0 item1=4 item2=0 </output>";

    #[test]
    fn dictionary_orders_by_frequency_after_specials() {
        let lines = vec![LINE.to_string()];
        let d = Dictionary::from_tag(&lines, "dialogue", 0, true).unwrap();
        assert_eq!(d.word(0), "<eos>");
        assert_eq!(d.word(4), "THEM:");
        assert_eq!(d.get_idx("never-seen"), d.get_idx(UNK));
        let items = Dictionary::from_tag(&lines, "output", -1, false).unwrap();
        assert!(items.get_idx("<unk>").is_none());
    }

    #[test]
    fn batches_are_time_major_and_padded() {
        let ex = |n: usize| Example { ctx: vec![1, 2], words: (0..n as u32).collect(), items: vec![7, 8] };
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (batches, stats) = make_batches(&[ex(3), ex(5)], 2, 99, &mut rng, &Device::Cpu).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(stats.n, 10);
        assert_eq!(stats.nonpadn, 8);
        let b = &batches[0];
        assert_eq!(b.inpt.dims(), &[4, 2]);
        let inpt: Vec<Vec<u32>> = b.inpt.to_vec2().unwrap();
        assert_eq!(inpt[3], vec![99, 3]);
        assert_eq!(b.sel_tgt.to_vec1::<u32>().unwrap(), vec![7, 7, 8, 8]);
    }
}
