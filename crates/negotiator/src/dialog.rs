//! Runs two agents through a negotiation and logs it.

use crate::agent::Agent;
use crate::data::read_lines;
use crate::domain::ObjectDivision;
use crate::metric::Metrics;
use anyhow::{ensure, Context, Result};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const CODE2ITEM: [(&str, &str); 3] = [("item0", "book"), ("item1", "hat"), ("item2", "ball")];

/// Writes transcripts to stderr (when verbose) and/or a log file.
pub struct DialogLogger {
    verbose: bool,
    file: Option<BufWriter<File>>,
}

impl DialogLogger {
    pub fn new(verbose: bool, log_file: Option<&Path>) -> Result<Self> {
        let file = log_file
            .map(|p| File::create(p).with_context(|| format!("creating {}", p.display())).map(BufWriter::new))
            .transpose()?;
        Ok(Self { verbose, file })
    }

    /// Logs `s`; `forced` also prints it to stdout.
    pub fn dump(&mut self, s: &str, forced: bool) -> Result<()> {
        if self.verbose {
            eprintln!("{s}");
        }
        if let Some(f) = &mut self.file {
            writeln!(f, "{s}")?;
            f.flush()?;
        }
        if forced {
            println!("{s}");
        }
        Ok(())
    }

    fn with_name(&mut self, name: &str, s: &str) -> Result<()> {
        self.dump(&format!("{name:<5} : {s}"), false)
    }

    pub fn dump_ctx(&mut self, name: &str, ctx: &[String]) -> Result<()> {
        ensure!(ctx.len() == 6, "we expect 3 objects");
        let s = CODE2ITEM
            .iter()
            .enumerate()
            .map(|(i, (_, item))| format!("{item}=(count:{} value:{})", ctx[2 * i], ctx[2 * i + 1]))
            .collect::<Vec<_>>()
            .join(" ");
        self.with_name(name, &s)
    }

    pub fn dump_sent(&mut self, name: &str, sent: &[String]) -> Result<()> {
        self.with_name(name, &sent.join(" "))
    }

    pub fn dump_choice(&mut self, name: &str, choice: &[String]) -> Result<()> {
        let rep = |w: &String| match w.split_once('=') {
            Some((code, n)) => CODE2ITEM
                .iter()
                .find(|(c, _)| *c == code)
                .map_or_else(|| w.clone(), |(_, item)| format!("{item}={n}")),
            None => w.clone(),
        };
        self.with_name(name, &choice.iter().map(rep).collect::<Vec<_>>().join(" "))
    }
}

pub struct Outcome {
    pub conv: Vec<Vec<String>>,
    pub agree: bool,
    pub rewards: Vec<i64>,
}

pub struct Dialog {
    domain: ObjectDivision,
    pub metrics: Metrics,
    max_turns: usize,
    has_ref_text: bool,
}

impl Dialog {
    pub fn new(names: [&str; 2], ref_text: Option<&Path>, max_turns: usize) -> Result<Self> {
        let mut m = Metrics::default();
        m.register_average("dialog_len");
        m.register_average("sent_len");
        m.register_percentage("agree");
        m.register_average("advantage");
        m.register_time("time");
        m.register_average("comb_rew");
        for name in names {
            m.register_average(&format!("{name}_rew"));
            m.register_percentage(&format!("{name}_sel"));
            m.register_uniqueness(&format!("{name}_unique"));
        }
        if let Some(path) = ref_text {
            m.register_ngram("full_match", read_lines(path)?.join(" "));
        }
        Ok(Self { domain: ObjectDivision, metrics: m, max_turns, has_ref_text: ref_text.is_some() })
    }

    pub fn show_metrics(&self) -> String {
        self.metrics.show()
    }

    pub fn run<R: Rng>(
        &mut self,
        agents: [&mut dyn Agent; 2],
        ctxs: &[Vec<String>; 2],
        logger: &mut DialogLogger,
        rng: &mut R,
    ) -> Result<Outcome> {
        let [a, b] = agents;
        for (agent, ctx) in [(&mut *a, &ctxs[0]), (&mut *b, &ctxs[1])] {
            agent.feed_context(ctx)?;
            logger.dump_ctx(agent.name(), ctx)?;
        }
        logger.dump(&"-".repeat(80), false)?;

        let names = [a.name().to_string(), b.name().to_string()];
        let mut agents: [&mut dyn Agent; 2] = [a, b];
        let mut writer = usize::from(rng.gen::<f64>() >= 0.5);
        let mut conv = Vec::new();
        let ngram = self.has_ref_text;
        self.metrics.reset();

        loop {
            let out = agents[writer].write()?;
            let wname = &names[writer];
            self.metrics.record("sent_len", out.len() as f64);
            if ngram {
                self.metrics.record_sent("full_match", &out);
            }
            self.metrics.record_sent(&format!("{wname}_unique"), &out);
            agents[1 - writer].read(&out)?;
            if !agents[writer].is_human() {
                logger.dump_sent(wname, &out)?;
            }
            let is_selection = out.len() == 1 && out[0] == "<selection>";
            conv.push(out);
            if is_selection {
                self.metrics.record(&format!("{wname}_sel"), 1.0);
                self.metrics.record(&format!("{}_sel", names[1 - writer]), 0.0);
                break;
            }
            if conv.len() >= self.max_turns {
                logger.dump(&format!("reached max_turns={}, forcing a choice", self.max_turns), false)?;
                break;
            }
            writer = 1 - writer;
        }

        let half = self.domain.selection_length() / 2;
        let mut choices = Vec::with_capacity(2);
        for (agent, name) in agents.iter_mut().zip(&names) {
            let choice = agent.choose()?;
            logger.dump_choice(name, &choice[..half.min(choice.len())])?;
            choices.push(choice);
        }

        let (agree, rewards) = self.domain.score_choices(&choices, ctxs)?;
        logger.dump(&"-".repeat(80), false)?;
        logger.dump(if agree { "Agreement!" } else { "Disagreement?!" }, false)?;
        for ((agent, name), &reward) in agents.iter_mut().zip(&names).zip(&rewards) {
            if agree {
                logger.dump(&format!("{name:<5} : {reward} points"), false)?;
            } else {
                logger.dump(&format!("{name:<5} : 0 (potential {reward})"), false)?;
            }
            agent.update(agree, reward)?;
        }

        if agree {
            self.metrics.record("advantage", (rewards[0] - rewards[1]) as f64);
        }
        self.metrics.record("time", 0.0);
        self.metrics.record("dialog_len", conv.len() as f64);
        self.metrics.record("agree", f64::from(u8::from(agree)));
        self.metrics.record("comb_rew", if agree { rewards.iter().sum::<i64>() as f64 } else { 0.0 });
        for (name, &reward) in names.iter().zip(&rewards) {
            self.metrics.record(&format!("{name}_rew"), if agree { reward as f64 } else { 0.0 });
        }

        logger.dump(&"-".repeat(80), false)?;
        logger.dump(&self.show_metrics(), false)?;
        logger.dump(&"-".repeat(80), false)?;
        for (ctx, choice) in ctxs.iter().zip(&choices) {
            logger.dump(&format!("debug: {} {}", ctx.join(" "), choice.join(" ")), false)?;
        }
        Ok(Outcome { conv, agree, rewards })
    }
}

/// Pairs of contexts (one per agent) read from a self-play context file.
pub struct ContextGenerator {
    ctxs: Vec<[Vec<String>; 2]>,
}

impl ContextGenerator {
    pub fn from_file(path: &Path) -> Result<Self> {
        let lines = read_lines(path)?;
        let ctxs = lines
            .chunks_exact(2)
            .map(|p| [0, 1].map(|i| p[i].split_whitespace().map(String::from).collect()))
            .collect::<Vec<_>>();
        ensure!(!ctxs.is_empty(), "no contexts in {}", path.display());
        Ok(Self { ctxs })
    }

    pub fn len(&self) -> usize {
        self.ctxs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ctxs.is_empty()
    }

    pub fn sample<R: Rng>(&self, rng: &mut R) -> &[Vec<String>; 2] {
        self.ctxs.choose(rng).expect("non-empty by construction")
    }

    /// All contexts, reshuffled each epoch.
    pub fn epochs<R: Rng>(&self, nepoch: usize, rng: &mut R) -> Vec<[Vec<String>; 2]> {
        let mut out = Vec::with_capacity(nepoch * self.ctxs.len());
        for _ in 0..nepoch {
            let mut epoch = self.ctxs.clone();
            epoch.shuffle(rng);
            out.extend(epoch);
        }
        out
    }
}
