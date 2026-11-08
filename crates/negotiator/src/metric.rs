//! Running dialogue statistics reported during self-play and RL.

use std::collections::HashSet;
use std::time::Instant;

enum Kind {
    Average { k: f64, n: f64 },
    Percentage { k: f64, n: f64 },
    Time { total: f64, n: f64, last: Instant },
    /// Fraction of generated sentences found verbatim in the reference text.
    NGram { text: String, k: f64, n: f64 },
    Uniqueness { seen: HashSet<String> },
}

impl Kind {
    fn show(&self) -> String {
        let ratio = |k: f64, n: f64| k / n.max(1.0);
        match self {
            Self::Average { k, n } => format!("{:.2}", ratio(*k, *n)),
            Self::Percentage { k, n } => format!("{:.2}%", 100.0 * ratio(*k, *n)),
            Self::Time { total, n, .. } => format!("{:.3}s", ratio(*total, *n)),
            Self::NGram { k, n, .. } => format!("{:.2}", ratio(*k, *n)),
            Self::Uniqueness { seen } => seen.len().to_string(),
        }
    }
}

/// An ordered set of named metrics; names are case-insensitive.
#[derive(Default)]
pub struct Metrics {
    entries: Vec<(String, Kind)>,
}

impl Metrics {
    fn register(&mut self, name: &str, kind: Kind) {
        let name = name.to_lowercase();
        assert!(self.find(&name).is_none(), "metric {name} registered twice");
        self.entries.push((name, kind));
    }

    fn find(&mut self, name: &str) -> Option<&mut Kind> {
        let name = name.to_lowercase();
        self.entries.iter_mut().find(|(n, _)| *n == name).map(|(_, k)| k)
    }

    fn get(&mut self, name: &str) -> &mut Kind {
        self.find(name).unwrap_or_else(|| panic!("unknown metric {name}"))
    }

    pub fn register_average(&mut self, name: &str) {
        self.register(name, Kind::Average { k: 0.0, n: 0.0 });
    }

    pub fn register_percentage(&mut self, name: &str) {
        self.register(name, Kind::Percentage { k: 0.0, n: 0.0 });
    }

    pub fn register_time(&mut self, name: &str) {
        self.register(name, Kind::Time { total: 0.0, n: 0.0, last: Instant::now() });
    }

    pub fn register_ngram(&mut self, name: &str, text: String) {
        self.register(name, Kind::NGram { text, k: 0.0, n: 0.0 });
    }

    pub fn register_uniqueness(&mut self, name: &str) {
        self.register(name, Kind::Uniqueness { seen: HashSet::new() });
    }

    /// Records a numeric observation for an average, percentage, or time metric.
    pub fn record(&mut self, name: &str, value: f64) {
        match self.get(name) {
            Kind::Average { k, n } | Kind::Percentage { k, n } => {
                *k += value;
                *n += 1.0;
            }
            Kind::Time { total, n, last } => {
                *total += last.elapsed().as_secs_f64();
                *n += 1.0;
            }
            _ => panic!("metric {name} does not take numbers"),
        }
    }

    /// Records a sentence for an n-gram or uniqueness metric.
    pub fn record_sent(&mut self, name: &str, sent: &[String]) {
        let joined = sent.join(" ");
        match self.get(name) {
            Kind::NGram { text, k, n } => {
                *n += 1.0;
                if text.contains(&joined) {
                    *k += 1.0;
                }
            }
            Kind::Uniqueness { seen } => {
                seen.insert(joined);
            }
            _ => panic!("metric {name} does not take sentences"),
        }
    }

    /// Restarts per-dialogue timers; accumulated values are kept.
    pub fn reset(&mut self) {
        for (_, kind) in &mut self.entries {
            if let Kind::Time { last, .. } = kind {
                *last = Instant::now();
            }
        }
    }

    pub fn show(&self) -> String {
        self.entries.iter().map(|(n, k)| format!("{n}={}", k.show())).collect::<Vec<_>>().join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_the_reference_implementation() {
        let mut m = Metrics::default();
        m.register_average("Len");
        m.register_percentage("agree");
        m.register_uniqueness("u");
        m.register_ngram("match", "i want the ball . <eos>".into());
        m.record("len", 3.0);
        m.record("len", 4.0);
        m.record("agree", 1.0);
        m.record("agree", 0.0);
        let s = |x: &str| x.split(' ').map(String::from).collect::<Vec<_>>();
        m.record_sent("u", &s("a b"));
        m.record_sent("u", &s("a b"));
        m.record_sent("match", &s("the ball ."));
        m.record_sent("match", &s("no deal"));
        assert_eq!(m.show(), "len=3.50 agree=50.00% u=1 match=0.50");
    }
}
