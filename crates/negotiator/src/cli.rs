//! Flags and setup shared by the command-line tools.

use crate::agent::Strategy;
use anyhow::Result;
use candle_core::Device;
use clap::Args;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Args, Debug, Clone)]
pub struct RuntimeArgs {
    /// Run on the CPU even when built with CUDA support.
    #[arg(long)]
    pub cpu: bool,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    pub gpu: usize,
    /// Random seed.
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

impl RuntimeArgs {
    /// Picks CUDA when available unless `--cpu` is given, and seeds every RNG.
    pub fn setup(&self) -> Result<(Device, StdRng)> {
        let use_cuda = !self.cpu && candle_core::utils::cuda_is_available();
        let device = crate::device::select(use_cuda, self.gpu)?;
        // candle cannot seed its CPU RNG; CPU runs are therefore not bit-reproducible.
        if !device.is_cpu() {
            device.set_seed(self.seed)?;
        }
        eprintln!("device: {}", if use_cuda { format!("cuda:{}", self.gpu) } else { "cpu".into() });
        Ok((device, StdRng::seed_from_u64(self.seed)))
    }
}

#[derive(Args, Debug, Clone)]
#[command(rename_all = "snake_case")]
pub struct RolloutArgs {
    /// Use batched GPU rollouts for planning agents.
    #[arg(long)]
    pub fast_rollout: bool,
    #[arg(long, default_value_t = 100)]
    pub rollout_bsz: usize,
    #[arg(long, default_value_t = 3)]
    pub rollout_count_threshold: usize,
}

impl RolloutArgs {
    pub fn strategy(&self, smart: bool) -> Strategy {
        match (smart, self.fast_rollout) {
            (false, _) => Strategy::Sample,
            (true, false) => Strategy::rollout(),
            (true, true) => Strategy::BatchedRollout {
                bsz: self.rollout_bsz,
                count_threshold: self.rollout_count_threshold,
            },
        }
    }
}

/// Stack size for the worker thread. candle's backward pass walks the graph
/// recursively, and unrolled RNNs over long dialogues are thousands of nodes
/// deep, far beyond Windows' 1 MiB default main-thread stack.
const STACK_BYTES: usize = 512 * 1024 * 1024;

/// Runs `f` on a thread with a stack large enough for deep autograd graphs.
pub fn run_with_big_stack<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    std::thread::Builder::new()
        .name("negotiator".into())
        .stack_size(STACK_BYTES)
        .spawn(f)?
        .join()
        .map_err(|_| anyhow::anyhow!("worker thread panicked"))?
}

/// Derives an independent RNG for a sub-component.
pub fn fork_rng(rng: &mut StdRng) -> StdRng {
    StdRng::seed_from_u64(rng.gen())
}
