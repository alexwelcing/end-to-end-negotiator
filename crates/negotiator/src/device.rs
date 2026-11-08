use anyhow::Result;
use candle_core::Device;

/// Picks the compute device: CUDA ordinal when built with `--features cuda`
/// and `use_cuda` is set, CPU otherwise.
pub fn select(use_cuda: bool, ordinal: usize) -> Result<Device> {
    if use_cuda {
        if !candle_core::utils::cuda_is_available() {
            anyhow::bail!("CUDA requested but this binary was built without the `cuda` feature");
        }
        return Ok(Device::new_cuda(ordinal)?);
    }
    Ok(Device::Cpu)
}
