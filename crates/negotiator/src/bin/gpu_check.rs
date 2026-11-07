use anyhow::Result;
use candle_core::{DType, Tensor};

fn main() -> Result<()> {
    let device = negotiator::device::select(true, 0)?;
    let a = Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
    let b = a.matmul(&a.t()?)?.sum_all()?.to_scalar::<f32>()?;
    let x = candle_core::Var::randn(0f32, 1.0, (64, 32), &device)?;
    let loss = x.as_tensor().sqr()?.sum_all()?;
    let grads = loss.backward()?;
    let g = grads.get(&x).unwrap().to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    println!("device={device:?} matmul_sum={b:.3} grad_sum={g:.3}");
    Ok(())
}
