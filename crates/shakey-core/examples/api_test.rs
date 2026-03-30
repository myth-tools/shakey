use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let w = candle_core::Var::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &dev)?;
    let x = Tensor::randn(0.0, 1.0, (2, 2), &dev)?;
    let y = w.as_tensor().matmul(&x)?;
    let loss = y.sum_all()?;

    // Check backward return type
    let grads = loss.backward()?;

    // Check the actual type by giving it a wrong one
    // let _ : i32 = grads;

    // Test Grads access
    if let Some(grad) = grads.get(&w) {
        println!("Grad found!");
        let _: Tensor = grad.clone();
    }

    println!("API Check Passed!");
    Ok(())
}
