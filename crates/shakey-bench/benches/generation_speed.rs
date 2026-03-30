use candle_core::Device;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use shakey_agent::generation::specular::SpecularEngine;
use shakey_core::inference::{InferenceEngine, SamplingParams};
use shakey_core::model::config::ModelConfig;
use shakey_core::model::transformer::TransformerModel;
use shakey_core::tokenizer::Tokenizer;

fn bench_generation(c: &mut Criterion) {
    let device = Device::Cpu;
    let config = ModelConfig::seed(); // Small model for CPU bench
    let map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&map, candle_core::DType::F32, &device);
    let model = TransformerModel::new(&config, vb).unwrap();

    // Byte-level tokenizer (real, production tokenizer — no mocks)
    let tokenizer = Tokenizer::byte_level();

    let engine = InferenceEngine::new(&model, &tokenizer, device.clone());
    let specular = SpecularEngine::new(&model, &tokenizer, device.clone());

    let prompt = "Explain the OODA loop in 3 sentences.";
    let params = SamplingParams {
        max_tokens: 50,
        temperature: 0.0, // Greedy for consistency
        ..Default::default()
    };

    let mut group = c.benchmark_group("WCSI Generation");

    group.bench_function("Standard Autoregressive", |b| {
        b.iter(|| {
            let _ = engine.generate(black_box(prompt), &params).unwrap();
        })
    });

    group.bench_function("WCSI Specular (3 Heads)", |b| {
        b.iter(|| {
            specular
                .generate_specular(black_box(prompt), &params, |_| Ok(()))
                .unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_generation);
criterion_main!(benches);
