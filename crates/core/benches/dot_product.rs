use criterion::{black_box, criterion_group, criterion_main, Criterion};
use unifyml::Vector;

/// Helper: build two vectors of a given size filled with small values.
/// We do this OUTSIDE the timed region so we only measure `dot`, not allocation.
fn make_pair(size: usize) -> (Vector<f32>, Vector<f32>) {
    let a = Vector::from_vec((0..size).map(|i| (i % 10) as f32).collect());
    let b = Vector::from_vec((0..size).map(|i| ((i + 3) % 7) as f32).collect());
    (a, b)
}

fn bench_dot(c: &mut Criterion, size: usize) {
    let (a, b) = make_pair(size);
    let name = format!("dot/size_{}", size);

    c.bench_function(&name, |bencher| {
        bencher.iter(|| {
            // black_box prevents the compiler from constant-folding the result away
            black_box(black_box(&a).dot(black_box(&b)))
        });
    });
}

fn bench_dot_sizes(c: &mut Criterion) {
    let sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000];
    for size in sizes {
        bench_dot(c, size);
    }
}

criterion_group!(benches, bench_dot_sizes);
criterion_main!(benches);
