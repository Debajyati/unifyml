use criterion::{black_box, criterion_group, criterion_main, Criterion};
use unifyml::Matrix;

/// Helper: build two square matrices of size N x N filled with small values.
/// We construct these from flat Vecs for fast initialization.
fn make_matrix_pair(n: usize) -> (Matrix<f32>, Matrix<f32>) {
    let size = n * n;

    let a_data: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| ((i + 3) % 7) as f32).collect();

    // Using unwrap() here since we guarantee the lengths match n * n
    let a = Matrix::from_vec(n, n, a_data).unwrap();
    let b = Matrix::from_vec(n, n, b_data).unwrap();

    (a, b)
}

fn bench_matmul(c: &mut Criterion, n: usize) {
    let (a, b) = make_matrix_pair(n);
    let name = format!("matmul/{}x{}", n, n);

    c.bench_function(&name, |bencher| {
        bencher.iter(|| {
            // black_box prevents the compiler from constant-folding or skipping the calculation
            black_box(black_box(&a).matmul(black_box(&b)))
        });
    });
}

fn bench_matmul_sizes(c: &mut Criterion) {
    // Matmul is O(N^3). N=256 means ~16.7 million multiplications and additions per iteration.
    // Adjust these sizes based on how long you are willing to wait for the benchmarks.
    let sizes = [10, 32, 64, 128, 256, 512, 1024];
    for size in sizes {
        bench_matmul(c, size);
    }
}

criterion_group!(benches, bench_matmul_sizes);
criterion_main!(benches);
