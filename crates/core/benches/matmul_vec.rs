use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use unifyml::{Matrix, Vector};

/// Helper: build an M x N matrix and an N-element vector filled with deterministic values.
fn make_matrix_vector_pair(rows: usize, cols: usize) -> (Matrix<f32>, Vector<f32>, Vec<f32>, Vec<f32>) {
    let size = rows * cols;
    let mat_data: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| ((i + 3) % 7) as f32).collect();

    let matrix = Matrix::from_vec(rows, cols, mat_data.clone()).unwrap();
    let vector = Vector::from_vec(vec_data.clone());

    (matrix, vector, mat_data, vec_data)
}

/// Baseline: raw slice iteration with zero abstraction overhead.
fn raw_matmul_vector(matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for i in 0..rows {
        let row_offset = i * cols;
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += matrix[row_offset + j] * vector[j];
        }
        out[i] = sum;
    }
    out
}

fn bench_matmul_vector_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_vector");

    // Square and rectangular dimension sweeps (rows, cols)
    let dimensions = [
        (32, 32),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 512), // Tall matrix
        (512, 2048), // Wide matrix
    ];

    for (rows, cols) in dimensions {
        let (matrix, vector, raw_mat, raw_vec) = make_matrix_vector_pair(rows, cols);
        let param_label = format!("{}x{}", rows, cols);

        // 1. Your abstraction: Matrix::matmul_vector
        group.bench_with_input(
            BenchmarkId::new("unifyml_struct", &param_label),
            &(matrix, vector),
            |b, (m, v)| {
                b.iter(|| black_box(m.matmul_vector(black_box(v))))
            },
        );

        // 2. Baseline: Raw slice loops
        group.bench_with_input(
            BenchmarkId::new("raw_slices", &param_label),
            &(raw_mat, raw_vec, rows, cols),
            |b, (m_slice, v_slice, r, c)| {
                b.iter(|| black_box(raw_matmul_vector(black_box(m_slice), black_box(v_slice), *r, *c)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul_vector_comparison);
criterion_main!(benches);
