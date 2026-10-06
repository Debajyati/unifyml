#[cfg(test)]
mod matrix_tests {
    use unifyml::*;

    // ===================== Construction =====================

    #[test]
    fn new_creates_empty_matrix() {
        let m: Matrix<f32> = Matrix::new();
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn default_creates_empty_matrix() {
        let m: Matrix<f32> = Matrix::default();
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 0);
    }

    #[test]
    fn zeros_creates_correct_shape_and_values() {
        let m = Matrix::<f32>::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.len(), 12);
        for &v in m.as_slice() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn zeros_with_zero_dimension() {
        let m = Matrix::<f32>::zeros(0, 5);
        assert_eq!(m.len(), 0);
        let m2 = Matrix::<f32>::zeros(5, 0);
        assert_eq!(m2.len(), 0);
    }

    #[test]
    fn ones_creates_correct_shape_and_values() {
        let m = Matrix::<f32>::ones(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        for &v in m.as_slice() {
            assert_eq!(v, 1.0);
        }
    }

    // ===================== Access =====================

    #[test]
    fn get_and_index_reads_correct_element() {
        // Build a 2x3 matrix with values 1..=6
        let mut m = Matrix::<f32>::zeros(2, 3);
        for (i, v) in (1..=6).enumerate() {
            m.as_mut_slice()[i] = v as f32;
        }
        // Layout:
        // [1 2 3]
        // [4 5 6]
        assert_eq!(*m.get(0, 0).unwrap(), 1.0);
        assert_eq!(*m.get(0, 2).unwrap(), 3.0);

        // Also test the Index trait
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn get_returns_none_on_out_of_bounds() {
        let mut m = Matrix::<f32>::zeros(2, 2);
        assert_eq!(m.get(5, 0), None);
        assert_eq!(m.get(0, 5), None);
        assert_eq!(m.get_mut(5, 0), None);
        assert_eq!(m.get_mut(0, 5), None);
    }

    #[test]
    fn index_mut_writes_correct_element() {
        let mut m = Matrix::<f32>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(1, 1)] = 9.0;
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 9.0);
    }

    #[test]
    #[should_panic]
    fn index_panics_on_out_of_bounds() {
        let m = Matrix::<f32>::zeros(2, 2);
        let _ = m[(5, 0)];
    }

    // ===================== Transpose =====================

    #[test]
    fn transpose_square_matrix() {
        let mut m = Matrix::<f32>::zeros(2, 2);
        m[(0, 0)] = 1.0; m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0; m[(1, 1)] = 4.0;

        let t = m.transpose();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 3.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(1, 1)], 4.0);
    }

    #[test]
    fn transpose_rectangular_matrix() {
        let mut m = Matrix::<f32>::zeros(2, 3);
        for (i, v) in (1..=6).enumerate() {
            m[(i / 3, i % 3)] = v as f32;
        }

        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(1, 1)], 5.0);
        assert_eq!(t[(2, 0)], 3.0);
        assert_eq!(t[(2, 1)], 6.0);
    }

    #[test]
    fn transpose_empty_matrix() {
        let m: Matrix<f32> = Matrix::new();
        let t = m.transpose();
        assert_eq!(t.rows(), 0);
        assert_eq!(t.cols(), 0);
    }

    // [A] INVARIANT: (A^T)^T = A
    #[test]
    fn transpose_twice_returns_original() {
        let mut a = Matrix::<f32>::zeros(2, 3);
        for (i, v) in (1..=6).enumerate() {
            a[(i / 3, i % 3)] = v as f32;
        }
        let result = a.transpose().transpose();
        assert_eq!(result, a);
    }

    // ===================== Math Operations & Invariants =====================

    #[test]
    fn add_two_matrices() {
        let mut a = Matrix::<f32>::zeros(2, 2);
        let mut b = Matrix::<f32>::zeros(2, 2);
        a[(0, 0)] = 1.0; a[(1, 1)] = 2.0;
        b[(0, 0)] = 10.0; b[(0, 1)] = 20.0;

        let c = a.add(&b).unwrap();
        assert_eq!(c[(0, 0)], 11.0);
        assert_eq!(c[(0, 1)], 20.0);
        assert_eq!(c[(1, 0)], 0.0);
        assert_eq!(c[(1, 1)], 2.0);
    }

    // [B] INVARIANT: A + 0 = A
    #[test]
    fn add_zero_matrix_returns_original() {
        let a = Matrix::<f32>::ones(3, 4);
        let zero = Matrix::<f32>::zeros(3, 4);
        assert_eq!(a.add(&zero).unwrap(), a);
    }

    // [B] INVARIANT: A + B = B + A
    #[test]
    fn addition_is_commutative() {
        let mut a = Matrix::<f32>::zeros(2, 2);
        a[(0, 0)] = 1.0; a[(1, 1)] = 2.0;

        let mut b = Matrix::<f32>::zeros(2, 2);
        b[(0, 1)] = 3.0; b[(1, 0)] = 4.0;

        assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());
    }

    #[test]
    fn add_dimension_mismatch() {
        let a = Matrix::<f32>::zeros(2, 3);
        let b = Matrix::<f32>::zeros(3, 2);
        let err = a.add(&b).unwrap_err();
        match err {
            MatrixError::DimensionMismatch { left, right } => {
                assert_eq!(left, (2, 3));
                assert_eq!(right, (3, 2));
            }
            _ => panic!("Expected DimensionMismatch"),
        }
    }

    // [C] INVARIANT: (A + B) - B = A
    #[test]
    fn subtraction_reverses_addition() {
        let mut a = Matrix::<f32>::zeros(2, 2);
        a[(0, 0)] = 5.0; a[(1, 1)] = 10.0;

        let mut b = Matrix::<f32>::ones(2, 2);
        b[(0, 1)] = 2.0;

        let added = a.add(&b).unwrap();
        let subtracted = added.sub(&b).unwrap();
        assert_eq!(subtracted, a);
    }

    #[test]
    fn sub_dimension_mismatch() {
        let a = Matrix::<f32>::zeros(2, 2);
        let b = Matrix::<f32>::zeros(2, 3);
        assert!(a.sub(&b).is_err());
    }

    // ===================== Matrix Multiplication =====================

    #[test]
    fn matmul_2x2_basic() {
        let mut a = Matrix::<f32>::zeros(2, 2);
        a[(0, 0)] = 1.0; a[(0, 1)] = 2.0;
        a[(1, 0)] = 3.0; a[(1, 1)] = 4.0;

        let mut b = Matrix::<f32>::zeros(2, 2);
        b[(0, 0)] = 5.0; b[(0, 1)] = 6.0;
        b[(1, 0)] = 7.0; b[(1, 1)] = 8.0;

        let c = a.matmul(&b).unwrap();
        assert_eq!(c[(0, 0)], 19.0);
        assert_eq!(c[(0, 1)], 22.0);
        assert_eq!(c[(1, 0)], 43.0);
        assert_eq!(c[(1, 1)], 50.0);
    }

    #[test]
    fn matmul_rectangular_2x3_times_3x2() {
        let mut a = Matrix::<f32>::zeros(2, 3);
        for (i, v) in (1..=6).enumerate() {
            a[(i / 3, i % 3)] = v as f32;
        }
        let mut b = Matrix::<f32>::zeros(3, 2);
        for (i, v) in (7..=12).enumerate() {
            b[(i / 2, i % 2)] = v as f32;
        }

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c[(0, 0)], 58.0);
        assert_eq!(c[(0, 1)], 64.0);
        assert_eq!(c[(1, 0)], 139.0);
        assert_eq!(c[(1, 1)], 154.0);
    }

    // [D] INVARIANT: Catching Indexing Errors (3x2 * 2x4 -> 3x4)
    #[test]
    fn matmul_rectangular_3x2_times_2x4() {
        let mut a = Matrix::<f32>::zeros(3, 2);
        for i in 0..6 { a[(i / 2, i % 2)] = (i + 1) as f32; }

        let mut b = Matrix::<f32>::zeros(2, 4);
        for i in 0..8 { b[(i / 4, i % 4)] = (i + 1) as f32; }

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 4);

        // Manual verification of a few spots:
        // C(0, 0) = A(row 0) • B(col 0) = [1, 2] • [1, 5] = 1*1 + 2*5 = 11
        // C(2, 3) = A(row 2) • B(col 3) = [5, 6] • [4, 8] = 5*4 + 6*8 = 68
        assert_eq!(c[(0, 0)], 11.0);
        assert_eq!(c[(2, 3)], 68.0);
    }

    // [E] INVARIANT: I * A = A and A * I = A
    #[test]
    fn matmul_identity_on_both_sides() {
        let mut a = Matrix::<f32>::zeros(2, 2);
        a[(0, 0)] = 1.0; a[(0, 1)] = 2.0;
        a[(1, 0)] = 3.0; a[(1, 1)] = 4.0;

        let mut i = Matrix::<f32>::zeros(2, 2);
        i[(0, 0)] = 1.0; i[(1, 1)] = 1.0;

        let left_identity = i.matmul(&a).unwrap();
        assert_eq!(left_identity, a);

        let right_identity = a.matmul(&i).unwrap();
        assert_eq!(right_identity, a);
    }

    // [F] INVARIANT: (A * B) * C = A * (B * C)
    #[test]
    fn matmul_associativity() {
        let mut a = Matrix::<f32>::zeros(2, 3);
        for i in 0..6 { a[(i / 3, i % 3)] = (i + 1) as f32; }

        let mut b = Matrix::<f32>::zeros(3, 2);
        for i in 0..6 { b[(i / 2, i % 2)] = (i + 1) as f32; }

        let mut c = Matrix::<f32>::zeros(2, 4);
        for i in 0..8 { c[(i / 4, i % 4)] = (i + 1) as f32; }

        // (A * B) * C
        let ab = a.matmul(&b).unwrap();
        let ab_c = ab.matmul(&c).unwrap();

        // A * (B * C)
        let bc = b.matmul(&c).unwrap();
        let a_bc = a.matmul(&bc).unwrap();

        // Using small integers cast to f32 means exact float equality is safe here
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn matmul_dimension_mismatch() {
        let a = Matrix::<f32>::zeros(2, 3);
        let b = Matrix::<f32>::zeros(2, 2);
        let err = a.matmul(&b).unwrap_err();
        match err {
            MatrixError::DimensionMismatch { left, right } => {
                assert_eq!(left, (2, 3));
                assert_eq!(right, (2, 2));
            }
            _ => panic!("Expected DimensionMismatch"),
        }
    }

    // ===================== Clone / PartialEq =====================

    #[test]
    fn clone_is_independent() {
        let mut a = Matrix::<f32>::zeros(2, 2);
        a[(0, 0)] = 5.0;
        let mut b = a.clone();
        b[(0, 0)] = 99.0;
        assert_eq!(a[(0, 0)], 5.0);
    }

    #[test]
    fn partial_eq() {
        let a = Matrix::<f32>::ones(2, 2);
        let b = Matrix::<f32>::ones(2, 2);
        let c = Matrix::<f32>::zeros(2, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

