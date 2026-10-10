use unifyml::Tensor;
use unifyml::TensorError;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tensor_creation_and_indexing() {
        // 2 x 3 x 4 tensor = 24 elements
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(vec![2, 3, 4], data).unwrap();

        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.len(), 24);
        assert_eq!(tensor.strides(), &[12, 4, 1]);

        // Check specific indices
        // [0, 0, 0] -> offset 0 -> value 0.0
        assert_eq!(tensor.get(&[0, 0, 0]), Some(&0.0));
        // [1, 2, 3] -> offset (1*12) + (2*4) + (3*1) = 12 + 8 + 3 = 23 -> value 23.0
        assert_eq!(tensor.get(&[1, 2, 3]), Some(&23.0));
        // [1, 0, 0] -> offset 12 -> value 12.0
        assert_eq!(tensor.get(&[1, 0, 0]), Some(&12.0));

        // Out of bounds checks
        assert!(tensor.get(&[2, 0, 0]).is_none());
        assert!(tensor.get(&[1, 3, 0]).is_none());
        assert!(tensor.get(&[1, 2]).is_none()); // Dimension rank mismatch
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0f32; 24];
        let mut tensor = Tensor::from_vec(vec![2, 3, 4], data).unwrap();

        // Reshape to [4, 6]
        assert!(tensor.reshape(vec![4, 6]).is_ok());
        assert_eq!(tensor.shape(), &[4, 6]);
        assert_eq!(tensor.strides(), &[6, 1]);

        // Invalid reshape size
        assert!(tensor.reshape(vec![5, 5]).is_err());
    }

    #[test]
    fn test_calculate_offset_rank_3() {
        // Shape: [2, 3, 4] -> Strides: [12, 4, 1] -> Total elements: 24
        let tensor = Tensor::<f32>::zeros(vec![2, 3, 4]);

        // 1. Origin [0, 0, 0]
        assert_eq!(tensor.calculate_offset(&[0, 0, 0]), Ok(0));

        // 2. Non-zero multi-dimensional indices
        // [1, 2, 3] -> (1 * 12) + (2 * 4) + (3 * 1) = 12 + 8 + 3 = 23
        assert_eq!(tensor.calculate_offset(&[1, 2, 3]), Ok(23));

        // [1, 0, 0] -> (1 * 12) + (0 * 4) + (0 * 1) = 12
        assert_eq!(tensor.calculate_offset(&[1, 0, 0]), Ok(12));

        // [0, 2, 1] -> (0 * 12) + (2 * 4) + (1 * 1) = 9
        assert_eq!(tensor.calculate_offset(&[0, 2, 1]), Ok(9));
    }

    #[test]
    fn test_calculate_offset_rank_4() {
        // Shape: [2, 3, 4, 5] -> Strides: [60, 20, 5, 1] -> Total elements: 120
        let tensor = Tensor::<f32>::zeros(vec![2, 3, 4, 5]);

        // 1. Origin [0, 0, 0, 0]
        assert_eq!(tensor.calculate_offset(&[0, 0, 0, 0]), Ok(0));

        // 2. Non-zero multi-dimensional indices
        // [1, 2, 3, 4] -> (1 * 60) + (2 * 20) + (3 * 5) + (4 * 1) = 60 + 40 + 15 + 4 = 119
        assert_eq!(tensor.calculate_offset(&[1, 2, 3, 4]), Ok(119));

        // [0, 1, 2, 3] -> (0 * 60) + (1 * 20) + (2 * 5) + (3 * 1) = 0 + 20 + 10 + 3 = 33
        assert_eq!(tensor.calculate_offset(&[0, 1, 2, 3]), Ok(33));

        // [1, 0, 3, 2] -> (1 * 60) + (0 * 20) + (3 * 5) + (2 * 1) = 60 + 0 + 15 + 2 = 77
        assert_eq!(tensor.calculate_offset(&[1, 0, 3, 2]), Ok(77));
    }

    #[test]
    fn test_calculate_offset_out_of_bounds() {
        // Rank-3 tensor with shape [2, 3, 4]
        let tensor = Tensor::<f32>::zeros(vec![2, 3, 4]);

        // Case 1: Index equals dimension size along axis 0 (valid range: 0..2)
        assert_eq!(
            tensor.calculate_offset(&[2, 0, 0]),
            Err(TensorError::IndexOutOfBounds {
                dims: 3,
                index: vec![2, 0, 0],
            })
        );

        // Case 2: Index exceeds dimension size along axis 1 (valid range: 0..3)
        assert_eq!(
            tensor.calculate_offset(&[0, 5, 0]),
            Err(TensorError::IndexOutOfBounds {
                dims: 3,
                index: vec![0, 5, 0],
            })
        );

        // Case 3: Index equals dimension size along axis 2 (valid range: 0..4)
        assert_eq!(
            tensor.calculate_offset(&[1, 2, 4]),
            Err(TensorError::IndexOutOfBounds {
                dims: 3,
                index: vec![1, 2, 4],
            })
        );

        // Case 4: Dimension count mismatch (too few indices: 2 indices for rank-3 shape)
        assert_eq!(
            tensor.calculate_offset(&[1, 2]),
            Err(TensorError::IndexOutOfBounds {
                dims: 3,
                index: vec![1, 2],
            })
        );

        // Case 5: Dimension count mismatch (too many indices: 4 indices for rank-3 shape)
        assert_eq!(
            tensor.calculate_offset(&[0, 0, 0, 0]),
            Err(TensorError::IndexOutOfBounds {
                dims: 3,
                index: vec![0, 0, 0, 0],
            })
        );
    }

    #[test]
    fn test_shape_contract_rank_0_scalar() {
        // shape = [] -> Rank-0 Scalar Tensor
        let tensor = Tensor::<f32>::from_vec(vec![], vec![42.0]).unwrap();

        assert_eq!(tensor.ndim(), 0);
        assert_eq!(tensor.len(), 1);
        assert_eq!(tensor.strides(), &[]);
        assert_eq!(tensor.as_slice(), &[42.0]);
        // Rank-0 indexing with empty index slice
        assert_eq!(tensor.get(&[]), Some(&42.0));
    }

    #[test]
    fn test_shape_contract_rank_1_unit() {
        // shape = [1] -> Rank-1 Unit Vector Tensor
        let tensor = Tensor::<f32>::from_vec(vec![1], vec![10.0]).unwrap();

        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.len(), 1);
        assert_eq!(tensor.strides(), &[1]);
        assert_eq!(tensor.get(&[0]), Some(&10.0));
    }

    #[test]
    fn test_shape_contract_zero_length() {
        // shape = [0] -> Rank-1 Empty Tensor
        let tensor = Tensor::<f32>::zeros(vec![0]);

        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.len(), 0);
        assert_eq!(tensor.strides(), &[1]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.get(&[0]), None);
    }

    #[test]
    fn test_shape_contract_degenerate_zero_dimension() {
        // shape = [2, 0, 4] -> Rank-3 Degenerate Empty Tensor
        let tensor = Tensor::<f32>::zeros(vec![2, 0, 4]);

        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.len(), 0);
        assert_eq!(tensor.strides(), &[0, 4, 1]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.get(&[0, 0, 0]), None);
    }

    #[test]
    fn test_from_slice_construction() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_slice(vec![2, 3], &data).unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_slice(), &data);
        assert_eq!(tensor.get(&[1, 2]), Some(&6.0));
    }

    #[test]
    fn test_reshape_is_zero_copy_and_mutates_underlying_data() {
        // 1. Construct initial 2x3x4 tensor
        let initial_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut tensor = Tensor::from_vec(vec![2, 3, 4], initial_data.clone()).unwrap();

        // Verify original element ordering
        assert_eq!(tensor.as_slice(), initial_data.as_slice());

        // 2. Perform zero-copy reshape to [4, 6]
        assert!(tensor.reshape(vec![4, 6]).is_ok());
        assert_eq!(tensor.shape(), &[4, 6]);
        assert_eq!(tensor.strides(), &[6, 1]);

        // Element ordering in raw flat memory MUST remain identical (0..23)
        assert_eq!(tensor.as_slice(), initial_data.as_slice());

        // 3. Mutate an element via reshaped index and verify original storage reflects mutation
        // In shape [4, 6], index [2, 3] corresponds to offset (2 * 6) + 3 = 15
        if let Some(val) = tensor.get_mut(&[2, 3]) {
            *val = 999.0;
        }

        // Verify flat storage reflects the mutation directly at index 15
        assert_eq!(tensor.as_slice()[15], 999.0);

        // Reshape back to [2, 3, 4] and verify mutated value is retrieved at original 3D index [1, 0, 3] (offset 15)
        assert!(tensor.reshape(vec![2, 3, 4]).is_ok());
        assert_eq!(tensor.get(&[1, 0, 3]), Some(&999.0));
    }
}

#[cfg(test)]
mod tensor_algebra_tests {
    use super::*;

    #[test]
    fn test_scale_properties() {
        let shape = vec![2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::from_vec(shape.clone(), data).unwrap();

        // 1. Identity law: A * 1 = A
        let scaled_one = a.scale(1.0);
        assert_eq!(scaled_one, a);

        // 2. Zero property: A * 0 = zero tensor
        let scaled_zero = a.scale(0.0);
        assert_eq!(scaled_zero, Tensor::zeros(shape.clone()));

        // 3. Scaling by constant c
        let scaled_two = a.scale(2.0);
        let expected = Tensor::from_vec(vec![2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
        assert_eq!(scaled_two, expected);
    }

    #[test]
    fn test_addition_properties() {
        let shape = vec![2, 2];
        let a = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_vec(shape.clone(), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = Tensor::from_vec(shape.clone(), vec![9.0, 10.0, 11.0, 12.0]).unwrap();
        let zeros = Tensor::zeros(shape.clone());

        // 1. Identity: A + 0 = A
        assert_eq!(a.add(&zeros).unwrap(), a);

        // 2. Commutativity: A + B = B + A
        assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());

        // 3. Associativity: (A + B) + C = A + (B + C)
        let left_assoc = a.add(&b).unwrap().add(&c).unwrap();
        let right_assoc = a.add(&b.add(&c).unwrap()).unwrap();
        assert_eq!(left_assoc, right_assoc);

        // 4. Shape mismatch error test
        let mismatched = Tensor::zeros(vec![2, 3]);
        let err = a.add(&mismatched);
        assert_eq!(
            err,
            Err(TensorError::ShapeMismatch {
                left: vec![2, 2],
                right: vec![2, 3]
            })
        );
    }

    #[test]
    fn test_subtraction_properties() {
        let shape = vec![2, 2];
        let a = Tensor::from_vec(shape.clone(), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let b = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let zeros = Tensor::zeros(shape.clone());

        // 1. Identity: A - 0 = A
        assert_eq!(a.sub(&zeros).unwrap(), a);

        // 2. Self-subtraction: A - A = 0
        assert_eq!(a.sub(&a).unwrap(), zeros);

        // 3. Inverse relation: (A + B) - B = A
        let added = a.add(&b).unwrap();
        assert_eq!(added.sub(&b).unwrap(), a);

        // 4. Shape mismatch error test
        let mismatched = Tensor::zeros(vec![4]);
        assert_eq!(
            a.sub(&mismatched),
            Err(TensorError::ShapeMismatch {
                left: vec![2, 2],
                right: vec![4]
            })
        );
    }

    #[test]
    fn test_mul_properties() {
        let shape = vec![2, 2];
        let a = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_vec(shape.clone(), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let ones = Tensor::ones(shape.clone());
        let zeros = Tensor::zeros(shape.clone());

        // 1. Multiplicative Identity: A ⊙ 1 = A
        assert_eq!(a.mul(&ones).unwrap(), a);

        // 2. Zero Property: A ⊙ 0 = 0
        assert_eq!(a.mul(&zeros).unwrap(), zeros);

        // 3. Commutativity: A ⊙ B = B ⊙ A
        assert_eq!(a.mul(&b).unwrap(), b.mul(&a).unwrap());

        // 4. Expected numerical outcome test
        let expected = Tensor::from_vec(shape, vec![5.0, 12.0, 21.0, 32.0]).unwrap();
        assert_eq!(a.mul(&b).unwrap(), expected);

        // 5. Shape mismatch error test
        let mismatched = Tensor::zeros(vec![2, 3]);

        assert_eq!(
            a.mul(&mismatched),
            Err(TensorError::ShapeMismatch {
                left: vec![2, 2],
                right: vec![2, 3]
            })
        );
    }

    #[test]
    fn test_tensor_matmul_2d() {
        // A [2, 3] * B [3, 2] -> C [2, 2]
        // A = [[1, 2, 3], [4, 5, 6]]
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // B = [[7, 8], [9, 1], [2, 3]]
        let b = Tensor::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // C[0, 0] = 1*7 + 2*9 + 3*2 = 31
        // C[0, 1] = 1*8 + 2*1 + 3*3 = 19
        // C[1, 0] = 4*7 + 5*9 + 6*2 = 85
        // C[1, 1] = 4*8 + 5*1 + 6*3 = 55
        assert_eq!(c.as_slice(), &[31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_tensor_matmul_transposed_views() {
        // Verify matmul works on non-contiguous transposed tensor views
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let a_t = a.transpose_2d().unwrap(); // Shape [3, 2]

        let x = vec![10.0, 20.0];
        let x_tensor = Tensor::from_vec(vec![2, 1], x).unwrap();

        let res = a_t.matmul(&x_tensor).unwrap();
        assert_eq!(res.shape(), &[3, 1]);
        // 1*10 + 4*20 = 90
        // 2*10 + 5*20 = 120
        // 3*10 + 6*20 = 150
        assert_eq!(res.as_slice(), &[90.0, 120.0, 150.0]);
    }

    #[test]
    fn test_tensor_matmul_batched_3d() {
        // Batch size = 2, Matmul [2, 2] * [2, 2] -> [2, 2, 2]
        let a = Tensor::from_vec(
            vec![2, 2, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, // Batch 0
                5.0, 6.0, 7.0, 8.0, // Batch 1
            ],
        )
        .unwrap();

        let identity_batch = Tensor::from_vec(
            vec![2, 2, 2],
            vec![
                1.0, 0.0, 0.0, 1.0, // Identity Batch 0
                1.0, 0.0, 0.0, 1.0, // Identity Batch 1
            ],
        )
        .unwrap();

        let res = a.matmul(&identity_batch).unwrap();
        assert_eq!(res.shape(), &[2, 2, 2]);
        assert_eq!(res, a); // A * I = A
    }

    #[test]
    fn test_tensor_matmul_dimension_mismatch() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let b = Tensor::from_vec(vec![4, 2], vec![1.0; 8]).unwrap();

        assert_eq!(
            a.matmul(&b),
            Err(TensorError::MatmulDimensionMismatch {
                left_shape: vec![2, 3],
                right_shape: vec![4, 2],
            })
        );
    }
}

#[cfg(test)]
mod min_max_dim_tests {
    use super::*;

    #[test]
    fn test_min_max_dim_rank_2() {
        // Matrix 2x3:
        // [[10.0,  2.0,  5.0],
        //  [ 4.0, 20.0,  1.0]]
        let data = vec![10.0, 2.0, 5.0, 4.0, 20.0, 1.0];
        let tensor = Tensor::from_vec(vec![2, 3], data).unwrap();

        // 1. Min along axis 0 (columns) -> Shape [3]
        let min_axis_0 = tensor.min_dim(0).unwrap();
        assert_eq!(min_axis_0.shape(), &[3]);
        assert_eq!(min_axis_0.as_slice(), &[4.0, 2.0, 1.0]);

        // 2. Max along axis 0 (columns) -> Shape [3]
        let max_axis_0 = tensor.max_dim(0).unwrap();
        assert_eq!(max_axis_0.shape(), &[3]);
        assert_eq!(max_axis_0.as_slice(), &[10.0, 20.0, 5.0]);

        // 3. Min along axis 1 (rows) -> Shape [2]
        let min_axis_1 = tensor.min_dim(1).unwrap();
        assert_eq!(min_axis_1.shape(), &[2]);
        assert_eq!(min_axis_1.as_slice(), &[2.0, 1.0]);

        // 4. Max along axis 1 (rows) -> Shape [2]
        let max_axis_1 = tensor.max_dim(1).unwrap();
        assert_eq!(max_axis_1.shape(), &[2]);
        assert_eq!(max_axis_1.as_slice(), &[10.0, 20.0]);
    }

    #[test]
    fn test_min_max_dim_rank_1_to_scalar() {
        // Vector [4] -> Reduces to Rank-0 Scalar Tensor []
        let tensor = Tensor::from_vec(vec![4], vec![7.0, -3.0, 12.0, 0.0]).unwrap();

        let min_res = tensor.min_dim(0).unwrap();
        assert_eq!(min_res.shape(), &[]);
        assert_eq!(min_res.as_slice(), &[-3.0]);

        let max_res = tensor.max_dim(0).unwrap();
        assert_eq!(max_res.shape(), &[]);
        assert_eq!(max_res.as_slice(), &[12.0]);
    }

    #[test]
    fn test_min_max_dim_invalid_axis() {
        let tensor: Tensor<f32> = Tensor::zeros(vec![2, 3]);

        // Out-of-bounds axis (axis=2 for rank-2 tensor)
        assert_eq!(
            tensor.min_dim(2),
            Err(TensorError::IndexOutOfBounds {
                dims: 2,
                index: vec![2],
            })
        );
        assert_eq!(
            tensor.max_dim(2),
            Err(TensorError::IndexOutOfBounds {
                dims: 2,
                index: vec![2],
            })
        );
    }

    #[test]
    fn test_sum_dim_rank_2() {
        // Matrix 2x3:
        // [[1.0, 2.0, 3.0],
        //  [4.0, 5.0, 6.0]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(vec![2, 3], data).unwrap();

        // Sum along axis 0 (columns) -> Shape [3]
        // [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0]
        let sum_0 = tensor.sum_dim(0).unwrap();
        assert_eq!(sum_0.shape(), &[3]);
        assert_eq!(sum_0.as_slice(), &[5.0, 7.0, 9.0]);

        // Sum along axis 1 (rows) -> Shape [2]
        // [1+2+3, 4+5+6] = [6.0, 15.0]
        let sum_1 = tensor.sum_dim(1).unwrap();
        assert_eq!(sum_1.shape(), &[2]);
        assert_eq!(sum_1.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn test_sum_dim_rank_1_to_scalar() {
        // Vector [4] -> Reduces to Rank-0 Scalar Tensor []
        let tensor = Tensor::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        let sum_res = tensor.sum_dim(0).unwrap();
        assert_eq!(sum_res.shape(), &[]);
        assert_eq!(sum_res.as_slice(), &[100.0]);
    }

    #[test]
    fn test_sum_dim_invalid_axis() {
        let tensor: Tensor<f32> = Tensor::zeros(vec![2, 3]);

        assert_eq!(
            tensor.sum_dim(2),
            Err(TensorError::IndexOutOfBounds {
                dims: 2,
                index: vec![2],
            })
        );
    }
}

#[cfg(test)]
mod contiguity_and_permutation_tests {
    use super::*;

    #[test]
    fn test_is_contiguous_and_reshape_contract() {
        // Standard initialized tensor is contiguous
        let mut t = Tensor::from_vec(vec![2, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap();
        assert!(t.is_contiguous());

        // Reshape succeeds on contiguous tensor
        assert!(t.reshape(vec![6, 4]).is_ok());
        assert_eq!(t.shape(), &[6, 4]);

        // Permute to non-contiguous shape [4, 2, 3] with strides [1, 12, 4]
        let permuted = t.permute(&[1, 0]).unwrap(); // [6, 4] permuted to [4, 6]
        assert!(!permuted.is_contiguous());

        // Reshape fails on non-contiguous tensor
        let mut permuted_clone = permuted.clone();
        assert_eq!(
            permuted_clone.reshape(vec![24]),
            Err(TensorError::NonContiguousReshape)
        );

        // Identity permutation preserves contiguity
        let identity_perm = t.permute(&[0, 1]).unwrap();
        assert!(identity_perm.is_contiguous());

        // Squeeze and unsqueeze preserve contiguity expectations
        let sq = Tensor::from_vec(vec![1, 3, 1, 4], (0..12).map(|i| i as f32).collect())
            .unwrap()
            .squeeze();
        assert!(sq.is_contiguous());

        let unsq = t.unsqueeze(1).unwrap();
        assert!(unsq.is_contiguous());
    }

    #[test]
    fn test_permute_logical_matrix_indexing() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::from_vec(vec![2, 3], data).unwrap();

        // Permute to [3, 2]
        let a_t = a.permute(&[1, 0]).unwrap();
        assert_eq!(a_t.shape(), &[3, 2]);
        assert_eq!(a_t.strides(), &[1, 3]);

        // Expected logical matrix layout:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        assert_eq!(a_t.get(&[0, 0]), Some(&1.0));
        assert_eq!(a_t.get(&[0, 1]), Some(&4.0));
        assert_eq!(a_t.get(&[1, 0]), Some(&2.0));
        assert_eq!(a_t.get(&[1, 1]), Some(&5.0));
        assert_eq!(a_t.get(&[2, 0]), Some(&3.0));
        assert_eq!(a_t.get(&[2, 1]), Some(&6.0));
    }

    #[test]
    fn test_general_transpose() {
        // Rank-3 tensor [2, 3, 4] -> transpose() -> Rank-3 tensor [4, 3, 2]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_vec(vec![2, 3, 4], data).unwrap();

        let x_t = x.transpose();
        assert_eq!(x_t.shape(), &[4, 3, 2]);
        assert_eq!(x_t.strides(), &[1, 4, 12]);

        // x[1, 2, 3] (offset 23) -> x_t[3, 2, 1] (offset 23)
        assert_eq!(x.get(&[1, 2, 3]), Some(&23.0));
        assert_eq!(x_t.get(&[3, 2, 1]), Some(&23.0));
    }

    #[test]
    fn test_permute_error_cases() {
        let x = Tensor::<f32>::zeros(vec![2, 3, 4]);

        // Wrong axis count
        assert!(x.permute(&[1, 0]).is_err());
        // Duplicate axes
        assert!(x.permute(&[0, 1, 1]).is_err());
        // Axis out of bounds
        assert!(x.permute(&[0, 1, 3]).is_err());
    }
}

#[cfg(test)]
mod non_contiguous_architecture_tests {
    use super::*;

    /// Helper to generate a matrix and its transposed (non-contiguous) view
    fn get_matrix_and_transposed_view() -> (Tensor<f32>, Tensor<f32>) {
        // [2, 3] contiguous
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // [3, 2] non-contiguous (strides [1, 3])
        // Logically:
        // [1, 4]
        // [2, 5]
        // [3, 6]
        let a_t = a.transpose_2d().unwrap();

        (a, a_t)
    }

    #[test]
    fn test_non_contiguous_elementwise() {
        let (_, a_t) = get_matrix_and_transposed_view();

        // Aᵀ + Aᵀ
        let added = a_t.add(&a_t).unwrap();
        assert_eq!(added.shape(), &[3, 2]);
        assert_eq!(added.get(&[0, 0]), Some(&2.0)); // 1 + 1
        assert_eq!(added.get(&[0, 1]), Some(&8.0)); // 4 + 4
        assert_eq!(added.get(&[2, 1]), Some(&12.0)); // 6 + 6

        // Aᵀ * Aᵀ
        let multiplied = a_t.mul(&a_t).unwrap();
        assert_eq!(multiplied.get(&[1, 1]), Some(&25.0)); // 5 * 5
    }

    #[test]
    fn test_non_contiguous_reductions() {
        let (_, a_t) = get_matrix_and_transposed_view();

        // Sum along axis 1 (the rows of Aᵀ, which were the columns of A)
        // Logically summing [1, 4], [2, 5], [3, 6]
        let sum_dim_1 = a_t.sum_dim(1).unwrap();
        assert_eq!(sum_dim_1.shape(), &[3]);
        assert_eq!(sum_dim_1.get(&[0]), Some(&5.0));  // 1+4
        assert_eq!(sum_dim_1.get(&[1]), Some(&7.0));  // 2+5
        assert_eq!(sum_dim_1.get(&[2]), Some(&9.0));  // 3+6
    }

    #[test]
    fn test_non_contiguous_matmul() {
        // Aᵀ is [3, 2] non-contiguous. B is [2, 2] contiguous.
        let (_, a_t) = get_matrix_and_transposed_view();
        let b = Tensor::from_vec(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        // [3, 2] × [2, 2] -> [3, 2]
        let c = a_t.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[3, 2]);

        // Row 0 of Aᵀ is [1, 4]
        // Col 0 of B is [10, 30]
        // 1*10 + 4*30 = 130
        assert_eq!(c.get(&[0, 0]), Some(&130.0));
    }

    #[test]
    fn test_dot_and_outer() {
        let v1 = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();

        let dot_res = v1.dot(&v2).unwrap();
        assert_eq!(dot_res.shape(), &[]);
        assert_eq!(dot_res.as_slice(), &[32.0]); // 4 + 10 + 18

        let v3 = Tensor::from_vec(vec![2], vec![10.0, 20.0]).unwrap();
        let outer_res = v1.outer(&v3).unwrap();
        assert_eq!(outer_res.shape(), &[3, 2]);
        // [[10, 20], [20, 40], [30, 60]]
        assert_eq!(outer_res.as_slice(), &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }
}

#[cfg(test)]
mod additional_math_and_reductions_tests {
    use super::*;

    #[test]
    fn test_elementwise_math_real() {
        let a = Tensor::from_vec(vec![2], vec![-1.0, 4.0]).unwrap();

        let neg = a.neg();
        assert_eq!(neg.as_slice(), &[1.0, -4.0]);

        let abs = a.abs();
        assert_eq!(abs.as_slice(), &[1.0, 4.0]);

        let sqrt = abs.sqrt();
        assert_eq!(sqrt.as_slice(), &[1.0, 2.0]);

        let pow = abs.pow(3.0);
        assert_eq!(pow.as_slice(), &[1.0, 64.0]);

        let ln = Tensor::from_vec(vec![1], vec![1.0_f32.exp()]).unwrap().ln();
        // Almost equal due to f32 precision
        assert!((ln.as_slice()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_prod_dim() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let prod_0 = a.prod_dim(0).unwrap(); // Columns
        assert_eq!(prod_0.shape(), &[3]);
        assert_eq!(prod_0.as_slice(), &[4.0, 10.0, 18.0]);

        let prod_1 = a.prod_dim(1).unwrap(); // Rows
        assert_eq!(prod_1.shape(), &[2]);
        assert_eq!(prod_1.as_slice(), &[6.0, 120.0]); // 1*2*3, 4*5*6
    }

    #[test]
    fn test_argmin_argmax_dim() {
        // [[10, 50, 30],
        //  [60, 20, 40]]
        let a = Tensor::from_vec(vec![2, 3], vec![10.0, 50.0, 30.0, 60.0, 20.0, 40.0]).unwrap();

        // Across rows (axis 1):
        // Row 0 min is 10 (idx 0), max is 50 (idx 1)
        // Row 1 min is 20 (idx 1), max is 60 (idx 0)
        let argmin_1 = a.argmin_dim(1).unwrap();
        assert_eq!(argmin_1.shape(), &[2]);
        assert_eq!(argmin_1.as_slice(), &[0, 1]);

        let argmax_1 = a.argmax_dim(1).unwrap();
        assert_eq!(argmax_1.shape(), &[2]);
        assert_eq!(argmax_1.as_slice(), &[1, 0]);

        // Across cols (axis 0):
        // Col 0: 10 vs 60 -> min idx 0, max idx 1
        // Col 1: 50 vs 20 -> min idx 1, max idx 0
        // Col 2: 30 vs 40 -> min idx 0, max idx 1
        let argmin_0 = a.argmin_dim(0).unwrap();
        assert_eq!(argmin_0.as_slice(), &[0, 1, 0]);

        let argmax_0 = a.argmax_dim(0).unwrap();
        assert_eq!(argmax_0.as_slice(), &[1, 0, 1]);
    }
}

#[cfg(test)]
mod ml_primitives_tests {
    use super::*;

    #[test]
    fn test_softmax_integration() {
        // Explicitly declare Tensor::<f32>
        let a = Tensor::<f32>::from_vec(
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0,
                1.0, 1.0, 1.0
            ]
        ).unwrap();

        let sm = a.softmax(1).unwrap();
        assert_eq!(sm.shape(), &[2, 3]);

        let row_sums = sm.sum_dim(1).unwrap();
        assert_eq!(row_sums.shape(), &[2]);

        // Add _f32 to the literals so Rust knows exactly what type to call .abs() on
        assert!((row_sums.as_slice()[0] - 1.0_f32).abs() < 1e-6);
        assert!((row_sums.as_slice()[1] - 1.0_f32).abs() < 1e-6);

        assert!((sm.get(&[1, 0]).unwrap() - 0.333333_f32).abs() < 1e-5);
    }

    #[test]
    fn test_relu_and_leaky_relu() {
        let a = Tensor::from_vec(vec![3], vec![-5.0, 0.0, 5.0]).unwrap();

        let r = a.relu();
        assert_eq!(r.as_slice(), &[0.0, 0.0, 5.0]);

        let lr = a.leaky_relu(0.1);
        assert_eq!(lr.as_slice(), &[-0.5, 0.0, 5.0]);
    }
}
