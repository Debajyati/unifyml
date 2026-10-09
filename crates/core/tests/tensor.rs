#[cfg(test)]
mod tests {
    use unifyml::Tensor;
    use unifyml::TensorError;
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
    use unifyml::Tensor;
    use unifyml::TensorError;

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
        assert_eq!(
            a.mul(&b).unwrap(),
            b.mul(&a).unwrap()
        );

        // 4. Expected numerical outcome test
        let expected = Tensor::from_vec(shape, vec![5.0, 12.0, 21.0, 32.0]).unwrap();
        assert_eq!(a.mul(&b).unwrap(), expected);

        // 5. Shape mismatch error test
        let mismatched = Tensor::zeros(vec![2, 2, 1]);
        assert_eq!(
            a.mul(&mismatched),
            Err(TensorError::ShapeMismatch {
                left: vec![2, 2],
                right: vec![2, 2, 1]
            })
        );
    }
}
