use crate::Scalar;
use crate::TensorError;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: Scalar> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Helper to compute total number of elements from a shape.
fn compute_total_elements(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1; // Scalar tensor (rank-0)
    }
    shape.iter().product()
}

/// Helper to compute row-major contiguous strides from a shape.
fn calculate_stride(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let n = shape.len();
    let mut stride: Vec<usize> = vec![1; n];

    for i in (1..n).rev() {
        stride[i - 1] = stride[i] * shape[i];
    }

    stride
}

/// Determines the target output shape when broadcasting two shapes right-to-left.
/// Returns TensorError::ShapeMismatch if the shapes are incompatible.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank_a = a.len();
    let rank_b = b.len();
    let max_rank = rank_a.max(rank_b);

    let mut target_shape = vec![0; max_rank];

    for i in 0..max_rank {
        // Index from right to left
        let dim_a = if i < rank_a { a[rank_a - 1 - i] } else { 1 };
        let dim_b = if i < rank_b { b[rank_b - 1 - i] } else { 1 };

        if dim_a == dim_b {
            target_shape[max_rank - 1 - i] = dim_a;
        } else if dim_a == 1 {
            target_shape[max_rank - 1 - i] = dim_b;
        } else if dim_b == 1 {
            target_shape[max_rank - 1 - i] = dim_a;
        } else {
            return Err(TensorError::ShapeMismatch {
                left: a.to_vec(),
                right: b.to_vec(),
            });
        }
    }

    Ok(target_shape)
}

/// Adjusts a tensor's original shape and strides to match a broadcasted target shape.
/// Axes with size 1 that are expanded receive a stride of 0 (virtual zero-copy broadcasting).
fn broadcast_strides(shape: &[usize], strides: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let target_rank = target_shape.len();
    let pad = target_rank - rank;

    let mut target_strides = vec![0; target_rank];

    for i in 0..target_rank {
        if i < pad {
            // Prepended dimensions have stride 0
            target_strides[i] = 0;
        } else {
            let orig_idx = i - pad;
            if shape[orig_idx] == 1 && target_shape[i] > 1 {
                // Expanded singleton dimension gets stride 0
                target_strides[i] = 0;
            } else {
                target_strides[i] = strides[orig_idx];
            }
        }
    }

    target_strides
}

impl<T: Scalar> Tensor<T> {
    /// Create a tensor from a shape and flat data vector with validation.
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Self, TensorError> {
        let expected_len = compute_total_elements(&shape);
        if data.len() != expected_len {
            return Err(TensorError::DataLengthMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }
        let strides = calculate_stride(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Construct a Tensor by copying elements from a slice given a shape.
    pub fn from_slice(shape: Vec<usize>, slice: &[T]) -> Result<Self, TensorError> {
        let expected_len = compute_total_elements(&shape);
        if slice.len() != expected_len {
            return Err(TensorError::DataLengthMismatch {
                expected: expected_len,
                actual: slice.len(),
            });
        }
        let strides = calculate_stride(&shape);
        Ok(Self {
            data: slice.to_vec(),
            shape,
            strides,
        })
    }

    /// Expose read-only access to underlying contiguous memory buffer.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Expose mutable access to underlying contiguous memory buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Create a tensor filled with zeros for a given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = compute_total_elements(&shape);
        let data = vec![T::zero(); len];
        let strides = calculate_stride(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with ones for a given shape.
    pub fn ones(shape: Vec<usize>) -> Self {
        let len = compute_total_elements(&shape);
        let data = vec![T::one(); len];
        let strides = calculate_stride(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Maps an N-dimensional index to a flat linear offset using strides.
    pub fn calculate_offset(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds {
                dims: self.shape.len(),
                index: indices.to_vec(),
            });
        }

        let mut offset = 0;
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    dims: self.shape.len(),
                    index: indices.to_vec(),
                });
            }
            offset += idx * self.strides[i];
        }
        Ok(offset)
    }

    /// Safely retrieve a reference to an element given multi-dimensional indices.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.calculate_offset(indices).ok()?;
        self.data.get(offset)
    }

    /// Safely retrieve a mutable reference to an element given multi-dimensional indices.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = self.calculate_offset(indices).ok()?;
        self.data.get_mut(offset)
    }

    /// Zero-copy reshape: alters shape and recalculates strides without moving underlying data.
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), TensorError> {
        let new_len = compute_total_elements(&new_shape);
        if new_len != self.len() {
            return Err(TensorError::ReshapeMismatch {
                old_len: self.len(),
                new_len,
            });
        }
        self.strides = calculate_stride(&new_shape);
        self.shape = new_shape;
        Ok(())
    }
    /// Multiplies every element in the tensor by a scalar factor.
    pub fn scale(&self, factor: T) -> Self {
        let scaled_data: Vec<T> = self.data.iter().map(|&x| x * factor).collect();
        Self {
            data: scaled_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Broadcasted element-wise addition between two tensors.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        let target_shape = broadcast_shape(self.shape(), other.shape())?;

        // Fast-path: if shapes match identically, compute via fast contiguous iteration
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            return Ok(Self {
                data: result_data,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
            });
        }

        // Virtual zero-copy broadcasted path via stride-0 offset calculations
        let strides_a = broadcast_strides(self.shape(), self.strides(), &target_shape);
        let strides_b = broadcast_strides(other.shape(), other.strides(), &target_shape);

        let total_elements: usize = target_shape.iter().product();
        let target_strides = calculate_stride(&target_shape);
        let mut result_data = Vec::with_capacity(total_elements);

        for linear_idx in 0..total_elements {
            // Unravel flat linear_idx into multi-dimensional index according to target_shape
            let mut offset_a = 0;
            let mut offset_b = 0;
            let mut temp_idx = linear_idx;

            for d in 0..target_shape.len() {
                let coord = temp_idx / target_strides[d];
                temp_idx %= target_strides[d];

                offset_a += coord * strides_a[d];
                offset_b += coord * strides_b[d];
            }

            let val_a = self.data[offset_a];
            let val_b = other.data[offset_b];
            result_data.push(val_a + val_b);
        }

        Ok(Self {
            data: result_data,
            strides: target_strides,
            shape: target_shape,
        })
    }

    /// Broadcasted element-wise subtraction between two tensors.
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        let target_shape = broadcast_shape(self.shape(), other.shape())?;

        // Fast path for identical shapes (contiguous slice iteration)
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect();

            return Ok(Self {
                data: result_data,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
            });
        }

        // Virtual zero-copy broadcasted path via stride-0 offset calculation
        let strides_a = broadcast_strides(self.shape(), self.strides(), &target_shape);
        let strides_b = broadcast_strides(other.shape(), other.strides(), &target_shape);

        let total_elements: usize = target_shape.iter().product();
        let target_strides = calculate_stride(&target_shape);
        let mut result_data = Vec::with_capacity(total_elements);

        for linear_idx in 0..total_elements {
            let mut offset_a = 0;
            let mut offset_b = 0;
            let mut temp_idx = linear_idx;

            for d in 0..target_shape.len() {
                let coord = temp_idx / target_strides[d];
                temp_idx %= target_strides[d];

                offset_a += coord * strides_a[d];
                offset_b += coord * strides_b[d];
            }

            let val_a = self.data[offset_a];
            let val_b = other.data[offset_b];
            result_data.push(val_a - val_b);
        }

        Ok(Self {
            data: result_data,
            strides: target_strides,
            shape: target_shape,
        })
    }

    /// Broadcasted element-wise multiplication between two tensors.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        let target_shape = broadcast_shape(self.shape(), other.shape())?;

        // Fast path for identical shapes (contiguous slice iteration)
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect();

            return Ok(Self {
                data: result_data,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
            });
        }

        // Virtual zero-copy broadcasted path via stride-0 offset calculation
        let strides_a = broadcast_strides(self.shape(), self.strides(), &target_shape);
        let strides_b = broadcast_strides(other.shape(), other.strides(), &target_shape);

        let total_elements: usize = target_shape.iter().product();
        let target_strides = calculate_stride(&target_shape);
        let mut result_data = Vec::with_capacity(total_elements);

        for linear_idx in 0..total_elements {
            let mut offset_a = 0;
            let mut offset_b = 0;
            let mut temp_idx = linear_idx;

            for d in 0..target_shape.len() {
                let coord = temp_idx / target_strides[d];
                temp_idx %= target_strides[d];

                offset_a += coord * strides_a[d];
                offset_b += coord * strides_b[d];
            }

            let val_a = self.data[offset_a];
            let val_b = other.data[offset_b];
            result_data.push(val_a * val_b);
        }

        Ok(Self {
            data: result_data,
            strides: target_strides,
            shape: target_shape,
        })
    }
}

impl<T: Scalar + PartialOrd> Tensor<T> {
    /// Reduces the tensor along a specified axis by taking the minimum value.
    pub fn min_dim(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.ndim() {
            return Err(TensorError::IndexOutOfBounds {
                dims: self.ndim(),
                index: vec![axis],
            });
        }

        let mut output_shape = Vec::with_capacity(self.ndim().saturating_sub(1));
        for (i, &d) in self.shape.iter().enumerate() {
            if i != axis {
                output_shape.push(d);
            }
        }

        let total_output_elements = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };

        let output_strides = calculate_stride(&output_shape);
        let mut result_data = Vec::with_capacity(total_output_elements);

        let axis_size = self.shape[axis];
        let axis_stride = self.strides[axis];

        for linear_out_idx in 0..total_output_elements {
            // Unravel output index and inject placeholder 0 at target axis
            let mut base_in_coords = Vec::with_capacity(self.ndim());
            let mut temp_idx = linear_out_idx;

            for &item in output_strides.iter().take(output_shape.len()) {
                if base_in_coords.len() == axis {
                    base_in_coords.push(0);
                }
                let coord = temp_idx / item;
                temp_idx %= item;
                base_in_coords.push(coord);
            }
            if base_in_coords.len() == axis {
                base_in_coords.push(0);
            }

            let base_offset = self.calculate_offset(&base_in_coords)?;

            if axis_size == 0 {
                return Err(TensorError::InvalidShape {
                    reason: "Cannot perform reduction on dimension of size 0".to_string(),
                });
            }

            // Iterate along axis and accumulate minimum
            let mut min_val = self.data[base_offset];
            for r in 1..axis_size {
                let val = self.data[base_offset + r * axis_stride];
                if val < min_val {
                    min_val = val;
                }
            }

            result_data.push(min_val);
        }

        Ok(Self {
            data: result_data,
            strides: output_strides,
            shape: output_shape,
        })
    }

    /// Reduces the tensor along a specified axis by taking the maximum value.
    pub fn max_dim(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.ndim() {
            return Err(TensorError::IndexOutOfBounds {
                dims: self.ndim(),
                index: vec![axis],
            });
        }

        let mut output_shape = Vec::with_capacity(self.ndim().saturating_sub(1));
        for (i, &d) in self.shape.iter().enumerate() {
            if i != axis {
                output_shape.push(d);
            }
        }

        let total_output_elements = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };

        let output_strides = calculate_stride(&output_shape);
        let mut result_data = Vec::with_capacity(total_output_elements);

        let axis_size = self.shape[axis];
        let axis_stride = self.strides[axis];

        for linear_out_idx in 0..total_output_elements {
            // Unravel output index and inject placeholder 0 at target axis
            let mut base_in_coords = Vec::with_capacity(self.ndim());
            let mut temp_idx = linear_out_idx;

            for &item in output_strides.iter().take(output_shape.len()) {
                if base_in_coords.len() == axis {
                    base_in_coords.push(0);
                }
                let coord = temp_idx / item;
                temp_idx %= item;
                base_in_coords.push(coord);
            }
            if base_in_coords.len() == axis {
                base_in_coords.push(0);
            }

            let base_offset = self.calculate_offset(&base_in_coords)?;

            if axis_size == 0 {
                return Err(TensorError::InvalidShape {
                    reason: "Cannot perform reduction on dimension of size 0".to_string(),
                });
            }

            // Iterate along axis and accumulate maximum
            let mut max_val = self.data[base_offset];
            for r in 1..axis_size {
                let val = self.data[base_offset + r * axis_stride];
                if val > max_val {
                    max_val = val;
                }
            }

            result_data.push(max_val);
        }

        Ok(Self {
            data: result_data,
            strides: output_strides,
            shape: output_shape,
        })
    }

    /// Reduces the tensor along a specified axis by summing all values.
    pub fn sum_dim(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.ndim() {
            return Err(TensorError::IndexOutOfBounds {
                dims: self.ndim(),
                index: vec![axis],
            });
        }

        let mut output_shape = Vec::with_capacity(self.ndim().saturating_sub(1));
        for (i, &d) in self.shape.iter().enumerate() {
            if i != axis {
                output_shape.push(d);
            }
        }

        let total_output_elements = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };

        let output_strides = calculate_stride(&output_shape);
        let mut result_data = Vec::with_capacity(total_output_elements);

        let axis_size = self.shape[axis];
        let axis_stride = self.strides[axis];

        for linear_out_idx in 0..total_output_elements {
            // Unravel output index and inject placeholder 0 at target axis
            let mut base_in_coords = Vec::with_capacity(self.ndim());
            let mut temp_idx = linear_out_idx;

            for &item in output_strides.iter().take(output_shape.len()) {
                if base_in_coords.len() == axis {
                    base_in_coords.push(0);
                }
                let coord = temp_idx / item;
                temp_idx %= item;
                base_in_coords.push(coord);
            }
            if base_in_coords.len() == axis {
                base_in_coords.push(0);
            }

            let base_offset = self.calculate_offset(&base_in_coords)?;

            if axis_size == 0 {
                result_data.push(T::zero());
                continue;
            }

            // Iterate along target axis and accumulate sum
            let mut sum_val = T::zero();
            for r in 0..axis_size {
                sum_val = sum_val + self.data[base_offset + r * axis_stride];
            }

            result_data.push(sum_val);
        }

        Ok(Self {
            data: result_data,
            strides: output_strides,
            shape: output_shape,
        })
    }
}

#[cfg(test)]
mod internal_test {
    use super::*;
    #[test]
    fn test_stride_calculation() {
        let shape = vec![2, 3, 4];
        let strides = calculate_stride(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }
}

#[cfg(test)]
mod broadcasting_tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_resolution() {
        // Equal shapes
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Ok(vec![2, 3]));

        // Right alignment (padding missing rank on left)
        assert_eq!(broadcast_shape(&[2, 3, 4], &[3, 4]), Ok(vec![2, 3, 4]));

        // Singleton dimensions expansion
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Ok(vec![3, 4]));
        assert_eq!(broadcast_shape(&[2, 3, 4], &[2, 1, 4]), Ok(vec![2, 3, 4]));

        // Rank-0 Scalar Tensor ([])
        assert_eq!(broadcast_shape(&[], &[2, 3]), Ok(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 3], &[]), Ok(vec![2, 3]));

        // Incompatible shapes
        assert_eq!(
            broadcast_shape(&[2, 3], &[4, 3]),
            Err(TensorError::ShapeMismatch {
                left: vec![2, 3],
                right: vec![4, 3]
            })
        );
        assert_eq!(
            broadcast_shape(&[2, 3], &[2, 4]),
            Err(TensorError::ShapeMismatch {
                left: vec![2, 3],
                right: vec![2, 4]
            })
        );
    }

    #[test]
    fn test_broadcasted_addition() {
        // 1. Matrix [3, 1] + Row Vector [1, 4] -> Matrix [3, 4]
        // A = [[1], [2], [3]]
        let a = Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        // B = [[10, 20, 30, 40]]
        let b = Tensor::from_vec(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[3, 4]);

        // Row 0: 1+10, 1+20, 1+30, 1+40 = 11, 21, 31, 41
        assert_eq!(c.get(&[0, 0]), Some(&11.0));
        assert_eq!(c.get(&[0, 3]), Some(&41.0));
        // Row 2: 3+10, 3+20, 3+30, 3+40 = 13, 23, 33, 43
        assert_eq!(c.get(&[2, 0]), Some(&13.0));
        assert_eq!(c.get(&[2, 3]), Some(&43.0));

        // 2. Rank-0 Scalar Tensor ([]) + Tensor [2, 3]
        let scalar = Tensor::from_vec(vec![], vec![5.0]).unwrap();
        let mat = Tensor::ones(vec![2, 3]);

        let res = mat.add(&scalar).unwrap();
        assert_eq!(res.shape(), &[2, 3]);
        assert_eq!(res.as_slice(), &[6.0, 6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_broadcasted_subtraction() {
        // [2, 3] - [1, 3] -> [2, 3]
        let a = Tensor::from_vec(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
        let b = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();

        let c = a.sub(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        // Row 0: 10-1, 20-2, 30-3 = [9, 18, 27]
        // Row 1: 40-1, 50-2, 60-3 = [39, 48, 57]
        assert_eq!(c.as_slice(), &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    }

    #[test]
    fn test_broadcasted_multiplication() {
        // [3, 1] * [1, 2] -> [3, 2]
        let a = Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::from_vec(vec![1, 2], vec![10.0, 100.0]).unwrap();

        let c = a.mul(&b).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        // Row 0: 1*10, 1*100 = [10, 100]
        // Row 1: 2*10, 2*100 = [20, 200]
        // Row 2: 3*10, 3*100 = [30, 300]
        assert_eq!(c.as_slice(), &[10.0, 100.0, 20.0, 200.0, 30.0, 300.0]);
    }
}
