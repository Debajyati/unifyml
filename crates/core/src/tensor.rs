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

    /// Performs element-wise addition between two tensors of identical shape.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Performs element-wise subtraction between two tensors of identical shape.
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Performs element-wise multiplication (Hadamard product) between two tensors of identical shape.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stride_calculation() {
        let shape = vec![2, 3, 4];
        let strides = calculate_stride(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }
}
