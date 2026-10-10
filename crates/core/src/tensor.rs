use crate::TensorError;
use crate::{Real, Scalar};

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

    /// Returns true if the tensor's memory layout is C-contiguous (row-major).
    ///
    /// A tensor is contiguous if its strides match the standard layout where moving along
    /// dimension `k` by 1 steps over `prod(shape[k+1..])` elements in memory.
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let mut expected_stride = 1;
        for i in (0..self.ndim()).rev() {
            // Ignore dimensions of size 1, as any stride is technically valid for length 1
            if self.shape[i] != 1 {
                if self.strides[i] != expected_stride {
                    return false;
                }
                expected_stride *= self.shape[i];
            }
        }

        true
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// Currently requires the tensor to be contiguous. Reshaping a non-contiguous tensor
    /// without contiguous memory layout will return `Err(TensorError::NonContiguousReshape)`.
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NonContiguousReshape);
        }

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

    /// Broadcasted element-wise division between two tensors.
    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        let target_shape = broadcast_shape(self.shape(), other.shape())?;

        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a / b)
                .collect();

            return Ok(Self {
                data: result_data,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
            });
        }

        let strides_a = broadcast_strides(self.shape(), self.strides(), &target_shape);
        let strides_b = broadcast_strides(other.shape(), other.strides(), &target_shape);
        let target_strides = calculate_stride(&target_shape);
        let total_elements: usize = target_shape.iter().product();
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
            result_data.push(self.data[offset_a] / other.data[offset_b]);
        }

        Ok(Self {
            data: result_data,
            strides: target_strides,
            shape: target_shape,
        })
    }

    /// Performs 2D or batched N-dimensional matrix multiplication.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        let ndim_a = self.ndim();
        let ndim_b = other.ndim();

        if ndim_a < 2 || ndim_b < 2 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "matmul requires at least rank-2 tensors, got rank-{} and rank-{}",
                    ndim_a, ndim_b
                ),
            });
        }

        let m = self.shape[ndim_a - 2];
        let k_a = self.shape[ndim_a - 1];
        let k_b = other.shape[ndim_b - 2];
        let n = other.shape[ndim_b - 1];

        // Validate inner matrix dimension match (K_a == K_b)
        if k_a != k_b {
            return Err(TensorError::MatmulDimensionMismatch {
                left_shape: self.shape.clone(),
                right_shape: other.shape.clone(),
            });
        }

        // Extract batch shapes (all axes except the last 2)
        let batch_shape_a = &self.shape[..ndim_a - 2];
        let batch_shape_b = &other.shape[..ndim_b - 2];

        // Resolve broadcasted batch shape
        let batch_target_shape = broadcast_shape(batch_shape_a, batch_shape_b)?;

        // Construct final output shape: [batch_dims..., M, N]
        let mut output_shape = batch_target_shape.clone();
        output_shape.push(m);
        output_shape.push(n);

        let output_strides = calculate_stride(&output_shape);
        let total_output_elements = output_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_output_elements);

        // Virtual stride alignment for batch dimensions
        let batch_strides_a = broadcast_strides(
            batch_shape_a,
            &self.strides[..ndim_a - 2],
            &batch_target_shape,
        );
        let batch_strides_b = broadcast_strides(
            batch_shape_b,
            &other.strides[..ndim_b - 2],
            &batch_target_shape,
        );

        // Matrix inner strides
        let stride_a_m = self.strides[ndim_a - 2];
        let stride_a_k = self.strides[ndim_a - 1];
        let stride_b_k = other.strides[ndim_b - 2];
        let stride_b_n = other.strides[ndim_b - 1];

        let batch_count: usize = batch_target_shape.iter().product();
        let batch_target_strides = calculate_stride(&batch_target_shape);

        // Compute matrix multiplication for every batch
        for b in 0..batch_count {
            // Compute base batch offset for A and B
            let mut base_offset_a = 0;
            let mut base_offset_b = 0;

            if !batch_target_shape.is_empty() {
                let mut temp_idx = b;
                for d in 0..batch_target_shape.len() {
                    let coord = temp_idx / batch_target_strides[d];
                    temp_idx %= batch_target_strides[d];

                    base_offset_a += coord * batch_strides_a[d];
                    base_offset_b += coord * batch_strides_b[d];
                }
            }

            // Perform 2D matmul for current batch slice
            for i in 0..m {
                let row_offset_a = base_offset_a + i * stride_a_m;
                for j in 0..n {
                    let col_offset_b = base_offset_b + j * stride_b_n;

                    let mut sum = T::zero();
                    for k in 0..k_a {
                        let offset_a = row_offset_a + k * stride_a_k;
                        let offset_b = col_offset_b + k * stride_b_k;

                        sum = sum + self.data[offset_a] * other.data[offset_b];
                    }
                    result_data.push(sum);
                }
            }
        }

        Ok(Self {
            data: result_data,
            strides: output_strides,
            shape: output_shape,
        })
    }
}

impl<T: Real> Tensor<T> {
    /// Reduces the tensor along a specified axis by taking the minimum value.
    pub fn min_dim(&self, axis: usize) -> Result<Self, TensorError> {
        self.reduce_dim(axis, |_step, acc, val| if val < acc { val } else { acc })
    }

    /// Reduces the tensor along a specified axis by taking the maximum value.
    pub fn max_dim(&self, axis: usize) -> Result<Self, TensorError> {
        self.reduce_dim(axis, |_step, acc, val| if val > acc { val } else { acc })
    }

    /// Reduces the tensor along a specified axis by summing all values.
    pub fn sum_dim(&self, axis: usize) -> Result<Self, TensorError> {
        self.reduce_dim(axis, |_step, acc, val| acc + val)
    }

    /// Computes the dot product (inner product) of two rank-1 tensors.
    /// Returns a rank-0 scalar tensor.
    pub fn dot(&self, other: &Self) -> Result<Self, TensorError> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "dot() requires rank-1 tensors, got rank-{} and rank-{}",
                    self.ndim(),
                    other.ndim()
                ),
            });
        }
        if self.shape[0] != other.shape[0] {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let len = self.shape[0];
        let stride_a = self.strides[0];
        let stride_b = other.strides[0];

        let mut sum = T::zero();
        for i in 0..len {
            let val_a = self.data[i * stride_a];
            let val_b = other.data[i * stride_b];
            sum = sum + val_a * val_b;
        }

        Ok(Self {
            data: vec![sum],
            shape: vec![],
            strides: vec![],
        })
    }

    /// Computes the outer product of two rank-1 tensors.
    /// Returns a rank-2 tensor (matrix).
    pub fn outer(&self, other: &Self) -> Result<Self, TensorError> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "outer() requires rank-1 tensors, got rank-{} and rank-{}",
                    self.ndim(),
                    other.ndim()
                ),
            });
        }

        let m = self.shape[0];
        let n = other.shape[0];
        let stride_a = self.strides[0];
        let stride_b = other.strides[0];

        let mut result_data = Vec::with_capacity(m * n);

        for i in 0..m {
            let val_a = self.data[i * stride_a];
            for j in 0..n {
                let val_b = other.data[j * stride_b];
                result_data.push(val_a * val_b);
            }
        }

        Ok(Self {
            data: result_data,
            shape: vec![m, n],
            strides: vec![n, 1],
        })
    }

    /// Reduces the tensor along a specified axis by computing the arithmetic mean.
    pub fn mean_dim(&self, axis: usize) -> Result<Self, TensorError> {
        let axis_size = self.shape.get(axis).copied().unwrap_or(0);
        if axis_size == 0 {
            return Err(TensorError::InvalidShape {
                reason: "Cannot calculate mean of dimension of size 0".to_string(),
            });
        }

        let summed = self.sum_dim(axis)?;
        let count_scalar = T::from_usize(axis_size);

        let result_data = summed
            .data
            .into_iter()
            .map(|val| val / count_scalar)
            .collect();

        Ok(Self {
            data: result_data,
            strides: summed.strides,
            shape: summed.shape,
        })
    }

    /// Reduces the tensor along an axis by computing the product of its elements.
    pub fn prod_dim(&self, axis: usize) -> Result<Self, TensorError> {
        // Our reduce_dim initializes with the first element and folds the rest.
        // This is mathematically identical to starting with T::one() and multiplying.
        self.reduce_dim(axis, |_step, acc, val| acc * val)
    }

    /// Element-wise negation (-x)
    pub fn neg(&self) -> Self {
        let zero = T::zero();
        let result_data: Vec<T> = self.data.iter().map(|&x| zero - x).collect();
        Self {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T: Scalar> Tensor<T> {
    /// Removes all dimensions of size 1 from the tensor's shape.
    /// Preserves original strides for non-1 axes.
    pub fn squeeze(&self) -> Self {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        for (i, &d) in self.shape.iter().enumerate() {
            if d != 1 {
                new_shape.push(d);
                new_strides.push(self.strides[i]);
            }
        }

        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Removes a single dimension of size 1 at the specified axis.
    /// Returns an error if axis is out of bounds or if shape[axis] != 1.
    pub fn squeeze_dim(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.ndim() {
            return Err(TensorError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        if self.shape[axis] != 1 {
            return Err(TensorError::CannotSqueezeDimension {
                axis,
                size: self.shape[axis],
            });
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.remove(axis);
        new_strides.remove(axis);

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Inserts a singleton dimension (size 1) at the specified axis.
    /// Accepts axis in range 0..=ndim.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, TensorError> {
        if axis > self.ndim() {
            return Err(TensorError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        // New dimension size is 1
        new_shape.insert(axis, 1);

        // Stride for singleton axis can be computed as product of subsequent dims
        // or set to the stride of dimension at position `axis` (or 1 if at trailing end).
        let inserted_stride = if axis < self.strides.len() {
            self.shape[axis] * self.strides[axis]
        } else {
            1
        };

        new_strides.insert(axis, inserted_stride);

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Flattens the entire tensor into a 1D tensor of shape [len].
    pub fn flatten(&self) -> Self {
        let len = self.len();
        Self {
            data: self.data.clone(),
            shape: vec![len],
            strides: vec![1],
        }
    }

    /// Reorders the axes of the tensor according to the specified permutation vector `axes`.
    /// Performs zero-copy reordering of `shape` and `strides`.
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.ndim() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Permutation length ({}) does not match tensor rank ({})",
                    axes.len(),
                    self.ndim()
                ),
            });
        }

        // Validate that axes is a valid permutation of 0..ndim
        let mut seen = vec![false; self.ndim()];
        for &axis in axes {
            if axis >= self.ndim() {
                return Err(TensorError::InvalidAxis {
                    axis,
                    ndim: self.ndim(),
                });
            }
            if seen[axis] {
                return Err(TensorError::InvalidShape {
                    reason: format!("Duplicate axis {} in permutation", axis),
                });
            }
            seen[axis] = true;
        }

        let mut new_shape = Vec::with_capacity(self.ndim());
        let mut new_strides = Vec::with_capacity(self.ndim());

        for &axis in axes {
            new_shape.push(self.shape[axis]);
            new_strides.push(self.strides[axis]);
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// General N-dimensional matrix transpose.
    /// Reverses the axes of the tensor (e.g. [2, 3, 4] -> [4, 3, 2]).
    pub fn transpose(&self) -> Self {
        let reversed_axes: Vec<usize> = (0..self.ndim()).rev().collect();
        self.permute(&reversed_axes)
            .expect("reversed axes must always form a valid permutation")
    }

    /// Convenience alias for 2D matrix transpose via permute(&[1, 0]).
    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "transpose_2d requires rank-2 tensor, got rank-{}",
                    self.ndim()
                ),
            });
        }
        self.permute(&[1, 0])
    }
}

impl<T: Scalar + PartialOrd> Tensor<T> {
    /// Reduces along an axis and returns the indices of the minimum values.
    pub fn argmin_dim(&self, axis: usize) -> Result<Tensor<usize>, TensorError> {
        self.reduce_index_dim(axis, |val, best| val < best)
    }

    /// Reduces along an axis and returns the indices of the maximum values.
    pub fn argmax_dim(&self, axis: usize) -> Result<Tensor<usize>, TensorError> {
        self.reduce_index_dim(axis, |val, best| val > best)
    }
}

impl<T: Real> Tensor<T> {
    pub fn abs(&self) -> Self {
        self.map_real(|x| x.abs())
    }

    pub fn sqrt(&self) -> Self {
        self.map_real(|x| x.sqrt())
    }

    pub fn exp(&self) -> Self {
        self.map_real(|x| x.exp())
    }

    pub fn ln(&self) -> Self {
        self.map_real(|x| x.ln())
    }

    pub fn pow(&self, exp: T) -> Self {
        self.map_real(|x| x.pow(exp))
    }

    // Internal helper for mapping Real operations
    fn map_real<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let result_data: Vec<T> = self.data.iter().map(|&x| f(x)).collect();
        Self {
            data: result_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
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

#[cfg(test)]
mod tensor_transformation_tests {
    use super::*;

    #[test]
    fn test_squeeze_and_squeeze_dim() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // Shape [1, 3, 1, 4] -> Strides [12, 4, 4, 1]
        let x = Tensor::from_vec(vec![1, 3, 1, 4], data).unwrap();

        // 1. Squeeze all 1s -> [3, 4]
        let squeezed = x.squeeze();
        assert_eq!(squeezed.shape(), &[3, 4]);
        assert_eq!(squeezed.strides(), &[4, 1]);
        assert_eq!(squeezed.get(&[2, 3]), Some(&11.0));

        // 2. Squeeze specific axis 0 -> [3, 1, 4]
        let sq_0 = x.squeeze_dim(0).unwrap();
        assert_eq!(sq_0.shape(), &[3, 1, 4]);

        // 3. Error when squeezing non-singleton dimension (axis 1 has size 3)
        assert_eq!(
            x.squeeze_dim(1),
            Err(TensorError::CannotSqueezeDimension { axis: 1, size: 3 })
        );
    }

    #[test]
    fn test_unsqueeze() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = Tensor::from_vec(vec![2, 3], data).unwrap();

        // Unsqueeze at axis 0 -> [1, 2, 3]
        let u0 = x.unsqueeze(0).unwrap();
        assert_eq!(u0.shape(), &[1, 2, 3]);
        assert_eq!(u0.get(&[0, 1, 2]), Some(&6.0));

        // Unsqueeze at axis 1 -> [2, 1, 3]
        let u1 = x.unsqueeze(1).unwrap();
        assert_eq!(u1.shape(), &[2, 1, 3]);
        assert_eq!(u1.get(&[1, 0, 2]), Some(&6.0));

        // Unsqueeze at trailing axis 2 -> [2, 3, 1]
        let u2 = x.unsqueeze(2).unwrap();
        assert_eq!(u2.shape(), &[2, 3, 1]);
        assert_eq!(u2.get(&[1, 2, 0]), Some(&6.0));

        // Out-of-bounds axis error (axis 3 for rank 2)
        assert_eq!(
            x.unsqueeze(3),
            Err(TensorError::InvalidAxis { axis: 3, ndim: 2 })
        );
    }

    #[test]
    fn test_flatten() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_vec(vec![2, 3, 4], data.clone()).unwrap();

        let flat = x.flatten();
        assert_eq!(flat.shape(), &[24]);
        assert_eq!(flat.as_slice(), data.as_slice());
    }

    #[test]
    fn test_permute_zero_copy() {
        // [2, 3, 4] with data 0..24
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = Tensor::from_vec(vec![2, 3, 4], data.clone()).unwrap();

        // Permute axes to [2, 0, 1] -> shape [4, 2, 3]
        let p = x.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);

        // Underlying data array remains 100% untouched (zero copy!)
        assert_eq!(p.as_slice(), data.as_slice());

        // Verify logical indexing through reordered strides
        // Original x[1, 2, 3] = offset (1*12 + 2*4 + 3*1) = 23 -> value 23.0
        // In permuted shape [4, 2, 3], original axis 2 is index 0 (val 3),
        // original axis 0 is index 1 (val 1), original axis 1 is index 2 (val 2)
        // Look up p[3, 1, 2]:
        assert_eq!(p.get(&[3, 1, 2]), Some(&23.0));
    }

    #[test]
    fn test_permute_invalid_axes() {
        let x = Tensor::<f32>::zeros(vec![2, 3, 4]);

        // Wrong length
        assert!(x.permute(&[1, 0]).is_err());
        // Duplicate axis
        assert!(x.permute(&[0, 1, 1]).is_err());
        // Out of bounds axis index
        assert!(x.permute(&[0, 1, 3]).is_err());
    }
}

impl<T: Scalar> Tensor<T> {
    /// Internal generic helper that returns the index of the best element along an axis.
    fn reduce_index_dim<F>(&self, axis: usize, is_better: F) -> Result<Tensor<usize>, TensorError>
    where
        F: Fn(T, T) -> bool, // returns true if the first arg is "better" than the second
    {
        if axis >= self.ndim() {
            return Err(TensorError::IndexOutOfBounds { dims: self.ndim(), index: vec![axis] });
        }
        let axis_size = self.shape[axis];
        if axis_size == 0 {
            return Err(TensorError::InvalidShape { reason: "Cannot perform arg reduction on size 0".to_string() });
        }

        let mut output_shape = self.shape.clone();
        output_shape.remove(axis);

        let total_output_elements = if output_shape.is_empty() { 1 } else { output_shape.iter().product() };
        let output_strides = calculate_stride(&output_shape);
        let mut result_data = Vec::with_capacity(total_output_elements);
        let axis_stride = self.strides[axis];

        for linear_out_idx in 0..total_output_elements {
            let mut base_in_coords = Vec::with_capacity(self.ndim());
            let mut temp_idx = linear_out_idx;

            for &item in output_strides.iter().take(output_shape.len()) {
                if base_in_coords.len() == axis { base_in_coords.push(0); }
                base_in_coords.push(temp_idx / item);
                temp_idx %= item;
            }
            if base_in_coords.len() == axis { base_in_coords.push(0); }

            let base_offset = self.calculate_offset(&base_in_coords)?;

            let mut best_idx = 0;
            let mut best_val = self.data[base_offset];

            for r in 1..axis_size {
                let val = self.data[base_offset + r * axis_stride];
                // In event of a tie, the strictly better (`<` or `>`) check fails, keeping the first index
                if is_better(val, best_val) {
                    best_val = val;
                    best_idx = r;
                }
            }
            result_data.push(best_idx);
        }

        Ok(Tensor {
            data: result_data,
            strides: output_strides,
            shape: output_shape,
        })
    }

    /// Internal generic helper that collapses a target axis by applying an aggregator closure.
    fn reduce_dim<F>(&self, axis: usize, aggregator: F) -> Result<Self, TensorError>
    where
        F: Fn(usize, T, T) -> T, // (step_index, accumulator, current_element) -> new_accumulator
    {
        if axis >= self.ndim() {
            return Err(TensorError::IndexOutOfBounds {
                dims: self.ndim(),
                index: vec![axis],
            });
        }

        let axis_size = self.shape[axis];
        if axis_size == 0 {
            return Err(TensorError::InvalidShape {
                reason: "Cannot perform reduction on dimension of size 0".to_string(),
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

            // Initialize accumulator with the first element along the reduction axis
            let mut acc = self.data[base_offset];

            // Fold remaining elements along the reduction axis
            for r in 1..axis_size {
                let val = self.data[base_offset + r * axis_stride];
                acc = aggregator(r, acc, val);
            }

            result_data.push(acc);
        }

        Ok(Self {
            data: result_data,
            strides: output_strides,
            shape: output_shape,
        })
    }
}

impl<T: PartialOrd + Scalar> Tensor<T> {
    pub fn relu(&self) -> Self {
        let zero = T::zero();
        let result_data = self.data.iter().map(|&x| if x > zero { x } else { zero }).collect();
        Self { data: result_data, shape: self.shape.clone(), strides: self.strides.clone() }
    }

    pub fn leaky_relu(&self, alpha: T) -> Self {
        let zero = T::zero();
        let result_data = self.data.iter().map(|&x| if x > zero { x } else { x * alpha }).collect();
        Self { data: result_data, shape: self.shape.clone(), strides: self.strides.clone() }
    }
}

impl<T: Real> Tensor<T> {
    pub fn sigmoid(&self) -> Self {
        let one = T::one();
        let zero = T::zero();
        // sigmoid(x) = 1 / (1 + exp(-x))
        self.map_real(|x| one / (one + (zero - x).exp()))
    }

    pub fn tanh(&self) -> Self {
        self.map_real(|x| x.tanh())
    }

    /// Computes the softmax along the specified axis in a numerically stable way.
    pub fn softmax(&self, axis: usize) -> Result<Self, TensorError> {
        // 1. max(x) -> Output loses the axis dimension
        let x_max = self.max_dim(axis)?;

        // 2. Unsqueeze back to size 1 so broadcasting align rules apply cleanly
        let x_max_unsq = x_max.unsqueeze(axis)?;

        // 3. x - x_max (Broadcasts cleanly across the axis!)
        let shifted = self.sub(&x_max_unsq)?;

        // 4. exp(x - x_max)
        let exps = shifted.exp();

        // 5. sum(exp(x - x_max))
        let sum_exps = exps.sum_dim(axis)?;
        let sum_exps_unsq = sum_exps.unsqueeze(axis)?;

        // 6. Divide element-wise with broadcasting
        exps.div(&sum_exps_unsq)
    }

    /// Computes the log_softmax along the specified axis in a numerically stable way.
    pub fn log_softmax(&self, axis: usize) -> Result<Self, TensorError> {
        // log_softmax(x) = (x - max(x)) - log(sum(exp(x - max(x))))
        let x_max = self.max_dim(axis)?.unsqueeze(axis)?;
        let shifted = self.sub(&x_max)?;

        let exps = shifted.exp();
        let sum_exps_unsq = exps.sum_dim(axis)?.unsqueeze(axis)?;
        let log_sum_exps = sum_exps_unsq.ln();

        shifted.sub(&log_sum_exps)
    }

}
