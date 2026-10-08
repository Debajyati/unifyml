use std::ops::{IndexMut, Index};
use crate::Vector;
use crate::MatrixError;
use crate::Scalar;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T: Scalar> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Scalar> Default for Matrix<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Matrix<T> {
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get(row * self.cols + col)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get_mut(row * self.cols + col)
    }

    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::DataLengthMismatch {
                expected: rows * cols,
                actual: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    pub fn from_slice(rows: usize, cols: usize, slice: &[T]) -> Result<Self, MatrixError> {
        if slice.len() != rows * cols {
            return Err(MatrixError::DataLengthMismatch {
                expected: rows * cols,
                actual: slice.len(),
            });
        }
        Ok(Self {
            data: slice.to_vec(),
            rows,
            cols,
        })
    }

    pub fn from_slices(rows: usize, cols: usize, slices: &[&[T]]) -> Result<Self, MatrixError> {
        if slices.len() != rows {
            return Err(MatrixError::DataLengthMismatch {
                expected: rows,
                actual: slices.len(),
            });
        }

        let mut data = Vec::with_capacity(rows * cols);
        for slice in slices {
            if slice.len() != cols {
                return Err(MatrixError::DataLengthMismatch {
                    expected: cols,
                    actual: slice.len(),
                });
            }
            data.extend_from_slice(slice);
        }

        Ok(Self { data, rows, cols })
    }

    pub fn from_rows(data: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        if data.is_empty() {
            return Ok(Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }

        let rows = data.len();
        let cols = data[0].len();

        for row in &data {
            if row.len() != cols {
                return Err(MatrixError::DataLengthMismatch {
                    expected: cols,
                    actual: row.len(),
                });
            }
        }

        let flat_data: Vec<T> = data.into_iter().flatten().collect();

        Ok(Self {
            data: flat_data,
            rows,
            cols,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn from_vector(v: Vector<T>, rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if v.len() != rows * cols {
            return Err(MatrixError::DataLengthMismatch {
                expected: rows * cols,
                actual: v.len(),
            });
        }
        Ok(Self {
            data: v.to_vec(),
            rows,
            cols,
        })
    }

    pub fn to_vector(self) -> Vector<T> {
        Vector::from_vec(self.data)
    }

    pub fn from_vector_column(v: Vector<T>) -> Self {
        let rows = v.len();
        let cols = 1;
        Self {
            data: v.to_vec(),
            rows,
            cols,
        }
    }

    pub fn from_vector_row(v: Vector<T>) -> Self {
        let rows = 1;
        let cols = v.len();
        Self {
            data: v.to_vec(),
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn scale(&self, scaling_factor: T) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * scaling_factor).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn with_value(rows: usize, cols: usize, value: T) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::zero(); rows * cols],
            rows,
            cols,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::one(); rows * cols],
            rows,
            cols,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut tp = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let src_idx = i * self.cols + j;
                let dest_idx = j * self.rows + i;

                tp.data[dest_idx] = self.data[src_idx];
            }
        }
        tp
    }

    pub fn add(&self, m: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.rows != m.rows || self.cols != m.cols {
            return Err(MatrixError::DimensionMismatch {
                left: (self.rows, self.cols),
                right: (m.rows, m.cols),
            });
        }
        let result: Vec<T> = self
            .data
            .iter()
            .zip(m.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Self {
            data: result,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn sub(&self, m: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.rows != m.rows || self.cols != m.cols {
            return Err(MatrixError::DimensionMismatch {
                left: (self.rows, self.cols),
                right: (m.rows, m.cols),
            });
        }
        let result: Vec<T> = self
            .data
            .iter()
            .zip(m.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Self {
            data: result,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn matmul(&self, m: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.cols != m.rows {
            return Err(MatrixError::DimensionMismatch {
                left: (self.rows, self.cols),
                right: (m.rows, m.cols),
            });
        }

        let mut result = Self::zeros(self.rows, m.cols);

        for i in 0..self.rows {
            for k in 0..self.cols {
                let val = self.data[i * self.cols + k];

                for j in 0..m.cols {
                    let idx = i * m.cols + j;
                    result.data[idx] = result.data[idx] + val * m.data[k * m.cols + j];
                }
            }
        }

        Ok(result)
    }

    pub fn matmul_vector(&self, v: &Vector<T>) -> Result<Vector<T>, MatrixError> {
        if self.cols == 0 {
            return Ok(Vector::from_vec(vec![T::zero(); self.rows]));
        }

        if self.cols != v.len() {
            return Err(MatrixError::VectorDimensionMismatch {
                matrix_cols: self.cols,
                vector_len: v.len(),
            });
        }

        let mat = self.as_slice();
        let vec = v.as_slice();
        let mut out = vec![T::zero(); self.rows];

        for (i, row_chunk) in mat.chunks_exact(self.cols).enumerate() {
            out[i] = row_chunk
                .iter()
                .zip(vec.iter())
                .map(|(&a, &b)| a * b)
                .fold(T::zero(), |acc , el| acc + el)
        }

        Ok(Vector::from_vec(out))
    }
}

impl<T: Scalar> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[row * self.cols + col]
    }
}

impl<T: Scalar> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.data[row * self.cols + col]
    }
}
