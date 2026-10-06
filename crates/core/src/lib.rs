use std::{
    ops::{Add, Div, Mul, Sub},
    ops::{Index, IndexMut},
    vec,
};

pub trait Scalar:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: Scalar> {
    data: Vec<T>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum VectorError {
    DimensionMismatch { left: usize, right: usize },
}

impl<T: Scalar> Default for Vector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Vector<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![T::zero(); size],
        }
    }

    pub fn ones(size: usize) -> Self {
        Self {
            data: vec![T::one(); size],
        }
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        self.data.get(idx)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn add(&self, v: &Vector<T>) -> Result<Vector<T>, VectorError> {
        if self.len() != v.len() {
            return Err(VectorError::DimensionMismatch {
                left: self.len(),
                right: v.len(),
            });
        }

        let result: Vec<T> = self
            .data
            .iter()
            .zip(v.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Ok(Self { data: result })
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            data: slice.to_vec(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn sub(&self, v: &Vector<T>) -> Result<Vector<T>, VectorError> {
        if self.len() != v.len() {
            return Err(VectorError::DimensionMismatch {
                left: self.len(),
                right: v.len(),
            });
        }

        let result: Vec<T> = self
            .data
            .iter()
            .zip(v.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        Ok(Self { data: result })
    }

    pub fn scale(&self, scaling_factor: T) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * scaling_factor).collect(),
        }
    }

    pub fn dot(&self, other: &Vector<T>) -> Result<T, VectorError> {
        if self.len() != other.len() {
            return Err(VectorError::DimensionMismatch {
                left: self.len(),
                right: other.len(),
            });
        }

        if self.is_empty() {
            return Ok(T::zero());
        }

        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x * y)
            .fold(T::zero(), |acc, x| acc + x);

        Ok(result)
    }

    pub fn sum(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        let mut res = T::zero();

        for &i in self.data.iter() {
            res = res + i;
        }

        res
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T: Scalar> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum MatrixError {
    DimensionMismatch {
        left: (usize, usize),
        right: (usize, usize),
    },
    DataLengthMismatch {
        expected: usize,
        actual: usize,
    },
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
            data: v.data,
            rows,
            cols,
        })
    }

    pub fn to_vector(self) -> Vector<T> {
        Vector { data: self.data }
    }

    pub fn from_vector_column(v: Vector<T>) -> Self {
        let rows = v.len();
        let cols = 1;
        Self {
            data: v.data,
            rows,
            cols,
        }
    }

    pub fn from_vector_row(v: Vector<T>) -> Self {
        let rows = 1;
        let cols = v.len();
        Self {
            data: v.data,
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
}

impl<T: Scalar> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Scalar> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
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
