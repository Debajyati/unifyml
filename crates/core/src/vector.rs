use crate::Real;
use crate::VectorError;
use crate::Scalar;
use std::{
    ops::{Index, IndexMut},
    vec,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: Scalar> {
    data: Vec<T>,
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

    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
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

    pub fn squared_norm(&self) -> T {
        self.dot(self).unwrap_or_else(|_| T::zero())
    }

    pub fn product(&self) -> T {
        if self.is_empty() {
            return T::one();
        }
        self.data.iter().fold(T::one(), |acc, &x| acc * x)
    }

    pub fn sum(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        self.data.iter().fold(T::zero(), |acc, &x| acc + x)
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

impl<T: Real> Vector<T> {
    pub fn min(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mut low = self.data[0];

        for &i in self.data.iter() {
            if i.lt(&low) {
                low = i;
            }
        }

        Some(low)
    }

    pub fn max(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mut high = self.data[0];

        for &i in self.data.iter() {
            if i.gt(&high) {
                high = i;
            }
        }

        Some(high)
    }
}

impl<T: Real> Vector<T> {
    pub fn norm_l1(&self) -> T {
        self.data.iter().fold(T::zero(), |acc, &x| acc + x.abs())
    }

    pub fn norm_l2(&self) -> T {
        self.squared_norm().sqrt()
    }
}

