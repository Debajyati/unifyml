use std::ops::{Add, Div, Mul, Sub};

pub trait Scalar:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(n: usize) -> Self;
}

impl Scalar for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_usize(n: usize) -> Self { n as f32 }
}

pub trait Real: Scalar + PartialOrd {
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn pow(self, exp: Self) -> Self;
    fn tanh(self) -> Self;
}

impl Real for f32 {
    fn abs(self) -> Self { f32::abs(self) }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn exp(self) -> Self { f32::exp(self) }
    fn ln(self) -> Self { f32::ln(self) }
    fn pow(self, exp: Self) -> Self { f32::powf(self, exp) }
    fn tanh(self) -> Self { f32::tanh(self) }
}

impl Scalar for usize {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn from_usize(n: usize) -> Self { n }
}

impl Scalar for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_usize(n: usize) -> Self { n as f64 }
}

impl Real for f64 {
    fn abs(self) -> Self { f64::abs(self) }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn exp(self) -> Self { f64::exp(self) }
    fn ln(self) -> Self { f64::ln(self) }
    fn pow(self, exp: Self) -> Self { f64::powf(self, exp) }
    fn tanh(self) -> Self { f64::tanh(self) }
}

