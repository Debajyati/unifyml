use std::ops::{Add, Div, Mul, Sub};

pub trait Scalar:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl Scalar for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

pub trait Real: Scalar + PartialOrd {
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

impl Real for f32 {
    fn abs(self) -> Self { f32::abs(self) }
    fn sqrt(self) -> Self { f32::sqrt(self) }
}
