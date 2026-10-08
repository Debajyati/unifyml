pub mod errors;
pub mod tensor;
pub mod matrix;
pub mod vector;
pub mod scalar;

pub use errors::{MatrixError, VectorError};
pub use matrix::Matrix;
pub use scalar::{Real, Scalar};
pub use vector::Vector;
pub use tensor::Tensor;
