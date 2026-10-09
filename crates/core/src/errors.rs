#[derive(Debug, PartialEq, Eq)]
pub enum VectorError {
    DimensionMismatch { left: usize, right: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum MatrixError {
    DimensionMismatch { left: (usize, usize), right: (usize, usize) },
    VectorDimensionMismatch { matrix_cols: usize, vector_len: usize },
    DataLengthMismatch { expected: usize, actual: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    DataLengthMismatch { expected: usize, actual: usize },
    InvalidShape { reason: String },
    IndexOutOfBounds { dims: usize, index: Vec<usize> },
    ReshapeMismatch { old_len: usize, new_len: usize },
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
}
