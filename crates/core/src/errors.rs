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
