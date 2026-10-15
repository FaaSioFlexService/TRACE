//! Representaciones dense, CSR y ELLPACK.

use crate::Vector;

/// Matriz densa row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>,
}

impl DenseMatrix {
    /// Crea una matriz validando el tamaño del buffer.
    pub fn new(rows: usize, columns: usize, data: Vec<f64>) -> Result<Self, String> {
        if data.len() != rows.saturating_mul(columns) {
            return Err("matrix data length does not match dimensions".into());
        }
        Ok(Self { rows, columns, data })
    }

    /// Devuelve una fila como slice contiguo.
    #[inline]
    pub fn row(&self, row: usize) -> &[f64] {
        let offset = row * self.columns;
        &self.data[offset..offset + self.columns]
    }
}

/// Matriz Compressed Sparse Row.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f64>,
    pub column_indices: Vec<usize>,
    pub row_pointers: Vec<usize>,
}

impl CsrMatrix {
    /// Construye CSR y valida la estructura de filas.
    pub fn new(
        rows: usize,
        columns: usize,
        values: Vec<f64>,
        column_indices: Vec<usize>,
        row_pointers: Vec<usize>,
    ) -> Result<Self, String> {
        if values.len() != column_indices.len() {
            return Err("CSR values and column indices must have the same length".into());
        }
        if row_pointers.len() != rows + 1 {
            return Err("CSR row pointer length must be rows + 1".into());
        }
        if row_pointers.last().copied().unwrap_or(0) != values.len() {
            return Err("CSR last row pointer must equal nnz".into());
        }
        if row_pointers.windows(2).any(|window| window[0] > window[1]) {
            return Err("CSR row pointers must be monotonic".into());
        }
        if column_indices.iter().any(|&column| column >= columns) {
            return Err("CSR column index out of bounds".into());
        }
        Ok(Self { rows, columns, values, column_indices, row_pointers })
    }
}


/// Matriz Compressed Sparse Column.
#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub column_pointers: Vec<usize>,
}

impl CscMatrix {
    /// Construye CSC validando la estructura de columnas.
    pub fn new(
        rows: usize,
        columns: usize,
        values: Vec<f64>,
        row_indices: Vec<usize>,
        column_pointers: Vec<usize>,
    ) -> Result<Self, String> {
        if values.len() != row_indices.len() {
            return Err("CSC values and row indices must have the same length".into());
        }
        if column_pointers.len() != columns + 1 {
            return Err("CSC column pointer length must be columns + 1".into());
        }
        if column_pointers.last().copied().unwrap_or(0) != values.len() {
            return Err("CSC last column pointer must equal nnz".into());
        }
        if column_pointers.windows(2).any(|window| window[0] > window[1]) {
            return Err("CSC column pointers must be monotonic".into());
        }
        if row_indices.iter().any(|&row| row >= rows) {
            return Err("CSC row index out of bounds".into());
        }
        Ok(Self { rows, columns, values, row_indices, column_pointers })
    }
}

/// Matriz ELLPACK con ancho fijo por fila.
#[derive(Clone, Debug, PartialEq)]
pub struct EllpackMatrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f64>,
    pub column_indices: Vec<isize>,
    pub max_non_zeros_per_row: usize,
    pub non_zero_count: usize,
}

impl EllpackMatrix {
    /// Construye ELLPACK y valida el almacenamiento regular.
    pub fn new(
        rows: usize,
        columns: usize,
        values: Vec<f64>,
        column_indices: Vec<isize>,
        max_non_zeros_per_row: usize,
        non_zero_count: usize,
    ) -> Result<Self, String> {
        let storage = rows.saturating_mul(max_non_zeros_per_row);
        if values.len() != storage || column_indices.len() != storage {
            return Err("ELLPACK storage length does not match dimensions".into());
        }
        if column_indices.iter().any(|&column| column >= columns as isize) {
            return Err("ELLPACK column index out of bounds".into());
        }
        if non_zero_count > storage {
            return Err("ELLPACK non-zero count exceeds storage".into());
        }
        Ok(Self { rows, columns, values, column_indices, max_non_zeros_per_row, non_zero_count })
    }
}

impl From<Vec<f64>> for Vector {
    fn from(data: Vec<f64>) -> Self {
        Self::new(data)
    }
}

/// Matriz en cualquiera de las representaciones soportadas por TRACE.
#[derive(Clone, Debug, PartialEq)]
pub enum Matrix {
    Dense(DenseMatrix),
    Csr(CsrMatrix),
    Csc(CscMatrix),
    Ellpack(EllpackMatrix),
}

/// Convierte una matriz densa a CSR.
pub fn dense_to_csr(matrix: &DenseMatrix, zero_tolerance: f64) -> CsrMatrix {
    let mut values = Vec::new();
    let mut column_indices = Vec::new();
    let mut row_pointers = Vec::with_capacity(matrix.rows + 1);
    row_pointers.push(0);
    for row in 0..matrix.rows {
        let offset = row * matrix.columns;
        for column in 0..matrix.columns {
            let value = matrix.data[offset + column];
            if value.abs() > zero_tolerance {
                values.push(value);
                column_indices.push(column);
            }
        }
        row_pointers.push(values.len());
    }
    CsrMatrix { rows: matrix.rows, columns: matrix.columns, values, column_indices, row_pointers }
}

/// Convierte una matriz densa a CSC.
pub fn dense_to_csc(matrix: &DenseMatrix, zero_tolerance: f64) -> CscMatrix {
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut column_pointers = Vec::with_capacity(matrix.columns + 1);
    column_pointers.push(0);
    for column in 0..matrix.columns {
        for row in 0..matrix.rows {
            let value = matrix.data[row * matrix.columns + column];
            if value.abs() > zero_tolerance {
                values.push(value);
                row_indices.push(row);
            }
        }
        column_pointers.push(values.len());
    }
    CscMatrix { rows: matrix.rows, columns: matrix.columns, values, row_indices, column_pointers }
}

/// Convierte una matriz densa a ELLPACK.
pub fn dense_to_ellpack(matrix: &DenseMatrix, zero_tolerance: f64) -> EllpackMatrix {
    let mut max_non_zeros_per_row = 0;
    let mut non_zero_count = 0;
    for row in 0..matrix.rows {
        let offset = row * matrix.columns;
        let count = (0..matrix.columns).filter(|&column| matrix.data[offset + column].abs() > zero_tolerance).count();
        max_non_zeros_per_row = max_non_zeros_per_row.max(count);
        non_zero_count += count;
    }
    let storage = matrix.rows * max_non_zeros_per_row;
    let mut values = vec![0.0; storage];
    let mut column_indices = vec![-1isize; storage];
    for row in 0..matrix.rows {
        let dense_offset = row * matrix.columns;
        let ell_offset = row * max_non_zeros_per_row;
        let mut slot = 0;
        for column in 0..matrix.columns {
            let value = matrix.data[dense_offset + column];
            if value.abs() <= zero_tolerance { continue; }
            values[ell_offset + slot] = value;
            column_indices[ell_offset + slot] = column as isize;
            slot += 1;
        }
    }
    EllpackMatrix { rows: matrix.rows, columns: matrix.columns, values, column_indices, max_non_zeros_per_row, non_zero_count }
}
