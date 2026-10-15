//! Operaciones algebraicas de alto nivel y kernels BLAS-like propios.
//!
//! La implementación evita depender de BLAS/LAPACK. Cada operación trabaja
//! sobre el almacenamiento row-major de TRACE y reutiliza los kernels internos
//! cuando la operación puede expresarse como combinación de primitives.

use crate::{DenseMatrix, RuntimePolicy, Vector};
use crate::operations::matrix_matrix_multiply;

/// Suma escalada `y = alpha * x + y`.
pub fn axpy(alpha: f64, x: &Vector, y: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    if x.len() != y.len() {
        return Err("vectors must have the same length".into());
    }
    let mut result = y.data.clone();
    if policy.plan_for(x.len()).threads <= 1 {
        for index in 0..x.len() {
            result[index] = alpha.mul_add(x.data[index], result[index]);
        }
    } else {
        use rayon::prelude::*;
        result.par_iter_mut().enumerate().for_each(|(index, value)| {
            *value = alpha.mul_add(x.data[index], *value);
        });
    }
    Ok(Vector::new(result))
}

/// Producto cruz tridimensional.
pub fn cross_product(left: &Vector, right: &Vector) -> Result<Vector, String> {
    if left.len() != 3 || right.len() != 3 {
        return Err("cross product requires vectors of length 3".into());
    }
    Ok(Vector::new(vec![
        left.data[1] * right.data[2] - left.data[2] * right.data[1],
        left.data[2] * right.data[0] - left.data[0] * right.data[2],
        left.data[0] * right.data[1] - left.data[1] * right.data[0],
    ]))
}

/// GEMV generalizado `y = alpha * A * x + beta * y`.
pub fn gemv(alpha: f64, matrix: &DenseMatrix, x: &Vector, beta: f64, y: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    if matrix.columns != x.len() || matrix.rows != y.len() {
        return Err("GEMV dimensions are incompatible".into());
    }
    let product = crate::operations::matrix_vector_multiply(matrix, x, policy)?;
    let mut result = vec![0.0; matrix.rows];
    for row in 0..matrix.rows {
        result[row] = alpha.mul_add(product.data[row], beta * y.data[row]);
    }
    Ok(Vector::new(result))
}

/// Actualización de rango uno `A = A + alpha * x * y^T`.
pub fn ger(alpha: f64, x: &Vector, y: &Vector, matrix: &DenseMatrix, policy: RuntimePolicy) -> Result<DenseMatrix, String> {
    if matrix.rows != x.len() || matrix.columns != y.len() {
        return Err("GER dimensions are incompatible".into());
    }
    let mut data = matrix.data.clone();
    let plan = policy.plan_for(matrix.rows * matrix.columns);
    if plan.threads <= 1 {
        for row in 0..matrix.rows {
            let row_offset = row * matrix.columns;
            for column in 0..matrix.columns {
                data[row_offset + column] += alpha * x.data[row] * y.data[column];
            }
        }
    } else {
        use rayon::prelude::*;
        data.par_chunks_mut(matrix.columns).enumerate().for_each(|(row, output)| {
            let scale = alpha * x.data[row];
            for column in 0..matrix.columns {
                output[column] += scale * y.data[column];
            }
        });
    }
    DenseMatrix::new(matrix.rows, matrix.columns, data)
}

/// Actualización simétrica `C = alpha * A * A^T + beta * C`.
pub fn syrk(alpha: f64, a: &DenseMatrix, beta: f64, c: &DenseMatrix, policy: RuntimePolicy) -> Result<DenseMatrix, String> {
    if c.rows != c.columns || a.rows != c.rows {
        return Err("SYRK dimensions are incompatible".into());
    }
    let product = matrix_matrix_multiply(a, &transpose(a)?, policy)?;
    let mut data = vec![0.0; c.data.len()];
    for row in 0..c.rows {
        for column in 0..c.columns {
            data[row * c.columns + column] = alpha * product.data[row * c.columns + column] + beta * c.data[row * c.columns + column];
        }
    }
    DenseMatrix::new(c.rows, c.columns, data)
}

/// Transposición out-of-place con escritura secuencial por filas de destino.
pub fn transpose(matrix: &DenseMatrix) -> Result<DenseMatrix, String> {
    let mut data = vec![0.0; matrix.rows * matrix.columns];
    for row in 0..matrix.rows {
        let source_offset = row * matrix.columns;
        for column in 0..matrix.columns {
            data[column * matrix.rows + row] = matrix.data[source_offset + column];
        }
    }
    DenseMatrix::new(matrix.columns, matrix.rows, data)
}

/// Traza de una matriz cuadrada.
pub fn trace(matrix: &DenseMatrix) -> Result<f64, String> {
    if matrix.rows != matrix.columns {
        return Err("trace requires a square matrix".into());
    }
    Ok((0..matrix.rows).map(|index| matrix.data[index * matrix.columns + index]).sum())
}

/// Norma de Frobenius.
pub fn matrix_norm_frobenius(matrix: &DenseMatrix, policy: RuntimePolicy) -> f64 {
    let mut sum = 0.0;
    if policy.plan_for(matrix.data.len()).threads <= 1 {
        for &value in &matrix.data { sum = value.mul_add(value, sum); }
    } else {
        use rayon::prelude::*;
        sum = matrix.data.par_iter().map(|value| value * value).sum();
    }
    sum.sqrt()
}

/// Norma 1: máximo de las sumas absolutas por columna.
pub fn matrix_norm_one(matrix: &DenseMatrix) -> f64 {
    let mut maximum: f64 = 0.0;
    for column in 0..matrix.columns {
        let mut sum = 0.0;
        for row in 0..matrix.rows { sum += matrix.data[row * matrix.columns + column].abs(); }
        maximum = maximum.max(sum);
    }
    maximum
}

/// Norma infinito: máximo de las sumas absolutas por fila.
pub fn matrix_norm_inf(matrix: &DenseMatrix) -> f64 {
    let mut maximum: f64 = 0.0;
    for row in 0..matrix.rows {
        let offset = row * matrix.columns;
        let sum: f64 = matrix.data[offset..offset + matrix.columns].iter().map(|value| value.abs()).sum();
        maximum = maximum.max(sum);
    }
    maximum
}

/// Norma espectral aproximada mediante iteración de potencia sobre A^T A.
pub fn matrix_norm_two(matrix: &DenseMatrix, policy: RuntimePolicy) -> Result<f64, String> {
    if matrix.rows == 0 || matrix.columns == 0 { return Ok(0.0); }
    let transpose_matrix = transpose(matrix)?;
    let mut vector = Vector::new(vec![1.0; matrix.columns]);
    let initial = crate::operations::vector_norm2(&vector, policy);
    if initial == 0.0 { return Ok(0.0); }
    for _ in 0..30 {
        let ax = crate::operations::matrix_vector_multiply(matrix, &vector, policy)?;
        vector = crate::operations::matrix_vector_multiply(&transpose_matrix, &ax, policy)?;
        let norm = crate::operations::vector_norm2(&vector, policy);
        if norm <= f64::EPSILON { return Ok(0.0); }
        for value in &mut vector.data { *value /= norm; }
    }
    let ax = crate::operations::matrix_vector_multiply(matrix, &vector, policy)?;
    Ok(crate::operations::vector_norm2(&ax, policy))
}

/// Producto de Kronecker entre dos matrices densas.
pub fn kronecker_product(left: &DenseMatrix, right: &DenseMatrix) -> Result<DenseMatrix, String> {
    let rows = left.rows.checked_mul(right.rows).ok_or("Kronecker row count overflow")?;
    let columns = left.columns.checked_mul(right.columns).ok_or("Kronecker column count overflow")?;
    let mut data = vec![0.0; rows * columns];
    for left_row in 0..left.rows {
        for left_column in 0..left.columns {
            let scale = left.data[left_row * left.columns + left_column];
            for right_row in 0..right.rows {
                let destination_row = left_row * right.rows + right_row;
                let destination_offset = destination_row * columns + left_column * right.columns;
                let source_offset = right_row * right.columns;
                for right_column in 0..right.columns {
                    data[destination_offset + right_column] = scale * right.data[source_offset + right_column];
                }
            }
        }
    }
    DenseMatrix::new(rows, columns, data)
}

/// Resuelve `A x = b` mediante eliminación con pivoteo parcial.
pub fn solve_linear_system(matrix: &DenseMatrix, b: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    if matrix.rows != matrix.columns || b.len() != matrix.rows {
        return Err("linear system dimensions are incompatible".into());
    }
    let factors = crate::decompositions::lu(matrix, policy)?;
    factors.solve(b)
}

/// Resuelve `A X = B` reutilizando una única factorización LU.
pub fn solve_multiple_rhs(matrix: &DenseMatrix, rhs: &DenseMatrix, policy: RuntimePolicy) -> Result<DenseMatrix, String> {
    if matrix.rows != matrix.columns || rhs.rows != matrix.rows {
        return Err("multiple-RHS system dimensions are incompatible".into());
    }
    let factors = crate::decompositions::lu(matrix, policy)?;
    let mut data = vec![0.0; rhs.rows * rhs.columns];
    for column in 0..rhs.columns {
        let vector = Vector::new((0..rhs.rows).map(|row| rhs.data[row * rhs.columns + column]).collect());
        let solution = factors.solve(&vector)?;
        for row in 0..rhs.rows { data[row * rhs.columns + column] = solution.data[row]; }
    }
    DenseMatrix::new(rhs.rows, rhs.columns, data)
}

/// Inversa mediante múltiples resoluciones sobre la identidad.
pub fn inverse(matrix: &DenseMatrix, policy: RuntimePolicy) -> Result<DenseMatrix, String> {
    if matrix.rows != matrix.columns { return Err("inverse requires a square matrix".into()); }
    let mut identity = vec![0.0; matrix.rows * matrix.columns];
    for index in 0..matrix.rows { identity[index * matrix.columns + index] = 1.0; }
    solve_multiple_rhs(matrix, &DenseMatrix::new(matrix.rows, matrix.columns, identity)?, policy)
}

/// Determinante calculado desde LU para evitar una eliminación duplicada.
pub fn determinant(matrix: &DenseMatrix, policy: RuntimePolicy) -> Result<f64, String> {
    if matrix.rows != matrix.columns { return Err("determinant requires a square matrix".into()); }
    let factors = crate::decompositions::lu(matrix, policy)?;
    let mut determinant = factors.pivot_sign as f64;
    for index in 0..matrix.rows { determinant *= factors.u.data[index * matrix.columns + index]; }
    Ok(determinant)
}

/// Resuelve un sistema triangular inferior o superior.
pub fn triangular_solve(matrix: &DenseMatrix, b: &Vector, lower: bool, unit_diagonal: bool) -> Result<Vector, String> {
    if matrix.rows != matrix.columns || b.len() != matrix.rows { return Err("triangular solve dimensions are incompatible".into()); }
    let n = matrix.rows;
    let mut result = b.data.clone();
    if lower {
        for row in 0..n {
            let offset = row * n;
            let mut value = result[row];
            for column in 0..row { value -= matrix.data[offset + column] * result[column]; }
            if !unit_diagonal {
                let diagonal = matrix.data[offset + row];
                if diagonal.abs() <= f64::EPSILON { return Err("singular triangular matrix".into()); }
                value /= diagonal;
            }
            result[row] = value;
        }
    } else {
        for row in (0..n).rev() {
            let offset = row * n;
            let mut value = result[row];
            for column in row + 1..n { value -= matrix.data[offset + column] * result[column]; }
            if !unit_diagonal {
                let diagonal = matrix.data[offset + row];
                if diagonal.abs() <= f64::EPSILON { return Err("singular triangular matrix".into()); }
                value /= diagonal;
            }
            result[row] = value;
        }
    }
    Ok(Vector::new(result))
}

/// Resolución TRSM por columnas para una matriz triangular izquierda.
pub fn trsm(left: &DenseMatrix, right: &DenseMatrix, lower: bool, unit_diagonal: bool) -> Result<DenseMatrix, String> {
    if left.rows != left.columns || right.rows != left.rows { return Err("TRSM dimensions are incompatible".into()); }
    let mut data = vec![0.0; right.data.len()];
    for column in 0..right.columns {
        let rhs = Vector::new((0..right.rows).map(|row| right.data[row * right.columns + column]).collect());
        let solution = triangular_solve(left, &rhs, lower, unit_diagonal)?;
        for row in 0..right.rows { data[row * right.columns + column] = solution.data[row]; }
    }
    DenseMatrix::new(right.rows, right.columns, data)
}
