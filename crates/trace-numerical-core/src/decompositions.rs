//! Factorizaciones matriciales implementadas directamente en Rust.
//!
//! Esta capa no llama a LAPACK/BLAS. Los algoritmos se mantienen explícitos
//! para poder medir por separado algoritmo, almacenamiento y hardware.

use crate::{DenseMatrix, RuntimePolicy, Vector};

/// Factorización LU con pivoteo parcial `P A = L U`.
#[derive(Clone, Debug)]
pub struct LuFactors {
    pub l: DenseMatrix,
    pub u: DenseMatrix,
    pub permutation: Vec<usize>,
    pub pivot_sign: i32,
}

impl LuFactors {
    /// Resuelve un sistema usando la factorización ya calculada.
    pub fn solve(&self, b: &Vector) -> Result<Vector, String> {
        if b.len() != self.l.rows { return Err("LU right-hand side has incompatible length".into()); }
        let n = self.l.rows;
        let mut permuted = vec![0.0; n];
        for row in 0..n { permuted[row] = b.data[self.permutation[row]]; }
        let y = solve_lower_raw(&self.l, &permuted, true)?;
        let x = solve_upper_raw(&self.u, &y)?;
        Ok(Vector::new(x))
    }
}

/// Factoriza una matriz cuadrada general mediante eliminación con pivoteo.
pub fn lu(matrix: &DenseMatrix, _policy: RuntimePolicy) -> Result<LuFactors, String> {
    if matrix.rows != matrix.columns { return Err("LU requires a square matrix".into()); }
    let n = matrix.rows;
    let mut u = matrix.data.clone();
    let mut l = vec![0.0; n * n];
    let mut permutation: Vec<usize> = (0..n).collect();
    for index in 0..n { l[index * n + index] = 1.0; }
    let mut pivot_sign = 1;

    for column in 0..n {
        let mut pivot = column;
        let mut maximum = u[column * n + column].abs();
        for row in column + 1..n {
            let candidate = u[row * n + column].abs();
            if candidate > maximum { maximum = candidate; pivot = row; }
        }
        if maximum <= f64::EPSILON { return Err("matrix is singular to working precision".into()); }
        if pivot != column {
            for k in 0..n { u.swap(column * n + k, pivot * n + k); }
            for k in 0..column { l.swap(column * n + k, pivot * n + k); }
            permutation.swap(column, pivot);
            pivot_sign = -pivot_sign;
        }
        for row in column + 1..n {
            let factor = u[row * n + column] / u[column * n + column];
            l[row * n + column] = factor;
            u[row * n + column] = 0.0;
            for k in column + 1..n {
                let index = row * n + k;
                u[index] -= factor * u[column * n + k];
            }
        }
    }
    Ok(LuFactors {
        l: DenseMatrix::new(n, n, l)?,
        u: DenseMatrix::new(n, n, u)?,
        permutation,
        pivot_sign,
    })
}

/// Factorización de Cholesky `A = L L^T` para matrices SPD.
pub fn cholesky(matrix: &DenseMatrix) -> Result<DenseMatrix, String> {
    if matrix.rows != matrix.columns { return Err("Cholesky requires a square matrix".into()); }
    let n = matrix.rows;
    let mut l = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..=row {
            let mut sum = matrix.data[row * n + column];
            for k in 0..column { sum -= l[row * n + k] * l[column * n + k]; }
            if row == column {
                if sum <= 0.0 { return Err("matrix is not positive definite".into()); }
                l[row * n + column] = sum.sqrt();
            } else {
                let diagonal = l[column * n + column];
                if diagonal.abs() <= f64::EPSILON { return Err("zero Cholesky pivot".into()); }
                l[row * n + column] = sum / diagonal;
            }
        }
    }
    DenseMatrix::new(n, n, l)
}

/// Factorización LDL^T para matrices simétricas sin pivoteo.
pub fn ldlt(matrix: &DenseMatrix) -> Result<(DenseMatrix, Vector), String> {
    if matrix.rows != matrix.columns { return Err("LDL^T requires a square matrix".into()); }
    let n = matrix.rows;
    let mut l = vec![0.0; n * n];
    let mut d = vec![0.0; n];
    for index in 0..n { l[index * n + index] = 1.0; }
    for row in 0..n {
        for column in 0..row {
            let mut value = matrix.data[row * n + column];
            for k in 0..column { value -= l[row * n + k] * d[k] * l[column * n + k]; }
            if d[column].abs() <= f64::EPSILON { return Err("zero LDL^T pivot; pivoted LDL^T required".into()); }
            l[row * n + column] = value / d[column];
        }
        let mut diagonal = matrix.data[row * n + row];
        for k in 0..row { diagonal -= l[row * n + k] * l[row * n + k] * d[k]; }
        if diagonal.abs() <= f64::EPSILON { return Err("singular LDL^T factorization".into()); }
        d[row] = diagonal;
    }
    Ok((DenseMatrix::new(n, n, l)?, Vector::new(d)))
}

/// Factorización QR mediante reflexiones de Householder.
pub fn qr(matrix: &DenseMatrix) -> Result<(DenseMatrix, DenseMatrix), String> {
    let m = matrix.rows;
    let n = matrix.columns;
    let rank = m.min(n);
    let mut r = matrix.data.clone();
    let mut q = identity(m);
    for k in 0..rank {
        let mut norm = 0.0;
        for row in k..m { norm = r[row * n + k].hypot(norm); }
        if norm <= f64::EPSILON { continue; }
        let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = vec![0.0; m - k];
        for row in k..m { v[row - k] = r[row * n + k]; }
        v[0] += sign * norm;
        let v_norm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
        if v_norm <= f64::EPSILON { continue; }
        for value in &mut v { *value /= v_norm; }
        for column in k..n {
            let mut projection = 0.0;
            for row in k..m { projection += v[row - k] * r[row * n + column]; }
            for row in k..m { r[row * n + column] -= 2.0 * v[row - k] * projection; }
        }
        for row in 0..m {
            let mut projection = 0.0;
            for column in k..m { projection += q.data[row * m + column] * v[column - k]; }
            for column in k..m { q.data[row * m + column] -= 2.0 * projection * v[column - k]; }
        }
    }
    let q_matrix = q;
    let mut r_data = vec![0.0; m * n];
    for row in 0..m { for column in 0..n { if row <= column { r_data[row * n + column] = r[row * n + column]; } } }
    Ok((q_matrix, DenseMatrix::new(m, n, r_data)?))
}

/// SVD propia basada en iteraciones Jacobi de una matriz de columnas.
pub fn svd(matrix: &DenseMatrix, _policy: RuntimePolicy) -> Result<(DenseMatrix, Vector, DenseMatrix), String> {
    let m = matrix.rows;
    let n = matrix.columns;
    let rank = m.min(n);
    let mut a = matrix.data.clone();
    let mut v = identity(n);
    let tolerance = 1e-12;
    let max_sweeps = 32usize.max(8 * n.max(1));

    for _ in 0..max_sweeps {
        let mut changed = false;
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for row in 0..m {
                    let ap = a[row * n + p];
                    let aq = a[row * n + q];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                if gamma.abs() <= tolerance * (alpha * beta).sqrt().max(1.0) { continue; }
                changed = true;
                let tau = (beta - alpha) / (2.0 * gamma);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for row in 0..m {
                    let p_index = row * n + p;
                    let q_index = row * n + q;
                    let ap = a[p_index];
                    let aq = a[q_index];
                    a[p_index] = c * ap - s * aq;
                    a[q_index] = s * ap + c * aq;
                }
                for row in 0..n {
                    let p_index = row * n + p;
                    let q_index = row * n + q;
                    let vp = v.data[p_index];
                    let vq = v.data[q_index];
                    v.data[p_index] = c * vp - s * vq;
                    v.data[q_index] = s * vp + c * vq;
                }
            }
        }
        if !changed { break; }
    }

    let mut singular_values: Vec<(f64, usize)> = (0..n)
        .map(|column| {
            let norm = (0..m).map(|row| a[row * n + column] * a[row * n + column]).sum::<f64>().sqrt();
            (norm, column)
        })
        .collect();
    singular_values.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut s = vec![0.0; rank];
    let mut u = vec![0.0; m * rank];
    let mut sorted_v = vec![0.0; n * rank];
    for output_column in 0..rank {
        let (sigma, source_column) = singular_values[output_column];
        s[output_column] = sigma;
        for row in 0..n { sorted_v[row * rank + output_column] = v.data[row * n + source_column]; }
        if sigma > f64::EPSILON {
            for row in 0..m { u[row * rank + output_column] = a[row * n + source_column] / sigma; }
        }
    }
    Ok((DenseMatrix::new(m, rank, u)?, Vector::new(s), DenseMatrix::new(n, rank, sorted_v)?))
}

/// QR con pivoteo de columnas para revelar el rango numérico.
///
/// Devuelve `A P = Q R`, la permutación y una estimación de rango basada en
/// el mayor valor diagonal de R.
pub fn rank_revealing_qr(matrix: &DenseMatrix) -> Result<(DenseMatrix, DenseMatrix, Vec<usize>, usize), String> {
    let m = matrix.rows;
    let n = matrix.columns;
    let rank_limit = m.min(n);
    let mut a = matrix.data.clone();
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut q = identity(m);
    let mut column_norms = vec![0.0; n];
    for column in 0..n {
        column_norms[column] = (0..m).map(|row| a[row * n + column].powi(2)).sum::<f64>().sqrt();
    }
    let mut diagonal = vec![0.0; rank_limit];

    for k in 0..rank_limit {
        let mut pivot = k;
        for column in k + 1..n { if column_norms[column] > column_norms[pivot] { pivot = column; } }
        if pivot != k {
            for row in 0..m { a.swap(row * n + k, row * n + pivot); }
            column_norms.swap(k, pivot);
            permutation.swap(k, pivot);
        }
        let mut norm = 0.0;
        for row in k..m { norm = a[row * n + k].hypot(norm); }
        diagonal[k] = norm;
        if norm <= f64::EPSILON { continue; }
        let sign = if a[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        let mut v = vec![0.0; m - k];
        for row in k..m { v[row - k] = a[row * n + k]; }
        v[0] += sign * norm;
        let vnorm = v.iter().map(|value| value * value).sum::<f64>().sqrt();
        if vnorm <= f64::EPSILON { continue; }
        for value in &mut v { *value /= vnorm; }
        for column in k..n {
            let mut projection = 0.0;
            for row in k..m { projection += v[row - k] * a[row * n + column]; }
            for row in k..m { a[row * n + column] -= 2.0 * v[row - k] * projection; }
        }
        for row in 0..m {
            let mut projection = 0.0;
            for column in k..m { projection += q.data[row * m + column] * v[column - k]; }
            for column in k..m { q.data[row * m + column] -= 2.0 * projection * v[column - k]; }
        }
        for column in k + 1..n {
            column_norms[column] = (k + 1..m).map(|row| a[row * n + column].powi(2)).sum::<f64>().sqrt();
        }
    }

    let threshold = diagonal.iter().copied().fold(0.0_f64, f64::max) * 1e-12;
    let numerical_rank = diagonal.iter().filter(|&&value| value > threshold.max(f64::EPSILON)).count();
    let mut r_data = vec![0.0; rank_limit * n];
    for row in 0..rank_limit { for column in row..n { r_data[row * n + column] = a[row * n + column]; } }
    let mut q_thin_data = vec![0.0; m * rank_limit];
    for row in 0..m { for column in 0..rank_limit { q_thin_data[row * rank_limit + column] = q.data[row * m + column]; } }
    let q_thin = DenseMatrix::new(m, rank_limit, q_thin_data)?;
    Ok((q_thin, DenseMatrix::new(rank_limit, n, r_data)?, permutation, numerical_rank))
}

/// Descomposición polar `A = U P` para matrices con al menos tantas filas como columnas.
///
/// Se construye desde la SVD propia para reutilizar la misma ruta numérica sin
/// depender de una implementación externa de álgebra lineal.
pub fn polar(matrix: &DenseMatrix, policy: RuntimePolicy) -> Result<(DenseMatrix, DenseMatrix), String> {
    if matrix.rows < matrix.columns {
        return Err("polar currently requires rows >= columns".into());
    }
    let (u, singular_values, v) = svd(matrix, policy)?;
    let rank = singular_values.len();
    let mut unitary = vec![0.0; matrix.rows * matrix.columns];
    for row in 0..matrix.rows {
        for column in 0..matrix.columns {
            let mut value = 0.0;
            for k in 0..rank { value += u.data[row * rank + k] * v.data[column * rank + k]; }
            unitary[row * matrix.columns + column] = value;
        }
    }
    let mut positive = vec![0.0; matrix.columns * matrix.columns];
    for row in 0..matrix.columns {
        for column in 0..matrix.columns {
            let mut value = 0.0;
            for k in 0..rank { value += v.data[row * rank + k] * singular_values.data[k] * v.data[column * rank + k]; }
            positive[row * matrix.columns + column] = value;
        }
    }
    Ok((DenseMatrix::new(matrix.rows, matrix.columns, unitary)?, DenseMatrix::new(matrix.columns, matrix.columns, positive)?))
}

/// Autovalores y autovectores de una matriz simétrica mediante Jacobi.
pub fn symmetric_eigen(matrix: &DenseMatrix, _policy: RuntimePolicy) -> Result<(Vector, DenseMatrix), String> {
    if matrix.rows != matrix.columns { return Err("symmetric eigen decomposition requires a square matrix".into()); }
    let n = matrix.rows;
    let mut a = matrix.data.clone();
    let mut vectors = identity(n);
    for _ in 0..64usize.max(8 * n.max(1)) {
        let mut p = 0;
        let mut q = 0;
        let mut maximum = 0.0;
        for row in 0..n { for column in row + 1..n {
            let value = a[row * n + column].abs();
            if value > maximum { maximum = value; p = row; q = column; }
        }}
        if maximum <= 1e-12 { break; }
        let theta = 0.5 * (2.0 * a[p * n + q]).atan2(a[q * n + q] - a[p * n + p]);
        let c = theta.cos();
        let s = theta.sin();
        for k in 0..n {
            let apk = a[p * n + k];
            let aqk = a[q * n + k];
            a[p * n + k] = c * apk - s * aqk;
            a[q * n + k] = s * apk + c * aqk;
        }
        for k in 0..n {
            let akp = a[k * n + p];
            let akq = a[k * n + q];
            a[k * n + p] = c * akp - s * akq;
            a[k * n + q] = s * akp + c * akq;
            let vkp = vectors.data[k * n + p];
            let vkq = vectors.data[k * n + q];
            vectors.data[k * n + p] = c * vkp - s * vkq;
            vectors.data[k * n + q] = s * vkp + c * vkq;
        }
    }
    let mut eigenvalues: Vec<(f64, usize)> = (0..n).map(|index| (a[index * n + index], index)).collect();
    eigenvalues.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut values = vec![0.0; n];
    let mut sorted_vectors = vec![0.0; n * n];
    for column in 0..n {
        values[column] = eigenvalues[column].0;
        let source = eigenvalues[column].1;
        for row in 0..n { sorted_vectors[row * n + column] = vectors.data[row * n + source]; }
    }
    Ok((Vector::new(values), DenseMatrix::new(n, n, sorted_vectors)?))
}

/// Reducción de Hessenberg por reflexiones de Householder.
pub fn hessenberg(matrix: &DenseMatrix) -> Result<(DenseMatrix, DenseMatrix), String> {
    if matrix.rows != matrix.columns { return Err("Hessenberg reduction requires a square matrix".into()); }
    let n = matrix.rows;
    let mut h = matrix.data.clone();
    let mut q = identity(n);
    for k in 0..n.saturating_sub(2) {
        let length = n - k - 1;
        let mut vector = vec![0.0; length];
        for row in 0..length { vector[row] = h[(k + 1 + row) * n + k]; }
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm <= f64::EPSILON { continue; }
        let sign = if vector[0] >= 0.0 { 1.0 } else { -1.0 };
        vector[0] += sign * norm;
        let vector_norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        for value in &mut vector { *value /= vector_norm; }
        for column in k..n {
            let mut projection = 0.0;
            for row in 0..length { projection += vector[row] * h[(k + 1 + row) * n + column]; }
            for row in 0..length { h[(k + 1 + row) * n + column] -= 2.0 * vector[row] * projection; }
        }
        for row in 0..n {
            let mut projection = 0.0;
            for column in 0..length { projection += q.data[row * n + (k + 1 + column)] * vector[column]; }
            for column in 0..length { q.data[row * n + (k + 1 + column)] -= 2.0 * projection * vector[column]; }
        }
        for row in 0..n {
            let mut projection = 0.0;
            for column in 0..length { projection += h[row * n + (k + 1 + column)] * vector[column]; }
            for column in 0..length { h[row * n + (k + 1 + column)] -= 2.0 * projection * vector[column]; }
        }
    }
    Ok((q, DenseMatrix::new(n, n, h)?))
}

fn identity(size: usize) -> DenseMatrix {
    let mut data = vec![0.0; size * size];
    for index in 0..size { data[index * size + index] = 1.0; }
    DenseMatrix::new(size, size, data).expect("identity dimensions are valid")
}

fn solve_lower_raw(matrix: &DenseMatrix, rhs: &[f64], unit_diagonal: bool) -> Result<Vec<f64>, String> {
    let n = matrix.rows;
    let mut result = rhs.to_vec();
    for row in 0..n {
        let offset = row * n;
        for column in 0..row { result[row] -= matrix.data[offset + column] * result[column]; }
        if !unit_diagonal {
            let diagonal = matrix.data[offset + row];
            if diagonal.abs() <= f64::EPSILON { return Err("singular lower factor".into()); }
            result[row] /= diagonal;
        }
    }
    Ok(result)
}

fn solve_upper_raw(matrix: &DenseMatrix, rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = matrix.rows;
    let mut result = rhs.to_vec();
    for row in (0..n).rev() {
        let offset = row * n;
        for column in row + 1..n { result[row] -= matrix.data[offset + column] * result[column]; }
        let diagonal = matrix.data[offset + row];
        if diagonal.abs() <= f64::EPSILON { return Err("singular upper factor".into()); }
        result[row] /= diagonal;
    }
    Ok(result)
}
