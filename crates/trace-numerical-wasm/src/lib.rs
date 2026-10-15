//! Bindings WebAssembly del core numérico TRACE.
//!
//! El módulo expone operaciones BLAS-like, factorizaciones y solvers propios.
//! Los buffers cruzan la frontera JS/WASM una sola vez por llamada; la región
//! de cálculo queda completamente dentro de Rust.

use js_sys::Float64Array;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use trace_numerical_core::{
    axpy, cholesky, cross_product, determinant, gemv, ger, hessenberg, inverse, kronecker_product,
    ldlt, lu, polar, matrix_norm_frobenius, matrix_norm_inf, matrix_norm_one, matrix_norm_two,
    matrix_vector_multiply, qr, rank_revealing_qr, solve_linear_system, solve_multiple_rhs, svd, symmetric_eigen,
    syrk, trace, transpose, triangular_solve, trsm, vector_add, vector_norm2, vector_scale,
    dot_product, DenseMatrix, ExecutionMode, RuntimePolicy, Vector,
};

fn cpu_policy() -> RuntimePolicy {
    // El navegador usa la ruta CPU propia. El build wasm puede activar SIMD
    // mediante target-feature=+simd128 cuando el aislamiento del documento lo permite.
    RuntimePolicy { mode: ExecutionMode::CpuOnly, ..RuntimePolicy::default() }
}

fn matrix(rows: usize, columns: usize, data: Float64Array) -> Result<DenseMatrix, JsValue> {
    DenseMatrix::new(rows, columns, data.to_vec()).map_err(|error| JsValue::from_str(&error))
}

fn vector(data: Float64Array) -> Vector { Vector::new(data.to_vec()) }

fn output_vector(result: Vector) -> Float64Array { Float64Array::from(result.data.as_slice()) }

fn output_matrix(result: DenseMatrix) -> Float64Array { Float64Array::from(result.data.as_slice()) }

/// Suma dos vectores f64.
#[wasm_bindgen]
pub fn vector_add_wasm(left: Float64Array, right: Float64Array) -> Result<Float64Array, JsValue> {
    vector_add(&vector(left), &vector(right), cpu_policy()).map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Escala un vector f64.
#[wasm_bindgen]
pub fn vector_scale_wasm(input: Float64Array, scalar: f64) -> Float64Array {
    output_vector(vector_scale(&vector(input), scalar, cpu_policy()))
}

/// Suma escalada `y = alpha*x + y`.
#[wasm_bindgen]
pub fn axpy_wasm(alpha: f64, x: Float64Array, y: Float64Array) -> Result<Float64Array, JsValue> {
    axpy(alpha, &vector(x), &vector(y), cpu_policy()).map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Producto cruz de dos vectores de longitud tres.
#[wasm_bindgen]
pub fn cross_product_wasm(left: Float64Array, right: Float64Array) -> Result<Float64Array, JsValue> {
    cross_product(&vector(left), &vector(right)).map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Producto escalar f64.
#[wasm_bindgen]
pub fn dot_product_wasm(left: Float64Array, right: Float64Array) -> Result<f64, JsValue> {
    dot_product(&vector(left), &vector(right), cpu_policy()).map_err(|error| JsValue::from_str(&error))
}

/// Norma euclídea del vector.
#[wasm_bindgen]
pub fn vector_norm2_wasm(input: Float64Array) -> f64 { vector_norm2(&vector(input), cpu_policy()) }

/// GEMV generalizado.
#[wasm_bindgen]
pub fn gemv_wasm(rows: usize, columns: usize, matrix_data: Float64Array, alpha: f64, x: Float64Array, beta: f64, y: Float64Array) -> Result<Float64Array, JsValue> {
    let matrix = matrix(rows, columns, matrix_data)?;
    gemv(alpha, &matrix, &vector(x), beta, &vector(y), cpu_policy()).map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Multiplicación densa matriz-vector.
#[wasm_bindgen]
pub fn matrix_vector_multiply_wasm(rows: usize, columns: usize, matrix_data: Float64Array, input: Float64Array) -> Result<Float64Array, JsValue> {
    let matrix = matrix(rows, columns, matrix_data)?;
    matrix_vector_multiply(&matrix, &vector(input), cpu_policy()).map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Multiplicación CSR por vector.
#[wasm_bindgen]
pub fn csr_matrix_vector_multiply_wasm(rows: usize, columns: usize, values: Float64Array, column_indices: js_sys::Int32Array, row_pointers: js_sys::Int32Array, input: Float64Array) -> Result<Float64Array, JsValue> {
    let matrix = CsrMatrix::new(
        rows,
        columns,
        values.to_vec(),
        column_indices.to_vec().into_iter().map(|value| value as usize).collect(),
        row_pointers.to_vec().into_iter().map(|value| value as usize).collect(),
    ).map_err(|error| JsValue::from_str(&error))?;
    trace_numerical_core::matrix_vector_multiply_any(&trace_numerical_core::Matrix::Csr(matrix), &vector(input), cpu_policy())
        .map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Multiplicación CSC por vector.
#[wasm_bindgen]
pub fn csc_matrix_vector_multiply_wasm(rows: usize, columns: usize, values: Float64Array, row_indices: js_sys::Int32Array, column_pointers: js_sys::Int32Array, input: Float64Array) -> Result<Float64Array, JsValue> {
    let matrix = CscMatrix::new(
        rows,
        columns,
        values.to_vec(),
        row_indices.to_vec().into_iter().map(|value| value as usize).collect(),
        column_pointers.to_vec().into_iter().map(|value| value as usize).collect(),
    ).map_err(|error| JsValue::from_str(&error))?;
    trace_numerical_core::matrix_vector_multiply_any(&trace_numerical_core::Matrix::Csc(matrix), &vector(input), cpu_policy())
        .map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Multiplicación ELLPACK por vector.
#[wasm_bindgen]
pub fn ellpack_matrix_vector_multiply_wasm(rows: usize, columns: usize, values: Float64Array, column_indices: js_sys::Int32Array, max_non_zeros_per_row: usize, non_zero_count: usize, input: Float64Array) -> Result<Float64Array, JsValue> {
    let matrix = EllpackMatrix::new(
        rows,
        columns,
        values.to_vec(),
        column_indices.to_vec().into_iter().map(|value| value as isize).collect(),
        max_non_zeros_per_row,
        non_zero_count,
    ).map_err(|error| JsValue::from_str(&error))?;
    trace_numerical_core::matrix_vector_multiply_any(&trace_numerical_core::Matrix::Ellpack(matrix), &vector(input), cpu_policy())
        .map(output_vector).map_err(|error| JsValue::from_str(&error))
}

/// Multiplicación densa matriz-matriz.
#[wasm_bindgen]
pub fn matrix_matrix_multiply_wasm(left_rows: usize, left_columns: usize, left_data: Float64Array, right_rows: usize, right_columns: usize, right_data: Float64Array) -> Result<Float64Array, JsValue> {
    let left = matrix(left_rows, left_columns, left_data)?;
    let right = matrix(right_rows, right_columns, right_data)?;
    trace_numerical_core::matrix_matrix_multiply(&left, &right, cpu_policy()).map(output_matrix).map_err(|error| JsValue::from_str(&error))
}

/// Actualización GER.
#[wasm_bindgen]
pub fn ger_wasm(rows: usize, columns: usize, matrix_data: Float64Array, alpha: f64, x: Float64Array, y: Float64Array) -> Result<Float64Array, JsValue> {
    let matrix = matrix(rows, columns, matrix_data)?;
    ger(alpha, &vector(x), &vector(y), &matrix, cpu_policy()).map(output_matrix).map_err(|error| JsValue::from_str(&error))
}

/// Actualización simétrica SYRK.
#[wasm_bindgen]
pub fn syrk_wasm(a_rows: usize, a_columns: usize, a_data: Float64Array, c_size: usize, c_data: Float64Array, alpha: f64, beta: f64) -> Result<Float64Array, JsValue> {
    let a = matrix(a_rows, a_columns, a_data)?;
    let c = matrix(c_size, c_size, c_data)?;
    syrk(alpha, &a, beta, &c, cpu_policy()).map(output_matrix).map_err(|error| JsValue::from_str(&error))
}

/// Transposición densa out-of-place.
#[wasm_bindgen]
pub fn transpose_wasm(rows: usize, columns: usize, data: Float64Array) -> Result<Float64Array, JsValue> {
    transpose(&matrix(rows, columns, data)?).map(output_matrix).map_err(|error| JsValue::from_str(&error))
}

/// Traza de una matriz cuadrada.
#[wasm_bindgen]
pub fn trace_wasm(size: usize, data: Float64Array) -> Result<f64, JsValue> { trace(&matrix(size, size, data)?).map_err(|error| JsValue::from_str(&error)) }

/// Norma de Frobenius.
#[wasm_bindgen]
pub fn matrix_norm_frobenius_wasm(rows: usize, columns: usize, data: Float64Array) -> f64 { matrix_norm_frobenius(&DenseMatrix::new(rows, columns, data.to_vec()).expect("valid matrix"), cpu_policy()) }

/// Norma uno.
#[wasm_bindgen]
pub fn matrix_norm_one_wasm(rows: usize, columns: usize, data: Float64Array) -> f64 { matrix_norm_one(&DenseMatrix::new(rows, columns, data.to_vec()).expect("valid matrix")) }

/// Norma infinito.
#[wasm_bindgen]
pub fn matrix_norm_inf_wasm(rows: usize, columns: usize, data: Float64Array) -> f64 { matrix_norm_inf(&DenseMatrix::new(rows, columns, data.to_vec()).expect("valid matrix")) }

/// Norma espectral aproximada.
#[wasm_bindgen]
pub fn matrix_norm_two_wasm(rows: usize, columns: usize, data: Float64Array) -> Result<f64, JsValue> { matrix_norm_two(&matrix(rows, columns, data)?, cpu_policy()).map_err(|error| JsValue::from_str(&error)) }

/// Producto de Kronecker.
#[wasm_bindgen]
pub fn kronecker_product_wasm(left_rows: usize, left_columns: usize, left_data: Float64Array, right_rows: usize, right_columns: usize, right_data: Float64Array) -> Result<Float64Array, JsValue> {
    let left = matrix(left_rows, left_columns, left_data)?;
    let right = matrix(right_rows, right_columns, right_data)?;
    kronecker_product(&left, &right).map(output_matrix).map_err(|error| JsValue::from_str(&error))
}

/// Determinante por LU.
#[wasm_bindgen]
pub fn determinant_wasm(size: usize, data: Float64Array) -> Result<f64, JsValue> { determinant(&matrix(size, size, data)?, cpu_policy()).map_err(|error| JsValue::from_str(&error)) }

/// Inversa por resolución de sistemas sobre la identidad.
#[wasm_bindgen]
pub fn inverse_wasm(size: usize, data: Float64Array) -> Result<Float64Array, JsValue> { inverse(&matrix(size, size, data)?, cpu_policy()).map(output_matrix).map_err(|error| JsValue::from_str(&error)) }

/// Resolución `A*x=b`.
#[wasm_bindgen]
pub fn solve_linear_system_wasm(size: usize, data: Float64Array, b: Float64Array) -> Result<Float64Array, JsValue> { solve_linear_system(&matrix(size, size, data)?, &vector(b), cpu_policy()).map(output_vector).map_err(|error| JsValue::from_str(&error)) }

/// Resolución `A*X=B`.
#[wasm_bindgen]
pub fn solve_multiple_rhs_wasm(size: usize, rhs_columns: usize, data: Float64Array, rhs: Float64Array) -> Result<Float64Array, JsValue> { solve_multiple_rhs(&matrix(size, size, data)?, &matrix(size, rhs_columns, rhs)?, cpu_policy()).map(output_matrix).map_err(|error| JsValue::from_str(&error)) }

/// Resolución triangular.
#[wasm_bindgen]
pub fn triangular_solve_wasm(size: usize, data: Float64Array, b: Float64Array, lower: bool, unit_diagonal: bool) -> Result<Float64Array, JsValue> { triangular_solve(&matrix(size, size, data)?, &vector(b), lower, unit_diagonal).map(output_vector).map_err(|error| JsValue::from_str(&error)) }

/// TRSM con múltiples RHS.
#[wasm_bindgen]
pub fn trsm_wasm(size: usize, rhs_columns: usize, data: Float64Array, rhs: Float64Array, lower: bool, unit_diagonal: bool) -> Result<Float64Array, JsValue> { trsm(&matrix(size, size, data)?, &matrix(size, rhs_columns, rhs)?, lower, unit_diagonal).map(output_matrix).map_err(|error| JsValue::from_str(&error)) }

#[derive(Serialize)]
struct LuOutput { l: Vec<f64>, u: Vec<f64>, permutation: Vec<usize>, pivot_sign: i32 }

#[derive(Serialize)]
struct QrOutput { q: Vec<f64>, r: Vec<f64>, q_rows: usize, q_columns: usize, r_rows: usize, r_columns: usize }

#[derive(Serialize)]
struct SvdOutput { u: Vec<f64>, singular_values: Vec<f64>, v: Vec<f64>, u_rows: usize, u_columns: usize, v_rows: usize, v_columns: usize }

#[derive(Serialize)]
struct EigenOutput { eigenvalues: Vec<f64>, eigenvectors: Vec<f64>, size: usize }

#[derive(Serialize)]
struct HessenbergOutput { q: Vec<f64>, h: Vec<f64>, size: usize }

/// Factorización LU con metadatos de pivoteo.
#[wasm_bindgen]
pub fn lu_wasm(size: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let factors = lu(&matrix(size, size, data)?, cpu_policy()).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&LuOutput { l: factors.l.data, u: factors.u.data, permutation: factors.permutation, pivot_sign: factors.pivot_sign }).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Factorización de Cholesky; devuelve L.
#[wasm_bindgen]
pub fn cholesky_wasm(size: usize, data: Float64Array) -> Result<Float64Array, JsValue> { cholesky(&matrix(size, size, data)?).map(output_matrix).map_err(|error| JsValue::from_str(&error)) }

/// Factorización LDL^T; devuelve L seguido de D.
#[wasm_bindgen]
pub fn ldlt_wasm(size: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (l, d) = ldlt(&matrix(size, size, data)?).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&(l.data, d.data)).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Factorización QR de Householder.
#[wasm_bindgen]
pub fn qr_wasm(rows: usize, columns: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (q, r) = qr(&matrix(rows, columns, data)?).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&QrOutput { q: q.data, r: r.data, q_rows: rows, q_columns: rows, r_rows: rows, r_columns: columns }).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// QR con pivoteo de columnas para revelar el rango.
#[wasm_bindgen]
pub fn rank_revealing_qr_wasm(rows: usize, columns: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (q, r, permutation, rank) = rank_revealing_qr(&matrix(rows, columns, data)?).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&(q.data, r.data, permutation, rank, q.rows, q.columns, r.rows, r.columns)).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// SVD basada en Jacobi unilateral.
#[wasm_bindgen]
pub fn svd_wasm(rows: usize, columns: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (u, s, v) = svd(&matrix(rows, columns, data)?, cpu_policy()).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&SvdOutput { u: u.data, singular_values: s.data, v: v.data, u_rows: u.rows, u_columns: u.columns, v_rows: v.rows, v_columns: v.columns }).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Descomposición polar.
#[wasm_bindgen]
pub fn polar_wasm(rows: usize, columns: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (u, p) = polar(&matrix(rows, columns, data)?, cpu_policy()).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&(u.data, p.data, u.rows, u.columns, p.rows, p.columns)).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Autodescomposición de una matriz simétrica.
#[wasm_bindgen]
pub fn symmetric_eigen_wasm(size: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (values, vectors) = symmetric_eigen(&matrix(size, size, data)?, cpu_policy()).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&EigenOutput { eigenvalues: values.data, eigenvectors: vectors.data, size }).map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Reducción de Hessenberg.
#[wasm_bindgen]
pub fn hessenberg_wasm(size: usize, data: Float64Array) -> Result<JsValue, JsValue> {
    let (q, h) = hessenberg(&matrix(size, size, data)?).map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&HessenbergOutput { q: q.data, h: h.data, size }).map_err(|error| JsValue::from_str(&error.to_string()))
}
