//! Kernels numericos CPU.
//!
//! Las operaciones grandes dividen el trabajo por filas para conservar acceso
//! contiguo al layout row-major. GEMM usa blocking para aumentar reutilizacion
//! de cache y evita el patron ingenuo i-k-j sin tiling cuando el problema crece.

use rayon::prelude::*;

use crate::{CscMatrix, CsrMatrix, DenseMatrix, EllpackMatrix, Vector};
use crate::runtime::RuntimePolicy;
use crate::simd;

/// Suma dos vectores.
pub fn vector_add(left: &Vector, right: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    validate_same_length(left, right)?;
    let mut result = vec![0.0; left.len()];
    let use_simd = policy.plan_for(left.len()).use_simd;
    parallel_chunks(left.len(), policy, &mut result, |output, start, end| {
        simd::add(&left.data[start..end], &right.data[start..end], output, plan_avx512(&policy, left.len()), use_simd);
    });
    Ok(Vector::new(result))
}

/// Escala un vector por un escalar.
pub fn vector_scale(vector: &Vector, scalar: f64, policy: RuntimePolicy) -> Vector {
    let mut result = vec![0.0; vector.len()];
    let use_simd = policy.plan_for(vector.len()).use_simd;
    parallel_chunks(vector.len(), policy, &mut result, |output, start, end| {
        simd::scale(&vector.data[start..end], scalar, output, plan_avx512(&policy, vector.len()), use_simd);
    });
    Vector::new(result)
}

/// Producto escalar con reduccion determinista por bloques.
pub fn dot_product(left: &Vector, right: &Vector, policy: RuntimePolicy) -> Result<f64, String> {
    validate_same_length(left, right)?;
    if left.is_empty() { return Ok(0.0); }
    let plan = policy.plan_for(left.len());
    if plan.threads == 1 {
        let capabilities = policy.plan_for(left.len()).capabilities;
        return Ok(simd::dot(&left.data, &right.data, capabilities.avx512f, capabilities.avx2 || capabilities.wasm_simd128, capabilities.fma));
    }
    let chunk = (left.len() + plan.threads - 1) / plan.threads;
    let simd_enabled = plan.capabilities.avx2 || plan.capabilities.wasm_simd128;
    let fma_enabled = plan.capabilities.fma;
    let partials: Vec<f64> = left.data.par_chunks(chunk)
        .zip(right.data.par_chunks(chunk))
        .map(|(left_chunk, right_chunk)| simd::dot(left_chunk, right_chunk, plan.capabilities.avx512f, simd_enabled, fma_enabled))
        .collect();
    Ok(partials.into_iter().sum())
}

/// Norma euclidea del vector.
pub fn vector_norm2(vector: &Vector, policy: RuntimePolicy) -> f64 {
    dot_product(vector, vector, policy).unwrap_or(0.0).sqrt()
}

/// Multiplicacion matriz-vector para dense, CSR o ELLPACK.
pub fn matrix_vector_multiply(matrix: &DenseMatrix, vector: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    if matrix.columns != vector.len() { return Err("matrix columns must match vector length".into()); }
    let mut result = vec![0.0; matrix.rows];
    let capabilities = policy.plan_for(matrix.columns);
    parallel_chunks(matrix.rows, policy, &mut result, |output, start, end| {
        for row in start..end {
            let matrix_offset = row * matrix.columns;
            output[row - start] = simd::dot(&matrix.data[matrix_offset..matrix_offset + matrix.columns], &vector.data, capabilities.capabilities.avx512f, capabilities.capabilities.avx2 || capabilities.capabilities.wasm_simd128, capabilities.capabilities.fma);
        }
    });
    Ok(Vector::new(result))
}

/// Multiplicacion matriz-vector con seleccion por representacion.
pub fn matrix_vector_multiply_any(matrix: &crate::Matrix, vector: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    match matrix {
        crate::Matrix::Dense(matrix) => matrix_vector_multiply(matrix, vector, policy),
        crate::Matrix::Csr(matrix) => csr_matrix_vector_multiply(matrix, vector, policy),
        crate::Matrix::Csc(matrix) => csc_matrix_vector_multiply(matrix, vector, policy),
        crate::Matrix::Ellpack(matrix) => ellpack_matrix_vector_multiply(matrix, vector, policy),
    }
}

/// Multiplicacion CSR por vector.
pub fn csr_matrix_vector_multiply(matrix: &CsrMatrix, vector: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    if matrix.columns != vector.len() { return Err("matrix columns must match vector length".into()); }
    let mut result = vec![0.0; matrix.rows];
    let capabilities = policy.plan_for(matrix.columns);
    parallel_chunks(matrix.rows, policy, &mut result, |output, start, end| {
        for row in start..end {
            let mut accumulator = 0.0;
            for index in matrix.row_pointers[row]..matrix.row_pointers[row + 1] {
                accumulator += matrix.values[index] * vector.data[matrix.column_indices[index]];
            }
            output[row - start] = accumulator;
        }
    });
    Ok(Vector::new(result))
}

/// Multiplicación CSC por vector.
pub fn csc_matrix_vector_multiply(matrix: &CscMatrix, vector: &Vector, _policy: RuntimePolicy) -> Result<Vector, String> {
    if matrix.columns != vector.len() { return Err("matrix columns must match vector length".into()); }
    let mut result = vec![0.0; matrix.rows];
    for column in 0..matrix.columns {
        let value = vector.data[column];
        for index in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            result[matrix.row_indices[index]] += matrix.values[index] * value;
        }
    }
    Ok(Vector::new(result))
}

/// Multiplicación ELLPACK por vector.
pub fn ellpack_matrix_vector_multiply(matrix: &EllpackMatrix, vector: &Vector, policy: RuntimePolicy) -> Result<Vector, String> {
    if matrix.columns != vector.len() { return Err("matrix columns must match vector length".into()); }
    let mut result = vec![0.0; matrix.rows];
    let capabilities = policy.plan_for(matrix.columns);
    parallel_chunks(matrix.rows, policy, &mut result, |output, start, end| {
        for row in start..end {
            let offset = row * matrix.max_non_zeros_per_row;
            let mut accumulator = 0.0;
            for slot in 0..matrix.max_non_zeros_per_row {
                let column = matrix.column_indices[offset + slot];
                if column >= 0 { accumulator += matrix.values[offset + slot] * vector.data[column as usize]; }
            }
            output[row - start] = accumulator;
        }
    });
    Ok(Vector::new(result))
}

/// Multiplicacion GEMM densa con blocking y paralelismo por bloques de filas.
pub fn matrix_matrix_multiply(left: &DenseMatrix, right: &DenseMatrix, policy: RuntimePolicy) -> Result<DenseMatrix, String> {
    if left.columns != right.rows { return Err("matrix dimensions are incompatible".into()); }
    let rows = left.rows;
    let columns = right.columns;
    let shared = left.columns;
    let mut result = vec![0.0; rows * columns];
    let block = choose_gemm_block(rows.max(columns).max(shared), policy.plan_for(rows * columns).capabilities);
    let plan = policy.plan_for(rows * columns);

    if plan.threads == 1 {
        gemm_blocked_range(left, right, &mut result, 0, rows, block, policy.plan_for(rows * columns).capabilities);
        return DenseMatrix::new(rows, columns, result);
    }

    let chunk_rows = (rows + plan.threads - 1) / plan.threads;
    result.par_chunks_mut(chunk_rows * columns).enumerate().for_each(|(chunk_index, result_slice)| {
        let start = chunk_index * chunk_rows;
        let end = start + result_slice.len() / columns;
        gemm_blocked_range_into(left, right, result_slice, start, end, block, policy.plan_for(rows * columns).capabilities);
    });
    DenseMatrix::new(rows, columns, result)
}

fn gemm_blocked_range(left: &DenseMatrix, right: &DenseMatrix, result: &mut [f64], start_row: usize, end_row: usize, block: usize, capabilities: crate::CpuCapabilities) {
    gemm_blocked_range_into(left, right, &mut result[start_row * right.columns..end_row * right.columns], start_row, end_row, block, capabilities);
}

fn gemm_blocked_range_into(left: &DenseMatrix, right: &DenseMatrix, result: &mut [f64], start_row: usize, end_row: usize, block: usize, capabilities: crate::CpuCapabilities) {
    let columns = right.columns;
    let mut packed_right = vec![0.0; block * block];
    for row_block in (start_row..end_row).step_by(block) {
        let row_end = (row_block + block).min(end_row);
        for shared_block in (0..left.columns).step_by(block) {
            let shared_end = (shared_block + block).min(left.columns);
            let shared_width = shared_end - shared_block;
            for column_block in (0..columns).step_by(block) {
                let column_end = (column_block + block).min(columns);
                let column_width = column_end - column_block;

                // Empaquetado de panel B para que el micro-kernel lea memoria
                // secuencial durante el bucle de producto interno.
                for column in 0..column_width {
                    for shared in 0..shared_width {
                        packed_right[column * shared_width + shared] =
                            right.data[(shared_block + shared) * columns + column_block + column];
                    }
                }

                for row in row_block..row_end {
                    let left_row = row * left.columns;
                    let result_row = (row - start_row) * columns;
                    for column in 0..column_width {
                        let left_panel = &left.data[left_row + shared_block..left_row + shared_end];
                        let right_panel = &packed_right[column * shared_width..column * shared_width + shared_width];
                        let accumulator = result[result_row + column_block + column];
                        let dot = simd::dot(
                            left_panel,
                            right_panel,
                            capabilities.avx512f,
                            capabilities.avx2 || capabilities.wasm_simd128,
                            capabilities.fma,
                        );
                        result[result_row + column_block + column] = accumulator + dot;
                    }
                }
            }
        }
    }
}

fn choose_gemm_block(size: usize, capabilities: crate::CpuCapabilities) -> usize {
    // El tamaño se ajusta a la capacidad SIMD y al working set esperado. No se
    // usa una dependencia BLAS: esta es la heurística propia de TRACE.
    if capabilities.avx512f {
        if size >= 2048 { 128 } else if size >= 512 { 96 } else { 48 }
    } else if capabilities.avx2 || capabilities.neon {
        if size >= 2048 { 96 } else if size >= 512 { 64 } else { 32 }
    } else if size >= 1024 {
        64
    } else {
        32
    }
}

fn plan_avx512(policy: &RuntimePolicy, elements: usize) -> bool {
    policy.plan_for(elements).capabilities.avx512f
}

fn validate_same_length(left: &Vector, right: &Vector) -> Result<(), String> {
    if left.len() != right.len() { Err("vectors must have the same length".into()) } else { Ok(()) }
}

fn dot_range(left: &[f64], right: &[f64], start: usize, end: usize) -> f64 {
    let mut accumulator = 0.0;
    for index in start..end { accumulator += left[index] * right[index]; }
    accumulator
}

fn parallel_chunks<F>(elements: usize, policy: RuntimePolicy, result: &mut [f64], operation: F)
where
    F: Fn(&mut [f64], usize, usize) + Sync + Send,
{
    if elements == 0 { return; }
    let plan = policy.plan_for(elements);
    if plan.threads == 1 {
        operation(result, 0, elements);
        return;
    }
    let chunk = (elements + plan.threads - 1) / plan.threads;
    result.par_chunks_mut(chunk).enumerate().for_each(|(chunk_index, output)| {
        let start = chunk_index * chunk;
        let end = start + output.len();
        operation(output, start, end);
    });
}

#[inline]
fn detect_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("avx2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[inline]
fn detect_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("fma") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}
