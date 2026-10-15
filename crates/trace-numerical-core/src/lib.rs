//! Core numerico nativo de TRACE.
//!
//! La implementacion mantiene row-major como layout canonico y selecciona
//! automaticamente un plan CPU segun arquitectura, caracteristicas SIMD y
//! numero de hilos disponibles. El plan nunca sacrifica portabilidad por una
//! optimizacion que el procesador no soporte.

mod advanced;
mod decompositions;
mod matrix;
mod operations;
mod runtime;
mod simd;
mod vector;

pub use matrix::{dense_to_csc, dense_to_csr, dense_to_ellpack, CscMatrix, CsrMatrix, DenseMatrix, EllpackMatrix, Matrix};
pub use advanced::{axpy, cross_product, determinant, gemv, ger, inverse, kronecker_product, matrix_norm_frobenius, matrix_norm_inf, matrix_norm_one, matrix_norm_two, solve_linear_system, solve_multiple_rhs, syrk, trace, transpose, triangular_solve, trsm};
pub use decompositions::{cholesky, hessenberg, ldlt, lu, polar, qr, rank_revealing_qr, svd, symmetric_eigen, LuFactors};
pub use operations::{
    dot_product, matrix_matrix_multiply, matrix_vector_multiply, matrix_vector_multiply_any, vector_add, vector_norm2,
    vector_scale,
};
pub use runtime::{CpuCapabilities, CpuPlan, ExecutionMode, RuntimePolicy};
pub use vector::Vector;
