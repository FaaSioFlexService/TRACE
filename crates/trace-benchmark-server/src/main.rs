//! Servidor HTTP nativo para el laboratorio TRACE.
//!
//! El servidor expone el mismo catálogo matemático que WASM, ejecutando los
//! kernels propios de Rust. No se usa BLAS/LAPACK ni se reporta una ejecución
//! GPU hasta que exista un executor GPU real para esa operación.

use axum::{extract::State, http::StatusCode, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use trace_numerical_core::{
    axpy, cholesky, cross_product, determinant, gemv, ger, hessenberg, inverse, kronecker_product,
    ldlt, lu, matrix_matrix_multiply, matrix_norm_frobenius, matrix_norm_inf, matrix_norm_one,
    matrix_norm_two, matrix_vector_multiply_any, qr, solve_linear_system, solve_multiple_rhs,
    syrk, trace, transpose, triangular_solve, trsm, vector_add, vector_norm2, vector_scale,
    dot_product, DenseMatrix, ExecutionMode, Matrix, RuntimePolicy, Vector,
    symmetric_eigen,
};

mod backend;

#[derive(Clone)]
struct AppState {
    gpu: trace_gpu::GpuDetection,
}

#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    operation: String,
    execution: String,
    vector: Option<Vec<f64>>,
    other_vector: Option<Vec<f64>>,
    matrix: Option<MatrixPayload>,
    right_matrix: Option<MatrixPayload>,
    scalar: Option<f64>,
    lower: Option<bool>,
    unit_diagonal: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct MatrixPayload {
    rows: usize,
    columns: usize,
    data: Vec<f64>,
    format: Option<String>,
    zero_tolerance: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ExecuteResponse {
    operation: String,
    execution: String,
    backend: String,
    elapsed_ns: u128,
    checksum: f64,
    values: Vec<f64>,
}

/// Punto de entrada del servidor nativo.
#[tokio::main]
async fn main() {
    let gpu = trace_gpu::detect();
    let state = Arc::new(AppState { gpu });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/execute", post(execute))
        .with_state(state);

    let address: SocketAddr = "0.0.0.0:8080".parse().expect("valid server address");
    let listener = tokio::net::TcpListener::bind(address).await.expect("bind server socket");
    println!("TRACE Benchmark Server listening on http://{address}");
    axum::serve(listener, app).await.expect("serve HTTP requests");
}

/// Endpoint de disponibilidad del servicio.
async fn health() -> &'static str { "ok" }

/// Expone las capacidades detectadas para reproducibilidad.
async fn capabilities(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let cpu = trace_numerical_core::CpuCapabilities::detect();
    let mut native_backends = backend::registry();
    for native_backend in &mut native_backends {
        if native_backend.id == "trace-rust-native-gpu" {
            native_backend.available = state.gpu.available;
        }
    }
    Json(serde_json::json!({
        "cpu": {
            "architecture": cpu.architecture,
            "logicalThreads": cpu.logical_threads,
            "avx2": cpu.avx2,
            "avx512f": cpu.avx512f,
            "fma": cpu.fma,
            "neon": cpu.neon,
            "wasmSimd128": cpu.wasm_simd128
        },
        "gpu": state.gpu,
        "nativeBackends": native_backends
    }))
}

/// Ejecuta una operación propia en CPU nativa y devuelve solo el coste de kernel.
async fn execute(Json(request): Json<ExecuteRequest>) -> Result<Json<ExecuteResponse>, (StatusCode, Json<serde_json::Value>)> {
    if request.execution == "gpu" {
        return Err((StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
            "error": "GPU execution is detected and exposed as a capability, but the production GPU executor is not enabled for this operation."
        }))));
    }

    let policy = RuntimePolicy {
        mode: match request.execution.as_str() {
            "cpu" => ExecutionMode::CpuParallel,
            "cpu-single" => ExecutionMode::CpuOnly,
            _ => ExecutionMode::Auto,
        },
        ..RuntimePolicy::default()
    };

    let operation = request.operation.clone();
    let started = std::time::Instant::now();
    let result = execute_cpu_operation(&request, policy);
    let elapsed_ns = started.elapsed().as_nanos();
    let values = match result {
        Ok(values) => values,
        Err(error) => {
            return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": error }))));
        }
    };
    let checksum = values.iter().fold(0.0, |sum, value| sum + value);

    Ok(Json(ExecuteResponse {
        operation,
        execution: request.execution,
        backend: "trace-rust-native-cpu".into(),
        elapsed_ns,
        checksum,
        values,
    }))
}

/// Ejecuta una operación y aplana sus resultados para el protocolo HTTP.
fn execute_cpu_operation(request: &ExecuteRequest, policy: RuntimePolicy) -> Result<Vec<f64>, String> {
    let vector = || Vector::new(request.vector.clone().unwrap_or_default());
    let other_vector = || Vector::new(request.other_vector.clone().unwrap_or_default());
    let dense = || matrix_from_payload(request.matrix.as_ref());
    let right = || matrix_from_payload(request.right_matrix.as_ref());

    match request.operation.as_str() {
        "vectorAdd" => Ok(vector_add(&vector(), &other_vector(), policy)?.data),
        "vectorScale" => Ok(vector_scale(&vector(), request.scalar.unwrap_or(1.0), policy).data),
        "axpy" => Ok(axpy(request.scalar.unwrap_or(1.0), &vector(), &other_vector(), policy)?.data),
        "crossProduct" => Ok(cross_product(&vector(), &other_vector())?.data),
        "dotProduct" => Ok(vec![dot_product(&vector(), &other_vector(), policy)?]),
        "vectorNorm2" => Ok(vec![vector_norm2(&vector(), policy)]),
        "gemv" => {
            let matrix = dense()?;
            let y = Vector::new(vec![0.0; matrix.rows]);
            Ok(gemv(request.scalar.unwrap_or(1.0), &matrix, &vector(), 0.0, &y, policy)?.data)
        }
        "matrixVectorMultiply" => {
            let payload = request.matrix.as_ref().ok_or("matrix is required")?;
            let dense_matrix = DenseMatrix::new(payload.rows, payload.columns, payload.data.clone())?;
            let tolerance = payload.zero_tolerance.unwrap_or(0.0);
            let matrix = match payload.format.as_deref() {
                Some("csr") => Matrix::Csr(trace_numerical_core::dense_to_csr(&dense_matrix, tolerance)),
                Some("csc") => Matrix::Csc(trace_numerical_core::dense_to_csc(&dense_matrix, tolerance)),
                Some("ellpack") => Matrix::Ellpack(trace_numerical_core::dense_to_ellpack(&dense_matrix, tolerance)),
                _ => Matrix::Dense(dense_matrix),
            };
            Ok(matrix_vector_multiply_any(&matrix, &vector(), policy)?.data)
        }
        "triangularSolve" => Ok(triangular_solve(&dense()?, &vector(), request.lower.unwrap_or(true), request.unit_diagonal.unwrap_or(false))?.data),
        "ger" => Ok(ger(request.scalar.unwrap_or(1.0), &vector(), &other_vector(), &dense()?, policy)?.data),
        "matrixMatrixMultiply" => Ok(matrix_matrix_multiply(&dense()?, &right()?, policy)?.data),
        "trsm" => Ok(trsm(&dense()?, &right()?, request.lower.unwrap_or(true), request.unit_diagonal.unwrap_or(false))?.data),
        "syrk" => Ok(syrk(request.scalar.unwrap_or(1.0), &dense()?, 1.0, &right()?, policy)?.data),
        "transpose" => Ok(transpose(&dense()?)?.data),
        "trace" => Ok(vec![trace(&dense()?)?]),
        "matrixNormFrobenius" => Ok(vec![matrix_norm_frobenius(&dense()?, policy)]),
        "matrixNormOne" => Ok(vec![matrix_norm_one(&dense()?)]),
        "matrixNormInf" => Ok(vec![matrix_norm_inf(&dense()?)]),
        "matrixNormTwo" => Ok(vec![matrix_norm_two(&dense()?, policy)?]),
        "kroneckerProduct" => {
            let identity = DenseMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0])?;
            Ok(kronecker_product(&dense()?, &identity)?.data)
        }
        "inverse" => Ok(inverse(&dense()?, policy)?.data),
        "determinant" => Ok(vec![determinant(&dense()?, policy)?]),
        "solveLinearSystem" => Ok(solve_linear_system(&dense()?, &vector(), policy)?.data),
        "solveMultipleRhs" => Ok(solve_multiple_rhs(&dense()?, &right()?, policy)?.data),
        "lu" => {
            let factors = lu(&dense()?, policy)?;
            let mut values = factors.l.data;
            values.extend(factors.u.data);
            values.extend(factors.permutation.into_iter().map(|value| value as f64));
            Ok(values)
        }
        "cholesky" => Ok(cholesky(&dense()?)?.data),
        "ldlt" => {
            let (l, d) = ldlt(&dense()?)?;
            let mut values = l.data;
            values.extend(d.data);
            Ok(values)
        }
        "qr" => {
            let (q, r) = qr(&dense()?)?;
            let mut values = q.data;
            values.extend(r.data);
            Ok(values)
        }
        "rankRevealing" => {
            let (q, r, permutation, rank) = trace_numerical_core::rank_revealing_qr(&dense()?)?;
            let mut values = q.data;
            values.extend(r.data);
            values.extend(permutation.into_iter().map(|value| value as f64));
            values.push(rank as f64);
            Ok(values)
        }
        "svd" => {
            let (u, singular_values, v) = trace_numerical_core::svd(&dense()?, policy)?;
            let mut values = u.data;
            values.extend(singular_values.data);
            values.extend(v.data);
            Ok(values)
        }
        "polar" => {
            let (u, p) = trace_numerical_core::polar(&dense()?, policy)?;
            let mut values = u.data;
            values.extend(p.data);
            Ok(values)
        }
        "symmetricEigen" => {
            let (values, vectors) = symmetric_eigen(&dense()?, policy)?;
            let mut output = values.data;
            output.extend(vectors.data);
            Ok(output)
        }
        "hessenberg" => {
            let (q, h) = hessenberg(&dense()?)?;
            let mut output = q.data;
            output.extend(h.data);
            Ok(output)
        }
        "bunchKaufman" | "eigen" | "schur" | "jordan" | "frobenius" | "hermite" | "smith" | "popov" => {
            Err("operation is present in the catalog but its production kernel is not enabled yet".into())
        }
        _ => Err(format!("unsupported operation: {}", request.operation)),
    }
}

/// Construye una matriz densa validando dimensiones y buffer antes de medir.
fn matrix_from_payload(payload: Option<&MatrixPayload>) -> Result<DenseMatrix, String> {
    let payload = payload.ok_or("matrix is required")?;
    DenseMatrix::new(payload.rows, payload.columns, payload.data.clone())
}
