//! Registro de backends nativos del servidor.
//!
//! Las implementaciones externas se activan solo cuando la biblioteca esta
//! disponible en la maquina. Nunca se presentan como ejecutables si no existe
//! un adapter real enlazado contra ella.

/// Identificador estable de backend nativo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeBackendId {
    TraceRustCpu,
    TraceRustGpu,
    OpenBlas,
    Blis,
    OneMkl,
    Aocl,
    Eigen,
    ArmPerformanceLibraries,
    Accelerate,
    Cublas,
    Rocblas,
    Nvpl,
}

/// Metadatos de un backend que el servidor puede exponer.
#[derive(Clone, Debug, serde::Serialize)]
pub struct NativeBackendDefinition {
    pub id: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    pub available: bool,
}

/// Construye el catalogo sin fingir disponibilidad de librerias no enlazadas.
pub fn registry() -> Vec<NativeBackendDefinition> {
    vec![
        NativeBackendDefinition { id: "trace-rust-native-cpu", name: "TRACE Rust Native CPU", family: "TRACE", available: true },
        NativeBackendDefinition { id: "trace-rust-native-gpu", name: "TRACE Rust Native GPU", family: "TRACE", available: false },
        NativeBackendDefinition { id: "openblas", name: "OpenBLAS", family: "BLAS/LAPACK", available: cfg!(feature = "openblas") },
        NativeBackendDefinition { id: "blis", name: "BLIS", family: "BLAS", available: cfg!(feature = "blis") },
        NativeBackendDefinition { id: "onemkl", name: "Intel oneMKL", family: "BLAS/LAPACK", available: cfg!(feature = "onemkl") },
        NativeBackendDefinition { id: "aocl", name: "AMD AOCL", family: "BLAS/LAPACK", available: cfg!(feature = "aocl") },
        NativeBackendDefinition { id: "eigen", name: "Eigen", family: "C++ linear algebra", available: cfg!(feature = "eigen") },
        NativeBackendDefinition { id: "armpl", name: "Arm Performance Libraries", family: "ARM HPC", available: cfg!(feature = "armpl") },
        NativeBackendDefinition { id: "accelerate", name: "Apple Accelerate", family: "Apple HPC", available: cfg!(feature = "accelerate") },
        NativeBackendDefinition { id: "cublas", name: "NVIDIA cuBLAS", family: "CUDA", available: cfg!(feature = "cublas") },
        NativeBackendDefinition { id: "rocblas", name: "AMD rocBLAS", family: "ROCm", available: cfg!(feature = "rocblas") },
        NativeBackendDefinition { id: "nvpl", name: "NVIDIA Performance Libraries", family: "NVIDIA HPC", available: cfg!(feature = "nvpl") },
    ]
}
