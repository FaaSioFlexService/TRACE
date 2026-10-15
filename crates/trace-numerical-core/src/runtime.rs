//! Deteccion de capacidades CPU y seleccion del plan de ejecucion.

use std::thread;

/// Modo solicitado por el usuario para el backend nativo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    /// Deja que TRACE elija scalar/SIMD y el grado de paralelismo.
    Auto,
    /// Fuerza CPU sin paralelismo de datos.
    CpuOnly,
    /// Permite CPU paralelo y SIMD, pero no GPU.
    CpuParallel,
}

/// Politica de runtime para decidir kernels y paralelismo.
#[derive(Clone, Copy, Debug)]
pub struct RuntimePolicy {
    pub mode: ExecutionMode,
    pub min_parallel_elements: usize,
    pub max_threads: Option<usize>,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self { mode: ExecutionMode::Auto, min_parallel_elements: 32 * 1024, max_threads: None }
    }
}

/// Capacidades hardware observables sin depender de una libreria externa.
#[derive(Clone, Debug)]
pub struct CpuCapabilities {
    pub architecture: &'static str,
    pub logical_threads: usize,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
    pub wasm_simd128: bool,
}

impl CpuCapabilities {
    /// Detecta las instrucciones SIMD disponibles en el procesador actual.
    pub fn detect() -> Self {
        Self {
            architecture: architecture_name(),
            logical_threads: thread::available_parallelism().map(|value| value.get()).unwrap_or(1),
            avx2: detect_avx2(),
            avx512f: detect_avx512f(),
            fma: detect_fma(),
            neon: detect_neon(),
            wasm_simd128: cfg!(all(target_arch = "wasm32", target_feature = "simd128")),
        }
    }
}

/// Plan concreto elegido para una operacion.
#[derive(Clone, Debug)]
pub struct CpuPlan {
    pub use_simd: bool,
    pub threads: usize,
    pub block_rows: usize,
    pub capabilities: CpuCapabilities,
}

impl RuntimePolicy {
    /// Construye el plan segun tamano y capacidades detectadas.
    pub fn plan_for(&self, elements: usize) -> CpuPlan {
        let capabilities = CpuCapabilities::detect();
        let use_simd = matches!(self.mode, ExecutionMode::Auto | ExecutionMode::CpuParallel)
            && (capabilities.avx512f || capabilities.avx2 || capabilities.wasm_simd128);
        let available_threads = self.max_threads.unwrap_or(capabilities.logical_threads).max(1);
        let threads = if matches!(self.mode, ExecutionMode::CpuOnly) || elements < self.min_parallel_elements {
            1
        } else {
            available_threads.min(elements.max(1))
        };
        let block_rows = (elements / threads.max(1)).max(64);
        CpuPlan { use_simd, threads, block_rows, capabilities }
    }
}

#[inline]
fn architecture_name() -> &'static str {
    if cfg!(target_arch = "x86_64") { "x86_64" }
    else if cfg!(target_arch = "aarch64") { "aarch64" }
    else if cfg!(target_arch = "x86") { "x86" }
    else { "other" }
}

#[inline]
fn detect_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("avx2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[inline]
fn detect_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("avx512f") }
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

#[inline]
fn detect_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    { std::arch::is_aarch64_feature_detected!("neon") }
    #[cfg(not(target_arch = "aarch64"))]
    { false }
}
