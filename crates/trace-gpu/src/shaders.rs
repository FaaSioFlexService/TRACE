//! Fuentes WGSL de los kernels GPU de TRACE.
//!
//! Se mantienen separadas del executor wgpu para poder probar y especializar
//! workgroup sizes, tiling y reducciones sin tocar la API del core.

pub const VECTOR: &str = include_str!("../shaders/vector.wgsl");
pub const DENSE: &str = include_str!("../shaders/dense.wgsl");
pub const SPARSE: &str = include_str!("../shaders/sparse.wgsl");
