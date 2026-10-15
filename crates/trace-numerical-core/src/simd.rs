//! Kernels SIMD portables por dispatch de CPU.
//!
//! Cada funcion marcada con target_feature solo se invoca despues de detectar
//! la capacidad correspondiente. Asi mantenemos un binario portable sin
//! ejecutar instrucciones no soportadas por el procesador anfitrion.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(left: &[f64], right: &[f64], output: &mut [f64]) {
    let mut index = 0;
    while index + 4 <= left.len() {
        let a = _mm256_loadu_pd(left.as_ptr().add(index));
        let b = _mm256_loadu_pd(right.as_ptr().add(index));
        _mm256_storeu_pd(output.as_mut_ptr().add(index), _mm256_add_pd(a, b));
        index += 4;
    }
    while index < left.len() { output[index] = left[index] + right[index]; index += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(input: &[f64], scalar: f64, output: &mut [f64]) {
    let factor = _mm256_set1_pd(scalar);
    let mut index = 0;
    while index + 4 <= input.len() {
        let values = _mm256_loadu_pd(input.as_ptr().add(index));
        _mm256_storeu_pd(output.as_mut_ptr().add(index), _mm256_mul_pd(values, factor));
        index += 4;
    }
    while index < input.len() { output[index] = input[index] * scalar; index += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(left: &[f64], right: &[f64]) -> f64 {
    let mut accumulator = _mm256_setzero_pd();
    let mut index = 0;
    while index + 4 <= left.len() {
        let a = _mm256_loadu_pd(left.as_ptr().add(index));
        let b = _mm256_loadu_pd(right.as_ptr().add(index));
        accumulator = _mm256_add_pd(accumulator, _mm256_mul_pd(a, b));
        index += 4;
    }
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator);
    let mut result = lanes.iter().sum::<f64>();
    while index < left.len() { result += left[index] * right[index]; index += 1; }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_fma_avx2(left: &[f64], right: &[f64]) -> f64 {
    let mut accumulator = _mm256_setzero_pd();
    let mut index = 0;
    while index + 4 <= left.len() {
        let a = _mm256_loadu_pd(left.as_ptr().add(index));
        let b = _mm256_loadu_pd(right.as_ptr().add(index));
        accumulator = _mm256_fmadd_pd(a, b, accumulator);
        index += 4;
    }
    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator);
    let mut result = lanes.iter().sum::<f64>();
    while index < left.len() { result += left[index] * right[index]; index += 1; }
    result
}


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_avx512(left: &[f64], right: &[f64], output: &mut [f64]) {
    let mut index = 0;
    while index + 8 <= left.len() {
        let a = _mm512_loadu_pd(left.as_ptr().add(index));
        let b = _mm512_loadu_pd(right.as_ptr().add(index));
        _mm512_storeu_pd(output.as_mut_ptr().add(index), _mm512_add_pd(a, b));
        index += 8;
    }
    while index < left.len() { output[index] = left[index] + right[index]; index += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scale_avx512(input: &[f64], scalar: f64, output: &mut [f64]) {
    let factor = _mm512_set1_pd(scalar);
    let mut index = 0;
    while index + 8 <= input.len() {
        let values = _mm512_loadu_pd(input.as_ptr().add(index));
        _mm512_storeu_pd(output.as_mut_ptr().add(index), _mm512_mul_pd(values, factor));
        index += 8;
    }
    while index < input.len() { output[index] = input[index] * scalar; index += 1; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(left: &[f64], right: &[f64]) -> f64 {
    let mut accumulator = _mm512_setzero_pd();
    let mut index = 0;
    while index + 8 <= left.len() {
        let a = _mm512_loadu_pd(left.as_ptr().add(index));
        let b = _mm512_loadu_pd(right.as_ptr().add(index));
        accumulator = _mm512_add_pd(accumulator, _mm512_mul_pd(a, b));
        index += 8;
    }
    let mut lanes = [0.0_f64; 8];
    _mm512_storeu_pd(lanes.as_mut_ptr(), accumulator);
    let mut result = lanes.iter().sum::<f64>();
    while index < left.len() { result += left[index] * right[index]; index += 1; }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_fma_avx512(left: &[f64], right: &[f64]) -> f64 {
    let mut accumulator = _mm512_setzero_pd();
    let mut index = 0;
    while index + 8 <= left.len() {
        let a = _mm512_loadu_pd(left.as_ptr().add(index));
        let b = _mm512_loadu_pd(right.as_ptr().add(index));
        accumulator = _mm512_fmadd_pd(a, b, accumulator);
        index += 8;
    }
    let mut lanes = [0.0_f64; 8];
    _mm512_storeu_pd(lanes.as_mut_ptr(), accumulator);
    let mut result = lanes.iter().sum::<f64>();
    while index < left.len() { result += left[index] * right[index]; index += 1; }
    result
}

/// Suma SIMD con fallback scalar.
pub fn add(left: &[f64], right: &[f64], output: &mut [f64], avx512: bool, avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if avx512 {
        unsafe { add_avx512(left, right, output); }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe { add_avx2(left, right, output); }
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if avx2 {
        unsafe { add_wasm(left, right, output); }
        return;
    }
    for index in 0..left.len() { output[index] = left[index] + right[index]; }
}

/// Escalado SIMD con fallback scalar.
pub fn scale(input: &[f64], scalar: f64, output: &mut [f64], avx512: bool, avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if avx512 {
        unsafe { scale_avx512(input, scalar, output); }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe { scale_avx2(input, scalar, output); }
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if avx2 {
        unsafe { scale_wasm(input, scalar, output); }
        return;
    }
    for index in 0..input.len() { output[index] = input[index] * scalar; }
}

/// Producto escalar SIMD con FMA cuando esta disponible.
pub fn dot(left: &[f64], right: &[f64], avx512: bool, avx2: bool, fma: bool) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512 && fma { return unsafe { dot_fma_avx512(left, right) }; }
        if avx512 { return unsafe { dot_avx512(left, right) }; }
        if avx2 && fma { return unsafe { dot_fma_avx2(left, right) }; }
        if avx2 { return unsafe { dot_avx2(left, right) }; }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if avx2 { return unsafe { dot_wasm(left, right) }; }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn add_wasm(left: &[f64], right: &[f64], output: &mut [f64]) {
    use std::arch::wasm32::*;
    let mut index = 0;
    while index + 2 <= left.len() {
        let a = v128_load(left.as_ptr().add(index) as *const v128);
        let b = v128_load(right.as_ptr().add(index) as *const v128);
        v128_store(output.as_mut_ptr().add(index) as *mut v128, f64x2_add(a, b));
        index += 2;
    }
    while index < left.len() { output[index] = left[index] + right[index]; index += 1; }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn scale_wasm(input: &[f64], scalar: f64, output: &mut [f64]) {
    use std::arch::wasm32::*;
    let factor = f64x2_splat(scalar);
    let mut index = 0;
    while index + 2 <= input.len() {
        let values = v128_load(input.as_ptr().add(index) as *const v128);
        v128_store(output.as_mut_ptr().add(index) as *mut v128, f64x2_mul(values, factor));
        index += 2;
    }
    while index < input.len() { output[index] = input[index] * scalar; index += 1; }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_wasm(left: &[f64], right: &[f64]) -> f64 {
    use std::arch::wasm32::*;
    let mut accumulator = f64x2_splat(0.0);
    let mut index = 0;
    while index + 2 <= left.len() {
        let a = v128_load(left.as_ptr().add(index) as *const v128);
        let b = v128_load(right.as_ptr().add(index) as *const v128);
        accumulator = f64x2_add(accumulator, f64x2_mul(a, b));
        index += 2;
    }
    let lanes = [f64x2_extract_lane::<0>(accumulator), f64x2_extract_lane::<1>(accumulator)];
    let mut result = lanes[0] + lanes[1];
    while index < left.len() { result += left[index] * right[index]; index += 1; }
    result
}
