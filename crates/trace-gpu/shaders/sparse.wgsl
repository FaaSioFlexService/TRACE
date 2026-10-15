// Kernels SpMV. CSR usa rangos por fila; ELLPACK usa ancho fijo por fila.
struct SparseDimensions { rows: u32, columns: u32, width: u32 };
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> dims: SparseDimensions;

@compute @workgroup_size(256)
fn csr_spmv(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= dims.rows) { return; }
    var sum = 0.0;
    for (var index = row_ptr[row]; index < row_ptr[row + 1u]; index++) { sum += values[index] * x[indices[index]]; }
    out[row] = sum;
}

@compute @workgroup_size(256)
fn ellpack_spmv(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= dims.rows) { return; }
    let base = row * dims.width;
    var sum = 0.0;
    for (var slot = 0u; slot < dims.width; slot++) {
        let column = indices[base + slot];
        if (column != 0xffffffffu) { sum += values[base + slot] * x[column]; }
    }
    out[row] = sum;
}
