// Dense row-major GPU kernels. Cada invocation produce un elemento de salida.
struct Dimensions { rows: u32, columns: u32, shared: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dimensions;

@compute @workgroup_size(256)
fn gemv(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= dims.rows) { return; }
    var sum = 0.0;
    let base = row * dims.columns;
    for (var column = 0u; column < dims.columns; column++) { sum += a[base + column] * x[column]; }
    out[row] = sum;
}

@compute @workgroup_size(8, 8, 1)
fn gemm(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let column = id.x;
    if (row >= dims.rows || column >= dims.columns) { return; }
    var sum = 0.0;
    for (var shared = 0u; shared < dims.shared; shared++) { sum += a[row * dims.shared + shared] * b[shared * dims.columns + column]; }
    out[row * dims.columns + column] = sum;
}
