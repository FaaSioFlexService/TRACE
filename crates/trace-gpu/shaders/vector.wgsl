// Kernels vectoriales f32. La reduccion de dot/norm se realiza en dos pasadas:
// una primera fase genera parciales por workgroup y una segunda los reduce.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> scalar: f32;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&out)) { out[id.x] = a[id.x] + b[id.x]; }
}

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&out)) { out[id.x] = a[id.x] * scalar; }
}
