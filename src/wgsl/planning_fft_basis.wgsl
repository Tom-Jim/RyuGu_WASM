// Zero-padded Newton convolution, 56 unit-density columns. Everything large
// stays on the GPU. A bounded source upload / one column per submission limits
// queue depth; workgroups cover entire FFT planes, not one CPU FFT per column.
// Complex values use two f32 words per real/imaginary component. This keeps
// transform / density summation rounding out of the final potential Hessian.
struct Params {
    n: u32, side: u32, axis: u32, inverse: u32,
    column: u32, bank_offset: u32, output_offset: u32, source_count: u32,
    model: u32, buffer_kind: u32, _pad0: u32, _pad1: u32,
    spacing: vec2<f32>, half_extent: f32, gravity: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
// Packed five words: x/y/z/volume bits followed by voxel index (no struct padding).
@group(0) @binding(1) var<storage, read> sources: array<u32>;
@group(0) @binding(2) var<storage, read_write> bank: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> work: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> kernel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> densities: array<f32>;
@group(0) @binding(6) var<storage, read_write> potential: array<f32>;
// 128 roots computed once in f64, split into high/low words; < 2 KiB CPU work.
@group(0) @binding(7) var<storage, read> roots: array<vec4<f32>>;

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    return vec2(s, (a - (s - v)) + (b - v));
}
fn add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    return two_sum(s.x, s.y + a.y + b.y);
}
fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = a.x * b.x;
    return two_sum(p, fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x);
}
fn cmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4(add(mul(a.xy, b.xy), -mul(a.zw, b.zw)),
        add(mul(a.xy, b.zw), mul(a.zw, b.xy)));
}
fn cadd(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4(add(a.xy, b.xy), add(a.zw, b.zw));
}
fn cell_count() -> u32 { return params.n * params.n * params.n; }
fn index3(p: vec3<u32>, n: u32) -> u32 { return (p.z * n + p.y) * n + p.x; }
fn coord(i: u32, n: u32) -> vec3<u32> { return vec3(i % n, (i / n) % n, i / (n * n)); }
fn bank_index(column: u32, i: u32) -> u32 {
    return 2u * (params.bank_offset + column * cell_count() + i);
}
fn read_bank(i: u32) -> vec2<f32> {
    return vec2(bitcast<f32>(atomicLoad(&bank[i])), bitcast<f32>(atomicLoad(&bank[i + 1u])));
}
fn write_bank(i: u32, v: vec2<f32>) {
    atomicStore(&bank[i], bitcast<u32>(v.x));
    atomicStore(&bank[i + 1u], bitcast<u32>(v.y));
}
// A CAS add returns its rounding residual. Deposit that residual separately in
// the low word. Unlike a non-atomic high/low update this cannot lose a concurrent
// writer's contribution. Inputs are finite, positive source volumes.
fn atomic_add_residual(i: u32, value: f32) -> f32 {
    var old = atomicLoad(&bank[i]);
    var residual = 0.0;
    loop {
        let sum = two_sum(bitcast<f32>(old), value);
        let exchanged = atomicCompareExchangeWeak(&bank[i], old, bitcast<u32>(sum.x));
        if exchanged.exchanged {
            residual = sum.y;
            break;
        }
        old = exchanged.old_value;
    }
    // Keep an explicit value return after the loop: Naga otherwise inserts
    // a void return at the function boundary, invalidating the entire module.
    return residual;
}
@compute @workgroup_size(128)
fn deposit(@builtin(global_invocation_id) id: vec3<u32>) {
    let source = id.x;
    if source >= params.source_count { return; }
    let base = source * 5u;
    let position = vec3(bitcast<f32>(sources[base]), bitcast<f32>(sources[base+1u]), bitcast<f32>(sources[base+2u]));
    let volume = bitcast<f32>(sources[base+3u]);
    let voxel = sources[base+4u];
    if voxel >= 56u || volume <= 0.0 { return; }
    let grid = (position + vec3(params.half_extent)) / params.spacing.x - vec3(0.5);
    let low = vec3<i32>(floor(grid));
    let fraction = grid - floor(grid);
    for (var corner = 0u; corner < 8u; corner += 1u) {
        let delta = vec3<u32>(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
        let node = low + vec3<i32>(delta);
        if any(node < vec3(0)) || any(node >= vec3<i32>(i32(params.n))) { continue; }
        let weights = select(vec3(1.0) - fraction, fraction, delta != vec3(0u));
        let mass = mul(mul(mul(vec2(volume, 0.0), vec2(weights.x, 0.0)), vec2(weights.y, 0.0)), vec2(weights.z, 0.0));
        let address = bank_index(voxel, index3(vec3<u32>(node), params.n));
        let residual = atomic_add_residual(address, mass.x);
        _ = atomic_add_residual(address + 1u, residual + mass.y);
    }
}
@compute @workgroup_size(128)
fn seed_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.side * params.side * params.side { return; }
    let c = coord(i, params.side);
    let d = vec3<f32>(select(vec3<i32>(c) - vec3<i32>(i32(params.side)), vec3<i32>(c), c < vec3(params.n)));
    // Distances in integer cell units avoid cancellation/rounding in r^2.
    let r2 = max(dot(d, d), 0.25);
    let r = sqrt(r2);
    let distance = mul(vec2(r, fma(-r, r, r2) / (2.0 * r)), params.spacing);
    let reciprocal = 1.0 / distance.x;
    let correction = add(vec2(1.0, 0.0), -mul(distance, vec2(reciprocal, 0.0)));
    kernel[i] = vec4(add(vec2(reciprocal, 0.0), correction * reciprocal), vec2(0.0));
}
@compute @workgroup_size(128)
fn load_column(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.side * params.side * params.side { return; }
    let c = coord(i, params.side);
    var value = vec2(0.0);
    if all(c < vec3(params.n)) { value = read_bank(bank_index(params.column, index3(c, params.n))); }
    work[i] = vec4(value, vec2(0.0));
}
var<workgroup> line: array<vec4<f32>, 128>;
fn line_index(group: u32, lane: u32) -> u32 {
    let a = group % params.side;
    let b = group / params.side;
    if params.axis == 0u { return index3(vec3(lane, a, b), params.side); }
    if params.axis == 1u { return index3(vec3(a, lane, b), params.side); }
    return index3(vec3(a, b, lane), params.side);
}
@compute @workgroup_size(128)
fn transform(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // Inactive lanes still take every barrier for the 32-point coarse level.
    let bits = select(5u, 7u, params.side == 128u);
    if lane < params.side {
        let source = line_index(group.x, reverseBits(lane) >> (32u - bits));
        var value = work[source];
        if params.buffer_kind == 1u { value = kernel[source]; }
        line[lane] = value;
    }
    workgroupBarrier();
    for (var width = 2u; width <= params.side; width *= 2u) {
        var result = vec4(0.0);
        if lane < params.side {
            let half_width = width / 2u;
            let offset = lane % half_width;
            let first = (lane / width) * width + offset;
            var root = roots[offset * (128u / width)];
            if params.inverse != 0u { root = vec4(root.xy, -root.zw); }
            let odd = cmul(line[first + half_width], root);
            result = cadd(line[first], select(odd, -odd, (lane % width) >= half_width));
        }
        workgroupBarrier();
        if lane < params.side { line[lane] = result; }
        workgroupBarrier();
    }
    if lane < params.side {
        var result = line[lane];
        if params.inverse != 0u { result /= f32(params.side); }
        let destination = line_index(group.x, lane);
        if params.buffer_kind == 1u { kernel[destination] = result; }
        else { work[destination] = result; }
    }
}
@compute @workgroup_size(128)
fn convolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.side * params.side * params.side { work[id.x] = cmul(work[id.x], kernel[id.x]); }
}
@compute @workgroup_size(128)
fn store_column(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cell_count() { return; }
    let source = index3(coord(id.x, params.n), params.side);
    write_bank(bank_index(params.column, id.x), mul(work[source].xy, vec2(params.gravity, 0.0)));
}
@compute @workgroup_size(128)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cell_count() { return; }
    var sum = vec2(0.0);
    for (var column = 0u; column < 56u; column += 1u) {
        sum = add(sum, mul(read_bank(bank_index(column, id.x)), vec2(densities[params.model * 56u + column], 0.0)));
    }
    potential[params.output_offset + id.x] = sum.x + sum.y;
}
