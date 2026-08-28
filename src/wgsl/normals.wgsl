struct Position { xyz: vec3<f32>, _pad: f32, }
struct Normal   { xyz: vec3<f32>, _pad: f32, }

@group(0) @binding(0) var<storage, read>       positions : array<Position>;
@group(0) @binding(1) var<storage, read>       offsets   : array<u32>;
@group(0) @binding(2) var<storage, read>       nbr_idx   : array<u32>;
@group(0) @binding(3) var<storage, read_write> normals   : array<Normal>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&positions) { return; }

    let start = offsets[i];
    let end   = offsets[i + 1u];
    let count = end - start;

    if count < 2u {
        normals[i] = Normal(vec3<f32>(0.0, 1.0, 0.0), 0.0);
        return;
    }

    let p   = positions[i].xyz;
    var acc = vec3<f32>(0.0);
    let last = end - 1u;

    for (var k = start; k < last; k += 1u) {
        let a = positions[nbr_idx[k]].xyz;
        let b = positions[nbr_idx[k + 1u]].xyz;
        acc += cross(a - p, b - p);
    }
    let a_last  = positions[nbr_idx[last]].xyz;
    let b_first = positions[nbr_idx[start]].xyz;
    acc += cross(a_last - p, b_first - p);

    let len = length(acc);
    let n = select(vec3<f32>(0.0, 1.0, 0.0), acc / len, len > 1e-6);
    normals[i] = Normal(n, 0.0);
}
