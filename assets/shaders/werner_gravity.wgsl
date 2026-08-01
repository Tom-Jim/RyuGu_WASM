struct GravParams {
    probe_pos: vec3<f32>,
    num_faces: u32,
};

@group(0) @binding(0) var<uniform> params: GravParams;
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> partial_sums: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> face_densities: array<f32>; // <--- 新增：每个面的专属密度！

const EPSILON: f32 = 1e-6;

fn compute_face_gravity(probe: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, r3: vec3<f32>) -> vec3<f32> {
    let v1 = r1 - probe;
    let v2 = r2 - probe;
    let v3 = r3 - probe;
    
    let d1 = length(v1);
    let d2 = length(v2);
    let d3 = length(v3);

    let e12 = r2 - r1;
    let e23 = r3 - r2;
    let e31 = r1 - r3;
    let face_normal = normalize(cross(e12, -e31));
    
    let num = dot(v1, cross(v2, v3));
    let den = d1 * d2 * d3 + d1 * dot(v2, v3) + d2 * dot(v3, v1) + d3 * dot(v1, v2);
    var omega = 0.0;
    if (abs(den) > EPSILON) { omega = 2.0 * atan2(num, den); }
    let face_contribution = face_normal * (dot(face_normal, v1) * omega);

    var edge_contribution = vec3<f32>(0.0);
    let len12 = length(e12);
    let edge_normal12 = cross(face_normal, e12 / len12);
    let Le12 = log((d1 + d2 + len12 + EPSILON) / (d1 + d2 - len12 + EPSILON));
    edge_contribution += face_normal * (dot(edge_normal12, v1) * Le12); 

    let len23 = length(e23);
    let edge_normal23 = cross(face_normal, e23 / len23);
    let Le23 = log((d2 + d3 + len23 + EPSILON) / (d2 + d3 - len23 + EPSILON));
    edge_contribution += face_normal * (dot(edge_normal23, v2) * Le23);

    let len31 = length(e31);
    let edge_normal31 = cross(face_normal, e31 / len31);
    let Le31 = log((d3 + d1 + len31 + EPSILON) / (d3 + d1 - len31 + EPSILON));
    edge_contribution += face_normal * (dot(edge_normal31, v3) * Le31);

    return edge_contribution - face_contribution;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let face_idx = global_id.x;
    if (face_idx >= params.num_faces) { return; }

    let idx1 = indices[face_idx * 3u];
    let idx2 = indices[face_idx * 3u + 1u];
    let idx3 = indices[face_idx * 3u + 2u];

    let r1 = vertices[idx1].xyz;
    let r2 = vertices[idx2].xyz;
    let r3 = vertices[idx3].xyz;

    let acc_contribution = compute_face_gravity(params.probe_pos, r1, r2, r3);
    
    // 取出这个面专属的 1/R 密度，并乘进去！
    let g_density = face_densities[face_idx]; 
    partial_sums[face_idx] = vec4<f32>(acc_contribution * g_density, 0.0);
}