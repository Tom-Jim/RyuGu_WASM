// Werner & Scheeres closed-polyhedron forward gravity. Shared edges are
// evaluated exactly once with both adjacent face dyads; faces use signed solid
// angles. This is a homogeneous-density reference method.

struct WernerParams {
    probe_pos: vec3<f32>,
    g_density: f32,
    edge_count: u32,
    face_count: u32,
    item_count: u32,
    _padding: u32,
};

struct WernerEdge {
    p0: vec4<f32>,
    p1: vec4<f32>,
    tensor_row0: vec4<f32>,
    tensor_row1: vec4<f32>,
    tensor_row2: vec4<f32>,
};

struct WernerFace {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: WernerParams;
@group(0) @binding(1) var<storage, read> edges: array<WernerEdge>;
@group(0) @binding(2) var<storage, read> faces: array<WernerFace>;
@group(0) @binding(3) var<storage, read_write> output_acc: array<vec4<f32>>;

var<workgroup> shared_acc: array<vec4<f32>, 64>;

fn edge_contribution(index: u32) -> vec3<f32> {
    let edge = edges[index];
    let r0 = edge.p0.xyz - params.probe_pos;
    let r1 = edge.p1.xyz - params.probe_pos;
    let length0 = length(r0);
    let length1 = length(r1);
    let edge_length = length(edge.p1.xyz - edge.p0.xyz);
    let sum_length = length0 + length1;
    let denominator = max(sum_length - edge_length, 1.0e-6 * max(sum_length, 1.0));
    let logarithm = log(max((sum_length + edge_length) / denominator, 1.0));
    let tensor_r = vec3<f32>(
        dot(edge.tensor_row0.xyz, r0),
        dot(edge.tensor_row1.xyz, r0),
        dot(edge.tensor_row2.xyz, r0)
    );
    return tensor_r * logarithm;
}

fn face_contribution(index: u32) -> vec3<f32> {
    let face = faces[index];
    let r0 = face.p0.xyz - params.probe_pos;
    let r1 = face.p1.xyz - params.probe_pos;
    let r2 = face.p2.xyz - params.probe_pos;
    let length0 = length(r0);
    let length1 = length(r1);
    let length2 = length(r2);
    let numerator = dot(r0, cross(r1, r2));
    let denominator = length0 * length1 * length2
        + length0 * dot(r1, r2)
        + length1 * dot(r2, r0)
        + length2 * dot(r0, r1);
    let solid_angle = 2.0 * atan2(numerator, denominator);
    let normal = face.normal.xyz;
    return normal * dot(normal, r0) * solid_angle;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let index = global_id.x;
    let lane = local_id.x;
    var edge_sum = vec3<f32>(0.0);
    var face_sum = vec3<f32>(0.0);
    if index < params.edge_count {
        edge_sum = edge_contribution(index);
    }
    if index < params.face_count {
        face_sum = face_contribution(index);
    }
    // ∇U for the conventional positive gravitational potential:
    // -Gρ Σ(E r L) + Gρ Σ(F r ω).
    shared_acc[lane] = vec4<f32>(params.g_density * (-edge_sum + face_sum), 0.0);
    workgroupBarrier();

    var stride = 32u;
    loop {
        if stride == 0u {
            break;
        }
        if lane < stride {
            shared_acc[lane] += shared_acc[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if lane == 0u {
        output_acc[workgroup_id.x] = shared_acc[0];
    }
}
