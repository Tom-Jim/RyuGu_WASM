#import bevy_pbr::forward_io::VertexOutput

struct DensitySliceParams {
    // xyz unused; w = body half-extent (metres) of the baked volume AABB.
    volume_min_extent: vec4<f32>,
    // xyz = colormap low; w unused (normalized density is baked into the volume).
    color_low: vec4<f32>,
    // xyz = colormap mid; w unused.
    color_mid: vec4<f32>,
    // xyz = colormap high; w = clip radius in body metres (outside → discard).
    color_high: vec4<f32>,
    // x/y unused, z unused, w = opacity. Density is pre-normalized into the texture.
    density_range: vec4<f32>,
    // Columns of body←world rotation; translation in w of each.
    body_from_world_x: vec4<f32>,
    body_from_world_y: vec4<f32>,
    body_from_world_z: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> material: DensitySliceParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(1)
var density_volume: texture_3d<f32>;

fn colormap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    if x < 0.5 {
        return mix(material.color_low.xyz, material.color_mid.xyz, x * 2.0);
    }
    return mix(material.color_mid.xyz, material.color_high.xyz, (x - 0.5) * 2.0);
}

fn world_to_body(world: vec3<f32>) -> vec3<f32> {
    let origin = vec3<f32>(
        material.body_from_world_x.w,
        material.body_from_world_y.w,
        material.body_from_world_z.w,
    );
    let delta = world - origin;
    return vec3<f32>(
        dot(material.body_from_world_x.xyz, delta),
        dot(material.body_from_world_y.xyz, delta),
        dot(material.body_from_world_z.xyz, delta),
    );
}

fn sample_normalized_density(body: vec3<f32>) -> f32 {
    let half_extent = max(material.volume_min_extent.w, 1.0e-3);
    let uvw = (body / half_extent) * 0.5 + vec3<f32>(0.5);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0;
    }
    // textureLoad avoids R32Float / non-filterable sample validation failures.
    let dims = vec3<f32>(textureDimensions(density_volume));
    let coord = vec3<i32>(clamp(floor(uvw * dims), vec3<f32>(0.0), dims - vec3<f32>(1.0)));
    return textureLoad(density_volume, coord, 0).r;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let body = world_to_body(in.world_position.xyz);
    let clip_radius = max(material.color_high.w, 1.0);
    if length(body) > clip_radius {
        discard;
    }

    let t = sample_normalized_density(body);
    let rgb = colormap(t);
    let light = normalize(vec3<f32>(0.35, 0.70, 0.55));
    let normal = normalize(in.world_normal);
    let shade = 0.55 + 0.45 * max(dot(normal, light), 0.0);
    return vec4<f32>(rgb * shade, material.density_range.w);
}
