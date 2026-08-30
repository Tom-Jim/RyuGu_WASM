#import bevy_pbr::forward_io::VertexOutput

struct MobileUnlitMaterial {
    color: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> material: MobileUnlitMaterial;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // A tiny view-independent diffuse term keeps Ryugu's relief readable on
    // mobile without pulling in Bevy's full PBR shader and light bindings.
    let normal = normalize(in.world_normal);
    let light_direction = normalize(vec3<f32>(0.35, 0.70, 0.55));
    let brightness = 0.24 + 0.76 * max(dot(normal, light_direction), 0.0);
    return vec4<f32>(material.color.rgb * brightness, material.color.a);
}
