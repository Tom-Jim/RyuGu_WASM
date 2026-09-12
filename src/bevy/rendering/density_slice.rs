use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;

const DENSITY_SLICE_SHADER: &str = "embedded://ryugu_wasm/wgsl/density_slice.wgsl";
pub(crate) const DENSITY_VOLUME_RESOLUTION: u32 = 48;
const SLICE_HALF_EXTENT: f32 = 550.0;

#[derive(Component)]
pub struct DensitySliceMarker;

#[derive(Resource, Default)]
pub struct DensityVolumeState {
    pub image: Option<Handle<Image>>,
    pub bake_key: u64,
    pub density_min: f32,
    pub density_max: f32,
    pub mean: f32,
    pub half_span: f32,
    pub mode: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct DensitySliceUniform {
    pub volume_min_extent: Vec4,
    pub color_low: Vec4,
    pub color_mid: Vec4,
    pub color_high: Vec4,
    pub density_range: Vec4,
    pub body_from_world_x: Vec4,
    pub body_from_world_y: Vec4,
    pub body_from_world_z: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct DensitySliceMaterial {
    #[uniform(0)]
    pub params: DensitySliceUniform,
    // Rgba8Unorm + textureLoad (no sampler) is the WebGPU-safe volume path.
    #[texture(1, dimension = "3d", sample_type = "float", filterable = false)]
    pub volume: Handle<Image>,
    pub alpha_mode: AlphaMode,
}

impl Material for DensitySliceMaterial {
    fn fragment_shader() -> ShaderRef {
        DENSITY_SLICE_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}

pub fn ensure_density_slice_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DensitySliceMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut volume: ResMut<DensityVolumeState>,
    existing: Query<Entity, With<DensitySliceMarker>>,
) {
    if !existing.is_empty() {
        return;
    }
    let image = images.add(blank_density_volume());
    volume.image = Some(image.clone());
    let material = materials.add(DensitySliceMaterial {
        params: DensitySliceUniform {
            volume_min_extent: Vec4::new(0.0, 0.0, 0.0, SECTION_CLIP_RADIUS),
            color_low: Vec4::new(0.45, 0.04, 0.02, 0.0),
            color_mid: Vec4::new(1.0, 0.18, 0.015, 1.0),
            color_high: Vec4::new(1.0, 0.95, 0.12, SECTION_CLIP_RADIUS),
            density_range: Vec4::new(0.0, 1.0, 0.0, 0.82),
            body_from_world_x: Vec4::X,
            body_from_world_y: Vec4::Y,
            body_from_world_z: Vec4::Z,
        },
        volume: image,
        alpha_mode: AlphaMode::Blend,
    });
    let mesh = meshes.add(Mesh::from(Plane3d::new(
        Vec3::Z,
        Vec2::splat(SLICE_HALF_EXTENT),
    )));
    commands.spawn((
        DensitySliceMarker,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Visibility::Hidden,
    ));
}

pub fn bake_density_volume_system(
    mut images: ResMut<Assets<Image>>,
    mut volume: ResMut<DensityVolumeState>,
    mut materials: ResMut<Assets<DensitySliceMaterial>>,
    slice_query: Query<&MeshMaterial3d<DensitySliceMaterial>, With<DensitySliceMarker>>,
    show_section: Res<ShowSection>,
    active_method: Res<ActiveGravityMethod>,
    density_mode: Res<DensityMode>,
    density_c: Option<Res<DensityC>>,
    werner_density: Option<Res<WernerDensity>>,
    inversion: Res<TrajectoryInversionState>,
) {
    let inferred = if show_section.0 {
        None
    } else {
        inversion
            .displayed_density
            .as_ref()
            .filter(|result| result.method == *active_method)
    };
    if !show_section.0 && inferred.is_none() {
        return;
    }

    let display_method = inferred.map_or(*active_method, |result| result.method);
    let c = inferred
        .filter(|result| result.method != ActiveGravityMethod::HomogeneousWerner)
        .map(|result| result.density)
        .unwrap_or_else(|| density_c.map(|r| r.0).unwrap_or(1.0));
    let uniform_forward_density = inferred.is_none() && *density_mode == DensityMode::Constant;
    let uniform_density = inferred
        .filter(|result| result.method == ActiveGravityMethod::HomogeneousWerner)
        .map(|result| result.density)
        .unwrap_or_else(|| werner_density.map(|r| r.0).unwrap_or(0.0));
    let homogeneous_display =
        display_method == ActiveGravityMethod::HomogeneousWerner || uniform_forward_density;

    let bake_key = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        show_section.0.hash(&mut hasher);
        method_key(display_method).hash(&mut hasher);
        density_mode_key(*density_mode).hash(&mut hasher);
        c.to_bits().hash(&mut hasher);
        uniform_density.to_bits().hash(&mut hasher);
        if let Some(result) = inferred {
            result.source_hash.hash(&mut hasher);
            result.voxels.len().hash(&mut hasher);
            for voxel in &result.voxels {
                voxel.density.to_bits().hash(&mut hasher);
                voxel.center.x.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    };
    if volume.bake_key == bake_key && volume.image.is_some() {
        return;
    }

    let half_extent = SECTION_CLIP_RADIUS;
    let resolution = DENSITY_VOLUME_RESOLUTION as usize;
    let mut raw = vec![0_u8; resolution * resolution * resolution * 4];
    let (density_min, density_max, mean, half_span, mode) = if let Some(result) = inferred {
        let (minimum, maximum) = result.voxels.iter().map(|voxel| voxel.density).fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), density| (minimum.min(density), maximum.max(density)),
        );
        let half_span = (maximum - result.density)
            .abs()
            .max((result.density - minimum).abs())
            .max(result.density.abs() * 0.05)
            .max(f32::EPSILON);
        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let body = voxel_body_position(x, y, z, resolution, half_extent);
                    let t = if body.length() > half_extent {
                        0.0
                    } else {
                        let density = interpolated_inverted_density(result, body);
                        (0.5 + (density - result.density) / (2.0 * half_span)).clamp(0.0, 1.0)
                    };
                    write_unorm8(&mut raw, x, y, z, resolution, t);
                }
            }
        }
        (minimum, maximum, result.density, half_span, 2.0)
    } else if homogeneous_display {
        let density = uniform_density.max(c.max(0.0));
        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let body = voxel_body_position(x, y, z, resolution, half_extent);
                    let t = if body.length() <= half_extent {
                        0.5
                    } else {
                        0.0
                    };
                    write_unorm8(&mut raw, x, y, z, resolution, t);
                }
            }
        }
        (density, density, density, density.max(1.0), 1.0)
    } else {
        let density_min = logarithmic_radial_density(0.0, c);
        let density_max = logarithmic_radial_density(half_extent, c);
        let span = (density_max - density_min).max(1.0e-6);
        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let body = voxel_body_position(x, y, z, resolution, half_extent);
                    let t = if body.length() > half_extent {
                        0.0
                    } else {
                        let density = logarithmic_radial_density(body.length(), c);
                        ((density - density_min) / span).clamp(0.0, 1.0)
                    };
                    write_unorm8(&mut raw, x, y, z, resolution, t);
                }
            }
        }
        (density_min, density_max, 0.0, 1.0, 0.0)
    };

    // Replace the Image asset so Bevy always re-uploads the 3-D texture to the GPU.
    let image = images.add(density_volume_from_raw(raw));
    volume.image = Some(image.clone());
    volume.bake_key = bake_key;
    volume.density_min = density_min;
    volume.density_max = density_max;
    volume.mean = mean;
    volume.half_span = half_span;
    volume.mode = mode;

    if let Ok(material_handle) = slice_query.single()
        && let Some(mut material) = materials.get_mut(&material_handle.0)
    {
        material.volume = image;
    }
}

pub fn update_density_slice_system(
    ryugu_query: Query<&Transform, (With<RyuguMarker>, Without<DensitySliceMarker>)>,
    camera_query: Query<
        &Transform,
        (
            With<Camera3d>,
            Without<RyuguMarker>,
            Without<DensitySliceMarker>,
        ),
    >,
    show_section: Res<ShowSection>,
    active_method: Res<ActiveGravityMethod>,
    inversion: Res<TrajectoryInversionState>,
    volume: Res<DensityVolumeState>,
    mut materials: ResMut<Assets<DensitySliceMaterial>>,
    mut slice_query: Query<
        (
            &MeshMaterial3d<DensitySliceMaterial>,
            &mut Transform,
            &mut Visibility,
        ),
        (With<DensitySliceMarker>, Without<RyuguMarker>),
    >,
) {
    let inferred = if show_section.0 {
        None
    } else {
        inversion
            .displayed_density
            .as_ref()
            .filter(|result| result.method == *active_method)
    };
    let visible = show_section.0 || inferred.is_some();
    let Ok((material_handle, mut transform, mut visibility)) = slice_query.single_mut() else {
        return;
    };
    *visibility = if visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    if !visible {
        return;
    }
    let Some(ryugu_tf) = ryugu_query.iter().next() else {
        return;
    };
    let Some(cam_tf) = camera_query.iter().next() else {
        return;
    };
    let Some(mut material) = materials.get_mut(&material_handle.0) else {
        return;
    };

    let com = ryugu_tf.translation;
    let plane_normal = (cam_tf.translation - com).normalize_or_zero();
    if plane_normal == Vec3::ZERO {
        return;
    }
    let up = if plane_normal.abs().dot(Vec3::Y) < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent_u = plane_normal.cross(up).normalize();
    let tangent_v = plane_normal.cross(tangent_u).normalize();
    // Plane3d defaults to +Z normal; map local Z → camera-facing plane normal.
    *transform = Transform::from_matrix(Mat4::from_cols(
        tangent_u.extend(0.0),
        tangent_v.extend(0.0),
        plane_normal.extend(0.0),
        com.extend(1.0),
    ));

    let inv = Mat3::from_quat(ryugu_tf.rotation.inverse());
    let display_method = inferred.map_or(*active_method, |result| result.method);
    let (low, mid, high) = density_palette(display_method);
    material.params.volume_min_extent = Vec4::new(0.0, 0.0, 0.0, SECTION_CLIP_RADIUS);
    material.params.color_low = Vec4::new(low.x, low.y, low.z, volume.mean);
    material.params.color_mid = Vec4::new(mid.x, mid.y, mid.z, volume.half_span);
    material.params.color_high = Vec4::new(high.x, high.y, high.z, SECTION_CLIP_RADIUS);
    material.params.density_range = Vec4::new(
        volume.density_min,
        volume.density_max,
        volume.mode,
        if inferred.is_some() { 0.72 } else { 0.84 },
    );
    material.params.body_from_world_x = Vec4::new(inv.x_axis.x, inv.y_axis.x, inv.z_axis.x, com.x);
    material.params.body_from_world_y = Vec4::new(inv.x_axis.y, inv.y_axis.y, inv.z_axis.y, com.y);
    material.params.body_from_world_z = Vec4::new(inv.x_axis.z, inv.y_axis.z, inv.z_axis.z, com.z);
    if let Some(image) = &volume.image {
        material.volume = image.clone();
    }
}

fn blank_density_volume() -> Image {
    density_volume_from_raw(vec![
        0_u8;
        (DENSITY_VOLUME_RESOLUTION * DENSITY_VOLUME_RESOLUTION * DENSITY_VOLUME_RESOLUTION * 4)
            as usize
    ])
}

fn density_volume_from_raw(raw: Vec<u8>) -> Image {
    let resolution = DENSITY_VOLUME_RESOLUTION;
    let mut image = Image::new(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: resolution,
        },
        TextureDimension::D3,
        raw,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = bevy::image::ImageSampler::nearest();
    image
}

fn voxel_body_position(x: usize, y: usize, z: usize, resolution: usize, half_extent: f32) -> Vec3 {
    let denom = (resolution.saturating_sub(1).max(1)) as f32;
    Vec3::new(
        (x as f32 / denom) * 2.0 - 1.0,
        (y as f32 / denom) * 2.0 - 1.0,
        (z as f32 / denom) * 2.0 - 1.0,
    ) * half_extent
}

fn write_unorm8(raw: &mut [u8], x: usize, y: usize, z: usize, resolution: usize, value: f32) {
    let index = ((z * resolution + y) * resolution + x) * 4;
    let byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    raw[index] = byte;
    raw[index + 1] = byte;
    raw[index + 2] = byte;
    raw[index + 3] = 255;
}

fn density_palette(method: ActiveGravityMethod) -> (Vec3, Vec3, Vec3) {
    match method {
        ActiveGravityMethod::RadialAnalytic => (
            Vec3::new(0.45, 0.04, 0.02),
            Vec3::new(1.0, 0.18, 0.015),
            Vec3::new(1.0, 0.95, 0.12),
        ),
        ActiveGravityMethod::FrequencyDomain => (
            Vec3::new(0.02, 0.35, 0.45),
            Vec3::new(0.02, 0.9, 0.42),
            Vec3::new(0.82, 1.0, 0.12),
        ),
        ActiveGravityMethod::MmfftCompressed => (
            Vec3::new(0.12, 0.22, 0.65),
            Vec3::new(0.02, 0.58, 1.0),
            Vec3::new(0.72, 1.0, 1.0),
        ),
        ActiveGravityMethod::Fmm => (
            Vec3::new(0.4, 0.05, 0.5),
            Vec3::new(0.95, 0.06, 0.72),
            Vec3::new(1.0, 0.78, 0.92),
        ),
        ActiveGravityMethod::HomogeneousWerner => (
            Vec3::new(0.08, 0.35, 0.65),
            Vec3::new(0.0, 0.75, 1.0),
            Vec3::new(0.88, 1.0, 1.0),
        ),
    }
}

fn method_key(method: ActiveGravityMethod) -> u8 {
    match method {
        ActiveGravityMethod::RadialAnalytic => 0,
        ActiveGravityMethod::HomogeneousWerner => 1,
        ActiveGravityMethod::FrequencyDomain => 2,
        ActiveGravityMethod::MmfftCompressed => 3,
        ActiveGravityMethod::Fmm => 4,
    }
}

fn density_mode_key(mode: DensityMode) -> u8 {
    match mode {
        DensityMode::Variable => 0,
        DensityMode::Constant => 1,
    }
}

/// Shared with the legacy contour helpers in `render_systems`.
pub(crate) fn interpolated_inverted_density(
    result: &DensityInversionResult,
    body_point: Vec3,
) -> f32 {
    let support = (result.voxel_size * 1.75).max(f32::MIN_POSITIVE);
    let support_squared = support * support;
    let mut weighted_density = 0.0_f32;
    let mut total_weight = 0.0_f32;
    let mut nearest = (f32::INFINITY, result.density);

    for voxel in &result.voxels {
        let distance_squared = body_point.distance_squared(voxel.center);
        if distance_squared < nearest.0 {
            nearest = (distance_squared, voxel.density);
        }
        if distance_squared < support_squared {
            let q_squared = distance_squared / support_squared;
            let weight = (1.0 - q_squared).powi(2);
            weighted_density += weight * voxel.density;
            total_weight += weight;
        }
    }

    if total_weight > f32::EPSILON {
        weighted_density / total_weight
    } else {
        nearest.1
    }
}
