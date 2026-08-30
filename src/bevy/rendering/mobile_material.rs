use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

const MOBILE_UNLIT_SHADER: &str = "shaders/mobile_unlit.wgsl";

/// A deliberately small material used only by mobile browsers. Unlike
/// `StandardMaterial { unlit: true }`, this does not compile Bevy's PBR
/// fragment shader, which is the pipeline rejected by the affected Vulkan
/// driver.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct MobileUnlitMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    pub alpha_mode: AlphaMode,
}

impl Material for MobileUnlitMaterial {
    fn fragment_shader() -> ShaderRef {
        MOBILE_UNLIT_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}

/// Swap each newly loaded glTF mesh away from `StandardMaterial` before the
/// render extraction schedule sees it. Desktop browsers never register this
/// system or this material plugin.
pub fn configure_mobile_materials_system(
    mut commands: Commands,
    mesh_materials: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>),
        Without<MeshMaterial3d<MobileUnlitMaterial>>,
    >,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut mobile_materials: ResMut<Assets<MobileUnlitMaterial>>,
) {
    for (entity, handle) in &mesh_materials {
        let Some(source) = standard_materials.get(&handle.0) else {
            continue;
        };
        let mut color = source.base_color.to_linear();
        // Some glTF materials store almost all visible color in textures. A
        // useful neutral fallback keeps the geometry visible without adding a
        // texture-sampling shader to the fragile mobile pipeline.
        if color.red + color.green + color.blue < 0.06 {
            color = LinearRgba::new(0.32, 0.38, 0.42, color.alpha);
        }
        let material = mobile_materials.add(MobileUnlitMaterial {
            color,
            alpha_mode: source.alpha_mode,
        });
        commands
            .entity(entity)
            .remove::<MeshMaterial3d<StandardMaterial>>()
            .insert(MeshMaterial3d(material));
    }
}

/// Mobile equivalent of `section_alpha_system` for the lightweight material.
pub fn mobile_section_alpha_system(
    show_section: Res<ShowSection>,
    inversion: Res<TrajectoryInversionState>,
    ryugu_query: Query<Entity, With<RyuguMarker>>,
    children_query: Query<&Children>,
    material_handles: Query<&MeshMaterial3d<MobileUnlitMaterial>>,
    mut materials: ResMut<Assets<MobileUnlitMaterial>>,
) {
    if !show_section.is_changed() && !inversion.is_changed() {
        return;
    }
    let section_visible = show_section.0 || inversion.displayed_density.is_some();
    let Some(root) = ryugu_query.iter().next() else {
        return;
    };

    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if let Ok(handle) = material_handles.get(entity)
            && let Some(mut material) = materials.get_mut(&handle.0)
        {
            material.color.alpha = if section_visible {
                if show_section.0 { 0.25 } else { 0.20 }
            } else {
                1.0
            };
            material.alpha_mode = if section_visible {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            };
        }
        if let Ok(children) = children_query.get(entity) {
            stack.extend(children.iter());
        }
    }
}
