use bevy::asset::LoadContext;
use bevy::gltf::{
    GltfAssetLabel, GltfMaterial,
    extensions::{ErasedGltfExtensionHandler, GltfExtensionHandler, GltfExtensionHandlers},
};
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

const MOBILE_UNLIT_SHADER: &str = "shaders/mobile_unlit.wgsl";
const MOBILE_MATERIAL_SUFFIX: &str = "mobile_unlit";

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

fn mobile_material_from_gltf(source: &GltfMaterial) -> MobileUnlitMaterial {
    let mut color = source.base_color.to_linear();
    // Some glTF materials store almost all visible color in textures. A
    // useful neutral fallback keeps the geometry visible without adding a
    // texture-sampling shader to the fragile mobile pipeline.
    if color.red + color.green + color.blue < 0.06 {
        color = LinearRgba::new(0.32, 0.38, 0.42, color.alpha);
    }
    MobileUnlitMaterial {
        color,
        alpha_mode: source.alpha_mode,
    }
}

fn mobile_material_label(material_label: &str) -> String {
    format!("{material_label}/{MOBILE_MATERIAL_SUFFIX}")
}

/// Creates and attaches the mobile material while the glTF scene is built.
/// This prevents a `StandardMaterial` entity from existing even for a single
/// render-extraction frame.
#[derive(Default, Clone)]
struct MobileGltfMaterialHandler;

impl GltfExtensionHandler for MobileGltfMaterialHandler {
    fn dyn_clone(&self) -> Box<dyn ErasedGltfExtensionHandler> {
        Box::new(self.clone())
    }

    fn on_root(
        &mut self,
        load_context: &mut LoadContext<'_>,
        _gltf: &bevy::gltf::gltf::Gltf,
        _settings: &bevy::gltf::GltfLoaderSettings,
    ) {
        let default_label = mobile_material_label(&GltfAssetLabel::DefaultMaterial.to_string());
        load_context.add_labeled_asset(
            default_label,
            mobile_material_from_gltf(&GltfMaterial::default()),
        );
    }

    fn on_material(
        &mut self,
        load_context: &mut LoadContext<'_>,
        _gltf_material: &bevy::gltf::gltf::Material,
        _material: Handle<GltfMaterial>,
        material_asset: &GltfMaterial,
        material_label: &str,
    ) {
        load_context.add_labeled_asset(
            mobile_material_label(material_label),
            mobile_material_from_gltf(material_asset),
        );
    }

    fn on_spawn_mesh_and_material(
        &mut self,
        load_context: &mut LoadContext<'_>,
        _primitive: &bevy::gltf::gltf::Primitive,
        _mesh: &bevy::gltf::gltf::Mesh,
        _material: &bevy::gltf::gltf::Material,
        entity: &mut EntityWorldMut,
        material_label: &str,
    ) {
        let handle = load_context.get_label_handle::<MobileUnlitMaterial>(
            mobile_material_label(material_label),
        );
        entity.insert(MeshMaterial3d(handle));
    }
}

/// Registers the handler after `GltfPlugin` has created its shared handler
/// list but before the loader starts processing any model assets.
pub fn register_mobile_gltf_materials(app: &mut App) {
    let handlers = app
        .world()
        .resource::<GltfExtensionHandlers>()
        .0
        .clone();
    bevy::tasks::block_on(async {
        handlers
            .write()
            .await
            .push(Box::new(MobileGltfMaterialHandler));
    });
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
    unmaterialized_meshes: Query<
        Entity,
        (With<Mesh3d>, Without<MeshMaterial3d<StandardMaterial>>, Without<MeshMaterial3d<MobileUnlitMaterial>>),
    >,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut mobile_materials: ResMut<Assets<MobileUnlitMaterial>>,
) {
    for (entity, handle) in &mesh_materials {
        let Some(source) = standard_materials.get(&handle.0) else {
            continue;
        };
        let material = mobile_materials.add(MobileUnlitMaterial {
            color: source.base_color.to_linear(),
            alpha_mode: source.alpha_mode,
        });
        commands
            .entity(entity)
            .remove::<MeshMaterial3d<StandardMaterial>>()
            .insert(MeshMaterial3d(material));
    }

    // On mobile the glTF loader is configured not to create StandardMaterial
    // at all. Add the neutral fallback directly to those mesh entities.
    let fallback = mobile_materials.add(MobileUnlitMaterial {
        color: LinearRgba::new(0.32, 0.38, 0.42, 1.0),
        alpha_mode: AlphaMode::Opaque,
    });
    for entity in &unmaterialized_meshes {
        commands.entity(entity).insert(MeshMaterial3d(fallback.clone()));
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
