use crate::components::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

pub fn setup_fps_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("FPS: --"),
        TextFont {
            font_size: bevy::text::FontSize::Px(16.0),
            ..default()
        },
        TextColor(Color::srgb(0.6, 1.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(15.0),
            right: Val::Px(15.0),
            ..default()
        },
        FpsTextMarker,
    ));
}

pub fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsTextMarker>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    if let Some(mut text) = query.iter_mut().next() {
        *text = Text::new(format!("FPS: {fps:.0}"));
    }
}

fn hint_text(
    mode: CameraMode,
    normals: bool,
    section: bool,
    method: ActiveGravityMethod,
) -> String {
    let mode_str = match mode {
        CameraMode::Overview => "Overview",
        CameraMode::FollowCassini => "Follow Cassini",
    };
    let n_str = if normals { "ON" } else { "OFF" };
    let s_str = if section { "ON" } else { "OFF" };
    format!(
        "Press 'S': View | 'F': Normals | 'D': Section | 'G': Method [{}] | Mode: [{}] | N: [{}] | S: [{}]",
        method.as_str(),
        mode_str,
        n_str,
        s_str
    )
}

pub fn normal_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    active_method: ResMut<ActiveGravityMethod>,
    mut show_normals: ResMut<ShowNormals>,
    show_section: Res<ShowSection>,
    mode: Res<CameraMode>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyF) {
        return;
    }
    show_normals.0 = !show_normals.0;
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}

pub fn section_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    active_method: ResMut<ActiveGravityMethod>,
    mut show_section: ResMut<ShowSection>,
    show_normals: Res<ShowNormals>,
    mode: Res<CameraMode>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyD) {
        return;
    }
    show_section.0 = !show_section.0;
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}
pub fn method_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut active_method: ResMut<ActiveGravityMethod>,
    mode: Res<CameraMode>,
    show_normals: Res<ShowNormals>,
    show_section: Res<ShowSection>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
    mut cassini_query: Query<
        (&mut Transform, &mut Velocity, &mut OrbitHistory),
        With<CassiniMarker>,
    >,
    mut ryugu_query: Query<&mut Transform, (With<RyuguMarker>, Without<CassiniMarker>)>,
) {
    if !keyboard.just_pressed(KeyCode::KeyG) {
        return;
    }
    *active_method = match *active_method {
        ActiveGravityMethod::VoxelStehfest => ActiveGravityMethod::DecomposedWerner,
        ActiveGravityMethod::DecomposedWerner => ActiveGravityMethod::VoxelStehfest,
    };
    if let Ok((mut c_transform, mut c_velocity, mut c_history)) = cassini_query.single_mut() {
        if let Some(mut r_transform) = ryugu_query.iter_mut().next() {
            c_transform.translation = PROBE_R0;
            c_velocity.0 = *PROBE_V_INIT;
            // Reset probe state so the new trajectory starts clean: drop the old
            // history line, undo accumulated spin, and keep Ryugu centered at CoM.
            c_history.0.clear();
            r_transform.rotation = Quat::IDENTITY;
            r_transform.translation = Vec3::ZERO;
        }
    }
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}
pub fn update_hint_on_mode_change(
    active_method: ResMut<ActiveGravityMethod>,
    mode: Res<CameraMode>,
    show_normals: Res<ShowNormals>,
    show_section: Res<ShowSection>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !mode.is_changed() && !show_normals.is_changed() && !show_section.is_changed() {
        return;
    }
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(
            *mode,
            show_normals.0,
            show_section.0,
            *active_method,
        ));
    }
}
