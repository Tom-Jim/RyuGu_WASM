use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use crate::components::*;

pub fn setup_fps_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("FPS: --"),
        TextFont { font_size: bevy::text::FontSize::Px(16.0), ..default() },
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

fn hint_text(mode: CameraMode, normals: bool, section: bool) -> String {
    let mode_str = match mode {
        CameraMode::Overview => "Overview",
        CameraMode::FollowCassini => "Follow Cassini",
    };
    let normals_str = if normals { "ON" } else { "OFF" };
    let section_str = if section { "ON" } else { "OFF" };
    format!(
        "Press 'S': View | Press 'F': Normals | Press 'D': Section | Mode: [{mode_str}] | Normals: [{normals_str}] | Section: [{section_str}]"
    )
}

pub fn normal_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut show_normals: ResMut<ShowNormals>,
    show_section: Res<ShowSection>,
    mode: Res<CameraMode>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyF) { return; }
    show_normals.0 = !show_normals.0;
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(*mode, show_normals.0, show_section.0));
    }
}

pub fn section_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut show_section: ResMut<ShowSection>,
    show_normals: Res<ShowNormals>,
    mode: Res<CameraMode>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyD) { return; }
    show_section.0 = !show_section.0;
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(*mode, show_normals.0, show_section.0));
    }
}

pub fn update_hint_on_mode_change(
    mode: Res<CameraMode>,
    show_normals: Res<ShowNormals>,
    show_section: Res<ShowSection>,
    mut text_query: Query<&mut Text, With<UiTextMarker>>,
) {
    if !mode.is_changed() && !show_normals.is_changed() && !show_section.is_changed() {
        return;
    }
    if let Some(mut text) = text_query.iter_mut().next() {
        *text = Text::new(hint_text(*mode, show_normals.0, show_section.0));
    }
}
