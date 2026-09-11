---
version: alpha
name: "Ryugu Dynamics Laboratory"
description: "A dense WebGPU research console for inspecting asteroid gravity, trajectory propagation, inversion, and numerical certification."
colors:
  background: "#010304"
  surface: "#050F13"
  surface-raised: "#09191E"
  primary: "#48E7E2"
  text: "#D9EEF0"
  muted: "#789097"
  success: "#43E58A"
  warning: "#FFE071"
  danger: "#FF7272"
  eq106: "#36E7F2"
  werner: "#FF7D89"
  fft: "#FFB23D"
  fmm: "#42DC77"
typography:
  sans:
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
  mono:
    fontFamily: "ui-monospace, monospace"
rounded:
  DEFAULT: "0.375rem"
  sm: "0.25rem"
  md: "0.375rem"
  lg: "0.75rem"
spacing:
  section-gap: "0.875rem"
  page-max: "100vw"
components:
  button: {}
  card: {}
  dialog: {}
  range: {}
  chart: {}
---

# Ryugu Dynamics Laboratory Design System

## Overview

### Creative North Star

The interface should feel like a flight-dynamics test bench: dark instrument glass, thin cyan geometry, compact numerical readouts, and focused telemetry plots. The 3D body remains the visual center; controls frame it instead of covering it.

### Product context and register

- **Audience and primary job:** numerical-method researchers comparing gravity evaluators, checking Eq.121 propagation quality and Eq.184 inversion, and running density inversion.
- **Target market(s) and evidence:** research and engineering use; no region-specific workflow is encoded.
- **Locale(s) and language policy:** current controls are concise technical English; the shell supports Chinese user context without changing symbols, units, or method names.
- **Usage scene:** desktop and mobile GPU-capable browsers, with readable controls and explicit gesture ownership during long calculations.
- **Register:** scientific product UI, not a marketing surface.
- **Memorable signature:** The scene caption exposes the active operator chain; telemetry tracks the selected observable.
- **Restraint:** buttons, sliders, errors, and benchmark controls remain conventional and explicitly labeled.
- **Anti-references:** avoid game HUD ornament, neon glow overload, pill-heavy dashboards, and decorative motion that competes with measured data.
- **Token ownership/runtime mapping:** `src/html/ui.css` owns instrument tokens and `src/html/tailwind.css` is compiled by Bun into `src/html/tailwind.generated.css`. This file mirrors accepted values and explains their use.

## Colors

Near-black `background` and restrained `surface` layers preserve contrast with the 3D view. `primary` is reserved for focus, selection, and primary actions. Method curves keep stable identities: frequency-domain (Eq.121/184) cyan, Werner red, packed FFT amber, and FMM green. Danger never relies on color alone and is paired with error text and a recovery action.

## Typography

Sans text labels controls and hierarchy; monospaced text carries measurements, units, status, and axes. Numeric labels use tabular-feeling monospace alignment and scientific notation outside compact human ranges. Method names and mathematical symbols retain their established casing.

## Layout

The workbench follows the current viewport dimensions, swapping width and height for quarter-turn display rotation. Desktop uses three columns: initial conditions, surface fields and trajectory editing at left; the asteroid and telemetry at center; inversion and planning at right. Method navigation occupies its own row. Container queries follow the rotated workbench width, not the unrotated screen width. At 820px and below, the scene, telemetry and control rails stack in a scrollable workbench. Safe-area padding keeps controls clear of screen cutouts. Floating panel dragging and resizing remain available, and Reset UI view restores their placement.

Gestures default to the native Bevy camera. The explicit Gestures button transfers scene pinch, wheel and pan input to the whole workbench. A gesture finishes before ownership changes, so camera touches cannot become stuck. Workbench zoom is anchored at the gesture center; panel dragging accounts for rotation and scale. Navigation is a standalone script and remains usable while the Vue interface initializes.

Background calculation retains the Worker/RAF scheduler, visibility recovery and local loopback keepalive. The optional Keep screen awake control requests a screen wake lock while visible and reacquires it after returning; it does not replace background scheduling. Status text reports browser support and acquisition failures.

## Elevation & Depth

Hierarchy comes from tonal surfaces, thin cyan borders, without costly panel backdrop blur. Modal dimming separates test workflows from the live scene. Charts and dense status blocks use flat inner surfaces; they do not receive independent shadows.

## Shapes

Controls use 6px corners and panels use 10–12px corners. Plot areas and result strips remain nearly square to read as instruments. Full pills and gratuitous circular containers are not part of the system.

## Components

### Foundational visual states

Hover strengthens the cyan border; `focus-visible` uses a two-pixel cyan outline; selected controls use a cyan fill with dark text. Disabled controls retain their geometry at reduced opacity. Running, ready, and failed operations combine text with warning, success, or danger color.

### Buttons and actions

Primary fill is limited to the highest-value action in a local group, such as trajectory inversion. Repeated benchmark and cancellation actions retain labels. Busy actions disable in place so the layout does not jump.

### Navigation and data display

Method and metric selectors behave as single-choice pressed controls. Workload selection and benchmark method inclusion expose pressed state. Charts share axis typography, grid treatment, and stable method colors; empty charts state which samples are pending.

### Forms and overlays

Probe and acceleration ranges display their current value beside the label. Benchmark and error dialogs are modal, close on Escape where recovery is safe, restore focus, and retain explicit close or reset buttons.

### Iconography

No external icon family is required. The close mark is the only symbol-only action and always has an accessible label; scientific actions remain text-labeled.

### Motion

Feedback transitions last about 160ms and communicate hover or press state only. Reduced-motion preference collapses transitions and animations.

### Content and data visualization

Voice is concise and diagnostic. Units appear in labels or values. Residual and timing magnitudes use logarithmic axes where their ranges demand it; status text reports accepted/rejected segments and solver iterations alongside the plots.

### Planning progress

The Vue planning-progress component shows First, Stress, and Quadrature as determinate completion and verification bars from the Rust UI snapshot. Unscheduled workloads remain visibly ready instead of pretending to progress.

## Do's and Don'ts

- **Do:** keep method colors identical across performance, residual, planning, and quadrature views.
- **Do:** preserve measured-versus-estimated wording and expose pending/failed states during long GPU work.
- **Don't:** hide calculation state behind an indefinite blank chart or spinner.
- **Don't:** add decorative cards, gradients, or animations that weaken the scene and telemetry hierarchy.
