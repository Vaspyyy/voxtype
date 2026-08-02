//! `voxtype-osd-gtk4` — GTK4 + gtk4-layer-shell on-screen mic visualizer
//! for the Voxtype daemon.
//!
//! Renders a click-through, layer-shell-anchored window containing a compact
//! rounded-bar dictation pill. Audio frames arrive on
//! the daemon's audio Unix socket via [`voxtype::osd::ipc::run_ipc_loop`],
//! decoded into [`AudioFrame`]s by a tokio runtime on a worker thread, and
//! pushed into a shared [`FrameRing`] + [`PeakHold`]. The GTK side polls a
//! ~60 Hz `glib::timeout_add_local` callback that redraws the
//! `DrawingArea` whenever new frames have arrived.
//!
//! When the IPC socket is silent for `idle_timeout_secs` (Idle proxy) the
//! window is hidden so the binary does no rendering work and consumes
//! effectively zero CPU. It reappears when frames resume.
//!
//! Run with `RUST_LOG=debug` for verbose logs.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cairo::{Context, RectangleInt, Region};
use clap::Parser;
use gtk4::glib;
use gtk4::prelude::*;
use gtk4::{Application, ApplicationWindow, CssProvider, DrawingArea};
use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

use voxtype::audio::levels::{AudioFrame, FRAME_HZ};
use voxtype::config::Config as VoxtypeConfig;
use voxtype::osd::config::{OsdConfig, OsdPosition};
use voxtype::osd::ipc::{resolve_socket_path, run_ipc_loop, FrameRing, DEFAULT_RING_DEPTH};
use voxtype::osd::theme::ThemeWatcher;
use voxtype::osd::visual::{project_pill_levels, Palette, PeakHold};

/// Load the `[osd]` section from the voxtype config file, falling back to
/// `OsdConfig::default()` on any error (file missing, unreadable, parse
/// failure, or `[osd]` section absent).
///
/// We deliberately ignore parse errors instead of returning them: the OSD
/// is a side car, and a malformed config shouldn't prevent it from running
/// with sensible defaults — the user will see the daemon complain about
/// the same file separately.
fn load_osd_config_from_file(explicit: Option<&std::path::Path>) -> OsdConfig {
    let path = explicit
        .map(std::path::Path::to_path_buf)
        .or_else(VoxtypeConfig::default_path);
    let Some(path) = path else {
        return OsdConfig::default();
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return OsdConfig::default(),
    };

    #[derive(serde::Deserialize, Default)]
    struct PartialConfig {
        #[serde(default)]
        osd: Option<OsdConfig>,
    }

    match toml::from_str::<PartialConfig>(&content) {
        Ok(p) => p.osd.unwrap_or_default(),
        Err(_) => OsdConfig::default(),
    }
}

/// Application id for the GTK4 frontend.
const APP_ID: &str = "io.voxtype.OsdGtk4";

/// Render tick period (~60 Hz). The redraw is gated on whether new frames
/// have arrived since the last paint, so this is a cheap upper bound.
const RENDER_TICK_MS: u32 = 16;

/// How long we wait without frames before treating the daemon as idle and
/// hiding the surface. Matches the BRIEF's "Idle: surface destroyed" rule.
const IDLE_TIMEOUT_SECS: f32 = 0.15;

/// Number of rounded columns in the compact dictation pill.
const PILL_BAR_COUNT: usize = 19;

/// Audio history shown by the bars. At 100 Hz this is just under half a
/// second, enough to feel fluid without looking like a dense oscilloscope.
const PILL_RECENT_FRAMES: usize = 44;

/// The old waveform gain was tuned for raw full-height samples. Dial it back
/// before the pill's response curve so normal speech has useful headroom.
const PILL_GAIN_SCALE: f32 = 0.45;

struct PillAnimation {
    levels: Vec<f32>,
}

#[derive(Clone, Copy)]
struct PillStyle {
    palette: Palette,
    gain: f32,
    opacity: f64,
}

impl PillAnimation {
    fn new() -> Self {
        Self {
            levels: vec![0.0; PILL_BAR_COUNT],
        }
    }

    fn approach(&mut self, targets: &[f32]) {
        for (level, target) in self.levels.iter_mut().zip(targets) {
            let response = if *target > *level { 0.62 } else { 0.24 };
            *level += (*target - *level) * response;
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "voxtype-osd-gtk4",
    version,
    about = "Voxtype on-screen mic visualizer (GTK4 + gtk4-layer-shell)"
)]
struct Args {
    /// Path to the voxtype config file. Defaults to
    /// `~/.config/voxtype/config.toml`. Only the `[osd]` section is read.
    #[arg(long, env = "VOXTYPE_CONFIG")]
    config: Option<PathBuf>,

    /// Path to the audio-frame Unix socket. Defaults to
    /// `$XDG_RUNTIME_DIR/voxtype/audio.sock`.
    #[arg(long, env = "VOXTYPE_OSD_SOCKET")]
    socket: Option<PathBuf>,

    /// Seconds to wait between reconnect attempts when the daemon is down.
    #[arg(long, default_value = "1.0", env = "VOXTYPE_OSD_RECONNECT_SECS")]
    reconnect_secs: f32,

    /// Print one debug line per N frames received (0 = quiet).
    #[arg(long, default_value = "0", env = "VOXTYPE_OSD_LOG_EVERY")]
    log_every: u32,

    /// Held-peak decay rate in dB/sec.
    #[arg(long, default_value = "6.0", env = "VOXTYPE_OSD_PEAK_DECAY")]
    peak_decay_db_per_sec: f32,

    /// Surface width in physical pixels.
    #[arg(long, env = "VOXTYPE_OSD_WIDTH")]
    width_px: Option<u32>,

    /// Surface height in physical pixels.
    #[arg(long, env = "VOXTYPE_OSD_HEIGHT")]
    height_px: Option<u32>,

    /// Margin from the screen edge in physical pixels.
    #[arg(long, env = "VOXTYPE_OSD_MARGIN")]
    margin_px: Option<u32>,

    /// Visual gain applied to audio samples before drawing the waveform.
    /// Higher = waveform fills more of the vertical for quiet inputs.
    /// Reduce for hot mics (e.g. 4.0); raise for quiet sources (e.g. 14.0).
    #[arg(long, env = "VOXTYPE_OSD_GAIN")]
    waveform_gain: Option<f32>,
}

/// State shared between the IPC worker and the GTK redraw timer.
struct SharedState {
    ring: Mutex<FrameRing>,
    peak: Mutex<PeakHold>,
    last_seq: Mutex<u64>,
    last_frame_at: Mutex<Instant>,
}

impl SharedState {
    fn new(decay_db_per_sec: f32) -> Self {
        Self {
            ring: Mutex::new(FrameRing::new(DEFAULT_RING_DEPTH)),
            peak: Mutex::new(PeakHold::new(decay_db_per_sec)),
            last_seq: Mutex::new(0),
            last_frame_at: Mutex::new(Instant::now() - Duration::from_secs(3600)),
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let socket_path = resolve_socket_path(args.socket.clone());

    // Layer config: defaults < config file [osd] < CLI/env overrides.
    let mut osd_cfg = load_osd_config_from_file(args.config.as_deref());
    if let Some(w) = args.width_px {
        osd_cfg.width_px = w;
    }
    if let Some(h) = args.height_px {
        osd_cfg.height_px = h;
    }
    if let Some(m) = args.margin_px {
        osd_cfg.margin_px = m;
    }
    if let Some(g) = args.waveform_gain {
        osd_cfg.waveform_gain = g;
    }
    // peak_decay_db_per_sec has a clap default value, so this always
    // overrides whatever the file said. That's intentional: if the user
    // passes the flag, honor it; if they don't, the clap default kicks in.
    osd_cfg.peak_decay_db_per_sec = args.peak_decay_db_per_sec;

    tracing::info!(
        "voxtype-osd-gtk4 starting; socket={:?} size={}x{} margin={} pos={:?}",
        socket_path,
        osd_cfg.width_px,
        osd_cfg.height_px,
        osd_cfg.margin_px,
        osd_cfg.position,
    );

    let theme = ThemeWatcher::new();
    let palette = theme.palette();

    let state = Arc::new(SharedState::new(osd_cfg.peak_decay_db_per_sec));

    // Spawn the tokio IPC worker on a side thread.
    spawn_ipc_worker(
        state.clone(),
        socket_path,
        args.reconnect_secs,
        args.log_every,
    );

    // GTK application owns the main thread.
    let app = Application::builder().application_id(APP_ID).build();

    let cfg = osd_cfg.clone();
    let state_for_activate = state.clone();
    app.connect_activate(move |app| {
        build_window(app, &cfg, palette, state_for_activate.clone());
    });

    // GTK's run() consumes argv; we've already parsed via clap, so feed
    // it an empty vector to keep it from re-parsing.
    let exit = app.run_with_args::<&str>(&[]);
    let code: u8 = exit.into();
    if code != 0 {
        anyhow::bail!("GTK application exited with status {}", code);
    }
    Ok(())
}

/// Spawn the tokio runtime + IPC loop on a dedicated thread.
fn spawn_ipc_worker(
    state: Arc<SharedState>,
    socket_path: PathBuf,
    reconnect_secs: f32,
    log_every: u32,
) {
    std::thread::Builder::new()
        .name("voxtype-osd-ipc".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::error!("Failed to build tokio runtime: {e}");
                    return;
                }
            };

            let dt_per_frame = 1.0 / FRAME_HZ as f32;
            let mut total: u64 = 0;
            let mut last_log = Instant::now();

            let on_frame = move |frame: AudioFrame| {
                if let Ok(mut r) = state.ring.lock() {
                    r.push(frame);
                }
                if let Ok(mut p) = state.peak.lock() {
                    p.update(frame.peak_dbfs, dt_per_frame);
                }
                if let Ok(mut s) = state.last_seq.lock() {
                    *s = s.wrapping_add(1);
                }
                if let Ok(mut t) = state.last_frame_at.lock() {
                    *t = Instant::now();
                }

                total += 1;
                if log_every > 0 && total.is_multiple_of(u64::from(log_every)) {
                    let elapsed = last_log.elapsed().as_secs_f32();
                    let rate = if elapsed > 0.0 {
                        log_every as f32 / elapsed
                    } else {
                        0.0
                    };
                    tracing::debug!(
                        target: "osd::frames",
                        frontend = "gtk4",
                        seq = frame.seq,
                        peak_dbfs = frame.peak_dbfs,
                        min = frame.min,
                        max = frame.max,
                        rate_hz = rate,
                        "frame batch"
                    );
                    last_log = Instant::now();
                }
            };

            rt.block_on(run_ipc_loop(socket_path, reconnect_secs, on_frame));
        })
        .expect("spawn ipc worker thread");
}

/// Best-effort monitor height in physical pixels for translating the
/// fractional `top_margin` config into a layer-shell pixel offset.
///
/// Tries the GDK display's currently-focused monitor first (which is the
/// monitor the user is most likely looking at, and the one swayosd targets
/// via `--monitor`). If that fails — display unavailable, no monitors
/// enumerated — returns None and the caller falls back to a conservative
/// default.
fn focused_monitor_height_px() -> Option<i32> {
    use gtk4::gdk;
    let display = gdk::Display::default()?;
    let monitors = display.monitors();
    // `monitors` is a GListModel; pull the first item with a non-zero
    // height. On multi-monitor setups this picks whichever the compositor
    // ordered first, which lines up with the layer-shell default in
    // practice.
    for i in 0..monitors.n_items() {
        if let Some(obj) = monitors.item(i) {
            if let Ok(monitor) = obj.downcast::<gdk::Monitor>() {
                let h = monitor.geometry().height();
                if h > 0 {
                    return Some(h);
                }
            }
        }
    }
    None
}

/// Build the GTK window, attach layer-shell config, mount the DrawingArea,
/// and start the redraw tick.
fn build_window(app: &Application, cfg: &OsdConfig, palette: Palette, state: Arc<SharedState>) {
    let window = ApplicationWindow::builder()
        .application(app)
        .default_width(cfg.width_px as i32)
        .default_height(cfg.height_px as i32)
        .resizable(false)
        .decorated(false)
        .build();

    // GTK otherwise paints the rectangular application-window background
    // behind Cairo. Keep it transparent so only the rounded pill is visible.
    let css = CssProvider::new();
    css.load_from_data(
        "window.voxtype-osd, \
         window.voxtype-osd.background, \
         window.voxtype-osd decoration { \
             background-color: transparent; \
             background-image: none; \
             border: none; \
             box-shadow: none; \
             outline: none; \
         }",
    );
    gtk4::style_context_add_provider_for_display(
        &gtk4::gdk::Display::default().expect("GTK display unavailable"),
        &css,
        gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
    );
    window.add_css_class("voxtype-osd");

    // Layer-shell setup: top layer, no keyboard, anchored per config.
    window.init_layer_shell();
    window.set_layer(Layer::Overlay);
    window.set_keyboard_mode(KeyboardMode::None);
    window.set_namespace(Some("voxtype-osd"));

    // Centered positions use swayosd-style fractional placement:
    // anchor only to Edge::Top with a margin of `top_margin × monitor_height`
    // and let the layer shell center horizontally. Matches what users
    // already see for volume/brightness/media-key feedback so the voxtype
    // OSD lands in the familiar band.
    //
    // Corner positions still use the absolute `margin_px` model — fractional
    // doesn't make sense there.
    let centered = matches!(
        cfg.position,
        OsdPosition::BottomCenter | OsdPosition::TopCenter
    );

    if centered {
        // Resolve monitor height to translate the fractional offset into
        // pixels. Falls back to a conservative 1080 if the display can't be
        // queried (extremely rare on Wayland-only systems where layer-shell
        // is supported at all).
        let monitor_height = focused_monitor_height_px().unwrap_or(1080);
        let top_px = (cfg.top_margin.clamp(0.0, 1.0) * monitor_height as f32) as i32;

        window.set_anchor(Edge::Top, true);
        window.set_anchor(Edge::Bottom, false);
        // Don't anchor Left/Right -> layer shell auto-centers horizontally.
        window.set_anchor(Edge::Left, false);
        window.set_anchor(Edge::Right, false);
        window.set_margin(Edge::Top, top_px);
    } else {
        // Corner positions: legacy anchor + uniform pixel margin behavior.
        let (anchor_top, anchor_bottom, anchor_left, anchor_right) = match cfg.position {
            OsdPosition::BottomLeft => (false, true, true, false),
            OsdPosition::BottomRight => (false, true, false, true),
            OsdPosition::TopLeft => (true, false, true, false),
            OsdPosition::TopRight => (true, false, false, true),
            // Centered branch is handled above; unreachable here.
            OsdPosition::BottomCenter | OsdPosition::TopCenter => unreachable!(),
        };
        window.set_anchor(Edge::Top, anchor_top);
        window.set_anchor(Edge::Bottom, anchor_bottom);
        window.set_anchor(Edge::Left, anchor_left);
        window.set_anchor(Edge::Right, anchor_right);

        let m = cfg.margin_px as i32;
        if anchor_top {
            window.set_margin(Edge::Top, m);
        }
        if anchor_bottom {
            window.set_margin(Edge::Bottom, m);
        }
        if anchor_left {
            window.set_margin(Edge::Left, m);
        }
        if anchor_right {
            window.set_margin(Edge::Right, m);
        }
    }

    // Don't reserve space on the output: the OSD floats over windows.
    window.set_exclusive_zone(0);

    // The drawing area fills the window.
    let drawing_area = DrawingArea::new();
    drawing_area.set_content_width(cfg.width_px as i32);
    drawing_area.set_content_height(cfg.height_px as i32);
    let state_for_draw = state.clone();
    let style = PillStyle {
        palette,
        gain: cfg.waveform_gain,
        opacity: cfg.opacity as f64,
    };
    let animation = Rc::new(RefCell::new(PillAnimation::new()));
    let animation_for_draw = animation.clone();
    drawing_area.set_draw_func(move |_area, cr, w, h| {
        draw(cr, w, h, &state_for_draw, &animation_for_draw, &style);
    });
    window.set_child(Some(&drawing_area));

    // Click-through: install an empty input region on the window's surface
    // once it's realized. Until then `realize` hasn't allocated a surface.
    {
        let window_ref = window.clone();
        window.connect_realize(move |_| {
            apply_click_through(&window_ref);
        });
    }

    // Redraw timer. We only call queue_draw() when the IPC has produced a
    // newer seq than the last paint, so this is cheap when idle.
    let redraw_state = state.clone();
    let redraw_area = drawing_area.clone();
    let redraw_window = window.clone();
    let redraw_animation = animation.clone();
    let last_drawn_seq = Cell::new(0u64);
    // Tracks GTK visibility. Starts true because `window.present()` below maps
    // the surface, the first tick's idle check then hides it.
    let visible = Cell::new(true);

    glib::timeout_add_local(Duration::from_millis(RENDER_TICK_MS as u64), move || {
        let cur_seq = redraw_state.last_seq.lock().map(|s| *s).unwrap_or(0);
        let last_at = redraw_state
            .last_frame_at
            .lock()
            .map(|t| *t)
            .unwrap_or_else(|_| Instant::now() - Duration::from_secs(3600));
        let idle = last_at.elapsed().as_secs_f32() > IDLE_TIMEOUT_SECS;

        if idle {
            if visible.get() {
                tracing::info!("hiding (idle for {:.2}s)", last_at.elapsed().as_secs_f32());
                redraw_window.set_visible(false);
                if let Ok(mut ring) = redraw_state.ring.lock() {
                    ring.clear();
                }
                redraw_animation.borrow_mut().levels.fill(0.0);
                visible.set(false);
            }
            return glib::ControlFlow::Continue;
        }

        if !visible.get() {
            tracing::info!(
                "showing (frame seq={}, last_at={:.3}s ago)",
                cur_seq,
                last_at.elapsed().as_secs_f32()
            );
            redraw_window.set_visible(true);
            visible.set(true);
        }

        // Decay the held peak even when no new frame arrived this tick.
        if let Ok(mut p) = redraw_state.peak.lock() {
            let dt = (RENDER_TICK_MS as f32) / 1000.0;
            // We pass the most recent peak from the ring as the "current"
            // value so a stale update doesn't snap the held value back up.
            let cur_peak = redraw_state
                .ring
                .lock()
                .ok()
                .and_then(|r| r.latest())
                .map(|f| f.peak_dbfs)
                .unwrap_or(-120.0);
            // Only decay; the IPC callback already snapped up on each
            // received frame. Calling update here with a non-louder peak
            // keeps the linear decay running at render rate.
            if cur_peak <= p.held_dbfs {
                p.update(cur_peak, dt);
            }
        }

        if cur_seq != last_drawn_seq.get() {
            redraw_area.queue_draw();
            last_drawn_seq.set(cur_seq);
        }
        glib::ControlFlow::Continue
    });

    // Map the layer-shell surface once. The redraw timer will hide it
    // immediately on its first tick (no frames yet → idle), and toggle
    // visibility from there. Mapping once at startup keeps Hyprland's
    // layer-shell state machine happy across hide/show cycles.
    window.present();
}

/// Set an empty input region on the GdkSurface so clicks pass through.
fn apply_click_through(window: &ApplicationWindow) {
    let Some(surface) = window.surface() else {
        tracing::warn!("Window has no surface yet; click-through not applied");
        return;
    };
    let empty = Region::create_rectangle(&RectangleInt::new(0, 0, 0, 0));
    surface.set_input_region(Some(&empty));
}

/// Render the compact rounded-bar dictation pill into the Cairo context.
fn draw(
    cr: &Context,
    width: i32,
    height: i32,
    state: &Arc<SharedState>,
    animation: &Rc<RefCell<PillAnimation>>,
    style: &PillStyle,
) {
    let w = width as f64;
    let h = height as f64;
    if w <= 0.0 || h <= 0.0 {
        return;
    }

    // Clear the transparent layer surface before painting the floating shape.
    cr.set_operator(cairo::Operator::Clear);
    cr.paint().ok();
    cr.set_operator(cairo::Operator::Over);

    // A small inset leaves room for antialiasing and a restrained drop shadow.
    let inset = 2.0;
    let pill_x = inset;
    let pill_y = inset;
    let pill_w = (w - inset * 2.0).max(1.0);
    let pill_h = (h - inset * 2.0).max(1.0);
    let radius = pill_h * 0.5;

    rounded_rectangle(cr, pill_x, pill_y + 1.0, pill_w, pill_h, radius);
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.22 * style.opacity.clamp(0.0, 1.0));
    cr.fill().ok();

    rounded_rectangle(cr, pill_x, pill_y, pill_w, pill_h, radius);
    cr.set_source_rgba(
        style.palette.background.r as f64,
        style.palette.background.g as f64,
        style.palette.background.b as f64,
        (style.palette.background.a as f64 * style.opacity).clamp(0.0, 1.0),
    );
    cr.fill().ok();

    let frames: Vec<AudioFrame> = match state.ring.lock() {
        Ok(r) => r.iter().collect(),
        Err(_) => return,
    };
    let targets = project_pill_levels(
        &frames,
        PILL_BAR_COUNT,
        PILL_RECENT_FRAMES,
        style.gain * PILL_GAIN_SCALE,
    );
    let mut animation = animation.borrow_mut();
    animation.approach(&targets);

    let horizontal_padding = (pill_h * 0.38).max(14.0);
    let bars_x = pill_x + horizontal_padding;
    let bars_w = (pill_w - horizontal_padding * 2.0).max(1.0);
    let bar_w = (pill_h * 0.095).clamp(4.0, 6.0);
    let gap = ((bars_w - bar_w * PILL_BAR_COUNT as f64) / PILL_BAR_COUNT.saturating_sub(1) as f64)
        .max(2.0);
    let total_w = bar_w * PILL_BAR_COUNT as f64 + gap * PILL_BAR_COUNT.saturating_sub(1) as f64;
    let start_x = bars_x + (bars_w - total_w) * 0.5;
    let center_y = pill_y + pill_h * 0.5;
    let min_bar_h = (pill_h * 0.13).max(bar_w);
    let max_bar_h = pill_h * 0.68;

    cr.set_source_rgba(
        style.palette.foreground.r as f64,
        style.palette.foreground.g as f64,
        style.palette.foreground.b as f64,
        0.96,
    );
    for (index, level) in animation.levels.iter().enumerate() {
        let bar_h = min_bar_h + (max_bar_h - min_bar_h) * f64::from(*level);
        let x = start_x + index as f64 * (bar_w + gap);
        let y = center_y - bar_h * 0.5;
        rounded_rectangle(cr, x, y, bar_w, bar_h, bar_w * 0.5);
        cr.fill().ok();
    }
}

fn rounded_rectangle(cr: &Context, x: f64, y: f64, w: f64, h: f64, radius: f64) {
    let radius = radius.min(w * 0.5).min(h * 0.5).max(0.0);
    let right = x + w;
    let bottom = y + h;

    cr.new_sub_path();
    cr.arc(
        right - radius,
        y + radius,
        radius,
        -std::f64::consts::FRAC_PI_2,
        0.0,
    );
    cr.arc(
        right - radius,
        bottom - radius,
        radius,
        0.0,
        std::f64::consts::FRAC_PI_2,
    );
    cr.arc(
        x + radius,
        bottom - radius,
        radius,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
    );
    cr.arc(
        x + radius,
        y + radius,
        radius,
        std::f64::consts::PI,
        std::f64::consts::PI + std::f64::consts::FRAC_PI_2,
    );
    cr.close_path();
}
