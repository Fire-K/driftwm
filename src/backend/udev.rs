use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Once;
use std::time::Duration;

use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::{
    backend::{
        allocator::{
            Format, Fourcc, Modifier,
            format::FormatSet,
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
        },
        drm::{
            DrmDevice, DrmDeviceFd, DrmDeviceNotifier, DrmEvent, DrmNode, NodeType,
            compositor::{DrmCompositor, FrameError, FrameFlags, PrimaryPlaneElement},
            exporter::gbm::GbmFramebufferExporter,
        },
        egl::{EGLContext, EGLDevice, EGLDisplay, context::ContextPriority, fence::EGLFence},
        libinput::{LibinputInputBackend, LibinputSessionInterface},
        renderer::{
            ImportDma, RendererSuper,
            gles::GlesRenderer,
            multigpu::{GpuManager, MultiFrame, MultiRenderer, gbm::GbmGlesBackend},
            sync::SyncPoint,
        },
        session::{Event as SessionEvent, Session, libseat::LibSeatSession},
        udev::{self, UdevBackend, UdevEvent},
    },
    output::{Mode, Output, PhysicalProperties, Subpixel},
    reexports::{
        calloop::{
            Dispatcher, EventLoop, RegistrationToken,
            channel::{Event as ChannelEvent, Sender, channel},
            timer::{TimeoutAction, Timer},
        },
        drm::control::{self, connector, crtc},
        input::Libinput,
        rustix::fs::OFlags,
    },
    utils::{DeviceFd, Transform},
    wayland::dmabuf::{DmabufFeedback, DmabufFeedbackBuilder},
};
use smithay::reexports::wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_feedback_v1::TrancheFlags;

use smithay_drm_extras::drm_scanner::{DrmScanEvent, DrmScanner};

use crate::backend::Backend;
use crate::backend::cvt;
use crate::backend::gamma::{GammaProps, set_gamma_for_crtc_legacy};
use crate::state::{CrtcKey, DriftWm};
use driftwm::config::{GpuScope, OutputMode as ConfigOutputMode};

const SUPPORTED_COLOR_FORMATS: &[Fourcc] = &[
    Fourcc::Xrgb8888,
    Fourcc::Xbgr8888,
    Fourcc::Argb8888,
    Fourcc::Abgr8888,
];

type GbmDrmCompositor = DrmCompositor<
    GbmAllocator<DrmDeviceFd>,
    GbmFramebufferExporter<DrmDeviceFd>,
    smithay::desktop::utils::OutputPresentationFeedback,
    DrmDeviceFd,
>;

// Multi-GPU rendering: render on the primary GPU, scan out (with an implicit
// PRIME copy) on outputs that live on other GPUs. Both type params are the same
// GBM-GLES backend; the render and target nodes differ at call time.
pub type MultiGpuRenderer<'render> = MultiRenderer<
    'render,
    'render,
    GbmGlesBackend<GlesRenderer, DrmDeviceFd>,
    GbmGlesBackend<GlesRenderer, DrmDeviceFd>,
>;

pub type MultiGpuFrame<'render, 'frame, 'buffer> = MultiFrame<
    'render,
    'render,
    'frame,
    'buffer,
    GbmGlesBackend<GlesRenderer, DrmDeviceFd>,
    GbmGlesBackend<GlesRenderer, DrmDeviceFd>,
>;

pub type MultiGpuRendererError<'render> = <MultiGpuRenderer<'render> as RendererSuper>::Error;

/// The udev backend's renderer state stored on `Backend::Udev`. Holds the
/// multi-GPU manager (one GLES renderer per DRM render node) and the primary
/// render node. `gpu_manager.single_renderer(&primary_render_node)` yields a
/// `MultiGpuRenderer` for same-GPU work; for an output on another GPU,
/// `gpu_manager.renderer(primary, target, fmt)` adds the implicit PRIME copy.
pub struct UdevRenderer {
    pub gpu_manager: GpuManager<GbmGlesBackend<GlesRenderer, DrmDeviceFd>>,
    pub primary_render_node: DrmNode,
    /// Render nodes of secondary GPUs currently registered on the manager
    /// (i.e. with an output attached). Empty in the common single-GPU and idle
    /// dGPU cases.
    pub secondary_render_nodes: HashSet<DrmNode>,
    /// KMS nodes of secondary GPUs whose GBM/EGL bring-up is running on a
    /// background thread (see `scan_device_connectors`). Guards against
    /// spawning a second probe for a device that is still waking up.
    gfx_probes_in_flight: HashSet<DrmNode>,
    /// Sends a finished background bring-up back to the main thread; cloned
    /// into each probe thread.
    gfx_probe_sender: Sender<GfxProbeResult>,
}

/// Result of a secondary GPU's GBM/EGL bring-up, run off the main thread so a
/// dGPU waking from runtime suspend (power-up + possible firmware reload)
/// never blocks rendering or input on other outputs. `None` means the probe
/// failed (logged on the worker thread already).
struct GfxProbeResult {
    node: DrmNode,
    gfx: Option<DeviceGfx>,
}

struct DeviceData {
    /// This device's KMS (primary) node — the key in `DriftWm::udev_devices`,
    /// and the allocation target advertised in scanout dmabuf-feedback tranches.
    kms_node: DrmNode,
    drm: DrmDevice,
    drm_fd: DrmDeviceFd,
    /// GBM/EGL/GLES state. Always present on the primary GPU; on secondary
    /// GPUs it exists only while an output is attached, so an idle dGPU can
    /// runtime-suspend (a live EGL context pins it awake).
    gfx: Option<DeviceGfx>,
    drm_scanner: DrmScanner,
    /// Failed bring-ups of a connector since the last success; bounds the
    /// delayed rescans that retry it (see `scan_device_connectors`).
    scan_retries: u8,
    surfaces: HashMap<crtc::Handle, SurfaceData>,
    /// Calloop token for this device's DRM (VBlank) event source; removed
    /// when the GPU is unplugged.
    drm_token: Option<RegistrationToken>,
}

struct DeviceGfx {
    gbm: GbmDevice<DrmDeviceFd>,
    /// This device's DRM render node (resolved via EGL; the KMS node itself on
    /// split-DRM systems without one). Outputs here scan out buffers rendered
    /// on the primary GPU — when the nodes differ, via an implicit PRIME copy.
    render_node: DrmNode,
    render_formats: Vec<Format>,
}

struct SurfaceData {
    compositor: GbmDrmCompositor,
    output: Output,
    connector: connector::Handle,
    make: String,
    model: String,
    serial_number: String,
    global: GlobalId,
    /// Atomic GAMMA_LUT/GAMMA_LUT_SIZE property handles. `None` if the driver
    /// doesn't expose them; in that case we fall back to legacy `set_gamma`.
    gamma_props: Option<GammaProps>,
    /// Gamma ramp queued while the session is inactive (VT switched away).
    /// Re-applied on session resume. `Some(Some(ramp))` = set to ramp,
    /// `Some(None)` = reset to identity, `None` = nothing pending.
    pending_gamma_change: Option<Option<Vec<u16>>>,
    /// Per-surface dmabuf feedback sent to clients shown on this output.
    /// `None` if building it failed — clients fall back to the default global.
    dmabuf_feedback: Option<SurfaceDmabufFeedback>,
}

/// Per-surface dmabuf feedback. `render` steers clients toward the primary
/// render node; `scanout` adds scanout-flagged tranches targeting this
/// output's KMS device so a fullscreen client allocates buffers its
/// DrmCompositor can promote to a plane. Which one a surface receives is
/// decided per frame from its render-element state (see
/// `render::send_dmabuf_feedbacks`).
pub struct SurfaceDmabufFeedback {
    pub render: DmabufFeedback,
    pub scanout: DmabufFeedback,
}

fn surface_dmabuf_feedback(
    compositor: &GbmDrmCompositor,
    primary_formats: FormatSet,
    primary_render_node: DrmNode,
    surface_render_node: DrmNode,
    surface_kms_node: DrmNode,
) -> Result<SurfaceDmabufFeedback, std::io::Error> {
    let surface = compositor.surface();
    let planes = surface.planes();

    let primary_plane_formats = surface.plane_info().formats.clone();
    let primary_or_overlay_plane_formats = primary_plane_formats
        .iter()
        .chain(planes.overlay.iter().flat_map(|p| p.formats.iter()))
        .copied()
        .collect::<FormatSet>();

    // Limit scanout tranches to formats we can also render from, so a buffer
    // that fails the scanout test still has a composite fallback path.
    let mut primary_scanout_formats = primary_plane_formats
        .intersection(&primary_formats)
        .copied()
        .collect::<Vec<_>>();
    let mut primary_or_overlay_scanout_formats = primary_or_overlay_plane_formats
        .intersection(&primary_formats)
        .copied()
        .collect::<Vec<_>>();

    // Cross-GPU scanout is only reliable with Linear buffers: iGPU+dGPU pairs
    // can share non-Linear modifiers on paper yet scan out glitched frames
    // (same workaround as niri).
    if surface_render_node != primary_render_node {
        primary_scanout_formats.retain(|f| f.modifier == Modifier::Linear);
        primary_or_overlay_scanout_formats.retain(|f| f.modifier == Modifier::Linear);
    }

    tracing::info!(
        "dmabuf feedback for {surface_kms_node}: {} plane formats, {} render-node formats, \
         scanout tranches {} + {}",
        primary_plane_formats.iter().count(),
        primary_formats.iter().count(),
        primary_scanout_formats.len(),
        primary_or_overlay_scanout_formats.len(),
    );

    let builder = DmabufFeedbackBuilder::new(primary_render_node.dev_id(), primary_formats);

    // Prefer primary-plane-only formats over primary-or-overlay: overlay
    // planes are disabled in render_frame, so this raises the chance a
    // client's buffer lands in the tranche that can actually scan out.
    let scanout = builder
        .clone()
        .add_preference_tranche(
            surface_kms_node.dev_id(),
            TrancheFlags::Scanout,
            primary_scanout_formats,
            4u32..=6,
        )
        .add_preference_tranche(
            surface_kms_node.dev_id(),
            TrancheFlags::Scanout,
            primary_or_overlay_scanout_formats,
            4u32..=6,
        )
        .build()?;

    // On the primary node the render path can scan out too — reuse the
    // scanout feedback to avoid advertising duplicate tranches.
    let render = if surface_render_node == primary_render_node {
        scanout.clone()
    } else {
        builder.build()?
    };

    Ok(SurfaceDmabufFeedback { render, scanout })
}

/// Opaque handle to udev backend device data, stored in
/// `DriftWm::udev_devices` keyed by KMS node. Rc-cloneable so the render loop
/// and gamma-control handler can each grab an independent `RefCell` borrow
/// without re-routing through DriftWm.
#[derive(Clone)]
pub(crate) struct UdevDevice(Rc<RefCell<DeviceData>>);

/// Apply (or clear, with `None`) a gamma ramp on `surface` via whichever
/// path the CRTC supports — atomic GAMMA_LUT first, legacy ioctl fallback.
fn apply_gamma(
    surface: &mut SurfaceData,
    drm: &DrmDevice,
    crtc: crtc::Handle,
    ramp: Option<&[u16]>,
) -> Option<()> {
    if let Some(gp) = &mut surface.gamma_props {
        gp.set_gamma(drm, ramp)
    } else {
        set_gamma_for_crtc_legacy(drm, crtc, ramp)
    }
}

impl UdevDevice {
    /// The CRTC driving `output`, for callers that hold output-keyed state and
    /// need to consult the CRTC-keyed frame bookkeeping.
    pub(crate) fn crtc_for_output(&self, output: &Output) -> Option<CrtcKey> {
        let dev = self.0.borrow();
        dev.surfaces
            .iter()
            .find(|(_, s)| s.output == *output)
            .map(|(crtc, _)| (dev.kms_node, *crtc))
    }

    /// Look up the per-output gamma LUT size. Prefers atomic GAMMA_LUT_SIZE;
    /// falls back to the CRTC's legacy `gamma_length`. Returns `None` if the
    /// CRTC reports size 0 (e.g. Apple DCP on Asahi, virtual outputs without
    /// gamma support) so the protocol cleanly fails the control rather than
    /// advertising a 0-entry LUT.
    pub(crate) fn get_gamma_size(&self, output: &Output) -> Option<u32> {
        use smithay::reexports::drm::control::Device as _;
        let dev = self.0.borrow();
        let (crtc, surface) = dev.surfaces.iter().find(|(_, s)| s.output == *output)?;
        let size = if let Some(gp) = &surface.gamma_props {
            gp.gamma_size(&dev.drm)?
        } else {
            dev.drm.get_crtc(*crtc).ok()?.gamma_length()
        };
        (size != 0).then_some(size)
    }

    /// Apply a gamma ramp (or reset to identity if `None`). Atomic path if
    /// the driver exposes GAMMA_LUT; legacy ioctl otherwise. If the session
    /// is inactive (VT switched away), the ramp is queued on the surface
    /// and re-applied on resume.
    pub(crate) fn set_gamma(&self, output: &Output, ramp: Option<Vec<u16>>) -> Option<()> {
        let mut dev = self.0.borrow_mut();
        let DeviceData { drm, surfaces, .. } = &mut *dev;
        let (crtc, surface) = surfaces.iter_mut().find(|(_, s)| s.output == *output)?;

        if !drm.is_active() {
            surface.pending_gamma_change = Some(ramp);
            return Some(());
        }

        apply_gamma(surface, drm, *crtc, ramp.as_deref())
    }
}

/// Tick animations once for all outputs, mark dirty CRTCs, then render.
///
/// Clones each device's `Rc` handle out of `data.udev_devices` first so a
/// borrow stays independent of mutations on `data`. Global per-frame work
/// (DPMS drain, mode changes, foreign-toplevel + output-management refresh)
/// runs once and is routed to the owning device; dirty-marking and rendering
/// iterate every device.
pub(crate) fn render_if_needed(data: &mut DriftWm) {
    // Fast path: nothing needs attention — skip all work when idle
    let any_chunked_pending = data
        .render
        .cached_tile_chunks
        .values()
        .any(|c| c.has_pending_loads())
        || data
            .render
            .cached_shader_chunks
            .values()
            .any(|c| c.has_pending_bakes());
    if data.redraws_needed.is_empty()
        && !data.has_active_animations()
        && !data.background_animation_due_any()
        && !data.output_config_dirty
        && data.pending_dpms.is_empty()
        && !any_chunked_pending
    {
        // A capped animated background still needs a wake-up for its next
        // tick: no event fires on an idle desktop, and without this the
        // animation would only advance alongside other redraws (stutter).
        let fps = data.config.background.animate_fps;
        let eligible: Vec<String> = data.background_render_eligible_output_names().collect();
        if data.render.background_is_animated
            && fps > 0
            && !data.render.background_tick_armed
            && !eligible.is_empty()
        {
            let interval = Duration::from_secs_f64(1.0 / fps as f64);
            // Wake for the soonest-due eligible output; stamps are per-output.
            // The stamp of an output that stopped rendering its background —
            // DPMS-off, or fullscreen with the canvas concealed — goes stale
            // forever; excluding it here keeps a long-dead stamp from
            // collapsing this to the 1ms floor and busy-rescheduling.
            let elapsed = data
                .render
                .background_last_animate
                .iter()
                .filter(|(name, _)| eligible.contains(name))
                .map(|(_, t)| t.elapsed())
                .max()
                .unwrap_or(interval);
            let wait = interval
                .saturating_sub(elapsed)
                .max(Duration::from_millis(1));
            if data
                .loop_handle
                .insert_source(Timer::from_duration(wait), |_, _, data: &mut DriftWm| {
                    data.render.background_tick_armed = false;
                    render_if_needed(data);
                    TimeoutAction::Drop
                })
                .is_ok()
            {
                data.render.background_tick_armed = true;
            }
        }
        return;
    }

    // Free capture textures left by finished screenshot/screencast clients
    // (kept warm while one renders into them). Only fires on render-active
    // cycles, so a fully-idle stop frees on next activity — memory, not battery.
    data.render
        .evict_idle_capture_state(data.start_time.elapsed());

    // Clone the device handles out so each borrow is independent of `data`.
    let devices: Vec<UdevDevice> = data.udev_devices.values().cloned().collect();
    if devices.is_empty() {
        return;
    }

    // 1. Tick animations once for all outputs (before device borrows)
    data.tick_all_animations();

    // After the tick, so a camera warp's motion and any cursor change it
    // implies go into this frame.
    data.refresh_pointer_focus();

    // 2. Drain pending DPMS transitions before animation marking so DPMS-off
    //    outputs don't get re-dirtied below. Each output lives on exactly one
    //    device, so find the surface across all devices.
    if !data.pending_dpms.is_empty() {
        let pending: Vec<(Output, bool)> = data.pending_dpms.drain().collect();
        for (output, on) in &pending {
            let mut handled = false;
            for device in &devices {
                let mut dev = device.0.borrow_mut();
                let kms_node = dev.kms_node;
                let Some((&crtc, surface)) =
                    dev.surfaces.iter_mut().find(|(_, s)| s.output == *output)
                else {
                    continue;
                };
                let key: CrtcKey = (kms_node, crtc);
                handled = true;
                if *on {
                    data.redraws_needed.insert(output.clone());
                } else {
                    let cleared = surface.compositor.clear();
                    match cleared {
                        Ok(()) => {
                            // Only now is the panel dark and idle: smithay drops its
                            // `pending_frame` inside a successful `clear`, so no flip
                            // is outstanding and nothing will render here again while
                            // the output is off. Forgetting a flight that is still in
                            // the air would let the render gate re-enter
                            // `render_frame` inside the in-flight window and credit
                            // the next VBlank to the wrong frame.
                            data.redraws_needed.remove(output);
                            data.frames_pending.remove(&key);
                            if let Some(token) = data.estimated_vblank_timers.remove(&key) {
                                data.loop_handle.remove(token);
                            }
                            // A dark panel satisfies the lock as well as a lock frame
                            // does, and this output will never render again while it
                            // is off — so an awaited one has to report in here or the
                            // confirmation waits out the whole timeout for a frame
                            // that cannot come.
                            data.forget_lock_frame(key);
                            data.stop_awaiting_lock_frame(output);
                        }
                        // The panel may still be lit on whatever it last scanned
                        // out, which on the backstop path is precisely the frame
                        // that was never a lock frame — confirming here would put
                        // the desktop behind a `locked` event.
                        Err(e) => {
                            tracing::error!(
                                "DPMS off: compositor.clear failed for '{}': {e:?} — leaving it \
                                 lit and awaiting a lock frame",
                                output.name()
                            );
                            // Recording an output as off when its panel is still lit
                            // would freeze it out of the render gate forever. Put the
                            // bookkeeping back on reality so the output keeps
                            // rendering; the `refresh` below then tells the client
                            // the output is still on. Only a lock retries from here —
                            // its backstop asks again next pass (and skips outputs
                            // already marked off, which is the other reason to undo
                            // the mark). An ordinary `wlopm --off` gets this error
                            // and nothing more.
                            data.dpms_off_outputs.remove(output);
                            data.redraws_needed.insert(output.clone());
                        }
                    }
                }
                break;
            }
            if !handled {
                // No surface left to apply the transition to, and no later pass
                // will find one — the request is simply lost. A lock is
                // unaffected: an output goes surface-less by being removed, and
                // removal already drops it from the awaited set.
                tracing::warn!(
                    "DPMS {}: no DRM surface for '{}', dropping the transition",
                    if *on { "on" } else { "off" },
                    output.name()
                );
            }
        }
        // Broadcast mode events for client-initiated changes (already sent
        // inline) plus anything else that drifted; idempotent.
        driftwm::protocols::output_power::OutputPowerState::refresh(data);
    }

    // 3. Mark outputs dirty for per-output animations / pending chunk uploads,
    //    across every active device.
    for device in &devices {
        let dev = device.0.borrow();
        if !dev.drm.is_active() {
            continue;
        }
        for surface in dev.surfaces.values() {
            if data.dpms_off_outputs.contains(&surface.output) {
                continue;
            }
            if data.output_has_active_animations(&surface.output) {
                data.redraws_needed.insert(surface.output.clone());
            }
            // Chunked-bg with tiles still to upload: keep firing frames until the
            // visible set fully resolves. Otherwise the loop idles after pan
            // stops and blurry chunks stay covered by the fallback plane until
            // unrelated damage (cursor, animation, client commit) wakes us.
            if let Some(cache) = data.render.cached_tile_chunks.get(&surface.output.name())
                && cache.has_pending_loads()
            {
                data.redraws_needed.insert(surface.output.clone());
            }
            // Same for chunked shader-bake: refine sharp chunks after pan stops.
            if let Some(cache) = data.render.cached_shader_chunks.get(&surface.output.name())
                && cache.has_pending_bakes()
            {
                data.redraws_needed.insert(surface.output.clone());
            }
        }
    }

    // Global animations (key repeat, cursor) → every output.
    if data.held_action.is_some()
        || data.cursor.exec_cursor_show_at.is_some()
        || data.cursor.exec_cursor_deadline.is_some()
        || data.cursor_is_animated()
    {
        data.mark_all_dirty();
    } else if data.render.background_is_animated {
        // An output whose fullscreen window conceals the canvas skips the
        // background entirely, so an animated bg gives it nothing to redraw —
        // marking it just burns battery. A translucent fullscreen window
        // conceals nothing, so its output keeps ticking.
        let dirty: Vec<_> = data
            .background_render_eligible_outputs()
            .filter(|o| data.background_animation_due(&o.name()))
            .cloned()
            .collect();
        data.redraws_needed.extend(dirty);
    }

    // 4. Foreign toplevel refresh (once per frame, not per-output)
    crate::render::refresh_foreign_toplevels(data);
    crate::render::refresh_ext_workspaces(data);

    // 4a. Drain queued mode changes before re-notifying clients so the
    // re-broadcast reflects the new mode state. Mode changes come from
    // wlr-output-management Apply or config reload. Each device claims the
    // entries for outputs it owns and hands the rest back; anything left
    // after every device had a look targets a gone output.
    if !data.pending_mode_changes.is_empty() {
        let mut pending = std::mem::take(&mut data.pending_mode_changes);
        for device in &devices {
            if pending.is_empty() {
                break;
            }
            let mut dev = device.0.borrow_mut();
            let DeviceData { drm, surfaces, .. } = &mut *dev;
            pending = apply_pending_mode_changes(drm, surfaces, data, pending);
        }
        for (name, _) in pending {
            tracing::warn!("Mode change for '{name}' dropped: output no longer present");
        }
    }

    // 4b. Re-notify output management clients after apply_output_config,
    //     aggregating heads across every device.
    if data.output_config_dirty {
        data.output_config_dirty = false;
        let head_state = collect_all_head_states(data);
        driftwm::protocols::output_management::notify_changes::<DriftWm>(
            &mut data.output_management_state,
            head_state,
        );
    }

    // 5. Render outputs that need it, per device.
    for device in &devices {
        let mut dev = device.0.borrow_mut();
        if !dev.drm.is_active() {
            continue;
        }
        let kms_node = dev.kms_node;
        let Some(render_node) = dev.gfx.as_ref().map(|g| g.render_node) else {
            continue;
        };
        for (&crtc, surface) in dev.surfaces.iter_mut() {
            let key: CrtcKey = (kms_node, crtc);
            if data.dpms_off_outputs.contains(&surface.output) {
                data.redraws_needed.remove(&surface.output);
                continue;
            }
            // An armed estimated-VBlank timer counts as waiting, like frames_pending:
            // re-rendering before either resolves spins render_frame past refresh rate.
            if data.redraws_needed.contains(&surface.output)
                && !data.frames_pending.contains(&key)
                && !data.estimated_vblank_timers.contains_key(&key)
            {
                render_frame(data, surface, key, render_node);
            }
        }
    }
}

pub fn init_udev(
    event_loop: &mut EventLoop<'static, DriftWm>,
    data: &mut DriftWm,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create libseat session
    let (mut session, session_notifier) = LibSeatSession::new()
        .map_err(|e| format!("Failed to create session (are you running from a TTY?): {e}"))?;
    let seat_name = session.seat();
    tracing::info!("Session created on seat: {seat_name}");
    tracing::info!(
        "Backend config: wait_for_frame_completion={}, disable_direct_scanout={}, disable_hardware_cursor={}",
        data.config.backend.wait_for_frame_completion,
        data.config.backend.disable_direct_scanout,
        data.config.backend.disable_hardware_cursor,
    );

    // 2. Enumerate GPUs — UdevBackend gives us all DRM devices (also used for hotplug later)
    let udev_backend = UdevBackend::new(&seat_name)?;
    let primary_gpu_path = udev::primary_gpu(&seat_name).ok().flatten();
    if let Some(ref p) = primary_gpu_path {
        tracing::info!("System primary GPU: {}", p.display());
    }

    // Build ordered candidate list: primary GPU first, then all others.
    // On hybrid graphics (iGPU + dGPU), the "primary" GPU may not have
    // the display outputs, so we fall back to other devices.
    let gpu_paths: Vec<PathBuf> = {
        let mut paths = Vec::new();
        if let Some(ref p) = primary_gpu_path {
            paths.push(p.clone());
        }
        for (_dev_id, path) in udev_backend.device_list() {
            let p = path.to_path_buf();
            if !paths.contains(&p) {
                paths.push(p);
            }
        }
        paths
    };
    tracing::info!("GPU candidates: {gpu_paths:?}");

    if gpu_paths.is_empty() {
        return Err("No GPUs found".into());
    }

    // 3. Open every usable GPU. The first that opens becomes the primary
    // render GPU (candidate order puts the system primary first); the others'
    // outputs scan out primary-rendered buffers via an implicit PRIME copy.

    // Multi-GPU manager: one GLES renderer per render node, created lazily as
    // nodes are added. The primary node's renderer is used for same-GPU work;
    // outputs on other GPUs go through a cross-GPU MultiRenderer (PRIME copy).
    let mut gpu_manager: GpuManager<GbmGlesBackend<GlesRenderer, DrmDeviceFd>> =
        GpuManager::new(GbmGlesBackend::with_context_priority(ContextPriority::High))
            .map_err(|e| format!("Failed to create GPU manager: {e}"))?;

    // The configured render GPU is tried first; the first GPU whose graphics
    // stack initialises becomes the primary. With `gpus = "render"` nothing
    // else is opened, so other GPUs are never woken. Otherwise the remaining
    // GPUs keep only their DRM fd open until an output needs them: an EGL
    // context on an NVIDIA dGPU keeps it out of runtime suspend.
    let render_only = data.config.backend.gpus == GpuScope::RenderOnly;
    let (gpu_paths, unmatched) = super::gpu_select::order_candidates(
        gpu_paths,
        &data.config.backend.render_gpu,
        super::gpu_select::classify,
    );
    if let Some(msg) = unmatched {
        tracing::warn!("{msg}");
    }
    let mut primary: Option<OpenedGpu> = None;
    let mut secondary: Vec<OpenedGpu> = Vec::new();
    for path in &gpu_paths {
        if primary.is_some() && render_only {
            break;
        }
        let Some(mut gpu) = open_drm(&mut session, path) else {
            continue;
        };
        if primary.is_none() {
            match init_gfx(&mut gpu_manager, gpu.node, &gpu.drm_fd) {
                Some(gfx) => {
                    tracing::info!(
                        "Primary render GPU: {} (render node {})",
                        gpu.node,
                        gfx.render_node
                    );
                    gpu.gfx = Some(gfx);
                    primary = Some(gpu);
                    continue;
                }
                None if render_only => continue,
                None => {}
            }
        }
        tracing::info!("Secondary GPU: {}", path.display());
        secondary.push(gpu);
    }
    let Some(primary) = primary else {
        return Err("No usable GPU found (are you running from a TTY?)".into());
    };
    let mut opened = vec![primary];
    opened.extend(secondary);
    let primary_gfx = opened[0].gfx.as_ref().unwrap();
    let primary_render_node = primary_gfx.render_node;
    let primary_render_formats = primary_gfx.render_formats.clone();

    // 4. Store renderer on state + create DMA-BUF global
    let (gfx_probe_sender, gfx_probe_channel) = channel::<GfxProbeResult>();
    data.backend = Some(Backend::Udev(Box::new(UdevRenderer {
        gpu_manager,
        primary_render_node,
        secondary_render_nodes: HashSet::new(),
        gfx_probes_in_flight: HashSet::new(),
        gfx_probe_sender,
    })));
    let formats = data
        .backend
        .as_mut()
        .unwrap()
        .with_renderer(|r| r.dmabuf_formats())
        .expect("primary renderer available at init");
    data.render_device = Some(primary_render_node.dev_id());
    // Capture clients allocate buffers we render INTO, so advertise the
    // render-target set (already CCS-filtered above) — not the wider
    // import set, which can include formats we can't bind as a target.
    data.render_dmabuf_formats = Some(primary_render_formats.iter().copied().collect());
    let default_feedback = DmabufFeedbackBuilder::new(primary_render_node.dev_id(), formats)
        .build()
        .expect("failed to build dmabuf feedback");
    let dmabuf_global = data
        .dmabuf_state
        .create_global_with_default_feedback::<DriftWm>(&data.display_handle, &default_feedback);
    data.dmabuf_global = Some(dmabuf_global);

    // Compile background/effect shaders on the primary renderer before any
    // frame renders (attach_gpu below queues the first frames).
    {
        let mut backend = data.backend.take().unwrap();
        backend
            .with_renderer(|r| {
                data.render.shadow_shader = crate::render::compile_shadow_shader(r);
                data.render.border_shader = crate::render::compile_border_shader(r);
                data.render.corner_clip_shader = crate::render::compile_corner_clip_shader(r);
                let (blur_down, blur_up, blur_mask) = crate::render::compile_blur_shaders(r);
                data.render.blur_down_shader = blur_down;
                data.render.blur_up_shader = blur_up;
                data.render.blur_mask_shader = blur_mask;
                // The blur's wrap mode is a property of the GL context, and this is a
                // new one.
                data.render.blur_wrap_mode = None;
            })
            .expect("primary renderer available at init");
        data.backend = Some(backend);
    }

    // 5. Set up libinput
    let libinput_session = LibinputSessionInterface::from(session.clone());
    let mut libinput = Libinput::new_with_udev(libinput_session);
    libinput
        .udev_assign_seat(&seat_name)
        .map_err(|_| "Failed to assign libinput seat")?;
    let libinput_backend = LibinputInputBackend::new(libinput.clone());

    event_loop
        .handle()
        .insert_source(libinput_backend, |mut event, _, data| {
            use smithay::backend::input::InputEvent;
            match &mut event {
                InputEvent::DeviceAdded { device } => {
                    data.configure_libinput_device(device);
                    data.input_devices.push(device.clone());
                }
                InputEvent::DeviceRemoved { device } => {
                    data.input_devices.retain(|d| d != device);
                }
                _ => {}
            }
            data.process_input_event(event);
        })?;

    // Store session on state so keyboard handler can call change_vt()
    data.session = Some(session);

    // 6. Register session notifier (VT switching). Pauses/resumes every DRM
    // device; libinput is seat-wide, so the one handle moves into the closure.
    event_loop
        .handle()
        .insert_source(session_notifier, move |event, _, data: &mut DriftWm| {
            match event {
                SessionEvent::PauseSession => {
                    tracing::info!("Session paused (VT switch away)");
                    libinput.suspend();
                    for device in data.udev_devices.values() {
                        device.0.borrow_mut().drm.pause();
                    }
                    for (_, token) in data.estimated_vblank_timers.drain() {
                        data.loop_handle.remove(token);
                    }
                    data.clear_lock_frames();
                    data.confirm_lock_on_session_pause();
                    // The only reset a switch we didn't initiate ourselves
                    // (`chvt`, logind) ever reaches — the keyboard handler's two
                    // copies both hang off a key we intercepted.
                    data.reset_held_input_state();
                }
                SessionEvent::ActivateSession => {
                    tracing::info!("Session resumed (VT switch back)");
                    if libinput.resume().is_err() {
                        tracing::warn!("Failed to resume libinput");
                    }
                    // VBlanks for pre-switch frames never arrive, so nothing
                    // would ever retire their provenance either.
                    data.frames_pending.clear();
                    // Whatever wedged the GPU has had a suspend/resume or a VT
                    // round-trip to clear, and the first frame back is the
                    // slowest of the session — the tier that survived the switch
                    // would be the one most likely to cut it short.
                    data.fence_failures.clear();
                    data.clear_lock_frames();
                    for (_, token) in data.estimated_vblank_timers.drain() {
                        data.loop_handle.remove(token);
                    }
                    // VT switch implicitly wakes the screen. Clear DPMS-off so
                    // the render loop below actually paints; the daemon will
                    // re-request off after idle if still applicable.
                    data.dpms_off_outputs.clear();
                    data.pending_dpms.clear();
                    driftwm::protocols::output_power::OutputPowerState::refresh(data);
                    let devices: Vec<UdevDevice> = data.udev_devices.values().cloned().collect();
                    for device in &devices {
                        let mut dev = device.0.borrow_mut();
                        if let Err(e) = dev.drm.activate(false) {
                            tracing::error!("Failed to activate DRM: {e}");
                            continue;
                        }
                        let kms_node = dev.kms_node;
                        let render_node = dev.gfx.as_ref().map(|g| g.render_node);
                        let DeviceData { drm, surfaces, .. } = &mut *dev;
                        for (&crtc, surface) in surfaces.iter_mut() {
                            if let Err(e) = surface.compositor.reset_state() {
                                tracing::warn!("Failed to reset DRM surface state: {e}");
                            }
                            let _ = surface.compositor.frame_submitted();
                            if let Some(ramp) = surface.pending_gamma_change.take() {
                                if apply_gamma(surface, drm, crtc, ramp.as_deref()).is_none() {
                                    tracing::warn!(
                                        "failed to re-apply gamma on session resume for crtc {crtc:?}"
                                    );
                                }
                            } else if let Some(gp) = &mut surface.gamma_props
                                && gp.has_previous_blob()
                            {
                                // VT switch clears CRTC gamma to default. Re-apply
                                // the last-set blob so a tint set before the switch
                                // doesn't silently vanish until the client re-polls.
                                // Legacy path has no equivalent — kernel doesn't
                                // retain the ramp and we don't shadow it.
                                if gp.restore_gamma(drm).is_none() {
                                    tracing::warn!(
                                        "failed to restore gamma on session resume for crtc {crtc:?}"
                                    );
                                }
                            }
                            if let Some(render_node) = render_node {
                                render_frame(data, surface, (kms_node, crtc), render_node);
                            }
                        }
                    }
                }
            }
        })?;

    // 6b. Register the secondary-GPU bring-up channel: a probe thread
    // spawned from `scan_device_connectors` sends its result here, off the
    // single event-loop thread it must never block.
    event_loop
        .handle()
        .insert_source(gfx_probe_channel, move |event, _, data: &mut DriftWm| {
            let ChannelEvent::Msg(GfxProbeResult { node, gfx }) = event else {
                return;
            };
            let Some(Backend::Udev(udev)) = data.backend.as_mut() else {
                return;
            };
            udev.gfx_probes_in_flight.remove(&node);
            let Some(device) = data.udev_devices.get(&node).cloned() else {
                tracing::debug!("gfx probe finished for a now-removed device {node}");
                return;
            };
            if let Some(probed) = gfx {
                let Some(Backend::Udev(udev)) = data.backend.as_mut() else {
                    return;
                };
                if let Some(ready) = register_gfx(&mut udev.gpu_manager, node, probed) {
                    if ready.render_node != udev.primary_render_node {
                        udev.secondary_render_nodes.insert(ready.render_node);
                    }
                    device.0.borrow_mut().gfx = Some(ready);
                }
            }
            // Picks the now-ready (or still-failed, logged already) gfx back
            // up immediately instead of waiting for the next retry tick.
            scan_device_connectors(data, &device);
        })?;

    // 7. Register udev backend for hotplug (connectors AND whole GPUs)
    let udev_dispatcher = Dispatcher::new(
        udev_backend,
        move |event: UdevEvent, _, data: &mut DriftWm| match event {
            UdevEvent::Changed { device_id } => {
                let Ok(node) = DrmNode::from_dev_id(device_id) else {
                    return;
                };
                let Some(device) = data.udev_devices.get(&node).cloned() else {
                    tracing::debug!("udev change for unknown device {device_id:?}");
                    return;
                };
                tracing::debug!("Udev device changed: {device_id:?}");
                scan_device_connectors(data, &device);
            }
            UdevEvent::Added { device_id, path } => {
                let Ok(node) = DrmNode::from_dev_id(device_id) else {
                    return;
                };
                if data.udev_devices.contains_key(&node) {
                    return;
                }
                tracing::info!("Udev device added: {}", path.display());
                gpu_added(data, &path);
            }
            UdevEvent::Removed { device_id } => {
                let Ok(node) = DrmNode::from_dev_id(device_id) else {
                    return;
                };
                gpu_removed(data, node);
            }
        },
    );
    event_loop.handle().register_dispatcher(udev_dispatcher)?;

    // 8. Attach every opened GPU: registers its VBlank source, scans its
    // connectors, creates outputs and queues their first frames.
    for gpu in opened {
        attach_gpu(data, gpu);
    }

    let total_surfaces: usize = data
        .udev_devices
        .values()
        .map(|d| d.0.borrow().surfaces.len())
        .sum();
    if total_surfaces == 0 {
        return Err("No GPU with connected displays found (are you running from a TTY?)".into());
    }

    Ok(())
}

/// A DRM device opened but not yet attached to the event loop or scanned for
/// outputs. `gfx` is `None` for GPUs that initialise lazily.
struct OpenedGpu {
    node: DrmNode,
    drm: DrmDevice,
    drm_notifier: DrmDeviceNotifier,
    drm_fd: DrmDeviceFd,
    gfx: Option<DeviceGfx>,
}

/// Open one DRM device (no EGL/GBM yet). Returns `None` (with a log line) on
/// any failure so callers can skip to the next candidate.
fn open_drm(session: &mut LibSeatSession, path: &Path) -> Option<OpenedGpu> {
    let open_flags = OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOCTTY | OFlags::NONBLOCK;

    let node = match DrmNode::from_path(path) {
        Ok(n) => n,
        Err(e) => {
            tracing::debug!("{}: not a DRM node ({e}), skipping", path.display());
            return None;
        }
    };
    if node.ty() != NodeType::Primary {
        tracing::debug!("{}: not a primary node, skipping", path.display());
        return None;
    }

    let fd = match session.open(path, open_flags) {
        Ok(fd) => fd,
        Err(e) => {
            tracing::warn!("{}: failed to open ({e})", path.display());
            return None;
        }
    };
    let drm_fd = DrmDeviceFd::new(DeviceFd::from(fd));

    // true = release existing CRTCs for a clean modeset (avoids conflicts
    // with previous session's DRM state)
    let (drm, drm_notifier) = match DrmDevice::new(drm_fd.clone(), true) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!("{}: failed to create DRM device ({e})", path.display());
            return None;
        }
    };

    Some(OpenedGpu {
        node,
        drm,
        drm_notifier,
        drm_fd,
        gfx: None,
    })
}

/// Create the GBM device, probe EGL and register the render node with the GPU
/// manager. Returns `None` (with a log line) on failure.
fn init_gfx(
    gpu_manager: &mut GpuManager<GbmGlesBackend<GlesRenderer, DrmDeviceFd>>,
    node: DrmNode,
    drm_fd: &DrmDeviceFd,
) -> Option<DeviceGfx> {
    let gbm = probe_gfx(node, drm_fd)?;
    register_gfx(gpu_manager, node, gbm)
}

/// The slow half of GPU bring-up: open GBM, create an EGL display/context and
/// resolve the render node and formats. Touches only the DRM fd and freshly
/// created EGL/GBM objects — no shared compositor state — so it's safe to run
/// on a background thread. For a dGPU waking from PCI runtime suspend this is
/// where the time goes (power-up, and on NVIDIA a GSP firmware reload), which
/// is why `scan_device_connectors` runs it off the main thread instead of
/// blocking rendering/input on every other output while it waits.
fn probe_gfx(node: DrmNode, drm_fd: &DrmDeviceFd) -> Option<DeviceGfx> {
    let gbm = match GbmDevice::new(drm_fd.clone()) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!("{node}: failed to create GBM device ({e})");
            return None;
        }
    };

    let egl_display = match unsafe { EGLDisplay::new(gbm.clone()) } {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("{node}: failed to create EGL display ({e})");
            return None;
        }
    };
    if EGLDevice::device_for_display(&egl_display).is_ok_and(|d| d.is_software()) {
        tracing::warn!("{node}: software EGL device, skipping");
        return None;
    }
    // High priority lets the compositor's composite preempt a
    // GPU-saturating client (shader compile, screen-share encode) instead
    // of queuing behind it. EGL_IMG_context_priority is best-effort:
    // smithay falls back to default priority if the extension is absent, and
    // some drivers (notably NVIDIA) may only partially honor it.
    let egl_context = match EGLContext::new_with_priority(&egl_display, ContextPriority::High) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("{}: failed to create EGL context ({e})", node);
            return None;
        }
    };
    let render_formats: Vec<Format> = egl_context
        .dmabuf_render_formats()
        .iter()
        .copied()
        .filter(|f| {
            // Intel CCS modifiers increase display link bandwidth, which can
            // prevent high-res/high-refresh modes from working (e.g. ultrawides
            // that need DSC). Filter them out — the GPU falls back to
            // uncompressed framebuffers with no visual difference.
            let is_ccs = matches!(
                f.modifier,
                Modifier::I915_y_tiled_ccs
                    | Modifier::I915_y_tiled_gen12_rc_ccs
                    | Modifier::I915_y_tiled_gen12_mc_ccs
                    // Yf_TILED_CCS
                    | Modifier::Unrecognized(0x100000000000005)
                    // Y_TILED_GEN12_RC_CCS_CC
                    | Modifier::Unrecognized(0x100000000000008)
                    // 4_TILED_DG2_RC_CCS
                    | Modifier::Unrecognized(0x10000000000000a)
                    // 4_TILED_DG2_MC_CCS
                    | Modifier::Unrecognized(0x10000000000000b)
                    // 4_TILED_DG2_RC_CCS_CC
                    | Modifier::Unrecognized(0x10000000000000c)
            );
            !is_ccs
        })
        .collect();
    // Ask EGL/Mesa for the actual rendering device — on split-DRM
    // systems the KMS node we opened has no render node, but Mesa
    // routes rendering through the right GPU under the hood. We need
    // to advertise that GPU's render node to clients (`zwp_linux_dmabuf_v1`
    // feedback, xdph-wlr) so they don't crash trying to use the
    // display-only node.
    let render_node = EGLDevice::device_for_display(&egl_display)
        .ok()
        .and_then(|d| d.try_get_render_node().ok().flatten())
        .or_else(|| node.node_with_type(NodeType::Render).and_then(|n| n.ok()))
        .unwrap_or_else(|| {
            tracing::warn!(
                "could not resolve a DRM render node; falling back to KMS node {node:?} \
                 — capture clients may misbehave"
            );
            node
        });

    // Drop the probe context before handing the GBM device to the GPU
    // manager, which creates its own EGL display + GLES renderer for the
    // render node (avoids two high-priority contexts on the same device).
    drop(egl_context);

    Some(DeviceGfx {
        gbm,
        render_node,
        render_formats,
    })
}

/// Register an already-probed GBM device's render node with the GPU manager
/// (its own EGL display + GLES renderer). Cheap once the GPU is actually
/// awake — the slow part already happened in `probe_gfx` — so this stays on
/// the main thread. `add_node` is a no-op if another KMS device already
/// registered this render node (split-DRM boards routing through one render
/// GPU). Returns `None` (with a log line) on failure.
fn register_gfx(
    gpu_manager: &mut GpuManager<GbmGlesBackend<GlesRenderer, DrmDeviceFd>>,
    node: DrmNode,
    gfx: DeviceGfx,
) -> Option<DeviceGfx> {
    if let Err(e) = gpu_manager
        .as_mut()
        .add_node(gfx.render_node, gfx.gbm.clone())
    {
        tracing::warn!("{node}: failed to add render node to GPU manager ({e})");
        return None;
    }
    // add_node only records the node; the GLES renderer is built on first use
    // and a failure there is otherwise silent. Build it now so a broken
    // secondary GPU is rejected here instead of retried on every frame.
    if let Err(e) = gpu_manager.single_renderer(&gfx.render_node) {
        tracing::warn!(
            "{node}: GLES renderer unavailable on {} ({e:?})",
            gfx.render_node
        );
        gpu_manager.as_mut().remove_node(&gfx.render_node);
        let _ = gpu_manager.devices();
        return None;
    }
    Some(gfx)
}

/// Wire an opened GPU into the compositor: register its VBlank event source,
/// store it in `udev_devices`, and scan its connectors (creating outputs and
/// queuing their first frames).
fn attach_gpu(data: &mut DriftWm, gpu: OpenedGpu) {
    let OpenedGpu {
        node,
        drm,
        drm_notifier,
        drm_fd,
        gfx,
    } = gpu;

    log_drm_connectors(&drm);

    let device = UdevDevice(Rc::new(RefCell::new(DeviceData {
        kms_node: node,
        drm,
        drm_fd,
        gfx,
        drm_scanner: DrmScanner::new(),
        scan_retries: 0,
        surfaces: HashMap::new(),
        drm_token: None,
    })));

    let device_for_drm = device.clone();
    let token =
        data.loop_handle
            .insert_source(drm_notifier, move |event, meta, data: &mut DriftWm| {
                let mut dev = device_for_drm.0.borrow_mut();
                let kms_node = dev.kms_node;
                let Some(render_node) = dev.gfx.as_ref().map(|g| g.render_node) else {
                    return;
                };
                match event {
                    DrmEvent::VBlank(crtc) => {
                        let key: CrtcKey = (kms_node, crtc);
                        let Some(surface) = dev.surfaces.get_mut(&crtc) else {
                            return;
                        };
                        match surface.compositor.frame_submitted() {
                            Ok(Some(mut feedback)) => {
                                deliver_presentation(&mut feedback, &surface.output, meta.as_ref());
                            }
                            Ok(None) => {}
                            Err(e) => tracing::warn!("frame_submitted error: {e:?}"),
                        }
                        data.frames_pending.remove(&key);
                        // The VBlank event itself is the proof the frame flipped —
                        // `frame_submitted` returning `Ok(None)` is a real flip that
                        // simply carries no feedback.
                        if data.lock_frame_queued.remove(&key) {
                            data.lock_frame_on_screen.insert(key);
                            data.stop_awaiting_lock_frame(&surface.output);
                        } else {
                            data.lock_frame_on_screen.remove(&key);
                        }
                        // Real VBlank beat any estimated-VBlank timer we might have armed.
                        if let Some(token) = data.estimated_vblank_timers.remove(&key) {
                            data.loop_handle.remove(token);
                        }
                        if data.redraws_needed.contains(&surface.output) {
                            render_frame(data, surface, (kms_node, crtc), render_node);
                        }
                    }
                    DrmEvent::Error(err) => {
                        tracing::error!("DRM error: {err}");
                    }
                }
            });
    match token {
        Ok(token) => device.0.borrow_mut().drm_token = Some(token),
        Err(e) => tracing::error!("Failed to register DRM event source for {node}: {e}"),
    }

    data.udev_devices.insert(node, device.clone());
    scan_device_connectors(data, &device);
}

/// Rescan a device's connectors, creating surfaces for newly connected ones
/// and tearing down disconnected ones. Shared between initial attach and
/// udev change events.
fn scan_device_connectors(data: &mut DriftWm, device: &UdevDevice) {
    {
        let mut dev = device.0.borrow_mut();
        let DeviceData {
            kms_node,
            ref mut drm_scanner,
            ref mut scan_retries,
            ref mut drm,
            ref drm_fd,
            ref mut gfx,
            ref mut surfaces,
            ..
        } = *dev;
        let scan_result = match drm_scanner.scan_connectors(&*drm) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Failed to scan connectors: {e}");
                return;
            }
        };
        let mut retry = false;
        for scan_event in scan_result {
            match scan_event {
                DrmScanEvent::Connected {
                    connector,
                    crtc: Some(crtc),
                } => {
                    if surfaces.contains_key(&crtc) {
                        continue;
                    }
                    tracing::info!(
                        "Connector connected: {}-{} (CRTC {:?})",
                        connector.interface().as_str(),
                        connector.interface_id(),
                        crtc,
                    );
                    if gfx.is_none() {
                        let Some(Backend::Udev(udev)) = data.backend.as_mut() else {
                            continue;
                        };
                        if !udev.gfx_probes_in_flight.contains(&kms_node) {
                            udev.gfx_probes_in_flight.insert(kms_node);
                            let sender = udev.gfx_probe_sender.clone();
                            let probe_drm_fd = drm_fd.clone();
                            std::thread::spawn(move || {
                                let gfx = probe_gfx(kms_node, &probe_drm_fd);
                                let _ = sender.send(GfxProbeResult {
                                    node: kms_node,
                                    gfx,
                                });
                            });
                        }
                        // Bring-up (GBM/EGL, and on a dGPU waking from runtime
                        // suspend possibly a firmware reload) runs on that
                        // worker thread so it never blocks the single event
                        // loop thread that also draws every other output and
                        // reads input. The channel handler re-scans as soon
                        // as the probe finishes; the retry timer below is
                        // just a safety net.
                        retry = true;
                        continue;
                    }
                    let Some(DeviceGfx {
                        gbm,
                        render_node,
                        render_formats,
                    }) = gfx.as_ref()
                    else {
                        tracing::warn!("{kms_node}: graphics init failed for this connector");
                        retry = true;
                        continue;
                    };
                    let render_node = *render_node;
                    // Placeholders are retired inside output_connected, after
                    // create_surface — the sequence is synchronous within this
                    // handler, so active_output() never observes a gap.
                    let saved = data.saved_camera_state();
                    let dh = data.display_handle.clone();
                    if let Some(sd) = create_surface(
                        drm,
                        gbm,
                        render_formats,
                        render_node,
                        kms_node,
                        &connector,
                        crtc,
                        &dh,
                        data,
                    ) {
                        surfaces.insert(crtc, sd);
                        let new_output = surfaces[&crtc].output.clone();
                        data.output_connected(&new_output, &saved);
                        data.active_outputs.insert(new_output);
                        let surface = surfaces.get_mut(&crtc).unwrap();
                        render_frame(data, surface, (kms_node, crtc), render_node);
                        *scan_retries = 0;
                    } else {
                        retry = true;
                    }
                }
                DrmScanEvent::Connected {
                    connector,
                    crtc: None,
                } => {
                    tracing::warn!(
                        "Connector {}-{} has no available CRTC",
                        connector.interface().as_str(),
                        connector.interface_id()
                    );
                }
                DrmScanEvent::Disconnected {
                    crtc: Some(crtc), ..
                } => {
                    tracing::info!("Hotplug: CRTC {crtc:?} disconnected");
                    if let Some(surface) = surfaces.remove(&crtc) {
                        // "Last output" spans all devices: another GPU's
                        // monitor keeps the canvas alive.
                        let is_last = surfaces.is_empty()
                            && data
                                .udev_devices
                                .values()
                                .filter(|d| !Rc::ptr_eq(&d.0, &device.0))
                                .all(|d| d.0.borrow().surfaces.is_empty());
                        teardown_output(data, surface, is_last);
                    }
                    let key: CrtcKey = (kms_node, crtc);
                    data.frames_pending.remove(&key);
                    data.fence_failures.remove(&key);
                    data.forget_lock_frame(key);
                    if let Some(token) = data.estimated_vblank_timers.remove(&key) {
                        data.loop_handle.remove(token);
                    }
                }
                _ => {}
            }
        }
        if surfaces.is_empty()
            && !data.config.backend.keep_secondary_gpu_awake
            && let Some(released) = gfx.as_ref().map(|g| g.render_node)
            && let Some(Backend::Udev(udev)) = data.backend.as_mut()
            && released != udev.primary_render_node
        {
            let shared = data
                .udev_devices
                .values()
                .filter(|d| !Rc::ptr_eq(&d.0, &device.0))
                .any(|d| {
                    d.0.borrow()
                        .gfx
                        .as_ref()
                        .is_some_and(|g| g.render_node == released)
                });
            if !shared {
                // Last output gone: drop the renderer so the dGPU can idle.
                tracing::info!("{kms_node}: no outputs left, releasing render node {released}");
                udev.gpu_manager.as_mut().remove_node(&released);
                let _ = udev.gpu_manager.devices();
                udev.secondary_render_nodes.remove(&released);
            }
            *gfx = None;
        }
        // The scanner already recorded a failed connector as connected and
        // would never report it again (e.g. a dGPU still waking from runtime
        // suspend): forget its state and rescan shortly, a few times.
        if retry && *scan_retries < 5 {
            *scan_retries += 1;
            *drm_scanner = DrmScanner::new();
            let device = device.clone();
            let timer = Timer::from_duration(Duration::from_secs(3));
            let _ = data
                .loop_handle
                .insert_source(timer, move |_, _, data: &mut DriftWm| {
                    let alive = data
                        .udev_devices
                        .values()
                        .any(|d| Rc::ptr_eq(&d.0, &device.0));
                    if alive {
                        scan_device_connectors(data, &device);
                    }
                    TimeoutAction::Drop
                });
        }
    }
    // Notify output management clients after connector changes; heads span
    // every device, so aggregate across all of them.
    let head_state = collect_all_head_states(data);
    driftwm::protocols::output_management::notify_changes::<DriftWm>(
        &mut data.output_management_state,
        head_state,
    );
}

/// A whole GPU appeared at runtime (eGPU dock, driver rebind): open it and
/// light up its outputs (graphics init happens on the first connected output).
fn gpu_added(data: &mut DriftWm, path: &Path) {
    if !matches!(data.backend, Some(Backend::Udev(_)))
        || data.config.backend.gpus == GpuScope::RenderOnly
    {
        return;
    }
    let Some(session) = data.session.as_mut() else {
        return;
    };
    let Some(gpu) = open_drm(session, path) else {
        return;
    };
    attach_gpu(data, gpu);
}

/// A whole GPU disappeared: tear down its outputs, drop its event source and
/// (unless shared or primary) its renderer.
fn gpu_removed(data: &mut DriftWm, node: DrmNode) {
    let Some(device) = data.udev_devices.remove(&node) else {
        tracing::debug!("udev removal for unknown device {node}");
        return;
    };
    tracing::info!("GPU removed: {node}");

    let (surfaces, token, render_node) = {
        let mut dev = device.0.borrow_mut();
        let surfaces: Vec<(crtc::Handle, SurfaceData)> = dev.surfaces.drain().collect();
        let render_node = dev.gfx.as_ref().map(|g| g.render_node);
        (surfaces, dev.drm_token.take(), render_node)
    };
    if let Some(token) = token {
        data.loop_handle.remove(token);
    }

    let surviving: usize = data
        .udev_devices
        .values()
        .map(|d| d.0.borrow().surfaces.len())
        .sum();
    let count = surfaces.len();
    for (i, (crtc, surface)) in surfaces.into_iter().enumerate() {
        let key: CrtcKey = (node, crtc);
        data.frames_pending.remove(&key);
        data.fence_failures.remove(&key);
        data.forget_lock_frame(key);
        if let Some(t) = data.estimated_vblank_timers.remove(&key) {
            data.loop_handle.remove(t);
        }
        teardown_output(data, surface, surviving == 0 && i + 1 == count);
    }

    // On split-DRM boards several KMS devices can route to one render node;
    // only release the render node with its last KMS device.
    let render_node_still_used = render_node.is_none_or(|rn| {
        data.udev_devices.values().any(|d| {
            d.0.borrow()
                .gfx
                .as_ref()
                .is_some_and(|g| g.render_node == rn)
        })
    });
    if let Some(Backend::Udev(udev)) = data.backend.as_mut()
        && let Some(render_node) = render_node
        && !render_node_still_used
    {
        if render_node == udev.primary_render_node {
            // We can't render without the primary GPU (no promotion of a new
            // primary — surviving outputs go dark until it returns or restart).
            // Withdraw what advertised the dead node so new clients don't
            // allocate on it: the dmabuf global (delayed destroy, like output
            // globals) and every surviving surface's dmabuf feedback.
            tracing::error!(
                "Primary render GPU {render_node} removed — rendering is unavailable \
                 until restart"
            );
            if let Some(global) = data.dmabuf_global.take() {
                data.dmabuf_state
                    .disable_global::<DriftWm>(&data.display_handle, &global);
                let timer = Timer::from_duration(Duration::from_secs(10));
                let _ = data
                    .loop_handle
                    .insert_source(timer, move |_, _, data: &mut DriftWm| {
                        let dh = data.display_handle.clone();
                        data.dmabuf_state.destroy_global::<DriftWm>(&dh, global);
                        TimeoutAction::Drop
                    });
            }
            data.render_device = None;
            data.render_dmabuf_formats = None;
            for device in data.udev_devices.values() {
                for surface in device.0.borrow_mut().surfaces.values_mut() {
                    surface.dmabuf_feedback = None;
                }
            }
        }
        udev.gpu_manager.as_mut().remove_node(&render_node);
        udev.secondary_render_nodes.remove(&render_node);
        // Trigger re-enumeration so the manager actually drops the device.
        let _ = udev.gpu_manager.devices();
    }

    let head_state = collect_all_head_states(data);
    driftwm::protocols::output_management::notify_changes::<DriftWm>(
        &mut data.output_management_state,
        head_state,
    );
}

/// Log all connectors and their states for the selected GPU.
fn log_drm_connectors(drm: &DrmDevice) {
    use smithay::reexports::drm::control::Device as ControlDevice;
    let Ok(res) = ControlDevice::resource_handles(drm) else {
        return;
    };
    tracing::info!(
        "DRM resources: {} connectors, {} CRTCs, {} encoders",
        res.connectors().len(),
        res.crtcs().len(),
        res.encoders().len(),
    );
    for &handle in res.connectors() {
        if let Ok(info) = ControlDevice::get_connector(drm, handle, true) {
            tracing::info!(
                "  connector {}-{}: state={:?}, modes={}",
                info.interface().as_str(),
                info.interface_id(),
                info.state(),
                info.modes().len(),
            );
        }
    }
}

/// Pick the mode with the highest resolution (w*h), then highest refresh.
fn pick_max_mode(modes: &[control::Mode]) -> Option<control::Mode> {
    modes
        .iter()
        .max_by_key(|m| {
            let (w, h) = m.size();
            (w as u64 * h as u64, m.vrefresh() as u64)
        })
        .copied()
}

/// Pick the best mode for a connector: prefer MODE_TYPE_PREFERRED,
/// fall back to the highest-resolution mode.
fn pick_preferred_mode(modes: &[control::Mode]) -> Option<control::Mode> {
    if let Some(preferred) = modes
        .iter()
        .find(|m| m.mode_type().contains(control::ModeTypeFlags::PREFERRED))
    {
        return Some(*preferred);
    }
    pick_max_mode(modes)
}

/// Where a chosen mode came from. `SynthesizedCvt` modes haven't been
/// validated by the kernel yet — callers should be prepared to retry with
/// `pick_preferred_mode` if the atomic-test fails.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModeSource {
    Edid,
    SynthesizedCvt,
}

/// Select a mode based on output config, falling back to preferred.
/// For `SizeRefresh` rules that don't match any EDID-advertised mode,
/// synthesize a CVT modeline — this lets users drive CRTs above their
/// EDID-reported refresh range.
pub(crate) fn pick_mode_for_config(
    modes: &[control::Mode],
    config: &ConfigOutputMode,
    connector_name: &str,
) -> Option<(control::Mode, ModeSource)> {
    match config {
        ConfigOutputMode::Preferred => pick_preferred_mode(modes).map(|m| (m, ModeSource::Edid)),
        ConfigOutputMode::Max => pick_max_mode(modes).map(|m| (m, ModeSource::Edid)),
        ConfigOutputMode::Size(w, h) => {
            let matched = modes
                .iter()
                .filter(|m| m.size() == (*w as u16, *h as u16))
                .max_by_key(|m| m.vrefresh() as u64);
            if matched.is_none() {
                tracing::warn!("No mode matching {w}x{h}, falling back to preferred");
            }
            matched
                .copied()
                .map(|m| (m, ModeSource::Edid))
                .or_else(|| pick_preferred_mode(modes).map(|m| (m, ModeSource::Edid)))
        }
        ConfigOutputMode::SizeRefresh(w, h, hz) => {
            if let Some(m) = modes
                .iter()
                .find(|m| m.size() == (*w as u16, *h as u16) && m.vrefresh() == *hz)
            {
                return Some((*m, ModeSource::Edid));
            }
            tracing::warn!(
                "Output {connector_name}: mode {w}x{h}@{hz}Hz not in EDID, synthesizing CVT modeline"
            );
            match cvt::synth_cvt(*w as u16, *h as u16, *hz) {
                Ok(raw) => Some((control::Mode::from(raw), ModeSource::SynthesizedCvt)),
                Err(e) => {
                    tracing::error!(
                        "Output {connector_name}: CVT synthesis failed ({e}), falling back to preferred"
                    );
                    pick_preferred_mode(modes).map(|m| (m, ModeSource::Edid))
                }
            }
        }
    }
}

/// Resolve a queued `ModeIntent` to a concrete `control::Mode` for the given
/// connector. `Custom` first looks for an exact EDID match; only synthesizes
/// CVT if nothing matches.
fn resolve_pending_mode(
    intent: &crate::state::ModeIntent,
    connector: &connector::Info,
    connector_name: &str,
) -> Option<control::Mode> {
    match intent {
        crate::state::ModeIntent::EdidIndex(idx) => connector.modes().get(*idx).copied(),
        crate::state::ModeIntent::Custom { w, h, refresh_mhz } => {
            let hz = (*refresh_mhz / 1000) as u32;
            if let Some(m) = connector
                .modes()
                .iter()
                .find(|m| m.size() == (*w as u16, *h as u16) && m.vrefresh() == hz)
            {
                return Some(*m);
            }
            match cvt::synth_cvt(*w as u16, *h as u16, hz) {
                Ok(raw) => Some(control::Mode::from(raw)),
                Err(e) => {
                    tracing::error!(
                        "Output {connector_name}: CVT synthesis failed ({e}) for {w}x{h}@{hz}Hz"
                    );
                    None
                }
            }
        }
        crate::state::ModeIntent::Preferred => pick_preferred_mode(connector.modes()),
        crate::state::ModeIntent::Max => pick_max_mode(connector.modes()),
    }
}

#[allow(clippy::too_many_arguments)]
fn create_surface(
    drm: &mut DrmDevice,
    gbm: &GbmDevice<DrmDeviceFd>,
    render_formats: &[Format],
    render_node: DrmNode,
    kms_node: DrmNode,
    connector: &connector::Info,
    crtc: crtc::Handle,
    dh: &smithay::reexports::wayland_server::DisplayHandle,
    state: &mut DriftWm,
) -> Option<SurfaceData> {
    let connector_name = format!(
        "{}-{}",
        connector.interface().as_str(),
        connector.interface_id()
    );

    let output_cfg = state.config.output_config(&connector_name);

    let config_mode = output_cfg
        .map(|c| &c.mode)
        .unwrap_or(&ConfigOutputMode::Preferred);
    let (mode, mode_source) =
        pick_mode_for_config(connector.modes(), config_mode, &connector_name)?;
    tracing::info!(
        "Output {connector_name}: mode {}x{}@{}Hz ({:?})",
        mode.size().0,
        mode.size().1,
        mode.vrefresh(),
        mode_source,
    );

    let (drm_surface, mode) = match drm.create_surface(crtc, mode, &[connector.handle()]) {
        Ok(s) => (s, mode),
        Err(e) if mode_source == ModeSource::SynthesizedCvt => {
            tracing::error!(
                "Output {connector_name}: synthesized CVT mode rejected by kernel ({e}), falling back to preferred"
            );
            let fallback = pick_preferred_mode(connector.modes())?;
            match drm.create_surface(crtc, fallback, &[connector.handle()]) {
                Ok(s) => (s, fallback),
                Err(e2) => {
                    tracing::error!("FAILED: drm.create_surface (preferred fallback): {e2}");
                    return None;
                }
            }
        }
        Err(e) => {
            tracing::error!("FAILED: drm.create_surface: {e}");
            return None;
        }
    };

    let (phys_w, phys_h) = connector.size().unwrap_or((0, 0));
    let edid = smithay_drm_extras::display_info::for_connector(drm, connector.handle());
    let make = edid
        .as_ref()
        .and_then(|i| i.make())
        .unwrap_or_else(|| "Unknown".to_string());
    let model = edid
        .as_ref()
        .and_then(|i| i.model())
        .unwrap_or_else(|| connector_name.clone());
    let serial_number = edid.as_ref().and_then(|i| i.serial()).unwrap_or_default();
    let output = Output::new(
        connector_name.clone(),
        PhysicalProperties {
            size: (phys_w as i32, phys_h as i32).into(),
            subpixel: convert_subpixel(connector.subpixel()),
            make: make.clone(),
            model: model.clone(),
            serial_number: serial_number.clone(),
        },
    );

    let output_mode = Mode {
        size: (mode.size().0 as i32, mode.size().1 as i32).into(),
        refresh: (mode.vrefresh() * 1000) as i32,
    };
    let scale_val = output_cfg.and_then(|c| c.scale).unwrap_or_else(|| {
        tracing::info!(
            "No [[outputs]] entry for '{}' — defaulting to scale 1.0. \
                 Add an [[outputs]] section to config.toml to set a custom scale.",
            connector_name,
        );
        1.0
    });
    let scale = smithay::output::Scale::Fractional(scale_val);
    let transform = output_cfg
        .and_then(|c| c.transform)
        .unwrap_or(Transform::Normal);
    // Mode/scale/transform are set here; the layout position and per-output
    // viewport state are owned by DriftWm::output_connected, called after this
    // returns.
    output.change_current_state(Some(output_mode), Some(transform), Some(scale), None);
    output.set_preferred(output_mode);
    let global = output.create_global::<DriftWm>(dh);

    let allocator = GbmAllocator::new(
        gbm.clone(),
        GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
    );
    // The exporter's import filter must name this device's render node:
    // client buffers allocated there can go to a KMS plane directly
    // (`NodeFilter::None` would veto direct scanout entirely).
    let compositor = match DrmCompositor::new(
        &output,
        drm_surface,
        None,
        allocator.clone(),
        GbmFramebufferExporter::new(gbm.clone(), render_node.into()),
        SUPPORTED_COLOR_FORMATS.iter().copied(),
        render_formats.iter().copied(),
        drm.cursor_size(),
        Some(gbm.clone()),
    ) {
        Ok(c) => c,
        Err(e) => {
            // DrmCompositor::new consumes the surface on error — recreate it.
            // Retry with Modifier::Invalid (implicit) only, which is the most
            // compatible option (lets the driver pick the layout).
            tracing::warn!("DrmCompositor failed ({e:?}), retrying with implicit modifier");
            let _ = std::fs::write("/tmp/driftwm-drm-error.txt", format!("{e:?}"));

            let fallback_surface = match drm.create_surface(crtc, mode, &[connector.handle()]) {
                Ok(s) => s,
                Err(e2) => {
                    tracing::error!("Failed to recreate DRM surface: {e2}");
                    return None;
                }
            };
            let fallback_formats: Vec<Format> = render_formats
                .iter()
                .copied()
                .filter(|f| f.modifier == Modifier::Invalid)
                .collect();

            match DrmCompositor::new(
                &output,
                fallback_surface,
                None,
                allocator,
                GbmFramebufferExporter::new(gbm.clone(), render_node.into()),
                SUPPORTED_COLOR_FORMATS.iter().copied(),
                fallback_formats,
                drm.cursor_size(),
                Some(gbm.clone()),
            ) {
                Ok(c) => c,
                Err(e2) => {
                    tracing::error!("DrmCompositor failed even with implicit modifier: {e2:?}");
                    let _ = std::fs::write(
                        "/tmp/driftwm-drm-error.txt",
                        format!("First: {e:?}\nFallback: {e2:?}"),
                    );
                    return None;
                }
            }
        }
    };

    let gamma_props = GammaProps::new(drm, crtc);
    if gamma_props.is_none() {
        tracing::info!(
            "GAMMA_LUT atomic property unavailable on CRTC {crtc:?} — falling back to legacy \
             drmModeCrtcSetGamma ioctl. Driver may not expose GAMMA_LUT/GAMMA_LUT_SIZE properties."
        );
    }

    let mut dmabuf_feedback = None;
    if let Some(crate::backend::Backend::Udev(udev)) = state.backend.as_mut() {
        let primary_render_node = udev.primary_render_node;
        if let Ok(renderer) = udev.gpu_manager.single_renderer(&primary_render_node) {
            let primary_formats = renderer.dmabuf_formats();
            drop(renderer);
            match surface_dmabuf_feedback(
                &compositor,
                primary_formats,
                primary_render_node,
                render_node,
                kms_node,
            ) {
                Ok(f) => dmabuf_feedback = Some(f),
                Err(e) => {
                    tracing::warn!("Failed to build dmabuf feedback for {connector_name}: {e}");
                }
            }
        }
    }

    Some(SurfaceData {
        compositor,
        output,
        connector: connector.handle(),
        make,
        model,
        serial_number,
        global,
        gamma_props,
        pending_gamma_change: None,
        dmabuf_feedback,
    })
}

/// Tear down a `wl_output` global. Disables it now so clients see the
/// removal event, then queues a delayed `remove_global` so any in-flight
/// bind requests don't hit a freed global and get protocol-killed.
///
/// Callers must send every event referencing this output (`wl_surface.leave`,
/// foreign-toplevel `output_leave`, …) before calling this — see the ordering
/// note in `teardown_output`.
fn remove_output_global(data: &mut DriftWm, global: GlobalId) {
    data.display_handle
        .disable_global::<DriftWm>(global.clone());
    let dh = data.display_handle.clone();
    let timer = Timer::from_duration(Duration::from_secs(10));
    if let Err(e) = data
        .loop_handle
        .insert_source(timer, move |_, _, _: &mut DriftWm| {
            dh.remove_global::<DriftWm>(global.clone());
            TimeoutAction::Drop
        })
    {
        tracing::warn!("Failed to schedule wl_output global removal: {e:?}");
    }
}

/// Drop everything bound to a disconnected output.
///
/// The backend-independent policy (client leaves, capture/grab/fullscreen
/// cleanup, placeholder-vs-drop split) lives in [`DriftWm::output_disconnected`].
/// Here we own the two udev-side pieces: the `wl_output` global teardown, run
/// *after* the policy so every client-facing leave is sent while the global is
/// still valid, and the `active_outputs` removal, symmetric with its insert.
fn teardown_output(data: &mut DriftWm, surface: SurfaceData, is_last: bool) {
    let SurfaceData { output, global, .. } = surface;

    data.output_disconnected(&output, is_last);
    remove_output_global(data, global);
    data.active_outputs.remove(&output);
}

/// How long to wait on a render fence that has been coming back normally.
///
/// Sits well above even a pathological composite (4K, blurred, full-output
/// redraw, several monitors in one pass): tripping this on a merely slow frame
/// would trade a stall for a corrupt one. It bounds each signal-free interval
/// rather than the call — Mesa's native-fence path ends in libsync's
/// `sync_wait`, which restarts `poll` with the full timeout on every `EINTR`.
const FENCE_WAIT_BUDGET: Duration = Duration::from_secs(2);

/// Budget after a single miss. Still far above any real frame: one miss is as
/// easily a GPU climbing out of a power state or a kernel-recovered hang as a
/// wedge, and dropping straight to [`FENCE_WAIT_WEDGED`] would let a legitimate
/// 60ms frame hold the output in the wedged tier from then on.
const FENCE_WAIT_SUSPECT: Duration = Duration::from_millis(250);

/// Budget once a fence has missed twice running. The "might just be slow"
/// reading is spent by then, and the loop is rendering every output serially —
/// at the full budget a wedged GPU leaves under a percent of the loop for the VT
/// switch this bound exists to permit.
const FENCE_WAIT_WEDGED: Duration = Duration::from_millis(50);

/// How often a wait in the wedged tier is taken at [`FENCE_WAIT_SUSPECT`]
/// instead. Every wait in that tier misses by construction once the GPU is
/// merely slower than its budget, so without a periodically longer one an output
/// that came back as slow-but-working would flip early forever.
const FENCE_REPROBE_INTERVAL: u32 = 8;

/// A fence kind the budget can't be applied to has been seen. Unlike the
/// per-CRTC failure counts this is a property of the build, not of a GPU, so one
/// report for the process is all it can ever be worth.
static UNKNOWN_FENCE_SEEN: Once = Once::new();

/// Wait for the GPU to finish the frame, but never indefinitely.
///
/// `SyncPoint::wait` is `eglClientWaitSync` with `EGL_FOREVER` on the
/// compositor's only thread, so a fence that never signals takes the event loop
/// with it — input, Wayland dispatch, and the session notifier a VT switch needs
/// — leaving a reboot as the only way out.
///
/// Giving up does not make the frame correct. This path runs precisely where KMS
/// can't be gated on the fence, so a flip that goes out early can show a partial
/// frame; that is the artifact `wait_for_frame_completion` exists to suppress.
/// The trade is a rare corrupt frame against a session that has to be
/// power-cycled.
///
/// Only bounds the wait it can see: when the renderer can't export a fence at
/// all, smithay falls back to `glFinish` inside `render_frame`, which has
/// already returned by the time this runs.
fn wait_for_fence(data: &mut DriftWm, key: CrtcKey, sync: &SyncPoint, output: &Output) {
    let Some(fence) = sync.get::<EGLFence>() else {
        // A sync point with no fence waits on nothing, but one holding a kind we
        // can't downcast to still has to be awaited: skipping it would tear with
        // nothing in the log to say why. Unreachable with the pinned smithay —
        // the swapchain's fence comes from the GLES renderer, which produces an
        // EGL fence or none — so this only fires after a bump.
        if sync.contains_fence() {
            UNKNOWN_FENCE_SEEN.call_once(|| {
                tracing::warn!(
                    "render fence on {} is not an EGL fence — waiting on it unbounded, which \
                     a wedged GPU can turn into a session that needs a power cycle",
                    output.name()
                );
            });
            let _ = sync.wait();
        }
        return;
    };
    let failures = data.fence_failures.get(&key).copied().unwrap_or(0);
    let budget = match failures {
        0 => FENCE_WAIT_BUDGET,
        n if n == 1 || n % FENCE_REPROBE_INTERVAL == 0 => FENCE_WAIT_SUSPECT,
        _ => FENCE_WAIT_WEDGED,
    };
    // Deliberately no retry on `Err`: EGL has no interrupted status, so this
    // fails only on a real EGL error, and a retry loop would hand back the whole
    // budget on every pass — the unbounded wait this exists to remove.
    let failed = match fence.client_wait(Some(budget), true) {
        Ok(true) => false,
        Ok(false) => {
            report_fence_failure(
                output,
                &format!("still unsignalled after {budget:?}"),
                failures == 0,
            );
            true
        }
        Err(err) => {
            report_fence_failure(output, &format!("wait failed: {err}"), failures == 0);
            true
        }
    };
    if failed {
        data.fence_failures.insert(key, failures + 1);
    } else {
        data.fence_failures.remove(&key);
    }
}

/// Warn on the frame a fence starts failing, then stay quiet until it recovers.
/// The condition persists while the GPU is wedged and the subscriber writes
/// synchronously on this thread, so warning per frame would itself cost frames.
fn report_fence_failure(output: &Output, what: &str, first: bool) {
    if !first {
        tracing::debug!("render fence on {}: {what}", output.name());
        return;
    }
    tracing::warn!(
        "render fence on {}: {what}. Flipping without it — the GPU is not \
         completing work, so expect missing or corrupt frames.",
        output.name()
    );
}

/// Render a single frame and queue it to the DRM compositor. `render_node` is
/// the owning device's render node, deciding same-GPU vs cross-GPU rendering.
fn render_frame(data: &mut DriftWm, surface: &mut SurfaceData, key: CrtcKey, render_node: DrmNode) {
    render_frame_inner(data, surface, key, render_node);
    // Read by the compositing code to skip blur; must not leak into
    // primary-GPU-only renders (IPC screenshots) that run between frames.
    data.render.frame_is_cross_gpu = false;
}

fn render_frame_inner(
    data: &mut DriftWm,
    surface: &mut SurfaceData,
    key: CrtcKey,
    render_node: DrmNode,
) {
    #[cfg(feature = "profile-with-tracy")]
    let _span = tracy_client::span!("udev::render_frame");

    let SurfaceData {
        compositor,
        output,
        dmabuf_feedback,
        ..
    } = surface;
    let output = &*output;

    #[cfg(feature = "profile-with-tracy")]
    {
        static COMMITS_PLOT: std::sync::OnceLock<tracy_client::PlotName> =
            std::sync::OnceLock::new();
        let commits = COMMITS_PLOT
            .get_or_init(|| tracy_client::PlotName::new_leak("frame.commits".to_string()));
        if let Some(client) = tracy_client::Client::running() {
            client.plot(*commits, data.commits_since_render as f64);
        }
    }
    data.commits_since_render = 0;

    data.redraws_needed.remove(output);

    // Flush Wayland clients
    data.display_handle.flush_clients().ok();

    // Read per-output state for this frame
    let (cur_camera, cur_zoom) = data.world_view(output);
    let (last_cam, last_zoom) = {
        let os = crate::state::output_state(output);
        (os.last_rendered_camera, os.last_rendered_zoom)
    };

    // Update background element
    let (camera_moved, zoom_changed, bg_animated) = crate::render::update_background_element(
        data, output, cur_camera, cur_zoom, last_cam, last_zoom,
    );

    // Force full redraw when viewport shifts — DrmCompositor's damage tracker
    // doesn't know all elements moved, so without this we get partial-update artifacts.
    if camera_moved || zoom_changed {
        compositor.reset_buffer_ages();
    }

    // Force full redraw when animated background is visible through transparent windows.
    // smithay's buffer-age optimisation skips recompositing windows whose surface content
    // didn't change — but transparent windows show the background through them, so when
    // the background shader advances a frame the stale composited result is reused and
    // the background appears "frozen" inside those windows.
    // Fix: reset buffer ages so every pixel is redrawn from scratch this frame.
    // Only on frames where the animation actually advanced — between capped
    // ticks the composited result is intentionally reused.
    if bg_animated {
        let focused = data.focus_root_window();
        let has_transparent = data
            .stage
            .windows()
            .filter_map(|w| w.client())
            .any(|w| data.effective_opacity_of(w, focused.as_ref()) < 1.0);
        if has_transparent {
            compositor.reset_buffer_ages();
        }
    }

    // Take the backend out to split the borrow from state, then grab a
    // MultiRenderer for the whole frame. Same GPU: plain single_renderer.
    // Output on another GPU: render on the primary, copy to the scanout GPU
    // in the compositor's framebuffer format (implicit PRIME).
    let mut backend = data.backend.take().unwrap();
    let Backend::Udev(udev) = &mut backend else {
        data.backend = Some(backend);
        return;
    };
    data.render.frame_is_cross_gpu = render_node != udev.primary_render_node;
    let renderer = if render_node == udev.primary_render_node {
        udev.gpu_manager.single_renderer(&udev.primary_render_node)
    } else {
        udev.gpu_manager
            .renderer(&udev.primary_render_node, &render_node, compositor.format())
    };
    let mut renderer = match renderer {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Failed to acquire renderer for {render_node}: {e:?}");
            data.backend = Some(backend);
            queue_estimated_vblank_timer(data, output, key);
            return;
        }
    };

    // Build cursor + compose frame
    let cursor_alpha = if data.active_output().as_ref() == Some(output) {
        1.0
    } else if data.is_output_fullscreen(output)
        || data.is_fullscreen()
        || data.session_lock.is_locked()
    {
        // The ghost cursor shows where the pointer sits on the shared canvas,
        // which only applies between canvas viewports. A fullscreen output is
        // not one — don't ghost the pointer onto a fullscreen output's window,
        // nor project a fullscreen output's pointer onto other monitors. Nor is
        // a locked one: the pointer's location is then screen-space on the
        // active output alone, and means nothing on any other.
        0.0
    } else {
        data.config.inactive_cursor_opacity as f32
    };
    #[cfg(feature = "profile-with-tracy")]
    let _cursor_span = tracy_client::span!("udev::build_cursor_elements");
    // The cursor tracks the live camera, not `world_view` — see its doc for why
    // the two can differ during a fullscreen entry.
    let (cursor_camera, cursor_zoom) = {
        let os = crate::state::output_state(output);
        (os.camera, os.zoom)
    };
    let cursor_elements = crate::render::build_cursor_elements(
        data,
        &mut renderer,
        cursor_camera,
        cursor_zoom,
        output.current_scale().fractional_scale(),
        cursor_alpha,
    );
    #[cfg(feature = "profile-with-tracy")]
    drop(_cursor_span);
    // Read the same predicate `compose_frame` is about to branch on, so the
    // bookkeeping below can never disagree with what was actually painted.
    let lock_frame = data.session_lock.renders_lock_frame();
    let elements = crate::render::compose_frame(data, &mut renderer, output, cursor_elements);

    // Overlay planes are left off — they cause hard-to-diagnose flicker on some
    // hardware. disable_hardware_cursor composites the cursor into the frame instead
    // of using the KMS cursor plane: a workaround for NVIDIA, where a system-memory
    // cursor buffer can't be scanned out (stutter/tearing), while keeping direct
    // scanout for fullscreen apps.
    //
    // Also skip cursor plane scanout when the cursor is dimmed: smithay's cursor plane
    // cache is keyed by element id + commit and ignores alpha, so a 1.0 → <1.0 change
    // reuses the previously-drawn opaque buffer. GPU compositing reapplies alpha.
    let mut frame_flags = FrameFlags::empty();
    if !data.config.backend.disable_direct_scanout {
        frame_flags |= FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY;
    }
    if cursor_alpha >= 1.0 && !data.config.backend.disable_hardware_cursor {
        frame_flags |= FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT;
    }

    // Render via DRM compositor (latency-sensitive — do first)
    #[cfg(feature = "profile-with-tracy")]
    let _composite_span = tracy_client::span!("udev::compositor_render_frame");
    let render_result = compositor.render_frame(
        &mut renderer,
        &elements,
        [0.0f32, 0.0, 0.0, 1.0],
        frame_flags,
    );
    #[cfg(feature = "profile-with-tracy")]
    drop(_composite_span);

    // CPU-wait on the GPU fence when KMS can't gate the flip on it
    // (typical on NVIDIA — EGL fence isn't exportable as IN_FENCE_FD).
    // Config flag forces the wait even when smithay says it's not needed.
    if let Ok(ref rr) = render_result
        && (rr.needs_sync() || data.config.backend.wait_for_frame_completion)
        && let PrimaryPlaneElement::Swapchain(ref element) = rr.primary_element
    {
        // `has_fence` distinguishes a real wait from the fenceless case, which
        // reports needs_sync but blocks on nothing — without it a smoke test
        // reads a permanent no-op as "the path is exercised".
        tracing::debug!(
            "Fence wait: needs_sync={}, force={}, has_fence={}",
            rr.needs_sync(),
            data.config.backend.wait_for_frame_completion,
            element.sync.contains_fence(),
        );
        wait_for_fence(data, key, &element.sync, output);
    }

    match render_result {
        Ok(render_result) => {
            crate::render::update_primary_scanout_output(data, output, &render_result.states);
            if let Some(surface_feedback) = dmabuf_feedback.as_ref() {
                crate::render::send_dmabuf_feedbacks(
                    data,
                    output,
                    surface_feedback,
                    &render_result.states,
                );
            }
            let feedback =
                crate::render::take_presentation_feedback(data, output, &render_result.states);
            let queue_result = {
                #[cfg(feature = "profile-with-tracy")]
                let _span = tracy_client::span!("udev::queue_frame");
                compositor.queue_frame(feedback)
            };
            match queue_result {
                Ok(()) => {
                    data.frames_pending.insert(key);
                    if lock_frame {
                        data.lock_frame_queued.insert(key);
                    } else {
                        data.lock_frame_queued.remove(&key);
                    }
                }
                Err(FrameError::EmptyFrame) => {
                    // No page flip - no real VBlank to wake us. Always arm the
                    // estimated timer so the render gate paces re-renders to the refresh
                    // period; otherwise a dirty-but-unchanged output spins render_frame.
                    //
                    // Nothing was queued, so whatever is scanned out stays — and
                    // if that was already a lock frame, this output owes nothing
                    // more. Load-bearing, not belt-and-braces: an output with no
                    // lock surface of its own paints black in both lock states,
                    // so the redraw that seeds the wait produces no damage, no
                    // flip and therefore no VBlank, ever. Reading
                    // `lock_frame_on_screen` is only sound because the render
                    // gate keeps `render_frame` out of the in-flight window, so
                    // smithay's `pending_frame` is `None` at every reachable
                    // `EmptyFrame` — an invariant of the gate, not of
                    // `EmptyFrame`, and it stops holding if triple-buffering
                    // relaxes the gate.
                    if lock_frame && data.lock_frame_on_screen.contains(&key) {
                        data.stop_awaiting_lock_frame(output);
                    }
                    queue_estimated_vblank_timer(data, output, key);
                }
                Err(e) => {
                    tracing::warn!("Failed to queue frame: {e:?}");
                    retry_dropped_lock_frame(data, output);
                    queue_estimated_vblank_timer(data, output, key);
                }
            }
        }
        Err(e) => {
            tracing::warn!("Render frame error: {e:?}");
            retry_dropped_lock_frame(data, output);
            queue_estimated_vblank_timer(data, output, key);
        }
    }

    // Fulfill capture requests after main render
    #[cfg(feature = "profile-with-tracy")]
    let _captures_span = tracy_client::span!("udev::captures");
    // Captures always run on the primary GPU: on a cross-GPU MultiRenderer,
    // bind/create_buffer target the scanout GPU, which can't bind client
    // buffers allocated on the render GPU, and cached capture textures would
    // hop between GL contexts.
    if data.render.frame_is_cross_gpu {
        // The elements above are typed to the cross-GPU renderer, whose
        // bind/create_buffer target the scanout GPU — which can't bind client
        // buffers allocated on the render GPU. Recompose on a primary-only
        // renderer, and only when someone is actually capturing.
        drop(elements);
        drop(renderer);
        if crate::render::capture_work_pending(data, output) {
            data.render.frame_is_cross_gpu = false;
            match udev.gpu_manager.single_renderer(&udev.primary_render_node) {
                Ok(mut primary) => {
                    let cursor = crate::render::build_cursor_elements(
                        data,
                        &mut primary,
                        cursor_camera,
                        cursor_zoom,
                        output.current_scale().fractional_scale(),
                        cursor_alpha,
                    );
                    let elements = crate::render::compose_frame(data, &mut primary, output, cursor);
                    run_captures(data, &mut primary, output, &elements);
                }
                Err(e) => tracing::warn!("Capture skipped, primary renderer unavailable: {e:?}"),
            }
        }
    } else {
        run_captures(data, &mut renderer, output, &elements);
        drop(elements);
        drop(renderer);
    }
    #[cfg(feature = "profile-with-tracy")]
    drop(_captures_span);

    // The renderer (borrowing the backend's GPU manager) is gone by now; put
    // the backend back on state.
    data.backend = Some(backend);
    finish_frame(data, output);

    #[cfg(feature = "profile-with-tracy")]
    {
        drop(_span);
        tracy_client::Client::running().map(|c| c.frame_mark());
    }
}

fn run_captures<R: crate::render::DriftRenderer>(
    data: &mut DriftWm,
    renderer: &mut R,
    output: &Output,
    elements: &[crate::render::OutputRenderElements<R>],
) where
    crate::render::OutputRenderElements<R>: smithay::backend::renderer::element::RenderElement<R>,
{
    crate::render::render_screencopy(data, renderer, output, elements);
    crate::render::render_capture_frames(data, renderer, output, elements);
    crate::render::render_toplevel_captures(data, renderer);
}

/// Bookkeeping after a frame's GPU work is done and the backend is back on
/// state.
fn finish_frame(data: &mut DriftWm, output: &Output) {
    // Record camera+zoom for next-frame change detection
    {
        let (camera, zoom) = data.world_view(output);
        let mut os = crate::state::output_state(output);
        os.last_rendered_camera = camera;
        os.last_rendered_zoom = zoom;
    }
    data.write_state_file_if_dirty();

    // Post-render
    #[cfg(feature = "profile-with-tracy")]
    let _post_span = tracy_client::span!("udev::post_render");
    crate::render::post_render(data, output);
    data.display_handle.flush_clients().ok();
    #[cfg(feature = "profile-with-tracy")]
    drop(_post_span);
}

/// Forward a page-flip's timing to all clients waiting on `wp_presentation`.
/// `meta` carries the kernel timestamp + sequence; if it's missing (rare on
/// some drivers) we discard rather than fabricate, per protocol guidance.
fn deliver_presentation(
    feedback: &mut smithay::desktop::utils::OutputPresentationFeedback,
    output: &Output,
    meta: Option<&smithay::backend::drm::DrmEventMetadata>,
) {
    use smithay::backend::drm::DrmEventTime as DrmTime;
    use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
    use smithay::wayland::presentation::Refresh;

    let Some(meta) = meta else {
        feedback.discarded();
        return;
    };

    let refresh_picos = output
        .current_mode()
        .map(|m| (1_000_000_000_000u64 / (m.refresh.max(1) as u64)) as u32)
        .unwrap_or(0);
    let refresh = Refresh::Fixed(Duration::from_nanos(refresh_picos as u64));

    let flags = wp_presentation_feedback::Kind::Vsync
        | wp_presentation_feedback::Kind::HwClock
        | wp_presentation_feedback::Kind::HwCompletion;

    match meta.time {
        DrmTime::Monotonic(time) => {
            feedback.presented::<_, smithay::utils::Monotonic>(
                time,
                refresh,
                meta.sequence as u64,
                flags,
            );
        }
        DrmTime::Realtime(_) => {
            // We advertised CLOCK_MONOTONIC; a realtime stamp from the kernel
            // can't be reported safely against that clock id.
            feedback.discarded();
        }
    }
}

/// A frame that never reached the screen leaves `redraws_needed` already drained
/// (`render_frame` removes the entry before compositing), so nothing retries it.
/// While the output still owes a lock frame that would stall the confirmation to
/// the timeout — re-arm it so the post-dispatch `render_if_needed` in `main`
/// composites again (the estimated-VBlank timer only supplies the wake-up; its
/// callback does nothing but drop its own token). Scoped to the awaiting case;
/// dropped frames are otherwise not retried.
fn retry_dropped_lock_frame(data: &mut DriftWm, output: &Output) {
    if data.is_awaiting_lock_frame(output) {
        data.redraws_needed.insert(output.clone());
    }
}

/// Wake the VBlank-driven loop at ~one refresh period when queue_frame returned
/// EmptyFrame, so ongoing animations keep ticking. Idempotent per CRTC.
fn queue_estimated_vblank_timer(data: &mut DriftWm, output: &Output, key: CrtcKey) {
    if data.estimated_vblank_timers.contains_key(&key) {
        return;
    }
    // Clamp refresh mHz before the cast: negative i32 would wrap to a huge u64 and
    // produce a near-zero-duration timer, spinning the loop.
    let duration = output
        .current_mode()
        .map(|m| m.refresh.max(1_000) as u64)
        .map(|mhz| Duration::from_nanos(1_000_000_000_000 / mhz))
        .unwrap_or_else(|| Duration::from_micros(16_667));

    let timer = Timer::from_duration(duration);
    match data
        .loop_handle
        .insert_source(timer, move |_, _, data: &mut DriftWm| {
            data.estimated_vblank_timers.remove(&key);
            TimeoutAction::Drop
        }) {
        Ok(tok) => {
            data.estimated_vblank_timers.insert(key, tok);
        }
        Err(e) => tracing::warn!("Failed to insert estimated VBlank timer: {e:?}"),
    }
}

use driftwm::protocols::output_management::{ModeInfo, OutputHeadState};

/// Apply queued mode changes for outputs on this device via
/// `DrmCompositor::use_mode`; entries for other devices' outputs are returned
/// unclaimed. Safe with a page flip in flight: `use_mode` only validates
/// (TEST_ONLY commit) and stages the mode; the real modeset lands with the next
/// frame commit.
fn apply_pending_mode_changes(
    drm: &DrmDevice,
    surfaces: &mut HashMap<crtc::Handle, SurfaceData>,
    data: &mut DriftWm,
    pending: HashMap<String, crate::state::ModeIntent>,
) -> HashMap<String, crate::state::ModeIntent> {
    use smithay::reexports::drm::control::Device as ControlDevice;

    let mut unclaimed = HashMap::new();
    for (name, intent) in pending {
        let Some((_, surface)) = surfaces.iter_mut().find(|(_, s)| s.output.name() == name) else {
            unclaimed.insert(name, intent);
            continue;
        };

        let connector = match ControlDevice::get_connector(drm, surface.connector, false) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Mode change for '{name}': get_connector failed: {e}");
                continue;
            }
        };

        let Some(mode) = resolve_pending_mode(&intent, &connector, &name) else {
            tracing::error!("Mode change for '{name}': could not resolve intent {intent:?}");
            continue;
        };

        let new_smithay_mode = Mode {
            size: (mode.size().0 as i32, mode.size().1 as i32).into(),
            refresh: (mode.vrefresh() * 1000) as i32,
        };

        // Config reload queues Preferred/Max on every reload, so skip the
        // modeset when it resolves to the mode already in use. Scoped to those
        // rule-derived intents: reload already change-detects Size/SizeRefresh,
        // and explicit EdidIndex/Custom requests are one-shot user actions
        // worth honoring even when nominally identical to the current mode.
        if matches!(
            intent,
            crate::state::ModeIntent::Preferred | crate::state::ModeIntent::Max
        ) && surface.output.current_mode() == Some(new_smithay_mode)
        {
            tracing::debug!("Mode change for '{name}': already at requested mode, skipping");
            continue;
        }

        match surface.compositor.use_mode(mode) {
            Ok(_) => {
                surface
                    .output
                    .change_current_state(Some(new_smithay_mode), None, None, None);
                surface.output.set_preferred(new_smithay_mode);
                // Re-anchor layer surfaces (waybar/mako/swaync) to the new
                // output dimensions. Without this they keep their old
                // geometry until the client re-anchors itself.
                {
                    let mut map = smithay::desktop::layer_map_for_output(&surface.output);
                    map.arrange();
                }
                // Resize fullscreen window (if any) to the new viewport.
                let new_size =
                    smithay::utils::Size::from((mode.size().0 as i32, mode.size().1 as i32));
                data.resize_fullscreen_for_output(&surface.output, new_size);
                data.render.remove_output(&name);
                data.redraws_needed.insert(surface.output.clone());
                data.output_config_dirty = true;
                tracing::info!(
                    "Mode change applied to '{name}': {}x{}@{}Hz",
                    mode.size().0,
                    mode.size().1,
                    mode.vrefresh(),
                );
            }
            Err(e) => {
                tracing::error!("Mode change rejected by kernel for '{name}': {e:?}");
                // Re-broadcast so clients see the state didn't actually move.
                data.output_config_dirty = true;
            }
        }
    }
    unclaimed
}

/// Aggregate wlr-output-management head state across every DRM device.
fn collect_all_head_states(data: &DriftWm) -> HashMap<String, OutputHeadState> {
    let mut head_state = HashMap::new();
    for device in data.udev_devices.values() {
        let dev = device.0.borrow();
        head_state.extend(collect_output_state_from_surfaces(&dev.surfaces, &dev.drm));
    }
    head_state
}

fn collect_output_state_from_surfaces(
    surfaces: &HashMap<crtc::Handle, SurfaceData>,
    drm: &DrmDevice,
) -> HashMap<String, OutputHeadState> {
    use smithay::reexports::drm::control::Device as ControlDevice;
    let mut result = HashMap::new();
    for surface in surfaces.values() {
        let output = &surface.output;
        let name = output.name();
        let mode = output.current_mode().unwrap();
        let transform = output.current_transform();
        let scale = output.current_scale().fractional_scale();
        let layout_pos = crate::state::output_state(output).layout_position;

        let mut modes: Vec<ModeInfo> =
            match ControlDevice::get_connector(drm, surface.connector, false) {
                Ok(info) => info
                    .modes()
                    .iter()
                    .map(|m| ModeInfo {
                        width: m.size().0 as i32,
                        height: m.size().1 as i32,
                        refresh: (m.vrefresh() as i32) * 1000,
                        preferred: m.mode_type().contains(control::ModeTypeFlags::PREFERRED),
                    })
                    .collect(),
                Err(_) => vec![],
            };

        // If the active mode is a CVT-synthesized one (not in the EDID list),
        // append it so `wlr-randr` can show it as current. Without this the
        // user runs `wlr-randr --custom-mode ...`, sees the display change,
        // and then sees the old mode list with nothing marked current — looks
        // broken.
        let mut current_mode_index = modes.iter().position(|m| {
            m.width == mode.size.w && m.height == mode.size.h && m.refresh == mode.refresh
        });
        if current_mode_index.is_none() {
            modes.push(ModeInfo {
                width: mode.size.w,
                height: mode.size.h,
                refresh: mode.refresh,
                preferred: false,
            });
            current_mode_index = Some(modes.len() - 1);
        }

        let phys = output.physical_properties().size;
        result.insert(
            name.clone(),
            OutputHeadState {
                name,
                description: format!("{} {} ({})", surface.make, surface.model, output.name()),
                make: surface.make.clone(),
                model: surface.model.clone(),
                serial_number: surface.serial_number.clone(),
                physical_size: (phys.w, phys.h),
                modes,
                current_mode_index,
                position: (layout_pos.x, layout_pos.y),
                transform,
                scale,
            },
        );
    }
    result
}

fn convert_subpixel(sp: connector::SubPixel) -> Subpixel {
    match sp {
        connector::SubPixel::Unknown => Subpixel::Unknown,
        connector::SubPixel::HorizontalRgb => Subpixel::HorizontalRgb,
        connector::SubPixel::HorizontalBgr => Subpixel::HorizontalBgr,
        connector::SubPixel::VerticalRgb => Subpixel::VerticalRgb,
        connector::SubPixel::VerticalBgr => Subpixel::VerticalBgr,
        connector::SubPixel::None => Subpixel::None,
        _ => Subpixel::Unknown,
    }
}
