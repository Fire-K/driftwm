pub mod cvt;
pub mod gamma;
pub mod gpu_select;
pub mod udev;
pub mod winit;

use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::winit::WinitGraphicsBackend;

use crate::backend::udev::UdevRenderer;
use crate::render::AsGlesRenderer;

/// Backend abstraction — winit (nested) or udev (real hardware).
/// Only the renderer lives here; udev-specific state (DRM, session, etc.)
/// is captured by calloop closures in udev.rs.
pub enum Backend {
    Winit(Box<WinitGraphicsBackend<GlesRenderer>>),
    Udev(Box<UdevRenderer>),
    /// A bare renderer for the test fixtures, which have no window or DRM
    /// device but still exercise the udev-only code paths.
    #[cfg(test)]
    Headless(Box<GlesRenderer>),
}

impl Backend {
    /// Run `f` with a primary-GPU [`GlesRenderer`] for one-off work (shader
    /// compilation, dmabuf import, off-screen screenshot). For udev this is the
    /// underlying GlesRenderer of the multi-GPU manager's primary render node.
    /// Returns `None` when that renderer is unavailable — e.g. after the
    /// primary GPU was unplugged, which keeps the compositor alive.
    ///
    /// The render loop does NOT go through this — it grabs a full
    /// `MultiGpuRenderer` via `single_renderer` so cross-GPU scanout works.
    pub fn with_renderer<T>(&mut self, f: impl FnOnce(&mut GlesRenderer) -> T) -> Option<T> {
        match self {
            Backend::Winit(backend) => Some(f(backend.renderer())),
            #[cfg(test)]
            Backend::Headless(renderer) => Some(f(renderer)),
            Backend::Udev(udev) => {
                match udev.gpu_manager.single_renderer(&udev.primary_render_node) {
                    Ok(mut renderer) => Some(f(renderer.as_gles_renderer())),
                    Err(err) => {
                        tracing::warn!("primary GPU renderer unavailable: {err:?}");
                        None
                    }
                }
            }
        }
    }

    /// The renderer of a headless test backend, for tests that compose and read
    /// back frames with a concrete `GlesRenderer`.
    #[cfg(test)]
    pub fn renderer(&mut self) -> &mut GlesRenderer {
        match self {
            Backend::Headless(renderer) => renderer,
            Backend::Winit(backend) => backend.renderer(),
            Backend::Udev(_) => panic!("the udev backend has no bare renderer"),
        }
    }

    /// Start importing a committed surface's buffer on the primary GPU before
    /// the next frame needs it, overlapping the import (and any cross-GPU copy)
    /// with the client's remaining work instead of paying it at render time.
    pub fn early_import(
        &mut self,
        surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    ) {
        // With a secondary GPU registered, smithay may pick it as the import
        // source for a buffer of unknown origin and read it back through
        // system memory; the import then happens at render time on the right
        // GPU instead.
        if let Backend::Udev(udev) = self
            && udev.secondary_render_nodes.is_empty()
            && let Err(err) = udev
                .gpu_manager
                .early_import(udev.primary_render_node, surface)
        {
            tracing::warn!("early import failed: {err:?}");
        }
    }
}
