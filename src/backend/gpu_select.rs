//! Choosing which GPU composites frames, from the `[backend]` config.

use std::fs;
use std::path::{Path, PathBuf};

use driftwm::config::RenderGpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuKind {
    Integrated,
    Discrete,
}

const VENDOR_NVIDIA: u32 = 0x10de;
const VENDOR_INTEL: u32 = 0x8086;

/// Heuristic: Intel iGPUs sit on the root PCI bus (00:02.0) while Arc cards sit
/// behind a bridge; NVIDIA GPUs in laptops and desktops are discrete; for AMD
/// (APU and dGPU look alike on the bus) the boot GPU is taken as the integrated
/// one, which is what hybrid laptops do. Anything else — platform devices on
/// ARM — counts as integrated.
pub fn classify_pci(vendor: u32, on_root_bus: bool, boot_vga: bool) -> GpuKind {
    match vendor {
        VENDOR_NVIDIA => GpuKind::Discrete,
        VENDOR_INTEL if on_root_bus => GpuKind::Integrated,
        VENDOR_INTEL => GpuKind::Discrete,
        _ if boot_vga => GpuKind::Integrated,
        _ => GpuKind::Discrete,
    }
}

/// Read sysfs (no device open, so it never wakes a suspended GPU).
pub fn classify(path: &Path) -> GpuKind {
    let Some(name) = path.file_name() else {
        return GpuKind::Integrated;
    };
    let device = Path::new("/sys/class/drm").join(name).join("device");
    let Some(vendor) = fs::read_to_string(device.join("vendor"))
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
    else {
        return GpuKind::Integrated;
    };
    let on_root_bus = fs::canonicalize(&device)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .and_then(|addr| addr.split(':').nth(1).map(str::to_owned))
        .is_some_and(|bus| bus == "00");
    let boot_vga = fs::read_to_string(device.join("boot_vga")).is_ok_and(|s| s.trim() == "1");
    classify_pci(vendor, on_root_bus, boot_vga)
}

/// Reorder `candidates` (system boot GPU first) so the configured render GPU
/// is tried first. Returns the new order and, when the setting matched no
/// device, a message for the log; the order is then left as is (auto).
pub fn order_candidates(
    candidates: Vec<PathBuf>,
    setting: &RenderGpu,
    kind_of: impl Fn(&Path) -> GpuKind,
) -> (Vec<PathBuf>, Option<String>) {
    let wanted: Box<dyn Fn(&PathBuf) -> bool> = match setting {
        RenderGpu::Auto => return (candidates, None),
        RenderGpu::Integrated => Box::new(|p| kind_of(p) == GpuKind::Integrated),
        RenderGpu::Discrete => Box::new(|p| kind_of(p) == GpuKind::Discrete),
        RenderGpu::Path(want) => Box::new(move |p| {
            // by-path/by-id symlinks resolve to the same card node.
            p == Path::new(want)
                || matches!(
                    (fs::canonicalize(p), fs::canonicalize(want)),
                    (Ok(a), Ok(b)) if a == b
                )
        }),
    };
    if !candidates.iter().any(&wanted) {
        let msg = format!("render_gpu {setting:?} matches no GPU in {candidates:?}, using auto");
        return (candidates, Some(msg));
    }
    let (mut first, rest): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|p| wanted(p));
    first.extend(rest);
    (first, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> Vec<PathBuf> {
        vec!["/dev/dri/card1".into(), "/dev/dri/card0".into()]
    }

    fn kind(p: &Path) -> GpuKind {
        if p == Path::new("/dev/dri/card0") {
            GpuKind::Discrete
        } else {
            GpuKind::Integrated
        }
    }

    #[test]
    fn hybrid_laptop_classification() {
        assert_eq!(classify_pci(VENDOR_INTEL, true, true), GpuKind::Integrated);
        assert_eq!(classify_pci(VENDOR_NVIDIA, false, false), GpuKind::Discrete);
        assert_eq!(classify_pci(VENDOR_NVIDIA, false, true), GpuKind::Discrete);
        assert_eq!(classify_pci(VENDOR_INTEL, false, false), GpuKind::Discrete);
        assert_eq!(classify_pci(0x1002, false, true), GpuKind::Integrated);
        assert_eq!(classify_pci(0x1002, false, false), GpuKind::Discrete);
    }

    #[test]
    fn auto_keeps_boot_gpu_first() {
        let (order, warn) = order_candidates(paths(), &RenderGpu::Auto, kind);
        assert_eq!(order, paths());
        assert!(warn.is_none());
    }

    #[test]
    fn discrete_moves_dgpu_first() {
        let (order, warn) = order_candidates(paths(), &RenderGpu::Discrete, kind);
        assert_eq!(order[0], PathBuf::from("/dev/dri/card0"));
        assert_eq!(order.len(), 2);
        assert!(warn.is_none());
    }

    #[test]
    fn integrated_keeps_igpu_first() {
        let (order, _) = order_candidates(paths(), &RenderGpu::Integrated, kind);
        assert_eq!(order[0], PathBuf::from("/dev/dri/card1"));
    }

    #[test]
    fn explicit_path_wins() {
        let setting = RenderGpu::Path("/dev/dri/card0".into());
        let (order, warn) = order_candidates(paths(), &setting, kind);
        assert_eq!(order[0], PathBuf::from("/dev/dri/card0"));
        assert!(warn.is_none());
    }

    #[test]
    fn unmatched_setting_falls_back_with_a_message() {
        let (order, warn) = order_candidates(
            vec!["/dev/dri/card0".into()],
            &RenderGpu::Integrated,
            |_| GpuKind::Discrete,
        );
        assert_eq!(order, vec![PathBuf::from("/dev/dri/card0")]);
        assert!(warn.is_some());
    }
}
