mod cosmic_text_system;
#[cfg(all(unix, not(target_family = "wasm")))]
mod dmabuf;
#[cfg(all(windows, not(target_family = "wasm")))]
mod win32;
mod wgpu_atlas;
mod wgpu_context;
mod wgpu_renderer;

pub use cosmic_text_system::*;
#[cfg(all(unix, not(target_family = "wasm")))]
pub use dmabuf::DmabufInfo;
#[cfg(all(windows, not(target_family = "wasm")))]
pub use win32::SharedTextureInfo;

// The embed export differs per OS in handle type and in what the consumer needs
// to import it (a dma-buf plane vs. a whole shareable allocation), but the
// renderer only ever stores targets and hands out handles. It uses these names
// so that stays one code path.
#[cfg(all(unix, not(target_family = "wasm")))]
pub(crate) use dmabuf as export;
#[cfg(all(windows, not(target_family = "wasm")))]
pub(crate) use win32 as export;
#[cfg(not(target_family = "wasm"))]
pub use export::{ExportHandle, ExportInfo};
pub use wgpu;
pub use wgpu_atlas::*;
pub use wgpu_context::*;
pub use wgpu_renderer::{GpuContext, WgpuRenderer, WgpuSurfaceConfig};
