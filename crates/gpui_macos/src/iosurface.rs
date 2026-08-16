//! The zero-copy half of the macOS embed transport: frames land in IOSurfaces
//! the consumer samples directly, instead of being read back and copied over the
//! socket.
//!
//! Linux hands the consumer DRM PRIME fds over `SCM_RIGHTS` and Windows
//! duplicates a D3D12 NT handle into it. Neither shape exists here: Darwin's
//! `sys/socket.h` defines no `SCM_*` type for mach ports, so the surface right
//! cannot ride the unix side channel that carries every other embed message.
//! What it can ride is the bootstrap namespace — this module registers a
//! per-window service name, the `IOSF` handshake sends that name as plain text,
//! and a consumer that looks the name up gets one send right per buffer back.
//!
//! `kIOSurfaceIsGlobal` would skip all of that (a `u32` surface ID over the
//! socket, no mach code at all), but it has been deprecated as insecure since
//! 10.11 — it publishes the surface to every process on the machine, not just
//! the host that asked for it.

use std::ffi::CString;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result, anyhow, bail};
use core_graphics::display::{CGDirectDisplayID, CGDisplay};
use foreign_types::ForeignType as _;
use mach2::bootstrap::{BOOTSTRAP_MAX_NAME_LEN, bootstrap_check_in, bootstrap_port};
use mach2::kern_return::KERN_SUCCESS;
use mach2::mach_port::{mach_port_deallocate, mach_port_mod_refs};
use mach2::message::{
    MACH_MSG_TYPE_COPY_SEND, MACH_MSGH_BITS, MACH_MSGH_BITS_COMPLEX, MACH_MSGH_BITS_REMOTE_MASK,
    MACH_RCV_MSG, MACH_RCV_TIMED_OUT, MACH_RCV_TIMEOUT, MACH_SEND_MSG, mach_msg, mach_msg_body_t,
    mach_msg_header_t, mach_msg_port_descriptor_t, mach_msg_trailer_t,
};
use mach2::port::{MACH_PORT_NULL, MACH_PORT_RIGHT_RECEIVE, mach_port_t};
use mach2::traps::mach_task_self;
use objc::{msg_send, sel, sel_impl};
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained};
use objc2_io_surface::{
    IOSurfaceRef, kIOSurfaceBytesPerElement, kIOSurfaceHeight, kIOSurfacePixelFormat,
    kIOSurfaceWidth,
};

/// Smallest over-allocation, and the figures the Linux producer hardcodes
/// (`EMBED_DMABUF_MAX_WIDTH` / `EMBED_DMABUF_MAX_HEIGHT` in
/// `gpui_linux/src/linux/headless/client.rs`).
const MAX_SIZE_FLOOR: (u32, u32) = (3840, 2160);

/// Largest over-allocation, so a bogus or absurd display mode cannot make the
/// producer ask for gigabytes. Each buffer costs `width * height * 4` bytes and
/// there are [`EMBED_BUFFER_COUNT`] of them, i.e. 265 MB at this ceiling.
const MAX_SIZE_CEILING: (u32, u32) = (7680, 4320);

/// Over-allocated surface size: the buffers are created once at this size and
/// each frame renders into their top-left, so a resize needs neither a second
/// handshake nor a reallocation.
///
/// Linux hardcodes 4K here and never notices the ceiling, because the path that
/// actually runs there today is the CPU one, which reallocates per resize and has
/// no cap at all. On this side the cap is visible: a pane wider than the surface
/// is stretched to fill rather than letterboxed — `CALayer`'s default `resize`
/// gravity, which the consumer picked to match GTK's `ContentFit::Fill` — so on a
/// display whose backing size exceeds 3840x2160 every glyph comes out
/// proportionally wide. Hence the maximum is taken from the largest attached
/// display's backing pixels, which is the same product the consumer derives its
/// resize requests from (`NSScreen` frame times `backingScaleFactor`), floored at
/// the Linux figure so the ordinary case allocates exactly what Linux does.
///
/// Computed once: a display that arrives mid-session cannot resize buffers a
/// consumer already holds send rights to, and re-handshaking is the thing this
/// over-allocation exists to avoid.
pub(crate) fn embed_max_size() -> (u32, u32) {
    static SIZE: OnceLock<(u32, u32)> = OnceLock::new();
    *SIZE.get_or_init(|| {
        let (mut width, mut height) = MAX_SIZE_FLOOR;
        for display in active_displays() {
            let (display_width, display_height) = backing_size(display);
            width = width.max(display_width);
            height = height.max(display_height);
        }
        (
            width.min(MAX_SIZE_CEILING.0),
            height.min(MAX_SIZE_CEILING.1),
        )
    })
}

fn active_displays() -> Vec<CGDirectDisplayID> {
    match CGDisplay::active_displays() {
        Ok(displays) => displays,
        Err(error) => {
            log::warn!(
                "[zed-embed] listing the active displays failed ({error}); \
                 sizing the embed surfaces at {}x{}",
                MAX_SIZE_FLOOR.0,
                MAX_SIZE_FLOOR.1
            );
            Vec::new()
        }
    }
}

/// A display's size in backing pixels.
///
/// Not `CGDisplayModeGetPixelWidth` directly: that reports the mode's own
/// orientation, while the bounds are rotation-aware, and a rotated display would
/// otherwise be measured across the wrong axis. The mode is used only for the
/// ratio between backing and logical pixels, which rotation leaves alone.
fn backing_size(display: CGDirectDisplayID) -> (u32, u32) {
    let display = CGDisplay::new(display);
    let Some(mode) = display.display_mode() else {
        return (0, 0);
    };
    let (logical_width, logical_height) = (mode.width(), mode.height());
    if logical_width == 0 || logical_height == 0 {
        return (0, 0);
    }
    let horizontal_scale = mode.pixel_width() as f64 / logical_width as f64;
    let vertical_scale = mode.pixel_height() as f64 / logical_height as f64;
    let bounds = display.bounds().size;
    // A saturating cast, so a nonsensical mode yields 0 or the ceiling rather
    // than wrapping into a plausible-looking size.
    let scaled = |extent: f64, scale: f64| (extent * scale).round() as u32;
    (
        scaled(bounds.width, horizontal_scale),
        scaled(bounds.height, vertical_scale),
    )
}

/// Frames alternate between two surfaces, so a frame is never rendered into the
/// one the consumer is still sampling. Stage A's CPU path needs only one target
/// because its readback finishes before `draw` returns; here the consumer reads
/// asynchronously, at whatever moment CoreAnimation next composites.
pub(crate) const EMBED_BUFFER_COUNT: usize = 2;

/// `'BGRA'` as a FourCC, matching the `BGRA8Unorm` textures the renderer draws
/// into and the byte order every embed transport puts on the wire.
const PIXEL_FORMAT_BGRA: i32 = 0x4247_5241;

/// Message ids for the rendezvous exchange. Only the pair defined here is
/// served: a stray message on the service port is far more likely to be a bug
/// than a request, and replying to it would hand out send rights.
const SURFACE_REQUEST_ID: i32 = 100;
const SURFACE_REPLY_ID: i32 = 200;

/// How long the rendezvous thread blocks in `mach_msg` before re-checking
/// whether its session is still alive.
const RECEIVE_TIMEOUT_MS: u32 = 500;

/// Distinguishes the services of concurrent panes. One embed process serves
/// every pane the host opens, and `bootstrap_check_in` fails with
/// `BOOTSTRAP_SERVICE_ACTIVE` if a live registration already holds the name, so
/// the pid alone is not enough.
static NEXT_SERVICE: AtomicU32 = AtomicU32::new(0);

/// The consumer's request. The body is empty — the reply port in the header is
/// the entire point of the message.
#[repr(C)]
#[derive(Default)]
struct SurfaceRequest {
    header: mach_msg_header_t,
    trailer: mach_msg_trailer_t,
}

/// The reply: one port descriptor per buffer, in buffer-index order, which is
/// the order `RDY!` indexes into.
#[repr(C)]
struct SurfaceReply {
    header: mach_msg_header_t,
    body: mach_msg_body_t,
    surfaces: [mach_msg_port_descriptor_t; EMBED_BUFFER_COUNT],
}

/// The surfaces a zero-copy session renders into, and the bootstrap service that
/// hands them to the consumer.
pub(crate) struct EmbedSurfaces {
    /// The surfaces themselves are not held separately: an `MTLTexture` created
    /// from one retains it (that is what its `iosurface` property returns), and so
    /// does every mach port made from it, so a third reference here would keep
    /// alive exactly what these already do.
    textures: [metal::Texture; EMBED_BUFFER_COUNT],
    service_name: String,
    /// Cleared on drop. The rendezvous thread notices within its receive timeout
    /// and releases the mach rights from there, because once it starts it is
    /// their only owner — a port name is per-task, so deallocating from both
    /// sides would be a double free of the same name.
    alive: Arc<AtomicBool>,
}

impl EmbedSurfaces {
    pub(crate) fn new(device: &metal::Device) -> Result<Self> {
        let mut textures = Vec::with_capacity(EMBED_BUFFER_COUNT);
        let mut ports = Vec::with_capacity(EMBED_BUFFER_COUNT);
        for index in 0..EMBED_BUFFER_COUNT {
            match create_buffer(device) {
                Ok((texture, port)) => {
                    textures.push(texture);
                    ports.push(port);
                }
                Err(error) => {
                    // Whatever succeeded before this failure is holding rights
                    // that nothing else will ever give back: the rendezvous
                    // thread, which owns them once it starts, does not exist yet.
                    unsafe { release_rights(MACH_PORT_NULL, &ports) };
                    return Err(error).with_context(|| format!("creating embed buffer {index}"));
                }
            }
        }
        let textures: [metal::Texture; EMBED_BUFFER_COUNT] = textures
            .try_into()
            .map_err(|_| anyhow!("expected {EMBED_BUFFER_COUNT} embed textures"))?;
        let ports: [mach_port_t; EMBED_BUFFER_COUNT] = ports
            .try_into()
            .map_err(|_| anyhow!("expected {EMBED_BUFFER_COUNT} embed surface ports"))?;

        let (service_name, alive) = match serve(ports) {
            Ok(served) => served,
            Err(error) => {
                unsafe { release_rights(MACH_PORT_NULL, &ports) };
                // Not fatal to the session: the caller falls back to CPU frames,
                // which is also what a consumer that never negotiated this
                // capability gets.
                return Err(error);
            }
        };

        let (max_width, max_height) = embed_max_size();
        log::info!(
            "[zed-embed] serving {EMBED_BUFFER_COUNT} IOSurfaces of \
             {max_width}x{max_height} as {service_name}"
        );
        Ok(Self {
            textures,
            service_name,
            alive,
        })
    }

    pub(crate) fn service_name(&self) -> &str {
        &self.service_name
    }

    pub(crate) fn texture(&self, index: usize) -> Option<&metal::TextureRef> {
        self.textures.get(index).map(|texture| texture.as_ref())
    }
}

impl Drop for EmbedSurfaces {
    fn drop(&mut self) {
        // Stopping the thread is all this has to do: it owns the mach rights and
        // releases them on its way out, and the surfaces and textures are
        // ordinary CoreFoundation and Metal objects that the consumer's own
        // references keep alive for as long as it is still sampling them.
        self.alive.store(false, Ordering::Release);
    }
}

/// One buffer: the surface, the texture the renderer draws into it through, and
/// the send right a consumer will be handed.
fn create_buffer(device: &metal::Device) -> Result<(metal::Texture, mach_port_t)> {
    let (max_width, max_height) = embed_max_size();
    let surface = create_surface(max_width, max_height)?;
    let texture = wrap_as_texture(device, &surface, max_width, max_height)
        .context("wrapping the surface as a texture")?;
    // Made here rather than in the rendezvous thread because an `IOSurfaceRef` is
    // not `Send`, while a port name is a plain integer that is. The right is
    // copied per request, never moved, so this one stays valid for every consumer
    // that ever looks the name up.
    let port = surface.create_mach_port();
    if port == MACH_PORT_NULL {
        bail!("IOSurfaceCreateMachPort returned a null port");
    }
    Ok((texture, port))
}

/// Register a service name for these ports and start answering lookups on it.
fn serve(ports: [mach_port_t; EMBED_BUFFER_COUNT]) -> Result<(String, Arc<AtomicBool>)> {
    let service_name = format!(
        "dev.slop2.zed.embed.{}.{}",
        std::process::id(),
        NEXT_SERVICE.fetch_add(1, Ordering::Relaxed)
    );
    // The name travels in the handshake as length-prefixed text, so the only limit
    // that matters is launchd's own.
    if service_name.len() >= BOOTSTRAP_MAX_NAME_LEN as usize {
        bail!("service name {service_name} exceeds BOOTSTRAP_MAX_NAME_LEN");
    }
    let mut service_port = MACH_PORT_NULL;
    let checked_in = {
        let name =
            CString::new(service_name.clone()).context("the name contained an interior NUL")?;
        unsafe { bootstrap_check_in(bootstrap_port, name.as_ptr(), &mut service_port) }
    };
    if checked_in != KERN_SUCCESS {
        bail!("bootstrap_check_in({service_name}) failed: {checked_in}");
    }

    let alive = Arc::new(AtomicBool::new(true));
    if let Err(error) = std::thread::Builder::new()
        .name("zed-embed-iosurface".into())
        .spawn({
            let alive = alive.clone();
            move || serve_surfaces(service_port, ports, alive)
        })
    {
        // The service right is this function's to give back only while no thread
        // has taken it over; the surface ports belong to the caller either way.
        unsafe { release_rights(service_port, &[]) };
        return Err(error).context("spawning the IOSurface rendezvous thread");
    }
    Ok((service_name, alive))
}

fn create_surface(width: u32, height: u32) -> Result<CFRetained<IOSurfaceRef>> {
    let width_value = CFNumber::new_i32(width as i32);
    let height_value = CFNumber::new_i32(height as i32);
    let bytes_per_element = CFNumber::new_i32(4);
    let pixel_format = CFNumber::new_i32(PIXEL_FORMAT_BGRA);
    // No `kIOSurfaceBytesPerRow`: IOSurface picks a row alignment the GPU is
    // happy with, and both consumers of the pixels — Metal here, CoreAnimation
    // there — read the stride off the surface rather than assuming width * 4.
    let properties = CFDictionary::from_slices(
        &[
            unsafe { kIOSurfaceWidth },
            unsafe { kIOSurfaceHeight },
            unsafe { kIOSurfaceBytesPerElement },
            unsafe { kIOSurfacePixelFormat },
        ],
        &[
            &*width_value,
            &*height_value,
            &*bytes_per_element,
            &*pixel_format,
        ],
    );
    unsafe { IOSurfaceRef::new(properties.as_opaque()) }
        .with_context(|| format!("IOSurfaceCreate({width}x{height}) returned null"))
}

fn wrap_as_texture(
    device: &metal::Device,
    surface: &IOSurfaceRef,
    width: u32,
    height: u32,
) -> Result<metal::Texture> {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
    // `Private` is not a legal storage mode for an IOSurface-backed texture —
    // the whole point of the surface is that something outside this device can
    // read it. `Managed` works on both unified and discrete memory, at the cost
    // of an explicit synchronize on the latter (see `render_scene_to_target`).
    descriptor.set_storage_mode(metal::MTLStorageMode::Managed);

    // metal-rs 0.33 binds no IOSurface constructor, so the selector is sent
    // directly. Same approach as the CoreVideo textures in `metal_renderer`.
    let texture: *mut objc::runtime::Object = unsafe {
        let device: *mut objc::runtime::Object = device.as_ptr().cast();
        let descriptor: *mut objc::runtime::Object = descriptor.as_ptr().cast();
        let surface: *const std::ffi::c_void = (surface as *const IOSurfaceRef).cast();
        msg_send![device, newTextureWithDescriptor: descriptor iosurface: surface plane: 0usize]
    };
    if texture.is_null() {
        bail!("newTextureWithDescriptor:iosurface:plane: returned nil");
    }
    Ok(unsafe { metal::Texture::from_ptr(texture.cast()) })
}

/// Answer lookups for as long as the session lives, then hand every right back.
fn serve_surfaces(
    service_port: mach_port_t,
    ports: [mach_port_t; EMBED_BUFFER_COUNT],
    alive: Arc<AtomicBool>,
) {
    while alive.load(Ordering::Acquire) {
        let mut request = SurfaceRequest::default();
        // A timed receive rather than an indefinite one: a pane can close before
        // any consumer ever asks, and an indefinite `mach_msg` would park this
        // thread on a dead session for the life of the process — and this
        // process outlives individual panes by design.
        let received = unsafe {
            mach_msg(
                &mut request.header,
                MACH_RCV_MSG | MACH_RCV_TIMEOUT,
                0,
                mem::size_of::<SurfaceRequest>() as u32,
                service_port,
                RECEIVE_TIMEOUT_MS,
                MACH_PORT_NULL,
            )
        };
        if received == MACH_RCV_TIMED_OUT {
            continue;
        }
        if received != KERN_SUCCESS {
            log::error!("[zed-embed] IOSurface rendezvous receive failed: {received:#x}");
            break;
        }
        if request.header.msgh_id != SURFACE_REQUEST_ID {
            log::warn!(
                "[zed-embed] ignoring an unexpected rendezvous message (id {})",
                request.header.msgh_id
            );
            continue;
        }
        let reply_port = request.header.msgh_remote_port;
        if reply_port == MACH_PORT_NULL {
            log::warn!("[zed-embed] a rendezvous request carried no reply port");
            continue;
        }

        let mut reply = SurfaceReply {
            header: mach_msg_header_t {
                // The request's remote disposition is reused as-is: it arrived
                // as a send-once right, and replying is what consumes it.
                msgh_bits: MACH_MSGH_BITS(request.header.msgh_bits & MACH_MSGH_BITS_REMOTE_MASK, 0)
                    | MACH_MSGH_BITS_COMPLEX,
                msgh_size: mem::size_of::<SurfaceReply>() as u32,
                msgh_remote_port: reply_port,
                msgh_local_port: MACH_PORT_NULL,
                msgh_voucher_port: MACH_PORT_NULL,
                msgh_id: SURFACE_REPLY_ID,
            },
            body: mach_msg_body_t {
                msgh_descriptor_count: EMBED_BUFFER_COUNT as u32,
            },
            // Copied, not moved: a move would consume this process's only right
            // to the surface, so a consumer that looked the name up a second
            // time — after its own restart, say — would find nothing there.
            surfaces: ports
                .map(|port| mach_msg_port_descriptor_t::new(port, MACH_MSG_TYPE_COPY_SEND)),
        };
        let sent = unsafe {
            mach_msg(
                &mut reply.header,
                MACH_SEND_MSG,
                mem::size_of::<SurfaceReply>() as u32,
                0,
                MACH_PORT_NULL,
                0,
                MACH_PORT_NULL,
            )
        };
        if sent != KERN_SUCCESS {
            // Nothing to unwind: with `COPY_SEND` the rights are only copied on
            // a successful send.
            log::error!("[zed-embed] handing over the IOSurface ports failed: {sent:#x}");
            continue;
        }
        log::info!("[zed-embed] handed {EMBED_BUFFER_COUNT} IOSurface ports to a consumer");
    }
    unsafe { release_rights(service_port, &ports) };
}

/// Give back the per-surface send rights and, if it was ever checked in, the
/// service's receive right — which is what returns the name to launchd.
unsafe fn release_rights(service_port: mach_port_t, ports: &[mach_port_t]) {
    for &port in ports {
        if port == MACH_PORT_NULL {
            continue;
        }
        let result = unsafe { mach_port_deallocate(mach_task_self(), port) };
        if result != KERN_SUCCESS {
            log::error!("[zed-embed] releasing an IOSurface port failed: {result:#x}");
        }
    }
    if service_port == MACH_PORT_NULL {
        return;
    }
    // `mach_port_deallocate` would be a no-op here: it drops send rights, and
    // this is the receive right.
    let result =
        unsafe { mach_port_mod_refs(mach_task_self(), service_port, MACH_PORT_RIGHT_RECEIVE, -1) };
    if result != KERN_SUCCESS {
        log::error!("[zed-embed] releasing the rendezvous service port failed: {result:#x}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mach2::bootstrap::bootstrap_look_up;
    use mach2::mach_port::mach_port_allocate;
    use mach2::message::MACH_MSG_TYPE_MAKE_SEND_ONCE;

    /// A consumer's view of the reply. Separate from [`SurfaceReply`] so that the
    /// room a receive needs for the kernel-appended trailer cannot leak into the
    /// `msgh_size` the sending side reports.
    #[repr(C)]
    struct ReceivedReply {
        reply: SurfaceReply,
        trailer: mach_msg_trailer_t,
    }

    /// Stands in for the `client-appkit` consumer: the exchange here is the one it
    /// performs after reading an `IOSF` handshake, so the reply's message layout is
    /// checked in the repository that defines it rather than only across the wire.
    #[test]
    fn a_consumer_can_look_up_the_surfaces() -> Result<()> {
        let Some(device) = metal::Device::system_default() else {
            // The transport is unavailable without a Metal device anyway, and the
            // producer falls back to CPU frames there.
            return Ok(());
        };
        let surfaces = EmbedSurfaces::new(&device)?;

        let mut service_port = MACH_PORT_NULL;
        let name = CString::new(surfaces.service_name())?;
        let found = unsafe { bootstrap_look_up(bootstrap_port, name.as_ptr(), &mut service_port) };
        anyhow::ensure!(found == KERN_SUCCESS, "bootstrap_look_up failed: {found}");

        let mut reply_port = MACH_PORT_NULL;
        let allocated = unsafe {
            mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &mut reply_port)
        };
        anyhow::ensure!(
            allocated == KERN_SUCCESS,
            "allocating a reply port failed: {allocated:#x}"
        );

        let mut request = mach_msg_header_t {
            // A send-once right to the reply port, which the producer consumes by
            // replying to it exactly once.
            msgh_bits: MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE),
            msgh_size: mem::size_of::<mach_msg_header_t>() as u32,
            msgh_remote_port: service_port,
            msgh_local_port: reply_port,
            msgh_voucher_port: MACH_PORT_NULL,
            msgh_id: SURFACE_REQUEST_ID,
        };
        let sent = unsafe {
            mach_msg(
                &mut request,
                MACH_SEND_MSG,
                mem::size_of::<mach_msg_header_t>() as u32,
                0,
                MACH_PORT_NULL,
                0,
                MACH_PORT_NULL,
            )
        };
        anyhow::ensure!(
            sent == KERN_SUCCESS,
            "sending the request failed: {sent:#x}"
        );

        let mut received: ReceivedReply = unsafe { mem::zeroed() };
        // Generously longer than the rendezvous thread's own receive timeout: the
        // request can land just after that thread has started waiting, so a reply
        // may be a whole timeout away.
        let result = unsafe {
            mach_msg(
                &mut received.reply.header,
                MACH_RCV_MSG | MACH_RCV_TIMEOUT,
                0,
                mem::size_of::<ReceivedReply>() as u32,
                reply_port,
                RECEIVE_TIMEOUT_MS * 4,
                MACH_PORT_NULL,
            )
        };
        anyhow::ensure!(
            result == KERN_SUCCESS,
            "receiving the reply failed: {result:#x}"
        );
        anyhow::ensure!(
            received.reply.header.msgh_id == SURFACE_REPLY_ID,
            "unexpected reply id {}",
            received.reply.header.msgh_id
        );
        anyhow::ensure!(
            received.reply.header.msgh_bits & MACH_MSGH_BITS_COMPLEX != 0,
            "the reply carried no port descriptors"
        );
        anyhow::ensure!(
            received.reply.body.msgh_descriptor_count == EMBED_BUFFER_COUNT as u32,
            "expected {EMBED_BUFFER_COUNT} descriptors, got {}",
            received.reply.body.msgh_descriptor_count
        );

        let (max_width, max_height) = embed_max_size();
        for (index, descriptor) in received.reply.surfaces.iter().enumerate() {
            let surface = IOSurfaceRef::lookup_from_mach_port(descriptor.name)
                .with_context(|| format!("looking up the surface behind buffer {index}"))?;
            anyhow::ensure!(
                surface.width() == max_width as usize && surface.height() == max_height as usize,
                "buffer {index} is {}x{}, not {max_width}x{max_height}",
                surface.width(),
                surface.height()
            );
        }

        let ports: Vec<mach_port_t> = received
            .reply
            .surfaces
            .iter()
            .map(|descriptor| descriptor.name)
            .chain([service_port])
            .collect();
        unsafe { release_rights(reply_port, &ports) };
        Ok(())
    }
}
