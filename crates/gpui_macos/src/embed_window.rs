//! The window an embed session renders into.
//!
//! Structurally the counterpart to `gpui_linux`'s `HeadlessWindow` and
//! `gpui_windows`'s `EmbedWindow`: a `PlatformWindow` with no `NSWindow` behind
//! it, keeping Zed's own Metal renderer and pointing it at an offscreen texture
//! instead of a `CAMetalLayer`.
//!
//! There is no OS focus or pointer here — the consumer forwards input over the
//! socket — so `is_active` and `is_hovered` report true unconditionally. Without
//! that GPUI routes neither typed text nor cursor-style changes to the embed.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use gpui::{
    Bounds, Capslock, DevicePixels, DispatchEventResult, DisplayId, GpuSpecs, Modifiers, Pixels,
    PlatformAtlas, PlatformDisplay, PlatformInput, PlatformInputHandler, PlatformWindow, Point,
    PromptButton, PromptLevel, RequestFrameOptions, Scene, Size, WindowAppearance,
    WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowParams, px, size,
};
use uuid::Uuid;

use crate::iosurface::{EMBED_BUFFER_COUNT, EmbedSurfaces, embed_max_size};
use crate::metal_renderer::MetalRenderer;
use crate::renderer;

#[derive(Debug)]
pub(crate) struct EmbedDisplay {
    bounds: Bounds<Pixels>,
}

impl EmbedDisplay {
    pub(crate) fn new() -> Self {
        Self {
            bounds: Bounds::from_corners(Point::default(), Point::new(px(1920.), px(1080.))),
        }
    }
}

impl PlatformDisplay for EmbedDisplay {
    fn id(&self) -> DisplayId {
        DisplayId::new(0)
    }

    fn uuid(&self) -> Result<Uuid> {
        // Stable identity: there is exactly one embed display.
        Ok(Uuid::nil())
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}

/// A `Managed` BGRA target the CPU can read back from.
///
/// `Managed` rather than `Private` because the mode-0 transport copies the pixels
/// out with `get_bytes`, which a private texture cannot serve.
fn new_render_target(device: &metal::Device, width: u32, height: u32) -> metal::Texture {
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
    descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
    device.new_texture(&descriptor)
}

/// Where finished frames land, which is the whole difference between the two
/// transport modes.
enum EmbedTarget {
    /// Mode 0. One content-sized texture, read back on the render thread and
    /// copied over the socket as a `FRAM` message.
    Readback {
        /// Single rather than double buffered: the readback finishes before
        /// `draw` returns, so nothing else is sampling this texture when the next
        /// frame overwrites it.
        target: metal::Texture,
        /// Reused across frames. A fresh allocation per frame is several
        /// megabytes at a full-window size, sixty times a second.
        readback: Vec<u8>,
    },
    /// Mode 1. Two IOSurface-backed textures the consumer samples in place; a
    /// finished frame is announced with `RDY!` and never copied.
    Surfaces {
        surfaces: EmbedSurfaces,
        /// Alternated per frame so a frame is never rendered into the surface the
        /// consumer is still compositing.
        next_buffer: usize,
    },
}

/// What the consumer needs in order to find the surfaces: the `IOSF` handshake's
/// payload.
pub(crate) struct ZeroCopyHandshake {
    pub(crate) service_name: String,
    pub(crate) buffer_count: u32,
    pub(crate) max_width: u32,
    pub(crate) max_height: u32,
}

struct EmbedWindowState {
    bounds: Bounds<Pixels>,
    display: Rc<dyn PlatformDisplay>,
    input_handler: Option<PlatformInputHandler>,
    title: Option<String>,
    is_fullscreen: bool,

    renderer: MetalRenderer,
    target: EmbedTarget,
    /// Force the next frame even if GPUI considers the window clean.
    ///
    /// The tick used to pass `force_render: true` unconditionally, which meant a
    /// full scene rebuild + rasterization on EVERY tick — and the tick runs off a
    /// fixed clock, so an idle embed re-rendered forever while a native window
    /// idles at zero.
    ///
    /// Dirty-tracking cannot cover the cases where the OUTPUT changed without the
    /// scene changing: a consumer that just connected has no pixels yet, and a
    /// resize or scale change reallocates the target. Those set this.
    force_frame: bool,
    /// Device-pixel size of the target, which is what the consumer sends. Kept so
    /// a scale change can recompute the logical size without waiting for the next
    /// resize.
    pixels: (u32, u32),
    /// Reported by `scale_factor`. The consumer's display scale times the user's
    /// editor multiplier -- 1.0 until it says otherwise.
    scale: f32,
    /// Whether the frames handed to the consumer carry a background of their own.
    ///
    /// A native window would ask the compositor for translucency here. There is no
    /// compositor on this path — the frame IS the answer — so `Transparent` means
    /// the renderer clears to transparent black and every pixel the scene did not
    /// cover leaves the consumer's own background showing through its embed pane.
    /// `Opaque` clears to opaque black, which is what every embed did before the
    /// host had a way to ask for anything else.
    background: WindowBackgroundAppearance,

    request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    input: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    resize: Option<Box<dyn FnMut(Size<Pixels>, f32)>>,
    close: Option<Box<dyn FnOnce()>>,
    active_status_change: Option<Box<dyn FnMut(bool)>>,
    activated: bool,
    hover_status_change: Option<Box<dyn FnMut(bool)>>,
    title_change: Option<Box<dyn FnMut(&str)>>,
    /// Fired with each finished frame's pixels and their dimensions. Mode 0 only.
    on_frame: Option<Box<dyn FnMut(&[u8], u32, u32)>>,
    /// Fired with a finished frame's buffer index and active dimensions, for
    /// `RDY!`. Mode 1 only.
    ///
    /// `Arc` rather than a boxed `FnMut` like the others because this one is
    /// invoked from Metal's completion handler, which runs on a driver thread: the
    /// callback has to be shared with a `'static + Send` block instead of borrowed
    /// out of the state for the duration of the call.
    on_ready: Option<Arc<dyn Fn(u32, u32, u32) + Send + Sync>>,
}

#[derive(Clone)]
pub(crate) struct EmbedWindow(Rc<RefCell<EmbedWindowState>>);

/// Weak handle so the frame clock can hold the window without keeping it alive:
/// once GPUI drops the window, `upgrade` fails and the session tears down.
#[derive(Clone)]
pub(crate) struct WeakEmbedWindow(std::rc::Weak<RefCell<EmbedWindowState>>);

impl WeakEmbedWindow {
    pub(crate) fn upgrade(&self) -> Option<EmbedWindow> {
        self.0.upgrade().map(EmbedWindow)
    }
}

impl raw_window_handle::HasWindowHandle for EmbedWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        // Embed windows are not backed by an NSWindow.
        Err(raw_window_handle::HandleError::NotSupported)
    }
}

impl raw_window_handle::HasDisplayHandle for EmbedWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::NotSupported)
    }
}

impl EmbedWindow {
    pub(crate) fn new(
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        instance_buffer_pool: renderer::Context,
    ) -> Self {
        let width = f32::from(params.bounds.size.width).max(1.0) as u32;
        let height = f32::from(params.bounds.size.height).max(1.0) as u32;
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        let target = new_render_target(renderer.device(), width, height);

        Self(Rc::new(RefCell::new(EmbedWindowState {
            bounds: params.bounds,
            display,
            input_handler: None,
            title: None,
            is_fullscreen: false,
            renderer,
            target: EmbedTarget::Readback {
                target,
                readback: Vec::new(),
            },
            // The first frame always happens: a fresh consumer has nothing.
            force_frame: true,
            pixels: (width, height),
            scale: 1.0,
            // Matches `MetalRenderer::new_headless`, which starts opaque.
            background: WindowBackgroundAppearance::Opaque,
            request_frame: None,
            input: None,
            resize: None,
            close: None,
            active_status_change: None,
            activated: false,
            hover_status_change: None,
            title_change: None,
            on_frame: None,
            on_ready: None,
        })))
    }

    pub(crate) fn downgrade(&self) -> WeakEmbedWindow {
        WeakEmbedWindow(Rc::downgrade(&self.0))
    }

    pub(crate) fn set_on_frame(&self, callback: Box<dyn FnMut(&[u8], u32, u32)>) {
        self.0.borrow_mut().on_frame = Some(callback);
    }

    pub(crate) fn set_on_ready(&self, callback: Arc<dyn Fn(u32, u32, u32) + Send + Sync>) {
        self.0.borrow_mut().on_ready = Some(callback);
    }

    pub(crate) fn set_on_title_change(&self, callback: Box<dyn FnMut(&str)>) {
        self.0.borrow_mut().title_change = Some(callback);
    }

    /// Move this window onto IOSurfaces the consumer can sample directly, and
    /// return what it needs to find them.
    ///
    /// Fallible on purpose, and called before the mode byte goes out: everything
    /// this touches — surface allocation, the Metal wrapping, the bootstrap
    /// registration — can fail on a machine where the CPU path still works fine,
    /// and the caller then stays on mode 0 rather than serving a session with no
    /// pixels in it.
    pub(crate) fn enable_zero_copy(&self) -> Result<ZeroCopyHandshake> {
        let mut state = self.0.borrow_mut();
        let surfaces = EmbedSurfaces::new(state.renderer.device())?;
        let (max_width, max_height) = embed_max_size();
        let handshake = ZeroCopyHandshake {
            service_name: surfaces.service_name().to_owned(),
            buffer_count: EMBED_BUFFER_COUNT as u32,
            max_width,
            max_height,
        };
        state.target = EmbedTarget::Surfaces {
            surfaces,
            next_buffer: 0,
        };
        state.force_frame = true;

        // The size the window was created at may already be past what the
        // surfaces can hold. Clamping here rather than waiting for the consumer's
        // first `Resize` keeps the very first frame inside the buffer; no resize
        // callback fires because this runs before gpui has opened the window.
        let (width, height) = state.pixels;
        let clamped = (width.min(max_width), height.min(max_height));
        if clamped != (width, height) {
            state.pixels = clamped;
            state.bounds.size = Size {
                width: px(clamped.0 as f32 / state.scale),
                height: px(clamped.1 as f32 / state.scale),
            };
        }
        Ok(handshake)
    }

    /// Ask GPUI to produce and present a frame. Driven by the session's clock.
    pub(crate) fn tick_frame(&self) {
        // Consume the one-shot force. Otherwise GPUI's invalidator decides, which
        // is what a native window does: no damage, no frame.
        let forced = {
            let mut state = self.0.borrow_mut();
            std::mem::replace(&mut state.force_frame, false)
        };
        let mut callback = self.0.borrow_mut().request_frame.take();
        if let Some(callback) = callback.as_mut() {
            callback(RequestFrameOptions {
                require_presentation: forced,
                force_render: forced,
            });
        }
        self.0.borrow_mut().request_frame = callback;
    }

    /// Match the consumer's display area.
    ///
    /// In mode 0 the target is reallocated at the new size, because the frame
    /// message carries the target's own dimensions; anything larger would arrive
    /// as content in the corner of an oversized image. Mode 1's surfaces are
    /// allocated once at the maximum and each frame renders into their top-left,
    /// so a resize there costs nothing but a clamp — which is also why it needs no
    /// second handshake, unlike the Windows producer's re-shared textures.
    pub(crate) fn resize_embed(&self, width: u32, height: u32) {
        let (width, height) = {
            let state = self.0.borrow();
            match &state.target {
                EmbedTarget::Readback { .. } => (width.max(1), height.max(1)),
                EmbedTarget::Surfaces { .. } => {
                    let (max_width, max_height) = embed_max_size();
                    (width.clamp(1, max_width), height.clamp(1, max_height))
                }
            }
        };
        // The consumer sends device pixels; the target is that size, while gpui
        // lays out against the logical extent.
        let scale = self.0.borrow().scale;
        let size = Size {
            width: px(width as f32 / scale),
            height: px(height as f32 / scale),
        };

        log::info!("[zed-embed] resize requested: {width}x{height} at scale {scale}");
        {
            let mut state = self.0.borrow_mut();
            if state.bounds.size == size {
                return;
            }
            state.bounds.size = size;
            state.pixels = (width, height);
            let state = &mut *state;
            if let EmbedTarget::Readback { target, .. } = &mut state.target {
                *target = new_render_target(state.renderer.device(), width, height);
            }
            // A freshly allocated target is blank, and the scene may be identical
            // to the last one.
            state.force_frame = true;
            log::info!("[zed-embed] resized to {width}x{height}");
        }

        let mut callback = self.0.borrow_mut().resize.take();
        if let Some(callback) = callback.as_mut() {
            callback(size, scale);
        }
        self.0.borrow_mut().resize = callback;
    }

    /// The scale the consumer reported. Pointer coordinates arrive in device
    /// pixels like sizes do, so they need dividing by this before they mean
    /// anything to gpui.
    pub(crate) fn scale(&self) -> f32 {
        self.0.borrow().scale
    }

    /// Adopt a new scale. The target keeps its pixel size -- only the logical
    /// extent gpui lays out against changes -- so nothing is reallocated.
    pub(crate) fn set_scale(&self, scale: f32) {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale.clamp(0.25, 8.0)
        } else {
            return;
        };
        let (size, changed) = {
            let mut state = self.0.borrow_mut();
            if (state.scale - scale).abs() < f32::EPSILON {
                (state.bounds.size, false)
            } else {
                state.scale = scale;
                // A scale change relays out the same content: the scene is
                // "unchanged" as far as dirty-tracking goes, but every pixel is.
                state.force_frame = true;
                let (width, height) = state.pixels;
                let size = Size {
                    width: px(width as f32 / scale),
                    height: px(height as f32 / scale),
                };
                state.bounds.size = size;
                (size, true)
            }
        };
        if !changed {
            return;
        }
        log::info!("[zed-embed] scale set to {scale}");
        let mut callback = self.0.borrow_mut().resize.take();
        if let Some(callback) = callback.as_mut() {
            callback(size, scale);
        }
        self.0.borrow_mut().resize = callback;
    }

    /// Route input forwarded by the consumer into GPUI.
    pub(crate) fn dispatch_input(&self, event: PlatformInput) {
        let mut callback = self.0.borrow_mut().input.take();
        if let Some(callback) = callback.as_mut() {
            callback(event);
        }
        self.0.borrow_mut().input = callback;
    }

    /// Mark the window active and hovered once. GPUI uses activation to focus
    /// the content and route typed text to the input handler — without it only
    /// mouse events land, and typing goes nowhere.
    ///
    /// Deliberately does not latch until the callbacks exist: this runs on a
    /// clock that can tick before GPUI has registered them, and latching then
    /// would leave the window permanently unfocused.
    pub(crate) fn ensure_active(&self) {
        let (mut active, mut hover) = {
            let mut state = self.0.borrow_mut();
            if state.activated {
                return;
            }
            if state.active_status_change.is_none() && state.hover_status_change.is_none() {
                return;
            }
            state.activated = true;
            // A consumer just became live; it has nothing to sample.
            state.force_frame = true;
            (
                state.active_status_change.take(),
                state.hover_status_change.take(),
            )
        };
        if let Some(callback) = active.as_mut() {
            callback(true);
        }
        if let Some(callback) = hover.as_mut() {
            callback(true);
        }
        log::info!("[zed-embed] window activated");
        let mut state = self.0.borrow_mut();
        state.active_status_change = active;
        state.hover_status_change = hover;
    }

    /// Tell GPUI the window is gone, so the session can end when the consumer
    /// disconnects.
    pub(crate) fn close(&self) {
        let callback = self.0.borrow_mut().close.take();
        if let Some(callback) = callback {
            callback();
        }
    }
}

impl PlatformWindow for EmbedWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().bounds
    }

    fn is_maximized(&self) -> bool {
        false
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.0.borrow().bounds)
    }

    fn content_size(&self) -> Size<Pixels> {
        self.0.borrow().bounds.size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        self.0.borrow_mut().bounds.size = size;
    }

    fn scale_factor(&self) -> f32 {
        // Sizes arrive in device pixels; this is what gpui divides them by to lay
        // out. Left at 1.0 the editor draws at 100% on a scaled display, which is
        // physically small next to the host's own chrome.
        self.0.borrow().scale
    }

    fn appearance(&self) -> WindowAppearance {
        WindowAppearance::Dark
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.0.borrow().display.clone())
    }

    fn mouse_position(&self) -> Point<Pixels> {
        Point::default()
    }

    fn modifiers(&self) -> Modifiers {
        Modifiers::default()
    }

    fn capslock(&self) -> Capslock {
        Capslock::default()
    }

    fn set_input_handler(&mut self, input_handler: PlatformInputHandler) {
        self.0.borrow_mut().input_handler = Some(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        self.0.borrow_mut().input_handler.take()
    }

    fn prompt(
        &self,
        _level: PromptLevel,
        _msg: &str,
        _detail: Option<&str>,
        _answers: &[PromptButton],
    ) -> Option<futures::channel::oneshot::Receiver<usize>> {
        // Fall back to GPUI's rendered prompts: the native ones are NSAlerts,
        // which would open a real window outside the host's pane.
        None
    }

    fn activate(&self) {}

    fn is_active(&self) -> bool {
        true
    }

    fn is_hovered(&self) -> bool {
        true
    }

    fn background_appearance(&self) -> WindowBackgroundAppearance {
        self.0.borrow().background
    }

    fn set_title(&mut self, title: &str) {
        self.0.borrow_mut().title = Some(title.to_owned());
        let mut callback = self.0.borrow_mut().title_change.take();
        if let Some(callback) = callback.as_mut() {
            callback(title);
        }
        self.0.borrow_mut().title_change = callback;
    }

    fn get_title(&self) -> String {
        self.0.borrow().title.clone().unwrap_or_default()
    }

    fn set_background_appearance(&self, background: WindowBackgroundAppearance) {
        let mut state = self.0.borrow_mut();
        if state.background == background {
            return;
        }
        state.background = background;
        // `Blurred` has no meaning without a compositor to blur; treat anything
        // that isn't opaque as "clear to transparent black and let the host decide
        // what is behind us".
        let transparent = background != WindowBackgroundAppearance::Opaque;
        state.renderer.update_transparency(transparent);
        // The scene has not changed, only what the untouched pixels will hold, so
        // dirty-tracking would skip the repaint that makes this visible.
        state.force_frame = true;
    }

    fn minimize(&self) {}

    fn zoom(&self) {}

    fn toggle_fullscreen(&self) {
        let mut state = self.0.borrow_mut();
        state.is_fullscreen = !state.is_fullscreen;
    }

    fn is_fullscreen(&self) -> bool {
        self.0.borrow().is_fullscreen
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.0.borrow_mut().request_frame = Some(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.0.borrow_mut().input = Some(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.0.borrow_mut().active_status_change = Some(callback);
    }

    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.0.borrow_mut().hover_status_change = Some(callback);
    }

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        self.0.borrow_mut().resize = Some(callback);
    }

    fn on_moved(&self, _callback: Box<dyn FnMut()>) {}

    fn on_should_close(&self, _callback: Box<dyn FnMut() -> bool>) {}

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        self.0.borrow_mut().close = Some(callback);
    }

    fn on_hit_test_window_control(&self, _callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
    }

    fn on_appearance_changed(&self, _callback: Box<dyn FnMut()>) {}

    fn draw(&self, scene: &Scene) {
        let mut state = self.0.borrow_mut();
        let state = &mut *state;

        let (width, height) = state.pixels;
        let render_size = size(DevicePixels(width as i32), DevicePixels(height as i32));

        match &mut state.target {
            EmbedTarget::Readback { target, readback } => {
                // Waiting for completion, unlike a windowed present: the pixels
                // are read back on this thread immediately below, so there is
                // nothing to overlap the GPU work with.
                if let Err(error) =
                    state
                        .renderer
                        .render_scene_to_target(scene, render_size, target, true, None)
                {
                    log::error!("[zed-embed] rendering a frame: {error:#}");
                    return;
                }

                let bytes_per_row = width as usize * 4;
                readback.resize(bytes_per_row * height as usize, 0);
                let region = metal::MTLRegion {
                    origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: metal::MTLSize {
                        width: width as u64,
                        height: height as u64,
                        depth: 1,
                    },
                };
                target.get_bytes(
                    readback.as_mut_ptr() as *mut std::ffi::c_void,
                    bytes_per_row as u64,
                    region,
                    0,
                );
                // `FRAM` is RGBA on the wire — the Linux producer reads back an
                // `Rgba8Unorm` offscreen texture and hands the bytes over
                // untouched — so a BGRA render target needs the same swap
                // `render_scene_to_image` does. The zero-copy path below is the
                // opposite case: there the surface stays BGRA, because that is
                // what a `CALayer` samples natively.
                for pixel in readback.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }

                if let Some(callback) = state.on_frame.as_mut() {
                    callback(readback, width, height);
                }
            }
            EmbedTarget::Surfaces {
                surfaces,
                next_buffer,
            } => {
                let index = *next_buffer;
                let Some(target) = surfaces.texture(index) else {
                    log::error!("[zed-embed] no IOSurface backs buffer {index}");
                    return;
                };
                *next_buffer = (index + 1) % EMBED_BUFFER_COUNT;

                // `RDY!` rides the GPU's completion handler instead of being sent
                // after this returns: `commit` only submits the work, so telling
                // the consumer to sample here would race the frame it is being
                // told about — which is what tearing and flashes to white on the
                // other platforms looked like.
                let on_complete = state.on_ready.clone().map(|on_ready| {
                    Box::new(move || on_ready(index as u32, width, height))
                        as Box<dyn FnOnce() + Send>
                });
                if let Err(error) = state.renderer.render_scene_to_target(
                    scene,
                    render_size,
                    target,
                    false,
                    on_complete,
                ) {
                    log::error!("[zed-embed] rendering a frame: {error:#}");
                }
            }
        }
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.0.borrow().renderer.sprite_atlas().clone()
    }

    /// No native window here, so nothing else will deliver typed characters.
    fn commits_key_char(&self) -> bool {
        true
    }

    fn is_subpixel_rendering_supported(&self) -> bool {
        // The consumer composites these pixels over its own background, so
        // subpixel coverage computed against an assumed opaque backdrop would
        // fringe.
        false
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        // Matching `MacWindow`, which reports none either: Metal exposes no
        // driver/device triple in the shape `GpuSpecs` wants.
        None
    }
}
