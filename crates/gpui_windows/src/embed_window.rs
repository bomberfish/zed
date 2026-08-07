//! The window an embed session renders into.
//!
//! Structurally the counterpart to `gpui_linux`'s `HeadlessWindow`, with one
//! deliberate difference: it keeps Zed's own D3D11 renderer rather than
//! swapping in a second GPU stack. The renderer draws into a texture shared
//! with the consumer process (see [`crate::shared_texture`]) instead of a swap
//! chain, so the pixels are handed over without a copy.
//!
//! There is no OS focus or pointer here — the consumer forwards input over the
//! pipe — so `is_active` and `is_hovered` report true unconditionally. Without
//! that GPUI routes neither typed text nor cursor-style changes to the embed.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use gpui::{
    Bounds, Capslock, DispatchEventResult, DisplayId, GpuSpecs, Modifiers, Pixels, PlatformAtlas,
    PlatformDisplay, PlatformInput, PlatformInputHandler, PlatformWindow, Point, PromptButton,
    PromptLevel, RequestFrameOptions, Scene, Size, WindowAppearance, WindowBackgroundAppearance,
    WindowBounds, WindowControlArea, WindowParams, px,
};
use gpui_util::ResultExt;
use windows::Win32::Foundation::HANDLE;
use uuid::Uuid;

use crate::directx_devices::DirectXDevices;
use crate::directx_renderer::DirectXRenderer;
use crate::shared_texture::{SharedTexture, SharedTextureDevices};

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

struct EmbedWindowState {
    bounds: Bounds<Pixels>,
    display: Rc<dyn PlatformDisplay>,
    input_handler: Option<PlatformInputHandler>,
    title: Option<String>,
    is_fullscreen: bool,

    devices: Rc<SharedTextureDevices>,
    renderer: DirectXRenderer,
    /// Two targets, alternated per frame, sized to the content rather than
    /// over-allocated: the consumer's importer takes the texture's dimensions
    /// from the resource itself, so anything larger would display as content in
    /// the corner of an oversized texture.
    ///
    /// Two and not one because the consumer samples asynchronously. With a single
    /// target the next frame's writes land in the very texture GTK is reading,
    /// which shows up as horizontal tearing; the fence only orders the announced
    /// frame's writes ahead of the read, it cannot hold back the frame after it.
    /// Mirrors the Linux path, which allocates two dma-buf planes for the same
    /// reason and sends the plane index per frame.
    shared: [SharedTexture; 2],
    /// Which of `shared` the next frame draws into.
    next_buffer: usize,
    /// Force the next frame even if GPUI considers the window clean.
    ///
    /// The tick used to pass `force_render: true` unconditionally, which meant a
    /// full scene rebuild + rasterization on EVERY tick — and the tick runs off
    /// the vsync thread, so an idle embed re-rendered at display refresh rate
    /// forever while a native window idles at zero. That is what made the embed
    /// slower than native despite the zero-copy handoff; the copy was never the
    /// cost.
    ///
    /// Dirty-tracking cannot cover the cases where the OUTPUT changed without the
    /// scene changing: a consumer that just connected has no pixels yet, and a
    /// resize or scale change reallocates the targets. Those set this.
    force_frame: bool,
    /// The consumer, kept so a resize can duplicate the new handle to it.
    consumer_process: HANDLE,
    /// Device-pixel size of the shared target, which is what the consumer sends.
    /// Kept so a scale change can recompute the logical size without waiting for
    /// the next resize.
    pixels: (u32, u32),
    /// Reported by `scale_factor`. The consumer's display scale times the user's
    /// editor multiplier -- 1.0 until it says otherwise.
    scale: f32,
    /// Sends a fresh handshake after the targets are reallocated: both texture
    /// handles, then the size.
    reshare: Option<Box<dyn FnMut([isize; 2], u32, u32)>>,

    request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    input: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    resize: Option<Box<dyn FnMut(Size<Pixels>, f32)>>,
    close: Option<Box<dyn FnOnce()>>,
    active_status_change: Option<Box<dyn FnMut(bool)>>,
    activated: bool,
    hover_status_change: Option<Box<dyn FnMut(bool)>>,
    title_change: Option<Box<dyn FnMut(&str)>>,
    /// Fired after each frame reaches a shared texture: which buffer, the active
    /// width/height (the sub-rect of it to sample), and the fence value that
    /// frame's writes complete at.
    on_frame: Option<Box<dyn FnMut(usize, u32, u32, u64)>>,
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
        // Embed windows are not backed by an HWND.
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
        devices: Rc<SharedTextureDevices>,
        directx_devices: &DirectXDevices,
        consumer_process: HANDLE,
    ) -> Result<Self> {
        let width = f32::from(params.bounds.size.width).max(1.0) as u32;
        let height = f32::from(params.bounds.size.height).max(1.0) as u32;
        let shared = [
            SharedTexture::new(&devices, width, height)
                .context("Creating the first shared embed texture")?,
            SharedTexture::new(&devices, width, height)
                .context("Creating the second shared embed texture")?,
        ];
        // Bound to buffer 0 here; `draw` rebinds per frame as it alternates.
        let renderer = DirectXRenderer::new_offscreen(
            directx_devices,
            shared[0].render_target_texture(),
            width,
            height,
        )
        .context("Creating the offscreen renderer")?;

        Ok(Self(Rc::new(RefCell::new(EmbedWindowState {
            bounds: params.bounds,
            display,
            input_handler: None,
            title: None,
            is_fullscreen: false,
            devices,
            renderer,
            shared,
            next_buffer: 0,
            // The first frame always happens: a fresh consumer has nothing.
            force_frame: true,
            consumer_process,
            pixels: (width, height),
            scale: 1.0,
            reshare: None,
            request_frame: None,
            input: None,
            resize: None,
            close: None,
            active_status_change: None,
            activated: false,
            hover_status_change: None,
            title_change: None,
            on_frame: None,
        }))))
    }

    pub(crate) fn downgrade(&self) -> WeakEmbedWindow {
        WeakEmbedWindow(Rc::downgrade(&self.0))
    }

    /// Both shared handles duplicated into the consumer, plus their dimensions.
    pub(crate) fn share(&self) -> Result<([isize; 2], u32, u32)> {
        let state = self.0.borrow();
        let handles = [
            state.shared[0].share_with(state.consumer_process)?,
            state.shared[1].share_with(state.consumer_process)?,
        ];
        Ok((handles, state.shared[0].width, state.shared[0].height))
    }

    /// The frame-completion fence, duplicated into the consumer. Sent once, with
    /// the handshake: the fence outlives target reallocation.
    pub(crate) fn share_fence(&self) -> Result<isize> {
        let state = self.0.borrow();
        state.devices.share_fence_with(state.consumer_process)
    }

    /// Called after a resize reallocates the shared targets, with the new handles.
    pub(crate) fn set_on_reshare(&self, callback: Box<dyn FnMut([isize; 2], u32, u32)>) {
        self.0.borrow_mut().reshare = Some(callback);
    }

    pub(crate) fn set_on_frame(&self, callback: Box<dyn FnMut(usize, u32, u32, u64)>) {
        self.0.borrow_mut().on_frame = Some(callback);
    }

    pub(crate) fn set_on_title_change(&self, callback: Box<dyn FnMut(&str)>) {
        self.0.borrow_mut().title_change = Some(callback);
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

    /// Make the next tick produce a frame even if nothing in the scene changed.
    pub(crate) fn force_next_frame(&self) {
        self.0.borrow_mut().force_frame = true;
    }

    /// Match the consumer's display area.
    ///
    /// The shared target is reallocated at the new size and re-shared, because
    /// the consumer derives the texture's dimensions from the resource. Frames
    /// drawn into an oversized target would appear as a small image in its
    /// corner rather than filling the pane.
    pub(crate) fn resize_embed(&self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        // The consumer sends device pixels; the shared target is that size, while
        // gpui lays out against the logical extent.
        let scale = self.0.borrow().scale;
        let size = Size {
            width: px(width as f32 / scale),
            height: px(height as f32 / scale),
        };

        log::info!("[zed-embed] resize requested: {width}x{height} at scale {scale}");
        let reshared = {
            let mut state = self.0.borrow_mut();
            if state.bounds.size == size {
                return;
            }
            state.bounds.size = size;
            state.pixels = (width, height);

            let state = &mut *state;
            // Both targets, since `draw` alternates between them; a stale one at
            // the old size would show up every other frame.
            let allocated = (|| -> Result<[SharedTexture; 2]> {
                Ok([
                    SharedTexture::new(&state.devices, width, height)?,
                    SharedTexture::new(&state.devices, width, height)?,
                ])
            })();
            match allocated {
                Ok(shared) => {
                    match state
                        .renderer
                        .set_offscreen_target(shared[0].render_target_texture(), width, height)
                    {
                        Ok(()) => {
                            state.next_buffer = 0;
                            // Freshly allocated targets are blank, and the scene
                            // may be identical to the last one.
                            state.force_frame = true;
                            // Null consumer: a detached window, so there is no
                            // process to duplicate the new handles into and no
                            // handshake to send. Still reallocated and
                            // retargeted, so it keeps rendering correctly.
                            let handles = if state.consumer_process.is_invalid() {
                                Ok([0, 0])
                            } else {
                                shared[0]
                                    .share_with(state.consumer_process)
                                    .and_then(|first| {
                                        shared[1]
                                            .share_with(state.consumer_process)
                                            .map(|second| [first, second])
                                    })
                            };
                            // The old targets cannot simply be dropped here. Their
                            // D3D11 side is an 11on12 *wrapped* resource, and
                            // releasing one while the immediate context still has
                            // queued work referencing it faults inside the interop
                            // layer -- an access violation, with no Rust panic to
                            // show for it. Retargeting unbound the old view, so
                            // flushing is enough to know the device is done.
                            unsafe { state.devices.device_context.Flush() };
                            let previous = std::mem::replace(&mut state.shared, shared);
                            drop(previous);
                            log::info!("[zed-embed] resized to {width}x{height}");
                            match handles {
                                Ok(handles) => Some((handles, width, height)),
                                Err(error) => {
                                    log::error!(
                                        "[zed-embed] re-sharing the resized targets: {error:#}"
                                    );
                                    None
                                }
                            }
                        }
                        Err(error) => {
                            log::error!("[zed-embed] retargeting the renderer: {error:#}");
                            None
                        }
                    }
                }
                Err(error) => {
                    log::error!("[zed-embed] reallocating the shared targets: {error:#}");
                    None
                }
            }
        };

        if let Some((handles, width, height)) = reshared {
            let mut callback = self.0.borrow_mut().reshare.take();
            if let Some(callback) = callback.as_mut() {
                callback(handles, width, height);
            }
            self.0.borrow_mut().reshare = callback;
        }

        let mut callback = self.0.borrow_mut().resize.take();
        if let Some(callback) = callback.as_mut() {
            // Scale factor is 1.0 for embed windows (see `scale_factor`).
            callback(size, 1.0);
        }
        self.0.borrow_mut().resize = callback;
    }

    /// The scale the consumer reported. Pointer coordinates arrive in device
    /// pixels like sizes do, so they need dividing by this before they mean
    /// anything to gpui.
    pub(crate) fn scale(&self) -> f32 {
        self.0.borrow().scale
    }

    /// Adopt a new scale. The shared target keeps its pixel size -- only the
    /// logical extent gpui lays out against changes -- so nothing is reallocated
    /// and the consumer needs no new handle.
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
        // Fall back to GPUI's rendered prompts.
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
        WindowBackgroundAppearance::Opaque
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

    fn set_background_appearance(&self, _background: WindowBackgroundAppearance) {}

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

        // Alternate targets so this frame never lands in the texture the
        // consumer is still sampling. Rebinding is view-only (`bind_offscreen_target`),
        // because the size has not changed.
        let index = state.next_buffer;
        if let Err(error) = state
            .renderer
            .bind_offscreen_target(state.shared[index].render_target_texture())
        {
            log::error!("[zed-embed] binding shared buffer {index}: {error:#}");
            return;
        }
        state.next_buffer = 1 - index;

        // Acquire/release bracket every frame: releasing is what flushes the
        // 11on12 layer, and therefore what makes these pixels visible to the
        // consumer's D3D12 device.
        state.shared[index].acquire(&state.devices);
        state
            .renderer
            .draw(scene, WindowBackgroundAppearance::Opaque)
            .log_err();
        state.shared[index].release(&state.devices);

        // Flush submits, it does not complete: without a value for the consumer
        // to wait on, GTK samples whatever the GPU happens to have written so far.
        let fence_value = match state.devices.signal_frame() {
            Ok(value) => value,
            Err(error) => {
                log::error!("[zed-embed] signalling the frame fence: {error:#}");
                return;
            }
        };

        // Pixels, not logical: the consumer samples a sub-rect of the shared
        // texture, which is sized in device pixels.
        let (width, height) = state.pixels;
        if let Some(callback) = state.on_frame.as_mut() {
            callback(index, width, height, fence_value);
        }
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.0.borrow().renderer.sprite_atlas()
    }

    /// No native window here, so nothing else will deliver typed characters.
    fn commits_key_char(&self) -> bool {
        true
    }

    fn is_subpixel_rendering_supported(&self) -> bool {
        // The consumer composites the shared texture over its own background,
        // so subpixel coverage computed against an assumed opaque backdrop
        // would fringe.
        false
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        self.0.borrow().renderer.gpu_specs().log_err()
    }

    /// Null: an embed window is not backed by an HWND. Callers use this to talk
    /// to the window manager, which has nothing to say about a texture living in
    /// another process.
    fn get_raw_handle(&self) -> windows::Win32::Foundation::HWND {
        windows::Win32::Foundation::HWND::default()
    }
}
