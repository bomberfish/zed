//! Windows for the headless platform client.
//!
//! A headless window has no compositor surface and no GPU: layout, text
//! shaping, and entity plumbing run normally, `draw` discards the scene, and
//! the sprite atlas hands out tiles without uploading pixels (mirroring
//! GPUI's `TestWindow`/`TestAtlas`). This lets command-line tools drive real
//! `Window`-based code paths without a display server.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use collections::HashMap;
use parking_lot::Mutex;
use uuid::Uuid;

use gpui::{
    AtlasKey, AtlasTextureId, AtlasTile, Bounds, Capslock, DevicePixels, DispatchEventResult,
    DisplayId, GpuSpecs, Modifiers, Pixels, PlatformAtlas, PlatformDisplay, PlatformInput,
    PlatformInputHandler, PlatformWindow, Point, PromptButton, PromptLevel, RequestFrameOptions,
    Scene, Size, TileId, WindowAppearance, WindowBackgroundAppearance, WindowBounds,
    WindowControlArea, WindowParams, px,
};

#[derive(Debug)]
pub(crate) struct HeadlessDisplay {
    bounds: Bounds<Pixels>,
}

impl HeadlessDisplay {
    pub(crate) fn new() -> Self {
        Self {
            bounds: Bounds::from_corners(Point::default(), Point::new(px(1920.), px(1080.))),
        }
    }
}

impl PlatformDisplay for HeadlessDisplay {
    fn id(&self) -> DisplayId {
        DisplayId::new(0)
    }

    fn uuid(&self) -> anyhow::Result<Uuid> {
        // Stable identity: there is exactly one headless display.
        Ok(Uuid::nil())
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}

struct HeadlessWindowState {
    bounds: Bounds<Pixels>,
    display: Rc<dyn PlatformDisplay>,
    input_handler: Option<PlatformInputHandler>,
    title: Option<String>,
    is_fullscreen: bool,
    // Embed mode (`ZED_EMBED_SOCKET`): render offscreen via wgpu and drive
    // frames/input over IPC instead of discarding everything.
    renderer: Option<EmbedRenderer>,
    request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    /// Force the next frame even when GPUI considers the window clean.
    ///
    /// The tick used to force EVERY frame, so an idle embed rebuilt and
    /// rasterized the whole scene at the consumer's tick rate while a native
    /// window idles at zero. Dirty-tracking handles the scene; this covers the
    /// cases where the OUTPUT changed without it — a consumer that just attached
    /// has no pixels, and a resize reallocates the buffers.
    force_frame: bool,
    input: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    resize: Option<Box<dyn FnMut(Size<Pixels>, f32)>>,
    // Fired when the host connection drops so GPUI removes this window.
    close: Option<Box<dyn FnOnce()>>,
    // Window activation. An embed window is always the active input target
    // (there's no OS focus); firing this with `true` is what lets the focused
    // editor accept typed text, not just keybindings.
    active_status_change: Option<Box<dyn FnMut(bool)>>,
    activated: bool,
    // Hover state. GPUI only pushes cursor-style changes while the window is
    // hovered, so an embed window (which always "has" the pointer) must report
    // hovered=true or forwarded cursor styles never apply.
    hover_status_change: Option<Box<dyn FnMut(bool)>>,
    // Fired when GPUI sets the window title, so the host can mirror it.
    title_change: Option<Box<dyn FnMut(&str)>>,
    /// Whether the frames handed to the consumer are meant to carry a background
    /// of their own.
    ///
    /// Stored rather than derived because there is no compositor to ask: the frame
    /// IS the answer, and the offscreen renderer already clears to transparent
    /// black, so what a `Transparent` theme changes is only whether the scene
    /// paints an opaque backdrop over that clear. GPUI still reads this back — for
    /// subpixel-rendering decisions, and to keep a window that asked for
    /// transparency from being told it is opaque.
    background: WindowBackgroundAppearance,
}

#[derive(Clone)]
pub(crate) struct HeadlessWindow(Rc<RefCell<HeadlessWindowState>>);

/// Weak handle so the frame clock can hold the window without keeping it alive —
/// when GPUI removes the window, `upgrade` returns `None` and the clock can tear
/// down the connection.
#[derive(Clone)]
pub(crate) struct WeakHeadlessWindow(std::rc::Weak<RefCell<HeadlessWindowState>>);

impl WeakHeadlessWindow {
    pub(crate) fn upgrade(&self) -> Option<HeadlessWindow> {
        self.0.upgrade().map(HeadlessWindow)
    }
}

impl raw_window_handle::HasWindowHandle for HeadlessWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        // Headless windows are not backed by a native window.
        Err(raw_window_handle::HandleError::NotSupported)
    }
}

impl raw_window_handle::HasDisplayHandle for HeadlessWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::NotSupported)
    }
}

/// The offscreen renderer the embed path drives.
///
/// `remote_server` links this crate with neither `wayland` nor `x11`, and both of
/// those are what pull in `gpui_wgpu` — a headless project host has no GPU stack
/// and no business rendering. The headless platform, however, is compiled in
/// EVERY configuration, so the embed code has to survive that build. It does by
/// swapping the renderer for an uninhabited stand-in: `renderer` is then provably
/// always `None`, the embed entry points are compiled out, and the few remaining
/// call sites keep their exact shape.
#[cfg(feature = "gpui_wgpu")]
pub(crate) type EmbedRenderer = gpui_wgpu::WgpuRenderer;

#[cfg(not(feature = "gpui_wgpu"))]
pub(crate) enum EmbedRenderer {}

#[cfg(not(feature = "gpui_wgpu"))]
impl EmbedRenderer {
    // Bodies are `match *self {}`: the type has no values, so these are
    // unreachable by construction and need no fallback behaviour.
    pub(crate) fn resize_offscreen(&mut self, _width: u32, _height: u32) {
        match *self {}
    }

    pub(crate) fn draw_offscreen(&mut self, _scene: &Scene) {
        match *self {}
    }

    pub(crate) fn sprite_atlas(&self) -> &Arc<dyn PlatformAtlas> {
        match *self {}
    }
}

impl HeadlessWindow {
    pub(crate) fn new(params: WindowParams, display: Rc<dyn PlatformDisplay>) -> Self {
        Self(Rc::new(RefCell::new(HeadlessWindowState {
            bounds: params.bounds,
            display,
            input_handler: None,
            title: None,
            is_fullscreen: false,
            renderer: None,
            request_frame: None,
            // The first frame always happens: a fresh consumer has nothing.
            force_frame: true,
            input: None,
            resize: None,
            close: None,
            active_status_change: None,
            activated: false,
            hover_status_change: None,
            title_change: None,
            background: WindowBackgroundAppearance::Opaque,
        })))
    }

    /// Embed-mode window: owns an offscreen wgpu renderer (already wired with an
    /// `on_frame` callback that ships pixels over IPC).
    #[cfg(feature = "gpui_wgpu")]
    pub(crate) fn new_embed(
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        renderer: EmbedRenderer,
    ) -> Self {
        Self(Rc::new(RefCell::new(HeadlessWindowState {
            bounds: params.bounds,
            display,
            input_handler: None,
            title: None,
            is_fullscreen: false,
            renderer: Some(renderer),
            request_frame: None,
            // The first frame always happens: a fresh consumer has nothing.
            force_frame: true,
            input: None,
            resize: None,
            close: None,
            active_status_change: None,
            activated: false,
            hover_status_change: None,
            title_change: None,
            background: WindowBackgroundAppearance::Opaque,
        })))
    }

    /// Ask GPUI to produce + present a frame (drives `draw`). Called from the
    /// embed frame clock.
    pub(crate) fn tick_frame(&self) {
        let forced = {
            let mut state = self.0.borrow_mut();
            std::mem::replace(&mut state.force_frame, false)
        };
        let mut cb = self.0.borrow_mut().request_frame.take();
        if let Some(cb) = cb.as_mut() {
            cb(RequestFrameOptions {
                require_presentation: forced,
                force_render: forced,
            });
        }
        self.0.borrow_mut().request_frame = cb;
    }

    /// Make the next tick produce a frame even if the scene did not change.
    pub(crate) fn force_next_frame(&self) {
        self.0.borrow_mut().force_frame = true;
    }

    /// Resize the embed surface to match the consumer's display area: resize the
    /// offscreen render target and notify GPUI so it relayouts at the new size.
    pub(crate) fn resize_embed(&self, width: u32, height: u32) {
        let size = Size {
            width: px(width as f32),
            height: px(height as f32),
        };
        {
            let mut state = self.0.borrow_mut();
            if state.bounds.size == size {
                return;
            }
            state.bounds.size = size;
            // The reallocated buffers are blank even if the scene is identical.
            state.force_frame = true;
            if let Some(renderer) = state.renderer.as_mut() {
                renderer.resize_offscreen(width, height);
            }
        }
        let mut cb = self.0.borrow_mut().resize.take();
        if let Some(cb) = cb.as_mut() {
            // Scale factor is 1.0 for embed windows (see `scale_factor`).
            cb(size, 1.0);
        }
        self.0.borrow_mut().resize = cb;
    }

    /// Enable zero-copy dma-buf export on the embed renderer, returning the
    /// plane geometry + fds to hand to the consumer.
    #[cfg(feature = "gpui_wgpu")]
    pub(crate) fn enable_dmabuf(
        &self,
        width: u32,
        height: u32,
        count: usize,
    ) -> Option<(gpui_wgpu::DmabufInfo, Vec<std::os::fd::RawFd>)> {
        let mut state = self.0.borrow_mut();
        let renderer = state.renderer.as_mut()?;
        match renderer.enable_dmabuf(width, height, count) {
            Ok(result) => Some(result),
            Err(error) => {
                log::error!("[zed-embed] enable_dmabuf failed: {error:#}");
                None
            }
        }
    }

    /// Register the per-frame dma-buf ready callback on the embed renderer
    /// (target index + active width/height).
    #[cfg(feature = "gpui_wgpu")]
    pub(crate) fn set_on_dmabuf(&self, callback: Box<dyn FnMut(usize, u32, u32)>) {
        if let Some(renderer) = self.0.borrow_mut().renderer.as_mut() {
            renderer.set_on_dmabuf(callback);
        }
    }

    /// Mark the window active once (embed windows are always the active input
    /// target). GPUI uses this to focus the content and route typed text to the
    /// editor's input handler — without it only keybindings work, not typing.
    pub(crate) fn ensure_active(&self) {
        let (mut active_cb, mut hover_cb) = {
            let mut state = self.0.borrow_mut();
            if state.activated {
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
        // Active → editor accepts typed text. Hovered → GPUI pushes cursor
        // styles (both gated on these states for a normal window).
        if let Some(cb) = active_cb.as_mut() {
            cb(true);
        }
        if let Some(cb) = hover_cb.as_mut() {
            cb(true);
        }
        let mut state = self.0.borrow_mut();
        state.active_status_change = active_cb;
        state.hover_status_change = hover_cb;
    }

    pub(crate) fn downgrade(&self) -> WeakHeadlessWindow {
        WeakHeadlessWindow(Rc::downgrade(&self.0))
    }

    /// Register a callback for window-title changes; fires immediately with the
    /// current title if one is already set.
    pub(crate) fn set_on_title(&self, mut callback: Box<dyn FnMut(&str)>) {
        let current = self.0.borrow().title.clone();
        if let Some(title) = current {
            callback(&title);
        }
        self.0.borrow_mut().title_change = Some(callback);
    }

    /// Fire the close callback (host connection dropped). Idempotent: the
    /// `FnOnce` is taken, so subsequent calls are no-ops. GPUI's callback
    /// removes the window app-side.
    pub(crate) fn fire_close(&self) {
        if let Some(callback) = self.0.borrow_mut().close.take() {
            callback();
        }
    }

    /// Dispatch a forwarded input event into GPUI.
    pub(crate) fn dispatch_input(&self, event: PlatformInput) {
        let mut cb = self.0.borrow_mut().input.take();
        if let Some(cb) = cb.as_mut() {
            cb(event);
        }
        self.0.borrow_mut().input = cb;
    }
}

impl PlatformWindow for HeadlessWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().bounds
    }

    fn is_maximized(&self) -> bool {
        false
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.bounds())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.bounds().size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        self.0.borrow_mut().bounds.size = size;
    }

    fn scale_factor(&self) -> f32 {
        1.0
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
        // Embed windows have no OS focus; treat them as always active so the
        // editor accepts typed input.
        true
    }

    fn is_hovered(&self) -> bool {
        // The host forwards pointer events, so treat the embed as hovered
        // (needed for GPUI to push cursor-style changes).
        true
    }

    fn background_appearance(&self) -> WindowBackgroundAppearance {
        self.0.borrow().background
    }

    fn set_title(&mut self, title: &str) {
        self.0.borrow_mut().title = Some(title.to_owned());
        let mut cb = self.0.borrow_mut().title_change.take();
        if let Some(cb) = cb.as_mut() {
            cb(title);
        }
        self.0.borrow_mut().title_change = cb;
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
        // The scene has not changed, only what the pixels it does not cover will
        // hold, so dirty-tracking would skip the repaint that makes this visible.
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

    // In embed mode the calloop frame clock drives these; otherwise (pure
    // headless) they're dropped and anything awaiting a frame never resolves.
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
        if let Some(renderer) = self.0.borrow_mut().renderer.as_mut() {
            renderer.draw_offscreen(scene);
        }
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        if let Some(renderer) = self.0.borrow().renderer.as_ref() {
            return renderer.sprite_atlas().clone();
        }
        Arc::new(HeadlessAtlas::default())
    }

    /// No native window here, so nothing else will deliver typed characters.
    fn commits_key_char(&self) -> bool {
        true
    }

    fn is_subpixel_rendering_supported(&self) -> bool {
        false
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        None
    }
}

/// Allocates atlas tiles without uploading pixels, so glyph and sprite
/// painting completes headlessly.
#[derive(Default)]
struct HeadlessAtlas(Mutex<HeadlessAtlasState>);

#[derive(Default)]
struct HeadlessAtlasState {
    next_id: u32,
    tiles: HashMap<AtlasKey, AtlasTile>,
}

impl PlatformAtlas for HeadlessAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        {
            let state = self.0.lock();
            if let Some(&tile) = state.tiles.get(key) {
                return Ok(Some(tile));
            }
        }

        let Some((size, _)) = build()? else {
            return Ok(None);
        };

        let mut state = self.0.lock();
        state.next_id += 1;
        let texture_id = state.next_id;
        state.next_id += 1;
        let tile_id = state.next_id;
        let tile = AtlasTile {
            texture_id: AtlasTextureId {
                index: texture_id,
                kind: key.texture_kind(),
            },
            tile_id: TileId(tile_id),
            padding: 0,
            bounds: Bounds {
                origin: Point::default(),
                size,
            },
        };
        state.tiles.insert(key.clone(), tile);
        Ok(Some(tile))
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().tiles.remove(key);
    }
}
