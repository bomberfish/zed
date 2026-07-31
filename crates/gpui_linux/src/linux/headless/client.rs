// The embed path (offscreen render + IPC) needs a GPU stack, so it is behind
// `gpui_wgpu` — see `open_window`. Its imports are gated with it, since a
// headless build (`remote_server`) compiles the path out entirely.
use std::cell::RefCell;
use std::collections::VecDeque;
#[cfg(feature = "gpui_wgpu")]
use std::io::IoSlice;
use std::io::{Read, Write};
#[cfg(feature = "gpui_wgpu")]
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::OnceLock;
use std::rc::Rc;
#[cfg(feature = "gpui_wgpu")]
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use calloop::{EventLoop, LoopHandle};
use gpui_util::ResultExt;
#[cfg(feature = "gpui_wgpu")]
use nix::sys::socket::{ControlMessage, MsgFlags, sendmsg};

use crate::linux::headless::window::{HeadlessDisplay, HeadlessWindow};
use crate::linux::{LinuxClient, LinuxCommon, LinuxKeyboardLayout};
#[cfg(feature = "gpui_wgpu")]
use gpui::{
    KeyDownEvent, KeyUpEvent, Keystroke, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    PlatformInput, Point, ScrollDelta, ScrollWheelEvent, TouchPhase, point, px,
};
use gpui::{
    AnyWindowHandle, CursorStyle, DisplayId, Modifiers, MouseButton, NavigationDirection,
    PlatformDisplay, PlatformKeyboardLayout, PlatformWindow, WindowParams,
};

// Modifier bitfield shared with slop2's `EmbedSurface` encoder.
const MOD_CTRL: u8 = 1 << 0;
const MOD_ALT: u8 = 1 << 1;
const MOD_SHIFT: u8 = 1 << 2;
const MOD_PLATFORM: u8 = 1 << 3;

// IPC input from the consumer (matches slop2's `EmbedSurface` encoder).
enum EmbedInput {
    PointerMove {
        x: f32,
        y: f32,
    },
    Button {
        button: u8,
        pressed: bool,
    },
    Key {
        // Already-translated GPUI keystroke fields (the consumer owns the
        // keyval→name mapping, since it has GDK's tables).
        key: String,
        key_char: Option<String>,
        modifiers: Modifiers,
        pressed: bool,
    },
    Scroll {
        dx: f32,
        dy: f32,
    },
    Resize {
        width: u32,
        height: u32,
    },
    // Host palette (slop2): 13 colors packed 0xRRGGBBAA, to theme the embed.
    Palette {
        colors: [u32; 13],
    },
    // Host system clipboard text (slop2): pushed on connect and whenever the
    // host's clipboard changes, so the embed's paste reads real system content.
    Clipboard {
        text: String,
    },
}

fn read_embed_input(r: &mut impl Read) -> std::io::Result<EmbedInput> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    let mut f32_le = |r: &mut dyn Read| -> std::io::Result<f32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    };
    match tag[0] {
        0x01 => Ok(EmbedInput::PointerMove {
            x: f32_le(r)?,
            y: f32_le(r)?,
        }),
        0x02 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(EmbedInput::Button {
                button: b[0],
                pressed: b[1] != 0,
            })
        }
        0x03 => {
            let mut header = [0u8; 3];
            r.read_exact(&mut header)?;
            let pressed = header[0] != 0;
            let mods = header[1];
            let key = read_length_prefixed(r, header[2] as usize)?;
            let mut char_len = [0u8; 1];
            r.read_exact(&mut char_len)?;
            let key_char = read_length_prefixed(r, char_len[0] as usize)?;
            Ok(EmbedInput::Key {
                key,
                key_char: (!key_char.is_empty()).then_some(key_char),
                modifiers: Modifiers {
                    control: mods & MOD_CTRL != 0,
                    alt: mods & MOD_ALT != 0,
                    shift: mods & MOD_SHIFT != 0,
                    platform: mods & MOD_PLATFORM != 0,
                    function: false,
                },
                pressed,
            })
        }
        0x04 => Ok(EmbedInput::Scroll {
            dx: f32_le(r)?,
            dy: f32_le(r)?,
        }),
        0x05 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(EmbedInput::Resize {
                width: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                height: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            })
        }
        0x06 => {
            let mut colors = [0u32; 13];
            for color in &mut colors {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                *color = u32::from_le_bytes(b);
            }
            Ok(EmbedInput::Palette { colors })
        }
        0x07 => {
            let mut len = [0u8; 4];
            r.read_exact(&mut len)?;
            let text = read_length_prefixed(r, u32::from_le_bytes(len) as usize)?;
            Ok(EmbedInput::Clipboard { text })
        }
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad input tag {other}"),
        )),
    }
}

fn read_length_prefixed(r: &mut impl Read, len: usize) -> std::io::Result<String> {
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

fn write_embed_frame(
    w: &mut impl Write,
    width: u32,
    height: u32,
    data: &[u8],
) -> std::io::Result<()> {
    w.write_all(b"FRAM")?;
    w.write_all(&width.to_le_bytes())?;
    w.write_all(&height.to_le_bytes())?;
    w.write_all(&(width * 4).to_le_bytes())?;
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    w.write_all(data)?;
    w.flush()
}

// dma-buf handshake (once): plane geometry + the DRM PRIME fds via SCM_RIGHTS.
//   b"DMAB" + num_fds u32 + width + height + stride + offset + modifier(u64) + fourcc
//
// Behind `gpui_wgpu` with the rest of the embed path: `DmabufInfo` comes from the
// renderer crate, which a headless build (`remote_server`) does not link.
#[cfg(feature = "gpui_wgpu")]
fn send_handshake(
    socket_fd: RawFd,
    info: &gpui_wgpu::DmabufInfo,
    fds: &[RawFd],
) -> std::io::Result<()> {
    let mut payload = Vec::with_capacity(36);
    payload.extend_from_slice(b"DMAB");
    payload.extend_from_slice(&(fds.len() as u32).to_le_bytes());
    payload.extend_from_slice(&info.width.to_le_bytes());
    payload.extend_from_slice(&info.height.to_le_bytes());
    payload.extend_from_slice(&info.stride.to_le_bytes());
    payload.extend_from_slice(&info.offset.to_le_bytes());
    payload.extend_from_slice(&info.modifier.to_le_bytes());
    payload.extend_from_slice(&info.fourcc.to_le_bytes());

    let iov = [IoSlice::new(&payload)];
    let cmsgs = [ControlMessage::ScmRights(fds)];
    sendmsg::<()>(socket_fd, &iov, &cmsgs, MsgFlags::empty(), None).map_err(std::io::Error::from)?;
    Ok(())
}

// Per-frame ready notification: b"RDY!" + buffer_index u32 + active width u32 +
// active height u32. The buffers are over-allocated to a fixed max size and the
// frame is rendered into the top-left; the active dims tell the consumer which
// sub-rect to sample.
fn send_ready(w: &mut impl Write, index: u32, width: u32, height: u32) -> std::io::Result<()> {
    w.write_all(b"RDY!")?;
    w.write_all(&index.to_le_bytes())?;
    w.write_all(&width.to_le_bytes())?;
    w.write_all(&height.to_le_bytes())?;
    w.flush()
}

// Window title (producer → consumer): b"TITL" + u32 len + UTF-8 title. The
// consumer mirrors it onto its tab.
fn write_title(w: &mut impl Write, title: &str) -> std::io::Result<()> {
    let bytes = title.as_bytes();
    let len = bytes.len().min(1024);
    w.write_all(b"TITL")?;
    w.write_all(&(len as u32).to_le_bytes())?;
    w.write_all(&bytes[..len])?;
    w.flush()
}

// Clipboard write (producer → consumer): b"CLIP" + u32 len + UTF-8 text. Sent
// when the embed copies, so the host puts the text on the real system clipboard.
fn write_clipboard(w: &mut impl Write, text: &str) -> std::io::Result<()> {
    let bytes = text.as_bytes();
    w.write_all(b"CLIP")?;
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)?;
    w.flush()
}

// Cursor style change (producer → consumer): b"CURS" + u8 name_len + CSS name.
// The consumer sets its widget cursor to the named GTK/CSS cursor.
fn write_cursor(w: &mut impl Write, name: &str) -> std::io::Result<()> {
    let bytes = name.as_bytes();
    w.write_all(b"CURS")?;
    w.write_all(&[bytes.len() as u8])?;
    w.write_all(bytes)?;
    w.flush()
}

/// Map a GPUI cursor style to a CSS/GTK cursor name the consumer can resolve
/// via `gdk::Cursor::from_name`.
fn cursor_name(style: CursorStyle) -> &'static str {
    use gpui::CursorStyle::*;
    match style {
        Arrow => "default",
        IBeam => "text",
        Crosshair => "crosshair",
        ClosedHand => "grabbing",
        OpenHand => "grab",
        PointingHand => "pointer",
        ResizeLeft => "w-resize",
        ResizeRight => "e-resize",
        ResizeLeftRight => "ew-resize",
        ResizeUp => "n-resize",
        ResizeDown => "s-resize",
        ResizeUpDown => "ns-resize",
        ResizeUpLeftDownRight => "nwse-resize",
        ResizeUpRightDownLeft => "nesw-resize",
        ResizeColumn => "col-resize",
        ResizeRow => "row-resize",
        IBeamCursorForVerticalLayout => "vertical-text",
        OperationNotAllowed => "not-allowed",
        _ => "default",
    }
}

// Cursor requested by the most recent `set_cursor_style` call. GPUI sets the
// cursor several times per paint (a default reset, then the hovered element's
// real cursor), so we don't forward each call — we stash the latest here and
// flush only the final value once per frame (see the frame clock), else the
// consumer flickers between the reset default and the real cursor every frame.
static PENDING_CURSOR: Mutex<Option<&'static str>> = Mutex::new(Option::None);
// System clipboard bridged from the host. The headless process has no display
// server, so `read_from_clipboard` returns this cached text (kept fresh by the
// host pushing `Clipboard` updates), and `write_to_clipboard` sends the text
// back to the host to place on the real system clipboard.
static CLIPBOARD_CACHE: Mutex<Option<String>> = Mutex::new(Option::None);
static CLIPBOARD_WRITER: Mutex<Option<UnixStream>> = Mutex::new(Option::None);

fn mouse_button(button: u8) -> MouseButton {
    // Button numbers are GTK/GDK's: 1 left, 2 middle, 3 right, 8 back, 9
    // forward (the X11/evdev convention GDK exposes for the side buttons).
    match button {
        3 => MouseButton::Right,
        2 => MouseButton::Middle,
        8 => MouseButton::Navigate(NavigationDirection::Back),
        9 => MouseButton::Navigate(NavigationDirection::Forward),
        _ => MouseButton::Left,
    }
}

// One embed process serves many host windows: the socket is bound once and its
// accept loop pushes connections here; each `open_window` binds to the next one.
static PENDING_CONNECTIONS: Mutex<VecDeque<UnixStream>> = Mutex::new(VecDeque::new());
static EMBED_LISTENER_STARTED: OnceLock<()> = OnceLock::new();
// Over-allocated dma-buf size (4K). Buffers are created once at this size and
// frames render into the top-left, so resizing needs no re-export. Consumers
// larger than this are clamped (rendered at max, stretched to fill).
const EMBED_DMABUF_MAX_WIDTH: u32 = 3840;
const EMBED_DMABUF_MAX_HEIGHT: u32 = 2160;
// Serializes enqueue of (project path -> gpui bridge, stream -> queue, counter)
// so a connection's path and stream stay index-correlated across the two queues.
static EMBED_ENQUEUE: Mutex<()> = Mutex::new(());

// First message on a new connection: the host's project dir. b"CWD " + u32 len
// + path bytes. Read before the window opens so it opens the right directory.
/// Read an embed connection's opening message: `CWD ` + u32 len + path.
///
/// `None` means the peer hung up without saying anything — the host polls this
/// socket to learn when we are accepting, and that liveness probe must not turn
/// into a window nobody asked for (it would also steal the next real consumer's
/// queue slot). Anything else is a consumer, with an empty path meaning "no
/// project named".
fn read_embed_cwd(stream: &mut UnixStream) -> Option<String> {
    let mut magic = [0u8; 4];
    stream.read_exact(&mut magic).ok()?;
    if &magic != b"CWD " {
        return Some(String::new());
    }
    let mut len = [0u8; 4];
    if stream.read_exact(&mut len).is_err() {
        return Some(String::new());
    }
    let n = u32::from_le_bytes(len) as usize;
    if n > 4096 {
        return Some(String::new());
    }
    let mut buf = vec![0u8; n];
    if stream.read_exact(&mut buf).is_err() {
        return Some(String::new());
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Start the embed accept loop once. Each accepted connection is queued and the
/// app is notified (via the gpui connection counter) to open a window for it.
fn ensure_embed_listener(socket_path: &str) {
    if EMBED_LISTENER_STARTED.set(()).is_err() {
        return;
    }
    let path = socket_path.to_owned();
    std::thread::spawn(move || {
        let _ = std::fs::remove_file(&path);
        let listener = match UnixListener::bind(&path) {
            Ok(listener) => listener,
            Err(error) => {
                log::error!("[zed-embed] failed to bind {path}: {error}");
                return;
            }
        };
        log::info!("[zed-embed] listening on {path}");
        for conn in listener.incoming() {
            match conn {
                Ok(mut stream) => {
                    // Read the project dir (first message) on a per-connection
                    // thread so a slow/absent cwd never stalls other accepts,
                    // then enqueue path + stream + counter atomically.
                    std::thread::spawn(move || {
                        let Some(path) = read_embed_cwd(&mut stream) else {
                            return;
                        };
                        let _guard = EMBED_ENQUEUE.lock();
                        gpui::embed_palette::push_embed_path(path);
                        if let Ok(mut queue) = PENDING_CONNECTIONS.lock() {
                            queue.push_back(stream);
                        }
                        gpui::embed_palette::note_embed_connection();
                    });
                }
                Err(error) => log::error!("[zed-embed] accept failed: {error}"),
            }
        }
    });
}

/// Pop the next queued connection. The app opens a window in response to the
/// connection counter, so a stream should already be queued; spin briefly to
/// cover the race where `open_window` runs just before the push completes.
fn take_pending_connection() -> Option<UnixStream> {
    for _ in 0..200 {
        if let Ok(mut queue) = PENDING_CONNECTIONS.lock()
            && let Some(stream) = queue.pop_front()
        {
            return Some(stream);
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    None
}

pub struct HeadlessClientState {
    pub(crate) _loop_handle: LoopHandle<'static, HeadlessClient>,
    pub(crate) event_loop: Option<calloop::EventLoop<'static, HeadlessClient>>,
    pub(crate) common: LinuxCommon,
    pub(crate) display: Rc<dyn PlatformDisplay>,
}

#[derive(Clone)]
pub(crate) struct HeadlessClient(Rc<RefCell<HeadlessClientState>>);

impl HeadlessClient {
    pub(crate) fn new() -> Self {
        // Embed mode opens a window per host connection, so the accept loop must
        // run from startup — not lazily in `open_window`, which only runs once a
        // connection has already been accepted (a chicken-and-egg deadlock).
        if let Ok(socket_path) = std::env::var("ZED_EMBED_SOCKET") {
            ensure_embed_listener(&socket_path);
        }

        let event_loop = EventLoop::try_new().unwrap();

        let (common, main_receiver, wake_receiver) = LinuxCommon::new(event_loop.get_signal());

        let handle = event_loop.handle();

        handle
            .insert_source(main_receiver, |event, _, _: &mut HeadlessClient| {
                if let calloop::channel::Event::Msg(runnable) = event {
                    runnable.run();
                }
            })
            .ok();

        handle
            .insert_source(wake_receiver, |event, _, client: &mut HeadlessClient| {
                if let calloop::channel::Event::Msg(()) = event {
                    client.with_common(|common| common.handle_system_wake());
                }
            })
            .ok();

        HeadlessClient(Rc::new(RefCell::new(HeadlessClientState {
            event_loop: Some(event_loop),
            _loop_handle: handle,
            common,
            display: Rc::new(HeadlessDisplay::new()),
        })))
    }
}

impl LinuxClient for HeadlessClient {
    fn with_common<R>(&self, f: impl FnOnce(&mut LinuxCommon) -> R) -> R {
        f(&mut self.0.borrow_mut().common)
    }

    fn keyboard_layout(&self) -> Box<dyn PlatformKeyboardLayout> {
        Box::new(LinuxKeyboardLayout::new("unknown".into()))
    }

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        vec![self.0.borrow().display.clone()]
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.0.borrow().display.clone())
    }

    fn display(&self, id: DisplayId) -> Option<Rc<dyn PlatformDisplay>> {
        let display = self.0.borrow().display.clone();
        (display.id() == id).then_some(display)
    }

    #[cfg(feature = "screen-capture")]
    fn screen_capture_sources(
        &self,
    ) -> futures::channel::oneshot::Receiver<anyhow::Result<Vec<Rc<dyn gpui::ScreenCaptureSource>>>>
    {
        let (tx, rx) = futures::channel::oneshot::channel();
        tx.send(Err(anyhow::anyhow!(
            "Headless mode does not support screen capture."
        )))
        .ok();
        rx
    }

    fn active_window(&self) -> Option<AnyWindowHandle> {
        None
    }

    fn window_stack(&self) -> Option<Vec<AnyWindowHandle>> {
        None
    }

    fn open_window(
        &self,
        _handle: AnyWindowHandle,
        params: WindowParams,
    ) -> anyhow::Result<Box<dyn PlatformWindow>> {
        let display = self.0.borrow().display.clone();

        // Embed mode: render offscreen and bridge frames/input over a unix
        // socket (the consumer, e.g. slop2, connects to it).
        //
        // Requires a GPU stack. `remote_server` links this crate with neither
        // `wayland` nor `x11` — the two features that bring in `gpui_wgpu` — and a
        // headless project host renders nothing, so the whole path compiles out
        // there and `open_window` falls through to the discard-everything window
        // below.
        #[cfg(feature = "gpui_wgpu")]
        if let Ok(socket_path) = std::env::var("ZED_EMBED_SOCKET") {
            // Opt into zero-copy dma-buf transport (needs Vulkan external-memory
            // support). Otherwise fall back to CPU readback frames.
            let dmabuf = std::env::var("ZED_EMBED_DMABUF").is_ok();
            let width = f32::from(params.bounds.size.width).max(1.0) as u32;
            let height = f32::from(params.bounds.size.height).max(1.0) as u32;

            let mut renderer = gpui_wgpu::WgpuRenderer::new_offscreen(width, height, dmabuf)?;

            // Bind this window to the next queued host connection. The listener
            // is shared across all embed windows in this process.
            ensure_embed_listener(&socket_path);
            let stream = match take_pending_connection() {
                Some(stream) => stream,
                None => {
                    log::error!("[zed-embed] no pending connection for new window");
                    return Ok(Box::new(HeadlessWindow::new(params, display)));
                }
            };
            log::info!("[zed-embed] binding window to a consumer (dmabuf={dmabuf})");

            // First byte tells the consumer which transport to expect.
            {
                let mut mode_writer = stream.try_clone()?;
                mode_writer.write_all(&[dmabuf as u8]).ok();
            }

            // Cursor changes are routed per-window from the frame clock below
            // (using this window's own writer), not through a global.
            // Route clipboard writes (copy in the embed) to this consumer.
            if let Ok(clip_clone) = stream.try_clone()
                && let Ok(mut guard) = CLIPBOARD_WRITER.lock()
            {
                *guard = Some(clip_clone);
            }
            let mut reader = stream.try_clone()?;

            if !dmabuf {
                let mut writer = stream.try_clone()?;
                renderer.set_on_frame(Box::new(move |bytes, w, h| {
                    let _ = write_embed_frame(&mut writer, w, h, bytes);
                }));
            }

            let window = HeadlessWindow::new_embed(params, display, renderer);

            // Mirror the window title (project/file name) onto the host's tab.
            if let Ok(mut title_writer) = stream.try_clone() {
                window.set_on_title(Box::new(move |title| {
                    let _ = write_title(&mut title_writer, title);
                }));
            }

            // Socket reader thread → shared input queue. Flips `connected` to
            // false on EOF (host closed the tab), which the frame clock turns
            // into a window close.
            let queue: Arc<Mutex<VecDeque<EmbedInput>>> = Arc::new(Mutex::new(VecDeque::new()));
            let connected = Arc::new(std::sync::atomic::AtomicBool::new(true));
            {
                let queue = Arc::clone(&queue);
                let connected = Arc::clone(&connected);
                std::thread::spawn(move || {
                    while let Ok(msg) = read_embed_input(&mut reader) {
                        if let Ok(mut q) = queue.lock() {
                            q.push_back(msg);
                        }
                    }
                    connected.store(false, std::sync::atomic::Ordering::Release);
                });
            }

            // dma-buf: create the (fixed max-size) exportable buffers and send
            // the handshake NOW — before any title/cursor traffic — so it's the
            // first producer→consumer message after the mode byte. Over-alloc
            // makes this size-independent; frames render at the active size into
            // each buffer's top-left. `RDY!` carries the active size per frame.
            if dmabuf
                && let Some((info, fds)) =
                    window.enable_dmabuf(EMBED_DMABUF_MAX_WIDTH, EMBED_DMABUF_MAX_HEIGHT, 2)
            {
                if send_handshake(stream.as_raw_fd(), &info, &fds).is_ok()
                    && let Ok(rdy_writer) = stream.try_clone()
                {
                    let rdy_writer = Arc::new(Mutex::new(rdy_writer));
                    window.set_on_dmabuf(Box::new(move |index, w, h| {
                        if let Ok(mut stream) = rdy_writer.lock() {
                            let _ = send_ready(&mut *stream, index as u32, w, h);
                        }
                    }));
                } else {
                    log::error!("[zed-embed] dma-buf handshake failed to send");
                }
            }

            // Frame clock: drain input + drive a frame every ~16ms. Hold the
            // window WEAKLY so that when Zed removes it (window closed), the
            // clock notices (upgrade fails) and tears down the connection —
            // which makes the host close the tab. `close_stream` is the handle
            // we shut down to signal that.
            let win_weak = window.downgrade();
            let mut close_stream = stream.try_clone()?;
            // Per-window cursor routing: this window's own writer + last-sent
            // cursor. `set_cursor_style` is client-global (no window arg), but the
            // frame clocks run sequentially on the one event-loop thread, so the
            // cursor GPUI sets during *this* window's tick belongs to it. We clear
            // the shared PENDING_CURSOR at the top of the tick and flush whatever
            // this tick produced to this window's consumer — so multiple embed
            // windows each get their own cursor instead of all funneling to the
            // last-opened one.
            let mut cursor_writer = stream.try_clone()?;
            let mut last_cursor: Option<&'static str> = None;
            let mut last_pos: Point<Pixels> = point(px(0.0), px(0.0));
            // Track the currently-held button so drags carry it on MouseMove
            // (GPUI needs `pressed_button` set for drag gestures to work).
            let mut pressed_button: Option<MouseButton> = None;
            self.0
                .borrow()
                ._loop_handle
                .insert_source(
                    calloop::timer::Timer::from_duration(Duration::from_millis(16)),
                    move |_, _, _client: &mut HeadlessClient| {
                        // Host closed the connection (tab closed) → close the
                        // window app-side and stop this frame clock.
                        if !connected.load(std::sync::atomic::Ordering::Acquire) {
                            if let Some(win) = win_weak.upgrade() {
                                win.fire_close();
                            }
                            return calloop::timer::TimeoutAction::Drop;
                        }
                        // Zed removed the window (window closed) → shut the socket
                        // so the host tears down its tab, and stop the clock.
                        let Some(win) = win_weak.upgrade() else {
                            let _ = close_stream.shutdown(std::net::Shutdown::Both);
                            return calloop::timer::TimeoutAction::Drop;
                        };
                        // Clear the shared cursor slot so this tick captures only
                        // the cursor GPUI sets for *this* window's paint/input.
                        if let Ok(mut pending) = PENDING_CURSOR.lock() {
                            *pending = None;
                        }
                        // Mark active once GPUI has registered its callbacks, so
                        // the editor accepts typed text (not just keybindings).
                        win.ensure_active();
                        let drained: Vec<EmbedInput> = {
                            match queue.lock() {
                                Ok(mut q) => q.drain(..).collect(),
                                Err(_) => Vec::new(),
                            }
                        };
                        for msg in drained {
                            match msg {
                                EmbedInput::PointerMove { x, y } => {
                                    last_pos = point(px(x), px(y));
                                    win.dispatch_input(PlatformInput::MouseMove(MouseMoveEvent {
                                        position: last_pos,
                                        pressed_button,
                                        modifiers: Modifiers::default(),
                                    }));
                                }
                                EmbedInput::Button { button, pressed } => {
                                    let button = mouse_button(button);
                                    if pressed {
                                        pressed_button = Some(button);
                                        win.dispatch_input(PlatformInput::MouseDown(
                                            MouseDownEvent {
                                                button,
                                                position: last_pos,
                                                modifiers: Modifiers::default(),
                                                click_count: 1,
                                                first_mouse: false,
                                            },
                                        ));
                                    } else {
                                        pressed_button = None;
                                        win.dispatch_input(PlatformInput::MouseUp(MouseUpEvent {
                                            button,
                                            position: last_pos,
                                            modifiers: Modifiers::default(),
                                            click_count: 1,
                                        }));
                                    }
                                }
                                EmbedInput::Resize { width, height } => {
                                    // Clamp to the over-allocated dma-buf max so the
                                    // blit into the fixed-size buffers never
                                    // overflows. The dma-buf handshake already
                                    // happened at connect; resize only re-sizes the
                                    // offscreen render target (rendered into each
                                    // buffer's top-left).
                                    let width = width.min(EMBED_DMABUF_MAX_WIDTH);
                                    let height = height.min(EMBED_DMABUF_MAX_HEIGHT);
                                    win.resize_embed(width, height);
                                }
                                EmbedInput::Palette { colors } => {
                                    // Hand the host palette to the app's theme
                                    // poller via the gpui bridge.
                                    gpui::embed_palette::set_embed_palette(
                                        gpui::embed_palette::EmbedPalette { colors },
                                    );
                                }
                                EmbedInput::Clipboard { text } => {
                                    // Host's system clipboard changed; cache it so
                                    // the embed's paste reads real system content.
                                    if let Ok(mut cache) = CLIPBOARD_CACHE.lock() {
                                        *cache = Some(text);
                                    }
                                }
                                EmbedInput::Scroll { dx, dy } => {
                                    // GTK delivers ~1.0 per wheel notch; native
                                    // GPUI converts a notch to `SCROLL_LINES`
                                    // lines, so match that or scrolling is ~1/3
                                    // as sensitive. Axes are negated (GTK's sign
                                    // is opposite GPUI's).
                                    let lines = crate::linux::platform::SCROLL_LINES;
                                    win.dispatch_input(PlatformInput::ScrollWheel(
                                        ScrollWheelEvent {
                                            position: last_pos,
                                            delta: ScrollDelta::Lines(point(
                                                -dx * lines,
                                                -dy * lines,
                                            )),
                                            modifiers: Modifiers::default(),
                                            touch_phase: TouchPhase::Moved,
                                        },
                                    ));
                                }
                                EmbedInput::Key {
                                    key,
                                    key_char,
                                    modifiers,
                                    pressed,
                                } => {
                                    let keystroke = Keystroke {
                                        modifiers,
                                        key,
                                        key_char,
                                    };
                                    if pressed {
                                        win.dispatch_input(PlatformInput::KeyDown(KeyDownEvent {
                                            keystroke,
                                            is_held: false,
                                            prefer_character_input: false,
                                        }));
                                    } else {
                                        win.dispatch_input(PlatformInput::KeyUp(KeyUpEvent {
                                            keystroke,
                                        }));
                                    }
                                }
                            }
                        }
                        win.tick_frame();
                        // Forward this window's final cursor (if it changed) to
                        // its own consumer — one CURS per frame, after the paint's
                        // set_cursor_style calls have settled.
                        if let Ok(pending) = PENDING_CURSOR.lock() {
                            if let Some(name) = *pending {
                                if last_cursor != Some(name) {
                                    last_cursor = Some(name);
                                    let _ = write_cursor(&mut cursor_writer, name);
                                }
                            }
                        }
                        calloop::timer::TimeoutAction::ToDuration(Duration::from_millis(16))
                    },
                )
                .ok();

            return Ok(Box::new(window));
        }

        Ok(Box::new(HeadlessWindow::new(params, display)))
    }

    fn compositor_name(&self) -> &'static str {
        "headless"
    }

    fn set_cursor_style(&self, style: CursorStyle) {
        // Just record the latest request; the frame clock forwards the final
        // value once per frame (coalescing the per-paint default-then-real
        // sequence into one, so the consumer doesn't flicker).
        if let Ok(mut pending) = PENDING_CURSOR.lock() {
            *pending = Some(cursor_name(style));
        }
    }

    fn open_uri(&self, _uri: &str) {}

    fn reveal_path(&self, _path: std::path::PathBuf) {}

    fn write_to_primary(&self, _item: gpui::ClipboardItem) {}

    fn write_to_clipboard(&self, item: gpui::ClipboardItem) {
        let Some(text) = item.text() else {
            return;
        };
        // Cache locally so an immediate read reflects the copy without a round
        // trip, then ask the host to place it on the real system clipboard.
        if let Ok(mut cache) = CLIPBOARD_CACHE.lock() {
            *cache = Some(text.clone());
        }
        if let Ok(mut guard) = CLIPBOARD_WRITER.lock()
            && let Some(writer) = guard.as_mut()
        {
            let _ = write_clipboard(writer, &text);
        }
    }

    fn read_from_primary(&self) -> Option<gpui::ClipboardItem> {
        None
    }

    fn read_from_clipboard(&self) -> Option<gpui::ClipboardItem> {
        CLIPBOARD_CACHE
            .lock()
            .ok()
            .and_then(|cache| cache.clone())
            .map(gpui::ClipboardItem::new_string)
    }

    fn run(&self) {
        let mut event_loop = self
            .0
            .borrow_mut()
            .event_loop
            .take()
            .expect("App is already running");

        event_loop.run(None, &mut self.clone(), |_| {}).log_err();
    }
}

#[cfg(all(test, feature = "gpui_wgpu"))]
mod tests {
    use super::*;

    /// The host polls the embed socket to learn when we accept connections. That
    /// probe used to be indistinguishable from a consumer, so it opened a window
    /// and took the next real consumer's queue slot — which broke every embed
    /// that needed its window to be created for a specific project.
    #[test]
    fn a_probe_is_not_a_consumer_but_a_pathless_client_is() {
        let (host, mut zed) = UnixStream::pair().expect("socketpair");
        drop(host);
        assert_eq!(
            read_embed_cwd(&mut zed),
            None,
            "connect-and-hang-up is a probe"
        );

        let (mut host, mut zed) = UnixStream::pair().expect("socketpair");
        host.write_all(b"CWD ").unwrap();
        host.write_all(&12u32.to_le_bytes()).unwrap();
        host.write_all(b"/home/velzie").unwrap();
        assert_eq!(read_embed_cwd(&mut zed).as_deref(), Some("/home/velzie"));

        // A consumer that names no project still gets its window.
        let (mut host, mut zed) = UnixStream::pair().expect("socketpair");
        host.write_all(b"CWD ").unwrap();
        host.write_all(&0u32.to_le_bytes()).unwrap();
        drop(host);
        assert_eq!(read_embed_cwd(&mut zed).as_deref(), Some(""));

        // So does one whose opening message we don't recognise, since it did
        // speak: dropping it would strand a peer that got here somehow.
        let (mut host, mut zed) = UnixStream::pair().expect("socketpair");
        host.write_all(b"RFB \x00").unwrap();
        drop(host);
        assert_eq!(read_embed_cwd(&mut zed).as_deref(), Some(""));
    }
}
