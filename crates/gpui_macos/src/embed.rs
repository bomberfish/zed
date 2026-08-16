//! Embed sessions: render Zed into a host application's window.
//!
//! The macOS counterpart to `gpui_linux`'s headless embed client and
//! `gpui_windows`'s embed pipe. The wire format is byte-identical to both, which
//! is what lets one consumer implementation drive all three: the transports
//! differ only in how the frame buffer itself is handed over (Linux passes DRM
//! PRIME fds as `SCM_RIGHTS`, Windows duplicates a D3D12 NT handle, and macOS
//! passes an IOSurface mach send right), and each announces its own handshake
//! after the shared mode byte.
//!
//! The accept loop follows Linux rather than Windows: the host spawns ONE zed and
//! points every pane at the same socket, so this process binds once at startup
//! and opens a window per accepted connection. Windows can afford to block in
//! `WindowsPlatform::new` until a consumer connects; here that would deadlock the
//! host, which connects to this socket purely to discover that we are up (see
//! [`read_embed_cwd`]) before it has any consumer to offer.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use gpui::{
    Capslock, KeyDownEvent, KeyUpEvent, Keystroke, Modifiers, ModifiersChangedEvent, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, NavigationDirection, Pixels, PlatformInput,
    Point, ScrollDelta, ScrollWheelEvent, TouchPhase, point, px,
};
use parking_lot::Mutex;

use crate::embed_window::EmbedWindow;

/// Lines per wheel notch. Windows reads `SPI_GETWHEELSCROLLLINES` for this;
/// macOS has no equivalent to read, because AppKit folds the setting into
/// `scrollingDeltaY` before a native window ever sees it — and an embed window
/// gets no `NSEvent` at all. 3.0 is what AppKit reports per detent for a plain
/// wheel, and what the Linux producer uses.
const SCROLL_LINES: f32 = 3.0;

/// Environment variable naming the endpoint to serve. Its presence is what puts
/// this process into embed mode.
///
/// The same variable on every platform: gpui, workspace, and zed's main all gate
/// embed behaviour on this name (dropping the title bar, applying the host
/// palette, skipping the boot workspace, committing typed characters). A second,
/// macOS-only name would leave every one of those silently switched off.
pub(crate) const EMBED_ENDPOINT_VAR: &str = "ZED_EMBED_SOCKET";

/// Set beside [`EMBED_ENDPOINT_VAR`] to offer the zero-copy transport, matching
/// Linux's `ZED_EMBED_DMABUF`. Presence alone is the signal, like there — the host
/// sets it only for a consumer that negotiated the capability, and its own RFB
/// consumer, which has no compositor to hand a surface to, wants CPU frames.
pub(crate) const EMBED_IOSURFACE_VAR: &str = "ZED_EMBED_IOSURFACE";

// Modifier bitfield shared with slop2's embed surface encoder.
const MOD_CTRL: u8 = 1 << 0;
const MOD_ALT: u8 = 1 << 1;
const MOD_SHIFT: u8 = 1 << 2;
const MOD_PLATFORM: u8 = 1 << 3;

/// Input forwarded by the consumer. Mirrors the Linux and Windows `EmbedInput`
/// exactly; the consumer owns keyval→name translation because it is the side
/// holding the platform's key tables.
pub(crate) enum EmbedInput {
    PointerMove {
        x: f32,
        y: f32,
    },
    Button {
        button: u8,
        pressed: bool,
    },
    Key {
        key: String,
        key_char: Option<String>,
        modifiers: Modifiers,
        pressed: bool,
    },
    Scroll {
        dx: f32,
        dy: f32,
    },
    /// Smooth scroll in pixels, from a precision touchpad. A separate message
    /// from `Scroll` because that one is in wheel notches, and the two differ by
    /// more than an order of magnitude.
    ScrollPixels {
        dx: f32,
        dy: f32,
    },
    Resize {
        width: u32,
        height: u32,
    },
    Palette {
        colors: [u32; 13],
    },
    /// A modifier pressed or released on its own; carries the resulting state.
    ModifiersChanged {
        modifiers: Modifiers,
    },
    Clipboard {
        text: String,
    },
    /// Scale as a percentage (150 = 1.5x): the consumer's display scale times the
    /// user's editor multiplier. Sizes stay in pixels; this is what turns them
    /// into the logical extent gpui lays out against.
    ScalePercent {
        pct: u32,
    },
    /// The CoreGraphics display the consumer is showing this pane on.
    ///
    /// The one input this platform has that Linux and Windows do not. An embed
    /// producer renders offscreen, so it is on no screen at all and cannot ask which
    /// one its pixels end up on; the frame clock has to guess, and the guess it
    /// started with (`CGMainDisplayID()`, resolved once at session start) is wrong in
    /// the direction that costs frames — a 120Hz built-in paces an embed shown on a
    /// 240Hz external at half its rate, and in clamshell it names a display that is
    /// off. The consumer knows the answer for free from `NSWindow.screen`.
    Display {
        display_id: u32,
    },
}

/// The write half of a consumer's socket.
///
/// Linux hands each producer→consumer message its own `try_clone`d `UnixStream`;
/// that works there because every writer runs on the one calloop thread. Here
/// the writers do not share a thread — clipboard writes come from
/// `Platform::write_to_clipboard`, and the zero-copy path's `RDY!` comes from a
/// Metal completion handler on a driver thread — so the socket is wrapped once
/// and each message is assembled into a single buffer written under the lock.
/// Two interleaved partial writes would desynchronize the stream permanently.
pub(crate) struct EmbedSocket {
    writer: Mutex<UnixStream>,
}

impl EmbedSocket {
    fn new(stream: UnixStream) -> Self {
        Self {
            writer: Mutex::new(stream),
        }
    }

    fn write_message(&self, payload: &[u8]) -> Result<()> {
        let mut writer = self.writer.lock();
        writer.write_all(payload)?;
        writer.flush()?;
        Ok(())
    }

    /// Drop the connection, which is how the consumer learns the editor closed
    /// its window and the host should tear the pane down.
    pub(crate) fn shutdown(&self) {
        if let Err(error) = self.writer.lock().shutdown(std::net::Shutdown::Both) {
            log::info!("[zed-embed] shutting the consumer socket down: {error}");
        }
    }
}

/// The first byte the consumer expects: 1 for zero-copy surface handoff, 0 for
/// CPU frames. Sending the handshake without it leaves the stream misaligned by
/// one byte.
pub(crate) fn send_mode(socket: &EmbedSocket, zero_copy: bool) -> Result<()> {
    socket.write_message(&[zero_copy as u8])
}

/// The zero-copy handshake, sent immediately after a mode byte of 1 — the slot
/// where Linux sends `DMAB` and Windows sends `SHTX`:
///
/// `IOSF` + u32 name length + bootstrap service name + u32 buffer count +
/// u32 max width + u32 max height
///
/// The name is plain text rather than a handle or descriptor because a mach send
/// right cannot cross this socket at all (see `iosurface.rs`); the consumer looks
/// the name up in the bootstrap namespace and collects one right per buffer from
/// there. The maximum dimensions are the surfaces' real size, which every frame
/// renders into the top-left corner of, so the consumer needs them to compute the
/// sub-rect the active size occupies.
pub(crate) fn send_iosurface_handshake(
    socket: &EmbedSocket,
    service_name: &str,
    buffer_count: u32,
    max_width: u32,
    max_height: u32,
) -> Result<()> {
    let name = service_name.as_bytes();
    let mut payload = Vec::with_capacity(20 + name.len());
    payload.extend_from_slice(b"IOSF");
    payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
    payload.extend_from_slice(name);
    payload.extend_from_slice(&buffer_count.to_le_bytes());
    payload.extend_from_slice(&max_width.to_le_bytes());
    payload.extend_from_slice(&max_height.to_le_bytes());
    socket.write_message(&payload)
}

/// A finished frame in the zero-copy transport: `RDY!` + buffer index + the
/// active width and height, which are what the consumer clips the oversized
/// surface down to.
pub(crate) fn send_ready(socket: &EmbedSocket, index: u32, width: u32, height: u32) -> Result<()> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(b"RDY!");
    payload.extend_from_slice(&index.to_le_bytes());
    payload.extend_from_slice(&width.to_le_bytes());
    payload.extend_from_slice(&height.to_le_bytes());
    socket.write_message(&payload)
}

/// A whole frame's pixels, for the CPU-readback transport: `FRAM` + width +
/// height + stride + byte count + RGBA bytes. The caller owns the byte order;
/// `embed_window.rs` swaps its BGRA render target into RGBA before calling here,
/// because that is what the Linux producer's `Rgba8Unorm` readback yields and the
/// consumers decode.
pub(crate) fn send_frame(socket: &EmbedSocket, width: u32, height: u32, data: &[u8]) -> Result<()> {
    let mut payload = Vec::with_capacity(20 + data.len());
    payload.extend_from_slice(b"FRAM");
    payload.extend_from_slice(&width.to_le_bytes());
    payload.extend_from_slice(&height.to_le_bytes());
    payload.extend_from_slice(&(width * 4).to_le_bytes());
    payload.extend_from_slice(&(data.len() as u32).to_le_bytes());
    payload.extend_from_slice(data);
    socket.write_message(&payload)
}

/// Cursor style change: `CURS` + u8 name length + CSS name, byte-identical to
/// the other producers'. The consumer resolves the name to a platform cursor.
pub(crate) fn send_cursor(socket: &EmbedSocket, name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    let len = bytes.len().min(64);
    let mut payload = Vec::with_capacity(5 + len);
    payload.extend_from_slice(b"CURS");
    payload.push(len as u8);
    payload.extend_from_slice(&bytes[..len]);
    socket.write_message(&payload)
}

/// The cursor the most recent `set_cursor_style` asked for.
///
/// GPUI sets the cursor several times per paint — a default reset, then the
/// hovered element's real one — so forwarding every call would put a burst of
/// `CURS` messages on the socket per frame. The latest wins; the frame clock
/// flushes it once per tick.
pub(crate) static PENDING_CURSOR: Mutex<Option<&'static str>> = Mutex::new(None);

/// Map a GPUI cursor style to the CSS name the consumer resolves. Kept in step
/// with the Linux and Windows producers' tables so every embed shows the same
/// pointer.
pub(crate) fn cursor_name(style: gpui::CursorStyle) -> &'static str {
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

pub(crate) fn send_title(socket: &EmbedSocket, title: &str) -> Result<()> {
    let bytes = title.as_bytes();
    let len = bytes.len().min(1024);
    let mut payload = Vec::with_capacity(8 + len);
    payload.extend_from_slice(b"TITL");
    payload.extend_from_slice(&(len as u32).to_le_bytes());
    payload.extend_from_slice(&bytes[..len]);
    socket.write_message(&payload)
}

pub(crate) fn send_clipboard(socket: &EmbedSocket, text: &str) -> Result<()> {
    let bytes = text.as_bytes();
    let mut payload = Vec::with_capacity(8 + bytes.len());
    payload.extend_from_slice(b"CLIP");
    payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    payload.extend_from_slice(bytes);
    socket.write_message(&payload)
}

/// The host's system clipboard, pushed on connect and whenever it changes, so
/// the embed's paste reads real system content rather than its own last copy.
pub(crate) static CLIPBOARD_CACHE: Mutex<Option<String>> = Mutex::new(None);

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_string(reader: &mut impl Read, len: usize) -> Result<String> {
    if len == 0 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Decode one input message. Tags match the Linux and Windows decoders byte for
/// byte.
fn read_input(reader: &mut impl Read) -> Result<EmbedInput> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        0x01 => Ok(EmbedInput::PointerMove {
            x: read_f32(reader)?,
            y: read_f32(reader)?,
        }),
        0x02 => {
            let mut bytes = [0u8; 2];
            reader.read_exact(&mut bytes)?;
            Ok(EmbedInput::Button {
                button: bytes[0],
                pressed: bytes[1] != 0,
            })
        }
        0x03 => {
            let mut header = [0u8; 3];
            reader.read_exact(&mut header)?;
            let pressed = header[0] != 0;
            let modifiers = header[1];
            let key = read_string(reader, header[2] as usize)?;
            let mut char_len = [0u8; 1];
            reader.read_exact(&mut char_len)?;
            let key_char = read_string(reader, char_len[0] as usize)?;
            let modifiers = Modifiers {
                control: modifiers & MOD_CTRL != 0,
                alt: modifiers & MOD_ALT != 0,
                shift: modifiers & MOD_SHIFT != 0,
                platform: modifiers & MOD_PLATFORM != 0,
                function: false,
            };
            // An empty key name means the consumer pressed or released a MODIFIER
            // on its own. That is not a keystroke in gpui's model — natively it
            // arrives as `ModifiersChanged`, and alt/ctrl/shift live in
            // `Modifiers` rather than being keys — so forwarding it as a KeyDown
            // named `alt_l` matched no binding and looked like the key was never
            // delivered at all. The length byte makes 0 a legal length, so this
            // needs no new tag.
            if key.is_empty() {
                return Ok(EmbedInput::ModifiersChanged { modifiers });
            }
            Ok(EmbedInput::Key {
                key,
                key_char: (!key_char.is_empty()).then_some(key_char),
                modifiers,
                pressed,
            })
        }
        0x04 => Ok(EmbedInput::Scroll {
            dx: read_f32(reader)?,
            dy: read_f32(reader)?,
        }),
        0x05 => Ok(EmbedInput::Resize {
            width: read_u32(reader)?,
            height: read_u32(reader)?,
        }),
        0x06 => {
            let mut colors = [0u32; 13];
            for color in colors.iter_mut() {
                *color = read_u32(reader)?;
            }
            Ok(EmbedInput::Palette { colors })
        }
        0x07 => {
            let len = read_u32(reader)? as usize;
            Ok(EmbedInput::Clipboard {
                text: read_string(reader, len)?,
            })
        }
        0x0B => Ok(EmbedInput::ScalePercent {
            pct: read_u32(reader)?,
        }),
        0x0C => Ok(EmbedInput::ScrollPixels {
            dx: read_f32(reader)?,
            dy: read_f32(reader)?,
        }),
        0x0E => Ok(EmbedInput::Display {
            display_id: read_u32(reader)?,
        }),
        other => bail!("unknown embed input tag {other:#04x}"),
    }
}

/// Reads the socket on a dedicated thread and forwards decoded input.
///
/// The socket cannot be serviced from the GPUI main thread: that thread is inside
/// `CFRunLoopRun`, and blocking it on a read would stop the run loop.
pub(crate) fn spawn_reader(mut reader: UnixStream) -> Receiver<EmbedInput> {
    let (sender, receiver): (Sender<EmbedInput>, Receiver<EmbedInput>) = channel();
    if let Err(error) = std::thread::Builder::new()
        .name("zed-embed-reader".into())
        .spawn(move || {
            loop {
                match read_input(&mut reader) {
                    Ok(input) => {
                        if sender.send(input).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        log::info!("[zed-embed] input stream ended: {error:#}");
                        break;
                    }
                }
            }
        })
    {
        // The sender goes with the closure, so the receiver reports a
        // disconnected consumer and the frame clock closes the window rather
        // than running a session that can never see input.
        log::error!("[zed-embed] spawning the input reader thread failed: {error}");
    }
    receiver
}

/// Button numbers are GTK/GDK's: 1 left, 2 middle, 3 right, 8 back, 9 forward
/// (the X11/evdev convention GDK exposes for the side buttons), because the
/// consumer's encoder is shared with the GTK client.
fn mouse_button(button: u8) -> MouseButton {
    match button {
        3 => MouseButton::Right,
        2 => MouseButton::Middle,
        8 => MouseButton::Navigate(NavigationDirection::Back),
        9 => MouseButton::Navigate(NavigationDirection::Forward),
        _ => MouseButton::Left,
    }
}

/// GPUI mouse events carry the current position and held button, but the wire
/// protocol reports moves and buttons separately — so the last of each has to be
/// remembered between messages.
#[derive(Default)]
pub(crate) struct EmbedInputState {
    position: Point<Pixels>,
    pressed_button: Option<MouseButton>,
}

impl EmbedInputState {
    /// Apply one decoded message to the window. Returns false when the message
    /// was consumed here (resize, palette, clipboard) rather than dispatched.
    pub(crate) fn apply(&mut self, input: EmbedInput, window: &EmbedWindow) -> bool {
        match input {
            EmbedInput::PointerMove { x, y } => {
                // Device pixels on the wire, logical points to gpui -- the same
                // conversion sizes get. Without it every click lands scale-times
                // too far from the pointer on a scaled display.
                let scale = window.scale();
                self.position = point(px(x / scale), px(y / scale));
                window.dispatch_input(PlatformInput::MouseMove(MouseMoveEvent {
                    position: self.position,
                    pressed_button: self.pressed_button,
                    modifiers: Modifiers::default(),
                }));
            }
            EmbedInput::Button { button, pressed } => {
                let button = mouse_button(button);
                if pressed {
                    self.pressed_button = Some(button);
                    window.dispatch_input(PlatformInput::MouseDown(MouseDownEvent {
                        button,
                        position: self.position,
                        modifiers: Modifiers::default(),
                        click_count: 1,
                        first_mouse: false,
                    }));
                } else {
                    self.pressed_button = None;
                    window.dispatch_input(PlatformInput::MouseUp(MouseUpEvent {
                        button,
                        position: self.position,
                        modifiers: Modifiers::default(),
                        click_count: 1,
                    }));
                }
            }
            EmbedInput::Scroll { dx, dy } => {
                // The consumer reports notches -- 1.0 per wheel click, fractions
                // of one from a precision touchpad -- and gpui takes the
                // fractional result as-is. Axes are negated: GTK's sign is the
                // opposite of GPUI's.
                window.dispatch_input(PlatformInput::ScrollWheel(ScrollWheelEvent {
                    position: self.position,
                    delta: ScrollDelta::Lines(point(-dx * SCROLL_LINES, -dy * SCROLL_LINES)),
                    modifiers: Modifiers::default(),
                    touch_phase: TouchPhase::Moved,
                }));
            }
            EmbedInput::ScrollPixels { dx, dy } => {
                // Pixels straight through: gpui scrolls by exactly this much, the
                // way a touchpad is meant to drive it. Divided by the scale for
                // the same reason pointer coordinates are -- the consumer measures
                // in device pixels. Axes negated, as with notch scrolling.
                let scale = window.scale();
                window.dispatch_input(PlatformInput::ScrollWheel(ScrollWheelEvent {
                    position: self.position,
                    delta: ScrollDelta::Pixels(point(px(-dx / scale), px(-dy / scale))),
                    modifiers: Modifiers::default(),
                    touch_phase: TouchPhase::Moved,
                }));
            }
            EmbedInput::ModifiersChanged { modifiers } => {
                window.dispatch_input(PlatformInput::ModifiersChanged(ModifiersChangedEvent {
                    modifiers,
                    capslock: Capslock { on: false },
                }));
                return false;
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
                // Just the keystroke: gpui commits the typed character itself for
                // embed sessions (`IS_EMBEDDED_INPUT`, keyed off
                // ZED_EMBED_SOCKET) because there is no OS IME to do it. Feeding
                // the input handler here as well types everything twice.
                if pressed {
                    window.dispatch_input(PlatformInput::KeyDown(KeyDownEvent {
                        keystroke,
                        is_held: false,
                        prefer_character_input: false,
                    }));
                } else {
                    window.dispatch_input(PlatformInput::KeyUp(KeyUpEvent { keystroke }));
                }
            }
            EmbedInput::Resize { width, height } => {
                window.resize_embed(width, height);
                return false;
            }
            EmbedInput::Palette { colors } => {
                gpui::embed_palette::set_embed_palette(gpui::embed_palette::EmbedPalette {
                    colors,
                });
                return false;
            }
            EmbedInput::Clipboard { text } => {
                *CLIPBOARD_CACHE.lock() = Some(text);
                return false;
            }
            EmbedInput::ScalePercent { pct } => {
                window.set_scale(pct as f32 / 100.0);
                return false;
            }
            EmbedInput::Display { display_id } => {
                crate::platform::retarget_embed_frame_clock(display_id);
                return false;
            }
        }
        true
    }
}

/// One embed process serves many host windows: the socket is bound once and its
/// accept loop pushes connections here; each `open_window` binds to the next one.
static PENDING_CONNECTIONS: Mutex<VecDeque<UnixStream>> = Mutex::new(VecDeque::new());
static EMBED_LISTENER_STARTED: OnceLock<()> = OnceLock::new();
/// Serializes enqueue of (project path -> gpui bridge, stream -> queue, counter)
/// so a connection's path and stream stay index-correlated across the two queues.
static EMBED_ENQUEUE: Mutex<()> = Mutex::new(());

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
    let length = u32::from_le_bytes(len) as usize;
    if length > 4096 {
        return Some(String::new());
    }
    let mut path = vec![0u8; length];
    if stream.read_exact(&mut path).is_err() {
        return Some(String::new());
    }
    Some(String::from_utf8_lossy(&path).into_owned())
}

/// Start the embed accept loop once. Each accepted connection is queued and the
/// app is notified (via the gpui connection counter) to open a window for it.
///
/// Called from `MacPlatform::new` rather than lazily from `open_window`, because
/// a window only opens once a connection has been accepted — starting the
/// listener there would be a deadlock.
pub(crate) fn ensure_embed_listener(socket_path: &str) {
    if EMBED_LISTENER_STARTED.set(()).is_err() {
        return;
    }
    let path = socket_path.to_owned();
    if let Err(error) = std::thread::Builder::new()
        .name("zed-embed-listener".into())
        .spawn(move || {
            // The host unlinks a stale socket before spawning us, but a previous
            // run that died without unbinding would otherwise make `bind` fail
            // with EADDRINUSE forever.
            if let Err(error) = std::fs::remove_file(&path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                log::warn!("[zed-embed] removing the stale socket {path}: {error}");
            }
            let listener = match UnixListener::bind(&path) {
                Ok(listener) => listener,
                Err(error) => {
                    log::error!("[zed-embed] failed to bind {path}: {error}");
                    return;
                }
            };
            log::info!("[zed-embed] listening on {path}");
            for connection in listener.incoming() {
                match connection {
                    Ok(mut stream) => {
                        // Read the project dir (first message) on a per-connection
                        // thread so a slow or absent cwd never stalls other
                        // accepts, then enqueue path + stream + counter atomically.
                        if let Err(error) = std::thread::Builder::new()
                            .name("zed-embed-accept".into())
                            .spawn(move || {
                                let Some(path) = read_embed_cwd(&mut stream) else {
                                    return;
                                };
                                let _guard = EMBED_ENQUEUE.lock();
                                gpui::embed_palette::push_embed_path(path);
                                PENDING_CONNECTIONS.lock().push_back(stream);
                                gpui::embed_palette::note_embed_connection();
                            })
                        {
                            log::error!("[zed-embed] spawning the accept thread failed: {error}");
                        }
                    }
                    Err(error) => log::error!("[zed-embed] accept failed: {error}"),
                }
            }
        })
    {
        log::error!("[zed-embed] spawning the listener thread failed: {error}");
    }
}

/// Pop the next queued connection, and the read half to service it with.
///
/// The app opens a window in response to the connection counter, so a stream
/// should already be queued; spin briefly to cover the race where `open_window`
/// runs just before the push completes.
pub(crate) fn take_pending_connection() -> Option<(EmbedSocket, UnixStream)> {
    for _ in 0..200 {
        let stream = PENDING_CONNECTIONS.lock().pop_front();
        if let Some(stream) = stream {
            let reader = match stream.try_clone().context("cloning the consumer socket") {
                Ok(reader) => reader,
                Err(error) => {
                    log::error!("[zed-embed] {error:#}");
                    return None;
                }
            };
            return Some((EmbedSocket::new(stream), reader));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host connects to this socket to find out whether we are accepting yet
    /// (macOS has no `/proc/net/unix` for it to read instead), so a connection
    /// that says nothing must not consume a window. A client that sends a
    /// zero-length path, or something that isn't `CWD ` at all, is still a
    /// consumer.
    #[test]
    fn a_probe_is_not_a_consumer_but_a_pathless_client_is() {
        let (mut producer, consumer) = UnixStream::pair().expect("creating a socket pair");
        drop(consumer);
        assert_eq!(read_embed_cwd(&mut producer), None);

        let (mut producer, mut consumer) = UnixStream::pair().expect("creating a socket pair");
        consumer.write_all(b"CWD ").expect("writing the magic");
        consumer
            .write_all(&12u32.to_le_bytes())
            .expect("writing the length");
        consumer.write_all(b"/home/velzie").expect("writing a path");
        assert_eq!(
            read_embed_cwd(&mut producer),
            Some("/home/velzie".to_owned())
        );

        let (mut producer, mut consumer) = UnixStream::pair().expect("creating a socket pair");
        consumer.write_all(b"CWD ").expect("writing the magic");
        consumer
            .write_all(&0u32.to_le_bytes())
            .expect("writing the length");
        assert_eq!(read_embed_cwd(&mut producer), Some(String::new()));

        let (mut producer, mut consumer) = UnixStream::pair().expect("creating a socket pair");
        consumer.write_all(b"RFB \x00").expect("writing a greeting");
        assert_eq!(read_embed_cwd(&mut producer), Some(String::new()));
    }

    /// Tag `0x0E` exists nowhere but here and in the macOS consumer's
    /// `send_display`, so unlike every other input tag there is no third
    /// implementation to catch a disagreement about it. These are the exact bytes
    /// that encoder writes.
    #[test]
    fn a_display_id_decodes_from_the_bytes_the_consumer_writes() {
        let mut bytes: &[u8] = &[0x0E, 0x00, 0x04, 0x28, 0x04];
        match read_input(&mut bytes).expect("decoding a display id") {
            EmbedInput::Display { display_id } => assert_eq!(display_id, 0x0428_0400),
            _ => panic!("tag 0x0E decoded as something other than a display id"),
        }
    }
}
