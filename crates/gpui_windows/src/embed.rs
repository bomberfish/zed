//! Embed sessions: render Zed into a host application's window.
//!
//! The Windows counterpart to `gpui_linux`'s headless embed client. The wire
//! format is byte-identical to the Linux one except for the handshake, because
//! the two platforms hand over the frame buffer differently: Linux passes DRM
//! PRIME fds as `SCM_RIGHTS` ancillary data, while here the producer duplicates
//! an NT handle into the consumer process (`DuplicateHandle`) and sends the
//! duplicate inline.
//!
//! Reads and writes use an overlapped pipe. A synchronous handle serializes all
//! I/O on the file object, so a blocking read on the reader thread would stall
//! every frame notification behind it.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use anyhow::{Context as _, Result, bail};
use gpui::{
    Capslock, KeyDownEvent, KeyUpEvent, Keystroke, Modifiers, ModifiersChangedEvent, MouseButton,
    MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, NavigationDirection, Pixels, PlatformInput, Point, ScrollDelta,
    ScrollWheelEvent, TouchPhase, point, px,
};
use parking_lot::Mutex;

use crate::embed_window::EmbedWindow;
use windows::Win32::Foundation::{CloseHandle, ERROR_IO_PENDING, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Storage::FileSystem::{
    FILE_FLAG_OVERLAPPED, PIPE_ACCESS_DUPLEX, ReadFile, WriteFile,
};
use windows::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeClientProcessId, PIPE_READMODE_BYTE,
    PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    SPI_GETWHEELSCROLLLINES, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SystemParametersInfoW,
};
use windows::Win32::System::Diagnostics::Debug::{
    EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS, SetUnhandledExceptionFilter,
};
use windows::Win32::System::Threading::{
    CreateEventW, INFINITE, OpenProcess, PROCESS_DUP_HANDLE, WaitForSingleObject,
};
use windows::core::HSTRING;

/// Lines per wheel notch, from the user's Windows setting -- the same value
/// `handle_mouse_wheel_msg` uses for a real window, so an embedded editor scrolls
/// like every other one. Hardcoding 3 (as the unix client does) ignores the
/// setting entirely.
fn wheel_scroll_lines() -> f32 {
    let mut lines: u32 = 3;
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETWHEELSCROLLLINES,
            0,
            Some(&mut lines as *mut u32 as *mut std::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS::default(),
        )
    };
    if ok.is_err() || lines == 0 {
        return 3.0;
    }
    lines as f32
}

/// Environment variable naming the endpoint to serve. Its presence is what puts
/// this process into embed mode.
///
/// The same variable on every platform even though the value is a pipe name here
/// and a socket path on unix: gpui, workspace, and zed's main all gate embed
/// behaviour on this name (dropping the title bar, applying the host palette,
/// skipping the boot workspace, committing typed characters). A second,
/// Windows-only name would leave every one of those silently switched off.
pub(crate) const EMBED_ENDPOINT_VAR: &str = "ZED_EMBED_SOCKET";

const PIPE_BUFFER: u32 = 512 * 1024;

// Modifier bitfield shared with slop2's `EmbedSurface` encoder.
const MOD_CTRL: u8 = 1 << 0;
const MOD_ALT: u8 = 1 << 1;
const MOD_SHIFT: u8 = 1 << 2;
const MOD_PLATFORM: u8 = 1 << 3;

/// Input forwarded by the consumer. Mirrors the Linux `EmbedInput` exactly; the
/// consumer owns keyval→name translation because it is the side holding GDK's
/// tables.
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
}

/// An overlapped named-pipe server, split so the reader thread and the main
/// thread can use it concurrently.
///
/// Each direction owns its own event: overlapped I/O signals completion through
/// the event in its `OVERLAPPED`, so sharing one between a concurrent read and
/// write would let each consume the other's completion.
pub(crate) struct EmbedPipe {
    handle: HANDLE,
    read_event: HANDLE,
    write_event: HANDLE,
}

// The handle is used from the reader thread and the main thread, which is safe
// precisely because the pipe is overlapped and the two directions are
// independent.
unsafe impl Send for EmbedPipe {}
unsafe impl Sync for EmbedPipe {}

impl EmbedPipe {
    /// Create the server end and block until a consumer connects.
    pub(crate) fn serve(name: &str) -> Result<Self> {
        let pipe = Self::create(name)?;
        pipe.accept()?;
        Ok(pipe)
    }

    /// Create one instance of the server pipe, not yet connected.
    ///
    /// `PIPE_UNLIMITED_INSTANCES`, because one zed serves every pane: each
    /// consumer gets its own instance and its own window, mirroring the unix
    /// accept loop. A single-instance pipe means the second pane hangs forever.
    pub(crate) fn create(name: &str) -> Result<Self> {
        let wide = HSTRING::from(name);
        let handle = unsafe {
            CreateNamedPipeW(
                &wide,
                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                PIPE_UNLIMITED_INSTANCES,
                PIPE_BUFFER,
                PIPE_BUFFER,
                0,
                None,
            )
        };
        if handle.is_invalid() {
            bail!(
                "CreateNamedPipeW({name}) failed: {}",
                std::io::Error::last_os_error()
            );
        }

        let read_event = unsafe { CreateEventW(None, true, false, None) }
            .context("creating the pipe read event")?;
        let write_event = unsafe { CreateEventW(None, true, false, None) }
            .context("creating the pipe write event")?;
        Ok(Self {
            handle,
            read_event,
            write_event,
        })
    }

    /// Block until a consumer connects to this instance.
    pub(crate) fn accept(&self) -> Result<()> {
        let mut overlapped = OVERLAPPED {
            hEvent: self.read_event,
            ..Default::default()
        };
        // ConnectNamedPipe on an overlapped pipe returns FALSE with
        // ERROR_IO_PENDING for the normal case; a consumer that connected in the
        // window between create and connect surfaces as ERROR_PIPE_CONNECTED.
        let connected = unsafe { ConnectNamedPipe(self.handle, Some(&mut overlapped)) };
        if let Err(error) = connected {
            const ERROR_PIPE_CONNECTED_CODE: i32 = 535;
            if error.code().0 & 0xffff == ERROR_IO_PENDING.0 as i32 {
                let mut transferred = 0u32;
                unsafe {
                    GetOverlappedResult(self.handle, &overlapped, &mut transferred, true)
                        .context("waiting for the embed consumer to connect")?
                };
            } else if error.code().0 & 0xffff != ERROR_PIPE_CONNECTED_CODE {
                return Err(error).context("ConnectNamedPipe");
            }
        }
        Ok(())
    }

    /// The consumer's process, opened for handle duplication.
    ///
    /// Taken from the pipe rather than exchanged in the protocol: the server end
    /// already knows who connected, so there is no window in which a wrong or
    /// stale pid could be supplied.
    pub(crate) fn consumer_process(&self) -> Result<HANDLE> {
        let mut pid = 0u32;
        unsafe { GetNamedPipeClientProcessId(self.handle, &mut pid) }
            .context("GetNamedPipeClientProcessId")?;
        unsafe { OpenProcess(PROCESS_DUP_HANDLE, false, pid) }
            .with_context(|| format!("opening the consumer process {pid} for handle duplication"))
    }

    /// Resolve one overlapped operation.
    ///
    /// An overlapped `ReadFile`/`WriteFile` reports "started, not finished" as a
    /// failure with `ERROR_IO_PENDING`; anything else is a genuine error and
    /// must not be waited on, or the wait blocks forever on an event that will
    /// never be signalled.
    fn wait(
        &self,
        overlapped: &mut OVERLAPPED,
        event: HANDLE,
        started: windows::core::Result<()>,
    ) -> Result<u32> {
        if let Err(error) = started
            && error.code().0 & 0xffff != ERROR_IO_PENDING.0 as i32
        {
            return Err(error).context("starting overlapped pipe I/O");
        }
        if unsafe { WaitForSingleObject(event, INFINITE) } != WAIT_OBJECT_0 {
            bail!("waiting on overlapped pipe I/O failed");
        }
        let mut transferred = 0u32;
        unsafe { GetOverlappedResult(self.handle, overlapped, &mut transferred, false) }
            .context("GetOverlappedResult")?;
        Ok(transferred)
    }

    pub(crate) fn read_exact(&self, buffer: &mut [u8]) -> Result<()> {
        let mut filled = 0usize;
        while filled < buffer.len() {
            let mut overlapped = OVERLAPPED {
                hEvent: self.read_event,
                ..Default::default()
            };
            let result = unsafe {
                ReadFile(
                    self.handle,
                    Some(&mut buffer[filled..]),
                    None,
                    Some(&mut overlapped),
                )
            };
            let read = self.wait(&mut overlapped, self.read_event, result)?;
            if read == 0 {
                bail!("embed consumer disconnected");
            }
            filled += read as usize;
        }
        Ok(())
    }

    pub(crate) fn write_all(&self, buffer: &[u8]) -> Result<()> {
        let mut written = 0usize;
        while written < buffer.len() {
            let mut overlapped = OVERLAPPED {
                hEvent: self.write_event,
                ..Default::default()
            };
            let result = unsafe {
                WriteFile(
                    self.handle,
                    Some(&buffer[written..]),
                    // None: with an OVERLAPPED the count comes from
                    // GetOverlappedResult, and passing both is invalid.
                    None,
                    Some(&mut overlapped),
                )
            };
            let sent = self.wait(&mut overlapped, self.write_event, result)?;
            if sent == 0 {
                bail!("embed consumer disconnected");
            }
            written += sent as usize;
        }
        Ok(())
    }
}

impl Drop for EmbedPipe {
    fn drop(&mut self) {
        unsafe {
            for handle in [self.handle, self.read_event, self.write_event] {
                if !handle.is_invalid()
                    && let Err(error) = CloseHandle(handle)
                {
                    log::error!("closing an embed pipe handle: {error}");
                }
            }
        }
    }
}

/// Log a stack trace when the process dies from an OS-level exception.
///
/// An access violation is not a Rust panic: nothing is printed, no hook runs,
/// and the process simply disappears -- which makes a fault inside the D3D11/
/// 11on12 interop layer invisible. This is the last code to run before that
/// happens, so it is the only place a stack can be captured.
///
/// Returns `EXCEPTION_CONTINUE_SEARCH`: the crash still happens, it is only
/// no longer silent.
unsafe extern "system" fn crash_filter(info: *const EXCEPTION_POINTERS) -> i32 {
    let (code, address, pc, lr, sp) = unsafe {
        let record = (*info).ExceptionRecord;
        let context = (*info).ContextRecord;
        (
            (*record).ExceptionCode.0,
            (*record).ExceptionAddress as usize,
            (*context).Pc as usize,
            // x30 is the link register: when the fault is a jump to a bad
            // address, the faulting frame itself is unwindable garbage but the
            // *caller* is still here, which is the only thing that identifies
            // the bad call.
            (*context).Anonymous.X[30] as usize,
            (*context).Sp as usize,
        )
    };
    let base = unsafe { GetModuleHandleW(None) }
        .map(|module| module.0 as usize)
        .unwrap_or(0);
    // Offsets, not absolute addresses: ASLR moves the image every run, and only
    // the offset can be fed back to llvm-symbolizer against zed.pdb.
    let offset = |value: usize| value.wrapping_sub(base);
    eprintln!(
        "[zed-embed] fatal exception {code:#010x} at {address:#x}\n\
         base={base:#x} pc={pc:#x} (+{:#x}) lr={lr:#x} (+{:#x}) sp={sp:#x}",
        offset(pc),
        offset(lr),
    );
    EXCEPTION_CONTINUE_SEARCH
}

/// Install [`crash_filter`]. Idempotent enough for one call at startup.
pub(crate) fn install_crash_filter() {
    unsafe { SetUnhandledExceptionFilter(Some(crash_filter)) };
}

/// The consumer's opening message: `"CWD " + u32 len + path`, naming the project
/// the window should open. It arrives before anything else, so it has to be
/// consumed before the input reader starts or its bytes are decoded as input.
///
/// A different magic is not an error: it means a consumer that names no project,
/// and the four bytes it did send are already spent either way.
pub(crate) fn read_cwd(pipe: &EmbedPipe) -> Result<String> {
    let mut magic = [0u8; 4];
    pipe.read_exact(&mut magic)?;
    if &magic != b"CWD " {
        return Ok(String::new());
    }
    let len = read_u32(pipe)? as usize;
    if len > 4096 {
        return Ok(String::new());
    }
    read_string(pipe, len)
}

/// The first byte the consumer expects: which transport the frames will use.
/// It reads this before any tagged message, so sending the handshake without it
/// leaves the stream misaligned by one byte.
pub(crate) fn send_mode(pipe: &EmbedPipe, zero_copy: bool) -> Result<()> {
    pipe.write_all(&[zero_copy as u8])
}

/// Handshake (once): the duplicated shared-texture handle and the target's
/// dimensions.
///
/// `SHTX` rather than Linux's `DMAB` because the payload is a different kind of
/// thing — one shareable D3D12 resource, not a dma-buf plane with stride and
/// modifier.
/// `SHTX` + handle u64 + width u32 + height u32 — the first buffer and the size.
///
/// Deliberately unchanged at 16 bytes: the consumer reads this body at a fixed
/// length with no length prefix, so appending fields here would simply block a
/// consumer that predates them. The second buffer and the fence travel as their
/// own message instead.
pub(crate) fn send_handshake(pipe: &EmbedPipe, handle: isize, width: u32, height: u32) -> Result<()> {
    let mut payload = Vec::with_capacity(20);
    payload.extend_from_slice(b"SHTX");
    payload.extend_from_slice(&(handle as u64).to_le_bytes());
    payload.extend_from_slice(&width.to_le_bytes());
    payload.extend_from_slice(&height.to_le_bytes());
    pipe.write_all(&payload)
}

/// `SHT2` + handle u64 + fence u64, sent immediately after every `SHTX`.
///
/// The second of the two targets frames alternate between, and the fence their
/// completion is signalled on. `fence` is 0 on a re-handshake after a resize:
/// the targets are reallocated but the fence is not, and re-sharing it would
/// leak a handle per resize for no gain.
pub(crate) fn send_shared_extra(pipe: &EmbedPipe, handle: isize, fence: isize) -> Result<()> {
    let mut payload = Vec::with_capacity(20);
    payload.extend_from_slice(b"SHT2");
    payload.extend_from_slice(&(handle as u64).to_le_bytes());
    payload.extend_from_slice(&(fence as u64).to_le_bytes());
    pipe.write_all(&payload)
}

/// The fence value the frame announced by the next `RDY!` completes at.
///
/// Sent as its own message immediately before it, rather than as an extra `RDY!`
/// field, so `RDY!` stays byte-identical to the Linux one and the consumer keeps
/// a single parser for both platforms.
pub(crate) fn send_fence_value(pipe: &EmbedPipe, value: u64) -> Result<()> {
    let mut payload = Vec::with_capacity(12);
    payload.extend_from_slice(b"FNCV");
    payload.extend_from_slice(&value.to_le_bytes());
    pipe.write_all(&payload)
}

/// Per-frame notification: which buffer, and the active sub-rect of it to sample.
pub(crate) fn send_ready(pipe: &EmbedPipe, index: u32, width: u32, height: u32) -> Result<()> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(b"RDY!");
    payload.extend_from_slice(&index.to_le_bytes());
    payload.extend_from_slice(&width.to_le_bytes());
    payload.extend_from_slice(&height.to_le_bytes());
    pipe.write_all(&payload)
}

/// Cursor style change: `CURS` + u8 name length + CSS name, byte-identical to
/// the Linux producer's. The consumer resolves it with `gdk::Cursor::from_name`.
pub(crate) fn send_cursor(pipe: &EmbedPipe, name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    let len = bytes.len().min(64);
    let mut payload = Vec::with_capacity(5 + len);
    payload.extend_from_slice(b"CURS");
    payload.push(len as u8);
    payload.extend_from_slice(&bytes[..len]);
    pipe.write_all(&payload)
}

/// The cursor the most recent `set_cursor_style` asked for.
///
/// GPUI sets the cursor several times per paint — a default reset, then the
/// hovered element's real one — so forwarding every call would put a burst of
/// `CURS` messages on the pipe per frame. The latest wins; the frame pump flushes
/// it once per tick. Same design as the Linux producer.
pub(crate) static PENDING_CURSOR: Mutex<Option<&'static str>> = Mutex::new(None);

/// Map a GPUI cursor style to the CSS name the consumer resolves. Kept in step
/// with the Linux producer's table so both embeds show the same pointer.
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

pub(crate) fn send_title(pipe: &EmbedPipe, title: &str) -> Result<()> {
    let bytes = title.as_bytes();
    let len = bytes.len().min(1024);
    let mut payload = Vec::with_capacity(8 + len);
    payload.extend_from_slice(b"TITL");
    payload.extend_from_slice(&(len as u32).to_le_bytes());
    payload.extend_from_slice(&bytes[..len]);
    pipe.write_all(&payload)
}

pub(crate) fn send_clipboard(pipe: &EmbedPipe, text: &str) -> Result<()> {
    let bytes = text.as_bytes();
    let mut payload = Vec::with_capacity(8 + bytes.len());
    payload.extend_from_slice(b"CLIP");
    payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    payload.extend_from_slice(bytes);
    pipe.write_all(&payload)
}

fn read_u32(pipe: &EmbedPipe) -> Result<u32> {
    let mut bytes = [0u8; 4];
    pipe.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(pipe: &EmbedPipe) -> Result<f32> {
    let mut bytes = [0u8; 4];
    pipe.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_string(pipe: &EmbedPipe, len: usize) -> Result<String> {
    if len == 0 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; len];
    pipe.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Decode one input message. Tags match the Linux decoder byte for byte.
pub(crate) fn read_input(pipe: &EmbedPipe) -> Result<EmbedInput> {
    let mut tag = [0u8; 1];
    pipe.read_exact(&mut tag)?;
    match tag[0] {
        0x01 => Ok(EmbedInput::PointerMove {
            x: read_f32(pipe)?,
            y: read_f32(pipe)?,
        }),
        0x02 => {
            let mut bytes = [0u8; 2];
            pipe.read_exact(&mut bytes)?;
            Ok(EmbedInput::Button {
                button: bytes[0],
                pressed: bytes[1] != 0,
            })
        }
        0x03 => {
            let mut header = [0u8; 3];
            pipe.read_exact(&mut header)?;
            let pressed = header[0] != 0;
            let modifiers = header[1];
            let key = read_string(pipe, header[2] as usize)?;
            let mut char_len = [0u8; 1];
            pipe.read_exact(&mut char_len)?;
            let key_char = read_string(pipe, char_len[0] as usize)?;
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
            // named `alt_l` (which is what the consumer used to send, from
            // `keyval.name()`) matched no binding and looked like the key was
            // never delivered at all. The length byte makes 0 a legal length, so
            // this needs no new tag.
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
            dx: read_f32(pipe)?,
            dy: read_f32(pipe)?,
        }),
        0x05 => Ok(EmbedInput::Resize {
            width: read_u32(pipe)?,
            height: read_u32(pipe)?,
        }),
        0x06 => {
            let mut colors = [0u32; 13];
            for color in colors.iter_mut() {
                *color = read_u32(pipe)?;
            }
            Ok(EmbedInput::Palette { colors })
        }
        0x07 => {
            let len = read_u32(pipe)? as usize;
            Ok(EmbedInput::Clipboard {
                text: read_string(pipe, len)?,
            })
        }
        0x0B => Ok(EmbedInput::ScalePercent { pct: read_u32(pipe)? }),
        0x0C => Ok(EmbedInput::ScrollPixels {
            dx: read_f32(pipe)?,
            dy: read_f32(pipe)?,
        }),
        other => bail!("unknown embed input tag {other:#04x}"),
    }
}

/// Reads the pipe on a dedicated thread and forwards decoded input.
///
/// The pipe cannot be serviced from the GPUI main thread: that thread is inside
/// `GetMessageW`, and blocking it on a read would stop the message loop.
pub(crate) fn spawn_reader(pipe: Arc<EmbedPipe>) -> Receiver<EmbedInput> {
    let (sender, receiver): (Sender<EmbedInput>, Receiver<EmbedInput>) = channel();
    std::thread::Builder::new()
        .name("zed-embed-reader".into())
        .spawn(move || {
            loop {
                match read_input(&pipe) {
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
        .expect("spawning the embed reader thread");
    receiver
}

/// Button numbers are GTK/GDK's: 1 left, 2 middle, 3 right, 8 back, 9 forward
/// (the X11/evdev convention GDK exposes for the side buttons).
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
                // of one from a precision touchpad -- so this is the same
                // arithmetic a real window does in `handle_mouse_wheel_msg`, and
                // gpui takes the fractional result as-is.
                let lines = wheel_scroll_lines();
                // Axes are negated: GTK's sign is the opposite of GPUI's.
                window.dispatch_input(PlatformInput::ScrollWheel(ScrollWheelEvent {
                    position: self.position,
                    delta: ScrollDelta::Lines(point(-dx * lines, -dy * lines)),
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
                window.dispatch_input(PlatformInput::ModifiersChanged(
                    ModifiersChangedEvent {
                        modifiers,
                        capslock: Capslock { on: false },
                    },
                ));
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
                // Just the keystroke: gpui commits the typed character itself
                // for embed sessions (`IS_EMBEDDED_INPUT`, keyed off
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
        }
        true
    }
}

/// The host's system clipboard, pushed on connect and whenever it changes, so
/// the embed's paste reads real system content rather than its own last copy.
pub(crate) static CLIPBOARD_CACHE: Mutex<Option<String>> = Mutex::new(None);
