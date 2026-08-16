//! Host-palette bridge for headless embedding.
//!
//! When a gpui app is embedded in a host application (e.g. slop2) and rendered
//! headless, the host can feed its color palette in so the app themes itself to
//! match. The platform layer writes the palette here (received over the embed
//! socket); the app polls [`embed_palette`] and rebuilds its theme when the
//! version changes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// Host palette: 13 colors packed as `0xRRGGBBAA`. Order matches slop2's
/// `Palette`: fg, fg_dim, fg_muted, fg_subtle, bg, bg_alt, accent, accent_alt,
/// purple, red, green, yellow, yellow_bright. The alpha byte of `bg`/`bg_alt`
/// carries the host's window transparency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbedPalette {
    /// The packed colors, in the order given above.
    pub colors: [u32; 13],
}

static PALETTE: Mutex<Option<EmbedPalette>> = Mutex::new(None);
static VERSION: AtomicU64 = AtomicU64::new(0);

/// Store a new host palette (called by the platform layer). Bumps the version
/// unless the palette is unchanged: hosts resend it whenever they restyle
/// themselves, and a version bump costs the app a full theme rebuild.
pub fn set_embed_palette(palette: EmbedPalette) {
    let mut current = PALETTE.lock();
    if *current == Some(palette) {
        return;
    }
    *current = Some(palette);
    VERSION.fetch_add(1, Ordering::Release);
}

/// The current host palette and its monotonic version, if one has been set.
/// The version changes whenever `set_embed_palette` receives new colors, so a
/// poller can detect updates without comparing all colors.
pub fn embed_palette() -> Option<(u64, EmbedPalette)> {
    let version = VERSION.load(Ordering::Acquire);
    (*PALETTE.lock()).map(|palette| (version, palette))
}

/// Total embed connections accepted so far. The platform bumps this per new
/// host connection; the app polls it and opens one window per connection.
static CONNECTION_COUNT: AtomicU64 = AtomicU64::new(0);

/// Record a newly-accepted embed connection; returns the new total.
pub fn note_embed_connection() -> u64 {
    CONNECTION_COUNT.fetch_add(1, Ordering::Release) + 1
}

/// The number of embed connections accepted so far.
pub fn embed_connection_count() -> u64 {
    CONNECTION_COUNT.load(Ordering::Acquire)
}

/// Per-connection project directory (the host's cwd when `embed` was typed),
/// queued in the same order as connections so the app can open the right dir.
/// An empty string means "no directory" (open a blank workspace).
static PENDING_PATHS: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// Queue the project dir for the next connection (platform side).
pub fn push_embed_path(path: String) {
    PENDING_PATHS.lock().push_back(path);
}

/// Take the project dir for the next connection to open (app side).
pub fn take_embed_path() -> Option<String> {
    PENDING_PATHS.lock().pop_front()
}
