// Synthetic-workload tests for the damage-tracked render planner.
//
// Each scenario feeds a fake VT byte stream (mimicking a TUI app, an
// ssh typing session, a scrolling build log, …) through a real
// `alacritty_terminal` Term + parser, then simulates the coalesced
// redraw pass the GTK layer performs: collect damage → hand it to the
// `RowPlanner` → count how many rows would actually be re-rasterized.
//
// The assertions encode the efficiency contract: typing repaints ≤2
// rows, a TUI tick repaints only the rows it touched, a no-op TUI
// refresh repaints nothing, and a scrolling firehose repaints only the
// newly appended rows regardless of how much output arrived.

use alacritty_terminal::event::VoidListener;
use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::index::{Column, Line, Point};
use alacritty_terminal::term::test::TermSize;
use alacritty_terminal::term::{Config as TermConfig, Term, TermDamage};
use alacritty_terminal::vte::ansi::{Color as AlacColor, Processor, StdSyncHandler};

use slop2::grid_cache::{hash_mix, hash_rgb, RowPlanner};

struct Sim {
    term: Term<VoidListener>,
    parser: Processor<StdSyncHandler>,
    planner: RowPlanner,
    cols: usize,
    rows: usize,
}

impl Sim {
    fn new(cols: usize, rows: usize) -> Self {
        let term = Term::new(
            TermConfig {
                scrolling_history: 10_000,
                ..Default::default()
            },
            &TermSize::new(cols, rows),
            VoidListener,
        );
        Self {
            term,
            parser: Processor::new(),
            planner: RowPlanner::new(),
            cols,
            rows,
        }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.parser.advance(&mut self.term, bytes);
    }

    fn feed_str(&mut self, s: &str) {
        self.feed(s.as_bytes());
    }

    /// One coalesced redraw pass, viewport pinned to the bottom (the
    /// common case for a terminal). Returns the render rows that would
    /// be re-rasterized.
    fn frame(&mut self) -> Vec<i64> {
        // Damage collection, exactly as perform_session_redraw does it.
        let history = self.term.grid().history_size();
        let mut full = false;
        let mut lines: Vec<usize> = Vec::new();
        match self.term.damage() {
            TermDamage::Full => full = true,
            TermDamage::Partial(iter) => lines.extend(iter.map(|b| b.line)),
        }
        self.term.reset_damage();
        if full {
            self.planner.add_damage_full();
        } else {
            self.planner.add_damage_rows(lines, history);
        }

        // Visible range: the last `rows` rows of render space
        // (history + screen), like a viewport scrolled to the bottom.
        let total = history as i64 + self.rows as i64;
        let first = (total - self.rows as i64).max(0);
        let count = (total - first) as usize;

        let grid = self.term.grid();
        let cols = self.cols;
        let history_i64 = history as i64;
        let plan = self.planner.begin_frame(
            1,
            first,
            count,
            history,
            None,
            Some(self.term.colors()),
            |row| {
                let line = Line((row - history_i64) as i32);
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for col in 0..cols {
                    let cell = &grid[line][Column(col)];
                    h = hash_mix(h, cell.c as u64);
                    h = hash_mix(h, cell.flags.bits() as u64);
                    h = hash_color(h, cell.fg);
                    h = hash_color(h, cell.bg);
                }
                h
            },
        );
        plan.repaint
    }

    /// Read a screen row back as a trimmed string (sanity checks).
    fn row_text(&self, line: i32) -> String {
        let grid = self.term.grid();
        (0..self.cols)
            .map(|c| grid[Line(line)][Column(c)].c)
            .collect::<String>()
            .trim_end()
            .to_string()
    }
}

fn hash_color(h: u64, color: AlacColor) -> u64 {
    match color {
        AlacColor::Spec(rgb) => hash_rgb(hash_mix(h, 1), rgb),
        AlacColor::Named(n) => hash_mix(hash_mix(h, 2), n as u64),
        AlacColor::Indexed(i) => hash_mix(hash_mix(h, 3), i as u64),
    }
}

// ---------------------------------------------------------------------------
// ssh / interactive shell typing
// ---------------------------------------------------------------------------

#[test]
fn typing_echo_repaints_at_most_two_rows() {
    let mut sim = Sim::new(120, 24);
    sim.feed_str("user@host:~$ ");
    sim.frame(); // settle

    for ch in "cargo build --release".chars() {
        sim.feed_str(&ch.to_string());
        let repaint = sim.frame();
        assert!(
            repaint.len() <= 2,
            "one echoed keystroke repainted {} rows: {repaint:?}",
            repaint.len()
        );
    }
}

#[test]
fn idle_frames_repaint_nothing() {
    let mut sim = Sim::new(120, 24);
    sim.feed_str("user@host:~$ ls\r\nsrc  target  Cargo.toml\r\nuser@host:~$ ");
    sim.frame();

    // Blink ticks / trail frames / expose events: no damage, no work.
    for _ in 0..10 {
        let repaint = sim.frame();
        assert!(repaint.is_empty(), "idle frame repainted {repaint:?}");
    }
}

// ---------------------------------------------------------------------------
// TUI apps (vim/htop-style alt-screen updates)
// ---------------------------------------------------------------------------

/// Paint a full-screen TUI frame: status bar + numbered content rows.
fn tui_full_frame(sim: &mut Sim, tick: u64) {
    sim.feed_str("\x1b[H"); // cursor home
    for row in 0..sim.rows {
        // Move to the row start and rewrite it.
        sim.feed_str(&format!("\x1b[{};1H", row + 1));
        if row == 0 {
            sim.feed_str(&format!("\x1b[7m htop-sim  tick {tick:<8}\x1b[27m"));
        } else {
            sim.feed_str(&format!("proc {row:>4}  cpu {:>3}%", (row as u64 * 7) % 100));
        }
        sim.feed_str("\x1b[K");
    }
}

#[test]
fn tui_cursor_motion_repaints_moved_rows_only() {
    let mut sim = Sim::new(120, 24);
    sim.feed_str("\x1b[?1049h"); // enter alt screen
    tui_full_frame(&mut sim, 0);
    sim.frame();

    // vim-style: move the cursor around without changing content.
    sim.feed_str("\x1b[10;5H");
    let repaint = sim.frame();
    assert!(
        repaint.len() <= 2,
        "pure cursor motion repainted {} rows: {repaint:?}",
        repaint.len()
    );
}

#[test]
fn tui_partial_update_repaints_touched_rows_only() {
    let mut sim = Sim::new(120, 24);
    sim.feed_str("\x1b[?1049h");
    tui_full_frame(&mut sim, 0);
    sim.frame();

    // htop updates three process rows in place.
    sim.feed_str("\x1b[5;1Hproc    4  cpu  93%\x1b[K");
    sim.feed_str("\x1b[9;1Hproc    8  cpu  12%\x1b[K");
    sim.feed_str("\x1b[13;1Hproc   12  cpu  55%\x1b[K");
    let repaint = sim.frame();
    assert!(
        repaint.len() <= 5,
        "3-row TUI update repainted {} rows: {repaint:?}",
        repaint.len()
    );
    assert!(
        repaint.len() >= 3,
        "3-row TUI update must repaint the touched rows, got {repaint:?}"
    );
}

#[test]
fn tui_identical_refresh_repaints_nothing() {
    // Many TUIs redraw the entire screen every tick even when nothing
    // changed. The term marks all those cells damaged; the row hashes
    // must recognize the content is identical and skip the raster.
    let mut sim = Sim::new(120, 24);
    sim.feed_str("\x1b[?1049h");
    tui_full_frame(&mut sim, 7);
    sim.frame();

    tui_full_frame(&mut sim, 7); // identical redraw
    let repaint = sim.frame();
    assert!(
        repaint.is_empty(),
        "identical TUI refresh repainted {} rows: {repaint:?}",
        repaint.len()
    );
}

#[test]
fn tui_tick_repaints_only_changed_rows() {
    let mut sim = Sim::new(120, 24);
    sim.feed_str("\x1b[?1049h");
    tui_full_frame(&mut sim, 1);
    sim.frame();

    // Next tick: only the status bar text (row 0) actually differs.
    tui_full_frame(&mut sim, 2);
    let repaint = sim.frame();
    assert!(
        repaint.len() <= 2,
        "single-row TUI tick repainted {} rows: {repaint:?}",
        repaint.len()
    );
}

// ---------------------------------------------------------------------------
// Scrolling output (build logs, cat, ssh burst)
// ---------------------------------------------------------------------------

#[test]
fn scrolling_log_repaints_only_new_rows() {
    let mut sim = Sim::new(120, 24);
    for i in 0..24 {
        sim.feed_str(&format!("build line {i}\r\n"));
    }
    sim.frame();

    // 5 more lines arrive before the next coalesced redraw.
    for i in 24..29 {
        sim.feed_str(&format!("build line {i}\r\n"));
    }
    let repaint = sim.frame();
    assert!(
        repaint.len() <= 7,
        "5 new lines repainted {} rows: {repaint:?}",
        repaint.len()
    );
    // Sanity: the content actually scrolled into history.
    assert!(sim.term.grid().history_size() >= 5);
}

#[test]
fn coalesced_firehose_repaints_at_most_the_viewport() {
    let mut sim = Sim::new(120, 24);
    sim.frame();

    // 10k lines land between two redraw passes (throttled pump).
    for i in 0..10_000 {
        sim.feed_str(&format!("log line {i}\r\n"));
    }
    let repaint = sim.frame();
    assert!(
        repaint.len() <= 24,
        "firehose frame repainted {} rows (> viewport)",
        repaint.len()
    );
}

#[test]
fn steady_scroll_amortizes_to_new_rows_per_frame() {
    let mut sim = Sim::new(120, 24);
    for i in 0..50 {
        sim.feed_str(&format!("warmup {i}\r\n"));
    }
    sim.frame();

    let mut total_repainted = 0usize;
    let frames = 30usize;
    let lines_per_frame = 3usize;
    for f in 0..frames {
        for l in 0..lines_per_frame {
            sim.feed_str(&format!("steady {f}-{l}\r\n"));
        }
        total_repainted += sim.frame().len();
    }
    // Ideal is lines_per_frame(+cursor row) per frame; allow slack but
    // reject anything close to full-viewport-per-frame (24 * 30 = 720).
    assert!(
        total_repainted <= frames * (lines_per_frame + 2),
        "steady scroll repainted {total_repainted} rows over {frames} frames"
    );
}

// ---------------------------------------------------------------------------
// Palette changes
// ---------------------------------------------------------------------------

#[test]
fn osc_palette_change_invalidates_everything() {
    let mut sim = Sim::new(80, 24);
    sim.feed_str("\x1b[31mred text\x1b[0m\r\nplain\r\n");
    sim.frame();

    // Redefine ANSI color 1: every cached row may now be stale even
    // though no cell content changed.
    sim.feed_str("\x1b]4;1;#00ff00\x07");
    let repaint = sim.frame();
    assert_eq!(
        repaint.len(),
        24,
        "palette change must invalidate all visible rows, got {}",
        repaint.len()
    );
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

#[test]
fn resize_repaints_everything_once_then_settles() {
    let mut sim = Sim::new(120, 24);
    sim.feed_str("user@host:~$ ");
    sim.frame();

    sim.term.resize(TermSize::new(100, 24));
    sim.cols = 100;
    let repaint = sim.frame();
    assert!(!repaint.is_empty(), "resize must repaint");

    let repaint = sim.frame();
    assert!(
        repaint.is_empty(),
        "post-resize idle frame repainted {repaint:?}"
    );
}

// ---------------------------------------------------------------------------
// Memory: frozen grids and capture caps
// ---------------------------------------------------------------------------

#[test]
fn frozen_grid_trims_trailing_blanks() {
    use slop2::render::FrozenGrid;

    let grid = FrozenGrid::from_bytes(b"hi there\r\nx\r\n", 200, 24);
    assert_eq!(grid.cols, 200);
    assert_eq!(grid.rows.len(), 2);
    assert!(
        grid.rows[0].len() <= 9,
        "row 0 kept {} cells for 8 chars of content",
        grid.rows[0].len()
    );
    assert!(grid.rows[1].len() <= 2);
    assert_eq!(grid.to_plain_text(), "hi there\nx");
}

#[test]
fn frozen_grid_keeps_styled_trailing_cells() {
    use slop2::render::FrozenGrid;

    // A colored-background band (e.g. a TUI status strip) must survive
    // trimming even though the trailing cells are spaces.
    let grid = FrozenGrid::from_bytes(b"\x1b[41m   x      \x1b[0m\r\n", 80, 24);
    assert!(
        grid.rows[0].len() >= 10,
        "colored-bg trailing spaces were trimmed: {} cells",
        grid.rows[0].len()
    );
}

#[test]
fn capture_cap_keeps_the_tail_at_a_line_boundary() {
    use slop2::term_loop::{cap_capture, CAPTURE_KEEP_BYTES, CAPTURE_MAX_BYTES};

    let mut capture: Vec<u8> = Vec::new();
    let mut i = 0u64;
    while capture.len() <= CAPTURE_MAX_BYTES {
        capture.extend_from_slice(format!("log line number {i}\n").as_bytes());
        i += 1;
    }
    capture.extend_from_slice(b"tail after cap\n");
    cap_capture(&mut capture);

    assert!(capture.len() <= CAPTURE_KEEP_BYTES + 64 * 1024);
    // The kept tail starts at a line boundary and still ends with the
    // most recent bytes.
    assert!(capture.starts_with(b"log line number"));
    assert!(capture.ends_with(b"tail after cap\n"));
}

#[test]
fn capture_under_cap_is_untouched() {
    use slop2::term_loop::cap_capture;

    let mut capture = b"short output\n".to_vec();
    cap_capture(&mut capture);
    assert_eq!(capture, b"short output\n");
}
