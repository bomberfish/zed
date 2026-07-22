// Pixel-parity test: the damage-cached render path (offscreen surface,
// row-hash skipping, scroll shift-blits) must produce byte-identical
// pixels to a naive full repaint of the same grid. Runs headless —
// cairo ImageSurfaces and a pangocairo fontmap need no display.

use alacritty_terminal::event::VoidListener;
use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::index::{Column, Line};
use alacritty_terminal::term::test::TermSize;
use alacritty_terminal::term::{Config as TermConfig, Term, TermDamage};
use alacritty_terminal::vte::ansi::{Color as AlacColor, Processor, StdSyncHandler};

use gtk4::cairo;
use gtk4::pango;
use pangocairo::prelude::FontMapExt;

use slop2::grid_cache::{hash_mix, hash_rgb, GridCache};
use slop2::render::{self, CellMetrics};

const COLS: usize = 90;
const ROWS: usize = 16;
const WASH: (f64, f64, f64, f64) = (0.08, 0.09, 0.11, 0.35);

fn hash_color(h: u64, color: AlacColor) -> u64 {
    match color {
        AlacColor::Spec(rgb) => hash_rgb(hash_mix(h, 1), rgb),
        AlacColor::Named(n) => hash_mix(hash_mix(h, 2), n as u64),
        AlacColor::Indexed(i) => hash_mix(hash_mix(h, 3), i as u64),
    }
}

struct Harness {
    term: Term<VoidListener>,
    parser: Processor<StdSyncHandler>,
    cache: GridCache,
    font_desc: pango::FontDescription,
    metrics: CellMetrics,
    width: f64,
}

impl Harness {
    fn new() -> Self {
        let font_desc = pango::FontDescription::from_string("Noto Sans Mono Medium 10");
        let font_map = pangocairo::FontMap::default();
        let pctx = font_map.create_context();
        let metrics = render::measure_cell(&pctx, &font_desc);
        Self {
            term: Term::new(
                TermConfig {
                    scrolling_history: 1000,
                    ..Default::default()
                },
                &TermSize::new(COLS, ROWS),
                VoidListener,
            ),
            parser: Processor::new(),
            cache: GridCache::new(),
            font_desc,
            metrics,
            width: (COLS as f64 * 8.0f64.max(1.0)).ceil(),
        }
    }

    fn feed(&mut self, s: &str) {
        self.parser.advance(&mut self.term, s.as_bytes());
    }

    fn visible_range(&self) -> (i64, usize) {
        let history = self.term.grid().history_size() as i64;
        let total = history + ROWS as i64;
        let first = (total - ROWS as i64).max(0);
        (first, (total - first) as usize)
    }

    fn paint_rows_onto(
        &self,
        cr: &cairo::Context,
        rows: impl Iterator<Item = i64>,
        y_of: impl Fn(i64) -> f64,
    ) {
        let layout = pangocairo::functions::create_layout(cr);
        layout.set_font_description(Some(&self.font_desc));
        let grid = self.term.grid();
        let colors = self.term.colors();
        let history = self.term.grid().history_size() as i64;
        for row in rows {
            let y = y_of(row);
            cr.save().unwrap();
            cr.rectangle(0.0, y, self.width, self.metrics.height);
            cr.clip();
            cr.set_operator(cairo::Operator::Source);
            cr.set_source_rgba(WASH.0, WASH.1, WASH.2, WASH.3);
            cr.paint().unwrap();
            cr.set_operator(cairo::Operator::Over);
            let line = Line((row - history) as i32);
            let mut cells = Vec::with_capacity(COLS);
            for col in 0..COLS {
                let cell = &grid[line][Column(col)];
                cells.push((col, render::freeze_cell(cell, colors), false));
            }
            render::paint_live_row(cr, &layout, &cells, y, &self.metrics);
            cr.restore().unwrap();
        }
    }

    /// Reference render: full repaint of the visible range, no cache.
    fn render_reference(&self) -> Vec<u8> {
        let (first, count) = self.visible_range();
        let surface = self.new_target(count);
        {
            let cr = cairo::Context::new(&surface).unwrap();
            self.paint_rows_onto(
                &cr,
                first..first + count as i64,
                |row| (row - first) as f64 * self.metrics.height,
            );
        }
        Self::pixels(surface)
    }

    /// Cached render: collect damage, plan, repaint only planned rows
    /// into the cache surface, blit to a target. Mirrors draw_live.
    fn render_cached(&mut self) -> Vec<u8> {
        let history = self.term.grid().history_size();
        let mut full = false;
        let mut lines: Vec<usize> = Vec::new();
        match self.term.damage() {
            TermDamage::Full => full = true,
            TermDamage::Partial(iter) => lines.extend(iter.map(|b| b.line)),
        }
        self.term.reset_damage();
        if full {
            self.cache.add_damage_full();
        } else {
            self.cache.add_damage_rows(lines, history);
        }

        let (first, count) = self.visible_range();
        let history_i64 = history as i64;
        let plan = {
            let grid = self.term.grid();
            let colors = self.term.colors();
            self.cache.begin_frame(
                1,
                first,
                count,
                history,
                None,
                Some(colors),
                self.width,
                self.metrics.height,
                1,
                |row| {
                    let line = Line((row - history_i64) as i32);
                    let mut h = 0xcbf2_9ce4_8422_2325u64;
                    for col in 0..COLS {
                        let cell = &grid[line][Column(col)];
                        h = hash_mix(h, cell.c as u64);
                        h = hash_mix(h, cell.flags.bits() as u64);
                        h = hash_color(h, cell.fg);
                        h = hash_color(h, cell.bg);
                    }
                    h
                },
            )
        };

        if !plan.repaint.is_empty() {
            let origin = self.cache.origin();
            let surface = self.cache.surface().unwrap().clone();
            let cr = cairo::Context::new(&surface).unwrap();
            self.paint_rows_onto(&cr, plan.repaint.iter().copied(), |row| {
                (row - origin) as f64 * self.metrics.height
            });
        }

        let target = self.new_target(count);
        {
            let cr = cairo::Context::new(&target).unwrap();
            cr.set_operator(cairo::Operator::Source);
            cr.set_source_surface(self.cache.surface().unwrap(), 0.0, 0.0)
                .unwrap();
            cr.paint().unwrap();
        }
        Self::pixels(target)
    }

    fn new_target(&self, count: usize) -> cairo::ImageSurface {
        cairo::ImageSurface::create(
            cairo::Format::ARgb32,
            self.width as i32,
            (count as f64 * self.metrics.height) as i32,
        )
        .unwrap()
    }

    fn pixels(surface: cairo::ImageSurface) -> Vec<u8> {
        let mut surface = surface;
        surface.flush();
        let out = surface.data().unwrap().to_vec();
        out
    }
}

#[test]
fn cached_pixels_match_reference_across_a_session() {
    let mut h = Harness::new();

    // Frame 1: fresh shell content.
    h.feed("user@host:~$ ls --color\r\n\x1b[34msrc\x1b[0m  \x1b[34mtarget\x1b[0m  Cargo.toml\r\nuser@host:~$ ");
    let cached = h.render_cached();
    assert_eq!(cached, h.render_reference(), "initial frame differs");

    // Frame 2: typing echo (partial repaint path).
    h.feed("cargo test");
    let cached = h.render_cached();
    assert_eq!(cached, h.render_reference(), "typing frame differs");

    // Frame 3: idle (pure blit path).
    let cached = h.render_cached();
    assert_eq!(cached, h.render_reference(), "idle frame differs");

    // Frame 4: enough output to scroll (shift-blit path).
    for i in 0..ROWS + 6 {
        h.feed(&format!("output line {i} with \x1b[31mcolor\x1b[0m\r\n"));
    }
    let cached = h.render_cached();
    assert_eq!(cached, h.render_reference(), "scrolled frame differs");

    // Frame 5: steady drip (shift by a few rows, repaint the tail).
    for i in 0..3 {
        h.feed(&format!("drip {i}\r\n"));
    }
    let cached = h.render_cached();
    assert_eq!(cached, h.render_reference(), "drip frame differs");

    // Frame 6: overwrite in place (TUI-ish damage without scroll).
    h.feed("\x1b[2;1H\x1b[7moverwritten status line\x1b[27m\x1b[K");
    let cached = h.render_cached();
    assert_eq!(cached, h.render_reference(), "overwrite frame differs");
}
