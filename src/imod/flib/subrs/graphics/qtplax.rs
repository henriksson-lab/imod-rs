//! Translation of `IMOD/flib/subrs/graphics/qtplax.cpp` and `qtplax.h`: the
//! graphics routines the Fortran plotting library draws through
//! (`plax_box`, `plax_vect`, `plax_sctext`, ... on a 1280 x 1024 virtual
//! screen with Y up), the display list they build, and the program start-up
//! that runs the Fortran main program on a second thread.
//!
//! The source draws with Qt (`QPainter` on a `QWidget`, a `QImage` for
//! saving).  Qt is not a dependency of this crate (CLAUDE.md), so the
//! drawing layer is Rust-native:
//!
//! - the display list, its encoding, the coordinate transforms, the pen and
//!   brush caching and the text placement are the source's, line for line;
//! - [`Painter`] stands in for `QPainter`: aliased lines, rectangles,
//!   ellipses and polygons on an RGB buffer, with Qt's geometry
//!   (`drawRect(x, y, w, h)` covers `x..=x+w`, a pen of width 0 or 1 is one
//!   pixel, wider pens have square caps), and antialiased text through
//!   `ab_glyph` from a system TrueType font (`PLAX_FONT`, else DejaVu Sans or
//!   Liberation Sans in the usual places; without one, text is not drawn);
//! - the window (`PlaxWindow`, with the left-click position popup, the
//!   right-click "Save to PNG" menu, `-message` and `-tooltip`) is Slint,
//!   built with the `gui` feature;
//! - saving to PNG renders the list into a buffer and writes it with the
//!   `image` crate; saving to TIFF goes through the translated
//!   `iiOpenNew`/`tiffWriteSection`, as the source does.
//!
//! Rust-only, and recorded as deviations: with no window available (the
//! `gui` feature off, no `DISPLAY`/`WAYLAND_DISPLAY`, or `PLAX_HEADLESS`
//! set) the program runs *headless*: the list is kept and saved exactly as
//! with a window of the requested size, and `plax_wait_for_close` acts as if
//! the window were closed at once (exit status 0).  `PLAX_CALL_LOG=<file>` writes one line per `plax_*` call (the
//! routine and its arguments), the verification log compared with native's
//! (recorded there by an `LD_PRELOAD` shim over `libdnmncar.so`).  The
//! "Print" menu item has no printer to go to and does nothing.

use std::io::Write as _;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::imod::libcfshr::b3dutil::{CArg, c_format_bytes, exit, program_args};

/// `#define LIST_CHUNK  1024` (`qtplax.cpp:46`).
const LIST_CHUNK: usize = 1024;
/// `#define FSTRING_LEN  80` (`qtplax.cpp:47`).
const FSTRING_LEN: usize = 80;
/// `TEXT_SIZE_SCALE 2.25` (`qtplax.cpp:59`, the non-Mac, non-Windows arm).
const TEXT_SIZE_SCALE: f64 = 2.25;
/// `DEFAULT_HEIGHT 640` (`qtplax.cpp:60`).
const DEFAULT_HEIGHT: i32 = 640;
/// `#define PLAX_LOWRAMP 40` (`qtplax.h:19`).
pub const PLAX_LOWRAMP: i32 = 40;
/// `#define PLAX_RAMPSIZE 256` (`qtplax.h:20`).
pub const PLAX_RAMPSIZE: usize = 256;

/// The display-list codes (`qtplax.cpp:150-151`).
const PCALL_MAPCOLOR: i32 = 0;
const PCALL_BOX: i32 = 1;
const PCALL_BOXO: i32 = 2;
const PCALL_VECT: i32 = 3;
const PCALL_VECTW: i32 = 4;
const PCALL_CIRC: i32 = 5;
const PCALL_CIRCO: i32 = 6;
const PCALL_POLY: i32 = 7;
const PCALL_POLYO: i32 = 8;
const PCALL_SCTEXT: i32 = 9;
const PCALL_ALIGN: i32 = 10;

/// Qt alignment flags used by `plax_draw_text`.
const ALIGN_LEFT: i32 = 0x1;
const ALIGN_RIGHT: i32 = 0x2;
const ALIGN_HCENTER: i32 = 0x4;
const ALIGN_TOP: i32 = 0x20;
const ALIGN_BOTTOM: i32 = 0x40;
const ALIGN_VCENTER: i32 = 0x80;
const TEXT_WORD_WRAP: i32 = 0x1000;

/// The logical resolution text point sizes are converted at: what the
/// reference Qt used under Xvfb (measured: its text is 72/96 of the size at
/// 96 dots per inch).
const FONT_DPI: f32 = 72.;

/// `qRgb(r, g, b)`.
fn q_rgb(r: i32, g: i32, b: i32) -> u32 {
    0xff000000 | (((r & 0xff) as u32) << 16) | (((g & 0xff) as u32) << 8) | ((b & 0xff) as u32)
}

/// The file-scope statics of `qtplax.cpp` (`:64-100`).
struct Plax {
    /// `static QRgb sRGB[PLAX_RAMPSIZE]`
    rgb: [u32; PLAX_RAMPSIZE],
    scale_x: f32,
    scale_y: f32,
    font_scale: f32,
    /// `sDrawList`, `sListSize` (its length) and `sListMax`.
    draw_list: Vec<i32>,
    list_max: usize,
    out_list_ind: usize,
    last_size: i32,
    prog_name: String,
    plax_width: i32,
    plax_height: i32,
    plax_top: i32,
    plax_left: i32,
    user_size_set: bool,
    user_pos_set: bool,
    plax_open: i32,
    plax_exposed: i32,
    pen_color: i32,
    pen_width: i32,
    brush_closed: i32,
    brush_color: i32,
    next_text_align: i32,
    no_graph: i32,
    message: Option<String>,
    tool_tip: Option<String>,
    exit_on_close: i32,
    user_scale_set: bool,
    user_xscale: f32,
    user_xadd: f32,
    user_yscale: f32,
    user_yadd: f32,
    com_file: [u8; FSTRING_LEN],
    /// `static int lastbold` in `plax_draw_text`.
    last_bold: i32,
    /// Rust-only: no window (see the module documentation).
    headless: bool,
    /// Rust-only: the redraw signal for the window (`PlaxThread::sendSignal`).
    signal: bool,
}

static PLAX: Mutex<Plax> = Mutex::new(Plax {
    rgb: [0; PLAX_RAMPSIZE],
    scale_x: 0.,
    scale_y: 0.,
    font_scale: 1.,
    draw_list: Vec::new(),
    list_max: 0,
    out_list_ind: 0,
    last_size: 0,
    prog_name: String::new(),
    plax_width: 5 * DEFAULT_HEIGHT / 4,
    plax_height: DEFAULT_HEIGHT,
    plax_top: 30,
    plax_left: 10,
    user_size_set: false,
    user_pos_set: false,
    plax_open: 0,
    plax_exposed: 0,
    pen_color: -1,
    pen_width: 0,
    brush_closed: 0,
    brush_color: -1,
    next_text_align: 0,
    no_graph: 0,
    message: None,
    tool_tip: None,
    exit_on_close: 0,
    user_scale_set: false,
    user_xscale: 0.,
    user_xadd: 0.,
    user_yscale: 0.,
    user_yadd: 0.,
    com_file: [b' '; FSTRING_LEN],
    last_bold: 0,
    headless: false,
    signal: false,
});

/// `sPlaxWidget->lock()`: the mutex over the display list (here over all of
/// the state, which the window thread also reads).
fn lock() -> MutexGuard<'static, Plax> {
    PLAX.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Rust-only: the `PLAX_CALL_LOG` verification log (module documentation).
fn call_log(line: &[u8]) {
    static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let log = LOG.get_or_init(|| {
        let name = std::env::var_os("PLAX_CALL_LOG")?;
        std::fs::File::create(name).ok().map(Mutex::new)
    });
    if let Some(file) = log {
        let mut file = file.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = file.write_all(line);
        let _ = file.write_all(b"\n");
    }
}

/// `f2cString`: a Fortran string with its trailing blanks removed.
fn f2c_string(string: &[u8]) -> Vec<u8> {
    let end = string.iter().rposition(|&c| c != b' ').map_or(0, |p| p + 1);
    string[..end].to_vec()
}

/// `sscanf(text, "%d%*c%d", &a, &b)`: the integers stored before the first
/// conversion that fails.
fn scan_two_ints(text: &str, a: &mut i32, b: &mut i32) {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    let mut scan_int = |pos: &mut usize| -> Option<i32> {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        let start = *pos;
        if *pos < bytes.len() && (bytes[*pos] == b'+' || bytes[*pos] == b'-') {
            *pos += 1;
        }
        let digits = *pos;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
        if *pos == digits {
            return None;
        }
        let value: i64 = std::str::from_utf8(&bytes[start..*pos])
            .ok()?
            .parse()
            .ok()?;
        Some(value as i32)
    };
    let Some(first) = scan_int(&mut pos) else {
        return;
    };
    *a = first;
    // `%*c`: one character, whatever it is
    if pos >= bytes.len() {
        return;
    }
    pos += 1;
    if let Some(second) = scan_int(&mut pos) {
        *b = second;
    }
}

/// Original `plax_initialize` (`qtplax.cpp:403`): reads the `-s`, `-p`,
/// `-message`, `-tooltip`, `-comfile` and `-nograph` arguments, opens the
/// window and runs `real_graphics_main` (the program's `realGraphicsMain`,
/// which the source calls by its linked name) on a second thread; never
/// returns.
pub fn plax_initialize(string: &str, real_graphics_main: fn(&[u8])) -> ! {
    let mut plax = lock();
    plax.com_file = [b' '; FSTRING_LEN];
    plax.prog_name = string.trim_end_matches(' ').to_owned();

    // Get the arguments.  Fortran numbers them 0 to iargc()
    let mut argv = program_args();
    if argv.is_empty() {
        argv.push(String::new());
    }
    argv.push("-style=windows".to_owned());
    let argc = argv.len();

    // Look for -s and -p arguments
    let mut i = 1usize;
    while i + 1 < argc {
        if argv[i] == "-s" {
            i += 1;
            let (mut w, mut h) = (plax.plax_width, plax.plax_height);
            scan_two_ints(&argv[i], &mut w, &mut h);
            plax.plax_width = w;
            plax.plax_height = h;
            plax.user_size_set = true;
            i += 1;
            continue;
        }
        if argv[i] == "-p" {
            i += 1;
            let (mut l, mut t) = (plax.plax_left, plax.plax_top);
            scan_two_ints(&argv[i], &mut l, &mut t);
            plax.plax_left = l;
            plax.plax_top = t;
            plax.user_pos_set = true;
        }

        // Get arguments for tooltip or message box or com file
        if argv[i] == "-message" {
            i += 1;
            plax.message = Some(argv[i].clone());
        }
        if argv[i] == "-tooltip" {
            i += 1;
            plax.tool_tip = Some(argv[i].clone());
        }
        if argv[i] == "-comfile" {
            i += 1;
            // `c2fString(argv[++i], sComFile, FSTRING_LEN)`
            let bytes = argv[i].as_bytes();
            let n = bytes.len().min(FSTRING_LEN);
            plax.com_file = [b' '; FSTRING_LEN];
            plax.com_file[..n].copy_from_slice(&bytes[..n]);
        }

        // Also look for an argument to avoid trying to start Qt app
        if argv[i] == "-nograph" {
            plax.no_graph = 1;
            let com_file = plax.com_file;
            drop(plax);
            real_graphics_main(&com_file);
            exit(0);
        }
        i += 1;
    }

    // Otherwise start the Qt application and start second thread that calls
    // Fortran
    let com_file = plax.com_file;
    let want_window = window_available();
    drop(plax);
    if want_window {
        #[cfg(feature = "gui")]
        {
            if window::start_plax_app(com_file, real_graphics_main) {
                exit(0);
            }
        }
    }

    // Rust-only headless start (module documentation): the window "has"
    // its requested size, as after its first resize event.
    {
        let mut plax = lock();
        plax.headless = true;
        start_plax_geometry(&mut plax);
        let (w, h) = (plax.plax_width, plax.plax_height);
        resize_event(&mut plax, w, h);
    }
    real_graphics_main(&com_file);
    exit(0);
}

/// Rust-only: whether a window can be opened (module documentation).
fn window_available() -> bool {
    if !cfg!(feature = "gui") {
        return false;
    }
    if std::env::var_os("PLAX_HEADLESS").is_some_and(|value| !value.is_empty() && value != "0") {
        return false;
    }
    std::env::var_os("DISPLAY").is_some_and(|v| !v.is_empty())
        || std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
}

/// The geometry part of `startPlaxApp` (`qtplax.cpp:555-584`).  The device
/// pixel ratio is 1 here: the Slint window scales its logical size itself.
fn start_plax_geometry(plax: &mut Plax) {
    let mut dpi_ratio: f32 = 1.;
    if plax.user_pos_set {
        plax.plax_left = (plax.plax_left as f32 / dpi_ratio) as i32;
        plax.plax_top = (plax.plax_top as f32 / dpi_ratio) as i32;
    }
    if !plax.user_size_set {
        dpi_ratio = dpi_ratio.powf(0.4);
    }
    plax.plax_width = (plax.plax_width as f32 / dpi_ratio) as i32;
    plax.plax_height = (plax.plax_height as f32 / dpi_ratio) as i32;
    plax.scale_x = 0.5 / dpi_ratio;
    plax.scale_y = 0.5 / dpi_ratio;
    plax.plax_exposed = 0;
}

/// `PlaxWindow::resizeEvent` (`qtplax.cpp:192`): record size and set scale.
fn resize_event(plax: &mut Plax, width: i32, height: i32) {
    plax.plax_width = width;
    plax.plax_height = height;
    plax.scale_x = width as f32 / 1280.0;
    plax.scale_y = height as f32 / 1024.0;

    // Make it repaint the whole thing
    plax.out_list_ind = 0;
}

/// `PlaxThread::sendSignal` (`qtplax.cpp:397`): asks the window thread to
/// run `redrawSlot`.
fn send_signal(plax: &mut Plax) {
    plax.signal = true;
}

/// Original `plax_open` (`qtplax.cpp:480`).
pub fn plax_open() -> i32 {
    call_log(b"open");
    let mut plax = lock();
    if plax.no_graph != 0 {
        return 0;
    }
    plax.plax_open = 1;

    // Qt in main thread: just show the widget now
    send_signal(&mut plax);
    0
}

/// Original `plax_close` (`qtplax.cpp:494`): "close" is really hiding the
/// widget.
pub fn plax_close() {
    call_log(b"close");
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    plax.plax_open = 0;
    send_signal(&mut plax);
}

/// Original `plax_flush` (`qtplax.cpp:504`): ask for a redraw.
pub fn plax_flush() {
    call_log(b"flush");
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    plax.plax_open = -1;
    send_signal(&mut plax);
}

/// Original `plax_wait_for_close` (`qtplax.cpp:522`): from here on closing
/// the window ends the program; this thread waits forever.
///
/// Fixed in translation (BUGS.md, `qtplax`): with `-nograph` the wait
/// condition was never created and the source dereferences NULL.  Here,
/// as with no window at all (headless), the window counts as closed at once:
/// the program exits with status 0, as the window's close event does.
pub fn plax_wait_for_close() {
    call_log(b"wait");
    {
        let mut plax = lock();
        if plax.no_graph != 0 || plax.headless {
            drop(plax);
            exit(0);
        }
        plax.exit_on_close = 1;
    }
    loop {
        std::thread::park();
    }
}

/*
 * The routines called from Fortran to put the drawing commands on the stack
 */

/// Original `plax_mapcolor` (`qtplax.cpp:604`).
pub fn plax_mapcolor(color: i32, ired: i32, igreen: i32, iblue: i32) {
    call_log(&c_format_bytes(
        "mapcolor %d %d %d %d",
        &[
            CArg::Int(color as i64),
            CArg::Int(ired as i64),
            CArg::Int(igreen as i64),
            CArg::Int(iblue as i64),
        ],
    ));
    add_four_args(&mut lock(), PCALL_MAPCOLOR, color, ired, igreen, iblue);
}

/// The log line of a five-argument call.
fn log_five(tag: &str, a: i32, b: i32, c: i32, d: i32, e: i32) {
    call_log(&c_format_bytes(
        "%s %d %d %d %d %d",
        &[
            CArg::Str(tag),
            CArg::Int(a as i64),
            CArg::Int(b as i64),
            CArg::Int(c as i64),
            CArg::Int(d as i64),
            CArg::Int(e as i64),
        ],
    ));
}

/// Original `plax_box` (`qtplax.cpp:609`): filled box.
pub fn plax_box(cindex: i32, ix1: i32, iy1: i32, ix2: i32, iy2: i32) {
    log_five("box", cindex, ix1, iy1, ix2, iy2);
    add_five_args(&mut lock(), PCALL_BOX, cindex, ix1, iy1, ix2, iy2);
}

/// Original `plax_boxo` (`qtplax.cpp:614`): open box.
pub fn plax_boxo(cindex: i32, ix1: i32, iy1: i32, ix2: i32, iy2: i32) {
    log_five("boxo", cindex, ix1, iy1, ix2, iy2);
    add_five_args(&mut lock(), PCALL_BOXO, cindex, ix1, iy1, ix2, iy2);
}

/// Original `plax_vect` (`qtplax.cpp:619`).
pub fn plax_vect(cindex: i32, ix1: i32, iy1: i32, ix2: i32, iy2: i32) {
    log_five("vect", cindex, ix1, iy1, ix2, iy2);
    add_five_args(&mut lock(), PCALL_VECT, cindex, ix1, iy1, ix2, iy2);
}

/// Original `plax_vectw` (`qtplax.cpp:624`).
pub fn plax_vectw(linewidth: i32, cindex: i32, ix1: i32, iy1: i32, ix2: i32, iy2: i32) {
    call_log(&c_format_bytes(
        "vectw %d %d %d %d %d %d",
        &[
            CArg::Int(linewidth as i64),
            CArg::Int(cindex as i64),
            CArg::Int(ix1 as i64),
            CArg::Int(iy1 as i64),
            CArg::Int(ix2 as i64),
            CArg::Int(iy2 as i64),
        ],
    ));
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    add_six_args(
        &mut plax,
        PCALL_VECTW,
        linewidth,
        cindex,
        ix1,
        iy1,
        ix2,
        iy2,
    );
}

/// Original `plax_circ` (`qtplax.cpp:635`): filled circle.
pub fn plax_circ(cindex: i32, radius: i32, ix: i32, iy: i32) {
    call_log(&c_format_bytes(
        "circ %d %d %d %d",
        &[
            CArg::Int(cindex as i64),
            CArg::Int(radius as i64),
            CArg::Int(ix as i64),
            CArg::Int(iy as i64),
        ],
    ));
    add_four_args(&mut lock(), PCALL_CIRC, cindex, radius, ix, iy);
}

/// Original `plax_circo` (`qtplax.cpp:641`): open circle.
pub fn plax_circo(cindex: i32, radius: i32, ix: i32, iy: i32) {
    call_log(&c_format_bytes(
        "circo %d %d %d %d",
        &[
            CArg::Int(cindex as i64),
            CArg::Int(radius as i64),
            CArg::Int(ix as i64),
            CArg::Int(iy as i64),
        ],
    ));
    add_four_args(&mut lock(), PCALL_CIRCO, cindex, radius, ix, iy);
}

/// The log line of a polygon call.
fn log_poly(tag: &str, cindex: i32, size: i32, vec: &[i16]) {
    let mut line = c_format_bytes(
        "%s %d %d",
        &[
            CArg::Str(tag),
            CArg::Int(cindex as i64),
            CArg::Int(size as i64),
        ],
    );
    for value in &vec[..(2 * size.max(0) as usize).min(vec.len())] {
        line.extend_from_slice(format!(" {value}").as_bytes());
    }
    call_log(&line);
}

/// The polygon's `4 * size` bytes as list words (`addBytesToList((char
/// *)vec, 4 * *size)`): two `b3dInt16`s per word, the first in the low half
/// as memory order on the reference platform puts it.
fn poly_bytes(size: i32, vec: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 * size.max(0) as usize);
    for value in &vec[..(2 * size.max(0) as usize).min(vec.len())] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Original `plax_poly` (`qtplax.cpp:647`): closed filled polygon.
pub fn plax_poly(cindex: i32, size: i32, vec: &[i16]) {
    log_poly("poly", cindex, size, vec);
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    if add_two_args(&mut plax, PCALL_POLY, cindex, size) == 0 {
        add_bytes_to_list(&mut plax, &poly_bytes(size, vec));
    }
}

/// Original `plax_polyo` (`qtplax.cpp:657`).
pub fn plax_polyo(cindex: i32, size: i32, vec: &[i16]) {
    log_poly("polyo", cindex, size, vec);
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    if add_two_args(&mut plax, PCALL_POLYO, cindex, size) == 0 {
        add_bytes_to_list(&mut plax, &poly_bytes(size, vec));
    }
}

/// Original `plax_sctext` (`qtplax.cpp:667`).  `string` is the whole Fortran
/// string, blanks included (`strsize` is its length).
pub fn plax_sctext(
    thickness: i32,
    xsize: i32,
    iysize: i32,
    cindex: i32,
    ix: i32,
    iy: i32,
    string: &[u8],
) {
    let mut line = c_format_bytes(
        "sctext %d %d %d %d %d %d %d [",
        &[
            CArg::Int(thickness as i64),
            CArg::Int(xsize as i64),
            CArg::Int(iysize as i64),
            CArg::Int(cindex as i64),
            CArg::Int(ix as i64),
            CArg::Int(iy as i64),
            CArg::Int(string.len() as i64),
        ],
    );
    line.extend_from_slice(string);
    line.push(b']');
    call_log(&line);
    let str_int = string.len() as i32;
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    if add_six_args(
        &mut plax,
        PCALL_SCTEXT,
        thickness,
        iysize,
        cindex,
        ix,
        iy,
        str_int,
    ) == 0
    {
        add_bytes_to_list(&mut plax, string);
    }
}

/// Original `plax_next_text_align` (`qtplax.cpp:685`).
pub fn plax_next_text_align(type_: i32) {
    call_log(&c_format_bytes("align %d", &[CArg::Int(type_ as i64)]));
    let mut plax = lock();
    if plax.no_graph != 0 {
        return;
    }
    if plax.draw_list.len() + 4 > plax.list_max && allocate_list_chunk(&mut plax) != 0 {
        return;
    }
    plax.draw_list.push(PCALL_ALIGN);
    plax.draw_list.push(type_);
}

/// Original `plax_drawing_scale` (`qtplax.cpp:696`).
pub fn plax_drawing_scale(xscale: f32, xadd: f32, yscale: f32, yadd: f32) {
    call_log(&c_format_bytes(
        "scale %g %g %g %g",
        &[
            CArg::Dbl(xscale as f64),
            CArg::Dbl(xadd as f64),
            CArg::Dbl(yscale as f64),
            CArg::Dbl(yadd as f64),
        ],
    ));
    let mut plax = lock();
    plax.user_scale_set = true;
    plax.user_xscale = xscale;
    plax.user_xadd = xadd;
    plax.user_yscale = yscale;
    plax.user_yadd = yadd;
}

/// Original `plax_erase` (`qtplax.cpp:705`).
pub fn plax_erase() {
    call_log(b"erase");
    let mut plax = lock();
    plax.out_list_ind = 0;
    plax.draw_list.clear();
}

/// Original `plax_putc` (`qtplax.cpp:712`).
pub fn plax_putc(f: u8) {
    let _ = std::io::stdout().write_all(&[f]);
}

/// Original `plaxSavePNG` (`qtplax.cpp:717`).
pub fn plax_save_png_c(filename: &str) {
    save_png_to_file(filename, false);
}

/// Original `plaxSaveTIFF` (`qtplax.cpp:722`).
pub fn plax_save_tiff_c(filename: &str) {
    save_png_to_file(filename, true);
}

/// The log line of a save call.
fn log_save(tag: &str, filename: &[u8]) {
    let mut line = c_format_bytes(
        "%s %d [",
        &[CArg::Str(tag), CArg::Int(filename.len() as i64)],
    );
    line.extend_from_slice(filename);
    line.push(b']');
    call_log(&line);
}

/// Original `plax_save_png` (`qtplax.cpp:727`): `filename` is the Fortran
/// string.
pub fn plax_save_png(filename: &[u8]) -> i32 {
    log_save("savepng", filename);
    let cname = f2c_string(filename);
    if cname.is_empty() {
        return 1;
    }
    save_png_to_file(&String::from_utf8_lossy(&cname), false);
    0
}

/// Original `plax_save_tiff` (`qtplax.cpp:739`).
pub fn plax_save_tiff(filename: &[u8]) -> i32 {
    log_save("savetiff", filename);
    let cname = f2c_string(filename);
    if cname.is_empty() {
        return 1;
    }
    save_png_to_file(&String::from_utf8_lossy(&cname), true);
    0
}

/*****************************************************************************/
/* Internal Functions                                                        */
/*****************************************************************************/

/// Original `addBytesToList` (`qtplax.cpp:757`): the bytes packed into list
/// words in memory order.
fn add_bytes_to_list(plax: &mut Plax, bytes: &[u8]) -> i32 {
    let num_int = bytes.len().div_ceil(4);
    while plax.draw_list.len() + num_int > plax.list_max {
        if allocate_list_chunk(plax) != 0 {
            return 1;
        }
    }
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        plax.draw_list.push(i32::from_le_bytes(word));
    }
    0
}

/// Original `addTwoArgs` (`qtplax.cpp:772`).
fn add_two_args(plax: &mut Plax, code: i32, i1: i32, i2: i32) -> i32 {
    if plax.draw_list.len() + 4 > plax.list_max && allocate_list_chunk(plax) != 0 {
        return 1;
    }
    plax.draw_list.extend_from_slice(&[code, i1, i2]);
    0
}

/// Original `addFourArgs` (`qtplax.cpp:783`).
fn add_four_args(plax: &mut Plax, code: i32, i1: i32, i2: i32, i3: i32, i4: i32) -> i32 {
    if plax.no_graph != 0 {
        return 0;
    }
    if plax.draw_list.len() + 5 > plax.list_max && allocate_list_chunk(plax) != 0 {
        return 1;
    }
    plax.draw_list.extend_from_slice(&[code, i1, i2, i3, i4]);
    0
}

/// Original `addFiveArgs` (`qtplax.cpp:800`).
fn add_five_args(plax: &mut Plax, code: i32, i1: i32, i2: i32, i3: i32, i4: i32, i5: i32) -> i32 {
    if plax.no_graph != 0 {
        return 0;
    }
    if plax.draw_list.len() + 6 > plax.list_max && allocate_list_chunk(plax) != 0 {
        return 1;
    }
    plax.draw_list
        .extend_from_slice(&[code, i1, i2, i3, i4, i5]);
    0
}

/// Original `addSixArgs` (`qtplax.cpp:818`).
#[allow(clippy::too_many_arguments)]
fn add_six_args(
    plax: &mut Plax,
    code: i32,
    i1: i32,
    i2: i32,
    i3: i32,
    i4: i32,
    i5: i32,
    i6: i32,
) -> i32 {
    if plax.draw_list.len() + 7 > plax.list_max && allocate_list_chunk(plax) != 0 {
        return 1;
    }
    plax.draw_list
        .extend_from_slice(&[code, i1, i2, i3, i4, i5, i6]);
    0
}

/// Original `allocate_list_chunk` (`qtplax.cpp:834`).
fn allocate_list_chunk(plax: &mut Plax) -> i32 {
    if plax.draw_list.try_reserve(LIST_CHUNK).is_err() {
        eprintln!("QTPLAX: Error getting memory for drawing list.");
        plax.draw_list.clear();
        plax.list_max = 0;
        plax.out_list_ind = 0;
        return 1;
    }
    plax.list_max += LIST_CHUNK;
    0
}

/// Original static `draw` (`qtplax.cpp:850`): draws the list from
/// `sOutListInd` on with `painter`.
fn draw(plax: &mut Plax, painter: &mut Painter) {
    plax.last_size = 0;

    // Draw starting after the last item drawn
    while plax.out_list_ind < plax.draw_list.len() {
        let mut ind = plax.out_list_ind;
        plax.out_list_ind += 1;
        let code = plax.draw_list[ind];
        ind += 1;
        let word = |plax: &Plax, k: usize| plax.draw_list.get(k).copied().unwrap_or(0);
        match code {
            PCALL_MAPCOLOR => {
                let color = word(plax, ind);
                if (0..PLAX_RAMPSIZE as i32).contains(&color) {
                    plax.rgb[color as usize] = q_rgb(
                        word(plax, ind + 1),
                        word(plax, ind + 2),
                        word(plax, ind + 3),
                    );
                }
                plax.pen_color = -1;
                plax.brush_color = -1;
                plax.out_list_ind += 4;
            }
            PCALL_BOX => {
                plax_set_brush(plax, painter, word(plax, ind), 1);
                let args = [
                    word(plax, ind),
                    word(plax, ind + 1),
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                    word(plax, ind + 4),
                ];
                plax_draw_box(plax, painter, args[0], args[1], args[2], args[3], args[4]);
                plax.out_list_ind += 5;
            }
            PCALL_BOXO => {
                plax_set_brush(plax, painter, word(plax, ind), 0);
                let args = [
                    word(plax, ind),
                    word(plax, ind + 1),
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                    word(plax, ind + 4),
                ];
                plax_draw_box(plax, painter, args[0], args[1], args[2], args[3], args[4]);
                plax.out_list_ind += 5;
            }
            PCALL_VECT => {
                plax_set_pen(plax, painter, word(plax, ind), 0);
                let args = [
                    word(plax, ind + 1),
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                    word(plax, ind + 4),
                ];
                plax_draw_vect(plax, painter, args[0], args[1], args[2], args[3]);
                plax.out_list_ind += 5;
            }
            PCALL_VECTW => {
                plax_set_pen(plax, painter, word(plax, ind + 1), word(plax, ind));
                let args = [
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                    word(plax, ind + 4),
                    word(plax, ind + 5),
                ];
                plax_draw_vect(plax, painter, args[0], args[1], args[2], args[3]);
                plax.out_list_ind += 6;
            }
            PCALL_CIRC => {
                plax_set_brush(plax, painter, word(plax, ind), 1);
                let args = [
                    word(plax, ind),
                    word(plax, ind + 1),
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                ];
                plax_draw_circ(plax, painter, args[0], args[1], args[2], args[3]);
                plax.out_list_ind += 4;
            }
            PCALL_CIRCO => {
                plax_set_brush(plax, painter, word(plax, ind), 0);
                let args = [
                    word(plax, ind),
                    word(plax, ind + 1),
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                ];
                plax_draw_circ(plax, painter, args[0], args[1], args[2], args[3]);
                plax.out_list_ind += 4;
            }
            PCALL_POLY | PCALL_POLYO => {
                let csize = word(plax, ind + 1);
                let mut vec = Vec::with_capacity(2 * csize.max(0) as usize);
                for k in 0..csize.max(0) as usize {
                    let packed = word(plax, ind + 2 + k);
                    vec.push((packed & 0xffff) as u16 as i16);
                    vec.push(((packed >> 16) & 0xffff) as u16 as i16);
                }
                let cindex = word(plax, ind);
                plax_draw_poly(
                    plax,
                    painter,
                    cindex,
                    csize,
                    &vec,
                    i32::from(code == PCALL_POLY),
                );
            }
            PCALL_SCTEXT => {
                let strsize = word(plax, ind + 5);
                let mut string = Vec::with_capacity(strsize.max(0) as usize);
                for k in 0..(strsize.max(0) as usize).div_ceil(4) {
                    string.extend_from_slice(&word(plax, ind + 6 + k).to_le_bytes());
                }
                string.truncate(strsize.max(0) as usize);
                let args = [
                    word(plax, ind),
                    word(plax, ind + 1),
                    word(plax, ind + 2),
                    word(plax, ind + 3),
                    word(plax, ind + 4),
                ];
                plax_draw_text(
                    plax, painter, args[0], args[1], args[2], args[3], args[4], strsize, &string,
                );
                plax.out_list_ind += 6 + (strsize.max(0) as usize).div_ceil(4);
            }
            PCALL_ALIGN => {
                plax.next_text_align = word(plax, ind);
                plax.out_list_ind += 1;
            }
            _ => {}
        }
    }
}

/// Original static `plax_draw_box` (`qtplax.cpp:948`).
fn plax_draw_box(
    plax: &mut Plax,
    painter: &mut Painter,
    cindex: i32,
    mut x1: i32,
    mut y1: i32,
    mut x2: i32,
    mut y2: i32,
) {
    plax_set_pen(plax, painter, cindex, 0);
    plax_transform(plax, &mut x1, &mut y1);
    plax_transform(plax, &mut x2, &mut y2);

    let x = if x1 > x2 { x2 } else { x1 };
    let y = if y1 > y2 { y2 } else { y1 };
    let mut width = x1 - x2;
    let mut height = y1 - y2;

    if width < 0 {
        width *= -1;
    }

    if height < 0 {
        height *= -1;
    }
    width += 1;
    height += 1;

    painter.draw_rect(x, y, width, height);
}

/// Original static `plax_draw_circ` (`qtplax.cpp:979`).
fn plax_draw_circ(
    plax: &mut Plax,
    painter: &mut Painter,
    cindex: i32,
    radius: i32,
    mut x: i32,
    mut y: i32,
) {
    let size = plax_scale(plax, radius);
    plax_set_pen(plax, painter, cindex, 0);

    plax_transform(plax, &mut x, &mut y);
    painter.draw_ellipse(x - size, y - size, size * 2, size * 2);
}

/// Original static `plax_draw_vect` (`qtplax.cpp:989`).
fn plax_draw_vect(
    plax: &mut Plax,
    painter: &mut Painter,
    mut x1: i32,
    mut y1: i32,
    mut x2: i32,
    mut y2: i32,
) {
    plax_transform(plax, &mut x1, &mut y1);
    plax_transform(plax, &mut x2, &mut y2);
    painter.draw_line(x1, y1, x2, y2);
}

/// Original static `plax_draw_poly` (`qtplax.cpp:996`).
fn plax_draw_poly(
    plax: &mut Plax,
    painter: &mut Painter,
    cindex: i32,
    csize: i32,
    vec: &[i16],
    iffill: i32,
) {
    let mut points: Vec<(i32, i32)> = Vec::with_capacity(csize.max(0) as usize);

    plax_set_pen(plax, painter, cindex, 0);
    plax_set_brush(plax, painter, cindex, iffill);

    for i in 0..csize.max(0) as usize {
        points.push((
            plax_transx(plax, vec[i * 2]) as i32,
            plax_transy(plax, vec[(i * 2) + 1]) as i32,
        ));
    }

    painter.draw_polygon(&points);

    plax.out_list_ind += 2 + csize.max(0) as usize;
}

/// Original static `plax_draw_text` (`qtplax.cpp:1017`).
#[allow(clippy::too_many_arguments)]
fn plax_draw_text(
    plax: &mut Plax,
    painter: &mut Painter,
    thickness: i32,
    iysize: i32,
    cindex: i32,
    mut x: i32,
    mut y: i32,
    _strsize: i32,
    string: &[u8],
) {
    let mut ysize = (iysize as f64 * TEXT_SIZE_SCALE * plax.font_scale as f64) as i32;
    let mut ifbold = 0;
    let cstring = f2c_string(string);

    if thickness > 1 {
        ifbold = 1;
    }

    ysize = plax_scale(plax, ysize);
    if ysize <= 1 {
        ysize = 1;
    }

    if plax.last_size != ysize || plax.last_bold != ifbold {
        painter.set_font(ysize, ifbold != 0);
    }
    plax.last_bold = ifbold;
    plax.last_size = ysize;

    plax_transform(plax, &mut x, &mut y);
    plax_set_pen(plax, painter, cindex, 0);

    let text = String::from_utf8_lossy(&cstring).into_owned();
    if plax.next_text_align % 10 != 0 {
        let (pw, ph) = (plax.plax_width, plax.plax_height);
        let max_width = 2 * x.min(pw - x);
        let max_height = 2 * (y - 1).min(ph - y - 1);
        let mut align;
        let width;
        let height;
        match plax.next_text_align % 10 {
            1 => {
                // Top center
                align = ALIGN_HCENTER | ALIGN_TOP;
                width = max_width.min(pw / 2);
                height = (ph - 1 - y).min(ph / 2);
                x -= width / 2;
            }
            2 => {
                // right center
                align = ALIGN_VCENTER | ALIGN_RIGHT;
                width = (x - 1).min(pw / 2);
                height = max_height.min(ph / 2);
                x -= width;
                y -= height / 2;
            }
            3 => {
                // Bottom center
                align = ALIGN_HCENTER | ALIGN_BOTTOM;
                width = max_width.min(pw / 2);
                height = (y - 1).min(ph / 2);
                x -= width / 2;
                y -= height;
            }
            4 => {
                // left center
                align = ALIGN_VCENTER | ALIGN_LEFT;
                width = (pw - x - 1).min(pw / 2);
                height = max_height.min(ph / 2);
                y -= height / 2;
            }
            _ => {
                // all centered
                align = ALIGN_VCENTER | ALIGN_HCENTER;
                height = max_height.min(ph / 2);
                width = max_width.min(pw / 2);
                x -= width / 2;
                y -= height / 2;
            }
        }
        if plax.next_text_align >= 10 {
            align |= TEXT_WORD_WRAP;
        }
        painter.draw_text_rect(x, y, width, height, align, &text);
        plax.next_text_align = 0;
    } else {
        painter.draw_text(x, y, &text);
    }
    let _ = std::io::stdout().flush();
}

/// Original static `plax_scale` (`qtplax.cpp:1117`).
fn plax_scale(plax: &Plax, size: i32) -> i32 {
    let mut nsize = if plax.scale_x > plax.scale_y {
        (size as f32 * plax.scale_y) as i32
    } else {
        (size as f32 * plax.scale_x) as i32
    };

    if nsize < 1 {
        nsize = 1;
    }
    nsize
}

/// Original static `plax_transx` (`qtplax.cpp:1147`).
fn plax_transx(plax: &Plax, ix: i16) -> i16 {
    let x = ix as f32 * plax.scale_x;
    x as i16
}

/// Original static `plax_transy` (`qtplax.cpp:1154`).
fn plax_transy(plax: &Plax, iy: i16) -> i16 {
    let y = plax.scale_y * (1023.0 - iy as f32);
    y as i16
}

/// Original static `plax_transform` (`qtplax.cpp:1162`).
fn plax_transform(plax: &Plax, x: &mut i32, y: &mut i32) {
    let ix = *x as f32;
    let iy = *y as f32;

    *x = (plax.scale_x * ix) as i32;
    *y = (plax.scale_y as f64 * (1023.0 - iy as f64)) as i32;
}

/// Original static `plax_set_pen` (`qtplax.cpp:1176`).
fn plax_set_pen(plax: &mut Plax, painter: &mut Painter, color: i32, width: i32) {
    if color == plax.pen_color && width == plax.pen_width {
        return;
    }
    painter.set_pen(rgb_at(plax, color), width);
    plax.pen_color = color;
    plax.pen_width = width;
}

/// Original static `plax_set_brush` (`qtplax.cpp:1185`).
fn plax_set_brush(plax: &mut Plax, painter: &mut Painter, color: i32, closed: i32) {
    if (closed == 0 || color == plax.brush_color) && closed == plax.brush_closed {
        return;
    }
    if closed != 0 {
        painter.set_brush(Some(rgb_at(plax, color)));
    } else {
        painter.set_brush(None);
    }
    plax.brush_color = color;
    plax.brush_closed = closed;
}

/// `sRGB[color]`.  Fixed in translation (BUGS.md, `qtplax`): an index
/// outside the 256-entry table reads past it in the source; here it is
/// black.
fn rgb_at(plax: &Plax, color: i32) -> u32 {
    if (0..PLAX_RAMPSIZE as i32).contains(&color) {
        plax.rgb[color as usize]
    } else {
        q_rgb(0, 0, 0)
    }
}

/// Original static `savePNGtoFile` (`qtplax.cpp:300`): renders the whole
/// list into an image of the window's size and saves it as PNG or TIFF.
///
/// Fixed in translation (BUGS.md, `qtplax`): with `-nograph` there is no
/// widget and the source dereferences NULL; here nothing is saved.
fn save_png_to_file(filename: &str, save_tiff: bool) {
    let (width, height, rgb) = {
        let mut plax = lock();
        if plax.no_graph != 0 {
            return;
        }
        let mut painter = Painter::new(plax.plax_width, plax.plax_height);
        plax.out_list_ind = 0;
        draw(&mut plax, &mut painter);
        (painter.width, painter.height, painter.rgb_bytes())
    };
    if save_tiff {
        // For save to TIFF, copy RGB into a buffer
        let mut newbuf = rgb;

        // Open the file, set it up for RGB and write it
        let ii_file = unsafe {
            crate::imod::libiimod::iimage::ii_open_new(
                filename.as_bytes(),
                "wb",
                crate::imod::libiimod::iimage::IIFILE_TIFF,
            )
        };
        if !ii_file.is_null() {
            let err = unsafe {
                let file = &mut *ii_file;
                file.format = crate::imod::libiimod::iimage::IIFORMAT_RGB;
                file.type_ = crate::imod::libiimod::iimage::ImageDataType::UnsignedByte;
                file.amin = 0.;
                file.amax = 255.;
                file.nx = width;
                file.ny = height;
                file.nz = 1;
                crate::imod::libiimod::iitif::tiff_write_section(file, &mut newbuf, 8, 1, 72, 100)
            };
            if err != 0 {
                print!("Error ({err}) writing image to file\n");
            }
            unsafe { crate::imod::libiimod::iimage::ii_delete(ii_file) };
        } else {
            print!(
                "Error opening file {}: {}\n",
                filename,
                crate::imod::libcfshr::b3dutil::b3d_get_error()
            );
        }
    } else {
        // `image->save(filename, "PNG")`: a failure is silent
        let _ = image::save_buffer_with_format(
            filename,
            &rgb,
            width.max(0) as u32,
            height.max(0) as u32,
            image::ExtendedColorType::Rgb8,
            image::ImageFormat::Png,
        );
    }
}

/// Rust-only: renders the whole list at the window's current size (a
/// `paintEvent`: `sOutListInd = 0; draw()`), for the window.
#[cfg(feature = "gui")]
fn paint_event() -> (i32, i32, Vec<u8>) {
    let mut plax = lock();
    plax.plax_exposed = 1;
    plax.out_list_ind = 0;
    let mut painter = Painter::new(plax.plax_width, plax.plax_height);
    draw(&mut plax, &mut painter);
    (painter.width, painter.height, painter.rgb_bytes())
}

/// The fonts text is drawn with: regular and bold.
struct PlaxFonts {
    regular: Option<ab_glyph::FontVec>,
    bold: Option<ab_glyph::FontVec>,
}

/// Rust-only: loads the text fonts once (module documentation).
fn plax_fonts() -> &'static PlaxFonts {
    static FONTS: OnceLock<PlaxFonts> = OnceLock::new();
    FONTS.get_or_init(|| {
        let load = |path: &str| -> Option<ab_glyph::FontVec> {
            let data = std::fs::read(path).ok()?;
            ab_glyph::FontVec::try_from_vec(data).ok()
        };
        let regular_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        ];
        let bold_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        ];
        let (regular, bold) = match std::env::var("PLAX_FONT") {
            Ok(path) if path == "none" => (None, None),
            Ok(path) => {
                let font = load(&path);
                let bold = bold_paths.iter().find_map(|p| load(p));
                (font, bold)
            }
            Err(_) => (
                regular_paths.iter().find_map(|p| load(p)),
                bold_paths.iter().find_map(|p| load(p)),
            ),
        };
        if regular.is_none() && std::env::var("PLAX_FONT").as_deref() != Ok("none") {
            eprintln!(
                "WARNING: no TrueType font found for plot text; set PLAX_FONT to a .ttf file"
            );
        }
        PlaxFonts { regular, bold }
    })
}

/// Rust-only stand-in for `QPainter` on a `QImage` of format RGB32 (module
/// documentation): aliased geometry with Qt's pixel coverage, antialiased
/// text.
struct Painter {
    width: i32,
    height: i32,
    pixels: Vec<u32>,
    pen_color: u32,
    pen_width: i32,
    brush: Option<u32>,
    font_px: f32,
    bold: bool,
}

impl Painter {
    /// A new painter: black pen of width 1, no brush, the default font
    /// (`QFont()`, 9 points).
    fn new(width: i32, height: i32) -> Painter {
        let width = width.max(1);
        let height = height.max(1);
        Painter {
            width,
            height,
            pixels: vec![0xff000000; (width * height) as usize],
            pen_color: 0xff000000,
            pen_width: 1,
            brush: None,
            font_px: 9. * FONT_DPI / 72.,
            bold: false,
        }
    }

    fn rgb_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 3);
        for &p in &self.pixels {
            out.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8]);
        }
        out
    }

    fn set_pen(&mut self, color: u32, width: i32) {
        self.pen_color = color;
        self.pen_width = width;
    }

    fn set_brush(&mut self, brush: Option<u32>) {
        self.brush = brush;
    }

    /// `QFont::setPointSize(size)` and `setBold`.
    fn set_font(&mut self, point_size: i32, bold: bool) {
        self.font_px = point_size as f32 * FONT_DPI / 72.;
        self.bold = bold;
    }

    fn plot(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && y >= 0 && x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize] = color;
        }
    }

    fn blend(&mut self, x: i32, y: i32, color: u32, coverage: f32) {
        if x < 0 || y < 0 || x >= self.width || y >= self.height || coverage <= 0. {
            return;
        }
        let a = coverage.min(1.);
        let idx = (y * self.width + x) as usize;
        let bg = self.pixels[idx];
        let mix = |shift: u32| -> u32 {
            let b = ((bg >> shift) & 0xff) as f32;
            let f = ((color >> shift) & 0xff) as f32;
            ((b + (f - b) * a).round() as u32).min(255) << shift
        };
        self.pixels[idx] = 0xff000000 | mix(16) | mix(8) | mix(0);
    }

    /// Fills pixels whose centres are inside the polygon (odd-even rule).
    fn fill_polygon_f(&mut self, points: &[(f32, f32)], color: u32) {
        if points.len() < 3 {
            return;
        }
        let ymin = points.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let ymax = points.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        let y0 = (ymin.floor() as i32).max(0);
        let y1 = (ymax.ceil() as i32).min(self.height - 1);
        let mut xs: Vec<f32> = Vec::new();
        for py in y0..=y1 {
            let cy = py as f32 + 0.5;
            xs.clear();
            for k in 0..points.len() {
                let (ax, ay) = points[k];
                let (bx, by) = points[(k + 1) % points.len()];
                if (ay <= cy && by > cy) || (by <= cy && ay > cy) {
                    xs.push(ax + (cy - ay) * (bx - ax) / (by - ay));
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let start = (pair[0] - 0.5).ceil() as i32;
                let end = (pair[1] - 0.5).ceil() as i32 - 1;
                for px in start.max(0)..=end.min(self.width - 1) {
                    self.plot(px, py, color);
                }
            }
        }
    }

    /// `drawLine`: a one-pixel line for a pen of width 0 or 1 (both ends
    /// included), otherwise a filled band with square caps.
    fn draw_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        let color = self.pen_color;
        if self.pen_width <= 1 {
            let (mut x, mut y) = (x1, y1);
            let dx = (x2 - x1).abs();
            let dy = -(y2 - y1).abs();
            let sx = if x1 < x2 { 1 } else { -1 };
            let sy = if y1 < y2 { 1 } else { -1 };
            let mut err = dx + dy;
            loop {
                self.plot(x, y, color);
                if x == x2 && y == y2 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    x += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y += sy;
                }
            }
            return;
        }
        let half = self.pen_width as f32 / 2.;
        let (ax, ay) = (x1 as f32 + 0.5, y1 as f32 + 0.5);
        let (bx, by) = (x2 as f32 + 0.5, y2 as f32 + 0.5);
        let len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
        let (ux, uy) = if len > 0. {
            ((bx - ax) / len, (by - ay) / len)
        } else {
            (1., 0.)
        };
        let (nx, ny) = (-uy, ux);
        let corners = [
            (ax - ux * half + nx * half, ay - uy * half + ny * half),
            (bx + ux * half + nx * half, by + uy * half + ny * half),
            (bx + ux * half - nx * half, by + uy * half - ny * half),
            (ax - ux * half - nx * half, ay - uy * half - ny * half),
        ];
        self.fill_polygon_f(&corners, color);
    }

    /// `drawRect(x, y, w, h)`: the brush fills the inside, the pen outlines
    /// `x..=x+w`, `y..=y+h`.
    fn draw_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        if let Some(fill) = self.brush {
            for py in y..y + h {
                for px in x..x + w {
                    self.plot(px, py, fill);
                }
            }
        }
        self.draw_line(x, y, x + w, y);
        self.draw_line(x + w, y, x + w, y + h);
        self.draw_line(x + w, y + h, x, y + h);
        self.draw_line(x, y + h, x, y);
    }

    /// `drawEllipse(x, y, w, h)` for the circles `plax_draw_circ` draws
    /// (`w == h`): a filled disc for a brush, a midpoint-circle outline.
    fn draw_ellipse(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let r = w.min(h) / 2;
        let (cx, cy) = (x + r, y + r);
        if let Some(fill) = self.brush {
            for py in cy - r..=cy + r {
                for px in cx - r..=cx + r {
                    let (dx, dy) = (px - cx, py - cy);
                    if dx * dx + dy * dy <= r * r {
                        self.plot(px, py, fill);
                    }
                }
            }
        }
        let color = self.pen_color;
        let (mut dx, mut dy) = (r, 0);
        let mut err = 1 - r;
        while dx >= dy {
            for (px, py) in [
                (dx, dy),
                (dy, dx),
                (-dy, dx),
                (-dx, dy),
                (-dx, -dy),
                (-dy, -dx),
                (dy, -dx),
                (dx, -dy),
            ] {
                self.plot(cx + px, cy + py, color);
            }
            dy += 1;
            if err < 0 {
                err += 2 * dy + 1;
            } else {
                dx -= 1;
                err += 2 * (dy - dx) + 1;
            }
        }
    }

    /// `drawPolygon`/`drawConvexPolygon`: brush fill, then the closed
    /// outline with the pen.
    fn draw_polygon(&mut self, points: &[(i32, i32)]) {
        if let Some(fill) = self.brush {
            let pts: Vec<(f32, f32)> = points
                .iter()
                .map(|&(x, y)| (x as f32 + 0.5, y as f32 + 0.5))
                .collect();
            self.fill_polygon_f(&pts, fill);
        }
        for k in 0..points.len() {
            let (a, b) = (points[k], points[(k + 1) % points.len()]);
            self.draw_line(a.0, a.1, b.0, b.1);
        }
    }

    fn font(&self) -> Option<&'static ab_glyph::FontVec> {
        let fonts = plax_fonts();
        if self.bold {
            fonts.bold.as_ref().or(fonts.regular.as_ref())
        } else {
            fonts.regular.as_ref()
        }
    }

    /// Width of a line of text and the font's ascent, descent (negative)
    /// and line gap, in pixels.
    fn measure(&self, text: &str) -> (f32, f32, f32, f32) {
        use ab_glyph::{Font as _, ScaleFont as _};
        let Some(font) = self.font() else {
            return (0., 0., 0., 0.);
        };
        let scaled = font.as_scaled(
            font.pt_to_px_scale(self.font_px)
                .unwrap_or(ab_glyph::PxScale::from(self.font_px)),
        );
        let mut width = 0.;
        let mut last: Option<ab_glyph::GlyphId> = None;
        for c in text.chars() {
            let id = scaled.glyph_id(c);
            if let Some(prev) = last {
                width += scaled.kern(prev, id);
            }
            width += scaled.h_advance(id);
            last = Some(id);
        }
        (width, scaled.ascent(), scaled.descent(), scaled.line_gap())
    }

    /// Draws one line of text with its baseline starting at `(x, y)`,
    /// clipped to `clip` (`x0, y0, x1, y1` exclusive) when given.
    fn draw_text_line(&mut self, x: f32, y: f32, text: &str, clip: Option<(i32, i32, i32, i32)>) {
        use ab_glyph::{Font as _, ScaleFont as _};
        let Some(font) = self.font() else {
            return;
        };
        let scale = font
            .pt_to_px_scale(self.font_px)
            .unwrap_or(ab_glyph::PxScale::from(self.font_px));
        let scaled = font.as_scaled(scale);
        let color = self.pen_color;
        let mut caret = x;
        let mut last: Option<ab_glyph::GlyphId> = None;
        for c in text.chars() {
            let id = scaled.glyph_id(c);
            if let Some(prev) = last {
                caret += scaled.kern(prev, id);
            }
            let glyph = id.with_scale_and_position(scale, ab_glyph::point(caret, y));
            caret += scaled.h_advance(id);
            last = Some(id);
            if let Some(outlined) = font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                let mut coverage: Vec<(i32, i32, f32)> = Vec::new();
                outlined.draw(|gx, gy, cov| {
                    coverage.push((
                        bounds.min.x as i32 + gx as i32,
                        bounds.min.y as i32 + gy as i32,
                        cov,
                    ));
                });
                for (px, py, cov) in coverage {
                    if let Some((x0, y0, x1, y1)) = clip
                        && (px < x0 || py < y0 || px >= x1 || py >= y1)
                    {
                        continue;
                    }
                    self.blend(px, py, color, cov);
                }
            }
        }
    }

    /// `drawText(x, y, text)`: baseline at `y`.
    fn draw_text(&mut self, x: i32, y: i32, text: &str) {
        self.draw_text_line(x as f32, y as f32, text, None);
    }

    /// `drawText(x, y, w, h, flags, text)`: aligned in the rectangle,
    /// word-wrapped for `Qt::TextWordWrap`, clipped to it.
    fn draw_text_rect(&mut self, x: i32, y: i32, w: i32, h: i32, align: i32, text: &str) {
        let (_, ascent, descent, line_gap) = self.measure("");
        let line_height = ascent - descent;
        let mut lines: Vec<String> = Vec::new();
        if align & TEXT_WORD_WRAP != 0 {
            let mut current = String::new();
            for word in text.split(' ') {
                let candidate = if current.is_empty() {
                    word.to_owned()
                } else {
                    format!("{current} {word}")
                };
                if !current.is_empty() && self.measure(&candidate).0 > w as f32 {
                    lines.push(std::mem::take(&mut current));
                    current = word.to_owned();
                } else {
                    current = candidate;
                }
            }
            lines.push(current);
        } else {
            lines.push(text.to_owned());
        }
        let total = lines.len() as f32 * line_height + (lines.len() as f32 - 1.) * line_gap;
        let top = if align & ALIGN_BOTTOM != 0 {
            (y + h) as f32 - total
        } else if align & ALIGN_VCENTER != 0 {
            y as f32 + (h as f32 - total) / 2.
        } else {
            y as f32
        };
        let clip = Some((x, y, x + w, y + h));
        for (k, line) in lines.iter().enumerate() {
            let lw = self.measure(line).0;
            let left = if align & ALIGN_RIGHT != 0 {
                (x + w) as f32 - lw
            } else if align & ALIGN_HCENTER != 0 {
                x as f32 + (w as f32 - lw) / 2.
            } else {
                x as f32
            };
            let baseline = top + ascent + k as f32 * (line_height + line_gap);
            self.draw_text_line(left.round(), baseline.round(), line, clip);
        }
    }
}

/// The window, with the `gui` feature: Slint in place of Qt.  The class
/// `PlaxWindow` (`qtplax.h:134`) is the Slint component of that name and the
/// closures `start_plax_app` registers on it; `PlaxThread` is the Fortran
/// thread it spawns.
#[cfg(feature = "gui")]
mod window {
    use super::{FSTRING_LEN, lock, paint_event, resize_event, start_plax_geometry};
    use crate::imod::libcfshr::b3dutil::{CArg, c_format_bytes, exit};
    use slint::ComponentHandle as _;

    slint::slint! {
        import { LineEdit, Button } from "std-widgets.slint";

        export component PlaxWindow inherits Window {
            in property <image> plot;
            in property <string> window-title;
            in property <string> tooltip-text;
            in property <string> coord-text;
            in-out property <bool> saving: false;
            in-out property <string> save-name;
            in-out property <length> click-x;
            in-out property <length> click-y;
            callback left-clicked(length, length);
            callback save-png(string);
            title: root.window-title;
            background: black;
            Image {
                x: 0; y: 0;
                width: parent.width;
                height: parent.height;
                source: root.plot;
                image-fit: fill;
                image-rendering: pixelated;
            }
            area := TouchArea {
                pointer-event(event) => {
                    if (event.kind == PointerEventKind.down) {
                        root.click-x = self.mouse-x;
                        root.click-y = self.mouse-y;
                        if (event.button == PointerEventButton.left) {
                            root.left-clicked(self.mouse-x, self.mouse-y);
                        } else if (event.button == PointerEventButton.right) {
                            menu.show();
                        }
                    }
                }
            }
            if area.has-hover && root.tooltip-text != "" && !root.saving : Rectangle {
                x: min(area.mouse-x + 12px, root.width - tip.preferred-width - 8px);
                y: min(area.mouse-y + 16px, root.height - tip.preferred-height - 8px);
                width: tip.preferred-width + 8px;
                height: tip.preferred-height + 6px;
                background: #ffffdc;
                border-width: 1px;
                border-color: #000000;
                tip := Text { text: root.tooltip-text; color: black; x: 4px; y: 3px; }
            }
            coord-popup := PopupWindow {
                x: root.click-x;
                y: root.click-y;
                Rectangle {
                    background: #f0f0f0;
                    border-width: 1px;
                    border-color: #808080;
                    width: coord.preferred-width + 16px;
                    height: coord.preferred-height + 10px;
                    coord := Text { text: root.coord-text; color: black; x: 8px; y: 5px; }
                }
            }
            menu := PopupWindow {
                x: root.click-x;
                y: root.click-y;
                width: 120px;
                height: 50px;
                close-policy: close-on-click-outside;
                Rectangle {
                    background: #f0f0f0;
                    border-width: 1px;
                    border-color: #808080;
                    width: 120px;
                    height: 50px;
                    save-item := TouchArea {
                        x: 0; y: 0; width: parent.width; height: 25px;
                        clicked => { menu.close(); root.save-name = ""; root.saving = true; }
                        Rectangle {
                            background: save-item.has-hover ? #3070c0 : transparent;
                            Text { x: 8px; text: "Save to PNG"; color: save-item.has-hover ? white : black; }
                        }
                    }
                    // "Print": Qt opened a print dialog; there is no printer
                    // backend here, so the item is shown and does nothing.
                    Rectangle {
                        x: 0; y: 25px; width: parent.width; height: 25px;
                        Text { x: 8px; text: "Print"; color: #909090; }
                    }
                }
            }
            if root.saving : Rectangle {
                width: 360px;
                height: 90px;
                background: #f0f0f0;
                border-width: 1px;
                border-color: #808080;
                Text { x: 10px; y: 8px; text: "File name for saving as PNG"; color: black; }
                name := LineEdit {
                    x: 10px; y: 28px; width: 340px; height: 26px;
                    text <=> root.save-name;
                    accepted => { root.save-png(root.save-name); root.saving = false; }
                    init => { self.focus(); }
                }
                Button {
                    x: 190px; y: 60px; width: 75px; height: 24px; text: "Save";
                    clicked => { root.save-png(root.save-name); root.saving = false; }
                }
                Button {
                    x: 275px; y: 60px; width: 75px; height: 24px; text: "Cancel";
                    clicked => { root.saving = false; }
                }
            }
            public function show-coords() { coord-popup.show(); }
        }

        // The `-message` label (`new QLabel(sMessage, NULL)`), a window of
        // its own beside the plot.
        export component PlaxMessage inherits Window {
            in property <string> message;
            background: #f0f0f0;
            VerticalLayout {
                padding: 6px;
                Text { text: root.message; color: black; }
            }
        }
    }

    /// `startPlaxApp` (`qtplax.cpp:542`), which makes the `PlaxWindow`
    /// (`PlaxWindow::PlaxWindow`, `qtplax.cpp:154`), and the main thread's part of
    /// `plax_initialize` (`:465-476`): creates the window, starts the
    /// Fortran thread (`PlaxThread::run`) and runs the event loop.  Returns
    /// false, before anything is started, when no window can be made.
    pub(super) fn start_plax_app(com_file: [u8; FSTRING_LEN], main: fn(&[u8])) -> bool {
        let Ok(widget) = PlaxWindow::new() else {
            return false;
        };
        let (width, height, left, top, title, tool_tip, message) = {
            let mut plax = lock();
            start_plax_geometry(&mut plax);
            (
                plax.plax_width,
                plax.plax_height,
                plax.plax_left,
                plax.plax_top,
                format!(
                    "{}   (Left click for position, right click to save as PNG or print)",
                    plax.prog_name
                ),
                plax.tool_tip.clone(),
                plax.message.clone(),
            )
        };
        widget
            .window()
            .set_size(slint::LogicalSize::new(width as f32, height as f32));
        widget
            .window()
            .set_position(slint::LogicalPosition::new(left as f32, top as f32));
        widget.set_window_title(title.into());
        if let Some(tip) = tool_tip {
            widget.set_tooltip_text(tip.into());
        }
        let message_window = message.and_then(|text| {
            let label = PlaxMessage::new().ok()?;
            label.set_message(text.into());
            label.window().set_position(slint::LogicalPosition::new(
                (left + width + 10) as f32,
                (top + height / 2) as f32,
            ));
            let _ = label.show();
            Some(label)
        });

        // `PlaxWindow::closeEvent`: ignore close events unless exiting
        widget.window().on_close_requested(close_event);

        // `mousePressEvent`: the left button shows the position in user
        // units once a drawing scale is set
        let weak = widget.as_weak();
        widget.on_left_clicked(move |mx, my| {
            let Some(widget) = weak.upgrade() else {
                return;
            };
            let text = {
                let plax = lock();
                if !plax.user_scale_set {
                    return;
                }
                let xx = mx / plax.scale_x;
                let yy = 1023. - my / plax.scale_y;
                let mut text = c_format_bytes(
                    "%.3g, %.3g",
                    &[
                        CArg::Dbl(((xx - plax.user_xadd) / plax.user_xscale) as f64),
                        CArg::Dbl(((yy - plax.user_yadd) / plax.user_yscale) as f64),
                    ],
                );
                text.retain(|&c| c != 0);
                String::from_utf8_lossy(&text).into_owned()
            };
            widget.set_coord_text(text.into());
            widget.invoke_show_coords();
        });

        // `savePNGslot`: the name entered, `.png` added when missing
        widget.on_save_png(|name| {
            let mut filename = name.to_string();
            if filename.is_empty() {
                return;
            }
            if !filename.to_lowercase().ends_with(".png") {
                filename += ".png";
            }
            super::save_png_to_file(&filename, false);
        });

        // `PlaxThread`: the Fortran program on its own thread
        let spawned = std::thread::Builder::new()
            .name("plax".into())
            .spawn(move || {
                main(&com_file);
                exit(0);
            });
        if spawned.is_err() {
            return false;
        }

        // `redrawSlot`, `timerEvent` (whose one-pixel resizes only forced Qt
        // to repaint; here the timer redraws directly) and the paint and
        // resize events, polled
        let weak = widget.as_weak();
        let timer = slint::Timer::default();
        let mut shown = false;
        let mut last_size = (0, 0);
        timer.start(
            slint::TimerMode::Repeated,
            std::time::Duration::from_millis(30),
            move || {
                if let Some(widget) = weak.upgrade() {
                    timer_event(&widget, &mut shown, &mut last_size);
                }
            },
        );
        let _ = slint::run_event_loop_until_quit();
        drop(message_window);
        true
    }

    /// Original `PlaxWindow::closeEvent` (`qtplax.cpp:171`): close events
    /// are ignored unless exiting.
    fn close_event() -> slint::CloseRequestResponse {
        if lock().exit_on_close != 0 {
            exit(0);
        }
        slint::CloseRequestResponse::KeepWindowShown
    }

    /// Original `PlaxWindow::timerEvent` (`qtplax.cpp:204`), here a 30 ms
    /// timer that delivers the redraw signal (`redrawSlot`: hide, show, or
    /// redraw) and the window's resize and paint events (`resizeEvent`,
    /// `paintEvent`).  The source's timer resized the window by a pixel to
    /// make Qt repaint; this one redraws directly.
    fn timer_event(widget: &PlaxWindow, shown: &mut bool, last_size: &mut (i32, i32)) {
        let (open, signal) = {
            let mut plax = lock();
            let signal = plax.signal;
            plax.signal = false;
            (plax.plax_open, signal)
        };
        let mut redraw = false;
        if signal {
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if open == 0 {
                if *shown {
                    let _ = widget.hide();
                    *shown = false;
                }
            } else {
                if !*shown {
                    let _ = widget.show();
                    *shown = true;
                }
                if open < 0 {
                    lock().plax_open = 1;
                }
                redraw = true;
            }
        }
        if *shown {
            let scale = widget.window().scale_factor();
            let size = widget.window().size().to_logical(scale);
            let size = (size.width as i32, size.height as i32);
            if size != *last_size && size.0 > 0 && size.1 > 0 {
                *last_size = size;
                resize_event(&mut lock(), size.0, size.1);
                redraw = true;
            }
        }
        if redraw {
            let (w, h, rgb) = paint_event();
            let buffer = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
                &rgb, w as u32, h as u32,
            );
            widget.set_plot(slint::Image::from_rgb8(buffer));
        }
    }
}
