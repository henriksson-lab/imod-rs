//! Translation of `IMOD/flib/subrs/graphics/psf.c`: the Fortran interface
//! to the PostScript functions of `ps.c` ([`super::ps`]).
//!
//! The file statics `ps` and `lastsize` are thread-local (the Fortran
//! program runs on one thread).

use super::ps::{
    Ps, ps_close, ps_draw_circle, ps_draw_point, ps_draw_quadrangle, ps_draw_text,
    ps_draw_triangle, ps_draw_vector, ps_open, ps_page, ps_set_color, ps_set_font,
    ps_set_line_width, ps_set_point,
};
use std::cell::{Cell, RefCell};

thread_local! {
    /// `static PS *ps = NULL;`
    static PS_FILE: RefCell<Option<Box<Ps>>> = const { RefCell::new(None) };
    /// `static int lastsize = 0;`
    static LASTSIZE: Cell<i32> = const { Cell::new(0) };
}

/// `static char defaultFont[] = "Helvetica";`
const DEFAULT_FONT: &[u8] = b"Helvetica";

fn with_ps(f: impl FnOnce(&mut Ps)) {
    PS_FILE.with_borrow_mut(|ps| {
        if let Some(ps) = ps.as_mut() {
            f(ps);
        }
    });
}

/// Original `psopen` (`psf.c:51`): `filename` is the Fortran string.
pub fn psopen(filename: &[u8], lm: f32, bm: f32, dpi: f32) -> i32 {
    let end = filename
        .iter()
        .rposition(|&c| c != b' ')
        .map_or(0, |p| p + 1);
    let fname = String::from_utf8_lossy(&filename[..end]).into_owned();
    let ddpi = dpi as f64;
    let dlm = lm as f64;
    let dbm = bm as f64;
    let ps = ps_open(&fname, ddpi, dlm, dbm);

    /* DNM 3/23/01: set font size to zero so a font specification will be
    output to every new file */
    LASTSIZE.set(0);
    let failed = ps.is_none();
    PS_FILE.with_borrow_mut(|slot| *slot = ps);
    if failed {
        return -1;
    }
    0
}

/// Original `pslinewidth` (`psf.c:70`).
pub fn pslinewidth(width: f32) {
    let dw = width as f64;
    with_ps(|ps| ps_set_line_width(ps, dw));
}

/// Original `pssetcolor` (`psf.c:77`).
pub fn pssetcolor(red: i32, green: i32, blue: i32) {
    with_ps(|ps| ps_set_color(ps, red, green, blue));
}

/// Original `pspoint` (`psf.c:83`).
pub fn pspoint(ix: f32, iy: f32) {
    let x = ix as f64;
    let y = iy as f64;
    with_ps(|ps| ps_draw_point(ps, x, y));
}

/// Original `psfirstpoint` (`psf.c:91`).
pub fn psfirstpoint(ix: f32, iy: f32) {
    let x = ix as f64;
    let y = iy as f64;
    with_ps(|ps| ps_set_point(ps, x, y));
}

/// Original `psvector` (`psf.c:99`).
pub fn psvector(ix: f32, iy: f32) {
    let x = ix as f64;
    let y = iy as f64;
    with_ps(|ps| ps_draw_vector(ps, x, y));
}

/// Original `pscircle` (`psf.c:107`).
pub fn pscircle(ix: f32, iy: f32, irad: f32, fill: i32) {
    let x = ix as f64;
    let y = iy as f64;
    let rad = irad as f64;
    with_ps(|ps| ps_draw_circle(ps, x, y, rad, fill));
}

/// Original `pstriangle` (`psf.c:116`).
pub fn pstriangle(ix: &[f32], iy: &[f32], fill: i32) {
    let mut x = [0f64; 3];
    let mut y = [0f64; 3];
    for i in 0..3 {
        x[i] = ix[i] as f64;
        y[i] = iy[i] as f64;
    }
    with_ps(|ps| ps_draw_triangle(ps, &x, &y, fill));
}

/// Original `psquadrangle` (`psf.c:128`).
pub fn psquadrangle(ix: &[f32], iy: &[f32], fill: i32) {
    let mut x = [0f64; 4];
    let mut y = [0f64; 4];
    for i in 0..4 {
        x[i] = ix[i] as f64;
        y[i] = iy[i] as f64;
    }
    with_ps(|ps| ps_draw_quadrangle(ps, &x, &y, fill));
}

/// Original `psframe` (`psf.c:140`).
pub fn psframe() {
    with_ps(ps_page);
}

/// Original `pswritetext` (`psf.c:146`): `text` is the Fortran string,
/// whole (`text_size` is its length).
pub fn pswritetext(ix: f32, iy: f32, text: &[u8], jsize: i32, jor: i32, jctr: i32) {
    let x = ix;
    let y = iy;

    let env_font = std::env::var_os("IMOD_PS_FONT").map(|v| v.into_encoded_bytes());
    let use_font: &[u8] = env_font.as_deref().unwrap_or(DEFAULT_FONT);
    // `memcpy(ctext, text, text_size); ctext[text_size] = 0x00;`: the C
    // string ends at a NUL inside the text, if any
    let ctext = &text[..text.iter().position(|&c| c == 0).unwrap_or(text.len())];

    if LASTSIZE.get() != jsize {
        LASTSIZE.set(jsize);
        with_ps(|ps| ps_set_font(ps, use_font, jsize));
    }

    with_ps(|ps| ps_draw_text(ps, ctext, x as f64, y as f64, jor, jctr));
}

/// Original `psclose` (`psf.c:172`).
///
/// Fixed in translation (BUGS.md, `psf.c`): the source frees the structure
/// and leaves the static pointer to it, so a drawing call before the next
/// `psopen` uses freed memory; here the pointer is cleared and such a call
/// draws nothing.
pub fn psclose() {
    PS_FILE.with_borrow_mut(|slot| {
        if let Some(ps) = slot.take() {
            ps_close(ps);
        }
    });
}
