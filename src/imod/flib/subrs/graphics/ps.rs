//! Translation of `IMOD/flib/subrs/graphics/ps.c` and `ps.h`: PostScript
//! drawing routines.

use crate::imod::libcfshr::b3dutil::{
    CArg, IMOD_BUILD_DATE, IMOD_BUILD_TIME, ImodFile, c_format_bytes,
};
use std::io::Write as _;

/// Text placement defines (`ps.h:27-32`).
pub const PS_CENTERED: i32 = 4;
pub const PS_LEFT_JUST: i32 = 3;
pub const PS_RIGHT_JUST: i32 = 2;
pub const PS_VERT_CENTERED: i32 = 0;
pub const PS_VERT_LEFT_JUST: i32 = -1;
pub const PS_VERT_RIGHT_JUST: i32 = 1;

/// Superscript, subscript, and symbol font defines (`ps.h:35-47`).
const ESC_CHAR: u8 = b'^';
const SUB_CHAR: u8 = b'B';
const SUP_CHAR: u8 = b'P';
const SYM_CHAR: u8 = b'S';
const PS_REGULAR: i32 = 0;
const PS_SUPERSCRIPT: i32 = 1;
const PS_SUBSCRIPT: i32 = 2;
const PS_SYMBOL: i32 = 3;
const MAXSEG: usize = 20;
const SUPSCALE: f32 = 0.8;
const SUBSCALE: f32 = 0.8;
const SYMSCALE: f32 = 1.0;
const SUPRISE: f32 = 0.33;
const SUBDROP: f32 = 0.33;

/// `typedef struct {...} PS` (`ps.h:6`).
///
/// Fixed in translation (BUGS.md, `ps.c`): `PSopen` `malloc`s the structure
/// and never sets `red`, `green`, `blue` or `fontSize`, which `PSsetColor`
/// and `PSdrawText` then read; here they start at 0, black being the color
/// the file sets (`0 setgray`) and a zero font size the "no font yet" state
/// `PSdrawText` tests for.
pub struct Ps {
    /// A C stdio stream (`ImodFile`), flushed by `exit` as `fopen`'s is.
    pub fp: ImodFile,
    /// resolution in dots per inch
    pub dpi: f64,
    /// Current position.
    pub cx: f64,
    pub cy: f64,
    pub xoffset: f64,
    pub yoffset: f64,
    pub line_width: f64,
    pub font_center: i32,
    pub font_size: f64,
    pub point_size: f64,
    pub point_style: i32,
    pub red: i32,
    pub green: i32,
    pub blue: i32,
    pub font_name: Vec<u8>,
}

impl Ps {
    fn put(&mut self, fmt: &str, args: &[CArg]) {
        let _ = self.fp.write_all(&c_format_bytes(fmt, args));
    }
}

/// Original `PSopen` (`ps.c:13`).
pub fn ps_open(filename: &str, dpi: f64, x: f64, y: f64) -> Option<Box<Ps>> {
    let Some(fp) = ImodFile::open(filename, "w") else {
        // `perror("PSopen: error getting output file. ")`
        let error = std::io::Error::last_os_error();
        let text = error.to_string();
        let text = match error.raw_os_error() {
            Some(errno) => text
                .strip_suffix(&format!(" (os error {errno})"))
                .unwrap_or(&text)
                .to_owned(),
            None => text,
        };
        eprintln!("PSopen: error getting output file. : {text}");
        return None;
    };
    let mut ps = Box::new(Ps {
        fp,
        dpi,
        cx: 0.0,
        cy: 0.0,
        xoffset: x,
        yoffset: y,
        line_width: 1.,
        font_center: 0,
        font_size: 0.,
        point_size: 1.0 / dpi,
        point_style: 0,
        red: 0,
        green: 0,
        blue: 0,
        font_name: Vec::new(),
    });

    ps.put("%%!PS-Boulder-Laboratory-for-3D-EM-of-Cells\n", &[]);
    ps.put(
        "%%%%Created by the BL3DEMC PS module. Compiled %s %s\n",
        &[CArg::Str(IMOD_BUILD_DATE), CArg::Str(IMOD_BUILD_TIME)],
    );

    ps.put("/inch {%g mul} def\n", &[CArg::Dbl(dpi)]);
    ps.put(
        "%g %g scale\n",
        &[CArg::Dbl(72.0 / dpi), CArg::Dbl(72.0 / dpi)],
    );
    ps.put("%g inch %g inch translate\n", &[CArg::Dbl(x), CArg::Dbl(y)]);
    ps.put("1 setlinewidth\n0 setgray\n", &[]);

    ps.put("\n%%%%-----Define procedures-----\n", &[]);
    ps.put(
        "/drawpoint %% stack: x y\n{newpath moveto 1 0 rlineto 0 1 rlineto -1 0 rlineto closepath fill} def\n",
        &[],
    );
    ps.put("/drawline { newpath moveto lineto stroke } def\n", &[]);
    ps.put("/opencircle { newpath 0 360 arc stroke } def\n", &[]);
    ps.put("/fillcircle { newpath 0 360 arc fill } def\n", &[]);
    ps.put(
        "/opentri { newpath moveto lineto lineto closepath stroke } def\n",
        &[],
    );
    ps.put(
        "/filltri { newpath moveto lineto lineto closepath fill } def\n",
        &[],
    );
    ps.put(
        "/openquad { newpath moveto lineto lineto lineto closepath stroke } def\n",
        &[],
    );
    ps.put(
        "/fillquad { newpath moveto lineto lineto lineto closepath fill } def\n",
        &[],
    );
    ps.put(
        "/rightshow %%stk: string\n{dup stringwidth pop neg 0 rmoveto show } def\n",
        &[],
    );
    ps.put(
        "/centershow %%stk: string\n{dup stringwidth pop neg 2 div 0 rmoveto show } def\n",
        &[],
    );
    ps.put("\n%%%%-----Graphics output-----\n", &[]);
    Some(ps)
}

/// Original `PSclose` (`ps.c:63`).
pub fn ps_close(ps: Box<Ps>) {
    let mut ps = ps;
    let _ = ps.fp.flush();
}

/// Original `PSpage` (`ps.c:69`).
pub fn ps_page(ps: &mut Ps) {
    ps.put("showpage\n", &[]);
    let (dpi, x, y) = (ps.dpi, ps.xoffset, ps.yoffset);
    ps.put(
        "%g %g scale\n",
        &[CArg::Dbl(72.0 / dpi), CArg::Dbl(72.0 / dpi)],
    );
    ps.put("%g inch %g inch translate\n", &[CArg::Dbl(x), CArg::Dbl(y)]);
}

/// Original `PSdrawPoint` (`ps.c:76`).
pub fn ps_draw_point(ps: &mut Ps, x: f64, y: f64) {
    ps.put("%g inch %g inch drawpoint\n", &[CArg::Dbl(x), CArg::Dbl(y)]);
}

/// Original `PSsetPoint` (`ps.c:81`).
pub fn ps_set_point(ps: &mut Ps, x: f64, y: f64) {
    ps.cx = x;
    ps.cy = y;
}

/// Original `PSsetLineWidth` (`ps.c:87`).
pub fn ps_set_line_width(ps: &mut Ps, width: f64) {
    if width != ps.line_width {
        ps.put("%g setlinewidth\n", &[CArg::Dbl(width)]);
    }
    ps.line_width = width;
}

/// Original `PSdrawVector` (`ps.c:94`).
pub fn ps_draw_vector(ps: &mut Ps, x: f64, y: f64) {
    let (cx, cy) = (ps.cx, ps.cy);
    ps.put(
        "%g inch %g inch %g inch %g inch drawline\n",
        &[CArg::Dbl(cx), CArg::Dbl(cy), CArg::Dbl(x), CArg::Dbl(y)],
    );
    ps.cx = x;
    ps.cy = y;
}

/// Original `PSdrawCircle` (`ps.c:102`).
pub fn ps_draw_circle(ps: &mut Ps, x: f64, y: f64, rad: f64, fill: i32) {
    ps.put(
        "%g inch %g inch %g inch %scircle\n",
        &[
            CArg::Dbl(x),
            CArg::Dbl(y),
            CArg::Dbl(rad),
            CArg::Str(if fill != 0 { "fill" } else { "open" }),
        ],
    );
}

/// Original `PSdrawTriangle` (`ps.c:108`).
pub fn ps_draw_triangle(ps: &mut Ps, x: &[f64; 3], y: &[f64; 3], fill: i32) {
    ps.put(
        "%g inch %g inch %g inch %g inch %g inch %g inch %stri\n",
        &[
            CArg::Dbl(x[0]),
            CArg::Dbl(y[0]),
            CArg::Dbl(x[1]),
            CArg::Dbl(y[1]),
            CArg::Dbl(x[2]),
            CArg::Dbl(y[2]),
            CArg::Str(if fill != 0 { "fill" } else { "open" }),
        ],
    );
}

/// Original `PSdrawQuadrangle` (`ps.c:114`).
pub fn ps_draw_quadrangle(ps: &mut Ps, x: &[f64; 4], y: &[f64; 4], fill: i32) {
    ps.put(
        "%g inch %g inch %g inch %g inch %g inch %g inch %g inch %g inch %squad\n",
        &[
            CArg::Dbl(x[0]),
            CArg::Dbl(y[0]),
            CArg::Dbl(x[1]),
            CArg::Dbl(y[1]),
            CArg::Dbl(x[2]),
            CArg::Dbl(y[2]),
            CArg::Dbl(x[3]),
            CArg::Dbl(y[3]),
            CArg::Str(if fill != 0 { "fill" } else { "open" }),
        ],
    );
}

/// Original `PSsetFont` (`ps.c:121`).
pub fn ps_set_font(ps: &mut Ps, name: &[u8], size: i32) {
    ps.font_size = size as f64 * ps.dpi / 72.0;
    ps.font_name = name.to_vec();
    let font_size = ps.font_size;
    ps.put(
        "/%s findfont %g scalefont setfont\n",
        &[CArg::Bytes(name), CArg::Dbl(font_size)],
    );
}

/// Original `PSsetColor` (`ps.c:129`).
pub fn ps_set_color(ps: &mut Ps, red: i32, green: i32, blue: i32) {
    if ps.red == red && ps.green == green && ps.blue == blue {
        return;
    }
    ps.red = 0.max(255.min(red));
    ps.green = 0.max(255.min(green));
    ps.blue = 0.max(255.min(blue));
    let (r, g, b) = (ps.red, ps.green, ps.blue);
    ps.put(
        "%f %f %f setrgbcolor\n",
        &[
            CArg::Dbl(r as f64 / 255.),
            CArg::Dbl(g as f64 / 255.),
            CArg::Dbl(b as f64 / 255.),
        ],
    );
}

/// Original `PSdrawText` (`ps.c:140`).
///
/// Fixed in translation (BUGS.md, `ps.c`): the segment arrays hold
/// `MAXSEG` (20) entries and the source writes past them for a string with
/// more special characters; here a special character that would need a
/// segment beyond them (one is kept for the text after the last special
/// character) stays in the regular text.  For an empty string the
/// source tests `typestrng[0]`, which nothing has set; here an empty string
/// is regular text.
pub fn ps_draw_text(ps: &mut Ps, text: &[u8], x: f64, y: f64, rotation: i32, placement: i32) {
    let mut indstr = [0usize; MAXSEG];
    let mut indend = [0usize; MAXSEG];
    let mut typestrng = [PS_REGULAR; MAXSEG];
    let mut nseg = 0usize;
    let mut lasttext = 0usize;

    if ps.font_size <= 0.0 {
        eprintln!("PSdrawText: bad font size");
        return;
    }

    ps.put("%g inch %g inch moveto\n", &[CArg::Dbl(x), CArg::Dbl(y)]);

    if rotation != 0 {
        ps.put("gsave\n %d rotate\n", &[CArg::Int(rotation as i64)]);
    }

    /* Copy text string, putting \ before any parentheses to keep PS happy
    and converting \nnn to single character */

    let mut lentext = text.len();
    let mut tcpy: Vec<u8> = Vec::with_capacity(lentext + 20);
    let mut i = 0usize;
    while i < lentext {
        if text[i] == b'(' || text[i] == b')' {
            tcpy.push(b'\\');
        }
        if text[i] == b'\\' {
            let mut ii = i + 1;
            let mut ndig = 0;
            let mut octcode: i32 = 0;
            while ndig < 3 && ii < lentext {
                if text[ii].is_ascii_digit() {
                    octcode = 8 * octcode + (text[ii] as i32 - 48);
                    ndig += 1;
                    ii += 1;
                } else {
                    break;
                }
            }
            if ndig == 3 {
                tcpy.push(octcode as u8);
                i = ii - 1;
            } else {
                tcpy.push(text[i]);
            }
        } else {
            tcpy.push(text[i]);
        }
        i += 1;
    }
    lentext = tcpy.len();
    let at = |k: usize| -> u8 { tcpy.get(k).copied().unwrap_or(0) };
    // `%s` (and `strncpy`) of the C string stops at a NUL that `\000` put in
    let c_str = |bytes: &[u8]| -> Vec<u8> {
        bytes[..bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len())].to_vec()
    };

    /* Parse the string into regular text and special characters */

    let mut i = 0usize;
    loop {
        let c = at(i);
        i += 1;
        if c == ESC_CHAR {
            let code = at(i);
            i += 1;
            if (code == SUP_CHAR || code == SUB_CHAR || code == SYM_CHAR) && nseg + 3 <= MAXSEG {
                if i >= 3 && i - 3 >= lasttext {
                    typestrng[nseg] = PS_REGULAR;
                    indstr[nseg] = lasttext;
                    indend[nseg] = i - 3;
                    nseg += 1;
                }
                typestrng[nseg] = match code {
                    SUP_CHAR => PS_SUPERSCRIPT,
                    SUB_CHAR => PS_SUBSCRIPT,
                    _ => PS_SYMBOL,
                };
                indstr[nseg] = i;
                nseg += 1;
                i += 1;
                lasttext = i;
            }
        }
        if i >= lentext {
            break;
        }
    }

    if lasttext < lentext && nseg < MAXSEG {
        typestrng[nseg] = PS_REGULAR;
        indstr[nseg] = lasttext;
        indend[nseg] = lentext - 1;
        nseg += 1;
    }

    match placement {
        PS_VERT_CENTERED | PS_VERT_RIGHT_JUST | PS_VERT_LEFT_JUST => {
            let size = ps.font_size;
            ps.put("0 %g rmoveto\n", &[CArg::Dbl(size * -0.33)]);
        }
        _ => {}
    }

    /* Use the defined procedures for pure text */

    let font_name = ps.font_name.clone();
    let font_size = ps.font_size;
    if nseg < 2 && typestrng[0] == PS_REGULAR {
        match placement {
            PS_CENTERED | PS_VERT_CENTERED => {
                ps.put("(%s) centershow\n", &[CArg::Bytes(&c_str(&tcpy))]);
            }
            PS_RIGHT_JUST | PS_VERT_RIGHT_JUST => {
                ps.put("(%s) rightshow\n", &[CArg::Bytes(&c_str(&tcpy))]);
            }
            PS_LEFT_JUST | PS_VERT_LEFT_JUST => {
                ps.put("(%s) show\n", &[CArg::Bytes(&c_str(&tcpy))]);
            }
            _ => {}
        }
    } else {
        let segment = |iseg: usize| -> &[u8] {
            let start = indstr[iseg].min(tcpy.len());
            let end = (indend[iseg] + 1).min(tcpy.len()).max(start);
            &tcpy[start..end]
        };

        /* Otherwise first output commands to add up the string width */

        if placement != PS_LEFT_JUST && placement != PS_VERT_LEFT_JUST {
            ps.put("0\n", &[]);
            for iseg in 0..nseg {
                if typestrng[iseg] == PS_REGULAR {
                    ps.put(
                        "(%s) stringwidth pop add\n",
                        &[CArg::Bytes(&c_str(segment(iseg)))],
                    );
                } else {
                    match typestrng[iseg] {
                        PS_SUPERSCRIPT => ps.put(
                            "/%s findfont %g scalefont setfont\n",
                            &[
                                CArg::Bytes(&font_name),
                                CArg::Dbl(font_size * SUPSCALE as f64),
                            ],
                        ),
                        PS_SUBSCRIPT => ps.put(
                            "/%s findfont %g scalefont setfont\n",
                            &[
                                CArg::Bytes(&font_name),
                                CArg::Dbl(font_size * SUBSCALE as f64),
                            ],
                        ),
                        _ => ps.put(
                            "/Symbol findfont %g scalefont setfont\n",
                            &[CArg::Dbl(font_size * SYMSCALE as f64)],
                        ),
                    }

                    ps.put("(%c) stringwidth pop add\n", &[CArg::Chr(at(indstr[iseg]))]);
                    ps.put(
                        "/%s findfont %g scalefont setfont\n",
                        &[CArg::Bytes(&font_name), CArg::Dbl(font_size)],
                    );
                }
            }

            /* Shift for centered and right justified text */

            if placement == PS_CENTERED || placement == PS_VERT_CENTERED {
                ps.put("neg 2 div 0 rmoveto\n", &[]);
            } else {
                ps.put("neg 0 rmoveto\n", &[]);
            }
        }

        /* Finally, output commands to do all text and special characters */

        for iseg in 0..nseg {
            if typestrng[iseg] == PS_REGULAR {
                ps.put("(%s) show\n", &[CArg::Bytes(&c_str(segment(iseg)))]);
            } else {
                let ch = at(indstr[iseg]);
                match typestrng[iseg] {
                    PS_SUPERSCRIPT => {
                        ps.put(
                            "/%s findfont %g scalefont setfont\n",
                            &[
                                CArg::Bytes(&font_name),
                                CArg::Dbl(font_size * SUPSCALE as f64),
                            ],
                        );
                        ps.put("0 %g rmoveto\n", &[CArg::Dbl(font_size * SUPRISE as f64)]);
                        ps.put("(%c) show\n", &[CArg::Chr(ch)]);
                        ps.put("0 %g rmoveto\n", &[CArg::Dbl(-font_size * SUPRISE as f64)]);
                    }
                    PS_SUBSCRIPT => {
                        ps.put(
                            "/%s findfont %g scalefont setfont\n",
                            &[
                                CArg::Bytes(&font_name),
                                CArg::Dbl(font_size * SUBSCALE as f64),
                            ],
                        );
                        ps.put("0 %g rmoveto\n", &[CArg::Dbl(-font_size * SUBDROP as f64)]);
                        ps.put("(%c) show\n", &[CArg::Chr(ch)]);
                        ps.put("0 %g rmoveto\n", &[CArg::Dbl(font_size * SUBDROP as f64)]);
                    }
                    _ => {
                        ps.put(
                            "/Symbol findfont %g scalefont setfont\n",
                            &[CArg::Dbl(font_size * SYMSCALE as f64)],
                        );
                        ps.put("(%c) show\n", &[CArg::Chr(ch)]);
                    }
                }
                ps.put(
                    "/%s findfont %g scalefont setfont\n",
                    &[CArg::Bytes(&font_name), CArg::Dbl(font_size)],
                );
            }
        }
    }
    if rotation != 0 {
        ps.put("grestore\n", &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drawn(text: &[u8], placement: i32) -> String {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-ps-text-{}-{}.ps",
            std::process::id(),
            text.len()
        ));
        let mut ps = ps_open(path.to_str().unwrap(), 300., 0.5, 1.75).unwrap();
        ps_set_font(&mut ps, b"Helvetica", 12);
        let start = {
            let _ = ps.fp.flush();
            std::fs::read(&path).unwrap().len()
        };
        ps_draw_text(&mut ps, text, 1., 2., 0, placement);
        ps_close(ps);
        let all = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        String::from_utf8_lossy(&all[start..]).into_owned()
    }

    /// `BUGS.md` (`ps.c`), defined behaviour: an empty string is regular
    /// text (the source tests a segment type it never set).
    #[test]
    fn empty_text_is_regular_text() {
        assert_eq!(
            drawn(b"", PS_CENTERED),
            "1 inch 2 inch moveto\n() centershow\n"
        );
    }

    /// `BUGS.md` (`ps.c`), defined behaviour: special characters past the
    /// 20 segments stay in the regular text instead of being written past
    /// the arrays.
    #[test]
    fn segments_past_the_limit_stay_text() {
        let text = b"a^Pb".repeat(15);
        let out = drawn(&text, PS_LEFT_JUST);
        // 19 segments: 9 pairs of text and superscript, then the rest as text
        assert_eq!(out.matches("(b) show").count(), 9, "{out}");
        assert!(
            out.ends_with(&format!("({}) show\n", "a^Pb".repeat(6))),
            "{out}"
        );
    }

    /// `BUGS.md` (`ps.c`), defined behaviour: the color starts as black
    /// (`0 setgray`), so setting black first writes nothing.
    #[test]
    fn color_starts_black() {
        let path = std::env::temp_dir().join(format!("imod-rs-ps-color-{}.ps", std::process::id()));
        let mut ps = ps_open(path.to_str().unwrap(), 300., 0.5, 1.75).unwrap();
        ps_set_color(&mut ps, 0, 0, 0);
        ps_set_color(&mut ps, 255, 0, 0);
        ps_close(ps);
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(text.matches("setrgbcolor").count(), 1);
        assert!(text.ends_with("1.000000 0.000000 0.000000 setrgbcolor\n"));
    }
}
