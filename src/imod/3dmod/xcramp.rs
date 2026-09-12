//! Translation of `IMOD/3dmod/xcramp.cpp` and `xcramp.h`.
//!
//! The obsolete indexed-Qt-colormap branch is represented by `rgba == 0`.
//! Its `QGLColormap::setEntry` calls are an explicit viewer/Qt boundary; the
//! ramp construction shared by both source branches remains here.
#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};

use crate::imod::libcfshr::colormap::{cmap_convert_ramp, cmap_read_convert, cmap_standard_ramp};

/// `xbldrcoloramp` (`Cramp`) from `xcramp.h`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cramp {
    pub depth: i32,
    pub rgba: i32,
    pub scale: f32,
    pub rampsize: i32,
    pub rampbase: i32,
    pub blacklevel: i32,
    pub whitelevel: i32,
    pub reverse: i32,
    pub falsecolor: i32,
    pub noflevels: i32,
    pub clevel: i32,
    pub blacks: Vec<i32>,
    pub whites: Vec<i32>,
    pub minlevel: i32,
    pub maxlevel: i32,
    pub mapsize: i32,
    pub ramp: Vec<u32>,
    pub bramp: [u8; 256],
    pub cmap: Vec<u16>,
}

impl Default for Cramp {
    fn default() -> Self {
        Self {
            depth: 0,
            rgba: 0,
            scale: 0.,
            rampsize: 0,
            rampbase: 0,
            blacklevel: 0,
            whitelevel: 0,
            reverse: 0,
            falsecolor: 0,
            noflevels: 0,
            clevel: 0,
            blacks: Vec::new(),
            whites: Vec::new(),
            minlevel: 0,
            maxlevel: 0,
            mapsize: 0,
            ramp: Vec::new(),
            bramp: [0; 256],
            cmap: Vec::new(),
        }
    }
}

/// The QGLColormap calls in the non-`NEW_QTOPENGL` source branch.
pub trait XcrampColormapBoundary {
    fn set_entry(&mut self, index: i32, red: i32, green: i32, blue: i32);
}

/// `xcramp_allinit`.
///
/// `dummy` is the unused `int *dummy` argument of the `NEW_QTOPENGL` source
/// signature.  Passing `Some` has no effect, exactly as in that branch.
pub fn xcramp_allinit(
    depth: i32,
    _dummy: Option<&mut i32>,
    low: i32,
    high: i32,
    ushort: i32,
) -> Option<Cramp> {
    let mut cramp = Cramp {
        rgba: 1,
        blacklevel: 0,
        whitelevel: 255,
        minlevel: 0,
        maxlevel: 255,
        mapsize: 256,
        ..Cramp::default()
    };
    if ushort != 0 {
        cramp.rgba = 2;
        cramp.whitelevel = 65535;
        cramp.maxlevel = 65535;
        cramp.mapsize = 65536;
    }
    let mapsize = usize::try_from(cramp.mapsize).ok()?;
    cramp.ramp.try_reserve_exact(mapsize).ok()?;
    cramp.cmap.try_reserve_exact(mapsize).ok()?;
    cramp.ramp.resize(mapsize, 0);
    cramp.cmap.resize(mapsize, 0);
    cramp.reverse = 0;
    cramp.falsecolor = 0;
    cramp.depth = depth;
    cramp.rampbase = low;
    cramp.rampsize = high - low + 1;
    cramp.scale = (cramp.rampsize - 1) as f32 / 255.;
    cramp.clevel = 0;
    cramp.noflevels = 0;
    if xcramp_store_init(&mut cramp, 4) != 0 {
        return None;
    }
    Some(cramp)
}

/// `xcrampStoreInit`.
pub fn xcramp_store_init(cramp: &mut Cramp, size: i32) -> i32 {
    if size < 1 {
        return -1;
    }
    let Ok(size) = usize::try_from(size) else {
        return -1;
    };
    let mut blacks = Vec::new();
    let mut whites = Vec::new();
    if blacks.try_reserve_exact(size).is_err() || whites.try_reserve_exact(size).is_err() {
        return -1;
    }
    blacks.resize(size, 0);
    whites.resize(size, 255);
    blacks[0] = cramp.blacklevel;
    whites[0] = cramp.whitelevel;
    cramp.noflevels = size as i32;
    cramp.clevel = 0;
    cramp.blacks = blacks;
    cramp.whites = whites;
    0
}

/// `xcrampSelectIndex`.
pub fn xcramp_select_index(cramp: &mut Cramp, index: i32) -> i32 {
    if index < 0 || index >= cramp.noflevels {
        return -1;
    }
    let old = cramp.clevel as usize;
    cramp.blacks[old] = cramp.blacklevel;
    cramp.whites[old] = cramp.whitelevel;
    cramp.clevel = index;
    let current = index as usize;
    cramp.blacklevel = cramp.blacks[current];
    cramp.whitelevel = cramp.whites[current];
    0
}

/// `xcramp_level`.
pub fn xcramp_level(xcramp: &mut Cramp, black: i32, white: i32) -> i32 {
    xcramp.blacklevel += black;
    if xcramp.blacklevel < xcramp.minlevel || xcramp.blacklevel > xcramp.maxlevel {
        xcramp.blacklevel -= black;
    }
    xcramp.whitelevel += white;
    if xcramp.whitelevel < xcramp.minlevel || xcramp.whitelevel > xcramp.maxlevel {
        xcramp.whitelevel -= white;
    }
    xcramp_ramp(xcramp)
}

/// `xcramp_falsecolor`.
pub fn xcramp_falsecolor(xcramp: &mut Cramp, flag: i32) -> i32 {
    xcramp.falsecolor = flag;
    xcramp_ramp(xcramp)
}

/// `xcramp_reverse`.
pub fn xcramp_reverse(xcramp: &mut Cramp, flag: i32) -> i32 {
    xcramp.reverse = flag;
    xcramp_ramp(xcramp)
}

/// `xcramp_getlevels`.
pub fn xcramp_getlevels(xcramp: &Cramp) -> (i32, i32) {
    (xcramp.blacklevel, xcramp.whitelevel)
}

/// `xcramp_setlevels`.
pub fn xcramp_setlevels(xcramp: &mut Cramp, black: i32, white: i32) {
    xcramp.blacklevel = black;
    xcramp.whitelevel = white;
    xcramp_ramp(xcramp);
}

/// `xcramp_ramp` for the `NEW_QTOPENGL` (RGBA) branch.
pub fn xcramp_ramp(cr: &mut Cramp) -> i32 {
    if cr.falsecolor < 2 {
        if cr.blacklevel < cr.minlevel {
            cr.blacklevel = cr.minlevel;
        }
        if cr.whitelevel > cr.maxlevel {
            cr.whitelevel = cr.maxlevel;
        }
        if cr.blacklevel > cr.whitelevel {
            cr.blacklevel = cr.whitelevel;
        }
        for value in &mut cr.cmap[..cr.blacklevel as usize] {
            *value = 0;
        }
        for value in &mut cr.cmap[cr.whitelevel as usize..] {
            *value = 255;
        }
        let mut rampsize = cr.whitelevel - cr.blacklevel;
        if rampsize < 1 {
            rampsize = 1;
        }
        let slope = 256. / rampsize as f32;
        for i in cr.blacklevel..cr.whitelevel {
            cr.cmap[i as usize] = ((i - cr.blacklevel) as f32 * slope) as u16;
        }
        if cr.reverse != 0 {
            for value in &mut cr.cmap {
                *value = 255 - *value;
            }
        }
        if cr.scale <= 0. {
            return -1;
        }
    } else {
        for i in 0..256 {
            cr.cmap[i] = i as u16;
        }
    }
    for i in 0..256 {
        cr.bramp[i] = cr.cmap[i * cr.mapsize as usize / 256] as u8;
    }
    if cr.rgba != 0 {
        match cr.falsecolor {
            0 => {
                for ival in 0..cr.mapsize as usize {
                    let val = cr.cmap[ival] as u8;
                    cr.ramp[ival] = u32::from_ne_bytes([val, val, val, 0]);
                }
            }
            1 | 2 => {
                for ival in 0..cr.mapsize as usize {
                    let (red, green, blue) = xcramp_mapfalsecolor(cr.cmap[ival] as i32);
                    cr.ramp[ival] = u32::from_ne_bytes([red as u8, green as u8, blue as u8, 0]);
                }
            }
            _ => {}
        }
        return 0;
    }
    0
}

/// `xcramp_ramp`'s non-`NEW_QTOPENGL` `QGLColormap::setEntry` tail.
pub fn xcramp_ramp_indexed(cr: &mut Cramp, colormap: &mut dyn XcrampColormapBoundary) -> i32 {
    let result = xcramp_ramp(cr);
    if result != 0 || cr.rgba != 0 {
        return result;
    }
    match cr.falsecolor {
        0 => {
            for i in 0..cr.rampsize {
                let value = cr.cmap[(i as f32 / cr.scale) as usize] as i32;
                colormap.set_entry(i + cr.rampbase, value, value, value);
            }
        }
        1 => {
            for i in 0..cr.rampsize {
                let (red, green, blue) =
                    xcramp_mapfalsecolor(cr.cmap[(i as f32 / cr.scale) as usize] as i32);
                colormap.set_entry(i + cr.rampbase, red, green, blue);
            }
        }
        _ => {}
    }
    0
}

static CMAP: OnceLock<Mutex<([[u8; 256]; 3], i32)>> = OnceLock::new();

/// `xcramp_mapfalsecolor`.
pub fn xcramp_mapfalsecolor(gray: i32) -> (i32, i32, i32) {
    let cmap = CMAP.get_or_init(|| Mutex::new(([[0; 256]; 3], 0)));
    let mut cmap = cmap.lock().expect("xcramp false-color map mutex poisoned");
    if cmap.1 == 0 {
        cmap_convert_ramp(cmap_standard_ramp(), &mut cmap.0);
        cmap.1 = 1;
    }
    (
        cmap.0[0][gray as usize] as i32,
        cmap.0[1][gray as usize] as i32,
        cmap.0[2][gray as usize] as i32,
    )
}

/// `xcramp_readfalsemap`.
pub fn xcramp_readfalsemap(filename: &str) -> i32 {
    let cmap = CMAP.get_or_init(|| Mutex::new(([[0; 256]; 3], 0)));
    let mut cmap = cmap.lock().expect("xcramp false-color map mutex poisoned");
    let error = cmap_read_convert(filename, &mut cmap.0);
    if error == 0 {
        cmap.1 = 1;
    }
    error
}

/// `xcramp_copyfalsemap`.
pub fn xcramp_copyfalsemap(inmap: &[[u8; 256]; 3]) {
    let cmap = CMAP.get_or_init(|| Mutex::new(([[0; 256]; 3], 0)));
    let mut cmap = cmap.lock().expect("xcramp false-color map mutex poisoned");
    cmap.0 = *inmap;
    cmap.1 = 1;
}

/// `xcramp_restorefalsemap`.
pub fn xcramp_restorefalsemap() {
    let cmap = CMAP.get_or_init(|| Mutex::new(([[0; 256]; 3], 0)));
    let mut cmap = cmap.lock().expect("xcramp false-color map mutex poisoned");
    cmap_convert_ramp(cmap_standard_ramp(), &mut cmap.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_gray_ramp_and_stored_levels() {
        let mut ramp = xcramp_allinit(24, None, 0, 255, 0).unwrap();
        xcramp_setlevels(&mut ramp, 32, 223);
        assert_eq!(ramp.cmap[0], 0);
        assert_eq!(ramp.cmap[31], 0);
        assert_eq!(ramp.cmap[223], 255);
        assert_eq!(ramp.bramp[255], 255);
        assert_eq!(ramp.ramp[0].to_ne_bytes(), [0, 0, 0, 0]);
        assert_eq!(xcramp_select_index(&mut ramp, 1), 0);
        xcramp_setlevels(&mut ramp, 10, 200);
        assert_eq!(xcramp_select_index(&mut ramp, 0), 0);
        assert_eq!(xcramp_getlevels(&ramp), (32, 223));
    }

    #[test]
    fn source_false_color_and_reverse_paths() {
        xcramp_restorefalsemap();
        let mut ramp = xcramp_allinit(24, None, 0, 255, 0).unwrap();
        assert_eq!(xcramp_falsecolor(&mut ramp, 1), 0);
        let color = xcramp_mapfalsecolor(128);
        assert_eq!(
            ramp.ramp[128].to_ne_bytes(),
            [color.0 as u8, color.1 as u8, color.2 as u8, 0]
        );
        assert_eq!(xcramp_reverse(&mut ramp, 1), 0);
        assert_eq!(ramp.reverse, 1);
    }
}
