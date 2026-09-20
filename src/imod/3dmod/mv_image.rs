//! Translation of `IMOD/3dmod/mv_image.cpp` together with `mv_image.h`.
//!
//! The source is a compatibility-profile texture mapper.  Image cache access,
//! Qt controls, and the current GL context are deliberately passed at their
//! source boundaries; no alternate image renderer is introduced here.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::zoomdown::{
    SLICE_MODE_RGB, ZoomLines, ZoomOut, select_zoom_filter, zoom_with_filter,
};
use crate::imod::libimod::imat::{
    B3D_X, B3D_Y, B3D_Z, imod_mat_new, imod_mat_rot, imod_mat_transform,
};
use crate::imod::libimod::imodel::{Iclip_planes, Ipoint};
use crate::imod::three_dmod::imodv::ImodvApp;
use crate::imod::three_dmod::imodview::{
    ImodView, ivw_get_location, ivw_get_z_section, ivw_set_location,
    ivw_ushort_in_range_to_byte_map,
};

pub const IMODV_DRAW_CZ: i32 = 1;
pub const IMODV_DRAW_CY: i32 = 1 << 1;
pub const IMODV_DRAW_CX: i32 = 1 << 2;
pub const IMODV_CLIP_CZ: i32 = 1 << 3;
pub const IMODV_CLIP_CY: i32 = 1 << 4;
pub const IMODV_CLIP_CX: i32 = 1 << 5;
pub const IMODV_DRAW_CXYZ: i32 = IMODV_DRAW_CX | IMODV_DRAW_CY | IMODV_DRAW_CZ;
pub const MAX_SLICES: i32 = 1024;
pub const IMODV_MOVIE_END_STATE: i32 = 1;

/// `MovieTerminus` from the included `mv_movie.h` source boundary.
#[derive(Clone, Debug)]
pub struct MovieTerminus {
    pub img_xcenter: i32,
    pub img_ycenter: i32,
    pub img_zcenter: i32,
    pub img_slices: i32,
    pub img_transparency: i32,
}
impl Default for MovieTerminus {
    fn default() -> Self {
        Self {
            img_xcenter: 1,
            img_ycenter: 1,
            img_zcenter: 1,
            img_slices: 1,
            img_transparency: 0,
        }
    }
}
/// Fields of `MovieSegment` read or written by this translation unit.
#[derive(Clone, Debug, Default)]
pub struct MovieSegment {
    pub start: MovieTerminus,
    pub end: MovieTerminus,
    pub img_axis_flags: i32,
    pub img_white_level: i32,
    pub img_black_level: i32,
    pub img_false_color: i32,
    pub img_xsize: i32,
    pub img_ysize: i32,
    pub img_zsize: i32,
    pub img_clip_offset: i32,
}
/// `FastSegment` from `pyramidcache.h`, with the C pointer/stride ownership retained.
#[derive(Clone, Copy, Debug)]
pub struct FastSegment {
    pub xor_y: i32,
    pub length: i32,
    pub line: *const u8,
    pub stride: i32,
}

/// Value returned by the `iview`/pyramid-cache image access boundary.  It
/// preserves the three storage forms selected by `rgbStore` and `ushortStore`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImagePixel {
    Byte(u8),
    UShort(u16),
    Rgb([u8; 3]),
}

/// Direct commands issued by `imodvDrawTImage`, `setAlpha`, and texture setup.
pub trait MvImageGl {
    fn texture_capacity(&mut self, requested: i32) -> i32;
    fn create_texture_bgra(&mut self, width: i32, height: i32, data: &[u8]) -> u32;
    fn delete_texture(&mut self, texture: u32);
    fn upload_bgra(&mut self, width: i32, height: i32, data: &[u8]);
    fn texture_parameters(&mut self);
    fn enable_texture(&mut self, enabled: bool);
    fn set_alpha_blend(&mut self, alpha: f32, enabled: bool);
    fn draw_textured_quad(&mut self, points: [Ipoint; 4], clamp: (f32, f32), texel: f32);
    fn flush(&mut self);
}
/// Pixel/cache access supplied by `iview`, `pyramidcache`, and `cachefill`.
pub trait MvImageSource {
    fn dimensions(&self) -> (i32, i32, i32);
    fn location(&self) -> (i32, i32, i32);
    fn set_location(&mut self, x: i32, y: i32, z: i32);
    fn z_section(&mut self, z: i32) -> Option<&[u8]>;
    fn rgb_store(&self) -> bool {
        false
    }
    fn ushort_store(&self) -> bool {
        false
    }
    fn ushort_to_byte_map(&self) -> Option<&[u8]> {
        None
    }
    /// `ivwGetZSection`/`ivwSetupFastAccess` value lookup.  Implementors for
    /// 16-bit and RGB stores override this; byte storage retains `idata[j][i]`.
    fn pixel(&mut self, x: i32, y: i32, z: i32) -> Option<ImagePixel> {
        let (xs, ys, zs) = self.dimensions();
        if x < 0 || y < 0 || z < 0 || x >= xs || y >= ys || z >= zs {
            return None;
        }
        let section = self.z_section(z)?;
        section
            .get((y * xs + x) as usize)
            .copied()
            .map(ImagePixel::Byte)
    }
    /// `PyramidCache::pickBestCache`.
    fn pick_best_cache(&mut self, _zoom: f64, _up: f32, _down: f32) -> Option<(i32, i32)> {
        None
    }
    /// `scaledAreaSize`/`scaledRangeInZ`; identity is the source no-cache path.
    fn adjust_limits_for_tiles(
        &mut self,
        _cache: i32,
        x: &mut (i32, i32),
        y: &mut (i32, i32),
        z: &mut (i32, i32),
    ) -> (f32, f32, f32, i32) {
        let _ = (x, y, z);
        (0., 0., 0., 1)
    }
}

/// `imodvDrawImage`'s direct `ViewInfo` image/cache source.
///
/// The C++ unit obtains line pointers with `ivwGetZSection` for the Z plane
/// and with `ivwSetupFastAccess` for the X/Y planes.  The Rust `ImodView`
/// keeps both representations behind `ivw_get_z_section`; this adapter makes
/// that one source route available to all three plane loops without copying a
/// volume or introducing a second image cache.  `ushort_map` owns the result
/// of the source's `ivwUShortInRangeToByteMap` allocation for as long as the
/// texture operation uses it.
pub struct ImodViewImageSource<'a> {
    view: &'a mut ImodView,
    ushort_map: Option<Vec<u8>>,
}

impl<'a> ImodViewImageSource<'a> {
    pub fn new(view: &'a mut ImodView) -> Self {
        let ushort_map = (view.ushort_store != 0).then(|| ivw_ushort_in_range_to_byte_map(view));
        Self { view, ushort_map }
    }
}

impl MvImageSource for ImodViewImageSource<'_> {
    fn dimensions(&self) -> (i32, i32, i32) {
        (self.view.xsize, self.view.ysize, self.view.zsize)
    }

    fn location(&self) -> (i32, i32, i32) {
        let mut x = 0;
        let mut y = 0;
        let mut z = 0;
        ivw_get_location(self.view, &mut x, &mut y, &mut z);
        (x, y, z)
    }

    fn set_location(&mut self, x: i32, y: i32, z: i32) {
        ivw_set_location(self.view, x, y, z)
    }

    // `imodv_draw_image` overrides `pixel`, so a contiguous section is never
    // required here: `ivw_get_z_section` deliberately returns source line
    // pointers so that cached, flipped, and tiled image storage remains valid.
    fn z_section(&mut self, _z: i32) -> Option<&[u8]> {
        None
    }

    fn rgb_store(&self) -> bool {
        self.view.rgb_store != 0
    }

    fn ushort_store(&self) -> bool {
        self.view.ushort_store != 0
    }

    fn ushort_to_byte_map(&self) -> Option<&[u8]> {
        self.ushort_map.as_deref()
    }

    fn pixel(&mut self, x: i32, y: i32, z: i32) -> Option<ImagePixel> {
        let (xs, ys, zs) = self.dimensions();
        if x < 0 || y < 0 || z < 0 || x >= xs || y >= ys || z >= zs {
            return None;
        }
        let lines = ivw_get_z_section(self.view, z);
        if lines.is_null() {
            return None;
        }
        // `ivwGetZSection` has checked and built the `ysize` row table above.
        let line = unsafe { *lines.add(y as usize) };
        if line.is_null() {
            return None;
        }
        unsafe {
            if self.view.rgb_store != 0 {
                let pixel = line.add(3 * x as usize);
                return Some(ImagePixel::Rgb([*pixel, *pixel.add(1), *pixel.add(2)]));
            }
            if self.view.ushort_store != 0 {
                let pixel = line.add(2 * x as usize);
                return Some(ImagePixel::UShort(u16::from_ne_bytes([
                    *pixel,
                    *pixel.add(1),
                ])));
            }
            Some(ImagePixel::Byte(*line.add(x as usize)))
        }
    }
}

/// Native viewer and Qt calls made by `ImodvImage` event handlers.  These
/// retain the source ownership of the image cache and docking-dialog manager.
pub trait MvImageNativeBoundary {
    /// `imodCacheFill(Imodv->vi, -1)`.
    fn cache_fill(&mut self, section: i32);
    /// `ImodPrefs->getRoundedStyle()`.
    fn rounded_style(&self) -> bool;
    /// `DialogFrame::changeEvent`.
    fn dialog_change_event(&mut self);
    /// `ivwCheckAndSetMacMenu` for the model-view dialog.
    fn check_and_set_mac_menu(&mut self);
    /// `imodvDialogManager.remove(sTopWin)`.
    fn remove_dialog(&mut self);
    /// `QCloseEvent::accept`.
    fn accept_close_event(&mut self);
}

/// Static variables in `mv_image.cpp`, collected without changing their lifetime.
#[derive(Clone, Debug)]
pub struct MvImageState {
    pub tdata: Vec<u8>,
    pub tex_image_size: i32,
    pub zoom_buffer: Vec<u8>,
    pub zoom_buf_size: i32,
    pub zoom_scale: f64,
    pub zoom_filter: i32,
    pub tex_name: u32,
    pub cmap: [[u8; 256]; 3],
    pub black_level: i32,
    pub white_level: i32,
    pub falsecolor: i32,
    pub use_zoom_down: i32,
    pub clip_offset: i32,
    pub image_trans: i32,
    pub cmap_init: i32,
    pub cmap_z: i32,
    pub cmap_time: i32,
    pub num_slices: i32,
    pub xdraw_size: i32,
    pub ydraw_size: i32,
    pub zdraw_size: i32,
    pub last_ysize: i32,
    pub pyr_zoom_up_limit: f32,
    pub pyr_zoom_down_limit: f32,
    pub clip_planes: Iclip_planes,
    pub flags: i32,
    pub wall_load: f64,
    pub wall_draw: f64,
    pub wall_fill: f64,
}

impl MvImageState {
    /// `mvImageAnyDrawing`.
    pub fn mv_image_any_drawing(&self) -> bool {
        (self.flags & IMODV_DRAW_CXYZ) != 0
    }
    /// `mvImageAnyClipping`.
    pub fn mv_image_any_clipping(&self) -> bool {
        ((self.flags & IMODV_DRAW_CX) != 0 && (self.flags & IMODV_CLIP_CX) != 0)
            || ((self.flags & IMODV_DRAW_CY) != 0 && (self.flags & IMODV_CLIP_CY) != 0)
            || ((self.flags & IMODV_DRAW_CZ) != 0 && (self.flags & IMODV_CLIP_CZ) != 0)
    }
    /// `mvImageDrawingZplanes`.
    pub fn mv_image_drawing_zplanes(&self, a: &ImodvApp, closed: bool) -> bool {
        !closed && a.tex_map != 0 && (self.flags & IMODV_DRAW_CZ) != 0
    }
    /// `mvImageGetFlags`.
    pub fn mv_image_get_flags(&self) -> i32 {
        self.flags
    }
    /// `mvImageGetClipPlanes`.
    pub fn mv_image_get_clip_planes(&mut self) -> &mut Iclip_planes {
        &mut self.clip_planes
    }
    /// `mvImageSetAntialias`.
    pub fn mv_image_set_antialias(&mut self, value: i32) {
        self.use_zoom_down = value
    }
    /// `mvImageGetAntialias`.
    pub fn mv_image_get_antialias(&self) -> i32 {
        self.use_zoom_down
    }
    /// `mvImageGetThickness`.
    pub fn mv_image_get_thickness(&self) -> i32 {
        self.num_slices
    }
    /// `mvImageGetTransparency`.
    pub fn mv_image_get_transparency(&self) -> i32 {
        self.image_trans
    }

    /// `makeColorMap`; false color follows IMOD's BGR output ordering.
    pub fn make_color_map(&mut self, colormap_image: bool) {
        let (mut black, mut white) = if colormap_image {
            (0, 255)
        } else {
            (self.black_level, self.white_level)
        };
        let reverse = black > white;
        if reverse {
            std::mem::swap(&mut black, &mut white);
        }
        let ramp = (white - black).max(1) as f32;
        for i in 0usize..256 {
            let level = i as i32;
            let mut v = if level < black {
                0
            } else if level >= white {
                255
            } else {
                ((level - black) as f32 * 256. / ramp) as u8
            };
            if reverse && !colormap_image {
                v = 255 - v;
            }
            if self.falsecolor != 0 || colormap_image {
                let (r, g, b) = xcramp_mapfalsecolor(v);
                self.cmap[0][i] = b;
                self.cmap[1][i] = g;
                self.cmap[2][i] = r;
            } else {
                self.cmap[0][i] = v;
                self.cmap[1][i] = v;
                self.cmap[2][i] = v;
            }
        }
        self.cmap_init = 1;
    }
    /// `setSliceLimits`.
    pub fn set_slice_limits(
        &self,
        ciz: i32,
        zsize: i32,
        invert_z: bool,
        draw_trans: i32,
    ) -> (i32, i32, i32) {
        let mut st = (ciz - self.num_slices / 2).max(0);
        let mut nd = (st + self.num_slices - 1).min(zsize - 1);
        if ((nd > st || self.image_trans != 0) && draw_trans == 0)
            || (st == nd && self.image_trans == 0 && draw_trans != 0)
        {
            nd = st - 1
        }
        if invert_z { (nd, st, -1) } else { (st, nd, 1) }
    }
    /// `setAlpha`.
    pub fn set_alpha(&self, iz: i32, zst: i32, znd: i32, izdir: i32, gl: &mut dyn MvImageGl) {
        let n = (znd - zst) / izdir + 1;
        let m = (iz - zst) / izdir + 1;
        let b = 0.01 * (100 - self.image_trans) as f32 / n as f32;
        gl.set_alpha_blend(
            b / (1.0 - (n - m) as f32 * b),
            self.image_trans != 0 || n > 1,
        );
    }
    /// `mvImageSetThickTrans`.
    pub fn mv_image_set_thick_trans(&mut self, zsize: i32, slices: i32, trans: i32) {
        self.num_slices = slices.clamp(1, zsize.min(MAX_SLICES));
        self.image_trans = trans.clamp(0, 100)
    }
    /// `mvImageSetPlaneFlag`.
    pub fn mv_image_set_plane_flag(&mut self, a: &mut ImodvApp, on: bool, flag: i32) {
        if on {
            self.flags |= flag
        } else {
            self.flags &= !flag
        };
        a.tex_map = self.mv_image_any_drawing() as i32
    }
    /// `mvImageUpdate`.
    pub fn mv_image_update(&mut self, a: &mut ImodvApp) {
        if a.tex_map != 0 && !self.mv_image_any_drawing() {
            self.flags |= IMODV_DRAW_CZ
        } else if a.tex_map == 0 && self.mv_image_any_drawing() {
            self.flags &= !IMODV_DRAW_CXYZ
        }
    }
    /// `mvImageSubsetLimits`.
    pub fn mv_image_subset_limits(
        &self,
        a: &ImodvApp,
        source: &dyn MvImageSource,
        closed: bool,
    ) -> Option<(f64, f32, f32, i32, i32, i32, i32)> {
        if closed || a.tex_map == 0 || !self.mv_image_any_drawing() || self.xdraw_size <= 0 {
            return None;
        }
        let (x, y, _) = source.location();
        let (xs, ys, _) = source.dimensions();
        let (xb, xe) = set_coord_limits(x, xs, self.xdraw_size);
        let (yb, ye) = set_coord_limits(y, ys, self.ydraw_size);
        let zoom = 0.5 * (a.winx.min(a.winy) as f64)
            / unsafe { a.imod.as_ref() }?.view.first()?.rad as f64;
        Some((
            zoom,
            self.pyr_zoom_up_limit,
            self.pyr_zoom_down_limit,
            xb,
            yb,
            xe + 1 - xb,
            ye + 1 - yb,
        ))
    }
    /// `mvImageGetMovieState`.
    pub fn mv_image_get_movie_state(&self, a: &ImodvApp, segment: &mut MovieSegment) {
        segment.img_axis_flags = if a.tex_map == 0 { 0 } else { self.flags };
        if a.tex_map != 0 {
            segment.img_white_level = self.white_level;
            segment.img_black_level = self.black_level;
            segment.img_false_color = self.falsecolor;
            segment.img_clip_offset = self.clip_offset;
            segment.img_xsize = self.xdraw_size;
            segment.img_ysize = self.ydraw_size;
            segment.img_zsize = self.zdraw_size
        }
    }
    /// `imodvDrawTImage`.
    pub fn imodv_draw_timage(
        &mut self,
        points: [Ipoint; 4],
        clamp: Ipoint,
        data: &[u8],
        width: i32,
        height: i32,
        gl: &mut dyn MvImageGl,
    ) {
        let mut upload = data.to_vec();
        let mut upload_width = width;
        let mut upload_height = height;
        let mut upload_clamp = (clamp.x, clamp.y);
        // The `sZoomBuffer` / `zoomWithFilter` path.  `zoomdown.rs` is the direct
        // translation of b3dgfx's filtering implementation, so preserve its
        // source-selected filter instead of replacing it with a Rust resampler.
        if !self.zoom_buffer.is_empty() && self.zoom_scale < 1.0 {
            let nx =
                ((width as f64 * self.zoom_scale).floor() as i32).clamp(1, self.tex_image_size);
            let ny =
                ((height as f64 * self.zoom_scale).floor() as i32).clamp(1, self.tex_image_size);
            let lines: Vec<&[u8]> = (0..height)
                .map(|row| &upload[(row * width * 4) as usize..])
                .collect();
            let mut reduced = vec![0u8; (nx * ny * 4) as usize];
            let dtype = -(SLICE_MODE_RGB + if self.falsecolor != 0 { 2 } else { 3 });
            let err = zoom_with_filter(
                ZoomLines::Byte(&lines),
                width,
                height,
                0.,
                0.,
                nx,
                ny,
                nx,
                0,
                dtype,
                &mut ZoomOut::Byte(&mut reduced),
                None,
                None,
            );
            if err == 0 {
                upload = reduced;
                upload_width = nx;
                upload_height = ny;
                upload_clamp = (
                    (nx - 1) as f32 / self.tex_image_size as f32,
                    (ny - 1) as f32 / self.tex_image_size as f32,
                );
            }
        }
        gl.upload_bgra(upload_width, upload_height, &upload);
        gl.draw_textured_quad(points, upload_clamp, 1. / self.tex_image_size as f32);
        gl.flush()
    }
    /// `initTexMapping`.
    pub fn init_tex_mapping(&mut self, a: &ImodvApp, gl: &mut dyn MvImageGl) -> i32 {
        let mut step = if a.gl_ext_flags != 0 { 528 } else { 512 };
        let mut usable = 0;
        while step >= 64 {
            usable = gl.texture_capacity(step);
            if usable > 0 {
                break;
            }
            step /= 2
        }
        if usable <= 0 {
            return 1;
        }
        self.tdata = vec![0; (4 * step * step) as usize];
        self.tex_image_size = step;
        self.tex_name = gl.create_texture_bgra(step, step, &self.tdata);
        gl.texture_parameters();
        0
    }
    /// `mvImageCleanup`.
    pub fn mv_image_cleanup(&mut self, gl: &mut dyn MvImageGl) {
        self.tdata.clear();
        if self.tex_image_size != 0 {
            gl.delete_texture(self.tex_name)
        }
        self.tex_image_size = 0;
        self.tex_name = 0
    }
    /// Clipping section in `imodvDrawImage`.  The resulting `IclipPlanes` is
    /// consumed unchanged by `mv_ogl::clip_obj` when it draws model primitives.
    pub fn setup_clip_planes(&mut self, a: &ImodvApp, source: &dyn MvImageSource, draw_trans: i32) {
        self.clip_planes.count = 0;
        if draw_trans != 0 || !self.mv_image_any_clipping() {
            return;
        }
        let Some(model) = (unsafe { a.imod.as_ref() }) else {
            return;
        };
        let Some(view) = model.view.first() else {
            return;
        };
        let Some(mut mat) = imod_mat_new(3) else {
            return;
        };
        imod_mat_rot(&mut mat, view.rot.z as f64, B3D_Z);
        imod_mat_rot(&mut mat, view.rot.y as f64, B3D_Y);
        imod_mat_rot(&mut mat, view.rot.x as f64, B3D_X);
        let (ix, iy, iz) = source.location();
        let (xs, ys, zs) = source.dimensions();
        let mut count = 0usize;
        for axis in 0..3usize {
            if self.flags & (1 << axis) == 0 || self.flags & (1 << (axis + 3)) == 0 {
                continue;
            }
            let normal = Ipoint {
                x: (axis == 2) as i32 as f32,
                y: (axis == 1) as i32 as f32,
                z: (axis == 0) as i32 as f32,
            };
            let mut transformed = Ipoint::default();
            imod_mat_transform(&mat, &normal, &mut transformed);
            let flip = (self.clip_offset < 0 && transformed.z < 0.)
                || (self.clip_offset >= 0 && transformed.z > 0.);
            let n = if flip {
                Ipoint {
                    x: -normal.x,
                    y: -normal.y,
                    z: -normal.z,
                }
            } else {
                normal
            };
            self.clip_planes.normal[count] = n;
            let gap = self.clip_offset.max(0) as f32 + 0.5;
            self.clip_planes.point[count] = Ipoint {
                x: if axis == 2 {
                    -(ix as f32)
                } else {
                    -(xs as f32) / 2.
                } + n.x * gap,
                y: if axis == 1 {
                    -(iy as f32)
                } else {
                    -(ys as f32) / 2.
                } + n.y * gap,
                z: if axis == 0 {
                    -(iz as f32)
                } else {
                    -(zs as f32) / 2.
                } + n.z * gap,
            };
            count += 1;
        }
        if self.clip_offset >= 0 {
            for index in 0..count {
                if count + index >= self.clip_planes.normal.len() {
                    break;
                }
                let n = self.clip_planes.normal[index];
                let p = self.clip_planes.point[index];
                self.clip_planes.normal[count + index] = Ipoint {
                    x: -n.x,
                    y: -n.y,
                    z: -n.z,
                };
                self.clip_planes.point[count + index] = Ipoint {
                    x: p.x - n.x * (self.clip_offset + 1) as f32,
                    y: p.y - n.y * (self.clip_offset + 1) as f32,
                    z: p.z - n.z * (self.clip_offset + 1) as f32,
                };
            }
            count *= 2;
        }
        self.clip_planes.count = count.min(self.clip_planes.normal.len()) as u8;
    }
    /// Source `FILLDATA` macro.
    pub fn fill_pixel(&mut self, width: i32, x: i32, y: i32, value: u8) {
        let p = (4 * (width * y + x)) as usize;
        if p + 3 >= self.tdata.len() {
            return;
        }
        self.tdata[p] = self.cmap[0][value as usize];
        self.tdata[p + 1] = self.cmap[1][value as usize];
        self.tdata[p + 2] = self.cmap[2][value as usize];
        self.tdata[p + 3] = 255
    }

    /// `mvImageTogglePlane`.
    pub fn mv_image_toggle_plane(&mut self, a: &mut ImodvApp, flag: i32) {
        self.mv_image_set_plane_flag(a, self.flags & flag == 0, flag)
    }
    /// `mvImageEditDialog`; native dialog construction remains `mv_window`/Qt boundary.
    pub fn mv_image_edit_dialog(
        &mut self,
        a: &mut ImodvApp,
        dialog: &mut Option<ImodvImage>,
        open: bool,
    ) {
        if !open {
            *dialog = None;
            return;
        }
        if dialog.is_none() {
            *dialog = Some(ImodvImage::default());
            self.make_color_map(false)
        }
        self.mv_image_update(a)
    }
    /// `mvImageSetMovieDrawState`.
    pub fn mv_image_set_movie_draw_state(
        &mut self,
        a: &mut ImodvApp,
        segment: &MovieSegment,
        source: &dyn MvImageSource,
    ) -> i32 {
        if segment.img_axis_flags & IMODV_DRAW_CXYZ == 0 {
            a.tex_map = 0;
            if self.mv_image_any_drawing() {
                self.mv_image_update(a)
            };
            return 1;
        }
        self.flags = segment.img_axis_flags;
        a.tex_map = 1;
        self.white_level = segment.img_white_level;
        self.black_level = segment.img_black_level;
        self.falsecolor = segment.img_false_color;
        self.clip_offset = segment.img_clip_offset;
        let (x, y, z) = source.dimensions();
        self.xdraw_size = segment.img_xsize.min(x);
        self.ydraw_size = segment.img_ysize.min(y);
        self.zdraw_size = segment.img_zsize.min(z);
        self.make_color_map(false);
        0
    }
    /// `fillPatchFromTiles`.
    pub fn fill_patch_from_tiles(
        &mut self,
        segments: &[FastSegment],
        starts: &[i32],
        ushort: bool,
        bmap: Option<&[u8]>,
        fill_x: i32,
        istart: i32,
        iend: i32,
        jstart: i32,
        jend: i32,
    ) {
        for j in jstart.max(0)..jend.min(starts.len().saturating_sub(1) as i32) {
            let mut next = istart;
            for si in starts[j as usize]..starts[j as usize + 1] {
                let s = segments[si as usize];
                if s.xor_y >= iend {
                    break;
                }
                if s.xor_y + s.length <= istart {
                    continue;
                }
                for i in next..s.xor_y.min(iend) {
                    self.fill_pixel(fill_x, i - istart, j - jstart, 0)
                }
                let ss = s.xor_y.max(istart);
                let se = (s.xor_y + s.length).min(iend);
                for i in ss..se {
                    let off = ((i - s.xor_y) * s.stride * (if ushort { 2 } else { 1 })) as usize;
                    let raw = unsafe { *s.line.add(off) };
                    let value = if ushort {
                        bmap.and_then(|m| {
                            m.get(unsafe { *(s.line.add(off) as *const u16) } as usize)
                        })
                        .copied()
                        .unwrap_or(0)
                    } else {
                        raw
                    };
                    self.fill_pixel(fill_x, i - istart, j - jstart, value)
                }
                next = se
            }
            for i in next..iend {
                self.fill_pixel(fill_x, i - istart, j - jstart, 0)
            }
        }
    }
    /// `endTexMapping`.
    pub fn end_tex_mapping(&mut self, a: &mut ImodvApp, gl: &mut dyn MvImageGl) {
        self.flags &= !IMODV_DRAW_CXYZ;
        a.tex_map = 0;
        self.mv_image_cleanup(gl)
    }
    /// Source `FILLRGB` macro and `ivwUShortInRangeToByteMap` indexing.
    pub fn fill_image_pixel(
        &mut self,
        width: i32,
        x: i32,
        y: i32,
        pixel: Option<ImagePixel>,
        ushort_map: Option<&[u8]>,
    ) {
        match pixel {
            Some(ImagePixel::Byte(value)) => self.fill_pixel(width, x, y, value),
            Some(ImagePixel::UShort(value)) => self.fill_pixel(
                width,
                x,
                y,
                ushort_map
                    .and_then(|map| map.get(value as usize))
                    .copied()
                    .unwrap_or(0),
            ),
            Some(ImagePixel::Rgb([blue, green, red])) => {
                let p = (4 * (width * y + x)) as usize;
                if p + 3 < self.tdata.len() {
                    self.tdata[p] = self.cmap[0][red as usize];
                    self.tdata[p + 1] = self.cmap[1][green as usize];
                    self.tdata[p + 2] = self.cmap[2][blue as usize];
                    self.tdata[p + 3] = 255;
                }
            }
            None => self.fill_pixel(width, x, y, 0),
        }
    }

    /// `mvImageSetMovieEndState`.
    pub fn mv_image_set_movie_end_state(
        &mut self,
        a: &mut ImodvApp,
        source: &mut dyn MvImageSource,
        start_end: i32,
        segment: &MovieSegment,
    ) {
        if self.mv_image_set_movie_draw_state(a, segment, source) != 0 {
            return;
        }
        let term = if start_end == IMODV_MOVIE_END_STATE {
            &segment.end
        } else {
            &segment.start
        };
        self.image_trans = term.img_transparency;
        source.set_location(
            term.img_xcenter - 1,
            term.img_ycenter - 1,
            term.img_zcenter - 1,
        );
        self.num_slices = term.img_slices.min(source.dimensions().2.min(MAX_SLICES));
    }
    /// `imodvDrawImage`: Z, X, and Y texture-plane loops.  Image/cache storage is
    /// read only through the direct `iview`/`PyramidCache` boundary above.
    pub fn imodv_draw_image(
        &mut self,
        a: &mut ImodvApp,
        source: &mut dyn MvImageSource,
        draw_trans: i32,
        gl: &mut dyn MvImageGl,
    ) {
        if !self.mv_image_any_drawing() {
            self.mv_image_cleanup(gl);
            return;
        }
        let (xs, ys, zs) = source.dimensions();
        if self.xdraw_size < 0 {
            self.xdraw_size = xs;
            self.ydraw_size = ys;
            self.zdraw_size = zs
        }
        if self.tex_image_size == 0 && self.init_tex_mapping(a, gl) != 0 {
            return;
        }
        // `pickBestCache` and the source antialiased-reduction selection.  A
        // selected cache remains responsible for its scaled sample coordinates
        // through the `MvImageSource` implementation.
        let rad = unsafe { a.imod.as_ref() }
            .and_then(|model| model.view.first())
            .map(|view| view.rad as f64)
            .unwrap_or(1.0)
            .max(f64::MIN_POSITIVE);
        let zoom = 0.5 * a.winx.min(a.winy) as f64 / rad;
        let cache_selection =
            source.pick_best_cache(zoom, self.pyr_zoom_up_limit, self.pyr_zoom_down_limit);
        let cache_scale = cache_selection.map(|(_, scale)| scale).unwrap_or(1).max(1);
        self.zoom_scale = zoom * cache_scale as f64;
        let max_size = self.xdraw_size.max(self.ydraw_size).max(self.zdraw_size) / cache_scale;
        self.zoom_buffer.clear();
        if self.use_zoom_down != 0
            && self.zoom_scale < 0.75
            && max_size as f64 * self.zoom_scale > 20.0
        {
            self.zoom_buf_size =
                (((self.tex_image_size as f64 / self.zoom_scale) as i32) / 2 * 2).min(max_size);
            if self.zoom_buf_size > 0 {
                let filters = [5, 4, 1, 0];
                let mut width = 0;
                for filter in filters {
                    if unsafe { select_zoom_filter(filter, self.zoom_scale, &mut width) } == 0 {
                        self.zoom_filter = filter;
                        break;
                    }
                }
                self.zoom_buffer
                    .resize((4 * self.zoom_buf_size * self.zoom_buf_size) as usize, 0);
            }
        }
        self.make_color_map(false);
        let (cx, cy, cz) = source.location();
        let mut xlimits = set_coord_limits(cx, xs, self.xdraw_size);
        let mut ylimits = set_coord_limits(cy, ys, self.ydraw_size);
        let (zfirst, zlast, zdir) = self.set_slice_limits(cz, zs, a.invert_z != 0, draw_trans);
        let mut zlimits = (zfirst, zlast);
        let (_tile_x_offset, _tile_y_offset, tile_z_offset, tile_z_scale) = adjust_limits_for_tiles(
            source,
            cache_selection.map(|(cache, _)| cache),
            zdir,
            &mut xlimits,
            &mut ylimits,
            &mut zlimits,
        );
        let (xst, xnd) = xlimits;
        let (yst, ynd) = ylimits;
        let (zst, znd) = zlimits;
        self.setup_clip_planes(a, source, draw_trans);
        gl.enable_texture(true);
        let mut z = zst;
        while zdir * (znd - z) >= 0 {
            self.set_alpha(z, zst, znd, zdir, gl);
            let width = xnd - xst + 2;
            let height = ynd - yst + 2;
            let needed = (4 * width * height) as usize;
            if self.tdata.len() < needed {
                self.tdata.resize(needed, 0)
            }
            for yy in 0..height {
                for xx in 0..width {
                    let ix = (xst - 1 + xx).clamp(0, xs - 1);
                    let iy = (yst - 1 + yy).clamp(0, ys - 1);
                    let pixel = source.pixel(ix, iy, z);
                    let map = source.ushort_to_byte_map();
                    self.fill_image_pixel(width, xx, yy, pixel, map)
                }
            }
            let p = [
                Ipoint {
                    x: xst as f32,
                    y: yst as f32,
                    z: z as f32 * tile_z_scale as f32 + tile_z_offset,
                },
                Ipoint {
                    x: xnd as f32,
                    y: yst as f32,
                    z: z as f32 * tile_z_scale as f32 + tile_z_offset,
                },
                Ipoint {
                    x: xnd as f32,
                    y: ynd as f32,
                    z: z as f32 * tile_z_scale as f32 + tile_z_offset,
                },
                Ipoint {
                    x: xst as f32,
                    y: ynd as f32,
                    z: z as f32 * tile_z_scale as f32 + tile_z_offset,
                },
            ];
            let data = self.tdata[..needed].to_vec();
            self.imodv_draw_timage(
                p,
                Ipoint {
                    x: (width - 1) as f32 / width as f32,
                    y: (height - 1) as f32 / height as f32,
                    z: 0.,
                },
                &data,
                width,
                height,
                gl,
            );
            z += zdir;
        }
        // Draw Current X image.  This is the `IMODV_DRAW_CX` loop in the source:
        // texture U is Y and texture V is Z.
        if self.flags & IMODV_DRAW_CX != 0 && !(a.stereo != 0 && a.image_stereo != 0) {
            let (xfirst, xlast, xdir) = self.set_slice_limits(cx, xs, a.invert_z != 0, draw_trans);
            let (yfirst, ylast) = set_coord_limits(cy, ys, self.ydraw_size);
            let (zfirst, zlast) = set_coord_limits(cz, zs, self.zdraw_size);
            let mut xplane = xfirst;
            while xdir * (xlast - xplane) >= 0 {
                self.set_alpha(xplane, xfirst, xlast, xdir, gl);
                let mut zpatch = zfirst;
                while zpatch < zlast {
                    let zend = (zpatch + self.tex_image_size - 2).min(zlast);
                    let mut ypatch = yfirst;
                    while ypatch < ylast {
                        let yend = (ypatch + self.tex_image_size - 2).min(ylast);
                        let width = yend - ypatch + 2;
                        let height = zend - zpatch + 2;
                        let needed = (4 * width * height) as usize;
                        if self.tdata.len() < needed {
                            self.tdata.resize(needed, 0);
                        }
                        for vz in 0..height {
                            for uy in 0..width {
                                let pixel = source.pixel(
                                    (xplane).clamp(0, xs - 1),
                                    (ypatch - 1 + uy).clamp(0, ys - 1),
                                    (zpatch - 1 + vz).clamp(0, zs - 1),
                                );
                                let map = source.ushort_to_byte_map();
                                self.fill_image_pixel(width, uy, vz, pixel, map);
                            }
                        }
                        let data = self.tdata[..needed].to_vec();
                        self.imodv_draw_timage(
                            [
                                Ipoint {
                                    x: xplane as f32,
                                    y: ypatch as f32,
                                    z: zpatch as f32,
                                },
                                Ipoint {
                                    x: xplane as f32,
                                    y: yend as f32,
                                    z: zpatch as f32,
                                },
                                Ipoint {
                                    x: xplane as f32,
                                    y: yend as f32,
                                    z: zend as f32,
                                },
                                Ipoint {
                                    x: xplane as f32,
                                    y: ypatch as f32,
                                    z: zend as f32,
                                },
                            ],
                            Ipoint {
                                x: (width - 1) as f32 / width as f32,
                                y: (height - 1) as f32 / height as f32,
                                z: 0.,
                            },
                            &data,
                            width,
                            height,
                            gl,
                        );
                        ypatch += self.tex_image_size - 2;
                    }
                    zpatch += self.tex_image_size - 2;
                }
                xplane += xdir;
            }
        }

        // Draw Current Y image.  This is the source `IMODV_DRAW_CY` loop: U is X
        // and V is Z, with fast-access storage represented by `MvImageSource`.
        if self.flags & IMODV_DRAW_CY != 0 && !(a.stereo != 0 && a.image_stereo != 0) {
            let (yfirst, ylast, ydir) = self.set_slice_limits(cy, ys, a.invert_z != 0, draw_trans);
            let (xfirst, xlast) = set_coord_limits(cx, xs, self.xdraw_size);
            let (zfirst, zlast) = set_coord_limits(cz, zs, self.zdraw_size);
            let mut yplane = yfirst;
            while ydir * (ylast - yplane) >= 0 {
                self.set_alpha(yplane, yfirst, ylast, ydir, gl);
                let mut zpatch = zfirst;
                while zpatch < zlast {
                    let zend = (zpatch + self.tex_image_size - 2).min(zlast);
                    let mut xpatch = xfirst;
                    while xpatch < xlast {
                        let xend = (xpatch + self.tex_image_size - 2).min(xlast);
                        let width = xend - xpatch + 2;
                        let height = zend - zpatch + 2;
                        let needed = (4 * width * height) as usize;
                        if self.tdata.len() < needed {
                            self.tdata.resize(needed, 0);
                        }
                        for vz in 0..height {
                            for ux in 0..width {
                                let pixel = source.pixel(
                                    (xpatch - 1 + ux).clamp(0, xs - 1),
                                    yplane.clamp(0, ys - 1),
                                    (zpatch - 1 + vz).clamp(0, zs - 1),
                                );
                                let map = source.ushort_to_byte_map();
                                self.fill_image_pixel(width, ux, vz, pixel, map);
                            }
                        }
                        let data = self.tdata[..needed].to_vec();
                        self.imodv_draw_timage(
                            [
                                Ipoint {
                                    x: xpatch as f32,
                                    y: yplane as f32,
                                    z: zpatch as f32,
                                },
                                Ipoint {
                                    x: xend as f32,
                                    y: yplane as f32,
                                    z: zpatch as f32,
                                },
                                Ipoint {
                                    x: xend as f32,
                                    y: yplane as f32,
                                    z: zend as f32,
                                },
                                Ipoint {
                                    x: xpatch as f32,
                                    y: yplane as f32,
                                    z: zend as f32,
                                },
                            ],
                            Ipoint {
                                x: (width - 1) as f32 / width as f32,
                                y: (height - 1) as f32 / height as f32,
                                z: 0.,
                            },
                            &data,
                            width,
                            height,
                            gl,
                        );
                        xpatch += self.tex_image_size - 2;
                    }
                    zpatch += self.tex_image_size - 2;
                }
                yplane += ydir;
            }
        }
        gl.set_alpha_blend(1., false);
        gl.enable_texture(false)
    }
}

impl Default for MvImageState {
    fn default() -> Self {
        Self {
            tdata: Vec::new(),
            tex_image_size: 0,
            zoom_buffer: Vec::new(),
            zoom_buf_size: 0,
            zoom_scale: 0.,
            zoom_filter: 0,
            tex_name: 0,
            cmap: [[0; 256]; 3],
            black_level: 0,
            white_level: 255,
            falsecolor: 0,
            use_zoom_down: 1,
            clip_offset: 0,
            image_trans: 0,
            cmap_init: 0,
            cmap_z: 0,
            cmap_time: 0,
            num_slices: 1,
            xdraw_size: -1,
            ydraw_size: -1,
            zdraw_size: -1,
            last_ysize: -1,
            pyr_zoom_up_limit: 1.05,
            pyr_zoom_down_limit: 0.5,
            clip_planes: Iclip_planes::default(),
            flags: 0,
            wall_load: 0.,
            wall_draw: 0.,
            wall_fill: 0.,
        }
    }
}

/// Qt-independent state of `ImodvImage` / the paired dialog frame.
#[derive(Clone, Debug, Default)]
pub struct ImodvImage {
    pub ctrl_pressed: bool,
    pub rounded_style: bool,
    /// Mirrors the source clearing `sDia`/`sTopWin` after unregistering the
    /// docking dialog.  The actual dialog pointer remains native-owned.
    pub closed: bool,
    pub view_x: bool,
    pub view_y: bool,
    pub view_z: bool,
    pub clip_x: bool,
    pub clip_y: bool,
    pub clip_z: bool,
    pub apply_clip: bool,
}

/// `xcramp_mapfalsecolor` source boundary equation (rainbow ramp).
pub fn xcramp_mapfalsecolor(value: u8) -> (u8, u8, u8) {
    let x = value as f32 / 255.;
    let r = (1.5 - (4. * x - 3.).abs()).clamp(0., 1.);
    let g = (1.5 - (4. * x - 2.).abs()).clamp(0., 1.);
    let b = (1.5 - (4. * x - 1.).abs()).clamp(0., 1.);
    ((r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8)
}

/// `setCoordLimits`.
pub fn set_coord_limits(cur: i32, max_size: i32, draw_size: i32) -> (i32, i32) {
    let mut st = (cur - draw_size / 2).max(1);
    let end = (st + draw_size).min(max_size - 1);
    st = st.max(1).max(end - draw_size);
    (st, end)
}

/// `adjustLimitsForTiles`.  The cache unit owns the coordinate conversion;
/// this source unit preserves its call point and returned display offsets.
pub fn adjust_limits_for_tiles(
    source: &mut dyn MvImageSource,
    cache: Option<i32>,
    main_dir: i32,
    x: &mut (i32, i32),
    y: &mut (i32, i32),
    z: &mut (i32, i32),
) -> (f32, f32, f32, i32) {
    if let Some(cache) = cache {
        if main_dir <= 0 || (x.0 <= x.1 && y.0 <= y.1 && z.0 <= z.1) {
            return source.adjust_limits_for_tiles(cache, x, y, z);
        }
    }
    (0., 0., 0., 1)
}

impl ImodvImage {
    /// `ImodvImage()` source constructor.
    pub fn new() -> Self {
        Self::default()
    }
    pub fn update_coords(&mut self, state: &mut MvImageState, source: &dyn MvImageSource) {
        let (_, y, z) = source.dimensions();
        if state.last_ysize != y {
            std::mem::swap(&mut state.ydraw_size, &mut state.zdraw_size);
            state.last_ysize = y;
            state.num_slices = state.num_slices.min(z.min(MAX_SLICES))
        }
    }
    pub fn manage_clip_enables(&mut self, state: &MvImageState) {
        self.clip_x &= state.flags & IMODV_DRAW_CX != 0;
        self.clip_y &= state.flags & IMODV_DRAW_CY != 0;
        self.clip_z &= state.flags & IMODV_DRAW_CZ != 0
    }
    pub fn set_view_clip_check_boxes(&mut self, state: &MvImageState, a: &ImodvApp) {
        self.view_x = state.flags & IMODV_DRAW_CX != 0 && !(a.stereo != 0 && a.image_stereo != 0);
        self.view_y = state.flags & IMODV_DRAW_CY != 0 && !(a.stereo != 0 && a.image_stereo != 0);
        self.view_z = state.flags & IMODV_DRAW_CZ != 0;
        self.clip_x = state.flags & IMODV_CLIP_CX != 0;
        self.clip_y = state.flags & IMODV_CLIP_CY != 0;
        self.clip_z = state.flags & IMODV_CLIP_CZ != 0;
        self.manage_clip_enables(state)
    }
    pub fn view_x_toggled(&mut self, s: &mut MvImageState, a: &mut ImodvApp, on: bool) {
        s.mv_image_set_plane_flag(a, on, IMODV_DRAW_CX)
    }
    pub fn view_y_toggled(&mut self, s: &mut MvImageState, a: &mut ImodvApp, on: bool) {
        s.mv_image_set_plane_flag(a, on, IMODV_DRAW_CY)
    }
    pub fn view_z_toggled(&mut self, s: &mut MvImageState, a: &mut ImodvApp, on: bool) {
        s.mv_image_set_plane_flag(a, on, IMODV_DRAW_CZ)
    }
    pub fn clip_x_toggled(&mut self, s: &mut MvImageState, a: &mut ImodvApp, on: bool) {
        s.mv_image_set_plane_flag(a, on, IMODV_CLIP_CX)
    }
    pub fn clip_y_toggled(&mut self, s: &mut MvImageState, a: &mut ImodvApp, on: bool) {
        s.mv_image_set_plane_flag(a, on, IMODV_CLIP_CY)
    }
    pub fn clip_z_toggled(&mut self, s: &mut MvImageState, a: &mut ImodvApp, on: bool) {
        s.mv_image_set_plane_flag(a, on, IMODV_CLIP_CZ)
    }
    pub fn clip_offset_changed(&mut self, s: &mut MvImageState, value: i32) {
        s.clip_offset = value
    }
    pub fn false_toggled(&mut self, s: &mut MvImageState, on: bool) {
        s.falsecolor = on as i32;
        s.make_color_map(false)
    }
    pub fn use_zoom_toggled(&mut self, s: &mut MvImageState, on: bool) {
        s.use_zoom_down = on as i32
    }
    pub fn apply_clip_toggled(&mut self, on: bool) {
        self.apply_clip = on
    }
    pub fn slider_moved(&mut self, s: &mut MvImageState, which: i32, value: i32, _dragging: bool) {
        match which {
            3 => s.xdraw_size = value,
            4 => s.ydraw_size = value,
            5 => s.zdraw_size = value,
            6 => s.num_slices = value,
            7 => s.image_trans = value,
            8 => {
                s.black_level = value;
                s.make_color_map(false)
            }
            9 => {
                s.white_level = value;
                s.make_color_map(false)
            }
            _ => {}
        }
    }
    pub fn copy_bw_clicked(&mut self, s: &mut MvImageState, black: i32, white: i32) {
        s.black_level = black;
        s.white_level = white;
        s.make_color_map(false)
    }
    /// `ImodvImage::buttonPressed`.
    pub fn button_pressed(&mut self, native: &mut dyn MvImageNativeBoundary) {
        native.cache_fill(-1)
    }
    /// `ImodvImage::topChangeEvent`.
    pub fn top_change_event(&mut self, native: &mut dyn MvImageNativeBoundary) {
        self.rounded_style = native.rounded_style();
        native.dialog_change_event();
        native.check_and_set_mac_menu();
    }
    /// `ImodvImage::topCloseEvent`.
    pub fn top_close_event(&mut self, native: &mut dyn MvImageNativeBoundary) {
        native.remove_dialog();
        self.closed = true;
        native.accept_close_event();
    }
    pub fn key_press_event(&mut self, control: bool) {
        self.ctrl_pressed = control
    }
    pub fn key_release_event(&mut self) {
        self.ctrl_pressed = false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libcfshr::islice::slice_create;
    use crate::imod::libiimod::mrcfiles::{LoadInfo, MRC_MODE_BYTE, MRC_MODE_RGB, MRC_MODE_USHORT};
    use crate::imod::three_dmod::imodview::IvwSlice;
    struct Src {
        data: Vec<u8>,
    }
    impl MvImageSource for Src {
        fn dimensions(&self) -> (i32, i32, i32) {
            (4, 4, 4)
        }
        fn location(&self) -> (i32, i32, i32) {
            (2, 2, 2)
        }
        fn set_location(&mut self, _: i32, _: i32, _: i32) {}
        fn z_section(&mut self, z: i32) -> Option<&[u8]> {
            self.data.get((z * 16) as usize..((z + 1) * 16) as usize)
        }
    }
    struct Gl {
        quads: usize,
        texel: f32,
    }
    impl MvImageGl for Gl {
        fn texture_capacity(&mut self, n: i32) -> i32 {
            n
        }
        fn create_texture_bgra(&mut self, _: i32, _: i32, _: &[u8]) -> u32 {
            1
        }
        fn delete_texture(&mut self, _: u32) {}
        fn upload_bgra(&mut self, _: i32, _: i32, _: &[u8]) {}
        fn texture_parameters(&mut self) {}
        fn enable_texture(&mut self, _: bool) {}
        fn set_alpha_blend(&mut self, _: f32, _: bool) {}
        fn draw_textured_quad(&mut self, _: [Ipoint; 4], _: (f32, f32), texel: f32) {
            self.quads += 1;
            self.texel = texel;
        }
        fn flush(&mut self) {}
    }

    fn cached_view(mode: i32, bytes: Vec<u8>) -> (ImodView, Box<LoadInfo>) {
        let mut load_info = Box::new(LoadInfo {
            axis: 3,
            ..Default::default()
        });
        let mut section = slice_create(2, 1, mode).expect("test section");
        section.data.bytes_mut().copy_from_slice(&bytes);
        let view = ImodView {
            fp: Some(ImodFile::Token(1)),
            li: &mut *load_info,
            xsize: 2,
            ysize: 1,
            zsize: 1,
            vm_size: 1,
            vm_tdim: 1,
            cache_index: vec![0],
            vm_cache: vec![IvwSlice {
                cz: 0,
                ct: 0,
                used: 0,
                sec: section,
            }],
            ..Default::default()
        };
        (view, load_info)
    }

    #[test]
    fn imod_view_source_reads_byte_ushort_and_bgr_cache_pixels() {
        let (mut bytes, _byte_load) = cached_view(MRC_MODE_BYTE, vec![7, 9]);
        let mut source = ImodViewImageSource::new(&mut bytes);
        assert_eq!(source.pixel(0, 0, 0), Some(ImagePixel::Byte(7)));
        assert_eq!(source.pixel(1, 0, 0), Some(ImagePixel::Byte(9)));
        assert_eq!(source.pixel(2, 0, 0), None);

        let (mut ushorts, _ushort_load) = cached_view(
            MRC_MODE_USHORT,
            [500u16.to_ne_bytes(), 1000u16.to_ne_bytes()].concat(),
        );
        ushorts.ushort_store = 1;
        ushorts.range_low = 0;
        ushorts.range_high = 65535;
        let mut source = ImodViewImageSource::new(&mut ushorts);
        assert_eq!(source.pixel(0, 0, 0), Some(ImagePixel::UShort(500)));
        assert_eq!(source.pixel(1, 0, 0), Some(ImagePixel::UShort(1000)));
        assert_eq!(source.ushort_to_byte_map().unwrap()[500], 2);

        let (mut rgb, _rgb_load) = cached_view(MRC_MODE_RGB, vec![3, 2, 1, 6, 5, 4]);
        rgb.rgb_store = 1;
        let mut source = ImodViewImageSource::new(&mut rgb);
        assert_eq!(source.pixel(0, 0, 0), Some(ImagePixel::Rgb([3, 2, 1])));
        assert_eq!(source.pixel(1, 0, 0), Some(ImagePixel::Rgb([6, 5, 4])));
    }
    #[derive(Default)]
    struct EventBoundary {
        calls: Vec<&'static str>,
        cache_section: Option<i32>,
        rounded: bool,
    }
    impl MvImageNativeBoundary for EventBoundary {
        fn cache_fill(&mut self, section: i32) {
            self.calls.push("cache_fill");
            self.cache_section = Some(section);
        }
        fn rounded_style(&self) -> bool {
            self.rounded
        }
        fn dialog_change_event(&mut self) {
            self.calls.push("base_change");
        }
        fn check_and_set_mac_menu(&mut self) {
            self.calls.push("mac_menu");
        }
        fn remove_dialog(&mut self) {
            self.calls.push("remove_dialog");
        }
        fn accept_close_event(&mut self) {
            self.calls.push("accept_close");
        }
    }
    #[test]
    fn limits_match_source() {
        assert_eq!(set_coord_limits(0, 100, 20), (1, 21));
        let s = MvImageState::default();
        assert_eq!(s.set_slice_limits(0, 10, false, 0), (0, 0, 1));
    }
    #[test]
    fn texture_path_uploads() {
        let mut s = MvImageState::default();
        s.flags = IMODV_DRAW_CZ;
        let mut a = ImodvApp::default();
        a.gl_ext_flags = 1;
        let mut src = Src {
            data: (0..64).collect(),
        };
        let mut gl = Gl {
            quads: 0,
            texel: 0.,
        };
        s.imodv_draw_image(&mut a, &mut src, 0, &mut gl);
        assert_ne!(s.tex_name, 0);
        assert_eq!(gl.texel, 1. / s.tex_image_size as f32);
    }
    #[test]
    fn all_orthogonal_planes_issue_real_texture_quads() {
        let mut s = MvImageState::default();
        s.flags = IMODV_DRAW_CX | IMODV_DRAW_CY | IMODV_DRAW_CZ;
        let mut a = ImodvApp::default();
        a.gl_ext_flags = 1;
        let mut src = Src {
            data: (0..64).collect(),
        };
        let mut gl = Gl {
            quads: 0,
            texel: 0.,
        };
        s.imodv_draw_image(&mut a, &mut src, 0, &mut gl);
        assert!(gl.quads >= 3, "drawn quads: {}", gl.quads);
    }
    #[test]
    fn ushort_and_rgb_fill_follow_source_bgr_order() {
        let mut state = MvImageState::default();
        state.make_color_map(false);
        state.tdata.resize(4, 0);
        let map = vec![0u8; 256];
        state.fill_image_pixel(1, 0, 0, Some(ImagePixel::Rgb([11, 22, 33])), Some(&map));
        assert_eq!(&state.tdata, &[33, 22, 11, 255]);
        state.fill_image_pixel(1, 0, 0, Some(ImagePixel::UShort(10)), Some(&vec![9; 11]));
        assert_eq!(state.tdata[3], 255);
    }
    #[test]
    fn image_dialog_event_handlers_follow_native_event_order() {
        let mut dialog = ImodvImage::new();
        let mut native = EventBoundary {
            rounded: true,
            ..Default::default()
        };

        dialog.button_pressed(&mut native);
        dialog.top_change_event(&mut native);
        dialog.top_close_event(&mut native);

        assert_eq!(native.cache_section, Some(-1));
        assert!(dialog.rounded_style);
        assert!(dialog.closed);
        assert_eq!(
            native.calls,
            [
                "cache_fill",
                "base_change",
                "mac_menu",
                "remove_dialog",
                "accept_close"
            ]
        );
    }
}
