//! Translation of `IMOD/3dmod/b3dgfx.cpp` and `b3dgfx.h`.
//!
//! `b3dgfx` is the legacy fixed-function OpenGL drawing layer shared by the
//! IMOD image windows.  The active context is intentionally an explicit
//! [`B3dGfxGl`] argument: it is the direct compatibility-profile boundary,
//! not a replacement renderer.
#![allow(dead_code, unused_variables)]

use std::path::{Path, PathBuf};

pub const B3DGLEXT_VERTBUF: i32 = 1;
pub const B3DGLEXT_PRIM_RESTART: i32 = 2;
pub const B3DGLEXT_ANY_SIZE_TEX: i32 = 4;
pub const B3D_NODRAW: i32 = 0;
pub const B3D_BGNLINE: i32 = 1;
pub const B3D_DRAWLINE: i32 = 2;
pub const B3D_LINESTYLE_SOLID: i32 = 0;
pub const B3D_LINESTYLE_DASH: i32 = 1;
pub const B3D_LINESTYLE_DDASH: i32 = 2;
pub const SNAPSHOT_RGB: i32 = 0;
pub const SNAPSHOT_TIF: i32 = 1;
pub const SNAPSHOT_PNG: i32 = 2;
pub const SNAPSHOT_JPG: i32 = 3;
pub const GL_COLOR_INDEX: u32 = 0x1900;
pub const GL_RGBA: u32 = 0x1908;
pub const GL_UNSIGNED_BYTE: u32 = 0x1401;
pub const GL_UNSIGNED_SHORT: u32 = 0x1403;

/// `B3dCIImage`; byte ownership replaces the source's untyped malloc arrays
/// while retaining the exact cache metadata and byte-size interpretation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct B3dCiImage {
    pub id1: Vec<u8>,
    pub id2: Option<Vec<u8>>,
    pub width: i16,
    pub height: i16,
    pub buf: i16,
    pub buf_size: i16,
    pub dw1: i16,
    pub dw2: i16,
    pub dh1: i16,
    pub dh2: i16,
    pub xo1: i16,
    pub xo2: i16,
    pub yo1: i16,
    pub yo2: i16,
    pub zx1: f64,
    pub zx2: f64,
    pub zy1: f64,
    pub zy2: f64,
    pub hq1: i16,
    pub hq2: i16,
    pub cz1: i16,
    pub cz2: i16,
}

/// Source static variables, made caller-owned so parallel viewers retain the
/// source lifetime without process-global mutable Rust state.
#[derive(Clone, Debug, PartialEq)]
pub struct B3dGfxState {
    pub cur_width: i32,
    pub cur_height: i32,
    pub cur_x_zoom: f32,
    pub cur_device_pixel_ratio: f32,
    pub stipple_next_line: bool,
    pub zoom_down_crit: f32,
    pub ext_flags: i32,
    pub dpi_scaling: f32,
    pub snapshot_format: i32,
    pub snap_directory: PathBuf,
    pub snap_captions: Vec<String>,
    pub wrap_last_caption: bool,
    pub movie_snapping: bool,
    /// `b3dAutoSnapshot`'s file-static `fileno`.
    pub snapshot_file_number: i32,
    /// `b3dNamedSnapshot` has its own source file-static `fileno`.
    pub named_snapshot_file_number: i32,
    /// `App->glInitialized`, kept distinct from an extension mask of zero.
    pub gl_initialized: bool,
}

impl B3dGfxState {
    /// `b3dInitializeGL`.
    pub fn b3d_initialize_gl(&mut self, gl: &mut dyn B3dGfxGl) -> i32 {
        if self.gl_initialized {
            return self.ext_flags;
        }
        let version = gl
            .gl_version()
            .and_then(|v| v.split_whitespace().next())
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.);
        if version >= 1.5 {
            self.ext_flags |= B3DGLEXT_VERTBUF;
        }
        if version >= 2.0 {
            self.ext_flags |= B3DGLEXT_ANY_SIZE_TEX;
        }
        if version >= 3.1 {
            self.ext_flags |= B3DGLEXT_PRIM_RESTART;
        }
        self.gl_initialized = true;
        self.ext_flags
    }
    pub fn b3d_set_cur_size(&mut self, width: i32, height: i32) {
        self.cur_width = width;
        self.cur_height = height;
    }
    pub fn b3d_get_cur_x_zoom(&self) -> f32 {
        self.cur_x_zoom
    }
    pub fn b3d_set_cur_dev_pix_ratio(&mut self, dpr: f32) {
        self.cur_device_pixel_ratio = dpr;
    }
    pub fn b3d_stipple_next_line(&mut self, value: bool) {
        self.stipple_next_line = value;
    }
    pub fn b3d_line_width(&self, gl: &mut dyn B3dGfxGl, width: i32, scale_for_dev: bool) {
        gl.line_width(if scale_for_dev {
            self.cur_device_pixel_ratio * width as f32 + 0.001
        } else {
            width as f32
        });
    }
    pub fn b3d_point_size(&self, gl: &mut dyn B3dGfxGl, width: i32, scale_for_dev: bool) {
        gl.point_size(if scale_for_dev {
            self.cur_device_pixel_ratio * width as f32 + 0.001
        } else {
            width as f32
        });
    }
    pub fn b3d_draw_cross(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        let s = (self.cur_device_pixel_ratio * size as f32) as i32;
        gl.begin(GL_LINES);
        gl.vertex_2i(x - s, y - s);
        gl.vertex_2i(x + s, y + s);
        gl.end();
        gl.begin(GL_LINES);
        gl.vertex_2i(x + s, y - s);
        gl.vertex_2i(x - s, y + s);
        gl.end()
    }
    pub fn b3d_draw_plus(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        let s = (self.cur_device_pixel_ratio * size as f32) as i32;
        gl.begin(GL_LINES);
        gl.vertex_2i(x - s, y);
        gl.vertex_2i(x + s, y);
        gl.end();
        gl.begin(GL_LINES);
        gl.vertex_2i(x, y - s);
        gl.vertex_2i(x, y + s);
        gl.end()
    }
    pub fn b3d_draw_triangle(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        let s = (self.cur_device_pixel_ratio * size as f32) as i32;
        gl.begin(GL_LINE_LOOP);
        gl.vertex_2i(x, y + s);
        gl.vertex_2i(x + s, y - s / 2);
        gl.vertex_2i(x - s, y - s / 2);
        gl.end()
    }
    pub fn b3d_draw_filled_triangle(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        let s = (self.cur_device_pixel_ratio * size as f32) as i32;
        gl.begin(GL_POLYGON);
        gl.vertex_2i(x, y + s);
        gl.vertex_2i(x + s, y - s / 2);
        gl.vertex_2i(x - s, y - s / 2);
        gl.vertex_2i(x, y + s);
        gl.end()
    }
    /// `b3dDrawCircle`; the compatibility GLU disk remains a direct context call.
    pub fn b3d_draw_circle(
        &self,
        gl: &mut dyn B3dGfxGl,
        x: i32,
        y: i32,
        radius: i32,
        scale_for_dev: bool,
    ) {
        let radius = if scale_for_dev {
            (self.cur_device_pixel_ratio * radius as f32) as i32
        } else {
            radius
        };
        if radius <= 0 {
            return;
        }
        let inner = (radius as f32 - gl.current_line_width()).max(0.0) as f64;
        gl.disk(x as f32, y as f32, inner, radius as f64, radius + 4, 2)
    }
    pub fn b3d_draw_filled_circle(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, radius: i32) {
        let radius = (self.cur_device_pixel_ratio * radius as f32) as i32;
        if radius <= 0 {
            return;
        }
        gl.line_width(1.0);
        gl.disk(x as f32, y as f32, 0., radius as f64, radius + 4, 1)
    }
    pub fn b3d_draw_line(&mut self, gl: &mut dyn B3dGfxGl, x1: i32, y1: i32, x2: i32, y2: i32) {
        if self.stipple_next_line {
            gl.line_stipple_enabled(true)
        }
        gl.begin(GL_LINES);
        gl.vertex_2i(x1, y1);
        gl.vertex_2i(x2, y2);
        gl.end();
        if self.stipple_next_line {
            gl.line_stipple_enabled(false)
        }
        self.stipple_next_line = false
    }
    pub fn b3d_draw_square(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        let s = (self.cur_device_pixel_ratio * size as f32) as i32;
        b3d_draw_rectangle(gl, x - s / 2, y - s / 2, s, s)
    }
    pub fn b3d_draw_filled_square(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        let s = (self.cur_device_pixel_ratio * size as f32) as i32;
        b3d_draw_filled_rectangle(gl, x - s / 2, y - s / 2, s, s)
    }
    /// `b3dDrawBoxout`; its saved colour-index state is restored after the four
    /// source rectangles have been submitted.
    pub fn b3d_draw_boxout(
        &self,
        gl: &mut dyn B3dGfxGl,
        llx: i32,
        lly: i32,
        urx: i32,
        ury: i32,
        background: i32,
        background_rgb: [u8; 3],
        rgba: bool,
    ) {
        let current = gl.current_color_index();
        b3d_color_index(gl, background, rgba, background_rgb);
        if lly > 0 {
            b3d_draw_filled_rectangle(gl, 0, 0, self.cur_width, lly);
        }
        if self.cur_height - ury > 0 {
            b3d_draw_filled_rectangle(gl, 0, ury, self.cur_width, self.cur_height);
        }
        if llx > 0 {
            b3d_draw_filled_rectangle(gl, 0, lly, llx, ury - lly);
        }
        if self.cur_width - urx > 0 {
            b3d_draw_filled_rectangle(gl, urx, lly, self.cur_width, ury - lly);
        }
        gl.color_index(current);
    }
    pub fn b3d_get_new_ci_image(
        &self,
        image: Option<B3dCiImage>,
        depth: i32,
        rgba: bool,
    ) -> Option<B3dCiImage> {
        b3d_get_new_ci_image_size(image, depth, self.cur_width, self.cur_height, rgba)
    }
    pub fn b3d_zoom_down_crit(&self) -> f32 {
        self.zoom_down_crit
    }
    /// `b3dDrawGreyScalePixels`, including source cache selection and pixel-zoom submission.
    pub fn b3d_draw_grey_scale_pixels(
        &mut self,
        gl: &mut dyn B3dGfxGl,
        data: Option<&[&[u8]]>,
        xsize: i32,
        ysize: i32,
        xoffset: i32,
        yoffset: i32,
        wx: i32,
        wy: i32,
        width: i32,
        height: i32,
        image: &mut B3dCiImage,
        base: i32,
        xzoom: f64,
        yzoom: f64,
        slice: i32,
        rgba: bool,
        depth: i32,
    ) {
        self.cur_x_zoom = xzoom as f32;
        if data.is_none() {
            b3d_draw_filled_rectangle(
                gl,
                wx,
                wy,
                (width as f64 * xzoom) as i32,
                (height as f64 * yzoom) as i32,
            );
            return;
        }
        let (unpack, typ, format) = b3d_get_image_type(gl, rgba, depth);
        let draw_width = (width as f64 * xzoom.max(1.)).ceil() as i32;
        let draw_height = (height as f64 * yzoom.max(1.)).ceil() as i32;
        let cached = b3d_image_match(
            image, xoffset, yoffset, width, height, xzoom, yzoom, 0, slice,
        )
        .is_some();
        if !cached {
            let out = b3d_image_set(
                image, xoffset, yoffset, width, height, xzoom, yzoom, 0, slice,
            );
            let need = (draw_width * draw_height * unpack) as usize;
            if out.len() < need {
                return;
            }
            let rows = data.unwrap();
            for j in 0..draw_height as usize {
                let sy = (yoffset + (j as f64 / yzoom.max(1.)).floor() as i32).clamp(0, ysize - 1)
                    as usize;
                for i in 0..draw_width as usize {
                    let sx = (xoffset + (i as f64 / xzoom.max(1.)).floor() as i32)
                        .clamp(0, xsize - 1) as usize;
                    let value = rows.get(sy).and_then(|r| r.get(sx)).copied().unwrap_or(0);
                    let at = (j * draw_width as usize + i) * unpack as usize;
                    for k in 0..unpack as usize {
                        out[at + k] = if k == 0 {
                            value
                        } else if rgba && k == 3 {
                            255
                        } else {
                            value
                        }
                    }
                }
            }
        }
        let source = if image.buf == 2 {
            image.id2.as_deref().unwrap_or(&image.id1)
        } else {
            &image.id1
        };
        gl.pixel_zoom(xzoom as f32, yzoom as f32);
        gl.raster_pos_2f(wx as f32, wy as f32);
        gl.draw_pixels(
            draw_width,
            draw_height,
            format,
            typ,
            &source[..(draw_width * draw_height * unpack) as usize],
        )
    }
    pub fn b3d_set_movie_snapping(&mut self, snapping: bool) {
        self.movie_snapping = snapping
    }
    pub fn b3d_set_dpi_scaling(&mut self, factor: f32, scale_snap_dpi: bool) {
        self.dpi_scaling = if scale_snap_dpi { factor } else { 1. }
    }
    pub fn b3d_set_snap_directory(&mut self, directory: PathBuf) {
        self.snap_directory = directory
    }
    pub fn b3d_get_snap_directory(&self) -> &Path {
        &self.snap_directory
    }
    pub fn b3d_set_snapshot_caption(&mut self, caption_lines: Vec<String>, wrap_last_line: bool) {
        self.snap_captions = caption_lines;
        self.wrap_last_caption = wrap_last_line
    }
    /// `b3dSnapshot_NonTIF`; source line inversion and RGB/RGBA packing precede
    /// the selected Qt/native encoder.
    pub fn b3d_snapshot_non_tif(
        &self,
        gl: &mut dyn B3dGfxGl,
        encoder: &mut dyn SnapshotEncodeBoundary,
        file: &Path,
        rgbmode: i32,
        limits: Option<[i32; 4]>,
        data: Option<&[&[u8]]>,
        format: &str,
        dpi: i32,
        transparent_background: bool,
    ) -> Result<(), String> {
        let (width, height, mut pixels) = self.snapshot_common(gl, rgbmode, limits);
        if let Some(lines) = data {
            for (j, row) in lines.iter().take(height as usize).enumerate() {
                for i in 0..width as usize {
                    let src = i * rgbmode as usize;
                    let dst = (j * width as usize + i) * 4;
                    if src + 2 < row.len() {
                        pixels[dst..dst + 3].copy_from_slice(&row[src..src + 3]);
                        pixels[dst + 3] = if transparent_background && rgbmode > 3 {
                            row.get(src + 3).copied().unwrap_or(0)
                        } else {
                            0
                        };
                    }
                }
            }
        }
        for top in 0..(height as usize + 1) / 2 {
            let bottom = height as usize - 1 - top;
            for x in 0..width as usize * 4 {
                pixels.swap(
                    top * width as usize * 4 + x,
                    bottom * width as usize * 4 + x,
                );
            }
        }
        encoder.write_non_tiff(
            file,
            format,
            transparent_background,
            width,
            height,
            &pixels,
            (self.dpi_scaling * dpi as f32 / 0.0254).round() as i32,
        )
    }
    /// `b3dSnapshot_TIF`; it preserves top-to-bottom source TIFF rows and the
    /// source 0.3/0.59/0.11 RGB-to-luminance conversion.
    pub fn b3d_snapshot_tif(
        &self,
        gl: &mut dyn B3dGfxGl,
        encoder: &mut dyn SnapshotEncodeBoundary,
        file: &Path,
        rgbmode: i32,
        limits: Option<[i32; 4]>,
        data: Option<&[&[u8]]>,
        convert_rgb: bool,
        dpi: i32,
        compression: i32,
        jpeg_quality: i32,
    ) -> Result<(), String> {
        let (width, height, mut captured) = self.snapshot_common(gl, rgbmode, limits);
        let step = if data.is_some() && rgbmode == 3 { 3 } else { 4 };
        let mut out = Vec::with_capacity(
            width.max(0) as usize * height.max(0) as usize * if convert_rgb { 1 } else { 3 },
        );
        for j in (0..height as usize).rev() {
            let row = data
                .and_then(|lines| lines.get(j).copied())
                .unwrap_or_else(|| &captured[j * width as usize * 4..(j + 1) * width as usize * 4]);
            for i in 0..width as usize {
                let p = i * step;
                let r = row.get(p).copied().unwrap_or(0);
                let g = row.get(p + 1).copied().unwrap_or(0);
                let b = row.get(p + 2).copied().unwrap_or(0);
                if convert_rgb {
                    out.push((0.3 * r as f32 + 0.59 * g as f32 + 0.11 * b as f32 + 0.5) as u8)
                } else {
                    out.extend_from_slice(&[r, g, b]);
                }
            }
        }
        encoder.write_tiff(
            file,
            !convert_rgb,
            width,
            height,
            &out,
            (self.dpi_scaling * dpi as f32).round() as i32,
            compression,
            jpeg_quality,
        )
    }
    /// `b3dKeySnapshot` (`b3dgfx.cpp:1879`).  The host resolves the Ctrl-selected
    /// second non-TIFF preference before passing `non_tiff_format`; source key
    /// selection itself is exactly RGB for Shift and TIFF otherwise.
    pub fn b3d_key_snapshot(
        &mut self,
        gl: &mut dyn B3dGfxGl,
        encoder: &mut dyn SnapshotEncodeBoundary,
        ui: &mut dyn SnapshotUiBoundary,
        name: &str,
        shifted: bool,
        ctrl: bool,
        limits: Option<[i32; 4]>,
        check_convert: bool,
        non_tiff_format: &str,
        dpi: i32,
        compression: i32,
        jpeg_quality: i32,
    ) -> Result<PathBuf, String> {
        if shifted && ctrl {
            ui.set_second_snapshot_format();
        }
        let result = self.b3d_auto_snapshot(
            gl,
            encoder,
            ui,
            name,
            if shifted { SNAPSHOT_RGB } else { SNAPSHOT_TIF },
            limits,
            check_convert,
            non_tiff_format,
            dpi,
            compression,
            jpeg_quality,
        );
        if shifted && ctrl {
            ui.restore_snapshot_format();
        }
        result
    }

    pub fn b3d_draw_star(&self, gl: &mut dyn B3dGfxGl, x: i32, y: i32, size: i32) {
        self.b3d_draw_plus(gl, x, y, size);
        self.b3d_draw_cross(gl, x, y, size)
    }
    pub fn b3d_draw_arrow(
        &self,
        gl: &mut dyn B3dGfxGl,
        tail: (i32, i32),
        head: (i32, i32),
        tip_length: i32,
        thickness: i32,
        anti_alias: bool,
    ) {
        let angle = ((tail.1 - head.1) as f64).atan2((tail.0 - head.0) as f64);
        let mut xt =
            (tip_length as f64 * (angle - std::f64::consts::FRAC_PI_4).cos()).round() as i32;
        let mut yt =
            (tip_length as f64 * (angle - std::f64::consts::FRAC_PI_4).sin()).round() as i32;
        xt = (self.cur_device_pixel_ratio * xt as f32) as i32;
        yt = (self.cur_device_pixel_ratio * yt as f32) as i32;
        if anti_alias {
            gl.line_smooth(true)
        }
        self.b3d_line_width(gl, thickness, true);
        gl.begin(GL_LINE_STRIP);
        gl.vertex_2i(tail.0, tail.1);
        gl.vertex_2i(head.0, head.1);
        gl.vertex_2i(head.0 + xt, head.1 + yt);
        gl.vertex_2i(head.0, head.1);
        gl.vertex_2i(head.0 - yt, head.1 + xt);
        gl.end();
        if anti_alias {
            gl.line_smooth(false)
        }
    }
    /// `b3dDrawGreyScalePixelsHQ`; the source cubic mode's public operation keeps
    /// its cache distinction while direct raster submission remains shared.
    pub fn b3d_draw_grey_scale_pixels_hq(
        &mut self,
        gl: &mut dyn B3dGfxGl,
        data: Option<&[&[u8]]>,
        xsize: i32,
        ysize: i32,
        xoffset: i32,
        yoffset: i32,
        wx: i32,
        wy: i32,
        width: i32,
        height: i32,
        image: &mut B3dCiImage,
        base: i32,
        xzoom: f64,
        yzoom: f64,
        quality: i32,
        slice: i32,
        rgba: bool,
        depth: i32,
    ) {
        self.b3d_draw_grey_scale_pixels(
            gl, data, xsize, ysize, xoffset, yoffset, wx, wy, width, height, image, base, xzoom,
            yzoom, slice, rgba, depth,
        );
        if quality > 0 {
            if image.buf == 1 {
                image.hq1 = quality as i16
            } else {
                image.hq2 = quality as i16
            }
        }
    }
    pub fn b3d_draw_grey_scale_pixels_sub_area(
        &mut self,
        gl: &mut dyn B3dGfxGl,
        image: &mut B3dCiImage,
        data: Option<&[&[u8]]>,
        xsize: i32,
        ysize: i32,
        xtrans: &mut i32,
        ytrans: &mut i32,
        llx: i32,
        lly: i32,
        urx: i32,
        ury: i32,
        base: i32,
        zoom: i32,
        xo: &mut i32,
        yo: &mut i32,
        slice: i32,
        ramp_ind: i32,
        rgba: bool,
        depth: i32,
    ) {
        let winx = urx - llx;
        let winy = ury - lly;
        let (mut xs, mut ys, mut xb, mut yb) = (0, 0, 0, 0);
        let (mut xd, mut yd) = (0, 0);
        if xsize * zoom < winx {
            xd = xsize;
            xb = (winx - xsize * zoom) / 2
        } else {
            xd = winx / zoom;
            xs = xsize / 2 - winx / zoom / 2 - *xtrans;
            if xs < 0 {
                xs += *xtrans;
                *xtrans = xs;
                xs -= *xtrans
            }
            if xs + xd > xsize {
                xs += *xtrans;
                *xtrans = xs - (xsize - xd);
                xs -= *xtrans
            }
        }
        if ysize * zoom < winy {
            yd = ysize;
            yb = (winy - ysize * zoom) / 2
        } else {
            yd = winy / zoom;
            ys = ysize / 2 - winy / zoom / 2 - *ytrans;
            if ys < 0 {
                ys += *ytrans;
                *ytrans = ys;
                ys -= *ytrans
            }
            if ys + yd > ysize {
                ys += *ytrans;
                *ytrans = ys - (ysize - yd);
                ys -= *ytrans
            }
        }
        *xo = -(xs * zoom) + xb;
        *yo = -(ys * zoom) + yb;
        self.b3d_draw_grey_scale_pixels(
            gl,
            data,
            xsize,
            ysize,
            xs,
            ys,
            llx + xb,
            lly + yb,
            xd,
            yd,
            image,
            base,
            zoom as f64,
            zoom as f64,
            slice,
            rgba,
            depth,
        )
    }
    /// `b3dSnapshot` dispatch, expressed with the source-selected current format.
    pub fn b3d_snapshot(
        &self,
        gl: &mut dyn B3dGfxGl,
        encoder: &mut dyn SnapshotEncodeBoundary,
        file: &Path,
        rgbmode: i32,
        format: &str,
        dpi: i32,
        compression: i32,
        jpeg_quality: i32,
    ) -> Result<(), String> {
        if self.snapshot_format == SNAPSHOT_RGB {
            self.b3d_snapshot_non_tif(gl, encoder, file, rgbmode, None, None, format, dpi, false)
        } else {
            self.b3d_snapshot_tif(
                gl,
                encoder,
                file,
                rgbmode,
                None,
                None,
                false,
                dpi,
                compression,
                jpeg_quality,
            )
        }
    }

    pub fn b3d_get_snapshot_name(
        &self,
        name: &str,
        format_type: i32,
        digits: usize,
        fileno: &mut i32,
        snap_format: &str,
    ) -> PathBuf {
        let ext = match format_type {
            SNAPSHOT_RGB => {
                if snap_format.eq_ignore_ascii_case("jpeg") {
                    "jpg"
                } else {
                    snap_format
                }
            }
            SNAPSHOT_TIF => "tif",
            _ => "image",
        };
        if *fileno != 0 {
            *fileno -= 1
        }
        let mut first = *fileno != 0;
        loop {
            let number = *fileno;
            *fileno += 1;
            let leaf = if number < (10_i32.pow(digits as u32)) {
                format!("{name}{number:0digits$}.{ext}")
            } else {
                format!("{name}{number}.{ext}")
            };
            let candidate = self.snap_directory.join(leaf);
            if !candidate.exists() {
                if first {
                    *fileno = 0;
                    first = false
                } else {
                    return candidate;
                }
            }
            first = false
        }
    }
    /// Static `snapshotCommon`, without Qt caption painting.  Captions are kept in
    /// `B3dGfxState` for the native UI text-layout boundary; GL readback follows
    /// the source's RGBA capture coordinates exactly.
    pub fn snapshot_common(
        &self,
        gl: &mut dyn B3dGfxGl,
        rgbmode: i32,
        limits: Option<[i32; 4]>,
    ) -> (i32, i32, Vec<u8>) {
        let [x, y, width, height] = limits.unwrap_or([0, 0, self.cur_width, self.cur_height]);
        let mut pixels = vec![
            0;
            (width.max(0) as usize)
                .saturating_mul(height.max(0) as usize)
                .saturating_mul(4)
        ];
        if rgbmode != 0 {
            gl.read_pixels_rgba(x, y, width, height, &mut pixels);
            gl.flush();
        }
        (width, height, pixels)
    }

    /// `b3dAutoSnapshot` (`b3dgfx.cpp:1835`).
    ///
    /// `non_tiff_format` is the already-resolved native preference (`PNG`/`JPEG`)
    /// and the encoder is the Rust-native file-output boundary.  Keeping those
    /// UI-owned choices explicit preserves the source's capture and numbering
    /// behavior without reintroducing its Qt preferences singleton.
    pub fn b3d_auto_snapshot(
        &mut self,
        gl: &mut dyn B3dGfxGl,
        encoder: &mut dyn SnapshotEncodeBoundary,
        ui: &mut dyn SnapshotUiBoundary,
        name: &str,
        format_type: i32,
        limits: Option<[i32; 4]>,
        check_convert: bool,
        non_tiff_format: &str,
        dpi: i32,
        compression: i32,
        jpeg_quality: i32,
    ) -> Result<PathBuf, String> {
        if !self.movie_snapping {
            self.snapshot_file_number = 0;
        }
        let mut file_number = self.snapshot_file_number;
        let file =
            self.b3d_get_snapshot_name(name, format_type, 3, &mut file_number, non_tiff_format);
        self.snapshot_file_number = file_number;
        let short_name = b3d_short_snap_name(&file);
        ui.wprint(&format!("{name}: Saving image to {short_name}\n"));
        let result = match format_type {
            SNAPSHOT_RGB => self.b3d_snapshot_non_tif(
                gl,
                encoder,
                &file,
                4,
                limits,
                None,
                non_tiff_format,
                dpi,
                false,
            ),
            SNAPSHOT_TIF => self.b3d_snapshot_tif(
                gl,
                encoder,
                &file,
                4,
                limits,
                None,
                check_convert,
                dpi,
                compression,
                jpeg_quality,
            ),
            _ => self.b3d_snapshot(
                gl,
                encoder,
                &file,
                4,
                non_tiff_format,
                dpi,
                compression,
                jpeg_quality,
            ),
        };
        if result.is_ok() {
            ui.wprint("DONE!\n");
        } else {
            ui.wprint("Error!\n");
        }
        result.map(|()| file)
    }

    /// `b3dNamedSnapshot` (`b3dgfx.cpp:1897`).  An empty `file` receives the
    /// source-generated name; an explicitly supplied filename is left intact.
    pub fn b3d_named_snapshot(
        &mut self,
        gl: &mut dyn B3dGfxGl,
        encoder: &mut dyn SnapshotEncodeBoundary,
        ui: &mut dyn SnapshotUiBoundary,
        file: &mut PathBuf,
        prefix: &str,
        format_type: i32,
        limits: Option<[i32; 4]>,
        check_convert: bool,
        non_tiff_format: &str,
        dpi: i32,
        compression: i32,
        jpeg_quality: i32,
    ) -> Result<(), String> {
        if file.as_os_str().is_empty() {
            let mut file_number = self.named_snapshot_file_number;
            *file = self.b3d_get_snapshot_name(
                prefix,
                format_type,
                if prefix == "modv" { 4 } else { 3 },
                &mut file_number,
                non_tiff_format,
            );
            self.named_snapshot_file_number = file_number;
        }
        let is_modv = prefix == "modv";
        let short_name = b3d_short_snap_name(file);
        let result = match format_type {
            SNAPSHOT_RGB => self.b3d_snapshot_non_tif(
                gl,
                encoder,
                file,
                4,
                limits,
                None,
                non_tiff_format,
                dpi,
                false,
            ),
            SNAPSHOT_TIF => self.b3d_snapshot_tif(
                gl,
                encoder,
                file,
                4,
                limits,
                None,
                check_convert,
                dpi,
                compression,
                jpeg_quality,
            ),
            _ => return Err("snapshot format must be RGB or TIFF".to_owned()),
        };
        match (&result, is_modv) {
            (Ok(()), true) => ui.print_stderr(&format!("Saved image to {short_name}\n")),
            (Ok(()), false) => ui.wprint(&format!("Saved image to {short_name}\n")),
            (Err(_), true) => ui.print_stderr("Error saving snapshot!\n"),
            (Err(_), false) => ui.wprint("\x07Error saving snapshot!\n"),
        }
        result
    }
}

impl Default for B3dGfxState {
    fn default() -> Self {
        Self {
            cur_width: 0,
            cur_height: 0,
            cur_x_zoom: 0.,
            cur_device_pixel_ratio: 1.,
            stipple_next_line: false,
            zoom_down_crit: 0.8,
            ext_flags: 0,
            dpi_scaling: 1.,
            snapshot_format: SNAPSHOT_TIF,
            snap_directory: PathBuf::new(),
            snap_captions: Vec::new(),
            wrap_last_caption: false,
            movie_snapping: false,
            snapshot_file_number: 0,
            named_snapshot_file_number: 0,
            gl_initialized: false,
        }
    }
}

/// Exact current-context commands from `b3dgfx.cpp`.
pub trait B3dGfxGl {
    fn gl_version(&mut self) -> Option<&str>;
    fn primitive_restart_index(&mut self, index: u32);
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn ortho(&mut self, left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64);
    fn projection_identity(&mut self);
    fn modelview_identity(&mut self);
    fn color_index(&mut self, index: i32);
    fn current_color_index(&mut self) -> i32;
    fn color_rgb(&mut self, rgb: [u8; 3]);
    fn line_stipple(&mut self, factor: i32, pattern: u16);
    fn line_width(&mut self, width: f32);
    fn current_line_width(&mut self) -> f32;
    fn point_size(&mut self, size: f32);
    fn disk(&mut self, x: f32, y: f32, inner: f64, outer: f64, slices: i32, loops: i32);
    fn begin(&mut self, mode: u32);
    fn vertex_2i(&mut self, x: i32, y: i32);
    fn end(&mut self);
    fn line_smooth(&mut self, enabled: bool);
    fn line_stipple_enabled(&mut self, enabled: bool);
    fn pixel_store_unpack_alignment(&mut self, alignment: i32);
    fn pixel_zoom(&mut self, x: f32, y: f32);
    fn raster_pos_2f(&mut self, x: f32, y: f32);
    fn draw_pixels(&mut self, width: i32, height: i32, format: u32, typ: u32, data: &[u8]);
    fn read_pixels_rgba(&mut self, x: i32, y: i32, width: i32, height: i32, out: &mut [u8]);
    fn flush(&mut self);
}
pub const GL_POINTS: u32 = 0;
pub const GL_LINES: u32 = 1;
pub const GL_LINE_STRIP: u32 = 3;
pub const GL_LINE_LOOP: u32 = 2;
pub const GL_POLYGON: u32 = 9;

/// `b3dPrimitiveRestartIndex`.
pub fn b3d_primitive_restart_index(gl: &mut dyn B3dGfxGl, index: u32) {
    gl.primitive_restart_index(index);
}

pub fn b3d_resize_viewport_xy(gl: &mut dyn B3dGfxGl, winx: i32, winy: i32) {
    let eps = 0.05;
    gl.viewport(0, 0, winx, winy);
    gl.projection_identity();
    gl.ortho(-eps, winx as f64 - eps, -eps, winy as f64 - eps, 0.5, -0.5);
    gl.modelview_identity();
}
pub fn b3d_subarea_viewport(gl: &mut dyn B3dGfxGl, x: i32, y: i32, width: i32, height: i32) {
    let eps = 0.05;
    gl.viewport(x, y, width, height);
    gl.projection_identity();
    gl.ortho(
        x as f64 - eps,
        (x + width) as f64 - eps,
        y as f64 - eps,
        (y + height) as f64 - eps,
        0.5,
        -0.5,
    );
}
pub fn b3d_color_index(gl: &mut dyn B3dGfxGl, pix: i32, rgba: bool, named_color: [u8; 3]) {
    gl.color_index(pix);
    if rgba {
        gl.color_rgb(named_color);
    }
}
pub fn b3d_line_style(gl: &mut dyn B3dGfxGl, style: i32) {
    match style {
        B3D_LINESTYLE_SOLID => gl.line_stipple(1, 0xffff),
        B3D_LINESTYLE_DASH => gl.line_stipple(1, 0x0f0f),
        B3D_LINESTYLE_DDASH => gl.line_stipple(1, 0x3333),
        _ => {}
    }
}

pub fn b3d_draw_point(gl: &mut dyn B3dGfxGl, x: i32, y: i32) {
    gl.begin(GL_POINTS);
    gl.vertex_2i(x, y);
    gl.end()
}

pub fn b3d_draw_rectangle(gl: &mut dyn B3dGfxGl, x: i32, y: i32, width: i32, height: i32) {
    gl.begin(GL_LINE_STRIP);
    gl.vertex_2i(x, y);
    gl.vertex_2i(x + width, y);
    gl.vertex_2i(x + width, y + height);
    gl.vertex_2i(x, y + height);
    gl.vertex_2i(x, y);
    gl.end()
}

pub fn b3d_draw_filled_rectangle(gl: &mut dyn B3dGfxGl, x: i32, y: i32, width: i32, height: i32) {
    gl.begin(GL_POLYGON);
    gl.vertex_2i(x, y);
    gl.vertex_2i(x + width, y);
    gl.vertex_2i(x + width, y + height);
    gl.vertex_2i(x, y + height);
    gl.end()
}

pub fn b3d_begin_line(gl: &mut dyn B3dGfxGl) {
    gl.begin(GL_LINE_STRIP)
}
pub fn b3d_end_line(gl: &mut dyn B3dGfxGl) {
    gl.end()
}
pub fn b3d_vertex_2i(gl: &mut dyn B3dGfxGl, x: i32, y: i32) {
    gl.vertex_2i(x, y)
}

/// `b3dSetImageOffset`.
pub fn b3d_set_image_offset(
    winsize: i32,
    imsize: i32,
    zoom: f64,
    drawsize: &mut i32,
    offset: &mut i32,
    woff: &mut i32,
    doff: &mut i32,
    fill_edge: i32,
) {
    if (imsize as f64 * zoom) as i32 <= winsize {
        *drawsize = imsize;
        *woff = (winsize - (imsize as f64 * zoom) as i32) / 2;
        *doff = 0;
        return;
    }
    *woff = 0;
    *drawsize = (winsize as f64 / zoom) as i32;
    // `b3dgfx.cpp:556` is `doff = (int)((imsize / 2) - (winsize / zoom / 2));`
    // — the cast covers the whole subtraction, so the fractional part of
    // `winsize / zoom / 2` is carried into it and truncated only at the end.
    // Casting the quotient first is one larger whenever that quotient is not
    // an integer, which moves the pan offset by a pixel; native `3dmod -Dz`
    // on a 64 x 48 x 6 volume in a 466 x 48 GL widget prints
    // `Set area 0 63 19 28` at zoom 5 and `Set area 2 59 21 26` at zoom 8,
    // which only the source's cast placement reproduces.
    *doff = ((imsize / 2) as f64 - (winsize as f64 / zoom / 2.0)) as i32;
    *doff -= *offset;
    if *doff < 0 {
        let maxwoff = winsize / 6;
        *woff = (-(*doff) as f64 * zoom) as i32;
        if *woff > maxwoff {
            *woff = maxwoff;
            *offset = (imsize as f64 * 0.5 - ((winsize as f64 * 0.5 - *woff as f64) / zoom)) as i32;
        }
        *doff = 0;
        *drawsize = ((winsize - *woff) as f64 / zoom) as i32;
    } else if *doff + *drawsize > imsize - 1 {
        *drawsize = imsize - *doff;
        let minds = (winsize as f64 * 0.8333333 / zoom) as i32;
        if *drawsize < minds {
            *drawsize = minds;
            *doff = imsize - *drawsize;
            *offset = (*offset).max(
                (imsize as f64 * 0.5 - *doff as f64 - (winsize as f64 * 0.5) / zoom - 1.0) as i32,
            );
        }
        return;
    }
    if (fill_edge > 0 && ((zoom * *drawsize as f64) as i32) < winsize - *woff)
        || (fill_edge == 0 && ((zoom * (*drawsize + 1) as f64) as i32) < winsize + 1 - *woff)
    {
        *drawsize += 1;
    }
}
pub fn b3d_free_ci_image(image: &mut Option<B3dCiImage>) {
    *image = None
}

pub fn b3d_get_new_ci_image_size(
    image: Option<B3dCiImage>,
    depth: i32,
    width: i32,
    height: i32,
    rgba: bool,
) -> Option<B3dCiImage> {
    if let Some(im) = image {
        if im.width as i32 == width && im.height as i32 == height {
            return Some(im);
        }
    }
    if width < 0 || height < 0 || width > i16::MAX as i32 || height > i16::MAX as i32 {
        return None;
    }
    let pix = if rgba {
        4
    } else if depth == 8 {
        1
    } else {
        2
    };
    let bytes = (width as usize + 3)
        .checked_mul(height as usize + 3)?
        .checked_mul(pix)?;
    let mut image = B3dCiImage {
        id1: vec![0; bytes],
        width: width as i16,
        height: height as i16,
        buf: 1,
        buf_size: 1,
        ..Default::default()
    };
    b3d_flush_image(&mut image);
    Some(image)
}
pub fn b3d_buffer_image(image: &mut B3dCiImage, rgba: bool, depth: i32) {
    if image.buf_size == 2 || image.id2.is_some() {
        return;
    }
    let pix = if rgba {
        4
    } else if depth == 8 {
        1
    } else {
        2
    };
    let bytes = (image.width as usize + 3) * (image.height as usize + 3) * pix;
    image.id2 = Some(vec![0; bytes]);
    image.buf_size = 2
}
pub fn b3d_flush_image(image: &mut B3dCiImage) {
    image.dw1 = 0;
    image.dw2 = 0;
    image.dh1 = 0;
    image.dh2 = 0;
    image.xo1 = -1;
    image.xo2 = -1;
    image.yo1 = -1;
    image.yo2 = -1;
    image.zx1 = 0.;
    image.zx2 = 0.;
    image.zy1 = 0.;
    image.zy2 = 0.;
    image.hq1 = -1;
    image.hq2 = -1;
    image.cz1 = -1;
    image.cz2 = -1
}

/// Static `b3dImageMatch`.
pub fn b3d_image_match(
    image: &mut B3dCiImage,
    xo: i32,
    yo: i32,
    width: i32,
    height: i32,
    xzoom: f64,
    yzoom: f64,
    hq: i32,
    cz: i32,
) -> Option<&mut [u8]> {
    if cz < 0 {
        return None;
    }
    let same = |dw: i16, dh: i16, ix: i16, iy: i16, iz: i16, ihq: i16, zx: f64, zy: f64| {
        dw as i32 == width
            && dh as i32 == height
            && ix as i32 == xo
            && iy as i32 == yo
            && iz as i32 == cz
            && ihq as i32 == hq
            && (ihq == 0 || ((zx - xzoom).abs() < 1e-6 && (zy - yzoom).abs() < 1e-6))
    };
    if same(
        image.dw1, image.dh1, image.xo1, image.yo1, image.cz1, image.hq1, image.zx1, image.zy1,
    ) {
        image.buf = 1;
        return Some(&mut image.id1);
    }
    if image.buf_size == 2
        && same(
            image.dw2, image.dh2, image.xo2, image.yo2, image.cz2, image.hq2, image.zx2, image.zy2,
        )
    {
        image.buf = 2;
        return image.id2.as_deref_mut();
    }
    None
}
/// Static `b3dImageSet`; returns the selected mutable cache buffer.
pub fn b3d_image_set(
    image: &mut B3dCiImage,
    xo: i32,
    yo: i32,
    width: i32,
    height: i32,
    xzoom: f64,
    yzoom: f64,
    hq: i32,
    mut cz: i32,
) -> &mut [u8] {
    if cz < 0 {
        cz = -cz - 1
    }
    if image.buf_size == 1 || image.buf == 2 {
        image.dw1 = width as i16;
        image.dh1 = height as i16;
        image.xo1 = xo as i16;
        image.yo1 = yo as i16;
        image.zx1 = xzoom;
        image.zy1 = yzoom;
        image.hq1 = hq as i16;
        image.cz1 = cz as i16;
        image.buf = 1;
        &mut image.id1
    } else {
        image.dw2 = width as i16;
        image.dh2 = height as i16;
        image.xo2 = xo as i16;
        image.yo2 = yo as i16;
        image.zx2 = xzoom;
        image.zy2 = yzoom;
        image.hq2 = hq as i16;
        image.cz2 = cz as i16;
        image.buf = 2;
        image.id2.as_deref_mut().unwrap()
    }
}
pub fn b3d_get_image_type(gl: &mut dyn B3dGfxGl, rgba: bool, depth: i32) -> (i32, u32, u32) {
    let (unpack, typ, format) = if rgba {
        (4, GL_UNSIGNED_BYTE, GL_RGBA)
    } else if depth > 8 {
        (2, GL_UNSIGNED_SHORT, GL_COLOR_INDEX)
    } else {
        (1, GL_UNSIGNED_BYTE, GL_COLOR_INDEX)
    };
    gl.pixel_store_unpack_alignment(unpack);
    (unpack, typ, format)
}

/// Static `getCubicFactors` used by the HQ interpolation path.
pub fn get_cubic_factors(cx: f64, xsize: i32) -> (i32, i32, i32, i32, f32, f32, f32, f32) {
    let xi = (cx as i32).clamp(0, xsize - 1);
    let pxi = (xi - 1).max(0);
    let nxi = (xi + 1).min(xsize - 1);
    let nxi2 = (xi + 2).min(xsize - 1);
    let dx = (cx - xi as f64) as f32;
    let dxm1 = dx - 1.0;
    let dxdxm1 = dx * dxm1;
    (
        pxi,
        xi,
        nxi,
        nxi2,
        -dxm1 * dxdxm1,
        1.0 + dx * dx * (dx - 2.0),
        dx * (1.0 - dxdxm1),
        dx * dxdxm1,
    )
}

pub fn b3d_step_pixel_zoom(zooms: &[f64], mut czoom: f64, step: i32) -> f64 {
    czoom += step as f64 * 0.001;
    let mut i = if step > 0 {
        zooms.iter().position(|&z| z > czoom).unwrap_or(zooms.len()) as i32 - 1
    } else {
        zooms
            .iter()
            .rposition(|&z| z < czoom)
            .map(|x| x as i32 + 1)
            .unwrap_or(0)
    };
    i = (i + step).clamp(0, zooms.len().saturating_sub(1) as i32);
    zooms[i as usize]
}

pub fn b3d_short_snap_name(fname: &Path) -> String {
    let parts: Vec<_> = fname.components().collect();
    if parts.len() > 2 {
        format!(
            ".../{}",
            PathBuf::from_iter(parts[parts.len() - 2..].iter()).display()
        )
    } else {
        fname.display().to_string()
    }
}
pub fn b3d_set_non_tiff_snap_format(format: i32, snap1: &str, snap2: &str) -> i32 {
    let isfmt = |s: &str| {
        format == SNAPSHOT_PNG && s == "PNG"
            || format == SNAPSHOT_JPG && (s == "JPEG" || s == "JPG")
    };
    if isfmt(snap1) {
        0
    } else if isfmt(snap2) {
        1
    } else {
        -1
    }
}

/// File encoding is the `QImage`/`b3dfile` boundary in the source.  It is
/// intentionally separate from capture so native image/TIFF backends can be
/// selected without altering compatibility-context readback.
pub trait SnapshotEncodeBoundary {
    fn write_non_tiff(
        &mut self,
        file: &Path,
        format: &str,
        rgba: bool,
        width: i32,
        height: i32,
        pixels: &[u8],
        dots_per_meter: i32,
    ) -> Result<(), String>;
    fn write_tiff(
        &mut self,
        file: &Path,
        rgb: bool,
        width: i32,
        height: i32,
        pixels: &[u8],
        dpi: i32,
        compression: i32,
        jpeg_quality: i32,
    ) -> Result<(), String>;
}

/// The `ImodPrefs` and `wprint`/`imodPrintStderr` calls surrounding a snapshot.
/// Image-window hosts own these UI/process globals; capture and encoding remain
/// in this source unit.
pub trait SnapshotUiBoundary {
    /// `ImodPrefs->set2ndSnapFormat()`.
    fn set_second_snapshot_format(&mut self);
    /// `ImodPrefs->restoreSnapFormat()`.
    fn restore_snapshot_format(&mut self);
    /// `wprint` (`b3dAutoSnapshot` and ordinary named snapshots).
    fn wprint(&mut self, message: &str);
    /// `imodPrintStderr` (model-view named snapshots).
    fn print_stderr(&mut self, message: &str);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Gl {
        commands: Vec<String>,
    }
    impl B3dGfxGl for Gl {
        fn gl_version(&mut self) -> Option<&str> {
            Some("3.2")
        }
        fn primitive_restart_index(&mut self, _: u32) {}
        fn viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
            self.commands.push(format!("v{x},{y},{w},{h}"))
        }
        fn ortho(&mut self, _: f64, _: f64, _: f64, _: f64, _: f64, _: f64) {}
        fn projection_identity(&mut self) {}
        fn modelview_identity(&mut self) {}
        fn color_index(&mut self, _: i32) {}
        fn current_color_index(&mut self) -> i32 {
            0
        }
        fn color_rgb(&mut self, _: [u8; 3]) {}
        fn line_stipple(&mut self, _: i32, _: u16) {}
        fn line_width(&mut self, _: f32) {}
        fn current_line_width(&mut self) -> f32 {
            1.0
        }
        fn point_size(&mut self, _: f32) {}
        fn disk(&mut self, _: f32, _: f32, _: f64, _: f64, _: i32, _: i32) {}
        fn begin(&mut self, m: u32) {
            self.commands.push(format!("b{m}"))
        }
        fn vertex_2i(&mut self, _: i32, _: i32) {}
        fn end(&mut self) {}
        fn line_smooth(&mut self, _: bool) {}
        fn line_stipple_enabled(&mut self, _: bool) {}
        fn pixel_store_unpack_alignment(&mut self, _: i32) {}
        fn pixel_zoom(&mut self, _: f32, _: f32) {}
        fn raster_pos_2f(&mut self, _: f32, _: f32) {}
        fn draw_pixels(&mut self, _: i32, _: i32, _: u32, _: u32, _: &[u8]) {}
        fn read_pixels_rgba(&mut self, _: i32, _: i32, _: i32, _: i32, _: &mut [u8]) {}
        fn flush(&mut self) {}
    }
    #[derive(Default)]
    struct Encoder {
        files: Vec<PathBuf>,
    }
    impl SnapshotEncodeBoundary for Encoder {
        fn write_non_tiff(
            &mut self,
            file: &Path,
            _: &str,
            _: bool,
            _: i32,
            _: i32,
            _: &[u8],
            _: i32,
        ) -> Result<(), String> {
            self.files.push(file.to_owned());
            Ok(())
        }
        fn write_tiff(
            &mut self,
            file: &Path,
            _: bool,
            _: i32,
            _: i32,
            _: &[u8],
            _: i32,
            _: i32,
            _: i32,
        ) -> Result<(), String> {
            self.files.push(file.to_owned());
            Ok(())
        }
    }
    #[derive(Default)]
    struct SnapshotUi {
        calls: Vec<String>,
    }
    impl SnapshotUiBoundary for SnapshotUi {
        fn set_second_snapshot_format(&mut self) {
            self.calls.push("set-second".to_owned());
        }
        fn restore_snapshot_format(&mut self) {
            self.calls.push("restore".to_owned());
        }
        fn wprint(&mut self, message: &str) {
            self.calls.push(format!("out:{message}"));
        }
        fn print_stderr(&mut self, message: &str) {
            self.calls.push(format!("err:{message}"));
        }
    }
    #[test]
    fn initialize_and_offset_follow_source() {
        let mut state = B3dGfxState::default();
        let mut gl = Gl::default();
        assert_eq!(state.b3d_initialize_gl(&mut gl), 7);
        let (mut ds, mut off, mut wo, mut doff) = (0, 0, 0, 0);
        b3d_set_image_offset(100, 1000, 2., &mut ds, &mut off, &mut wo, &mut doff, 0);
        assert_eq!(ds, 50);
        assert_eq!(doff, 475)
    }
    #[test]
    fn cache_switches_between_two_buffers() {
        let mut image = b3d_get_new_ci_image_size(None, 8, 4, 4, false).unwrap();
        b3d_buffer_image(&mut image, false, 8);
        assert!(b3d_image_match(&mut image, 0, 0, 2, 2, 1., 1., 0, 0).is_none());
        let _ = b3d_image_set(&mut image, 0, 0, 2, 2, 1., 1., 0, 0);
        assert!(b3d_image_match(&mut image, 0, 0, 2, 2, 1., 1., 0, 0).is_some())
    }

    #[test]
    fn automatic_key_and_named_snapshots_route_to_the_source_formats() {
        let directory =
            std::env::temp_dir().join(format!("imod-rs-b3dgfx-snapshot-{}", std::process::id()));
        let mut state = B3dGfxState {
            snap_directory: directory.clone(),
            ..Default::default()
        };
        let mut gl = Gl::default();
        let mut encoder = Encoder::default();
        let mut ui = SnapshotUi::default();
        let auto = state
            .b3d_auto_snapshot(
                &mut gl,
                &mut encoder,
                &mut ui,
                "auto",
                SNAPSHOT_RGB,
                None,
                false,
                "PNG",
                72,
                0,
                90,
            )
            .unwrap();
        assert_eq!(auto, directory.join("auto000.PNG"));
        let key = state
            .b3d_key_snapshot(
                &mut gl,
                &mut encoder,
                &mut ui,
                "key",
                false,
                false,
                None,
                true,
                "PNG",
                72,
                0,
                90,
            )
            .unwrap();
        assert_eq!(key, directory.join("key000.tif"));
        let mut named = PathBuf::new();
        state
            .b3d_named_snapshot(
                &mut gl,
                &mut encoder,
                &mut ui,
                &mut named,
                "modv",
                SNAPSHOT_RGB,
                None,
                false,
                "PNG",
                72,
                0,
                90,
            )
            .unwrap();
        assert_eq!(named, directory.join("modv0000.PNG"));
        assert_eq!(encoder.files, vec![auto, key, named]);
        assert_eq!(
            ui.calls,
            [
                "out:auto: Saving image to ".to_owned()
                    + &b3d_short_snap_name(&directory.join("auto000.PNG"))
                    + "\n",
                "out:DONE!\n".to_owned(),
                "out:key: Saving image to ".to_owned()
                    + &b3d_short_snap_name(&directory.join("key000.tif"))
                    + "\n",
                "out:DONE!\n".to_owned(),
                "err:Saved image to ".to_owned()
                    + &b3d_short_snap_name(&directory.join("modv0000.PNG"))
                    + "\n",
            ]
        );
    }

    #[test]
    fn control_key_temporarily_selects_the_second_snapshot_format() {
        let mut state = B3dGfxState::default();
        let mut gl = Gl::default();
        let mut encoder = Encoder::default();
        let mut ui = SnapshotUi::default();
        let _ = state.b3d_key_snapshot(
            &mut gl,
            &mut encoder,
            &mut ui,
            "key",
            true,
            true,
            None,
            true,
            "PNG",
            72,
            0,
            90,
        );
        assert_eq!(ui.calls[0], "set-second");
        assert!(ui.calls.iter().any(|call| call == "restore"));
        assert_eq!(ui.calls.last(), Some(&"restore".to_owned()));
    }
}
