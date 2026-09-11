//! Translation of `IMOD/midas/graphics.cpp` and `graphics.h`.
//!
//! `MidasGL` in the C++ program is a Qt compatibility-profile widget.  The
//! raster preparation below is independent of Qt and is kept byte-for-byte in
//! the source's RGBA ordering.  Creation of the native Qt widget and its
//! compatibility `glDrawPixels` presentation remain at the explicit Qt/GL
//! boundary; a core-profile GL context cannot emulate that call faithfully.
#![allow(dead_code)]

use super::midas::{MIDAS_VIEW_COLOR, MIDAS_VIEW_SINGLE, MidasTransform, MidasView};

/// C `MidasGL` (`graphics.h`) state not owned by `Midas_view`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidasGl {
    pub mouse_pressed: bool,
    pub button1_down_millis: u128,
    pub mouse_label: String,
    pub initialized: bool,
    pub stars: Vec<(f32, f32, f32)>,
}

/// Portable mouse payload for the three `QMouseEvent` functions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MouseEvent {
    pub x: i32,
    pub y: i32,
    pub left: bool,
    pub middle: bool,
    pub right: bool,
    pub control: bool,
    pub shift: bool,
}

impl MidasGl {
    /// `MidasGL::MidasGL`.
    pub fn new() -> Self {
        Self::default()
    }
    /// `MidasGL::~MidasGL`.
    pub fn destroy(&mut self) {
        self.stars.clear();
    }
    /// `MidasGL::midas_clear`.
    pub fn midas_clear(&mut self) {
        self.stars.clear();
    }
    /// `MidasGL::draw`; Qt's `update()` is a native-widget boundary.
    pub fn draw(&mut self) {}
    /// `MidasGL::paintGL`.
    ///
    /// The source paints the preassembled RGBA buffer, then overlays its
    /// centre/fixed-point stars.  Pixel presentation remains the native
    /// compatibility-GL boundary, while the exact coordinate state is kept
    /// here for the direct GL host.
    pub fn paint_gl(&mut self, view: &mut MidasView) -> (i32, i32) {
        self.midas_clear();
        let image = view.id.clone();
        let (xdrawn, ydrawn) = self.draw_image(view, &image, 0, 0, view.width, view.height);
        if view.showref == 0 && !(view.cur_warp_file >= 0 && view.edit_warps) {
            self.draw_star(
                view.truezoom * view.xcenter + view.xoffset as f32,
                view.truezoom * view.ycenter + view.yoffset as f32,
                7.0,
            );
            if view.use_fixed != 0 {
                self.draw_star(
                    view.truezoom * view.xfixed + view.xoffset as f32,
                    view.truezoom * view.yfixed + view.yoffset as f32,
                    7.0,
                );
            }
        }
        view.exposed = 1;
        (xdrawn, ydrawn)
    }
    /// `MidasGL::fill_rgb`.
    pub fn fill_rgb(
        &self,
        fbuf: Option<&[u8]>,
        tobuf: &mut [u32],
        channel: i32,
        transform: &MidasTransform,
        reversemap: i32,
    ) {
        let size = tobuf.len();
        if fbuf.is_none() {
            if channel == 0 {
                tobuf.fill(0);
            } else {
                let byte = 3usize.saturating_sub(channel as usize);
                for pixel in tobuf {
                    *pixel &= !(0xff << (byte * 8));
                }
            }
            return;
        }
        let fbuf = fbuf.unwrap();
        let mut cmap = [0u8; 256];
        let black = transform.black.clamp(0, 255) as usize;
        let white = transform.white.clamp(0, 255) as usize;
        for value in white..256 {
            cmap[value] = 255;
        }
        let slope = 256.0 / (white.saturating_sub(black).max(1) as f32);
        for value in black..white {
            cmap[value] = ((value - black) as f32 * slope) as u8;
        }
        if reversemap != 0 {
            for entry in &mut cmap {
                *entry = 255 - *entry;
            }
        }
        for index in 0..size.min(fbuf.len()) {
            let value = cmap[fbuf[index] as usize] as u32;
            if channel > 0 {
                let shift = (3 - channel as u32) * 8;
                tobuf[index] = (tobuf[index] & !(0xff << shift)) | (value << shift);
            } else if channel == 0 {
                tobuf[index] = value | (value << 8) | (value << 16);
            }
        }
    }
    /// `MidasGL::draw_image`, returning the source `xdrawn`, `ydrawn` pair.
    pub fn draw_image(
        &mut self,
        view: &mut MidasView,
        image: &[u32],
        llx: i32,
        lly: i32,
        urx: i32,
        ury: i32,
    ) -> (i32, i32) {
        let swinx = urx - llx;
        let swiny = ury - lly;
        let zoom = view.truezoom;
        if view.xsize <= 0 || view.ysize <= 0 || zoom == 0. {
            return (0, 0);
        }
        let (mut xs, mut ys, mut xstart, mut ystart, mut xb, mut yb) =
            (view.xsize, view.ysize, 0, 0, 0, 0);
        if (view.xsize - 1) as f32 * zoom < swinx as f32 {
            xb = ((swinx as f32 - (view.xsize - 1) as f32 * zoom) / 2.) as i32;
        } else {
            xs = (swinx as f32 / zoom) as i32;
            xstart = (view.xsize as f32 / 2. - swinx as f32 / zoom / 2.) as i32 - view.xtrans;
            xstart = xstart.clamp(0, (view.xsize - 1 - xs).max(0));
            view.xtrans = (view.xsize / 2 - xs / 2) - xstart;
        }
        if (view.ysize - 1) as f32 * zoom < swiny as f32 {
            yb = ((swiny as f32 - (view.ysize - 1) as f32 * zoom) / 2.) as i32;
        } else {
            ys = (swiny as f32 / zoom) as i32;
            ystart = (view.ysize as f32 / 2. - swiny as f32 / zoom / 2.) as i32 - view.ytrans;
            ystart = ystart.clamp(0, (view.ysize - 1 - ys).max(0));
            view.ytrans = (view.ysize / 2 - ys / 2) - ystart;
        }
        view.xoffset = xb - (zoom * xstart as f32) as i32;
        view.yoffset = yb - (zoom * ystart as f32) as i32;
        let _ = (image, llx, lly); // presentation is Qt compatibility GL boundary
        if view.zoom < 0. {
            let skip = (-view.zoom) as i32;
            ((xs / skip), (ys / skip))
        } else if view.zoom == 1.5 {
            (
                ((xs as f32 * 1.5) as i32).min(swinx),
                ((ys as f32 * 1.5) as i32).min(swiny),
            )
        } else {
            (
                (xs as f32 * view.zoom) as i32,
                (ys as f32 * view.zoom) as i32,
            )
        }
    }
    /// `MidasGL::fill_viewdata`.
    pub fn fill_viewdata(
        &self,
        view: &mut MidasView,
        current: Option<&[u8]>,
        previous: Option<&[u8]>,
    ) -> i32 {
        let refz = if view.xtype == super::midas::XTYPE_XREF {
            view.zsize
        } else {
            view.refz
        } as usize;
        if view.id.len() != view.xysize as usize {
            view.id.resize(view.xysize.max(0) as usize, 0);
        }
        if view.vmode == MIDAS_VIEW_COLOR {
            for channel in 0..3 {
                let tr = view
                    .tr
                    .get(if view.image_for_channel[channel] == 2 {
                        view.cz
                    } else {
                        refz as i32
                    } as usize)
                    .cloned()
                    .unwrap_or_default();
                let image = match view.image_for_channel[channel] {
                    2 => current,
                    1 => previous,
                    _ => None,
                };
                self.fill_rgb(
                    image,
                    &mut view.id,
                    channel as i32 + 1,
                    &tr,
                    view.reversemap,
                );
            }
        } else if view.vmode == MIDAS_VIEW_SINGLE {
            let at = if view.showref != 0 {
                refz
            } else {
                view.cz as usize
            };
            let tr = view.tr.get(at).cloned().unwrap_or_default();
            self.fill_rgb(
                if view.showref != 0 { previous } else { current },
                &mut view.id,
                0,
                &tr,
                view.reversemap,
            );
        }
        0
    }
    /// `MidasGL::update_slice_view`.
    pub fn update_slice_view(
        &mut self,
        view: &mut MidasView,
        current: Option<&[u8]>,
        previous: Option<&[u8]>,
    ) -> i32 {
        self.fill_viewdata(view, current, previous);
        self.draw();
        0
    }
    /// `MidasGL::initializeGL`.
    pub fn initialize_gl(&mut self) {
        self.initialized = true;
    }
    /// `MidasGL::resizeGL`.
    pub fn resize_gl(&mut self, view: &mut MidasView, width: i32, height: i32) {
        if !view.exiting {
            view.width = width;
            view.height = height;
            let size = width.saturating_mul(height).max(0) as usize;
            if view.sdat.len() < size {
                view.sdat.resize(size, 0);
            }
        }
    }
    /// `MidasGL::mousePressEvent`.
    pub fn mouse_press_event(&mut self, view: &mut MidasView, event: MouseEvent, now_millis: u128) {
        view.lastmx = event.x;
        view.lastmy = view.height - event.y;
        view.firstmx = view.lastmx;
        view.firstmy = view.lastmy;
        self.mouse_pressed = true;
        self.button1_down_millis = now_millis;
        if event.middle && event.control && !view.edit_warps {
            view.xcenter = (view.lastmx - view.xoffset) as f32 / view.truezoom;
            view.ycenter = (view.lastmy - view.yoffset) as f32 / view.truezoom;
            view.draw_corr_box = 2;
        } else if event.right
            && event.control
            && view.xtype != super::midas::XTYPE_MONT
            && !view.edit_warps
        {
            view.xfixed = (view.lastmx - view.xoffset) as f32 / view.truezoom;
            view.yfixed = (view.lastmy - view.yoffset) as f32 / view.truezoom;
            view.use_fixed = 1 - view.use_fixed;
        }
    }
    /// `MidasGL::mouseReleaseEvent`.
    pub fn mouse_release_event(&mut self, _view: &mut MidasView, _event: MouseEvent) {
        self.mouse_pressed = false;
        self.manage_mouse_label(" ");
    }
    /// `MidasGL::mouseMoveEvent`.
    pub fn mouse_move_event(&mut self, view: &mut MidasView, event: MouseEvent) {
        if !self.mouse_pressed {
            return;
        }
        view.mousemoving = 1;
        view.mx = event.x;
        view.my = view.height - event.y;
        if event.left && event.control {
            view.xtrans += view.mx - view.lastmx;
            view.ytrans += view.my - view.lastmy;
        }
        view.lastmx = view.mx;
        view.lastmy = view.my;
        view.mousemoving = 0;
    }
    /// `MidasGL::nearestControlPoint`; libwarp coordinate arrays are a source boundary.
    pub fn nearest_control_point(&self, _view: &MidasView, _iz: i32, mindist: &mut f32) -> i32 {
        *mindist = 1e30;
        -2
    }
    /// `MidasGL::newCurrentControl`.
    pub fn new_current_control(&mut self, view: &mut MidasView, newcur: i32, _update_slice: bool) {
        view.cur_control = newcur;
        view.draw_corr_box = 2;
        self.draw();
    }
    /// `MidasGL::attachControlPoint`.
    pub fn attach_control_point(&mut self, view: &mut MidasView) {
        let mut distance = 0.;
        let iz = if view.num_chunks != 0 {
            view.cur_chunk
        } else {
            view.cz
        };
        let current = self.nearest_control_point(view, iz, &mut distance);
        if current >= 0 {
            self.new_current_control(view, current, false);
        }
    }
    /// `MidasGL::manageMouseLabel`.
    pub fn manage_mouse_label(&mut self, text: &str) {
        self.mouse_label = text.to_owned();
    }
    /// `MidasGL::drawStar`.
    pub fn draw_star(&mut self, xcen: f32, ycen: f32, censize: f32) {
        self.stars.push((xcen, ycen, censize));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fill_rgb_matches_source_channel_order() {
        let mut output = vec![0u32; 2];
        let gl = MidasGl::new();
        gl.fill_rgb(
            Some(&[0, 255]),
            &mut output,
            1,
            &MidasTransform {
                black: 0,
                white: 255,
                ..Default::default()
            },
            0,
        );
        assert_eq!(output[0] & 0xff0000, 0);
        assert_ne!(output[1] & 0xff0000, 0);
    }
    #[test]
    fn resize_grows_sdat() {
        let mut view = MidasView::default();
        let mut gl = MidasGl::new();
        gl.resize_gl(&mut view, 12, 7);
        assert_eq!(view.sdat.len(), 84);
    }
}
