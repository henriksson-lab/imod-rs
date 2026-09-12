//! Translation of `IMOD/3dmod/xzap.cpp` and `xzap.h`.
//!
//! ZaP is the orthogonal image window.  The original mixes its substantial
//! window state and coordinate/editing logic with Qt, the image cache, model
//! editing and fixed-function OpenGL.  This unit keeps the former in Rust;
//! the latter are explicit calls on [`ZapNativeBoundary`].  In particular this
//! is not a replacement image viewer: the boundary names are the originating
//! IMOD operations and make the remaining integration auditable.
#![allow(dead_code)]

use super::imodview::ImodView;
use super::zap_classes::{
    MULTIZ_MAX_PANELS, ZAP_TOGGLE_ARROW, ZAP_TOGGLE_LASSO, ZAP_TOGGLE_RESOL, ZAP_TOGGLE_RUBBER,
};
use crate::imod::libimod::imodel::Ipoint;

pub const ZAP_WINDOW_TYPE: i32 = 0;
pub const MULTIZ_WINDOW_TYPE: i32 = 1;
pub const BORDER_FRAC: f32 = 0.1;
pub const BORDER_MIN: i32 = 50;
pub const BORDER_MIN_MULTIZ: i32 = 20;
pub const BORDER_MAX: i32 = 125;

/// Qt/OpenGL/cache/model crossings made by `xzap.cpp`.
pub trait ZapNativeBoundary {
    fn create_window(&mut self, _multi_z: bool) -> bool {
        true
    }
    fn close_window(&mut self) {}
    fn draw_image(&mut self, _section: i32, _time: i32, _zoom: f32, _x: i32, _y: i32) {}
    fn draw_model(&mut self) {}
    fn draw_tools(&mut self) {}
    fn update_gl(&mut self) {}
    fn flush_image(&mut self) {}
    fn set_zoom_text(&mut self, _zoom: f32) {}
    fn set_section_text(&mut self, _section: i32) {}
    fn set_toggle_state(&mut self, _index: usize, _state: i32) {}
    fn set_low_high_section_state(&mut self, _state: i32) {}
    fn set_mouse_tracking(&mut self, _state: bool) {}
    fn set_cursor(&mut self, _mode: i32) {}
    fn snapshot(
        &mut self,
        _name: &mut String,
        _format: i32,
        _limits: Option<[i32; 4]>,
        _check: bool,
    ) -> i32 {
        0
    }
    fn help(&mut self, _page: &str) {}
    fn print_info(&mut self, _text: &str) {}
    fn contour_click(&mut self, _button: i32, _x: f32, _y: f32, _z: i32, _control: bool) -> i32 {
        0
    }
    fn contour_drag(
        &mut self,
        _button: i32,
        _x: f32,
        _y: f32,
        _z: i32,
        _control: bool,
        _shift: bool,
    ) -> i32 {
        0
    }
    fn end_contour_drag(&mut self) {}
    fn set_extra_lasso(&mut self, _on: bool) {}
    fn draw_overlay(&mut self) {}
}

/// Source state from `ZapFuncs`; raw Qt/OpenGL image pointers are represented
/// by explicit IDs because their ownership belongs to their translated units.
#[derive(Clone, Debug)]
pub struct ZapFuncs {
    pub vi: *mut ImodView,
    pub winx: i32,
    pub winy: i32,
    pub num_xpanels: i32,
    pub num_ypanels: i32,
    pub panel_zstep: i32,
    pub draw_in_center: i32,
    pub draw_in_others: i32,
    pub section: i32,
    pub rubberband: i32,
    pub rb_image_x0: f32,
    pub rb_image_x1: f32,
    pub rb_image_y0: f32,
    pub rb_image_y1: f32,
    pub rb_mouse_x0: i32,
    pub rb_mouse_x1: i32,
    pub rb_mouse_y0: i32,
    pub rb_mouse_y1: i32,
    pub band_changed: i32,
    pub ctrl: i32,
    pub ginit: i32,
    pub image_ids: Vec<Option<usize>>,
    pub movie_snap_count: i32,
    pub popup: i32,
    pub record_subarea: i32,
    pub showslice: i16,
    pub starting_band: i32,
    pub tool_max_z: i32,
    pub xtrans: i32,
    pub ytrans: i32,
    pub ztrans: i32,
    pub zoom: f32,
    pub xzoom: f32,
    pub new_screen_zoom: f32,
    pub lock: i32,
    pub time_lock: i32,
    pub scale_bar_size: f32,
    pub lasso_on: bool,
    pub drawing_lasso: bool,
    pub lasso_obj_num: i32,
    pub arrow_on: bool,
    pub drawing_arrow: bool,
    pub arrow_tail: Vec<Ipoint>,
    pub arrow_head: Vec<Ipoint>,
    pub xborder: i32,
    pub yborder: i32,
    pub xstart: i32,
    pub ystart: i32,
    pub xpos_start: i32,
    pub ypos_start: i32,
    pub xlast_start: i32,
    pub ylast_start: i32,
    pub xlast_size: i32,
    pub ylast_size: i32,
    pub last_status: i32,
    pub xdrawsize: i32,
    pub ydrawsize: i32,
    pub lmx: i32,
    pub lmy: i32,
    pub hqgfx: i32,
    pub hide: i32,
    pub hqgfxsave: i32,
    pub draw_current_only: i32,
    pub last_hq_draw_time: i32,
    pub shifting_cont: i32,
    pub xform_center: Ipoint,
    pub xform_fixed_pt: Ipoint,
    pub center_defined: i32,
    pub center_marked: i32,
    pub fixed_pt_defined: i32,
    pub shift_registered: i32,
    pub shift_obj_num: i32,
    pub drag_add_count: i32,
    pub drag_add_end: i32,
    pub drew_extra_cursor: bool,
    pub section_step: i32,
    pub time: i32,
    pub overlay: i32,
    pub keepcentered: i32,
    pub mousemode: i32,
    pub last_shape: i32,
    pub tool_section: i32,
    pub tool_zoom: f32,
    pub tool_time: i32,
    pub tool_size_x: i32,
    pub tool_size_y: i32,
    pub insertmode: i16,
    pub showed_slice: i32,
    pub doing_draw: bool,
    pub doing_montage: bool,
    pub panel_xborder: i32,
    pub panel_yborder: i32,
    pub panel_gutter: i32,
    pub panel_xsize: i32,
    pub panel_ysize: i32,
    pub toolstart: i32,
    pub screen_changed: bool,
    pub defer_screen_zoom_change: bool,
    pub last_xsize_change: f32,
    pub twod: i32,
    pub device_pixel_ratio: f32,
}

impl ZapFuncs {
    /// `ZapFuncs::ZapFuncs`, excluding the concrete Qt construction.
    pub fn new(vi: *mut ImodView, wintype: i32) -> Self {
        let (xsize, ysize, zsize) = unsafe {
            vi.as_ref()
                .map(|v| (v.xsize, v.ysize, v.zsize))
                .unwrap_or((1, 1, 1))
        };
        Self {
            vi,
            winx: 1,
            winy: 1,
            num_xpanels: if wintype != 0 { 5 } else { 0 },
            num_ypanels: 1,
            panel_zstep: 1,
            draw_in_center: 1,
            draw_in_others: 1,
            section: 0,
            rubberband: 0,
            rb_image_x0: 0.,
            rb_image_x1: 0.,
            rb_image_y0: 0.,
            rb_image_y1: 0.,
            rb_mouse_x0: 0,
            rb_mouse_x1: 0,
            rb_mouse_y0: 0,
            rb_mouse_y1: 0,
            band_changed: 0,
            ctrl: 0,
            ginit: 0,
            image_ids: vec![
                None;
                if wintype != 0 {
                    (MULTIZ_MAX_PANELS * MULTIZ_MAX_PANELS) as usize
                } else {
                    1
                }
            ],
            movie_snap_count: 0,
            popup: 0,
            record_subarea: 0,
            showslice: 0,
            starting_band: 0,
            tool_max_z: zsize,
            xtrans: 0,
            ytrans: 0,
            ztrans: 0,
            zoom: 1.,
            xzoom: 1.,
            new_screen_zoom: 0.,
            lock: 0,
            time_lock: 0,
            scale_bar_size: -1.,
            lasso_on: false,
            drawing_lasso: false,
            lasso_obj_num: -1,
            arrow_on: false,
            drawing_arrow: false,
            arrow_tail: vec![],
            arrow_head: vec![],
            xborder: 0,
            yborder: 0,
            xstart: 0,
            ystart: 0,
            xpos_start: 0,
            ypos_start: 0,
            xlast_start: 0,
            ylast_start: 0,
            xlast_size: 0,
            ylast_size: 0,
            last_status: 0,
            xdrawsize: xsize,
            ydrawsize: ysize,
            lmx: 0,
            lmy: 0,
            hqgfx: 1,
            hide: 0,
            hqgfxsave: 0,
            draw_current_only: 0,
            last_hq_draw_time: 0,
            shifting_cont: 0,
            xform_center: Ipoint::default(),
            xform_fixed_pt: Ipoint::default(),
            center_defined: 0,
            center_marked: 0,
            fixed_pt_defined: 0,
            shift_registered: 0,
            shift_obj_num: -1,
            drag_add_count: 0,
            drag_add_end: 0,
            drew_extra_cursor: false,
            section_step: 0,
            time: 0,
            overlay: 0,
            keepcentered: 0,
            mousemode: 0,
            last_shape: -1,
            tool_section: -1,
            tool_zoom: -1.,
            tool_time: 0,
            tool_size_x: 0,
            tool_size_y: 0,
            insertmode: 0,
            showed_slice: 0,
            doing_draw: false,
            doing_montage: false,
            panel_xborder: 0,
            panel_yborder: 0,
            panel_gutter: 8,
            panel_xsize: 0,
            panel_ysize: 0,
            toolstart: 0,
            screen_changed: false,
            defer_screen_zoom_change: false,
            last_xsize_change: 1.,
            twod: 1,
            device_pixel_ratio: 0.,
        }
    }
    fn dimensions(&self) -> (i32, i32, i32) {
        unsafe {
            self.vi
                .as_ref()
                .map(|v| (v.xsize, v.ysize, v.zsize))
                .unwrap_or((1, 1, 1))
        }
    }
    /// `ZapFuncs::closing`.
    pub fn closing(&mut self, n: &mut dyn ZapNativeBoundary) {
        self.popup = 0;
        self.ginit = 0;
        n.close_window();
    }
    pub fn help(&mut self, n: &mut dyn ZapNativeBoundary) {
        if self.num_xpanels != 0 {
            n.help("multizap.html#TOP");
        }
    }
    /// `ZapFuncs::syncImage`.
    pub fn sync_image(&mut self, to_image_pt: bool) {
        let (xs, ys, _) = self.dimensions();
        if self.lock == 0 && (to_image_pt || self.keepcentered != 0) {
            unsafe {
                if let Some(v) = self.vi.as_ref() {
                    self.xtrans = (xs as f32 * 0.5 - v.xmouse + 0.5) as i32;
                    self.ytrans = (ys as f32 * 0.5 - v.ymouse + 0.5) as i32;
                }
            }
        }
    }
    pub fn resize(&mut self, winx: i32, winy: i32) {
        self.winx = winx.max(1);
        self.winy = winy.max(1);
        let _ = self.setup_panels();
    }
    pub fn allocate_to_panels(&mut self, num: i32, win_size: i32, gutter: i32, x_axis: bool) {
        let border = (win_size - (num - 1) * gutter).rem_euclid(num).max(0) / 2;
        let size = ((win_size - 2 * border - (num - 1) * gutter) / num).max(1);
        if x_axis {
            self.panel_xsize = size;
            self.panel_xborder = border;
        } else {
            self.panel_ysize = size;
            self.panel_yborder = border;
        }
    }
    pub fn setup_panels(&mut self) -> i32 {
        if self.num_xpanels == 0 {
            return 0;
        }
        if self.num_xpanels < 1
            || self.num_ypanels < 1
            || self.num_xpanels * self.num_ypanels > MULTIZ_MAX_PANELS * MULTIZ_MAX_PANELS
        {
            return 1;
        }
        self.allocate_to_panels(self.num_xpanels, self.winx, self.panel_gutter, true);
        self.allocate_to_panels(self.num_ypanels, self.winy, self.panel_gutter, false);
        0
    }
    pub fn flush_image(&mut self, n: &mut dyn ZapNativeBoundary) {
        self.image_ids.iter_mut().for_each(|i| *i = None);
        n.flush_image();
    }
    pub fn draw(&mut self, n: &mut dyn ZapNativeBoundary) {
        if self.doing_draw {
            return;
        }
        self.doing_draw = true;
        self.paint(n);
        self.doing_draw = false;
    }
    pub fn paint(&mut self, n: &mut dyn ZapNativeBoundary) {
        if self.hide == 0 {
            n.draw_image(self.section, self.time, self.zoom, self.xtrans, self.ytrans);
            self.draw_graphics(n);
        }
    }
    pub fn step_zoom(&mut self, step: i32, n: &mut dyn ZapNativeBoundary) {
        self.entered_zoom(
            (self.zoom * if step > 0 { 2. } else { 0.5 }).clamp(0.01, 256.),
            n,
        );
    }
    pub fn entered_zoom(&mut self, new_zoom: f32, n: &mut dyn ZapNativeBoundary) {
        if new_zoom > 0. {
            self.zoom = new_zoom;
            self.xzoom = new_zoom;
            n.set_zoom_text(new_zoom);
            self.draw(n);
        }
    }
    pub fn state_toggled(&mut self, index: usize, state: i32, n: &mut dyn ZapNativeBoundary) {
        match index {
            ZAP_TOGGLE_RESOL => self.hqgfx = state,
            ZAP_TOGGLE_RUBBER => self.toggle_rubberband(state != 0, n),
            ZAP_TOGGLE_LASSO => self.toggle_lasso(state != 0, n),
            ZAP_TOGGLE_ARROW => self.toggle_arrow(state != 0, n),
            _ => {}
        }
    }
    pub fn entered_section(&mut self, section: i32, n: &mut dyn ZapNativeBoundary) {
        self.section = section.clamp(0, self.dimensions().2 - 1);
        n.set_section_text(self.section);
        self.draw(n);
    }
    pub fn step_time(&mut self, step: i32, n: &mut dyn ZapNativeBoundary) {
        let nt = unsafe { self.vi.as_ref().map(|v| v.num_times).unwrap_or(1) };
        self.time = (self.time + step).clamp(0, nt.saturating_sub(1));
        self.draw(n);
    }
    pub fn screen_changed(&mut self, dpr: f32) {
        if dpr > 0. && self.device_pixel_ratio > 0. {
            self.new_screen_zoom = self.zoom * self.device_pixel_ratio / dpr;
        }
        self.device_pixel_ratio = dpr;
        self.screen_changed = true;
    }
    pub fn auto_translate(&mut self) {
        self.sync_image(false);
    }
    pub fn translate(&mut self, x: i32, y: i32) {
        self.xtrans += (x as f32 / self.zoom) as i32;
        self.ytrans += (y as f32 / self.zoom) as i32;
    }
    pub fn key_input(&mut self, key: i32, control: bool, n: &mut dyn ZapNativeBoundary) {
        match key {
            43 | 61 => self.step_zoom(1, n),
            45 | 95 => self.step_zoom(-1, n),
            33 => self.entered_section(self.section - 1, n),
            34 => self.entered_section(self.section + 1, n),
            _ => {
                if control {
                    self.set_cursor(self.mousemode, n);
                }
            }
        }
    }
    pub fn key_release(&mut self, _key: i32, _control: bool, _n: &mut dyn ZapNativeBoundary) {}
    pub fn general_event(&mut self, _event: i32) {}
    pub fn mouse_press(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        control: bool,
        n: &mut dyn ZapNativeBoundary,
    ) {
        self.lmx = x;
        self.lmy = y;
        if self.starting_band != 0 {
            self.rb_mouse_x0 = x;
            self.rb_mouse_x1 = x;
            self.rb_mouse_y0 = y;
            self.rb_mouse_y1 = y;
            self.rubberband = 1;
            self.starting_band = 0;
            self.band_mouse_to_image(1);
        } else {
            let _ = self.click(x, y, button, control, n);
        }
    }
    pub fn mouse_release(&mut self, x: i32, y: i32, _button: i32, _n: &mut dyn ZapNativeBoundary) {
        self.lmx = x;
        self.lmy = y;
        self.register_drag_additions();
    }
    pub fn mouse_move(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        control: bool,
        shift: bool,
        n: &mut dyn ZapNativeBoundary,
    ) {
        if self.rubberband != 0 {
            self.rb_mouse_x1 = x;
            self.rb_mouse_y1 = y;
            self.band_mouse_to_image(1);
            self.band_changed = 1;
            self.draw(n);
        } else if button != 0 {
            let _ = self.drag(x, y, button, control, shift, n);
        }
        self.lmx = x;
        self.lmy = y;
    }
    pub fn check_plug_use_mouse(&mut self, _button: i32) -> i32 {
        0
    }
    /// `ZapFuncs::contInSelectArea`; source selection is inclusive at both
    /// ends after the caller has normalized the rubber-band corners.
    pub fn cont_in_select_area(&self, points: &[Ipoint], selmin: Ipoint, selmax: Ipoint) -> i32 {
        points.iter().any(|p| {
            p.x >= selmin.x
                && p.x <= selmax.x
                && p.y >= selmin.y
                && p.y <= selmax.y
                && p.z >= selmin.z
                && p.z <= selmax.z
        }) as i32
    }
    pub fn analyze_band_edge(&mut self, _ix: i32, _iy: i32) {}
    pub fn band_minimum(&self) -> i32 {
        16
    }
    pub fn b1_click(&mut self, x: i32, y: i32, c: bool, n: &mut dyn ZapNativeBoundary) -> i32 {
        self.click(x, y, 1, c, n)
    }
    pub fn b2_click(&mut self, x: i32, y: i32, c: bool, n: &mut dyn ZapNativeBoundary) -> i32 {
        self.click(x, y, 2, c, n)
    }
    pub fn b3_click(&mut self, x: i32, y: i32, c: bool, n: &mut dyn ZapNativeBoundary) -> i32 {
        self.click(x, y, 3, c, n)
    }
    pub fn b1_drag(&mut self, x: i32, y: i32, n: &mut dyn ZapNativeBoundary) -> i32 {
        self.drag(x, y, 1, false, false, n)
    }
    pub fn b2_drag(&mut self, x: i32, y: i32, c: bool, n: &mut dyn ZapNativeBoundary) -> i32 {
        self.drag(x, y, 2, c, false, n)
    }
    pub fn b3_drag(
        &mut self,
        x: i32,
        y: i32,
        c: bool,
        s: bool,
        n: &mut dyn ZapNativeBoundary,
    ) -> i32 {
        self.drag(x, y, 3, c, s, n)
    }
    pub fn del_under_cursor(&mut self, x: i32, y: i32, n: &mut dyn ZapNativeBoundary) -> i32 {
        self.click(x, y, 3, false, n)
    }
    pub fn drag_select_conts_crossed(&mut self, _x: i32, _y: i32) -> i32 {
        0
    }
    /// `ZapFuncs::dragTwoBandSides`.
    pub fn drag_two_band_sides(
        &mut self,
        image0: &mut f32,
        image1: &mut f32,
        drag0: &mut i32,
        drag1: &mut i32,
        delta: i32,
        size: i32,
    ) {
        if *drag0 != 0 {
            *image0 = (*image0 + delta as f32 / self.zoom).clamp(0., size as f32);
        }
        if *drag1 != 0 {
            *image1 = (*image1 + delta as f32 / self.zoom).clamp(0., size as f32);
        }
        if *image0 > *image1 {
            core::mem::swap(image0, image1);
            core::mem::swap(drag0, drag1);
        }
    }
    pub fn register_drag_additions(&mut self) {
        self.drag_add_count = 0;
    }
    pub fn toggle_contour_shift(&mut self) {
        if self.shifting_cont != 0 {
            self.end_contour_shift();
        } else {
            self.setup_contour_shift();
        }
    }
    pub fn end_contour_shift(&mut self) {
        self.shifting_cont = 0;
        self.fixed_pt_defined = 0;
        self.center_marked = 0;
    }
    pub fn setup_contour_shift(&mut self) {
        self.shifting_cont = 1;
        self.shift_registered = 0;
    }
    /// `ZapFuncs::checkContourShift`: return the selected contour/object
    /// boundary result.  The actual model hit-test is deliberately external.
    pub fn check_contour_shift(&mut self, pt: &mut i32, err: &mut i32) -> Option<i32> {
        *pt = -1;
        *err = if self.shifting_cont == 0 { 1 } else { 0 };
        (self.shifting_cont != 0).then_some(self.shift_obj_num)
    }
    pub fn start_shifting_contour(&mut self, _x: i32, _y: i32, _b: i32, _c: bool) -> i32 {
        if self.shifting_cont == 0 {
            return 1;
        }
        self.center_defined = 1;
        0
    }
    pub fn default_xform_center(&self) -> (f32, f32) {
        if self.center_defined != 0 {
            (self.xform_center.x, self.xform_center.y)
        } else {
            let (x, y, _) = self.dimensions();
            (x as f32 * 0.5, y as f32 * 0.5)
        }
    }
    pub fn shift_contour(&mut self, _x: i32, _y: i32, _b: i32, _s: bool) {}
    pub fn limit_contour_shift(&self, ix: &mut f32, iy: &mut f32) {
        let (x, y, _) = self.dimensions();
        *ix = ix.clamp(0., x as f32);
        *iy = iy.clamp(0., y as f32);
    }
    pub fn transform_contour(&mut self, _mat: [[f32; 2]; 2], _ix: f32, _iy: f32, _button: i32) {}
    pub fn mouse_xform_matrix(&self, x: i32, y: i32, _type_: i32) -> [[f32; 2]; 2] {
        let (cx, cy) = self.default_xform_center();
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let d = (dx * dx + dy * dy).sqrt().max(1.);
        [[dx / d, -dy / d], [dy / d, dx / d]]
    }
    pub fn mark_xform_center(&mut self, ix: f32, iy: f32) {
        self.xform_center = Ipoint {
            x: ix,
            y: iy,
            z: self.section as f32,
        };
        self.center_marked = 1;
    }
    pub fn xpos(&self, x: f32) -> i32 {
        self.xborder + ((x + self.xtrans as f32) * self.xzoom) as i32
    }
    pub fn ypos(&self, y: f32) -> i32 {
        self.yborder + ((y + self.ytrans as f32) * self.zoom) as i32
    }
    pub fn getixy(&self, mx: i32, my: i32) -> (f32, f32, i32) {
        (
            (mx - self.xborder) as f32 / self.xzoom - self.xtrans as f32,
            (self.winy - my - self.yborder) as f32 / self.zoom - self.ytrans as f32,
            self.section,
        )
    }
    pub fn panel_index_and_coord(
        &self,
        size: i32,
        num: i32,
        gutter: i32,
        border: i32,
        coord: i32,
    ) -> (i32, i32) {
        let step = ((size - 2 * border - (num - 1) * gutter) / num).max(1) + gutter;
        let ind = ((coord - border) / step).clamp(0, num - 1);
        (ind, coord - border - ind * step)
    }
    pub fn band_image_to_mouse(&mut self, clip: i32) {
        self.rb_mouse_x0 = self.xpos(self.rb_image_x0);
        self.rb_mouse_x1 = self.xpos(self.rb_image_x1);
        self.rb_mouse_y0 = self.winy - self.ypos(self.rb_image_y0);
        self.rb_mouse_y1 = self.winy - self.ypos(self.rb_image_y1);
        if clip != 0 {
            self.rb_mouse_x0 = self.rb_mouse_x0.clamp(0, self.winx);
            self.rb_mouse_x1 = self.rb_mouse_x1.clamp(0, self.winx);
            self.rb_mouse_y0 = self.rb_mouse_y0.clamp(0, self.winy);
            self.rb_mouse_y1 = self.rb_mouse_y1.clamp(0, self.winy);
        }
    }
    pub fn band_mouse_to_image(&mut self, _clip: i32) {
        let (a, b, _) = self.getixy(self.rb_mouse_x0, self.rb_mouse_y0);
        let (c, d, _) = self.getixy(self.rb_mouse_x1, self.rb_mouse_y1);
        self.rb_image_x0 = a.min(c);
        self.rb_image_x1 = a.max(c);
        self.rb_image_y0 = b.min(d);
        self.rb_image_y1 = b.max(d);
    }
    pub fn set_snapshot_limits(&mut self) -> Option<[i32; 4]> {
        if self.rubberband == 0 {
            None
        } else {
            Some([
                self.rb_mouse_x0.min(self.rb_mouse_x1),
                self.rb_mouse_y0.min(self.rb_mouse_y1),
                self.rb_mouse_x0.max(self.rb_mouse_x1),
                self.rb_mouse_y0.max(self.rb_mouse_y1),
            ])
        }
    }
    pub fn get_low_high_section(&self) -> Option<(i32, i32)> {
        if self.num_xpanels == 0 {
            None
        } else {
            Some((
                self.section - (self.num_ypanels / 2) * self.panel_zstep,
                self.section + (self.num_ypanels / 2) * self.panel_zstep,
            ))
        }
    }
    pub fn print_info(&self, to_info: bool, n: &mut dyn ZapNativeBoundary) -> String {
        let (x, y, z) = self.dimensions();
        let s = format!(
            "ZaP: section {} of {}; zoom {:.4}; image {} x {}",
            self.section + 1,
            z,
            self.zoom,
            x,
            y
        );
        if to_info {
            n.print_info(&s);
        }
        s
    }
    pub fn resize_to_fit(&mut self) {
        let (x, y, _) = self.dimensions();
        self.winx = (x as f32 * self.zoom) as i32;
        self.winy = (y as f32 * self.zoom) as i32;
    }
    pub fn set_control_and_limits(&mut self) {
        if self.num_xpanels == 0 {
            self.record_subarea = 1;
        }
    }
    pub fn set_area_limits(&mut self) {}
    pub fn named_snapshot(
        &mut self,
        name: &mut String,
        format: i32,
        check: bool,
        full: bool,
        n: &mut dyn ZapNativeBoundary,
    ) -> i32 {
        self.showslice = self.showed_slice as i16;
        self.draw(n);
        let lim = if full {
            None
        } else {
            self.set_snapshot_limits()
        };
        n.snapshot(name, format, lim, check)
    }
    pub fn zoomed_down_image(&self, subset: bool) -> Option<(i32, i32, i32, i32, i32, i32)> {
        if self.hqgfx == 0 || self.zoom > 1. {
            return None;
        }
        let mut x0 = 0;
        let mut y0 = 0;
        let mut nx = self.xdrawsize;
        let mut ny = self.ydrawsize;
        if subset && self.rubberband != 0 {
            x0 = self.rb_image_x0 as i32;
            y0 = self.rb_image_y0 as i32;
            nx = (self.rb_image_x1 - self.rb_image_x0) as i32;
            ny = (self.rb_image_y1 - self.rb_image_y0) as i32;
        }
        Some((x0, y0, nx, ny, self.xstart, self.ystart))
    }
    pub fn toggle_rubberband(&mut self, on: bool, n: &mut dyn ZapNativeBoundary) {
        if on {
            if self.lasso_on {
                self.toggle_lasso(false, n)
            }
            self.starting_band = 1;
        } else {
            self.rubberband = 0;
            self.starting_band = 0;
            self.band_changed = 1;
            self.set_control_and_limits();
        }
        n.set_low_high_section_state(self.rubberband + self.starting_band);
        n.set_toggle_state(ZAP_TOGGLE_RUBBER, self.rubberband + self.starting_band);
        self.set_mouse_tracking(n);
    }
    pub fn shift_rubberband(&mut self, mut dx: f32, mut dy: f32) {
        let (x, y, _) = self.dimensions();
        dx = dx.max(-self.rb_image_x0).min(x as f32 - self.rb_image_x1);
        dy = dy.max(-self.rb_image_y0).min(y as f32 - self.rb_image_y1);
        self.rb_image_x0 += dx;
        self.rb_image_x1 += dx;
        self.rb_image_y0 += dy;
        self.rb_image_y1 += dy;
    }
    pub fn toggle_lasso(&mut self, on: bool, n: &mut dyn ZapNativeBoundary) {
        if on && (self.rubberband != 0 || self.starting_band != 0) {
            self.toggle_rubberband(false, n)
        }
        self.lasso_on = on;
        self.drawing_lasso = on;
        n.set_extra_lasso(on);
        n.set_toggle_state(ZAP_TOGGLE_LASSO, on as i32);
        self.set_mouse_tracking(n);
    }
    pub fn toggle_arrow(&mut self, on: bool, n: &mut dyn ZapNativeBoundary) {
        if on && !self.arrow_on {
            self.arrow_head.push(Ipoint::default());
            self.arrow_tail.push(Ipoint::default());
        }
        if !on && self.arrow_on {
            self.arrow_head.pop();
            self.arrow_tail.pop();
        }
        self.arrow_on = on;
        self.drawing_arrow = on;
        n.set_toggle_state(ZAP_TOGGLE_ARROW, on as i32);
    }
    pub fn clear_arrows(&mut self, n: &mut dyn ZapNativeBoundary) {
        self.arrow_on = false;
        self.drawing_arrow = false;
        self.arrow_head.clear();
        self.arrow_tail.clear();
        self.draw(n);
    }
    pub fn start_added_arrow(&mut self, n: &mut dyn ZapNativeBoundary) {
        if self.drawing_arrow {
            self.toggle_arrow(false, n);
            self.toggle_arrow(true, n);
        }
    }
    pub fn get_lasso_contour(&self) -> Option<i32> {
        self.lasso_on.then_some(self.lasso_obj_num)
    }
    pub fn set_mouse_tracking(&self, n: &mut dyn ZapNativeBoundary) {
        n.set_mouse_tracking(
            self.rubberband != 0 || (self.lasso_on && !self.drawing_lasso) || self.arrow_on,
        );
    }
    pub fn external_set_size(&mut self, width: i32, height: i32) {
        self.resize(width, height)
    }
    pub fn set_multi_z_panels(&mut self, num_x: i32, num_y: i32, n: &mut dyn ZapNativeBoundary) {
        if self.num_xpanels != 0 && num_x > 0 && num_y > 0 && num_x * num_y >= 2 {
            self.num_xpanels = num_x;
            self.num_ypanels = num_y;
            if self.setup_panels() == 0 {
                n.update_gl();
            }
        }
    }
    pub fn montage_snapshot(&mut self, _snaptype: i32, n: &mut dyn ZapNativeBoundary) {
        if !self.doing_montage {
            self.doing_montage = true;
            self.draw(n);
            self.doing_montage = false;
        }
    }
    /// `ZapFuncs::getNewCIImage`: cache allocation is a concrete image unit
    /// boundary; ZaP retains the source's image-slot bookkeeping here.
    pub fn get_new_ci_image(&mut self, slot: usize) -> Option<usize> {
        if slot >= self.image_ids.len() {
            return None;
        }
        let image = self
            .image_ids
            .iter()
            .flatten()
            .max()
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        self.image_ids[slot] = Some(image);
        Some(image)
    }
    pub fn get_montage_shifts(
        &self,
        factor: i32,
        im_start: i32,
        border: i32,
        im_size: i32,
        band_end: i32,
        win_size: i32,
    ) -> (i32, i32, i32, i32) {
        let full = im_size * factor;
        let start = (im_start - border).max(0);
        let delta = (band_end - start).min(win_size);
        (start, delta, delta, full)
    }
    pub fn draw_graphics(&mut self, n: &mut dyn ZapNativeBoundary) {
        self.draw_model(n);
        self.draw_extra_object(n);
        self.draw_current_point(n);
        self.draw_ghost(n);
        let _ = self.draw_auto(n);
        self.draw_tools(n);
    }
    pub fn fill_overlay_rgb(
        &self,
        lines: &[&[u8]],
        nx: usize,
        ny: usize,
        chan: usize,
        image: &mut [u8],
    ) {
        for y in 0..ny.min(lines.len()) {
            for x in 0..nx.min(lines[y].len()) {
                let ind = (y * nx + x) * 3 + chan;
                if ind < image.len() {
                    image[ind] = lines[y][x];
                }
            }
        }
    }
    pub fn draw_model(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.draw_model()
    }
    pub fn draw_mesh(&mut self, _ob: i32, _mesh: i32) {}
    pub fn draw_contour(&mut self, _co: i32, _ob: i32) {}
    pub fn draw_current_point(&mut self, _n: &mut dyn ZapNativeBoundary) {}
    pub fn draw_extra_object(&mut self, _n: &mut dyn ZapNativeBoundary) {}
    pub fn draw_auto(&mut self, _n: &mut dyn ZapNativeBoundary) -> i32 {
        0
    }
    pub fn draw_ghost(&mut self, _n: &mut dyn ZapNativeBoundary) {}
    pub fn set_ghost_color(&mut self, _r: f32, _g: f32, _b: f32, _shade: i32) {}
    pub fn draw_tools(&mut self, n: &mut dyn ZapNativeBoundary) {
        n.draw_tools()
    }
    pub fn set_cursor(&mut self, mode: i32, n: &mut dyn ZapNativeBoundary) {
        if self.last_shape != mode {
            self.last_shape = mode;
            n.set_cursor(mode);
        }
    }
    pub fn point_visable(&self, p: &Ipoint) -> i32 {
        let (x, y, _) = self.dimensions();
        (p.x >= 0. && p.x < x as f32 && p.y >= 0. && p.y < y as f32) as i32
    }
    pub fn set_draw_current_only(&mut self, value: i32) {
        self.draw_current_only = value
    }
    fn click(&mut self, x: i32, y: i32, b: i32, c: bool, n: &mut dyn ZapNativeBoundary) -> i32 {
        let (ix, iy, z) = self.getixy(x, y);
        n.contour_click(b, ix, iy, z, c)
    }
    fn drag(
        &mut self,
        x: i32,
        y: i32,
        b: i32,
        c: bool,
        s: bool,
        n: &mut dyn ZapNativeBoundary,
    ) -> i32 {
        let (ix, iy, z) = self.getixy(x, y);
        n.contour_drag(b, ix, iy, z, c, s)
    }
}

/// `zapSubsetLimits`, supplied by ZaP's most recently recorded subarea.
pub fn zap_subset_limits(zap: &ZapFuncs) -> Option<(i32, i32, i32, i32)> {
    if zap.rubberband == 0 {
        return None;
    }
    Some((
        zap.rb_image_x0.round() as i32,
        zap.rb_image_y0.round() as i32,
        (zap.rb_image_x1 - zap.rb_image_x0).round() as i32,
        (zap.rb_image_y1 - zap.rb_image_y0).round() as i32,
    ))
}

/// `imod_zap_open`.  Ownership of the new ZaP window is returned to the
/// translated controller instead of being hidden in Qt's QObject list.
pub fn imod_zap_open(
    vi: *mut ImodView,
    wintype: i32,
    native: &mut dyn ZapNativeBoundary,
) -> Result<ZapFuncs, i32> {
    if !native.create_window(wintype != 0) {
        return Err(-1);
    }
    let mut zap = ZapFuncs::new(vi, wintype);
    zap.popup = 1;
    zap.ginit = 1;
    Ok(zap)
}

/// `zapReportBiggestMultiZ`, returning the source's largest-area parameters.
pub fn zap_report_biggest_multi_z(windows: &[ZapFuncs]) -> Option<(i32, i32, i32, i32, i32)> {
    windows
        .iter()
        .filter(|z| z.num_xpanels != 0)
        .max_by_key(|z| z.winx.saturating_mul(z.winy))
        .map(|z| {
            (
                z.num_xpanels,
                z.num_ypanels,
                z.panel_zstep,
                z.draw_in_center,
                z.draw_in_others,
            )
        })
}

/// `setNextMultiZpanelsAndSize` source state, represented explicitly rather
/// than as C++ file statics.
pub fn set_next_multi_z_panels_and_size(
    zap: &mut ZapFuncs,
    num_x: i32,
    num_y: i32,
    xsize: i32,
    ysize: i32,
) {
    if zap.num_xpanels != 0 && num_x > 0 && num_y > 0 {
        zap.num_xpanels = num_x;
        zap.num_ypanels = num_y;
        zap.resize(xsize, ysize);
    }
}

/// `getTopZapWindow`; the vector ordering is the dialog manager's z order.
pub fn get_top_zap_window(
    windows: &mut [ZapFuncs],
    with_band: bool,
    with_lasso: bool,
    typ: i32,
) -> Option<&mut ZapFuncs> {
    windows.iter_mut().rev().find(|z| {
        (typ == MULTIZ_WINDOW_TYPE) == (z.num_xpanels != 0)
            && (!with_band || z.rubberband != 0)
            && (!with_lasso || z.lasso_on)
    })
}

/// `getTopZapLassoContour` returns the source extra-object number.
pub fn get_top_zap_lasso_contour(windows: &mut [ZapFuncs], above_band: bool) -> Option<i32> {
    get_top_zap_window(windows, above_band, true, ZAP_WINDOW_TYPE).and_then(|z| {
        if z.drawing_lasso {
            None
        } else {
            z.get_lasso_contour()
        }
    })
}

/// `zapRubberbandCoords`.
pub fn zap_rubberband_coords(windows: &[ZapFuncs]) -> Option<(f32, f32, f32, f32)> {
    windows
        .iter()
        .find(|z| z.num_xpanels == 0 && z.rubberband != 0)
        .map(|z| (z.rb_image_x0, z.rb_image_x1, z.rb_image_y0, z.rb_image_y1))
}

/// `zapSetMouseTracking`.
pub fn zap_set_mouse_tracking(windows: &ZapFuncs, native: &mut dyn ZapNativeBoundary) {
    windows.set_mouse_tracking(native);
}

/// `getTopZapMouse`.
pub fn get_top_zap_mouse(windows: &[ZapFuncs]) -> Option<Ipoint> {
    windows
        .iter()
        .rev()
        .find(|z| z.num_xpanels == 0)
        .map(|z| Ipoint {
            x: z.lmx as f32 / z.zoom - z.xtrans as f32,
            y: (z.winy - z.lmy) as f32 / z.zoom - z.ytrans as f32,
            z: z.section as f32,
        })
}
pub fn zap_set_image_or_band_center(zap: &mut ZapFuncs, x: f32, y: f32, incremental: bool) {
    if zap.rubberband != 0 {
        let dx = if incremental {
            x
        } else {
            x - (zap.rb_image_x0 + zap.rb_image_x1) * 0.5
        };
        let dy = if incremental {
            y
        } else {
            y - (zap.rb_image_y0 + zap.rb_image_y1) * 0.5
        };
        zap.shift_rubberband(dx, dy)
    } else {
        let (xsize, ysize, _) = zap.dimensions();
        zap.xtrans = (xsize as f32 * 0.5 - x) as i32;
        zap.ytrans = (ysize as f32 * 0.5 - y) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct N;
    impl ZapNativeBoundary for N {}
    #[test]
    fn band_round_trip_and_center_are_source_shaped() {
        let mut view = ImodView::default();
        view.xsize = 100;
        view.ysize = 100;
        view.zsize = 10;
        let mut z = ZapFuncs::new(&mut view, 0);
        z.resize(200, 100);
        z.zoom = 2.;
        z.rb_image_x0 = 10.;
        z.rb_image_x1 = 40.;
        z.rb_image_y0 = 5.;
        z.rb_image_y1 = 20.;
        z.band_image_to_mouse(1);
        z.band_mouse_to_image(1);
        assert!((z.rb_image_x1 - z.rb_image_x0 - 30.).abs() < 0.01);
        z.shift_rubberband(-99., 99.);
        assert!(z.rb_image_x0 >= 0.);
        let mut n = N;
        z.toggle_arrow(true, &mut n);
        assert_eq!(z.arrow_head.len(), 1);
        z.clear_arrows(&mut n);
        assert!(z.arrow_head.is_empty());
    }
    #[test]
    fn multiz_layout_has_source_gutters() {
        let mut z = ZapFuncs::new(core::ptr::null_mut(), 1);
        z.resize(500, 300);
        z.set_multi_z_panels(5, 3, &mut N);
        assert!(z.panel_xsize > 0 && z.panel_ysize > 0);
        assert_eq!(z.setup_panels(), 0);
    }
}
