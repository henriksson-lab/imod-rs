//! Translation of `IMOD/3dmod/linegui.cpp` and `linegui.h`.
//!
//! The line-following numerical routines are Fortran entry points in the
//! upstream distribution, while the dialog, model, undo, image-cache, and
//! docking operations belong to 3dmod/Qt.  They are deliberately retained as
//! direct calls on [`LineTrackNativeBoundary`].  The source-owned plugin state,
//! parameter validation, key dispatch, track/copy sequencing, and settings
//! serialization are Rust.
#![allow(dead_code)]

use crate::imod::libimod::imodel::Ipoint;

pub const MAX_EDIT_BOXES: usize = 12;
pub const CONTOUR_POINT_MAX: usize = 1000;
pub const MAX_SETTINGS: usize = 13;
pub const IMOD_PLUG_MENU: i32 = 1;
pub const IMOD_PLUG_KEYS: i32 = 2;

/// `LineTrackModule` from `linegui.h`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LineTrackModule;

impl LineTrackModule {
    /// `LineTrackModule::LineTrackModule`.
    pub fn new() -> Self {
        Self
    }
}

/// `PlugData` from `linegui.cpp`.  Viewer/model pointers are represented by
/// the methods of the native boundary; all data owned by this compilation unit
/// remains explicit here.
#[derive(Clone, Debug, PartialEq)]
pub struct PlugData {
    pub window_open: bool,
    pub undo_cont: Option<Vec<Ipoint>>,
    pub tmp_cont: Option<Vec<Ipoint>>,
    pub ob: i32,
    pub co: i32,
    pub pt: i32,
    pub csection: i32,
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub idata: Vec<u8>,
    pub idata_xsize: i32,
    pub idata_ysize: i32,
    pub idata_sec: i32,
    pub idata_flipped: i32,
    pub cmax: usize,
    pub ksize: i32,
    pub knum: i32,
    pub sigma: f32,
    pub h: f32,
    pub ifdark: i32,
    pub stepsize: f32,
    pub redtol: f32,
    pub ifreplace: i32,
    pub offset: f32,
    pub closecont: i32,
    pub copytol: i32,
    pub docopy: i32,
    pub copypool: i32,
    pub copyfit: i32,
    pub copiedco: i32,
    pub copysize: i32,
    pub left: i32,
    pub top: i32,
}

impl Default for PlugData {
    fn default() -> Self {
        Self {
            window_open: false,
            undo_cont: None,
            tmp_cont: None,
            ob: 0,
            co: 0,
            pt: 0,
            csection: 0,
            xsize: 0,
            ysize: 0,
            zsize: 0,
            idata: Vec::new(),
            idata_xsize: 0,
            idata_ysize: 0,
            idata_sec: 0,
            idata_flipped: 0,
            cmax: 0,
            ksize: 0,
            knum: 0,
            sigma: 0.,
            h: 0.,
            ifdark: 0,
            stepsize: 0.,
            redtol: 0.,
            ifreplace: 0,
            offset: 0.,
            closecont: 0,
            copytol: 0,
            docopy: 0,
            copypool: 0,
            copyfit: 0,
            copiedco: 0,
            copysize: 0,
            left: 0,
            top: 0,
        }
    }
}

/// File-static `thisPlug`, `sTopWin`, and `first`.
#[derive(Clone, Debug, PartialEq)]
pub struct LineTrackState {
    pub this_plug: PlugData,
    pub top_window_open: bool,
    pub first: bool,
}

impl Default for LineTrackState {
    fn default() -> Self {
        Self {
            this_plug: PlugData::default(),
            top_window_open: false,
            first: true,
        }
    }
}

/// One of the stored `mIntPtr`/`mFloatPtr` bindings in `LineTrack`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineTrackValue {
    Ifdark,
    Offset,
    Sigma,
    H,
    Ksize,
    Redtol,
    Copytol,
    Copypool,
    Copyfit,
    Knum,
    Stepsize,
}

/// `ToolEdit` value and the parallel arrays populated by `makeEditRow`.
#[derive(Clone, Debug, PartialEq)]
pub struct LineTrackEditRow {
    pub label: String,
    pub value: String,
    pub value_kind: LineTrackValue,
    pub is_int: bool,
    pub min: f32,
    pub max: f32,
    pub tip: String,
}

/// Qt/3dmod and the external `linetrack_`/`conttrack_` boundaries used by the
/// exact source unit.  An implementation binds these calls to the actual
/// viewer rather than maintaining a second model representation.
pub trait LineTrackNativeBoundary {
    fn print(&mut self, text: &str);
    fn rgb_or_fake_image(&mut self) -> bool;
    fn raise_top_window(&mut self);
    fn image_size(&mut self) -> (i32, i32, i32);
    fn load_settings(&mut self, name: &str) -> Vec<f64>;
    fn create_window(&mut self, title: &str, help: &str, key: char);
    fn set_window_position(&mut self, left: i32, top: i32);
    fn show_window(&mut self);
    fn selected_contour(&mut self) -> Option<Vec<Ipoint>>;
    fn contour_index(&mut self) -> (i32, i32, i32);
    fn current_location(&mut self) -> (i32, i32, i32);
    fn current_section(&mut self, z: i32, cached: bool) -> Option<Vec<u8>>;
    fn tile_or_strip_cache(&mut self) -> bool;
    fn model_flipped(&mut self) -> i32;
    fn copy_image_to_byte_buffer(&mut self, image: &[u8], destination: &mut [u8]) -> bool;
    fn contour_addition(&mut self, contour_number: i32);
    fn object_max_contour(&mut self) -> i32;
    fn new_contour(&mut self);
    fn replace_selected_contour(&mut self, contour: Vec<Ipoint>);
    fn set_new_contour_time(&mut self);
    fn copy_contour_properties(&mut self, source: i32, destination: i32);
    fn contour_data_change(&mut self);
    fn flush_undo_unit(&mut self);
    fn finish_undo_unit(&mut self);
    fn undo(&mut self);
    fn set_index(&mut self, object: i32, contour: i32, point: i32);
    fn draw_model(&mut self);
    fn line_track(
        &mut self,
        image: &[u8],
        nx: i32,
        ny: i32,
        points: &mut Vec<Ipoint>,
        curpt: &mut i32,
        ksize: i32,
        knum: i32,
        sigma: f32,
        h: f32,
        ifdark: i32,
        stepsize: f32,
        redtol: f32,
        ifreplace: i32,
        offset: f32,
        closecont: i32,
    ) -> i32;
    fn contour_track(
        &mut self,
        image: &[u8],
        nx: i32,
        ny: i32,
        points: &mut Vec<Ipoint>,
        curpt: &mut i32,
        ksize: i32,
        knum: i32,
        sigma: f32,
        h: f32,
        ifdark: i32,
        stepsize: f32,
        redtol: f32,
        offset: f32,
        copytol: i32,
        copypool: i32,
        copyfit: i32,
    );
    fn geometry(&mut self) -> (i32, i32);
    fn save_settings(&mut self, name: &str, values: &[f64]);
    fn remove_window(&mut self);
    fn free_tile_cached_section(&mut self);
    fn accept_close(&mut self);
    fn rounded_style(&mut self);
    fn dialog_change_event(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn font_change_event(&mut self) -> bool;
    fn close_key(&mut self) -> bool;
    fn close_window(&mut self);
    fn control_key(&mut self, release: bool);
}

/// `setDefaults`.
pub fn set_defaults(plug: &mut PlugData) {
    plug.ksize = 7;
    plug.knum = 30;
    plug.sigma = 2.5;
    plug.h = 7.;
    plug.ifdark = 1;
    plug.stepsize = 3.;
    plug.redtol = 0.3;
    plug.offset = 0.;
    plug.copytol = 3;
    plug.copypool = 5;
    plug.copyfit = 5;
}

/// `imodPlugInfo`.
pub fn imod_plug_info(plug_type: Option<&mut i32>) -> &'static str {
    if let Some(plug_type) = plug_type {
        *plug_type = IMOD_PLUG_MENU + IMOD_PLUG_KEYS;
    }
    "Line Track"
}

/// `imodPlugKeys`.
pub fn imod_plug_keys(
    state: &mut LineTrackState,
    key: char,
    ctrl: bool,
    shift: bool,
    form: &mut LineTrack,
    native: &mut dyn LineTrackNativeBoundary,
) -> i32 {
    if !state.this_plug.window_open {
        return 0;
    }
    match key {
        '\'' | ' ' => {
            if key == '\'' {
                state.this_plug.docopy = state.this_plug.copytol;
            }
            if ctrl || shift {
                state.this_plug.closecont = 1;
            }
            form.track(0, state, native);
            1
        }
        ';' | 'u' | 'U' => {
            form.undo(native);
            1
        }
        _ => 0,
    }
}

/// `imodPlugExecute`.
pub fn imod_plug_execute(state: &mut LineTrackState, native: &mut dyn LineTrackNativeBoundary) {
    if native.rgb_or_fake_image() {
        native.print("\x07Line tracker will not work on RGB or blank data\n");
        return;
    }
    if state.this_plug.window_open {
        native.raise_top_window();
        return;
    }
    let (xsize, ysize, zsize) = native.image_size();
    let plug = &mut state.this_plug;
    plug.xsize = xsize;
    plug.ysize = ysize;
    plug.zsize = zsize;
    plug.idata.clear();
    plug.undo_cont = None;
    plug.tmp_cont = Some(Vec::new());
    plug.idata_xsize = 0;
    plug.idata_ysize = 0;
    plug.cmax = CONTOUR_POINT_MAX;
    plug.ifreplace = 1;
    plug.closecont = 0;
    plug.docopy = 0;
    plug.copiedco = -1;
    let values = native.load_settings("LineTracker");
    if state.first {
        set_defaults(plug);
        if values.len() > 12 {
            plug.left = values[0] as i32;
            plug.top = values[1] as i32;
            plug.ksize = values[2] as i32;
            plug.knum = values[3] as i32;
            plug.sigma = values[4] as f32;
            plug.h = values[5] as f32;
            plug.ifdark = values[6] as i32;
            plug.stepsize = values[7] as f32;
            plug.redtol = values[8] as f32;
            plug.offset = values[9] as f32;
            plug.copytol = values[10] as i32;
            plug.copypool = values[11] as i32;
            plug.copyfit = values[12] as i32;
        }
    }
    native.create_window("Line Tracker", "lineTracker.html#TOP", 'T');
    plug.window_open = true;
    state.top_window_open = true;
    if !state.first || values.len() > 12 {
        native.set_window_position(plug.left, plug.top);
    }
    state.first = false;
    native.show_window();
}

/// `LineTrack` from `linegui.h`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LineTrack {
    pub m_edit: Vec<LineTrackEditRow>,
    pub m_undo_enabled: bool,
    pub m_rounded_style: bool,
}

impl LineTrack {
    /// `LineTrack::LineTrack`.
    pub fn new(plug: &PlugData) -> Self {
        let mut line = Self::default();
        line.make_edit_row(
            0,
            "0 Light, 1 Dark Lines",
            LineTrackValue::Ifdark,
            true,
            0.,
            1.,
            "0 or 1 if lines are lighter or darker than background",
            plug,
        );
        line.make_edit_row(
            1,
            "Offset from Line",
            LineTrackValue::Offset,
            false,
            -5.,
            5.,
            "Offset from center to place contour on edge of feature",
            plug,
        );
        line.make_edit_row(
            2,
            "Sigma (Line width)",
            LineTrackValue::Sigma,
            false,
            0.5,
            7.,
            "Half-width of smoothing gaussian across the line",
            plug,
        );
        line.make_edit_row(
            3,
            "H (half-length)",
            LineTrackValue::H,
            false,
            1.,
            20.,
            "Half-length of smoothing gaussian along the line",
            plug,
        );
        line.make_edit_row(
            4,
            "Kernel Half-size",
            LineTrackValue::Ksize,
            true,
            1.,
            10.,
            "Determines total size of kernels",
            plug,
        );
        line.make_edit_row(
            5,
            "Reduction Tolerance",
            LineTrackValue::Redtol,
            false,
            0.,
            5.,
            "Maximum error in removing points along path",
            plug,
        );
        line.make_edit_row(
            6,
            "Copy Tolerance",
            LineTrackValue::Copytol,
            true,
            0.,
            10.,
            "Maximum shift of points when copying",
            plug,
        );
        line.make_edit_row(
            7,
            "Copy Pooling",
            LineTrackValue::Copypool,
            true,
            -5.,
            5.,
            "# of points to consider when shifting after a copy",
            plug,
        );
        line.make_edit_row(
            8,
            "Copy Smoothing",
            LineTrackValue::Copyfit,
            true,
            0.,
            9.,
            "# of points to smooth over after a copy",
            plug,
        );
        line.make_edit_row(
            9,
            "Number of Kernels",
            LineTrackValue::Knum,
            true,
            9.,
            40.,
            "# of directions to analyze",
            plug,
        );
        line.make_edit_row(
            10,
            "Step Size",
            LineTrackValue::Stepsize,
            false,
            1.,
            20.,
            "Pixels to move between points when tracking",
            plug,
        );
        line
    }

    /// `LineTrack::~LineTrack`.
    pub fn destroy(&mut self) {}

    /// `LineTrack::makeEditRow`.
    pub fn make_edit_row(
        &mut self,
        row: usize,
        label: &str,
        value_kind: LineTrackValue,
        is_int: bool,
        min: f32,
        max: f32,
        tip: &str,
        plug: &PlugData,
    ) {
        let edit = LineTrackEditRow {
            label: label.into(),
            value: String::new(),
            value_kind,
            is_int,
            min,
            max,
            tip: tip.into(),
        };
        if row == self.m_edit.len() {
            self.m_edit.push(edit);
        } else if row < self.m_edit.len() {
            self.m_edit[row] = edit;
        }
        self.fillin_value(row, plug);
    }

    /// `LineTrack::fillinValue`.
    pub fn fillin_value(&mut self, row: usize, plug: &PlugData) {
        let edit = &mut self.m_edit[row];
        let value = match edit.value_kind {
            LineTrackValue::Ifdark => plug.ifdark as f32,
            LineTrackValue::Offset => plug.offset,
            LineTrackValue::Sigma => plug.sigma,
            LineTrackValue::H => plug.h,
            LineTrackValue::Ksize => plug.ksize as f32,
            LineTrackValue::Redtol => plug.redtol,
            LineTrackValue::Copytol => plug.copytol as f32,
            LineTrackValue::Copypool => plug.copypool as f32,
            LineTrackValue::Copyfit => plug.copyfit as f32,
            LineTrackValue::Knum => plug.knum as f32,
            LineTrackValue::Stepsize => plug.stepsize,
        };
        edit.value = if edit.is_int {
            format!("{}", value as i32)
        } else {
            format!("{}", value)
        };
    }

    /// `LineTrack::valueEntered`.
    pub fn value_entered(&mut self, which: usize, text: &str, plug: &mut PlugData) {
        let edit = &mut self.m_edit[which];
        if edit.is_int {
            let mut value = text.trim().parse::<i32>().unwrap_or(0);
            value = value
                .max((edit.min + 0.5).floor() as i32)
                .min((edit.max + 0.5).floor() as i32);
            match edit.value_kind {
                LineTrackValue::Ifdark => plug.ifdark = value,
                LineTrackValue::Ksize => plug.ksize = value,
                LineTrackValue::Copytol => plug.copytol = value,
                LineTrackValue::Copypool => plug.copypool = value,
                LineTrackValue::Copyfit => plug.copyfit = value,
                LineTrackValue::Knum => plug.knum = value,
                _ => unreachable!(),
            }
            edit.value = value.to_string();
        } else {
            let value = text
                .trim()
                .parse::<f32>()
                .unwrap_or(0.)
                .max(edit.min)
                .min(edit.max);
            match edit.value_kind {
                LineTrackValue::Offset => plug.offset = value,
                LineTrackValue::Sigma => plug.sigma = value,
                LineTrackValue::H => plug.h = value,
                LineTrackValue::Redtol => plug.redtol = value,
                LineTrackValue::Stepsize => plug.stepsize = value,
                _ => unreachable!(),
            }
            edit.value = value.to_string();
        }
    }

    /// `LineTrack::track`.
    pub fn track(
        &mut self,
        client: i32,
        state: &mut LineTrackState,
        native: &mut dyn LineTrackNativeBoundary,
    ) {
        let plug = &mut state.this_plug;
        let closecont = plug.closecont;
        plug.closecont = 0;
        let mut copytol = plug.docopy;
        plug.docopy = 0;
        if client > 1 {
            copytol = plug.copytol;
        }
        let mut points = match native.selected_contour() {
            Some(points) => points,
            None => {
                native.print("\x07Line Track Error:\n  No Contour Selected.\n");
                return;
            }
        };
        let maxpoint = points.len();
        if maxpoint < 2 {
            native.print("\x07Line Track Error:\n  Contour must have at least 2 points.\n");
            return;
        }
        let (xsize, ysize, zsize) = native.image_size();
        plug.xsize = xsize;
        plug.ysize = ysize;
        plug.zsize = zsize;
        let (_, _, curz) = native.current_location();
        let cached = native.tile_or_strip_cache();
        let image = match native.current_section(curz, cached) {
            Some(image) => image,
            None => {
                native.print("\x07Line Track Error:\n  No current image data.\n");
                return;
            }
        };
        if plug.xsize != plug.idata_xsize || plug.ysize != plug.idata_ysize {
            let size = (plug.xsize as usize).saturating_mul(plug.ysize as usize);
            plug.idata = vec![0; size];
            plug.idata_xsize = plug.xsize;
            plug.idata_ysize = plug.ysize;
            plug.idata_sec = -1;
        }
        let flipped = native.model_flipped();
        if curz != plug.idata_sec
            || flipped != plug.idata_flipped
                && native.copy_image_to_byte_buffer(&image, &mut plug.idata)
        {
            native.print("\x07Line Track failed to get memory for short to byte map.\n");
            return;
        }
        plug.idata_sec = curz;
        plug.idata_flipped = flipped;
        let mut curpt;
        if copytol != 0 {
            let zdiff = points[0].z.round() as i32 - curz;
            if zdiff != 1 && zdiff != -1 {
                native.print("\x07Contour Copy Error:\n  Copy must be to adjacent section\n");
                return;
            }
            if maxpoint < 4 {
                native.print("\x07Contour Copy Error:\n  Contour must have at least 4 points.\n");
                return;
            }
            let (_, copiedco, point) = native.contour_index();
            plug.copiedco = copiedco;
            plug.pt = point;
            let max_contour = native.object_max_contour();
            native.contour_addition(max_contour);
            native.new_contour();
            for point in &mut points {
                point.z = curz as f32;
            }
            native.replace_selected_contour(points.clone());
            native.set_new_contour_time();
            let (ob, co, _) = native.contour_index();
            plug.ob = ob;
            plug.co = co;
            curpt = plug.pt;
            let destination = native.object_max_contour() - 1;
            native.copy_contour_properties(plug.copiedco, destination);
        } else {
            let (ob, co, point) = native.contour_index();
            plug.ob = ob;
            plug.co = co;
            plug.pt = point;
            curpt = point;
            plug.copiedco = -1;
            plug.undo_cont = Some(points.clone());
        }
        self.m_undo_enabled = true;
        points.reserve(CONTOUR_POINT_MAX);
        if copytol == 0 {
            native.contour_data_change();
            if native.line_track(
                &plug.idata,
                plug.xsize,
                plug.ysize,
                &mut points,
                &mut curpt,
                plug.ksize,
                plug.knum,
                plug.sigma,
                plug.h,
                plug.ifdark,
                plug.stepsize,
                plug.redtol,
                plug.ifreplace,
                plug.offset,
                closecont,
            ) != 0
            {
                native.print("\x07LineTrack failed to find path\n");
                native.flush_undo_unit();
            }
        } else {
            native.contour_track(
                &plug.idata,
                plug.xsize,
                plug.ysize,
                &mut points,
                &mut curpt,
                plug.ksize,
                plug.knum,
                plug.sigma,
                plug.h,
                plug.ifdark,
                plug.stepsize,
                plug.redtol,
                plug.offset,
                copytol,
                plug.copypool,
                plug.copyfit,
            );
        }
        plug.copysize = points.len() as i32;
        native.replace_selected_contour(points);
        native.set_index(plug.ob, plug.co, curpt);
        native.finish_undo_unit();
        native.draw_model();
    }

    /// `LineTrack::undo`.
    pub fn undo(&mut self, native: &mut dyn LineTrackNativeBoundary) {
        native.undo();
    }

    /// `LineTrack::buttonPressed`.
    pub fn button_pressed(
        &mut self,
        which: i32,
        state: &mut LineTrackState,
        native: &mut dyn LineTrackNativeBoundary,
    ) {
        match which {
            0 => self.track(1, state, native),
            1 => self.track(2, state, native),
            2 => self.undo(native),
            3 => {
                set_defaults(&mut state.this_plug);
                for i in 0..10 {
                    self.fillin_value(i, &state.this_plug);
                }
            }
            _ => {}
        }
    }

    /// `LineTrack::topCloseEvent`.
    pub fn top_close_event(
        &mut self,
        state: &mut LineTrackState,
        native: &mut dyn LineTrackNativeBoundary,
    ) {
        let (left, top) = native.geometry();
        let plug = &mut state.this_plug;
        plug.left = left;
        plug.top = top;
        native.save_settings(
            "LineTracker",
            &[
                left as f64,
                top as f64,
                plug.ksize as f64,
                plug.knum as f64,
                plug.sigma as f64,
                plug.h as f64,
                plug.ifdark as f64,
                plug.stepsize as f64,
                plug.redtol as f64,
                plug.offset as f64,
                plug.copytol as f64,
                plug.copypool as f64,
                plug.copyfit as f64,
            ],
        );
        native.remove_window();
        if native.tile_or_strip_cache() {
            native.free_tile_cached_section();
        }
        plug.window_open = false;
        state.top_window_open = false;
        plug.undo_cont = None;
        plug.tmp_cont = None;
        plug.idata.clear();
        native.accept_close();
    }

    /// `LineTrack::topChangeEvent`.
    pub fn top_change_event(&mut self, native: &mut dyn LineTrackNativeBoundary) {
        native.rounded_style();
        native.dialog_change_event();
        native.check_and_set_mac_menu();
        if native.font_change_event() {
            return;
        }
    }
    /// `LineTrack::keyPressEvent`.
    pub fn key_press_event(&mut self, native: &mut dyn LineTrackNativeBoundary) {
        if native.close_key() {
            native.close_window();
        } else {
            native.control_key(false);
        }
    }
    /// `LineTrack::keyReleaseEvent`.
    pub fn key_release_event(&mut self, native: &mut dyn LineTrackNativeBoundary) {
        native.control_key(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_and_value_ranges_match_linegui() {
        let mut plug = PlugData::default();
        set_defaults(&mut plug);
        let mut line = LineTrack::new(&plug);
        line.value_entered(0, "4garbage", &mut plug);
        assert_eq!(plug.ifdark, 0);
        line.value_entered(4, "100", &mut plug);
        assert_eq!(plug.ksize, 10);
        line.value_entered(1, "-9", &mut plug);
        assert_eq!(plug.offset, -5.);
        assert_eq!(line.m_edit.len(), 11);
    }
    #[test]
    fn plugin_info_matches_source() {
        let mut typ = 0;
        assert_eq!(imod_plug_info(Some(&mut typ)), "Line Track");
        assert_eq!(typ, IMOD_PLUG_MENU + IMOD_PLUG_KEYS);
    }
}
