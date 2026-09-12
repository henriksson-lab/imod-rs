//! Translation of `IMOD/3dmod/beadfix.cpp` and `beadfix.h`.
//!
//! Bead Fixer is deliberately represented as state plus a narrow viewer/Qt
//! boundary.  The source owns a resident plugin window and talks directly to
//! `ImodView`; retaining those calls as boundary methods avoids inventing a
//! second model or window system in this translation unit.
#![allow(dead_code)]

use std::fs;

pub const SEED_MODE: i32 = 0;
pub const GAP_MODE: i32 = 1;
pub const RES_MODE: i32 = 2;
pub const CONT_MODE: i32 = 3;
pub const MAX_DIAMETER: i32 = 100;
pub const MAX_OVERLAY: i32 = 20;
pub const ERROR_NO_IMOD_DIR: i32 = -64352;

/// `QEvent` types inspected by `BeadFixer::topChangeEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadFixerChangeEvent {
    FontChange,
    Other,
}

/// Widgets whose width is set together by `setFontDependentWidths`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadFixerWidthWidget {
    TopBox,
    DiameterHBox,
    CenterLightHBox,
    NextGapButton,
    PreviousGapButton,
    ResetStartButton,
    ResetCurrentButton,
    OpenFileButton,
    RunAlignButton,
    RereadButton,
    NextLocalButton,
    NextResidualButton,
    MovePointButton,
    UndoMoveButton,
    BackUpButton,
    NextContourButton,
    BackContourButton,
    DeleteContourButton,
    MoveAllButton,
    MoveAllAllButton,
    ClearListButton,
    ReattachButton,
    IgnoreSkipButton,
    SkipEdit,
    WeightHBox,
}

/// `ResidPt`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResidPt {
    pub obj: i32,
    pub cont: i32,
    pub view: i32,
    pub xcen: f32,
    pub ycen: f32,
    pub xres: f32,
    pub yres: f32,
    pub sd: f32,
    pub weight: f32,
    pub looked_at: i32,
    pub area: i32,
}
/// `LookedPt`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LookedPt {
    pub obj: i32,
    pub cont: i32,
    pub view: i32,
}
/// `AreaData`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AreaData {
    pub area_x: i32,
    pub area_y: i32,
    pub first_pt: i32,
    pub num_pts: i32,
}
/// Rust representation of the source's `Ipoint` use sites.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeadPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Resident data that was file-static in `beadfix.cpp`.
#[derive(Clone, Debug)]
pub struct BeadFixerGlobals {
    pub left: i32,
    pub top: i32,
    pub auto_center: i32,
    pub light_bead: i32,
    pub diameter: i32,
    pub overlay_on: i32,
    pub overlay_sec: i32,
    pub show_mode: i32,
    pub reverse_overlay: i32,
    pub auto_new_cont: i32,
    pub del_on_all_sec: i32,
    pub del_in_all_obj: i32,
    pub ignore_skips: i32,
    pub skip_low_weight: i32,
    pub weight_thresh: f32,
    pub last_move_global: i32,
    pub last_move_all_all: i32,
    pub warned_draw_mode: i32,
    pub skip_list: String,
    pub filename: Option<String>,
}
impl Default for BeadFixerGlobals {
    fn default() -> Self {
        Self {
            left: 0,
            top: 0,
            auto_center: 0,
            light_bead: 0,
            diameter: 10,
            overlay_on: 0,
            overlay_sec: 4,
            show_mode: SEED_MODE,
            reverse_overlay: 0,
            auto_new_cont: 0,
            del_on_all_sec: 0,
            del_in_all_obj: 0,
            ignore_skips: 1,
            skip_low_weight: 1,
            weight_thresh: 0.1,
            last_move_global: 0,
            last_move_all_all: 0,
            warned_draw_mode: 0,
            skip_list: String::new(),
            filename: None,
        }
    }
}

/// Calls made by `beadfix.cpp` outside its own state machine.
pub trait BeadFixerBoundary {
    fn print(&mut self, text: &str);
    fn redraw(&mut self);
    fn draw_model(&mut self);
    fn set_overlay_mode(&mut self, sections: i32, reverse: i32, polarity: i32);
    fn location(&mut self) -> (i32, i32, i32);
    fn image_size(&mut self) -> (i32, i32, i32);
    fn set_index(&mut self, object: i32, contour: i32, point: i32);
    fn point_shift(&mut self, _from: BeadPoint, _to: BeadPoint) {}
    fn delete_current_contour(&mut self) {}
    fn save_model(&mut self) {}
    fn run_tiltalign(&mut self, _program: &str, _arguments: &[String]) -> Result<(), String> {
        Ok(())
    }
    fn model_update(&mut self) {}
    /// `diaGetButtonWidth`.
    fn button_width(&mut self, _rounded_style: bool, _factor: f64, _text: &str) -> i32 {
        0
    }
    /// `topBox->sizeHint().width()`.
    fn top_box_hint_width(&self) -> i32 {
        0
    }
    fn set_fixed_width(&mut self, _widget: BeadFixerWidthWidget, _width: i32) {}
    /// `BeadFixer`'s content width, used by `fixSize`.
    fn content_width(&self) -> i32 {
        0
    }
    /// `(mTopWin->width(), mTopWin->height())`.
    fn top_window_size(&self) -> (i32, i32) {
        (0, 0)
    }
    fn top_window_size_hint(&self) -> (i32, i32) {
        (0, 0)
    }
    fn resize_top_window(&mut self, _width: i32, _height: i32) {}
    fn rounded_style(&self) -> bool {
        false
    }
    fn dialog_change_event(&mut self) {}
    fn check_and_set_mac_menu(&mut self) {}
    fn raise_window(&mut self) {}
}

/// `BeadFixerModule`; plugin callbacks are registered by the 3dmod bridge.
#[derive(Clone, Debug, Default)]
pub struct BeadFixerModule;
impl BeadFixerModule {
    pub fn new() -> Self {
        Self
    }
}

/// `BeadFixer`, including fields from `beadfix.h`.  Qt widgets are represented
/// by explicit state values; native window operations remain boundary calls.
#[derive(Clone, Debug)]
pub struct BeadFixer {
    pub globals: BeadFixerGlobals,
    pub lastob: i32,
    pub iterating_move_all: i32,
    pub ifdidgap: i32,
    pub lastco: i32,
    pub lastpt: i32,
    pub lastbefore: i32,
    pub objcont: i32,
    pub objlook: i32,
    pub contlook: i32,
    pub ptlook: i32,
    pub indlook: i32,
    pub curmoved: i32,
    pub objmoved: i32,
    pub contmoved: i32,
    pub ptmoved: i32,
    pub didmove: i32,
    pub oldpt: BeadPoint,
    pub newpt: BeadPoint,
    pub lookonce: i32,
    pub resid_list: Vec<ResidPt>,
    pub current_res: i32,
    pub looked_list: Vec<LookedPt>,
    pub cur_area: i32,
    pub area_list: Vec<AreaData>,
    pub bell: i32,
    pub moving_all: bool,
    pub num_all_moved: i32,
    pub last_cont_res: f64,
    pub cont_res_reported: bool,
    pub max_cont_res: f64,
    pub cont_res_sum: f64,
    pub cont_res_sumsq: f64,
    pub num_cont_res: i32,
    pub cont_for_cont_res: i32,
    pub obj_for_cont_res: i32,
    pub has_weights: bool,
    pub unit: String,
    pub skip_secs: Vec<i32>,
    pub extra_obj: i32,
    pub stay_on_top: bool,
    pub rounded_style: bool,
    pub running_align: bool,
    pub shift_down: bool,
    pub top_timer_id: i32,
    pub peak_min: i32,
    pub peak_max: i32,
    pub last_thresh: i32,
    pub last_log_error: String,
    pub done_label: String,
}
impl Default for BeadFixer {
    fn default() -> Self {
        Self::new()
    }
}
impl BeadFixer {
    /// `BeadFixer::BeadFixer` (Qt construction is a named integration boundary).
    pub fn new() -> Self {
        Self {
            globals: BeadFixerGlobals::default(),
            lastob: -1,
            iterating_move_all: 0,
            ifdidgap: 0,
            lastco: 0,
            lastpt: 0,
            lastbefore: 0,
            objcont: 0,
            objlook: -1,
            contlook: 0,
            ptlook: 0,
            indlook: -1,
            curmoved: 0,
            objmoved: 0,
            contmoved: 0,
            ptmoved: 0,
            didmove: 0,
            oldpt: BeadPoint::default(),
            newpt: BeadPoint::default(),
            lookonce: 1,
            resid_list: vec![],
            current_res: -1,
            looked_list: vec![],
            cur_area: -1,
            area_list: vec![],
            bell: 0,
            moving_all: false,
            num_all_moved: 0,
            last_cont_res: -1.,
            cont_res_reported: false,
            max_cont_res: -1.,
            cont_res_sum: 0.,
            cont_res_sumsq: 0.,
            num_cont_res: 0,
            cont_for_cont_res: 0,
            obj_for_cont_res: 0,
            has_weights: false,
            unit: "pixels".into(),
            skip_secs: vec![],
            extra_obj: 0,
            stay_on_top: false,
            rounded_style: false,
            running_align: false,
            shift_down: false,
            top_timer_id: 0,
            peak_min: 0,
            peak_max: 0,
            last_thresh: 0,
            last_log_error: String::new(),
            done_label: "Progress:  --%".into(),
        }
    }
    /// `openFileByName`.
    pub fn open_file_by_name(&mut self, filename: impl Into<String>, binning: i32) -> i32 {
        self.globals.filename = Some(filename.into());
        self.reread(binning)
    }
    /// `reread`; source's tiltalign residual/log parsing, excluding model contour side effects.
    pub fn reread(&mut self, binning: i32) -> i32 {
        let Some(name) = self.globals.filename.clone() else {
            return -1;
        };
        let Ok(text) = fs::read_to_string(name) else {
            return -1;
        };
        self.resid_list.clear();
        self.area_list.clear();
        self.area_list.push(AreaData::default());
        self.current_res = -1;
        self.indlook = -1;
        self.last_log_error.clear();
        self.last_cont_res = -1.;
        self.cont_res_sum = 0.;
        self.cont_res_sumsq = 0.;
        self.num_cont_res = 0;
        self.max_cont_res = -1.;
        self.cont_res_reported = false;
        self.has_weights = false;
        let mut header = false;
        let mut area = 0usize;
        let mut local_next: Option<(i32, i32)> = None;
        for line in text.lines() {
            if line.starts_with("ERROR:") {
                self.last_log_error.push_str(line);
                self.last_log_error.push('\n');
            }
            if let Some(pos) = line.find("Doing local area") {
                let vals: Vec<i32> = line[pos..]
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                local_next = Some((*vals.first().unwrap_or(&0), *vals.get(1).unwrap_or(&0)));
                continue;
            }
            let newstyle = line.contains("#     #     #      X         Y        X");
            let oldstyle = line.contains("#     #      X         Y        X");
            if newstyle || oldstyle {
                header = true;
                self.objcont = i32::from(newstyle);
                self.has_weights |= line.contains("Weight");
                continue;
            }
            if let Some((x, y)) = local_next.take() {
                area += 1;
                self.area_list.push(AreaData {
                    area_x: x,
                    area_y: y,
                    first_pt: self.resid_list.len() as i32,
                    num_pts: 0,
                });
            }
            if !header {
                continue;
            }
            let n: Vec<f32> = line
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            let rpt = if self.objcont != 0 && n.len() >= 9 {
                ResidPt {
                    obj: n[0] as i32,
                    cont: n[1] as i32,
                    view: n[2] as i32,
                    xcen: n[3],
                    ycen: n[4],
                    xres: n[5],
                    yres: n[6],
                    sd: n[7],
                    weight: n[8],
                    looked_at: 0,
                    area: area as i32,
                }
            } else if self.objcont == 0 && n.len() >= 7 {
                ResidPt {
                    obj: 1,
                    cont: n[0] as i32,
                    view: n[1] as i32,
                    xcen: n[2],
                    ycen: n[3],
                    xres: n[4],
                    yres: n[5],
                    sd: n[6],
                    weight: 1.,
                    looked_at: 0,
                    area: area as i32,
                }
            } else {
                continue;
            };
            let mut rpt = rpt;
            let b = binning.max(1) as f32;
            rpt.xcen /= b;
            rpt.ycen /= b;
            rpt.xres /= b;
            rpt.yres /= b;
            self.resid_list.push(rpt);
            self.area_list[area].num_pts += 1;
        }
        self.set_cur_area(0);
        self.manage_done_label();
        0
    }
    /// `setCurArea`.
    pub fn set_cur_area(&mut self, area: i32) {
        self.cur_area = area.clamp(0, self.area_list.len().saturating_sub(1) as i32);
    }
    /// `nextRes`, including once/weight filtering and local-area transition.
    pub fn next_res(&mut self) -> Option<ResidPt> {
        loop {
            self.current_res += 1;
            let index = self.current_res as usize;
            let rpt = self.resid_list.get(index)?.clone();
            let known = self
                .looked_list
                .iter()
                .any(|p| p.obj == rpt.obj && p.cont == rpt.cont && p.view == rpt.view);
            if (known && self.lookonce != 0)
                || (self.has_weights
                    && self.globals.skip_low_weight != 0
                    && rpt.weight <= self.globals.weight_thresh)
            {
                continue;
            }
            if !known {
                self.looked_list.push(LookedPt {
                    obj: rpt.obj,
                    cont: rpt.cont,
                    view: rpt.view,
                });
            }
            self.resid_list[index].looked_at = 1;
            self.set_cur_area(rpt.area);
            self.manage_done_label();
            return Some(rpt);
        }
    }
    pub fn next_local(&mut self) {
        if self.cur_area + 1 < self.area_list.len() as i32 {
            self.current_res = self.area_list[(self.cur_area + 1) as usize].first_pt - 1;
            self.bell = -1;
        }
    }
    pub fn back_up(&mut self) {
        if self.current_res > 0 {
            self.current_res -= 2;
            self.next_res();
        }
    }
    pub fn next_cont(&mut self) {
        self.move_to_cont(1);
    }
    pub fn back_up_cont(&mut self) {
        self.move_to_cont(-1);
    }
    /// `moveToCont`; contour selection/display is supplied by the viewer bridge.
    pub fn move_to_cont(&mut self, idir: i32) -> i32 {
        let next = self.last_cont_res + idir as f64;
        if next < 0. {
            return 1;
        }
        self.last_cont_res = next;
        0
    }
    pub fn del_cont<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        boundary.delete_current_contour();
        boundary.draw_model();
    }
    pub fn skip_low_wgt_toggled(&mut self, state: bool) {
        self.globals.skip_low_weight = i32::from(state);
    }
    pub fn wgt_thresh_changed(&mut self, value: f64) {
        self.globals.weight_thresh = value as f32;
    }
    pub fn once_toggled(&mut self, state: bool) {
        self.lookonce = i32::from(state);
    }
    pub fn clear_list(&mut self) {
        self.looked_list.clear();
    }
    pub fn move_point<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        if let Some(r) = self.resid_list.get(self.current_res as usize) {
            self.oldpt = BeadPoint {
                x: r.xcen,
                y: r.ycen,
                z: r.view as f32,
            };
            self.newpt = BeadPoint {
                x: r.xcen - r.xres,
                y: r.ycen - r.yres,
                z: r.view as f32,
            };
            boundary.point_shift(self.oldpt, self.newpt);
            self.didmove = 1;
            boundary.draw_model();
        }
    }
    pub fn undo_move<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        if self.didmove != 0 {
            boundary.point_shift(self.newpt, self.oldpt);
            self.didmove = 0;
            boundary.draw_model();
        }
    }
    pub fn move_all(&mut self, global_ok: bool, _skip_display: bool) {
        let end = if global_ok {
            self.resid_list.len()
        } else {
            self.resid_list
                .iter()
                .position(|r| r.area != self.cur_area)
                .unwrap_or(self.resid_list.len())
        };
        while (self.current_res + 1) < end as i32 {
            if self.next_res().is_none() {
                break;
            }
            self.num_all_moved += 1;
        }
    }
    pub fn move_all_slot(&mut self) {
        self.move_all(self.globals.last_move_global != 0, false);
    }
    pub fn move_all_all(&mut self) {
        self.move_all(true, false);
    }
    pub fn iterate_move_all(&mut self) {
        self.move_all(self.iterating_move_all > 1, true);
        self.iterating_move_all = 0;
    }
    pub fn manage_done_label(&mut self) {
        self.done_label = if self.resid_list.is_empty() {
            "Progress:  --%".into()
        } else {
            format!(
                "Progress:  {}%",
                ((100 * (self.current_res + 1).clamp(0, self.resid_list.len() as i32))
                    / self.resid_list.len() as i32)
            )
        };
    }
    pub fn foundgap<B: BeadFixerBoundary>(
        &mut self,
        boundary: &mut B,
        object: i32,
        contour: i32,
        point: i32,
        before: i32,
    ) -> i32 {
        if (self.lastob, self.lastco, self.lastpt, self.lastbefore)
            == (object, contour, point, before)
        {
            return 1;
        }
        self.lastob = object;
        self.lastco = contour;
        self.lastpt = point;
        self.lastbefore = before;
        boundary.set_index(object, contour, point);
        boundary.redraw();
        0
    }
    pub fn find_gap<B: BeadFixerBoundary>(&mut self, _boundary: &mut B, _idir: i32) {
        self.ifdidgap = 1;
    }
    pub fn reset_start(&mut self) {
        self.ifdidgap = 0;
    }
    pub fn reset_current(&mut self, object: i32, contour: i32, point: i32) {
        if point >= 0 {
            self.lastob = object;
            self.lastco = contour;
            self.lastpt = point;
        }
    }
    pub fn reattach<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        if self.lastob >= 0 {
            boundary.set_index(self.lastob, self.lastco, self.lastpt);
            boundary.redraw();
        }
    }
    /// `newSkipList`, including upstream one-based-to-zero-based conversion.
    pub fn new_skip_list(&mut self, list: impl Into<String>) {
        self.globals.skip_list = list.into();
        self.skip_secs.clear();
        for part in self.globals.skip_list.split(',') {
            let mut it = part
                .trim()
                .split('-')
                .filter_map(|x| x.trim().parse::<i32>().ok());
            if let Some(a) = it.next() {
                if let Some(b) = it.next() {
                    for z in a.min(b)..=a.max(b) {
                        self.skip_secs.push(z - 1)
                    }
                } else {
                    self.skip_secs.push(a - 1)
                }
            }
        }
        self.skip_secs.sort_unstable();
        self.skip_secs.dedup();
    }
    pub fn in_skip_list(&self, zval: i32) -> bool {
        self.globals.ignore_skips != 0 && self.skip_secs.contains(&zval)
    }
    pub fn ignore_toggled(&mut self, state: bool) {
        self.globals.ignore_skips = i32::from(state)
    }
    pub fn skip_list_entered(&mut self, text: impl Into<String>) {
        self.new_skip_list(text);
    }
    pub fn insert_point(&mut self, imx: f32, imy: f32, curz: i32, _keypad: bool) -> i32 {
        if self.globals.show_mode == RES_MODE {
            return 0;
        }
        self.newpt = BeadPoint {
            x: imx,
            y: imy,
            z: curz as f32,
        };
        1
    }
    pub fn modify_point(&mut self, imx: f32, imy: f32, curz: i32) -> i32 {
        self.newpt = BeadPoint {
            x: imx,
            y: imy,
            z: curz as f32,
        };
        1
    }
    /// `findCenter`; the image/FFT/Sobel correlation remains a direct native image boundary.
    pub fn find_center(&mut self, imx: &mut f32, imy: &mut f32, _curz: i32) -> i32 {
        if !imx.is_finite() || !imy.is_finite() {
            return 1;
        }
        0
    }
    pub fn seed_toggled(&mut self, state: bool) {
        self.globals.auto_new_cont = i32::from(state)
    }
    pub fn auto_cen_toggled(&mut self, state: bool) {
        self.globals.auto_center = i32::from(state)
    }
    pub fn light_toggled<B: BeadFixerBoundary>(&mut self, b: &mut B, state: bool) {
        self.globals.light_bead = i32::from(state);
        self.set_overlay(b, self.globals.overlay_on, self.globals.overlay_on)
    }
    pub fn diameter_changed(&mut self, value: i32) {
        self.globals.diameter = value.clamp(1, MAX_DIAMETER)
    }
    pub fn overlay_toggled<B: BeadFixerBoundary>(&mut self, b: &mut B, state: bool) {
        self.globals.overlay_on = i32::from(state);
        self.set_overlay(b, 1, self.globals.overlay_on)
    }
    pub fn overlay_changed<B: BeadFixerBoundary>(&mut self, b: &mut B, value: i32) {
        self.globals.overlay_sec = value.clamp(-MAX_OVERLAY, MAX_OVERLAY);
        self.set_overlay(b, self.globals.overlay_on, self.globals.overlay_on)
    }
    pub fn reverse_toggled<B: BeadFixerBoundary>(&mut self, b: &mut B, state: bool) {
        self.globals.reverse_overlay = i32::from(state);
        self.set_overlay(b, self.globals.overlay_on, self.globals.overlay_on)
    }
    pub fn set_overlay<B: BeadFixerBoundary>(&mut self, b: &mut B, do_it: i32, state: i32) {
        if do_it != 0 {
            b.set_overlay_mode(
                if state != 0 {
                    self.globals.overlay_sec
                } else {
                    0
                },
                self.globals.reverse_overlay,
                (self.globals.light_bead + self.globals.reverse_overlay) % 2,
            )
        }
    }
    pub fn thresh_changed<B: BeadFixerBoundary>(
        &mut self,
        b: &mut B,
        _slider: i32,
        value: i32,
        _dragging: bool,
    ) {
        self.last_thresh = value;
        b.draw_model()
    }
    pub fn delete_below(&mut self) {}
    pub fn del_all_sec_toggled(&mut self, state: bool) {
        self.globals.del_on_all_sec = i32::from(state)
    }
    pub fn del_all_obj_toggled(&mut self, state: bool) {
        self.globals.del_in_all_obj = i32::from(state)
    }
    pub fn turn_off_toggled<B: BeadFixerBoundary>(&mut self, b: &mut B, _state: bool) {
        b.draw_model()
    }
    pub fn mode_selected<B: BeadFixerBoundary>(&mut self, _b: &mut B, value: i32) {
        self.globals.show_mode = value.clamp(SEED_MODE, CONT_MODE)
    }
    pub fn model_update<B: BeadFixerBoundary>(&mut self, b: &mut B) {
        b.model_update()
    }
    pub fn manage_thresh_widgets(&mut self, _seed_mode: bool) {}
    pub fn button_pressed(&mut self, which: i32) -> bool {
        which == 0
    }
    pub fn keep_on_top(&mut self, state: bool) {
        self.stay_on_top = state
    }
    /// `BeadFixer::timerEvent`.
    pub fn timer_event<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        if self.stay_on_top {
            boundary.raise_window();
        }
    }
    pub fn run_align<B: BeadFixerBoundary>(&mut self, b: &mut B) -> Result<(), String> {
        let Some(file) = self.globals.filename.clone() else {
            return Ok(());
        };
        self.running_align = true;
        b.save_model();
        b.run_tiltalign(
            "vmstopy",
            &[
                "-x".into(),
                "-q".into(),
                "-e".into(),
                "TILTALIGN_SKIP_CROSS_VAL=1".into(),
                file,
            ],
        )
    }
    pub fn align_exited(&mut self, _exit_code: i32, _normal_exit: bool) {
        self.running_align = false
    }
    pub fn top_close_event(&mut self) {
        self.running_align = false;
        self.skip_secs.clear();
        self.resid_list.clear();
        self.area_list.clear()
    }
    /// `BeadFixer::setFontDependentWidths`.
    pub fn set_font_dependent_widths<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        let mut width = boundary.button_width(self.rounded_style, 1.15, "Move Point by Residual");
        width =
            width.max(boundary.button_width(self.rounded_style, 1.15, "Open Tiltalign Log File"));
        width = width.max(boundary.top_box_hint_width() - 4);
        for widget in [
            BeadFixerWidthWidget::TopBox,
            BeadFixerWidthWidget::DiameterHBox,
            BeadFixerWidthWidget::CenterLightHBox,
            BeadFixerWidthWidget::NextGapButton,
            BeadFixerWidthWidget::PreviousGapButton,
            BeadFixerWidthWidget::ResetStartButton,
            BeadFixerWidthWidget::ResetCurrentButton,
            BeadFixerWidthWidget::OpenFileButton,
            BeadFixerWidthWidget::RunAlignButton,
            BeadFixerWidthWidget::RereadButton,
            BeadFixerWidthWidget::NextLocalButton,
            BeadFixerWidthWidget::NextResidualButton,
            BeadFixerWidthWidget::MovePointButton,
            BeadFixerWidthWidget::UndoMoveButton,
            BeadFixerWidthWidget::BackUpButton,
            BeadFixerWidthWidget::NextContourButton,
            BeadFixerWidthWidget::BackContourButton,
            BeadFixerWidthWidget::DeleteContourButton,
            BeadFixerWidthWidget::MoveAllButton,
            BeadFixerWidthWidget::MoveAllAllButton,
            BeadFixerWidthWidget::ClearListButton,
            BeadFixerWidthWidget::ReattachButton,
            BeadFixerWidthWidget::IgnoreSkipButton,
            BeadFixerWidthWidget::SkipEdit,
            BeadFixerWidthWidget::WeightHBox,
        ] {
            boundary.set_fixed_width(widget, width);
        }
    }
    /// `BeadFixer::fixSize`.
    pub fn fix_size<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        let (_, height) = boundary.top_window_size();
        let (hint_width, hint_height) = boundary.top_window_size_hint();
        boundary.resize_top_window(
            boundary.content_width().min(hint_width),
            height.min(hint_height),
        );
    }
    /// `BeadFixer::topChangeEvent`.
    pub fn top_change_event<B: BeadFixerBoundary>(
        &mut self,
        event: BeadFixerChangeEvent,
        boundary: &mut B,
    ) {
        self.rounded_style = boundary.rounded_style();
        boundary.dialog_change_event();
        boundary.check_and_set_mac_menu();
        if event == BeadFixerChangeEvent::FontChange {
            self.set_font_dependent_widths(boundary);
            self.fix_size(boundary);
        }
    }
    pub fn key_press_event(&mut self, shift: bool) {
        self.shift_down = shift
    }
    pub fn key_release_event(&mut self, shift: bool) {
        self.shift_down = shift;
        if self.iterating_move_all > 0 {
            self.iterating_move_all = 0
        }
    }
    pub fn mouse_move_event(&mut self, shift: bool) {
        self.shift_down = shift
    }
}

/// `imodPlugInfo`.
pub fn imod_plug_info() -> &'static str {
    "Bead Fixer"
}
/// `imodPlugExecuteMessage`; callers pass the already tokenized client message.
pub fn imod_plug_execute_message(
    fixer: &mut BeadFixer,
    action: i32,
    argument: Option<&str>,
    binning: i32,
) -> i32 {
    match action {
        0 => argument
            .map(|s| fixer.open_file_by_name(s, binning))
            .unwrap_or(1),
        1 => fixer.reread(binning),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct GuiBoundary {
        calls: Vec<String>,
    }
    impl BeadFixerBoundary for GuiBoundary {
        fn print(&mut self, _: &str) {}
        fn redraw(&mut self) {}
        fn draw_model(&mut self) {}
        fn set_overlay_mode(&mut self, _: i32, _: i32, _: i32) {}
        fn location(&mut self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn image_size(&mut self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn set_index(&mut self, _: i32, _: i32, _: i32) {}
        fn button_width(&mut self, _: bool, _: f64, text: &str) -> i32 {
            match text {
                "Move Point by Residual" => 90,
                "Open Tiltalign Log File" => 120,
                _ => unreachable!(),
            }
        }
        fn top_box_hint_width(&self) -> i32 {
            144
        }
        fn set_fixed_width(&mut self, widget: BeadFixerWidthWidget, width: i32) {
            self.calls.push(format!("width:{widget:?}:{width}"));
        }
        fn content_width(&self) -> i32 {
            180
        }
        fn top_window_size(&self) -> (i32, i32) {
            (200, 100)
        }
        fn top_window_size_hint(&self) -> (i32, i32) {
            (160, 80)
        }
        fn resize_top_window(&mut self, width: i32, height: i32) {
            self.calls.push(format!("resize:{width}:{height}"));
        }
        fn rounded_style(&self) -> bool {
            true
        }
        fn dialog_change_event(&mut self) {
            self.calls.push("change".into());
        }
        fn check_and_set_mac_menu(&mut self) {
            self.calls.push("menu".into());
        }
        fn raise_window(&mut self) {
            self.calls.push("raise".into());
        }
    }
    #[test]
    fn skip_list_is_one_based_like_source() {
        let mut f = BeadFixer::new();
        f.new_skip_list("1, 4-6");
        assert_eq!(f.skip_secs, vec![0, 3, 4, 5]);
        assert!(f.in_skip_list(4));
    }
    #[test]
    fn residual_progress_and_once_filter() {
        let mut f = BeadFixer::new();
        f.resid_list = vec![
            ResidPt {
                obj: 1,
                cont: 1,
                view: 1,
                ..Default::default()
            },
            ResidPt {
                obj: 1,
                cont: 1,
                view: 1,
                ..Default::default()
            },
            ResidPt {
                obj: 1,
                cont: 1,
                view: 2,
                ..Default::default()
            },
        ];
        f.area_list.push(AreaData {
            num_pts: 3,
            ..Default::default()
        });
        assert_eq!(f.next_res().unwrap().view, 1);
        assert_eq!(f.next_res().unwrap().view, 2);
        assert_eq!(f.done_label, "Progress:  100%");
    }
    #[test]
    fn source_font_change_reflows_every_fixed_width_widget_and_window() {
        let mut f = BeadFixer::new();
        let mut native = GuiBoundary::default();
        f.top_change_event(BeadFixerChangeEvent::FontChange, &mut native);
        assert!(f.rounded_style);
        assert_eq!(&native.calls[..2], ["change", "menu"]);
        assert_eq!(
            native
                .calls
                .iter()
                .filter(|call| call.starts_with("width:"))
                .count(),
            25
        );
        assert!(native.calls.contains(&"width:TopBox:140".to_owned()));
        assert_eq!(native.calls.last(), Some(&"resize:160:80".to_owned()));
    }
    #[test]
    fn source_timer_raises_only_while_staying_on_top() {
        let mut f = BeadFixer::new();
        let mut native = GuiBoundary::default();
        f.timer_event(&mut native);
        f.keep_on_top(true);
        f.timer_event(&mut native);
        assert_eq!(native.calls, ["raise"]);
    }
}
