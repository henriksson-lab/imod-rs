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

/// Reasons dispatched to the Bead Fixer plugin by 3dmod.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadFixerPluginReason {
    ModelUpdate,
    NewModel,
    Other,
}

/// Qt-independent inputs consumed by the Bead Fixer plugin callbacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadFixerPluginKey {
    Apostrophe,
    QuoteDbl,
    Space,
    Semicolon,
    Colon,
    U,
    Slash,
    Insert,
    Key9,
    Key3,
    PageUp,
    PageDown,
    Other,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadFixerPluginAction {
    NextResidual,
    NextContour,
    BackResidual,
    BackContour,
    NextGap,
    MovePoint,
    MoveAll,
    MoveAllAll,
    UndoMove,
    ToggleOverlay,
    InsertPoint,
    NextSection,
    PreviousSection,
    ModifyPoint,
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

/// Model data read by `BeadFixer::manageThreshWidgets`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeadThresholdObject {
    pub uses_value: bool,
    pub value_min: f32,
    pub value_max: f32,
    pub valblack: u8,
    pub skip_low: bool,
}

/// Widget state derived by native `manageThreshWidgets`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeadThresholdState {
    pub enabled: bool,
    pub peak_min: i32,
    pub peak_max: i32,
    pub slider_value: i32,
    pub skip_low: bool,
}

/// The contour information `moveToCont` reads from the model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeadContourResidual {
    pub object: i32,
    pub contour: i32,
    pub point_count: usize,
    pub mean_residual: f64,
}

/// Model values consumed by native `deleteBelow`, kept independent of the GUI
/// selection implementation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeadValueContour {
    pub object: i32,
    pub contour: i32,
    pub value: f32,
    pub value_min: f32,
    pub value_max: f32,
    pub valblack: u8,
    pub uses_value: bool,
    pub intersects_selection: bool,
}

/// Contour geometry used by the model-independent portion of `findGap`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeadGapContour {
    pub object: i32,
    pub contour: i32,
    pub points: Vec<BeadPoint>,
}

/// A point adjacent to a missing, non-skipped section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeadGap {
    pub object: i32,
    pub contour: i32,
    pub point: i32,
    pub before: i32,
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
    /// Native visibility operation for a source `showWidget` call.
    fn set_widget_visible(&mut self, _widget: &'static str, _visible: bool) {}
    fn print(&mut self, text: &str);
    fn redraw(&mut self);
    fn draw_model(&mut self);
    fn set_overlay_mode(&mut self, sections: i32, reverse: i32, polarity: i32);
    fn location(&mut self) -> (i32, i32, i32);
    fn image_size(&mut self) -> (i32, i32, i32);
    fn set_index(&mut self, object: i32, contour: i32, point: i32);
    fn point_shift(&mut self, _from: BeadPoint, _to: BeadPoint) {}
    fn delete_current_contour(&mut self) {}
    /// Native `deleteBelow` model/store scan input, after rubber-band or
    /// lasso selection has been evaluated by the image-view host.
    fn bead_value_contours(&mut self) -> Vec<BeadValueContour> {
        Vec::new()
    }
    fn current_object(&mut self) -> i32 {
        -1
    }
    /// Clear/add the native selection list, delete contours in one undo unit,
    /// then clear the selection list as `deleteBelow` does.
    fn delete_below_contours(&mut self, _contours: &[(i32, i32)]) {}
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
    /// `BeadFixerModule()` source constructor.
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
    pub contour_residuals: Vec<BeadContourResidual>,
    pub gap_contours: Vec<BeadGapContour>,
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
            contour_residuals: Vec::new(),
            gap_contours: Vec::new(),
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
    /// `reportContRes`; returns the native report text for the rewritten GUI to
    /// present, and records that the report has been consumed.
    pub fn report_cont_res(&mut self) -> Option<String> {
        if self.globals.filename.is_none()
            || self.globals.show_mode != CONT_MODE
            || self.cont_res_reported
        {
            return None;
        }
        self.cont_res_reported = true;
        if self.num_cont_res <= 0 {
            return Some("\x07No contour mean residual data found.\n".into());
        }
        let count = self.num_cont_res as f64;
        let average = self.cont_res_sum / count;
        let variance = if self.num_cont_res > 1 {
            (self.cont_res_sumsq - self.cont_res_sum * average) / (count - 1.)
        } else {
            0.
        };
        Some(format!(
            "{} contour mean residuals:\n Average {:.2}  SD {:.2}  Max {:.2} {}\n",
            self.num_cont_res,
            average,
            variance.max(0.).sqrt(),
            self.max_cont_res,
            self.unit,
        ))
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
        self.contour_residuals.clear();
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
        let mut contour_block = false;
        for line in text.lines() {
            if line.starts_with("ERROR:") {
                self.last_log_error.push_str(line);
                self.last_log_error.push('\n');
            }
            if contour_block {
                let values: Vec<f64> = line
                    .split_whitespace()
                    .filter_map(|value| value.parse().ok())
                    .collect();
                if values.len() >= 7 {
                    self.contour_residuals.push(BeadContourResidual {
                        object: values[4] as i32 - 1,
                        contour: values[5] as i32 - 1,
                        mean_residual: values[6],
                        ..Default::default()
                    });
                    continue;
                }
                contour_block = false;
            }
            if line.contains(" Z ")
                && line.contains(" obj ")
                && (line.contains("mean resid") || line.contains("resid-"))
            {
                if line.contains("resid-nm") {
                    self.unit = "nm".into();
                }
                contour_block = true;
                continue;
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
        // `reread` breaks equal contour values by small ordered offsets, while
        // retaining the original values for its reported statistics.
        let base_values: Vec<f64> = self
            .contour_residuals
            .iter()
            .map(|contour| contour.mean_residual)
            .collect();
        for (index, value) in base_values.iter().copied().enumerate() {
            if value <= 0. {
                continue;
            }
            self.max_cont_res = self.max_cont_res.max(value);
            self.cont_res_sum += value;
            self.cont_res_sumsq += value * value;
            self.num_cont_res += 1;
            let rank = base_values[..=index]
                .iter()
                .filter(|&&other| other == value)
                .count();
            self.contour_residuals[index].mean_residual = value + rank as f64 * 0.00001;
        }
        if self.num_cont_res > 0 {
            self.last_cont_res = self.max_cont_res + 1.;
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
    /// Native `nextRes` model lookup after a residual record has been chosen.
    /// Coordinates in residual logs are one-based for direct object/contour
    /// listings; model indices returned here are zero-based.
    pub fn locate_residual_in_contours(
        &self,
        residual: &ResidPt,
        contours: &[BeadGapContour],
    ) -> Option<(i32, i32, i32)> {
        let contour = if self.objcont != 0 {
            contours.iter().find(|contour| {
                contour.object == residual.obj - 1 && contour.contour == residual.cont - 1
            })?
        } else {
            let mut number = 0;
            contours.iter().find(|contour| {
                if contour.points.len() > 1 {
                    number += 1;
                }
                number == residual.cont
            })?
        };
        let point = contour.points.iter().position(|point| {
            (point.z + 1.5).floor() as i32 == residual.view
                && (point.x - residual.xcen).powi(2) + (point.y - residual.ycen).powi(2) <= 225.
        })?;
        Some((contour.object, contour.contour, point as i32))
    }
    /// Viewer-bound completion of `nextRes` after selecting a residual.
    pub fn next_res_with_contours<B: BeadFixerBoundary>(
        &mut self,
        contours: &[BeadGapContour],
        boundary: &mut B,
    ) -> Option<ResidPt> {
        let residual = self.next_res()?;
        let (object, contour, point) = self.locate_residual_in_contours(&residual, contours)?;
        self.indlook = self.current_res;
        self.objlook = object;
        self.contlook = contour;
        self.ptlook = point;
        self.curmoved = 0;
        boundary.set_index(object, contour, point);
        boundary.redraw();
        Some(residual)
    }
    pub fn next_local(&mut self) {
        if self.cur_area + 1 < self.area_list.len() as i32 {
            self.current_res = self.area_list[(self.cur_area + 1) as usize].first_pt - 1;
            self.bell = -1;
        }
    }
    /// `backUp`: find the previous eligible residual, make the current one
    /// eligible again, then revisit the selected residual through `nextRes`.
    pub fn back_up(&mut self) -> bool {
        if self.resid_list.is_empty() {
            return false;
        }
        let mut previous = None;
        for index in (0..self.current_res.max(0) as usize).rev() {
            let residual = &self.resid_list[index];
            let eligible_once = self.lookonce == 0 || residual.looked_at != 0;
            let eligible_weight = !self.has_weights
                || self.globals.skip_low_weight == 0
                || residual.weight > self.globals.weight_thresh;
            if eligible_once && eligible_weight {
                previous = Some(index);
                break;
            }
        }
        let Some(previous) = previous else {
            return false;
        };
        if let Some(current) = self.resid_list.get(self.current_res as usize) {
            if let Some(looked) = self.looked_list.iter_mut().find(|looked| {
                looked.obj == current.obj
                    && looked.cont == current.cont
                    && looked.view == current.view
            }) {
                looked.obj = -1;
            }
        }
        let area = self.resid_list[previous].area;
        if area != self.cur_area {
            self.set_cur_area(area);
            self.bell = 1;
        }
        self.current_res = previous as i32 - 1;
        let look_once = self.lookonce;
        self.lookonce = 0;
        let found = self.next_res().is_some();
        self.lookonce = look_once;
        found
    }
    pub fn next_cont(&mut self) {
        self.move_to_cont(1);
    }
    pub fn back_up_cont(&mut self) {
        self.move_to_cont(-1);
    }
    /// `moveToCont`: select the nearest contour residual in the requested
    /// direction.  The GUI host applies the selected index and redraws it.
    pub fn move_to_cont(&mut self, idir: i32) -> i32 {
        self.obj_for_cont_res = -1;
        let mut selected = None;
        for contour in &self.contour_residuals {
            let residual = contour.mean_residual;
            if residual > 0.
                && (idir as f64) * (residual - self.last_cont_res) < 0.
                && selected
                    .as_ref()
                    .map_or(true, |best: &BeadContourResidual| {
                        (idir as f64) * (residual - best.mean_residual) > 0.
                    })
            {
                selected = Some(contour.clone());
            }
        }
        let Some(contour) = selected else {
            return 0;
        };
        self.obj_for_cont_res = contour.object;
        self.cont_for_cont_res = contour.contour;
        self.last_cont_res = contour.mean_residual;
        self.obj_for_cont_res + 1
    }
    /// Viewer-bound portion of `moveToCont`: select the contour midpoint and
    /// request a redraw after the pure residual search above succeeds.
    pub fn move_to_cont_with_boundary<B: BeadFixerBoundary>(
        &mut self,
        idir: i32,
        boundary: &mut B,
    ) -> i32 {
        let result = self.move_to_cont(idir);
        if result <= 0 {
            return result;
        }
        let contour = self
            .contour_residuals
            .iter()
            .find(|contour| {
                contour.object == self.obj_for_cont_res && contour.contour == self.cont_for_cont_res
            })
            .expect("selected contour must remain in contour_residuals");
        boundary.print(&format!(
            "Contour mean residual {:.2} {}\n",
            contour.mean_residual, self.unit
        ));
        boundary.set_index(
            contour.object,
            contour.contour,
            contour.point_count as i32 / 2 - 1,
        );
        boundary.redraw();
        result
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
        if self.curmoved != 0 || self.objlook < 0 || self.indlook < 0 {
            return;
        }
        if let Some(r) = self.resid_list.get(self.indlook as usize) {
            self.oldpt = BeadPoint {
                x: r.xcen,
                y: r.ycen,
                z: r.view as f32,
            };
            self.newpt = BeadPoint {
                x: r.xcen + r.xres,
                y: r.ycen + r.yres,
                z: r.view as f32,
            };
            boundary.point_shift(self.oldpt, self.newpt);
            self.objmoved = self.objlook;
            self.contmoved = self.contlook;
            self.ptmoved = self.ptlook;
            self.curmoved = 1;
            self.didmove = 1;
            boundary.draw_model();
        }
    }
    pub fn undo_move<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        if self.didmove != 0 {
            boundary.point_shift(self.newpt, self.oldpt);
            self.didmove = 0;
            self.curmoved = 0;
            boundary.draw_model();
        }
    }
    pub fn move_all(&mut self, global_ok: bool, _skip_display: bool) {
        self.num_all_moved = 0;
        if (self.cur_area <= 0 && (self.area_list.len() > 1 || !global_ok))
            || self.current_res >= self.resid_list.len() as i32
            || self.resid_list.is_empty()
        {
            return;
        }
        let start_area = self.cur_area;
        self.moving_all = true;
        let end = if global_ok {
            self.resid_list.len()
        } else {
            self.resid_list
                .iter()
                .position(|r| r.area != self.cur_area)
                .unwrap_or(self.resid_list.len())
        };
        while self.cur_area == start_area && (self.current_res + 1) < end as i32 {
            if self.next_res().is_none() {
                break;
            }
            self.num_all_moved += 1;
        }
        self.moving_all = false;
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
                (100. * (self.current_res + 1).clamp(0, self.resid_list.len() as i32) as f64
                    / self.resid_list.len() as f64
                    + 0.5) as i32
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
    pub fn find_gap<B: BeadFixerBoundary>(&mut self, boundary: &mut B, idir: i32) {
        self.ifdidgap = 1;
        let (_, _, zsize) = boundary.image_size();
        if let Some(gap) = self.find_gap_from_last(&self.gap_contours, zsize, idir) {
            self.foundgap(boundary, gap.object, gap.contour, gap.point, gap.before);
        }
    }
    /// Stateful source order for `findGap`: resume at the last reported
    /// point, and retain a contour's point order rather than sorting points
    /// by Z.  The supplied list is the model's object/contour order.
    pub fn find_gap_from_last(
        &self,
        contours: &[BeadGapContour],
        zsize: i32,
        idir: i32,
    ) -> Option<BeadGap> {
        let low = (0..zsize).find(|&z| !self.in_skip_list(z))?;
        let high = (0..zsize).rev().find(|&z| !self.in_skip_list(z))?;
        let forward = idir >= 0;
        let start =
            if self.ifdidgap != 0 && self.lastob >= 0 && self.lastco >= 0 && self.lastpt >= 0 {
                contours.iter().position(|contour| {
                    contour.object == self.lastob && contour.contour == self.lastco
                })
            } else {
                None
            };
        let mut positions: Vec<usize> = if forward {
            (start.unwrap_or(0)..contours.len()).collect()
        } else {
            (0..=start.unwrap_or(contours.len().saturating_sub(1)))
                .rev()
                .collect()
        };
        if contours.is_empty() {
            positions.clear();
        }
        for (number, index) in positions.into_iter().enumerate() {
            let contour = &contours[index];
            if contour.points.is_empty() {
                continue;
            }
            let is_start = Some(index) == start;
            let point_start = if is_start {
                if self.lastbefore != 0 {
                    0
                } else {
                    self.lastpt.clamp(0, contour.points.len() as i32 - 1) as usize
                }
            } else if forward {
                0
            } else {
                contour.points.len() - 1
            };
            let mut min = (contour.points[0].z, 0_usize);
            let mut max = min;
            for (point, value) in contour.points.iter().enumerate() {
                if value.z < min.0 {
                    min = (value.z, point);
                }
                if value.z > max.0 {
                    max = (value.z, point);
                }
            }
            // `lookback` is false only while continuing the same contour.
            if (!is_start || self.ifdidgap == 0 || number > 0) && min.0 > low as f32 + 0.5 {
                return Some(BeadGap {
                    object: contour.object,
                    contour: contour.contour,
                    point: min.1 as i32,
                    before: 1,
                });
            }
            let points: Box<dyn Iterator<Item = usize>> = if forward {
                Box::new(point_start..contour.points.len())
            } else {
                Box::new((0..=point_start).rev())
            };
            for point in points {
                if point == max.1 {
                    continue;
                }
                let mut next_z = (contour.points[point].z + 1.5) as i32;
                while next_z < high && self.in_skip_list(next_z) {
                    next_z += 1;
                }
                if next_z < high
                    && !contour
                        .points
                        .iter()
                        .any(|value| (value.z + 0.5) as i32 == next_z)
                {
                    return Some(BeadGap {
                        object: contour.object,
                        contour: contour.contour,
                        point: point as i32,
                        before: 0,
                    });
                }
            }
            if forward && max.0 + 0.1 < high as f32 {
                return Some(BeadGap {
                    object: contour.object,
                    contour: contour.contour,
                    point: max.1 as i32,
                    before: 0,
                });
            }
            if !forward && min.0 > low as f32 + 0.5 {
                return Some(BeadGap {
                    object: contour.object,
                    contour: contour.contour,
                    point: min.1 as i32,
                    before: 1,
                });
            }
        }
        None
    }
    /// Pure search performed by `findGap`; the caller supplies the model's
    /// contours in model order and the image Z size.
    pub fn find_gap_in_contours(
        &self,
        contours: &[BeadGapContour],
        zsize: i32,
        idir: i32,
    ) -> Option<BeadGap> {
        let low = (0..zsize).find(|&z| !self.in_skip_list(z))?;
        let high = (0..zsize).rev().find(|&z| !self.in_skip_list(z))?;
        let iter: Box<dyn Iterator<Item = &BeadGapContour>> = if idir >= 0 {
            Box::new(contours.iter())
        } else {
            Box::new(contours.iter().rev())
        };
        for contour in iter {
            if contour.points.is_empty() {
                continue;
            }
            let mut indexed: Vec<(usize, i32)> = contour
                .points
                .iter()
                .enumerate()
                .map(|(i, point)| (i, (point.z + 0.5) as i32))
                .collect();
            indexed.sort_by_key(|entry| entry.1);
            if idir < 0 {
                indexed.reverse();
            }
            let (edge_index, edge_z) = indexed[0];
            if (idir >= 0 && edge_z > low) || (idir < 0 && edge_z < high) {
                return Some(BeadGap {
                    object: contour.object,
                    contour: contour.contour,
                    point: edge_index as i32,
                    before: i32::from(idir >= 0),
                });
            }
            for pair in indexed.windows(2) {
                let z = pair[0].1;
                let next = if idir >= 0 { z + 1 } else { z - 1 };
                if next >= low && next <= high && !self.in_skip_list(next) && pair[1].1 != next {
                    return Some(BeadGap {
                        object: contour.object,
                        contour: contour.contour,
                        point: pair[0].0 as i32,
                        before: 0,
                    });
                }
            }
        }
        None
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
    /// Pure value-update portion of native `threshChanged`.  The nudge keeps
    /// a real byte-valued property change for a slider move between adjacent
    /// representable threshold values.
    pub fn threshold_black_value(&mut self, value: i32, current: u8) -> u8 {
        let range = self.peak_max - self.peak_min;
        if range == 0 {
            return current;
        }
        let mut black = (255. * (value - self.peak_min) as f64 / range as f64 + 0.5) as i32;
        if black == current as i32 && value != self.last_thresh {
            black += if value < self.last_thresh { -1 } else { 1 };
        }
        self.last_thresh = value;
        black.clamp(u8::MIN as i32, u8::MAX as i32) as u8
    }
    /// Select contours that native `deleteBelow` would delete.  The GUI host
    /// supplies whether each contour lies in its lasso/section selection and
    /// performs the undoable deletion.
    pub fn delete_below_from_contours(
        &self,
        contours: &[BeadValueContour],
        current_object: i32,
    ) -> Vec<(i32, i32)> {
        contours
            .iter()
            .filter(|contour| {
                (contour.object == current_object || self.globals.del_in_all_obj != 0)
                // The native store loop explicitly requires `index.contour > 0`.
                && contour.contour > 0 && contour.uses_value && contour.intersects_selection
                && contour.value < contour.valblack as f32
                    * (contour.value_max - contour.value_min) / 255. + contour.value_min
            })
            .map(|contour| (contour.object, contour.contour))
            .collect()
    }
    /// `deleteBelow`.
    pub fn delete_below<B: BeadFixerBoundary>(&mut self, boundary: &mut B) {
        let current_object = boundary.current_object();
        let contours = boundary.bead_value_contours();
        let selected = self.delete_below_from_contours(&contours, current_object);
        if !selected.is_empty() {
            boundary.delete_below_contours(&selected);
        }
    }
    pub fn del_all_sec_toggled(&mut self, state: bool) {
        self.globals.del_on_all_sec = i32::from(state)
    }
    pub fn del_all_obj_toggled(&mut self, state: bool) {
        self.globals.del_in_all_obj = i32::from(state)
    }
    pub fn turn_off_toggled<B: BeadFixerBoundary>(&mut self, b: &mut B, _state: bool) {
        b.draw_model()
    }
    /// Object-property portion of `turnOffToggled`.
    pub fn turn_off_matflags(matflags2: u32, state: bool) -> u32 {
        let bit = crate::imod::libimod::iobj::MATFLAGS2_SKIP_LOW;
        if state {
            matflags2 | bit
        } else {
            matflags2 & !bit
        }
    }
    /// `makeUpDownArrow()` source method.  Clearing/configuring the existing
    /// extra object is viewer ownership; this preserves the source's five
    /// contour vertices for its yellow open direction arrow.
    pub fn make_up_down_arrow(current: BeadPoint, before: bool) -> [BeadPoint; 5] {
        let size = 12.0;
        let idir = if before { -1.0 } else { 1.0 };
        let mut point = current;
        point.y += idir * size / 2.0;
        let first = point;
        point.y += idir * size;
        let second = point;
        point.x -= idir * size / 3.0;
        point.y -= idir * size / 3.0;
        let third = point;
        point.x += idir * size / 3.0;
        point.y += idir * size / 3.0;
        let fourth = point;
        point.x += idir * size / 3.0;
        point.y -= idir * size / 3.0;
        [first, second, third, fourth, point]
    }
    /// `showWidget()` source helper, represented by the replacement GUI boundary.
    fn show_widget<B: BeadFixerBoundary>(b: &mut B, widget: &'static str, visible: bool) {
        b.set_widget_visible(widget, visible)
    }
    /// Native mode transition.  The returned contour report is the source's
    /// post-transition `reportContRes` output.
    pub fn mode_selected<B: BeadFixerBoundary>(&mut self, b: &mut B, value: i32) -> Option<String> {
        let value = value.clamp(SEED_MODE, CONT_MODE);
        let res_or_cont = value == RES_MODE || value == CONT_MODE;
        for (widget, visible) in [
            ("seedModeBox", value == SEED_MODE),
            ("overlayHbox", value == SEED_MODE),
            ("reverseBox", value == SEED_MODE),
            ("ignoreSkipBut", value == GAP_MODE),
            ("skipEdit", value == GAP_MODE),
            ("nextGapBut", value == GAP_MODE),
            ("prevGapBut", value == GAP_MODE),
            ("reattachBut", value == GAP_MODE),
            ("resetStartBut", value == GAP_MODE),
            ("resetCurrentBut", value == GAP_MODE),
            ("cenLightHbox", !res_or_cont),
            ("diameterHbox", !res_or_cont),
            ("openFileBut", res_or_cont),
            ("runAlignBut", res_or_cont),
            ("rereadBut", res_or_cont),
            ("nextResBut", value == RES_MODE),
            ("backUpBut", value == RES_MODE),
            ("nextLocalBut", value == RES_MODE),
            ("movePointBut", value == RES_MODE),
            ("undoMoveBut", value == RES_MODE),
            ("moveAllBut", value == RES_MODE),
            ("moveAllAllBut", value == RES_MODE),
            ("clearListBut", value == RES_MODE),
            ("examineBox", value == RES_MODE),
            ("doneLabel", value == RES_MODE),
            ("weightHbox", value == RES_MODE),
            ("skipLowWgtBox", value == RES_MODE),
            ("nextContBut", value == CONT_MODE),
            ("backContBut", value == CONT_MODE),
            ("delContBut", value == CONT_MODE),
        ] {
            Self::show_widget(b, widget, visible);
        }
        if (value == SEED_MODE || self.globals.show_mode == SEED_MODE)
            && self.globals.overlay_on != 0
        {
            self.set_overlay(b, 1, i32::from(value == SEED_MODE));
        }
        self.globals.show_mode = value;
        self.report_cont_res()
    }
    pub fn model_update<B: BeadFixerBoundary>(&mut self, b: &mut B) {
        b.model_update()
    }
    /// Pure state portion of `manageThreshWidgets`; the Qt show/range/value
    /// calls consume the returned state at the host boundary.
    pub fn manage_thresh_widgets_for_object(
        &mut self,
        seed_mode: bool,
        object: Option<BeadThresholdObject>,
    ) -> BeadThresholdState {
        let Some(object) = object.filter(|object| object.uses_value && seed_mode) else {
            return BeadThresholdState::default();
        };
        // `B3DNINT` is the source's `int(value + .5)` conversion.
        self.peak_min = (object.value_min * 1000. + 0.5) as i32;
        self.peak_max = (object.value_max * 1000. + 0.5) as i32;
        self.last_thresh = ((object.valblack as f64 * (self.peak_max - self.peak_min) as f64
            / 255.
            + self.peak_min as f64)
            + 0.5) as i32;
        BeadThresholdState {
            enabled: true,
            peak_min: self.peak_min,
            peak_max: self.peak_max,
            slider_value: self.last_thresh,
            skip_low: object.skip_low,
        }
    }
    pub fn manage_thresh_widgets(&mut self, seed_mode: bool) -> BeadThresholdState {
        self.manage_thresh_widgets_for_object(seed_mode, None)
    }
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
/// `imodPlugExecuteType()` source plugin callback.
pub fn imod_plug_execute_type<B: BeadFixerBoundary>(
    fixer: Option<&mut BeadFixer>,
    reason: BeadFixerPluginReason,
    boundary: &mut B,
) {
    let Some(fixer) = fixer else { return };
    match reason {
        BeadFixerPluginReason::ModelUpdate => fixer.model_update(boundary),
        BeadFixerPluginReason::NewModel => fixer.lastob = -1,
        BeadFixerPluginReason::Other => {}
    }
}
/// `imodPlugKeys()` source plugin callback.  The caller supplies the model-mode
/// and cursor-point prerequisites owned by the replacement viewer.
pub fn imod_plug_keys(
    fixer: Option<&mut BeadFixer>,
    key: BeadFixerPluginKey,
    ctrl: bool,
    shift: bool,
    keypad: bool,
    model_mode: bool,
) -> Option<BeadFixerPluginAction> {
    let fixer = fixer?;
    match key {
        BeadFixerPluginKey::Apostrophe
            if matches!(fixer.globals.show_mode, RES_MODE | CONT_MODE) =>
        {
            Some(if fixer.globals.show_mode == RES_MODE {
                BeadFixerPluginAction::NextResidual
            } else {
                BeadFixerPluginAction::NextContour
            })
        }
        BeadFixerPluginKey::QuoteDbl if matches!(fixer.globals.show_mode, RES_MODE | CONT_MODE) => {
            Some(if fixer.globals.show_mode == RES_MODE {
                BeadFixerPluginAction::BackResidual
            } else {
                BeadFixerPluginAction::BackContour
            })
        }
        BeadFixerPluginKey::Space if fixer.globals.show_mode == GAP_MODE => {
            Some(BeadFixerPluginAction::NextGap)
        }
        BeadFixerPluginKey::Semicolon if fixer.globals.show_mode == RES_MODE => {
            Some(BeadFixerPluginAction::MovePoint)
        }
        BeadFixerPluginKey::Colon if fixer.globals.show_mode == RES_MODE => {
            fixer.iterating_move_all = -1;
            Some(if ctrl {
                BeadFixerPluginAction::MoveAllAll
            } else {
                BeadFixerPluginAction::MoveAll
            })
        }
        BeadFixerPluginKey::U if fixer.globals.show_mode == RES_MODE => {
            Some(BeadFixerPluginAction::UndoMove)
        }
        BeadFixerPluginKey::Slash if fixer.globals.show_mode == SEED_MODE => {
            fixer.globals.overlay_on = 1 - fixer.globals.overlay_on;
            Some(BeadFixerPluginAction::ToggleOverlay)
        }
        BeadFixerPluginKey::Insert if keypad && model_mode => {
            Some(BeadFixerPluginAction::InsertPoint)
        }
        BeadFixerPluginKey::Key9 if keypad && shift => Some(BeadFixerPluginAction::NextSection),
        BeadFixerPluginKey::Key3 if keypad && shift => Some(BeadFixerPluginAction::PreviousSection),
        BeadFixerPluginKey::PageUp => Some(BeadFixerPluginAction::NextSection),
        BeadFixerPluginKey::PageDown => Some(BeadFixerPluginAction::PreviousSection),
        _ => None,
    }
}
/// `imodPlugMouse()` source plugin callback.
pub fn imod_plug_mouse(
    fixer: Option<&BeadFixer>,
    model_mode: bool,
    press: bool,
    middle: bool,
    right: bool,
    shift: bool,
) -> Option<BeadFixerPluginAction> {
    let fixer = fixer?;
    if !model_mode || !press {
        return None;
    }
    if middle {
        Some(BeadFixerPluginAction::InsertPoint)
    } else if right
        && shift
        && fixer.globals.auto_center != 0
        && fixer.globals.show_mode != RES_MODE
    {
        Some(BeadFixerPluginAction::ModifyPoint)
    } else {
        None
    }
}
/// `executeMessage()` source method.
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
        fn print(&mut self, text: &str) {
            self.calls.push(format!("print:{text}"));
        }
        fn redraw(&mut self) {
            self.calls.push("redraw".into());
        }
        fn draw_model(&mut self) {}
        fn model_update(&mut self) {
            self.calls.push("model_update".into());
        }
        fn set_overlay_mode(&mut self, sections: i32, reverse: i32, polarity: i32) {
            self.calls
                .push(format!("overlay:{sections}:{reverse}:{polarity}"));
        }
        fn location(&mut self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn image_size(&mut self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn set_index(&mut self, object: i32, contour: i32, point: i32) {
            self.calls.push(format!("index:{object}:{contour}:{point}"));
        }
        fn point_shift(&mut self, from: BeadPoint, to: BeadPoint) {
            self.calls.push(format!(
                "shift:{:.1}:{:.1}:{:.1}:{:.1}",
                from.x, from.y, to.x, to.y
            ));
        }
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
    fn threshold_widget_state_uses_native_peak_and_slider_conversions() {
        let mut f = BeadFixer::new();
        let state = f.manage_thresh_widgets_for_object(
            true,
            Some(BeadThresholdObject {
                uses_value: true,
                value_min: 1.234,
                value_max: 4.567,
                valblack: 128,
                skip_low: true,
            }),
        );
        assert_eq!(
            state,
            BeadThresholdState {
                enabled: true,
                peak_min: 1234,
                peak_max: 4567,
                slider_value: 2907,
                skip_low: true,
            }
        );
        assert_eq!(
            f.manage_thresh_widgets_for_object(
                false,
                Some(BeadThresholdObject {
                    uses_value: true,
                    ..Default::default()
                })
            ),
            BeadThresholdState::default()
        );
    }
    #[test]
    fn threshold_change_uses_native_byte_rounding_and_duplicate_nudge() {
        let mut f = BeadFixer::new();
        f.peak_min = 0;
        f.peak_max = 1000;
        f.last_thresh = 500;
        assert_eq!(f.threshold_black_value(600, 128), 153);
        f.peak_max = 10000;
        f.last_thresh = 5000;
        // 5001 maps back to 128, so the source forces a visible increment.
        assert_eq!(f.threshold_black_value(5001, 128), 129);
    }
    #[test]
    fn turn_off_preserves_other_material_flags() {
        assert_eq!(BeadFixer::turn_off_matflags(0b100, true), 0b101);
        assert_eq!(BeadFixer::turn_off_matflags(0b101, false), 0b100);
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
    fn progress_label_uses_native_nearest_integer_rounding() {
        let mut f = BeadFixer::new();
        f.resid_list = vec![ResidPt::default(), ResidPt::default(), ResidPt::default()];
        f.current_res = 1;
        f.manage_done_label();
        assert_eq!(f.done_label, "Progress:  67%");
    }
    #[test]
    fn residual_lookup_uses_one_based_log_ids_z_and_distance_limit() {
        let mut f = BeadFixer::new();
        f.objcont = 1;
        let contours = vec![BeadGapContour {
            object: 0,
            contour: 1,
            points: vec![
                BeadPoint {
                    x: 20.,
                    y: 30.,
                    z: 4.,
                },
                BeadPoint {
                    x: 50.,
                    y: 30.,
                    z: 4.,
                },
            ],
        }];
        let residual = ResidPt {
            obj: 1,
            cont: 2,
            view: 5,
            xcen: 21.,
            ycen: 29.,
            ..Default::default()
        };
        assert_eq!(
            f.locate_residual_in_contours(&residual, &contours),
            Some((0, 1, 0))
        );
        let too_far = ResidPt {
            xcen: 80.,
            ..residual
        };
        assert_eq!(f.locate_residual_in_contours(&too_far, &contours), None);
    }
    #[test]
    fn next_res_with_contours_selects_the_resolved_model_point() {
        let mut f = BeadFixer::new();
        f.objcont = 1;
        f.resid_list = vec![ResidPt {
            obj: 1,
            cont: 1,
            view: 1,
            xcen: 3.,
            ycen: 4.,
            ..Default::default()
        }];
        let contours = vec![BeadGapContour {
            object: 0,
            contour: 0,
            points: vec![BeadPoint {
                x: 3.,
                y: 4.,
                z: 0.,
            }],
        }];
        let mut native = GuiBoundary::default();
        assert!(f.next_res_with_contours(&contours, &mut native).is_some());
        assert_eq!((f.indlook, f.objlook, f.contlook, f.ptlook), (0, 0, 0, 0));
        assert!(native.calls.contains(&"index:0:0:0".to_owned()));
    }
    #[test]
    fn back_up_revisits_the_previous_eligible_residual() {
        let mut f = BeadFixer::new();
        f.resid_list = vec![
            ResidPt {
                obj: 1,
                cont: 1,
                view: 1,
                looked_at: 1,
                area: 0,
                ..Default::default()
            },
            ResidPt {
                obj: 2,
                cont: 1,
                view: 1,
                looked_at: 1,
                area: 1,
                ..Default::default()
            },
        ];
        f.area_list = vec![
            AreaData {
                first_pt: 0,
                num_pts: 1,
                ..Default::default()
            },
            AreaData {
                first_pt: 1,
                num_pts: 1,
                ..Default::default()
            },
        ];
        f.looked_list = vec![
            LookedPt {
                obj: 1,
                cont: 1,
                view: 1,
            },
            LookedPt {
                obj: 2,
                cont: 1,
                view: 1,
            },
        ];
        f.current_res = 1;
        f.cur_area = 1;
        f.lookonce = 1;

        assert!(f.back_up());
        assert_eq!(f.current_res, 0);
        assert_eq!(f.cur_area, 0);
        assert_eq!(f.bell, 1);
        assert_eq!(f.looked_list[1].obj, -1);
    }
    #[test]
    fn move_point_applies_residual_offset_from_fitted_center() {
        let mut f = BeadFixer::new();
        f.resid_list = vec![ResidPt {
            xcen: 10.,
            ycen: 20.,
            xres: 1.5,
            yres: -2.,
            view: 4,
            ..Default::default()
        }];
        f.indlook = 0;
        f.objlook = 0;
        let mut native = GuiBoundary::default();

        f.move_point(&mut native);
        assert_eq!(
            f.newpt,
            BeadPoint {
                x: 11.5,
                y: 18.,
                z: 4.
            }
        );
        assert!(
            native
                .calls
                .contains(&"shift:10.0:20.0:11.5:18.0".to_owned())
        );
        assert_eq!(f.curmoved, 1);
    }
    #[test]
    fn move_point_requires_an_active_residual_selection() {
        let mut f = BeadFixer::new();
        f.resid_list = vec![ResidPt {
            xcen: 10.,
            ..Default::default()
        }];
        let mut native = GuiBoundary::default();
        f.move_point(&mut native);
        assert_eq!(f.didmove, 0);
        assert!(native.calls.is_empty());
    }
    #[test]
    fn move_all_rejects_global_residuals_when_local_areas_exist() {
        let mut f = BeadFixer::new();
        f.resid_list = vec![ResidPt::default()];
        f.area_list = vec![AreaData::default(), AreaData::default()];
        f.current_res = -1;
        f.move_all(true, false);
        assert_eq!(f.current_res, -1);
        assert_eq!(f.num_all_moved, 0);
    }
    #[test]
    fn delete_below_filters_values_objects_and_selection_like_native() {
        let mut f = BeadFixer::new();
        f.globals.del_in_all_obj = 0;
        let contours = vec![
            BeadValueContour {
                object: 1,
                contour: 2,
                value: 3.,
                value_min: 0.,
                value_max: 10.,
                valblack: 128,
                uses_value: true,
                intersects_selection: true,
            },
            BeadValueContour {
                object: 1,
                contour: 0,
                value: 1.,
                value_min: 0.,
                value_max: 10.,
                valblack: 128,
                uses_value: true,
                intersects_selection: true,
            },
            BeadValueContour {
                object: 1,
                contour: 3,
                value: 8.,
                value_min: 0.,
                value_max: 10.,
                valblack: 128,
                uses_value: true,
                intersects_selection: true,
            },
            BeadValueContour {
                object: 2,
                contour: 1,
                value: 2.,
                value_min: 0.,
                value_max: 10.,
                valblack: 128,
                uses_value: true,
                intersects_selection: true,
            },
        ];
        assert_eq!(f.delete_below_from_contours(&contours, 1), vec![(1, 2)]);
        f.globals.del_in_all_obj = 1;
        assert_eq!(
            f.delete_below_from_contours(&contours, 1),
            vec![(1, 2), (2, 1)]
        );
    }
    #[test]
    fn gap_search_skips_excluded_sections_and_finds_missing_section() {
        let mut f = BeadFixer::new();
        f.new_skip_list("3");
        let contours = vec![BeadGapContour {
            object: 1,
            contour: 2,
            points: vec![
                BeadPoint {
                    z: 0.,
                    ..Default::default()
                },
                BeadPoint {
                    z: 3.,
                    ..Default::default()
                },
            ],
        }];
        assert_eq!(
            f.find_gap_in_contours(&contours, 5, 1),
            Some(BeadGap {
                object: 1,
                contour: 2,
                point: 0,
                before: 0,
            })
        );
    }
    #[test]
    fn gap_search_resumes_at_last_reported_model_point() {
        let mut f = BeadFixer::new();
        f.ifdidgap = 1;
        f.lastob = 1;
        f.lastco = 2;
        f.lastpt = 1;
        let contours = vec![BeadGapContour {
            object: 1,
            contour: 2,
            points: vec![
                BeadPoint {
                    z: 0.,
                    ..Default::default()
                },
                BeadPoint {
                    z: 1.,
                    ..Default::default()
                },
                BeadPoint {
                    z: 3.,
                    ..Default::default()
                },
            ],
        }];
        assert_eq!(
            f.find_gap_from_last(&contours, 5, 1),
            Some(BeadGap {
                object: 1,
                contour: 2,
                point: 1,
                before: 0,
            })
        );
    }
    #[test]
    fn gap_search_does_not_repeat_a_reported_gap_before_contour_start() {
        let mut f = BeadFixer::new();
        f.ifdidgap = 1;
        f.lastob = 1;
        f.lastco = 2;
        f.lastpt = 0;
        f.lastbefore = 1;
        let contours = vec![BeadGapContour {
            object: 1,
            contour: 2,
            points: vec![
                BeadPoint {
                    z: 2.,
                    ..Default::default()
                },
                BeadPoint {
                    z: 3.,
                    ..Default::default()
                },
            ],
        }];
        assert_eq!(f.find_gap_from_last(&contours, 4, 1), None);
    }
    #[test]
    fn contour_navigation_selects_the_nearest_residual_in_each_direction() {
        let mut f = BeadFixer::new();
        f.contour_residuals = vec![
            BeadContourResidual {
                object: 2,
                contour: 4,
                point_count: 10,
                mean_residual: 2.0,
            },
            BeadContourResidual {
                object: 1,
                contour: 3,
                point_count: 7,
                mean_residual: 4.0,
            },
            BeadContourResidual {
                object: 3,
                contour: 2,
                point_count: 9,
                mean_residual: 6.0,
            },
        ];
        f.last_cont_res = 5.0;
        let mut native = GuiBoundary::default();

        assert_eq!(f.move_to_cont_with_boundary(1, &mut native), 2);
        assert_eq!(
            (f.obj_for_cont_res, f.cont_for_cont_res, f.last_cont_res),
            (1, 3, 4.0)
        );
        assert!(native.calls.contains(&"index:1:3:2".to_owned()));
        assert_eq!(f.move_to_cont(-1), 4);
        assert_eq!(
            (f.obj_for_cont_res, f.cont_for_cont_res, f.last_cont_res),
            (3, 2, 6.0)
        );
    }
    #[test]
    fn contour_residual_report_uses_sample_standard_deviation_once() {
        let mut f = BeadFixer::new();
        f.globals.filename = Some("resid.log".into());
        f.globals.show_mode = CONT_MODE;
        f.num_cont_res = 2;
        f.cont_res_sum = 6.0;
        f.cont_res_sumsq = 20.0;
        f.max_cont_res = 4.0;

        assert_eq!(
            f.report_cont_res(),
            Some("2 contour mean residuals:\n Average 3.00  SD 1.41  Max 4.00 pixels\n".into())
        );
        assert_eq!(f.report_cont_res(), None);
    }
    #[test]
    fn reread_ingests_contour_mean_residual_block_and_breaks_ties() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-beadfix-contour-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::write(
            &path,
            " Z  obj mean resid\n1 0 0 0 1 2 3.5\n2 0 0 0 2 3 3.5\n\n",
        )
        .unwrap();
        let mut f = BeadFixer::new();
        f.globals.filename = Some(path.to_string_lossy().into_owned());
        assert_eq!(f.reread(1), 0);
        std::fs::remove_file(path).unwrap();
        assert_eq!(f.num_cont_res, 2);
        assert_eq!(f.cont_res_sum, 7.0);
        assert_eq!(f.max_cont_res, 3.5);
        assert_eq!(f.last_cont_res, 4.5);
        assert_eq!(f.contour_residuals[0].mean_residual, 3.50001);
        assert_eq!(f.contour_residuals[1].mean_residual, 3.50002);
    }
    #[test]
    fn mode_change_disables_seed_overlay_and_reports_contours() {
        let mut f = BeadFixer::new();
        f.globals.overlay_on = 1;
        f.globals.overlay_sec = 3;
        f.globals.filename = Some("resid.log".into());
        f.num_cont_res = 1;
        f.cont_res_sum = 2.0;
        f.cont_res_sumsq = 4.0;
        f.max_cont_res = 2.0;
        let mut native = GuiBoundary::default();

        let report = f.mode_selected(&mut native, CONT_MODE);
        assert_eq!(native.calls, ["overlay:0:0:0"]);
        assert!(report.unwrap().starts_with("1 contour mean residuals:"));
        assert_eq!(f.globals.show_mode, CONT_MODE);
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
    #[test]
    fn plugin_execute_type_routes_model_and_new_model_reasons() {
        let mut f = BeadFixer::new();
        f.lastob = 4;
        let mut native = GuiBoundary::default();
        imod_plug_execute_type(
            Some(&mut f),
            BeadFixerPluginReason::ModelUpdate,
            &mut native,
        );
        imod_plug_execute_type(Some(&mut f), BeadFixerPluginReason::NewModel, &mut native);
        assert_eq!(f.lastob, -1);
        assert_eq!(native.calls, ["model_update"]);
    }
    #[test]
    fn gap_arrow_matches_source_five_point_geometry() {
        let points = BeadFixer::make_up_down_arrow(
            BeadPoint {
                x: 10.,
                y: 20.,
                z: 3.,
            },
            false,
        );
        assert_eq!(
            points[0],
            BeadPoint {
                x: 10.,
                y: 26.,
                z: 3.
            }
        );
        assert_eq!(
            points[1],
            BeadPoint {
                x: 10.,
                y: 38.,
                z: 3.
            }
        );
        assert_eq!(
            points[4],
            BeadPoint {
                x: 14.,
                y: 34.,
                z: 3.
            }
        );
        let up = BeadFixer::make_up_down_arrow(BeadPoint::default(), true);
        assert_eq!(up[0].y, -6.);
        assert_eq!(
            up[4],
            BeadPoint {
                x: -4.,
                y: -14.,
                z: 0.
            }
        );
    }
}
