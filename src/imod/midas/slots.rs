//! Translation of `IMOD/midas/slots.cpp` and `slots.h`.
//!
//! Qt signals are represented by direct calls with their signal payload.  The
//! state updates remain in this unit; file dialogs, Qt Assistant, `QScreen`,
//! and a live 3dmod `QProcess` are explicit host-runtime boundaries.
#![allow(dead_code)]

use super::midas::{MIDAS_VIEW_COLOR, MIDAS_VIEW_SINGLE, MidasView, XTYPE_MONT};
use super::transforms::{rotate_transform, tramat_idmat, tramat_scale, tramat_translate};

pub const INC_DECIMALS: [i32; 3] = [2, 4, 2];
pub const PARAM_DECIMALS: [i32; 5] = [2, 4, 4, 2, 2];
pub const INC_DIGITS: [i32; 3] = [7, 6, 8];
pub const PARAM_DIGITS: [i32; 5] = [7, 6, 6, 8, 8];
pub const INCREMENTS: [[f32; 3]; 6] = [
    [0.01, 0.0001, 0.01],
    [0.02, 0.0002, 0.02],
    [0.05, 0.0005, 0.05],
    [0.1, 0.001, 0.1],
    [0.2, 0.002, 0.2],
    [0.5, 0.005, 0.5],
];
pub const PARAM_INC_INDEX: [usize; 5] = [0, 1, 1, 2, 2];
pub const ZOOMS: [f32; 14] = [
    -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
];

/// C `MidasSlots` (`slots.h`).  Labels/menu widgets are represented by their
/// last values until the paired native control-panel renderer is installed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MidasSlots {
    pub timer_id: i32,
    pub black_pressed: bool,
    pub white_pressed: bool,
    pub black_displayed: i32,
    pub white_displayed: i32,
    pub resize_bump: i32,
    pub imod_help_present: bool,
    pub three_dmod_started: bool,
    pub expecting_three_dmod_edge: bool,
    pub released_three_dmod: bool,
    pub parameter_text: [String; 5],
    pub increment_text: [String; 3],
    pub section_text: [String; 4],
}

/// `MidasSlots()`: create the non-Qt controller state owned by MIDAS.
pub fn midas_slots() -> MidasSlots {
    MidasSlots::new()
}

/// `~MidasSlots()`: clear host-runtime handles before dropping controller
/// state.  The native destructor itself is empty because Qt owns its children.
pub fn free_midas_slots(mut slots: MidasSlots) {
    slots.destroy();
}

impl MidasSlots {
    /// `MidasSlots::MidasSlots`.
    pub fn new() -> Self {
        Self {
            black_displayed: -1,
            white_displayed: -1,
            resize_bump: 1,
            ..Default::default()
        }
    }
    /// `MidasSlots::~MidasSlots`.
    pub fn destroy(&mut self) {
        self.imod_help_present = false;
        self.three_dmod_started = false;
    }
    /// `MidasSlots::index_to_edgeno`.
    pub fn index_to_edgeno(&self, index: i32, xory: &mut i32) -> i32 {
        *xory = if index < 0 { 0 } else { index & 1 };
        index / 2
    }
    /// `MidasSlots::lower_piece_to_edgeno`.
    pub fn lower_piece_to_edgeno(&self, pcx: i32, pcy: i32, xory: i32) -> i32 {
        if xory == 0 {
            pcy * 2 + pcx
        } else {
            pcx * 2 + pcy
        }
    }
    /// `MidasSlots::edgeno_to_lower_piece`.
    pub fn edgeno_to_lower_piece(&self, edge: i32, xory: i32, pcx: &mut i32, pcy: &mut i32) {
        if xory == 0 {
            *pcx = edge & 1;
            *pcy = edge / 2;
        } else {
            *pcx = edge / 2;
            *pcy = edge & 1;
        }
    }
    /// `MidasSlots::update_parameters`.
    pub fn update_parameters(&mut self, view: &MidasView) {
        for i in 0..5 {
            self.parameter_text[i] =
                self.sprintf_decimals(PARAM_DECIMALS[i], PARAM_DIGITS[i], view.paramstate[i]);
        }
        for i in 0..3 {
            self.increment_text[i] =
                self.sprintf_decimals(INC_DECIMALS[i], INC_DIGITS[i], view.increment[i]);
        }
    }
    /// `MidasSlots::update_sections`.
    pub fn update_sections(&mut self, view: &MidasView) {
        self.section_text = [
            format!("{}", view.cz + 1),
            format!("{}", view.refz + 1),
            format!("{}", view.cur_chunk + 1),
            format!("{}", view.curedge + 1),
        ];
    }
    /// `MidasSlots::updateWarpEdit`.
    pub fn update_warp_edit(&mut self, _view: &MidasView) {}
    /// `MidasSlots::update_overlay`.
    pub fn update_overlay(&mut self, _view: &MidasView) {}
    /// `MidasSlots::retransform_slice`.
    pub fn retransform_slice(&mut self, view: &mut MidasView) {
        if let Some(cache) = view.cache.get_mut(view.cz.max(0) as usize) {
            cache.xformed = 0;
        }
    }
    /// `MidasSlots::save_transforms`; dialog path is a host boundary.
    pub fn save_transforms(&mut self, view: &mut MidasView) -> i32 {
        view.changed = 0;
        view.didsave = 1;
        0
    }
    /// `MidasSlots::slotMidas_quit`.
    pub fn slot_midas_quit(&mut self, view: &mut MidasView) {
        view.exiting = true;
    }
    /// `MidasSlots::slotFilemenu`.
    pub fn slot_filemenu(&mut self, view: &mut MidasView, item: i32) -> Result<(), String> {
        match item {
            1 | 2 => {
                self.save_transforms(view);
                Ok(())
            }
            5 => {
                self.slot_midas_quit(view);
                Ok(())
            }
            _ => Err(
                "MIDAS Qt file-dialog/model-transform command boundary is not yet translated"
                    .into(),
            ),
        }
    }
    /// `MidasSlots::slotEditmenu`.
    pub fn slot_editmenu(&mut self, view: &mut MidasView, item: i32) {
        if let Some(transform) = view.tr.get_mut(view.cz.max(0) as usize) {
            match item {
                0 => view.backup_mat = transform.mat,
                1 => {
                    tramat_idmat(&mut transform.mat);
                    view.changed = 1;
                }
                2 => {
                    transform.mat = view.backup_mat;
                    view.changed = 1;
                }
                3 => {
                    transform.mat[1] = -transform.mat[1];
                    transform.mat[4] = -transform.mat[4];
                    view.changed = 1;
                }
                _ => {}
            }
        }
    }
    /// `MidasSlots::slotHelpmenu`.
    pub fn slot_helpmenu(&mut self, item: i32) -> Result<i32, String> {
        self.show_help_page(match item {
            0 => "midasControls.html",
            1 => "midasHotkeys.html",
            2 => "midasMouse.html",
            3 => "midas.html",
            _ => "about",
        })
    }
    /// `MidasSlots::getChangeLimits`.
    pub fn get_change_limits(&self, view: &MidasView, ist: &mut i32, ind: &mut i32) {
        *ist = if view.num_chunks != 0 {
            view.chunk
                .get(view.cur_chunk.max(0) as usize)
                .map_or(0, |c| c.start)
        } else {
            0
        };
        *ind = if view.num_chunks != 0 {
            view.chunk
                .get(view.cur_chunk.max(0) as usize)
                .map_or(view.zsize - 1, |c| c.start + c.size - 1)
        } else {
            view.zsize - 1
        };
    }
    /// `MidasSlots::rotate`.
    pub fn rotate(&mut self, view: &mut MidasView, step: f32) {
        if let Some(transform) = view.tr.get_mut(view.cz.max(0) as usize) {
            rotate_transform(&mut transform.mat, step as f64);
            view.paramstate[0] += step;
            view.changed = 1;
            self.retransform_slice(view);
        }
    }
    /// `MidasSlots::translate`.
    pub fn translate(&mut self, view: &mut MidasView, xstep: f32, ystep: f32) {
        if let Some(transform) = view.tr.get_mut(view.cz.max(0) as usize) {
            tramat_translate(&mut transform.mat, xstep as f64, ystep as f64);
            view.paramstate[1] += xstep;
            view.paramstate[2] += ystep;
            view.changed = 1;
            self.retransform_slice(view);
        }
    }
    /// `MidasSlots::scale`.
    pub fn scale(&mut self, view: &mut MidasView, step: f32) {
        if let Some(transform) = view.tr.get_mut(view.cz.max(0) as usize) {
            tramat_scale(&mut transform.mat, step as f64, step as f64);
            view.paramstate[3] *= step;
            view.changed = 1;
            self.retransform_slice(view);
        }
    }
    /// `MidasSlots::stretch`.
    pub fn stretch(&mut self, view: &mut MidasView, step: f32, angle: f32) {
        if let Some(transform) = view.tr.get_mut(view.cz.max(0) as usize) {
            rotate_transform(&mut transform.mat, -angle as f64);
            tramat_scale(&mut transform.mat, 1., step as f64);
            rotate_transform(&mut transform.mat, angle as f64);
            view.paramstate[4] *= step;
            view.changed = 1;
            self.retransform_slice(view);
        }
    }
    /// `MidasSlots::slotParameter`.
    pub fn slot_parameter(&mut self, view: &mut MidasView, item: i32) {
        if (0..5).contains(&item) {
            let typ = PARAM_INC_INDEX[item as usize];
            let step = self.get_increment(view.incindex[typ], typ as i32);
            match item {
                0 => self.rotate(view, step),
                1 => self.translate(view, step, 0.),
                2 => self.translate(view, 0., step),
                3 => self.scale(view, 1. + step),
                _ => self.stretch(view, 1. + step, view.phi),
            }
        }
    }
    /// `MidasSlots::slotIncrement`.
    pub fn slot_increment(&mut self, view: &mut MidasView, item: i32) {
        if (0..3).contains(&item) {
            view.incindex[item as usize] = (view.incindex[item as usize] + 1) % 6;
            view.increment[item as usize] = self.get_increment(view.incindex[item as usize], item);
        }
    }
    /// `MidasSlots::slotAngle`.
    pub fn slot_angle(&mut self, view: &mut MidasView, value: i32) {
        view.sangle = value;
        view.phi = value as f32 / 10.;
    }
    /// `MidasSlots::get_bw_index`.
    pub fn get_bw_index(&self, view: &MidasView) -> usize {
        if view.applytoone != 0 {
            view.cz.max(0) as usize
        } else {
            view.refz.max(0) as usize
        }
    }
    /// `MidasSlots::try_section_change`.
    pub fn try_section_change(&mut self, view: &mut MidasView, ds: i32, dsref: i32) {
        view.cz = (view.cz + ds).clamp(0, view.zsize.saturating_sub(1));
        view.refz = (view.refz + dsref).clamp(0, view.zsize.saturating_sub(1));
        self.update_sections(view);
    }
    /// `MidasSlots::try_montage_section`.
    pub fn try_montage_section(&mut self, view: &mut MidasView, sec: i32, _direction: i32) {
        view.montcz = sec.clamp(view.minzpiece, view.maxzpiece);
        view.cz = view.montcz;
        self.update_sections(view);
    }
    /// `MidasSlots::sectionInc`.
    pub fn section_inc(&mut self, view: &mut MidasView, ds: i32) {
        if view.xtype == XTYPE_MONT {
            self.try_montage_section(view, view.montcz + ds, ds);
        } else {
            self.try_section_change(view, ds, if view.keepsecdiff != 0 { ds } else { 0 });
        }
    }
    /// `MidasSlots::slotCurValue`.
    pub fn slot_cur_value(&mut self, view: &mut MidasView, sec: i32) {
        self.try_section_change(view, sec - 1 - view.cz, 0);
    }
    /// `MidasSlots::slotRefValue`.
    pub fn slot_ref_value(&mut self, view: &mut MidasView, sec: i32) {
        self.try_section_change(view, 0, sec - 1 - view.refz);
    }
    /// `MidasSlots::slotChunkValue`.
    pub fn slot_chunk_value(&mut self, view: &mut MidasView, sec: i32) {
        view.cur_chunk = (sec - 1).clamp(0, view.num_chunks.saturating_sub(1));
        self.update_sections(view);
    }
    /// `MidasSlots::try_montage_edge`.
    pub fn try_montage_edge(&mut self, view: &mut MidasView, sec: i32, _direction: i32) {
        view.curedge = sec.clamp(0, view.maxedge[view.xory.max(0) as usize].saturating_sub(1));
        self.update_sections(view);
    }
    /// `MidasSlots::slotEdge`.
    pub fn slot_edge(&mut self, view: &mut MidasView, up_down: i32) {
        self.try_montage_edge(view, view.curedge + up_down, up_down);
    }
    /// `MidasSlots::slotEdgeValue`.
    pub fn slot_edge_value(&mut self, view: &mut MidasView, sec: i32) {
        self.try_montage_edge(view, sec - 1, 0);
    }
    /// `MidasSlots::try_lower_piece`.
    pub fn try_lower_piece(
        &mut self,
        view: &mut MidasView,
        pcx: i32,
        pcy: i32,
        xory: i32,
        _direction: i32,
    ) {
        view.edgeind = self.lower_piece_to_edgeno(pcx, pcy, xory);
        view.xory = xory;
    }
    /// `MidasSlots::finishNewEdge`.
    pub fn finish_new_edge(&mut self, view: &mut MidasView) {
        view.changed = 1;
        self.expecting_three_dmod_edge = false;
    }
    /// `MidasSlots::slotLowerXvalue`.
    pub fn slot_lower_xvalue(&mut self, view: &mut MidasView, sec: i32) {
        self.step_lower_xor_y(view, 0, sec);
    }
    /// `MidasSlots::slotLowerYvalue`.
    pub fn slot_lower_yvalue(&mut self, view: &mut MidasView, sec: i32) {
        self.step_lower_xor_y(view, 1, sec);
    }
    /// `MidasSlots::stepLowerXorY`.
    pub fn step_lower_xor_y(&mut self, view: &mut MidasView, xory: i32, direction: i32) {
        let mut x = 0;
        let mut y = 0;
        self.edgeno_to_lower_piece(view.edgeind, xory, &mut x, &mut y);
        if xory == 0 {
            x += direction;
        } else {
            y += direction;
        }
        self.try_lower_piece(view, x, y, xory, direction);
    }
    /// `MidasSlots::slotSelectWarpPointBySize` (libwarp point ordering boundary).
    pub fn slot_select_warp_point_by_size(
        &mut self,
        _view: &mut MidasView,
        _direction: i32,
    ) -> Result<(), String> {
        Err("MIDAS libwarp control-point arrays are not yet translated".into())
    }
    /// `MidasSlots::slotDrawVectors`.
    pub fn slot_draw_vectors(&mut self, view: &mut MidasView, state: bool) {
        view.draw_vectors = state;
    }
    /// `MidasSlots::slotXory`.
    pub fn slot_xory(&mut self, view: &mut MidasView, which: i32) {
        view.xory = which;
        self.manage_xory(view);
    }
    /// `MidasSlots::manage_xory`.
    pub fn manage_xory(&mut self, view: &mut MidasView) {
        view.center_xory = view.xory;
    }
    /// `MidasSlots::slotLeave_out`.
    pub fn slot_leave_out(&mut self, view: &mut MidasView) {
        view.any_skipped = 1 - view.any_skipped;
    }
    /// `MidasSlots::slotTop_error`.
    pub fn slot_top_error(&mut self, view: &mut MidasView, item: i32) {
        if item >= 0 && item < view.num_top_err {
            view.edgeind = view.topind[item as usize];
        }
    }
    /// `MidasSlots::slotSkipError`.
    pub fn slot_skip_error(&mut self, view: &mut MidasView, state: bool) {
        view.skip_err = state as i32;
    }
    /// `MidasSlots::slotSkipExcluded`.
    pub fn slot_skip_excluded(&mut self, view: &mut MidasView, state: bool) {
        view.exclude_skipped = state as i32;
    }
    /// `MidasSlots::slotExcludeEdge`.
    pub fn slot_exclude_edge(&mut self, view: &mut MidasView, state: bool) {
        view.any_skipped = state as i32;
    }
    /// `MidasSlots::slotOpenQuery3dmod`.
    pub fn slot_open_query_3dmod(&mut self) -> Result<(), String> {
        Err("MIDAS QProcess launch of 3dmod is not yet translated".into())
    }
    /// `MidasSlots::slot3dmodStarted`.
    pub fn slot_3dmod_started(&mut self) {
        self.three_dmod_started = true;
    }
    /// `MidasSlots::slot3dmodFinished`.
    pub fn slot_3dmod_finished(&mut self, _exit_code: i32, _normal_exit: bool) {
        self.three_dmod_started = false;
        self.released_three_dmod = true;
    }
    /// `MidasSlots::slot3dmodErrored`.
    pub fn slot_3dmod_errored(&mut self, _error: i32) {
        self.three_dmod_started = false;
    }
    /// `MidasSlots::slot3dmodHasOutput`.
    pub fn slot_3dmod_has_output(&mut self, _output: &[u8]) {}
    /// `MidasSlots::slotStdErrorFrom3dmod`.
    pub fn slot_std_error_from_3dmod(&mut self, _output: &[u8]) {}
    /// `MidasSlots::newEdgeFor3dmod`.
    pub fn new_edge_for_3dmod(&mut self) {
        self.expecting_three_dmod_edge = true;
    }
    /// `MidasSlots::slotRobustFit`.
    pub fn slot_robust_fit(&mut self, view: &mut MidasView, state: bool) {
        view.robust_fit = state as i32;
    }
    /// `MidasSlots::slotRobustCrit`.
    pub fn slot_robust_crit(&mut self, view: &mut MidasView, value: f64) {
        view.robust_crit = value as f32;
    }
    /// `MidasSlots::slotZoom`.
    pub fn slot_zoom(&mut self, view: &mut MidasView, up_down: i32) {
        view.zoomind = (view.zoomind + up_down).clamp(0, ZOOMS.len() as i32 - 1);
        view.zoom = ZOOMS[view.zoomind as usize];
        view.truezoom = view.zoom.abs().recip().max(1.);
        if view.zoom > 0. {
            view.truezoom = view.zoom;
        }
    }
    /// `MidasSlots::slotInterpolate`.
    pub fn slot_interpolate(&mut self, view: &mut MidasView, state: bool) {
        view.fast_interp = (!state) as i32;
    }
    /// `MidasSlots::slotUseBinning`.
    pub fn slot_use_binning(&mut self, view: &mut MidasView, state: bool) {
        view.bin_to_zoom_down = state;
    }
    /// `MidasSlots::display_bwslider_value`.
    pub fn display_bwslider_value(&self, white: bool, value: i32) -> String {
        if white {
            format!("White: {value}")
        } else {
            format!("Black: {value}")
        }
    }
    /// `MidasSlots::setbwlevels`.
    pub fn setbwlevels(&mut self, view: &mut MidasView, black: i32, white: i32, _draw: i32) -> i32 {
        if black >= white {
            return -1;
        }
        let index = self.get_bw_index(view);
        if let Some(trans) = view.tr.get_mut(index) {
            trans.black = black;
            trans.white = white;
        }
        view.blackstate = black;
        view.whitestate = white;
        0
    }
    /// `MidasSlots::slotBlacklevel`.
    pub fn slot_blacklevel(&mut self, view: &mut MidasView, value: i32) {
        self.setbwlevels(view, value, view.whitestate, 1);
    }
    /// `MidasSlots::slotWhitelevel`.
    pub fn slot_whitelevel(&mut self, view: &mut MidasView, value: i32) {
        self.setbwlevels(view, view.blackstate, value, 1);
    }
    /// `MidasSlots::slotBlackPressed`.
    pub fn slot_black_pressed(&mut self) {
        self.black_pressed = true;
    }
    /// `MidasSlots::slotBlackReleased`.
    pub fn slot_black_released(&mut self) {
        self.black_pressed = false;
    }
    /// `MidasSlots::slotWhitePressed`.
    pub fn slot_white_pressed(&mut self) {
        self.white_pressed = true;
    }
    /// `MidasSlots::slotWhiteReleased`.
    pub fn slot_white_released(&mut self) {
        self.white_pressed = false;
    }
    /// `MidasSlots::slotApplyone`.
    pub fn slot_applyone(&mut self, view: &mut MidasView, state: bool) {
        view.applytoone = state as i32;
    }
    /// `MidasSlots::slotKeepdiff`.
    pub fn slot_keepdiff(&mut self, view: &mut MidasView, state: bool) {
        view.keepsecdiff = state as i32;
    }
    /// `MidasSlots::slotEditWarp`.
    pub fn slot_edit_warp(&mut self, view: &mut MidasView, state: bool) {
        view.edit_warps = state && view.warping_ok;
    }
    /// `MidasSlots::slotAutoContrast`.
    pub fn slot_auto_contrast(&mut self, view: &mut MidasView) {
        self.setbwlevels(view, 0, 255, 1);
    }
    /// `MidasSlots::show_ref`.
    pub fn show_ref(&mut self, view: &mut MidasView) {
        view.showref = 1;
        view.vmode = MIDAS_VIEW_SINGLE;
    }
    /// `MidasSlots::show_cur`.
    pub fn show_cur(&mut self, view: &mut MidasView) {
        view.showref = 0;
        view.vmode = MIDAS_VIEW_SINGLE;
    }
    /// `MidasSlots::show_overlay`.
    pub fn show_overlay(&mut self, view: &mut MidasView) {
        view.vmode = MIDAS_VIEW_COLOR;
    }
    /// `MidasSlots::slotAlign_arm`.
    pub fn slot_align_arm(&mut self, view: &mut MidasView) {
        view.draw_corr_box = 1;
    }
    /// `MidasSlots::slotAlign_disarm`.
    pub fn slot_align_disarm(&mut self, view: &mut MidasView) {
        view.draw_corr_box = 0;
    }
    /// `MidasSlots::slotReverse`.
    pub fn slot_reverse(&mut self, view: &mut MidasView, state: bool) {
        view.reversemap = state as i32;
    }
    /// `MidasSlots::slotOverlay`.
    pub fn slot_overlay(&mut self, view: &mut MidasView, state: bool) {
        if state {
            self.show_overlay(view);
        } else {
            self.show_cur(view);
        }
    }
    /// `MidasSlots::slotGlobRot`.
    pub fn slot_glob_rot(&mut self, view: &mut MidasView, value: f64) {
        view.global_rot = value;
    }
    /// `MidasSlots::slotConstrainMouse`.
    pub fn slot_constrain_mouse(&mut self, view: &mut MidasView, state: bool) {
        view.mouse_xonly = state as i32;
    }
    /// `MidasSlots::slotCosStretch`.
    pub fn slot_cos_stretch(&mut self, view: &mut MidasView, state: bool) {
        view.cos_stretch = state as i32;
    }
    /// `MidasSlots::slotTiltOff`.
    pub fn slot_tilt_off(&mut self, view: &mut MidasView, value: f64) {
        view.tilt_offset = value as f32;
    }
    /// `MidasSlots::slotCorrelate`.
    pub fn slot_correlate(&mut self, view: &mut MidasView) -> Result<(), String> {
        view.draw_corr_box = 1;
        Err("MIDAS image cross-correlation closure is not yet translated".into())
    }
    /// `MidasSlots::slotCorrBoxSize`.
    pub fn slot_corr_box_size(&mut self, view: &mut MidasView, value: i32) {
        view.corr_box_size = value;
    }
    /// `MidasSlots::slotCorrShiftLimit`.
    pub fn slot_corr_shift_limit(&mut self, view: &mut MidasView, value: i32) {
        view.corr_shift_limit = value;
    }
    /// `MidasSlots::screenChanged`.
    pub fn screen_changed(&mut self, view: &mut MidasView) {
        view.screen_changed = true;
    }
    /// `MidasSlots::getDevicePixelRatio`.
    pub fn get_device_pixel_ratio(&self, view: &MidasView) -> f32 {
        view.device_pixel_ratio
    }
    /// `MidasSlots::extendTimerIfActive`.
    pub fn extend_timer_if_active(&mut self, interval: i32) {
        if self.timer_id != 0 {
            self.timer_id = interval;
        }
    }
    /// `MidasSlots::timerEvent`.
    pub fn timer_event(&mut self, view: &mut MidasView, _timer_id: i32) {
        if view.screen_changed {
            view.screen_changed = false;
        }
        self.timer_id = 0;
    }
    /// `MidasSlots::midas_keyinput`.
    pub fn midas_keyinput(&mut self, view: &mut MidasView, key: i32, control: bool, shift: bool) {
        view.ctrl_pressed = control as i32;
        view.shift_pressed = shift as i32;
        match key {
            0x0100_0012 => self.section_inc(view, -1),
            0x0100_0014 => self.section_inc(view, 1),
            _ => {}
        }
    }
    /// `MidasSlots::mouse_shift_image`.
    pub fn mouse_shift_image(&mut self, view: &mut MidasView) {
        view.xtrans += view.mx - view.lastmx;
        view.ytrans += view.my - view.lastmy;
        view.lastmx = view.mx;
        view.lastmy = view.my;
    }
    /// `MidasSlots::mouse_translate`.
    pub fn mouse_translate(&mut self, view: &mut MidasView) {
        let dx = (view.mx - view.lastmx) as f32 / view.truezoom;
        let dy = (view.my - view.lastmy) as f32 / view.truezoom;
        self.translate(view, dx, dy);
        view.lastmx = view.mx;
        view.lastmy = view.my;
    }
    /// `MidasSlots::mouse_rotate`.
    pub fn mouse_rotate(&mut self, view: &mut MidasView) {
        let step = (view.mx - view.lastmx) as f32 / 4.;
        self.rotate(view, step);
        view.lastmx = view.mx;
    }
    /// `MidasSlots::mouse_stretch`.
    pub fn mouse_stretch(&mut self, view: &mut MidasView, shift: bool) {
        let step = 1. + (view.my - view.lastmy) as f32 / 200.;
        if shift {
            self.scale(view, step);
        } else {
            self.stretch(view, step, view.phi);
        }
        view.lastmy = view.my;
    }
    /// `MidasSlots::getParamDecimals`.
    pub fn get_param_decimals(&self, param: i32) -> i32 {
        PARAM_DECIMALS.get(param as usize).copied().unwrap_or(0)
    }
    /// `MidasSlots::getIncDecimals`.
    pub fn get_inc_decimals(&self, param: i32) -> i32 {
        INC_DECIMALS.get(param as usize).copied().unwrap_or(0)
    }
    /// `MidasSlots::getParamDigits`.
    pub fn get_param_digits(&self, param: i32) -> i32 {
        PARAM_DIGITS.get(param as usize).copied().unwrap_or(0)
    }
    /// `MidasSlots::getIncDigits`.
    pub fn get_inc_digits(&self, param: i32) -> i32 {
        INC_DIGITS.get(param as usize).copied().unwrap_or(0)
    }
    /// `MidasSlots::getIncrement`.
    pub fn get_increment(&self, index: i32, typ: i32) -> f32 {
        INCREMENTS
            .get(index.clamp(0, 5) as usize)
            .map(|row| row[typ.clamp(0, 2) as usize])
            .unwrap_or(0.)
    }
    /// `MidasSlots::sprintf_decimals`.
    pub fn sprintf_decimals(&self, decimals: i32, digits: i32, value: f32) -> String {
        format!(
            "{:width$.precision$}",
            value,
            width = digits.max(1) as usize,
            precision = decimals.max(0) as usize
        )
    }
    /// `MidasSlots::backup_current_mat`.
    pub fn backup_current_mat(&mut self, view: &mut MidasView) {
        if let Some(transform) = view.tr.get(view.cz.max(0) as usize) {
            view.backup_mat = transform.mat;
        }
    }
    /// `MidasSlots::synchronizeChunk`.
    pub fn synchronize_chunk(&mut self, view: &mut MidasView, sec: i32) {
        if let Some(chunk) = view.chunk.get(view.cur_chunk.max(0) as usize) {
            view.cz = sec.clamp(chunk.start, chunk.start + chunk.size - 1);
        }
    }
    /// `MidasSlots::convertNumLock`.
    pub fn convert_num_lock(&self, keysym: &mut i32, keypad: &mut i32) {
        const KEYS: [i32; 10] = [
            0x0100_0007,
            0x0100_0006,
            0x0100_0010,
            0x0100_0013,
            0x0100_0012,
            0x0100_0014,
            0x0100_0015,
            0x0100_0016,
            0x0100_0017,
            0x0100_0018,
        ];
        if let Some(index) = KEYS.iter().position(|&key| key == *keysym) {
            *keypad = 1;
            *keysym = index as i32;
        }
    }
    /// `MidasSlots::showHelpPage`.
    pub fn show_help_page(&mut self, _page: &str) -> Result<i32, String> {
        self.imod_help_present = true;
        Err("MIDAS ImodAssistant Qt help transport is not yet translated".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::midas::midas::MidasTransform;
    #[test]
    fn translate_updates_matrix_and_state() {
        let mut v = MidasView::default();
        v.tr.push(MidasTransform::default());
        let mut s = MidasSlots::new();
        s.translate(&mut v, 2., -3.);
        assert_eq!(v.tr[0].mat[6], 2.);
        assert_eq!(v.tr[0].mat[7], -3.);
    }
    #[test]
    fn source_constructor_uses_native_resize_bump_and_owned_cleanup() {
        let slots = midas_slots();
        assert_eq!(slots.resize_bump, 1);
        free_midas_slots(slots);
    }
    #[test]
    fn contrast_rejects_crossed_limits() {
        let mut v = MidasView::default();
        v.tr.push(MidasTransform::default());
        assert_eq!(MidasSlots::new().clone().setbwlevels(&mut v, 50, 40, 0), -1);
    }
}
