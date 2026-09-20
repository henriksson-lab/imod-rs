//! Translation of `IMOD/3dmod/info_cb.cpp` and `info_cb.h`.
//!
//! `info_cb.cpp` is the stateful half of the information window.  Its image
//! sampling, Qt widget and other-window calls are represented by the direct
//! [`InfoCbBoundary`] boundary; the contrast/range and floating state follow
//! the original source rather than being delegated to a replacement UI.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imodel::{Iindex, Imod, imod_set_index};
use crate::imod::three_dmod::imodview::{
    IMOD_DRAW_ALL, IMOD_DRAW_MOD, IMOD_DRAW_NOSYNC, IMOD_DRAW_XYZ, IMOD_MM_TOGGLE, IMOD_MMODEL,
    IMOD_MMOVIE, ImodView, ivw_bind_mouse,
};

/// `MeanSDData` in the source.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanSdData {
    pub mean: f32,
    pub sd: f32,
    pub ix_start: i32,
    pub iy_start: i32,
    pub nx_use: i32,
    pub ny_use: i32,
    pub cache_sum: i32,
}
/// `TimeRampData` in the source.
#[derive(Clone, Copy, Debug)]
pub struct TimeRampData {
    pub last_section: i32,
    pub ref_black: f32,
    pub ref_white: f32,
    pub black_in_range: i32,
    pub white_in_range: i32,
    pub range_low: i32,
    pub range_high: i32,
    pub subsets: i32,
    pub float_on: i32,
    pub reverse: i32,
    pub false_color: i32,
    pub cramp_ind: i32,
}
impl Default for TimeRampData {
    fn default() -> Self {
        Self {
            // `sTRampData` is initialized with an unvisited time marker; zero
            // is a valid section and would suppress the source global-float path.
            last_section: -1,
            ref_black: 0.,
            ref_white: 0.,
            black_in_range: 0,
            white_in_range: 0,
            range_low: 0,
            range_high: 0,
            subsets: 0,
            float_on: 0,
            reverse: 0,
            false_color: 0,
            cramp_ind: 0,
        }
    }
}
/// Source `Cramp` fields accessed by this unit.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cramp {
    pub blacklevel: i32,
    pub whitelevel: i32,
    pub reverse: i32,
    pub falsecolor: i32,
}

/// Lock state snapshot of one Zap, Slicer, or XYZ window consumed by
/// `makeListOfTimesToFloat`.  The rewritten GUI host supplies these in its
/// dialog-manager order; `is_top_window` corresponds to `objList.at(0)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimeLockedImageWindow {
    pub section_locked: bool,
    pub time_lock: i32,
    pub is_top_window: bool,
}

/// The static variables of `info_cb.cpp`, held by the owning viewer instead of C globals.
#[derive(Clone, Debug)]
pub struct InfoCbState {
    pub forbid_level: i32,
    pub ref_black: f32,
    pub ref_white: f32,
    pub last_section: i32,
    pub last_time: i32,
    pub last_ramp_time: i32,
    pub sec_data: Vec<MeanSdData>,
    pub table_size: i32,
    pub tdim: i32,
    pub single_cleared: i32,
    pub save_next_clear: i32,
    pub cleared_section: i32,
    pub cleared_time: i32,
    pub cleared_mean: f32,
    pub cleared_sd: f32,
    pub cleared_black: f32,
    pub cleared_white: f32,
    pub t_ramp_data: Vec<TimeRampData>,
    pub ctrl_pressed: i32,
    pub imod_obj_cnum: i32,
    pub float_on: i32,
    pub doing_float: i32,
    pub float_subsets: i32,
    pub last_subsets: i32,
    pub last_reverse: i32,
    pub float_matt: f32,
    pub t_ramps_on: i32,
    pub dump_cache: i32,
    pub start_dump: i32,
    pub update_info_only: i32,
}

impl InfoCbState {
    /// `imodInfoSubset`.
    pub fn imod_info_subset(&mut self, value: i32) {
        self.float_subsets = value;
    }
    /// `imodInfoTRamps`.
    pub fn imod_info_t_ramps(&mut self, value: i32) {
        self.t_ramps_on = value;
    }
    /// `imodInfoCtrlPress`.
    pub fn imod_info_ctrl_press(&mut self, pressed: i32) {
        self.ctrl_pressed = pressed;
    }
    /// `imod_info_setocp`.
    pub fn imod_info_setocp(&mut self, model: &Imod, b: &mut dyn InfoCbBoundary) {
        let oi = model.cindex.object;
        let obj = model.obj.get(oi.max(0) as usize);
        let cont = obj.and_then(|o| o.cont.get(model.cindex.contour.max(0) as usize));
        let max = [
            model.obj.len() as i32,
            obj.map_or(-1, |o| o.cont.len() as i32),
            cont.map_or(-1, |c| c.pts.len() as i32),
        ];
        let val = [
            if obj.is_some() { oi + 1 } else { 0 },
            if cont.is_some() {
                model.cindex.contour + 1
            } else {
                -1
            },
            if cont.map_or(false, |c| !c.pts.is_empty()) {
                model.cindex.point + 1
            } else {
                if cont.is_some() { 0 } else { -1 }
            },
        ];
        b.update_ocp(val, max);
        if self.imod_obj_cnum != oi && oi >= 0 {
            self.imod_obj_cnum = oi;
            imod_info_setobjcolor(model, b)
        }
        if self.update_info_only == 0 {
            b.object_changed();
        }
    }
    /// `imodInfoUpdateOnly`.
    pub fn imod_info_update_only(&mut self, value: i32) {
        self.update_info_only = value;
    }
    /// `imod_info_setbw`.
    pub fn imod_info_setbw(
        &mut self,
        vi: &mut ImodView,
        black: i32,
        white: i32,
        b: &mut dyn InfoCbBoundary,
    ) {
        let changed = vi.black != black || vi.white != white;
        vi.black = black;
        vi.white = white;
        if vi.ushort_store != 0 {
            if black < vi.range_low {
                vi.range_low = black
            }
            if white > vi.range_high {
                vi.range_high = white
            }
            vi.black_in_range = info_level_to_slider(vi, black);
            vi.white_in_range = info_level_to_slider(vi, white);
            b.set_lh_sliders(vi.range_low, vi.range_high)
        }
        b.set_bw_sliders(vi.black_in_range, vi.white_in_range);
        if self.doing_float == 0 {
            self.ref_black = black as f32;
            self.ref_white = white as f32
        }
        if changed {
            b.draw(if false { IMOD_DRAW_ALL } else { IMOD_DRAW_MOD });
        }
    }
    /// `getSampleLimits`.
    pub fn get_sample_limits(&self, vi: &ImodView) -> (i32, i32, i32, i32, f32) {
        let ix = (self.float_matt * vi.xsize as f32) as i32;
        let iy = (self.float_matt * vi.ysize as f32) as i32;
        let nx = vi.xsize - 2 * ix;
        let ny = vi.ysize - 2 * iy;
        let sample = (10000. / (nx.max(1) * ny.max(1)) as f32).min(1.);
        (ix, iy, nx, ny, sample)
    }
    /// `setTimeRampDataFromGlobal`.
    pub fn set_time_ramp_data_from_global(
        &self,
        vi: &ImodView,
        cramp: &Cramp,
        trd: &mut TimeRampData,
    ) {
        if vi.ushort_store != 0 {
            trd.range_low = vi.range_low;
            trd.range_high = vi.range_high
        }
        trd.last_section = self.last_section;
        trd.subsets = self.float_subsets;
        trd.float_on = self.float_on;
        trd.reverse = cramp.reverse;
        trd.false_color = cramp.falsecolor;
    }
    /// `imodInfoSaveNextClear`.
    pub fn imod_info_save_next_clear(&mut self) {
        self.save_next_clear = 1;
    }
    /// `imod_info_float_clear`.
    pub fn imod_info_float_clear(&mut self, section: i32, time: i32) {
        self.single_cleared = 0;
        if section < 0 && time < 0 {
            self.sec_data.clear();
            self.table_size = 0;
            self.last_section = -1;
            return;
        }
        if section < 0 {
            for i in 0..-section {
                if let Some(d) = self.sec_data.get_mut((i * self.tdim + time) as usize) {
                    d.mean = -1.;
                    d.sd = -1.
                }
            }
            return;
        }
        let index = section * self.tdim + time;
        if let Some(d) = self.sec_data.get_mut(index as usize) {
            if self.save_next_clear != 0 {
                self.cleared_section = section;
                self.cleared_time = time;
                self.cleared_mean = d.mean;
                self.cleared_sd = d.sd;
                self.save_next_clear = 0
            }
            d.mean = -1.;
            d.sd = -1.;
            self.single_cleared = i32::from(self.cleared_sd >= 0.)
        }
    }
    /// `imodInfoTimeRampIndex`.
    pub fn imod_info_time_ramp_index(
        &mut self,
        vi: &ImodView,
        time: i32,
        allow_no_last_sec: bool,
    ) -> i32 {
        if self.t_ramps_on == 0 || time < 1 || time == vi.cur_time {
            return -1;
        }
        if time as usize >= self.t_ramp_data.len() {
            return -1;
        }
        let ramp_count = self
            .t_ramp_data
            .iter()
            .filter(|data| data.cramp_ind >= 0)
            .count() as i32;
        let trd = &mut self.t_ramp_data[time as usize];
        if trd.last_section < 0 && !allow_no_last_sec {
            return -1;
        }
        if trd.cramp_ind < 0 {
            trd.cramp_ind = ramp_count
        }
        trd.cramp_ind
    }
    /// `imodInfoSyncToTimeRamp`.
    pub fn imod_info_sync_to_time_ramp(&mut self, vi: &ImodView, cramp: &Cramp, time: i32) {
        if self.t_ramps_on == 0 || time < 1 {
            return;
        }
        if let Some(trd) = self.t_ramp_data.get_mut(time as usize) {
            if trd.cramp_ind < 0 || trd.last_section < 0 {
                return;
            }
            if time == vi.cur_time {
                trd.false_color = cramp.falsecolor;
                trd.reverse = cramp.reverse;
                trd.black_in_range = vi.black_in_range;
                trd.white_in_range = vi.white_in_range;
            }
        }
    }
    /// `imodInfoSyncRampToTimeData`.
    pub fn imod_info_sync_ramp_to_time_data(&mut self, ramp: &Cramp, time: i32) {
        if self.t_ramps_on == 0 || time < 1 {
            return;
        }
        if let Some(trd) = self.t_ramp_data.get_mut(time as usize) {
            if trd.cramp_ind >= 0 && trd.last_section >= 0 {
                trd.false_color = ramp.falsecolor;
                trd.reverse = ramp.reverse;
                trd.ref_black = ramp.blacklevel as f32;
                trd.ref_white = ramp.whitelevel as f32;
                trd.black_in_range = ramp.blacklevel;
                trd.white_in_range = ramp.whitelevel;
            }
        }
    }
    /// `imodInfoGetLowHighRange`.
    pub fn imod_info_get_low_high_range(&self, vi: &ImodView, time: i32) -> (i32, i32) {
        if self.t_ramps_on == 0 || time < 1 || time == vi.cur_time {
            (vi.range_low, vi.range_high)
        } else {
            self.t_ramp_data
                .get(time as usize)
                .map(|d| (d.range_low, d.range_high))
                .unwrap_or((vi.range_low, vi.range_high))
        }
    }
    /// `makeListOfTimesToFloat`; window enumeration remains at the Zap/Slicer/Xyz boundary.
    pub fn make_list_of_times_to_float(
        &self,
        vi: &ImodView,
        section: i32,
        windows: &[TimeLockedImageWindow],
    ) -> Vec<i32> {
        let mut times = Vec::new();
        for window in windows {
            if window.section_locked || window.time_lock == 0 || window.time_lock == vi.cur_time {
                continue;
            }
            let Some(time_data) = self.t_ramp_data.get(window.time_lock as usize) else {
                continue;
            };
            // `info_cb.cpp:1244-1247`: a saved floating time is reconsidered on
            // a new section; otherwise global float initializes only the current
            // top image window for an unvisited time.
            let should_float = (time_data.float_on != 0
                && time_data.last_section >= 0
                && section != self.last_section)
                || (self.float_on != 0 && time_data.last_section < 0 && window.is_top_window);
            if should_float && !times.contains(&window.time_lock) {
                times.push(window.time_lock);
            }
        }
        times
    }
    /// `imodInfoSetFloatFlags`.
    pub fn imod_info_set_float_flags(&mut self, float_on: i32, subset: i32, ramps: i32) {
        self.float_on = float_on;
        self.float_subsets = subset;
        self.t_ramps_on = ramps;
    }
    /// `imodInfoGetFloatFlags`.
    pub fn imod_info_get_float_flags(&self) -> (i32, i32, i32) {
        (self.float_on, self.float_subsets, self.t_ramps_on)
    }
    /// `imod_info_forbid`.
    pub fn imod_info_forbid(&mut self) {
        self.forbid_level += 1;
    }
    /// `imod_info_enable`.
    pub fn imod_info_enable(&mut self) {
        self.forbid_level = (self.forbid_level - 1).max(0);
    }
    /// `imodStartAutoDumpCache`.
    pub fn imod_start_auto_dump_cache(&mut self) {
        self.dump_cache = 1;
        self.start_dump = 2;
    }

    /// `imodInfoNewOCP`.  The C source obtains `Imod` from `App->cvi`; Rust
    /// receives the owner explicitly so object/contour/point selection is real
    /// state mutation instead of an opaque UI notification.
    pub fn imod_info_new_ocp(
        &mut self,
        model: &mut Imod,
        which: i32,
        value: i32,
        no_show: i32,
        b: &mut dyn InfoCbBoundary,
    ) {
        b.set_focus();
        let value = value - 1;
        if value < 0 {
            self.imod_info_setocp(model, b);
            return;
        }
        let old = model.cindex;
        match which {
            // Object: with Show disabled detach contour and point; otherwise the
            // shared input controller preserves same-time contour attachment.
            0 if no_show != 0 => {
                model.cindex.object = value;
                model.cindex.contour = -1;
                model.cindex.point = -1;
            }
            0 => {
                imod_set_index(model, value, old.contour, old.point);
                b.keep_contour_at_same_time();
            }
            1 => {
                imod_set_index(model, old.object, value, old.point);
                b.restore_point_index(old);
            }
            2 => imod_set_index(model, old.object, old.contour, value),
            _ => return,
        }
        if no_show != 0 && model.cindex.point > 0 {
            // `IMOD_DRAW_NOSYNC` suppresses the controller sync that the normal
            // branch performs below; the boundary's draw flag is the direct host
            // equivalent for the rewritten GUI.
            b.draw(IMOD_DRAW_ALL | IMOD_DRAW_NOSYNC);
            return;
        }
        b.sync_model_index_to_mouse();
    }
    /// `imodInfoNewBW`.
    pub fn imod_info_new_bw(
        &mut self,
        vi: &mut ImodView,
        cramp: &mut Cramp,
        which: i32,
        value: i32,
        dragging: i32,
        b: &mut dyn InfoCbBoundary,
    ) {
        if self.forbid_level != 0 {
            b.set_bw_sliders(vi.black_in_range, vi.white_in_range);
            return;
        }
        let mut black = vi.black;
        let mut white = vi.white;
        if which != 0 {
            vi.white_in_range = value;
            white = imod_info_slider_to_level(vi, value);
            if black > white || vi.black_in_range > vi.white_in_range {
                black = white;
                vi.black_in_range = vi.white_in_range;
                b.set_bw_sliders(vi.black_in_range, vi.white_in_range);
            }
        } else {
            vi.black_in_range = value;
            black = imod_info_slider_to_level(vi, value);
            if black > white || vi.black_in_range > vi.white_in_range {
                white = black;
                vi.white_in_range = vi.black_in_range;
                b.set_bw_sliders(vi.black_in_range, vi.white_in_range);
            }
        }
        cramp.blacklevel = black;
        cramp.whitelevel = white;
        vi.black = black;
        vi.white = white;
        let save = self.float_on;
        self.float_on = 0;
        self.imod_info_setbw(vi, black, white, b);
        self.float_on = save;
    }
    /// `imodInfoNewLH`.
    pub fn imod_info_new_lh(
        &mut self,
        vi: &mut ImodView,
        cramp: &mut Cramp,
        which: i32,
        value: i32,
        dragging: i32,
        b: &mut dyn InfoCbBoundary,
    ) {
        let (mut low, mut high) = (vi.range_low, vi.range_high);
        if which != 0 {
            high = value;
            if high <= low {
                low = high - 1;
                b.set_lh_sliders(low, high)
            }
        } else {
            low = value;
            if high <= low {
                high = low + 1;
                b.set_lh_sliders(low, high)
            }
        }
        vi.range_low = low;
        vi.range_high = high;
        vi.black = imod_info_slider_to_level(vi, vi.black_in_range);
        vi.white = imod_info_slider_to_level(vi, vi.white_in_range);
        cramp.blacklevel = vi.black;
        cramp.whitelevel = vi.white;
        let save = self.float_on;
        self.float_on = 0;
        self.imod_info_setbw(vi, vi.black, vi.white, b);
        self.float_on = save;
    }
    /// `imod_info_bwfloat`; image/pyramid sampling is supplied by the source boundary.
    pub fn imod_info_bwfloat(
        &mut self,
        vi: &mut ImodView,
        section: i32,
        time: i32,
        b: &mut dyn InfoCbBoundary,
    ) -> i32 {
        let limits = self.get_sample_limits(vi);
        let Some((mean, sd)) =
            b.mean_sd(vi, section, time, (limits.0, limits.1, limits.2, limits.3))
        else {
            self.last_section = section;
            self.last_time = time;
            return 0;
        };
        let sd = sd.max(0.1);
        let (ref_mean, ref_sd) = if self.last_section >= 0 {
            b.mean_sd(
                vi,
                self.last_section,
                self.last_time,
                (limits.0, limits.1, limits.2, limits.3),
            )
            .unwrap_or((mean, sd))
        } else {
            (mean, sd)
        };
        let slope = sd / ref_sd.max(0.1);
        let mut black = (mean - (ref_mean - self.ref_black) * slope).round() as i32;
        let mut white = (black as f32 + slope * (self.ref_white - self.ref_black)).round() as i32;
        let max = if vi.ushort_store != 0 { 65535 } else { 255 };
        black = black.clamp(0, max);
        white = white.clamp(black, max);
        if white < black + 2 {
            black = (black - 1).max(0);
            white = (white + 1).min(max)
        }
        let changed = black != vi.black || white != vi.white;
        vi.black = black;
        vi.white = white;
        self.ref_black = black as f32;
        self.ref_white = white as f32;
        self.last_section = section;
        self.last_time = time;
        if changed {
            self.doing_float = 1;
            self.imod_info_setbw(vi, black, white, b);
            self.doing_float = 0;
            1
        } else {
            0
        }
    }
    /// `imodInfoCurrentMeanSD`.
    pub fn imod_info_current_mean_sd(
        &self,
        vi: &ImodView,
        b: &mut dyn InfoCbBoundary,
    ) -> Option<(f32, f32, f32, f32)> {
        let l = self.get_sample_limits(vi);
        let (mean, sd) = b.mean_sd(
            vi,
            (vi.zmouse + 0.5) as i32,
            vi.cur_time,
            (l.0, l.1, l.2, l.3),
        )?;
        let (mut lo, mut hi) = (mean - 3. * sd, mean + 3. * sd);
        if vi.ushort_store != 0 {
            lo = lo.clamp(0., 65535.);
            hi = hi.clamp(0., 65535.);
            if lo as i32 >= hi as i32 - 1 {
                lo = (0.5 * (lo + hi) - 1.).clamp(0., 65533.);
                hi = lo + 2.
            }
        }
        Some((mean, sd, lo, hi))
    }
    /// `imod_draw_window`.
    pub fn imod_draw_window(
        &mut self,
        model: Option<&Imod>,
        vi: Option<&mut ImodView>,
        b: &mut dyn InfoCbBoundary,
    ) {
        if let Some(m) = model {
            self.imod_info_setocp(m, b)
        }
        if let Some(v) = vi {
            imod_info_setxyz(v, b)
        }
    }

    /// `imodInfoFloat`.
    pub fn imod_info_float(
        &mut self,
        vi: &mut ImodView,
        b: &mut dyn InfoCbBoundary,
        state_in: i32,
    ) {
        self.float_on = state_in;
        if state_in != 0 {
            let section = (vi.zmouse + 0.5) as i32;
            let _ = self.imod_info_bwfloat(vi, section, vi.cur_time, b);
            self.imod_info_setbw(vi, vi.black, vi.white, b);
        }
    }
    /// `imodInfoAutoContrast`.
    pub fn imod_info_auto_contrast(
        &mut self,
        vi: &mut ImodView,
        cramp: &mut Cramp,
        target_mean: i32,
        target_sd: i32,
        b: &mut dyn InfoCbBoundary,
    ) {
        let Some((mean, sd, lo, hi)) = self.imod_info_current_mean_sd(vi, b) else {
            return;
        };
        let max = if vi.ushort_store != 0 { 65535 } else { 255 };
        let (mut black, mut white) = if cramp.reverse != 0 {
            (
                (mean - sd * (255 - target_mean) as f32 / target_sd as f32).round() as i32,
                (mean + sd * target_mean as f32 / target_sd as f32).round() as i32,
            )
        } else {
            (
                (mean - sd * target_mean as f32 / target_sd as f32).round() as i32,
                (mean + sd * (255 - target_mean) as f32 / target_sd as f32).round() as i32,
            )
        };
        black = black.clamp(0, max);
        white = white.clamp(0, max);
        if white - black < 4 {
            let mid = (white + black) / 2;
            black = (mid - 2).max(0);
            white = (mid + 2).min(max)
        }
        if vi.ushort_store != 0 {
            vi.range_low = lo as i32;
            vi.range_high = hi as i32
        }
        vi.black = black;
        vi.white = white;
        cramp.blacklevel = black;
        cramp.whitelevel = white;
        let save = self.float_on;
        self.float_on = 0;
        self.imod_info_setbw(vi, black, white, b);
        self.float_on = save;
        let _ = imod_info_input(b);
    }
}

impl Default for InfoCbState {
    fn default() -> Self {
        Self {
            forbid_level: 0,
            ref_black: 0.,
            ref_white: 255.,
            last_section: -1,
            last_time: 0,
            last_ramp_time: 0,
            sec_data: Vec::new(),
            table_size: 0,
            tdim: 0,
            single_cleared: 0,
            save_next_clear: 0,
            cleared_section: 0,
            cleared_time: 0,
            cleared_mean: 0.,
            cleared_sd: -1.,
            cleared_black: 0.,
            cleared_white: 0.,
            t_ramp_data: Vec::new(),
            ctrl_pressed: 0,
            imod_obj_cnum: -1,
            float_on: 0,
            doing_float: 0,
            float_subsets: 0,
            last_subsets: 0,
            last_reverse: -1,
            float_matt: 0.05,
            t_ramps_on: 0,
            dump_cache: 0,
            start_dump: 0,
            update_info_only: 0,
        }
    }
}

/// Native Qt/viewer calls made by `info_cb.cpp`.  The trait is intentionally a
/// direct boundary, so every translated source callback remains visible.
pub trait InfoCbBoundary {
    fn set_focus(&mut self) {}
    fn set_bw_sliders(&mut self, _: i32, _: i32) {}
    fn set_lh_sliders(&mut self, _: i32, _: i32) {}
    fn set_float(&mut self, _: i32) {}
    fn set_subarea(&mut self, _: i32) {}
    fn set_movie_model(&mut self, _: i32) {}
    fn update_ocp(&mut self, _: [i32; 3], _: [i32; 3]) {}
    fn update_xyz(&mut self, _: [i32; 3], _: [i32; 3]) {}
    fn set_object_color(&mut self, _: u32, _: u32) {}
    fn draw(&mut self, _: i32) {}
    fn process_events(&mut self) {}
    fn object_changed(&mut self) {}
    /// `inputKeepContourAtSameTime` after an object selection.
    fn keep_contour_at_same_time(&mut self) {}
    /// `inputRestorePointIndex` after a contour selection.
    fn restore_point_index(&mut self, _: Iindex) {}
    /// `ivwControlActive(vi, 0); imod_setxyzmouse()`.
    fn sync_model_index_to_mouse(&mut self) {}
    fn xyz_changed(&mut self) {}
    fn message(&mut self, _: &str) {}
    fn mean_sd(
        &mut self,
        _: &ImodView,
        _: i32,
        _: i32,
        _: (i32, i32, i32, i32),
    ) -> Option<(f32, f32)> {
        None
    }
    fn quit(&mut self) {}
    fn cache_dump(&mut self) {}
    fn cache_start_position(&mut self) {}
}

/// `infoSliderInRangeToLevel`.
pub fn info_slider_in_range_to_level(slider: i32, low: i32, high: i32) -> i32 {
    (slider as f64 * (high - low) as f64 / 255. + low as f64).round() as i32
}
/// `infoLevelToSliderInRange`.
pub fn info_level_to_slider_in_range(level: i32, low: i32, high: i32) -> i32 {
    (255. * (level - low) as f64 / (high - low).max(1) as f64).round() as i32
}
/// `imodInfoSliderToLevel`.
pub fn imod_info_slider_to_level(vi: &ImodView, slider: i32) -> i32 {
    if vi.ushort_store == 0 {
        slider
    } else {
        info_slider_in_range_to_level(slider, vi.range_low, vi.range_high)
    }
}
/// `infoLevelToSlider`.
pub fn info_level_to_slider(vi: &ImodView, level: i32) -> i32 {
    if vi.ushort_store == 0 {
        level
    } else {
        info_level_to_slider_in_range(level, vi.range_low, vi.range_high).clamp(0, 255)
    }
}

/// `imodInfoNewXYZ`.
pub fn imod_info_new_xyz(vi: &mut ImodView, values: [i32; 3], b: &mut dyn InfoCbBoundary) {
    b.set_focus();
    vi.xmouse = (values[0] - 1) as f32;
    vi.ymouse = (values[1] - 1) as f32;
    vi.zmouse = (values[2] - 1) as f32;
    b.draw(IMOD_DRAW_XYZ);
}

/// `imodInfoMMSelected`.
pub fn imod_info_mm_selected(model: &mut Imod, mode: i32, b: &mut dyn InfoCbBoundary) {
    imod_set_mmode(model, mode, b);
    b.draw(IMOD_DRAW_MOD);
}

/// `imodInfoQuit`.
pub fn imod_info_quit(b: &mut dyn InfoCbBoundary) {
    b.quit();
}

/// `imod_info_setobjcolor`.
pub fn imod_info_setobjcolor(model: &Imod, b: &mut dyn InfoCbBoundary) {
    let (obj, red, green, blue) = match model.obj.get(model.cindex.object.max(0) as usize) {
        Some(o) => (
            true,
            (255. * o.red) as i32,
            (255. * o.green) as i32,
            (255. * o.blue) as i32,
        ),
        None => (false, 128, 128, 128),
    };
    let th = 32.;
    let clev = (if red as f64 > th {
        (red as f64 - th) / (240. - th)
    } else {
        0.
    }) + (if green as f64 > th {
        (green as f64 - th) / (175. - th)
    } else {
        0.
    }) + (if blue as f64 > th {
        (blue as f64 - th) / (495. - th)
    } else {
        0.
    });
    let fore = if clev > 1. { 0 } else { 255 };
    b.set_object_color(
        (fore as u32) << 16 | (fore as u32) << 8 | fore as u32,
        (red as u32) << 16 | (green as u32) << 8 | blue as u32,
    );
}

/// `imod_info_setxyz`.
pub fn imod_info_setxyz(vi: &mut ImodView, b: &mut dyn InfoCbBoundary) {
    ivw_bind_mouse(vi);
    b.update_xyz(
        [
            (vi.xmouse + 1.) as i32,
            (vi.ymouse + 1.) as i32,
            (vi.zmouse + 1.5) as i32,
        ],
        [vi.xsize, vi.ysize, vi.zsize],
    );
    b.xyz_changed();
}

/// `imodInfoLimitSubarea`.
pub fn imod_info_limit_subarea(
    left_xpad: i32,
    right_x: i32,
    left_ypad: i32,
    right_y: i32,
    ix_start: &mut i32,
    iy_start: &mut i32,
    nx_use: &mut i32,
    ny_use: &mut i32,
) {
    let end = (*nx_use + *ix_start).min(right_x);
    if (*ix_start).max(left_xpad) < end - 4 {
        *ix_start = (*ix_start).max(left_xpad);
        *nx_use = end - *ix_start
    }
    let end = (*ny_use + *iy_start).min(right_y);
    if (*iy_start).max(left_ypad) < end - 4 {
        *iy_start = (*iy_start).max(left_ypad);
        *ny_use = end - *iy_start
    }
}

/// `setRampFromTimeData`.
pub fn set_ramp_from_time_data(vi: &ImodView, trd: &TimeRampData, ramp: &mut Cramp) {
    ramp.falsecolor = trd.false_color;
    ramp.reverse = trd.reverse;
    ramp.blacklevel = if vi.ushort_store != 0 {
        info_slider_in_range_to_level(trd.black_in_range, trd.range_low, trd.range_high)
    } else {
        trd.black_in_range
    };
    ramp.whitelevel = if vi.ushort_store != 0 {
        info_slider_in_range_to_level(trd.white_in_range, trd.range_low, trd.range_high)
    } else {
        trd.white_in_range
    };
}

/// `imod_info_input`.
pub fn imod_info_input(b: &mut dyn InfoCbBoundary) -> i32 {
    b.process_events();
    0
}
/// `show_status`.
pub fn show_status(info: Option<&str>, b: &mut dyn InfoCbBoundary) {
    if let Some(info) = info {
        imod_info_msg(Some(info), Some(" "), b)
    }
}
/// `imod_show_info`.
pub fn imod_show_info(info: Option<&str>, line: i32, b: &mut dyn InfoCbBoundary) {
    if let Some(info) = info {
        if line == 1 {
            imod_info_msg(Some(info), None, b)
        }
        if line == 2 {
            imod_info_msg(None, Some(info), b)
        }
    }
}
/// `imod_info_msg`.
pub fn imod_info_msg(top: Option<&str>, bot: Option<&str>, b: &mut dyn InfoCbBoundary) {
    if let Some(s) = top {
        b.message(s)
    }
    if let Some(s) = bot {
        b.message(s)
    }
}

/// `imod_set_mmode`.
pub fn imod_set_mmode(model: &mut Imod, mut mode: i32, b: &mut dyn InfoCbBoundary) {
    if mode == IMOD_MM_TOGGLE {
        mode = if model.mousemode == IMOD_MMOVIE {
            IMOD_MMODEL
        } else {
            IMOD_MMOVIE
        }
    }
    model.mousemode = mode;
    b.set_movie_model(if mode == IMOD_MMOVIE { 0 } else { 1 });
    b.draw(IMOD_DRAW_MOD);
}

/// `imodQuitCheck`.
pub fn imod_quit_check(b: &mut dyn InfoCbBoundary) -> i32 {
    imod_imgcnt(None, b);
    0
}
/// `imod_imgcnt`; Qt timing/cache calls are explicit boundary calls.
pub fn imod_imgcnt(string: Option<&str>, b: &mut dyn InfoCbBoundary) {
    if let Some(s) = string {
        b.message(s)
    }
    b.process_events();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct B {
        draws: Vec<i32>,
        means: Option<(f32, f32)>,
        keep_same_time: usize,
        restored: Vec<Iindex>,
        synced: usize,
    }
    impl InfoCbBoundary for B {
        fn draw(&mut self, x: i32) {
            self.draws.push(x)
        }
        fn keep_contour_at_same_time(&mut self) {
            self.keep_same_time += 1;
        }
        fn restore_point_index(&mut self, old: Iindex) {
            self.restored.push(old);
        }
        fn sync_model_index_to_mouse(&mut self) {
            self.synced += 1;
        }
        fn mean_sd(
            &mut self,
            _: &ImodView,
            _: i32,
            _: i32,
            _: (i32, i32, i32, i32),
        ) -> Option<(f32, f32)> {
            self.means
        }
    }
    #[test]
    fn slider_and_subarea_match_source_rounding() {
        assert_eq!(info_slider_in_range_to_level(128, 100, 1000), 552);
        let (mut x, mut y, mut nx, mut ny) = (0, 0, 100, 100);
        imod_info_limit_subarea(10, 80, 20, 90, &mut x, &mut y, &mut nx, &mut ny);
        assert_eq!((x, y, nx, ny), (10, 20, 70, 70));
    }
    #[test]
    fn auto_contrast_and_float_update_levels() {
        let mut s = InfoCbState::default();
        let mut v = ImodView {
            xsize: 100,
            ysize: 100,
            zsize: 1,
            ..Default::default()
        };
        let mut r = Cramp::default();
        let mut b = B {
            means: Some((100., 10.)),
            ..Default::default()
        };
        s.imod_info_auto_contrast(&mut v, &mut r, 128, 32, &mut b);
        assert!(v.white > v.black);
        s.float_on = 1;
        assert_eq!(s.imod_info_bwfloat(&mut v, 0, 0, &mut b), 0);
    }
    #[test]
    fn ocp_selection_mutates_model_and_uses_native_sync_routes() {
        let contour = || crate::imod::libimod::imodel::Icont {
            pts: vec![Default::default(), Default::default()],
            ..Default::default()
        };
        let mut model = Imod::default();
        model.obj = vec![
            crate::imod::libimod::imodel::Iobj {
                cont: vec![contour(), contour()],
                ..Default::default()
            },
            crate::imod::libimod::imodel::Iobj {
                cont: vec![contour()],
                ..Default::default()
            },
        ];
        model.cindex = Iindex {
            object: 0,
            contour: 0,
            point: 0,
        };
        let mut state = InfoCbState::default();
        let mut boundary = B::default();

        state.imod_info_new_ocp(&mut model, 0, 2, 1, &mut boundary);
        assert_eq!(
            model.cindex,
            Iindex {
                object: 1,
                contour: -1,
                point: -1
            }
        );
        assert_eq!(boundary.synced, 1);

        model.cindex = Iindex {
            object: 0,
            contour: 0,
            point: 1,
        };
        state.imod_info_new_ocp(&mut model, 1, 2, 0, &mut boundary);
        assert_eq!(model.cindex.contour, 1);
        assert_eq!(
            boundary.restored,
            vec![Iindex {
                object: 0,
                contour: 0,
                point: 1
            }]
        );

        state.imod_info_new_ocp(&mut model, 2, 2, 1, &mut boundary);
        assert_eq!(model.cindex.point, 1);
        assert_eq!(boundary.draws, vec![IMOD_DRAW_ALL | IMOD_DRAW_NOSYNC]);
    }
    #[test]
    fn time_float_list_follows_window_locks_and_top_window_rule() {
        let mut state = InfoCbState {
            last_section: 2,
            float_on: 1,
            t_ramp_data: vec![TimeRampData::default(); 5],
            ..Default::default()
        };
        state.t_ramp_data[2] = TimeRampData {
            float_on: 1,
            last_section: 1,
            ..Default::default()
        };
        let vi = ImodView {
            cur_time: 1,
            ..Default::default()
        };
        let times = state.make_list_of_times_to_float(
            &vi,
            3,
            &[
                TimeLockedImageWindow {
                    time_lock: 2,
                    ..Default::default()
                },
                TimeLockedImageWindow {
                    time_lock: 3,
                    is_top_window: true,
                    ..Default::default()
                },
                // Duplicate time and a section-locked window do not add a
                // second entry.
                TimeLockedImageWindow {
                    time_lock: 2,
                    ..Default::default()
                },
                TimeLockedImageWindow {
                    time_lock: 4,
                    section_locked: true,
                    is_top_window: true,
                },
            ],
        );
        assert_eq!(times, vec![2, 3]);
    }
}
