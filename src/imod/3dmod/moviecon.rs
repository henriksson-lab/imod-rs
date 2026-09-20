//! Translation of `IMOD/3dmod/moviecon.cpp` and `moviecon.h`.
#![allow(dead_code)]
use crate::imod::three_dmod::imodview::ImodView;
use std::time::Instant;

/// `imcUpdateDialog`'s form operation at the native GUI boundary.
pub trait MovieConNativeBoundary {
    fn set_non_tif_label(&mut self);
}
pub const BASEINT: f32 = 50.;
pub const RATEFAC: f32 = 1.25992;
pub const MINRATE: f32 = 0.2;
pub const MAXRATE: f32 = 200.;
/// Source file statics gathered without creating a replacement movie engine.
#[derive(Clone, Debug)]
pub struct MovieConState {
    pub start: [i32; 5],
    pub endsec: [i32; 5],
    pub increment: [i32; 4],
    pub maxend: [i32; 4],
    pub firsttime: bool,
    pub axiscon: usize,
    pub dialog_open: bool,
    pub realrate: f32,
    pub realint: f32,
    pub looponeway: i32,
    pub start_here: i32,
    pub autosnap: i32,
    pub movie_count: i32,
    pub starter_ctrl_id: i32,
    pub special_axis: i32,
    pub movie_current: i32,
    pub movie_start: i32,
    pub movie_last: i32,
    pub montage_fac: i32,
    pub scale_sizes: bool,
    pub size_scaling: i32,
    pub whole_mont: i32,
    pub snap_montage: bool,
    pub slicer_mont_fac: i32,
    pub scale_thicks: bool,
    pub thick_scaling: i32,
    pub slicer_montage: bool,
    pub timer: Option<Instant>,
}

impl MovieConState {
    pub fn imc_set_snap_montage(&mut self, v: bool) {
        self.snap_montage = v
    }
    pub fn imc_set_snap_whole_mont(&mut self, v: i32) {
        self.whole_mont = v
    }
    pub fn imc_set_montage_factor(&mut self, v: i32) {
        self.montage_fac = v.max(2)
    }
    pub fn imc_set_scale_sizes(&mut self, v: bool) {
        self.scale_sizes = v
    }
    pub fn imc_set_size_scaling(&mut self, v: i32) {
        self.size_scaling = v
    }
    pub fn imc_set_slicer_montage(&mut self, v: bool) {
        self.slicer_montage = v
    }
    pub fn imc_set_slicer_mont_factor(&mut self, v: i32) {
        self.slicer_mont_fac = v.max(2)
    }
    pub fn imc_set_scale_thicks(&mut self, v: bool) {
        self.scale_thicks = v
    }
    pub fn imc_set_thick_scaling(&mut self, v: i32) {
        self.thick_scaling = v
    }
    pub fn imc_set_special_limits(&mut self, axis: i32, start: i32, end: i32) {
        self.special_axis = axis;
        self.start[4] = start;
        self.endsec[4] = end
    }
    pub fn imc_start_timer(&mut self) {
        self.movie_count = 0;
        self.timer = Some(Instant::now());
        self.movie_start = 0;
        self.movie_current = 0
    }

    pub fn set_sliders(&self) -> (i32, i32, i32, i32, i32, i32, i32) {
        let mut max = self.maxend[self.axiscon];
        let on = (max >= 2) as i32;
        if on == 0 {
            max = 2
        }
        let (mut min, mut end) = (2, self.maxend[self.axiscon] + 1);
        if on == 0 {
            min = end;
            end += 1
        }
        (
            self.start[self.axiscon] + 1,
            max,
            self.endsec[self.axiscon] + 1,
            min,
            end,
            self.increment[self.axiscon],
            on,
        )
    }
    pub fn imc_reset_all(&mut self, v: &ImodView) {
        self.start = [0; 5];
        self.increment = [1; 4];
        self.endsec = [
            (v.xsize - 1).max(0),
            (v.ysize - 1).max(0),
            (v.zsize - 1).max(0),
            (v.num_times - 1).max(0),
            0,
        ];
        self.maxend.copy_from_slice(&self.endsec[..4]);
        self.looponeway = 0;
        self.autosnap = 0;
        self.firsttime = false;
    }
    pub fn imc_update_dialog(&self, n: &mut dyn MovieConNativeBoundary) {
        if self.dialog_open {
            n.set_non_tif_label();
        }
    }
    pub fn imc_start_snap_here(&self, _: &ImodView) -> i32 {
        self.start_here
    }
    pub fn imc_get_starter_id(&self) -> i32 {
        self.starter_ctrl_id
    }
    pub fn imc_set_starter_id(&mut self, v: i32) {
        self.starter_ctrl_id = v;
        self.special_axis = -1
    }
    pub fn imc_get_interval(&self) -> f32 {
        self.realint
    }
    pub fn imc_get_snap_montage(&self, doing: bool) -> bool {
        self.snap_montage && (!doing || self.dialog_open)
    }
    pub fn imc_get_snap_whole_mont(&self) -> i32 {
        self.whole_mont
    }
    pub fn imc_get_montage_factor(&self) -> i32 {
        self.montage_fac
    }
    pub fn imc_get_scale_sizes(&self) -> bool {
        self.scale_sizes
    }
    pub fn imc_get_size_scaling(&self) -> i32 {
        self.size_scaling
    }
    pub fn imc_get_slicer_montage(&self, doing: bool) -> bool {
        self.slicer_montage && (!doing || self.dialog_open)
    }
    pub fn imc_get_slicer_mont_factor(&self) -> i32 {
        self.slicer_mont_fac
    }
    pub fn imc_get_scale_thicks(&self) -> bool {
        self.scale_thicks
    }
    pub fn imc_get_thick_scaling(&self) -> i32 {
        self.thick_scaling
    }
    pub fn imc_set_movierate(&mut self, v: &mut ImodView, mut rate: i32) {
        loop {
            v.movierate = rate;
            self.realint = BASEINT * RATEFAC.powi(rate);
            self.realrate = 1000. / self.realint;
            if self.realrate > MAXRATE {
                rate += 1
            } else if self.realrate < MINRATE {
                rate -= 1
            } else {
                break;
            }
        }
    }
    pub fn imc_slider_changed(&mut self, which: i32, value: i32) {
        let a = self.axiscon;
        match which {
            0 => {
                self.start[a] = value - 1;
                if self.start[a] >= self.endsec[a] && self.start[a] + 1 <= self.maxend[a] {
                    self.endsec[a] = self.start[a] + 1
                }
            }
            1 => {
                self.endsec[a] = value - 1;
                if self.endsec[a] <= self.start[a] {
                    self.start[a] = self.endsec[a] - 1
                }
            }
            2 => self.increment[a] = value,
            _ => {}
        }
    }
    pub fn imc_axis_selected(&mut self, v: i32) {
        self.axiscon = v.clamp(0, 3) as usize
    }
    pub fn imc_extent_selected(&mut self, v: i32) {
        self.looponeway = v
    }
    pub fn imc_snap_selected(&mut self, v: i32) {
        self.autosnap = v
    }
    pub fn imc_start_here_selected(&mut self, v: i32) {
        self.start_here = v
    }

    pub fn imc_get_increment(&mut self, v: &ImodView, axis: i32) -> i32 {
        if self.firsttime {
            self.imc_reset_all(v)
        }
        self.increment[axis as usize]
    }
    pub fn imc_get_loop_mode(&mut self, v: &ImodView) -> i32 {
        if self.firsttime {
            self.imc_reset_all(v)
        }
        self.looponeway
    }
    pub fn imc_get_snapshot(&mut self, v: &ImodView) -> i32 {
        if self.firsttime {
            self.imc_reset_all(v)
        }
        self.autosnap
    }
    pub fn imc_get_start_end(&mut self, v: &ImodView, axis: i32) -> (i32, i32) {
        if self.firsttime {
            self.imc_reset_all(v)
        }
        let i = axis as usize;
        if axis == self.special_axis {
            (
                self.start[i].max(self.start[4]),
                self.endsec[i].min(self.endsec[4]),
            )
        } else {
            (self.start[i], self.endsec[i])
        }
    }
    pub fn imod_movie_con_dialog(&mut self, v: &ImodView) {
        self.dialog_open = true;
        self.imc_reset_all(v);
        self.axiscon = 2
    }
    pub fn imc_closing(&mut self, v: &ImodView) {
        self.dialog_open = false;
        self.imc_reset_all(v)
    }
    pub fn imc_reset_pressed(&mut self, v: &ImodView) {
        self.imc_reset_all(v)
    }
    pub fn imc_rate_entered(&mut self, v: &mut ImodView, mut rate: f32) {
        if rate <= 0. {
            self.imc_set_movierate(v, 0);
            return;
        }
        rate = rate.clamp(MINRATE, MAXRATE);
        self.realrate = rate;
        self.realint = 1000. / rate;
        v.movierate = ((self.realint / BASEINT).ln() / RATEFAC.ln() + 0.5) as i32
    }
    pub fn imc_increment_rate(&mut self, v: &mut ImodView, dir: i32) {
        self.imc_set_movierate(v, v.movierate + dir)
    }

    pub fn imc_read_timer(&mut self) -> Option<String> {
        self.movie_count += 1;
        let elapsed = self.timer?.elapsed().as_secs_f32();
        if elapsed == 0. {
            return None;
        }
        Some(format!(
            "Actual FPS: {:5.2} (avg)",
            self.movie_count as f32 / elapsed
        ))
    }
}

impl Default for MovieConState {
    fn default() -> Self {
        Self {
            start: [0; 5],
            endsec: [0; 5],
            increment: [0; 4],
            maxend: [0; 4],
            firsttime: true,
            axiscon: 0,
            dialog_open: false,
            realrate: 0.,
            realint: 0.,
            looponeway: 0,
            start_here: 0,
            autosnap: 0,
            movie_count: 0,
            starter_ctrl_id: 0,
            special_axis: -1,
            movie_current: 0,
            movie_start: 0,
            movie_last: 0,
            montage_fac: 2,
            scale_sizes: false,
            size_scaling: 1,
            whole_mont: 0,
            snap_montage: false,
            slicer_mont_fac: 2,
            scale_thicks: false,
            thick_scaling: 1,
            slicer_montage: false,
            timer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native(usize);
    impl MovieConNativeBoundary for Native {
        fn set_non_tif_label(&mut self) {
            self.0 += 1;
        }
    }
    #[test]
    fn ranges_rates_and_special_limits_match_source() {
        let v = ImodView {
            xsize: 10,
            ysize: 20,
            zsize: 30,
            num_times: 2,
            ..Default::default()
        };
        let mut s = MovieConState::default();
        s.imc_reset_all(&v);
        assert_eq!(s.imc_get_start_end(&v, 2), (0, 29));
        s.imc_set_special_limits(2, 4, 8);
        assert_eq!(s.imc_get_start_end(&v, 2), (4, 8));
        let mut v = v;
        s.imc_set_movierate(&mut v, 0);
        assert!((MINRATE..=MAXRATE).contains(&s.realrate));
    }
    #[test]
    fn open_controller_refreshes_the_source_non_tif_label() {
        let mut n = Native::default();
        (&MovieConState::default()).imc_update_dialog(&mut n);
        (&MovieConState {
            dialog_open: true,
            ..Default::default()
        })
            .imc_update_dialog(&mut n);
        assert_eq!(n.0, 1);
    }
}
