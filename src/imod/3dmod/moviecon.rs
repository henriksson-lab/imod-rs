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
pub fn set_sliders(state: &MovieConState) -> (i32, i32, i32, i32, i32, i32, i32) {
    let mut max = state.maxend[state.axiscon];
    let on = (max >= 2) as i32;
    if on == 0 {
        max = 2
    }
    let (mut min, mut end) = (2, state.maxend[state.axiscon] + 1);
    if on == 0 {
        min = end;
        end += 1
    }
    (
        state.start[state.axiscon] + 1,
        max,
        state.endsec[state.axiscon] + 1,
        min,
        end,
        state.increment[state.axiscon],
        on,
    )
}
pub fn imc_reset_all(s: &mut MovieConState, v: &ImodView) {
    s.start = [0; 5];
    s.increment = [1; 4];
    s.endsec = [
        (v.xsize - 1).max(0),
        (v.ysize - 1).max(0),
        (v.zsize - 1).max(0),
        (v.num_times - 1).max(0),
        0,
    ];
    s.maxend.copy_from_slice(&s.endsec[..4]);
    s.looponeway = 0;
    s.autosnap = 0;
    s.firsttime = false;
}
pub fn imc_update_dialog(s: &MovieConState, n: &mut dyn MovieConNativeBoundary) {
    if s.dialog_open {
        n.set_non_tif_label();
    }
}
pub fn imc_get_increment(s: &mut MovieConState, v: &ImodView, axis: i32) -> i32 {
    if s.firsttime {
        imc_reset_all(s, v)
    }
    s.increment[axis as usize]
}
pub fn imc_get_loop_mode(s: &mut MovieConState, v: &ImodView) -> i32 {
    if s.firsttime {
        imc_reset_all(s, v)
    }
    s.looponeway
}
pub fn imc_start_snap_here(s: &MovieConState, _: &ImodView) -> i32 {
    s.start_here
}
pub fn imc_get_snapshot(s: &mut MovieConState, v: &ImodView) -> i32 {
    if s.firsttime {
        imc_reset_all(s, v)
    }
    s.autosnap
}
pub fn imc_get_starter_id(s: &MovieConState) -> i32 {
    s.starter_ctrl_id
}
pub fn imc_set_starter_id(s: &mut MovieConState, v: i32) {
    s.starter_ctrl_id = v;
    s.special_axis = -1
}
pub fn imc_get_start_end(s: &mut MovieConState, v: &ImodView, axis: i32) -> (i32, i32) {
    if s.firsttime {
        imc_reset_all(s, v)
    }
    let i = axis as usize;
    if axis == s.special_axis {
        (s.start[i].max(s.start[4]), s.endsec[i].min(s.endsec[4]))
    } else {
        (s.start[i], s.endsec[i])
    }
}
pub fn imc_get_interval(s: &MovieConState) -> f32 {
    s.realint
}
pub fn imc_get_snap_montage(s: &MovieConState, doing: bool) -> bool {
    s.snap_montage && (!doing || s.dialog_open)
}
pub fn imc_set_snap_montage(s: &mut MovieConState, v: bool) {
    s.snap_montage = v
}
pub fn imc_get_snap_whole_mont(s: &MovieConState) -> i32 {
    s.whole_mont
}
pub fn imc_set_snap_whole_mont(s: &mut MovieConState, v: i32) {
    s.whole_mont = v
}
pub fn imc_get_montage_factor(s: &MovieConState) -> i32 {
    s.montage_fac
}
pub fn imc_set_montage_factor(s: &mut MovieConState, v: i32) {
    s.montage_fac = v.max(2)
}
pub fn imc_get_scale_sizes(s: &MovieConState) -> bool {
    s.scale_sizes
}
pub fn imc_set_scale_sizes(s: &mut MovieConState, v: bool) {
    s.scale_sizes = v
}
pub fn imc_get_size_scaling(s: &MovieConState) -> i32 {
    s.size_scaling
}
pub fn imc_set_size_scaling(s: &mut MovieConState, v: i32) {
    s.size_scaling = v
}
pub fn imc_get_slicer_montage(s: &MovieConState, doing: bool) -> bool {
    s.slicer_montage && (!doing || s.dialog_open)
}
pub fn imc_set_slicer_montage(s: &mut MovieConState, v: bool) {
    s.slicer_montage = v
}
pub fn imc_get_slicer_mont_factor(s: &MovieConState) -> i32 {
    s.slicer_mont_fac
}
pub fn imc_set_slicer_mont_factor(s: &mut MovieConState, v: i32) {
    s.slicer_mont_fac = v.max(2)
}
pub fn imc_get_scale_thicks(s: &MovieConState) -> bool {
    s.scale_thicks
}
pub fn imc_set_scale_thicks(s: &mut MovieConState, v: bool) {
    s.scale_thicks = v
}
pub fn imc_get_thick_scaling(s: &MovieConState) -> i32 {
    s.thick_scaling
}
pub fn imc_set_thick_scaling(s: &mut MovieConState, v: i32) {
    s.thick_scaling = v
}
pub fn imc_set_special_limits(s: &mut MovieConState, axis: i32, start: i32, end: i32) {
    s.special_axis = axis;
    s.start[4] = start;
    s.endsec[4] = end
}
pub fn imc_set_movierate(s: &mut MovieConState, v: &mut ImodView, mut rate: i32) {
    loop {
        v.movierate = rate;
        s.realint = BASEINT * RATEFAC.powi(rate);
        s.realrate = 1000. / s.realint;
        if s.realrate > MAXRATE {
            rate += 1
        } else if s.realrate < MINRATE {
            rate -= 1
        } else {
            break;
        }
    }
}
pub fn imc_start_timer(s: &mut MovieConState) {
    s.movie_count = 0;
    s.timer = Some(Instant::now());
    s.movie_start = 0;
    s.movie_current = 0
}
pub fn imc_read_timer(s: &mut MovieConState) -> Option<String> {
    s.movie_count += 1;
    let elapsed = s.timer?.elapsed().as_secs_f32();
    if elapsed == 0. {
        return None;
    }
    Some(format!(
        "Actual FPS: {:5.2} (avg)",
        s.movie_count as f32 / elapsed
    ))
}
pub fn imod_movie_con_dialog(s: &mut MovieConState, v: &ImodView) {
    s.dialog_open = true;
    imc_reset_all(s, v);
    s.axiscon = 2
}
pub fn imc_closing(s: &mut MovieConState, v: &ImodView) {
    s.dialog_open = false;
    imc_reset_all(s, v)
}
pub fn imc_reset_pressed(s: &mut MovieConState, v: &ImodView) {
    imc_reset_all(s, v)
}
pub fn imc_slider_changed(s: &mut MovieConState, which: i32, value: i32) {
    let a = s.axiscon;
    match which {
        0 => {
            s.start[a] = value - 1;
            if s.start[a] >= s.endsec[a] && s.start[a] + 1 <= s.maxend[a] {
                s.endsec[a] = s.start[a] + 1
            }
        }
        1 => {
            s.endsec[a] = value - 1;
            if s.endsec[a] <= s.start[a] {
                s.start[a] = s.endsec[a] - 1
            }
        }
        2 => s.increment[a] = value,
        _ => {}
    }
}
pub fn imc_axis_selected(s: &mut MovieConState, v: i32) {
    s.axiscon = v.clamp(0, 3) as usize
}
pub fn imc_extent_selected(s: &mut MovieConState, v: i32) {
    s.looponeway = v
}
pub fn imc_snap_selected(s: &mut MovieConState, v: i32) {
    s.autosnap = v
}
pub fn imc_start_here_selected(s: &mut MovieConState, v: i32) {
    s.start_here = v
}
pub fn imc_rate_entered(s: &mut MovieConState, v: &mut ImodView, mut rate: f32) {
    if rate <= 0. {
        imc_set_movierate(s, v, 0);
        return;
    }
    rate = rate.clamp(MINRATE, MAXRATE);
    s.realrate = rate;
    s.realint = 1000. / rate;
    v.movierate = ((s.realint / BASEINT).ln() / RATEFAC.ln() + 0.5) as i32
}
pub fn imc_increment_rate(s: &mut MovieConState, v: &mut ImodView, dir: i32) {
    imc_set_movierate(s, v, v.movierate + dir)
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
        imc_reset_all(&mut s, &v);
        assert_eq!(imc_get_start_end(&mut s, &v, 2), (0, 29));
        imc_set_special_limits(&mut s, 2, 4, 8);
        assert_eq!(imc_get_start_end(&mut s, &v, 2), (4, 8));
        let mut v = v;
        imc_set_movierate(&mut s, &mut v, 0);
        assert!((MINRATE..=MAXRATE).contains(&s.realrate));
    }
    #[test]
    fn open_controller_refreshes_the_source_non_tif_label() {
        let mut n = Native::default();
        imc_update_dialog(&MovieConState::default(), &mut n);
        imc_update_dialog(
            &MovieConState {
                dialog_open: true,
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(n.0, 1);
    }
}
