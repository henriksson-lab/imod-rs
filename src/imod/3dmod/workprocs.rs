//! Translation of `IMOD/3dmod/workprocs.cpp` and `workprocs.h`.
//!
//! Qt's `QTimer`, application event delivery, dock managers, image drawing,
//! and tile I/O are deliberately represented by [`WorkprocsNativeBoundary`].
//! The source-owned timer state and movie scheduling policy stay in this unit.
#![allow(dead_code)]

use std::time::Instant;

use crate::imod::three_dmod::control::{IMOD_DIALOG, IMODV_DIALOG};
use crate::imod::three_dmod::imodview::{IMOD_DRAW_IMAGE, IMOD_DRAW_XYZ, ImodView, ivw_set_time};
use crate::imod::three_dmod::moviecon::{
    MovieConState, imc_get_increment, imc_get_interval, imc_get_loop_mode, imc_get_start_end,
    imc_read_timer, imc_start_timer,
};
use crate::imod::three_dmod::preferences::ImodPreferences;

/// `MININTERVAL` in `workprocs.cpp` when `NO_ZERO_INTERVAL` is not defined.
pub const MIN_INTERVAL: i32 = 0;
/// `MOVIE_DEFAULT` from `moviecon.h`.
pub const MOVIE_DEFAULT: i32 = 52_965;

/// Source-visible state of a `QTimer`; actual timer delivery belongs to Qt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkprocTimer {
    pub active: bool,
    pub interval: i32,
    pub single_shot: bool,
}

impl WorkprocTimer {
    /// `QTimer::start`.
    pub fn start(&mut self, interval: i32) {
        self.active = true;
        self.interval = interval;
    }

    /// `QTimer::stop`.
    pub fn stop(&mut self) {
        self.active = false;
    }
}

/// Calls that cross from `workprocs.cpp` to Qt, image I/O, or a viewer window.
pub trait WorkprocsNativeBoundary {
    fn imod_autosave(&mut self, _view: &mut ImodView) {}
    fn ivw_work_proc(&mut self, _view: &mut ImodView) {}
    fn start_tile_loading(&mut self) {}
    fn ivw_switch_mac_menus(&mut self) {}
    fn imodv_closed(&self) -> bool {
        false
    }
    fn imod_dialog_stack_change_timeout(&mut self) {}
    fn imodv_dialog_stack_change_timeout(&mut self) {}
    fn imod_quit(&mut self) {}
    fn imod_info_input(&mut self) {}
    fn imod_draw(&mut self, _view: &mut ImodView, _flag: i32) {}
    fn milli_sleep(&mut self, _milliseconds: i32) {}
}

/// `ImodWorkproc` (`workprocs.h`).  The C++ `ImodView *mVi` is an opaque
/// identity here: Rust callers pass the borrowed view at each timer callback.
#[derive(Clone, Debug)]
pub struct ImodWorkproc {
    pub m_auto_save_timer: WorkprocTimer,
    pub m_movie_timer: WorkprocTimer,
    pub m_control_timer: WorkprocTimer,
    pub m_display_busy: i32,
    /// `ImodWorkproc::mVi`.  `ImodView` owns this `ImodWorkproc`, so the C++
    /// back-pointer cannot be a Rust borrow; it is never dereferenced here
    /// (every method takes the view as an argument), so only the identity is
    /// kept.
    pub m_vi: usize,
    pub m_timer_fired: i32,
    pub m_exit_on_timeout: bool,
    pub m_dlg_dock_timers: [Option<WorkprocTimer>; 2],
    pub m_dlg_dock_run_times: [Option<Instant>; 2],
    pub m_last_dlg_timeout: [i32; 2],
    /// `first_frame` is file-static in C++; keeping it per translated timer
    /// owner prevents cross-view state leakage in safe Rust.
    pub first_frame: i32,
    /// Source calls into the file-static `moviecon.cpp` state.
    pub movie_con: MovieConState,
}

impl ImodWorkproc {
    /// `ImodWorkproc::ImodWorkproc`.
    pub fn new(vw: &ImodView) -> Self {
        Self {
            m_auto_save_timer: WorkprocTimer::default(),
            m_movie_timer: WorkprocTimer::default(),
            m_control_timer: WorkprocTimer::default(),
            m_display_busy: 0,
            m_vi: core::ptr::from_ref(vw).addr(),
            m_timer_fired: 0,
            m_exit_on_timeout: false,
            m_dlg_dock_timers: [None, None],
            m_dlg_dock_run_times: [None, None],
            m_last_dlg_timeout: [0, 0],
            first_frame: 0,
            movie_con: MovieConState::default(),
        }
    }

    /// `ImodWorkproc::autoSaveTimeout`.
    pub fn auto_save_timeout(
        &mut self,
        view: &mut ImodView,
        native: &mut dyn WorkprocsNativeBoundary,
    ) {
        if !view.imod.is_null() {
            native.imod_autosave(view);
        }
    }

    /// `ImodWorkproc::controlTimeout`.
    pub fn control_timeout(
        &mut self,
        view: &mut ImodView,
        native: &mut dyn WorkprocsNativeBoundary,
    ) {
        native.ivw_work_proc(view);
    }

    /// `ImodWorkproc::startTileLoading`.
    pub fn start_tile_loading(&mut self) {
        // `QTimer::singleShot(1, ..., loadTileTimeout())`.
        self.m_control_timer.single_shot = true;
        self.m_control_timer.start(1);
    }

    /// `ImodWorkproc::loadTileTimeout`.
    pub fn load_tile_timeout(&mut self, native: &mut dyn WorkprocsNativeBoundary) {
        native.start_tile_loading();
    }

    /// `ImodWorkproc::startMenuSwitch`.
    pub fn start_menu_switch(&mut self) {
        // The Qt single-shot registration is kept as a native-delivery boundary.
        self.m_control_timer.single_shot = true;
        self.m_control_timer.start(1);
    }

    /// `ImodWorkproc::switchMenuTimeout`.
    pub fn switch_menu_timeout(&mut self, native: &mut dyn WorkprocsNativeBoundary) {
        native.ivw_switch_mac_menus();
    }

    /// `ImodWorkproc::startDlgDockTimer`.
    pub fn start_dlg_dock_timer(
        &mut self,
        dlg_class: i32,
        mut timeout: i32,
        native: &mut dyn WorkprocsNativeBoundary,
    ) {
        let index = dlg_class as usize;
        if index >= self.m_dlg_dock_timers.len()
            || (dlg_class == IMODV_DIALOG && native.imodv_closed())
        {
            return;
        }
        if self.m_dlg_dock_timers[index].is_none() {
            self.m_dlg_dock_timers[index] = Some(WorkprocTimer {
                single_shot: true,
                ..Default::default()
            });
            self.m_dlg_dock_run_times[index] = Some(Instant::now());
        } else if self.m_dlg_dock_run_times[index].is_some_and(|time| {
            self.m_last_dlg_timeout[index] - time.elapsed().as_millis() as i32 > timeout
        }) {
            return;
        }
        if timeout < 0 {
            timeout = -timeout;
            self.m_exit_on_timeout = true;
        }
        self.m_dlg_dock_timers[index]
            .as_mut()
            .unwrap()
            .start(timeout);
        self.m_dlg_dock_run_times[index] = Some(Instant::now());
        self.m_last_dlg_timeout[index] = timeout;
    }

    /// `ImodWorkproc::imodDockTimeout`.
    pub fn imod_dock_timeout(&mut self, native: &mut dyn WorkprocsNativeBoundary) {
        self.m_dlg_dock_timers[IMOD_DIALOG as usize] = None;
        self.m_dlg_dock_run_times[IMOD_DIALOG as usize] = None;
        if self.m_exit_on_timeout {
            self.m_exit_on_timeout = false;
            native.imod_quit();
            return;
        }
        native.imod_dialog_stack_change_timeout();
    }

    /// `ImodWorkproc::imodvDockTimeout`.
    pub fn imodv_dock_timeout(&mut self, native: &mut dyn WorkprocsNativeBoundary) {
        self.m_dlg_dock_timers[IMODV_DIALOG as usize] = None;
        self.m_dlg_dock_run_times[IMODV_DIALOG as usize] = None;
        native.imodv_dialog_stack_change_timeout();
    }

    /// `ImodWorkproc::movie_inc`.
    pub fn movie_inc(
        &mut self,
        view: &ImodView,
        mouse: &mut f32,
        movie: &mut i32,
        axis: i32,
        show: &mut i32,
    ) {
        if *movie != 0 {
            *show = 1;
            *mouse += *movie as f32 * imc_get_increment(&mut self.movie_con, view, axis) as f32;
            let (mut start, mut end) = imc_get_start_end(&mut self.movie_con, view, axis);
            if axis == 3 {
                start += 1;
                end += 1;
            }
            if *mouse < start as f32 {
                if imc_get_loop_mode(&mut self.movie_con, view) != 0 {
                    *mouse = end as f32;
                } else {
                    *mouse = start as f32;
                    *movie *= -1;
                }
            }
            if *mouse > end as f32 {
                if imc_get_loop_mode(&mut self.movie_con, view) != 0 {
                    *mouse = start as f32;
                } else {
                    *mouse = end as f32;
                    *movie *= -1;
                }
            }
        }
    }

    /// `ImodWorkproc::movieTimeout`.
    pub fn movie_timeout(&mut self, view: &mut ImodView, native: &mut dyn WorkprocsNativeBoundary) {
        if view.movie_running < 0 || view.doing_snap_draw != 0 {
            self.movie_timer(view);
        } else if view.movie_running > 0 {
            self.movie_proc(view, native);
        } else {
            self.m_movie_timer.stop();
        }
    }

    /// `ImodWorkproc::movieTimer`.
    pub fn movie_timer(&mut self, view: &mut ImodView) {
        if view.doing_snap_draw != 0 {
            return;
        }
        if self.m_display_busy != 0 {
            self.m_timer_fired = 1;
            view.movie_running = 0;
        } else {
            view.movie_running = 1;
            self.m_movie_timer.start(MIN_INTERVAL);
        }
    }

    /// `ImodWorkproc::movieProc`.
    pub fn movie_proc(&mut self, view: &mut ImodView, native: &mut dyn WorkprocsNativeBoundary) {
        self.m_display_busy = 1;
        self.m_timer_fired = 0;
        let interval = (imc_get_interval(&self.movie_con) - 0.5) as i32;
        view.movie_running = -1;
        self.m_movie_timer.start(interval);
        let mut show = 0;
        let mut x = view.xmouse;
        let mut xm = view.xmovie;
        self.movie_inc(view, &mut x, &mut xm, 0, &mut show);
        view.xmouse = x;
        view.xmovie = xm;
        let mut y = view.ymouse;
        let mut ym = view.ymovie;
        self.movie_inc(view, &mut y, &mut ym, 1, &mut show);
        view.ymouse = y;
        view.ymovie = ym;
        let mut z = view.zmouse;
        let mut zm = view.zmovie;
        self.movie_inc(view, &mut z, &mut zm, 2, &mut show);
        view.zmouse = z;
        view.zmovie = zm;
        let mut time = view.cur_time as f32;
        let mut tm = view.tmovie;
        self.movie_inc(view, &mut time, &mut tm, 3, &mut show);
        view.tmovie = tm;
        let mut draw_flag = IMOD_DRAW_XYZ;
        if view.tmovie != 0 {
            ivw_set_time(view, time as i32);
            draw_flag |= IMOD_DRAW_IMAGE;
        }
        if self.first_frame != 0 {
            imc_start_timer(&mut self.movie_con);
        } else {
            let _ = imc_read_timer(&mut self.movie_con);
        }
        self.first_frame = 0;
        if show == 0 {
            return;
        }
        native.imod_info_input();
        native.imod_draw(view, draw_flag);
        for _ in 0..50 {
            native.imod_info_input();
            if self.m_display_busy < 2 {
                break;
            }
            native.milli_sleep(50);
        }
        self.m_display_busy = 0;
        if self.m_timer_fired != 0 {
            view.movie_running = 1;
            self.m_movie_timer.start(MIN_INTERVAL);
        }
    }
}

/// `imod_start_autosave`.
pub fn imod_start_autosave(
    work: &mut ImodWorkproc,
    preferences: &ImodPreferences,
    imod_autosave: Option<&str>,
) -> i32 {
    let autosave_timeout = preferences.autosave_sec(imod_autosave);
    work.m_auto_save_timer.stop();
    if autosave_timeout > 0 {
        work.m_auto_save_timer.start(1000 * autosave_timeout);
    }
    0
}

/// `setmovievar` (file-static in `workprocs.cpp`).
pub fn set_movie_var(set: i32, movie: i32) -> i32 {
    match set {
        0 => 0,
        MOVIE_DEFAULT => movie,
        _ if movie != 0 => 0,
        _ => set,
    }
}

/// `imodMovieXYZT`.
pub fn imod_movie_xyzt(
    work: &mut ImodWorkproc,
    view: &mut ImodView,
    x: i32,
    y: i32,
    z: i32,
    t: i32,
) -> i32 {
    view.xmovie = set_movie_var(x, view.xmovie);
    view.ymovie = set_movie_var(y, view.ymovie);
    view.zmovie = set_movie_var(z, view.zmovie);
    view.tmovie = set_movie_var(t, view.tmovie);
    if view.movie_running != 0 {
        work.m_movie_timer.stop();
        view.movie_running = 0;
    }
    let interval = (imc_get_interval(&work.movie_con) + 0.5) as i32;
    if view.xmovie != 0 || view.ymovie != 0 || view.zmovie != 0 || view.tmovie != 0 {
        work.first_frame = 1;
    }
    view.movie_running = 1;
    work.m_movie_timer.start(interval);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        draws: Vec<i32>,
        inputs: i32,
        quit: i32,
    }
    impl WorkprocsNativeBoundary for Native {
        fn imod_draw(&mut self, _: &mut ImodView, flag: i32) {
            self.draws.push(flag)
        }
        fn imod_info_input(&mut self) {
            self.inputs += 1
        }
        fn imod_quit(&mut self) {
            self.quit += 1
        }
    }
    #[test]
    fn upstream_movie_flag_and_axis_policy() {
        let mut view = ImodView {
            xsize: 10,
            ysize: 10,
            zsize: 4,
            num_times: 2,
            ..Default::default()
        };
        let mut work = ImodWorkproc::new(&view);
        work.movie_con.realint = 5.;
        assert_eq!(
            imod_movie_xyzt(&mut work, &mut view, 1, 0, MOVIE_DEFAULT, 0),
            0
        );
        assert_eq!(view.xmovie, 1);
        assert_eq!(view.zmovie, 0);
        assert_eq!(work.m_movie_timer.interval, 5);
        assert_eq!(set_movie_var(3, 2), 0);
        assert_eq!(set_movie_var(3, 0), 3);
    }
    #[test]
    fn movie_bounces_and_draws_like_source() {
        let mut view = ImodView {
            xsize: 3,
            ysize: 3,
            zsize: 2,
            xmouse: 2.,
            xmovie: 1,
            ..Default::default()
        };
        let mut work = ImodWorkproc::new(&view);
        work.movie_con.realint = 1.;
        let mut native = Native::default();
        work.movie_proc(&mut view, &mut native);
        assert_eq!(view.xmouse, 2.);
        assert_eq!(view.xmovie, -1);
        assert_eq!(native.draws, vec![IMOD_DRAW_XYZ]);
    }
    #[test]
    fn negative_dock_timer_exits_on_timeout() {
        let view = ImodView::default();
        let mut work = ImodWorkproc::new(&view);
        let mut native = Native::default();
        work.start_dlg_dock_timer(IMOD_DIALOG, -2, &mut native);
        work.imod_dock_timeout(&mut native);
        assert_eq!(native.quit, 1);
    }
}
