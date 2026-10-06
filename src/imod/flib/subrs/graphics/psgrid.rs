//! Translation of `IMOD/flib/subrs/graphics/psgrid.f90`: grid lines with
//! equally or logarithmically spaced ticks.

use super::psplotpak::{ps_move_abs, ps_vect_abs};

/// Original `psGridLine` (`psgrid.f90:4`).
pub fn ps_grid_line(
    x_start: f32,
    y_start: f32,
    x_range: f32,
    y_range: f32,
    n_ticks: i32,
    tick_size: f32,
) {
    if n_ticks == 0 {
        return;
    }
    let num_ticks = n_ticks.abs() + 1;
    let mut x = vec![0f32; num_ticks as usize];
    let mut y = vec![0f32; num_ticks as usize];
    for i in 1..=num_ticks {
        x[(i - 1) as usize] = x_start + (i - 1) as f32 * x_range / (num_ticks - 1) as f32;
        y[(i - 1) as usize] = y_start + (i - 1) as f32 * y_range / (num_ticks - 1) as f32;
    }
    ps_grid_base(&x, &y, x_range, n_ticks, num_ticks, tick_size);
}

/// Original `psLogGrid` (`psgrid.f90:16`).
pub fn ps_log_grid(
    x_start: f32,
    y_start: f32,
    x_range: f32,
    y_range: f32,
    tick_vals: &[f32],
    n_ticks: i32,
    tick_size: f32,
) {
    let num_ticks = n_ticks.abs();
    let mut x = vec![0f32; num_ticks as usize + 1];
    let mut y = vec![0f32; num_ticks as usize + 1];
    for i in 1..=num_ticks as usize {
        let alog_tick = (tick_vals[i - 1] / tick_vals[0]).log10();
        x[i - 1] = x_start + x_range * alog_tick;
        y[i - 1] = y_start + y_range * alog_tick;
    }
    ps_grid_base(&x, &y, x_range, n_ticks, num_ticks, tick_size);
}

/// Original `psGridBase` (`psgrid.f90:28`).
pub fn ps_grid_base(
    x: &[f32],
    y: &[f32],
    x_range: f32,
    n_ticks: i32,
    num_ticks: i32,
    tick_size: f32,
) {
    let mut if_half = 0;
    if n_ticks < 0 {
        if_half = 1;
    }
    let mut dx = tick_size;
    let mut dy: f32 = 0.;
    if x_range != 0. {
        dy = tick_size;
        dx = 0.;
    }
    for it in 1..=num_ticks.max(0) as usize {
        let xx = x[it - 1];
        let yy = y[it - 1];
        if it == 1 {
            ps_move_abs(xx, yy);
        }
        ps_vect_abs(xx, yy);
        ps_move_abs(xx + dx, yy + dy);
        if if_half != 0 {
            ps_vect_abs(xx, yy);
        }
        if if_half == 0 {
            ps_vect_abs(xx - dx, yy - dy);
        }
        ps_move_abs(xx, yy);
    }
}
