//! Translation of `IMOD/flib/subrs/graphics/plotvars.f90`: the `plotvars`
//! module shared by `genhstplt` and the plotting routines, and
//! `lookupColorIndex`.
//!
//! The module's variables are one thread-local [`PlotVars`] (the Fortran
//! program runs on one thread; a fresh thread starts with the `data`
//! values, as a fresh process does).  `icolors(j, i)` is `icolors[i - 1][j
//! - 1]`.

use std::cell::RefCell;

/// `LIM_KEYS = 10` (`plotvars.f90:3`).
pub const LIM_KEYS: usize = 10;
/// `LIM_COLORS = 40` (`plotvars.f90:3`).
pub const LIM_COLORS: usize = 40;

/// The variables of module `plotvars` (`plotvars.f90:2-9`).
pub struct PlotVars {
    /// `character*80 keys(LIM_KEYS)`
    pub keys: [[u8; 80]; LIM_KEYS],
    /// `character*80 xaxisLabel/' '/`
    pub xaxis_label: [u8; 80],
    /// `igenPltType/0/`
    pub igen_plt_type: i32,
    /// `ifNoTerm/0/`
    pub if_no_term: i32,
    /// `ifConnect/0/`
    pub if_connect: i32,
    /// `numKeys/0/`
    pub num_keys: i32,
    /// `numColors/0/`
    pub num_colors: i32,
    /// `integer*4 icolors(6, LIM_COLORS)`
    pub icolors: [[i32; 6]; LIM_COLORS],
    /// `symConnectGap/1.1/`
    pub sym_connect_gap: f32,
    /// `zeroLineDashLen/.02/`
    pub zero_line_dash_len: f32,
    /// `screenYmin/0./, screenYmax/0./, screenXmin/0./, screenXmax/0./`
    pub screen_ymin: f32,
    pub screen_ymax: f32,
    pub screen_xmin: f32,
    pub screen_xmax: f32,
}

impl Default for PlotVars {
    fn default() -> Self {
        PlotVars {
            keys: [[b' '; 80]; LIM_KEYS],
            xaxis_label: [b' '; 80],
            igen_plt_type: 0,
            if_no_term: 0,
            if_connect: 0,
            num_keys: 0,
            num_colors: 0,
            icolors: [[0; 6]; LIM_COLORS],
            sym_connect_gap: 1.1,
            zero_line_dash_len: 0.02,
            screen_ymin: 0.,
            screen_ymax: 0.,
            screen_xmin: 0.,
            screen_xmax: 0.,
        }
    }
}

thread_local! {
    /// Module `plotvars`.
    pub static PLOTVARS: RefCell<PlotVars> = RefCell::new(PlotVars::default());
}

/// Original `lookupColorIndex` (`plotvars.f90:12`): the index (1-based) of
/// the color whose entry in column `icolumn` of `icolors` is `igroup`, or -1.
pub fn lookup_color_index(icolumn: i32, igroup: i32) -> i32 {
    PLOTVARS.with_borrow(|p| {
        for index in 1..=p.num_colors {
            if igroup == p.icolors[(index - 1) as usize][(icolumn - 1) as usize] {
                return index;
            }
        }
        -1
    })
}
