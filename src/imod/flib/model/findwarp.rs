//! Translation of `IMOD/flib/model/findwarp.f90`.
//!
//! The main program is [`findwarp`].  Its variables are the fields of
//! [`Fw`], because the four contained procedures ([`fit_local_patches`],
//! [`count_extra_eliminations`], [`set_auto_fits`], [`save_and_terminate`])
//! read and write them by host association -- including loop indices such as
//! `i`, `j`, `ind`, `locX` that the host also uses.  The external
//! subroutine `outputPatchRes` and function `determ` are
//! [`output_patch_res`] and [`determ`].
//!
//! The `equivalence`s of `numXpatchTot`..`numZpatchTot` with
//! `numXYZpatch(3)`, of `numXfit`..`numZfit` with both `numXYXfit(3)` and
//! `numXYZfit(3)`, of `numXpatchUse`.. with `numXYZpatchUse(3)` and of
//! `numXfitIn`.. with `numXYZfitIn(3)` are single three-element arrays here,
//! named after the array; the scalar names are its elements.
//!
//! Arrays keep the source's column-major linear addressing: `cenXYZ(ind, j)`
//! is `cen_xyz[(ind - 1) + (j - 1) * limPatch]`, `fitMat(j, i)` is
//! `fit_mat[(j - 1) + (i - 1) * matColDim]`, `amatSave(i, j, l)` is
//! `amat_save[(i - 1) + (j - 1) * 3 + (l - 1) * 9]`, and so on.
//!
//! Output is written through Rust's stdout, as `dopen` and the library units
//! write theirs.  The fitting goes through `solve_wo_outliers`/`multRegress`
//! (Gauss-Jordan, no LAPACK), so every value is expected to be bit-identical.
//! gfortran `MIN`/`MAX` of reals are written as `f32::min`/`max`, which
//! differ from them only for NaN operands.

use crate::imod::flib::model::get_region_contours::{
    check_boundary_conts, get_contour_array_sizes, get_extra_selections, get_region_contours,
    summarize_drops,
};
use crate::imod::flib::model::solve_wo_outliers::solve_wo_outliers;
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, frefor, frefor2, list_read};
use crate::imod::flib::subrs::hvem::get_nxyz::get_nxyz;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::readnumpatches::read_num_patches;
use crate::imod::flib::subrs::hvem::warpfile3d::{read_warp_file_header, read_warp_transforms};
use crate::imod::flib::subrs::hvem::xfinv3d::xfinv3d;
use crate::imod::flib::subrs::hvem::xfmult3d::xfmult3d;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::{exit, fortran_string, imodgetenv, numberinlist};
use crate::imod::libcfshr::parse_params::{
    pip_get_float, pip_get_float_array, pip_get_integer, pip_get_three_floats, pip_get_two_floats,
    pip_get_two_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::simplestat::sums_to_avg_sd_dbl;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// `parameter (LIMTARG = 30, LIMEXTRA = 20)` (`findwarp.f90:16`).
const LIMTARG: i32 = 30;
const LIMEXTRA: i32 = 20;
/// `parameter (numOptions = 26)` (`findwarp.f90:85`).
const NUM_OPTIONS: i32 = 26;
/// Fallback PIP table `options(1)` (`findwarp.f90:93-104`).
const OPTIONS: &str = "patch:PatchFile:FN:@output:OutputFile:FN:@region:RegionModel:FN:@\
volume:VolumeOrSizeXYZ:FN:@initial:InitialTransformFile:FN:@\
residual:ResidualPatchOutput:FN:@target:TargetMeanResidual:FA:@\
measured:MeasuredRatioMinAndMax:FP:@legacy:LegacyRatioEvaluation:I:@\
xskip:XSkipLeftAndRight:IP:@yskip:YSkipLowerAndUpper:IP:@\
zskip:ZSkipLowerAndUpper:IP:@extra:ExtraValueSelection:IPM:@\
select:SelectionCriteria:FAM:@desired:DesiredMaxResidual:F:@\
rowcol:LocalRowsAndColumns:IP:@slabs:LocalSlabs:I:@extent:MinExtentToFit:I:@\
maxfrac:MaxFractionToDrop:F:@minresid:MinResidualToDrop:F:@\
prob:CriterionProbabilities:FP:@discount:DiscountIfZeroVectors:F:@\
zero:ZeroExitCodeIfTooHigh:B:@debug:DebugAtXYZ:FT:@param:ParameterFile:PF:@\
help:usage:B:";

/// `character*5 rowSlabText(2) /'rows ', 'slabs'/`.
const ROW_SLAB_TEXT: [&str; 2] = ["rows ", "slabs"];
/// `character*5 rowSlabCapText(2) /'ROWS ', 'SLABS'/`.
const ROW_SLAB_CAP_TEXT: [&str; 2] = ["ROWS ", "SLABS"];
/// `character*1 yzText(2) /'Y', 'Z'/`.
const YZ_TEXT: [&str; 2] = ["Y", "Z"];

/// The variables of program `findwarp` (`findwarp.f90:13-80`) that its
/// contained procedures reach by host association.  Arrays are 0-based
/// `Vec`s addressed through the source's 1-based column-major arithmetic.
#[derive(Default)]
pub struct Fw {
    pub first_amat: [f32; 9],
    pub first_delta: [f32; 3],
    pub a: [f32; 9],
    pub del_xyz: [f32; 3],
    pub cen_save_min: [f32; 3],
    pub cen_save_max: [f32; 3],
    pub dev_xyz_max: [f32; 3],
    pub cen_local: [f32; 3],
    pub amat_tmp: [f32; 9],
    pub del_tmp: [f32; 3],
    pub cen_xyz_sum: [f32; 3],
    pub nxyz_vol: [i32; 3],
    pub mod_match: [i32; 3],
    pub debug_xyz: [f32; 3],
    pub dx_local: f32,
    pub dy_local: f32,
    pub dz_local: f32,
    pub ind_dropped: Vec<i32>,
    pub num_times_dropped: Vec<i32>,
    pub idrop: Vec<i32>,
    pub x_verts: Vec<f32>,
    pub y_verts: Vec<f32>,
    pub contour_z: Vec<f32>,
    pub drop_sum: Vec<f32>,
    pub fit_mat: Vec<f32>,
    pub ind_vert_start: Vec<i32>,
    pub num_verts: Vec<i32>,
    pub amat_save: Vec<f32>,
    pub cen_to_save: Vec<f32>,
    pub cen_xyz: Vec<f32>,
    pub vec_xyz: Vec<f32>,
    pub del_xyz_save: Vec<f32>,
    pub resid_sum: Vec<f32>,
    /// `devMeanAuto(indAuto, max(1, numSelectCrit))`, first dimension `lim_auto`.
    pub dev_mean_auto: Vec<f32>,
    pub dev_max_auto: Vec<f32>,
    pub lim_auto: i32,
    pub solved: Vec<bool>,
    pub exists: Vec<bool>,
    pub num_xauto_fit: Vec<i32>,
    pub num_yauto_fit: Vec<i32>,
    pub num_zauto_fit: Vec<i32>,
    pub in_row_x: Vec<i32>,
    pub in_row_y: Vec<i32>,
    pub in_row_z: Vec<i32>,
    pub num_resid: Vec<i32>,
    /// `inDiag1(0:limDiag)`: index 0 is element 0.
    pub in_diag1: Vec<i32>,
    pub in_diag2: Vec<i32>,
    pub extra_vals: Vec<f32>,
    pub auto_mean_patches: Vec<f32>,
    pub first_aloc: Vec<f32>,
    pub first_dloc: Vec<f32>,
    pub first_solve: Vec<bool>,
    /// `character*320 filename, residFile`.
    pub filename: Vec<u8>,
    pub resid_file: Vec<u8>,
    pub target_resid: Vec<f32>,
    /// `selectCrit(LIMTARG, LIMEXTRA)`.
    pub select_crit: Vec<f32>,
    pub num_select_crit: i32,
    pub ind_select: i32,
    pub icol_select: [i32; LIMEXTRA as usize],
    pub isign_select: [i32; LIMEXTRA as usize],
    /// `numXYZpatch(3)` = `numXpatchTot`, `numYpatchTot`, `numZpatchTot`.
    pub num_xyz_patch: [i32; 3],
    pub lim_patch: i32,
    pub lim_fit: i32,
    /// `numXYZfit(3)` = `numXYXfit(3)` = `numXfit`, `numYfit`, `numZfit`.
    pub num_xyz_fit: [i32; 3],
    /// `numXYZpatchUse(3)` = `numXpatchUse`, `numYpatchUse`, `numZpatchUse`.
    pub num_xyz_patch_use: [i32; 3],
    /// `numXYZfitIn(3)` = `numXfitIn`, `numYfitIn`, `numZfitIn`.
    pub num_xyz_fit_in: [i32; 3],
    pub num_col_select: i32,
    pub i_dextra: [i32; LIMEXTRA as usize],
    pub num_data: i32,
    pub num_pos_in_file: i32,
    pub i: i32,
    pub j: i32,
    pub ind: i32,
    pub num_conts: i32,
    pub ierr: i32,
    pub ind_y: i32,
    pub ind_z: i32,
    pub itmp: i32,
    pub num_xoffset: i32,
    pub num_yoffset: i32,
    pub num_zoffset: i32,
    pub if_local_slabs: i32,
    pub if_debug: i32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    pub frac_drop: f32,
    pub prob_crit: f32,
    pub abs_prob_crit: f32,
    pub elim_min_resid: f32,
    pub if_auto: i32,
    pub num_auto: i32,
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
    pub mat_col_dim: i32,
    pub max_extra: i32,
    pub num_local_done: i32,
    pub num_xlocal: i32,
    pub num_ylocal: i32,
    pub num_zlocal: i32,
    pub num_zero: i32,
    pub num_dev_sum: i32,
    pub min_extent: i32,
    pub ind_use: i32,
    pub nlist_dropped: i32,
    pub num_drop_tot: i32,
    pub loc_x: i32,
    pub loc_y: i32,
    pub loc_z: i32,
    pub lx: i32,
    pub ly: i32,
    pub lz: i32,
    pub if_use: i32,
    pub dev_mean_sum: f32,
    pub dev_max_sum: f32,
    pub dev_max_max: f32,
    pub patch_mean_num: f32,
    pub dev_max: f32,
    pub dev_mean: f32,
    pub dev_sd: f32,
    pub discount: f32,
    pub dev_mean_max: f32,
    pub determ_mean: f32,
    pub ind_lcl: i32,
    pub ipnt_max: i32,
    pub max_drop: i32,
    pub num_low_determ: i32,
    pub max_auto_patch: i32,
    pub if_in_drop: i32,
    pub num_drop: i32,
    pub if_flip: i32,
    pub icol_fixed: i32,
    pub ny_diag: i32,
    pub num_elim_select: i32,
    pub first_num_loc_x: i32,
    pub first_num_loc_y: i32,
    pub first_num_loc_z: i32,
    pub first_x_loc_start: f32,
    pub first_y_loc_start: f32,
    pub first_z_loc_start: f32,
    pub first_dx_loc: f32,
    pub first_dy_loc: f32,
    pub first_dz_loc: f32,
    pub debug_here: bool,
    pub legacy_ratios: bool,
    pub warp_read_in: bool,
    pub pip_input: bool,
}

impl Fw {
    /// Statement function `indPatch(ix, iy, iz)` (`findwarp.f90:90`).
    fn ind_patch(&self, ix: i32, iy: i32, iz: i32) -> i32 {
        ix + (iy - 1) * self.num_xyz_patch[0]
            + (iz - 1) * self.num_xyz_patch[0] * self.num_xyz_patch[1]
    }
    /// Statement function `indLocal(ix, iy, iz)` (`findwarp.f90:92`).
    fn ind_local(&self, ix: i32, iy: i32, iz: i32) -> i32 {
        ix + (iy - 1) * self.num_xlocal + (iz - 1) * self.num_xlocal * self.num_ylocal
    }
    /// `cenXYZ(ind, j)` as a linear index.
    fn ci(&self, ind: i32, j: i32) -> usize {
        (ind - 1) as usize + (j - 1) as usize * self.lim_patch as usize
    }
    /// `fitMat(j, i)` as a linear index.
    fn fi(&self, j: i32, i: i32) -> usize {
        (j - 1) as usize + (i - 1) as usize * self.mat_col_dim as usize
    }
    /// `extraVals(j, ind)` as a linear index.
    fn ei(&self, j: i32, ind: i32) -> usize {
        (j - 1) as usize + (ind - 1) as usize * self.max_extra.max(0) as usize
    }
    /// `selectCrit(k, i)` as a linear index.
    fn si(&self, k: i32, i: i32) -> usize {
        (k - 1) as usize + (i - 1) as usize * LIMTARG as usize
    }
}

/// Original program `findwarp` (`findwarp.f90:13`).
///
/// FINDWARP will solve for a series of general 3-dimensional linear
/// transformations that can then be used by WARPVOL to align two volumes to
/// each other.  It performs a series of multiple linear regression on
/// subsets of the displacements between the volumes determined at a matrix
/// of positions (patches).
/// Rust-only: one `[FWP1]` report of `findwarp` (`findwarp.f90:722-725`),
/// the averages and maxima of the mean and max residuals over the local fits.
#[derive(Clone, Copy, Debug, Default)]
pub struct FindwarpFit {
    pub dev_mean_avg: f32,
    pub dev_mean_max: f32,
    pub dev_max_avg: f32,
    pub dev_max_max: f32,
}

/// Rust-only: the values `findwarp` reports for a caller that used to parse
/// its tagged output lines (`matchorwarp`): every `[FWP1]` report in order,
/// and, for the `[FWP2]` failure (`findwarp.f90:702-707`), the lowest mean
/// residual reached.  The program's own output is unchanged; it records
/// these at the point it prints them (`CLAUDE.md`, "Wherever we control both
/// sides, use a direct function call now").
#[derive(Clone, Debug, Default)]
pub struct FindwarpResult {
    pub fits: Vec<FindwarpFit>,
    pub failed_above_target: Option<f32>,
}

thread_local! {
    /// Where [`findwarp`] records its [`FindwarpResult`] on this thread, when
    /// a direct caller set it through [`findwarp_recording`].
    static RESULT_SINK: std::cell::RefCell<Option<std::sync::Arc<std::sync::Mutex<FindwarpResult>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Rust-only: runs program [`findwarp`] on this thread with its reported
/// values recorded into `sink`.  The program ends through `exit`, so the
/// values reach the caller through `sink` rather than a return value; run it
/// under `commands::call_in_process`.
pub fn findwarp_recording(sink: std::sync::Arc<std::sync::Mutex<FindwarpResult>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    findwarp();
}

/// Rust-only: records into the direct caller's [`FindwarpResult`], if any.
fn record_result(update: impl FnOnce(&mut FindwarpResult)) {
    RESULT_SINK.with_borrow(|slot| {
        if let Some(sink) = slot {
            update(&mut sink.lock().expect("findwarp result sink"));
        }
    });
}

pub fn findwarp() {
    let mut v = Fw::default();
    let mut list_positions: Vec<i32>;
    let mut fre_num = [0.0_f32; LIMEXTRA as usize];
    let mut line: Vec<u8>;
    let mut ind_xyz = [0_i32; 3];
    let mut lim_axis: i32;
    let mut lim_diag: i32;
    let mut numeric = [0_i32; LIMEXTRA as usize];
    let mut num_target: i32 = 0;
    let mut ind_target: i32 = 1;
    let mut int_cen_pos: i32;
    let mut k: i32;
    let mut num_xexcl_high: i32;
    let mut num_yexcl_high: i32;
    let mut num_zexcl_high: i32;
    let mut if_subset: i32;
    let mut ind_auto: i32 = 0;
    let mut lim_vert = 0_i32;
    let mut lim_cont = 0_i32;
    let mut num_fields = 0_i32;
    let mut num_extra_id = 0_i32;
    let mut if_diddle: i32 = 0;
    let mut dev_mean_min: f32 = 0.;
    let mut desired_max_max: f32;
    let mut dev_mean_avg: f32;
    let mut dev_max_avg: f32;
    let mut diff_xyz = [0.0_f32; 3];
    let mut num_sum: i32;
    let mut vec_len: f32;
    let mut vec_len_sq: f32;
    let mut vec_max: f32;
    let mut sum_len: f64;
    let mut sum_len_sq: f64;
    let mut zero_exit_if_high: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // `use fortmodel` (through `get_region_contours`)
    let mut fm = FortModel::default();
    v.filename = vec![b' '; 320];
    v.resid_file = vec![b' '; 320];
    v.target_resid = vec![0.0; LIMTARG as usize];
    v.select_crit = vec![0.0; (LIMTARG * LIMEXTRA) as usize];

    //
    v.mat_col_dim = 20;
    v.num_xoffset = 0;
    v.num_yoffset = 0;
    v.num_zoffset = 0;
    num_xexcl_high = 0;
    num_yexcl_high = 0;
    num_zexcl_high = 0;
    if_subset = 0;
    v.if_debug = 0;
    v.ratio_min = 4.0;
    v.ratio_max = 20.0;
    v.frac_drop = 0.1;
    v.prob_crit = 0.01;
    v.abs_prob_crit = 0.002;
    v.elim_min_resid = 0.5;
    desired_max_max = 0.;
    v.if_local_slabs = 0;
    v.resid_file.fill(b' ');
    v.discount = 0.;
    v.min_extent = 2;
    v.lim_fit = 40000;
    v.num_auto = 0;
    v.num_select_crit = 0;
    v.num_col_select = 0;
    v.ind_select = 1;
    v.warp_read_in = false;
    v.mod_match = [-1; 3];
    zero_exit_if_high = false;
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "findwarp",
        "ERROR: FINDWARP - ",
        true,
        3,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    v.pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    // Open patch file and figure out sampled positions
    //
    let mut patch_name = String::new();
    if pip_get_in_out_file(
        "PatchFile",
        1,
        "Name of file with correlation positions and results",
        &mut patch_name,
        320,
    ) != 0
    {
        exit_error("No input patch file specified");
    }
    set_record(&mut v.filename, patch_name.as_bytes());

    let mut unit1 = BufReader::new(dopen(1, &fortran_string(&v.filename), "old", "f"));
    v.i_dextra.fill(0);
    if read_num_patches(
        &mut unit1,
        &mut v.num_data,
        &mut num_extra_id,
        &mut v.i_dextra,
        LIMEXTRA,
    ) != 0
    {
        exit_error("Reading first line of patch file");
    }
    let lp_dim = (v.num_data + 4).max(0) as usize;
    list_positions = vec![0; lp_dim * 3];
    memory_error(0, "array for position list");
    v.legacy_ratios = num_extra_id == 0;
    // `listPositions(k, j)`
    let li = |k: i32, j: i32| (k - 1) as usize + (j - 1) as usize * lp_dim;

    v.num_xyz_patch = [0; 3];
    v.max_extra = 0;
    for _i in 1..=v.num_data {
        line = read_a(&mut unit1, 320).unwrap_or_else(|err| read_runtime_error(err));
        frefor2(
            &String::from_utf8_lossy(&line),
            &mut fre_num,
            &mut numeric,
            &mut num_fields,
            LIMEXTRA,
        );
        v.max_extra = v.max_extra.max(num_fields - 6);
        for j in 1..=3 {
            int_cen_pos = fre_num[(j - 1) as usize].round() as i32;
            if numberinlist(
                &int_cen_pos,
                &list_positions[li(1, j)..],
                &v.num_xyz_patch[(j - 1) as usize],
                &0,
            ) == 0
            {
                v.num_xyz_patch[(j - 1) as usize] += 1;
                list_positions[li(v.num_xyz_patch[(j - 1) as usize], j)] = int_cen_pos;
            }
        }
    }
    //
    // sort the position lists
    //
    for i in 1..=3 {
        for j in 1..=v.num_xyz_patch[(i - 1) as usize] - 1 {
            for k in j..=v.num_xyz_patch[(i - 1) as usize] {
                if list_positions[li(j, i)] > list_positions[li(k, i)] {
                    v.itmp = list_positions[li(j, i)];
                    list_positions[li(j, i)] = list_positions[li(k, i)];
                    list_positions[li(k, i)] = v.itmp;
                }
            }
        }
    }
    //
    // Allocate arrays based on full set of patch positions
    v.lim_patch = v.num_xyz_patch[0] * v.num_xyz_patch[1] * v.num_xyz_patch[2] + 10;
    lim_axis = *v.num_xyz_patch.iter().max().unwrap() + 10;
    lim_diag = 2 * lim_axis;
    let lp = v.lim_patch as usize;
    v.amat_save = vec![0.0; 9 * lp];
    v.cen_to_save = vec![0.0; 3 * lp];
    v.del_xyz_save = vec![0.0; 3 * lp];
    v.resid_sum = vec![0.0; lp];
    v.solved = vec![false; lp];
    v.exists = vec![false; lp];
    v.cen_xyz = vec![0.0; lp * 3];
    v.vec_xyz = vec![0.0; lp * 3];
    v.in_row_x = vec![0; lim_axis as usize];
    v.in_row_y = vec![0; lim_axis as usize];
    v.in_row_z = vec![0; lim_axis as usize];
    v.num_resid = vec![0; lp];
    v.in_diag1 = vec![0; (lim_diag + 1) as usize];
    v.in_diag2 = vec![0; (lim_diag + 1) as usize];
    v.ind_dropped = vec![0; lp];
    v.num_times_dropped = vec![0; lp];
    v.drop_sum = vec![0.0; lp];
    memory_error(0, "arrays for patches");
    if v.max_extra > 0 {
        v.extra_vals = vec![0.0; v.max_extra as usize * lp];
        memory_error(0, "array for extra values");
    }
    //
    v.num_xyz_patch_use = v.num_xyz_patch;
    println!(
        " Number of patches in X, Y and Z is:{}{}{}",
        ld_int(v.num_xyz_patch[0]),
        ld_int(v.num_xyz_patch[1]),
        ld_int(v.num_xyz_patch[2])
    );
    // `rewind(1)`, `read(1,*) numData`
    drop(unit1);
    let mut unit1 = BufReader::new(
        File::open(patch_name.trim_end_matches(' '))
            .unwrap_or_else(|_| read_runtime_error(ListReadError::Error)),
    );
    if let Err(err) = list_read(&mut unit1, &mut [ListItem::Integer(&mut v.num_data)]) {
        read_runtime_error(err);
    }
    //
    if !v.pip_input {
        println!(" Enter NX, NY, NZ or name of file for tomogram being matched to");
    }
    get_nxyz(
        v.pip_input,
        "VolumeOrSizeXYZ",
        "FINDWARP",
        1,
        &mut v.nxyz_vol,
    );
    //
    // mark positions as nonexistent and fill cx from list positions
    //
    for k in 1..=v.num_xyz_patch[2] {
        for j in 1..=v.num_xyz_patch[1] {
            for i in 1..=v.num_xyz_patch[0] {
                v.ind = v.ind_patch(i, j, k);
                v.exists[(v.ind - 1) as usize] = false;
                let (c1, c2, c3) = (v.ci(v.ind, 1), v.ci(v.ind, 2), v.ci(v.ind, 3));
                v.cen_xyz[c1] = list_positions[li(i, 1)] as f32;
                v.cen_xyz[c2] = list_positions[li(j, 2)] as f32;
                v.cen_xyz[c3] = list_positions[li(k, 3)] as f32;
            }
        }
    }
    //
    // read each line, look up positions in list and store in right place
    //
    sum_len = 0.;
    sum_len_sq = 0.;
    num_sum = 0;
    vec_max = 0.;
    for _i in 1..=v.num_data {
        //
        // these are center coordinates and location of the second volume
        // relative to the first volume
        //
        line = read_a(&mut unit1, 320).unwrap_or_else(|err| read_runtime_error(err));
        frefor2(
            &String::from_utf8_lossy(&line),
            &mut fre_num,
            &mut numeric,
            &mut num_fields,
            LIMEXTRA,
        );
        for j in 1..=3 {
            int_cen_pos = fre_num[(j - 1) as usize].round() as i32;
            k = 1;
            while k <= v.num_xyz_patch[(j - 1) as usize] && int_cen_pos > list_positions[li(k, j)] {
                k += 1;
            }
            ind_xyz[(j - 1) as usize] = k;
        }
        v.ind = v.ind_patch(ind_xyz[0], ind_xyz[1], ind_xyz[2]);
        v.exists[(v.ind - 1) as usize] = true;
        for j in 1..=3 {
            let (c, jj) = (v.ci(v.ind, j), (j - 1) as usize);
            v.cen_xyz[c] = fre_num[jj];
            v.vec_xyz[c] = fre_num[jj + 3];
        }
        let (c1, c2, c3) = (v.ci(v.ind, 1), v.ci(v.ind, 2), v.ci(v.ind, 3));
        vec_len_sq = v.vec_xyz[c1] * v.vec_xyz[c1]
            + v.vec_xyz[c2] * v.vec_xyz[c2]
            + v.vec_xyz[c3] * v.vec_xyz[c3];
        vec_len = vec_len_sq.sqrt();
        sum_len += vec_len as f64;
        sum_len_sq += vec_len_sq as f64;
        num_sum += 1;
        // `max(vecMax, vecLen)` (`findwarp.f90:254`): `maxss vecLen, vecMax`
        // in the reference object.
        vec_max = maxss(vec_len, vec_max);
        for j in 7..=num_fields {
            let e = v.ei(j - 6, v.ind);
            v.extra_vals[e] = fre_num[(j - 1) as usize];
        }
    }
    //
    v.num_pos_in_file = v.num_data;
    drop(unit1);
    sums_to_avg_sd_dbl(
        sum_len,
        sum_len_sq,
        1,
        num_sum,
        &mut v.dev_mean,
        &mut v.dev_sd,
    );
    println!(
        "Patch vector length mean, SD and max:{}{}{}",
        format_f(v.dev_mean as f64, 8, 2),
        format_f(v.dev_sd as f64, 8, 2),
        format_f(vec_max as f64, 8, 2)
    );
    //
    // try to read in warp file
    if v.pip_input && pipgetstring_(b"InitialTransformFile", &mut v.filename) == 0 {
        let mut num_input = 0_i32;
        let unit_warp = read_warp_file_header(
            &mut v.filename,
            &mut num_input,
            &mut v.first_num_loc_y,
            &mut v.first_num_loc_x,
            &mut v.first_num_loc_z,
            &mut v.first_x_loc_start,
            &mut v.first_y_loc_start,
            &mut v.first_z_loc_start,
            &mut v.first_dx_loc,
            &mut v.first_dy_loc,
            &mut v.first_dz_loc,
            1.,
        );
        v.itmp = num_input;
        v.iy = v.first_num_loc_x * v.first_num_loc_y * v.first_num_loc_z;
        if v.iy == 0 {
            //
            // Regular 3D transform, just close file and read later
            drop(unit_warp);
        } else {
            //
            // Otherwise allocate needed arrays
            v.first_aloc = vec![0.0; 9 * v.iy as usize];
            v.first_dloc = vec![0.0; 3 * v.iy as usize];
            v.first_solve = vec![false; v.iy as usize];
            let (mut x_max, mut y_max, mut z_max) = (0.0_f32, 0.0_f32, 0.0_f32);
            read_warp_transforms(
                unit_warp,
                v.itmp,
                v.iy,
                v.first_num_loc_y,
                v.first_num_loc_x,
                v.first_num_loc_z,
                &mut v.first_x_loc_start,
                &mut v.first_y_loc_start,
                &mut v.first_z_loc_start,
                &mut v.first_dx_loc,
                &mut v.first_dy_loc,
                &mut v.first_dz_loc,
                1.,
                &mut x_max,
                &mut y_max,
                &mut z_max,
                &mut v.first_aloc,
                &mut v.first_dloc,
                &mut v.first_solve,
            );
            for ix in 1..=v.iy {
                v.ix = ix;
                if !v.first_solve[(ix - 1) as usize] {
                    exit_error(
                        "Warp file provided for initial transforms is not complete (filled in)",
                    );
                }
            }
            v.warp_read_in = true;
            //
            // now do sanity checks and figure out even/odd constraints
            v.filename.fill(b' ');
            // `212 format('The transform spacing in ', a,' in the the read-in file (',
            // f7.1, ') does not match the patch spacing (', f7.1, ')')`
            // (`findwarp.f90:295-305`): the third item is an INTEGER, which
            // gfortran's F descriptor rejects at run time (native exits 2 with
            // "Expected REAL for item 3").  Fixed in translation (BUGS.md): the
            // spacing is edited as a real and the intended message ends the
            // program through exitError (status 1).
            let spacing_error = |axis: &str, spacing: f32, patch: i32| -> ! {
                exit_error(&format!(
                    "The transform spacing in {axis} in the the read-in file ({}) does not match the patch spacing ({})",
                    format_f(spacing as f64, 7, 1),
                    format_f(patch as f32 as f64, 7, 1)
                ));
            };
            if v.first_num_loc_x > 1
                && (((list_positions[li(2, 1)] - list_positions[li(1, 1)]) as f32 - v.first_dx_loc)
                    .abs())
                    > 0.0002
            {
                spacing_error(
                    "X",
                    v.first_dx_loc,
                    list_positions[li(2, 1)] - list_positions[li(1, 1)],
                );
            } else if v.first_num_loc_y > 1
                && (((list_positions[li(2, 2)] - list_positions[li(1, 2)]) as f32 - v.first_dy_loc)
                    .abs())
                    > 0.0002
            {
                spacing_error(
                    "Y",
                    v.first_dy_loc,
                    list_positions[li(2, 2)] - list_positions[li(1, 2)],
                );
            } else if v.first_num_loc_z > 1
                && (((list_positions[li(2, 3)] - list_positions[li(1, 3)]) as f32 - v.first_dz_loc)
                    .abs())
                    > 0.0002
            {
                spacing_error(
                    "Z",
                    v.first_dz_loc,
                    list_positions[li(2, 3)] - list_positions[li(1, 3)],
                );
            }
            if !is_blank(&v.filename) {
                exit_error(&fortran_string(&v.filename));
            }
            //
            // Get difference between last patch position and starting solution position,
            // divide by interval between patch positions: this needs to be close to an
            // integer or close to halfway between, depending on whether an even or odd
            // number of patches were fit in that dimension
            diff_xyz[0] = -v.first_x_loc_start;
            diff_xyz[1] = -v.first_y_loc_start;
            diff_xyz[2] = -v.first_z_loc_start;
            for i in 1..=3 {
                let iu = (i - 1) as usize;
                diff_xyz[iu] = diff_xyz[iu] + list_positions[li(v.num_xyz_patch[iu], i)] as f32
                    - 0.5_f32 * v.nxyz_vol[iu] as f32;
                diff_xyz[iu] /= (list_positions[li(2, i)] - list_positions[li(1, i)]) as f32;
                diff_xyz[iu] = (diff_xyz[iu] - diff_xyz[iu].round() as i32 as f32).abs();
                if diff_xyz[iu] < 0.05 {
                    v.mod_match[iu] = 1;
                } else if diff_xyz[iu] > 0.45 {
                    v.mod_match[iu] = 0;
                } else {
                    let text = format!(
                        "The read-in transform positions on axis{}are not in register with the patch positions (fractional diff{})",
                        fmt_i(i, 2),
                        format_f(diff_xyz[iu] as f64, 6, 3)
                    );
                    // Written into `filename` and never reported
                    // (`findwarp.f90:325-327`); `modMatch` stays -1, so the
                    // search simply runs without the parity constraint.  Kept
                    // as written (BUGS.md): whether this was meant as an
                    // error, a warning or nothing is not knowable.
                    set_record(&mut v.filename, text.as_bytes());
                }
            }
        }
    }
    drop(list_positions);
    //
    // get patch region model, which should give flip value; otherwise set flip from patches
    //
    if v.pip_input {
        v.filename.fill(b' ');
        v.ierr = pipgetstring_(b"RegionModel", &mut v.filename);
    } else {
        print!(
            " Enter name of model file with contour enclosing area to use,\n or Return to use all patches: "
        );
        let _ = std::io::stdout().flush();
        line =
            read_a(&mut std::io::stdin().lock(), 320).unwrap_or_else(|err| read_runtime_error(err));
        v.filename.copy_from_slice(&line);
    }
    v.num_conts = 0;
    if !is_blank(&v.filename) {
        get_contour_array_sizes(
            &mut fm,
            &fortran_string(&v.filename),
            0,
            &mut lim_cont,
            &mut lim_vert,
        );
        v.x_verts = vec![0.0; lim_vert as usize];
        v.y_verts = vec![0.0; lim_vert as usize];
        v.contour_z = vec![0.0; lim_cont as usize];
        v.ind_vert_start = vec![0; lim_cont as usize];
        v.num_verts = vec![0; lim_cont as usize];
        memory_error(0, "arrays for boundary contours");
        get_region_contours(
            &mut fm,
            &fortran_string(&v.filename),
            "FINDWARP",
            &mut v.x_verts,
            &mut v.y_verts,
            &mut v.num_verts,
            &mut v.ind_vert_start,
            &mut v.contour_z,
            &mut v.num_conts,
            &mut v.if_flip,
            &mut lim_cont,
            &mut lim_vert,
            0,
        );
    } else {
        v.if_flip = 0;
        if v.num_xyz_patch[1] < v.num_xyz_patch[2] {
            v.if_flip = 1;
        }
    }
    //
    // Set indexes to Y-extent (row) and thickness (slab) variables
    v.ind_y = 2;
    if v.if_flip != 0 {
        v.ind_y = 3;
    }
    v.ind_z = 5 - v.ind_y;
    let izu = (v.ind_z - 1) as usize;
    let iyu = (v.ind_y - 1) as usize;
    v.num_xyz_fit_in[izu] = v.num_xyz_patch_use[izu];
    v.num_xyz_fit[izu] = v.num_xyz_patch_use[izu];
    //
    // Get most other PIP entries
    if v.pip_input {
        v.ierr = pip_get_float(b"MaxFractionToDrop", &mut v.frac_drop);
        v.ierr = pip_get_float(b"MinResidualToDrop", &mut v.elim_min_resid);
        v.ierr = pip_get_two_floats(
            b"CriterionProbabilities",
            &mut v.prob_crit,
            &mut v.abs_prob_crit,
        );
        v.ierr = pipgetstring_(b"ResidualPatchOutput", &mut v.resid_file);
        v.ierr = pip_get_float(b"DiscountIfZeroVectors", &mut v.discount);
        v.ierr = pip_get_integer(b"MinExtentToFit", &mut v.min_extent);
        v.ierr = pip_get_logical("ZeroExitCodeIfTooHigh", &mut zero_exit_if_high);
        v.min_extent = 2.max(v.min_extent);
        let mut env_line = [b' '; 320];
        if imodgetenv(b"FINDWARP_LEGACY_RATIOS", &mut env_line) == 0 {
            let mut ix = v.ix;
            let result = list_read(&mut &env_line[..], &mut [ListItem::Integer(&mut ix)]);
            v.ix = ix;
            v.ierr = if result.is_ok() { 0 } else { 1 };
            if v.ierr == 0 && v.ix != 0 {
                v.legacy_ratios = v.ix > 0;
            }
        }
        if pip_get_integer(b"LegacyRatioEvaluation", &mut v.ix) == 0 && v.ix != 0 {
            v.legacy_ratios = v.ix > 0;
        }
        let [d1, d2, d3] = &mut v.debug_xyz;
        v.if_debug = 1 - pip_get_three_floats(b"DebugAtXYZ", d1, d2, d3);
        //
        // Get, translate and validate the extra selection entries
        get_extra_selections(
            &mut v.icol_select,
            &mut v.isign_select,
            &mut v.num_col_select,
            &mut v.num_select_crit,
            &v.i_dextra,
            v.max_extra,
            &mut v.select_crit,
            LIMEXTRA,
            LIMTARG,
        );
    }
    //
    // aspectmax=3.
    //
    // THIS POINT IS LOOPED BACK TO INTERACTIVELY TO CHANGE PARAMETERS OR GO TO AUTOFIT
    //
    'label8: loop {
        if v.pip_input {
            let (mut fx, mut fy) = (v.num_xyz_fit_in[0], v.num_xyz_fit_in[iyu]);
            v.if_auto = pip_get_two_integers(b"LocalRowsAndColumns", &mut fx, &mut fy);
            v.num_xyz_fit_in[0] = fx;
            v.num_xyz_fit_in[iyu] = fy;
        } else {
            prompt(" 1 to find best warping automatically, 0 to proceed interactively: ");
            let mut value = v.if_auto;
            stdin_list(&mut [ListItem::Integer(&mut value)]);
            v.if_auto = value;
        }
        //
        if v.if_auto != 0 {
            //
            // initialize auto 1: get target and ratio parameters
            //
            if v.pip_input {
                num_target = 0;
                if pip_get_float_array(
                    b"TargetMeanResidual",
                    &mut v.target_resid,
                    &mut num_target,
                    LIMTARG,
                ) > 0
                {
                    exit_error("Target mean residual must be entered for automatic fits");
                }
                desired_max_max = 8.0_f32 * v.target_resid[(num_target - 1) as usize];
                v.ierr = pip_get_float(b"DesiredMaxResidual", &mut desired_max_max);
                v.ierr = pip_get_two_floats(
                    b"MeasuredRatioMinAndMax",
                    &mut v.ratio_min,
                    &mut v.ratio_max,
                );
            } else {
                prompt(" One or more mean residuals to achieve: ");
                line = read_a(&mut std::io::stdin().lock(), 320)
                    .unwrap_or_else(|err| read_runtime_error(err));
                v.filename.copy_from_slice(&line);
                frefor(
                    &String::from_utf8_lossy(&v.filename),
                    &mut v.target_resid,
                    &mut num_target,
                );
                prompt(&format!(
                    " Minimum and maximum ratio of measurements to unknowns to test\n  (/ for{}{}): ",
                    format_f(v.ratio_min as f64, 5, 1),
                    format_f(v.ratio_max as f64, 5, 1)
                ));
                let (mut rmin, mut rmax) = (v.ratio_min, v.ratio_max);
                stdin_list(&mut [ListItem::Real(&mut rmin), ListItem::Real(&mut rmax)]);
                v.ratio_min = rmin;
                v.ratio_max = rmax;
            }
        }
        //
        // Get rows, columns, slabs to exclude and check entries
        //
        if v.pip_input {
            v.ierr = pip_get_two_integers(
                b"XSkipLeftAndRight",
                &mut v.num_xoffset,
                &mut num_xexcl_high,
            ) + pip_get_two_integers(
                b"YSkipLowerAndUpper",
                &mut v.num_yoffset,
                &mut num_yexcl_high,
            ) + pip_get_two_integers(
                b"ZSkipLowerAndUpper",
                &mut v.num_zoffset,
                &mut num_zexcl_high,
            );
            if_subset = 3 - v.ierr;
        } else {
            prompt(" 0 to include all positions, or 1 to exclude rows or columns of patches: ");
            stdin_list(&mut [ListItem::Integer(&mut if_subset)]);
        }
        //
        if if_subset != 0 {
            loop {
                // 10
                if !v.pip_input {
                    num_xexcl_high = v.num_xyz_patch[0] - v.num_xoffset - v.num_xyz_patch_use[0];
                    prompt(&format!(
                        " # of columns to exclude on the left and right in X (/ for{}{}): ",
                        fmt_i(v.num_xoffset, 3),
                        fmt_i(num_xexcl_high, 3)
                    ));
                    stdin_list(&mut [
                        ListItem::Integer(&mut v.num_xoffset),
                        ListItem::Integer(&mut num_xexcl_high),
                    ]);
                }
                v.num_xyz_patch_use[0] = v.num_xyz_patch[0] - v.num_xoffset - num_xexcl_high;
                if v.num_xyz_patch_use[0] + v.num_xoffset > v.num_xyz_patch[0]
                    || v.num_xyz_patch_use[0] < 2
                {
                    if v.if_auto != 0 {
                        exit_error("Illegal entry for number of columns to exclude in X");
                    }
                    println!(" Illegal entry");
                    v.num_xoffset = 0;
                    v.num_xyz_patch_use[0] = v.num_xyz_patch[0];
                    continue;
                }
                break;
            }

            loop {
                // 12
                if !v.pip_input {
                    num_yexcl_high = v.num_xyz_patch[1] - v.num_yoffset - v.num_xyz_patch_use[1];
                    prompt(&format!(
                        " # of {} to exclude on the bottom and top in Y (/ for{}{}): ",
                        ROW_SLAB_TEXT[v.if_flip as usize],
                        fmt_i(v.num_yoffset, 3),
                        fmt_i(num_yexcl_high, 3)
                    ));
                    stdin_list(&mut [
                        ListItem::Integer(&mut v.num_yoffset),
                        ListItem::Integer(&mut num_yexcl_high),
                    ]);
                }
                v.num_xyz_patch_use[1] = v.num_xyz_patch[1] - v.num_yoffset - num_yexcl_high;
                if v.num_xyz_patch_use[1] + v.num_yoffset > v.num_xyz_patch[1]
                    || v.num_xyz_patch_use[1] < 1
                {
                    if v.if_auto != 0 {
                        exit_error(&format!(
                            "Illegal entry for number of {} to exclude in Y",
                            ROW_SLAB_CAP_TEXT[v.if_flip as usize]
                        ));
                    }
                    println!(" Illegal entry");
                    v.num_yoffset = 0;
                    v.num_xyz_patch_use[1] = v.num_xyz_patch[1];
                    continue;
                }
                break;
            }

            loop {
                // 14
                if !v.pip_input {
                    num_zexcl_high = v.num_xyz_patch[2] - v.num_zoffset - v.num_xyz_patch_use[2];
                    prompt(&format!(
                        " # of {} to exclude on the bottom and top in Z (/ for{}{}): ",
                        ROW_SLAB_TEXT[(1 - v.if_flip) as usize],
                        fmt_i(v.num_zoffset, 3),
                        fmt_i(num_zexcl_high, 3)
                    ));
                    stdin_list(&mut [
                        ListItem::Integer(&mut v.num_zoffset),
                        ListItem::Integer(&mut num_zexcl_high),
                    ]);
                }
                v.num_xyz_patch_use[2] = v.num_xyz_patch[2] - v.num_zoffset - num_zexcl_high;
                if v.num_xyz_patch_use[2] + v.num_zoffset > v.num_xyz_patch[2]
                    || v.num_xyz_patch_use[2] < 2
                {
                    if v.if_auto != 0 {
                        exit_error(&format!(
                            "Illegal entry for number of {} to exclude in Z",
                            ROW_SLAB_CAP_TEXT[(1 - v.if_flip) as usize]
                        ));
                    }
                    println!(" Illegal entry");
                    v.num_zoffset = 0;
                    v.num_xyz_patch_use[2] = v.num_xyz_patch[2];
                    continue;
                }
                break;
            }
            println!(
                " Remaining # of patches in X, Y and Z is:{}{}{}",
                ld_int(v.num_xyz_patch_use[0]),
                ld_int(v.num_xyz_patch_use[1]),
                ld_int(v.num_xyz_patch_use[2])
            );
        } else {
            v.num_xyz_patch_use = v.num_xyz_patch;
            v.num_xoffset = 0;
            v.num_yoffset = 0;
            v.num_zoffset = 0;
        }
        //
        // Initialize nfit for slabs before checking on slabs
        if v.pip_input && v.if_auto != 0 {
            v.num_xyz_fit[izu] = v.num_xyz_patch_use[izu];
            v.num_xyz_fit_in[izu] = v.num_xyz_fit[izu];
        }
        //
        // Figure out if doing subsets of slabs but not for interactive when
        // auto was already selected
        if (v.pip_input || v.if_auto == 0) && v.num_xyz_patch_use[izu] > 2 {
            if v.pip_input {
                v.if_local_slabs = 1 - pip_get_integer(b"LocalSlabs", &mut v.num_xyz_fit[izu]);
                if v.if_local_slabs > 0 {
                    if v.num_xyz_fit[izu] < 2 || v.num_xyz_fit[izu] > v.num_xyz_patch[izu] {
                        exit_error("Number of local slabs out of allowed range");
                    }
                    v.num_xyz_fit_in[izu] = v.num_xyz_fit[izu];
                }
            } else if v.if_auto == 0 {
                prompt(&format!(
                    " 0 to fit to all patches in {}, or 1 to fit to subsets in {}: ",
                    YZ_TEXT[(1 - v.if_flip) as usize],
                    YZ_TEXT[(1 - v.if_flip) as usize]
                ));
                stdin_list(&mut [ListItem::Integer(&mut v.if_local_slabs)]);
            }
        }

        if v.if_auto != 0 {
            //
            // set up parameters for automatic finding: make list of possible
            // nxfit, nyfit, nzfit values
            //
            ind_auto = -1;
            v.num_xyz_fit[izu] = v.num_xyz_fit[izu].min(v.num_xyz_patch_use[izu]);
            set_auto_fits(&mut v, &mut ind_auto);
            if v.num_auto == 0 {
                exit_error(
                    "No fitting parameters give the required ratio of measurements to unknowns - there are probably too few patches",
                );
            }
            v.lim_auto = ind_auto;
            let na = ind_auto as usize;
            let ns = 1.max(v.num_select_crit) as usize;
            v.num_xauto_fit = vec![0; na];
            v.num_yauto_fit = vec![0; na];
            v.num_zauto_fit = vec![0; na];
            v.auto_mean_patches = vec![0.0; na];
            v.dev_mean_auto = vec![0.0; na * ns];
            v.dev_max_auto = vec![0.0; na * ns];
            memory_error(0, "arrays for autofits");
            set_auto_fits(&mut v, &mut ind_auto);
            //
            // set up for first round and skip to set up this round's patches
            //
            ind_auto = 1;
            v.num_xyz_fit = [-100; 3];
            v.num_local_done = 0;
            dev_mean_min = 10000.;
            ind_target = 1;
            println!(
                "\nSeeking a warping with mean residual below{}",
                format_f(v.target_resid[0] as f64, 9, 3)
            );
            if v.num_select_crit > 0 {
                print_132(&v, 1);
            }
            count_extra_eliminations(&mut v);
        } else {
            //
            // Get parameter to control outlier elimination
            //
            if !v.pip_input {
                prompt(" 1 to enter parameters to control outlier elimination, 0 not to: ");
                stdin_list(&mut [ListItem::Integer(&mut if_diddle)]);
                if if_diddle != 0 {
                    prompt(&format!(
                        " Maximum fraction of patches to eliminate (/ for{}): ",
                        format_f(v.frac_drop as f64, 5, 2)
                    ));
                    stdin_list(&mut [ListItem::Real(&mut v.frac_drop)]);
                    prompt(&format!(
                        " Minimum residual needed to do any elimination (/ for{}): ",
                        format_f(v.elim_min_resid as f64, 5, 2)
                    ));
                    stdin_list(&mut [ListItem::Real(&mut v.elim_min_resid)]);
                    prompt(&format!(
                        " Criterion probability for candidates for elimination (/ for{}): ",
                        format_f(v.prob_crit as f64, 6, 3)
                    ));
                    stdin_list(&mut [ListItem::Real(&mut v.prob_crit)]);
                    prompt(&format!(
                        " Criterion probability for enforced elimination (/ for{}): ",
                        format_f(v.abs_prob_crit as f64, 6, 3)
                    ));
                    stdin_list(&mut [ListItem::Real(&mut v.abs_prob_crit)]);
                }
            }
            //
            // Initialize for first time in or back after parameter setting
            //
            v.num_xyz_fit = [-100; 3];
            v.num_local_done = 0;
            if !v.pip_input {
                println!(
                    "\n Enter 0 for the number of patches in X to loop back and find best fit \n  automatically, include a different subset of patches \n  or specify new outlier elimination parameters\n"
                );
            }
        }
        //
        // THIS POINT IS LOOPED BACK TO AFTER EVERY INTERACTIVE OR AUTO FIT
        //
        'label20: loop {
            //
            // Get the input number of local patches unless doing auto
            //
            if v.if_auto == 0 && v.if_local_slabs != 0 {
                if !v.pip_input {
                    if v.num_local_done == 0 {
                        prompt(" Number of local patches for fit in X, Y and Z: ");
                    } else {
                        prompt(
                            " Number of local patches for fit in X, Y and Z,\n    or / to redo and save last result: ",
                        );
                    }
                    let [mut f1, mut f2, mut f3] = v.num_xyz_fit_in;
                    stdin_list(&mut [
                        ListItem::Integer(&mut f1),
                        ListItem::Integer(&mut f2),
                        ListItem::Integer(&mut f3),
                    ]);
                    v.num_xyz_fit_in = [f1, f2, f3];
                }
            } else if v.if_auto == 0 {
                if !v.pip_input {
                    if v.num_local_done == 0 {
                        prompt(&format!(
                            " Number of local patches for fit in X and {}: ",
                            YZ_TEXT[v.if_flip as usize]
                        ));
                    } else {
                        prompt(&format!(
                            " Number of local patches for fit in X and {},\n    or / to redo and save last result: ",
                            YZ_TEXT[v.if_flip as usize]
                        ));
                    }
                    let (mut f1, mut f2) = (v.num_xyz_fit_in[0], v.num_xyz_fit_in[iyu]);
                    stdin_list(&mut [ListItem::Integer(&mut f1), ListItem::Integer(&mut f2)]);
                    v.num_xyz_fit_in[0] = f1;
                    v.num_xyz_fit_in[iyu] = f2;
                }
                v.num_xyz_fit_in[izu] = v.num_xyz_patch_use[izu];
            } else {
                let au = (ind_auto - 1) as usize;
                v.num_xyz_fit_in[0] = v.num_xauto_fit[au];
                v.num_xyz_fit_in[1] = v.num_yauto_fit[au];
                v.num_xyz_fit_in[2] = v.num_zauto_fit[au];
                v.patch_mean_num = v.auto_mean_patches[au];
            }
            //
            // Loop back to earlier parameter entries on a zero entry
            if v.num_xyz_fit_in[0] == 0 {
                continue 'label8;
            }
            //
            // Save data and terminate on a duplicate entry
            if v.num_xyz_fit_in == v.num_xyz_fit && v.num_local_done > 0 {
                save_and_terminate(&mut v);
            }
            //
            // Otherwise set up number of locations to fit and check entries
            v.num_xlocal = v.num_xyz_patch_use[0] + 1 - v.num_xyz_fit_in[0];
            v.num_ylocal = v.num_xyz_patch_use[1] + 1 - v.num_xyz_fit_in[1];
            v.num_zlocal = v.num_xyz_patch_use[2] + 1 - v.num_xyz_fit_in[2];
            if v.num_xyz_fit_in[0] < 2
                || v.num_xyz_fit_in[iyu] < 2
                || v.num_xlocal < 1
                || v.num_zlocal < 1
                || v.num_xyz_fit_in[izu] < 1
                || v.num_ylocal < 1
            {
                // `findwarp.f90:640-644` exits only for an automatic run; a
                // one-shot PIP run goes back to 20, which re-reads nothing, so
                // native prints "Illegal entry, try again" until killed.
                // Fixed in translation (BUGS.md): a PIP run exits here too.
                if v.if_auto != 0 || v.pip_input {
                    exit_error("Improper number to include in fit");
                }
                println!(" Illegal entry, try again");
                continue 'label20;
            }
            //
            // If arrays not allocated yet, do them to big size for interactive or current
            // size for one-shot run from PIP or biggest size from auto setup
            if v.fit_mat.is_empty() {
                if v.pip_input {
                    v.lim_fit =
                        v.num_xyz_fit_in[0] * v.num_xyz_fit_in[1] * v.num_xyz_fit_in[2] + 10;
                    if v.if_auto != 0 {
                        v.lim_fit = v.max_auto_patch + 10;
                    }
                }
                v.fit_mat = vec![0.0; (v.mat_col_dim * v.lim_fit) as usize];
                v.idrop = vec![0; v.lim_fit as usize];
                memory_error(0, "arrays for fitting");
            }
            if v.num_xyz_fit_in[0] * v.num_xyz_fit_in[1] * v.num_xyz_fit_in[2] > v.lim_fit {
                // Same endless loop natively for a PIP run
                // (`findwarp.f90:657-659`); fixed in translation (BUGS.md).
                if v.if_auto != 0 || v.pip_input {
                    exit_error("Too many patches for array sizes");
                }
                println!(" Too many patches for array sizes, try again");
                continue 'label20;
            }
            v.num_xyz_fit = v.num_xyz_fit_in;
            if v.if_auto == 0 {
                v.patch_mean_num = (v.num_xyz_fit[0] * v.num_xyz_fit[1] * v.num_xyz_fit[2]) as f32;
            }
            //
            // Do the fits finally
            //
            fit_local_patches(&mut v);
            //
            // check for auto control
            //
            if v.if_auto != 0 {
                dev_mean_avg = 10000.;
                if v.num_dev_sum > 0 {
                    dev_mean_avg = v.dev_mean_sum / v.num_dev_sum as f32;
                }
                // `findwarp.f90:675`: `minss devMeanMin, devMeanAvg`.
                dev_mean_min = minss(dev_mean_min, dev_mean_avg);
                let dai =
                    (ind_auto - 1) as usize + (v.ind_select - 1) as usize * v.lim_auto as usize;
                v.dev_mean_auto[dai] = dev_mean_avg;
                v.dev_max_auto[dai] = v.dev_max_max;
                if dev_mean_avg <= v.target_resid[(ind_target - 1) as usize]
                    && (v.ind_select >= v.num_select_crit
                        || desired_max_max <= 0.
                        || v.dev_max_max <= desired_max_max)
                {
                    if v.num_select_crit > 0 && ind_target == 1 {
                        println!(
                            "    - eliminates{} of{} patches",
                            fmt_i(v.num_elim_select, 6),
                            fmt_i(v.num_pos_in_file, 6)
                        );
                    }
                    //
                    // done: set nauto to zero to allow printing of results
                    //
                    v.num_auto = 0;
                    println!(
                        "\nDesired residual achieved with fits to{},{}, and{} patches in X, Y, Z\n",
                        fmt_i(v.num_xyz_fit[0], 3),
                        fmt_i(v.num_xyz_fit[1], 3),
                        fmt_i(v.num_xyz_fit[2], 3)
                    );
                } else {
                    ind_auto += 1;
                    if ind_auto > v.num_auto {
                        if v.num_select_crit > 0 && ind_target == 1 {
                            println!(
                                "    - eliminates{} of{} patches, gives best mean residual{}",
                                fmt_i(v.num_elim_select, 6),
                                fmt_i(v.num_pos_in_file, 6),
                                format_f(dev_mean_min as f64, 8, 3)
                            );
                        }
                        //
                        // If there are multiple selection criteria, first redo all fits
                        // with the next
                        if v.ind_select < v.num_select_crit {
                            ind_auto = 1;
                            v.num_xyz_fit[0] = -100;
                            v.num_local_done = 0;
                            v.ind_select += 1;
                            print_132(&v, v.ind_select);
                            count_extra_eliminations(&mut v);
                            continue 'label20;
                        }
                        //
                        // See if any other criteria have been met and loop back if so
                        //
                        for j in ind_target + 1..=num_target {
                            v.j = j;
                            println!(
                                "\nSeeking a warping with mean residual below{}",
                                format_f(v.target_resid[(j - 1) as usize] as f64, 9, 3)
                            );
                            v.lx = 1;
                            v.ly = 1.max(v.num_select_crit);
                            for ix in v.lx..=v.ly {
                                v.ix = ix;
                                if v.num_select_crit > 0 {
                                    print_132(&v, ix);
                                }
                                for i in 1..=v.num_auto {
                                    v.i = i;
                                    let dai =
                                        (i - 1) as usize + (ix - 1) as usize * v.lim_auto as usize;
                                    if v.dev_mean_auto[dai] <= v.target_resid[(j - 1) as usize]
                                        && (ix == v.ly
                                            || desired_max_max <= 0.
                                            || v.dev_max_auto[dai] <= desired_max_max)
                                    {
                                        ind_target = j;
                                        v.ind_select = ix;
                                        ind_auto = i;
                                        v.num_xyz_fit[0] = -100;
                                        v.num_local_done = 0;
                                        continue 'label20;
                                    }
                                }
                            }
                        }
                        //
                        // write patch file if desired on last fit before error message
                        //
                        output_patch_res(
                            &v.resid_file,
                            v.num_pos_in_file,
                            v.num_xyz_patch[0] * v.num_xyz_patch[1] * v.num_xyz_patch[2],
                            &v.exists,
                            &v.resid_sum,
                            &v.num_resid,
                            &v.ind_dropped,
                            &v.num_times_dropped,
                            v.nlist_dropped,
                            &v.cen_xyz,
                            &v.vec_xyz,
                            v.lim_patch,
                            v.max_extra,
                            &v.extra_vals,
                            &v.i_dextra,
                        );
                        if v.discount > 0. && v.num_local_done > 0 && v.num_dev_sum == 0 {
                            exit_error(
                                "All fits had too many zero vectors: raise -discount fraction or set it to zero",
                            );
                        }
                        println!(
                            "\nERROR: FINDWARP - Failed to find a warping with a mean residual below{}    [FWP2]",
                            format_f(dev_mean_min as f64, 9, 3)
                        );
                        record_result(|result| result.failed_above_target = Some(dev_mean_min));
                        if zero_exit_if_high {
                            exit(0);
                        }
                        exit(2);
                    }
                }
            }

            if v.nlist_dropped > 0 && v.num_auto == 0 {
                println!(
                    "{} separate patches eliminated as outliers a total of{} times:",
                    fmt_i(v.nlist_dropped, 5),
                    fmt_i(v.num_drop_tot, 6)
                );
                if v.nlist_dropped <= 10 {
                    println!("     patch position   # of times   mean residual");
                    for i in 1..=v.nlist_dropped {
                        let iu = (i - 1) as usize;
                        let indd = v.ind_dropped[iu];
                        println!(
                            "{}{}{}{}{}",
                            format_f(v.cen_xyz[v.ci(indd, 1)] as f64, 7, 0),
                            format_f(v.cen_xyz[v.ci(indd, 2)] as f64, 7, 0),
                            format_f(v.cen_xyz[v.ci(indd, 3)] as f64, 7, 0),
                            fmt_i(v.num_times_dropped[iu], 8),
                            format_f(
                                (v.drop_sum[iu] / v.num_times_dropped[iu] as f32) as f64,
                                12,
                                2
                            )
                        );
                    }
                } else {
                    for i in 1..=v.nlist_dropped {
                        let iu = (i - 1) as usize;
                        v.drop_sum[iu] /= v.num_times_dropped[iu] as f32;
                    }
                    summarize_drops(&v.drop_sum, v.nlist_dropped, "mean ");
                }
                println!();
            }
            //
            if v.num_local_done > 0 && v.num_auto == 0 {
                dev_mean_avg = v.dev_mean_sum / 1.max(v.num_dev_sum) as f32;
                dev_max_avg = v.dev_max_sum / 1.max(v.num_dev_sum) as f32;
                println!(
                    "Mean residual has an average of{} and a maximum of{}    [FWP1]\nMax  residual has an average of{} and a maximum of{}",
                    format_f(dev_mean_avg as f64, 8, 3),
                    format_f(v.dev_mean_max as f64, 8, 3),
                    format_f(dev_max_avg as f64, 8, 3),
                    format_f(v.dev_max_max as f64, 8, 3)
                );
                let fit = FindwarpFit {
                    dev_mean_avg,
                    dev_mean_max: v.dev_mean_max,
                    dev_max_avg,
                    dev_max_max: v.dev_max_max,
                };
                record_result(|result| result.fits.push(fit));
                if v.discount > 0. {
                    println!(
                        "\nThese averages are based on{} of{} fits",
                        fmt_i(v.num_dev_sum, 7),
                        fmt_i(v.num_local_done, 7)
                    );
                }
                //
                // finish up after auto fits: this call is unneeded and is just here to
                // make it clear
                if v.if_auto != 0 {
                    save_and_terminate(&mut v);
                }
            } else if v.if_auto == 0 {
                println!(" No locations could be solved for");
            }
            continue 'label20;
        }
    }
}

/// Original contained subroutine `fitLocalPatches` (`findwarp.f90:791`).
///
/// fitLocalPatches does the fits to all the local patches.
pub fn fit_local_patches(v: &mut Fw) {
    v.dev_mean_sum = 0.;
    v.dev_max_sum = 0.;
    v.dev_max_max = 0.;
    v.dev_mean_max = 0.;
    v.nlist_dropped = 0;
    v.num_drop_tot = 0;
    v.num_local_done = 0;
    v.num_dev_sum = 0;
    v.determ_mean = 0.;
    v.cen_save_min = [1.0e30; 3];
    v.cen_save_max = [-1.0e30; 3];
    v.num_resid.fill(0);
    v.resid_sum.fill(0.);
    let (num_xfit, num_yfit, num_zfit) = (v.num_xyz_fit[0], v.num_xyz_fit[1], v.num_xyz_fit[2]);

    v.loc_z = 1;
    while v.loc_z <= v.num_zlocal {
        v.loc_y = 1;
        while v.loc_y <= v.num_ylocal {
            v.loc_x = 1;
            while v.loc_x <= v.num_xlocal {
                v.num_data = 0;
                v.num_zero = 0;
                v.cen_xyz_sum = [0.; 3];
                //
                // count up number in each row in each dimension and on each
                // diagonal in the major dimensions
                //
                for i in 1..=num_xfit.max(num_yfit).max(num_zfit) {
                    v.i = i;
                    let iu = (i - 1) as usize;
                    v.in_row_x[iu] = 0;
                    v.in_row_y[iu] = 0;
                    v.in_row_z[iu] = 0;
                }
                v.ny_diag = num_yfit;
                if v.if_flip == 1 {
                    v.ny_diag = num_zfit;
                }
                //
                // Zero the parts of the arrays corresponding to corners, which
                // won't be tested
                for i in 0..=num_xfit + v.ny_diag - 2 {
                    v.i = i;
                    v.in_diag1[i as usize] = 0;
                    v.in_diag2[i as usize] = 0;
                }
                v.lz = v.loc_z + v.num_zoffset;
                while v.lz <= v.loc_z + v.num_zoffset + num_zfit - 1 {
                    v.ly = v.loc_y + v.num_yoffset;
                    while v.ly <= v.loc_y + v.num_yoffset + num_yfit - 1 {
                        v.lx = v.loc_x + v.num_xoffset;
                        while v.lx <= v.loc_x + v.num_xoffset + num_xfit - 1 {
                            v.if_use = 0;
                            v.ind = v.ind_patch(v.lx, v.ly, v.lz);
                            let ind = v.ind;
                            if v.exists[(ind - 1) as usize] {
                                v.if_use = 1;
                            }
                            for i in 1..=3 {
                                v.i = i;
                                let iu = (i - 1) as usize;
                                v.cen_xyz_sum[iu] = v.cen_xyz_sum[iu] + v.cen_xyz[v.ci(ind, i)]
                                    - 0.5_f32 * v.nxyz_vol[iu] as f32;
                            }
                            if v.num_conts > 0 && v.if_use > 0 {
                                let mut if_use = v.if_use;
                                check_boundary_conts(
                                    v.cen_xyz[v.ci(ind, 1)],
                                    v.cen_xyz[v.ci(ind, v.ind_y)],
                                    v.cen_xyz[v.ci(ind, v.ind_z)],
                                    &mut if_use,
                                    v.num_conts,
                                    &v.num_verts,
                                    &v.x_verts,
                                    &v.y_verts,
                                    &v.contour_z,
                                    &v.ind_vert_start,
                                );
                                v.if_use = if_use;
                            }
                            //
                            // Eliminate points that do not pass selection criteria
                            if v.num_col_select > 0 && v.if_use > 0 {
                                for i in 1..=v.num_col_select {
                                    v.i = i;
                                    let iu = (i - 1) as usize;
                                    if (v.extra_vals[v.ei(v.icol_select[iu], ind)]
                                        - v.select_crit[v.si(v.ind_select, i)])
                                        * (v.isign_select[iu] as f32)
                                        < 0.
                                    {
                                        v.if_use = 0;
                                    }
                                }
                            }
                            //
                            if v.if_use > 0 {
                                v.num_data += 1;
                                let nd = v.num_data;
                                if v.vec_xyz[v.ci(ind, 1)] == 0.
                                    && v.vec_xyz[v.ci(ind, 2)] == 0.
                                    && v.vec_xyz[v.ci(ind, 3)] == 0.
                                {
                                    v.num_zero += 1;
                                }
                                for j in 1..=3 {
                                    v.j = j;
                                    //
                                    // the regression requires coordinates of second
                                    // volume as independent variables (columns 1-3),
                                    // those in first volume as dependent variables
                                    // (stored in 5-7), to obtain transformation to get
                                    // from second to first volume cx+dx in second
                                    // volume matches cx in first volume
                                    //
                                    let f4 = v.fi(j + 4, nd);
                                    let f0 = v.fi(j, nd);
                                    v.fit_mat[f4] = v.cen_xyz[v.ci(ind, j)]
                                        - 0.5_f32 * v.nxyz_vol[(j - 1) as usize] as f32;
                                    v.fit_mat[f0] = v.fit_mat[f4] + v.vec_xyz[v.ci(ind, j)];
                                }
                                //
                                // Solve_wo_outliers uses columns 8-17; save indexi in 18
                                // Add to row counts and to diagonal counts
                                //
                                let f18 = v.fi(18, nd);
                                v.fit_mat[f18] = ind as f32;
                                v.ix = v.lx + 1 - v.loc_x - v.num_xoffset;
                                v.iy = v.ly + 1 - v.loc_y - v.num_yoffset;
                                v.iz = v.lz + 1 - v.loc_z - v.num_zoffset;
                                v.in_row_x[(v.ix - 1) as usize] += 1;
                                v.in_row_y[(v.iy - 1) as usize] += 1;
                                v.in_row_z[(v.iz - 1) as usize] += 1;
                                if v.if_flip == 1 {
                                    v.iy = v.iz;
                                }
                                v.iz = (v.ix - v.iy) + v.ny_diag - 1;
                                v.in_diag1[v.iz as usize] += 1;
                                v.iz = v.ix + v.iy - 2;
                                v.in_diag2[v.iz as usize] += 1;
                            }
                            v.lx += 1;
                        }
                        v.ly += 1;
                    }
                    v.lz += 1;
                }
                v.ind_lcl = v.ind_local(v.loc_x, v.loc_y, v.loc_z);
                let lcl = v.ind_lcl;
                let lu = (lcl - 1) as usize;
                //
                // Need regular array of positions, so use the xyzsum to get
                // censave, not the cenloc values from the regression
                //
                v.debug_here = v.if_debug != 0;
                for i in 1..=3 {
                    v.i = i;
                    let iu = (i - 1) as usize;
                    v.cen_to_save[iu + 3 * lu] =
                        v.cen_xyz_sum[iu] / (num_xfit * num_yfit * num_zfit) as f32;
                    // `findwarp.f90:899-900`: running value as destination.
                    v.cen_save_min[iu] = minss(v.cen_save_min[iu], v.cen_to_save[iu + 3 * lu]);
                    v.cen_save_max[iu] = maxss(v.cen_save_max[iu], v.cen_to_save[iu + 3 * lu]);
                    if v.debug_here {
                        v.debug_here = (v.cen_to_save[iu + 3 * lu] - v.debug_xyz[iu]).abs() < 1.;
                    }
                }
                //
                // solve for this location if there are at least half of the
                // normal number of patches present and if there are guaranteed
                // to be at least 3 patches in a different row from the dominant
                // one, even if the max are dropped from other rows
                // But treat thickness differently: if there are not enough data
                // on another layer, or if there is only one layer being fit,
                // then set the appropriate column as fixed in the fits
                //
                v.solved[lu] = v.num_data as f32 >= v.patch_mean_num / 2.0_f32;
                v.max_drop = (v.frac_drop * v.num_data as f32).round() as i32;
                v.icol_fixed = 0;
                let lim = v.num_data - 3 - v.max_drop;
                for i in 1..=num_xfit.max(num_yfit).max(num_zfit) {
                    v.i = i;
                    let iu = (i - 1) as usize;
                    if v.debug_here {
                        println!(
                            " in row{} :{}{}{}",
                            ld_int(i),
                            ld_int(v.in_row_x[iu]),
                            ld_int(v.in_row_y[iu]),
                            ld_int(v.in_row_z[iu])
                        );
                    }
                    if v.if_flip == 1 {
                        if v.in_row_x[iu] > lim || v.in_row_z[iu] > lim {
                            v.solved[lu] = false;
                        }
                        if v.in_row_y[iu] > lim || num_yfit == 1 {
                            v.icol_fixed = 2;
                        }
                    } else {
                        if v.in_row_x[iu] > lim || v.in_row_y[iu] > lim {
                            v.solved[lu] = false;
                        }
                        if v.in_row_z[iu] > lim || num_zfit == 1 {
                            v.icol_fixed = 3;
                        }
                    }
                }
                for i in 1..=num_xfit + v.ny_diag - 3 {
                    v.i = i;
                    if v.in_diag1[i as usize] > lim || v.in_diag2[i as usize] > lim {
                        v.solved[lu] = false;
                    }
                }
                if v.solved[lu] {
                    solve_wo_outliers(
                        &mut v.fit_mat,
                        v.mat_col_dim,
                        v.num_data,
                        3,
                        v.icol_fixed,
                        v.max_drop,
                        v.prob_crit,
                        v.abs_prob_crit,
                        v.elim_min_resid,
                        &mut v.idrop,
                        &mut v.num_drop,
                        &mut v.a,
                        &mut v.del_xyz,
                        &mut v.cen_local,
                        &mut v.dev_mean,
                        &mut v.dev_sd,
                        &mut v.dev_max,
                        &mut v.ipnt_max,
                        &mut v.dev_xyz_max,
                    );
                }
                if v.del_xyz[0] > 0.8_f32 * v.nxyz_vol[0] as f32
                    || v.del_xyz[1] > 0.8_f32 * v.nxyz_vol[1] as f32
                    || v.del_xyz[2] > 0.8_f32 * v.nxyz_vol[2] as f32
                {
                    v.solved[lu] = false;
                }
                if v.solved[lu] {
                    if v.debug_here {
                        for i in 1..=v.num_data {
                            v.i = i;
                            let mut text = String::new();
                            for j in 1..=7 {
                                text += &format_f(v.fit_mat[v.fi(j, i)] as f64, 9, 2);
                            }
                            println!("{text}");
                        }
                        // `do3multr` sets `cenMeanLoc` only in its fixed-column
                        // branch (`solve_wo_outliers.f90:318`), so native
                        // prints stack residue here for an ordinary fit.
                        // Fixed in translation (BUGS.md): `cenLocal` starts at
                        // 0 and keeps the last value a fixed-column fit set.
                        println!(
                            " cenloc{}{}{}",
                            ld_real(v.cen_local[0]),
                            ld_real(v.cen_local[1]),
                            ld_real(v.cen_local[2])
                        );
                        println!(
                            " censave{}{}{}",
                            ld_real(v.cen_to_save[3 * lu]),
                            ld_real(v.cen_to_save[1 + 3 * lu]),
                            ld_real(v.cen_to_save[2 + 3 * lu])
                        );
                        let mut text = String::new();
                        for k in 0..9 {
                            text += &format_f(v.a[k] as f64, 8, 3);
                        }
                        println!("{text}");
                    }
                    //
                    // Accumulate information about dropped points
                    //
                    for i in 1..=v.num_drop {
                        v.i = i;
                        v.if_in_drop = 0;
                        let row = v.idrop[(i - 1) as usize];
                        let drop_ind = v.fit_mat[v.fi(18, row)].round() as i32;
                        let resid = v.fit_mat[v.fi(4, v.num_data + i - v.num_drop)];
                        for j in 1..=v.nlist_dropped {
                            v.j = j;
                            let ju = (j - 1) as usize;
                            if drop_ind == v.ind_dropped[ju] {
                                v.if_in_drop = 1;
                                v.num_times_dropped[ju] += 1;
                                v.drop_sum[ju] += resid;
                            }
                        }
                        // `findwarp.f90:960` tests `ifInDrop` twice; whatever
                        // bound on `nlistDropped` was meant is not knowable,
                        // and the arrays are sized by `limPatch`, which the
                        // number of distinct patches cannot exceed.  Kept as
                        // written (BUGS.md).
                        if v.if_in_drop == 0 && v.if_in_drop < v.lim_fit {
                            v.nlist_dropped += 1;
                            let nu = (v.nlist_dropped - 1) as usize;
                            v.ind_dropped[nu] = drop_ind;
                            v.num_times_dropped[nu] = 1;
                            v.drop_sum[nu] = resid;
                        }
                    }
                    v.num_drop_tot += v.num_drop;
                    //
                    // if residual output asked for, accumulate info about all resids
                    //
                    if !is_blank(&v.resid_file) {
                        for i in 1..=v.num_data {
                            v.i = i;
                            let row = v.fit_mat[v.fi(5, i)].round() as i32;
                            v.ind = v.fit_mat[v.fi(18, row)].round() as i32;
                            let iu = (v.ind - 1) as usize;
                            v.num_resid[iu] += 1;
                            v.resid_sum[iu] += v.fit_mat[v.fi(4, i)];
                        }
                    }
                    //
                    if v.discount == 0. || v.num_zero as f32 / v.num_data as f32 <= v.discount {
                        v.dev_mean_sum += v.dev_mean;
                        v.dev_max_sum += v.dev_max;
                        v.num_dev_sum += 1;
                    }
                    // `findwarp.f90:985-986`: one `maxps` with the running
                    // maxima as destination.
                    v.dev_mean_max = maxss(v.dev_mean_max, v.dev_mean);
                    v.dev_max_max = maxss(v.dev_max_max, v.dev_max);
                    //
                    // mark this location as solved and save the solution.
                    //
                    if v.debug_here {
                        println!(
                            "{}{}{}{}{}{}{}{}{}",
                            fmt_i(lcl, 4),
                            fmt_i(v.loc_x, 4),
                            fmt_i(v.loc_y, 4),
                            fmt_i(v.loc_z, 4),
                            fmt_i(v.num_data, 4),
                            fmt_i(v.num_drop, 4),
                            format_f(v.del_xyz[0] as f64, 8, 1),
                            format_f(v.del_xyz[1] as f64, 8, 1),
                            format_f(v.del_xyz[2] as f64, 8, 1)
                        );
                    }
                    for i in 1..=3usize {
                        v.i = i as i32;
                        v.del_xyz_save[(i - 1) + 3 * lu] = v.del_xyz[i - 1];
                        for j in 1..=3usize {
                            v.j = j as i32;
                            v.amat_save[(i - 1) + (j - 1) * 3 + 9 * lu] =
                                v.a[(i - 1) + (j - 1) * 3];
                        }
                        if v.debug_here {
                            println!(
                                "{}{}{}{}",
                                format_f(v.a[i - 1] as f64, 10, 6),
                                format_f(v.a[i - 1 + 3] as f64, 10, 6),
                                format_f(v.a[i - 1 + 6] as f64, 10, 6),
                                format_f(v.del_xyz[i - 1] as f64, 10, 3)
                            );
                        }
                    }
                    v.num_local_done += 1;
                    v.determ_mean += determ(&v.a).abs();
                } else if v.debug_here {
                    println!(" Not solved{}", ld_int(v.num_data));
                }
                v.loc_x += 1;
            }
            v.loc_y += 1;
        }
        v.loc_z += 1;
    }
    v.determ_mean /= 1.max(v.num_local_done) as f32;
}

/// Original contained subroutine `countExtraEliminations` (`findwarp.f90:1015`).
///
/// Count up eliminations by the selection criteria.
pub fn count_extra_eliminations(v: &mut Fw) {
    if v.num_col_select > 0 {
        v.num_elim_select = 0;
        for ind in 1..=v.num_xyz_patch[0] * v.num_xyz_patch[1] * v.num_xyz_patch[2] {
            v.ind = ind;
            if v.exists[(ind - 1) as usize] {
                v.if_use = 1;
                for i in 1..=v.num_col_select {
                    v.i = i;
                    let iu = (i - 1) as usize;
                    if (v.extra_vals[v.ei(v.icol_select[iu], ind)]
                        - v.select_crit[v.si(v.ind_select, i)])
                        * (v.isign_select[iu] as f32)
                        < 0.
                    {
                        v.if_use = 0;
                    }
                }
                if v.if_use == 0 {
                    v.num_elim_select += 1;
                }
            }
        }
    }
}

/// Original contained subroutine `setAutoFits` (`findwarp.f90:1037`).
///
/// Sets up a list of number of local patches for autofits, where each one
/// has a ratio of measured to unknown within the min and max range.  Call
/// with limAuto <= 0 to just count up the number and return a good value for
/// limAuto.
pub fn set_auto_fits(v: &mut Fw, lim_auto: &mut i32) {
    let mut min_in_xyz = [0_i32; 3];
    let mut n_xyz_usable = [0_i32; 3];
    let mut min_loc_xyz = [0_i32; 3];
    let mut max_loc_xyz = [0_i32; 3];
    let mut num_exists: i32;
    let mut max_in_xyz: [i32; 3];
    let mut ratio: f32;
    let mut ratio_fac: f32;
    let mut ratio_avg: f32;
    let mut patch_mean: f32;
    let mut rtmp: f32;
    let mut avg_relax: f32;
    let izu = (v.ind_z - 1) as usize;
    avg_relax = 0.8;
    v.num_auto = 0;
    v.max_auto_patch = 0;
    max_in_xyz = v.num_xyz_patch_use;
    for k in 0..3 {
        min_in_xyz[k] = v.num_xyz_patch_use[k].min(v.min_extent);
    }
    min_in_xyz[izu] = v.num_xyz_patch_use[izu];
    if v.if_local_slabs != 0 {
        min_in_xyz[izu] = v.num_xyz_fit[izu];
    }
    ratio_fac = 4.0;
    if v.num_xyz_patch_use[izu] == 1 {
        ratio_fac = 3.0;
    }
    max_in_xyz[0] =
        (5.0_f32 * v.ratio_max * ratio_fac / (min_in_xyz[1] * min_in_xyz[2]) as f32).round() as i32;
    max_in_xyz[1] =
        (5.0_f32 * v.ratio_max * ratio_fac / (min_in_xyz[0] * min_in_xyz[2]) as f32).round() as i32;
    max_in_xyz[2] =
        (5.0_f32 * v.ratio_max * ratio_fac / (min_in_xyz[0] * min_in_xyz[1]) as f32).round() as i32;
    for k in 0..3 {
        max_in_xyz[k] = v.num_xyz_patch_use[k].min(max_in_xyz[k].max(min_in_xyz[k]));
    }
    let mut nx_fit = max_in_xyz[0];
    while nx_fit >= min_in_xyz[0] {
        //
        // Skip sizes that don't match inferred odd or even size of read-in fits
        if v.mod_match[0] >= 0 && v.first_num_loc_x > 1 && nx_fit % 2 != v.mod_match[0] {
            nx_fit -= 1;
            continue;
        }
        let mut ny_fit = max_in_xyz[1];
        while ny_fit >= min_in_xyz[1] {
            if v.mod_match[1] >= 0 && v.first_num_loc_y > 1 && ny_fit % 2 != v.mod_match[1] {
                ny_fit -= 1;
                continue;
            }
            let mut nz_fit = max_in_xyz[2];
            while nz_fit >= min_in_xyz[2] {
                if v.mod_match[2] >= 0 && v.first_num_loc_z > 1 && nz_fit % 2 != v.mod_match[2] {
                    nz_fit -= 1;
                    continue;
                }

                v.num_xlocal = v.num_xyz_patch_use[0] + 1 - nx_fit;
                v.num_ylocal = v.num_xyz_patch_use[1] + 1 - ny_fit;
                v.num_zlocal = v.num_xyz_patch_use[2] + 1 - nz_fit;
                //
                // For legacy ratios, simply take the nominal # to fit as usable number
                if v.legacy_ratios {
                    n_xyz_usable = [nx_fit, ny_fit, nz_fit];
                    num_exists =
                        nx_fit * ny_fit * nz_fit * v.num_xlocal * v.num_ylocal * v.num_zlocal;
                    avg_relax = 1.;
                } else {
                    //
                    // Otherwise, loop on all local patches for this size and determine
                    // maximum existing size in each dimension
                    n_xyz_usable = [0; 3];
                    num_exists = 0;
                    // The host variables `locX..locZ`, `lx..lz` are carried in locals
                    // through the scan and stored back once it ends, with the values the
                    // DO loops leave; nothing reads them in between.  `indPatch` is
                    // `lx + (ly - 1) * nxTot + (lz - 1) * nxTot * nyTot`.
                    let nx_tot = v.num_xyz_patch[0];
                    let nxy_tot = v.num_xyz_patch[0] * v.num_xyz_patch[1];
                    let exists = &v.exists;
                    let (xoff, yoff, zoff) = (v.num_xoffset, v.num_yoffset, v.num_zoffset);
                    let (mut loc_x, mut loc_y, mut loc_z) = (v.loc_x, v.loc_y, 1);
                    let (mut lx, mut ly, mut lz) = (v.lx, v.ly, v.lz);
                    while loc_z <= v.num_zlocal {
                        loc_y = 1;
                        while loc_y <= v.num_ylocal {
                            loc_x = 1;
                            while loc_x <= v.num_xlocal {
                                max_loc_xyz = [0; 3];
                                min_loc_xyz = v.num_xyz_patch;
                                lz = loc_z + zoff;
                                while lz <= loc_z + zoff + nz_fit - 1 {
                                    ly = loc_y + yoff;
                                    while ly <= loc_y + yoff + ny_fit - 1 {
                                        lx = loc_x + xoff;
                                        let row = (ly - 1) * nx_tot + (lz - 1) * nxy_tot;
                                        while lx <= loc_x + xoff + nx_fit - 1 {
                                            if exists[(lx + row - 1) as usize] {
                                                num_exists += 1;
                                                min_loc_xyz[0] = min_loc_xyz[0].min(lx);
                                                max_loc_xyz[0] = max_loc_xyz[0].max(lx);
                                                min_loc_xyz[1] = min_loc_xyz[1].min(ly);
                                                max_loc_xyz[1] = max_loc_xyz[1].max(ly);
                                                min_loc_xyz[2] = min_loc_xyz[2].min(lz);
                                                max_loc_xyz[2] = max_loc_xyz[2].max(lz);
                                            }
                                            lx += 1;
                                        }
                                        ly += 1;
                                    }
                                    lz += 1;
                                }
                                for k in 0..3 {
                                    n_xyz_usable[k] =
                                        n_xyz_usable[k].max(max_loc_xyz[k] + 1 - min_loc_xyz[k]);
                                }
                                loc_x += 1;
                            }
                            loc_y += 1;
                        }
                        loc_z += 1;
                    }
                    (v.loc_x, v.loc_y, v.loc_z) = (loc_x, loc_y, loc_z);
                    (v.lx, v.ly, v.lz) = (lx, ly, lz);
                }
                //
                // Get the mean number of patches in each area and a ratio based on that,
                // plus a ratio based on the nominal patch number.  The fit is acceptable
                // if the nominal ratio is within limits and the average ratio is almost
                // within the limits; also allow larger fits where the average ratio is
                // well within the upper limit
                patch_mean =
                    num_exists as f32 / (v.num_xlocal * v.num_ylocal * v.num_zlocal) as f32;
                ratio_avg = patch_mean / ratio_fac;
                ratio = (n_xyz_usable[0] * n_xyz_usable[1] * n_xyz_usable[2]) as f32 / ratio_fac;
                if (nx_fit < v.num_xyz_patch_use[0]
                    || (v.if_flip == 0 && ny_fit < v.num_xyz_patch_use[1])
                    || (v.if_flip != 0 && nz_fit < v.num_xyz_patch_use[2]))
                    && ratio >= v.ratio_min
                    && ratio_avg >= avg_relax * v.ratio_min
                    && (ratio <= v.ratio_max
                        || ratio_avg <= avg_relax * v.ratio_max
                        || (nx_fit == min_in_xyz[0]
                            && ny_fit == min_in_xyz[1]
                            && nz_fit == min_in_xyz[2]))
                {
                    v.num_auto += 1;
                    if *lim_auto > 0 {
                        let au = (v.num_auto - 1) as usize;
                        v.num_xauto_fit[au] = nx_fit;
                        v.num_yauto_fit[au] = ny_fit;
                        v.num_zauto_fit[au] = nz_fit;
                        v.auto_mean_patches[au] = patch_mean;
                        v.max_auto_patch = v.max_auto_patch.max(nx_fit * ny_fit * nz_fit);
                    }
                }
                nz_fit -= 1;
            }
            ny_fit -= 1;
        }
        nx_fit -= 1;
    }
    if *lim_auto <= 0 {
        *lim_auto = v.num_auto + 10;
        return;
    }
    //
    // sort the list by size of area in inverted order
    for i in 1..=v.num_auto - 1 {
        v.i = i;
        for j in i + 1..=v.num_auto {
            v.j = j;
            let (iu, ju) = ((i - 1) as usize, (j - 1) as usize);
            if v.auto_mean_patches[iu] < v.auto_mean_patches[ju] {
                v.itmp = v.num_xauto_fit[iu];
                v.num_xauto_fit[iu] = v.num_xauto_fit[ju];
                v.num_xauto_fit[ju] = v.itmp;
                v.itmp = v.num_yauto_fit[iu];
                v.num_yauto_fit[iu] = v.num_yauto_fit[ju];
                v.num_yauto_fit[ju] = v.itmp;
                v.itmp = v.num_zauto_fit[iu];
                v.num_zauto_fit[iu] = v.num_zauto_fit[ju];
                v.num_zauto_fit[ju] = v.itmp;
                rtmp = v.auto_mean_patches[iu];
                v.auto_mean_patches[iu] = v.auto_mean_patches[ju];
                v.auto_mean_patches[ju] = rtmp;
            }
        }
    }
}

/// Original contained subroutine `saveAndTerminate` (`findwarp.f90:1171`).
///
/// saveAndTerminate saves the results (getting initial file and output
/// file) and exits.
pub fn save_and_terminate(v: &mut Fw) -> ! {
    let mut diff_x: f32;
    let mut diff_y: f32;
    let mut diff_z: f32;
    let mut first_afwd = [0.0_f32; 9];
    let mut first_dfwd = [0.0_f32; 3];
    let mut flx: i32;
    let mut fly: i32;
    let mut flz: i32;
    let mut ind_first: i32;
    if v.num_xlocal > 1 || v.num_ylocal > 1 || v.num_zlocal > 1 {
        //
        // Eliminate locations with low determinants.  This is very
        // conservative measure before observed failures were diagonal
        // degeneracies in 2x2 fits
        v.num_low_determ = 0;
        for loc_z in 1..=v.num_zlocal {
            v.loc_z = loc_z;
            for loc_y in 1..=v.num_ylocal {
                v.loc_y = loc_y;
                for loc_x in 1..=v.num_xlocal {
                    v.loc_x = loc_x;
                    v.ind = v.ind_local(loc_x, loc_y, loc_z);
                    let iu = (v.ind - 1) as usize;
                    if v.solved[iu]
                        && determ(&v.amat_save[9 * iu..9 * iu + 9]).abs() < 0.01_f32 * v.determ_mean
                    {
                        v.solved[iu] = false;
                        v.num_low_determ += 1;
                    }
                }
            }
        }
        //
        if v.num_low_determ > 0 {
            println!(
                "\n{} fits were eliminated due to low matrix determinant",
                fmt_i(v.num_low_determ, 4)
            );
        }
        //
        if v.if_auto != 0 {
            println!();
        }
        if !v.warp_read_in {
            if v.pip_input {
                // `findwarp.f90:1200`: with no `-initial` the call leaves
                // `filename` unchanged, and it still holds the `-region`
                // model's name, so native list-reads a transform from the
                // model and dies with a runtime error (status 2) after all
                // the fitting.  Fixed in translation (BUGS.md): the name is
                // cleared first, so no entry means no initial transform (the
                // unit transform, as for an empty interactive answer).
                v.filename.fill(b' ');
                v.ierr = pipgetstring_(b"InitialTransformFile", &mut v.filename);
            } else {
                println!(
                    " Enter name of file with initial transformation, typically solve.xf   (Return if none)"
                );
                let line = read_a(&mut std::io::stdin().lock(), 320)
                    .unwrap_or_else(|err| read_runtime_error(err));
                v.filename.copy_from_slice(&line);
            }
            if !is_blank(&v.filename) {
                let mut unit1 = BufReader::new(dopen(1, &fortran_string(&v.filename), "old", "f"));
                let mut vals = [0.0_f32; 12];
                let result = {
                    let mut items: Vec<ListItem> = vals.iter_mut().map(ListItem::Real).collect();
                    list_read(&mut unit1, &mut items)
                };
                if let Err(err) = result {
                    read_runtime_error(err);
                }
                for i in 0..3 {
                    for j in 0..3 {
                        v.first_amat[i + j * 3] = vals[i * 4 + j];
                    }
                    v.first_delta[i] = vals[i * 4 + 3];
                }
            } else {
                for i in 0..3 {
                    for j in 0..3 {
                        v.first_amat[i + j * 3] = 0.;
                    }
                    v.first_amat[i + i * 3] = 1.;
                    v.first_delta[i] = 0.;
                }
            }
        }
        //
        let mut out_name = fortran_string(&v.filename);
        if pip_get_in_out_file(
            "OutputFile",
            2,
            "Name of file to place warping transformations in",
            &mut out_name,
            320,
        ) == 0
        {
            set_record(&mut v.filename, out_name.as_bytes());
            let mut unit1 = BufWriter::new(dopen(1, &fortran_string(&v.filename), "new", "f"));
            //
            // Output new style header to allow missing data
            v.dx_local = 1.;
            v.dy_local = 1.;
            v.dz_local = 1.;
            if v.num_xlocal > 1 {
                v.dx_local = (v.cen_save_max[0] - v.cen_save_min[0]) / (v.num_xlocal - 1) as f32;
            }
            if v.num_ylocal > 1 {
                v.dy_local = (v.cen_save_max[1] - v.cen_save_min[1]) / (v.num_ylocal - 1) as f32;
            }
            if v.num_zlocal > 1 {
                v.dz_local = (v.cen_save_max[2] - v.cen_save_min[2]) / (v.num_zlocal - 1) as f32;
            }
            // `104 format(i5,2i6,3f11.2,3f10.4)`
            let header_104 = |nx: i32, ny: i32, nz: i32, s: [f32; 3], d: [f32; 3]| -> String {
                format!(
                    "{}{}{}{}{}{}{}{}{}",
                    fmt_i(nx, 5),
                    fmt_i(ny, 6),
                    fmt_i(nz, 6),
                    format_f(s[0] as f64, 11, 2),
                    format_f(s[1] as f64, 11, 2),
                    format_f(s[2] as f64, 11, 2),
                    format_f(d[0] as f64, 10, 4),
                    format_f(d[1] as f64, 10, 4),
                    format_f(d[2] as f64, 10, 4)
                )
            };
            //
            // check if read-in file covers all the positions and keep now-redundant check
            // on spacing and positions in register
            if v.warp_read_in {
                println!(" Header being written:");
                println!(
                    "{}",
                    header_104(
                        v.num_xlocal,
                        v.num_ylocal,
                        v.num_zlocal,
                        v.cen_save_min,
                        [v.dx_local, v.dy_local, v.dz_local]
                    )
                );
                println!(" Read-in header:");
                println!(
                    "{}",
                    header_104(
                        v.first_num_loc_x,
                        v.first_num_loc_y,
                        v.first_num_loc_z,
                        [
                            v.first_x_loc_start,
                            v.first_y_loc_start,
                            v.first_z_loc_start
                        ],
                        [v.first_dx_loc, v.first_dy_loc, v.first_dz_loc]
                    )
                );
                if (v.dx_local - v.first_dx_loc).abs() > 0.0002
                    || (v.dy_local - v.first_dy_loc).abs() > 0.0002
                    || (v.dz_local - v.first_dz_loc).abs() > 0.0002
                {
                    exit_error(
                        "The transform spacing in the read-in warp file does not match what is being output",
                    );
                }
                if (v.num_xlocal > 1 && v.cen_save_min[0] + 0.1 < v.first_x_loc_start)
                    || (v.num_ylocal > 1 && v.cen_save_min[1] + 0.1 < v.first_y_loc_start)
                    || (v.num_zlocal > 1 && v.cen_save_min[2] + 0.1 < v.first_z_loc_start)
                {
                    exit_error(
                        "The transforms that were just computed extend to lower coordinates than the read-in transforms",
                    );
                }
                if (v.num_xlocal > 1
                    && v.cen_save_max[0] - 0.1
                        > v.first_x_loc_start + (v.first_num_loc_x - 1) as f32 * v.first_dx_loc)
                    || (v.num_ylocal > 1
                        && v.cen_save_max[1] - 0.1
                            > v.first_y_loc_start + (v.first_num_loc_y - 1) as f32 * v.first_dy_loc)
                    || (v.num_zlocal > 1
                        && v.cen_save_max[2] - 0.1
                            > v.first_z_loc_start + (v.first_num_loc_z - 1) as f32 * v.first_dz_loc)
                {
                    exit_error(
                        "The transforms that were just computed extend to higher coordinates than the read-in transforms",
                    );
                }
                diff_x = (v.cen_save_min[0] - v.first_x_loc_start) / v.dx_local;
                diff_y = (v.cen_save_min[1] - v.first_y_loc_start) / v.dy_local;
                diff_z = (v.cen_save_min[2] - v.first_z_loc_start) / v.dz_local;
                if (diff_x - diff_x.round() as i32 as f32).abs() > 0.1
                    || (diff_y - diff_y.round() as i32 as f32).abs() > 0.1
                    || (diff_z - diff_z.round() as i32 as f32).abs() > 0.1
                {
                    exit_error(
                        "The read-in warp transforms are not close enough to the positions just computed",
                    );
                }
            }
            //
            write_record(
                &mut unit1,
                &header_104(
                    v.num_xlocal,
                    v.num_ylocal,
                    v.num_zlocal,
                    v.cen_save_min,
                    [v.dx_local, v.dy_local, v.dz_local],
                ),
            );
            //
            for loc_z in 1..=v.num_zlocal {
                v.loc_z = loc_z;
                for loc_y in 1..=v.num_ylocal {
                    v.loc_y = loc_y;
                    for loc_x in 1..=v.num_xlocal {
                        v.loc_x = loc_x;
                        v.ind = v.ind_local(loc_x, loc_y, loc_z);
                        v.ind_use = v.ind;
                        let iu = (v.ind - 1) as usize;
                        let uu = (v.ind_use - 1) as usize;
                        //
                        // If this location was solved, combine with first transform and
                        // invert
                        if v.solved[iu] {
                            if v.warp_read_in {
                                //
                                // If combining with read-in transforms already inverted,
                                // compute index of read-in transform, invert it to get back
                                // to forward transform multiply that by solved transform and
                                // it is ready to invert for saving
                                flx = 0.max(
                                    ((v.cen_to_save[3 * iu] - v.first_x_loc_start) / v.first_dx_loc)
                                        .round() as i32,
                                );
                                fly = 0.max(
                                    ((v.cen_to_save[1 + 3 * iu] - v.first_y_loc_start)
                                        / v.first_dy_loc)
                                        .round() as i32,
                                );
                                flz = 0.max(
                                    ((v.cen_to_save[2 + 3 * iu] - v.first_z_loc_start)
                                        / v.first_dz_loc)
                                        .round() as i32,
                                );
                                ind_first = flx
                                    + 1
                                    + fly * v.first_num_loc_x
                                    + flz * v.first_num_loc_x * v.first_num_loc_y;
                                let fu = (ind_first - 1) as usize;
                                xfinv3d(
                                    &v.first_aloc[9 * fu..9 * fu + 9],
                                    &v.first_dloc[3 * fu..3 * fu + 3],
                                    &mut first_afwd,
                                    &mut first_dfwd,
                                );
                                xfmult3d(
                                    &first_afwd,
                                    &first_dfwd,
                                    &v.amat_save[9 * uu..9 * uu + 9],
                                    &v.del_xyz_save[3 * uu..3 * uu + 3],
                                    &mut v.amat_tmp,
                                    &mut v.del_tmp,
                                );
                            } else {
                                xfmult3d(
                                    &v.first_amat,
                                    &v.first_delta,
                                    &v.amat_save[9 * uu..9 * uu + 9],
                                    &v.del_xyz_save[3 * uu..3 * uu + 3],
                                    &mut v.amat_tmp,
                                    &mut v.del_tmp,
                                );
                            }
                            xfinv3d(&v.amat_tmp, &v.del_tmp, &mut v.a, &mut v.del_xyz);
                            write_record(
                                &mut unit1,
                                &format!(
                                    "{}{}{}",
                                    format_f(v.cen_to_save[3 * iu] as f64, 10, 1),
                                    format_f(v.cen_to_save[1 + 3 * iu] as f64, 10, 1),
                                    format_f(v.cen_to_save[2 + 3 * iu] as f64, 10, 1)
                                ),
                            );
                            for i in 0..3 {
                                write_record(
                                    &mut unit1,
                                    &format!(
                                        "{}{}{}{}",
                                        format_f(v.a[i] as f64, 11, 6),
                                        format_f(v.a[i + 3] as f64, 11, 6),
                                        format_f(v.a[i + 6] as f64, 11, 6),
                                        format_f(v.del_xyz[i] as f64, 11, 3)
                                    ),
                                );
                            }
                        }
                    }
                }
            }
            let _ = unit1.flush();
        }
    } else {
        //
        // Save a single transform if fit to whole area; won't happen if auto
        //
        println!(" Enter name of file in which to place single refining transformation");
        let line =
            read_a(&mut std::io::stdin().lock(), 320).unwrap_or_else(|err| read_runtime_error(err));
        v.filename.copy_from_slice(&line);
        let mut unit1 = BufWriter::new(dopen(1, &fortran_string(&v.filename), "new", "f"));
        for i in 0..3 {
            write_record(
                &mut unit1,
                &format!(
                    "{}{}{}{}",
                    format_f(v.a[i] as f64, 11, 6),
                    format_f(v.a[i + 3] as f64, 11, 6),
                    format_f(v.a[i + 6] as f64, 11, 6),
                    format_f(v.del_xyz[i] as f64, 11, 3)
                ),
            );
        }
        let _ = unit1.flush();
    }

    output_patch_res(
        &v.resid_file,
        v.num_pos_in_file,
        v.num_xyz_patch[0] * v.num_xyz_patch[1] * v.num_xyz_patch[2],
        &v.exists,
        &v.resid_sum,
        &v.num_resid,
        &v.ind_dropped,
        &v.num_times_dropped,
        v.nlist_dropped,
        &v.cen_xyz,
        &v.vec_xyz,
        v.lim_patch,
        v.max_extra,
        &v.extra_vals,
        &v.i_dextra,
    );

    exit(0);
}

/// Original subroutine `outputPatchRes` (`findwarp.f90:1332`).
///
/// Output new patch file with mean residuals if requested.  `cenXYZ` and
/// `vecXYZ` are `(limPatch, 3)`, `extraVals` is `(maxExtra, limPatch)`.
#[allow(clippy::too_many_arguments)]
pub fn output_patch_res(
    resid_file: &[u8],
    num_pos_in_file: i32,
    num_patch_tot: i32,
    exists: &[bool],
    resid_sum: &[f32],
    num_resid: &[i32],
    ind_dropped: &[i32],
    num_times_dropped: &[i32],
    nlist_dropped: i32,
    cen_xyz: &[f32],
    vec_xyz: &[f32],
    lim_patch: i32,
    max_extra: i32,
    extra_vals: &[f32],
    i_dextra: &[i32],
) {
    let id_resid_col = 3_i32;
    let id_outlier_col = 4_i32;
    let lp = lim_patch as usize;
    let mut dist: f32;
    let mut drop_frac: f32;
    if is_blank(resid_file) {
        return;
    }
    let mut unit1 = BufWriter::new(dopen(1, &fortran_string(resid_file), "new", "f"));
    let mut text = format!(
        "{} positions{}{}",
        fmt_i(num_pos_in_file, 7),
        fmt_i(id_resid_col, 8),
        fmt_i(id_outlier_col, 8)
    );
    if max_extra > 0 {
        for i in 1..=max_extra {
            text += &fmt_i(i_dextra[(i - 1) as usize], 8);
        }
    }
    write_record(&mut unit1, &text);
    for ind in 1..=num_patch_tot {
        let iu = (ind - 1) as usize;
        if exists[iu] {
            drop_frac = 0.;
            for i in 1..=nlist_dropped {
                let du = (i - 1) as usize;
                if ind == ind_dropped[du] {
                    drop_frac = num_times_dropped[du] as f32
                        / num_times_dropped[du].max(1).max(num_resid[iu]) as f32;
                    break;
                }
            }
            dist = resid_sum[iu] / 1.max(num_resid[iu]) as f32;
            let mut text = String::new();
            for i in 0..3 {
                text += &fmt_i(cen_xyz[iu + i * lp].round() as i32, 6);
            }
            for i in 0..3 {
                text += &format_f(vec_xyz[iu + i * lp] as f64, 9, 2);
            }
            text += &format_f(dist as f64, 10, 2);
            text += &format_f(drop_frac as f64, 7, 3);
            if max_extra > 0 {
                for i in 0..max_extra as usize {
                    text += &format_f(extra_vals[i + iu * max_extra as usize] as f64, 12, 4);
                }
            }
            write_record(&mut unit1, &text);
        }
    }
    let _ = unit1.flush();
}

/// Original function `determ` (`findwarp.f90:1376`).  `a(3,3)` is column
/// major; the `real*4` products are evaluated left to right as written.
pub fn determ(a: &[f32]) -> f32 {
    let e = |i: usize, j: usize| a[(i - 1) + (j - 1) * 3];
    e(1, 1) * e(2, 2) * e(3, 3) + e(2, 1) * e(3, 2) * e(1, 3) + e(1, 2) * e(2, 3) * e(3, 1)
        - e(1, 3) * e(2, 2) * e(3, 1)
        - e(2, 1) * e(1, 2) * e(3, 3)
        - e(1, 1) * e(3, 2) * e(2, 3)
}

/// `write(*,132) (selectCrit(k, i), i = 1, numColSelect)` with
/// `132 format(2x,'with extra column selection criteria', 10f9.3)`: format
/// reversion starts a new record with the whole format after ten values.
fn print_132(v: &Fw, k: i32) {
    let mut text = String::from("  with extra column selection criteria");
    for i in 1..=v.num_col_select {
        if i > 1 && (i - 1) % 10 == 0 {
            println!("{text}");
            text = String::from("  with extra column selection criteria");
        }
        text += &format_f(v.select_crit[v.si(k, i)] as f64, 9, 3);
    }
    println!("{text}");
}

/// A `write(*,'(1x,a,$)')` prompt: no record end, flushed for the reader.
fn prompt(text: &str) {
    print!("{text}");
    let _ = std::io::stdout().flush();
}

/// `read(5,*) items` with no `END=`/`ERR=`.
fn stdin_list(items: &mut [ListItem]) {
    if let Err(err) = list_read(&mut std::io::stdin().lock(), items) {
        read_runtime_error(err);
    }
}

/// Whether a `character` variable equals `' '` (is all blanks).
fn is_blank(record: &[u8]) -> bool {
    record.iter().all(|&b| b == b' ')
}

/// Character assignment into a fixed-length variable: cut or blank padded.
fn set_record(record: &mut [u8], text: &[u8]) {
    record.fill(b' ');
    let count = text.len().min(record.len());
    record[..count].copy_from_slice(&text[..count]);
}

/// A formatted write of one record to a file unit.
fn write_record(unit: &mut BufWriter<File>, text: &str) {
    let _ = unit.write_all(text.as_bytes());
    let _ = unit.write_all(b"\n");
}

/// gfortran `read(unit, '(a)') var` into a `character*len` variable: the
/// next record, cut at `len` or blank padded to it.  End of file is `END`.
fn read_a<R: BufRead>(unit: &mut R, len: usize) -> Result<Vec<u8>, ListReadError> {
    let mut record: Vec<u8> = Vec::new();
    match unit.read_until(b'\n', &mut record) {
        Ok(0) => return Err(ListReadError::End),
        Err(_) => return Err(ListReadError::Error),
        Ok(_) => {}
    }
    if record.last() == Some(&b'\n') {
        record.pop();
    }
    record.resize(len, b' ');
    Ok(record)
}

/// A formatted or list-directed `read` with no `END=`/`ERR=` that fails:
/// libgfortran reports it and stops with status 2.
fn read_runtime_error(err: ListReadError) -> ! {
    let _ = std::io::stdout().flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}

/// gfortran `Iw` output editing.
fn fmt_i(value: i32, w: usize) -> String {
    let text = format!("{value}");
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// A list-directed (`print *`) `integer*4` item: a blank separator (the
/// record's leading blank when it is the first item) and `I11`.
fn ld_int(value: i32) -> String {
    format!("{value:>12}")
}

/// A list-directed `real*4` item (its blank separator included): `F` form
/// with nine significant digits and four trailing blanks for magnitudes in
/// [0.1, 1e9), `E` form with a two-digit exponent otherwise.
fn ld_real(value: f32) -> String {
    if value.is_nan() {
        return format!("{:>17}", "NaN");
    }
    if value.is_infinite() {
        return format!("{:>17}", if value < 0. { "-Infinity" } else { "Infinity" });
    }
    if value == 0. {
        return format!("{:>13}    ", format!("{value:.8}"));
    }
    let scientific = format!("{:.8e}", value.abs());
    let (mantissa, power) = scientific.split_once('e').unwrap();
    let k = power.parse::<i32>().unwrap() + 1;
    if (0..=9).contains(&k) {
        let mut text = format!("{:.*}", (9 - k) as usize, value);
        if k == 9 {
            text.push('.');
        }
        format!("{text:>13}    ")
    } else {
        let e = k - 1;
        format!(
            "{:>17}",
            format!(
                "{}{}E{}{:02}",
                if value < 0. { "-" } else { "" },
                mantissa,
                if e < 0 { '-' } else { '+' },
                e.abs()
            )
        )
    }
}
