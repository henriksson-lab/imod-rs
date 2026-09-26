//! Translation of `IMOD/flib/blend/blendmont.f90`: the main program
//! `blendmont`, its eight contained procedures, and the external subroutine
//! `getPieceIndicesAndWeighting` that follows it in the file.
//!
//! BLENDMONT takes montaged images, blends their overlapping edges together,
//! and outputs the blended images with essentially no overlap.  See the man
//! page for further details.  (David Mastronarde, February 1989.)
//!
//! **Structure.**  The main program is [`blendmont`].  The contained
//! procedures (`writeEdgeCorrelations`, `findSectionEdgeFunctions`,
//! `findMultinegTransforms`, `getBestPieceShifts`, `redoEdgeFunction`,
//! `computeDxyGridMean`, `computeEdgeFractions`, `getPixelFromPieces`) are
//! one Rust function each.  They reach the main program's variables by host
//! association; the variables that cross between the host and a contained
//! procedure (either direction, or from one contained procedure to another)
//! are the fields of [`Host`], which the main program owns and lends to each
//! of them.  A host variable that a contained procedure only uses as scratch
//! and that nothing reads afterwards is a local of that procedure here.
//!
//! **Module state.**  `use blendvars` is [`BlendVars`], created once by the
//! main program (`bv`), and every unit that `use`s it takes it first (design
//! note in [`super::blendvars`]).  The Fortran unit connections for the
//! direct-access edge-function and edge-density files (units `iunEdge(ixy)` =
//! 7/8, `iunDens(ixy)` = 17/18) and the patch dumps (units 10/11) are the
//! [`BlendUnits`] the main program owns and passes to `edgeSwap`, `doEdge`,
//! `readEdgeFunc`; they are opened here through [`DirectUnit`], which
//! reproduces gfortran's direct-access record semantics (`recl` is in bytes:
//! `recl_bytes.inc` has `nbytes_recl_item = 1`).  The other Fortran units are
//! plain files: unit 3 (piece list output) and 14 (aligned coordinates,
//! gradient file) are writers/readers local to the site, and 4/5 (edge
//! correlation input) readers.  Image units (1, 2, 3, 4 through `imopen`) are
//! IMOD's C unit table.
//!
//! **External process.**  `blendmont.f90:1085` runs `clip plane` through
//! `call system(...)`.  `clip` is one of this crate's commands, so it runs in
//! this process through [`crate::imod::commands::run_in_process`] (owner
//! decision, `CLAUDE.md`, "Our own commands are called in process"); the
//! command line is still built with the source's `write` so its words are
//! the same.  See the site.
//!
//! **gfortran runtime.**  `cosd`/`sind` are `_gfortran_cosd_r4`/`_sind_r4`
//! (`nm blendmont.o`), reproduced by
//! [`crate::imod::flib::subrs::compat::gfortran_rt`].  `nint` is `lroundf`
//! (`f32::round`); real-to-integer assignment truncates.  Reals `MIN`/`MAX`
//! on data and weights are `gfortran_rt::{minss,maxss}` in the operand order
//! read from the reference object at each site (see `bsubs.rs`); the
//! integer-derived ones (`gridScale`, `edgeStart`, the edge limits) keep
//! source order, which cannot differ there.  Formatted and
//! list-directed output use the gfortran editing functions at the end of
//! this file (runtime boundary code, not translated units).  Output goes to
//! C `stdout` ([`ImodFile::Stdout`]), the stream `bsubs.rs` and
//! `solvescaling.rs` write, so the program's own lines keep their order.
//!
//! **No OpenMP** in this unit; the parallel regions it reaches are in
//! `montagexcorr.c` (reductions, kept serial in `montagexcorr.rs`) and
//! `warpinterp.c`/`taperatfill.c`/`reduce_by_binning.c` (thread-count
//! independent, already parallel there).
//!
//! **Uninitialised locals.**  The Fortran main program's locals have no
//! defined initial value; the translation starts them at zero / `.false.`.
//! Where the source can read one before assigning it, the site says so.

use super::blendvars::{
    BlendVars, IFAST_SIZ, LIM_EDG_BF, MAX_BIN, MEM_MAXIMUM, MEM_MINIMUM, MEM_PREFERRED,
};
use super::bsubs::{
    BlendUnits, DirectUnit, countedges, crossvalue, doedge, dxydgrinterp, edgeswap, fast_interp,
    find_best_gradient, find_best_shifts, find_edge_to_use, get_extra_indents, init_near_list,
    iwr_binned, joint_to_rotrans, lincom_rotrans, oneintrp, position_in_piece, read_edge_func,
    read_exclusion_model, read_list, recen_rotrans, xcorr_edge,
};
use super::edgesubs::setgridchars;
use super::setoverlap::set_overlap;
use super::shuffler::{clear_shuffle, scale_cached_pieces, shuffler};
use super::solvescaling::solve_scaling;
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{
    gfortran_cosd_r4, gfortran_sind_r4, maxss, minss,
};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, frefor, list_read};
use crate::imod::flib::subrs::hvem::get_tilt_angles::read_tilt_file;
use crate::imod::flib::subrs::hvem::getbinnedsize::get_binned_size;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist, rdlist};
use crate::imod::flib::subrs::imsubs::convert_vms::{convert_floats, convert_longs};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdsec};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::xfsubs::readdistortions::read_mag_gradients;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::libcfshr::autodoc::{
    adoc_get_float, adoc_get_two_floats, adoc_lookup_by_name_value, adoc_open_image_metadata,
    adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::{
    ImodFile, b3d_lock_file, b3d_output_file_type, exit, imod_backup_file,
    set_float_output_for_entered_mode, walltime,
};
use crate::imod::libcfshr::linearxforms::{xfcopy, xfinvert, xfmult, xfunit};
use crate::imod::libcfshr::montagexcorr::{
    mont_xc_find_binning, mont_xc_find_binning2, mont_xc_set_dist_weight_half_fall,
    montxcbasicsizes, montxcorrgetmaxes,
};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_integer_array,
    pip_get_three_floats, pip_get_two_floats, pip_get_two_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::piecefuncs::{checklist, fill_listz};
use crate::imod::libcfshr::robuststat::{rs_mad_median_outliers, rs_median};
use crate::imod::libcfshr::writelist::wrlist;
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libiimod::iihdf::ii_test_if_hdf;
use crate::imod::libiimod::parallelwrite::{
    iiu_par_wrt_flush_buffers, iiu_par_wrt_initialize, iiu_par_wrt_reclose_hdf,
    iiu_write_dummy_sec_to_hdf, par_wrt_close, par_wrt_lin, par_wrt_posn, par_wrt_properties,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_print, iiu_close, iiu_file_type, iiu_ret_adoc_index, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_extended_type, iiu_alt_mode, iiu_alt_num_extended, iiu_alt_origin,
    iiu_alt_sample, iiu_alt_size, iiu_ret_delta, iiu_ret_origin, iiu_trans_header,
    iiu_write_header, iiu_write_header_str,
};
use crate::imod::libwarp::warpfiles::{
    get_grid_parameters, get_linear_transform, get_warp_grid, set_current_warp_file,
};
use crate::imod::libwarp::warputils::{
    find_max_grid_size, get_size_adjusted_grid, interpolate_grid, read_check_warp_file,
};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// `parameter (LIMNEG = 30)` (`blendmont.f90:54`).
const LIMNEG: i32 = 30;
/// `parameter (numOptions = 82)` (`blendmont.f90:159`).
const NUM_OPTIONS: i32 = 82;
/// Fallback PIP table `options(1)` (`blendmont.f90:161-196`).
const OPTIONS: &str = "imin:ImageInputFile:FN:@plin:PieceListInput:FN:@imout:ImageOutputFile:FN:@\
plout:PieceListOutput:FN:@aligned:AlignedPieceCoordFile:FN:@\
rootname:RootNameForEdges:CH:@oldedge:OldEdgeFunctions:B:@\
perneg:FramesPerNegativeXandY:IP:@missing:MissingFromFirstNegativeXandY:IP:@\
intensity:FixIntensityFromEdges:I:@base:BaseIntensityForScaling:F:@\
sum:SumPiecesForGradient:B:@other:OtherSumGradientFile:FN:@\
flatfield:FlatfieldFile:FN:@mode:ModeToOutput:I:@float:FloatToRange:B:@\
fill:FillValue:F:@xform:TransformFile:FN:@center:TransformCenterXandY:FP:@\
unaligned:UnalignedStartingXandY:IP:@order:InterpolationOrder:I:@\
sections:SectionsToDo:LI:@xminmax:StartingAndEndingX:IP:@\
yminmax:StartingAndEndingY:IP:@nofft:NoResizeForFFT:B:@origin:AdjustOrigin:B:@\
bin:BinByFactor:I:@maxsize:MaximumNewSizeXandY:IP:@\
minoverlap:MinimumOverlapXandY:IP:@distort:DistortionField:FN:@\
imagebinned:ImagesAreBinned:F:@gradient:GradientFile:FN:@\
adjusted:AdjustedFocus:B:@addgrad:AddToGradient:FP:@tiltfile:TiltFile:FN:@\
offset:OffsetTilts:F:@geometry:TiltGeometry:FT:@\
justUndistort:JustUndistort:B:@test:TestMode:B:@sloppy:SloppyMontage:B:@\
very:VerySloppyMontage:B:@emgrid:EMGridMapFilter:F:@shift:ShiftPieces:B:@\
edge:ShiftFromEdges:B:@xcorr:ShiftFromXcorrs:B:@readxcorr:ReadInXcorrs:B:@\
weight:WeightForExpectedShifts:F:@expected:ExpectedShiftsFromEcd:FN:@\
mdoc:MdocForExpectedShifts:B:@ecdbin:BinningForEdgeShifts:F:@\
overlap:OverlapForEdgeShifts:IP:@skip:SkipEdgeModelFile:FN:@\
nonzero:NonzeroSkippedEdgeUse:I:@robust:RobustFitCriterion:F:@\
width:BlendingWidthXandY:IP:@boxsize:BoxSizeShortAndLong:IP:@\
grid:GridSpacingShortAndLong:IP:@indents:IndentShortAndLong:IP:@\
goodedge:GoodEdgeLowAndHighZ:IP:@onegood:OneGoodEdgeLimits:IAM:@\
same:SameEdgeShifts:B:@exclude:ExcludeFillFromEdges:B:@\
unsmooth:UnsmoothedPatchFile:FN:@smooth:SmoothedPatchFile:FN:@\
parallel:ParallelMode:IP:@subset:SubsetToDo:LI:@lines:LineSubsetToDo:IP:@\
boundary:BoundaryInfoFile:FN:@functions:EdgeFunctionsOnly:I:@\
aspect:AspectRatioForXcorr:F:@pad:PadFraction:F:@extra:ExtraXcorrWidth:F:@\
numpeaks:NumberOfXcorrPeaks:I:@radius1:FilterRadius1:F:@\
radius2:FilterRadius2:F:@sigma1:FilterSigma1:F:@sigma2:FilterSigma2:F:@\
treat:TreatFillForXcorr:I:@xcdbg:XcorrDebug:B:@taper:TaperFraction:F:@\
param:ParameterFile:PF:@help:usage:B:";

/// `character*4 edgeExtension(2) /'.xef', '.yef'/` (`blendmont.f90:36`).
const EDGE_EXTENSION: [&str; 2] = [".xef", ".yef"];
/// `character*5 densExtension(2) /'.xaed', '.yaed'/` (`blendmont.f90:37`).
const DENS_EXTENSION: [&str; 2] = [".xaed", ".yaed"];
/// `character*5 xcorrExtension(0:2) /'.ecd', '.xecd', '.yecd'/`
/// (`blendmont.f90:38`); `trim` removes the blank padding of `'.ecd'`.
const XCORR_EXTENSION: [&str; 3] = [".ecd", ".xecd", ".yecd"];
/// `character*6 edgeXcorrText(2) /'xcorr:', 'edges:'/` (`blendmont.f90:84`).
const EDGE_XCORR_TEXT: [&str; 2] = ["xcorr:", "edges:"];
/// `integer*4 modePower(0:15) /8, 15, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 8, 0, 0, 0/`
/// (`blendmont.f90:89`).
const MODE_POWER: [i32; 16] = [8, 15, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 8, 0, 0, 0];

/// The main program's variables that its contained procedures reach by host
/// association and that cross between procedures (module note).  Field
/// names are the source's in snake case.  Two-dimensional arrays are flat
/// column-major with leading dimension `limEdge` (`bv.lim_edge`); `hxf` is
/// `real*4 hxf(2,3,limNpc)`, flat.
#[derive(Default)]
pub struct Host {
    /// `character*320 rootName`.
    pub root_name: String,
    /// `character*320 edgeName`, a file name and message buffer.
    pub edge_name: String,
    pub if_edge_func_only: i32,
    pub ixy_func_start: i32,
    pub ixy_func_end: i32,
    pub num_skipped_edges: i32,
    /// `edgeDisplaceX(limEdge, 2)`.
    pub edge_displace_x: Vec<f32>,
    /// `edgeDisplaceY(limEdge, 2)`.
    pub edge_displace_y: Vec<f32>,
    /// `dxGridMean(limEdge, 2)`.
    pub dx_grid_mean: Vec<f32>,
    /// `dyGridMean(limEdge, 2)`.
    pub dy_grid_mean: Vec<f32>,
    /// `logical edgeDone(limEdge, 2)`.
    pub edge_done: Vec<bool>,
    /// `maxSDtemp(maxSecEdges)`.
    pub max_sd_temp: Vec<f32>,
    /// `hxf(2, 3, limNpc)`.
    pub hxf: Vec<f32>,
    pub use_edges: bool,
    pub iz_sect: i32,
    pub x_is_long_dim: bool,
    pub do_cross: bool,
    pub if_sloppy: i32,
    pub shift_each: bool,
    pub any_neg: bool,
    pub sd_crit: f32,
    pub dev_crit: f32,
    /// `integer*4 numFit(2) /5, 7/`.
    pub num_fit: [i32; 2],
    pub ipoly_order: i32,
    /// `integer*4 nskipRegress(2) /1, 2/`.
    pub nskip_regress: [i32; 2],
    pub xc_read_in: bool,
    pub xc_legacy: bool,
    pub use_expected: bool,
    pub from_edge: bool,
    pub if_old_edge: i32,
    pub test_mode: bool,
    pub iedge: i32,
    pub ixy: i32,
    pub inde: i32,
    pub edges_separated: bool,
    pub num_edges_in: i32,
    /// `logical active4(3, 2)`, dimension-reversed: `active4(i, ixy)` is
    /// `active4[ixy - 1][i - 1]`.
    pub active4: [[bool; 3]; 2],
    pub ind_edge: i32,
    pub xg: f32,
    pub ysrc: f32,
    pub num_iter: i32,
    pub pix_val: f32,
    pub one_edge_just_avg_crit: f32,
}

/// Flat index of a `(limEdge, 2)` array element `(iedge, ixy)`.
macro_rules! e2 {
    ($bv:expr, $i:expr, $ixy:expr) => {
        (($i) - 1) as usize + $bv.lim_edge as usize * (($ixy) - 1) as usize
    };
}

/// Flat index of a two-dimensional module allocatable element `a(i, j)` with
/// its descriptor `ext`.
macro_rules! a2 {
    ($ext:expr, $i:expr, $j:expr) => {
        (($i) - 1) as usize + $ext[0] * (($j) - 1) as usize
    };
}

/// Flat index of `h(k, j, ipc)` in a `(2, 3, *)` transform array.
macro_rules! h3 {
    ($k:expr, $j:expr, $ipc:expr) => {
        (($k) - 1) as usize + 2 * ((($j) - 1) as usize + 3 * (($ipc) - 1) as usize)
    };
}

/// Original: `program blendmont` (`blendmont.f90:29`).
pub fn blendmont() {
    let mut bv = BlendVars::default();
    let mut units = BlendUnits::default();
    let mut h = Host::default();
    let mut fm = FortModel::default();
    let mut out = ImodFile::Stdout;

    let mut image_in_file = String::new();
    let mut file_name: String = String::new();
    let mut pl_out_file: String = String::new();
    let mut out_file: String = String::new();
    let mut edge_name2: String = String::new();
    let mut ali_coord_file: String = String::new();
    let mut ecd_for_expected: String = String::new();
    let action_str: &str;
    let mut mxyz_in = [0i32; 3];
    // `integer*4 nxyzst(3) /0, 0, 0/`
    let nxyzst = [0i32; 3];
    let mut nxyz_tmp = [0i32; 3];
    let mut mxyz_tmp = [0i32; 3];
    // `real*4 cell(6) /1., 1., 1., 0., 0., 0./`
    let mut cell = [1.0f32, 1., 1., 0., 0., 0.];
    let delta: [f32; 3];
    let (mut x_origin, mut y_origin, mut z_origin): (f32, f32, f32) = (0.0, 0.0, 0.0);
    let mut bin_line: Vec<f32> = Vec::new();
    let mut title_str: String = String::new();
    // `integer*4 nskipRegress(2) /1, 2/`, `numFit(2) /5, 7/`
    h.nskip_regress = [1, 2];
    h.num_fit = [5, 7];
    let mut igrid_start = [0i32; 2];
    let mut i_offset = [0i32; 2];
    // `ixPcLower(15)` etc. are the multinegative joint arrays of
    // `findMultinegTransforms`; the main program reuses `ixPcLower` and
    // `ixPcUpper` as scratch.
    let mut ix_pc_lower = [0i32; 15];
    let mut ix_pc_upper = [0i32; 15];
    // `integer*4 numEdgeTmp(5,2), numDenTmp(6,2)`, flat column-major.
    let mut num_edge_tmp = [0i32; 10];
    let mut num_den_tmp = [0i32; 12];
    let mut list_z: Vec<i32> = Vec::new();
    let mut iz_want: Vec<i32> = Vec::new();
    let mut iz_all_want: Vec<i32> = Vec::new();
    let mut skip_xforms = false;
    let mut undistort_only = false;
    let mut xc_write_out: bool = false;
    let mut same_edge_shifts = false;
    // `real*4 title(20)`: never assigned; the header writes pass it with
    // `labFlag = -1`, which does not read it.
    let title = [0u8; 80];
    // `real*4 gradTmp(2,2)`, flat column-major.
    let mut grad_tmp = [0.0f32; 4];
    let mut map_all_piece: Vec<i32> = Vec::new();
    let mut num_control: Vec<i32> = Vec::new();
    let mut ix_pc_temp: Vec<i32> = Vec::new();
    let mut iy_pc_temp: Vec<i32> = Vec::new();
    let mut iz_pc_temp: Vec<i32> = Vec::new();
    let mut neg_tmp: Vec<i32> = Vec::new();
    let mut multi_neg: Vec<bool> = Vec::new();
    let mut multi_temp: Vec<bool> = Vec::new();
    let (mut any_pixels, mut in_frame, mut do_fast, mut any_lines_out): (bool, bool, bool, bool) =
        (false, false, false, false);
    let mut exist: bool = false;
    let mut outputpl: bool = false;
    let mut parallel_hdf = false;
    let mut xcorr_debug = false;
    let mut very_sloppy = false;
    let mut adjust_origin = false;
    let mut use_fill = false;
    let mut same_pieces: bool = false;
    let mut no_fft_sizes = false;
    let mut y_chunks = false;
    let mut do_warp: bool = false;
    let mut sum_for_grad = false;
    let mut expected_from_mdoc = false;
    let hdf_or_idoc_file: bool;
    let mut print_and_exit = false;
    let mut other_grad_file: String = String::new();
    let mut fastf = [0.0f32; 6];
    let mut gxf_temp = [0.0f32; 6];
    let mut fast_temp = [0.0f32; 6];
    let mut gxf: Vec<f32> = Vec::new();
    let (mut min_xwant, mut max_xwant, mut mode_parallel, mut nz_all_want) =
        (0i32, 0i32, 0i32, 0i32);
    let (mut num_out, mut num_chunks) = (0i32, 0i32);
    let (mut min_ywant, mut max_ywant, mut num_extra) = (0i32, 0i32, 0i32);
    let mut nxy_box = [0i32; 2];
    let mut n_extra = [0i32; 2];
    let mut bound_file: String = String::new();
    let (mut idf_nx, mut idf_ny, mut idf_binning) = (0i32, 0i32, 0i32);
    let mut pixel_idf = 0.0f32;
    let mut bin_ratio: f32 = 0.0;
    let mut binning_of_input: f32 = 0.0;
    let mut mode_in = 0i32;
    let mut num_list_z: i32 = 0;
    let (mut min_zpiece, mut max_piece_z) = (0i32, 0i32);
    let num_sect: i32;
    let nx_total_pix: i32;
    let ny_total_pix: i32;
    let mut mode_out: i32 = 0;
    let mut if_float: i32 = 0;
    let mut i: i32 = 0;
    let (mut min_xoverlap, mut min_yoverlap, mut num_trials): (i32, i32, i32) = (0, 0, 0);
    let mut iopt_abs: i32 = 0;
    let (mut num_xmissing, mut num_ymissing): (i32, i32) = (0, 0);
    let (mut nx_total_want, mut ny_total_want) = (0i32, 0i32);
    let (mut new_xpieces, mut new_ypieces) = (0i32, 0i32);
    let (mut new_xtotal_pix, mut new_ytotal_pix) = (0i32, 0i32);
    let (mut new_xframe, mut new_yframe): (i32, i32) = (0, 0);
    let (mut new_min_xpiece, mut new_min_ypiece): (i32, i32) = (0, 0);
    let mut if_want: i32 = 0;
    let mut num_gxforms = 0i32;
    let mut ierr: i32 = 0;
    let (mut num_xframes_per_neg, mut num_yframes_per_neg) = (0i32, 0i32);
    let (mut dmin, mut dmax) = (0.0f32, 0.0f32);
    let (mut out_min, mut out_max): (f32, f32) = (0.0, 0.0);
    let mut dflt_in_min: f32 = 0.0;
    let mut pixel_tot: f32 = 0.0;
    let mut fill_val = 0.0f32;
    let dflt_in_max: f32;
    let (mut cur_in_min, mut cur_in_max): (f32, f32) = (0.0, 0.0);
    let (mut pixel_scale, mut pixel_add): (f32, f32) = (0.0, 0.0);
    let (mut tsum, mut cur_sum, mut grand_sum, mut real_nsum): (f64, f64, f64, f64) =
        (0.0, 0.0, 0.0, 0.0);
    let (mut ix_frame, mut iy_frame, mut ipc): (i32, i32, i32) = (0, 0, 0);
    let (mut num_short, mut num_long) = (0i32, 0i32);
    let mut ind_array = 0i32;
    let mut new_use_count: i32 = 0;
    let (mut new_pc_xlow_left, mut new_pc_ylow_left) = (0i32, 0i32);
    let (mut nx_fast, mut ny_fast): (i32, i32) = (0, 0);
    let (mut ind_ylow, mut ind_yhigh, mut num_lines_out): (i32, i32, i32) = (0, 0, 0);
    let (mut ind_xlow, mut ind_xhigh) = (0i32, 0i32);
    let mut in_one_piece: i32 = 0;
    let mut iopt_neg: i32 = 0;
    let (mut ix_neg, mut iy_neg, mut num_xneg): (i32, i32, i32) = (0, 0, 0);
    let mut ineg: i32 = 0;
    let mut list_first: i32 = 0;
    let mut ipc_high: i32 = 0;
    let (mut ix, mut iy) = (0i32, 0i32);
    let mut iline_out: i32 = 0;
    let mut ifill: i32 = 0;
    let (mut num_along, mut num_across, mut num_ed): (i32, i32, i32) = (0, 0, 0);
    let mut len_record: i32 = 0;
    let (mut dmin_out, mut dmax_out): (f32, f32) = (0.0, 0.0);
    let mut tmean: f32 = 0.0;
    let mut grid_scale: f32 = 0.0;
    let mut em_grid_filter: f32 = 0.0;
    let mut num_zwant: i32 = 0;
    let (mut new_xoverlap, mut new_yoverlap) = (0i32, 0i32);
    let mut iwant: i32 = 0;
    let mut if_diddle: i32 = 0;
    let (mut ix_out, mut iy_out) = (0i32, 0i32);
    let mut if_revise = 0i32;
    let mut ipc_lower: i32 = 0;
    let mut err_lim = 0.0f32;
    let mut warp_scale = 0.0f32;
    let (mut warp_xoffset, mut warp_yoffset) = (0.0f32, 0.0f32);
    let mut ind_gxf = 0i32;
    let mut line_base: i32 = 0;
    let (mut x1, mut y1, mut w1) = (0.0f32, 0.0f32, 0.0f32);
    let mut val: f32 = 0.0;
    let (mut nx_grid_in, mut ny_grid_in): (i32, i32) = (0, 0);
    let mut xtmp: f32 = 0.0;
    let (mut del_dmag_per_um, mut del_rot_per_um) = (0.0f32, 0.0f32);
    let mut tilt_offset = 0.0f32;
    let mut ecd_binning: f32 = 0.0;
    let mut i_binning: i32 = 0;
    let mut ny_write: i32 = 0;
    let mut num_zero: i32 = 0;
    let most_needed: i32;
    let mut lim_xcorr_peaks = 0i32;
    let (mut line_offset, mut lines_buffered, mut i_buffer_base): (i32, i32, i32) = (0, 0, 0);
    let (mut ix_offset, mut iy_offset): (i32, i32) = (0, 0);
    let max_xcorr_binning: i32;
    let nxy_xcorr_target: i32;
    let (mut line_start, mut line_end) = (0i32, 0i32);
    let (mut nxy_padded, mut nxy_boxed) = (0i32, 0i32);
    let mut if_use_adjusted: i32 = 0;
    let mut iy_out_offset: i32 = 0;
    let mut ip_first = [0i32; 4];
    let mut num_first = 0i32;
    let mut num_lines_write: i32 = 0;
    let (mut iedge_del_x, mut iedge_del_y): (i32, i32) = (0, 0);
    let (mut ix_unali_start, mut iy_unali_start) = (0i32, 0i32);
    let max_sec_edges: i32;
    let (mut iwarp_nx, mut iwarp_ny) = (0i32, 0i32);
    let mut ind_warp_file = 0i32;
    let min_sample_per_edge: i32;
    let mut max_sampling: i32 = 0;
    let num_optimal_samp: i32;
    let mut ind_adoc: i32 = 0;
    let (mut if_adoc_mont, mut num_adoc_sect, mut isect_type) = (0i32, 0i32, 0i32);
    let mut isect_num: i32 = 0;
    let mut indent_tmp = 0i32;
    let (mut axis_angle, mut tilt_angle) = (0.0f32, 0.0f32);
    let (mut xvec, mut yvec): (f32, f32) = (0.0, 0.0);
    let (mut xrot, mut yrot, mut xback, mut yback): (f32, f32, f32, f32) = (0.0, 0.0, 0.0, 0.0);
    let (mut cos_rot, mut sin_rot): (f32, f32) = (0.0, 0.0);
    let mut exp_shift_x = [0.0f32; 2];
    let mut exp_shift_y = [0.0f32; 2];
    let mut n_expected = [0i32; 2];
    let mut nbin_tmp: i32 = 0;
    let (mut num_pad_tmp, mut num_box_tmp) = (0i32, 0i32);
    let (mut nx_pad, mut ny_pad, mut max_long_tmp) = (0i32, 0i32, 0i32);
    let (mut wall_start, mut fast_cum, mut slow_cum): (f64, f64, f64) = (0.0, 0.0, 0.0);
    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0i32, 0i32);
    let mut ixy: i32 = 0;
    let mut iedge: i32 = 0;
    let mut jedge: i32 = 0;
    let mut ind_x: i32 = 0;
    let mut ind_y: i32 = 0;

    // gfortran runtime: a `read(5,*)` with no `END=`/`ERR=` that fails
    // reports and stops with status 2.
    let read5 = |items: &mut [ListItem]| {
        let _ = ImodFile::Stdout.flush();
        if let Err(err) = list_read(&mut std::io::stdin().lock(), items) {
            read_runtime_error(err);
        }
    };
    // `read(5, '(a)') name`, likewise.
    let read5_a = || -> String {
        let _ = ImodFile::Stdout.flush();
        let mut line = String::new();
        if matches!(std::io::stdin().lock().read_line(&mut line), Ok(0) | Err(_)) {
            read_runtime_error(ListReadError::End);
        }
        line.trim_end_matches(['\r', '\n'])
            .trim_end_matches(' ')
            .to_owned()
    };
    // The Fortran wrapper `pipgetstring` (`pip_fwrap.c`): the variable is left
    // untouched unless the option is found; trailing blanks are the
    // `character*320` padding, trimmed wherever the source uses `trim`.
    // `c2fString` there copies at most the variable's declared length
    // (blendmont.f90:34-35,80,93: every name read is `character*320`); a longer entry fails with `In PipGetString, string is too
    // long for character variable`, which exits under the exit prefix.
    let get_string = |option: &[u8], length: usize, string: &mut String| -> i32 {
        let mut record = vec![b' '; length];
        let err = crate::imod::libcfshr::pip_fwrap::pipgetstring_(option, &mut record);
        if err == 0 {
            *string = crate::imod::libcfshr::b3dutil::fortran_string(&record);
        }
        err
    };
    let blank = |string: &str| string.bytes().all(|b| b == b' ');
    let pdx = |bv: &BlendVars, ix: i32, iy: i32| -> usize { a2!(bv.map_piece_ext, ix, iy) };

    //
    // initialization of many things
    //
    bv.ix_debug = -12000;
    bv.iy_debug = -12000;
    bv.in_piece[0] = 0;
    bv.iun_edge = [7, 8];
    bv.iun_dens = [17, 18];
    bv.indent = [3, 3]; //minimum indent short & long
    bv.int_grid = [6, 6]; //grid interval short & long
    bv.ibox_siz = [10, 15]; //box size short & long
    if_float = 0;
    h.if_sloppy = 0;
    iopt_abs = 0;
    num_xmissing = 0;
    num_ymissing = 0;
    h.if_old_edge = 0;
    h.shift_each = false;
    h.xc_legacy = false;
    h.from_edge = false;
    h.xc_read_in = false;
    ecd_for_expected = String::new();
    expected_from_mdoc = false;
    h.use_expected = false;
    bv.do_mag_grad = false;
    bv.undistort = false;
    bv.focus_adjusted = false;
    binning_of_input = 1.;
    bv.interp_order = 2;
    h.test_mode = false;
    undistort_only = false;
    bv.limit_data = false;
    adjust_origin = false;
    no_fft_sizes = false;
    y_chunks = false;
    same_edge_shifts = false;
    parallel_hdf = false;
    i_binning = 1;
    bv.num_angles = 0;
    bv.num_use_edge = 0;
    bv.iz_use_def_low = -1;
    bv.iz_use_def_high = -1;
    mode_parallel = 0;
    out_file = String::new();
    bv.num_xcorr_peaks = 1;
    h.num_skipped_edges = 0;
    if_use_adjusted = 0;
    em_grid_filter = 0.;
    iy_out_offset = 0;
    h.if_edge_func_only = 0;
    edge_name2 = String::new();
    bound_file = String::new();
    bv.iz_unsmoothed_patch = -1;
    bv.iz_smoothed_patch = -1;
    bv.robust_crit = 0.;
    ecd_binning = 1.;
    //
    // Xcorr parameters
    // 11/5/05: increased taper fraction 0.05->0.1 to protect against
    // edge effects with default filter
    //
    xcorr_debug = false;
    bv.if_dump_xy = [-1, -1];
    bv.ifill_treatment = 1;
    bv.aspect_max = 2.0; //maximum aspect ratio of block
    bv.pad_frac = 0.45;
    bv.extra_width = 0.;
    bv.radius1 = 0.;
    bv.radius2 = 0.35;
    bv.sigma1 = 0.05;
    bv.sigma2 = 0.05;
    max_xcorr_binning = 3;
    nxy_xcorr_target = 1024;
    very_sloppy = false;
    bv.i_den_sample = 0;
    bv.i_dens_from_edges = 0;
    other_grad_file = String::new();
    sum_for_grad = false;
    bv.mult_by_flatfield = 0;
    min_sample_per_edge = 10;
    num_optimal_samp = 2000;
    h.one_edge_just_avg_crit = 2.5;
    bv.lm_field = 1;
    bv.max_fields = 1;
    use_fill = false;
    print_and_exit = false;
    bv.mem_lim = 256;
    i = 0;
    montxcorrgetmaxes(&mut lim_xcorr_peaks, &mut i);
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "blendmont",
        "ERROR: BLENDMONT - ",
        true,
        0,
        0,
        0,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    if pip_get_in_out_file(
        "ImageInputFile",
        num_non_opt_arg + 1,
        "Input image file",
        &mut image_in_file,
        320,
    ) != 0
    {
        exit_error("No input image file specified");
    }
    if pip_input {
        ierr = pip_get_logical("PrintXYSizeAndExit", &mut print_and_exit);
    }
    if print_and_exit {
        iiu_alt_print(0);
    }

    imopen(1, image_in_file.trim_end_matches(' '), "ro");
    unsafe {
        irdhdr(
            1,
            bv.nxyz_in.as_mut_ptr(),
            mxyz_in.as_mut_ptr(),
            &mut mode_in,
            &mut dmin,
            &mut dmax,
            &mut bv.dmean,
        );
    }
    delta = iiu_ret_delta(1);
    if pip_input {
        ierr = pip_get_integer(b"EdgeFunctionsOnly", &mut h.if_edge_func_only);
    }
    h.if_edge_func_only = 0.max(3.min(h.if_edge_func_only));
    if h.if_edge_func_only == 1 || h.if_edge_func_only == 2 {
        h.ixy_func_start = h.if_edge_func_only;
        h.ixy_func_end = h.if_edge_func_only;
    } else {
        h.ixy_func_start = 1;
        h.ixy_func_end = 2;
    }
    //
    if pip_get_in_out_file(
        "ImageOutputFile",
        num_non_opt_arg + 1,
        "Output image file",
        &mut out_file,
        320,
    ) != 0
        && h.if_edge_func_only == 0
    {
        exit_error("No output image file specified");
    }
    //
    mode_out = mode_in;
    if pip_input {
        if pip_get_integer(b"ModeToOutput", &mut mode_out) == 0 {
            mode_out = set_float_output_for_entered_mode(mode_out);
        }
    } else {
        let _ = write!(
            out,
            " Mode for output file [/ for{}]: ",
            i_edit(mode_out, 2)
        );
        read5(&mut [ListItem::Integer(&mut mode_out)]);
        mode_out = set_float_output_for_entered_mode(mode_out);
    }
    if mode_out != 0 && mode_out != 1 && mode_out != 2 && mode_out != 6 && mode_out != 12 {
        exit_error("Bad mode value");
    }
    //
    // set up output range, and default input range and minimum.  If real
    // input, use actual input range and min; otherwise use theoretical
    // range for the input mode.  This will preserve data values if no float
    //
    out_min = 0.;
    out_max = 2i32.wrapping_pow(MODE_POWER[mode_out as usize] as u32) as f32 - 1.;
    if mode_out == 1 {
        out_min = -out_max;
    }
    if mode_in == 2 {
        dflt_in_max = dmax;
        dflt_in_min = dmin;
        if mode_out == 2 {
            out_min = dmin;
            out_max = dmax;
        }
    } else {
        // `modePower(modeIn)`: an input mode outside 0:15 reads past the
        // array in the source; the translation uses 0 there.
        let power = MODE_POWER.get(mode_in as usize).copied().unwrap_or(0);
        dflt_in_max = 2i32.wrapping_pow(power as u32) as f32 - 1.;
        dflt_in_min = if mode_in == 1 { -dflt_in_max } else { 0. };
    }
    //
    file_name = String::new();
    if pip_input {
        ierr = pip_get_boolean(b"FloatToRange", &mut if_float);
        ierr = get_string(b"TransformFile", 320, &mut file_name);
        ierr = pip_get_logical("JustUndistort", &mut undistort_only);
    } else {
        let _ = write!(out, " 1 to float each section to maximum range, 0 not to: ");
        read5(&mut [ListItem::Integer(&mut if_float)]);
        //
        let _ = write!(out, " File of g transforms to apply (Return if none): ");
        file_name = read5_a();
    }
    //
    // Preserve legacy behavior of floating to positive range for mode 1
    if if_float != 0 && mode_in == 1 && mode_out == 1 {
        out_min = 0.;
        dflt_in_min = 0.;
    }

    bv.lim_sect = 100000;
    ix_pc_temp = vec![0; LIM_INIT_USIZE];
    iy_pc_temp = vec![0; LIM_INIT_USIZE];
    iz_pc_temp = vec![0; LIM_INIT_USIZE];
    neg_tmp = vec![0; LIM_INIT_USIZE];
    multi_temp = vec![false; bv.lim_sect as usize];
    //
    read_list(
        &mut ix_pc_temp,
        &mut iy_pc_temp,
        &mut iz_pc_temp,
        &mut neg_tmp,
        &mut multi_temp,
        &mut bv.npc_list,
        &mut min_zpiece,
        &mut max_piece_z,
        &mut h.any_neg,
        pip_input,
    );
    bv.lim_npc = bv.npc_list + 10;
    {
        let lnpc = bv.lim_npc as usize;
        bv.ix_pc_list = vec![0; lnpc];
        bv.iy_pc_list = vec![0; lnpc];
        bv.iz_pc_list = vec![0; lnpc];
        bv.neg_list = vec![0; lnpc];
        bv.lim_data_ind = vec![0; lnpc];
        bv.iedge_lower = vec![0; lnpc * 2];
        bv.iedge_lower_ext = [lnpc, 2];
        bv.iedge_upper = vec![0; lnpc * 2];
        bv.iedge_upper_ext = [lnpc, 2];
        bv.hinv = vec![0.; 6 * lnpc];
        bv.hinv_ext = [2, 3, lnpc];
        bv.mem_index = vec![0; lnpc];
        bv.htmp = vec![0.; 6 * lnpc];
        bv.htmp_ext = [2, 3, lnpc];
        h.hxf = vec![0.; 6 * lnpc];
        bv.ind_var = vec![0; lnpc];
    }
    let npc = bv.npc_list as usize;
    bv.ix_pc_list[..npc].copy_from_slice(&ix_pc_temp[..npc]);
    bv.iy_pc_list[..npc].copy_from_slice(&iy_pc_temp[..npc]);
    bv.iz_pc_list[..npc].copy_from_slice(&iz_pc_temp[..npc]);
    bv.neg_list[..npc].copy_from_slice(&neg_tmp[..npc]);
    //
    num_sect = max_piece_z + 1 - min_zpiece;
    bv.lim_sect = num_sect + 1;
    {
        let ls = bv.lim_sect as usize;
        bv.tilt_angles = vec![0.; ls];
        bv.dmag_per_um = vec![0.; ls];
        bv.rot_per_um = vec![0.; ls];
        list_z = vec![0; ls];
        iz_want = vec![0; ls];
        iz_all_want = vec![0; ls];
        gxf = vec![0.; 6 * ls];
        multi_neg = vec![false; ls];
        bv.iz_mem_list = vec![0; bv.mem_lim as usize];
        bv.last_used = vec![0; bv.mem_lim as usize];
    }
    multi_neg[..num_sect as usize].copy_from_slice(&multi_temp[..num_sect as usize]);
    //
    {
        let mut n = 0usize;
        fill_listz(&bv.iz_pc_list[..npc], &mut list_z, &mut n);
        num_list_z = n as i32;
    }
    //
    bv.do_gxforms = false;
    do_warp = false;
    if !blank(&file_name) && !undistort_only {
        bv.do_gxforms = true;
        //
        // Open as warping file if possible
        let mut warp_pixel = 0.0f32;
        ind_warp_file = read_check_warp_file(
            file_name.trim_end_matches(' '),
            0,
            1,
            &mut iwarp_nx,
            &mut iwarp_ny,
            &mut num_gxforms,
            &mut ix,
            &mut warp_pixel,
            &mut iy,
            &mut h.edge_name,
        );
        if ind_warp_file < -1 {
            exit_error(&h.edge_name);
        }
        do_warp = ind_warp_file >= 0;
        if do_warp {
            if num_gxforms > bv.lim_sect {
                exit_error("Too many sections in warping file for transform array");
            }
            warp_scale = warp_pixel / delta[0];
            for i in 1..=num_gxforms {
                if get_linear_transform(i - 1, &mut gxf[h3!(1, 1, i)..], 2) != 0 {
                    exit_error("Getting linear transform from warp file");
                }
                gxf[h3!(1, 3, i)] *= warp_scale;
                gxf[h3!(2, 3, i)] *= warp_scale;
            }
        } else {
            //
            // Get regular transforms
            let unit3 = dopen(3, file_name.trim_end_matches(' '), "ro", "f");
            let mut list: Vec<[f32; 6]> = Vec::new();
            ierr = xfrdall2(&mut BufReader::new(unit3), &mut list, bv.lim_sect);
            num_gxforms = list.len() as i32;
            for (k, xf) in list.iter().enumerate() {
                gxf[6 * k..6 * k + 6].copy_from_slice(xf);
            }
            // `close(3)` is the drop of the reader.
            //
            // It is OK if ierr=-1 : more transforms than sections
            if ierr > 0 {
                exit_error("Reading transforms");
            }
        }
    }

    if bv.do_gxforms {
        if num_list_z > num_gxforms {
            exit_error("More sections than G transforms");
        }

        skip_xforms = false;
        if num_list_z < num_sect {
            if num_gxforms == num_list_z {
                let _ = writeln!(
                    out,
                    " Looks like there are transforms only for the sections that exist in file"
                );
            } else if num_gxforms == num_sect {
                let _ = writeln!(
                    out,
                    " There seem to be transforms for each Z value, including ones missing from file"
                );
                skip_xforms = true;
            } else {
                exit_error(
                    "Cannot tell how transforms match up to sections, because of missing sections",
                );
            }
        }
    }
    //
    // now check lists and get basic properties of overlap etc
    //
    checklist(
        &bv.ix_pc_list[..npc],
        1,
        bv.nxyz_in[0],
        &mut bv.min_xpiece,
        &mut bv.nx_pieces,
        &mut bv.n_overlap[0],
    );
    checklist(
        &bv.iy_pc_list[..npc],
        1,
        bv.nxyz_in[1],
        &mut bv.min_ypiece,
        &mut bv.ny_pieces,
        &mut bv.n_overlap[1],
    );
    if bv.nx_pieces <= 0 || bv.ny_pieces <= 0 {
        exit_error("Checklist reported a problem with the piece list in one direction");
    }
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    //
    nx_total_pix = bv.nx_pieces * (nxin - bv.n_overlap[0]) + bv.n_overlap[0];
    ny_total_pix = bv.ny_pieces * (nyin - bv.n_overlap[1]) + bv.n_overlap[1];
    // `115 format(i7,' total ',a1,' pixels in',i4,' pieces of', i6,
    // ' pixels, with overlap of',i5)`
    let fmt115 = |ntot: i32, axis: &str, npcs: i32, nframe: i32, nover: i32| -> String {
        format!(
            "{} total {} pixels in{} pieces of{} pixels, with overlap of{}\n",
            i_edit(ntot, 7),
            axis,
            i_edit(npcs, 4),
            i_edit(nframe, 6),
            i_edit(nover, 5)
        )
    };
    if !print_and_exit {
        if pip_input {
            let _ = writeln!(out, " Input file:");
        }
        let _ = out
            .write_all(fmt115(nx_total_pix, "X", bv.nx_pieces, nxin, bv.n_overlap[0]).as_bytes());
        let _ = out
            .write_all(fmt115(ny_total_pix, "Y", bv.ny_pieces, nyin, bv.n_overlap[1]).as_bytes());
    }
    //
    bv.ix_dim_den_buf = bv.ny_pieces * (bv.nx_pieces - 1);
    max_sec_edges = bv.ix_dim_den_buf + bv.nx_pieces * (bv.ny_pieces - 1);
    bv.lim_edge = bv.lim_sect * 1.max(bv.nx_pieces * bv.ny_pieces);
    {
        let le = bv.lim_edge as usize;
        let mse = max_sec_edges as usize;
        h.dx_grid_mean = vec![0.; 2 * le];
        h.dy_grid_mean = vec![0.; 2 * le];
        h.edge_displace_x = vec![0.; 2 * le];
        h.edge_displace_y = vec![0.; 2 * le];
        h.edge_done = vec![false; 2 * le];
        bv.ipiece_lower = vec![0; 2 * le];
        bv.ipiece_lower_ext = [le, 2];
        bv.ipiece_upper = vec![0; 2 * le];
        bv.ipiece_upper_ext = [le, 2];
        bv.ibuf_edge = vec![0; 2 * le];
        bv.ibuf_edge_ext = [le, 2];
        bv.if_skip_edge = vec![0; 2 * le];
        bv.if_skip_edge_ext = [le, 2];
        bv.i_edge_zbase = vec![0; 2 * (num_sect as usize + 1)];
        bv.i_edge_zbase_ext = [num_sect as usize + 1, 2];
        bv.trimmed_max_sds = vec![0.; mse];
        bv.max_sd_to_edge_num = vec![0; mse];
        bv.max_sd_to_ixy_of_edge = vec![0; mse];
        h.max_sd_temp = vec![0.; mse];
        bv.altern_disps = vec![0.; le * 4 * 2];
        bv.iedge_alt_fixed = vec![0; le];
        bv.iedge_low_weight = vec![0; le];
    }
    //
    // find out if global multi-neg specifications are needed or desired
    // But first deal with correlation control parameters
    // Here are the defaults for VerySloppy
    //
    if pip_input {
        ierr = pip_get_logical("VerySloppyMontage", &mut very_sloppy);
        if very_sloppy {
            h.if_sloppy = 1;
            bv.aspect_max = 5.;
            bv.radius1 = -0.01;
            bv.extra_width = 0.25;
            bv.num_xcorr_peaks = 16;
        } else {
            ierr = pip_get_boolean(b"SloppyMontage", &mut h.if_sloppy);
        }

        if pip_get_float(b"EMGridMapFilter", &mut em_grid_filter) == 0 {
            if delta[0] == 1. {
                exit_error(
                    "You cannot use EMGridMapFilter; the image file does not have a pixel size defined",
                );
            }
            bv.num_xcorr_peaks = 50;
            if very_sloppy {
                bv.num_xcorr_peaks = 100;
            }
            bv.radius1 = 0.;
            bv.sigma1 = 0.01;
            h.use_expected = true;
        }
        h.shift_each = h.if_sloppy != 0;
        if pip_get_two_integers(
            b"FramesPerNegativeXandY",
            &mut num_xframes_per_neg,
            &mut num_yframes_per_neg,
        ) == 0
        {
            iopt_abs = 1;
        }
        if !h.shift_each {
            ierr = pip_get_logical("ShiftPieces", &mut h.shift_each);
        }
        ierr = pip_get_logical("ShiftFromEdges", &mut h.from_edge);
        ierr = pip_get_logical("ShiftFromXcorrs", &mut h.xc_legacy);
        ierr = pip_get_logical("ReadInXcorrs", &mut h.xc_read_in);
        if pip_get_float(b"WeightForExpectedShifts", &mut err_lim) == 0 {
            h.use_expected = err_lim > 0.;
            if err_lim > 2. {
                mont_xc_set_dist_weight_half_fall(err_lim);
            }
        }

        if !h.xc_read_in && h.shift_each {
            ierr = pip_get_logical("MdocForExpectedShifts", &mut expected_from_mdoc);
            ierr = get_string(b"ExpectedShiftsFromEcd", 320, &mut ecd_for_expected);
            if ierr == 0 && expected_from_mdoc {
                exit_error("You cannot enter both ExpectedShiftsFromEcd and MdocForExpectedShifts");
            }
            if ierr == 0 || expected_from_mdoc {
                h.use_expected = true;
            }
        }
        if h.use_expected && em_grid_filter == 0. {
            bv.num_xcorr_peaks = 16;
        }
        ierr = pip_get_float(b"AspectRatio", &mut bv.aspect_max);
        ierr = pip_get_integer(b"NumberOfXcorrPeaks", &mut bv.num_xcorr_peaks);
        bv.num_xcorr_peaks = 1.max(lim_xcorr_peaks.min(bv.num_xcorr_peaks));
        ierr = pip_get_float(b"PadFraction", &mut bv.pad_frac);
        ierr = pip_get_float(b"ExtraXcorrWidth", &mut bv.extra_width);
        ierr = pip_get_float(b"FilterSigma1", &mut bv.sigma1);
        ierr = pip_get_float(b"FilterRadius1", &mut bv.radius1);
        ierr = pip_get_float(b"FilterRadius2", &mut bv.radius2);
        ix = pip_get_float(b"FilterSigma2", &mut bv.sigma2);
        if em_grid_filter > 0. && (ierr == 0 || ix == 0) {
            exit_error("You cannot enter FilterRadius2 or FilterSigma2 with EMGridMapFilter");
        }
        ierr = pip_get_integer(b"NonzeroSkippedEdgeUse", &mut if_use_adjusted);
        ierr = pip_get_float(b"RobustFitCriterion", &mut bv.robust_crit);
        ierr = pip_get_logical("TestMode", &mut h.test_mode);
        use_fill = pip_get_float(b"FillValue", &mut fill_val) == 0;
        ierr = pip_get_float(b"BaseIntensityForScaling", &mut bv.den_zero_base);
        ierr = pip_get_integer(b"FixIntensityFromEdges", &mut bv.i_dens_from_edges);
        //if (iDensFromEdges > 0) &
        //ierr = PipGetInteger('EdgeIntensitySampling', iDenSample)
        ierr = pip_get_logical("SumPiecesForGradient", &mut sum_for_grad);
        ierr = get_string(b"OtherSumGradientFile", 320, &mut other_grad_file);
        if ierr == 0 && sum_for_grad {
            exit_error("You cannot enter both -SumPiecesForGradient and -OtherSumGradientFile");
        }
        if sum_for_grad && h.if_edge_func_only > 0 && h.if_edge_func_only < 3 {
            exit_error(
                "You cannot use -SumPiecesForGradient when making edge functions on only one axis",
            );
        }

        // Handle a flatfield file: check, open, allocate, read image, set flag
        if get_string(b"FlatfieldFile", 320, &mut h.edge_name) == 0 {
            if sum_for_grad || !blank(&other_grad_file) {
                exit_error(
                    "You cannot use a flatfield file with -SumPiecesForGradient or -OtherSumGradientFile",
                );
            }
            imopen(3, h.edge_name.trim_end_matches(' '), "ro");
            let (mut ff_mode, mut ff_min, mut ff_max) = (0i32, 0.0f32, 0.0f32);
            unsafe {
                irdhdr(
                    3,
                    nxyz_tmp.as_mut_ptr(),
                    mxyz_tmp.as_mut_ptr(),
                    &mut ff_mode,
                    &mut ff_min,
                    &mut ff_max,
                    &mut w1,
                );
            }
            ierr = ff_mode;
            x1 = ff_min;
            y1 = ff_max;
            if nxyz_tmp[0] != nxin || nxyz_tmp[1] != nyin {
                exit_error("Flatfield image must be the same size as the input pieces");
            }
            if ierr != 2 || (w1 - 1.).abs() > 1.5 {
                exit_error("Flatfield image must be floating point and have a mean near 1");
            }
            bv.flatfield = vec![0.; (nxin * nyin) as usize];
            memory_error(0, "array for flatfield image");
            bv.mult_by_flatfield = 1;
            // `irdsec` with no error branch: `iiuReadSection` exits on error
            // itself through the unit layer's exit-on-error setting.
            let _ = unsafe { irdsec(3, &mut bv.flatfield) };
            unsafe { iiu_close(3) };
        }

        ierr = pip_get_two_integers(
            b"GoodEdgeLowAndHighZ",
            &mut bv.iz_use_def_low,
            &mut bv.iz_use_def_high,
        );
        ierr = pip_number_of_entries(b"OneGoodEdgeLimits", &mut bv.num_use_edge);
        for i in 1..=bv.num_use_edge {
            let mut num_to_get = 5;
            ierr =
                pip_get_integer_array(b"OneGoodEdgeLimits", &mut ix_pc_lower, &mut num_to_get, 15);
            let iu = (i - 1) as usize;
            bv.ix_frm_use_edge[iu] = ix_pc_lower[0];
            bv.iy_frm_use_edge[iu] = ix_pc_lower[1];
            bv.ixy_use_edge[iu] = ix_pc_lower[2];
            bv.iz_low_use[iu] = ix_pc_lower[3];
            bv.iz_high_use[iu] = ix_pc_lower[4];
        }
        ierr = pip_get_logical("SameEdgeShifts", &mut same_edge_shifts);
        if (h.any_neg || iopt_abs != 0) && (h.shift_each || h.xc_read_in) && !undistort_only {
            exit_error("you cannot use ShiftPieces or ReadInXcorrs with multiple negatives");
        }
        if h.from_edge && h.xc_legacy && !undistort_only {
            exit_error("You cannot use both ShiftFromEdges and ShiftFromXcorrs");
        }
        if (bv.iz_use_def_low >= 0 || bv.num_use_edge > 0) && h.shift_each {
            if !same_edge_shifts {
                exit_error("You cannot use good edge limits when shifting pieces");
            }
            if h.from_edge {
                exit_error("You cannot use ShiftFromEdges with good edge limits");
            }
            h.xc_legacy = true;
        }
    } else if h.any_neg {
        let _ = writeln!(out, " There are multi-negative specifications in list file");
        let _ = write!(
            out,
            " 1 to do initial cross-correlations in overlap zones, 0 not to: "
        );
        read5(&mut [ListItem::Integer(&mut h.if_sloppy)]);
    } else {
        let _ = writeln!(
            out,
            // `print *` of two adjacent strings puts no blank between them.
            " Enter the negative of one of the following options to do initialcross-correlations in overlap zones in combination with the particular option"
        );
        let _ = write!(
            out,
            " Enter 1 to specify division into negatives to apply to all sections,\n      2 to use edge functions to find a shift for each frame to align frames\n      3 to use cross-correlation only to find a shift for each frame\n      4 to use only cross-correlation displacements read from a file\n      5 to use best shifts from edge functions and correlations\n      6 to use best shifts from edge functions and displacements read from a file: "
        );
        iopt_neg = 0;
        read5(&mut [ListItem::Integer(&mut iopt_neg)]);
        h.if_sloppy = 0;
        if iopt_neg < 0 {
            h.if_sloppy = 1;
        }
        iopt_abs = iopt_neg.abs();
        h.shift_each = iopt_abs >= 2;
        h.from_edge = iopt_abs == 2;
        h.xc_legacy = iopt_abs == 3 || iopt_abs == 4;
        h.xc_read_in = iopt_abs == 4 || iopt_abs == 6;
    }
    //
    if iopt_abs == 1 {
        if pip_input {
            // Fixed in translation (`BUGS.md`): `blendmont.f90:638` reads
            // `FramesPerNegativeXandY` a second time here, so
            // `-MissingFromFirstNegativeXandY` is never read and the missing
            // counts equal the frames per negative.  The option the counts
            // belong to is read (they stay 0 when it is not entered).
            ierr = pip_get_two_integers(
                b"MissingFromFirstNegativeXandY",
                &mut num_xmissing,
                &mut num_ymissing,
            );
        } else {
            let _ = write!(
                out,
                " # of frames per negative in X; # missing from left-most negative: "
            );
            read5(&mut [
                ListItem::Integer(&mut num_xframes_per_neg),
                ListItem::Integer(&mut num_xmissing),
            ]);
            let _ = write!(
                out,
                " # of frames per negative in Y; # missing from bottom-most negative: "
            );
            read5(&mut [
                ListItem::Integer(&mut num_yframes_per_neg),
                ListItem::Integer(&mut num_ymissing),
            ]);
        }
        num_xneg = (bv.nx_pieces + (num_xframes_per_neg - 1) + num_xmissing) / num_xframes_per_neg;
        //
        // derive frame number of each piece and assign negative #
        //
        for ipc in 1..=bv.npc_list {
            let iu = (ipc - 1) as usize;
            ix_frame = (bv.ix_pc_list[iu] - bv.min_xpiece) / (nxin - bv.n_overlap[0]);
            iy_frame = (bv.iy_pc_list[iu] - bv.min_ypiece) / (nyin - bv.n_overlap[1]);
            ix_neg = (ix_frame + num_xmissing) / num_xframes_per_neg;
            iy_neg = (iy_frame + num_ymissing) / num_yframes_per_neg;
            bv.neg_list[iu] = 1 + ix_neg + iy_neg * num_xneg;
        }
        //
        // now deduce true multi-neg character of each section
        //
        for iz in min_zpiece..=max_piece_z {
            ineg = iz + 1 - min_zpiece;
            multi_neg[(ineg - 1) as usize] = false;
            list_first = -100000;
            for ipc in 1..=bv.npc_list {
                let iu = (ipc - 1) as usize;
                if bv.iz_pc_list[iu] == iz {
                    if list_first == -100000 {
                        list_first = bv.neg_list[iu];
                    }
                    multi_neg[(ineg - 1) as usize] =
                        multi_neg[(ineg - 1) as usize] || (bv.neg_list[iu] != list_first);
                }
            }
            h.any_neg = h.any_neg || multi_neg[(ineg - 1) as usize];
        }
        //
    }
    //
    pl_out_file = String::new();
    if pip_input {
        ierr = get_string(b"PieceListOutput", 320, &mut pl_out_file);
    } else {
        let _ = write!(out, " Name of new piece list file (Return for none): ");
        pl_out_file = read5_a();
    }
    outputpl = !blank(&pl_out_file) && h.if_edge_func_only == 0;
    ali_coord_file = String::new();
    ierr = get_string(b"AlignedPieceCoordFile", 320, &mut ali_coord_file);
    //
    // find out center of transforms
    //
    bv.gx_cen = (bv.min_xpiece + nx_total_pix / 2) as f32;
    bv.gy_cen = (bv.min_ypiece + ny_total_pix / 2) as f32;
    if pip_input {
        ierr = pip_get_two_floats(b"TransformCenterXandY", &mut bv.gx_cen, &mut bv.gy_cen);
    } else if bv.do_gxforms {
        let _ = write!(
            out,
            " Enter true center coordinates of the transforms,\n    or / for the default{},{}, the center of the image area: ",
            f_edit(bv.gx_cen, 6, 1),
            f_edit(bv.gy_cen, 6, 1)
        );
        read5(&mut [
            ListItem::Real(&mut bv.gx_cen),
            ListItem::Real(&mut bv.gy_cen),
        ]);
    }
    //
    // Add 0.5 to get the center of rotation to be around the center of image
    bv.gx_cen += 0.5;
    bv.gy_cen += 0.5;
    //
    // get list of sections desired, set up default as all sections
    //
    for i in 1..=num_list_z {
        iz_pc_temp[(i - 1) as usize] = list_z[(i - 1) as usize];
    }
    num_zwant = num_list_z;
    if pip_input {
        if get_string(b"SectionsToDo", 320, &mut file_name) == 0 {
            let _ = parselist(
                file_name.trim_end_matches(' '),
                &mut iz_pc_temp,
                &mut num_zwant,
            );
        }
    } else {
        let _ = writeln!(
            out,
            " Enter list of sections to be included in output file (ranges ok)   or / to include all sections"
        );
        let _ = out.flush();
        let _ = rdlist(
            &mut std::io::stdin().lock(),
            &mut iz_pc_temp,
            &mut num_zwant,
        );
    }
    //
    // copy/pack list, eliminating non-existent sections and duplicates
    ix_out = 0;
    for ix in 1..=num_zwant {
        let want = iz_pc_temp[(ix - 1) as usize];
        ierr = 0;
        for i in 1..=num_list_z {
            if list_z[(i - 1) as usize] == want {
                ierr = 1;
            }
        }
        for i in 1..=ix_out {
            if iz_want[(i - 1) as usize] == want {
                ierr = 0;
            }
        }
        if ierr == 1 {
            ix_out += 1;
            iz_want[(ix_out - 1) as usize] = want;
        }
    }
    num_zwant = ix_out;
    if num_zwant == 0 {
        exit_error("Section output list does not include any actual sections");
    }
    nz_all_want = num_zwant;
    //
    // Set flag for whether to write out edge displacements
    // write edge correlations if they are not being read in and if they are
    // computed in their entirety: but this needs to be modified later
    xc_write_out = !h.test_mode
        && !h.xc_read_in
        && ((h.if_sloppy == 1 && h.shift_each)
            || (h.if_sloppy == 0 && h.shift_each && !h.from_edge));
    //
    drop(ix_pc_temp);
    drop(iy_pc_temp);
    drop(neg_tmp);
    drop(multi_temp);
    // `izPcTemp` is deallocated here too; nothing reads it again.
    drop(iz_pc_temp);
    //
    if undistort_only {
        //
        // If undistorting, copy sizes etc.
        //
        new_xframe = nxin;
        new_xoverlap = bv.n_overlap[0];
        new_xpieces = bv.nx_pieces;
        new_min_xpiece = bv.min_xpiece;
        new_yframe = nyin;
        new_yoverlap = bv.n_overlap[1];
        new_ypieces = bv.ny_pieces;
        new_min_ypiece = bv.min_ypiece;
        action_str = "undistorted only";
    } else {
        //
        // Set up output size
        //
        min_xoverlap = 2;
        min_yoverlap = 2;
        new_xframe = 100000000;
        new_yframe = 100000000;
        num_trials = 0;
        loop {
            // `32` continue: the interactive revision loop.
            min_xwant = bv.min_xpiece;
            min_ywant = bv.min_ypiece;
            max_xwant = min_xwant + nx_total_pix - 1;
            max_ywant = min_ywant + ny_total_pix - 1;
            if pip_input {
                ierr = pip_get_logical("XcorrDebug", &mut xcorr_debug);
                ierr = pip_get_logical("AdjustOrigin", &mut adjust_origin);
                ierr = pip_get_logical("ExcludeFillFromEdges", &mut bv.limit_data);
                ierr = pip_get_logical("NoResizeForFFT", &mut no_fft_sizes);
                ierr = pip_get_two_integers(b"StartingAndEndingX", &mut min_xwant, &mut max_xwant);
                ierr = pip_get_two_integers(b"StartingAndEndingY", &mut min_ywant, &mut max_ywant);
                ierr =
                    pip_get_two_integers(b"MaximumNewSizeXandY", &mut new_xframe, &mut new_yframe);
                ierr = pip_get_two_integers(
                    b"MinimumOverlapXandY",
                    &mut min_xoverlap,
                    &mut min_yoverlap,
                );
                ierr = pip_get_integer(b"BinByFactor", &mut i_binning);
                if i_binning < 1 {
                    exit_error("Binning must be positive");
                }
                if i_binning > MAX_BIN {
                    exit_error("Binning is too large");
                }
                if get_string(b"UnsmoothedPatchFile", 320, &mut h.edge_name) == 0 {
                    units.unit10 = Some(BufWriter::new(dopen(
                        10,
                        h.edge_name.trim_end_matches(' '),
                        "new",
                        "f",
                    )));
                    bv.iz_unsmoothed_patch = 0;
                }
                if get_string(b"SmoothedPatchFile", 320, &mut h.edge_name) == 0 {
                    units.unit11 = Some(BufWriter::new(dopen(
                        11,
                        h.edge_name.trim_end_matches(' '),
                        "new",
                        "f",
                    )));
                    bv.iz_smoothed_patch = 0;
                }
            } else {
                let _ = write!(
                    out,
                    " Enter Min X, Max X, Min Y, and Max Y coordinates of desired output section,\n    or / for whole input section [={}{}{}{}]: ",
                    i_edit(min_xwant, 6),
                    i_edit(max_xwant, 6),
                    i_edit(min_ywant, 6),
                    i_edit(max_ywant, 6)
                );
                read5(&mut [
                    ListItem::Integer(&mut min_xwant),
                    ListItem::Integer(&mut max_xwant),
                    ListItem::Integer(&mut min_ywant),
                    ListItem::Integer(&mut max_ywant),
                ]);
                let _ = write!(out, " Maximum new X and Y frame size, minimum overlap: ");
                read5(&mut [
                    ListItem::Integer(&mut new_xframe),
                    ListItem::Integer(&mut new_yframe),
                    ListItem::Integer(&mut min_xoverlap),
                    ListItem::Integer(&mut min_yoverlap),
                ]);
            }
            if num_trials <= 1 {
                //on first 2 trials, enforce min
                min_xoverlap = 2.max(min_xoverlap); //overlap of 2 so things look
                min_yoverlap = 2.max(min_yoverlap); //nice in wimp.  After that, let
            } //the user have it.
            num_trials += 1;
            //
            // If no resizing desired, take exactly what is requested in one frame
            if no_fft_sizes
                && max_xwant + 1 - min_xwant <= new_xframe
                && max_ywant + 1 - min_ywant <= new_yframe
            {
                nx_total_want = max_xwant + 1 - min_xwant;
                new_xframe = nx_total_want;
                new_xtotal_pix = nx_total_want;
                new_xpieces = 1;
                new_xoverlap = 2;
                ny_total_want = max_ywant + 1 - min_ywant;
                new_yframe = ny_total_want;
                new_ytotal_pix = ny_total_want;
                new_ypieces = 1;
                new_yoverlap = 2;
            } else {
                nx_total_want = 2 * ((max_xwant + 2 - min_xwant) / 2);
                ny_total_want = 2 * ((max_ywant + 2 - min_ywant) / 2);
                set_overlap(
                    nx_total_want,
                    min_xoverlap,
                    no_fft_sizes,
                    &mut new_xframe,
                    2,
                    &mut new_xpieces,
                    &mut new_xoverlap,
                    &mut new_xtotal_pix,
                );
                set_overlap(
                    ny_total_want,
                    min_yoverlap,
                    no_fft_sizes,
                    &mut new_yframe,
                    2,
                    &mut new_ypieces,
                    &mut new_yoverlap,
                    &mut new_ytotal_pix,
                );
            }
            //
            if !outputpl && h.if_edge_func_only == 0 && (new_xpieces > 1 || new_ypieces > 1) {
                exit_error(
                    "You must specify an output piece list file to have more than one output frame",
                );
            }
            if i_binning > 1 && (new_xpieces > 1 || new_ypieces > 1) {
                exit_error("With binning, output must be into a single frame");
            }
            if print_and_exit {
                let _ = writeln!(
                    out,
                    " Output size: {}{}",
                    ld_int(new_xtotal_pix),
                    ld_int(new_ytotal_pix)
                );
                exit(0);
            } else {
                if pip_input {
                    let _ = writeln!(out, " Output file:");
                }
                let _ = out.write_all(
                    fmt115(new_xtotal_pix, "X", new_xpieces, new_xframe, new_xoverlap).as_bytes(),
                );
                let _ = out.write_all(
                    fmt115(new_ytotal_pix, "Y", new_ypieces, new_yframe, new_yoverlap).as_bytes(),
                );
            }

            //
            if !pip_input {
                let _ = write!(out, " 1 to revise frame size/overlap: ");
                read5(&mut [ListItem::Integer(&mut if_revise)]);
                if if_revise != 0 {
                    continue;
                }
            }
            break;
        }
        //
        new_min_xpiece = min_xwant - (new_xtotal_pix - nx_total_want) / 2;
        new_min_ypiece = min_ywant - (new_ytotal_pix - ny_total_want) / 2;
        action_str = "blended and recut";
    }
    let _ = writeln!(
        out,
        "Starting coordinates of output in X and Y ={}{}",
        i_edit(new_min_xpiece, 7),
        i_edit(new_min_ypiece, 7)
    );
    //
    bv.nxyz_out[0] = new_xframe;
    bv.nxyz_out[1] = new_yframe;
    dmin_out = 1.0e10;
    dmax_out = -1.0e10;
    grand_sum = 0.;
    (bv.nxyz_bin[0], ix_offset) = get_binned_size(bv.nxyz_out[0], i_binning, 0);
    (bv.nxyz_bin[1], iy_offset) = get_binned_size(bv.nxyz_out[1], i_binning, 0);
    bv.nxyz_bin[2] = 0;
    num_lines_write = bv.nxyz_bin[1];
    bv.hx_cen = nxin as f32 / 2.;
    bv.hy_cen = nyin as f32 / 2.;
    //
    // get edge indexes for pieces and piece indexes for edges
    // first build a map in the array of all pieces present
    //
    let (nxp, nyp) = (bv.nx_pieces, bv.ny_pieces);
    let mapx = |ix: i32, iy: i32, iz: i32| -> usize {
        ((ix - 1) + nxp * ((iy - 1) + nyp * (iz - 1))) as usize
    };
    map_all_piece = vec![0; (nxp * nyp * num_sect) as usize];
    //
    for ipc in 1..=bv.npc_list {
        let iu = (ipc - 1) as usize;
        map_all_piece[mapx(
            1 + (bv.ix_pc_list[iu] - bv.min_xpiece) / (nxin - bv.n_overlap[0]),
            1 + (bv.iy_pc_list[iu] - bv.min_ypiece) / (nyin - bv.n_overlap[1]),
            bv.iz_pc_list[iu] + 1 - min_zpiece,
        )] = ipc;
        for i in 1..=2 {
            bv.iedge_lower[a2!(bv.iedge_lower_ext, ipc, i)] = 0;
            bv.iedge_upper[a2!(bv.iedge_upper_ext, ipc, i)] = 0;
            bv.lim_data_ind[iu] = -1;
        }
    }
    //
    // look at all the edges in turn, add to list if pieces on both sides
    //
    num_along = nyp;
    num_across = nxp;
    bv.i_edge_zbase.fill(-1);
    for ixy in 1..=2 {
        num_ed = 0;
        for iz in 1..=num_sect {
            bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, iz, ixy)] = num_ed;
            for iy in 1..=num_along {
                for ix in 2..=num_across {
                    if ixy == 1 {
                        ipc_lower = map_all_piece[mapx(ix - 1, iy, iz)];
                        ipc_high = map_all_piece[mapx(ix, iy, iz)];
                    } else {
                        ipc_lower = map_all_piece[mapx(iy, ix - 1, iz)];
                        ipc_high = map_all_piece[mapx(iy, ix, iz)];
                    }
                    if ipc_lower != 0 && ipc_high != 0 {
                        num_ed += 1;
                        bv.ipiece_lower[e2!(bv, num_ed, ixy)] = ipc_lower;
                        bv.ipiece_upper[e2!(bv, num_ed, ixy)] = ipc_high;
                        bv.iedge_lower[a2!(bv.iedge_lower_ext, ipc_high, ixy)] = num_ed;
                        bv.iedge_upper[a2!(bv.iedge_upper_ext, ipc_lower, ixy)] = num_ed;
                        h.edge_done[e2!(bv, num_ed, ixy)] = false;
                        bv.if_skip_edge[e2!(bv, num_ed, ixy)] = 0;
                        h.edge_displace_x[e2!(bv, num_ed, ixy)] = 0.;
                        h.edge_displace_y[e2!(bv, num_ed, ixy)] = 0.;
                    }
                }
            }
        }
        bv.nedge[(ixy - 1) as usize] = num_ed;
        bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, num_sect + 1, ixy)] = num_ed;
        num_along = nxp;
        num_across = nyp;
    }
    // The DO variables' exit values, which the source reads again later
    // (`iy` at `blendmont.f90:1399` when ExpectedShiftsFromEcd is entered).
    iy = nxp + 1;
    ix = nyp + 1;
    //
    // Allocate data depending on number of pieces (limvar)
    bv.lim_var = nxp * nyp;
    if !undistort_only {
        let lv = bv.lim_var as usize;
        bv.bb = vec![0.; 2 * lv];
        bv.bb_ext = [2, lv];
        bv.ivar_pc = vec![0; lv];
        bv.iall_var_pc = vec![0; lv];
        bv.ivar_group = vec![0; lv];
        bv.list_check = vec![0; lv];
        bv.fps_work = vec![0.; 20 * lv + lv / 2 + 2];
        bv.dxy_var = vec![0.; lv * 2];
        bv.dxy_var_ext = [lv, 2];
        bv.row_tmp = vec![0.; lv * 2];
        //
        if h.test_mode {
            let n = 2 * lv;
            let le = bv.lim_edge as usize;
            bv.grad_xcen_lo = vec![0.; n];
            bv.grad_xcen_hi = vec![0.; n];
            bv.grad_ycen_lo = vec![0.; n];
            bv.grad_ycen_hi = vec![0.; n];
            bv.over_xcen_lo = vec![0.; n];
            bv.over_xcen_hi = vec![0.; n];
            bv.over_ycen_lo = vec![0.; n];
            bv.over_ycen_hi = vec![0.; n];
            bv.dx_edge = vec![0.; le * 2];
            bv.dx_edge_ext = [le, 2];
            bv.dy_edge = vec![0.; le * 2];
            bv.dy_edge_ext = [le, 2];
            bv.dx_adj = vec![0.; le * 2];
            bv.dx_adj_ext = [le, 2];
            bv.dy_adj = vec![0.; le * 2];
            bv.dy_adj_ext = [le, 2];
        }
    }
    //
    // get edge file name root, parameters if doing a new one
    //
    if pip_input && !undistort_only {
        ierr = pip_get_boolean(b"OldEdgeFunctions", &mut h.if_old_edge);
        if get_string(b"RootNameForEdges", 320, &mut h.root_name) != 0 {
            exit_error("No root name for edge functions specified");
        }
    } else if !undistort_only {
        let _ = write!(out, " 0 for new edge files, 1 to read old ones: ");
        read5(&mut [ListItem::Integer(&mut h.if_old_edge)]);
        let _ = write!(out, " Root file name for edge function files: ");
        h.root_name = read5_a();
    }
    if h.if_old_edge != 0 && h.if_edge_func_only != 0 {
        exit_error("You cannot use old edge functions when just computing edge functions");
    }
    let root = h.root_name.trim_end_matches(' ').to_string();
    //
    // Find out about parallel mode
    if pip_input {
        ierr = pip_get_two_integers(b"ParallelMode", &mut mode_parallel, &mut ix);
        if mode_parallel != 0 {
            y_chunks = ix != 0;
            if outputpl || new_xpieces > 1 || new_ypieces > 1 {
                exit_error("Parallel mode requires output in one piece with no piece list");
            }
            if undistort_only || h.test_mode || xcorr_debug || h.if_edge_func_only != 0 {
                exit_error(
                    "No parallel mode allowedwith UndistortOnly, EdgeFunctionsOnly, TestMode, or XcorrDebug",
                );
            }
            if mode_parallel < 0 && h.if_old_edge == 0 {
                exit_error("Parallel mode allowed only when using old edge functions");
            }
            if mode_parallel < 0 && xc_write_out {
                exit_error("Parallel mode not allowed if writing edge correlation displacements");
            }
            if mode_parallel > 0 {
                //
                // Figure out lists to output
                if mode_parallel <= 1 {
                    exit_error("Target number of chunks should be at least 2");
                }
                //
                // Set up for breaking into Z chunks and modify for chunks in Y
                ix = num_zwant;
                ix_out = 1;
                if y_chunks {
                    ix = new_ytotal_pix / i_binning;
                    ix_out = new_min_ypiece;
                }
                num_chunks = mode_parallel.min(ix);
                num_extra = ix % num_chunks;
                for i in 1..=num_chunks {
                    num_out = ix / num_chunks;
                    if i <= num_extra {
                        num_out += 1;
                    }
                    if y_chunks {
                        iy = ix_out + i_binning * num_out - 1;
                        if i == num_chunks {
                            iy = new_min_ypiece + new_ytotal_pix - 1;
                        }
                        let _ =
                            writeln!(out, "LineSubsetToDo {}{}", i_edit(ix_out, 9), i_edit(iy, 9));
                        ind_xlow = (ix_out - new_min_ypiece) / i_binning;
                        ind_xhigh = (iy - new_min_ypiece) / i_binning;
                        ix_out += i_binning * num_out;
                    } else {
                        let _ = write!(out, "SubsetToDo ");
                        wrlist(&iz_want[(ix_out - 1) as usize..], &num_out);
                        ind_xlow = ix_out - 1;
                        ind_xhigh = ix_out + num_out - 2;
                        ix_out += num_out;
                    }
                    if i == 1 {
                        ind_xlow = -1;
                    }
                    if i == num_chunks {
                        ind_xhigh = -1;
                    }
                    let _ = writeln!(
                        out,
                        "ChunkBoundary  {}{}",
                        i_edit(ind_xlow, 9),
                        i_edit(ind_xhigh, 9)
                    );
                }
                //
                // Output the image size for convenient parsing
                let _ = writeln!(
                    out,
                    "Output image size:{}{}",
                    i_edit(bv.nxyz_bin[0], 9),
                    i_edit(bv.nxyz_bin[1], 9)
                );
                //
                // Or get subset list
            } else if mode_parallel < -1 {
                if y_chunks {
                    if pip_get_two_integers(b"LineSubsetToDo", &mut line_start, &mut line_end) != 0
                    {
                        exit_error("You must enter LineSubsetToDo with this parallel mode");
                    }
                    if line_start < new_min_ypiece || line_end > new_min_ypiece + new_ytotal_pix - 1
                    {
                        exit_error("The starting or ending line of the subset is out of range");
                    }
                    if mode_parallel < -2 {
                        exit_error(
                            "Only direct writingto a single file is allowed with subsets of lines",
                        );
                    }
                    if (line_start - new_min_ypiece) % i_binning != 0
                        || ((line_end + 1 - line_start) % i_binning != 0
                            && line_end != new_min_ypiece + new_ytotal_pix - 1)
                    {
                        exit_error(
                            "Starting line and number of lines must be a multiple of the binning",
                        );
                    }
                    iy_out_offset = (line_start - new_min_ypiece) / i_binning;
                    num_lines_write = (line_end + 1 - line_start) / i_binning;
                } else {
                    if get_string(b"SubsetToDo", 320, &mut file_name) != 0 {
                        exit_error("You must enter SubsetToDo with this parallel mode");
                    }
                    //
                    // For direct writing, save full want list as all want list
                    if mode_parallel == -2 {
                        let n = num_zwant as usize;
                        let (src, dst) = (&iz_want[..n], &mut iz_all_want[..n]);
                        dst.copy_from_slice(src);
                    }
                    //
                    // In either case, replace the actual want list
                    let _ = parselist(
                        file_name.trim_end_matches(' '),
                        &mut iz_want,
                        &mut num_zwant,
                    );
                }
                parallel_hdf = mode_parallel == -2
                    && ii_test_if_hdf(out_file.trim_end_matches(' ').as_bytes()) > 0;
                ierr = get_string(b"BoundaryInfoFile", 320, &mut bound_file);
                if parallel_hdf && blank(&bound_file) {
                    exit_error(
                        "A boundary info file must be entered for parallel mode -2 if output file type is HDF",
                    );
                }
            } else {
                //
                // Open output file: set Z size for the file (nzbin is kept as a
                // running count of sections output otherwise)
                bv.nxyz_bin[2] = num_zwant;
                parallel_hdf = b3d_output_file_type() == 5;
            }
        }
    }
    //
    // Initialize parallel writing if there is a boundary file
    ixy = bv.nxyz_bin[0];
    if parallel_hdf {
        ixy = -bv.nxyz_bin[0];
    }
    ierr = iiu_par_wrt_initialize(
        bound_file.trim_end_matches(' '),
        5,
        ixy,
        bv.nxyz_bin[1],
        nz_all_want,
    );
    if ierr != 0 {
        let _ = writeln!(
            out,
            "\nERROR: BLENDMONT - Initializing parallel write boundary file, error{}",
            i_edit(ierr, 3)
        );
        exit(1);
    }
    //
    // Make gradient file
    if sum_for_grad {
        other_grad_file = format!("{root}_grad.txt");
        if h.if_old_edge == 0 {
            // `write(edgeName, '(a,f20.10,1x,a,1x,a)') 'clip plane -l ',
            // denZeroBase, trim(imageInFile), trim(otherGradFile)`, then
            // `call system(trim(edgeName))`.  `clip` is our own command, so it
            // runs in this process (owner decision, `CLAUDE.md`, "Our own
            // commands are called in process"): the words `sh` would split the
            // line into are the program's `argv`, with `argv[0]` the path a
            // command link would have.  A line that needs more of the shell
            // than word splitting (a file name with a quote, `$`, glob, ...)
            // still goes to `/bin/sh -c` as `system()` would.  `system`'s
            // status is not tested in the source; neither is it here.
            h.edge_name = format!(
                "clip plane -l {} {} {}",
                f_edit(bv.den_zero_base, 20, 10),
                image_in_file.trim_end_matches(' '),
                other_grad_file.trim_end_matches(' ')
            );
            let _ = out.flush();
            let line = h.edge_name.trim_end_matches(' ');
            let plain = !line.chars().any(|c| "|&;<>()$`\\\"'*?[#~=%{}!".contains(c));
            if let (true, Some(clip)) = (plain, crate::imod::commands::find("clip")) {
                let mut argv: Vec<std::ffi::OsString> = Vec::new();
                for (k, word) in line.split_ascii_whitespace().enumerate() {
                    if k == 0 {
                        argv.push(match std::env::current_exe() {
                            Ok(path) => path.with_file_name(word).into_os_string(),
                            Err(_) => word.into(),
                        });
                    } else {
                        argv.push(word.into());
                    }
                }
                let _ = crate::imod::commands::run_in_process(clip, argv, None, false);
            } else {
                use std::os::unix::process::CommandExt;
                let _ = std::process::Command::new("/bin/sh")
                    .arg0("sh")
                    .arg("-c")
                    .arg(line)
                    .status();
            }
            exist = std::path::Path::new(other_grad_file.trim_end_matches(' ')).exists();
            if !exist {
                exit_error("Failed to find gradient file after running \"clip plane\"");
            }
        }
    }

    // Read gradient file
    if !blank(&other_grad_file) && (!sum_for_grad || h.if_old_edge == 0) {
        let name = other_grad_file.trim_end_matches(' ').to_string();
        let mut unit14 = BufReader::new(dopen(14, &name, "old", "f"));
        if let Err(err) = list_read(
            &mut unit14,
            &mut [
                ListItem::Real(&mut bv.x_base_grad_scale),
                ListItem::Real(&mut bv.y_base_grad_scale),
            ],
        ) {
            read_runtime_error(err);
        }
        // `close(14)` is the drop of the reader.
    }
    //
    let mut edges_incomplete = false;
    let mut old_edge_open_failed = false;
    if h.if_old_edge != 0 && !undistort_only {
        //
        // for old files, open, get edge count and # of grids in X and Y
        //
        // `err = 53` on the opens: any failure to open an old file branches to
        // the recovery below with `ixy` at the failing axis.
        let mut fail_ixy = 0;
        'open_old: {
            for ixy in 1..=2 {
                let xyu = (ixy - 1) as usize;
                len_record = 24;
                h.edge_name = format!("{root}{}", EDGE_EXTENSION[xyu]);
                let unit = match DirectUnit::open(bv.iun_edge[xyu], &h.edge_name, false, len_record)
                {
                    Ok(unit) => unit,
                    Err(_) => {
                        fail_ixy = ixy;
                        break 'open_old;
                    }
                };
                let rec = unit
                    .read_record(1)
                    .unwrap_or_else(|err| unit.runtime_error(err));
                for i in 0..5 {
                    num_edge_tmp[i + 5 * xyu] =
                        i32::from_ne_bytes(rec[4 * i..4 * i + 4].try_into().unwrap());
                }
                drop(unit);
                //
                // Do same for density samples
                if bv.i_dens_from_edges > 0 {
                    len_record = 36;
                    h.edge_name = format!("{root}{}", DENS_EXTENSION[xyu]);
                    let unit =
                        match DirectUnit::open(bv.iun_dens[xyu], &h.edge_name, false, len_record) {
                            Ok(unit) => unit,
                            Err(_) => {
                                exit_error(&format!("Opening edge density file {}", h.edge_name))
                            }
                        };
                    let rec = unit
                        .read_record(1)
                        .unwrap_or_else(|err| unit.runtime_error(err));
                    for i in 0..6 {
                        num_den_tmp[i + 6 * xyu] =
                            i32::from_ne_bytes(rec[4 * i..4 * i + 4].try_into().unwrap());
                    }
                    for i in 0..2 {
                        grad_tmp[i + 2 * xyu] =
                            f32::from_ne_bytes(rec[24 + 4 * i..28 + 4 * i].try_into().unwrap());
                    }
                    drop(unit);
                }
            }
            //
            // make sure edge counts match and intgrid was consistent
            //
            if num_edge_tmp[0] != bv.nedge[0] || num_edge_tmp[5] != bv.nedge[1] {
                let mut bytes: Vec<u8> =
                    num_edge_tmp.iter().flat_map(|v| v.to_ne_bytes()).collect();
                convert_longs(&mut bytes, 10);
                for (k, c) in bytes.chunks(4).enumerate() {
                    num_edge_tmp[k] = i32::from_ne_bytes(c.try_into().unwrap());
                }
                if num_edge_tmp[0] != bv.nedge[0] || num_edge_tmp[5] != bv.nedge[1] {
                    exit_error("Wrong # of edges in edge function file");
                }
                bv.need_byte_swap = 1;
                // Fixed in translation (`BUGS.md`): the source swaps only the
                // edge-function counts (`blendmont.f90:1127-1131`); the edge
                // density files written with them (header at `:1118`, records
                // at `:1954`) are used as read.  Their header is swapped here
                // and their records where they are read.
                let mut bytes: Vec<u8> = num_den_tmp.iter().flat_map(|v| v.to_ne_bytes()).collect();
                convert_longs(&mut bytes, 12);
                for (k, c) in bytes.chunks(4).enumerate() {
                    num_den_tmp[k] = i32::from_ne_bytes(c.try_into().unwrap());
                }
                let mut bytes: Vec<u8> = grad_tmp.iter().flat_map(|v| v.to_ne_bytes()).collect();
                convert_floats(&mut bytes, 4);
                for (k, c) in bytes.chunks(4).enumerate() {
                    grad_tmp[k] = f32::from_ne_bytes(c.try_into().unwrap());
                }
            }
            if num_edge_tmp[3] != num_edge_tmp[9] || num_edge_tmp[4] != num_edge_tmp[8] {
                exit_error("Inconsistent grid spacings between edge function files");
            }
            //
            if bv.i_dens_from_edges > 0 {
                if num_den_tmp[0] != bv.nedge[0] || num_den_tmp[6] != bv.nedge[1] {
                    exit_error("Wrong # of edges in edge density file");
                }
                if num_den_tmp[1] != num_den_tmp[7] {
                    exit_error("Inconsistent sampling factor between edge density files");
                }
                if bv.i_den_sample > 0 && num_den_tmp[1] != bv.i_den_sample {
                    exit_error(
                        "Sampling factor entered for fixing gradients does not match value in files",
                    );
                }
                if num_den_tmp[4] != num_den_tmp[11] || num_den_tmp[5] != num_den_tmp[10] {
                    exit_error("Inconsistent grid spacings between edge density files");
                }
                if !blank(&other_grad_file)
                    && (!sum_for_grad || h.if_old_edge == 0)
                    && (grad_tmp[0] != bv.x_base_grad_scale || grad_tmp[1] != bv.y_base_grad_scale)
                {
                    exit_error(
                        "The intensity gradient from image analysis does not match the value in the edge density files",
                    );
                }
                bv.i_den_sample = num_den_tmp[1];
            }
            //
            // set up record size and reopen with right record size
            //
            for ixy in 1..=2 {
                let xyu = (ixy - 1) as usize;
                bv.nx_grid[xyu] = num_edge_tmp[1 + 5 * xyu];
                bv.ny_grid[xyu] = num_edge_tmp[2 + 5 * xyu];
                len_record = 4 * 6.max(3 * (bv.nx_grid[xyu] * bv.ny_grid[xyu] + 2));
                h.edge_name = format!("{root}{}", EDGE_EXTENSION[xyu]);
                match DirectUnit::open(bv.iun_edge[xyu], &h.edge_name, false, len_record) {
                    Ok(unit) => units.edge[xyu] = Some(unit),
                    Err(_) => {
                        fail_ixy = ixy;
                        break 'open_old;
                    }
                }

                if bv.i_dens_from_edges > 0 {
                    bv.nx_den_grid[xyu] = num_den_tmp[2 + 6 * xyu];
                    bv.ny_den_grid[xyu] = num_den_tmp[3 + 6 * xyu];
                    len_record = 4 * 8.max(2 * (bv.nx_den_grid[xyu] * bv.ny_den_grid[xyu] + 3));
                    //print *,ixy,lenRecord, nxDenGrid(ixy), nyDenGrid(ixy)
                    h.edge_name = format!("{root}{}", DENS_EXTENSION[xyu]);
                    match DirectUnit::open(bv.iun_dens[xyu], &h.edge_name, false, len_record) {
                        Ok(unit) => units.dens[xyu] = Some(unit),
                        Err(_) => exit_error(&format!("Opening edge density file {}", h.edge_name)),
                    }
                }
            }
            //
            // Read edge headers to set up edgedone array
            num_zero = 0;
            for ixy in 1..=2 {
                let xyu = (ixy - 1) as usize;
                let mut read_all = true;
                for iedge in 1..=bv.nedge[xyu] {
                    // `err = 52`: a failed read ends this axis without counting it.
                    let Ok(rec) = units.edge[xyu].as_ref().unwrap().read_record(1 + iedge) else {
                        read_all = false;
                        break;
                    };
                    let mut bytes = rec[..8].to_vec();
                    if bv.need_byte_swap != 0 {
                        convert_longs(&mut bytes, 2);
                    }
                    ix_pc_lower[0] = i32::from_ne_bytes(bytes[0..4].try_into().unwrap());
                    ix_pc_lower[1] = i32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                    // print *,'edge', ixy, iedge, ixpclo(1), ixpclo(2)
                    if ix_pc_lower[0] >= 0 && ix_pc_lower[1] >= 0 {
                        h.edge_done[e2!(bv, iedge, ixy)] = true;
                    }
                }
                if read_all {
                    num_zero += 1;
                }
                // `52 continue`
            }
            //
            // If either set of edge functions is incomplete, set flag; only set
            // the interval now if edges are complete
            if num_zero < 2 {
                edges_incomplete = true;
            } else {
                bv.int_grid[0] = num_edge_tmp[3];
                bv.int_grid[1] = num_edge_tmp[4];
                if bv.i_dens_from_edges > 0 {
                    bv.interval_den[0] = num_den_tmp[4];
                    bv.interval_den[1] = num_den_tmp[5];
                }
            }
        }
        if fail_ixy > 0 {
            old_edge_open_failed = true;
            //
            // 8/8/03: if there is an error opening old files, just build new ones
            // This is to allow command files to say use old functions even if
            // a previous command file might not get run
            //
            // `53`
            h.if_old_edge = 0;
            if fail_ixy == 2 {
                units.edge[0] = None;
            }
            let _ = writeln!(
                out,
                "\nWARNING: BLENDMONT - Error opening old edge function file; new edge functions will be computed"
            );
            for ixy in 1..=2 {
                for iedge in 1..=bv.nedge[(ixy - 1) as usize] {
                    h.edge_done[e2!(bv, iedge, ixy)] = false;
                }
            }
        }
    }
    let _ = old_edge_open_failed;

    // `54`
    if mode_parallel < 0 && (h.if_old_edge == 0 || edges_incomplete) {
        exit_error("Parallel mode not allowed if edge functions are incomplete or nonexistent");
    }
    if (h.if_old_edge == 0 || edges_incomplete) && !undistort_only {
        if_diddle = 0;
        // write(*,'(1x,a,$)') &
        // '1 to diddle with edge function parameters: '
        // read(5,*) ifdiddle
        h.sd_crit = 2.;
        h.dev_crit = 2.;
        h.ipoly_order = 2;
        //
        // make smart defaults for grid parameters
        //
        {
            let big = if nxin > nyin { nxin } else { nyin } as f32 / 512.;
            let lo = if 1. > big { 1. } else { big };
            grid_scale = if 8. < lo { 8. } else { lo };
        }
        for ixy in 1..=2 {
            let xyu = (ixy - 1) as usize;
            bv.ibox_siz[xyu] = (bv.ibox_siz[xyu] as f32 * grid_scale).round() as i32;
            bv.indent[xyu] = (bv.indent[xyu] as f32 * grid_scale).round() as i32;
            bv.int_grid[xyu] = (bv.int_grid[xyu] as f32 * grid_scale).round() as i32;
            bv.last_written[xyu] = 0;
        }
        //
        if pip_input {
            let mut two = 2;
            ierr = pip_get_integer_array(b"BoxSizeShortAndLong", &mut bv.ibox_siz, &mut two, 2);
            two = 2;
            ierr = pip_get_integer_array(b"IndentShortAndLong", &mut bv.indent, &mut two, 2);
            two = 2;
            ierr = pip_get_integer_array(b"GridSpacingShortAndLong", &mut bv.int_grid, &mut two, 2);
        }
        // print *,'box size', (iboxsiz(i), i=1, 2), '  grid', (intgrid(i), i=1, 2)

        if if_diddle != 0 {
            let _ = write!(
                out,
                " criterion # of sds away from mean of sd and deviation: "
            );
            read5(&mut [
                ListItem::Real(&mut h.sd_crit),
                ListItem::Real(&mut h.dev_crit),
            ]);
            let _ = write!(
                out,
                " # grid positions in short and long directions to include in regression: "
            );
            let (a, b) = h.num_fit.split_at_mut(1);
            read5(&mut [ListItem::Integer(&mut a[0]), ListItem::Integer(&mut b[0])]);
            let _ = write!(out, " order of polynomial: ");
            read5(&mut [ListItem::Integer(&mut h.ipoly_order)]);
            let _ = write!(
                out,
                " intervals at which to do regression in short and long directions: "
            );
            let (a, b) = h.nskip_regress.split_at_mut(1);
            read5(&mut [ListItem::Integer(&mut a[0]), ListItem::Integer(&mut b[0])]);
        }
        //
        // get edge function characteristics for x and y edges
        //
        // print *,(iboxsiz(I), indent(i), intgrid(i), i=1, 2)
        for ixy in 1..=2 {
            let xyu = (ixy - 1) as usize;
            let (mut nxg, mut nyg) = (0i32, 0i32);
            setgridchars(
                &bv.nxyz_in,
                &bv.n_overlap,
                &bv.ibox_siz,
                &bv.indent,
                &bv.int_grid,
                ixy,
                0,
                0,
                0,
                0,
                &mut nxg,
                &mut nyg,
                &mut igrid_start,
                &mut i_offset,
            );
            bv.nx_grid[xyu] = nxg;
            bv.ny_grid[xyu] = nyg;
            // print *,ixy, nxgrid(ixy), nygrid(ixy), nedgetmp(2, ixy), nedgetmp(3, ixy)
            if edges_incomplete
                && (bv.nx_grid[xyu] != num_edge_tmp[1 + 5 * xyu]
                    || bv.ny_grid[xyu] != num_edge_tmp[2 + 5 * xyu])
            {
                exit_error("Cannot use incomplete old edge function file with current parameters");
            }
        }
        //
        // Figure out default sampling
        if bv.i_dens_from_edges > 0 {
            if bv.i_den_sample <= 0 {
                max_sampling = bv.ny_grid[0].min(bv.nx_grid[1]) / min_sample_per_edge;
                xtmp = (bv.nx_grid[0] * bv.ny_grid[0] * (nxp - 1)
                    + bv.nx_grid[1] * bv.ny_grid[1] * (nyp - 1)) as f32;
                bv.i_den_sample = (xtmp / num_optimal_samp as f32).sqrt() as i32;
                bv.i_den_sample = 1.max(bv.i_den_sample.min(max_sampling));
                let _ = out.flush();
            }
            //
            // Set up density sampling parameters
            for xyu in 0..2 {
                bv.nx_den_grid[xyu] = 1.max(bv.nx_grid[xyu] / bv.i_den_sample);
                bv.ny_den_grid[xyu] = 1.max(bv.ny_grid[xyu] / bv.i_den_sample);
                bv.interval_den[xyu] = bv.int_grid[xyu] * bv.i_den_sample;
            }
        }
    }

    if h.if_old_edge == 0 && !undistort_only && mode_parallel <= 0 {
        bv.need_byte_swap = 0;
        // open file, write header record
        // set record length to total bytes divided by system-dependent
        // number of bytes per item
        //
        for ixy in h.ixy_func_start..=h.ixy_func_end {
            let xyu = (ixy - 1) as usize;
            let yxu = (2 - ixy) as usize;
            len_record = 4 * 6.max(3 * (bv.nx_grid[xyu] * bv.ny_grid[xyu] + 2));
            h.edge_name = format!("{root}{}", EDGE_EXTENSION[xyu]);
            ierr = imod_backup_file(&h.edge_name);
            if ierr != 0 {
                let _ = writeln!(
                    out,
                    "\n WARNING: BLENDMONT - error renaming existing edge function file"
                );
            }
            let unit = DirectUnit::open(bv.iun_edge[xyu], &h.edge_name, true, len_record)
                .unwrap_or_else(|err| open_runtime_error(bv.iun_edge[xyu], &h.edge_name, err));
            // write header record
            let rec: Vec<u8> = [
                bv.nedge[xyu],
                bv.nx_grid[xyu],
                bv.ny_grid[xyu],
                bv.int_grid[xyu],
                bv.int_grid[yxu],
            ]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
            if let Err(err) = unit.write_record(1, &rec) {
                unit.runtime_error(err);
            }
            units.edge[xyu] = Some(unit);
            //
            // Same for density files
            if bv.i_den_sample > 0 {
                len_record = 4 * 8.max(2 * (bv.nx_den_grid[xyu] * bv.ny_den_grid[xyu] + 3));
                h.edge_name = format!("{root}{}", DENS_EXTENSION[xyu]);
                ierr = imod_backup_file(&h.edge_name);
                if ierr != 0 {
                    let _ = writeln!(
                        out,
                        "\n WARNING: BLENDMONT - error renaming existing edge density file"
                    );
                }
                let unit = DirectUnit::open(bv.iun_dens[xyu], &h.edge_name, true, len_record)
                    .unwrap_or_else(|err| open_runtime_error(bv.iun_dens[xyu], &h.edge_name, err));
                let mut rec: Vec<u8> = [
                    bv.nedge[xyu],
                    bv.i_den_sample,
                    bv.nx_den_grid[xyu],
                    bv.ny_den_grid[xyu],
                    bv.interval_den[xyu],
                    bv.interval_den[yxu],
                ]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
                rec.extend_from_slice(&bv.x_base_grad_scale.to_ne_bytes());
                rec.extend_from_slice(&bv.y_base_grad_scale.to_ne_bytes());
                if let Err(err) = unit.write_record(1, &rec) {
                    unit.runtime_error(err);
                }
                units.dens[xyu] = Some(unit);
            }
        }
    }
    //
    // Do not write ecd file if not all sections are being computed
    if h.if_old_edge == 1 && num_zwant != num_list_z {
        xc_write_out = false;
    }
    //
    // Read old .ecd file(s) or expected shifts
    if (h.xc_read_in || !blank(&ecd_for_expected)) && !undistort_only {
        ierr = pip_get_float(b"BinningForEdgeShifts", &mut ecd_binning);
        iedge_del_x = 0;
        iedge_del_y = 0;
        if pip_get_two_integers(b"OverlapForEdgeShifts", &mut iedge_del_x, &mut iedge_del_y) == 0 {
            iedge_del_x = bv.n_overlap[0] - (ecd_binning * iedge_del_x as f32).round() as i32;
            iedge_del_y = bv.n_overlap[1] - (ecd_binning * iedge_del_y as f32).round() as i32;
            let _ = writeln!(
                out,
                " Adjusting by {}{}",
                ld_int(iedge_del_x),
                ld_int(iedge_del_y)
            );
        }
        let mut unit5: Option<BufReader<File>> = None;
        if !blank(&ecd_for_expected) {
            // `iy` is not set on this branch in the source
            // (`blendmont.f90:1343-1349`): it keeps the exit value of the
            // edge-listing loop (`nxPieces + 1`) or of the parallel chunk
            // loop, and becomes the unit the Y edges are read from below --
            // an unconnected `fort.<n>` (End of file), or standard input for
            // a 4-wide montage.  Fixed in translation (`BUGS.md`): the Y
            // edges follow the X edges in the one `.ecd` file, unit 4.
            h.edge_name = ecd_for_expected.trim_end_matches(' ').to_string();
            iy = 4;
        } else {
            h.edge_name = format!("{root}{}", XCORR_EXTENSION[0]);
            exist = std::path::Path::new(&h.edge_name).exists();
            iy = 4;
            if !exist {
                //
                // If the file does not exist, look for the two separate files
                // and put second name in a different variable.  Set unit number to
                // read one file then the other.
                h.edge_name = format!("{root}{}", XCORR_EXTENSION[1]);
                exist = std::path::Path::new(&h.edge_name).exists();
                if exist {
                    edge_name2 = format!("{root}{}", XCORR_EXTENSION[2]);
                    // The source inquires `edgeName` again here, not
                    // `edgeName2` (so a missing `.yecd` is found by `dopen`).
                    exist = std::path::Path::new(&h.edge_name).exists();
                }
                if !exist {
                    h.edge_name = format!("{root}{}", XCORR_EXTENSION[0]);
                    let _ = writeln!(
                        out,
                        "\nERROR: BLENDMONT - Edge correlation file does not exist: {}",
                        h.edge_name
                    );
                    exit(1);
                }
                unit5 = Some(BufReader::new(dopen(5, &edge_name2, "ro", "f")));
                iy = 5;
            }
        }
        let mut unit4 = BufReader::new(dopen(4, &h.edge_name, "ro", "f"));
        {
            let (a, b) = num_edge_tmp.split_at_mut(5);
            if let Err(err) = list_read(
                &mut unit4,
                &mut [ListItem::Integer(&mut a[0]), ListItem::Integer(&mut b[0])],
            ) {
                read_runtime_error(err);
            }
        }
        if num_edge_tmp[0] != bv.nedge[0] || num_edge_tmp[5] != bv.nedge[1] {
            exit_error("Wrong # of edges in edge correlation file");
        }
        ix = 4;

        for ixy in 1..=2 {
            for i in 1..=bv.nedge[(ixy - 1) as usize] {
                let mut line = String::new();
                // `read(ix, '(a)') titleStr`.  Unit 5 is standard input unless
                // it was reconnected to the second file; any other unit number
                // (see `iy` above) is not connected, and gfortran connects it
                // to `fort.<unit>`, which ends at once.
                let got = if ix == 4 {
                    unit4.read_line(&mut line)
                } else if ix == 5 {
                    match unit5.as_mut() {
                        Some(reader) => reader.read_line(&mut line),
                        None => {
                            let _ = out.flush();
                            std::io::stdin().lock().read_line(&mut line)
                        }
                    }
                } else {
                    let fort = format!("fort.{ix}");
                    match std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .open(&fort)
                    {
                        Ok(file) => BufReader::new(file).read_line(&mut line),
                        Err(err) => Err(err),
                    }
                };
                if matches!(got, Ok(0) | Err(_)) {
                    read_runtime_error(ListReadError::End);
                }
                // `character*80 titleStr`: the record is truncated to 80.
                let line = line.trim_end_matches(['\r', '\n']);
                let title_rec: String = line.chars().take(80).collect();
                let mut title_vals = [0.0f32; 20];
                ix_out = 0;
                frefor(&title_rec, &mut title_vals, &mut ix_out);
                let k = e2!(bv, i, ixy);
                h.edge_displace_x[k] = title_vals[0] * ecd_binning;
                h.edge_displace_y[k] = title_vals[1] * ecd_binning;
                if ixy == 1 {
                    h.edge_displace_x[k] += iedge_del_x as f32;
                }
                if ixy == 2 {
                    h.edge_displace_y[k] += iedge_del_y as f32;
                }
                //
                // Read in skip edge flag and treat it same as when using an
                // exclusion model
                if ix_out > 2 && title_vals[2] != 0. {
                    bv.if_skip_edge[k] = 2;
                    if (h.edge_displace_x[k] != 0. || h.edge_displace_y[k] != 0.)
                        && if_use_adjusted > 0
                    {
                        bv.if_skip_edge[k] = 1;
                        if if_use_adjusted > 1 {
                            bv.if_skip_edge[k] = 0;
                        }
                        if bv.if_skip_edge[k] > 0 {
                            h.num_skipped_edges += 1;
                        }
                    }
                }
            }
            ix = iy;
        }
        // `close(4)`, `close(5)`: closing unit 5 closes standard input when
        // it was never reconnected, and a later `read(5, ...)` (the
        // interactive blending-width prompt) then finds it closed
        // (`blendmont.f90:1402`).  Fixed in translation (`BUGS.md`): unit 5
        // is closed only when it was reconnected to the second file, and
        // standard input stays open otherwise.
        drop(unit4);
        drop(unit5);
    }

    // Or get adjusted overlaps from mdoc file and set up edge displacements
    if expected_from_mdoc {
        // open the autodoc in HDF/IDOC or open the mdoc file
        let ftype = unsafe { iiu_file_type(1) };
        hdf_or_idoc_file = ftype == 5 || ftype == 7;
        if hdf_or_idoc_file {
            // Fortran wrapper `iiuretadocindex`: 1-based, errors unchanged.
            let err = unsafe { iiu_ret_adoc_index(1, 0, 0) };
            ind_adoc = if err < 0 { err } else { err + 1 };
            if ind_adoc <= 0 {
                exit_error("Getting autodoc index for HDF or IDOC file");
            }
        } else {
            // Fortran wrapper `adocopenimagemetadata`: 1-based on success.
            let err = adoc_open_image_metadata(
                image_in_file.trim_end_matches(' ').as_bytes(),
                1,
                &mut if_adoc_mont,
                &mut num_adoc_sect,
                &mut isect_type,
            );
            ind_adoc = if err >= 0 { err + 1 } else { err };
            if ind_adoc < 0 {
                h.edge_name = format!(
                    "Error{} trying to access metadata file {}.mdoc",
                    i_edit(ind_adoc, 3),
                    image_in_file.trim_end_matches(' ')
                );
                exit_error(&h.edge_name);
            }
        }
        let _ = adoc_set_current(ind_adoc - 1);
        bv.ilistz = 1;
        while bv.ilistz <= num_list_z {
            h.iz_sect = list_z[(bv.ilistz - 1) as usize];

            // Get adoc section
            let err = adoc_lookup_by_name_value(b"MontSection", h.iz_sect);
            isect_num = if err >= 0 { err + 1 } else { err };
            if isect_num <= 0 {
                h.edge_name = format!(
                    "No MontSection for section {} in mdoc file",
                    i_edit(h.iz_sect, 5)
                );
                exit_error(&h.edge_name);
            }

            // Get expected shifts, or fall back to angle and tilt angle
            let (ex1, ex2) = exp_shift_x.split_at_mut(1);
            let (ey1, ey2) = exp_shift_y.split_at_mut(1);
            if adoc_get_two_floats(
                b"MontSection",
                isect_num - 1,
                b"XEdgeExpectedShifts",
                &mut ex1[0],
                &mut ey1[0],
            ) != 0
                && adoc_get_two_floats(
                    b"MontSection",
                    isect_num - 1,
                    b"YEdgeExpectedShifts",
                    &mut ex2[0],
                    &mut ey2[0],
                ) != 0
            {
                if adoc_get_float(
                    b"MontSection",
                    isect_num - 1,
                    b"RotationAngle",
                    &mut axis_angle,
                ) != 0
                    || adoc_get_float(b"MontSection", isect_num - 1, b"TiltAngle", &mut tilt_angle)
                        != 0
                {
                    h.edge_name = format!(
                        "Getting TiltAngle or RotationAngle from MontSection for section {} in mdoc file",
                        i_edit(h.iz_sect, 5)
                    );
                    exit_error(&h.edge_name);
                }

                // Tilt-foreshorten by rotating axis to X, contracting Y, rotating back
                //print *,ilistz,izSect,isectNum,tiltAngle
                for ixy in 1..=2 {
                    if ixy == 1 {
                        xvec = (nxin - bv.n_overlap[0]) as f32;
                        yvec = 0.;
                    } else {
                        yvec = (nyin - bv.n_overlap[1]) as f32;
                        xvec = 0.;
                    }
                    cos_rot = gfortran_cosd_r4(axis_angle);
                    sin_rot = gfortran_sind_r4(axis_angle);
                    xrot = xvec * cos_rot + yvec * sin_rot;
                    yrot = -xvec * sin_rot + yvec * cos_rot;
                    yrot *= gfortran_cosd_r4(tilt_angle);
                    xback = xrot * cos_rot - yrot * sin_rot;
                    yback = xrot * sin_rot + yrot * cos_rot;
                    exp_shift_x[(ixy - 1) as usize] = xback - xvec;
                    exp_shift_y[(ixy - 1) as usize] = yback - yvec;
                }
            }

            // Fill in expected shifts for this section
            let iz = h.iz_sect + 1 - min_zpiece;
            for ixy in 1..=2 {
                for i in bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, iz, ixy)] + 1
                    ..=bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, iz + 1, ixy)]
                {
                    h.edge_displace_x[e2!(bv, i, ixy)] = exp_shift_x[(ixy - 1) as usize];
                    h.edge_displace_y[e2!(bv, i, ixy)] = exp_shift_y[(ixy - 1) as usize];
                }
                //print *, iEdgeZbase(iz, ixy), iEdgeZbase(iz + 1, ixy) - 1, expShiftX(ixy), &
                //    expShiftY(ixy)
            }
            bv.ilistz += 1;
        }
    }

    //
    // Determine size needed for output and correlation arrays
    bv.max_line_length = new_xframe + 32;
    bv.max_bsiz = (IFAST_SIZ + MAX_BIN) * bv.max_line_length;
    //
    // Find binning up to limit that will get padded size down to target
    bv.nbin_xcorr = mont_xc_find_binning(
        max_xcorr_binning,
        nxy_xcorr_target,
        0,
        &bv.nxyz_in,
        &bv.n_overlap,
        bv.aspect_max,
        bv.extra_width,
        bv.pad_frac,
        nice_fft_limit(),
        &mut nxy_padded,
        &mut nxy_boxed,
    );
    //
    // Take account of expected shifts
    if h.use_expected {
        nxy_padded = 0;
        nxy_boxed = 0;
        for bin_loop in 1..=2 {
            for iz in 1..=num_sect {
                for ixy in 1..=2 {
                    for i in bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, iz, ixy)] + 1
                        ..=bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, iz + 1, ixy)]
                    {
                        n_expected[0] = h.edge_displace_x[e2!(bv, i, ixy)].round() as i32;
                        n_expected[1] = h.edge_displace_y[e2!(bv, i, ixy)].round() as i32;
                        //
                        // First determine maximum binning needed
                        if bin_loop == 1 {
                            nbin_tmp = mont_xc_find_binning2(
                                max_xcorr_binning,
                                nxy_xcorr_target,
                                0,
                                ixy - 1,
                                &bv.nxyz_in,
                                &bv.n_overlap,
                                &n_expected,
                                bv.aspect_max,
                                bv.extra_width,
                                bv.pad_frac,
                                nice_fft_limit(),
                                &mut num_pad_tmp,
                                &mut num_box_tmp,
                            );
                            bv.nbin_xcorr = bv.nbin_xcorr.max(nbin_tmp);
                        } else {
                            // Then with that, get the box sizes
                            let xyu = (ixy - 1) as usize;
                            n_expected[xyu] = bv.n_overlap[xyu] + 0.max(-n_expected[xyu]);
                            montxcbasicsizes(
                                &(ixy + 2),
                                &bv.nbin_xcorr,
                                &0,
                                &bv.nxyz_in,
                                &n_expected,
                                &bv.aspect_max,
                                &bv.extra_width,
                                &bv.pad_frac,
                                &nice_fft_limit(),
                                &mut indent_tmp,
                                &mut nxy_box,
                                &mut n_extra,
                                &mut nx_pad,
                                &mut ny_pad,
                                &mut max_long_tmp,
                            );
                            nxy_padded = nxy_padded.max((nx_pad + 8) * (ny_pad + 8));
                            nxy_boxed = nxy_boxed.max((nxy_box[0] + 4) * (nxy_box[1] + 4));
                        }
                    }
                }
            }
        }
    }

    bv.idimc = nxy_padded;
    bv.max_bsiz = bv.max_bsiz.max(2 * nxy_boxed * bv.nbin_xcorr.pow(2));
    //print *,'nbinxcorr, dims', nbinXcorr, idimc, maxbsiz
    bin_line = vec![0.; bv.max_line_length as usize];
    bv.brray = vec![0.; bv.max_bsiz as usize];

    // Now that binning is known, set up grid map filter: convert from 1/micron value to
    // 1/pixel value using pixel size in microns
    if em_grid_filter > 0. {
        bv.radius2 = em_grid_filter * delta[0] * bv.nbin_xcorr as f32 / 10000.;
        bv.sigma2 = bv.radius2 / 8.;
    }

    bv.ixg_dim = 0;
    bv.iyg_dim = 0;
    for xyu in 0..2 {
        bv.ixg_dim = bv.ixg_dim.max(bv.nx_grid[xyu]);
        bv.iyg_dim = bv.iyg_dim.max(bv.ny_grid[xyu]);
    }
    //
    // Allocate arrays for density samples
    if bv.i_den_sample > 0 {
        ix = (bv.nx_den_grid[0] * bv.ny_den_grid[0]).max(bv.nx_den_grid[1] * bv.ny_den_grid[1]);
        let (ixu, mse) = (ix as usize, max_sec_edges as usize);
        bv.nx_den_buf = vec![0; mse];
        bv.ny_den_buf = vec![0; mse];
        bv.ix_den_start = vec![0; mse];
        bv.iy_den_start = vec![0; mse];
        bv.ix_den_offset = vec![0; mse];
        bv.iy_den_offset = vec![0; mse];
        bv.den_abuf = vec![0.; ixu * mse];
        bv.den_abuf_ext = [ixu, mse];
        bv.den_bbuf = vec![0.; ixu * mse];
        bv.den_bbuf_ext = [ixu, mse];
        bv.den_solution = vec![0.; bv.lim_npc as usize];
        bv.delta_den_buf = vec![0.; ixu * mse];
        bv.delta_den_buf_ext = [ixu, mse];
        bv.piece_scaling = vec![0.; (nxp * nyp) as usize];
        bv.piece_scaling_ext = [nxp as usize, nyp as usize];
        memory_error(0, "arrays for density buffers");
    }
    //
    // make default blending width be 80% of overlap up to 50, then
    // half of overlap above 50
    bv.iblend[0] = (bv.n_overlap[0] / 2)
        .max(50)
        .max((4 * bv.n_overlap[0] / 5).min(50));
    bv.iblend[1] = (bv.n_overlap[1] / 2)
        .max(50)
        .max((4 * bv.n_overlap[1] / 5).min(50));
    if pip_input {
        let mut two = 2;
        ierr = pip_get_integer_array(b"BlendingWidthXandY", &mut bv.iblend, &mut two, 2);
    } else {
        let _ = write!(
            out,
            " Blending width in X & Y (/ for{}{}): ",
            i_edit(bv.iblend[0], 5),
            i_edit(bv.iblend[1], 5)
        );
        // `read(5,*) iblend` after the `.ecd` read's `close(5)`: native then
        // reads `fort.5` (End of file).  Standard input stays connected here
        // (fixed in translation, `BUGS.md`).
        let (a, b) = bv.iblend.split_at_mut(1);
        read5(&mut [ListItem::Integer(&mut a[0]), ListItem::Integer(&mut b[0])]);
    }
    //
    // Do other Pip-only options,
    // Read in mag gradient and distortion field files if specified
    //
    if pip_input {
        bv.interp_order = 3;
        ierr = pip_get_integer(b"InterpolationOrder", &mut bv.interp_order);
        ierr = pip_get_logical("AdjustedFocus", &mut bv.focus_adjusted);
        if get_string(b"GradientFile", 320, &mut file_name) == 0 {
            bv.do_mag_grad = true;
            read_mag_gradients(
                file_name.trim_end_matches(' '),
                bv.lim_sect,
                &mut bv.pixel_mag_grad,
                &mut bv.axis_rot,
                &mut bv.tilt_angles,
                &mut bv.dmag_per_um,
                &mut bv.rot_per_um,
                &mut bv.num_mag_grad,
            );
            if bv.num_mag_grad != num_list_z {
                let _ = writeln!(
                    out,
                    " WARNING: BLENDMONT - # of mag gradients ({} ) does not match # of sections ({} )",
                    ld_int(bv.num_mag_grad),
                    ld_int(num_list_z)
                );
            }
            bv.num_angles = bv.num_mag_grad;
        }
        //
        // Look for tilt angles if no mag gradients, then adjust if any
        //
        if bv.num_angles == 0 && get_string(b"TiltFile", 320, &mut file_name) == 0 {
            let _ = out.flush();
            read_tilt_file(
                &mut bv.num_angles,
                14,
                file_name.trim_end_matches(' '),
                &mut bv.tilt_angles,
                bv.lim_sect,
            );
            if bv.num_angles != num_list_z {
                let _ = writeln!(
                    out,
                    " WARNING: BLENDMONT - # of tilt angles ({} ) does not match # of sections ({} )",
                    ld_int(bv.num_angles),
                    ld_int(num_list_z)
                );
            }
        }
        if bv.num_angles > 0 && pip_get_float(b"OffsetTilts", &mut tilt_offset) == 0 {
            for i in 1..=bv.num_angles {
                bv.tilt_angles[(i - 1) as usize] += tilt_offset;
            }
        }
        //
        // If doing added gradients, use a tiltgeometry entry only if no
        // gradient file
        //
        if pip_get_two_floats(b"AddToGradient", &mut del_dmag_per_um, &mut del_rot_per_um) == 0 {
            if bv.do_mag_grad {
                for i in 1..=bv.num_mag_grad {
                    bv.dmag_per_um[(i - 1) as usize] += del_dmag_per_um;
                    bv.rot_per_um[(i - 1) as usize] += del_rot_per_um;
                }
            } else {
                if pip_get_three_floats(
                    b"TiltGeometry",
                    &mut bv.pixel_mag_grad,
                    &mut bv.axis_rot,
                    &mut tilt_offset,
                ) != 0
                {
                    exit_error("-tilt or -gradient must be entered with -add");
                }
                bv.pixel_mag_grad *= 10.;
                bv.num_mag_grad = 1;
                bv.dmag_per_um[0] = del_dmag_per_um;
                bv.rot_per_um[0] = del_rot_per_um;
                bv.do_mag_grad = true;
                if bv.num_angles == 0 {
                    bv.num_angles = 1;
                    bv.tilt_angles[0] = tilt_offset;
                }
            }
        }
        if bv.do_mag_grad {
            bv.lm_field = 200;
            bv.max_fields = 16;
        }
        //
        if get_string(b"DistortionField", 320, &mut file_name) == 0 {
            bv.undistort = true;
            ierr = read_check_warp_file(
                file_name.trim_end_matches(' '),
                1,
                1,
                &mut idf_nx,
                &mut idf_ny,
                &mut ix,
                &mut idf_binning,
                &mut pixel_idf,
                &mut iy,
                &mut h.edge_name,
            );
            if ierr < 0 {
                exit_error(&h.edge_name);
            }

            ierr = get_grid_parameters(
                0,
                &mut bv.nx_field,
                &mut bv.ny_field,
                &mut bv.x_field_strt,
                &mut bv.y_field_strt,
                &mut bv.x_field_intrv,
                &mut bv.y_field_intrv,
            );
            bv.lm_field = bv.lm_field.max(bv.nx_field).max(bv.ny_field);

            if pip_get_float(b"ImagesAreBinned", &mut binning_of_input) != 0 {
                //
                // If input binning was not specified object if it is ambiguous
                //
                if nxin <= idf_nx * idf_binning / 2 && nyin <= idf_ny * idf_binning / 2 {
                    exit_error(
                        "You must specify binning of images because they are not larger than half the camera size",
                    );
                }
            }
            if binning_of_input <= 0. {
                exit_error("Image binning must be a positive number");
            }
        }
        //
        // Allocate field arrays and load distortion field
        if bv.do_mag_grad || bv.undistort {
            let lm = bv.lm_field as usize;
            let mf = bv.max_fields as usize;
            bv.dist_dx = vec![0.; lm * lm];
            bv.dist_dx_ext = [lm, lm];
            bv.dist_dy = vec![0.; lm * lm];
            bv.dist_dy_ext = [lm, lm];
            bv.field_dx = vec![0.; lm * lm * mf];
            bv.field_dx_ext = [lm, lm, mf];
            bv.field_dy = vec![0.; lm * lm * mf];
            bv.field_dy_ext = [lm, lm, mf];
            memory_error(0, "arrays for distortion fields");
        }
        if bv.undistort {
            if get_warp_grid(
                0,
                &mut bv.nx_field,
                &mut bv.ny_field,
                &mut bv.x_field_strt,
                &mut bv.y_field_strt,
                &mut bv.x_field_intrv,
                &mut bv.y_field_intrv,
                &mut bv.dist_dx,
                &mut bv.dist_dy,
                bv.lm_field,
            ) != 0
            {
                exit_error("Getting distortion field from warp file");
            }
            //
            // Adjust grid start and interval and field itself for the
            // overall binning
            //
            bin_ratio = idf_binning as f32 / binning_of_input;
            bv.x_field_strt *= bin_ratio;
            bv.y_field_strt *= bin_ratio;
            bv.x_field_intrv *= bin_ratio;
            bv.y_field_intrv *= bin_ratio;
            //
            // if images are not full field, adjust grid start by half the
            // difference between field and image size
            //
            bv.x_field_strt -= (idf_nx as f32 * bin_ratio - nxin as f32) / 2.;
            bv.y_field_strt -= (idf_ny as f32 * bin_ratio - nyin as f32) / 2.;
            //
            // scale field
            for iy in 1..=bv.ny_field {
                for i in 1..=bv.nx_field {
                    let k = a2!(bv.dist_dx_ext, i, iy);
                    bv.dist_dx[k] *= bin_ratio;
                    bv.dist_dy[k] *= bin_ratio;
                }
            }
        }
        // print *,xFieldStrt, yfieldStrt, xFieldIntrv, yFieldIntrv
        // write(*,'(10f7.2)') (distDx(i, 5), distDy(i, 5), i=1, min(nxField, 10))
        //
        // Handle debug output - open files and set flags
        //
        if xcorr_debug {
            if nxp > 1 {
                h.edge_name = format!("{root}.xdbg");
                imopen(3, &h.edge_name, "new");
                bv.if_dump_xy[0] = 0;
            }
            if nyp > 1 {
                h.edge_name = format!("{root}.ydbg");
                imopen(4, &h.edge_name, "new");
                bv.if_dump_xy[1] = 0;
            }
        }
        //
        // Get fill treatment; if not entered and very sloppy and distortion,
        // set for taper
        if pip_get_integer(b"TreatFillForXcorr", &mut bv.ifill_treatment) != 0
            && very_sloppy
            && (bv.undistort || bv.do_mag_grad)
        {
            bv.ifill_treatment = 2;
        }
        //
        // Check for model of edges to exclude
        if get_string(b"SkipEdgeModelFile", 320, &mut file_name) == 0 && !undistort_only {
            let lim_edge = bv.lim_edge;
            read_exclusion_model(
                &mut bv,
                &mut fm,
                file_name.trim_end_matches(' '),
                &h.edge_displace_x,
                &h.edge_displace_y,
                lim_edge,
                if_use_adjusted,
                &map_all_piece,
                nxp,
                nyp,
                min_zpiece,
                &mut h.num_skipped_edges,
            );
        }
    }
    //
    // Now that skip flags have been set, write ecd file if
    // it was read in from two halves
    if !blank(&edge_name2) {
        write_edge_correlations(&mut h, &bv);
    }
    //
    bv.do_fields = bv.undistort || bv.do_mag_grad;
    if undistort_only && !bv.do_fields {
        exit_error("You mustenter -gradient and/or -distort with -justUndistort");
    }
    if undistort_only && h.test_mode {
        exit_error("You cannot enter both -test and -justUndistort");
    }

    //
    // Set up for warping
    if do_warp {
        if pip_get_two_integers(
            b"UnalignedStartingXandY",
            &mut ix_unali_start,
            &mut iy_unali_start,
        ) != 0
        {
            //
            // If there is no unaligned start entered and the output area matches the
            // warp file area, then assume the same start as the current output
            ix_unali_start = new_min_xpiece;
            iy_unali_start = new_min_ypiece;
            //
            // Otherwise issue warning if sizes don't match and assume it was centered on
            // input / full output
            if (warp_scale * iwarp_nx as f32).round() as i32 != new_xtotal_pix
                || (warp_scale * iwarp_ny as f32).round() as i32 != new_ytotal_pix
            {
                ix_unali_start = (bv.min_xpiece as f32 + nx_total_pix as f32 / 2.
                    - warp_scale * iwarp_nx as f32 / 2.)
                    .round() as i32;
                iy_unali_start = (bv.min_ypiece as f32 + ny_total_pix as f32 / 2.
                    - warp_scale * iwarp_ny as f32 / 2.)
                    .round() as i32;
                h.edge_name = format!(
                    "WARNING: BLENDMONT - Area being output is different size from unaligned area; you may need to enter UnalignedStartingXandY for warping to work right; assuming starts of{}{}",
                    i_edit(ix_unali_start, 7),
                    i_edit(iy_unali_start, 7)
                );
                let _ = writeln!(out, "\n{}", h.edge_name);
            }
        }
        //
        // Given starting coordinate for warping coordinates, get an offset from that
        // Just divide the offset by warpScale here, it's never needed otherwise
        warp_xoffset = (new_min_xpiece as f32 + new_xtotal_pix as f32 / 2.
            - (ix_unali_start as f32 + warp_scale * iwarp_nx as f32 / 2.))
            / warp_scale;
        warp_yoffset = (new_min_ypiece as f32 + new_ytotal_pix as f32 / 2.
            - (iy_unali_start as f32 + warp_scale * iwarp_ny as f32 / 2.))
            / warp_scale;
        if set_current_warp_file(ind_warp_file) != 0 {
            exit_error("Setting current warp file");
        }
        num_control = vec![0; num_gxforms as usize];
        memory_error(0, "array for number of control points");
        if find_max_grid_size(
            warp_xoffset,
            warp_xoffset + new_xtotal_pix as f32 / warp_scale,
            warp_yoffset,
            warp_yoffset + new_ytotal_pix as f32 / warp_scale,
            &mut num_control,
            &mut bv.lm_warp_x,
            &mut bv.lm_warp_y,
            &mut h.edge_name,
        ) != 0
        {
            exit_error(&h.edge_name);
        }
        let (lx, ly) = (bv.lm_warp_x as usize, bv.lm_warp_y as usize);
        bv.warp_dx = vec![0.; lx * ly];
        bv.warp_dx_ext = [lx, ly];
        bv.warp_dy = vec![0.; lx * ly];
        bv.warp_dy_ext = [lx, ly];
        memory_error(0, "arrays for warping grids");
    }

    pip_done();
    //
    // Allocate more arrays now
    drop(map_all_piece);
    {
        let half = (bv.idimc / 2) as usize;
        // `complex*8 xcray(idimc / 2)`: interleaved (re, im), twice the count.
        bv.xcray = vec![0.; 2 * half];
        bv.xdray = vec![0.; 2 * half];
        bv.map_piece = vec![0; (nxp * nyp) as usize];
        bv.map_piece_ext = [nxp as usize, nyp as usize];
        bv.map_disjoint = vec![0; (nxp * nyp) as usize];
        bv.map_disjoint_ext = [nxp as usize, nyp as usize];
        bv.any_disjoint = vec![false; (nxp * nyp) as usize];
        bv.any_disjoint_ext = [nxp as usize, nyp as usize];
        let lv = bv.lim_var as usize;
        bv.lim_data_lo = vec![0; lv * 4];
        bv.lim_data_lo_ext = [lv, 2, 2];
        bv.lim_data_hi = vec![0; lv * 4];
        bv.lim_data_hi_ext = [lv, 2, 2];
        let (gx, gy, lb) = (
            bv.ixg_dim as usize,
            bv.iyg_dim as usize,
            LIM_EDG_BF as usize,
        );
        bv.dx_gr_bf = vec![0.; gx * gy * lb];
        bv.dx_gr_bf_ext = [gx, gy, lb];
        bv.dy_gr_bf = vec![0.; gx * gy * lb];
        bv.dy_gr_bf_ext = [gx, gy, lb];
        bv.dden_gr_bf = vec![0.; gx * gy * lb];
        bv.dden_gr_bf_ext = [gx, gy, lb];
        bv.dx_grid = vec![0.; gx * gy];
        bv.dx_grid_ext = [gx, gy];
        bv.dy_grid = vec![0.; gx * gy];
        bv.dy_grid_ext = [gx, gy];
        bv.dden_grid = vec![0.; gx * gy];
        bv.dden_grid_ext = [gx, gy];
        bv.sd_grid = vec![0.; gx * gy];
        bv.sd_grid_ext = [gx, gy];
        memory_error(0, "correlation or edge buffer arrays");
    }
    ix = bv.idimc / 4;
    if bv.num_xcorr_peaks > 1 && !h.xc_legacy {
        ix = bv.idimc / 2;
    }
    bv.xeray = vec![0.; 2 * ix as usize];
    memory_error(0, "correlation array");
    //
    // Set maximum load; allow one extra slot if doing fields
    // Limit by the maximum number of pieces that could be needed, which is
    // the full array, or just one if undistorting
    bv.npix_in = nxin as i64 * nyin as i64;
    most_needed = if undistort_only { 1 } else { 2.max(nxp * nyp) };
    //
    // First compute a limit based on the minimum memory we would like to use
    if bv.do_fields {
        bv.max_load = (MEM_MINIMUM as i64 / bv.npix_in - 1)
            .min(bv.mem_lim as i64)
            .min(bv.max_fields as i64)
            .min(most_needed as i64) as i32;
        ix = 1;
    } else {
        bv.max_load = (MEM_MINIMUM as i64 / bv.npix_in)
            .min(bv.mem_lim as i64)
            .min(most_needed as i64) as i32;
        ix = 0;
    }
    //
    // Increase that to 4 or whatever is needed if it is below that
    // but if this doesn't fit into the preferred memory limit, then
    // just go for 2 pieces and make sure that fits too!
    bv.max_load = bv.max_load.max(4.min(most_needed));
    if (bv.max_load + ix) as i64 * bv.npix_in > MEM_PREFERRED as i64 {
        bv.max_load = 2.min(most_needed);
    }
    if (bv.max_load + ix) as i64 * bv.npix_in > MEM_MAXIMUM as i64 {
        exit_error("Images too large for 32-bit array indexes");
    }
    bv.max_siz = ((bv.max_load + ix) as i64 * bv.npix_in) as i32;
    // `shuffler` keeps one distortion field per cache slot,
    // `fieldDx(:, :, ioldest)` (`shuffler.f90:95-96`), but with `-distort` and
    // no gradients `maxFields` is 1 while the `max(maxLoad, min(4, ...))` above
    // allows up to 4 slots: the source writes the fields of slots 2-4 past the
    // end of `fieldDx`/`fieldDy` (a heap overflow; `BUGS.md`).  Each slot's
    // grid is written just before `warpInterp` reads it, so where native
    // survives it computes what per-slot storage would.  Fixed in
    // translation (`BUGS.md`): the arrays get one field per cache slot.
    if bv.do_fields && bv.max_load > bv.max_fields {
        let (lm, slots) = (bv.lm_field as usize, bv.max_load as usize);
        bv.field_dx.resize(lm * lm * slots, 0.);
        bv.field_dx_ext[2] = slots;
        bv.field_dy.resize(lm * lm * slots, 0.);
        bv.field_dy_ext[2] = slots;
    }
    bv.array = vec![0.; bv.max_siz as usize];
    let _ = writeln!(
        out,
        "Allocated{} MB of memory for main image array",
        i_edit(bv.max_siz / (1024 * 256), 5)
    );
    //
    // initialize memory allocator and dx, dy lists for looking for near piece
    //
    clear_shuffle(&mut bv);
    init_near_list(&mut bv);
    //
    bv.int_gr_copy = bv.int_grid;
    //
    // All errors checked, exit if setting up parallel
    if mode_parallel > 0 {
        exit(0);
    }
    //
    // Now open output files after errors have been checked, unless direct
    // writing in parallel mode
    if mode_parallel != -2 && h.if_edge_func_only == 0 {
        //
        // Do as much of header as possible, shift origin same as in newstack
        imopen(2, out_file.trim_end_matches(' '), "new");
        iiu_trans_header(2, 1);
        iiu_alt_num_extended(2, 0);
        iiu_alt_extended_type(2, &[0, 0]);
        iiu_alt_mode(2, mode_out);
        iiu_alt_size(2, &bv.nxyz_bin, &nxyzst);
        [x_origin, y_origin, z_origin] = iiu_ret_origin(1);
        x_origin -= delta[0] * ix_offset as f32;
        y_origin -= delta[1] * iy_offset as f32;
        if adjust_origin {
            x_origin -= delta[0] * (new_min_xpiece - bv.min_xpiece) as f32;
            y_origin -= delta[1] * (new_min_ypiece - bv.min_ypiece) as f32;
            z_origin -= delta[2] * (iz_want[0] - list_z[0]) as f32;
        }
        iiu_alt_origin(2, &[x_origin, y_origin, z_origin]);
        cell[0] = bv.nxyz_bin[0] as f32 * delta[0] * i_binning as f32;
        cell[1] = bv.nxyz_bin[1] as f32 * delta[1] * i_binning as f32;
        //
        // Finish it and exit if setting up for direct writing
        if mode_parallel == -1 {
            iiu_alt_sample(2, &bv.nxyz_bin);
            cell[2] = bv.nxyz_bin[2] as f32 * delta[2];
            iiu_alt_cell(2, &cell);
        }
        //
        let mut dat = [b' '; 9];
        let mut tim = [b' '; 8];
        b3d_date(&mut dat);
        time(&mut tim);
        // `90 format( 'BLENDMONT: Montage pieces ',a, t57, a9, 2x, a8 )`
        // into `character*80 titleStr`.
        title_str = format!(
            "{:<56}{}  {}",
            format!("BLENDMONT: Montage pieces {:<18}", action_str),
            String::from_utf8_lossy(&dat),
            String::from_utf8_lossy(&tim)
        );
        if parallel_hdf {
            unsafe { iiu_write_dummy_sec_to_hdf(2) };
        }
        iiu_write_header_str(2, &title_str, 1, dmin, dmax, bv.dmean);
        if mode_parallel == -1 {
            unsafe { iiu_close(2) };
            exit(0);
        }
    } else if h.if_edge_func_only == 0 {
        let (mut all_sec, mut lines_bound, mut nfiles) = (0i32, 0i32, 0i32);
        if parallel_hdf {
            par_wrt_properties(&mut all_sec, &mut lines_bound, &mut nfiles);
            ixy = all_sec;
            ipc = lines_bound;
            ix = nfiles;
            if b3d_lock_file(ixy) != 0 {
                exit_error("Could not get lock for opening HDF file");
            }
        }
        imopen(2, out_file.trim_end_matches(' '), "old");
        let (mut hmode, mut hmin, mut hmax, mut hmean) = (0i32, 0.0f32, 0.0f32, 0.0f32);
        unsafe {
            irdhdr(
                2,
                ix_pc_lower.as_mut_ptr(),
                ix_pc_upper.as_mut_ptr(),
                &mut hmode,
                &mut hmin,
                &mut hmax,
                &mut hmean,
            );
        }
        ix_out = hmode;
        bv.wll = hmin;
        bv.wlr = hmax;
        bv.wul = hmean;
        if ix_pc_lower[0] != bv.nxyz_bin[0]
            || ix_pc_lower[1] != bv.nxyz_bin[1]
            || ix_out != mode_out
        {
            exit_error("Existing output file does not have right size or mode");
        }
        if parallel_hdf {
            unsafe { iiu_par_wrt_reclose_hdf(2, 1) };
        }
    }
    //
    let mut unit3: Option<BufWriter<File>> = None;
    if outputpl {
        unit3 = Some(BufWriter::new(dopen(
            3,
            pl_out_file.trim_end_matches(' '),
            "new",
            "f",
        )));
    }
    fast_cum = 0.;
    slow_cum = 0.;
    if bv.i_dens_from_edges > 1 {
        h.one_edge_just_avg_crit = 20.;
    }
    //
    // loop on z: do everything within each section for maximum efficiency
    //
    // The DO variable is the module variable `ilistz` (`getBestPieceShifts`
    // reads it); nothing in the body assigns it.
    'sections: for ilistz in 1..=num_list_z {
        bv.ilistz = ilistz;
        bv.doing_edge_func = true;
        h.iz_sect = list_z[(ilistz - 1) as usize];
        bv.i_apply_dens_scaling = 0;
        bv.num_max_sds = 0;
        //
        // test if this section is wanted in output: if not, skip in test mode
        //
        if_want = 0;
        for iwant in 1..=num_zwant {
            if iz_want[(iwant - 1) as usize] == h.iz_sect {
                if_want = 1;
            }
        }
        if if_want == 0 && (h.test_mode || undistort_only) {
            continue 'sections;
        }
        //
        // Look up actual section to write if doing direct parallel writes
        if if_want != 0 && mode_parallel == -2 && !y_chunks {
            for i in 1..=nz_all_want {
                if iz_all_want[(i - 1) as usize] == h.iz_sect {
                    num_out = i - 1;
                }
            }
        }
        //
        let _ = writeln!(out, " working on section #{}", i_edit(h.iz_sect, 5));
        let _ = out.flush();
        bv.multng = multi_neg[(h.iz_sect - min_zpiece) as usize];
        h.x_is_long_dim = nxp > nyp;
        bv.iedge_cur_base[0] =
            bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, h.iz_sect + 1 - min_zpiece, 1)];
        bv.iedge_cur_base[1] =
            bv.i_edge_zbase[a2!(bv.i_edge_zbase_ext, h.iz_sect + 1 - min_zpiece, 2)];
        //
        // make a map of pieces in this section and set up index to data limits
        //
        bv.map_piece.fill(0);
        ipc_lower = 0;
        for ipc in 1..=bv.npc_list {
            let iu = (ipc - 1) as usize;
            if bv.iz_pc_list[iu] == h.iz_sect {
                ix_frame = 1 + (bv.ix_pc_list[iu] - bv.min_xpiece) / (nxin - bv.n_overlap[0]);
                iy_frame = 1 + (bv.iy_pc_list[iu] - bv.min_ypiece) / (nyin - bv.n_overlap[1]);
                let k = pdx(&bv, ix_frame, iy_frame);
                bv.map_piece[k] = ipc;
                ipc_lower += 1;
                bv.lim_data_ind[iu] = ipc_lower;
                let lv = bv.lim_data_lo_ext[0];
                for i in 1..=2usize {
                    let base = (ipc_lower - 1) as usize + lv * (i - 1);
                    bv.lim_data_lo[base] = -1;
                    bv.lim_data_lo[base + 2 * lv] = -1;
                    bv.lim_data_hi[base] = -1;
                    bv.lim_data_hi[base + 2 * lv] = -1;
                }
            }
        }
        //
        if !undistort_only {
            //
            // First get edge functions for the section if they haven't been done
            find_section_edge_functions(&mut h, &mut bv, &mut units);
            //
            // Load all the edge density values into the buffers if using old edges
            if h.if_old_edge != 0 && bv.i_den_sample > 0 {
                for ixy in 1..=2 {
                    let xyu = (ixy - 1) as usize;
                    for iy_frame in 1..=nyp {
                        for ix_frame in 1..=nxp {
                            ipc = bv.map_piece[pdx(&bv, ix_frame, iy_frame)];
                            if ipc > 0 {
                                jedge = bv.iedge_upper[a2!(bv.iedge_upper_ext, ipc, ixy)];
                                if jedge > 0 && bv.if_skip_edge[e2!(bv, jedge, ixy)] == 0 {
                                    let ixb = (ixy - 1) * bv.ix_dim_den_buf + jedge
                                        - bv.iedge_cur_base[xyu];
                                    let iu = (ixb - 1) as usize;
                                    let unit = units.dens[xyu].as_ref().unwrap();
                                    let rec = unit
                                        .read_record(jedge + 1)
                                        .unwrap_or_else(|err| unit.runtime_error(err));
                                    // Swapped for an old file of the other byte
                                    // order (fixed in translation, `BUGS.md`).
                                    let swap = bv.need_byte_swap != 0;
                                    let word = |k: usize| -> [u8; 4] {
                                        let mut w: [u8; 4] =
                                            rec[4 * k..4 * k + 4].try_into().unwrap();
                                        if swap {
                                            w.reverse();
                                        }
                                        w
                                    };
                                    bv.nx_den_buf[iu] = i32::from_ne_bytes(word(0));
                                    bv.ny_den_buf[iu] = i32::from_ne_bytes(word(1));
                                    bv.ix_den_start[iu] = i32::from_ne_bytes(word(2));
                                    bv.iy_den_start[iu] = i32::from_ne_bytes(word(3));
                                    bv.ix_den_offset[iu] = i32::from_ne_bytes(word(4));
                                    bv.iy_den_offset[iu] = i32::from_ne_bytes(word(5));
                                    let npts = bv.nx_den_buf[iu] * bv.ny_den_buf[iu];
                                    if 4 * (6 + 2 * npts.max(0) as usize) > rec.len() {
                                        unit.runtime_error(std::io::Error::other(
                                            "Short record on unformatted read",
                                        ));
                                    }
                                    for i in 1..=npts {
                                        let k = (i - 1) as usize + bv.den_abuf_ext[0] * iu;
                                        bv.den_abuf[k] =
                                            f32::from_ne_bytes(word(6 + 2 * (i - 1) as usize));
                                        bv.den_bbuf[k] =
                                            f32::from_ne_bytes(word(7 + 2 * (i - 1) as usize));
                                    }
                                    let _ = writeln!(
                                        out,
                                        "{}{}{}{}",
                                        ld_int(ixy),
                                        ld_int(ix_frame),
                                        ld_int(iy_frame),
                                        ld_int(npts)
                                    );
                                    for i in 1..=npts {
                                        let k = (i - 1) as usize + bv.den_abuf_ext[0] * iu;
                                        if bv.den_abuf[k].is_nan() {
                                            let _ = writeln!(
                                                out,
                                                " A{}{}",
                                                ld_int(ixb),
                                                ld_real(bv.den_abuf[k])
                                            );
                                        }
                                        if bv.den_bbuf[k].is_nan() {
                                            let _ = writeln!(
                                                out,
                                                " B{}{}",
                                                ld_int(ixb),
                                                ld_real(bv.den_bbuf[k])
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //
            // Solve for the gradient and individual piece factors
            if bv.i_den_sample > 0 {
                solve_scaling(&mut bv, &mut h.dx_grid_mean, &mut h.dy_grid_mean, h.iz_sect);
                bv.i_apply_dens_scaling = bv.i_dens_from_edges;
                scale_cached_pieces(&mut bv, h.iz_sect);
            }
        }
        //
        // now if doing multinegatives, need to solve for h transforms
        //
        if bv.multng && !undistort_only {
            find_multineg_transforms(&mut h, &mut bv, &mut units);
        } //end of multineg stuff

        if !undistort_only {
            //
            // initialize edge buffer allocation
            //
            bv.jus_edg_ct = 0;
            for ixy in 1..=2 {
                for i in 1..=bv.nedge[(ixy - 1) as usize] {
                    bv.ibuf_edge[e2!(bv, i, ixy)] = 0;
                }
            }
            for i in 0..LIM_EDG_BF as usize {
                bv.iedg_bf_list[i] = -1;
                bv.las_edg_use[i] = 0;
            }
            //
            // if this section is not wanted in output, skip out
            //
            if if_want == 0 {
                continue 'sections;
            }
            //
            // scan through all edges in this section to determine limits for
            // when a point is near an edge
            //
            for ixy in h.ixy_func_start..=h.ixy_func_end {
                let xyu = (ixy - 1) as usize;
                bv.edge_lo_near[xyu] = 0.;
                bv.edge_hi_near[xyu] = bv.nxyz_in[xyu] as f32 - 1.;
                for iedge in 1..=bv.nedge[xyu] {
                    if bv.iz_pc_list[(bv.ipiece_lower[e2!(bv, iedge, ixy)] - 1) as usize]
                        == h.iz_sect
                    {
                        jedge = iedge;
                        if h.use_edges {
                            find_edge_to_use(&bv, iedge, ixy, &mut jedge);
                        }
                        // print *,'reading header of edge', ixy, jedge, ' for edge', ixy, iedge
                        let unit = units.edge[xyu].as_ref().unwrap();
                        let rec = unit
                            .read_record(1 + jedge)
                            .unwrap_or_else(|err| unit.runtime_error(err));
                        let mut bytes = rec[..24].to_vec();
                        if bv.need_byte_swap != 0 {
                            convert_longs(&mut bytes, 6);
                        }
                        let word = |k: usize| {
                            i32::from_ne_bytes(bytes[4 * k..4 * k + 4].try_into().unwrap())
                        };
                        nx_grid_in = word(0);
                        ny_grid_in = word(1);
                        igrid_start[0] = word(2);
                        i_offset[0] = word(3);
                        igrid_start[1] = word(4);
                        i_offset[1] = word(5);
                        let hi = igrid_start[xyu] as f32;
                        bv.edge_hi_near[xyu] = if hi < bv.edge_hi_near[xyu] {
                            hi
                        } else {
                            bv.edge_hi_near[xyu]
                        };
                        let lo = ((nx_grid_in.min(ny_grid_in) - 1) * bv.int_grid[0]
                            + i_offset[xyu]
                            + 10) as f32;
                        bv.edge_lo_near[xyu] = if lo > bv.edge_lo_near[xyu] {
                            lo
                        } else {
                            bv.edge_lo_near[xyu]
                        };
                        // print *,nxgr, nygr, (igridstr(i), iofset(i), i=1, 2)
                    }
                }
                // print *,ixy, edgelonear(ixy), edgehinear(ixy)
            }
            h.edges_separated = bv.edge_hi_near[0] - bv.edge_lo_near[0] > 0.05 * nxin as f32
                && bv.edge_hi_near[1] - bv.edge_lo_near[1] > 0.05 * nyin as f32;
        }
        //
        // Now analyze for h transforms if need to shift each piece - this sets
        // multng because it is a flag that hinv exists and needs to be used
        //
        bv.map_disjoint.fill(0);
        bv.any_disjoint.fill(false);
        if h.shift_each && !undistort_only && h.if_edge_func_only != 1 && h.if_edge_func_only != 2 {
            get_best_piece_shifts(&mut h, &mut bv, &mut units);
            //
            // Analyze for disjoint edges on this section
            // Look for non-overlap between cross-corner pieces and code by type
            for iy_frame in 1..=nyp - 1 {
                for ix_frame in 1..=nxp - 1 {
                    let m = |bv: &BlendVars, x: i32, y: i32| bv.map_piece[pdx(bv, x, y)];
                    if m(&bv, ix_frame, iy_frame) > 0
                        && m(&bv, ix_frame + 1, iy_frame) > 0
                        && m(&bv, ix_frame, iy_frame + 1) > 0
                        && m(&bv, ix_frame + 1, iy_frame + 1) > 0
                    {
                        let kd = pdx(&bv, ix_frame, iy_frame);
                        ix = m(&bv, ix_frame, iy_frame);
                        iy = m(&bv, ix_frame + 1, iy_frame + 1);
                        let (ixu, iyu) = ((ix - 1) as usize, (iy - 1) as usize);
                        if bv.ix_pc_list[ixu] as f32 + h.hxf[h3!(1, 3, ix)] + nxin as f32
                            <= bv.ix_pc_list[iyu] as f32 + h.hxf[h3!(1, 3, iy)]
                        {
                            bv.map_disjoint[kd] = 1;
                        }
                        if bv.iy_pc_list[ixu] as f32 + h.hxf[h3!(2, 3, ix)] + nyin as f32
                            <= bv.iy_pc_list[iyu] as f32 + h.hxf[h3!(2, 3, iy)]
                        {
                            bv.map_disjoint[kd] = 3;
                        }
                        ix = m(&bv, ix_frame, iy_frame + 1);
                        iy = m(&bv, ix_frame + 1, iy_frame);
                        let (ixu, iyu) = ((ix - 1) as usize, (iy - 1) as usize);
                        if bv.ix_pc_list[ixu] as f32 + h.hxf[h3!(1, 3, ix)] + nxin as f32
                            <= bv.ix_pc_list[iyu] as f32 + h.hxf[h3!(1, 3, iy)]
                        {
                            bv.map_disjoint[kd] = 2;
                        }
                        if bv.iy_pc_list[iyu] as f32 + h.hxf[h3!(2, 3, iy)] + nyin as f32
                            <= bv.iy_pc_list[ixu] as f32 + h.hxf[h3!(2, 3, ix)]
                        {
                            bv.map_disjoint[kd] = 4;
                        }
                        if bv.map_disjoint[kd] != 0 {
                            for yy in iy_frame..=iy_frame + 1 {
                                for xx in ix_frame..=ix_frame + 1 {
                                    let k = a2!(bv.any_disjoint_ext, xx, yy);
                                    bv.any_disjoint[k] = true;
                                }
                            }
                        }
                        // if (mapDisjoint(ixfrm, iyfrm) .ne. 0) print *,'disjoint', ixfrm, iyfrm
                    }
                }
            }
        }
        //
        // if doing g transforms, get inverse and recenter it from center
        // of output image to corner of image
        //
        if bv.do_gxforms {
            if skip_xforms {
                ind_gxf = h.iz_sect + 1 - min_zpiece;
            } else {
                for ilis in 1..=num_list_z {
                    if list_z[(ilis - 1) as usize] == h.iz_sect {
                        ind_gxf = ilis;
                    }
                }
            }
            xfcopy(&gxf[h3!(1, 1, ind_gxf)..], &mut gxf_temp);
            gxf_temp[4] = gxf_temp[4] + (1. - gxf_temp[0]) * bv.gx_cen - gxf_temp[2] * bv.gy_cen;
            gxf_temp[5] = gxf_temp[5] + (1. - gxf_temp[3]) * bv.gy_cen - gxf_temp[1] * bv.gx_cen;
            xfinvert(&gxf_temp, bv.ginv.as_flattened_mut());
        }
        //
        // Adjust flag for shuffler to know whether to undistort, and clear
        // out the undistorted pieces in memory
        //
        bv.doing_edge_func = false;
        if bv.do_fields {
            clear_shuffle(&mut bv);
        }
        if h.test_mode || h.if_edge_func_only != 0 {
            continue 'sections; // To end of section loop
        }
        //
        // If warping, find out if there is warping on this section and get grid
        bv.sec_has_warp = false;
        if do_warp {
            bv.sec_has_warp = num_control[(ind_gxf - 1) as usize] > 2;
        }
        if bv.sec_has_warp {
            //
            // Get the grid without adjustment of coordinates for any offset, then adjust
            // the grid start for the old starting X coordinate
            if get_size_adjusted_grid(
                ind_gxf - 1,
                new_xtotal_pix as f32 / warp_scale,
                new_ytotal_pix as f32 / warp_scale,
                warp_xoffset,
                warp_yoffset,
                0,
                warp_scale,
                1,
                &mut bv.nx_warp,
                &mut bv.ny_warp,
                &mut bv.x_warp_strt,
                &mut bv.y_warp_strt,
                &mut bv.x_warp_intrv,
                &mut bv.y_warp_intrv,
                &mut bv.warp_dx,
                &mut bv.warp_dy,
                bv.lm_warp_x,
                bv.lm_warp_y,
                &mut h.edge_name,
            ) != 0
            {
                exit_error(&h.edge_name);
            }
            bv.x_warp_strt += ix_unali_start as f32;
            bv.y_warp_strt += iy_unali_start as f32;
            // print *,nxWarp, nyWarp, xWarpStrt, yWarpStrt, xWarpIntrv, yWarpIntrv
            // write(*,'(10f7.2)') ((warpDx(ix, iy), warpDy(ix, iy), ix=1, 10), iy=3, 4)
        }
        //
        // if floating, need to get current input min and max
        // To pad edges properly, need current mean: put it into dmean
        //
        crossvalue(h.x_is_long_dim, nxp, nyp, &mut num_short, &mut num_long);
        cur_in_max = -1.0e10;
        cur_in_min = 1.0e10;
        cur_sum = 0.;
        real_nsum = 0.;
        for ilong in (1..=num_long).rev() {
            for ishort in (1..=num_short).rev() {
                crossvalue(h.x_is_long_dim, ishort, ilong, &mut ix_frame, &mut iy_frame);
                let mp = bv.map_piece[pdx(&bv, ix_frame, iy_frame)];
                if mp > 0 {
                    shuffler(&mut bv, mp, &mut ind_array);
                    for iy in 1..=nyin {
                        tsum = 0.;
                        let start = (ind_array + (iy - 1) * nxin - 1) as usize;
                        for &a in &bv.array[start..start + nxin as usize] {
                            // `blendmont.f90:2134-2135`: `minss curInMin, a` /
                            // `maxss curInMax, a` in the reference object, so a
                            // NaN pixel replaces the running extreme.
                            cur_in_min = minss(cur_in_min, a);
                            cur_in_max = maxss(cur_in_max, a);
                            tsum += a as f64;
                        }
                        cur_sum += tsum;
                        real_nsum += nxin as f64;
                    }
                }
            }
        }
        bv.dmean = (cur_sum / real_nsum) as f32;
        bv.dfill = bv.dmean;
        if use_fill {
            bv.dfill = fill_val;
        }
        if if_float == 0 {
            cur_in_min = dflt_in_min;
            cur_in_max = dflt_in_max;
        }
        //
        // now get output scaling factor and additive factor
        //
        {
            let range = cur_in_max - cur_in_min;
            // `blendmont.f90:2154`: `maxss curInMax - curInMin, 1.`.
            pixel_scale = (out_max - out_min) / maxss(range, 1.);
        }
        pixel_add = out_min - pixel_scale * cur_in_min;
        //
        // look through memory list and renumber them with priorities
        // backwards from the first needed piece
        //
        new_use_count = bv.juse_count - 1;
        for ilong in 1..=num_long {
            for ishort in 1..=num_short {
                crossvalue(h.x_is_long_dim, ishort, ilong, &mut ix_frame, &mut iy_frame);
                let mp = bv.map_piece[pdx(&bv, ix_frame, iy_frame)];
                if mp > 0 {
                    for i in 0..bv.max_load as usize {
                        if bv.iz_mem_list[i] == mp {
                            bv.last_used[i] = new_use_count;
                            new_use_count -= 1;
                        }
                    }
                }
            }
        }
        //
        // UNDISTORTING ONLY: loop on all frames in section, in order as
        // they were in input file
        //
        if undistort_only {
            clear_shuffle(&mut bv);
            bv.doing_edge_func = true;
            for ipc in 1..=bv.npc_list {
                if bv.iz_pc_list[(ipc - 1) as usize] == h.iz_sect {
                    shuffler(&mut bv, ipc, &mut ind_array);
                    tsum = 0.;
                    let base = (ind_array - 1) as usize;
                    for v in &mut bv.array[base..base + bv.npix_in as usize] {
                        val = pixel_scale * *v + pixel_add;
                        *v = val;
                        // `blendmont.f90:2188-2189`: running extreme as
                        // destination.
                        dmin_out = minss(dmin_out, val);
                        dmax_out = maxss(dmax_out, val);
                        tsum += val as f64;
                    }
                    grand_sum += tsum;
                    bv.nxyz_bin[2] += 1;
                    unsafe {
                        iiu_write_section(2, bv.array[base..].as_mut_ptr().cast());
                    }
                    //
                    if let Some(w) = unit3.as_mut() {
                        // `newPcXlowLeft` and `newPcYlowLeft` are never set on
                        // this path; they are 0 as the translation starts them
                        // (uninitialised in the source).
                        let _ = writeln!(
                            w,
                            "{}{}{}",
                            i_edit(new_pc_xlow_left, 9),
                            i_edit(new_pc_ylow_left, 9),
                            i_edit(h.iz_sect, 7)
                        );
                    }
                }
            }
            continue 'sections; // To end of section loop
        }
        //
        // GET THE PIXEL OUT
        // -  loop on output frames; within each frame loop on little boxes
        //
        crossvalue(
            h.x_is_long_dim,
            new_xpieces,
            new_ypieces,
            &mut num_short,
            &mut num_long,
        );
        //
        let nx_out = bv.nxyz_out[0];
        for ilong in 1..=num_long {
            for ishort in 1..=num_short {
                crossvalue(h.x_is_long_dim, ishort, ilong, &mut ix_out, &mut iy_out);
                // write(*,'(a,2i4)') ' composing frame at', ixout, iyout
                new_pc_ylow_left = new_min_ypiece + (iy_out - 1) * (new_yframe - new_yoverlap);
                new_pc_xlow_left = new_min_xpiece + (ix_out - 1) * (new_xframe - new_xoverlap);
                any_pixels = i_binning > 1 || new_xpieces * new_ypieces == 1;
                any_lines_out = false;
                tsum = 0.;
                line_offset = iy_offset;
                lines_buffered = 0;
                i_buffer_base = 0;
                //
                // do fast little boxes
                //
                nx_fast = (new_xframe + (IFAST_SIZ - 1)) / IFAST_SIZ;
                ny_fast = (new_yframe + (IFAST_SIZ - 1)) / IFAST_SIZ;
                if y_chunks {
                    ny_fast = (line_end - line_start + IFAST_SIZ) / IFAST_SIZ;
                }
                //
                // loop on boxes, get lower & upper limits in each box
                //
                for iy_fast in 1..=ny_fast {
                    ind_ylow = new_pc_ylow_left + (iy_fast - 1) * IFAST_SIZ;
                    ind_yhigh = (ind_ylow + IFAST_SIZ).min(new_pc_ylow_left + new_yframe) - 1;
                    if y_chunks {
                        ind_ylow = line_start + (iy_fast - 1) * IFAST_SIZ;
                        ind_yhigh = (ind_ylow + IFAST_SIZ - 1).min(line_end);
                    }
                    num_lines_out = ind_yhigh + 1 - ind_ylow;
                    //
                    // fill array with dfill
                    //
                    {
                        let base = i_buffer_base as usize;
                        let n = (nx_out * num_lines_out).max(0) as usize;
                        bv.brray[base..base + n].fill(bv.dfill);
                    }
                    //
                    for ix_fast in 1..=nx_fast {
                        ind_xlow = new_pc_xlow_left + (ix_fast - 1) * IFAST_SIZ;
                        ind_xhigh = (ind_xlow + IFAST_SIZ).min(new_pc_xlow_left + new_xframe) - 1;
                        //
                        // check # of edges, and prime piece number, for each corner
                        //
                        do_fast = true;
                        in_frame = false;
                        in_one_piece = 0;
                        same_pieces = true;
                        bv.debug = (bv.ix_debug >= ind_xlow && bv.ix_debug <= ind_xhigh)
                            || (bv.iy_debug >= ind_ylow && bv.iy_debug <= ind_yhigh);
                        ind_y = ind_ylow;
                        let step_y = 1.max(ind_yhigh - ind_ylow);
                        let step_x = 1.max(ind_xhigh - ind_xlow);
                        while ind_y <= ind_yhigh {
                            ind_x = ind_xlow;
                            while ind_x <= ind_xhigh {
                                countedges(
                                    &mut bv,
                                    ind_x,
                                    ind_y,
                                    &mut h.xg,
                                    &mut h.ysrc,
                                    h.use_edges,
                                );
                                if bv.num_pieces > 0 {
                                    if in_one_piece == 0 {
                                        in_one_piece = bv.in_piece[1];
                                        num_first = bv.num_pieces;
                                        ip_first.copy_from_slice(&bv.in_piece[1..5]);
                                    }
                                    do_fast = do_fast
                                        && bv.num_pieces == 1
                                        && bv.in_piece[1] == in_one_piece;
                                    same_pieces = same_pieces && bv.num_pieces == num_first;
                                    for i in 1..=bv.num_pieces {
                                        let iu = (i - 1) as usize;
                                        in_frame = in_frame
                                            || (bv.x_in_piece[iu] >= 0.
                                                && bv.x_in_piece[iu] <= nxin as f32 - 1.
                                                && bv.y_in_piece[iu] >= 0.
                                                && bv.y_in_piece[iu] <= nyin as f32 - 1.);
                                        // `ipFirst(4)`: with more than four
                                        // pieces the source reads past it (stack
                                        // residue); here such a piece never matches.
                                        same_pieces = same_pieces
                                            && ip_first.get(iu).copied()
                                                == Some(bv.in_piece[i as usize]);
                                    }
                                } else {
                                    same_pieces = false;
                                }
                                ind_x += step_x;
                            }
                            ind_y += step_y;
                        }
                        if bv.debug {
                            let _ = writeln!(
                                out,
                                "{}{}{}{}{}",
                                ld_int(ind_ylow),
                                ld_int(ind_yhigh),
                                ld_int(bv.num_pieces),
                                ld_real(bv.x_in_piece[0]),
                                ld_real(bv.y_in_piece[0])
                            );
                        }
                        //
                        // ALL ON ONE PIECE: do a fast transform of whole box
                        //
                        wall_start = walltime();
                        if do_fast && in_frame {
                            if bv.debug {
                                let _ = writeln!(
                                    out,
                                    " fast box{}{}{}{}",
                                    ld_int(ind_xlow),
                                    ld_int(ind_xhigh),
                                    ld_int(ind_ylow),
                                    ld_int(ind_yhigh)
                                );
                            }
                            xfunit(&mut fastf, 1.); //start with unit xform
                            //
                            // if doing g xforms, put operations into xform that will
                            // perform inverse of xform
                            //
                            if bv.do_gxforms {
                                xfcopy(bv.ginv.as_flattened(), &mut fastf);
                            }
                            //
                            // shift coordinates down to be within piece
                            //
                            fastf[4] -= bv.ix_pc_list[(in_one_piece - 1) as usize] as f32;
                            fastf[5] -= bv.iy_pc_list[(in_one_piece - 1) as usize] as f32;
                            //
                            // if doing h's, implement the h inverse
                            //
                            if bv.multng {
                                xfmult(&fastf, &bv.hinv[h3!(1, 1, in_one_piece)..], &mut fast_temp);
                                xfcopy(&fast_temp, &mut fastf);
                            }
                            //
                            // now add 1 to get array index
                            //
                            fastf[4] += 1.;
                            fastf[5] += 1.;
                            //
                            shuffler(&mut bv, in_one_piece, &mut ind_array);
                            // `fastInterp(brray(iBufferBase + 1), ..., array(indArray), ...)`:
                            // `fastInterp` reads the module only through `bv`, and
                            // `brray`/`array` are lent out of it for the call.
                            let mut brray = std::mem::take(&mut bv.brray);
                            let array = std::mem::take(&mut bv.array);
                            fast_interp(
                                &bv,
                                &mut brray[i_buffer_base as usize..],
                                nx_out,
                                num_lines_out,
                                &array[(ind_array - 1) as usize..],
                                nxin,
                                nyin,
                                ind_xlow,
                                ind_xhigh,
                                ind_ylow,
                                ind_yhigh,
                                new_pc_xlow_left,
                                &[[fastf[0], fastf[1]], [fastf[2], fastf[3]]],
                                fastf[4],
                                fastf[5],
                                in_one_piece,
                            );
                            bv.brray = brray;
                            bv.array = array;
                            any_pixels = true;
                            fast_cum += walltime() - wall_start;
                            //
                        } else if in_frame {
                            //
                            // in or near an edge: loop on each pixel
                            //
                            h.num_iter = 10;
                            line_base = i_buffer_base + 1 - new_pc_xlow_left;
                            for indy in ind_ylow..=ind_yhigh {
                                for indx in ind_xlow..=ind_xhigh {
                                    // Set debug .or. or .and. here
                                    bv.debug = indx == bv.ix_debug || indy == bv.iy_debug;
                                    if same_pieces {
                                        //
                                        // If it is all the same pieces in this box, then update
                                        // the positions in the pieces
                                        h.xg = indx as f32;
                                        h.ysrc = indy as f32;
                                        if bv.sec_has_warp {
                                            let (xin, yin) = (h.xg - 0.5, h.ysrc - 0.5);
                                            interpolate_grid(
                                                xin,
                                                yin,
                                                &bv.warp_dx,
                                                &bv.warp_dy,
                                                bv.lm_warp_x,
                                                bv.nx_warp,
                                                bv.ny_warp,
                                                bv.x_warp_strt,
                                                bv.y_warp_strt,
                                                bv.x_warp_intrv,
                                                bv.y_warp_intrv,
                                                &mut h.xg,
                                                &mut h.ysrc,
                                            );
                                            h.xg += indx as f32;
                                            h.ysrc += indy as f32;
                                        }
                                        if bv.do_gxforms {
                                            let g = &bv.ginv;
                                            xtmp = g[0][0] * h.xg + g[1][0] * h.ysrc + g[2][0];
                                            h.ysrc = g[0][1] * h.xg + g[1][1] * h.ysrc + g[2][1];
                                            h.xg = xtmp;
                                        }
                                        for i in 1..=bv.num_pieces {
                                            let iu = (i - 1) as usize;
                                            let (mut xp, mut yp) = (0.0f32, 0.0f32);
                                            position_in_piece(
                                                &bv,
                                                h.xg,
                                                h.ysrc,
                                                bv.in_piece[i as usize],
                                                &mut xp,
                                                &mut yp,
                                            );
                                            bv.x_in_piece[iu] = xp;
                                            bv.y_in_piece[iu] = yp;
                                        }
                                    } else {
                                        countedges(
                                            &mut bv,
                                            indx,
                                            indy,
                                            &mut h.xg,
                                            &mut h.ysrc,
                                            h.use_edges,
                                        );
                                    }
                                    if bv.debug {
                                        let mut s = format!(
                                            "{}{}{}{} edges{} pieces",
                                            i_edit(indx, 6),
                                            i_edit(indy, 6),
                                            i_edit(bv.num_edges[0], 7),
                                            i_edit(bv.num_edges[1], 2),
                                            i_edit(bv.num_pieces, 3)
                                        );
                                        for i in 1..=bv.num_pieces {
                                            s.push_str(&i_edit(bv.in_piece[i as usize], 5));
                                        }
                                        let _ = writeln!(out, "{s}");
                                    }
                                    //
                                    // load the edges and compute edge fractions
                                    //
                                    compute_edge_fractions(&mut h, &mut bv, &mut units);
                                    //
                                    // get indices of pieces and edges and the weighting
                                    // of each piece: for now, numbers
                                    get_piece_indices_and_weighting(
                                        &mut bv,
                                        &mut units,
                                        h.use_edges,
                                    );
                                    //
                                    // NOW SORT OUT THE CASES OF 1, 2, 3 or 4 PIECES

                                    get_pixel_from_pieces(&mut h, &mut bv);
                                    //
                                    // stick limited pixval into array and mark as output
                                    //
                                    // `max(curInMin, min(curInMax, pixVal))`
                                    // (`blendmont.f90:2373`): `minss pixVal,
                                    // curInMax` then `maxss ., curInMin`.
                                    bv.brray[(line_base + indx - 1) as usize] =
                                        maxss(minss(h.pix_val, cur_in_max), cur_in_min);
                                    any_pixels = true;
                                }
                                line_base += nx_out;
                            }
                            slow_cum += walltime() - wall_start;
                        }
                    }
                    //
                    // if any pixels have been present in this frame, write line out
                    //
                    if mode_parallel != -2 || y_chunks {
                        num_out = bv.nxyz_bin[2];
                    }
                    if any_pixels {
                        {
                            let base = i_buffer_base as usize;
                            let n = (nx_out * num_lines_out).max(0) as usize;
                            for v in &mut bv.brray[base..base + n] {
                                *v = pixel_scale * *v + pixel_add;
                            }
                        }
                        //
                        // Set line position based on binned pixels output
                        // Set lines to write based on what is in buffer already
                        // Get new number of lines left in buffer; if at top of
                        // frame, increment lines to write if there are any lines left
                        //
                        iline_out = ((iy_fast - 1) * IFAST_SIZ - iy_offset) / i_binning;
                        unsafe { par_wrt_posn(2, num_out, iline_out + iy_out_offset) };
                        iline_out = lines_buffered + num_lines_out;
                        ny_write = (iline_out - line_offset) / i_binning;
                        lines_buffered = (iline_out - line_offset) % i_binning;
                        if ind_yhigh == new_pc_ylow_left + new_yframe - 1 && lines_buffered > 0 {
                            ny_write += 1;
                        }
                        //
                        // write data
                        //
                        iwr_binned(
                            2,
                            &mut bv.brray,
                            &mut bin_line,
                            nx_out,
                            bv.nxyz_bin[0],
                            ix_offset,
                            iline_out,
                            ny_write,
                            line_offset,
                            i_binning,
                            &mut dmin_out,
                            &mut dmax_out,
                            &mut tsum,
                        );
                        //
                        // Set Y offset to zero after first time, set base and
                        // move remaining lines to bottom
                        //
                        line_offset = 0;
                        ifill = (iline_out - lines_buffered) * nx_out;
                        i_buffer_base = nx_out * lines_buffered;
                        for i in 1..=i_buffer_base {
                            bv.brray[(i - 1) as usize] = bv.brray[(i + ifill - 1) as usize];
                        }
                        //
                        // if this is the first time anything is written, and it
                        // wasn't the first set of lines, then need to go back and
                        // fill the lower part of frame with mean values
                        //
                        if !any_lines_out && iy_fast > 1 {
                            val = bv.dfill * pixel_scale + pixel_add;
                            bv.brray[..nx_out as usize].fill(val);
                            unsafe { par_wrt_posn(2, num_out, iy_out_offset) };
                            for _ifill in 1..=(iy_fast - 1) * IFAST_SIZ {
                                unsafe { par_wrt_lin(2, bv.brray.as_mut_ptr().cast()) };
                            }
                            tsum += (val * nx_out as f32 * (iy_fast - 1) as f32 * IFAST_SIZ as f32)
                                as f64;
                        }
                        any_lines_out = true;
                    }
                }
                //
                // if any pixels present, write piece coordinates
                //
                if any_pixels {
                    grand_sum += tsum;
                    // write(*,'(a,i5)') ' wrote new frame #', nzbin
                    bv.nxyz_bin[2] += 1;
                    //
                    if let Some(w) = unit3.as_mut() {
                        let _ = writeln!(
                            w,
                            "{}{}{}",
                            i_edit(new_pc_xlow_left, 9),
                            i_edit(new_pc_ylow_left, 9),
                            i_edit(h.iz_sect, 7)
                        );
                    }
                }
                //
            } // Loop on frames - short dim
        } // Loop on frames - long dim
    } // Loop on sections
    //
    // `close(3)`
    if let Some(mut w) = unit3.take() {
        let _ = w.flush();
    }
    // write(*,'(a,2f12.6)') 'fast box and single pixel times:', fastcum, slowcum
    let _ = (fast_cum, slow_cum);
    if h.if_edge_func_only == 0 && !h.test_mode {
        //
        // If direct parallel, output stats
        pixel_tot = (bv.nxyz_bin[0] as f32 * num_lines_write as f32) * bv.nxyz_bin[2] as f32;
        tmean = (grand_sum / pixel_tot as f64) as f32;
        if mode_parallel == -2 {
            let _ = writeln!(
                out,
                "Min, max, mean, # pixels={}{}{}{}",
                g_edit(dmin_out, 15, 7),
                g_edit(dmax_out, 15, 7),
                g_edit(tmean, 15, 7),
                f_edit(pixel_tot, 15, 0)
            );
        } else {
            //
            // otherwise finalize the header
            //
            iiu_alt_size(2, &bv.nxyz_bin, &nxyzst);
            iiu_alt_sample(2, &bv.nxyz_bin);
            cell[2] = bv.nxyz_bin[2] as f32 * delta[2];
            iiu_alt_cell(2, &cell);
            iiu_write_header(2, &title, -1, dmin_out, dmax_out, tmean);
        }
        if parallel_hdf {
            if unsafe { iiu_par_wrt_flush_buffers(2) } != 0 {
                exit_error("Finishing writing to output HDF file");
            }
            par_wrt_close();
        }
        unsafe { iiu_close(2) };
    }
    if undistort_only {
        exit(0);
    }
    //
    // write edge correlations
    //
    if xc_write_out {
        write_edge_correlations(&mut h, &bv);
    }
    //
    // Write aligned piece coordinates
    if !blank(&ali_coord_file) {
        if ali_coord_file.starts_with('.') {
            ali_coord_file = format!("{root}{}", ali_coord_file.trim_end_matches(' '));
        }
        let mut unit14 =
            BufWriter::new(dopen(14, ali_coord_file.trim_end_matches(' '), "new", "f"));
        for ipc in 1..=bv.npc_list {
            let iu = (ipc - 1) as usize;
            ix_frame = (bv.ix_pc_list[iu] - bv.min_xpiece) / (nxin - bv.n_overlap[0]) + 1;
            iy_frame = (bv.iy_pc_list[iu] - bv.min_ypiece) / (nyin - bv.n_overlap[1]) + 1;
            ix = bv.ix_pc_list[iu] + h.hxf[h3!(1, 3, ipc)].round() as i32;
            iy = bv.iy_pc_list[iu] + h.hxf[h3!(2, 3, ipc)].round() as i32;
            let mut line = format!(
                "{}{}{}{}{}{}{}",
                i_edit(ix, 9),
                i_edit(iy, 9),
                i_edit(bv.iz_pc_list[iu], 6),
                i_edit(ix_frame, 6),
                i_edit(iy_frame, 6),
                i_edit(ix - bv.min_xpiece, 9),
                i_edit(iy - bv.min_ypiece, 9)
            );
            if ipc == 1 {
                line.push_str(&i_edit(nxin, 6));
                line.push_str(&i_edit(nyin, 6));
            }
            let _ = writeln!(unit14, "{line}");
        }
        // `close(14)`
        let _ = unit14.flush();
    }

    //
    // rewrite header for new edge functions so that they have later date
    // than the edge correlations; close files
    //
    for ixy in h.ixy_func_start..=h.ixy_func_end {
        let xyu = (ixy - 1) as usize;
        let yxu = (2 - ixy) as usize;
        if h.if_old_edge == 0
            && let Some(unit) = units.edge[xyu].as_ref()
        {
            let rec: Vec<u8> = [
                bv.nedge[xyu],
                bv.nx_grid[xyu],
                bv.ny_grid[xyu],
                bv.int_grid[xyu],
                bv.int_grid[yxu],
            ]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
            if let Err(err) = unit.write_record(1, &rec) {
                unit.runtime_error(err);
            }
        }
        units.edge[xyu] = None;
    }
    for ixy in 1..=2 {
        if bv.if_dump_xy[(ixy - 1) as usize] > 0 {
            iiu_write_header(2 + ixy, &title, -1, 0., 255., 128.);
            unsafe { iiu_close(2 + ixy) };
        }
    }
    if bv.iz_unsmoothed_patch >= 0 {
        if let Some(mut w) = units.unit10.take() {
            let _ = w.flush();
        }
    }
    if bv.iz_smoothed_patch >= 0 {
        if let Some(mut w) = units.unit11.take() {
            let _ = w.flush();
        }
    }
    //
    let _ = (
        ierr,
        grid_scale,
        max_sampling,
        xtmp,
        nbin_tmp,
        ecd_binning,
        bin_ratio,
        if_revise,
    );
    exit(0);
}

/// Original: `subroutine writeEdgeCorrelations()`, contained in `blendmont`
/// (`blendmont.f90:2529`).
///
/// Writes the full edge correlation file, or the X or Y component only.
pub fn write_edge_correlations(h: &mut Host, bv: &BlendVars) {
    h.edge_name = format!(
        "{}{}",
        h.root_name.trim_end_matches(' '),
        XCORR_EXTENSION[(h.if_edge_func_only % 3) as usize]
    );
    let mut unit4 = BufWriter::new(dopen(4, &h.edge_name, "new", "f"));
    if h.ixy_func_start == 1 {
        let _ = writeln!(
            unit4,
            "{}{}",
            i_edit(bv.nedge[0], 7),
            i_edit(bv.nedge[1], 7)
        );
    }
    for ixy in h.ixy_func_start..=h.ixy_func_end {
        let nedge = bv.nedge[(ixy - 1) as usize];
        if h.num_skipped_edges == 0 && nedge > 0 {
            for i in 1..=nedge {
                let k = e2!(bv, i, ixy);
                let _ = writeln!(
                    unit4,
                    "{}{}",
                    f_edit(h.edge_displace_x[k], 9, 3),
                    f_edit(h.edge_displace_y[k], 10, 3)
                );
            }
        } else if nedge > 0 {
            for i in 1..=nedge {
                let k = e2!(bv, i, ixy);
                let _ = writeln!(
                    unit4,
                    "{}{}{}",
                    f_edit(h.edge_displace_x[k], 9, 3),
                    f_edit(h.edge_displace_y[k], 10, 3),
                    i_edit(bv.if_skip_edge[k], 4)
                );
            }
        }
    }
    // `close(4)`
    let _ = unit4.flush();
}

/// Original: `subroutine findSectionEdgeFunctions()`, contained in
/// `blendmont` (`blendmont.f90:2549`).
///
/// Gets edge functions for a section if they haven't been done yet.  Sets
/// the host's `useEdges`, which the pixel loop reads.
pub fn find_section_edge_functions(h: &mut Host, bv: &mut BlendVars, units: &mut BlendUnits) {
    let mut jedge = 0i32;
    let (mut ixy, mut iyx) = (0i32, 0i32);
    let lim_edge = bv.lim_edge;
    //
    // Check if any edges are to be substituted on this section
    // Do so if any use values are different from original and none are 0
    h.use_edges = false;
    if bv.num_use_edge > 0 || bv.iz_use_def_low >= 0 {
        let mut num_zero = 0;
        for ixy in 1..=2 {
            for iedge in 1..=bv.nedge[(ixy - 1) as usize] {
                if h.iz_sect == bv.iz_pc_list[(bv.ipiece_lower[e2!(bv, iedge, ixy)] - 1) as usize] {
                    find_edge_to_use(bv, iedge, ixy, &mut jedge);
                    if jedge == 0 {
                        num_zero += 1;
                    }
                    if jedge != iedge {
                        h.use_edges = true;
                    }
                }
            }
        }
        if num_zero > 0 {
            h.use_edges = false;
        }
        // print *,'numzero, useedges', numZero, useEdges
    }
    //
    // loop on short then long direction
    //
    for iedge_dir in 1..=2 {
        crossvalue(
            h.x_is_long_dim,
            iedge_dir,
            3 - iedge_dir,
            &mut ixy,
            &mut iyx,
        );
        if iyx == h.if_edge_func_only {
            continue;
        }
        //
        // loop on all edges of that type with pieces in section that are
        // not done yet
        //
        for iedge in 1..=bv.nedge[(ixy - 1) as usize] {
            if h.iz_sect == bv.iz_pc_list[(bv.ipiece_lower[e2!(bv, iedge, ixy)] - 1) as usize] {
                jedge = iedge;
                if h.use_edges {
                    find_edge_to_use(bv, iedge, ixy, &mut jedge);
                }
                // print *,'checking edge', ixy, jedge, ' for edge', ixy, iedge
                if !h.edge_done[e2!(bv, jedge, ixy)] {
                    //
                    // do cross-correlation if the sloppy flag is set and either
                    // we are shifting each piece or the pieces are't on same neg
                    //
                    // print *,'Doing edge ', ixy, jedge
                    h.do_cross = h.if_sloppy != 0
                        && (h.shift_each
                            || (h.any_neg
                                && bv.neg_list
                                    [(bv.ipiece_lower[e2!(bv, jedge, ixy)] - 1) as usize]
                                    != bv.neg_list
                                        [(bv.ipiece_upper[e2!(bv, jedge, ixy)] - 1) as usize]));
                    doedge(
                        bv,
                        units,
                        jedge,
                        ixy,
                        &mut h.edge_done,
                        h.sd_crit,
                        h.dev_crit,
                        &h.num_fit,
                        h.ipoly_order,
                        &h.nskip_regress,
                        h.do_cross,
                        h.xc_read_in,
                        h.xc_legacy,
                        h.use_expected,
                        &mut h.edge_displace_x,
                        &mut h.edge_displace_y,
                        lim_edge,
                    );
                    //
                    // after each one, check memory list to see if there's any
                    // pieces with undone lower edge in orthogonal direction
                    //
                    if !h.use_edges && ixy != h.if_edge_func_only {
                        for imem in 1..=bv.max_load {
                            let ipc = bv.iz_mem_list[(imem - 1) as usize];
                            if ipc > 0 {
                                jedge = bv.iedge_lower[a2!(bv.iedge_lower_ext, ipc, iyx)];
                                if jedge > 0 && !h.edge_done[e2!(bv, jedge, iyx)] {
                                    // print *,'Doing edge ', iyx, jedge
                                    h.do_cross = h.if_sloppy != 0
                                        && (h.shift_each
                                            || (h.any_neg
                                                && bv.neg_list[(bv.ipiece_lower
                                                    [e2!(bv, jedge, iyx)]
                                                    - 1)
                                                    as usize]
                                                    != bv.neg_list[(bv.ipiece_upper
                                                        [e2!(bv, jedge, iyx)]
                                                        - 1)
                                                        as usize]));
                                    doedge(
                                        bv,
                                        units,
                                        jedge,
                                        iyx,
                                        &mut h.edge_done,
                                        h.sd_crit,
                                        h.dev_crit,
                                        &h.num_fit,
                                        h.ipoly_order,
                                        &h.nskip_regress,
                                        h.do_cross,
                                        h.xc_read_in,
                                        h.xc_legacy,
                                        h.use_expected,
                                        &mut h.edge_displace_x,
                                        &mut h.edge_displace_y,
                                        lim_edge,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Original: `subroutine findMultinegTransforms()`, contained in
/// `blendmont` (`blendmont.f90:2628`).
///
/// Finds h transforms if multiple negatives (UNTESTED! in the source).  The
/// negative and joint tables are host variables used only here, so they are
/// locals.  Two defects of the source are fixed in translation (`BUGS.md`):
/// the iteration count `ishift` is never incremented there, so the loop runs
/// until `errAdjust <= errLim`; and the final recentring subtracts the mean X
/// shift from the *angle* (`hCum(1, i)`) and the mean Y shift from the X
/// shift (`hCum(2, i)`).  Here the loop stops after `numShiftNeg` iterations
/// and the shifts `hCum(2:3, i)` are recentred.
pub fn find_multineg_transforms(h: &mut Host, bv: &mut BlendVars, units: &mut BlendUnits) {
    let limneg = LIMNEG as usize;
    let mut min_neg_x = [0i32; LIMNEG as usize];
    let mut max_neg_x = [0i32; LIMNEG as usize];
    let mut min_neg_y = [0i32; LIMNEG as usize];
    let mut max_neg_y = [0i32; LIMNEG as usize];
    // `(LIMNEG, 2)` arrays, flat column-major.
    let mut joint_lower = [0i32; 2 * LIMNEG as usize];
    let mut joint_upper = [0i32; 2 * LIMNEG as usize];
    let mut neg_lower = [0i32; 2 * LIMNEG as usize];
    let mut neg_upper = [0i32; 2 * LIMNEG as usize];
    let mut neg_index = [0i32; LIMNEG as usize];
    let mut num_in_h_adj = [0i32; LIMNEG as usize];
    let mut num_joint = [0i32; 2];
    let mut num_edge_on_joint = [0i32; 15];
    // `listOnJoint(15, LIMNEG)`, flat column-major.
    let mut list_on_joint = [0i32; 15 * LIMNEG as usize];
    let mut ix_pc_lower = [0i32; 15];
    let mut ix_pc_upper = [0i32; 15];
    let mut iy_pc_lower = [0i32; 15];
    let mut iy_pc_upper = [0i32; 15];
    // `hCum(6, LIMNEG)`, `hAdj(6, LIMNEG)`, `r(6, LIMNEG, 2)`, flat.
    let mut h_cum = [0.0f32; 6 * LIMNEG as usize];
    let mut h_adj = [0.0f32; 6 * LIMNEG as usize];
    let mut r = [0.0f32; 6 * 2 * LIMNEG as usize];
    let mut h_frame = [0.0f32; 6];
    let mut rotx_net = [0.0f32; 6];
    let (mut ind_low, mut ind_upper) = (0i32, 0i32);
    let (mut ipc_lower, mut ipc_upper) = (0i32, 0i32);
    let jn = |i: i32, ixy: i32| (i - 1) as usize + limneg * (ixy - 1) as usize;
    let rr = |joint: i32, ixy: i32| 6 * ((joint - 1) as usize + limneg * (ixy - 1) as usize);
    let hc = |i: i32| 6 * (i - 1) as usize;
    let mut out = ImodFile::Stdout;
    //
    // first make index of negatives and find coordinates of each
    //
    let mut num_negatives = 0i32;
    for i in 0..limneg {
        max_neg_x[i] = -1000000;
        min_neg_x[i] = 1000000;
        max_neg_y[i] = -1000000;
        min_neg_y[i] = 1000000;
    }
    for ipc in 1..=bv.npc_list {
        let iu = (ipc - 1) as usize;
        if bv.iz_pc_list[iu] == h.iz_sect {
            let mut iwhich = 0;
            for i in 1..=num_negatives {
                if bv.neg_list[iu] == neg_index[(i - 1) as usize] {
                    iwhich = i;
                }
            }
            if iwhich == 0 {
                num_negatives += 1;
                neg_index[(num_negatives - 1) as usize] = bv.neg_list[iu];
                iwhich = num_negatives;
            }
            // point max coordinate 1 past end: it'll be easier
            let w = (iwhich - 1) as usize;
            min_neg_x[w] = min_neg_x[w].min(bv.ix_pc_list[iu]);
            max_neg_x[w] = max_neg_x[w].max(bv.ix_pc_list[iu] + bv.nxyz_in[0]);
            min_neg_y[w] = min_neg_y[w].min(bv.iy_pc_list[iu]);
            max_neg_y[w] = max_neg_y[w].max(bv.iy_pc_list[iu] + bv.nxyz_in[1]);
        }
    }
    //
    // look at all edges, make tables of joints between negs
    //
    for ixy in 1..=2 {
        let xyu = (ixy - 1) as usize;
        num_joint[xyu] = 0;
        for i in 1..=num_negatives {
            joint_upper[jn(i, ixy)] = 0;
            joint_lower[jn(i, ixy)] = 0;
        }
        //
        for iedge in 1..=bv.nedge[xyu] {
            let neg_low = bv.neg_list[(bv.ipiece_lower[e2!(bv, iedge, ixy)] - 1) as usize];
            let neg_up = bv.neg_list[(bv.ipiece_upper[e2!(bv, iedge, ixy)] - 1) as usize];
            if bv.iz_pc_list[(bv.ipiece_lower[e2!(bv, iedge, ixy)] - 1) as usize] == h.iz_sect
                && neg_low != neg_up
            {
                // convert neg #'s to neg indexes
                for i in 1..=num_negatives {
                    if neg_index[(i - 1) as usize] == neg_low {
                        ind_low = i;
                    }
                    if neg_index[(i - 1) as usize] == neg_up {
                        ind_upper = i;
                    }
                }
                // see if joint already on list
                let mut joint = 0;
                for j in 1..=num_joint[xyu] {
                    if neg_lower[jn(j, ixy)] == ind_low && neg_upper[jn(j, ixy)] == ind_upper {
                        joint = j;
                    }
                }
                // if not, add to list
                if joint == 0 {
                    num_joint[xyu] += 1;
                    joint = num_joint[xyu];
                    neg_lower[jn(joint, ixy)] = ind_low;
                    neg_upper[jn(joint, ixy)] = ind_upper;
                    // point the negatives to the joint
                    joint_lower[jn(ind_low, ixy)] = joint;
                    joint_upper[jn(ind_upper, ixy)] = joint;
                    num_edge_on_joint[(joint - 1) as usize] = 0;
                }
                // add edge to list of ones on that joint
                num_edge_on_joint[(joint - 1) as usize] += 1;
                list_on_joint[(num_edge_on_joint[(joint - 1) as usize] - 1) as usize
                    + 15 * (joint - 1) as usize] = iedge;
            }
        }
        //
        // now process all edges on each joint to get a rotrans
        //
        for joint in 1..=num_joint[xyu] {
            for ied in 1..=num_edge_on_joint[(joint - 1) as usize] {
                let iedge = list_on_joint[(ied - 1) as usize + 15 * (joint - 1) as usize];
                read_edge_func(bv, units, iedge, ixy, ied);

                let d = (ied - 1) as usize;
                ipc_lower = bv.ipiece_lower[e2!(bv, iedge, ixy)];
                ix_pc_lower[d] = bv.ix_grd_st_bf[d] + bv.ix_pc_list[(ipc_lower - 1) as usize];
                iy_pc_lower[d] = bv.iy_grd_st_bf[d] + bv.iy_pc_list[(ipc_lower - 1) as usize];
                ipc_upper = bv.ipiece_upper[e2!(bv, iedge, ixy)];
                ix_pc_upper[d] = bv.ix_ofs_bf[d] + bv.ix_pc_list[(ipc_upper - 1) as usize];
                iy_pc_upper[d] = bv.iy_ofs_bf[d] + bv.iy_pc_list[(ipc_upper - 1) as usize];
            }
            let k = rr(joint, ixy);
            joint_to_rotrans(
                &bv.dx_gr_bf,
                &bv.dy_gr_bf,
                bv.ixg_dim,
                bv.iyg_dim,
                &bv.nx_gr_bf,
                &bv.ny_gr_bf,
                bv.int_grid[xyu],
                bv.int_grid[(2 - ixy) as usize],
                &ix_pc_lower,
                &iy_pc_lower,
                &ix_pc_upper,
                &iy_pc_upper,
                num_edge_on_joint[(joint - 1) as usize],
                &mut r[k..k + 6],
            );
            let _ = writeln!(
                out,
                " {} joint, negatives{}{}   theta ={}   dx, dy ={}{}",
                (ixy as u8 + b'W') as char,
                i_edit(bv.neg_list[(ipc_lower - 1) as usize], 4),
                i_edit(bv.neg_list[(ipc_upper - 1) as usize], 4),
                f_edit(r[k], 6, 2),
                f_edit(r[k + 1], 6, 1),
                f_edit(r[k + 2], 6, 1)
            );
        }
    }
    //
    // Resolve the edges into rotrans centered on each negative
    // Initially set the rotrans for each negative to null
    //
    let mut max_swing = 0i32;
    for i in 1..=num_negatives {
        let (k, u) = (hc(i), (i - 1) as usize);
        h_cum[k] = 0.;
        h_cum[k + 1] = 0.;
        h_cum[k + 2] = 0.;
        h_cum[k + 3] = 0.5 * (max_neg_x[u] + min_neg_x[u]) as f32;
        h_cum[k + 4] = 0.5 * (max_neg_y[u] + min_neg_y[u]) as f32;
        max_swing = max_swing
            .max((max_neg_x[u] - min_neg_x[u]) / 2)
            .max((max_neg_y[u] - min_neg_y[u]) / 2);
    }
    //
    // start loop - who knows how long this will take
    //
    // Fixed in translation (`BUGS.md`): the source never increments `ishift`
    // (`blendmont.f90:2743-2796`), so the `numShiftNeg` limit never applies
    // and a non-converging adjustment loops forever.  It counts iterations
    // here.
    let mut ishift = 1;
    let mut err_adjust = 1.0e10f32;
    let num_shift_neg = 100;
    let err_lim = 0.1 * num_negatives as f32;
    let (mut dx_sum, mut dy_sum) = (0.0f32, 0.0f32);
    while ishift <= num_shift_neg && err_adjust > err_lim {
        //
        // set sum of adjustment h's to null
        //
        err_adjust = 0.;
        for i in 1..=num_negatives {
            let k = hc(i);
            let src = h_cum;
            lincom_rotrans(&src[k..k + 6], 0., &src[k..k + 6], 0., &mut h_adj[k..k + 6]);
            num_in_h_adj[(i - 1) as usize] = 0;
        }
        //
        // loop on joints, adding up net adjustments still needed
        //
        for ixy in 1..=2 {
            for joint in 1..=num_joint[(ixy - 1) as usize] {
                //
                // get net rotrans still needed at this joint: subtract upper
                // h and add lower h to joint rotrans
                //
                let neg_up = neg_upper[jn(joint, ixy)];
                let neg_low = neg_lower[jn(joint, ixy)];
                let k = rr(joint, ixy);
                lincom_rotrans(
                    &h_cum[hc(neg_up)..hc(neg_up) + 6],
                    -1.,
                    &r[k..k + 6],
                    -1.,
                    &mut rotx_net,
                );
                // The output aliases the second input: pass a copy of it
                // (`bsubs.rs` module note).
                let copy = rotx_net;
                lincom_rotrans(
                    &h_cum[hc(neg_low)..hc(neg_low) + 6],
                    1.,
                    &copy,
                    1.,
                    &mut rotx_net,
                );
                //
                // now add half of the net needed to the upper adjustment
                // subtract half from the lower adjustment
                //
                let ku = hc(neg_up);
                let copy: [f32; 6] = h_adj[ku..ku + 6].try_into().unwrap();
                lincom_rotrans(&rotx_net, 0.5, &copy, 1., &mut h_adj[ku..ku + 6]);
                num_in_h_adj[(neg_up - 1) as usize] += 1;
                let kl = hc(neg_low);
                let copy: [f32; 6] = h_adj[kl..kl + 6].try_into().unwrap();
                lincom_rotrans(&rotx_net, -0.5, &copy, 1., &mut h_adj[kl..kl + 6]);
                num_in_h_adj[(neg_low - 1) as usize] += 1;
            }
        }
        //
        // get the average adjustment, adjust hcum with it
        //
        dx_sum = 0.;
        dy_sum = 0.;
        for i in 1..=num_negatives {
            let (k, u) = (hc(i), (i - 1) as usize);
            err_adjust +=
                (h_adj[k + 1].abs() + h_adj[k + 2].abs() + h_adj[k].abs() * max_swing as f32)
                    / num_in_h_adj[u] as f32;
            let copy: [f32; 6] = h_cum[k..k + 6].try_into().unwrap();
            lincom_rotrans(
                &h_adj[k..k + 6],
                1. / num_in_h_adj[u] as f32,
                &copy,
                1.,
                &mut h_cum[k..k + 6],
            );
            dx_sum += h_cum[k + 1];
            dy_sum += h_cum[k + 2];
        }
        ishift += 1;
        //
    } //end of cycle
    //
    // shift all dx and dy to have a mean of zero
    //
    // Fixed in translation (`BUGS.md`): `blendmont.f90:2800-2803` subtracts
    // the mean X shift from `hCum(1, i)` (the angle) and the mean Y shift
    // from `hCum(2, i)` (the X shift); the sums were taken over `hCum(2, i)`
    // and `hCum(3, i)`, which are recentred here.
    for i in 1..=num_negatives {
        let k = hc(i);
        h_cum[k + 1] -= dx_sum / num_negatives as f32;
        h_cum[k + 2] -= dy_sum / num_negatives as f32;
    }
    //
    // compute the h function (and hinv) centered on corner of frame
    //
    for ipc in 1..=bv.npc_list {
        let iu = (ipc - 1) as usize;
        if bv.iz_pc_list[iu] == h.iz_sect {
            for i in 1..=num_negatives {
                if neg_index[(i - 1) as usize] == bv.neg_list[iu] {
                    let k = hc(i);
                    recen_rotrans(
                        &h_cum[k..k + 6],
                        bv.ix_pc_list[iu] as f32,
                        bv.iy_pc_list[iu] as f32,
                        &mut h_frame,
                    );
                    h.hxf[h3!(1, 1, ipc)] = gfortran_cosd_r4(h_frame[0]);
                    h.hxf[h3!(2, 1, ipc)] = gfortran_sind_r4(h_frame[0]);
                    h.hxf[h3!(1, 2, ipc)] = -h.hxf[h3!(2, 1, ipc)];
                    h.hxf[h3!(2, 2, ipc)] = h.hxf[h3!(1, 1, ipc)];
                    h.hxf[h3!(1, 3, ipc)] = h_frame[1];
                    h.hxf[h3!(2, 3, ipc)] = h_frame[2];
                    let off = h3!(1, 1, ipc);
                    xfinvert(&h.hxf[off..off + 6], &mut bv.hinv[off..off + 6]);
                }
            }
        }
    }
    let _ = (joint_lower, joint_upper);
}

/// Original: `subroutine getBestPieceShifts()`, contained in `blendmont`
/// (`blendmont.f90:2830`).
///
/// Finds the shifts of each piece that best align the pieces.  The host
/// variables it alone uses (`xDisplace`, `beforeMean`, `afterMean`, ...) are
/// locals; `iedge` and `ixy` are the host's, for `redoEdgeFunction` and
/// `computeDxyGridMean`.
pub fn get_best_piece_shifts(h: &mut Host, bv: &mut BlendVars, units: &mut BlendUnits) {
    let mut max_sd_median = 0.0f32;
    let (mut x_displace, mut y_displace) = (0.0f32, 0.0f32);
    let (mut ind_low, mut ind_upper) = (0i32, 0i32);
    let mut del_indent = [0.0f32; 2];
    let mut num_best_edge = 0i32;
    let (mut before_mean, mut before_max) = (0.0f32, 0.0f32);
    let mut after_mean = [0.0f32; 2];
    let mut after_max = [0.0f32; 2];
    let mut ind_best = 0usize;
    let (mut dmag_new, mut drot_new) = (0.0f32, 0.0f32);
    let lim_edge = bv.lim_edge;
    let mut out = ImodFile::Stdout;
    //
    // Analyze max SDs for low outliers and set up to skip those edges
    if bv.num_max_sds > 5 {
        rs_median(
            &bv.trimmed_max_sds,
            bv.num_max_sds,
            &mut h.max_sd_temp,
            &mut max_sd_median,
        );
        rs_mad_median_outliers(
            &bv.trimmed_max_sds,
            bv.num_max_sds,
            2.24,
            &mut h.max_sd_temp,
        );
        for ix in 1..=bv.num_max_sds {
            let iu = (ix - 1) as usize;
            if h.max_sd_temp[iu] < 0. && bv.trimmed_max_sds[iu] < max_sd_median / 4. {
                let k = e2!(bv, bv.max_sd_to_edge_num[iu], bv.max_sd_to_ixy_of_edge[iu]);
                bv.if_skip_edge[k] = 2;
                h.num_skipped_edges += 1;
                //print *,'skipping ',trimmedMaxSDs(ix), maxSDtoEdgeNum(ix), maxSDtoIXYofEdge(ix)
            }
        }
    }
    //
    // Set the multineg flag to indicate that there are h transforms
    bv.multng = true;
    for ixy in 1..=2 {
        h.ixy = ixy;
        let mut edge_disp_mean = 0.0f32;
        let nedge = bv.nedge[(ixy - 1) as usize];
        for iedge in 1..=nedge {
            h.iedge = iedge;
            let k = e2!(bv, iedge, ixy);
            if bv.iz_pc_list[(bv.ipiece_lower[k] - 1) as usize] == h.iz_sect {
                //
                // need displacements implied by edges, unless this is to be
                // done by old cross-correlation only
                //
                if !h.xc_legacy {
                    compute_dxy_grid_mean(h, bv, units);
                }
                //
                if !h.from_edge && !h.xc_read_in && !(h.if_sloppy == 1 && h.if_old_edge == 0) {
                    //
                    // If the edges of the image are lousy, it's better to use
                    // correlation, so here is this option.  Compute the
                    // correlations unless doing this by edges only,
                    // if they aren't already available
                    //
                    if bv.if_skip_edge[k] > 0 {
                        x_displace = 0.;
                        y_displace = 0.;
                    } else {
                        shuffler(bv, bv.ipiece_lower[k], &mut ind_low);
                        shuffler(bv, bv.ipiece_upper[k], &mut ind_upper);

                        get_extra_indents(
                            bv,
                            bv.ipiece_lower[k],
                            bv.ipiece_upper[k],
                            ixy,
                            &mut del_indent,
                        );
                        let mut indent_xc = 0;
                        if del_indent[(ixy - 1) as usize] > 0. && bv.ifill_treatment == 1 {
                            indent_xc = del_indent[(ixy - 1) as usize] as i32 + 1;
                        }
                        // `xcorrEdge(array(indLow), array(indUpper), ...)`:
                        // `xcorrEdge` does not touch `array` through the module,
                        // so it is lent out of it for the call.
                        let array = std::mem::take(&mut bv.array);
                        xcorr_edge(
                            bv,
                            &array,
                            ind_low,
                            ind_upper,
                            ixy,
                            &mut x_displace,
                            &mut y_displace,
                            h.xc_legacy,
                            h.use_expected,
                            indent_xc,
                        );
                        bv.array = array;
                    }
                    h.edge_displace_x[k] = x_displace;
                    h.edge_displace_y[k] = y_displace;
                }
                // write(*,'(1x,a,2i4,a,2f8.2,a,2f8.2)') &
                // char(ixy+ichar('W')) //' edge, pieces' &
                // , ipiecelower(iedge, ixy), ipieceupper(iedge, ixy), &
                // '  dxygridmean:', dxgridmean(iedge, ixy), &
                // dygridmean(iedge, ixy), '  xcorr:', -xdisp, -ydisp
                // dxgridmean(iedge, ixy) =-xdisp
                // dygridmean(iedge, ixy) =-ydisp
                // endif
            }
            if ixy == 1 {
                edge_disp_mean += h.edge_displace_x[k] / nedge as f32;
            }
            if ixy == 2 {
                edge_disp_mean += h.edge_displace_y[k] / nedge as f32;
            }
        }
        if (ixy == 1 && (edge_disp_mean * bv.nx_pieces as f32).abs() > (bv.nxyz_in[0] * 3) as f32)
            || (ixy == 2
                && (edge_disp_mean * bv.ny_pieces as f32).abs() > (bv.nxyz_in[1] * 3) as f32)
        {
            let _ = writeln!(
                out,
                "\nWARNING: mean edge shift of{} in {} may give artifacts; consider adjusting overlaps with edpiecepoint and -overlap option",
                f_edit(edge_disp_mean, 8, 0),
                (ixy as u8 + b'W') as char
            );
        }
    }
    //
    // If there is only one piece in one direction, then do it from
    // correlation only unless directed to use the edge, because the error
    // is zero in either case and it is impossible to tell which is better
    let from_corr_only = h.xc_legacy || (!h.from_edge && (bv.nx_pieces == 1 || bv.ny_pieces == 1));
    //
    if !h.from_edge {
        let (am, ax) = (&mut after_mean[0], &mut after_max[0]);
        find_best_shifts(
            bv,
            &mut h.edge_displace_x,
            &mut h.edge_displace_y,
            lim_edge,
            -1,
            h.iz_sect,
            &mut h.hxf,
            &mut num_best_edge,
            &mut before_mean,
            &mut before_max,
            am,
            ax,
            bv.num_xcorr_peaks > 1,
        );
        ind_best = 1;
        //
        // Redo the edge functions were an alternative shift was substituted
        if bv.num_xcorr_peaks > 1 && bv.num_alt_fixed > 0 {
            for ifix in 1..=bv.num_alt_fixed {
                let fixed = bv.iedge_alt_fixed[(ifix - 1) as usize];
                h.ixy = fixed / lim_edge + 1;
                h.iedge = fixed + 1 - (h.ixy - 1) * lim_edge;
                redo_edge_function(h, bv, units);
            }
        }
    }
    if !from_corr_only {
        let (am, ax) = (&mut after_mean[1], &mut after_max[1]);
        find_best_shifts(
            bv,
            &mut h.dx_grid_mean,
            &mut h.dy_grid_mean,
            lim_edge,
            1,
            h.iz_sect,
            &mut h.hxf,
            &mut num_best_edge,
            &mut before_mean,
            &mut before_max,
            am,
            ax,
            false,
        );
        ind_best = 2;
    }
    //
    // if first one was better based upon mean, redo it and reset the
    // index to 1
    //
    if !(from_corr_only || h.from_edge) && after_mean[0] < after_mean[1] {
        let (am, ax) = (&mut after_mean[0], &mut after_max[0]);
        find_best_shifts(
            bv,
            &mut h.edge_displace_x,
            &mut h.edge_displace_y,
            lim_edge,
            -1,
            h.iz_sect,
            &mut h.hxf,
            &mut num_best_edge,
            &mut before_mean,
            &mut before_max,
            am,
            ax,
            false,
        );
        ind_best = 1;
    }

    // Now fix the edges that have low weights. replacing shifts and redoing edge function
    for ifix in 1..=bv.num_low_weight {
        let low = bv.iedge_low_weight[(ifix - 1) as usize];
        h.ixy = low / lim_edge + 1;
        h.iedge = low - (h.ixy - 1) * lim_edge;
        let k = e2!(bv, h.iedge, h.ixy);
        let ipc_upper = bv.ipiece_upper[k];
        let ipc_lower = bv.ipiece_lower[k];
        let dx2 = h.hxf[h3!(1, 3, ipc_upper)] - h.hxf[h3!(1, 3, ipc_lower)];
        let dy2 = h.hxf[h3!(2, 3, ipc_upper)] - h.hxf[h3!(2, 3, ipc_lower)];
        //
        // Do it only if the difference in positions is bigger than the mean error
        let ex = h.edge_displace_x[k] - dx2;
        let ey = h.edge_displace_y[k] - dy2;
        // `ind_best` is 1 or 2 here (see the note on `indBest` below).
        if (ex * ex + ey * ey).sqrt() > after_mean[ind_best.max(1) - 1] {
            h.edge_displace_x[k] = dx2;
            h.edge_displace_y[k] = dy2;
            redo_edge_function(h, bv, units);
        }
    }

    // `indBest` is always set: `fromEdge` implies `.not. fromCorrOnly`,
    // because ShiftFromEdges with ShiftFromXcorrs is refused earlier.
    let ib = ind_best.max(1) - 1;
    if num_best_edge > 0 {
        let _ = writeln!(
            out,
            "{} edges, mean&max error before:{}{}, after by {}{}{}",
            i_edit(num_best_edge, 5),
            f_edit(before_mean, 7, 1),
            f_edit(before_max, 7, 1),
            EDGE_XCORR_TEXT[ib],
            f_edit(after_mean[ib], 7, 2),
            f_edit(after_max[ib], 7, 2)
        );
    }
    if h.test_mode {
        // `dmagPerUm(min(ilistz, numMagGrad))`: with no gradients this is
        // element 0, before the array; the source reads the 4 bytes in front
        // of the allocation (the high half of the malloc size word, 0 for
        // these sizes).  Fixed in translation (`BUGS.md`): with no gradients
        // the gradient is defined as 0.
        let ig = bv.ilistz.min(bv.num_mag_grad);
        let dmag = if ig >= 1 {
            bv.dmag_per_um[(ig - 1) as usize]
        } else {
            0.
        };
        let drot = if ig >= 1 {
            bv.rot_per_um[(ig - 1) as usize]
        } else {
            0.
        };
        let _ = writeln!(
            out,
            " section:{}  gradient:{}{}  mean, max error:{}{}",
            i_edit(h.iz_sect, 4),
            f_edit(dmag, 8, 3),
            f_edit(drot, 8, 3),
            f_edit(after_mean[ib], 9, 4),
            f_edit(after_max[ib], 9, 4)
        );
        if ind_best == 1 {
            find_best_gradient(
                bv,
                &h.edge_displace_x,
                &h.edge_displace_y,
                lim_edge,
                -1,
                h.iz_sect,
                &mut dmag_new,
                &mut drot_new,
            );
        } else {
            find_best_gradient(
                bv,
                &h.dx_grid_mean,
                &h.dy_grid_mean,
                lim_edge,
                1,
                h.iz_sect,
                &mut dmag_new,
                &mut drot_new,
            );
        }
        let _ = writeln!(
            out,
            " Total gradient implied by displacements:{}{}",
            f_edit(dmag + dmag_new, 9, 4),
            f_edit(drot + drot_new, 9, 4)
        );
    }
}

/// Original: `subroutine redoEdgeFunction()`, contained in `blendmont`
/// (`blendmont.f90:2988`).
///
/// Recomputes an edge function because the starting shift has changed.
/// Reads the host's `iedge`, `ixy` and `doCross` (the value the last
/// `findSectionEdgeFunctions` left there).
pub fn redo_edge_function(h: &mut Host, bv: &mut BlendVars, units: &mut BlendUnits) {
    let lim_edge = bv.lim_edge;
    doedge(
        bv,
        units,
        h.iedge,
        h.ixy,
        &mut h.edge_done,
        h.sd_crit,
        h.dev_crit,
        &h.num_fit,
        h.ipoly_order,
        &h.nskip_regress,
        h.do_cross,
        true,
        h.xc_legacy,
        h.use_expected,
        &mut h.edge_displace_x,
        &mut h.edge_displace_y,
        lim_edge,
    );
    let k = a2!(bv.ibuf_edge_ext, h.iedge, h.ixy);
    h.inde = bv.ibuf_edge[k];
    if h.inde > 0 {
        bv.ibuf_edge[k] = 0;
        bv.las_edg_use[(h.inde - 1) as usize] = 0;
    }
    compute_dxy_grid_mean(h, bv, units);
}

/// Original: `subroutine computeDxyGridMean()`, contained in `blendmont`
/// (`blendmont.f90:3004`).
///
/// Computes the mean displacement of the edge function grid for the host's
/// `iedge`, `ixy`.
pub fn compute_dxy_grid_mean(h: &mut Host, bv: &mut BlendVars, units: &mut BlendUnits) {
    let (iedge, ixy) = (h.iedge, h.ixy);
    edgeswap(bv, units, iedge, ixy, &mut h.inde);
    let inde = h.inde;
    let ib = (inde - 1) as usize;
    //
    // compute the mean current displacement of upper relative
    // to lower implied by the d[xy]mean
    //
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for ix in 1..=bv.nx_gr_bf[ib] {
        for iy in 1..=bv.ny_gr_bf[ib] {
            let k = (ix - 1) as usize
                + bv.dx_gr_bf_ext[0] * ((iy - 1) as usize + bv.dx_gr_bf_ext[1] * ib);
            sum_x += bv.dx_gr_bf[k];
            sum_y += bv.dy_gr_bf[k];
        }
    }
    let k = e2!(bv, iedge, ixy);
    let npts = (bv.nx_gr_bf[ib] * bv.ny_gr_bf[ib]) as f32;
    h.dx_grid_mean[k] = sum_x / npts;
    h.dy_grid_mean[k] = sum_y / npts;
    //
    // adjust these by the current displacements implied by
    // starting and offset coordinates of the edge areas
    //
    // Each `+`/`-` with an integer operand is a real operation, left to right.
    if ixy == 1 {
        h.dx_grid_mean[k] = h.dx_grid_mean[k] + bv.ix_ofs_bf[ib] as f32
            - bv.ix_grd_st_bf[ib] as f32
            + bv.nxyz_in[0] as f32
            - bv.n_overlap[0] as f32;
        h.dy_grid_mean[k] =
            h.dy_grid_mean[k] + bv.iy_ofs_bf[ib] as f32 - bv.iy_grd_st_bf[ib] as f32;
    } else {
        h.dx_grid_mean[k] =
            h.dx_grid_mean[k] + bv.ix_ofs_bf[ib] as f32 - bv.ix_grd_st_bf[ib] as f32;
        h.dy_grid_mean[k] = h.dy_grid_mean[k] + bv.iy_ofs_bf[ib] as f32
            - bv.iy_grd_st_bf[ib] as f32
            + bv.nxyz_in[1] as f32
            - bv.n_overlap[1] as f32;
    }
    //
    // But if this edge is skipped for edge functions and included
    // for finding shifts, substitute the (read-in) correlation shift
    if bv.if_skip_edge[k] == 1 {
        h.dx_grid_mean[k] = -h.edge_displace_x[k];
        h.dy_grid_mean[k] = -h.edge_displace_y[k];
    }
    //print *,'edge mean', iedge, ixy, dxGridMean(iedge, ixy), dyGridMean(iedge, ixy)
}

/// Original: `subroutine computeEdgeFractions()`, contained in `blendmont`
/// (`blendmont.f90:3049`).
///
/// Loads the edges and computes edge fractions for a pixel.  Sets the host's
/// `numEdgesIn` and `active4` for `getPixelFromPieces`.
pub fn compute_edge_fractions(h: &mut Host, bv: &mut BlendVars, units: &mut BlendUnits) {
    let mut out = ImodFile::Stdout;
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let intg = bv.int_grid[0];
    h.num_edges_in = 0;
    for ixy in 1..=2 {
        let xyu = (ixy - 1) as usize;
        for ied in 1..=bv.num_edges[xyu] {
            let du = (ied - 1) as usize;
            let iedge = bv.in_edge[xyu][du];
            edgeswap(bv, units, iedge, ixy, &mut h.ind_edge);
            let ind_edge = h.ind_edge;
            let ie = (ind_edge - 1) as usize;
            let ind_lower = bv.in_ed_lower[xyu][du];
            bv.ind_edge4[xyu][du] = ind_edge;
            if h.edges_separated {
                if ixy == 1 {
                    let lim = (bv.nx_gr_bf[ie] * intg).min(bv.iblend[0]);
                    bv.edge_frac4[xyu][du] = 0.5
                        + (bv.x_in_piece[(ind_lower - 1) as usize]
                            - (bv.ix_grd_st_bf[ie] as f32
                                + ((bv.nx_gr_bf[ie] - 1) * intg) as f32 / 2.))
                            / lim as f32;
                } else {
                    let lim = (bv.ny_gr_bf[ie] * intg).min(bv.iblend[1]);
                    bv.edge_frac4[xyu][du] = 0.5
                        + (bv.y_in_piece[(ind_lower - 1) as usize]
                            - (bv.iy_grd_st_bf[ie] as f32
                                + ((bv.ny_gr_bf[ie] - 1) * intg) as f32 / 2.))
                            / lim as f32;
                }
            } else {
                let (bw_offset, edge_start, edge_end);
                if ixy == 1 {
                    let bwo = (0.max(bv.nx_gr_bf[ie] * intg - bv.iblend[0]) / 2) as f32;
                    let a = 0.55f32 * nxin as f32;
                    let b = bv.ix_grd_st_bf[ie] as f32 - intg as f32 / 2. + bwo;
                    edge_start = if a > b { a } else { b };
                    let a = 0.45f32 * nxin as f32;
                    let b = bv.ix_ofs_bf[ie] as f32 + (bv.nx_gr_bf[ie] as f32 - 0.5) * intg as f32
                        - bwo;
                    edge_end = (if a < b { a } else { b }) + bv.ix_grd_st_bf[ie] as f32
                        - bv.ix_ofs_bf[ie] as f32;
                    bw_offset = (bv.x_in_piece[(ind_lower - 1) as usize] - edge_start)
                        / (edge_end - edge_start);
                } else {
                    let bwo = (0.max(bv.ny_gr_bf[ie] * intg - bv.iblend[1]) / 2) as f32;
                    let a = 0.55f32 * nyin as f32;
                    let b = bv.iy_grd_st_bf[ie] as f32 - intg as f32 / 2. + bwo;
                    edge_start = if a > b { a } else { b };
                    let a = 0.45f32 * nyin as f32;
                    let b = bv.iy_ofs_bf[ie] as f32 + (bv.ny_gr_bf[ie] as f32 - 0.5) * intg as f32
                        - bwo;
                    edge_end = (if a < b { a } else { b }) + bv.iy_grd_st_bf[ie] as f32
                        - bv.iy_ofs_bf[ie] as f32;
                    bw_offset = (bv.y_in_piece[(ind_lower - 1) as usize] - edge_start)
                        / (edge_end - edge_start);
                }
                if bv.debug {
                    let _ = writeln!(
                        out,
                        "{}{}{}{}",
                        ld_int(ied),
                        ld_int(ixy),
                        ld_real(bv.edge_frac4[xyu][du]),
                        ld_real(bw_offset)
                    );
                }
                let _ = (edge_start, edge_end);
                bv.edge_frac4[xyu][du] = bw_offset;
            }
            h.active4[xyu][du] = bv.edge_frac4[xyu][du] < 0.999 && bv.edge_frac4[xyu][du] > 0.001;
            if h.active4[xyu][du] {
                h.num_edges_in += 1;
            }
            if bv.edge_frac4[xyu][du] < 0. {
                bv.edge_frac4[xyu][du] = 0.;
            }
            if bv.edge_frac4[xyu][du] > 1. {
                bv.edge_frac4[xyu][du] = 1.;
            }
            if bv.debug {
                let _ = writeln!(
                    out,
                    "{}{}{}{}",
                    ld_int(ied),
                    ld_int(ixy),
                    ld_real(bv.edge_frac4[xyu][du]),
                    ld_logical(h.active4[xyu][du])
                );
            }
        }
    }
}

/// Original: `subroutine getPixelFromPieces()`, contained in `blendmont`
/// (`blendmont.f90:3099`).
///
/// Sorts out the cases of 1, 2, 3 or 4 pieces and computes the pixel into
/// the host's `pixVal`.  Reads the host's `xg`, `ysrc`, `numIter`,
/// `numEdgesIn`, `active4`, `hxf` and `oneEdgeJustAvgCrit`; the rest of the
/// host variables it assigns are scratch, locals here.
pub fn get_pixel_from_pieces(h: &mut Host, bv: &mut BlendVars) {
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let (xg, ysrc) = (h.xg, h.ysrc);
    let num_iter = h.num_iter;
    let [indp1, indp2, indp3, indp4] = [
        bv.indp1234[0],
        bv.indp1234[1],
        bv.indp1234[2],
        bv.indp1234[3],
    ];
    let (wll, wlr, wul, wur) = (bv.wll, bv.wlr, bv.wul, bv.wur);
    let mut ind_array = 0i32;
    let (mut x1, mut y1, mut x2, mut y2, mut x3, mut y3, mut x4, mut y4) = (
        0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
    );
    let (mut x3t, mut y3t) = (0.0f32, 0.0f32);
    let (mut dden, mut dden12, mut dden13, mut dden23, mut dden14, mut dden43) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut c11, mut c12, mut c21, mut c22, mut denom) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut fb11, mut fb12, mut fb21, mut fb22) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut f2b11, mut f2b12, mut f2b21, mut f2b22) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut f3b11, mut f3b12, mut f3b21, mut f3b22) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut f4b11, mut f4b12, mut f4b21, mut f4b22) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut ind12edg, mut ind13_edge, mut ind23edg, mut ind14edg, mut ind43edg) =
        (0i32, 0i32, 0i32, 0i32, 0i32);
    let hxf = |k: i32, j: i32, ipc: i32| h.hxf[h3!(k, j, ipc)];
    let ixpc = |ipc: i32| bv.ix_pc_list[(ipc - 1) as usize] as f32;
    let iypc = |ipc: i32| bv.iy_pc_list[(ipc - 1) as usize] as f32;
    let multng = bv.multng;
    //
    if bv.n_active_p <= 1 {
        //ONE PIECE
        let mut ind_pc_one = if wll > 0. {
            indp1
        } else if wlr > 0. {
            indp2
        } else if wul > 0. {
            indp3
        } else {
            indp4
        };
        if bv.n_active_p == 0 {
            ind_pc_one = 1;
        }
        //
        let ipc = bv.in_piece[ind_pc_one as usize];
        shuffler(bv, ipc, &mut ind_array);
        let u = (ind_pc_one - 1) as usize;
        h.pix_val = oneintrp(
            bv,
            &bv.array[(ind_array - 1) as usize..],
            nxin,
            nyin,
            bv.x_in_piece[u],
            bv.y_in_piece[u],
            ipc,
        );
        //
        // ONE EDGE, TWO PIECES
        //
    } else if bv.n_active_p == 2 {
        //
        // find the pieces around the edge, set edge index
        //
        let (ind_pc_lo, ind_pc_up, w1, ind_edge);
        if wll > 0. && wlr > 0. {
            ind_pc_lo = indp1;
            ind_pc_up = indp2;
            w1 = wll;
            ind_edge = bv.ind_edge4[0][(bv.inde12 - 1) as usize];
        } else if wul > 0. && wur > 0. {
            ind_pc_lo = indp3;
            ind_pc_up = indp4;
            w1 = wul;
            ind_edge = bv.ind_edge4[0][(bv.inde34 - 1) as usize];
        } else if wll > 0. && wul > 0. {
            ind_pc_lo = indp1;
            ind_pc_up = indp3;
            w1 = wll;
            ind_edge = bv.ind_edge4[1][(bv.inde13 - 1) as usize];
        } else {
            ind_pc_lo = indp2;
            ind_pc_up = indp4;
            w1 = wlr;
            ind_edge = bv.ind_edge4[1][(bv.inde24 - 1) as usize];
        }
        h.ind_edge = ind_edge;
        //
        // set up pieces #'s, and starting coords
        //
        let ipiece1 = bv.in_piece[ind_pc_lo as usize];
        let ipiece2 = bv.in_piece[ind_pc_up as usize];
        x1 = bv.x_in_piece[(ind_pc_lo - 1) as usize];
        y1 = bv.y_in_piece[(ind_pc_lo - 1) as usize];
        let w2 = 1. - w1;
        //
        // set up to solve equation for (x1, y1) and (x2, y2)
        // given their difference and the desired xg, yg
        //
        let mut x_src_const = xg - w1 * ixpc(ipiece1) - w2 * ixpc(ipiece2);
        let mut y_src_const = ysrc - w1 * iypc(ipiece1) - w2 * iypc(ipiece2);
        if multng {
            x_src_const = x_src_const - w1 * hxf(1, 3, ipiece1) - w2 * hxf(1, 3, ipiece2);
            y_src_const = y_src_const - w1 * hxf(2, 3, ipiece1) - w2 * hxf(2, 3, ipiece2);
            c11 = w1 * hxf(1, 1, ipiece1) + w2 * hxf(1, 1, ipiece2);
            c12 = w1 * hxf(1, 2, ipiece1) + w2 * hxf(1, 2, ipiece2);
            c21 = w1 * hxf(2, 1, ipiece1) + w2 * hxf(2, 1, ipiece2);
            c22 = w1 * hxf(2, 2, ipiece1) + w2 * hxf(2, 2, ipiece2);
            denom = c11 * c22 - c12 * c21;
            fb11 = w2 * hxf(1, 1, ipiece2);
            fb12 = w2 * hxf(1, 2, ipiece2);
            fb21 = w2 * hxf(2, 1, ipiece2);
            fb22 = w2 * hxf(2, 2, ipiece2);
        }
        //
        // in this loop, use the difference between (x1, y1)
        // and (x2, y2) at the one place to solve for values of
        // those coords; iterate until stabilize
        //
        let mut x1last = -100.0f32;
        let mut y1last = -100.0f32;
        let mut iter = 1;
        while iter <= num_iter && ((x1 - x1last).abs() > 0.01 || (y1 - y1last).abs() > 0.01) {
            dxydgrinterp(bv, x1, y1, ind_edge, &mut x2, &mut y2, &mut dden);
            x1last = x1;
            y1last = y1;
            let dx2 = x2 - x1;
            let dy2 = y2 - y1;
            if multng {
                let bx = x_src_const - fb11 * dx2 - fb12 * dy2;
                let by = y_src_const - fb21 * dx2 - fb22 * dy2;
                x1 = (bx * c22 - by * c12) / denom;
                y1 = (by * c11 - bx * c21) / denom;
            } else {
                x1 = x_src_const - w2 * dx2;
                y1 = y_src_const - w2 * dy2;
            }
            x2 = x1 + dx2;
            y2 = y1 + dy2;
            iter += 1;
        }
        //
        // get the pixel from the piece with the bigger weight
        // and adjust density by the difference across edge
        // Except average them within 4 pixels of center!
        // Make it wider if gradients solved here, because density correction is not perfect
        //
        let ib = bv.iblend[0].max(bv.iblend[1]);
        if (w1 - w2).abs() * (ib as f32) < h.one_edge_just_avg_crit {
            let mut ind_array1 = 0i32;
            let mut ind_array2 = 0i32;
            shuffler(bv, ipiece1, &mut ind_array1);
            shuffler(bv, ipiece2, &mut ind_array2);
            h.pix_val = w1
                * oneintrp(
                    bv,
                    &bv.array[(ind_array1 - 1) as usize..],
                    nxin,
                    nyin,
                    x1,
                    y1,
                    ipiece1,
                )
                + w2 * oneintrp(
                    bv,
                    &bv.array[(ind_array2 - 1) as usize..],
                    nxin,
                    nyin,
                    x2,
                    y2,
                    ipiece2,
                );
        } else if w1 > w2 {
            shuffler(bv, ipiece1, &mut ind_array);
            h.pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x1,
                y1,
                ipiece1,
            ) + w2 * dden;
        } else {
            shuffler(bv, ipiece2, &mut ind_array);
            h.pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x2,
                y2,
                ipiece2,
            ) - w1 * dden;
        }
        //
        // THREE PIECES AND TWO EDGES
        //
    } else if bv.n_active_p == 3 {
        //
        // now decide between the three cases of divergent, serial, or
        // convergent functions, and assign pieces and weights accordingly
        //
        // DIVERGENT: if lower piece is same for x and y edge
        //
        let (w1, w2, w3, jnd_p1, jnd_p2, jnd_p3, i_corn_type);
        if wur <= 0. {
            w1 = wll;
            w2 = wlr;
            w3 = wul;
            jnd_p1 = indp1;
            jnd_p2 = indp2;
            jnd_p3 = indp3;
            ind12edg = bv.ind_edge4[0][(bv.inde12 - 1) as usize];
            ind13_edge = bv.ind_edge4[1][(bv.inde13 - 1) as usize];
            i_corn_type = 1;
            //
            // CONVERGENT: if upper piece same for x and y edge
            //
        } else if wll <= 0. {
            w3 = wur;
            w1 = wul;
            w2 = wlr;
            jnd_p1 = indp3;
            jnd_p2 = indp2;
            jnd_p3 = indp4;
            ind13_edge = bv.ind_edge4[0][(bv.inde34 - 1) as usize];
            ind23edg = bv.ind_edge4[1][(bv.inde24 - 1) as usize];
            i_corn_type = 2;
            //
            // SERIAL: lower of one is the upper of the other
            //
        } else {
            w1 = wll;
            w3 = wur;
            jnd_p1 = indp1;
            jnd_p3 = indp4;
            if wlr <= 0. {
                w2 = wul;
                jnd_p2 = indp3;
                ind12edg = bv.ind_edge4[1][(bv.inde13 - 1) as usize];
                ind23edg = bv.ind_edge4[0][(bv.inde34 - 1) as usize];
            } else {
                w2 = wlr;
                jnd_p2 = indp2;
                ind12edg = bv.ind_edge4[0][(bv.inde12 - 1) as usize];
                ind23edg = bv.ind_edge4[1][(bv.inde24 - 1) as usize];
            }
            i_corn_type = 3;
        }

        let ipiece1 = bv.in_piece[jnd_p1 as usize];
        let ipiece2 = bv.in_piece[jnd_p2 as usize];
        let ipiece3 = bv.in_piece[jnd_p3 as usize];
        x1 = bv.x_in_piece[(jnd_p1 - 1) as usize];
        y1 = bv.y_in_piece[(jnd_p1 - 1) as usize];
        // `max(w1, w2, w3)` (`blendmont.f90:3278`): `maxss(maxss(w1, w2), w3)`
        // in the reference object.
        let w_max = maxss(maxss(w1, w2), w3);
        //
        // set up to solve equations for new (x1, y1), (x2, y2)
        // and (x3, y3) given the differences between them and
        // the desired weighted coordinate (xg, yg)
        //
        let mut x_src_const = xg - w1 * ixpc(ipiece1) - w2 * ixpc(ipiece2) - w3 * ixpc(ipiece3);
        let mut y_src_const = ysrc - w1 * iypc(ipiece1) - w2 * iypc(ipiece2) - w3 * iypc(ipiece3);
        if multng {
            x_src_const = x_src_const
                - w1 * hxf(1, 3, ipiece1)
                - w2 * hxf(1, 3, ipiece2)
                - w3 * hxf(1, 3, ipiece3);
            y_src_const = y_src_const
                - w1 * hxf(2, 3, ipiece1)
                - w2 * hxf(2, 3, ipiece2)
                - w3 * hxf(2, 3, ipiece3);
            c11 = w1 * hxf(1, 1, ipiece1) + w2 * hxf(1, 1, ipiece2) + w3 * hxf(1, 1, ipiece3);
            c12 = w1 * hxf(1, 2, ipiece1) + w2 * hxf(1, 2, ipiece2) + w3 * hxf(1, 2, ipiece3);
            c21 = w1 * hxf(2, 1, ipiece1) + w2 * hxf(2, 1, ipiece2) + w3 * hxf(2, 1, ipiece3);
            c22 = w1 * hxf(2, 2, ipiece1) + w2 * hxf(2, 2, ipiece2) + w3 * hxf(2, 2, ipiece3);
            denom = c11 * c22 - c12 * c21;
            f2b11 = w2 * hxf(1, 1, ipiece2);
            f2b12 = w2 * hxf(1, 2, ipiece2);
            f2b21 = w2 * hxf(2, 1, ipiece2);
            f2b22 = w2 * hxf(2, 2, ipiece2);
            f3b11 = w3 * hxf(1, 1, ipiece3);
            f3b12 = w3 * hxf(1, 2, ipiece3);
            f3b21 = w3 * hxf(2, 1, ipiece3);
            f3b22 = w3 * hxf(2, 2, ipiece3);
        }
        //
        // do iteration, starting with coordinates and solving
        // for new coordinates until convergence
        //
        let mut x1last = -100.0f32;
        let mut y1last = -100.0f32;
        let mut iter = 1;
        while iter <= num_iter && ((x1 - x1last).abs() > 0.01 || (y1 - y1last).abs() > 0.01) {
            if i_corn_type == 1 {
                //
                // divergent case
                //
                dxydgrinterp(bv, x1, y1, ind12edg, &mut x2, &mut y2, &mut dden12);
                dxydgrinterp(bv, x1, y1, ind13_edge, &mut x3, &mut y3, &mut dden13);
            } else if i_corn_type == 2 {
                //
                // convergent case
                //
                dxydgrinterp(bv, x1, y1, ind13_edge, &mut x3, &mut y3, &mut dden13);
                //
                if iter == 1 {
                    let e = (ind23edg - 1) as usize;
                    x2 = x3 + bv.ix_grd_st_bf[e] as f32 - bv.ix_ofs_bf[e] as f32;
                    y2 = y3 + bv.iy_grd_st_bf[e] as f32 - bv.iy_ofs_bf[e] as f32;
                }
                dxydgrinterp(bv, x2, y2, ind23edg, &mut x3t, &mut y3t, &mut dden23);
                x2 = x2 + x3 - x3t;
                y2 = y2 + y3 - y3t;
            } else {
                //
                // serial case
                //
                dxydgrinterp(bv, x1, y1, ind12edg, &mut x2, &mut y2, &mut dden12);
                dxydgrinterp(bv, x2, y2, ind23edg, &mut x3, &mut y3, &mut dden23);
            }
            //
            // solve equations for new coordinates
            //
            x1last = x1;
            y1last = y1;
            let dx2 = x2 - x1;
            let dy2 = y2 - y1;
            let dx3 = x3 - x1;
            let dy3 = y3 - y1;
            if multng {
                let bx = x_src_const - f2b11 * dx2 - f2b12 * dy2 - f3b11 * dx3 - f3b12 * dy3;
                let by = y_src_const - f2b21 * dx2 - f2b22 * dy2 - f3b21 * dx3 - f3b22 * dy3;
                x1 = (bx * c22 - by * c12) / denom;
                y1 = (by * c11 - bx * c21) / denom;
            } else {
                x1 = x_src_const - w2 * dx2 - w3 * dx3;
                y1 = y_src_const - w2 * dy2 - w3 * dy3;
            }
            x2 = x1 + dx2;
            y2 = y1 + dy2;
            x3 = x1 + dx3;
            y3 = y1 + dy3;
            iter += 1;
        }
        //
        // take pixel from the piece with the highest weight
        //
        let mut pix_val;
        if w1 == w_max {
            shuffler(bv, ipiece1, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x1,
                y1,
                ipiece1,
            );
        } else if w2 == w_max {
            shuffler(bv, ipiece2, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x2,
                y2,
                ipiece2,
            );
        } else {
            shuffler(bv, ipiece3, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x3,
                y3,
                ipiece3,
            );
        }
        //
        // Adjust for differences in mean density: divergent
        //
        if i_corn_type == 1 {
            if w1 == w_max {
                pix_val = pix_val + w2 * dden12 + w3 * dden13;
            } else if w2 == w_max {
                pix_val = pix_val + (w2 - 1.) * dden12 + w3 * dden13;
            } else {
                pix_val = pix_val + w2 * dden12 + (w3 - 1.) * dden13;
            }
            //
            // convergent
            //
        } else if i_corn_type == 2 {
            if w1 == w_max {
                pix_val = pix_val - (w1 - 1.) * dden13 - w2 * dden23;
            } else if w2 == w_max {
                pix_val = pix_val - w1 * dden13 - (w2 - 1.) * dden23;
            } else {
                pix_val = pix_val - w1 * dden13 - w2 * dden23;
            }
            //
            // serial
            //
        } else {
            //
            if w1 == w_max {
                pix_val = pix_val - (w1 - 1.) * dden12 + w3 * dden23;
            } else if w2 == w_max {
                pix_val = pix_val - w1 * dden12 + w3 * dden23;
            } else {
                pix_val = pix_val - w1 * dden12 + (w3 - 1.) * dden23;
            }
        }
        h.pix_val = pix_val;
    } else {
        //
        // FOUR PIECES, THREE EDGES USED FOR SOLUTION
        //
        // First, need to have only 3 active edges, so if
        // there are four, knock one out based on ex and ey
        //
        let (ex, ey) = (bv.ex, bv.ey);
        let (i12, i34, i13, i24) = (
            (bv.inde12 - 1) as usize,
            (bv.inde34 - 1) as usize,
            (bv.inde13 - 1) as usize,
            (bv.inde24 - 1) as usize,
        );
        if h.num_edges_in == 4 {
            // `min(ex, 1. - ex, ey, 1. - ey)` (`blendmont.f90:3422`): the
            // reference object computes
            // `minss(minss(1. - ey, 1. - ex), minss(ex, ey))`.
            let emin = minss(minss(1. - ey, 1. - ex), minss(ex, ey));
            if ex == emin {
                h.active4[0][i34] = false;
            } else if 1. - ex == emin {
                h.active4[0][i12] = false;
            } else if ey == emin {
                h.active4[1][i24] = false;
            } else {
                h.active4[1][i13] = false;
            }
        }
        //
        // here there is always a serial chain from ll to ur, through either ul
        // or lr, and in each case the fourth piece is either divergent from
        // the first or converging on the fourth
        //
        let w1 = wll;
        let w3 = wur;
        let (w2, w4, jnd_p2, jnd_p4, i_corn_type);
        if !h.active4[0][i12] || !h.active4[1][i24] {
            w2 = wul;
            w4 = wlr;
            jnd_p2 = indp3;
            jnd_p4 = indp2;
            ind12edg = bv.ind_edge4[1][i13];
            ind23edg = bv.ind_edge4[0][i34];
            if !h.active4[1][i24] {
                ind14edg = bv.ind_edge4[0][i12];
                i_corn_type = 1; //divergent
            } else {
                ind43edg = bv.ind_edge4[1][i24];
                i_corn_type = 2; //convergent
            }
        } else {
            w2 = wlr;
            w4 = wul;
            jnd_p2 = indp2;
            jnd_p4 = indp3;
            ind12edg = bv.ind_edge4[0][i12];
            ind23edg = bv.ind_edge4[1][i24];
            if !h.active4[0][i34] {
                ind14edg = bv.ind_edge4[1][i13];
                i_corn_type = 1;
            } else {
                ind43edg = bv.ind_edge4[0][i34];
                i_corn_type = 2;
            }
        }
        //
        let ipiece1 = bv.in_piece[indp1 as usize];
        let ipiece2 = bv.in_piece[jnd_p2 as usize];
        let ipiece3 = bv.in_piece[indp4 as usize];
        let ipiece4 = bv.in_piece[jnd_p4 as usize];
        x1 = bv.x_in_piece[(indp1 - 1) as usize];
        y1 = bv.y_in_piece[(indp1 - 1) as usize];
        // `max(w1, w2, w3, w4)` (`blendmont.f90:3477`): the reference object
        // computes `maxss(maxss(w2, w4), maxss(w3, w1))` (the last pair hoisted
        // above the branch that picks `w2`/`w4`).
        let w_max = maxss(maxss(w2, w4), maxss(w3, w1));
        //
        // set up to solve equations for new (x1, y1), (x2, y2)
        // (x3, y3), and (x4, y4) given the differences between
        // them and the desired weighted coordinate (xg, yg)
        //
        let mut x_src_const =
            xg - w1 * ixpc(ipiece1) - w2 * ixpc(ipiece2) - w3 * ixpc(ipiece3) - w4 * ixpc(ipiece4);
        let mut y_src_const = ysrc
            - w1 * iypc(ipiece1)
            - w2 * iypc(ipiece2)
            - w3 * iypc(ipiece3)
            - w4 * iypc(ipiece4);
        if multng {
            x_src_const = x_src_const
                - w1 * hxf(1, 3, ipiece1)
                - w2 * hxf(1, 3, ipiece2)
                - w3 * hxf(1, 3, ipiece3)
                - w4 * hxf(1, 3, ipiece4);
            y_src_const = y_src_const
                - w1 * hxf(2, 3, ipiece1)
                - w2 * hxf(2, 3, ipiece2)
                - w3 * hxf(2, 3, ipiece3)
                - w4 * hxf(2, 3, ipiece4);
            c11 = w1 * hxf(1, 1, ipiece1)
                + w2 * hxf(1, 1, ipiece2)
                + w3 * hxf(1, 1, ipiece3)
                + w4 * hxf(1, 1, ipiece4);
            c12 = w1 * hxf(1, 2, ipiece1)
                + w2 * hxf(1, 2, ipiece2)
                + w3 * hxf(1, 2, ipiece3)
                + w4 * hxf(1, 2, ipiece4);
            c21 = w1 * hxf(2, 1, ipiece1)
                + w2 * hxf(2, 1, ipiece2)
                + w3 * hxf(2, 1, ipiece3)
                + w4 * hxf(2, 1, ipiece4);
            c22 = w1 * hxf(2, 2, ipiece1)
                + w2 * hxf(2, 2, ipiece2)
                + w3 * hxf(2, 2, ipiece3)
                + w4 * hxf(2, 2, ipiece4);
            denom = c11 * c22 - c12 * c21;
            f2b11 = w2 * hxf(1, 1, ipiece2);
            f2b12 = w2 * hxf(1, 2, ipiece2);
            f2b21 = w2 * hxf(2, 1, ipiece2);
            f2b22 = w2 * hxf(2, 2, ipiece2);
            f3b11 = w3 * hxf(1, 1, ipiece3);
            f3b12 = w3 * hxf(1, 2, ipiece3);
            f3b21 = w3 * hxf(2, 1, ipiece3);
            f3b22 = w3 * hxf(2, 2, ipiece3);
            f4b11 = w4 * hxf(1, 1, ipiece4);
            f4b12 = w4 * hxf(1, 2, ipiece4);
            f4b21 = w4 * hxf(2, 1, ipiece4);
            f4b22 = w4 * hxf(2, 2, ipiece4);
        }
        //
        // do iteration, starting with coordinates and solving
        // for new coordinates until convergence
        //
        let mut x1last = -100.0f32;
        let mut y1last = -100.0f32;
        let mut iter = 1;
        while iter <= num_iter && ((x1 - x1last).abs() > 0.01 || (y1 - y1last).abs() > 0.01) {
            dxydgrinterp(bv, x1, y1, ind12edg, &mut x2, &mut y2, &mut dden12);
            dxydgrinterp(bv, x2, y2, ind23edg, &mut x3, &mut y3, &mut dden23);
            if i_corn_type == 1 {
                //
                // divergent case
                //
                dxydgrinterp(bv, x1, y1, ind14edg, &mut x4, &mut y4, &mut dden14);
            } else {
                //
                // convergent case
                //
                if iter == 1 {
                    let e = (ind43edg - 1) as usize;
                    x4 = x3 + bv.ix_grd_st_bf[e] as f32 - bv.ix_ofs_bf[e] as f32;
                    y4 = y3 + bv.iy_grd_st_bf[e] as f32 - bv.iy_ofs_bf[e] as f32;
                }
                dxydgrinterp(bv, x4, y4, ind43edg, &mut x3t, &mut y3t, &mut dden43);
                x4 = x4 + x3 - x3t;
                y4 = y4 + y3 - y3t;
            }
            //
            // solve equations for new coordinates
            //
            x1last = x1;
            y1last = y1;
            let dx2 = x2 - x1;
            let dy2 = y2 - y1;
            let dx3 = x3 - x1;
            let dy3 = y3 - y1;
            let dx4 = x4 - x1;
            let dy4 = y4 - y1;
            if multng {
                let bx = x_src_const
                    - f2b11 * dx2
                    - f2b12 * dy2
                    - f3b11 * dx3
                    - f3b12 * dy3
                    - f4b11 * dx4
                    - f4b12 * dy4;
                let by = y_src_const
                    - f2b21 * dx2
                    - f2b22 * dy2
                    - f3b21 * dx3
                    - f3b22 * dy3
                    - f4b21 * dx4
                    - f4b22 * dy4;
                x1 = (bx * c22 - by * c12) / denom;
                y1 = (by * c11 - bx * c21) / denom;
            } else {
                x1 = x_src_const - w2 * dx2 - w3 * dx3 - w4 * dx4;
                y1 = y_src_const - w2 * dy2 - w3 * dy3 - w4 * dy4;
            }
            x2 = x1 + dx2;
            y2 = y1 + dy2;
            x3 = x1 + dx3;
            y3 = y1 + dy3;
            x4 = x1 + dx4;
            y4 = y1 + dy4;
            iter += 1;
        }
        //
        // take pixel from the piece with the highest weight
        // and adjust density appropriately for case
        //
        let mut pix_val;
        if w1 == w_max {
            shuffler(bv, ipiece1, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x1,
                y1,
                ipiece1,
            );
            if i_corn_type == 1 {
                pix_val = pix_val + (w2 + w3) * dden12 + w3 * dden23 + w4 * dden14;
            } else {
                pix_val = pix_val + (1. - w1) * dden12 + (w3 + w4) * dden23 - w4 * dden43;
            }
        } else if w2 == w_max {
            shuffler(bv, ipiece2, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x2,
                y2,
                ipiece2,
            );
            if i_corn_type == 1 {
                pix_val = pix_val + (w2 + w3 - 1.) * dden12 + w3 * dden23 + w4 * dden14;
            } else {
                pix_val = pix_val - w1 * dden12 + (w3 + w4) * dden23 - w4 * dden43;
            }
        } else if w3 == w_max {
            shuffler(bv, ipiece3, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x3,
                y3,
                ipiece3,
            );
            if i_corn_type == 1 {
                pix_val = pix_val + (w2 + w3 - 1.) * dden12 + (w3 - 1.) * dden23 + w4 * dden14;
            } else {
                pix_val = pix_val - w1 * dden12 + (w3 + w4 - 1.) * dden23 - w4 * dden43;
            }
        } else {
            shuffler(bv, ipiece4, &mut ind_array);
            pix_val = oneintrp(
                bv,
                &bv.array[(ind_array - 1) as usize..],
                nxin,
                nyin,
                x4,
                y4,
                ipiece4,
            );
            if i_corn_type == 1 {
                pix_val = pix_val + (w2 + w3) * dden12 + w3 * dden23 + (w4 - 1.) * dden14;
            } else {
                pix_val = pix_val - w1 * dden12 + (w3 + w4 - 1.) * dden23 - (w4 - 1.) * dden43;
            }
        }
        h.pix_val = pix_val;
    }
}

/// Original: `subroutine getPieceIndicesAndWeighting(useEdges)`
/// (`blendmont.f90:3616`), the external subroutine after the main program.
///
/// Gets indices of pieces and edges and the weighting of each piece.
/// `use blendvars`, so the module struct comes first; it calls `edgeSwap`,
/// so it also takes the unit connections.  Its locals are not SAVE; none is
/// read before it is assigned on any path.  In the X-disjoint branch for a
/// missing upper-right piece the source computes `er1` from `dr3`, not the
/// `dr1` assigned on the line before (`blendmont.f90:3927-3928`); here from
/// `dr1` (fixed in translation, `BUGS.md`).
pub fn get_piece_indices_and_weighting(
    bv: &mut BlendVars,
    units: &mut BlendUnits,
    use_edges: bool,
) {
    let (mut er3, mut eb3, mut el4, mut eb4) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut fx, mut fy) = (0.0f32, 0.0f32);
    let (mut dr1, mut dt1, mut dl2, mut dt2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut dr3, mut db3, mut dl4, mut db4) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut f12, mut f13, mut f34, mut f24) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut er1, mut et1, mut el2, mut et2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut ip_from, mut ip_xto, mut ip_yto) = (0i32, 0i32, 0i32);
    let mut ixy_disjoint: i32;
    let mut ind_edge = 0i32;
    let mut jedge = 0i32;
    let (mut bw_offset, mut edge_start, mut edge_end): (f32, f32, f32);
    let mut out = ImodFile::Stdout;
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let intg = bv.int_grid[0];
    let iblend = bv.iblend;

    // for now, numbers 1 to 4 represent lower left, lower right, upper left,
    // and upper right
    if bv.num_pieces > 1 {
        bv.indp1234[0] = 0; //index of piece 1, 2, 3, or 4
        bv.indp1234[1] = 0; //if the point is in them
        bv.indp1234[2] = 0;
        bv.indp1234[3] = 0;
        bv.inde12 = 0; //index of edges between 1 and 2
        bv.inde13 = 0; //1 and 3, etc
        bv.inde34 = 0;
        bv.inde24 = 0;
        f12 = 0.; //edge fractions between 1 and 2
        f13 = 0.; //1 and 3, etc
        f34 = 0.;
        f24 = 0.;
        er1 = 0.; //end fractions to right and top
        et1 = 0.; //of piece 1, left and bottom of
        el2 = 0.; //piece 3, etc
        et2 = 0.;
        er3 = 0.;
        eb3 = 0.;
        el4 = 0.;
        eb4 = 0.;
        fx = 0.; //the composite edge fractions
        fy = 0.; //in x and y directions
        if bv.num_pieces > 2 {
            for ipc in 1..=bv.num_pieces {
                let u = (ipc - 1) as usize;
                let i =
                    2 * (bv.inp_yframe[u] - bv.min_yframe) + 1 + bv.inp_xframe[u] - bv.min_xframe;
                bv.indp1234[(i - 1) as usize] = ipc;
            }
            let [indp1, indp2, indp3, _] = [
                bv.indp1234[0],
                bv.indp1234[1],
                bv.indp1234[2],
                bv.indp1234[3],
            ];
            for ied in 1..=bv.num_edges[0] {
                if bv.in_ed_lower[0][(ied - 1) as usize] == indp1 {
                    bv.inde12 = ied;
                }
                if bv.in_ed_lower[0][(ied - 1) as usize] == indp3 {
                    bv.inde34 = ied;
                }
            }
            for ied in 1..=bv.num_edges[1] {
                if bv.in_ed_lower[1][(ied - 1) as usize] == indp1 {
                    bv.inde13 = ied;
                }
                if bv.in_ed_lower[1][(ied - 1) as usize] == indp2 {
                    bv.inde24 = ied;
                }
            }
            if bv.debug {
                let _ = writeln!(
                    out,
                    "{}{}{}{}",
                    ld_int(bv.inde12),
                    ld_int(bv.inde34),
                    ld_int(bv.inde13),
                    ld_int(bv.inde24)
                );
            }
            if bv.inde12 != 0 {
                f12 = bv.edge_frac4[0][(bv.inde12 - 1) as usize];
            }
            if bv.inde34 != 0 {
                f34 = bv.edge_frac4[0][(bv.inde34 - 1) as usize];
            }
            if bv.inde13 != 0 {
                f13 = bv.edge_frac4[1][(bv.inde13 - 1) as usize];
            }
            if bv.inde24 != 0 {
                f24 = bv.edge_frac4[1][(bv.inde24 - 1) as usize];
            }
            if bv.debug {
                let p = |k: usize| bv.in_piece[bv.indp1234[k] as usize];
                let _ = writeln!(
                    out,
                    "piece 1234{}{}{}{}  edge fractions{}{}{}{}",
                    i_edit(p(0), 6),
                    i_edit(p(1), 6),
                    i_edit(p(2), 6),
                    i_edit(p(3), 6),
                    f_edit(f12, 8, 4),
                    f_edit(f34, 8, 4),
                    f_edit(f13, 8, 4),
                    f_edit(f24, 8, 4)
                );
            }
        } else {
            //
            // two piece case - identify as upper or lower to
            // simplify computation of end fractions
            //
            if bv.num_edges[0] > 0 {
                fx = bv.edge_frac4[0][0];
                if 0.5 * (bv.y_in_piece[0] + bv.y_in_piece[1]) > (nyin / 2) as f32 {
                    bv.inde12 = 1;
                    if bv.x_in_piece[0] > bv.x_in_piece[1] {
                        bv.indp1234[0] = 1;
                        bv.indp1234[1] = 2;
                    } else {
                        bv.indp1234[0] = 2;
                        bv.indp1234[1] = 1;
                    }
                } else {
                    bv.inde34 = 1;
                    fy = 1.;
                    if bv.x_in_piece[0] > bv.x_in_piece[1] {
                        bv.indp1234[2] = 1;
                        bv.indp1234[3] = 2;
                    } else {
                        bv.indp1234[2] = 2;
                        bv.indp1234[3] = 1;
                    }
                }
            } else {
                fy = bv.edge_frac4[1][0];
                if 0.5 * (bv.x_in_piece[0] + bv.x_in_piece[1]) > (nxin / 2) as f32 {
                    bv.inde13 = 1;
                    if bv.y_in_piece[0] > bv.y_in_piece[1] {
                        bv.indp1234[0] = 1;
                        bv.indp1234[2] = 2;
                    } else {
                        bv.indp1234[0] = 2;
                        bv.indp1234[2] = 1;
                    }
                } else {
                    bv.inde24 = 1;
                    fx = 1.;
                    if bv.y_in_piece[0] > bv.y_in_piece[1] {
                        bv.indp1234[1] = 1;
                        bv.indp1234[3] = 2;
                    } else {
                        bv.indp1234[1] = 2;
                        bv.indp1234[3] = 1;
                    }
                }
            }
        }
        let [indp1, indp2, indp3, indp4] = [
            bv.indp1234[0],
            bv.indp1234[1],
            bv.indp1234[2],
            bv.indp1234[3],
        ];
        let xin = |bv: &BlendVars, k: i32| bv.x_in_piece[(k - 1) as usize];
        let yin = |bv: &BlendVars, k: i32| bv.y_in_piece[(k - 1) as usize];
        //
        // get distance to top, right, bottom, or left edges
        // as needed for each piece, and compute end fractions
        //
        if indp1 > 0 {
            dr1 = nxin as f32 - 1. - xin(bv, indp1);
            dt1 = nyin as f32 - 1. - yin(bv, indp1);
            er1 = (dr1 + 1.) / iblend[0] as f32;
            et1 = dt1 / iblend[1] as f32;
        }
        if indp2 > 0 {
            dl2 = xin(bv, indp2);
            dt2 = nyin as f32 - 1. - yin(bv, indp2);
            el2 = (dl2 + 1.) / iblend[0] as f32;
            et2 = (dt2 + 1.) / iblend[1] as f32;
        }
        if indp3 > 0 {
            dr3 = nxin as f32 - 1. - xin(bv, indp3);
            db3 = yin(bv, indp3);
            er3 = (dr3 + 1.) / iblend[0] as f32;
            eb3 = (db3 + 1.) / iblend[1] as f32;
        }
        if indp4 > 0 {
            dl4 = xin(bv, indp4);
            db4 = yin(bv, indp4);
            el4 = (dl4 + 1.) / iblend[0] as f32;
            eb4 = (db4 + 1.) / iblend[1] as f32;
        }
        //
        // If 3 pieces, check for a disjoint edge: use previous state if pieces
        // match the last time
        if bv.num_pieces == 3
            && bv.any_disjoint[a2!(bv.any_disjoint_ext, bv.min_xframe, bv.min_yframe)]
        {
            let pc = |bv: &BlendVars, k: i32| bv.in_piece[k as usize];
            if pc(bv, indp1) == bv.lastp1
                && pc(bv, indp2) == bv.lastp2
                && pc(bv, indp3) == bv.lastp3
                && pc(bv, indp4) == bv.lastp4
            {
                ixy_disjoint = bv.lastxy_disjoint;
            } else {
                //
                // if pieces have changed, analyze edge starts and ends.
                // First find the missing edges and set some indexes
                ixy_disjoint = 0;
                let lo = |bv: &BlendVars, ipc: i32, ixy: i32| {
                    bv.iedge_lower[a2!(bv.iedge_lower_ext, ipc, ixy)]
                };
                let up = |bv: &BlendVars, ipc: i32, ixy: i32| {
                    bv.iedge_upper[a2!(bv.iedge_upper_ext, ipc, ixy)]
                };
                if indp1 == 0 {
                    bv.in_edge[0][1] = lo(bv, pc(bv, indp2), 1);
                    bv.in_edge[1][1] = lo(bv, pc(bv, indp3), 2);
                    ip_from = indp4;
                    ip_xto = indp2;
                    ip_yto = indp3;
                } else if indp2 == 0 {
                    bv.in_edge[0][1] = up(bv, pc(bv, indp1), 1);
                    bv.in_edge[1][1] = lo(bv, pc(bv, indp4), 2);
                    ip_from = indp3;
                    ip_xto = indp1;
                    ip_yto = indp4;
                } else if indp3 == 0 {
                    bv.in_edge[0][1] = lo(bv, pc(bv, indp4), 1);
                    bv.in_edge[1][1] = up(bv, pc(bv, indp1), 2);
                    ip_from = indp2;
                    ip_xto = indp4;
                    ip_yto = indp1;
                } else if indp4 == 0 {
                    bv.in_edge[0][1] = up(bv, pc(bv, indp3), 1);
                    bv.in_edge[1][1] = up(bv, pc(bv, indp2), 2);
                    ip_from = indp1;
                    ip_xto = indp3;
                    ip_yto = indp2;
                }
                //
                // Replace with used edge numbers
                if use_edges {
                    for ixy in 1..=2 {
                        let xyu = (ixy - 1) as usize;
                        find_edge_to_use(bv, bv.in_edge[xyu][1], ixy, &mut jedge);
                        if jedge != 0 {
                            bv.in_edge[xyu][1] = jedge;
                        }
                    }
                }
                //
                // Analyze X edge first then Y edge - don't worry if both are bad
                if bv.in_edge[0][1] > 0 && bv.in_edge[1][1] > 0 {
                    //
                    // Get start of already included X edge and translate it into
                    // piece on missing X edge, and get end of the missing X edge
                    // in that piece.  If end is before start, it is disjoint
                    if indp1 == 0 || indp3 == 0 {
                        ind_edge = bv.ind_edge4[0][0];
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.nx_gr_bf[e] * intg - iblend[0]) / 2) as f32;
                        edge_start =
                            bv.ix_ofs_bf[e] as f32 - intg as f32 / 2. + bw_offset + xin(bv, ip_xto)
                                - xin(bv, ip_from);
                        let ie = bv.in_edge[0][1];
                        edgeswap(bv, units, ie, 1, &mut ind_edge);
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.nx_gr_bf[e] * intg - iblend[0]) / 2) as f32;
                        edge_end = bv.ix_ofs_bf[e] as f32
                            + (bv.nx_gr_bf[e] as f32 - 0.5) * intg as f32
                            - bw_offset;
                        if edge_end <= edge_start {
                            ixy_disjoint = 1;
                        }
                    } else {
                        //
                        // Other two cases, compare end of existing to start of missing
                        ind_edge = bv.ind_edge4[0][0];
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.nx_gr_bf[e] * intg - iblend[0]) / 2) as f32;
                        edge_end = bv.ix_grd_st_bf[e] as f32
                            + (bv.nx_gr_bf[e] as f32 - 0.5) * intg as f32
                            - bw_offset
                            + xin(bv, ip_xto)
                            - xin(bv, ip_from);
                        let ie = bv.in_edge[0][1];
                        edgeswap(bv, units, ie, 1, &mut ind_edge);
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.nx_gr_bf[e] * intg - iblend[0]) / 2) as f32;
                        edge_start = bv.ix_grd_st_bf[e] as f32 - intg as f32 / 2. + bw_offset;
                        if edge_end <= edge_start {
                            ixy_disjoint = 1;
                        }
                    }
                    //
                    // Similarly two cases for Y
                    if indp1 == 0 || indp2 == 0 {
                        ind_edge = bv.ind_edge4[1][0];
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.ny_gr_bf[e] * intg - iblend[1]) / 2) as f32;
                        edge_start =
                            bv.iy_ofs_bf[e] as f32 - intg as f32 / 2. + bw_offset + yin(bv, ip_yto)
                                - yin(bv, ip_from);
                        let ie = bv.in_edge[1][1];
                        edgeswap(bv, units, ie, 2, &mut ind_edge);
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.ny_gr_bf[e] * intg - iblend[1]) / 2) as f32;
                        edge_end = bv.iy_ofs_bf[e] as f32
                            + (bv.ny_gr_bf[e] as f32 - 0.5) * intg as f32
                            - bw_offset;
                        if edge_end <= edge_start {
                            ixy_disjoint = 2;
                        }
                    } else {
                        ind_edge = bv.ind_edge4[1][0];
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.ny_gr_bf[e] * intg - iblend[1]) / 2) as f32;
                        edge_end = bv.iy_grd_st_bf[e] as f32
                            + (bv.ny_gr_bf[e] as f32 - 0.5) * intg as f32
                            - bw_offset
                            + yin(bv, ip_yto)
                            - yin(bv, ip_from);
                        let ie = bv.in_edge[1][1];
                        edgeswap(bv, units, ie, 2, &mut ind_edge);
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.ny_gr_bf[e] * intg - iblend[1]) / 2) as f32;
                        edge_start = bv.iy_grd_st_bf[e] as f32 - intg as f32 / 2. + bw_offset;
                        if edge_end <= edge_start {
                            ixy_disjoint = 2;
                        }
                    }
                    //
                    // If one was found, compute the start and end for the half of
                    // the edge to be used on the other axis
                    if ixy_disjoint == 1 {
                        ind_edge = bv.ind_edge4[1][0];
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.ny_gr_bf[e] * intg - iblend[1]) / 2) as f32;
                        if indp1 == 0 || indp2 == 0 {
                            bv.start_skew =
                                bv.iy_ofs_bf[e] as f32 + ((bv.ny_gr_bf[e] - 1) * intg) as f32 / 2.;
                            bv.end_skew = bv.iy_ofs_bf[e] as f32
                                + (bv.ny_gr_bf[e] as f32 - 0.5) * intg as f32
                                - bw_offset;
                        } else {
                            bv.start_skew =
                                bv.iy_grd_st_bf[e] as f32 - intg as f32 / 2. + bw_offset;
                            bv.end_skew = bv.iy_grd_st_bf[e] as f32
                                + ((bv.ny_gr_bf[e] - 1) * intg) as f32 / 2.;
                        }
                    } else if ixy_disjoint == 2 {
                        ind_edge = bv.ind_edge4[0][0];
                        let e = (ind_edge - 1) as usize;
                        bw_offset = (0.max(bv.nx_gr_bf[e] * intg - iblend[0]) / 2) as f32;
                        if indp1 == 0 || indp3 == 0 {
                            bv.start_skew =
                                bv.ix_ofs_bf[e] as f32 + ((bv.nx_gr_bf[e] - 1) * intg) as f32 / 2.;
                            bv.end_skew = bv.ix_ofs_bf[e] as f32
                                + (bv.nx_gr_bf[e] as f32 - 0.5) * intg as f32
                                - bw_offset;
                        } else {
                            bv.start_skew =
                                bv.ix_grd_st_bf[e] as f32 - intg as f32 / 2. + bw_offset;
                            bv.end_skew = bv.ix_grd_st_bf[e] as f32
                                + ((bv.nx_gr_bf[e] - 1) * intg) as f32 / 2.;
                        }
                    }
                }
                //
                // Save for future use
                bv.lastp1 = pc(bv, indp1);
                bv.lastp2 = pc(bv, indp2);
                bv.lastp3 = pc(bv, indp3);
                bv.lastp4 = pc(bv, indp4);
                bv.lastxy_disjoint = ixy_disjoint;
            }
            //
            // Now if there is disjoint edge, modify edge and end fractions to
            // start at the midpoint of the piece
            let (ss, es) = (bv.start_skew, bv.end_skew);
            // `max(0., min(1., ...))` and `max(0., d)` (`blendmont.f90:3886-3927`):
            // `minss expr, 1.` then `maxss ., 0.`, and `maxss d, 0.`, in the
            // reference object.
            let frac = |v: f32| maxss(minss((v - ss) / (es - ss), 1.0f32), 0.0f32);
            if ixy_disjoint == 1 {
                if indp1 == 0 {
                    bv.edge_frac4[1][0] = frac(yin(bv, indp4));
                    db4 = maxss(yin(bv, indp4) - ss, 0.0f32);
                    eb4 = (db4 + 1.) / iblend[1] as f32;
                    if bv.debug {
                        let _ = writeln!(
                            out,
                            "s&e skew, y{}{}{} mod ef, db4, eb4{}{}{}",
                            f_edit(ss, 8, 1),
                            f_edit(es, 8, 1),
                            f_edit(yin(bv, indp4), 8, 1),
                            f_edit(bv.edge_frac4[1][0], 8, 4),
                            f_edit(db4, 8, 1),
                            f_edit(eb4, 8, 4)
                        );
                    }
                } else if indp2 == 0 {
                    bv.edge_frac4[1][0] = frac(yin(bv, indp3));
                    db3 = maxss(yin(bv, indp3) - ss, 0.0f32);
                    eb3 = (db3 + 1.) / iblend[1] as f32;
                } else if indp3 == 0 {
                    bv.edge_frac4[1][0] = frac(yin(bv, indp2));
                    dt2 = maxss(es - yin(bv, indp2), 0.0f32);
                    et2 = (dt2 + 1.) / iblend[1] as f32;
                } else {
                    bv.edge_frac4[1][0] = frac(yin(bv, indp1));
                    dt1 = maxss(es - yin(bv, indp1), 0.0f32);
                    et1 = (dt1 + 1.) / iblend[1] as f32;
                }
            } else if ixy_disjoint == 2 {
                if indp1 == 0 {
                    bv.edge_frac4[0][0] = frac(xin(bv, indp4));
                    dl4 = maxss(xin(bv, indp4) - ss, 0.0f32);
                    el4 = (dl4 + 1.) / iblend[0] as f32;
                } else if indp2 == 0 {
                    bv.edge_frac4[0][0] = frac(xin(bv, indp3));
                    dr3 = maxss(es - xin(bv, indp3), 0.0f32);
                    er3 = (dr3 + 1.) / iblend[0] as f32;
                } else if indp3 == 0 {
                    bv.edge_frac4[0][0] = frac(xin(bv, indp2));
                    dl2 = maxss(xin(bv, indp2) - ss, 0.0f32);
                    el2 = (dl2 + 1.) / iblend[0] as f32;
                } else {
                    bv.edge_frac4[0][0] = frac(xin(bv, indp1));
                    dr1 = maxss(es - xin(bv, indp1), 0.0f32);
                    // Fixed in translation (`BUGS.md`): the source has
                    // `er1 = (dr3 + 1.) / iblend(1)` (`blendmont.f90:3928`),
                    // the wrong variable -- `dr1` was just computed for it, as
                    // every sibling branch pairs `dXn` with `eXn`.
                    er1 = (dr1 + 1.) / iblend[0] as f32;
                }
            }
        }
        //
        // If there are 4 pieces, fx and fy are weighted sum of the f's in the
        // two overlapping edges.  The weights ex and ey are modified
        // fractional distances across the overlap zone.  First get the
        // distances to the borders of the overlap zone (dla, etc) and
        // use them to get absolute fractional distances across zone, ax in the
        // y direction and ay in the x direction.  They are used to make ex
        // slide from being a distance across the lower overlap to being
        // a distance across the upper overlap.  This gives continuity with the
        // edge in 2 or 3-piece cases
        //
        if bv.num_pieces == 4 {
            // `blendmont.f90:3944-3947`, operand order from the reference
            // object: only `dla` comes out reversed.
            let dla = minss(dl4, dl2);
            let dra = minss(dr1, dr3);
            let dba = minss(db3, db4);
            let dta = minss(dt1, dt2);
            let ax = dba / (dta + dba);
            let ay = dla / (dra + dla);
            bv.ex = ((1. - ay) * db3 + ay * db4) / ((1. - ay) * (dt1 + db3) + ay * (dt2 + db4));
            bv.ey = ((1. - ax) * dl2 + ax * dl4) / ((1. - ax) * (dr1 + dl2) + ax * (dr3 + dl4));
            fx = (1. - bv.ex) * f12 + bv.ex * f34;
            fy = (1. - bv.ey) * f13 + bv.ey * f24;
        } else if bv.num_pieces == 3 {
            //
            // Three-piece case is simple, only two edges
            //
            fx = bv.edge_frac4[0][0];
            fy = bv.edge_frac4[1][0];
        }
        //
        // weighting factors are a product of the two f's,
        // attenuated if necessary by fractional distance to
        // end of piece, then normalized to sum to 1.
        //
        // `blendmont.f90:3966-3969`, each `min` in the operand order of the
        // reference object (`-fverbose-asm`).
        bv.wll = minss(et1, 1. - fx) * minss(er1, 1. - fy);
        bv.wlr = minss(et2, fx) * minss(1. - fy, el2);
        bv.wul = minss(1. - fx, eb3) * minss(er3, fy);
        bv.wur = minss(eb4, fx) * minss(fy, el4);
        let w_sum = bv.wll + bv.wlr + bv.wul + bv.wur;
        if w_sum > 0. {
            bv.wll /= w_sum;
            bv.wlr /= w_sum;
            bv.wul /= w_sum;
            bv.wur /= w_sum;
        }
        //
        // count up active pieces implied by the w's
        //
        bv.n_active_p = 0;
        if bv.wll > 0. {
            bv.n_active_p += 1;
        }
        if bv.wlr > 0. {
            bv.n_active_p += 1;
        }
        if bv.wul > 0. {
            bv.n_active_p += 1;
        }
        if bv.wur > 0. {
            bv.n_active_p += 1;
        }
        //
        // filter out cross-corner 2-piece cases, pick the
        // piece where the point is most interior
        //
        if bv.n_active_p == 2 && bv.wll * bv.wur > 0. {
            // `blendmont.f90:3990, 3997`: `minss dr1, dt1` < `minss dl4, db4`,
            // and `minss dl2, dt2` < `minss db3, dr3`.
            if minss(dr1, dt1) < minss(dl4, db4) {
                bv.wll = 0.;
            } else {
                bv.wur = 0.;
            }
            bv.n_active_p = 1;
        } else if bv.n_active_p == 2 && bv.wul * bv.wlr > 0. {
            if minss(dl2, dt2) < minss(db3, dr3) {
                bv.wlr = 0.;
            } else {
                bv.wul = 0.;
            }
            bv.n_active_p = 1;
        }
        let _ = (dr1, ind_edge);
    } else {
        //
        // the one-piece case, avoid all that computation
        //
        bv.n_active_p = 1;
        bv.indp1234[0] = 1;
        bv.wll = 1.;
    }
    if bv.debug {
        let _ = writeln!(
            out,
            "Active{}  weights{}{}{}{}",
            i_edit(bv.n_active_p, 3),
            f_edit(bv.wll, 8, 4),
            f_edit(bv.wlr, 8, 4),
            f_edit(bv.wul, 8, 4),
            f_edit(bv.wur, 8, 4)
        );
    }
}

// ---------------------------------------------------------------------------
// gfortran runtime boundary: formatted and list-directed output editing and
// the runtime errors of a failed transfer.  Not translations of source units
// (as `gfortran_rt` is not): what libgfortran does for the edit descriptors
// and `print *` items this program uses.
// ---------------------------------------------------------------------------

/// `Iw` editing: right-justified in `w`; `w` asterisks if it does not fit.
fn i_edit(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// `Fw.d` editing: the value rounded to `d` decimals (round-half-even on the
/// exact binary value, as libgfortran does), a trailing point for `d = 0`,
/// the leading zero dropped when that is what makes it fit, `w` asterisks
/// when it still does not.  `NaN`/`Infinity` are right-justified.
fn f_edit(value: f32, w: usize, d: usize) -> String {
    if value.is_nan() {
        return format!("{:>w$}", "NaN");
    }
    if value.is_infinite() {
        let mut text = if value < 0. { "-Infinity" } else { "Infinity" };
        if text.len() > w {
            text = if value < 0. { "-Inf" } else { "Inf" };
        }
        if text.len() > w {
            return "*".repeat(w);
        }
        return format!("{text:>w$}");
    }
    let mut text = format!("{value:.d$}");
    if d == 0 {
        text.push('.');
    }
    if text.len() > w {
        if let Some(rest) = text.strip_prefix("0.") {
            text = format!(".{rest}");
        } else if let Some(rest) = text.strip_prefix("-0.") {
            text = format!("-.{rest}");
        }
    }
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// `Gw.d` editing with no exponent part, as `flib/image/header.rs` carries
/// it: `F(w-4).(d-k)` plus four blanks when the value rounded to `d`
/// significant digits has `0 <= k <= d` integer digits, else `Ew.d`.
fn g_edit(value: f32, w: usize, d: i32) -> String {
    if value.is_nan() {
        return format!("{:>w$}", "NaN");
    }
    if value.is_infinite() {
        let text = match (value < 0.0, w) {
            (false, 8..) => "Infinity",
            (false, _) => "Inf",
            (true, 9..) => "-Infinity",
            (true, _) => "-Inf",
        };
        return format!("{text:>w$}");
    }
    let magnitude = value.abs();
    let mut digits = String::new();
    let mut exponent = 1_i32;
    if magnitude != 0.0 {
        let scientific = format!("{:.*e}", (d - 1) as usize, magnitude);
        let (mantissa, power) = scientific.split_once('e').unwrap();
        digits = mantissa.replace('.', "");
        exponent = power.parse::<i32>().unwrap() + 1;
    }
    if (0..=d).contains(&exponent) {
        let mut text = format!("{:.*}", (d - exponent) as usize, value);
        if exponent == d {
            text.push('.');
        }
        format!("{:>1$}    ", text, w - 4)
    } else {
        format!(
            "{:>1$}",
            format!(
                "{}0.{}E{}{:02}",
                if value < 0.0 { "-" } else { "" },
                digits,
                if exponent < 0 { '-' } else { '+' },
                exponent.abs()
            ),
            w
        )
    }
}

/// A list-directed (`print *`) `integer*4` item: a blank separator (the
/// record's leading blank when it is the first item) and `I11`, so 12
/// columns.
fn ld_int(value: i32) -> String {
    format!("{value:>12}")
}

/// A list-directed `real*4` item (its blank separator, or the record's
/// leading blank, included): `G16.9` style (`F` form with
/// nine significant digits and four trailing blanks for magnitudes in
/// [0.1, 1e9), `E` form with a two-digit exponent otherwise); `NaN` and
/// `Infinity` right-justified in the 17 columns.
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

/// A list-directed `logical` item: a blank and `T` or `F`.
fn ld_logical(value: bool) -> &'static str {
    if value { " T" } else { " F" }
}

/// A formatted or list-directed `read` with no `END=`/`ERR=` that fails:
/// libgfortran reports it and stops with status 2.
fn read_runtime_error(err: ListReadError) -> ! {
    let _ = ImodFile::Stdout.flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}

/// An `open` with no `ERR=`/`IOSTAT=` that fails: libgfortran reports it and
/// stops with status 2.
fn open_runtime_error(iunit: i32, name: &str, err: std::io::Error) -> ! {
    let _ = ImodFile::Stdout.flush();
    eprintln!("Fortran runtime error: Cannot open file '{name}': {err}");
    let _ = iunit;
    exit(2);
}

/// `limInit` as an array extent.
const LIM_INIT_USIZE: usize = super::blendvars::LIM_INIT as usize;
