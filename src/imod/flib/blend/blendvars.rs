//! Translation of `IMOD/flib/blend/blendvars.f90`, the Fortran module shared by
//! the blendmont program units.
//!
//! # Design note — read this before translating any blend unit
//!
//! `module blendvars` holds every variable that `blendmont.f90`, `bsubs.f90`,
//! `shuffler.f90`, `solvescaling.f90` (and `reducemont.f90`, out of scope)
//! share through `use blendvars` (26 `use` sites; `nm` shows each as a
//! `__blendvars_MOD_*` symbol).  It becomes one struct, [`BlendVars`], with one
//! field per module variable in declaration order, following the `FortModel`
//! precedent (`flib/subrs/model/fortmodel.rs`).
//!
//! **Ownership and passing.**  `blendmont`'s main program creates one
//! `BlendVars::default()` and owns it for the run.  Every subroutine or
//! function whose source contains `use blendvars` takes `bv: &mut BlendVars`
//! (or `&BlendVars` if it only reads) as its **first** parameter, ahead of the
//! source's own dummy arguments; a unit without `use blendvars` does not take
//! it.  A Fortran module variable written `nxin` in the source is `bv.nxyz_in[0]`
//! in the translation (see the equivalences below).  Do not introduce a
//! `static mut` or `thread_local!`: the struct is the module.
//!
//! **Borrowing.**  The source routinely passes a module array as an actual
//! argument to a routine that also `use`s the module (e.g. `bsubs.f90`'s
//! `montXCorrEdge(brray, brray(maxbsiz / 2 + 1), ..., xcray, xdray, xeray, ...)`
//! at :2044).  Rust will not lend `bv` mutably twice.  Where the callee does
//! not itself touch the aliased variable through the module, split the borrow
//! at the call (`&mut bv.xcray`, `&mut bv.xdray` are disjoint fields).  Where
//! it does, take the array out with `std::mem::take`, pass it, and put it back
//! — and say so at the site.  Never clone: a clone silently drops the callee's
//! writes.
//!
//! **Initial values / SAVE.**  `blendvars.f90` has no `DATA` statements and no
//! declaration initialisers.  Module variables have static storage (implicit
//! SAVE, persisting for the whole run) and gfortran places uninitialised ones
//! in `.bss`, so every scalar and fixed array starts at zero / `.false.`, and
//! every allocatable starts unallocated.  [`Default`] reproduces exactly that:
//! zeros, `false`, empty `Vec`s.  A program unit that holds a *local* with a
//! declaration initialiser gets SAVE semantics there (see `CLAUDE.md`, "A
//! Fortran declaration initialiser makes the variable `SAVE`") — that lives in
//! the unit's own translation, not here.
//!
//! **Parameters** are `pub const` of Fortran's default integer kind, `i32`;
//! array bounds spell `MAX_IN_PC as usize`.
//!
//! **Types.**  `integer*4` → `i32`, `integer(kind = 8)` → `i64`, `real*4` →
//! `f32`, `logical` → `bool`.  `complex * 8` allocatables (`xcray`, `xdray`,
//! `xeray`) are `Vec<f32>` holding interleaved (re, im) pairs, **twice** the
//! Fortran element count: `allocate(xcray(idimc / 2))` is
//! `vec![0.0; 2 * (idimc / 2)]`.  Every use of them is sequence association
//! with a `real` dummy (`montXCorrEdge`'s `lowerPad: &mut [f32]`,
//! `libcfshr/montagexcorr.rs`), so the real view is the one the callee needs.
//!
//! **Indexing (1-based source, 0-based Rust).**  All arrays are stored 0-based
//! and a source subscript `a(i)` is written `a[(i - 1) as usize]`, with one
//! exception: `inPiece(0:maxInPc)` has lower bound 0, so it is stored with
//! `maxInPc + 1` elements and `inPiece(i)` is `in_piece[i as usize]` — no
//! shift.
//!
//! **Fixed-size multi-dimensional arrays** are nested Rust arrays with the
//! Fortran dimensions **reversed**, which is column-major memory order:
//! `integer*4 inEdge(maxInPc,2)` is `[[i32; 100]; 2]` and `inEdge(i, j)` is
//! `in_edge[j - 1][i - 1]`; `real*4 ginv(2,3)` is `[[f32; 2]; 3]` and
//! `ginv(i, j)` is `ginv[j - 1][i - 1]`.  When the source passes the whole
//! array (or `ginv(1,1)`) to a routine that takes it as a flat `real` array —
//! `xfinvert`, `xfmult`, `xfapply` take `&[f32]` with a row count
//! (`libcfshr/linearxforms.rs`) — pass `ginv.as_flattened()`.
//!
//! **Allocatable multi-dimensional arrays** are one flat column-major `Vec`
//! plus a companion `<name>_ext: [usize; R]` field holding the extents given to
//! `allocate`.  The companion is the Fortran array descriptor: element
//! addressing uses the extents *at allocation time*, not the current value of
//! whatever variable was used to compute them — and some were computed from
//! `blendmont` locals the module does not hold (`denAbuf(ix, maxSecEdges)`,
//! `blendmont.f90:1546-1549`, is indexed by `solvescaling.f90:106` and
//! `bsubs.f90:812` with no other record of `ix`).  So:
//! `hinv(i, j, k)` is
//! `hinv[(i - 1) + hinv_ext[0] * ((j - 1) + hinv_ext[1] * (k - 1))]`, and
//! `allocate(hinv(2, 3, limNpc))` sets `hinv_ext = [2, 3, limNpc as usize]`
//! and `hinv = vec![0.0; 2 * 3 * limNpc as usize]`.  An array-section actual
//! argument such as `hinv(1,1,ipc)` or `dxGrBf(1,1,i)` is the slice starting at
//! that element's flat offset — which is exactly Fortran's sequence
//! association, and why a flat `Vec` (not `Vec<Vec<_>>`) is the representation.
//! `deallocate` is `v = Vec::new()` (and the `_ext` may be left as is: nothing
//! reads a deallocated array's shape).  `allocated(x)` is `!x.is_empty()`
//! except for a zero-size allocation, which no `allocate` in these units makes.
//!
//! **`allocate` fills.**  Fortran `allocate` does not initialise; a `Vec` is
//! zeroed.  Where native output would depend on reading an element never
//! written, the translation is deterministic and native is not (`NATIVE.md`
//! §4) — record it, do not imitate it.
//!
//! **Equivalences** (`blendvars.f90:36-39`, `:122-123`) are storage aliases,
//! so only the array is a field; the scalar names map to elements.  Do **not**
//! add getter/setter methods for them: assigning through an accessor's
//! return value is a silent no-op (`CLAUDE.md`, "Assigning to the result of an
//! accessor").  Write the element:
//!
//! | source scalar | field element |
//! |---|---|
//! | `nxin`, `nyin`, `nzin` | `nxyz_in[0]`, `nxyz_in[1]`, `nxyz_in[2]` |
//! | `nxBin`, `nyBin`, `nzBin` | `nxyz_bin[0]`, `[1]`, `[2]` |
//! | `nxOut`, `nyOut`, `nzOut` | `nxyz_out[0]`, `[1]`, `[2]` |
//! | `nXoverlap`, `nyOverlap` | `n_overlap[0]`, `n_overlap[1]` |
//! | `indp1`, `indp2`, `indp3`, `indp4` | `indp1234[0]`, `[1]`, `[2]`, `[3]` |
//!
//! Fortran names are case-insensitive: `maxbsiz` in `bsubs.f90` is `maxBsiz`
//! here (`max_bsiz`), `iedgeZbase` at `blendmont.f90:1911` is `iEdgeZbase`.

/// Original: `parameter (ifastSiz = 32)` (`blendvars.f90:12`).
pub const IFAST_SIZ: i32 = 32;
/// Original: `parameter (maxBin = 8)` (`blendvars.f90:12`).
pub const MAX_BIN: i32 = 8;
/// Original: `parameter (limInit = 2500000)` (`blendvars.f90:12`).
pub const LIM_INIT: i32 = 2500000;
/// Original: `parameter (maxDistNear = 5)` (`blendvars.f90:13`).
pub const MAX_DIST_NEAR: i32 = 5;
/// Original: `parameter (maxInPc = 100)` (`blendvars.f90:13`).
pub const MAX_IN_PC: i32 = 100;
/// Original: `parameter (maxPcNear = (2 * maxDistNear + 1) * (2 * maxDistNear + 1))`
/// (`blendvars.f90:14`) — 121.
pub const MAX_PC_NEAR: i32 = (2 * MAX_DIST_NEAR + 1) * (2 * MAX_DIST_NEAR + 1);
/// Original: `parameter (maxUseEdge = 100)` (`blendvars.f90:15`).
pub const MAX_USE_EDGE: i32 = 100;
/// Original: `parameter (memMaximum = 2000000000)` (`blendvars.f90:68`).
pub const MEM_MAXIMUM: i32 = 2000000000;
/// Original: `parameter (memMinimum = 32000000)` (`blendvars.f90:69`).
pub const MEM_MINIMUM: i32 = 32000000;
/// Original: `parameter (memPreferred = 340000000)` (`blendvars.f90:69`).
pub const MEM_PREFERRED: i32 = 340000000;
/// Original: `parameter (limEdgBf = 20)` (`blendvars.f90:74`).
pub const LIM_EDG_BF: i32 = 20;

/// Original: `module blendvars` (`blendvars.f90:3-128`).
///
/// Field order is declaration order.  Multi-dimensional allocatables carry a
/// `<name>_ext` companion immediately after them (see the module note).
#[derive(Clone, Debug)]
pub struct BlendVars {
    // ---- blendvars.f90:16-17
    /// Original: `real*4, allocatable :: array(:)` (`blendvars.f90:16`).
    pub array: Vec<f32>,
    /// Original: `real*4, allocatable :: brray(:)` (`blendvars.f90:16`).
    pub brray: Vec<f32>,
    /// Original: `complex * 8, allocatable :: xcray(:)` (`blendvars.f90:17`) —
    /// interleaved (re, im), `2 *` the Fortran size.
    pub xcray: Vec<f32>,
    /// Original: `complex * 8, allocatable :: xdray(:)` (`blendvars.f90:17`) —
    /// interleaved (re, im), `2 *` the Fortran size.
    pub xdray: Vec<f32>,
    /// Original: `complex * 8, allocatable :: xeray(:)` (`blendvars.f90:17`) —
    /// interleaved (re, im), `2 *` the Fortran size.
    pub xeray: Vec<f32>,

    // ---- blendvars.f90:19-21
    /// Original: `integer*4 nxyzIn(3)` (`blendvars.f90:19`); `nxin`, `nyin`,
    /// `nzin` are equivalenced to its elements (`:36`).
    pub nxyz_in: [i32; 3],
    /// Original: `integer*4 nxyzOut(3)` (`blendvars.f90:19`); `nxOut`,
    /// `nyOut`, `nzOut` are equivalenced to its elements (`:38`).
    pub nxyz_out: [i32; 3],
    /// Original: `integer*4 idimc` (`blendvars.f90:19`).
    pub idimc: i32,
    /// Original: `integer*4 nxyzBin(3)` (`blendvars.f90:20`); `nxBin`,
    /// `nyBin`, `nzBin` are equivalenced to its elements (`:37`).
    pub nxyz_bin: [i32; 3],
    /// Original: `integer*4 limNpc` (`blendvars.f90:20`).
    pub lim_npc: i32,
    /// Original: `integer*4 limSect` (`blendvars.f90:20`).
    pub lim_sect: i32,
    /// Original: `integer*4 limEdge` (`blendvars.f90:20`).
    pub lim_edge: i32,
    /// Original: `integer*4 maxSiz` (`blendvars.f90:20`).
    pub max_siz: i32,
    /// Original: `integer*4 nOverlap(2)` (`blendvars.f90:21`); `nXoverlap`,
    /// `nyOverlap` are equivalenced to its elements (`:39`).
    pub n_overlap: [i32; 2],
    /// Original: `integer*4 nedge(2)` (`blendvars.f90:21`).
    pub nedge: [i32; 2],
    /// Original: `integer*4 maxLineLength` (`blendvars.f90:21`).
    pub max_line_length: i32,
    /// Original: `integer*4 maxBsiz` (`blendvars.f90:21`).
    pub max_bsiz: i32,

    // ---- blendvars.f90:22-30
    /// Original: `integer*4, allocatable :: ixPcList(:)` (`blendvars.f90:22`) —
    /// piece coords in x.
    pub ix_pc_list: Vec<i32>,
    /// Original: `integer*4, allocatable :: iyPcList(:)` (`blendvars.f90:22`) —
    /// piece coords in y.
    pub iy_pc_list: Vec<i32>,
    /// Original: `integer*4, allocatable :: izPcList(:)` (`blendvars.f90:23`) —
    /// section #.
    pub iz_pc_list: Vec<i32>,
    /// Original: `integer*4, allocatable :: negList(:)` (`blendvars.f90:23`) —
    /// negative #.
    pub neg_list: Vec<i32>,
    /// Original: `integer*4, allocatable :: limDataLo(:,:,:)`
    /// (`blendvars.f90:24`); allocated `(limVar, 2, 2)` (`blendmont.f90:1770`).
    pub lim_data_lo: Vec<i32>,
    /// Allocation extents of `limDataLo`.
    pub lim_data_lo_ext: [usize; 3],
    /// Original: `integer*4, allocatable :: limDataHi(:,:,:)`
    /// (`blendvars.f90:24`); allocated `(limVar, 2, 2)`.
    pub lim_data_hi: Vec<i32>,
    /// Allocation extents of `limDataHi`.
    pub lim_data_hi_ext: [usize; 3],
    /// Original: `integer*4, allocatable :: iedgeLower(:,:)`
    /// (`blendvars.f90:25`); allocated `(limNpc, 2)` (`blendmont.f90:384`).
    pub iedge_lower: Vec<i32>,
    /// Allocation extents of `iedgeLower`.
    pub iedge_lower_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: iedgeUpper(:,:)`
    /// (`blendvars.f90:25`); allocated `(limNpc, 2)`.
    pub iedge_upper: Vec<i32>,
    /// Allocation extents of `iedgeUpper`.
    pub iedge_upper_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: limDataInd(:)` (`blendvars.f90:25`).
    pub lim_data_ind: Vec<i32>,
    /// Original: `integer*4, allocatable :: ipieceLower(:,:)`
    /// (`blendvars.f90:26`); allocated `(limEdge, 2)` (`blendmont.f90:479`).
    pub ipiece_lower: Vec<i32>,
    /// Allocation extents of `ipieceLower`.
    pub ipiece_lower_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: ipieceUpper(:,:)`
    /// (`blendvars.f90:26`); allocated `(limEdge, 2)`.
    pub ipiece_upper: Vec<i32>,
    /// Allocation extents of `ipieceUpper`.
    pub ipiece_upper_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: ibufEdge(:,:)`
    /// (`blendvars.f90:26`); allocated `(limEdge, 2)`.
    pub ibuf_edge: Vec<i32>,
    /// Allocation extents of `ibufEdge`.
    pub ibuf_edge_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: ifSkipEdge(:,:)`
    /// (`blendvars.f90:27`); allocated `(limEdge, 2)`.
    pub if_skip_edge: Vec<i32>,
    /// Allocation extents of `ifSkipEdge`.
    pub if_skip_edge_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: maxSDtoEdgeNum(:)` (`blendvars.f90:27`).
    pub max_sd_to_edge_num: Vec<i32>,
    /// Original: `integer*4, allocatable :: maxSDtoIXYofEdge(:)` (`blendvars.f90:27`).
    pub max_sd_to_ixy_of_edge: Vec<i32>,
    /// Original: `real*4, allocatable :: trimmedMaxSDs(:)` (`blendvars.f90:28`).
    pub trimmed_max_sds: Vec<f32>,
    /// Original: `real*4, allocatable :: alternDisps(:)` (`blendvars.f90:28`).
    pub altern_disps: Vec<f32>,
    /// Original: `integer*4, allocatable :: iedgeAltFixed(:)` (`blendvars.f90:29`).
    pub iedge_alt_fixed: Vec<i32>,
    /// Original: `integer*4, allocatable :: iedgeLowWeight(:)` (`blendvars.f90:29`).
    pub iedge_low_weight: Vec<i32>,
    /// Original: `integer*4 numAltFixed` (`blendvars.f90:30`).
    pub num_alt_fixed: i32,
    /// Original: `integer*4 numLowWeight` (`blendvars.f90:30`).
    pub num_low_weight: i32,

    // ---- blendvars.f90:31-35
    /// Original: `integer*4 indent(2)` (`blendvars.f90:31`) — minimum indent
    /// short & long.
    pub indent: [i32; 2],
    /// Original: `integer*4 intGrid(2)` (`blendvars.f90:32`) — grid interval
    /// short & long.
    pub int_grid: [i32; 2],
    /// Original: `integer*4 iboxSiz(2)` (`blendvars.f90:33`) — box size short
    /// & long.
    pub ibox_siz: [i32; 2],
    /// Original: `integer*4 nxGrid(2)` (`blendvars.f90:34`) — # of grid points
    /// for x&y edges.
    pub nx_grid: [i32; 2],
    /// Original: `integer*4 nyGrid(2)` (`blendvars.f90:34`).
    pub ny_grid: [i32; 2],
    /// Original: `real*4 edgeLoNear(2)` (`blendvars.f90:35`) — limits for
    /// values near edges.
    pub edge_lo_near: [f32; 2],
    /// Original: `real*4 edgeHiNear(2)` (`blendvars.f90:35`).
    pub edge_hi_near: [f32; 2],

    // ---- blendvars.f90:40-44
    /// Original: `integer*4 npcList` (`blendvars.f90:40`).
    pub npc_list: i32,
    /// Original: `integer*4 minXpiece` (`blendvars.f90:40`).
    pub min_xpiece: i32,
    /// Original: `integer*4 minYpiece` (`blendvars.f90:40`).
    pub min_ypiece: i32,
    /// Original: `integer*4 nxPieces` (`blendvars.f90:40`).
    pub nx_pieces: i32,
    /// Original: `integer*4 nyPieces` (`blendvars.f90:40`).
    pub ny_pieces: i32,
    /// Original: `integer*4 interpOrder` (`blendvars.f90:40`).
    pub interp_order: i32,
    /// Original: `integer*4 numMaxSDs` (`blendvars.f90:40`).
    pub num_max_sds: i32,
    /// Original: `real*4 dmean` (`blendvars.f90:41`).
    pub dmean: f32,
    /// Original: `real*4 dfill` (`blendvars.f90:41`).
    pub dfill: f32,
    /// Original: `integer*4 izUseDefLow` (`blendvars.f90:42`).
    pub iz_use_def_low: i32,
    /// Original: `integer*4 izUseDefHigh` (`blendvars.f90:42`).
    pub iz_use_def_high: i32,
    /// Original: `integer*4 numUseEdge` (`blendvars.f90:42`).
    pub num_use_edge: i32,
    /// Original: `integer*4 ixyUseEdge(maxUseEdge)` (`blendvars.f90:42`).
    pub ixy_use_edge: [i32; MAX_USE_EDGE as usize],
    /// Original: `integer*4 ixFrmUseEdge(maxUseEdge)` (`blendvars.f90:43`).
    pub ix_frm_use_edge: [i32; MAX_USE_EDGE as usize],
    /// Original: `integer*4 iyFrmUseEdge(maxUseEdge)` (`blendvars.f90:43`).
    pub iy_frm_use_edge: [i32; MAX_USE_EDGE as usize],
    /// Original: `integer*4 izLowUse(maxUseEdge)` (`blendvars.f90:44`).
    pub iz_low_use: [i32; MAX_USE_EDGE as usize],
    /// Original: `integer*4 izHighUse(maxUseEdge)` (`blendvars.f90:44`).
    pub iz_high_use: [i32; MAX_USE_EDGE as usize],
    /// Original: `integer*4 lastWritten(2)` (`blendvars.f90:44`).
    pub last_written: [i32; 2],

    // ---- blendvars.f90:46-56
    /// Original: `integer*4 numPieces` (`blendvars.f90:46`) — # of pieces point
    /// is in.
    pub num_pieces: i32,
    /// Original: `integer*4 inPiece(0:maxInPc)` (`blendvars.f90:47`) — piece #
    /// of pieces point in.  Lower bound 0: `inPiece(i)` is `in_piece[i]`, no
    /// shift (`blendmont.f90:202` writes `inPiece(0)`).
    pub in_piece: [i32; MAX_IN_PC as usize + 1],
    /// Original: `real*4 xInPiece(maxInPc)` (`blendvars.f90:48`) — x coordinate
    /// within piece.
    pub x_in_piece: [f32; MAX_IN_PC as usize],
    /// Original: `real*4 yInPiece(maxInPc)` (`blendvars.f90:48`) — y coordinate
    /// within piece.
    pub y_in_piece: [f32; MAX_IN_PC as usize],
    /// Original: `integer*4 inpXframe(maxInPc)` (`blendvars.f90:50`) — frame
    /// number of each piece in X.
    pub inp_xframe: [i32; MAX_IN_PC as usize],
    /// Original: `integer*4 inpYframe(maxInPc)` (`blendvars.f90:50`) — frame
    /// number of each piece in Y.
    pub inp_yframe: [i32; MAX_IN_PC as usize],
    /// Original: `integer*4 maxXframe` (`blendvars.f90:51`).
    pub max_xframe: i32,
    /// Original: `integer*4 maxYframe` (`blendvars.f90:51`).
    pub max_yframe: i32,
    /// Original: `integer*4 minXframe` (`blendvars.f90:51`).
    pub min_xframe: i32,
    /// Original: `integer*4 minYframe` (`blendvars.f90:51`).
    pub min_yframe: i32,
    /// Original: `integer*4 numEdges(2)` (`blendvars.f90:52`) — number of edges
    /// in x and y.
    pub num_edges: [i32; 2],
    /// Original: `integer*4 inEdge(maxInPc,2)` (`blendvars.f90:53`) — edge # of
    /// edges point is in.  `inEdge(i, j)` is `in_edge[j - 1][i - 1]`.
    pub in_edge: [[i32; MAX_IN_PC as usize]; 2],
    /// Original: `integer*4 inEdLower(maxInPc,2)` (`blendvars.f90:55`) — list
    /// index of piece on lower side of edge.
    pub in_ed_lower: [[i32; MAX_IN_PC as usize]; 2],
    /// Original: `integer*4 inEdUpper(maxInPc,2)` (`blendvars.f90:55`) — list
    /// index of piece on upper side of edge.
    pub in_ed_upper: [[i32; MAX_IN_PC as usize]; 2],
    /// Original: `integer*4 numPcNear` (`blendvars.f90:56`).
    pub num_pc_near: i32,
    /// Original: `integer*4 idxPcNear(maxPcNear)` (`blendvars.f90:56`).
    pub idx_pc_near: [i32; MAX_PC_NEAR as usize],
    /// Original: `integer*4 idyPcNear(maxPcNear)` (`blendvars.f90:56`).
    pub idy_pc_near: [i32; MAX_PC_NEAR as usize],

    // ---- blendvars.f90:58-66
    /// Original: `integer*4, allocatable :: mapPiece(:,:)` (`blendvars.f90:58`) —
    /// map of pieces in this section; allocated `(nxPieces, nyPieces)`
    /// (`blendmont.f90:1768`).
    pub map_piece: Vec<i32>,
    /// Allocation extents of `mapPiece`.
    pub map_piece_ext: [usize; 2],
    /// Original: `logical, allocatable :: anyDisjoint(:,:)` (`blendvars.f90:59`)
    /// — if any corner is disjoint; allocated `(nxPieces, nyPieces)`.
    pub any_disjoint: Vec<bool>,
    /// Allocation extents of `anyDisjoint`.
    pub any_disjoint_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: mapDisjoint(:,:)`
    /// (`blendvars.f90:60`) — type of disjoint edge (X/Y); allocated
    /// `(nxPieces, nyPieces)`.
    pub map_disjoint: Vec<i32>,
    /// Allocation extents of `mapDisjoint`.
    pub map_disjoint_ext: [usize; 2],
    /// Original: `logical doGxforms` (`blendvars.f90:61`) — if doing g's.
    pub do_gxforms: bool,
    /// Original: `logical multng` (`blendvars.f90:61`) — if negs in sect.
    pub multng: bool,
    /// Original: `logical limitData` (`blendvars.f90:62`) — if limiting data in
    /// X / Y.
    pub limit_data: bool,
    /// Original: `real*4 hxCen` (`blendvars.f90:63`) — coord of center of input
    /// frame.
    pub hx_cen: f32,
    /// Original: `real*4 hyCen` (`blendvars.f90:63`).
    pub hy_cen: f32,
    /// Original: `real*4 gxCen` (`blendvars.f90:64`) — coord of output image
    /// center.
    pub gx_cen: f32,
    /// Original: `real*4 gyCen` (`blendvars.f90:64`).
    pub gy_cen: f32,
    /// Original: `real*4 ginv(2,3)` (`blendvars.f90:65`) — inverse of g and h
    /// xforms.  `ginv(i, j)` is `ginv[j - 1][i - 1]`; flat view
    /// `ginv.as_flattened()`.
    pub ginv: [[f32; 2]; 3],
    /// Original: `real*4, allocatable :: hinv(:,:,:)` (`blendvars.f90:66`);
    /// allocated `(2, 3, limNpc)` (`blendmont.f90:385`).
    pub hinv: Vec<f32>,
    /// Allocation extents of `hinv`.
    pub hinv_ext: [usize; 3],
    /// Original: `real*4, allocatable :: htmp(:,:,:)` (`blendvars.f90:66`);
    /// allocated `(2, 3, limNpc)`.
    pub htmp: Vec<f32>,
    /// Allocation extents of `htmp`.
    pub htmp_ext: [usize; 3],

    // ---- blendvars.f90:70-72
    /// Original: `integer*4, allocatable :: memIndex(:)` (`blendvars.f90:70`).
    pub mem_index: Vec<i32>,
    /// Original: `integer*4, allocatable :: izMemList(:)` (`blendvars.f90:70`).
    pub iz_mem_list: Vec<i32>,
    /// Original: `integer*4, allocatable :: lastUsed(:)` (`blendvars.f90:70`).
    pub last_used: Vec<i32>,
    /// Original: `integer*4 maxLoad` (`blendvars.f90:71`).
    pub max_load: i32,
    /// Original: `integer*4 juseCount` (`blendvars.f90:71`).
    pub juse_count: i32,
    /// Original: `integer*4 ilistz` (`blendvars.f90:71`).
    pub ilistz: i32,
    /// Original: `integer*4 memLim` (`blendvars.f90:71`).
    pub mem_lim: i32,
    /// Original: `integer(kind = 8) npixIn` (`blendvars.f90:72`).
    pub npix_in: i64,

    // ---- blendvars.f90:75-82
    /// Original: `integer*4 iedgBfList(limEdgBf)` (`blendvars.f90:75`).
    pub iedg_bf_list: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 ixyBfList(limEdgBf)` (`blendvars.f90:75`).
    pub ixy_bf_list: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 lasEdgUse(limEdgBf)` (`blendvars.f90:75`).
    pub las_edg_use: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 iunEdge(2)` (`blendvars.f90:76`).
    pub iun_edge: [i32; 2],
    /// Original: `integer*4 ixgDim` (`blendvars.f90:76`).
    pub ixg_dim: i32,
    /// Original: `integer*4 iygDim` (`blendvars.f90:76`).
    pub iyg_dim: i32,
    /// Original: `integer*4 nxGrBf(limEdgBf)` (`blendvars.f90:77`).
    pub nx_gr_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 nyGrBf(limEdgBf)` (`blendvars.f90:77`).
    pub ny_gr_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 ixGrdStBf(limEdgBf)` (`blendvars.f90:77`).
    pub ix_grd_st_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 iyGrdStBf(limEdgBf)` (`blendvars.f90:78`).
    pub iy_grd_st_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 ixOfsBf(limEdgBf)` (`blendvars.f90:78`).
    pub ix_ofs_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 iyOfsBf(limEdgBf)` (`blendvars.f90:78`).
    pub iy_ofs_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 intGrCopy(2)` (`blendvars.f90:79`).
    pub int_gr_copy: [i32; 2],
    /// Original: `integer*4 intXgrBf(limEdgBf)` (`blendvars.f90:79`).
    pub int_xgr_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `integer*4 intYgrBf(limEdgBf)` (`blendvars.f90:79`).
    pub int_ygr_bf: [i32; LIM_EDG_BF as usize],
    /// Original: `real*4, allocatable :: dxGrBf(:,:,:)` (`blendvars.f90:80`);
    /// allocated `(ixgDim, iygDim, limEdgBf)` (`blendmont.f90:1771`).
    pub dx_gr_bf: Vec<f32>,
    /// Allocation extents of `dxGrBf`.
    pub dx_gr_bf_ext: [usize; 3],
    /// Original: `real*4, allocatable :: dyGrBf(:,:,:)` (`blendvars.f90:80`);
    /// allocated `(ixgDim, iygDim, limEdgBf)`.
    pub dy_gr_bf: Vec<f32>,
    /// Allocation extents of `dyGrBf`.
    pub dy_gr_bf_ext: [usize; 3],
    /// Original: `real*4, allocatable :: ddenGrBf(:,:,:)` (`blendvars.f90:80`);
    /// allocated `(ixgDim, iygDim, limEdgBf)`.
    pub dden_gr_bf: Vec<f32>,
    /// Allocation extents of `ddenGrBf`.
    pub dden_gr_bf_ext: [usize; 3],
    /// Original: `real*4, allocatable :: dxGrid(:,:)` (`blendvars.f90:81`);
    /// allocated `(ixgDim, iygDim)`.
    pub dx_grid: Vec<f32>,
    /// Allocation extents of `dxGrid`.
    pub dx_grid_ext: [usize; 2],
    /// Original: `real*4, allocatable :: dyGrid(:,:)` (`blendvars.f90:81`);
    /// allocated `(ixgDim, iygDim)`.
    pub dy_grid: Vec<f32>,
    /// Allocation extents of `dyGrid`.
    pub dy_grid_ext: [usize; 2],
    /// Original: `real*4, allocatable :: ddenGrid(:,:)` (`blendvars.f90:81`);
    /// allocated `(ixgDim, iygDim)`.
    pub dden_grid: Vec<f32>,
    /// Allocation extents of `ddenGrid`.
    pub dden_grid_ext: [usize; 2],
    /// Original: `real*4, allocatable :: sdGrid(:,:)` (`blendvars.f90:81`);
    /// allocated `(ixgDim, iygDim)`.
    pub sd_grid: Vec<f32>,
    /// Allocation extents of `sdGrid`.
    pub sd_grid_ext: [usize; 2],
    /// Original: `integer*4 jusEdgCt` (`blendvars.f90:82`).
    pub jus_edg_ct: i32,
    /// Original: `integer*4 needByteSwap` (`blendvars.f90:82`).
    pub need_byte_swap: i32,
    /// Original: `integer*4 izUnsmoothedPatch` (`blendvars.f90:82`).
    pub iz_unsmoothed_patch: i32,
    /// Original: `integer*4 izSmoothedPatch` (`blendvars.f90:82`).
    pub iz_smoothed_patch: i32,

    // ---- blendvars.f90:84-91
    /// Original: `integer*4 iedgeCurBase(2)` (`blendvars.f90:84`).
    pub iedge_cur_base: [i32; 2],
    /// Original: `integer*4 iunDens(2)` (`blendvars.f90:84`).
    pub iun_dens: [i32; 2],
    /// Original: `integer*4 ixDimDenBuf` (`blendvars.f90:84`).
    pub ix_dim_den_buf: i32,
    /// Original: `integer*4 iApplyDensScaling` (`blendvars.f90:84`).
    pub i_apply_dens_scaling: i32,
    /// Original: `integer*4 iDensFromEdges` (`blendvars.f90:85`).
    pub i_dens_from_edges: i32,
    /// Original: `integer*4 multByFlatfield` (`blendvars.f90:85`).
    pub mult_by_flatfield: i32,
    /// Original: `integer*4 iDenSample` (`blendvars.f90:86`).
    pub i_den_sample: i32,
    /// Original: `integer*4 intervalDen(2)` (`blendvars.f90:86`).
    pub interval_den: [i32; 2],
    /// Original: `integer*4 nxDenGrid(2)` (`blendvars.f90:86`).
    pub nx_den_grid: [i32; 2],
    /// Original: `integer*4 nyDenGrid(2)` (`blendvars.f90:86`).
    pub ny_den_grid: [i32; 2],
    /// Original: `integer*4, allocatable :: iEdgeZbase(:,:)`
    /// (`blendvars.f90:87`); allocated `(numSect + 1, 2)` from a `blendmont`
    /// local (`blendmont.f90:480`).
    pub i_edge_zbase: Vec<i32>,
    /// Allocation extents of `iEdgeZbase`.
    pub i_edge_zbase_ext: [usize; 2],
    /// Original: `integer*4, allocatable :: ixDenOffset(:)` (`blendvars.f90:87`).
    pub ix_den_offset: Vec<i32>,
    /// Original: `integer*4, allocatable :: iyDenOffset(:)` (`blendvars.f90:87`).
    pub iy_den_offset: Vec<i32>,
    /// Original: `integer*4, allocatable :: nxDenBuf(:)` (`blendvars.f90:88`).
    pub nx_den_buf: Vec<i32>,
    /// Original: `integer*4, allocatable :: nyDenBuf(:)` (`blendvars.f90:88`).
    pub ny_den_buf: Vec<i32>,
    /// Original: `integer*4, allocatable :: ixDenStart(:)` (`blendvars.f90:88`).
    pub ix_den_start: Vec<i32>,
    /// Original: `integer*4, allocatable :: iyDenStart(:)` (`blendvars.f90:88`).
    pub iy_den_start: Vec<i32>,
    /// Original: `real*4, allocatable :: denAbuf(:,:)` (`blendvars.f90:89`);
    /// allocated `(ix, maxSecEdges)` from `blendmont` locals
    /// (`blendmont.f90:1546-1549`).
    pub den_abuf: Vec<f32>,
    /// Allocation extents of `denAbuf`.
    pub den_abuf_ext: [usize; 2],
    /// Original: `real*4, allocatable :: denBbuf(:,:)` (`blendvars.f90:89`);
    /// allocated `(ix, maxSecEdges)`.
    pub den_bbuf: Vec<f32>,
    /// Allocation extents of `denBbuf`.
    pub den_bbuf_ext: [usize; 2],
    /// Original: `real*4, allocatable :: pieceScaling(:,:)`
    /// (`blendvars.f90:89`); allocated `(nxPieces, nyPieces)`.
    pub piece_scaling: Vec<f32>,
    /// Allocation extents of `pieceScaling`.
    pub piece_scaling_ext: [usize; 2],
    /// Original: `real*4, allocatable :: deltaDenBuf(:,:)`
    /// (`blendvars.f90:89`); allocated `(ix, maxSecEdges)`.
    pub delta_den_buf: Vec<f32>,
    /// Allocation extents of `deltaDenBuf`.
    pub delta_den_buf_ext: [usize; 2],
    /// Original: `real*4, allocatable :: denSolution(:)` (`blendvars.f90:90`).
    pub den_solution: Vec<f32>,
    /// Original: `real*4, allocatable :: flatfield(:)` (`blendvars.f90:90`).
    pub flatfield: Vec<f32>,
    /// Original: `real*4 xGradScaling` (`blendvars.f90:91`).
    pub x_grad_scaling: f32,
    /// Original: `real*4 yGradScaling` (`blendvars.f90:91`).
    pub y_grad_scaling: f32,
    /// Original: `real*4 denZeroBase` (`blendvars.f90:91`).
    pub den_zero_base: f32,
    /// Original: `real*4 xBaseGradScale` (`blendvars.f90:91`).
    pub x_base_grad_scale: f32,
    /// Original: `real*4 yBaseGradScale` (`blendvars.f90:91`).
    pub y_base_grad_scale: f32,

    // ---- blendvars.f90:93-100
    /// Original: `real*4, allocatable :: distDx(:,:)` (`blendvars.f90:93`);
    /// allocated `(lmField, lmField)` (`blendmont.f90:1649`).
    pub dist_dx: Vec<f32>,
    /// Allocation extents of `distDx`.
    pub dist_dx_ext: [usize; 2],
    /// Original: `real*4, allocatable :: distDy(:,:)` (`blendvars.f90:93`);
    /// allocated `(lmField, lmField)`.
    pub dist_dy: Vec<f32>,
    /// Allocation extents of `distDy`.
    pub dist_dy_ext: [usize; 2],
    /// Original: `real*4, allocatable :: fieldDx(:,:,:)` (`blendvars.f90:93`);
    /// allocated `(lmField, lmField, maxFields)`.
    pub field_dx: Vec<f32>,
    /// Allocation extents of `fieldDx`.
    pub field_dx_ext: [usize; 3],
    /// Original: `real*4, allocatable :: fieldDy(:,:,:)` (`blendvars.f90:93`);
    /// allocated `(lmField, lmField, maxFields)`.
    pub field_dy: Vec<f32>,
    /// Allocation extents of `fieldDy`.
    pub field_dy_ext: [usize; 3],
    /// Original: `real*4, allocatable :: warpDx(:,:)` (`blendvars.f90:94`);
    /// allocated `(lmWarpX, lmWarpY)` (`blendmont.f90:1760`).
    pub warp_dx: Vec<f32>,
    /// Allocation extents of `warpDx`.
    pub warp_dx_ext: [usize; 2],
    /// Original: `real*4, allocatable :: warpDy(:,:)` (`blendvars.f90:94`);
    /// allocated `(lmWarpX, lmWarpY)`.
    pub warp_dy: Vec<f32>,
    /// Allocation extents of `warpDy`.
    pub warp_dy_ext: [usize; 2],
    /// Original: `logical doFields` (`blendvars.f90:95`).
    pub do_fields: bool,
    /// Original: `logical undistort` (`blendvars.f90:95`).
    pub undistort: bool,
    /// Original: `logical doMagGrad` (`blendvars.f90:95`).
    pub do_mag_grad: bool,
    /// Original: `logical focusAdjusted` (`blendvars.f90:95`).
    pub focus_adjusted: bool,
    /// Original: `logical doingEdgeFunc` (`blendvars.f90:95`).
    pub doing_edge_func: bool,
    /// Original: `logical debug` (`blendvars.f90:95`).
    pub debug: bool,
    /// Original: `logical secHasWarp` (`blendvars.f90:95`).
    pub sec_has_warp: bool,
    /// Original: `real*4 pixelMagGrad` (`blendvars.f90:96`).
    pub pixel_mag_grad: f32,
    /// Original: `real*4 axisRot` (`blendvars.f90:96`).
    pub axis_rot: f32,
    /// Original: `real*4 xFieldStrt` (`blendvars.f90:96`).
    pub x_field_strt: f32,
    /// Original: `real*4 yFieldStrt` (`blendvars.f90:96`).
    pub y_field_strt: f32,
    /// Original: `real*4 xFieldIntrv` (`blendvars.f90:96`).
    pub x_field_intrv: f32,
    /// Original: `real*4 yFieldIntrv` (`blendvars.f90:96`).
    pub y_field_intrv: f32,
    /// Original: `real*4, allocatable :: tiltAngles(:)` (`blendvars.f90:97`).
    pub tilt_angles: Vec<f32>,
    /// Original: `real*4, allocatable :: dmagPerUm(:)` (`blendvars.f90:97`).
    pub dmag_per_um: Vec<f32>,
    /// Original: `real*4, allocatable :: rotPerUm(:)` (`blendvars.f90:97`).
    pub rot_per_um: Vec<f32>,
    /// Original: `integer*4 nxField` (`blendvars.f90:98`).
    pub nx_field: i32,
    /// Original: `integer*4 nyField` (`blendvars.f90:98`).
    pub ny_field: i32,
    /// Original: `integer*4 numMagGrad` (`blendvars.f90:98`).
    pub num_mag_grad: i32,
    /// Original: `integer*4 lmField` (`blendvars.f90:98`).
    pub lm_field: i32,
    /// Original: `integer*4 maxFields` (`blendvars.f90:98`).
    pub max_fields: i32,
    /// Original: `integer*4 nxWarp` (`blendvars.f90:98`).
    pub nx_warp: i32,
    /// Original: `integer*4 nyWarp` (`blendvars.f90:98`).
    pub ny_warp: i32,
    /// Original: `integer*4 numAngles` (`blendvars.f90:99`).
    pub num_angles: i32,
    /// Original: `integer*4 lmWarpX` (`blendvars.f90:99`).
    pub lm_warp_x: i32,
    /// Original: `integer*4 lmWarpY` (`blendvars.f90:99`).
    pub lm_warp_y: i32,
    /// Original: `real*4 xWarpStrt` (`blendvars.f90:100`).
    pub x_warp_strt: f32,
    /// Original: `real*4 yWarpStrt` (`blendvars.f90:100`).
    pub y_warp_strt: f32,
    /// Original: `real*4 xWarpIntrv` (`blendvars.f90:100`).
    pub x_warp_intrv: f32,
    /// Original: `real*4 yWarpIntrv` (`blendvars.f90:100`).
    pub y_warp_intrv: f32,

    // ---- blendvars.f90:102-105
    /// Original: `integer*4 ifDumpXY(2)` (`blendvars.f90:102`).
    pub if_dump_xy: [i32; 2],
    /// Original: `integer*4 nzOutXY(2)` (`blendvars.f90:102`).
    pub nz_out_xy: [i32; 2],
    /// Original: `integer*4 nxOutXY(2)` (`blendvars.f90:102`).
    pub nx_out_xy: [i32; 2],
    /// Original: `integer*4 nyOutXY(2)` (`blendvars.f90:102`).
    pub ny_out_xy: [i32; 2],
    /// Original: `integer*4 ipcBelowEdge` (`blendvars.f90:102`).
    pub ipc_below_edge: i32,
    /// Original: `integer*4 ifillTreatment` (`blendvars.f90:103`).
    pub ifill_treatment: i32,
    /// Original: `integer*4 numXcorrPeaks` (`blendvars.f90:103`).
    pub num_xcorr_peaks: i32,
    /// Original: `integer*4 nbinXcorr` (`blendvars.f90:103`).
    pub nbin_xcorr: i32,
    /// Original: `integer*4 ixDebug` (`blendvars.f90:103`).
    pub ix_debug: i32,
    /// Original: `integer*4 iyDebug` (`blendvars.f90:103`).
    pub iy_debug: i32,
    /// Original: `real*4 padFrac` (`blendvars.f90:104`).
    pub pad_frac: f32,
    /// Original: `real*4 aspectMax` (`blendvars.f90:104`).
    pub aspect_max: f32,
    /// Original: `real*4 extraWidth` (`blendvars.f90:104`).
    pub extra_width: f32,
    /// Original: `real*4 radius1` (`blendvars.f90:105`).
    pub radius1: f32,
    /// Original: `real*4 radius2` (`blendvars.f90:105`).
    pub radius2: f32,
    /// Original: `real*4 sigma1` (`blendvars.f90:105`).
    pub sigma1: f32,
    /// Original: `real*4 sigma2` (`blendvars.f90:105`).
    pub sigma2: f32,
    /// Original: `real*4 robustCrit` (`blendvars.f90:105`).
    pub robust_crit: f32,

    // ---- blendvars.f90:107-115: variables for finding shifts and gradients
    /// Original: `integer*4 limVar` (`blendvars.f90:108`).
    pub lim_var: i32,
    /// Original: `real*4, allocatable :: rowTmp(:)` (`blendvars.f90:109`).
    pub row_tmp: Vec<f32>,
    /// Original: `real*4, allocatable :: dxyVar(:,:)` (`blendvars.f90:109`);
    /// allocated `(limVar, 2)` (`blendmont.f90:945`).
    pub dxy_var: Vec<f32>,
    /// Allocation extents of `dxyVar`.
    pub dxy_var_ext: [usize; 2],
    /// Original: `real*4, allocatable :: bb(:,:)` (`blendvars.f90:109`);
    /// allocated `(2, limVar)` (`blendmont.f90:943`).
    pub bb: Vec<f32>,
    /// Allocation extents of `bb`.
    pub bb_ext: [usize; 2],
    /// Original: `real*4, allocatable :: fpsWork(:)` (`blendvars.f90:109`).
    pub fps_work: Vec<f32>,
    /// Original: `integer*4, allocatable :: ivarPc(:)` (`blendvars.f90:110`).
    pub ivar_pc: Vec<i32>,
    /// Original: `integer*4, allocatable :: indVar(:)` (`blendvars.f90:110`).
    pub ind_var: Vec<i32>,
    /// Original: `integer*4, allocatable :: iallVarPc(:)` (`blendvars.f90:111`).
    pub iall_var_pc: Vec<i32>,
    /// Original: `integer*4, allocatable :: ivarGroup(:)` (`blendvars.f90:111`).
    pub ivar_group: Vec<i32>,
    /// Original: `integer*4, allocatable :: listCheck(:)` (`blendvars.f90:111`).
    pub list_check: Vec<i32>,
    /// Original: `real*4, allocatable :: gradXcenLo(:)` (`blendvars.f90:112`).
    pub grad_xcen_lo: Vec<f32>,
    /// Original: `real*4, allocatable :: gradXcenHi(:)` (`blendvars.f90:112`).
    pub grad_xcen_hi: Vec<f32>,
    /// Original: `real*4, allocatable :: gradYcenLo(:)` (`blendvars.f90:112`).
    pub grad_ycen_lo: Vec<f32>,
    /// Original: `real*4, allocatable :: gradYcenHi(:)` (`blendvars.f90:113`).
    pub grad_ycen_hi: Vec<f32>,
    /// Original: `real*4, allocatable :: overXcenLo(:)` (`blendvars.f90:113`).
    pub over_xcen_lo: Vec<f32>,
    /// Original: `real*4, allocatable :: overXcenHi(:)` (`blendvars.f90:113`).
    pub over_xcen_hi: Vec<f32>,
    /// Original: `real*4, allocatable :: overYcenLo(:)` (`blendvars.f90:114`).
    pub over_ycen_lo: Vec<f32>,
    /// Original: `real*4, allocatable :: overYcenHi(:)` (`blendvars.f90:114`).
    pub over_ycen_hi: Vec<f32>,
    /// Original: `real*4, allocatable :: dxEdge(:,:)` (`blendvars.f90:115`);
    /// allocated `(limEdge, 2)` (`blendmont.f90:952`).
    pub dx_edge: Vec<f32>,
    /// Allocation extents of `dxEdge`.
    pub dx_edge_ext: [usize; 2],
    /// Original: `real*4, allocatable :: dyEdge(:,:)` (`blendvars.f90:115`);
    /// allocated `(limEdge, 2)`.
    pub dy_edge: Vec<f32>,
    /// Allocation extents of `dyEdge`.
    pub dy_edge_ext: [usize; 2],
    /// Original: `real*4, allocatable :: dxAdj(:,:)` (`blendvars.f90:115`);
    /// allocated `(limEdge, 2)`.
    pub dx_adj: Vec<f32>,
    /// Allocation extents of `dxAdj`.
    pub dx_adj_ext: [usize; 2],
    /// Original: `real*4, allocatable :: dyAdj(:,:)` (`blendvars.f90:115`);
    /// allocated `(limEdge, 2)`.
    pub dy_adj: Vec<f32>,
    /// Allocation extents of `dyAdj`.
    pub dy_adj_ext: [usize; 2],

    // ---- blendvars.f90:117-126: variables added when internal sub didn't work
    /// Original: `integer*4 iblend(2)` (`blendvars.f90:118`) — blending width in
    /// x and y.
    pub iblend: [i32; 2],
    /// Original: `integer*4 indEdge4(3,2)` (`blendvars.f90:119`).
    /// `indEdge4(i, j)` is `ind_edge4[j - 1][i - 1]`.
    pub ind_edge4: [[i32; 3]; 2],
    /// Original: `real*4 edgeFrac4(3,2)` (`blendvars.f90:120`).
    /// `edgeFrac4(i, j)` is `edge_frac4[j - 1][i - 1]`.
    pub edge_frac4: [[f32; 3]; 2],
    /// Original: `integer*4 indp1234(8)` (`blendvars.f90:121`); `indp1`,
    /// `indp2`, `indp3`, `indp4` (`:124`) are equivalenced to its first four elements (`:122`),
    /// so they are `indp1234[0]`, `[1]`, `[2]`, `[3]` and have no field of their
    /// own.
    pub indp1234: [i32; 8],
    /// Original: `integer*4 inde12` (`blendvars.f90:124`).
    pub inde12: i32,
    /// Original: `integer*4 inde13` (`blendvars.f90:124`).
    pub inde13: i32,
    /// Original: `integer*4 inde34` (`blendvars.f90:124`).
    pub inde34: i32,
    /// Original: `integer*4 inde24` (`blendvars.f90:124`).
    pub inde24: i32,
    /// Original: `integer*4 nActiveP` (`blendvars.f90:124`).
    pub n_active_p: i32,
    /// Original: `integer*4 lastxyDisjoint` (`blendvars.f90:125`).
    pub lastxy_disjoint: i32,
    /// Original: `integer*4 lastp1` (`blendvars.f90:125`).
    pub lastp1: i32,
    /// Original: `integer*4 lastp2` (`blendvars.f90:125`).
    pub lastp2: i32,
    /// Original: `integer*4 lastp3` (`blendvars.f90:125`).
    pub lastp3: i32,
    /// Original: `integer*4 lastp4` (`blendvars.f90:125`).
    pub lastp4: i32,
    /// Original: `real*4 wll` (`blendvars.f90:126`).
    pub wll: f32,
    /// Original: `real*4 wlr` (`blendvars.f90:126`).
    pub wlr: f32,
    /// Original: `real*4 wul` (`blendvars.f90:126`).
    pub wul: f32,
    /// Original: `real*4 wur` (`blendvars.f90:126`).
    pub wur: f32,
    /// Original: `real*4 startSkew` (`blendvars.f90:126`).
    pub start_skew: f32,
    /// Original: `real*4 endSkew` (`blendvars.f90:126`).
    pub end_skew: f32,
    /// Original: `real*4 ex` (`blendvars.f90:126`).
    pub ex: f32,
    /// Original: `real*4 ey` (`blendvars.f90:126`).
    pub ey: f32,
}

impl Default for BlendVars {
    /// The module's state at program start: static storage, zero / `.false.`
    /// scalars and fixed arrays, unallocated (empty) allocatables.  Written by
    /// hand because arrays longer than 32 have no `Default`.
    fn default() -> Self {
        BlendVars {
            array: Vec::new(),
            brray: Vec::new(),
            xcray: Vec::new(),
            xdray: Vec::new(),
            xeray: Vec::new(),
            nxyz_in: [0; 3],
            nxyz_out: [0; 3],
            idimc: 0,
            nxyz_bin: [0; 3],
            lim_npc: 0,
            lim_sect: 0,
            lim_edge: 0,
            max_siz: 0,
            n_overlap: [0; 2],
            nedge: [0; 2],
            max_line_length: 0,
            max_bsiz: 0,
            ix_pc_list: Vec::new(),
            iy_pc_list: Vec::new(),
            iz_pc_list: Vec::new(),
            neg_list: Vec::new(),
            lim_data_lo: Vec::new(),
            lim_data_lo_ext: [0; 3],
            lim_data_hi: Vec::new(),
            lim_data_hi_ext: [0; 3],
            iedge_lower: Vec::new(),
            iedge_lower_ext: [0; 2],
            iedge_upper: Vec::new(),
            iedge_upper_ext: [0; 2],
            lim_data_ind: Vec::new(),
            ipiece_lower: Vec::new(),
            ipiece_lower_ext: [0; 2],
            ipiece_upper: Vec::new(),
            ipiece_upper_ext: [0; 2],
            ibuf_edge: Vec::new(),
            ibuf_edge_ext: [0; 2],
            if_skip_edge: Vec::new(),
            if_skip_edge_ext: [0; 2],
            max_sd_to_edge_num: Vec::new(),
            max_sd_to_ixy_of_edge: Vec::new(),
            trimmed_max_sds: Vec::new(),
            altern_disps: Vec::new(),
            iedge_alt_fixed: Vec::new(),
            iedge_low_weight: Vec::new(),
            num_alt_fixed: 0,
            num_low_weight: 0,
            indent: [0; 2],
            int_grid: [0; 2],
            ibox_siz: [0; 2],
            nx_grid: [0; 2],
            ny_grid: [0; 2],
            edge_lo_near: [0.0; 2],
            edge_hi_near: [0.0; 2],
            npc_list: 0,
            min_xpiece: 0,
            min_ypiece: 0,
            nx_pieces: 0,
            ny_pieces: 0,
            interp_order: 0,
            num_max_sds: 0,
            dmean: 0.0,
            dfill: 0.0,
            iz_use_def_low: 0,
            iz_use_def_high: 0,
            num_use_edge: 0,
            ixy_use_edge: [0; MAX_USE_EDGE as usize],
            ix_frm_use_edge: [0; MAX_USE_EDGE as usize],
            iy_frm_use_edge: [0; MAX_USE_EDGE as usize],
            iz_low_use: [0; MAX_USE_EDGE as usize],
            iz_high_use: [0; MAX_USE_EDGE as usize],
            last_written: [0; 2],
            num_pieces: 0,
            in_piece: [0; MAX_IN_PC as usize + 1],
            x_in_piece: [0.0; MAX_IN_PC as usize],
            y_in_piece: [0.0; MAX_IN_PC as usize],
            inp_xframe: [0; MAX_IN_PC as usize],
            inp_yframe: [0; MAX_IN_PC as usize],
            max_xframe: 0,
            max_yframe: 0,
            min_xframe: 0,
            min_yframe: 0,
            num_edges: [0; 2],
            in_edge: [[0; MAX_IN_PC as usize]; 2],
            in_ed_lower: [[0; MAX_IN_PC as usize]; 2],
            in_ed_upper: [[0; MAX_IN_PC as usize]; 2],
            num_pc_near: 0,
            idx_pc_near: [0; MAX_PC_NEAR as usize],
            idy_pc_near: [0; MAX_PC_NEAR as usize],
            map_piece: Vec::new(),
            map_piece_ext: [0; 2],
            any_disjoint: Vec::new(),
            any_disjoint_ext: [0; 2],
            map_disjoint: Vec::new(),
            map_disjoint_ext: [0; 2],
            do_gxforms: false,
            multng: false,
            limit_data: false,
            hx_cen: 0.0,
            hy_cen: 0.0,
            gx_cen: 0.0,
            gy_cen: 0.0,
            ginv: [[0.0; 2]; 3],
            hinv: Vec::new(),
            hinv_ext: [0; 3],
            htmp: Vec::new(),
            htmp_ext: [0; 3],
            mem_index: Vec::new(),
            iz_mem_list: Vec::new(),
            last_used: Vec::new(),
            max_load: 0,
            juse_count: 0,
            ilistz: 0,
            mem_lim: 0,
            npix_in: 0,
            iedg_bf_list: [0; LIM_EDG_BF as usize],
            ixy_bf_list: [0; LIM_EDG_BF as usize],
            las_edg_use: [0; LIM_EDG_BF as usize],
            iun_edge: [0; 2],
            ixg_dim: 0,
            iyg_dim: 0,
            nx_gr_bf: [0; LIM_EDG_BF as usize],
            ny_gr_bf: [0; LIM_EDG_BF as usize],
            ix_grd_st_bf: [0; LIM_EDG_BF as usize],
            iy_grd_st_bf: [0; LIM_EDG_BF as usize],
            ix_ofs_bf: [0; LIM_EDG_BF as usize],
            iy_ofs_bf: [0; LIM_EDG_BF as usize],
            int_gr_copy: [0; 2],
            int_xgr_bf: [0; LIM_EDG_BF as usize],
            int_ygr_bf: [0; LIM_EDG_BF as usize],
            dx_gr_bf: Vec::new(),
            dx_gr_bf_ext: [0; 3],
            dy_gr_bf: Vec::new(),
            dy_gr_bf_ext: [0; 3],
            dden_gr_bf: Vec::new(),
            dden_gr_bf_ext: [0; 3],
            dx_grid: Vec::new(),
            dx_grid_ext: [0; 2],
            dy_grid: Vec::new(),
            dy_grid_ext: [0; 2],
            dden_grid: Vec::new(),
            dden_grid_ext: [0; 2],
            sd_grid: Vec::new(),
            sd_grid_ext: [0; 2],
            jus_edg_ct: 0,
            need_byte_swap: 0,
            iz_unsmoothed_patch: 0,
            iz_smoothed_patch: 0,
            iedge_cur_base: [0; 2],
            iun_dens: [0; 2],
            ix_dim_den_buf: 0,
            i_apply_dens_scaling: 0,
            i_dens_from_edges: 0,
            mult_by_flatfield: 0,
            i_den_sample: 0,
            interval_den: [0; 2],
            nx_den_grid: [0; 2],
            ny_den_grid: [0; 2],
            i_edge_zbase: Vec::new(),
            i_edge_zbase_ext: [0; 2],
            ix_den_offset: Vec::new(),
            iy_den_offset: Vec::new(),
            nx_den_buf: Vec::new(),
            ny_den_buf: Vec::new(),
            ix_den_start: Vec::new(),
            iy_den_start: Vec::new(),
            den_abuf: Vec::new(),
            den_abuf_ext: [0; 2],
            den_bbuf: Vec::new(),
            den_bbuf_ext: [0; 2],
            piece_scaling: Vec::new(),
            piece_scaling_ext: [0; 2],
            delta_den_buf: Vec::new(),
            delta_den_buf_ext: [0; 2],
            den_solution: Vec::new(),
            flatfield: Vec::new(),
            x_grad_scaling: 0.0,
            y_grad_scaling: 0.0,
            den_zero_base: 0.0,
            x_base_grad_scale: 0.0,
            y_base_grad_scale: 0.0,
            dist_dx: Vec::new(),
            dist_dx_ext: [0; 2],
            dist_dy: Vec::new(),
            dist_dy_ext: [0; 2],
            field_dx: Vec::new(),
            field_dx_ext: [0; 3],
            field_dy: Vec::new(),
            field_dy_ext: [0; 3],
            warp_dx: Vec::new(),
            warp_dx_ext: [0; 2],
            warp_dy: Vec::new(),
            warp_dy_ext: [0; 2],
            do_fields: false,
            undistort: false,
            do_mag_grad: false,
            focus_adjusted: false,
            doing_edge_func: false,
            debug: false,
            sec_has_warp: false,
            pixel_mag_grad: 0.0,
            axis_rot: 0.0,
            x_field_strt: 0.0,
            y_field_strt: 0.0,
            x_field_intrv: 0.0,
            y_field_intrv: 0.0,
            tilt_angles: Vec::new(),
            dmag_per_um: Vec::new(),
            rot_per_um: Vec::new(),
            nx_field: 0,
            ny_field: 0,
            num_mag_grad: 0,
            lm_field: 0,
            max_fields: 0,
            nx_warp: 0,
            ny_warp: 0,
            num_angles: 0,
            lm_warp_x: 0,
            lm_warp_y: 0,
            x_warp_strt: 0.0,
            y_warp_strt: 0.0,
            x_warp_intrv: 0.0,
            y_warp_intrv: 0.0,
            if_dump_xy: [0; 2],
            nz_out_xy: [0; 2],
            nx_out_xy: [0; 2],
            ny_out_xy: [0; 2],
            ipc_below_edge: 0,
            ifill_treatment: 0,
            num_xcorr_peaks: 0,
            nbin_xcorr: 0,
            ix_debug: 0,
            iy_debug: 0,
            pad_frac: 0.0,
            aspect_max: 0.0,
            extra_width: 0.0,
            radius1: 0.0,
            radius2: 0.0,
            sigma1: 0.0,
            sigma2: 0.0,
            robust_crit: 0.0,
            lim_var: 0,
            row_tmp: Vec::new(),
            dxy_var: Vec::new(),
            dxy_var_ext: [0; 2],
            bb: Vec::new(),
            bb_ext: [0; 2],
            fps_work: Vec::new(),
            ivar_pc: Vec::new(),
            ind_var: Vec::new(),
            iall_var_pc: Vec::new(),
            ivar_group: Vec::new(),
            list_check: Vec::new(),
            grad_xcen_lo: Vec::new(),
            grad_xcen_hi: Vec::new(),
            grad_ycen_lo: Vec::new(),
            grad_ycen_hi: Vec::new(),
            over_xcen_lo: Vec::new(),
            over_xcen_hi: Vec::new(),
            over_ycen_lo: Vec::new(),
            over_ycen_hi: Vec::new(),
            dx_edge: Vec::new(),
            dx_edge_ext: [0; 2],
            dy_edge: Vec::new(),
            dy_edge_ext: [0; 2],
            dx_adj: Vec::new(),
            dx_adj_ext: [0; 2],
            dy_adj: Vec::new(),
            dy_adj_ext: [0; 2],
            iblend: [0; 2],
            ind_edge4: [[0; 3]; 2],
            edge_frac4: [[0.0; 3]; 2],
            indp1234: [0; 8],
            inde12: 0,
            inde13: 0,
            inde34: 0,
            inde24: 0,
            n_active_p: 0,
            lastxy_disjoint: 0,
            lastp1: 0,
            lastp2: 0,
            lastp3: 0,
            lastp4: 0,
            wll: 0.0,
            wlr: 0.0,
            wul: 0.0,
            wur: 0.0,
            start_skew: 0.0,
            end_skew: 0.0,
            ex: 0.0,
            ey: 0.0,
        }
    }
}
