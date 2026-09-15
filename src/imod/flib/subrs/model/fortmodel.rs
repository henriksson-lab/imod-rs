//! Fortran model common storage from `IMOD/include/fortmodel.f90`.
//!
//! `IMOD/include/fortmodel.f90` is a data module: it declares the model arrays
//! that `read_mod`, `store_mod` and `readw_or_imod` share through `use
//! fortmodel`, plus the single `allocateFortModel` subroutine that sizes them.
//! The crate has no `src/imod/include` directory yet, so the module lives
//! beside its only users in `flib/subrs/model`; move it to a mirrored
//! `src/imod/include/fortmodel.rs` when that directory is created.
#![allow(dead_code)]

/// Original: `max_clabel` (`fortmodel.f90:9`) — max # of text labels.
pub const MAX_CLABEL: usize = 200;
/// Original: `maxTypes` (`fortmodel.f90:10`).
pub const MAX_TYPES: usize = 2;

/// Original: module `fortmodel` (`fortmodel.f90:5`).
///
/// The Fortran module variables are program-wide state reached through `use
/// fortmodel`; the translated program units take this struct by reference in
/// the same order the source reads and writes the module variables.
pub struct FortModel {
    /// `max_obj_num` (`fortmodel.f90:12`), `data max_obj_num /1000000/`.
    pub max_obj_num: i32,
    /// `max_pt` (`fortmodel.f90:14`), `data max_pt /20000000/`.
    pub max_pt: i32,
    /// `len_object` (`fortmodel.f90:16`), `data len_object /24000000/`.
    pub len_object: i32,
    /// `max_obj_order` (`fortmodel.f90:18`), `data max_obj_order /1200000/`.
    pub max_obj_order: i32,
    /// `fmInitialized` (`fortmodel.f90:20`).
    pub fm_initialized: bool,
    /// `fmMaxObjNumDflt` (`fortmodel.f90:21`).
    pub fm_max_obj_num_dflt: [i32; MAX_TYPES],
    /// `fmMaxPtNumDflt` (`fortmodel.f90:22`).
    pub fm_max_pt_num_dflt: [i32; MAX_TYPES],
    /// `fmModSizeType` (`fortmodel.f90:35`).
    pub fm_mod_size_type: i32,
    /// `fmNeedObjects` (`fortmodel.f90:37`).
    pub fm_need_objects: i32,
    /// `fmNeedPoints` (`fortmodel.f90:39`).
    pub fm_need_points: i32,
    /// `fmIncReadObjBy` (`fortmodel.f90:41`).
    pub fm_inc_read_obj_by: i32,
    /// `fmIncReadPointsBy` (`fortmodel.f90:43`).
    pub fm_inc_read_points_by: i32,
    /// `fmMaxObjLoaded` (`fortmodel.f90:45`).
    pub fm_max_obj_loaded: i32,
    /// `fmBoostReadInBy` (`fortmodel.f90:47`).
    pub fm_boost_read_in_by: f32,
    /// `object(len_object)` (`fortmodel.f90:52`) — pointer into `p_coord`.
    pub object: Vec<i32>,
    /// `npt_in_obj(max_obj_num)` (`fortmodel.f90:55`).
    pub npt_in_obj: Vec<i32>,
    /// `ibase_obj(max_obj_num)` (`fortmodel.f90:58`).
    pub ibase_obj: Vec<i32>,
    /// `obj_color(2, max_obj_num)` (`fortmodel.f90:61`) — column major, so
    /// `obj_color(1,i)` and `obj_color(2,i)` are `obj_color[i - 1][0..2]`.
    pub obj_color: Vec<[i32; 2]>,
    /// `obj_order(max_obj_order)` (`fortmodel.f90:64`).
    pub obj_order: Vec<i32>,
    /// `ndx_order(max_obj_num)` (`fortmodel.f90:67`).
    pub ndx_order: Vec<i32>,
    /// `p_coord(3, max_pt)` (`fortmodel.f90:70`) — column major.
    pub p_coord: Vec<[f32; 3]>,
    /// Per-point marker values.  The source uses `integer(c_char)` only as a
    /// compact integer array; this is crate-owned marker data, not a C string
    /// or an ABI buffer.
    pub pt_label: Vec<u8>,
    /// `clabel(max_clabel)` (`fortmodel.f90:75`), `character*10`.
    pub clabel: Vec<[u8; 10]>,
    /// `label_list(max_clabel)` (`fortmodel.f90:76`).
    pub label_list: Vec<i32>,
    /// `n_point` (`fortmodel.f90:78`) — highest point # in `p_coord`.
    pub n_point: i32,
    /// `n_object` (`fortmodel.f90:80`) — total # of non-zero objects.
    pub n_object: i32,
    /// `ibase_free` (`fortmodel.f90:82`) — base index of free area in `object`.
    pub ibase_free: i32,
    /// `ntot_in_obj` (`fortmodel.f90:84`) — total entries in `object`.
    pub ntot_in_obj: i32,
    /// `nin_order` (`fortmodel.f90:86`) — # of entries in `obj_order`.
    pub nin_order: i32,
    /// `max_mod_obj` (`fortmodel.f90:88`) — highest object # used so far.
    pub max_mod_obj: i32,
    /// `n_clabel` (`fortmodel.f90:90`) — number of text labels.
    pub n_clabel: i32,
}

impl Default for FortModel {
    /// The `data` statements of `fortmodel.f90`; the allocatable arrays start
    /// unallocated, exactly as `fmInitialized = .false.` records.
    fn default() -> Self {
        Self {
            max_obj_num: 1000000,
            max_pt: 20000000,
            len_object: 24000000,
            max_obj_order: 1200000,
            fm_initialized: false,
            fm_max_obj_num_dflt: [1000000, 100000],
            fm_max_pt_num_dflt: [20000000, 1000000],
            fm_mod_size_type: 1,
            fm_need_objects: 0,
            fm_need_points: 0,
            fm_inc_read_obj_by: 0,
            fm_inc_read_points_by: 0,
            fm_max_obj_loaded: 0,
            fm_boost_read_in_by: 1.5,
            object: Vec::new(),
            npt_in_obj: Vec::new(),
            ibase_obj: Vec::new(),
            obj_color: Vec::new(),
            obj_order: Vec::new(),
            ndx_order: Vec::new(),
            p_coord: Vec::new(),
            pt_label: Vec::new(),
            clabel: Vec::new(),
            label_list: Vec::new(),
            n_point: 0,
            n_object: 0,
            ibase_free: 0,
            ntot_in_obj: 0,
            nin_order: 0,
            max_mod_obj: 0,
            n_clabel: 0,
        }
    }
}

/// Original: `allocateFortModel` (`fortmodel.f90:97`).
pub fn allocate_fort_model(fm: &mut FortModel) {
    // The Fortran `deallocate` of the previous arrays is the drop that
    // replacing each vector below performs.
    fm.fm_mod_size_type = 1.max(2.min(fm.fm_mod_size_type));
    fm.max_obj_num =
        fm.fm_max_obj_num_dflt[fm.fm_mod_size_type as usize - 1].max(fm.fm_need_objects);
    fm.max_pt = fm.fm_max_pt_num_dflt[fm.fm_mod_size_type as usize - 1].max(fm.fm_need_points);
    fm.len_object = fm.max_pt + fm.max_pt / 5;
    fm.max_obj_order = fm.max_obj_num + fm.max_obj_num / 5;
    fm.object = vec![0; fm.len_object as usize];
    fm.npt_in_obj = vec![0; fm.max_obj_num as usize];
    fm.ibase_obj = vec![0; fm.max_obj_num as usize];
    fm.obj_color = vec![[0; 2]; fm.max_obj_num as usize];
    fm.obj_order = vec![0; fm.max_obj_order as usize];
    fm.ndx_order = vec![0; fm.max_obj_num as usize];
    fm.p_coord = vec![[0.0; 3]; fm.max_pt as usize];
    fm.pt_label = vec![0; fm.max_pt as usize];
    // `clabel` and `label_list` are fixed-size module arrays, not allocatables;
    // they are materialised here so the module state is complete.
    fm.clabel = vec![[b' '; 10]; MAX_CLABEL];
    fm.label_list = vec![0; MAX_CLABEL];
    fm.fm_initialized = true;
    // `imodArrayLimits(max_pt, max_obj_num)` informs libimod of these limits;
    // the translated `libimod` model reader carries no such global limit.
}
