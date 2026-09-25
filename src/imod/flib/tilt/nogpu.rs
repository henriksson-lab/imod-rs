//! Translation of `IMOD/flib/tilt/nogpu.cpp` ("Has stubs to allow compilation when
//! no CUDA is available"), with its paired header `IMOD/flib/tilt/gpubp.h` merged in
//! (`ORDER.md` §9: `gpubp.h`→`nogpu.rs`).
//!
//! The reference build is CUDA-free (`configure: TILTGPUOBJ = nogpu.$(OBJEXT)`), so
//! this is what `tilt` links.  `gpubp.h` holds only the prototypes below, one per
//! entry point, and no types or constants; the header's guard and its "IF YOU
//! CHANGE SOMETHING, COPY IT TO nogpu.cpp" note have nothing to translate.
//!
//! Every stub returns 1 and touches none of its arguments.  Note that for
//! `gpuAvailable` 1 is **not** a failure: `tilt.cpp:4688,4839` read
//! `gpuAvailable(...) != 0` as "a GPU is available", leaving `gpuMemory`
//! unwritten (it is uninitialised in `Tilt::inputParameters`), and the later
//! `gpuAllocArrays(...) == 0` / `gpuLoadFilter(...) == 0` tests are what turn
//! `mUseGPU` off.  `tilt.rs` must reproduce that sequence.
//!
//! Pointer parameters become slices: `&mut` where the CUDA implementation
//! (`gpubp.cu`) writes host memory back (`slice` outputs, `gpuFilterLines`'s
//! `lines`, `gpuFilterRawImage`'s `rawProj`, the reprojection `lines`), `&`
//! where it only uploads.  `float &memory` is `&mut f32`; `int *maxTex2D`,
//! `*maxTexLayer`, `*maxTex3D` are `tilt.cpp:2770`'s `int[2]`, `int[3]`, `int[3]`.

/// C `gpuAllocArrays` (`nogpu.cpp:4`, `gpubp.h:4`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_alloc_arrays(
    _width: i32,
    _nyout: i32,
    _nx_proj_pad: i32,
    _ny_proj: i32,
    _nplanes: i32,
    _nviews: i32,
    _num_warps: i32,
    _num_delz: i32,
    _nfilt: i32,
    _nreproj: i32,
    _ny_raw_pad: i32,
    _nx_filt_dim: i32,
    _super_samp: i32,
    _super_border: i32,
    _nx_crop_pad: i32,
    _ny_crop_pad: i32,
    _clean_super: i32,
    _first_npl: i32,
    _last_npl: i32,
    _use_3d: i32,
) -> i32 {
    1
}

/// C `gpuLoadLocals` (`nogpu.cpp:13`, `gpubp.h:9`).
pub fn gpu_load_locals(_packed: &[f32], _num_warps: i32) -> i32 {
    1
}

/// C `gpuAvailable` (`nogpu.cpp:18`, `gpubp.h:10`).
pub fn gpu_available(
    _n_gpu: i32,
    _memory: &mut f32,
    _max_tex_2d: &mut [i32],
    _max_tex_layer: &mut [i32],
    _max_tex_3d: &mut [i32],
    _debug: i32,
) -> i32 {
    1
}

/// C `gpuLoadFilter` (`nogpu.cpp:24`, `gpubp.h:12`).
pub fn gpu_load_filter(_lines: &[f32]) -> i32 {
    1
}

/// C `gpuLoadRawFiltMap` (`nogpu.cpp:29`, `gpubp.h:13`).
pub fn gpu_load_raw_filt_map(_map: &[f32]) -> i32 {
    1
}

/// C `gpuFilterRawImage` (`nogpu.cpp:34`, `gpubp.h:14`).
pub fn gpu_filter_raw_image(_raw_proj: &mut [f32], _view_ind: i32) -> i32 {
    1
}

/// C `gpuReprojLocal` (`nogpu.cpp:39`, `gpubp.h:15`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_reproj_local(
    _lines: &mut [f32],
    _sin_beta: f32,
    _cos_beta: f32,
    _sin_alpha: f32,
    _cos_alpha: f32,
    _xzfac: f32,
    _yzfac: f32,
    _nx_warp: i32,
    _ny_warp: i32,
    _ix_start_warp: i32,
    _iy_start_warp: i32,
    _i_del_xwarp: i32,
    _i_del_ywarp: i32,
    _warp_delz: &[f32],
    _n_warp_delz: i32,
    _dx_warp_delz: f32,
    _xproj_min: f32,
    _xproj_max: f32,
    _lslice_start: i32,
    _lslice_end: i32,
    _ithick: i32,
    _iview: i32,
    _xcen_out: f32,
    _xcen_in: f32,
    _axis_xoffset: f32,
    _min_xload: i32,
    _x_proj_offset: f32,
    _ycen_adj: f32,
    _y_proj_offset: f32,
    _center_slice: f32,
    _pmean: f32,
) -> i32 {
    1
}

/// C `gpuBpLocal` (`nogpu.cpp:51`, `gpubp.h:23`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_bp_local(
    _slice: &mut [f32],
    _lslice: i32,
    _nx_warp: i32,
    _ny_warp: i32,
    _ix_start_warp: i32,
    _iy_start_warp: i32,
    _i_del_xwarp: i32,
    _i_del_ywarp: i32,
    _nx_proj: i32,
    _xcen_out: f32,
    _xcen_in: f32,
    _axis_xoffset: f32,
    _ycen_out: f32,
    _center_slice: f32,
    _edgefill: f32,
) -> i32 {
    1
}

/// C `gpuBpXtilt` (`nogpu.cpp:59`, `gpubp.h:27`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_bp_xtilt(
    _slice: &mut [f32],
    _sin_beta: &[f32],
    _cos_beta: &[f32],
    _sin_alpha: &[f32],
    _cos_alpha: &[f32],
    _xzfac: &[f32],
    _yzfac: &[f32],
    _nx_proj: i32,
    _ny_proj: i32,
    _xcen_in: f32,
    _xcen_out: f32,
    _ycen_out: f32,
    _lslice: i32,
    _center_slice: f32,
    _edgefill: f32,
) -> i32 {
    1
}

/// C `gpuBpNoX` (`nogpu.cpp:67`, `gpubp.h:31`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_bp_no_x(
    _slice: &mut [f32],
    _lines: &[f32],
    _sin_beta: &[f32],
    _cos_beta: &[f32],
    _nx_proj: i32,
    _xcen_in: f32,
    _xcen_out: f32,
    _ycen_out: f32,
    _edgefill: f32,
) -> i32 {
    1
}

/// C `gpuShiftProj` (`nogpu.cpp:74`, `gpubp.h:34`).
pub fn gpu_shift_proj(_num_planes: i32, _lslice_start: i32, _load_start: i32) -> i32 {
    1
}

/// C `gpuLoadProj` (`nogpu.cpp:79`, `gpubp.h:35`).
pub fn gpu_load_proj(
    _lines: &[f32],
    _num_planes: i32,
    _lslice_start: i32,
    _load_start: i32,
) -> i32 {
    1
}

/// C `gpuFilterLines` (`nogpu.cpp:84`, `gpubp.h:36`).
pub fn gpu_filter_lines(_lines: &mut [f32], _lslice: i32, _filter_set: i32) -> i32 {
    1
}

/// C `gpuReproject` (`nogpu.cpp:89`, `gpubp.h:37`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_reproject(
    _lines: &mut [f32],
    _sin_beta: f32,
    _cos_beta: f32,
    _sin_alpha: f32,
    _cos_alpha: f32,
    _xzfac: f32,
    _yzfac: f32,
    _delz: f32,
    _lslice_start: i32,
    _lslice_end: i32,
    _ithick: i32,
    _xcen_out: f32,
    _xcen_paxis_ofs: f32,
    _min_xreproj: i32,
    _x_proj_offset: f32,
    _ycen_out: f32,
    _min_yreproj: i32,
    _y_proj_offset: f32,
    _center_slice: f32,
    _if_alpha: i32,
    _pmean: f32,
) -> i32 {
    1
}

/// C `gpuReprojOneSlice` (`nogpu.cpp:99`, `gpubp.h:43`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_reproj_one_slice(
    _slice: &[f32],
    _lines: &mut [f32],
    _sin_beta: &[f32],
    _cos_beta: &[f32],
    _ycen: f32,
    _num_proj: i32,
    _pmean: f32,
) -> i32 {
    1
}

/// C `gpuDone` (`nogpu.cpp:105`, `gpubp.h:45`).
pub fn gpu_done() {}
