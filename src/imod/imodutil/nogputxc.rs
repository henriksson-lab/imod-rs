//! Translation of `IMOD/imodutil/nogputxc.cpp`, the non-GPU stubs of the
//! `TxcGPU` class, with its class header `IMOD/imodutil/gputiltxc.h` merged in.
//!
//! The reference build links these stubs (the CUDA `gputiltxc.cu` is out of
//! scope), so `gpuAvailable` always reports no GPU and every other member
//! reports failure; `tiltxcorr` falls back to the CPU before it can reach any
//! of them.

use super::tiltxcorr::MAX_FULL_CACHE;

/// `gputiltxc.h:9`.
pub const MAX_PLAN_ARRAYS: i32 = 20;

/// `class TxcGPU` (`gputiltxc.h:11-77`).  The private members are declared
/// as in the header; the stub constructor sets none of them.
#[allow(dead_code)]
#[derive(Default)]
pub struct TxcGPU {
    m_debug: i32,
    m_cur_taper_pad: Vec<f32>,
    m_last_taper_pad: Vec<f32>,
    m_cur_extract: Vec<f32>,
    m_last_copy_for_iter: Vec<f32>,
    m_last_copy_for_ccc: Vec<f32>,
    m_corr_packed: Vec<f32>,
    m_limit_shift: bool,
    m_use_ccc: i32,
    m_binning: i32,
    m_iterate: i32,
    m_extract_size: i32,
    m_pad_size: i32,
    m_nx_full: i32,
    m_ny_full: i32,
    m_nx_patch: i32,
    m_nx_taper: i32,
    m_nx_pad: i32,
    m_ny_patch: i32,
    m_ny_taper: i32,
    m_ny_pad: i32,
    m_filt_delta: f32,
    m_last_xstart: i32,
    m_last_ystart: i32,
    m_last_xpad: i32,
    m_last_ypad: i32,
    m_last_iz_val: i32,
    m_max_plans: f32,
    m_nx_plan: [i32; MAX_PLAN_ARRAYS as usize],
    m_ny_plan: [i32; MAX_PLAN_ARRAYS as usize],
    m_plan_used: [i32; MAX_PLAN_ARRAYS as usize],
    m_use_count: i32,
    m_iz_loaded: [i32; MAX_FULL_CACHE],
    m_cur_plan_ind: i32,
    m_full_cache_size: i32,
    m_track_time: bool,
    m_wall_copy: f64,
    m_wall_plan: f64,
    m_wall_extract: f64,
    m_wall_xform: f64,
    m_wall_fft: f64,
    m_wall_filter: f64,
    m_wall_conj: f64,
    m_wall_temp: f64,
    m_wall_start: f64,
}

impl TxcGPU {
    /// `TxcGPU::TxcGPU` (`nogputxc.cpp:2`), the constructor `TxcGPU`.
    pub fn new() -> Self {
        TxcGPU::default()
    }

    /// `TxcGPU::gpuAvailable` (`nogputxc.cpp:3`).
    pub fn gpu_available(&mut self, _n_gpu: i32, memory: &mut f32, _debug: i32) -> i32 {
        *memory = 0.;
        0
    }

    /// `TxcGPU::initialize` (`nogputxc.cpp:8`).
    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        &mut self,
        _nx_full: i32,
        _ny_full: i32,
        _extract_size: i32,
        _pad_size: i32,
        _max_plans: i32,
        _max_fulls: i32,
        _ctf: &[f32],
        _ctf_size: i32,
        _delta: f32,
        _binning: i32,
        _use_ccc: i32,
        _iterate: i32,
        _limiting_shift: bool,
    ) -> i32 {
        1
    }

    /// `TxcGPU::allocatePatchArrays` (`nogputxc.cpp:11`).
    pub fn allocate_patch_arrays(&mut self) -> i32 {
        1
    }

    /// `TxcGPU::loadFullArray` (`nogputxc.cpp:12`).
    pub fn load_full_array(&mut self, _iz_read: i32, _load_ind: i32, _full_arr: &[f32]) -> i32 {
        1
    }

    /// `TxcGPU::setupForPatch` (`nogputxc.cpp:13`).
    pub fn setup_for_patch(
        &mut self,
        _nx_patch: i32,
        _ny_patch: i32,
        _nx_pad: i32,
        _ny_pad: i32,
        _nx_taper: i32,
        _ny_taper: i32,
    ) -> i32 {
        1
    }

    /// `TxcGPU::extractPatch` (`nogputxc.cpp:15`).
    #[allow(clippy::too_many_arguments)]
    pub fn extract_patch(
        &mut self,
        _iz_val: i32,
        _is_last: bool,
        _dmean: f32,
        _ix_start: i32,
        _iy_start: i32,
        _fs: &[f32],
        _xtrans: f32,
        _ytrans: f32,
        _n_taper: i32,
        _fill: f32,
        _good_xstart: i32,
        _good_xend: i32,
        _good_ystart: i32,
        _good_yend: i32,
        _dump_images: bool,
    ) -> i32 {
        1
    }

    /// `TxcGPU::getCorrelation` (`nogputxc.cpp:20`).
    #[allow(clippy::too_many_arguments)]
    pub fn get_correlation(
        &mut self,
        _iteration: i32,
        _corr_arr: &mut [f32],
        _last_filt: &mut [f32],
        _cur_filt: &mut [f32],
        _corr_xdim: i32,
        _corr_ypad: i32,
        _dump_corr: bool,
    ) -> i32 {
        1
    }

    /// `TxcGPU::freeCudaArray` (`nogputxc.cpp:22`).
    pub fn free_cuda_array(&mut self, _array: &mut Vec<f32>) {}

    /// `TxcGPU::isInitialized` (`nogputxc.cpp:23`).
    pub fn is_initialized(&self) -> bool {
        false
    }

    /// `TxcGPU::reportTimes` (`nogputxc.cpp:24`).
    pub fn report_times(&self) {}
}
