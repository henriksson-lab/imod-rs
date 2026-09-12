//! Translation of `IMOD/3dmod/iproc.cpp` and `iproc.h`.
//!
//! The image-processing state and byte-domain algorithms live here.  Qt
//! widgets, the 3dmod image store, and sliceproc's FFT/edge implementations
//! are deliberately represented by [`IprocBoundary`], rather than hidden
//! behind a replacement GUI or image store.
#![allow(dead_code)]

pub const PROC_BACKGROUND: u8 = 0;
pub const PROC_FOREGROUND: u8 = 255;
pub const NO_KERNEL_SIGMA: f32 = 0.4;
pub const KERNEL_MAXSIZE: usize = 7;
pub const APPLY_BUT: usize = 0;
pub const MORE_BUT: usize = 1;
pub const LESS_BUT: usize = 2;
pub const DO_SAME_BUT: usize = 3;
pub const TOGGLE_BUT: usize = 4;
pub const RESET_BUT: usize = 5;
pub const SAVE_BUT: usize = 6;
pub const LIST_BUT: usize = 7;

/// `IProcParam`.
#[derive(Clone, Debug, PartialEq)]
pub struct IprocParam {
    pub proc_num: i32,
    pub threshold: i32,
    pub thresh_grow: bool,
    pub thresh_shrink: bool,
    pub edge: i32,
    pub kernel_sigma: f32,
    pub smooth_3d: bool,
    pub rescale_smooth: bool,
    pub radius1: f32,
    pub radius2: f32,
    pub sigma1: f32,
    pub sigma2: f32,
    pub fft_binning: i32,
    pub fft_subset: bool,
    pub median_3d: bool,
    pub median_size: i32,
    pub andf_iterations: i32,
    pub andf_iter_done: i32,
    pub andf_k: f64,
    pub andf_lambda: f64,
    pub andf_stop_func: i32,
}
impl Default for IprocParam {
    fn default() -> Self {
        Self {
            proc_num: 0,
            threshold: 128,
            thresh_grow: false,
            thresh_shrink: false,
            edge: 0,
            kernel_sigma: NO_KERNEL_SIGMA,
            smooth_3d: false,
            rescale_smooth: true,
            radius1: 0.,
            radius2: 0.5,
            sigma1: 0.,
            sigma2: 0.05,
            fft_binning: 1,
            fft_subset: false,
            median_3d: true,
            median_size: 3,
            andf_iterations: 5,
            andf_iter_done: 0,
            andf_k: 2.,
            andf_lambda: 0.2,
            andf_stop_func: 0,
        }
    }
}

/// Source `ImodIProc`, with byte buffers owned by Rust instead of `malloc`.
#[derive(Clone, Debug, Default)]
pub struct ImodIproc {
    pub idata_sec: i32,
    pub idata_time: i32,
    pub modified: bool,
    pub auto_apply: bool,
    pub auto_save: bool,
    pub filter_slicer: bool,
    pub slicer_filt_applied: bool,
    pub toggling: bool,
    pub apply_thresh_change: bool,
    pub file_threshold: f32,
    pub range_low: i32,
    pub range_high: i32,
    pub input_mode: i32,
    pub output_mode: i32,
    pub was_byte: bool,
    pub timer_for_fft: bool,
    pub save_proc_num: i32,
    pub fft_scale: f32,
    pub fft_xrange: f32,
    pub fft_yrange: f32,
    pub fft_xcen: i32,
    pub fft_ycen: i32,
    pub isaved: Vec<u8>,
    pub iwork: Vec<u8>,
    pub andf_image: Vec<f32>,
    pub andf_image2: Vec<f32>,
    pub median_vol: Vec<Vec<u8>>,
}

/// Image/Qt/viewer calls made by this source unit.
pub trait IprocBoundary {
    fn dimensions(&self) -> (usize, usize, usize);
    fn current_section_time(&self) -> (i32, i32);
    fn image_mode(&self) -> i32;
    fn image_range(&self) -> (f32, f32);
    fn display_range(&self) -> (i32, i32);
    fn image_bytes(&mut self, section: i32, time: i32) -> Option<Vec<u8>>;
    fn replace_image_bytes(&mut self, section: i32, time: i32, data: &[u8]);
    fn redraw_image(&mut self);
    fn apply_edge(&mut self, _which: i32, _data: &mut [u8]) {}
    fn apply_fourier(&mut self, _data: &mut [u8], _p: &IprocParam) {}
    fn apply_fft(&mut self, _data: &mut [u8], _p: &IprocParam) -> (f32, i32, i32) {
        (0., 0, 0)
    }
    fn apply_median(&mut self, _data: &mut [u8], _volume: &[Vec<u8>], _p: &IprocParam) {}
    fn apply_aniso_diffusion(&mut self, _data: &mut [u8], _p: &mut IprocParam) {}
    fn dialog_message(&mut self, _message: &str) {}
}

/// `IProcWindow` sans Qt's generated widget ownership.
#[derive(Clone, Debug)]
pub struct IprocWindow {
    pub running_proc: bool,
    pub use_stack_ind: i32,
    pub command_list: Vec<String>,
    pub callback: Option<fn()>,
    pub four_filt_sliders: bool,
    pub sigma1_scale: f32,
    pub last_four_filt_range: f32,
    pub timer_id: i32,
    pub param_stack: Vec<IprocParam>,
    pub saved_param: IprocParam,
    pub data_modes: Vec<i32>,
    pub selected_filter: i32,
}
impl Default for IprocWindow {
    fn default() -> Self {
        Self {
            running_proc: false,
            use_stack_ind: -1,
            command_list: vec![],
            callback: None,
            four_filt_sliders: false,
            sigma1_scale: 0.,
            last_four_filt_range: 0.2,
            timer_id: 0,
            param_stack: vec![],
            saved_param: IprocParam::default(),
            data_modes: vec![],
            selected_filter: 0,
        }
    }
}

/// Source `IProcThread`; execution is intentionally scheduled by the host.
#[derive(Default)]
pub struct IprocThread;
impl IprocThread {
    pub fn run(
        &self,
        window: &mut IprocWindow,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
    ) {
        window.run_current_filter(proc, param, boundary);
    }
}

/// `edge_cb`.
pub fn edge_cb(
    proc: &mut ImodIproc,
    param: &IprocParam,
    boundary: &mut dyn IprocBoundary,
) -> String {
    boundary.apply_edge(param.edge, &mut proc.iwork);
    if param.edge == 0 {
        proc.output_mode = 0;
        proc.was_byte = true;
        "clip sobel".into()
    } else if param.edge == 1 {
        proc.output_mode = 0;
        proc.was_byte = true;
        "clip prewitt".into()
    } else if param.edge == 2 {
        "clip laplac".to_string()
    } else if param.edge == 3 {
        proc.output_mode = 0;
        proc.was_byte = true;
        "clip graham".into()
    } else {
        "clip gradient".into()
    }
}
/// `thresh_cb`.
pub fn thresh_cb(
    proc: &mut ImodIproc,
    param: &IprocParam,
    depth: i32,
    rampbase: i32,
    rampsize: i32,
) -> String {
    let (mut threshold, min, max) = if depth == 8 {
        (
            ((rampsize as f32 / 256.) * param.threshold as f32 + rampbase as f32) as i32,
            rampbase as u8,
            (rampsize + rampbase - 1) as u8,
        )
    } else {
        (param.threshold, 0, 255)
    };
    threshold = threshold.clamp(0, 255);
    for value in &mut proc.iwork {
        *value = if *value > threshold as u8 { max } else { min };
    }
    if param.thresh_grow || param.thresh_shrink {
        "Cannot do thresholding with grow or shrink in clip".into()
    } else {
        format!("clip threshold -t {}", proc.file_threshold)
    }
}
/// `smooth_cb`; sliceproc's 2-D Gaussian path is a boundary, while 3-D byte smoothing is native.
pub fn smooth_cb(proc: &mut ImodIproc, param: &IprocParam, nx: usize, ny: usize) -> String {
    if param.smooth_3d && !proc.median_vol.is_empty() {
        let _ = vol_byte_smooth(
            &mut proc.median_vol,
            nx,
            ny,
            param.kernel_sigma.max(0.85),
            true,
        );
        if let Some(mid) = proc.median_vol.get(proc.median_vol.len() / 2) {
            proc.iwork.clone_from(mid);
        }
    } else {
        let kernel = gaussian_kernel(param.kernel_sigma, KERNEL_MAXSIZE);
        let mut result = vec![0i16; nx * ny];
        byte_kernel_filter(
            &proc.iwork,
            &mut result,
            nx,
            nx,
            nx,
            ny,
            &kernel,
            kernel_dimension(param.kernel_sigma, KERNEL_MAXSIZE),
        );
        for (out, val) in proc.iwork.iter_mut().zip(result) {
            *out = val.clamp(0, 255) as u8;
        }
    }
    format!("clip smooth -l {}", param.kernel_sigma)
}
/// `sharpen_cb`.
pub fn sharpen_cb(proc: &mut ImodIproc) -> String {
    let original = proc.iwork.clone();
    for i in 1..proc.iwork.len().saturating_sub(1) {
        proc.iwork[i] = (2 * original[i] as i16
            - ((original[i - 1] as i16 + original[i + 1] as i16) / 2))
            .clamp(0, 255) as u8;
    }
    "clip sharpen".into()
}
/// `fourFilt_cb`.
pub fn four_filt_cb(
    proc: &mut ImodIproc,
    param: &IprocParam,
    boundary: &mut dyn IprocBoundary,
) -> String {
    let command = format!(
        "mtffilter -high {:.3} -low {:.3},{:.3}",
        param.sigma1, param.radius2, param.sigma2
    );
    if proc.filter_slicer {
        proc.slicer_filt_applied = true;
    } else {
        boundary.apply_fourier(&mut proc.iwork, param);
    }
    command
}
/// `fft_cb`.
pub fn fft_cb(
    proc: &mut ImodIproc,
    param: &IprocParam,
    boundary: &mut dyn IprocBoundary,
) -> String {
    let (scale, xcen, ycen) = boundary.apply_fft(&mut proc.iwork, param);
    proc.fft_scale = scale;
    proc.fft_xcen = xcen;
    proc.fft_ycen = ycen;
    proc.output_mode = 4;
    if param.fft_binning > 1 {
        "Cannot do FFT with binning in one operation".into()
    } else if param.fft_subset {
        "Cannot do FFT on subset in one operation".into()
    } else {
        "clip fft -2d".into()
    }
}
/// `median_cb`.
pub fn median_cb(
    proc: &mut ImodIproc,
    param: &IprocParam,
    boundary: &mut dyn IprocBoundary,
) -> String {
    boundary.apply_median(&mut proc.iwork, &proc.median_vol, param);
    format!(
        "clip median -{}d -n {}",
        if param.median_3d { 3 } else { 2 },
        param.median_size
    )
}
/// `anisoDiff_cb`.
pub fn aniso_diff_cb(
    proc: &mut ImodIproc,
    param: &mut IprocParam,
    boundary: &mut dyn IprocBoundary,
) -> String {
    boundary.apply_aniso_diffusion(&mut proc.iwork, param);
    format!(
        "clip diffusion -cc {} -k {:.5} -l {:.3} -n {}",
        param.andf_stop_func + 2,
        param.andf_k,
        param.andf_lambda,
        param.andf_iterations
    )
}

/// `setSliceMinMax`.
pub fn set_slice_min_max(
    actual: bool,
    data: &[u8],
    depth: i32,
    rampbase: i32,
    rampsize: i32,
) -> (f32, f32) {
    if actual {
        data.iter().fold((255., 0.), |(lo, hi), &v| {
            (lo.min(v as f32), hi.max(v as f32))
        })
    } else if depth == 8 {
        (rampbase as f32, (rampbase + rampsize - 1) as f32)
    } else {
        (0., 255.)
    }
}
/// `modeChangeStr`.
pub fn mode_change_str(
    proc: &mut ImodIproc,
    param: &IprocParam,
    amin: f32,
    amax: f32,
    option: &str,
) -> String {
    if proc.input_mode == 0 {
        proc.output_mode = 1;
    } else if proc.input_mode == 4 {
        proc.output_mode = 2;
    } else if proc.input_mode == 1 || proc.input_mode == 6 {
        if !proc.was_byte
            && ((param.proc_num != 1 && (amin < -10000. || amax > 10000.))
                || (param.proc_num == 1 && (amin < -30000. || amax > 30000.)))
        {
            proc.output_mode = 2;
        } else if proc.input_mode == 6 {
            proc.output_mode = 1;
        }
    }
    if proc.output_mode != proc.input_mode {
        format!(" -{} {}", option, proc.output_mode)
    } else {
        String::new()
    }
}
/// `clipFFTtoRealStr`.
pub fn clip_fft_to_real_str(proc: &mut ImodIproc) -> String {
    if proc.input_mode == 4 {
        proc.output_mode = 2;
        " -m 2".into()
    } else {
        String::new()
    }
}
/// `cannotDoFFTStr`.
pub fn cannot_do_fft_str(proc: &ImodIproc, command: &mut String, operation: &str) {
    if proc.input_mode == 4 {
        *command = format!("Cannot run {} on FFT data", operation);
    }
}

impl IprocWindow {
    /// `IProcWindow::IProcWindow`.
    pub fn new() -> Self {
        Self::default()
    }
    /// `autoApplyToggled` / `autoSaveToggled` / `applyThreshToggled`.
    pub fn auto_apply_toggled(&mut self, proc: &mut ImodIproc, state: bool) {
        proc.auto_apply = state;
    }
    pub fn auto_save_toggled(&mut self, proc: &mut ImodIproc, state: bool) {
        proc.auto_save = state;
    }
    pub fn apply_thresh_toggled(&mut self, proc: &mut ImodIproc, state: bool) {
        proc.apply_thresh_change = state;
    }
    pub fn thresh_changed(&mut self, param: &mut IprocParam, value: i32) {
        param.threshold = value;
    }
    pub fn four_filt_changed(&mut self, param: &mut IprocParam, which: i32, value: i32) {
        if which == 0 {
            self.set_sigma1_slider(param, value as f32 / self.sigma1_scale);
        } else if which == 1 {
            param.radius2 = 0.001 * value as f32;
        } else if which == 2 {
            param.sigma2 = 0.001 * value as f32;
        }
    }
    pub fn filt_slicer_toggled(&mut self, proc: &mut ImodIproc, state: bool) {
        proc.filter_slicer = state;
    }
    /// `setSigma1Slider`.
    pub fn set_sigma1_slider(&mut self, param: &mut IprocParam, mut value: f32) {
        let (scale, max) = if value <= 0.00101 {
            (10000., 360)
        } else if value <= 0.00201 {
            (5000., 300)
        } else if value <= 0.00501 {
            (2000., 240)
        } else {
            (1000., 200)
        };
        let range = max as f32 / scale;
        if self.last_four_filt_range < 0.19 && value > 0.0055 {
            value = range * value / self.last_four_filt_range;
        }
        self.last_four_filt_range = range;
        self.sigma1_scale = scale;
        let mut new_value = (scale * value).round() as i32;
        if ((new_value as f32 / scale) - param.sigma1).abs() < 1.0e-5 {
            if value > param.sigma1 {
                new_value += 1;
            }
            if value < param.sigma1 {
                new_value = (new_value - 1).max(0);
            }
        }
        param.sigma1 = new_value as f32 / scale;
    }
    pub fn kernel_changed(&mut self, param: &mut IprocParam, value: f64) {
        param.kernel_sigma = value as f32;
    }
    pub fn smooth_3d_changed(&mut self, param: &mut IprocParam, state: bool) {
        param.smooth_3d = state;
    }
    pub fn scale_smth_toggled(&mut self, param: &mut IprocParam, state: bool) {
        param.rescale_smooth = state;
    }
    pub fn binning_changed(&mut self, param: &mut IprocParam, value: i32) {
        param.fft_binning = value;
    }
    pub fn subset_changed(&mut self, param: &mut IprocParam, state: bool) {
        param.fft_subset = state;
    }
    pub fn grow_changed(&mut self, param: &mut IprocParam, state: bool) {
        param.thresh_grow = state;
    }
    pub fn shrink_changed(&mut self, param: &mut IprocParam, state: bool) {
        param.thresh_shrink = state;
    }
    pub fn filter_highlighted(&mut self, param: &mut IprocParam, which: i32) {
        param.proc_num = which;
        self.selected_filter = which;
    }
    pub fn filter_selected(&mut self, param: &mut IprocParam, which: i32) {
        self.filter_highlighted(param, which);
    }
    pub fn edge_selected(&mut self, param: &mut IprocParam, which: i32) {
        param.edge = which;
    }
    pub fn med_size_changed(&mut self, param: &mut IprocParam, value: i32) {
        param.median_size = value;
    }
    pub fn med_3d_changed(&mut self, param: &mut IprocParam, state: bool) {
        param.median_3d = state;
    }
    pub fn andf_iter_changed(&mut self, param: &mut IprocParam, value: i32) {
        param.andf_iterations = value;
    }
    pub fn andf_func_clicked(&mut self, param: &mut IprocParam, value: i32) {
        param.andf_stop_func = value;
    }
    pub fn andf_k_entered(&mut self, param: &mut IprocParam, value: f64) {
        param.andf_k = value;
    }
    /// `limitFFTbinning`.
    pub fn limit_fft_binning(&mut self, param: &mut IprocParam, nx: usize, ny: usize) {
        param.fft_binning = param.fft_binning.min(16.min(nx.min(ny)) as i32);
    }
    /// `calcFileThreshold`.
    pub fn calc_file_threshold(
        &mut self,
        proc: &mut ImodIproc,
        param: &IprocParam,
        ushort: bool,
        smin: f32,
        smax: f32,
        mode: i32,
    ) {
        let (range, low, high) = if ushort {
            (65535., proc.range_low as f32, proc.range_high as f32)
        } else {
            (255., 0., 255.)
        };
        let mut value =
            ((param.threshold as f32 + 0.5) * (high - low) / 255. + low) * (smax - smin) / range
                + smin;
        if mode != 2 && mode != 4 {
            let old = value;
            value = value.floor();
            if old - value < 1.0e-6 * value {
                value -= 1.;
            }
        }
        proc.file_threshold = value;
    }
    /// `apply`, `startProcess`, and `finishProcess` in source order, synchronously scheduled by the host.
    pub fn apply(
        &mut self,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
        use_stack: bool,
    ) {
        let (section, time) = boundary.current_section_time();
        if use_stack && !self.param_stack.is_empty() {
            self.saved_param = param.clone();
            if self
                .param_stack
                .last()
                .is_some_and(|p| p.proc_num == param.proc_num)
            {
                *self.param_stack.last_mut().unwrap() = param.clone();
            }
            *param = self.param_stack[0].clone();
            self.use_stack_ind = 0;
        } else {
            self.param_stack.clear();
            self.param_stack.push(param.clone());
            self.use_stack_ind = -1;
        }
        clearsec(proc, boundary);
        self.command_list.clear();
        self.data_modes = vec![boundary.image_mode()];
        proc.was_byte = boundary.image_mode() == 0;
        param.andf_iter_done = 0;
        if section != proc.idata_sec || time != proc.idata_time {
            proc.idata_sec = section;
            proc.idata_time = time;
            savesec(proc, boundary);
        }
        self.start_process(proc, param, boundary);
    }
    pub fn start_process(
        &mut self,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
    ) {
        proc.fft_scale = 0.;
        proc.input_mode = *self.data_modes.last().unwrap_or(&0);
        proc.output_mode = proc.input_mode;
        self.running_proc = true;
        self.run_current_filter(proc, param, boundary);
        self.finish_process(proc, param, boundary);
    }
    pub fn run_current_filter(
        &mut self,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
    ) {
        let (nx, ny, _) = boundary.dimensions();
        let command = match param.proc_num {
            0 => fft_cb(proc, param, boundary),
            1 => four_filt_cb(proc, param, boundary),
            2 => smooth_cb(proc, param, nx, ny),
            3 => median_cb(proc, param, boundary),
            4 => aniso_diff_cb(proc, param, boundary),
            5 => edge_cb(proc, param, boundary),
            6 => sharpen_cb(proc),
            7 => thresh_cb(proc, param, 16, 0, 256),
            _ => String::new(),
        };
        self.command_list.push(command);
    }
    pub fn finish_process(
        &mut self,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
    ) {
        proc.modified = true;
        copy_and_display(proc, boundary);
        self.data_modes.push(proc.output_mode);
        self.running_proc = false;
        if self.use_stack_ind >= 0 {
            self.use_stack_ind += 1;
            if (self.use_stack_ind as usize) < self.param_stack.len() {
                *param = self.param_stack[self.use_stack_ind as usize].clone();
                self.start_process(proc, param, boundary);
            } else {
                self.use_stack_ind = -1;
                *param = self.saved_param.clone();
            }
        }
        if let Some(callback) = self.callback.take() {
            callback();
        }
    }
    pub fn button_clicked(
        &mut self,
        which: usize,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
    ) {
        match which {
            APPLY_BUT => {
                proc.save_proc_num = -1;
                self.apply(proc, param, boundary, false);
            }
            MORE_BUT => {
                self.param_stack.push(param.clone());
                self.start_process(proc, param, boundary);
            }
            LESS_BUT if self.param_stack.len() > 1 => {
                self.param_stack.pop();
                self.command_list.pop();
                self.data_modes.pop();
                self.apply(proc, param, boundary, true);
            }
            DO_SAME_BUT if !self.param_stack.is_empty() => self.apply(proc, param, boundary, true),
            TOGGLE_BUT => {
                proc.toggling = false;
                copy_and_display(proc, boundary);
            }
            RESET_BUT => {
                clearsec(proc, boundary);
                boundary.redraw_image();
            }
            SAVE_BUT => {
                proc.modified = false;
                proc.idata_sec = -1;
            }
            LIST_BUT => boundary.dialog_message(&self.command_list.join("\n")),
            _ => {}
        }
    }
    pub fn button_pressed(
        &mut self,
        which: usize,
        proc: &mut ImodIproc,
        boundary: &mut dyn IprocBoundary,
    ) {
        if which == TOGGLE_BUT && proc.modified {
            saved_to_image(proc, boundary);
            proc.toggling = true;
            boundary.redraw_image();
        }
    }
    pub fn timer_event(&mut self) {}
    pub fn top_change_event(&mut self) {}
    pub fn top_close_event(&mut self, proc: &mut ImodIproc, boundary: &mut dyn IprocBoundary) {
        if !self.running_proc {
            clearsec(proc, boundary);
            free_arrays(proc);
            boundary.redraw_image();
        }
    }
    pub fn key_press_event(
        &mut self,
        key: char,
        proc: &mut ImodIproc,
        param: &mut IprocParam,
        boundary: &mut dyn IprocBoundary,
    ) {
        if key == 'A' {
            self.apply(proc, param, boundary, false);
        } else if key == 'B' {
            self.button_clicked(MORE_BUT, proc, param, boundary);
        }
    }
    pub fn key_release_event(&mut self) {}
}

/// `iprocRethink`.
pub fn iproc_rethink(proc: &mut ImodIproc, nx: usize, ny: usize) -> i32 {
    let Some(n) = nx.checked_mul(ny) else {
        return 1;
    };
    proc.isaved = vec![0; n];
    proc.iwork = vec![0; n];
    0
}
/// `iprocUpdate`.
pub fn iproc_update(
    window: &mut IprocWindow,
    proc: &mut ImodIproc,
    param: &mut IprocParam,
    boundary: &mut dyn IprocBoundary,
) {
    let (sec, time) = boundary.current_section_time();
    if !window.running_proc
        && window.use_stack_ind < 0
        && (sec != proc.idata_sec || time != proc.idata_time)
    {
        if proc.auto_save {
            proc.modified = false;
            proc.idata_sec = -1;
        }
        if proc.auto_apply {
            window.apply(proc, param, boundary, true);
        }
    }
}
pub fn iproc_apply(
    window: &mut IprocWindow,
    proc: &mut ImodIproc,
    param: &mut IprocParam,
    boundary: &mut dyn IprocBoundary,
) {
    if !window.running_proc {
        window.apply(proc, param, boundary, false);
    }
}
pub fn iproc_is_open(open: bool) -> bool {
    open
}
pub fn iproc_command_list(window: Option<&IprocWindow>) -> Vec<String> {
    window.map_or_else(Vec::new, |w| w.command_list.clone())
}
pub fn iproc_filter_for_slicer(
    window: Option<&IprocWindow>,
    proc: &ImodIproc,
    param: &IprocParam,
) -> (bool, f32, f32, f32, f32) {
    (
        window.is_some_and(|w| {
            (proc.filter_slicer || w.param_stack.len() > 1)
                && proc.slicer_filt_applied
                && !proc.toggling
        }),
        param.radius1,
        param.sigma1,
        param.radius2,
        param.sigma2,
    )
}
pub fn input_iproc_open(
    proc: &mut ImodIproc,
    nx: usize,
    ny: usize,
    pixel_bytes: usize,
) -> Result<IprocWindow, i32> {
    if nx.saturating_mul(ny) > 2_147_000_000
        || nx
            .checked_mul(ny)
            .and_then(|n| n.checked_mul(pixel_bytes))
            .is_none()
    {
        return Err(1);
    }
    iproc_rethink(proc, nx, ny);
    Ok(IprocWindow::new())
}
pub fn iproc_toggle_full_fft(
    window: &mut IprocWindow,
    proc: &mut ImodIproc,
    param: &mut IprocParam,
    boundary: &mut dyn IprocBoundary,
) {
    if proc.modified {
        clearsec(proc, boundary);
        boundary.redraw_image();
    } else {
        proc.save_proc_num = param.proc_num;
        param.proc_num = 0;
        window.apply(proc, param, boundary, false);
    }
}
pub fn iproc_busy(window: Option<&IprocWindow>) -> bool {
    window.is_some_and(|w| w.running_proc)
}
pub fn iproc_call_when_free(window: Option<&mut IprocWindow>, callback: fn()) {
    if let Some(w) = window.filter(|w| w.running_proc) {
        w.callback = Some(callback);
    } else {
        callback();
    }
}

/// `cpdslice`.
pub fn cpdslice(source: &[u8], proc: &mut ImodIproc, depth: i32, rampbase: u8) {
    proc.iwork.clear();
    proc.iwork.extend(source.iter().map(|&v| {
        if depth > 8 {
            v
        } else {
            v.wrapping_add(rampbase)
        }
    }));
}
/// `copyAndDisplay`.
pub fn copy_and_display(proc: &ImodIproc, boundary: &mut dyn IprocBoundary) {
    boundary.replace_image_bytes(proc.idata_sec, proc.idata_time, &proc.iwork);
    boundary.redraw_image();
}
/// `clearsec`.
pub fn clearsec(proc: &mut ImodIproc, boundary: &mut dyn IprocBoundary) {
    if proc.idata_sec >= 0 && proc.modified {
        proc.iwork.clone_from(&proc.isaved);
        let _ = saved_to_image(proc, boundary);
        proc.modified = false;
        proc.slicer_filt_applied = false;
    }
}
/// `savesec`.
pub fn savesec(proc: &mut ImodIproc, boundary: &mut dyn IprocBoundary) {
    if proc.idata_sec >= 0 {
        if let Some(image) = boundary.image_bytes(proc.idata_sec, proc.idata_time) {
            proc.iwork = image.clone();
            proc.isaved = image;
        }
    }
}
/// `imageToBuffer`.
pub fn image_to_buffer(proc: &mut ImodIproc, image: &[u8]) {
    proc.iwork.clear();
    proc.iwork.extend_from_slice(image);
}
/// `savedToImage`.
pub fn saved_to_image(proc: &ImodIproc, boundary: &mut dyn IprocBoundary) -> i32 {
    if proc.idata_sec < 0 {
        1
    } else {
        boundary.replace_image_bytes(proc.idata_sec, proc.idata_time, &proc.isaved);
        0
    }
}
/// `fillMedianVol`.
pub fn fill_median_vol(
    proc: &mut ImodIproc,
    z_start: i32,
    z_end: i32,
    boundary: &mut dyn IprocBoundary,
) -> i32 {
    proc.median_vol.clear();
    for z in z_start..=z_end {
        let Some(data) = boundary.image_bytes(z, proc.idata_time) else {
            proc.median_vol.clear();
            return 1;
        };
        proc.median_vol.push(data);
    }
    0
}
/// `freeMedianVol`.
pub fn free_median_vol(proc: &mut ImodIproc) {
    proc.median_vol.clear();
}
/// `freeArrays`.
pub fn free_arrays(proc: &mut ImodIproc) {
    proc.isaved.clear();
    proc.iwork.clear();
    proc.andf_image.clear();
    proc.andf_image2.clear();
    proc.median_vol.clear();
}
/// `setUnscaledK`.
pub fn set_unscaled_k(param: &IprocParam, slope: f32) -> String {
    format!("unscaled: {:.5}", param.andf_k / slope as f64)
}

/// `volByteSmooth`.
pub fn vol_byte_smooth(
    vol: &mut [Vec<u8>],
    nx: usize,
    ny: usize,
    sigma: f32,
    mid_z_only: bool,
) -> i32 {
    if vol.is_empty() || vol.iter().any(|p| p.len() < nx * ny) {
        return 1;
    }
    let dim = kernel_dimension(sigma, KERNEL_MAXSIZE);
    let kernel = gaussian_kernel(sigma, dim);
    let mut filtered: Vec<Vec<i16>> = vol
        .iter()
        .map(|p| {
            let mut out = vec![0; nx * ny];
            byte_kernel_filter(p, &mut out, nx, nx, nx, ny, &kernel, dim);
            out
        })
        .collect();
    let zkernel: Vec<i32> = (0..dim).map(|i| kernel[i + (dim / 2) * dim]).collect();
    let zsum: i32 = zkernel.iter().sum();
    let kernel_sum: i32 = kernel.iter().sum::<i32>() * zsum;
    if kernel_sum == 0 {
        return 1;
    }
    let start = if mid_z_only { vol.len() / 2 } else { 0 };
    let end = if mid_z_only { start } else { vol.len() - 1 };
    for z in start..=end {
        for ind in 0..nx * ny {
            let sum: i32 = (0..dim)
                .map(|k| {
                    let zi = (z as isize + k as isize - dim as isize / 2)
                        .clamp(0, vol.len() as isize - 1) as usize;
                    zkernel[k] * filtered[zi][ind] as i32
                })
                .sum();
            vol[z][ind] = ((sum + (kernel_sum - 1) / 2) / kernel_sum).clamp(0, 255) as u8;
        }
    }
    0
}
/// `byteKernelFilter`.
pub fn byte_kernel_filter(
    array: &[u8],
    brray: &mut [i16],
    nx_adim: usize,
    nx_bdim: usize,
    nx: usize,
    ny: usize,
    mat: &[i32],
    kdim: usize,
) {
    let below = kdim / 2;
    for iyo in 0..ny {
        for ixo in 0..nx {
            let mut sum = 0i32;
            for iy in 0..kdim {
                let aiy = (iyo as isize + iy as isize - below as isize).clamp(0, ny as isize - 1)
                    as usize;
                for ix in 0..kdim {
                    let aix = (ixo as isize + ix as isize - below as isize)
                        .clamp(0, nx as isize - 1) as usize;
                    sum += mat[ix + iy * kdim] * array[aix + aiy * nx_adim] as i32;
                }
            }
            brray[ixo + nx_bdim * iyo] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

/// Private source-equivalent kernel setup used by `smooth_cb`/`volByteSmooth`.
fn kernel_dimension(sigma: f32, max: usize) -> usize {
    if sigma <= 0.75 {
        3
    } else if sigma <= 1.25 {
        5.min(max)
    } else {
        7.min(max)
    }
}
/// Private equivalent of sliceproc's `scaledGaussianKernel`.
fn gaussian_kernel(sigma: f32, dim: usize) -> Vec<i32> {
    let sigma = sigma.max(0.01);
    let half = dim as f32 / 2.;
    let vals: Vec<f32> = (0..dim)
        .flat_map(|y| {
            (0..dim).map(move |x| {
                let dx = x as f32 - half.floor();
                let dy = y as f32 - half.floor();
                (-((dx * dx + dy * dy) / (2. * sigma * sigma))).exp()
            })
        })
        .collect();
    vals.into_iter()
        .map(|v| (v * 100.).round() as i32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_kernel_clamps_edges_as_source() {
        let a = [1, 2, 3, 4];
        let mut b = [0; 4];
        byte_kernel_filter(&a, &mut b, 2, 2, 2, 2, &[1; 9], 3);
        assert_eq!(b, [18, 21, 24, 27]);
    }
    #[test]
    fn smoothing_middle_plane_only() {
        let mut v = vec![vec![0; 9], vec![255; 9], vec![0; 9]];
        assert_eq!(vol_byte_smooth(&mut v, 3, 3, 0.4, true), 0);
        assert_eq!(v[0], vec![0; 9]);
        assert!(v[1][4] < 255);
    }
    #[test]
    fn slider_has_source_scales() {
        let mut w = IprocWindow::new();
        let mut p = IprocParam::default();
        w.set_sigma1_slider(&mut p, 0.001);
        assert_eq!(w.sigma1_scale, 10000.);
        w.set_sigma1_slider(&mut p, 0.01);
        assert_eq!(w.sigma1_scale, 1000.);
    }
}
