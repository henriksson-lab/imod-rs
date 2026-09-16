//! Owned request/reply translation of `shrmemframe.{h,cpp}`.
pub const SHRMEMFRAME_VERSION: i32 = 101;
pub const MESS_BUF_SIZE: usize = 4096;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShrMemFrameAction {
    Done,
    Initialize,
    NextFrame,
    Finish,
    FrcCross,
    GpuAvail,
    Cleanup,
    Version,
    Exit,
}
#[derive(Clone, Debug)]
pub struct InitializeParams {
    pub ret_val: i32,
    pub bin_sum: i32,
    pub bin_align: i32,
    pub trim_frac: f32,
    pub num_all_vs_all: i32,
    pub cum_align_at_end: i32,
    pub use_hybrid: i32,
    pub defer_sum: i32,
    pub group_size: i32,
    pub nx: usize,
    pub ny: usize,
    pub pad_frac: f32,
    pub taper_frac: f32,
    pub anti_filt_type: i32,
    pub radius1: f32,
    pub radius2: Vec<f32>,
    pub sigma1: f32,
    pub sigma2: Vec<f32>,
    pub num_filters: i32,
    pub max_shift: i32,
    pub k_factor: f32,
    pub max_max_weight: f32,
    pub summing_mode: i32,
    pub expected_z: i32,
    pub make_unwgt_sum: i32,
    pub gpu_flags: i32,
    pub debug: i32,
    pub messages: String,
}
#[derive(Clone, Debug)]
pub struct NextFrameParams {
    pub ret_val: i32,
    pub frame: Vec<u8>,
    pub pixel_type: i32,
    pub gain_reference: Option<Vec<f32>>,
    pub defects: Option<String>,
    pub trunc_limit: f32,
    pub camera_size_x: i32,
    pub camera_size_y: i32,
    pub defect_binning: i32,
    pub shift_x: f32,
    pub shift_y: f32,
    pub messages: String,
}
#[derive(Clone, Debug)]
pub struct FinishParams {
    pub ret_val: i32,
    pub aligned_sum: Vec<f32>,
    pub x_shifts: Vec<f32>,
    pub y_shifts: Vec<f32>,
    pub raw_x_shifts: Vec<f32>,
    pub raw_y_shifts: Vec<f32>,
    pub ring_corrs: Vec<f32>,
    pub refine_radius2: f32,
    pub refine_sigma2: f32,
    pub iter_crit: f32,
    pub group_refine: i32,
    pub do_spline: i32,
    pub delta_r: f32,
    pub best_filt: i32,
    pub smooth_dist: Vec<f32>,
    pub raw_dist: Vec<f32>,
    pub res_mean: Vec<f32>,
    pub res_sd: Vec<f32>,
    pub mean_res_max: Vec<f32>,
    pub max_res_max: Vec<f32>,
    pub mean_raw_max: Vec<f32>,
    pub max_raw_max: Vec<f32>,
    pub messages: String,
}
#[derive(Clone, Debug)]
pub struct FrcParams {
    pub ring_corrs: Vec<f32>,
    pub frc_delta: f32,
    pub half_cross: f32,
    pub quart_cross: f32,
    pub eighth_cross: f32,
    pub half_nyq: f32,
}
#[derive(Clone, Debug)]
pub struct GpuParams {
    pub ret_val: i32,
    pub n_gpu: i32,
    pub memory: f32,
    pub debug: i32,
    pub messages: String,
}
#[derive(Clone, Debug)]
pub enum ShrMemFrameRequest {
    Initialize(InitializeParams),
    NextFrame(NextFrameParams),
    Finish(FinishParams),
    FrcCross(FrcParams),
    GpuAvail(GpuParams),
    Cleanup,
    Version,
    Exit,
    Done,
}
#[derive(Clone, Debug)]
pub struct ShrMemFrameReply {
    pub id: i32,
    pub action: ShrMemFrameAction,
    pub request: ShrMemFrameRequest,
}
/// Safe replacement for the source `FrameAlign` dependency and Windows mapping transport.
pub trait FrameAlignBackend {
    fn initialize(&mut self, params: &mut InitializeParams);
    fn next_frame(&mut self, params: &mut NextFrameParams);
    fn finish_align_and_sum(&mut self, params: &mut FinishParams);
    fn analyze_frc_crossings(&mut self, params: &mut FrcParams);
    fn gpu_available(&mut self, params: &mut GpuParams);
    fn cleanup(&mut self);
}
pub struct ShrMemFrame<B> {
    pub id_value: i32,
    pub backend: B,
    pub nx: usize,
    pub ny: usize,
    pub num_frames: usize,
    pub defect_string: Option<String>,
    pub messages: String,
    pub running: bool,
}
impl<B: FrameAlignBackend> ShrMemFrame<B> {
    pub fn new(id_value: i32, backend: B) -> Self {
        Self {
            id_value,
            backend,
            nx: 0,
            ny: 0,
            num_frames: 0,
            defect_string: None,
            messages: String::new(),
            running: true,
        }
    }
    pub fn frame_print(&mut self, message: &str) {
        if self.messages.len() + message.len() < MESS_BUF_SIZE - 1 {
            self.messages.push_str(message)
        }
    }
    pub fn handle(&mut self, id: i32, mut request: ShrMemFrameRequest) -> Option<ShrMemFrameReply> {
        if id != self.id_value {
            self.running = false;
            return None;
        }
        match &mut request {
            ShrMemFrameRequest::Initialize(p) => {
                self.messages.clear();
                self.backend.initialize(p);
                p.messages.clone_from(&self.messages);
                self.nx = p.nx;
                self.ny = p.ny;
                self.num_frames = 0
            }
            ShrMemFrameRequest::NextFrame(p) => {
                self.messages.clear();
                if self.num_frames == 0 {
                    self.defect_string = p.defects.clone()
                }
                self.backend.next_frame(p);
                p.messages.clone_from(&self.messages);
                self.num_frames += 1
            }
            ShrMemFrameRequest::Finish(p) => {
                self.messages.clear();
                self.backend.finish_align_and_sum(p);
                p.messages.clone_from(&self.messages)
            }
            ShrMemFrameRequest::FrcCross(p) => self.backend.analyze_frc_crossings(p),
            ShrMemFrameRequest::GpuAvail(p) => {
                self.messages.clear();
                self.backend.gpu_available(p);
                p.messages.clone_from(&self.messages)
            }
            ShrMemFrameRequest::Cleanup => {
                self.messages.clear();
                self.backend.cleanup()
            }
            ShrMemFrameRequest::Version => {
                return Some(ShrMemFrameReply {
                    id: self.id_value,
                    action: ShrMemFrameAction::Done,
                    request: ShrMemFrameRequest::Version,
                });
            }
            ShrMemFrameRequest::Exit => self.running = false,
            ShrMemFrameRequest::Done => {}
        }
        Some(ShrMemFrameReply {
            id: self.id_value,
            action: ShrMemFrameAction::Done,
            request,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Backend;
    impl FrameAlignBackend for Backend {
        fn initialize(&mut self, _: &mut InitializeParams) {}
        fn next_frame(&mut self, _: &mut NextFrameParams) {}
        fn finish_align_and_sum(&mut self, _: &mut FinishParams) {}
        fn analyze_frc_crossings(&mut self, _: &mut FrcParams) {}
        fn gpu_available(&mut self, _: &mut GpuParams) {}
        fn cleanup(&mut self) {}
    }
    #[test]
    fn foreign_id_stops_service() {
        let mut s = ShrMemFrame::new(3, Backend);
        assert!(s.handle(4, ShrMemFrameRequest::Done).is_none());
        assert!(!s.running)
    }
}
