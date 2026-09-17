//! Safe, in-process translation of the `ShrMemClient` protocol client.
//!
//! The C++ unit is a Windows plug-in transport: it creates a process, maps a
//! byte buffer, and signals two Win32 events.  The protocol itself is retained
//! here, while Rust ownership replaces the process handles and mapped pointers.

use super::shrmemframe::{
    FinishParams, FrameAlignBackend, FrcParams, GpuParams, InitializeParams, NextFrameParams,
    ShrMemFrame, ShrMemFrameRequest,
};

#[derive(Clone, Debug, Default)]
pub struct FinishAlignment {
    pub aligned_sum: Vec<f32>,
    pub x_shifts: Vec<f32>,
    pub y_shifts: Vec<f32>,
    pub raw_x_shifts: Vec<f32>,
    pub raw_y_shifts: Vec<f32>,
    pub ring_corrs: Vec<f32>,
    pub best_filt: i32,
    pub smooth_dist: Vec<f32>,
    pub raw_dist: Vec<f32>,
    pub res_mean: Vec<f32>,
    pub res_sd: Vec<f32>,
    pub mean_res_max: Vec<f32>,
    pub max_res_max: Vec<f32>,
    pub mean_raw_max: Vec<f32>,
    pub max_raw_max: Vec<f32>,
}

/// `TickInterval()`: elapsed milliseconds between Win32 `GetTickCount`
/// readings.  `wrapping_sub` retains the native 32-bit tick-counter rollover
/// behavior without requiring a platform clock in protocol tests.
pub fn tick_interval(start_tick: u32, current_tick: u32) -> f64 {
    current_tick.wrapping_sub(start_tick) as f64
}

pub struct ShrMemClient<B> {
    server: ShrMemFrame<B>,
    num_frames: usize,
    nx: usize,
    ny: usize,
    server_id: i32,
    messages: Vec<String>,
    frame_buffer: Vec<u8>,
}

/// `ShrMemClient()`: construct the client around its owned Rust service.
/// The original no-argument constructor subsequently launched a Windows
/// process; requiring that service here makes the replacement transport
/// explicit and keeps the client usable on every platform.
pub fn shr_mem_client<B: FrameAlignBackend>(server: ShrMemFrame<B>) -> ShrMemClient<B> {
    ShrMemClient::new(server)
}

/// `~ShrMemClient()`: first send the native exit action, then release the
/// owned service and protocol buffers.
pub fn free_shr_mem_client<B: FrameAlignBackend>(mut client: ShrMemClient<B>) {
    client.disconnect(50);
}

impl<B: FrameAlignBackend> ShrMemClient<B> {
    /// Source constructor.  The supplied owned service is the Rust equivalent
    /// of launching and connecting to `shrmemframe.exe`.
    pub fn new(server: ShrMemFrame<B>) -> Self {
        let server_id = server.id_value;
        Self {
            server,
            num_frames: 0,
            nx: 0,
            ny: 0,
            server_id,
            messages: Vec::new(),
            frame_buffer: Vec::new(),
        }
    }

    /// Source `ConnectIfNeeded` checks that the remote endpoint is live.
    pub fn connect_if_needed(&self) -> Result<(), String> {
        if self.server.running {
            Ok(())
        } else {
            Err("shrmemframe is not running".into())
        }
    }

    /// Source `Disconnect`: route an exit action, then let Rust release state.
    pub fn disconnect(&mut self, _wait_millis: i32) {
        let _ = self.server.handle(self.server_id, ShrMemFrameRequest::Exit);
        self.num_frames = 0;
        self.nx = 0;
        self.ny = 0;
    }

    /// Source `initialize`.
    pub fn initialize(&mut self, params: InitializeParams) -> Result<i32, String> {
        self.connect_if_needed()?;
        let nx = params.nx;
        let ny = params.ny;
        let reply = self
            .server
            .handle(self.server_id, ShrMemFrameRequest::Initialize(params))
            .ok_or_else(|| "shrmemframe rejected initialization".to_owned())?;
        let ShrMemFrameRequest::Initialize(params) = reply.request else {
            return Err("shrmemframe returned an invalid initialization reply".into());
        };
        self.relay_messages(&params.messages);
        self.num_frames = 0;
        self.nx = nx;
        self.ny = ny;
        self.frame_buffer
            .resize(nx.saturating_mul(ny).saturating_mul(4), 0);
        Ok(params.ret_val)
    }

    /// Source `gpuAvailable`.
    pub fn gpu_available(&mut self, params: GpuParams) -> Result<GpuParams, String> {
        self.connect_if_needed()?;
        let reply = self
            .server
            .handle(self.server_id, ShrMemFrameRequest::GpuAvail(params))
            .ok_or_else(|| "shrmemframe rejected GPU query".to_owned())?;
        let ShrMemFrameRequest::GpuAvail(params) = reply.request else {
            return Err("shrmemframe returned an invalid GPU reply".into());
        };
        self.relay_messages(&params.messages);
        Ok(params)
    }

    /// Source `nextFrame`.  The frame and optional gain reference are owned by
    /// the request instead of copied through offsets in a shared byte mapping.
    pub fn next_frame(&mut self, mut params: NextFrameParams) -> Result<i32, String> {
        self.check_if_process_died("shrmemframe died while processing a frame")?;
        if self.nx == 0 || self.ny == 0 {
            return Err("frame alignment was not initialized".into());
        }
        if params.frame.is_empty() {
            params.frame = self.frame_buffer.clone();
        } else {
            self.frame_buffer.clone_from(&params.frame);
        }
        let reply = self
            .server
            .handle(self.server_id, ShrMemFrameRequest::NextFrame(params))
            .ok_or_else(|| "shrmemframe rejected frame data".to_owned())?;
        let ShrMemFrameRequest::NextFrame(params) = reply.request else {
            return Err("shrmemframe returned an invalid frame reply".into());
        };
        self.relay_messages(&params.messages);
        self.num_frames += 1;
        Ok(params.ret_val)
    }

    /// Source `getFrameBuffer`; a caller can write directly into owned frame
    /// storage, then call `next_frame` with an empty frame vector.
    pub fn get_frame_buffer(&mut self) -> &mut Vec<u8> {
        &mut self.frame_buffer
    }

    /// Source `finishAlignAndSum`.
    pub fn finish_align_and_sum(
        &mut self,
        params: FinishParams,
    ) -> Result<(i32, FinishAlignment), String> {
        self.check_if_process_died("shrmemframe died after the last frame")?;
        let reply = self
            .server
            .handle(self.server_id, ShrMemFrameRequest::Finish(params))
            .ok_or_else(|| "shrmemframe rejected finish request".to_owned())?;
        let ShrMemFrameRequest::Finish(params) = reply.request else {
            return Err("shrmemframe returned an invalid finish reply".into());
        };
        self.relay_messages(&params.messages);
        Ok((
            params.ret_val,
            FinishAlignment {
                aligned_sum: params.aligned_sum,
                x_shifts: params.x_shifts,
                y_shifts: params.y_shifts,
                raw_x_shifts: params.raw_x_shifts,
                raw_y_shifts: params.raw_y_shifts,
                ring_corrs: params.ring_corrs,
                best_filt: params.best_filt,
                smooth_dist: params.smooth_dist,
                raw_dist: params.raw_dist,
                res_mean: params.res_mean,
                res_sd: params.res_sd,
                mean_res_max: params.mean_res_max,
                max_res_max: params.max_res_max,
                mean_raw_max: params.mean_raw_max,
                max_raw_max: params.max_raw_max,
            },
        ))
    }

    /// Source `cleanup`.
    pub fn cleanup(&mut self) -> Result<(), String> {
        self.check_if_process_died("shrmemframe died before cleanup")?;
        self.server
            .handle(self.server_id, ShrMemFrameRequest::Cleanup)
            .ok_or_else(|| "shrmemframe rejected cleanup".to_owned())?;
        Ok(())
    }

    /// Source `analyzeFRCcrossings`.
    pub fn analyze_frc_crossings(&mut self, params: FrcParams) -> Result<FrcParams, String> {
        self.check_if_process_died("shrmemframe died before FRC analysis")?;
        let reply = self
            .server
            .handle(self.server_id, ShrMemFrameRequest::FrcCross(params))
            .ok_or_else(|| "shrmemframe rejected FRC analysis".to_owned())?;
        let ShrMemFrameRequest::FrcCross(params) = reply.request else {
            return Err("shrmemframe returned an invalid FRC reply".into());
        };
        Ok(params)
    }

    /// Source `CheckIfProcessDied`.
    pub fn check_if_process_died(&self, message: &str) -> Result<(), String> {
        if self.server.running {
            Ok(())
        } else {
            Err(message.to_owned())
        }
    }

    /// Source `ClearProcessVars`; owned Rust state has no handles to close.
    pub fn clear_process_vars(&mut self, terminate: bool) {
        if terminate {
            self.server.running = false;
        }
    }

    /// Source `SetCodeAndWaitForReply`; dispatch is synchronous for the owned
    /// in-process server, so there are no platform events or timeout handles.
    pub fn set_code_and_wait_for_reply(
        &mut self,
        request: ShrMemFrameRequest,
        _timeout_millis: i32,
    ) -> Result<(), String> {
        self.server
            .handle(self.server_id, request)
            .map(|_| ())
            .ok_or_else(|| "shrmemframe returned no reply".to_owned())
    }

    /// Source `RelayMessages`.
    pub fn relay_messages(&mut self, message: &str) {
        if !message.is_empty() {
            self.messages.push(message.to_owned());
        }
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
    fn disconnect_stops_owned_service() {
        let mut client = ShrMemClient::new(ShrMemFrame::new(19, Backend));
        client.disconnect(0);
        assert!(client.connect_if_needed().is_err());
    }

    #[test]
    fn tick_interval_and_constructor_handle_tick_counter_rollover() {
        assert_eq!(tick_interval(u32::MAX - 4, 3), 8.0);
        let client = shr_mem_client(ShrMemFrame::new(20, Backend));
        free_shr_mem_client(client);
    }
}
