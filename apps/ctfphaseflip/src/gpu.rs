use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Uniform parameters passed to the CTF correction compute shader.
/// Must match the WGSL `Params` struct layout exactly (16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CtfParams {
    pub sw: u32,
    pub sh: u32,
    pub pixel_a: f32,
    pub wavelength: f32,
    pub cs_a: f32,
    pub amp_contrast: f32,
    pub cuton_freq: f32,
    pub defocus1: f32,
    pub defocus2: f32,
    pub astig_angle_rad: f32,
    pub has_astigmatism: u32,
    pub plate_phase_rad: f32,
    pub tilt_axis_rad: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Manages GPU resources for CTF phase-flip correction of 2D FFT strips.
pub struct GpuCtfSession {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuCtfSession {
    /// Try to initialise the GPU pipeline. Returns `None` if no suitable adapter
    /// is found or device creation fails.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        eprintln!("ctfphaseflip: GPU adapter: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ctfphaseflip-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        // --- Shader module ---
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ctf_correct"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("ctf_correct.wgsl"))),
        });

        // --- Bind group layout ---
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ctf_layout"),
                entries: &[
                    bgl_storage_rw(0), // strip_re
                    bgl_storage_rw(1), // strip_im
                    bgl_uniform(2),    // params
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ctf_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ctf_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("ctf_correct"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Apply CTF phase-flip correction on the GPU for one strip.
    /// `strip_re` and `strip_im` are the real and imaginary parts of the 2D FFT
    /// (row-major, sw*sh elements). They are modified in-place.
    pub fn correct_strip(
        &self,
        strip_re: &mut [f32],
        strip_im: &mut [f32],
        params: CtfParams,
    ) {
        let sw = params.sw as usize;
        let sh = params.sh as usize;
        let n = sw * sh;
        assert_eq!(strip_re.len(), n);
        assert_eq!(strip_im.len(), n);

        let buf_size = (n * std::mem::size_of::<f32>()) as u64;

        // Upload real and imaginary parts
        let re_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("strip_re"),
                contents: bytemuck::cast_slice(strip_re),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let im_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("strip_im"),
                contents: bytemuck::cast_slice(strip_im),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ctf_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ctf_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ctf_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ctf_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (sw as u32 + 15) / 16;
            let wg_y = (sh as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy results to staging buffers for readback
        let staging_re = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_re"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_im = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_im"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&re_buffer, 0, &staging_re, 0, buf_size);
        encoder.copy_buffer_to_buffer(&im_buffer, 0, &staging_im, 0, buf_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back real part
        {
            let slice = staging_re.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            receiver.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            strip_re.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging_re.unmap();

        // Read back imaginary part
        {
            let slice = staging_im.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            receiver.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            strip_im.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging_im.unmap();
    }
}

// --- Helper functions for bind group layout entries ---

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
