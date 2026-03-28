use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Uniform parameters passed to the compute shader.
/// Must match the WGSL `Params` struct layout exactly.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    nx: u32,
    nz: u32,
    n_projs: u32,
    in_nx: u32,
    in_ny: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    center_x: f32,
    center_z: f32,
    out_center_x: f32,
    inv_n: f32,
}

/// Manages all GPU resources for back-projection reconstruction.
/// Create once, then call `reconstruct_row` for each Y row.
pub struct GpuSession {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    row_offset_buffer: wgpu::Buffer,
    padded_entry: usize,
    out_nx: usize,
    out_nz: usize,
}

impl GpuSession {
    /// Try to initialise the GPU pipeline. Returns `None` if no suitable adapter
    /// is found or device creation fails.
    pub fn new(
        projections: &[f32],
        tilt_cos_sin: &[(f32, f32)],
        in_nx: usize,
        in_ny: usize,
        n_projs: usize,
        out_nx: usize,
        out_nz: usize,
    ) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        eprintln!("tilt: GPU adapter: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tilt-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        // --- Shader module ---
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backproject"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("backproject.wgsl"))),
        });

        // --- Bind group layout ---
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bp_layout"),
                entries: &[
                    bgl_storage_ro(0),
                    bgl_storage_ro(1),
                    bgl_storage_rw(2),
                    bgl_uniform(3),
                    bgl_uniform_dynamic(4, std::mem::size_of::<u32>() as u64),
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bp_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("backproject"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Buffers ---
        let proj_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("projections"),
            contents: bytemuck::cast_slice(projections),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tilt_flat: Vec<f32> = tilt_cos_sin
            .iter()
            .flat_map(|&(c, s)| [c, s])
            .collect();
        let tilt_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tilt_params"),
            contents: bytemuck::cast_slice(&tilt_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params = Params {
            nx: out_nx as u32,
            nz: out_nz as u32,
            n_projs: n_projs as u32,
            in_nx: in_nx as u32,
            in_ny: in_ny as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            center_x: in_nx as f32 / 2.0,
            center_z: out_nz as f32 / 2.0,
            out_center_x: out_nx as f32 / 2.0,
            inv_n: 1.0 / n_projs as f32,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Row offset buffer with dynamic offsets. Pre-fill all Y indices.
        // Dynamic uniform offsets must be aligned to minUniformBufferOffsetAlignment.
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let padded_entry = align.max(std::mem::size_of::<u32>());
        let mut row_data = vec![0u8; padded_entry * in_ny];
        for iy in 0..in_ny {
            let off = iy * padded_entry;
            row_data[off..off + 4].copy_from_slice(&(iy as u32).to_le_bytes());
        }
        let row_offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_offsets"),
            contents: &row_data,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_size = (out_nx * out_nz * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bp_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tilt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &row_offset_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as u64),
                    }),
                },
            ],
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group,
            output_buffer,
            staging_buffer,
            row_offset_buffer,
            padded_entry,
            out_nx,
            out_nz,
        })
    }

    /// Reconstruct one XZ slice for the given Y row index.
    pub fn reconstruct_row(&self, iy: usize) -> Vec<f32> {
        let output_size = (self.out_nx * self.out_nz * std::mem::size_of::<f32>()) as u64;

        // Clear output buffer
        let zeros = vec![0u8; output_size as usize];
        self.queue.write_buffer(&self.output_buffer, 0, &zeros);

        let dynamic_offset = (iy * self.padded_entry) as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bp_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
            let wg_x = (self.out_nx as u32 + 15) / 16;
            let wg_z = (self.out_nz as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_z, 1);
        }

        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Map staging buffer and read results
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        result
    }
}

// --- Helper functions for bind group layout entries ---

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

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

fn bgl_uniform_dynamic(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(min_size),
        },
        count: None,
    }
}
