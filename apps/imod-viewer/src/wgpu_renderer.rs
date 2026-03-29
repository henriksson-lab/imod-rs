//! GPU-accelerated 3D renderer for IMOD models using wgpu.
//!
//! Replaces the software rasterizer with a proper GPU pipeline featuring
//! Phong lighting, transparency (sorted back-to-front), per-object materials,
//! and depth-cue fog. Renders to an off-screen texture and reads back to a
//! Slint Image.

use bytemuck::{Pod, Zeroable};
use imod_mesh::IsosurfaceMesh;
use imod_model::{ImatData, ImodModel};
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};

// ---------------------------------------------------------------------------
// Uniform buffer layout  (must match model.wgsl)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model_rot: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad0: f32,
    light_dir: [f32; 3],
    _pad1: f32,
    ambient_strength: f32,
    diffuse_strength: f32,
    specular_strength: f32,
    shininess: f32,
    fog_near: f32,
    fog_far: f32,
    _pad2: f32,
    _pad3: f32,
    bg_color: [f32; 4],
}

// ---------------------------------------------------------------------------
// Per-vertex layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    alpha: f32,
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &[
            // position
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            // normal
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            // color
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
            // alpha
            wgpu::VertexAttribute {
                offset: 36,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

// ---------------------------------------------------------------------------
// A draw batch: a range of vertices with a material
// ---------------------------------------------------------------------------

struct DrawBatch {
    vertex_offset: u32,
    vertex_count: u32,
    mat: MaterialParams,
    transparent: bool,
    centroid_z: f32, // for sorting transparent batches
}

#[derive(Clone)]
struct MaterialParams {
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            ambient: 0.4,
            diffuse: 0.7,
            specular: 0.3,
            shininess: 32.0,
        }
    }
}

impl From<&ImatData> for MaterialParams {
    fn from(m: &ImatData) -> Self {
        Self {
            ambient: m.ambient as f32 / 255.0,
            diffuse: m.diffuse as f32 / 255.0,
            specular: m.specular as f32 / 255.0,
            shininess: (m.shininess as f32 / 255.0) * 128.0 + 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// WgpuRenderer
// ---------------------------------------------------------------------------

pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_opaque: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    width: u32,
    height: u32,
    // Camera state
    pub rot_x: f32,
    pub rot_y: f32,
    pub zoom: f32,
    center: [f32; 3],
    radius: f32,
}

impl WgpuRenderer {
    /// Create a new GPU renderer. Returns `None` if wgpu initialisation fails.
    pub fn new(width: u32, height: u32) -> Option<Self> {
        let w = width.max(1);
        let h = height.max(1);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("imod-viewer"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("model.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniforms_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = Vertex::layout();

        // Opaque pipeline: depth write + test, no blending
        let pipeline_opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opaque_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // two-sided
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Transparent pipeline: depth test but no depth write, alpha blending
        let pipeline_transparent =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transparent_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false, // no depth write for transparency
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let (depth_texture, depth_view) = Self::create_depth_texture(&device, w, h);
        let (output_texture, output_view) = Self::create_output_texture(&device, w, h);

        Some(Self {
            device,
            queue,
            pipeline_opaque,
            pipeline_transparent,
            bind_group_layout,
            depth_texture,
            depth_view,
            output_texture,
            output_view,
            width: w,
            height: h,
            rot_x: 20.0,
            rot_y: 30.0,
            zoom: 1.0,
            center: [0.0; 3],
            radius: 1.0,
        })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        let (dt, dv) = Self::create_depth_texture(&self.device, w, h);
        let (ot, ov) = Self::create_output_texture(&self.device, w, h);
        self.depth_texture = dt;
        self.depth_view = dv;
        self.output_texture = ot;
        self.output_view = ov;
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.rot_y += dx * 0.5;
        self.rot_x += dy * 0.5;
        self.rot_x = self.rot_x.clamp(-89.0, 89.0);
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.max(0.01);
    }

    // -----------------------------------------------------------------------
    // Render an IMOD model
    // -----------------------------------------------------------------------

    pub fn render_model(&mut self, model: &ImodModel) -> Image {
        self.compute_bounds_model(model);
        let (vertices, batches) = self.build_model_geometry(model);
        self.render_batches(&vertices, batches)
    }

    // -----------------------------------------------------------------------
    // Render an isosurface mesh
    // -----------------------------------------------------------------------

    pub fn render_isosurface(&mut self, mesh: &IsosurfaceMesh) -> Image {
        self.compute_bounds_isosurface(mesh);
        let (vertices, batches) = self.build_isosurface_geometry(mesh);
        self.render_batches(&vertices, batches)
    }

    // -----------------------------------------------------------------------
    // Internal: create textures
    // -----------------------------------------------------------------------

    fn create_depth_texture(
        device: &wgpu::Device,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    fn create_output_texture(
        device: &wgpu::Device,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    // -----------------------------------------------------------------------
    // Bounds computation
    // -----------------------------------------------------------------------

    fn compute_bounds_model(&mut self, model: &ImodModel) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let mut any = false;

        for obj in &model.objects {
            for cont in &obj.contours {
                for pt in &cont.points {
                    let p = [pt.x, pt.y, pt.z];
                    for i in 0..3 {
                        min[i] = min[i].min(p[i]);
                        max[i] = max[i].max(p[i]);
                    }
                    any = true;
                }
            }
            for mesh in &obj.meshes {
                for v in &mesh.vertices {
                    let p = [v.x, v.y, v.z];
                    for i in 0..3 {
                        min[i] = min[i].min(p[i]);
                        max[i] = max[i].max(p[i]);
                    }
                    any = true;
                }
            }
        }

        if !any {
            self.center = [0.0; 3];
            self.radius = 1.0;
            return;
        }

        for i in 0..3 {
            self.center[i] = (min[i] + max[i]) * 0.5;
        }
        let dx = max[0] - min[0];
        let dy = max[1] - min[1];
        let dz = max[2] - min[2];
        self.radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;
        if self.radius < 1e-6 {
            self.radius = 1.0;
        }
    }

    fn compute_bounds_isosurface(&mut self, mesh: &IsosurfaceMesh) {
        if mesh.vertices.is_empty() {
            self.center = [0.0; 3];
            self.radius = 1.0;
            return;
        }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &mesh.vertices {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
        for i in 0..3 {
            self.center[i] = (min[i] + max[i]) * 0.5;
        }
        let dx = max[0] - min[0];
        let dy = max[1] - min[1];
        let dz = max[2] - min[2];
        self.radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;
        if self.radius < 1e-6 {
            self.radius = 1.0;
        }
    }

    // -----------------------------------------------------------------------
    // Matrix math (minimal, no dependency)
    // -----------------------------------------------------------------------

    fn build_mvp(&self) -> ([[f32; 4]; 4], [[f32; 4]; 4], [f32; 3]) {
        let ax = self.rot_x.to_radians();
        let ay = self.rot_y.to_radians();
        let (sx, cx) = ax.sin_cos();
        let (sy, cy) = ay.sin_cos();

        // Model rotation: Y then X
        let rot = [
            [cy, 0.0, sy, 0.0],
            [sx * sy, cx, -sx * cy, 0.0],
            [-cx * sy, sx, cx * cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Camera distance
        let dist = self.radius * 3.0 / self.zoom;
        let cam_pos = [0.0f32, 0.0, dist];

        // Orthographic projection that fits the model
        let half_extent = self.radius / self.zoom;
        let aspect = self.width as f32 / self.height as f32;
        let (l, r, b, t) = if aspect >= 1.0 {
            (
                -half_extent * aspect,
                half_extent * aspect,
                -half_extent,
                half_extent,
            )
        } else {
            (
                -half_extent,
                half_extent,
                -half_extent / aspect,
                half_extent / aspect,
            )
        };
        let near = 0.01;
        let far = dist * 3.0;

        // Orthographic projection matrix
        let proj = [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, -1.0 / (far - near), 0.0],
            [
                -(r + l) / (r - l),
                -(t + b) / (t - b),
                -near / (far - near),
                1.0,
            ],
        ];

        // View matrix: translate center to origin, then apply rotation, then translate by -cam_pos
        // Combined: MVP = proj * view_translate * rot * model_translate
        // model_translate: move center to origin
        // view_translate: move camera back

        // Build MVP step by step:
        // 1) translate center to origin
        let tc = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-self.center[0], -self.center[1], -self.center[2], 1.0],
        ];

        // 2) rotate
        // 3) translate camera (view)
        let tv = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -dist, 1.0],
        ];

        // MVP = proj * tv * rot * tc
        let rot_tc = mat4_mul(&rot, &tc);
        let tv_rot_tc = mat4_mul(&tv, &rot_tc);
        let mvp = mat4_mul(&proj, &tv_rot_tc);

        (mvp, rot, cam_pos)
    }

    // -----------------------------------------------------------------------
    // Geometry building: IMOD model
    // -----------------------------------------------------------------------

    fn build_model_geometry(&self, model: &ImodModel) -> (Vec<Vertex>, Vec<DrawBatch>) {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut batches: Vec<DrawBatch> = Vec::new();

        for obj in &model.objects {
            let color = [obj.red, obj.green, obj.blue];
            let alpha = 1.0 - (obj.trans as f32 / 255.0);
            let mat = obj
                .imat
                .as_ref()
                .map(MaterialParams::from)
                .unwrap_or_default();
            let transparent = alpha < 0.999;

            // --- Meshes ---
            for mesh in &obj.meshes {
                let start = vertices.len() as u32;
                self.tessellate_imod_mesh(
                    &mesh.vertices,
                    &mesh.indices,
                    color,
                    alpha,
                    &mut vertices,
                );
                let count = vertices.len() as u32 - start;
                if count > 0 {
                    // Compute centroid for sorting
                    let centroid_z = self.batch_centroid_z(&vertices, start, count);
                    batches.push(DrawBatch {
                        vertex_offset: start,
                        vertex_count: count,
                        mat: mat.clone(),
                        transparent,
                        centroid_z,
                    });
                }
            }

            // --- Contours as 3D tubes (rendered as thick lines via small quads) ---
            for cont in &obj.contours {
                let pts: Vec<[f32; 3]> = cont.points.iter().map(|p| [p.x, p.y, p.z]).collect();
                if pts.len() < 2 {
                    // Single-point contours: render as small sphere-like billboards
                    for &p in &pts {
                        let start = vertices.len() as u32;
                        self.emit_point_sphere(p, color, alpha, &mut vertices, obj.pdrawsize);
                        let count = vertices.len() as u32 - start;
                        if count > 0 {
                            batches.push(DrawBatch {
                                vertex_offset: start,
                                vertex_count: count,
                                mat: mat.clone(),
                                transparent,
                                centroid_z: self.batch_centroid_z(&vertices, start, count),
                            });
                        }
                    }
                    continue;
                }
                let start = vertices.len() as u32;
                let line_w = if obj.linewidth > 0 {
                    obj.linewidth as f32
                } else {
                    1.0
                };
                self.emit_line_strip(&pts, line_w, color, alpha, &mut vertices);
                // Also emit point markers
                let pt_radius = if obj.pdrawsize > 0 {
                    obj.pdrawsize as f32
                } else {
                    1.0
                };
                for &p in &pts {
                    self.emit_point_sphere(p, color, alpha, &mut vertices, pt_radius as i32);
                }
                let count = vertices.len() as u32 - start;
                if count > 0 {
                    batches.push(DrawBatch {
                        vertex_offset: start,
                        vertex_count: count,
                        mat: mat.clone(),
                        transparent,
                        centroid_z: self.batch_centroid_z(&vertices, start, count),
                    });
                }
            }
        }

        (vertices, batches)
    }

    /// Tessellate an IMOD mesh (with sentinel-based index list) into triangles.
    fn tessellate_imod_mesh(
        &self,
        verts: &[imod_core::Point3f],
        indices: &[i32],
        color: [f32; 3],
        alpha: f32,
        out: &mut Vec<Vertex>,
    ) {
        if verts.is_empty() || indices.is_empty() {
            return;
        }
        let nv = verts.len() as i32;

        let mut i = 0;
        while i < indices.len() {
            let cmd = indices[i];
            if cmd == -1 {
                break;
            }
            if cmd == -23 || cmd == -24 {
                // BGNPOLY / BGNBIGPOLY
                i += 1;
                let mut poly: Vec<usize> = Vec::new();
                while i < indices.len() {
                    let idx = indices[i];
                    if idx < 0 {
                        break;
                    }
                    if idx < nv {
                        poly.push(idx as usize);
                    }
                    i += 1;
                }
                self.emit_polygon_fan(verts, &poly, color, alpha, out);
                if i < indices.len() && (indices[i] == -22 || indices[i] == -21) {
                    i += 1;
                }
            } else if cmd == -25 {
                // POLYNORM: pairs of (normal_idx, vertex_idx)
                i += 1;
                let mut poly: Vec<usize> = Vec::new();
                let mut normals: Vec<usize> = Vec::new();
                while i + 1 < indices.len() {
                    let ni = indices[i];
                    let vi = indices[i + 1];
                    if ni < 0 || vi < 0 {
                        break;
                    }
                    if vi < nv {
                        poly.push(vi as usize);
                        normals.push(if ni < nv { ni as usize } else { vi as usize });
                    }
                    i += 2;
                }
                // Use normals from the vertex list as direction vectors
                self.emit_polygon_fan_with_normals(verts, &poly, &normals, color, alpha, out);
                if i < indices.len() && (indices[i] == -22 || indices[i] == -21) {
                    i += 1;
                }
            } else if cmd == -20 {
                i += 1;
            } else if cmd >= 0 {
                let mut tri_indices: Vec<i32> = Vec::new();
                while i < indices.len() && indices[i] >= 0 {
                    tri_indices.push(indices[i]);
                    i += 1;
                }
                for chunk in tri_indices.chunks(3) {
                    if chunk.len() == 3 {
                        let (i0, i1, i2) = (chunk[0], chunk[1], chunk[2]);
                        if i0 < nv && i1 < nv && i2 < nv {
                            let v0 = &verts[i0 as usize];
                            let v1 = &verts[i1 as usize];
                            let v2 = &verts[i2 as usize];
                            let p0 = [v0.x, v0.y, v0.z];
                            let p1 = [v1.x, v1.y, v1.z];
                            let p2 = [v2.x, v2.y, v2.z];
                            let n = face_normal(p0, p1, p2);
                            out.push(Vertex {
                                position: p0,
                                normal: n,
                                color,
                                alpha,
                            });
                            out.push(Vertex {
                                position: p1,
                                normal: n,
                                color,
                                alpha,
                            });
                            out.push(Vertex {
                                position: p2,
                                normal: n,
                                color,
                                alpha,
                            });
                        }
                    }
                }
                if i < indices.len() && indices[i] == -1 {
                    break;
                }
            } else {
                i += 1;
            }
        }
    }

    fn emit_polygon_fan(
        &self,
        verts: &[imod_core::Point3f],
        poly: &[usize],
        color: [f32; 3],
        alpha: f32,
        out: &mut Vec<Vertex>,
    ) {
        if poly.len() < 3 {
            return;
        }
        let v0 = &verts[poly[0]];
        let p0 = [v0.x, v0.y, v0.z];
        for j in 1..poly.len() - 1 {
            let v1 = &verts[poly[j]];
            let v2 = &verts[poly[j + 1]];
            let p1 = [v1.x, v1.y, v1.z];
            let p2 = [v2.x, v2.y, v2.z];
            let n = face_normal(p0, p1, p2);
            out.push(Vertex {
                position: p0,
                normal: n,
                color,
                alpha,
            });
            out.push(Vertex {
                position: p1,
                normal: n,
                color,
                alpha,
            });
            out.push(Vertex {
                position: p2,
                normal: n,
                color,
                alpha,
            });
        }
    }

    fn emit_polygon_fan_with_normals(
        &self,
        verts: &[imod_core::Point3f],
        poly: &[usize],
        normals: &[usize],
        color: [f32; 3],
        alpha: f32,
        out: &mut Vec<Vertex>,
    ) {
        if poly.len() < 3 || normals.len() < poly.len() {
            // Fallback to flat shading
            self.emit_polygon_fan(verts, poly, color, alpha, out);
            return;
        }
        let v0 = &verts[poly[0]];
        let p0 = [v0.x, v0.y, v0.z];
        let nv0 = &verts[normals[0]];
        let n0 = normalize3([nv0.x, nv0.y, nv0.z]);

        for j in 1..poly.len() - 1 {
            let v1 = &verts[poly[j]];
            let v2 = &verts[poly[j + 1]];
            let p1 = [v1.x, v1.y, v1.z];
            let p2 = [v2.x, v2.y, v2.z];
            let nv1 = &verts[normals[j]];
            let nv2 = &verts[normals[j + 1]];
            let n1 = normalize3([nv1.x, nv1.y, nv1.z]);
            let n2 = normalize3([nv2.x, nv2.y, nv2.z]);

            out.push(Vertex {
                position: p0,
                normal: n0,
                color,
                alpha,
            });
            out.push(Vertex {
                position: p1,
                normal: n1,
                color,
                alpha,
            });
            out.push(Vertex {
                position: p2,
                normal: n2,
                color,
                alpha,
            });
        }
    }

    /// Emit a line strip as camera-facing quads (billboard tubes).
    fn emit_line_strip(
        &self,
        pts: &[[f32; 3]],
        width: f32,
        color: [f32; 3],
        alpha: f32,
        out: &mut Vec<Vertex>,
    ) {
        let half_w = width * 0.5;
        for w in pts.windows(2) {
            let a = w[0];
            let b = w[1];
            let dir = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            if len < 1e-8 {
                continue;
            }
            // Pick a perpendicular direction: cross with up or right
            let up = if dir[1].abs() < 0.9 * len {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let perp = normalize3(cross3(dir, up));
            let normal = normalize3(cross3(perp, dir));

            let offset = [perp[0] * half_w, perp[1] * half_w, perp[2] * half_w];

            let p0 = [a[0] - offset[0], a[1] - offset[1], a[2] - offset[2]];
            let p1 = [a[0] + offset[0], a[1] + offset[1], a[2] + offset[2]];
            let p2 = [b[0] + offset[0], b[1] + offset[1], b[2] + offset[2]];
            let p3 = [b[0] - offset[0], b[1] - offset[1], b[2] - offset[2]];

            // Two triangles
            for &pos in &[p0, p1, p2, p0, p2, p3] {
                out.push(Vertex {
                    position: pos,
                    normal,
                    color,
                    alpha,
                });
            }
        }
    }

    /// Emit a small octahedron to represent a point marker.
    fn emit_point_sphere(
        &self,
        center: [f32; 3],
        color: [f32; 3],
        alpha: f32,
        out: &mut Vec<Vertex>,
        size: i32,
    ) {
        let r = if size > 0 { size as f32 } else { 1.0 };
        // 6 vertices of an octahedron
        let top = [center[0], center[1] + r, center[2]];
        let bot = [center[0], center[1] - r, center[2]];
        let lft = [center[0] - r, center[1], center[2]];
        let rgt = [center[0] + r, center[1], center[2]];
        let fwd = [center[0], center[1], center[2] + r];
        let bck = [center[0], center[1], center[2] - r];

        let faces = [
            (top, fwd, rgt),
            (top, rgt, bck),
            (top, bck, lft),
            (top, lft, fwd),
            (bot, rgt, fwd),
            (bot, bck, rgt),
            (bot, lft, bck),
            (bot, fwd, lft),
        ];

        for (a, b, c) in &faces {
            let n = face_normal(*a, *b, *c);
            out.push(Vertex {
                position: *a,
                normal: n,
                color,
                alpha,
            });
            out.push(Vertex {
                position: *b,
                normal: n,
                color,
                alpha,
            });
            out.push(Vertex {
                position: *c,
                normal: n,
                color,
                alpha,
            });
        }
    }

    fn batch_centroid_z(&self, vertices: &[Vertex], start: u32, count: u32) -> f32 {
        if count == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in start..(start + count) {
            sum += vertices[i as usize].position[2];
        }
        sum / count as f32
    }

    // -----------------------------------------------------------------------
    // Geometry building: Isosurface
    // -----------------------------------------------------------------------

    fn build_isosurface_geometry(&self, mesh: &IsosurfaceMesh) -> (Vec<Vertex>, Vec<DrawBatch>) {
        let color = [100.0 / 255.0, 180.0 / 255.0, 220.0 / 255.0];
        let alpha = 1.0f32;
        let mut vertices = Vec::with_capacity(mesh.indices.len());

        for tri in mesh.indices.chunks(3) {
            if tri.len() < 3 {
                break;
            }
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len()
            {
                continue;
            }
            let p0 = mesh.vertices[i0];
            let p1 = mesh.vertices[i1];
            let p2 = mesh.vertices[i2];
            let n0 = mesh.normals[i0];
            let n1 = mesh.normals[i1];
            let n2 = mesh.normals[i2];

            vertices.push(Vertex {
                position: p0,
                normal: n0,
                color,
                alpha,
            });
            vertices.push(Vertex {
                position: p1,
                normal: n1,
                color,
                alpha,
            });
            vertices.push(Vertex {
                position: p2,
                normal: n2,
                color,
                alpha,
            });
        }

        let count = vertices.len() as u32;
        let batches = if count > 0 {
            vec![DrawBatch {
                vertex_offset: 0,
                vertex_count: count,
                mat: MaterialParams::default(),
                transparent: false,
                centroid_z: 0.0,
            }]
        } else {
            vec![]
        };

        (vertices, batches)
    }

    // -----------------------------------------------------------------------
    // Core render: upload geometry, dispatch draw calls, read back pixels
    // -----------------------------------------------------------------------

    fn render_batches(&self, vertices: &[Vertex], mut batches: Vec<DrawBatch>) -> Image {
        if vertices.is_empty() || batches.is_empty() {
            return self.blank_image();
        }

        // Upload vertex buffer
        let vb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertices"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        // Build MVP
        let (mvp, model_rot, cam_pos) = self.build_mvp();

        // Separate opaque and transparent batches
        let (opaque, mut transparent): (Vec<_>, Vec<_>) =
            batches.drain(..).partition(|b| !b.transparent);

        // Sort transparent batches back-to-front (largest centroid_z first in camera space)
        // We need to transform centroid_z by the rotation to get camera-space Z
        transparent.sort_by(|a, b| {
            b.centroid_z
                .partial_cmp(&a.centroid_z)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render"),
            });

        // Render pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.133,
                            g: 0.133,
                            b: 0.133,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_vertex_buffer(0, vb.slice(..));

            // Draw opaque batches
            pass.set_pipeline(&self.pipeline_opaque);
            for batch in &opaque {
                let bg = self.create_batch_bind_group(&batch.mat, &mvp, &model_rot, &cam_pos);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(
                    batch.vertex_offset..(batch.vertex_offset + batch.vertex_count),
                    0..1,
                );
            }

            // Draw transparent batches (sorted back-to-front)
            pass.set_pipeline(&self.pipeline_transparent);
            for batch in &transparent {
                let bg = self.create_batch_bind_group(&batch.mat, &mvp, &model_rot, &cam_pos);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(
                    batch.vertex_offset..(batch.vertex_offset + batch.vertex_count),
                    0..1,
                );
            }
        }

        // Copy output texture to staging buffer
        let bytes_per_row = align_to_256(self.width * 4);
        let staging_size = (bytes_per_row * self.height) as u64;
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back
        let buf_slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            return self.blank_image();
        }

        let data = buf_slice.get_mapped_range();
        let mut pixel_buf =
            SharedPixelBuffer::<Rgba8Pixel>::new(self.width, self.height);
        let out_pixels = pixel_buf.make_mut_bytes();
        let row_bytes = self.width as usize * 4;
        for y in 0..self.height as usize {
            let src_off = y * bytes_per_row as usize;
            let dst_off = y * row_bytes;
            out_pixels[dst_off..dst_off + row_bytes]
                .copy_from_slice(&data[src_off..src_off + row_bytes]);
        }
        drop(data);
        staging_buf.unmap();

        Image::from_rgba8(pixel_buf)
    }

    fn create_batch_bind_group(
        &self,
        mat: &MaterialParams,
        mvp: &[[f32; 4]; 4],
        model_rot: &[[f32; 4]; 4],
        cam_pos: &[f32; 3],
    ) -> wgpu::BindGroup {
        let dist = self.radius * 3.0 / self.zoom;
        let uniforms = Uniforms {
            mvp: *mvp,
            model_rot: *model_rot,
            camera_pos: *cam_pos,
            _pad0: 0.0,
            light_dir: [0.3, 0.6, 1.0], // slightly above and to the right
            _pad1: 0.0,
            ambient_strength: mat.ambient,
            diffuse_strength: mat.diffuse,
            specular_strength: mat.specular,
            shininess: mat.shininess,
            fog_near: dist * 0.3,
            fog_far: dist * 2.5,
            _pad2: 0.0,
            _pad3: 0.0,
            bg_color: [0.133, 0.133, 0.133, 1.0],
        };

        let ub = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_bg"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            }],
        })
    }

    fn blank_image(&self) -> Image {
        let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(self.width, self.height);
        let pixels = buf.make_mut_bytes();
        for i in 0..(self.width * self.height) as usize {
            let off = i * 4;
            pixels[off] = 0x22;
            pixels[off + 1] = 0x22;
            pixels[off + 2] = 0x22;
            pixels[off + 3] = 0xFF;
        }
        Image::from_rgba8(buf)
    }
}

// ---------------------------------------------------------------------------
// Helper math functions
// ---------------------------------------------------------------------------

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalize3(cross3(e1, e2))
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

/// Align a value up to the next multiple of 256 (wgpu row alignment requirement).
fn align_to_256(v: u32) -> u32 {
    (v + 255) & !255
}

// Bring in the buffer init utility
use wgpu::util::DeviceExt;
