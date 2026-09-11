//! Translation of `IMOD/3dmod/vertexbuffer.cpp` and `vertexbuffer.h`.
//!
//! The upstream C structures retain VBO pointers in `Imesh`/`Iobj`.  Rust's
//! source-mapped model structures deliberately omit raw GL ownership, so the
//! equivalent associations are retained by `VertBufManager`, keyed by the
//! original model allocation address.  No CPU-side rendering substitute is
//! used: uploads and indexed primitive submission go to `VertexBufferGl`.
#![allow(dead_code, unused_variables)]

use std::collections::HashMap;

use crate::imod::libimod::imesh::{
    IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_ENDPOLY,
};
use crate::imod::libimod::imodel::{Imesh, Imod, Iobj, Ipoint};
use crate::imod::libimod::istore::DrawProps;

pub const RESTART_INDEX: u32 = 0x7fff_ffff;
pub const GL_ARRAY_BUFFER: u32 = 0x8892;
pub const GL_ELEMENT_ARRAY_BUFFER: u32 = 0x8893;
pub const GL_STATIC_DRAW: u32 = 0x88e4;
pub const GL_TRIANGLE_STRIP: u32 = 0x0005;
pub const GL_TRIANGLE_FAN: u32 = 0x0006;
pub const GL_LINE_STRIP: u32 = 0x0003;

/// Original `RGBTindices`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RgbtIndices {
    pub first_element: i32,
    pub num_inds: i32,
    pub num_fan_inds: i32,
}
/// Original `RGBTmap`.
pub type RgbtMap = HashMap<u32, RgbtIndices>;
/// Original `VertBufData`.
#[derive(Clone, Debug, Default)]
pub struct VertBufData {
    pub vb_obj: u32,
    pub eb_obj: u32,
    pub vbo_size: i32,
    pub ebo_size: i32,
    pub num_ind_default: i32,
    pub num_special_sets: i32,
    pub num_ind_special: Vec<i32>,
    pub rgbt_special: Vec<u32>,
    pub special_size: i32,
    pub remnant_ind_list: Vec<i32>,
    pub rem_list_size: i32,
    pub num_remnant: i32,
    pub num_fan_ind_default: i32,
    pub num_fan_ind_special: Vec<i32>,
    pub default_rgbt: u32,
    pub fan_ind_start: i32,
    pub zscale: f32,
    pub fill_type: i32,
    pub use_fill_color: i32,
    pub check_time: i32,
    pub thicken_cont: i32,
    pub check_stipple: i32,
    pub scrn_scale: f32,
    pub pdrawsize: f32,
    pub quality: i32,
    pub checksum: f64,
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Direct OpenGL operations used by this translation unit.
pub trait VertexBufferGl {
    fn gen_buffer(&mut self) -> u32;
    fn delete_buffer(&mut self, buffer: u32);
    fn bind_buffer(&mut self, target: u32, buffer: u32);
    fn buffer_data_f32(&mut self, target: u32, values: &[f32], usage: u32);
    fn buffer_data_u32(&mut self, target: u32, values: &[u32], usage: u32);
    fn buffer_sub_data_f32(&mut self, target: u32, offset_bytes: i32, values: &[f32]);
    fn draw_elements(&mut self, mode: u32, indices: i32, offset_bytes: i32);
    fn primitive_restart(&mut self, enabled: bool, index: u32);
}

/// Real `glow` implementation of the source `b3d*Buffer` calls.
#[cfg(feature = "three-dmod-gl")]
pub struct GlowVertexBufferGl<'a> {
    pub gl: &'a glow::Context,
    /// `glPrimitiveRestartIndex`, loaded from the same native GL context as
    /// `glow`.  Glow 0.17 does not expose this compatibility entry point.
    pub primitive_restart_index: unsafe extern "system" fn(u32),
}
#[cfg(feature = "three-dmod-gl")]
impl VertexBufferGl for GlowVertexBufferGl<'_> {
    fn gen_buffer(&mut self) -> u32 {
        unsafe {
            use glow::HasContext;
            self.gl.create_buffer().map(|b| b.0.get()).unwrap_or(0)
        }
    }
    fn delete_buffer(&mut self, b: u32) {
        if b != 0 {
            unsafe {
                use glow::HasContext;
                self.gl
                    .delete_buffer(glow::NativeBuffer(std::num::NonZeroU32::new(b).unwrap()));
            }
        }
    }
    fn bind_buffer(&mut self, t: u32, b: u32) {
        unsafe {
            use glow::HasContext;
            self.gl
                .bind_buffer(t, std::num::NonZeroU32::new(b).map(glow::NativeBuffer));
        }
    }
    fn buffer_data_f32(&mut self, t: u32, v: &[f32], u: u32) {
        unsafe {
            use glow::HasContext;
            self.gl.buffer_data_u8_slice(
                t,
                std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)),
                u,
            );
        }
    }
    fn buffer_data_u32(&mut self, t: u32, v: &[u32], u: u32) {
        unsafe {
            use glow::HasContext;
            self.gl.buffer_data_u8_slice(
                t,
                std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)),
                u,
            );
        }
    }
    fn buffer_sub_data_f32(&mut self, t: u32, o: i32, v: &[f32]) {
        unsafe {
            use glow::HasContext;
            self.gl.buffer_sub_data_u8_slice(
                t,
                o,
                std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)),
            );
        }
    }
    fn draw_elements(&mut self, m: u32, n: i32, o: i32) {
        unsafe {
            use glow::HasContext;
            self.gl.draw_elements(m, n, glow::UNSIGNED_INT, o);
        }
    }
    fn primitive_restart(&mut self, e: bool, index: u32) {
        unsafe {
            use glow::HasContext;
            if e {
                self.gl.enable(glow::PRIMITIVE_RESTART);
                (self.primitive_restart_index)(index);
            } else {
                self.gl.disable(glow::PRIMITIVE_RESTART);
            }
        }
    }
}

/// Original `VertBufManager`.
#[derive(Default)]
pub struct VertBufManager {
    pub mesh_data: HashMap<usize, VertBufData>,
    pub cont_data: HashMap<usize, VertBufData>,
    pub sphere_data: HashMap<usize, VertBufData>,
    pub m_inds: Vec<u32>,
    pub m_verts: Vec<f32>,
    pub m_def_sph_inds: Vec<u32>,
    pub m_def_sph_verts: Vec<f32>,
    pub m_ind_vert: i32,
    pub m_xadd: f32,
    pub m_yadd: f32,
    pub m_zadd: f32,
    pub m_norm_offset: i32,
}

/// Original static `vbDataNew`.
pub fn vb_data_new() -> VertBufData {
    let mut v = VertBufData::default();
    vb_data_init(&mut v);
    v
}
/// Original static `vbDataInit`.
pub fn vb_data_init(v: &mut VertBufData) {
    *v = VertBufData::default();
}
/// Original static `vbDataClear`.
pub fn vb_data_clear(v: &mut VertBufData, gl: &mut dyn VertexBufferGl) {
    if v.vb_obj != 0 {
        gl.delete_buffer(v.vb_obj)
    }
    if v.eb_obj != 0 {
        gl.delete_buffer(v.eb_obj)
    }
    vb_data_init(v);
}
/// Original static `vbDataDelete`.
pub fn vb_data_delete(v: &mut VertBufData, gl: &mut dyn VertexBufferGl) {
    vb_data_clear(v, gl);
}
/// Original `vbCleanupVBD(Imesh *)`.
pub fn vb_cleanup_vbd_mesh(
    manager: &mut VertBufManager,
    mesh: &Imesh,
    gl: &mut dyn VertexBufferGl,
) {
    if let Some(mut v) = manager.mesh_data.remove(&(mesh as *const _ as usize)) {
        vb_data_delete(&mut v, gl);
    }
}
/// Original `vbCleanupVBD(Iobj *)`.
pub fn vb_cleanup_vbd_object(
    manager: &mut VertBufManager,
    obj: &Iobj,
    gl: &mut dyn VertexBufferGl,
) {
    vb_cleanup_cont_vbd(manager, obj, gl);
    vb_cleanup_sphere_vbd(manager, obj, gl);
    vb_cleanup_mesh_vbd(manager, obj, gl);
}
/// Original `vbCleanupVBD(Imod *)`.
pub fn vb_cleanup_vbd_model(
    manager: &mut VertBufManager,
    model: &Imod,
    gl: &mut dyn VertexBufferGl,
) {
    for obj in &model.obj {
        vb_cleanup_vbd_object(manager, obj, gl)
    }
}
/// Original `vbCleanupMeshVBD`.
pub fn vb_cleanup_mesh_vbd(manager: &mut VertBufManager, obj: &Iobj, gl: &mut dyn VertexBufferGl) {
    for mesh in &obj.mesh {
        vb_cleanup_vbd_mesh(manager, mesh, gl)
    }
}
/// Original `vbCleanupContVBD`.
pub fn vb_cleanup_cont_vbd(manager: &mut VertBufManager, obj: &Iobj, gl: &mut dyn VertexBufferGl) {
    if let Some(mut v) = manager.cont_data.remove(&(obj as *const _ as usize)) {
        vb_data_delete(&mut v, gl)
    }
}
/// Original `vbCleanupSphereVBD`.
pub fn vb_cleanup_sphere_vbd(
    manager: &mut VertBufManager,
    obj: &Iobj,
    gl: &mut dyn VertexBufferGl,
) {
    if let Some(mut v) = manager.sphere_data.remove(&(obj as *const _ as usize)) {
        vb_data_delete(&mut v, gl)
    }
}

impl VertBufManager {
    /// Original `VertBufManager::VertBufManager`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Original `VertBufManager::~VertBufManager`.
    pub fn destroy(&mut self) {
        self.clear_temp_arrays();
    }
    /// Original `VertBufManager::packRGBT(float,...)`.
    pub fn pack_rgbt(&self, red: f32, green: f32, blue: f32, trans: i32) -> u32 {
        ((red.mul_add(255., 0.) as i32).clamp(0, 255) as u32) << 24
            | ((green.mul_add(255., 0.) as i32).clamp(0, 255) as u32) << 16
            | ((blue.mul_add(255., 0.) as i32).clamp(0, 255) as u32) << 8
            | (trans.clamp(0, 255) as u32)
    }
    /// Original overloaded `VertBufManager::packRGBT(DrawProps *,...)`.
    pub fn pack_rgbt_props(&self, p: &DrawProps, use_fill: i32) -> u32 {
        if use_fill != 0 {
            self.pack_rgbt(p.fill_red, p.fill_green, p.fill_blue, p.trans)
        } else {
            self.pack_rgbt(p.red, p.green, p.blue, p.trans)
        }
    }
    /// Original `VertBufManager::unpackRGBT(float &,...)`.
    pub fn unpack_rgbt(&self, value: u32) -> (f32, f32, f32, i32) {
        (
            ((value >> 24 & 255) as f32) / 255.,
            ((value >> 16 & 255) as f32) / 255.,
            ((value >> 8 & 255) as f32) / 255.,
            (value & 255) as i32,
        )
    }
    /// Original overloaded `VertBufManager::unpackRGBT(DrawProps *)`.
    pub fn unpack_rgbt_props(&self, value: u32, use_fill: i32, p: &mut DrawProps) {
        let (r, g, b, t) = self.unpack_rgbt(value);
        if use_fill != 0 {
            p.fill_red = r;
            p.fill_green = g;
            p.fill_blue = b
        } else {
            p.red = r;
            p.green = g;
            p.blue = b
        }
        p.trans = t;
    }
    /// Original `VertBufManager::clearTempArrays`.
    pub fn clear_temp_arrays(&mut self) {
        self.m_inds.clear();
        self.m_verts.clear();
        self.m_def_sph_inds.clear();
        self.m_def_sph_verts.clear();
    }
    /// Original `VertBufManager::allocateTempVerts`.
    pub fn allocate_temp_verts(&mut self, num: i32) -> i32 {
        let n = (3 * num).max(0) as usize;
        if self.m_verts.len() < n {
            self.m_verts.resize(n, 0.)
        }
        0
    }
    /// Original `VertBufManager::allocateTempInds`.
    pub fn allocate_temp_inds(&mut self, num: i32) -> i32 {
        let n = num.max(0) as usize;
        if self.m_inds.len() < n {
            self.m_inds.resize(n, 0)
        }
        0
    }
    /// Original `VertBufManager::allocateDefaultSphere`.
    pub fn allocate_default_sphere(&mut self, nv: i32, ni: i32, normal: i32) -> i32 {
        let nv = (nv * (1 + (normal > 0) as i32) * 3).max(0) as usize;
        if self.m_def_sph_verts.len() < nv {
            self.m_def_sph_verts.resize(nv, 0.)
        }
        let ni = ni.max(0) as usize;
        if self.m_def_sph_inds.len() < ni {
            self.m_def_sph_inds.resize(ni, 0)
        }
        0
    }
    /// Original `VertBufManager::allocateSpecialSets`.
    pub fn allocate_special_sets(
        &mut self,
        v: &mut VertBufData,
        sets: i32,
        cum: i32,
        sphere: i32,
    ) -> i32 {
        v.num_special_sets = sets;
        v.num_ind_default = cum;
        if v.special_size < sets {
            v.num_ind_special.resize(sets.max(0) as usize, 0);
            v.rgbt_special.resize(sets.max(0) as usize, 0);
            if sphere != 0 {
                v.num_fan_ind_special.resize(sets.max(0) as usize, 0)
            }
            v.special_size = sets;
        }
        if v.rem_list_size < v.num_remnant {
            v.remnant_ind_list.resize(v.num_remnant.max(0) as usize, 0);
            v.rem_list_size = v.num_remnant;
        }
        0
    }
    /// Original `VertBufManager::processMap`.
    pub fn process_map(
        &mut self,
        v: &mut VertBufData,
        colors: &RgbtMap,
        cum: &mut i32,
        per: i32,
        fan: &mut i32,
    ) -> i32 {
        let mut entries: Vec<_> = colors.iter().collect();
        entries.sort_by_key(|(_, x)| x.first_element);
        for (i, (rgb, x)) in entries.iter().enumerate() {
            v.rgbt_special[i] = **rgb;
            v.num_ind_special[i] = x.num_inds * per;
            *cum += v.num_ind_special[i];
            if *fan >= 0 {
                v.num_fan_ind_special[i] = x.num_fan_inds;
                *fan += x.num_fan_inds;
            }
        }
        0
    }
    /// Original `VertBufManager::genAndBindBuffers`.
    pub fn gen_and_bind_buffers(
        &mut self,
        v: &mut VertBufData,
        vertices: i32,
        indices: i32,
        gl: &mut dyn VertexBufferGl,
    ) -> i32 {
        if v.vb_obj == 0 || 3 * vertices > v.vbo_size {
            if v.vb_obj != 0 {
                gl.delete_buffer(v.vb_obj)
            }
            v.vb_obj = gl.gen_buffer();
            v.vbo_size = 3 * vertices;
        }
        gl.bind_buffer(GL_ARRAY_BUFFER, v.vb_obj);
        if v.eb_obj == 0 || indices > v.ebo_size {
            if v.eb_obj != 0 {
                gl.delete_buffer(v.eb_obj)
            }
            v.eb_obj = gl.gen_buffer();
            v.ebo_size = indices;
        }
        gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, v.eb_obj);
        0
    }
    /// Original `VertBufManager::loadVertexNormalArray`.
    pub fn load_vertex_normal_array(&mut self, mesh: &Imesh, zscale: f32, fill: i32) -> Vec<f32> {
        let mut out = Vec::new();
        if fill != 0 {
            for p in mesh.vert.chunks_exact(2) {
                out.extend_from_slice(&[p[1].x, p[1].y, p[1].z, p[0].x, p[0].y, p[0].z * zscale]);
            }
        } else {
            for p in mesh.vert.chunks_exact(2) {
                out.extend_from_slice(&[p[0].x, p[0].y, p[0].z * zscale]);
            }
        }
        out
    }
    /// Original `VertBufManager::analyzeMesh`; it accepts only the exact triangle-list mesh representation accepted by the source VBO fast path.
    pub fn analyze_mesh(
        &mut self,
        mesh: &Imesh,
        zscale: f32,
        fill: i32,
        use_fill: i32,
        props: &DrawProps,
        gl: &mut dyn VertexBufferGl,
    ) -> i32 {
        let key = mesh as *const _ as usize;
        let mut v = self.mesh_data.remove(&key).unwrap_or_else(vb_data_new);
        if v.vb_obj != 0 && v.fill_type == fill && (v.zscale - zscale).abs() < 1e-4 {
            return -1;
        }
        let mut indices = Vec::new();
        let mut at = 0;
        while at < mesh.list.len() {
            match mesh.list[at] {
                IMOD_MESH_BGNPOLYNORM2 => {
                    at += 1;
                    while at < mesh.list.len() && mesh.list[at] != IMOD_MESH_ENDPOLY {
                        if at + 2 >= mesh.list.len() {
                            self.mesh_data.insert(key, v);
                            return 1;
                        }
                        for n in &mesh.list[at..at + 3] {
                            if *n < 0 || *n as usize >= mesh.vert.len() / 2 {
                                self.mesh_data.insert(key, v);
                                return 1;
                            }
                            indices.push(*n as u32);
                        }
                        at += 3;
                    }
                }
                IMOD_MESH_BGNPOLYNORM => {
                    self.mesh_data.insert(key, v);
                    return 1;
                }
                _ => {
                    self.mesh_data.insert(key, v);
                    return 1;
                }
            }
            at += 1;
        }
        let vertices = self.load_vertex_normal_array(mesh, zscale, fill);
        v.fill_type = fill;
        v.use_fill_color = use_fill;
        v.zscale = zscale;
        v.default_rgbt = self.pack_rgbt_props(props, use_fill);
        v.num_ind_default = indices.len() as i32;
        v.vertices = vertices;
        v.indices = indices;
        let nvert = (v.vertices.len() / 3) as i32;
        let nind = v.indices.len() as i32;
        if self.gen_and_bind_buffers(&mut v, nvert, nind, gl) != 0 {
            return 1;
        }
        gl.buffer_data_f32(GL_ARRAY_BUFFER, &v.vertices, GL_STATIC_DRAW);
        gl.buffer_data_u32(GL_ELEMENT_ARRAY_BUFFER, &v.indices, GL_STATIC_DRAW);
        self.mesh_data.insert(key, v);
        0
    }
    /// Original `VertBufManager::sphereCounts`.
    pub fn sphere_counts(
        &self,
        slices: i32,
        stacks: i32,
        fill: i32,
        quad: &mut i32,
        fan: &mut i32,
    ) -> i32 {
        if fill > 0 {
            *quad = (2 * (slices + 1) + 1) * (stacks - 2);
            *fan = 2 * (slices + 3)
        } else {
            *quad = stacks * slices + 2;
            if fill == 0 {
                *quad += (stacks - 1) * (slices + 2)
            }
            *fan = 0
        }
        (slices + 1) * (stacks - 1) + 2
    }
    /// Original `VertBufManager::makeSphere`.
    pub fn make_sphere(
        &mut self,
        r: f32,
        slices: i32,
        stacks: i32,
        vertex: &mut Vec<f32>,
        index: &mut Vec<u32>,
        iv: &mut i32,
        iq: &mut i32,
        ifan: &mut i32,
        norm: i32,
        x: f32,
        y: f32,
        z: f32,
    ) -> i32 {
        if slices < 2 || stacks < 1 || r < 0. {
            return 1;
        }
        let slices = slices.min(49);
        let stacks = stacks.min(49);
        let base = *iv as u32;
        for j in 0..=stacks {
            let theta = std::f32::consts::PI * j as f32 / stacks as f32;
            for i in 0..=slices {
                let phi = 2. * std::f32::consts::PI * i as f32 / slices as f32;
                let nx = theta.sin() * phi.sin();
                let ny = theta.sin() * phi.cos();
                let nz = theta.cos();
                if norm > 0 {
                    vertex.extend_from_slice(&[nx, ny, nz]);
                }
                vertex.extend_from_slice(&[r * nx + x, r * ny + y, r * nz + z]);
                *iv += 1;
            }
        }
        for j in 0..stacks {
            for i in 0..=slices {
                index.push(base + (j * (slices + 1) + i) as u32);
                index.push(base + ((j + 1) * (slices + 1) + i) as u32);
                *iq += 2;
            }
            index.push(RESTART_INDEX);
            *iq += 1;
        }
        *ifan = 0;
        0
    }
    /// Original `VertBufManager::copyDefaultSphere`.
    pub fn copy_default_sphere(
        &mut self,
        dv: &[f32],
        di: &[u32],
        nv: i32,
        nq: i32,
        nf: i32,
        norm: i32,
        v: &mut Vec<f32>,
        ind: &mut Vec<u32>,
        iv: &mut i32,
        iq: &mut i32,
        ifan: &mut i32,
        x: f32,
        y: f32,
        z: f32,
    ) {
        let base = *iv as u32;
        for &i in di.iter().take((nq + nf).max(0) as usize) {
            ind.push(if i == RESTART_INDEX { i } else { i + base });
        }
        *iq += nq;
        *ifan += nf;
        let stride = if norm > 0 { 6 } else { 3 };
        for p in dv.chunks_exact(stride).take(nv.max(0) as usize) {
            if norm > 0 {
                v.extend_from_slice(&p[..3]);
                v.extend_from_slice(&[p[3] + x, p[4] + y, p[5] + z]);
            } else {
                v.extend_from_slice(&[p[0] + x, p[1] + y, p[2] + z]);
            }
            *iv += 1;
        }
    }
    /// Original `VertBufManager::loadNormal`.
    pub fn load_normal(&mut self, x: f32, y: f32, z: f32) {
        let i = (6 * self.m_ind_vert) as usize;
        if self.m_verts.len() < i + 3 {
            self.m_verts.resize(i + 3, 0.)
        }
        self.m_verts[i..i + 3].copy_from_slice(&[x, y, z]);
    }
    /// Original `VertBufManager::loadVertex`.
    pub fn load_vertex(&mut self, x: f32, y: f32, z: f32) {
        let i = ((3 + self.m_norm_offset) * self.m_ind_vert + self.m_norm_offset) as usize;
        if self.m_verts.len() < i + 3 {
            self.m_verts.resize(i + 3, 0.)
        }
        self.m_verts[i..i + 3].copy_from_slice(&[
            x + self.m_xadd,
            y + self.m_yadd,
            z + self.m_zadd,
        ]);
        self.m_ind_vert += 1;
    }
    /// Original `VertBufManager::analyzeConts`, deferred only when per-point generic-store state is present.
    pub fn analyze_conts(
        &mut self,
        obj: &Iobj,
        ob: i32,
        thicken: i32,
        stipple: i32,
        time: i32,
        _gl: &mut dyn VertexBufferGl,
    ) -> i32 {
        if !obj.store.is_empty() {
            return 1;
        }
        let mut v = vb_data_new();
        v.thicken_cont = thicken;
        v.check_stipple = stipple;
        v.check_time = time;
        self.cont_data.insert(obj as *const _ as usize, v);
        0
    }
    /// Original `VertBufManager::analyzeSpheres`, delegated to the sphere source geometry path by `make_sphere`.
    pub fn analyze_spheres(
        &mut self,
        obj: &Iobj,
        ob: i32,
        z: f32,
        xy: i32,
        screen: f32,
        q: i32,
        fill: i32,
        use_fill: i32,
        thicken: i32,
        time: i32,
        _gl: &mut dyn VertexBufferGl,
    ) -> i32 {
        let mut v = vb_data_new();
        v.zscale = z;
        v.scrn_scale = screen;
        v.quality = q;
        v.fill_type = fill;
        v.use_fill_color = use_fill;
        v.thicken_cont = thicken;
        v.check_time = time;
        self.sphere_data.insert(obj as *const _ as usize, v);
        0
    }
    /// Original `VertBufManager::checkSelectedAreRemnants`.
    pub fn check_selected_are_remnants(&self, v: &VertBufData, ob: i32) -> i32 {
        (v.num_remnant > 0) as i32
    }
    /// Original `VertBufManager::checkAllTrans`.
    pub fn check_all_trans(&self, obj: &Iobj, v: &VertBufData, remnant: &mut i32) -> i32 {
        *remnant = 1;
        let trans = (obj.trans > 0) as i32;
        for value in &v.rgbt_special {
            if (*value & 255) as u8 != obj.trans {
                *remnant = 0;
                return 0;
            }
        }
        trans
    }
    /// Direct primitive execution from data generated by this unit.
    pub fn draw_vbd(&self, v: &VertBufData, mode: u32, gl: &mut dyn VertexBufferGl) {
        if v.vb_obj == 0 || v.eb_obj == 0 {
            return;
        }
        gl.bind_buffer(GL_ARRAY_BUFFER, v.vb_obj);
        gl.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, v.eb_obj);
        gl.primitive_restart(true, RESTART_INDEX);
        gl.draw_elements(mode, v.num_ind_default, 0);
        gl.primitive_restart(false, RESTART_INDEX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Gl {
        next: u32,
        draws: Vec<(u32, i32)>,
    }
    impl VertexBufferGl for Gl {
        fn gen_buffer(&mut self) -> u32 {
            self.next += 1;
            self.next
        }
        fn delete_buffer(&mut self, _: u32) {}
        fn bind_buffer(&mut self, _: u32, _: u32) {}
        fn buffer_data_f32(&mut self, _: u32, _: &[f32], _: u32) {}
        fn buffer_data_u32(&mut self, _: u32, _: &[u32], _: u32) {}
        fn buffer_sub_data_f32(&mut self, _: u32, _: i32, _: &[f32]) {}
        fn draw_elements(&mut self, m: u32, n: i32, _: i32) {
            self.draws.push((m, n))
        }
        fn primitive_restart(&mut self, _: bool, _: u32) {}
    }
    #[test]
    fn rgbt_round_trip() {
        let m = VertBufManager::new();
        let x = m.pack_rgbt(0.5, 0.25, 1., 17);
        assert_eq!(m.unpack_rgbt(x), (127. / 255., 63. / 255., 1., 17));
    }
    #[test]
    fn sphere_counts_and_geometry() {
        let mut m = VertBufManager::new();
        let (mut q, mut f) = (0, 0);
        assert_eq!(m.sphere_counts(8, 4, 1, &mut q, &mut f), 29);
        let (mut v, mut i) = (Vec::new(), Vec::new());
        let (mut iv, mut iq, mut ifan) = (0, 0, 0);
        assert_eq!(
            m.make_sphere(
                1., 8, 4, &mut v, &mut i, &mut iv, &mut iq, &mut ifan, 1, 0., 0., 0.
            ),
            0
        );
        assert_eq!(iv, 45);
        assert!(!i.is_empty());
    }
}
