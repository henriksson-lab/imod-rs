//! Translation of `IMOD/3dmod/isothread.cpp` and `isothread.h`.
//!
//! `IsoThread` is the per-Z-subslice worker owned by `ImodvIsosurface`.  Qt's
//! `QThread` supplies `start` and `wait` in the C++ source.  [`IsoThread::run`]
//! is its direct source method; [`IsoThread::start`] is the corresponding Rust
//! worker handoff.  The templated `mcubes.h::surface<unsigned char>` call is
//! represented by [`McubesBoundary`], until that complete template unit is
//! translated.  It deliberately does not hide or replace marching-cubes
//! geometry generation.
#![allow(dead_code)]

use std::thread::{self, JoinHandle};

use crate::imod::libimod::imodel::Ipoint;
use crate::imod::three_dmod::isosurface::{IsoColor, IsoPaintPoint};

/// `mcubescpp.h::smooth_vertex_positions`.  Marching-cubes vertices are
/// interleaved `[coordinate, normal]`; triangle entries name the even slot of
/// each vertex, as in native `Contour_Surface::geometry`.
pub fn smooth_vertex_positions(
    vertices: &mut [Ipoint],
    triangles: &[i32],
    smoothing_factor: f32,
    iterations: i32,
) {
    let vertex_count = vertices.len() / 2;
    if vertex_count == 0 || triangles.len() % 3 != 0 {
        return;
    }
    let triangle_vertices: Option<Vec<[usize; 3]>> = triangles
        .chunks_exact(3)
        .map(|triangle| {
            let indices = [triangle[0], triangle[1], triangle[2]];
            let mut output = [0; 3];
            for (slot, index) in indices.into_iter().enumerate() {
                if index < 0 || index % 2 != 0 || index as usize / 2 >= vertex_count {
                    return None;
                }
                output[slot] = index as usize / 2;
            }
            Some(output)
        })
        .collect();
    let Some(triangle_vertices) = triangle_vertices else {
        return;
    };
    let mut counts = vec![0_u32; vertex_count];
    for triangle in &triangle_vertices {
        for &vertex in triangle {
            counts[vertex] += 2;
        }
    }
    let fixed = 1. - smoothing_factor;
    for _ in 0..iterations.max(0) {
        let mut sums = vec![Ipoint::default(); vertex_count];
        for &[first, second, third] in &triangle_vertices {
            let a = vertices[2 * first];
            let b = vertices[2 * second];
            let c = vertices[2 * third];
            sums[first].x += b.x + c.x;
            sums[first].y += b.y + c.y;
            sums[first].z += b.z + c.z;
            sums[second].x += a.x + c.x;
            sums[second].y += a.y + c.y;
            sums[second].z += a.z + c.z;
            sums[third].x += a.x + b.x;
            sums[third].y += a.y + b.y;
            sums[third].z += a.z + b.z;
        }
        for vertex in 0..vertex_count {
            if counts[vertex] == 0 {
                continue;
            }
            let average = &sums[vertex];
            let coordinate = &mut vertices[2 * vertex];
            coordinate.x =
                fixed * coordinate.x + smoothing_factor * average.x / counts[vertex] as f32;
            coordinate.y =
                fixed * coordinate.y + smoothing_factor * average.y / counts[vertex] as f32;
            coordinate.z =
                fixed * coordinate.z + smoothing_factor * average.z / counts[vertex] as f32;
        }
    }
    for vertex in 0..vertex_count {
        vertices[2 * vertex + 1] = Ipoint::default();
    }
    for &[first, second, third] in &triangle_vertices {
        let a = vertices[2 * first];
        let b = vertices[2 * second];
        let c = vertices[2 * third];
        let ux = b.x - a.x;
        let uy = b.y - a.y;
        let uz = b.z - a.z;
        let vx = c.x - a.x;
        let vy = c.y - a.y;
        let vz = c.z - a.z;
        let normal = Ipoint {
            x: uy * vz - uz * vy,
            y: uz * vx - ux * vz,
            z: ux * vy - uy * vx,
        };
        for vertex in [first, second, third] {
            vertices[2 * vertex + 1].x += normal.x;
            vertices[2 * vertex + 1].y += normal.y;
            vertices[2 * vertex + 1].z += normal.z;
        }
    }
}

/// `MAX_THREADS` from `isosurface.h`.
pub const MAX_THREADS: usize = 16;

/// The fields of `ImodvIsosurface` read by `IsoThread::run`.
///
/// The complete dialog/controller class belongs to `isosurface.cpp`; keeping
/// this subset under its upstream type name means the later source translation
/// can extend it rather than introduce a parallel worker-only representation.
#[derive(Clone, Debug, PartialEq)]
pub struct ImodvIsosurface {
    pub m_box_origin: [i32; 3],
    pub m_bin_box_size: [i32; 3],
    pub m_sub_z_ends: [i32; MAX_THREADS + 1],
    pub m_bin_volume: Vec<u8>,
    pub m_n_threads: i32,
    pub m_threshold: f32,
    /// Backing state of source `getBinning`, which is read from the
    /// isosurface dialog's file-static parameter store in C++.
    pub binning: i32,
    // The remaining source-owned fields are initialized and manipulated by
    // `isosurface.cpp`; keeping them on this exact upstream class avoids a
    // second, worker-only controller representation.
    pub m_local_x: i32,
    pub m_local_y: i32,
    pub m_local_z: i32,
    pub m_mask_obj: i32,
    pub m_mask_cont: i32,
    pub m_mask_psize: i32,
    pub m_mask_checksum: f64,
    pub m_curr_time: i32,
    pub m_curr_stack_index: i32,
    pub m_box_ends: [i32; 3],
    pub m_range_low: i32,
    pub m_range_high: i32,
    pub m_stack_thresholds: Vec<f32>,
    pub m_median: f32,
    pub m_stack_outer_lims: Vec<i32>,
    pub m_outer_limit: i32,
    pub m_volume: Vec<u8>,
    pub m_true_bin_vol: Vec<u8>,
    pub m_paint_vol: Vec<u8>,
    /// Native `mColorList` and `mPaintPoints`, retained by `fillPaintVol` to
    /// detect whether the rasterized mask is still current.
    pub m_color_list: Vec<IsoColor>,
    pub m_paint_points: Vec<IsoPaintPoint>,
    pub m_box_size: [i32; 3],
    pub m_bin_box_ends: [i32; 3],
    pub m_paint_size: [i32; 3],
    pub m_vol_min: i32,
    pub m_vol_max: i32,
    pub m_init_n_threads: i32,
    pub m_last_objsize: i32,
    pub m_paint_obj_list: Vec<i32>,
    pub m_param_load_confirmed: i32,
    pub m_new_model_opened: bool,
    pub m_ctrl_pressed: bool,
    /// `DialogFrame::mRoundedStyle`, refreshed by `topChangeEvent`.
    pub m_rounded_style: bool,
    /// The file-static top-window pointer is cleared by `topCloseEvent`.
    pub m_top_window_open: bool,
    pub m_extra_obj_num: i32,
    pub m_box_obj_num: i32,
    /// `ImodvIsosurface::mMadeFiltered`: whether `mFilteredMesh` is a
    /// separately allocated filtered mesh rather than `mOrigMesh`.
    pub m_made_filtered: bool,
}

impl Default for ImodvIsosurface {
    fn default() -> Self {
        Self {
            m_box_origin: [0; 3],
            m_bin_box_size: [0; 3],
            m_sub_z_ends: [0; MAX_THREADS + 1],
            m_bin_volume: Vec::new(),
            m_n_threads: 1,
            m_threshold: 0.,
            binning: 1,
            m_local_x: 0,
            m_local_y: 0,
            m_local_z: 0,
            m_mask_obj: -1,
            m_mask_cont: -1,
            m_mask_psize: 0,
            m_mask_checksum: 0.,
            m_curr_time: 0,
            m_curr_stack_index: 0,
            m_box_ends: [0; 3],
            m_range_low: 0,
            m_range_high: 255,
            m_stack_thresholds: vec![-1.],
            m_median: -1.,
            m_stack_outer_lims: vec![-1],
            m_outer_limit: -1,
            m_volume: Vec::new(),
            m_true_bin_vol: Vec::new(),
            m_paint_vol: Vec::new(),
            m_color_list: Vec::new(),
            m_paint_points: Vec::new(),
            m_box_size: [0; 3],
            m_bin_box_ends: [0; 3],
            m_paint_size: [0; 3],
            m_vol_min: 255,
            m_vol_max: 0,
            m_init_n_threads: 1,
            m_last_objsize: 0,
            m_paint_obj_list: Vec::new(),
            m_param_load_confirmed: 1,
            m_new_model_opened: false,
            m_ctrl_pressed: false,
            m_rounded_style: false,
            m_top_window_open: true,
            m_extra_obj_num: -1,
            m_box_obj_num: -1,
            m_made_filtered: false,
        }
    }
}

/// The `Contour_Surface` virtual interface from `mcubes.h`.
///
/// `vertex_xyz` has exactly `2 * vertex_count` points: even points are vertex
/// coordinates and odd points are their normals.  This is the nonstandard
/// layout written by the upstream `mcubescpp.h::normals` implementation.
pub trait ContourSurface {
    fn vertex_count(&self) -> u32;
    fn triangle_count(&self) -> u32;
    fn geometry(
        &mut self,
        vertex_xyz: &mut [Ipoint],
        triangle_vertex_indices: &mut [i32],
        origin: [i32; 3],
        bin_num: i32,
    );
    fn normals(&mut self, vertex_xyz: &mut [Ipoint]);
}

/// Boundary at the templated `surface<unsigned char>` invocation in
/// `mcubes.h`.  Size and stride retain the source's `Index` (`unsigned int`)
/// representation.
pub trait McubesBoundary {
    type Surface: ContourSurface;

    fn surface(
        &mut self,
        grid: &[u8],
        size: [u32; 3],
        stride: [u32; 3],
        threshold: f32,
        cap_faces: bool,
    ) -> Self::Surface;
}

/// `IsoThread` from `isothread.h`.
///
/// The C++ worker retains an `ImodvIsosurface *`.  Rust snapshots only the
/// immutable fields read by `run`, allowing the same worker to be sent to a
/// native thread without aliasing the live Qt dialog.
pub struct IsoThread<B: McubesBoundary> {
    m_which_subslice: i32,
    m_iso: ImodvIsosurface,
    m_n_vertex: i32,
    m_n_triangle: i32,
    m_vertex_xyz: Vec<Ipoint>,
    m_triangle: Vec<i32>,
    mcubes: B,
}

impl<B: McubesBoundary> IsoThread<B> {
    /// `IsoThread()`, the C++ `IsoThread::IsoThread` constructor.
    pub fn new(subslice_index: i32, iso: &ImodvIsosurface, mcubes: B) -> Self {
        Self {
            m_which_subslice: subslice_index,
            m_iso: iso.clone(),
            m_n_vertex: 0,
            m_n_triangle: 0,
            m_vertex_xyz: Vec::new(),
            m_triangle: Vec::new(),
            mcubes,
        }
    }

    /// `IsoThread::getVertex_xyz`.
    pub fn get_vertex_xyz(&self) -> &[Ipoint] {
        &self.m_vertex_xyz
    }

    /// `IsoThread::getTriangle`.
    pub fn get_triangle(&self) -> &[i32] {
        &self.m_triangle
    }

    /// `IsoThread::getNVertex`.
    pub fn get_n_vertex(&self) -> i32 {
        self.m_n_vertex
    }

    /// `IsoThread::getNTriangle`.
    pub fn get_n_triangle(&self) -> i32 {
        self.m_n_triangle
    }

    /// Rust counterpart of the inherited `QThread::start`.  The caller joins
    /// this handle where upstream calls `QThread::wait`.
    pub fn start(self) -> JoinHandle<Self>
    where
        B: Send + 'static,
        B::Surface: Send + 'static,
    {
        thread::spawn(move || {
            let mut worker = self;
            worker.run();
            worker
        })
    }

    /// `IsoThread::run`.
    pub fn run(&mut self) {
        let box_size = self.m_iso.m_bin_box_size;
        let box_origin = self.m_iso.m_box_origin;
        let sub_z_ends = self.m_iso.m_sub_z_ends;
        let volume = &self.m_iso.m_bin_volume;

        let mut curr_size = [0; 3];
        curr_size[0] = box_size[0];
        curr_size[1] = box_size[1];
        let mut curr_origin = [0; 3];
        curr_origin[0] = box_origin[0];
        curr_origin[1] = box_origin[1];
        let stride = [1, box_size[0], box_size[0] * box_size[1]];

        let i = self.m_which_subslice;
        let bin_num = self.m_iso.get_binning();
        if self.m_iso.m_n_threads == 1 {
            curr_size[2] = sub_z_ends[(i + 1) as usize] - sub_z_ends[i as usize] + 1;
        } else if i == 0 || i == self.m_iso.m_n_threads - 1 {
            curr_size[2] = sub_z_ends[(i + 1) as usize] - sub_z_ends[i as usize] + 1 + 1;
        } else {
            curr_size[2] = sub_z_ends[(i + 1) as usize] - sub_z_ends[i as usize] + 1 + 1 + 1;
        }

        let sub_volume_ptr = if i == 0 {
            curr_origin[2] = sub_z_ends[i as usize];
            &volume[..]
        } else {
            curr_origin[2] =
                sub_z_ends[0] + bin_num * (sub_z_ends[i as usize] - sub_z_ends[0]) - bin_num;
            let offset =
                (box_size[0] * box_size[1] * (sub_z_ends[i as usize] - sub_z_ends[0] - 1)) as usize;
            &volume[offset..]
        };

        let mut cs = self.mcubes.surface(
            sub_volume_ptr,
            [
                curr_size[0] as u32,
                curr_size[1] as u32,
                curr_size[2] as u32,
            ],
            [stride[0] as u32, stride[1] as u32, stride[2] as u32],
            self.m_iso.m_threshold,
            false,
        );
        self.m_n_vertex = cs.vertex_count() as i32;
        self.m_n_triangle = cs.triangle_count() as i32;
        self.m_vertex_xyz = vec![Ipoint::default(); 2 * self.m_n_vertex as usize];
        if i == 0 {
            self.m_triangle = vec![0; 3 * self.m_n_triangle as usize + 3];
            cs.geometry(
                &mut self.m_vertex_xyz,
                &mut self.m_triangle[1..1 + 3 * self.m_n_triangle as usize],
                curr_origin,
                bin_num,
            );
        } else {
            self.m_triangle = vec![0; 3 * self.m_n_triangle as usize];
            cs.geometry(
                &mut self.m_vertex_xyz,
                &mut self.m_triangle,
                curr_origin,
                bin_num,
            );
        }
        cs.normals(&mut self.m_vertex_xyz);
    }
}

/// `IsoThread::~IsoThread`.
///
/// The C++ destructor is empty.  Rust drops the owned worker snapshot, output
/// arrays, and marching-cubes boundary at this same lifetime point.
impl<B: McubesBoundary> Drop for IsoThread<B> {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::{
        ContourSurface, ImodvIsosurface, IsoThread, McubesBoundary, smooth_vertex_positions,
    };
    use crate::imod::libimod::imodel::Ipoint;

    #[test]
    fn smoothing_uses_even_marching_cubes_indices_and_recomputes_normals() {
        let mut vertices = vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint::default(),
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint::default(),
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
            Ipoint::default(),
        ];
        smooth_vertex_positions(&mut vertices, &[0, 2, 4], 0.3, 1);
        assert_eq!(
            vertices[0],
            Ipoint {
                x: 0.15,
                y: 0.15,
                z: 0.
            }
        );
        assert!((vertices[1].z - 0.3025).abs() < 1.0e-5);
    }

    #[derive(Clone, Debug)]
    struct MockSurface {
        vertices: u32,
        triangles: u32,
    }

    impl ContourSurface for MockSurface {
        fn vertex_count(&self) -> u32 {
            self.vertices
        }
        fn triangle_count(&self) -> u32 {
            self.triangles
        }
        fn geometry(
            &mut self,
            vertex_xyz: &mut [Ipoint],
            triangle_vertex_indices: &mut [i32],
            origin: [i32; 3],
            bin_num: i32,
        ) {
            vertex_xyz[0] = Ipoint {
                x: origin[0] as f32,
                y: origin[1] as f32,
                z: origin[2] as f32,
            };
            for (index, value) in triangle_vertex_indices.iter_mut().enumerate() {
                *value = index as i32 + bin_num;
            }
        }
        fn normals(&mut self, vertex_xyz: &mut [Ipoint]) {
            vertex_xyz[1] = Ipoint {
                x: 1.,
                y: 2.,
                z: 3.,
            };
        }
    }

    #[derive(Clone, Debug)]
    struct MockMcubes;

    impl McubesBoundary for MockMcubes {
        type Surface = MockSurface;
        fn surface(
            &mut self,
            _grid: &[u8],
            _size: [u32; 3],
            _stride: [u32; 3],
            _threshold: f32,
            _cap_faces: bool,
        ) -> Self::Surface {
            MockSurface {
                vertices: 2,
                triangles: 1,
            }
        }
    }

    #[test]
    fn first_subslice_keeps_the_source_one_index_triangle_offset() {
        let mut iso = ImodvIsosurface {
            m_box_origin: [10, 20, 30],
            m_bin_box_size: [2, 3, 4],
            m_bin_volume: vec![0; 24],
            m_n_threads: 1,
            binning: 2,
            ..Default::default()
        };
        iso.m_sub_z_ends[0] = 4;
        iso.m_sub_z_ends[1] = 7;
        let mut thread = IsoThread::new(0, &iso, MockMcubes);
        thread.run();
        assert_eq!(thread.get_n_vertex(), 2);
        assert_eq!(thread.get_n_triangle(), 1);
        assert_eq!(thread.get_vertex_xyz()[0].z, 4.);
        assert_eq!(thread.get_vertex_xyz()[1].z, 3.);
        assert_eq!(thread.get_triangle(), &[0, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn later_subslice_uses_preceding_plane_and_can_run_on_a_worker() {
        let mut iso = ImodvIsosurface {
            m_box_origin: [10, 20, 30],
            m_bin_box_size: [2, 3, 5],
            m_bin_volume: vec![0; 30],
            m_n_threads: 3,
            binning: 2,
            ..Default::default()
        };
        iso.m_sub_z_ends[0] = 2;
        iso.m_sub_z_ends[1] = 4;
        iso.m_sub_z_ends[2] = 6;
        iso.m_sub_z_ends[3] = 8;
        let thread = IsoThread::new(1, &iso, MockMcubes);
        let thread = thread.start().join().expect("IsoThread worker panicked");
        assert_eq!(thread.get_vertex_xyz()[0].z, 4.);
        assert_eq!(thread.get_triangle(), &[2, 3, 4]);
    }
}
