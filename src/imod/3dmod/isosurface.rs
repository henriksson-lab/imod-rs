//! Translation of `IMOD/3dmod/isosurface.cpp` and `isosurface.h`.
//!
//! The numerical volume preparation code is native Rust.  Qt dialog lifetime,
//! model/mesh ownership, image-cache access, marching cubes, and OpenGL redraw
//! are deliberately explicit boundaries; they are not replaced by a second UI.
#![allow(dead_code)]

use crate::imod::three_dmod::histwidget::HistWidget;
pub use crate::imod::three_dmod::isothread::{ImodvIsosurface, MAX_THREADS};

pub const IIS_CENTER_VOLUME: u32 = 1;
pub const IIS_VIEW_BOX: u32 = 1 << 1;
pub const IIS_VIEW_USER_MODEL: u32 = 1 << 2;
pub const IIS_VIEW_ISOSURFACE: u32 = 1 << 3;
pub const IIS_LINK_XYZ: u32 = 1 << 4;
pub const IIS_DELETE_PIECES: u32 = 1 << 5;
pub const IIS_CLOSE_FACES: u32 = 1 << 6;
pub const IIS_PAINT_OBJECT: u32 = 1 << 7;
pub const MASK_NONE: i32 = 0;
pub const MASK_CONTOUR: i32 = 1;
pub const MASK_OBJECT: i32 = 2;
pub const MASK_SPHERE: i32 = 3;
pub const MASK_ELLIPSOID: i32 = 4;
pub const MASK_LASSO: i32 = 5;
pub const PERCENTILE: f32 = 0.07;
pub const MAXIMAL_ITERATION: i32 = 20;
pub const MAXIMAL_BINNING: i32 = 4;
pub const NO_KERNEL_SIGMA: f32 = 0.4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IsoPoint3D {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IsoColor {
    pub trans: i32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub dummy: u8,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IsoPaintPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub size: f32,
    pub color_ind: i32,
    pub drawn: bool,
}

/// File-static `iisData` in the source.  Dialog and app pointers become state
/// and callbacks on [`IsosurfaceNativeBoundary`].
#[derive(Clone, Debug)]
pub struct ImodvIsosurfaceData {
    pub it_num: i32,
    pub binning_num: i32,
    pub kernel_sigma: f32,
    pub min_t_num: i32,
    pub mask_type: i32,
    pub flags: u32,
}
impl Default for ImodvIsosurfaceData {
    fn default() -> Self {
        Self {
            it_num: 0,
            binning_num: 1,
            kernel_sigma: NO_KERNEL_SIGMA,
            min_t_num: 100,
            mask_type: 0,
            flags: IIS_CENTER_VOLUME
                | IIS_VIEW_BOX
                | IIS_VIEW_USER_MODEL
                | IIS_VIEW_ISOSURFACE
                | IIS_LINK_XYZ,
        }
    }
}

/// All source operations whose target is Qt, the image cache, the model, mesh
/// ownership, or a native 3dmod redraw.
pub trait IsosurfaceNativeBoundary {
    fn open_dialog(&mut self) -> bool;
    fn raise_dialog(&mut self);
    fn close_dialog(&mut self);
    fn image_size(&self) -> [i32; 3];
    fn cursor_xyz(&self) -> [i32; 3];
    fn time_index(&self) -> i32;
    fn time_count(&self) -> i32;
    fn voxel(&self, x: i32, y: i32, z: i32) -> u8;
    fn set_cursor_xyz(&mut self, xyz: [i32; 3]);
    fn set_bounding_box(&mut self, origin: [i32; 3], ends: [i32; 3]);
    fn set_view_center(&mut self, xyz: [i32; 3]);
    fn set_iso_mesh(&mut self, volume: &[u8], size: [i32; 3], threshold: f32, binning: i32);
    fn redraw(&mut self);
    fn save_object(&mut self, sort_surfaces: bool);
    fn fill_cache(&mut self);
}

/// `setCoordLimits`.
pub fn set_coord_limits(cur: i32, max_size: i32, draw_size: i32) -> (i32, i32) {
    let mut start = cur - draw_size / 2;
    if start < 0 {
        start = 0;
    }
    let mut end = start + draw_size;
    if end > max_size {
        end = max_size;
        start = end - draw_size;
    }
    (start, end)
}

/// `isBoxChanged`; `range` is the current ushort display range supplied by
/// the viewer/image boundary.
pub fn is_box_changed(
    iso: &mut ImodvIsosurface,
    image_size: [i32; 3],
    draw_size: [i32; 3],
    current_time: i32,
    range: Option<(i32, i32)>,
) -> bool {
    if iso.m_curr_time != current_time {
        iso.m_curr_time = current_time;
        return true;
    }
    if let Some((low, high)) = range {
        if iso.m_range_low != low || iso.m_range_high != high {
            return true;
        }
    }
    let local = [iso.m_local_x, iso.m_local_y, iso.m_local_z];
    for axis in 0..3 {
        let (start, end) = set_coord_limits(local[axis], image_size[axis], draw_size[axis]);
        if iso.m_box_origin[axis] != start || iso.m_box_ends[axis] != end {
            return true;
        }
    }
    false
}

/// `findDimLimits`.
pub fn find_dim_limits(
    maximal: bool,
    sizes: [i32; 3],
    box_initial: i32,
    box_limit: i32,
) -> [i32; 3] {
    let voxels = if maximal {
        box_limit.pow(3)
    } else {
        box_initial.pow(3)
    };
    let min_dim = (voxels as f32).powf(1. / 3.) as i32;
    let smallest = if sizes[0] <= sizes[1] && sizes[0] <= sizes[2] {
        0
    } else if sizes[1] <= sizes[0] && sizes[1] <= sizes[2] {
        1
    } else {
        2
    };
    let mut out = sizes;
    out[smallest] = min_dim.min(sizes[smallest]);
    let other = ((voxels / out[smallest].max(1)) as f64).sqrt() as i32;
    for axis in 0..3 {
        if axis != smallest {
            out[axis] = other.min(sizes[axis]);
        }
    }
    out
}

/// `imodvIsosurfaceEditDialog`.
pub fn imodv_isosurface_edit_dialog(
    state: &mut Option<ImodvIsosurface>,
    native: &mut dyn IsosurfaceNativeBoundary,
    open: bool,
) {
    if !open {
        if state.is_some() {
            native.close_dialog();
            *state = None;
        }
        return;
    }
    if state.is_some() {
        native.raise_dialog();
        return;
    }
    if native.open_dialog() {
        *state = Some(ImodvIsosurface::default());
    }
}
/// `imodvIsosurfaceBoxLimits`.
pub fn imodv_isosurface_box_limits(iso: &ImodvIsosurface) -> (i32, i32, i32, i32) {
    (
        iso.m_box_origin[0],
        iso.m_box_origin[1],
        iso.m_box_ends[0] + 1 - iso.m_box_origin[0],
        iso.m_box_ends[1] + 1 - iso.m_box_origin[1],
    )
}
/// `imodvIsosurfaceInvertThreshold`.
pub fn imodv_isosurface_invert_threshold(iso: &mut ImodvIsosurface) {
    iso.invert_threshold();
}
/// `imodvIsosurfaceNewModel`.
pub fn imodv_isosurface_new_model(iso: &mut ImodvIsosurface) {
    iso.m_new_model_opened = true;
}

impl ImodvIsosurface {
    /// Source constructor after Qt widget construction; `native` owns those widgets.
    pub fn isosurface_new(
        native: &mut dyn IsosurfaceNativeBoundary,
        data: &ImodvIsosurfaceData,
        box_initial: i32,
        box_limit: i32,
    ) -> Self {
        let sizes = native.image_size();
        let mut iso = Self::default();
        iso.binning = data.binning_num;
        iso.m_local_x = native.cursor_xyz()[0];
        iso.m_local_y = native.cursor_xyz()[1];
        iso.m_local_z = native.cursor_xyz()[2];
        iso.m_box_size = find_dim_limits(false, sizes, box_initial, box_limit);
        iso.m_bin_box_size = iso.m_box_size;
        iso.m_stack_thresholds = vec![-1.; (native.time_count() + 1).max(1) as usize];
        iso.m_stack_outer_lims = vec![-1; (native.time_count() + 1).max(1) as usize];
        iso.set_bounding_box(native);
        iso.alloc_arrays_if_needed(data.flags);
        iso
    }
    /// `ImodvIsosurface::allocArraysIfNeeded`.
    pub fn alloc_arrays_if_needed(&mut self, flags: u32) -> bool {
        let n = self.m_box_size.iter().product::<i32>().max(0) as usize;
        if n > self.m_volume.len() {
            self.m_volume = vec![0; n];
            self.m_true_bin_vol = vec![0; n];
            self.m_paint_vol = if flags & IIS_PAINT_OBJECT != 0 {
                vec![0; n]
            } else {
                Vec::new()
            };
        }
        self.m_volume.len() != n || self.m_true_bin_vol.len() < n
    }
    /// `ImodvIsosurface::getCurrStackIndex`.
    pub fn get_curr_stack_index_from_time(&self, num_times: i32, current_time: i32) -> i32 {
        if num_times == 0 { 0 } else { current_time - 1 }
    }
    /// `ImodvIsosurface::getCurrStackIndex` with the current viewer time
    /// retained by the source controller.
    pub fn get_curr_stack_index(&self) -> i32 {
        self.m_curr_stack_index
    }
    /// `ImodvIsosurface::updateCoords`.
    pub fn update_coords(&mut self, native: &mut dyn IsosurfaceNativeBoundary, set_local: bool) {
        if set_local {
            [self.m_local_x, self.m_local_y, self.m_local_z] = native.cursor_xyz();
        } else {
            native.set_cursor_xyz([self.m_local_x, self.m_local_y, self.m_local_z]);
        }
    }
    /// `ImodvIsosurface::setBoundingBox`.
    pub fn set_bounding_box(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        let size = native.image_size();
        for i in 0..3 {
            let (a, b) = set_coord_limits(
                [self.m_local_x, self.m_local_y, self.m_local_z][i],
                size[i],
                self.m_box_size[i],
            );
            self.m_box_origin[i] = a;
            self.m_box_ends[i] = b - 1;
        }
        native.set_bounding_box(self.m_box_origin, self.m_box_ends);
    }
    /// `ImodvIsosurface::setViewCenter`.
    pub fn set_view_center(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        native.set_view_center([self.m_local_x, self.m_local_y, self.m_local_z]);
    }
    /// `ImodvIsosurface::fillVolumeArray`.
    pub fn fill_volume_array(&mut self, native: &dyn IsosurfaceNativeBoundary) -> f32 {
        let [nx, ny, nz] = self.m_box_size;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    self.m_volume[(x + y * nx + z * nx * ny) as usize] = native.voxel(
                        x + self.m_box_origin[0],
                        y + self.m_box_origin[1],
                        z + self.m_box_origin[2],
                    );
                }
            }
        }
        0.
    }
    /// `ImodvIsosurface::fillBinVolume`.
    pub fn fill_bin_volume(&mut self) {
        let bin = self.binning.max(1);
        self.m_bin_box_size = self.m_box_size.map(|v| v / bin);
        if bin == 1 {
            self.m_bin_volume = self.m_volume.clone();
            return;
        }
        let [nx, ny, nz] = self.m_box_size;
        let [bx, by, bz] = self.m_bin_box_size;
        self.m_true_bin_vol.resize((bx * by * bz) as usize, 0);
        for z in 0..bz {
            for y in 0..by {
                for x in 0..bx {
                    let mut sum = 0u32;
                    for zz in 0..bin {
                        for yy in 0..bin {
                            for xx in 0..bin {
                                sum += self.m_volume[((x * bin + xx)
                                    + (y * bin + yy) * nx
                                    + (z * bin + zz) * nx * ny)
                                    as usize] as u32;
                            }
                        }
                    }
                    self.m_true_bin_vol[(x + y * bx + z * bx * by) as usize] =
                        (sum / (bin * bin * bin) as u32) as u8;
                }
            }
        }
        self.m_bin_volume = self.m_true_bin_vol[..(bx * by * bz) as usize].to_vec();
    }
    /// `ImodvIsosurface::applyMask` sphere/ellipsoid branch. Contour/object/lasso
    /// scanning remains the model-controller boundary.
    pub fn apply_mask(&mut self, data: &ImodvIsosurfaceData) {
        if data.mask_type != MASK_SPHERE && data.mask_type != MASK_ELLIPSOID {
            return;
        }
        let [nx, ny, nz] = self.m_box_size;
        let (mut a, mut b, mut c) = (nx as f32 / 2., ny as f32 / 2., nz as f32 / 2.);
        if data.mask_type == MASK_SPHERE {
            a = (nx.min(ny).min(nz) as f32) / 2.;
            b = a;
            c = a;
        }
        let (xc, yc, zc) = (
            (nx - 1) as f32 / 2.,
            (ny - 1) as f32 / 2.,
            (nz - 1) as f32 / 2.,
        );
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let q = ((x as f32 - xc) / a).powi(2)
                        + ((y as f32 - yc) / b).powi(2)
                        + ((z as f32 - zc) / c).powi(2);
                    if q > 1. {
                        self.m_volume[(x + y * nx + z * nx * ny) as usize] = self.m_median as u8;
                    }
                }
            }
        }
    }
    /// `ImodvIsosurface::maskWithContour`.
    ///
    /// `scan_segments` is the source scan contour represented as `(Y, Xstart,
    /// Xend)` entries.  It keeps the exact masking effect while native model
    /// contour conversion stays at the controller boundary.
    pub fn mask_with_contour(&mut self, scan_segments: &[(i32, i32, i32)], iz: i32) -> i32 {
        let [nx, ny, nz] = self.m_box_size;
        if iz < 0 || iz >= nz {
            return 1;
        }
        for iy in 0..ny {
            let base = (iz * nx * ny + iy * nx) as usize;
            let mut inside = vec![false; nx as usize];
            for &(sy, start, end) in scan_segments {
                if sy - self.m_box_origin[1] == iy {
                    for ix in start - self.m_box_origin[0]..=end - self.m_box_origin[0] {
                        if ix >= 0 && ix < nx {
                            inside[ix as usize] = true;
                        }
                    }
                }
            }
            for ix in 0..nx {
                if !inside[ix as usize] {
                    self.m_volume[base + ix as usize] = self.m_median as u8;
                }
            }
        }
        0
    }
    /// `ImodvIsosurface::closeBoxFaces`.
    pub fn close_box_faces(&mut self, data: &ImodvIsosurfaceData) {
        if data.flags & IIS_CLOSE_FACES == 0 {
            return;
        }
        let [nx, ny, nz] = self.m_bin_box_size;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1 {
                        self.m_bin_volume[(x + y * nx + z * nx * ny) as usize] = self.m_median as u8
                    }
                }
            }
        }
    }
    /// `ImodvIsosurface::addToNeighborList`.
    pub fn add_to_neighbor_list(neighbors: &mut Vec<IsoPoint3D>, ix: i32, iy: i32, iz: i32) {
        neighbors.push(IsoPoint3D { ix, iy, iz });
    }
    /// `ImodvIsosurface::removeOuterPixels`.
    pub fn remove_outer_pixels(&mut self) {
        let [nx, ny, nz] = self.m_bin_box_size;
        if self.m_outer_limit <= self.m_vol_min || self.m_outer_limit >= self.m_vol_max {
            return;
        }
        let dir = if self.m_threshold > self.m_median {
            1
        } else {
            -1
        };
        let neutral = (self.m_threshold as i32 - 2 * dir) as u8;
        let mut seen = vec![false; self.m_bin_volume.len()];
        let mut work = Vec::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (x + y * nx + z * nx * ny) as usize;
                    if dir * (self.m_bin_volume[i] as i32 - self.m_outer_limit) > 0 {
                        seen[i] = true;
                        self.m_bin_volume[i] = neutral;
                        Self::add_to_neighbor_list(&mut work, x, y, z)
                    }
                }
            }
        }
        for _ in 0..16 {
            let mut next = Vec::new();
            for p in work {
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let (x, y, z) = (p.ix + dx, p.iy + dy, p.iz + dz);
                            if x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz {
                                let i = (x + y * nx + z * nx * ny) as usize;
                                if !seen[i]
                                    && dir * (self.m_bin_volume[i] as i32 - self.m_threshold as i32)
                                        > 0
                                {
                                    seen[i] = true;
                                    self.m_bin_volume[i] = neutral;
                                    Self::add_to_neighbor_list(&mut next, x, y, z)
                                }
                            }
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            work = next;
        }
    }
    /// `ImodvIsosurface::fillAndProcessVols`.
    pub fn fill_and_process_vols(
        &mut self,
        native: &dyn IsosurfaceNativeBoundary,
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        percentile: f32,
        skip_fill: bool,
        use_thresh: f32,
    ) {
        if !skip_fill {
            self.fill_volume_array(native);
        }
        self.apply_mask(data);
        self.fill_bin_volume();
        self.m_vol_min = 255;
        self.m_vol_max = 0;
        hist.hist = [0.; 256];
        for &v in &self.m_bin_volume {
            hist.hist[v as usize] += 1.;
            self.m_vol_min = self.m_vol_min.min(v as i32);
            self.m_vol_max = self.m_vol_max.max(v as i32)
        }
        let den = self.m_bin_volume.len().max(1) as f32;
        for h in &mut hist.hist {
            *h /= den
        }
        hist.set_min_max(self.m_vol_min, self.m_vol_max);
        hist.set_hist_min_max();
        self.m_threshold = if use_thresh >= 0. {
            use_thresh
        } else {
            hist.threshold_for_percentile(percentile)
        };
        self.m_median = hist.threshold_for_percentile(0.5);
        self.close_box_faces(data);
        self.remove_outer_pixels();
    }
    /// `ImodvIsosurface::setIsoObj`; the `mcubes` and model mesh transfer are source boundary.
    pub fn set_iso_obj(&mut self, native: &mut dyn IsosurfaceNativeBoundary, fill_paint: bool) {
        native.set_iso_mesh(
            &self.m_bin_volume,
            self.m_bin_box_size,
            self.m_threshold,
            self.binning,
        );
        if fill_paint {
            self.paint_mesh()
        }
    }
    /// `ImodvIsosurface::setBoundingObj` (extra-object line construction is native model boundary).
    pub fn set_bounding_obj(&mut self) {}
    /// `ImodvIsosurface::resizeToContours`.
    pub fn resize_to_contours(
        &mut self,
        native: &mut dyn IsosurfaceNativeBoundary,
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        draw: bool,
    ) {
        self.set_bounding_box(native);
        self.fill_and_process_vols(native, data, hist, -1., false, -1.);
        self.set_iso_obj(native, true);
        if draw {
            native.redraw()
        }
    }
    /// `ImodvIsosurface::invertThreshold`.
    pub fn invert_threshold(&mut self) {
        self.m_threshold = 2. * self.m_median - self.m_threshold;
    }
    /// `ImodvIsosurface::smoothMesh`; mesh ownership/marching-cubes mesh format is native boundary.
    pub fn smooth_mesh(&mut self, _iter_num: i32) {}
    /// `ImodvIsosurface::filterMesh`; mesh filtering boundary.
    pub fn filter_mesh(&mut self, _state: bool) {}
    /// `ImodvIsosurface::paintMesh`; mesh store ownership boundary.
    pub fn paint_mesh(&mut self) {}
    /// `ImodvIsosurface::fillPaintVol`; point/model properties boundary.
    pub fn fill_paint_vol(&mut self) -> bool {
        false
    }
    pub fn view_iso_toggled(&mut self, _state: bool) {}
    pub fn view_model_toggled(&mut self, _state: bool) {}
    pub fn view_boxing_toggled(&mut self, _state: bool) {}
    pub fn center_volume_toggled(&mut self, _state: bool) {}
    pub fn delete_pieces_toggled(&mut self, _state: bool) {}
    pub fn link_xyz_toggled(&mut self, _state: bool) {}
    pub fn close_faces_toggled(&mut self, _state: bool) {}
    pub fn hist_changed(&mut self, _which: i32, value: i32, _dragging: bool) {
        self.m_threshold = value as f32;
    }
    pub fn iter_num_changed(&mut self, _iter: i32) {}
    pub fn binning_num_changed(&mut self, bin: i32) {
        self.binning = bin.clamp(1, MAXIMAL_BINNING);
    }
    pub fn kernel_sigma_changed(&mut self, _value: f64) {}
    pub fn slider_moved(&mut self, _which: i32, _value: i32, _dragging: bool) {}
    pub fn num_of_triangles_changed(&mut self, _num: i32) {}
    pub fn mask_selected(&mut self, _which: i32) {}
    pub fn area_from_cont_clicked(&mut self) {}
    pub fn paint_obj_toggled(&mut self, _state: bool) {}
    pub fn paint_list_edit_finished(&mut self) {}
    pub fn show_rubber_band_area(&mut self) {}
    pub fn button_pressed(&mut self, which: i32, native: &mut dyn IsosurfaceNativeBoundary) {
        if which == 2 {
            native.fill_cache()
        } else if which < 2 {
            native.save_object(which == 1)
        }
    }
    pub fn manage_paint_object(&mut self) {}
    pub fn model_to_paint_list(&mut self) {}
    pub fn paint_list_to_string(&self) -> String {
        self.m_paint_obj_list
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
    pub fn get_parameters_from_store(&mut self, _threshold_only: bool, _set_controls: bool) -> i32 {
        0
    }
    pub fn set_parameters_in_store(&mut self) {}
    pub fn file_value_of_threshold(&self) -> f32 {
        self.m_threshold
    }
    pub fn threshold_from_file_value(&self, value: f32) -> f32 {
        value
    }
    pub fn find_closest_z(&self, iz: i32, listz: &[i32]) -> (i32, i32) {
        let mut best = i32::MAX;
        let mut other = i32::MAX;
        for &z in listz {
            if (z - iz).abs() < (best - iz).abs() {
                other = best;
                best = z
            } else if (z - iz).abs() < (other - iz).abs() {
                other = z
            }
        }
        (best, other)
    }
    pub fn show_defined_area(&mut self, _x0: f32, _x1: f32, _y0: f32, _y1: f32, _draw: bool) {}
    pub fn top_change_event(&mut self) {}
    /// `ImodvIsosurface::setFontDependentWidths`; Qt font/layout adjustment is
    /// performed by the native dialog boundary.
    pub fn set_font_dependent_widths(&mut self) {}
    pub fn top_close_event(&mut self) {}
    pub fn key_press_event(&mut self, key: i32, hot_key: i32) {
        if key == hot_key {
            self.m_ctrl_pressed = true
        }
    }
    pub fn key_release_event(&mut self, key: i32, hot_key: i32) {
        if key == hot_key {
            self.m_ctrl_pressed = false
        }
    }
    pub fn dump_volume(&self, _filename: &str, _binned: bool) {}
}

/// `imodvIsosurfaceUpdate`.
pub fn imodv_isosurface_update(
    iso: &mut ImodvIsosurface,
    native: &mut dyn IsosurfaceNativeBoundary,
    data: &ImodvIsosurfaceData,
    hist: &mut HistWidget,
    draw_flags: i32,
) -> bool {
    if draw_flags != 0 {
        iso.update_coords(native, data.flags & IIS_LINK_XYZ != 0);
        iso.set_bounding_box(native);
        if data.flags & IIS_CENTER_VOLUME != 0 {
            iso.set_view_center(native)
        }
        iso.fill_and_process_vols(native, data, hist, PERCENTILE, false, -1.);
        iso.set_iso_obj(native, true);
        true
    } else if iso.fill_paint_vol() {
        iso.paint_mesh();
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct N;
    impl IsosurfaceNativeBoundary for N {
        fn open_dialog(&mut self) -> bool {
            true
        }
        fn raise_dialog(&mut self) {}
        fn close_dialog(&mut self) {}
        fn image_size(&self) -> [i32; 3] {
            [4, 4, 4]
        }
        fn cursor_xyz(&self) -> [i32; 3] {
            [2, 2, 2]
        }
        fn time_index(&self) -> i32 {
            0
        }
        fn time_count(&self) -> i32 {
            0
        }
        fn voxel(&self, x: i32, y: i32, z: i32) -> u8 {
            (x + y + z) as u8
        }
        fn set_cursor_xyz(&mut self, _: [i32; 3]) {}
        fn set_bounding_box(&mut self, _: [i32; 3], _: [i32; 3]) {}
        fn set_view_center(&mut self, _: [i32; 3]) {}
        fn set_iso_mesh(&mut self, _: &[u8], _: [i32; 3], _: f32, _: i32) {}
        fn redraw(&mut self) {}
        fn save_object(&mut self, _: bool) {}
        fn fill_cache(&mut self) {}
    }
    #[test]
    fn binned_volume_is_source_average() {
        let mut i = ImodvIsosurface::default();
        i.m_box_size = [4, 4, 4];
        i.binning = 2;
        i.m_volume = (0..64).collect();
        i.fill_bin_volume();
        assert_eq!(i.m_bin_box_size, [2, 2, 2]);
        assert_eq!(i.m_bin_volume[0], 10);
    }
    #[test]
    fn outer_limit_propagates_from_extreme() {
        let mut i = ImodvIsosurface::default();
        i.m_bin_box_size = [3, 3, 3];
        i.m_bin_volume = vec![0; 27];
        i.m_bin_volume[13] = 255;
        i.m_threshold = 100.;
        i.m_median = 50.;
        i.m_outer_limit = 200;
        i.m_vol_min = 0;
        i.m_vol_max = 255;
        i.remove_outer_pixels();
        assert!(i.m_bin_volume.iter().all(|&v| v < 200));
    }
}
