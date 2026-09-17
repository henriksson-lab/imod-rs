//! Translation of `IMOD/3dmod/isosurface.cpp` and `isosurface.h`.
//!
//! The numerical volume preparation code is native Rust.  Qt dialog lifetime,
//! model/mesh ownership, image-cache access, marching cubes, and OpenGL redraw
//! are deliberately explicit boundaries; they are not replaced by a second UI.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MrcHeader, mrc_head_new, mrc_head_write, mrc_write_idata,
};
use crate::imod::libimod::istore::{
    GEN_STORE_BYTE, GEN_STORE_COLOR, GEN_STORE_FLOAT, GEN_STORE_INT, GEN_STORE_NOINDEX,
    GEN_STORE_SHORT, GEN_STORE_TRANS, Istore, StoreUnion, istore_add_one_index_item, istore_insert,
};
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
pub const IIS_X_COORD: i32 = 0;
pub const IIS_Y_COORD: i32 = 1;
pub const IIS_Z_COORD: i32 = 2;
pub const IIS_X_SIZE: i32 = 3;
pub const IIS_Y_SIZE: i32 = 4;
pub const IIS_Z_SIZE: i32 = 5;
/// `ISO_STORE_*` and their packed-union offsets from `include/istore.h`.
pub const ISO_STORE_PARAMS: i16 = 22;
pub const ISO_STORE_THRESH: i16 = 23;
pub const ISO_STORE_CAP: u16 = 1 << 5;
pub const ISO_STORE_DELETE: u16 = 1 << 6;
pub const ISO_STORE_HAS_OUTER: u16 = 1 << 7;
pub const ISO_STORE_BIN_IND: usize = 0;
pub const ISO_STORE_SMOOTH_IND: usize = 1;
pub const ISO_STORE_OUTER_IND: usize = 2;
pub const ISO_STORE_SIGMA_IND: usize = 0;
pub const ISO_STORE_MINSIZE_IND: usize = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IsoPoint3D {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

/// The four source contours constructed by `ImodvIsosurface::setBoundingObj`.
/// The first is the lower Z face and carries `ICONT_DRAW_ALLZ` in the model
/// host; the remaining three are the upper face and the two side loops.
pub fn isosurface_bounding_box_contours(origin: [i32; 3], ends: [i32; 3]) -> [[IsoPoint3D; 4]; 4] {
    let [x0, y0, z0] = origin;
    let [x1, y1, z1] = ends;
    let corners = [
        IsoPoint3D {
            ix: x0,
            iy: y0,
            iz: z0,
        },
        IsoPoint3D {
            ix: x1,
            iy: y0,
            iz: z0,
        },
        IsoPoint3D {
            ix: x1,
            iy: y1,
            iz: z0,
        },
        IsoPoint3D {
            ix: x0,
            iy: y1,
            iz: z0,
        },
        IsoPoint3D {
            ix: x0,
            iy: y0,
            iz: z1,
        },
        IsoPoint3D {
            ix: x1,
            iy: y0,
            iz: z1,
        },
        IsoPoint3D {
            ix: x1,
            iy: y1,
            iz: z1,
        },
        IsoPoint3D {
            ix: x0,
            iy: y1,
            iz: z1,
        },
    ];
    [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[3], corners[2], corners[6], corners[7]],
    ]
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

/// Model-derived inputs gathered by the host for `fillPaintVol`.  Coordinates
/// are local to the current isosurface box and color indices are one-based,
/// matching the source's `mPaintPoints`/`mColorList` vectors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IsosurfacePaintInput {
    pub colors: Vec<IsoColor>,
    pub points: Vec<IsoPaintPoint>,
    pub zscale: f32,
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

/// `setOrClearFlags` as used by the native isosurface toggle slots.
pub fn set_isosurface_flag(data: &mut ImodvIsosurfaceData, flag: u32, state: bool) {
    if state {
        data.flags |= flag;
    } else {
        data.flags &= !flag;
    }
}

/// GUI-neutral portions of the native isosurface dialog slots.
pub fn set_isosurface_view_iso(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_VIEW_ISOSURFACE, state);
}
pub fn set_isosurface_view_model(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_VIEW_USER_MODEL, state);
}
pub fn set_isosurface_view_box(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_VIEW_BOX, state);
}
pub fn set_isosurface_center_volume(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_CENTER_VOLUME, state);
}
pub fn set_isosurface_delete_pieces(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_DELETE_PIECES, state);
}
pub fn set_isosurface_link_xyz(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_LINK_XYZ, state);
}
pub fn set_isosurface_close_faces(data: &mut ImodvIsosurfaceData, state: bool) {
    set_isosurface_flag(data, IIS_CLOSE_FACES, state);
}
/// Native `paintObjToggled`, with Rust-owned paint-volume allocation.
pub fn set_isosurface_paint_object(
    iso: &mut ImodvIsosurface,
    data: &mut ImodvIsosurfaceData,
    state: bool,
) -> bool {
    if !state {
        set_isosurface_flag(data, IIS_PAINT_OBJECT, false);
        iso.m_paint_vol.clear();
        return true;
    }
    let length = iso.m_box_size.iter().try_fold(1_usize, |count, &size| {
        count.checked_mul(size.max(0) as usize)
    });
    let Some(length) = length else {
        set_isosurface_flag(data, IIS_PAINT_OBJECT, false);
        return false;
    };
    let mut volume = Vec::new();
    if volume.try_reserve_exact(length).is_err() {
        set_isosurface_flag(data, IIS_PAINT_OBJECT, false);
        return false;
    }
    volume.resize(length, 0);
    iso.m_paint_vol = volume;
    set_isosurface_flag(data, IIS_PAINT_OBJECT, true);
    true
}
pub fn set_isosurface_iterations(data: &mut ImodvIsosurfaceData, iterations: i32) {
    data.it_num = iterations;
}
pub fn set_isosurface_kernel_sigma(data: &mut ImodvIsosurfaceData, sigma: f64) {
    data.kernel_sigma = sigma as f32;
}
pub fn set_isosurface_min_triangles(data: &mut ImodvIsosurfaceData, triangles: i32) {
    data.min_t_num = triangles;
    set_isosurface_delete_pieces(data, true);
}
pub fn set_isosurface_mask(data: &mut ImodvIsosurfaceData, mask: i32) {
    data.mask_type = mask;
}

/// State-update half of native `sliderMoved`.  Returns whether a linked XYZ
/// cursor must be updated by the host.
pub fn isosurface_slider_move(iso: &mut ImodvIsosurface, which: i32, value: i32) -> bool {
    match which {
        IIS_X_COORD => iso.m_local_x = value - 1,
        IIS_Y_COORD => iso.m_local_y = value - 1,
        IIS_Z_COORD => iso.m_local_z = value - 1,
        IIS_X_SIZE => iso.m_box_size[0] = value,
        IIS_Y_SIZE => iso.m_box_size[1] = value,
        IIS_Z_SIZE => iso.m_box_size[2] = value,
        _ => return false,
    }
    which <= IIS_Z_COORD
}

/// Pure `findClosestZ`: candidates carry whether their Z plane has a contour
/// with at least three points.  The first result is closest with ties below;
/// the second is the closest eligible plane on the opposite side.
pub fn isosurface_find_closest_z(
    local_z: i32,
    box_origin_z: i32,
    candidates: &[(i32, bool)],
) -> (i32, i32) {
    let z = local_z + box_origin_z;
    let mut below: Option<(i32, i32)> = None;
    let mut above: Option<(i32, i32)> = None;
    for &(candidate, eligible) in candidates {
        if !eligible {
            continue;
        }
        let delta = z - candidate;
        if delta >= 0 && below.map_or(true, |(distance, _)| delta < distance) {
            below = Some((delta, candidate));
        } else if delta < 0 && above.map_or(true, |(distance, _)| -delta < distance) {
            above = Some((-delta, candidate));
        }
    }
    match (below, above) {
        (Some((bd, bz)), Some((ad, az))) if bd <= ad => (bz, az),
        (Some((_, bz)), Some((_, az))) => (az, bz),
        (Some((_, bz)), None) => (bz, i32::MAX),
        (None, Some((_, az))) => (az, i32::MAX),
        (None, None) => (i32::MAX, i32::MAX),
    }
}

/// Source `parselist`/`listToString` convention for paint-object IDs: text is
/// one-based, resident object indices are zero-based.
pub fn parse_isosurface_paint_list(text: &str) -> Vec<i32> {
    let mut out = Vec::new();
    for token in text.split(',') {
        let values: Vec<i32> = token
            .trim()
            .splitn(2, '-')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        match values.as_slice() {
            [value] if *value > 0 => out.push(value - 1),
            [first, last] if *first > 0 && *last > 0 => {
                let step = if first <= last { 1 } else { -1 };
                let mut value = *first;
                loop {
                    out.push(value - 1);
                    if value == *last {
                        break;
                    }
                    value += step;
                }
            }
            _ => {}
        }
    }
    out
}

pub fn isosurface_paint_list_to_string(indices: &[i32]) -> String {
    let values: Vec<i32> = indices.iter().map(|index| index + 1).collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < values.len() {
        let mut end = start;
        while end + 1 < values.len() && values[end + 1] == values[end] + 1 {
            end += 1;
        }
        chunks.push(if end == start {
            values[start].to_string()
        } else {
            format!("{}-{}", values[start], values[end])
        });
        start = end + 1;
    }
    chunks.join(",")
}

/// `managePaintObject`'s model-independent filtering rule.  Each entry in
/// `scattered` corresponds to a zero-based model object index.
pub fn filter_isosurface_paint_list(indices: &mut Vec<i32>, scattered: &[bool]) -> bool {
    let old_len = indices.len();
    indices.retain(|&index| index >= 0 && scattered.get(index as usize).copied().unwrap_or(false));
    indices.len() != old_len
}

/// Pure state portion of native `histChanged`.  Returns whether the volume
/// must be refilled because the outer-limit mask changed.
pub fn isosurface_hist_change(iso: &mut ImodvIsosurface, which: i32, value: i32) -> bool {
    let mut refill = false;
    if which == 0 {
        iso.m_threshold = value as f32 + 0.5;
        if iso.m_outer_limit >= 0 {
            if iso.m_threshold < iso.m_median {
                iso.m_outer_limit = if iso.m_outer_limit >= iso.m_median as i32 {
                    -1
                } else {
                    iso.m_outer_limit.min(value)
                };
            } else {
                iso.m_outer_limit = if iso.m_outer_limit < iso.m_median as i32 {
                    -1
                } else {
                    iso.m_outer_limit.max(value + 1)
                };
            }
            if iso.m_outer_limit <= iso.m_vol_min || iso.m_outer_limit >= iso.m_vol_max {
                iso.m_outer_limit = -1;
            }
            refill = true;
        }
    } else {
        iso.m_outer_limit = if iso.m_threshold < iso.m_median {
            value.min(iso.m_threshold as i32)
        } else {
            value.max(iso.m_threshold as i32 + 1)
        };
        if iso.m_outer_limit <= iso.m_vol_min || iso.m_outer_limit >= iso.m_vol_max {
            iso.m_outer_limit = -1;
        }
        refill = true;
    }
    refill
}

/// Native `ISO_STORE_PARAMS` fields unpacked from their deliberately compact
/// `Istore` representation.  Thresholds are already converted to the loaded
/// image's 0..255 display range by the callback to [`read_isosurface_store`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IsosurfaceStoredParameters {
    pub binning_num: i32,
    pub iteration_num: i32,
    pub kernel_sigma: f32,
    pub min_triangles: i32,
    pub outer_limit: Option<i32>,
    pub delete_pieces: bool,
    pub close_faces: bool,
    pub thresholds: Vec<Option<f32>>,
}

/// Read the isosurface records from a model general store.  `time_count` is
/// `mVi->numTimes`; a zero count has the native single-stack convention.  The
/// callback is `thresholdFromFileValue` and keeps image/window conversion at
/// the image boundary instead of inventing a GUI image type here.
pub fn read_isosurface_store<F>(
    store: &[Istore],
    time_count: usize,
    mut threshold_from_file: F,
) -> Option<IsosurfaceStoredParameters>
where
    F: FnMut(usize, f32) -> f32,
{
    // Native scans only the trailing no-index section; it deliberately stops
    // at the first indexed item rather than finding an older parameter record.
    let mut param = None;
    for item in store.iter().rev() {
        if item.flags & GEN_STORE_NOINDEX == 0 {
            break;
        }
        if item.type_ == ISO_STORE_PARAMS {
            param = Some(*item);
            break;
        }
    }
    let param = param?;
    let index = param.index.b();
    let value = param.value.us();
    let number_of_times = time_count.max(1);
    let mut result = IsosurfaceStoredParameters {
        binning_num: index[ISO_STORE_BIN_IND] as i32,
        iteration_num: index[ISO_STORE_SMOOTH_IND] as i32,
        kernel_sigma: value[ISO_STORE_SIGMA_IND] as f32 / 10_000.,
        min_triangles: value[ISO_STORE_MINSIZE_IND] as i32,
        outer_limit: (param.flags & ISO_STORE_HAS_OUTER != 0)
            .then_some(index[ISO_STORE_OUTER_IND] as i32),
        delete_pieces: param.flags & ISO_STORE_DELETE != 0,
        close_faces: param.flags & ISO_STORE_CAP != 0,
        thresholds: vec![None; number_of_times],
    };
    for time in 0..number_of_times {
        let store_index = time as i32 + if time_count > 0 { 1 } else { 0 };
        if let Some(item) = store.iter().find(|item| {
            item.type_ == ISO_STORE_THRESH
                && item.flags & GEN_STORE_NOINDEX == 0
                && item.index.i() == store_index
        }) {
            result.thresholds[time] = Some(threshold_from_file(time, item.value.f()));
        }
    }
    Some(result)
}

/// Apply model-store state to the resident controller and its file-static
/// dialog data.  It is the non-Qt half of `getParametersFromStore` once a host
/// has handled any confirmation prompt.  Returns whether anything differed.
pub fn apply_isosurface_store(
    stored: &IsosurfaceStoredParameters,
    iso: &mut ImodvIsosurface,
    data: &mut ImodvIsosurfaceData,
) -> bool {
    let outer = stored.outer_limit.unwrap_or(-1);
    let differs = outer != iso.m_outer_limit
        || stored.binning_num != data.binning_num
        || stored.iteration_num != data.it_num
        || (stored.kernel_sigma - data.kernel_sigma).abs() > f32::EPSILON
        || stored.min_triangles != data.min_t_num
        || stored.delete_pieces != (data.flags & IIS_DELETE_PIECES != 0)
        || stored.close_faces != (data.flags & IIS_CLOSE_FACES != 0)
        || stored
            .thresholds
            .iter()
            .enumerate()
            .any(|(time, threshold)| {
                threshold.is_some_and(|value| {
                    iso.m_stack_thresholds
                        .get(time)
                        .is_some_and(|current| *current >= 0. && (value - *current).abs() > 0.01)
                })
            });
    if !differs {
        return false;
    }
    iso.m_outer_limit = outer;
    data.binning_num = stored.binning_num;
    data.it_num = stored.iteration_num;
    data.kernel_sigma = stored.kernel_sigma;
    data.min_t_num = stored.min_triangles;
    set_isosurface_delete_pieces(data, stored.delete_pieces);
    set_isosurface_close_faces(data, stored.close_faces);
    if iso.m_stack_thresholds.len() < stored.thresholds.len() {
        iso.m_stack_thresholds.resize(stored.thresholds.len(), -1.);
    }
    for (time, threshold) in stored.thresholds.iter().enumerate() {
        if let Some(value) = threshold {
            iso.m_stack_thresholds[time] = *value;
        }
    }
    true
}

/// Write/replace the native `ISO_STORE_PARAMS` record and the threshold record
/// for `current_time`, exactly as `setParametersInStore` does.  `file_threshold`
/// must be produced by [`isosurface_file_value_of_threshold`] at the image
/// boundary.
pub fn write_isosurface_store(
    store: &mut Vec<Istore>,
    iso: &ImodvIsosurface,
    data: &ImodvIsosurfaceData,
    current_time: i32,
    file_threshold: f32,
) {
    let mut index = [0_u8; 4];
    index[ISO_STORE_BIN_IND] = data.binning_num as u8;
    index[ISO_STORE_SMOOTH_IND] = data.it_num as u8;
    let mut flags = GEN_STORE_BYTE | (GEN_STORE_SHORT << 2) | GEN_STORE_NOINDEX;
    if data.flags & IIS_CLOSE_FACES != 0 {
        flags |= ISO_STORE_CAP;
    }
    if data.flags & IIS_DELETE_PIECES != 0 {
        flags |= ISO_STORE_DELETE;
    }
    if iso.m_outer_limit >= iso.m_vol_min && iso.m_outer_limit <= iso.m_vol_max {
        flags |= ISO_STORE_HAS_OUTER;
        index[ISO_STORE_OUTER_IND] = iso.m_outer_limit as u8;
    }
    let parameter = Istore {
        type_: ISO_STORE_PARAMS,
        flags,
        index: StoreUnion::from_b(index),
        value: StoreUnion::from_us([
            (10_000. * data.kernel_sigma).round() as u16,
            data.min_t_num as u16,
        ]),
    };
    let mut replaced = false;
    for item in store.iter_mut().rev() {
        if item.flags & GEN_STORE_NOINDEX == 0 {
            break;
        }
        if item.type_ == ISO_STORE_PARAMS {
            *item = parameter;
            replaced = true;
            break;
        }
    }
    if !replaced {
        istore_insert(store, parameter);
    }
    istore_add_one_index_item(
        store,
        Istore {
            type_: ISO_STORE_THRESH,
            flags: GEN_STORE_INT | (GEN_STORE_FLOAT << 2),
            index: StoreUnion::from_i(current_time),
            value: StoreUnion::from_f(file_threshold),
        },
    );
}

/// Native conversion between displayed 0..255 threshold values and file
/// values, including the ushort display window.
pub fn isosurface_file_value_of_threshold(
    threshold: f32,
    ushort: bool,
    range_low: i32,
    range_high: i32,
    smin: f32,
    smax: f32,
) -> f32 {
    if ushort {
        let display = threshold * (range_high - range_low) as f32 / 255. + range_low as f32;
        display * (smax - smin) / 65535. + smin
    } else {
        threshold * (smax - smin) / 255. + smin
    }
}
pub fn isosurface_threshold_from_file_value(
    value: f32,
    ushort: bool,
    range_low: i32,
    range_high: i32,
    smin: f32,
    smax: f32,
) -> f32 {
    let threshold = if ushort {
        let uint = 65535. * (value - smin) / (smax - smin);
        255. * (uint - range_low as f32) / (range_high - range_low) as f32
    } else {
        255. * (value - smin) / (smax - smin)
    };
    threshold.clamp(0., 255.)
}

/// Select the exact byte stack native `dumpVolume` writes.  A missing or
/// short resident buffer is rejected instead of exposing an incomplete MRC
/// slice list to the file-I/O boundary.
pub fn isosurface_dump_volume_data(
    iso: &ImodvIsosurface,
    binned: bool,
) -> Option<(&[u8], [i32; 3])> {
    let size = if binned {
        iso.m_bin_box_size
    } else {
        iso.m_box_size
    };
    let length = size.iter().try_fold(1_usize, |total, &axis| {
        total.checked_mul(axis.max(0) as usize)
    })?;
    let data = if binned {
        &iso.m_bin_volume
    } else {
        &iso.m_volume
    };
    if data.len() < length {
        None
    } else {
        Some((&data[..length], size))
    }
}

/// The component-selection core of native `filterMesh`.  `triangles` is the
/// mesh list after its leading `-25` opcode; output uses that same plain
/// triangle-index form.  Native `Surface_Pieces` orders components smallest
/// first, then `filterMesh` keeps only trailing pieces with `area > min`.
pub fn filter_isosurface_triangles(
    vertices: &[crate::imod::libimod::imodel::Ipoint],
    triangles: &[i32],
    min_triangles: i32,
) -> Vec<i32> {
    if triangles.len() % 3 != 0 || triangles.is_empty() {
        return Vec::new();
    }
    let mut sorted = vec![0; triangles.len()];
    let pieces = crate::imod::three_dmod::surfpieces::SurfacePieces::new(
        vertices,
        triangles,
        (triangles.len() / 3) as i32,
        &mut sorted,
    );
    let kept = pieces
        .pieces
        .iter()
        .rev()
        .take_while(|piece| piece.area > min_triangles as f32)
        .map(|piece| piece.t_list.len())
        .sum::<usize>();
    if kept == 0 {
        Vec::new()
    } else {
        sorted[sorted.len() - 3 * kept..].to_vec()
    }
}

/// Rasterize the source paint-point spheres into `mPaintVol`.  Points are in
/// local box coordinates and color indices are one-based, matching native
/// `fillPaintVol`; overlapping colors select the closest normalized sphere.
pub fn rasterize_isosurface_paint_volume(iso: &mut ImodvIsosurface, zscale: f32) -> bool {
    let size = iso.m_box_size;
    let Some(length) = size.iter().try_fold(1_usize, |total, &axis| {
        total.checked_mul(axis.max(0) as usize)
    }) else {
        return false;
    };
    if iso.m_paint_vol.len() != length {
        iso.m_paint_vol.resize(length, 0);
    }
    iso.m_paint_vol.fill(0);
    for z in 0..size[2].max(0) {
        for y in 0..size[1].max(0) {
            for x in 0..size[0].max(0) {
                let mut candidates: Vec<(usize, f32)> = Vec::new();
                for (index, point) in iso.m_paint_points.iter().enumerate() {
                    if point.size <= 0. || point.color_ind <= 0 {
                        continue;
                    }
                    let dx = x as f32 + 0.5 - point.x;
                    let dy = y as f32 + 0.5 - point.y;
                    let dz = zscale * (z as f32 + 0.5 - point.z);
                    let distance = (dx * dx + dy * dy + dz * dz) / (point.size * point.size);
                    if distance <= 1. {
                        candidates.push((index, distance));
                    }
                }
                let mut excluded = vec![false; candidates.len()];
                // Native sorts a cluster from smaller to larger and excludes
                // a larger sphere when it wholly contains the smaller one.
                for (left, &(li, _)) in candidates.iter().enumerate() {
                    for (right, &(ri, _)) in candidates.iter().enumerate() {
                        if left == right {
                            continue;
                        }
                        let small = &iso.m_paint_points[li];
                        let large = &iso.m_paint_points[ri];
                        if small.size > large.size || small.color_ind == large.color_ind {
                            continue;
                        }
                        let dx = small.x - large.x;
                        let dy = small.y - large.y;
                        let dz = small.z - large.z;
                        if dx * dx + dy * dy + dz * dz <= (large.size - small.size).powi(2) {
                            excluded[right] = true;
                        }
                    }
                }
                let color = candidates
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !excluded[*index])
                    .min_by(|(_, left), (_, right)| left.1.total_cmp(&right.1))
                    .map_or(0, |(_, &(index, _))| {
                        iso.m_paint_points[index].color_ind.min(u8::MAX as i32) as u8
                    });
                iso.m_paint_vol[(x + y * size[0] + z * size[0] * size[1]) as usize] = color;
            }
        }
    }
    iso.m_paint_size = size;
    true
}

/// Build the mesh-store records made by native `paintMesh`.  `list` retains
/// the marching-cubes leading opcode, so painted vertex records start at list
/// index one.  Invalid mesh indices or vertices outside the paint box are
/// ignored rather than indexing outside the native paint volume.
pub fn isosurface_paint_mesh_store(
    vertices: &[crate::imod::libimod::imodel::Ipoint],
    list: &[i32],
    box_origin: [i32; 3],
    box_size: [i32; 3],
    paint_volume: &[u8],
    colors: &[IsoColor],
) -> Vec<Istore> {
    let Some(volume_len) = box_size.iter().try_fold(1_usize, |total, &axis| {
        total.checked_mul(axis.max(0) as usize)
    }) else {
        return Vec::new();
    };
    if list.len() < 2 || paint_volume.len() < volume_len || colors.is_empty() {
        return Vec::new();
    }
    let color_at = |list_pos: usize| -> Option<usize> {
        let &vertex_index = list.get(list_pos)?;
        let point = vertices.get(usize::try_from(vertex_index).ok()?)?;
        let x = (point.x as i32).checked_sub(box_origin[0])?;
        let y = (point.y as i32).checked_sub(box_origin[1])?;
        let z = (point.z as i32).checked_sub(box_origin[2])?;
        if x < 0 || x >= box_size[0] || y < 0 || y >= box_size[1] || z < 0 || z >= box_size[2] {
            return None;
        }
        let color =
            *paint_volume.get((x + y * box_size[0] + z * box_size[0] * box_size[1]) as usize)?;
        if color == 0 {
            return None;
        }
        let index = (color - 1) as usize;
        (index < colors.len()).then_some(index)
    };
    let color_store = |list_pos: usize, color: IsoColor| Istore {
        type_: GEN_STORE_COLOR,
        flags: GEN_STORE_BYTE << 2,
        index: StoreUnion::from_i(list_pos as i32),
        value: StoreUnion::from_b([color.r, color.g, color.b, u8::from(color.trans != 0)]),
    };
    let trans_store = |list_pos: usize, trans: i32| Istore {
        type_: GEN_STORE_TRANS,
        flags: 0,
        index: StoreUnion::from_i(list_pos as i32),
        value: StoreUnion::from_i(trans),
    };
    let any_transparent = colors.iter().any(|color| color.trans != 0);
    let mut stores = Vec::new();
    if !any_transparent {
        for list_pos in 1..list.len() {
            if let Some(color_index) = color_at(list_pos) {
                stores.push(color_store(list_pos, colors[color_index]));
            }
        }
        return stores;
    }
    // The C code processes triangle triplets when any paint color is
    // transparent, then supplies transparency 1 to unpainted vertices of a
    // partially transparent triangle.  This prevents interpolation from
    // turning its exposed faces opaque.
    for first in (1..list.len().saturating_sub(2)).step_by(3) {
        let mut transparent = Vec::new();
        for list_pos in first..first + 3 {
            if let Some(color_index) = color_at(list_pos) {
                let color = colors[color_index];
                stores.push(color_store(list_pos, color));
                if color.trans != 0 {
                    stores.push(trans_store(list_pos, color.trans));
                    transparent.push(list_pos);
                }
            }
        }
        if !transparent.is_empty() && transparent.len() < 3 {
            for list_pos in first..first + 3 {
                if !transparent.contains(&list_pos) {
                    stores.push(trans_store(list_pos, 1));
                }
            }
        }
    }
    stores
}

/// Native `ImodvIsosurface::dumpVolume`: write the selected resident volume
/// as an unsigned-byte MRC stack.  The return value follows the MRC helpers
/// (zero on success); an unavailable buffer or failed `fopen` is an error.
pub fn dump_isosurface_volume(
    iso: &ImodvIsosurface,
    filename: impl AsRef<std::path::Path>,
    binned: bool,
) -> i32 {
    let Some((data, [nx, ny, nz])) = isosurface_dump_volume_data(iso, binned) else {
        return 1;
    };
    let Some(mut output) = ImodFile::open(filename, "wb") else {
        return 1;
    };
    let plane = match (nx as usize).checked_mul(ny as usize) {
        Some(length) if length > 0 => length,
        _ => return 1,
    };
    let slices: Vec<&[u8]> = data.chunks_exact(plane).take(nz as usize).collect();
    if slices.len() != nz as usize {
        return 1;
    }
    let mut header = MrcHeader::default();
    mrc_head_new(&mut header, nx, ny, nz, MRC_MODE_BYTE);
    header.amin = 0.;
    header.amax = 255.;
    header.amean = 128.;
    if mrc_head_write(&mut output, &mut header) != 0 {
        return 1;
    }
    i32::from(mrc_write_idata(&mut output, &mut header, &slices) != 0)
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
    /// `setBoundingObj`, which builds the box extra-object in the model host.
    fn set_bounding_object(&mut self, _origin: [i32; 3], _ends: [i32; 3]) {}
    /// Full `setBoundingObj` topology.  Limits-only hosts retain the former
    /// callback while the rewritten renderer can consume source-identical
    /// contour ordering directly.
    fn set_bounding_object_contours(
        &mut self,
        origin: [i32; 3],
        ends: [i32; 3],
        _contours: [[IsoPoint3D; 4]; 4],
    ) {
        self.set_bounding_object(origin, ends);
    }
    fn set_view_center(&mut self, xyz: [i32; 3]);
    fn set_iso_mesh(&mut self, volume: &[u8], size: [i32; 3], threshold: f32, binning: i32);
    /// `ImodvIsosurface::smoothMesh` over the host-owned generated mesh.
    /// Hosts can use `isothread::smooth_vertex_positions` on their interleaved
    /// marching-cubes vertex/normal buffer.
    fn smooth_iso_mesh(&mut self, _iterations: i32) {}
    /// `filterMesh`: duplicate/filter the host-owned marching-cubes mesh.
    /// The host can apply [`filter_isosurface_triangles`] to its triangle list.
    fn filter_iso_mesh(&mut self, _enabled: bool, _minimum_triangles: i32) {}
    /// Clear the displayed extra object and replace it with a duplicate of
    /// the host-owned filtered mesh, as `iterNumChanged` does before smoothing.
    fn replace_iso_mesh_with_filtered(&mut self) {}
    /// `setParametersInStore` at the model-store boundary.
    fn persist_isosurface_parameters(&mut self) {}
    /// Set visibility of the extra object that holds the isosurface mesh.
    fn set_isosurface_visible(&mut self, _visible: bool) {}
    /// `ImodvApp::drawExtraOnly` from `viewModelToggled`.
    fn set_user_model_visible(&mut self, _visible: bool) {}
    /// Toggle the bounding-box extra object's off flag.
    fn set_isosurface_box_visible(&mut self, _visible: bool) {}
    /// `paintMesh`: attach the translated colour/transparency store generated
    /// from this paint volume to the host-owned isosurface mesh.
    fn paint_iso_mesh(
        &mut self,
        _paint_volume: &[u8],
        _paint_size: [i32; 3],
        _colors: &[IsoColor],
    ) {
    }
    /// Model/store traversal half of `fillPaintVol`.  `None` means no paint
    /// object is active or no eligible sized points were found.
    fn collect_isosurface_paint_points(
        &mut self,
        _origin: [i32; 3],
        _ends: [i32; 3],
        _object_list: &[i32],
    ) -> Option<IsosurfacePaintInput> {
        None
    }
    /// Zero-based scattered model-object indices marked with
    /// `IOBJ_EXFLAG_ISO_PAINT` for `modelToPaintList`.
    fn marked_isosurface_paint_objects(&mut self) -> Vec<i32> {
        Vec::new()
    }
    /// Per-object scattered qualification for `managePaintObject`.
    fn isosurface_scattered_objects(&mut self) -> Vec<bool> {
        Vec::new()
    }
    fn set_isosurface_paint_list_enabled(&mut self, _enabled: bool) {}
    /// Image coordinates from the highest-priority Zap or Slicer rubber band.
    fn isosurface_rubber_band_area(&mut self) -> Option<(f32, f32, f32, f32)> {
        None
    }
    fn redraw(&mut self);
    /// The source's final `imodDraw(... IMOD_DRAW_XYZ | IMOD_DRAW_SKIPMODV)`.
    fn redraw_image(&mut self) {}
    fn save_object(&mut self, sort_surfaces: bool);
    fn fill_cache(&mut self);
    /// `ImodPrefs->getRoundedStyle()`.
    fn rounded_style(&self) -> bool {
        false
    }
    /// `DialogFrame::changeEvent`.
    fn dialog_change_event(&mut self) {}
    /// `ivwCheckAndSetMacMenu`.
    fn check_and_set_mac_menu(&mut self) {}
    /// `diaSetButtonWidth` for a source-named dialog button.
    fn set_button_width(&mut self, _button: &str, _rounded: bool, _factor: f32, _label: &str) {}
    /// `diaLimitEditStretch` for the paint-object list edit/layout.
    fn limit_edit_stretch(&mut self, _edit: &str, _layout: &str, _stretch: i32) {}
    /// `imodvDialogManager.remove(sTopWin)`, `iisData.dia = NULL`, and
    /// `sTopWin = NULL`.
    fn remove_isosurface_dialog(&mut self) {}
    /// `QCloseEvent::accept`.
    fn accept_close_event(&mut self) {}
    /// `ivwFreeExtraObject`.
    fn free_extra_object(&mut self, _object_number: i32) {}
    /// `imodvObjedNewView`.
    fn object_editor_new_view(&mut self) {}
    /// `iisData.a->drawExtraOnly = 0`.
    fn enable_user_model_drawing(&mut self) {}
    /// `imodDraw(mVi, IMOD_DRAW_MOD)`.
    fn draw_model(&mut self) {}
    /// `imodMeshDelete(mOrigMesh)` and, when present,
    /// `imodMeshDelete(mFilteredMesh)` plus `delete mSurfPieces`.
    fn release_isosurface_meshes(&mut self) {}
}

/// The portion of `QEvent` relevant to `ImodvIsosurface::topChangeEvent`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IsosurfaceChangeEvent {
    Other,
    FontChange,
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

/// Geometry selected by native `showDefinedArea`, before volume allocation,
/// mesh construction, and redraw are delegated to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsosurfaceDefinedArea {
    pub center: [i32; 2],
    pub size: [i32; 2],
    pub origin: [i32; 2],
    pub ends: [i32; 2],
}

pub fn isosurface_defined_area(
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
    image_size: [i32; 3],
    maximal_size: [i32; 3],
) -> IsosurfaceDefinedArea {
    let size = [
        (x1 - x0).min(maximal_size[0] as f32) as i32,
        (y1 - y0).min(maximal_size[1] as f32) as i32,
    ];
    let center = [((x0 + x1) / 2.) as i32, ((y0 + y1) / 2.) as i32];
    let (xstart, xend) = set_coord_limits(center[0], image_size[0], size[0]);
    let (ystart, yend) = set_coord_limits(center[1], image_size[1], size[1]);
    IsosurfaceDefinedArea {
        center,
        size,
        origin: [xstart, ystart],
        ends: [xend, yend],
    }
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
    /// `ImodvIsosurface::getBinning` (`isosurface.cpp:1970`).
    pub fn get_binning(&self) -> i32 {
        self.binning
    }

    /// Apply the pure geometry result from `showDefinedArea` before the host
    /// reallocates buffers, rebuilds the mesh, and redraws.
    pub fn apply_defined_area(&mut self, area: IsosurfaceDefinedArea) {
        self.m_local_x = area.center[0];
        self.m_local_y = area.center[1];
        self.m_box_size[0] = area.size[0];
        self.m_box_size[1] = area.size[1];
        self.m_box_origin[0] = area.origin[0];
        self.m_box_origin[1] = area.origin[1];
        self.m_box_ends[0] = area.ends[0];
        self.m_box_ends[1] = area.ends[1];
    }
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
        if data.flags & IIS_CENTER_VOLUME != 0 {
            iso.set_view_center(native);
        }
        iso.set_bounding_obj(native);
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
        native.set_view_center(
            [0, 1, 2].map(|axis| (self.m_box_origin[axis] + self.m_box_ends[axis]) / 2),
        );
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
            self.paint_mesh(native)
        }
    }
    /// `ImodvIsosurface::setBoundingObj`.  The host owns the extra object,
    /// while this controller supplies the source's eight-corner box limits.
    pub fn set_bounding_obj(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        native.set_bounding_object_contours(
            self.m_box_origin,
            self.m_box_ends,
            isosurface_bounding_box_contours(self.m_box_origin, self.m_box_ends),
        );
    }
    /// `ImodvIsosurface::resizeToContours`.
    pub fn resize_to_contours(
        &mut self,
        native: &mut dyn IsosurfaceNativeBoundary,
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        draw: bool,
    ) {
        self.set_bounding_box(native);
        self.set_bounding_obj(native);
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
    /// `ImodvIsosurface::smoothMesh`.  Marching-cubes mesh ownership remains
    /// at the model host, which applies `smooth_vertex_positions` to the
    /// source interleaved vertex/normal layout.
    pub fn smooth_mesh(&mut self, iterations: i32, native: &mut dyn IsosurfaceNativeBoundary) {
        if iterations != 0 {
            native.smooth_iso_mesh(iterations);
        }
    }
    /// `ImodvIsosurface::filterMesh`; mesh allocation remains in the model
    /// host, while this controller retains the source ownership state.
    pub fn filter_mesh(
        &mut self,
        state: bool,
        minimum_triangles: i32,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        self.m_made_filtered = state;
        native.filter_iso_mesh(state, minimum_triangles);
    }
    /// `ImodvIsosurface::paintMesh`; the source mesh's general-store
    /// ownership is at the model host, but all paint state remains here.
    pub fn paint_mesh(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        native.paint_iso_mesh(&self.m_paint_vol, self.m_paint_size, &self.m_color_list);
    }
    /// `ImodvIsosurface::fillPaintVol`.  The host performs model-store
    /// traversal; this method preserves the native cache comparison and runs
    /// the translated local-sphere rasterizer only when inputs changed.
    pub fn fill_paint_vol(&mut self, native: &mut dyn IsosurfaceNativeBoundary) -> bool {
        let Some(input) = native.collect_isosurface_paint_points(
            self.m_box_origin,
            self.m_box_ends,
            &self.m_paint_obj_list,
        ) else {
            let changed = !self.m_color_list.is_empty() || !self.m_paint_points.is_empty();
            self.m_color_list.clear();
            self.m_paint_points.clear();
            return changed;
        };
        if input.colors.is_empty() || input.points.is_empty() {
            let changed = !self.m_color_list.is_empty() || !self.m_paint_points.is_empty();
            self.m_color_list.clear();
            self.m_paint_points.clear();
            return changed;
        }
        if self.m_color_list == input.colors
            && self.m_paint_points == input.points
            && self.m_paint_size == self.m_box_size
        {
            return false;
        }
        self.m_color_list = input.colors;
        self.m_paint_points = input.points;
        rasterize_isosurface_paint_volume(self, input.zscale)
    }
    /// `viewIsoToggled`.
    pub fn view_iso_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_view_iso(data, state);
        native.set_isosurface_visible(state);
        native.redraw();
    }
    /// `viewModelToggled`.
    pub fn view_model_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_view_model(data, state);
        native.set_user_model_visible(state);
        native.redraw();
    }
    /// `viewBoxingToggled`.
    pub fn view_boxing_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_view_box(data, state);
        native.set_isosurface_box_visible(state);
        native.draw_model();
    }
    /// `centerVolumeToggled`.
    pub fn center_volume_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_center_volume(data, state);
        if state {
            self.set_view_center(native);
            native.redraw();
        }
    }
    /// `deletePiecesToggled`, including the native filter/rebuild sequence.
    pub fn delete_pieces_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_delete_pieces(data, state);
        native.persist_isosurface_parameters();
        self.filter_mesh(state, data.min_t_num, native);
        self.iter_num_changed(data.it_num, data, native);
    }
    /// `linkXYZToggled`: refresh paint state from the linked model and redraw
    /// the image-view model layer when enabling the link.
    pub fn link_xyz_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_link_xyz(data, state);
        if state {
            if self.fill_paint_vol(native) {
                self.paint_mesh(native);
            }
            native.draw_model();
        }
    }
    /// `closeFacesToggled`: regenerate the resident volume and mesh because
    /// boundary capping changes the marching-cubes input.
    pub fn close_faces_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_close_faces(data, state);
        self.fill_and_process_vols(native, data, hist, -1., false, -1.);
        self.set_iso_obj(native, false);
        native.redraw();
    }
    pub fn hist_changed(&mut self, which: i32, value: i32, dragging: bool) {
        self.hist_changed_with_hot(which, value, dragging, false);
    }
    pub fn hist_changed_with_hot(
        &mut self,
        which: i32,
        value: i32,
        dragging: bool,
        hot: bool,
    ) -> bool {
        if dragging && !hot {
            return false;
        }
        isosurface_hist_change(self, which, value)
    }
    /// `iterNumChanged`: restore the filtered mesh, smooth its duplicate,
    /// repaint its store, and draw the new result.
    pub fn iter_num_changed(
        &mut self,
        iterations: i32,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        data.it_num = iterations;
        native.persist_isosurface_parameters();
        native.replace_iso_mesh_with_filtered();
        self.smooth_mesh(iterations, native);
        self.paint_mesh(native);
        native.redraw();
    }
    /// `binningNumChanged`.
    pub fn binning_num_changed(
        &mut self,
        binning: i32,
        data: &mut ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        let old_binning = data.binning_num;
        data.binning_num = binning.clamp(1, MAXIMAL_BINNING);
        self.binning = data.binning_num;
        self.set_bounding_box(native);
        self.set_bounding_obj(native);
        let percentile = hist.percentile_at_point(self.m_threshold);
        let skip_fill = !(old_binning == 1
            && (data.flags & IIS_CLOSE_FACES != 0
                || data.kernel_sigma > 0.
                || (self.m_outer_limit > self.m_vol_min && self.m_outer_limit < self.m_vol_max)));
        self.fill_and_process_vols(native, data, hist, percentile, skip_fill, -1.);
        self.set_iso_obj(native, false);
        native.redraw();
    }
    /// `kernelSigmaChanged`: retain the histogram percentile through a
    /// smoothing change and skip source-image refilling only when the native
    /// binning/mask conditions permit it.
    pub fn kernel_sigma_changed(
        &mut self,
        value: f64,
        data: &mut ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_kernel_sigma(data, value);
        let percentile = hist.percentile_at_point(self.m_threshold);
        let skip_fill = !(data.binning_num == 1
            && (data.flags & IIS_CLOSE_FACES != 0
                || data.kernel_sigma > 0.
                || (self.m_outer_limit > self.m_vol_min && self.m_outer_limit < self.m_vol_max)));
        self.fill_and_process_vols(native, data, hist, percentile, skip_fill, -1.);
        self.set_iso_obj(native, false);
        native.redraw();
    }
    /// `sliderMoved`.  A normal release, or the controller's hot-slider
    /// state, rebuilds the source volume/mesh after updating a box coordinate
    /// or dimension.
    pub fn slider_moved(
        &mut self,
        which: i32,
        value: i32,
        dragging: bool,
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) -> bool {
        let linked_cursor = isosurface_slider_move(self, which, value);
        if linked_cursor && data.flags & IIS_LINK_XYZ != 0 {
            native.set_cursor_xyz([self.m_local_x, self.m_local_y, self.m_local_z]);
        }
        if dragging && !self.m_ctrl_pressed {
            return false;
        }
        self.set_bounding_box(native);
        if data.flags & IIS_CENTER_VOLUME != 0 {
            self.set_view_center(native);
        }
        self.set_bounding_obj(native);
        if self.alloc_arrays_if_needed(data.flags) {
            return false;
        }
        self.fill_and_process_vols(native, data, hist, -1., false, -1.);
        self.set_iso_obj(native, true);
        native.redraw();
        if which <= IIS_Z_COORD || data.flags & IIS_VIEW_BOX != 0 {
            native.redraw_image();
        }
        true
    }
    /// `numOfTrianglesChanged` always enables small-piece deletion after
    /// storing the new strict component-size threshold.
    pub fn num_of_triangles_changed(
        &mut self,
        triangles: i32,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        set_isosurface_min_triangles(data, triangles);
        self.delete_pieces_toggled(true, data, native);
    }
    /// `maskSelected`.
    pub fn mask_selected(&mut self, which: i32, data: &mut ImodvIsosurfaceData) {
        set_isosurface_mask(data, which);
    }
    /// `areaFromContClicked`.
    pub fn area_from_cont_clicked(
        &mut self,
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        self.resize_to_contours(native, data, hist, true);
    }
    /// `paintObjToggled`, including allocation, paint refresh, and redraw.
    pub fn paint_obj_toggled(
        &mut self,
        state: bool,
        data: &mut ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) -> bool {
        let allocated = set_isosurface_paint_object(self, data, state);
        self.manage_paint_object(data, native);
        let _ = self.fill_paint_vol(native);
        self.paint_mesh(native);
        native.redraw();
        allocated
    }
    /// `paintListEditFinished`: parse the one-based user list, retain only
    /// scattered objects, then refresh the source paint mask and mesh store.
    pub fn paint_list_edit_finished(
        &mut self,
        text: &str,
        scattered: &[bool],
        native: &mut dyn IsosurfaceNativeBoundary,
    ) -> String {
        let normalized = self.set_paint_list_for_objects(text, scattered);
        let _ = self.fill_paint_vol(native);
        self.paint_mesh(native);
        native.redraw();
        normalized
    }
    /// Data-transfer portion of native `paintListEditFinished`; model object
    /// qualification and repainting remain host/model operations.
    pub fn set_paint_list_text(&mut self, text: &str) -> String {
        self.m_paint_obj_list = parse_isosurface_paint_list(text);
        self.paint_list_to_string()
    }
    pub fn set_paint_list_for_objects(&mut self, text: &str, scattered: &[bool]) -> String {
        self.m_paint_obj_list = parse_isosurface_paint_list(text);
        filter_isosurface_paint_list(&mut self.m_paint_obj_list, scattered);
        self.paint_list_to_string()
    }
    /// `showRubberBandArea`.
    pub fn show_rubber_band_area(
        &mut self,
        maximal_size: [i32; 3],
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) -> bool {
        let Some((x0, x1, y0, y1)) = native.isosurface_rubber_band_area() else {
            return false;
        };
        if x1 - x0 < 8. || y1 - y0 < 8. {
            return false;
        }
        self.show_defined_area(x0, x1, y0, y1, maximal_size, data, hist, native, true)
    }
    pub fn button_pressed(&mut self, which: i32, native: &mut dyn IsosurfaceNativeBoundary) {
        if which == 2 {
            native.fill_cache()
        } else if which < 2 {
            native.save_object(which == 1)
        }
    }
    /// `managePaintObject`.
    pub fn manage_paint_object(
        &mut self,
        data: &ImodvIsosurfaceData,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) -> String {
        let scattered = native.isosurface_scattered_objects();
        filter_isosurface_paint_list(&mut self.m_paint_obj_list, &scattered);
        native.set_isosurface_paint_list_enabled(data.flags & IIS_PAINT_OBJECT != 0);
        self.paint_list_to_string()
    }
    /// `modelToPaintList`.
    pub fn model_to_paint_list(&mut self, native: &mut dyn IsosurfaceNativeBoundary) -> String {
        self.m_paint_obj_list = native.marked_isosurface_paint_objects();
        self.paint_list_to_string()
    }
    pub fn paint_list_to_string(&self) -> String {
        isosurface_paint_list_to_string(&self.m_paint_obj_list)
    }
    pub fn get_parameters_from_store(&mut self, _threshold_only: bool, _set_controls: bool) -> i32 {
        0
    }
    /// `setParametersInStore`; packed record ownership is at the model host.
    pub fn set_parameters_in_store(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        native.persist_isosurface_parameters();
    }
    pub fn file_value_of_threshold(&self) -> f32 {
        self.m_threshold
    }
    pub fn threshold_from_file_value(&self, value: f32) -> f32 {
        value
    }
    pub fn find_closest_z(&self, iz: i32, listz: &[i32]) -> (i32, i32) {
        let candidates: Vec<(i32, bool)> = listz.iter().copied().map(|z| (z, true)).collect();
        isosurface_find_closest_z(iz, self.m_box_origin[2], &candidates)
    }
    /// `showDefinedArea`.
    pub fn show_defined_area(
        &mut self,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        maximal_size: [i32; 3],
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
        draw: bool,
    ) -> bool {
        self.show_defined_area_with_boundary(x0, x1, y0, y1, maximal_size, data, hist, native, draw)
    }
    /// Host-callable `showDefinedArea`.  The rewritten GUI supplies its
    /// maximal volume dimensions and uses boundary callbacks for the two
    /// renderer/model operations that the C++ code sends to 3dmod.
    pub fn show_defined_area_with_boundary(
        &mut self,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        maximal_size: [i32; 3],
        data: &ImodvIsosurfaceData,
        hist: &mut HistWidget,
        native: &mut dyn IsosurfaceNativeBoundary,
        draw: bool,
    ) -> bool {
        let area = isosurface_defined_area(x0, x1, y0, y1, native.image_size(), maximal_size);
        self.apply_defined_area(area);
        if self.alloc_arrays_if_needed(data.flags) {
            return false;
        }
        if data.flags & IIS_CENTER_VOLUME != 0 {
            self.set_view_center(native);
        }
        if data.flags & IIS_LINK_XYZ != 0 {
            let mut cursor = native.cursor_xyz();
            cursor[0] = self.m_local_x;
            cursor[1] = self.m_local_y;
            native.set_cursor_xyz(cursor);
        }
        self.set_bounding_box(native);
        self.set_bounding_obj(native);
        self.fill_and_process_vols(native, data, hist, -1., false, -1.);
        self.set_iso_obj(native, true);
        if draw {
            native.redraw();
            if data.flags & IIS_VIEW_BOX != 0 {
                native.redraw_image();
            }
        }
        true
    }
    /// `ImodvIsosurface::topChangeEvent`.
    pub fn top_change_event(
        &mut self,
        event: IsosurfaceChangeEvent,
        native: &mut dyn IsosurfaceNativeBoundary,
    ) {
        self.m_rounded_style = native.rounded_style();
        native.dialog_change_event();
        native.check_and_set_mac_menu();
        if event == IsosurfaceChangeEvent::FontChange {
            self.set_font_dependent_widths(native);
        }
    }
    /// `ImodvIsosurface::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        native.set_button_width("use_rubber", self.m_rounded_style, 1.2, "Rubberband");
        native.set_button_width("size_contours", self.m_rounded_style, 1.2, "Contour");
        native.limit_edit_stretch("paint_list_edit", "paint_layout", 4);
    }
    /// `ImodvIsosurface::topCloseEvent`.
    pub fn top_close_event(&mut self, native: &mut dyn IsosurfaceNativeBoundary) {
        native.remove_isosurface_dialog();
        self.m_top_window_open = false;
        native.accept_close_event();
        native.free_extra_object(self.m_box_obj_num);
        native.free_extra_object(self.m_extra_obj_num);
        native.object_editor_new_view();
        native.enable_user_model_drawing();
        native.draw_model();
        self.m_volume.clear();
        self.m_true_bin_vol.clear();
        self.m_paint_vol.clear();
        native.release_isosurface_meshes();
    }
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
    pub fn dump_volume(&self, filename: &str, binned: bool) {
        let _ = dump_isosurface_volume(self, filename, binned);
    }
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
        iso.set_bounding_obj(native);
        if data.flags & IIS_CENTER_VOLUME != 0 {
            iso.set_view_center(native)
        }
        iso.fill_and_process_vols(native, data, hist, PERCENTILE, false, -1.);
        iso.set_iso_obj(native, true);
        true
    } else if iso.fill_paint_vol(native) {
        iso.paint_mesh(native);
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
    #[derive(Default)]
    struct EventBoundary {
        rounded: bool,
        calls: Vec<String>,
        paint_input: Option<IsosurfacePaintInput>,
    }
    impl EventBoundary {
        fn call(&mut self, value: impl Into<String>) {
            self.calls.push(value.into());
        }
    }
    impl IsosurfaceNativeBoundary for EventBoundary {
        fn open_dialog(&mut self) -> bool {
            true
        }
        fn raise_dialog(&mut self) {}
        fn close_dialog(&mut self) {}
        fn image_size(&self) -> [i32; 3] {
            [0; 3]
        }
        fn cursor_xyz(&self) -> [i32; 3] {
            [0; 3]
        }
        fn time_index(&self) -> i32 {
            0
        }
        fn time_count(&self) -> i32 {
            0
        }
        fn voxel(&self, _: i32, _: i32, _: i32) -> u8 {
            0
        }
        fn set_cursor_xyz(&mut self, _: [i32; 3]) {}
        fn set_bounding_box(&mut self, _: [i32; 3], _: [i32; 3]) {}
        fn set_bounding_object(&mut self, origin: [i32; 3], ends: [i32; 3]) {
            self.call(format!("box_object:{origin:?}:{ends:?}"))
        }
        fn set_view_center(&mut self, _: [i32; 3]) {}
        fn set_iso_mesh(&mut self, _: &[u8], _: [i32; 3], _: f32, _: i32) {}
        fn smooth_iso_mesh(&mut self, iterations: i32) {
            self.call(format!("smooth:{iterations}"))
        }
        fn filter_iso_mesh(&mut self, enabled: bool, minimum_triangles: i32) {
            self.call(format!("filter:{enabled}:{minimum_triangles}"))
        }
        fn replace_iso_mesh_with_filtered(&mut self) {
            self.call("replace_filtered")
        }
        fn persist_isosurface_parameters(&mut self) {
            self.call("persist")
        }
        fn set_isosurface_visible(&mut self, visible: bool) {
            self.call(format!("iso_visible:{visible}"))
        }
        fn set_user_model_visible(&mut self, visible: bool) {
            self.call(format!("model_visible:{visible}"))
        }
        fn set_isosurface_box_visible(&mut self, visible: bool) {
            self.call(format!("box_visible:{visible}"))
        }
        fn paint_iso_mesh(&mut self, volume: &[u8], size: [i32; 3], colors: &[IsoColor]) {
            self.call(format!("paint:{}:{size:?}:{}", volume.len(), colors.len()))
        }
        fn collect_isosurface_paint_points(
            &mut self,
            _: [i32; 3],
            _: [i32; 3],
            _: &[i32],
        ) -> Option<IsosurfacePaintInput> {
            self.paint_input.clone()
        }
        fn redraw(&mut self) {
            self.call("redraw")
        }
        fn save_object(&mut self, _: bool) {}
        fn fill_cache(&mut self) {}
        fn rounded_style(&self) -> bool {
            self.rounded
        }
        fn dialog_change_event(&mut self) {
            self.call("base_change")
        }
        fn check_and_set_mac_menu(&mut self) {
            self.call("mac_menu")
        }
        fn set_button_width(&mut self, button: &str, rounded: bool, factor: f32, label: &str) {
            self.call(format!("width:{button}:{rounded}:{factor}:{label}"))
        }
        fn limit_edit_stretch(&mut self, edit: &str, layout: &str, stretch: i32) {
            self.call(format!("stretch:{edit}:{layout}:{stretch}"))
        }
        fn remove_isosurface_dialog(&mut self) {
            self.call("remove_dialog")
        }
        fn accept_close_event(&mut self) {
            self.call("accept_close")
        }
        fn free_extra_object(&mut self, object: i32) {
            self.call(format!("free:{object}"))
        }
        fn object_editor_new_view(&mut self) {
            self.call("new_view")
        }
        fn enable_user_model_drawing(&mut self) {
            self.call("enable_model")
        }
        fn draw_model(&mut self) {
            self.call("draw_model")
        }
        fn release_isosurface_meshes(&mut self) {
            self.call("release_meshes")
        }
    }
    #[test]
    fn smoothing_action_skips_zero_and_delegates_nonzero_iterations() {
        let mut iso = ImodvIsosurface::default();
        let mut native = EventBoundary::default();
        iso.smooth_mesh(0, &mut native);
        iso.smooth_mesh(3, &mut native);
        assert_eq!(native.calls, ["smooth:3"]);
    }
    #[test]
    fn filtering_tracks_owned_filtered_mesh_and_forwards_triangle_limit() {
        let mut iso = ImodvIsosurface::default();
        let mut native = EventBoundary::default();

        iso.filter_mesh(true, 100, &mut native);
        assert!(iso.m_made_filtered);
        iso.filter_mesh(false, 100, &mut native);

        assert!(!iso.m_made_filtered);
        assert_eq!(native.calls, ["filter:true:100", "filter:false:100"]);
    }
    #[test]
    fn iteration_change_rebuilds_smooths_paints_and_draws_in_source_order() {
        let mut iso = ImodvIsosurface::default();
        let mut data = ImodvIsosurfaceData::default();
        let mut native = EventBoundary::default();

        iso.iter_num_changed(3, &mut data, &mut native);

        assert_eq!(data.it_num, 3);
        assert_eq!(
            native.calls,
            [
                "persist",
                "replace_filtered",
                "smooth:3",
                "paint:0:[0, 0, 0]:0",
                "redraw"
            ]
        );
    }
    #[test]
    fn delete_pieces_rebuilds_the_mesh_with_the_saved_smoothing_level() {
        let mut iso = ImodvIsosurface::default();
        let mut data = ImodvIsosurfaceData {
            it_num: 2,
            min_t_num: 50,
            ..Default::default()
        };
        let mut native = EventBoundary::default();

        iso.delete_pieces_toggled(true, &mut data, &mut native);

        assert_ne!(data.flags & IIS_DELETE_PIECES, 0);
        assert_eq!(
            native.calls,
            [
                "persist",
                "filter:true:50",
                "persist",
                "replace_filtered",
                "smooth:2",
                "paint:0:[0, 0, 0]:0",
                "redraw"
            ]
        );
    }
    #[test]
    fn view_toggles_apply_source_flags_and_host_draws() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_origin = [2, 4, 6];
        iso.m_box_ends = [4, 8, 10];
        let mut data = ImodvIsosurfaceData::default();
        let mut native = EventBoundary::default();

        iso.view_iso_toggled(false, &mut data, &mut native);
        iso.view_model_toggled(false, &mut data, &mut native);
        iso.view_boxing_toggled(false, &mut data, &mut native);
        iso.center_volume_toggled(true, &mut data, &mut native);

        assert_eq!(data.flags & IIS_VIEW_ISOSURFACE, 0);
        assert_eq!(data.flags & IIS_VIEW_USER_MODEL, 0);
        assert_eq!(data.flags & IIS_VIEW_BOX, 0);
        assert_ne!(data.flags & IIS_CENTER_VOLUME, 0);
        assert_eq!(
            native.calls,
            [
                "iso_visible:false",
                "redraw",
                "model_visible:false",
                "redraw",
                "box_visible:false",
                "draw_model",
                "redraw"
            ]
        );
    }
    #[test]
    fn linking_xyz_refreshes_paint_and_draws_the_model_layer() {
        let mut iso = ImodvIsosurface::default();
        let mut data = ImodvIsosurfaceData::default();
        let mut native = EventBoundary::default();

        iso.link_xyz_toggled(true, &mut data, &mut native);

        assert_ne!(data.flags & IIS_LINK_XYZ, 0);
        assert_eq!(native.calls, ["draw_model"]);
    }
    #[test]
    fn painting_forwards_the_source_paint_volume_and_color_table() {
        let mut iso = ImodvIsosurface::default();
        iso.m_paint_vol = vec![1, 2, 3, 4];
        iso.m_paint_size = [2, 2, 1];
        iso.m_color_list.push(IsoColor {
            r: 1,
            g: 2,
            b: 3,
            ..Default::default()
        });
        let mut native = EventBoundary::default();

        iso.paint_mesh(&mut native);

        assert_eq!(native.calls, ["paint:4:[2, 2, 1]:1"]);
    }
    #[test]
    fn paint_volume_rasterizes_changed_host_input_and_caches_equal_input() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_size = [1, 1, 1];
        let mut native = EventBoundary {
            paint_input: Some(IsosurfacePaintInput {
                colors: vec![IsoColor {
                    r: 10,
                    g: 20,
                    b: 30,
                    ..Default::default()
                }],
                points: vec![IsoPaintPoint {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                    size: 1.,
                    color_ind: 1,
                    ..Default::default()
                }],
                zscale: 1.,
            }),
            ..Default::default()
        };

        assert!(iso.fill_paint_vol(&mut native));
        assert_eq!(iso.m_paint_vol, [1]);
        assert!(!iso.fill_paint_vol(&mut native));
        native.paint_input = None;
        assert!(iso.fill_paint_vol(&mut native));
        assert!(iso.m_color_list.is_empty() && iso.m_paint_points.is_empty());
        assert!(!iso.fill_paint_vol(&mut native));
    }
    #[test]
    fn bounding_object_uses_the_current_source_box_limits() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_origin = [2, 3, 4];
        iso.m_box_ends = [12, 13, 14];
        let mut native = EventBoundary::default();

        iso.set_bounding_obj(&mut native);

        assert_eq!(native.calls, ["box_object:[2, 3, 4]:[12, 13, 14]"]);
    }
    #[test]
    fn bounding_object_contours_follow_native_corner_order() {
        let contours = isosurface_bounding_box_contours([2, 3, 4], [12, 13, 14]);

        assert_eq!(
            contours[0],
            [
                IsoPoint3D {
                    ix: 2,
                    iy: 3,
                    iz: 4
                },
                IsoPoint3D {
                    ix: 12,
                    iy: 3,
                    iz: 4
                },
                IsoPoint3D {
                    ix: 12,
                    iy: 13,
                    iz: 4
                },
                IsoPoint3D {
                    ix: 2,
                    iy: 13,
                    iz: 4
                },
            ]
        );
        assert_eq!(
            contours[2][2],
            IsoPoint3D {
                ix: 12,
                iy: 3,
                iz: 14
            }
        );
        assert_eq!(
            contours[3][3],
            IsoPoint3D {
                ix: 2,
                iy: 13,
                iz: 14
            }
        );
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
    #[test]
    fn native_toggle_flags_set_and_clear_independently() {
        let mut data = ImodvIsosurfaceData::default();
        set_isosurface_flag(&mut data, IIS_VIEW_ISOSURFACE, false);
        set_isosurface_flag(&mut data, IIS_DELETE_PIECES, true);
        assert_eq!(data.flags & IIS_VIEW_ISOSURFACE, 0);
        assert_ne!(data.flags & IIS_DELETE_PIECES, 0);
        assert_ne!(data.flags & IIS_LINK_XYZ, 0);
    }
    #[test]
    fn isosurface_dialog_slots_update_native_resident_parameters() {
        let mut data = ImodvIsosurfaceData::default();
        set_isosurface_view_model(&mut data, false);
        set_isosurface_close_faces(&mut data, true);
        set_isosurface_iterations(&mut data, 99);
        set_isosurface_kernel_sigma(&mut data, -2.);
        set_isosurface_min_triangles(&mut data, 42);
        set_isosurface_mask(&mut data, 99);
        assert_eq!(data.flags & IIS_VIEW_USER_MODEL, 0);
        assert_ne!(data.flags & IIS_CLOSE_FACES, 0);
        assert_eq!(data.it_num, 99);
        assert_eq!(data.kernel_sigma, -2.);
        assert_eq!(data.min_t_num, 42);
        assert_ne!(data.flags & IIS_DELETE_PIECES, 0);
        assert_eq!(data.mask_type, 99);
    }
    #[test]
    fn defined_area_clamps_size_and_shifts_box_inside_image() {
        assert_eq!(
            isosurface_defined_area(70., 120., 2., 22., [100, 80, 30], [30, 25, 30]),
            IsosurfaceDefinedArea {
                center: [95, 12],
                size: [30, 20],
                origin: [70, 2],
                ends: [100, 22]
            }
        );
    }
    #[test]
    fn applying_defined_area_updates_native_box_and_local_coordinates() {
        let mut iso = ImodvIsosurface::default();
        iso.apply_defined_area(IsosurfaceDefinedArea {
            center: [12, 13],
            size: [20, 21],
            origin: [2, 3],
            ends: [22, 24],
        });
        assert_eq!((iso.m_local_x, iso.m_local_y), (12, 13));
        assert_eq!(iso.m_box_size, [20, 21, 0]);
        assert_eq!(iso.m_box_origin, [2, 3, 0]);
        assert_eq!(iso.m_box_ends, [22, 24, 0]);
    }
    #[test]
    fn defined_area_host_bridge_rebuilds_box_volume_mesh_and_draws() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_size[2] = 4;
        let data = ImodvIsosurfaceData::default();
        let mut hist = HistWidget::default();
        let mut native = N;
        assert!(iso.show_defined_area_with_boundary(
            1.,
            3.,
            1.,
            3.,
            [4, 4, 4],
            &data,
            &mut hist,
            &mut native,
            true,
        ));
        assert_eq!((iso.m_local_x, iso.m_local_y), (2, 2));
        assert_eq!(iso.m_box_size, [2, 2, 4]);
        assert_eq!(iso.m_box_origin, [1, 1, 0]);
        assert_eq!(iso.m_box_ends, [2, 2, 3]);
        assert_eq!(iso.m_volume.len(), 16);
        assert_eq!(iso.m_bin_volume.len(), 16);
    }
    #[test]
    fn slider_move_uses_one_based_coordinates_and_direct_sizes() {
        let mut iso = ImodvIsosurface::default();
        assert!(isosurface_slider_move(&mut iso, IIS_Z_COORD, 8));
        assert!(!isosurface_slider_move(&mut iso, IIS_Y_SIZE, 19));
        assert_eq!(iso.m_local_z, 7);
        assert_eq!(iso.m_box_size[1], 19);
    }
    #[test]
    fn closest_z_prefers_nearest_eligible_below_on_ties_and_reports_other_side() {
        assert_eq!(
            isosurface_find_closest_z(5, 10, &[(13, true), (15, false), (17, true), (20, true),]),
            (13, 17)
        );
    }
    #[test]
    fn closest_z_method_accounts_for_box_origin() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_origin[2] = 10;
        assert_eq!(iso.find_closest_z(5, &[13, 17]), (13, 17));
    }
    #[test]
    fn paint_object_list_round_trips_one_based_ranges() {
        let indices = parse_isosurface_paint_list("1,3-5,8");
        assert_eq!(indices, [0, 2, 3, 4, 7]);
        assert_eq!(isosurface_paint_list_to_string(&indices), "1,3-5,8");
    }
    #[test]
    fn paint_object_toggle_owns_and_releases_paint_volume() {
        let mut iso = ImodvIsosurface::default();
        let mut data = ImodvIsosurfaceData::default();
        iso.m_box_size = [2, 3, 4];
        assert!(set_isosurface_paint_object(&mut iso, &mut data, true));
        assert_eq!(iso.m_paint_vol.len(), 24);
        assert_ne!(data.flags & IIS_PAINT_OBJECT, 0);
        assert!(set_isosurface_paint_object(&mut iso, &mut data, false));
        assert!(iso.m_paint_vol.is_empty());
        assert_eq!(data.flags & IIS_PAINT_OBJECT, 0);
    }
    #[test]
    fn paint_list_text_updates_resident_object_indices() {
        let mut iso = ImodvIsosurface::default();
        assert_eq!(iso.set_paint_list_text("2-4,8"), "2-4,8");
        assert_eq!(iso.m_paint_obj_list, [1, 2, 3, 7]);
    }
    #[test]
    fn paint_list_discards_invalid_and_non_scattered_objects() {
        let mut list = vec![-1, 0, 1, 2, 5];
        assert!(filter_isosurface_paint_list(
            &mut list,
            &[true, false, true]
        ));
        assert_eq!(list, [0, 2]);
    }
    #[test]
    fn paint_list_edit_normalizes_against_current_model_objects() {
        let mut iso = ImodvIsosurface::default();
        assert_eq!(
            iso.set_paint_list_for_objects("1-4", &[true, false, true]),
            "1,3"
        );
        assert_eq!(iso.m_paint_obj_list, [0, 2]);
    }
    #[test]
    fn histogram_threshold_change_keeps_outer_limit_on_correct_side() {
        let mut iso = ImodvIsosurface::default();
        iso.m_median = 100.;
        iso.m_outer_limit = 40;
        iso.m_vol_min = 0;
        iso.m_vol_max = 255;
        assert!(isosurface_hist_change(&mut iso, 0, 80));
        assert_eq!(iso.m_threshold, 80.5);
        assert_eq!(iso.m_outer_limit, 40);
        assert!(isosurface_hist_change(&mut iso, 1, 120));
        assert_eq!(iso.m_outer_limit, 80);
    }
    #[test]
    fn dragging_histogram_slider_requires_hot_state() {
        let mut iso = ImodvIsosurface::default();
        iso.m_threshold = 17.;
        assert!(!iso.hist_changed_with_hot(0, 30, true, false));
        assert_eq!(iso.m_threshold, 17.);
        assert!(!iso.hist_changed_with_hot(0, 30, true, true));
        assert_eq!(iso.m_threshold, 30.5);
    }
    #[test]
    fn threshold_file_conversion_handles_ushort_window_and_clamping() {
        let value = isosurface_file_value_of_threshold(127.5, true, 1000, 3000, 0., 65535.);
        assert_eq!(value, 2000.);
        assert_eq!(
            isosurface_threshold_from_file_value(value, true, 1000, 3000, 0., 65535.),
            127.5
        );
        assert_eq!(
            isosurface_threshold_from_file_value(-10., false, 0, 0, 0., 100.),
            0.
        );
    }
    #[test]
    fn dump_volume_selects_native_binned_or_unbinned_byte_stack() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_size = [2, 2, 1];
        iso.m_bin_box_size = [1, 1, 1];
        iso.m_volume = vec![1, 2, 3, 4];
        iso.m_bin_volume = vec![9];
        assert_eq!(
            isosurface_dump_volume_data(&iso, false),
            Some((&[1, 2, 3, 4][..], [2, 2, 1]))
        );
        assert_eq!(
            isosurface_dump_volume_data(&iso, true),
            Some((&[9][..], [1, 1, 1]))
        );
        iso.m_volume.truncate(3);
        assert_eq!(isosurface_dump_volume_data(&iso, false), None);
    }
    #[test]
    fn dump_volume_writes_native_byte_mrc_header_and_slices() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_size = [2, 2, 1];
        iso.m_volume = vec![4, 5, 6, 7];
        let path = std::env::temp_dir().join(format!(
            "imod-rs-isovol-{}-{}.mrc",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        assert_eq!(dump_isosurface_volume(&iso, &path, false), 0);
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(bytes.len(), 1028);
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
        assert_eq!(
            i32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            MRC_MODE_BYTE
        );
        // `mrc_head_new` follows the process's native signed-byte write
        // convention; this test configuration stores signed bytes with 128
        // added by the shared MRC writer, just like the C call path.
        assert_eq!(&bytes[1024..], &[132, 133, 134, 135]);
    }
    #[test]
    fn mesh_filter_keeps_only_components_strictly_above_triangle_limit() {
        let triangles = [0, 1, 2, 2, 1, 3, 4, 5, 6];
        assert_eq!(
            filter_isosurface_triangles(&[], &triangles, 1),
            vec![0, 1, 2, 2, 1, 3]
        );
        assert!(filter_isosurface_triangles(&[], &triangles, 2).is_empty());
    }
    #[test]
    fn paint_rasterizer_uses_voxel_centers_and_nearest_overlapping_sphere() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_size = [2, 1, 1];
        iso.m_paint_points = vec![
            IsoPaintPoint {
                x: 0.5,
                y: 0.5,
                z: 0.5,
                size: 1.,
                color_ind: 1,
                ..Default::default()
            },
            IsoPaintPoint {
                x: 1.5,
                y: 0.5,
                z: 0.5,
                size: 1.,
                color_ind: 2,
                ..Default::default()
            },
        ];
        assert!(rasterize_isosurface_paint_volume(&mut iso, 1.));
        assert_eq!(iso.m_paint_vol, [1, 2]);
        assert_eq!(iso.m_paint_size, [2, 1, 1]);
    }
    #[test]
    fn paint_rasterizer_prefers_a_contained_smaller_color_sphere() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_size = [2, 1, 1];
        iso.m_paint_points = vec![
            IsoPaintPoint {
                x: 0.5,
                y: 0.5,
                z: 0.5,
                size: 1.,
                color_ind: 1,
                ..Default::default()
            },
            IsoPaintPoint {
                x: 2.5,
                y: 0.5,
                z: 0.5,
                size: 4.,
                color_ind: 2,
                ..Default::default()
            },
        ];
        assert!(rasterize_isosurface_paint_volume(&mut iso, 1.));
        assert_eq!(iso.m_paint_vol, [1, 1]);
    }
    #[test]
    fn paint_mesh_store_uses_native_list_positions_and_rgb_bytes() {
        let vertices = vec![
            crate::imod::libimod::imodel::Ipoint {
                x: 10.,
                y: 20.,
                z: 30.,
            },
            crate::imod::libimod::imodel::Ipoint {
                x: 11.,
                y: 20.,
                z: 30.,
            },
            crate::imod::libimod::imodel::Ipoint {
                x: 12.,
                y: 20.,
                z: 30.,
            },
        ];
        let stores = isosurface_paint_mesh_store(
            &vertices,
            &[-25, 0, 1, 2],
            [10, 20, 30],
            [3, 1, 1],
            &[1, 0, 1],
            &[IsoColor {
                r: 8,
                g: 9,
                b: 10,
                ..Default::default()
            }],
        );
        assert_eq!(stores.len(), 2);
        assert_eq!(stores[0].type_, GEN_STORE_COLOR);
        assert_eq!(stores[0].flags, GEN_STORE_BYTE << 2);
        assert_eq!(stores[0].index.i(), 1);
        assert_eq!(stores[0].value.b(), [8, 9, 10, 0]);
        assert_eq!(stores[1].index.i(), 3);
    }
    #[test]
    fn paint_mesh_store_completes_partial_transparent_triangles() {
        let vertices = vec![
            crate::imod::libimod::imodel::Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            crate::imod::libimod::imodel::Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            crate::imod::libimod::imodel::Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
        ];
        let stores = isosurface_paint_mesh_store(
            &vertices,
            &[-25, 0, 1, 2],
            [0, 0, 0],
            [3, 1, 1],
            &[1, 0, 0],
            &[IsoColor {
                trans: 77,
                r: 1,
                g: 2,
                b: 3,
                ..Default::default()
            }],
        );
        assert_eq!(
            stores
                .iter()
                .filter(|store| store.type_ == GEN_STORE_COLOR)
                .count(),
            1
        );
        let trans: Vec<_> = stores
            .iter()
            .filter(|store| store.type_ == GEN_STORE_TRANS)
            .collect();
        assert_eq!(trans.len(), 3);
        assert_eq!(trans[0].index.i(), 1);
        assert_eq!(trans[0].value.i(), 77);
        assert_eq!((trans[1].index.i(), trans[1].value.i()), (2, 1));
        assert_eq!((trans[2].index.i(), trans[2].value.i()), (3, 1));
    }
    #[test]
    fn model_store_round_trips_packed_isosurface_parameters_and_time_threshold() {
        let mut iso = ImodvIsosurface::default();
        let mut data = ImodvIsosurfaceData::default();
        iso.m_outer_limit = 77;
        iso.m_vol_min = 0;
        iso.m_vol_max = 255;
        data.binning_num = 3;
        data.it_num = 9;
        data.kernel_sigma = 1.2345;
        data.min_t_num = 456;
        set_isosurface_delete_pieces(&mut data, true);
        set_isosurface_close_faces(&mut data, true);
        let mut store = Vec::new();
        write_isosurface_store(&mut store, &iso, &data, 3, 81.25);
        // Rewriting the same time replaces rather than accumulates a threshold.
        write_isosurface_store(&mut store, &iso, &data, 3, 92.5);
        assert_eq!(
            store
                .iter()
                .filter(|item| item.type_ == ISO_STORE_PARAMS)
                .count(),
            1
        );
        assert_eq!(
            store
                .iter()
                .filter(|item| item.type_ == ISO_STORE_THRESH)
                .count(),
            1
        );
        let loaded = read_isosurface_store(&store, 3, |_, value| value).unwrap();
        assert_eq!(loaded.binning_num, 3);
        assert_eq!(loaded.iteration_num, 9);
        assert_eq!(loaded.kernel_sigma, 1.2345);
        assert_eq!(loaded.min_triangles, 456);
        assert_eq!(loaded.outer_limit, Some(77));
        assert!(loaded.delete_pieces && loaded.close_faces);
        assert_eq!(loaded.thresholds, [None, None, Some(92.5)]);
    }
    #[test]
    fn applying_store_updates_only_present_thresholds() {
        let stored = IsosurfaceStoredParameters {
            binning_num: 2,
            iteration_num: 4,
            kernel_sigma: 0.6,
            min_triangles: 80,
            outer_limit: None,
            delete_pieces: true,
            close_faces: true,
            thresholds: vec![Some(44.), None],
        };
        let mut iso = ImodvIsosurface::default();
        iso.m_stack_thresholds = vec![20., 30.];
        let mut data = ImodvIsosurfaceData::default();
        assert!(apply_isosurface_store(&stored, &mut iso, &mut data));
        assert_eq!(iso.m_stack_thresholds, [44., 30.]);
        assert_eq!(iso.m_outer_limit, -1);
        assert_eq!((data.binning_num, data.it_num, data.min_t_num), (2, 4, 80));
        assert_ne!(data.flags & (IIS_DELETE_PIECES | IIS_CLOSE_FACES), 0);
    }
    #[test]
    fn lifecycle_events_follow_source_order_and_release_owned_buffers() {
        let mut iso = ImodvIsosurface::default();
        iso.m_box_obj_num = 3;
        iso.m_extra_obj_num = 7;
        iso.m_volume = vec![1];
        iso.m_true_bin_vol = vec![2];
        iso.m_paint_vol = vec![3];
        let mut native = EventBoundary {
            rounded: true,
            ..Default::default()
        };

        iso.top_change_event(IsosurfaceChangeEvent::FontChange, &mut native);
        iso.top_close_event(&mut native);

        assert!(iso.m_rounded_style);
        assert!(!iso.m_top_window_open);
        assert!(iso.m_volume.is_empty());
        assert!(iso.m_true_bin_vol.is_empty());
        assert!(iso.m_paint_vol.is_empty());
        assert_eq!(
            native.calls,
            [
                "base_change",
                "mac_menu",
                "width:use_rubber:true:1.2:Rubberband",
                "width:size_contours:true:1.2:Contour",
                "stretch:paint_list_edit:paint_layout:4",
                "remove_dialog",
                "accept_close",
                "free:3",
                "free:7",
                "new_view",
                "enable_model",
                "draw_model",
                "release_meshes",
            ]
        );
    }
}
