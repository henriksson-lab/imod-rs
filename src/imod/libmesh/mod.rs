//! Translation boundary for `IMOD/libmesh`.
//!
//! This library owns contour preparation, skinning, remeshing, and the
//! minimum-area contour-pair mesher used by 3dmod and `imodmesh`.  It is kept
//! separate from `libimod::imesh`, which owns only mesh data structures and
//! basic mesh operations.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, OnceLock};

// Original static `newPolyNorm` (`libmesh/remesh.c`).
static NEW_POLY_NORM: AtomicI32 = AtomicI32::new(1);
static SKIN_FLAGS: AtomicI32 = AtomicI32::new(0);
static FAST_MESH: AtomicI32 = AtomicI32::new(0);
static MESH_LIMITS: OnceLock<
    Mutex<(
        crate::imod::libimod::imodel::Ipoint,
        crate::imod::libimod::imodel::Ipoint,
    )>,
> = OnceLock::new();
static SKIN_REPORT_TIME: OnceLock<Mutex<Option<std::time::Instant>>> = OnceLock::new();

/// Native static `imeshDefaultCallback` (`skinobj.c`).
pub fn imesh_default_callback(_status: i32) -> i32 {
    0
}

/// Native `skin_report_time` (`skinobj.c`).  Reporting is enabled by the
/// same `IMOD_MESH_TIMING` environment variable; the returned elapsed time is
/// useful to Rust callers while the C routine printed it directly.
pub fn skin_report_time(_label: &str) -> Option<f32> {
    if std::env::var_os("IMOD_MESH_TIMING").is_none() {
        return None;
    }
    let clock = SKIN_REPORT_TIME.get_or_init(|| Mutex::new(None));
    let mut previous = clock.lock().ok()?;
    let now = std::time::Instant::now();
    let elapsed = previous
        .replace(now)
        .map(|then| now.duration_since(then).as_secs_f32())
        .unwrap_or(0.);
    Some(elapsed)
}

/// Text produced by native diagnostic `dump_lists` (`skinobj.c`).  The C
/// helper is only called from commented debugging sites; returning the text
/// keeps it available to Rust callers without unsolicited stdout output.
pub fn skin_dump_lists(
    message: &str,
    bottom: &[i32],
    top: &[i32],
    bottom_lookup: Option<&[i32]>,
    top_lookup: Option<&[i32]>,
) -> String {
    let format_list = |list: &[i32], lookup: Option<&[i32]>| {
        list.iter()
            .map(|&item| {
                lookup
                    .and_then(|values| {
                        usize::try_from(item)
                            .ok()
                            .and_then(|index| values.get(index))
                            .copied()
                    })
                    .unwrap_or(item)
                    + 1
            })
            .map(|item| format!(" {item}"))
            .collect::<String>()
    };
    format!(
        "{message}:\nbottom:{}\ntop:{}\n",
        format_list(bottom, bottom_lookup),
        format_list(top, top_lookup)
    )
}

/// Original static `AllPixelsProcessed` (`libmesh/skeletonize.c`).
/// Returns true only when every occupancy counter is nonzero.
pub fn all_pixels_processed(use_count: &[i32]) -> bool {
    use_count.iter().all(|&count| count != 0)
}

/// Original static `FindStartPoint` (`libmesh/skeletonize.c`).
/// Finds the first endpoint or diagonal-corner point in a medial-axis bitmap.
pub fn find_skeleton_start_point(
    width: usize,
    height: usize,
    axis: &[u8],
) -> Option<(usize, usize)> {
    if width < 3 || height < 3 || axis.len() < width.checked_mul(height)? {
        return None;
    }
    let on = |x: usize, y: usize| axis[y * width + x] != 0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            if !on(x, y) {
                continue;
            }
            let p1 = on(x - 1, y);
            let p2 = on(x + 1, y);
            let p3 = on(x, y - 1);
            let p4 = on(x, y + 1);
            let p5 = on(x - 1, y - 1);
            let p6 = on(x + 1, y - 1);
            let p7 = on(x - 1, y + 1);
            let p8 = on(x + 1, y + 1);
            let sum = [p1, p2, p3, p4, p5, p6, p7, p8]
                .into_iter()
                .filter(|value| *value)
                .count();
            if sum == 1
                || (sum == 2
                    && ((p1 && p5)
                        || (p5 && p3)
                        || (p3 && p6)
                        || (p6 && p2)
                        || (p2 && p8)
                        || (p8 && p4)
                        || (p4 && p7)
                        || (p7 && p1)))
            {
                return Some((x, y));
            }
        }
    }
    None
}

/// Original static `TakeFreemanStep` (`libmesh/skeletonize.c`).
/// Advances a bitmap coordinate by one Freeman-chain direction.
pub fn take_freeman_step(freeman: usize, x: &mut isize, y: &mut isize) {
    const DX: [isize; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    const DY: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
    if let (Some(dx), Some(dy)) = (DX.get(freeman), DY.get(freeman)) {
        *x += dx;
        *y += dy;
    }
}

/// Original static `ptcompare` (`libmesh/remesh.c`).
///
/// Orders points by X then Y.  As in the C helper, values which are not
/// ordered by either comparison (including NaN) compare equal.
pub fn compare_points_xy(
    left: &crate::imod::libimod::imodel::Ipoint,
    right: &crate::imod::libimod::imodel::Ipoint,
) -> std::cmp::Ordering {
    if left.x < right.x {
        std::cmp::Ordering::Less
    } else if left.x > right.x {
        std::cmp::Ordering::Greater
    } else if left.y > right.y {
        std::cmp::Ordering::Greater
    } else if left.y < right.y {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Original static `floatcmp` (`libmesh/objprep.c`).
///
/// This preserves C's qsort ordering for unordered values: neither comparison
/// succeeds for NaN, so such values compare equal rather than being given a
/// Rust total order.
pub fn compare_f32_native(left: f32, right: f32) -> std::cmp::Ordering {
    if left < right {
        std::cmp::Ordering::Less
    } else if left > right {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Original static `EliminateLinearPart` (`libmesh/skeletonize.c`).
/// Removes matching, nearly linear arms around a skeleton endpoint.  Point
/// arrays omit the duplicate closing point, as do the native work arrays.
pub fn eliminate_skeleton_linear_part(
    start: usize,
    opposite: usize,
    x: &mut Vec<i32>,
    y: &mut Vec<i32>,
) -> bool {
    let count = x.len();
    if count != y.len() || count < 3 || start >= count || opposite > count {
        return false;
    }
    let same = |offset: usize, x: &[i32], y: &[i32]| {
        offset < count
            && x[(start + offset) % count] == x[(count + opposite - offset) % count]
            && y[(start + offset) % count] == y[(count + opposite - offset) % count]
    };
    let mut samples = Vec::new();
    let mut offset = 1;
    while same(offset, x, y) {
        samples.push((start + offset) % count);
        offset += 1;
    }
    let n = samples.len();
    if n <= 1 || count <= 2 * n {
        return false;
    }
    let (mut sx, mut sy, mut sxy, mut sxx, mut syy) = (0_f64, 0_f64, 0_f64, 0_f64, 0_f64);
    for &index in &samples {
        let (px, py) = (x[index] as f64, y[index] as f64);
        sx += px;
        sy += py;
        sxy += px * py;
        sxx += px * px;
        syy += py * py;
    }
    let determinant = n as f64 * sxx - sx * sx;
    if determinant.abs() <= f64::EPSILON {
        return false;
    }
    let intercept = (sxx * sy - sx * sxy) / determinant;
    let slope = (n as f64 * sxy - sx * sy) / determinant;
    let residual: f64 = samples
        .iter()
        .map(|&i| {
            let d = y[i] as f64 - (intercept + slope * x[i] as f64);
            d * d
        })
        .sum();
    let variance = syy - sy * sy / n as f64;
    let r_squared = if variance > 1.0e-8 {
        1. - residual / variance
    } else {
        1.
    };
    if r_squared <= 0.99 {
        return false;
    }
    let mut nx = Vec::with_capacity(count - 2 * n);
    let mut ny = Vec::with_capacity(count - 2 * n);
    for i in 0..count {
        if (i < start || i >= start + n) && (i < opposite.saturating_sub(n) || i > opposite) {
            nx.push(x[i]);
            ny.push(y[i]);
        }
    }
    if nx.len() != count - 2 * n {
        return false;
    }
    *x = nx;
    *y = ny;
    true
}

/// Original static `EliminateStraightSegments` (`libmesh/skeletonize.c`).
pub fn eliminate_skeleton_straight_segments(
    x: &mut Vec<i32>,
    y: &mut Vec<i32>,
    endpoints: &[(i32, i32)],
) {
    for &(end_x, end_y) in endpoints {
        let Some(start) = x
            .iter()
            .zip(y.iter())
            .position(|(&px, &py)| px == end_x && py == end_y)
        else {
            continue;
        };
        let opposite = if start == 0 { x.len() } else { start };
        let _ = eliminate_skeleton_linear_part(start, opposite, x, y);
    }
}

/// Bitmap-tracing portion of `SkeletonToContour` (`libmesh/skeletonize.c`).
pub fn trace_skeleton_axis(width: usize, height: usize, axis: &mut [u8]) -> Vec<(i32, i32)> {
    if width < 3 || height < 3 || axis.len() < width * height {
        return Vec::new();
    }
    for x in 0..width {
        axis[x] = 0;
        axis[(height - 1) * width + x] = 0;
    }
    for y in 0..height {
        axis[y * width] = 0;
        axis[y * width + width - 1] = 0;
    }
    let Some((start_x, start_y)) = find_skeleton_start_point(width, height, axis) else {
        return Vec::new();
    };
    let (mut x, mut y, mut freeman) = (start_x as isize, start_y as isize, 0usize);
    let mut points = Vec::new();
    loop {
        points.push((x as i32, y as i32));
        let mut candidates = Vec::new();
        let mut next = (freeman + 2) % 8;
        for _ in 0..8 {
            const DX: [isize; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
            const DY: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
            if axis[((y + DY[next]) as usize) * width + (x + DX[next]) as usize] != 0 {
                candidates.push(next);
                if candidates.len() == 2 {
                    break;
                }
            }
            next = (next + 7) % 8;
        }
        let Some(&first) = candidates.first() else {
            break;
        };
        freeman = if candidates.len() > 1 && first % 2 == 1 && (8 + first - candidates[1]) % 8 == 1
        {
            candidates[1]
        } else {
            first
        };
        take_freeman_step(freeman, &mut x, &mut y);
        if x as usize == start_x && y as usize == start_y {
            break;
        }
    }
    points
}

/// Coordinate-mapping tail of `SkeletonToContour` (`libmesh/skeletonize.c`).
pub fn skeleton_trace_to_contour(
    trace: &[(i32, i32)],
    width: usize,
    height: usize,
    minimum_x: i32,
    minimum_y: i32,
    range_x: i32,
    range_y: i32,
    center_z: f32,
) -> crate::imod::libimod::imodel::Icont {
    // The C expression divides two integers before assigning to `float`.
    let scale_x = if width > 3 {
        ((range_x - 1) / (width - 3) as i32) as f32
    } else {
        1.
    };
    let scale_y = if height > 3 {
        ((range_y - 1) / (height - 3) as i32) as f32
    } else {
        1.
    };
    let mut contour = crate::imod::libimod::imodel::Icont::default();
    contour.pts = trace
        .iter()
        .map(|&(x, y)| crate::imod::libimod::imodel::Ipoint {
            x: minimum_x as f32 + (x - 1) as f32 * scale_x,
            y: minimum_y as f32 + (y - 1) as f32 * scale_y,
            z: center_z,
        })
        .collect();
    if contour.pts.len() > 1 {
        let first = contour.pts[0];
        if contour.pts.last() != Some(&first) {
            contour.pts.push(first);
        }
    }
    contour
}

/// Raster-input stage of `skeletonize` (`libmesh/skeletonize.c`).
pub fn skeleton_bitmap_from_scan(
    scan: &crate::imod::libimod::imodel::Icont,
) -> Option<(Vec<u8>, usize, usize, i32, i32, i32, i32)> {
    use crate::imod::libimod::icont::imod_contour_get_bbox;
    if scan.pts.len() < 2 {
        return None;
    }
    let (mut low, mut high) = (
        crate::imod::libimod::imodel::Ipoint::default(),
        crate::imod::libimod::imodel::Ipoint::default(),
    );
    imod_contour_get_bbox(Some(scan), &mut low, &mut high);
    let (range_x, range_y) = (
        (high.x - low.x).abs().ceil() as i32 + 1,
        (high.y - low.y).abs().ceil() as i32 + 1,
    );
    let (minimum_x, minimum_y) = (low.x.floor() as i32, low.y.floor() as i32);
    let divisor = if range_x > 100 && range_y > 100 { 2 } else { 1 };
    let (width, height) = (
        ((range_x + divisor - 1) / divisor + 2) as usize,
        ((range_y + divisor - 1) / divisor + 2) as usize,
    );
    let mut bitmap = vec![0; width * height];
    for pair in scan.pts.chunks_exact(2) {
        let pixels = 1 + pair[1].x.round() as i32 - pair[0].x.round() as i32;
        let (x, y) = (
            1 + pair[0].x.round() as i32 - minimum_x,
            1 + pair[0].y.round() as i32 - minimum_y,
        );
        for offset in 0..(pixels / divisor).max(0) {
            let index = (y / divisor) as usize * width + (x / divisor + offset) as usize;
            if index < bitmap.len() {
                bitmap[index] = 1;
            }
        }
    }
    Some((
        bitmap, width, height, minimum_x, minimum_y, range_x, range_y,
    ))
}

/// `SliceMedialAxis` / `Skeletonize` (`libmesh/skeletonize.c`).
pub fn skeletonize_bitmap(bitmap: &[u8], width: usize, height: usize) -> Option<Vec<u8>> {
    if width < 3 || height < 3 || bitmap.len() < width * height {
        return None;
    }
    let mut out: Vec<u8> = bitmap[..width * height]
        .iter()
        .map(|&v| u8::from(v != 0))
        .collect();
    loop {
        let mut changed = false;
        for pass in 0..2 {
            let mut remove = Vec::new();
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let i = y * width + x;
                    if out[i] == 0 {
                        continue;
                    }
                    let p2 = out[i - width];
                    let p3 = out[i - width + 1];
                    let p4 = out[i + 1];
                    let p5 = out[i + width + 1];
                    let p6 = out[i + width];
                    let p7 = out[i + width - 1];
                    let p8 = out[i - 1];
                    let p9 = out[i - width - 1];
                    let n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    let ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
                    let transitions = (0..8).filter(|&k| ring[k] == 0 && ring[k + 1] != 0).count();
                    let keep = if pass == 0 {
                        p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0
                    } else {
                        p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0
                    };
                    if (2..=6).contains(&n) && transitions == 1 && keep {
                        remove.push(i);
                    }
                }
            }
            if !remove.is_empty() {
                changed = true;
                for i in remove {
                    out[i] = 0;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Some(out)
}

/// Original `skeletonize` (`libmesh/skeletonize.c`).
pub fn skeletonize_contour(
    contour: &crate::imod::libimod::imodel::Icont,
    scan_contour: Option<&crate::imod::libimod::imodel::Icont>,
    center: crate::imod::libimod::imodel::Ipoint,
) -> Option<crate::imod::libimod::imodel::Icont> {
    use crate::imod::libimod::icont::imodel_contour_scan;
    let owned_scan;
    let scan = if let Some(scan) = scan_contour {
        scan
    } else {
        owned_scan = imodel_contour_scan(Some(contour))?;
        &owned_scan
    };
    let (bitmap, width, height, minimum_x, minimum_y, range_x, range_y) =
        skeleton_bitmap_from_scan(scan)?;
    let mut axis = skeletonize_bitmap(&bitmap, width, height)?;
    let trace = trace_skeleton_axis(width, height, &mut axis);
    if trace.is_empty() {
        return None;
    }
    Some(skeleton_trace_to_contour(
        &trace, width, height, minimum_x, minimum_y, range_x, range_y, center.z,
    ))
}

fn mesh_limits() -> &'static Mutex<(
    crate::imod::libimod::imodel::Ipoint,
    crate::imod::libimod::imodel::Ipoint,
)> {
    MESH_LIMITS.get_or_init(|| {
        Mutex::new((
            crate::imod::libimod::imodel::Ipoint {
                x: -1.0e30,
                y: -1.0e30,
                z: -1.0e30,
            },
            crate::imod::libimod::imodel::Ipoint {
                x: 1.0e30,
                y: 1.0e30,
                z: 1.0e30,
            },
        ))
    })
}

/// Original `imeshSetMinMax` (`libmesh/mkmesh.c`).
pub fn imesh_set_min_max(
    minimum: crate::imod::libimod::imodel::Ipoint,
    maximum: crate::imod::libimod::imodel::Ipoint,
) {
    *mesh_limits().lock().unwrap() = (minimum, maximum);
}

/// Original static `getnextz` (`skinobj.c`).
///
/// With skip-slice meshing, only an exact entry advances to the next occupied
/// Z.  A missing or final entry deliberately advances past the final Z; native
/// callers use that sentinel to end their connection pass.
pub fn get_next_z(zlist: &[i32], current_z: i32) -> i32 {
    let Some(&last) = zlist.last() else {
        return current_z.saturating_add(1);
    };
    for pair in zlist.windows(2) {
        if pair[0] == current_z {
            return pair[1];
        }
    }
    last.saturating_add(1)
}

/// Original static `outsideMeshLimits` (`libmesh/mkmesh.c`).
pub fn outside_mesh_limits(
    p1: &crate::imod::libimod::imodel::Ipoint,
    p2: &crate::imod::libimod::imodel::Ipoint,
    p3: &crate::imod::libimod::imodel::Ipoint,
) -> bool {
    let (minimum, maximum) = *mesh_limits().lock().unwrap();
    [p1, p2, p3].iter().all(|point| {
        (point.x < minimum.x || point.x > maximum.x) && (point.y < minimum.y || point.y > maximum.y)
    })
}

/// Original `chunkMeshAddIndex` (`libmesh/mkmesh.c`).
///
/// Rust's `Vec` owns the allocation; `max_list` retains the source chunk-size
/// growth state used by callers that build a mesh incrementally.
pub fn chunk_mesh_add_index(
    mesh: &mut crate::imod::libimod::imodel::Imesh,
    index: i32,
    max_list: &mut usize,
) {
    const CHUNK_SIZE: usize = 8192;
    if mesh.list.len() >= *max_list {
        *max_list = if mesh.list.is_empty() {
            CHUNK_SIZE
        } else {
            *max_list + CHUNK_SIZE
        };
        mesh.list.reserve(*max_list - mesh.list.len());
    }
    mesh.list.push(index);
}

/// Original static `chunkAddTriangle` (`libmesh/mkmesh.c`).
pub fn chunk_add_triangle(
    mesh: &mut crate::imod::libimod::imodel::Imesh,
    i1: i32,
    i2: i32,
    i3: i32,
    max_list: &mut usize,
    inside: bool,
) {
    use crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY;
    let (i2, i3) = if inside { (i3, i2) } else { (i2, i3) };
    if mesh.list.is_empty() {
        chunk_mesh_add_index(mesh, IMOD_MESH_BGNPOLY, max_list);
    }
    chunk_mesh_add_index(mesh, i1, max_list);
    chunk_mesh_add_index(mesh, i2, max_list);
    chunk_mesh_add_index(mesh, i3, max_list);
}

/// Original static `pointAreaQuick` (`libmesh/mkmesh.c`).
/// Returns twice the scale-adjusted triangle area, matching the C fast path.
pub fn point_area_quick(
    p1: &crate::imod::libimod::imodel::Ipoint,
    p2: &crate::imod::libimod::imodel::Ipoint,
    p3: &crate::imod::libimod::imodel::Ipoint,
    zscale: f32,
) -> f32 {
    let (x1, y1, z1) = (p1.x - p2.x, p1.y - p2.y, (p1.z - p2.z) * zscale);
    let (x2, y2, z2) = (p3.x - p2.x, p3.y - p2.y, (p3.z - p2.z) * zscale);
    let normal = (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2);
    (normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2).sqrt()
}

/// Original static `build_area_matrices` (`libmesh/mkmesh.c`).
///
/// `up` and `down` are returned in the source row-major layout: each of the
/// `top.pts.len()` rows has `bottom.pts.len()` entries.  Open objects retain
/// their final point as an endpoint, so the corresponding wraparound triangle
/// entries are zeroed just as in the C implementation.
pub fn build_area_matrices(
    bottom: &crate::imod::libimod::imodel::Icont,
    bottom_direction: i32,
    top: &crate::imod::libimod::imodel::Icont,
    top_direction: i32,
    scale: &crate::imod::libimod::imodel::Ipoint,
    open_object: bool,
) -> (Vec<f32>, Vec<f32>) {
    use crate::imod::libimod::ipoint::imod_point_area_scale;

    let bottom_size = bottom.pts.len();
    let top_size = top.pts.len();
    assert!(
        bottom_size > 0 && top_size > 0,
        "native mesh area matrices require nonempty contours"
    );
    let bottom_dim = bottom_size - usize::from(open_object);
    let top_dim = top_size - usize::from(open_object);
    let mut up = vec![0.; bottom_size * top_size];
    let mut down = vec![0.; bottom_size * top_size];
    let zscale = if (scale.x - scale.y).abs() < 1.0e-5 * scale.x {
        scale.z / scale.x
    } else {
        0.
    };
    let mut i = if bottom_direction < 0 {
        bottom_size - 1
    } else {
        0
    };
    let mut j = if top_direction < 0 { top_size - 1 } else { 0 };
    for l in 0..top_size {
        let mut next_j = j as i32 + top_direction;
        if next_j == top_size as i32 {
            next_j = 0;
        }
        if next_j < 0 {
            next_j = top_size as i32 - 1;
        }
        let next_j = next_j as usize;
        for k in 0..bottom_size {
            let mut next_i = i as i32 + bottom_direction;
            if next_i == bottom_size as i32 {
                next_i = 0;
            }
            if next_i < 0 {
                next_i = bottom_size as i32 - 1;
            }
            let next_i = next_i as usize;
            let index = k + l * bottom_size;
            if k != bottom_dim {
                up[index] = if zscale != 0. {
                    point_area_quick(&top.pts[j], &bottom.pts[i], &bottom.pts[next_i], zscale)
                } else {
                    imod_point_area_scale(&top.pts[j], &bottom.pts[i], &bottom.pts[next_i], scale)
                };
            }
            if l != top_dim {
                down[index] = if zscale != 0. {
                    point_area_quick(&top.pts[j], &top.pts[next_j], &bottom.pts[i], zscale)
                } else {
                    imod_point_area_scale(&top.pts[j], &top.pts[next_j], &bottom.pts[i], scale)
                };
            }
            i = next_i;
        }
        j = next_j;
    }
    (up, down)
}

/// Original `makeCapMesh` (`libmesh/mkmesh.c`).
///
/// The center is vertex zero, followed by one copy of each contour point.
/// Per-vertex general-storage items are generated at the exact list offsets
/// used by native IMOD when the supplied state has any requested bits.
pub fn make_cap_mesh(
    contour: &crate::imod::libimod::imodel::Icont,
    center: &crate::imod::libimod::imodel::Ipoint,
    mesh_direction: i32,
    props: &crate::imod::libimod::istore::DrawProps,
    state: i32,
    state_test: i32,
) -> Option<crate::imod::libimod::imodel::Imesh> {
    use crate::imod::libimod::imesh::{
        IMOD_MESH_BGNPOLY, IMOD_MESH_END, IMOD_MESH_ENDPOLY, imod_mesh_add_index,
        imod_mesh_add_vert,
    };
    use crate::imod::libimod::istore::istore_generate_items;

    let mut mesh = crate::imod::libimod::imodel::Imesh::default();
    imod_mesh_add_vert(&mut mesh, center);
    for point in 0..contour.pts.len() {
        let next = (point + 1) % contour.pts.len();
        imod_mesh_add_vert(&mut mesh, &contour.pts[point]);
        if outside_mesh_limits(center, &contour.pts[point], &contour.pts[next]) {
            continue;
        }
        if state & state_test != 0 {
            for offset in 1..=3 {
                if istore_generate_items(
                    &mut mesh.store,
                    props,
                    state,
                    mesh.list.len() as i32 + offset,
                    state_test,
                ) != 0
                {
                    return None;
                }
            }
        }
        imod_mesh_add_index(&mut mesh, IMOD_MESH_BGNPOLY);
        imod_mesh_add_index(
            &mut mesh,
            if mesh_direction != 0 {
                0
            } else {
                point as i32 + 1
            },
        );
        imod_mesh_add_index(&mut mesh, next as i32 + 1);
        imod_mesh_add_index(
            &mut mesh,
            if mesh_direction != 0 {
                point as i32 + 1
            } else {
                0
            },
        );
        imod_mesh_add_index(&mut mesh, IMOD_MESH_ENDPOLY);
    }
    imod_mesh_add_index(&mut mesh, IMOD_MESH_END);
    Some(mesh)
}

/// Source fallback of static `imeshContourCap` (`skinobj.c`).
///
/// Uses the robust contour center and native winding/inside orientation for a
/// fan cap.  Skeleton caps require `skeletonize`, which is a separate native
/// contour operation; this function is the common non-skeleton path.
pub fn contour_cap_mesh(
    contour: &crate::imod::libimod::imodel::Icont,
    scan_contour: Option<&mut crate::imod::libimod::imodel::Icont>,
    side: i32,
    inside: bool,
) -> Option<crate::imod::libimod::imodel::Imesh> {
    use crate::imod::libimod::icont::{
        IMOD_CONTOUR_CLOCKWISE, IMOD_CONTOUR_COUNTER_CLOCKWISE, imod_cont_z_direction,
    };
    if contour.pts.is_empty() {
        return None;
    }
    let mut working = contour.clone();
    let mut center = crate::imod::libimod::imodel::Ipoint::default();
    robust_center_of_mass(&mut working, scan_contour, &mut center);
    if side > 0 {
        center.z += 0.5;
    } else {
        center.z -= 0.5;
    }
    let direction = imod_cont_z_direction(Some(contour));
    let mut mesh_direction = i32::from(
        (side < 0 && direction == IMOD_CONTOUR_COUNTER_CLOCKWISE)
            || (side > 0 && direction == IMOD_CONTOUR_CLOCKWISE),
    );
    if inside {
        mesh_direction = 1 - mesh_direction;
    }
    make_cap_mesh(
        contour,
        &center,
        mesh_direction,
        &crate::imod::libimod::istore::DrawProps::default(),
        0,
        0,
    )
}

/// Original `makeTubeCont` (`libmesh/mkmesh.c`).
///
/// Appends a circular tube ring centered at `location`, perpendicular to the
/// supplied (scaled-coordinate) normal.  The append behavior is deliberate:
/// native callers reuse an initially empty contour and receive `slices` new
/// points on success.
pub fn make_tube_cont(
    contour: &mut crate::imod::libimod::imodel::Icont,
    location: &crate::imod::libimod::imodel::Ipoint,
    normal: &crate::imod::libimod::imodel::Ipoint,
    scale: &crate::imod::libimod::imodel::Ipoint,
    tube_diameter: f32,
    slices: i32,
) -> i32 {
    use crate::imod::libimod::imat::{
        B3D_Y, B3D_Z, imod_mat_delete, imod_mat_new, imod_mat_rot, imod_mat_scale,
        imod_mat_transform,
    };
    use crate::imod::libimod::ipoint::imod_point_append;

    let mut matrix = match imod_mat_new(3) {
        Some(matrix) => matrix,
        None => return -1,
    };
    let mut rotation = match imod_mat_new(3) {
        Some(matrix) => matrix,
        None => return -1,
    };
    let step = 360.0 / slices as f64;
    let b = (normal.z as f64).acos().to_degrees();
    let a = (normal.y as f64).atan2(normal.x as f64).to_degrees();
    imod_mat_rot(&mut matrix, b, B3D_Y);
    imod_mat_rot(&mut matrix, a, B3D_Z);
    let inverse_scale = crate::imod::libimod::imodel::Ipoint {
        x: 1. / scale.x,
        y: 1. / scale.y,
        z: 1. / scale.z,
    };
    imod_mat_scale(&mut matrix, &inverse_scale);
    let source = crate::imod::libimod::imodel::Ipoint {
        x: 0.,
        y: tube_diameter * 0.5,
        z: 0.,
    };
    for _ in 0..slices {
        let mut turned = crate::imod::libimod::imodel::Ipoint::default();
        let mut point = crate::imod::libimod::imodel::Ipoint::default();
        imod_mat_rot(&mut rotation, step, B3D_Z);
        imod_mat_transform(&rotation, &source, &mut turned);
        imod_mat_transform(&matrix, &turned, &mut point);
        point.x += location.x;
        point.y += location.y;
        point.z += location.z;
        imod_point_append(contour, point);
    }
    imod_mat_delete(&mut matrix);
    imod_mat_delete(&mut rotation);
    0
}

/// Original static `circle_top_and_direction` (`libmesh/mkmesh.c`).
fn circle_top_and_direction(
    contour: &crate::imod::libimod::imodel::Icont,
    matrix: &crate::imod::libimod::imat::Imat,
) -> (bool, usize) {
    use crate::imod::libimod::imat::imod_mat_transform;

    let mut top = 0;
    let mut top_y = 0.;
    let mut top_x = 0.;
    let mut reverse = false;
    for point in 0..=contour.pts.len() {
        let index = if point == contour.pts.len() { 0 } else { point };
        let mut transformed = crate::imod::libimod::imodel::Ipoint::default();
        imod_mat_transform(matrix, &contour.pts[index], &mut transformed);
        if point == 0 || transformed.y > top_y {
            top_y = transformed.y;
            top_x = transformed.x;
            top = index;
        }
        if point == top + 1 {
            reverse = transformed.x > top_x;
        }
    }
    (reverse, top)
}

/// Original `joinTubeCont` (`libmesh/mkmesh.c`).
///
/// Joins two tube rings with the source's proportional point matching.  The
/// input drawing properties are mutable because native code synchronizes the
/// transparency property between the two rings before generating store items.
pub fn join_tube_cont(
    first: &crate::imod::libimod::imodel::Icont,
    second: &crate::imod::libimod::imodel::Icont,
    normal: &crate::imod::libimod::imodel::Ipoint,
    first_props: &mut crate::imod::libimod::istore::DrawProps,
    mut first_state: i32,
    second_props: &mut crate::imod::libimod::istore::DrawProps,
    mut second_state: i32,
) -> Option<crate::imod::libimod::imodel::Imesh> {
    use crate::imod::libimod::imat::{B3D_Y, B3D_Z, imod_mat_delete, imod_mat_new, imod_mat_rot};
    use crate::imod::libimod::imesh::{IMOD_MESH_BGNPOLY, IMOD_MESH_END, IMOD_MESH_ENDPOLY};
    use crate::imod::libimod::istore::istore_generate_items;

    if first.pts.is_empty() || second.pts.is_empty() {
        return None;
    }
    const STATE_TEST: i32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 9);
    let generate_items = (first_state | second_state) & STATE_TEST != 0;
    let mut matrix = imod_mat_new(3)?;
    let b = (normal.z as f64).acos().to_degrees();
    let a = (normal.y as f64).atan2(normal.x as f64).to_degrees();
    imod_mat_rot(&mut matrix, -a, B3D_Z);
    imod_mat_rot(&mut matrix, -b, B3D_Y);
    let (reverse_first, mut first_point) = circle_top_and_direction(first, &matrix);
    let (reverse_second, mut second_point) = circle_top_and_direction(second, &matrix);
    let second_direction: i32 = if reverse_first == reverse_second {
        1
    } else {
        -1
    };
    if (first_state | second_state) & (1 << 2) != 0 {
        if first_props.trans != 0 && second_props.trans == 0 {
            second_props.trans = 1;
        }
        if second_props.trans != 0 && first_props.trans == 0 {
            first_props.trans = 1;
        }
        first_state |= 1 << 2;
        second_state |= 1 << 2;
    }
    let first_size = first.pts.len();
    let second_size = second.pts.len();
    let max_size = first_size.max(second_size);
    let mut mesh = crate::imod::libimod::imodel::Imesh::default();
    for _ in 0..first_size {
        mesh.vert.push(first.pts[first_point]);
        first_point = (first_point + 1) % first_size;
    }
    for _ in 0..second_size {
        mesh.vert.push(second.pts[second_point]);
        second_point =
            ((second_point as i32 + second_direction).rem_euclid(second_size as i32)) as usize;
    }
    let mut last_first = 0;
    let mut last_second = 0;
    mesh.list.push(IMOD_MESH_BGNPOLY);
    for point in 0..max_size {
        let next = (point + 1) % max_size;
        let next_first =
            (((first_size * next) as f64 / max_size as f64 + 0.5).floor() as usize) % first_size;
        let next_second =
            (((second_size * next) as f64 / max_size as f64 + 0.5).floor() as usize) % second_size;
        if last_first != next_first {
            mesh.list.extend_from_slice(&[
                first_size as i32 + last_second as i32,
                last_first as i32,
                next_first as i32,
            ]);
            if generate_items {
                let base = mesh.list.len() as i32 - 3;
                if istore_generate_items(
                    &mut mesh.store,
                    second_props,
                    second_state,
                    base,
                    STATE_TEST,
                ) != 0
                    || istore_generate_items(
                        &mut mesh.store,
                        first_props,
                        first_state,
                        base + 1,
                        STATE_TEST,
                    ) != 0
                    || istore_generate_items(
                        &mut mesh.store,
                        first_props,
                        first_state,
                        base + 2,
                        STATE_TEST,
                    ) != 0
                {
                    return None;
                }
            }
        }
        if next_second != last_second {
            mesh.list.extend_from_slice(&[
                first_size as i32 + last_second as i32,
                next_first as i32,
                first_size as i32 + next_second as i32,
            ]);
            if generate_items {
                let base = mesh.list.len() as i32 - 3;
                if istore_generate_items(
                    &mut mesh.store,
                    second_props,
                    second_state,
                    base,
                    STATE_TEST,
                ) != 0
                    || istore_generate_items(
                        &mut mesh.store,
                        first_props,
                        first_state,
                        base + 1,
                        STATE_TEST,
                    ) != 0
                    || istore_generate_items(
                        &mut mesh.store,
                        second_props,
                        second_state,
                        base + 2,
                        STATE_TEST,
                    ) != 0
                {
                    return None;
                }
            }
        }
        last_first = next_first;
        last_second = next_second;
    }
    mesh.list.push(IMOD_MESH_ENDPOLY);
    mesh.list.push(IMOD_MESH_END);
    imod_mat_delete(&mut matrix);
    Some(mesh)
}

/// Original static `endsOfWholeGap` (`libmesh/mkmesh.c`).
///
/// Starting at an endpoint in a closed contour, finds the non-gap point after
/// the complete gap and returns the first gap point encountered while walking
/// backward as `new_start`.  The half-contour bounds are deliberately those
/// of native IMOD, which prevent a malformed all-gap contour from looping.
pub fn ends_of_whole_gap(contour: &crate::imod::libimod::imodel::Icont, point: i32) -> (i32, i32) {
    use crate::imod::libimod::istore::istore_point_is_gap;

    let size = contour.pts.len() as i32;
    assert!(
        size > 0,
        "native endsOfWholeGap requires a nonempty contour"
    );
    let mut next = point;
    let mut new_start = point;
    let mut count = 0;
    while count < size / 2 {
        next = if next != 0 { next - 1 } else { size - 1 };
        if istore_point_is_gap(&contour.store, next) == 0 {
            break;
        }
        new_start = next;
        count += 1;
    }
    next = (point + 1) % size;
    count = 0;
    while count < size / 2 && istore_point_is_gap(&contour.store, next) != 0 {
        next = (next + 1) % size;
        count += 1;
    }
    (next, new_start)
}

/// Original static `addConnectorIfNone` (`libmesh/mkmesh.c`).
///
/// Inserts paired one-point connector records around an unconnected gap.  The
/// C routine's store-copy bookkeeping is unnecessary with Rust-owned vectors;
/// callers already have exclusive mutable access to the contours.
pub fn add_connector_if_none(
    bottom: &mut crate::imod::libimod::imodel::Icont,
    bottom_point: i32,
    top: &mut crate::imod::libimod::imodel::Icont,
    top_point: i32,
    maximum_connector: &mut i32,
    direction: [i32; 2],
) -> i32 {
    use crate::imod::libimod::istore::{
        GEN_STORE_CONNECT, GEN_STORE_ONEPOINT, Istore, StoreUnion, istore_connect_number,
        istore_insert,
    };

    let (bottom_next, _) = ends_of_whole_gap(bottom, bottom_point);
    let (top_next, _) = ends_of_whole_gap(top, top_point);
    if istore_connect_number(&bottom.store, bottom_point) >= 0
        || istore_connect_number(&bottom.store, bottom_next) >= 0
        || istore_connect_number(&top.store, top_point) >= 0
        || istore_connect_number(&top.store, top_next) >= 0
    {
        return 0;
    }
    for which in 0..2 {
        *maximum_connector += 1;
        let value = StoreUnion::from_i(*maximum_connector);
        let bottom_index = if which == 0 {
            bottom_point
        } else {
            bottom_next
        };
        if istore_insert(
            &mut bottom.store,
            Istore {
                type_: GEN_STORE_CONNECT,
                flags: GEN_STORE_ONEPOINT,
                index: StoreUnion::from_i(bottom_index),
                value,
            },
        ) != 0
        {
            return 1;
        }
        let top_index = if direction[0] != direction[1] {
            if which == 0 { top_next } else { top_point }
        } else if which == 0 {
            top_point
        } else {
            top_next
        };
        if istore_insert(
            &mut top.store,
            Istore {
                type_: GEN_STORE_CONNECT,
                flags: GEN_STORE_ONEPOINT,
                index: StoreUnion::from_i(top_index),
                value,
            },
        ) != 0
        {
            return 1;
        }
    }
    0
}

/// Original static `manageGaps` (`libmesh/mkmesh.c`).
///
/// Adds endpoint gaps for open contours in a closed object, then pairs gap
/// openings by scaled midpoint distance and inserts implicit connectors where
/// native IMOD permits them.  This mutates working contour stores; callers of
/// the full skinning routine use cloned contours and discard these changes
/// after mesh construction, corresponding to C's `dupStoreIfNeeded` cleanup.
pub fn manage_gaps(
    bottom: &mut crate::imod::libimod::imodel::Icont,
    top: &mut crate::imod::libimod::imodel::Icont,
    closed_object: bool,
    direction: [i32; 2],
    scale: &crate::imod::libimod::imodel::Ipoint,
) -> i32 {
    use crate::imod::libimod::imodel::ICONT_OPEN;
    use crate::imod::libimod::ipoint::{imod_point_distance, imod_point3d_scale_distance};
    use crate::imod::libimod::istore::{
        GEN_STORE_CONNECT, GEN_STORE_GAP, GEN_STORE_ONEPOINT, Istore, StoreUnion,
        istore_count_items, istore_insert, istore_point_is_gap,
    };
    if bottom.pts.is_empty() || top.pts.is_empty() {
        return 1;
    }
    let mut bottom_gaps = istore_count_items(&bottom.store, GEN_STORE_GAP, 0);
    let mut top_gaps = istore_count_items(&top.store, GEN_STORE_GAP, 0);
    let no_gaps = bottom_gaps + top_gaps == 0;
    if closed_object
        && bottom.flags & ICONT_OPEN != 0
        && istore_point_is_gap(&bottom.store, bottom.pts.len() as i32 - 1) == 0
    {
        if istore_insert(
            &mut bottom.store,
            Istore {
                type_: GEN_STORE_GAP,
                flags: GEN_STORE_ONEPOINT,
                index: StoreUnion::from_i(bottom.pts.len() as i32 - 1),
                value: StoreUnion::default(),
            },
        ) != 0
        {
            return 1;
        }
        bottom_gaps += 1;
    }
    if closed_object
        && top.flags & ICONT_OPEN != 0
        && istore_point_is_gap(&top.store, top.pts.len() as i32 - 1) == 0
    {
        if istore_insert(
            &mut top.store,
            Istore {
                type_: GEN_STORE_GAP,
                flags: GEN_STORE_ONEPOINT,
                index: StoreUnion::from_i(top.pts.len() as i32 - 1),
                value: StoreUnion::default(),
            },
        ) != 0
        {
            return 1;
        }
        top_gaps += 1;
    }
    if bottom_gaps == 0 || top_gaps == 0 {
        return 0;
    }
    let gather_gaps = |contour: &crate::imod::libimod::imodel::Icont| {
        let mut gaps = Vec::new();
        let mut large_open = None;
        let mut maximum = 0;
        for store in &contour.store {
            if store.type_ == GEN_STORE_GAP && store.index.i() >= 0 {
                let (next, start) = ends_of_whole_gap(contour, store.index.i());
                if !gaps.contains(&start) {
                    if next < start && (next > 0 || start < contour.pts.len() as i32 - 1) {
                        large_open = Some(gaps.len());
                    }
                    gaps.push(start);
                }
            }
            if store.type_ == GEN_STORE_CONNECT {
                maximum = maximum.max(store.value.i());
            }
        }
        (gaps, large_open, maximum)
    };
    let (mut bottom_list, bottom_open, bottom_maximum) = gather_gaps(bottom);
    let (mut top_list, top_open, top_maximum) = gather_gaps(top);
    let mut maximum = bottom_maximum.max(top_maximum);
    let mut bottom_left = bottom_list.len();
    let mut top_left = top_list.len();
    let mut end_connected = false;
    if let (Some(bottom_open), Some(top_open)) = (bottom_open, top_open) {
        if add_connector_if_none(
            bottom,
            bottom_list[bottom_open],
            top,
            top_list[top_open],
            &mut maximum,
            direction,
        ) != 0
        {
            return 1;
        }
        end_connected = true;
        bottom_list[bottom_open] = -1;
        top_list[top_open] = -1;
        bottom_left -= 1;
        top_left -= 1;
    }
    while bottom_left != 0 && top_left != 0 {
        let mut minimum_distance = 1.0e30_f32;
        let mut selected_bottom = 0;
        let mut selected_top = 0;
        for (bottom_item, &bottom_start) in bottom_list.iter().enumerate() {
            if bottom_start < 0 {
                continue;
            }
            let (bottom_next, _) = ends_of_whole_gap(bottom, bottom_start);
            let bottom_midpoint = crate::imod::libimod::imodel::Ipoint {
                x: (bottom.pts[bottom_start as usize].x + bottom.pts[bottom_next as usize].x) / 2.,
                y: (bottom.pts[bottom_start as usize].y + bottom.pts[bottom_next as usize].y) / 2.,
                z: (bottom.pts[bottom_start as usize].z + bottom.pts[bottom_next as usize].z) / 2.,
            };
            for (top_item, &top_start) in top_list.iter().enumerate() {
                if top_start < 0 {
                    continue;
                }
                let (top_next, _) = ends_of_whole_gap(top, top_start);
                let top_midpoint = crate::imod::libimod::imodel::Ipoint {
                    x: (top.pts[top_start as usize].x + top.pts[top_next as usize].x) / 2.,
                    y: (top.pts[top_start as usize].y + top.pts[top_next as usize].y) / 2.,
                    z: (top.pts[top_start as usize].z + top.pts[top_next as usize].z) / 2.,
                };
                let distance = imod_point3d_scale_distance(&bottom_midpoint, &top_midpoint, scale);
                if distance < minimum_distance {
                    minimum_distance = distance;
                    selected_bottom = bottom_item;
                    selected_top = top_item;
                }
            }
        }
        let bottom_start = bottom_list[selected_bottom];
        let top_start = top_list[selected_top];
        let (bottom_next, _) = ends_of_whole_gap(bottom, bottom_start);
        let (top_next, _) = ends_of_whole_gap(top, top_start);
        let maximum_gap = imod_point_distance(
            &bottom.pts[bottom_start as usize],
            &bottom.pts[bottom_next as usize],
        )
        .max(imod_point_distance(
            &top.pts[top_start as usize],
            &top.pts[top_next as usize],
        ));
        if minimum_distance < maximum_gap || no_gaps {
            if add_connector_if_none(
                bottom,
                bottom_start,
                top,
                top_start,
                &mut maximum,
                direction,
            ) != 0
            {
                return 1;
            }
            if bottom_start == bottom.pts.len() as i32 - 1 || top_start == top.pts.len() as i32 - 1
            {
                end_connected = true;
            }
        }
        bottom_list[selected_bottom] = -1;
        top_list[selected_top] = -1;
        bottom_left -= 1;
        top_left -= 1;
    }
    if closed_object
        && bottom.flags & ICONT_OPEN != 0
        && top.flags & ICONT_OPEN != 0
        && !end_connected
    {
        return add_connector_if_none(
            bottom,
            bottom.pts.len() as i32 - 1,
            top,
            top.pts.len() as i32 - 1,
            &mut maximum,
            direction,
        );
    }
    0
}

/// Original `makeConnectors` (`libmesh/mkmesh.c`).
///
/// A `None` result has the C routine's NULL meaning: no shared connector, or
/// an inconsistent connector ordering.  Each returned item carries explicit
/// third-point/gap expansion and skip information for `imeshContoursCost`.
pub fn make_connectors(
    bottom: &crate::imod::libimod::imodel::Icont,
    top: &crate::imod::libimod::imodel::Icont,
    closed_object: bool,
    direction_product: i32,
) -> Option<Vec<Connector>> {
    use crate::imod::libimod::istore::{
        GEN_STORE_CONNECT, GEN_STORE_NOINDEX, istore_connect_number, istore_count_items,
        istore_point_is_gap,
    };
    if bottom.pts.is_empty() || top.pts.is_empty() {
        return None;
    }
    let maximum = istore_count_items(&bottom.store, GEN_STORE_CONNECT, 0).min(istore_count_items(
        &top.store,
        GEN_STORE_CONNECT,
        0,
    ));
    if maximum == 0 {
        return None;
    }
    let min_bridge = std::env::var("MIN_BRIDGE_CONNECT_NUM")
        .ok()
        .and_then(|text| text.parse::<i32>().ok())
        .unwrap_or(0);
    let bottom_size = bottom.pts.len() as i32;
    let top_size = top.pts.len() as i32;
    let open_object = !closed_object;
    let mut connectors = Vec::with_capacity(maximum as usize);
    let mut extraction_open_direction = 0;

    for store in &bottom.store {
        if store.type_ != GEN_STORE_CONNECT {
            continue;
        }
        let number = store.value.i();
        if direction_product == 0 && number < min_bridge {
            continue;
        }
        if connectors
            .iter()
            .any(|connector: &Connector| connector.connect == number)
        {
            continue;
        }
        let Some(top_store) = top.store.iter().find(|candidate| {
            candidate.type_ == GEN_STORE_CONNECT && candidate.value.i() == number
        }) else {
            continue;
        };
        if !connectors.is_empty() {
            if direction_product == 0 {
                return None;
            }
            let start = connectors[0].t1;
            let last = connectors.last().unwrap().t1;
            let middle = top_store.index.i();
            if closed_object {
                if connectors.len() > 1
                    && ((direction_product * (middle - last) + top_size) % top_size
                        + (direction_product * (start - middle) + top_size) % top_size
                        != (direction_product * (start - last) + top_size) % top_size)
                {
                    continue;
                }
            } else {
                if connectors.len() == 2 {
                    extraction_open_direction = if last - start > 0 { 1 } else { -1 };
                }
                if (middle - last) * extraction_open_direction < 0 {
                    continue;
                }
            }
        }
        let mut connector = Connector {
            b1: store.index.i(),
            b2: store.index.i(),
            t1: top_store.index.i(),
            t2: top_store.index.i(),
            connect: number,
            ..Default::default()
        };
        let mut used = false;
        for candidate in &bottom.store {
            if candidate.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                break;
            }
            if candidate.type_ == GEN_STORE_CONNECT && candidate.value.i() == number {
                if candidate.index.i() == connector.b1 + 1 {
                    connector.b2 += 1;
                    used = true;
                }
                if candidate.index.i() == bottom_size - 1 && connector.b1 == 0 && closed_object {
                    connector.b1 = bottom_size - 1;
                    used = true;
                }
            }
        }
        if !used {
            for candidate in &top.store {
                if candidate.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                    break;
                }
                if candidate.type_ == GEN_STORE_CONNECT && candidate.value.i() == number {
                    if candidate.index.i() == connector.t1 + 1 {
                        connector.t2 += 1;
                        used = true;
                    }
                    if candidate.index.i() == top_size - 1 && connector.t1 == 0 && closed_object {
                        connector.t1 = top_size - 1;
                        used = true;
                    }
                }
            }
        }
        if (connector.b1 != connector.b2 && istore_point_is_gap(&bottom.store, connector.b1) != 0)
            || (connector.t1 != connector.t2 && istore_point_is_gap(&top.store, connector.t1) != 0)
        {
            connector.gap = 1;
        }
        connectors.push(connector);
    }
    if direction_product == 0 {
        return (!connectors.is_empty()).then_some(connectors);
    }

    // Native's implied third point on the far side of a gap.
    for current in 0..connectors.len() {
        let mut connector = connectors[current].clone();
        if !(connector.gap == 0 && (connector.b1 != connector.b2 || connector.t1 != connector.t2)) {
            let other_bottom = |index: i32, connectors: &[Connector]| {
                connectors.iter().enumerate().any(|(item, candidate)| {
                    item != current && (candidate.b1 == index || candidate.b2 == index)
                })
            };
            let other_top = |index: i32, connectors: &[Connector]| {
                connectors.iter().enumerate().any(|(item, candidate)| {
                    item != current && (candidate.t1 == index || candidate.t2 == index)
                })
            };
            let forward = (connector.b1 + 1) % bottom_size;
            if connector.b1 == connector.b2
                && istore_point_is_gap(&bottom.store, connector.b1) != 0
                && !(connector.b1 == bottom_size - 1 && open_object)
                && !other_bottom(forward, &connectors)
                && ((closed_object && istore_connect_number(&bottom.store, forward) >= 0)
                    || istore_connect_number(&bottom.store, forward) == connector.connect)
            {
                connector.b2 = forward;
                connector.gap = 1;
            }
            let backward = (bottom_size + connector.b1 - 1) % bottom_size;
            if connector.b1 == connector.b2
                && istore_point_is_gap(&bottom.store, backward) != 0
                && !(backward == bottom_size - 1 && open_object)
                && !other_bottom(backward, &connectors)
                && ((closed_object && istore_connect_number(&bottom.store, backward) >= 0)
                    || istore_connect_number(&bottom.store, backward) == connector.connect)
            {
                connector.b1 = backward;
                connector.gap = 1;
            }
            let forward = (connector.t1 + 1) % top_size;
            if connector.t1 == connector.t2
                && istore_point_is_gap(&top.store, connector.t1) != 0
                && !(connector.t1 == top_size - 1 && open_object)
                && !other_top(forward, &connectors)
                && ((closed_object && istore_connect_number(&top.store, forward) >= 0)
                    || istore_connect_number(&top.store, forward) == connector.connect)
            {
                connector.t2 = forward;
                connector.gap = 1;
            }
            let backward = (top_size + connector.t1 - 1) % top_size;
            if connector.t1 == connector.t2
                && istore_point_is_gap(&top.store, backward) != 0
                && !(backward == top_size - 1 && open_object)
                && !other_top(backward, &connectors)
                && ((closed_object && istore_connect_number(&top.store, backward) >= 0)
                    || istore_connect_number(&top.store, backward) == connector.connect)
            {
                connector.t1 = backward;
                connector.gap = 1;
            }
        }
        connectors[current] = connector;
    }
    let open_direction = if closed_object {
        direction_product
    } else {
        extraction_open_direction
    };
    for current in 0..connectors.len() {
        let next = (current + 1) % connectors.len();
        let mut connector = connectors[current].clone();
        if current == 0 && open_object {
            if ((connector.b1 > 0 && connector.t1 == 0 && open_direction >= 0)
                || (connector.t2 == top_size - 1 && open_direction <= 0))
            {
                let number = istore_connect_number(&bottom.store, connector.b1 - 1);
                if number >= 0 && number != connector.connect {
                    connector.skip_from_start = 1;
                    connector.skip_index = connector.b1 - 1;
                }
            }
            if connector.b1 == 0 && connector.t1 > 0 && connector.t2 < top_size - 1 {
                if open_direction >= 0 {
                    let number = istore_connect_number(&top.store, connector.t1 - 1);
                    if number >= 0 && number != connector.connect {
                        connector.skip_from_start = 1;
                        connector.skip_index = connector.t1 - 1;
                    }
                }
                if connector.skip_from_start == 0 && open_direction <= 0 {
                    let number = istore_connect_number(&top.store, connector.t2 + 1);
                    if number >= 0 && number != connector.connect {
                        connector.skip_from_start = 1;
                        connector.skip_index = connector.t2 + 1;
                    }
                }
            }
        }
        if current < connectors.len() - 1 || closed_object {
            let first_number =
                istore_connect_number(&bottom.store, (connector.b2 + 1) % bottom_size);
            let second_number = istore_connect_number(
                &bottom.store,
                (bottom_size + connectors[next].b1 - 1) % bottom_size,
            );
            if first_number >= 0
                && second_number >= 0
                && first_number != connector.connect
                && first_number != connectors[next].connect
                && second_number != connector.connect
                && second_number != connectors[next].connect
            {
                connector.skip_to_next = 1;
            }
            if connector.skip_to_next == 0 {
                let first_index = if direction_product > 0 {
                    (connector.t2 + 1) % top_size
                } else {
                    (top_size + connector.t1 - 1) % top_size
                };
                let second_index = if direction_product < 0 {
                    (connectors[next].t2 + 1) % top_size
                } else {
                    (top_size + connectors[next].t1 - 1) % top_size
                };
                let first_number = istore_connect_number(&top.store, first_index);
                let second_number = istore_connect_number(&top.store, second_index);
                if first_number >= 0
                    && second_number >= 0
                    && first_number != connector.connect
                    && first_number != connectors[next].connect
                    && second_number != connector.connect
                    && second_number != connectors[next].connect
                {
                    connector.skip_to_next = 1;
                }
            }
        } else if current != 0 || connector.skip_from_start == 0 {
            if ((connector.b2 < bottom_size - 1 && connector.t1 == 0 && open_direction <= 0)
                || (connector.t2 == top_size - 1 && open_direction >= 0))
            {
                let number = istore_connect_number(&bottom.store, connector.b2 + 1);
                if number >= 0 && number != connector.connect {
                    connector.skip_to_end = 1;
                    connector.skip_index = connector.b2 + 1;
                }
            }
            if connector.b2 == bottom_size - 1 && connector.t1 > 0 && connector.t2 < top_size - 1 {
                if open_direction >= 0 {
                    let number = istore_connect_number(&top.store, connector.t2 + 1);
                    if number >= 0 && number != connector.connect {
                        connector.skip_to_end = 1;
                        connector.skip_index = connector.t2 + 1;
                    }
                }
                if connector.skip_to_end == 0 && open_direction <= 0 {
                    let number = istore_connect_number(&top.store, connector.t1 - 1);
                    if number >= 0 && number != connector.connect {
                        connector.skip_to_end = 1;
                        connector.skip_index = connector.t1 - 1;
                    }
                }
            }
        }
        connectors[current] = connector;
    }
    Some(connectors)
}

/// One backtrace decision from `imeshContoursCost` (`mkmesh.c`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostPathStep {
    /// `path[i + j * (bdim + 1)]` selected an up triangle and decrements `i`.
    Up,
    /// It selected a down triangle and decrements `j`.
    Down,
}

/// Extracts the native cost-matrix backtrace in forward mesh-emission order.
///
/// In `imeshContoursCost`, this is the `while (i || j)` loop's path decision
/// before it maps each step to contour indices and emits a triangle.
pub fn backtrace_cost_path(
    path: &[u8],
    bottom_dimension: usize,
    bottom_length: usize,
    top_length: usize,
) -> Vec<CostPathStep> {
    let stride = bottom_dimension + 1;
    assert!(
        path.len() >= stride * (top_length + 1),
        "native path matrix is undersized"
    );
    let mut bottom = bottom_length;
    let mut top = top_length;
    let mut steps = Vec::with_capacity(bottom + top);
    while bottom != 0 || top != 0 {
        if path[bottom + top * stride] != 0 {
            assert!(bottom != 0, "native path cannot select up at left edge");
            steps.push(CostPathStep::Up);
            bottom -= 1;
        } else {
            assert!(top != 0, "native path cannot select down at bottom edge");
            steps.push(CostPathStep::Down);
            top -= 1;
        }
    }
    steps
}

/// Geometric, unconnected branch of `imeshContoursCost` (`mkmesh.c`).
///
/// This is the source path after contour direction/start-point selection and
/// before connector segmentation or per-point display-store propagation.  It
/// retains native point ordering, gap suppression, clipping, and polygon
/// markers, making it the reusable core for the full skinning entry point.
pub fn mesh_unconnected_contour_pair(
    bottom: &crate::imod::libimod::imodel::Icont,
    top: &crate::imod::libimod::imodel::Icont,
    scale: &crate::imod::libimod::imodel::Ipoint,
    inside: bool,
    direction: [i32; 2],
    start: [usize; 2],
    open_object: bool,
) -> Option<crate::imod::libimod::imodel::Imesh> {
    use crate::imod::libimod::imesh::{IMOD_MESH_END, IMOD_MESH_ENDPOLY};
    use crate::imod::libimod::imodel::ICONT_OPEN;
    use crate::imod::libimod::istore::istore_point_is_gap;

    let bottom_size = bottom.pts.len();
    let top_size = top.pts.len();
    if bottom_size == 0 || top_size == 0 || start[0] >= bottom_size || start[1] >= top_size {
        return None;
    }
    if (!open_object && bottom_size < 3 && top_size < 3)
        || (open_object && bottom_size < 2 && top_size < 2)
    {
        return None;
    }
    let bottom_dimension = bottom_size - usize::from(open_object);
    let top_dimension = top_size - usize::from(open_object);
    let (up, down) =
        build_area_matrices(bottom, direction[0], top, direction[1], scale, open_object);
    let bottom_start = if direction[0] < 0 {
        bottom_size - 1 - start[0]
    } else {
        start[0]
    };
    let top_start = if direction[1] < 0 {
        top_size - 1 - start[1]
    } else {
        start[1]
    };
    let (_, path) = cost_from_area_matrices(
        &up,
        &down,
        bottom_dimension,
        top_dimension,
        bottom_size,
        bottom_start,
        top_start,
        bottom_dimension,
        top_dimension,
        -1.,
    );
    let mut mesh = crate::imod::libimod::imodel::Imesh::default();
    let mut bottom_skip = None;
    let mut bottom_point = start[0] as i32;
    for index in 0..bottom_size {
        mesh.vert.push(bottom.pts[bottom_point as usize]);
        bottom_point += direction[0];
        if bottom_point == bottom_size as i32 {
            bottom_point = 0;
            bottom_skip = Some(index);
        } else if bottom_point < 0 {
            bottom_point = bottom_size as i32 - 1;
            bottom_skip = Some(index);
        }
    }
    mesh.vert.push(bottom.pts[start[0]]);
    let top_offset = bottom_size + 1;
    let mut top_skip = None;
    let mut top_point = start[1] as i32;
    for index in 0..top_size {
        mesh.vert.push(top.pts[top_point as usize]);
        top_point += direction[1];
        if top_point == top_size as i32 {
            top_point = 0;
            top_skip = Some(index);
        } else if top_point < 0 {
            top_point = top_size as i32 - 1;
            top_skip = Some(index);
        }
    }
    mesh.vert.push(top.pts[start[1]]);
    let mut max_list = 0;
    let mut bottom_count = bottom_dimension;
    let mut top_count = top_dimension;
    while bottom_count != 0 || top_count != 0 {
        let path_index = bottom_count + top_count * (bottom_dimension + 1);
        if path[path_index] != 0 {
            let last_bottom = bottom_count - 1;
            let top_point = ((start[1] as i32 + direction[1] * top_count as i32)
                .rem_euclid(top_size as i32)) as usize;
            let last_point = ((start[0] as i32 + direction[0] * last_bottom as i32)
                .rem_euclid(bottom_size as i32)) as usize;
            let point = ((start[0] as i32 + direction[0] * bottom_count as i32)
                .rem_euclid(bottom_size as i32)) as usize;
            if !((bottom.flags & ICONT_OPEN != 0 && bottom_skip == Some(last_bottom))
                || istore_point_is_gap(
                    &bottom.store,
                    if direction[0] > 0 {
                        last_point as i32
                    } else {
                        point as i32
                    },
                ) != 0
                || outside_mesh_limits(
                    &top.pts[top_point],
                    &bottom.pts[last_point],
                    &bottom.pts[point],
                ))
            {
                chunk_add_triangle(
                    &mut mesh,
                    (top_count + top_offset) as i32,
                    last_bottom as i32,
                    bottom_count as i32,
                    &mut max_list,
                    inside,
                );
            }
            bottom_count -= 1;
        } else {
            let last_top = top_count - 1;
            let point = ((start[1] as i32 + direction[1] * top_count as i32)
                .rem_euclid(top_size as i32)) as usize;
            let last_point = ((start[1] as i32 + direction[1] * last_top as i32)
                .rem_euclid(top_size as i32)) as usize;
            let bottom_point = ((start[0] as i32 + direction[0] * bottom_count as i32)
                .rem_euclid(bottom_size as i32)) as usize;
            if !((top.flags & ICONT_OPEN != 0 && top_skip == Some(last_top))
                || istore_point_is_gap(
                    &top.store,
                    if direction[1] > 0 {
                        last_point as i32
                    } else {
                        point as i32
                    },
                ) != 0
                || outside_mesh_limits(
                    &top.pts[last_point],
                    &bottom.pts[bottom_point],
                    &top.pts[point],
                ))
            {
                chunk_add_triangle(
                    &mut mesh,
                    (last_top + top_offset) as i32,
                    bottom_count as i32,
                    (top_count + top_offset) as i32,
                    &mut max_list,
                    inside,
                );
            }
            top_count -= 1;
        }
    }
    if !mesh.list.is_empty() {
        chunk_mesh_add_index(&mut mesh, IMOD_MESH_ENDPOLY, &mut max_list);
    }
    chunk_mesh_add_index(&mut mesh, IMOD_MESH_END, &mut max_list);
    Some(mesh)
}

/// Original `imodMeshesDeleteRes` (`libmesh/objprep.c`).
///
/// Removes meshes at one resolution while retaining the order and all data of
/// the other resolutions.  Dropping removed `Imesh` values replaces the C
/// routine's explicit vertex/list/store frees.
pub fn imod_meshes_delete_res(
    meshes: &mut Vec<crate::imod::libimod::imodel::Imesh>,
    resolution: i32,
) -> i32 {
    use crate::imod::libimod::imesh::imesh_resol;
    if meshes.is_empty() {
        return -1;
    }
    meshes.retain(|mesh| imesh_resol(mesh.flag) != resolution);
    0
}

/// Original `imeshDupMarkedConts` (`libmesh/objprep.c`).
///
/// Produces the temporary contour-only object used for meshing.  A nonzero
/// `flag` selects contours bearing that bit and clears it in the source,
/// exactly as the native one-shot work marker does; meshes, labels, and mesh
/// parameters are not copied.
pub fn imesh_dup_marked_conts(
    object: &mut crate::imod::libimod::imodel::Iobj,
    flag: u32,
) -> Option<crate::imod::libimod::imodel::Iobj> {
    use crate::imod::libimod::icont::imod_contour_dup;
    use crate::imod::libimod::iobj::imod_object_add_contour;
    use crate::imod::libimod::istore::{istore_copy_cont_surf_items, istore_copy_non_index};

    let mut output = object.clone();
    output.cont.clear();
    output.mesh.clear();
    output.store.clear();
    output.label = None;
    output.mesh_param = None;
    let mut maximum_surface = 0;
    for index in 0..object.cont.len() {
        if flag != 0 && object.cont[index].flags & flag == 0 {
            continue;
        }
        if flag != 0 {
            object.cont[index].flags &= !flag;
        }
        if object.cont[index].pts.is_empty() {
            continue;
        }
        let contour = imod_contour_dup(&object.cont[index])?;
        maximum_surface = maximum_surface.max(contour.surf);
        let new_index = imod_object_add_contour(&mut output, contour);
        if new_index < 0
            || istore_copy_cont_surf_items(
                &object.store,
                &mut output.store,
                index as i32,
                new_index,
                0,
            ) != 0
        {
            return None;
        }
    }
    if istore_copy_non_index(&object.store, &mut output.store) != 0 {
        return None;
    }
    for surface in 0..=maximum_surface {
        if istore_copy_cont_surf_items(&object.store, &mut output.store, surface, surface, 1) != 0 {
            return None;
        }
    }
    Some(output)
}

/// Original static `resecobj` (`libmesh/objprep.c`).
///
/// Removes empty contours and contours outside the selected Z range or not on
/// the requested Z increment.  `DEFAULT_VALUE` for both bounds retains the
/// full range, exactly as in the native fast-return condition.
pub fn resection_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    minimum_z: i32,
    maximum_z: i32,
    increment_z: i32,
) -> i32 {
    use crate::imod::libimod::icont::imod_contour_z_value;
    use crate::imod::libimod::imesh::DEFAULT_VALUE;
    let increment_z = increment_z.max(1);
    if minimum_z == DEFAULT_VALUE && maximum_z == DEFAULT_VALUE && increment_z == 1 {
        return 0;
    }
    let mut index = 0;
    while index < object.cont.len() {
        let contour = &object.cont[index];
        let remove = contour.pts.is_empty() || {
            let z = imod_contour_z_value(Some(contour));
            (minimum_z != DEFAULT_VALUE && z < minimum_z)
                || (maximum_z != DEFAULT_VALUE && z > maximum_z)
                || z % increment_z != 0
        };
        if remove {
            object.cont.remove(index);
        } else {
            index += 1;
        }
    }
    0
}

/// Original static `ReduceObj` (`libmesh/objprep.c`).
///
/// Reduces a contour only when it is not a loopback and preserves at least
/// four points by halving tolerance until a viable reduction is found.
pub fn reduce_object_contours(
    object: &mut crate::imod::libimod::imodel::Iobj,
    distance: f32,
) -> i32 {
    use crate::imod::libimod::icont::{imod_contour_dup, imod_contour_reduce};
    if distance <= 0. {
        return 0;
    }
    for contour in &mut object.cont {
        let count = contour.pts.len();
        let number_to_test = 4_i32.min(count as i32 / 2 - 1);
        let mut loopback = true;
        for point in 1..=number_to_test.max(0) as usize {
            if (contour.pts[point].x - contour.pts[count - point].x).abs() > 0.001
                || (contour.pts[point].y - contour.pts[count - point].y).abs() > 0.001
            {
                loopback = false;
                break;
            }
        }
        if count > 4 && !loopback {
            let mut tolerance = distance;
            while tolerance > 0.01 * distance {
                let Some(mut duplicate) = imod_contour_dup(contour) else {
                    return 1;
                };
                imod_contour_reduce(Some(&mut duplicate), tolerance);
                if duplicate.pts.len() < 4 {
                    tolerance *= 0.5;
                } else {
                    *contour = duplicate;
                    break;
                }
            }
        }
    }
    0
}

/// Original static `cleanzero` (`libmesh/objprep.c`).
pub fn clean_zero_coordinates(object: &mut crate::imod::libimod::imodel::Iobj) {
    for contour in &mut object.cont {
        for point in &mut contour.pts {
            if point.x < 0.001 && point.x > -0.001 {
                point.x = 0.;
            }
            if point.y < 0.001 && point.y > -0.001 {
                point.y = 0.;
            }
            if point.z < 0.001 && point.z > -0.001 {
                point.z = 0.;
            }
        }
    }
}

/// Original static `extendOpenEnds` (`libmesh/objprep.c`).
///
/// Extends eligible open contours from the phantom gap ends of the nearest-Z
/// companion contour, marking each inserted point as a gap.
pub fn extend_open_ends(object: &mut crate::imod::libimod::imodel::Iobj) -> i32 {
    use crate::imod::libimod::icont::{imod_contour_length, imod_contour_z_value};
    use crate::imod::libimod::imodel::ICONT_OPEN;
    use crate::imod::libimod::ipoint::{imod_point_add, imod_point_append, imod_point_distance};
    use crate::imod::libimod::istore::{
        GEN_STORE_GAP, GEN_STORE_ONEPOINT, Istore, StoreUnion, istore_insert, istore_point_is_gap,
    };

    let z_values: Vec<i32> = object
        .cont
        .iter()
        .map(|contour| imod_contour_z_value(Some(contour)))
        .collect();
    let mut phantom = Vec::new();
    for (index, contour) in object.cont.iter().enumerate() {
        if contour.flags & ICONT_OPEN != 0 && contour.pts.len() > 1 {
            let start_gap = istore_point_is_gap(&contour.store, 0) != 0;
            let end_gap = istore_point_is_gap(&contour.store, contour.pts.len() as i32 - 2) != 0;
            if (start_gap && end_gap)
                || (start_gap && istore_point_is_gap(&contour.store, 1) != 0)
                || (end_gap
                    && istore_point_is_gap(&contour.store, contour.pts.len() as i32 - 3) != 0)
            {
                phantom.push(index);
            }
        }
    }
    if phantom.is_empty() {
        return 0;
    }
    for index in 0..object.cont.len() {
        let contour = &object.cont[index];
        let eligible = contour.flags & ICONT_OPEN != 0
            && contour.pts.len() > 1
            && istore_point_is_gap(&contour.store, 0) == 0
            && istore_point_is_gap(&contour.store, contour.pts.len() as i32 - 2) == 0
            && imod_point_distance(&contour.pts[0], contour.pts.last().unwrap())
                > 0.33 * imod_contour_length(Some(contour), 0);
        if !eligible {
            continue;
        }
        let near_index = *phantom
            .iter()
            .min_by_key(|&&candidate| (z_values[candidate] - z_values[index]).abs())
            .unwrap();
        let near = object.cont[near_index].clone();
        let mut start = -1;
        while istore_point_is_gap(&near.store, start + 1) != 0 && start < near.pts.len() as i32 / 2
        {
            start += 1;
        }
        let mut end = near.pts.len() as i32;
        while istore_point_is_gap(&near.store, end - 2) != 0 && end > near.pts.len() as i32 / 2 {
            end -= 1;
        }
        let contour = &mut object.cont[index];
        for point in 0..=start {
            let mut added = near.pts[point as usize];
            added.z = z_values[index] as f32;
            if imod_point_add(contour, Some(added), point) == 0
                || istore_insert(
                    &mut contour.store,
                    Istore {
                        type_: GEN_STORE_GAP,
                        flags: GEN_STORE_ONEPOINT,
                        index: StoreUnion::from_i(point),
                        value: StoreUnion::default(),
                    },
                ) != 0
            {
                return 1;
            }
        }
        for point in end as usize..near.pts.len() {
            let mut added = near.pts[point];
            added.z = z_values[index] as f32;
            if imod_point_append(contour, added) == 0
                || istore_insert(
                    &mut contour.store,
                    Istore {
                        type_: GEN_STORE_GAP,
                        flags: GEN_STORE_ONEPOINT,
                        index: StoreUnion::from_i(contour.pts.len() as i32 - 2),
                        value: StoreUnion::default(),
                    },
                ) != 0
            {
                return 1;
            }
        }
    }
    0
}

/// Original `imeshPrepContours` (`libmesh/objprep.c`).
///
/// Flattens contours, extends eligible open ends in closed objects, filters
/// sections, reduces points, and clears near-zero coordinates.  In mean-Z
/// mode the first point deliberately keeps its original Z: this follows the
/// C loop, which assigns the calculated mean starting at point one.
pub fn imesh_prep_contours(
    object: &mut crate::imod::libimod::imodel::Iobj,
    minimum_z: i32,
    maximum_z: i32,
    increment_z: i32,
    tolerance: f32,
    use_mean_z: bool,
) -> i32 {
    use crate::imod::libimod::iobj::iobj_close;
    for contour in &mut object.cont {
        if contour.pts.is_empty() {
            return 1;
        }
        let mut z = contour.pts[0].z;
        if use_mean_z {
            for point in contour.pts.iter().skip(1) {
                z += point.z;
            }
            z /= contour.pts.len() as f32;
        }
        for point in contour.pts.iter_mut().skip(1) {
            point.z = z;
        }
    }
    if iobj_close(object.flags) != 0 && extend_open_ends(object) != 0 {
        return 1;
    }
    if resection_object(object, minimum_z, maximum_z, increment_z) != 0 {
        return 1;
    }
    if reduce_object_contours(object, tolerance) != 0 {
        return 1;
    }
    clean_zero_coordinates(object);
    0
}

/// Non-tilted-object path of native `analyzePrepSkinObj` (`objprep.c`).
///
/// It consumes the persisted mesh parameters, chooses the requested
/// resolution, prepares a temporary contour copy unless `IMESH_MK_IS_COPY`
/// is set, and transfers the resulting meshes back to the source object.
/// Tilted-contour subset rotation is intentionally rejected with `-2` until
/// that distinct source branch is translated; flattening such contours here
/// would silently produce a different mesh.
pub fn analyze_prep_skin_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    resolution: i32,
    scale: &crate::imod::libimod::imodel::Ipoint,
    callback: Option<fn(i32) -> i32>,
) -> i32 {
    use crate::imod::libimod::imesh::{IMESH_MK_IS_COPY, IMESH_MK_TUBE};
    use crate::imod::libimod::imodel::IMOD_OBJFLAG_OPEN;
    let Some(params) = object.mesh_param.clone() else {
        return 1;
    };
    let make_tubes = object.flags & IMOD_OBJFLAG_OPEN != 0 && params.flags & IMESH_MK_TUBE != 0;
    if !make_tubes
        && params.flat_crit > 0.
        && object.cont.iter().any(|contour| {
            contour
                .pts
                .iter()
                .map(|point| point.z)
                .fold(None, |bounds: Option<(f32, f32)>, z| {
                    Some(match bounds {
                        Some((low, high)) => (low.min(z), high.max(z)),
                        None => (z, z),
                    })
                })
                .is_some_and(|(low, high)| high - low >= params.flat_crit)
        })
    {
        return -2;
    }
    let (increment_z, tolerance) = if resolution != 0 {
        (params.incz_low_res, params.tol_low_res)
    } else {
        (params.incz_high_res, params.tol_high_res)
    };
    if params.flags & IMESH_MK_IS_COPY != 0 {
        if !make_tubes
            && imesh_prep_contours(
                object,
                params.minz,
                params.maxz,
                increment_z,
                tolerance,
                params.flags & crate::imod::libimod::imesh::IMESH_MK_USE_MEAN != 0,
            ) != 0
        {
            return 1;
        }
        return imesh_skin_object(
            object,
            scale,
            params.overlap as f64,
            params.cap,
            params.cap_skip_zlist.as_deref(),
            increment_z,
            params.flags,
            params.passes,
            params.tube_diameter as f64,
            callback,
        );
    }
    let mut work = object.clone();
    work.mesh.clear();
    if !make_tubes
        && imesh_prep_contours(
            &mut work,
            params.minz,
            params.maxz,
            increment_z,
            tolerance,
            params.flags & crate::imod::libimod::imesh::IMESH_MK_USE_MEAN != 0,
        ) != 0
    {
        return 1;
    }
    let status = imesh_skin_object(
        &mut work,
        scale,
        params.overlap as f64,
        params.cap,
        params.cap_skip_zlist.as_deref(),
        increment_z,
        params.flags,
        params.passes,
        params.tube_diameter as f64,
        callback,
    );
    if status == 0 {
        object.mesh = work.mesh;
    }
    status
}

/// Original static `getnextz` (`libmesh/skinobj.c`).
///
/// Returns the following listed Z value, or one past the final listed value
/// when the current value is final or absent; callers intentionally do not
/// treat the latter as an error.
pub fn next_mesh_z(z_list: &[i32], current_z: i32) -> i32 {
    assert!(
        !z_list.is_empty(),
        "native getnextz requires a nonempty Z list"
    );
    for index in 0..z_list.len() - 1 {
        if current_z == z_list[index] {
            return z_list[index + 1];
        }
    }
    z_list[z_list.len() - 1] + 1
}

/// Original static `segment_separation` (`libmesh/skinobj.c`).
///
/// Returns the native integer-truncated separation/overlap ranking metric for
/// two projected contour bounding segments.
pub fn segment_separation(lower_one: f32, upper_one: f32, lower_two: f32, upper_two: f32) -> f32 {
    let mut minimum_length = (upper_one - lower_one) as i32;
    if upper_two - lower_two < minimum_length as f32 {
        minimum_length = (upper_two - lower_two) as i32;
    }
    if (lower_one >= lower_two && upper_one <= upper_two)
        || (lower_two >= lower_one && upper_two <= upper_one)
    {
        return 0.;
    }
    if lower_one > upper_two {
        return (minimum_length as f32 + lower_one - upper_two);
    }
    if lower_two > upper_one {
        return (minimum_length as f32 + lower_two - upper_one);
    }
    if upper_one > upper_two {
        minimum_length = (upper_one - upper_two) as i32;
        if lower_one - lower_two < minimum_length as f32 {
            minimum_length = (lower_one - lower_two) as i32;
        }
    } else {
        minimum_length = (upper_two - upper_one) as i32;
        if lower_two - lower_one < minimum_length as f32 {
            minimum_length = (lower_two - lower_one) as i32;
        }
    }
    minimum_length as f32
}

/// Original static `addMeshToObject` (`libmesh/skinobj.c`).
pub fn add_mesh_to_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    contour: &crate::imod::libimod::imodel::Icont,
    mut mesh: crate::imod::libimod::imodel::Imesh,
    flags: u32,
) -> i32 {
    use crate::imod::libimod::imesh::{IMESH_MK_SURF, IMESH_MK_TIME, imodel_mesh_add};
    mesh.surf = if flags & IMESH_MK_SURF != 0 {
        contour.surf as i16
    } else {
        0
    };
    mesh.time = if flags & IMESH_MK_TIME != 0 {
        contour.time as i16
    } else {
        0
    };
    imodel_mesh_add(Some(&mesh), &mut object.mesh)
}

/// Original static `interpolate_point` (`libmesh/skinobj.c`).
pub fn interpolate_point(
    first: crate::imod::libimod::imodel::Ipoint,
    second: crate::imod::libimod::imodel::Ipoint,
    fraction: f32,
) -> crate::imod::libimod::imodel::Ipoint {
    crate::imod::libimod::imodel::Ipoint {
        x: fraction * second.x + (1. - fraction) * first.x,
        y: fraction * second.y + (1. - fraction) * first.y,
        z: fraction * second.z + (1. - fraction) * first.z,
    }
}

/// Original static `backoff_overlap` (`libmesh/skinobj.c`).
///
/// Removes the section of `first` inside `second`, retaining interpolated
/// points just outside both crossing edges in the source's one-tenth steps.
pub fn backoff_overlap(
    first: &mut crate::imod::libimod::imodel::Icont,
    second: &crate::imod::libimod::imodel::Icont,
) {
    use crate::imod::libimod::ipoint::{imod_point_add, imod_point_delete, imod_point_inside_cont};
    let count = first.pts.len();
    let Some(mut first_out) =
        (0..count).find(|&index| imod_point_inside_cont(second, &first.pts[index]) == 0)
    else {
        return;
    };
    let mut last_out = first_out;
    let mut first_in = last_out + 1;
    let mut turns = 0;
    while turns < count {
        if first_in == count {
            first_in = 0;
        }
        if imod_point_inside_cont(second, &first.pts[first_in]) != 0 {
            break;
        }
        last_out = first_in;
        first_in += 1;
        turns += 1;
    }
    if turns == count {
        return;
    }
    let mut before = None;
    for tenth in (2..=9).rev() {
        let point = interpolate_point(first.pts[last_out], first.pts[first_in], 0.1 * tenth as f32);
        if imod_point_inside_cont(second, &point) == 0 {
            before = Some(interpolate_point(
                first.pts[last_out],
                first.pts[first_in],
                0.1 * (tenth - 1) as f32,
            ));
            break;
        }
    }
    let mut last_in = first_in;
    first_out = last_in + 1;
    turns = 0;
    while turns < count {
        if first_out == count {
            first_out = 0;
        }
        if imod_point_inside_cont(second, &first.pts[first_out]) == 0 {
            break;
        }
        last_in = first_out;
        first_out += 1;
        turns += 1;
    }
    let mut after = None;
    for tenth in (2..=9).rev() {
        let point = interpolate_point(first.pts[first_out], first.pts[last_in], 0.1 * tenth as f32);
        if imod_point_inside_cont(second, &point) == 0 {
            after = Some(interpolate_point(
                first.pts[first_out],
                first.pts[last_in],
                0.1 * (tenth - 1) as f32,
            ));
            break;
        }
    }
    let mut insertion = last_out + 1;
    if let Some(point) = before {
        imod_point_add(first, Some(point), insertion as i32);
        insertion += 1;
    }
    if let Some(point) = after {
        imod_point_add(first, Some(point), insertion as i32);
    }
    let mut index = 0;
    while index < first.pts.len() {
        if imod_point_inside_cont(second, &first.pts[index]) != 0 {
            imod_point_delete(first, index as i32);
        } else {
            index += 1;
        }
    }
}

/// Original static `eliminate_overlap` (`libmesh/skinobj.c`).
///
/// Returns the native iteration count: zero/one when no further overlap is
/// detected before that pass, or one after the two allowed backoff passes.
pub fn eliminate_overlap(
    first: &mut crate::imod::libimod::imodel::Icont,
    second: &mut crate::imod::libimod::imodel::Icont,
) -> i32 {
    use crate::imod::libimod::icont::imodel_contour_overlap;
    for pass in 0..2 {
        if imodel_contour_overlap(first, second) == 0 {
            return pass;
        }
        let copy = first.clone();
        backoff_overlap(first, second);
        backoff_overlap(second, &copy);
    }
    1
}

/// Original static `isContConvexIfSimple` (`libmesh/skinobj.c`).
pub fn contour_convex_if_simple(contour: Option<&crate::imod::libimod::imodel::Icont>) -> i32 {
    let Some(contour) = contour else {
        return -1;
    };
    let mut all_positive = false;
    for index in 0..contour.pts.len() {
        let next = (index + 1) % contour.pts.len();
        let second = (next + 1) % contour.pts.len();
        let cross = (contour.pts[next].x - contour.pts[index].x)
            * (contour.pts[second].y - contour.pts[next].y)
            - (contour.pts[next].y - contour.pts[index].y)
                * (contour.pts[second].x - contour.pts[next].x);
        if index == 0 {
            all_positive = cross > 0.;
        }
        if (cross > 0. && !all_positive) || (cross < 0. && all_positive) {
            return 0;
        }
    }
    1
}

/// Original static `concavityAreaFraction` (`skinobj.c`).
/// Returns `(fraction_smaller_than_hull, hull_area_minus_contour_area)`.
pub fn concavity_area_fraction(
    contour: Option<&crate::imod::libimod::imodel::Icont>,
) -> (f32, f32) {
    let Some(contour) = contour else {
        return (-1., 0.);
    };
    if contour.pts.len() < 4 {
        return (0., 0.);
    }
    let area = |points: &[crate::imod::libimod::imodel::Ipoint]| -> f32 {
        0.5 * points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let next = points[(index + 1) % points.len()];
                point.x * next.y - point.y * next.x
            })
            .sum::<f32>()
            .abs()
    };
    let contour_area = area(&contour.pts);
    let mut points = contour.pts.clone();
    points.sort_by(compare_points_xy);
    points.dedup_by(|left, right| left.x == right.x && left.y == right.y);
    let cross = |origin: crate::imod::libimod::imodel::Ipoint,
                 first: crate::imod::libimod::imodel::Ipoint,
                 second: crate::imod::libimod::imodel::Ipoint| {
        (first.x - origin.x) * (second.y - origin.y) - (first.y - origin.y) * (second.x - origin.x)
    };
    let mut hull = Vec::new();
    for point in points.iter().copied() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0. {
            hull.pop();
        }
        hull.push(point);
    }
    let lower_len = hull.len();
    for point in points.iter().rev().copied().skip(1) {
        while hull.len() > lower_len
            && cross(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.
        {
            hull.pop();
        }
        hull.push(point);
    }
    if hull.len() > 1 {
        hull.pop();
    }
    if hull.len() < 3 {
        return (-1., 0.);
    }
    let hull_area = area(&hull);
    let difference = hull_area - contour_area;
    (
        if hull_area > 0. {
            difference / hull_area
        } else {
            0.
        },
        difference,
    )
}

/// Original static `segment_mm` (`skinobj.c`).
/// Returns `(xmin, xmax, ymin, ymax)` for the inclusive cyclic segment.
pub fn contour_segment_min_max(
    contour: Option<&crate::imod::libimod::imodel::Icont>,
    start: usize,
    end: usize,
) -> Option<(f32, f32, f32, f32)> {
    let contour = contour?;
    if contour.pts.is_empty() || start >= contour.pts.len() || end >= contour.pts.len() {
        return None;
    }
    let mut index = start;
    let mut xmin = contour.pts[index].x;
    let mut xmax = xmin;
    let mut ymin = contour.pts[index].y;
    let mut ymax = ymin;
    loop {
        let point = contour.pts[index];
        xmin = xmin.min(point.x);
        xmax = xmax.max(point.x);
        ymin = ymin.min(point.y);
        ymax = ymax.max(point.y);
        if index == end {
            break;
        }
        index = (index + 1) % contour.pts.len();
    }
    Some((xmin, xmax, ymin, ymax))
}

/// Original static `cross_cont` (`skinobj.c`).
/// Tests whether a 2-D segment crosses a contour edge excluding edges incident
/// to either supplied endpoint index.
pub fn segment_crosses_contour(
    contour: Option<&crate::imod::libimod::imodel::Icont>,
    x_start: f32,
    y_start: f32,
    x_end: f32,
    y_end: f32,
    skip_one: usize,
    skip_two: usize,
) -> i32 {
    let Some(contour) = contour else {
        return 0;
    };
    if contour.pts.is_empty() {
        return 0;
    }
    let dx_one = x_end - x_start;
    let dy_one = y_end - y_start;
    for point in 0..contour.pts.len() {
        let next = (point + 1) % contour.pts.len();
        if point == skip_one || point == skip_two || next == skip_one || next == skip_two {
            continue;
        }
        let edge_start = contour.pts[point];
        let edge_end = contour.pts[next];
        let dx_two = edge_start.x - edge_end.x;
        let dy_two = edge_start.y - edge_end.y;
        let dx_start = edge_start.x - x_start;
        let dy_start = edge_start.y - y_start;
        let denominator = dx_one * dy_two - dy_one * dx_two;
        let t_numerator = dx_start * dy_two - dy_start * dx_two;
        let u_numerator = dx_one * dy_start - dy_one * dx_start;
        if (denominator < 0.
            && t_numerator <= 0.
            && u_numerator <= 0.
            && t_numerator >= denominator
            && u_numerator >= denominator)
            || (denominator >= 0.
                && t_numerator >= 0.
                && u_numerator >= 0.
                && t_numerator <= denominator
                && u_numerator <= denominator)
        {
            return 1;
        }
    }
    0
}

/// Original static `check_legal_joiner` (`skinobj.c`).
///
/// Determines whether a proposed connector can join contours without crossing
/// either endpoint contour or violating the nesting relation on the adjacent
/// slice.  `other_indices[..other_same_count]` names same-level contours in
/// `object`; the remainder names inner/outer levels, exactly as constructed by
/// the native nesting pass.  Invalid indices are ignored rather than turning a
/// corrupt model into a Rust bounds panic.
pub fn check_legal_joiner(
    point_one: crate::imod::libimod::imodel::Ipoint,
    point_two: crate::imod::libimod::imodel::Ipoint,
    contour_one: &crate::imod::libimod::imodel::Icont,
    contour_two: &crate::imod::libimod::imodel::Icont,
    same_level: bool,
    object: &crate::imod::libimod::imodel::Iobj,
    other_indices: &[i32],
    other_same_count: usize,
    other: Option<&crate::imod::libimod::imodel::Icont>,
) -> i32 {
    use crate::imod::libimod::ipoint::imod_point_inside_cont;
    const CHECK_MIN_FRACTION: f32 = 0.45;
    const CHECK_MIN_DISTANCE: f32 = 2.0;
    const INSIDE_MAX_DISTANCE: f32 = 0.1;
    const INSIDE_MAX_FRACTION: f32 = 0.1;
    let dx = point_one.x - point_two.x;
    let dy = point_one.y - point_two.y;
    let length = (dx * dx + dy * dy).sqrt();
    if length > 1.0e-5 {
        let fraction = (INSIDE_MAX_DISTANCE / length).min(INSIDE_MAX_FRACTION);
        let start_x = (1. - fraction) * point_one.x + fraction * point_two.x;
        let start_y = (1. - fraction) * point_one.y + fraction * point_two.y;
        let end_x = (1. - fraction) * point_two.x + fraction * point_one.x;
        let end_y = (1. - fraction) * point_two.y + fraction * point_one.y;
        if segment_crosses_contour(
            Some(contour_one),
            start_x,
            start_y,
            end_x,
            end_y,
            usize::MAX,
            usize::MAX,
        ) != 0
            || segment_crosses_contour(
                Some(contour_two),
                start_x,
                start_y,
                end_x,
                end_y,
                usize::MAX,
                usize::MAX,
            ) != 0
        {
            return 0;
        }
    }
    let min_fraction = if length > 1.0e-5 {
        (CHECK_MIN_DISTANCE / length).max(CHECK_MIN_FRACTION)
    } else {
        CHECK_MIN_FRACTION
    }
    .min(0.5);
    let midpoint = crate::imod::libimod::imodel::Ipoint {
        x: (1. - min_fraction) * point_one.x + min_fraction * point_two.x,
        y: (1. - min_fraction) * point_one.y + min_fraction * point_two.y,
        z: (1. - min_fraction) * point_one.z + min_fraction * point_two.z,
    };
    let same_count = other_same_count.min(other_indices.len());
    let contains = |index: i32| -> bool {
        usize::try_from(index)
            .ok()
            .and_then(|index| object.cont.get(index))
            .is_some_and(|contour| imod_point_inside_cont(contour, &midpoint) != 0)
    };
    let initial = if same_level { 1 } else { 0 };
    if other.is_some_and(|contour| imod_point_inside_cont(contour, &midpoint) != 0)
        || other_indices[..same_count].iter().copied().any(contains)
    {
        return initial;
    }
    let inverse = 1 - initial;
    // This scan is intentionally retained even though CHECK_DIVISIONS is zero:
    // an other-level hit returns `inverse` in the C code as well.
    let _inside_other_level = other_indices[same_count..].iter().copied().any(contains);
    inverse
}

/// A closest point from one contour vertex to an edge of another contour.
/// This is the value accumulated by native `scan_points_to_segments`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContourSegmentCandidate {
    pub point_index: usize,
    pub segment_index: usize,
    pub t: f32,
    pub close: crate::imod::libimod::imodel::Ipoint,
    pub distance_squared: f32,
}

/// Original static `scan_points_to_segments` (`skinobj.c`).
///
/// The C implementation partitions both contours into bounding boxes before
/// visiting the same candidates.  This direct iterator retains the exact 2-D
/// projection and endpoint/wrap semantics while making the two source outputs
/// explicit: the nearest legal connector and the nearest connector regardless
/// of legality.  The latter is the source routine's fallback when no legal
/// candidate has been found.
pub fn scan_points_to_segments(
    contour_one: &crate::imod::libimod::imodel::Icont,
    contour_two: &crate::imod::libimod::imodel::Icont,
    check_legality: bool,
    same_level: bool,
    object: &crate::imod::libimod::imodel::Iobj,
    other_indices: &[i32],
    other_same_count: usize,
    other: Option<&crate::imod::libimod::imodel::Icont>,
) -> (
    Option<ContourSegmentCandidate>,
    Option<ContourSegmentCandidate>,
) {
    if contour_one.pts.is_empty() || contour_two.pts.is_empty() {
        return (None, None);
    }
    let mut legal: Option<ContourSegmentCandidate> = None;
    let mut fallback: Option<ContourSegmentCandidate> = None;
    for (point_index, point) in contour_one.pts.iter().copied().enumerate() {
        for segment_index in 0..contour_two.pts.len() {
            let start = contour_two.pts[segment_index];
            let end = contour_two.pts[(segment_index + 1) % contour_two.pts.len()];
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let t = if dx != 0. || dy != 0. {
                (((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy))
                    .clamp(0., 1.)
            } else {
                0.
            };
            let close = crate::imod::libimod::imodel::Ipoint {
                x: start.x + t * dx,
                y: start.y + t * dy,
                z: start.z,
            };
            let distance_squared = (close.x - point.x).powi(2) + (close.y - point.y).powi(2);
            let candidate = ContourSegmentCandidate {
                point_index,
                segment_index,
                t,
                close,
                distance_squared,
            };
            if fallback.is_none_or(|best| distance_squared < best.distance_squared) {
                fallback = Some(candidate);
            }
            let permitted = !check_legality
                || check_legal_joiner(
                    point,
                    close,
                    contour_one,
                    contour_two,
                    same_level,
                    object,
                    other_indices,
                    other_same_count,
                    other,
                ) != 0;
            if permitted && legal.is_none_or(|best| distance_squared < best.distance_squared) {
                legal = Some(candidate);
            }
        }
    }
    (legal, fallback)
}

/// The selected connection returned by `find_closest_contour`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestContourMatch {
    /// Position in the native caller's `list` (one for a direct `onecont`).
    pub list_position: usize,
    /// Object contour index when the candidate came from `list`.
    pub contour_index: Option<usize>,
    /// True when `point_one_index` belongs to the accumulated contour.
    pub first_contour_point: bool,
    pub point_one_index: usize,
    pub point_two_index: usize,
    pub t: f32,
    pub close: crate::imod::libimod::imodel::Ipoint,
    pub distance_squared: f32,
}

/// Original static `find_closest_contour` (`skinobj.c`).
///
/// Selects a direct connector if one was designated in the stores; otherwise
/// considers point-to-segment approaches in both directions.  It first
/// preserves the source's nearest-distance choice, validates that connector,
/// then falls back to the nearest legal candidate when nesting disallows it.
pub fn find_closest_contour(
    target: &crate::imod::libimod::imodel::Icont,
    one_contour: Option<&crate::imod::libimod::imodel::Icont>,
    object: &crate::imod::libimod::imodel::Iobj,
    list: &[i32],
    used: &[bool],
    other_indices: &[i32],
    other_same_count: usize,
    other: Option<&crate::imod::libimod::imodel::Icont>,
    same_level_count: usize,
) -> Option<ClosestContourMatch> {
    #[derive(Clone, Copy)]
    struct Pair<'a> {
        position: usize,
        index: Option<usize>,
        contour: &'a crate::imod::libimod::imodel::Icont,
        same_level: bool,
    }
    let mut choices = Vec::new();
    if let Some(contour) = one_contour {
        choices.push(Pair {
            position: 1,
            index: None,
            contour,
            same_level: true,
        });
    } else {
        for (position, &raw_index) in list.iter().enumerate().skip(1) {
            if used.get(position).copied().unwrap_or(false) {
                continue;
            }
            let Ok(index) = usize::try_from(raw_index) else {
                continue;
            };
            let Some(contour) = object.cont.get(index) else {
                continue;
            };
            choices.push(Pair {
                position,
                index: Some(index),
                contour,
                same_level: position < same_level_count,
            });
        }
    }
    let mut unrestricted: Option<ClosestContourMatch> = None;
    let mut designated: Option<ClosestContourMatch> = None;
    for choice in &choices {
        if let Some(connector) = make_connectors(target, choice.contour, true, 0)
            .and_then(|mut items| items.drain(..).next())
        {
            let point_one_index = usize::try_from(connector.b1).ok()?;
            let point_two_index = usize::try_from(connector.t1).ok()?;
            let close = *choice.contour.pts.get(point_two_index)?;
            designated = Some(ClosestContourMatch {
                list_position: choice.position,
                contour_index: choice.index,
                first_contour_point: true,
                point_one_index,
                point_two_index,
                t: 0.,
                close,
                distance_squared: 0.,
            });
            break;
        }
        let (forward, _) = scan_points_to_segments(
            target,
            choice.contour,
            false,
            false,
            object,
            other_indices,
            other_same_count,
            other,
        );
        if let Some(candidate) = forward {
            let value = ClosestContourMatch {
                list_position: choice.position,
                contour_index: choice.index,
                first_contour_point: true,
                point_one_index: candidate.point_index,
                point_two_index: candidate.segment_index,
                t: candidate.t,
                close: candidate.close,
                distance_squared: candidate.distance_squared,
            };
            if unrestricted.is_none_or(|best| value.distance_squared < best.distance_squared) {
                unrestricted = Some(value);
            }
        }
        let (reverse, _) = scan_points_to_segments(
            choice.contour,
            target,
            false,
            false,
            object,
            other_indices,
            other_same_count,
            other,
        );
        if let Some(candidate) = reverse {
            let value = ClosestContourMatch {
                list_position: choice.position,
                contour_index: choice.index,
                first_contour_point: false,
                point_one_index: candidate.segment_index,
                point_two_index: candidate.point_index,
                t: candidate.t,
                close: candidate.close,
                distance_squared: candidate.distance_squared,
            };
            if unrestricted.is_none_or(|best| value.distance_squared < best.distance_squared) {
                unrestricted = Some(value);
            }
        }
    }
    if designated.is_some() {
        unrestricted = designated;
    }
    let nearest = unrestricted?;
    let selected = choices
        .iter()
        .find(|choice| choice.position == nearest.list_position)?;
    let target_point = if nearest.first_contour_point {
        target.pts.get(nearest.point_one_index)?
    } else {
        &nearest.close
    };
    let joined_point = if nearest.first_contour_point {
        &nearest.close
    } else {
        selected.contour.pts.get(nearest.point_two_index)?
    };
    if check_legal_joiner(
        *target_point,
        *joined_point,
        target,
        selected.contour,
        selected.same_level,
        object,
        other_indices,
        other_same_count,
        other,
    ) != 0
    {
        return Some(nearest);
    }

    let mut legal: Option<ClosestContourMatch> = None;
    for choice in &choices {
        let (forward, _) = scan_points_to_segments(
            target,
            choice.contour,
            true,
            choice.same_level,
            object,
            other_indices,
            other_same_count,
            other,
        );
        if let Some(candidate) = forward {
            let value = ClosestContourMatch {
                list_position: choice.position,
                contour_index: choice.index,
                first_contour_point: true,
                point_one_index: candidate.point_index,
                point_two_index: candidate.segment_index,
                t: candidate.t,
                close: candidate.close,
                distance_squared: candidate.distance_squared,
            };
            if legal.is_none_or(|best| value.distance_squared < best.distance_squared) {
                legal = Some(value);
            }
        }
        let (reverse, _) = scan_points_to_segments(
            choice.contour,
            target,
            true,
            choice.same_level,
            object,
            other_indices,
            other_same_count,
            other,
        );
        if let Some(candidate) = reverse {
            let value = ClosestContourMatch {
                list_position: choice.position,
                contour_index: choice.index,
                first_contour_point: false,
                point_one_index: candidate.segment_index,
                point_two_index: candidate.point_index,
                t: candidate.t,
                close: candidate.close,
                distance_squared: candidate.distance_squared,
            };
            if legal.is_none_or(|best| value.distance_squared < best.distance_squared) {
                legal = Some(value);
            }
        }
    }
    legal.or(Some(nearest))
}

/// Original static `join_all_contours` (`skinobj.c`).
///
/// Joins the contours named by `list` into one contour, choosing each next
/// contour by legal closest approach.  Object/contour draw-property promotion
/// belongs to the surrounding skinning pass; the resulting contour preserves
/// its point-store edits through `imod_contour_join`.
pub fn join_all_contours(
    object: &crate::imod::libimod::imodel::Iobj,
    list: &[i32],
    same_level_count: usize,
    fill: i32,
    other_indices: &[i32],
    other_same_count: usize,
    other: Option<&crate::imod::libimod::imodel::Icont>,
) -> Option<crate::imod::libimod::imodel::Icont> {
    use crate::imod::libimod::icont::imod_contour_join;
    use crate::imod::libimod::ipoint::imod_point_add;
    let (&first, _) = list.split_first()?;
    let mut other_for_checks = other;
    let mut joined = if first >= 0 {
        object.cont.get(usize::try_from(first).ok()?)?.clone()
    } else {
        // The native caller uses `other` itself as the initial contour and
        // clears its special adjacent-slice meaning thereafter.
        other_for_checks = None;
        other?.clone()
    };
    if list.len() < 2 {
        return Some(joined);
    }
    let mut used = vec![false; list.len()];
    for _ in 1..list.len() {
        let mut selected = find_closest_contour(
            &joined,
            None,
            object,
            list,
            &used,
            other_indices,
            other_same_count,
            other_for_checks,
            same_level_count,
        )?;
        let object_index = selected.contour_index?;
        let mut next = object.cont.get(object_index)?.clone();
        if selected.list_position < same_level_count
            && eliminate_overlap(&mut joined, &mut next) != 0
        {
            let selected_position = selected.list_position;
            let mut projection = find_closest_contour(
                &joined,
                Some(&next),
                object,
                list,
                &used,
                other_indices,
                other_same_count,
                other_for_checks,
                same_level_count,
            )?;
            // `onecont` is evaluated in a synthetic two-entry list by the C
            // helper, but it still joins the originally selected object item.
            projection.list_position = selected_position;
            projection.contour_index = Some(object_index);
            selected = projection;
        }
        if selected.t > 0. && selected.t < 1. {
            if selected.first_contour_point {
                let insertion = selected.point_two_index.checked_add(1)?;
                if imod_point_add(&mut next, Some(selected.close), insertion as i32) == 0 {
                    return None;
                }
            } else {
                let insertion = selected.point_one_index.checked_add(1)?;
                if imod_point_add(&mut joined, Some(selected.close), insertion as i32) == 0 {
                    return None;
                }
            }
        }
        let counter_direction = i32::from(selected.list_position >= same_level_count);
        joined = imod_contour_join(
            Some(&mut joined),
            Some(&mut next),
            selected.point_one_index as i32,
            selected.point_two_index as i32,
            fill,
            counter_direction,
        )?;
        used[selected.list_position] = true;
    }
    Some(joined)
}

/// Original static `subtract_scan_contours` (`skinobj.c`).
///
/// Scan contours encode each horizontal interval as two consecutive points.
/// Removes from `first` the area represented by `second`, preserving point
/// store indices through the standard contour point edit helpers.
pub fn subtract_scan_contours(
    first: &mut crate::imod::libimod::imodel::Icont,
    second: &crate::imod::libimod::imodel::Icont,
) {
    use crate::imod::libimod::ipoint::{imod_point_add, imod_point_delete};
    let mut first_index = 0usize;
    let mut second_start = 0usize;
    while first_index + 1 < first.pts.len() {
        let mut second_index = second_start;
        let mut removed = false;
        while second_index + 1 < second.pts.len() {
            let first_start = first.pts[first_index];
            let first_end = first.pts[first_index + 1];
            let second_start_point = second.pts[second_index];
            let second_end_point = second.pts[second_index + 1];
            if first_start.y == second_start_point.y {
                if first_start.x < second_end_point.x && second_start_point.x < first_end.x {
                    if first_start.x < second_start_point.x && first_end.x <= second_end_point.x {
                        first.pts[first_index + 1].x = second_start_point.x;
                    } else if first_start.x >= second_start_point.x
                        && first_end.x > second_end_point.x
                    {
                        first.pts[first_index].x = second_end_point.x;
                    } else if first_start.x >= second_start_point.x
                        && first_end.x < second_end_point.x
                    {
                        imod_point_delete(first, first_index as i32);
                        imod_point_delete(first, first_index as i32);
                        removed = true;
                        break;
                    } else {
                        // The second interval is strictly inside the first:
                        // split it into the two surviving horizontal spans.
                        let _ = imod_point_add(
                            first,
                            Some(second_start_point),
                            (first_index + 1) as i32,
                        );
                        let _ =
                            imod_point_add(first, Some(second_end_point), (first_index + 2) as i32);
                    }
                }
            } else if first_start.y > second_start_point.y {
                second_start = second_index;
            } else {
                break;
            }
            second_index += 2;
        }
        if !removed {
            first_index += 2;
        }
    }
}

/// Original static `break_contour_inout` (`skinobj.c`).
///
/// Splits a contour along two points for nested in/out skinning.  Unlike the
/// general contour-break operation, optional fill inserts the same displaced
/// midpoint in both results, and a reversed input pair reverses the returned
/// contour order.
pub fn break_contour_inout(
    contour: Option<&crate::imod::libimod::imodel::Icont>,
    first_point: i32,
    second_point: i32,
    fill: i32,
) -> Option<(
    crate::imod::libimod::imodel::Icont,
    crate::imod::libimod::imodel::Icont,
)> {
    use crate::imod::libimod::imodel::{ICONT_OPEN, Icont};
    use crate::imod::libimod::istore::istore_extract_changes;
    let contour = contour?;
    let mut first_point = usize::try_from(first_point).ok()?;
    let mut second_point = usize::try_from(second_point).ok()?;
    if first_point >= contour.pts.len() || second_point >= contour.pts.len() {
        return None;
    }
    let mut reversed = false;
    if second_point < first_point {
        std::mem::swap(&mut first_point, &mut second_point);
        reversed = true;
    }
    let mut midpoint = crate::imod::libimod::imodel::Ipoint {
        x: (contour.pts[first_point].x + contour.pts[second_point].x) * 0.5,
        y: (contour.pts[first_point].y + contour.pts[second_point].y) * 0.5,
        z: (contour.pts[first_point].z + contour.pts[second_point].z) * 0.5,
    };
    if fill == 1 {
        midpoint.z += 0.75;
    }
    if fill == -1 {
        midpoint.z -= 0.75;
    }
    let mut first = Icont::default();
    let mut second = Icont::default();
    if contour.flags & ICONT_OPEN != 0 {
        first.flags |= ICONT_OPEN;
    }
    first.pts.extend_from_slice(&contour.pts[..=first_point]);
    if fill != 0 {
        first.pts.push(midpoint);
    }
    first.pts.extend_from_slice(&contour.pts[second_point..]);
    if fill != 0 {
        second.pts.push(midpoint);
    }
    second
        .pts
        .extend_from_slice(&contour.pts[first_point..=second_point]);
    let size = contour.pts.len() as i32;
    let first_point = first_point as i32;
    let second_point = second_point as i32;
    if istore_extract_changes(&contour.store, &mut first.store, 0, first_point, 0, size) != 0
        || (fill != 0
            && istore_extract_changes(
                &contour.store,
                &mut first.store,
                first_point,
                first_point,
                first_point + 1,
                size,
            ) != 0)
        || istore_extract_changes(
            &contour.store,
            &mut first.store,
            second_point,
            size - 1,
            first_point + if fill != 0 { 2 } else { 1 },
            size,
        ) != 0
        || (fill != 0
            && istore_extract_changes(
                &contour.store,
                &mut second.store,
                first_point,
                first_point,
                0,
                size,
            ) != 0)
        || istore_extract_changes(
            &contour.store,
            &mut second.store,
            first_point,
            second_point,
            if fill != 0 { 1 } else { 0 },
            size,
        ) != 0
    {
        return None;
    }
    if reversed {
        Some((second, first))
    } else {
        Some((first, second))
    }
}

/// Original static `smoothReduceSkeleton` (`skinobj.c`).
///
/// Smooths runs between paired pixel-boundary points with overlapping local
/// quadratic fits, reduces each run, then removes the duplicated return path
/// characteristic of a skeleton contour.
pub fn smooth_reduce_skeleton(contour: Option<&mut crate::imod::libimod::imodel::Icont>) -> i32 {
    use crate::imod::libcfshr::simplestat::ls_fit2;
    use crate::imod::libimod::icont::{imod_contour_reduce, imod_contour_unique};
    let Some(contour) = contour else {
        return 1;
    };
    let count = contour.pts.len();
    if count < 5 {
        return 1;
    }
    let mut paired = vec![-1isize; count];
    let mut base = 0usize;
    while base + 1 < count {
        if paired[base] >= 0 {
            base += 1;
            continue;
        }
        let mut found = count;
        for index in base + 1..count {
            if paired[index] < 0
                && (contour.pts[base].x - contour.pts[index].x).abs() < 0.01
                && (contour.pts[base].y - contour.pts[index].y).abs() < 0.01
            {
                found = index;
                break;
            }
        }
        if found < count {
            paired[base] = found as isize;
            paired[found] = base as isize;
            while base + 1 < found.saturating_sub(1)
                && paired[base + 1] < 0
                && paired[found - 1] < 0
                && (contour.pts[base + 1].x - contour.pts[found - 1].x).abs() < 0.01
                && (contour.pts[base + 1].y - contour.pts[found - 1].y).abs() < 0.01
            {
                base += 1;
                found -= 1;
                paired[base] = found as isize;
                paired[found] = base as isize;
            }
        }
        base += 1;
    }
    base = 0;
    while base + 1 < count {
        if !(paired[base + 1] >= 0 && (paired[base] == -1 || paired[base] != paired[base + 1] + 1))
        {
            base += 1;
            continue;
        }
        let mut end_break = count;
        for index in base + 2..count {
            if paired[index] < 0 || paired[index] != paired[index - 1] - 1 {
                end_break = index;
                break;
            }
        }
        if end_break == count {
            end_break = count - 1;
        }
        let mut segment_start = base + 1;
        let mut segment_end = end_break - 1;
        if paired[base] == -1 && paired[segment_start] == ((base + count - 1) % count) as isize {
            segment_start -= 1;
        }
        if paired[end_break] == -1 && paired[segment_end] == (end_break + 1) as isize {
            segment_end += 1;
        }
        let segment_size = segment_end + 1 - segment_start;
        if segment_size < 3 {
            base += 1;
            continue;
        }
        if segment_size >= 4 {
            for fit_start in 0..=segment_size.saturating_sub(5) {
                let fit_end = (fit_start + 4).min(segment_size - 1);
                let fit_size = fit_end + 1 - fit_start;
                let start = contour.pts[segment_start + fit_start];
                let finish = contour.pts[segment_start + fit_end];
                let angle = (finish.y - start.y).atan2(finish.x - start.x);
                let (cosine, sine) = (angle.cos(), angle.sin());
                let mut xrot = Vec::with_capacity(fit_size);
                let mut yrot = Vec::with_capacity(fit_size);
                for point in &contour.pts[segment_start + fit_start..=segment_start + fit_end] {
                    xrot.push(cosine * point.x + sine * point.y);
                    yrot.push(-sine * point.x + cosine * point.y);
                }
                let x: Vec<f32> = xrot.iter().map(|value| *value - xrot[0]).collect();
                let xsquared: Vec<f32> = x.iter().map(|value| value * value).collect();
                let (mut a, mut b, mut c) = (0., 0., 0.);
                ls_fit2(
                    &x,
                    &xsquared,
                    &yrot,
                    fit_size as i32,
                    &mut a,
                    &mut b,
                    Some(&mut c),
                );
                let mut fill_start = fit_size / 2;
                let mut fill_end = fill_start;
                if fit_start == 0 {
                    fill_start = 1;
                }
                if fit_end == segment_size - 1 {
                    fill_end = fit_size - 2;
                }
                for fit_index in fill_start..=fill_end {
                    let yfit = a * x[fit_index] + b * xsquared[fit_index] + c;
                    let index = segment_start + fit_start + fit_index;
                    contour.pts[index].x = cosine * xrot[fit_index] - sine * yfit;
                    contour.pts[index].y = sine * xrot[fit_index] + cosine * yfit;
                    if let Ok(pair) = usize::try_from(paired[index]) {
                        contour.pts[pair] = contour.pts[index];
                    }
                }
            }
        }
        let mut reduced = crate::imod::libimod::imodel::Icont {
            pts: contour.pts[segment_start..=segment_end].to_vec(),
            ..Default::default()
        };
        imod_contour_reduce(Some(&mut reduced), 0.25);
        for index in segment_start + 1..segment_end {
            let replacement = reduced.pts[(index - segment_start).min(reduced.pts.len() - 1)];
            contour.pts[index] = replacement;
            if let Ok(pair) = usize::try_from(paired[index]) {
                contour.pts[pair] = replacement;
            }
        }
        for index in base + 1..end_break {
            if let Ok(pair) = usize::try_from(paired[index]) {
                paired[pair] = -2;
            }
            paired[index] = -2;
        }
        base = segment_end;
    }
    imod_contour_unique(contour);
    0
}

/// Original static `evaluate_break` (`skinobj.c`).
///
/// Scores a possible in/out break in an outer contour.  `orphan_scans` and
/// their corresponding bounds are the caller's cache indexed by
/// `just_inside`; scan conversion performed by overlap testing is retained in
/// that cache just as in the native routine.
pub fn evaluate_break(
    outer: &crate::imod::libimod::imodel::Icont,
    used: &[bool],
    just_inside: &[usize],
    orphan_scans: &mut [crate::imod::libimod::imodel::Icont],
    orphan_bounds: &[(
        crate::imod::libimod::imodel::Ipoint,
        crate::imod::libimod::imodel::Ipoint,
    )],
    first_point: usize,
    second_point: usize,
    current_minimum: f32,
    path_length: f32,
    full_area: f32,
    join_direction: i32,
    area_only: bool,
) -> (f32, f32) {
    use crate::imod::libimod::icont::{
        imod_contour_area, imod_contour_get_bbox, imodel_contour_scan, imodel_overlap_fractions,
    };
    const INVALID: f32 = 1.0e20;
    const FAILURE: f32 = 1.0e30;
    let (Some(start), Some(end)) = (outer.pts.get(first_point), outer.pts.get(second_point)) else {
        return (FAILURE, 0.);
    };
    let distance =
        ((start.x - end.x).powi(2) + (start.y - end.y).powi(2) + (start.z - end.z).powi(2)).sqrt();
    let maximum_area = (distance + path_length).powi(2) / 12.5664;
    let mut ratio = if maximum_area > 0. {
        distance / maximum_area.powi(2)
    } else {
        INVALID
    };
    if (!area_only && ratio > current_minimum) || (area_only && maximum_area < current_minimum) {
        return (ratio, maximum_area);
    }
    if segment_crosses_contour(
        Some(outer),
        start.x,
        start.y,
        end.x,
        end.y,
        first_point,
        second_point,
    ) != 0
    {
        return (INVALID, 0.);
    }
    let Some((first, second)) = break_contour_inout(
        Some(outer),
        first_point as i32,
        second_point as i32,
        -join_direction,
    ) else {
        return (FAILURE, 0.);
    };
    let first_area = imod_contour_area(Some(&first));
    let second_area = imod_contour_area(Some(&second));
    let (inner, inner_area) = if first_area > full_area && second_area < first_area {
        (second, second_area)
    } else if second_area > full_area {
        (first, first_area)
    } else {
        return (INVALID, 0.);
    };
    ratio = if inner_area > 0. {
        distance / inner_area.powi(2)
    } else {
        INVALID
    };
    if (!area_only && ratio > current_minimum) || (area_only && inner_area < current_minimum) {
        return (ratio, inner_area);
    }
    let Some(mut inner_scan) = imodel_contour_scan(Some(&inner)) else {
        return (FAILURE, 0.);
    };
    let mut inner_minimum = crate::imod::libimod::imodel::Ipoint::default();
    let mut inner_maximum = crate::imod::libimod::imodel::Ipoint::default();
    imod_contour_get_bbox(Some(&inner), &mut inner_minimum, &mut inner_maximum);
    let mut area_sum = 0.;
    for &index in just_inside {
        if used.get(index).copied().unwrap_or(false) {
            continue;
        }
        let (Some(scan), Some(&(minimum, maximum))) =
            (orphan_scans.get_mut(index), orphan_bounds.get(index))
        else {
            continue;
        };
        let (mut first_fraction, mut second_fraction) = (0., 0.);
        imodel_overlap_fractions(
            scan,
            minimum,
            maximum,
            &mut inner_scan,
            inner_minimum,
            inner_maximum,
            &mut first_fraction,
            &mut second_fraction,
        );
        area_sum += second_fraction * inner_area;
    }
    ratio = if area_sum > 0. {
        distance / area_sum.powi(2)
    } else {
        INVALID
    };
    (ratio, area_sum)
}

/// Original static `add_whole_nest` (`skinobj.c`).
///
/// Finds the level-one enclosing nest, then appends its primary and inside
/// contours once.  `flag` is the caller's temporary contour-flag bit.
pub fn add_whole_nest(
    nest_index: usize,
    object: &mut crate::imod::libimod::imodel::Iobj,
    nests: &[crate::imod::libimod::icont::Nesting],
    nest_indices: &[i32],
    flag: u32,
    target_list: &mut Vec<i32>,
) {
    let Some(mut nest) = nests.get(nest_index) else {
        return;
    };
    for &outside in &nest.outside {
        let Some(&raw_index) = usize::try_from(outside)
            .ok()
            .and_then(|index| nest_indices.get(index))
        else {
            continue;
        };
        let Some(candidate) = usize::try_from(raw_index)
            .ok()
            .and_then(|index| nests.get(index))
        else {
            continue;
        };
        if candidate.level == 1 {
            nest = candidate;
            break;
        }
    }
    let mut append = |contour_index: i32| {
        let Some(contour) = usize::try_from(contour_index)
            .ok()
            .and_then(|index| object.cont.get_mut(index))
        else {
            return;
        };
        if contour.flags & flag == 0 {
            contour.flags |= flag;
            target_list.push(contour_index);
        }
    };
    append(nest.co);
    for &inside in &nest.inside {
        append(inside);
    }
}

/// Original static `robustCenterOfMass` (`libmesh/skinobj.c`).
///
/// Uses a scan contour when supplied and falls back to the geometric bounding
/// box center of the original contour if the native center-of-mass routine
/// reports failure.
pub fn robust_center_of_mass(
    contour: &mut crate::imod::libimod::imodel::Icont,
    scan_contour: Option<&mut crate::imod::libimod::imodel::Icont>,
    center: &mut crate::imod::libimod::imodel::Ipoint,
) -> i32 {
    use crate::imod::libimod::icont::{imod_contour_center_of_mass, imod_contour_get_bbox};
    if imod_contour_center_of_mass(scan_contour.or(Some(contour)), center) != 0 {
        let mut lower = crate::imod::libimod::imodel::Ipoint::default();
        let mut upper = crate::imod::libimod::imodel::Ipoint::default();
        imod_contour_get_bbox(Some(contour), &mut lower, &mut upper);
        center.x = (lower.x + upper.x) / 2.;
        center.y = (lower.y + upper.y) / 2.;
        center.z = (lower.z + upper.z) / 2.;
        return 1;
    }
    0
}

/// Object-level unconnected branch of `mesh_contours` / `imeshContoursCost`.
///
/// This commits the fully translated geometric contour-pair path for pairs
/// that have already been classified as unconnected.  Connector/nesting
/// dispatch remains in the higher `skinobj.c` orchestration layer.
pub fn mesh_unconnected_contours_to_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    bottom: &crate::imod::libimod::imodel::Icont,
    top: &crate::imod::libimod::imodel::Icont,
    scale: &crate::imod::libimod::imodel::Ipoint,
    mut inside: bool,
    flags: u32,
) -> i32 {
    use crate::imod::libimod::icont::imod_cont_z_direction;
    use crate::imod::libimod::imodel::IMOD_OBJFLAG_OUT;
    let mut bottom = bottom.clone();
    let mut top = top.clone();
    if bottom.pts.len() == 1 {
        bottom.pts.push(bottom.pts[0]);
    }
    if top.pts.len() == 1 {
        top.pts.push(top.pts[0]);
    }
    if object.flags & IMOD_OBJFLAG_OUT != 0 {
        inside = !inside;
    }
    let mut direction = [
        imod_cont_z_direction(Some(&bottom)),
        imod_cont_z_direction(Some(&top)),
    ];
    if direction[0] == 0 {
        direction[0] = 1;
    }
    if direction[1] == 0 {
        direction[1] = 1;
    }
    let open = crate::imod::libimod::iobj::iobj_close(object.flags) == 0;
    let Some(mesh) =
        mesh_unconnected_contour_pair(&bottom, &top, scale, inside, direction, [0, 0], open)
    else {
        return -1;
    };
    add_mesh_to_object(object, &bottom, mesh, flags)
}
/// Native static `mesh_contours` (`skinobj.c`).
///
/// Unlike the flag-oriented public pairing wrapper, the complex skinning
/// path supplies the selected surface and time explicitly after joining a
/// group of contours.  This preserves that native ownership boundary.
pub fn mesh_contours(
    object: &mut crate::imod::libimod::imodel::Iobj,
    bottom: &crate::imod::libimod::imodel::Icont,
    top: &crate::imod::libimod::imodel::Icont,
    surface: i32,
    time: i32,
    scale: &crate::imod::libimod::imodel::Ipoint,
    inside: bool,
) -> i32 {
    let mut bottom = bottom.clone();
    let mut top = top.clone();
    if bottom.pts.len() == 1 {
        bottom.pts.push(bottom.pts[0]);
    }
    if top.pts.len() == 1 {
        top.pts.push(top.pts[0]);
    }
    let mut direction = [
        crate::imod::libimod::icont::imod_cont_z_direction(Some(&bottom)),
        crate::imod::libimod::icont::imod_cont_z_direction(Some(&top)),
    ];
    for value in &mut direction {
        if *value == 0 {
            *value = 1;
        }
    }
    let inside = if object.flags & crate::imod::libimod::imodel::IMOD_OBJFLAG_OUT != 0 {
        !inside
    } else {
        inside
    };
    let open = crate::imod::libimod::iobj::iobj_close(object.flags) == 0;
    let Some(mut mesh) =
        mesh_unconnected_contour_pair(&bottom, &top, scale, inside, direction, [0, 0], open)
    else {
        return -1;
    };
    mesh.surf = surface as i16;
    mesh.time = time as i16;
    crate::imod::libimod::imesh::imodel_mesh_add(Some(&mesh), &mut object.mesh)
}

/// Closed-object nesting prescan from `imeshSkinObject` (`skinobj.c`).
///
/// Builds scan contours, bounding boxes, and the source nesting-level table
/// while honoring time-based separation.  The returned scan contours are the
/// working inputs for the subsequent connector/orphan pass.
pub fn skin_object_nesting(
    object: &crate::imod::libimod::imodel::Iobj,
    flags: u32,
) -> Option<(
    Vec<crate::imod::libimod::imodel::Icont>,
    Vec<crate::imod::libimod::icont::Nesting>,
    Vec<i32>,
)> {
    use crate::imod::libimod::icont::{
        imod_contour_check_nesting, imod_contour_get_bbox, imod_contour_nest_levels,
        imod_contour_z_value, imodel_contour_scan,
    };
    use crate::imod::libimod::imesh::IMESH_MK_TIME;
    let mut scan = object
        .cont
        .iter()
        .map(|contour| imodel_contour_scan(Some(contour)))
        .collect::<Option<Vec<_>>>()?;
    let mut low = vec![crate::imod::libimod::imodel::Ipoint::default(); object.cont.len()];
    let mut high = low.clone();
    for (index, contour) in object.cont.iter().enumerate() {
        imod_contour_get_bbox(Some(contour), &mut low[index], &mut high[index]);
    }
    let mut nests = Vec::new();
    let mut indices = vec![-1; object.cont.len()];
    let (mut count, mut warnings) = (0, 0);
    for first in 0..object.cont.len() {
        if object.cont[first].pts.is_empty() {
            continue;
        }
        for second in first + 1..object.cont.len() {
            if object.cont[second].pts.is_empty()
                || imod_contour_z_value(Some(&object.cont[first]))
                    != imod_contour_z_value(Some(&object.cont[second]))
                || (flags & IMESH_MK_TIME != 0
                    && object.cont[first].time != object.cont[second].time)
            {
                continue;
            }
            if imod_contour_check_nesting(
                first as i32,
                second as i32,
                &mut scan,
                &low,
                &high,
                &mut nests,
                &mut indices,
                &mut count,
                &mut warnings,
            ) != 0
            {
                return None;
            }
        }
    }
    imod_contour_nest_levels(&mut nests, &indices, count);
    // `imeshSkinObject`: compose each nested contour's scan representation
    // with immediate interior areas removed before matching it across Z.
    for nest in &mut nests {
        if nest.inside.is_empty() {
            continue;
        }
        let mut inside_scan = scan.get(nest.co as usize)?.clone();
        for &inner in &nest.inside {
            subtract_scan_contours(&mut inside_scan, scan.get(inner as usize)?);
        }
        nest.inscan = Some(inside_scan);
    }
    Some((scan, nests, indices))
}

/// Per-section contour groups used by `imeshSkinObject` (`skinobj.c`).
/// Empty contours are retained in neither group, matching the native early
/// connection marking before the main Z-pass loop.
pub fn skin_object_z_groups(
    object: &crate::imod::libimod::imodel::Iobj,
    increment_z: i32,
) -> Vec<(i32, Vec<i32>)> {
    use crate::imod::libimod::icont::imod_contour_z_value;
    let increment = increment_z.max(1);
    let mut groups: std::collections::BTreeMap<i32, Vec<i32>> = std::collections::BTreeMap::new();
    for (index, contour) in object.cont.iter().enumerate() {
        if contour.pts.is_empty() {
            continue;
        }
        let z = imod_contour_z_value(Some(contour));
        if z.rem_euclid(increment) != 0 {
            continue;
        }
        groups.entry(z).or_default().push(index as i32);
    }
    groups.into_iter().collect()
}

/// Non-nested multi-contour portion of closed `imeshSkinObject`.
///
/// Native closed-object skinning for groups of contours on adjacent sections.
/// The nesting-aware path first partitions matching contours by nesting level
/// and scan overlap, then meshes each matched component with the appropriate
/// inside/outside winding.
pub fn skin_closed_contour_groups(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    increment_z: i32,
    flags: u32,
) -> i32 {
    let Some((scan, nests, nest_indices)) = skin_object_nesting(object, flags) else {
        return -1;
    };
    let groups = skin_object_z_groups(object, increment_z);
    object.mesh.clear();
    for pair in groups.windows(2) {
        if pair[1].0 - pair[0].0 != increment_z.max(1) {
            continue;
        }
        let Some(&bottom_first) = pair[0].1.first() else {
            continue;
        };
        let Some(&top_first) = pair[1].1.first() else {
            continue;
        };
        let (surface, time, top_surface, top_time) = {
            let bottom = &object.cont[bottom_first as usize];
            let top = &object.cont[top_first as usize];
            (bottom.surf as i32, bottom.time as i32, top.surf, top.time)
        };
        if flags & crate::imod::libimod::imesh::IMESH_MK_SURF != 0 && surface != top_surface {
            continue;
        }
        if flags & crate::imod::libimod::imesh::IMESH_MK_TIME != 0 && time != top_time {
            continue;
        }
        if nests.is_empty() {
            if mesh_joined_contour_groups(object, &pair[0].1, &pair[1].1, scale, false, flags) != 0
            {
                return -1;
            }
            continue;
        }
        let Some((minimum, maximum)) =
            skin_overlap_matrix(object, &scan, &nests, &nest_indices, &pair[0].1, &pair[1].1)
        else {
            return -1;
        };
        let bottom_levels = skin_group_nesting_levels(&pair[0].1, &nest_indices, &nests);
        let top_levels = skin_group_nesting_levels(&pair[1].1, &nest_indices, &nests);
        let components =
            skin_overlap_components(&minimum, &maximum, &bottom_levels, &top_levels, 0.);
        let Some(components) = skin_component_contour_lists(&components, &pair[0].1, &pair[1].1)
        else {
            return -1;
        };
        let matched_bottom = components
            .iter()
            .flat_map(|(bottom, _)| bottom.iter())
            .copied()
            .collect::<Vec<_>>();
        let matched_top = components
            .iter()
            .flat_map(|(_, top)| top.iter())
            .copied()
            .collect::<Vec<_>>();
        let mut connector_scans = scan.clone();
        let connector_bounds = object
            .cont
            .iter()
            .map(|contour| {
                let (mut low, mut high) = (Default::default(), Default::default());
                crate::imod::libimod::icont::imod_contour_get_bbox(
                    Some(contour),
                    &mut low,
                    &mut high,
                );
                (low, high)
            })
            .collect::<Vec<_>>();
        for (bottom_list, top_list) in components {
            let Some(level) = skin_group_nesting_levels(&bottom_list, &nest_indices, &nests)
                .first()
                .copied()
            else {
                return -1;
            };
            let Some((bottom_joined, top_joined)) =
                skin_join_component_contours(object, &bottom_list, &top_list, 1)
            else {
                return -1;
            };
            let (bottom_joined, top_joined) = if level.rem_euclid(2) == 1 {
                let top_just_inside = skin_immediate_inner_positions(
                    &pair[1].1,
                    &top_list,
                    &matched_top,
                    &nest_indices,
                    &nests,
                );
                let bottom_just_inside = skin_immediate_inner_positions(
                    &pair[0].1,
                    &bottom_list,
                    &matched_bottom,
                    &nest_indices,
                    &nests,
                );
                let mut top_used = vec![1; pair[1].1.len()];
                let mut bottom_used = vec![1; pair[0].1.len()];
                for &position in &top_just_inside {
                    top_used[position] = 0;
                }
                for &position in &bottom_just_inside {
                    bottom_used[position] = 0;
                }
                let Some(bottom_joined) = skin_connect_orphans(
                    object,
                    bottom_joined,
                    &pair[1].1,
                    &mut top_used,
                    &top_just_inside,
                    -1,
                    true,
                    surface,
                    time,
                    scale,
                    &mut connector_scans,
                    &connector_bounds,
                ) else {
                    return -1;
                };
                let Some(top_joined) = skin_connect_orphans(
                    object,
                    top_joined,
                    &pair[0].1,
                    &mut bottom_used,
                    &bottom_just_inside,
                    1,
                    true,
                    surface,
                    time,
                    scale,
                    &mut connector_scans,
                    &connector_bounds,
                ) else {
                    return -1;
                };
                (bottom_joined, top_joined)
            } else {
                (bottom_joined, top_joined)
            };
            if mesh_contours(
                object,
                &bottom_joined,
                &top_joined,
                surface,
                time,
                scale,
                level.rem_euclid(2) == 0,
            ) != 0
            {
                return -1;
            }
        }
    }
    if flags & crate::imod::libimod::imesh::IMESH_MK_NORM != 0 {
        imesh_remesh_normal(&mut object.mesh, scale, 0);
    }
    0
}

/// Overlap tables built by the nested branch of `imeshSkinObject`.
/// Returns row-major `(minimum, maximum)` fractions for `top × bottom`.
pub fn skin_overlap_matrix(
    object: &crate::imod::libimod::imodel::Iobj,
    scan: &[crate::imod::libimod::imodel::Icont],
    nests: &[crate::imod::libimod::icont::Nesting],
    nest_indices: &[i32],
    bottom: &[i32],
    top: &[i32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    use crate::imod::libimod::icont::{imod_contour_get_bbox, imodel_overlap_fractions};
    let mut minimum = Vec::with_capacity(bottom.len() * top.len());
    let mut maximum = Vec::with_capacity(bottom.len() * top.len());
    for &top_index in top {
        let top_index = usize::try_from(top_index).ok()?;
        let top_nest = nest_indices.get(top_index).copied().unwrap_or(-1);
        let top_contour = if top_nest >= 0 {
            nests
                .get(top_nest as usize)?
                .inscan
                .as_ref()
                .unwrap_or(scan.get(top_index)?)
        } else {
            scan.get(top_index)?
        };
        let (mut top_low, mut top_high) = (Default::default(), Default::default());
        imod_contour_get_bbox(Some(top_contour), &mut top_low, &mut top_high);
        for &bottom_index in bottom {
            let bottom_index = usize::try_from(bottom_index).ok()?;
            let bottom_nest = nest_indices.get(bottom_index).copied().unwrap_or(-1);
            let bottom_contour = if bottom_nest >= 0 {
                nests
                    .get(bottom_nest as usize)?
                    .inscan
                    .as_ref()
                    .unwrap_or(scan.get(bottom_index)?)
            } else {
                scan.get(bottom_index)?
            };
            let (mut bottom_low, mut bottom_high) = (Default::default(), Default::default());
            imod_contour_get_bbox(Some(bottom_contour), &mut bottom_low, &mut bottom_high);
            let (mut one, mut two) = (0., 0.);
            let (mut bottom_work, mut top_work) = (bottom_contour.clone(), top_contour.clone());
            imodel_overlap_fractions(
                &mut bottom_work,
                bottom_low,
                bottom_high,
                &mut top_work,
                top_low,
                top_high,
                &mut one,
                &mut two,
            );
            minimum.push(one.min(two));
            maximum.push(one.max(two));
        }
    }
    let _ = object; // retained for the source-shaped object-level API.
    Some((minimum, maximum))
}

/// First component-building loop of nested `imeshSkinObject` matching.
/// It groups same-parity, same-level contour indices that overlap above the
/// configured threshold, starting with the greatest minimum overlap.
pub fn skin_overlap_components(
    minimum: &[f32],
    maximum: &[f32],
    bottom_levels: &[i32],
    top_levels: &[i32],
    overlap: f32,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let width = bottom_levels.len();
    if width == 0 || minimum.len() != width * top_levels.len() || maximum.len() != minimum.len() {
        return Vec::new();
    }
    let (mut top_used, mut bottom_used) = (
        vec![false; top_levels.len()],
        vec![false; bottom_levels.len()],
    );
    let mut output = Vec::new();
    for parity in [1, 0] {
        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for top in 0..top_levels.len() {
                for bottom in 0..width {
                    let value = minimum[top * width + bottom];
                    if !top_used[top]
                        && !bottom_used[bottom]
                        && top_levels[top].rem_euclid(2) == parity
                        && bottom_levels[bottom].rem_euclid(2) == parity
                        && value > overlap
                        && best.is_none_or(|(_, _, prior)| value > prior)
                    {
                        best = Some((top, bottom, value));
                    }
                }
            }
            let Some((seed_top, seed_bottom, _)) = best else {
                break;
            };
            let (top_level, bottom_level) = (top_levels[seed_top], bottom_levels[seed_bottom]);
            let (mut tops, mut bottoms) = (vec![seed_top], vec![seed_bottom]);
            top_used[seed_top] = true;
            bottom_used[seed_bottom] = true;
            let mut changed = true;
            while changed {
                changed = false;
                for top in 0..top_levels.len() {
                    if !top_used[top]
                        && top_levels[top] == top_level
                        && bottoms
                            .iter()
                            .any(|&bottom| maximum[top * width + bottom] > overlap)
                    {
                        top_used[top] = true;
                        tops.push(top);
                        changed = true;
                    }
                }
                for bottom in 0..width {
                    if !bottom_used[bottom]
                        && bottom_levels[bottom] == bottom_level
                        && tops
                            .iter()
                            .any(|&top| maximum[top * width + bottom] > overlap)
                    {
                        bottom_used[bottom] = true;
                        bottoms.push(bottom);
                        changed = true;
                    }
                }
            }
            output.push((bottoms, tops));
        }
    }
    output
}

/// Maps a native contour-index list to the nesting levels used by the
/// overlap-component matcher in `imeshSkinObject`.
pub fn skin_group_nesting_levels(
    indices: &[i32],
    nest_indices: &[i32],
    nests: &[crate::imod::libimod::icont::Nesting],
) -> Vec<i32> {
    indices
        .iter()
        .map(|&index| {
            let nest = usize::try_from(index)
                .ok()
                .and_then(|value| nest_indices.get(value))
                .copied()
                .unwrap_or(-1);
            if nest >= 0 {
                nests
                    .get(nest as usize)
                    .map(|entry| entry.level)
                    .unwrap_or(1)
            } else {
                1
            }
        })
        .collect()
}

/// Complete overlap analysis stage for one pair of Z groups in nested
/// `imeshSkinObject` processing.  Component entries are positions within the
/// supplied bottom/top lists, as in the native work arrays.
pub fn skin_nested_group_components(
    object: &crate::imod::libimod::imodel::Iobj,
    bottom: &[i32],
    top: &[i32],
    flags: u32,
    overlap: f32,
) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
    let (scan, nests, nest_indices) = skin_object_nesting(object, flags)?;
    let (minimum, maximum) =
        skin_overlap_matrix(object, &scan, &nests, &nest_indices, bottom, top)?;
    let bottom_levels = skin_group_nesting_levels(bottom, &nest_indices, &nests);
    let top_levels = skin_group_nesting_levels(top, &nest_indices, &nests);
    Some(skin_overlap_components(
        &minimum,
        &maximum,
        &bottom_levels,
        &top_levels,
        overlap,
    ))
}

/// Converts the position-based nested matching result back into the source
/// contour-index lists consumed by `join_all_contours`.
pub fn skin_component_contour_lists(
    components: &[(Vec<usize>, Vec<usize>)],
    bottom: &[i32],
    top: &[i32],
) -> Option<Vec<(Vec<i32>, Vec<i32>)>> {
    components
        .iter()
        .map(|(bottom_positions, top_positions)| {
            let bottom_list = bottom_positions
                .iter()
                .map(|&position| bottom.get(position).copied())
                .collect::<Option<Vec<_>>>()?;
            let top_list = top_positions
                .iter()
                .map(|&position| top.get(position).copied())
                .collect::<Option<Vec<_>>>()?;
            Some((bottom_list, top_list))
        })
        .collect()
}

/// Positions in one Z-group that are immediately inside an outer component
/// and have not already been claimed by an overlap component.  This mirrors
/// the native `*JustInList` construction before `connect_orphans` runs.
pub fn skin_immediate_inner_positions(
    group: &[i32],
    outer_component: &[i32],
    matched: &[i32],
    nest_indices: &[i32],
    nests: &[crate::imod::libimod::icont::Nesting],
) -> Vec<usize> {
    group
        .iter()
        .enumerate()
        .filter_map(|(position, &contour)| {
            if matched.contains(&contour) {
                return None;
            }
            let nest_index = usize::try_from(contour)
                .ok()
                .and_then(|index| nest_indices.get(index))
                .copied()?;
            let nest = nests.get(usize::try_from(nest_index).ok()?)?;
            let parent_level = outer_component
                .iter()
                .filter_map(|&outer| {
                    usize::try_from(outer)
                        .ok()
                        .and_then(|index| nest_indices.get(index))
                        .copied()
                        .and_then(|index| usize::try_from(index).ok())
                        .and_then(|index| nests.get(index))
                        .map(|entry| entry.level)
                })
                .next()?;
            (nest.level == parent_level + 1
                && nest
                    .outside
                    .iter()
                    .any(|outside| outer_component.contains(outside)))
            .then_some(position)
        })
        .collect()
}

/// Builds the two joined contour paths for one matched skinning component.
/// This is the paired `join_all_contours` operation used repeatedly by the
/// nested branch before it decides whether to mesh an outer or inner region.
pub fn skin_join_component_contours(
    object: &crate::imod::libimod::imodel::Iobj,
    bottom: &[i32],
    top: &[i32],
    fill: i32,
) -> Option<(
    crate::imod::libimod::imodel::Icont,
    crate::imod::libimod::imodel::Icont,
)> {
    let bottom_joined =
        join_all_contours(object, bottom, bottom.len(), fill, top, top.len(), None)?;
    let top_joined = join_all_contours(object, top, top.len(), -fill, bottom, bottom.len(), None)?;
    Some((bottom_joined, top_joined))
}

/// First stage of native `connect_orphans` (`skinobj.c`): mark still-unused
/// just-inside contours for capping when their scan overlap with the current
/// outer contour exceeds the native 0.8 threshold.  A `used` value of `-1`
/// carries the source's "cap it" state; zero remains available for joining.
pub fn skin_mark_capped_orphans(
    outer: &crate::imod::libimod::imodel::Icont,
    contour_list: &[i32],
    used: &mut [i32],
    just_inside: &[usize],
    scans: &mut [crate::imod::libimod::imodel::Icont],
    bounds: &[(
        crate::imod::libimod::imodel::Ipoint,
        crate::imod::libimod::imodel::Ipoint,
    )],
) -> bool {
    use crate::imod::libimod::icont::{
        imod_contour_get_bbox, imodel_contour_scan, imodel_overlap_fractions,
    };
    let Some(mut outer_scan) = imodel_contour_scan(Some(outer)) else {
        return false;
    };
    let (mut low, mut high) = (Default::default(), Default::default());
    imod_contour_get_bbox(Some(outer), &mut low, &mut high);
    for &position in just_inside {
        if used.get(position).copied().unwrap_or(1) != 0 {
            continue;
        }
        let Some(&contour_index) = contour_list.get(position) else {
            return false;
        };
        let Ok(contour_index) = usize::try_from(contour_index) else {
            return false;
        };
        let (Some(scan), Some(&(scan_low, scan_high))) =
            (scans.get_mut(contour_index), bounds.get(contour_index))
        else {
            return false;
        };
        let (mut first, mut second) = (0., 0.);
        imodel_overlap_fractions(
            scan,
            scan_low,
            scan_high,
            &mut outer_scan,
            low,
            high,
            &mut first,
            &mut second,
        );
        if first > 0.8 {
            used[position] = -1;
        }
    }
    true
}

/// Candidate break pairs generated by the first scan in `connect_orphans`.
/// Each item is `(outer_start, outer_end, shortest_outer_path, path_sign,
/// inner_start, inner_end)`; `path_sign` retains the native orientation test.
pub fn skin_orphan_break_candidates(
    outer: &crate::imod::libimod::imodel::Icont,
    inner: &crate::imod::libimod::imodel::Icont,
) -> Vec<(usize, usize, f32, f32, usize, usize)> {
    use crate::imod::libimod::icont::{imod_contour_length, imod_contour_nearest};
    const SECTIONS: usize = 38;
    if outer.pts.len() < 2 || inner.pts.len() < 2 {
        return Vec::new();
    }
    let total_inner = imod_contour_length(Some(inner), 1);
    if total_inner <= 0. {
        return Vec::new();
    }
    let step = total_inner / (SECTIONS - 2) as f32;
    let mut samples = vec![(
        imod_contour_nearest(Some(outer), &inner.pts[0]).max(0) as usize,
        0usize,
    )];
    let (mut travelled, mut next_at) = (0., step);
    let mut point = 0usize;
    loop {
        let next = (point + 1) % inner.pts.len();
        let dx = inner.pts[next].x - inner.pts[point].x;
        let dy = inner.pts[next].y - inner.pts[point].y;
        travelled += (dx * dx + dy * dy).sqrt();
        if travelled > next_at || next == 0 {
            samples.push((
                imod_contour_nearest(Some(outer), &inner.pts[next]).max(0) as usize,
                next,
            ));
            next_at += step;
            if samples.len() == SECTIONS {
                break;
            }
        }
        point = next;
        if next == 0 {
            break;
        }
    }
    samples.sort_unstable_by_key(|sample| sample.0);
    samples.dedup_by_key(|sample| sample.0);
    if samples.len() < 2 {
        return Vec::new();
    }
    let outer_length = imod_contour_length(Some(outer), 1);
    let mut forward = vec![0.; samples.len() - 1];
    for index in 0..samples.len() - 1 {
        for point in samples[index].0..samples[index + 1].0 {
            let next = (point + 1) % outer.pts.len();
            let dx = outer.pts[next].x - outer.pts[point].x;
            let dy = outer.pts[next].y - outer.pts[point].y;
            forward[index] += (dx * dx + dy * dy).sqrt();
        }
    }
    let mut candidates = Vec::new();
    for first in 0..samples.len() - 1 {
        let mut distance = 0.;
        for second in first + 1..samples.len() {
            distance += forward[second - 1];
            let (shortest, sign) = if distance < outer_length / 2. {
                (distance, 1.)
            } else {
                (outer_length - distance, -1.)
            };
            candidates.push((
                samples[first].0,
                samples[second].0,
                shortest,
                sign,
                samples[first].1,
                samples[second].1,
            ));
        }
    }
    candidates.sort_unstable_by(|left, right| compare_f32_native(right.2, left.2));
    candidates
}

/// Choose the first `connect_orphans` break before the native local-neighbour
/// refinement.  Candidates are considered in decreasing outer separation;
/// the native tie break retains the largest joinable inner area, then the
/// smallest separation-to-area score.
pub fn skin_select_orphan_break(
    outer: &crate::imod::libimod::imodel::Icont,
    inner: &crate::imod::libimod::imodel::Icont,
    used: &[bool],
    just_inside: &[usize],
    orphan_scans: &mut [crate::imod::libimod::imodel::Icont],
    orphan_bounds: &[(
        crate::imod::libimod::imodel::Ipoint,
        crate::imod::libimod::imodel::Ipoint,
    )],
    join_direction: i32,
) -> Option<(usize, usize, f32, f32, i32)> {
    use crate::imod::libimod::icont::imod_contour_area;
    const FAILURE: f32 = 1.0e25;
    let full_area = imod_contour_area(Some(outer));
    let mut best: Option<(usize, usize, f32, f32, i32)> = None;
    let mut maximum_area = -1.;
    for (first, second, path_length, sign, _inner_first, _inner_second) in
        skin_orphan_break_candidates(outer, inner)
    {
        let (ratio, inner_area) = evaluate_break(
            outer,
            used,
            just_inside,
            orphan_scans,
            orphan_bounds,
            first,
            second,
            maximum_area,
            path_length,
            full_area,
            join_direction,
            true,
        );
        if ratio > FAILURE {
            return None;
        }
        if inner_area > maximum_area
            || (inner_area == maximum_area && best.is_some_and(|current| ratio < current.2))
        {
            maximum_area = inner_area;
            best = Some((first, second, ratio, path_length, sign as i32));
        }
    }
    best
}

/// Directional local search following the coarse candidate selection in
/// `connect_orphans`.  The thirteen offsets are the native `dst1`/`dst2`
/// stencil; previously tested pairs are cached so a sweep cannot oscillate.
pub fn skin_refine_orphan_break(
    outer: &crate::imod::libimod::imodel::Icont,
    inner: &crate::imod::libimod::imodel::Icont,
    used: &[bool],
    just_inside: &[usize],
    orphan_scans: &mut [crate::imod::libimod::imodel::Icont],
    orphan_bounds: &[(
        crate::imod::libimod::imodel::Ipoint,
        crate::imod::libimod::imodel::Ipoint,
    )],
    join_direction: i32,
) -> Option<(usize, usize, f32)> {
    use crate::imod::libimod::icont::imod_contour_area;
    const OFFSETS: [(i32, i32); 13] = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (0, 2),
        (2, 0),
        (-1, 0),
        (0, -1),
        (-1, -1),
        (-1, 1),
        (1, -1),
    ];
    let (mut first, mut second, mut best_ratio, base_path, sign) = skin_select_orphan_break(
        outer,
        inner,
        used,
        just_inside,
        orphan_scans,
        orphan_bounds,
        join_direction,
    )?;
    let full_area = imod_contour_area(Some(outer));
    let count = outer.pts.len();
    if count < 2 {
        return None;
    }
    let sweep = -sign;
    let mut best_path = base_path;
    let mut tested = std::collections::HashMap::from([((first, second), best_ratio)]);
    loop {
        let mut improved = false;
        for (first_offset, second_offset) in OFFSETS {
            let trial_first =
                (first as i32 + first_offset * sweep).rem_euclid(count as i32) as usize;
            let trial_second =
                (second as i32 - second_offset * sweep).rem_euclid(count as i32) as usize;
            let ratio = if let Some(&cached) = tested.get(&(trial_first, trial_second)) {
                cached
            } else {
                let path = skin_break_path_length(
                    outer,
                    first,
                    second,
                    best_path,
                    first_offset,
                    second_offset,
                    sweep,
                );
                let (ratio, _) = evaluate_break(
                    outer,
                    used,
                    just_inside,
                    orphan_scans,
                    orphan_bounds,
                    trial_first,
                    trial_second,
                    best_ratio,
                    path,
                    full_area,
                    join_direction,
                    false,
                );
                if ratio > 1.0e25 {
                    return None;
                }
                tested.insert((trial_first, trial_second), ratio);
                ratio
            };
            if ratio < best_ratio {
                best_path = skin_break_path_length(
                    outer,
                    first,
                    second,
                    best_path,
                    first_offset,
                    second_offset,
                    sweep,
                );
                first = trial_first;
                second = trial_second;
                best_ratio = ratio;
                improved = true;
                break;
            }
        }
        if !improved {
            break;
        }
    }
    Some((first, second, best_ratio))
}

fn skin_break_path_length(
    contour: &crate::imod::libimod::imodel::Icont,
    first: usize,
    second: usize,
    base: f32,
    first_offset: i32,
    second_offset: i32,
    sweep: i32,
) -> f32 {
    let count = contour.pts.len() as i32;
    let mut path = base;
    for (start, offset) in [(first, first_offset), (second, second_offset)] {
        let direction = if offset > 0 { 1 } else { -1 };
        let mut point = start as i32;
        for _ in 0..offset.abs() {
            let next = (point + direction * sweep).rem_euclid(count);
            let one = contour.pts[point as usize];
            let two = contour.pts[next as usize];
            path += direction as f32 * ((one.x - two.x).powi(2) + (one.y - two.y).powi(2)).sqrt();
            point = next;
        }
    }
    path
}

/// Connect the orphan contours nested immediately inside one outer contour.
///
/// This is the ownership-changing portion of `connect_orphans`: cap-worthy
/// orphans remain marked `-1`, while each viable break peels off the smaller
/// side of the outer contour, joins every orphan whose scan overlaps that new
/// inside piece, and meshes the two paths.  The returned contour is the
/// remaining outer path for the caller's next nesting pass.
pub fn skin_connect_orphans(
    object: &mut crate::imod::libimod::imodel::Iobj,
    mut outer: crate::imod::libimod::imodel::Icont,
    contour_list: &[i32],
    used: &mut [i32],
    just_inside: &[usize],
    join_direction: i32,
    inside: bool,
    surface: i32,
    time: i32,
    scale: &crate::imod::libimod::imodel::Ipoint,
    scans: &mut [crate::imod::libimod::imodel::Icont],
    bounds: &[(
        crate::imod::libimod::imodel::Ipoint,
        crate::imod::libimod::imodel::Ipoint,
    )],
) -> Option<crate::imod::libimod::imodel::Icont> {
    use crate::imod::libimod::icont::{
        imod_contour_area, imod_contour_get_bbox, imodel_contour_scan, imodel_scans_overlap,
    };
    if !skin_mark_capped_orphans(&outer, contour_list, used, just_inside, scans, bounds) {
        return None;
    }
    for &position in just_inside {
        if used.get(position).copied().unwrap_or(1) != 0 {
            continue;
        }
        let contour_index = usize::try_from(*contour_list.get(position)?).ok()?;
        let inner_source = object.cont.get(contour_index)?.clone();
        let active = just_inside
            .iter()
            .filter_map(|&candidate| {
                (used.get(candidate).copied() == Some(0))
                    .then(|| contour_list.get(candidate).copied())
                    .flatten()
                    .and_then(|index| usize::try_from(index).ok())
            })
            .collect::<Vec<_>>();
        let mut used_by_contour = vec![true; object.cont.len()];
        for &index in &active {
            if index < used_by_contour.len() {
                used_by_contour[index] = false;
            }
        }
        let (first, second, ratio) = skin_refine_orphan_break(
            &outer,
            &inner_source,
            &used_by_contour,
            &active,
            scans,
            bounds,
            join_direction,
        )?;
        if ratio >= 1.0e20 {
            continue;
        }
        let (first_piece, second_piece) =
            break_contour_inout(Some(&outer), first as i32, second as i32, -join_direction)?;
        let (inner_piece, outer_piece) =
            if imod_contour_area(Some(&second_piece)) < imod_contour_area(Some(&first_piece)) {
                (second_piece, first_piece)
            } else {
                (first_piece, second_piece)
            };
        let inner_scan = imodel_contour_scan(Some(&inner_piece))?;
        let (mut inner_low, mut inner_high) = (Default::default(), Default::default());
        imod_contour_get_bbox(Some(&inner_piece), &mut inner_low, &mut inner_high);
        let mut joining = Vec::new();
        for &candidate in just_inside {
            if used.get(candidate).copied().unwrap_or(1) != 0 {
                continue;
            }
            let index = usize::try_from(*contour_list.get(candidate)?).ok()?;
            if imodel_scans_overlap(
                scans.get(index),
                bounds.get(index)?.0,
                bounds.get(index)?.1,
                Some(&inner_scan),
                inner_low,
                inner_high,
            ) != 0
            {
                joining.push(index as i32);
                used[candidate] = 1;
            }
        }
        if joining.is_empty() {
            continue;
        }
        let joined = join_all_contours(
            object,
            &joining,
            joining.len(),
            join_direction,
            &[],
            0,
            Some(&inner_piece),
        )?;
        let (bottom, top) = if join_direction < 0 {
            (&inner_piece, &joined)
        } else {
            (&joined, &inner_piece)
        };
        if mesh_contours(object, bottom, top, surface, time, scale, inside) != 0 {
            return None;
        }
        outer = outer_piece;
    }
    Some(outer)
}

/// Joined outer-contour operation used by the complex `imeshSkinObject`
/// branch: combine each same-level Z group and mesh the two combined paths.
pub fn mesh_joined_contour_groups(
    object: &mut crate::imod::libimod::imodel::Iobj,
    bottom_indices: &[i32],
    top_indices: &[i32],
    scale: &crate::imod::libimod::imodel::Ipoint,
    inside: bool,
    flags: u32,
) -> i32 {
    if bottom_indices.is_empty() || top_indices.is_empty() {
        return 0;
    }
    let view = object.clone();
    let Some(bottom) = join_all_contours(
        &view,
        bottom_indices,
        bottom_indices.len(),
        1,
        top_indices,
        top_indices.len(),
        None,
    ) else {
        return -1;
    };
    let Some(top) = join_all_contours(
        &view,
        top_indices,
        top_indices.len(),
        -1,
        bottom_indices,
        bottom_indices.len(),
        None,
    ) else {
        return -1;
    };
    mesh_unconnected_contours_to_object(object, &bottom, &top, scale, inside, flags)
}

/// Open-surface branch of `mesh_open_obj` (`skinobj.c`).
///
/// For every requested Z-pass, pairs still-unconnected contours by their
/// minimum projected bounding-box separation, then makes each pair available
/// to no further pair in that pass.  Tube and dome meshing is deliberately a
/// separate source branch selected by `IMESH_MK_TUBE`.
pub fn mesh_open_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    increment_z: i32,
    flags: u32,
    skip_passes: i32,
) -> i32 {
    mesh_open_object_in_range(
        object,
        scale,
        increment_z,
        flags,
        skip_passes,
        crate::imod::libimod::imesh::DEFAULT_VALUE,
        crate::imod::libimod::imesh::DEFAULT_VALUE,
    )
}

/// Range-aware `mesh_open_obj` branch for `MeshParams::minz`/`maxz`.
pub fn mesh_open_object_in_range(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    increment_z: i32,
    flags: u32,
    skip_passes: i32,
    minimum_z: i32,
    maximum_z: i32,
) -> i32 {
    use crate::imod::libimod::icont::{
        ICONT_CONNECT_BOTTOM, ICONT_CONNECT_TOP, imod_contour_get_bbox, imod_contour_make_z_tables,
    };
    use crate::imod::libimod::imesh::{DEFAULT_VALUE, IMESH_MK_SKIP, IMESH_MK_SURF, IMESH_MK_TIME};
    use crate::imod::libimod::imodel::{ICONT_OPEN, IMOD_OBJFLAG_OUT};
    let mut contz = Vec::new();
    let mut zlist = Vec::new();
    let mut numatz = Vec::new();
    let mut contatz = Vec::new();
    let mut zmin = 0;
    let mut zmax = 0;
    let mut zlsize = 0;
    let mut nummax = 0;
    if imod_contour_make_z_tables(
        object,
        increment_z.max(1),
        ICONT_CONNECT_TOP | ICONT_CONNECT_BOTTOM,
        &mut contz,
        &mut zlist,
        &mut numatz,
        &mut contatz,
        &mut zmin,
        &mut zmax,
        &mut zlsize,
        &mut nummax,
    ) != 0
        || zmin > zmax
    {
        return -1;
    }
    object.mesh.clear();
    let inside = object.flags & IMOD_OBJFLAG_OUT != 0;
    let passes = skip_passes.max(1) as usize;
    for zpass in 1..=passes {
        for iz in 0..=(zmax - zmin) as usize {
            let lower_z = zmin + iz as i32;
            if minimum_z != DEFAULT_VALUE && lower_z < minimum_z {
                continue;
            }
            if maximum_z != DEFAULT_VALUE && lower_z > maximum_z {
                continue;
            }
            let mut upper_z = lower_z;
            for _ in 0..zpass {
                if flags & IMESH_MK_SKIP != 0 {
                    upper_z = get_next_z(&zlist, upper_z);
                    if upper_z > zmax {
                        break;
                    }
                } else {
                    upper_z += increment_z.max(1);
                }
            }
            if upper_z > zmax {
                continue;
            }
            if minimum_z != DEFAULT_VALUE && upper_z < minimum_z {
                continue;
            }
            if maximum_z != DEFAULT_VALUE && upper_z > maximum_z {
                continue;
            }
            let lower = &contatz[(lower_z - zmin) as usize];
            let upper = &contatz[(upper_z - zmin) as usize];
            let mut candidates = Vec::new();
            for (li, &lower_index) in lower.iter().enumerate() {
                let lower_index = lower_index as usize;
                if contz.get(lower_index) != Some(&lower_z) {
                    continue;
                }
                if object.cont[lower_index].flags & ICONT_CONNECT_TOP != 0 {
                    continue;
                }
                for (ui, &upper_index) in upper.iter().enumerate() {
                    let upper_index = upper_index as usize;
                    if contz.get(upper_index) != Some(&upper_z) {
                        continue;
                    }
                    if object.cont[upper_index].flags & ICONT_CONNECT_BOTTOM != 0 {
                        continue;
                    }
                    let bottom = &object.cont[lower_index];
                    let top = &object.cont[upper_index];
                    if flags & IMESH_MK_SURF != 0 && bottom.surf != top.surf {
                        continue;
                    }
                    if flags & IMESH_MK_TIME != 0 && bottom.time != top.time {
                        continue;
                    }
                    let (mut bmin, mut bmax) = (Default::default(), Default::default());
                    let (mut tmin, mut tmax) = (Default::default(), Default::default());
                    imod_contour_get_bbox(Some(bottom), &mut bmin, &mut bmax);
                    imod_contour_get_bbox(Some(top), &mut tmin, &mut tmax);
                    let separation = segment_separation(bmin.x, bmax.x, tmin.x, tmax.x)
                        .max(segment_separation(bmin.y, bmax.y, tmin.y, tmax.y));
                    candidates.push((
                        separation,
                        li,
                        ui,
                        lower_index,
                        upper_index,
                        bottom.pts.len() == 1 || top.pts.len() == 1,
                    ));
                }
            }
            let mut used_lower = vec![false; lower.len()];
            let mut used_upper = vec![false; upper.len()];
            loop {
                let best_regular = candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, candidate)| {
                        !candidate.5 && !used_lower[candidate.1] && !used_upper[candidate.2]
                    })
                    .min_by(|a, b| a.1.0.total_cmp(&b.1.0));
                let best_single = candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, candidate)| {
                        candidate.5 && !used_lower[candidate.1] && !used_upper[candidate.2]
                    })
                    .min_by(|a, b| a.1.0.total_cmp(&b.1.0));
                let Some((_, candidate)) = best_regular.or(best_single) else {
                    break;
                };
                let (li, ui, lower_index, upper_index) =
                    (candidate.1, candidate.2, candidate.3, candidate.4);
                used_lower[li] = true;
                used_upper[ui] = true;
                let (bottom, top) = (
                    object.cont[lower_index].clone(),
                    object.cont[upper_index].clone(),
                );
                object.cont[lower_index].flags |= ICONT_OPEN | ICONT_CONNECT_TOP;
                object.cont[upper_index].flags |= ICONT_OPEN | ICONT_CONNECT_BOTTOM;
                if mesh_unconnected_contours_to_object(object, &bottom, &top, scale, inside, flags)
                    != 0
                {
                    return -1;
                }
            }
        }
    }
    if flags & crate::imod::libimod::imesh::IMESH_MK_NORM != 0 {
        imesh_remesh_normal(&mut object.mesh, scale, 0);
    }
    0
}

/// Core tube branch of `mesh_open_tube_obj` (`skinobj.c`).
///
/// Builds one scaled circular ring per contour point and joins adjacent rings.
/// `mesh_diameter` has the source meanings: positive is fixed diameter, zero
/// uses object line width, less than `-1` uses symbol size, and `-1` uses the
/// per-point size.  The source dome-cap and normal-remesh steps are applied
/// below when their corresponding mesh flags are set.
pub fn mesh_open_tube_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    flags: u32,
    mesh_diameter: f64,
) -> i32 {
    use crate::imod::libimod::imesh::{IMESH_MK_CAP_DOME, IMESH_MK_CAP_TUBE};
    use crate::imod::libimod::ipoint::{imod_point_get_size, imod_point_normalize};
    use crate::imod::libimod::istore::{
        DrawProps, istore_cont_surf_draw_props, istore_default_draw_props,
        istore_first_change_index, istore_next_change, istore_point_is_gap,
    };
    let mut default_props = DrawProps::default();
    istore_default_draw_props(object, &mut default_props);
    object.mesh.clear();
    for contour_index in 0..object.cont.len() {
        let contour = object.cont[contour_index].clone();
        if contour.pts.len() < 2 {
            continue;
        }
        let mut contour_props = default_props;
        let (mut contour_state, mut surface_state) = (0, 0);
        let _ = istore_cont_surf_draw_props(
            &object.store,
            &default_props,
            &mut contour_props,
            contour_index as i32,
            contour.surf,
            &mut contour_state,
            &mut surface_state,
        );
        let mut point_props = contour_props;
        let (mut point_state, mut point_changes, mut change_cursor) = (0, 0, 0usize);
        let mut next_change = istore_first_change_index(&contour.store);
        let mut rings = Vec::with_capacity(contour.pts.len());
        let mut ring_normals = Vec::with_capacity(contour.pts.len());
        for point_index in 0..contour.pts.len() {
            if point_index as i32 == next_change {
                next_change = istore_next_change(
                    &contour.store,
                    &mut change_cursor,
                    &contour_props,
                    &mut point_props,
                    &mut point_state,
                    &mut point_changes,
                );
            }
            let diameter = if mesh_diameter > 0. {
                mesh_diameter as f32
            } else if mesh_diameter == 0. {
                point_props.linewidth as f32
            } else if mesh_diameter < -1.0001 {
                point_props.symsize as f32
            } else {
                2. * imod_point_get_size(object, &contour, point_index as i32)
            };
            let diameter = diameter.max(1.);
            let slices = ((diameter / 2.) as i32).clamp(12, 50);
            let previous = point_index.saturating_sub(1);
            let next = (point_index + 1).min(contour.pts.len() - 1);
            let mut tangent = crate::imod::libimod::imodel::Ipoint {
                x: scale.x * (contour.pts[next].x - contour.pts[previous].x),
                y: scale.y * (contour.pts[next].y - contour.pts[previous].y),
                z: scale.z * (contour.pts[next].z - contour.pts[previous].z),
            };
            imod_point_normalize(&mut tangent);
            let mut ring = crate::imod::libimod::imodel::Icont::default();
            if make_tube_cont(
                &mut ring,
                &contour.pts[point_index],
                &tangent,
                scale,
                diameter,
                slices,
            ) != 0
            {
                return -1;
            }
            rings.push(ring);
            ring_normals.push(tangent);
        }
        if flags & IMESH_MK_CAP_DOME != 0 && contour_props.no_cap == 0 {
            let diameter = if mesh_diameter > 0. {
                mesh_diameter as f32
            } else {
                contour_props.linewidth.max(1) as f32
            };
            let (start_domes, start_cap) = make_dome_contours(
                &rings[0],
                &contour.pts[0],
                &ring_normals[0],
                scale,
                diameter,
                -1.,
            );
            if let Some(last_dome) = start_domes.last() {
                if let Some(mesh) = make_cap_mesh(last_dome, &start_cap, 1, &default_props, 0, 0) {
                    if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                        return -1;
                    }
                }
            }
            for dome_index in (0..start_domes.len()).rev() {
                let next = if dome_index == 0 {
                    &rings[0]
                } else {
                    &start_domes[dome_index - 1]
                };
                let mut first_props = default_props;
                let mut second_props = default_props;
                if let Some(mesh) = join_tube_cont(
                    &start_domes[dome_index],
                    next,
                    &ring_normals[0],
                    &mut first_props,
                    0,
                    &mut second_props,
                    0,
                ) {
                    if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                        return -1;
                    }
                }
            }
        } else if flags & IMESH_MK_CAP_TUBE != 0 && contour_props.no_cap == 0 {
            if let Some(mesh) = make_cap_mesh(&rings[0], &contour.pts[0], 1, &default_props, 0, 0) {
                if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                    return -1;
                }
            }
        }
        for point_index in 0..rings.len() - 1 {
            if istore_point_is_gap(&contour.store, point_index as i32) != 0 {
                continue;
            }
            let mut tangent = crate::imod::libimod::imodel::Ipoint {
                x: scale.x * (contour.pts[point_index + 1].x - contour.pts[point_index].x),
                y: scale.y * (contour.pts[point_index + 1].y - contour.pts[point_index].y),
                z: scale.z * (contour.pts[point_index + 1].z - contour.pts[point_index].z),
            };
            imod_point_normalize(&mut tangent);
            let mut first_props = default_props;
            let mut second_props = default_props;
            if let Some(mesh) = join_tube_cont(
                &rings[point_index],
                &rings[point_index + 1],
                &tangent,
                &mut first_props,
                0,
                &mut second_props,
                0,
            ) {
                if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                    return -1;
                }
            }
        }
        if flags & IMESH_MK_CAP_DOME != 0 && contour_props.no_cap == 0 {
            let last = contour.pts.len() - 1;
            let diameter = if mesh_diameter > 0. {
                mesh_diameter as f32
            } else {
                contour_props.linewidth.max(1) as f32
            };
            let (end_domes, end_cap) = make_dome_contours(
                &rings[last],
                &contour.pts[last],
                &ring_normals[last],
                scale,
                diameter,
                1.,
            );
            for dome_index in 0..end_domes.len() {
                let previous = if dome_index == 0 {
                    &rings[last]
                } else {
                    &end_domes[dome_index - 1]
                };
                let mut first_props = default_props;
                let mut second_props = default_props;
                if let Some(mesh) = join_tube_cont(
                    previous,
                    &end_domes[dome_index],
                    &ring_normals[last],
                    &mut first_props,
                    0,
                    &mut second_props,
                    0,
                ) {
                    if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                        return -1;
                    }
                }
            }
            if let Some(last_dome) = end_domes.last() {
                if let Some(mesh) = make_cap_mesh(last_dome, &end_cap, 0, &default_props, 0, 0) {
                    if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                        return -1;
                    }
                }
            }
        } else if flags & IMESH_MK_CAP_TUBE != 0 && contour_props.no_cap == 0 {
            let last = contour.pts.len() - 1;
            if let Some(mesh) =
                make_cap_mesh(&rings[last], &contour.pts[last], 0, &default_props, 0, 0)
            {
                if add_mesh_to_object(object, &contour, mesh, flags) != 0 {
                    return -1;
                }
            }
        }
    }
    if flags & crate::imod::libimod::imesh::IMESH_MK_NORM != 0 {
        imesh_remesh_normal(&mut object.mesh, scale, 0);
    }
    0
}

/// Original static `makeDomeConts` (`skinobj.c`).
/// Returns successive hemispherical rings and the terminal cap point.
pub fn make_dome_contours(
    last_contour: &crate::imod::libimod::imodel::Icont,
    last_point: &crate::imod::libimod::imodel::Ipoint,
    normal: &crate::imod::libimod::imodel::Ipoint,
    scale: &crate::imod::libimod::imodel::Ipoint,
    diameter: f32,
    direction: f32,
) -> (
    Vec<crate::imod::libimod::imodel::Icont>,
    crate::imod::libimod::imodel::Ipoint,
) {
    let positions = ((last_contour.pts.len() as f32 / 4.).round() as usize).max(1);
    let mut rings = Vec::with_capacity(positions.saturating_sub(1));
    let mut cap = *last_point;
    for position in 1..=positions {
        let angle = std::f32::consts::FRAC_PI_2 * position as f32 / positions as f32;
        let distance = 0.5 * diameter * direction * angle.sin();
        cap = crate::imod::libimod::imodel::Ipoint {
            x: (last_point.x * scale.x + normal.x * distance) / scale.x,
            y: (last_point.y * scale.y + normal.y * distance) / scale.y,
            z: (last_point.z * scale.z + normal.z * distance) / scale.z,
        };
        if position < positions {
            let radius = 0.5 * diameter * angle.cos();
            let mut ring = crate::imod::libimod::imodel::Icont::default();
            let slices = (radius as i32).clamp(12, 50);
            if make_tube_cont(&mut ring, &cap, normal, scale, 2. * radius, slices) == 0 {
                rings.push(ring);
            }
        }
    }
    (rings, cap)
}

/// Simple unconnected-stack branch of `imeshSkinObject` (`skinobj.c`).
///
/// Meshes successive nonempty contours in Z order when each layer is already
/// known to be an unbranched, unnested pair.  The full skinning dispatcher
/// extends this with connectors, nesting, orphan joins, tubes, and caps.
/// Returns `-2` when a Z section has multiple contours, which must be handled
/// by that remaining dispatcher rather than silently treated as success.
///
/// This is selected by [`imesh_skin_object`] for closed objects after its
/// source-level validation and open-object dispatch.
pub fn skin_unconnected_contour_stack(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    flags: u32,
) -> i32 {
    skin_unconnected_contour_stack_in_range_with_caps(
        object,
        scale,
        flags,
        crate::imod::libimod::imesh::DEFAULT_VALUE,
        crate::imod::libimod::imesh::DEFAULT_VALUE,
        crate::imod::libimod::imesh::IMESH_CAP_OFF,
    )
}

/// Native `imeshSkinObject` dispatcher (`skinobj.c`).
///
/// The native entry point first rejects scatter objects, clears prior meshes,
/// then selects tube meshing for open tube objects, ordinary open-object
/// meshing for other open objects, or closed-object skinning.  The closed
/// branch currently delegates to the translated unconnected-stack path; it
/// returns `-2` for a same-Z multi-contour topology so callers cannot mistake
/// an as-yet-untranslated nesting/connector case for a successful mesh.
/// A callback return value is propagated exactly as the native early status
/// checks do.
pub fn imesh_skin_object(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    _overlap: f64,
    do_cap: i32,
    cap_skip_zlist: Option<&[i32]>,
    z_increment: i32,
    flags: u32,
    skip_passes: i32,
    tube_diameter: f64,
    callback: Option<fn(i32) -> i32>,
) -> i32 {
    use crate::imod::libimod::imesh::IMESH_MK_TUBE;
    use crate::imod::libimod::imodel::{IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT};
    if object.flags & IMOD_OBJFLAG_SCAT != 0 {
        return -1;
    }
    object.mesh.clear();
    if object.cont.is_empty() {
        return 0;
    }
    if object.flags & IMOD_OBJFLAG_OPEN != 0 {
        if flags & IMESH_MK_TUBE != 0 {
            return mesh_open_tube_object(object, scale, flags, tube_diameter);
        }
        return mesh_open_object(object, scale, z_increment, flags, skip_passes);
    }
    if let Some(report) = callback {
        let status = report(1);
        if status != 0 {
            return status;
        }
        let status = report(25);
        if status != 0 {
            return status;
        }
    }
    let has_multiple_per_section = skin_object_z_groups(object, z_increment)
        .iter()
        .any(|(_, contours)| contours.len() > 1);
    let result = if has_multiple_per_section {
        // The complex native path owns caps after connection/orphan passes;
        // defer caps until that nested branch is complete.
        skin_closed_contour_groups(object, scale, z_increment, flags)
    } else {
        skin_unconnected_contour_stack_in_range_with_caps_and_skip(
            object,
            scale,
            flags,
            crate::imod::libimod::imesh::DEFAULT_VALUE,
            crate::imod::libimod::imesh::DEFAULT_VALUE,
            do_cap,
            cap_skip_zlist,
        )
    };
    if result == 0 {
        if let Some(report) = callback {
            return report(100);
        }
    }
    result
}

/// Range-aware simple-stack branch used by the object editor's `minz`/`maxz`
/// meshing controls.  `DEFAULT_VALUE` retains the source's unrestricted
/// range semantics.
pub fn skin_unconnected_contour_stack_in_range(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    flags: u32,
    minimum_z: i32,
    maximum_z: i32,
) -> i32 {
    skin_unconnected_contour_stack_in_range_with_caps(
        object,
        scale,
        flags,
        minimum_z,
        maximum_z,
        crate::imod::libimod::imesh::IMESH_CAP_OFF,
    )
}

/// Simple-stack branch with source `doCap` endpoint behavior.
pub fn skin_unconnected_contour_stack_in_range_with_caps(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    flags: u32,
    minimum_z: i32,
    maximum_z: i32,
    do_cap: i32,
) -> i32 {
    skin_unconnected_contour_stack_in_range_with_caps_and_skip(
        object, scale, flags, minimum_z, maximum_z, do_cap, None,
    )
}

/// Cap-aware simple stack with the native cap-skip Z list applied to endpoints.
pub fn skin_unconnected_contour_stack_in_range_with_caps_and_skip(
    object: &mut crate::imod::libimod::imodel::Iobj,
    scale: &crate::imod::libimod::imodel::Ipoint,
    flags: u32,
    minimum_z: i32,
    maximum_z: i32,
    do_cap: i32,
    cap_skip_zlist: Option<&[i32]>,
) -> i32 {
    use crate::imod::libimod::icont::imod_contour_z_value;
    use crate::imod::libimod::imesh::{DEFAULT_VALUE, IMESH_MK_SURF, IMESH_MK_TIME};
    let mut contours: Vec<_> = object
        .cont
        .iter()
        .enumerate()
        .filter(|(_, contour)| {
            let z = imod_contour_z_value(Some(contour));
            !contour.pts.is_empty()
                && (minimum_z == DEFAULT_VALUE || z >= minimum_z)
                && (maximum_z == DEFAULT_VALUE || z <= maximum_z)
        })
        .map(|(index, contour)| (index, contour.clone()))
        .collect();
    contours.sort_by_key(|(_, contour)| imod_contour_z_value(Some(contour)));
    object.mesh.clear();
    if contours.windows(2).any(|pair| {
        imod_contour_z_value(Some(&pair[0].1)) == imod_contour_z_value(Some(&pair[1].1))
    }) {
        return -2;
    }
    for pair in contours.windows(2) {
        if flags & IMESH_MK_SURF != 0 && pair[0].1.surf != pair[1].1.surf {
            continue;
        }
        if flags & IMESH_MK_TIME != 0 && pair[0].1.time != pair[1].1.time {
            continue;
        }
        if mesh_unconnected_contours_to_object(object, &pair[0].1, &pair[1].1, scale, false, flags)
            != 0
        {
            return -1;
        }
    }
    if do_cap != crate::imod::libimod::imesh::IMESH_CAP_OFF && !contours.is_empty() {
        let cap_inside = object.flags & crate::imod::libimod::imodel::IMOD_OBJFLAG_OUT != 0;
        let skipped = |contour: &crate::imod::libimod::imodel::Icont| {
            cap_skip_zlist.is_some_and(|list| list.contains(&imod_contour_z_value(Some(contour))))
        };
        if !skipped(&contours[0].1) {
            if let Some(mesh) = contour_cap_mesh(&contours[0].1, None, -1, cap_inside) {
                if add_mesh_to_object(object, &contours[0].1, mesh, flags) != 0 {
                    return -1;
                }
            }
        }
        let last = contours.len() - 1;
        if !skipped(&contours[last].1) {
            if let Some(mesh) = contour_cap_mesh(&contours[last].1, None, 1, cap_inside) {
                if add_mesh_to_object(object, &contours[last].1, mesh, flags) != 0 {
                    return -1;
                }
            }
        }
    }
    if flags & crate::imod::libimod::imesh::IMESH_MK_NORM != 0 {
        imesh_remesh_normal(&mut object.mesh, scale, 0);
    }
    0
}

/// Original static `cost_from_area_matrices` (`libmesh/mkmesh.c`).
/// Returns the row-major cost and predecessor-path matrices (0 = down, 1 = up).
pub fn cost_from_area_matrices(
    up: &[f32],
    down: &[f32],
    bdim: usize,
    tdim: usize,
    bsize: usize,
    sb: usize,
    st: usize,
    bmax: usize,
    tmax: usize,
    curmin: f64,
) -> (Vec<f64>, Vec<u8>) {
    let stride = bdim + 1;
    let mut cost = vec![0.; stride * (tmax + 1)];
    let mut path = vec![0; stride * (tmax + 1)];
    let mut j = if st == tdim { 0 } else { st };
    let mut jl = 0;
    for l in 0..=tmax {
        let mut i = if sb == bdim { 0 } else { sb };
        let mut il = 0;
        let mut rowmin = 0.;
        for k in 0..=bmax {
            let ind = k + l * stride;
            if k == 0 {
                if l != 0 {
                    cost[ind] = cost[ind - stride] + down[i + jl * bsize] as f64;
                    path[ind] = 0;
                    rowmin = cost[ind];
                }
            } else if l == 0 {
                cost[ind] = cost[ind - 1] + up[il + j * bsize] as f64;
                path[ind] = 1;
            } else {
                let costdown = cost[ind - stride] + down[i + jl * bsize] as f64;
                let costup = cost[ind - 1] + up[il + j * bsize] as f64;
                if costdown < costup {
                    cost[ind] = costdown;
                    path[ind] = 0;
                } else {
                    cost[ind] = costup;
                    path[ind] = 1;
                }
                if cost[ind] < rowmin {
                    rowmin = cost[ind];
                }
            }
            il = i;
            i = (i + 1) % bdim;
        }
        jl = j;
        j = (j + 1) % tdim;
        if curmin >= 0. && rowmin > curmin {
            cost[bmax + tmax * stride] = rowmin;
            break;
        }
    }
    (cost, path)
}

/// Original `imeshSetNewPolyNorm` (`libmesh/remesh.c`).
pub fn imesh_set_new_poly_norm(value: i32) {
    NEW_POLY_NORM.store(value, Ordering::Relaxed);
}

/// Original `imeshSetSkinFlags` (`libmesh/mkmesh.c`).
pub fn imesh_set_skin_flags(flags: i32, fast: i32) {
    SKIN_FLAGS.store(flags, Ordering::Relaxed);
    FAST_MESH.store(fast, Ordering::Relaxed);
}

/// Current source-compatible mkmesh skinning configuration.
pub fn imesh_skin_flags() -> (i32, i32) {
    (
        SKIN_FLAGS.load(Ordering::Relaxed),
        FAST_MESH.load(Ordering::Relaxed),
    )
}

/// Current source-compatible remesh normal encoding selection.
pub fn imesh_new_poly_norm() -> i32 {
    NEW_POLY_NORM.load(Ordering::Relaxed)
}

/// Original `Connector` (`include/mkmesh.h`).
///
/// Connector construction belongs to `mkmesh.c`; keeping the source layout
/// here establishes its owned Rust representation before that algorithm is
/// translated.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Connector {
    pub b1: i32,
    pub t1: i32,
    pub b2: i32,
    pub t2: i32,
    pub gap: i32,
    pub connect: i32,
    pub skip_to_next: i32,
    pub skip_to_end: i32,
    pub skip_from_start: i32,
    pub skip_index: i32,
}

/// Original static `invertConnectors` (`libmesh/mkmesh.c`).
pub fn invert_connectors(connectors: &mut [Connector], direction: [i32; 2]) {
    if connectors.is_empty() {
        return;
    }
    if direction[0] < 0 {
        connectors.reverse();
        let last = connectors.len() - 1;
        let skip_to_end = connectors[0].skip_to_end;
        if connectors[last].skip_from_start != 0 {
            connectors[last].skip_to_end = 1;
            connectors[last].skip_from_start = 0;
        }
        if skip_to_end != 0 {
            connectors[0].skip_from_start = 1;
            connectors[0].skip_to_end = 0;
        }
        let zero_skip = connectors[0].skip_to_next;
        connectors[0].skip_to_next = 0;
        for index in 0..connectors.len() {
            (connectors[index].b1, connectors[index].b2) =
                (connectors[index].b2, connectors[index].b1);
            let next = (index + 1) % connectors.len();
            if next != 0 && connectors[next].skip_to_next != 0 {
                connectors[next].skip_to_next = 0;
                connectors[index].skip_to_next = 1;
            } else if next == 0 {
                connectors[index].skip_to_next = zero_skip;
            }
        }
    }
    if direction[1] < 0 {
        for connector in connectors {
            (connector.t1, connector.t2) = (connector.t2, connector.t1);
        }
    }
}

/// Original `imeshNormal` (`libmesh/remesh.c`).
///
/// Unlike the general `imod_point_normalize` helper, this source routine uses
/// a -Z normal for a degenerate triangle.
pub fn imesh_normal(
    normal: &mut crate::imod::libimod::imodel::Ipoint,
    p1: &crate::imod::libimod::imodel::Ipoint,
    p2: &crate::imod::libimod::imodel::Ipoint,
    p3: &crate::imod::libimod::imodel::Ipoint,
    scale: Option<&crate::imod::libimod::imodel::Ipoint>,
) {
    let mut v1 = crate::imod::libimod::imodel::Ipoint {
        x: p3.x - p2.x,
        y: p3.y - p2.y,
        z: p3.z - p2.z,
    };
    let mut v2 = crate::imod::libimod::imodel::Ipoint {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
        z: p1.z - p2.z,
    };
    if let Some(scale) = scale {
        v1.x *= scale.x;
        v1.y *= scale.y;
        v1.z *= scale.z;
        v2.x *= scale.x;
        v2.y *= scale.y;
        v2.z *= scale.z;
    }
    normal.x = v1.y * v2.z - v1.z * v2.y;
    normal.y = v1.z * v2.x - v1.x * v2.z;
    normal.z = v1.x * v2.y - v1.y * v2.x;
    let distance = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
    if distance == 0. {
        *normal = crate::imod::libimod::imodel::Ipoint {
            x: 0.,
            y: 0.,
            z: -1.,
        };
    } else {
        normal.x /= distance;
        normal.y /= distance;
        normal.z /= distance;
    }
}

/// Generated-polygon portion of `imeshReMeshNormal` (`remesh.c`).
///
/// Mesh chunks made by the skinning paths use `BGNPOLY`; this remaps their
/// repeated vertices into source-style vertex/normal pairs per surface/time
/// and accumulates scaled triangle normals.  Non-polygon or other-resolution
/// meshes are retained unchanged for the remaining complete remesh path.
pub fn imesh_remesh_normal(
    meshes: &mut Vec<crate::imod::libimod::imodel::Imesh>,
    scale: &crate::imod::libimod::imodel::Ipoint,
    resolution: i32,
) {
    use crate::imod::libimod::imesh::{
        IMESH_FLAG_RES_SHIFT, IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2,
        IMOD_MESH_END, IMOD_MESH_ENDPOLY, imesh_resol,
    };
    let input = std::mem::take(meshes);
    let mut output = Vec::new();
    let mut groups = Vec::<(i16, i16)>::new();
    for mesh in &input {
        if imesh_resol(mesh.flag) == resolution && !groups.contains(&(mesh.surf, mesh.time)) {
            groups.push((mesh.surf, mesh.time));
        }
    }
    for mesh in &input {
        if imesh_resol(mesh.flag) != resolution {
            output.push(mesh.clone());
        }
    }
    for (surface, time) in groups {
        let mut triangles = Vec::<[crate::imod::libimod::imodel::Ipoint; 3]>::new();
        let mut unsupported = Vec::new();
        for mesh in input.iter().filter(|mesh| {
            mesh.surf == surface && mesh.time == time && imesh_resol(mesh.flag) == resolution
        }) {
            let mut index = 0;
            let mut found = false;
            while index < mesh.list.len() {
                if mesh.list[index] != IMOD_MESH_BGNPOLY {
                    index += 1;
                    continue;
                }
                found = true;
                index += 1;
                while index < mesh.list.len() && mesh.list[index] != IMOD_MESH_ENDPOLY {
                    if index + 2 >= mesh.list.len() {
                        break;
                    }
                    let ids = [mesh.list[index], mesh.list[index + 1], mesh.list[index + 2]];
                    if ids
                        .iter()
                        .all(|&id| id >= 0 && (id as usize) < mesh.vert.len())
                    {
                        triangles.push([
                            mesh.vert[ids[0] as usize],
                            mesh.vert[ids[1] as usize],
                            mesh.vert[ids[2] as usize],
                        ]);
                    }
                    index += 3;
                }
                index += 1;
            }
            if !found {
                unsupported.push(mesh.clone());
            }
        }
        if triangles.is_empty() {
            output.extend(unsupported);
            continue;
        }
        let mut remesh = crate::imod::libimod::imodel::Imesh {
            surf: surface,
            time,
            flag: (resolution as u32) << IMESH_FLAG_RES_SHIFT,
            ..Default::default()
        };
        let new_encoding = imesh_new_poly_norm() != 0;
        remesh.list.push(if new_encoding {
            IMOD_MESH_BGNPOLYNORM2
        } else {
            IMOD_MESH_BGNPOLYNORM
        });
        for triangle in triangles {
            let mut ids = [0usize; 3];
            for point in 0..3 {
                ids[point] = remesh
                    .vert
                    .iter()
                    .step_by(2)
                    .position(|vertex| *vertex == triangle[point])
                    .map_or_else(
                        || {
                            let index = remesh.vert.len();
                            remesh.vert.push(triangle[point]);
                            remesh.vert.push(Default::default());
                            index
                        },
                        |position| position * 2,
                    );
            }
            if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
                continue;
            }
            let mut normal = Default::default();
            imesh_normal(
                &mut normal,
                &triangle[2],
                &triangle[0],
                &triangle[1],
                Some(scale),
            );
            for id in ids {
                remesh.vert[id + 1].x += normal.x;
                remesh.vert[id + 1].y += normal.y;
                remesh.vert[id + 1].z += normal.z;
                if new_encoding {
                    remesh.list.push(id as i32);
                } else {
                    remesh.list.extend_from_slice(&[(id + 1) as i32, id as i32]);
                }
            }
        }
        remesh
            .list
            .extend_from_slice(&[IMOD_MESH_ENDPOLY, IMOD_MESH_END]);
        output.push(remesh);
        output.extend(unsupported);
    }
    *meshes = output;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Ipoint;

    #[test]
    fn normal_matches_native_scaling_and_degenerate_fallback() {
        let p1 = Ipoint {
            x: 1.,
            y: 0.,
            z: 0.,
        };
        let p2 = Ipoint::default();
        let p3 = Ipoint {
            x: 0.,
            y: 1.,
            z: 0.,
        };
        let mut normal = Ipoint::default();
        imesh_normal(&mut normal, &p1, &p2, &p3, None);
        assert_eq!(
            normal,
            Ipoint {
                x: 0.,
                y: 0.,
                z: -1.
            }
        );
        imesh_normal(&mut normal, &p1, &p1, &p1, None);
        assert_eq!(
            normal,
            Ipoint {
                x: 0.,
                y: 0.,
                z: -1.
            }
        );
    }

    #[test]
    fn remesh_normal_encoding_selector_roundtrips() {
        imesh_set_new_poly_norm(0);
        assert_eq!(imesh_new_poly_norm(), 0);
        imesh_set_new_poly_norm(1);
        assert_eq!(imesh_new_poly_norm(), 1);
    }

    #[test]
    fn remesh_normal_merges_vertices_and_emits_normal_pairs() {
        use crate::imod::libimod::imesh::{
            IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_END, IMOD_MESH_ENDPOLY,
        };
        let triangle = crate::imod::libimod::imodel::Imesh {
            vert: vec![
                Ipoint::default(),
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            list: vec![IMOD_MESH_BGNPOLY, 0, 1, 2, IMOD_MESH_ENDPOLY, IMOD_MESH_END],
            ..Default::default()
        };
        let mut meshes = vec![triangle.clone(), triangle];
        imesh_set_new_poly_norm(1);
        imesh_remesh_normal(
            &mut meshes,
            &Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            0,
        );
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].vert.len(), 6);
        assert_eq!(meshes[0].list[0], IMOD_MESH_BGNPOLYNORM2);
        assert!(meshes[0].vert[1].z > 1.9);
    }

    #[test]
    fn remesh_normal_retains_requested_resolution() {
        use crate::imod::libimod::imesh::{
            IMESH_FLAG_RES_SHIFT, IMOD_MESH_BGNPOLY, IMOD_MESH_END, IMOD_MESH_ENDPOLY,
        };
        let mut meshes = vec![crate::imod::libimod::imodel::Imesh {
            vert: vec![
                Ipoint::default(),
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            list: vec![IMOD_MESH_BGNPOLY, 0, 1, 2, IMOD_MESH_ENDPOLY, IMOD_MESH_END],
            flag: 1 << IMESH_FLAG_RES_SHIFT,
            ..Default::default()
        }];
        imesh_remesh_normal(
            &mut meshes,
            &Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            1,
        );
        assert_eq!(crate::imod::libimod::imesh::imesh_resol(meshes[0].flag), 1);
    }

    #[test]
    fn skinning_configuration_retains_flags_and_fast_mode() {
        imesh_set_skin_flags(0x123, 1);
        assert_eq!(imesh_skin_flags(), (0x123, 1));
        imesh_set_skin_flags(0, 0);
    }

    #[test]
    fn mesh_limits_reject_only_triangles_wholly_outside_xy_bounds() {
        imesh_set_min_max(
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 1.,
                z: 0.,
            },
        );
        let outside = Ipoint {
            x: 2.,
            y: 2.,
            z: 0.,
        };
        assert!(outside_mesh_limits(&outside, &outside, &outside));
        let inside = Ipoint {
            x: 0.5,
            y: 0.5,
            z: 0.,
        };
        assert!(!outside_mesh_limits(&inside, &outside, &outside));
        imesh_set_min_max(
            Ipoint {
                x: -1.0e30,
                y: -1.0e30,
                z: -1.0e30,
            },
            Ipoint {
                x: 1.0e30,
                y: 1.0e30,
                z: 1.0e30,
            },
        );
    }

    #[test]
    fn chunk_index_append_initializes_native_growth_chunk() {
        let mut mesh = crate::imod::libimod::imodel::Imesh::default();
        let mut max_list = 0;
        chunk_mesh_add_index(&mut mesh, 17, &mut max_list);
        assert_eq!(mesh.list, vec![17]);
        assert_eq!(max_list, 8192);
    }

    #[test]
    fn triangle_append_reverses_inside_winding() {
        let mut mesh = crate::imod::libimod::imodel::Imesh::default();
        let mut max_list = 0;
        chunk_add_triangle(&mut mesh, 1, 2, 3, &mut max_list, true);
        assert_eq!(
            mesh.list,
            vec![crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY, 1, 3, 2]
        );
    }

    #[test]
    fn quick_area_returns_twice_scaled_triangle_area() {
        let origin = Ipoint::default();
        let x = Ipoint {
            x: 3.,
            y: 0.,
            z: 0.,
        };
        let y = Ipoint {
            x: 0.,
            y: 4.,
            z: 0.,
        };
        assert_eq!(point_area_quick(&x, &origin, &y, 1.), 12.);
    }

    #[test]
    fn area_matrices_zero_open_contour_wraparound_entries() {
        let bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint::default(),
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let top = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.,
                },
            ],
            ..Default::default()
        };
        let (up, down) = build_area_matrices(
            &bottom,
            1,
            &top,
            1,
            &Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            true,
        );
        assert_eq!(up[1], 0.);
        assert_eq!(up[3], 0.);
        assert_eq!(down[2], 0.);
        assert_eq!(down[3], 0.);
        assert!(up[0] > 0. && down[0] > 0.);
    }

    #[test]
    fn area_matrices_follow_reversed_contour_direction() {
        let bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let top = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 1.,
                },
            ],
            ..Default::default()
        };
        let scale = Ipoint {
            x: 2.,
            y: 1.,
            z: 1.,
        };
        let (forward, _) = build_area_matrices(&bottom, 1, &top, 1, &scale, false);
        let (backward, _) = build_area_matrices(&bottom, -1, &top, 1, &scale, false);
        assert_ne!(forward, backward);
        assert!(backward.iter().all(|area| *area > 0.));
    }

    #[test]
    fn cap_mesh_encodes_triangles_and_style_at_native_list_offsets() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let props = crate::imod::libimod::istore::DrawProps {
            red: 1.,
            ..Default::default()
        };
        let mesh = make_cap_mesh(
            &contour,
            &Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
            0,
            &props,
            1,
            1,
        )
        .unwrap();
        assert_eq!(mesh.vert.len(), 4);
        assert_eq!(
            mesh.list,
            vec![
                crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY,
                1,
                2,
                0,
                crate::imod::libimod::imesh::IMOD_MESH_ENDPOLY,
                crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY,
                2,
                3,
                0,
                crate::imod::libimod::imesh::IMOD_MESH_ENDPOLY,
                crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY,
                3,
                1,
                0,
                crate::imod::libimod::imesh::IMOD_MESH_ENDPOLY,
                crate::imod::libimod::imesh::IMOD_MESH_END,
            ]
        );
        assert_eq!(mesh.store.len(), 9);
        assert_eq!(mesh.store[0].index.i(), 1);
        assert_eq!(mesh.store[8].index.i(), 13);
    }

    #[test]
    fn cap_mesh_omits_triangles_wholly_outside_mesh_bounds() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 0.,
                },
                Ipoint {
                    x: 3.,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        imesh_set_min_max(
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
        );
        let mesh = make_cap_mesh(
            &contour,
            &Ipoint {
                x: 2.,
                y: 3.,
                z: 0.,
            },
            1,
            &crate::imod::libimod::istore::DrawProps::default(),
            0,
            0,
        )
        .unwrap();
        assert_eq!(mesh.list, vec![crate::imod::libimod::imesh::IMOD_MESH_END]);
        imesh_set_min_max(
            Ipoint {
                x: -1.0e30,
                y: -1.0e30,
                z: -1.0e30,
            },
            Ipoint {
                x: 1.0e30,
                y: 1.0e30,
                z: 1.0e30,
            },
        );
    }

    #[test]
    fn tube_ring_has_requested_count_center_and_radius() {
        let mut contour = crate::imod::libimod::imodel::Icont::default();
        assert_eq!(
            make_tube_cont(
                &mut contour,
                &Ipoint {
                    x: 4.,
                    y: 5.,
                    z: 6.
                },
                &Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.
                },
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                4.,
                4,
            ),
            0
        );
        assert_eq!(contour.pts.len(), 4);
        for point in &contour.pts {
            let dx = point.x - 4.;
            let dy = point.y - 5.;
            assert!((dx * dx + dy * dy - 4.).abs() < 1.0e-5);
            assert!((point.z - 6.).abs() < 1.0e-5);
        }
        assert_ne!(contour.pts[0], contour.pts[1]);
    }

    #[test]
    fn tube_join_uses_proportional_triangle_strip_and_syncs_transparency() {
        let first = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
                Ipoint {
                    x: -1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let second = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 1.,
                },
                Ipoint {
                    x: -1.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 1.,
                },
            ],
            ..Default::default()
        };
        let mut first_props = crate::imod::libimod::istore::DrawProps {
            trans: 1,
            ..Default::default()
        };
        let mut second_props = crate::imod::libimod::istore::DrawProps::default();
        let mesh = join_tube_cont(
            &first,
            &second,
            &Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
            &mut first_props,
            1 << 2,
            &mut second_props,
            0,
        )
        .unwrap();
        assert_eq!(mesh.vert.len(), 6);
        assert_eq!(mesh.list.len(), 21);
        assert_eq!(mesh.list[0], crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY);
        assert_eq!(&mesh.list[1..4], &[3, 0, 1]);
        assert_eq!(&mesh.list[4..7], &[3, 1, 4]);
        assert_eq!(
            mesh.list[19],
            crate::imod::libimod::imesh::IMOD_MESH_ENDPOLY
        );
        assert_eq!(mesh.list[20], crate::imod::libimod::imesh::IMOD_MESH_END);
        assert_eq!(second_props.trans, 1);
        assert_eq!(mesh.store.len(), 18);
    }

    #[test]
    fn whole_gap_endpoint_walks_both_sides_without_wrapping_forever() {
        use crate::imod::libimod::istore::{GEN_STORE_GAP, Istore, StoreUnion};
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 6],
            store: vec![
                Istore {
                    type_: GEN_STORE_GAP,
                    flags: 0,
                    index: StoreUnion::from_i(1),
                    value: StoreUnion::default(),
                },
                Istore {
                    type_: GEN_STORE_GAP,
                    flags: 0,
                    index: StoreUnion::from_i(2),
                    value: StoreUnion::default(),
                },
                Istore {
                    type_: GEN_STORE_GAP,
                    flags: 0,
                    index: StoreUnion::from_i(3),
                    value: StoreUnion::default(),
                },
            ],
            ..Default::default()
        };
        // From point 2, native scans backward through 1 to establish the new
        // start, then forward through 3 to return the non-gap endpoint 4.
        assert_eq!(ends_of_whole_gap(&contour, 2), (4, 1));
    }

    #[test]
    fn implicit_connector_inserts_paired_numbered_endpoints_in_direction_order() {
        use crate::imod::libimod::istore::istore_connect_number;
        let mut bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 4],
            ..Default::default()
        };
        let mut top = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 4],
            ..Default::default()
        };
        let mut maximum = 7;
        assert_eq!(
            add_connector_if_none(&mut bottom, 0, &mut top, 0, &mut maximum, [1, -1]),
            0
        );
        assert_eq!(maximum, 9);
        assert_eq!(istore_connect_number(&bottom.store, 0), 8);
        assert_eq!(istore_connect_number(&bottom.store, 1), 9);
        assert_eq!(istore_connect_number(&top.store, 1), 8);
        assert_eq!(istore_connect_number(&top.store, 0), 9);
        // Existing endpoint numbers inhibit the native fallback.
        assert_eq!(
            add_connector_if_none(&mut bottom, 0, &mut top, 0, &mut maximum, [1, -1]),
            0
        );
        assert_eq!(maximum, 9);
    }

    #[test]
    fn explicit_connectors_match_numbers_and_expand_gap_third_points() {
        use crate::imod::libimod::istore::{GEN_STORE_CONNECT, GEN_STORE_GAP, Istore, StoreUnion};
        let connector = |index, number| Istore {
            type_: GEN_STORE_CONNECT,
            flags: 0,
            index: StoreUnion::from_i(index),
            value: StoreUnion::from_i(number),
        };
        let gap = |index| Istore {
            type_: GEN_STORE_GAP,
            flags: 0,
            index: StoreUnion::from_i(index),
            value: StoreUnion::default(),
        };
        let bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 4],
            store: vec![connector(1, 14), gap(1), connector(2, 14)],
            ..Default::default()
        };
        let top = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 4],
            store: vec![connector(1, 14), gap(1), connector(2, 14)],
            ..Default::default()
        };
        let connectors = make_connectors(&bottom, &top, true, 1).unwrap();
        assert_eq!(connectors.len(), 1);
        assert_eq!(
            (
                connectors[0].b1,
                connectors[0].b2,
                connectors[0].t1,
                connectors[0].t2
            ),
            (1, 2, 1, 2)
        );
        assert_eq!(connectors[0].gap, 1);
    }

    #[test]
    fn same_plane_connector_mode_rejects_multiple_matches() {
        use crate::imod::libimod::istore::{GEN_STORE_CONNECT, Istore, StoreUnion};
        let connector = |index, number| Istore {
            type_: GEN_STORE_CONNECT,
            flags: 0,
            index: StoreUnion::from_i(index),
            value: StoreUnion::from_i(number),
        };
        let bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 3],
            store: vec![connector(0, 1), connector(1, 2)],
            ..Default::default()
        };
        let top = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 3],
            store: vec![connector(0, 1), connector(1, 2)],
            ..Default::default()
        };
        assert!(make_connectors(&bottom, &top, true, 0).is_none());
    }

    #[test]
    fn cost_backtrace_emits_native_reverse_path_order() {
        let (_, path) = cost_from_area_matrices(&[1.], &[1.], 1, 1, 1, 0, 0, 1, 1, -1.);
        assert_eq!(
            backtrace_cost_path(&path, 1, 1, 1),
            vec![CostPathStep::Up, CostPathStep::Down]
        );
    }

    #[test]
    fn unconnected_contour_pair_emits_native_marked_triangle_mesh() {
        let bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let top = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 1.,
                },
            ],
            ..Default::default()
        };
        let mesh = mesh_unconnected_contour_pair(
            &bottom,
            &top,
            &Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            false,
            [1, 1],
            [0, 0],
            false,
        )
        .unwrap();
        assert_eq!(mesh.vert.len(), 8);
        assert_eq!(mesh.list.len(), 21);
        assert_eq!(mesh.list[0], crate::imod::libimod::imesh::IMOD_MESH_BGNPOLY);
        assert_eq!(
            mesh.list[19],
            crate::imod::libimod::imesh::IMOD_MESH_ENDPOLY
        );
        assert_eq!(mesh.list[20], crate::imod::libimod::imesh::IMOD_MESH_END);
    }

    #[test]
    fn gap_management_pairs_open_contour_endpoints_in_closed_object() {
        use crate::imod::libimod::imodel::ICONT_OPEN;
        use crate::imod::libimod::istore::{istore_connect_number, istore_point_is_gap};
        let mut bottom = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint::default(),
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 0.,
                },
            ],
            flags: ICONT_OPEN,
            ..Default::default()
        };
        let mut top = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 1.,
                },
            ],
            flags: ICONT_OPEN,
            ..Default::default()
        };
        assert_eq!(
            manage_gaps(
                &mut bottom,
                &mut top,
                true,
                [1, 1],
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                }
            ),
            0
        );
        assert_eq!(istore_point_is_gap(&bottom.store, 2), 1);
        assert_eq!(istore_point_is_gap(&top.store, 2), 1);
        assert!(istore_connect_number(&bottom.store, 2) > 0);
        assert_eq!(
            istore_connect_number(&bottom.store, 2),
            istore_connect_number(&top.store, 2)
        );
    }

    #[test]
    fn delete_meshes_at_resolution_retains_other_resolution_order() {
        use crate::imod::libimod::imesh::IMESH_FLAG_RES_SHIFT;
        let mut meshes = vec![
            crate::imod::libimod::imodel::Imesh {
                flag: 0 << IMESH_FLAG_RES_SHIFT,
                surf: 1,
                ..Default::default()
            },
            crate::imod::libimod::imodel::Imesh {
                flag: 1 << IMESH_FLAG_RES_SHIFT,
                surf: 2,
                ..Default::default()
            },
            crate::imod::libimod::imodel::Imesh {
                flag: 0 << IMESH_FLAG_RES_SHIFT,
                surf: 3,
                ..Default::default()
            },
        ];
        assert_eq!(imod_meshes_delete_res(&mut meshes, 0), 0);
        assert_eq!(
            meshes.iter().map(|mesh| mesh.surf).collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(imod_meshes_delete_res(&mut Vec::new(), 0), -1);
    }

    #[test]
    fn marked_contour_duplication_consumes_marker_and_omits_mesh_data() {
        let marker = 1 << 27;
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            crate::imod::libimod::imodel::Icont {
                pts: vec![Ipoint::default()],
                flags: marker,
                surf: 2,
                ..Default::default()
            },
            crate::imod::libimod::imodel::Icont {
                pts: vec![Ipoint::default()],
                surf: 3,
                ..Default::default()
            },
            crate::imod::libimod::imodel::Icont {
                flags: marker,
                surf: 4,
                ..Default::default()
            },
        ];
        object
            .mesh
            .push(crate::imod::libimod::imodel::Imesh::default());
        object.mesh_param = Some(crate::imod::libimod::imesh::MeshParams::default());
        let duplicate = imesh_dup_marked_conts(&mut object, marker).unwrap();
        assert_eq!(duplicate.cont.len(), 1);
        assert_eq!(duplicate.cont[0].surf, 2);
        assert!(duplicate.mesh.is_empty());
        assert!(duplicate.mesh_param.is_none());
        assert_eq!(object.cont[0].flags & marker, 0);
        assert_eq!(object.cont[2].flags & marker, 0);
    }

    #[test]
    fn resection_removes_empty_out_of_range_and_off_increment_contours() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint { x: 0., y: 0., z }],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            contour(0.),
            contour(1.),
            contour(2.),
            contour(3.),
            crate::imod::libimod::imodel::Icont::default(),
        ];
        assert_eq!(resection_object(&mut object, 0, 2, 2), 0);
        assert_eq!(
            object
                .cont
                .iter()
                .map(|contour| contour.pts[0].z)
                .collect::<Vec<_>>(),
            vec![0., 2.]
        );
    }

    #[test]
    fn point_reduction_preserves_loopback_contour() {
        let points = vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
        ];
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: points.clone(),
            ..Default::default()
        });
        assert_eq!(reduce_object_contours(&mut object, 5.), 0);
        assert_eq!(object.cont[0].pts, points);
    }

    #[test]
    fn clean_zero_coordinates_only_snaps_strictly_small_components() {
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint {
                x: 0.0009,
                y: -0.0009,
                z: 0.001,
            }],
            ..Default::default()
        });
        clean_zero_coordinates(&mut object);
        assert_eq!(
            object.cont[0].pts[0],
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.001
            }
        );
    }

    #[test]
    fn open_end_extension_projects_nearby_phantom_gap_to_matching_z() {
        use crate::imod::libimod::imodel::ICONT_OPEN;
        use crate::imod::libimod::istore::{
            GEN_STORE_GAP, GEN_STORE_ONEPOINT, Istore, StoreUnion, istore_point_is_gap,
        };
        let gap = |index| Istore {
            type_: GEN_STORE_GAP,
            flags: GEN_STORE_ONEPOINT,
            index: StoreUnion::from_i(index),
            value: StoreUnion::default(),
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            crate::imod::libimod::imodel::Icont {
                pts: vec![
                    Ipoint {
                        x: -1.,
                        y: 0.,
                        z: 0.,
                    },
                    Ipoint {
                        x: 0.,
                        y: 0.,
                        z: 0.,
                    },
                    Ipoint {
                        x: 2.,
                        y: 0.,
                        z: 0.,
                    },
                ],
                flags: ICONT_OPEN,
                store: vec![gap(0), gap(1)],
                ..Default::default()
            },
            crate::imod::libimod::imodel::Icont {
                pts: vec![
                    Ipoint {
                        x: 0.,
                        y: 0.,
                        z: 1.,
                    },
                    Ipoint {
                        x: 2.,
                        y: 0.,
                        z: 1.,
                    },
                    Ipoint {
                        x: 4.,
                        y: 0.,
                        z: 1.,
                    },
                ],
                flags: ICONT_OPEN,
                ..Default::default()
            },
        ];
        assert_eq!(extend_open_ends(&mut object), 0);
        assert_eq!(object.cont[1].pts.len(), 7);
        assert_eq!(
            object.cont[1].pts[0],
            Ipoint {
                x: -1.,
                y: 0.,
                z: 1.
            }
        );
        assert_eq!(istore_point_is_gap(&object.cont[1].store, 0), 1);
    }

    #[test]
    fn contour_preparation_preserves_native_first_point_mean_z_quirk() {
        use crate::imod::libimod::imesh::DEFAULT_VALUE;
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.0005,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 3.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 5.,
                },
            ],
            ..Default::default()
        });
        assert_eq!(
            imesh_prep_contours(&mut object, DEFAULT_VALUE, DEFAULT_VALUE, 1, 0., true),
            0
        );
        assert_eq!(
            object.cont[0].pts[0],
            Ipoint {
                x: 0.,
                y: 0.,
                z: 1.
            }
        );
        assert_eq!(object.cont[0].pts[1].z, 3.);
        assert_eq!(object.cont[0].pts[2].z, 3.);
    }

    #[test]
    fn next_mesh_z_returns_next_listed_or_one_past_maximum() {
        assert_eq!(next_mesh_z(&[2, 5, 9], 2), 5);
        assert_eq!(next_mesh_z(&[2, 5, 9], 9), 10);
        assert_eq!(next_mesh_z(&[2, 5, 9], 4), 10);
    }

    #[test]
    fn segment_separation_preserves_native_integer_truncation_ranking() {
        assert_eq!(segment_separation(1., 3., 0., 4.), 0.);
        assert_eq!(segment_separation(0., 2.9, 5., 8.), 4.1);
        assert_eq!(segment_separation(0., 5., 3., 7.), 2.);
    }

    #[test]
    fn mesh_commit_applies_requested_surface_and_time_tags() {
        use crate::imod::libimod::imesh::{IMESH_MK_SURF, IMESH_MK_TIME};
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        let contour = crate::imod::libimod::imodel::Icont {
            surf: 7,
            time: 3,
            ..Default::default()
        };
        assert_eq!(
            add_mesh_to_object(
                &mut object,
                &contour,
                crate::imod::libimod::imodel::Imesh::default(),
                IMESH_MK_SURF | IMESH_MK_TIME
            ),
            0
        );
        assert_eq!((object.mesh[0].surf, object.mesh[0].time), (7, 3));
    }

    #[test]
    fn interpolation_uses_native_first_to_second_fraction() {
        assert_eq!(
            interpolate_point(
                Ipoint {
                    x: 2.,
                    y: 4.,
                    z: 6.
                },
                Ipoint {
                    x: 10.,
                    y: 14.,
                    z: 18.
                },
                0.25,
            ),
            Ipoint {
                x: 4.,
                y: 6.5,
                z: 9.
            }
        );
    }

    #[test]
    fn overlap_elimination_returns_immediately_for_disjoint_contours() {
        let contour = |offset| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: offset,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: offset + 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: offset,
                    y: 1.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let mut first = contour(0.);
        let mut second = contour(10.);
        assert_eq!(eliminate_overlap(&mut first, &mut second), 0);
    }

    #[test]
    fn simple_convexity_uses_consistent_turn_orientation() {
        let contour = |points| crate::imod::libimod::imodel::Icont {
            pts: points,
            ..Default::default()
        };
        let square = contour(vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 1.,
                z: 0.,
            },
            Ipoint {
                x: 0.,
                y: 1.,
                z: 0.,
            },
        ]);
        let concave = contour(vec![
            Ipoint {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
            Ipoint {
                x: 1.,
                y: 0.5,
                z: 0.,
            },
            Ipoint {
                x: 2.,
                y: 2.,
                z: 0.,
            },
            Ipoint {
                x: 0.,
                y: 2.,
                z: 0.,
            },
        ]);
        assert_eq!(contour_convex_if_simple(Some(&square)), 1);
        assert_eq!(contour_convex_if_simple(Some(&concave)), 0);
        assert_eq!(contour_convex_if_simple(None), -1);
    }

    #[test]
    fn concavity_fraction_matches_hull_area_difference() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let (fraction, difference) = concavity_area_fraction(Some(&contour));
        assert!((difference - 1.).abs() < 1.0e-6);
        assert!((fraction - 0.25).abs() < 1.0e-6);
        assert_eq!(concavity_area_fraction(None), (-1., 0.));
    }

    #[test]
    fn cyclic_segment_bounds_include_wrapped_endpoints() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 4.,
                    y: 1.,
                    z: 0.,
                },
                Ipoint {
                    x: 3.,
                    y: 5.,
                    z: 0.,
                },
                Ipoint {
                    x: -2.,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            contour_segment_min_max(Some(&contour), 3, 1),
            Some((-2., 4., 0., 2.))
        );
        assert_eq!(contour_segment_min_max(None, 0, 0), None);
    }

    #[test]
    fn segment_crossing_skips_incident_contour_edges() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            segment_crosses_contour(Some(&contour), -1., 1., 3., 1., 4, 5),
            1
        );
        assert_eq!(
            segment_crosses_contour(Some(&contour), 0., 0., 2., 0., 0, 1),
            0
        );
    }

    #[test]
    fn legal_joiner_applies_adjacent_slice_nesting_rule() {
        let contour = |xmin: f32, xmax: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: xmin,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: xmax,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: xmax,
                    y: 2.,
                    z: 0.,
                },
                Ipoint {
                    x: xmin,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let first = contour(0., 2.);
        let second = contour(8., 10.);
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(contour(2., 8.));
        let left = Ipoint {
            x: 2.,
            y: 1.,
            z: 0.,
        };
        let right = Ipoint {
            x: 8.,
            y: 1.,
            z: 0.,
        };
        assert_eq!(
            check_legal_joiner(left, right, &first, &second, true, &object, &[0], 1, None),
            1
        );
        assert_eq!(
            check_legal_joiner(left, right, &first, &second, false, &object, &[0], 1, None),
            0
        );
        assert_eq!(
            check_legal_joiner(left, right, &first, &second, true, &object, &[99], 1, None),
            0
        );
    }

    #[test]
    fn contour_segment_scan_reports_closest_projected_connector() {
        let first = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint {
                x: 0.,
                y: 0.,
                z: 7.,
            }],
            ..Default::default()
        };
        let second = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 3.,
                    y: -1.,
                    z: 4.,
                },
                Ipoint {
                    x: 3.,
                    y: 1.,
                    z: 4.,
                },
            ],
            ..Default::default()
        };
        let object = crate::imod::libimod::imodel::Iobj::default();
        let (legal, fallback) =
            scan_points_to_segments(&first, &second, false, false, &object, &[], 0, None);
        let candidate = legal.unwrap();
        assert_eq!(Some(candidate), fallback);
        assert_eq!(candidate.point_index, 0);
        assert_eq!(candidate.segment_index, 0);
        assert!((candidate.t - 0.5).abs() < 1.0e-6);
        assert_eq!(
            candidate.close,
            Ipoint {
                x: 3.,
                y: 0.,
                z: 4.
            }
        );
        assert!((candidate.distance_squared - 9.).abs() < 1.0e-6);
    }

    #[test]
    fn closest_contour_uses_directional_scan_after_nesting_check() {
        let contour = |xmin: f32, xmax: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: xmin,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: xmax,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: xmax,
                    y: 1.,
                    z: 0.,
                },
                Ipoint {
                    x: xmax,
                    y: 2.,
                    z: 0.,
                },
                Ipoint {
                    x: xmin,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let target = contour(0., 2.);
        let joined = contour(8., 10.);
        let adjacent = contour(2., 8.);
        let object = crate::imod::libimod::imodel::Iobj::default();
        let matched = find_closest_contour(
            &target,
            Some(&joined),
            &object,
            &[],
            &[],
            &[],
            0,
            Some(&adjacent),
            0,
        )
        .unwrap();
        assert_eq!(matched.list_position, 1);
        assert!(matched.first_contour_point);
        assert_eq!(matched.point_one_index, 2);
        assert_eq!(
            matched.close,
            Ipoint {
                x: 8.,
                y: 1.,
                z: 0.
            }
        );
    }

    #[test]
    fn join_all_contours_applies_closest_connector_and_join() {
        let triangle = |x: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x, y: 0., z: 0. },
                Ipoint {
                    x: x + 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint { x, y: 1., z: 0. },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![triangle(0.), triangle(5.)];
        let joined = join_all_contours(&object, &[0, 1], 2, 0, &[], 0, None).unwrap();
        // The nearest approach is an interior edge projection, so the native
        // join path inserts its connector point before combining contours.
        assert_eq!(joined.pts.len(), 8);
        assert!(joined.pts.iter().any(|point| point.x >= 5.));
    }

    #[test]
    fn scan_contour_subtraction_splits_truncates_and_removes_intervals() {
        let intervals = |values: &[(f32, f32)]| crate::imod::libimod::imodel::Icont {
            pts: values
                .iter()
                .flat_map(|&(start, end)| {
                    [
                        Ipoint {
                            x: start,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: end,
                            y: 0.,
                            z: 0.,
                        },
                    ]
                })
                .collect(),
            ..Default::default()
        };
        let cutter = intervals(&[(3., 7.)]);
        let mut split = intervals(&[(0., 10.)]);
        subtract_scan_contours(&mut split, &cutter);
        assert_eq!(
            split.pts.iter().map(|point| point.x).collect::<Vec<_>>(),
            vec![0., 3., 7., 10.]
        );

        let mut truncated = intervals(&[(0., 5.)]);
        subtract_scan_contours(&mut truncated, &cutter);
        assert_eq!(
            truncated
                .pts
                .iter()
                .map(|point| point.x)
                .collect::<Vec<_>>(),
            vec![0., 3.]
        );

        let mut removed = intervals(&[(4., 6.)]);
        subtract_scan_contours(&mut removed, &cutter);
        assert!(removed.pts.is_empty());
    }

    #[test]
    fn inout_break_adds_fill_midpoint_and_preserves_return_order() {
        use crate::imod::libimod::imodel::ICONT_OPEN;
        let contour = crate::imod::libimod::imodel::Icont {
            flags: ICONT_OPEN,
            pts: (0..5)
                .map(|x| Ipoint {
                    x: x as f32,
                    y: 0.,
                    z: 2.,
                })
                .collect(),
            ..Default::default()
        };
        let (first, second) = break_contour_inout(Some(&contour), 1, 3, 1).unwrap();
        assert_eq!(first.pts.len(), 5);
        assert_eq!(second.pts.len(), 4);
        assert_ne!(first.flags & ICONT_OPEN, 0);
        assert_eq!(second.flags & ICONT_OPEN, 0);
        assert_eq!(
            first.pts[2],
            Ipoint {
                x: 2.,
                y: 0.,
                z: 2.75
            }
        );
        assert_eq!(
            second.pts[0],
            Ipoint {
                x: 2.,
                y: 0.,
                z: 2.75
            }
        );

        let (reversed_first, reversed_second) =
            break_contour_inout(Some(&contour), 3, 1, 0).unwrap();
        assert_eq!(reversed_first.pts.len(), 3);
        assert_eq!(reversed_second.pts.len(), 4);
    }

    #[test]
    fn skip_slice_next_z_requires_an_exact_current_entry() {
        assert_eq!(get_next_z(&[2, 5, 9], 2), 5);
        assert_eq!(get_next_z(&[2, 5, 9], 5), 9);
        assert_eq!(get_next_z(&[2, 5, 9], 9), 10);
        assert_eq!(get_next_z(&[2, 5, 9], 3), 10);
        assert_eq!(get_next_z(&[], -1), 0);
    }

    #[test]
    fn skeleton_pixel_completion_requires_all_zero_counts() {
        assert!(all_pixels_processed(&[]));
        assert!(all_pixels_processed(&[1, -1, 2]));
        assert!(!all_pixels_processed(&[0, 1, 0]));
        let mut axis = vec![0u8; 25];
        axis[2 * 5 + 2] = 1;
        axis[2 * 5 + 3] = 1;
        assert_eq!(find_skeleton_start_point(5, 5, &axis), Some((2, 2)));
    }

    #[test]
    fn freeman_step_and_xy_comparator_match_native_helpers() {
        let mut x = 3;
        let mut y = 4;
        take_freeman_step(1, &mut x, &mut y);
        assert_eq!((x, y), (4, 3));

        let low = Ipoint {
            x: 1.,
            y: 9.,
            z: 0.,
        };
        let high = Ipoint {
            x: 2.,
            y: 0.,
            z: 0.,
        };
        let same_x_lower_y = Ipoint {
            x: 1.,
            y: 2.,
            z: 0.,
        };
        // Native `remesh.c:ptcompare` returns -1 and 1 respectively for
        // these two probes (compiled against the vendored IMOD headers).
        assert!(compare_points_xy(&low, &high).is_lt());
        assert!(compare_points_xy(&low, &same_x_lower_y).is_gt());
        assert!(compare_f32_native(1., 2.).is_lt());
        assert_eq!(compare_f32_native(f32::NAN, 1.), std::cmp::Ordering::Equal);
    }

    #[test]
    fn skeleton_linear_arm_reducer_removes_mirrored_straight_points() {
        let mut x = vec![0, 1, 2, 10, 9, 2, 1];
        let mut y = vec![0; 7];
        assert!(eliminate_skeleton_linear_part(0, 7, &mut x, &mut y));
        assert_eq!(x, vec![2, 10, 9]);
        assert_eq!(y, vec![0; 3]);
    }

    #[test]
    fn skeleton_axis_trace_follows_and_closes_simple_line() {
        let mut axis = vec![0u8; 25];
        axis[2 * 5 + 1] = 1;
        axis[2 * 5 + 2] = 1;
        axis[2 * 5 + 3] = 1;
        assert_eq!(
            trace_skeleton_axis(5, 5, &mut axis),
            vec![(1, 2), (2, 2), (3, 2), (2, 2)]
        );
    }

    #[test]
    fn skeleton_trace_mapping_scales_and_closes_contour() {
        let contour = skeleton_trace_to_contour(&[(1, 1), (3, 2)], 5, 5, 10, 20, 9, 5, 7.);
        assert_eq!(
            contour.pts,
            vec![
                Ipoint {
                    x: 10.,
                    y: 20.,
                    z: 7.
                },
                Ipoint {
                    x: 18.,
                    y: 22.,
                    z: 7.
                },
                Ipoint {
                    x: 10.,
                    y: 20.,
                    z: 7.
                },
            ]
        );
    }

    #[test]
    fn skeleton_trace_mapping_uses_native_integer_scale() {
        let contour = skeleton_trace_to_contour(&[(1, 1), (4, 4)], 6, 6, 0, 0, 9, 9, 0.);
        assert_eq!(
            contour.pts[1],
            Ipoint {
                x: 6.,
                y: 6.,
                z: 0.
            }
        );
    }

    #[test]
    fn zhang_suen_thinning_preserves_center_of_solid_block() {
        let mut bitmap = vec![0u8; 49];
        for y in 2..5 {
            for x in 2..5 {
                bitmap[y * 7 + x] = 1;
            }
        }
        let thinned = skeletonize_bitmap(&bitmap, 7, 7).unwrap();
        assert_ne!(thinned[3 * 7 + 3], 0);
        assert!(thinned.iter().filter(|&&value| value != 0).count() < 9);
    }

    #[test]
    fn skeletonize_contour_composes_scan_thinning_trace_and_mapping() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 3.,
                },
                Ipoint {
                    x: 4.,
                    y: 0.,
                    z: 3.,
                },
            ],
            ..Default::default()
        };
        let result = skeletonize_contour(
            &contour,
            Some(&contour),
            Ipoint {
                x: 2.,
                y: 0.,
                z: 9.,
            },
        )
        .unwrap();
        assert!(result.pts.len() >= 2);
        assert_eq!(result.pts.first(), result.pts.last());
        assert!(result.pts.iter().all(|point| point.z == 9.));
    }

    #[test]
    fn skeleton_smoothing_handles_paired_pixel_boundary_path() {
        let mut contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 1.,
                    z: 0.,
                },
                Ipoint {
                    x: 3.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 1.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        assert_eq!(smooth_reduce_skeleton(Some(&mut contour)), 0);
        assert!(contour.pts.len() <= 7);
        assert!(contour.pts.windows(2).all(|pair| pair[0] != pair[1]));
        let mut short = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 4],
            ..Default::default()
        };
        assert_eq!(smooth_reduce_skeleton(Some(&mut short)), 1);
    }

    #[test]
    fn break_evaluation_rejects_when_geometric_area_bound_cannot_improve() {
        let outer = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 4.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 4.,
                    y: 4.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 4.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let (score, inner_area) =
            evaluate_break(&outer, &[], &[], &mut [], &[], 0, 1, 1., 0., 16., 1, false);
        assert!(score > 1.);
        assert!((inner_area - 16. / 12.5664).abs() < 1.0e-5);
    }

    #[test]
    fn whole_nest_selects_level_one_outer_group_once() {
        use crate::imod::libimod::icont::Nesting;
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![Default::default(), Default::default(), Default::default()];
        let nests = vec![
            Nesting {
                co: 0,
                outside: vec![1],
                ..Default::default()
            },
            Nesting {
                co: 1,
                level: 1,
                inside: vec![2],
                ..Default::default()
            },
        ];
        let mut list = Vec::new();
        add_whole_nest(0, &mut object, &nests, &[0, 1, 1], 0x40, &mut list);
        assert_eq!(list, vec![1, 2]);
        add_whole_nest(0, &mut object, &nests, &[0, 1, 1], 0x40, &mut list);
        assert_eq!(list, vec![1, 2]);
    }

    #[test]
    fn skin_nesting_builds_interior_subtracted_scan_contours() {
        let square = |low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: low,
                    y: low,
                    z: 0.,
                },
                Ipoint {
                    x: high,
                    y: low,
                    z: 0.,
                },
                Ipoint {
                    x: high,
                    y: high,
                    z: 0.,
                },
                Ipoint {
                    x: low,
                    y: high,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![square(0., 6.), square(2., 4.)];
        let (_scan, nests, indices) = skin_object_nesting(&object, 0).unwrap();
        assert!(indices.iter().all(|&index| index >= 0));
        let outer = nests.iter().find(|nest| nest.co == 0).unwrap();
        assert_eq!(outer.inside, vec![1]);
        assert!(outer.inscan.is_some());
    }

    #[test]
    fn skin_z_groups_sort_and_filter_native_sections() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint { x: 0., y: 0., z }, Ipoint { x: 1., y: 0., z }],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            contour(4.),
            contour(0.),
            Default::default(),
            contour(2.),
            contour(1.),
        ];
        assert_eq!(
            skin_object_z_groups(&object, 2),
            vec![(0, vec![1]), (2, vec![3]), (4, vec![0])]
        );
    }

    #[test]
    fn closed_skin_groups_mesh_non_nested_multi_contour_sections() {
        let contour = |z, x| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x, y: 0., z },
                Ipoint {
                    x: x + 1.,
                    y: 0.,
                    z,
                },
                Ipoint { x, y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            contour(0., 0.),
            contour(0., 4.),
            contour(1., 0.),
            contour(1., 4.),
        ];
        assert_eq!(
            skin_closed_contour_groups(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                1,
                0
            ),
            0
        );
        assert!(!object.mesh.is_empty());
    }

    #[test]
    fn closed_skin_groups_mesh_matched_nested_levels() {
        let square = |z, low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: low, y: low, z },
                Ipoint { x: high, y: low, z },
                Ipoint {
                    x: high,
                    y: high,
                    z,
                },
                Ipoint { x: low, y: high, z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            square(0., 0., 8.),
            square(0., 2., 6.),
            square(1., 0., 8.),
            square(1., 2., 6.),
        ];
        assert_eq!(
            skin_closed_contour_groups(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                1,
                0
            ),
            0
        );
        assert_eq!(object.mesh.len(), 2);
    }

    #[test]
    fn closed_skin_groups_routes_unmatched_inner_contour_to_orphan_path() {
        let square = |z, low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: low, y: low, z },
                Ipoint { x: high, y: low, z },
                Ipoint {
                    x: high,
                    y: high,
                    z,
                },
                Ipoint { x: low, y: high, z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![square(0., 0., 8.), square(0., 2., 6.), square(1., 0., 8.)];
        assert_eq!(
            skin_closed_contour_groups(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                1,
                0
            ),
            0
        );
        assert_eq!(object.mesh.len(), 1);
    }

    #[test]
    fn skin_object_dispatches_non_nested_multi_contour_sections() {
        use crate::imod::libimod::imesh::IMESH_CAP_OFF;
        let contour = |z, x| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x, y: 0., z },
                Ipoint {
                    x: x + 1.,
                    y: 0.,
                    z,
                },
                Ipoint { x, y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            contour(0., 0.),
            contour(0., 4.),
            contour(1., 0.),
            contour(1., 4.),
        ];
        assert_eq!(
            imesh_skin_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                0.,
                IMESH_CAP_OFF,
                None,
                1,
                0,
                1,
                0.,
                None
            ),
            0
        );
        assert!(!object.mesh.is_empty());
    }

    #[test]
    fn skin_overlap_matrix_tracks_matching_nested_working_scans() {
        let square = |z, low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: low, y: low, z },
                Ipoint { x: high, y: low, z },
                Ipoint {
                    x: high,
                    y: high,
                    z,
                },
                Ipoint { x: low, y: high, z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            square(0., 0., 4.),
            square(0., 1., 3.),
            square(1., 0., 4.),
            square(1., 1., 3.),
        ];
        let (scan, nests, indices) = skin_object_nesting(&object, 0).unwrap();
        let (minimum, maximum) =
            skin_overlap_matrix(&object, &scan, &nests, &indices, &[0, 1], &[2, 3]).unwrap();
        assert_eq!(minimum.len(), 4);
        assert_eq!(maximum.len(), 4);
        assert!(maximum.iter().all(|value| *value >= 0.));
    }

    #[test]
    fn skin_overlap_components_group_same_level_connected_pairs() {
        let minimum = vec![0.8, 0.1, 0.7, 0.2];
        let maximum = vec![0.9, 0.2, 0.8, 0.8];
        assert_eq!(
            skin_overlap_components(&minimum, &maximum, &[1, 1], &[1, 1], 0.5),
            vec![(vec![0, 1], vec![0, 1])]
        );
    }

    #[test]
    fn skin_group_levels_use_nesting_records_or_outer_default() {
        use crate::imod::libimod::icont::Nesting;
        let nests = vec![Nesting {
            level: 2,
            ..Default::default()
        }];
        assert_eq!(
            skin_group_nesting_levels(&[0, 1, 2], &[-1, 0, -1], &nests),
            vec![1, 2, 1]
        );
    }

    #[test]
    fn nested_group_analysis_matches_outer_and_inner_levels() {
        let square = |z, low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: low, y: low, z },
                Ipoint { x: high, y: low, z },
                Ipoint {
                    x: high,
                    y: high,
                    z,
                },
                Ipoint { x: low, y: high, z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            square(0., 0., 6.),
            square(0., 2., 4.),
            square(1., 0., 6.),
            square(1., 2., 4.),
        ];
        let components = skin_nested_group_components(&object, &[0, 1], &[2, 3], 0, 0.).unwrap();
        assert!(!components.is_empty());
    }

    #[test]
    fn immediate_inner_positions_exclude_already_matched_holes() {
        use crate::imod::libimod::icont::Nesting;
        let nests = vec![
            Nesting {
                co: 0,
                level: 1,
                inside: vec![1, 2],
                ..Default::default()
            },
            Nesting {
                co: 1,
                level: 2,
                outside: vec![0],
                ..Default::default()
            },
            Nesting {
                co: 2,
                level: 2,
                outside: vec![0],
                ..Default::default()
            },
        ];
        assert_eq!(
            skin_immediate_inner_positions(&[0, 1, 2], &[0], &[1], &[0, 1, 2], &nests),
            vec![2]
        );
    }

    #[test]
    fn skin_dump_lists_uses_one_based_direct_or_lookup_indices() {
        assert_eq!(
            skin_dump_lists("state", &[0, 2], &[1], Some(&[9, 8, 7]), None),
            "state:\nbottom: 10 8\ntop: 2\n"
        );
    }

    #[test]
    fn skin_component_lists_preserve_object_indices() {
        let components = vec![(vec![1, 0], vec![0])];
        assert_eq!(
            skin_component_contour_lists(&components, &[10, 11], &[20]),
            Some(vec![(vec![11, 10], vec![20])])
        );
    }

    #[test]
    fn skin_component_join_builds_both_group_paths() {
        let contour = |z, x| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x, y: 0., z },
                Ipoint {
                    x: x + 1.,
                    y: 0.,
                    z,
                },
                Ipoint { x, y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            contour(0., 0.),
            contour(0., 3.),
            contour(1., 0.),
            contour(1., 3.),
        ];
        let (bottom, top) = skin_join_component_contours(&object, &[0, 1], &[2, 3], 1).unwrap();
        assert!(bottom.pts.len() >= 3 && top.pts.len() >= 3);
    }

    #[test]
    fn skin_orphan_cap_classifier_marks_nearly_contained_contours() {
        use crate::imod::libimod::icont::{imod_contour_get_bbox, imodel_contour_scan};
        let square = |low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: low,
                    y: low,
                    z: 0.,
                },
                Ipoint {
                    x: high,
                    y: low,
                    z: 0.,
                },
                Ipoint {
                    x: high,
                    y: high,
                    z: 0.,
                },
                Ipoint {
                    x: low,
                    y: high,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let outer = square(0., 10.);
        let inner = square(1., 9.);
        let mut scans = vec![imodel_contour_scan(Some(&inner)).unwrap()];
        let (mut low, mut high) = (Ipoint::default(), Ipoint::default());
        imod_contour_get_bbox(Some(&inner), &mut low, &mut high);
        let mut used = vec![0];
        assert!(skin_mark_capped_orphans(
            &outer,
            &[0],
            &mut used,
            &[0],
            &mut scans,
            &[(low, high)]
        ));
        assert_eq!(used, vec![-1]);
    }

    #[test]
    fn orphan_break_candidates_are_ordered_by_outer_path_length() {
        let square = |offset: f32, size: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: offset,
                    y: offset,
                    z: 0.,
                },
                Ipoint {
                    x: offset + size,
                    y: offset,
                    z: 0.,
                },
                Ipoint {
                    x: offset + size,
                    y: offset + size,
                    z: 0.,
                },
                Ipoint {
                    x: offset,
                    y: offset + size,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let candidates = skin_orphan_break_candidates(&square(0., 10.), &square(3., 4.));
        assert!(!candidates.is_empty());
        assert!(candidates.windows(2).all(|pair| pair[0].2 >= pair[1].2));
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.0 < 4 && candidate.1 < 4)
        );
    }

    #[test]
    fn orphan_connector_keeps_cap_classified_contours_out_of_joining() {
        use crate::imod::libimod::icont::{imod_contour_get_bbox, imodel_contour_scan};
        let square = |low: f32, high: f32| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: low,
                    y: low,
                    z: 0.,
                },
                Ipoint {
                    x: high,
                    y: low,
                    z: 0.,
                },
                Ipoint {
                    x: high,
                    y: high,
                    z: 0.,
                },
                Ipoint {
                    x: low,
                    y: high,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let outer = square(0., 10.);
        let inner = square(1., 9.);
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(inner.clone());
        let mut scans = vec![imodel_contour_scan(Some(&inner)).unwrap()];
        let (mut low, mut high) = (Ipoint::default(), Ipoint::default());
        imod_contour_get_bbox(Some(&inner), &mut low, &mut high);
        let mut used = vec![0];
        let remaining = skin_connect_orphans(
            &mut object,
            outer.clone(),
            &[0],
            &mut used,
            &[0],
            1,
            false,
            0,
            0,
            &Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
            &mut scans,
            &[(low, high)],
        )
        .unwrap();
        assert_eq!(used, vec![-1]);
        assert_eq!(remaining.pts, outer.pts);
        assert!(object.mesh.is_empty());
    }

    #[test]
    fn contour_cap_uses_center_and_native_side_offset() {
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 3.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 3.,
                },
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 3.,
                },
                Ipoint {
                    x: 0.,
                    y: 2.,
                    z: 3.,
                },
            ],
            ..Default::default()
        };
        let mesh = contour_cap_mesh(&contour, None, -1, false).unwrap();
        assert_eq!(
            mesh.vert[0],
            Ipoint {
                x: 1.,
                y: 1.,
                z: 2.5
            }
        );
        let inside_mesh = contour_cap_mesh(&contour, None, -1, true).unwrap();
        assert_ne!(mesh.list, inside_mesh.list);
    }

    #[test]
    fn simple_stack_caps_propagate_object_inside_out_winding() {
        use crate::imod::libimod::imesh::IMESH_CAP_ALL;
        use crate::imod::libimod::imodel::IMOD_OBJFLAG_OUT;
        let contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 2.,
                    z: 0.,
                },
            ],
            ..Default::default()
        };
        let mut normal = crate::imod::libimod::imodel::Iobj::default();
        normal.cont.push(contour.clone());
        let mut inside_out = crate::imod::libimod::imodel::Iobj::default();
        inside_out.flags |= IMOD_OBJFLAG_OUT;
        inside_out.cont.push(contour);
        let scale = Ipoint {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        assert_eq!(
            skin_unconnected_contour_stack_in_range_with_caps(
                &mut normal,
                &scale,
                0,
                -999,
                999,
                IMESH_CAP_ALL
            ),
            0
        );
        assert_eq!(
            skin_unconnected_contour_stack_in_range_with_caps(
                &mut inside_out,
                &scale,
                0,
                -999,
                999,
                IMESH_CAP_ALL
            ),
            0
        );
        assert_eq!(normal.mesh.len(), 2);
        assert_eq!(inside_out.mesh.len(), 2);
        assert_ne!(normal.mesh[0].list, inside_out.mesh[0].list);
    }

    #[test]
    fn robust_center_uses_contour_center_of_mass_when_available() {
        let mut contour = crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 2.,
                },
                Ipoint {
                    x: 2.,
                    y: 0.,
                    z: 2.,
                },
                Ipoint {
                    x: 2.,
                    y: 2.,
                    z: 2.,
                },
                Ipoint {
                    x: 0.,
                    y: 2.,
                    z: 2.,
                },
            ],
            ..Default::default()
        };
        let mut center = Ipoint::default();
        assert_eq!(robust_center_of_mass(&mut contour, None, &mut center), 0);
        assert_eq!(
            center,
            Ipoint {
                x: 1.,
                y: 1.,
                z: 2.
            }
        );
    }

    #[test]
    fn unconnected_pair_commits_source_format_mesh_to_object() {
        use crate::imod::libimod::imesh::IMESH_MK_SURF;
        let contour = |z, surf| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            surf,
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        assert_eq!(
            mesh_unconnected_contours_to_object(
                &mut object,
                &contour(0., 4),
                &contour(1., 4),
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                false,
                IMESH_MK_SURF
            ),
            0
        );
        assert_eq!(object.mesh.len(), 1);
        assert_eq!(object.mesh[0].surf, 4);
        assert_eq!(
            *object.mesh[0].list.last().unwrap(),
            crate::imod::libimod::imesh::IMOD_MESH_END
        );
    }

    #[test]
    fn simple_stack_skinning_connects_each_neighboring_z_pair() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(2.), contour(0.), contour(1.)];
        assert_eq!(
            skin_unconnected_contour_stack(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                0
            ),
            0
        );
        assert_eq!(object.mesh.len(), 2);
    }

    #[test]
    fn simple_stack_defers_multi_contour_section_to_nesting_dispatch() {
        let contour = |z, offset| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: offset,
                    y: 0.,
                    z,
                },
                Ipoint {
                    x: offset + 1.,
                    y: 0.,
                    z,
                },
                Ipoint {
                    x: offset,
                    y: 1.,
                    z,
                },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(0., 0.), contour(0., 3.), contour(1., 0.)];
        assert_eq!(
            skin_unconnected_contour_stack(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                0
            ),
            -2
        );
        assert!(object.mesh.is_empty());
    }

    #[test]
    fn simple_stack_range_limits_connected_sections() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(0.), contour(1.), contour(2.)];
        assert_eq!(
            skin_unconnected_contour_stack_in_range(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                0,
                1,
                2
            ),
            0
        );
        assert_eq!(object.mesh.len(), 1);
    }

    #[test]
    fn simple_stack_cap_mode_adds_both_endpoint_caps() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(0.), contour(1.)];
        assert_eq!(
            skin_unconnected_contour_stack_in_range_with_caps(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                0,
                0,
                1,
                crate::imod::libimod::imesh::IMESH_CAP_ALL
            ),
            0
        );
        assert_eq!(object.mesh.len(), 3);
    }

    #[test]
    fn simple_stack_cap_skip_list_omits_matching_endpoint() {
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(0.), contour(1.)];
        assert_eq!(
            skin_unconnected_contour_stack_in_range_with_caps_and_skip(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                0,
                0,
                1,
                crate::imod::libimod::imesh::IMESH_CAP_ALL,
                Some(&[0])
            ),
            0
        );
        assert_eq!(object.mesh.len(), 2);
    }

    #[test]
    fn skin_object_dispatches_closed_stacks_and_propagates_callback_stop() {
        use crate::imod::libimod::imesh::IMESH_CAP_OFF;
        use crate::imod::libimod::imodel::IMOD_OBJFLAG_SCAT;
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let scale = Ipoint {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(0.), contour(1.)];
        assert_eq!(
            imesh_skin_object(
                &mut object,
                &scale,
                0.,
                IMESH_CAP_OFF,
                None,
                1,
                0,
                1,
                0.,
                None
            ),
            0
        );
        assert!(!object.mesh.is_empty());

        fn stop_at_quarter(status: i32) -> i32 {
            if status == 25 { 17 } else { 0 }
        }
        assert_eq!(
            imesh_skin_object(
                &mut object,
                &scale,
                0.,
                IMESH_CAP_OFF,
                None,
                1,
                0,
                1,
                0.,
                Some(stop_at_quarter)
            ),
            17
        );
        assert!(object.mesh.is_empty());

        object.flags |= IMOD_OBJFLAG_SCAT;
        assert_eq!(
            imesh_skin_object(
                &mut object,
                &scale,
                0.,
                IMESH_CAP_OFF,
                None,
                1,
                0,
                1,
                0.,
                None
            ),
            -1
        );
    }

    #[test]
    fn analyze_prep_uses_mesh_parameters_and_rejects_tilted_subset_path() {
        use crate::imod::libimod::imesh::MeshParams;
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let scale = Ipoint {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![contour(0.), contour(1.)];
        object.mesh_param = Some(MeshParams {
            passes: 1,
            ..MeshParams::default()
        });
        assert_eq!(analyze_prep_skin_object(&mut object, 0, &scale, None), 0);
        assert!(!object.mesh.is_empty());

        let mut tilted = crate::imod::libimod::imodel::Iobj::default();
        tilted.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 2.,
                },
                Ipoint {
                    x: 0.,
                    y: 1.,
                    z: 0.,
                },
            ],
            ..Default::default()
        });
        tilted.mesh_param = Some(MeshParams {
            flat_crit: 1.,
            ..MeshParams::default()
        });
        assert_eq!(analyze_prep_skin_object(&mut tilted, 0, &scale, None), -2);
    }

    #[test]
    fn joined_contour_groups_mesh_multiple_outer_contours() {
        let contour = |z, x| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x, y: 0., z },
                Ipoint {
                    x: x + 1.,
                    y: 0.,
                    z,
                },
                Ipoint { x, y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont = vec![
            contour(0., 0.),
            contour(0., 4.),
            contour(1., 0.),
            contour(1., 4.),
        ];
        assert_eq!(
            mesh_joined_contour_groups(
                &mut object,
                &[0, 1],
                &[2, 3],
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                false,
                0
            ),
            0
        );
        assert!(!object.mesh.is_empty());
    }

    #[test]
    fn open_surface_pairs_nearest_contours_once_per_section() {
        use crate::imod::libimod::icont::{ICONT_CONNECT_BOTTOM, ICONT_CONNECT_TOP};
        use crate::imod::libimod::imodel::IMOD_OBJFLAG_OPEN;
        let contour = |z, offset| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: offset,
                    y: 0.,
                    z,
                },
                Ipoint {
                    x: offset + 1.,
                    y: 0.,
                    z,
                },
                Ipoint {
                    x: offset,
                    y: 1.,
                    z,
                },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.flags |= IMOD_OBJFLAG_OPEN;
        object.cont = vec![
            contour(0., 0.),
            contour(0., 20.),
            contour(1., 0.2),
            contour(1., 20.2),
        ];
        assert_eq!(
            mesh_open_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                1,
                0,
                1
            ),
            0
        );
        // A connection can be emitted as multiple mesh chunks by the cost
        // mesher; the connection flags are the one-pair-per-contour invariant.
        assert!(!object.mesh.is_empty());
        assert_ne!(object.cont[0].flags & ICONT_CONNECT_TOP, 0);
        assert_ne!(object.cont[1].flags & ICONT_CONNECT_TOP, 0);
        assert_ne!(object.cont[2].flags & ICONT_CONNECT_BOTTOM, 0);
        assert_ne!(object.cont[3].flags & ICONT_CONNECT_BOTTOM, 0);
    }

    #[test]
    fn open_surface_range_excludes_connections_outside_bounds() {
        use crate::imod::libimod::imodel::IMOD_OBJFLAG_OPEN;
        let contour = |z| crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint { x: 0., y: 0., z },
                Ipoint { x: 1., y: 0., z },
                Ipoint { x: 0., y: 1., z },
            ],
            ..Default::default()
        };
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.flags |= IMOD_OBJFLAG_OPEN;
        object.cont = vec![contour(0.), contour(1.), contour(2.)];
        assert_eq!(
            mesh_open_object_in_range(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                1,
                0,
                1,
                1,
                2
            ),
            0
        );
        assert!(!object.mesh.is_empty());
        assert_eq!(
            object.cont[0].flags & crate::imod::libimod::icont::ICONT_CONNECT_TOP,
            0
        );
    }

    #[test]
    fn open_tube_builds_rings_joins_and_flat_end_caps() {
        use crate::imod::libimod::imesh::{IMESH_MK_CAP_TUBE, IMESH_MK_TUBE};
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 2.,
                },
            ],
            ..Default::default()
        });
        assert_eq!(
            mesh_open_tube_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                IMESH_MK_TUBE | IMESH_MK_CAP_TUBE,
                24.
            ),
            0
        );
        assert_eq!(object.mesh.len(), 4);
        assert_eq!(object.mesh[1].vert.len(), 24);
    }

    #[test]
    fn open_tube_skips_join_at_general_storage_gap() {
        use crate::imod::libimod::imesh::IMESH_MK_TUBE;
        use crate::imod::libimod::istore::{GEN_STORE_GAP, Istore, StoreUnion};
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 2.,
                },
            ],
            store: vec![Istore {
                type_: GEN_STORE_GAP,
                index: StoreUnion::from_i(0),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert_eq!(
            mesh_open_tube_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                IMESH_MK_TUBE,
                24.
            ),
            0
        );
        assert_eq!(object.mesh.len(), 1);
    }

    #[test]
    fn open_tube_uses_contour_storage_width() {
        use crate::imod::libimod::imesh::IMESH_MK_TUBE;
        use crate::imod::libimod::istore::{GEN_STORE_3DWIDTH, Istore, StoreUnion};
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.store.push(Istore {
            type_: GEN_STORE_3DWIDTH,
            index: StoreUnion::from_i(0),
            value: StoreUnion::from_i(48),
            ..Default::default()
        });
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
            ],
            ..Default::default()
        });
        assert_eq!(
            mesh_open_tube_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                IMESH_MK_TUBE,
                0.
            ),
            0
        );
        assert_eq!(object.mesh.len(), 1);
        assert_eq!(object.mesh[0].vert.len(), 48);
    }

    #[test]
    fn open_tube_applies_point_storage_width_changes() {
        use crate::imod::libimod::imesh::IMESH_MK_TUBE;
        use crate::imod::libimod::istore::{GEN_STORE_3DWIDTH, Istore, StoreUnion};
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
            ],
            store: vec![Istore {
                type_: GEN_STORE_3DWIDTH,
                index: StoreUnion::from_i(1),
                value: StoreUnion::from_i(48),
                ..Default::default()
            }],
            ..Default::default()
        });
        assert_eq!(
            mesh_open_tube_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                IMESH_MK_TUBE,
                0.
            ),
            0
        );
        // The initial 12-point ring joins the 24-point ring selected at point 1.
        assert_eq!(object.mesh[0].vert.len(), 36);
    }

    #[test]
    fn open_tube_contour_no_cap_suppresses_flat_caps() {
        use crate::imod::libimod::imesh::{IMESH_MK_CAP_TUBE, IMESH_MK_TUBE};
        use crate::imod::libimod::istore::{GEN_STORE_NO_CAP, Istore, StoreUnion};
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.store.push(Istore {
            type_: GEN_STORE_NO_CAP,
            index: StoreUnion::from_i(0),
            ..Default::default()
        });
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
            ],
            ..Default::default()
        });
        assert_eq!(
            mesh_open_tube_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                IMESH_MK_TUBE | IMESH_MK_CAP_TUBE,
                24.
            ),
            0
        );
        assert_eq!(object.mesh.len(), 1);
    }

    #[test]
    fn dome_contours_and_tube_domes_follow_scaled_endpoints() {
        use crate::imod::libimod::imesh::{IMESH_MK_CAP_DOME, IMESH_MK_TUBE};
        let ring = crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default(); 24],
            ..Default::default()
        };
        let scale = Ipoint {
            x: 2.,
            y: 2.,
            z: 2.,
        };
        let (_, cap) = make_dome_contours(
            &ring,
            &Ipoint::default(),
            &Ipoint {
                x: 0.,
                y: 0.,
                z: 1.,
            },
            &scale,
            24.,
            -1.,
        );
        assert!((cap.z + 6.).abs() < 1.0e-5);
        let mut object = crate::imod::libimod::imodel::Iobj::default();
        object.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                },
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 2.,
                },
            ],
            ..Default::default()
        });
        assert_eq!(
            mesh_open_tube_object(
                &mut object,
                &Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 1.
                },
                IMESH_MK_TUBE | IMESH_MK_CAP_DOME,
                24.
            ),
            0
        );
        assert_eq!(object.mesh.len(), 8);
    }

    #[test]
    fn cost_matrix_prefers_up_path_on_equal_costs() {
        let (cost, path) = cost_from_area_matrices(&[1.], &[1.], 1, 1, 1, 0, 0, 1, 1, -1.);
        assert_eq!(cost, vec![0., 1., 1., 2.]);
        assert_eq!(path, vec![0, 1, 0, 1]);
    }

    #[test]
    fn cost_matrix_records_row_minimum_when_pruned() {
        let (cost, _) = cost_from_area_matrices(&[3.], &[2.], 1, 1, 1, 0, 0, 1, 1, 0.5);
        assert_eq!(cost[3], 2.);
    }

    #[test]
    fn connector_inversion_reverses_bottom_and_swaps_both_endpoints() {
        let mut connectors = vec![
            Connector {
                b1: 1,
                b2: 2,
                t1: 3,
                t2: 4,
                ..Default::default()
            },
            Connector {
                b1: 5,
                b2: 6,
                t1: 7,
                t2: 8,
                ..Default::default()
            },
        ];
        invert_connectors(&mut connectors, [-1, -1]);
        assert_eq!(
            (
                connectors[0].b1,
                connectors[0].b2,
                connectors[0].t1,
                connectors[0].t2
            ),
            (6, 5, 8, 7)
        );
    }
}
