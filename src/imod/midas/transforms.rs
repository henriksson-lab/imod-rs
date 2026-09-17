//! Translation of the non-Qt portions of `IMOD/midas/transforms.cpp`.
//!
//! The source uses a nine-element, column-major homogeneous transform.  This
//! unit deliberately retains that representation: it is the file format and
//! is also what the rest of MIDAS passes to the display code.

use super::midas::{MIDAS_DEBUG, MidasTransform, MidasView};
use crate::imod::libcfshr::find_piece_shifts::find_piece_shifts;
use crate::imod::libcfshr::linearxforms::{xf_invert, xf_mult};
use std::path::Path;
use std::sync::atomic::Ordering;

/// `new_view` (`transforms.cpp:33`), relocated to the owning MIDAS state
/// module and re-exported here for source-compatible call sites.
pub fn new_view() -> MidasView {
    super::midas::new_view()
}

/// `load_view` (`transforms.cpp:143`), delegated to the owned MRC loading
/// boundary in `file_io` rather than retaining C FILE/Qt string handling.
pub fn load_view(view: &mut MidasView, filename: &Path) -> Result<i32, String> {
    super::file_io::load_image(view, filename)
}

/// C `Islice` subset consumed by `midas_transform`.
#[derive(Clone, Debug, PartialEq)]
pub struct MidasSlice {
    pub xsize: i32,
    pub ysize: i32,
    pub mean: f32,
    pub data: Vec<u8>,
}

/// C `tramat_create` (`transforms.cpp:1234`).
pub fn tramat_create() -> [f32; 9] {
    let mut mat = [0.0; 9];
    tramat_idmat(&mut mat);
    mat
}

/// C `tramat_free` (`transforms.cpp:1254`).  Rust owns matrix storage.
pub fn tramat_free(_mat: [f32; 9]) {}

/// C `tramat_idmat` (`transforms.cpp:1260`).
pub fn tramat_idmat(mat: &mut [f32; 9]) -> i32 {
    *mat = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    0
}

/// C `tramat_copy` (`transforms.cpp:1266`).
pub fn tramat_copy(fmat: &[f32; 9], tomat: &mut [f32; 9]) -> i32 {
    *tomat = *fmat;
    0
}

/// C `tramat_multiply` (`transforms.cpp:1272`), matching `xfMult` order.
pub fn tramat_multiply(m2: &[f32; 9], m1: &[f32; 9], out: &mut [f32; 9]) -> i32 {
    xf_mult(m2, m1, out, 3);
    0
}

/// C `tramat_inverse` (`transforms.cpp:1279`).
pub fn tramat_inverse(mat: &[f32; 9]) -> Option<[f32; 9]> {
    let determinant = mat[0] * mat[4] - mat[3] * mat[1];
    if determinant == 0.0 {
        return None;
    }
    let mut inverse = [0.; 9];
    xf_invert(mat, &mut inverse, 3);
    Some(inverse)
}

/// C `tramat_translate` (`transforms.cpp:1287`).
pub fn tramat_translate(mat: &mut [f32; 9], x: f64, y: f64) -> i32 {
    mat[6] += x as f32;
    mat[7] += y as f32;
    0
}

/// C `tramat_scale` (`transforms.cpp:1294`).
pub fn tramat_scale(mat: &mut [f32; 9], x: f64, y: f64) -> i32 {
    let mut scale = tramat_create();
    scale[0] = x as f32;
    scale[4] = y as f32;
    let mut output = [0.; 9];
    tramat_multiply(mat, &scale, &mut output);
    *mat = output;
    0
}

/// C `tramat_rot` (`transforms.cpp:1310`).
pub fn tramat_rot(mat: &mut [f32; 9], angle: f64) -> i32 {
    let radians = angle * 0.017453293;
    let (sine, cosine) = radians.sin_cos();
    let mut rotation = tramat_create();
    rotation[0] = cosine as f32;
    rotation[1] = sine as f32;
    rotation[3] = -sine as f32;
    rotation[4] = cosine as f32;
    let mut output = [0.; 9];
    tramat_multiply(mat, &rotation, &mut output);
    *mat = output;
    0
}

/// C `rotate_transform` (`transforms.cpp:1396`).
pub fn rotate_transform(mat: &mut [f32; 9], angle: f64) {
    let mut rotation = tramat_create();
    tramat_rot(&mut rotation, -angle);
    let mut product = [0.; 9];
    tramat_multiply(&rotation, mat, &mut product);
    tramat_rot(&mut product, angle);
    *mat = product;
}

/// C `rotate_all_transforms` (`transforms.cpp:1406`).
pub fn rotate_all_transforms(view: &mut MidasView, angle: f64) {
    for transform in &mut view.tr {
        rotate_transform(&mut transform.mat, angle);
    }
}

/// C `stretch_transform` (`transforms.cpp:1416`).
pub fn stretch_transform(view: &MidasView, mat: &mut [f32; 9], index: usize, destretch: i32) {
    if index == 0 || index >= view.tilt_angles.len() {
        return;
    }
    let previous = (view.tilt_angles[index - 1] - view.tilt_offset).to_radians();
    let current = (view.tilt_angles[index] - view.tilt_offset).to_radians();
    let stretch = previous.cos() / current.cos();
    if destretch != 0 {
        mat[6] /= stretch;
    } else {
        mat[6] *= stretch;
    }
}

/// C `stretch_all_transforms` (`transforms.cpp:1431`).
pub fn stretch_all_transforms(view: &mut MidasView, destretch: i32) {
    if !MIDAS_DEBUG.load(Ordering::Relaxed) {
        eprintln!("{}tretching all", if destretch != 0 { "Des" } else { "S" });
    }
    let angles = view.tilt_angles.clone();
    let offset = view.tilt_offset;
    for (index, transform) in view.tr.iter_mut().enumerate() {
        if index != 0 && index < angles.len() {
            let stretch = (angles[index - 1] - offset).to_radians().cos()
                / (angles[index] - offset).to_radians().cos();
            if destretch != 0 {
                transform.mat[6] /= stretch;
            } else {
                transform.mat[6] *= stretch;
            }
        }
    }
}

/// C `translate_slice` (`transforms.cpp:720`), source fill/shift order.
pub fn translate_slice(slice: &mut MidasSlice, xt: i32, yt: i32) -> i32 {
    if slice.xsize <= 0
        || slice.ysize <= 0
        || slice.data.len() != (slice.xsize * slice.ysize) as usize
    {
        return -1;
    }
    let mean = slice.mean as u8;
    let source = slice.data.clone();
    for y in 0..slice.ysize {
        for x in 0..slice.xsize {
            let sx = x - xt;
            let sy = y - yt;
            slice.data[(x + y * slice.xsize) as usize] =
                if sx >= 0 && sx < slice.xsize && sy >= 0 && sy < slice.ysize {
                    source[(sx + sy * slice.xsize) as usize]
                } else {
                    mean
                };
        }
    }
    0
}

/// C `midas_transform` regular affine branch (`transforms.cpp:801`).  Warping
/// is deliberately rejected until the whole `libwarp` grid closure is active.
pub fn midas_transform(
    view: &MidasView,
    _zval: i32,
    input: &MidasSlice,
    output: &mut MidasSlice,
    matrix: &[f32; 9],
    izwarp: i32,
) -> i32 {
    if izwarp >= 0 {
        return -2;
    }
    let Some(inverse) = tramat_inverse(matrix) else {
        return -1;
    };
    if input.xsize != output.xsize
        || input.ysize != output.ysize
        || input.data.len() != output.data.len()
    {
        return -1;
    }
    let xcenter = view.xsize as f32 / 2.;
    let ycenter = view.ysize as f32 / 2.;
    let xbase = inverse[6] + xcenter - xcenter * inverse[0] - ycenter * inverse[3];
    let ybase = inverse[7] + ycenter - xcenter * inverse[1] - ycenter * inverse[4];
    let mean = input.mean as u8;
    for y in 0..input.ysize {
        for x in 0..input.xsize {
            let fx = xbase + x as f32 * inverse[0] + y as f32 * inverse[3];
            let fy = ybase + x as f32 * inverse[1] + y as f32 * inverse[4];
            let pos = (x + y * input.xsize) as usize;
            if view.fast_interp != 0 {
                let sx = (fx + 0.5) as i32;
                let sy = (fy + 0.5) as i32;
                output.data[pos] = if sx >= 0 && sx < input.xsize && sy >= 0 && sy < input.ysize {
                    input.data[(sx + sy * input.xsize) as usize]
                } else {
                    mean
                };
            } else {
                let sx = fx as i32;
                let sy = fy as i32;
                output.data[pos] =
                    if sx >= 0 && sx < input.xsize - 1 && sy >= 0 && sy < input.ysize - 1 {
                        let dx = fx - sx as f32;
                        let dy = fy - sy as f32;
                        let at = |xx, yy| input.data[(xx + yy * input.xsize) as usize] as f32;
                        ((1. - dy) * ((1. - dx) * at(sx, sy) + dx * at(sx + 1, sy))
                            + dy * ((1. - dx) * at(sx, sy + 1) + dx * at(sx + 1, sy + 1)))
                            as u8
                    } else {
                        mean
                    };
            }
        }
    }
    0
}

/// C `getXformSlice`: transform an owned cached source section using the
/// section's active affine matrix.  Cache eviction and image-file reading stay
/// with the caller's owned cache boundary.
pub fn get_xform_slice(
    view: &MidasView,
    input: &MidasSlice,
    zval: usize,
    _shift_ok: bool,
) -> Result<(MidasSlice, bool), String> {
    let transform = view
        .tr
        .get(zval)
        .ok_or_else(|| format!("no transform for MIDAS section {zval}"))?;
    let identity = transform.mat == [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    if identity {
        return Ok((input.clone(), false));
    }
    let mut output = input.clone();
    if midas_transform(view, zval as i32, input, &mut output, &transform.mat, -1) != 0 {
        return Err(format!("cannot transform MIDAS section {zval}"));
    }
    Ok((output, true))
}

/// C `flush_xformed` (`transforms.cpp:696`).
pub fn flush_xformed(view: &mut MidasView) {
    for cache in &mut view.cache {
        cache.xformed = 0;
    }
}
/// C `midasGetSize` (`transforms.cpp:706`).
pub fn midas_get_size(view: &MidasView, xs: &mut i32, ys: &mut i32) {
    *xs = view.xsize;
    *ys = view.ysize;
}
/// C `includedEdge` (`transforms.cpp:1703`), currently no montage map closure.
pub fn included_edge(_mapind: i32, _xory: i32) -> i32 {
    0
}
/// C `nearest_section` (`transforms.cpp:1786`).
pub fn nearest_section(view: &MidasView, section: i32, direction: i32) -> i32 {
    let next = section + direction;
    if next >= 0 && next < view.zsize {
        next
    } else {
        section
    }
}
/// C `set_mont_pieces` (`transforms.cpp:1835`).
pub fn set_mont_pieces(_view: &mut MidasView) {}

/// C `getRawSlice` (`transforms.cpp`): cache ownership is in the source
/// `Islice`/Midas GL closure and is therefore an explicit boundary.
pub fn get_raw_slice(_view: &mut MidasView, _zval: i32) -> Result<MidasSlice, String> {
    Err("MIDAS Islice cache ownership is not yet translated".into())
}
/// C `midasGetSlice` (`transforms.cpp`).
pub fn midas_get_slice(_view: &mut MidasView, _slice_type: i32) -> Result<MidasSlice, String> {
    Err("MIDAS display slice cache is not yet translated".into())
}
/// C `midasGetPrevImage` (`transforms.cpp`).
pub fn midas_get_prev_image(_view: &mut MidasView) -> Result<Vec<u8>, String> {
    Err("MIDAS previous-image cache is not yet translated".into())
}
/// C `fillWarpingGrid` (`transforms.cpp:1196`).
pub fn fill_warping_grid(_iz: i32) -> Result<(i32, i32, f32, f32, f32, f32), String> {
    Err("MIDAS warping grid requires the libwarp storage closure".into())
}
/// C `global_rot_transform` (`transforms.cpp`).
pub fn global_rot_transform(
    _view: &MidasView,
    _input: &MidasSlice,
    _output: &mut MidasSlice,
    _zval: i32,
) -> Result<i32, String> {
    Err("MIDAS global rotation needs display/cache slice ownership".into())
}
/// C `transform_model` (`transforms.cpp:1440`).
pub fn transform_model(_input: &str, _output: &str, _view: &MidasView) -> Result<(), String> {
    Err("MIDAS model transformation requires the imodel object-I/O closure".into())
}
/// C `reduceControlPoints` (`transforms.cpp:1468`).
pub fn reduce_control_points(_view: &mut MidasView) -> Result<(), String> {
    Err("MIDAS control-point reduction requires the libwarp closure".into())
}
/// C `adjustControlPoints` (`transforms.cpp:1607`).
pub fn adjust_control_points(_view: &mut MidasView) -> Result<(), String> {
    Err("MIDAS control-point adjustment requires the libwarp closure".into())
}
/// C `nearest_edge` (`transforms.cpp:1714`).
pub fn nearest_edge(
    _view: &MidasView,
    _z: i32,
    _xory: i32,
    _edgeno: i32,
    _direction: i32,
    _edgeind: &mut i32,
) -> Result<i32, String> {
    Err("MIDAS montage edge graph is not yet translated".into())
}
/// C `find_best_shifts` (`transforms.cpp:1924`).
pub fn find_best_shifts(_view: &mut MidasView) -> Result<(), String> {
    Err("MIDAS montage least-squares closure is not yet translated".into())
}
/// C `find_local_errors` (`transforms.cpp:2032`).
pub fn find_local_errors(_view: &mut MidasView) -> Result<(), String> {
    Err("MIDAS montage local-error closure is not yet translated".into())
}
/// C `crossCorrelate` (`transforms.cpp:2285`).
pub fn cross_correlate(_view: &mut MidasView) -> Result<(), String> {
    Err("MIDAS correlation uses the source FFT/display cache closure".into())
}

/// `xcorrRange` (`transforms.cpp:2271`).  Returns the inclusive portion of a
/// correlation box that remains valid after a rounded image translation.
pub fn xcorr_range(
    size: i32,
    shift: f32,
    center: i32,
    border: i32,
    box_size: i32,
) -> Option<(i32, i32)> {
    if size <= 0 || border < 0 || box_size <= 0 {
        return None;
    }
    let rounded_shift = shift.round() as i32;
    let box_low = center - box_size / 2;
    let box_high = box_low + box_size - 1;
    let first = (border + rounded_shift.max(0)).max(box_low);
    let last = (size - 1 - border - rounded_shift.min(0)).min(box_high);
    (first <= last).then_some((first, last))
}

/// Result of `checklist` (`transforms.cpp:1651`), which determines whether a
/// set of montage coordinates forms a regular one-dimensional grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MidasPieceChecklist {
    pub minimum_piece: i32,
    /// `-1` retains the source marker for an irregular grid or a pitch larger
    /// than the source frame.
    pub pieces: i32,
    pub overlap: i32,
}

/// One montage adjacency relation used by `piecesNotConnected`.  The source
/// stores horizontal/vertical links in parallel arrays; an owned edge record
/// makes the same connectivity relation explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MidasMontageEdge {
    pub lower_piece: usize,
    pub upper_piece: usize,
    pub skipped: bool,
}

/// `lowerEdgeIfIncluded` (`transforms.cpp:1850`), with `None` replacing the
/// source's negative edge sentinel.
pub fn lower_edge_if_included(
    edge: Option<usize>,
    skipped: bool,
    exclude_skipped: bool,
) -> Option<usize> {
    (!exclude_skipped || !skipped).then_some(edge).flatten()
}

/// `upperEdgeIfIncluded` (`transforms.cpp:1858`), identical policy applied
/// to the upper adjacency table in the native representation.
pub fn upper_edge_if_included(
    edge: Option<usize>,
    skipped: bool,
    exclude_skipped: bool,
) -> Option<usize> {
    (!exclude_skipped || !skipped).then_some(edge).flatten()
}

/// `checkPathList` (`transforms.cpp:1868`).  `labels` replaces `leaveType`
/// and `pending` replaces the C `pathList`; true means this edge joined the
/// two endpoint searches, so an alternate path exists.
pub fn check_path_list(
    labels: &mut [u8],
    pending: &mut Vec<usize>,
    old_piece: usize,
    new_piece: usize,
) -> Result<bool, ()> {
    let label = *labels.get(old_piece).ok_or(())?;
    if label == 0 {
        return Err(());
    }
    let target = labels.get_mut(new_piece).ok_or(())?;
    if *target != 0 {
        return Ok(*target != label);
    }
    *target = label;
    pending.push(new_piece);
    Ok(false)
}

/// `piecesNotConnected` (`transforms.cpp:1886`).  Returns true when omitting
/// `leave_edge` disconnects its endpoints, respecting the source option that
/// omits skipped edges from the path search.
pub fn pieces_not_connected(
    piece_count: usize,
    edges: &[MidasMontageEdge],
    leave_edge: usize,
    exclude_skipped: bool,
) -> Result<bool, ()> {
    let leaving = edges.get(leave_edge).ok_or(())?;
    if leaving.lower_piece >= piece_count || leaving.upper_piece >= piece_count {
        return Err(());
    }
    let mut seen = vec![false; piece_count];
    let mut queue = std::collections::VecDeque::from([leaving.lower_piece]);
    seen[leaving.lower_piece] = true;
    while let Some(piece) = queue.pop_front() {
        for (index, edge) in edges.iter().enumerate() {
            if index == leave_edge || (exclude_skipped && edge.skipped) {
                continue;
            }
            if edge.lower_piece >= piece_count || edge.upper_piece >= piece_count {
                return Err(());
            }
            let neighbor = if edge.lower_piece == piece {
                Some(edge.upper_piece)
            } else if edge.upper_piece == piece {
                Some(edge.lower_piece)
            } else {
                None
            };
            if let Some(neighbor) = neighbor {
                if neighbor == leaving.upper_piece {
                    return Ok(false);
                }
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    Ok(true)
}

/// Borrowed montage-edge data consumed by `solveForShifts`.  These are the
/// typed equivalents of the parallel arrays on C's `MidasView`.
pub struct MidasShiftSolveRequest<'a> {
    pub variable_pieces: &'a [i32],
    pub variable_for_piece: &'a [i32],
    pub x_piece_coordinates: &'a [i32],
    pub y_piece_coordinates: &'a [i32],
    pub edge_dx: &'a [f32],
    pub edge_dy: &'a [f32],
    pub piece_lower: &'a [i32],
    pub piece_upper: &'a [i32],
    pub skipped_edges: &'a [i32],
    pub edge_lower: &'a [i32],
    pub edge_upper: &'a [i32],
    pub leave_index: i32,
    pub exclude_skipped: bool,
    pub robust_criterion: f32,
}

/// Result from `solveForShifts`, including the source solver's per-variable
/// X/Y shifts, convergence counters, and returned mean edge weights.
#[derive(Clone, Debug, PartialEq)]
pub struct MidasShiftSolveResult {
    pub shifts: Vec<f32>,
    pub edge_weights: Vec<f32>,
    pub iterations: i32,
    pub weighted_error_mean: f32,
    pub weighted_error_max: f32,
}

/// `solve_for_shifts` (`transforms.cpp:2244`).  MIDAS uses the shared
/// `findPieceShifts` implementation with a fixed direction/layout and these
/// source-local convergence parameters.
pub fn solve_for_shifts(request: MidasShiftSolveRequest<'_>) -> Result<MidasShiftSolveResult, ()> {
    let count = i32::try_from(request.variable_pieces.len()).map_err(|_| ())?;
    if count < 2
        || request.variable_for_piece.len() != request.x_piece_coordinates.len()
        || request.x_piece_coordinates.len() != request.y_piece_coordinates.len()
    {
        return Err(());
    }
    let mut shifts = vec![0.0; request.variable_pieces.len() * 2];
    let mut weights = vec![0.0; request.variable_pieces.len() * 2];
    let mut iterations = 0;
    let mut weighted_error_mean = 0.0;
    let mut weighted_error_max = 0.0;
    let status = find_piece_shifts(
        request.variable_pieces,
        count,
        request.variable_for_piece,
        request.x_piece_coordinates,
        request.y_piece_coordinates,
        request.edge_dx,
        request.edge_dy,
        -1,
        request.piece_lower,
        request.piece_upper,
        request.skipped_edges,
        1,
        &mut shifts,
        1,
        request.edge_lower,
        request.edge_upper,
        1,
        &mut weights,
        0,
        request.leave_index,
        if request.exclude_skipped { 1 } else { 3 },
        request.robust_criterion,
        5.0e-4,
        5.0e-6,
        20 + count,
        7,
        50,
        &mut iterations,
        &mut weighted_error_mean,
        &mut weighted_error_max,
    );
    if status != 0 {
        return Err(());
    }
    Ok(MidasShiftSolveResult {
        shifts,
        edge_weights: weights,
        iterations,
        weighted_error_mean,
        weighted_error_max,
    })
}

/// `checklist` (`transforms.cpp:1651`).
pub fn checklist(piece_coordinates: &[i32], frame_size: i32) -> Option<MidasPieceChecklist> {
    let &minimum_piece = piece_coordinates.iter().min()?;
    let pitch = piece_coordinates
        .iter()
        .map(|&piece| piece - minimum_piece)
        .filter(|&difference| difference > 0)
        .min();
    let Some(pitch) = pitch else {
        return Some(MidasPieceChecklist {
            minimum_piece,
            pieces: 1,
            overlap: 0,
        });
    };
    if frame_size < pitch
        || piece_coordinates
            .iter()
            .any(|&piece| (piece - minimum_piece) % pitch != 0)
    {
        return Some(MidasPieceChecklist {
            minimum_piece,
            pieces: -1,
            overlap: 0,
        });
    }
    let pieces = piece_coordinates
        .iter()
        .map(|&piece| (piece - minimum_piece) / pitch + 1)
        .max()
        .unwrap_or(1);
    Some(MidasPieceChecklist {
        minimum_piece,
        pieces,
        overlap: frame_size - pitch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_matrix_matches_midas_layout() {
        let mut matrix = tramat_create();
        tramat_translate(&mut matrix, 2., -3.);
        assert_eq!(matrix[6..8], [2., -3.]);
        let inverse = tramat_inverse(&matrix).unwrap();
        assert_eq!(inverse[6..8], [-2., 3.]);
    }
    #[test]
    fn source_view_constructor_delegates_to_owned_midas_state() {
        let view = new_view();
        assert_eq!((view.xsize, view.ysize, view.zsize), (0, 0, 0));
    }
    #[test]
    fn translation_fills_mean() {
        let mut slice = MidasSlice {
            xsize: 2,
            ysize: 2,
            mean: 7.,
            data: vec![1, 2, 3, 4],
        };
        assert_eq!(translate_slice(&mut slice, 1, 0), 0);
        assert_eq!(slice.data, vec![7, 1, 7, 3]);
    }
    #[test]
    fn xform_slice_preserves_identity_and_marks_affine_output() {
        let mut view = new_view();
        view.xsize = 2;
        view.ysize = 2;
        view.tr = vec![MidasTransform {
            mat: tramat_create(),
            ..Default::default()
        }];
        let input = MidasSlice {
            xsize: 2,
            ysize: 2,
            mean: 9.,
            data: vec![1, 2, 3, 4],
        };
        assert_eq!(
            get_xform_slice(&view, &input, 0, false).unwrap(),
            (input.clone(), false)
        );
        view.tr[0].mat[6] = 1.;
        let (output, transformed) = get_xform_slice(&view, &input, 0, false).unwrap();
        assert!(transformed);
        assert_eq!(output.data, vec![1, 1, 3, 3]);
    }
    #[test]
    fn correlation_range_intersects_shift_border_and_box() {
        assert_eq!(xcorr_range(100, 4.6, 50, 3, 100), Some((8, 96)));
        assert_eq!(xcorr_range(100, -4.6, 50, 3, 100), Some((3, 99)));
        assert_eq!(xcorr_range(10, 0., 5, 6, 4), None);
    }
    #[test]
    fn montage_checklist_reports_regular_spacing_and_irregular_sentinel() {
        assert_eq!(
            checklist(&[100, 0, 50, 50], 80),
            Some(MidasPieceChecklist {
                minimum_piece: 0,
                pieces: 3,
                overlap: 30
            })
        );
        assert_eq!(
            checklist(&[0, 30, 50], 80),
            Some(MidasPieceChecklist {
                minimum_piece: 0,
                pieces: -1,
                overlap: 0
            })
        );
        assert_eq!(
            checklist(&[7, 7], 80),
            Some(MidasPieceChecklist {
                minimum_piece: 7,
                pieces: 1,
                overlap: 0
            })
        );
    }
    #[test]
    fn omitted_montage_edge_reports_alternate_paths_and_disconnections() {
        let edges = [
            MidasMontageEdge {
                lower_piece: 0,
                upper_piece: 1,
                skipped: false,
            },
            MidasMontageEdge {
                lower_piece: 1,
                upper_piece: 2,
                skipped: false,
            },
            MidasMontageEdge {
                lower_piece: 0,
                upper_piece: 2,
                skipped: false,
            },
        ];
        assert_eq!(pieces_not_connected(3, &edges, 0, false), Ok(false));
        assert_eq!(pieces_not_connected(3, &edges[..2], 0, false), Ok(true));
        let skipped = [
            edges[0],
            MidasMontageEdge {
                skipped: true,
                ..edges[2]
            },
        ];
        assert_eq!(pieces_not_connected(3, &skipped, 0, true), Ok(true));
        assert_eq!(lower_edge_if_included(Some(4), true, true), None);
        assert_eq!(upper_edge_if_included(Some(4), true, false), Some(4));
        let mut labels = [1, 0, 2];
        let mut pending = vec![0, 2];
        assert_eq!(check_path_list(&mut labels, &mut pending, 0, 1), Ok(false));
        assert_eq!(pending, [0, 2, 1]);
        assert_eq!(check_path_list(&mut labels, &mut pending, 1, 2), Ok(true));
    }
    #[test]
    fn midas_shift_adapter_invokes_shared_piece_solver() {
        let result = solve_for_shifts(MidasShiftSolveRequest {
            variable_pieces: &[0, 1],
            variable_for_piece: &[0, 1],
            x_piece_coordinates: &[0, 1],
            y_piece_coordinates: &[0, 0],
            edge_dx: &[4., 0.],
            edge_dy: &[0., 0.],
            piece_lower: &[0, 1],
            piece_upper: &[1, 0],
            skipped_edges: &[0, 0],
            edge_lower: &[-1, -1, 0, -1],
            edge_upper: &[0, -1, -1, -1],
            leave_index: -1,
            exclude_skipped: false,
            robust_criterion: 0.,
        })
        .unwrap();
        assert!(result.iterations > 0);
        assert!(result.shifts.iter().all(|value| value.is_finite()));
    }
}
