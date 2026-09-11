//! Translation of the non-Qt portions of `IMOD/midas/transforms.cpp`.
//!
//! The source uses a nine-element, column-major homogeneous transform.  This
//! unit deliberately retains that representation: it is the file format and
//! is also what the rest of MIDAS passes to the display code.

use super::midas::{MIDAS_DEBUG, MidasTransform, MidasView};
use crate::imod::libcfshr::linearxforms::{xf_invert, xf_mult};
use std::sync::atomic::Ordering;

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
}
