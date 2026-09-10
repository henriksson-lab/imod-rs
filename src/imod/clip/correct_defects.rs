//! Scaffold for `IMOD/clip/CorrectDefects.cpp`.
#![allow(dead_code)]

/// C++ `CameraDefects` from `include/CorrectDefects.h`, in declaration order.
pub struct CameraDefects {
    pub was_scaled: i32,
    pub rotation_flip: i32,
    pub k2_type: i32,
    pub falcon_type: i32,
    pub usable_top: i32,
    pub usable_left: i32,
    pub usable_bottom: i32,
    pub usable_right: i32,
    pub num_avg_super_res: i32,
    pub bad_column_start: Vec<u16>,
    pub bad_column_width: Vec<i16>,
    pub partial_bad_col: Vec<u16>,
    pub partial_bad_width: Vec<i16>,
    pub partial_bad_start_y: Vec<u16>,
    pub partial_bad_end_y: Vec<u16>,
    pub bad_row_start: Vec<u16>,
    pub bad_row_height: Vec<i16>,
    pub partial_bad_row: Vec<u16>,
    pub partial_bad_height: Vec<i16>,
    pub partial_bad_start_x: Vec<u16>,
    pub partial_bad_end_x: Vec<u16>,
    pub bad_pixel_x: Vec<u16>,
    pub bad_pixel_y: Vec<u16>,
    pub pix_use_mean: Vec<i8>,
}

/// C++ `CorDefCorrectDefects` (`CorrectDefects.cpp:78`).
pub unsafe fn cor_def_correct_defects(
    defects: *const crate::imod::clip::clip::CameraDefects,
    array: *mut core::ffi::c_void,
    data_type: i32,
    binning: i32,
    top: i32,
    left: i32,
    bottom: i32,
    right: i32,
) {
    let size_x = right - left;
    let size_y = bottom - top;
    let sum_x = ((size_x + 9) / 10).min(50);
    let sum_y = ((size_y + 9) / 10).min(50);
    let defects_ref = unsafe { &*defects };
    let super_fac = if defects_ref.falcon_type != 0 && defects_ref.was_scaled == 1 {
        2
    } else if defects_ref.falcon_type != 0 && defects_ref.was_scaled == 2 {
        4
    } else {
        0
    };
    if !defects_ref.pix_use_mean.is_empty() {
        let mut mean = 0.;
        let mut sd = 0.;
        unsafe { cor_def_sample_mean_sd_1(array, data_type, size_x, size_y, &mut mean, &mut sd) };
        unsafe {
            correct_pixels_3_ways(
                defects, array, data_type, size_x, size_y, binning, top, left, 1, mean,
            )
        };
    }
    let mut num_bad = (defects_ref.usable_top - 1) / binning + 1 - top;
    if defects_ref.usable_top > 0 && num_bad > 0 {
        unsafe {
            correct_edge(
                array,
                data_type,
                num_bad,
                5,
                size_x,
                sum_x,
                num_bad * size_x,
                1,
                -size_x,
            )
        };
    }
    let mut first_bad = (defects_ref.usable_bottom + 1) / binning - top;
    num_bad = size_y - first_bad;
    if defects_ref.usable_bottom > 0 && num_bad > 0 {
        unsafe {
            correct_edge(
                array,
                data_type,
                num_bad,
                5,
                size_x,
                sum_x,
                (first_bad - 1) * size_x,
                1,
                size_x,
            )
        };
    }
    num_bad = (defects_ref.usable_left - 1) / binning + 1 - left;
    if defects_ref.usable_left > 0 && num_bad > 0 {
        unsafe {
            correct_edge(
                array, data_type, num_bad, 5, size_y, sum_y, num_bad, size_x, -1,
            )
        };
    }
    first_bad = (defects_ref.usable_right + 1) / binning - left;
    num_bad = size_x - first_bad;
    if defects_ref.usable_right > 0 && num_bad > 0 {
        unsafe {
            correct_edge(
                array,
                data_type,
                num_bad,
                5,
                size_y,
                sum_y,
                first_bad - 1,
                size_x,
                1,
            )
        };
    }
    for i in 0..defects_ref.bad_column_start.len() {
        let start = defects_ref.bad_column_start[i] as i32 / binning;
        let end = (defects_ref.bad_column_start[i] as i32 + defects_ref.bad_column_width[i] as i32
            - 1)
            / binning;
        unsafe {
            correct_column(
                array,
                data_type,
                size_x,
                size_y,
                1,
                size_x,
                start - left,
                end + 1 - start,
                0,
                size_y - 1,
                super_fac,
                defects_ref.num_avg_super_res,
            )
        };
    }
    for i in 0..defects_ref.partial_bad_col.len() {
        let start = defects_ref.partial_bad_col[i] as i32 / binning;
        let end = (defects_ref.partial_bad_col[i] as i32 + defects_ref.partial_bad_width[i] as i32
            - 1)
            / binning;
        let ys = defects_ref.partial_bad_start_y[i] as i32 / binning - top;
        let ye = defects_ref.partial_bad_end_y[i] as i32 / binning - top;
        if ys < size_y && ye >= 0 && ys <= ye {
            unsafe {
                correct_column(
                    array,
                    data_type,
                    size_x,
                    size_y,
                    1,
                    size_x,
                    start - left,
                    end + 1 - start,
                    ys.max(0),
                    ye.min(size_y - 1),
                    super_fac,
                    defects_ref.num_avg_super_res,
                )
            };
        }
    }
    for i in 0..defects_ref.bad_row_start.len() {
        let start = defects_ref.bad_row_start[i] as i32 / binning;
        let end = (defects_ref.bad_row_start[i] as i32 + defects_ref.bad_row_height[i] as i32 - 1)
            / binning;
        unsafe {
            correct_column(
                array,
                data_type,
                size_y,
                size_x,
                size_x,
                1,
                start - top,
                end + 1 - start,
                0,
                size_x - 1,
                super_fac,
                defects_ref.num_avg_super_res,
            )
        };
    }
    for i in 0..defects_ref.partial_bad_row.len() {
        let start = defects_ref.partial_bad_row[i] as i32 / binning;
        let end =
            (defects_ref.partial_bad_row[i] as i32 + defects_ref.partial_bad_height[i] as i32 - 1)
                / binning;
        let ys = defects_ref.partial_bad_start_x[i] as i32 / binning - left;
        let ye = defects_ref.partial_bad_end_x[i] as i32 / binning - left;
        if ys < size_x && ye >= 0 && ys <= ye {
            unsafe {
                correct_column(
                    array,
                    data_type,
                    size_y,
                    size_x,
                    size_x,
                    1,
                    start - top,
                    end + 1 - start,
                    ys.max(0),
                    ye.min(size_x - 1),
                    super_fac,
                    defects_ref.num_avg_super_res,
                )
            };
        }
    }
    unsafe {
        correct_pixels_3_ways(
            defects, array, data_type, size_x, size_y, binning, top, left, 0, 0.,
        )
    };
}
/// C++ `CorrectEdge` (`CorrectDefects.cpp:190`).
pub unsafe fn correct_edge(
    array: *mut core::ffi::c_void,
    data_type: i32,
    num_bad: i32,
    taper: i32,
    length: i32,
    mut sum_length: i32,
    ind_start: i32,
    step_along: i32,
    step_between: i32,
) {
    unsafe {
        if sum_length > length {
            sum_length = length;
        }
        let add_start = sum_length / 2;
        let add_end = length - (sum_length - sum_length / 2);
        for row in 0..num_bad {
            let mut sum = 0_f64;
            match data_type {
                0 => {
                    for i in 0..sum_length {
                        sum += *array
                            .cast::<u8>()
                            .offset((ind_start + i * step_along) as isize)
                            as f64
                    }
                }
                1 => {
                    for i in 0..sum_length {
                        sum += *array
                            .cast::<i16>()
                            .offset((ind_start + i * step_along) as isize)
                            as f64
                    }
                }
                6 => {
                    for i in 0..sum_length {
                        sum += *array
                            .cast::<u16>()
                            .offset((ind_start + i * step_along) as isize)
                            as f64
                    }
                }
                2 => {
                    for i in 0..sum_length {
                        sum += *array
                            .cast::<f32>()
                            .offset((ind_start + i * step_along) as isize)
                            as f64
                    }
                }
                _ => return,
            }
            let mut ind_drop = ind_start;
            let mut ind_add = ind_start + sum_length * step_along;
            let mut ind_dest = ind_start + (row + 1) * step_between;
            let mut fraction = 1_f32;
            let mut sum_factor = 0_f64;
            if taper > 0 {
                fraction = ((taper as f32 - row as f32 - 1.) / taper as f32).max(0.);
                sum_factor = (1. - fraction as f64) / sum_length as f64;
            }
            for i in 0..length {
                if i >= add_start && i < add_end {
                    match data_type {
                        0 => {
                            sum += *array.cast::<u8>().offset(ind_add as isize) as f64
                                - *array.cast::<u8>().offset(ind_drop as isize) as f64
                        }
                        1 => {
                            sum += *array.cast::<i16>().offset(ind_add as isize) as f64
                                - *array.cast::<i16>().offset(ind_drop as isize) as f64
                        }
                        6 => {
                            sum += *array.cast::<u16>().offset(ind_add as isize) as f64
                                - *array.cast::<u16>().offset(ind_drop as isize) as f64
                        }
                        2 => {
                            sum += *array.cast::<f32>().offset(ind_add as isize) as f64
                                - *array.cast::<f32>().offset(ind_drop as isize) as f64
                        }
                        _ => unreachable!(),
                    }
                    ind_add += step_along;
                    ind_drop += step_along;
                }
                let value = fraction
                    * match data_type {
                        0 => *array
                            .cast::<u8>()
                            .offset((ind_start + i * step_along) as isize)
                            as f32,
                        1 => *array
                            .cast::<i16>()
                            .offset((ind_start + i * step_along) as isize)
                            as f32,
                        6 => *array
                            .cast::<u16>()
                            .offset((ind_start + i * step_along) as isize)
                            as f32,
                        2 => *array
                            .cast::<f32>()
                            .offset((ind_start + i * step_along) as isize),
                        _ => unreachable!(),
                    }
                    + (sum_factor * sum) as f32;
                match data_type {
                    0 => {
                        *array.cast::<u8>().offset(ind_dest as isize) =
                            random_int_fill_from_float(value) as u8
                    }
                    1 => {
                        *array.cast::<i16>().offset(ind_dest as isize) =
                            random_int_fill_from_float(value) as i16
                    }
                    6 => {
                        *array.cast::<u16>().offset(ind_dest as isize) =
                            random_int_fill_from_float(value) as u16
                    }
                    2 => *array.cast::<f32>().offset(ind_dest as isize) = value,
                    _ => unreachable!(),
                }
                ind_dest += step_along;
            }
        }
    }
}
/// C++ `CorrectColumn` (`CorrectDefects.cpp:429`).
unsafe fn correct_column(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nx: i32,
    ny: i32,
    x_stride: i32,
    y_stride: i32,
    mut ind_start: i32,
    mut num: i32,
    y_start: i32,
    y_end: i32,
    super_fac: i32,
    num_avg_super: i32,
) {
    static mut PSEUDO: i32 = 456_789;
    if ind_start < 0 {
        num += ind_start;
        ind_start = 0;
    }
    if ind_start + num > nx {
        num = nx - ind_start;
    }
    if num <= 0 {
        return;
    }
    macro_rules! run_column {
        ($ty:ty, $integer:expr) => {{
            let data = array.cast::<$ty>();
            if super_fac > 0 {
                let mut sides = Vec::with_capacity((2 * num_avg_super) as usize);
                for i in 0..num_avg_super {
                    let il = ind_start - (i + 1) * super_fac;
                    if il >= 0 {
                        sides.push(il);
                    }
                    let ir = ind_start + num + i * super_fac;
                    if ir < nx {
                        sides.push(ir);
                    }
                }
                for side in sides {
                    for iy in y_start..=y_end {
                        let ind = side * x_stride + iy * y_stride;
                        let count = if super_fac == 2 { 2 } else { 4 };
                        let mut sum: i32 = 0;
                        let mut fsum: f32 = 0.;
                        for j in 0..count {
                            if $integer {
                                sum += *data.offset((ind + j * x_stride) as isize) as i32;
                            } else {
                                fsum += *data.offset((ind + j * x_stride) as isize) as f32;
                            }
                        }
                        if $integer {
                            let mean = sum / count;
                            for j in 0..count {
                                *data.offset((ind + j * x_stride) as isize) = mean as $ty;
                            }
                            let remainder = sum % count;
                            for _ in 0..remainder {
                                let next = unsafe {
                                    PSEUDO = (197 * (PSEUDO + 1)) & 0x000f_ffff;
                                    PSEUDO
                                };
                                let which = (next >> 2) & (count - 1);
                                let ptr = data.offset((ind + which * x_stride) as isize);
                                *ptr = (*ptr as i32 + 1) as $ty;
                            }
                        } else {
                            let mean = fsum / count as f32;
                            for j in 0..count {
                                *data.offset((ind + j * x_stride) as isize) = mean as $ty;
                            }
                        }
                    }
                }
            }
            for col in 0..num {
                let f_right = (col + 1) as f32 / (num + 1) as f32;
                let f_left = 1. - f_right;
                let mut left = ind_start - 1;
                let mut right = ind_start + num;
                if left < 0 {
                    left = right;
                }
                if right >= nx {
                    right = left;
                }
                let five_plus_ok = left > 14 && right < nx - 15;
                let mut ind = (ind_start + col) * x_stride + y_stride * y_start;
                let mut il = left * x_stride + y_stride * y_start;
                let mut ir = right * x_stride + y_stride * y_start;
                let full_start = if y_start < 7 { 7 } else { y_start };
                let full_end = if y_end >= ny - 7 { ny - 8 } else { y_end };
                for y in y_start..=y_end {
                    let fill: f32;
                    if num >= 3 && y_end - y_start >= 15 && five_plus_ok {
                        if y < full_start {
                            fill = (f_left
                                * (*data.offset(il as isize) as f32
                                    + *data.offset((il + y_stride) as isize) as f32
                                    + *data.offset((il - x_stride) as isize) as f32
                                    + *data.offset((il + y_stride - x_stride) as isize) as f32)
                                + f_right
                                    * (*data.offset(ir as isize) as f32
                                        + *data.offset((ir + y_stride) as isize) as f32
                                        + *data.offset((ir + x_stride) as isize) as f32
                                        + *data.offset((ir + y_stride + x_stride) as isize)
                                            as f32))
                                / 4.;
                        } else if y <= full_end {
                            let next = unsafe {
                                PSEUDO = (197 * (PSEUDO + 1)) & 0x000f_ffff;
                                PSEUDO
                            };
                            let iy = (next >> 2) % 15;
                            let ix = (next >> 6) & 15;
                            fill = if next & 2048 != 0 {
                                *data.offset((il + (iy - 7) * y_stride - ix * x_stride) as isize)
                                    as f32
                            } else {
                                *data.offset((ir + (iy - 7) * y_stride + ix * x_stride) as isize)
                                    as f32
                            };
                        } else {
                            fill = (f_left
                                * (*data.offset((il - y_stride) as isize) as f32
                                    + *data.offset(il as isize) as f32
                                    + *data.offset((il - x_stride - y_stride) as isize) as f32
                                    + *data.offset((il - x_stride) as isize) as f32)
                                + f_right
                                    * (*data.offset((ir - y_stride) as isize) as f32
                                        + *data.offset(ir as isize) as f32
                                        + *data.offset((ir + x_stride - y_stride) as isize)
                                            as f32
                                        + *data.offset((ir + x_stride) as isize) as f32))
                                / 4.;
                        }
                    } else if num >= 3 && y_end - y_start >= 1 {
                        if y < full_start {
                            fill = (f_left
                                * (*data.offset(il as isize) as f32
                                    + *data.offset((il + y_stride) as isize) as f32)
                                + f_right
                                    * (*data.offset(ir as isize) as f32
                                        + *data.offset((ir + y_stride) as isize) as f32))
                                / 2.;
                        } else if y <= full_end {
                            fill = (f_left
                                * (*data.offset((il - y_stride) as isize) as f32
                                    + *data.offset(il as isize) as f32
                                    + *data.offset((il + y_stride) as isize) as f32)
                                + f_right
                                    * (*data.offset((ir - y_stride) as isize) as f32
                                        + *data.offset(ir as isize) as f32
                                        + *data.offset((ir + y_stride) as isize) as f32))
                                / 3.;
                        } else {
                            fill = (f_left
                                * (*data.offset((il - y_stride) as isize) as f32
                                    + *data.offset(il as isize) as f32)
                                + f_right
                                    * (*data.offset((ir - y_stride) as isize) as f32
                                        + *data.offset(ir as isize) as f32))
                                / 2.;
                        }
                    } else {
                        fill = f_left * *data.offset(il as isize) as f32
                            + f_right * *data.offset(ir as isize) as f32;
                    }
                    *data.offset(ind as isize) = if $integer {
                        random_int_fill_from_float(fill) as $ty
                    } else {
                        fill as $ty
                    };
                    ind += y_stride;
                    il += y_stride;
                    ir += y_stride;
                }
            }
        }};
    }
    match data_type {
        0 => run_column!(u8, true),
        1 => run_column!(i16, true),
        6 => run_column!(u16, true),
        2 => run_column!(f32, false),
        _ => {}
    }
}
/// Matches C++ `RandomIntFillFromIntSum`.
pub fn random_int_fill_from_int_sum(integer_sum: i32, number_summed: i32) -> i32 {
    static mut PSEUDO: i32 = 482_945;
    let pseudo = unsafe {
        PSEUDO = (197 * (PSEUDO + 1)) & 0x000f_ffff;
        PSEUDO
    };
    let random_integer = (pseudo >> 2) % number_summed;
    let mut result = integer_sum / number_summed;
    if integer_sum % number_summed > random_integer {
        result += 1;
    }
    result
}
/// Matches C++ `RandomIntFillFromFloat`.
pub fn random_int_fill_from_float(value: f32) -> i32 {
    static mut PSEUDO: i32 = 843_295;
    let pseudo = unsafe {
        PSEUDO = (197 * (PSEUDO + 1)) & 0x000f_ffff;
        PSEUDO
    };
    let mut result = value as i32;
    if (value - result as f32) * 0x3ffff as f32 > (pseudo & 0x3ffff) as f32 {
        result += 1;
    }
    result
}
/// C++ `CorrectPixel` (`CorrectDefects.cpp:581`).
pub unsafe fn correct_pixel(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    xpix: i32,
    ypix: i32,
    use_mean: i32,
    mean: f32,
) {
    unsafe {
        let index = xpix + ypix * nxdim;
        if use_mean != 0 {
            match data_type {
                0 => {
                    *array.cast::<u8>().offset(index as isize) =
                        random_int_fill_from_float(mean) as u8
                }
                1 => {
                    *array.cast::<i16>().offset(index as isize) =
                        random_int_fill_from_float(mean) as i16
                }
                6 => {
                    *array.cast::<u16>().offset(index as isize) =
                        random_int_fill_from_float(mean) as u16
                }
                2 => *array.cast::<f32>().offset(index as isize) = mean,
                _ => {}
            }
            return;
        }
        if xpix > 0 && xpix < nx - 1 && ypix > 0 && ypix < ny - 1 {
            match data_type {
                0 => {
                    let data = array.cast::<u8>();
                    *data.offset(index as isize) = random_int_fill_from_int_sum(
                        *data.offset((index - 1) as isize) as i32
                            + *data.offset((index + 1) as isize) as i32
                            + *data.offset((index - nxdim) as isize) as i32
                            + *data.offset((index + nxdim) as isize) as i32,
                        4,
                    ) as u8;
                }
                1 => {
                    let data = array.cast::<i16>();
                    *data.offset(index as isize) = random_int_fill_from_int_sum(
                        *data.offset((index - 1) as isize) as i32
                            + *data.offset((index + 1) as isize) as i32
                            + *data.offset((index - nxdim) as isize) as i32
                            + *data.offset((index + nxdim) as isize) as i32,
                        4,
                    ) as i16;
                }
                6 => {
                    let data = array.cast::<u16>();
                    *data.offset(index as isize) = random_int_fill_from_int_sum(
                        *data.offset((index - 1) as isize) as i32
                            + *data.offset((index + 1) as isize) as i32
                            + *data.offset((index - nxdim) as isize) as i32
                            + *data.offset((index + nxdim) as isize) as i32,
                        4,
                    ) as u16;
                }
                2 => {
                    let data = array.cast::<f32>();
                    *data.offset(index as isize) = (*data.offset((index - 1) as isize)
                        + *data.offset((index + 1) as isize)
                        + *data.offset((index - nxdim) as isize)
                        + *data.offset((index + nxdim) as isize))
                        / 4.;
                }
                _ => {}
            }
            return;
        }
        let mut integer_sum = 0_i32;
        let mut float_sum = 0_f32;
        let mut count = 0_i32;
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let x = xpix + dx;
            let y = ypix + dy;
            if x >= 0 && x < nx && y >= 0 && y < ny {
                count += 1;
                let ind = x + y * nxdim;
                match data_type {
                    1 => integer_sum += *array.cast::<i16>().offset(ind as isize) as i32,
                    6 => integer_sum += *array.cast::<u16>().offset(ind as isize) as i32,
                    2 => float_sum += *array.cast::<f32>().offset(ind as isize),
                    _ => {}
                }
            }
        }
        match data_type {
            0 => {
                *array.cast::<u8>().offset(index as isize) =
                    random_int_fill_from_int_sum(integer_sum, count) as u8
            }
            1 => {
                *array.cast::<i16>().offset(index as isize) =
                    random_int_fill_from_int_sum(integer_sum, count) as i16
            }
            6 => {
                *array.cast::<u16>().offset(index as isize) =
                    random_int_fill_from_int_sum(integer_sum, count) as u16
            }
            2 => *array.cast::<f32>().offset(index as isize) = float_sum / count as f32,
            _ => {}
        }
    }
}

/// C++ `CorrectSuperPixel` (`CorrectDefects.cpp:651`).
pub unsafe fn correct_super_pixel(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    xpix: i32,
    ypix: i32,
    use_mean: i32,
    mean: f32,
) {
    unsafe {
        let index = xpix + ypix * nxdim;
        let destinations = [index, index + 1, index + nxdim, index + nxdim + 1];
        if use_mean != 0 {
            match data_type {
                0 => {
                    for ind in destinations {
                        *array.cast::<u8>().offset(ind as isize) =
                            random_int_fill_from_float(mean) as u8
                    }
                }
                1 => {
                    for ind in destinations {
                        *array.cast::<i16>().offset(ind as isize) =
                            random_int_fill_from_float(mean) as i16
                    }
                }
                6 => {
                    for ind in destinations {
                        *array.cast::<u16>().offset(ind as isize) =
                            random_int_fill_from_float(mean) as u16
                    }
                }
                2 => {
                    for ind in destinations {
                        *array.cast::<f32>().offset(ind as isize) = mean
                    }
                }
                _ => {}
            }
            return;
        }
        let deltas = [
            (-2, 0),
            (-1, 0),
            (2, 0),
            (3, 0),
            (-2, 1),
            (-1, 1),
            (2, 1),
            (3, 1),
            (0, -2),
            (1, -2),
            (0, -1),
            (1, -1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
        ];
        let interior = xpix > 1 && xpix < nx - 3 && ypix > 1 && ypix < ny - 3;
        let mut integer_sum = 0_i32;
        let mut float_sum = 0_f32;
        let mut count = 0_i32;
        for (dx, dy) in deltas {
            let x = xpix + dx;
            let y = ypix + dy;
            if interior || (x >= 0 && x < nx && y >= 0 && y < ny) {
                let ind = x + y * nxdim;
                match data_type {
                    0 => integer_sum += *array.cast::<u8>().offset(ind as isize) as i32,
                    1 => integer_sum += *array.cast::<i16>().offset(ind as isize) as i32,
                    6 => integer_sum += *array.cast::<u16>().offset(ind as isize) as i32,
                    2 => float_sum += *array.cast::<f32>().offset(ind as isize),
                    _ => {}
                }
                count += 1;
            }
        }
        match data_type {
            0 => {
                let value = random_int_fill_from_int_sum(integer_sum, count) as u8;
                for ind in destinations {
                    *array.cast::<u8>().offset(ind as isize) = value
                }
            }
            1 => {
                let value = random_int_fill_from_int_sum(integer_sum, count) as i16;
                for ind in destinations {
                    *array.cast::<i16>().offset(ind as isize) = value
                }
            }
            6 => {
                let value = random_int_fill_from_int_sum(integer_sum, count) as u16;
                for ind in destinations {
                    *array.cast::<u16>().offset(ind as isize) = value
                }
            }
            2 => {
                let value = float_sum / count as f32;
                for ind in destinations {
                    *array.cast::<f32>().offset(ind as isize) = value
                }
            }
            _ => {}
        }
    }
}

/// C++ `CorrectJumboPixel` (`CorrectDefects.cpp:755`).
pub unsafe fn correct_jumbo_pixel(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    xpix: i32,
    ypix: i32,
    use_mean: i32,
    mean: f32,
) {
    unsafe {
        let index = xpix + ypix * nxdim;
        if use_mean != 0 {
            for y in 0..4 {
                for x in 0..4 {
                    let ind = index + x + y * nxdim;
                    match data_type {
                        0 => {
                            *array.cast::<u8>().offset(ind as isize) =
                                random_int_fill_from_float(mean) as u8
                        }
                        1 => {
                            *array.cast::<i16>().offset(ind as isize) =
                                random_int_fill_from_float(mean) as i16
                        }
                        6 => {
                            *array.cast::<u16>().offset(ind as isize) =
                                random_int_fill_from_float(mean) as u16
                        }
                        2 => *array.cast::<f32>().offset(ind as isize) = mean,
                        _ => {}
                    }
                }
            }
            return;
        }
        let mut deltas = [(0_i32, 0_i32); 64];
        let mut at = 0;
        for (base_x, base_y) in [(-4, 0), (4, 0), (0, -4), (0, 4)] {
            for y in 0..4 {
                for x in 0..4 {
                    deltas[at] = (base_x + x, base_y + y);
                    at += 1;
                }
            }
        }
        let interior = xpix > 5 && xpix < nx - 7 && ypix > 5 && ypix < ny - 7;
        let mut integer_sum = 0_i32;
        let mut float_sum = 0_f32;
        let mut count = 0_i32;
        for (dx, dy) in deltas {
            let x = xpix + dx;
            let y = ypix + dy;
            if interior || (x >= 0 && x < nx && y >= 0 && y < ny) {
                let ind = x + y * nxdim;
                match data_type {
                    0 => integer_sum += *array.cast::<u8>().offset(ind as isize) as i32,
                    1 => integer_sum += *array.cast::<i16>().offset(ind as isize) as i32,
                    6 => integer_sum += *array.cast::<u16>().offset(ind as isize) as i32,
                    2 => float_sum += *array.cast::<f32>().offset(ind as isize),
                    _ => {}
                };
                count += 1;
            }
        }
        for y in 0..4 {
            for x in 0..4 {
                let ind = index + x + y * nxdim;
                match data_type {
                    0 => {
                        *array.cast::<u8>().offset(ind as isize) =
                            random_int_fill_from_int_sum(integer_sum, count) as u8
                    }
                    1 => {
                        *array.cast::<i16>().offset(ind as isize) =
                            random_int_fill_from_int_sum(integer_sum, count) as i16
                    }
                    6 => {
                        *array.cast::<u16>().offset(ind as isize) =
                            random_int_fill_from_int_sum(integer_sum, count) as u16
                    }
                    2 => *array.cast::<f32>().offset(ind as isize) = float_sum / count as f32,
                    _ => {}
                }
            }
        }
    }
}
/// C++ `CorrectPixels3Ways` (`CorrectDefects.cpp:843`).
pub unsafe fn correct_pixels_3_ways(
    defects: *const crate::imod::clip::clip::CameraDefects,
    array: *mut core::ffi::c_void,
    data_type: i32,
    size_x: i32,
    size_y: i32,
    binning: i32,
    top: i32,
    left: i32,
    use_mean: i32,
    mean: f32,
) {
    unsafe {
        let defects = &*defects;
        for i in 0..defects.bad_pixel_x.len() {
            if (i < defects.pix_use_mean.len() && defects.pix_use_mean[i] == use_mean as i8)
                || (i >= defects.pix_use_mean.len() && use_mean == 0)
            {
                let x = defects.bad_pixel_x[i] as i32 / binning - left;
                let y = defects.bad_pixel_y[i] as i32 / binning - top;
                if x >= 0 && y >= 0 && x < size_x && y < size_y {
                    if defects.was_scaled > 1 {
                        correct_jumbo_pixel(
                            array, data_type, size_x, size_x, size_y, x, y, use_mean, mean,
                        );
                    }
                    if defects.was_scaled > 0 && binning == 1 {
                        correct_super_pixel(
                            array, data_type, size_x, size_x, size_y, x, y, use_mean, mean,
                        );
                    } else {
                        correct_pixel(
                            array, data_type, size_x, size_x, size_y, x, y, use_mean, mean,
                        );
                    }
                }
            }
        }
    }
}
/// Matches C++ `CorDefSurroundingMean`.
pub unsafe fn cor_def_surrounding_mean(
    frame: *const core::ffi::c_void,
    pixel_type: i32,
    nx: i32,
    ny: i32,
    truncation_limit: f32,
    ix: i32,
    iy: i32,
) -> f32 {
    unsafe {
        let ix_start = (ix - 3).max(0);
        let ix_end = (ix + 3).min(nx - 1);
        let iy_start = (iy - 3).max(0);
        let iy_end = (iy + 3).min(ny - 1);
        let mut number_pixels = 0;
        let mut sum = 0.;
        for ay in iy_start..=iy_end {
            for ax in ix_start..=ix_end {
                if (ax - ix).abs() > 1 || (ay - iy).abs() > 1 {
                    let value = match pixel_type {
                        0 => *frame.cast::<u8>().add((ax + ay * nx) as usize) as f32,
                        1 => *frame.cast::<i16>().add((ax + ay * nx) as usize) as f32,
                        6 => *frame.cast::<u16>().add((ax + ay * nx) as usize) as f32,
                        2 => *frame.cast::<f32>().add((ax + ay * nx) as usize),
                        _ => 0.,
                    };
                    if value <= truncation_limit {
                        sum += value;
                        number_pixels += 1;
                    }
                }
            }
        }
        sum / number_pixels.max(1) as f32
    }
}
/// C++ `CorDefScaleDefectsForK2` (`CorrectDefects.cpp:933`).
pub fn cor_def_scale_defects_for_k2(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    scale_down: bool,
) {
    let up_factor = if scale_down { 1 } else { 2 };
    let down_factor = if scale_down { 2 } else { 1 };
    let add_factor = if scale_down { 0 } else { 1 };
    defects.was_scaled = if scale_down { -1 } else { 1 };
    scale_defects_by_factors(defects, up_factor, down_factor, add_factor);
}
/// Matches C++ `CorDefScaleDefectsForFalcon`.
pub fn cor_def_scale_defects_for_falcon(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    factor: i32,
) {
    defects.was_scaled = factor / 2;
    if factor > 0 {
        scale_defects_by_factors(defects, factor, 1, factor - 1);
    } else {
        scale_defects_by_factors(defects, 1, -factor, 0);
    }
}
/// Matches C++ `ScaleDefectsByFactors`.
pub fn scale_defects_by_factors(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    up_factor: i32,
    down_factor: i32,
    add_factor: i32,
) {
    defects.usable_top = defects.usable_top * up_factor / down_factor;
    defects.usable_left = defects.usable_left * up_factor / down_factor;
    defects.usable_bottom = defects.usable_bottom * up_factor / down_factor;
    if defects.usable_bottom != 0 {
        defects.usable_bottom += add_factor;
    }
    defects.usable_right = defects.usable_right * up_factor / down_factor;
    if defects.usable_right != 0 {
        defects.usable_right += add_factor;
    }
    scale_rows_or_columns(
        &mut defects.bad_column_start,
        &mut defects.bad_column_width,
        &mut defects.partial_bad_col,
        &mut defects.partial_bad_width,
        &mut defects.partial_bad_start_y,
        &mut defects.partial_bad_end_y,
        up_factor,
        down_factor,
        add_factor,
    );
    scale_rows_or_columns(
        &mut defects.bad_row_start,
        &mut defects.bad_row_height,
        &mut defects.partial_bad_row,
        &mut defects.partial_bad_height,
        &mut defects.partial_bad_start_x,
        &mut defects.partial_bad_end_x,
        up_factor,
        down_factor,
        add_factor,
    );
    for index in 0..defects.bad_pixel_x.len() {
        defects.bad_pixel_x[index] =
            (defects.bad_pixel_x[index] as i32 * up_factor / down_factor) as u16;
        defects.bad_pixel_y[index] =
            (defects.bad_pixel_y[index] as i32 * up_factor / down_factor) as u16;
    }
}
/// Matches C++ `ScaleRowsOrColumns`.
pub fn scale_rows_or_columns(
    column_start: &mut [u16],
    column_width: &mut [i16],
    partial: &mut [u16],
    partial_width: &mut [i16],
    start_y: &mut [u16],
    end_y: &mut [u16],
    up_factor: i32,
    down_factor: i32,
    add_factor: i32,
) {
    for index in 0..column_start.len() {
        column_start[index] = (column_start[index] as i32 * up_factor / down_factor) as u16;
        column_width[index] = (column_width[index] as i32 * up_factor / down_factor).max(1) as i16;
    }
    for index in 0..partial.len() {
        partial[index] = (partial[index] as i32 * up_factor / down_factor) as u16;
        partial_width[index] =
            (partial_width[index] as i32 * up_factor / down_factor).max(1) as i16;
        start_y[index] = (start_y[index] as i32 * up_factor / down_factor) as u16;
        end_y[index] = (end_y[index] as i32 * up_factor / down_factor + add_factor) as u16;
    }
}
/// Matches C++ `CorDefFlipDefectsInY`.
pub fn cor_def_flip_defects_in_y(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    _cam_size_x: i32,
    cam_size_y: i32,
    mut was_scaled: i32,
) {
    let y_flip = cam_size_y - 1;
    if was_scaled == 0 {
        was_scaled = defects.was_scaled;
    }
    let mut pixel_flip = y_flip;
    if was_scaled > 0 {
        pixel_flip = cam_size_y - if was_scaled > 1 { 4 } else { 2 };
    }
    let temporary = if defects.usable_bottom != 0 {
        y_flip - defects.usable_bottom
    } else {
        0
    };
    defects.usable_bottom = if defects.usable_top != 0 {
        y_flip - defects.usable_top
    } else {
        0
    };
    defects.usable_top = temporary;
    for index in 0..defects.bad_row_start.len() {
        defects.bad_row_start[index] = (y_flip
            - (defects.bad_row_start[index] as i32 + defects.bad_row_height[index] as i32 - 1))
            as u16;
    }
    for index in 0..defects.partial_bad_row.len() {
        defects.partial_bad_row[index] = (y_flip
            - (defects.partial_bad_row[index] as i32 + defects.partial_bad_height[index] as i32
                - 1)) as u16;
    }
    for index in 0..defects.partial_bad_col.len() {
        let temporary = y_flip - defects.partial_bad_end_y[index] as i32;
        defects.partial_bad_end_y[index] =
            (y_flip - defects.partial_bad_start_y[index] as i32) as u16;
        defects.partial_bad_start_y[index] = temporary as u16;
    }
    for index in 0..defects.bad_pixel_y.len() {
        defects.bad_pixel_y[index] = (pixel_flip - defects.bad_pixel_y[index] as i32) as u16;
    }
}
/// Matches C++ `CorDefRotateFlipDefects`.
pub fn cor_def_rotate_flip_defects(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    rotation_flip: i32,
    camera_size_x: i32,
    camera_size_y: i32,
) {
    let mut operation = rotation_flip;
    let pixel_add = if defects.was_scaled > 0 {
        if defects.was_scaled > 1 { 3 } else { 1 }
    } else {
        0
    };
    if defects.rotation_flip == rotation_flip {
        return;
    }
    if operation == 0 {
        operation = defects.rotation_flip;
        if operation == 1 || operation == 3 {
            operation = 4 - operation;
        }
    }
    for index in 0..defects.bad_pixel_x.len() {
        let (mut x1, mut y1) = (
            defects.bad_pixel_x[index] as i32,
            defects.bad_pixel_y[index] as i32,
        );
        let (mut x2, mut y2) = (x1 + pixel_add, y1 + pixel_add);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x1, &mut y1);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x2, &mut y2);
        defects.bad_pixel_x[index] = x1.min(x2) as u16;
        defects.bad_pixel_y[index] = y1.min(y2) as u16;
    }
    let mut bad_column_start = Vec::new();
    let mut bad_column_width = Vec::new();
    let mut bad_row_start = Vec::new();
    let mut bad_row_height = Vec::new();
    let mut partial_bad_col = Vec::new();
    let mut partial_bad_width = Vec::new();
    let mut partial_bad_start_y = Vec::new();
    let mut partial_bad_end_y = Vec::new();
    let mut partial_bad_row = Vec::new();
    let mut partial_bad_height = Vec::new();
    let mut partial_bad_start_x = Vec::new();
    let mut partial_bad_end_x = Vec::new();
    for index in 0..defects.bad_column_start.len() {
        let mut x1 = defects.bad_column_start[index] as i32;
        let mut x2 = x1 + defects.bad_column_width[index] as i32 - 1;
        let (mut y1, mut y2) = (0, 0);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x1, &mut y1);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x2, &mut y2);
        if operation % 2 != 0 {
            bad_row_start.push(y1.min(y2) as u16);
            bad_row_height.push(defects.bad_column_width[index]);
        } else {
            bad_column_start.push(x1.min(x2) as u16);
            bad_column_width.push(defects.bad_column_width[index]);
        }
    }
    for index in 0..defects.bad_row_start.len() {
        let mut y1 = defects.bad_row_start[index] as i32;
        let mut y2 = y1 + defects.bad_row_height[index] as i32 - 1;
        let (mut x1, mut x2) = (0, 0);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x1, &mut y1);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x2, &mut y2);
        if operation % 2 != 0 {
            bad_column_start.push(x1.min(x2) as u16);
            bad_column_width.push(defects.bad_row_height[index]);
        } else {
            bad_row_start.push(y1.min(y2) as u16);
            bad_row_height.push(defects.bad_row_height[index]);
        }
    }
    for index in 0..defects.partial_bad_col.len() {
        let mut x1 = defects.partial_bad_col[index] as i32;
        let mut x2 = x1 + defects.partial_bad_width[index] as i32 - 1;
        let mut y1 = defects.partial_bad_start_y[index] as i32;
        let mut y2 = defects.partial_bad_end_y[index] as i32;
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x1, &mut y1);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x2, &mut y2);
        if operation % 2 != 0 {
            partial_bad_row.push(y1.min(y2) as u16);
            partial_bad_height.push(defects.partial_bad_width[index]);
            partial_bad_start_x.push(x1.min(x2) as u16);
            partial_bad_end_x.push(x1.max(x2) as u16);
        } else {
            partial_bad_col.push(x1.min(x2) as u16);
            partial_bad_width.push(defects.partial_bad_width[index]);
            partial_bad_start_y.push(y1.min(y2) as u16);
            partial_bad_end_y.push(y1.max(y2) as u16);
        }
    }
    for index in 0..defects.partial_bad_row.len() {
        let mut y1 = defects.partial_bad_row[index] as i32;
        let mut y2 = y1 + defects.partial_bad_height[index] as i32 - 1;
        let mut x1 = defects.partial_bad_start_x[index] as i32;
        let mut x2 = defects.partial_bad_end_x[index] as i32;
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x1, &mut y1);
        cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x2, &mut y2);
        if operation % 2 != 0 {
            partial_bad_col.push(x1.min(x2) as u16);
            partial_bad_width.push(defects.partial_bad_height[index]);
            partial_bad_start_y.push(y1.min(y2) as u16);
            partial_bad_end_y.push(y1.max(y2) as u16);
        } else {
            partial_bad_row.push(y1.min(y2) as u16);
            partial_bad_height.push(defects.partial_bad_height[index]);
            partial_bad_start_x.push(x1.min(x2) as u16);
            partial_bad_end_x.push(x1.max(x2) as u16);
        }
    }
    let mut x1 = defects.usable_left;
    let mut x2 = if defects.usable_right != 0 {
        defects.usable_right
    } else if operation % 2 != 0 {
        camera_size_y
    } else {
        camera_size_x
    };
    let mut y1 = defects.usable_top;
    let mut y2 = if defects.usable_bottom != 0 {
        defects.usable_bottom
    } else if operation % 2 != 0 {
        camera_size_x
    } else {
        camera_size_y
    };
    cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x1, &mut y1);
    cor_def_rot_flip_ccdcoord(operation, camera_size_x, camera_size_y, &mut x2, &mut y2);
    defects.usable_top = y1.min(y2);
    defects.usable_bottom = y1.max(y2);
    defects.usable_left = x1.min(x2);
    defects.usable_right = x1.max(x2);
    defects.bad_column_start = bad_column_start;
    defects.bad_column_width = bad_column_width;
    defects.bad_row_start = bad_row_start;
    defects.bad_row_height = bad_row_height;
    defects.partial_bad_col = partial_bad_col;
    defects.partial_bad_width = partial_bad_width;
    defects.partial_bad_start_y = partial_bad_start_y;
    defects.partial_bad_end_y = partial_bad_end_y;
    defects.partial_bad_row = partial_bad_row;
    defects.partial_bad_height = partial_bad_height;
    defects.partial_bad_start_x = partial_bad_start_x;
    defects.partial_bad_end_x = partial_bad_end_x;
    defects.rotation_flip = rotation_flip;
}
/// Matches C++ `CorDefFindTouchingPixels`.
pub fn cor_def_find_touching_pixels(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    camera_size_x: i32,
    camera_size_y: i32,
    mut was_scaled: i32,
) {
    use std::collections::BTreeSet;
    if was_scaled == 0 {
        was_scaled = defects.was_scaled;
    }
    let x_difference = if was_scaled > 0 {
        if was_scaled > 1 { 4 } else { 2 }
    } else {
        1
    };
    let y_difference = 65_536_u32 * x_difference as u32;
    defects.pix_use_mean.clear();
    let mut point_map = BTreeSet::new();
    for index in 0..defects.bad_pixel_x.len() {
        point_map
            .insert((defects.bad_pixel_x[index] as u32) << 16 | defects.bad_pixel_y[index] as u32);
    }
    for index in 0..defects.bad_pixel_x.len() {
        let x = defects.bad_pixel_x[index] as i32;
        let y = defects.bad_pixel_y[index] as i32;
        let key = (x as u32) << 16 | y as u32;
        if (defects.usable_left > 0 && x <= defects.usable_left)
            || (defects.usable_right > 0
                && defects.usable_right < camera_size_x - 1
                && x >= defects.usable_right)
            || (defects.usable_top > 0 && y <= defects.usable_top)
            || (defects.usable_bottom > 0
                && defects.usable_bottom < camera_size_y - 1
                && y >= defects.usable_bottom)
            || check_point_near_full_lines(
                x,
                &defects.bad_column_start,
                &defects.bad_column_width,
                x_difference,
            ) != 0
            || check_point_near_full_lines(
                y,
                &defects.bad_row_start,
                &defects.bad_row_height,
                x_difference,
            ) != 0
            || check_point_near_partial_lines(
                x,
                y,
                &defects.partial_bad_col,
                &defects.partial_bad_width,
                &defects.partial_bad_start_y,
                &defects.partial_bad_end_y,
                x_difference,
            ) != 0
            || check_point_near_partial_lines(
                y,
                x,
                &defects.partial_bad_row,
                &defects.partial_bad_height,
                &defects.partial_bad_start_x,
                &defects.partial_bad_end_x,
                x_difference,
            ) != 0
            || (x > 0 && point_map.contains(&(key - y_difference)))
            || (x < camera_size_x - 1 && point_map.contains(&(key + y_difference)))
            || (y > 0 && point_map.contains(&(key - x_difference as u32)))
            || (y < camera_size_y - 1 && point_map.contains(&(key + x_difference as u32)))
        {
            if defects.pix_use_mean.is_empty() {
                defects.pix_use_mean.resize(defects.bad_pixel_x.len(), 0);
            }
            defects.pix_use_mean[index] = 1;
        }
    }
}
/// Matches C++ `CheckPointNearFullLines`.
pub fn check_point_near_full_lines(
    point: i32,
    columns: &[u16],
    widths: &[i16],
    left_difference: i32,
) -> i32 {
    for index in 0..columns.len() {
        if point >= columns[index] as i32 - left_difference
            && point <= columns[index] as i32 + widths[index] as i32
        {
            return 1;
        }
    }
    0
}
/// Matches C++ `CheckPointNearPartialLines`.
pub fn check_point_near_partial_lines(
    point: i32,
    other: i32,
    columns: &[u16],
    widths: &[i16],
    start_y: &[u16],
    end_y: &[u16],
    left_difference: i32,
) -> i32 {
    for index in 0..columns.len() {
        if point >= columns[index] as i32 - left_difference
            && point <= columns[index] as i32 + widths[index] as i32
            && other >= start_y[index] as i32 - left_difference
            && other <= end_y[index] as i32 + 1
        {
            return 1;
        }
    }
    0
}
/// Matches C++ `CorDefMergeDefectLists`.
pub fn cor_def_merge_defect_lists(
    defects: &mut crate::imod::clip::clip::CameraDefects,
    xy_pairs: &mut [u16],
    number_points: i32,
    camera_size_x: i32,
    camera_size_y: i32,
    rotation_flip: i32,
) {
    use std::collections::{BTreeMap, BTreeSet};
    defects.rotation_flip = rotation_flip;
    let mut point_map = BTreeMap::<u32, usize>::new();
    let mut source_xx = 0_i32;
    let mut source_yy = 0_i32;
    for point in 0..number_points as usize {
        let pair = 2 * point;
        let (mut x, mut y) = (xy_pairs[pair] as i32, xy_pairs[pair + 1] as i32);
        cor_def_rot_flip_ccdcoord(rotation_flip, camera_size_x, camera_size_y, &mut x, &mut y);
        source_xx = x;
        source_yy = y;
        xy_pairs[pair] = x as u16;
        xy_pairs[pair + 1] = y as u16;
        if (defects.usable_left > 0 && x < defects.usable_left)
            || (defects.usable_right > 0 && x > defects.usable_right)
            || (defects.usable_top > 0 && y < defects.usable_top)
            || (defects.usable_bottom > 0 && y > defects.usable_bottom)
            || check_if_point_in_full_lines(x, &defects.bad_column_start, &defects.bad_column_width)
                != 0
            || check_if_point_in_full_lines(y, &defects.bad_row_start, &defects.bad_row_height) != 0
            || check_if_point_in_partial_lines(
                x,
                y,
                &defects.partial_bad_col,
                &defects.partial_bad_width,
                &defects.partial_bad_start_y,
                &defects.partial_bad_end_y,
            ) != 0
            || check_if_point_in_partial_lines(
                y,
                x,
                &defects.partial_bad_row,
                &defects.partial_bad_height,
                &defects.partial_bad_start_x,
                &defects.partial_bad_end_x,
            ) != 0
            || defects
                .bad_pixel_x
                .iter()
                .zip(&defects.bad_pixel_y)
                .any(|(&px, &py)| px as i32 == x && py as i32 == y)
        {
            xy_pairs[pair] = u16::MAX;
        }
        if xy_pairs[pair] != u16::MAX {
            let key = (x as u32) << 16 | y as u32;
            if point_map.insert(key, pair).is_some() {
                xy_pairs[pair] = u16::MAX;
            }
        }
    }
    for point in 0..number_points as usize {
        let pair = 2 * point;
        if xy_pairs[pair] == u16::MAX {
            continue;
        }
        let mut in_group = vec![pair];
        let mut group_set = BTreeSet::<u32>::new();
        group_set.insert((source_xx as u32) << 16 | source_yy as u32);
        let mut checked = 0;
        while checked < in_group.len() {
            let member = in_group[checked];
            let x = xy_pairs[member] as i32;
            let y = xy_pairs[member + 1] as i32;
            let key = (x as u32) << 16 | y as u32;
            for neighbor in [
                (x > 0).then_some(key - 65_536),
                (x < camera_size_x - 1).then_some(key + 65_536),
                (y > 0).then_some(key - 1),
                (y < camera_size_y - 1).then_some(key + 1),
            ]
            .into_iter()
            .flatten()
            {
                if !group_set.contains(&neighbor) {
                    if let Some(&entry) = point_map.get(&neighbor) {
                        group_set.insert(neighbor);
                        in_group.push(entry);
                    }
                }
            }
            checked += 1;
        }
        let mut xmin = xy_pairs[pair] as i32;
        let mut xmax = xmin;
        let mut ymin = xy_pairs[pair + 1] as i32;
        let mut ymax = ymin;
        for &member in in_group.iter().skip(1) {
            let x = xy_pairs[member] as i32;
            let y = xy_pairs[member + 1] as i32;
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        if in_group.len() == 1 {
            defects.bad_pixel_x.push(xy_pairs[pair]);
            defects.bad_pixel_y.push(xy_pairs[pair + 1]);
        } else if ymax - ymin > xmax - xmin {
            add_rows_or_columns(
                xmin,
                xmax,
                ymin,
                ymax,
                camera_size_y,
                xy_pairs,
                &in_group
                    .iter()
                    .map(|&value| value as i32)
                    .collect::<Vec<_>>(),
                0,
                &mut defects.bad_column_start,
                &mut defects.bad_column_width,
                &mut defects.partial_bad_col,
                &mut defects.partial_bad_width,
                &mut defects.partial_bad_start_y,
                &mut defects.partial_bad_end_y,
                &mut defects.bad_pixel_x,
                &mut defects.bad_pixel_y,
            );
        } else {
            add_rows_or_columns(
                ymin,
                ymax,
                xmin,
                xmax,
                camera_size_x,
                xy_pairs,
                &in_group
                    .iter()
                    .map(|&value| value as i32)
                    .collect::<Vec<_>>(),
                1,
                &mut defects.bad_row_start,
                &mut defects.bad_row_height,
                &mut defects.partial_bad_row,
                &mut defects.partial_bad_height,
                &mut defects.partial_bad_start_x,
                &mut defects.partial_bad_end_x,
                &mut defects.bad_pixel_x,
                &mut defects.bad_pixel_y,
            );
        }
        for member in in_group {
            let key = (xy_pairs[member] as u32) << 16 | xy_pairs[member + 1] as u32;
            point_map.remove(&key);
            xy_pairs[member] = u16::MAX;
        }
    }
}
/// Matches C++ `CheckIfPointInFullLines`.
pub fn check_if_point_in_full_lines(point: i32, columns: &[u16], widths: &[i16]) -> i32 {
    for index in 0..columns.len() {
        if point >= columns[index] as i32
            && point <= columns[index] as i32 + widths[index] as i32 - 1
        {
            return 1;
        }
    }
    0
}
/// Matches C++ `CheckIfPointInPartialLines`.
pub fn check_if_point_in_partial_lines(
    point: i32,
    other: i32,
    columns: &[u16],
    widths: &[i16],
    start_y: &[u16],
    end_y: &[u16],
) -> i32 {
    for index in 0..columns.len() {
        if point >= columns[index] as i32
            && point <= columns[index] as i32 + widths[index] as i32 - 1
            && other >= start_y[index] as i32
            && other <= end_y[index] as i32
        {
            return 1;
        }
    }
    0
}
/// Matches C++ `AddRowsOrColumns`.
pub fn add_rows_or_columns(
    xmin: i32,
    xmax: i32,
    _ymin: i32,
    _ymax: i32,
    camera_size: i32,
    xy_pairs: &[u16],
    in_group: &[i32],
    coordinate_index: i32,
    column_start: &mut Vec<u16>,
    column_width: &mut Vec<i16>,
    partial: &mut Vec<u16>,
    partial_width: &mut Vec<i16>,
    start_y: &mut Vec<u16>,
    end_y: &mut Vec<u16>,
    bad_pixel_x: &mut Vec<u16>,
    bad_pixel_y: &mut Vec<u16>,
) {
    let width = xmax + 1 - xmin;
    let mut column_histogram = vec![0; width as usize];
    let mut column_ymin = vec![0; width as usize];
    let mut column_ymax = vec![0; width as usize];
    let mut maximum_histogram = 0;
    for &group_index in in_group {
        let column = xy_pairs[(group_index + coordinate_index) as usize] as i32 - xmin;
        let y = xy_pairs[(group_index + 1 - coordinate_index) as usize] as i32;
        if column_histogram[column as usize] != 0 {
            column_ymin[column as usize] = column_ymin[column as usize].min(y);
            column_ymax[column as usize] = column_ymax[column as usize].max(y);
        } else {
            column_ymin[column as usize] = y;
            column_ymax[column as usize] = y;
        }
        column_histogram[column as usize] += 1;
        maximum_histogram = maximum_histogram.max(column_histogram[column as usize]);
    }
    let mut one_width = 0;
    let mut one_xmin = 0;
    let mut one_ymin = 0;
    let mut one_ymax = 0;
    for column in 0..width {
        if column_histogram[column as usize] < maximum_histogram / 50 {
            for &group_index in in_group {
                if xy_pairs[(group_index + coordinate_index) as usize] as i32 - xmin == column {
                    bad_pixel_x.push(xy_pairs[group_index as usize]);
                    bad_pixel_y.push(xy_pairs[(group_index + 1) as usize]);
                }
            }
            if one_width != 0 {
                add_one_row_or_column(
                    one_xmin,
                    one_xmin + one_width - 1,
                    one_ymin,
                    one_ymax,
                    camera_size,
                    column_start,
                    column_width,
                    partial,
                    partial_width,
                    start_y,
                    end_y,
                );
                one_width = 0;
            }
        } else {
            if one_width == 0 {
                one_xmin = xmin + column;
                one_ymin = column_ymin[column as usize];
                one_ymax = column_ymax[column as usize];
            } else {
                one_ymin = one_ymin.min(column_ymin[column as usize]);
                one_ymax = one_ymax.max(column_ymax[column as usize]);
            }
            one_width += 1;
        }
    }
    if one_width != 0 {
        add_one_row_or_column(
            one_xmin,
            one_xmin + one_width - 1,
            one_ymin,
            one_ymax,
            camera_size,
            column_start,
            column_width,
            partial,
            partial_width,
            start_y,
            end_y,
        );
    }
}
/// Matches C++ `AddOneRowOrColumn`.
pub fn add_one_row_or_column(
    xmin: i32,
    xmax: i32,
    ymin: i32,
    ymax: i32,
    camera_size: i32,
    column_start: &mut Vec<u16>,
    column_width: &mut Vec<i16>,
    partial: &mut Vec<u16>,
    partial_width: &mut Vec<i16>,
    start_y: &mut Vec<u16>,
    end_y: &mut Vec<u16>,
) {
    let width = xmax + 1 - xmin;
    if ymin == 0 && ymax == camera_size - 1 {
        for index in 0..column_start.len() {
            if xmax == column_start[index] as i32 - 1
                || xmin == column_start[index] as i32 + column_width[index] as i32
            {
                column_start[index] = column_start[index].min(xmin as u16);
                column_width[index] += width as i16;
                return;
            }
        }
        column_start.push(xmin as u16);
        column_width.push(width as i16);
    } else {
        partial.push(xmin as u16);
        partial_width.push(width as i16);
        start_y.push(ymin as u16);
        end_y.push(ymax as u16);
    }
}
/// Matches C++ `CorDefAddBadColumn`.
pub fn cor_def_add_bad_column(
    column: i32,
    bad_column_start: &mut Vec<u16>,
    bad_column_width: &mut Vec<i16>,
) {
    let mut index = 0;
    while index < bad_column_start.len() {
        if column == bad_column_start[index] as i32 + bad_column_width[index] as i32 {
            bad_column_width[index] += 1;
            break;
        }
        index += 1;
    }
    if index == bad_column_start.len() {
        bad_column_start.push(column as u16);
        bad_column_width.push(1);
    }
}
/// Matches C++ `CorDefAddPartialBadCol`.
pub fn cor_def_add_partial_bad_col(
    values: &[i32; 4],
    partial_bad_column: &mut Vec<u16>,
    partial_bad_width: &mut Vec<i16>,
    partial_bad_start_y: &mut Vec<u16>,
    partial_bad_end_y: &mut Vec<u16>,
) {
    partial_bad_column.push(values[0] as u16);
    partial_bad_width.push(values[1] as i16);
    partial_bad_start_y.push(values[2] as u16);
    partial_bad_end_y.push(values[3] as u16);
}
/// Matches C++ `CorDefDefectsToString`.
pub fn cor_def_defects_to_string(
    defects: &crate::imod::clip::clip::CameraDefects,
    output: &mut String,
    camera_size_x: i32,
    camera_size_y: i32,
) {
    output.push_str(&format!(
        "CameraSizeX {camera_size_x}\nCameraSizeY {camera_size_y}\n"
    ));
    output.push_str(&format!(
        "RotationAndFlip {}\nWasScaled {}\nK2Type {}\nFalconType {}\n",
        defects.rotation_flip, defects.was_scaled, defects.k2_type, defects.falcon_type
    ));
    output.push_str(&format!(
        "NumToAvgSuperRes {}\nUsableArea {} {} {} {}\n",
        defects.num_avg_super_res,
        defects.usable_top,
        defects.usable_left,
        defects.usable_bottom,
        defects.usable_right
    ));
    for index in 0..defects.partial_bad_col.len() {
        output.push_str(&format!(
            "PartialBadColumn {} {} {} {}\n",
            defects.partial_bad_col[index],
            defects.partial_bad_width[index],
            defects.partial_bad_start_y[index],
            defects.partial_bad_end_y[index]
        ));
    }
    for index in 0..defects.partial_bad_row.len() {
        output.push_str(&format!(
            "PartialBadRow {} {} {} {}\n",
            defects.partial_bad_row[index],
            defects.partial_bad_height[index],
            defects.partial_bad_start_x[index],
            defects.partial_bad_end_x[index]
        ));
    }
    let mut count = 0;
    let mut buffer = String::new();
    for index in 0..defects.bad_pixel_x.len() {
        if count == 0 {
            buffer = "BadPixels".to_owned();
        }
        buffer.push_str(&format!(
            " {} {}",
            defects.bad_pixel_x[index], defects.bad_pixel_y[index]
        ));
        count += 1;
        if count == 10 || index == defects.bad_pixel_x.len() - 1 {
            buffer.push('\n');
            output.push_str(&buffer);
            count = 0;
        }
    }
    bad_rows_or_cols_to_string(
        &defects.bad_column_start,
        &defects.bad_column_width,
        output,
        "BadColumns",
    );
    bad_rows_or_cols_to_string(
        &defects.bad_row_start,
        &defects.bad_row_height,
        output,
        "BadRows",
    );
}
/// Matches C++ `BadRowsOrColsToString`.
pub fn bad_rows_or_cols_to_string(starts: &[u16], widths: &[i16], output: &mut String, name: &str) {
    let mut count = 0;
    let mut buffer = String::new();
    for index in 0..starts.len() {
        for column in 0..widths[index] {
            if count == 0 {
                buffer = name.to_owned();
            }
            buffer.push_str(&format!(" {}", starts[index] as i32 + column as i32));
            count += 1;
            if count == 20 {
                buffer.push('\n');
                output.push_str(&buffer);
                count = 0;
            }
        }
    }
    if count != 0 {
        buffer.push('\n');
        output.push_str(&buffer);
    }
}
/// C++ `CorDefParseDefects` (`CorrectDefects.cpp:1624`).
pub fn cor_def_parse_defects(
    source: &str,
    from_string: bool,
    defects: &mut crate::imod::clip::clip::CameraDefects,
    camera_size_x: &mut i32,
    camera_size_y: &mut i32,
) -> i32 {
    *camera_size_x = 0;
    *camera_size_y = 0;
    clear_defect_list(defects);
    let text = if from_string {
        source.as_bytes().to_vec()
    } else {
        match std::fs::read(source) {
            Ok(text) => text,
            Err(_) => return 1,
        }
    };
    let mut records = Vec::new();
    if from_string {
        for raw_line in text.split(|byte| *byte == b'\n') {
            records.push(&raw_line[..raw_line.len().min(511)]);
        }
    } else {
        let mut start = 0;
        while start < text.len() {
            let end = (start + 511).min(text.len());
            if let Some(relative_newline) = text[start..end].iter().position(|byte| *byte == b'\n')
            {
                records.push(&text[start..start + relative_newline]);
                start += relative_newline + 1;
            } else {
                records.push(&text[start..end]);
                // C `fgetline` evaluates getc before its `i < limit - 1`
                // condition, so a full 511-byte record consumes one further
                // byte before the next call.
                start = if end < text.len() { end + 1 } else { end };
            }
        }
    }
    for raw_line in records {
        let raw_line = if raw_line.last() == Some(&b'\r') {
            &raw_line[..raw_line.len() - 1]
        } else {
            raw_line
        };
        let line = match core::str::from_utf8(raw_line) {
            Ok(line) => line,
            Err(_) => return 2,
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(end_tag) = line.find(|character: char| character == ' ' || character == '\t')
        else {
            continue;
        };
        let tag = &line[..end_tag];
        let value_string = match std::ffi::CString::new(&line[end_tag..]) {
            Ok(value) => value,
            Err(_) => return 2,
        };
        let mut value_array = [0_i32; 100];
        let mut number_to_get = 0;
        let pip_error = unsafe {
            crate::imod::libcfshr::parse_params::pip_get_line_of_values(
                c" ".as_ptr(),
                value_string.as_ptr(),
                value_array.as_mut_ptr().cast(),
                1,
                &mut number_to_get,
                value_array.len() as i32,
            )
        };
        if pip_error != 0 {
            return 2;
        }
        let values = &value_array[..number_to_get as usize];
        match tag {
            tag if "CameraSizeX".starts_with(tag) => {
                if let Some(value) = values.first() {
                    *camera_size_x = *value
                } else {
                    return 2;
                }
            }
            tag if "CameraSizeY".starts_with(tag) => {
                if let Some(value) = values.first() {
                    *camera_size_y = *value
                } else {
                    return 2;
                }
            }
            tag if "RotationAndFlip".starts_with(tag) => {
                if let Some(value) = values.first() {
                    defects.rotation_flip = *value
                } else {
                    return 2;
                }
            }
            tag if "WasScaled".starts_with(tag) => {
                if let Some(value) = values.first() {
                    defects.was_scaled = *value
                } else {
                    return 2;
                }
            }
            tag if "K2Type".starts_with(tag) => {
                if let Some(value) = values.first() {
                    defects.k2_type = *value
                } else {
                    return 2;
                }
            }
            tag if "FalconType".starts_with(tag) => {
                if let Some(value) = values.first() {
                    defects.falcon_type = *value
                } else {
                    return 2;
                }
            }
            tag if "NumToAvgSuperRes".starts_with(tag) => {
                if let Some(value) = values.first() {
                    defects.num_avg_super_res = *value
                } else {
                    return 2;
                }
            }
            tag if "UsableArea".starts_with(tag) => {
                if values.len() >= 4 {
                    defects.usable_top = values[0];
                    defects.usable_left = values[1];
                    defects.usable_bottom = values[2];
                    defects.usable_right = values[3];
                } else {
                    return 2;
                }
            }
            tag if "PartialBadColumn".starts_with(tag) => {
                if values.len() >= 4 {
                    cor_def_add_partial_bad_col(
                        &[values[0], values[1], values[2], values[3]],
                        &mut defects.partial_bad_col,
                        &mut defects.partial_bad_width,
                        &mut defects.partial_bad_start_y,
                        &mut defects.partial_bad_end_y,
                    );
                } else {
                    return 2;
                }
            }
            tag if "PartialBadRow".starts_with(tag) => {
                if values.len() >= 4 {
                    cor_def_add_partial_bad_col(
                        &[values[0], values[1], values[2], values[3]],
                        &mut defects.partial_bad_row,
                        &mut defects.partial_bad_height,
                        &mut defects.partial_bad_start_x,
                        &mut defects.partial_bad_end_x,
                    );
                } else {
                    return 2;
                }
            }
            tag if "BadPixels".starts_with(tag) => {
                if values.len() < 2 || values.len() % 2 != 0 {
                    return 2;
                }
                for pair in values.chunks_exact(2) {
                    defects.bad_pixel_x.push(pair[0] as u16);
                    defects.bad_pixel_y.push(pair[1] as u16);
                }
            }
            tag if "BadColumns".starts_with(tag) => {
                if values.is_empty() {
                    return 2;
                } else {
                    for value in values {
                        cor_def_add_bad_column(
                            *value,
                            &mut defects.bad_column_start,
                            &mut defects.bad_column_width,
                        );
                    }
                }
            }
            tag if "BadRows".starts_with(tag) => {
                if values.is_empty() {
                    return 2;
                } else {
                    for value in values {
                        cor_def_add_bad_column(
                            *value,
                            &mut defects.bad_row_start,
                            &mut defects.bad_row_height,
                        );
                    }
                }
            }
            _ => {}
        }
    }
    0
}
/// Matches C++ `clearDefectList`.
pub fn clear_defect_list(defects: &mut crate::imod::clip::clip::CameraDefects) {
    defects.usable_top = 0;
    defects.usable_left = 0;
    defects.usable_bottom = 0;
    defects.usable_right = 0;
    defects.rotation_flip = 0;
    defects.was_scaled = 0;
    defects.k2_type = 0;
    defects.falcon_type = 0;
    defects.bad_column_start.clear();
    defects.bad_column_width.clear();
    defects.partial_bad_col.clear();
    defects.partial_bad_width.clear();
    defects.partial_bad_start_y.clear();
    defects.partial_bad_end_y.clear();
    defects.bad_row_start.clear();
    defects.bad_row_height.clear();
    defects.partial_bad_row.clear();
    defects.partial_bad_height.clear();
    defects.partial_bad_start_x.clear();
    defects.partial_bad_end_x.clear();
    defects.bad_pixel_x.clear();
    defects.bad_pixel_y.clear();
    defects.pix_use_mean.clear();
}
/// C++ `CorDefParseFeiXml` (`CorrectDefects.cpp:1802`).
pub unsafe fn cor_def_parse_fei_xml(
    text: *const core::ffi::c_char,
    defects: &mut CameraDefects,
    mut pad: i32,
) -> i32 {
    let column_pad = pad / 10;
    pad %= 10;
    clear_defect_list(defects);
    let mut root = core::ptr::null_mut();
    let xml_ind = unsafe { crate::imod::libcfshr::mxmlwrap::ixml_load_string(text, 0, &mut root) };
    if xml_ind < 0 {
        return xml_ind;
    }
    if root.is_null() || unsafe { libc::strcmp(root, c"defects".as_ptr()) } != 0 {
        unsafe {
            libc::free(root.cast());
            crate::imod::libcfshr::mxmlwrap::ixml_clear(xml_ind);
        }
        return -4;
    }
    unsafe {
        libc::free(root.cast());
    }
    let tags = [
        c"row".as_ptr(),
        c"col".as_ptr(),
        c"area".as_ptr(),
        c"nonmaskingpoint".as_ptr(),
        c"point".as_ptr(),
        c"nonmaskingpoint".as_ptr(),
    ];
    for tag_ind in 0..6 {
        let mut start_ind = 0;
        let mut number = 0;
        let err = unsafe {
            crate::imod::libcfshr::mxmlwrap::ixml_find_elements(
                xml_ind,
                0,
                tags[tag_ind],
                &mut start_ind,
                &mut number,
            )
        };
        if err != 0 {
            unsafe {
                crate::imod::libcfshr::mxmlwrap::ixml_clear(xml_ind);
            }
            return -err;
        }
        for elem_ind in 0..number {
            let mut value = core::ptr::null_mut();
            let mut err = unsafe {
                crate::imod::libcfshr::mxmlwrap::ixml_get_string_value(
                    xml_ind,
                    start_ind + elem_ind,
                    &mut value,
                )
            };
            if err != 0 || value.is_null() {
                if err == 0 {
                    err = -5;
                }
                unsafe {
                    libc::free(value.cast());
                    crate::imod::libcfshr::mxmlwrap::ixml_clear(xml_ind);
                }
                return -err;
            }
            if unsafe { !libc::strchr(value, b'/' as i32).is_null() } {
                unsafe {
                    libc::free(value.cast());
                }
                continue;
            }
            let mut num_list = 0;
            let values =
                unsafe { crate::imod::libcfshr::parselist::parselist(value, &mut num_list) };
            unsafe {
                libc::free(value.cast());
            }
            if values.is_null() {
                unsafe {
                    crate::imod::libcfshr::mxmlwrap::ixml_clear(xml_ind);
                }
                return 6 - num_list;
            }
            if ((tag_ind == 2 || tag_ind == 3) && num_list != 4)
                || ((tag_ind == 4 || tag_ind == 5) && num_list != 2)
            {
                unsafe {
                    libc::free(values.cast());
                    crate::imod::libcfshr::mxmlwrap::ixml_clear(xml_ind);
                }
                return 10;
            }
            let adjusted = 0.max(unsafe { *values } - column_pad);
            unsafe {
                match tag_ind {
                    0 => {
                        defects.bad_row_start.push(adjusted as u16);
                        defects
                            .bad_row_height
                            .push((num_list + column_pad + *values - adjusted) as i16);
                    }
                    1 => {
                        defects.bad_column_start.push(adjusted as u16);
                        defects
                            .bad_column_width
                            .push((num_list + column_pad + *values - adjusted) as i16);
                    }
                    2 | 3 => {
                        if *values.add(2) - *values > *values.add(3) - *values.add(1) {
                            defects.partial_bad_row.push(*values.add(1) as u16);
                            defects
                                .partial_bad_height
                                .push((1 + *values.add(3) - *values.add(1)) as i16);
                            defects.partial_bad_start_x.push(*values as u16);
                            defects.partial_bad_end_x.push(*values.add(2) as u16);
                        } else {
                            defects.partial_bad_col.push(*values as u16);
                            defects
                                .partial_bad_width
                                .push((1 + *values.add(2) - *values) as i16);
                            defects.partial_bad_start_y.push(*values.add(1) as u16);
                            defects.partial_bad_end_y.push(*values.add(3) as u16);
                        }
                    }
                    4 | 5 => {
                        defects.bad_pixel_x.push(*values as u16);
                        defects.bad_pixel_y.push(*values.add(1) as u16);
                    }
                    _ => {}
                }
                libc::free(values.cast());
            }
        }
    }
    unsafe {
        crate::imod::libcfshr::mxmlwrap::ixml_clear(xml_ind);
    }
    defects.falcon_type = 1;
    defects.num_avg_super_res = pad.clamp(0, 4);
    0
}
/// C++ `CorDefProcessFeiDefects` (`CorrectDefects.cpp:1916`).
pub unsafe fn cor_def_process_fei_defects(
    ii_file: *mut crate::imod::libiimod::iimage::ImodImageFile,
    defects: &mut crate::imod::clip::clip::CameraDefects,
    nx: i32,
    ny: i32,
    flip_y: bool,
    super_fac: i32,
    fei_def_pad: i32,
    dump_defect_name: *const core::ffi::c_char,
    mess_buf: *mut core::ffi::c_char,
    buf_len: i32,
) -> i32 {
    let mut count = 0_u32;
    let mut text: *mut core::ffi::c_char = core::ptr::null_mut();
    if unsafe {
        crate::imod::libiimod::iitif::tiff_get_array(
            ii_file,
            65_100,
            &mut count,
            (&mut text).cast(),
        )
    } <= 0
        || count == 0
    {
        return -1;
    }
    let bytes = unsafe { core::slice::from_raw_parts(text.cast::<u8>(), count as usize) };
    let mut copy = Vec::with_capacity(count as usize + 1);
    copy.extend_from_slice(bytes);
    copy.push(0);
    let parsed = cor_def_parse_fei_xml(copy.as_ptr().cast(), defects, fei_def_pad);
    if parsed != 0 {
        libc::snprintf(
            mess_buf,
            buf_len as usize,
            c"Parsing defect string from TIFF file (error %d)".as_ptr(),
            parsed,
        );
        return 1;
    }
    if !dump_defect_name.is_null() {
        let mut text = String::new();
        cor_def_defects_to_string(defects, &mut text, nx, ny);
        crate::imod::libcfshr::b3dutil::imod_backup_file(dump_defect_name);
        let file = libc::fopen(dump_defect_name, c"w".as_ptr());
        if file.is_null() {
            libc::snprintf(
                mess_buf,
                buf_len as usize,
                c"Opening file to write defects to, %s".as_ptr(),
                dump_defect_name,
            );
            return 1;
        }
        libc::fputs(std::ffi::CString::new(text).unwrap().as_ptr(), file);
        libc::fclose(file);
    }
    if flip_y {
        cor_def_flip_defects_in_y(defects, nx, ny, 0);
    }
    cor_def_find_touching_pixels(defects, nx, ny, 0);
    if super_fac != 1 {
        cor_def_scale_defects_for_falcon(defects, super_fac);
    }
    0
}
/// Matches C++ `CorDefFillDefectArray`.
pub unsafe fn cor_def_fill_defect_array(
    defects: *const crate::imod::clip::clip::CameraDefects,
    camera_size_x: i32,
    camera_size_y: i32,
    array: *mut u8,
    nx: i32,
    ny: i32,
    do_falcon_pad: bool,
) -> i32 {
    unsafe {
        let defects = &*defects;
        if nx <= 0 || ny <= 0 || camera_size_x <= 0 || camera_size_y <= 0 {
            return 1;
        }
        core::ptr::write_bytes(array, 0, (nx * ny) as usize);
        let ix_offset = (nx - camera_size_x) / 2;
        let iy_offset = (ny - camera_size_y) / 2;
        let number_pixels = if (*defects).was_scaled > 1 { 4 } else { 2 };
        let do_falcon_pad = do_falcon_pad
            && (*defects).falcon_type != 0
            && (*defects).num_avg_super_res > 0
            && (*defects).was_scaled != 0;
        let (mut x_low_extra, mut x_high_extra) = (0, camera_size_x);
        let (mut y_low_extra, mut y_high_extra) = (0, camera_size_y);
        if ix_offset > 0 {
            x_low_extra = -ix_offset;
            x_high_extra = nx - ix_offset;
        }
        if iy_offset > 0 {
            y_low_extra = -iy_offset;
            y_high_extra = ny - iy_offset;
        }
        macro_rules! set_pixel {
            ($x:expr, $y:expr, $value:expr) => {{
                let ix = $x + ix_offset;
                let iy = $y + iy_offset;
                if ix >= 0 && ix < nx && iy >= 0 && iy < ny {
                    *array.add((ix + iy * nx) as usize) = $value;
                }
            }};
        }
        for index in 0..(*defects).bad_pixel_x.len() {
            let bad_x = (*defects).bad_pixel_x[index] as i32;
            let bad_y = (*defects).bad_pixel_y[index] as i32;
            if (*defects).was_scaled > 0 {
                for jy in 0..number_pixels {
                    for jx in 0..number_pixels {
                        set_pixel!(bad_x + jx, bad_y + jy, 1);
                    }
                }
            } else {
                set_pixel!(bad_x, bad_y, 1);
            }
        }
        for column in 0..(*defects).bad_column_start.len() {
            for offset in 0..(*defects).bad_column_width[column] as i32 {
                let bad_x = (*defects).bad_column_start[column] as i32 + offset;
                for bad_y in y_low_extra..y_high_extra {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
            if do_falcon_pad {
                for offset in 0..(*defects).num_avg_super_res {
                    let bad_x =
                        (*defects).bad_column_start[column] as i32 - (offset + 1) * number_pixels;
                    if bad_x >= 0 {
                        for bad_y in y_low_extra..y_high_extra {
                            set_pixel!(bad_x, bad_y, 254);
                        }
                    }
                    let bad_x = (*defects).bad_column_start[column] as i32
                        + (*defects).bad_column_width[column] as i32
                        + offset * number_pixels;
                    if bad_x < nx {
                        for bad_y in y_low_extra..y_high_extra {
                            set_pixel!(bad_x, bad_y, 254);
                        }
                    }
                }
            }
        }
        for row in 0..(*defects).bad_row_start.len() {
            for offset in 0..(*defects).bad_row_height[row] as i32 {
                let bad_y = (*defects).bad_row_start[row] as i32 + offset;
                for bad_x in x_low_extra..x_high_extra {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
            if do_falcon_pad {
                for offset in 0..(*defects).num_avg_super_res {
                    let bad_y = (*defects).bad_row_start[row] as i32 - (offset + 1) * number_pixels;
                    if bad_y >= 0 {
                        for bad_x in x_low_extra..x_high_extra {
                            set_pixel!(bad_x, bad_y, 255);
                        }
                    }
                    let bad_y = (*defects).bad_row_start[row] as i32
                        + (*defects).bad_row_height[row] as i32
                        + offset * number_pixels;
                    if bad_y < ny {
                        for bad_x in x_low_extra..x_high_extra {
                            set_pixel!(bad_x, bad_y, 255);
                        }
                    }
                }
            }
        }
        for column in 0..(*defects).partial_bad_col.len() {
            for offset in 0..(*defects).partial_bad_width[column] as i32 {
                let bad_x = (*defects).partial_bad_col[column] as i32 + offset;
                for bad_y in (*defects).partial_bad_start_y[column] as i32
                    ..=(*defects).partial_bad_end_y[column] as i32
                {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
            if do_falcon_pad {
                for offset in 0..(*defects).num_avg_super_res {
                    let bad_x =
                        (*defects).bad_column_start[column] as i32 - (offset + 1) * number_pixels;
                    if bad_x >= 0 {
                        for bad_y in (*defects).partial_bad_start_y[column] as i32
                            ..=(*defects).partial_bad_end_y[column] as i32
                        {
                            set_pixel!(bad_x, bad_y, 254);
                        }
                    }
                    let bad_x = (*defects).bad_column_start[column] as i32
                        + (*defects).bad_column_width[column] as i32
                        + offset * number_pixels;
                    if bad_x < nx {
                        for bad_y in (*defects).partial_bad_start_y[column] as i32
                            ..=(*defects).partial_bad_end_y[column] as i32
                        {
                            set_pixel!(bad_x, bad_y, 254);
                        }
                    }
                }
            }
        }
        for row in 0..(*defects).partial_bad_row.len() {
            for offset in 0..(*defects).partial_bad_height[row] as i32 {
                let bad_y = (*defects).partial_bad_row[row] as i32 + offset;
                for bad_x in (*defects).partial_bad_start_x[row] as i32
                    ..=(*defects).partial_bad_end_x[row] as i32
                {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
            if do_falcon_pad {
                for offset in 0..(*defects).num_avg_super_res {
                    let bad_y = (*defects).bad_row_start[row] as i32 - (offset + 1) * number_pixels;
                    if bad_y >= 0 {
                        for bad_x in (*defects).partial_bad_start_x[row] as i32
                            ..=(*defects).partial_bad_end_x[row] as i32
                        {
                            set_pixel!(bad_x, bad_y, 255);
                        }
                    }
                    let bad_y = (*defects).bad_row_start[row] as i32
                        + (*defects).bad_row_height[row] as i32
                        + offset * number_pixels;
                    if bad_y < ny {
                        for bad_x in (*defects).partial_bad_start_x[row] as i32
                            ..=(*defects).partial_bad_end_x[row] as i32
                        {
                            set_pixel!(bad_x, bad_y, 255);
                        }
                    }
                }
            }
        }
        if (*defects).usable_left > 0 {
            for bad_x in x_low_extra..(*defects).usable_left {
                for bad_y in y_low_extra..y_high_extra {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
        }
        if (*defects).usable_right > 0 {
            for bad_x in (*defects).usable_right + 1..x_high_extra {
                for bad_y in y_low_extra..y_high_extra {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
        }
        if (*defects).usable_top > 0 {
            for bad_y in y_low_extra..(*defects).usable_top {
                for bad_x in x_low_extra..x_high_extra {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
        }
        if (*defects).usable_bottom > 0 {
            for bad_y in (*defects).usable_bottom + 1..x_high_extra {
                for bad_x in x_low_extra..x_high_extra {
                    set_pixel!(bad_x, bad_y, 1);
                }
            }
        }
    }
    0
}
/// C++ `CorDefExpandGainReference` (`CorrectDefects.cpp:2125`).
///
/// # Safety
///
/// `reference_in` contains `nx_in * ny_in` floats and `reference_out` contains
/// `(nx_in * factor) * (ny_in * factor)` floats.
pub unsafe fn cor_def_expand_gain_reference(
    reference_in: *const f32,
    nx_in: i32,
    ny_in: i32,
    factor: i32,
    reference_out: *mut f32,
) {
    unsafe {
        let nx_out = nx_in * factor;
        for iy in (0..ny_in).rev() {
            let input_line = reference_in.add((iy * nx_in) as usize);
            let output_line = reference_out.add(((iy * factor + factor - 1) * nx_out) as usize);
            let mut output = output_line;
            for ix in 0..nx_in {
                for _ in 0..factor {
                    *output = *input_line.add(ix as usize);
                    output = output.add(1);
                }
            }
            core::ptr::copy_nonoverlapping(
                output_line,
                output_line.sub(nx_out as usize),
                nx_out as usize,
            );
            if factor > 2 {
                core::ptr::copy_nonoverlapping(
                    output_line,
                    output_line.sub((2 * nx_out) as usize),
                    nx_out as usize,
                );
                core::ptr::copy_nonoverlapping(
                    output_line,
                    output_line.sub((3 * nx_out) as usize),
                    nx_out as usize,
                );
            }
        }
    }
}
/// C++ `CorDefReadSuperGain` (`CorrectDefects.cpp:2158`).
pub fn cor_def_read_super_gain(
    filename: &str,
    super_fac: i32,
    biases: &mut Vec<Vec<f32>>,
    num_in_x: &mut i32,
    x_start: &mut i32,
    x_spacing: &mut i32,
    num_in_y: &mut i32,
    y_start: &mut i32,
    y_spacing: &mut i32,
) -> i32 {
    let name = match std::ffi::CString::new(filename) {
        Ok(value) => value,
        Err(_) => return 1,
    };
    let file = unsafe { libc::fopen(name.as_ptr(), c"r".as_ptr()) };
    if file.is_null() {
        return 1;
    }
    let mut version = 0;
    let mut done_at_fac = 0;
    let scan = unsafe { libc::fscanf(file, c"%d %d".as_ptr(), &mut version, &mut done_at_fac) };
    if scan == libc::EOF {
        return 2;
    }
    if scan != 0 && scan < 2 {
        return 4;
    }
    if version != 1 || !matches!(done_at_fac, 2 | 4) || done_at_fac < super_fac {
        return 3;
    }
    let mut header = [0_i32; 6];
    let scan = unsafe {
        libc::fscanf(
            file,
            c"%d %d %d %d %d %d".as_ptr(),
            header.as_mut_ptr().add(0),
            header.as_mut_ptr().add(1),
            header.as_mut_ptr().add(2),
            header.as_mut_ptr().add(3),
            header.as_mut_ptr().add(4),
            header.as_mut_ptr().add(5),
        )
    };
    if scan == libc::EOF {
        return 2;
    }
    if scan != 0 && scan < 6 {
        return 4;
    }
    (
        *num_in_x, *x_start, *x_spacing, *num_in_y, *y_start, *y_spacing,
    ) = (
        header[0], header[1], header[2], header[3], header[4], header[5],
    );
    biases.clear();
    if super_fac != done_at_fac {
        let divisor = super_fac / done_at_fac;
        *x_start /= divisor;
        *y_start /= divisor;
        *x_spacing /= divisor;
        *y_spacing /= divisor;
    }
    for _ in 0..*num_in_y {
        for _ in 0..*num_in_x {
            let mut bias16 = Vec::with_capacity(16);
            for _ in 0..16 {
                let mut value = 0.;
                let scan = unsafe { libc::fscanf(file, c"%f".as_ptr(), &mut value) };
                if scan == libc::EOF {
                    return 2;
                }
                if scan != 0 && scan < 1 {
                    return 4;
                }
                bias16.push(value);
            }
            let mut bias4 = Vec::with_capacity(4);
            for _ in 0..4 {
                let mut value = 0.;
                let scan = unsafe { libc::fscanf(file, c"%f".as_ptr(), &mut value) };
                if scan == libc::EOF {
                    return 2;
                }
                if scan != 0 && scan < 1 {
                    return 4;
                }
                bias4.push(value);
            }
            biases.push(if super_fac == 2 { bias4 } else { bias16 });
        }
    }
    unsafe {
        libc::fclose(file);
    }
    0
}
/// C++ `CorDefRefineSuperResRef` (`CorrectDefects.cpp:2220`).
pub fn cor_def_refine_super_res_ref(
    reference: &mut [f32],
    nx: i32,
    ny: i32,
    super_fac: i32,
    biases: &[Vec<f32>],
    num_in_x: i32,
    x_start: i32,
    x_spacing: i32,
    num_in_y: i32,
    y_start: i32,
    y_spacing: i32,
) {
    let _xbase = x_spacing / 2 - x_start;
    let _ybase = y_spacing / 2 - x_start;
    for ydiv in 0..num_in_y {
        let iy0 = 0.max(y_start + ydiv * y_spacing - y_spacing / 2);
        let iy1 = (ny - 1).min(y_start + (ydiv + 1) * y_spacing - y_spacing / 2);
        for xdiv in 0..num_in_x {
            let ix0 = 0.max(x_start + xdiv * x_spacing - x_spacing / 2);
            let ix1 = (nx - 1).min(x_start + (xdiv + 1) * x_spacing - x_spacing / 2);
            let bias = &biases[(xdiv + ydiv * num_in_x) as usize];
            for iy in iy0..iy1 {
                let ind_base = super_fac * (iy % super_fac);
                let ix_base = iy * nx;
                for ix in ix0..ix1 {
                    reference[(ix + ix_base) as usize] *=
                        bias[(ix % super_fac + ind_base) as usize];
                }
            }
        }
    }
}
/// C++ `CorDefFindDriftCorrEdges` (`CorrectDefects.cpp:2286`).
pub unsafe fn cor_def_find_drift_corr_edges(
    array: *const core::ffi::c_void,
    data_type: i32,
    nx: i32,
    ny: i32,
    analyze_len: i32,
    max_width: i32,
    crit_madns: f32,
    x_low: &mut i32,
    x_high: &mut i32,
    y_low: &mut i32,
    y_high: &mut i32,
) -> i32 {
    if !matches!(data_type, 1 | 6 | 2) || nx < 2 || ny < 2 || max_width < 1 {
        return 1;
    }
    let x_start = (nx / 2 - analyze_len / 2).max(0);
    let y_start = (ny / 2 - analyze_len / 2).max(0);
    let x_end = (x_start + analyze_len).min(nx);
    let y_end = (y_start + analyze_len).min(ny);
    if x_end - x_start < 2 || y_end - y_start < 2 || max_width >= nx || max_width >= ny {
        return 1;
    }
    let mut line_mean = vec![0_f32; (4 * max_width) as usize];
    let mut diff_mean = vec![0_f32; (4 * max_width) as usize];
    let mut diff_sd = vec![0_f32; (4 * max_width) as usize];
    let mut ratio = vec![0_f32; (4 * max_width) as usize];
    macro_rules! value {
        ($index:expr) => {
            match data_type {
                1 => unsafe { *array.cast::<i16>().add($index as usize) as f32 },
                6 => unsafe { *array.cast::<u16>().add($index as usize) as f32 },
                _ => unsafe { *array.cast::<f32>().add($index as usize) },
            }
        };
    }
    for wid in 0..max_width {
        for idir in 0..2 {
            let iy = if idir != 0 { ny - 1 - wid } else { wid };
            let step = 1 - 2 * idir;
            let mut sum = 0_f64;
            let mut dsum = 0_f64;
            let mut dsq = 0_f64;
            for ix in x_start..x_end {
                let current = value!(ix + iy * nx) as f64;
                let difference = value!(ix + (iy + step) * nx) as f64 - current;
                sum += current;
                dsum += difference;
                dsq += difference * difference;
            }
            let count = (x_end - x_start) as f64;
            let ind = (idir * max_width + wid) as usize;
            line_mean[ind] = (sum / count) as f32;
            crate::imod::libcfshr::simplestat::sums_to_avg_sd_dbl(
                dsum,
                dsq,
                1,
                x_end - x_start,
                &mut diff_mean[ind],
                &mut diff_sd[ind],
            );
            ratio[ind] = diff_mean[ind] / diff_sd[ind].max(0.1);
        }
    }
    for loop_index in 0..2 {
        let idir = 1 - 2 * loop_index;
        for wid in 0..max_width {
            let mut sum = 0_f64;
            let mut dsum = 0_f64;
            let mut dsq = 0_f64;
            for iy in y_start..y_end {
                let ix = if loop_index != 0 { nx - 1 - wid } else { wid };
                let current = value!(ix + iy * nx) as f64;
                let difference = value!(ix + idir + iy * nx) as f64 - current;
                sum += current;
                dsum += difference;
                dsq += difference * difference;
            }
            let count = (y_end - y_start) as f64;
            let ind = (2 * max_width + loop_index * max_width + wid) as usize;
            line_mean[ind] = (sum / count) as f32;
            crate::imod::libcfshr::simplestat::sums_to_avg_sd_dbl(
                dsum,
                dsq,
                1,
                y_end - y_start,
                &mut diff_mean[ind],
                &mut diff_sd[ind],
            );
            ratio[ind] = diff_mean[ind] / diff_sd[ind].max(0.1);
        }
    }
    let mut temporary = vec![0_f32; (4 * max_width) as usize];
    let mut median = 0_f32;
    let mut madn = 0_f32;
    crate::imod::libcfshr::robuststat::rs_median(
        ratio.as_mut_ptr(),
        4 * max_width,
        temporary.as_mut_ptr(),
        &mut median,
    );
    crate::imod::libcfshr::robuststat::rs_madn(
        ratio.as_mut_ptr(),
        4 * max_width,
        median,
        temporary.as_mut_ptr(),
        &mut madn,
    );
    for index in (0..4 * max_width).rev() {
        temporary[index as usize] = (ratio[index as usize] - median) / madn;
    }
    let mut above = [-1_i32; 4];
    for edge in 0..4 {
        for width in (0..max_width).rev() {
            if temporary[(edge * max_width + width) as usize] > crit_madns {
                above[edge as usize] = width;
                break;
            }
        }
    }
    *y_low = above[0] + 1;
    *y_high = ny - 1 - (above[1] + 1);
    *x_low = above[2] + 1;
    *x_high = nx - 1 - (above[3] + 1);
    0
}
/// C++ overload 1 `CorDefSampleMeanSD` (`CorrectDefects.cpp:2551`).
pub unsafe fn cor_def_sample_mean_sd_1(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nx: i32,
    ny: i32,
    mean: *mut f32,
    sd: *mut f32,
) {
    unsafe {
        cor_def_sample_mean_sd_2(array, data_type, nx, nx, ny, mean, sd);
    }
}
/// C++ overload 2 `CorDefSampleMeanSD` (`CorrectDefects.cpp:2557`).
pub unsafe fn cor_def_sample_mean_sd_2(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    mean: *mut f32,
    sd: *mut f32,
) {
    unsafe {
        cor_def_sample_mean_sd_3(array, data_type, nxdim, nx, ny, 0, 0, nx, ny, mean, sd);
    }
}
/// C++ overload 3 `CorDefSampleMeanSD` (`CorrectDefects.cpp:2564`).
pub unsafe fn cor_def_sample_mean_sd_3(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nxdim: i32,
    nx: i32,
    ny: i32,
    ix_start: i32,
    iy_start: i32,
    nx_use: i32,
    ny_use: i32,
    mean: *mut f32,
    sd: *mut f32,
) {
    unsafe {
        let (call_type, dsize) = match data_type {
            0 => (0, 1),
            1 => (3, 2),
            6 => (2, 2),
            2 => (6, 4),
            _ => return,
        };
        let lines = crate::imod::libcfshr::b3dutil::make_line_pointers(array, nxdim, ny, dsize);
        if lines.is_null() {
            return;
        }
        let sample = (30000. / (nx * ny) as f32).min(1.);
        crate::imod::libcfshr::samplemeansd::sample_mean_sd(
            lines, call_type, nx, ny, sample, ix_start, iy_start, nx_use, ny_use, mean, sd,
        );
        libc::free(lines.cast());
    }
}
/// Matches C++ `CorDefSetupToCorrect`.
pub fn cor_def_setup_to_correct(
    nx_full: i32,
    ny_full: i32,
    defects: &mut crate::imod::clip::clip::CameraDefects,
    camera_size_x: &mut i32,
    camera_size_y: &mut i32,
    scale_defects: i32,
    set_binning: f32,
    use_binning: &mut i32,
    bin_option: Option<&str>,
) -> i32 {
    if defects.falcon_type != 0 && (nx_full > *camera_size_x || ny_full > *camera_size_y) {
        if defects.was_scaled > 1 {
            return 1;
        }
        let scaling = (nx_full / *camera_size_x) as i32;
        if *camera_size_x * scaling != nx_full
            || *camera_size_y * scaling != ny_full
            || (scaling != 2 && scaling != 4)
        {
            return 1;
        }
        cor_def_scale_defects_for_falcon(defects, scaling);
        *camera_size_x *= scaling;
        *camera_size_y *= scaling;
    }
    if defects.falcon_type != 0 && (nx_full < *camera_size_x && ny_full < *camera_size_y) {
        if defects.was_scaled < 0 {
            return 1;
        }
        let scaling = (*camera_size_x / nx_full) as i32;
        if *camera_size_x * scaling != nx_full
            || *camera_size_y * scaling != ny_full
            || (scaling != 2 && scaling != 4 && scaling != 8)
        {
            return 1;
        }
        cor_def_scale_defects_for_falcon(defects, -scaling);
        *camera_size_x /= scaling;
        *camera_size_y /= scaling;
    }
    if nx_full > *camera_size_x * 2 || ny_full > *camera_size_y * 2 {
        return 1;
    }
    if nx_full > *camera_size_x
        || ny_full > *camera_size_y
        || (scale_defects != 0 && defects.was_scaled <= 0)
    {
        if !(scale_defects != 0 && defects.was_scaled <= 0) && bin_option.is_some() {
            println!(
                "Scaling defect list up by 2 because images are larger than camera size in list\n"
            );
        }
        cor_def_scale_defects_for_k2(defects, false);
        *camera_size_x *= 2;
        *camera_size_y *= 2;
    }
    let scaled_for_k2 = defects.k2_type > 0 && defects.was_scaled > 0;
    if set_binning <= 0. {
        *use_binning = 1;
        while nx_full * (*use_binning + 1) <= *camera_size_x
            && ny_full * (*use_binning + 1) <= *camera_size_y
        {
            *use_binning += 1;
        }
        if *use_binning > 1 && bin_option.is_some() {
            let bin_text = if scaled_for_k2 {
                format!("{:.1}", *use_binning as f32 / 2.)
            } else {
                use_binning.to_string()
            };
            println!(
                "Assuming binning of {bin_text} for defect correction instead of a small subarea;\n    use the {} option to set a binning if this is incorrect.",
                bin_option.unwrap()
            );
        }
    } else {
        *use_binning = ((if scaled_for_k2 { 2. } else { 1. }) * set_binning + 0.02).round() as i32;
    }
    0
}
/// C++ `CorDefUserToRotFlipCCD` (`CorrectDefects.cpp:2450`).
pub fn cor_def_user_to_rot_flip_ccd(
    operation: i32,
    binning: i32,
    cam_size_x: &mut i32,
    cam_size_y: &mut i32,
    im_size_x: &mut i32,
    im_size_y: &mut i32,
    top: &mut i32,
    left: &mut i32,
    bottom: &mut i32,
    right: &mut i32,
) {
    if operation & 8 != 0 {
        cor_def_mirror_coords(binning, *cam_size_x, left, right);
    }
    match operation % 4 {
        1 => cor_def_rotate_coords_cw(
            binning, cam_size_x, cam_size_y, im_size_x, im_size_y, top, left, bottom, right,
        ),
        2 => {
            cor_def_mirror_coords(binning, *cam_size_y, top, bottom);
            cor_def_mirror_coords(binning, *cam_size_x, left, right);
        }
        3 => cor_def_rotate_coords_ccw(
            binning, cam_size_x, cam_size_y, im_size_x, im_size_y, top, left, bottom, right,
        ),
        _ => {}
    }
    if operation & 4 != 0 {
        cor_def_mirror_coords(binning, *cam_size_x, left, right);
    }
}
/// Matches C++ `CorDefRotFlipCCDtoUser`.
pub fn cor_def_rot_flip_ccd_to_user(
    operation: i32,
    binning: i32,
    cam_size_x: &mut i32,
    cam_size_y: &mut i32,
    im_size_x: &mut i32,
    im_size_y: &mut i32,
    top: &mut i32,
    left: &mut i32,
    bottom: &mut i32,
    right: &mut i32,
) {
    if operation & 4 != 0 {
        cor_def_mirror_coords(binning, *cam_size_x, left, right);
    }
    match operation % 4 {
        1 => cor_def_rotate_coords_ccw(
            binning, cam_size_x, cam_size_y, im_size_x, im_size_y, top, left, bottom, right,
        ),
        2 => {
            cor_def_mirror_coords(binning, *cam_size_y, top, bottom);
            cor_def_mirror_coords(binning, *cam_size_x, left, right);
        }
        3 => cor_def_rotate_coords_cw(
            binning, cam_size_x, cam_size_y, im_size_x, im_size_y, top, left, bottom, right,
        ),
        _ => {}
    }
    if operation & 8 != 0 {
        cor_def_mirror_coords(binning, *cam_size_x, left, right);
    }
}
/// Matches C++ `CorDefRotFlipCCDcoord`.
pub fn cor_def_rot_flip_ccdcoord(
    operation: i32,
    cam_size_x: i32,
    cam_size_y: i32,
    xx: &mut i32,
    yy: &mut i32,
) {
    let mut bottom = *yy + 1;
    let mut right = *xx + 1;
    let mut im_size_x = cam_size_x;
    let mut im_size_y = cam_size_y;
    if operation == 0 {
        return;
    }
    if operation % 2 != 0 {
        im_size_x = cam_size_y;
        im_size_y = cam_size_x;
    }
    let mut mutable_cam_size_x = im_size_x;
    let mut mutable_cam_size_y = im_size_y;
    cor_def_rot_flip_ccd_to_user(
        operation,
        1,
        &mut mutable_cam_size_x,
        &mut mutable_cam_size_y,
        &mut im_size_x,
        &mut im_size_y,
        yy,
        xx,
        &mut bottom,
        &mut right,
    );
    *xx = (*xx).min(right);
    *yy = (*yy).min(bottom);
}
/// Matches C++ `CorDefMirrorCoords`.
pub fn cor_def_mirror_coords(binning: i32, size: i32, start: &mut i32, end: &mut i32) {
    let temporary = size / binning - *start;
    *start = size / binning - *end;
    *end = temporary;
}
/// Matches C++ `CorDefRotateCoordsCW`.
pub fn cor_def_rotate_coords_cw(
    binning: i32,
    cam_size_x: &mut i32,
    cam_size_y: &mut i32,
    im_size_x: &mut i32,
    im_size_y: &mut i32,
    top: &mut i32,
    left: &mut i32,
    bottom: &mut i32,
    right: &mut i32,
) {
    let temporary_top = *left;
    let temporary_bottom = *right;
    *left = *cam_size_y / binning - *bottom;
    *right = *cam_size_y / binning - *top;
    *top = temporary_top;
    *bottom = temporary_bottom;
    core::mem::swap(cam_size_x, cam_size_y);
    core::mem::swap(im_size_x, im_size_y);
}
/// Matches C++ `CorDefRotateCoordsCCW`.
pub fn cor_def_rotate_coords_ccw(
    binning: i32,
    cam_size_x: &mut i32,
    cam_size_y: &mut i32,
    im_size_x: &mut i32,
    im_size_y: &mut i32,
    top: &mut i32,
    left: &mut i32,
    bottom: &mut i32,
    right: &mut i32,
) {
    let temporary_top = *cam_size_x / binning - *right;
    let temporary_bottom = *cam_size_x / binning - *left;
    *left = *top;
    *right = *bottom;
    *top = temporary_top;
    *bottom = temporary_bottom;
    core::mem::swap(cam_size_x, cam_size_y);
    core::mem::swap(im_size_x, im_size_y);
}

#[cfg(test)]
mod tests {
    use super::{
        cor_def_correct_defects, cor_def_defects_to_string, cor_def_fill_defect_array,
        cor_def_find_drift_corr_edges, cor_def_parse_defects, cor_def_parse_fei_xml,
        cor_def_read_super_gain, cor_def_sample_mean_sd_1, cor_def_setup_to_correct, correct_edge,
        correct_pixel,
    };
    use crate::imod::clip::clip::CameraDefects;

    #[test]
    fn correct_edge_copies_good_float_row_without_taper() {
        let mut image = [10_f32, 20., 30., 0., 0., 0.];
        unsafe {
            correct_edge(image.as_mut_ptr().cast(), 2, 1, 0, 3, 3, 0, 1, 3);
        }
        assert_eq!(image, [10., 20., 30., 10., 20., 30.]);
    }

    #[test]
    fn correct_edge_uses_source_double_sum_factor_before_float_store() {
        let mut image = [10_f32, 20., 30., 0., 0., 0.];
        unsafe {
            correct_edge(image.as_mut_ptr().cast(), 2, 1, 5, 3, 3, 0, 1, 3);
        }
        assert_eq!(image, [10., 20., 30., 12., 20., 28.]);
    }

    #[test]
    fn correct_pixel_preserves_source_byte_edge_path() {
        let mut image = [9_u8, 8, 7, 6];
        unsafe {
            correct_pixel(image.as_mut_ptr().cast(), 0, 2, 2, 2, 0, 0, 0, 0.);
        }
        assert_eq!(image[0], 0);
    }

    #[test]
    fn master_byte_edge_defect_matches_native_correct_defects_fixture() {
        let mut image = [1_u8, 2, 3, 4, 99, 6, 7, 8, 9];
        let defects = CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: vec![],
            bad_column_width: vec![],
            partial_bad_col: vec![],
            partial_bad_width: vec![],
            partial_bad_start_y: vec![],
            partial_bad_end_y: vec![],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![0],
            bad_pixel_y: vec![0],
            pix_use_mean: vec![],
        };
        unsafe {
            cor_def_correct_defects(&defects, image.as_mut_ptr().cast(), 0, 1, 0, 0, 3, 3);
        }
        assert_eq!(image, [0, 2, 3, 4, 99, 6, 7, 8, 9]);
    }

    #[test]
    fn master_corrects_full_column_by_neighbor_interpolation() {
        let mut image = [
            1_f32, 2., 99., 4., 5., 1., 2., 99., 4., 5., 1., 2., 99., 4., 5.,
        ];
        let mut list = CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: Vec::new(),
            bad_column_width: Vec::new(),
            partial_bad_col: Vec::new(),
            partial_bad_width: Vec::new(),
            partial_bad_start_y: Vec::new(),
            partial_bad_end_y: Vec::new(),
            bad_row_start: Vec::new(),
            bad_row_height: Vec::new(),
            partial_bad_row: Vec::new(),
            partial_bad_height: Vec::new(),
            partial_bad_start_x: Vec::new(),
            partial_bad_end_x: Vec::new(),
            bad_pixel_x: Vec::new(),
            bad_pixel_y: Vec::new(),
            pix_use_mean: Vec::new(),
        };
        list.bad_column_start.push(2);
        list.bad_column_width.push(1);
        unsafe { cor_def_correct_defects(&list, image.as_mut_ptr().cast(), 2, 1, 0, 0, 3, 5) };
        assert_eq!(image[2], 3.);
        assert_eq!(image[7], 3.);
        assert_eq!(image[12], 3.);
    }

    #[test]
    fn fei_xml_uses_translated_mxml_node_list() {
        let mut defects = CameraDefects {
            was_scaled: 9,
            rotation_flip: 9,
            k2_type: 9,
            falcon_type: 9,
            usable_top: 9,
            usable_left: 9,
            usable_bottom: 9,
            usable_right: 9,
            num_avg_super_res: 9,
            bad_column_start: vec![],
            bad_column_width: vec![],
            partial_bad_col: vec![],
            partial_bad_width: vec![],
            partial_bad_start_y: vec![],
            partial_bad_end_y: vec![],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![],
            bad_pixel_y: vec![],
            pix_use_mean: vec![],
        };
        let xml = c"<defects><row>3,4</row><col>6,7</col><area>1,2,4,3</area><point>8,9</point></defects>";
        assert_eq!(
            unsafe { cor_def_parse_fei_xml(xml.as_ptr(), &mut defects, 56) },
            0
        );
        assert_eq!(defects.falcon_type, 1);
        assert_eq!(defects.num_avg_super_res, 4);
        assert_eq!(
            (&defects.bad_row_start[..], &defects.bad_row_height[..]),
            (&[0_u16][..], &[10_i16][..])
        );
        assert_eq!(
            (&defects.bad_column_start[..], &defects.bad_column_width[..]),
            (&[1_u16][..], &[12_i16][..])
        );
        assert_eq!(
            (
                &defects.partial_bad_row[..],
                &defects.partial_bad_height[..]
            ),
            (&[2_u16][..], &[2_i16][..])
        );
        assert_eq!(
            (&defects.bad_pixel_x[..], &defects.bad_pixel_y[..]),
            (&[8_u16][..], &[9_u16][..])
        );
    }

    #[test]
    fn super_gain_text_file_preserves_source_grid_layout() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-supergain-{}.txt", std::process::id()));
        let values = (1..=20)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(&path, format!("1 4\n1 2 3 1 4 5\n{values}\n")).unwrap();
        let (mut nx, mut xs, mut xd, mut ny, mut ys, mut yd) = (0, 0, 0, 0, 0, 0);
        let mut biases = Vec::new();
        assert_eq!(
            cor_def_read_super_gain(
                path.to_str().unwrap(),
                4,
                &mut biases,
                &mut nx,
                &mut xs,
                &mut xd,
                &mut ny,
                &mut ys,
                &mut yd
            ),
            0
        );
        assert_eq!((nx, xs, xd, ny, ys, yd), (1, 2, 3, 1, 4, 5));
        assert_eq!(
            biases,
            vec![(1..=16).map(|value| value as f32).collect::<Vec<_>>()]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn super_gain_source_scan_return_codes_are_preserved() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-supergain-incomplete-{}.txt",
            std::process::id()
        ));
        let (mut nx, mut xs, mut xd, mut ny, mut ys, mut yd) = (0, 0, 0, 0, 0, 0);
        let mut biases = Vec::new();
        std::fs::write(&path, "").unwrap();
        assert_eq!(
            cor_def_read_super_gain(
                path.to_str().unwrap(),
                4,
                &mut biases,
                &mut nx,
                &mut xs,
                &mut xd,
                &mut ny,
                &mut ys,
                &mut yd
            ),
            2
        );
        std::fs::write(&path, "1\n").unwrap();
        assert_eq!(
            cor_def_read_super_gain(
                path.to_str().unwrap(),
                4,
                &mut biases,
                &mut nx,
                &mut xs,
                &mut xd,
                &mut ny,
                &mut ys,
                &mut yd
            ),
            4
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn defect_text_uses_source_pip_value_separators() {
        let mut defects = CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: vec![],
            bad_column_width: vec![],
            partial_bad_col: vec![],
            partial_bad_width: vec![],
            partial_bad_start_y: vec![],
            partial_bad_end_y: vec![],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![],
            bad_pixel_y: vec![],
            pix_use_mean: vec![],
        };
        let (mut cx, mut cy) = (0, 0);
        assert_eq!(
            cor_def_parse_defects(
                "CameraSize\t4\nCameraSizeY 4\nBadColumns 1, 3\n",
                true,
                &mut defects,
                &mut cx,
                &mut cy
            ),
            0
        );
        assert_eq!((cx, cy), (4, 4));
        assert_eq!(
            (&defects.bad_column_start[..], &defects.bad_column_width[..]),
            (&[1_u16, 3_u16][..], &[1_i16, 1_i16][..])
        );
        defects.bad_column_start = vec![u16::MAX];
        defects.bad_column_width = vec![1];
        let mut text = String::new();
        cor_def_defects_to_string(&defects, &mut text, 4, 4);
        assert!(text.contains("BadColumns 65535\n"));
        assert_eq!(
            cor_def_parse_defects("CameraSizeX /\n", true, &mut defects, &mut cx, &mut cy),
            2
        );
        let long_line = format!("{} CameraSizeX 99\n", "x".repeat(511));
        assert_eq!(
            cor_def_parse_defects(&long_line, true, &mut defects, &mut cx, &mut cy),
            0
        );
        assert_eq!((cx, cy), (0, 0));
    }

    #[test]
    fn fill_defect_array_keeps_source_partial_falcon_padding() {
        let defects = CameraDefects {
            was_scaled: 1,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 1,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 1,
            bad_column_start: vec![4],
            bad_column_width: vec![1],
            partial_bad_col: vec![4],
            partial_bad_width: vec![1],
            partial_bad_start_y: vec![2],
            partial_bad_end_y: vec![3],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![],
            bad_pixel_y: vec![],
            pix_use_mean: vec![],
        };
        let mut map = [0_u8; 64];
        assert_eq!(
            unsafe { cor_def_fill_defect_array(&defects, 8, 8, map.as_mut_ptr(), 8, 8, true) },
            0
        );
        assert_eq!(map[4 + 2 * 8], 1);
        assert_eq!(map[4 + 3 * 8], 1);
        assert_eq!(map[2 + 2 * 8], 254);
        assert_eq!(map[2 + 3 * 8], 254);
        assert_eq!(map[5 + 2 * 8], 254);
        assert_eq!(map[5 + 3 * 8], 254);
    }

    #[test]
    fn sample_mean_sd_uses_source_slice_type_mapping() {
        let mut image = [1_f32, 2., 3., 4., 5., 6., 7., 8., 9.];
        let (mut mean, mut sd) = (0_f32, 0_f32);
        unsafe {
            cor_def_sample_mean_sd_1(image.as_mut_ptr().cast(), 2, 3, 3, &mut mean, &mut sd);
        }
        assert_eq!(mean, 5.);
        assert!((sd - 2.738_613).abs() < 1.0e-5);
    }

    #[test]
    fn setup_to_correct_deduces_unscaled_camera_binning() {
        let mut defects = CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: vec![],
            bad_column_width: vec![],
            partial_bad_col: vec![],
            partial_bad_width: vec![],
            partial_bad_start_y: vec![],
            partial_bad_end_y: vec![],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![],
            bad_pixel_y: vec![],
            pix_use_mean: vec![],
        };
        let (mut camera_x, mut camera_y, mut binning) = (8, 8, 0);
        assert_eq!(
            cor_def_setup_to_correct(
                4,
                4,
                &mut defects,
                &mut camera_x,
                &mut camera_y,
                0,
                0.,
                &mut binning,
                None
            ),
            0
        );
        assert_eq!((camera_x, camera_y, binning), (8, 8, 2));
    }

    #[test]
    fn drift_edges_uniform_image_has_full_good_limits() {
        let image = [4_f32; 64];
        let (mut xl, mut xh, mut yl, mut yh) = (-1, -1, -1, -1);
        assert_eq!(
            unsafe {
                cor_def_find_drift_corr_edges(
                    image.as_ptr().cast(),
                    2,
                    8,
                    8,
                    4,
                    2,
                    3.,
                    &mut xl,
                    &mut xh,
                    &mut yl,
                    &mut yh,
                )
            },
            0
        );
        assert_eq!((xl, xh, yl, yh), (0, 7, 0, 7));
    }
}

#[cfg(test)]
mod coordinate_tests {
    use super::{cor_def_mirror_coords, cor_def_rotate_coords_ccw, cor_def_rotate_coords_cw};

    #[test]
    fn mirror_coords_matches_exclusive_end_convention() {
        let mut start = 2;
        let mut end = 7;
        cor_def_mirror_coords(1, 10, &mut start, &mut end);
        assert_eq!((start, end), (3, 8));
    }

    #[test]
    fn rotations_are_inverses() {
        let (mut cx, mut cy, mut ix, mut iy) = (10, 20, 10, 20);
        let (mut top, mut left, mut bottom, mut right) = (2, 3, 7, 9);
        cor_def_rotate_coords_cw(
            1,
            &mut cx,
            &mut cy,
            &mut ix,
            &mut iy,
            &mut top,
            &mut left,
            &mut bottom,
            &mut right,
        );
        cor_def_rotate_coords_ccw(
            1,
            &mut cx,
            &mut cy,
            &mut ix,
            &mut iy,
            &mut top,
            &mut left,
            &mut bottom,
            &mut right,
        );
        assert_eq!(
            (cx, cy, ix, iy, top, left, bottom, right),
            (10, 20, 10, 20, 2, 3, 7, 9)
        );
    }
}
