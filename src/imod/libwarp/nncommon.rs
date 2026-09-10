//! Translation of `IMOD/libwarp/nncommon.c`.
#![allow(dead_code)]

use crate::imod::libwarp::delaunay::Circle;
use crate::imod::libwarp::nn::{NN_VERBOSE, Point};
use core::ffi::c_char;

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
    static mut stdin: *mut libc::FILE;
}

/// Original C variadic `nn_quit` (`nncommon.c:54`).
///
/// The C ABI's variadic formatter cannot be defined on stable Rust.  Direct
/// source call sites retain their fixed `"%s: %s\\n"` formatting inline.
pub unsafe fn nn_quit(format: *const c_char) -> ! {
    unsafe {
        libc::fflush(core::ptr::null_mut());
        libc::fputs(c"  error: libnn: ".as_ptr(), stderr);
        libc::fputs(format, stderr);
        libc::exit(1)
    }
}

/// Original `circle_contains` (`nncommon.c:69`).
pub unsafe fn circle_contains(circle: *mut Circle, point: *mut Point) -> i32 {
    unsafe { (((*circle).x - (*point).x).hypot((*circle).y - (*point).y) <= (*circle).r) as i32 }
}

/// Original `points_thingrid` (`nncommon.c:85`).
pub unsafe fn points_thingrid(
    point_count: *mut i32,
    points_pointer: *mut *mut Point,
    nx: i32,
    ny: i32,
) {
    unsafe {
        let point_total = *point_count;
        let points = *points_pointer;
        let mut xmin = f64::MAX;
        let mut xmax = -f64::MAX;
        let mut ymin = f64::MAX;
        let mut ymax = -f64::MAX;
        let nxy = nx * ny;
        let sum_x = libc::calloc(nxy as usize, core::mem::size_of::<f64>()).cast::<f64>();
        let sum_y = libc::calloc(nxy as usize, core::mem::size_of::<f64>()).cast::<f64>();
        let sum_z = libc::calloc(nxy as usize, core::mem::size_of::<f64>()).cast::<f64>();
        let count = libc::calloc(nxy as usize, core::mem::size_of::<i32>()).cast::<i32>();
        if NN_VERBOSE != 0 {
            libc::fprintf(stderr, c"thinned: %d points -> ".as_ptr(), *point_count);
        }
        if nx < 1 || ny < 1 {
            libc::free(points.cast());
            *points_pointer = core::ptr::null_mut();
            *point_count = 0;
            if NN_VERBOSE != 0 {
                libc::fprintf(stderr, c"0 points".as_ptr());
            }
            return;
        }
        for index in 0..point_total {
            let point = points.add(index as usize);
            if (*point).x < xmin {
                xmin = (*point).x;
            }
            if (*point).x > xmax {
                xmax = (*point).x;
            }
            if (*point).y < ymin {
                ymin = (*point).y;
            }
            if (*point).y > ymax {
                ymax = (*point).y;
            }
        }
        let stepx = if nx > 1 {
            (xmax - xmin) / nx as f64
        } else {
            0.0
        };
        let stepy = if ny > 1 {
            (ymax - ymin) / ny as f64
        } else {
            0.0
        };
        for index in 0..point_total {
            let point = points.add(index as usize);
            let mut ix = if nx == 1 {
                0
            } else {
                let grid_x = ((*point).x - xmin) / stepx;
                if (grid_x.round() - grid_x).abs() < 1.0e-15 {
                    grid_x.round() as i32
                } else {
                    grid_x.floor() as i32
                }
            };
            let mut iy = if ny == 1 {
                0
            } else {
                let grid_y = ((*point).y - ymin) / stepy;
                if (grid_y.round() - grid_y).abs() < 1.0e-15 {
                    grid_y.round() as i32
                } else {
                    grid_y.floor() as i32
                }
            };
            if ix == nx {
                ix -= 1;
            }
            if iy == ny {
                iy -= 1;
            }
            let cell = ix + iy * nx;
            *sum_x.add(cell as usize) += (*point).x;
            *sum_y.add(cell as usize) += (*point).y;
            *sum_z.add(cell as usize) += (*point).z;
            *count.add(cell as usize) += 1;
        }
        let mut new_count = 0;
        for iy in 0..ny {
            for ix in 0..nx {
                if *count.add((ix + iy * nx) as usize) > 0 {
                    new_count += 1;
                }
            }
        }
        let new_points =
            libc::malloc(new_count as usize * core::mem::size_of::<Point>()).cast::<Point>();
        let mut output_index = 0;
        for iy in 0..ny {
            for ix in 0..nx {
                let cell = ix + iy * nx;
                let cell_count = *count.add(cell as usize);
                if cell_count > 0 {
                    let point = new_points.add(output_index as usize);
                    (*point).x = *sum_x.add(cell as usize) / cell_count as f64;
                    (*point).y = *sum_y.add(cell as usize) / cell_count as f64;
                    (*point).z = *sum_z.add(cell as usize) / cell_count as f64;
                    output_index += 1;
                }
            }
        }
        libc::free(sum_x.cast());
        libc::free(sum_y.cast());
        libc::free(sum_z.cast());
        libc::free(count.cast());
        libc::free(points.cast());
        *points_pointer = new_points;
        *point_count = new_count;
        if NN_VERBOSE != 0 {
            libc::fprintf(stderr, c"%d points\n".as_ptr(), new_count);
        }
    }
}

/// Original `points_thinlin` (`nncommon.c:217`).
pub unsafe fn points_thinlin(
    input_count: *mut i32,
    input_points: *mut *mut Point,
    maximum_radius: f64,
) {
    unsafe {
        let mut output_count: i32 = 0;
        let mut allocated: i32 = 1024;
        let mut output =
            libc::malloc(allocated as usize * core::mem::size_of::<Point>()).cast::<Point>();
        let mut count: f64 = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_z = 0.0;
        let mut sum_radius = 0.0;
        let mut previous: *mut Point = core::ptr::null_mut();
        for index in 0..*input_count {
            let point = (*input_points).add(index as usize);
            if (*point).x.is_nan() || (*point).y.is_nan() || (*point).z.is_nan() {
                if !previous.is_null() {
                    if output_count == allocated {
                        allocated *= 2;
                        output = libc::realloc(
                            output.cast(),
                            allocated as usize * core::mem::size_of::<Point>(),
                        )
                        .cast();
                    }
                    (*output.add(output_count as usize)).x = sum_x / count;
                    (*output.add(output_count as usize)).y = sum_y / count;
                    (*output.add(output_count as usize)).z = sum_z / count;
                    output_count += 1;
                    previous = core::ptr::null_mut();
                }
                continue;
            }
            if previous.is_null() {
                sum_x = (*point).x;
                sum_y = (*point).y;
                sum_z = (*point).z;
                sum_radius = 0.0;
                count = 1.0;
                previous = point;
                continue;
            }
            let distance = ((*point).x - (*previous).x).hypot((*point).y - (*previous).y);
            if sum_radius + distance > maximum_radius {
                if output_count == allocated {
                    allocated *= 2;
                    output = libc::realloc(
                        output.cast(),
                        allocated as usize * core::mem::size_of::<Point>(),
                    )
                    .cast();
                }
                (*output.add(output_count as usize)).x = sum_x / count;
                (*output.add(output_count as usize)).y = sum_y / count;
                (*output.add(output_count as usize)).z = sum_z / count;
                output_count += 1;
                previous = core::ptr::null_mut();
            } else {
                sum_x += (*point).x;
                sum_y += (*point).y;
                sum_z += (*point).z;
                sum_radius += distance;
                count += 1.0;
                previous = point;
            }
        }
        libc::free((*input_points).cast());
        *input_points = libc::realloc(
            output.cast(),
            output_count as usize * core::mem::size_of::<Point>(),
        )
        .cast();
        *input_count = output_count;
    }
}

/// Original `points_getrange` (`nncommon.c:313`).
pub unsafe fn points_getrange(
    count: i32,
    points: *mut Point,
    zoom: f64,
    xmin: *mut f64,
    xmax: *mut f64,
    ymin: *mut f64,
    ymax: *mut f64,
) {
    unsafe {
        let mut xmin = xmin;
        let mut xmax = xmax;
        let mut ymin = ymin;
        let mut ymax = ymax;
        if !xmin.is_null() {
            if (*xmin).is_nan() {
                *xmin = f64::MAX;
            } else {
                xmin = core::ptr::null_mut();
            }
        }
        if !xmax.is_null() {
            if (*xmax).is_nan() {
                *xmax = -f64::MAX;
            } else {
                xmax = core::ptr::null_mut();
            }
        }
        if !ymin.is_null() {
            if (*ymin).is_nan() {
                *ymin = f64::MAX;
            } else {
                ymin = core::ptr::null_mut();
            }
        }
        if !ymax.is_null() {
            if (*ymax).is_nan() {
                *ymax = -f64::MAX;
            } else {
                ymax = core::ptr::null_mut();
            }
        }
        for index in 0..count {
            let point = points.add(index as usize);
            if !xmin.is_null() && (*point).x < *xmin {
                *xmin = (*point).x;
            }
            if !xmax.is_null() && (*point).x > *xmax {
                *xmax = (*point).x;
            }
            if !ymin.is_null() && (*point).y < *ymin {
                *ymin = (*point).y;
            }
            if !ymax.is_null() && (*point).y > *ymax {
                *ymax = (*point).y;
            }
        }
        if zoom.is_nan() || zoom <= 0.0 || zoom == 1.0 {
            return;
        }
        if !xmin.is_null() && !xmax.is_null() {
            let half_difference = (*xmax - *xmin) / 2.0;
            let average = (*xmax + *xmin) / 2.0;
            *xmin = average - half_difference * zoom;
            *xmax = average + half_difference * zoom;
        }
        if !ymin.is_null() && !ymax.is_null() {
            let half_difference = (*ymax - *ymin) / 2.0;
            let average = (*ymax + *ymin) / 2.0;
            *ymin = average - half_difference * zoom;
            *ymax = average + half_difference * zoom;
        }
    }
}

/// Original `points_generate` (`nncommon.c:387`).
pub unsafe fn points_generate(
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: i32,
    ny: i32,
    output_count: *mut i32,
    output_points: *mut *mut Point,
) {
    unsafe {
        if nx < 1 || ny < 1 {
            *output_points = core::ptr::null_mut();
            *output_count = 0;
            return;
        }
        *output_count = nx * ny;
        *output_points =
            libc::malloc(*output_count as usize * core::mem::size_of::<Point>()).cast();
        let step_x = if nx > 1 {
            (xmax - xmin) / (nx - 1) as f64
        } else {
            0.0
        };
        let step_y = if ny > 1 {
            (ymax - ymin) / (ny - 1) as f64
        } else {
            0.0
        };
        let x0 = if nx > 1 { xmin } else { (xmin + xmax) / 2.0 };
        let mut y = if ny > 1 { ymin } else { (ymin + ymax) / 2.0 };
        let mut index = 0;
        for _ in 0..ny {
            let mut x = x0;
            for _ in 0..nx {
                let point = (*output_points).add(index as usize);
                (*point).x = x;
                (*point).y = y;
                x += step_x;
                index += 1;
            }
            y += step_y;
        }
    }
}

/// Original `str2double` (`nncommon.c:422`).
pub unsafe fn str2double(token: *mut c_char, value: *mut f64) -> i32 {
    unsafe {
        if token.is_null() {
            *value = f64::NAN;
            return 0;
        }
        let mut end: *mut c_char = core::ptr::null_mut();
        *value = libc::strtod(token, &mut end);
        if end == token {
            *value = f64::NAN;
            return 0;
        }
        1
    }
}

/// Original `points_read` (`nncommon.c:448`).
pub unsafe fn points_read(
    filename: *mut c_char,
    dimension: i32,
    count: *mut i32,
    points: *mut *mut Point,
) {
    unsafe {
        let mut file: *mut libc::FILE = core::ptr::null_mut();
        let mut allocated = 1024_i32;
        let mut buffer = [0_i8; 1024];
        let separators = c" ,;\t";
        if dimension < 2 || dimension > 3 {
            *count = 0;
            *points = core::ptr::null_mut();
            return;
        }
        if filename.is_null() {
            file = stdin;
        } else if libc::strcmp(filename, c"stdin".as_ptr()) == 0
            || libc::strcmp(filename, c"-".as_ptr()) == 0
        {
            file = stdin;
        } else {
            file = libc::fopen(filename, c"r".as_ptr());
            if file.is_null() {
                libc::fflush(core::ptr::null_mut());
                libc::fprintf(
                    stderr,
                    c"  error: libnn: %s: %s\n".as_ptr(),
                    filename,
                    libc::strerror(*libc::__errno_location()),
                );
                libc::exit(1);
            }
        }
        *points = libc::malloc(allocated as usize * core::mem::size_of::<Point>()).cast();
        *count = 0;
        while !libc::fgets(buffer.as_mut_ptr(), 1024, file).is_null() {
            if *count == allocated {
                allocated *= 2;
                *points = libc::realloc(
                    (*points).cast(),
                    allocated as usize * core::mem::size_of::<Point>(),
                )
                .cast();
            }
            if buffer[0] == b'#' as i8 {
                continue;
            }
            let mut token = libc::strtok(buffer.as_mut_ptr(), separators.as_ptr());
            if token.is_null() {
                continue;
            }
            let point = (*points).add(*count as usize);
            if str2double(token, &mut (*point).x) == 0 {
                continue;
            }
            token = libc::strtok(core::ptr::null_mut(), separators.as_ptr());
            if token.is_null() || str2double(token, &mut (*point).y) == 0 {
                continue;
            }
            if dimension == 2 {
                (*point).z = f64::NAN;
            } else {
                token = libc::strtok(core::ptr::null_mut(), separators.as_ptr());
                if token.is_null() || str2double(token, &mut (*point).z) == 0 {
                    continue;
                }
            }
            *count += 1;
        }
        if *count == 0 {
            libc::free((*points).cast());
            *points = core::ptr::null_mut();
        } else {
            *points = libc::realloc(
                (*points).cast(),
                *count as usize * core::mem::size_of::<Point>(),
            )
            .cast();
        }
        if file != stdin && libc::fclose(file) != 0 {
            libc::fflush(core::ptr::null_mut());
            libc::fprintf(
                stderr,
                c"  error: libnn: %s: %s\n".as_ptr(),
                filename,
                libc::strerror(*libc::__errno_location()),
            );
            libc::exit(1);
        }
    }
}

/// Original `points_scaletosquare` (`nncommon.c:525`).
pub unsafe fn points_scaletosquare(count: i32, points: *mut Point) -> f64 {
    unsafe {
        if count <= 0 {
            return f64::NAN;
        }
        let mut xmin = (*points).x;
        let mut xmax = (*points).x;
        let mut ymin = (*points).y;
        let mut ymax = (*points).y;
        for index in 1..count {
            let point = points.add(index as usize);
            if (*point).x < xmin {
                xmin = (*point).x;
            } else if (*point).x > xmax {
                xmax = (*point).x;
            }
            if (*point).y < ymin {
                ymin = (*point).y;
            } else if (*point).y > ymax {
                ymax = (*point).y;
            }
        }
        if xmin == xmax || ymin == ymax {
            return f64::NAN;
        }
        let multiplier = (ymax - ymin) / (xmax - xmin);
        for index in 0..count {
            (*points.add(index as usize)).y /= multiplier;
        }
        multiplier
    }
}

/// Original `points_scale` (`nncommon.c:567`).
pub unsafe fn points_scale(count: i32, points: *mut Point, multiplier: f64) {
    unsafe {
        for index in 0..count {
            (*points.add(index as usize)).y /= multiplier;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_grid_range_circle_and_scale_paths_are_preserved() {
        unsafe {
            let mut count = 0;
            let mut points = core::ptr::null_mut();
            points_generate(0.0, 2.0, 4.0, 6.0, 2, 2, &mut count, &mut points);
            assert_eq!(count, 4);
            assert_eq!(((*points).x, (*points.add(3)).y), (0.0, 6.0));
            let mut xmin = f64::NAN;
            let mut xmax = f64::NAN;
            let mut ymin = f64::NAN;
            let mut ymax = f64::NAN;
            points_getrange(
                count, points, 2.0, &mut xmin, &mut xmax, &mut ymin, &mut ymax,
            );
            assert_eq!((xmin, xmax, ymin, ymax), (-1.0, 3.0, 3.0, 7.0));
            let circle = Circle {
                x: 1.0,
                y: 1.0,
                r: 1.0,
            };
            assert_eq!(
                circle_contains((&circle as *const Circle).cast_mut(), points),
                0
            );
            let mut square_points = [
                Point {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Point {
                    x: 2.0,
                    y: 4.0,
                    z: 0.0,
                },
            ];
            assert_eq!(points_scaletosquare(2, square_points.as_mut_ptr()), 2.0);
            assert_eq!(square_points[1].y, 2.0);
            libc::free(points.cast());
        }
    }

    #[test]
    fn reads_the_vendored_imod_columnar_fixture_with_source_token_rules() {
        unsafe {
            let filename = c"IMOD/Etomo/uitestData/BB/BBafid_simple-align.xyz";
            let mut count = -1;
            let mut points = core::ptr::null_mut();
            points_read(filename.as_ptr().cast_mut(), 3, &mut count, &mut points);
            assert_eq!(count, 23);
            assert_eq!(
                ((*points).x, (*points).y, (*points).z),
                (1.0, 244.53, 242.58)
            );
            libc::free(points.cast());
        }
    }
}
