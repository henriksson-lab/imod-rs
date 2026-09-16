//! Translation of `IMOD/mrc/fakevolume.cpp`.

use crate::imod::libcfshr::b3dutil::{ImodFile, set_float_output_for_entered_mode};
use crate::imod::libcfshr::islice::{slice_create, slice_mode_if_real, slice_put_val};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_float, pip_get_in_out_file, pip_get_integer, pip_get_three_floats,
    pip_get_three_integers, pip_get_two_floats, pip_number_of_entries, pip_read_or_parse_options,
};
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_head_label, mrc_head_new, mrc_head_write, mrc_write_slice,
};

const LIMOBJ: usize = 200;

#[derive(Clone, Copy)]
struct Sphere {
    center: [f32; 3],
    kind: i32,
    radius: [f32; 2],
    density: [f32; 3],
}

#[derive(Clone, Copy)]
struct Cylinder {
    start: [f32; 3],
    end: [f32; 3],
    truncated: bool,
    radius: [f32; 2],
    density: [f32; 2],
}

/// `point_to_line` in `fakevolume.cpp`.
pub fn point_to_line(aa: f32, bb: f32, cc: f32, dd: f32, x: f32, y: f32, z: f32) -> f32 {
    (aa * x + cc * y + z - aa * bb - cc * dd) / (1. + aa * aa + cc * cc)
}

/// `pointLineSegDist` in `fakevolume.cpp`.
pub fn point_line_seg_dist(start: [f32; 3], end: [f32; 3], point: [f32; 3]) -> (f32, f32) {
    let direction = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let denominator = direction.iter().map(|value| value * value).sum::<f32>();
    let t = if denominator == 0. {
        0.
    } else {
        ((direction[0] * (point[0] - start[0])
            + direction[1] * (point[1] - start[1])
            + direction[2] * (point[2] - start[2]))
            / denominator)
            .clamp(0., 1.)
    };
    let distance_squared = (0..3)
        .map(|index| (point[index] - direction[index] * t - start[index]).powi(2))
        .sum();
    (t, distance_squared)
}

/// C `main` in `fakevolume.cpp`.
pub fn fakevolume(arguments: &[String]) -> i32 {
    const OPTIONS: [&[u8]; 16] = [
        b"output:OutputFile:FN:",
        b"size:VolumeSizeInXYZ:IT:",
        b"offsets:OffsetsInXYZ:FT:",
        b"back:BackgroundDensity:F:",
        b"stype:SphereType:IM:",
        b"scen:SphereCenterInXYZ:FTM:",
        b"sradii:SphereRadii:FPM:",
        b"sdens:SphereDensities:FPM:",
        b"trunc:CylinderIsTruncated:IM:",
        b"cstart:CylinderStartInXYZ:FTM:",
        b"cend:CylinderEndInXYZ:FTM:",
        b"cradii:CylinderRadii:FPM:",
        b"cdens:CylinderDensities:FPM:",
        b"mode:ModeToOutput:I:",
        b"param:ParameterFile:PF:",
        b"help:usage:B:",
    ];
    let argv = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    let program = argv.first().map_or(b"fakevolume".as_slice(), Vec::as_slice);
    let (mut optional, mut positional) = (0, 0);
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &OPTIONS,
        OPTIONS.len() as i32,
        program,
        3,
        1,
        1,
        &mut optional,
        &mut positional,
        None,
    );
    let mut output = Vec::new();
    if pip_get_in_out_file(b"OutputFile", 0, &mut output) != 0 {
        pip_done();
        return 1;
    }
    let output = String::from_utf8_lossy(&output).into_owned();
    let (mut nx, mut ny, mut nz) = (0, 0, 0);
    if pip_get_three_integers(b"VolumeSizeInXYZ", &mut nx, &mut ny, &mut nz) != 0
        || [nx, ny, nz].iter().any(|&value| value < 1)
        || (nx as f64 * ny as f64 > 2_000_000_000.)
    {
        pip_done();
        return 1;
    }
    let (mut offset_x, mut offset_y, mut offset_z) = (0., 0., 0.);
    let _ = pip_get_three_floats(b"OffsetsInXYZ", &mut offset_x, &mut offset_y, &mut offset_z);
    let offsets = [offset_x, offset_y, offset_z];
    let dimensions = [nx, ny, nz];
    let mut background = 0.;
    if pip_get_float(b"BackgroundDensity", &mut background) != 0 {
        pip_done();
        return 1;
    }
    let mut mode = 2;
    let _ = pip_get_integer(b"ModeToOutput", &mut mode);
    if slice_mode_if_real(mode) < 0 {
        pip_done();
        return 1;
    }
    mode = set_float_output_for_entered_mode(mode);

    let mut count = 0;
    let _ = pip_number_of_entries(b"SphereCenterInXYZ", &mut count);
    if !(0..=LIMOBJ as i32).contains(&count) {
        pip_done();
        return 1;
    }
    let sphere_count = count as usize;
    let mut type_count = 0;
    let mut radius_count = 0;
    let mut density_count = 0;
    let _ = pip_number_of_entries(b"SphereType", &mut type_count);
    let _ = pip_number_of_entries(b"SphereRadii", &mut radius_count);
    let _ = pip_number_of_entries(b"SphereDensities", &mut density_count);
    if sphere_count == 0 && (type_count != 0 || radius_count != 0 || density_count != 0) {
        pip_done();
        return 1;
    }
    if sphere_count > 0
        && ([type_count, radius_count, density_count]
            .iter()
            .any(|&n| n != 1 && n != sphere_count as i32))
    {
        pip_done();
        return 1;
    }
    let mut spheres = vec![
        Sphere {
            center: [0.; 3],
            kind: 0,
            radius: [0.; 2],
            density: [0.; 3]
        };
        sphere_count
    ];
    for sphere in &mut spheres {
        let (mut x, mut y, mut z) = (0., 0., 0.);
        if pip_get_three_floats(b"SphereCenterInXYZ", &mut x, &mut y, &mut z) != 0 {
            pip_done();
            return 1;
        }
        sphere.center = [x, y, z];
        for index in 0..3 {
            sphere.center[index] += offsets[index];
        }
    }
    for index in 0..sphere_count {
        if index < type_count as usize
            && pip_get_integer(b"SphereType", &mut spheres[index].kind) != 0
        {
            pip_done();
            return 1;
        }
        if index >= type_count as usize {
            spheres[index].kind = spheres[0].kind;
        }
        if index > 0
            && index < type_count as usize
            && spheres[index].kind != spheres[index - 1].kind
            && (type_count != sphere_count as i32
                || radius_count != sphere_count as i32
                || density_count != sphere_count as i32)
        {
            pip_done();
            return 1;
        }
    }
    for index in 0..sphere_count {
        if index < radius_count as usize {
            if spheres[index].kind <= 1 {
                spheres[index].radius[0] = 0.;
                if pip_get_float(b"SphereRadii", &mut spheres[index].radius[1]) != 0 {
                    pip_done();
                    return 1;
                }
            } else {
                let (mut inner, mut outer) = (0., 0.);
                if pip_get_two_floats(b"SphereRadii", &mut inner, &mut outer) != 0 {
                    pip_done();
                    return 1;
                }
                spheres[index].radius = [inner, outer];
            }
        } else {
            spheres[index].radius = spheres[0].radius;
        }
        if index < density_count as usize {
            if spheres[index].kind <= 1 {
                if pip_get_float(b"SphereDensities", &mut spheres[index].density[1]) != 0 {
                    pip_done();
                    return 1;
                };
                spheres[index].density[0] = spheres[index].density[1];
                spheres[index].density[2] = spheres[index].density[1];
            } else {
                let (mut inner, mut outer) = (0., 0.);
                if pip_get_two_floats(b"SphereDensities", &mut inner, &mut outer) != 0 {
                    pip_done();
                    return 1;
                }
                spheres[index].density[0] = inner;
                spheres[index].density[2] = outer;
                spheres[index].density[1] = if spheres[index].kind == 2 {
                    spheres[index].density[0]
                } else {
                    spheres[index].density[2]
                };
            }
        } else {
            spheres[index].density = spheres[0].density;
        }
    }

    let mut cylinder_count = 0;
    let mut end_count = 0;
    let _ = pip_number_of_entries(b"CylinderStartInXYZ", &mut cylinder_count);
    let _ = pip_number_of_entries(b"CylinderEndInXYZ", &mut end_count);
    if cylinder_count != end_count || !(0..=LIMOBJ as i32).contains(&cylinder_count) {
        pip_done();
        return 1;
    }
    let mut trunc_count = 0;
    radius_count = 0;
    density_count = 0;
    let _ = pip_number_of_entries(b"CylinderIsTruncated", &mut trunc_count);
    let _ = pip_number_of_entries(b"CylinderRadii", &mut radius_count);
    let _ = pip_number_of_entries(b"CylinderDensities", &mut density_count);
    if cylinder_count == 0 && (trunc_count != 0 || radius_count != 0 || density_count != 0) {
        pip_done();
        return 1;
    }
    let mut cylinders = vec![
        Cylinder {
            start: [0.; 3],
            end: [0.; 3],
            truncated: false,
            radius: [0.; 2],
            density: [0.; 2]
        };
        cylinder_count as usize
    ];
    for index in 0..cylinders.len() {
        let (mut x1, mut y1, mut z1, mut x2, mut y2, mut z2) = (0., 0., 0., 0., 0., 0.);
        if pip_get_three_floats(b"CylinderStartInXYZ", &mut x1, &mut y1, &mut z1) != 0
            || pip_get_three_floats(b"CylinderEndInXYZ", &mut x2, &mut y2, &mut z2) != 0
        {
            pip_done();
            return 1;
        }
        let first = cylinders.first().copied().unwrap_or(cylinders[index]);
        let cylinder = &mut cylinders[index];
        cylinder.start = [x1, y1, z1];
        cylinder.end = [x2, y2, z2];
        if index < trunc_count as usize {
            let mut value = 0;
            if pip_get_integer(b"CylinderIsTruncated", &mut value) != 0 {
                pip_done();
                return 1;
            };
            cylinder.truncated = value != 0;
        } else if index > 0 {
            cylinder.truncated = first.truncated;
        }
        if index < radius_count as usize {
            let (mut inner, mut outer) = (0., 0.);
            if pip_get_two_floats(b"CylinderRadii", &mut inner, &mut outer) != 0 {
                pip_done();
                return 1;
            }
            cylinder.radius = [inner, outer];
        } else if index > 0 {
            cylinder.radius = first.radius;
        }
        if index < density_count as usize {
            let (mut inner, mut outer) = (0., 0.);
            if pip_get_two_floats(b"CylinderDensities", &mut inner, &mut outer) != 0 {
                pip_done();
                return 1;
            }
            cylinder.density = [inner, outer];
        } else if index > 0 {
            cylinder.density = first.density;
        }
    }
    pip_done();

    let Some(mut file) = ImodFile::open(&output, "wb") else {
        return 1;
    };
    let mut header = MrcHeader::default();
    if mrc_head_new(
        &mut header,
        dimensions[0],
        dimensions[1],
        dimensions[2],
        mode,
    ) != 0
        || mrc_head_write(&mut file, &mut header) != 0
    {
        return 1;
    }
    let (mut minimum, mut maximum, mut total) = (f32::INFINITY, f32::NEG_INFINITY, 0_f64);
    for z in 0..dimensions[2] {
        let Some(mut slice) = slice_create(dimensions[0], dimensions[1], mode) else {
            return 1;
        };
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let mut value = background;
                for sphere in &spheres {
                    let position = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                    if (0..3)
                        .map(|axis| (sphere.center[axis] - position[axis]).powi(2))
                        .sum::<f32>()
                        <= (sphere.radius[1] + 1.5).powi(2)
                    {
                        let mut sum = 0.;
                        for dz in 0..5 {
                            for dy in 0..5 {
                                for dx in 0..5 {
                                    let sample = [
                                        x as f32 + (dx as f32 + 0.5) / 5.,
                                        y as f32 + (dy as f32 + 0.5) / 5.,
                                        z as f32 + (dz as f32 + 0.5) / 5.,
                                    ];
                                    let radius = (0..3)
                                        .map(|axis| (sphere.center[axis] - sample[axis]).powi(2))
                                        .sum::<f32>()
                                        .sqrt();
                                    if radius <= sphere.radius[0] {
                                        sum += sphere.density[0];
                                    } else if radius <= sphere.radius[1] {
                                        sum += sphere.density[1]
                                            + (sphere.density[2] - sphere.density[1])
                                                * (radius - sphere.radius[0])
                                                / (sphere.radius[1] - sphere.radius[0]);
                                    }
                                }
                            }
                        }
                        value += sum / 125.;
                    }
                }
                for cylinder in &cylinders {
                    let point = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                    let distance_squared = if cylinder.truncated {
                        point_line_seg_dist(cylinder.start, cylinder.end, point).1
                    } else {
                        let aa = (cylinder.end[0] - cylinder.start[0])
                            / (cylinder.end[2] - cylinder.start[2]);
                        let bb =
                            cylinder.start[0] + offsets[0] - aa * (cylinder.start[2] + offsets[2]);
                        let cc = (cylinder.end[1] - cylinder.start[1])
                            / (cylinder.end[2] - cylinder.start[2]);
                        let dd =
                            cylinder.start[1] + offsets[1] - cc * (cylinder.start[2] + offsets[2]);
                        let t = point_to_line(aa, bb, cc, dd, point[0], point[1], point[2]);
                        (aa * t + bb - point[0]).powi(2)
                            + (cc * t + dd - point[1]).powi(2)
                            + (t - point[2]).powi(2)
                    };
                    if distance_squared <= (cylinder.radius[1] + 1.5).powi(2) {
                        let mut sum = 0.;
                        for dz in 0..5 {
                            for dy in 0..5 {
                                for dx in 0..5 {
                                    let sample = [
                                        x as f32 + (dx as f32 + 0.5) / 5.,
                                        y as f32 + (dy as f32 + 0.5) / 5.,
                                        z as f32 + (dz as f32 + 0.5) / 5.,
                                    ];
                                    let radius = if cylinder.truncated {
                                        point_line_seg_dist(cylinder.start, cylinder.end, sample)
                                            .1
                                            .sqrt()
                                    } else {
                                        let aa = (cylinder.end[0] - cylinder.start[0])
                                            / (cylinder.end[2] - cylinder.start[2]);
                                        let bb = cylinder.start[0] + offsets[0]
                                            - aa * (cylinder.start[2] + offsets[2]);
                                        let cc = (cylinder.end[1] - cylinder.start[1])
                                            / (cylinder.end[2] - cylinder.start[2]);
                                        let dd = cylinder.start[1] + offsets[1]
                                            - cc * (cylinder.start[2] + offsets[2]);
                                        let t = point_to_line(
                                            aa, bb, cc, dd, sample[0], sample[1], sample[2],
                                        );
                                        ((aa * t + bb - sample[0]).powi(2)
                                            + (cc * t + dd - sample[1]).powi(2)
                                            + (t - sample[2]).powi(2))
                                        .sqrt()
                                    };
                                    if radius <= cylinder.radius[0] {
                                        sum += cylinder.density[0];
                                    } else if radius <= cylinder.radius[1] {
                                        sum += cylinder.density[1];
                                    }
                                }
                            }
                        }
                        value += sum / 125.;
                    }
                }
                let _ = slice_put_val(&mut slice, x, y, [value, 0., 0., 0.]);
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                total += value as f64;
            }
        }
        if mrc_write_slice(&slice.data, &mut file, &mut header, z, b'Z') != 0 {
            return 1;
        }
    }
    header.amin = minimum;
    header.amax = maximum;
    header.amean =
        (total / (dimensions[0] as f64 * dimensions[1] as f64 * dimensions[2] as f64)) as f32;
    mrc_head_label(&mut header, b"fakevolume: Volume with synthesized features");
    if mrc_head_write(&mut file, &mut header) != 0 {
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read, mrc_read_slice};
    #[test]
    fn line_and_segment_distance_match_source_geometry() {
        assert_eq!(point_to_line(0., 0., 0., 0., 2., 3., 4.), 4.);
        let (t, distance) = point_line_seg_dist([0., 0., 0.], [2., 0., 0.], [1., 3., 0.]);
        assert_eq!((t, distance), (0.5, 9.));
    }

    #[test]
    fn command_writes_a_real_sphere_volume() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-fakevolume-{}.mrc", std::process::id()));
        let arguments = vec![
            "fakevolume".into(),
            "-output".into(),
            path.display().to_string(),
            "-size".into(),
            "5,5,1".into(),
            "-back".into(),
            "1".into(),
            "-scen".into(),
            "2.5,2.5,0.5".into(),
            "-stype".into(),
            "1".into(),
            "-sradii".into(),
            "1".into(),
            "-sdens".into(),
            "4".into(),
        ];
        assert_eq!(fakevolume(&arguments), 0);
        let mut file = ImodFile::open(&path, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (5, 5, 1));
        let mut data = vec![0_u8; 5 * 5 * 4];
        assert_eq!(
            mrc_read_slice(&mut data, &mut file, &mut header, 0, b'Z'),
            0
        );
        assert!(f32::from_ne_bytes(data[12 * 4..13 * 4].try_into().unwrap()) > 1.);
        std::fs::remove_file(path).unwrap();
    }
}
