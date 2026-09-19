//! Translation of `IMOD/clip/fft.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, IP_DEFAULT};
use crate::imod::clip::file_io::{
    clip_write_slice, grap_volume_free, grap_volume_read, grap_volume_write, set_input_options,
    set_output_options,
};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libcfshr::islice::{Islice, Istack, slice_create};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MrcHeader, mrc_coord_cp,
    mrc_head_label, mrc_head_label_cp, mrc_head_new, mrc_head_write,
};
use crate::imod::libiimod::mrcslice::{
    slice_box_in, slice_complex_float, slice_float, slice_read_subm, slice_wrap_fft_lines,
};
use std::io::Write as _;

/// C++ `clip_fft` (`fft.cpp:21`).
pub fn clip_fft(input: &mut MrcHeader, output: &mut MrcHeader, options: &mut ClipOptions) -> i32 {
    let complex = input.mode == MRC_MODE_COMPLEX_FLOAT || input.mode == MRC_MODE_COMPLEX_SHORT;
    if complex {
        if options.ix != IP_DEFAULT
            || options.iy != IP_DEFAULT
            || options.cx != IP_DEFAULT as f32
            || options.cy != IP_DEFAULT as f32
        {
            crate::imod::clip::clip::show_warning(
                "clip inverse fft - input sizes or centers are ignored",
            );
        }
        options.ix = IP_DEFAULT;
        options.iy = IP_DEFAULT;
        options.cx = IP_DEFAULT as f32;
        options.cy = IP_DEFAULT as f32;
        if options.mode == IP_DEFAULT {
            options.mode = MRC_MODE_FLOAT;
        }
    } else {
        if options.ox != IP_DEFAULT
            || options.oy != IP_DEFAULT
            || options.oz != IP_DEFAULT
            || options.mode != IP_DEFAULT
        {
            crate::imod::clip::clip::show_warning(
                "clip forward fft - output sizes or mode are ignored",
            );
        }
        options.ox = IP_DEFAULT;
        options.oy = IP_DEFAULT;
        options.oz = IP_DEFAULT;
        options.mode = MRC_MODE_COMPLEX_FLOAT;
        options.ocanresize = 0;
    }
    if options.dim == 3 {
        return clip_3dfft(input, output, options);
    }
    if complex && options.ox == IP_DEFAULT {
        options.ox = 2 * (input.nx - 1);
    }
    set_input_options(options, input);
    if !complex
        && (clip_nicesize(options.ix) == 0 || clip_nicesize(options.iy) == 0 || options.ix % 2 != 0)
    {
        if crate::imod::libfft::using_fftw() != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "ERROR: clip - fft input size in X (%d) must be even.\n",
                    &[CArg::Int((options.ix) as i64)],
                )
                .as_bytes(),
            );
        } else {
            let _ = ImodFile::Stdout.write_all(c_format("ERROR: clip - fft input size (%d, %d) is odd and/or has factors greater than 19.\n", &[CArg::Int((options.ix) as i64), CArg::Int((options.iy) as i64)]).as_bytes());
        }
        return -1;
    }
    if !complex {
        options.ox = options.ix / 2 + 1;
        options.oy = options.iy;
        options.oz = options.nofsecs;
        if options.add2file != 0
            && (output.mode != options.mode || output.nx != options.ox || output.ny != options.oy)
        {
            crate::imod::clip::clip::show_error(
                "clip forward 2d fft - cannot append to output file of different size or mode",
            );
            return -1;
        }
    }
    let mut z = set_output_options(options, output);
    if z < 0 {
        return z;
    }
    mrc_head_label_cp(input, output);
    mrc_head_label(output, b"clip: 2d fft");
    crate::imod::clip::clip::show_status("Doing fast fourier transform...\n");
    for k in 0..options.nofsecs {
        let slice = slice_read_subm(
            input,
            options.secs[(k) as usize],
            b'z',
            options.ix,
            options.iy,
            options.cx as i32,
            options.cy as i32,
        );
        let Some(mut slice) = slice else {
            crate::imod::clip::clip::show_error("fft: Error reading slice.");
            return -1;
        };
        slice_fft(&mut slice);
        if clip_write_slice(&mut slice, output, options, k, &mut z, 1) != 0 {
            return -1;
        }
    }
    mrc_coord_cp(output, input);
    if input.nx == input.mx && input.ny == input.my && input.nz == input.mz {
        output.mx = output.nx;
    }
    let mut fp = output.fp.clone().unwrap();
    mrc_head_write(&mut fp, output)
}
/// C++ `slice_fft` (`fft.cpp:111`).  The numerical backend is supplied by
/// IMOD's translated `cfft` layer; this preserves its input/output layout.
pub fn slice_fft(slice: &mut Islice) -> i32 {
    let nx2 = slice.xsize + 2;
    let ncs = slice.xsize as usize * core::mem::size_of::<f32>();
    let idir = if slice.mode == MRC_MODE_COMPLEX_FLOAT || slice.mode == MRC_MODE_COMPLEX_SHORT {
        -1
    } else {
        1
    };
    if idir == 1 {
        slice_float(slice);
        let width = slice.xsize as usize;
        let height = slice.ysize as usize;
        // `fft.cpp:130-136`: one `malloc`, the rows copied into it from the
        // slice's own data, the old data freed, and the buffer handed to
        // `sliceInit` as the slice's data.  Allocating it as bytes lets it
        // become `slice.data` with no copy back.
        let buffer_len = nx2 as usize * height * core::mem::size_of::<f32>();
        if slice.data.len() != width * height * core::mem::size_of::<f32>() {
            return -1;
        }
        let mut buffer = Vec::new();
        if buffer.try_reserve_exact(buffer_len).is_err() {
            return -1;
        }
        buffer.resize(buffer_len, 0_u8);
        for row in 0..height {
            let source_start = row * ncs;
            let buffer_start = row * nx2 as usize * core::mem::size_of::<f32>();
            buffer[buffer_start..buffer_start + ncs]
                .copy_from_slice(&slice.data[source_start..source_start + ncs]);
        }
        slice.data = buffer;
        let nx = slice.xsize;
        let ny = slice.ysize;
        let (head, floats, tail) = unsafe { slice.data.align_to_mut::<f32>() };
        if head.is_empty() && tail.is_empty() {
            mrc_to_dfft(floats, nx, ny, 0);
        } else {
            // A whole `Vec<u8>` allocation from the global allocator is always
            // sufficiently aligned; transcode rather than read unaligned if
            // that ever fails to hold.
            let mut float_data = slice
                .data
                .chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            mrc_to_dfft(&mut float_data, nx, ny, 0);
            slice.data = float_data.into_iter().flat_map(f32::to_ne_bytes).collect();
        }
        slice.xsize = slice.xsize / 2 + 1;
        slice.mode = MRC_MODE_COMPLEX_FLOAT;
        slice.csize = 2;
        slice.dsize = core::mem::size_of::<f32>() as i32;
        slice_wrap_fft_lines(slice, 0);
    } else {
        slice_complex_float(slice);
        slice_wrap_fft_lines(slice, 1);
        let new_xsize = 2 * (slice.xsize - 1);
        let width = new_xsize as usize;
        let height = slice.ysize as usize;
        let packed_width = width + 2;
        if slice.data.len() != packed_width * height * core::mem::size_of::<f32>() {
            return -1;
        }
        // `fft.cpp:161-170` transforms `slice->data.f` in place and then
        // repacks the rows within that same buffer; reinterpret the byte
        // buffer instead of transcoding the slice in both directions.
        let ny = slice.ysize;
        let (head, floats, tail) = unsafe { slice.data.align_to_mut::<f32>() };
        if head.is_empty() && tail.is_empty() {
            mrc_to_dfft(floats, new_xsize, ny, 1);
        } else {
            // See the forward branch: unreachable with the global allocator.
            let mut packed = slice
                .data
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            mrc_to_dfft(&mut packed, new_xsize, ny, 1);
            slice.data = packed.into_iter().flat_map(f32::to_ne_bytes).collect();
        }
        for row in 0..height {
            let source_start = row * packed_width * core::mem::size_of::<f32>();
            let dest_start = row * width * core::mem::size_of::<f32>();
            slice.data.copy_within(
                source_start..source_start + width * core::mem::size_of::<f32>(),
                dest_start,
            );
        }
        slice
            .data
            .truncate(width * height * core::mem::size_of::<f32>());
        slice.xsize = new_xsize;
        slice.mode = MRC_MODE_FLOAT;
        slice.csize = 1;
    }
    0
}
/// C++ `clip_3dfft` (`fft.cpp:191`).
pub fn clip_3dfft(input: &mut MrcHeader, output: &mut MrcHeader, options: &mut ClipOptions) -> i32 {
    if input.mode != MRC_MODE_COMPLEX_FLOAT {
        if options.ix == IP_DEFAULT {
            options.ix = input.nx;
        }
        if options.iy == IP_DEFAULT {
            options.iy = input.ny;
        }
        if options.iz == IP_DEFAULT {
            options.iz = input.nz;
        }
        if clip_nicesize(options.ix) == 0
            || clip_nicesize(options.iy) == 0
            || clip_nicesize(options.iz) == 0
            || options.ix % 2 != 0
        {
            let _ = ImodFile::Stdout.write_all(c_format("ERROR: clip - fft input size %dx%dx%d is odd and/or has factors greater than 19.\n", &[CArg::Int((options.ix) as i64), CArg::Int((options.iy) as i64), CArg::Int((options.iz) as i64)]).as_bytes());
            return -1;
        }
    }
    let Some(mut volume) = grap_volume_read(input, options) else {
        return -1;
    };
    if input.mode != MRC_MODE_COMPLEX_FLOAT {
        for slice in &mut volume.slices {
            if input.mode == MRC_MODE_COMPLEX_SHORT {
                slice_complex_float(slice);
            } else {
                slice_float(slice);
            }
        }
    }
    crate::imod::clip::clip::show_status("Doing 3d fast fourier transform in core...\n");
    clip_fftvol(&mut volume);
    let Some(first) = volume.slices.first() else {
        return -1;
    };
    mrc_head_new(
        output,
        first.xsize,
        first.ysize,
        volume.slices.len() as i32,
        first.mode,
    );
    mrc_head_label_cp(input, output);
    mrc_head_label(
        output,
        if input.mode != MRC_MODE_COMPLEX_FLOAT {
            b"Clip: Forward 3D FFT"
        } else {
            b"Clip: Inverse 3D FFT"
        },
    );
    if grap_volume_write(&mut volume, output, options) != 0 {
        return -1;
    }
    mrc_coord_cp(output, input);
    if input.nx == input.mx && input.ny == input.my && input.nz == input.mz {
        output.mx = output.nx;
    }
    let mut fp = output.fp.clone().unwrap();
    if mrc_head_write(&mut fp, output) != 0 {
        return -1;
    }
    0
}
/// C++ `clip_fftvol` (`fft.cpp:258`).
pub fn clip_fftvol(volume: &mut Istack) -> i32 {
    // `fft.cpp:258` guards on `!v`; a reference is never null.
    if volume.slices.is_empty() {
        return -1;
    }
    let first = volume.slices.first().unwrap();
    let ncs = first.xsize as usize * core::mem::size_of::<f32>();
    if first.mode == MRC_MODE_COMPLEX_FLOAT {
        clip_wrapvol(volume, 1);
        clip_fftvol3(volume, -2);
        for slice in &mut volume.slices {
            // `fft.cpp:265` transforms `v->vol[z]->data.f` in place.  The
            // translated `Islice` keeps its pixels as bytes, so reinterpret
            // that buffer as `f32` for the call rather than transcoding the
            // whole slice into a second array and back.
            let nx = (slice.xsize - 1) * 2;
            let ny = slice.ysize;
            let (head, floats, tail) = unsafe { slice.data.align_to_mut::<f32>() };
            if head.is_empty() && tail.is_empty() {
                mrc_to_dfft(floats, nx, ny, -1);
            } else {
                // A whole `Vec<u8>` allocation from the global allocator is
                // always sufficiently aligned; transcode rather than read
                // unaligned if that ever fails to hold.
                let mut float_data = slice
                    .data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                if float_data.len() * 4 != slice.data.len() {
                    return -1;
                }
                mrc_to_dfft(&mut float_data, nx, ny, -1);
                slice.data = float_data.into_iter().flat_map(f32::to_ne_bytes).collect();
            }
            slice.mode = MRC_MODE_FLOAT;
            slice.csize = 1;
            slice.xsize *= 2;
            let xsize = slice.xsize;
            let ysize = slice.ysize;
            slice_box_in(slice, 0, 0, xsize - 2, ysize);
        }
    } else {
        let nx2 = first.xsize + 2;
        for slice in &mut volume.slices {
            slice_float(slice);
            // `fft.cpp:278-284`: one `malloc` per slice, filled row by row from
            // the slice's own data, which is then freed and *replaced* by that
            // buffer.  Allocating it as bytes lets it become `slice.data`
            // without a copy back.
            let buffer_len = (nx2 * slice.ysize) as usize * core::mem::size_of::<f32>();
            let mut buffer = Vec::new();
            if buffer.try_reserve_exact(buffer_len).is_err() {
                return -1;
            }
            buffer.resize(buffer_len, 0_u8);
            if slice.data.len() % 4 != 0 {
                return -1;
            }
            for row in 0..slice.ysize as usize {
                let source_start = row * slice.xsize as usize * core::mem::size_of::<f32>();
                let dest_start = row * nx2 as usize * core::mem::size_of::<f32>();
                buffer[dest_start..dest_start + ncs]
                    .copy_from_slice(&slice.data[source_start..source_start + ncs]);
            }
            slice.data = buffer;
            slice.xsize += 2;
            let nx = slice.xsize - 2;
            let ny = slice.ysize;
            let (head, floats, tail) = unsafe { slice.data.align_to_mut::<f32>() };
            if head.is_empty() && tail.is_empty() {
                mrc_to_dfft(floats, nx, ny, 0);
            } else {
                // See the inverse branch: unreachable with the global allocator.
                let mut float_data = slice
                    .data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                mrc_to_dfft(&mut float_data, nx, ny, 0);
                slice.data = float_data.into_iter().flat_map(f32::to_ne_bytes).collect();
            }
            slice.xsize /= 2;
            slice.mode = MRC_MODE_COMPLEX_FLOAT;
            slice.csize = 2;
        }
        clip_fftvol3(volume, -1);
        clip_wrapvol(volume, 0);
    }
    0
}
/// C++ `clip_fftvol3` (`fft.cpp:309`).
pub fn clip_fftvol3(volume: &mut Istack, idir: i32) -> i32 {
    if volume.slices.is_empty() {
        return -1;
    }
    let first_xsize = volume.slices[0].xsize as usize;
    let first_ysize = volume.slices[0].ysize as usize;
    let depth = volume.slices.len();
    for slice in &volume.slices {
        if slice.data.len() != 2 * first_xsize * first_ysize * core::mem::size_of::<f32>() {
            return -1;
        }
    }
    // `fft.cpp:310` allocates one `zsize` by `vxsize` complex slice and moves
    // values through `v->vol[z]->data.f` in place, so that transposed line is
    // the routine's only allocation.  The translated `Islice` keeps its pixels
    // as bytes; read and write them where they lie rather than staging a
    // second copy of the whole volume.
    let mut work = vec![0_f32; 2 * depth * first_xsize];
    for y in 0..first_ysize {
        for (z, slice) in volume.slices.iter().enumerate() {
            let inp = &slice.data[8 * y * first_xsize..][..8 * first_xsize];
            for x in 0..first_xsize {
                work[2 * (x * depth + z)] =
                    f32::from_ne_bytes(inp[8 * x..8 * x + 4].try_into().unwrap());
                work[2 * (x * depth + z) + 1] =
                    f32::from_ne_bytes(inp[8 * x + 4..8 * x + 8].try_into().unwrap());
            }
        }
        mrc_odfft(&mut work, depth as i32, first_xsize as i32, idir);
        for (z, slice) in volume.slices.iter_mut().enumerate() {
            let outp = &mut slice.data[8 * y * first_xsize..][..8 * first_xsize];
            for x in 0..first_xsize {
                outp[8 * x..8 * x + 4].copy_from_slice(&work[2 * (x * depth + z)].to_ne_bytes());
                outp[8 * x + 4..8 * x + 8]
                    .copy_from_slice(&work[2 * (x * depth + z) + 1].to_ne_bytes());
            }
        }
    }
    0
}
/// C++ `clip_wrapvol` (`fft.cpp:350`).
pub fn clip_wrapvol(volume: &mut Istack, direction: i32) -> i32 {
    // `fft.cpp:326` guards on `!v`; a reference is never null.
    let Some(first) = volume.slices.first() else {
        return -1;
    };
    let first_xsize = first.xsize;
    let first_ysize = first.ysize;
    let depth = volume.slices.len() as i32;
    let temporary_len = (2 * first_xsize) as usize;
    let mut temporary = Vec::new();
    if temporary.try_reserve_exact(temporary_len).is_err() {
        crate::imod::clip::clip::show_error(
            "fft: Memory error getting array for temporary line of data\n",
        );
        return 1;
    }
    temporary.resize(temporary_len, 0.);
    let wrapped = (2 * first_xsize * first_ysize) as usize;
    for slice in &mut volume.slices {
        if slice.data.len() % core::mem::size_of::<f32>() != 0
            || slice.data.len() / core::mem::size_of::<f32>() < wrapped
        {
            return -1;
        }
        // `fft.cpp:358` wraps `v->vol[k]->data.f` in place; reinterpret the
        // byte buffer rather than transcoding the slice in both directions.
        let (head, floats, _) = unsafe { slice.data.align_to_mut::<f32>() };
        if head.is_empty() {
            crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
                &mut floats[..wrapped],
                temporary.as_mut_slice(),
                first_xsize,
                first_ysize,
                direction,
            );
        } else {
            // Unreachable with the global allocator; see `clip_fftvol`.
            let mut data: Vec<f32> = slice
                .data
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
                &mut data[..wrapped],
                temporary.as_mut_slice(),
                first_xsize,
                first_ysize,
                direction,
            );
            for (bytes, value) in slice
                .data
                .chunks_exact_mut(core::mem::size_of::<f32>())
                .zip(data)
            {
                bytes.copy_from_slice(&value.to_ne_bytes());
            }
        }
    }
    let mut output = 0;
    let mut low = 0;
    let mut high = 0;
    let increment = crate::imod::libcfshr::filtxcorr::indices_for_fft_wrap(
        depth,
        direction,
        &mut output,
        &mut low,
        &mut high,
    );
    if depth % 2 != 0 {
        for _ in 0..depth / 2 {
            volume.slices.swap(output as usize, high as usize);
            volume.slices.swap(high as usize, low as usize);
            output += increment;
            high += increment;
            low += increment;
        }
    } else {
        for _ in 0..depth / 2 {
            volume.slices.swap(low as usize, high as usize);
            high += 1;
            low += 1;
        }
    }
    0
}
/// C++ `mrcToDFFT` (`fft.cpp:391`).
///
/// The IMOD packed-real layout has two padding values at the end of each row,
/// so `buffer` must contain at least `(nx + 2) * ny` values.
pub fn mrc_to_dfft(buffer: &mut [f32], nx: i32, ny: i32, idir: i32) {
    let required = usize::try_from(nx)
        .ok()
        .and_then(|width| width.checked_add(2))
        .and_then(|width| {
            usize::try_from(ny)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .expect("FFT dimensions must be nonnegative and fit in memory");
    assert!(
        buffer.len() >= required,
        "packed FFT buffer is shorter than its dimensions require"
    );
    crate::imod::libfft::todfft_c(buffer, nx, ny, idir);
}
/// C++ static `mrcODFFT` (`fft.cpp:397`).
///
/// `buffer` contains `nx * ny` complex values, represented as interleaved
/// real and imaginary `f32` values.
pub fn mrc_odfft(buffer: &mut [f32], nx: i32, ny: i32, idir: i32) {
    let required = usize::try_from(nx)
        .ok()
        .and_then(|width| {
            usize::try_from(ny)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|complex_values| complex_values.checked_mul(2))
        .expect("FFT dimensions must be nonnegative and fit in memory");
    assert!(
        buffer.len() >= required,
        "complex FFT buffer is shorter than its dimensions require"
    );
    crate::imod::libfft::odfft_c(buffer, nx, ny, idir);
}
/// C++ `clip_nicesize` (`fft.cpp:405`).
pub fn clip_nicesize(mut size: i32) -> i32 {
    if crate::imod::libfft::using_fftw() != 0 {
        return 1;
    }
    for factor in 2..20 {
        while size % factor == 0 {
            size /= factor;
        }
    }
    if size > 1 { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::{clip_nicesize, mrc_to_dfft, slice_fft};
    use crate::imod::libcfshr::islice::slice_create;
    use crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;

    #[test]
    fn nice_sizes_match_clip_factor_rule() {
        assert_eq!(clip_nicesize(19 * 19 * 18), 1);
        assert_eq!(clip_nicesize(23), 0);
    }

    #[test]
    fn packed_fft_slice_round_trip_preserves_real_values() {
        let mut values = vec![0.0_f32; 6 * 4];
        for y in 0..4 {
            for x in 0..4 {
                values[y * 6 + x] = (3 * y + x) as f32;
            }
        }
        let expected = values.clone();
        mrc_to_dfft(&mut values, 4, 4, 0);
        mrc_to_dfft(&mut values, 4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                let index = y * 6 + x;
                assert!((values[index] - expected[index]).abs() < 1.0e-4);
            }
        }
    }

    #[test]
    fn slice_fft_round_trip_uses_owned_byte_storage() {
        let mut slice = slice_create(4, 4, MRC_MODE_FLOAT).unwrap();
        let expected: Vec<f32> = (0..16).map(|value| value as f32 - 5.0).collect();
        slice.data = expected
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        assert_eq!(slice_fft(&mut slice), 0);
        assert_eq!(slice_fft(&mut slice), 0);
        assert_eq!(
            (slice.xsize, slice.ysize, slice.mode),
            (4, 4, MRC_MODE_FLOAT)
        );
        let restored: Vec<f32> = slice
            .data
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        for (actual, original) in restored.iter().zip(expected) {
            assert!((actual - original).abs() < 1.0e-4);
        }
    }
}
