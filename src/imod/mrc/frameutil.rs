//! Safe translation of `IMOD/mrc/frameutil.cpp` and `.h`.

use crate::imod::libcfshr::b3dutil::{ImodFile, data_size_for_mode};
use crate::imod::libcfshr::filtxcorr::wrap_fft_slice;
use crate::imod::libfft::todfft::todfft_c;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_write, mrc_write_slice,
};
use std::sync::{
    LazyLock, Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

static DUMP_INDEX: AtomicUsize = AtomicUsize::new(0);
static PRINT_FUNCTION: LazyLock<Mutex<Option<fn(&str)>>> = LazyLock::new(|| Mutex::new(None));
static DUMP_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

/// C `utilCoordsForWrap`.
pub fn util_coords_for_wrap(
    nx_from: i32,
    ny_from: i32,
    nx_to: i32,
    ny_to: i32,
    x_offset: i32,
    y_offset: i32,
) -> (
    [i32; 4],
    [i32; 4],
    [i32; 4],
    [i32; 4],
    [i32; 4],
    [i32; 4],
    [i32; 4],
    [i32; 4],
) {
    let mut xf0 = [0; 4];
    let mut xt0 = [0; 4];
    let mut yf0 = [0; 4];
    let mut yt0 = [0; 4];
    let mut xf1 = [0; 4];
    let mut xt1 = [0; 4];
    let mut yf1 = [0; 4];
    let mut yt1 = [0; 4];
    xf0[0] = 0.max(x_offset - nx_to / 2);
    xf0[3] = xf0[0];
    xf1[0] = nx_to / 2 + x_offset - 1;
    xf1[3] = xf1[0];
    let mut num = xf1[0] + 1 - xf0[0];
    xt1[0] = nx_to - 1;
    xt1[3] = xt1[0];
    xt0[0] = nx_to - num;
    xt0[3] = xt0[0];
    yf0[0] = 0.max(y_offset - ny_to / 2);
    yf0[1] = yf0[0];
    yf1[0] = ny_to / 2 + y_offset - 1;
    yf1[1] = yf1[0];
    num = yf1[0] + 1 - yf0[0];
    yt1[0] = ny_to - 1;
    yt1[1] = yt1[0];
    yt0[0] = ny_to - num;
    yt0[1] = yt0[0];
    xf0[1] = x_offset + nx_from - nx_to / 2;
    xf0[2] = xf0[1];
    xf1[1] = (nx_from - 1).min(xf0[1] + nx_to - 1);
    xf1[2] = xf1[1];
    num = xf1[1] + 1 - xf0[1];
    xt0[1] = 0;
    xt0[2] = 0;
    xt1[1] = num - 1;
    xt1[2] = num - 1;
    yf0[2] = y_offset + ny_from - ny_to / 2;
    yf0[3] = yf0[2];
    yf1[2] = (ny_from - 1).min(yf0[2] + ny_to - 1);
    yf1[3] = yf1[2];
    num = yf1[2] + 1 - yf0[2];
    yt0[2] = 0;
    yt0[3] = 0;
    yt1[2] = num - 1;
    yt1[3] = num - 1;
    (xf0, xt0, yf0, yt0, xf1, xt1, yf1, yt1)
}

/// C `utilRollSavedFrames`.
pub fn util_roll_saved_frames<T>(saved: &mut [T], num_frames: usize) {
    if num_frames > 1 && num_frames <= saved.len() {
        saved[..num_frames].rotate_left(1);
    }
}

fn dump_path(prefix: &str) -> std::path::PathBuf {
    let index = DUMP_INDEX.fetch_add(1, Ordering::Relaxed);
    check_dump_dir().join(format!("{prefix}-{index}.mrc"))
}

/// C `checkDumpDir`: capture `FRAMEALIGN_DUMPDIR` once for all image dumps.
fn check_dump_dir() -> &'static std::path::PathBuf {
    DUMP_DIR.get_or_init(|| {
        std::env::var_os("FRAMEALIGN_DUMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    })
}

/// C `utilDumpImage`, with the source's raw pixel storage represented by a
/// bounded byte slice.  A negative `if_corr` selects `-if_corr - 1` as the
/// MRC mode, exactly as in the C entry point.
pub fn util_dump_image(
    buf: &[u8],
    nx_dim: usize,
    nx_pad: usize,
    ny_pad: usize,
    if_corr: i32,
    descrip: &str,
    frame: i32,
) -> Option<std::path::PathBuf> {
    let mode = if if_corr < 0 {
        -if_corr - 1
    } else {
        MRC_MODE_FLOAT
    };
    let (mut pixel_bytes, mut channels) = (0, 0);
    if data_size_for_mode(mode, &mut pixel_bytes, &mut channels) != 0 || nx_dim < nx_pad {
        return None;
    }
    let pixel_bytes = usize::try_from(pixel_bytes.checked_mul(channels)?).ok()?;
    if buf.len() < nx_dim.checked_mul(ny_pad)?.checked_mul(pixel_bytes)? {
        return None;
    }
    let mut data = vec![0_u8; nx_pad.checked_mul(ny_pad)?.checked_mul(pixel_bytes)?];
    for y in 0..ny_pad {
        for x in 0..nx_pad {
            let (sx, sy) = if if_corr > 0 {
                ((x + nx_pad / 2) % nx_pad, (y + ny_pad / 2) % ny_pad)
            } else {
                (x, y)
            };
            let source = (sx + sy * nx_dim) * pixel_bytes;
            let target = (x + y * nx_pad) * pixel_bytes;
            data[target..target + pixel_bytes].copy_from_slice(&buf[source..source + pixel_bytes]);
        }
    }
    let path = dump_path("faimg");
    let mut file = ImodFile::open(&path, "wb")?;
    let mut header = MrcHeader::default();
    if mrc_head_new(&mut header, nx_pad as i32, ny_pad as i32, 1, mode) != 0 {
        return None;
    };
    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.);
    for pixel in data.chunks_exact(pixel_bytes) {
        let value = match mode {
            0 => pixel[0] as f32,
            1 => i16::from_ne_bytes([pixel[0], pixel[1]]) as f32,
            6 => u16::from_ne_bytes([pixel[0], pixel[1]]) as f32,
            2 => f32::from_ne_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]),
            _ => 0.,
        };
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    header.amin = min;
    header.amax = max;
    header.amean = sum / data.len() as f32;
    if mrc_head_write(&mut file, &mut header) != 0
        || mrc_write_slice(&data, &mut file, &mut header, 0, b'Z') != 0
    {
        return None;
    };
    util_print(&format!(
        "Saved {descrip} image frame {frame} in {}    mean {:.2}\n",
        path.display(),
        header.amean
    ));
    Some(path)
}

/// Typed convenience entry for native float callers.
pub fn util_dump_float_image(
    buf: &[f32],
    nx_dim: usize,
    nx_pad: usize,
    ny_pad: usize,
    if_corr: i32,
    descrip: &str,
    frame: i32,
) -> Option<std::path::PathBuf> {
    let bytes = buf
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    util_dump_image(&bytes, nx_dim, nx_pad, ny_pad, if_corr, descrip, frame)
}

/// C `utilDumpFFT`; complex storage is dumped as its interleaved float image.
pub fn util_dump_fft(
    fft: &mut [f32],
    nx_pad: usize,
    ny_pad: usize,
    descrip: &str,
    real: bool,
    frame: i32,
    scale: bool,
) -> Option<std::path::PathBuf> {
    let count = (nx_pad + 2).checked_mul(ny_pad)?;
    if fft.len() < count {
        return None;
    };
    if real {
        todfft_c(&mut fft[..count], nx_pad as i32, ny_pad as i32, 1);
        let path =
            util_dump_float_image(&fft[..count], nx_pad + 2, nx_pad, ny_pad, 0, descrip, frame);
        todfft_c(&mut fft[..count], nx_pad as i32, ny_pad as i32, 0);
        return path;
    }
    let mut temporary = vec![0.0; 2 * nx_pad + 4];
    wrap_fft_slice(
        &mut fft[..count],
        &mut temporary,
        ((nx_pad + 2) / 2) as i32,
        ny_pad as i32,
        0,
    );
    let factor = 1.0 / ((nx_pad * ny_pad) as f32).sqrt();
    if scale {
        for value in &mut fft[..count] {
            *value *= factor;
        }
    }
    let path = util_dump_float_image(
        &fft[..count],
        nx_pad + 2,
        nx_pad + 2,
        ny_pad,
        0,
        descrip,
        frame,
    );
    if scale {
        for value in &mut fft[..count] {
            *value /= factor;
        }
    }
    wrap_fft_slice(
        &mut fft[..count],
        &mut temporary,
        ((nx_pad + 2) / 2) as i32,
        ny_pad as i32,
        1,
    );
    path
}

/// C `utilSetPrintFunc`.
pub fn util_set_print_func(function: Option<fn(&str)>) {
    *PRINT_FUNCTION.lock().unwrap_or_else(|p| p.into_inner()) = function;
}
/// C `utilPrint`, expressed as already formatted Rust text.
pub fn util_print(message: &str) {
    if let Some(function) = *PRINT_FUNCTION.lock().unwrap_or_else(|p| p.into_inner()) {
        function(message)
    } else {
        print!("{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::mrc_head_read;
    #[test]
    fn wrap_coordinates_match_four_quadrants() {
        let (xf0, xt0, yf0, yt0, xf1, xt1, yf1, yt1) = util_coords_for_wrap(100, 80, 20, 16, 0, 0);
        assert_eq!((xf0[0], xf1[0], xt0[0], xt1[0]), (0, 9, 10, 19));
        assert_eq!((yf0[0], yf1[0], yt0[0], yt1[0]), (0, 7, 8, 15));
        assert_eq!((xf0[2], xf1[2], xt0[2], xt1[2]), (90, 99, 0, 9));
        assert_eq!((yf0[3], yf1[3], yt0[3], yt1[3]), (72, 79, 0, 7));
    }
    #[test]
    fn rolls_owned_saved_frames() {
        let mut v = vec![1, 2, 3];
        util_roll_saved_frames(&mut v, 3);
        assert_eq!(v, [2, 3, 1]);
    }

    #[test]
    fn dump_image_preserves_nonfloat_short_pixels() {
        let values = [-3_i16, 12, 8, -1];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let path = util_dump_image(&bytes, 2, 2, 2, -2, "short", 0).unwrap();
        let mut file = ImodFile::open(&path, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(header.mode, 1);
        assert_eq!((header.amin, header.amax), (-3., 12.));
        let _ = std::fs::remove_file(path);
    }
}
