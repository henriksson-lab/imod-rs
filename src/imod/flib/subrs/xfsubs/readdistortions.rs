//! Translation of `IMOD/flib/subrs/xfsubs/readdistortions.f`.
#![allow(dead_code)]

use crate::imod::flib::subrs::hvem::dopen::dopen;
use std::io::{BufRead, BufReader};

/// The list-directed `read(14, *)` reader these four subroutines share.
///
/// Fortran list-directed input starts each READ statement at a new record and
/// then consumes as many further records as the item list needs, treating
/// blanks and commas as separators.  A single tokenizer over the whole file
/// would silently let one statement pick up where the last one stopped inside
/// a record, which is not what the source does.
struct ListDirected {
    lines: std::vec::IntoIter<String>,
    pending: std::collections::VecDeque<String>,
}

impl ListDirected {
    fn new(file: std::fs::File) -> Self {
        let lines: Vec<String> = BufReader::new(file)
            .lines()
            .map(|line| line.unwrap_or_default())
            .collect();
        Self {
            lines: lines.into_iter(),
            pending: std::collections::VecDeque::new(),
        }
    }

    /// Start a new READ statement: any tokens left over in the current record
    /// are discarded, as Fortran discards the rest of a record.
    fn statement(&mut self) {
        self.pending.clear();
    }

    fn next_token(&mut self) -> Option<String> {
        loop {
            if let Some(token) = self.pending.pop_front() {
                return Some(token);
            }
            let line = self.lines.next()?;
            for token in line.split([' ', '\t', ',', '\r']).filter(|s| !s.is_empty()) {
                self.pending.push_back(token.to_owned());
            }
        }
    }

    fn integer(&mut self) -> i32 {
        self.next_token()
            .and_then(|token| token.parse::<f64>().ok())
            .map_or(0, |value| value as i32)
    }

    fn real(&mut self) -> f32 {
        self.next_token()
            .and_then(|token| token.parse::<f32>().ok())
            .unwrap_or(0.)
    }
}

/// Original `readDistortions` (`readdistortions.f:32`).
#[allow(clippy::too_many_arguments)]
pub fn read_distortions(
    idf_file: &str,
    field_dx: &mut [f32],
    field_dy: &mut [f32],
    lm_grid: i32,
    idf_nx: &mut i32,
    idf_ny: &mut i32,
    idf_binning: &mut i32,
    pixel_idf: &mut f32,
    ix_grid_strt: &mut i32,
    x_grid_intrv: &mut f32,
    nx_grid: &mut i32,
    iy_grid_strt: &mut i32,
    y_grid_intrv: &mut f32,
    ny_grid: &mut i32,
) {
    let mut reader = ListDirected::new(dopen(14, idf_file, "ro", "f"));
    reader.statement();
    let idf_version = reader.integer();
    if idf_version == 1 {
        reader.statement();
        *idf_nx = reader.integer();
        *idf_ny = reader.integer();
        *idf_binning = reader.integer();
        *pixel_idf = reader.real();
    } else if idf_version == 2 {
        reader.statement();
        *idf_nx = reader.integer();
        *idf_ny = reader.integer();
        let _idf_nz = reader.integer();
        *idf_binning = reader.integer();
        *pixel_idf = reader.real();
    } else {
        println!();
        println!("ERROR: readDistortions - version{idf_version:12} of idf file not recognized");
        std::process::exit(1);
    }
    reader.statement();
    *ix_grid_strt = reader.integer();
    *x_grid_intrv = reader.real();
    *nx_grid = reader.integer();
    *iy_grid_strt = reader.integer();
    *y_grid_intrv = reader.real();
    *ny_grid = reader.integer();
    if *nx_grid > lm_grid || *ny_grid > lm_grid {
        println!();
        println!("ERROR: readDistortions - too many grid points for arrays");
        std::process::exit(1);
    }
    for j in 1..=*ny_grid {
        reader.statement();
        for i in 1..=*nx_grid {
            // `fieldDx(lmGrid, lmGrid)` is column-major.
            let index = ((j - 1) * lm_grid + i - 1) as usize;
            field_dx[index] = reader.real();
            field_dy[index] = reader.real();
        }
    }
}

/// Original `loadDistortions` (`readdistortions.f:87`).
#[allow(clippy::too_many_arguments)]
pub fn load_distortions(
    idf_file: &str,
    field_dx: &mut [f32],
    field_dy: &mut [f32],
    lm_all_grid: i32,
    idf_nx: &mut i32,
    idf_ny: &mut i32,
    idf_nz: &mut i32,
    idf_binning: &mut i32,
    pixel_idf: &mut f32,
    ix_grid_strt: &mut [i32],
    x_grid_intrv: &mut [f32],
    nx_grid: &mut [i32],
    iy_grid_strt: &mut [i32],
    y_grid_intrv: &mut [f32],
    ny_grid: &mut [i32],
    lm_sec: i32,
) {
    let mut reader = ListDirected::new(dopen(14, idf_file, "ro", "f"));
    reader.statement();
    let idf_version = reader.integer();
    if idf_version == 1 {
        reader.statement();
        *idf_nx = reader.integer();
        *idf_ny = reader.integer();
        *idf_binning = reader.integer();
        *pixel_idf = reader.real();
        *idf_nz = 1;
    } else if idf_version == 2 {
        reader.statement();
        *idf_nx = reader.integer();
        *idf_ny = reader.integer();
        *idf_nz = reader.integer();
        *idf_binning = reader.integer();
        *pixel_idf = reader.real();
    } else {
        println!();
        println!("ERROR: loadDistortions - version{idf_version:12} of idf file not recognized");
        std::process::exit(1);
    }
    if *idf_nz > lm_sec {
        println!();
        println!("ERROR: loadDistortions - too many distortion fields for arrays");
        std::process::exit(1);
    }
    //
    // Read each distortion field sequentially into field arrays
    //
    let mut ind_base = 0_i32;
    for iz in 1..=*idf_nz as usize {
        reader.statement();
        ix_grid_strt[iz - 1] = reader.integer();
        x_grid_intrv[iz - 1] = reader.real();
        nx_grid[iz - 1] = reader.integer();
        iy_grid_strt[iz - 1] = reader.integer();
        y_grid_intrv[iz - 1] = reader.real();
        ny_grid[iz - 1] = reader.integer();
        if ind_base + nx_grid[iz - 1] * ny_grid[iz - 1] > lm_all_grid {
            println!();
            println!("ERROR: loadDistortions - too many grid points for arrays");
            std::process::exit(1);
        }
        for _ in 1..=ny_grid[iz - 1] {
            reader.statement();
            for i in 1..=nx_grid[iz - 1] {
                field_dx[(ind_base + i - 1) as usize] = reader.real();
                field_dy[(ind_base + i - 1) as usize] = reader.real();
            }
            ind_base += nx_grid[iz - 1];
        }
    }
}

/// Original `getSecDistortions` (`readdistortions.f:157`).
pub fn get_sec_distortions(
    iz_sec: i32,
    all_dx: &[f32],
    all_dy: &[f32],
    nx_grid: &[i32],
    ny_grid: &[i32],
    field_dx: &mut [f32],
    field_dy: &mut [f32],
    lm_grid: i32,
) {
    if nx_grid[iz_sec as usize - 1] > lm_grid || ny_grid[iz_sec as usize - 1] > lm_grid {
        println!();
        println!("ERROR: getSecDistortions - too many grid points for arrays");
        std::process::exit(1);
    }
    let mut ind_base = 0_i32;
    for i in 1..iz_sec as usize {
        ind_base += nx_grid[i - 1] * ny_grid[i - 1];
    }
    for j in 1..=ny_grid[iz_sec as usize - 1] {
        for i in 1..=nx_grid[iz_sec as usize - 1] {
            let index = ((j - 1) * lm_grid + i - 1) as usize;
            field_dx[index] = all_dx[(ind_base + i - 1) as usize];
            field_dy[index] = all_dy[(ind_base + i - 1) as usize];
        }
        ind_base += nx_grid[iz_sec as usize - 1];
    }
}

/// Original `readMagGradients` (`readdistortions.f:198`).
#[allow(clippy::too_many_arguments)]
pub fn read_mag_gradients(
    mag_file: &str,
    max_vals: i32,
    pixel_size: &mut f32,
    axis_rot: &mut f32,
    tilt: &mut [f32],
    dmag_per_um: &mut [f32],
    rot_per_um: &mut [f32],
    num_vals: &mut i32,
) {
    let mut reader = ListDirected::new(dopen(14, mag_file, "ro", "f"));
    reader.statement();
    let mag_version = reader.integer();
    if mag_version == 1 {
        reader.statement();
        *num_vals = reader.integer();
        *pixel_size = reader.real();
        *axis_rot = reader.real();
        if *num_vals > max_vals {
            println!();
            println!("ERROR: readMagGradients - too many values for arrays");
            std::process::exit(1);
        }
        for i in 1..=*num_vals as usize {
            reader.statement();
            tilt[i - 1] = reader.real();
            dmag_per_um[i - 1] = reader.real();
            rot_per_um[i - 1] = reader.real();
        }
    } else {
        println!();
        println!(
            "ERROR: readMagGradients - version{mag_version:12} of gradient file not recognized"
        );
        std::process::exit(1);
    }
}
