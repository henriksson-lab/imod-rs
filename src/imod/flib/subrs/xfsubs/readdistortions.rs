//! Translation of `IMOD/flib/subrs/xfsubs/readdistortions.f`.
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
