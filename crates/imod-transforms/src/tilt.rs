use std::io::{self, BufRead, Write};
use std::path::Path;

/// Read tilt angles from a .tlt or .rawtlt file.
/// Format: one angle per line (degrees).
pub fn read_tilt_file(path: impl AsRef<Path>) -> io::Result<Vec<f32>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = io::BufReader::new(file);
    let mut angles = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(angle) = trimmed.parse::<f32>() {
            angles.push(angle);
        }
    }

    Ok(angles)
}

/// Write tilt angles to a file.
/// Format: one angle per line.
pub fn write_tilt_file(path: impl AsRef<Path>, angles: &[f32]) -> io::Result<()> {
    let mut file = std::fs::File::create(path.as_ref())?;
    for &angle in angles {
        writeln!(file, "{:10.2}", angle)?;
    }
    Ok(())
}

/// Generate evenly spaced tilt angles.
pub fn generate_tilt_angles(start: f32, increment: f32, count: usize) -> Vec<f32> {
    (0..count).map(|i| start + i as f32 * increment).collect()
}
