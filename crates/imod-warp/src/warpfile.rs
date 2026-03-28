use std::io::{self, Write};
use std::path::Path;

use imod_transforms::LinearTransform;

/// A warp transform for a single section, consisting of control points
/// with local linear transforms at each point.
#[derive(Debug, Clone)]
pub struct WarpTransform {
    /// Section Z index.
    pub z: i32,
    /// Number of control points.
    pub nx: i32,
    pub ny: i32,
    /// Control point positions.
    pub control_x: Vec<f32>,
    pub control_y: Vec<f32>,
    /// Local linear transform at each control point.
    pub transforms: Vec<LinearTransform>,
}

/// A warp file containing transforms for multiple sections.
#[derive(Debug, Clone)]
pub struct WarpFile {
    pub nx: i32,
    pub ny: i32,
    pub binning: i32,
    pub pixel_size: f32,
    pub version: i32,
    pub flags: i32,
    pub sections: Vec<WarpTransform>,
}

impl WarpFile {
    /// Read a warp file.
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let mut lines = content.lines();

        // Header line: version flags nx ny binning pixelSize numSections
        let header = lines.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "empty warp file")
        })?;
        let vals: Vec<&str> = header.split_whitespace().collect();
        if vals.len() < 7 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid warp file header",
            ));
        }

        let version: i32 = vals[0].parse().unwrap_or(1);
        let flags: i32 = vals[1].parse().unwrap_or(0);
        let nx: i32 = vals[2].parse().unwrap_or(0);
        let ny: i32 = vals[3].parse().unwrap_or(0);
        let binning: i32 = vals[4].parse().unwrap_or(1);
        let pixel_size: f32 = vals[5].parse().unwrap_or(1.0);
        let num_sections: usize = vals[6].parse().unwrap_or(0);

        let mut sections = Vec::with_capacity(num_sections);

        for _ in 0..num_sections {
            // Section header: z nControl
            let sec_header = match lines.next() {
                Some(l) => l,
                None => break,
            };
            let svals: Vec<&str> = sec_header.split_whitespace().collect();
            if svals.len() < 2 {
                break;
            }
            let z: i32 = svals[0].parse().unwrap_or(0);
            let n_control: usize = svals[1].parse().unwrap_or(0);

            let mut control_x = Vec::with_capacity(n_control);
            let mut control_y = Vec::with_capacity(n_control);
            let mut transforms = Vec::with_capacity(n_control);

            for _ in 0..n_control {
                let line = match lines.next() {
                    Some(l) => l,
                    None => break,
                };
                let fvals: Vec<f32> = line
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if fvals.len() >= 8 {
                    control_x.push(fvals[0]);
                    control_y.push(fvals[1]);
                    transforms.push(LinearTransform {
                        a11: fvals[2],
                        a12: fvals[3],
                        a21: fvals[4],
                        a22: fvals[5],
                        dx: fvals[6],
                        dy: fvals[7],
                    });
                }
            }

            sections.push(WarpTransform {
                z,
                nx,
                ny,
                control_x,
                control_y,
                transforms,
            });
        }

        Ok(WarpFile {
            nx,
            ny,
            binning,
            pixel_size,
            version,
            flags,
            sections,
        })
    }

    /// Write a warp file.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut f = std::fs::File::create(path.as_ref())?;

        writeln!(
            f,
            "{} {} {} {} {} {} {}",
            self.version,
            self.flags,
            self.nx,
            self.ny,
            self.binning,
            self.pixel_size,
            self.sections.len()
        )?;

        for sec in &self.sections {
            writeln!(f, "{} {}", sec.z, sec.transforms.len())?;
            for i in 0..sec.transforms.len() {
                let xf = &sec.transforms[i];
                writeln!(
                    f,
                    "{:.2} {:.2} {:.7} {:.7} {:.7} {:.7} {:.3} {:.3}",
                    sec.control_x[i],
                    sec.control_y[i],
                    xf.a11,
                    xf.a12,
                    xf.a21,
                    xf.a22,
                    xf.dx,
                    xf.dy
                )?;
            }
        }

        Ok(())
    }
}
