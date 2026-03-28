use clap::Parser;
use imod_model::read_model;
use std::io::Write;

/// Solve for the transformation to match two volumes based on corresponding
/// point positions from corrsearch3d or manually placed fiducials.
///
/// Reads a model with displacement vectors (pairs of points per contour)
/// and fits a 3D affine transform or reports the needed shifts.
#[derive(Parser)]
#[command(name = "solvematch", about = "Solve for 3D volume matching transform")]
struct Args {
    /// Model file with correspondence points (from corrsearch3d)
    #[arg(short = 'm', long)]
    model: String,

    /// Output transform file (3x4 matrix)
    #[arg(short = 'o', long)]
    output: String,

    /// Maximum residual to keep a point
    #[arg(short = 'r', long, default_value_t = 20.0)]
    max_residual: f32,
}

fn main() {
    let args = Args::parse();

    let model = read_model(&args.model).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    // Extract point pairs: each contour has 2 points (source, target)
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for obj in &model.objects {
        for cont in &obj.contours {
            if cont.points.len() >= 2 {
                let p0 = &cont.points[0];
                let p1 = &cont.points[1];
                src.push([p0.x as f64, p0.y as f64, p0.z as f64]);
                dst.push([p1.x as f64, p1.y as f64, p1.z as f64]);
            }
        }
    }

    eprintln!("solvematch: {} correspondence pairs", src.len());

    if src.len() < 4 {
        eprintln!("Error: need at least 4 pairs");
        std::process::exit(1);
    }

    // Compute centroids
    let n = src.len() as f64;
    let mut cs = [0.0f64; 3];
    let mut cd = [0.0f64; 3];
    for i in 0..src.len() {
        for j in 0..3 { cs[j] += src[i][j]; cd[j] += dst[i][j]; }
    }
    for j in 0..3 { cs[j] /= n; cd[j] /= n; }

    // Center
    let ds: Vec<[f64; 3]> = src.iter().map(|p| [p[0] - cs[0], p[1] - cs[1], p[2] - cs[2]]).collect();
    let dd: Vec<[f64; 3]> = dst.iter().map(|p| [p[0] - cd[0], p[1] - cd[1], p[2] - cd[2]]).collect();

    // A^T*A and B^T*A
    let mut ata = [[0.0f64; 3]; 3];
    let mut bta = [[0.0f64; 3]; 3];
    for i in 0..src.len() {
        for r in 0..3 {
            for c in 0..3 {
                ata[r][c] += ds[i][r] * ds[i][c];
                bta[r][c] += dd[i][r] * ds[i][c];
            }
        }
    }

    let inv = invert_3x3(&ata);
    let mut m = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 { m[r][c] += bta[r][k] * inv[k][c]; }
        }
    }

    let mut t = [0.0f64; 3];
    for r in 0..3 {
        t[r] = cd[r];
        for c in 0..3 { t[r] -= m[r][c] * cs[c]; }
    }

    // Residuals
    let mut total = 0.0f64;
    for i in 0..src.len() {
        let mut pred = [0.0f64; 3];
        for r in 0..3 {
            pred[r] = t[r];
            for c in 0..3 { pred[r] += m[r][c] * src[i][c]; }
        }
        let res = ((dst[i][0] - pred[0]).powi(2) + (dst[i][1] - pred[1]).powi(2) + (dst[i][2] - pred[2]).powi(2)).sqrt();
        total += res * res;
    }
    let rms = (total / src.len() as f64).sqrt();
    eprintln!("solvematch: RMS residual = {:.3} pixels", rms);

    // Write output
    let mut f = std::fs::File::create(&args.output).unwrap();
    for r in 0..3 {
        writeln!(f, "{:12.7} {:12.7} {:12.7} {:12.3}", m[r][0], m[r][1], m[r][2], t[r]).unwrap();
    }
    eprintln!("solvematch: wrote transform to {}", args.output);
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = 1.0 / det;
    [
        [(m[1][1]*m[2][2]-m[1][2]*m[2][1])*d, (m[0][2]*m[2][1]-m[0][1]*m[2][2])*d, (m[0][1]*m[1][2]-m[0][2]*m[1][1])*d],
        [(m[1][2]*m[2][0]-m[1][0]*m[2][2])*d, (m[0][0]*m[2][2]-m[0][2]*m[2][0])*d, (m[0][2]*m[1][0]-m[0][0]*m[1][2])*d],
        [(m[1][0]*m[2][1]-m[1][1]*m[2][0])*d, (m[0][1]*m[2][0]-m[0][0]*m[2][1])*d, (m[0][0]*m[1][1]-m[0][1]*m[1][0])*d],
    ]
}
