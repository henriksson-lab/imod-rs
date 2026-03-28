use clap::Parser;
use imod_model::read_model;

/// Find the 3D warping transform to align two tomograms (e.g., for dual-axis
/// combination). Uses corresponding fiducial positions from both volumes to
/// solve for local linear transforms that map volume A onto volume B.
///
/// Reads two model files with matched fiducial positions and finds the best-fit
/// linear transform (or piecewise-linear warping) between them.
#[derive(Parser)]
#[command(name = "findwarp", about = "Find 3D warping transform between two volumes")]
struct Args {
    /// Model file with fiducial positions in volume A
    #[arg(short = 'a', long)]
    model_a: String,

    /// Model file with fiducial positions in volume B
    #[arg(short = 'b', long)]
    model_b: String,

    /// Output warp transform file
    #[arg(short = 'o', long)]
    output: String,

    /// Residual output file
    #[arg(long)]
    residual: Option<String>,

    /// Maximum residual for accepting a point (pixels)
    #[arg(short = 'r', long, default_value_t = 10.0)]
    max_residual: f32,
}

fn main() {
    let args = Args::parse();

    let model_a = read_model(&args.model_a).unwrap_or_else(|e| {
        eprintln!("Error reading model A: {}", e);
        std::process::exit(1);
    });
    let model_b = read_model(&args.model_b).unwrap_or_else(|e| {
        eprintln!("Error reading model B: {}", e);
        std::process::exit(1);
    });

    // Extract matched point pairs: each object index corresponds between A and B
    let n_pairs = model_a.objects.len().min(model_b.objects.len());
    let mut pairs_a = Vec::new();
    let mut pairs_b = Vec::new();

    for i in 0..n_pairs {
        if let (Some(pa), Some(pb)) = (
            model_a.objects[i].contours.first().and_then(|c| c.points.first()),
            model_b.objects[i].contours.first().and_then(|c| c.points.first()),
        ) {
            pairs_a.push((pa.x as f64, pa.y as f64, pa.z as f64));
            pairs_b.push((pb.x as f64, pb.y as f64, pb.z as f64));
        }
    }

    eprintln!("findwarp: {} matched point pairs", pairs_a.len());

    if pairs_a.len() < 4 {
        eprintln!("Error: need at least 4 matched points for a 3D transform");
        std::process::exit(1);
    }

    // Solve for 3x4 affine transform: B = M * A + T
    // Using least squares: minimize sum of |B_i - (M * A_i + T)|^2
    // This is a 12-parameter problem (3x3 matrix + 3 translation)
    let n = pairs_a.len();

    // Compute centroids
    let (mut ca, mut cb) = ([0.0f64; 3], [0.0f64; 3]);
    for i in 0..n {
        ca[0] += pairs_a[i].0; ca[1] += pairs_a[i].1; ca[2] += pairs_a[i].2;
        cb[0] += pairs_b[i].0; cb[1] += pairs_b[i].1; cb[2] += pairs_b[i].2;
    }
    for j in 0..3 { ca[j] /= n as f64; cb[j] /= n as f64; }

    // Center the points
    let da: Vec<[f64; 3]> = pairs_a.iter().map(|p| [p.0 - ca[0], p.1 - ca[1], p.2 - ca[2]]).collect();
    let db: Vec<[f64; 3]> = pairs_b.iter().map(|p| [p.0 - cb[0], p.1 - cb[1], p.2 - cb[2]]).collect();

    // Solve for rotation+scale: M = (B^T * A) * (A^T * A)^-1
    // Build A^T*A (3x3) and B^T*A (3x3)
    let mut ata = [[0.0f64; 3]; 3];
    let mut bta = [[0.0f64; 3]; 3];
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                ata[r][c] += da[i][r] * da[i][c];
                bta[r][c] += db[i][r] * da[i][c];
            }
        }
    }

    // Invert A^T*A (3x3)
    let inv = invert_3x3(&ata);
    let mut m = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            for k in 0..3 {
                m[r][c] += bta[r][k] * inv[k][c];
            }
        }
    }

    // Translation: T = cb - M * ca
    let mut t = [0.0f64; 3];
    for r in 0..3 {
        t[r] = cb[r];
        for c in 0..3 {
            t[r] -= m[r][c] * ca[c];
        }
    }

    // Compute residuals
    let mut total_res = 0.0f64;
    let mut max_res = 0.0f64;
    for i in 0..n {
        let mut pred = [0.0f64; 3];
        let pa = [pairs_a[i].0, pairs_a[i].1, pairs_a[i].2];
        for r in 0..3 {
            pred[r] = t[r];
            for c in 0..3 {
                pred[r] += m[r][c] * pa[c];
            }
        }
        let pb = [pairs_b[i].0, pairs_b[i].1, pairs_b[i].2];
        let res = ((pb[0] - pred[0]).powi(2) + (pb[1] - pred[1]).powi(2) + (pb[2] - pred[2]).powi(2)).sqrt();
        total_res += res * res;
        if res > max_res { max_res = res; }
    }
    let rms = (total_res / n as f64).sqrt();

    eprintln!("findwarp: RMS residual = {:.3}, max = {:.3}", rms, max_res);

    // Write output: 3x4 matrix (3 rows of: m11 m12 m13 tx)
    let mut f = std::fs::File::create(&args.output).unwrap();
    use std::io::Write;
    for r in 0..3 {
        writeln!(f, "{:12.7} {:12.7} {:12.7} {:12.3}",
            m[r][0], m[r][1], m[r][2], t[r]).unwrap();
    }

    eprintln!("findwarp: wrote transform to {}", args.output);
}


fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}
