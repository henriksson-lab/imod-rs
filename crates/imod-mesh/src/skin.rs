use imod_core::Point3f;

/// A triangle mesh built from contours on consecutive Z sections.
#[derive(Debug, Clone)]
pub struct ContourMesh {
    pub vertices: Vec<Point3f>,
    pub normals: Vec<Point3f>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

/// A 2D contour on a given Z section.
#[derive(Debug, Clone)]
pub struct Contour2d {
    pub points: Vec<[f32; 2]>,
    pub z: f32,
    pub closed: bool,
}

/// Skin two contours on adjacent Z sections by building a triangle strip
/// between them. Uses a simple nearest-point correspondence.
pub fn skin_contours(c1: &Contour2d, c2: &Contour2d) -> ContourMesh {
    if c1.points.is_empty() || c2.points.is_empty() {
        return ContourMesh {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        };
    }

    let n1 = c1.points.len();
    let n2 = c2.points.len();

    // Build vertex list: c1 points then c2 points
    let mut vertices = Vec::with_capacity(n1 + n2);
    for p in &c1.points {
        vertices.push(Point3f { x: p[0], y: p[1], z: c1.z });
    }
    for p in &c2.points {
        vertices.push(Point3f { x: p[0], y: p[1], z: c2.z });
    }

    // Walk both contours, creating triangles
    let mut indices = Vec::new();
    let offset2 = n1 as u32;
    let mut i1: usize = 0;
    let mut i2: usize = 0;

    while i1 < n1 || i2 < n2 {
        let next1 = if i1 < n1 { (i1 + 1) % n1 } else { 0 };
        let next2 = if i2 < n2 { (i2 + 1) % n2 } else { 0 };

        if i1 >= n1 {
            // Advance on contour 2
            indices.push(i1 as u32 % n1 as u32);
            indices.push(offset2 + i2 as u32);
            indices.push(offset2 + next2 as u32);
            i2 += 1;
        } else if i2 >= n2 {
            // Advance on contour 1
            indices.push(i1 as u32);
            indices.push(offset2 + i2 as u32 % n2 as u32);
            indices.push(next1 as u32);
            i1 += 1;
        } else {
            // Choose which contour to advance based on shorter diagonal
            let p1 = &c1.points[i1];
            let p2_curr = &c2.points[i2];
            let p1_next = &c1.points[next1];
            let p2_next = &c2.points[next2];

            let d_advance1 = dist2d(p1_next, p2_curr);
            let d_advance2 = dist2d(p1, p2_next);

            if d_advance1 <= d_advance2 {
                indices.push(i1 as u32);
                indices.push(offset2 + i2 as u32);
                indices.push(next1 as u32);
                i1 += 1;
            } else {
                indices.push(i1 as u32);
                indices.push(offset2 + i2 as u32);
                indices.push(offset2 + next2 as u32);
                i2 += 1;
            }
        }
    }

    // Compute flat normals per vertex (placeholder: use Z direction)
    let normals = vec![Point3f { x: 0.0, y: 0.0, z: 1.0 }; vertices.len()];

    ContourMesh {
        vertices,
        normals,
        indices,
    }
}

fn dist2d(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Reduce the number of points in a contour using the Douglas-Peucker algorithm.
pub fn simplify_contour(points: &[[f32; 2]], tolerance: f32) -> Vec<[f32; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    douglas_peucker(points, 0, points.len() - 1, tolerance, &mut keep);
    points
        .iter()
        .zip(keep.iter())
        .filter(|&(_, k)| *k)
        .map(|(&p, _)| p)
        .collect()
}

fn douglas_peucker(points: &[[f32; 2]], start: usize, end: usize, tol: f32, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let mut max_dist = 0.0f32;
    let mut max_idx = start;
    let ax = points[start][0];
    let ay = points[start][1];
    let bx = points[end][0];
    let by = points[end][1];
    let len_sq = (bx - ax) * (bx - ax) + (by - ay) * (by - ay);

    for i in (start + 1)..end {
        let d = if len_sq < 1e-10 {
            let dx = points[i][0] - ax;
            let dy = points[i][1] - ay;
            (dx * dx + dy * dy).sqrt()
        } else {
            let t = ((points[i][0] - ax) * (bx - ax) + (points[i][1] - ay) * (by - ay)) / len_sq;
            let t = t.clamp(0.0, 1.0);
            let px = ax + t * (bx - ax);
            let py = ay + t * (by - ay);
            let dx = points[i][0] - px;
            let dy = points[i][1] - py;
            (dx * dx + dy * dy).sqrt()
        };
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }
    if max_dist > tol {
        keep[max_idx] = true;
        douglas_peucker(points, start, max_idx, tol, keep);
        douglas_peucker(points, max_idx, end, tol, keep);
    }
}
