/// A 2D point for triangulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

/// A triangle defined by three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

/// Result of Delaunay triangulation.
#[derive(Debug, Clone)]
pub struct Triangulation {
    pub points: Vec<Point2d>,
    pub triangles: Vec<Triangle>,
}

impl Triangulation {
    /// Find which triangle contains the given point.
    /// Returns the triangle index, or None if outside the convex hull.
    pub fn find_containing_triangle(&self, x: f64, y: f64) -> Option<usize> {
        for (i, tri) in self.triangles.iter().enumerate() {
            if self.point_in_triangle(x, y, tri) {
                return Some(i);
            }
        }
        None
    }

    fn point_in_triangle(&self, px: f64, py: f64, tri: &Triangle) -> bool {
        let a = &self.points[tri.a];
        let b = &self.points[tri.b];
        let c = &self.points[tri.c];

        let d1 = sign(px, py, a.x, a.y, b.x, b.y);
        let d2 = sign(px, py, b.x, b.y, c.x, c.y);
        let d3 = sign(px, py, c.x, c.y, a.x, a.y);

        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

        !(has_neg && has_pos)
    }
}

fn sign(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
}

/// Compute the Delaunay triangulation of a set of 2D points.
///
/// Uses the Bowyer-Watson incremental algorithm.
pub fn triangulate(points: &[Point2d]) -> Triangulation {
    if points.len() < 3 {
        return Triangulation {
            points: points.to_vec(),
            triangles: Vec::new(),
        };
    }

    // Find bounding box
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dmax = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Create super-triangle that contains all points
    let mut all_points = points.to_vec();
    let st0 = all_points.len();
    all_points.push(Point2d { x: mid_x - 20.0 * dmax, y: mid_y - dmax });
    all_points.push(Point2d { x: mid_x, y: mid_y + 20.0 * dmax });
    all_points.push(Point2d { x: mid_x + 20.0 * dmax, y: mid_y - dmax });

    let mut triangles = vec![Triangle { a: st0, b: st0 + 1, c: st0 + 2 }];

    // Insert points one at a time
    for i in 0..points.len() {
        let px = all_points[i].x;
        let py = all_points[i].y;

        // Find triangles whose circumcircle contains the new point
        let mut bad_triangles = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            if in_circumcircle(&all_points, tri, px, py) {
                bad_triangles.push(ti);
            }
        }

        // Find the boundary polygon (edges of bad triangles not shared)
        let mut polygon = Vec::new();
        for &ti in &bad_triangles {
            let tri = &triangles[ti];
            let edges = [(tri.a, tri.b), (tri.b, tri.c), (tri.c, tri.a)];
            for &(ea, eb) in &edges {
                let shared = bad_triangles.iter().any(|&oi| {
                    oi != ti && {
                        let ot = &triangles[oi];
                        edge_in_triangle(ea, eb, ot)
                    }
                });
                if !shared {
                    polygon.push((ea, eb));
                }
            }
        }

        // Remove bad triangles (reverse order to maintain indices)
        let mut bad_sorted = bad_triangles.clone();
        bad_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for ti in bad_sorted {
            triangles.swap_remove(ti);
        }

        // Create new triangles from polygon edges to new point
        for &(ea, eb) in &polygon {
            triangles.push(Triangle { a: ea, b: eb, c: i });
        }
    }

    // Remove triangles that reference super-triangle vertices
    triangles.retain(|t| t.a < st0 && t.b < st0 && t.c < st0);

    Triangulation {
        points: points.to_vec(),
        triangles,
    }
}

fn in_circumcircle(points: &[Point2d], tri: &Triangle, px: f64, py: f64) -> bool {
    let a = &points[tri.a];
    let b = &points[tri.b];
    let c = &points[tri.c];

    // Ensure consistent orientation: compute signed area
    let cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);

    let (a, b, c) = if cross < 0.0 {
        (b, a, c) // swap to make CCW
    } else {
        (a, b, c)
    };

    let ax = a.x - px;
    let ay = a.y - py;
    let bx = b.x - px;
    let by = b.y - py;
    let cx = c.x - px;
    let cy = c.y - py;

    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx);

    det > 0.0
}

fn edge_in_triangle(ea: usize, eb: usize, t: &Triangle) -> bool {
    let verts = [t.a, t.b, t.c];
    let has_a = verts.contains(&ea);
    let has_b = verts.contains(&eb);
    has_a && has_b
}
