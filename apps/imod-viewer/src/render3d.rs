//! Software 3D renderer for IMOD models.
//!
//! Renders meshes with flat shading and contours as 3D lines into an RGBA
//! pixel buffer, using a Z-buffer for hidden-surface removal.

use imod_model::ImodModel;
use slint::{Image, Rgba8Pixel, SharedPixelBuffer};

/// A software 3D renderer that produces RGBA images of IMOD models.
pub struct Renderer3D {
    width: usize,
    height: usize,
    /// Rotation around X axis (degrees)
    pub rot_x: f32,
    /// Rotation around Y axis (degrees)
    pub rot_y: f32,
    /// Zoom factor (1.0 = fit model in view)
    pub zoom: f32,
    /// Center of rotation (model space)
    center: [f32; 3],
    /// Model bounding-box radius (for auto-fit)
    radius: f32,
    zbuf: Vec<f32>,
    pixels: Vec<u8>,
}

impl Renderer3D {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            rot_x: 20.0,
            rot_y: 30.0,
            zoom: 1.0,
            center: [0.0; 3],
            radius: 1.0,
            zbuf: vec![f32::INFINITY; n],
            pixels: vec![0u8; n * 4],
        }
    }

    /// Resize the output buffer if dimensions changed.
    pub fn resize(&mut self, w: usize, h: usize) {
        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            let n = w * h;
            self.zbuf.resize(n, f32::INFINITY);
            self.pixels.resize(n * 4, 0);
        }
    }

    /// Trackball-style rotation: dx/dy in pixels of mouse drag.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.rot_y += dx * 0.5;
        self.rot_x += dy * 0.5;
        // Clamp X rotation to avoid gimbal flipping
        self.rot_x = self.rot_x.clamp(-89.0, 89.0);
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom.max(0.01);
    }

    /// Render the model and return a Slint Image.
    pub fn render_model(&mut self, model: &ImodModel) -> Image {
        self.compute_bounds(model);
        self.clear();
        self.draw_model(model);

        let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(self.width as u32, self.height as u32);
        buf.make_mut_bytes().copy_from_slice(&self.pixels);
        Image::from_rgba8(buf)
    }

    // ---- internals ----

    fn clear(&mut self) {
        // Dark gray background #222222
        for i in 0..self.width * self.height {
            let off = i * 4;
            self.pixels[off] = 0x22;
            self.pixels[off + 1] = 0x22;
            self.pixels[off + 2] = 0x22;
            self.pixels[off + 3] = 0xFF;
            self.zbuf[i] = f32::INFINITY;
        }
    }

    fn compute_bounds(&mut self, model: &ImodModel) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let mut any = false;

        for obj in &model.objects {
            for cont in &obj.contours {
                for pt in &cont.points {
                    let p = [pt.x, pt.y, pt.z];
                    for i in 0..3 {
                        min[i] = min[i].min(p[i]);
                        max[i] = max[i].max(p[i]);
                    }
                    any = true;
                }
            }
            for mesh in &obj.meshes {
                for v in &mesh.vertices {
                    let p = [v.x, v.y, v.z];
                    for i in 0..3 {
                        min[i] = min[i].min(p[i]);
                        max[i] = max[i].max(p[i]);
                    }
                    any = true;
                }
            }
        }

        if !any {
            self.center = [0.0; 3];
            self.radius = 1.0;
            return;
        }

        for i in 0..3 {
            self.center[i] = (min[i] + max[i]) * 0.5;
        }
        let dx = max[0] - min[0];
        let dy = max[1] - min[1];
        let dz = max[2] - min[2];
        self.radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;
        if self.radius < 1e-6 {
            self.radius = 1.0;
        }
    }

    /// Transform a model-space point to screen (x, y, depth).
    /// Uses perspective-like orthographic: rotate around center, then scale to fit.
    fn project(&self, p: [f32; 3]) -> (f32, f32, f32) {
        // Translate to center
        let x = p[0] - self.center[0];
        let y = p[1] - self.center[1];
        let z = p[2] - self.center[2];

        // Rotate around Y axis
        let ay = self.rot_y.to_radians();
        let (sy, cy) = ay.sin_cos();
        let x1 = x * cy + z * sy;
        let z1 = -x * sy + z * cy;
        let y1 = y;

        // Rotate around X axis
        let ax = self.rot_x.to_radians();
        let (sx, cx) = ax.sin_cos();
        let y2 = y1 * cx - z1 * sx;
        let z2 = y1 * sx + z1 * cx;
        let x2 = x1;

        // Scale to fit in viewport
        let half = self.width.min(self.height) as f32 * 0.5;
        let scale = half / self.radius * self.zoom * 0.9;

        let sx_screen = self.width as f32 * 0.5 + x2 * scale;
        let sy_screen = self.height as f32 * 0.5 - y2 * scale; // flip Y
        (sx_screen, sy_screen, z2)
    }

    fn set_pixel_z(&mut self, x: i32, y: i32, depth: f32, r: u8, g: u8, b: u8) {
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
            return;
        }
        let idx = y as usize * self.width + x as usize;
        if depth < self.zbuf[idx] {
            self.zbuf[idx] = depth;
            let off = idx * 4;
            self.pixels[off] = r;
            self.pixels[off + 1] = g;
            self.pixels[off + 2] = b;
            self.pixels[off + 3] = 255;
        }
    }

    fn draw_triangle(&mut self, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], r: u8, g: u8, b: u8) {
        // Compute face normal in model space for Lambertian shading
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-12 {
            return; // degenerate
        }
        let (nx, ny, nz) = (nx / len, ny / len, nz / len);

        // Rotate the normal the same way we rotate vertices
        let ay = self.rot_y.to_radians();
        let (sy, cy) = ay.sin_cos();
        let nx1 = nx * cy + nz * sy;
        let nz1 = -nx * sy + nz * cy;
        let ny1 = ny;
        let ax = self.rot_x.to_radians();
        let (sx, cx) = ax.sin_cos();
        let _ny2 = ny1 * cx - nz1 * sx;
        let nz2 = ny1 * sx + nz1 * cx;
        let _nx2 = nx1;

        // Light direction: towards viewer (0, 0, 1) in screen space
        let ndotl = nz2.abs(); // two-sided lighting
        // Ambient + diffuse
        let shade = 0.25 + 0.75 * ndotl;
        let sr = (r as f32 * shade).min(255.0) as u8;
        let sg = (g as f32 * shade).min(255.0) as u8;
        let sb = (b as f32 * shade).min(255.0) as u8;

        // Project vertices
        let (sx0, sy0, d0) = self.project(v0);
        let (sx1, sy1, d1) = self.project(v1);
        let (sx2, sy2, d2) = self.project(v2);

        // Rasterize using scanline
        self.fill_triangle_scanline(sx0, sy0, d0, sx1, sy1, d1, sx2, sy2, d2, sr, sg, sb);
    }

    fn fill_triangle_scanline(
        &mut self,
        x0: f32, y0: f32, z0: f32,
        x1: f32, y1: f32, z1: f32,
        x2: f32, y2: f32, z2: f32,
        r: u8, g: u8, b: u8,
    ) {
        // Sort by y
        let mut pts = [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2)];
        if pts[0].1 > pts[1].1 { pts.swap(0, 1); }
        if pts[0].1 > pts[2].1 { pts.swap(0, 2); }
        if pts[1].1 > pts[2].1 { pts.swap(1, 2); }

        let (ax, ay, az) = pts[0];
        let (bx, by, bz) = pts[1];
        let (cx, cy, cz) = pts[2];

        let total_h = cy - ay;
        if total_h < 0.5 {
            return;
        }

        let y_start = ay.ceil().max(0.0) as i32;
        let y_end = cy.floor().min(self.height as f32 - 1.0) as i32;

        for y in y_start..=y_end {
            let yf = y as f32 + 0.5;
            // Interpolate edges: long edge A->C, short edges A->B then B->C
            let t_ac = (yf - ay) / total_h;
            let x_ac = ax + (cx - ax) * t_ac;
            let z_ac = az + (cz - az) * t_ac;

            let (x_short, z_short) = if yf < by {
                let h = by - ay;
                if h < 0.5 { continue; }
                let t = (yf - ay) / h;
                (ax + (bx - ax) * t, az + (bz - az) * t)
            } else {
                let h = cy - by;
                if h < 0.5 { continue; }
                let t = (yf - by) / h;
                (bx + (cx - bx) * t, bz + (cz - bz) * t)
            };

            let (lx, lz, rx, rz) = if x_ac < x_short {
                (x_ac, z_ac, x_short, z_short)
            } else {
                (x_short, z_short, x_ac, z_ac)
            };

            let xi_start = lx.ceil().max(0.0) as i32;
            let xi_end = rx.floor().min(self.width as f32 - 1.0) as i32;
            let span = rx - lx;

            for xi in xi_start..=xi_end {
                let t = if span > 0.5 { (xi as f32 + 0.5 - lx) / span } else { 0.5 };
                let depth = lz + (rz - lz) * t;
                self.set_pixel_z(xi, y, depth, r, g, b);
            }
        }
    }

    fn draw_line_3d(&mut self, p0: [f32; 3], p1: [f32; 3], r: u8, g: u8, b: u8) {
        let (sx0, sy0, d0) = self.project(p0);
        let (sx1, sy1, d1) = self.project(p1);

        // Bresenham with depth interpolation
        let dx = (sx1 - sx0).abs();
        let dy = (sy1 - sy0).abs();
        let steps = dx.max(dy).max(1.0) as i32;

        for i in 0..=steps {
            let t = if steps > 0 { i as f32 / steps as f32 } else { 0.0 };
            let x = sx0 + (sx1 - sx0) * t;
            let y = sy0 + (sy1 - sy0) * t;
            let d = d0 + (d1 - d0) * t;
            // Draw a 2-pixel wide line for visibility
            for ox in -1..=1 {
                for oy in -1..=1 {
                    self.set_pixel_z(x as i32 + ox, y as i32 + oy, d, r, g, b);
                }
            }
        }
    }

    fn draw_point_3d(&mut self, p: [f32; 3], radius: f32, r: u8, g: u8, b: u8) {
        let (sx, sy, depth) = self.project(p);
        let half = self.width.min(self.height) as f32 * 0.5;
        let scale = half / self.radius * self.zoom * 0.9;
        let screen_r = (radius * scale).max(2.0);
        let ir = screen_r.ceil() as i32;

        for dy in -ir..=ir {
            for dx in -ir..=ir {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= screen_r * screen_r {
                    self.set_pixel_z(sx as i32 + dx, sy as i32 + dy, depth, r, g, b);
                }
            }
        }
    }

    fn draw_model(&mut self, model: &ImodModel) {
        for obj in &model.objects {
            let r = (obj.red * 255.0) as u8;
            let g = (obj.green * 255.0) as u8;
            let b = (obj.blue * 255.0) as u8;

            // Draw meshes (triangle strips / polygons from IMOD index list)
            for mesh in &obj.meshes {
                self.draw_mesh(&mesh.vertices, &mesh.indices, r, g, b);
            }

            // Draw contours as 3D polylines
            for cont in &obj.contours {
                let pts: Vec<[f32; 3]> = cont.points.iter().map(|p| [p.x, p.y, p.z]).collect();
                // Draw connecting lines
                for w in pts.windows(2) {
                    self.draw_line_3d(w[0], w[1], r, g, b);
                }
                // Draw individual points as small circles
                let pt_radius = if obj.pdrawsize > 0 { obj.pdrawsize as f32 } else { 1.0 };
                for &p in &pts {
                    self.draw_point_3d(p, pt_radius, r, g, b);
                }
            }
        }
    }

    /// Draw an IMOD mesh. The index list uses special negative sentinel values:
    /// -1  = end of list
    /// -20 = normal indices follow (skip pairs of normal_idx, vert_idx)
    /// -21 = end of polygon / concave polygon marker
    /// -22 = end of polygon (ENDPOLY)
    /// -23 = begin polygon (BGNPOLY) -- next index is normal, then vertex indices
    /// -24 = begin big polygon (BGNBIGPOLY)
    /// -25 = normal index (POLYNORM) -- next two are normal indices then vertex pairs
    fn draw_mesh(&mut self, vertices: &[imod_core::Point3f], indices: &[i32], r: u8, g: u8, b: u8) {
        if vertices.is_empty() || indices.is_empty() {
            return;
        }
        let nv = vertices.len() as i32;

        // Simple approach: scan for triangle fans / strips.
        // IMOD meshes commonly use BGNPOLY (-23) followed by vertex indices,
        // ending with ENDPOLY (-22) or negative sentinels.
        // We also handle plain index lists terminated by -1 (triangle list).
        let mut i = 0;
        while i < indices.len() {
            let cmd = indices[i];
            if cmd == -1 {
                break; // end of list
            }
            if cmd == -23 || cmd == -24 {
                // BGNPOLY / BGNBIGPOLY
                // Collect vertex indices until next sentinel
                i += 1;
                let mut poly_verts: Vec<[f32; 3]> = Vec::new();
                while i < indices.len() {
                    let idx = indices[i];
                    if idx < 0 {
                        break;
                    }
                    if idx < nv {
                        let v = &vertices[idx as usize];
                        poly_verts.push([v.x, v.y, v.z]);
                    }
                    i += 1;
                }
                // Triangulate polygon as a fan from first vertex
                if poly_verts.len() >= 3 {
                    for j in 1..poly_verts.len() - 1 {
                        self.draw_triangle(poly_verts[0], poly_verts[j], poly_verts[j + 1], r, g, b);
                    }
                }
                // Skip the sentinel
                if i < indices.len() && (indices[i] == -22 || indices[i] == -21) {
                    i += 1;
                }
            } else if cmd == -25 {
                // POLYNORM: pairs of (normal_idx, vertex_idx)
                i += 1;
                let mut poly_verts: Vec<[f32; 3]> = Vec::new();
                while i + 1 < indices.len() {
                    let ni = indices[i];
                    let vi = indices[i + 1];
                    if ni < 0 || vi < 0 { break; }
                    if vi < nv {
                        let v = &vertices[vi as usize];
                        poly_verts.push([v.x, v.y, v.z]);
                    }
                    i += 2;
                }
                if poly_verts.len() >= 3 {
                    for j in 1..poly_verts.len() - 1 {
                        self.draw_triangle(poly_verts[0], poly_verts[j], poly_verts[j + 1], r, g, b);
                    }
                }
                if i < indices.len() && (indices[i] == -22 || indices[i] == -21) {
                    i += 1;
                }
            } else if cmd == -20 {
                // Normal section -- skip
                i += 1;
            } else if cmd >= 0 {
                // Plain triangle list: take triples of indices
                let mut tri_indices: Vec<i32> = Vec::new();
                while i < indices.len() && indices[i] >= 0 {
                    tri_indices.push(indices[i]);
                    i += 1;
                }
                // Render as triangles (every 3 indices)
                for chunk in tri_indices.chunks(3) {
                    if chunk.len() == 3 {
                        let i0 = chunk[0];
                        let i1 = chunk[1];
                        let i2 = chunk[2];
                        if i0 < nv && i1 < nv && i2 < nv {
                            let v0 = &vertices[i0 as usize];
                            let v1 = &vertices[i1 as usize];
                            let v2 = &vertices[i2 as usize];
                            self.draw_triangle(
                                [v0.x, v0.y, v0.z],
                                [v1.x, v1.y, v1.z],
                                [v2.x, v2.y, v2.z],
                                r, g, b,
                            );
                        }
                    }
                }
                // Skip end sentinel
                if i < indices.len() && indices[i] == -1 {
                    break;
                }
            } else {
                // Unknown sentinel, skip
                i += 1;
            }
        }
    }
}
