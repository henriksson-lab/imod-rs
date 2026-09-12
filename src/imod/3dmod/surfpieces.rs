//! Translation of `IMOD/3dmod/surfpieces.cpp` and `surfpieces.h`.
//!
//! This unit partitions an isosurface triangle list into vertex-connected
//! pieces, orders the pieces by their triangle count, and rewrites the
//! triangle list in that order.  `vertex_xyz` is deliberately retained in
//! [`SurfacePieces::new`]: the upstream area calculation that used it is
//! commented out, and the live source orders by triangle count instead.
#![allow(dead_code)]

use crate::imod::libimod::imodel::Ipoint;

/// `Surface_Piece` (`surfpieces.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfacePiece {
    pub area: f32,
    pub v_list: Vec<i32>,
    pub t_list: Vec<i32>,
}

impl SurfacePiece {
    /// `Surface_Piece::Surface_Piece`.
    pub fn new(v_list: Vec<i32>, t_list: Vec<i32>) -> Self {
        Self {
            area: 0.0,
            v_list,
            t_list,
        }
    }
}

/// `Surface_Pieces` (`surfpieces.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfacePieces {
    pub pieces: Vec<SurfacePiece>,
}

/// `maximum_triangle_vertex_index`.
pub fn maximum_triangle_vertex_index(tarray: &[i32], tc: i32) -> i32 {
    let mut vmax = 0;
    let s0 = 3;
    let s1 = 1;
    for t in 0..tc as usize {
        let v0 = tarray[s0 * t];
        let v1 = tarray[s0 * t + s1];
        let v2 = tarray[s0 * t + 2 * s1];
        if v0 > vmax {
            vmax = v0;
        }
        if v1 > vmax {
            vmax = v1;
        }
        if v2 > vmax {
            vmax = v2;
        }
    }
    vmax
}

/// `min_connected_vertex`.
pub fn min_connected_vertex(v: i32, vmap: &mut [i32]) -> i32 {
    let mut v0 = v;
    while vmap[v0 as usize] < v0 {
        v0 = vmap[v0 as usize];
    }
    // Collapse the chain, exactly as the source does, for future traversals.
    let mut v1 = v;
    while v1 > v0 {
        let next = vmap[v1 as usize];
        vmap[v1 as usize] = v0;
        v1 = next;
    }
    v0
}

/// `calculate_components`.
pub fn calculate_components(tarray: &[i32], tc: i32, vmap: &mut [i32], vc: i32) -> i32 {
    for v in 0..vc as usize {
        vmap[v] = vc;
    }

    let s0 = 3;
    let s1 = 1;
    for t in 0..tc as usize {
        let v0 = min_connected_vertex(tarray[s0 * t], vmap);
        let v1 = min_connected_vertex(tarray[s0 * t + s1], vmap);
        let v2 = min_connected_vertex(tarray[s0 * t + 2 * s1], vmap);
        let v01 = if v0 < v1 { v0 } else { v1 };
        let vmin = if v2 < v01 { v2 } else { v01 };
        vmap[v0 as usize] = vmin;
        vmap[v1 as usize] = vmin;
        vmap[v2 as usize] = vmin;
    }

    let mut cc = 0;
    for v in 0..vc as usize {
        let vm = vmap[v];
        if vm < v as i32 {
            vmap[v] = vmap[vm as usize];
        } else if vm == v as i32 {
            vmap[v] = cc;
            cc += 1;
        }
    }
    cc
}

/// `sortbyArea`.
pub fn sort_by_area(left: &SurfacePiece, right: &SurfacePiece) -> bool {
    left.area < right.area
}

impl SurfacePieces {
    /// `Surface_Pieces::Surface_Pieces`.
    ///
    /// `sorted_triangle` is the caller-owned equivalent of the source output
    /// buffer.  As in the C++ routine, it must have room for `3 * tc` indices.
    pub fn new(
        _vertex_xyz: &[Ipoint],
        tarray: &[i32],
        tc: i32,
        sorted_triangle: &mut [i32],
    ) -> Self {
        let mut surface_pieces = Self::default();
        if tc <= 0 {
            return surface_pieces;
        }

        let vc = maximum_triangle_vertex_index(tarray, tc) + 1;
        let mut vmap = vec![0; vc as usize];
        let cc = calculate_components(tarray, tc, &mut vmap, vc);

        for _c in 0..cc {
            surface_pieces
                .pieces
                .push(SurfacePiece::new(Vec::new(), Vec::new()));
        }

        for v in 0..vc as usize {
            if vmap[v] < vc {
                surface_pieces.pieces[vmap[v] as usize]
                    .v_list
                    .push(v as i32);
            }
        }

        let s0 = 3;
        for t in 0..tc as usize {
            surface_pieces.pieces[vmap[tarray[s0 * t] as usize] as usize]
                .t_list
                .push(t as i32);
        }

        for ci in 0..cc as usize {
            // The geometric area calculation is commented out in the original.
            surface_pieces.pieces[ci].area = surface_pieces.pieces[ci].t_list.len() as f32;
        }

        // This corresponds to C++ `std::sort`, which is likewise unstable for
        // pieces with equal areas.
        surface_pieces
            .pieces
            .sort_unstable_by(|left, right| left.area.total_cmp(&right.area));

        let mut t_counter = 0;
        for ci in 0..cc as usize {
            let n_t = surface_pieces.pieces[ci].t_list.len();
            let curr_t_list = &surface_pieces.pieces[ci].t_list;
            for t in 0..n_t {
                let triangle = curr_t_list[t] as usize;
                sorted_triangle[t_counter] = tarray[3 * triangle];
                t_counter += 1;
                sorted_triangle[t_counter] = tarray[3 * triangle + 1];
                t_counter += 1;
                sorted_triangle[t_counter] = tarray[3 * triangle + 2];
                t_counter += 1;
            }
        }
        surface_pieces
    }
}

/// `Surface_Pieces::~Surface_Pieces`.
///
/// The vectors own the allocations corresponding to the C++ heap vectors, so
/// Rust drops them after this source-shaped destructor hook returns.
impl Drop for SurfacePieces {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_components_and_orders_smallest_piece_first() {
        // Two triangles form one piece; the last triangle is disconnected.
        let triangles = [0, 1, 2, 2, 1, 3, 4, 5, 6];
        let mut sorted = [0; 9];
        let pieces = SurfacePieces::new(&[], &triangles, 3, &mut sorted);

        assert_eq!(pieces.pieces.len(), 2);
        assert_eq!(pieces.pieces[0].area, 1.0);
        assert_eq!(pieces.pieces[0].t_list, vec![2]);
        assert_eq!(pieces.pieces[1].area, 2.0);
        assert_eq!(pieces.pieces[1].t_list, vec![0, 1]);
        assert_eq!(sorted, [4, 5, 6, 0, 1, 2, 2, 1, 3]);
    }

    #[test]
    fn empty_triangle_input_has_no_pieces_or_output() {
        let mut sorted = [];
        let pieces = SurfacePieces::new(&[], &[], 0, &mut sorted);
        assert!(pieces.pieces.is_empty());
    }

    #[test]
    fn chain_compression_finds_lowest_connected_vertex() {
        let mut vmap = [0, 0, 1, 2];
        assert_eq!(min_connected_vertex(3, &mut vmap), 0);
        assert_eq!(vmap, [0, 0, 0, 0]);
    }
}
