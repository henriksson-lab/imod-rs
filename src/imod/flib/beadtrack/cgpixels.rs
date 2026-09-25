//! Translation of `IMOD/flib/beadtrack/cgpixels.h`.
//!
//! Type-only header: the pixel lists and kernels for centroid finding, reached
//! in `tracksubs.cpp` through `static CGPixels *cp;` (`tracksubs.cpp:18`, set by
//! `tracksubsSetPointers`) and owned by `beadtrack.cpp` as a file-scope
//! `static CGPixels cgPixels;` (`beadtrack.cpp:32`).  The translation owns one
//! `CGPixels` in the program state and passes `&mut CGPixels` to the functions
//! whose C body dereferences `cp`; see `tiltalign/alivar.rs` for the shared
//! conventions (`Vec` for `B3DMALLOC`ed pointers, empty for `NULL`, zeroed
//! where the C is uninitialised).
//!
//! The two kernels are fixed arrays filled by `scaledGaussianKernel` with
//! maximum dimension 7 and 9 (`beadtrack.cpp:1310,1317`), stored row by row
//! in a flat array exactly as in C.  `edgeMedian` and `getEdgeSD` are `int`
//! flags assigned `true`/`false` (`tracksubs.cpp:690-694`); they stay `i32`.

/// Original: `struct CGPixels` (`cgpixels.h:3-28`).
///
/// `Default` is the zero-initialised file-scope static.  It is written by hand
/// because `[f32; 49]` and `[f32; 81]` have no `Default`.
#[derive(Clone, Debug)]
pub struct CGPixels {
    /// Original: `int *idyEdge` (`cgpixels.h:5`).
    pub idy_edge: Vec<i32>,
    /// Original: `int *idxEdge` (`cgpixels.h:6`).
    pub idx_edge: Vec<i32>,
    /// Original: `int *idyin` (`cgpixels.h:7`).
    pub idyin: Vec<i32>,
    /// Original: `int *idxIn` (`cgpixels.h:8`).
    pub idx_in: Vec<i32>,
    /// Original: `float *outerPixels` (`cgpixels.h:9`).
    pub outer_pixels: Vec<f32>,
    /// Original: `float *elongSmooth` (`cgpixels.h:10`).
    pub elong_smooth: Vec<f32>,
    /// Original: `float *edgePixels` (`cgpixels.h:11`).
    pub edge_pixels: Vec<f32>,
    /// Original: `int *elongMask` (`cgpixels.h:12`).
    pub elong_mask: Vec<i32>,
    /// Original: `int *idyOuter` (`cgpixels.h:13`).
    pub idy_outer: Vec<i32>,
    /// Original: `int *idxOuter` (`cgpixels.h:14`).
    pub idx_outer: Vec<i32>,
    /// Original: `int *iyElong` (`cgpixels.h:15`).
    pub iy_elong: Vec<i32>,
    /// Original: `int *ixElong` (`cgpixels.h:16`).
    pub ix_elong: Vec<i32>,
    /// Original: `float elongKernel[49]` (`cgpixels.h:17`).
    pub elong_kernel: [f32; 49],
    /// Original: `float outerKernel[81]` (`cgpixels.h:18`).
    pub outer_kernel: [f32; 81],
    /// Original: `int numInside` (`cgpixels.h:19`).
    pub num_inside: i32,
    /// Original: `int numEdge` (`cgpixels.h:20`).
    pub num_edge: i32,
    /// Original: `int iPolarity` (`cgpixels.h:21`).
    pub i_polarity: i32,
    /// Original: `int kernDimElong` (`cgpixels.h:22`).
    pub kern_dim_elong: i32,
    /// Original: `int numOuter` (`cgpixels.h:23`).
    pub num_outer: i32,
    /// Original: `int kernDimOuter` (`cgpixels.h:24`).
    pub kern_dim_outer: i32,
    /// Original: `int numPixForBestCGcen` (`cgpixels.h:25`).
    pub num_pix_for_best_cgcen: i32,
    /// Original: `int edgeMedian` (`cgpixels.h:26`).
    pub edge_median: i32,
    /// Original: `int getEdgeSD` (`cgpixels.h:27`).
    pub get_edge_sd: i32,
}

impl Default for CGPixels {
    fn default() -> Self {
        CGPixels {
            idy_edge: Vec::new(),
            idx_edge: Vec::new(),
            idyin: Vec::new(),
            idx_in: Vec::new(),
            outer_pixels: Vec::new(),
            elong_smooth: Vec::new(),
            edge_pixels: Vec::new(),
            elong_mask: Vec::new(),
            idy_outer: Vec::new(),
            idx_outer: Vec::new(),
            iy_elong: Vec::new(),
            ix_elong: Vec::new(),
            elong_kernel: [0.0; 49],
            outer_kernel: [0.0; 81],
            num_inside: 0,
            num_edge: 0,
            i_polarity: 0,
            kern_dim_elong: 0,
            num_outer: 0,
            kern_dim_outer: 0,
            num_pix_for_best_cgcen: 0,
            edge_median: 0,
            get_edge_sd: 0,
        }
    }
}
