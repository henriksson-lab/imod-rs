//! Translation of `IMOD/flib/tiltalign/arraymaxes.h`.
//!
//! The header has no implementation pair: three macro constants and the
//! `ArrayMaxes` struct.  The macros are untyped in C and every use is in `int`
//! arithmetic (`mx->maxView * maxWgtRings`, `utilfuncs.cpp:69`;
//! `B3DMIN(mx->maxReal, maxRealForDirectInit)`, `tiltalign.cpp:567`) or as an
//! array bound (`int nmapSpecMag[MAXGRP]`, `tltcntrl.h:15`), so they are `i32`
//! here and array bounds spell `MAXGRP as usize`.

/// Original: `#define MAXGRP 20` (`arraymaxes.h:4`).
pub const MAXGRP: i32 = 20;
/// Original: `#define maxWgtRings 10` (`arraymaxes.h:5`).
pub const MAX_WGT_RINGS: i32 = 10;
/// Original: `#define maxRealForDirectInit 200` (`arraymaxes.h:6`).
pub const MAX_REAL_FOR_DIRECT_INIT: i32 = 200;

/// Original: `struct ArrayMaxes` (`arraymaxes.h:7-12`).
///
/// Both programs hold one as a file-scope `static ArrayMaxes arrayMaxes;`
/// (`tiltalign.cpp:31`, `beadtrack.cpp:38`), so it starts zero-initialised,
/// which is what `Default` gives.  `allocateAlivar` (`utilfuncs.cpp:54-57`)
/// sets all three fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArrayMaxes {
    /// Original: `maxProjPt` (`arraymaxes.h:9`).
    pub max_proj_pt: i32,
    /// Original: `maxReal` (`arraymaxes.h:10`).
    pub max_real: i32,
    /// Original: `maxView` (`arraymaxes.h:11`).
    pub max_view: i32,
}
