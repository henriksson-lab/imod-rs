//! Translation of `IMOD/flib/tiltalign/mapsepgroups.h`.
//!
//! Type-only header; `allocateMapsep` (`utilfuncs.cpp:121-126`) is what sizes
//! the two arrays: `iviewsInGroup` to `mx->maxView * MAXGRP` and
//! `numSepInGroup` to `MAXGRP`.

/// Original: `struct MapSepGroups` (`mapsepgroups.h:6-11`).
///
/// Held as a file-scope `static MapSepGroups sepGroups;` in both programs
/// (`tiltalign.cpp:33`, `beadtrack.cpp:40`), so zero-initialised: `Default`.
/// A C `int *` that `B3DMALLOC` fills is a `Vec<i32>`; an unallocated
/// (`NULL`) pointer is an empty `Vec`.  `B3DMALLOC` leaves the memory
/// uninitialised where a `Vec` is zeroed (`NATIVE.md` §4).
#[derive(Clone, Debug, Default)]
pub struct MapSepGroups {
    /// Original: `int *numSepInGroup` (`mapsepgroups.h:8`), `MAXGRP` long.
    pub num_sep_in_group: Vec<i32>,
    /// Original: `int numSeparateGroups` (`mapsepgroups.h:9`).
    pub num_separate_groups: i32,
    /// Original: `int *iviewsInGroup` (`mapsepgroups.h:10`),
    /// `maxView * MAXGRP` long.
    pub iviews_in_group: Vec<i32>,
}
