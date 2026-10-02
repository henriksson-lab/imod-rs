//! Translation units from `IMOD/raptor/suitesparse` (CSparse 2.2.3), the
//! parts RAPTOR reaches.  `cs.rs` is `cs.h`; every other module is the
//! source file of the same name.  The routines RAPTOR never calls
//! (`cs_lu`, `cs_qr`, `cs_dmperm`, ...) are recorded in `DEAD_CODE.md`.
//!
//! Ownership replaces CSparse's `malloc`/`free` discipline: a `cs` is a
//! [`cs::Cs`] owning its arrays in `Vec`s, `cs_spfree`/`cs_sfree`/`cs_nfree`
//! are `Drop`, and a routine's workspaces are dropped when it returns.  A
//! `NULL` result (bad input, or a numeric failure such as a matrix that is
//! not positive definite) is `None`.  Every array keeps the length CSparse
//! allocated for it (`CS_MAX(n,1)` elements), and memory CSparse leaves
//! uninitialised is zero here; no reached path reads it before writing it.

pub mod cs;
pub mod cs_add;
pub mod cs_amd;
pub mod cs_chol;
pub mod cs_cholsol;
pub mod cs_compress;
pub mod cs_counts;
pub mod cs_cumsum;
pub mod cs_droptol;
pub mod cs_dropzeros;
pub mod cs_entry;
pub mod cs_ereach;
pub mod cs_etree;
pub mod cs_fkeep;
pub mod cs_gaxpy;
pub mod cs_ipvec;
pub mod cs_leaf;
pub mod cs_lsolve;
pub mod cs_ltsolve;
pub mod cs_malloc;
pub mod cs_multiply;
pub mod cs_pinv;
pub mod cs_post;
pub mod cs_pvec;
pub mod cs_scatter;
pub mod cs_schol;
pub mod cs_symperm;
pub mod cs_tdfs;
pub mod cs_transpose;
pub mod cs_util;
