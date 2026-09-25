//! Translation of `IMOD/flib/tiltalign/evalfunct.h`.
//!
//! The header declares `class EvalFunct`.  Its only inline body is the empty
//! constructor `EvalFunct() {};` (`evalfunct.h:10`), translated here as
//! [`EvalFunct::new`].  Every other member function is defined in
//! `funct.cpp` (`allocateFunctVars` :72, `freeThreadAllocations` :130,
//! `funct` :145, `gradientSum` :841, `remap_params` :862, `map_one_var` :934,
//! `matrix_to_coef` :962, `solveLeftOutXYZs` :988), so those land as an
//! `impl EvalFunct` block in `tiltalign/funct.rs` (ORDER.md §9 wave 3).  The
//! data members are private in C++; here they are `pub(super)`, which makes
//! them visible to `funct.rs` (a sibling under `tiltalign`) and nowhere else.
//!
//! # Guidance for `funct.rs`
//!
//! - The class is reached through three file-scope statics in `funct.cpp`
//!   (`sEvalFunct`, `av`, `mx`, :56-58), set by `allocateFunctVars`, plus the
//!   free function `funct()` (:66) that `metroSearch` calls back
//!   (`tiltali.cpp:193`).  In the translation the callback is a closure that
//!   captures `&mut EvalFunct` and `&mut AlignVariables`/`&ArrayMaxes`; the
//!   statics hold nothing.
//! - Every pointer member is a `Vec`, `B3DMALLOC`ed in `allocateFunctVars`
//!   with these sizes (`ms = maxView`): `mRealInView` `maxReal * ms` `char`s,
//!   `mIndvReal` `maxReal * ms` `short`s, `mIndvProj` `maxReal * ms`,
//!   `mXproj`/`mYproj` `maxProjPt`, `mResProd` `6 * maxProjPt`,
//!   `mDmat`/`mXTmat`/`mYTmat`/`mBeamInv` `9 * ms`, `mRmat` `4 * ms`,
//!   `mBeamMat` `6 * ms`, `mXYZmat` `8 * ms`, the rest `ms`.  The constructor
//!   leaves them uninitialised (not even `NULL`); `new` gives empty `Vec`s.
//!   `B3DMALLOC` does not zero, a `Vec` does (`NATIVE.md` §4).
//! - `char` is signed on this platform, so `mRealInView` is `i8`; it only
//!   holds 0/1 flags (`funct.cpp:229,237`).

/// Original: `class EvalFunct` (`evalfunct.h:7-39`).
#[derive(Clone, Debug, Default)]
pub struct EvalFunct {
    /// Original: `char *mRealInView` (`evalfunct.h:25`).
    pub(super) m_real_in_view: Vec<i8>,
    /// Original: `float *mXbar` (`evalfunct.h:26`).
    pub(super) m_xbar: Vec<f32>,
    /// Original: `float *mYbar` (`evalfunct.h:26`).
    pub(super) m_ybar: Vec<f32>,
    /// Original: `float *mXproj` (`evalfunct.h:26`).
    pub(super) m_xproj: Vec<f32>,
    /// Original: `float *mYproj` (`evalfunct.h:26`).
    pub(super) m_yproj: Vec<f32>,
    /// Original: `float *mXcen` (`evalfunct.h:27`).
    pub(super) m_xcen: Vec<f32>,
    /// Original: `float *mYcen` (`evalfunct.h:27`).
    pub(super) m_ycen: Vec<f32>,
    /// Original: `float *mZcen` (`evalfunct.h:27`).
    pub(super) m_zcen: Vec<f32>,
    /// Original: `float *mA` (`evalfunct.h:28`).
    pub(super) m_a: Vec<f32>,
    /// Original: `float *mB` (`evalfunct.h:28`).
    pub(super) m_b: Vec<f32>,
    /// Original: `float *mC` (`evalfunct.h:28`).
    pub(super) m_c: Vec<f32>,
    /// Original: `float *mD` (`evalfunct.h:28`).
    pub(super) m_d: Vec<f32>,
    /// Original: `float *mE` (`evalfunct.h:28`).
    pub(super) m_e: Vec<f32>,
    /// Original: `float *mF` (`evalfunct.h:28`).
    pub(super) m_f: Vec<f32>,
    /// Original: `float *mAOverN` (`evalfunct.h:28`).
    pub(super) m_a_over_n: Vec<f32>,
    /// Original: `float *mBOverN` (`evalfunct.h:29`).
    pub(super) m_b_over_n: Vec<f32>,
    /// Original: `float *mCOverN` (`evalfunct.h:29`).
    pub(super) m_c_over_n: Vec<f32>,
    /// Original: `float *mDOverN` (`evalfunct.h:29`).
    pub(super) m_d_over_n: Vec<f32>,
    /// Original: `float *mEOverN` (`evalfunct.h:29`).
    pub(super) m_e_over_n: Vec<f32>,
    /// Original: `float *mFOverN` (`evalfunct.h:29`).
    pub(super) m_f_over_n: Vec<f32>,
    /// Original: `float *mCosBet` (`evalfunct.h:30`).
    pub(super) m_cos_bet: Vec<f32>,
    /// Original: `float *mSinBet` (`evalfunct.h:30`).
    pub(super) m_sin_bet: Vec<f32>,
    /// Original: `float *mCosAlf` (`evalfunct.h:30`).
    pub(super) m_cos_alf: Vec<f32>,
    /// Original: `float *mSinAlf` (`evalfunct.h:30`).
    pub(super) m_sin_alf: Vec<f32>,
    /// Original: `float *mCosGam` (`evalfunct.h:31`).
    pub(super) m_cos_gam: Vec<f32>,
    /// Original: `float *mSinGam` (`evalfunct.h:31`).
    pub(super) m_sin_gam: Vec<f32>,
    /// Original: `float *mCosDel` (`evalfunct.h:31`).
    pub(super) m_cos_del: Vec<f32>,
    /// Original: `float *mSinDel` (`evalfunct.h:31`).
    pub(super) m_sin_del: Vec<f32>,
    /// Original: `float *mMag` (`evalfunct.h:31`).
    pub(super) m_mag: Vec<f32>,
    /// Original: `short *mIndvReal` (`evalfunct.h:32`).
    pub(super) m_indv_real: Vec<i16>,
    /// Original: `int *mNptInView` (`evalfunct.h:33`).
    pub(super) m_npt_in_view: Vec<i32>,
    /// Original: `int *mIndvProj` (`evalfunct.h:33`).
    pub(super) m_indv_proj: Vec<i32>,
    /// Original: `float *mResProd` (`evalfunct.h:34`).
    pub(super) m_res_prod: Vec<f32>,
    /// Original: `float *mDmat` (`evalfunct.h:35`).
    pub(super) m_dmat: Vec<f32>,
    /// Original: `float *mXTmat` (`evalfunct.h:35`).
    pub(super) m_xtmat: Vec<f32>,
    /// Original: `float *mYTmat` (`evalfunct.h:35`).
    pub(super) m_ytmat: Vec<f32>,
    /// Original: `float *mRmat` (`evalfunct.h:35`).
    pub(super) m_rmat: Vec<f32>,
    /// Original: `float *mBeamInv` (`evalfunct.h:36`).
    pub(super) m_beam_inv: Vec<f32>,
    /// Original: `float *mBeamMat` (`evalfunct.h:36`).
    pub(super) m_beam_mat: Vec<f32>,
    /// Original: `float *mXYZmat` (`evalfunct.h:37`).
    pub(super) m_xyzmat: Vec<f32>,
}

impl EvalFunct {
    /// Original: constructor `EvalFunct() {};` (`evalfunct.h:10`).
    ///
    /// The C++ body is empty and leaves every member uninitialised; the
    /// members are only read after `allocateFunctVars` assigns them.
    pub fn new() -> EvalFunct {
        EvalFunct::default()
    }
}
