//! Translation of `IMOD/libcfshr/montagexcorr.c` — functions for correlation of overlap
//! zones between montage pieces.
//!
//! Author of the original: David Mastronarde.  One Rust function per C function, with the
//! original identifier named in each doc comment.
#![allow(non_snake_case, dead_code, unused_variables, unused_assignments)]

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use std::io::Write;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

// ---------------------------------------------------------------------------------------
// Foreign boundary: the C stdout stream.
//
// The six debug format strings are rendered by `b3dutil::c_format`, the tree's verified
// translation of glibc's `printf` formatting (they use `%14.7e`, `%14.7g` and `%g`, which
// Rust's own formatting does not reproduce).  The *stream* is a separate matter: the debug
// lines still go out through libc stdio rather than `println!`, because C stdio is
// block-buffered when redirected and Rust's stdout is not, so a program that mixed the two
// would reorder its own output under `>` while looking correct on a terminal.
// ---------------------------------------------------------------------------------------

pub const MONTXC_MAX_PEAKS: i32 = 100;
pub const MONTXC_MAX_DEBUG_LINE: i32 = 90;
pub const MAX_RUNNERS_UP: i32 = 2;
pub const SLICE_MODE_FLOAT: i32 = 2;

/// C `static float sDistWeightHalfFall`, held as raw bits so the file-scope global keeps the
/// process-wide sharing the C has without needing a lock.
static S_DIST_WEIGHT_HALF_FALL: AtomicU32 = AtomicU32::new(0.0f32.to_bits());
/// C `static float sLastTrimmedMaxSD`.
static S_LAST_TRIMMED_MAX_SD: AtomicU32 = AtomicU32::new((-1.0f32).to_bits());
/// C `static float sLastRunnersUp[MAX_RUNNERS_UP * 2]`.
static S_LAST_RUNNERS_UP: [AtomicU32; 4] = [
    AtomicU32::new(0.0f32.to_bits()),
    AtomicU32::new(0.0f32.to_bits()),
    AtomicU32::new(0.0f32.to_bits()),
    AtomicU32::new(0.0f32.to_bits()),
];

/// C `montXCBasicSizes`.
///
/// Sets up most of the sizes for the overlap zone correlations.
pub fn mont_xc_basic_sizes(
    mut ixy: i32,
    nbin: i32,
    indentXC: i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    aspectMax: f32,
    extraWidth: f32,
    padFrac: f32,
    niceLimit: i32,
    indentUse: &mut i32,
    nxyBox: &mut [i32],
    numExtra: &mut [i32],
    nxPad: &mut i32,
    nyPad: &mut i32,
    maxLongShift: &mut i32,
) {
    let iyx: i32;
    let mut nxyBorder: [i32; 2] = [0; 2];
    let mut shiftInOverlap: i32 = 0;
    if ixy > 1 {
        ixy = ixy % 2;
        shiftInOverlap = if nxyOverlap[(1 - ixy) as usize] >= 0 {
            nxyOverlap[(1 - ixy) as usize]
        } else {
            -nxyOverlap[(1 - ixy) as usize]
        };
    }
    iyx = 1 - ixy;
    *indentUse = indentXC.min((nxyOverlap[ixy as usize] - 8) / 2);
    nxyBox[ixy as usize] = (nxyOverlap[ixy as usize] - *indentUse * 2) / nbin;
    nxyBox[iyx as usize] =
        (nxyPiece[iyx as usize] - shiftInOverlap - (2 * nbin).max(nxyPiece[ixy as usize] / 20))
            .min((aspectMax * nxyOverlap[ixy as usize] as f32) as i32)
            / nbin;
    numExtra[iyx as usize] = 0;
    numExtra[ixy as usize] =
        (2 * (((extraWidth * nxyBox[ixy as usize] as f32) as f64 + 0.5).floor() as i32 / 2)).min(
            (nxyPiece[ixy as usize]
                - crate::imod::libcfshr::b3dutil::b3d_i_max(&[
                    nbin,
                    *indentUse,
                    nxyPiece[ixy as usize] / 20,
                ]) * 2)
                / nbin
                - nxyBox[ixy as usize],
        );
    nxyBox[ixy as usize] = nxyBox[ixy as usize] + numExtra[ixy as usize];
    *maxLongShift = ((if 1.9f64 * nxyOverlap[ixy as usize] as f64 / nbin as f64
        > 1.5f64 * nxyBox[ixy as usize] as f64
    {
        1.9f64 * nxyOverlap[ixy as usize] as f64 / nbin as f64
    } else {
        1.5f64 * nxyBox[ixy as usize] as f64
    }) + 0.5f64)
        .floor() as i32;

    /* get the padded size */
    /* Limit the long dimension padding to that needed for the maximum shift */
    nxyBorder[ixy as usize] =
        5.max(((padFrac * nxyBox[ixy as usize] as f32) as f64 + 0.5).floor() as i32);
    nxyBorder[iyx as usize] = 5
        .max(((padFrac * nxyBox[iyx as usize] as f32) as f64 + 0.5).floor() as i32)
        .min(5.max((0.45f64 * *maxLongShift as f64 + 0.5).floor() as i32));
    *nxPad =
        crate::imod::libcfshr::filtxcorr::nice_frame(nxyBox[0] + 2 * nxyBorder[0], 2, niceLimit);
    *nyPad =
        crate::imod::libcfshr::filtxcorr::nice_frame(nxyBox[1] + 2 * nxyBorder[1], 2, niceLimit);
}

/// C `montxcbasicsizes` — Fortran wrapper for `montXCBasicSizes`.  `ixy` should be 1 or 2.
pub fn montxcbasicsizes(
    ixy: &i32,
    nbin: &i32,
    indentXC: &i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    aspectMax: &f32,
    extraWidth: &f32,
    padFrac: &f32,
    niceLimit: &i32,
    indentUse: &mut i32,
    nxyBox: &mut [i32],
    numExtra: &mut [i32],
    nxPad: &mut i32,
    nyPad: &mut i32,
    maxLongShift: &mut i32,
) {
    mont_xc_basic_sizes(
        *ixy - 1,
        *nbin,
        *indentXC,
        nxyPiece,
        nxyOverlap,
        *aspectMax,
        *extraWidth,
        *padFrac,
        *niceLimit,
        indentUse,
        nxyBox,
        numExtra,
        nxPad,
        nyPad,
        maxLongShift,
    );
}

/// C `montXCIndsAndCTF`.
///
/// Sets up indices for extracting boxes, and the filter function.
pub fn mont_xc_inds_and_ctf(
    mut ixy: i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    nxyBox: &[i32],
    nbin: i32,
    indentUse: i32,
    numExtra: &[i32],
    nxPad: i32,
    nyPad: i32,
    numSmooth: i32,
    sigma1: f32,
    sigma2: f32,
    radius1: f32,
    radius2: f32,
    evalCCC: i32,
    ind0Lower: &mut [i32],
    ind1Lower: &mut [i32],
    ind0Upper: &mut [i32],
    ind1Upper: &mut [i32],
    nxSmooth: &mut i32,
    nySmooth: &mut i32,
    ctf: &mut [f32],
    delta: &mut f32,
) {
    let mut iyx: i32;
    let mut shiftInOverlap: i32 = 0;
    let mut longShift: i32 = 0;
    if ixy > 1 {
        ixy = ixy % 2;
        longShift = 1;
        shiftInOverlap = -nxyOverlap[(1 - ixy) as usize];
    }
    iyx = 1 - ixy;
    ind0Lower[iyx as usize] =
        (nxyPiece[iyx as usize] / 2 - shiftInOverlap) - (nbin * nxyBox[iyx as usize]) / 2;
    ind1Lower[iyx as usize] = ind0Lower[iyx as usize] + nbin * nxyBox[iyx as usize] - 1;
    ind0Lower[ixy as usize] = nxyPiece[ixy as usize] - nxyOverlap[ixy as usize] + indentUse
        - nbin * numExtra[ixy as usize];
    ind1Lower[ixy as usize] = ind0Lower[ixy as usize] + nbin * nxyBox[ixy as usize] - 1;
    ind0Upper[0] = indentUse;
    ind1Upper[0] = indentUse + nbin * nxyBox[ixy as usize] - 1;
    if longShift != 0 {
        if ixy != 0 {
            ind0Upper[1] = ind0Upper[0];
            ind1Upper[1] = ind1Upper[0];
        }
        ind0Upper[iyx as usize] = ind0Lower[iyx as usize] + shiftInOverlap;
        ind1Upper[iyx as usize] = ind1Lower[iyx as usize] + shiftInOverlap;
    }

    /* Set up smoothing over some pixels, but no more than half of the pad */
    *nxSmooth = nxyBox[0] + (2 * numSmooth).min((nxPad - nxyBox[0]) / 2);
    *nySmooth = nxyBox[1] + (2 * numSmooth).min((nyPad - nxyBox[1]) / 2);

    /* Multiply high-frequency filtering parameters by the binning so they are equivalent
    to frequencies in unbinned images */
    unsafe {
        crate::imod::libcfshr::filtxcorr::xcorr_set_ctf(
            sigma1,
            nbin as f32 * sigma2,
            radius1,
            nbin as f32 * radius2,
            ctf,
            nxPad,
            nyPad,
            delta,
        );
    }
    if evalCCC != 0 {
        iyx = 0;
        while iyx < 8193 {
            ctf[iyx as usize] = (ctf[iyx as usize] as f64).sqrt() as f32;
            iyx += 1;
        }
    }
}

/// C `montxcindsandctf` — Fortran wrapper for `montXCIndsAndCTF`.  `ixy` should be 1 or 2.
pub fn montxcindsandctf(
    ixy: &i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    nxyBox: &[i32],
    nbin: &i32,
    indentUse: &i32,
    numExtra: &[i32],
    nxPad: &i32,
    nyPad: &i32,
    numSmooth: &i32,
    sigma1: &f32,
    sigma2: &f32,
    radius1: &f32,
    radius2: &f32,
    evalCCC: &i32,
    ind0Lower: &mut [i32],
    ind1Lower: &mut [i32],
    ind0Upper: &mut [i32],
    ind1Upper: &mut [i32],
    nxSmooth: &mut i32,
    nySmooth: &mut i32,
    ctf: &mut [f32],
    delta: &mut f32,
) {
    mont_xc_inds_and_ctf(
        *ixy - 1,
        nxyPiece,
        nxyOverlap,
        nxyBox,
        *nbin,
        *indentUse,
        numExtra,
        *nxPad,
        *nyPad,
        *numSmooth,
        *sigma1,
        *sigma2,
        *radius1,
        *radius2,
        *evalCCC,
        ind0Lower,
        ind1Lower,
        ind0Upper,
        ind1Upper,
        nxSmooth,
        nySmooth,
        ctf,
        delta,
    );
}

/// C `montXCFindBinning`.
///
/// Finds binning needed to keep boxed out area smaller than a target size.
pub fn mont_xc_find_binning(
    maxBin: i32,
    targetSize: i32,
    indentXC: i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    aspectMax: f32,
    extraWidth: f32,
    padFrac: f32,
    niceLimit: i32,
    numPaddedPix: &mut i32,
    numBoxedPix: &mut i32,
) -> i32 {
    let mut nxPad: i32 = 0;
    let mut nyPad: i32 = 0;
    let mut indentUse: i32 = 0;
    let mut nxyBox: [i32; 2] = [0; 2];
    let mut numExtra: [i32; 2] = [0; 2];
    let mut maxLongShift: i32 = 0;
    let mut ixy: i32;
    let mut nbin: i32 = 1;
    while nbin <= maxBin {
        *numPaddedPix = 0;
        *numBoxedPix = 0;
        ixy = 0;
        while ixy < 2 {
            mont_xc_basic_sizes(
                ixy,
                nbin,
                indentXC,
                nxyPiece,
                nxyOverlap,
                aspectMax,
                extraWidth,
                padFrac,
                niceLimit,
                &mut indentUse,
                &mut nxyBox[0..],
                &mut numExtra[0..],
                &mut nxPad,
                &mut nyPad,
                &mut maxLongShift,
            );
            *numPaddedPix = (*numPaddedPix).max((nxPad + 8) * (nyPad + 8));
            *numBoxedPix = (*numBoxedPix).max((nxyBox[0] + 4) * (nxyBox[1] + 4));
            ixy += 1;
        }
        if *numBoxedPix <= targetSize * targetSize {
            return nbin;
        }
        nbin += 1;
    }
    maxBin
}

/// C `montxcfindbinning` — Fortran wrapper for `montXCFindBinning`.
pub fn montxcfindbinning(
    maxBin: &i32,
    targetSize: &i32,
    indentXC: &i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    aspectMax: &f32,
    extraWidth: &f32,
    padFrac: &f32,
    niceLimit: &i32,
    numPaddedPix: &mut i32,
    numBoxedPix: &mut i32,
) -> i32 {
    mont_xc_find_binning(
        *maxBin,
        *targetSize,
        *indentXC,
        nxyPiece,
        nxyOverlap,
        *aspectMax,
        *extraWidth,
        *padFrac,
        *niceLimit,
        numPaddedPix,
        numBoxedPix,
    )
}

/// C `montXCFindBinning2`.
///
/// Finds binning needed to keep boxed out area smaller than a target size along one edge,
/// with expected shift at the edge taken into account.
pub fn mont_xc_find_binning2(
    maxBin: i32,
    targetSize: i32,
    indentXC: i32,
    ixy: i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    expectedShift: &[i32],
    aspectMax: f32,
    extraWidth: f32,
    padFrac: f32,
    niceLimit: i32,
    numPaddedPix: &mut i32,
    numBoxedPix: &mut i32,
) -> i32 {
    let mut nxPad: i32 = 0;
    let mut nyPad: i32 = 0;
    let mut indentUse: i32 = 0;
    let mut nxyBox: [i32; 2] = [0; 2];
    let mut numExtra: [i32; 2] = [0; 2];
    let mut maxLongShift: i32 = 0;
    let mut nbin: i32;
    let mut overlapUse: [i32; 2] = [0; 2];
    overlapUse[ixy as usize] = nxyOverlap[ixy as usize] + 0.max(-expectedShift[ixy as usize]);
    overlapUse[(1 - ixy) as usize] = expectedShift[(1 - ixy) as usize];
    nbin = 1;
    while nbin <= maxBin {
        mont_xc_basic_sizes(
            ixy + 2,
            nbin,
            indentXC,
            nxyPiece,
            &overlapUse,
            aspectMax,
            extraWidth,
            padFrac,
            niceLimit,
            &mut indentUse,
            &mut nxyBox[0..],
            &mut numExtra[0..],
            &mut nxPad,
            &mut nyPad,
            &mut maxLongShift,
        );
        *numPaddedPix = (nxPad + 8) * (nyPad + 8);
        *numBoxedPix = (nxyBox[0] + 4) * (nxyBox[1] + 4);
        if *numBoxedPix <= targetSize * targetSize {
            return nbin;
        }
        nbin += 1;
    }
    maxBin
}

/// C `montxcfindbinning2` — Fortran wrapper for `montXCFindBinning2`.
pub fn montxcfindbinning2(
    maxBin: &i32,
    targetSize: &i32,
    indentXC: &i32,
    ixy: &i32,
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    expectedShift: &[i32],
    aspectMax: &f32,
    extraWidth: &f32,
    padFrac: &f32,
    niceLimit: &i32,
    numPaddedPix: &mut i32,
    numBoxedPix: &mut i32,
) -> i32 {
    mont_xc_find_binning2(
        *maxBin,
        *targetSize,
        *indentXC,
        *ixy - 1,
        nxyPiece,
        nxyOverlap,
        expectedShift,
        *aspectMax,
        *extraWidth,
        *padFrac,
        *niceLimit,
        numPaddedPix,
        numBoxedPix,
    )
}

/// C `montXCorrEdge`.
///
/// Performs Fourier correlations and evaluation of real-space correlations to find
/// displacement on an edge.
///
/// `lowerCopy` is the C `float *lowerCopy` that may be NULL; `dumpEdge` is the C function
/// pointer that may be NULL.  `debugStr`/`debugLen` are the C debug buffer and its length.
pub fn mont_xcorr_edge(
    lowerIn: &[f32],
    upperIn: &[f32],
    nxyBox: &[i32],
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    nxSmooth: i32,
    nySmooth: i32,
    mut nxPad: i32,
    mut nyPad: i32,
    lowerPad: &mut [f32],
    upperPad: &mut [f32],
    mut lowerCopy: Option<&mut [f32]>,
    numXcorrPeaks: i32,
    legacy: i32,
    ctf: &[f32],
    delta: f32,
    inExtra: &[i32],
    nbin: i32,
    ixy: i32,
    maxLongShift: i32,
    weightCCC: i32,
    xDisplace: &mut f32,
    yDisplace: &mut f32,
    CCC: &mut f32,
    twoDfft: &mut dyn FnMut(&mut [f32], &mut i32, &mut i32, &mut i32),
    mut dumpEdge: Option<
        &mut dyn FnMut(&mut [f32], &mut i32, &mut i32, &mut i32, &mut i32, &mut i32),
    >,
    debugStr: &mut [u8],
    debugLen: i32,
    debugLevel: i32,
) {
    let mut ind: i32;
    let mut i: i32;
    let mut nxTrim: i32 = 0;
    let mut nyTrim: i32 = 0;
    let mut numPixel: i32 = 0;
    let mut indPeak: i32;
    let mut indSecond: i32;
    let mut indThird: i32;
    let mut curDebugLen: i32 = 0;
    let mut nxPadDim: i32 = nxPad + 2;
    let mut xpeak: [f32; MONTXC_MAX_PEAKS as usize] = [0.; MONTXC_MAX_PEAKS as usize];
    let mut ypeak: [f32; MONTXC_MAX_PEAKS as usize] = [0.; MONTXC_MAX_PEAKS as usize];
    let mut peak: [f32; MONTXC_MAX_PEAKS as usize] = [0.; MONTXC_MAX_PEAKS as usize];
    let mut wgtOrderInds: [i32; MONTXC_MAX_PEAKS as usize] = [0; MONTXC_MAX_PEAKS as usize];
    let mut wgtPeaks: [f32; MONTXC_MAX_PEAKS as usize] = [0.; MONTXC_MAX_PEAKS as usize];
    let mut gaussPeakProbs: [f32; MONTXC_MAX_PEAKS as usize] = [0.; MONTXC_MAX_PEAKS as usize];
    let mut sumArray: [f64; 7] = [0.; 7];
    let mut grandSums: [f64; 7] = [0.; 7];
    let mut cccSecond: f64 = 0.;
    let mut cccThird: f64 = 0.;
    let mut xTemp: f32;
    let mut yTemp: f32;
    let mut newCCC: f32 = 0.;
    let mut zero: i32 = 0;
    let mut one: i32 = 1;
    let jxy: i32 = 0;
    let mut ixyP1: i32;
    let mut numInSum: i32;
    let mut aWeights: Option<Vec<f32>> = None;
    let mut bWeights: Option<Vec<f32>> = None;
    let nxWgt: i32;
    let nyWgt: i32;
    let mut numSamp: i32 = 0;
    let binWgt: i32 = 2;
    let mut wgtXoffset: i32 = 0;
    let mut wgtYoffset: i32 = 0;
    let wgtBox: i32 = 10;
    let evalCCC: i32 = if numXcorrPeaks > 1 && legacy == 0 {
        1
    } else {
        0
    };
    let mut ccc: f64 = 0.;
    let mut cccMax: f64 = 0.;
    let wgtCCC: f64 = 0.;
    let mut fracArea: f64 = 0.;
    let mut sigma: f64 = 0.;
    let mut gaussProb: f64;
    let mut expectDist: [f64; 2] = [0.; 2];
    let mut delx: i32;
    let mut dely: i32;
    let mut xStart: i32;
    let mut xEnd: i32;
    let mut yStart: i32;
    let mut yEnd: i32;
    let mut fullPixel: i32 = 0;
    let wgtTrim: i32;
    let mut nyLocal: i32 = 0;
    let mut nxLocal: i32 = 0;
    let mut numLocalX: i32 = 0;
    let mut localXoverlap: i32 = 0;
    let mut numLocalY: i32 = 0;
    let mut localYoverlap: i32 = 0;
    let mut lyStart: i32;
    let mut lyEnd: i32;
    let mut lxStart: i32;
    let mut lxEnd: i32;
    let mut localX: i32;
    let mut localY: i32;
    let mut loc: i32;
    let mut indOrd: i32;
    let mut localXseq: [i32; 100] = [0; 100];
    let mut localYseq: [i32; 100] = [0; 100];
    let mut localAspect: f32;
    let maxLocalAspect: f32 = 2.;
    let mut maxWsum: f32 = 0.;
    let mut distLimit: f32 = 0.;
    let mut expectedXpeak: f32 = 0.;
    let mut expectedYpeak: f32 = 0.;
    let mut wsumAtMax: f32;
    let mut wsum: f32 = 0.;
    let delExtent: f32 = 0.;
    let wgtThresh: f32 = 0.;
    let fracDiffCrit: f32 = 0.95;
    let minWsumRatio: f32 = 0.33;
    let runnerUpThreshFac: f32 = 0.8;
    let mut longShiftToAdd: [i32; 2] = [0, 0];
    let mut extraFromExpected: [i32; 2] = [0, 0];
    let mut expectedLeft: [f32; 2] = [0., 0.];
    let mut numExtra: [i32; 2] = [0; 2];
    let edgeDisplace: f32 = if ixy != 0 { *yDisplace } else { *xDisplace };
    let longDisplace: f32 = if ixy != 0 { *xDisplace } else { *yDisplace };
    let overlapPow: f64 = 0.166667;
    // C `static int first = 1;` — read only by the commented-out SD-map image dumps.
    static FIRST: AtomicI32 = AtomicI32::new(1);
    let wallStart: f64;

    numExtra[0] = inExtra[0];
    numExtra[1] = if inExtra[1] >= 0 {
        inExtra[1]
    } else {
        -inExtra[1]
    };

    if weightCCC > 0 {
        // These are all unbinned
        extraFromExpected[ixy as usize] = ((if 0.0f64 > -edgeDisplace as f64 {
            0.0f64
        } else {
            -edgeDisplace as f64
        }) + 0.5f64)
            .floor() as i32;
        expectedLeft[ixy as usize] = (if 0.0f64 > edgeDisplace as f64 {
            0.0f64
        } else {
            edgeDisplace as f64
        }) as f32;
        longShiftToAdd[(1 - ixy) as usize] = (longDisplace as f64 + 0.5f64).floor() as i32;

        // Probability down to half at the overlap plus extra extent: has to be binned
        let sDistWeightHalfFall = f32::from_bits(S_DIST_WEIGHT_HALF_FALL.load(Ordering::Relaxed));
        sigma = 2.0f64
            * (if sDistWeightHalfFall as f64 > 0.0f64 {
                sDistWeightHalfFall / nbin as f32
            } else {
                (nxyOverlap[ixy as usize] / nbin + numExtra[ixy as usize]) as f32
            }) as f64
            / 2.355f64;
        distLimit = (2.03f64 * sigma) as f32;
        expectedXpeak = expectedLeft[0] / nbin as f32 + numExtra[0] as f32;
        expectedYpeak = expectedLeft[1] / nbin as f32 + inExtra[1] as f32;
        if debugLevel > 1 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
            let text = c_format(
                "sigma %.2f limit %.1f  expected %.1f %.1f\n",
                &[
                    CArg::Dbl(sigma),
                    CArg::Dbl(distLimit as f64),
                    CArg::Dbl(expectedXpeak as f64),
                    CArg::Dbl(expectedYpeak as f64),
                ],
            );
            let at = curDebugLen as usize;
            debugStr[at..at + text.len()].copy_from_slice(text.as_bytes());
            debugStr[at + text.len()] = 0;
            curDebugLen += text.len() as i32;
        }
    }
    ixyP1 = ixy + 1;

    // Clear out the 2nd and third peaks
    i = 0;
    while i < 4 {
        S_LAST_RUNNERS_UP[i as usize].store((-1.0e30f64 as f32).to_bits(), Ordering::Relaxed);
        i += 1;
    }

    // Set coordinate limits for weighting/SD maps
    xStart = (nxPad - nxyBox[0]) / 2;
    xEnd = nxPad - xStart;
    yStart = (nyPad - nxyBox[1]) / 2;
    yEnd = nyPad - yStart;
    nxWgt = (xEnd - xStart + binWgt - 1) / binWgt;
    xEnd = xStart + binWgt * nxWgt - 1;
    nyWgt = (yEnd - yStart + binWgt - 1) / binWgt;
    yEnd = yStart + binWgt * nyWgt - 1;
    wgtTrim = 5.min(nxWgt.min(nyWgt) / 20);

    // Set up to get SD maps if there is an array
    if lowerCopy.is_some() {
        if numXcorrPeaks > 1 {
            aWeights = Some(vec![0.0f32; (nxWgt * nyWgt) as usize]);
        }
        bWeights = Some(vec![0.0f32; (nxWgt * nyWgt) as usize]);
        // A Vec allocation failure aborts rather than returning NULL, so the `!bWeights` half
        // of the source's test can no longer fire; the `weightCCC && !aWeights` half still
        // does, when numXcorrPeaks <= 1.
        if (weightCCC != 0 && aWeights.is_none()) || bWeights.is_none() {
            aWeights = None;
            bWeights = None;
        }
    }

    // Loop on lower and upper piece, extracting etc.
    S_LAST_TRIMMED_MAX_SD.store((-1.0f32).to_bits(), Ordering::Relaxed);
    ind = 0;
    while ind < 2 {
        // The C walks `arrayIn`/`arrayOut`/`weights` from lower to upper at the end of the
        // iteration; Rust cannot re-seat a `&mut` from one parameter to another, so the same
        // choice is made at the top of the body instead.
        let arrayIn: &[f32] = if ind == 0 { lowerIn } else { upperIn };
        let arrayOut: &mut [f32] = if ind == 0 {
            &mut *lowerPad
        } else {
            &mut *upperPad
        };
        let weights: Option<&mut [f32]> = if ind == 0 {
            aWeights.as_deref_mut()
        } else {
            bWeights.as_deref_mut()
        };
        if nxSmooth > nxyBox[0] && nySmooth > nxyBox[1] {
            crate::imod::libcfshr::taperpad::slice_smooth_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Float(arrayIn),
                SLICE_MODE_FLOAT,
                nxyBox[0],
                nxyBox[1],
                arrayOut,
                nxSmooth,
                nxSmooth,
                nySmooth,
            );
            // The source passes `arrayOut` as both source and destination here;
            // `PadIn::InPlace` is that case.
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::InPlace,
                SLICE_MODE_FLOAT,
                nxSmooth,
                nySmooth,
                arrayOut,
                nxPadDim,
                nxPad,
                nyPad,
                0,
                0.,
            );
        } else {
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Float(arrayIn),
                SLICE_MODE_FLOAT,
                nxyBox[0],
                nxyBox[1],
                arrayOut,
                nxPadDim,
                nxPad,
                nyPad,
                0,
                0.,
            );
        }
        unsafe {
            crate::imod::libcfshr::filtxcorr::xcorr_mean_zero(arrayOut, nxPadDim, nxPad, nyPad);
        }
        if let Some(f) = &mut dumpEdge {
            f(
                arrayOut,
                &mut nxPadDim,
                &mut nxPad,
                &mut nyPad,
                &mut ixyP1,
                &mut zero,
            );
        }

        // Make the weighting/SD map
        // No variant on using simple SDs gave better results
        if let Some(w) = weights {
            let lc = lowerCopy.as_deref_mut().unwrap();
            {
                let (lcSum, lcSqr) = lc.split_at_mut((nxWgt * nyWgt) as usize);
                crate::imod::libcfshr::multibinstat::make_standard_dev_map(
                    &arrayOut[..],
                    nxPadDim,
                    xStart,
                    xEnd,
                    yStart,
                    yEnd,
                    binWgt,
                    wgtBox,
                    &mut w[..],
                    lcSum,
                    lcSqr,
                    &mut wgtXoffset,
                    &mut wgtYoffset,
                );
            }
            unsafe {
                crate::imod::libcfshr::samplemeansd::get_sample_of_array(
                    core::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len() * 4),
                    2,
                    nxWgt,
                    nyWgt,
                    1.,
                    wgtTrim,
                    wgtTrim,
                    nxWgt - 2 * wgtTrim,
                    nyWgt - 2 * wgtTrim,
                    -1.,
                    lc,
                    10000.min(nxPad * nyPad),
                    &mut numSamp,
                );
            }
            if numSamp > 0 {
                if numSamp <= 20 {
                    S_LAST_TRIMMED_MAX_SD
                        .store(lc[(numSamp - 1) as usize].to_bits(), Ordering::Relaxed);
                } else {
                    let v = crate::imod::libcfshr::percentile::percentile_float(
                        (0.95f64 * numSamp as f64) as i32,
                        lc,
                        numSamp,
                    );
                    S_LAST_TRIMMED_MAX_SD.store(v.to_bits(), Ordering::Relaxed);
                }
            }
            /*if (first) {
            if (ind)
              mrcWriteImageToFile("bstddev.mrc", weights, 2, nxWgt, nyWgt);
            else
              mrcWriteImageToFile("astddev.mrc", weights, 2, nxWgt, nyWgt);
              }*/
        }

        twoDfft(arrayOut, &mut nxPad, &mut nyPad, &mut zero);

        /* If filtering, apply to lower, and to upper as well if evaluating CCC's */
        if delta as f64 > 0. && (ind == 0 || evalCCC != 0) {
            // `XCorrFilterPart(arrayOut, arrayOut, ...)`: source and destination alias in C.
            let p = arrayOut.as_mut_ptr();
            unsafe {
                crate::imod::libcfshr::filtxcorr::xcorr_filter_part(
                    crate::imod::libcfshr::filtxcorr::FilterIn::InPlace,
                    core::slice::from_raw_parts_mut(p, ((nxPad + 2) * nyPad) as usize),
                    nxPad,
                    nyPad,
                    &ctf,
                    delta,
                );
            }
        }
        ind += 1;
    }
    FIRST.store(0, Ordering::Relaxed);
    if delta as f64 > 0. && evalCCC != 0 {
        let n = (nxPadDim * nyPad) as usize;
        lowerCopy.as_deref_mut().unwrap()[..n].copy_from_slice(&lowerPad[..n]);
    }

    /* multiply lower by complex conjugate of upper, put back in lower */
    unsafe {
        crate::imod::libcfshr::filtxcorr::conjugate_product(lowerPad, upperPad, nxPad, nyPad);
    }
    twoDfft(lowerPad, &mut nxPad, &mut nyPad, &mut one);
    if weightCCC != 0 {
        crate::imod::libcfshr::filtxcorr::set_peak_find_limits(
            (expectedXpeak - distLimit) as i32,
            (expectedXpeak + distLimit) as i32,
            (expectedYpeak - distLimit) as i32,
            (expectedYpeak + distLimit) as i32,
            1,
        );
    }
    unsafe {
        crate::imod::libcfshr::filtxcorr::xcorr_peak_find(
            lowerPad,
            nxPadDim,
            nyPad,
            &mut xpeak,
            &mut ypeak,
            &mut peak,
            16.max(numXcorrPeaks),
        );
    }

    /* Eliminate any peaks that shift beyond maximum along edge */
    /* leave indPeak pointing to first good peak */
    indThird = -1;
    indSecond = -1;
    indPeak = -1;
    i = 0;
    while i < 16.max(numXcorrPeaks) {
        if ixy == 0 && (ypeak[i as usize] as f64).abs() > maxLongShift as f64
            || ixy == 1 && (xpeak[i as usize] as f64).abs() > maxLongShift as f64
        {
            peak[i as usize] = -1.0e30f64 as f32;
            if debugLevel > 2 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
                let text = c_format(
                    "Eliminated peak %d at %.1f %.1f\n",
                    &[
                        CArg::Int(i as i64),
                        CArg::Dbl(xpeak[i as usize] as f64),
                        CArg::Dbl(ypeak[i as usize] as f64),
                    ],
                );
                let at = curDebugLen as usize;
                debugStr[at..at + text.len()].copy_from_slice(text.as_bytes());
                debugStr[at + text.len()] = 0;
                curDebugLen += text.len() as i32;
            }
        } else if indPeak == -1 && peak[i as usize] as f64 > -1.0e29f64 {
            indPeak = i;
        }
        i += 1;
    }

    /* But if no peak was legal, zero out the shift */
    if indPeak == -1 {
        indPeak = 0;
        xpeak[0] = numExtra[0] as f32;
        ypeak[0] = inExtra[1] as f32;
    }
    *CCC = -1.5f32;
    if evalCCC != 0 {
        /* If there was no filtering, simply pad images again */
        if delta == 0 as i32 as f32 {
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Float(lowerIn),
                SLICE_MODE_FLOAT,
                nxyBox[0],
                nxyBox[1],
                lowerCopy.as_deref_mut().unwrap(),
                nxPadDim,
                nxPad,
                nyPad,
                0,
                0.,
            );
            crate::imod::libcfshr::taperpad::slice_taper_out_pad(
                crate::imod::libcfshr::taperpad::PadIn::Float(upperIn),
                SLICE_MODE_FLOAT,
                nxyBox[0],
                nxyBox[1],
                upperPad,
                nxPadDim,
                nxPad,
                nyPad,
                0,
                0.,
            );
        } else {
            /* Otherwise, back-transform the filtered images */
            twoDfft(
                lowerCopy.as_deref_mut().unwrap(),
                &mut nxPad,
                &mut nyPad,
                &mut one,
            );
            twoDfft(upperPad, &mut nxPad, &mut nyPad, &mut one);
            if let Some(f) = &mut dumpEdge {
                f(
                    lowerCopy.as_deref_mut().unwrap(),
                    &mut nxPadDim,
                    &mut nxPad,
                    &mut nyPad,
                    &mut ixyP1,
                    &mut zero,
                );
                f(
                    upperPad,
                    &mut nxPadDim,
                    &mut nxPad,
                    &mut nyPad,
                    &mut ixyP1,
                    &mut zero,
                );
            }
        }

        // Set up local box sizes
        cccThird = -1.5;
        cccSecond = -1.5;
        cccMax = -1.5;
        wsumAtMax = 0.;
        nxTrim = 4.min(nxyBox[0] / 8) + (nxPad - nxyBox[0]) / 2;
        nyTrim = 4.min(nxyBox[1] / 8) + (nyPad - nxyBox[1]) / 2;
        fullPixel = (nxPad - 2 * nxTrim) * (nyPad - 2 * nyTrim);
        localAspect = ((nxyBox[(1 - ixy) as usize] as f32 / nxyBox[ixy as usize] as f32) as f64
            / 1.4f64) as f32;
        localAspect = if localAspect < maxLocalAspect {
            localAspect
        } else {
            maxLocalAspect
        };
        if ixy > 0 {
            nyLocal = (nyPad - 2 * nyTrim) / 2;
            nxLocal = (localAspect * nyLocal as f32) as i32;
        } else {
            nxLocal = (nxPad - 2 * nxTrim) / 2;
            nyLocal = (localAspect * nxLocal as f32) as i32;
        }
        i = 0;
        while i < numXcorrPeaks {
            wgtOrderInds[i as usize] = i;
            gaussPeakProbs[i as usize] = 1.;
            i += 1;
        }

        // Set up weighting for all peaks here so that they can be sorted by raw peak
        // strength times weighting
        if weightCCC != 0 {
            i = 0;
            while i < numXcorrPeaks {
                expectDist[0] = (xpeak[i as usize] - expectedXpeak) as f64;
                expectDist[1] = (ypeak[i as usize] - expectedYpeak) as f64;
                gaussPeakProbs[i as usize] = (-0.5f64
                    * ((expectDist[0] / sigma).powf(2.0f64) + (expectDist[1] / sigma).powf(2.0f64)))
                .exp() as f32;
                if debugLevel > 2 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
                    let text = c_format(
                        "%d: expected dist %.1f %.1f  prob %.3f\n",
                        &[
                            CArg::Int(i as i64),
                            CArg::Dbl(expectDist[0]),
                            CArg::Dbl(expectDist[1]),
                            CArg::Dbl(gaussPeakProbs[i as usize] as f64),
                        ],
                    );
                    let at = curDebugLen as usize;
                    debugStr[at..at + text.len()].copy_from_slice(text.as_bytes());
                    debugStr[at + text.len()] = 0;
                    curDebugLen += text.len() as i32;
                }
                wgtPeaks[i as usize] = -gaussPeakProbs[i as usize] * peak[i as usize];
                i += 1;
            }
            unsafe {
                crate::imod::libcfshr::robuststat::rs_sort_indexed_floats(
                    &wgtPeaks,
                    &mut wgtOrderInds,
                    numXcorrPeaks,
                );
            }
        }

        /*
         * Loop on peaks
         */
        wallStart = crate::imod::libcfshr::b3dutil::wall_time();
        indOrd = 0;
        while indOrd < numXcorrPeaks {
            i = wgtOrderInds[indOrd as usize];
            if peak[i as usize] as f64 <= -1.0e29f64 {
                indOrd += 1;
                continue;
            }

            // Skip if a CCC of 1 cannot possibly beat the current max
            gaussProb = gaussPeakProbs[i as usize] as f64;
            if i != 0 && gaussProb < 1.01f64 * cccMax {
                indOrd += 1;
                continue;
            }

            /* Reject peak at zero image offset from fixed pattern noise */
            if ixy == 0
                && ((nxyPiece[0] as f32 + nbin as f32 * (xpeak[i as usize] - numExtra[0] as f32)
                    - extraFromExpected[0] as f32
                    - nxyOverlap[0] as f32) as f64)
                    .abs()
                    <= 3.0f64
                || ixy == 1
                    && ((nxyPiece[1] as f32 + nbin as f32 * (ypeak[i as usize] - inExtra[1] as f32)
                        - extraFromExpected[1] as f32
                        - nxyOverlap[1] as f32) as f64)
                        .abs()
                        <= 3.0f64
            {
                indOrd += 1;
                continue;
            }

            // Ends are non-inclusive to avoid all the + 1's below
            delx = (xpeak[i as usize] as f64 + 0.5f64).floor() as i32;
            xStart = nxTrim.max(nxTrim + delx);
            xEnd = (nxPad - nxTrim).min(nxPad - nxTrim + delx);
            dely = (ypeak[i as usize] as f64 + 0.5f64).floor() as i32;
            yStart = nyTrim.max(nyTrim + dely);
            yEnd = (nyPad - nyTrim).min(nyPad - nyTrim + dely);
            numPixel = (yEnd - yStart) * (xEnd - xStart);
            fracArea = numPixel as f64 / fullPixel as f64;
            if i != 0 && fracArea < 0.125f64 {
                indOrd += 1;
                continue;
            }

            // What we used to do
            /*ccc = XCorrCCCoefficient(lowerCopy, upperPad, nxPadDim, nxPad, nyPad,
            xpeak[i], ypeak[i], nxTrim, nyTrim, &numPixel);
            printf(" %d:  %.4f\n", i, ccc);*/

            // Set up local areas and sequence for doing them
            local_num_and_overlap(xEnd - xStart, nxLocal, &mut numLocalX, &mut localXoverlap);
            local_num_and_overlap(yEnd - yStart, nyLocal, &mut numLocalY, &mut localYoverlap);
            if ixy > 0 {
                setup_local_sequence(numLocalX, numLocalY, &mut localXseq, &mut localYseq);
            } else {
                setup_local_sequence(numLocalY, numLocalX, &mut localYseq, &mut localXseq);
            }

            // Loop on local areas from middle out and sum component of CCC
            ind = 0;
            while ind < 7 {
                grandSums[ind as usize] = 0.;
                ind += 1;
            }
            numInSum = 0;
            loc = 0;
            while loc < numLocalX * numLocalY {
                localX = localXseq[loc as usize];
                localY = localYseq[loc as usize];
                if numLocalY == 1 {
                    lyStart = yStart + 0.max((yEnd - yStart - nyLocal) / 2);
                    lyEnd = yEnd.min(yStart + nyLocal);
                } else {
                    lyStart = yStart + localY * (nyLocal - localYoverlap);
                    lyEnd = yEnd.min(lyStart + nyLocal);
                }
                if numLocalX == 1 {
                    lxStart = xStart + 0.max((xEnd - xStart - nxLocal) / 2);
                    lxEnd = xEnd.min(xStart + nxLocal);
                } else {
                    lxStart = xStart + localX * (nxLocal - localXoverlap);
                    lxEnd = xEnd.min(lxStart + nxLocal - 1);
                }

                // Do search for best correlation in this area
                xTemp = xpeak[i as usize];
                yTemp = ypeak[i as usize];
                mont_xc_find_best_corr(
                    lowerCopy.as_deref().unwrap(),
                    upperPad,
                    nxPadDim,
                    nxPad,
                    nyPad,
                    nxTrim,
                    nyTrim,
                    lxStart,
                    lxEnd - 1,
                    lyStart,
                    lyEnd - 1,
                    &mut xTemp,
                    &mut yTemp,
                    &mut newCCC,
                    10.,
                    aWeights.as_deref(),
                    bWeights.as_deref(),
                    nxWgt,
                    binWgt,
                    wgtXoffset,
                    wgtYoffset,
                    (if 0.02f64 < cccMax / 10.0f64 {
                        0.02f64
                    } else {
                        cccMax / 10.0f64
                    }) as f32,
                    Some(&mut sumArray[..]),
                );

                if newCCC != 0 as i32 as f32 {
                    numInSum += 1;
                    ind = 0;
                    while ind < 6 {
                        grandSums[ind as usize] += sumArray[ind as usize];
                        ind += 1;
                    }

                    wsum = grandSums[5] as f32;
                    ccc = unsafe {
                        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
                            grandSums[0],
                            grandSums[1],
                            grandSums[2],
                            grandSums[3],
                            grandSums[4],
                            wsum as f64,
                            None,
                            "",
                        )
                    };

                    // Give up if the ccc after enough area is much lower than the max
                    if indOrd > 0
                        && (wsum as f64 > maxWsum as f64 / 10.0f64
                            && gaussProb * ccc < 0.5f64 * cccMax
                            || wsum as f64 > maxWsum as f64 / 5.0f64
                                && gaussProb * ccc < 0.75f64 * cccMax
                            || wsum as f64 > maxWsum as f64 / 3.0f64
                                && gaussProb * ccc < 0.85f64 * cccMax
                            || wsum as f64 > maxWsum as f64 / 2.0f64
                                && gaussProb * ccc < 0.9f64 * cccMax)
                    {
                        break;
                    }
                }
                loc += 1;
            }
            maxWsum = if maxWsum > wsum { maxWsum } else { wsum };

            // Handle a new max or a new 2nd or 3rd place one
            if wsum > minWsumRatio * wsumAtMax {
                if gaussProb * ccc > cccMax {
                    if cccMax > -1 as i32 as f64 {
                        indThird = indSecond;
                        indSecond = indPeak;
                        cccThird = cccSecond;
                        cccSecond = cccMax;
                    }
                    cccMax = gaussProb * ccc;
                    indPeak = i;
                    wsumAtMax = wsum;
                } else if gaussProb * ccc > cccSecond {
                    cccThird = cccSecond;
                    cccSecond = gaussProb * ccc;
                    indThird = indSecond;
                    indSecond = i;
                } else if gaussProb * ccc > cccThird {
                    cccThird = gaussProb * ccc;
                    indThird = i;
                }
            }

            if debugLevel > 1 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
                let text = c_format(
                    "%2d: at %7.1f %7.1f peak %14.7e  frac %.3f CCC %.5f%s wgt %.5f%s %g\n",
                    &[
                        CArg::Int(i as i64),
                        CArg::Dbl(xpeak[i as usize] as f64),
                        CArg::Dbl(ypeak[i as usize] as f64),
                        CArg::Dbl(peak[i as usize] as f64),
                        CArg::Dbl(fracArea),
                        CArg::Dbl(ccc),
                        CArg::Str(if weightCCC == 0 && indPeak == i {
                            "*"
                        } else {
                            " "
                        }),
                        CArg::Dbl(gaussProb * ccc),
                        CArg::Str(if weightCCC != 0 && indPeak == i {
                            "*"
                        } else {
                            " "
                        }),
                        CArg::Dbl(wsum as f64),
                    ],
                );
                let at = curDebugLen as usize;
                debugStr[at..at + text.len()].copy_from_slice(text.as_bytes());
                debugStr[at + text.len()] = 0;
                curDebugLen += text.len() as i32;
            }
            indOrd += 1;
        }
        i = indPeak;
        if debugLevel == 1 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
            let text = c_format(
                "Peak %d at %7.1f %7.1f  peak = %14.7g  CCC = %.5f\n",
                &[
                    CArg::Int(i as i64),
                    CArg::Dbl(xpeak[i as usize] as f64),
                    CArg::Dbl(ypeak[i as usize] as f64),
                    CArg::Dbl(peak[i as usize] as f64),
                    CArg::Dbl(cccMax),
                ],
            );
            let at = curDebugLen as usize;
            debugStr[at..at + text.len()].copy_from_slice(text.as_bytes());
            debugStr[at + text.len()] = 0;
            curDebugLen += text.len() as i32;
        }
        *CCC = cccMax as f32;

        // Save the runners up
        if indSecond >= 0 && cccSecond > runnerUpThreshFac as f64 * cccMax {
            S_LAST_RUNNERS_UP[0].store(
                (nbin as f32 * (xpeak[indSecond as usize] - numExtra[0] as f32)
                    + longShiftToAdd[0] as f32
                    - extraFromExpected[0] as f32)
                    .to_bits(),
                Ordering::Relaxed,
            );
            S_LAST_RUNNERS_UP[1].store(
                (nbin as f32 * (ypeak[indSecond as usize] - inExtra[1] as f32)
                    + longShiftToAdd[1] as f32
                    - extraFromExpected[1] as f32)
                    .to_bits(),
                Ordering::Relaxed,
            );
        }
        if indThird >= 0 && cccThird > runnerUpThreshFac as f64 * cccMax {
            S_LAST_RUNNERS_UP[2].store(
                (nbin as f32 * (xpeak[indThird as usize] - numExtra[0] as f32)
                    + longShiftToAdd[0] as f32
                    - extraFromExpected[0] as f32)
                    .to_bits(),
                Ordering::Relaxed,
            );
            S_LAST_RUNNERS_UP[3].store(
                (nbin as f32 * (ypeak[indThird as usize] - inExtra[1] as f32)
                    + longShiftToAdd[1] as f32
                    - extraFromExpected[1] as f32)
                    .to_bits(),
                Ordering::Relaxed,
            );
        }
    }
    if let Some(f) = &mut dumpEdge {
        f(
            lowerPad,
            &mut nxPadDim,
            &mut nxPad,
            &mut nyPad,
            &mut ixyP1,
            &mut one,
        );
    }

    /* return the amount to shift upper to align it to lower (verified) */
    // Subtract both extra width and the increase in width from expected shift, add the
    // shift from extracting overlaps in long direction
    *xDisplace = nbin as f32 * (xpeak[indPeak as usize] - numExtra[0] as f32)
        + longShiftToAdd[0] as f32
        - extraFromExpected[0] as f32;
    *yDisplace = nbin as f32 * (ypeak[indPeak as usize] - inExtra[1] as f32)
        + longShiftToAdd[1] as f32
        - extraFromExpected[1] as f32;
    if debugLevel != 0 && debugLen > curDebugLen + MONTXC_MAX_DEBUG_LINE {
        // The source does not advance curDebugLen after this last line.
        let text = c_format(
            "Peak at %8.2f %8.2f  Displacement %8.2f %8.2f\n",
            &[
                CArg::Dbl(xpeak[indPeak as usize] as f64),
                CArg::Dbl(ypeak[indPeak as usize] as f64),
                CArg::Dbl(*xDisplace as f64),
                CArg::Dbl(*yDisplace as f64),
            ],
        );
        let at = curDebugLen as usize;
        debugStr[at..at + text.len()].copy_from_slice(text.as_bytes());
        debugStr[at + text.len()] = 0;
    }
    // B3DFREE(aWeights); B3DFREE(bWeights) — the Vecs drop here.
}

/// C `montxcorredge` — Fortran wrapper for `montXCorrEdge`.  The wrapper collects and prints
/// the debug strings if `debugLevel` is non-zero.
pub fn montxcorredge(
    lowerIn: &[f32],
    upperIn: &[f32],
    nxyBox: &[i32],
    nxyPiece: &[i32],
    nxyOverlap: &[i32],
    nxSmooth: &i32,
    nySmooth: &i32,
    nxPad: &i32,
    nyPad: &i32,
    lowerPad: &mut [f32],
    upperPad: &mut [f32],
    lowerCopy: Option<&mut [f32]>,
    numXcorrPeaks: &i32,
    legacy: &i32,
    ctf: &[f32],
    delta: &f32,
    numExtra: &[i32],
    nbin: &i32,
    ixy: &i32,
    maxLongShift: &i32,
    weightCCC: &i32,
    xDisplace: &mut f32,
    yDisplace: &mut f32,
    CCC: &mut f32,
    twoDfft: &mut dyn FnMut(&mut [f32], &mut i32, &mut i32, &mut i32),
    dumpEdge: Option<&mut dyn FnMut(&mut [f32], &mut i32, &mut i32, &mut i32, &mut i32, &mut i32)>,
    debugLevel: &i32,
) {
    let debugLen: i32 = MONTXC_MAX_PEAKS * MONTXC_MAX_DEBUG_LINE;
    let mut debugStr: [u8; (MONTXC_MAX_PEAKS * MONTXC_MAX_DEBUG_LINE) as usize] =
        [0; (MONTXC_MAX_PEAKS * MONTXC_MAX_DEBUG_LINE) as usize];
    mont_xcorr_edge(
        lowerIn,
        upperIn,
        nxyBox,
        nxyPiece,
        nxyOverlap,
        *nxSmooth,
        *nySmooth,
        *nxPad,
        *nyPad,
        lowerPad,
        upperPad,
        lowerCopy,
        *numXcorrPeaks,
        *legacy,
        ctf,
        *delta,
        numExtra,
        *nbin,
        *ixy - 1,
        *maxLongShift,
        *weightCCC,
        xDisplace,
        yDisplace,
        CCC,
        twoDfft,
        dumpEdge,
        &mut debugStr,
        debugLen,
        *debugLevel,
    );
    if *debugLevel != 0 {
        // `curDebug` is an index rather than a pointer; `strchr` scans to the NUL, and the
        // source overwrites each newline with one, so the bound is recomputed each pass.
        let mut curDebug: usize = 0;
        loop {
            let nul = debugStr[curDebug..]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(debugStr.len() - curDebug);
            let Some(lineEnd) = debugStr[curDebug..curDebug + nul]
                .iter()
                .position(|&b| b == b'\n')
            else {
                break;
            };
            let lineEnd = curDebug + lineEnd;
            debugStr[lineEnd] = 0x00;
            // The C stream, not Rust's: mixing the two reorders output under
            // redirection (CLAUDE.md, "printf vs println!").  The line is the
            // NUL-delimited C string starting at `curDebug`.
            let line = &debugStr[curDebug..];
            let line = &line[..line.iter().position(|&b| b == 0).unwrap_or(line.len())];
            let _ = ImodFile::Stdout.write_all(&c_format_bytes("%s\n", &[CArg::Bytes(line)]));
            curDebug = lineEnd + 1;
        }
        let _ = ImodFile::Stdout.flush();
    }
}

/// C `localNumAndOverlap` (static).
///
/// Get number of boxes and their overlap on one axis.
fn local_num_and_overlap(extent: i32, nxLocal: i32, numLocalXp: &mut i32, nxOverlap: &mut i32) {
    let targetOverlap: f32 = 0.35;
    let minOverlap: f32 = 0.2;
    let maxOverlap: f32 = 0.5;
    let mut numLocalX: i32;
    numLocalX = ((extent - nxLocal) as f64 / (nxLocal as f64 * (1.0f64 - targetOverlap as f64))
        + 0.5f64)
        .floor() as i32
        + 1;
    numLocalX = if numLocalX > 1 { numLocalX } else { 1 };
    while numLocalX > 1
        && (nxLocal as f64 - (extent - nxLocal) as f64 / (numLocalX as f64 - 1.0f64))
            / nxLocal as f64
            > maxOverlap as f64
    {
        numLocalX -= 1;
    }

    while numLocalX > 1
        && ((nxLocal as f64 - (extent - nxLocal) as f64 / (numLocalX as f64 - 1.0f64))
            / nxLocal as f64)
            < minOverlap as f64
    {
        numLocalX += 1;
    }
    *nxOverlap = nxLocal - (extent - nxLocal) / 1.max(numLocalX - 1);
    *numLocalXp = numLocalX;
}

/// C `setupLocalSequence` (static).
///
/// Set up sequence from middle out.
fn setup_local_sequence(
    numLocalX: i32,
    numLocalY: i32,
    localXseq: &mut [i32],
    localYseq: &mut [i32],
) {
    let mut numLocalSeq: i32 = 0;
    let mut ind: i32;
    let mut localX: i32;
    let mut localY: i32;
    let mut dir: i32 = -1;
    ind = 0;
    while ind < numLocalX + 2 {
        if ind != 0 {
            localX = numLocalX / 2 + dir * ((ind + 1) / 2);
            dir = -dir;
        } else {
            localX = numLocalX / 2;
        }
        if localX >= 0 && localX < numLocalX {
            localY = 0;
            while localY < numLocalY {
                localXseq[numLocalSeq as usize] = localX;
                localYseq[numLocalSeq as usize] = localY;
                numLocalSeq += 1;
                localY += 1;
            }
        }
        ind += 1;
    }
}

/// C `montxcorrgetmaxes` — Fortran-callable routine returning the maximum number of peaks
/// allowed in `maxPeak` and the maximum number of debug lines in `maxLines`.
pub fn montxcorrgetmaxes(maxPeak: &mut i32, maxLines: &mut i32) {
    *maxPeak = MONTXC_MAX_PEAKS;
    *maxLines = MONTXC_MAX_DEBUG_LINE;
}

/// C `montXCSetDistWeightHalfFall`.
///
/// Sets the distance in pixels at which weighting by distance from expected shift falls by
/// half to `inVal` (callable from Fortran or C).
pub fn mont_xc_set_dist_weight_half_fall(inVal: &f32) {
    S_DIST_WEIGHT_HALF_FALL.store(inVal.to_bits(), Ordering::Relaxed);
}

/// C `montXCGetLastTrimmedMaxSD`.
///
/// Returns the 95th percentile value of the SD map used for weighted cross-correlation.
pub fn mont_xc_get_last_trimmed_max_sd() -> f64 {
    f32::from_bits(S_LAST_TRIMMED_MAX_SD.load(Ordering::Relaxed)) as f64
}

/// C `montXCGetLastRunnersUp`.
///
/// Returns up to `maxPairs` X,Y alternative displacements into `disps`.
pub fn mont_xc_get_last_runners_up(disps: &mut [f32], maxPairs: i32) {
    let mut i: i32;
    i = 0;
    while i < 2 * MAX_RUNNERS_UP.min(maxPairs) {
        disps[i as usize] = f32::from_bits(S_LAST_RUNNERS_UP[i as usize].load(Ordering::Relaxed));
        i += 1;
    }
    i = 2 * MAX_RUNNERS_UP;
    while i < 2 * maxPairs {
        disps[i as usize] = -1.0e30f64 as f32;
        i += 1;
    }
}

/// C `montxcgetlastrunnersup` — Fortran wrapper for `montXCGetLastRunnersUp`.
pub fn montxcgetlastrunnersup(disps: &mut [f32], maxPairs: &i32) {
    mont_xc_get_last_runners_up(disps, *maxPairs);
}

/// C `rowOfThreeCorrs`.
///
/// Computes cross-correlation coefficients at three adjacent shifts in X in subareas of two
/// images.
pub fn row_of_three_corrs(
    array: &[f32],
    brray: &[f32],
    nxDim: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    delX: i32,
    delY: i32,
    aWeights: Option<&[f32]>,
    bWeights: Option<&[f32]>,
    nxWgt: i32,
    mut binning: i32,
    wgtXoffset: i32,
    wgtYoffset: i32,
    corr1: &mut f32,
    corr2: &mut f32,
    corr3: &mut f32,
    mut sumArr1: Option<&mut [f64]>,
    mut sumArr2: Option<&mut [f64]>,
    mut sumArr3: Option<&mut [f64]>,
) {
    let mut ix: i32;
    let mut iy: i32;
    let mut aBase: i32;
    let nsum: i32 = 0;
    let end: i32 = 0;
    let mut bBase: i32;
    let mut aWgtBase: i32;
    let mut bWgtBase: i32;
    let mut abSum1: f64;
    let mut abSum2: f64;
    let mut abSum3: f64;
    let mut aSumSq1: f64;
    let mut aSumSq2: f64;
    let mut aSumSq3: f64;
    let mut aSum1: f64;
    let mut aSum2: f64;
    let mut aSum3: f64;
    let mut bSumSq1: f64;
    let mut bSumSq2: f64;
    let mut bSumSq3: f64;
    let mut bSumSq: f64;
    let denom: f64 = 0.;
    let amean: f64 = 0.;
    let bmean: f64 = 0.;
    let mut bSum1: f64;
    let mut bSum2: f64;
    let mut bSum3: f64;
    let mut bSum: f64;
    let mut wSum2: f64;
    let mut wSum1: f64;
    let mut wSum3: f64;
    let mut abTmp1: f64;
    let mut abTmp2: f64;
    let mut abTmp3: f64;
    let mut aTmp1: f64;
    let mut aTmpSq1: f64;
    let mut aTmp2: f64;
    let mut aTmpSq2: f64;
    let mut aTmp3: f64;
    let mut aTmpSq3: f64;
    let mut bTmp1: f64;
    let mut bTmp2: f64;
    let mut bTmp3: f64;
    let mut bTmpSq1: f64;
    let mut bTmpSq2: f64;
    let mut bTmpSq3: f64;
    let mut wTmp2: f64;
    let mut wTmp1: f64;
    let mut wTmp3: f64;
    let mut aval: f32;
    let mut bval1: f32;
    let mut bval2: f32;
    let mut bval3: f32;
    let mut bval: f32;
    let mut wgt: f32;
    let mut wgt1: f32;
    let mut wgt3: f32;
    let mut awgt: f32;
    let mut numThreads: i32;
    let maxThreads: i32 = 8;

    numThreads = ((((ix1 - ix0) as f64 * (iy1 - iy0) as f64).sqrt() / 80.0f64 + 0.5f64).floor()
        as i32 as f64
        * (if aWeights.is_some() { 2.0f64 } else { 1.0f64 })) as i32;
    numThreads = numThreads.clamp(1, maxThreads);
    numThreads = crate::imod::libcfshr::b3dutil::num_omp_threads(numThreads);

    binning = 1.max(binning);

    abSum1 = 0.;
    abSum2 = 0.;
    abSum3 = 0.;
    aSum2 = 0.;
    aSumSq1 = 0.;
    aSum1 = 0.;
    aSumSq2 = 0.;
    aSum3 = 0.;
    aSumSq3 = 0.;
    bSumSq1 = 0.;
    bSumSq2 = 0.;
    bSumSq3 = 0.;
    bSum1 = 0.;
    bSum2 = 0.;
    bSum3 = 0.;
    wSum1 = 0.;
    wSum2 = 0.;
    wSum3 = 0.;

    // The source's `#pragma omp parallel for` reduction runs this loop serially here; the
    // reduction order is the same as an OpenMP run with one thread.
    iy = iy0;
    while iy <= iy1 {
        aBase = iy * nxDim;
        bBase = (iy - delY) * nxDim - delX;
        aWgtBase = (iy / binning + wgtYoffset) * nxWgt + wgtXoffset;
        bWgtBase = ((iy - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
        bSumSq = 0.;
        bSum = 0.;
        abTmp1 = 0.;
        abTmp2 = 0.;
        abTmp3 = 0.;
        wTmp2 = 0.;
        wTmp1 = 0.;
        wTmp3 = 0.;
        aTmp1 = 0.;
        aTmpSq1 = 0.;
        aTmp2 = 0.;
        aTmpSq2 = 0.;
        aTmp3 = 0.;
        aTmpSq3 = 0.;
        bTmp1 = 0.;
        bTmp2 = 0.;
        bTmp3 = 0.;
        bTmpSq1 = 0.;
        bTmpSq2 = 0.;
        bTmpSq3 = 0.;

        if let (Some(aw), Some(bw)) = (aWeights, bWeights) {
            // Do fast loop
            ix = ix0;
            while ix <= ix1 {
                awgt = aw[(ix / binning + aWgtBase) as usize];
                wgt1 = awgt * bw[((ix + 1 - delX) / binning + bWgtBase) as usize];
                wgt = awgt * bw[((ix - delX) / binning + bWgtBase) as usize];
                wgt3 = awgt * bw[((ix - 1 - delX) / binning + bWgtBase) as usize];
                wTmp1 += wgt1 as f64;
                wTmp2 += wgt as f64;
                wTmp3 += wgt3 as f64;
                aval = array[(ix + aBase) as usize];
                bval = brray[(ix + bBase) as usize];
                bval1 = brray[(ix + bBase + 1) as usize];
                bval3 = brray[(ix + bBase - 1) as usize];
                aTmp1 += (aval * wgt1) as f64;
                aTmp2 += (aval * wgt) as f64;
                aTmp3 += (aval * wgt3) as f64;
                bTmp1 += (bval1 * wgt1) as f64;
                bTmp2 += (bval * wgt) as f64;
                bTmp3 += (bval3 * wgt3) as f64;
                aTmpSq1 += (aval * aval * wgt1) as f64;
                aTmpSq2 += (aval * aval * wgt) as f64;
                aTmpSq3 += (aval * aval * wgt3) as f64;
                bTmpSq1 += (bval1 * bval1 * wgt1) as f64;
                bTmpSq2 += (bval * bval * wgt) as f64;
                bTmpSq3 += (bval3 * bval3 * wgt3) as f64;
                abTmp1 += (aval * bval1 * wgt1) as f64;
                abTmp2 += (aval * bval * wgt) as f64;
                abTmp3 += (aval * bval3 * wgt3) as f64;
                ix += 1;
            }
        } else {
            // Add first position
            aval = array[(ix0 + aBase) as usize];
            bval1 = brray[(ix0 + bBase + 1) as usize];
            bval2 = brray[(ix0 + bBase) as usize];
            bval3 = brray[(ix0 + bBase - 1) as usize];
            aTmp2 += aval as f64;
            aTmpSq2 += (aval * aval) as f64;
            abTmp1 += (aval * bval1) as f64;
            abTmp2 += (aval * bval2) as f64;
            abTmp3 += (aval * bval3) as f64;
            bTmp2 += bval2 as f64;
            bTmp3 += (bval3 + bval2) as f64;
            bTmpSq2 += (bval2 * bval2) as f64;
            bTmpSq3 += (bval3 * bval3 + bval2 * bval2) as f64;

            // Do fast loop
            ix = ix0 + 1;
            while ix < ix1 {
                aval = array[(ix + aBase) as usize];
                bval = brray[(ix + bBase) as usize];
                aTmp2 += aval as f64;
                bSum += bval as f64;
                aTmpSq2 += (aval * aval) as f64;
                bSumSq += (bval * bval) as f64;
                abTmp1 += (aval * brray[(ix + bBase + 1) as usize]) as f64;
                abTmp2 += (aval * bval) as f64;
                abTmp3 += (aval * brray[(ix + bBase - 1) as usize]) as f64;
                ix += 1;
            }
            bTmp1 += bSum;
            bTmp2 += bSum;
            bTmp3 += bSum;
            bTmpSq1 += bSumSq;
            bTmpSq2 += bSumSq;
            bTmpSq3 += bSumSq;

            // Add last position
            aval = array[(ix1 + aBase) as usize];
            bval1 = brray[(ix1 + bBase + 1) as usize];
            bval2 = brray[(ix1 + bBase) as usize];
            bval3 = brray[(ix1 + bBase - 1) as usize];
            aTmp2 += aval as f64;
            aTmpSq2 += (aval * aval) as f64;
            abTmp1 += (aval * bval1) as f64;
            abTmp2 += (aval * bval2) as f64;
            abTmp3 += (aval * bval3) as f64;
            bTmp1 += (bval1 + bval2) as f64;
            bTmp2 += bval2 as f64;
            bTmpSq1 += (bval1 * bval1 + bval2 * bval2) as f64;
            bTmpSq2 += (bval2 * bval2) as f64;

            wTmp1 += (ix1 + 1 - ix0) as f64;
            wTmp2 += (ix1 + 1 - ix0) as f64;
            wTmp3 += (ix1 + 1 - ix0) as f64;
            aTmp3 = aTmp2;
            aTmp1 = aTmp3;
            aTmpSq3 = aTmpSq2;
            aTmpSq1 = aTmpSq3;
        }

        // Accumulate
        wSum1 += wTmp1;
        wSum2 += wTmp2;
        wSum3 += wTmp3;
        aSum1 += aTmp1;
        aSum2 += aTmp2;
        aSum3 += aTmp3;
        aSumSq1 += aTmpSq1;
        aSumSq2 += aTmpSq2;
        aSumSq3 += aTmpSq3;
        abSum1 += abTmp1;
        abSum2 += abTmp2;
        abSum3 += abTmp3;
        bSum1 += bTmp1;
        bSum2 += bTmp2;
        bSum3 += bTmp3;
        bSumSq1 += bTmpSq1;
        bSumSq2 += bTmpSq2;
        bSumSq3 += bTmpSq3;
        iy += 1;
    }

    *corr1 = unsafe {
        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
            aSum1,
            aSumSq1,
            bSum1,
            bSumSq1,
            abSum1,
            wSum1,
            sumArr1.as_deref_mut(),
            "",
        )
    } as f32;
    *corr2 = unsafe {
        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
            aSum2,
            aSumSq2,
            bSum2,
            bSumSq2,
            abSum2,
            wSum2,
            sumArr2.as_deref_mut(),
            "",
        )
    } as f32;
    *corr3 = unsafe {
        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
            aSum3,
            aSumSq3,
            bSum3,
            bSumSq3,
            abSum3,
            wSum3,
            sumArr3.as_deref_mut(),
            "",
        )
    } as f32;
}

/// C `columnOfThreeCorrs`.
///
/// Computes cross-correlation coefficients at three adjacent shifts in Y in subareas of two
/// images.  Arguments are the same as for `rowOfThreeCorrs`.
pub fn column_of_three_corrs(
    array: &[f32],
    brray: &[f32],
    nxDim: i32,
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    delX: i32,
    delY: i32,
    aWeights: Option<&[f32]>,
    bWeights: Option<&[f32]>,
    nxWgt: i32,
    binning: i32,
    wgtXoffset: i32,
    wgtYoffset: i32,
    corr1: &mut f32,
    corr2: &mut f32,
    corr3: &mut f32,
    mut sumArr1: Option<&mut [f64]>,
    mut sumArr2: Option<&mut [f64]>,
    mut sumArr3: Option<&mut [f64]>,
) {
    let mut ix: i32;
    let mut iy: i32;
    let mut aBase: i32;
    let nsum: i32 = 0;
    let end: i32 = 0;
    let mut bBase: i32;
    let mut aWgtBase: i32;
    let mut bWgtBase: i32;
    let mut bWgtBase1: i32;
    let mut bWgtBase3: i32;
    let mut wInd: i32;
    let mut abSum1: f64;
    let mut abSum2: f64;
    let mut abSum3: f64;
    let mut aSumSq1: f64;
    let mut aSumSq2: f64;
    let mut aSumSq3: f64;
    let mut aSum1: f64;
    let mut aSum2: f64;
    let mut aSum3: f64;
    let mut bSumSq1: f64;
    let mut bSumSq2: f64;
    let mut bSumSq3: f64;
    let mut bSumSq: f64;
    let denom: f64 = 0.;
    let amean: f64 = 0.;
    let bmean: f64 = 0.;
    let mut bSum1: f64;
    let mut bSum2: f64;
    let mut bSum3: f64;
    let mut bSum: f64;
    let mut wSum2: f64;
    let mut wSum1: f64;
    let mut wSum3: f64;
    let mut abTmp1: f64;
    let mut abTmp2: f64;
    let mut abTmp3: f64;
    let mut aTmp1: f64;
    let mut aTmpSq1: f64;
    let mut aTmp2: f64;
    let mut aTmpSq2: f64;
    let mut aTmp3: f64;
    let mut aTmpSq3: f64;
    let mut aTmpSq: f64;
    let mut bTmp1: f64;
    let mut bTmp2: f64;
    let mut bTmp3: f64;
    let mut bTmpSq1: f64;
    let mut bTmpSq2: f64;
    let mut bTmpSq3: f64;
    let mut wTmp2: f64;
    let mut wTmp1: f64;
    let mut wTmp3: f64;
    let mut aTmp: f64;
    let mut aval: f32;
    let mut bval1: f32;
    let mut bval2: f32;
    let mut bval3: f32;
    let mut bval: f32;
    let mut wgt: f32;
    let mut wgt1: f32;
    let mut wgt3: f32;
    let mut awgt: f32;
    let mut numThreads: i32;
    let maxThreads: i32 = 8;

    numThreads = ((((ix1 - ix0) as f64 * (iy1 - iy0) as f64).sqrt() / 80.0f64 + 0.5f64).floor()
        as i32 as f64
        * (if aWeights.is_some() { 2.0f64 } else { 1.0f64 })) as i32;
    numThreads = numThreads.clamp(1, maxThreads);
    numThreads = crate::imod::libcfshr::b3dutil::num_omp_threads(numThreads);

    abSum1 = 0.;
    abSum2 = 0.;
    abSum3 = 0.;
    aSum2 = 0.;
    aSumSq1 = 0.;
    aSum1 = 0.;
    aSumSq2 = 0.;
    aSum3 = 0.;
    aSumSq3 = 0.;
    bSumSq1 = 0.;
    bSumSq2 = 0.;
    bSumSq3 = 0.;
    bSum1 = 0.;
    bSum2 = 0.;
    bSum3 = 0.;
    wSum1 = 0.;
    wSum2 = 0.;
    wSum3 = 0.;

    iy = iy0 + (if aWeights.is_some() { 0 } else { 1 });
    while iy <= iy1 - (if aWeights.is_some() { 0 } else { 1 }) {
        aBase = iy * nxDim;
        bBase = (iy - delY) * nxDim - delX;
        bSumSq = 0.;
        bSum = 0.;
        abTmp1 = 0.;
        abTmp2 = 0.;
        abTmp3 = 0.;
        wTmp2 = 0.;
        wTmp1 = 0.;
        wTmp3 = 0.;
        aTmp1 = 0.;
        aTmpSq1 = 0.;
        aTmp2 = 0.;
        aTmpSq2 = 0.;
        aTmp3 = 0.;
        aTmpSq3 = 0.;
        bTmp1 = 0.;
        bTmp2 = 0.;
        bTmp3 = 0.;
        bTmpSq1 = 0.;
        bTmpSq2 = 0.;
        bTmpSq3 = 0.;

        if let (Some(aw), Some(bw)) = (aWeights, bWeights) {
            // Weighting: Set up weight bases
            aWgtBase = (iy / binning + wgtYoffset) * nxWgt + wgtXoffset;
            bWgtBase1 = ((iy + 1 - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
            bWgtBase = ((iy - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;
            bWgtBase3 = ((iy - 1 - delY) / binning + wgtYoffset) * nxWgt + wgtXoffset;

            ix = ix0;
            while ix <= ix1 {
                awgt = aw[(ix / binning + aWgtBase) as usize];
                wInd = (ix - delX) / binning;
                wgt1 = awgt * bw[(wInd + bWgtBase1) as usize];
                wgt = awgt * bw[(wInd + bWgtBase) as usize];
                wgt3 = awgt * bw[(wInd + bWgtBase3) as usize];
                wTmp1 += wgt1 as f64;
                wTmp2 += wgt as f64;
                wTmp3 += wgt3 as f64;
                aval = array[(ix + aBase) as usize];
                bval = brray[(ix + bBase) as usize];
                bval1 = brray[(ix + bBase + nxDim) as usize];
                bval3 = brray[(ix + bBase - nxDim) as usize];
                aTmp1 += (aval * wgt1) as f64;
                aTmp2 += (aval * wgt) as f64;
                aTmp3 += (aval * wgt3) as f64;
                bTmp1 += (bval1 * wgt1) as f64;
                bTmp2 += (bval * wgt) as f64;
                bTmp3 += (bval3 * wgt3) as f64;
                aTmpSq1 += (aval * aval * wgt1) as f64;
                aTmpSq2 += (aval * aval * wgt) as f64;
                aTmpSq3 += (aval * aval * wgt3) as f64;
                bTmpSq1 += (bval1 * bval1 * wgt1) as f64;
                bTmpSq2 += (bval * bval * wgt) as f64;
                bTmpSq3 += (bval3 * bval3 * wgt3) as f64;
                abTmp1 += (aval * bval1 * wgt1) as f64;
                abTmp2 += (aval * bval * wgt) as f64;
                abTmp3 += (aval * bval3 * wgt3) as f64;
                ix += 1;
            }
        } else {
            // No weights: do the restricted range
            aBase = iy * nxDim;
            bBase = (iy - delY) * nxDim - delX;
            bSumSq = 0.;
            bSum = 0.;
            abTmp1 = 0.;
            abTmp2 = 0.;
            abTmp3 = 0.;
            aTmp = 0.;
            aTmpSq = 0.;
            ix = ix0;
            while ix <= ix1 {
                aval = array[(ix + aBase) as usize];
                aTmp += aval as f64;
                bSum += brray[(ix + bBase) as usize] as f64;
                aTmpSq += (aval * aval) as f64;
                bSumSq += (brray[(ix + bBase) as usize] * brray[(ix + bBase) as usize]) as f64;
                abTmp1 += (aval * brray[(ix + bBase + nxDim) as usize]) as f64;
                abTmp2 += (aval * brray[(ix + bBase) as usize]) as f64;
                abTmp3 += (aval * brray[(ix + bBase - nxDim) as usize]) as f64;
                ix += 1;
            }
            wTmp1 += (ix1 + 1 - ix0) as f64;
            wTmp2 += (ix1 + 1 - ix0) as f64;
            wTmp3 += (ix1 + 1 - ix0) as f64;
            bTmp3 = bSum;
            bTmp2 = bTmp3;
            bTmp1 = bTmp2;
            bTmpSq3 = bSumSq;
            bTmpSq2 = bTmpSq3;
            bTmpSq1 = bTmpSq2;
            aTmp3 = aTmp;
            aTmp2 = aTmp3;
            aTmp1 = aTmp2;
            aTmpSq3 = aTmpSq;
            aTmpSq2 = aTmpSq3;
            aTmpSq1 = aTmpSq2;
        }

        // Accumulate
        wSum1 += wTmp1;
        wSum2 += wTmp2;
        wSum3 += wTmp3;
        aSum1 += aTmp1;
        aSum2 += aTmp2;
        aSum3 += aTmp3;
        aSumSq1 += aTmpSq1;
        aSumSq2 += aTmpSq2;
        aSumSq3 += aTmpSq3;
        abSum1 += abTmp1;
        abSum2 += abTmp2;
        abSum3 += abTmp3;
        bSum1 += bTmp1;
        bSum2 += bTmp2;
        bSum3 += bTmp3;
        bSumSq1 += bTmpSq1;
        bSumSq2 += bTmpSq2;
        bSumSq3 += bTmpSq3;

        iy += 1;
    }

    if aWeights.is_none() {
        // For no weights, complete the first row
        aBase = iy0 * nxDim;
        bBase = (iy0 - delY) * nxDim - delX;

        bTmp1 = 0.;
        bTmp2 = 0.;
        bTmp3 = 0.;
        bTmpSq1 = 0.;
        bTmpSq2 = 0.;
        bTmpSq3 = 0.;
        abTmp1 = 0.;
        abTmp2 = 0.;
        abTmp3 = 0.;
        aTmp = 0.;
        aTmpSq = 0.;
        ix = ix0;
        while ix <= ix1 {
            aval = array[(ix + aBase) as usize];
            bval1 = brray[(ix + bBase + nxDim) as usize];
            bval2 = brray[(ix + bBase) as usize];
            bval3 = brray[(ix + bBase - nxDim) as usize];
            aTmp += aval as f64;
            aTmpSq += (aval * aval) as f64;
            abTmp1 += (aval * bval1) as f64;
            abTmp2 += (aval * bval2) as f64;
            abTmp3 += (aval * bval3) as f64;
            bTmp2 += bval2 as f64;
            bTmp3 += (bval3 + bval2) as f64;
            bTmpSq2 += (bval2 * bval2) as f64;
            bTmpSq3 += (bval3 * bval3 + bval2 * bval2) as f64;
            ix += 1;
        }

        // Complete the last row
        aBase = iy1 * nxDim;
        bBase = (iy1 - delY) * nxDim - delX;
        ix = ix0;
        while ix <= ix1 {
            aval = array[(ix + aBase) as usize];
            bval1 = brray[(ix + bBase + nxDim) as usize];
            bval2 = brray[(ix + bBase) as usize];
            bval3 = brray[(ix + bBase - nxDim) as usize];
            aTmp += aval as f64;
            aTmpSq += (aval * aval) as f64;
            abTmp1 += (aval * bval1) as f64;
            abTmp2 += (aval * bval2) as f64;
            abTmp3 += (aval * bval3) as f64;
            bTmp1 += (bval1 + bval2) as f64;
            bTmp2 += bval2 as f64;
            bTmpSq1 += (bval1 * bval1 + bval2 * bval2) as f64;
            bTmpSq2 += (bval2 * bval2) as f64;
            ix += 1;
        }

        // Accumulate
        wSum1 += (2 * (ix1 + 1 - ix0)) as f64;
        wSum2 += (2 * (ix1 + 1 - ix0)) as f64;
        wSum3 += (2 * (ix1 + 1 - ix0)) as f64;
        aSum1 += aTmp;
        aSum2 += aTmp;
        aSum3 += aTmp;
        aSumSq1 += aTmpSq;
        aSumSq2 += aTmpSq;
        aSumSq3 += aTmpSq;
        abSum1 += abTmp1;
        abSum2 += abTmp2;
        abSum3 += abTmp3;
        bSum1 += bTmp1;
        bSum2 += bTmp2;
        bSum3 += bTmp3;
        bSumSq1 += bTmpSq1;
        bSumSq2 += bTmpSq2;
        bSumSq3 += bTmpSq3;
    }

    *corr1 = unsafe {
        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
            aSum1,
            aSumSq1,
            bSum1,
            bSumSq1,
            abSum1,
            wSum1,
            sumArr1.as_deref_mut(),
            "",
        )
    } as f32;
    *corr2 = unsafe {
        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
            aSum2,
            aSumSq2,
            bSum2,
            bSumSq2,
            abSum2,
            wSum2,
            sumArr2.as_deref_mut(),
            "",
        )
    } as f32;
    *corr3 = unsafe {
        crate::imod::libcfshr::filtxcorr::weighted_corr_from_sums(
            aSum3,
            aSumSq3,
            bSum3,
            bSumSq3,
            abSum3,
            wSum3,
            sumArr3.as_deref_mut(),
            "",
        )
    } as f32;
}

/// C `montXCFindBestCorr`.
///
/// Finds the shift with the best cross-correlation coefficient between overlapping subareas
/// on two images.
pub fn mont_xc_find_best_corr(
    array: &[f32],
    brray: &[f32],
    nxDim: i32,
    nx: i32,
    ny: i32,
    nxTrim: i32,
    nyTrim: i32,
    ixStart: i32,
    ixEnd: i32,
    iyStart: i32,
    iyEnd: i32,
    delX: &mut f32,
    delY: &mut f32,
    corr: &mut f32,
    maxDist: f32,
    aWeights: Option<&[f32]>,
    bWeights: Option<&[f32]>,
    nxWgt: i32,
    binning: i32,
    wgtXoffset: i32,
    wgtYoffset: i32,
    threshCCC: f32,
    mut bestSumArr: Option<&mut [f64]>,
) {
    let mut corrs: [[f32; 3]; 3] = [[0.; 3]; 3];
    let mut corrTmp: [[f32; 3]; 3] = [[0.; 3]; 3];
    let mut cccMax: f32;
    let mut sumArrs: [[[f64; 6]; 3]; 3] = [[[0.; 6]; 3]; 3];
    let mut sumTmp: [[[f64; 6]; 3]; 3] = [[[0.; 6]; 3]; 3];
    let mut done: [[i32; 3]; 3] = [[0; 3]; 3];
    let first: i32 = 1;
    let mut curDelX: i32 = (*delX as f64 + 0.5f64).floor() as i32;
    let mut curDelY: i32 = (*delY as f64 + 0.5f64).floor() as i32;
    let mut ix: i32;
    let mut iy: i32;
    let mut ixMax: i32;
    let mut iyMax: i32;
    let mut ix0: i32;
    let mut ix1: i32;
    let mut iy0: i32;
    let mut iy1: i32;
    let nc: i32;
    let mut ind: i32;
    let mut needCol: i32 = -1;

    ix0 = ixStart.max(nxTrim + curDelX);
    ix1 = ixEnd.min(nx + curDelX - nxTrim);
    iy0 = iyStart.max(nyTrim + curDelY);
    iy1 = iyEnd.min(ny + curDelY - nyTrim);
    // C `fflush(stdout)` at montagexcorr.c:1349.
    let _ = ImodFile::Stdout.flush();
    iy = 0;
    while iy < 3 {
        done[iy as usize][0] = 0;
        done[iy as usize][1] = 0;
        done[iy as usize][2] = 0;
        iy += 1;
    }

    while ((curDelX as f32 - *delX) as f64).powf(2.0f64)
        + ((curDelY as f32 - *delY) as f64).powf(2.0f64)
        < (maxDist * maxDist) as f64
    {
        ix0 = ixStart.max(nxTrim + curDelX - 1);
        ix1 = ixEnd.min(nx + curDelX + 1 - nxTrim);
        iy0 = iyStart.max(nyTrim + curDelY - 1);
        iy1 = iyEnd.min(ny + curDelY + 1 - nyTrim);

        if needCol >= 0 {
            let nc = needCol;
            // `&corrs[0][nc]`, `&corrs[1][nc]`, `&corrs[2][nc]` are three disjoint elements
            // in three different rows; destructuring the outer array splits the borrow at
            // exactly the boundary the C pointers do.
            let [c0, c1, c2] = &mut corrs;
            let [s0, s1, s2] = &mut sumArrs;
            column_of_three_corrs(
                array,
                brray,
                nxDim,
                ix0,
                ix1,
                iy0,
                iy1,
                curDelX + nc - 1,
                curDelY,
                aWeights,
                bWeights,
                nxWgt,
                binning,
                wgtXoffset,
                wgtYoffset,
                &mut c0[nc as usize],
                &mut c1[nc as usize],
                &mut c2[nc as usize],
                Some(&mut s0[nc as usize][..]),
                Some(&mut s1[nc as usize][..]),
                Some(&mut s2[nc as usize][..]),
            );
            done[0][nc as usize] = 1;
            done[1][nc as usize] = 1;
            done[2][nc as usize] = 1;
        }

        iy = 0;
        while iy < 3 {
            if !(done[iy as usize][0] != 0
                && done[iy as usize][1] != 0
                && done[iy as usize][2] != 0)
            {
                let [c0, c1, c2] = &mut corrs[iy as usize];
                let [s0, s1, s2] = &mut sumArrs[iy as usize];
                row_of_three_corrs(
                    array,
                    brray,
                    nxDim,
                    ix0,
                    ix1,
                    iy0,
                    iy1,
                    curDelX,
                    curDelY + iy - 1,
                    aWeights,
                    bWeights,
                    nxWgt,
                    binning,
                    wgtXoffset,
                    wgtYoffset,
                    c0,
                    c1,
                    c2,
                    Some(&mut s0[..]),
                    Some(&mut s1[..]),
                    Some(&mut s2[..]),
                );
                done[iy as usize][0] = 1;
                done[iy as usize][1] = 1;
                done[iy as usize][2] = 1;
            }
            iy += 1;
        }

        // Find maximum and copy all the correlations and clear out done to prepare for shift
        ixMax = 1;
        iyMax = 1;
        cccMax = corrs[1][1];
        iy = 0;
        while iy < 3 {
            ix = 0;
            while ix < 3 {
                corrTmp[iy as usize][ix as usize] = corrs[iy as usize][ix as usize];
                done[iy as usize][ix as usize] = 0;
                ind = 0;
                while ind < 6 {
                    sumTmp[iy as usize][ix as usize][ind as usize] =
                        sumArrs[iy as usize][ix as usize][ind as usize];
                    ind += 1;
                }

                // Do not move away from the middle as the maximum if two are equal
                if corrs[iy as usize][ix as usize] > cccMax
                    || corrs[iy as usize][ix as usize] == cccMax && (ixMax != 1 || iyMax != 1)
                {
                    ixMax = ix;
                    iyMax = iy;
                    cccMax = corrs[iy as usize][ix as usize];
                }
                ix += 1;
            }
            iy += 1;
        }

        // Done if the max is still in the middle
        curDelX += ixMax - 1;
        curDelY += iyMax - 1;
        if ixMax == 1 && iyMax == 1 || cccMax < threshCCC {
            *corr = cccMax;
            *delX = (curDelX as f64
                + (if ixMax == 1 && iyMax == 1 {
                    0.
                } else {
                    crate::imod::libcfshr::filtxcorr::parabolic_fit_position(
                        corrs[1][0],
                        corrs[1][1],
                        corrs[1][2],
                    )
                })) as f32;
            *delY = (curDelY as f64
                + (if ixMax == 1 && iyMax == 1 {
                    0.
                } else {
                    crate::imod::libcfshr::filtxcorr::parabolic_fit_position(
                        corrs[0][1],
                        corrs[1][1],
                        corrs[2][1],
                    )
                })) as f32;
            if aWeights.is_some() {
                let bsa = bestSumArr.as_deref_mut().unwrap();
                ind = 0;
                while ind < 6 {
                    // The source indexes [ixMax][iyMax] here while it fills sumArrs as
                    // [iy][ix] everywhere else; that transposition is preserved.
                    bsa[ind as usize] = sumArrs[ixMax as usize][iyMax as usize][ind as usize];
                    ind += 1;
                }
            }
            return;
        }

        // Set up to do a column (first) if shifting in X, and shift the correlations
        if ixMax != 1 {
            needCol = ixMax;
        }
        iy = 0.max(iyMax - 1);
        while iy <= 2.min(iyMax + 1) {
            ix = 0.max(ixMax - 1);
            while ix <= 2.min(ixMax + 1) {
                done[(iy + 1 - iyMax) as usize][(ix + 1 - ixMax) as usize] = 1;
                corrs[(iy + 1 - iyMax) as usize][(ix + 1 - ixMax) as usize] =
                    corrTmp[iy as usize][ix as usize];
                ind = 0;
                while ind < 6 {
                    sumArrs[(iy + 1 - iyMax) as usize][(ix + 1 - ixMax) as usize][ind as usize] =
                        sumTmp[iy as usize][ix as usize][ind as usize];
                    ind += 1;
                }
                ix += 1;
            }
            iy += 1;
        }
    }

    // Too far, return 0.
    *corr = 0 as i32 as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_sizes_and_binning_follow_source_edge_geometry() {
        let pieces = [100, 120];
        let overlap = [20, 25];
        let mut use_indent = 0;
        let mut boxed = [0; 2];
        let mut extra = [0; 2];
        let mut xpad = 0;
        let mut ypad = 0;
        let mut maximum = 0;
        mont_xc_basic_sizes(
            0,
            1,
            4,
            &pieces,
            &overlap,
            2.0,
            0.2,
            0.1,
            5,
            &mut use_indent,
            &mut boxed,
            &mut extra,
            &mut xpad,
            &mut ypad,
            &mut maximum,
        );
        assert_eq!(use_indent, 4);
        assert!(boxed[0] >= 12 && boxed[1] > 0 && xpad >= boxed[0] && ypad >= boxed[1]);
        let mut padded = 0;
        let mut area = 0;
        assert!(
            mont_xc_find_binning(
                4,
                200,
                4,
                &pieces,
                &overlap,
                2.,
                0.2,
                0.1,
                5,
                &mut padded,
                &mut area
            ) >= 1
        );
    }
}
