//! Translation of `IMOD/libcfshr/surfacesort.c`: sorting points onto two
//! surfaces.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use core::cell::Cell;
use std::io::Write;

use super::robuststat::{rs_mad_median_outliers, rs_median, rs_sort_indexed_floats};
use super::simplestat::{ls_fit2, ls_fit3, sums_to_avg_sd};

/// Complete function inventory for `surfacesort.c`.
pub const SURFACE_SORT_SOURCE_FUNCTIONS: &[&str] = &[
    "setSurfSortParam",
    "setsurfsortparam",
    "surfaceSort",
    "surfacesort",
];

/// Original `DTOR` (`surfacesort.c:21`).
pub const DTOR: f64 = 0.017453293;

/* DOC_SECTION PARAMETERS */
/* DOC_CODE Algorithm and Parameters for surfaceSort */
/*
This routine works as follows: ^
First it fits a plane to all of the points and rotates the points to make that plane
level.  Analysis proceeds with rotated positions. ^
For rapid access to neighboring points, it sets up a grid of squares (at sGridSpacing)
and list of delta values to grid positions in successively wider rings around one
grid square.  Then it makes lists of points within each grid square. ^
For each point (not marked as an outlier by values of -1 in the group array) it then
searches for the neighboring point at the steepest angle with respect to it, looking
at up to sMaxAngleNeigh neighbors. ^
Duplicate pairs are allowed to form, and are eliminated after sorting. ^
The points are sorted by steepest angle, and then it determines a set of pairs to use
for building clusters, as controlled by parameters sSteepestRatio, sNumMinForAmax,
sAngleMax, sNumMinForArelax, and sAngleRelax. ^
Given this set of pairs, it then makes a list of the separation in Z between each pair
and analyzes for MAD-Median outliers with criterion sOutlierCrit.  These outliers are
eliminated from cluster formation, as are the ones identified by the caller.
Pairs are always considered in order from the steepest angle downward.  To build a
cluster, it starts with the next point that hasn't been added to a cluster yet, putting
it and its pair on a list of cluster points to check.  Each point on the cluster list
is checked for whether it occurs in a pair at a steep angle, and if so its mate is
then added to the opposite group. ^
For each cluster, a central position and an average delta Z are computed. ^
Finally, it searches in progressively bigger rings around the clusters for grid
squares where there are unassigned points.  When it finds an unassigned point, it
collects the nearest assigned points in the neighborhood up to sMaxNumFit or up to
a distance of sMaxFitDist.  If this includes points from both surfaces and there are
at least sBiplaneMinFit points, it fits a pair of planes to all points; otherwise it
fits a single plane to just the points from one surface if there are at least
sPlaneMinFit points for this; otherwise it simply gets a mean of Z positions on the
top and bottom if possible.  One way or another this gives an estimate of the top and
bottom Z position at the unassigned point, and it assigns the point based on which Z
it is closer to.  In fact, this process is done in two rounds, and on the first round
assignment is deferred if the disparity between the point and estimated surface Z
values as a percentage of the Z separation exceeds sMaxRound1Dist. ^
^
The numbers in the comments below are index values for setSurfSortParam */

thread_local! {
    /// Spacing of squares for sorting points and accessing rings of points (0)
    static S_GRID_SPACING: Cell<f32> = const { Cell::new(50.) };
    /// Maximum neighbors to evaluate for finding neighbor with steepest angle (1)
    static S_MAX_ANGLE_NEIGH: Cell<i32> = const { Cell::new(50) };
    /// Use pairs with angle more than this fraction of very steepest angle (2)
    static S_STEEPEST_RATIO: Cell<f32> = const { Cell::new(0.5) };
    /// If there are fewer than this number of pairs, take pairs down to sAngleMax (3)
    static S_NUM_MIN_FOR_AMAX: Cell<i32> = const { Cell::new(10) };
    /// Angle to go down to if sSteepestRatio doesn't give enough pairs (4)
    static S_ANGLE_MAX: Cell<f32> = const { Cell::new(20.) };
    /// If fewer than this number of pairs, take pairs down to sAngleRelax (5)
    static S_NUM_MIN_FOR_ARELAX: Cell<i32> = const { Cell::new(3) };
    /// Angle to go down to if sAngleMax doesn't give enough pairs (6)
    static S_ANGLE_RELAX: Cell<f32> = const { Cell::new(5.) };
    /// Criterion for MAD-Median outlier elimination based on delta Z of a pair (7)
    static S_OUTLIER_CRIT: Cell<f32> = const { Cell::new(3.) };
    /// Maximum distance to search for neighboring points in plane fits (8)
    static S_MAX_FIT_DIST: Cell<f32> = const { Cell::new(2048.) };
    /// Maximum number of points in plane fits (9)
    static S_MAX_NUM_FIT: Cell<i32> = const { Cell::new(15) };
    /// Minimum # of points for fitting parallel planes (10)
    static S_BIPLANE_MIN_FIT: Cell<i32> = const { Cell::new(5) };
    /// Minimum # of points for fitting one plane (11)
    static S_PLANE_MIN_FIT: Cell<i32> = const { Cell::new(4) };
    /// Maximum % distance of Z value between nearest and other plane for
    /// deferring to 2nd round: values > 50 disable deferring points (12)
    static S_MAX_ROUND1_DIST: Cell<f32> = const { Cell::new(100.) };
    /// 1 for minimal output, 2 for exhaustive output (13)
    static S_DEBUG_LEVEL: Cell<i32> = const { Cell::new(0) };
}
/* END_CODE */
/* END_SECTION */

/// Original `setSurfSortParam` (`surfacesort.c:99`).
///
/// `B3DNINT` is `(int)floor(a + 0.5)`, not `round()`.
pub fn set_surf_sort_param(index: i32, value: f32) -> i32 {
    let nint = |v: f32| (v as f64 + 0.5).floor() as i32;
    match index {
        0 => S_GRID_SPACING.with(|c| c.set(value)),
        1 => S_MAX_ANGLE_NEIGH.with(|c| c.set(nint(value))),
        2 => S_STEEPEST_RATIO.with(|c| c.set(value)),
        3 => S_NUM_MIN_FOR_AMAX.with(|c| c.set(nint(value))),
        4 => S_ANGLE_MAX.with(|c| c.set(value)),
        5 => S_NUM_MIN_FOR_ARELAX.with(|c| c.set(nint(value))),
        6 => S_ANGLE_RELAX.with(|c| c.set(value)),
        7 => S_OUTLIER_CRIT.with(|c| c.set(value)),
        8 => S_MAX_FIT_DIST.with(|c| c.set(value)),
        9 => S_MAX_NUM_FIT.with(|c| c.set(nint(value))),
        10 => S_BIPLANE_MIN_FIT.with(|c| c.set(nint(value))),
        11 => S_PLANE_MIN_FIT.with(|c| c.set(nint(value))),
        12 => S_MAX_ROUND1_DIST.with(|c| c.set(value)),
        13 => S_DEBUG_LEVEL.with(|c| c.set(nint(value))),
        _ => return 1,
    }
    0
}

/// Original `setsurfsortparam` (`surfacesort.c:125`).
pub fn setsurfsortparam(index: &i32, value: &f32) -> i32 {
    set_surf_sort_param(*index, *value)
}

/// Original `surfaceSort` (`surfacesort.c:139`).
///
/// The source's twenty-one `B3DMALLOC` failure returns (error 1) are
/// unreachable here: a `Vec` allocation aborts rather than returning null.
pub fn surface_sort(xyz: &[f32], num_pts: i32, markers_in_group: i32, group: &mut [i32]) -> i32 {
    let s_grid_spacing = S_GRID_SPACING.with(|c| c.get());
    let s_max_angle_neigh = S_MAX_ANGLE_NEIGH.with(|c| c.get());
    let s_steepest_ratio = S_STEEPEST_RATIO.with(|c| c.get());
    let s_num_min_for_amax = S_NUM_MIN_FOR_AMAX.with(|c| c.get());
    let s_angle_max = S_ANGLE_MAX.with(|c| c.get());
    let s_num_min_for_arelax = S_NUM_MIN_FOR_ARELAX.with(|c| c.get());
    let s_angle_relax = S_ANGLE_RELAX.with(|c| c.get());
    let s_outlier_crit = S_OUTLIER_CRIT.with(|c| c.get());
    let s_max_fit_dist = S_MAX_FIT_DIST.with(|c| c.get());
    let s_max_num_fit = S_MAX_NUM_FIT.with(|c| c.get());
    let s_biplane_min_fit = S_BIPLANE_MIN_FIT.with(|c| c.get());
    let s_plane_min_fit = S_PLANE_MIN_FIT.with(|c| c.get());
    let s_max_round1_dist = S_MAX_ROUND1_DIST.with(|c| c.get());
    let s_debug_level = S_DEBUG_LEVEL.with(|c| c.get());

    let (mut afit, mut bfit, mut cfit) = (0.0f32, 0.0f32, 0.0f32);

    if num_pts < 0 {
        return 1;
    }
    if num_pts == 0 {
        return 0;
    }
    let Some(coordinate_count) = num_pts.checked_mul(3) else {
        return 1;
    };
    if xyz.len() < coordinate_count as usize
        || group.len() < num_pts as usize
        || !s_grid_spacing.is_finite()
        || s_grid_spacing <= 0.0
        || s_max_angle_neigh <= 0
        || s_max_num_fit <= 0
    {
        return 1;
    }
    if num_pts < 3 {
        group[0] = 1;
        if num_pts > 1 {
            group[1] = 2;
        }
        return 0;
    }

    /* Copy to rot arrays and fit a plane to all of the points */
    let mut xrot = vec![0.0f32; num_pts as usize];
    let mut yrot = vec![0.0f32; num_pts as usize];
    let mut zrot = vec![0.0f32; num_pts as usize];
    for i in 0..num_pts as usize {
        xrot[i] = xyz[3 * i];
        yrot[i] = xyz[3 * i + 1];
        zrot[i] = xyz[3 * i + 2];
    }
    ls_fit2(
        &xrot,
        &yrot,
        &zrot,
        num_pts,
        &mut afit,
        &mut bfit,
        Some(&mut cfit),
    );

    /* Find rotation angles and cosine and sines */
    let alpha = (bfit as f64).atan();
    let cosal = alpha.cos();
    let sinal = alpha.sin();
    let slope = afit as f64 / (cosal - bfit as f64 * sinal);
    let theta = -slope.atan();
    let costh = theta.cos();
    let sinth = theta.sin();

    /* Back-rotate by -alpha around X then -theta around Y */
    let mut xmin: f32 = 1.0e30;
    let mut ymin: f32 = 1.0e30;
    let mut xmax: f32 = -1.0e30;
    let mut ymax: f32 = -1.0e30;
    for i in 0..num_pts as usize {
        yrot[i] = (xyz[3 * i + 1] as f64 * cosal + xyz[3 * i + 2] as f64 * sinal) as f32;
        let zp: f32 = (-(xyz[3 * i + 1] as f64) * sinal + xyz[3 * i + 2] as f64 * cosal) as f32;
        xrot[i] = (xyz[3 * i] as f64 * costh - zp as f64 * sinth) as f32;
        zrot[i] = (xyz[3 * i] as f64 * sinth + zp as f64 * costh) as f32;
        // `B3DMIN`/`B3DMAX` keep the second operand when the comparison is
        // false, which is not what `f32::min`/`max` do on a NaN.
        xmin = if xmin < xrot[i] { xmin } else { xrot[i] };
        xmax = if xmax > xrot[i] { xmax } else { xrot[i] };
        ymin = if ymin < yrot[i] { ymin } else { yrot[i] };
        ymax = if ymax > yrot[i] { ymax } else { yrot[i] };
    }

    /* Set up grid and get arrays */
    let num_grid_x = (((xmax - xmin) / s_grid_spacing) as f64 + 1.) as i32;
    let num_grid_y = (((ymax - ymin) / s_grid_spacing) as f64 + 1.) as i32;
    let num_squares = num_grid_x * num_grid_y;
    let diagonal: f32 = ((num_grid_x as f64 - 1.) * (num_grid_x as f64 - 1.)
        + (num_grid_y as f64 - 1.) * (num_grid_y as f64 - 1.))
        .sqrt() as f32;
    let num_rings = (diagonal as f64 + 0.5).floor() as i32 + 1;
    let mut idx = vec![0i16; (4 * num_squares) as usize];
    let mut idy = vec![0i16; (4 * num_squares) as usize];
    let mut ring_start = vec![0i32; (num_rings + 2) as usize];
    let mut num_in_square = vec![0i32; num_squares as usize];
    let mut square_ind = vec![0i32; num_squares as usize];
    let mut point_lists = vec![0i32; num_pts as usize];
    let mut square_done = vec![0u8; num_squares as usize];
    let mut steep_angle = vec![0.0f32; num_pts as usize];
    let mut steep_neigh = vec![0i32; num_pts as usize];
    let mut sort_ind = vec![0i32; num_pts as usize];
    let mut xfit = vec![0.0f32; s_max_num_fit as usize];
    let mut yfit = vec![0.0f32; s_max_num_fit as usize];
    let mut zfit = vec![0.0f32; s_max_num_fit as usize];
    let mut grpfit = vec![0.0f32; s_max_num_fit as usize];

    /* Set up rings of delta values */
    let mut ind: i32 = 0;
    for ring in 0..num_rings {
        ring_start[ring as usize] = ind;
        for dy in -(num_grid_y - 1)..num_grid_y {
            for dx in -(num_grid_x - 1)..num_grid_x {
                if (((dx * dx + dy * dy) as f64).sqrt() + 0.5).floor() as i32 == ring {
                    idx[ind as usize] = dx as i16;
                    idy[ind as usize] = dy as i16;
                    ind += 1;
                }
            }
        }
    }
    ring_start[num_rings as usize] = ind;

    /* Sort the points into grid - first count how many in each square so
    indexes can be set up, then make the indexes, then put points into index
    list */
    for i in 0..num_squares as usize {
        num_in_square[i] = 0;
    }
    for i in 0..num_pts as usize {
        let sx = ((xrot[i] - xmin) / s_grid_spacing) as i32;
        let sy = ((yrot[i] - ymin) / s_grid_spacing) as i32;
        num_in_square[(sx + sy * num_grid_x) as usize] += 1;
    }
    let mut ind: i32 = 0;
    for i in 0..num_squares as usize {
        square_ind[i] = ind;
        ind += num_in_square[i];
        num_in_square[i] = 0;
        square_done[i] = 0;
    }
    for i in 0..num_pts as usize {
        let sx = ((xrot[i] - xmin) / s_grid_spacing) as i32;
        let sy = ((yrot[i] - ymin) / s_grid_spacing) as i32;
        let ind = sx + sy * num_grid_x;
        point_lists[(square_ind[ind as usize] + num_in_square[ind as usize]) as usize] = i as i32;
        num_in_square[ind as usize] += 1;
    }

    /* For each point, find the neighbor with the steepest angle
    Loop on the squares; loop on each point in the square
    For each point, loop on sequence of neighboring squares and on points in
    each square until reach maximum number of neighors */
    for ind in 0..num_squares {
        let sx = ind % num_grid_x;
        let sy = ind / num_grid_x;
        for isq in 0..num_in_square[ind as usize] {
            let ipt = point_lists[(square_ind[ind as usize] + isq) as usize];
            steep_angle[ipt as usize] = -1.;
            steep_neigh[ipt as usize] = -1;
            sort_ind[ipt as usize] = ipt;
            if markers_in_group != 0 && group[ipt as usize] < 0 {
                if s_debug_level > 1 {
                    let _ = ImodFile::Stdout
                        .write_all(format!("Skipping search for {ipt}\n").as_bytes());
                }
                continue;
            }
            let mut num_neigh = 0;
            let mut jdxy = 0;
            while jdxy < ring_start[num_rings as usize] && num_neigh < s_max_angle_neigh {
                let ix = sx + idx[jdxy as usize] as i32;
                let iy = sy + idy[jdxy as usize] as i32;
                if ix < 0 || ix >= num_grid_x || iy < 0 || iy >= num_grid_y {
                    jdxy += 1;
                    continue;
                }
                let jnd = ix + iy * num_grid_x;
                let mut jsq = 0;
                while jsq < num_in_square[jnd as usize] && num_neigh < s_max_angle_neigh {
                    let jpt = point_lists[(square_ind[jnd as usize] + jsq) as usize];
                    if ipt == jpt || (markers_in_group != 0 && group[jpt as usize] < 0) {
                        jsq += 1;
                        continue;
                    }
                    let pdx = xrot[ipt as usize] - xrot[jpt as usize];
                    let pdy = yrot[ipt as usize] - yrot[jpt as usize];
                    let angle = (((zrot[ipt as usize] - zrot[jpt as usize]) as f64).abs())
                        .atan2(((pdx * pdx + pdy * pdy) as f64).sqrt())
                        / DTOR;
                    let angle = angle as f32;
                    if angle > steep_angle[ipt as usize] {
                        steep_angle[ipt as usize] = angle;
                        steep_neigh[ipt as usize] = jpt;
                    }
                    num_neigh += 1;
                    jsq += 1;
                }
                jdxy += 1;
            }
        }
    }

    /* Zero group values now that markers have been used, if any */
    for i in 0..num_pts as usize {
        group[i] = 0;
    }

    /* Sort the points by steepest angle */
    rs_sort_indexed_floats(&steep_angle, &mut sort_ind, num_pts);
    if s_debug_level > 1 {
        for i in 0..num_pts as usize {
            let ipt = sort_ind[i];
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "pt %d  neigh %d  angle %f\n",
                    &[
                        CArg::Int(ipt as i64),
                        CArg::Int(steep_neigh[ipt as usize] as i64),
                        CArg::Dbl(steep_angle[ipt as usize] as f64),
                    ],
                )
                .as_bytes(),
            );
        }
    }
    let very_steepest = steep_angle[sort_ind[(num_pts - 1) as usize] as usize];

    /* Find pairs that are sufficiently steep */
    let mut first_steep = num_pts - 1;
    let mut num_steep = 1;
    for ind in (0..=num_pts - 2).rev() {
        let angle = steep_angle[sort_ind[ind as usize] as usize];

        /* Termination conditions: */
        if (angle < s_steepest_ratio * very_steepest && num_steep >= s_num_min_for_amax)
            || (angle < s_angle_max && num_steep >= s_num_min_for_arelax)
            || (angle < s_angle_relax)
        {
            break;
        }

        /* Check for duplicate pair */
        let mut ifdup = 0;
        for i in ind + 1..num_pts {
            // `steepAngle[sortInd[i]] > angle + 1.e-5` -- the sum is a double.
            if (steep_angle[sort_ind[i as usize] as usize] as f64) > angle as f64 + 1.0e-5 {
                break;
            }
            if sort_ind[i as usize] == steep_neigh[sort_ind[ind as usize] as usize]
                && sort_ind[ind as usize] == steep_neigh[sort_ind[i as usize] as usize]
            {
                ifdup = 1;
                if s_debug_level > 1 {
                    let _ = ImodFile::Stdout.write_all(
                        c_format(
                            "Duplicate  %d  %d  %f\n",
                            &[
                                CArg::Int(sort_ind[ind as usize] as i64),
                                CArg::Int(sort_ind[i as usize] as i64),
                                CArg::Dbl(steep_angle[sort_ind[ind as usize] as usize] as f64),
                            ],
                        )
                        .as_bytes(),
                    );
                }
                steep_angle[sort_ind[ind as usize] as usize] = -1.;
                break;
            }
        }
        if ifdup != 0 {
            continue;
        }

        /* Add this point as start of steep ones to use */
        num_steep += 1;
        first_steep = ind;
    }

    /* Collect the delta Z values to get median and outlier evaluation */
    // `B3DMALLOC(..., numPts - firstSteep)` (`surfacesort.c:351-355`).  That
    // size is too small: with a single steep pair `firstSteep == numPts - 1`,
    // so the allocation holds one element and `cluster[1] = jpt`
    // (`surfacesort.c:388`) writes past it.  The source overruns the heap
    // there; these are sized `numPts` instead, which is an upper bound on
    // every index used and reproduces what the C's malloc slack gives it.
    let alloc = if num_pts > num_pts - first_steep {
        num_pts
    } else {
        num_pts - first_steep
    } as usize;
    let mut del_z = vec![0.0f32; alloc];
    let mut outlie = vec![0.0f32; alloc];
    let mut cluster = vec![0i32; alloc];
    let mut cluster_sx = vec![0i32; alloc];
    let mut cluster_sy = vec![0i32; alloc];
    let mut median_del_z = 0.0f32;
    let mut i = 0usize;
    for ind in first_steep..num_pts {
        if steep_angle[sort_ind[ind as usize] as usize] > 0. {
            del_z[i] = ((zrot[sort_ind[ind as usize] as usize]
                - zrot[steep_neigh[sort_ind[ind as usize] as usize] as usize])
                as f64)
                .abs() as f32;
            i += 1;
        }
    }
    rs_median(&del_z, num_steep, &mut outlie, &mut median_del_z);
    if s_debug_level != 0 {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Steep pairs: n = %d  median delz = %f\n",
                &[CArg::Int(i as i64), CArg::Dbl(median_del_z as f64)],
            )
            .as_bytes(),
        );
    }
    if num_steep > 2 {
        /* If more than 2 points, identify outliers with criterion, and then
        eliminate them too */
        rs_mad_median_outliers(&del_z, num_steep, s_outlier_crit, &mut outlie);
        let mut i = 0usize;
        for ind in first_steep..num_pts {
            if steep_angle[sort_ind[ind as usize] as usize] > 0. {
                if outlie[i] != 0. {
                    steep_angle[sort_ind[ind as usize] as usize] = -1.;
                    num_steep -= 1;
                }
                i += 1;
            }
        }
    }

    /* Build clusters from the top of the list down */
    let mut num_cluster = 0i32;
    let mut num_done = 0i32;
    loop {
        /* Look for first remaining pair */
        let mut ind = num_pts - 1;
        while ind >= first_steep {
            if steep_angle[sort_ind[ind as usize] as usize] > 0. {
                break;
            }
            ind -= 1;
        }
        if ind < first_steep {
            break;
        }

        /* Start a cluster and list of points to check */
        let mut num_in_clust = 2usize;
        let ipt = sort_ind[ind as usize];
        let jpt = steep_neigh[ipt as usize];
        cluster[0] = ipt;
        cluster[1] = jpt;
        group[ipt as usize] = if zrot[ipt as usize] < zrot[jpt as usize] {
            1
        } else {
            2
        };
        group[jpt as usize] = 3 - group[ipt as usize];
        let mut check_ind = 0usize;
        steep_angle[ipt as usize] = -1.;
        while check_ind < num_in_clust {
            /* To check a point in cluster list, loop on the rest of the best pairs looking
            for one that includes this point */
            for jnd in (first_steep..=ind - 1).rev() {
                let ipt = sort_ind[jnd as usize];
                let jpt = steep_neigh[ipt as usize];
                if steep_angle[ipt as usize] > 0.
                    && (ipt == cluster[check_ind] || jpt == cluster[check_ind])
                {
                    /* oldpt is index of one already in cluster, newpt is its pair */
                    let mut newpt = ipt;
                    let mut oldpt = jpt;
                    if ipt == cluster[check_ind] {
                        oldpt = ipt;
                        newpt = jpt;
                    }

                    /* Check consistency with existing entry */
                    let knd = if zrot[oldpt as usize] < zrot[newpt as usize] {
                        1
                    } else {
                        2
                    };
                    if knd != group[oldpt as usize] {
                        let _ = ImodFile::Stdout.write_all(
                            c_format(
                                "INCONSISTENCY IN INITIAL STEEP PAIRS IN SURFACE SORT.\n",
                                &[],
                            )
                            .as_bytes(),
                        );
                    } else {
                        /* See if other point is already in cluster; if so check its
                        surface consistency too */
                        let mut ifdup = 0;
                        let knd = if zrot[oldpt as usize] >= zrot[newpt as usize] {
                            1
                        } else {
                            2
                        };
                        for j in 0..num_in_clust {
                            if newpt == cluster[j] {
                                ifdup = 1;
                                if knd != group[newpt as usize] {
                                    let _ = ImodFile::Stdout.write_all(
                                        c_format(
                                            "INCONSISTENCY IN INITIAL STEEP PAIRS IN SURFACE \
                                             SORT.\n",
                                            &[],
                                        )
                                        .as_bytes(),
                                    );
                                }
                                break;
                            }
                        }

                        /* Add new point to cluster and set its angle -1 to mark its pair as used */
                        if ifdup == 0 {
                            cluster[num_in_clust] = newpt;
                            num_in_clust += 1;
                            group[newpt as usize] = knd;
                        }
                    }
                    steep_angle[ipt as usize] = -1.;
                }
            }
            check_ind += 1;
        }

        /* A cluster is done.  Get its delta Z overall and central square */
        let mut nbot = 0i32;
        let mut ntop = 0i32;
        let mut zbot = 0.0f32;
        let mut ztop = 0.0f32;
        let mut xsum = 0.0f32;
        let mut ysum = 0.0f32;
        for i in 0..num_in_clust {
            let ipt = cluster[i];
            if group[ipt as usize] == 2 {
                ntop += 1;
                ztop += zrot[ipt as usize];
            } else {
                nbot += 1;
                zbot += zrot[ipt as usize];
            }
            xsum += xrot[ipt as usize];
            ysum += yrot[ipt as usize];
        }

        del_z[num_cluster as usize] = ztop / ntop as f32 - zbot / nbot as f32;
        cluster_sx[num_cluster as usize] =
            ((xsum / num_in_clust as f32 - xmin) / s_grid_spacing) as i32;
        cluster_sy[num_cluster as usize] =
            ((ysum / num_in_clust as f32 - ymin) / s_grid_spacing) as i32;
        num_cluster += 1;
        num_done += num_in_clust as i32;
        if s_debug_level != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "cluster %d  num  %d  delz  %f  sx, sy %d %d, done %d\n",
                    &[
                        CArg::Int(num_cluster as i64),
                        CArg::Int(num_in_clust as i64),
                        CArg::Dbl(del_z[(num_cluster - 1) as usize] as f64),
                        CArg::Int(cluster_sx[(num_cluster - 1) as usize] as i64),
                        CArg::Int(cluster_sy[(num_cluster - 1) as usize] as i64),
                        CArg::Int(num_done as i64),
                    ],
                )
                .as_bytes(),
            );
        }
        if s_debug_level > 1 {
            for i in 0..num_in_clust {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%d  %.1f  %.1f  %.1f  %d\n",
                        &[
                            CArg::Int(cluster[i] as i64),
                            CArg::Dbl(xrot[cluster[i] as usize] as f64),
                            CArg::Dbl(yrot[cluster[i] as usize] as f64),
                            CArg::Dbl(zrot[cluster[i] as usize] as f64),
                            CArg::Int(group[cluster[i] as usize] as i64),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
    }

    let mut num_err = 0i32;
    let mut errmax = -1000.0f32;
    let mut errsum = 0.0f32;
    let mut errsq = 0.0f32;

    /* Loop on rings, in each ring loop on clusters */
    for round in 0..2 {
        let mut ring = 0;
        while ring < num_rings && num_done < num_pts {
            let mut ind = 0;
            while ind < num_cluster && num_done < num_pts {
                let sx = cluster_sx[ind as usize];
                let sy = cluster_sy[ind as usize];

                /* Loop on the squares in the ring and on the points in the squares */
                let mut jdxy = ring_start[ring as usize];
                while jdxy < ring_start[(ring + 1) as usize] && num_done < num_pts {
                    let ix = sx + idx[jdxy as usize] as i32;
                    let iy = sy + idy[jdxy as usize] as i32;
                    if ix < 0 || ix >= num_grid_x || iy < 0 || iy >= num_grid_y {
                        jdxy += 1;
                        continue;
                    }
                    let jnd = ix + iy * num_grid_x;

                    /* If square done, skip; otherwise set to 1 and reset to 0 when
                    find a point that needs doing */
                    if square_done[jnd as usize] != 0 {
                        jdxy += 1;
                        continue;
                    }
                    square_done[jnd as usize] = 1;
                    let mut jsq = 0;
                    while jsq < num_in_square[jnd as usize] && num_done < num_pts {
                        let jpt = point_lists[(square_ind[jnd as usize] + jsq) as usize];
                        if group[jpt as usize] == 0 {
                            square_done[jnd as usize] = 0;

                            /* Now do rings around this point to collect identified
                            neighbors within a certain range up to a certain count */
                            let mut max_rings =
                                ((s_max_fit_dist / s_grid_spacing) as f64 + 0.5).floor() as i32;
                            max_rings = if max_rings < num_rings {
                                max_rings
                            } else {
                                num_rings
                            };
                            let mut nfit = 0i32;
                            let mut grpsum = 0i32;
                            let tx = ((xrot[jpt as usize] - xmin) / s_grid_spacing) as i32;
                            let ty = ((yrot[jpt as usize] - ymin) / s_grid_spacing) as i32;
                            let mut jring = 0;
                            while (jring < max_rings || nfit < 2) && nfit < s_max_num_fit {
                                let mut kdxy = ring_start[jring as usize];
                                while kdxy < ring_start[(jring + 1) as usize]
                                    && nfit < s_max_num_fit
                                {
                                    let jx = tx + idx[kdxy as usize] as i32;
                                    let jy = ty + idy[kdxy as usize] as i32;
                                    if jx < 0 || jx >= num_grid_x || jy < 0 || jy >= num_grid_y {
                                        kdxy += 1;
                                        continue;
                                    }
                                    let knd = jx + jy * num_grid_x;
                                    let mut isq = 0;
                                    while isq < num_in_square[knd as usize] && nfit < s_max_num_fit
                                    {
                                        let kpt =
                                            point_lists[(square_ind[knd as usize] + isq) as usize];
                                        if group[kpt as usize] != 0 {
                                            xfit[nfit as usize] = xrot[kpt as usize];
                                            yfit[nfit as usize] = yrot[kpt as usize];
                                            zfit[nfit as usize] = zrot[kpt as usize];
                                            grpfit[nfit as usize] =
                                                (group[kpt as usize] - 1) as f32;
                                            nfit += 1;
                                            grpsum += group[kpt as usize] - 1;
                                        }
                                        isq += 1;
                                    }
                                    kdxy += 1;
                                }
                                jring += 1;
                            }
                            if s_debug_level > 1 {
                                let _ = ImodFile::Stdout.write_all(
                                    c_format(
                                        "For %d  %.1f %.1f  %.1f  nfit %d  ntop %d\n",
                                        &[
                                            CArg::Int(jpt as i64),
                                            CArg::Dbl(xrot[jpt as usize] as f64),
                                            CArg::Dbl(yrot[jpt as usize] as f64),
                                            CArg::Dbl(zrot[jpt as usize] as f64),
                                            CArg::Int(nfit as i64),
                                            CArg::Int(grpsum as i64),
                                        ],
                                    )
                                    .as_bytes(),
                                );
                            }

                            let mut zbot: f32;
                            let mut ztop: f32;
                            /* Fit a biplane if there are enough points and 2 surfaces */
                            if nfit >= s_biplane_min_fit && grpsum != 0 && grpsum != nfit {
                                let (mut a1, mut a2, mut dzfit, mut con) = (0.0f32, 0., 0., 0.);
                                ls_fit3(
                                    &xfit, &yfit, &grpfit, &zfit, nfit, &mut a1, &mut a2,
                                    &mut dzfit, &mut con,
                                );

                                /* Get bottom and top predicted values.  Store current DZ as
                                delz for this area if there are at least 2 on each surface*/
                                zbot = a1 * xrot[jpt as usize] + a2 * yrot[jpt as usize] + con;
                                ztop = zbot + dzfit;
                                if s_debug_level > 1 {
                                    let _ = ImodFile::Stdout.write_all(
                                        c_format(
                                            "fit3 %f  %f %f %f\n",
                                            &[
                                                CArg::Dbl(a1 as f64),
                                                CArg::Dbl(a2 as f64),
                                                CArg::Dbl(dzfit as f64),
                                                CArg::Dbl(con as f64),
                                            ],
                                        )
                                        .as_bytes(),
                                    );
                                }
                                if grpsum > 1 && nfit - grpsum > 1 {
                                    del_z[ind as usize] = dzfit;
                                }
                            } else {
                                /* Need to keep just one group for single plane fit - so find
                                out how many points this leaves */
                                let mut keep_group = 1;
                                if grpsum <= nfit / 2 {
                                    keep_group = 0;
                                }
                                if (keep_group != 0 && grpsum >= s_plane_min_fit)
                                    || (keep_group == 0 && nfit - grpsum >= s_plane_min_fit)
                                {
                                    /* If this leaves enough points for a fit, repack the array
                                    with that group */
                                    if grpsum != 0 && grpsum != nfit {
                                        let mut j = 0i32;
                                        for i in 0..nfit as usize {
                                            if grpfit[i] == keep_group as f32 {
                                                xfit[j as usize] = xfit[i];
                                                yfit[j as usize] = yfit[i];
                                                zfit[j as usize] = zfit[i];
                                                grpfit[j as usize] = grpfit[i];
                                                j += 1;
                                            }
                                        }
                                        nfit = j;
                                        grpsum = j * keep_group;
                                    }

                                    /* Do fit and get upper and lower Z using the delz for area*/
                                    let (mut a1, mut a2, mut con) = (0.0f32, 0., 0.);
                                    ls_fit2(
                                        &xfit,
                                        &yfit,
                                        &zfit,
                                        nfit,
                                        &mut a1,
                                        &mut a2,
                                        Some(&mut con),
                                    );
                                    zbot = a1 * xrot[jpt as usize] + a2 * yrot[jpt as usize] + con;
                                    ztop = zbot;
                                    if keep_group != 0 {
                                        zbot -= del_z[ind as usize];
                                    } else {
                                        ztop += del_z[ind as usize];
                                    }
                                    if s_debug_level > 1 {
                                        let _ = ImodFile::Stdout.write_all(
                                            c_format(
                                                "fit2  %f  %f  %f  zbot %.1f  ztop  %.1f\n",
                                                &[
                                                    CArg::Dbl(a1 as f64),
                                                    CArg::Dbl(a2 as f64),
                                                    CArg::Dbl(con as f64),
                                                    CArg::Dbl(zbot as f64),
                                                    CArg::Dbl(ztop as f64),
                                                ],
                                            )
                                            .as_bytes(),
                                        );
                                    }
                                } else {
                                    /* Otherwise get mean of bottom and top */
                                    let mut nbot = 0i32;
                                    let mut ntop = 0i32;
                                    zbot = 0.;
                                    ztop = 0.;
                                    for i in 0..nfit as usize {
                                        if grpfit[i] != 0. {
                                            ntop += 1;
                                            ztop += zfit[i];
                                        } else {
                                            nbot += 1;
                                            zbot += zfit[i];
                                        }
                                    }

                                    /* Use both means if they exist; otherwise use delta Z and
                                    one mean to get the two Z values */
                                    if nbot != 0 {
                                        zbot /= nbot as f32;
                                    }
                                    if ntop != 0 {
                                        ztop /= ntop as f32;
                                    }
                                    if nbot != 0 && ntop == 0 {
                                        ztop = zbot + del_z[ind as usize];
                                    }
                                    if nbot == 0 && ntop != 0 {
                                        zbot = ztop - del_z[ind as usize];
                                    }
                                    if s_debug_level > 1 {
                                        let _ = ImodFile::Stdout.write_all(
                                            c_format(
                                                "means  %d  %d  zbot %.1f  ztop  %.1f\n",
                                                &[
                                                    CArg::Int(nbot as i64),
                                                    CArg::Int(ntop as i64),
                                                    CArg::Dbl(zbot as f64),
                                                    CArg::Dbl(ztop as f64),
                                                ],
                                            )
                                            .as_bytes(),
                                        );
                                    }
                                }
                            }

                            /* At last, assign point to group based on which Z is closest */
                            let xsum: f32;
                            if ((zrot[jpt as usize] - zbot) as f64).abs()
                                < ((zrot[jpt as usize] - ztop) as f64).abs()
                            {
                                group[jpt as usize] = 1;
                                // `100.` is a double literal, so the whole
                                // expression is evaluated in double.
                                xsum = (100. * (zrot[jpt as usize] - zbot) as f64
                                    / (ztop - zbot) as f64)
                                    as f32;
                            } else {
                                group[jpt as usize] = 2;
                                xsum = (100. * (ztop - zrot[jpt as usize]) as f64
                                    / (ztop - zbot) as f64)
                                    as f32;
                            }
                            if round != 0 || xsum < s_max_round1_dist {
                                if s_debug_level > 1 {
                                    let _ = ImodFile::Stdout.write_all(
                                        c_format(
                                            "Assign to %d   distance %.1f%%\n",
                                            &[
                                                CArg::Int(group[jpt as usize] as i64),
                                                CArg::Dbl(xsum as f64),
                                            ],
                                        )
                                        .as_bytes(),
                                    );
                                }
                                num_done += 1;
                                errsum += xsum;
                                errmax = if errmax > xsum { errmax } else { xsum };
                                errsq += xsum * xsum;
                                num_err += 1;
                            } else {
                                if s_debug_level > 1 {
                                    let _ = ImodFile::Stdout.write_all(
                                        c_format(
                                            "Defer %d because distance is %.1f%%\n",
                                            &[CArg::Int(jpt as i64), CArg::Dbl(xsum as f64)],
                                        )
                                        .as_bytes(),
                                    );
                                }
                                group[jpt as usize] = 0;
                            }
                        }
                        jsq += 1;
                    }
                    jdxy += 1;
                }
                ind += 1;
            }
            ring += 1;
        }
    }
    let (mut xsum, mut ysum) = (0.0f32, 0.0f32);
    sums_to_avg_sd(errsum, errsq, num_err, &mut xsum, &mut ysum);
    if s_debug_level != 0 {
        let mut out = ImodFile::Stdout;
        let _ = out.write_all(
            c_format(
                "Distance from nearest plane for %d points: mean %.1f%%  SD %.1f%%  max %.1f%%\n",
                &[
                    CArg::Int(num_err as i64),
                    CArg::Dbl(xsum as f64),
                    CArg::Dbl(ysum as f64),
                    CArg::Dbl(errmax as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = out.flush();
    }
    let _ = ImodFile::Stdout.flush();
    0
}

/// Original `surfacesort` (`surfacesort.c:706`).
pub fn surfacesort(xyz: &[f32], num_pts: &i32, markers_in_group: &i32, group: &mut [i32]) -> i32 {
    surface_sort(xyz, *num_pts, *markers_in_group, group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_dispatch_and_small_point_source_path() {
        assert_eq!(set_surf_sort_param(0, 25.), 0);
        assert_eq!(set_surf_sort_param(14, 0.), 1);
        let xyz = [0_f32; 6];
        let mut group = [0_i32; 2];
        assert_eq!(surface_sort(&xyz, 2, 0, &mut group), 0);
        assert_eq!(group, [1, 2]);
        set_surf_sort_param(0, 50.);
    }

    #[test]
    fn fortran_parameter_wrapper_preserves_rounding() {
        let index = 1;
        let value = 7.6;
        assert_eq!(setsurfsortparam(&index, &value), 0);
        assert_eq!(S_MAX_ANGLE_NEIGH.with(|c| c.get()), 8);
        let index = 1;
        let value = 50.;
        setsurfsortparam(&index, &value);
    }

    #[test]
    fn rejects_invalid_owned_input_without_indexing() {
        let mut empty = [];
        assert_eq!(surface_sort(&[], 0, 0, &mut empty), 0);
        assert_eq!(surface_sort(&[], 1, 0, &mut empty), 1);
        assert_eq!(surface_sort(&[0.0; 3], -1, 0, &mut [0]), 1);
        assert_eq!(SURFACE_SORT_SOURCE_FUNCTIONS.len(), 4);
    }
}
