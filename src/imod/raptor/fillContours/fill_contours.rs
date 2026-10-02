//! Translation of `IMOD/raptor/fillContours/fillContours.h` and
//! `fillContours.cpp`: fills gaps in the trajectories using the reprojection
//! model, and merges trajectories that follow the same marker.  `print` is
//! not reached (`DEAD_CODE.md`).

use crate::imod::c_sort::std_sort;
use crate::imod::cxx_stream::{cout, ostream_double};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::raptor::main_classes::constants::PEAK_THRESHOLD_FILL_CONTOURS;
use crate::imod::raptor::main_classes::io_mrc_vol::IoMrc;
use crate::imod::raptor::opencv::cvtemplmatch::{CV_TM_CCOEFF_NORMED, cv_match_template};
use crate::imod::raptor::opencv::cxtypes::{CV_64FC1, CvMat, CvSize, IPL_DEPTH_32F, IplImage};
use crate::imod::raptor::optimization::contour::Contour;
use crate::imod::raptor::optimization::sfm_data::SfmData;
use crate::imod::raptor::template::template::find_peaks;

/// `struct centroid` (`fillContours.h:8`).  The C++ array is `new
/// centroid[memory]`, uninitialised; every member read is written first.
#[derive(Clone, Copy, Debug, Default)]
pub struct Centroid {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

/// `fillContours(SFMdata* sfm, ioMRC* vol, float** templ, int* templSize,
/// bool debugMode)` (`fillContours.cpp:16`).
pub fn fill_contours(
    sfm: &mut SfmData,
    vol: &IoMrc,
    templ: &[Vec<f32>],
    templ_size: &[i32],
    debug_mode: bool,
) {
    cout("Starting filling contours\n");
    // maximum number of images use to use as a template matching
    let memory = 3i32;

    let m = sfm.contour_x.scores.rows;
    let t = sfm.contour_x.scores.cols;

    // we try to expand each trajectory: first left to right
    for ii in 0..m {
        let m_type = sfm.marker_type[ii as usize];
        let ts = templ_size[m_type as usize];
        // copy templ to IplImage for beadCenter method
        let mut perfect_marker_templ_ipl = IplImage::create(CvSize::new(ts, ts), IPL_DEPTH_32F, 1);
        perfect_marker_templ_ipl.image_data[..(ts * ts) as usize]
            .copy_from_slice(&templ[m_type as usize][..(ts * ts) as usize]);
        // we need at least "memory" previous neigbors to try to locate the
        // gold bead
        // `for (int jj=memory,pos=(ii*T)+memory;jj<T;jj++,pos++)`
        'frames: for jj in memory..t {
            let pos = (ii * t + jj) as usize;
            if sfm.contour_x.scores.data[pos] > 1e-6 {
                continue;
            }
            // we need at least two previous neigbors to try to locate the gold bead
            let mut try_tracking = true;
            for kk in 1..=memory {
                if sfm.contour_x.scores.data[pos - kk as usize] < 1e-6 {
                    try_tracking = false;
                    break;
                }
            }
            if !try_tracking {
                continue;
            }
            // now apparently we have enough history to try to continue the track

            // create patch around around this contour
            let mut patch_ipl = IplImage::create(CvSize::new(2 * ts, 2 * ts), IPL_DEPTH_32F, 1);
            try_tracking = vol.read_mrc_patch(
                jj,
                sfm.reproj_x.scores.data[pos].ceil() as i32,
                sfm.reproj_y.scores.data[pos].ceil() as i32,
                ts,
                ts,
                &mut patch_ipl.image_data,
            );
            if !try_tracking {
                continue;
            }

            // for each neighboring image find translation peak
            let mut peak = vec![Centroid::default(); memory as usize];
            let mut templ_ipl =
                IplImage::create(CvSize::new(2 * (ts / 2), 2 * (ts / 2)), IPL_DEPTH_32F, 1);
            for kk in 1..=memory {
                let n = pos - kk as usize;
                let frame = jj - kk;
                try_tracking = vol.read_mrc_patch(
                    frame,
                    sfm.contour_x.scores.data[n].ceil() as i32,
                    sfm.contour_y.scores.data[n].ceil() as i32,
                    ts / 2,
                    ts / 2,
                    &mut templ_ipl.image_data,
                );
                if !try_tracking {
                    // we can check variable used==false to see if we were here
                    peak[(kk - 1) as usize].x = -1.0;
                    continue;
                }
                let p = &mut peak[(kk - 1) as usize];
                find_matching_translation(
                    &patch_ipl,
                    &templ_ipl,
                    &mut p.x,
                    &mut p.y,
                    &mut p.score,
                    m_type,
                );
                if p.score < PEAK_THRESHOLD_FILL_CONTOURS {
                    p.x = -1.0;
                }
            }
            // see if they agree (`float += double` computes in double)
            let mut x_mean = 0.0f32;
            let mut y_mean = 0.0f32;
            let mut score_sum = 0.0f32;
            let mut count = 0i32;
            for kk in 0..memory as usize {
                if peak[kk].x >= 0.0 {
                    count += 1;
                    x_mean = (x_mean as f64 + peak[kk].x) as f32;
                    y_mean = (y_mean as f64 + peak[kk].y) as f32;
                    score_sum = (score_sum as f64 + peak[kk].score) as f32;
                }
            }
            if count == 0 {
                // non o fteh peaks could be calculated
                continue;
            }
            x_mean /= count as f32;
            y_mean /= count as f32;

            // double check if NCC agrees
            let mut xx = 0.0f32;
            let mut yy = 0.0f32;
            for kk in 0..memory as usize {
                if peak[kk].x >= 0.0 {
                    if (x_mean as f64 - peak[kk].x).abs() > 3.0 {
                        continue 'frames;
                    }
                    xx = (xx as f64 + peak[kk].score * peak[kk].x) as f32;
                    yy = (yy as f64 + peak[kk].score * peak[kk].y) as f32;
                }
            }
            xx /= score_sum; // weighted average
            yy /= score_sum;

            // peak search return (x,y) coordinates in array *result. We have to
            // obtain the relative coordinate of displacement (delta_x,delta_y)
            xx = (xx as f64 - 0.5 * (patch_ipl.width - templ_ipl.width) as f64) as f32;
            yy = (yy as f64 - 0.5 * (patch_ipl.height - templ_ipl.height) as f64) as f32;
            // improve centering of the bead
            bead_center(
                &patch_ipl,
                &perfect_marker_templ_ipl,
                &mut xx,
                &mut yy,
                m_type,
            );

            // delta_x and delta_y are switched
            sfm.contour_x.scores.data[pos] = yy as f64 + sfm.reproj_x.scores.data[pos].ceil();
            sfm.contour_y.scores.data[pos] = xx as f64 + sfm.reproj_y.scores.data[pos].ceil();

            if debug_mode {
                cout(&format!(
                    "Filled trajectory={ii} frame={jj} with scoreSum={} and new (x,y)={},{}.Reproj_(x,y)={},{}\n",
                    ostream_double(score_sum as f64),
                    ostream_double(sfm.contour_x.scores.data[pos]),
                    ostream_double(sfm.contour_y.scores.data[pos]),
                    ostream_double(sfm.reproj_x.scores.data[pos]),
                    ostream_double(sfm.reproj_y.scores.data[pos])
                ));
            }
        }
        if debug_mode {
            cout(&format!(
                "Finished checking trajectory={ii} for from left to right\n"
            ));
        }
    }

    // SECOND FROM RIGHT TO LEFT
    for ii in 0..m {
        let m_type = sfm.marker_type[ii as usize];
        let ts = templ_size[m_type as usize];
        let mut perfect_marker_templ_ipl = IplImage::create(CvSize::new(ts, ts), IPL_DEPTH_32F, 1);
        perfect_marker_templ_ipl.image_data[..(ts * ts) as usize]
            .copy_from_slice(&templ[m_type as usize][..(ts * ts) as usize]);
        // `for (int jj=T-memory-1,pos=((ii+1)*T)-memory-1;jj>=0;jj--,pos--)`
        'frames: for jj in (0..t - memory).rev() {
            let pos = (ii * t + jj) as usize;
            if sfm.contour_x.scores.data[pos] > 1e-6 {
                continue;
            }
            // we need at least two previous neigbors to try to locate the gold bead
            let mut try_tracking = true;
            for kk in 1..=memory {
                if sfm.contour_x.scores.data[pos + kk as usize] < 1e-6 {
                    try_tracking = false;
                    break;
                }
            }
            if !try_tracking {
                continue;
            }
            // now apparently we have enough history to try to continue the track

            // create patch around around this contour
            let mut patch_ipl = IplImage::create(CvSize::new(2 * ts, 2 * ts), IPL_DEPTH_32F, 1);
            try_tracking = vol.read_mrc_patch(
                jj,
                sfm.reproj_x.scores.data[pos].ceil() as i32,
                sfm.reproj_y.scores.data[pos].ceil() as i32,
                ts,
                ts,
                &mut patch_ipl.image_data,
            );
            if !try_tracking {
                continue;
            }

            // for each neighboring image find translation peak
            let mut peak = vec![Centroid::default(); memory as usize];
            let mut templ_ipl =
                IplImage::create(CvSize::new(2 * (ts / 2), 2 * (ts / 2)), IPL_DEPTH_32F, 1);
            for kk in 1..=memory {
                let n = pos + kk as usize;
                let frame = jj + kk;
                try_tracking = vol.read_mrc_patch(
                    frame,
                    sfm.contour_x.scores.data[n].ceil() as i32,
                    sfm.contour_y.scores.data[n].ceil() as i32,
                    ts / 2,
                    ts / 2,
                    &mut templ_ipl.image_data,
                );
                if !try_tracking {
                    // we can check variable used==false to see if we were here
                    peak[(kk - 1) as usize].x = -1.0;
                    continue;
                }
                let p = &mut peak[(kk - 1) as usize];
                find_matching_translation(
                    &patch_ipl,
                    &templ_ipl,
                    &mut p.x,
                    &mut p.y,
                    &mut p.score,
                    m_type,
                );
                if p.score < PEAK_THRESHOLD_FILL_CONTOURS {
                    p.x = -1.0;
                }
            }
            // see if they agree (`float += double` computes in double)
            let mut x_mean = 0.0f32;
            let mut y_mean = 0.0f32;
            let mut score_sum = 0.0f32;
            let mut count = 0i32;
            for kk in 0..memory as usize {
                if peak[kk].x >= 0.0 {
                    count += 1;
                    x_mean = (x_mean as f64 + peak[kk].x) as f32;
                    y_mean = (y_mean as f64 + peak[kk].y) as f32;
                    score_sum = (score_sum as f64 + peak[kk].score) as f32;
                }
            }
            if count == 0 {
                // non o fteh peaks could be calculated
                continue;
            }
            x_mean /= count as f32;
            y_mean /= count as f32;

            // double check if NCC agrees
            let mut xx = 0.0f32;
            let mut yy = 0.0f32;
            for kk in 0..memory as usize {
                if peak[kk].x >= 0.0 {
                    if (x_mean as f64 - peak[kk].x).abs() > 3.0 {
                        continue 'frames;
                    }
                    xx = (xx as f64 + peak[kk].score * peak[kk].x) as f32;
                    yy = (yy as f64 + peak[kk].score * peak[kk].y) as f32;
                }
            }
            xx /= score_sum; // weighted average
            yy /= score_sum;

            // peak search return (x,y) coordinates in array *result. We have to
            // obtain the relative coordinate of displacement (delta_x,delta_y)
            xx = (xx as f64 - 0.5 * (patch_ipl.width - templ_ipl.width) as f64) as f32;
            yy = (yy as f64 - 0.5 * (patch_ipl.height - templ_ipl.height) as f64) as f32;
            // improve centering of the bead
            bead_center(
                &patch_ipl,
                &perfect_marker_templ_ipl,
                &mut xx,
                &mut yy,
                m_type,
            );

            // delta_x and delta_y are switched
            sfm.contour_x.scores.data[pos] = yy as f64 + sfm.reproj_x.scores.data[pos].ceil();
            sfm.contour_y.scores.data[pos] = xx as f64 + sfm.reproj_y.scores.data[pos].ceil();

            if debug_mode {
                cout(&format!(
                    "Filled trajectory={ii} frame={jj} with scoreSum={} and new (x,y)={},{}.Reproj_(x,y)={},{}\n",
                    ostream_double(score_sum as f64),
                    ostream_double(sfm.contour_x.scores.data[pos]),
                    ostream_double(sfm.contour_y.scores.data[pos]),
                    ostream_double(sfm.reproj_x.scores.data[pos]),
                    ostream_double(sfm.reproj_y.scores.data[pos])
                ));
            }
        }
        if debug_mode {
            cout(&format!(
                "Finished checking trajectory={ii} for from right to left\n"
            ));
        }
    }

    cout("Finished filling contours\n");
}

/// `findMatchingTranslation(IplImage* imageIpl, IplImage* templIpl, double*
/// x, double* y, double* score, int mType)` (`fillContours.cpp:282`): given
/// a large patch and a smaller patch, finds the peak of the NCC value to
/// compute translation between patches.
pub fn find_matching_translation(
    image_ipl: &IplImage,
    templ_ipl: &IplImage,
    x: &mut f64,
    y: &mut f64,
    score: &mut f64,
    m_type: i32,
) {
    let lp_width = image_ipl.width;
    let lp_height = image_ipl.height;
    let sp_width = templ_ipl.width;
    let result_width = lp_width - sp_width + 1;
    let result_height = lp_height - sp_width + 1;

    let mut result_ipl =
        IplImage::create(CvSize::new(result_width, result_height), IPL_DEPTH_32F, 1);

    cv_match_template(image_ipl, templ_ipl, &mut result_ipl, CV_TM_CCOEFF_NORMED);

    let peaks = find_peaks(
        &result_ipl.image_data,
        result_width,
        result_height,
        PEAK_THRESHOLD_FILL_CONTOURS as f32,
        1,
        0,
        sp_width,
        0,
        m_type,
    );

    drop(result_ipl);

    if !peaks.is_empty() {
        *x = peaks[0].x as f64;
        *y = peaks[0].y as f64;
        *score = peaks[0].score;
    } else {
        *x = -1.0;
        *y = -1.0;
        *score = -1.0;
    }
}

/// `joinSimilarContours(SFMdata* sfm)` (`fillContours.cpp:348`): merges
/// trajectories closer than 3 pixels.  `sfm` is deleted inside.
pub fn join_similar_contours(mut sfm: SfmData) -> SfmData {
    let t = sfm.contour_x.scores.cols;
    let m = sfm.contour_x.scores.rows;
    let min_dist_contour = 3.0f64;

    // contours that need to be erased because we have merged them
    let mut erase_contour: Vec<i32> = Vec::new();
    for ii in 0..m - 1 {
        if erase_contour.contains(&ii) {
            // we don't need to check this contour because it was merged already
            continue;
        }
        let row1 = (ii * t) as usize;
        for jj in (ii + 1)..m {
            if erase_contour.contains(&jj) {
                continue;
            }
            let row2 = (jj * t) as usize;
            let dist = {
                let cx = &sfm.contour_x.scores.data;
                let cy = &sfm.contour_y.scores.data;
                distance_trajectory(
                    &cx[row1..row1 + t as usize],
                    &cy[row1..row1 + t as usize],
                    &cx[row2..row2 + t as usize],
                    &cy[row2..row2 + t as usize],
                    t,
                )
            };
            if dist < min_dist_contour {
                // contours need to be merged
                let cx = &mut sfm.contour_x.scores.data;
                let cy = &mut sfm.contour_y.scores.data;
                for kk in 0..t as usize {
                    if cx[row1 + kk] < 1e-6 {
                        if cx[row2 + kk] > 1e-6 {
                            cx[row1 + kk] = cx[row2 + kk];
                            cy[row1 + kk] = cy[row2 + kk];
                            cx[row2 + kk] = 0.0;
                            cy[row2 + kk] = 0.0;
                        }
                    } else if cx[row2 + kk] > 1e-6 {
                        cx[row2 + kk] = 0.0;
                        cy[row2 + kk] = 0.0;
                    }
                }
                erase_contour.push(jj);
            }
        }
    }

    // redo sfm structure to delete trajectories that were merged
    if !erase_contour.is_empty() {
        let keep = m - erase_contour.len() as i32;
        let mut c_x = CvMat::create(keep, t, CV_64FC1);
        let mut r_x = CvMat::create(keep, t, CV_64FC1);
        let mut c_y = CvMat::create(keep, t, CV_64FC1);
        let mut r_y = CvMat::create(keep, t, CV_64FC1);
        let mut m_type = vec![0i32; keep.max(0) as usize];

        let mut count = 0i32;
        for ii in 0..m {
            if erase_contour.contains(&ii) {
                // we don't need to copy this contour
                continue;
            }
            let mut pos = (count * t) as usize;
            let mut pos2 = (ii * t) as usize;
            for _jj in 0..t {
                c_x.data[pos] = sfm.contour_x.scores.data[pos2];
                r_x.data[pos] = sfm.reproj_x.scores.data[pos2];
                c_y.data[pos] = sfm.contour_y.scores.data[pos2];
                r_y.data[pos] = sfm.reproj_y.scores.data[pos2];
                pos += 1;
                pos2 += 1;
            }
            m_type[count as usize] = sfm.marker_type[ii as usize];
            count += 1;
        }

        if count != keep {
            cout(&format!(
                "ERROR: joinSimilarContours. Count={count} does not agree with number of contours to keep {keep}\n"
            ));
            exit(-1);
        }

        drop(sfm);
        SfmData::with_reprojection(
            &Contour::from_mat(&c_x, 1),
            &Contour::from_mat(&c_y, 2),
            &Contour::from_mat(&r_x, 1),
            &Contour::from_mat(&r_y, 2),
            &m_type,
        )
    } else {
        // `new SFMdata(*sfm); delete sfm;`
        sfm
    }
}

/// `distanceTrajectory(double* x1, double* y1, double* x2, double* y2, int
/// length)` (`fillContours.cpp:453`): the median L1 distance between two
/// trajectories over the frames both have.
pub fn distance_trajectory(x1: &[f64], y1: &[f64], x2: &[f64], y2: &[f64], length: i32) -> f64 {
    let mut dist: Vec<f64> = Vec::new();

    for kk in 0..length as usize {
        if x1[kk] > 1e-6 && x2[kk] > 1e-6 {
            dist.push((x1[kk] - x2[kk]).abs() + (y1[kk] - y2[kk]).abs());
        }
    }

    if dist.is_empty() {
        1e20
    } else {
        // compute median
        std_sort(&mut dist, &mut |a, b| a < b);
        dist[dist.len() / 2]
    }
}

/// `beadCenter(IplImage* patchIpl, IplImage* perfectMarkerTemplIpl, float*
/// xx, float* yy, int mType)` (`fillContours.cpp:472`).
///
/// `dist` is the running minimum over all the peaks, measured from the
/// centre as already moved, so once one peak lies within `minDist` every
/// later (weaker) peak replaces the centre too, wherever it lies.  The
/// comments describe a closest-peak rule, but that rule gave worse
/// alignments on TS_01, so the source's behaviour is kept (owner decision
/// 2026-09-27; `BUGS.md`, RAPTOR, item 17, with the measurements).
pub fn bead_center(
    patch_ipl: &IplImage,
    perfect_marker_templ_ipl: &IplImage,
    xx: &mut f32,
    yy: &mut f32,
    m_type: i32,
) {
    let result_width = patch_ipl.width - perfect_marker_templ_ipl.width + 1;
    let result_height = patch_ipl.height - perfect_marker_templ_ipl.height + 1;

    let mut result_ipl =
        IplImage::create(CvSize::new(result_width, result_height), IPL_DEPTH_32F, 1);
    cv_match_template(
        patch_ipl,
        perfect_marker_templ_ipl,
        &mut result_ipl,
        CV_TM_CCOEFF_NORMED,
    );

    // we find the closest peak to (xx,yy); highest peak is not always the
    // best option (for example, when two markers are together)
    let max_num_peaks = 5u32;
    let peaks = find_peaks(
        &result_ipl.image_data,
        result_width,
        result_height,
        PEAK_THRESHOLD_FILL_CONTOURS as f32,
        max_num_peaks,
        0,
        perfect_marker_templ_ipl.width,
        0,
        m_type,
    );

    // at the most we allow a move as largar as ther adius of a marker
    let min_dist = 0.25 * (perfect_marker_templ_ipl.width - 1) as f64;
    let mut dist = 1e11f64;
    // the center is 0.5*(result->width-1)
    let cx = 0.5f32 * (result_width - 1) as f32;
    let cy = 0.5f32 * (result_height - 1) as f32;
    for peak in &peaks {
        // `(double)(a)*(a) + (b)*(b)`: the cast covers only the first
        // factor, so the second product is `float`.
        let dx = *xx - (peak.x - cx);
        let dy = *yy - (peak.y - cy);
        let candidate = (dx as f64 * dx as f64 + (dy * dy) as f64).sqrt();
        dist = if candidate < dist { candidate } else { dist };
        if dist < min_dist {
            *xx = peak.x - cx;
            *yy = peak.y - cy;
        }
    }
}
