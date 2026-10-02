//! Translation of `IMOD/raptor/template/template.h` and `template.cpp`:
//! marker template creation and refinement, and peak detection by normalised
//! cross-correlation.
//!
//! The file-scope globals `min_potential_peak` and `min_index` are
//! thread-local state here (a program run in process starts with its own),
//! and `findPeaks` returns its `new Point2D`s by value: the callers that keep
//! them move them into the point arena, the others drop them where the C++
//! deletes them.  `diffVector`, `nnz`, `mldivide` (empty) and the
//! commented-out `cropBorders` are not reached and are recorded in
//! `DEAD_CODE.md`.

use crate::imod::c_sort::std_sort;
use crate::imod::cxx_stream::cout;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::raptor::main::{DIAMETER, MAX_MARKERS_NEXT_FRAME, ZEROTILT};
use crate::imod::raptor::main_classes::constants::MAX_TARGETS_PREV_FRAME;
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::io_mrc_vol::IoMrc;
use crate::imod::raptor::main_classes::point2d::{Point2D, PointId};
use crate::imod::raptor::opencv::cvtemplmatch::{CV_TM_CCOEFF_NORMED, cv_match_template};
use crate::imod::raptor::opencv::cxmatrix::{CV_SVD, cv_solve};
use crate::imod::raptor::opencv::cxtypes::{CV_64FC1, CvMat, CvSize, IPL_DEPTH_32F, IplImage};
use std::cell::Cell;

thread_local! {
    /// `float min_potential_peak` (`template.cpp:15`).
    static MIN_POTENTIAL_PEAK: Cell<f32> = const { Cell::new(0.0) };
    /// `unsigned int min_index` (`template.cpp:16`).
    static MIN_INDEX: Cell<u32> = const { Cell::new(0) };
}

/// `struct simplePoint2D` (`template.h:13`).
#[derive(Clone, Copy, Debug)]
pub struct SimplePoint2D {
    pub x: i32,
    pub y: i32,
    pub score: f64,
}

impl SimplePoint2D {
    /// `simplePoint2D(int x, int y, double score)`.
    pub fn new(x: i32, y: i32, score: f64) -> SimplePoint2D {
        SimplePoint2D { x, y, score }
    }
}

/// `createSyntheticTemplate(vector<frame>* frames, int diameter, int
/// frameNumber, ioMRC* vol, bool white, int mType)` (`template.cpp:17`).
pub fn create_synthetic_template(
    frames: &[Frame],
    diameter: i32,
    frame_number: i32,
    vol: &IoMrc,
    white: bool,
    m_type: i32,
) -> Vec<f32> {
    // the slice to compute NCC with
    let mut slice = vec![0f32; vol.header_get_nx() as usize * vol.header_get_ny() as usize];
    vol.read_mrc_slice(frame_number, &mut slice);
    let n = 2 * diameter + 1; // width and height of the template
    let mut templ = vec![0f32; (n * n) as usize]; // the template for computing NCC
    let m = ((n + 1) / 2) as f64;
    let background: f32;
    let marker: f32;
    if white {
        background = 0.0;
        marker = 1.0;
    } else {
        background = 1.0;
        marker = 0.0;
    }
    for i in 0..n {
        for j in 0..n {
            let norm = ((i as f64 + 1.0 - m).abs().powf(2.0)
                + (j as f64 + 1.0 - m).abs().powf(2.0))
            .sqrt();
            if norm <= (diameter / 2) as f64 {
                templ[(i * n + j) as usize] = marker;
            } else {
                templ[(i * n + j) as usize] = background;
            }
        }
    }

    // `frames->at(frameNumber)`: `readMRCSlice` above has already rejected
    // an index outside the stack.
    let frame = &frames[frame_number as usize];
    let image_width = frame.width;
    let image_height = frame.height;
    let template_side = 2 * diameter + 1;
    let result_width = image_width - template_side + 1;
    let result_height = image_height - template_side + 1;
    let mut result = vec![0f32; (result_width * result_height) as usize];

    let mut image_ipl = IplImage::create(CvSize::new(image_width, image_height), IPL_DEPTH_32F, 1);
    let mut templ_ipl =
        IplImage::create(CvSize::new(template_side, template_side), IPL_DEPTH_32F, 1);
    let mut result_ipl =
        IplImage::create(CvSize::new(result_width, result_height), IPL_DEPTH_32F, 1);
    vol.read_mrc_slice(frame_number, &mut image_ipl.image_data);
    let side2 = (template_side * template_side) as usize;
    templ_ipl.image_data[..side2].copy_from_slice(&templ[..side2]);

    cv_match_template(&image_ipl, &templ_ipl, &mut result_ipl, CV_TM_CCOEFF_NORMED);

    let nres = (result_width * result_height) as usize;
    result.copy_from_slice(&result_ipl.image_data[..nres]);
    drop(image_ipl);
    drop(templ_ipl);
    drop(result_ipl);

    // suppress negative values and find the maximal value
    let mut max = -10.0f32;
    for i in 0..nres {
        if result[i] > max {
            max = result[i];
        }
    }
    // normalize cross correlation values to avoid threshold depending on
    // patch size
    for i in 0..nres {
        result[i] /= max;
    }

    // select the 5 highest peaks
    let peaks = find_peaks(
        &result,
        result_width,
        result_height,
        0.5,
        5,
        3,
        template_side,
        frame_number,
        m_type,
    );
    if peaks.len() < 3 {
        cout(
            "ERROR: unable to create marker template from synthetic template. Try to change teh specified diameter\n",
        );
        exit(-1);
    }

    // reset template to zero
    for i in 0..side2 {
        templ[i] = 0.0;
    }
    for k in 0..=2 {
        for i in 0..template_side {
            for j in 0..template_side {
                let x = (peaks[k].x + i as f32) as i32;
                let y = (peaks[k].y + j as f32) as i32;
                // The source tests only the upper bounds; a peak whose
                // sub-pixel centroid moved below 0 (the interpolation weights
                // are correlation values and can be negative) read before the
                // slice (`BUGS.md`, RAPTOR).
                if x >= 0 && y >= 0 && x < image_width && y < image_height {
                    templ[(i + j * template_side) as usize] +=
                        slice[(x + y * image_width) as usize];
                }
            }
        }
    }
    for i in 0..side2 {
        templ[i] /= 3.0; // perform average
    }
    templ
}

/// `computeNCC(vector<frame>* frames, float* templ, ioMRC* vol, int mType)`
/// (`template.cpp:113`): refines the template using the MRC volume.  The
/// starting point is a synthetic template.
pub fn compute_ncc(frames: &[Frame], templ: &mut [f32], vol: &IoMrc, m_type: i32) {
    let threshold = 0.3f32;
    let zerotilt = ZEROTILT.get();
    let mut l = zerotilt + 1 - (frames.len() % 2) as i32;
    let mut avg_factor = 3i32;
    // to refine the template we don't need all the projections
    for k in 0..(frames.len() as i32).min(20) {
        if (k % 2) == 1 {
            l -= k;
        } else {
            l += k;
        }
        let mut image = vec![0f32; vol.header_get_nx() as usize * vol.header_get_ny() as usize];
        vol.read_mrc_slice(l, &mut image);
        let diameter = DIAMETER.with(|d| d.borrow()[m_type as usize]);
        let template_side = diameter * 2 + 1;
        let image_width = frames[k as usize].width;
        let image_height = frames[k as usize].height;
        let result_width = frames[k as usize].width - template_side + 1;
        let result_height = frames[k as usize].height - template_side + 1;
        let nres = (result_width * result_height) as usize;
        let mut result = vec![0f32; nres];
        let mut image_ipl =
            IplImage::create(CvSize::new(image_width, image_height), IPL_DEPTH_32F, 1);
        let mut templ_ipl =
            IplImage::create(CvSize::new(template_side, template_side), IPL_DEPTH_32F, 1);
        let mut result_ipl =
            IplImage::create(CvSize::new(result_width, result_height), IPL_DEPTH_32F, 1);
        let nimg = (image_width * image_height) as usize;
        image_ipl.image_data[..nimg].copy_from_slice(&image[..nimg]);
        let side2 = (template_side * template_side) as usize;
        templ_ipl.image_data[..side2].copy_from_slice(&templ[..side2]);

        cv_match_template(&image_ipl, &templ_ipl, &mut result_ipl, CV_TM_CCOEFF_NORMED);

        result.copy_from_slice(&result_ipl.image_data[..nres]);
        drop(image_ipl);
        drop(templ_ipl);
        drop(result_ipl);
        // suppress negative values and find the maximal value
        let mut max = -10.0f32;
        for i in 0..nres {
            if result[i] > max {
                max = result[i];
            }
        }
        for i in 0..nres {
            result[i] /= max;
        }
        if k + 1 < vol.header_get_nz() {
            for i in 0..template_side {
                for j in 0..template_side {
                    templ[(i + j * template_side) as usize] *= (5 * k + 3) as f32;
                }
            }
            let mut peaks = find_peaks(
                &result,
                result_width,
                result_height,
                threshold,
                5,
                0,
                template_side,
                k,
                m_type,
            );
            for peak in peaks.iter_mut() {
                peak.x += diameter as f32;
                peak.y += diameter as f32;
                for i in 0..template_side {
                    for j in 0..template_side {
                        let x = (peak.x + i as f32 - diameter as f32) as i32;
                        let y = (peak.y + j as f32 - diameter as f32) as i32;
                        // The source adds `image[x + y*image_width]` unguarded;
                        // a peak near the edge (its centroid moved outward by
                        // the sub-pixel interpolation) puts the window partly
                        // outside the image and reads past the buffer.  Only
                        // pixels inside the image are added, the guard
                        // `createSyntheticTemplate` uses (`BUGS.md`, RAPTOR).
                        if x >= 0 && x < image_width && y >= 0 && y < image_height {
                            templ[(i + j * template_side) as usize] +=
                                image[(x + y * image_width) as usize];
                        }
                    }
                }
            }
            avg_factor += peaks.len() as i32;
            for i in 0..template_side {
                for j in 0..template_side {
                    templ[(i + j * template_side) as usize] /= avg_factor as f32;
                }
            }
        }
    }
}

/// `findPeaks(float* fv, int width, int height, float threshold, unsigned
/// int maxMarkers, unsigned int minMarkers, int template_side, int frameID,
/// int mType)` (`template.cpp:229`): finds the peaks in the given frame.
#[allow(clippy::too_many_arguments)]
pub fn find_peaks(
    fv: &[f32],
    width: i32,
    height: i32,
    mut threshold: f32,
    max_markers: u32,
    min_markers: u32,
    template_side: i32,
    frame_id: i32,
    m_type: i32,
) -> Vec<Point2D> {
    let mut peaks: Vec<Point2D> = Vec::new();

    // find the peaks
    MIN_POTENTIAL_PEAK.set(f32::MIN_POSITIVE);
    MIN_INDEX.set(u32::MAX);
    let mut potential_peaks: Vec<SimplePoint2D> = Vec::new();
    for i in 0..width {
        for j in 0..height {
            if fv[(i + j * width) as usize] > MIN_POTENTIAL_PEAK.get() {
                let p = SimplePoint2D::new(i, j, fv[(i + j * width) as usize] as f64);
                process_simple_point2d(&mut potential_peaks, p, max_markers, m_type);
            }
        }
    }
    // peak[i].score>=peak[i+1].score
    std_sort(&mut potential_peaks, &mut |a, b| simplepointcmp(*a, *b));
    let points = if (potential_peaks.len() as u32) < max_markers {
        potential_peaks.len() as u32
    } else {
        max_markers
    };
    for i in 0..points as usize {
        let p = potential_peaks[i];
        if p.score < threshold as f64 {
            break;
        }
        let mut point = Point2D::with_position(p.x, p.y, frame_id);
        point.score = p.score;
        peaks.push(point);
    }

    let mut interp = true;
    if (peaks.len() as u32) < min_markers && threshold > 0.2 {
        threshold -= 0.1;
        peaks.clear();
        peaks = find_peaks(
            fv,
            width,
            height,
            threshold,
            max_markers,
            min_markers,
            template_side,
            frame_id,
            m_type,
        ); // nested function
        interp = false; // if we enter here we don't need interpolation because of recursion
    }

    // interpolation to achieve subpixel accuracy
    let radius = 2i32;
    if interp {
        for peak in peaks.iter_mut() {
            let ii = peak.x as i32;
            let jj = peak.y as i32;
            if ii < radius || jj < radius || ii > width - radius - 1 || jj > height - radius - 1 {
                continue; // avoid segmentation fault
            }
            // we use center mass to find the center
            let mut w_total = 0.0f64;
            let mut posx = 0.0f64;
            let mut posy = 0.0f64;
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    let w = fv[((ii + dx) + (jj + dy) * width) as usize] as f64;
                    w_total += w;
                    posx += dx as f64 * w;
                    posy += dy as f64 * w;
                }
            }
            posx /= w_total;
            posy /= w_total;
            // update location
            peak.x = (peak.x as f64 - posx) as f32;
            peak.y = (peak.y as f64 - posy) as f32;
        }
    }

    peaks
}

/// `simplepointcmp(simplePoint2D a, simplePoint2D b)` (`template.cpp:407`).
pub fn simplepointcmp(a: SimplePoint2D, b: SimplePoint2D) -> bool {
    a.score > b.score
}

/// `pointcmp(Point2D* a, Point2D* b)` (`template.cpp:412`).
pub fn pointcmp(a: &Point2D, b: &Point2D) -> bool {
    a.score > b.score
}

/// `estimateNumberOfMarkers(vector<frame>* frames, float** templ, ioMRC* vol,
/// int numDiffMarkerSize)` (`template.cpp:422`).
pub fn estimate_number_of_markers(
    frames: &[Frame],
    templ: &[Vec<f32>],
    vol: &IoMrc,
    num_diff_marker_size: i32,
) -> i32 {
    let mut targets = -1i32;
    let thr_abs_dist = 0.05f64; // to decide where the kink is
    let min_ncc_val = 0.8f64; // any value above that is considered a marker

    let max_markers = 120i32; // we plot a profile with a hundred markers
    let zerotilt = (frames.len() / 2) as u32;
    let peak_threshold = 0.1f32;
    for tt in 0..num_diff_marker_size {
        let mut xx = CvMat::create(max_markers, 1, CV_64FC1);
        let mut xx_w = CvMat::create(max_markers, 1, CV_64FC1);
        for ii in 0..max_markers {
            xx.cvm_set(ii, 0, 0.0);
            xx_w.cvm_set(ii, 0, 0.0);
        }
        let diameter = DIAMETER.with(|d| d.borrow()[tt as usize]);
        let template_side = diameter * 2 + 1;
        // we use 9 frames to average the peak profile (the loop bound is
        // `unsigned`: with fewer than 8 frames `zerotilt-4` wraps and the
        // loop does not run)
        let mut i = zerotilt.wrapping_sub(4);
        while i < zerotilt.wrapping_add(4) {
            let frame = &frames[i as usize];
            let image_width = frame.width;
            let image_height = frame.height;
            let result_width = image_width - template_side + 1;
            let result_height = image_height - template_side + 1;
            let nres = (result_width * result_height) as usize;

            let mut result = vec![0f32; nres];
            let mut image_ipl =
                IplImage::create(CvSize::new(image_width, image_height), IPL_DEPTH_32F, 1);
            let mut templ_ipl =
                IplImage::create(CvSize::new(template_side, template_side), IPL_DEPTH_32F, 1);
            let mut result_ipl =
                IplImage::create(CvSize::new(result_width, result_height), IPL_DEPTH_32F, 1);
            vol.read_mrc_slice(i as i32, &mut image_ipl.image_data);
            let side2 = (template_side * template_side) as usize;
            templ_ipl.image_data[..side2].copy_from_slice(&templ[tt as usize][..side2]);

            cv_match_template(&image_ipl, &templ_ipl, &mut result_ipl, CV_TM_CCOEFF_NORMED);

            result.copy_from_slice(&result_ipl.image_data[..nres]);
            drop(image_ipl);
            drop(templ_ipl);
            drop(result_ipl);

            // suppress negative values and find the maximal value
            let mut max = -10.0f32;
            for kk in 0..nres {
                if result[kk] > max {
                    max = result[kk];
                }
            }
            // normalize cross correlation values to avoid threshold depending
            // on patch size
            for kk in 0..nres {
                result[kk] /= max;
            }

            let peaks = find_peaks(
                &result,
                result_width,
                result_height,
                peak_threshold,
                max_markers as u32,
                6,
                template_side,
                i as i32,
                tt,
            );

            for (kk, peak) in peaks.iter().enumerate() {
                xx_w.data[kk] += 1.0;
                xx.data[kk] += peak.score;
            }
            i += 1;
        }

        // estimate number of marker using least squares
        for kk in 0..max_markers as usize {
            if xx_w.data[kk] > 0.5 {
                xx.data[kk] /= xx_w.data[kk];
            }
        }

        // fit least squares to positions 9-19
        let mut a = CvMat::create(11, 2, CV_64FC1);
        let mut b = CvMat::create(11, 1, CV_64FC1);
        let mut pp = CvMat::create(2, 1, CV_64FC1);
        for kk in 0..11usize {
            a.data[kk + kk] = (kk + 9) as f64;
            a.data[kk + kk + 1] = 1.0;
            b.data[kk] = xx.data[kk + 9];
        }
        cv_solve(a.view(), b.view(), &mut pp, CV_SVD);
        let mut targets_aux = 0i32;
        for kk in 20..max_markers {
            targets_aux = kk;
            if (kk as f64 * pp.data[0] + pp.data[1] - xx.data[kk as usize]).abs() > thr_abs_dist
                || xx.data[kk as usize] < 0.6
            {
                if xx.data[kk as usize] < min_ncc_val {
                    break;
                }
            }
        }
        // we choose the max out of all the possible diameter sizes
        targets = targets.max(targets_aux);
    }

    targets.min(MAX_TARGETS_PREV_FRAME)
}

/// `findAllPeaks(vector<frame>* frames, float** templ, ioMRC* vol, int
/// numDiffMarkerSize, bool xRay)` (`template.cpp:531`).  The peaks a frame
/// keeps are moved into the point arena `points`.
pub fn find_all_peaks(
    frames: &mut [Frame],
    templ: &[Vec<f32>],
    vol: &IoMrc,
    num_diff_marker_size: i32,
    x_ray: bool,
    points: &mut Vec<Point2D>,
) {
    let max_markers_next_frame = MAX_MARKERS_NEXT_FRAME.get();
    for tt in 0..num_diff_marker_size {
        let diameter = DIAMETER.with(|d| d.borrow()[tt as usize]);
        let template_side = diameter * 2 + 1;
        let peak_threshold = 0.3f32;
        for i in 0..frames.len() {
            let image_width = frames[i].width;
            let image_height = frames[i].height;
            let result_width = image_width - template_side + 1;
            let result_height = image_height - template_side + 1;
            let nres = (result_width * result_height) as usize;

            let mut result = vec![0f32; nres];
            let mut image_ipl =
                IplImage::create(CvSize::new(image_width, image_height), IPL_DEPTH_32F, 1);
            let mut templ_ipl =
                IplImage::create(CvSize::new(template_side, template_side), IPL_DEPTH_32F, 1);
            let mut result_ipl =
                IplImage::create(CvSize::new(result_width, result_height), IPL_DEPTH_32F, 1);
            vol.read_mrc_slice(i as i32, &mut image_ipl.image_data);
            let side2 = (template_side * template_side) as usize;
            templ_ipl.image_data[..side2].copy_from_slice(&templ[tt as usize][..side2]);

            // remove lines from image to avoid missguiding template matching.
            // This is especially useful for soft X-ray where two parallel
            // straight lines from cylinder will ruin the template matching
            let mut mask = vec![1.0f32; (image_width * image_height) as usize];
            if x_ray {
                remove_lines_xray(&image_ipl, &mut mask);
            }

            cv_match_template(&image_ipl, &templ_ipl, &mut result_ipl, CV_TM_CCOEFF_NORMED);

            result.copy_from_slice(&result_ipl.image_data[..nres]);

            // apply mask
            // WARNING: IMAGE SIZE IS DIFFERENT THAN RESULT
            let mut pos_r = 0usize;
            let cc = (template_side - 1) / 2;
            for aa in 0..result_height {
                let mut pos_m = ((aa + cc) * image_width + cc) as usize;
                for _bb in 0..result_width {
                    result[pos_r] *= mask[pos_m];
                    pos_r += 1;
                    pos_m += 1;
                }
            }

            drop(mask);
            drop(image_ipl);
            drop(templ_ipl);
            drop(result_ipl);

            // suppress negative values and find the maximal value
            let mut max = -10.0f32;
            for kk in 0..nres {
                if result[kk] > max {
                    max = result[kk];
                }
            }
            // normalize cross correlation values to avoid threshold depending
            // on patch size
            for kk in 0..nres {
                result[kk] /= max;
            }

            let peaks = find_peaks(
                &result,
                result_width,
                result_height,
                peak_threshold,
                max_markers_next_frame,
                6,
                template_side,
                i as i32,
                tt,
            );

            for mut peak in peaks {
                peak.x += diameter as f32;
                peak.y += diameter as f32;
                peak.marker_type = tt;
                if tt != 0 && is_peak_in_frame(&frames[i], &peak, points) {
                    continue; // avoid repeating teh same peak
                }
                points.push(peak);
                frames[i].p.push(points.len() - 1);
            }
            // we need to sort them out based on score
            if tt != 0 {
                let arena: &[Point2D] = points;
                std_sort(&mut frames[i].p, &mut |a: &PointId, b: &PointId| {
                    pointcmp(&arena[*a], &arena[*b])
                });
            }
        }
    }
}

/// `isPeakInFrame(frame* frames, Point2D* p)` (`template.cpp:629`).
pub fn is_peak_in_frame(frames: &Frame, p: &Point2D, points: &[Point2D]) -> bool {
    let marker_type = p.marker_type;
    for &iter in &frames.p {
        if within_diameter(points[iter].x, points[iter].y, p.x, p.y, marker_type) {
            return true;
        }
    }
    false
}

/// `withinDiameter(float x1, float y1, float x2, float y2, int diamNum)`
/// (`template.cpp:642`).
pub fn within_diameter(x1: f32, y1: f32, x2: f32, y2: f32, diam_num: i32) -> bool {
    let d = DIAMETER.with(|d| d.borrow()[diam_num as usize]);
    let dist_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    (d.wrapping_mul(d)) as f32 > dist_sq
}

/// `processSimplePoint2D(vector<simplePoint2D>* potential_peaks,
/// simplePoint2D p, unsigned int maxMarkers, int mType)`
/// (`template.cpp:652`).
pub fn process_simple_point2d(
    potential_peaks: &mut Vec<SimplePoint2D>,
    p: SimplePoint2D,
    max_markers: u32,
    m_type: i32,
) -> bool {
    for i in 0..potential_peaks.len() {
        let target = potential_peaks[i];
        if within_diameter(
            p.x as f32,
            p.y as f32,
            target.x as f32,
            target.y as f32,
            m_type,
        ) {
            if p.score > target.score {
                potential_peaks[i].x = p.x;
                potential_peaks[i].y = p.y;
                potential_peaks[i].score = p.score;
                if i as u32 == MIN_INDEX.get() {
                    find_min(potential_peaks);
                }
                return true;
            } else {
                return false;
            }
        }
    }
    if potential_peaks.len() as u32 >= max_markers {
        potential_peaks[MIN_INDEX.get() as usize] = p;
    } else {
        potential_peaks.push(p);
    }
    find_min(potential_peaks);
    true
}

/// `findMin(vector<simplePoint2D>* potential_peaks)` (`template.cpp:684`).
pub fn find_min(potential_peaks: &[SimplePoint2D]) {
    MIN_INDEX.set(0);
    MIN_POTENTIAL_PEAK.set(potential_peaks[0].score as f32);
    for (i, p) in potential_peaks.iter().enumerate() {
        if p.score < MIN_POTENTIAL_PEAK.get() as f64 {
            MIN_POTENTIAL_PEAK.set(p.score as f32);
            MIN_INDEX.set(i as u32);
        }
    }
}

/// `removeLinesXray(IplImage* image, float* mask)` (`template.cpp:701`):
/// masks the bright borders of each row (the walls of a soft X-ray
/// capillary).  The source's two `while` scans stop only at a pixel not
/// above the row's central mean, so on a row brighter than its centre out to
/// the edge they run into the neighbouring rows and, on the first or last
/// row, off the image; they stop at the image bounds here (`BUGS.md`,
/// RAPTOR).
pub fn remove_lines_xray(image: &IplImage, mask: &mut [f32]) {
    let width = image.width;
    let safety_pixels = width / 100; // to remove further teh cylinder borders
    let total = (image.width * image.height) as i64;

    let mut mean: f32;
    // for each x-line find the two droping areas
    let line_profile = &image.image_data;
    for y in 0..image.height {
        let mut pos = (y * width) as i64;

        // compute mean in the center
        mean = 0.0;
        let mut count = 0i32;
        let mut ii = pos + (width / 4) as i64;
        while ii < pos + (3 * width / 4) as i64 {
            mean += line_profile[ii as usize];
            count += 1;
            ii += 1;
        }
        mean /= count as f32;

        // find white areas in the extremes
        while pos < total && line_profile[pos as usize] > mean {
            mask[pos as usize] = 0.0;
            pos += 1;
        }
        let mut ii = pos;
        while ii < pos + safety_pixels as i64 && ii < total {
            mask[ii as usize] = 0.0;
            ii += 1;
        }
        // start at the other end
        pos = ((y + 1) * width - 1) as i64;
        while pos >= 0 && line_profile[pos as usize] > mean {
            mask[pos as usize] = 0.0;
            pos -= 1;
        }
        let mut ii = pos;
        while ii > pos - safety_pixels as i64 && ii >= 0 {
            mask[ii as usize] = 0.0;
            ii -= 1;
        }
    }
}
