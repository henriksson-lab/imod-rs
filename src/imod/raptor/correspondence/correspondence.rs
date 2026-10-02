//! Translation of `IMOD/raptor/correspondence/correspondence.h` and
//! `correspondence.cpp`: pairwise correspondence of the markers of two
//! projections.
//!
//! `correspondenceRegions` writes the singleton potentials and marker
//! positions to `<output>/temp/<idx>FFcorrespondence.*`, runs
//! `MarkersCorrespond` on them and reads back the marginal beliefs it writes.
//! The source runs `"<binPATH>/MarkersCorrespond" <name> > <idx>_temp.txt
//! 2>&1` through `popen`; `MarkersCorrespond` is translated in this crate
//! (`markers_correspond_main_test.rs`), so it is called directly, in process,
//! with its standard output and error sent to the same `_temp.txt` file.
//! The file interface between the two programs is kept as the source has it,
//! so every temporary file is the one native RAPTOR writes.
//!
//! Frames are indices into the program's `vector<frame>` and points ids in
//! the point arena (see `point2d.rs`).  `prevPair` is only ever tested
//! against NULL, so it arrives as a flag.
//!
//! Two upstream defects are fixed (`BUGS.md`, RAPTOR): the `.cfg` file's
//! first line, the marker count `MarkersCorrespond` sizes its per-marker
//! distance and lock arrays by, was written before `M` was updated to the
//! number of reference peaks actually written (one more than `M` whenever the
//! previous pair supplied `M + 1`), so `MarkersCorrespond` read those arrays
//! one element past their end; and the beliefs line is copied into a buffer
//! without a terminating NUL before `strtok` scans it.

use crate::imod::cxx_stream::{IStream, cout, ostream_double, redirect_standard_streams};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::raptor::correspondence::markers_correspond_main_test;
use crate::imod::raptor::main_classes::constants::INITIAL_PAIRWISE_SCORE;
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::io_mrc_vol::IoMrc;
use crate::imod::raptor::main_classes::paircorrespondence::PairCorrespondence;
use crate::imod::raptor::main_classes::point2d::{Point2D, PointId};
use crate::imod::raptor::main_classes::stat::{average, max, mean_and_variance};
use std::io::Write as _;

/// Writes one of the temporary files, as the C++ `ofstream` does when it is
/// closed; a file that cannot be created is the source's "could not be
/// opened" error.
fn write_temp_file(filename: &str, text: &[u8]) {
    match std::fs::File::create(filename) {
        Ok(mut out) => {
            let _ = out.write_all(text);
        }
        Err(_) => {
            cout(&format!(
                "ERROR:file {filename} could not be opened at correspondenceRegions(...)\n"
            ));
            exit(-1);
        }
    }
}

/// `correspondenceRegions(frame* frame1, frame* frame2,
/// vector<pairCorrespondence>* correspondences, int Ntemplate, unsigned int
/// M, unsigned int K, int maxNCliques, int dmax, int Q1, int W1, int
/// maxPWTable, int minPWTable, string idxStr, int diameter, ioMRC* vol,
/// vector<pairCorrespondence>* prevPair, string outputDir, string binPATH)`
/// (`correspondence.cpp:25`).  `binPATH` located the `MarkersCorrespond`
/// executable; the translation calls its own.
#[allow(clippy::too_many_arguments)]
pub fn correspondence_regions(
    frames: &mut [Frame],
    frame1: usize,
    frame2: usize,
    points: &mut [Point2D],
    correspondences: &mut Vec<PairCorrespondence>,
    _n_template: i32,
    mut m: u32,
    mut k: u32,
    max_n_cliques: i32,
    dmax: i32,
    _q1: i32,
    _w1: i32,
    max_pw_table: i32,
    min_pw_table: i32,
    idx_str: &str,
    _diameter: i32,
    vol: &IoMrc,
    prev_pair: bool,
    output_dir: &str,
    _bin_path: &str,
) -> bool {
    let diff_threshold = 1.0f64;
    // minimum number of pairwise cliques per marker
    let min_cliques = 8u32;

    let frame1_size = if (frames[frame1].p.len() as u32) < m {
        frames[frame1].p.len() as u32
    } else {
        m
    };
    let frame2_size = if (frames[frame2].p.len() as u32) < k {
        frames[frame2].p.len() as u32
    } else {
        k
    };

    k = frame2_size; // to avoid segmentation fault

    // write files for Farshid's software
    let cfg_filename = format!("{output_dir}temp/{idx_str}FFcorrespondence.cfg");
    // The source writes `M` as the first line here, before `M` becomes the
    // number of reference peaks; the line is written with that final count
    // below (see the module documentation).
    let mut cfg_rest = String::new();
    cfg_rest.push_str(" 3\n");
    cfg_rest.push_str(" 8\n");
    cfg_rest.push_str(" 7\n");
    let mut n_cliques = max_n_cliques + 10;
    let mut reference_peaks: Vec<PointId> = Vec::new();
    if !prev_pair {
        for i in 0..frame1_size.min(m) {
            reference_peaks.push(frames[frame1].p[i as usize]);
        }
    } else {
        let mut j = 0u32;
        for i in 0..frames[frame1].p.len() {
            if points[frames[frame1].p[i]].used_pair_correspondence {
                reference_peaks.push(frames[frame1].p[i]);
                j += 1;
                if j > m {
                    break;
                }
            }
        }
        if j < m {
            for i in 0..frame1_size {
                if !points[frames[frame1].p[i as usize]].used_pair_correspondence {
                    reference_peaks.push(frames[frame1].p[i as usize]);
                    j += 1;
                }
                if j > m {
                    break;
                }
            }
        }
    }

    // update final number of targets (sometimes we might not have M)
    m = reference_peaks.len() as u32;

    let w1 = frames[frame1].width;
    let h1 = frames[frame1].height;
    let mut d = ((w1 * w1) as f64 + (h1 * h1) as f64).sqrt() + 50.0;
    while n_cliques > max_n_cliques {
        d -= 50.0;
        n_cliques = 0;
        // `for (unsigned int i = 0; i < M-1; i++)`: with no reference peak
        // `M-1` wraps and the source indexes an empty vector; no pair exists.
        for i in 0..m.saturating_sub(1) {
            let x1 = points[reference_peaks[i as usize]].x as f64;
            let y1 = points[reference_peaks[i as usize]].y as f64;
            for j in (i + 1)..m {
                let x2 = points[reference_peaks[j as usize]].x as f64;
                let y2 = points[reference_peaks[j as usize]].y as f64;
                let temp = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)).sqrt();
                if temp < d {
                    n_cliques += 1;
                }
            }
        }
    }
    cfg_rest.push_str(&format!(" {}\n", ostream_double(d)));
    cfg_rest.push_str(&format!(" {}\n", ostream_double(0.0001)));
    cfg_rest.push_str(&format!(" {}\n", 5));
    if m <= min_cliques {
        cfg_rest.push_str(&format!(" {}\n", m.wrapping_sub(1)));
    } else {
        cfg_rest.push_str(&format!(" {min_cliques}\n"));
    }
    cfg_rest.push_str(&format!(" {max_pw_table}\n"));
    cfg_rest.push_str(&format!(" {min_pw_table}\n"));
    for _ in 0..m {
        cfg_rest.push_str(&format!("{}\n", ostream_double(-(dmax as f32) as f64)));
        cfg_rest.push_str(&format!("{}\n", ostream_double(dmax as f32 as f64)));
    }
    write_temp_file(&cfg_filename, format!("{m}\n{cfg_rest}").as_bytes());

    let mut positions = format!("{m}\n");
    for i in 0..m {
        positions.push_str(&format!(
            "{}\n",
            ostream_double(points[reference_peaks[i as usize]].x as f64)
        ));
    }
    for i in 0..m {
        positions.push_str(&format!(
            "{}\n",
            ostream_double(points[reference_peaks[i as usize]].y as f64)
        ));
    }
    let mrkr_filename = format!("{output_dir}temp/{idx_str}FFcorrespondence.mrkr");
    write_temp_file(&mrkr_filename, positions.as_bytes());

    let mrkr_trans_filename = format!("{output_dir}temp/{idx_str}FFcorrespondence.mrkr_trans");
    write_temp_file(&mrkr_trans_filename, positions.as_bytes());

    let cand_filename = format!("{output_dir}temp/{idx_str}FFcorrespondence.cand");
    let mut cand = format!("{k}\n");
    for i in 0..k {
        cand.push_str(&format!(
            "{}\n",
            ostream_double(points[frames[frame2].p[i as usize]].x as f64)
        ));
    }
    for i in 0..k {
        cand.push_str(&format!(
            "{}\n",
            ostream_double(points[frames[frame2].p[i as usize]].y as f64)
        ));
    }
    write_temp_file(&cand_filename, cand.as_bytes());

    // `new float[M * K]` is uninitialised, and `singlePotentials` reads
    // entries it has not written (`S[i + j*S_width] != threshold`); zero is
    // a value that test treats as the source's garbage does.
    let mut s = vec![0f32; (m * k) as usize];
    single_potentials(
        &mut s,
        &frames[frame1],
        &frames[frame2],
        points,
        15,
        dmax,
        2,
        1,
        vol,
        m,
        k,
        &reference_peaks,
    );

    let sp_filename = format!("{output_dir}temp/{idx_str}FFcorrespondence.sp");
    let mut sp = format!("{k}\n");
    let mut pos_s = 0usize;
    for _i in 0..m {
        for _j in 0..k {
            sp.push_str(&ostream_double(s[pos_s] as f64));
            sp.push(' ');
            pos_s += 1;
        }
        sp.push('\n');
    }
    sp.push('\n');
    write_temp_file(&sp_filename, sp.as_bytes());

    // `popen("\"<binPATH>MarkersCorrespond\" <name> > <idx>_temp.txt 2>&1")`
    let test_name = format!("{output_dir}temp/{idx_str}FFcorrespondence");
    let temp_txt = format!("{output_dir}temp/{idx_str}_temp.txt");
    let argv = vec!["MarkersCorrespond".to_string(), test_name.clone()];
    let _ = crate::imod::commands::call_in_process(
        &["MarkersCorrespond", &test_name],
        None,
        false,
        move || {
            if let Ok(file) = std::fs::File::create(&temp_txt) {
                redirect_standard_streams(Some(file));
            }
            markers_correspond_main_test::main(&argv)
        },
    );

    let belief_filename = format!("{output_dir}temp/{idx_str}FFcorrespondence_final_beliefs_scr.m");
    let Some(mut input) = IStream::open(&belief_filename) else {
        cout(&format!(
            "ERROR:file {belief_filename} could not be opened at correspondenceRegions(...) to read inference results\n"
        ));
        exit(-1);
    };

    let mut c = vec![0f64; (m * k) as usize];
    let mut buffer = String::new();
    input.getline(&mut buffer);
    for i in 0..m {
        input.getline(&mut buffer);
        // `strtok(temp, " ")` over a copy of the line; the line is the whole
        // of the copy (see the module documentation).
        let mut tokens = buffer.split(' ').filter(|t| !t.is_empty());
        // we don't read the garbage potential at position K+1
        for j in 0..k {
            let token = tokens.next().unwrap_or("");
            let mut ss = IStream::from_bytes(token.as_bytes().to_vec());
            ss.read_f64(&mut c[(j + i * k) as usize]);
        }
    }
    drop(input);
    let mut max_c = vec![-1E+37f64; m as usize];
    let mut second_max = vec![-1E+37f64; m as usize];
    let mut max_index = vec![-1i32; m as usize];

    for i in 0..m as usize {
        for j in 0..k as usize {
            if c[j + i * k as usize] > max_c[i] {
                second_max[i] = max_c[i];
                max_c[i] = c[j + i * k as usize];
                max_index[i] = j as i32;
            }
        }
    }
    let mut num = 0i32;
    for i in 0..m as usize {
        if points[reference_peaks[i]].pairwise_score == INITIAL_PAIRWISE_SCORE {
            points[reference_peaks[i]].pairwise_score = max_c[i];
        }
        // log(0.1)=-2.3026
        if max_c[i] - second_max[i] > diff_threshold && max_c[i] > -2.3026 {
            let mut index = 0i32;
            for t in 0..frames[frame1].p.len() {
                if reference_peaks[i] == frames[frame1].p[t] {
                    index = t as i32;
                }
            }
            let temp = PairCorrespondence::new(
                index,
                max_index[i],
                frame1,
                frame2,
                c[i * k as usize + max_index[i] as usize],
            );
            points[reference_peaks[i]].index0.push(temp);
            points[reference_peaks[i]].used_pair_correspondence = true;
            let target = frames[frame2].p[max_index[i] as usize];
            points[target].pairwise_score = max_c[i];
            points[target].used_pair_correspondence = true;
            points[target].index1.push(temp);
            correspondences.push(temp);
            num += 1;
        }
    }

    let id1 = frames[frame1].frame_id;
    let id2 = frames[frame2].frame_id;
    if num < 4 && (id1 - id2).abs() < 2 {
        cout(&format!(
            "WARNING: pairwise correspondence between projections{id1}->{id2} (first projection is 0) contains only {num} correspondences\n"
        ));
        cout(
            "All the projections from now on are going to be ignored. Alignment will be produced with less projections.\n",
        );
        return false;
    } else {
        cout(&format!(
            "Found {num} pairwise correspondences out of {m} targeted markers between projections{id1}->{id2} (first projection is 0)\n"
        ));
    }
    true
}

/// `singlePotentials(float* S, frame* frame1, frame* frame2, int win, int
/// dmax, int mode, int option, ioMRC* vol, unsigned int
/// maxMarkersPrevFrame_, unsigned int maxMarkersNextFrame_,
/// vector<Point2D*> referencePeaks)` (`correspondence.cpp:342`).
#[allow(clippy::too_many_arguments)]
pub fn single_potentials(
    s: &mut [f32],
    frame1: &Frame,
    frame2: &Frame,
    points: &[Point2D],
    win: i32,
    dmax: i32,
    mode: i32,
    option: i32,
    vol: &IoMrc,
    max_markers_prev_frame_: u32,
    max_markers_next_frame_: u32,
    reference_peaks: &[PointId],
) {
    let threshold = 0.01f32;
    let npix = vol.header_get_nx() as usize * vol.header_get_ny() as usize;
    let mut image1 = vec![0f32; npix];
    vol.read_mrc_slice(frame1.frame_id, &mut image1);
    let mut image2 = vec![0f32; npix];
    vol.read_mrc_slice(frame2.frame_id, &mut image2);
    let mut aux_mv = 0f32;
    let mut var_frame1 = 0f32;
    mean_and_variance(
        &image1,
        frame1.width * frame1.height,
        &mut aux_mv,
        &mut var_frame1,
    );
    let mut var_frame2 = 0f32;
    mean_and_variance(
        &image2,
        frame2.width * frame2.height,
        &mut aux_mv,
        &mut var_frame2,
    );
    if max_markers_prev_frame_.wrapping_mul(max_markers_next_frame_) == 0 {
        return;
    }

    let s_width = max_markers_next_frame_ as i32;
    let s_height = max_markers_prev_frame_ as i32;

    let side = win * 2 + 1;
    let two_side = (side * side) as usize;
    // `new float[...]`: rows the source never fills are never read.
    let mut patches_frame1 = vec![0f32; max_markers_prev_frame_ as usize * two_side];
    let mut patches_frame2 = vec![0f32; max_markers_next_frame_ as usize * two_side];
    let mut vars_frame1 = vec![0f32; max_markers_prev_frame_ as usize];
    let mut vars_frame2 = vec![0f32; max_markers_next_frame_ as usize];
    let mut norms_frame1 = vec![0f32; max_markers_prev_frame_ as usize];
    let mut norms_frame2 = vec![0f32; max_markers_next_frame_ as usize];
    let mut mean_patch = 0f32;

    let mut aux = win;
    for j in 0..max_markers_prev_frame_ as usize {
        let rp = &points[reference_peaks[j]];
        while (rp.x - aux as f32) < 1.5
            || rp.x + aux as f32 > (frame1.width - 1) as f32
            || rp.y + aux as f32 > (frame1.height - 1) as f32
            || (rp.y - aux as f32) < 1.5
        {
            aux -= 1;
        }
        if aux != win {
            for i in 0..s_width {
                s[(i + j as i32 * s_width) as usize] = threshold;
            }
        } else {
            let img_x = (rp.x - win as f32) as i32;
            let img_y = (rp.y - win as f32) as i32;
            let img = &image1;
            for x in 0..side {
                for y in 0..side {
                    patches_frame1[j * two_side + (x + y * side) as usize] =
                        img[(img_x + x + (img_y + y) * frame1.width) as usize];
                }
            }
            mean_and_variance(
                &patches_frame1[j * two_side..],
                two_side as i32,
                &mut mean_patch,
                &mut vars_frame1[j],
            );
            for i in 0..two_side {
                patches_frame1[j * two_side + i] -= mean_patch;
            }
            norms_frame1[j] = (vars_frame1[j] * (two_side as i32 - 1) as f32).sqrt();
        }
        aux = win;
    }
    for i in 0..max_markers_next_frame_ as usize {
        let p2 = &points[frame2.p[i]];
        while (p2.x - aux as f32) < 1.5
            || p2.x + aux as f32 > (frame2.width - 1) as f32
            || (p2.y - aux as f32) < 1.5
            || p2.y + aux as f32 > (frame2.height - 1) as f32
        {
            aux -= 1;
        }
        if aux != win {
            for j in 0..s_height {
                s[(i as i32 + j * s_width) as usize] = threshold;
            }
        } else {
            let img_x = (p2.x - win as f32) as i32;
            let img_y = (p2.y - win as f32) as i32;
            let img = &image2;
            for x in 0..side {
                for y in 0..side {
                    patches_frame2[i * two_side + (x + y * side) as usize] =
                        img[(img_x + x + (img_y + y) * frame2.width) as usize];
                }
            }
            mean_and_variance(
                &patches_frame2[i * two_side..],
                two_side as i32,
                &mut mean_patch,
                &mut vars_frame2[i],
            );
            for j in 0..two_side {
                patches_frame2[i * two_side + j] -= mean_patch;
            }
            norms_frame2[i] = (vars_frame2[i] * (two_side as i32 - 1) as f32).sqrt();
        }
        aux = win;
    }
    for j in 0..s_height {
        if average(&s[(j * s_width) as usize..], s_width) != threshold as f64 {
            for i in 0..s_width {
                let idx = (i + j * s_width) as usize;
                if s[idx] != threshold {
                    let rp = &points[reference_peaks[j as usize]];
                    let p2 = &points[frame2.p[i as usize]];
                    let vx = rp.x - p2.x;
                    let vy = rp.y - p2.y;
                    let d: f64 = if option == 1 || option == 3 {
                        2.0 * dmax as f64
                    } else {
                        dmax as f64
                    };
                    let v_norm = ((vx * vx) + (vy * vy)).sqrt() as f64;
                    if v_norm > d {
                        // norm
                        s[idx] = 1E-10;
                    } else if rp.marker_type != p2.marker_type {
                        // both points have to come from the same marker
                        s[idx] = 1E-10;
                    } else if mode == 1 {
                        cout(
                            "Error: trying to calculate singleton potentials using Mutual Information. Not implemented yet.\n",
                        );
                        exit(-1);
                    } else if mode == 2 {
                        let mut m = 0.0f32;
                        for p1 in 0..two_side {
                            m += patches_frame1[j as usize * two_side + p1]
                                * patches_frame2[i as usize * two_side + p1];
                        }
                        m /= norms_frame1[j as usize] * norms_frame2[i as usize];
                        s[idx] = if m as f64 > 1E-10 { m } else { 1E-10 };
                    }
                }
            }
        }
    }
    // SECOND PASS TO ADD TEXTURE INFORMATION
    let a: f32;
    if max(s, s_width * s_height) > threshold {
        a = threshold;
    } else {
        let mut sum = 0f32;
        let mut num_bigger = 0f32;
        for i in 0..(s_width * s_height) as usize {
            if s[i] > threshold {
                sum += s[i];
                num_bigger += 1.0;
            }
        }
        a = sum / num_bigger;
    }
    for j in 0..max_markers_prev_frame_ {
        let row = (j * max_markers_next_frame_) as usize;
        if average(&s[row..], max_markers_next_frame_ as i32) != threshold as f64 {
            for i in 0..max_markers_next_frame_ {
                let idx = (i as i32 + j as i32 * s_width) as usize;
                if s[idx] != threshold {
                    let rp = &points[reference_peaks[j as usize]];
                    let p2 = &points[frame2.p[i as usize]];
                    let vx = rp.x - p2.x;
                    let vy = rp.y - p2.y;
                    let d: f64 = if option == 1 || option == 3 {
                        2.0 * dmax as f64
                    } else {
                        dmax as f64
                    };
                    let v_norm = (vx * vx + vy * vy).sqrt() as f64;
                    if v_norm > d {
                        // norm
                        s[idx] = 1E-10;
                    } else {
                        // texture information
                        let t = (0.5
                            * (vars_frame1[j as usize] / var_frame1
                                + vars_frame2[i as usize] / var_frame2)
                                as f64) as f32;
                        s[idx] += (1.0 - t) * a;
                        if option == 1 {
                            s[idx] *= (-(v_norm / dmax as f64).powf(2.0)).exp() as f32;
                        }
                    }
                }
            }
        }
    }
    let mut max_s = -1e+10f32;
    for i in 0..(max_markers_prev_frame_ * max_markers_next_frame_) as usize {
        if s[i] > max_s {
            max_s = s[i];
        }
    }
    for i in 0..(max_markers_prev_frame_ * max_markers_next_frame_) as usize {
        s[i] /= max_s;
    }
}
