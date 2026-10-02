//! Translation of `IMOD/raptor/trajectory/trajectory.h` and `trajectory.cpp`:
//! building marker trajectories from the pairwise correspondences, and
//! writing the IMOD text model, `tiltalign` parameter file and `tilt.com`.
//!
//! `trajectory::parent` and `trajectory::child` are set by the constructors
//! and read nowhere, so they are not kept; a trajectory is its list of
//! points (arena ids, see `point2d.rs`).  `writeMATLABcontour`,
//! `fiducialModel2Trajectory` and the `trajectory(trajectory*)` constructor
//! are not reached (`DEAD_CODE.md`).
//!
//! `ostream& out` parameters are byte buffers the caller writes to the file.

use crate::imod::cxx_stream::{IStream, cout, ostream_double};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::raptor::main_classes::constants::MAX_JUMPS;
use crate::imod::raptor::main_classes::frame::Frame;
use crate::imod::raptor::main_classes::paircorrespondence::PairCorrespondence;
use crate::imod::raptor::main_classes::point2d::{Point2D, PointId};
use std::io::Write as _;

/// `class trajectory`.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub p: Vec<PointId>,
}

/// `findTrajectories(vector<frame>* frames, vector<vector<pairCorrespondence>
/// >* correspondences, int numFrames, unsigned int maxMarkersPrevFrame)`
/// (`trajectory.cpp:46`).  The correspondences are read through the points'
/// `index0` lists, not through the parameter, which the source does not use.
pub fn find_trajectories(
    frames: &[Frame],
    _correspondences: &[Vec<PairCorrespondence>],
    num_frames: i32,
    max_markers_prev_frame: u32,
    points: &mut [Point2D],
) -> Vec<Trajectory> {
    let zerotilt = num_frames / 2 + num_frames % 2 - 1;
    let mut min_trajectory_length: u32 = if 10 > (num_frames as f64 * 0.1) as i32 {
        10
    } else {
        (num_frames as f64 * 0.1) as i32 as u32
    };
    if min_trajectory_length > num_frames as u32 {
        min_trajectory_length = num_frames as u32;
    }

    let mut trajectories: Vec<Trajectory> = Vec::new();
    let mut i = zerotilt as u32;
    while (i as usize) < frames.len() - 1 {
        if frames[i as usize].discard {
            i += 1;
            continue;
        }
        for j in 0..frames[i as usize].p.len() {
            if (j as u32) < max_markers_prev_frame || !points[frames[i as usize].p[j]].used {
                let temp = build_trajectory_forward(
                    frames[i as usize].p[j],
                    num_frames,
                    min_trajectory_length,
                    frames,
                    points,
                );
                // otherwise we are counting that min trajectory has to be
                // for half the trajectories
                if temp.p.len() as u32 > min_trajectory_length / 2 {
                    trajectories.push(temp);
                }
            }
        }
        i += 1;
    }
    build_trajectory_backward(
        &mut trajectories,
        frames,
        num_frames,
        zerotilt,
        max_markers_prev_frame,
        min_trajectory_length,
        points,
    );

    if trajectories.is_empty() {
        cout(&format!(
            "ERROR: after pairwise correspondence no trajectory with minimum length of {min_trajectory_length} projections could be found\n"
        ));
        cout("Thus, RAPTOR has to stop before optimization to find microscope alignment\n");
        exit(2);
    }
    trajectories
}

/// `buildTrajectoryForward(Point2D* point, int numFrames, unsigned int
/// minTrajectoryLength)` (`trajectory.cpp:82`).
pub fn build_trajectory_forward(
    point: PointId,
    num_frames: i32,
    min_trajectory_length: u32,
    frames: &[Frame],
    points: &mut [Point2D],
) -> Trajectory {
    let mut ans = Trajectory::default();
    ans.p.push(point);
    let mut cur = point;
    let mut prev_jump = 0u32;
    let mut next_jump = 1u32;
    loop {
        if !ans.p.is_empty() {
            cur = ans.p[ans.p.len() - 1 - prev_jump as usize];
        }
        let temp = find_marker_in_next_frame(cur, next_jump as i32, num_frames, frames, points);
        match temp {
            None => {
                next_jump += 1;
                if next_jump > MAX_JUMPS {
                    prev_jump += 1;
                    if prev_jump as usize >= ans.p.len() || prev_jump > MAX_JUMPS {
                        break;
                    }
                    next_jump = prev_jump + 1;
                }
            }
            Some(temp) => {
                prev_jump = 0;
                next_jump = 1;
                points[cur].used = true;
                cur = temp;
                ans.p.push(cur);
            }
        }
    }
    if (ans.p.len() as u32) < min_trajectory_length / 2 {
        for &p in &ans.p {
            points[p].used = false;
        }
    }
    ans
}

/// `buildTrajectoryBackward(vector<trajectory>* t, vector<frame>* frames, int
/// numFrames, int zerotilt, unsigned int maxMarkersPrevFrame, unsigned int
/// minTrajectoryLength)` (`trajectory.cpp:121`).
///
/// The second loop starts a trajectory at every usable point of the frames
/// below zero tilt and prepends the points it finds in earlier frames.  The
/// source never puts the starting point itself into `ans.p`
/// (`trajectory.cpp:167-170`), where `buildTrajectoryForward` starts with
/// `ans.p.push_back(point)`; fixed in translation (owner decision
/// 2026-09-27, `BUGS.md`, RAPTOR): the seed is the first element pushed, so
/// it stays the last one (the latest frame) as earlier frames are inserted
/// at the front.  With it:
/// - the trajectory keeps its seed, and the `minTrajectoryLength/2` test
///   counts it, as the forward builder's does;
/// - the `prevJump` walk mirrors the forward builder's.  Here index 0 is the
///   frontier (earliest frame) and `ans.p[prevJump]` is `prevJump` points
///   back towards the seed, as `ans.p[size()-1-prevJump]` is in the forward
///   builder.  After the first point is found (`ans.p = [A, seed]`) a failed
///   single jump from A now retries a double jump from the seed; the source,
///   with `ans.p = [A]`, broke off there because `prevJump >= size()`.
///   Further along the walk the source's `ans.p[1]` was already the right
///   point, so only that first retry changes;
/// - a rejected trajectory clears the seed's `used` flag with the others,
///   as the forward builder clears its seed's.  (The seed was marked `used`
///   only when a point was found from it.)  A seed among a frame's first
///   `maxMarkersPrevFrame` points is taken even when a forward trajectory
///   already holds it, exactly as the forward builder takes such seeds, so an
///   accepted backward trajectory can share its seed with another trajectory
///   -- the sharing `freeTrajectoryVector` already provides for.
pub fn build_trajectory_backward(
    t: &mut Vec<Trajectory>,
    frames: &[Frame],
    _num_frames: i32,
    zerotilt: i32,
    max_markers_prev_frame: u32,
    min_trajectory_length: u32,
    points: &mut [Point2D],
) {
    for i in 0..t.len() {
        let mut next_jump = 1u32;
        let mut prev_jump = 0u32;
        loop {
            let cur = t[i].p[prev_jump as usize];
            if points[cur].frame_id <= 0 {
                break;
            }
            let temp = find_marker_in_prev_frame(cur, next_jump as i32, frames, points);
            match temp {
                None => {
                    if prev_jump == MAX_JUMPS - 1 {
                        break;
                    }
                    next_jump += 1;
                    if next_jump == MAX_JUMPS {
                        prev_jump += 1;
                        if prev_jump as usize >= t[i].p.len() {
                            break;
                        }
                        next_jump = prev_jump + 1;
                    }
                }
                Some(temp) => {
                    next_jump = 1;
                    prev_jump = 0;
                    points[temp].used = true;
                    t[i].p.insert(0, temp);
                }
            }
        }
    }
    let mut i = zerotilt;
    while i > 0 {
        if frames[i as usize].discard {
            i -= 1;
            continue;
        }
        for j in 0..frames[i as usize].p.len() {
            if (j as u32) < max_markers_prev_frame || !points[frames[i as usize].p[j]].used {
                let mut ans = Trajectory::default();
                let mut cur = frames[i as usize].p[j];
                // The seed (not in the source's `ans.p`; see above).
                ans.p.push(cur);
                let mut prev_jump = 0u32;
                let mut next_jump = 1u32;
                loop {
                    let temp = find_marker_in_prev_frame(cur, next_jump as i32, frames, points);
                    match temp {
                        None => {
                            if prev_jump == MAX_JUMPS - 1 {
                                break;
                            }
                            next_jump += 1;
                            if next_jump == MAX_JUMPS {
                                prev_jump += 1;
                                if prev_jump as usize >= ans.p.len() {
                                    break;
                                }
                                cur = ans.p[prev_jump as usize];
                                next_jump = prev_jump + 1;
                            }
                        }
                        Some(temp) => {
                            next_jump = 1;
                            prev_jump = 0;
                            points[cur].used = true;
                            ans.p.insert(0, temp);
                            cur = temp;
                        }
                    }
                }
                if (ans.p.len() as u32) < min_trajectory_length / 2 {
                    for &p in &ans.p {
                        points[p].used = false;
                    }
                } else {
                    t.push(ans);
                }
            }
        }
        i -= 1;
    }
}

/// `findMarkerInPrevFrame(Point2D* cur, int jump)` (`trajectory.cpp:211`).
pub fn find_marker_in_prev_frame(
    cur: PointId,
    jump: i32,
    frames: &[Frame],
    points: &[Point2D],
) -> Option<PointId> {
    let mut ans: Option<PointId> = None;
    let mut max_score = -1E+12f64;
    let frame_id = points[cur].frame_id - jump;
    for temp in &points[cur].index0 {
        let target = frames[temp.frame2].p[temp.point2_index as usize];
        if temp.score > max_score
            && !points[target].used
            && frames[temp.frame2].frame_id == frame_id
        {
            max_score = temp.score;
            ans = Some(target);
        }
    }
    ans
}

/// `findMarkerInNextFrame(Point2D* cur, int jump, int numFrames)`
/// (`trajectory.cpp:228`).
pub fn find_marker_in_next_frame(
    cur: PointId,
    jump: i32,
    num_frames: i32,
    frames: &[Frame],
    points: &[Point2D],
) -> Option<PointId> {
    let mut ans: Option<PointId> = None;
    if points[cur].frame_id + jump >= num_frames {
        return ans;
    }
    let mut max_score = -1E+12f64;
    for temp in &points[cur].index0 {
        let target = frames[temp.frame2].p[temp.point2_index as usize];
        if temp.score > max_score
            && !points[target].used
            && frames[temp.frame2].frame_id == points[cur].frame_id + jump
        {
            max_score = temp.score;
            ans = Some(target);
        }
    }
    ans
}

/// `writeIMODfidModel(vector<trajectory> T, int width, int height, int
/// num_frames, string basename, ostream& out)` (`trajectory.cpp:245`).
pub fn write_imod_fid_model(
    t: &[Trajectory],
    width: i32,
    height: i32,
    num_frames: i32,
    _basename: &str,
    out: &mut Vec<u8>,
    points: &[Point2D],
) {
    let _ = write!(out, "imod 1\n");
    let _ = write!(out, "max {width} {height} {num_frames}\n");
    let _ = write!(out, "offsets 0 0 0\n");
    let _ = write!(out, "angles 0 0 0\n");
    let _ = write!(out, "scale 1 1 1\n");
    let _ = write!(out, "mousemode 1\n");
    let _ = write!(out, "drawmode 1\n");
    let _ = write!(out, "b&w_level 0,200\n");
    let _ = write!(out, "resolution 3\n");
    let _ = write!(out, "threshold 128\n");
    let _ = write!(out, "pixsize 1\n");
    let _ = write!(out, "units pixels\n");
    let _ = write!(out, "symbol circle\n"); // so imod displays all the markers at once
    let _ = write!(out, "size 7\n");

    let _ = write!(out, "object 0 {} 0\n", t.len());
    let _ = write!(out, "name\n");
    let _ = write!(out, "color 0 1 0 0\n");
    let _ = write!(out, "open\n");
    let _ = write!(out, "linewidth 1\n");
    let _ = write!(out, "surfsize  0\n");
    let _ = write!(out, "pointsize 0\n");
    let _ = write!(out, "axis      0\n");
    let _ = write!(out, "drawmode  1\n");

    for (i, traj) in t.iter().enumerate() {
        let _ = write!(out, "contour {} 0 {}\n", i, traj.p.len());
        for &p in &traj.p {
            let _ = write!(
                out,
                "{} {} {}\n",
                ostream_double((points[p].x + 1.0f32) as f64),
                ostream_double((points[p].y + 1.0f32) as f64),
                points[p].frame_id
            );
        }
    }
}

/// `writeIMODtiltalign(double alpha, string path, string basename, ostream&
/// out, vector<frame>* frames, int tiltOption, int rotOption, int
/// MagOption)` (`trajectory.cpp:340`).
#[allow(clippy::too_many_arguments)]
pub fn write_imod_tiltalign(
    alpha: f64,
    path: &str,
    basename: &str,
    out: &mut Vec<u8>,
    frames: &[Frame],
    tilt_option: i32,
    rot_option: i32,
    mag_option: i32,
) {
    let mut w = |s: &str| out.extend_from_slice(s.as_bytes());
    w("#If you know how to change the options for tiltalign in IMOD just edit this txt file\n");
    w(&format!(
        "#Execute from command line tiltalign -param {basename}_tiltalignScript.txt to align images with IMOD\n"
    ));
    w(&format!(
        "#Then execute from command line newstack -input ../prealign/{basename}.preali -output {basename}.ali -offset 0,0 -xform {basename}.xf to align images with IMOD\n\n\n"
    ));
    w(&format!("ModelFile\t{path}{basename}.fid.txt\n"));
    w(&format!("#ImageFile\t{basename}.preali\n"));
    w(&format!("OutputTiltFile\t{path}{basename}.tlt\n"));
    w(&format!("OutputTransformFile\t{path}{basename}.xf\n"));

    w("#THOSE ARE THE TWO MAIN PARAMETERS THAT NEED TO BE CHANGED!!!!!!!!!\n");
    w("#estimated from our correspondence\n");
    w(&format!("RotationAngle\t{}\n", ostream_double(alpha)));

    // list of views to be included in alignment
    w("IncludeList ");
    let mut flag_first = true;
    for (kk, frame) in frames.iter().enumerate() {
        if !frame.discard {
            if flag_first {
                // IMOD's tiltalign script considers 1 as the first section
                w(&format!("{}", kk + 1));
                flag_first = false;
            } else {
                w(&format!(",{}", kk + 1));
            }
        }
    }
    w("\n");

    w(&format!("TiltFile\t{path}{basename}.rawtlt\n\n"));

    w("#\n");
    w("# ADD a recommended tilt angle change to the existing AngleOffset value: our estimation\n");
    w("# should be pretty accurate\n");
    w("#\n");
    w("AngleOffset\t0.0\n");
    w(&format!("RotOption\t{rot_option}\n"));
    w("RotationFixedView 1\n");
    w("RotDefaultGrouping\t5\n\n");

    w("#\n");
    w("# TiltOption 0 fixes tilts, 2 solves for all tilt angles; change to 5 to solve\n");
    w("# for fewer tilts by grouping views by the amount in TiltDefaultGrouping\n");
    w("#\n");
    w(&format!("TiltOption\t{tilt_option}\n"));
    w("TiltFixedView 1\n");
    w("TiltDefaultGrouping\t5\n");
    w("MagReferenceView\t1\n");
    w(&format!("MagOption\t{mag_option}\n"));
    w("MagDefaultGrouping\t4\n\n");

    w("CompOption 0\n");
    w("#\n");
    w("# To solve for distortion, change both XStretchOption and SkewOption to 3;\n");
    w("# to solve for skew only leave XStretchOption at 0\n");
    w("#\n");
    w("XStretchOption\t0\n");
    w("XStretchDefaultGrouping\t7\n");
    w("SkewOption\t0\n");
    w("SkewDefaultGrouping\t11\n");
    w("# \n");
    w("# Criterion # of S.Dfprintf above mean residual to report (- for local mean)\n");
    w("#\n");
    w("ResidualReportCriterion\t2.0\n");
    w("SurfacesToAnalyze\t0\n");
    w("MetroFactor\t0.25\n");
    w("MaximumCycles\t5000\n");
    w("#\n");
    w("# ADD a recommended amount to shift up to the existing AxisZShift value\n");
    w("#\n");
    w("AxisZShift\t0\n");
    w("#\n");
    w("# Set to 1 to do local alignments\n");
    w("#\n");
    w("LocalAlignments\t0\n");
    w("OutputLocalFile\tDeinoG9blocal.xf\n");
    w("#\n");
    w("# Number of local patches to solve for in X and Y\n");
    w("#\n");
    w("NumberOfLocalPatchesXandY\t5,5\n");
    w("MinSizeOrOverlapXandY\t0.5,0.5\n");
    w("#\n");
    w("# Minimum fiducials total and on one surface if two surfaces\n");
    w("#\n");
    w("MinFidsTotalAndEachSurface\t8,3\n");
    w("FixXYZCoordinates\t0\n");
    w("LocalOutputOptions\t1,0,1\n");
    w("LocalRotOption\t3\n");
    w("LocalRotDefaultGrouping\t6\n");
    w("LocalTiltOption\t5\n");
    w("LocalTiltDefaultGrouping\t6\n");
    w("LocalMagReferenceView\t1\n");
    w("LocalMagOption\t3\n");
    w("LocalMagDefaultGrouping\t7\n");
    w("LocalXStretchOption\t0\n");
    w("LocalXStretchDefaultGrouping\t7\n");
    w("LocalSkewOption\t0\n");
    w("LocalSkewDefaultGrouping\t11\n");
}

/// `freeTrajectoryVector(vector<trajectory> T)` (`trajectory.cpp:450`).
/// The C++ first marks every point (`x = -10`) so a point two trajectories
/// share is deleted once, then deletes them; in the arena deleting is a
/// no-op, and the marking -- visible to every other holder of the point --
/// is what remains.
pub fn free_trajectory_vector(t: &[Trajectory], points: &mut [Point2D]) {
    // first pass is to identify repeated points
    for traj in t {
        for &p in &traj.p {
            // `fabs(x+10)` is the `float` overload
            if (points[p].x + 10.0f32).abs() as f64 >= 1e-3 {
                points[p].x = -10.0;
            }
        }
    }
}

/// `writeIMODtiltScript(int W, int H, string pathTilt, string pathAli,
/// string basename, int recoThickness, vector<frame>* frames, string
/// endingRec)` (`trajectory.cpp:528`): writes `<pathAli>tilt.com`.
#[allow(clippy::too_many_arguments)]
pub fn write_imod_tilt_script(
    w_: i32,
    h: i32,
    path_tilt: &str,
    path_ali: &str,
    basename: &str,
    reco_thickness: i32,
    frames: &[Frame],
    ending_rec: &str,
) {
    // create tilt.com script to run reconstruction
    let fid_name = format!("{path_ali}tilt.com");
    let mut fid: Vec<u8> = Vec::new();
    let mut w = |s: &str| fid.extend_from_slice(s.as_bytes());

    w("# Command file to run Tilt \n");
    w("#\n");
    w("####CreatedVersion#### 3.6.9\n");
    w("# \n");
    w("# RADIAL specifies the frequency at which the Gaussian low pass filter begins\n");
    w("#   followed by the standard deviation of the Gaussian roll-off\n");
    w("#\n");
    w("# LOG takes the logarithm of tilt data after adding the given value\n");
    w("#\n");

    w("$tilt\n");
    w(&format!("{path_ali}{basename}.ali\n"));
    w(&format!("{path_ali}{basename}{ending_rec}\n"));
    w("IMAGEBINNED 1\n");
    w(&format!("FULLIMAGE {w_} {h}\n"));
    w("LOG 0.0\n");
    w("MODE 2\n"); // always mode 2 and then we rescale if necessary using trimvol
    w("OFFSET 0.0\n");
    w("PARALLEL\n");
    w("RADIAL 0.35 0.05\n");
    w("SCALE 1.39 500\n");
    w("SHIFT 0.0 0.0\n");
    w("SUBSETSTART 0 0\n");
    w(&format!("THICKNESS {reco_thickness}\n"));
    let tlt_name = format!("{path_tilt}{basename}.tlt");
    let Some(mut in_tlt) = IStream::open(&tlt_name) else {
        // `ofstream fid` was created before this check, and `exit` does not
        // flush its buffer: the file is left empty.
        let _ = std::fs::write(&fid_name, b"");
        cout(&format!(
            "ERROR: opening tilt file {path_tilt}{basename}.tlt\n"
        ));
        cout("RAPTOR can not proceed with the reconstruction\n");
        exit(-1);
    };
    let mut tlt_angle = 0f32;
    w("ANGLES ");
    let mut flag_first = true;
    for frame in frames {
        in_tlt.read_f32(&mut tlt_angle);
        if !frame.discard {
            if flag_first {
                w(&ostream_double(tlt_angle as f64));
                flag_first = false;
            } else {
                w(&format!(",{}", ostream_double(tlt_angle as f64)));
            }
        }
    }
    w("\n");

    w("XAXISTILT 0.0\n");
    w("DONE\n");
    w("$if (-e ./savework) ./savework\n");

    // `ofstream fid` silently writes nothing when it could not be created.
    let _ = std::fs::write(&fid_name, &fid);
}

/// `getMarkerTypeFromTrajectory(vector<trajectory> T)`
/// (`trajectory.cpp:612`): the marker type of each trajectory, the rounded
/// average over its points.
pub fn get_marker_type_from_trajectory(t: &[Trajectory], points: &[Point2D]) -> Vec<i32> {
    let mut m_type = vec![0i32; t.len()];
    for (count, traj) in t.iter().enumerate() {
        m_type[count] = 0;
        for &p in &traj.p {
            m_type[count] += points[p].marker_type;
        }
        m_type[count] = (0.5 + (m_type[count] as f32 / traj.p.len() as f32) as f64).floor() as i32;
    }
    m_type
}
