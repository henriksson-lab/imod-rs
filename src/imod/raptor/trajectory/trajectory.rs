//! Trajectory graph and model writers from RAPTOR.
use std::path::Path;
/// An owned point, replacing the source's independently allocated `Point2D *`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryPoint {
    pub x: f32,
    pub y: f32,
    pub frame_id: i32,
    pub marker_type: i32,
    pub used: bool,
}
/// C++ `trajectory`; `points` are owned point IDs, not aliases requiring `delete`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trajectory {
    pub points: Vec<usize>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

/// C++ `trajectory::trajectory()`: an unlinked, empty trajectory.
///
/// The native class stored non-owning pointers for `parent`, `p`, and `child`.
/// This port represents those relationships by arena indexes, so the empty
/// constructor has no aliases to clean up.
pub fn trajectory() -> Trajectory {
    Trajectory::default()
}

/// C++ `trajectory::trajectory(trajectory *parent)`.
pub fn trajectory_with_parent(parent: usize) -> Trajectory {
    Trajectory {
        parent: Some(parent),
        ..Trajectory::default()
    }
}

/// C++ `trajectory::trajectory(const trajectory &)`.  Index relationships are
/// deliberately copied, as were the native non-owning pointer vectors.
pub fn trajectory_copy(source: &Trajectory) -> Trajectory {
    source.clone()
}

/// C++ `trajectory::~trajectory()`, which intentionally did not delete its
/// non-owning points or children.  Rust's owned vectors are released here.
pub fn free_trajectory(trajectory: Trajectory) {
    drop(trajectory);
}
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryCorrespondence {
    pub target: usize,
    pub score: f64,
}
pub fn find_marker(
    points: &[TrajectoryPoint],
    edges: &[Vec<TrajectoryCorrespondence>],
    current: usize,
    jump: i32,
    next: bool,
) -> Option<usize> {
    let frame = points[current].frame_id + if next { jump } else { -jump };
    edges[current]
        .iter()
        .filter(|e| !points[e.target].used && points[e.target].frame_id == frame)
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .map(|e| e.target)
}

/// `findMarkerInPrevFrame` / `findMarkerInNextFrame`.  Correspondence targets
/// are owned point indices; selecting the highest-score unused point retains
/// the native search behavior without frame-owned pointer traversal.
pub fn find_marker_in_prev_frame(
    points: &[TrajectoryPoint],
    edges: &[Vec<TrajectoryCorrespondence>],
    current: usize,
    jump: i32,
) -> Option<usize> {
    find_marker(points, edges, current, jump, false)
}
pub fn find_marker_in_next_frame(
    points: &[TrajectoryPoint],
    edges: &[Vec<TrajectoryCorrespondence>],
    current: usize,
    jump: i32,
    num_frames: i32,
) -> Option<usize> {
    (points.get(current)?.frame_id + jump < num_frames)
        .then(|| find_marker(points, edges, current, jump, true))
        .flatten()
}
pub fn build_trajectory_forward(
    points: &mut [TrajectoryPoint],
    edges: &[Vec<TrajectoryCorrespondence>],
    start: usize,
    num_frames: i32,
    min_length: usize,
    max_jumps: usize,
) -> Trajectory {
    let mut answer = Trajectory {
        points: vec![start],
        ..Default::default()
    };
    let (mut previous, mut next) = (0usize, 1i32);
    loop {
        let current = answer.points[answer.points.len() - 1 - previous];
        if let Some(found) = find_marker(points, edges, current, next, true) {
            points[current].used = true;
            answer.points.push(found);
            previous = 0;
            next = 1;
        } else {
            next += 1;
            if next as usize > max_jumps {
                previous += 1;
                if previous >= answer.points.len() || previous > max_jumps {
                    break;
                }
                next = previous as i32 + 1;
            }
        }
        if points[current].frame_id + next >= num_frames && next as usize > max_jumps {
            break;
        }
    }
    if answer.points.len() < min_length / 2 {
        for &id in &answer.points {
            points[id].used = false;
        }
    }
    answer
}
pub fn build_trajectory_backward(
    points: &mut [TrajectoryPoint],
    edges: &[Vec<TrajectoryCorrespondence>],
    trajectory: &mut Trajectory,
    max_jumps: usize,
) {
    let (mut previous, mut jump) = (0usize, 1i32);
    loop {
        let current = trajectory.points[previous];
        if points[current].frame_id <= 0 {
            break;
        }
        if let Some(found) = find_marker(points, edges, current, jump, false) {
            points[found].used = true;
            trajectory.points.insert(0, found);
            previous = 0;
            jump = 1;
        } else {
            jump += 1;
            if jump as usize >= max_jumps {
                previous += 1;
                if previous >= trajectory.points.len() {
                    break;
                }
                jump = previous as i32 + 1;
            }
        }
    }
}
pub fn find_trajectories(
    points: &mut [TrajectoryPoint],
    edges: &[Vec<TrajectoryCorrespondence>],
    num_frames: i32,
    max_markers_prev_frame: usize,
    max_jumps: usize,
) -> Result<Vec<Trajectory>, String> {
    let minimum = ((num_frames as f32 * 0.1) as usize)
        .max(10)
        .min(num_frames as usize);
    let zero = num_frames / 2 + num_frames % 2 - 1;
    let mut result = Vec::new();
    for frame in zero..num_frames - 1 {
        for id in 0..points.len() {
            if points[id].frame_id == frame && (!points[id].used || id < max_markers_prev_frame) {
                let mut t =
                    build_trajectory_forward(points, edges, id, num_frames, minimum, max_jumps);
                if t.points.len() > minimum / 2 {
                    build_trajectory_backward(points, edges, &mut t, max_jumps);
                    result.push(t);
                }
            }
        }
    }
    if result.is_empty() {
        Err(format!(
            "after pairwise correspondence no trajectory with minimum length of {minimum} projections could be found"
        ))
    } else {
        Ok(result)
    }
}
pub fn write_imod_tiltalign(
    alpha: f64,
    path: &str,
    basename: &str,
    discard: &[bool],
    tilt: i32,
    rotation: i32,
    magnification: i32,
) -> String {
    let include = discard
        .iter()
        .enumerate()
        .filter(|(_, d)| !**d)
        .map(|(i, _)| (i + 1).to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "ModelFile\t{path}{basename}.fid.txt\nOutputTiltFile\t{path}{basename}.tlt\nOutputTransformFile\t{path}{basename}.xf\nRotationAngle\t{alpha}\nIncludeList {include}\nTiltFile\t{path}{basename}.rawtlt\nAngleOffset\t0.0\nRotOption\t{rotation}\nRotationFixedView 1\nTiltOption\t{tilt}\nTiltFixedView 1\nMagReferenceView\t1\nMagOption\t{magnification}\nMagDefaultGrouping\t4\nCompOption 0\nXStretchOption\t0\nSkewOption\t0\nResidualReportCriterion\t2.0\nSurfacesToAnalyze\t0\nMetroFactor\t0.25\nMaximumCycles\t5000\nAxisZShift\t0\nLocalAlignments\t0\n"
    )
}
/// C `getMarkerTypeFromTrajectory`.
pub fn get_marker_type_from_trajectory(
    trajectories: &[Trajectory],
    points: &[TrajectoryPoint],
) -> Vec<i32> {
    trajectories
        .iter()
        .map(|t| {
            ((t.points.iter().map(|&i| points[i].marker_type).sum::<i32>() as f32
                / t.points.len() as f32)
                + 0.5)
                .floor() as i32
        })
        .collect()
}
/// C `writeIMODfidModel`, rendered as owned text.
pub fn write_imod_fid_model(
    trajectories: &[Trajectory],
    points: &[TrajectoryPoint],
    width: i32,
    height: i32,
    num_frames: i32,
) -> String {
    let mut s = format!(
        "imod 1\nmax {width} {height} {num_frames}\noffsets 0 0 0\nangles 0 0 0\nscale 1 1 1\nmousemode 1\ndrawmode 1\ndrawmode 1\nb&w_level 0,200\nresolution 3\nthreshold 128\npixsize 1\nunits pixels\nsymbol circle\nsize 7\nobject 0 {} 0\nname\ncolor 0 1 0 0\nopen\nlinewidth 1\nsurfsize  0\npointsize 0\naxis      0\ndrawmode  1\n",
        trajectories.len()
    );
    for (i, t) in trajectories.iter().enumerate() {
        s.push_str(&format!("contour {i} 0 {}\n", t.points.len()));
        for &id in &t.points {
            let p = &points[id];
            s.push_str(&format!("{} {} {}\n", p.x + 1., p.y + 1., p.frame_id));
        }
    }
    s
}
/// C `fiducialModel2Trajectory`, returning an owned point arena plus trajectory indexes.
pub fn fiducial_model_to_trajectory(
    input: &str,
) -> Result<(Vec<Trajectory>, Vec<TrajectoryPoint>), String> {
    let mut words = input.split_whitespace();
    while words.next() != Some("object") {}
    let _ = words.next();
    let count: usize = words
        .next()
        .ok_or("missing object count")?
        .parse()
        .map_err(|_| "bad object count")?;
    let mut result = Vec::new();
    let mut points = Vec::new();
    for _ in 0..count {
        while words.next() != Some("contour") {}
        let _ = words.next();
        let _ = words.next();
        let n: usize = words
            .next()
            .ok_or("missing contour count")?
            .parse()
            .map_err(|_| "bad contour count")?;
        let mut t = Trajectory::default();
        for _ in 0..n {
            let x = words
                .next()
                .ok_or("missing x")?
                .parse()
                .map_err(|_| "bad x")?;
            let y = words
                .next()
                .ok_or("missing y")?
                .parse()
                .map_err(|_| "bad y")?;
            let frame_id = words
                .next()
                .ok_or("missing frame")?
                .parse()
                .map_err(|_| "bad frame")?;
            t.points.push(points.len());
            points.push(TrajectoryPoint {
                x,
                y,
                frame_id,
                marker_type: 0,
                used: false,
            });
        }
        result.push(t);
    }
    Ok((result, points))
}
/// C `writeMATLABcontour` text transformation.
pub fn write_matlab_contour(
    fiducial: &str,
    dataset: &str,
    num_frames: usize,
) -> Result<String, String> {
    let (t, p) = fiducial_model_to_trajectory(fiducial)?;
    let mut s = format!("{dataset} = [\n");
    for tr in t {
        let mut row = vec![0f32; num_frames];
        for id in tr.points {
            let q = &p[id];
            if q.frame_id > 0 && (q.frame_id as usize) <= num_frames {
                row[q.frame_id as usize - 1] = q.x;
            }
        }
        for value in row {
            s.push_str(&format!("{value},"));
        }
        s.push_str(";\n");
    }
    s.push_str("];\n");
    Ok(s)
}
/// C `writeIMODtiltScript`, using explicit file ownership instead of C streams.
pub fn write_imod_tilt_script(
    path: &Path,
    width: i32,
    height: i32,
    aligned: &str,
    reconstruction: &str,
    thickness: i32,
    tilts: &[f32],
) -> std::io::Result<()> {
    std::fs::write(
        path,
        format!(
            "# Command file to run Tilt\n$tilt\n{aligned}\n{reconstruction}\nIMAGEBINNED 1\nFULLIMAGE {width} {height}\nLOG 0.0\nMODE 2\nOFFSET 0.0\nPARALLEL\nRADIAL 0.35 0.05\nSCALE 1.39 500\nSHIFT 0.0 0.0\nSUBSETSTART 0 0\nTHICKNESS {thickness}\nANGLES {}\nXAXISTILT 0.0\nDONE\n$if (-e ./savework) ./savework\n",
            tilts
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    )
}
/// C `freeTrajectoryVector`; owned Rust trajectories and their point arena drop exactly once.
pub fn free_trajectory_vector(trajectories: Vec<Trajectory>, points: Vec<TrajectoryPoint>) {
    drop(trajectories);
    drop(points);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructors_keep_index_relationships_non_owning() {
        let mut original = trajectory_with_parent(7);
        original.points.push(3);
        original.children.push(9);
        assert_eq!(trajectory(), Trajectory::default());
        assert_eq!(trajectory_copy(&original), original);
        free_trajectory(original);
    }
    #[test]
    fn model_roundtrip_and_marker_type() {
        let p = vec![TrajectoryPoint {
            x: 2.,
            y: 3.,
            frame_id: 1,
            marker_type: 2,
            used: false,
        }];
        let t = vec![Trajectory {
            points: vec![0],
            ..Default::default()
        }];
        let text = write_imod_fid_model(&t, &p, 10, 10, 1);
        let (r, q) = fiducial_model_to_trajectory(&text).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(q[0].x, 3.);
        assert_eq!(get_marker_type_from_trajectory(&t, &p), vec![2]);
    }
}
