//! Translation of `IMOD/3dmod/mv_movie.cpp` together with `mv_movie.h`.
//!
//! Qt dialogs, GL readback and snapshot writing are owned by the caller through
//! `MvMovieNativeBoundary`.  This keeps the movie interpolation and montage
//! state exactly in this translation unit without substituting a renderer.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::imat::{
    B3D_X, B3D_Y, B3D_Z, imod_mat_find_vector, imod_mat_get_nat_angles, imod_mat_id, imod_mat_mult,
    imod_mat_new, imod_mat_rot, imod_mat_rotate_vector,
};
use crate::imod::libimod::imodel::{IMOD_CLIPSIZE, IMOD_OBJFLAG_OFF, Iclip_planes, Ipoint};

pub const IMODV_MOVIE_FULLAXIS_X: i32 = -1;
pub const IMODV_MOVIE_FULLAXIS_Y: i32 = 1;
pub const IMODV_MOVIE_START_STATE: i32 = 0;
pub const IMODV_MOVIE_END_STATE: i32 = 1;
pub const VMOVIE_MAX_TRANS_CHANGES: usize = 64;
pub const VMOVIE_FLAG_LONG_WAY: i32 = 1 << 16;
pub const VMOVIE_FLAG_REVERSE: i32 = 1 << 17;
pub const MAX_OBJ_ONOFF: usize = 5000;

/// Original: `MovieTerminus` (`mv_movie.h`).
#[derive(Clone, Debug, PartialEq)]
pub struct MovieTerminus {
    pub rotation: Ipoint,
    pub translate: Ipoint,
    pub clip_point: [Ipoint; IMOD_CLIPSIZE],
    pub zoom_rad: f32,
    pub img_xcenter: i32,
    pub img_ycenter: i32,
    pub img_zcenter: i32,
    pub img_slices: i32,
    pub obj_trans: [u8; VMOVIE_MAX_TRANS_CHANGES],
    pub img_transparency: i32,
}
impl Default for MovieTerminus {
    fn default() -> Self {
        Self {
            rotation: Ipoint::default(),
            translate: Ipoint::default(),
            clip_point: [Ipoint::default(); IMOD_CLIPSIZE],
            zoom_rad: 0.,
            img_xcenter: 0,
            img_ycenter: 0,
            img_zcenter: 0,
            img_slices: 0,
            obj_trans: [0; VMOVIE_MAX_TRANS_CHANGES],
            img_transparency: 0,
        }
    }
}

/// Original: `MovieSegment` (`mv_movie.h`).
#[derive(Clone, Debug, PartialEq)]
pub struct MovieSegment {
    pub num_frames: i32,
    pub view_num: i32,
    pub full_axis: i32,
    pub num_clips: i32,
    pub clip_flags: i32,
    pub clip_normal: [Ipoint; IMOD_CLIPSIZE],
    pub obj_states: Vec<u8>,
    pub trans_change_objs: Vec<i32>,
    pub label: String,
    pub start: MovieTerminus,
    pub end: MovieTerminus,
    pub img_axis_flags: i32,
    pub img_white_level: i32,
    pub img_black_level: i32,
    pub img_false_color: i32,
    pub img_xsize: i32,
    pub img_ysize: i32,
    pub img_zsize: i32,
    pub img_clip_offset: i32,
}
impl Default for MovieSegment {
    fn default() -> Self {
        Self {
            num_frames: 0,
            view_num: 0,
            full_axis: 0,
            num_clips: 0,
            clip_flags: 0,
            clip_normal: [Ipoint::default(); IMOD_CLIPSIZE],
            obj_states: Vec::new(),
            trans_change_objs: Vec::new(),
            label: String::new(),
            start: MovieTerminus::default(),
            end: MovieTerminus::default(),
            img_axis_flags: 0,
            img_white_level: 0,
            img_black_level: 0,
            img_false_color: 0,
            img_xsize: 0,
            img_ysize: 0,
            img_zsize: 0,
            img_clip_offset: 0,
        }
    }
}

/// Original static variables in `mv_movie.cpp`.
#[derive(Clone, Debug)]
pub struct MvMovieState {
    pub movie_dialog_open: bool,
    pub sequence_dialog_open: bool,
    pub saved: i32,
    pub reverse: i32,
    pub longway: i32,
    pub montage: i32,
    pub file_format: i32,
    pub fullaxis: i32,
    pub abort: i32,
    pub frames: i32,
    pub mont_frames: i32,
    pub overlap: i32,
    pub trial_fps: i32,
    pub start_clips: Iclip_planes,
    pub end_clips: Iclip_planes,
    pub segments: Vec<MovieSegment>,
    pub start_obj_trans: Vec<u8>,
    pub end_obj_trans: Vec<u8>,
    pub last_make_nothing: i32,
}
impl Default for MvMovieState {
    fn default() -> Self {
        Self {
            movie_dialog_open: false,
            sequence_dialog_open: false,
            saved: 0,
            reverse: 0,
            longway: 0,
            montage: 0,
            file_format: 0,
            fullaxis: 0,
            abort: 0,
            frames: 10,
            mont_frames: 2,
            overlap: 4,
            trial_fps: 20,
            start_clips: Iclip_planes::default(),
            end_clips: Iclip_planes::default(),
            segments: Vec::new(),
            start_obj_trans: Vec::new(),
            end_obj_trans: Vec::new(),
            last_make_nothing: 0,
        }
    }
}

/// Direct counterpart of the Qt, imodv, image, snapshot, and GL calls made
/// by this source file.  The production bridge must make these actual calls.
pub trait MvMovieNativeBoundary {
    fn close_movie_dialog(&mut self);
    fn remove_movie_dialog(&mut self);
    fn raise_movie_dialog(&mut self);
    fn create_movie_dialog(&mut self) -> bool;
    fn set_sequence_open(&mut self, open: bool);
    fn close_sequence_dialog(&mut self);
    fn remove_sequence_dialog(&mut self);
    fn raise_sequence_dialog(&mut self);
    fn create_sequence_dialog(&mut self) -> bool;
    fn sequence_set_for_four_rows(&mut self);
    fn read_start_end(&mut self, index: usize) -> (f32, f32);
    fn set_start(&mut self, index: usize, value: f32);
    fn set_end(&mut self, index: usize, value: f32);
    fn set_button_states(
        &mut self,
        longway: i32,
        reverse: i32,
        montage: i32,
        format: i32,
        saved: i32,
        fps: i32,
    );
    fn get_button_states(&mut self) -> (i32, i32, i32, i32, i32, i32);
    fn get_frame_boxes(&mut self) -> (i32, i32);
    fn set_frame_boxes(&mut self, frames: i32, mont_frames: i32);
    fn sequence_update_enables(&mut self, movie_enabled: bool, making: bool);
    fn set_non_tif_label(&mut self);
    fn image_movie_state(&mut self, segment: &mut MovieSegment);
    fn image_set_movie_draw_state(&mut self, segment: &MovieSegment);
    fn image_set_movie_end_state(&mut self, end: i32, segment: &MovieSegment);
    fn image_set_thick_trans(&mut self, thickness: i32, transparency: i32);
    fn image_dimensions(&self) -> (i32, i32, i32);
    fn image_location(&self) -> (i32, i32, i32);
    fn image_set_location(&mut self, x: i32, y: i32, z: i32);
    fn image_transparency(&self) -> i32;
    fn image_thickness(&self) -> i32;
    fn model_view(&mut self) -> &mut MovieView;
    fn model_objects(&mut self) -> &mut [MovieObject];
    fn standalone(&self) -> bool;
    fn draw(&mut self);
    fn draw_images(&mut self);
    fn objed_new_view(&mut self);
    fn input(&mut self);
    fn auto_snapshot(&mut self, format: i32);
    fn sleep_millis(&mut self, millis: i32);
    fn ask_duplicate_movie(&mut self) -> i32;
    fn montage_begin(&mut self, _width: i32, _height: i32, _zoom: f32) -> bool {
        false
    }
    fn montage_tile(&mut self, _x: i32, _y: i32, _frames: i32) {}
    fn montage_finish(&mut self, _format: i32) {}
    fn montage_end(&mut self) {}
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MovieView {
    pub rot: Ipoint,
    pub trans: Ipoint,
    pub rad: f32,
    pub clips: Iclip_planes,
    pub fovy: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MovieObject {
    pub flags: u32,
    pub trans: u8,
}

/// Original static: `xinput`.
fn xinput(n: &mut dyn MvMovieNativeBoundary) {
    n.input()
}
/// Original static: `setOneStartOrEnd`.
fn set_one_start_or_end(
    n: &mut dyn MvMovieNativeBoundary,
    start_end: i32,
    index: usize,
    value: f32,
) {
    if start_end == IMODV_MOVIE_START_STATE {
        n.set_start(index, value)
    } else {
        n.set_end(index, value)
    }
}
/// Original static: `readStartEndInts`.
fn read_start_end_ints(n: &mut dyn MvMovieNativeBoundary, index: usize) -> (i32, i32) {
    let (s, e) = n.read_start_end(index);
    (s.round() as i32, e.round() as i32)
}
/// Original static: `setAllStartOrEnd`.
fn set_all_start_or_end(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    start_end: i32,
    obj_trans: &mut Vec<u8>,
) {
    state.fullaxis = 0;
    let (rot, trans, rad, clips) = {
        let v = n.model_view();
        (v.rot, v.trans, v.rad, v.clips.clone())
    };
    for (i, value) in [rot.x, rot.y, rot.z, trans.x, trans.y, trans.z, rad]
        .into_iter()
        .enumerate()
    {
        set_one_start_or_end(n, start_end, i, value);
    }
    if !n.standalone() {
        let (x, y, z) = n.image_location();
        for (i, value) in [
            x + 1,
            y + 1,
            z + 1,
            n.image_transparency(),
            n.image_thickness(),
        ]
        .into_iter()
        .enumerate()
        {
            set_one_start_or_end(n, start_end, i + 7, value as f32);
        }
    }
    obj_trans.clear();
    obj_trans.extend(n.model_objects().iter().map(|o| o.trans));
    let _ = clips;
}
/// Original: `mvMovieSetStart`.
pub fn mv_movie_set_start(state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary) {
    let mut values = std::mem::take(&mut state.start_obj_trans);
    set_all_start_or_end(state, n, IMODV_MOVIE_START_STATE, &mut values);
    state.start_obj_trans = values;
    state.start_clips = n.model_view().clips.clone();
}
/// Original: `mvMovieSetEnd`.
pub fn mv_movie_set_end(state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary) {
    let mut values = std::mem::take(&mut state.end_obj_trans);
    set_all_start_or_end(state, n, IMODV_MOVIE_END_STATE, &mut values);
    state.end_obj_trans = values;
    state.end_clips = n.model_view().clips.clone();
}
/// Original: `mvMovieFullAxis(int)`.
pub fn mv_movie_full_axis(state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary, ixy: i32) {
    mv_movie_set_start(state, n);
    mv_movie_set_end(state, n);
    state.fullaxis = ixy;
}
/// Original: `mvMovieMontSelection`.
pub fn mv_movie_mont_selection(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    mont: i32,
) {
    state.montage = mont;
    n.sequence_update_enables(mont != 0, state.abort == 0)
}
/// Original: `mvMovieQuit`.
pub fn mv_movie_quit(_state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary) {
    n.close_movie_dialog()
}
/// Original: `mvMovieClosing`.
pub fn mv_movie_closing(state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary) {
    (
        state.longway,
        state.reverse,
        state.montage,
        state.file_format,
        state.saved,
        state.trial_fps,
    ) = n.get_button_states();
    (state.frames, state.mont_frames) = n.get_frame_boxes();
    n.remove_movie_dialog();
    state.movie_dialog_open = false;
    state.abort = 1;
    n.sequence_update_enables(false, false)
}
/// Original: `mvMovieStop`.
pub fn mv_movie_stop(state: &mut MvMovieState) {
    state.abort = 1
}
/// Original: `mvMovieUpdate`.
pub fn mv_movie_update(_state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary) {
    n.set_non_tif_label()
}

/// Original static: `setstep`.
fn setstep(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    index: usize,
    frame: i32,
    lo: i32,
    hi: i32,
) -> (f32, f32) {
    let (mut min, mut max) = n.read_start_end(index);
    if index < 3 && (min - max).abs() > 0.05 && state.fullaxis != 0 {
        state.fullaxis = 0;
    }
    if hi != 0 {
        if min < lo as f32 {
            min = lo as f32;
            n.set_start(index, min)
        }
        if min > hi as f32 {
            min = hi as f32;
            n.set_start(index, min)
        }
        if max < lo as f32 {
            max = lo as f32;
            n.set_end(index, max)
        }
        if max > hi as f32 {
            max = hi as f32;
            n.set_end(index, max)
        }
    }
    if state.reverse != 0 {
        (max, (min - max) / frame as f32)
    } else {
        (min, (max - min) / frame as f32)
    }
}
/// Original static: `makeMovie`.
fn make_movie(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    frames: i32,
    from_sequence: bool,
) -> i32 {
    if frames <= 0 {
        return 0;
    }
    let steps = (frames - 1).max(1);
    n.sequence_update_enables(true, true);
    let mut p = [(0., 0.); 7];
    for (i, e) in p.iter_mut().enumerate() {
        *e = setstep(state, n, i, steps, 0, 0);
    }
    let (dims_x, dims_y, dims_z) = n.image_dimensions();
    let mut ip = [(0., 0.); 5];
    if !n.standalone() {
        for (j, (lo, hi)) in [(1, dims_x), (1, dims_y), (1, dims_z), (0, 100), (1, dims_z)]
            .into_iter()
            .enumerate()
        {
            ip[j] = setstep(state, n, j + 7, steps, lo, hi);
        }
    }
    {
        let v = n.model_view();
        v.rad = p[6].0;
        v.rot = Ipoint {
            x: p[0].0,
            y: p[1].0,
            z: p[2].0,
        };
        v.trans = Ipoint {
            x: p[3].0,
            y: p[4].0,
            z: p[5].0,
        };
        v.clips = state.start_clips.clone();
    }
    if !n.standalone() {
        n.image_set_location(
            (ip[0].0 - 0.5) as i32,
            (ip[1].0 - 0.5) as i32,
            (ip[2].0 - 0.5) as i32,
        );
        n.image_set_thick_trans((ip[4].0 + 0.5) as i32, (ip[3].0 + 0.5) as i32);
    }
    let count = state
        .start_obj_trans
        .len()
        .min(state.end_obj_trans.len())
        .min(3);
    let changed = (0..count).any(|i| state.start_obj_trans[i] != state.end_obj_trans[i]);
    {
        let o = n.model_objects();
        for i in 0..count {
            o[i].trans = state.start_obj_trans[i];
        }
    }
    let has_clip = (0..state.start_clips.count.min(state.end_clips.count) as usize)
        .any(|i| state.start_clips.point[i] != state.end_clips.point[i]);
    if p.iter().all(|x| x.1 == 0.)
        && !has_clip
        && ip.iter().all(|x| x.1 == 0.)
        && !from_sequence
        && !changed
        && state.last_make_nothing < 2
    {
        state.last_make_nothing = n.ask_duplicate_movie();
        if state.last_make_nothing == 0 {
            return 0;
        }
    }
    // The source applies an incremental matrix so its Euler representations
    // follow IMOD's rotation continuity rather than a linear angle path.
    let Some(mut mat) = imod_mat_new(3) else {
        return 0;
    };
    let Some(mut mati) = imod_mat_new(3) else {
        return 0;
    };
    let Some(mut matp) = imod_mat_new(3) else {
        return 0;
    };
    let mut delangle = 360. / (frames - 1).max(1) as f64;
    if state.reverse != 0 {
        delangle *= -1.;
    }
    if state.fullaxis == IMODV_MOVIE_FULLAXIS_X {
        imod_mat_rot(&mut mati, delangle, B3D_X);
    } else if state.fullaxis == IMODV_MOVIE_FULLAXIS_Y {
        imod_mat_rot(&mut mati, delangle, B3D_Y);
    } else {
        imod_mat_rot(&mut mat, -p[0].0 as f64, B3D_X);
        imod_mat_rot(&mut mat, -p[1].0 as f64, B3D_Y);
        imod_mat_rot(&mut mat, -p[2].0 as f64, B3D_Z);
        imod_mat_rot(&mut mat, (p[2].0 + steps as f32 * p[2].1) as f64, B3D_Z);
        imod_mat_rot(&mut mat, (p[1].0 + steps as f32 * p[1].1) as f64, B3D_Y);
        imod_mat_rot(&mut mat, (p[0].0 + steps as f32 * p[0].1) as f64, B3D_X);
        let mut axis = Ipoint::default();
        let mut angle = 0.;
        imod_mat_find_vector(&mat, &mut angle, &mut axis);
        delangle = angle / steps as f64;
        if state.longway != 0 {
            delangle = (angle - 360.) / steps as f64;
        }
        imod_mat_rotate_vector(&mut mati, delangle, &axis);
    }
    state.abort = 0;
    let interval = (1000 / state.trial_fps.max(1)).max(0);
    for frame in 1..=frames {
        if state.saved != 0 {
            n.auto_snapshot(state.file_format)
        } else {
            n.draw();
            n.sleep_millis(interval)
        };
        xinput(n);
        if state.abort != 0 {
            break;
        }
        if frame < frames {
            let frac = frame as f32 / (frames - 1).max(1) as f32;
            {
                let v = n.model_view();
                v.rad *= ((p[6].0 + p[6].1 * steps as f32) / p[6].0).powf(1. / steps as f32);
                imod_mat_id(&mut mat);
                imod_mat_rot(&mut mat, v.rot.z as f64, B3D_Z);
                imod_mat_rot(&mut mat, v.rot.y as f64, B3D_Y);
                imod_mat_rot(&mut mat, v.rot.x as f64, B3D_X);
                imod_mat_mult(&mati, &mat, &mut matp);
                let (mut alpha, mut beta, mut gamma) = (0., 0., 0.);
                imod_mat_get_nat_angles(&matp, &mut alpha, &mut beta, &mut gamma);
                v.rot = Ipoint {
                    x: alpha as f32,
                    y: beta as f32,
                    z: gamma as f32,
                };
                v.trans = Ipoint {
                    x: p[3].0 + frame as f32 * p[3].1,
                    y: p[4].0 + frame as f32 * p[4].1,
                    z: p[5].0 + frame as f32 * p[5].1,
                };
                for pl in 0..state.start_clips.count.min(state.end_clips.count) as usize {
                    v.clips.point[pl] = Ipoint {
                        x: state.start_clips.point[pl].x
                            + frac * (state.end_clips.point[pl].x - state.start_clips.point[pl].x),
                        y: state.start_clips.point[pl].y
                            + frac * (state.end_clips.point[pl].y - state.start_clips.point[pl].y),
                        z: state.start_clips.point[pl].z
                            + frac * (state.end_clips.point[pl].z - state.start_clips.point[pl].z),
                    };
                }
            }
            if !n.standalone() {
                n.image_set_location(
                    (ip[0].0 + frame as f32 * ip[0].1 - 0.5) as i32,
                    (ip[1].0 + frame as f32 * ip[1].1 - 0.5) as i32,
                    (ip[2].0 + frame as f32 * ip[2].1 - 0.5) as i32,
                );
                n.image_set_thick_trans(
                    (ip[4].0 + frame as f32 * ip[4].1 + 0.5) as i32,
                    (ip[3].0 + frame as f32 * ip[3].1 + 0.5) as i32,
                )
            }
            let o = n.model_objects();
            for i in 0..count {
                if state.start_obj_trans[i] != state.end_obj_trans[i] {
                    o[i].trans = (state.start_obj_trans[i] as f32
                        + frac * (state.end_obj_trans[i] as f32 - state.start_obj_trans[i] as f32))
                        .round() as u8;
                }
            }
        }
    }
    let result = state.abort;
    state.abort = 1;
    n.sequence_update_enables(true, false);
    n.objed_new_view();
    result
}
/// Original static: `makeMontage`; GL readback/scale-bar composition remain direct boundary calls.
fn make_montage(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    frames: i32,
    mut overlap: i32,
) {
    if frames <= 1 {
        return;
    }
    let (x, y, _) = n.image_dimensions();
    overlap = overlap.max(0).min(x / 2).min(y / 2);
    let width = frames * x - (frames - 1) * overlap;
    let height = frames * y - (frames - 1) * overlap;
    let zoom = ((width as f32 / x.max(1) as f32).min(height as f32 / y.max(1) as f32)).max(1.);
    if !n.montage_begin(width, height, zoom) {
        return;
    }
    state.abort = 0;
    for iy in 0..frames {
        for ix in 0..frames {
            n.montage_tile(ix, iy, frames);
            xinput(n);
            if state.abort != 0 {
                break;
            }
        }
        if state.abort != 0 {
            break;
        }
    }
    if state.abort == 0 {
        n.montage_finish(state.file_format)
    }
    n.montage_end();
    state.abort = 1;
    n.draw();
}
/// Original: `mvMovieMake`.
pub fn mv_movie_make(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    from_sequence: bool,
) -> i32 {
    (
        state.longway,
        state.reverse,
        state.montage,
        state.file_format,
        state.saved,
        state.trial_fps,
    ) = n.get_button_states();
    (state.frames, state.mont_frames) = n.get_frame_boxes();
    if state.abort != 0 {
        if state.montage != 0 {
            make_montage(state, n, state.mont_frames, state.overlap);
            0
        } else {
            make_movie(state, n, state.frames, from_sequence)
        }
    } else {
        0
    }
}
/// Original: `mvMovieDialog`.
pub fn mv_movie_dialog(state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary, state_on: i32) {
    if state_on == 0 {
        if state.movie_dialog_open {
            n.close_movie_dialog();
        }
        return;
    }
    if state.movie_dialog_open {
        n.raise_movie_dialog();
        return;
    }
    state.saved = 0;
    state.abort = 1;
    if !n.create_movie_dialog() {
        return;
    }
    state.movie_dialog_open = true;
    mv_movie_set_start(state, n);
    mv_movie_set_end(state, n);
    n.set_button_states(
        state.longway,
        state.reverse,
        state.montage,
        state.file_format,
        state.saved,
        state.trial_fps,
    );
    n.set_frame_boxes(state.frames, state.mont_frames);
    n.set_sequence_open(state.sequence_dialog_open);
    if state.sequence_dialog_open {
        n.sequence_update_enables(state.montage == 0, false);
    }
}
/// Original: `mvMovieSequenceDialog`.
pub fn mv_movie_sequence_dialog(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    state_on: i32,
) {
    if state_on == 0 {
        if state.sequence_dialog_open {
            n.close_sequence_dialog();
        }
        return;
    }
    if state.sequence_dialog_open {
        n.raise_sequence_dialog();
        return;
    }
    if !n.create_sequence_dialog() {
        return;
    }
    state.sequence_dialog_open = true;
    n.sequence_set_for_four_rows();
    n.sequence_update_enables(
        state.movie_dialog_open && state.montage == 0,
        state.abort == 0,
    );
    if state.movie_dialog_open {
        n.set_sequence_open(true);
    }
}
/// Original: `mvMovieSequenceClosing`.
pub fn mv_movie_sequence_closing(state: &mut MvMovieState, n: &mut dyn MvMovieNativeBoundary) {
    n.remove_sequence_dialog();
    state.sequence_dialog_open = false;
    if state.movie_dialog_open {
        n.set_sequence_open(false);
    }
}
/// Original: `mvMovieGetSegment`.
pub fn mv_movie_get_segment(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
) -> MovieSegment {
    let (mut segment) = MovieSegment::default();
    (state.frames, state.mont_frames) = n.get_frame_boxes();
    let mut vals = [(0., 0.); 7];
    for (i, v) in vals.iter_mut().enumerate() {
        *v = n.read_start_end(i)
    }
    segment.start.rotation = Ipoint {
        x: vals[0].0,
        y: vals[1].0,
        z: vals[2].0,
    };
    segment.end.rotation = Ipoint {
        x: vals[0].1,
        y: vals[1].1,
        z: vals[2].1,
    };
    segment.start.translate = Ipoint {
        x: vals[3].0,
        y: vals[4].0,
        z: vals[5].0,
    };
    segment.end.translate = Ipoint {
        x: vals[3].1,
        y: vals[4].1,
        z: vals[5].1,
    };
    segment.start.zoom_rad = vals[6].0;
    segment.end.zoom_rad = vals[6].1;
    if !n.standalone() {
        n.image_movie_state(&mut segment);
        for i in 0..5 {
            let (a, b) = read_start_end_ints(n, i + 7);
            match i {
                0 => {
                    segment.start.img_xcenter = a;
                    segment.end.img_xcenter = b
                }
                1 => {
                    segment.start.img_ycenter = a;
                    segment.end.img_ycenter = b
                }
                2 => {
                    segment.start.img_zcenter = a;
                    segment.end.img_zcenter = b
                }
                3 => {
                    segment.start.img_transparency = a;
                    segment.end.img_transparency = b
                }
                _ => {
                    segment.start.img_slices = a;
                    segment.end.img_slices = b
                }
            }
        }
    }
    segment.num_frames = state.frames;
    segment.full_axis = state.fullaxis;
    segment.num_clips = state.start_clips.count.min(state.end_clips.count) as i32;
    segment.clip_flags = state.start_clips.flags as i32;
    for i in 0..segment.num_clips as usize {
        segment.clip_normal[i] = state.start_clips.normal[i];
        segment.start.clip_point[i] = state.start_clips.point[i];
        segment.end.clip_point[i] = state.end_clips.point[i]
    }
    if state.longway != 0 {
        segment.clip_flags |= VMOVIE_FLAG_LONG_WAY
    }
    if state.reverse != 0 {
        segment.clip_flags |= VMOVIE_FLAG_REVERSE
    }
    let objects = n.model_objects();
    segment.obj_states = objects
        .iter()
        .take(MAX_OBJ_ONOFF)
        .map(|o| {
            if o.flags & IMOD_OBJFLAG_OFF == 0 {
                1
            } else {
                0
            }
        })
        .collect();
    for (i, (&a, &b)) in state
        .start_obj_trans
        .iter()
        .zip(&state.end_obj_trans)
        .enumerate()
    {
        if a != b && segment.trans_change_objs.len() < VMOVIE_MAX_TRANS_CHANGES {
            let j = segment.trans_change_objs.len();
            segment.start.obj_trans[j] = a;
            segment.end.obj_trans[j] = b;
            segment.trans_change_objs.push(i as i32)
        }
    }
    if state.fullaxis != 0 && segment.label.is_empty() {
        segment.label = format!(
            "Full 360 {}",
            if state.fullaxis == IMODV_MOVIE_FULLAXIS_X {
                "X"
            } else {
                "Y"
            }
        )
    }
    segment
}
/// Original static: `setSegmentState`.
fn set_segment_state(
    _state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    segment: &MovieSegment,
    term: &MovieTerminus,
) {
    let v = n.model_view();
    v.clips.count = segment.num_clips.clamp(0, IMOD_CLIPSIZE as i32) as u8;
    v.clips.flags = segment.clip_flags as u8;
    for i in 0..v.clips.count as usize {
        v.clips.normal[i] = segment.clip_normal[i];
        v.clips.point[i] = term.clip_point[i]
    }
    let objects = n.model_objects();
    for (i, &on) in segment.obj_states.iter().enumerate().take(objects.len()) {
        if on != 0 {
            objects[i].flags &= !IMOD_OBJFLAG_OFF
        } else {
            objects[i].flags |= IMOD_OBJFLAG_OFF
        }
    }
    for (i, &object) in segment.trans_change_objs.iter().enumerate() {
        if let Some(o) = objects.get_mut(object.max(0) as usize) {
            o.trans = term.obj_trans[i]
        }
    }
}
/// Original: `mvMovieSetSegment`.
pub fn mv_movie_set_segment(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    segment: &MovieSegment,
) {
    (
        state.longway,
        state.reverse,
        state.montage,
        state.file_format,
        state.saved,
        state.trial_fps,
    ) = n.get_button_states();
    for (term, start_end) in [
        (&segment.start, IMODV_MOVIE_START_STATE),
        (&segment.end, IMODV_MOVIE_END_STATE),
    ] {
        for (i, v) in [
            term.rotation.x,
            term.rotation.y,
            term.rotation.z,
            term.translate.x,
            term.translate.y,
            term.translate.z,
            term.zoom_rad,
        ]
        .into_iter()
        .enumerate()
        {
            set_one_start_or_end(n, start_end, i, v)
        }
    }
    n.image_set_movie_draw_state(segment);
    state.frames = segment.num_frames;
    state.fullaxis = segment.full_axis;
    n.set_frame_boxes(state.frames, state.mont_frames);
    set_segment_state(state, n, segment, &segment.start);
    state.start_clips.count = segment.num_clips.clamp(0, IMOD_CLIPSIZE as i32) as u8;
    state.end_clips.count = state.start_clips.count;
    state.start_clips.flags = segment.clip_flags as u8;
    state.end_clips.flags = state.start_clips.flags;
    for i in 0..state.start_clips.count as usize {
        state.start_clips.normal[i] = segment.clip_normal[i];
        state.end_clips.normal[i] = segment.clip_normal[i];
        state.start_clips.point[i] = segment.start.clip_point[i];
        state.end_clips.point[i] = segment.end.clip_point[i]
    }
    state.longway = if segment.clip_flags & VMOVIE_FLAG_LONG_WAY != 0 {
        1
    } else {
        0
    };
    state.reverse = if segment.clip_flags & VMOVIE_FLAG_REVERSE != 0 {
        1
    } else {
        0
    };
    n.set_button_states(
        state.longway,
        state.reverse,
        state.montage,
        state.file_format,
        state.saved,
        state.trial_fps,
    );
    let objects = n.model_objects();
    state.start_obj_trans = objects.iter().map(|o| o.trans).collect();
    state.end_obj_trans = state.start_obj_trans.clone();
    for (i, &obj) in segment.trans_change_objs.iter().enumerate() {
        if let Some(x) = state.start_obj_trans.get_mut(obj.max(0) as usize) {
            *x = segment.start.obj_trans[i]
        }
        if let Some(x) = state.end_obj_trans.get_mut(obj.max(0) as usize) {
            *x = segment.end.obj_trans[i]
        }
    }
    n.draw();
    n.draw_images();
    n.objed_new_view()
}
/// Original: `mvMovieSetTerminus`.
pub fn mv_movie_set_terminus(
    state: &mut MvMovieState,
    n: &mut dyn MvMovieNativeBoundary,
    start_end: i32,
    segment: &MovieSegment,
) {
    let term = if start_end == IMODV_MOVIE_END_STATE {
        &segment.end
    } else {
        &segment.start
    };
    set_segment_state(state, n, segment, term);
    {
        let v = n.model_view();
        v.rot = term.rotation;
        v.trans = term.translate;
        v.rad = term.zoom_rad
    }
    if !n.standalone() {
        n.image_set_movie_end_state(start_end, segment)
    }
    n.draw();
    n.draw_images();
    n.objed_new_view()
}
/// Original: `mvMovieSegmentArray`.
pub fn mv_movie_segment_array(state: &mut MvMovieState) -> &mut Vec<MovieSegment> {
    &mut state.segments
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        start: [f32; 12],
        end: [f32; 12],
        view: MovieView,
        objects: Vec<MovieObject>,
        draws: i32,
        asked: i32,
        movie_create: bool,
        sequence_create: bool,
        movie_closes: usize,
        movie_raises: usize,
        sequence_closes: usize,
        sequence_raises: usize,
        sequence_open: Vec<bool>,
        sequence_enables: Vec<(bool, bool)>,
    }
    impl MvMovieNativeBoundary for N {
        fn close_movie_dialog(&mut self) {
            self.movie_closes += 1;
        }
        fn remove_movie_dialog(&mut self) {}
        fn raise_movie_dialog(&mut self) {
            self.movie_raises += 1;
        }
        fn create_movie_dialog(&mut self) -> bool {
            self.movie_create
        }
        fn set_sequence_open(&mut self, open: bool) {
            self.sequence_open.push(open);
        }
        fn close_sequence_dialog(&mut self) {
            self.sequence_closes += 1;
        }
        fn remove_sequence_dialog(&mut self) {}
        fn raise_sequence_dialog(&mut self) {
            self.sequence_raises += 1;
        }
        fn create_sequence_dialog(&mut self) -> bool {
            self.sequence_create
        }
        fn sequence_set_for_four_rows(&mut self) {}
        fn read_start_end(&mut self, i: usize) -> (f32, f32) {
            (self.start[i], self.end[i])
        }
        fn set_start(&mut self, i: usize, x: f32) {
            self.start[i] = x
        }
        fn set_end(&mut self, i: usize, x: f32) {
            self.end[i] = x
        }
        fn set_button_states(&mut self, _: i32, _: i32, _: i32, _: i32, _: i32, _: i32) {}
        fn get_button_states(&mut self) -> (i32, i32, i32, i32, i32, i32) {
            (0, 0, 0, 0, 0, 20)
        }
        fn get_frame_boxes(&mut self) -> (i32, i32) {
            (3, 2)
        }
        fn set_frame_boxes(&mut self, _: i32, _: i32) {}
        fn sequence_update_enables(&mut self, movie: bool, making: bool) {
            self.sequence_enables.push((movie, making));
        }
        fn set_non_tif_label(&mut self) {}
        fn image_movie_state(&mut self, _: &mut MovieSegment) {}
        fn image_set_movie_draw_state(&mut self, _: &MovieSegment) {}
        fn image_set_movie_end_state(&mut self, _: i32, _: &MovieSegment) {}
        fn image_set_thick_trans(&mut self, _: i32, _: i32) {}
        fn image_dimensions(&self) -> (i32, i32, i32) {
            (10, 10, 10)
        }
        fn image_location(&self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn image_set_location(&mut self, _: i32, _: i32, _: i32) {}
        fn image_transparency(&self) -> i32 {
            0
        }
        fn image_thickness(&self) -> i32 {
            1
        }
        fn model_view(&mut self) -> &mut MovieView {
            &mut self.view
        }
        fn model_objects(&mut self) -> &mut [MovieObject] {
            &mut self.objects
        }
        fn standalone(&self) -> bool {
            true
        }
        fn draw(&mut self) {
            self.draws += 1
        }
        fn draw_images(&mut self) {}
        fn objed_new_view(&mut self) {}
        fn input(&mut self) {}
        fn auto_snapshot(&mut self, _: i32) {}
        fn sleep_millis(&mut self, _: i32) {}
        fn ask_duplicate_movie(&mut self) -> i32 {
            self.asked += 1;
            1
        }
    }
    #[test]
    fn full_axis_and_segment_round_trip() {
        let mut s = MvMovieState {
            abort: 1,
            ..Default::default()
        };
        let mut n = N {
            objects: vec![MovieObject::default()],
            ..Default::default()
        };
        n.view.rad = 4.;
        mv_movie_full_axis(&mut s, &mut n, IMODV_MOVIE_FULLAXIS_X);
        let seg = mv_movie_get_segment(&mut s, &mut n);
        assert_eq!(seg.full_axis, IMODV_MOVIE_FULLAXIS_X);
        assert_eq!(seg.label, "Full 360 X");
    }
    #[test]
    fn movie_steps_display() {
        let mut s = MvMovieState {
            abort: 1,
            ..Default::default()
        };
        let mut n = N::default();
        n.start[0] = 0.;
        n.end[0] = 10.;
        n.start[6] = 2.;
        n.end[6] = 4.;
        assert_eq!(mv_movie_make(&mut s, &mut n, false), 0);
        assert_eq!(n.draws, 3);
        assert!((n.view.rot.x - 10.).abs() < 0.001);
    }
    #[test]
    fn movie_and_sequence_dialog_lifecycles_follow_source_branches() {
        let mut s = MvMovieState::default();
        let mut n = N {
            movie_create: true,
            sequence_create: true,
            ..Default::default()
        };
        mv_movie_dialog(&mut s, &mut n, 1);
        assert!(s.movie_dialog_open);
        assert_eq!(n.sequence_open, vec![false]);
        mv_movie_dialog(&mut s, &mut n, 1);
        assert_eq!(n.movie_raises, 1);
        mv_movie_sequence_dialog(&mut s, &mut n, 1);
        assert!(s.sequence_dialog_open);
        assert_eq!(n.sequence_enables, vec![(true, false)]);
        assert_eq!(n.sequence_open, vec![false, true]);
        mv_movie_sequence_dialog(&mut s, &mut n, 1);
        assert_eq!(n.sequence_raises, 1);
        mv_movie_sequence_closing(&mut s, &mut n);
        assert!(!s.sequence_dialog_open);
        assert_eq!(n.sequence_open, vec![false, true, false]);
        mv_movie_dialog(&mut s, &mut n, 0);
        assert_eq!(n.movie_closes, 1);
    }
    #[test]
    fn montage_selection_uses_the_source_sequence_enable_predicate() {
        let mut s = MvMovieState {
            abort: 0,
            ..Default::default()
        };
        let mut n = N::default();
        mv_movie_mont_selection(&mut s, &mut n, 1);
        mv_movie_mont_selection(&mut s, &mut n, 0);
        assert_eq!(n.sequence_enables, vec![(true, true), (false, true)]);
    }
}
