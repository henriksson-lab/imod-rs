//! Translation of `IMOD/3dmod/formv_sequence.cpp` and `formv_sequence.h`.
//! Autodoc and Qt table/file-dialog calls remain explicit native boundaries.
#![allow(dead_code)]
use crate::imod::libimod::imodel::IMOD_CLIPSIZE;
use crate::imod::three_dmod::mv_image::IMODV_DRAW_CXYZ;
pub const MAX_OBJ_ONOFF: usize = 5000;
pub use crate::imod::three_dmod::mv_movie::{
    MovieSegment, MovieTerminus, VMOVIE_MAX_TRANS_CHANGES,
};
pub trait MovieSequenceAutodocBoundary {
    fn new_autodoc(&mut self) -> i32;
    fn add_section(&mut self, section: &str, name: &str) -> i32;
    fn set_key_value(&mut self, section: &str, index: usize, key: &str, value: String) -> i32;
    fn write(&mut self) -> i32;
    fn done(&mut self);
}
pub trait MovieSequenceNativeBoundary: MovieSequenceAutodocBoundary {
    fn setup_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_sequence_signals(&mut self);
    fn current_row(&self) -> i32;
    fn select_row(&mut self, row: i32);
    fn set_rows(&mut self, segments: &[MovieSegment]);
    fn set_enabled(&mut self, which: i32, value: bool);
    fn movie_get_segment(&self) -> MovieSegment;
    fn movie_set_segment(&mut self, segment: &MovieSegment);
    fn movie_set_terminus(&mut self, which: i32, segment: &MovieSegment);
    fn movie_make(&mut self, all: bool) -> bool;
    fn open_sequence(&mut self) -> Option<String>;
    fn read(&mut self, path: &str) -> i32;
    fn autodoc_read_error(&mut self);
    fn no_movie_segments_error(&mut self);
    fn movie_sequence_conversion_error(&mut self);
    fn number_of_sections(&self, section: &str) -> i32;
    fn get_integer(&self, section: &str, index: usize, key: &str) -> Option<i32>;
    fn get_two_integers(&self, section: &str, index: usize, key: &str) -> Option<(i32, i32)>;
    fn get_three_integers(&self, section: &str, index: usize, key: &str)
    -> Option<(i32, i32, i32)>;
    fn get_float(&self, section: &str, index: usize, key: &str) -> Option<f32>;
    fn get_three_floats(&self, section: &str, index: usize, key: &str) -> Option<(f32, f32, f32)>;
    fn get_string(&self, section: &str, index: usize, key: &str) -> Option<String>;
    fn get_integer_array(
        &self,
        section: &str,
        index: usize,
        key: &str,
        maximum: usize,
    ) -> Option<Vec<i32>>;
    fn choice_save_before_load(&mut self) -> i32;
    fn choice_save_before_close(&mut self) -> i32;
    fn accept_close(&mut self);
    fn ignore_close(&mut self);
    fn movie_sequence_closing(&mut self);
    fn close(&mut self);
    fn close_key(&self) -> bool;
    fn imodv_key_press(&mut self);
    fn imodv_key_release(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn widget_change_event(&mut self);
    fn rounded_style(&self) -> bool;
    fn set_button_width(&mut self, button: &str, rounded: bool, factor: f32, text: &str) -> i32;
    fn set_fixed_button_width(&mut self, button: &str, width: i32);
    fn table_font_width(&self, text: &str) -> i32;
    fn set_table_column_width(&mut self, column: i32, width: i32);
    fn set_table_minimum_width(&mut self, width: i32);
    fn retranslate_ui(&mut self);
    fn top_window_size_hint(&self) -> (i32, i32);
    fn table_size_hint_height(&self) -> i32;
    fn table_font_height(&self) -> i32;
    fn resize_top_window(&mut self, width: i32, height: i32);
}
pub const ADD_AFTER: i32 = 0;
pub const ADD_BEFORE: i32 = 1;
pub const REPLACE: i32 = 2;
pub const DELETE: i32 = 3;
pub const SET_MOVIE: i32 = 4;
pub const SET_START: i32 = 5;
pub const SET_END: i32 = 6;
pub const RUN_ALL: i32 = 7;
pub const SAVE: i32 = 8;
pub const IMODV_MOVIE_START_STATE: i32 = 0;
pub const IMODV_MOVIE_END_STATE: i32 = 1;
#[derive(Debug, Default)]
pub struct MovieSequenceForm {
    pub m_segments: Vec<MovieSegment>,
    pub m_making_movie: bool,
    pub m_movie_enabled: bool,
    pub m_modified: bool,
    pub max_view: i32,
}
impl MovieSequenceForm {
    /// `MovieSequenceForm()` source constructor.
    pub fn new(
        segments: Vec<MovieSegment>,
        max_view: i32,
        n: &mut dyn MovieSequenceNativeBoundary,
    ) -> Self {
        n.setup_ui();
        let mut f = Self {
            m_segments: segments,
            max_view,
            ..Default::default()
        };
        f.init(n);
        f
    }
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.retranslate_ui()
    }
    pub fn top_change_event(&mut self, font: bool, n: &mut dyn MovieSequenceNativeBoundary) {
        n.widget_change_event();
        n.check_and_set_mac_menu();
        if font {
            self.set_font_dependent_widths(n)
        }
    }
    pub fn init(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.set_delete_on_close();
        n.set_always_show_tool_tips();
        n.connect_sequence_signals();
        self.set_font_dependent_widths(n);
        self.m_modified = false
    }
    pub fn set_for_four_rows(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let (width, height) = n.top_window_size_hint();
        let desired = (5.6 * (n.table_font_height() + 3) as f32) as i32;
        n.resize_top_window(width, height + desired - n.table_size_hint_height());
    }
    pub fn update_enables(
        &mut self,
        movie_enabled: bool,
        making_movie: bool,
        n: &mut dyn MovieSequenceNativeBoundary,
    ) {
        let size = self.m_segments.len();
        let selected = size > 0 && n.current_row() >= 0;
        n.set_enabled(
            ADD_AFTER,
            movie_enabled && (!size.ne(&0) || n.current_row() >= 0),
        );
        n.set_enabled(ADD_BEFORE, movie_enabled && selected);
        n.set_enabled(REPLACE, movie_enabled && selected);
        n.set_enabled(DELETE, selected);
        n.set_enabled(SET_MOVIE, movie_enabled && !making_movie && selected);
        n.set_enabled(SET_START, !making_movie && selected);
        n.set_enabled(SET_END, !making_movie && selected);
        n.set_enabled(RUN_ALL, movie_enabled && !making_movie);
        n.set_enabled(SAVE, size > 0);
        self.m_making_movie = making_movie;
        self.m_movie_enabled = movie_enabled
    }
    pub fn add_at_index(&mut self, index: i32, n: &mut dyn MovieSequenceNativeBoundary) {
        let segment = n.movie_get_segment();
        let index = index.max(0) as usize;
        if index >= self.m_segments.len() {
            self.m_segments.push(segment)
        } else {
            self.m_segments.insert(index, segment)
        }
        self.load_table(n);
        n.select_row(index as i32);
        self.update_enables(self.m_movie_enabled, self.m_making_movie, n);
        self.m_modified = true
    }
    pub fn add_after_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let index = n.current_row() + 1;
        if index == 0 && !self.m_segments.is_empty() {
            return;
        }
        self.add_at_index(index, n)
    }
    pub fn add_before_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let index = n.current_row();
        if index < 0 || self.m_segments.is_empty() {
            return;
        }
        self.add_at_index(index, n)
    }
    pub fn replace_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let index = n.current_row();
        if index < 0 || index as usize >= self.m_segments.len() {
            return;
        }
        let mut segment = n.movie_get_segment();
        if !self.m_segments[index as usize].label.is_empty() {
            segment.label = self.m_segments[index as usize].label.clone()
        }
        self.m_segments[index as usize] = segment;
        self.load_row(index, n);
        self.m_modified = true
    }
    pub fn delete_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let mut index = n.current_row();
        if index < 0 || index as usize >= self.m_segments.len() {
            return;
        }
        self.m_segments.remove(index as usize);
        self.load_table(n);
        if index as usize >= self.m_segments.len() {
            index = self.m_segments.len() as i32 - 1
        }
        if index >= 0 {
            n.select_row(index)
        }
        self.update_enables(self.m_movie_enabled, self.m_making_movie, n);
        self.m_modified = true
    }
    pub fn set_movie_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        if let Some(s) = self.m_segments.get(n.current_row().max(0) as usize) {
            n.movie_set_segment(s)
        }
    }
    pub fn set_start_or_end(&mut self, which: i32, n: &mut dyn MovieSequenceNativeBoundary) {
        if let Some(s) = self.m_segments.get(n.current_row().max(0) as usize) {
            n.movie_set_terminus(which, s)
        }
    }
    pub fn set_start_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        self.set_start_or_end(IMODV_MOVIE_START_STATE, n)
    }
    pub fn set_end_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        self.set_start_or_end(IMODV_MOVIE_END_STATE, n)
    }
    pub fn run_all_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        for s in &self.m_segments {
            n.movie_set_segment(s);
            if n.movie_make(true) {
                break;
            }
        }
    }
    pub fn save_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        self.save_sequence(n);
    }
    pub fn save_sequence(&mut self, n: &mut dyn MovieSequenceNativeBoundary) -> i32 {
        if n.new_autodoc() < 0 {
            n.done();
            return -1;
        }
        for (index, segment) in self.m_segments.iter().enumerate() {
            if n.add_section("MovieSegment", &(index + 1).to_string()) < 0 {
                n.done();
                return -1;
            }
            for (key, value) in [
                ("NumFrames", segment.num_frames.to_string()),
                ("ViewNum", segment.view_num.to_string()),
                ("FullAxis", segment.full_axis.to_string()),
                ("NumClips", segment.num_clips.to_string()),
                ("ClipFlags", segment.clip_flags.to_string()),
                ("ImgAxis", segment.img_axis_flags.to_string()),
                ("NumOnOff", segment.obj_states.len().to_string()),
                (
                    "NumTransChangeObjs",
                    segment.trans_change_objs.len().to_string(),
                ),
            ] {
                if n.set_key_value("MovieSegment", index, key, value) != 0 {
                    n.done();
                    return -1;
                }
            }
            if segment.img_axis_flags & IMODV_DRAW_CXYZ != 0 {
                for (key, value) in [
                    (
                        "ImgSize",
                        format!(
                            "{} {} {}",
                            segment.img_xsize, segment.img_ysize, segment.img_zsize
                        ),
                    ),
                    (
                        "ImgLevels",
                        format!("{} {}", segment.img_black_level, segment.img_white_level),
                    ),
                    ("ImageFalseColor", segment.img_false_color.to_string()),
                    ("ImgClipOffset", segment.img_clip_offset.to_string()),
                ] {
                    if n.set_key_value("MovieSegment", index, key, value) != 0 {
                        n.done();
                        return -1;
                    }
                }
            }
            for clip in 0..segment.num_clips.max(0) as usize {
                let point = segment.clip_normal[clip];
                if n.set_key_value(
                    "MovieSegment",
                    index,
                    &format!("ClipNormal{}", clip + 1),
                    format!("{} {} {}", point.x, point.y, point.z),
                ) != 0
                {
                    n.done();
                    return -1;
                }
            }
            if !segment.label.is_empty()
                && n.set_key_value("MovieSegment", index, "Label", segment.label.clone()) != 0
            {
                n.done();
                return -1;
            }
            if n.set_key_value(
                "MovieSegment",
                index,
                "ObjectOnOff",
                segment
                    .obj_states
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            ) != 0
            {
                n.done();
                return -1;
            }
            if n.set_key_value(
                "MovieSegment",
                index,
                "TransChangeObjs",
                segment
                    .trans_change_objs
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            ) != 0
            {
                n.done();
                return -1;
            }
            for (end, term) in [(false, &segment.start), (true, &segment.end)] {
                let suffix = if end { "End" } else { "Start" };
                for (key, value) in [
                    (
                        format!("Rotation{suffix}"),
                        format!(
                            "{} {} {}",
                            term.rotation.x, term.rotation.y, term.rotation.z
                        ),
                    ),
                    (
                        format!("Translation{suffix}"),
                        format!(
                            "{} {} {}",
                            term.translate.x, term.translate.y, term.translate.z
                        ),
                    ),
                    (format!("Zoom{suffix}"), term.zoom_rad.to_string()),
                ] {
                    if n.set_key_value("MovieSegment", index, &key, value) != 0 {
                        n.done();
                        return -1;
                    }
                }
                if segment.img_axis_flags & IMODV_DRAW_CXYZ != 0 {
                    for (key, value) in [
                        (
                            format!("ImgCenter{suffix}"),
                            format!(
                                "{} {} {}",
                                term.img_xcenter, term.img_ycenter, term.img_zcenter
                            ),
                        ),
                        (format!("ImgSlices{suffix}"), term.img_slices.to_string()),
                        (
                            format!("ImgTransparency{suffix}"),
                            term.img_transparency.to_string(),
                        ),
                    ] {
                        if n.set_key_value("MovieSegment", index, &key, value) != 0 {
                            n.done();
                            return -1;
                        }
                    }
                }
                for clip in 0..segment.num_clips.max(0) as usize {
                    let point = term.clip_point[clip];
                    if n.set_key_value(
                        "MovieSegment",
                        index,
                        &format!("ClipPoint{suffix}{}", clip + 1),
                        format!("{} {} {}", point.x, point.y, point.z),
                    ) != 0
                    {
                        n.done();
                        return -1;
                    }
                }
                if !segment.trans_change_objs.is_empty()
                    && n.set_key_value(
                        "MovieSegment",
                        index,
                        &format!("ObjTransparency{suffix}"),
                        term.obj_trans[..segment.trans_change_objs.len()]
                            .iter()
                            .map(|value| value.to_string())
                            .collect::<Vec<_>>()
                            .join(" "),
                    ) != 0
                {
                    n.done();
                    return -1;
                }
            }
        }
        let err = n.write();
        n.done();
        self.m_modified = err < 0;
        err
    }
    pub fn load_clicked(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        if self.m_modified {
            let c = n.choice_save_before_load();
            if c == 3 || (c == 1 && self.save_sequence(n) < 0) {
                return;
            }
        }
        let Some(path) = n.open_sequence() else {
            return;
        };
        if n.read(&path) < 0 {
            n.autodoc_read_error();
            n.done();
            return;
        }
        self.m_segments.clear();
        let num_seg = n.number_of_sections("MovieSegment");
        if num_seg <= 0 {
            if num_seg == 0 {
                n.no_movie_segments_error();
            } else {
                n.movie_sequence_conversion_error();
            }
            self.load_table(n);
            return;
        }
        let mut error = false;
        for iseg in 0..num_seg as usize {
            let mut segment = MovieSegment::default();
            let Some(value) = n.get_integer("MovieSegment", iseg, "NumFrames") else {
                error = true;
                break;
            };
            segment.num_frames = value;
            let Some(value) = n.get_integer("MovieSegment", iseg, "ViewNum") else {
                error = true;
                break;
            };
            segment.view_num = value;
            let Some(value) = n.get_integer("MovieSegment", iseg, "FullAxis") else {
                error = true;
                break;
            };
            segment.full_axis = value;
            let Some(value) = n.get_integer("MovieSegment", iseg, "NumClips") else {
                error = true;
                break;
            };
            if value < 0 || value as usize > IMOD_CLIPSIZE {
                error = true;
                break;
            }
            segment.num_clips = value;
            let Some(value) = n.get_integer("MovieSegment", iseg, "ClipFlags") else {
                error = true;
                break;
            };
            segment.clip_flags = value;
            let Some(value) = n.get_integer("MovieSegment", iseg, "ImgAxis") else {
                error = true;
                break;
            };
            segment.img_axis_flags = value;
            segment.img_clip_offset = 0;
            if segment.img_axis_flags & IMODV_DRAW_CXYZ != 0 {
                let Some((x, y, z)) = n.get_three_integers("MovieSegment", iseg, "ImgSize") else {
                    error = true;
                    break;
                };
                segment.img_xsize = x;
                segment.img_ysize = y;
                segment.img_zsize = z;
                let Some((black, white)) = n.get_two_integers("MovieSegment", iseg, "ImgLevels")
                else {
                    error = true;
                    break;
                };
                segment.img_black_level = black;
                segment.img_white_level = white;
                let Some(value) = n.get_integer("MovieSegment", iseg, "ImageFalseColor") else {
                    error = true;
                    break;
                };
                segment.img_false_color = value;
                let Some(value) = n.get_integer("MovieSegment", iseg, "ImgClipOffset") else {
                    error = true;
                    break;
                };
                segment.img_clip_offset = value;
            }
            for cl in 0..segment.num_clips as usize {
                let key = format!("ClipNormal{}", cl + 1);
                let Some((x, y, z)) = n.get_three_floats("MovieSegment", iseg, &key) else {
                    error = true;
                    break;
                };
                segment.clip_normal[cl].x = x;
                segment.clip_normal[cl].y = y;
                segment.clip_normal[cl].z = z;
            }
            if error {
                break;
            }
            if let Some(value) = n.get_string("MovieSegment", iseg, "Label") {
                segment.label = value;
            }
            let Some(num_obj) = n.get_integer("MovieSegment", iseg, "NumOnOff") else {
                error = true;
                break;
            };
            if num_obj <= 0 || num_obj as usize > MAX_OBJ_ONOFF {
                error = true;
                break;
            }
            let Some(value) = n.get_string("MovieSegment", iseg, "ObjectOnOff") else {
                error = true;
                break;
            };
            if value.len() < 2 * num_obj as usize - 1 {
                error = true;
                break;
            }
            segment.obj_states = (0..num_obj as usize)
                .map(|cl| {
                    if value.as_bytes()[2 * cl] == b'0' {
                        0
                    } else {
                        1
                    }
                })
                .collect();
            let Some(num_obj) = n.get_integer("MovieSegment", iseg, "NumTransChangeObjs") else {
                error = true;
                break;
            };
            if num_obj < 0 || num_obj as usize > VMOVIE_MAX_TRANS_CHANGES {
                error = true;
                break;
            }
            if num_obj > 0 {
                let Some(values) =
                    n.get_integer_array("MovieSegment", iseg, "TransChangeObjs", num_obj as usize)
                else {
                    error = true;
                    break;
                };
                if values.len() != num_obj as usize {
                    error = true;
                    break;
                }
                segment.trans_change_objs = values;
            }
            for se in 0..2 {
                let suffix = if se == 0 { "Start" } else { "End" };
                let term = if se == 0 {
                    &mut segment.start
                } else {
                    &mut segment.end
                };
                let key = format!("Rotation{suffix}");
                let Some((x, y, z)) = n.get_three_floats("MovieSegment", iseg, &key) else {
                    error = true;
                    break;
                };
                term.rotation.x = x;
                term.rotation.y = y;
                term.rotation.z = z;
                let key = format!("Translation{suffix}");
                let Some((x, y, z)) = n.get_three_floats("MovieSegment", iseg, &key) else {
                    error = true;
                    break;
                };
                term.translate.x = x;
                term.translate.y = y;
                term.translate.z = z;
                let key = format!("Zoom{suffix}");
                let Some(value) = n.get_float("MovieSegment", iseg, &key) else {
                    error = true;
                    break;
                };
                term.zoom_rad = value;
                if segment.img_axis_flags & IMODV_DRAW_CXYZ != 0 {
                    let key = format!("ImgCenter{suffix}");
                    let Some((x, y, z)) = n.get_three_integers("MovieSegment", iseg, &key) else {
                        error = true;
                        break;
                    };
                    term.img_xcenter = x;
                    term.img_ycenter = y;
                    term.img_zcenter = z;
                    let key = format!("ImgSlices{suffix}");
                    let Some(value) = n.get_integer("MovieSegment", iseg, &key) else {
                        error = true;
                        break;
                    };
                    term.img_slices = value;
                    let key = format!("ImgTransparency{suffix}");
                    let Some(value) = n.get_integer("MovieSegment", iseg, &key) else {
                        error = true;
                        break;
                    };
                    term.img_transparency = value;
                }
                for cl in 0..segment.num_clips as usize {
                    let key = format!("ClipPoint{suffix}{}", cl + 1);
                    let Some((x, y, z)) = n.get_three_floats("MovieSegment", iseg, &key) else {
                        error = true;
                        break;
                    };
                    term.clip_point[cl].x = x;
                    term.clip_point[cl].y = y;
                    term.clip_point[cl].z = z;
                }
                if error {
                    break;
                }
                if num_obj > 0 {
                    let key = format!("ObjTransparency{suffix}");
                    let Some(values) =
                        n.get_integer_array("MovieSegment", iseg, &key, num_obj as usize)
                    else {
                        error = true;
                        break;
                    };
                    if values.len() != num_obj as usize {
                        error = true;
                        break;
                    }
                    for (cl, value) in values.into_iter().enumerate() {
                        term.obj_trans[cl] = value as u8;
                    }
                }
            }
            if error {
                break;
            }
            self.m_segments.push(segment);
        }
        if error {
            n.movie_sequence_conversion_error();
            self.m_segments.clear();
        }
        self.load_table(n);
        n.select_row(0);
        self.update_enables(self.m_movie_enabled, self.m_making_movie, n);
        self.m_modified = false
    }
    pub fn load_table(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.set_rows(&self.m_segments)
    }
    pub fn load_row(&mut self, _: i32, n: &mut dyn MovieSequenceNativeBoundary) {
        self.load_table(n)
    }
    pub fn entry_changed(
        &mut self,
        row: i32,
        column: i32,
        text: &str,
        n: &mut dyn MovieSequenceNativeBoundary,
    ) {
        let Some(s) = self.m_segments.get_mut(row.max(0) as usize) else {
            return;
        };
        if column > 1 {
            s.label = text.trim().into()
        } else if column == 0 {
            s.num_frames = text.trim().parse().unwrap_or(0).max(2)
        } else {
            s.view_num = text.trim().parse().unwrap_or(0).clamp(1, self.max_view - 1)
        }
        self.load_row(row, n);
        self.m_modified = true
    }
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        let rounded = n.rounded_style();
        n.set_button_width("addBeforeButton", rounded, 1.2, "Add Before");
        n.set_button_width("addAfterButton", rounded, 1.2, "Add After");
        n.set_button_width("replaceButton", rounded, 1.2, "Replace");
        n.set_button_width("deleteButton", rounded, 1.2, "Delete");
        n.set_button_width("setMovieButton", rounded, 1.2, "Use as Movie");
        n.set_button_width("setStartButton", rounded, 1.2, "Set to Start");
        n.set_button_width("setEndButton", rounded, 1.2, "Set to End");
        let width = n.set_button_width("runAllButton", rounded, 1.3, "Run All");
        n.set_fixed_button_width("saveButton", width);
        n.set_fixed_button_width("loadButton", width);
        let extra = 8;
        let mut width = (1.2 * n.table_font_width("# Frames") as f32) as i32 + extra;
        n.set_table_column_width(0, width);
        let mut sum = width;
        width = (1.2 * n.table_font_width("View #") as f32) as i32 + extra;
        n.set_table_column_width(1, width);
        sum += width;
        width = n.table_font_width("3 objects w/ extra half-spin") + extra;
        n.set_table_column_width(2, width);
        sum += width;
        n.set_table_minimum_width(sum);
    }
    pub fn top_close_event(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        if self.m_modified {
            let c = n.choice_save_before_close();
            if c == 3 || (c == 1 && self.save_sequence(n) < 0) {
                n.ignore_close();
                return;
            }
        }
        n.movie_sequence_closing();
        self.m_segments.clear();
        n.accept_close()
    }
    pub fn key_press_event(&mut self, navigation: bool, n: &mut dyn MovieSequenceNativeBoundary) {
        if navigation {
            return;
        }
        if n.close_key() {
            n.close()
        } else {
            n.imodv_key_press()
        }
    }
    pub fn key_release_event(&mut self, n: &mut dyn MovieSequenceNativeBoundary) {
        n.imodv_key_release()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    #[derive(Default)]
    struct N {
        row: i32,
        rows: Vec<MovieSegment>,
        current: MovieSegment,
        keys: Vec<String>,
        values: BTreeMap<(String, usize, String), String>,
        sections: i32,
        button_widths: Vec<(String, i32)>,
        column_widths: Vec<(i32, i32)>,
        minimum_width: i32,
        retranslated: usize,
        resized: Vec<(i32, i32)>,
    }
    impl MovieSequenceAutodocBoundary for N {
        fn new_autodoc(&mut self) -> i32 {
            0
        }
        fn add_section(&mut self, _: &str, _: &str) -> i32 {
            self.sections += 1;
            0
        }
        fn set_key_value(&mut self, section: &str, index: usize, key: &str, value: String) -> i32 {
            self.keys.push(key.into());
            self.values
                .insert((section.into(), index, key.into()), value);
            0
        }
        fn write(&mut self) -> i32 {
            0
        }
        fn done(&mut self) {}
    }
    impl MovieSequenceNativeBoundary for N {
        fn setup_ui(&mut self) {}
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_sequence_signals(&mut self) {}
        fn current_row(&self) -> i32 {
            self.row
        }
        fn select_row(&mut self, x: i32) {
            self.row = x
        }
        fn set_rows(&mut self, x: &[MovieSegment]) {
            self.rows = x.into()
        }
        fn set_enabled(&mut self, _: i32, _: bool) {}
        fn movie_get_segment(&self) -> MovieSegment {
            self.current.clone()
        }
        fn movie_set_segment(&mut self, _: &MovieSegment) {}
        fn movie_set_terminus(&mut self, _: i32, _: &MovieSegment) {}
        fn movie_make(&mut self, _: bool) -> bool {
            false
        }
        fn open_sequence(&mut self) -> Option<String> {
            Some("fixture.adoc".into())
        }
        fn read(&mut self, _: &str) -> i32 {
            0
        }
        fn autodoc_read_error(&mut self) {}
        fn no_movie_segments_error(&mut self) {}
        fn movie_sequence_conversion_error(&mut self) {}
        fn number_of_sections(&self, _: &str) -> i32 {
            self.sections
        }
        fn get_integer(&self, section: &str, index: usize, key: &str) -> Option<i32> {
            self.values
                .get(&(section.into(), index, key.into()))?
                .parse()
                .ok()
        }
        fn get_two_integers(&self, section: &str, index: usize, key: &str) -> Option<(i32, i32)> {
            let mut values = self
                .values
                .get(&(section.into(), index, key.into()))?
                .split_whitespace();
            Some((values.next()?.parse().ok()?, values.next()?.parse().ok()?))
        }
        fn get_three_integers(
            &self,
            section: &str,
            index: usize,
            key: &str,
        ) -> Option<(i32, i32, i32)> {
            let mut values = self
                .values
                .get(&(section.into(), index, key.into()))?
                .split_whitespace();
            Some((
                values.next()?.parse().ok()?,
                values.next()?.parse().ok()?,
                values.next()?.parse().ok()?,
            ))
        }
        fn get_float(&self, section: &str, index: usize, key: &str) -> Option<f32> {
            self.values
                .get(&(section.into(), index, key.into()))?
                .parse()
                .ok()
        }
        fn get_three_floats(
            &self,
            section: &str,
            index: usize,
            key: &str,
        ) -> Option<(f32, f32, f32)> {
            let mut values = self
                .values
                .get(&(section.into(), index, key.into()))?
                .split_whitespace();
            Some((
                values.next()?.parse().ok()?,
                values.next()?.parse().ok()?,
                values.next()?.parse().ok()?,
            ))
        }
        fn get_string(&self, section: &str, index: usize, key: &str) -> Option<String> {
            self.values
                .get(&(section.into(), index, key.into()))
                .cloned()
        }
        fn get_integer_array(
            &self,
            section: &str,
            index: usize,
            key: &str,
            maximum: usize,
        ) -> Option<Vec<i32>> {
            let values = self
                .values
                .get(&(section.into(), index, key.into()))?
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            if values.len() > maximum {
                return None;
            }
            Some(values)
        }
        fn choice_save_before_load(&mut self) -> i32 {
            2
        }
        fn choice_save_before_close(&mut self) -> i32 {
            2
        }
        fn accept_close(&mut self) {}
        fn ignore_close(&mut self) {}
        fn movie_sequence_closing(&mut self) {}
        fn close(&mut self) {}
        fn close_key(&self) -> bool {
            false
        }
        fn imodv_key_press(&mut self) {}
        fn imodv_key_release(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
        fn widget_change_event(&mut self) {}
        fn rounded_style(&self) -> bool {
            true
        }
        fn set_button_width(&mut self, button: &str, _: bool, _: f32, text: &str) -> i32 {
            let width = text.len() as i32;
            self.button_widths.push((button.into(), width));
            width
        }
        fn set_fixed_button_width(&mut self, button: &str, width: i32) {
            self.button_widths.push((button.into(), width));
        }
        fn table_font_width(&self, text: &str) -> i32 {
            text.len() as i32
        }
        fn set_table_column_width(&mut self, column: i32, width: i32) {
            self.column_widths.push((column, width));
        }
        fn set_table_minimum_width(&mut self, width: i32) {
            self.minimum_width = width;
        }
        fn retranslate_ui(&mut self) {
            self.retranslated += 1;
        }
        fn top_window_size_hint(&self) -> (i32, i32) {
            (200, 100)
        }
        fn table_size_hint_height(&self) -> i32 {
            30
        }
        fn table_font_height(&self) -> i32 {
            10
        }
        fn resize_top_window(&mut self, width: i32, height: i32) {
            self.resized.push((width, height));
        }
    }
    #[test]
    fn add_replace_delete_and_entry_limits_match_source() {
        let mut n = N {
            row: -1,
            current: MovieSegment {
                num_frames: 9,
                view_num: 2,
                label: "x".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut f = MovieSequenceForm::new(vec![], 4, &mut n);
        f.add_after_clicked(&mut n);
        f.entry_changed(0, 0, "1", &mut n);
        f.entry_changed(0, 1, "9", &mut n);
        assert_eq!(
            (f.m_segments[0].num_frames, f.m_segments[0].view_num),
            (2, 3)
        );
        f.delete_clicked(&mut n);
        assert!(f.m_segments.is_empty());
    }
    #[test]
    fn font_widths_use_source_button_and_table_order() {
        let mut n = N::default();
        let mut form = MovieSequenceForm::new(vec![], 3, &mut n);
        assert_eq!(n.button_widths.len(), 10);
        assert_eq!(n.button_widths[0].0, "addBeforeButton");
        assert_eq!(n.button_widths[7].0, "runAllButton");
        assert_eq!(n.button_widths[8].0, "saveButton");
        assert_eq!(
            n.column_widths
                .iter()
                .map(|&(column, _)| column)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            n.minimum_width,
            n.column_widths.iter().map(|&(_, width)| width).sum()
        );
        form.top_change_event(true, &mut n);
        assert_eq!(n.button_widths.len(), 20);
    }
    #[test]
    fn language_and_four_row_sizing_follow_source_geometry() {
        let mut n = N::default();
        let mut form = MovieSequenceForm::new(vec![], 0, &mut n);
        form.language_change(&mut n);
        form.set_for_four_rows(&mut n);
        assert_eq!(n.retranslated, 1);
        // 100 + int(5.6 * (10 + 3)) - 30
        assert_eq!(n.resized, vec![(200, 142)]);
    }
    #[test]
    fn save_sequence_uses_source_autodoc_keys_for_full_movie_state() {
        let mut segment = MovieSegment {
            num_frames: 4,
            num_clips: 1,
            img_axis_flags: IMODV_DRAW_CXYZ,
            img_xsize: 10,
            img_ysize: 11,
            img_zsize: 12,
            obj_states: vec![1],
            trans_change_objs: vec![0],
            ..Default::default()
        };
        segment.start.rotation.x = 2.;
        segment.end.clip_point[0].z = 3.;
        let mut n = N::default();
        let mut form = MovieSequenceForm::new(vec![segment], 3, &mut n);
        assert_eq!(form.save_sequence(&mut n), 0);
        for key in [
            "NumFrames",
            "ImgSize",
            "ClipNormal1",
            "ObjectOnOff",
            "RotationStart",
            "RotationEnd",
            "ClipPointEnd1",
            "ObjTransparencyStart",
        ] {
            assert!(n.keys.iter().any(|saved| saved == key), "{key}");
        }
    }
    #[test]
    fn load_clicked_round_trips_general_image_clip_object_and_terminus_fields() {
        let mut segment = MovieSegment {
            num_frames: 7,
            view_num: 3,
            full_axis: 2,
            num_clips: 1,
            clip_flags: 5,
            img_axis_flags: IMODV_DRAW_CXYZ,
            img_xsize: 40,
            img_ysize: 50,
            img_zsize: 60,
            img_black_level: 2,
            img_white_level: 90,
            img_false_color: 1,
            img_clip_offset: 4,
            obj_states: vec![1, 0],
            trans_change_objs: vec![9],
            label: "representative".into(),
            ..Default::default()
        };
        segment.clip_normal[0].x = 1.5;
        segment.start.rotation.y = 2.5;
        segment.start.img_xcenter = 4;
        segment.start.clip_point[0].z = 3.5;
        segment.start.obj_trans[0] = 7;
        segment.end.translate.z = 4.5;
        segment.end.img_transparency = 8;
        segment.end.clip_point[0].x = 5.5;
        segment.end.obj_trans[0] = 6;
        let mut n = N::default();
        let mut form = MovieSequenceForm::new(vec![segment.clone()], 4, &mut n);
        assert_eq!(form.save_sequence(&mut n), 0);
        form.m_segments.clear();
        form.load_clicked(&mut n);
        assert_eq!(form.m_segments, vec![segment]);
    }
    #[test]
    fn load_clicked_clears_all_segments_when_a_required_field_is_missing() {
        let segment = MovieSegment {
            num_frames: 7,
            num_clips: 0,
            obj_states: vec![1],
            ..Default::default()
        };
        let mut n = N::default();
        let mut form = MovieSequenceForm::new(vec![segment], 4, &mut n);
        assert_eq!(form.save_sequence(&mut n), 0);
        n.values
            .remove(&("MovieSegment".into(), 0, "NumFrames".into()));
        form.load_clicked(&mut n);
        assert!(form.m_segments.is_empty());
    }
}
