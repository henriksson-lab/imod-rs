//! Translation of `IMOD/3dmod/rescale.cpp` and `rescale.h`.
//!
//! The Qt docking dialog and the image/cache services are represented by one
//! explicit boundary.  The dialog state and scaling arithmetic stay in this
//! translation unit, as in the C++ source.
#![allow(dead_code)]

use crate::imod::libiimod::iimage::IIFORMAT_COMPLEX;
use crate::imod::libiimod::mrcfiles::{mrc_complex_smin_smax, mrc_get_complex_scale};

pub const BLACKNEW: i32 = 32;
pub const WHITENEW: i32 = 223;

/// The fields of an `ImodImageFile` used by `rescale.cpp`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageScaleImage {
    pub filename: String,
    pub amin: f32,
    pub amax: f32,
    pub amean: f32,
    pub smin: f32,
    pub smax: f32,
    pub format: i32,
    pub type_: i32,
    pub file_open: bool,
}

/// Source fields from `ViewInfo` and `LoadInfo` consumed by this unit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageScaleView {
    pub image: ImageScaleImage,
    /// `ViewInfo::imageList`; indexes retain the C image-list coordinate.
    pub image_list: Vec<ImageScaleImage>,
    /// `ViewInfo::fileCopies`; only copies 1 through `num_read_threads - 1`
    /// are touched when the current image is copy zero.
    pub file_copies: Vec<ImageScaleImage>,
    pub zmin: i32,
    pub zsize: i32,
    pub zmouse: f32,
    pub multi_file_z: i32,
    pub loading_image: bool,
    pub ushort_store: bool,
    pub black: i32,
    pub white: i32,
    pub range_low: i32,
    pub range_high: i32,
    pub smin: f32,
    pub smax: f32,
    pub axis: i32,
    pub vm_size: i32,
    pub full_cache_flipped: bool,
    pub cur_time: i32,
    pub volume_stack: bool,
    pub num_times: i32,
    pub keep_cache_full: bool,
    pub num_read_threads: i32,
    pub image_is_first_copy: bool,
    /// Whether `image` is already `imageList[cz]`; this preserves the source
    /// pointer comparison before closing and reopening a multi-file image.
    pub image_is_selected_file: bool,
    pub depth: i32,
}

/// Native Qt widgets and the direct image/cache/viewer integration seam.
pub trait ImageScaleNativeBoundary {
    fn raise_dialog(&mut self);
    fn create_dialog(&mut self, title: &str, help: &str);
    fn show_dialog(&mut self);
    fn remove_dialog(&mut self);
    fn set_title(&mut self, title: &str);
    fn set_file_label(&mut self, text: &str);
    fn set_mmm_label(&mut self, text: &str);
    fn set_limit_text(&mut self, which: usize, text: &str);
    fn limit_text(&self, which: usize) -> String;
    fn start_timer(&mut self, milliseconds: i32) -> i32;
    fn kill_timer(&mut self, timer_id: i32);
    fn close_dialog(&mut self);
    fn set_focus(&mut self);
    fn control_key(&mut self, release: bool);
    fn rounded_style(&self) -> bool;
    fn dialog_change_event(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn set_ushort_sliders(
        &mut self,
        range_low: i32,
        range_high: i32,
        smin: f32,
        smax: f32,
        is_float: bool,
    );
    fn set_image_mm(&mut self, image: &mut ImageScaleImage, smin: f32, smax: f32, scale_max: f32);
    fn close_image(&mut self, image: &mut ImageScaleImage);
    fn reopen_image(&mut self, image: &mut ImageScaleImage);
    fn flip(&mut self, view: &mut ImageScaleView);
    fn flush_cache(&mut self, view: &mut ImageScaleView, time: i32);
    fn cache_fill(&mut self, view: &mut ImageScaleView) -> Result<(), String>;
    fn free_data_memory(&mut self, view: &mut ImageScaleView);
    fn image_load(&mut self, view: &mut ImageScaleView) -> Result<(), String>;
    fn scale(&mut self, view: &mut ImageScaleView);
    fn clear_float_info(&mut self, zsize: i32, time: i32);
    fn info_set_bw(&mut self, black: i32, white: i32);
    fn cramp_set_levels(&mut self, black: i32, white: i32);
    fn draw_image(&mut self);
    fn fatal_error(&mut self, message: &str);
}

/// `iscaleDataStruct` plus `sTopWin` from the source file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageScaleData {
    pub dia_open: bool,
    pub vi: ImageScaleView,
    pub min: f32,
    pub max: f32,
    pub black_new: i32,
    pub white_new: i32,
}

/// `ImageScaleWindow` (`rescale.h`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageScaleWindow {
    pub m_timer_id: i32,
    pub m_just_reload: bool,
    pub rounded_style: bool,
}

/// `ImageScaleWindow()`: initialize rescale state through the native image
/// and dialog boundary retained by this translation.
pub fn image_scale_window(
    data: &mut ImageScaleData,
    native: &mut dyn ImageScaleNativeBoundary,
) -> ImageScaleWindow {
    ImageScaleWindow::new(data, native)
}

/// `imodImageScaleDialog`.
pub fn imod_image_scale_dialog(
    data: &mut ImageScaleData,
    native: &mut dyn ImageScaleNativeBoundary,
) {
    if data.dia_open {
        native.raise_dialog();
        return;
    }
    data.black_new = BLACKNEW * if data.vi.ushort_store { 256 } else { 1 };
    data.white_new = WHITENEW * if data.vi.ushort_store { 256 } else { 1 };
    if data.vi.smin == 0. && data.vi.smax == 0. {
        data.vi.smin = data.vi.image.amin;
        data.vi.smax = data.vi.image.amax;
    }
    native.create_dialog("Reload Image", "imageScale.html#TOP");
    data.dia_open = true;
    native.show_dialog();
}

/// `imodImageScaleUpdate`.
pub fn imod_image_scale_update(
    data: &mut ImageScaleData,
    window: &mut ImageScaleWindow,
    native: &mut dyn ImageScaleNativeBoundary,
) {
    if data.dia_open {
        window.show_file_and_mmm(data, native);
    }
}

impl ImageScaleWindow {
    /// `ImageScaleWindow::ImageScaleWindow`.
    pub fn new(data: &mut ImageScaleData, native: &mut dyn ImageScaleNativeBoundary) -> Self {
        let mut window = Self::default();
        window.compute_scale(data);
        window.show_file_and_mmm(data, native);
        window.update_limits(data, native);
        native.set_title("Image Scale");
        window
    }

    /// `ImageScaleWindow::~ImageScaleWindow`.
    pub fn destroy(&mut self) {}

    /// `ImageScaleWindow::buttonPressed`.
    pub fn button_pressed(
        &mut self,
        which: i32,
        data: &mut ImageScaleData,
        native: &mut dyn ImageScaleNativeBoundary,
    ) {
        native.set_focus();
        self.m_just_reload = false;
        match which {
            2 => {
                self.m_just_reload = true;
                self.m_timer_id = native.start_timer(10);
                if self.m_timer_id == 0 {
                    self.apply_limits(data, native);
                }
            }
            0 => {
                self.m_timer_id = native.start_timer(10);
                if self.m_timer_id == 0 {
                    self.apply_limits(data, native);
                }
            }
            1 => {
                self.compute_scale(data);
                self.update_limits(data, native);
            }
            _ => {}
        }
    }

    /// `ImageScaleWindow::timerEvent`.
    pub fn timer_event(
        &mut self,
        data: &mut ImageScaleData,
        native: &mut dyn ImageScaleNativeBoundary,
    ) {
        native.kill_timer(self.m_timer_id);
        self.apply_limits(data, native);
    }

    /// `ImageScaleWindow::updateLimits`.
    pub fn update_limits(
        &mut self,
        data: &ImageScaleData,
        native: &mut dyn ImageScaleNativeBoundary,
    ) {
        let min = format!("{}", data.min);
        let max = format!("{}", data.max);
        native.set_limit_text(0, &min);
        native.set_limit_text(1, &max);
    }

    /// `ImageScaleWindow::showFileAndMMM`.
    pub fn show_file_and_mmm(
        &mut self,
        data: &mut ImageScaleData,
        native: &mut dyn ImageScaleNativeBoundary,
    ) {
        let view = &mut data.vi;
        if view.multi_file_z > 0 {
            if view.loading_image {
                return;
            }
            let cz = (view.zmouse + 0.5 + view.zmin as f32) as i32;
            if cz >= 0 && cz < view.zsize && !view.image_is_selected_file {
                if let Some(image) = view.image_list.get_mut(cz as usize) {
                    native.close_image(&mut view.image);
                    view.image = image.clone();
                    native.reopen_image(&mut view.image);
                    view.image_is_selected_file = true;
                }
            }
        }
        native.set_file_label(&format!("File: {}", view.image.filename));
        native.set_mmm_label(&format!(
            "Min: {}    Max: {}    Mean: {}",
            view.image.amin, view.image.amax, view.image.amean
        ));
    }

    /// `ImageScaleWindow::computeScale`.
    pub fn compute_scale(&mut self, data: &mut ImageScaleData) {
        let view = &data.vi;
        let (mut smin, mut smax) = (view.image.smin, view.image.smax);
        let kscale = mrc_get_complex_scale();
        if view.image.format == IIFORMAT_COMPLEX {
            (smin, smax) = mrc_complex_smin_smax(smin, smax);
        }
        let slidecur = (view.white - view.black) as f32;
        let rangecur = smax - smin;
        let slidenew = (data.white_new - data.black_new) as f32;
        let rangenew = slidecur * rangecur / slidenew;
        let slide_max = if view.ushort_store { 65535. } else { 255. };
        data.min = smin
            + (slidenew * rangenew / slide_max)
                * (view.black as f32 / slidecur - data.black_new as f32 / slidenew);
        data.max = data.min + rangenew;
        if view.image.format == IIFORMAT_COMPLEX {
            data.max = ((data.max as f64).exp() - 1.) as f32 / kscale;
            let min_sign = if data.min < 0. { -1. } else { 1. };
            data.min = data.min.abs();
            data.min = ((data.min as f64).exp() - 1.) as f32 / (kscale * min_sign);
        }
    }

    /// `ImageScaleWindow::applyLimits`.
    pub fn apply_limits(
        &mut self,
        data: &mut ImageScaleData,
        native: &mut dyn ImageScaleNativeBoundary,
    ) {
        if data.vi.loading_image {
            return;
        }
        data.min = native.limit_text(0).parse().unwrap_or(0.);
        data.max = native.limit_text(1).parse().unwrap_or(0.);
        let black = data.black_new;
        let white = data.white_new;
        let scale_max = if data.vi.ushort_store { 65535. } else { 255. };
        if !self.m_just_reload {
            data.vi.black = black;
            data.vi.white = white;
            data.vi.smin = data.min;
            data.vi.smax = data.max;
            native.set_image_mm(&mut data.vi.image, data.min, data.max, scale_max);
            if data.vi.ushort_store {
                native.set_ushort_sliders(
                    data.vi.range_low,
                    data.vi.range_high,
                    data.vi.image.smin,
                    data.vi.image.smax,
                    data.vi.image.type_ == 6,
                );
            }
            if data.vi.multi_file_z > 0 {
                for k in 0..data.vi.zsize {
                    if let Some(image) = data.vi.image_list.get_mut((k + data.vi.zmin) as usize) {
                        native.set_image_mm(image, data.min, data.max, scale_max);
                    }
                }
            }
            if data.vi.num_read_threads > 1 && data.vi.image_is_first_copy {
                for k in 1..data.vi.num_read_threads as usize {
                    if let Some(image) = data.vi.file_copies.get_mut(k) {
                        native.set_image_mm(image, data.min, data.max, scale_max);
                    }
                }
            }
        }
        let reflip = data.vi.axis == 2 && (data.vi.vm_size == 0 || data.vi.full_cache_flipped);
        if reflip {
            native.flip(&mut data.vi);
        }
        if data.vi.vm_size != 0 {
            let time = data.vi.cur_time;
            native.flush_cache(&mut data.vi, time);
            if self.m_just_reload
                && !data.vi.volume_stack
                && data.vi.num_times == 0
                && data.vi.image.file_open
            {
                native.close_image(&mut data.vi.image);
                native.reopen_image(&mut data.vi.image);
            }
            if data.vi.keep_cache_full && native.cache_fill(&mut data.vi).is_err() {
                native.fatal_error("3DMOD: Fatal error rereading image file\\n");
                return;
            }
        } else {
            native.free_data_memory(&mut data.vi);
            if data.vi.image.file_open {
                native.close_image(&mut data.vi.image);
            }
            if native.image_load(&mut data.vi).is_err() {
                native.fatal_error("3DMOD: Fatal error rereading image file\\n");
                return;
            }
            if data.vi.depth == 8 {
                native.scale(&mut data.vi);
            }
        }
        if reflip {
            native.flip(&mut data.vi);
        }
        native.clear_float_info(-data.vi.zsize, data.vi.cur_time);
        if !self.m_just_reload {
            native.info_set_bw(black, white);
            native.cramp_set_levels(black, white);
        }
        native.draw_image();
    }

    /// `ImageScaleWindow::topChangeEvent`.
    pub fn top_change_event(&mut self, native: &mut dyn ImageScaleNativeBoundary) {
        self.rounded_style = native.rounded_style();
        native.dialog_change_event();
        native.check_and_set_mac_menu();
    }

    /// `ImageScaleWindow::topCloseEvent`.
    pub fn top_close_event(
        &mut self,
        data: &mut ImageScaleData,
        native: &mut dyn ImageScaleNativeBoundary,
    ) {
        native.remove_dialog();
        data.dia_open = false;
    }

    /// `ImageScaleWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, native: &mut dyn ImageScaleNativeBoundary, close_key: bool) {
        if close_key {
            native.close_dialog();
        } else {
            native.control_key(false);
        }
    }

    /// `ImageScaleWindow::keyReleaseEvent`.
    pub fn key_release_event(&mut self, native: &mut dyn ImageScaleNativeBoundary) {
        native.control_key(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        texts: [String; 2],
        draws: i32,
        mm: i32,
        closes: i32,
        reopens: i32,
        focuses: i32,
    }
    impl ImageScaleNativeBoundary for Native {
        fn raise_dialog(&mut self) {}
        fn create_dialog(&mut self, _: &str, _: &str) {}
        fn show_dialog(&mut self) {}
        fn remove_dialog(&mut self) {}
        fn set_title(&mut self, _: &str) {}
        fn set_file_label(&mut self, _: &str) {}
        fn set_mmm_label(&mut self, _: &str) {}
        fn set_limit_text(&mut self, i: usize, s: &str) {
            self.texts[i] = s.into()
        }
        fn limit_text(&self, i: usize) -> String {
            self.texts[i].clone()
        }
        fn start_timer(&mut self, _: i32) -> i32 {
            0
        }
        fn kill_timer(&mut self, _: i32) {}
        fn close_dialog(&mut self) {}
        fn set_focus(&mut self) {
            self.focuses += 1
        }
        fn control_key(&mut self, _: bool) {}
        fn rounded_style(&self) -> bool {
            false
        }
        fn dialog_change_event(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
        fn set_ushort_sliders(&mut self, _: i32, _: i32, _: f32, _: f32, _: bool) {}
        fn set_image_mm(&mut self, i: &mut ImageScaleImage, a: f32, b: f32, _: f32) {
            i.smin = a;
            i.smax = b;
            self.mm += 1
        }
        fn close_image(&mut self, _: &mut ImageScaleImage) {
            self.closes += 1
        }
        fn reopen_image(&mut self, _: &mut ImageScaleImage) {
            self.reopens += 1
        }
        fn flip(&mut self, _: &mut ImageScaleView) {}
        fn flush_cache(&mut self, _: &mut ImageScaleView, _: i32) {}
        fn cache_fill(&mut self, _: &mut ImageScaleView) -> Result<(), String> {
            Ok(())
        }
        fn free_data_memory(&mut self, _: &mut ImageScaleView) {}
        fn image_load(&mut self, _: &mut ImageScaleView) -> Result<(), String> {
            Ok(())
        }
        fn scale(&mut self, _: &mut ImageScaleView) {}
        fn clear_float_info(&mut self, _: i32, _: i32) {}
        fn info_set_bw(&mut self, _: i32, _: i32) {}
        fn cramp_set_levels(&mut self, _: i32, _: i32) {}
        fn draw_image(&mut self) {
            self.draws += 1
        }
        fn fatal_error(&mut self, _: &str) {}
    }
    #[test]
    fn source_window_constructor_initializes_limits() {
        let mut data = ImageScaleData {
            vi: ImageScaleView {
                image: ImageScaleImage {
                    smin: 2.,
                    smax: 8.,
                    ..Default::default()
                },
                black: 0,
                white: 255,
                ..Default::default()
            },
            black_new: 0,
            white_new: 255,
            ..Default::default()
        };
        let mut native = Native::default();
        let _window = image_scale_window(&mut data, &mut native);
        assert_eq!(native.texts, [String::from("2"), String::from("8")]);
    }
    #[test]
    fn computes_and_applies_source_limits() {
        let mut d = ImageScaleData {
            vi: ImageScaleView {
                image: ImageScaleImage {
                    amin: 0.,
                    amax: 100.,
                    smin: 0.,
                    smax: 100.,
                    ..Default::default()
                },
                black: 0,
                white: 255,
                depth: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut n = Native::default();
        imod_image_scale_dialog(&mut d, &mut n);
        let mut w = ImageScaleWindow::new(&mut d, &mut n);
        w.button_pressed(1, &mut d, &mut n);
        assert!(d.max > d.min);
        n.texts = ["10".into(), "20".into()];
        w.button_pressed(0, &mut d, &mut n);
        assert_eq!((d.vi.image.smin, d.vi.image.smax), (10., 20.));
        assert_eq!(n.draws, 1);
        assert_eq!(n.focuses, 2);
    }
    #[test]
    fn complex_scale_round_trips_source_formula() {
        let mut d = ImageScaleData {
            black_new: BLACKNEW,
            white_new: WHITENEW,
            vi: ImageScaleView {
                image: ImageScaleImage {
                    smin: 1.,
                    smax: 20.,
                    format: IIFORMAT_COMPLEX,
                    ..Default::default()
                },
                black: 0,
                white: 255,
                ..Default::default()
            },
            ..Default::default()
        };
        ImageScaleWindow::default().compute_scale(&mut d);
        assert!(d.max > d.min);
    }

    #[test]
    fn current_multifile_image_is_not_closed_or_reopened() {
        let mut data = ImageScaleData {
            vi: ImageScaleView {
                multi_file_z: 1,
                zsize: 1,
                image_is_selected_file: true,
                image_list: vec![ImageScaleImage {
                    filename: "section.mrc".into(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut native = Native::default();
        ImageScaleWindow::default().show_file_and_mmm(&mut data, &mut native);
        assert_eq!((native.closes, native.reopens), (0, 0));
    }
}
