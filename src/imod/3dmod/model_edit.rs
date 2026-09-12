//! Translation of `IMOD/3dmod/model_edit.cpp` and `model_edit.h`.
//!
//! This retains the two dialog state records from the C++ unit.  Qt widgets,
//! global dialog management, drawing, vertex-buffer disposal, image-header
//! scaling, and undo are represented by the explicit boundary below; the
//! source-owned model mutations and their call order remain in this module.
#![allow(dead_code)]

use crate::imod::libimod::imodel::{
    IMOD_UNIT_ANGSTROM, IMOD_UNIT_CM, IMOD_UNIT_KILO, IMOD_UNIT_METER, IMOD_UNIT_MM, IMOD_UNIT_NM,
    IMOD_UNIT_PIXEL, IMOD_UNIT_PM, IMOD_UNIT_UM, Imod, Ipoint,
};

/// `model_edit.cpp`'s direct `ImodView`, undo, drawing, image, VBO, and Qt
/// calls.  This is deliberately a boundary rather than a replacement editor.
pub trait ModelEditNativeBoundary {
    fn raise_model_header(&mut self);
    fn show_model_header(&mut self);
    fn remove_model_header(&mut self);
    fn raise_model_offset(&mut self);
    fn show_model_offset(&mut self);
    fn remove_model_offset(&mut self);
    fn model_change(&mut self);
    fn model_shift(&mut self, offset: &Ipoint);
    fn finish_undo_unit(&mut self);
    fn pixel_changed(&mut self);
    fn set_model_scales_from_image(&mut self, model: &mut Imod, use_z_scale: bool);
    fn draw_model(&mut self);
    fn cleanup_vertex_buffers(&mut self, model: &Imod);
    fn rounded_style(&mut self) -> bool;
    fn dialog_change_event(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn close_dialog(&mut self);
    fn control_key(&mut self, release: bool);
}

/// `ModelHeaderWindow` (`model_edit.h`), including values held by its widgets.
#[derive(Clone, Debug)]
pub struct ModelHeaderWindow {
    pub m_pix_ratio: f32,
    pub m_setting_inc: bool,
    pub draw_checked: bool,
    pub set_incremental_checked: bool,
    pub edit_box: [String; 4],
    pub edit_enabled: [bool; 4],
    pub rounded_style: bool,
    pub closed: bool,
}

impl ModelHeaderWindow {
    /// `ModelHeaderWindow::ModelHeaderWindow`.
    pub fn new(fake_image: bool, zscale: f32, xscale: f32, model: &Imod) -> Self {
        let mut m_pix_ratio = 1.;
        if !fake_image && zscale != 0. && xscale != 0. {
            m_pix_ratio = zscale / xscale;
        }
        m_pix_ratio = 0.001 * (1000. * m_pix_ratio + 0.5).floor();
        let mut window = Self {
            m_pix_ratio,
            m_setting_inc: false,
            draw_checked: false,
            set_incremental_checked: false,
            edit_box: std::array::from_fn(|_| String::new()),
            edit_enabled: [false, true, true, true],
            rounded_style: false,
            closed: false,
        };
        window.update(model);
        window
    }

    /// `ModelHeaderWindow::~ModelHeaderWindow`.
    pub fn destroy(&mut self) {}

    /// `ModelHeaderWindow::valueEntered`.
    pub fn value_entered(&mut self, model: &mut Imod, n: &mut dyn ModelEditNativeBoundary) {
        n.model_change();
        if self.m_setting_inc {
            model.zscale = self.m_pix_ratio * self.edit_box[0].parse::<f32>().unwrap_or(0.);
        } else {
            model.zscale = self.edit_box[1].parse::<f32>().unwrap_or(0.);
        }
        model.res = self.edit_box[2].trim().parse::<i32>().unwrap_or(0);
        set_pixsize_and_units(model, &self.edit_box[3]);
        self.update(model);
        n.pixel_changed();
        n.finish_undo_unit();
        n.draw_model();
    }

    /// `ModelHeaderWindow::setPixelClicked`.
    pub fn set_pixel_clicked(&mut self, model: &mut Imod, n: &mut dyn ModelEditNativeBoundary) {
        n.model_change();
        n.set_model_scales_from_image(model, false);
        n.pixel_changed();
        self.update(model);
        n.finish_undo_unit();
    }

    /// `ModelHeaderWindow::drawToggled`.
    pub fn draw_toggled(
        &mut self,
        state: bool,
        model: &mut Imod,
        n: &mut dyn ModelEditNativeBoundary,
    ) {
        self.draw_checked = state;
        if (state && model.drawmode <= 0) || (!state && model.drawmode > 0) {
            model.drawmode = if state { 1 } else { -1 };
        }
        n.draw_model();
    }

    /// `ModelHeaderWindow::setIncToggled`.
    pub fn set_inc_toggled(&mut self, state: bool) {
        self.m_setting_inc = state;
        self.set_incremental_checked = state;
        self.edit_enabled[0] = state;
        self.edit_enabled[1] = !state;
    }

    /// `ModelHeaderWindow::update`.
    pub fn update(&mut self, model: &Imod) {
        self.draw_checked = model.drawmode > 0;
        self.edit_box[0] = format!("{}", model.zscale / self.m_pix_ratio);
        self.edit_box[1] = format!("{}", model.zscale);
        self.edit_box[2] = format!("{}", model.res as f32);
        let units = match model.units {
            IMOD_UNIT_PIXEL => "pixels",
            IMOD_UNIT_KILO => "km",
            IMOD_UNIT_METER => "m",
            IMOD_UNIT_CM => "cm",
            IMOD_UNIT_MM => "mm",
            IMOD_UNIT_UM => "um",
            IMOD_UNIT_NM => "nm",
            IMOD_UNIT_ANGSTROM => "A",
            IMOD_UNIT_PM => "pm",
            _ => "unknown units",
        };
        self.edit_box[3] = format!("{} {}", model.pixsize, units);
    }

    /// `ModelHeaderWindow::topChangeEvent`.
    pub fn top_change_event(&mut self, n: &mut dyn ModelEditNativeBoundary) {
        self.rounded_style = n.rounded_style();
        n.dialog_change_event();
        n.check_and_set_mac_menu();
    }

    /// `ModelHeaderWindow::topCloseEvent`.
    pub fn top_close_event(&mut self, model: &mut Imod, n: &mut dyn ModelEditNativeBoundary) {
        self.value_entered(model, n);
        n.remove_model_header();
        self.closed = true;
    }

    /// `ModelHeaderWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, close_key: bool, n: &mut dyn ModelEditNativeBoundary) {
        if close_key {
            n.close_dialog();
        } else {
            n.control_key(false);
        }
    }

    /// `ModelHeaderWindow::keyReleaseEvent`.
    pub fn key_release_event(&mut self, n: &mut dyn ModelEditNativeBoundary) {
        n.control_key(true);
    }
}

/// Rust state for static `sData` in `model_edit.cpp`.
#[derive(Clone, Debug, Default)]
pub struct ModelEditData {
    pub dia: Option<ModelHeaderWindow>,
}

/// `openModelEdit` (`model_edit.cpp:45`).
pub fn open_model_edit(
    state: &mut ModelEditData,
    fake_image: bool,
    zscale: f32,
    xscale: f32,
    model: &Imod,
    n: &mut dyn ModelEditNativeBoundary,
) -> i32 {
    if state.dia.as_ref().is_some_and(|dia| !dia.closed) {
        n.raise_model_header();
        return 0;
    }
    state.dia = Some(ModelHeaderWindow::new(fake_image, zscale, xscale, model));
    n.show_model_header();
    0
}

/// `imodModelEditUpdate` (`model_edit.cpp:67`).
pub fn imod_model_edit_update(state: &mut ModelEditData, model: &Imod) {
    if let Some(dia) = &mut state.dia {
        dia.update(model);
    }
}

/// `setPixsizeAndUnits` (`model_edit.cpp:77`).
pub fn set_pixsize_and_units(model: &mut Imod, string: &str) {
    if string.is_empty() {
        return;
    }
    // `atof` stops at the first non-number, whereas Rust's `parse` requires
    // the whole string.  Keep the numeric prefix in this source function.
    let trimmed = string.trim_start();
    let mut end = 0;
    for (index, character) in trimmed.char_indices() {
        if character.is_ascii_digit() || matches!(character, '+' | '-' | '.' | 'e' | 'E') {
            end = index + character.len_utf8();
        } else {
            break;
        }
    }
    let fscale = trimmed[..end].parse::<f32>().unwrap_or(0.);
    if fscale == 0. {
        return;
    }
    model.pixsize = fscale;
    model.units = IMOD_UNIT_PIXEL;
    // Preserve upstream's independent strstr sequence, including overlaps.
    if string.contains("km") {
        model.units = IMOD_UNIT_KILO;
    }
    if string.contains('m') {
        model.units = IMOD_UNIT_METER;
    }
    if string.contains("cm") {
        model.units = IMOD_UNIT_CM;
    }
    if string.contains("mm") {
        model.units = IMOD_UNIT_MM;
    }
    if string.contains("um") {
        model.units = IMOD_UNIT_UM;
    }
    if string.contains("nm") {
        model.units = IMOD_UNIT_NM;
    }
    if string.contains('A') {
        model.units = IMOD_UNIT_ANGSTROM;
    }
    if string.contains("pm") {
        model.units = IMOD_UNIT_PM;
    }
}

/// `scaleModelRes` (`model_edit.cpp:106`).
pub fn scale_model_res(res: i32, zoom: f32) -> f32 {
    let zoom_min = 0.75;
    let zoom_max = 1.5;
    let mut ret = res as f32;
    if zoom < zoom_min {
        ret *= zoom_min / zoom;
    }
    if zoom > zoom_max {
        ret *= zoom_max / zoom;
    }
    ret
}

/// `ModelOffsetWindow` (`model_edit.h`), including the dialog's labels and edits.
#[derive(Clone, Debug)]
pub struct ModelOffsetWindow {
    pub m_base_label: [String; 3],
    pub m_edit_box: [String; 3],
    pub m_applied_label: String,
    pub rounded_style: bool,
    pub closed: bool,
}

impl Default for ModelOffsetWindow {
    fn default() -> Self {
        Self {
            m_base_label: std::array::from_fn(|_| String::new()),
            m_edit_box: std::array::from_fn(|_| String::new()),
            m_applied_label: String::new(),
            rounded_style: false,
            closed: false,
        }
    }
}

/// Rust state for static `OffsetDialog` in `model_edit.cpp`.
#[derive(Clone, Debug, Default)]
pub struct ModelOffsetData {
    pub dia: Option<ModelOffsetWindow>,
    pub applied: Ipoint,
    pub base: Ipoint,
}

/// `openModelOffset` (`model_edit.cpp:307`).
pub fn open_model_offset(state: &mut ModelOffsetData, n: &mut dyn ModelEditNativeBoundary) -> i32 {
    if state.dia.as_ref().is_some_and(|dia| !dia.closed) {
        n.raise_model_offset();
        return 0;
    }
    state.base = Ipoint::default();
    state.applied = state.base;
    state.dia = Some(ModelOffsetWindow::new(state));
    n.show_model_offset();
    0
}

/// `imodTransXYZ` (`model_edit.cpp:333`).
pub fn imod_trans_xyz(
    model: &mut Imod,
    trans: Ipoint,
    offset: Option<&mut ModelOffsetData>,
    n: &mut dyn ModelEditNativeBoundary,
) {
    n.cleanup_vertex_buffers(model);
    for obj in &mut model.obj {
        for cont in &mut obj.cont {
            for point in &mut cont.pts {
                point.x += trans.x;
                point.y += trans.y;
                point.z += trans.z;
            }
        }
        for mesh in &mut obj.mesh {
            for vertex in mesh.vert.iter_mut().step_by(2) {
                vertex.x += trans.x;
                vertex.y += trans.y;
                vertex.z += trans.z;
            }
        }
    }
    if let Some(offset) = offset {
        // `offset` is supplied only for the source's live OffsetDialog.  The
        // Rust caller temporarily owns its window while dispatching a slot.
        offset.applied.x += trans.x;
        offset.applied.y += trans.y;
        offset.applied.z += trans.z;
        if offset.dia.is_some() {
            let mut dia = offset.dia.take().expect("checked above");
            dia.update_labels(offset);
            offset.dia = Some(dia);
        }
    }
}

impl ModelOffsetWindow {
    /// `ModelOffsetWindow::ModelOffsetWindow`.
    pub fn new(state: &ModelOffsetData) -> Self {
        let mut window = Self::default();
        window.update_labels(state);
        window
    }

    /// `ModelOffsetWindow::~ModelOffsetWindow`.
    pub fn destroy(&mut self) {}

    /// `ModelOffsetWindow::buttonPressed`.
    pub fn button_pressed(
        &mut self,
        which: i32,
        state: &mut ModelOffsetData,
        model: &mut Imod,
        n: &mut dyn ModelEditNativeBoundary,
    ) {
        match which {
            0 => {
                let mut offset = Ipoint {
                    x: self.m_edit_box[0].parse().unwrap_or(0.),
                    y: self.m_edit_box[1].parse().unwrap_or(0.),
                    z: self.m_edit_box[2].parse().unwrap_or(0.),
                };
                offset.x += state.base.x - state.applied.x;
                offset.y += state.base.y - state.applied.y;
                offset.z += state.base.z - state.applied.z;
                n.model_shift(&offset);
                imod_trans_xyz(model, offset, Some(state), n);
                n.finish_undo_unit();
                n.draw_model();
                self.update_labels(state);
            }
            1 => {
                let offset = Ipoint {
                    x: -state.applied.x,
                    y: -state.applied.y,
                    z: -state.applied.z,
                };
                n.model_shift(&offset);
                imod_trans_xyz(model, offset, Some(state), n);
                n.finish_undo_unit();
                state.base = Ipoint::default();
                state.applied = state.base;
                n.draw_model();
                self.update_labels(state);
            }
            2 => {
                state.base = state.applied;
                self.m_edit_box = std::array::from_fn(|_| String::new());
                self.update_labels(state);
            }
            _ => {}
        }
    }

    /// `ModelOffsetWindow::valueEntered`.
    pub fn value_entered(
        &mut self,
        state: &mut ModelOffsetData,
        model: &mut Imod,
        n: &mut dyn ModelEditNativeBoundary,
    ) {
        self.button_pressed(0, state, model, n);
    }

    /// `ModelOffsetWindow::updateLabels`.
    pub fn update_labels(&mut self, state: &ModelOffsetData) {
        self.m_base_label[0] = format!("+{:9.2} base offset", state.base.x);
        self.m_base_label[1] = format!("+{:9.2} base offset", state.base.y);
        self.m_base_label[2] = format!("+{:9.2} base offset", state.base.z);
        self.m_applied_label = format!(
            "{:9.2},{:9.2},{:9.2}",
            state.applied.x, state.applied.y, state.applied.z
        );
    }

    /// `ModelOffsetWindow::topChangeEvent`.
    pub fn top_change_event(&mut self, n: &mut dyn ModelEditNativeBoundary) {
        self.rounded_style = n.rounded_style();
        n.dialog_change_event();
        n.check_and_set_mac_menu();
    }

    /// `ModelOffsetWindow::topCloseEvent`.
    pub fn top_close_event(&mut self, n: &mut dyn ModelEditNativeBoundary) {
        n.remove_model_offset();
        self.closed = true;
    }

    /// `ModelOffsetWindow::keyPressEvent`.
    pub fn key_press_event(&mut self, close_key: bool, n: &mut dyn ModelEditNativeBoundary) {
        if close_key {
            n.close_dialog();
        } else {
            n.control_key(false);
        }
    }

    /// `ModelOffsetWindow::keyReleaseEvent`.
    pub fn key_release_event(&mut self, n: &mut dyn ModelEditNativeBoundary) {
        n.control_key(true);
    }
}

/// `imodModelEditNewModel` (`model_edit.cpp:530`).
pub fn imod_model_edit_new_model(
    header: &mut ModelEditData,
    offset: &mut ModelOffsetData,
    model: &Imod,
) {
    imod_model_edit_update(header, model);
    if offset.dia.as_ref().is_none_or(|dia| dia.closed) {
        return;
    }
    offset.base = Ipoint::default();
    offset.applied = offset.base;
    let mut dia = offset.dia.take().expect("checked above");
    // Source calls buttonPressed(2); the state transition is equivalent and
    // has no native calls in that button arm.
    dia.m_edit_box = std::array::from_fn(|_| String::new());
    dia.update_labels(offset);
    offset.dia = Some(dia);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Imesh, Iobj};

    #[derive(Default)]
    struct Native {
        calls: Vec<String>,
    }
    impl ModelEditNativeBoundary for Native {
        fn raise_model_header(&mut self) {
            self.calls.push("raise-header".into());
        }
        fn show_model_header(&mut self) {
            self.calls.push("show-header".into());
        }
        fn remove_model_header(&mut self) {
            self.calls.push("remove-header".into());
        }
        fn raise_model_offset(&mut self) {
            self.calls.push("raise-offset".into());
        }
        fn show_model_offset(&mut self) {
            self.calls.push("show-offset".into());
        }
        fn remove_model_offset(&mut self) {
            self.calls.push("remove-offset".into());
        }
        fn model_change(&mut self) {
            self.calls.push("change".into());
        }
        fn model_shift(&mut self, _: &Ipoint) {
            self.calls.push("shift".into());
        }
        fn finish_undo_unit(&mut self) {
            self.calls.push("finish".into());
        }
        fn pixel_changed(&mut self) {
            self.calls.push("pixel".into());
        }
        fn set_model_scales_from_image(&mut self, _: &mut Imod, _: bool) {}
        fn draw_model(&mut self) {
            self.calls.push("draw".into());
        }
        fn cleanup_vertex_buffers(&mut self, _: &Imod) {
            self.calls.push("cleanup".into());
        }
        fn rounded_style(&mut self) -> bool {
            true
        }
        fn dialog_change_event(&mut self) {
            self.calls.push("change-event".into());
        }
        fn check_and_set_mac_menu(&mut self) {
            self.calls.push("mac-menu".into());
        }
        fn close_dialog(&mut self) {
            self.calls.push("close".into());
        }
        fn control_key(&mut self, release: bool) {
            self.calls
                .push(if release { "release" } else { "press" }.into());
        }
    }

    #[test]
    fn pixel_units_follow_source_substring_order() {
        let mut model = Imod::default();
        set_pixsize_and_units(&mut model, "2.5 nm");
        assert_eq!(model.pixsize, 2.5);
        assert_eq!(model.units, IMOD_UNIT_NM);
        set_pixsize_and_units(&mut model, "3 km");
        assert_eq!(model.units, IMOD_UNIT_METER);
    }

    #[test]
    fn both_dialogs_route_source_change_and_key_events() {
        let model = Imod::default();
        let mut native = Native::default();
        let mut header = ModelHeaderWindow::new(true, 0., 0., &model);
        header.top_change_event(&mut native);
        header.key_press_event(false, &mut native);
        header.key_release_event(&mut native);
        let mut offset = ModelOffsetWindow::default();
        offset.top_change_event(&mut native);
        offset.key_press_event(true, &mut native);
        assert!(header.rounded_style && offset.rounded_style);
        assert_eq!(
            native.calls,
            [
                "change-event",
                "mac-menu",
                "press",
                "release",
                "change-event",
                "mac-menu",
                "close"
            ]
        );
    }

    #[test]
    fn translation_moves_contours_and_even_mesh_vertices() {
        let mut model = Imod::default();
        let mut obj = Iobj::default();
        obj.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint {
                x: 1.,
                y: 2.,
                z: 3.,
            }],
            ..Default::default()
        });
        obj.mesh.push(Imesh {
            vert: vec![
                Ipoint::default(),
                Ipoint::default(),
                Ipoint {
                    x: 4.,
                    y: 5.,
                    z: 6.,
                },
            ],
            ..Default::default()
        });
        model.obj.push(obj);
        let mut native = Native::default();
        imod_trans_xyz(
            &mut model,
            Ipoint {
                x: 1.,
                y: 2.,
                z: 3.,
            },
            None,
            &mut native,
        );
        assert_eq!(
            model.obj[0].cont[0].pts[0],
            Ipoint {
                x: 2.,
                y: 4.,
                z: 6.
            }
        );
        assert_eq!(
            model.obj[0].mesh[0].vert[0],
            Ipoint {
                x: 1.,
                y: 2.,
                z: 3.
            }
        );
        assert_eq!(model.obj[0].mesh[0].vert[1], Ipoint::default());
        assert_eq!(
            model.obj[0].mesh[0].vert[2],
            Ipoint {
                x: 5.,
                y: 7.,
                z: 9.
            }
        );
    }

    #[test]
    fn offset_apply_then_revert_preserves_source_undo_order() {
        let mut model = Imod::default();
        let mut obj = Iobj::default();
        obj.cont.push(crate::imod::libimod::imodel::Icont {
            pts: vec![Ipoint::default()],
            ..Default::default()
        });
        model.obj.push(obj);
        let mut state = ModelOffsetData::default();
        let mut native = Native::default();
        open_model_offset(&mut state, &mut native);
        let mut dia = state.dia.take().unwrap();
        dia.m_edit_box = ["2".into(), "3".into(), "4".into()];
        dia.button_pressed(0, &mut state, &mut model, &mut native);
        assert_eq!(
            state.applied,
            Ipoint {
                x: 2.,
                y: 3.,
                z: 4.
            }
        );
        dia.button_pressed(1, &mut state, &mut model, &mut native);
        assert_eq!(model.obj[0].cont[0].pts[0], Ipoint::default());
        assert_eq!(state.applied, Ipoint::default());
    }
}
