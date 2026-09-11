//! Translation of `IMOD/3dmod/form_object_edit.cpp` and `form_object_edit.h`.
//!
//! `ObjectEditForm` retains the generated Qt form's widget state.  The
//! `ObjectEditFormNativeBoundary` calls are the direct Qt docking, key-routing,
//! and preferences boundary from the source unit.
#![allow(dead_code)]

use crate::imod::libimod::imodel::Imod;
use crate::imod::three_dmod::object_edit::{
    ObjectEdit, ioew_arrow, ioew_closing, ioew_copy_obj, ioew_draw, ioew_draw_labels, ioew_ends,
    ioew_fill, ioew_fill_trans, ioew_get_copy_color_name, ioew_label_size, ioew_linewidth,
    ioew_nametext, ioew_open, ioew_outline, ioew_planar, ioew_point_limit, ioew_pointsize,
    ioew_quit, ioew_scale_for_dpi, ioew_set_copy_color_name, ioew_sphere_on_sec, ioew_surface,
    ioew_symbol, ioew_symsize, ioew_time,
};

/// Qt `DockingDialog`, keyboard routing, and `ImodPrefs` source boundary.
pub trait ObjectEditFormNativeBoundary {
    fn setup_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_signals(&mut self);
    fn set_focus(&mut self);
    fn set_default_obj_props(&mut self);
    fn restore_default_obj_props(&mut self);
    fn close_key(&self) -> bool;
    fn ivw_control_key(&mut self, release: bool);
    fn accept_close(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn is_font_change(&self) -> bool;
    fn retranslate_ui(&mut self);
}

/// `objectEditForm`, including the value and visibility state owned by its `.ui` form.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectEditForm {
    pub object_name: String,
    pub object_num_text: String,
    pub draw: bool,
    pub copy_object: i32,
    pub copy_object_maximum: i32,
    pub copy_object_enabled: bool,
    pub copy_color: bool,
    pub copy_name: bool,
    pub object_type: i32,
    pub front_surface: i32,
    pub symbol: i32,
    pub fill: bool,
    pub mark_ends: bool,
    pub arrow_at_end: bool,
    pub symbol_size: i32,
    pub symbol_size_text: String,
    pub draw_labels: bool,
    pub label_size: i32,
    pub time: bool,
    pub time_enabled: bool,
    pub point_radius: i32,
    pub on_section: bool,
    pub planar: bool,
    pub planar_enabled: bool,
    pub point_limit: i32,
    pub line_width: i32,
    pub scale_for_dpi: bool,
    pub scale_for_dpi_visible: bool,
    pub fill_trans: i32,
    pub fill_trans_text: String,
    pub outline: bool,
    pub fill_controls_enabled: bool,
}

impl ObjectEditForm {
    /// `objectEditForm::objectEditForm`.
    pub fn new(edit: &ObjectEdit, native: &mut dyn ObjectEditFormNativeBoundary) -> Self {
        native.setup_ui();
        native.set_delete_on_close();
        native.set_always_show_tool_tips();
        native.connect_signals();
        Self {
            copy_color: ioew_get_copy_color_name(edit) & 1 != 0,
            copy_name: ioew_get_copy_color_name(edit) & 2 != 0,
            copy_object: 1,
            copy_object_maximum: 1,
            copy_object_enabled: false,
            ..Default::default()
        }
    }
    /// `objectEditForm::~objectEditForm`.
    pub fn destroy(&mut self) {}
    /// `objectEditForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn ObjectEditFormNativeBoundary) {
        native.retranslate_ui()
    }
    /// `objectEditForm::nameChanged`.
    pub fn name_changed(&mut self, model: &mut Imod, edit: &mut ObjectEdit, new_name: &str) {
        self.object_name = new_name.into();
        ioew_nametext(model, edit, new_name)
    }
    /// `objectEditForm::symbolChanged`.
    pub fn symbol_changed(&mut self, model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
        self.symbol = value;
        ioew_symbol(model, edit, value)
    }
    /// `objectEditForm::OKPressed`.
    pub fn ok_pressed(&mut self, edit: &mut ObjectEdit) {
        ioew_quit(edit)
    }
    /// `objectEditForm::radiusChanged`.
    pub fn radius_changed(
        &mut self,
        model: &mut Imod,
        edit: &mut ObjectEdit,
        value: i32,
        native: &mut dyn ObjectEditFormNativeBoundary,
    ) {
        self.point_radius = value;
        ioew_pointsize(model, edit, value);
        native.set_focus()
    }
    /// `objectEditForm::ptLimitChanged`.
    pub fn pt_limit_changed(&mut self, model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
        self.point_limit = value;
        ioew_point_limit(model, edit, value)
    }
    /// `objectEditForm::selectedSurface`.
    pub fn selected_surface(&mut self, model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
        self.front_surface = value;
        ioew_surface(model, edit, value)
    }
    /// `objectEditForm::selectedType`.
    pub fn selected_type(&mut self, model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
        self.object_type = value;
        ioew_open(model, edit, value)
    }
    /// `objectEditForm::sizeChanged`.
    pub fn size_changed(&mut self, model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
        self.symbol_size = value;
        self.symbol_size_text = value.to_string();
        ioew_symsize(model, edit, value)
    }
    /// `objectEditForm::toggledDraw`.
    pub fn toggled_draw(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.draw = state;
        ioew_draw(model, edit, state as i32)
    }
    /// `objectEditForm::toggledDrawLabels`.
    pub fn toggled_draw_labels(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.draw_labels = state;
        ioew_draw_labels(model, edit, state as i32)
    }
    /// `objectEditForm::toggledScaleForDpi`.
    pub fn toggled_scale_for_dpi(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.scale_for_dpi = state;
        ioew_scale_for_dpi(model, edit, state as i32)
    }
    /// `objectEditForm::labelSizeChanged`.
    pub fn label_size_changed(
        &mut self,
        model: &mut Imod,
        edit: &mut ObjectEdit,
        value: i32,
        native: &mut dyn ObjectEditFormNativeBoundary,
    ) {
        self.label_size = value;
        ioew_label_size(model, edit, value);
        native.set_focus()
    }
    /// `objectEditForm::toggledFill`.
    pub fn toggled_fill(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.fill = state;
        ioew_fill(model, edit, state as i32)
    }
    /// `objectEditForm::toggledMarkEnds`.
    pub fn toggled_mark_ends(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.mark_ends = state;
        ioew_ends(model, edit, state as i32)
    }
    /// `objectEditForm::toggledArrowAtEnd`.
    pub fn toggled_arrow_at_end(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.arrow_at_end = state;
        ioew_arrow(model, edit, state as i32)
    }
    /// `objectEditForm::toggledTime`.
    pub fn toggled_time(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.time = state;
        ioew_time(model, edit, state as i32)
    }
    /// `objectEditForm::toggledOnSection`.
    pub fn toggled_on_section(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.on_section = state;
        ioew_sphere_on_sec(model, edit, state as i32)
    }
    /// `objectEditForm::widthChanged`.
    pub fn width_changed(
        &mut self,
        model: &mut Imod,
        edit: &mut ObjectEdit,
        value: i32,
        native: &mut dyn ObjectEditFormNativeBoundary,
    ) {
        self.line_width = value;
        ioew_linewidth(model, edit, value);
        native.set_focus()
    }
    /// `objectEditForm::toggledPlanar`.
    pub fn toggled_planar(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.planar = state;
        ioew_planar(model, edit, state as i32)
    }
    /// `objectEditForm::transChanged`.
    pub fn trans_changed(&mut self, model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
        self.fill_trans = value;
        self.fill_trans_text = value.to_string();
        ioew_fill_trans(model, edit, value)
    }
    /// `objectEditForm::toggledOutline`.
    pub fn toggled_outline(&mut self, model: &mut Imod, edit: &mut ObjectEdit, state: bool) {
        self.outline = state;
        ioew_outline(model, edit, state as i32)
    }
    /// `objectEditForm::toggledCopyColor`.
    pub fn toggled_copy_color(&mut self, edit: &mut ObjectEdit, state: bool) {
        self.copy_color = state;
        let mut current = ioew_get_copy_color_name(edit);
        current = if state { current | 1 } else { current & !1 };
        ioew_set_copy_color_name(edit, current)
    }
    /// `objectEditForm::toggledCopyName`.
    pub fn toggled_copy_name(&mut self, edit: &mut ObjectEdit, state: bool) {
        self.copy_name = state;
        let mut current = ioew_get_copy_color_name(edit);
        current = if state { current | 2 } else { current & !2 };
        ioew_set_copy_color_name(edit, current)
    }
    /// `objectEditForm::copyClicked`.
    pub fn copy_clicked(
        &mut self,
        model: &mut Imod,
        edit: &mut ObjectEdit,
        native: &mut dyn ObjectEditFormNativeBoundary,
    ) -> Result<(), String> {
        let result = ioew_copy_obj(model, edit, self.copy_object);
        native.set_focus();
        result
    }
    /// `objectEditForm::setDefaultsClicked`.
    pub fn set_defaults_clicked(&mut self, native: &mut dyn ObjectEditFormNativeBoundary) {
        native.set_default_obj_props()
    }
    /// `objectEditForm::restoreClicked`.
    pub fn restore_clicked(&mut self, native: &mut dyn ObjectEditFormNativeBoundary) {
        native.restore_default_obj_props()
    }
    /// `objectEditForm::setSymbolProperties`.
    pub fn set_symbol_properties(
        &mut self,
        which: i32,
        fill: bool,
        mark_ends: bool,
        arrow_at_end: bool,
        size: i32,
    ) {
        self.symbol = which;
        self.fill = fill;
        self.mark_ends = mark_ends;
        self.arrow_at_end = arrow_at_end;
        self.symbol_size = size;
        self.symbol_size_text = size.to_string()
    }
    /// `objectEditForm::setCopyObjLimit`.
    pub fn set_copy_obj_limit(&mut self, value: i32) {
        self.copy_object = self.copy_object.min(value);
        self.copy_object_maximum = value;
        self.copy_object_enabled = value > 1
    }
    /// `objectEditForm::setDrawBox`.
    pub fn set_draw_box(&mut self, state: bool) {
        self.draw = state
    }
    /// `objectEditForm::setDrawLabelsBox`.
    pub fn set_draw_labels_box(&mut self, state: bool) {
        self.draw_labels = state
    }
    /// `objectEditForm::setLabelSize`.
    pub fn set_label_size(&mut self, value: i32) {
        self.label_size = value
    }
    /// `objectEditForm::setObjectName`.
    pub fn set_object_name(&mut self, name: &str) {
        self.object_name = name.into()
    }
    /// `objectEditForm::setObjectNum`.
    pub fn set_object_num(&mut self, num: i32) {
        self.object_num_text = format!("# {}", num + 1)
    }
    /// `objectEditForm::setTimeBox`.
    pub fn set_time_box(&mut self, state: bool, enabled: bool) {
        self.time = state;
        self.time_enabled = enabled
    }
    /// `objectEditForm::setOnSecBox`.
    pub fn set_on_sec_box(&mut self, state: bool) {
        self.on_section = state
    }
    /// `objectEditForm::setPointRadius`.
    pub fn set_point_radius(&mut self, value: i32) {
        self.point_radius = value
    }
    /// `objectEditForm::setFrontSurface`.
    pub fn set_front_surface(&mut self, value: i32) {
        self.front_surface = value
    }
    /// `objectEditForm::setObjectType`.
    pub fn set_object_type(&mut self, value: i32) {
        self.object_type = value
    }
    /// `objectEditForm::setLineWidth`.
    pub fn set_line_width(&mut self, value: i32) {
        self.line_width = value
    }
    /// `objectEditForm::setScaleForDpi`.
    pub fn set_scale_for_dpi(&mut self, state: bool, show: bool) {
        self.scale_for_dpi = state;
        self.scale_for_dpi_visible = show
    }
    /// `objectEditForm::setPlanarBox`.
    pub fn set_planar_box(&mut self, state: bool, enabled: bool) {
        self.planar = state;
        self.planar_enabled = enabled
    }
    /// `objectEditForm::setPointLimit`.
    pub fn set_point_limit(&mut self, value: i32) {
        self.point_limit = value
    }
    /// `objectEditForm::setFillTrans`.
    pub fn set_fill_trans(&mut self, value: i32, state: bool, enabled: bool) {
        self.fill_trans = value;
        self.fill_trans_text = value.to_string();
        self.outline = state;
        self.fill_controls_enabled = enabled
    }
    /// `objectEditForm::topCloseEvent`.
    pub fn top_close_event(
        &mut self,
        edit: &mut ObjectEdit,
        native: &mut dyn ObjectEditFormNativeBoundary,
    ) {
        ioew_closing(edit);
        native.accept_close()
    }
    /// `objectEditForm::keyPressEvent`.
    pub fn key_press_event(
        &mut self,
        edit: &mut ObjectEdit,
        native: &mut dyn ObjectEditFormNativeBoundary,
    ) {
        if native.close_key() {
            ioew_quit(edit)
        } else {
            native.ivw_control_key(false)
        }
    }
    /// `objectEditForm::keyReleaseEvent`.
    pub fn key_release_event(&mut self, native: &mut dyn ObjectEditFormNativeBoundary) {
        native.ivw_control_key(true)
    }
    /// `objectEditForm::topChangeEvent`.
    pub fn top_change_event(&mut self, native: &mut dyn ObjectEditFormNativeBoundary) {
        native.check_and_set_mac_menu();
        if !native.is_font_change() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_SCAT, Iobj};
    use crate::imod::libimod::iobj::IOBJ_SYMF_ARROW;
    #[derive(Default)]
    struct Native {
        focus: usize,
        accepted: bool,
        defaults: usize,
        restored: usize,
    }
    impl ObjectEditFormNativeBoundary for Native {
        fn setup_ui(&mut self) {}
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_signals(&mut self) {}
        fn set_focus(&mut self) {
            self.focus += 1
        }
        fn set_default_obj_props(&mut self) {
            self.defaults += 1
        }
        fn restore_default_obj_props(&mut self) {
            self.restored += 1
        }
        fn close_key(&self) -> bool {
            false
        }
        fn ivw_control_key(&mut self, _: bool) {}
        fn accept_close(&mut self) {
            self.accepted = true
        }
        fn check_and_set_mac_menu(&mut self) {}
        fn is_font_change(&self) -> bool {
            false
        }
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn source_slots_update_selected_object_and_form_state() {
        let mut model = Imod::default();
        model.obj.push(Iobj::default());
        let mut edit = ObjectEdit {
            current_object: 0,
            ..Default::default()
        };
        let mut n = Native::default();
        let mut form = ObjectEditForm::new(&edit, &mut n);
        form.selected_type(&mut model, &mut edit, 2);
        form.toggled_arrow_at_end(&mut model, &mut edit, true);
        form.size_changed(&mut model, &mut edit, 9);
        assert_eq!(
            model.obj[0].flags & (IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT),
            IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT
        );
        assert_ne!(model.obj[0].symflags & IOBJ_SYMF_ARROW as u8, 0);
        assert_eq!(
            (form.symbol_size_text.as_str(), model.obj[0].symsize),
            ("9", 9)
        );
    }
    #[test]
    fn copy_checkbox_bits_and_setters_follow_source() {
        let mut e = ObjectEdit::default();
        let mut n = Native::default();
        let mut f = ObjectEditForm::new(&e, &mut n);
        f.toggled_copy_color(&mut e, true);
        f.toggled_copy_name(&mut e, true);
        f.set_copy_obj_limit(4);
        f.set_fill_trans(75, true, false);
        assert_eq!(ioew_get_copy_color_name(&e), 3);
        assert_eq!(
            (
                f.copy_object_maximum,
                f.copy_object_enabled,
                f.fill_trans_text.as_str(),
                f.fill_controls_enabled
            ),
            (4, true, "75", false)
        );
    }
}
