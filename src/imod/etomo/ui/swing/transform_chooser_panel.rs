//! `IMOD/Etomo/src/etomo/ui/swing/TransformChooserPanel.java`.
//!
//! Swing layout, components, and action delivery remain an explicit GUI
//! boundary.  The source panel's transform state and enablement policy are
//! retained directly.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::r#type::transform::Transform;

use super::check_box::CheckBox;
use super::radio_button::{RadioButton, RadioButtonGroup};

const SEARCH_LABEL: &str = "Search For:";

/// Native Swing `JPanel`/`BoxLayout` construction boundary, retaining the
/// source-visible hierarchy and axis/alignment settings.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TransformChooserPanelBoundary {
    pub root_layout_x_axis: bool,
    pub chooser_layout_y_axis: bool,
    pub chooser_alignment_x_center: bool,
    pub root_components: Vec<String>,
    pub chooser_components: Vec<String>,
}

/// Java `TransformChooserPanel`.
pub struct TransformChooserPanel {
    pub pnl_root: TransformChooserPanelBoundary,
    pub bg_transform: Rc<RefCell<RadioButtonGroup>>,
    pub rb_full_linear_transformation: RadioButton,
    pub rb_rotation_translation_magnification: RadioButton,
    pub rb_rotation_translation: RadioButton,
    pub rb_translation: Option<RadioButton>,
    pub cb_search: Option<CheckBox>,
}

impl TransformChooserPanel {
    fn new(allow_translations_alone: bool, optional_search: bool) -> Self {
        let bg_transform = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root: TransformChooserPanelBoundary::default(),
            rb_full_linear_transformation: RadioButton::new_in_group(
                "Full linear transformation",
                bg_transform.clone(),
            ),
            rb_rotation_translation_magnification: RadioButton::new_in_group(
                "Rotation/translation/magnification",
                bg_transform.clone(),
            ),
            rb_rotation_translation: RadioButton::new_in_group(
                "Rotation/translation",
                bg_transform.clone(),
            ),
            rb_translation: allow_translations_alone
                .then(|| RadioButton::new_in_group("Translation", bg_transform.clone())),
            cb_search: optional_search.then(|| CheckBox::new_with_text(SEARCH_LABEL)),
            bg_transform,
        }
    }

    pub fn get_join_model_instance() -> Self {
        let mut instance = Self::new(true, false);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    pub fn get_join_align_instance() -> Self {
        let mut instance = Self::new(false, false);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    pub fn get_serial_sections_instance() -> Self {
        let mut instance = Self::new(false, true);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    fn create_panel(&mut self) {
        self.set_transform(None);
        self.pnl_root.root_layout_x_axis = true;
        self.pnl_root.root_components = vec!["pnlChooser".into(), "horizontalGlue".into()];
        self.pnl_root.chooser_layout_y_axis = true;
        self.pnl_root.chooser_alignment_x_center = true;
        if self.cb_search.is_some() {
            self.pnl_root.chooser_components.push("cbSearch".into());
        } else {
            self.pnl_root
                .chooser_components
                .push(format!("JLabel:{SEARCH_LABEL}"));
        }
        self.pnl_root
            .chooser_components
            .push("rbFullLinearTransformation".into());
        self.pnl_root
            .chooser_components
            .push("rbRotationTranslationMagnification".into());
        self.pnl_root
            .chooser_components
            .push("rbRotationTranslation".into());
        if self.rb_translation.is_some() {
            self.pnl_root
                .chooser_components
                .push("rbTranslation".into());
        }
    }

    fn add_listeners(&mut self) {
        if let Some(cb_search) = &mut self.cb_search {
            cb_search.add_action_listener();
        }
    }

    pub fn get_search_action_command(&self) -> Option<&str> {
        self.cb_search
            .as_ref()
            .and_then(CheckBox::get_action_command)
    }

    pub fn add_search_listener(&mut self) {
        self.cb_search
            .as_mut()
            .expect("Java TransformChooserPanel.addSearchListener null cbSearch")
            .add_action_listener();
    }

    pub fn get_component(&self) -> &TransformChooserPanelBoundary {
        &self.pnl_root
    }

    fn action(&mut self) {
        self.update_display();
    }

    fn update_display(&mut self) {
        let enable = self
            .cb_search
            .as_ref()
            .expect("Java TransformChooserPanel.updateDisplay null cbSearch")
            .is_selected();
        self.rb_full_linear_transformation.set_enabled(enable);
        self.rb_rotation_translation_magnification
            .set_enabled(enable);
        self.rb_rotation_translation.set_enabled(enable);
        if let Some(rb_translation) = &mut self.rb_translation {
            rb_translation.set_enabled(enable);
        }
    }

    fn set_tooltips(&mut self) {
        if let Some(cb_search) = &mut self.cb_search {
            cb_search.set_tool_tip_text(Some(
                "Use iterative search to find best transformation for aligning images.",
            ));
        }
        self.rb_full_linear_transformation.set_tool_tip_text(Some(
            "Use rotation, translation, magnification, and stretching to align images.",
        ));
        self.rb_rotation_translation_magnification
            .set_tool_tip_text(Some(
                "Use translation, rotation, and magnification to align images.",
            ));
        self.rb_rotation_translation
            .set_tool_tip_text(Some("Use translation and rotation to align images."));
        if let Some(rb_translation) = &mut self.rb_translation {
            rb_translation.set_tool_tip_text(Some("Use translation to align images."));
        }
    }

    pub fn is_search(&self) -> bool {
        self.cb_search.as_ref().is_none_or(CheckBox::is_selected)
    }

    pub fn get_transform(&self) -> Transform {
        if self
            .cb_search
            .as_ref()
            .is_some_and(|cb_search| !cb_search.is_selected())
        {
            return Transform::SkipSearch;
        }
        if self.rb_full_linear_transformation.is_selected() {
            return Transform::FullLinearTransformation;
        }
        if self.rb_rotation_translation_magnification.is_selected() {
            return Transform::RotationTranslationMagnification;
        }
        if self.rb_rotation_translation.is_selected() {
            return Transform::RotationTranslation;
        }
        if self
            .rb_translation
            .as_ref()
            .expect("Java TransformChooserPanel.getTransform null rbTranslation")
            .is_selected()
        {
            return Transform::Translation;
        }
        Transform::DEFAULT
    }

    pub fn set_transform(&mut self, mut transform: Option<Transform>) {
        if let Some(cb_search) = &mut self.cb_search {
            cb_search.set_selected(transform != Some(Transform::SkipSearch));
            self.update_display();
        }
        if transform != Some(Transform::SkipSearch) {
            let transform = transform.get_or_insert(Transform::DEFAULT);
            match *transform {
                Transform::FullLinearTransformation => {
                    self.rb_full_linear_transformation.set_selected(true)
                }
                Transform::RotationTranslationMagnification => self
                    .rb_rotation_translation_magnification
                    .set_selected(true),
                Transform::RotationTranslation => self.rb_rotation_translation.set_selected(true),
                Transform::Translation => self
                    .rb_translation
                    .as_mut()
                    .expect("Java TransformChooserPanel.setTransform null rbTranslation")
                    .set_selected(true),
                Transform::SkipSearch => {}
            }
        }
    }
}

/// Java private `TransformChooserListener`, whose Swing event delivery is the
/// GUI boundary and whose source action call remains direct.
struct TransformChooserListener;

impl TransformChooserListener {
    fn action_performed(&self, panel: &mut TransformChooserPanel) {
        panel.action();
    }
}

#[cfg(test)]
mod tests {
    use super::{Transform, TransformChooserListener, TransformChooserPanel};

    #[test]
    fn join_model_supports_all_source_transform_choices() {
        let mut panel = TransformChooserPanel::get_join_model_instance();
        assert!(panel.is_search());
        assert_eq!(panel.get_transform(), Transform::FullLinearTransformation);
        panel.set_transform(Some(Transform::Translation));
        assert_eq!(panel.get_transform(), Transform::Translation);
        assert_eq!(panel.get_component().chooser_components.len(), 5);
    }

    #[test]
    fn optional_search_disables_buttons_and_returns_skip_search() {
        let mut panel = TransformChooserPanel::get_serial_sections_instance();
        panel.set_transform(Some(Transform::SkipSearch));
        assert!(!panel.is_search());
        assert_eq!(panel.get_transform(), Transform::SkipSearch);
        assert!(!panel.rb_full_linear_transformation.is_enabled());
        assert!(!panel.rb_rotation_translation_magnification.is_enabled());
        assert!(!panel.rb_rotation_translation.is_enabled());
        panel.cb_search.as_mut().unwrap().set_selected(true);
        TransformChooserListener.action_performed(&mut panel);
        assert!(panel.rb_full_linear_transformation.is_enabled());
    }

    #[test]
    #[should_panic(expected = "null rbTranslation")]
    fn source_null_translation_access_is_retained() {
        let mut panel = TransformChooserPanel::get_join_align_instance();
        panel.set_transform(Some(Transform::Translation));
    }
}
