//! `IMOD/Etomo/src/etomo/ui/swing/TransformChooserPanel.java`.
//!
//! The "Search For:" radio buttons choosing an alignment transform (full linear,
//! rotation/translation/magnification, rotation/translation and, for the join model,
//! translation), with an optional "Search For:" check box (Serial Sections).  An event
//! dispatch thread object, created as `Rc<Self>`; the listener class
//! `TransformChooserListener` is a closure holding a weak reference to the panel.

use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::radio_button::RadioButton;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::r#type::transform::Transform;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `SEARCH_LABEL`.
const SEARCH_LABEL: &str = "Search For:";

/// Java package-private `final class TransformChooserPanel`.
pub struct TransformChooserPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `bgTransform = new ButtonGroup()` (the buttons hold it too).
    #[allow(dead_code)]
    bg_transform: Rc<ButtonGroup>,
    /// Java private final `rbFullLinearTransformation`.
    rb_full_linear_transformation: Rc<RadioButton>,
    /// Java private final `rbRotationTranslationMagnification`.
    rb_rotation_translation_magnification: Rc<RadioButton>,
    /// Java private final `rbRotationTranslation`.
    rb_rotation_translation: Rc<RadioButton>,
    /// Java private final `rbTranslation`; null unless translations alone are
    /// allowed.
    rb_translation: Option<Rc<RadioButton>>,
    /// Java private final `cbSearch`; null unless the search is optional.
    cb_search: Option<Rc<CheckBox>>,
    /// Rust-only: Java `this` for the listener.
    self_ref: Weak<TransformChooserPanel>,
}

impl TransformChooserPanel {
    /// Java private `TransformChooserPanel(boolean, boolean)`.
    fn new(allow_translations_alone: bool, optional_search: bool) -> Rc<TransformChooserPanel> {
        Rc::new_cyclic(|self_ref| {
            let bg_transform = ButtonGroup::new();
            let rb_full_linear_transformation = RadioButton::new_string_button_group(
                Some("Full linear transformation"),
                Some(&bg_transform),
            );
            let rb_rotation_translation_magnification = RadioButton::new_string_button_group(
                Some("Rotation/translation/magnification"),
                Some(&bg_transform),
            );
            let rb_rotation_translation = RadioButton::new_string_button_group(
                Some("Rotation/translation"),
                Some(&bg_transform),
            );
            let rb_translation = if allow_translations_alone {
                Some(RadioButton::new_string_button_group(
                    Some("Translation"),
                    Some(&bg_transform),
                ))
            } else {
                None
            };
            let cb_search = if optional_search {
                Some(CheckBox::new_string(Some(SEARCH_LABEL)))
            } else {
                None
            };
            TransformChooserPanel {
                pnl_root: JComponent::new_panel(),
                bg_transform,
                rb_full_linear_transformation,
                rb_rotation_translation_magnification,
                rb_rotation_translation,
                rb_translation,
                cb_search,
                self_ref: self_ref.clone(),
            }
        })
    }

    /// Java static `getJoinModelInstance()`.
    pub fn get_join_model_instance() -> Rc<TransformChooserPanel> {
        let instance = TransformChooserPanel::new(true, false);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java static `getJoinAlignInstance()`.
    pub fn get_join_align_instance() -> Rc<TransformChooserPanel> {
        let instance = TransformChooserPanel::new(false, false);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java static `getSerialSectionsInstance()`.
    pub fn get_serial_sections_instance() -> Rc<TransformChooserPanel> {
        let instance = TransformChooserPanel::new(false, true);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.set_transform(None);
        // root panel
        // Swing layout: pnlRoot BoxLayout X_AXIS.
        let pnl_chooser = JComponent::new_panel();
        self.pnl_root.add(&pnl_chooser);
        // Swing layout: Box.createHorizontalGlue().
        // choooser panel
        // Swing layout: pnlChooser BoxLayout Y_AXIS, CENTER_ALIGNMENT.
        match &self.cb_search {
            Some(cb_search) => pnl_chooser.add(&cb_search.get_component()),
            None => pnl_chooser.add(&JComponent::new_label(SEARCH_LABEL)),
        }
        pnl_chooser.add(&self.rb_full_linear_transformation.get_component());
        pnl_chooser.add(&self.rb_rotation_translation_magnification.get_component());
        pnl_chooser.add(&self.rb_rotation_translation.get_component());
        if let Some(rb_translation) = &self.rb_translation {
            pnl_chooser.add(&rb_translation.get_component());
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        if let Some(cb_search) = &self.cb_search {
            // Java `new TransformChooserListener(this)`.
            let panel = self.self_ref.clone();
            let listener: ActionListener = Rc::new(move |_event: &ActionEvent| {
                if let Some(panel) = panel.upgrade() {
                    panel.action();
                }
            });
            cb_search.add_action_listener(Some(listener));
        }
    }

    /// Java package-private `getSearchActionCommand()`.
    pub fn get_search_action_command(&self) -> Option<String> {
        if let Some(cb_search) = &self.cb_search {
            return cb_search.get_action_command();
        }
        None
    }

    /// Java package-private `addSearchListener(ActionListener)`.
    ///
    /// Fixed in translation: the source dereferences `cbSearch`, which is null for
    /// the join instances (NullPointerException); its only caller passes a join
    /// instance only when `joinInterface` is false, and a null check box adds nothing
    /// here.
    pub fn add_search_listener(&self, listener: ActionListener) {
        if let Some(cb_search) = &self.cb_search {
            cb_search.add_action_listener(Some(listener));
        }
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `action()`.
    fn action(&self) {
        self.update_display();
    }

    /// Java private `updateDisplay()`.  Only reached with a search check box (from
    /// its listener and from `setTransform`).
    fn update_display(&self) {
        let Some(cb_search) = &self.cb_search else {
            return;
        };
        let enable = cb_search.is_selected();
        self.rb_full_linear_transformation.set_enabled(enable);
        self.rb_rotation_translation_magnification
            .set_enabled(enable);
        self.rb_rotation_translation.set_enabled(enable);
        if let Some(rb_translation) = &self.rb_translation {
            rb_translation.set_enabled(enable);
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        if let Some(cb_search) = &self.cb_search {
            cb_search.set_tool_tip_text_string(Some(
                "Use iterative search to find best transformation for aligning images.",
            ));
        }
        self.rb_full_linear_transformation
            .set_tool_tip_text_string(Some(
                "Use rotation, translation, magnification, and stretching to align images.",
            ));
        self.rb_rotation_translation_magnification
            .set_tool_tip_text_string(Some(
                "Use translation, rotation, and magnification to align images.",
            ));
        self.rb_rotation_translation
            .set_tool_tip_text_string(Some("Use translation and rotation to align images."));
        if let Some(rb_translation) = &self.rb_translation {
            rb_translation.set_tool_tip_text_string(Some("Use translation to align images."));
        }
    }

    /// Java package-private `isSearch()`.
    pub fn is_search(&self) -> bool {
        match &self.cb_search {
            // Search is always on
            None => true,
            Some(cb_search) => cb_search.is_selected(),
        }
    }

    /// Java package-private `getTransform()`.
    ///
    /// Fixed in translation: with no radio button selected the source dereferences
    /// `rbTranslation`, which is null for the align instances
    /// (NullPointerException); a missing button is not selected here.
    pub fn get_transform(&self) -> Transform {
        if let Some(cb_search) = &self.cb_search
            && !cb_search.is_selected()
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
            .is_some_and(|rb_translation| rb_translation.is_selected())
        {
            return Transform::Translation;
        }
        Transform::DEFAULT
    }

    /// Java package-private `setTransform(Transform)`; null is `None`.
    ///
    /// Fixed in translation: `Transform.TRANSLATION` on an instance without the
    /// translation button dereferences null in the source; nothing is selected here.
    pub fn set_transform(&self, transform: Option<Transform>) {
        if let Some(cb_search) = &self.cb_search {
            if transform == Some(Transform::SkipSearch) {
                cb_search.set_selected_boolean(false);
            } else {
                cb_search.set_selected_boolean(true);
            }
            self.update_display();
        }
        if transform != Some(Transform::SkipSearch) {
            let transform = transform.unwrap_or(Transform::DEFAULT);
            if transform == Transform::FullLinearTransformation {
                self.rb_full_linear_transformation
                    .set_selected_boolean(true);
            } else if transform == Transform::RotationTranslationMagnification {
                self.rb_rotation_translation_magnification
                    .set_selected_boolean(true);
            } else if transform == Transform::RotationTranslation {
                self.rb_rotation_translation.set_selected_boolean(true);
            } else if transform == Transform::Translation
                && let Some(rb_translation) = &self.rb_translation
            {
                rb_translation.set_selected_boolean(true);
            }
        }
    }
}
