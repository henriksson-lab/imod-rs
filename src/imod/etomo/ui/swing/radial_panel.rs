//! `IMOD/Etomo/src/etomo/ui/swing/RadialPanel.java`.
//!
//! Java `final class RadialPanel implements ActionListener` (radial filter
//! panel of the tilt panels).  An EDT object: created as `Rc<Self>` by
//! [`RadialPanel::get_instance`]; every method takes `&self`.  The Java
//! `implements ActionListener` (`actionPerformed` -> `updateDisplay()`) is a
//! closure registered on the radio buttons, holding a weak reference to the
//! panel.  The parent (`RadialParent`, the owning tilt panel) is held weakly.
//!
//! The fields Java creates only for `PanelId.TILT` (and sets to null
//! otherwise) are `Option`s.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::etched_border::EtchedBorder;
use super::filter_type::FilterType;
use super::labeled_text_field::LabeledTextField;
use super::radial_parent::RadialParent;
use super::radio_button::RadioButton;
use super::text_field::TextField;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::sirtsetup_param::{self, SirtsetupParam};
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;

/// Java private static final `RADIAL_FALLOFF_OLD_LABEL`.
const RADIAL_FALLOFF_OLD_LABEL: &str = " Falloff (1.4 * sigma): ";
/// Java private static final `RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL`.
const RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL: &str = " Falloff (sigma): ";
/// Java private static final `FAKE_SIRT_ITERATIONS_LABEL`.
const FAKE_SIRT_ITERATIONS_LABEL: &str = "Use SIRT-like filter equivalent to: ";
/// Java private static final `EXACT_FILTER_SIZE_LABEL`.
const EXACT_FILTER_SIZE_LABEL: &str = "Use 'exact filter' functions with 'object size' of: ";
/// Java private static final `HAMMING_LIKE_FITLER_LABEL` (unused in the Java).
#[allow(dead_code)]
const HAMMING_LIKE_FITLER_LABEL: &str = "Hamming-like filter (as in tomo3d) starting from: ";
/// Java private static final `RADIAL_BUTTON_LABEL`.
const RADIAL_BUTTON_LABEL: &str = "Standard Gaussian,";
/// Java private static final `RADIAL_MAX_LABEL`.
const RADIAL_MAX_LABEL: &str = "cutoff: ";
/// Java private static final `RADIAL_LABEL = RADIAL_BUTTON_LABEL + " " +
/// RADIAL_MAX_LABEL`.
const RADIAL_LABEL: &str = "Standard Gaussian, cutoff: ";

/// Java `final class RadialPanel implements ActionListener`.
pub struct RadialPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `ltfRadialMax`.
    ltf_radial_max: Rc<LabeledTextField>,
    /// Java private final `ltfRadialFallOff`.
    ltf_radial_fall_off: Rc<LabeledTextField>,
    /// Java private final `pnlHighFrequencyFiltering = new JPanel()`.
    pnl_high_frequency_filtering: Rc<JComponent>,

    /// Java private final `tfHammingLikeFilter` (null unless `PanelId.TILT`).
    tf_hamming_like_filter: Option<Rc<TextField>>,
    /// Java private final `tfExactFilterSize` (null unless `PanelId.TILT`).
    tf_exact_filter_size: Option<Rc<TextField>>,
    /// Java private final `panelId`.
    panel_id: PanelId,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `rbDefault` (null unless `PanelId.TILT`).
    rb_default: Option<Rc<RadioButton>>,
    /// Java private final `tfFakeSIRTiterations` (null unless `PanelId.TILT`).
    tf_fake_sirt_iterations: Option<Rc<TextField>>,
    /// Java private final `rbFakeSIRTiterations` (null unless `PanelId.TILT`).
    rb_fake_sirt_iterations: Option<Rc<RadioButton>>,
    /// Java private final `rbExactFilterSize` (null unless `PanelId.TILT`).
    rb_exact_filter_size: Option<Rc<RadioButton>>,
    /// Java private final `rbRadial` (null unless `PanelId.TILT`).
    rb_radial: Option<Rc<RadioButton>>,
    /// Java private final `rbHammingLikeFilter` (null unless `PanelId.TILT`).
    rb_hamming_like_filter: Option<Rc<RadioButton>>,
    /// Java private final `pnlForm` (null unless `PanelId.TILT`).
    pnl_form: Option<Rc<JComponent>>,
    /// Java private final `parent` (held weakly: the parent owns this panel).
    parent: Weak<dyn RadialParent>,
    /// Java private final `lFakeSIRTiterations` (a `JLabel`; null unless
    /// `PanelId.TILT`).
    l_fake_sirt_iterations: Option<Rc<JComponent>>,
    /// Java private final `pnlExactFilterSize` (null unless `PanelId.TILT`).
    pnl_exact_filter_size: Option<Rc<JComponent>>,

    /// Java private `multifiltFilterType = null`.
    multifilt_filter_type: RefCell<Option<Rc<dyn FilterType>>>,
    /// Java private `debug = false`.
    debug: Cell<bool>,
    /// Java private `init = true`.
    init: Cell<bool>,
}

impl RadialPanel {
    /// Java private constructor `RadialPanel(BaseManager, AxisID, PanelId,
    /// RadialParent)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        panel_id: PanelId,
        parent: Weak<dyn RadialParent>,
    ) -> Rc<RadialPanel> {
        // Field initializers.
        let pnl_root = JComponent::new_panel();
        let ltf_radial_max =
            LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some(RADIAL_LABEL));
        let ltf_radial_fall_off = LabeledTextField::new_field_type_string(
            FieldType::FloatingPoint,
            Some(RADIAL_FALLOFF_OLD_LABEL),
        );
        let pnl_high_frequency_filtering = JComponent::new_panel();
        // Constructor body.
        let rb_default;
        let rb_fake_sirt_iterations;
        let tf_fake_sirt_iterations;
        let rb_exact_filter_size;
        let tf_exact_filter_size;
        let rb_radial;
        let rb_hamming_like_filter;
        let tf_hamming_like_filter;
        let pnl_form;
        let pnl_exact_filter_size;
        let l_fake_sirt_iterations;
        if panel_id == PanelId::Tilt {
            let bg_form = ButtonGroup::new();
            rb_default = Some(RadioButton::new_string_button_group(
                Some("Standard ramp function"),
                Some(&bg_form),
            ));
            rb_fake_sirt_iterations = Some(RadioButton::new_string_button_group(
                Some(FAKE_SIRT_ITERATIONS_LABEL),
                Some(&bg_form),
            ));
            tf_fake_sirt_iterations = Some(TextField::new(
                FieldType::Integer,
                Some(FAKE_SIRT_ITERATIONS_LABEL),
                None,
            ));
            rb_exact_filter_size = Some(RadioButton::new_string_button_group(
                Some(EXACT_FILTER_SIZE_LABEL),
                Some(&bg_form),
            ));
            tf_exact_filter_size = Some(TextField::new(
                FieldType::Integer,
                Some(EXACT_FILTER_SIZE_LABEL),
                None,
            ));
            let bg_high_frequency_filtering = ButtonGroup::new();
            let radial = RadioButton::new_string_button_group(
                Some(RADIAL_BUTTON_LABEL),
                Some(&bg_high_frequency_filtering),
            );
            radial.set_name(Some(RADIAL_LABEL));
            rb_radial = Some(radial);
            ltf_radial_max.set_label(Some(RADIAL_MAX_LABEL));
            ltf_radial_max.set_name(Some(RADIAL_LABEL));
            rb_hamming_like_filter = Some(RadioButton::new_string_button_group(
                Some("Hamming-like filter (as in tomo3d) starting from: "),
                Some(&bg_high_frequency_filtering),
            ));
            tf_hamming_like_filter = Some(TextField::new(
                FieldType::FloatingPoint,
                Some("Hamming-like filter (as in tomo3d) starting from: "),
                None,
            ));
            pnl_form = Some(JComponent::new_panel());
            pnl_exact_filter_size = Some(JComponent::new_panel());
            l_fake_sirt_iterations = Some(JComponent::new_label(" iterations"));
        } else {
            rb_default = None;
            rb_fake_sirt_iterations = None;
            tf_fake_sirt_iterations = None;
            rb_exact_filter_size = None;
            tf_exact_filter_size = None;
            rb_radial = None;
            rb_hamming_like_filter = None;
            tf_hamming_like_filter = None;
            pnl_form = None;
            pnl_exact_filter_size = None;
            l_fake_sirt_iterations = None;
        }
        Rc::new(RadialPanel {
            pnl_root,
            ltf_radial_max,
            ltf_radial_fall_off,
            pnl_high_frequency_filtering,
            tf_hamming_like_filter,
            tf_exact_filter_size,
            panel_id,
            manager,
            axis_id,
            rb_default,
            tf_fake_sirt_iterations,
            rb_fake_sirt_iterations,
            rb_exact_filter_size,
            rb_radial,
            rb_hamming_like_filter,
            pnl_form,
            parent,
            l_fake_sirt_iterations,
            pnl_exact_filter_size,
            multifilt_filter_type: RefCell::new(None),
            debug: Cell::new(false),
            init: Cell::new(true),
        })
    }

    /// Java static `getInstance(BaseManager, AxisID, PanelId, RadialParent)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        panel_id: PanelId,
        parent: Weak<dyn RadialParent>,
    ) -> Rc<RadialPanel> {
        let instance = RadialPanel::new(manager, axis_id, panel_id, parent);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        if self.panel_id == PanelId::Tilt {
            self.rb_default().set_selected_boolean(true);
        }
        // panel
        let pnl_radial = JComponent::new_panel();
        let mut pnl_fake_sirt_iterations: Option<Rc<JComponent>> = None;
        let mut pnl_default: Option<Rc<JComponent>> = None;
        let mut pnl_hamming_like_filter: Option<Rc<JComponent>> = None;
        if self.panel_id == PanelId::Tilt {
            pnl_fake_sirt_iterations = Some(JComponent::new_panel());
            pnl_default = Some(JComponent::new_panel());
            pnl_hamming_like_filter = Some(JComponent::new_panel());
        }
        // Root
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        self.pnl_root.set_border_title(
            EtchedBorder::new(Some("Radial Filtering"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        if self.panel_id == PanelId::Tilt {
            self.pnl_root.add(self.pnl_form());
        }
        self.pnl_root.add(&self.pnl_high_frequency_filtering);
        if self.panel_id == PanelId::Tilt {
            let pnl_form = self.pnl_form();
            let pnl_default = pnl_default.as_ref().unwrap();
            let pnl_fake_sirt_iterations = pnl_fake_sirt_iterations.as_ref().unwrap();
            let pnl_exact_filter_size = self.pnl_exact_filter_size.as_ref().unwrap();
            // Form
            // Swing layout: pnlForm BoxLayout Y_AXIS.
            pnl_form.set_border_title(
                EtchedBorder::new(Some("Form of Radial Filter"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
            pnl_form.add(pnl_default);
            pnl_form.add(pnl_fake_sirt_iterations);
            pnl_form.add(pnl_exact_filter_size);
            // Default
            // Swing layout: pnlDefault BoxLayout X_AXIS; trailing horizontal glue.
            pnl_default.add(&self.rb_default().get_component());
            // FakeSIRTiterations
            // Swing layout: pnlFakeSIRTiterations BoxLayout X_AXIS; trailing glue.
            pnl_fake_sirt_iterations.add(&self.rb_fake_sirt_iterations().get_component());
            pnl_fake_sirt_iterations.add(&self.tf_fake_sirt_iterations().get_component());
            pnl_fake_sirt_iterations.add(self.l_fake_sirt_iterations.as_ref().unwrap());
            // ExactFilterSize
            // Swing layout: pnlExactFilterSize BoxLayout X_AXIS; trailing glue.
            pnl_exact_filter_size.add(&self.rb_exact_filter_size().get_component());
            pnl_exact_filter_size.add(&self.tf_exact_filter_size().get_component());
        }
        // HighFrequencyFiltering
        // Swing layout: pnlHighFrequencyFiltering BoxLayout Y_AXIS.
        self.pnl_high_frequency_filtering.set_border_title(
            EtchedBorder::new(Some("High-Frequency Filtering"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        self.pnl_high_frequency_filtering.add(&pnl_radial);
        if self.panel_id == PanelId::Tilt {
            self.pnl_high_frequency_filtering
                .add(pnl_hamming_like_filter.as_ref().unwrap());
        }
        // Radial
        // Swing layout: pnlRadial BoxLayout X_AXIS; trailing horizontal glue.
        if self.panel_id == PanelId::Tilt {
            pnl_radial.add(&self.rb_radial().get_component());
        }
        pnl_radial.add(&self.ltf_radial_max.get_container());
        pnl_radial.add(&self.ltf_radial_fall_off.get_container());
        // HammingLikeFilter
        if self.panel_id == PanelId::Tilt {
            let pnl_hamming_like_filter = pnl_hamming_like_filter.as_ref().unwrap();
            // Swing layout: pnlHammingLikeFilter BoxLayout X_AXIS; trailing glue.
            pnl_hamming_like_filter.add(&self.rb_hamming_like_filter().get_component());
            pnl_hamming_like_filter.add(&self.tf_hamming_like_filter().get_component());
        }
        self.update_display();
        self.init.set(false);
    }

    /// Java private `addListeners()`.  `this` (the `ActionListener`) is a
    /// closure calling [`RadialPanel::action_performed`].
    fn add_listeners(self: &Rc<Self>) {
        if self.panel_id == PanelId::Tilt {
            let weak = Rc::downgrade(self);
            let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(panel) = weak.upgrade() {
                    panel.action_performed(event);
                }
            });
            self.rb_default().add_action_listener(listener.clone());
            self.rb_fake_sirt_iterations()
                .add_action_listener(listener.clone());
            self.rb_exact_filter_size()
                .add_action_listener(listener.clone());
            self.rb_radial().add_action_listener(listener.clone());
            self.rb_hamming_like_filter().add_action_listener(listener);
        }
    }

    /// Java `setMultifiltFilterType(FilterType)`.  Disable filter types on this
    /// panel if they are selected in the multifilt panel.
    pub fn set_multifilt_filter_type(&self, multifilt_filter_type: Option<Rc<dyn FilterType>>) {
        *self.multifilt_filter_type.borrow_mut() = multifilt_filter_type;
        self.update_display();
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java `getRoot()`.
    pub fn get_root(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `actionPerformed(ActionEvent)` (implements `ActionListener`).
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.update_display();
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.ltf_radial_max.set_editable(editable);
        self.ltf_radial_fall_off.set_editable(editable);
        if self.panel_id == PanelId::Tilt {
            self.rb_default().set_editable(editable);
            self.rb_fake_sirt_iterations().set_editable(editable);
            self.tf_fake_sirt_iterations().set_editable(editable);
            self.rb_exact_filter_size().set_editable(editable);
            self.tf_exact_filter_size().set_editable(editable);
            self.rb_radial().set_editable(editable);
            self.rb_hamming_like_filter().set_editable(editable);
            self.tf_hamming_like_filter().set_editable(editable);
        }
    }

    /// Java final `msgMethodChanged()`.
    pub fn msg_method_changed(&self) {
        self.update_display();
    }

    /// Java `updateDisplay()`.  `init` is true when called from getInstance or
    /// the constructor - used to avoid querying other classes before they are
    /// complete.
    pub fn update_display(&self) {
        // The parent owns this panel, so it outlives it; the Java reference is
        // never null.
        let Some(parent) = self.parent.upgrade() else {
            return;
        };
        let mut pack = false;
        let advanced = parent.is_advanced();
        if let Some(pnl_form) = &self.pnl_form {
            let pnl_exact_filter_size = self.pnl_exact_filter_size.as_ref().unwrap();
            if !parent.is_ctf3d() {
                pnl_form.set_visible(true);
                pnl_exact_filter_size.set_visible(advanced || parent.is_multifilt());
            } else {
                // If it's not defaulted, it should be visible in Basic.
                pnl_form.set_visible(advanced || !self.rb_default().is_selected());
                pnl_exact_filter_size.set_visible(true);
            }
            pack = true;
        }
        // multifiltFilterType disables the same type of filter in this panel when they are
        // displayed together.
        let multifilt_filter_type = self.multifilt_filter_type.borrow().clone();
        let enable_high_frequency_filter = self.init.get()
            || !parent.is_multifilt()
            || match &multifilt_filter_type {
                Some(multifilt_filter_type) => !multifilt_filter_type.is_high_frequency_filter(),
                None => true,
            };
        self.pnl_high_frequency_filtering
            .set_enabled(enable_high_frequency_filter);
        // radial
        let mut enable_radial = enable_high_frequency_filter;
        if self.panel_id == PanelId::Tilt {
            self.rb_radial().set_enabled(enable_high_frequency_filter);
            enable_radial = enable_radial && self.rb_radial().is_selected();
        }
        self.ltf_radial_max.set_enabled(enable_radial);
        self.ltf_radial_fall_off.set_enabled(enable_radial);
        if self.panel_id == PanelId::Tilt {
            // hammingLikeFilter
            self.rb_hamming_like_filter()
                .set_enabled(enable_high_frequency_filter);
            self.tf_hamming_like_filter().set_enabled(
                enable_high_frequency_filter && self.rb_hamming_like_filter().is_selected(),
            );
            //
            let enable_radial_filter = self.init.get()
                || !parent.is_multifilt()
                || match &multifilt_filter_type {
                    Some(multifilt_filter_type) => !multifilt_filter_type.is_radial_filter(),
                    None => true,
                };
            self.pnl_form().set_enabled(enable_radial_filter);
            // Default
            self.rb_default().set_enabled(enable_radial_filter);
            // FakeSIRTiterations
            self.rb_fake_sirt_iterations()
                .set_enabled(enable_radial_filter);
            self.tf_fake_sirt_iterations()
                .set_enabled(enable_radial_filter && self.rb_fake_sirt_iterations().is_selected());
            self.l_fake_sirt_iterations
                .as_ref()
                .unwrap()
                .set_enabled(enable_radial_filter);
            // ExactFilterSize
            self.rb_exact_filter_size()
                .set_enabled(enable_radial_filter);
            self.tf_exact_filter_size()
                .set_enabled(enable_radial_filter && self.rb_exact_filter_size().is_selected());
        }
        if pack {
            ui_harness::with(|harness| {
                harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
            });
        }
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_radial_radius(
            self.panel_id,
            self.axis_id,
            self.ltf_radial_max.get_text_void().as_deref(),
        );
        meta_data.set_radial_sigma(
            self.panel_id,
            self.axis_id,
            self.ltf_radial_fall_off.get_text_void().as_deref(),
        );
        if self.panel_id == PanelId::Tilt {
            meta_data.set_hamming_like_filter(
                self.panel_id,
                self.axis_id,
                self.tf_hamming_like_filter().get_text_void().as_deref(),
            );
            meta_data.set_fake_sirt_iterations(
                self.panel_id,
                self.axis_id,
                self.tf_fake_sirt_iterations().get_text_void().as_deref(),
            );
            meta_data.set_exact_filter_size(
                self.panel_id,
                self.axis_id,
                self.tf_exact_filter_size().get_text_void().as_deref(),
            );
        }
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.ltf_radial_max.set_text_string(
            meta_data
                .get_radial_radius(self.panel_id, self.axis_id)
                .as_deref(),
        );
        self.ltf_radial_fall_off.set_text_string(
            meta_data
                .get_radial_sigma(self.panel_id, self.axis_id)
                .as_deref(),
        );
        if self.panel_id == PanelId::Tilt {
            self.tf_hamming_like_filter().set_text_string(
                meta_data
                    .get_hamming_like_filter(self.panel_id, self.axis_id)
                    .as_deref(),
            );
            self.tf_fake_sirt_iterations().set_text_string(
                meta_data
                    .get_fake_sirt_iterations(self.panel_id, self.axis_id)
                    .as_deref(),
            );
            self.tf_exact_filter_size().set_text_string(
                meta_data
                    .get_exact_filter_size(self.panel_id, self.axis_id)
                    .as_deref(),
            );
        }
        self.update_display();
    }

    /// Java `setParameters(ConstTiltParam)`.
    pub fn set_parameters_const_tilt_param(&self, tilt_param: &dyn ConstTiltParam) {
        if tilt_param.has_radial_weighting_function() {
            self.ltf_radial_max
                .set_non_empty_text(Some(&tilt_param.get_radial_bandwidth()));
            self.ltf_radial_fall_off
                .set_non_empty_text(Some(&tilt_param.get_radial_falloff()));
        }
        if tilt_param.is_falloff_is_true_sigma() {
            self.ltf_radial_fall_off
                .set_label(Some(RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL));
        } else {
            self.ltf_radial_fall_off
                .set_label(Some(RADIAL_FALLOFF_OLD_LABEL));
        }
        if self.panel_id == PanelId::Tilt {
            if tilt_param.is_hamming_like_filter() {
                self.rb_hamming_like_filter().set_selected_boolean(true);
                self.tf_hamming_like_filter()
                    .set_text_string(Some(&tilt_param.get_hamming_like_filter()));
            } else {
                // Default
                self.rb_radial().set_selected_boolean(true);
            }
            if tilt_param.is_fake_sirt_iterations() {
                self.rb_fake_sirt_iterations().set_selected_boolean(true);
                self.tf_fake_sirt_iterations()
                    .set_text_string(Some(&tilt_param.get_fake_sirt_iterations()));
            }
            if tilt_param.is_exact_filter_size() {
                self.rb_exact_filter_size().set_selected_boolean(true);
                self.tf_exact_filter_size()
                    .set_text_string(Some(&tilt_param.get_exact_filter_size()));
            }
        }
        self.update_display();
    }

    /// Java `getParameters(SirtsetupParam, boolean)`.
    pub fn get_parameters_sirtsetup_param_boolean(
        &self,
        param: &mut SirtsetupParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        if self.ltf_radial_max.is_editable() && self.ltf_radial_max.is_enabled() {
            let Ok(text) = self.ltf_radial_max.get_text_boolean(do_validation) else {
                return false;
            };
            param.set_radius_and_sigma(0, text.as_deref());
        }
        if self.ltf_radial_fall_off.is_enabled() {
            let Ok(text) = self.ltf_radial_fall_off.get_text_boolean(do_validation) else {
                return false;
            };
            param.set_radius_and_sigma(1, text.as_deref());
        }
        true
    }

    /// Java `setParameters(SirtsetupParam)`.
    pub fn set_parameters_sirtsetup_param(&self, param: &SirtsetupParam) {
        self.ltf_radial_max
            .set_text_string(Some(&param.get_radius_and_sigma(0)));
        self.ltf_radial_fall_off
            .set_text_string(Some(&param.get_radius_and_sigma(1)));
        if param.is_falloff_is_true_sigma() {
            self.ltf_radial_fall_off
                .set_label(Some(RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL));
        } else {
            self.ltf_radial_fall_off
                .set_label(Some(RADIAL_FALLOFF_OLD_LABEL));
        }
    }

    /// Java public `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.set(debug);
    }

    /// Java `getParameters(TiltParam, boolean) throws NumberFormatException`.
    /// Nothing in the body throws `NumberFormatException`, so no error
    /// channel is needed.
    pub fn get_parameters_tilt_param_boolean(
        &self,
        tilt_param: &mut TiltParam,
        do_validation: bool,
    ) -> bool {
        // Bug# 2098 comments 6 and 7. Ignore whether the panels these fields are on are
        // disabled. They should be saved to tilt.com as usual.
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        if ((self.panel_id == PanelId::Tilt && self.rb_radial().is_selected())
            || self.panel_id != PanelId::Tilt)
            && (!self.ltf_radial_max.is_empty() || !self.ltf_radial_fall_off.is_empty())
        {
            let Ok(radial_bandwidth) = self.ltf_radial_max.get_text_boolean(do_validation) else {
                return false;
            };
            tilt_param.set_radial_bandwidth(radial_bandwidth.as_deref());
            let Ok(radial_falloff) = self.ltf_radial_fall_off.get_text_boolean(do_validation)
            else {
                return false;
            };
            tilt_param.set_radial_falloff(radial_falloff.as_deref());
        } else {
            tilt_param.reset_radial_filter();
        }
        if self.panel_id == PanelId::Tilt {
            if self.rb_hamming_like_filter().is_selected() {
                let Ok(text) = self
                    .tf_hamming_like_filter()
                    .get_text_boolean(do_validation)
                else {
                    return false;
                };
                tilt_param.set_hamming_like_filter(text.as_deref());
            } else {
                tilt_param.reset_hamming_like_filter();
            }
            if self.rb_fake_sirt_iterations().is_selected() {
                let Ok(text) = self
                    .tf_fake_sirt_iterations()
                    .get_text_boolean(do_validation)
                else {
                    return false;
                };
                tilt_param.set_fake_sirt_iterations(text.as_deref());
            } else {
                tilt_param.reset_fake_sirt_iterations();
            }
            if self.rb_exact_filter_size().is_selected() {
                let Ok(text) = self.tf_exact_filter_size().get_text_boolean(do_validation) else {
                    return false;
                };
                tilt_param.set_exact_filter_size(text.as_deref());
            } else {
                tilt_param.reset_exact_filter_size();
            }
        } else {
            tilt_param.reset_exact_filter_size();
        }
        true
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        if self.panel_id == PanelId::Sirtsetup {
            // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
            let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
            match unsafe {
                autodoc_factory::get_instance(
                    Some(self.manager),
                    Some(autodoc_factory::SIRTSETUP),
                    self.axis_id,
                    false,
                )
            } {
                Ok(instance) => autodoc = instance as *const Autodoc,
                // `catch (final LockException except) {}`.
                Err(LogFileError::Lock(_)) => {}
                // `catch (final LogFileException | IOException except)`:
                // `except.printStackTrace()`.
                Err(except) => eprintln!("{}", except),
            }
            // SAFETY: `autodoc` is null or an autodoc the factory keeps for the
            // life of the process.
            let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
                None
            } else {
                Some(unsafe { &*autodoc })
            };
            let tooltip =
                etomo_autodoc::get_tooltip(autodoc, Some(sirtsetup_param::RADIUS_AND_SIGMA_KEY));
            self.ltf_radial_max.set_tool_tip_text(tooltip.as_deref());
            self.ltf_radial_fall_off
                .set_tool_tip_text(tooltip.as_deref());
        } else {
            self.ltf_radial_max.set_tool_tip_text(Some(
                "The spatial frequency at which to switch from the R-weighted radial \
                 filter to a Gaussian falloff.  Frequency is in cycles/pixel and \
                 ranges from 0-0.5.  Both a cutoff and a falloff must be entered.",
            ));
            self.ltf_radial_fall_off.set_tool_tip_text(Some(
                "The sigma value of a Gaussian which determines how fast the radial \
                 filter falls off at spatial frequencies above the cutoff frequency.  \
                 Frequency is in cycles/pixel and ranges from 0-0.5.  Both a \
                 cutoff and a falloff must be entered ",
            ));
            if self.panel_id == PanelId::Tilt {
                self.rb_default().set_tool_tip_text_string(Some(
                    "Use the standard ramp filter linearly proportional to spatial frequency \
                     until the high-frequency filter starts",
                ));
                self.rb_radial().set_tool_tip_text_string(Some(
                    shared_strings::GAUSSIAN_FILTER_RADIO_BUTTON_TOOLTIP,
                ));
                self.rb_hamming_like_filter().set_tool_tip_text_string(Some(
                    shared_strings::HAMMING_LIKE_FILTER_RADIO_BUTTON_TOOLTIP,
                ));
                self.tf_hamming_like_filter().set_tool_tip_text(Some(
                    "Spatial frequency at which to start multiplying by an approximate \
                     Hamming window that attenuates by 0.06 at 0.5 cycles/pixel.  Frequency \
                     ranges from 0 to 0.5.",
                ));
                self.rb_fake_sirt_iterations()
                    .set_tool_tip_text_string(Some(
                        shared_strings::SIRT_LIKE_FILTER_RADIO_BUTTON_TOOLTIP,
                    ));
                self.tf_fake_sirt_iterations().set_tool_tip_text(Some(
                    "Number of iterations of SIRT that the filtered back-projection should \
                     match approximately",
                ));
                self.rb_exact_filter_size().set_tool_tip_text_string(Some(
                    shared_strings::EXACT_FILTER_RADIO_BUTTON_TOOLTIP,
                ));
                self.tf_exact_filter_size().set_tool_tip_text(Some(
                    "A size in unbinned pixels that determines at what frequency the filter \
                     functions reach a plateau",
                ));
            }
        }
    }

    // Rust-only accessors for the `PanelId.TILT` fields.  Java dereferences
    // them only under `panelId == PanelId.TILT` (where the constructor set
    // them), so `unwrap` cannot fail on any path the Java takes.

    fn rb_default(&self) -> &Rc<RadioButton> {
        self.rb_default.as_ref().unwrap()
    }
    fn rb_fake_sirt_iterations(&self) -> &Rc<RadioButton> {
        self.rb_fake_sirt_iterations.as_ref().unwrap()
    }
    fn tf_fake_sirt_iterations(&self) -> &Rc<TextField> {
        self.tf_fake_sirt_iterations.as_ref().unwrap()
    }
    fn rb_exact_filter_size(&self) -> &Rc<RadioButton> {
        self.rb_exact_filter_size.as_ref().unwrap()
    }
    fn tf_exact_filter_size(&self) -> &Rc<TextField> {
        self.tf_exact_filter_size.as_ref().unwrap()
    }
    fn rb_radial(&self) -> &Rc<RadioButton> {
        self.rb_radial.as_ref().unwrap()
    }
    fn rb_hamming_like_filter(&self) -> &Rc<RadioButton> {
        self.rb_hamming_like_filter.as_ref().unwrap()
    }
    fn tf_hamming_like_filter(&self) -> &Rc<TextField> {
        self.tf_hamming_like_filter.as_ref().unwrap()
    }
    fn pnl_form(&self) -> &Rc<JComponent> {
        self.pnl_form.as_ref().unwrap()
    }
}
