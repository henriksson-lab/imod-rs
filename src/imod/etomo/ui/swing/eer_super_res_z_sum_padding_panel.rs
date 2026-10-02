//! `IMOD/Etomo/src/etomo/ui/swing/EERSuperResZSumPaddingPanel.java`.
//!
//! Java `final class EERSuperResZSumPaddingPanel implements Expandable,
//! ActionListener`: the "Reading EER Files" panel of the Align Frames tool
//! (`AlignFramesPanel`), which sets alignframes' `EERSuperResZSumPadding`
//! option.  An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`EERSuperResZSumPaddingPanel::get_instance`]; every method takes `&self`.
//! The panel is its own `ActionListener`; that listener is a closure holding a
//! weak reference.

use std::rc::{Rc, Weak};

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::etomo_button_group::EtomoButtonGroup;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::label::Label;
use super::panel_header::PanelHeader;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_ebutton::RadioEbutton;
use super::spinner::Spinner;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::align_frames_param::{self, AlignFramesParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, ButtonModel, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::eer_super_res::EERSuperRes;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;

/// Java private static final `EER_Z_SUM_FRAMES_MIN`.
const EER_Z_SUM_FRAMES_MIN: i32 = 2;
/// Java private static final `EER_Z_SUM_FRAMES_MAX`.
const EER_Z_SUM_FRAMES_MAX: i32 = 100;
/// Java private static final `EER_Z_SUM_SETS_MIN`.
const EER_Z_SUM_SETS_MIN: i32 = 1;
/// Java private static final `EER_Z_SUM_SETS_MAX`.
const EER_Z_SUM_SETS_MAX: i32 = 1000;

/// Java `final class EERSuperResZSumPaddingPanel`.
pub struct EERSuperResZSumPaddingPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlBody = new JPanel()`.
    pnl_body: Rc<JComponent>,
    /// Java private final `lSuperRes = new Label("Frame size to read in:")`.
    l_super_res: Rc<Label>,
    /// Java private final `bgSuperRes = new ButtonGroup()`.
    bg_super_res: Rc<ButtonGroup>,
    /// Java private final `rbSuperResNone`.
    rb_super_res_none: Rc<RadioButton>,
    /// Java private final `rbSuperRes2x`.
    rb_super_res_2x: Rc<RadioButton>,
    /// Java private final `rbSuperRes4x`.
    rb_super_res_4x: Rc<RadioButton>,
    /// Java private final `bgZSum = new EtomoButtonGroup()`.
    #[allow(dead_code)]
    bg_z_sum: Rc<EtomoButtonGroup>,
    /// Java private final `rbZSumFrames`.
    rb_z_sum_frames: Rc<RadioEbutton>,
    /// Java private final `spZSumFrames`.
    sp_z_sum_frames: Rc<Spinner>,
    /// Java private final `lZSumFrames = new Label("images to align")`.
    l_z_sum_frames: Rc<Label>,
    /// Java private final `rbZSumSets`.
    rb_z_sum_sets: Rc<RadioEbutton>,
    /// Java private final `spZSumSets`.
    sp_z_sum_sets: Rc<Spinner>,
    /// Java private final `lZSumSets = new Label("frames")`.
    l_z_sum_sets: Rc<Label>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Rust-only: Java `this` as the `ActionListener` it registers.
    action_listener: ActionListener,
}

impl EERSuperResZSumPaddingPanel {
    /// Java private constructor `EERSuperResZSumPaddingPanel(BaseManager, AxisID,
    /// DialogType)`, with the field initializers.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<EERSuperResZSumPaddingPanel> {
        Rc::new_cyclic(|self_ref: &Weak<EERSuperResZSumPaddingPanel>| {
            let bg_super_res = ButtonGroup::new();
            let rb_super_res_none = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(EERSuperRes::NONE),
                Some(&bg_super_res),
            );
            let rb_super_res_2x = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(EERSuperRes::TWO_X),
                Some(&bg_super_res),
            );
            let rb_super_res_4x = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(EERSuperRes::FOUR_X),
                Some(&bg_super_res),
            );
            let bg_z_sum = EtomoButtonGroup::new();
            let rb_z_sum_frames = RadioEbutton::get_enum_instance(
                Some(EnumeratedTypeRef::new(EERZSum::FRAMES)),
                Some(&bg_z_sum),
            );
            // Java `AlignFramesParam.getEERZSumFramesDefault()` is an Integer that is
            // never null here (unboxed to the int parameter).
            let sp_z_sum_frames = Spinner::get_instance_string_int_int_int(
                EERZSum::FRAMES.get_label().as_deref(),
                AlignFramesParam::get_eer_z_sum_frames_default().unwrap_or_default(),
                EER_Z_SUM_FRAMES_MIN,
                EER_Z_SUM_FRAMES_MAX,
            );
            let l_z_sum_frames = Label::new_string(Some("images to align"));
            let rb_z_sum_sets = RadioEbutton::get_enum_instance(
                Some(EnumeratedTypeRef::new(EERZSum::SETS)),
                Some(&bg_z_sum),
            );
            let sp_z_sum_sets = Spinner::get_instance_string_int_int_int(
                EERZSum::SETS.get_label().as_deref(),
                align_frames_param::EER_Z_SUM_SETS_DEFAULT,
                EER_Z_SUM_SETS_MIN,
                EER_Z_SUM_SETS_MAX,
            );
            let l_z_sum_sets = Label::new_string(Some("frames"));
            // Constructor body.
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            let header = PanelHeader::get_instance(
                Some("Reading EER Files"),
                Some(expandable),
                Some(dialog_type),
            );
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(event);
                }
            });
            EERSuperResZSumPaddingPanel {
                pnl_root: JComponent::new_panel(),
                pnl_body: JComponent::new_panel(),
                l_super_res: Label::new_string(Some("Frame size to read in:")),
                bg_super_res,
                rb_super_res_none,
                rb_super_res_2x,
                rb_super_res_4x,
                bg_z_sum,
                rb_z_sum_frames,
                sp_z_sum_frames,
                l_z_sum_frames,
                rb_z_sum_sets,
                sp_z_sum_sets,
                l_z_sum_sets,
                manager,
                axis_id,
                header,
                action_listener,
            }
        })
    }

    /// Java package-private static `getInstance(BaseManager, AxisID, DialogType)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<EERSuperResZSumPaddingPanel> {
        let instance = EERSuperResZSumPaddingPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.set_tootips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_super_res = JComponent::new_panel();
        let pnl_z_sum_frames = JComponent::new_panel();
        let pnl_z_sum_sets = JComponent::new_panel();
        // init
        self.header.set_open(false);
        // Root
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        // Swing painting: pnlRoot.setBorder(BorderFactory.createEtchedBorder())
        // (untitled).
        self.pnl_root.add(&self.header.get_component());
        self.pnl_root.add(&self.pnl_body);
        // Body
        // Swing layout: pnlBody BoxLayout Y_AXIS; vertical struts (2) after the
        // two ZSum rows.
        self.pnl_body.add(&pnl_super_res);
        self.pnl_body.add(&pnl_z_sum_frames);
        self.pnl_body.add(&pnl_z_sum_sets);
        // SuperRes
        // Swing layout: pnlSuperRes BoxLayout X_AXIS; horizontal strut (2) first.
        pnl_super_res.add(&self.l_super_res.get_component());
        pnl_super_res.add(&self.rb_super_res_none.get_component());
        pnl_super_res.add(&self.rb_super_res_2x.get_component());
        pnl_super_res.add(&self.rb_super_res_4x.get_component());
        // ZSumFrames
        // Swing layout: pnlZSumFrames BoxLayout X_AXIS; horizontal struts (3)
        // before the label and (2) after it.
        pnl_z_sum_frames.add(&self.rb_z_sum_frames.get_component());
        pnl_z_sum_frames.add(&self.sp_z_sum_frames.get_component());
        pnl_z_sum_frames.add(&self.l_z_sum_frames.get_component());
        // ZSumSets
        // Swing layout: pnlZSumSets BoxLayout X_AXIS; horizontal struts (3)
        // before the label and (2) after it.
        pnl_z_sum_sets.add(&self.rb_z_sum_sets.get_component());
        pnl_z_sum_sets.add(&self.sp_z_sum_sets.get_component());
        pnl_z_sum_sets.add(&self.l_z_sum_sets.get_component());

        self.update_display();
    }

    /// Java public `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `addListeners()`.
    fn add_listeners(&self) {
        self.rb_z_sum_frames
            .add_action_listener(self.action_listener.clone());
        self.rb_z_sum_sets
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.update_display();
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let z_sum_frames = self.rb_z_sum_frames.is_selected();
        self.sp_z_sum_frames.set_enabled(z_sum_frames);
        self.l_z_sum_frames
            .get_component()
            .set_enabled(z_sum_frames);
        let z_sum_sets = self.rb_z_sum_sets.is_selected();
        self.sp_z_sum_sets.set_enabled(z_sum_sets);
        self.l_z_sum_sets.get_component().set_enabled(z_sum_sets);
    }

    /// Java package-private `setParameters(AlignFramesParam)`.
    pub fn set_parameters(&self, param: &AlignFramesParam) {
        let eer_super_res = EERSuperRes::get_instance(Some(param.get_eer_super_res()));
        if eer_super_res == Some(EERSuperRes::NONE) {
            self.rb_super_res_none.set_selected_boolean(true);
        } else if eer_super_res == Some(EERSuperRes::TWO_X) {
            self.rb_super_res_2x.set_selected_boolean(true);
        } else if eer_super_res == Some(EERSuperRes::FOUR_X) {
            self.rb_super_res_4x.set_selected_boolean(true);
        }
        if param.is_eer_z_sum_frames_set() {
            self.rb_z_sum_frames.set_selected(true);
            self.sp_z_sum_frames
                .set_value_int(param.get_eer_z_sum_frames());
        } else {
            self.rb_z_sum_sets.set_selected(true);
            self.sp_z_sum_sets.set_value_int(param.get_eer_z_sum_sets());
        }
        self.update_display();
    }

    /// Java package-private `getParameters(AlignFramesParam)`.
    pub fn get_parameters(&self, param: &mut AlignFramesParam) {
        // Java `(RadioButton.RadioButtonModel) bgSuperRes.getSelection()`.
        let model = self
            .bg_super_res
            .get_selection()
            .and_then(|button| button.get_model());
        if let Some(model) = model {
            if let Some(model) = model.as_any().downcast_ref::<RadioButtonModel>() {
                let super_res = AbstractRadioButtonModel::get_enumerated_type(model);
                if let Some(super_res) = super_res {
                    param.set_eer_super_res(Some(&super_res.get_value()));
                }
            }
        }
        if self.rb_z_sum_frames.is_selected() {
            param.set_eer_z_sum_frames(self.sp_z_sum_frames.get_int_value());
        } else if self.rb_z_sum_sets.is_selected() {
            param.set_eer_z_sum_sets(self.sp_z_sum_sets.get_int_value());
        }
    }

    /// Java private `setTootips()` (the source's spelling).
    fn set_tootips(&self) {
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        // Java passes a null AxisID; ALIGN_FRAMES is not a per-axis autodoc, so
        // `AxisID::Only` stands in for it.
        // SAFETY: the factory keeps every autodoc it returns (and its sections)
        // for the life of the process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::ALIGN_FRAMES),
                AxisID::Only,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except):
            // except.printStackTrace().
            Err(except) => eprintln!("{except}"),
        }
        if autodoc.is_null() {
            return;
        }
        // SAFETY: see above.
        let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
        self.rb_super_res_none
            .set_tool_tip_text_string(Some(&EERSuperRes::NONE.get_tooltip()));
        self.rb_super_res_2x
            .set_tool_tip_text_string(Some(&EERSuperRes::TWO_X.get_tooltip()));
        self.rb_super_res_4x
            .set_tool_tip_text_string(Some(&EERSuperRes::FOUR_X.get_tooltip()));

        let autodoc_name = autodoc.get_autodoc_name();
        // SAFETY: see above.
        let section = unsafe {
            autodoc.get_section(
                Some(etomo_autodoc::FIELD_SECTION_NAME),
                Some(align_frames_param::EER_SUPER_RES_Z_SUM_PADDING_KEY),
            )
        };
        // Upstream bug fixed in translation (EERSuperResZSumPaddingPanel.java:217):
        // with no EERSuperResZSumPadding section in the autodoc, Java's
        // `RadioEbutton.setTooltip(String, ReadOnlySection)` dereferences the null
        // section; the radio button tooltips are left unset instead.  (The
        // `EtomoAutodoc.getTooltip` calls already answer null for a null section.)
        let section: Option<&dyn ReadOnlySection> = if section.is_null() {
            None
        } else {
            // SAFETY: see above.
            Some(unsafe { &*section })
        };

        if let Some(section) = section {
            self.rb_z_sum_frames
                .set_tooltip_string_read_only_section(Some(autodoc_factory::ALIGN_FRAMES), section);
        }
        let mut tooltip = section.and_then(|section| {
            etomo_autodoc::get_tooltip_enum_value_name(
                Some(&autodoc_name),
                section,
                Some(EERZSum::FRAMES.spinner_enum_value_name),
            )
        });
        self.sp_z_sum_frames.set_tool_tip_text(tooltip.as_deref());
        self.l_z_sum_frames
            .get_component()
            .set_tool_tip_text(tooltip.as_deref());

        if let Some(section) = section {
            self.rb_z_sum_sets
                .set_tooltip_string_read_only_section(Some(autodoc_factory::ALIGN_FRAMES), section);
        }
        tooltip = section.and_then(|section| {
            etomo_autodoc::get_tooltip_enum_value_name(
                Some(&autodoc_name),
                section,
                Some(EERZSum::SETS.spinner_enum_value_name),
            )
        });
        self.sp_z_sum_sets.set_tool_tip_text(tooltip.as_deref());
        self.l_z_sum_sets
            .get_component()
            .set_tool_tip_text(tooltip.as_deref());
    }
}

impl Expandable for EERSuperResZSumPaddingPanel {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        }
        let manager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

/// Java private static nested class `EERZSum implements EnumeratedType`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct EERZSum {
    /// Java private final `radioEnumValueName`.
    radio_enum_value_name: &'static str,
    /// Java private final `spinnerEnumValueName`.
    spinner_enum_value_name: &'static str,
    /// Java private final `label`.
    label: &'static str,
}

impl EERZSum {
    /// Java `FRAMES = new EERZSum("framesradio", "framesspin", "Sum frames to
    /// make")`.
    const FRAMES: EERZSum = EERZSum {
        radio_enum_value_name: "framesradio",
        spinner_enum_value_name: "framesspin",
        label: "Sum frames to make",
    };
    /// Java `SETS = new EERZSum("setsradio", "setsspin", "Sum successive sets
    /// of")`.
    const SETS: EERZSum = EERZSum {
        radio_enum_value_name: "setsradio",
        spinner_enum_value_name: "setsspin",
        label: "Sum successive sets of",
    };
    /// Java `DEFAULT = FRAMES`.
    const DEFAULT: EERZSum = EERZSum::FRAMES;
}

/// Java `toString()`: `radioEnumValueName`.
impl std::fmt::Display for EERZSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.radio_enum_value_name)
    }
}

impl EnumeratedType for EERZSum {
    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        *self == EERZSum::DEFAULT
    }

    /// Java `getValue()`: null.  The trait returns a number by value; an unset
    /// `EtomoNumber` (the null number) stands for Java's null.
    fn get_value(&self) -> ConstEtomoNumber {
        EtomoNumber::new().base
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        Some(self.label.to_string())
    }
}
