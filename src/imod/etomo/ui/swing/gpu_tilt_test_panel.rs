//! `IMOD/Etomo/src/etomo/ui/swing/GpuTiltTestPanel.java`.
//!
//! The Tools interface's "Test GPU" panel: a number of minutes, a GPU number
//! and a button that runs `gputilttest` through `ToolsManager.gpuTiltTest`.
//! An event-dispatch-thread object (`Rc`, `&self` methods).

use std::path::Path;
use std::rc::{Rc, Weak};

use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::tool_panel::ToolPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::gpu_tilt_test_param::GpuTiltTestParam;
use crate::imod::etomo::jdk::{ActionEvent, JComponent, MouseEvent, MouseListener};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java private static final `DATASET_ROOT`.
const DATASET_ROOT: &str = "gputest";

/// Java package-private `class GpuTiltTestPanel implements ToolPanel, ContextMenu`.
pub struct GpuTiltTestPanel {
    /// Java `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java `ltfNMinutes = new LabeledTextField(FieldType.FLOATING_POINT,
    /// "# of minutes: ")`.
    ltf_n_minutes: Rc<LabeledTextField>,
    /// Java `spGpuNumber = LabeledSpinner.getInstance("GPU #: ", 0, 0, 8, 1)`.
    sp_gpu_number: Rc<LabeledSpinner>,
    /// Java `btnRunTest = new MultiLineButton("Run GPU Test")`.
    btn_run_test: Rc<MultiLineButton>,

    /// Java private final `manager`.
    manager: &'static ToolsManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
}

impl GpuTiltTestPanel {
    /// Java private `GpuTiltTestPanel(ToolsManager, AxisID)`.
    fn new(manager: &'static ToolsManager, axis_id: AxisID) -> Rc<GpuTiltTestPanel> {
        Rc::new(GpuTiltTestPanel {
            pnl_root: JComponent::new_panel(),
            ltf_n_minutes: LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("# of minutes: "),
            ),
            sp_gpu_number: LabeledSpinner::get_instance_string_int_int_int_int(
                Some("GPU #: "),
                0,
                0,
                8,
                1,
            ),
            btn_run_test: MultiLineButton::new_string(Some("Run GPU Test")),
            manager,
            axis_id,
        })
    }

    /// Java package-private static `getInstance(ToolsManager, AxisID)`.
    pub fn get_instance(manager: &'static ToolsManager, axis_id: AxisID) -> Rc<GpuTiltTestPanel> {
        let instance = GpuTiltTestPanel::new(manager, axis_id);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners(&instance);
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.ltf_n_minutes.set_text_string(Some("1"));
        self.ltf_n_minutes.set_preferred_width(80);
        // Swing layout: btnRunTest.setAlignmentX(Box.CENTER_ALIGNMENT).
        // panels
        let pnl_fields = JComponent::new_panel();
        // Root
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS));
        // pnlRoot.add(Box.createRigidArea(FixedDim.x0_y5)).
        self.pnl_root.add(&pnl_fields);
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y23)).
        self.pnl_root.add(&self.btn_run_test.get_component());
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y15)).
        // Fields
        // Swing layout: pnlFields.setLayout(new BoxLayout(pnlFields, BoxLayout.X_AXIS));
        // pnlFields.add(Box.createRigidArea(FixedDim.x10_y0)).
        pnl_fields.add(&self.ltf_n_minutes.get_container());
        // Swing layout: pnlFields.add(Box.createRigidArea(FixedDim.x15_y0)).
        pnl_fields.add(&self.sp_gpu_number.get_container());
        // Swing layout: pnlFields.add(Box.createRigidArea(FixedDim.x10_y0)).
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self, this: &Rc<GpuTiltTestPanel>) {
        let context_menu: Weak<dyn ContextMenu> = Rc::downgrade(this) as Weak<dyn ContextMenu>;
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.pnl_root.add_mouse_listener(mouse_adapter);
        // new GpuTiltTestActionListener(this)
        let listener = GpuTiltTestActionListener::new(Rc::downgrade(this));
        self.btn_run_test
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                listener.action_performed(event)
            }));
    }

    /// Java package-private `getParameters(GpuTiltTestParam, boolean)`.
    pub fn get_parameters(&self, param: &mut GpuTiltTestParam, do_validation: bool) -> bool {
        match self.ltf_n_minutes.get_text_boolean(do_validation) {
            Ok(n_minutes) => {
                param.set_n_minutes(n_minutes.as_deref());
                param.set_gpu_number(Some(self.sp_gpu_number.get_value()));
                true
            }
            // catch (FieldValidationFailedException e)
            Err(_) => false,
        }
    }

    /// Java private `action()`.
    fn action(&self) {
        let user_dir = self.manager.get_property_user_dir().unwrap_or_default();
        if !dataset_tool::validate_dataset_name_directory(
            self.manager,
            self.axis_id,
            Path::new(&user_dir),
            Some(DATASET_ROOT),
            DataFileType::Tools,
            None,
            true,
        ) {
            return;
        }
        self.manager.gpu_tilt_test(self.axis_id);
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.ltf_n_minutes
            .set_tool_tip_text(Some("The number of minutes to run the test"));
        self.sp_gpu_number.set_tool_tip_text(Some(
            "The GPU number, numbered from 1.  When 0 is selected, the fastest GPU will be used.",
        ));
        self.btn_run_test.set_tool_tip_text(Some(
            "Test the reliability of the GPU by using gputilttest to run the Tilt program repeatedly.",
        ));
    }
}

impl ToolPanel for GpuTiltTestPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        Rc::clone(&self.pnl_root)
    }
}

impl ContextMenu for GpuTiltTestPanel {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["gputilttest".to_string()];
        let man_page = ["gputilttest.html".to_string()];
        let log_file_label = ["GPU test".to_string()];
        let log_file = [format!("{DATASET_ROOT}.log")];
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("GPU Test"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            self.manager,
            self.axis_id,
        );
    }
}

/// Java private final inner class `GpuTiltTestActionListener implements
/// ActionListener`.
struct GpuTiltTestActionListener {
    /// Java private final `adaptee`.
    adaptee: Weak<GpuTiltTestPanel>,
}

impl GpuTiltTestActionListener {
    /// Java private `GpuTiltTestActionListener(GpuTiltTestPanel)`.
    fn new(adaptee: Weak<GpuTiltTestPanel>) -> GpuTiltTestActionListener {
        GpuTiltTestActionListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.action();
        }
    }
}
