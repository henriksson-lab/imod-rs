//! `IMOD/Etomo/src/etomo/ui/swing/ProcessDialog.java`, plus the
//! `DialogExitState` value type it carries.
//!
//! Abstract base of the reconstruction process dialogs (Pre-processing ...
//! Clean Up): the root panel, the exit buttons (Cancel, Postpone, Execute
//! (labelled "Done" through the subclasses' button text), Advanced) with
//! their action adapters, and the dialog exit state the buttons set before
//! calling the subclass's `done()`.
//!
//! Object model (see `ui.md`): [`ProcessDialog`] is created as an `Rc` (its
//! button listeners hold it); a subclass holds it as `base: Rc<ProcessDialog>`
//! and derefs to it, and implements [`ProcessDialogVirtual`] for `done()` and
//! any overridden action.  The subclass constructor calls
//! [`ProcessDialog::set_this`] right after creating itself.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::etomo_panel::EtomoPanel;
use super::global_expand_button::GlobalExpandButton;
use super::single_line_button::SingleLineButton;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::util::utilities;

/// `etomo.type.DialogExitState`, which the rest of the tree also imports from here.
pub use crate::imod::etomo::r#type::dialog_exit_state::DialogExitState;

/// The `ProcessDialog` methods a subclass implements or overrides,
/// dispatched virtually.  Defaults are the `ProcessDialog` bodies.
pub trait ProcessDialogVirtual: std::any::Any {
    /// The embedded `ProcessDialog` (the Java superclass part).
    fn process_dialog(&self) -> &ProcessDialog;

    /// Java abstract `done()`.
    fn done(&self);

    /// Java `buttonCancelAction(ActionEvent)`.  Action to take when the cancel
    /// button is pressed, the default action is to set the exitState
    /// attribute to CANCEL.  (`ApplicationManager` calls it with null.)
    fn button_cancel_action(&self, event: Option<&ActionEvent>) {
        self.process_dialog().button_cancel_action_super(event);
    }

    /// Java `buttonPostponeAction(ActionEvent)`.  Action to take when the
    /// postpone button is pressed, the default action is to set the exitState
    /// attribute to POSTPONE.
    fn button_postpone_action(&self, event: Option<&ActionEvent>) {
        self.process_dialog().button_postpone_action_super(event);
    }

    /// Java `buttonExecuteAction()`.  Action to take when the execute button
    /// is pressed, the default action is to set the exitState attribute to
    /// EXECUTE.
    fn button_execute_action(&self) -> bool {
        self.process_dialog().button_execute_action_super()
    }

    /// Java `saveAction()`.
    fn save_action(&self) {
        self.process_dialog().save_action_super();
    }

    /// Java `getParameters(ParallelParam)`; empty in `ProcessDialog`.
    fn get_parameters(&self, _param: &mut dyn ParallelParam) {}
}

/// Java public abstract class `ProcessDialog implements
/// AbstractParallelDialog`.
pub struct ProcessDialog {
    /// This dialog's own handle (Java `this`, for the action adapters).
    self_ref: Weak<ProcessDialog>,
    /// The subclass object, for virtual dispatch.
    this: RefCell<Weak<dyn ProcessDialogVirtual>>,

    /// Java package-private `applicationManager`.
    pub application_manager: &'static ApplicationManager,
    /// Java package-private `axisID`.
    pub axis_id: AxisID,
    /// Java package-private `dialogType`.
    pub dialog_type: DialogType,
    /// Java `rootPanel = new EtomoPanel()`.
    pub root_panel: Rc<EtomoPanel>,
    // Exit buttons
    /// Java `pnlExitButtons = new JPanel()`.
    pub pnl_exit_buttons: Rc<JComponent>,
    /// Java `btnCancel = new SingleLineButton("Cancel")`.
    pub btn_cancel: Rc<SingleLineButton>,
    /// Java `btnExecute = new SingleLineButton("Execute")`.
    pub btn_execute: Rc<SingleLineButton>,
    /// Java `btnAdvanced = GlobalExpandButton.getInstance("Advanced", "Basic")`.
    pub btn_advanced: Rc<GlobalExpandButton>,

    /// Java `btnPostpone`; null unless the dialog uses Postpone.
    pub btn_postpone: Option<Rc<SingleLineButton>>,

    /// Java `exitState`, initialised to `DialogExitState.SAVE`.
    exit_state: Cell<DialogExitState>,
    /// Java `displayed`, initialised to false.
    displayed: Cell<bool>,
}

impl ProcessDialog {
    /// Java constructor `ProcessDialog(ApplicationManager, AxisID, DialogType)`.
    pub fn new_application_manager_axis_id_dialog_type(
        app_manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<ProcessDialog> {
        Self::new_application_manager_axis_id_dialog_type_boolean(
            app_manager,
            axis_id,
            dialog_type,
            true,
        )
    }

    /// Java constructor `ProcessDialog(ApplicationManager, AxisID, DialogType,
    /// boolean)`.  Create a new process dialog with a set of exit buttons
    /// (cancel, postpone, execute, and advanced) available for use.  The
    /// action adapters for the buttons are already implemented.
    pub fn new_application_manager_axis_id_dialog_type_boolean(
        app_manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        use_postpone: bool,
    ) -> Rc<ProcessDialog> {
        // Java field initialisers.
        let root_panel = EtomoPanel::new();
        let pnl_exit_buttons = JComponent::new_panel();
        let btn_cancel = SingleLineButton::new_string(Some("Cancel"));
        let btn_execute = SingleLineButton::new_string(Some("Execute"));
        let btn_advanced = GlobalExpandButton::get_instance(Some("Advanced"), Some("Basic"));

        eprintln!(
            "\n{}\nDialog: {}",
            utilities::get_date_time_stamp(),
            dialog_type
        );
        let displayed = true;
        // Get the default initial advanced state - dialog must set themselves up
        // according to this state.
        btn_advanced.change_state(app_manager.is_advanced(dialog_type, axis_id));
        // Upstream bug fixed in translation (ProcessDialog.java:78-84): the Java
        // calls setToolTipText() before assigning btnPostpone, so its
        // `btnPostpone != null` branch never runs and the Postpone button never
        // gets its tooltip.  The button is created first here, so the tooltip the
        // source writes for it is set.
        let btn_postpone = if use_postpone {
            Some(SingleLineButton::new_string(Some("Postpone")))
        } else {
            None
        };

        let this = Rc::new_cyclic(|self_ref| ProcessDialog {
            self_ref: self_ref.clone(),
            this: RefCell::new(Weak::<NoSubclass>::new() as Weak<dyn ProcessDialogVirtual>),
            application_manager: app_manager,
            axis_id,
            dialog_type,
            root_panel,
            pnl_exit_buttons,
            btn_cancel,
            btn_execute,
            btn_advanced,
            btn_postpone,
            exit_state: Cell::new(DialogExitState::Save),
            displayed: Cell::new(displayed),
        });
        this.set_tool_tip_text();

        // Layout the buttons
        // Swing layout: pnlExitButtons.setLayout(new BoxLayout(pnlExitButtons,
        // BoxLayout.X_AXIS)); horizontal glue between and around the buttons.
        this.pnl_exit_buttons.add(&this.btn_cancel.get_component());
        if let Some(btn_postpone) = &this.btn_postpone {
            this.pnl_exit_buttons.add(&btn_postpone.get_component());
        }
        this.pnl_exit_buttons.add(&this.btn_execute.get_component());
        this.pnl_exit_buttons
            .add(&this.btn_advanced.get_component());

        // Swing layout: UIUtilities.setButtonSizeAll(pnlExitButtons,
        // UIParameters.getInstance().getNarrowButtonDimension()).

        // Exit action listeners
        // Java `new buttonCancelActionAdapter(this)`.
        let weak = this.self_ref.clone();
        this.btn_cancel
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = weak.upgrade() {
                    adaptee.button_cancel_action(Some(event));
                }
            }));
        if let Some(btn_postpone) = &this.btn_postpone {
            // Java `new buttonPostponeActionAdapter(this)`.
            let weak = this.self_ref.clone();
            btn_postpone.add_action_listener(Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = weak.upgrade() {
                    adaptee.button_postpone_action(Some(event));
                }
            }));
        }
        // Java `new buttonExecuteActionAdapter(this)`.
        let weak = this.self_ref.clone();
        this.btn_execute
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = weak.upgrade() {
                    adaptee.button_execute_action();
                }
            }));
        this
    }

    /// Installs the subclass object for virtual dispatch (Rust-only; the Java
    /// `this` is the subclass object from the start).
    pub fn set_this(&self, this: Weak<dyn ProcessDialogVirtual>) {
        *self.this.borrow_mut() = this;
    }

    /// The subclass object (Java `this` seen through a virtual call).
    fn this(&self) -> Option<Rc<dyn ProcessDialogVirtual>> {
        self.this.borrow().upgrade()
    }

    /// Java abstract `done()`, dispatched to the subclass.
    pub fn done(&self) {
        if let Some(this) = self.this() {
            this.done();
        }
    }

    /// Java `setExitState(DialogExitState)`.
    pub fn set_exit_state(&self, exit_state: DialogExitState) {
        self.exit_state.set(exit_state);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.get_component()
    }

    /// Java `addExitButtons()`.
    pub fn add_exit_buttons(&self) {
        // Swing layout: rootPanel.add(Box.createVerticalGlue());
        // rootPanel.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.root_panel.get_component().add(&self.pnl_exit_buttons);
    }

    /// Java `isAdvanced()`.
    pub fn is_advanced(&self) -> bool {
        self.btn_advanced.is_expanded()
    }

    /// Java `setDisplayed(boolean)`.
    pub fn set_displayed(&self, displayed: bool) {
        self.displayed.set(displayed);
    }

    /// Java `isDisplayed()`.
    pub fn is_displayed(&self) -> bool {
        self.displayed.get()
    }

    /// Java `buttonCancelAction(ActionEvent)`, dispatched to the subclass.
    pub fn button_cancel_action(&self, event: Option<&ActionEvent>) {
        match self.this() {
            Some(this) => this.button_cancel_action(event),
            None => self.button_cancel_action_super(event),
        }
    }

    /// The `ProcessDialog` body of Java `buttonCancelAction(ActionEvent)`.
    pub fn button_cancel_action_super(&self, _event: Option<&ActionEvent>) {
        utilities::button_timestamp_container(Some("cancel"), Some(&self.dialog_type.to_string()));
        self.exit_state.set(DialogExitState::Cancel);
        self.done();
    }

    /// Java `buttonPostponeAction(ActionEvent)`, dispatched to the subclass.
    pub fn button_postpone_action(&self, event: Option<&ActionEvent>) {
        match self.this() {
            Some(this) => this.button_postpone_action(event),
            None => self.button_postpone_action_super(event),
        }
    }

    /// The `ProcessDialog` body of Java `buttonPostponeAction(ActionEvent)`.
    pub fn button_postpone_action_super(&self, _event: Option<&ActionEvent>) {
        utilities::button_timestamp_container(
            Some("postpone"),
            Some(&self.dialog_type.to_string()),
        );
        self.exit_state.set(DialogExitState::Postpone);
        self.done();
    }

    /// Java `queueTableEventAction(QueueTableEvent)`; empty.
    pub fn queue_table_event_action(&self, _event: &QueueTableEvent) {}

    /// Java `addQueueTableListener(QueueTableListener)`; empty.
    pub fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java `removeQueueTableListener(QueueTableListener)`; empty.
    pub fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}

    /// Java `buttonExecuteAction()`, dispatched to the subclass.
    pub fn button_execute_action(&self) -> bool {
        match self.this() {
            Some(this) => this.button_execute_action(),
            None => self.button_execute_action_super(),
        }
    }

    /// The `ProcessDialog` body of Java `buttonExecuteAction()`.
    pub fn button_execute_action_super(&self) -> bool {
        utilities::button_timestamp_container(Some("done"), Some(&self.dialog_type.to_string()));
        self.exit_state.set(DialogExitState::Execute);
        self.done();
        true
    }

    /// Java `saveAction()`, dispatched to the subclass.
    pub fn save_action(&self) {
        match self.this() {
            Some(this) => this.save_action(),
            None => self.save_action_super(),
        }
    }

    /// The `ProcessDialog` body of Java `saveAction()`.
    pub fn save_action_super(&self) {
        utilities::timestamp_command_status(Some("save"), Some(&self.dialog_type.to_string()));
        self.exit_state.set(DialogExitState::Save);
        self.done();
    }

    /// Java `getExitState()`.
    pub fn get_exit_state(&self) -> DialogExitState {
        self.exit_state.get()
    }

    /// Java private `setToolTipText()`.  Default tool tip text for the buttons.
    fn set_tool_tip_text(&self) {
        let mut line1 = "This button will abort any changes to the parameters ";
        let mut line2 = "in this dialog box and return you to the main window.";
        self.btn_cancel
            .set_tool_tip_text(Some(&format!("{line1}{line2}")));
        let mut line3;
        let mut line4;
        if let Some(btn_postpone) = &self.btn_postpone {
            line1 = "This button will save any changes to the parameters ";
            line2 = "in this dialog box and return you to the main window ";
            line3 = "without executing any of the processing.  Any parameter ";
            line4 = "changes will also be written to the com scripts.";
            btn_postpone.set_tool_tip_text(Some(&format!("{line1}{line2}{line3}{line4}")));
        }
        line1 = "This button will save any changes to the parameters ";
        line2 = "in this dialog box and execute the specified operation ";
        line3 = "on the data.  Any parameter changes will also be written ";
        line4 = "to the com scripts.";
        self.btn_execute
            .set_tool_tip_text(Some(&format!("{line1}{line2}{line3}{line4}")));

        line1 = "This button will present a more detailed set of ";
        line2 = "options for each of the underlying processes.";
        self.btn_advanced
            .set_tool_tip_text(Some(&format!("{line1}{line2}")));
    }
}

impl AbstractParallelDialog for ProcessDialog {
    /// Java `getParameters(ParallelParam)`, dispatched to the subclass (empty
    /// in `ProcessDialog`).
    fn get_parameters(&self, param: &mut dyn ParallelParam) {
        if let Some(this) = self.this() {
            this.get_parameters(param);
        }
    }

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        self.dialog_type
    }
}

/// Placeholder type for the empty `Weak<dyn ProcessDialogVirtual>` held
/// before the subclass installs itself.
struct NoSubclass;

impl ProcessDialogVirtual for NoSubclass {
    fn process_dialog(&self) -> &ProcessDialog {
        unreachable!("an empty Weak never upgrades")
    }
    fn done(&self) {}
}
