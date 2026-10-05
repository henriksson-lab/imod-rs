//! `IMOD/Etomo/src/etomo/ui/swing/BusyStatusPanel.java`.
//!
//! The busy icon of a startup dialog: a `JLabel` named `lb.busy`, enabled while a
//! process runs on its axis.  An event dispatch thread object (`Rc`); the manager's
//! busy status mediator holds it as a listener through an `EdtRef`.

use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::logic::busy_status_mediator::BusyStatusListener;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::event_queue::{self, EdtRef};

/// Java `LABEL`.
pub const LABEL: &str = "lb.busy";
/// Java package-private `ICON = CompleteIcon.createIcon("busy.png")`; the icon is
/// the Slint side's.
pub const ICON: &str = "busy.png";

/// Java `public final class BusyStatusPanel implements BusyStatusListener`.
pub struct BusyStatusPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `lBusyStatus = new JLabel(ICON)`.
    l_busy_status: Rc<JComponent>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// `this`, for the listener registration and the posted `SetBusyStatus`.
    this: Weak<BusyStatusPanel>,
    /// The `EdtRef` registered with the manager, so `removeListeners` removes the
    /// same listener (Java removes by identity).
    listener: RefCell<Option<Arc<EdtRef<dyn BusyStatusListener>>>>,
}

impl BusyStatusPanel {
    /// Java private `BusyStatusPanel(AxisID)`.
    fn new(axis_id: AxisID) -> Rc<BusyStatusPanel> {
        Rc::new_cyclic(|this| BusyStatusPanel {
            pnl_root: JComponent::new_panel(),
            l_busy_status: JComponent::new_label(""),
            axis_id,
            this: this.clone(),
            listener: RefCell::new(None),
        })
    }

    /// Java static package-private `getInstance(BaseManager, AxisID)`.
    pub fn get_instance(manager: &'static dyn BaseManager, axis_id: AxisID) -> Rc<BusyStatusPanel> {
        let instance = BusyStatusPanel::new(axis_id);
        instance.create_panel();
        instance.add_listeners(manager);
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Swing layout: BorderLayout with hgap 2.
        self.l_busy_status.set_name(Some(LABEL));
        self.l_busy_status.set_enabled(false);
        // Root
        // Swing layout: pnlRoot.add(lBusyStatus, BorderLayout.EAST).
        self.pnl_root.add(&self.l_busy_status);
    }

    /// Java package-private `addListeners(BaseManager)`.
    pub fn add_listeners(&self, manager: &'static dyn BaseManager) {
        let Some(this) = self.this.upgrade() else {
            return;
        };
        let listener: Arc<EdtRef<dyn BusyStatusListener>> =
            Arc::new(EdtRef::new(this as Rc<dyn BusyStatusListener>));
        *self.listener.borrow_mut() = Some(Arc::clone(&listener));
        manager.add_busy_status_listener(Some(listener));
    }

    /// Java package-private `removeListeners(BaseManager)`.
    pub fn remove_listeners(&self, manager: &'static dyn BaseManager) {
        let listener = self.listener.borrow().clone();
        manager.remove_busy_status_listener(listener.as_ref());
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl BusyStatusListener for BusyStatusPanel {
    /// Java `msgBusyStatusChanged(AxisID, boolean)`.
    fn msg_busy_status_changed(&self, axis_id: AxisID, process_status: bool) {
        if self.axis_id.is_same_axis(Some(axis_id)) {
            // SwingUtilities.invokeLater(new SetBusyStatus(processStatus))
            let Some(this) = self.this.upgrade() else {
                return;
            };
            let this = EdtRef::new(this);
            event_queue::invoke_later(move || {
                // Java private final class `SetBusyStatus.run()`.
                this.get().l_busy_status.set_enabled(process_status);
            });
        }
    }
}
