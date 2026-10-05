//! `IMOD/Etomo/src/etomo/ui/swing/MainFrame_AboutBox.java` (the Java class is
//! `MainFrame_AboutBox`; the Rust name drops the underscore).
//!
//! The Help > About dialog: the eTomo version, the IMOD build information and the
//! PEET version, with an OK button.
//!
//! Java `class MainFrame_AboutBox extends JDialog`: the dialog is the Swing
//! stand-in's `jdk::JDialog` (content pane, title, visibility; the Slint window
//! draws a showing one over the frame), with the modality the Java sets after
//! construction kept here.  Layout (`BorderLayout`, `BoxLayout`, rigid areas,
//! `setResizable`, `pack`) is not modelled.  The overridden `processWindowEvent`
//! is reached through a window listener on `WINDOW_CLOSING`, which the stand-in
//! delivers before its default close operation, as `Window.processWindowEvent`
//! does from the override's `super` call.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::imod::etomo::jdk::{ActionEvent, JComponent, JDialog, WindowListener};
use crate::imod::etomo::logic::version_control;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::imod_version;

thread_local! {
    /// The displayable About boxes.  The Java object is the `JDialog` itself,
    /// which AWT's window list keeps until `dispose()`; here the wrapper holding
    /// the dialog's listeners is kept the same way.
    static DISPLAYABLE: RefCell<Vec<Rc<MainFrameAboutBox>>> = const { RefCell::new(Vec::new()) };
}

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `WindowEvent.WINDOW_CLOSING`.
pub const WINDOW_CLOSING: i32 = 201;

/// Java package-private `class MainFrame_AboutBox extends JDialog`.
pub struct MainFrameAboutBox {
    /// The `JDialog` itself (`super(parent)`); its content pane is the Java
    /// `pnlRoot`.
    dialog: Rc<JDialog>,
    /// `Dialog.setModal`.
    modal: Cell<bool>,
    /// Java `pnlAbout`.
    pnl_about: Rc<JComponent>,
    /// Java `btnOK`.
    btn_ok: Rc<JComponent>,
}

impl MainFrameAboutBox {
    /// Java `MainFrame_AboutBox(Frame, AxisID)`.  `parent` is the owner frame
    /// (`super(parent)`), which only positions the dialog.
    pub fn new(_parent: &Rc<JComponent>, axis_id: AxisID) -> Rc<MainFrameAboutBox> {
        let instance = Rc::new(MainFrameAboutBox {
            // A JDialog is created invisible (and, with `super(parent)`, not modal).
            dialog: JDialog::new("", false),
            modal: Cell::new(false),
            pnl_about: JComponent::new_panel(),
            btn_ok: JComponent::new_button("OK"),
        });
        // `processWindowEvent` override, reached on WINDOW_CLOSING.
        struct Closing(Weak<MainFrameAboutBox>);
        impl WindowListener for Closing {
            fn window_closing(&self) {
                if let Some(about_box) = self.0.upgrade() {
                    about_box.process_window_event(WINDOW_CLOSING);
                }
            }
        }
        instance
            .dialog
            .add_window_listener(Rc::new(Closing(Rc::downgrade(&instance))));
        let pnl_root = instance.dialog.get_content_pane();
        let pnl_text = JComponent::new_panel();
        let pnl_button = JComponent::new_panel();
        // Swing layout: pnlRoot.setLayout(new BorderLayout()).
        instance.set_title("About");
        // Swing layout: setResizable(false).

        // Swing layout: pnlText and pnlAbout get vertical BoxLayouts.

        let lbl_etomo = JComponent::new_label("Etomo: The IMOD Tomography GUI");
        let lbl_version = JComponent::new_label(&format!(
            "Version {} {}",
            imod_version::CURRENT_VERSION,
            version_control::TIME_STAMP
        ));
        let lbl_authors = JComponent::new_label("Written by: Rick Gaudette & Sue Held");

        // btnOK.addActionListener(new AboutActionListener(this))
        let adaptee: Weak<MainFrameAboutBox> = Rc::downgrade(&instance);
        instance.btn_ok.add_action_listener(Rc::new(move |event| {
            // AboutActionListener.actionPerformed
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.button_action(event);
            }
        }));

        // Swing layout: pnlAbout.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_text.add(&lbl_etomo);
        // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y5)).
        pnl_text.add(&lbl_version);
        let imod_info: Option<Vec<String>> = version_control::get_imod_info(axis_id);
        if let Some(imod_info) = imod_info.as_ref().filter(|imod_info| imod_info.len() > 2) {
            // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y10)).
            pnl_text.add(&JComponent::new_label(&imod_info[1]));
            // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y5)).
            pnl_text.add(&JComponent::new_label(&imod_info[2]));
        }
        // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_text.add(&lbl_authors);
        if let Some(imod_info) = imod_info.as_ref().filter(|imod_info| !imod_info.is_empty()) {
            // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y20)).
            pnl_text.add(&JComponent::new_label(&format!(
                "IMOD Version: {}",
                imod_info[0]
            )));
        }
        let version: Option<String> = version_control::get_peet_version();
        if version.is_some() {
            // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y5)).
            // Java string concatenation writes a null version as "null".
            pnl_text.add(&JComponent::new_label(&format!(
                "PEET Version: {}",
                version_control::get_peet_version()
                    .as_deref()
                    .unwrap_or("null")
            )));
        }
        // Swing layout: pnlText.add(Box.createRigidArea(FixedDim.x0_y10)).
        pnl_button.add(&instance.btn_ok);

        instance.pnl_about.add(&pnl_text);
        instance.pnl_about.add(&pnl_button);
        // pnlRoot.add(pnlAbout, BorderLayout.CENTER)
        pnl_root.add(&instance.pnl_about);
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x20_y0),
        // BorderLayout.WEST) and the same at BorderLayout.EAST; pack().
        instance
    }

    /// Java `@Override processWindowEvent(WindowEvent)`: overridden so we can exit
    /// when window is closed.  `event_id` is `e.getID()`.
    pub fn process_window_event(&self, event_id: i32) {
        if event_id == WINDOW_CLOSING {
            self.cancel();
        }
        // super.processWindowEvent(e): the window listeners (none are registered).
    }

    /// Java `cancel()`.  Close the dialog.
    pub fn cancel(&self) {
        self.dispose();
    }

    /// Java `buttonAction(ActionEvent)`.  Close the dialog on a button event.
    pub fn button_action(&self, e: &ActionEvent) {
        if Rc::ptr_eq(e.get_source(), &self.btn_ok) {
            self.cancel();
        }
    }

    // --- javax.swing.JDialog / java.awt.Dialog / java.awt.Window members ---

    /// Java `JDialog.getContentPane()`.
    pub fn get_content_pane(&self) -> Rc<JComponent> {
        self.dialog.get_content_pane()
    }

    /// Java `Dialog.setTitle(String)`.
    pub fn set_title(&self, title: &str) {
        self.dialog.set_title(title);
    }

    /// Java `Dialog.getTitle()`.
    pub fn get_title(&self) -> String {
        self.dialog.get_title()
    }

    /// Java `Dialog.setModal(boolean)`.
    pub fn set_modal(&self, modal: bool) {
        self.modal.set(modal);
    }

    /// Java `Dialog.isModal()`.
    pub fn is_modal(&self) -> bool {
        self.modal.get()
    }

    /// Java `Dialog.setVisible(boolean)`.  (A modal Swing dialog blocks here until
    /// it is closed; the stand-in returns, and nothing follows the call.)
    pub fn set_visible(self: &Rc<Self>, visible: bool) {
        if visible {
            DISPLAYABLE.with(|displayable| {
                let mut displayable = displayable.borrow_mut();
                if !displayable
                    .iter()
                    .any(|about_box| Rc::ptr_eq(about_box, self))
                {
                    displayable.push(self.clone());
                }
            });
        }
        self.dialog.set_visible(visible);
    }

    /// Java `Window.isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.dialog.is_visible()
    }

    /// Java `Window.dispose()`.
    pub fn dispose(&self) {
        self.dialog.dispose();
        DISPLAYABLE.with(|displayable| {
            displayable
                .borrow_mut()
                .retain(|about_box| !std::ptr::eq(about_box.as_ref(), self))
        });
    }
}
