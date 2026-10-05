//! `IMOD/Etomo/src/etomo/ui/swing/ContextPopup.java`.
//!
//! The right-click menu every eTomo dialog and panel opens: man pages (shown
//! by the `imodqtassist` help viewer), log files (`TextPageWindow`), tabbed
//! log file sets (`TabbedTextWindow`), `tomodataplots` graphs, the guides, and
//! PEET's `PEETHelp` program.
//!
//! The Java compares `actionEvent.getActionCommand() == item.getText()` by
//! reference.  An item's action command is its text object, so the test is
//! "this item fired the event"; it is translated as an identity test on the
//! event source, which is what the reference comparison decides for every
//! item (two items with equal text stay distinct, as they do in Java).

use std::cell::{Cell, OnceCell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

use super::constants::TOP_ANCHOR;
use super::menu_item::MenuItem;
use super::tabbed_text_window::TabbedTextWindow;
use super::text_page_window::TextPageWindow;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::tomodataplots_param::Task;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent};
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::imodqtassist_process;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::environment_variable::{self, PARTICLE_DIR};

/// Java `public static final String TOMO_GUIDE`.
pub const TOMO_GUIDE: &str = "tomoguide.html";
/// Java `public static final String JOIN_GUIDE`.
pub const JOIN_GUIDE: &str = "tomojoin.html";
/// Java `public static final String SERIAL_GUIDE`.
pub const SERIAL_GUIDE: &str = "serialalign.html";
/// Java `public static final String BATCHRUNTOMO_GUIDE`.
pub const BATCHRUNTOMO_GUIDE: &str = "batchGuide.html";
/// Java `public static final String ALIGNFRAMES_GUIDE`.
pub const ALIGNFRAMES_GUIDE: &str = "alignframesGuide.html";
/// Java `private static final String TOMO_GUIDE_LABEL`.
const TOMO_GUIDE_LABEL: &str = "Tomography Guide";

thread_local! {
    /// Stand-in for Swing's popup layer.  Java discards the `ContextPopup` after
    /// constructing it; the showing `JPopupMenu` keeps its items, their
    /// listener and through it this object alive until the popup is gone.
    /// Swing shows one popup menu at a time, so a new popup replaces the old.
    static SHOWING: RefCell<Option<Rc<ContextPopup>>> = const { RefCell::new(None) };
}

/// Java `public final class ContextPopup`.
pub struct ContextPopup {
    /// Java `private final JPopupMenu contextMenu = new JPopupMenu("Help Documents")`.
    context_menu: Rc<JComponent>,
    serial_sections_guide_item: Rc<MenuItem>,
    model_guide_item: Rc<MenuItem>,
    it_3dmod_guide: Rc<MenuItem>,
    etomo_guide_item: Rc<MenuItem>,
    join_guide_item: Rc<MenuItem>,
    peet_guide_item: Rc<MenuItem>,
    peet_help_item: Rc<MenuItem>,
    batch_guide_item: Rc<MenuItem>,
    alignframes_guide_item: Rc<MenuItem>,

    /// Java `private final ActionListener actionListener`, assigned once in
    /// every constructor.
    action_listener: OnceCell<ActionListener>,
    /// Java `private final MouseEvent mouseEvent`.
    mouse_event: MouseEvent,

    tomo_guide_item: RefCell<Rc<MenuItem>>,
    man_page_name: RefCell<Option<Vec<String>>>,
    log_file_name: RefCell<Option<Vec<String>>>,
    man_page_item: RefCell<Option<Vec<Rc<MenuItem>>>>,
    log_file_item: RefCell<Option<Vec<Rc<MenuItem>>>>,
    log_file_set_item: RefCell<Option<Vec<Rc<MenuItem>>>>,
    // private String imodURL = null;
    anchor: RefCell<Option<String>>,
    tomo_guide_alt_label: RefCell<Option<String>>,
    graph_item: RefCell<Option<Vec<Rc<MenuItem>>>>,
    /// Java `private List<TaskInterface> graphTask`; every element is a
    /// `TomodataplotsParam.Task`.
    graph_task: RefCell<Option<Vec<Task>>>,
    serial_sections: Cell<bool>,
    anchor2: RefCell<Option<String>>,
    tomo_guide_alt_label2: RefCell<Option<String>>,
    tomo_guide_item2: RefCell<Option<Rc<MenuItem>>>,
}

impl ContextPopup {
    /// The Java field initializers, which run first in every constructor.
    fn init_fields(mouse_event: &MouseEvent) -> ContextPopup {
        ContextPopup {
            context_menu: JComponent::new_popup_menu("Help Documents"),
            serial_sections_guide_item: MenuItem::new_string("Serial Section Guide ..."),
            model_guide_item: MenuItem::new_string("IMOD Users Guide ..."),
            it_3dmod_guide: MenuItem::new_string("3dmod Users Guide ..."),
            etomo_guide_item: MenuItem::new_string("Etomo Users Guide ..."),
            join_guide_item: MenuItem::new_string("Join Users Guide ..."),
            peet_guide_item: MenuItem::new_string("PEET Users Guide ..."),
            peet_help_item: MenuItem::new_string("PEET Help ..."),
            batch_guide_item: MenuItem::new_string("Batch Interface Guide ..."),
            alignframes_guide_item: MenuItem::new_string("Align Frames Guide ..."),
            action_listener: OnceCell::new(),
            mouse_event: mouse_event.clone(),
            tomo_guide_item: RefCell::new(MenuItem::new_string(&format!(
                "{} ...",
                TOMO_GUIDE_LABEL
            ))),
            man_page_name: RefCell::new(None),
            log_file_name: RefCell::new(None),
            man_page_item: RefCell::new(None),
            log_file_item: RefCell::new(None),
            log_file_set_item: RefCell::new(None),
            anchor: RefCell::new(None),
            tomo_guide_alt_label: RefCell::new(None),
            graph_item: RefCell::new(None),
            graph_task: RefCell::new(None),
            serial_sections: Cell::new(false),
            anchor2: RefCell::new(None),
            tomo_guide_alt_label2: RefCell::new(None),
            tomo_guide_item2: RefCell::new(None),
        }
    }

    /// Java `ContextPopup(Component, MouseEvent, String, BaseManager, AxisID)`.
    ///
    /// Simple context popup constructor.  Only the default menu items are
    /// displayed.
    pub fn new_component_mouse_event_string_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Rc<ContextPopup> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let mut tomo_guide_location = TOMO_GUIDE.to_owned();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                tomo_guide_location.push_str(&format!("#{}", anchor));
            }
            this.global_item_action_action_event_string_base_manager_axis_id(
                action_event,
                Some(tomo_guide_location),
                manager,
                axis_id,
            );
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        // add the menu items
        this.add_standard_menu_items(false, None);
        this.show_menu(component);
        this
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, String[],
    /// String[], BaseManager, AxisID)`.
    ///
    /// Constructor to show a man page list in addition to the the standard menu
    /// items.  `Err` is Java's `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_string_array_string_array_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let guide_to_anchor = guide_to_anchor.map(str::to_owned);
        let listener_guide_to_anchor = guide_to_anchor.clone();
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let guide_to_anchor = listener_guide_to_anchor.clone();
            let mut guide_location = guide_to_anchor.clone();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                // Java string concatenation renders a null guide as "null".
                guide_location = Some(format!(
                    "{}#{}",
                    guide_location.as_deref().unwrap_or("null"),
                    anchor
                ));
            }

            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                }
            }

            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                guide_location,
                None,
                guide_to_anchor.as_deref(),
                manager,
                axis_id,
            );

            // Close the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, guide_to_anchor.as_deref());
        this.show_menu(component);
        Ok(this)
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, BaseManager,
    /// AxisID)`.
    ///
    /// Constructor to show the standard items with a anchor into one of the
    /// guides.
    pub fn new_component_mouse_event_string_string_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Rc<ContextPopup> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let guide_to_anchor = guide_to_anchor.map(str::to_owned);
        let listener_guide_to_anchor = guide_to_anchor.clone();
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let guide_to_anchor = listener_guide_to_anchor.clone();
            let mut guide_location = guide_to_anchor.clone();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                guide_location = Some(format!(
                    "{}#{}",
                    guide_location.as_deref().unwrap_or("null"),
                    anchor
                ));
            }
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                guide_location,
                None,
                guide_to_anchor.as_deref(),
                manager,
                axis_id,
            );
            // Close the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, guide_to_anchor.as_deref());
        this.show_menu(component);
        this
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, String[],
    /// String[], String[], String[], BaseManager, AxisID)`.
    ///
    /// Constructor to show a man page list and log file items in addition to
    /// the the standard menu items.  `Err` is Java's `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: Option<&[String]>,
        log_file: Option<&[String]>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        // Upstream bug fixed (ContextPopup.java:359): a null logFile with a
        // non-null logFileLabel throws NullPointerException; it is taken as
        // empty.
        if let Some(log_file_label) = log_file_label
            && log_file_label.len() != log_file.map_or(0, <[String]>::len)
        {
            let mut message = String::new();
            message.push_str(
                "log file label and log file arrays must be the same length\nlogFileLabel=\n",
            );
            for label in log_file_label {
                message.push_str(&format!("{}\n", label));
            }
            message.push_str("logFile=\n");
            for file in log_file.unwrap_or_default() {
                message.push_str(&format!("{}\n", file));
            }
            return Err(message);
        }

        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let guide_to_anchor = guide_to_anchor.map(str::to_owned);
        let listener_guide_to_anchor = guide_to_anchor.clone();
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let guide_to_anchor = listener_guide_to_anchor.clone();
            let mut guide_location = guide_to_anchor.clone();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                guide_location = Some(format!(
                    "{}#{}",
                    guide_location.as_deref().unwrap_or("null"),
                    anchor
                ));
            }
            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(
                    // getImodURL() + "man/" + getManPageName()[i]); manpage.setVisible(true);
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item().unwrap_or_default();
            let log_file_name = this.get_log_file_name().unwrap_or_default();
            for i in 0..log_file_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                    // TextPageWindow(): the font size comes from the first UIManager
                    // FontUIResource, which is not modelled; UIParameters' default stands in.
                    let mut log_file_window =
                        TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                    let visible = log_file_window.set_file_from_file_name(format!(
                        "{}{}{}",
                        manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        std::path::MAIN_SEPARATOR,
                        log_file_name[i]
                    ));
                    // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                    let _ = visible;
                }
            }

            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                guide_location,
                None,
                guide_to_anchor.as_deref(),
                manager,
                axis_id,
            );

            // Close the the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.add_log_file_menu_items(log_file_label, log_file);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, guide_to_anchor.as_deref());
        this.show_menu(component);
        Ok(this)
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, String, String,
    /// String[], String[], String[], String[], BaseManager, AxisID)`.
    ///
    /// Constructor to show a man page list and log file items in addition to
    /// the the standard menu items.  Contains two anchors, both of which are
    /// for the Tomography Guide.  `Err` is Java's `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        tomo_guide_label_suffix: Option<&str>,
        tomo_anchor2: Option<&str>,
        tomo_guide_label_suffix2: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: Option<&[String]>,
        log_file: Option<&[String]>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        // Upstream bug fixed (ContextPopup.java:445): a null logFile with a
        // non-null logFileLabel throws NullPointerException; it is taken as
        // empty.
        if let Some(log_file_label) = log_file_label
            && log_file_label.len() != log_file.map_or(0, <[String]>::len)
        {
            let mut message = String::new();
            message.push_str(
                "log file label and log file arrays must be the same length\nlogFileLabel=\n",
            );
            for label in log_file_label {
                message.push_str(&format!("{}\n", label));
            }
            message.push_str("logFile=\n");
            for file in log_file.unwrap_or_default() {
                message.push_str(&format!("{}\n", file));
            }
            return Err(message);
        }

        let guide_to_anchor = TOMO_GUIDE;
        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        *this.tomo_guide_alt_label.borrow_mut() =
            this.build_tomo_guide_alt_label(tomo_guide_label_suffix);
        *this.anchor2.borrow_mut() = tomo_anchor2.map(str::to_owned);
        *this.tomo_guide_alt_label2.borrow_mut() =
            this.build_tomo_guide_alt_label(tomo_guide_label_suffix2);

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let mut guide_location = guide_to_anchor.to_owned();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                guide_location.push_str(&format!("#{}", anchor));
            }
            let mut guide_location2 = guide_to_anchor.to_owned();
            let anchor2 = this.get_anchor2();
            if let Some(anchor2) = anchor2.as_deref()
                && !anchor2.is_empty()
            {
                guide_location2.push_str(&format!("#{}", anchor2));
            }

            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item().unwrap_or_default();
            let log_file_name = this.get_log_file_name().unwrap_or_default();
            for i in 0..log_file_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                    // TextPageWindow(): the font size comes from the first UIManager
                    // FontUIResource, which is not modelled; UIParameters' default stands in.
                    let mut log_file_window =
                        TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                    let visible = log_file_window.set_file_from_file_name(format!(
                        "{}{}{}",
                        manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        std::path::MAIN_SEPARATOR,
                        log_file_name[i]
                    ));
                    // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                    let _ = visible;
                }
            }

            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                Some(guide_location),
                Some(guide_location2),
                Some(guide_to_anchor),
                manager,
                axis_id,
            );

            // Close the the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.add_log_file_menu_items(log_file_label, log_file);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, Some(guide_to_anchor));
        this.show_menu(component);
        Ok(this)
    }

    /// Java private `buildTomoGuideAltLabel(String)`.
    fn build_tomo_guide_alt_label(&self, tomo_guide_label_suffix: Option<&str>) -> Option<String> {
        let tomo_guide_label_suffix = tomo_guide_label_suffix?;
        Some(format!(
            "{} ({}) ...",
            TOMO_GUIDE_LABEL, tomo_guide_label_suffix
        ))
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, String[],
    /// String[], String[], String[], TomodataplotsParam.Task[], File[],
    /// BaseManager, AxisID)`.
    ///
    /// Constructor to show a man page list and log file items in addition to
    /// the the standard menu items, plus graphs.  `Err` is Java's
    /// `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_task_array_file_array_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: &[String],
        log_file: &[String],
        graph: Option<&[Task]>,
        graph_input_file: Option<&[Option<PathBuf>]>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        if log_file_label.len() != log_file.len() {
            let mut message = String::new();
            message.push_str(
                "log file label and log file arrays must be the same length\nlogFileLabel=\n",
            );
            for label in log_file_label {
                message.push_str(&format!("{}\n", label));
            }
            message.push_str("logFile=\n");
            for file in log_file {
                message.push_str(&format!("{}\n", file));
            }
            return Err(message);
        }

        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let guide_to_anchor = guide_to_anchor.map(str::to_owned);
        let listener_guide_to_anchor = guide_to_anchor.clone();
        let listener_graph_input_file = graph_input_file.map(<[Option<PathBuf>]>::to_vec);
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            this.set_visible(false);
            let guide_to_anchor = listener_guide_to_anchor.clone();
            let graph_input_file = &listener_graph_input_file;
            let mut guide_location = guide_to_anchor.clone();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                guide_location = Some(format!(
                    "{}#{}",
                    guide_location.as_deref().unwrap_or("null"),
                    anchor
                ));
            }
            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(
                    // getImodURL() + "man/" + getManPageName()[i]); manpage.setVisible(true);
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                    return;
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item().unwrap_or_default();
            let log_file_name = this.get_log_file_name().unwrap_or_default();
            for i in 0..log_file_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                    // TextPageWindow(): the font size comes from the first UIManager
                    // FontUIResource, which is not modelled; UIParameters' default stands in.
                    let mut log_file_window =
                        TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                    let visible = log_file_window.set_file_from_file_name(format!(
                        "{}{}{}",
                        manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        std::path::MAIN_SEPARATOR,
                        log_file_name[i]
                    ));
                    // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                    let _ = visible;
                    return;
                }
            }

            let graph_item = this.get_graph_item();
            let graph_task = this.get_graph_task().unwrap_or_default();
            if let Some(graph_item) = graph_item {
                for i in 0..graph_item.len() {
                    if Rc::ptr_eq(action_event.get_source(), &graph_item[i].get_component()) {
                        let alternative_input_file = match graph_input_file {
                            Some(files) if files.len() > i => files[i].as_ref().map(|file| {
                                std::path::absolute(file)
                                    .unwrap_or_else(|_| file.clone())
                                    .to_string_lossy()
                                    .into_owned()
                            }),
                            _ => None,
                        };
                        manager.tomodataplots(
                            Some(&graph_task[i]),
                            Some(axis_id),
                            None,
                            alternative_input_file.as_deref(),
                        );
                        return;
                    }
                }
            }
            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                guide_location,
                None,
                guide_to_anchor.as_deref(),
                manager,
                axis_id,
            );
        });
        let _ = this.action_listener.set(action_listener);

        this.add_log_file_menu_items(Some(log_file_label), Some(log_file));
        if let Some(graph) = graph
            && !graph.is_empty()
        {
            this.context_menu.add(&JComponent::new_popup_separator());
            this.add_graph_menu_items(manager, axis_id, graph, graph_input_file);
        }
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, guide_to_anchor.as_deref());
        this.show_menu(component);
        Ok(this)
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, String[],
    /// String[], String[], String[], TomodataplotsParam.Task[], BaseManager,
    /// AxisID, boolean)`.
    ///
    /// Constructor to show a man page list and log file items in addition to
    /// the the standard menu items, plus graphs, optionally for serial
    /// sections.  `Err` is Java's `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_task_array_base_manager_axis_id_boolean(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: &[String],
        log_file: &[String],
        graph: Option<&[Task]>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        serial_sections: bool,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.serial_sections.set(serial_sections);
        this.validate(man_page_label, man_page)?;
        if log_file_label.len() != log_file.len() {
            let mut message = String::new();
            message.push_str(
                "log file label and log file arrays must be the same length\nlogFileLabel=\n",
            );
            for label in log_file_label {
                message.push_str(&format!("{}\n", label));
            }
            message.push_str("logFile=\n");
            for file in log_file {
                message.push_str(&format!("{}\n", file));
            }
            return Err(message);
        }

        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let guide_to_anchor = guide_to_anchor.map(str::to_owned);
        let listener_guide_to_anchor = guide_to_anchor.clone();
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            this.set_visible(false);
            let guide_to_anchor = listener_guide_to_anchor.clone();
            let mut guide_location = guide_to_anchor.clone();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                guide_location = Some(format!(
                    "{}#{}",
                    guide_location.as_deref().unwrap_or("null"),
                    anchor
                ));
            }
            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(
                    // getImodURL() + "man/" + getManPageName()[i]); manpage.setVisible(true);
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                    return;
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item().unwrap_or_default();
            let log_file_name = this.get_log_file_name().unwrap_or_default();
            for i in 0..log_file_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                    // TextPageWindow(): the font size comes from the first UIManager
                    // FontUIResource, which is not modelled; UIParameters' default stands in.
                    let mut log_file_window =
                        TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                    let visible = log_file_window.set_file_from_file_name(format!(
                        "{}{}{}",
                        manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        std::path::MAIN_SEPARATOR,
                        log_file_name[i]
                    ));
                    // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                    let _ = visible;
                    return;
                }
            }

            let graph_item = this.get_graph_item();
            let graph_task = this.get_graph_task().unwrap_or_default();
            if let Some(graph_item) = graph_item {
                for i in 0..graph_item.len() {
                    if Rc::ptr_eq(action_event.get_source(), &graph_item[i].get_component()) {
                        manager.tomodataplots(Some(&graph_task[i]), Some(axis_id), None, None);
                        return;
                    }
                }
            }
            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                guide_location,
                None,
                guide_to_anchor.as_deref(),
                manager,
                axis_id,
            );
        });
        let _ = this.action_listener.set(action_listener);

        this.add_log_file_menu_items(Some(log_file_label), Some(log_file));
        if let Some(graph) = graph
            && !graph.is_empty()
        {
            this.context_menu.add(&JComponent::new_popup_separator());
            this.add_graph_menu_items(manager, axis_id, graph, None);
        }
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, guide_to_anchor.as_deref());
        this.show_menu(component);
        Ok(this)
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String, String[],
    /// String[], String[], String[], BaseManager, AxisID, String)`.
    ///
    /// Constructor to show a man page list and log file items in addition to
    /// the the standard menu items; the .log files are in `subdirName`.  `Err`
    /// is Java's `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id_string(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: &[String],
        log_file: &[String],
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        subdir_name: Option<&str>,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        if log_file_label.len() != log_file.len() {
            let message = "log file label and log file arrays must be the same length";
            return Err(message.to_owned());
        }

        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let guide_to_anchor = guide_to_anchor.map(str::to_owned);
        let listener_guide_to_anchor = guide_to_anchor.clone();
        let subdir_name = subdir_name.map(str::to_owned);
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let guide_to_anchor = listener_guide_to_anchor.clone();
            let mut guide_location = guide_to_anchor.clone();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                guide_location = Some(format!(
                    "{}#{}",
                    guide_location.as_deref().unwrap_or("null"),
                    anchor
                ));
            }
            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(
                    // getImodURL() + "man/" + getManPageName()[i]); manpage.setVisible(true);
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item().unwrap_or_default();
            let log_file_name = this.get_log_file_name().unwrap_or_default();
            for i in 0..log_file_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                    // TextPageWindow(): the font size comes from the first UIManager
                    // FontUIResource, which is not modelled; UIParameters' default stands in.
                    let mut log_file_window =
                        TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                    let visible = log_file_window.set_file_from_file_name(format!(
                        "{}{}{}{}",
                        manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        match &subdir_name {
                            Some(subdir_name) =>
                                format!("{}{}", std::path::MAIN_SEPARATOR, subdir_name),
                            None => String::new(),
                        },
                        std::path::MAIN_SEPARATOR,
                        log_file_name[i]
                    ));
                    // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                    let _ = visible;
                }
            }

            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                guide_location,
                None,
                guide_to_anchor.as_deref(),
                manager,
                axis_id,
            );

            // Close the the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.add_log_file_menu_items(Some(log_file_label), Some(log_file));
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, guide_to_anchor.as_deref());
        this.show_menu(component);
        Ok(this)
    }

    /// Java `ContextPopup(Component, MouseEvent, String[], String[], boolean,
    /// BaseManager, AxisID)`.
    ///
    /// Shows a man page list and log file items in addition to the standard
    /// menu items (including the peet guide if required).  `Err` is Java's
    /// `IllegalArgumentException`.
    pub fn new_component_mouse_event_string_array_string_array_boolean_base_manager_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        man_page_label: &[String],
        man_page: &[String],
        add_peet_guide: bool,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let _anchor = this.get_anchor();
            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(
                    // getImodURL() + "man/" + getManPageName()[i]); manpage.setVisible(true);
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item();
            if let Some(log_file_item) = log_file_item {
                let log_file_name = this.get_log_file_name().unwrap_or_default();
                for i in 0..log_file_item.len() {
                    if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                        // TextPageWindow(): the font size comes from the first UIManager
                        // FontUIResource, which is not modelled; UIParameters' default stands in.
                        let mut log_file_window =
                            TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                        let visible = log_file_window.set_file_from_file_name(format!(
                            "{}{}{}",
                            manager
                                .get_property_user_dir()
                                .unwrap_or_else(|| "null".to_owned()),
                            std::path::MAIN_SEPARATOR,
                            log_file_name[i]
                        ));
                        // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                        let _ = visible;
                    }
                }
            }

            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                None,
                None,
                None,
                manager,
                axis_id,
            );

            // Close the the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(add_peet_guide, None);
        this.show_menu(component);
        Ok(this)
    }

    /// Java `ContextPopup(Component, MouseEvent, BaseManager, AxisID, boolean)`.
    ///
    /// Shows the standard menu items (the serial sections guide when
    /// `serialSections`).
    pub fn new_component_mouse_event_base_manager_axis_id_boolean(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        serial_sections: bool,
    ) -> Rc<ContextPopup> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.serial_sections.set(serial_sections);
        // calcImodURL();

        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let _anchor = this.get_anchor();
            let man_page_item = this.get_man_page_item();
            if let Some(man_page_item) = man_page_item {
                let man_page_name = this.get_man_page_name().unwrap_or_default();
                for i in 0..man_page_item.len() {
                    if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                        imodqtassist_process::INSTANCE.open(
                            Some(manager),
                            &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                            axis_id,
                        );
                    }
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item();
            if let Some(log_file_item) = log_file_item {
                let log_file_name = this.get_log_file_name().unwrap_or_default();
                for i in 0..log_file_item.len() {
                    if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                        // TextPageWindow(): the font size comes from the first UIManager
                        // FontUIResource, which is not modelled; UIParameters' default stands in.
                        let mut log_file_window =
                            TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                        let visible = log_file_window.set_file_from_file_name(format!(
                            "{}{}{}",
                            manager
                                .get_property_user_dir()
                                .unwrap_or_else(|| "null".to_owned()),
                            std::path::MAIN_SEPARATOR,
                            log_file_name[i]
                        ));
                        // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                        let _ = visible;
                    }
                }
            }

            // Search the standard items
            this.global_item_action_action_event_string_string_string_base_manager_axis_id(
                action_event,
                None,
                None,
                None,
                manager,
                axis_id,
            );

            // Close the the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, None);
        this.show_menu(component);
        this
    }

    /// Java `ContextPopup(Component, MouseEvent, String, String[], String[],
    /// String[], Vector, Vector, String[], String[], TomodataplotsParam.Task[],
    /// File[], ApplicationManager, String, AxisID)`.
    ///
    /// Constructor to show a man page list and tabbed log file items in
    /// addition to the the standard menu items.  `logFileSetLabel` and
    /// `logFileSet` are Vectors of `String[]`.  `updateLogCommandName` names the
    /// log that must be updated before it is displayed.  `Err` is Java's
    /// `IllegalArgumentException`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_component_mouse_event_string_string_array_string_array_string_array_vector_vector_string_array_string_array_task_array_file_array_application_manager_string_axis_id(
        component: &Rc<JComponent>,
        mouse_event: &MouseEvent,
        tomo_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_set_window_label: &[String],
        log_file_set_label: &[Vec<String>],
        log_file_set: &[Vec<String>],
        log_file_label: Option<&[String]>,
        log_file: Option<&[String]>,
        graph: Option<&[Task]>,
        graph_input_file: Option<&[Option<PathBuf>]>,
        application_manager: &'static ApplicationManager,
        update_log_command_name: &str,
        axis_id: AxisID,
    ) -> Result<Rc<ContextPopup>, String> {
        let this = Rc::new(ContextPopup::init_fields(mouse_event));
        this.validate(man_page_label, man_page)?;
        if log_file_set_label.len() != log_file_set.len() {
            let message = "log file label and log file vectors must be the same length";
            return Err(message.to_owned());
        }
        // Upstream bug fixed (ContextPopup.java:943): a null logFile with a
        // non-null logFileLabel throws NullPointerException; it is taken as
        // empty.
        if let Some(log_file_label) = log_file_label
            && log_file_label.len() != log_file.map_or(0, <[String]>::len)
        {
            let mut message = String::new();
            message.push_str(
                "log file label and log file arrays must be the same length\nlogFileLabel=\n",
            );
            for label in log_file_label {
                message.push_str(&format!("{}\n", label));
            }
            message.push_str("logFile=\n");
            for file in log_file.unwrap_or_default() {
                message.push_str(&format!("{}\n", file));
            }
            return Err(message);
        }

        *this.anchor.borrow_mut() = tomo_anchor.map(str::to_owned);
        // calcImodURL();

        let manager: &'static dyn BaseManager = application_manager;
        // Instantiate a new ActionListener to handle the menu selection
        let weak = Rc::downgrade(&this);
        let log_file_set_window_label_owned = log_file_set_window_label.to_vec();
        let log_file_set_label = log_file_set_label.to_vec();
        let log_file_set = log_file_set.to_vec();
        let listener_graph_input_file = graph_input_file.map(<[Option<PathBuf>]>::to_vec);
        let update_log_command_name = update_log_command_name.to_owned();
        let action_listener: ActionListener = Rc::new(move |action_event: &ActionEvent| {
            let Some(this) = weak.upgrade() else {
                return;
            };
            let graph_input_file = &listener_graph_input_file;
            let mut tomo_guide_location = TOMO_GUIDE.to_owned();
            let anchor = this.get_anchor();
            if let Some(anchor) = anchor.as_deref()
                && !anchor.is_empty()
            {
                tomo_guide_location.push_str(&format!("#{}", anchor));
            }

            let man_page_item = this.get_man_page_item().unwrap_or_default();
            let man_page_name = this.get_man_page_name().unwrap_or_default();
            for i in 0..man_page_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &man_page_item[i].get_component()) {
                    imodqtassist_process::INSTANCE.open(
                        Some(manager),
                        &format!("{}/{}", ProcessName::MAN, man_page_name[i]),
                        axis_id,
                    );
                }
            }

            // Search the logfile items
            let log_file_item = this.get_log_file_item().unwrap_or_default();
            let log_file_name = this.get_log_file_name().unwrap_or_default();
            for i in 0..log_file_item.len() {
                if Rc::ptr_eq(action_event.get_source(), &log_file_item[i].get_component()) {
                    // TextPageWindow(): the font size comes from the first UIManager
                    // FontUIResource, which is not modelled; UIParameters' default stands in.
                    let mut log_file_window =
                        TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
                    let visible = log_file_window.set_file_from_file_name(format!(
                        "{}{}{}",
                        application_manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        std::path::MAIN_SEPARATOR,
                        log_file_name[i]
                    ));
                    // Swing: logFileWindow.setVisible(visible) - the JFrame is not modelled.
                    let _ = visible;
                }
            }

            // Search the logfileset items
            let log_file_set_item = this.get_log_file_set_item().unwrap_or_default();
            for i in 0..log_file_set_item.len() {
                if Rc::ptr_eq(
                    action_event.get_source(),
                    &log_file_set_item[i].get_component(),
                ) {
                    if action_event
                        .get_action_command()
                        .is_some_and(|command| command.starts_with(&update_log_command_name))
                    {
                        application_manager.update_log(&update_log_command_name, axis_id);
                    }
                    // Create full path to the appropriate log file set items
                    let log_file_set_list = &log_file_set[i];
                    let mut log_file_set_full_path = vec![String::new(); log_file_set_list.len()];
                    let path = format!(
                        "{}{}",
                        application_manager
                            .get_property_user_dir()
                            .unwrap_or_else(|| "null".to_owned()),
                        std::path::MAIN_SEPARATOR
                    );
                    for j in 0..log_file_set_list.len() {
                        log_file_set_full_path[j] = format!("{}{}", path, log_file_set_list[j]);
                    }
                    let log_file_window =
                        TabbedTextWindow::new(log_file_set_window_label_owned[i].clone(), axis_id);
                    match log_file_window.open_files(
                        Some(manager),
                        &log_file_set_full_path,
                        &log_file_set_label[i],
                        axis_id,
                    ) {
                        Ok(true) => log_file_window.set_visible(true),
                        Ok(false) => log_file_window.dispose(),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            // catch (FileNotFoundException e)
                            eprintln!("{}", e);
                            // Java prints the array reference; the paths are printed.
                            eprintln!("File not file exception: {:?}", log_file_set_full_path);
                        }
                        Err(e) => {
                            // catch (IOException e)
                            eprintln!("{}", e);
                            eprintln!("IO exception: {:?}", log_file_set_full_path);
                        }
                    }
                    // catch (OutOfMemoryError e): dispose the window, tell the user
                    // "WARNING:  Ran out of memory.  Will not display log file.\n
                    // Please close open windows or exit Etomo." ("Out of Memory")
                    // and rethrow.  Rust aborts on allocation failure, so there is
                    // nothing to catch.
                }
            }

            let graph_item = this.get_graph_item();
            let graph_task = this.get_graph_task().unwrap_or_default();
            if let Some(graph_item) = graph_item {
                for i in 0..graph_item.len() {
                    if Rc::ptr_eq(action_event.get_source(), &graph_item[i].get_component()) {
                        let alternative_input_file = match graph_input_file {
                            Some(files) if files.len() > i => files[i].as_ref().map(|file| {
                                std::path::absolute(file)
                                    .unwrap_or_else(|_| file.clone())
                                    .to_string_lossy()
                                    .into_owned()
                            }),
                            _ => None,
                        };
                        application_manager.tomodataplots(
                            Some(&graph_task[i]),
                            Some(axis_id),
                            None,
                            alternative_input_file.as_deref(),
                        );
                        return;
                    }
                }
            }
            // Search the standard items
            this.global_item_action_action_event_string_base_manager_axis_id(
                action_event,
                Some(tomo_guide_location),
                manager,
                axis_id,
            );

            // Close the the menu
            this.set_visible(false);
        });
        let _ = this.action_listener.set(action_listener);

        this.add_tabbed_log_file_set_menu_items(log_file_set_window_label);
        this.add_log_file_menu_items(log_file_label, log_file);
        if let Some(graph) = graph
            && !graph.is_empty()
        {
            this.context_menu.add(&JComponent::new_popup_separator());
            this.add_graph_menu_items(manager, axis_id, graph, graph_input_file);
        }
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_man_page_menu_items(man_page_label, man_page);
        this.context_menu.add(&JComponent::new_popup_separator());
        this.add_standard_menu_items(false, None);
        this.show_menu(component);
        Ok(this)
    }

    /// Java private `addStandardMenuItems(boolean, String)`.
    fn add_standard_menu_items(&self, add_peet_guide: bool, guide_to_anchor: Option<&str>) {
        let action_listener = self
            .action_listener
            .get()
            .expect("actionListener is assigned before the menu items are added")
            .clone();
        let batch = guide_to_anchor == Some(BATCHRUNTOMO_GUIDE);
        let alignframes = guide_to_anchor == Some(ALIGNFRAMES_GUIDE);
        if batch {
            self.context_menu
                .add(&self.batch_guide_item.get_component());
            self.batch_guide_item
                .add_action_listener(action_listener.clone());
        } else if alignframes {
            self.context_menu
                .add(&self.alignframes_guide_item.get_component());
            self.alignframes_guide_item
                .add_action_listener(action_listener.clone());
        }
        // Construct the context menu
        if add_peet_guide {
            self.context_menu.add(&self.peet_guide_item.get_component());
            self.context_menu.add(&self.peet_help_item.get_component());
            self.peet_guide_item
                .add_action_listener(action_listener.clone());
            self.peet_help_item
                .add_action_listener(action_listener.clone());
            if !environment_variable::INSTANCE.exists(None, None, PARTICLE_DIR, None) {
                self.peet_help_item.set_enabled(false);
            }
        }
        if self.serial_sections.get() {
            self.context_menu
                .add(&self.serial_sections_guide_item.get_component());
            self.serial_sections_guide_item
                .add_action_listener(action_listener.clone());
        } else {
            // Currently only the tomo guide can have two entries which require
            // special labels.
            let tomo_guide_alt_label = self.tomo_guide_alt_label.borrow().clone();
            if let Some(tomo_guide_alt_label) = tomo_guide_alt_label {
                *self.tomo_guide_item.borrow_mut() = MenuItem::new_string(&tomo_guide_alt_label);
            }
            let tomo_guide_alt_label2 = self.tomo_guide_alt_label2.borrow().clone();
            if let Some(tomo_guide_alt_label2) = tomo_guide_alt_label2 {
                *self.tomo_guide_item2.borrow_mut() =
                    Some(MenuItem::new_string(&tomo_guide_alt_label2));
            }
            let tomo_guide_item = self.tomo_guide_item.borrow().clone();
            self.context_menu.add(&tomo_guide_item.get_component());
            tomo_guide_item.add_action_listener(action_listener.clone());
            let tomo_guide_item2 = self.tomo_guide_item2.borrow().clone();
            if let Some(tomo_guide_item2) = tomo_guide_item2 {
                self.context_menu.add(&tomo_guide_item2.get_component());
                tomo_guide_item2.add_action_listener(action_listener.clone());
            }
        }
        self.context_menu
            .add(&self.model_guide_item.get_component());
        self.model_guide_item
            .add_action_listener(action_listener.clone());
        self.context_menu.add(&self.it_3dmod_guide.get_component());
        self.it_3dmod_guide
            .add_action_listener(action_listener.clone());
        self.context_menu
            .add(&self.etomo_guide_item.get_component());
        self.etomo_guide_item
            .add_action_listener(action_listener.clone());
        if !self.serial_sections.get() {
            self.context_menu.add(&self.join_guide_item.get_component());
            self.join_guide_item
                .add_action_listener(action_listener.clone());
        }
        if !batch {
            self.context_menu
                .add(&self.batch_guide_item.get_component());
            self.batch_guide_item
                .add_action_listener(action_listener.clone());
        }
        if !alignframes {
            self.context_menu
                .add(&self.alignframes_guide_item.get_component());
            self.alignframes_guide_item
                .add_action_listener(action_listener);
        }
    }

    /// Java private `addManPageMenuItems(String[], String[])`.
    fn add_man_page_menu_items(&self, man_page_label: &[String], man_page: &[String]) {
        let action_listener = self
            .action_listener
            .get()
            .expect("actionListener is assigned before the menu items are added")
            .clone();
        let mut man_page_item: Vec<Rc<MenuItem>> = Vec::with_capacity(man_page_label.len());
        let mut man_page_name = vec![String::new(); man_page.len()];
        for i in 0..man_page_label.len() {
            let item = MenuItem::new_void();
            item.set_text(&format!("{} man page ...", man_page_label[i]));
            item.add_action_listener(action_listener.clone());
            self.context_menu.add(&item.get_component());
            man_page_item.push(item);
            man_page_name[i] = format!("{}{}", man_page[i], TOP_ANCHOR);
        }
        *self.man_page_item.borrow_mut() = Some(man_page_item);
        *self.man_page_name.borrow_mut() = Some(man_page_name);
    }

    /// Java private `globalItemAction(ActionEvent, String, BaseManager, AxisID)`.
    fn global_item_action_action_event_string_base_manager_axis_id(
        &self,
        action_event: &ActionEvent,
        tomo_guide_location: Option<String>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) {
        self.global_item_action_action_event_string_string_string_base_manager_axis_id(
            action_event,
            tomo_guide_location,
            None,
            Some(TOMO_GUIDE),
            manager,
            axis_id,
        );
    }

    /// Java private `globalItemAction(ActionEvent, String, String, String,
    /// BaseManager, AxisID)`.
    ///
    /// Open the appropriate file if the event is one of the global menu items.
    fn global_item_action_action_event_string_string_string_base_manager_axis_id(
        &self,
        action_event: &ActionEvent,
        mut guide_location: Option<String>,
        guide_location2: Option<String>,
        guide: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) {
        // Add TOP anchor when no anchor has been set.
        // `!guideLocation.matches("\\s*")`: Java `\s` is [ \t\n\x0B\f\r].
        if let Some(location) = guide_location.as_mut()
            && !location
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
            && !location.contains('#')
        {
            location.push_str(TOP_ANCHOR);
        }
        let source = action_event.get_source();
        let tomo_guide_item = self.tomo_guide_item.borrow().clone();
        let tomo_guide_item2 = self.tomo_guide_item2.borrow().clone();
        // `guideLocation` is null only when `guide` is; Java would hand null on.
        let guide_location_text = guide_location.as_deref().unwrap_or("null");
        if Rc::ptr_eq(source, &tomo_guide_item.get_component()) {
            // HTMLPageWindow manpage = new HTMLPageWindow(); if (guide.equals(
            // TOMO_GUIDE)) { manpage.openURL(imodURL + guideLocation); } else {
            // manpage.openURL(imodURL + TOMO_GUIDE); } manpage.setVisible(true);
            if guide == Some(TOMO_GUIDE) {
                imodqtassist_process::INSTANCE.open(Some(manager), guide_location_text, axis_id);
            } else {
                imodqtassist_process::INSTANCE.open(
                    Some(manager),
                    &format!("{}{}", TOMO_GUIDE, TOP_ANCHOR),
                    axis_id,
                );
            }
        } else if let Some(tomo_guide_item2) = &tomo_guide_item2
            && Rc::ptr_eq(source, &tomo_guide_item2.get_component())
        {
            if let Some(guide_location2) = guide_location2.as_deref()
                && guide == Some(TOMO_GUIDE)
            {
                imodqtassist_process::INSTANCE.open(Some(manager), guide_location2, axis_id);
            } else {
                imodqtassist_process::INSTANCE.open(
                    Some(manager),
                    &format!("{}{}", TOMO_GUIDE, TOP_ANCHOR),
                    axis_id,
                );
            }
        }

        if Rc::ptr_eq(source, &self.model_guide_item.get_component()) {
            // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(imodURL +
            // "guide.html"); manpage.setVisible(true);
            imodqtassist_process::INSTANCE.open(
                Some(manager),
                &format!("guide.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if Rc::ptr_eq(source, &self.it_3dmod_guide.get_component()) {
            // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(imodURL +
            // "3dmodguide.html"); manpage.setVisible(true);
            imodqtassist_process::INSTANCE.open(
                Some(manager),
                &format!("3dmodguide.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if Rc::ptr_eq(source, &self.etomo_guide_item.get_component()) {
            // HTMLPageWindow manpage = new HTMLPageWindow(); manpage.openURL(imodURL +
            // "UsingEtomo.html"); manpage.setVisible(true);
            imodqtassist_process::INSTANCE.open(
                Some(manager),
                &format!("UsingEtomo.html{}", TOP_ANCHOR),
                axis_id,
            );
        }

        if Rc::ptr_eq(source, &self.join_guide_item.get_component()) {
            // HTMLPageWindow manpage = new HTMLPageWindow(); if (guide.equals(
            // JOIN_GUIDE)) { manpage.openURL(imodURL + guideLocation); } else {
            // manpage.openURL(imodURL + JOIN_GUIDE); } manpage.setVisible(true);
            if guide == Some(JOIN_GUIDE) {
                imodqtassist_process::INSTANCE.open(Some(manager), guide_location_text, axis_id);
            } else {
                imodqtassist_process::INSTANCE.open(
                    Some(manager),
                    &format!("{}{}", JOIN_GUIDE, TOP_ANCHOR),
                    axis_id,
                );
            }
        }
        if Rc::ptr_eq(source, &self.batch_guide_item.get_component()) {
            if guide == Some(BATCHRUNTOMO_GUIDE) {
                imodqtassist_process::INSTANCE.open(Some(manager), guide_location_text, axis_id);
            } else {
                imodqtassist_process::INSTANCE.open(
                    Some(manager),
                    &format!("{}{}", BATCHRUNTOMO_GUIDE, TOP_ANCHOR),
                    axis_id,
                );
            }
        }
        if Rc::ptr_eq(source, &self.peet_guide_item.get_component()) {
            imodqtassist_process::INSTANCE.open(
                Some(manager),
                &format!("PEETmanual.html{}", TOP_ANCHOR),
                axis_id,
            );
        }
        if Rc::ptr_eq(source, &self.peet_help_item.get_component()) {
            // new File(new File(new File($PARTICLE_DIR), "bin"), "PEETHelp")
            // .getAbsolutePath(): a File built on the empty path resolves its
            // child against "/".
            let particle_dir =
                environment_variable::INSTANCE.get_value(None, None, PARTICLE_DIR, None);
            let bin = if particle_dir.is_empty() {
                PathBuf::from("/bin")
            } else {
                PathBuf::from(&particle_dir).join("bin")
            };
            let peet_help = bin.join("PEETHelp");
            let peet_help = std::path::absolute(&peet_help).unwrap_or(peet_help);
            BaseProcessManager::start_system_program_thread(
                vec![peet_help.to_string_lossy().into_owned()],
                axis_id,
                Some(manager),
            );
        }
        if Rc::ptr_eq(source, &self.serial_sections_guide_item.get_component()) {
            if guide == Some(SERIAL_GUIDE) {
                imodqtassist_process::INSTANCE.open(Some(manager), guide_location_text, axis_id);
            } else {
                imodqtassist_process::INSTANCE.open(
                    Some(manager),
                    &format!("{}{}", SERIAL_GUIDE, TOP_ANCHOR),
                    axis_id,
                );
            }
        }
        if Rc::ptr_eq(source, &self.alignframes_guide_item.get_component()) {
            if guide == Some(ALIGNFRAMES_GUIDE) {
                imodqtassist_process::INSTANCE.open(Some(manager), guide_location_text, axis_id);
            } else {
                imodqtassist_process::INSTANCE.open(
                    Some(manager),
                    &format!("{}{}", ALIGNFRAMES_GUIDE, TOP_ANCHOR),
                    axis_id,
                );
            }
        }
    }

    /// Java private `addLogFileMenuItems(String[], String[])`.
    ///
    /// Upstream bug fixed (ContextPopup.java:1273): Java dereferences both
    /// arrays unchecked although three constructors accept a null
    /// `logFileLabel`, so those popups throw NullPointerException while being
    /// built.  A null array is taken as empty (no log file items).
    fn add_log_file_menu_items(
        &self,
        log_file_label: Option<&[String]>,
        log_file: Option<&[String]>,
    ) {
        let action_listener = self
            .action_listener
            .get()
            .expect("actionListener is assigned before the menu items are added")
            .clone();
        let log_file_label = log_file_label.unwrap_or_default();
        let log_file = log_file.unwrap_or_default();
        let mut log_file_item: Vec<Rc<MenuItem>> = Vec::with_capacity(log_file_label.len());
        let mut log_file_name = vec![String::new(); log_file.len()];
        for i in 0..log_file_label.len() {
            let item = MenuItem::new_void();
            item.set_text(&format!("{} log file ...", log_file_label[i]));
            item.add_action_listener(action_listener.clone());
            self.context_menu.add(&item.get_component());
            log_file_item.push(item);
            log_file_name[i] = log_file[i].clone();
        }
        *self.log_file_item.borrow_mut() = Some(log_file_item);
        *self.log_file_name.borrow_mut() = Some(log_file_name);
    }

    /// Java private `addGraphMenuItems(BaseManager, AxisID,
    /// TomodataplotsParam.Task[], File[])`.
    fn add_graph_menu_items(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        graph: &[Task],
        graph_input_file: Option<&[Option<PathBuf>]>,
    ) {
        let action_listener = self
            .action_listener
            .get()
            .expect("actionListener is assigned before the menu items are added")
            .clone();
        let mut graph_task: Vec<Task> = Vec::new();
        for i in 0..graph.len() {
            // Upstream bug fixed (ContextPopup.java:1289): Java reads
            // `graphInputFile[i].exists()` without the length and null checks its
            // listeners make, so a short array or a null entry throws while the
            // popup is built.  Those checks are applied here too.
            if graph[i].is_available(manager, axis_id)
                || graph_input_file.is_some_and(|files| {
                    files.len() > i && files[i].as_ref().is_some_and(|file| file.exists())
                })
            {
                graph_task.push(graph[i]);
            }
        }
        let mut graph_item: Vec<Rc<MenuItem>> = Vec::with_capacity(graph_task.len());
        for task in &graph_task {
            let item = MenuItem::new_void();
            item.set_text(&task.to_string());
            item.add_action_listener(action_listener.clone());
            self.context_menu.add(&item.get_component());
            graph_item.push(item);
        }
        *self.graph_task.borrow_mut() = Some(graph_task);
        *self.graph_item.borrow_mut() = Some(graph_item);
    }

    /// Java private `addTabbedLogFileSetMenuItems(String[])`.
    fn add_tabbed_log_file_set_menu_items(&self, log_file_set_window_label: &[String]) {
        let action_listener = self
            .action_listener
            .get()
            .expect("actionListener is assigned before the menu items are added")
            .clone();
        let mut log_file_set_item: Vec<Rc<MenuItem>> =
            Vec::with_capacity(log_file_set_window_label.len());
        for label in log_file_set_window_label {
            let item = MenuItem::new_void();
            item.set_text(&format!("{} log file ...", label));
            item.add_action_listener(action_listener.clone());
            self.context_menu.add(&item.get_component());
            log_file_set_item.push(item);
        }
        *self.log_file_set_item.borrow_mut() = Some(log_file_set_item);
    }

    /// Java private `addTabbedLogFileMenuItems(String[])` (no caller in the
    /// Java either).
    fn add_tabbed_log_file_menu_items(&self, log_window_label: &[String]) {
        let action_listener = self
            .action_listener
            .get()
            .expect("actionListener is assigned before the menu items are added")
            .clone();
        let mut log_file_item: Vec<Rc<MenuItem>> = Vec::with_capacity(log_window_label.len());
        for label in log_window_label {
            let item = MenuItem::new_void();
            item.set_text(&format!("{} log file ...", label));
            item.add_action_listener(action_listener.clone());
            self.context_menu.add(&item.get_component());
            log_file_item.push(item);
        }
        *self.log_file_item.borrow_mut() = Some(log_file_item);
    }

    /// Java private `showMenu(Component)`.
    fn show_menu(self: &Rc<Self>, component: &Rc<JComponent>) {
        self.context_menu
            .show(component, self.mouse_event.x, self.mouse_event.y);
        // The Swing popup layer now holds the showing popup (see SHOWING).
        SHOWING.with(|showing| *showing.borrow_mut() = Some(self.clone()));
    }

    // Calculate the IMOD URL: `calcImodURL()` is commented out in the Java.

    /// Java private `validate(String[], String[])`.  `Err` is Java's
    /// `IllegalArgumentException`.
    fn validate(&self, man_page_label: &[String], man_page: &[String]) -> Result<(), String> {
        // Check to make sure that the menu label and man page arrays are the same
        // length
        if man_page_label.len() != man_page.len() {
            let message = "menu label and man page arrays must be the same length";
            return Err(message.to_owned());
        }
        Ok(())
    }

    /// Java private `getAnchor()`.
    fn get_anchor(&self) -> Option<String> {
        self.anchor.borrow().clone()
    }

    /// Java private `getAnchor2()`.
    fn get_anchor2(&self) -> Option<String> {
        self.anchor2.borrow().clone()
    }

    /// Java private `setVisible(boolean)`.
    fn set_visible(&self, visible: bool) {
        self.context_menu.set_visible(visible);
    }

    /// Java private `getManPageItem()`.
    fn get_man_page_item(&self) -> Option<Vec<Rc<MenuItem>>> {
        self.man_page_item.borrow().clone()
    }

    /// Java private `getManPageName()`.
    fn get_man_page_name(&self) -> Option<Vec<String>> {
        self.man_page_name.borrow().clone()
    }

    /// Java private `getLogFileSetItem()`.
    fn get_log_file_set_item(&self) -> Option<Vec<Rc<MenuItem>>> {
        self.log_file_set_item.borrow().clone()
    }

    /// Java private `getLogFileItem()`.
    fn get_log_file_item(&self) -> Option<Vec<Rc<MenuItem>>> {
        self.log_file_item.borrow().clone()
    }

    /// Java private `getGraphItem()`.
    fn get_graph_item(&self) -> Option<Vec<Rc<MenuItem>>> {
        self.graph_item.borrow().clone()
    }

    /// Java private `getGraphTask()`.
    fn get_graph_task(&self) -> Option<Vec<Task>> {
        self.graph_task.borrow().clone()
    }

    /// Java private `getLogFileName()`.
    fn get_log_file_name(&self) -> Option<Vec<String>> {
        self.log_file_name.borrow().clone()
    }

    /// The popup menu (Java `contextMenu`), for a driver that picks an item.
    /// Not a Java member.
    pub fn get_context_menu(&self) -> Rc<JComponent> {
        self.context_menu.clone()
    }
}
