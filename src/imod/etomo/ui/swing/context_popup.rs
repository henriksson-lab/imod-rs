//! `IMOD/Etomo/src/etomo/ui/swing/ContextPopup.java`.
//!
//! `JPopupMenu`, `TextPageWindow`, `TabbedTextWindow`, and the process launchers are
//! presentation/process boundaries.  This module keeps the menu construction and its
//! source action-command dispatch in one state object; the caller performs the returned
//! boundary operation.  In particular, an unavailable Swing implementation must not
//! make a selected help item disappear or change its target.
#![allow(dead_code)]

use super::etomo_menu::MenuItem;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use std::path::PathBuf;

pub const TOMO_GUIDE: &str = "tomoguide.html";
pub const JOIN_GUIDE: &str = "tomojoin.html";
pub const SERIAL_GUIDE: &str = "serialalign.html";
pub const BATCHRUNTOMO_GUIDE: &str = "batchGuide.html";
pub const ALIGNFRAMES_GUIDE: &str = "alignframesGuide.html";
const TOMO_GUIDE_LABEL: &str = "Tomography Guide";
const TOP_ANCHOR: &str = "#TOP";

/// Java `MouseEvent`, restricted to the two values `showMenu` reads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MouseEvent {
    pub x: i32,
    pub y: i32,
}

/// A task supplied by `TomodataplotsParam.Task[]`.  Availability has already been
/// evaluated at the `TomodataplotsParam.Task.isAvailable` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphTask {
    pub description: String,
    pub available: bool,
    pub input_file: Option<PathBuf>,
}

/// The operation which Java sends across a direct dependency boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContextPopupTarget {
    Imodqtassist {
        path: String,
        axis_id: AxisID,
    },
    TextPageWindow {
        path: PathBuf,
    },
    TabbedTextWindow {
        title: String,
        paths: Vec<PathBuf>,
        labels: Vec<String>,
        /// Java calls `ApplicationManager.updateLog` first only when this is set.
        update_log_command_name: Option<String>,
    },
    Tomodataplots {
        task: String,
        axis_id: AxisID,
        input_file: Option<PathBuf>,
    },
    PeetHelp {
        program: PathBuf,
        axis_id: AxisID,
    },
    /// Java branches independently inspect every standard item, so a command which
    /// happens not to name one produces no operation.
    None,
}

/// Java `ContextPopup` fields.  `menu` preserves JPopupMenu order; `None` is its
/// `JPopupMenu.Separator`.
#[derive(Clone, Debug)]
pub struct ContextPopup {
    pub context_menu_visible: bool,
    pub popup_title: &'static str,
    pub menu: Vec<Option<MenuItem>>,
    pub mouse_event: MouseEvent,
    pub popup_position: Option<(i32, i32)>,
    pub tomo_guide_item: MenuItem,
    pub serial_sections_guide_item: MenuItem,
    pub model_guide_item: MenuItem,
    pub it_3dmod_guide_item: MenuItem,
    pub etomo_guide_item: MenuItem,
    pub join_guide_item: MenuItem,
    pub peet_guide_item: MenuItem,
    pub peet_help_item: MenuItem,
    pub batch_guide_item: MenuItem,
    pub alignframes_guide_item: MenuItem,
    pub tomo_guide_item_2: Option<MenuItem>,
    pub man_page_name: Option<Vec<String>>,
    pub log_file_name: Option<Vec<String>>,
    pub man_page_item: Option<Vec<MenuItem>>,
    pub log_file_item: Option<Vec<MenuItem>>,
    pub log_file_set_item: Option<Vec<MenuItem>>,
    pub anchor: Option<String>,
    pub tomo_guide_alt_label: Option<String>,
    pub graph_item: Option<Vec<MenuItem>>,
    pub graph_task: Option<Vec<GraphTask>>,
    pub serial_sections: bool,
    pub anchor_2: Option<String>,
    pub tomo_guide_alt_label_2: Option<String>,
    pub axis_id: AxisID,
    pub guide_to_anchor: Option<String>,
    pub subdir_name: Option<String>,
    pub log_file_set_window_label: Option<Vec<String>>,
    pub log_file_set_label: Option<Vec<Vec<String>>>,
    pub log_file_set: Option<Vec<Vec<String>>>,
    pub update_log_command_name: Option<String>,
}

impl ContextPopup {
    /// Java constructor `(Component, MouseEvent, String, BaseManager, AxisID)`.
    pub fn new(mouse_event: MouseEvent, tomo_anchor: Option<&str>, axis_id: AxisID) -> Self {
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.anchor = tomo_anchor.map(str::to_owned);
        popup.guide_to_anchor = Some(TOMO_GUIDE.into());
        popup.add_standard_menu_items(false, None, None);
        popup.show_menu();
        popup
    }

    /// Java constructor with `manPageLabel` and `manPage`.
    pub fn new_man_pages(
        mouse_event: MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: &str,
        man_page_label: &[String],
        man_page: &[String],
        axis_id: AxisID,
    ) -> Result<Self, String> {
        Self::validate(man_page_label, man_page)?;
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.anchor = tomo_anchor.map(str::to_owned);
        popup.guide_to_anchor = Some(guide_to_anchor.into());
        popup.add_man_page_menu_items(man_page_label, man_page);
        popup.menu.push(None);
        popup.add_standard_menu_items(false, Some(guide_to_anchor), None);
        popup.show_menu();
        Ok(popup)
    }

    /// Java constructor with a guide anchor but no local items.
    pub fn new_guide(
        mouse_event: MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: &str,
        axis_id: AxisID,
    ) -> Self {
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.anchor = tomo_anchor.map(str::to_owned);
        popup.guide_to_anchor = Some(guide_to_anchor.into());
        popup.menu.push(None);
        popup.add_standard_menu_items(false, Some(guide_to_anchor), None);
        popup.show_menu();
        popup
    }

    /// Java constructor with man pages and ordinary log files.  `subdir_name` models
    /// the source's final overload as well, without changing its dispatch semantics.
    pub fn new_log_files(
        mouse_event: MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: &str,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: &[String],
        log_file: &[String],
        axis_id: AxisID,
        subdir_name: Option<&str>,
    ) -> Result<Self, String> {
        Self::validate(man_page_label, man_page)?;
        Self::validate(log_file_label, log_file)?;
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.anchor = tomo_anchor.map(str::to_owned);
        popup.guide_to_anchor = Some(guide_to_anchor.into());
        popup.subdir_name = subdir_name.map(str::to_owned);
        popup.add_log_file_menu_items(log_file_label, log_file);
        popup.menu.push(None);
        popup.add_man_page_menu_items(man_page_label, man_page);
        popup.menu.push(None);
        popup.add_standard_menu_items(false, Some(guide_to_anchor), None);
        popup.show_menu();
        Ok(popup)
    }

    /// Java two-Tomography-guide constructor.
    pub fn new_two_tomo_anchors(
        mouse_event: MouseEvent,
        tomo_anchor: Option<&str>,
        tomo_guide_label_suffix: Option<&str>,
        tomo_anchor_2: Option<&str>,
        tomo_guide_label_suffix_2: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: &[String],
        log_file: &[String],
        axis_id: AxisID,
    ) -> Result<Self, String> {
        let mut popup = Self::new_log_files(
            mouse_event,
            tomo_anchor,
            TOMO_GUIDE,
            man_page_label,
            man_page,
            log_file_label,
            log_file,
            axis_id,
            None,
        )?;
        popup.tomo_guide_alt_label = popup.build_tomo_guide_alt_label(tomo_guide_label_suffix);
        popup.anchor_2 = tomo_anchor_2.map(str::to_owned);
        popup.tomo_guide_alt_label_2 = popup.build_tomo_guide_alt_label(tomo_guide_label_suffix_2);
        // Java creates labels immediately before `addStandardMenuItems`; rebuild the
        // standard tail because this constructor follows the same object sequence.
        let standard_start = popup
            .menu
            .iter()
            .rposition(|item| item.is_none())
            .unwrap_or(0);
        popup.menu.truncate(standard_start + 1);
        popup.add_standard_menu_items(false, Some(TOMO_GUIDE), None);
        Ok(popup)
    }

    /// Java graph constructor.  The graph availability test is represented in
    /// `add_graph_menu_items` and so is performed before menu construction.
    pub fn new_graphs(
        mouse_event: MouseEvent,
        tomo_anchor: Option<&str>,
        guide_to_anchor: &str,
        man_page_label: &[String],
        man_page: &[String],
        log_file_label: &[String],
        log_file: &[String],
        graph: &[GraphTask],
        axis_id: AxisID,
        serial_sections: bool,
    ) -> Result<Self, String> {
        Self::validate(man_page_label, man_page)?;
        Self::validate(log_file_label, log_file)?;
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.anchor = tomo_anchor.map(str::to_owned);
        popup.guide_to_anchor = Some(guide_to_anchor.into());
        popup.serial_sections = serial_sections;
        popup.add_log_file_menu_items(log_file_label, log_file);
        if !graph.is_empty() {
            popup.menu.push(None);
            popup.add_graph_menu_items(graph);
        }
        popup.menu.push(None);
        popup.add_man_page_menu_items(man_page_label, man_page);
        popup.menu.push(None);
        popup.add_standard_menu_items(false, Some(guide_to_anchor), None);
        popup.show_menu();
        Ok(popup)
    }

    /// Java `(String[], String[], boolean, BaseManager, AxisID)` constructor.
    pub fn new_peet_man_pages(
        mouse_event: MouseEvent,
        man_page_label: &[String],
        man_page: &[String],
        add_peet_guide: bool,
        peet_directory_exists: bool,
        axis_id: AxisID,
    ) -> Result<Self, String> {
        Self::validate(man_page_label, man_page)?;
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.add_man_page_menu_items(man_page_label, man_page);
        popup.menu.push(None);
        popup.add_standard_menu_items(add_peet_guide, None, Some(peet_directory_exists));
        popup.show_menu();
        Ok(popup)
    }

    /// Java `(Component, MouseEvent, BaseManager, AxisID, boolean)` constructor.
    pub fn new_serial_sections(
        mouse_event: MouseEvent,
        axis_id: AxisID,
        serial_sections: bool,
    ) -> Self {
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.serial_sections = serial_sections;
        popup.menu.push(None);
        popup.add_standard_menu_items(false, None, None);
        popup.show_menu();
        popup
    }

    /// Java tabbed-log constructor.  The IOException/OutOfMemory UI branches remain
    /// an explicit window boundary; files and labels are retained exactly for it.
    pub fn new_tabbed_log_files(
        mouse_event: MouseEvent,
        tomo_anchor: Option<&str>,
        man_page_label: &[String],
        man_page: &[String],
        window_labels: &[String],
        file_set_labels: &[Vec<String>],
        file_sets: &[Vec<String>],
        log_file_label: &[String],
        log_file: &[String],
        graph: &[GraphTask],
        update_log_command_name: Option<&str>,
        axis_id: AxisID,
    ) -> Result<Self, String> {
        Self::validate(man_page_label, man_page)?;
        Self::validate(log_file_label, log_file)?;
        if file_set_labels.len() != file_sets.len() {
            return Err("log file label and log file vectors must be the same length".into());
        }
        if window_labels.len() != file_sets.len() {
            return Err(
                "log file window label and log file vectors must be the same length".into(),
            );
        }
        let mut popup = Self::empty(mouse_event, axis_id);
        popup.anchor = tomo_anchor.map(str::to_owned);
        popup.log_file_set_window_label = Some(window_labels.to_vec());
        popup.log_file_set_label = Some(file_set_labels.to_vec());
        popup.log_file_set = Some(file_sets.to_vec());
        popup.update_log_command_name = update_log_command_name.map(str::to_owned);
        popup.add_tabbed_log_file_set_menu_items(window_labels);
        popup.add_log_file_menu_items(log_file_label, log_file);
        if !graph.is_empty() {
            popup.menu.push(None);
            popup.add_graph_menu_items(graph);
        }
        popup.menu.push(None);
        popup.add_man_page_menu_items(man_page_label, man_page);
        popup.menu.push(None);
        popup.add_standard_menu_items(false, None, None);
        popup.show_menu();
        Ok(popup)
    }

    /// Java listener `actionPerformed`.  `ImodqtassistProcess`, text windows and the
    /// PEET system thread deliberately return their source arguments as targets.
    pub fn action_performed(
        &mut self,
        action_command: &str,
        manager: &dyn BaseManager,
    ) -> ContextPopupTarget {
        self.set_visible(false);
        if let (Some(items), Some(names)) = (&self.man_page_item, &self.man_page_name) {
            if let Some(index) = items
                .iter()
                .position(|item| item.action_command == action_command)
            {
                return ContextPopupTarget::Imodqtassist {
                    path: format!("man/{}", names[index]),
                    axis_id: self.axis_id,
                };
            }
        }
        if let (Some(items), Some(names)) = (&self.log_file_item, &self.log_file_name) {
            if let Some(index) = items
                .iter()
                .position(|item| item.action_command == action_command)
            {
                let mut path = PathBuf::from(manager.get_property_user_dir().unwrap_or_default());
                if let Some(subdir) = &self.subdir_name {
                    path.push(subdir);
                }
                path.push(&names[index]);
                return ContextPopupTarget::TextPageWindow { path };
            }
        }
        if let (Some(items), Some(windows), Some(labels), Some(files)) = (
            &self.log_file_set_item,
            &self.log_file_set_window_label,
            &self.log_file_set_label,
            &self.log_file_set,
        ) {
            if let Some(index) = items
                .iter()
                .position(|item| item.action_command == action_command)
            {
                let root = PathBuf::from(manager.get_property_user_dir().unwrap_or_default());
                return ContextPopupTarget::TabbedTextWindow {
                    title: windows[index].clone(),
                    paths: files[index].iter().map(|file| root.join(file)).collect(),
                    labels: labels[index].clone(),
                    update_log_command_name: self
                        .update_log_command_name
                        .as_ref()
                        .filter(|name| action_command.starts_with(name.as_str()))
                        .cloned(),
                };
            }
        }
        if let (Some(items), Some(tasks)) = (&self.graph_item, &self.graph_task) {
            if let Some(index) = items
                .iter()
                .position(|item| item.action_command == action_command)
            {
                return ContextPopupTarget::Tomodataplots {
                    task: tasks[index].description.clone(),
                    axis_id: self.axis_id,
                    input_file: tasks[index].input_file.clone(),
                };
            }
        }
        self.global_item_action(action_command)
    }

    /// Java private `addStandardMenuItems`.
    pub fn add_standard_menu_items(
        &mut self,
        add_peet_guide: bool,
        guide_to_anchor: Option<&str>,
        peet_directory_exists: Option<bool>,
    ) {
        let batch = guide_to_anchor == Some(BATCHRUNTOMO_GUIDE);
        let alignframes = guide_to_anchor == Some(ALIGNFRAMES_GUIDE);
        if batch {
            self.menu.push(Some(self.batch_guide_item.clone()));
        } else if alignframes {
            self.menu.push(Some(self.alignframes_guide_item.clone()));
        }
        if add_peet_guide {
            self.peet_help_item.enabled = peet_directory_exists.unwrap_or(false);
            self.menu.push(Some(self.peet_guide_item.clone()));
            self.menu.push(Some(self.peet_help_item.clone()));
        }
        if self.serial_sections {
            self.menu
                .push(Some(self.serial_sections_guide_item.clone()));
        } else {
            if let Some(label) = &self.tomo_guide_alt_label {
                self.tomo_guide_item = MenuItem::new(label);
            }
            if let Some(label) = &self.tomo_guide_alt_label_2 {
                self.tomo_guide_item_2 = Some(MenuItem::new(label));
            }
            self.menu.push(Some(self.tomo_guide_item.clone()));
            if let Some(item) = &self.tomo_guide_item_2 {
                self.menu.push(Some(item.clone()));
            }
        }
        self.menu.push(Some(self.model_guide_item.clone()));
        self.menu.push(Some(self.it_3dmod_guide_item.clone()));
        self.menu.push(Some(self.etomo_guide_item.clone()));
        if !self.serial_sections {
            self.menu.push(Some(self.join_guide_item.clone()));
        }
        if !batch {
            self.menu.push(Some(self.batch_guide_item.clone()));
        }
        if !alignframes {
            self.menu.push(Some(self.alignframes_guide_item.clone()));
        }
    }

    /// Java private `addManPageMenuItems`.
    pub fn add_man_page_menu_items(&mut self, labels: &[String], pages: &[String]) {
        let items: Vec<_> = labels
            .iter()
            .map(|label| MenuItem::new(&format!("{} man page ...", label)))
            .collect();
        self.man_page_name = Some(
            pages
                .iter()
                .map(|page| format!("{}{}", page, TOP_ANCHOR))
                .collect(),
        );
        self.menu.extend(items.iter().cloned().map(Some));
        self.man_page_item = Some(items);
    }

    /// Java private `addLogFileMenuItems`.
    pub fn add_log_file_menu_items(&mut self, labels: &[String], files: &[String]) {
        let items: Vec<_> = labels
            .iter()
            .map(|label| MenuItem::new(&format!("{} log file ...", label)))
            .collect();
        self.log_file_name = Some(files.to_vec());
        self.menu.extend(items.iter().cloned().map(Some));
        self.log_file_item = Some(items);
    }

    /// Java private `addGraphMenuItems`.
    pub fn add_graph_menu_items(&mut self, graph: &[GraphTask]) {
        let tasks: Vec<_> = graph
            .iter()
            .filter(|task| {
                task.available || task.input_file.as_ref().is_some_and(|file| file.exists())
            })
            .cloned()
            .collect();
        let items: Vec<_> = tasks
            .iter()
            .map(|task| MenuItem::new(&task.description))
            .collect();
        self.menu.extend(items.iter().cloned().map(Some));
        self.graph_task = Some(tasks);
        self.graph_item = Some(items);
    }

    /// Java private `addTabbedLogFileSetMenuItems`.
    pub fn add_tabbed_log_file_set_menu_items(&mut self, labels: &[String]) {
        let items: Vec<_> = labels
            .iter()
            .map(|label| MenuItem::new(&format!("{} log file ...", label)))
            .collect();
        self.menu.extend(items.iter().cloned().map(Some));
        self.log_file_set_item = Some(items);
    }

    /// Java private unused `addTabbedLogFileMenuItems`.
    pub fn add_tabbed_log_file_menu_items(&mut self, labels: &[String]) {
        let items: Vec<_> = labels
            .iter()
            .map(|label| MenuItem::new(&format!("{} log file ...", label)))
            .collect();
        self.menu.extend(items.iter().cloned().map(Some));
        self.log_file_item = Some(items);
    }

    /// Java private `showMenu`.
    pub fn show_menu(&mut self) {
        self.popup_position = Some((self.mouse_event.x, self.mouse_event.y));
        self.context_menu_visible = true;
    }
    /// Java private `validate`.
    pub fn validate(labels: &[String], values: &[String]) -> Result<(), String> {
        if labels.len() == values.len() {
            Ok(())
        } else {
            Err("menu label and man page arrays must be the same length".into())
        }
    }
    /// Java private `buildTomoGuideAltLabel`.
    pub fn build_tomo_guide_alt_label(&self, suffix: Option<&str>) -> Option<String> {
        suffix.map(|suffix| format!("{} ({}) ...", TOMO_GUIDE_LABEL, suffix))
    }
    /// Java private `getAnchor`.
    pub fn get_anchor(&self) -> Option<&str> {
        self.anchor.as_deref()
    }
    /// Java private `getAnchor2`.
    pub fn get_anchor_2(&self) -> Option<&str> {
        self.anchor_2.as_deref()
    }
    /// Java private `setVisible`.
    pub fn set_visible(&mut self, visible: bool) {
        self.context_menu_visible = visible;
    }
    /// Java private getters, with Java null represented by `Option`.
    pub fn get_man_page_item(&self) -> Option<&[MenuItem]> {
        self.man_page_item.as_deref()
    }
    pub fn get_man_page_name(&self) -> Option<&[String]> {
        self.man_page_name.as_deref()
    }
    pub fn get_log_file_set_item(&self) -> Option<&[MenuItem]> {
        self.log_file_set_item.as_deref()
    }
    pub fn get_log_file_item(&self) -> Option<&[MenuItem]> {
        self.log_file_item.as_deref()
    }
    pub fn get_graph_item(&self) -> Option<&[MenuItem]> {
        self.graph_item.as_deref()
    }
    pub fn get_graph_task(&self) -> Option<&[GraphTask]> {
        self.graph_task.as_deref()
    }
    pub fn get_log_file_name(&self) -> Option<&[String]> {
        self.log_file_name.as_deref()
    }

    /// Java private `globalItemAction` overloads, returning the direct process boundary.
    pub fn global_item_action(&self, action_command: &str) -> ContextPopupTarget {
        let guide = self.guide_to_anchor.as_deref();
        let location = self.guide_location(self.anchor.as_deref());
        let location_2 = self.guide_location(self.anchor_2.as_deref());
        let open = |path: String| ContextPopupTarget::Imodqtassist {
            path,
            axis_id: self.axis_id,
        };
        if action_command == self.tomo_guide_item.action_command {
            return open(if guide == Some(TOMO_GUIDE) {
                location
            } else {
                format!("{}{}", TOMO_GUIDE, TOP_ANCHOR)
            });
        }
        if self
            .tomo_guide_item_2
            .as_ref()
            .is_some_and(|item| action_command == item.action_command)
        {
            return open(if guide == Some(TOMO_GUIDE) {
                location_2
            } else {
                format!("{}{}", TOMO_GUIDE, TOP_ANCHOR)
            });
        }
        if action_command == self.model_guide_item.action_command {
            return open(format!("guide.html{}", TOP_ANCHOR));
        }
        if action_command == self.it_3dmod_guide_item.action_command {
            return open(format!("3dmodguide.html{}", TOP_ANCHOR));
        }
        if action_command == self.etomo_guide_item.action_command {
            return open(format!("UsingEtomo.html{}", TOP_ANCHOR));
        }
        if action_command == self.join_guide_item.action_command {
            return open(if guide == Some(JOIN_GUIDE) {
                location
            } else {
                format!("{}{}", JOIN_GUIDE, TOP_ANCHOR)
            });
        }
        if action_command == self.batch_guide_item.action_command {
            return open(if guide == Some(BATCHRUNTOMO_GUIDE) {
                location
            } else {
                format!("{}{}", BATCHRUNTOMO_GUIDE, TOP_ANCHOR)
            });
        }
        if action_command == self.peet_guide_item.action_command {
            return open(format!("PEETmanual.html{}", TOP_ANCHOR));
        }
        if action_command == self.peet_help_item.action_command {
            return ContextPopupTarget::PeetHelp {
                program: std::env::var_os("PARTICLE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_default()
                    .join("bin")
                    .join("PEETHelp"),
                axis_id: self.axis_id,
            };
        }
        if action_command == self.serial_sections_guide_item.action_command {
            return open(if guide == Some(SERIAL_GUIDE) {
                location
            } else {
                format!("{}{}", SERIAL_GUIDE, TOP_ANCHOR)
            });
        }
        if action_command == self.alignframes_guide_item.action_command {
            return open(if guide == Some(ALIGNFRAMES_GUIDE) {
                location
            } else {
                format!("{}{}", ALIGNFRAMES_GUIDE, TOP_ANCHOR)
            });
        }
        ContextPopupTarget::None
    }

    fn empty(mouse_event: MouseEvent, axis_id: AxisID) -> Self {
        Self {
            context_menu_visible: false,
            popup_title: "Help Documents",
            menu: Vec::new(),
            mouse_event,
            popup_position: None,
            tomo_guide_item: MenuItem::new(&(TOMO_GUIDE_LABEL.to_owned() + " ...")),
            serial_sections_guide_item: MenuItem::new("Serial Section Guide ..."),
            model_guide_item: MenuItem::new("IMOD Users Guide ..."),
            it_3dmod_guide_item: MenuItem::new("3dmod Users Guide ..."),
            etomo_guide_item: MenuItem::new("Etomo Users Guide ..."),
            join_guide_item: MenuItem::new("Join Users Guide ..."),
            peet_guide_item: MenuItem::new("PEET Users Guide ..."),
            peet_help_item: MenuItem::new("PEET Help ..."),
            batch_guide_item: MenuItem::new("Batch Interface Guide ..."),
            alignframes_guide_item: MenuItem::new("Align Frames Guide ..."),
            tomo_guide_item_2: None,
            man_page_name: None,
            log_file_name: None,
            man_page_item: None,
            log_file_item: None,
            log_file_set_item: None,
            anchor: None,
            tomo_guide_alt_label: None,
            graph_item: None,
            graph_task: None,
            serial_sections: false,
            anchor_2: None,
            tomo_guide_alt_label_2: None,
            axis_id,
            guide_to_anchor: None,
            subdir_name: None,
            log_file_set_window_label: None,
            log_file_set_label: None,
            log_file_set: None,
            update_log_command_name: None,
        }
    }
    fn guide_location(&self, anchor: Option<&str>) -> String {
        let mut location = self
            .guide_to_anchor
            .clone()
            .unwrap_or_else(|| TOMO_GUIDE.into());
        if let Some(anchor) = anchor.filter(|anchor| !anchor.trim().is_empty()) {
            location.push('#');
            location.push_str(anchor);
        }
        if !location.trim().is_empty() && !location.contains('#') {
            location.push_str(TOP_ANCHOR);
        }
        location
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guide_target_adds_top_anchor_and_hides_menu() {
        let mut popup = ContextPopup::new_guide(
            MouseEvent { x: 4, y: 9 },
            Some("setup"),
            TOMO_GUIDE,
            AxisID::Only,
        );
        assert_eq!(popup.popup_position, Some((4, 9)));
        assert_eq!(
            popup.global_item_action("Tomography Guide ..."),
            ContextPopupTarget::Imodqtassist {
                path: "tomoguide.html#setup".into(),
                axis_id: AxisID::Only
            }
        );
        popup.set_visible(false);
        assert!(!popup.context_menu_visible);
    }
    #[test]
    fn ordinary_log_file_keeps_manager_directory_and_subdir_boundary() {
        // Constructor validation and menu model are testable without manufacturing a
        // BaseManager implementation merely for the Swing TextPageWindow boundary.
        let popup = ContextPopup::new_log_files(
            MouseEvent::default(),
            None,
            TOMO_GUIDE,
            &["Tilt".into()],
            &["tilt".into()],
            &["Output".into()],
            &["tilt.log".into()],
            AxisID::First,
            Some("logs"),
        )
        .unwrap();
        assert_eq!(popup.get_man_page_name().unwrap(), ["tilt#TOP"]);
        assert_eq!(popup.get_log_file_name().unwrap(), ["tilt.log"]);
        assert_eq!(popup.subdir_name.as_deref(), Some("logs"));
    }
    #[test]
    fn graph_filter_and_two_tomo_labels_follow_source() {
        let mut popup = ContextPopup::new_two_tomo_anchors(
            MouseEvent::default(),
            Some("a"),
            Some("A"),
            Some("b"),
            Some("B"),
            &[],
            &[],
            &[],
            &[],
            AxisID::Second,
        )
        .unwrap();
        assert_eq!(
            popup.global_item_action("Tomography Guide (B) ..."),
            ContextPopupTarget::Imodqtassist {
                path: "tomoguide.html#b".into(),
                axis_id: AxisID::Second
            }
        );
        popup.add_graph_menu_items(&[
            GraphTask {
                description: "hidden".into(),
                available: false,
                input_file: None,
            },
            GraphTask {
                description: "shown".into(),
                available: true,
                input_file: None,
            },
        ]);
        assert_eq!(popup.get_graph_item().unwrap().len(), 1);
        assert_eq!(popup.get_graph_item().unwrap()[0].action_command, "shown");
    }
}
