//! `IMOD/Etomo/src/etomo/logic/DirectiveEditorBuilder.java`.
//!
//! `Directive`, `DirectiveMap`, `DirectiveDescrFile`, `ComFile`, and the local
//! directive-file `Autodoc` reader are separate storage source units.  They have not
//! all acquired one shared concrete Rust representation yet.  `DirectiveEditorBuilderSource`
//! is therefore the direct boundary for exactly those source calls; this unit retains
//! the builder's ordering, filtering, value precedence, dropped-directive handling,
//! and default save-location policy.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::directive_file_type::{DirectiveFileType, NUM};
use crate::imod::etomo::ui::swing::directive_editor_dialog::DirectiveEditorBuilder as DirectiveEditorBuilderBoundary;
use crate::imod::etomo::ui::swing::directive_section_panel::{
    Directive as UiDirective, DirectiveDescrSection, DirectiveMap as UiDirectiveMap,
};

/// Java private static `SECTION_OTHER_HEADER`.
const SECTION_OTHER_HEADER: &str = "Other Directives";
/// Java private static `SPECIAL_CASE_PROGRAM_NAME`.
const SPECIAL_CASE_PROGRAM_NAME: &str = "ctfplotter";

/// The `DirectiveType` values which `DirectiveEditorBuilder` tests directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveType {
    /// Java `DirectiveType.COM_PARAM`.
    ComParam,
    /// Java `DirectiveType.SETUP_SET`.
    SetupSet,
    /// Java `DirectiveType.RUN_TIME`.
    RunTime,
    /// A storage type which this source unit does not handle specially.
    Other,
}

/// Java `DirectiveValueType`, as needed by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveValueType {
    /// Java `BOOLEAN`.
    Boolean,
    /// Every non-boolean source value type.
    Other,
}

/// Java `DirectiveValues.Value`, at this unit's storage boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectiveStoredValue {
    /// Java `setValue(String)` or `setDefaultValue(String)`.
    String(String),
    /// Java `setValue(true)` or `setDefaultValue(true)`.
    Boolean(bool),
}

/// Java `DirectiveDescrElement`, projected to the fields read by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectiveDescrElement {
    /// Java `element.isSection()` and `getSectionHeader()`.
    Section(String),
    /// Java `element.isDirective()` and the `Directive(DirectiveDescrElement)` data.
    Directive(DirectiveDescr),
    /// A non-section/non-directive CSV element.
    Other,
}

/// The `DirectiveDescr` data read by Java's `Directive` constructor here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectiveDescr {
    pub name: String,
    pub directive_type: DirectiveType,
    pub value_type: DirectiveValueType,
    pub batch: bool,
    pub template: bool,
}

/// A name/value autodoc statement after Java's `ReadOnlyStatement` filter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectiveStatement {
    pub left_side: String,
    pub right_side: Option<String>,
}

/// Java `DirectiveName`, reduced only to the source calls made in this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectiveName {
    pub key: Option<String>,
    pub directive_type: Option<DirectiveType>,
    pub com_file_name: Option<String>,
    pub program_name: Option<String>,
    pub parameter_name: Option<String>,
}

impl DirectiveName {
    /// Java `DirectiveName()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Java `setKey(String)`.
    pub fn set_key(&mut self, input: &str) -> Option<AxisID> {
        let mut part: Vec<String> = input.split('.').map(str::to_string).collect();
        self.directive_type = match part.first().map(String::as_str) {
            Some("comparam") => Some(DirectiveType::ComParam),
            Some("setupset") => Some(DirectiveType::SetupSet),
            Some("runtime") => Some(DirectiveType::RunTime),
            Some(_) => Some(DirectiveType::Other),
            None => None,
        };
        let mut axis_id = None;
        if self.directive_type == Some(DirectiveType::ComParam) && part.len() > 1 {
            if part[1].ends_with('a') {
                axis_id = Some(AxisID::First);
                part[1].pop();
            } else if part[1].ends_with('b') {
                axis_id = Some(AxisID::Second);
                part[1].pop();
            }
        } else if self.directive_type == Some(DirectiveType::RunTime) && part.len() > 2 {
            if part[2] == "a" {
                axis_id = Some(AxisID::First);
                part[2] = "any".to_string();
            } else if part[2] == "b" {
                axis_id = Some(AxisID::Second);
                part[2] = "any".to_string();
            }
        }
        self.com_file_name = (self.directive_type == Some(DirectiveType::ComParam))
            .then(|| part.get(1).cloned())
            .flatten();
        self.program_name = (self.directive_type == Some(DirectiveType::ComParam))
            .then(|| part.get(2).cloned())
            .flatten();
        self.parameter_name = match self.directive_type {
            Some(DirectiveType::SetupSet) if part.get(1).map(String::as_str) == Some("copyarg") => {
                part.get(2).cloned()
            }
            Some(DirectiveType::SetupSet) => part.get(1).cloned(),
            Some(DirectiveType::ComParam) | Some(DirectiveType::RunTime) => part.get(3).cloned(),
            _ => None,
        };
        self.key = (!part.is_empty()).then(|| part.join("."));
        axis_id
    }
}

/// Java `Directive`, with the fields and value mutations used by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Directive {
    pub name: DirectiveName,
    pub batch: bool,
    pub template: bool,
    pub value_type: DirectiveValueType,
    pub in_directive_file: [bool; NUM as usize],
    pub value: Option<DirectiveStoredValue>,
    pub default_value: Option<DirectiveStoredValue>,
}

impl Directive {
    /// Java `Directive(DirectiveDescr)`.
    pub fn from_descr(descr: &DirectiveDescr) -> Self {
        let mut name = DirectiveName::new();
        name.set_key(&descr.name);
        Self {
            name,
            batch: descr.batch,
            template: descr.template,
            value_type: descr.value_type,
            in_directive_file: [false; NUM as usize],
            value: None,
            default_value: None,
        }
    }

    /// Java `Directive(DirectiveName)`.
    pub fn from_name(name: &DirectiveName) -> Self {
        Self {
            name: name.clone(),
            batch: true,
            template: true,
            value_type: DirectiveValueType::Other,
            in_directive_file: [false; NUM as usize],
            value: None,
            default_value: None,
        }
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        self.name.directive_type.is_some()
            && self
                .name
                .key
                .as_ref()
                .map(|key| key.split('.').count() > 1)
                .unwrap_or(false)
    }

    /// Java `getKey()`.
    pub fn get_key(&self) -> Option<&str> {
        self.name.key.as_deref()
    }

    /// Java `getType()`.
    pub fn get_type(&self) -> Option<DirectiveType> {
        self.name.directive_type
    }

    /// Java `setInDirectiveFile(DirectiveFileType, boolean)`.
    pub fn set_in_directive_file(&mut self, file_type: DirectiveFileType, input: bool) {
        self.in_directive_file[file_type.get_index() as usize] = input;
    }
}

/// Java `ComFile`, at this source unit's direct boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComFile {
    pub axis_id: AxisID,
    pub directory: Option<String>,
    pub com_file_name: Option<String>,
    pub program_name: Option<String>,
}

impl ComFile {
    /// Java `ComFile(BaseManager, AxisID)` and `ComFile(BaseManager, AxisID, String)`.
    pub fn new(axis_id: AxisID, directory: Option<&str>) -> Self {
        Self {
            axis_id,
            directory: directory.map(str::to_string),
            com_file_name: None,
            program_name: None,
        }
    }

    /// Java `equalsComFileName(String)`.
    pub fn equals_com_file_name(&self, name: Option<&str>) -> bool {
        self.com_file_name.is_some() && self.com_file_name.as_deref() == name
    }

    /// Java `setComFileName(String)`.
    pub fn set_com_file_name(&mut self, name: Option<&str>) {
        if !self.equals_com_file_name(name) {
            self.com_file_name = name.map(str::to_string);
            self.program_name = None;
        }
    }

    /// Java `equalsProgramName(String)`.
    pub fn equals_program_name(&self, name: Option<&str>) -> bool {
        self.program_name.is_some() && self.program_name.as_deref() == name
    }
}

/// Direct storage and manager calls made by `DirectiveEditorBuilder`.
pub trait DirectiveEditorBuilderSource: Send + Sync {
    /// Java `DirectiveDescrFile.INSTANCE.getIterator` after the title and column header.
    fn get_directive_descr_elements(
        &self,
        manager: &'static dyn BaseManager,
    ) -> Option<Vec<DirectiveDescrElement>>;
    /// Java local-file existence, `AutodocFactory.getInstance`, and statement iteration.
    fn get_local_directive_statements(
        &self,
        manager: &'static dyn BaseManager,
        file_type: DirectiveFileType,
    ) -> Result<Option<Vec<DirectiveStatement>>, String>;
    /// Java `manager.updateDirectiveMap`.
    fn update_directive_map(
        &self,
        manager: &'static dyn BaseManager,
        directive_map: &mut BTreeMap<String, Directive>,
        errmsg: &mut String,
    );
    /// Java `ComFile.getCommandMap`.
    fn get_command_map(
        &self,
        manager: &'static dyn BaseManager,
        com_file: &ComFile,
        program_name: Option<&str>,
        errmsg: &mut String,
    ) -> Option<HashMap<String, Option<String>>>;
    /// Java `EtomoDirector.INSTANCE.getUserConfiguration` and `ConfigTool.getDefaultUserTemplateDir`.
    fn get_user_template_dir(&self) -> Option<PathBuf>;
}

/// Java final `DirectiveEditorBuilder`.
pub struct DirectiveEditorBuilder {
    pub directive_map: BTreeMap<String, Directive>,
    pub section_array: Vec<DirectiveDescrSection>,
    pub file_type_exists: [bool; NUM as usize],
    pub dropped_directives: Vec<String>,
    pub manager: &'static dyn BaseManager,
    pub directive_file_type: DirectiveFileType,
    pub debug: bool,
    pub other_section: Option<DirectiveDescrSection>,
    pub source: &'static dyn DirectiveEditorBuilderSource,
}

impl DirectiveEditorBuilder {
    /// Java `DirectiveEditorBuilder(BaseManager, DirectiveFileType)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        directive_file_type: DirectiveFileType,
        source: &'static dyn DirectiveEditorBuilderSource,
    ) -> Self {
        Self {
            directive_map: BTreeMap::new(),
            section_array: Vec::new(),
            file_type_exists: [false; NUM as usize],
            dropped_directives: Vec::new(),
            manager,
            directive_file_type,
            debug: false,
            other_section: None,
            source,
        }
    }

    /// Java `build(AxisType, StringBuffer)`.
    pub fn build(&mut self, source_axis_type: AxisType, mut errmsg: Option<String>) -> String {
        self.directive_map.clear();
        self.section_array.clear();
        self.file_type_exists = [false; NUM as usize];
        self.other_section = None;
        let mut errmsg = errmsg.take().unwrap_or_default();
        if let Some(element_array) = self.source.get_directive_descr_elements(self.manager) {
            let mut section_index: Option<usize> = None;
            let mut directive_count = 0;
            let mut matches_type = false;
            for element in element_array {
                match element {
                    DirectiveDescrElement::Section(header) => {
                        if let Some(index) = section_index {
                            self.section_array[index].contains_editable_directives = matches_type;
                            if directive_count == 0 {
                                self.section_array.remove(index);
                            }
                        }
                        section_index = Some(self.section_array.len());
                        directive_count = 0;
                        matches_type = false;
                        self.section_array.push(DirectiveDescrSection {
                            title: header,
                            names: Vec::new(),
                            contains_editable_directives: true,
                        });
                    }
                    DirectiveDescrElement::Directive(descr) => {
                        let directive = Directive::from_descr(&descr);
                        if directive.is_valid() {
                            if (!matches_type
                                && self.directive_file_type != DirectiveFileType::Batch
                                && (directive.template || !directive.batch))
                                || (self.directive_file_type == DirectiveFileType::Batch
                                    && (!directive.template || directive.batch))
                            {
                                matches_type = true;
                            }
                            directive_count += 1;
                            let key = directive.get_key().unwrap().to_string();
                            self.directive_map.insert(key.clone(), directive);
                            if let Some(index) = section_index {
                                self.section_array[index].names.push(key);
                            } else {
                                eprintln!(
                                    "Error: directive {}, not inside a section in the directives.csv file.  It cannot be edited.",
                                    descr.name
                                );
                            }
                        }
                    }
                    DirectiveDescrElement::Other => {}
                }
            }
        }
        self.update_from_local_directive_file(DirectiveFileType::Scope, &mut errmsg);
        self.update_from_local_directive_file(DirectiveFileType::System, &mut errmsg);
        self.update_from_local_directive_file(DirectiveFileType::User, &mut errmsg);
        self.update_from_local_directive_file(DirectiveFileType::Batch, &mut errmsg);
        self.source
            .update_directive_map(self.manager, &mut self.directive_map, &mut errmsg);
        let first_axis_id = if source_axis_type == AxisType::DualAxis {
            AxisID::First
        } else {
            AxisID::Only
        };
        let mut com_file = ComFile::new(first_axis_id, None);
        let mut command_map = None;
        let mut com_file_defaults = ComFile::new(first_axis_id, Some("origcoms"));
        let mut command_map_defaults = None;
        for directive_name in self
            .directive_map
            .values()
            .filter(|directive| directive.get_type() == Some(DirectiveType::ComParam))
            .map(|directive| directive.name.clone())
            .collect::<Vec<_>>()
        {
            command_map = self.get_command_map(
                &directive_name,
                &mut com_file,
                command_map,
                &mut errmsg,
                false,
            );
            if let Some(ref command_map) = command_map {
                self.set_directive_value(command_map, &directive_name, false);
            }
            command_map_defaults = self.get_command_map(
                &directive_name,
                &mut com_file_defaults,
                command_map_defaults,
                &mut errmsg,
                true,
            );
            if let Some(ref command_map) = command_map_defaults {
                self.set_directive_value(command_map, &directive_name, true);
            }
        }
        errmsg
    }

    /// Java `updateFromLocalDirectiveFile(DirectiveFileType, StringBuffer)`.
    fn update_from_local_directive_file(
        &mut self,
        file_type: DirectiveFileType,
        errmsg: &mut String,
    ) -> bool {
        let statement_array = match self
            .source
            .get_local_directive_statements(self.manager, file_type)
        {
            Ok(Some(statement_array)) => statement_array,
            Ok(None) => return false,
            Err(error) => {
                errmsg.push_str(&format!("Unable to load {}.  {}  ", file_type, error));
                return false;
            }
        };
        self.file_type_exists[file_type.get_index() as usize] = true;
        for statement in statement_array {
            let mut directive_name = DirectiveName::new();
            let axis_id = directive_name.set_key(&statement.left_side);
            let key = directive_name.key.clone().unwrap();
            if !self.directive_map.contains_key(&key) {
                let directive = Directive::from_name(&directive_name);
                if axis_id == Some(AxisID::Second) {
                    break;
                }
                if directive.is_valid() && directive.get_type() == Some(DirectiveType::ComParam) {
                    if self.other_section.is_none() {
                        let other_section = DirectiveDescrSection {
                            title: SECTION_OTHER_HEADER.to_string(),
                            names: Vec::new(),
                            contains_editable_directives: true,
                        };
                        self.section_array.push(other_section.clone());
                        self.other_section = Some(other_section);
                    }
                    self.other_section.as_mut().unwrap().names.push(key.clone());
                    let index = self.section_array.len() - 1;
                    self.section_array[index].names.push(key.clone());
                    self.directive_map.insert(key.clone(), directive);
                } else {
                    self.dropped_directives.push(statement.left_side);
                    continue;
                }
            }
            let directive = self.directive_map.get_mut(&key).unwrap();
            directive.set_in_directive_file(file_type, true);
            if directive.value_type != DirectiveValueType::Boolean || statement.right_side.is_some()
            {
                directive.value = Some(match statement.right_side {
                    Some(value) => DirectiveStoredValue::String(value),
                    None => DirectiveStoredValue::Boolean(true),
                });
            }
        }
        true
    }

    /// Java `getCommandMap(DirectiveName, ComFile, Map, StringBuffer, boolean)`.
    fn get_command_map(
        &mut self,
        directive_name: &DirectiveName,
        com_file: &mut ComFile,
        mut command_map: Option<HashMap<String, Option<String>>>,
        errmsg: &mut String,
        origcoms: bool,
    ) -> Option<HashMap<String, Option<String>>> {
        if !com_file.equals_com_file_name(directive_name.com_file_name.as_deref()) {
            com_file.set_com_file_name(directive_name.com_file_name.as_deref());
        }
        if !com_file.equals_program_name(directive_name.program_name.as_deref()) {
            com_file.program_name = directive_name.program_name.clone();
            command_map = self.source.get_command_map(
                self.manager,
                com_file,
                directive_name.program_name.as_deref(),
                errmsg,
            );
            if origcoms
                && com_file.equals_com_file_name(Some(SPECIAL_CASE_PROGRAM_NAME))
                && com_file.equals_program_name(Some(SPECIAL_CASE_PROGRAM_NAME))
            {
                self.set_special_case_default_values(command_map.as_ref());
            }
        }
        command_map
    }

    /// Java `setSpecialCaseDefaultValues(Map)`.
    fn set_special_case_default_values(
        &mut self,
        command_map: Option<&HashMap<String, Option<String>>>,
    ) {
        let mut from_directive_name = DirectiveName::new();
        let mut to_directive_name = DirectiveName::new();
        from_directive_name.set_key("comparam.ctfplotter.ctfplotter.ExpectedDefocus");
        to_directive_name.set_key("setupset.copyarg.defocus");
        self.set_default_directive_value(command_map, &from_directive_name, &to_directive_name);
        from_directive_name.set_key("comparam.ctfplotter.ctfplotter.Voltage");
        to_directive_name.set_key("setupset.copyarg.voltage");
        self.set_default_directive_value(command_map, &from_directive_name, &to_directive_name);
        from_directive_name.set_key("comparam.ctfplotter.ctfplotter.SphericalAberration");
        to_directive_name.set_key("setupset.copyarg.Cs");
        self.set_default_directive_value(command_map, &from_directive_name, &to_directive_name);
    }

    /// Java `setDefaultDirectiveValue(Map, DirectiveName, DirectiveName)`.
    fn set_default_directive_value(
        &mut self,
        command_map: Option<&HashMap<String, Option<String>>>,
        from_directive_name: &DirectiveName,
        to_directive_name: &DirectiveName,
    ) {
        let Some(command_map) = command_map else {
            return;
        };
        let Some(parameter_name) = from_directive_name.parameter_name.as_deref() else {
            return;
        };
        if let Some(value) = command_map.get(parameter_name) {
            let directive = self
                .directive_map
                .get_mut(to_directive_name.key.as_deref().unwrap())
                .unwrap();
            directive.default_value = Some(match value {
                Some(value) => DirectiveStoredValue::String(value.clone()),
                None => DirectiveStoredValue::Boolean(true),
            });
        }
    }

    /// Java `setDirectiveValue(Map, DirectiveName, boolean)`.
    fn set_directive_value(
        &mut self,
        command_map: &HashMap<String, Option<String>>,
        directive_name: &DirectiveName,
        is_default_value: bool,
    ) {
        let Some(parameter_name) = directive_name.parameter_name.as_deref() else {
            return;
        };
        if let Some(value) = command_map.get(parameter_name) {
            let directive = self
                .directive_map
                .get_mut(directive_name.key.as_deref().unwrap())
                .unwrap();
            let value = match value {
                Some(value) => DirectiveStoredValue::String(value.clone()),
                None => DirectiveStoredValue::Boolean(true),
            };
            if is_default_value {
                directive.default_value = Some(value);
            } else {
                directive.value = Some(value);
            }
        }
    }

    /// Java `getSectionArray()`.
    pub fn get_section_array(&self) -> Vec<DirectiveDescrSection> {
        self.section_array.clone()
    }

    /// Java `getFileTypeExists()`.
    pub fn get_file_type_exists(&self) -> [bool; NUM as usize] {
        self.file_type_exists
    }

    /// Java `getDroppedDirectives()`.
    pub fn get_dropped_directives(&self) -> Vec<String> {
        self.dropped_directives.clone()
    }

    /// Java `getDefaultSaveLocation()`.
    pub fn get_default_save_location(&self) -> Option<PathBuf> {
        if self.directive_file_type == DirectiveFileType::User {
            if let Some(directory) = self.source.get_user_template_dir() {
                if !directory.exists() {
                    eprintln!("Creating user template directory:{}", directory.display());
                    let _ = std::fs::create_dir_all(&directory);
                }
                return Some(directory);
            }
        }
        self.manager.get_property_user_dir().map(PathBuf::from)
    }

    /// Java `getDirectiveMap()`.
    pub fn get_directive_map(&self) -> BTreeMap<String, Directive> {
        self.directive_map.clone()
    }
}

impl DirectiveEditorBuilderBoundary for DirectiveEditorBuilder {
    fn get_file_type_exists(&self) -> [bool; NUM as usize] {
        self.get_file_type_exists()
    }

    fn get_section_array(&self) -> Vec<DirectiveDescrSection> {
        self.get_section_array()
    }

    fn get_directive_map(&self) -> UiDirectiveMap {
        UiDirectiveMap {
            directives: self
                .directive_map
                .iter()
                .map(|(key, _)| (key.clone(), UiDirective { name: key.clone() }))
                .collect(),
        }
    }

    fn get_default_save_location(&self) -> Option<PathBuf> {
        self.get_default_save_location()
    }

    fn get_dropped_directives(&self) -> Vec<String> {
        self.get_dropped_directives()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
    use crate::imod::etomo::r#type::interface_type::InterfaceType;

    struct Manager(BaseManagerBase);
    impl BaseManager for Manager {
        fn base(&self) -> &BaseManagerBase {
            &self.0
        }
        fn this(&'static self) -> &'static dyn BaseManager {
            self
        }
        fn get_interface_type(&self) -> Option<InterfaceType> {
            None
        }
        fn create_main_panel(&self) {}
        fn get_base_meta_data(
            &self,
        ) -> Option<&dyn crate::imod::etomo::r#type::base_meta_data::BaseMetaData> {
            None
        }
        fn get_main_panel(&self) -> Option<std::convert::Infallible> {
            None
        }
        fn get_process_manager(&self) -> Option<std::convert::Infallible> {
            None
        }
        fn get_storables_with_offset(
            &self,
            _offset: i32,
        ) -> Option<Vec<Box<dyn crate::imod::etomo::storage::storable::Storable>>> {
            None
        }
        fn get_name(&self) -> Option<String> {
            None
        }
    }

    struct Source;
    impl DirectiveEditorBuilderSource for Source {
        fn get_directive_descr_elements(
            &self,
            _manager: &'static dyn BaseManager,
        ) -> Option<Vec<DirectiveDescrElement>> {
            Some(vec![
                DirectiveDescrElement::Section("Main".to_string()),
                DirectiveDescrElement::Directive(DirectiveDescr {
                    name: "comparam.test.test.Value".to_string(),
                    directive_type: DirectiveType::ComParam,
                    value_type: DirectiveValueType::Other,
                    batch: false,
                    template: true,
                }),
                DirectiveDescrElement::Section("Empty".to_string()),
            ])
        }
        fn get_local_directive_statements(
            &self,
            _manager: &'static dyn BaseManager,
            file_type: DirectiveFileType,
        ) -> Result<Option<Vec<DirectiveStatement>>, String> {
            if file_type == DirectiveFileType::Scope {
                Ok(Some(vec![DirectiveStatement {
                    left_side: "comparam.test.test.Value".to_string(),
                    right_side: Some("local".to_string()),
                }]))
            } else {
                Ok(None)
            }
        }
        fn update_directive_map(
            &self,
            _manager: &'static dyn BaseManager,
            _directive_map: &mut BTreeMap<String, Directive>,
            _errmsg: &mut String,
        ) {
        }
        fn get_command_map(
            &self,
            _manager: &'static dyn BaseManager,
            com_file: &ComFile,
            _program_name: Option<&str>,
            _errmsg: &mut String,
        ) -> Option<HashMap<String, Option<String>>> {
            (com_file.directory.is_none())
                .then(|| HashMap::from([("Value".to_string(), Some("com".to_string()))]))
        }
        fn get_user_template_dir(&self) -> Option<PathBuf> {
            None
        }
    }

    #[test]
    fn build_leaves_the_final_empty_section_as_the_java_loop_does() {
        static MANAGER: std::sync::LazyLock<Manager> =
            std::sync::LazyLock::new(|| Manager(BaseManagerBase::initial()));
        static SOURCE: Source = Source;
        let mut builder = DirectiveEditorBuilder::new(&*MANAGER, DirectiveFileType::Scope, &SOURCE);
        assert_eq!(builder.build(AxisType::SingleAxis, None), "");
        assert_eq!(builder.section_array.len(), 2);
        assert_eq!(builder.section_array[0].names, ["comparam.test.test.Value"]);
        assert_eq!(
            builder.directive_map["comparam.test.test.Value"]
                .value
                .as_ref(),
            Some(&DirectiveStoredValue::String("com".to_string()))
        );
        assert!(builder.file_type_exists[DirectiveFileType::Scope.get_index() as usize]);
    }

    #[test]
    fn local_undefined_comparam_is_added_to_other_directives_but_b_axis_stops_reading() {
        static MANAGER: std::sync::LazyLock<Manager> =
            std::sync::LazyLock::new(|| Manager(BaseManagerBase::initial()));
        struct LocalSource;
        impl DirectiveEditorBuilderSource for LocalSource {
            fn get_directive_descr_elements(
                &self,
                _manager: &'static dyn BaseManager,
            ) -> Option<Vec<DirectiveDescrElement>> {
                Some(Vec::new())
            }
            fn get_local_directive_statements(
                &self,
                _manager: &'static dyn BaseManager,
                file_type: DirectiveFileType,
            ) -> Result<Option<Vec<DirectiveStatement>>, String> {
                Ok((file_type == DirectiveFileType::Scope).then(|| {
                    vec![
                        DirectiveStatement {
                            left_side: "comparam.item.item.Value".to_string(),
                            right_side: Some("x".to_string()),
                        },
                        DirectiveStatement {
                            left_side: "comparam.itemb.itemb.Value".to_string(),
                            right_side: Some("b".to_string()),
                        },
                        DirectiveStatement {
                            left_side: "comparam.later.later.Value".to_string(),
                            right_side: Some("later".to_string()),
                        },
                    ]
                }))
            }
            fn update_directive_map(
                &self,
                _manager: &'static dyn BaseManager,
                _directive_map: &mut BTreeMap<String, Directive>,
                _errmsg: &mut String,
            ) {
            }
            fn get_command_map(
                &self,
                _manager: &'static dyn BaseManager,
                _com_file: &ComFile,
                _program_name: Option<&str>,
                _errmsg: &mut String,
            ) -> Option<HashMap<String, Option<String>>> {
                None
            }
            fn get_user_template_dir(&self) -> Option<PathBuf> {
                None
            }
        }
        static SOURCE: LocalSource = LocalSource;
        let mut builder = DirectiveEditorBuilder::new(&*MANAGER, DirectiveFileType::Scope, &SOURCE);
        builder.build(AxisType::SingleAxis, None);
        assert!(
            builder
                .directive_map
                .contains_key("comparam.item.item.Value")
        );
        assert!(
            !builder
                .directive_map
                .contains_key("comparam.later.later.Value")
        );
        assert_eq!(builder.section_array[0].title, SECTION_OTHER_HEADER);
    }
}
