//! `IMOD/Etomo/src/etomo/logic/DirectiveEditorBuilder.java`.
//!
//! Builds the list of directives to use in the directive editor: the directive names,
//! descriptions and sections from `$IMOD_DIR/com/directives.csv`
//! (`DirectiveDescrFile`), values from the local directive files (scope, system, user
//! and batch), the setupset/runtime values from the source manager, and the comparam
//! values and default values from the dataset's `.com` and `origcoms/*.com` files.
//!
//! An event dispatch thread object: `DirectiveEditorManager` builds it and the
//! `DirectiveEditorDialog` keeps it (`Rc<DirectiveEditorBuilder>`), so its mutable
//! fields are cells.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::config_tool;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_statement::ReadOnlyStatement;
use crate::imod::etomo::storage::autodoc::read_only_statement_list::ReadOnlyStatementList;
use crate::imod::etomo::storage::autodoc::statement;
use crate::imod::etomo::storage::com_file::ComFile;
use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::storage::directive_descr_file;
use crate::imod::etomo::storage::directive_descr_section::DirectiveDescrSection;
use crate::imod::etomo::storage::directive_map::DirectiveMap;
use crate::imod::etomo::storage::directive_name::DirectiveName;
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::directive_file_type::{self, DirectiveFileType};
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java private static final `SECTION_OTHER_HEADER`.
const SECTION_OTHER_HEADER: &str = "Other Directives";
/// Java private static final `AXID_ID`.
const AXID_ID: AxisID = AxisID::Only;
/// Java private static final `SPECIAL_CASE_PROGRAM_NAME`.
const SPECIAL_CASE_PROGRAM_NAME: &str = "ctfplotter";

/// A command map: Java `Map<String, String>` whose values may be null.
type CommandMap = HashMap<String, Option<String>>;

/// Java `public final class DirectiveEditorBuilder`.
pub struct DirectiveEditorBuilder {
    /// Java private final `directiveMap = new DirectiveMap()`.
    directive_map: RefCell<DirectiveMap>,
    /// Java private final `sectionArray = new ArrayList<DirectiveDescrSection>()`.
    section_array: RefCell<Vec<Rc<DirectiveDescrSection>>>,
    /// Java private `fileTypeExists = new boolean[DirectiveFileType.NUM]`.
    file_type_exists: RefCell<Vec<bool>>,
    /// Java private final `droppedDirectives = new ArrayList<String>()`.
    dropped_directives: RefCell<Vec<String>>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `type`.
    r#type: Option<DirectiveFileType>,
    /// Java private `debug`, initialised to false.  Never read in the source.
    debug: Cell<bool>,
    /// Java private `otherSection`, initialised to null.
    other_section: RefCell<Option<Rc<DirectiveDescrSection>>>,
}

impl DirectiveEditorBuilder {
    /// Java `DirectiveEditorBuilder(BaseManager, DirectiveFileType)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        r#type: Option<DirectiveFileType>,
    ) -> DirectiveEditorBuilder {
        DirectiveEditorBuilder {
            directive_map: RefCell::new(DirectiveMap::new()),
            section_array: RefCell::new(Vec::new()),
            file_type_exists: RefCell::new(vec![false; directive_file_type::NUM as usize]),
            dropped_directives: RefCell::new(Vec::new()),
            manager,
            r#type,
            debug: Cell::new(false),
            other_section: RefCell::new(None),
        }
    }

    /// Java `build(AxisType, StringBuffer)`.  Builds a list of directives.  Gets the
    /// directive names and descriptions from directives.csv and the local directive
    /// file matching the type member variable.  Gets the values and default values from
    /// ApplicationManager and the .com and origcoms/.com files.  `errmsg` may be null;
    /// returns errmsg.
    pub fn build(&self, source_axis_type: AxisType, errmsg: Option<String>) -> String {
        // reset
        self.directive_map.borrow_mut().clear();
        self.section_array.borrow_mut().clear();
        for i in 0..directive_file_type::NUM as usize {
            self.file_type_exists.borrow_mut()[i] = false;
        }
        *self.other_section.borrow_mut() = None;
        // Load and update directives, and load sections.
        let mut errmsg = errmsg.unwrap_or_default();
        // Load the directives from directives.csv.
        let descr_iterator =
            directive_descr_file::INSTANCE.get_iterator(Some(self.manager), Some(AXID_ID));
        if let Some(mut descr_iterator) = descr_iterator {
            // Skip title and column header
            descr_iterator.has_next();
            descr_iterator.has_next();
            let mut section: Option<Rc<DirectiveDescrSection>> = None;
            let mut d_count = 0;
            let mut matches_type = false;
            while descr_iterator.has_next() {
                let Some(element) = descr_iterator.next_element() else {
                    break;
                };
                // Get the section - directives following this section are associated
                // with it.
                if element.is_section() {
                    if let Some(section) = &section {
                        // Record whether any of the directives in the section match the
                        // directive file type. Directives that don't match the type
                        // cannot be edited in this editor.
                        section.set_contains_editable_directives(matches_type);
                        if d_count == 0 {
                            let mut section_array = self.section_array.borrow_mut();
                            if let Some(index) = section_array
                                .iter()
                                .position(|listed| Rc::ptr_eq(listed, section))
                            {
                                section_array.remove(index);
                            }
                        }
                    }
                    let new_section = Rc::new(DirectiveDescrSection::new(
                        element.get_section_header().as_deref(),
                    ));
                    matches_type = false;
                    d_count = 0;
                    self.section_array.borrow_mut().push(new_section.clone());
                    section = Some(new_section);
                } else if element.is_directive() {
                    // Get the directive
                    let directive = Directive::new_directive_descr(element);
                    // Only save valid directives.
                    if directive.is_valid() {
                        // Check to see if this directive matches the type
                        if !matches_type
                            && (self.r#type != Some(DirectiveFileType::Batch)
                                && (directive.is_template() || !directive.is_batch()))
                            || (self.r#type == Some(DirectiveFileType::Batch)
                                && (!directive.is_template() || directive.is_batch()))
                        {
                            matches_type = true;
                        }
                        d_count += 1;
                        let directive = Arc::new(directive);
                        let key = directive.get_key();
                        // Store the directive in the map
                        self.directive_map
                            .borrow_mut()
                            .put(key.as_deref().unwrap_or("null"), directive.clone());
                        // Store the directive under current section - directives are
                        // always stored under sections, so section should not be null.
                        if let Some(section) = &section {
                            section.add_directive(Some(&directive));
                        } else {
                            eprintln!(
                                "Error: directive {}, not inside a section in the directives.csv file.  It cannot be edited.",
                                directive.get_title().as_deref().unwrap_or("null")
                            );
                        }
                    }
                }
            }
            directive_descr_file::INSTANCE.release_iterator(&descr_iterator);
        }
        // Add information and undocumented directives from the current directive files.
        self.update_from_local_directive_file(Some(DirectiveFileType::Scope), &mut errmsg);
        self.update_from_local_directive_file(Some(DirectiveFileType::System), &mut errmsg);
        self.update_from_local_directive_file(Some(DirectiveFileType::User), &mut errmsg);
        self.update_from_local_directive_file(Some(DirectiveFileType::Batch), &mut errmsg);
        // update setupset and runtime
        self.manager
            .update_directive_map_directive_map(&self.directive_map.borrow(), &mut errmsg);
        // update paramMap from *.com and origcoms/*.com
        // Get a sorted list of comparam directive names
        let mut iterator = self
            .directive_map
            .borrow()
            .key_set(Some(DirectiveType::COM_PARAM))
            .iterator();
        let first_axis_id = if source_axis_type == AxisType::DualAxis {
            AxisID::First
        } else {
            AxisID::Only
        };
        // A or only axis
        let mut com_file = ComFile::new(self.manager, first_axis_id);
        let mut command_map: Option<CommandMap> = None;
        let mut com_file_defaults =
            ComFile::new_subdirectory(self.manager, first_axis_id, Some("origcoms"));
        let mut command_map_defaults: Option<CommandMap> = None;
        let mut directive_name = DirectiveName::new();
        while iterator.has_next() {
            directive_name.set_key_string(iterator.next().as_deref());
            // save values
            command_map = self.get_command_map(
                &directive_name,
                &mut com_file,
                command_map,
                &mut errmsg,
                false,
            );
            if let Some(command_map) = &command_map {
                self.set_directive_value(Some(command_map), &directive_name, false);
            }
            // save default values
            command_map_defaults = self.get_command_map(
                &directive_name,
                &mut com_file_defaults,
                command_map_defaults,
                &mut errmsg,
                true,
            );
            if let Some(command_map_defaults) = &command_map_defaults {
                self.set_directive_value(Some(command_map_defaults), &directive_name, true);
            }
        }
        errmsg
    }

    // Updates done

    /// Java private `updateFromLocalDirectiveFile(DirectiveFileType, StringBuffer)`.
    /// Updates existing directives and loads ones that are not in directive.csv.
    /// Returns true if the local directive file exists.
    fn update_from_local_directive_file(
        &self,
        r#type: Option<DirectiveFileType>,
        errmsg: &mut String,
    ) -> bool {
        let Some(r#type) = r#type else {
            return false;
        };
        let Some(directive_file) = r#type.get_local_file(Some(self.manager), Some(AXID_ID)) else {
            return false;
        };
        if !directive_file.exists() {
            return false;
        }
        self.file_type_exists.borrow_mut()[r#type.get_index() as usize] = true;
        let result: Result<bool, LogFileError> = (|| {
            let statement_list = unsafe {
                autodoc_factory::get_instance_file(Some(self.manager), Some(&directive_file), false)
            }?;
            if statement_list.is_null() {
                return Ok(false);
            }
            let statement_list = unsafe { &*statement_list };
            let mut location = ReadOnlyStatementList::get_statement_location(statement_list);
            let mut directive_name = DirectiveName::new();
            loop {
                let statement = unsafe {
                    ReadOnlyStatementList::next_statement(statement_list, location.as_mut())
                };
                if statement.is_null() {
                    break;
                }
                let statement = unsafe { &*statement };
                if statement.get_type() != statement::Type::NameValuePair {
                    continue;
                }
                let left_side = statement.get_left_side();
                let axis_id = directive_name.set_key_string(left_side.as_deref());
                let mut directive = self
                    .directive_map
                    .borrow()
                    .get_directive_string(directive_name.get_key().as_deref());
                if directive.is_none() {
                    // Handle undefined directives.
                    let undefined = Arc::new(Directive::new_directive_name(&directive_name));
                    // Ignoring B axis values and directives.
                    if axis_id == Some(AxisID::Second) {
                        break;
                    }
                    // Only comparam directives are used generically by batchruntomo, so
                    // undefined ones may be useful.
                    if undefined.is_valid()
                        && undefined.get_type() == Some(DirectiveType::COM_PARAM)
                    {
                        // If otherSection hasn't been created, create it and add it to
                        // sectionArray.
                        let other_section = self
                            .other_section
                            .borrow_mut()
                            .get_or_insert_with(|| {
                                let other_section =
                                    Rc::new(DirectiveDescrSection::new(Some(SECTION_OTHER_HEADER)));
                                self.section_array.borrow_mut().push(other_section.clone());
                                other_section
                            })
                            .clone();
                        let key = undefined.get_key();
                        other_section.add_string(key.as_deref());
                        self.directive_map
                            .borrow_mut()
                            .put(key.as_deref().unwrap_or("null"), undefined.clone());
                        directive = Some(undefined);
                    } else {
                        // Save directives that don't go into the editor.
                        self.dropped_directives
                            .borrow_mut()
                            .push(left_side.clone().unwrap_or_else(|| "null".to_string()));
                    }
                }
                // Upstream bug fixed in translation (DirectiveEditorBuilder.java:243):
                // for a dropped directive Java goes on to `directive.getValueType()`
                // with `directive` null, and the NullPointerException aborts building
                // the editor.  A dropped directive has no value to set; the next
                // statement is read.
                let Some(directive) = directive else {
                    continue;
                };
                directive.set_in_directive_file(Some(r#type), true);
                // Set the value from the file. This value will be overridden by
                // subsequent templates, and then from the coms or from the non-generic
                // code in the manager.
                let value_type = directive.get_value_type();
                let value = statement.get_right_side();
                if value_type != Some(DirectiveValueType::Boolean) {
                    // A blank value is an override.
                    directive.set_value_string(value.as_deref());
                } else if value.is_some() {
                    // A boolean value cannot be empty.
                    directive.set_value_string(value.as_deref());
                }
            }
            Ok(true)
        })();
        match result {
            Ok(retval) => retval,
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => false,
            // `catch (final LogFileException e)` and `catch (final IOException e)`.
            Err(e) => {
                if matches!(e, LogFileError::Io(_)) {
                    // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                    eprintln!("{}", e);
                }
                errmsg.push_str(&format!(
                    "Unable to load {}.  {}  ",
                    directive_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    e.get_message()
                ));
                false
            }
        }
    }

    /// Java private `getCommandMap(DirectiveName, ComFile, Map, StringBuffer, boolean)`.
    /// Uses the directiveName as a guide and decides whether to open a com file or
    /// create a new commandMap.  If neither of these things are necessary, returns the
    /// existing commandMap.
    fn get_command_map(
        &self,
        directive_name: &DirectiveName,
        com_file: &mut ComFile,
        mut command_map: Option<CommandMap>,
        errmsg: &mut String,
        origcoms: bool,
    ) -> Option<CommandMap> {
        // Get a new comfile and origcoms comfile each time the comfile name changes in
        // the sorted list of comparam directives.
        let com_file_name = directive_name.get_com_file_name();
        if !com_file.equals_com_file_name(com_file_name.as_deref()) {
            com_file.set_com_file_name(com_file_name.as_deref());
        }
        // Get a new program from the comfile and origcoms comfile each time the program
        // name or comfile name changes in the sorted list of comparam directives.
        let program_name = directive_name.get_program_name();
        if !com_file.equals_program_name(program_name.as_deref()) {
            command_map = com_file.get_command_map(program_name.as_deref(), errmsg);
            // Handled special case setupset directives.
            if origcoms
                && com_file.equals_com_file_name(Some(SPECIAL_CASE_PROGRAM_NAME))
                && com_file.equals_program_name(Some(SPECIAL_CASE_PROGRAM_NAME))
            {
                self.set_special_case_default_values(command_map.as_ref());
            }
        }
        command_map
    }

    /// Java private `setSpecialCaseDefaultValues(Map)`.  Handles cases where a setupset
    /// directive has a default value that is stored in a .com file.
    fn set_special_case_default_values(&self, command_map: Option<&CommandMap>) {
        let mut from_directive_name = DirectiveName::new();
        let mut to_directive_name = DirectiveName::new();
        let from_prefix = "comparam.";
        let to_prefix = "setupset.copyarg.";
        from_directive_name.set_key_string(Some(&format!(
            "{from_prefix}{SPECIAL_CASE_PROGRAM_NAME}.{SPECIAL_CASE_PROGRAM_NAME}.ExpectedDefocus"
        )));
        to_directive_name.set_key_string(Some(&format!("{to_prefix}defocus")));
        self.set_default_directive_value(command_map, &from_directive_name, &to_directive_name);
        from_directive_name.set_key_string(Some(&format!(
            "{from_prefix}{SPECIAL_CASE_PROGRAM_NAME}.{SPECIAL_CASE_PROGRAM_NAME}.Voltage"
        )));
        to_directive_name.set_key_string(Some(&format!("{to_prefix}voltage")));
        self.set_default_directive_value(command_map, &from_directive_name, &to_directive_name);
        from_directive_name.set_key_string(Some(&format!(
            "{from_prefix}{SPECIAL_CASE_PROGRAM_NAME}.{SPECIAL_CASE_PROGRAM_NAME}.SphericalAberration"
        )));
        to_directive_name.set_key_string(Some(&format!("{to_prefix}Cs")));
        self.set_default_directive_value(command_map, &from_directive_name, &to_directive_name);
    }

    /// Java private `setDefaultDirectiveValue(Map, DirectiveName, DirectiveName)`.
    /// Using fromDirectiveName as a key, gets a value from commandMap.  Using
    /// toDirectiveName as a key, gets a directive from directiveMap.  Places the value
    /// in the default value of the directive.
    fn set_default_directive_value(
        &self,
        command_map: Option<&CommandMap>,
        from_directive_name: &DirectiveName,
        to_directive_name: &DirectiveName,
    ) {
        let Some(command_map) = command_map else {
            return;
        };
        // Pull out the default value from the command map, using the "from" directive
        // name.
        let parameter_name = from_directive_name.get_parameter_name();
        if let Some(value) = parameter_name
            .as_deref()
            .and_then(|parameter_name| command_map.get(parameter_name))
        {
            // Upstream bug fixed in translation (DirectiveEditorBuilder.java:325): Java
            // dereferences the "to" directive unchecked; a directives.csv without it
            // would throw NullPointerException.  Nothing is set then.
            let Some(to_directive) = self
                .directive_map
                .borrow()
                .get_directive_string(to_directive_name.get_key().as_deref())
            else {
                return;
            };
            // Set the command map value in the "to" directive
            match value {
                Some(value) => to_directive.set_default_value_string(Some(value)),
                // no value - treat as a boolean
                None => to_directive.set_default_value_boolean(true),
            }
        }
    }

    /// Java private `setDirectiveValue(Map, DirectiveName, boolean)`.  Using
    /// directiveName as the key, gets a value from commandMap and gets a directive from
    /// directiveMap.  Places the value in the directive.
    fn set_directive_value(
        &self,
        command_map: Option<&CommandMap>,
        directive_name: &DirectiveName,
        is_default_value: bool,
    ) {
        let Some(command_map) = command_map else {
            return;
        };
        // Pull out the value or default value from the program command.
        let parameter_name = directive_name.get_parameter_name();
        if let Some(value) = parameter_name
            .as_deref()
            .and_then(|parameter_name| command_map.get(parameter_name))
        {
            // The key comes from the map's own comparam key set, so the directive is
            // there.
            let Some(directive) = self
                .directive_map
                .borrow()
                .get_directive_string(directive_name.get_key().as_deref())
            else {
                return;
            };
            // Set value in the directive
            match value {
                Some(value) => {
                    if !is_default_value {
                        directive.set_value_string(Some(value));
                    } else {
                        directive.set_default_value_string(Some(value));
                    }
                }
                None => {
                    // no value - treat as a boolean
                    if !is_default_value {
                        directive.set_value_boolean(true);
                    } else {
                        directive.set_default_value_boolean(true);
                    }
                }
            }
        }
    }

    /// Java `getSectionArray()`.
    pub fn get_section_array(&self) -> Vec<Rc<DirectiveDescrSection>> {
        self.section_array.borrow().clone()
    }

    /// Java `getFileTypeExists()`.
    pub fn get_file_type_exists(&self) -> Vec<bool> {
        self.file_type_exists.borrow().clone()
    }

    /// Java `getDroppedDirectives()`.
    pub fn get_dropped_directives(&self) -> Vec<String> {
        self.dropped_directives.borrow().clone()
    }

    /// Java `getDefaultSaveLocation()`.  If the type is USER, will attempt to create the
    /// user directory if necessary.
    pub fn get_default_save_location(&self) -> PathBuf {
        let mut dir: Option<PathBuf> = None;
        if self.r#type == Some(DirectiveFileType::User) {
            dir = etomo_director::INSTANCE.with_user_configuration(|user_config| {
                if user_config.is_user_template_dir_set() {
                    user_config.get_user_template_dir().map(PathBuf::from)
                } else {
                    config_tool::get_default_user_template_dir()
                }
            });
            if let Some(dir) = &dir
                && !dir.exists()
            {
                eprintln!(
                    "Creating user template directory:{}",
                    java_io_file_get_absolute_path(&dir.to_string_lossy())
                );
                let _ = std::fs::create_dir_all(dir);
            }
        }
        if let Some(dir) = dir {
            return dir;
        }
        PathBuf::from(
            self.manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string()),
        )
    }

    /// Java `getDirectiveMap()`.
    pub fn get_directive_map(&self) -> std::cell::Ref<'_, DirectiveMap> {
        self.directive_map.borrow()
    }
}
