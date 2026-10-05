//! `IMOD/Etomo/src/etomo/storage/DirectiveFile.java` (with its nested
//! `StatementIterator`, `Module`, `Comfile` and `Command`).
//!
//! Copyright: Copyright 2012 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  The autodoc translation keeps its registry per thread and hands out raw
//! pointers, so the autodoc and the cached parent attributes (`copyArg`, `runtime`,
//! `setupSet`, `comparam`) are raw pointers here (null for Java null), and a
//! `DirectiveFile` belongs to the thread that loaded it.  `getParentAttribute` fills the
//! caches lazily from methods the source calls on a shared instance, so the caches are
//! `Cell`s and those methods take `&self`.  Java overloads carry descriptive suffixes
//! naming their extra parameters (`contains_axis_template`, `get_value_index`, ...); a
//! null `DirectiveDef` is `None`.  The class implements `DirectiveFileInterface`
//! (`directive_file_interface.rs`); `DirectiveAttribute`'s `getMatch`,
//! `AttributeMatch` and `Match` are in `directive_attribute.rs`.

use super::autodoc::attribute::Attribute;
use super::autodoc::autodoc::Autodoc;
use super::autodoc::autodoc_factory;
use super::autodoc::read_only_attribute::ReadOnlyAttribute;
use super::autodoc::read_only_attribute_iterator::ReadOnlyAttributeIterator;
use super::autodoc::read_only_attribute_list::ReadOnlyAttributeList;
use super::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use super::autodoc::read_only_statement_list::ReadOnlyStatementList;
use super::autodoc::statement::Statement;
use super::autodoc::statement_location::StatementLocation;
use super::directive_attribute::{self, AttributeMatch, Match};
use super::directive_def::DirectiveDef;
use super::directive_type::DirectiveType;
use super::directive_value::DirectiveValue;
use super::log_file::LogFileError;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_string_split};
use regex::Regex;
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Java `AUTO_FIT_RANGE_INDEX`.
pub const AUTO_FIT_RANGE_INDEX: i32 = 0;
/// Java `AUTO_FIT_STEP_INDEX`.
pub const AUTO_FIT_STEP_INDEX: i32 = 1;

/// Java `","` used as a `split` regex.
static COMMA: LazyLock<Regex> = LazyLock::new(|| Regex::new(",").unwrap());

/// Java `DirectiveFile`.
pub struct DirectiveFile {
    /// Java private final field `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final field `manager`.
    manager: &'static dyn BaseManager,
    /// Java private field `file`, initialised to null.
    file: Option<PathBuf>,
    /// Java private field `directiveFileType`, initialised to null.
    directive_file_type: Option<DirectiveFileType>,
    /// Java private field `copyArg`, initialised to null.
    copy_arg: Cell<*mut Attribute>,
    /// Java private field `runtime`, initialised to null.
    runtime: Cell<*mut Attribute>,
    /// Java private field `setupSet`, initialised to null.
    setup_set: Cell<*mut Attribute>,
    /// Java private field `comparam`, initialised to null.
    comparam: Cell<*mut Attribute>,
    /// Java private field `copyArgSet`, initialised to false.
    copy_arg_set: Cell<bool>,
    /// Java private field `runtimeSet`, initialised to false.
    runtime_set: Cell<bool>,
    /// Java private field `setupSetSet`, initialised to false.
    setup_set_set: Cell<bool>,
    /// Java private field `comparamSet`, initialised to false.
    comparam_set: Cell<bool>,
    /// Java private field `autodoc`, initialised to null.
    autodoc: *mut Autodoc,
    /// Java private field `debug`, initialised to false.
    debug: bool,
}

impl DirectiveFile {
    /// Java private constructor `DirectiveFile(BaseManager, AxisID)`.
    fn new(manager: &'static dyn BaseManager, axis_id: Option<AxisID>) -> DirectiveFile {
        DirectiveFile {
            axis_id,
            manager,
            file: None,
            directive_file_type: None,
            copy_arg: Cell::new(std::ptr::null_mut()),
            runtime: Cell::new(std::ptr::null_mut()),
            setup_set: Cell::new(std::ptr::null_mut()),
            comparam: Cell::new(std::ptr::null_mut()),
            copy_arg_set: Cell::new(false),
            runtime_set: Cell::new(false),
            setup_set_set: Cell::new(false),
            comparam_set: Cell::new(false),
            autodoc: std::ptr::null_mut(),
            debug: false,
        }
    }

    /// Java `iterator(boolean)` (DirectiveFileInterface).
    pub fn iterator(&self, template_only: bool) -> Option<StatementIterator<'_>> {
        if self
            .directive_file_type
            .is_some_and(|directive_file_type| directive_file_type.is_batch())
            && template_only
        {
            return None;
        }
        Some(StatementIterator::new(self))
    }

    /// Java static `getArgInstance(BaseManager, AxisID)`.  Returns an instance loaded
    /// with the batch directive from the etomo parameters.  Returns null if an autodoc
    /// could not be loaded.
    pub fn get_arg_instance(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> Option<DirectiveFile> {
        let mut instance = DirectiveFile::new(manager, axis_id);
        let directive = etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_directive()
            .map(Path::to_path_buf);
        if instance.set_file(directive.as_deref(), Some(DirectiveFileType::Batch)) {
            return Some(instance);
        }
        None
    }

    /// Java static `getInstance(BaseManager, AxisID, File, DirectiveFileType)`.  Returns
    /// an instance loaded with the file parameter.  Returns null if an autodoc could not
    /// be loaded.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file: Option<&Path>,
        directive_file_type: DirectiveFileType,
    ) -> Option<DirectiveFile> {
        let mut instance = DirectiveFile::new(manager, axis_id);
        if instance.set_file(file, Some(directive_file_type)) {
            return Some(instance);
        }
        None
    }

    /// Java package-private `getAttribute(Match, DirectiveDef, AxisID, boolean)`.
    /// Returns an attribute from this directive file that matches the parameters.
    pub(crate) fn get_attribute_with_match(
        &self,
        r#match: Match,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        ignore_file_type: bool,
    ) -> Option<AttributeMatch> {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(axis_id));
        }
        let batch = self
            .directive_file_type
            .is_some_and(|directive_file_type| directive_file_type.is_batch());
        let directive_def = directive_def?;
        if !ignore_file_type
            && ((batch && !directive_def.is_batch(axis_id))
                || (!batch && !directive_def.is_template(axis_id)))
        {
            return None;
        }
        let parent_attribute = self.get_parent_attribute_def(Some(directive_def));
        directive_attribute::get_match(
            r#match,
            self,
            unsafe { parent_attribute.as_ref() },
            directive_def,
            axis_id,
        )
    }

    /// Java package-private `getAttribute(DirectiveDef, AxisID, boolean, boolean)`.
    /// Returns the best matching AttributeMatch.
    pub(crate) fn get_attribute(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        ignore_file_type: bool,
        include_override: bool,
    ) -> Option<AttributeMatch> {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(axis_id));
        }
        let batch = self
            .directive_file_type
            .is_some_and(|directive_file_type| directive_file_type.is_batch());
        let def = directive_def?;
        if !ignore_file_type
            && ((batch && !def.is_batch(axis_id)) || (!batch && !def.is_template(axis_id)))
        {
            return None;
        }
        let attribute_match =
            self.get_attribute_with_match(Match::Primary, directive_def, axis_id, ignore_file_type);
        if let Some(attribute_match) = attribute_match
            && !attribute_match.is_empty()
        {
            if attribute_match.is_override() && !include_override {
                return None;
            }
            return Some(attribute_match);
        }
        let attribute_match = self.get_attribute_with_match(
            Match::Secondary,
            directive_def,
            axis_id,
            ignore_file_type,
        );
        if let Some(attribute_match) = attribute_match
            && !attribute_match.is_empty()
        {
            if attribute_match.is_override() && !include_override {
                return None;
            }
            return Some(attribute_match);
        }
        None
    }

    /// Java `contains(DirectiveDef, AxisID)`.
    pub fn contains_axis(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.contains_axis_template(directive_def, axis_id, false)
    }

    /// Java `contains(DirectiveDef, AxisID, boolean)` (DirectiveFileInterface).
    pub fn contains_axis_template(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
    ) -> bool {
        self.contains_axis_template_ignore(directive_def, axis_id, template_only, false)
    }

    /// Java `contains(DirectiveDef, AxisID, boolean, boolean)` (DirectiveFileInterface).
    pub fn contains_axis_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> bool {
        self.contains_axis_template_ignore_override(
            directive_def,
            axis_id,
            template_only,
            ignore_file_type,
            false,
        )
    }

    /// Java `contains(DirectiveDef, AxisID, boolean, boolean, boolean)`.  Returns true
    /// if the directive file contains the directive attribute and does not override it
    /// (an empty value for a non-boolean directive).
    pub fn contains_axis_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> bool {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(axis_id));
        }
        if template_only
            && self
                .directive_file_type
                .is_some_and(|directive_file_type| directive_file_type.is_batch())
        {
            return false;
        }
        let attribute =
            self.get_attribute(directive_def, axis_id, ignore_file_type, include_override);
        attribute.is_some()
    }

    /// Java `containsValue(DirectiveDef)`.
    pub fn contains_value(&self, directive_def: Option<DirectiveDef>) -> bool {
        let attribute = self.get_attribute(directive_def, None, false, false);
        if let Some(attribute) = attribute {
            let value = attribute.get_value();
            return value
                .as_deref()
                .is_some_and(|value| !java_lang_string_matches_whitespace(value));
        }
        false
    }

    /// Java `contains(DirectiveDef)` (DirectiveFileInterface).  Returns true if the
    /// directive file contains the directive attribute and does not override it (an
    /// empty value for a non-boolean directive).
    pub fn contains(&self, directive_def: Option<DirectiveDef>) -> bool {
        self.contains_axis_template(directive_def, None, false)
    }

    /// Java `contains(DirectiveDef, boolean)` (DirectiveFileInterface).
    pub fn contains_template(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
    ) -> bool {
        // template/batch issues are already being handled in getAttribute
        self.contains_axis_template(directive_def, None, template_only)
    }

    /// Java `contains(DirectiveDef, boolean, boolean)` (DirectiveFileInterface).
    pub fn contains_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> bool {
        // template/batch issues are already being handled in getAttribute
        self.contains_axis_template_ignore(directive_def, None, template_only, ignore_file_type)
    }

    /// Java `contains(DirectiveDef, boolean, boolean, boolean)` (DirectiveFileInterface).
    pub fn contains_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> bool {
        // template/batch issues are already being handled in getAttribute
        self.contains_axis_template_ignore_override(
            directive_def,
            None,
            template_only,
            ignore_file_type,
            include_override,
        )
    }

    /// Java package-private `getDirectiveFileType`.
    pub(crate) fn get_directive_file_type(&self) -> Option<DirectiveFileType> {
        self.directive_file_type
    }

    /// Java `getValue(DirectiveDef, AxisID)`.
    pub fn get_value_axis(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        let directive_value =
            self.get_value_axis_template_ignore(directive_def, axis_id, false, false)?;
        directive_value.get_value()
    }

    /// Java `getValue(DirectiveDef, AxisID, boolean, boolean)` (DirectiveFileInterface).
    pub fn get_value_axis_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> Option<AttributeMatch> {
        self.get_value_axis_template_ignore_override(
            directive_def,
            axis_id,
            template_only,
            ignore_file_type,
            false,
        )
    }

    /// Java `getValue(DirectiveDef, AxisID, boolean, boolean, boolean)`.  Return the
    /// value of the attribute in the directive file.
    pub fn get_value_axis_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> Option<AttributeMatch> {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(axis_id));
        }
        if template_only
            && self
                .directive_file_type
                .is_some_and(|directive_file_type| directive_file_type.is_batch())
        {
            return None;
        }
        let attribute =
            self.get_attribute(directive_def, axis_id, ignore_file_type, include_override);
        if attribute.is_some() {
            return attribute;
        }
        None
    }

    /// Java `getValue(DirectiveDef)` (DirectiveFileInterface).  Return the value of the
    /// attribute in the directive file.
    pub fn get_value(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(None));
        }
        let directive_value =
            self.get_value_axis_template_ignore(directive_def, None, false, false)?;
        directive_value.get_value()
    }

    /// Java `getValue(DirectiveDef, boolean)` (DirectiveFileInterface).
    pub fn get_value_template(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
    ) -> Option<String> {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(None));
        }
        // template/batch issues are already being handled in getAttribute
        let directive_value =
            self.get_value_axis_template_ignore(directive_def, None, template_only, false)?;
        directive_value.get_value()
    }

    /// Java `getValue(DirectiveDef, boolean, boolean)` (DirectiveFileInterface).
    pub fn get_value_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> Option<AttributeMatch> {
        // template/batch issues are already being handled in getAttribute
        self.get_value_axis_template_ignore(directive_def, None, template_only, ignore_file_type)
    }

    /// Java `getValue(DirectiveDef, boolean, boolean, boolean)` (DirectiveFileInterface).
    pub fn get_value_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> Option<AttributeMatch> {
        // template/batch issues are already being handled in getAttribute
        self.get_value_axis_template_ignore_override(
            directive_def,
            None,
            template_only,
            ignore_file_type,
            include_override,
        )
    }

    /// Java `getValue(DirectiveDef, int)` (DirectiveFileInterface).
    pub fn get_value_index(
        &self,
        directive_def: Option<DirectiveDef>,
        index: i32,
    ) -> Option<String> {
        let mut directive_def = directive_def;
        if let Some(def) = directive_def {
            directive_def = Some(def.get_axis_id_instance(None));
        }
        if index < 0 {
            return None;
        }
        let value = self.get_value(directive_def);
        // Get the element specified by index
        if let Some(value) = value {
            let divider = ",";
            if value.contains(divider) {
                let array = java_lang_string_split(&value, &COMMA);
                if (index as usize) < array.len() {
                    return Some(array[index as usize].clone());
                }
            }
            if index == 0 {
                return Some(value);
            }
        }
        None
    }

    /// Java `isValue(DirectiveDef, AxisID)`.
    pub fn is_value_axis(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.is_value_axis_template(directive_def, axis_id, false)
    }

    /// Java `isValue(DirectiveDef, AxisID, boolean)` (DirectiveFileInterface).
    pub fn is_value_axis_template(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
    ) -> bool {
        self.is_value_axis_template_ignore(directive_def, axis_id, template_only, false)
    }

    /// Java `isValue(DirectiveDef, AxisID, boolean, boolean)`.  Return the boolean value
    /// of the attribute in the directive file.
    pub fn is_value_axis_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> bool {
        if template_only
            && self
                .directive_file_type
                .is_some_and(|directive_file_type| directive_file_type.is_batch())
        {
            return false;
        }
        let attribute = self.get_attribute(directive_def, axis_id, ignore_file_type, false);
        if let Some(attribute) = attribute {
            return attribute.is_value();
        }
        false
    }

    /// Java `isValue(DirectiveDef)` (DirectiveFileInterface).
    pub fn is_value(&self, directive_def: Option<DirectiveDef>) -> bool {
        self.is_value_axis_template(directive_def, None, false)
    }

    /// Java `isValue(DirectiveDef, boolean)`.
    pub fn is_value_template(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
    ) -> bool {
        self.is_value_axis_template_ignore(directive_def, None, template_only, false)
    }

    /// Java `isValue(DirectiveDef, boolean, boolean)` (DirectiveFileInterface).
    pub fn is_value_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> bool {
        self.is_value_axis_template_ignore(directive_def, None, template_only, ignore_file_type)
    }

    /// Java `setFile(File, DirectiveFileType)`.  Loads the file and opens the autodoc.
    /// Resets the instance so that it will read from the new autodoc.  Returns true if
    /// the autodoc was opened successfully, false if the autodoc open failed.
    ///
    /// The Rust autodoc factory takes a non-null `AxisID`; a null axis is passed as
    /// `AxisID.ONLY`.
    pub fn set_file(
        &mut self,
        directive_file: Option<&Path>,
        directive_file_type: Option<DirectiveFileType>,
    ) -> bool {
        self.file = directive_file.map(Path::to_path_buf);
        self.directive_file_type = directive_file_type;
        self.copy_arg_set.set(false);
        self.runtime_set.set(false);
        self.setup_set_set.set(false);
        self.comparam_set.set(false);
        self.copy_arg.set(std::ptr::null_mut());
        self.runtime.set(std::ptr::null_mut());
        self.setup_set.set(std::ptr::null_mut());
        self.comparam.set(std::ptr::null_mut());
        match unsafe {
            autodoc_factory::get_instance_file_axis_id(
                Some(self.manager),
                self.file.as_deref(),
                self.axis_id.unwrap_or(AxisID::Only),
                false,
            )
        } {
            Ok(autodoc) => self.autodoc = autodoc,
            // `catch (final LockException e) { return false; }`.
            Err(LogFileError::Lock(_)) => return false,
            Err(e) => {
                ui_harness::post_message_dialog(
                    Some(self.manager),
                    e.to_string(),
                    "Directive File Read Failure".to_string(),
                    None,
                );
                return false;
            }
        }
        true
    }

    /// Java private `getParentAttribute(DirectiveDef)`.
    fn get_parent_attribute_def(&self, directive_def: Option<DirectiveDef>) -> *mut Attribute {
        match directive_def {
            None => std::ptr::null_mut(),
            Some(directive_def) => {
                self.get_parent_attribute(Some(directive_def.get_directive_type()))
            }
        }
    }

    /// Java private `getParentAttribute(DirectiveType)`.
    fn get_parent_attribute(&self, r#type: Option<DirectiveType>) -> *mut Attribute {
        let mut parent_attribute: *mut Attribute = std::ptr::null_mut();
        let autodoc: Option<&Autodoc> = unsafe { self.autodoc.as_ref() };
        if r#type == Some(DirectiveType::COPY_ARG) {
            if !self.copy_arg_set.get() {
                self.copy_arg_set.set(true);
                self.setup_set
                    .set(self.get_parent_attribute(Some(DirectiveType::SETUP_SET)));
                if let Some(setup_set) = unsafe { self.setup_set.get().as_ref() }
                    && autodoc.is_some()
                {
                    self.copy_arg.set(unsafe {
                        setup_set.get_attribute_by_name(Some(&DirectiveType::COPY_ARG.to_string()))
                    });
                }
            }
            parent_attribute = self.copy_arg.get();
        } else if r#type == Some(DirectiveType::SETUP_SET) {
            if !self.setup_set_set.get() {
                self.setup_set_set.set(true);
                if let Some(autodoc) = autodoc {
                    self.setup_set.set(unsafe {
                        ReadOnlyAutodoc::get_attribute(
                            autodoc,
                            Some(&DirectiveType::SETUP_SET.to_string()),
                        )
                    });
                }
            }
            parent_attribute = self.setup_set.get();
        } else if r#type == Some(DirectiveType::RUN_TIME) {
            if !self.runtime_set.get() {
                self.runtime_set.set(true);
                if let Some(autodoc) = autodoc {
                    self.runtime.set(unsafe {
                        ReadOnlyAutodoc::get_attribute(
                            autodoc,
                            Some(&DirectiveType::RUN_TIME.to_string()),
                        )
                    });
                }
            }
            parent_attribute = self.runtime.get();
        } else if r#type == Some(DirectiveType::COM_PARAM) {
            if !self.comparam_set.get() {
                self.comparam_set.set(true);
                if let Some(autodoc) = autodoc {
                    self.comparam.set(unsafe {
                        ReadOnlyAutodoc::get_attribute(
                            autodoc,
                            Some(&DirectiveType::COM_PARAM.to_string()),
                        )
                    });
                }
            }
            parent_attribute = self.comparam.get();
        }
        parent_attribute
    }

    /// Java `setDebug(boolean)` (DirectiveFileInterface).
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java package-private `getCopyArgIterator`.
    pub(crate) fn get_copy_arg_iterator(&self) -> Option<ReadOnlyAttributeIterator<'_>> {
        let copy_arg = unsafe { self.copy_arg.get().as_ref() }?;
        let list = unsafe { copy_arg.get_children().as_ref() }?;
        Some(list.iterator())
    }

    /// Java `getFile`.
    pub fn get_file(&self) -> Option<PathBuf> {
        self.file.clone()
    }
}

/// Java `toString`.  `Object.toString`'s identity hash is a JVM value; the instance
/// address stands in for it.
impl std::fmt::Display for DirectiveFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{},directiveFileType:{}]",
            match &self.file {
                Some(file) => format!(
                    "file:{}",
                    java_io_file_get_absolute_path(&file.to_string_lossy())
                ),
                None => format!(
                    "etomo.storage.DirectiveFile@{:x}",
                    self as *const DirectiveFile as usize
                ),
            },
            match self.directive_file_type {
                None => "null".to_string(),
                Some(directive_file_type) => directive_file_type.to_string(),
            }
        )
    }
}

/// Java private final inner class `StatementIterator implements
/// Iterator<ReadOnlyStatement>`.  Yields the autodoc's statements (non-null
/// `*mut dyn Statement`s, owned by the autodoc) in order.
pub struct StatementIterator<'a> {
    /// The outer instance.
    directive_file: &'a DirectiveFile,
    /// Java private field `loc`, initialised to null.
    loc: Option<StatementLocation>,
    /// Java private field `next`, initialised to null.
    next: Option<*mut dyn Statement>,
}

impl<'a> StatementIterator<'a> {
    /// Java private constructor `StatementIterator()`.
    fn new(directive_file: &'a DirectiveFile) -> StatementIterator<'a> {
        StatementIterator {
            directive_file,
            loc: None,
            next: None,
        }
    }

    /// Java `hasNext`.
    pub fn has_next(&mut self) -> bool {
        if self.next.is_some() {
            return true;
        }
        self.next = self.get_next();
        self.next.is_some()
    }

    /// Java private `getNext`.
    fn get_next(&mut self) -> Option<*mut dyn Statement> {
        let autodoc: &Autodoc = unsafe { self.directive_file.autodoc.as_ref() }?;
        if self.loc.is_none() {
            self.loc = ReadOnlyStatementList::get_statement_location(autodoc);
            self.loc.as_ref()?;
        }
        let temp = unsafe { ReadOnlyStatementList::next_statement(autodoc, self.loc.as_mut()) };
        if temp.is_null() {
            return None;
        }
        Some(temp)
    }
}

/// Java `next` (with `hasNext` folded into the `Option`).  `remove` is a no-op in the
/// source and has no counterpart.
impl Iterator for StatementIterator<'_> {
    type Item = *mut dyn Statement;

    fn next(&mut self) -> Option<*mut dyn Statement> {
        if let Some(temp) = self.next.take() {
            return Some(temp);
        }
        self.get_next()
    }
}

/// Java package-private static final nested class `Module`, a typesafe enum; the
/// instances are associated constants with their Java names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Module(&'static str);

impl Module {
    pub const ALIGNED_STACK: Module = Module("AlignedStack");
    pub const BEAD_TRACKING: Module = Module("BeadTracking");
    pub const COMBINE: Module = Module("Combine");
    pub const CTF_CORRECTION: Module = Module("CTFcorrection");
    pub const CTF_PLOTTING: Module = Module("CTFplotting");
    pub const EXCLUDE_VIEWS: Module = Module("Excludeviews");
    pub const FIDUCIALS: Module = Module("Fiducials");
    pub const GOLD_ERASING: Module = Module("GoldErasing");
    pub const NAD: Module = Module("NAD");
    pub const PATCH_TRACKING: Module = Module("PatchTracking");
    pub const POSITIONING: Module = Module("Positioning");
    pub const POSTPROCESS: Module = Module("Postprocess");
    pub const PREPROCESSING: Module = Module("Preprocessing");
    pub const RAPTOR: Module = Module("RAPTOR");
    pub const RECONSTRUCTION: Module = Module("Reconstruction");
    pub const RESTRICT_ALIGN: Module = Module("RestrictAlign");
    pub const REPLACE_STEP: Module = Module("ReplaceStep");
    pub const RUN_AFTER_STEP: Module = Module("RunAfterStep");
    pub const SEED_FINDING: Module = Module("SeedFinding");
    pub const TILT_ALIGNMENT: Module = Module("TiltAlignment");
    pub const TRIMVOL: Module = Module("Trimvol");
}

/// Java `Module.toString`: the tag.
impl std::fmt::Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Java package-private static final nested class `Comfile`, a typesafe enum; the
/// instances are associated constants with their Java names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Comfile(&'static str);

impl Comfile {
    pub const ALIGN: Comfile = Comfile("align");
    pub const AUTOFIDSEED: Comfile = Comfile("autofidseed");
    pub const CTF_3D_SETUP: Comfile = Comfile("ctf3dsetup");
    pub const CTF_PLOTTER: Comfile = Comfile("ctfplotter");
    pub const ERASER: Comfile = Comfile("eraser");
    pub const GOLD_ERASER: Comfile = Comfile("golderaser");
    pub const PREBLEND: Comfile = Comfile("preblend");
    pub const PRENEWST: Comfile = Comfile("prenewst");
    pub const SIRTSETUP: Comfile = Comfile("sirtsetup");
    pub const TILT: Comfile = Comfile("tilt");
    pub const TRACK: Comfile = Comfile("track");
    pub const XCORR: Comfile = Comfile("xcorr");
    pub const XCORR_PT: Comfile = Comfile("xcorr_pt");
}

/// Java `Comfile.toString`: the tag.
impl std::fmt::Display for Comfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// Java package-private static final nested class `Command`, a typesafe enum; the
/// instances are associated constants with their Java names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command(&'static str);

impl Command {
    pub const AUTOFIDSEED: Command = Command("autofidseed");
    pub const BLENDMONT: Command = Command("blendmont");
    pub const BEADTRACK: Command = Command("beadtrack");
    pub const CCDERASER: Command = Command("ccderaser");
    pub const CTF_3D_SETUP: Command = Command("ctf3dsetup");
    pub const CTF_PLOTTER: Command = Command("ctfplotter");
    pub const IMODCHOPCONTS: Command = Command("imodchopconts");
    pub const NEWSTACK: Command = Command("newstack");
    pub const SIRTSETUP: Command = Command("sirtsetup");
    pub const TILT: Command = Command("tilt");
    pub const TILTALIGN: Command = Command("tiltalign");
    pub const TILTXCORR: Command = Command("tiltxcorr");
}

/// Java `Command.toString`: the tag.
impl std::fmt::Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
