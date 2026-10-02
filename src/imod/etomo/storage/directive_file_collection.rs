//! `IMOD/Etomo/src/etomo/storage/DirectiveFileCollection.java`.
//!
//! **Shape.**  The Java class implements `SetupReconInterface` and
//! `DirectiveFileInterface`; neither interface has a Rust trait yet, so their methods
//! are inherent methods here.  Java overloads carry descriptive suffixes naming the
//! extra parameters (`contains_axis_template`, `get_value_index`, ...).  A null Java
//! `DirectiveDef` argument is `None`.  The directive files are shared
//! (`Arc<DirectiveFile>`), as the Java array holds references.
//!
//! The two value maps (`copyArgExtraValues`, `copyArgCommandLineValues`) and
//! `CopyArgEntrySet.pairMap` are `HashMap`s in Java whose values may be null; they are
//! `BTreeMap<String, Option<String>>` here, so their iteration order is key order
//! instead of Java's hash-bucket order.  `CopyArgEntrySet.init` walks
//! `copyArgExtraValues` while adding to `pairMap`, and the tilt angle directives
//! exclude each other there, so when the extra values hold more than one tilt angle
//! directive for an axis, which one survives follows key order.
// TODO(unit): needs etomo/storage/DirectiveFile.java - `getInstance(BaseManager, AxisID,
// File, DirectiveFileType)` (DirectiveFile::get_instance(&'static dyn BaseManager,
// Option<AxisID>, Option<&Path>, DirectiveFileType) -> Option<DirectiveFile>),
// `getAttribute(Match, DirectiveDef, AxisID, boolean)`
// (get_attribute_with_match(&self, Match, Option<DirectiveDef>, Option<AxisID>, bool) ->
// Option<AttributeMatch>), `getAttribute(DirectiveDef, AxisID, boolean, boolean)`
// (get_attribute(&self, Option<DirectiveDef>, Option<AxisID>, bool, bool) ->
// Option<AttributeMatch>), `iterator(boolean)` (iterator(&self, bool) ->
// Option<directive_file::StatementIterator<'_>>, an `Iterator`),
// `getCopyArgIterator()` (get_copy_arg_iterator(&self) ->
// Option<ReadOnlyAttributeIterator<'_>>), and Display for `toString`.
// TODO(unit): needs etomo/storage/DirectiveAttribute.java - `AttributeMatch`
// (storage::directive_attribute::AttributeMatch: DirectiveValue + 'static, with
// is_empty(), is_override(), is_value()), `Match` (variants Primary, Secondary) and
// static `toBoolean` (directive_attribute::to_boolean(Option<&str>) -> bool).
// TODO(unit): needs etomo/logic/UserEnv.java - `isGpuProcessing`,
// `isParallelProcessing` (logic::user_env::is_gpu_processing / is_parallel_processing
// (&'static dyn BaseManager, Option<AxisID>, Option<&str>) -> bool).
// TODO(unit): needs etomo/logic/DatasetTool.java - `validateTiltAngle`
// (logic::dataset_tool::validate_tilt_angle(&'static dyn BaseManager, AxisID,
// Option<&str>, Option<AxisID>, bool, Option<&str>, Option<&str>) -> bool).
// TODO(unit): needs etomo/ui/SetupReconInterface.java and
// etomo/storage/DirectiveFileInterface.java - the interfaces this class implements.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::exclude_views_param::ExcludeViewsParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::user_env;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::directive_attribute::{self, AttributeMatch, Match};
use crate::imod::etomo::storage::directive_def::{self, DirectiveDef};
use crate::imod::etomo::storage::directive_file::{self, DirectiveFile};
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::storage::directive_value::DirectiveValue;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_string_matches_whitespace, java_lang_string_trim,
};
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::extension::EXTENSION_DIVIDER;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_type::CollectionType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java `DirectiveFileCollection`.
pub struct DirectiveFileCollection {
    /// Java field `directiveFileArray`, `{ null, null, null, null, null }`.
    directive_file_array: [Option<Arc<DirectiveFile>>; 5],
    /// Java field `binningValidationSet`.
    binning_validation_set: Option<ValidationSet>,
    /// Java field `copyArgExtraValues`: scan header and user preference values - lowest
    /// priority.
    copy_arg_extra_values: Option<BTreeMap<String, Option<String>>>,
    /// Java field `copyArgCommandLineValues`: command line copyarg directive values -
    /// highest priority.
    copy_arg_command_line_values: Option<BTreeMap<String, Option<String>>>,
    /// Java field `overrideSkipA`.
    override_skip_a: bool,
    /// Java field `overrideSkipB`.
    override_skip_b: bool,
    /// Java field `manager`.
    manager: &'static dyn BaseManager,
    /// Java field `axisID`.
    axis_id: Option<AxisID>,
    /// Java field `debug`.
    debug: bool,
}

/// Java `toString`.
impl std::fmt::Display for DirectiveFileCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.directive_file_array.len() {
            if let Some(directive_file) = &self.directive_file_array[i] {
                write!(f, "{}:{}   ", i, directive_file)?;
            }
        }
        Ok(())
    }
}

impl DirectiveFileCollection {
    /// Java private `DirectiveFileCollection(BaseManager, AxisID, ValidationSet,
    /// boolean)`.
    fn new_private(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        binning_validation_set: Option<ValidationSet>,
        include_batch_defaults: bool,
    ) -> DirectiveFileCollection {
        let mut instance = DirectiveFileCollection {
            directive_file_array: [None, None, None, None, None],
            binning_validation_set,
            copy_arg_extra_values: None,
            copy_arg_command_line_values: None,
            override_skip_a: false,
            override_skip_b: false,
            manager,
            axis_id,
            debug: false,
        };
        if include_batch_defaults {
            let file = file_type::CLASS
                .default_batch_run_tomo_autodoc
                .get_file(Some(manager), axis_id);
            instance.directive_file_array[DirectiveFileType::BatchDefaults.get_index() as usize] =
                DirectiveFile::get_instance(
                    manager,
                    axis_id,
                    file.as_deref(),
                    DirectiveFileType::BatchDefaults,
                )
                .map(Arc::new);
        }
        // When a directive file is set, get the values of the command line options which
        // correspond to copyarg directives.
        let (directive, axis, raw_image_stack, fiducial, frame) = {
            let arguments = etomo_director::ARGUMENTS.lock().unwrap();
            (
                arguments.get_directive().map(Path::to_path_buf),
                arguments.get_axis(),
                arguments.get_raw_image_stack().map(str::to_string),
                arguments.get_fiducial(),
                arguments.get_frame(),
            )
        };
        if directive.is_some() {
            if axis == Some(AxisType::DualAxis) {
                instance.set_copy_arg_command_line_value(Some(DirectiveDef::DUAL), Some("1"));
            }
            let name = raw_image_stack;
            if let Some(name) = name {
                if !java_lang_string_matches_whitespace(&name) {
                    instance.set_copy_arg_command_line_value(Some(DirectiveDef::NAME), Some(&name));
                }
            }
            // `Arguments.getFiducial()` is a ConstEtomoNumber of type double in Java; a
            // set value prints as `Double.toString`.
            let gold = fiducial;
            if let Some(gold) = gold {
                instance.set_copy_arg_command_line_value(
                    Some(DirectiveDef::GOLD),
                    Some(&java_lang_double_to_string(gold)),
                );
            }
            if frame == Some(ViewType::Montage) {
                instance.set_copy_arg_command_line_value(Some(DirectiveDef::MONTAGE), Some("1"));
            }
        }
        instance
    }

    /// Java `DirectiveFileCollection(BaseManager, AxisID)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> DirectiveFileCollection {
        DirectiveFileCollection::new_private(manager, axis_id, None, false)
    }

    /// Java `DirectiveFileCollection(BaseManager, AxisID, ValidationSet)`.
    pub fn new_with_validation_set(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        binning_validation_set: Option<ValidationSet>,
    ) -> DirectiveFileCollection {
        DirectiveFileCollection::new_private(manager, axis_id, binning_validation_set, false)
    }

    /// Java static `getBatchInstance(BaseManager, AxisID)`.
    pub fn get_batch_instance(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
    ) -> DirectiveFileCollection {
        DirectiveFileCollection::new_private(manager, axis_id, None, true)
    }

    /// Java static `getBatchInstance(BaseManager, AxisID, ValidationSet)`.
    pub fn get_batch_instance_with_validation_set(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        binning_validation_set: Option<ValidationSet>,
    ) -> DirectiveFileCollection {
        DirectiveFileCollection::new_private(manager, axis_id, binning_validation_set, true)
    }

    /// Java `iterator(boolean)` (DirectiveFileInterface).
    pub fn iterator(&self, template_only: bool) -> StatementIterator<'_> {
        StatementIterator::new(self, template_only)
    }

    /// Java `getParameters(ExcludeViewsParam, AxisID, boolean, boolean)`
    /// (SetupReconInterface).
    pub fn get_parameters(
        &self,
        param: &mut ExcludeViewsParam,
        axis_id: Option<AxisID>,
        _dual_axis: bool,
        _do_validation: bool,
    ) -> bool {
        let name = self.get_value_axis(Some(DirectiveDef::NAME), axis_id);
        if name.is_some() {
            let stack_file_name = file_type::CLASS
                .raw_stack
                .get_file_name(Some(self.manager), axis_id);
            param.set_stack_name(stack_file_name.as_deref());
        }
        param.set_montaged_images(self.is_value_axis(Some(DirectiveDef::MONTAGE), axis_id));
        param.set_delete_old_files(self.is_value(Some(DirectiveDef::DELETE_OLD_FILES)));
        true
    }

    /// Java `getDatasetName(AxisID)`.
    pub fn get_dataset_name(&self, axis_id: Option<AxisID>) -> Option<String> {
        self.get_value_axis(Some(DirectiveDef::NAME), axis_id)
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }

    /// Java `getSkip(AxisID, boolean)`.  Bug# 2322 done.
    pub fn get_skip(&self, axis_id: Option<AxisID>, _do_validation: bool) -> Option<String> {
        self.get_value_axis(Some(DirectiveDef::SKIP), axis_id)
    }

    /// Java private `getAttribute(DirectiveFileArrayIterator, DirectiveDef, AxisID,
    /// boolean)`.  Get the attribute matching directiveDef.  If the primary match is not
    /// found, use the secondary match.  The secondary match less closely matches the
    /// axisID.
    fn get_attribute_from_iterator(
        &self,
        iterator: &mut DirectiveFileArrayIterator,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        ignore_file_type: bool,
    ) -> Option<AttributeMatch> {
        let directive_def =
            directive_def.map(|directive_def| directive_def.get_axis_id_instance(axis_id));
        iterator.restart();
        let mut secondary_match: Option<AttributeMatch> = None;
        // Search for a primary match starting with the highest priority file
        while iterator.has_next(self) {
            let directive_file = match iterator.next(self) {
                None => continue,
                Some(directive_file) => directive_file,
            };
            let primary_match = directive_file.get_attribute_with_match(
                Match::Primary,
                directive_def,
                axis_id,
                ignore_file_type,
            );
            if let Some(primary_match) = primary_match {
                if !primary_match.is_empty() {
                    return Some(primary_match);
                }
            }
            // Get the secondary match in case no primary match is found. Lower priority
            // files with a primary match outweigh higher priority files with a secondary
            // match.
            if secondary_match
                .as_ref()
                .is_none_or(|secondary_match| secondary_match.is_empty())
            {
                secondary_match = directive_file.get_attribute_with_match(
                    Match::Secondary,
                    directive_def,
                    axis_id,
                    ignore_file_type,
                );
            }
        }
        // A primary match was not found. Use the secondary match if it was found.
        if let Some(secondary_match) = secondary_match {
            if !secondary_match.is_empty() {
                return Some(secondary_match);
            }
        }
        None
    }

    /// Java package-private `getAttribute(DirectiveDef, AxisID, boolean, boolean)`.
    pub(crate) fn get_attribute(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> Option<AttributeMatch> {
        self.get_attribute_include_override(
            directive_def,
            axis_id,
            template_only,
            ignore_file_type,
            false,
        )
    }

    /// Java package-private `getAttribute(DirectiveDef, AxisID, boolean, boolean,
    /// boolean)`.  Returns a non-empty attributeMatch or null.  Directive files are
    /// checked in order of priority (batch to scope) and the highest priority file
    /// containing the directive, is used.  An overridden directive (no value and
    /// non-boolean) causes a null to be returned.  Two different matches are used -
    /// first the primary match, and then the secondary match.  The difference between
    /// them is how closely they match the axisID.  The primary match has priority over
    /// the secondary match, even when the primary match is found in a lower priority
    /// file.  If the match is in override, return null.
    pub(crate) fn get_attribute_include_override(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> Option<AttributeMatch> {
        let directive_def = directive_def?.get_axis_id_instance(axis_id);
        let mut iterator = DirectiveFileArrayIterator::new(
            self,
            Some(directive_def),
            template_only,
            ignore_file_type,
        );
        let attribute_match = self.get_attribute_from_iterator(
            &mut iterator,
            Some(directive_def),
            axis_id,
            ignore_file_type,
        )?;
        if attribute_match.is_override() && !include_override {
            return None;
        }
        if !attribute_match.is_empty() {
            return Some(attribute_match);
        }
        None
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
    /// if a non-empty, non-overriding directive, or an extraValue is found.
    /// `templateOnly`: if true, batch directive file, copyArgExtraValues, and
    /// copyArgCommandLineValues are ignored.
    pub fn contains_axis_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> bool {
        let directive_def = match directive_def {
            None => return false,
            Some(directive_def) => directive_def.get_axis_id_instance(axis_id),
        };
        // Before looking in the directive files look in copyArgCommandLineValues.
        if directive_def.get_directive_type() == DirectiveType::COPY_ARG {
            if let Some(values) = &self.copy_arg_command_line_values {
                if values.contains_key(&directive_def.get_name_for_axis(axis_id)) {
                    return true;
                }
            }
        }
        let attribute_match = self.get_attribute_include_override(
            Some(directive_def),
            axis_id,
            template_only,
            ignore_file_type,
            include_override,
        );
        if attribute_match.is_some() {
            return true;
        }
        if template_only {
            return false;
        }
        // An attribute has not been found - look for an extra value - these are
        // associated with the directive file.
        if directive_def.get_directive_type() == DirectiveType::COPY_ARG {
            if let Some(values) = &self.copy_arg_extra_values {
                return values.contains_key(&directive_def.get_name_for_axis(axis_id));
            }
        }
        false
    }

    /// Java `containsDirective(DirectiveDef, boolean)`.  Returns true if the directive
    /// is found.  True whether or not directive overrides.
    pub fn contains_directive(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
    ) -> bool {
        let directive_def = match directive_def {
            None => return false,
            Some(directive_def) => directive_def.get_axis_id_instance(self.axis_id),
        };
        let mut iterator =
            DirectiveFileArrayIterator::new(self, Some(directive_def), template_only, false);
        let attribute_match = self.get_attribute_from_iterator(
            &mut iterator,
            Some(directive_def),
            self.axis_id,
            false,
        );
        attribute_match.is_some()
    }

    /// Java `contains(DirectiveDef, AxisID)`.
    pub fn contains_axis(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.contains_axis_template(directive_def, axis_id, false)
    }

    /// Java `contains(DirectiveDef)` (DirectiveFileInterface).  Returns true a
    /// non-empty, non-overriding directive, or an extraValue is found.
    pub fn contains(&self, directive_def: Option<DirectiveDef>) -> bool {
        self.contains_axis_template(directive_def, None, false)
    }

    /// Java `contains(DirectiveDef, boolean)` (DirectiveFileInterface).
    pub fn contains_template(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
    ) -> bool {
        self.contains_axis_template(directive_def, None, template_only)
    }

    /// Java `contains(DirectiveDef, boolean, boolean)` (DirectiveFileInterface).
    pub fn contains_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> bool {
        self.contains_axis_template_ignore(directive_def, None, template_only, ignore_file_type)
    }

    /// Java `contains(DirectiveDef, boolean, boolean, boolean)`
    /// (DirectiveFileInterface).
    pub fn contains_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> bool {
        self.contains_axis_template_ignore_override(
            directive_def,
            None,
            template_only,
            ignore_file_type,
            include_override,
        )
    }

    /// Java `getValue(DirectiveDef, AxisID, boolean, boolean)` (DirectiveFileInterface).
    pub fn get_value_axis_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> Option<Box<dyn DirectiveValue>> {
        self.get_value_axis_template_ignore_override(
            directive_def,
            axis_id,
            template_only,
            ignore_file_type,
            false,
        )
    }

    /// Java `getValue(DirectiveDef, AxisID, boolean, boolean, boolean)`.  Returns the
    /// directive value if a non-empty, non-overriding directive, or an extraValue is
    /// found.
    ///
    /// Fixed in translation: DirectiveFileCollection.java:514 calls
    /// `directiveDef.getDirectiveType()` after only guarding the `getAxisIDInstance`
    /// call, a NullPointerException for a null directiveDef.  A null directiveDef has no
    /// value.
    pub fn get_value_axis_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> Option<Box<dyn DirectiveValue>> {
        let directive_def = directive_def?.get_axis_id_instance(axis_id);
        // Look for a command line value before checking the directive files.
        if directive_def.get_directive_type() == DirectiveType::COPY_ARG {
            if let Some(values) = &self.copy_arg_command_line_values {
                let name = directive_def.get_name_for_axis(axis_id);
                if values.contains_key(&name) {
                    return Some(Box::new(CopyArgValue::new(
                        values.get(&name).cloned().flatten(),
                    )));
                }
            }
        }
        let attribute_match = self.get_attribute_include_override(
            Some(directive_def),
            axis_id,
            template_only,
            ignore_file_type,
            include_override,
        );
        if let Some(attribute_match) = attribute_match {
            return Some(Box::new(attribute_match));
        }
        if template_only {
            return None;
        }
        // An attribute has not been found - look for an extra value.
        if directive_def.get_directive_type() == DirectiveType::COPY_ARG {
            if let Some(values) = &self.copy_arg_extra_values {
                return Some(Box::new(CopyArgValue::new(
                    values
                        .get(&directive_def.get_name_for_axis(axis_id))
                        .cloned()
                        .flatten(),
                )));
            }
        }
        None
    }

    /// Java `getValue(DirectiveDef, AxisID)`.  TODO Bug# 2322 - need to get the paired
    /// directive that matches axisID?
    pub fn get_value_axis(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        let directive_value =
            self.get_value_axis_template_ignore(directive_def, axis_id, false, false)?;
        directive_value.get_value()
    }

    /// Java `getValue(DirectiveDef)` (DirectiveFileInterface).  Returns the directive
    /// value if a non-empty, non-overriding directive, or an extraValue is found.
    pub fn get_value(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
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
    ) -> Option<Box<dyn DirectiveValue>> {
        self.get_value_axis_template_ignore(directive_def, None, template_only, ignore_file_type)
    }

    /// Java `getValue(DirectiveDef, boolean, boolean, boolean)`
    /// (DirectiveFileInterface).  TODO 2342.
    pub fn get_value_template_ignore_override(
        &self,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
        include_override: bool,
    ) -> Option<Box<dyn DirectiveValue>> {
        self.get_value_axis_template_ignore_override(
            directive_def,
            None,
            template_only,
            ignore_file_type,
            include_override,
        )
    }

    /// Java `getValue(DirectiveDef, int)` (DirectiveFileInterface).  Return the
    /// specified element of the directive value if a non-empty, non-overriding
    /// directive, or an extraValue is found.
    pub fn get_value_index(
        &self,
        directive_def: Option<DirectiveDef>,
        index: i32,
    ) -> Option<String> {
        let directive_def =
            directive_def.map(|directive_def| directive_def.get_axis_id_instance(self.axis_id));
        if index < 0 {
            return None;
        }
        let value = self.get_value(directive_def);
        // Get the element specified by index
        if let Some(value) = value {
            let divider = ",";
            if value.contains(divider) {
                let array = java_lang_string_split(&value, &Regex::new(divider).unwrap());
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

    /// Java `isValue(DirectiveDef, AxisID, boolean)` (DirectiveFileInterface).
    pub fn is_value_axis_template(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
    ) -> bool {
        self.is_value_axis_template_ignore(directive_def, axis_id, template_only, false)
    }

    /// Java `isValue(DirectiveDef, AxisID, boolean, boolean)`.  Returns the boolean
    /// version of a directive value if a non-empty, non-overriding directive, or an
    /// extraValue is found.  `templateOnly` causes function to ignore the batch
    /// directive file and copyrgExtraValues.
    ///
    /// Two source details are kept: `ignoreFileType` is not passed on (the attribute is
    /// looked up with `false`), and a null directiveDef, a NullPointerException at
    /// DirectiveFileCollection.java:660 in the source, is fixed in translation to
    /// return false.
    pub fn is_value_axis_template_ignore(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        template_only: bool,
        _ignore_file_type: bool,
    ) -> bool {
        let directive_def = match directive_def {
            None => return false,
            Some(directive_def) => directive_def.get_axis_id_instance(axis_id),
        };
        // Check the command line before checking the file
        if directive_def.get_directive_type() == DirectiveType::COPY_ARG {
            if let Some(values) = &self.copy_arg_command_line_values {
                let name = directive_def.get_name_for_axis(axis_id);
                if values.contains_key(&name) {
                    return directive_attribute::to_boolean(
                        values.get(&name).cloned().flatten().as_deref(),
                    );
                }
            }
        }
        let attribute_match =
            self.get_attribute(Some(directive_def), axis_id, template_only, false);
        if let Some(attribute_match) = attribute_match {
            return attribute_match.is_value();
        }
        if template_only {
            return false;
        }
        // An attribute has not been found - look for scan header values.
        if directive_def.get_directive_type() == DirectiveType::COPY_ARG {
            if let Some(values) = &self.copy_arg_extra_values {
                return directive_attribute::to_boolean(
                    values
                        .get(&directive_def.get_name_for_axis(axis_id))
                        .cloned()
                        .flatten()
                        .as_deref(),
                );
            }
        }
        false
    }

    /// Java `isValue(DirectiveDef, AxisID)`.
    pub fn is_value_axis(
        &self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
    ) -> bool {
        self.is_value_axis_template(directive_def, axis_id, false)
    }

    /// Java `isValue(DirectiveDef)` (DirectiveFileInterface).  Returns the boolean
    /// version of a directive value if a non-empty, non-overriding directive, or an
    /// extraValue is found.
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

    /// Java `getTiltAngleFields(AxisID, TiltAngleSpec, boolean)` (SetupReconInterface).
    /// `doValidation` has no effect.  Returns true.
    ///
    /// Fixed in translation: DirectiveFileCollection.java:735-739 passes the directive's
    /// elements to `TiltAngleSpec.setRangeMin(String)` / `setRangeStep(String)`, whose
    /// `Double.parseDouble` throws an uncaught NumberFormatException on a malformed
    /// value.  A malformed element leaves that field unchanged here.
    pub fn get_tilt_angle_fields(
        &self,
        axis_id: Option<AxisID>,
        tilt_angle_spec: Option<&mut TiltAngleSpec>,
        _do_validation: bool,
    ) -> bool {
        let tilt_angle_spec = match tilt_angle_spec {
            None => return true,
            Some(tilt_angle_spec) => tilt_angle_spec,
        };
        let first_inc_directive_def = DirectiveDef::FIRST_INC.get_axis_id_instance(axis_id);
        if self.contains_axis(Some(first_inc_directive_def), axis_id) {
            tilt_angle_spec.set_type(TiltAngleType::Range);
            let value = self.get_value_axis(Some(first_inc_directive_def), axis_id);
            let mut array_value: Option<Vec<String>> = None;
            if let Some(value) = value {
                array_value = Some(java_lang_string_split(
                    java_lang_string_trim(&value),
                    &Regex::new(CollectionType::Array.get_splitter()).unwrap(),
                ));
            }
            if let Some(array_value) = &array_value {
                if !array_value.is_empty() {
                    let _ = tilt_angle_spec.set_range_min_string(&array_value[0]);
                }
                if array_value.len() > 1 {
                    let _ = tilt_angle_spec.set_range_step_string(&array_value[1]);
                }
            }
        } else if self.is_value_axis(
            Some(DirectiveDef::EXTRACT.get_axis_id_instance(axis_id)),
            axis_id,
        ) {
            tilt_angle_spec.set_type(TiltAngleType::Extract);
        } else if self.is_value_axis(
            Some(DirectiveDef::USE_RAW_TLT.get_axis_id_instance(axis_id)),
            axis_id,
        ) {
            tilt_angle_spec.set_type(TiltAngleType::File);
        } else {
            // Must set something here, so use the settings values
            let tilt_angles_rawtlt_file = etomo_director::INSTANCE.with_user_configuration(|c| c.is_tilt_angles_rawtlt_file());
            if tilt_angles_rawtlt_file {
                tilt_angle_spec.set_type(TiltAngleType::File);
            } else {
                // Default
                tilt_angle_spec.set_type(TiltAngleType::Extract);
            }
        }
        true
    }

    /// Java `msgExcludeViewsSucceeded(AxisID, boolean, boolean)`
    /// (SetupReconInterface).
    pub fn msg_exclude_views_succeeded(
        &mut self,
        axis_id: Option<AxisID>,
        _process_running: bool,
        _process_done: bool,
    ) {
        if axis_id == Some(AxisID::Second) {
            self.override_skip_b = true;
        } else if axis_id.is_none() {
            self.override_skip_a = true;
            self.override_skip_b = true;
        } else {
            self.override_skip_a = true;
        }
    }

    /// Java private `setCopyArgCommandLineValue(DirectiveDef, String)`.  Puts the
    /// DirectiveDef/value pair into commandLineValues.
    fn set_copy_arg_command_line_value(
        &mut self,
        directive_def: Option<DirectiveDef>,
        value: Option<&str>,
    ) {
        let directive_def = match directive_def {
            None => return,
            Some(directive_def) => directive_def.get_axis_id_instance(self.axis_id),
        };
        let values = self
            .copy_arg_command_line_values
            .get_or_insert_with(BTreeMap::new);
        values.insert(
            directive_def.get_name_for_axis(None),
            value.map(|value| value.to_string()),
        );
    }

    /// Java private `setCopyArgExtraValue(DirectiveDef, AxisID, String)`.  Puts the
    /// DirectiveDef/value pair into extraValues.
    fn set_copy_arg_extra_value(
        &mut self,
        directive_def: Option<DirectiveDef>,
        axis_id: Option<AxisID>,
        value: Option<&str>,
    ) {
        let directive_def = match directive_def {
            None => return,
            Some(directive_def) => directive_def.get_axis_id_instance(axis_id),
        };
        let values = self.copy_arg_extra_values.get_or_insert_with(BTreeMap::new);
        values.insert(
            directive_def.get_name_for_axis(axis_id),
            value.map(|value| value.to_string()),
        );
    }

    /// Java `setBinning(String)` (SetupReconInterface).
    pub fn set_binning(&mut self, input: Option<&str>) {
        self.set_copy_arg_extra_value(Some(DirectiveDef::BINNING), None, input);
    }

    /// Java `setImageRotation(String)` (SetupReconInterface).
    pub fn set_image_rotation(&mut self, input: Option<&str>) {
        self.set_copy_arg_extra_value(Some(DirectiveDef::ROTATION), Some(AxisID::First), input);
        // not valid for B if there is a brotation directive in this collection.
        // if (isValue(DirectiveDef.DUAL)) {
        // setCopyArgExtraValue(DirectiveDef.ROTATION, AxisID.SECOND, input);
        // }
    }

    /// Java `setPixelSize(double)` (SetupReconInterface).
    pub fn set_pixel_size(&mut self, input: f64) {
        self.set_copy_arg_extra_value(
            Some(DirectiveDef::PIXEL),
            None,
            Some(&java_lang_double_to_string(input)),
        );
    }

    /// Java `setHalfFloatModeOutput(Integer)` (SetupReconInterface).
    /// `String.valueOf(Integer)` of a null Integer is `"null"`.
    pub fn set_half_float_mode_output(&mut self, input: Option<i32>) {
        let value = match input {
            None => "null".to_string(),
            Some(input) => input.to_string(),
        };
        self.set_copy_arg_extra_value(Some(DirectiveDef::HALF_FLOAT), None, Some(&value));
    }

    /// Java `setTwodir(AxisID, double)` (SetupReconInterface).
    pub fn set_twodir(&mut self, axis_id: Option<AxisID>, input: f64) {
        self.set_copy_arg_extra_value(
            Some(DirectiveDef::TWODIR),
            axis_id,
            Some(&java_lang_double_to_string(input)),
        );
    }

    /// Java `initTiltAngleFields(AxisID, TiltAngleSpec, UserConfiguration)`
    /// (SetupReconInterface).  Add tilt angle directives to copyArgExtraValues.  Behaves
    /// as an init function by only adding directives that are not in the collection.
    /// Can be called at any time without overriding other settings.
    pub fn init_tilt_angle_fields(
        &mut self,
        axis_id: Option<AxisID>,
        tilt_angle_spec: &TiltAngleSpec,
        user_configuration: &UserConfiguration,
    ) {
        if self.copy_arg_extra_values.is_none() {
            self.copy_arg_extra_values = Some(BTreeMap::new());
        }
        if tilt_angle_spec.get_type() == TiltAngleType::File
            || user_configuration.is_tilt_angles_rawtlt_file()
        {
            let key = DirectiveDef::USE_RAW_TLT
                .get_axis_id_instance(axis_id)
                .get_name_for_axis(axis_id);
            if !self
                .copy_arg_extra_values
                .as_ref()
                .unwrap()
                .contains_key(&key)
            {
                self.set_copy_arg_extra_value(
                    Some(DirectiveDef::USE_RAW_TLT),
                    axis_id,
                    Some(directive_def::TRUE_VALUE),
                );
            }
        } else if tilt_angle_spec.get_type() == TiltAngleType::Extract {
            let key = DirectiveDef::EXTRACT
                .get_axis_id_instance(axis_id)
                .get_name_for_axis(axis_id);
            if !self
                .copy_arg_extra_values
                .as_ref()
                .unwrap()
                .contains_key(&key)
            {
                self.set_copy_arg_extra_value(
                    Some(DirectiveDef::EXTRACT),
                    axis_id,
                    Some(directive_def::TRUE_VALUE),
                );
            }
        } else if tilt_angle_spec.get_type() == TiltAngleType::Range {
            let first_inc_directive_def = DirectiveDef::FIRST_INC.get_axis_id_instance(axis_id);
            if !self
                .copy_arg_extra_values
                .as_ref()
                .unwrap()
                .contains_key(&first_inc_directive_def.get_name_for_axis(axis_id))
            {
                self.set_copy_arg_extra_value(
                    Some(first_inc_directive_def),
                    axis_id,
                    Some(&format!(
                        "{}, {}",
                        java_lang_double_to_string(tilt_angle_spec.get_range_min()),
                        java_lang_double_to_string(tilt_angle_spec.get_range_step())
                    )),
                );
            }
        }
    }

    /// Java `containsTiltAngleSpec(AxisID)`.
    pub fn contains_tilt_angle_spec(&self, axis_id: Option<AxisID>) -> bool {
        self.contains_axis(
            Some(DirectiveDef::FIRST_INC.get_axis_id_instance(axis_id)),
            axis_id,
        ) || self.contains_axis(
            Some(DirectiveDef::USE_RAW_TLT.get_axis_id_instance(axis_id)),
            axis_id,
        ) || self.contains_axis(
            Some(DirectiveDef::EXTRACT.get_axis_id_instance(axis_id)),
            axis_id,
        )
    }

    /// Java `getBackupDirectory()` (SetupReconInterface).
    pub fn get_backup_directory(&self) -> Option<String> {
        None
    }

    /// Java `getBinning()`.
    pub fn get_binning(&self) -> Option<String> {
        match self.get_binning_validate(false) {
            Ok(value) => value,
            Err(_) => None,
        }
    }

    /// Java `getBinning(boolean)` (SetupReconInterface).
    ///
    /// Fixed in translation: DirectiveFileCollection.java:917 calls
    /// `binningValidationSet.validate(value)` on a collection constructed without a
    /// validation set, a NullPointerException.  With no validation set there is nothing
    /// to validate.
    pub fn get_binning_validate(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let value = self.get_value(Some(DirectiveDef::BINNING));
        if do_validation {
            if let Some(binning_validation_set) = &self.binning_validation_set {
                let errmsg = binning_validation_set.validate(value.as_deref());
                if let Some(errmsg) = errmsg {
                    let errmsg = format!("Error: binning directive value is invalid:  {}", errmsg);
                    eprintln!("{}", errmsg);
                    return Err(FieldValidationFailedException::new(Some(&errmsg)));
                }
            }
        }
        Ok(value)
    }

    /// Java `getDataset()` (SetupReconInterface), deprecated 4/8/2019.  No need to get
    /// the dataset name.  It can be derived from the file name.
    pub fn get_dataset(&self) -> Option<String> {
        self.get_raw_image_stack()
    }

    /// Java `getRawImageStack()` (SetupReconInterface).  Returns the raw image stack
    /// file name recreated from the directives NAME and STACK_EXT.
    pub fn get_raw_image_stack(&self) -> Option<String> {
        let mut axis_string = String::new();
        if self.is_dual_axis_selected() {
            axis_string = AxisID::First.get_extension();
        }
        let mut name = self.get_value(Some(DirectiveDef::NAME));
        let mut stack_ext = self.get_value(Some(DirectiveDef::STACK_EXT));
        if name
            .as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
            && stack_ext
                .as_deref()
                .is_none_or(java_lang_string_matches_whitespace)
        {
            return None;
        }
        if name.is_none() {
            name = Some(String::new());
        }
        if stack_ext
            .as_deref()
            .is_some_and(java_lang_string_matches_whitespace)
        {
            stack_ext = None;
        }
        Some(format!(
            "{}{}{}{}",
            name.unwrap(),
            axis_string,
            EXTENSION_DIVIDER,
            match stack_ext {
                Some(stack_ext) => stack_ext,
                None => ImageFilenameStyle::DEFAULT
                    .get_default_raw_image_stack_extension()
                    .to_string(),
            }
        ))
    }

    /// Java `getDirectiveFile(DirectiveFileType)`.
    pub fn get_directive_file(&self, r#type: DirectiveFileType) -> Option<Arc<DirectiveFile>> {
        self.directive_file_array[r#type.get_index() as usize].clone()
    }

    /// Java `getDirectiveFileCollection()` (SetupReconInterface).
    pub fn get_directive_file_collection(&self) -> &DirectiveFileCollection {
        self
    }

    /// Java `getDistortionFile()` (SetupReconInterface).
    pub fn get_distortion_file(&self) -> Option<String> {
        self.get_value(Some(DirectiveDef::DISTORT))
    }

    /// Java `getCurrentStackExt()`.
    pub fn get_current_stack_ext(&self) -> Option<String> {
        let current_stack_ext = self.get_value(Some(DirectiveDef::CURRENT_STACK_EXT));
        if current_stack_ext.is_some() {
            return current_stack_ext;
        }
        // Use currentBStackExt if currentStackExt wasn't set.
        self.get_value(Some(DirectiveDef::CURRENT_B_STACK_EXT))
    }

    /// Java `getExcludeList(AxisID, boolean)` (SetupReconInterface).  `doValidation` has
    /// no effect.
    pub fn get_exclude_list(
        &self,
        axis_id: Option<AxisID>,
        _do_validation: bool,
    ) -> Option<String> {
        self.get_value_axis(Some(DirectiveDef::SKIP), axis_id)
    }

    /// Java `getTwodir(AxisID, boolean)` (SetupReconInterface).  `doValidation` has no
    /// effect.
    pub fn get_twodir(&self, axis_id: Option<AxisID>, _do_validation: bool) -> Option<String> {
        self.get_value_axis(Some(DirectiveDef::TWODIR), axis_id)
    }

    /// Java `isTwodir(AxisID)` (SetupReconInterface).
    pub fn is_twodir(&self, axis_id: Option<AxisID>) -> bool {
        // Contains returns true if a directive contains this attribute, and it is set to
        // a value. Since twodir is not a boolean, no value would mean that it is
        // overridden - which causes contains() to return false.
        self.contains_axis(Some(DirectiveDef::TWODIR), axis_id)
    }

    /// Java `getFiducialDiameter(boolean)` (SetupReconInterface).  `doValidation` has no
    /// effect.
    pub fn get_fiducial_diameter(&self, _do_validation: bool) -> Option<String> {
        self.get_value(Some(DirectiveDef::GOLD))
    }

    /// Java `getImageRotation(AxisID, boolean)` (SetupReconInterface).  Return rotation
    /// or brotation.  For the B axis use rotation if brotation is not set.  Rotation is
    /// required.  `doValidation` has no effect.
    pub fn get_image_rotation(
        &self,
        axis_id: Option<AxisID>,
        _do_validation: bool,
    ) -> Option<String> {
        let value = self.get_value_axis(Some(DirectiveDef::ROTATION), axis_id);
        // This function should never return null if any rotation exists. Use A rotation
        // when B rotation is missing.
        if axis_id == Some(AxisID::Second) && value.is_none() {
            return self.get_value_axis(Some(DirectiveDef::ROTATION), Some(AxisID::First));
        }
        value
    }

    /// Java `getMagGradientFile()` (SetupReconInterface).
    pub fn get_mag_gradient_file(&self) -> Option<String> {
        self.get_value(Some(DirectiveDef::GRADIENT))
    }

    /// Java `getPixelSize(boolean)` (SetupReconInterface).
    pub fn get_pixel_size(&self, _do_validation: bool) -> Option<String> {
        self.get_value(Some(DirectiveDef::PIXEL))
    }

    /// Java `getHalfFloatModeOutput()` (SetupReconInterface).
    pub fn get_half_float_mode_output(&self) -> Option<i32> {
        converter::to_integer(self.get_value(Some(DirectiveDef::HALF_FLOAT)).as_deref())
    }

    /// Java `getStackExt()`.
    pub fn get_stack_ext(&self) -> Option<String> {
        self.get_value(Some(DirectiveDef::STACK_EXT))
    }

    /// Java `isAdjustedFocusSelected(AxisID)` (SetupReconInterface).
    pub fn is_adjusted_focus_selected(&self, axis_id: Option<AxisID>) -> bool {
        self.is_value_axis(Some(DirectiveDef::FOCUS), axis_id)
    }

    /// Java `isDualAxisSelected()` (SetupReconInterface).
    pub fn is_dual_axis_selected(&self) -> bool {
        self.is_value(Some(DirectiveDef::DUAL))
    }

    /// Java `isGpuProcessingSelected(String)` (SetupReconInterface).
    pub fn is_gpu_processing_selected(&self, property_user_dir: Option<&str>) -> bool {
        user_env::is_gpu_processing(self.manager, self.axis_id, property_user_dir)
    }

    /// Java `isParallelProcessSelected(String)` (SetupReconInterface).
    pub fn is_parallel_process_selected(&self, property_user_dir: Option<&str>) -> bool {
        user_env::is_parallel_processing(self.manager, self.axis_id, property_user_dir)
    }

    /// Java `isSingleAxisSelected()` (SetupReconInterface).
    pub fn is_single_axis_selected(&self) -> bool {
        !self.is_value(Some(DirectiveDef::DUAL))
    }

    /// Java `isSingleViewSelected()` (SetupReconInterface).
    pub fn is_single_view_selected(&self) -> bool {
        !self.is_value(Some(DirectiveDef::MONTAGE))
    }

    /// Java `setup(DirectiveFile)`.  Sets the batch directive file, and then sets the
    /// template files from the the batch directive file.
    pub fn setup(&mut self, batch_directive_file: Option<Arc<DirectiveFile>>) {
        self.directive_file_array[DirectiveFileType::Batch.get_index() as usize] =
            batch_directive_file.clone();
        if let Some(batch_directive_file) = batch_directive_file {
            self.set_directive_file_from_attribute(
                batch_directive_file.get_attribute(
                    Some(DirectiveDef::SCOPE_TEMPLATE),
                    None,
                    false,
                    false,
                ),
                DirectiveFileType::Scope,
            );
            self.set_directive_file_from_attribute(
                batch_directive_file.get_attribute(
                    Some(DirectiveDef::SYSTEM_TEMPLATE),
                    None,
                    false,
                    false,
                ),
                DirectiveFileType::System,
            );
            self.set_directive_file_from_attribute(
                batch_directive_file.get_attribute(
                    Some(DirectiveDef::USER_TEMPLATE),
                    None,
                    false,
                    false,
                ),
                DirectiveFileType::User,
            );
        }
    }

    /// Java private `setDirectiveFile(AttributeMatch, DirectiveFileType)`.  Sets a
    /// directive file.  No directive file other then the one being set is changed by
    /// this function.
    fn set_directive_file_from_attribute(
        &mut self,
        attribute: Option<AttributeMatch>,
        r#type: DirectiveFileType,
    ) {
        let index = r#type.get_index() as usize;
        match attribute {
            None => {
                self.directive_file_array[index] = None;
            }
            Some(attribute) => {
                let abs_path = attribute.get_value();
                match abs_path {
                    None => {
                        self.directive_file_array[index] = None;
                    }
                    Some(abs_path) => {
                        self.directive_file_array[index] = DirectiveFile::get_instance(
                            self.manager,
                            self.axis_id,
                            Some(PathBuf::from(abs_path).as_path()),
                            r#type,
                        )
                        .map(Arc::new);
                    }
                }
            }
        }
    }

    /// Java `setDirectiveFile(File, DirectiveFileType)`.  Sets a directive file.  No
    /// directive file other then the one being set is changed by this function.
    pub fn set_directive_file(&mut self, file: Option<&Path>, r#type: DirectiveFileType) {
        let index = r#type.get_index() as usize;
        match file {
            None => {
                self.directive_file_array[index] = None;
            }
            Some(file) => {
                self.directive_file_array[index] =
                    DirectiveFile::get_instance(self.manager, self.axis_id, Some(file), r#type)
                        .map(Arc::new);
            }
        }
    }

    /// Java `validateTiltAngle(AxisID, String)` (SetupReconInterface).
    pub fn validate_tilt_angle(&self, axis_id: Option<AxisID>, error_title: Option<&str>) -> bool {
        let mut tilt_angle_spec = TiltAngleSpec::new();
        self.get_tilt_angle_fields(axis_id, Some(&mut tilt_angle_spec), false);
        dataset_tool::validate_tilt_angle(
            self.manager,
            AxisID::Only,
            error_title,
            axis_id,
            tilt_angle_spec.get_type() == TiltAngleType::Range,
            Some(&java_lang_double_to_string(tilt_angle_spec.get_range_min())),
            Some(&java_lang_double_to_string(
                tilt_angle_spec.get_range_step(),
            )),
        )
    }

    /// Java `getCopyArgCommandLineSet()`.
    pub fn get_copy_arg_command_line_set(&self) -> Option<Vec<(String, Option<String>)>> {
        let values = self.copy_arg_command_line_values.as_ref()?;
        Some(
            values
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        )
    }

    /// Java `getCopyArgEntrySet()`.  This function will not return null.
    pub fn get_copy_arg_entry_set(&self) -> CopyArgEntrySet<'_> {
        CopyArgEntrySet::get_instance(self)
    }

    /// Java `getDoseSym(AxisID, boolean)` (SetupReconInterface).
    pub fn get_dose_sym(&self, axis_id: Option<AxisID>, _do_validation: bool) -> Option<String> {
        self.get_value_axis(Some(DirectiveDef::DOSESYM), axis_id)
    }

    /// Java `isDoseSym(AxisID)` (SetupReconInterface).
    pub fn is_dose_sym(&self, axis_id: Option<AxisID>) -> bool {
        self.contains_axis(Some(DirectiveDef::DOSESYM), axis_id)
    }

    /// Java `setDoseSym(AxisID, double)` (SetupReconInterface).
    pub fn set_dose_sym(&mut self, axis_id: Option<AxisID>, input: f64) {
        self.set_copy_arg_extra_value(
            Some(DirectiveDef::DOSESYM),
            axis_id,
            Some(&java_lang_double_to_string(input)),
        );
    }
}

/// Java private inner class `StatementIterator`.  Iterates through all the global
/// name/value pairs in all files.  Duplicates from different files are NOT excluded:
/// this class does not function the way the keyed retrieval does.  This iterator is
/// read-only: name/value pairs cannot be removed from a file using this class.
pub struct StatementIterator<'a> {
    /// The enclosing instance.
    collection: &'a DirectiveFileCollection,
    /// Java field `fileIterator`.
    file_iterator: DirectiveFileArrayIterator,
    /// Java field `templateOnly`.
    template_only: bool,
    /// Java field `iterator`.
    iterator: Option<std::iter::Peekable<directive_file::StatementIterator<'a>>>,
}

impl<'a> StatementIterator<'a> {
    /// Java private `StatementIterator(boolean)`.
    fn new(collection: &'a DirectiveFileCollection, template_only: bool) -> StatementIterator<'a> {
        StatementIterator {
            collection,
            file_iterator: DirectiveFileArrayIterator::new(collection, None, template_only, true),
            template_only,
            iterator: None,
        }
    }

    /// Java `hasNext`.
    pub fn has_next(&mut self) -> bool {
        // Find a file with something in it
        while self
            .iterator
            .as_mut()
            .is_none_or(|iterator| iterator.peek().is_none())
        {
            if !self.file_iterator.has_next(self.collection) {
                return false;
            } else {
                // `hasNext` guarantees a file.
                let directive_file = self.file_iterator.next(self.collection).unwrap();
                self.iterator = directive_file
                    .iterator(self.template_only)
                    .map(|iterator| iterator.peekable());
            }
        }
        true
    }

    /// Java `next`.
    pub fn next(&mut self) -> Option<<directive_file::StatementIterator<'a> as Iterator>::Item> {
        if self.has_next() {
            return self.iterator.as_mut().unwrap().next();
        }
        None
    }
}

/// Java private inner class `DirectiveFileArrayIterator`.  Iterates through the
/// directive files from high to low.  If hasNext is true, next will not return null.
/// The enclosing instance is passed to `has_next` and `next`.
struct DirectiveFileArrayIterator {
    /// Java field `templateOnly`.
    template_only: bool,
    /// Java field `ignoreFileType`.
    ignore_file_type: bool,
    /// Java field `start`.
    start: i32,
    /// Java field `end`.
    end: i32,
    /// Java field `next`.
    next: i32,
}

impl DirectiveFileArrayIterator {
    /// Java private `DirectiveFileArrayIterator(DirectiveDef, boolean, boolean)`.
    /// `directiveDef`: to set template and batch if ignoreFileType is false, may be
    /// null.  `templateOnly`: batch file is not retrieved.  `ignoreFileType`:
    /// directiveDef is ignored.
    fn new(
        collection: &DirectiveFileCollection,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_file_type: bool,
    ) -> DirectiveFileArrayIterator {
        let directive_def = directive_def
            .map(|directive_def| directive_def.get_axis_id_instance(collection.axis_id));
        //
        // Get directive permission (permission to be in a specific type of directive
        // file).  If ignorePermissions is on, the directive can be in either type of
        // file.
        let mut template = true;
        let mut batch = true;
        if !ignore_file_type {
            if let Some(directive_def) = directive_def {
                template = directive_def.is_template(collection.axis_id);
                batch = directive_def.is_batch(collection.axis_id);
            }
        }
        //
        // Set start and end
        let length = collection.directive_file_array.len() as i32;
        let start;
        let end;
        if template_only && batch && !template {
            // Retrieve nothing. Its asking for templates only but the directive
            // permission is batch only.
            start = -1;
            end = -1;
        } else if template_only || (!batch && template) {
            // Retrieve templates.
            start = length - 2;
            end = 0;
        } else {
            start = length - 1;
            if batch && !template {
                // Retrieve batch
                end = start;
            } else {
                // Retrieve everything
                end = 0;
            }
        }
        DirectiveFileArrayIterator {
            template_only,
            ignore_file_type,
            start,
            end,
            next: start,
        }
    }

    /// Java private `iterator(DirectiveDef, boolean, boolean)`.
    fn iterator(
        collection: &DirectiveFileCollection,
        directive_def: Option<DirectiveDef>,
        template_only: bool,
        ignore_permissions: bool,
    ) -> DirectiveFileArrayIterator {
        DirectiveFileArrayIterator::new(
            collection,
            directive_def,
            template_only,
            ignore_permissions,
        )
    }

    /// Java private `restart`.
    fn restart(&mut self) {
        self.next = self.start;
    }

    /// Java `hasNext`.  Returns true if the next member variable is the index of a
    /// non-null directiveFileArray element.  If it is not will attempt to make this
    /// happen by decrementing next.
    fn has_next(&mut self, collection: &DirectiveFileCollection) -> bool {
        let length = collection.directive_file_array.len() as i32;
        if self.start == -1 || self.end == -1 || self.next < 0 || self.next >= length {
            return false;
        }
        if collection.directive_file_array[self.next as usize].is_some() {
            return true;
        }
        let mut i = self.next;
        while i >= self.end {
            if collection.directive_file_array[i as usize].is_some() {
                self.next = i;
                return true;
            }
            i -= 1;
        }
        false
    }

    /// Java `next`.  If hasNext is true, returns what the next member variable is
    /// pointing to and decrements next.  Returns null if hasNExt is false.
    fn next<'a>(&mut self, collection: &'a DirectiveFileCollection) -> Option<&'a DirectiveFile> {
        if self.has_next(collection) {
            let index = self.next;
            self.next -= 1;
            return collection.directive_file_array[index as usize].as_deref();
        }
        None
    }
}

/// Java private static class `CopyArgValue implements DirectiveValue`.
struct CopyArgValue {
    /// Java field `value`.
    value: Option<String>,
}

impl CopyArgValue {
    /// Java private `CopyArgValue(String)`.
    fn new(value: Option<String>) -> CopyArgValue {
        CopyArgValue { value }
    }
}

impl DirectiveValue for CopyArgValue {
    fn get_value(&self) -> Option<String> {
        self.value.clone()
    }

    fn is_batch(&self) -> bool {
        true
    }

    fn is_override(&self) -> bool {
        self.value.is_none()
    }
}

/// Java public static nested class `CopyArgEntrySet`.
pub struct CopyArgEntrySet<'a> {
    /// Java field `pairMap`.
    pair_map: BTreeMap<String, Option<String>>,
    /// Java field `directiveFileCollection`.
    directive_file_collection: &'a DirectiveFileCollection,
}

impl<'a> CopyArgEntrySet<'a> {
    /// Java private `CopyArgEntrySet(DirectiveFileCollection)`.  Don't call constructor
    /// directly.
    fn new(directive_file_collection: &'a DirectiveFileCollection) -> CopyArgEntrySet<'a> {
        CopyArgEntrySet {
            pair_map: BTreeMap::new(),
            directive_file_collection,
        }
    }

    /// Java private `contains(String)`.  Returns true if pairMap contains name.  Tilt
    /// angle spec directives exclude each other.
    fn contains(&self, name: &str) -> bool {
        if self.pair_map.contains_key(name) {
            return true;
        }
        let mut axis_id = AxisID::First;
        if self.is_tilt_angle_spec(name, Some(axis_id)) {
            return self.contains_tilt_angle_spec(Some(axis_id));
        }
        axis_id = AxisID::Second;
        if self.is_tilt_angle_spec(name, Some(axis_id)) {
            return self.contains_tilt_angle_spec(Some(axis_id));
        }
        false
    }

    /// Java private `isTiltAngleSpec(String, AxisID)`.
    fn is_tilt_angle_spec(&self, name: &str, pair_axis_id: Option<AxisID>) -> bool {
        DirectiveDef::USE_RAW_TLT
            .get_axis_id_instance(pair_axis_id)
            .equals_name(Some(name), pair_axis_id)
            || DirectiveDef::EXTRACT
                .get_axis_id_instance(pair_axis_id)
                .equals_name(Some(name), pair_axis_id)
            || DirectiveDef::FIRST_INC
                .get_axis_id_instance(pair_axis_id)
                .equals_name(Some(name), pair_axis_id)
    }

    /// Java private `containsTiltAngleSpec(AxisID)`.  `axisID` is the axis of the tilt
    /// angle spec directives; returns true if pairMap contains any of them.
    fn contains_tilt_angle_spec(&self, axis_id: Option<AxisID>) -> bool {
        self.pair_map.contains_key(
            &DirectiveDef::USE_RAW_TLT
                .get_axis_id_instance(axis_id)
                .get_name_for_axis(axis_id),
        ) || self.pair_map.contains_key(
            &DirectiveDef::EXTRACT
                .get_axis_id_instance(axis_id)
                .get_name_for_axis(axis_id),
        ) || self.pair_map.contains_key(
            &DirectiveDef::FIRST_INC
                .get_axis_id_instance(axis_id)
                .get_name_for_axis(axis_id),
        )
    }

    /// Java private static `getInstance(DirectiveFileCollection)`.  This function should
    /// not return null.
    fn get_instance(directive_file_collection: &'a DirectiveFileCollection) -> CopyArgEntrySet<'a> {
        let mut instance = CopyArgEntrySet::new(directive_file_collection);
        instance.init(&directive_file_collection.directive_file_array);
        instance
    }

    /// Java private `init(DirectiveFile[])`.  Load all of the directive file copyarg
    /// values into pairMap.  Pairs with the same name as a previously saved pair
    /// overrides the previous pair.  A pair with a blank value is not saved and causes
    /// the pair with the same name in the map to be removed.  After loading all of
    /// copyarg values, load the scan header output from the directive file if scan
    /// header is in the map and is set to "1".  Only load pairs with names that are not
    /// already in the map, because the directive files all override scan header.
    fn init(&mut self, directive_file_array: &[Option<Arc<DirectiveFile>>]) {
        for i in 0..directive_file_array.len() {
            if let Some(directive_file) = &directive_file_array[i] {
                let mut iterator = match directive_file.get_copy_arg_iterator() {
                    None => continue,
                    Some(iterator) => iterator,
                };
                while iterator.has_next() {
                    let attribute = match iterator.next() {
                        None => break,
                        Some(attribute) => *attribute,
                    };
                    // SAFETY: the iterator hands out the address of an attribute owned by
                    // the directive file's autodoc, which lives while the file is held.
                    let (name, value) = unsafe {
                        (
                            ReadOnlyAttribute::get_name(&*attribute),
                            ReadOnlyAttribute::get_value(&*attribute),
                        )
                    };
                    self.pair_map.remove(&name);
                    // A blank value means remove the previously added pair, otherwise
                    // override the previous added pair with the new value.
                    if value.is_some() {
                        self.pair_map.insert(name, value);
                    }
                }
            }
        }
        // Get values from the .etomo file and scanning the header which aren't already in
        // the map.
        let directive_file_collection = self.directive_file_collection;
        if let Some(copy_arg_extra_values) = &directive_file_collection.copy_arg_extra_values {
            for (name, value) in copy_arg_extra_values.iter() {
                if !self.contains(name) {
                    self.pair_map.insert(name.clone(), value.clone());
                }
            }
        }
        if self.directive_file_collection.override_skip_a {
            let key = DirectiveDef::SKIP.get_name_for_axis(Some(AxisID::First));
            if self.pair_map.contains_key(&key) {
                self.pair_map.remove(&key);
            }
        }
        if self.directive_file_collection.override_skip_b {
            let key = DirectiveDef::BSKIP.get_name_for_axis(Some(AxisID::Second));
            if self.pair_map.contains_key(&key) {
                self.pair_map.remove(&key);
            }
        }
    }

    /// Java `iterator()`: the entries of `pairMap`.  The Java method never returns
    /// null; the `Option` is always `Some`.
    pub fn iterator(&self) -> Option<Vec<(String, Option<String>)>> {
        Some(
            self.pair_map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        )
    }
}
