//! `IMOD/Etomo/src/etomo/util/RemotePath.java` (with its nested
//! `InvalidMountRuleException`, `MountRule`, `MountRuleStringIterator` and
//! `MountRuleIterator`).
//!
//! Description: A singleton class which loads mount rules from the cpu.adoc file and
//! uses them to translate local paths into paths which can be used on remote computers
//! which share a file system with the current host computer.  Currently this class only
//! works with Linux and MacIntosh.  (The source's class comment, with its worked
//! examples of mount rules, `%mountname`, rule ordering and overriding, applies
//! unchanged.)
//!
//! Warnings of problems with the mount rules in cpu.adoc will be placed in
//! etomo_err.log.
//!
//! Copyright: Copyright 2005 - 2024 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  `INSTANCE` is shared by every thread and `loadMountRules` is
//! `synchronized`, so the fields sit behind one `Mutex` and every method takes `&self`.
//! The Java field `localSection` holds a `ReadOnlySection` of the (per-thread,
//! raw-pointer) autodoc; only its name is ever read (`isLocalSection`), so the name is
//! what is kept.  The public iterator methods return the strings the Java `Iterator`
//! would yield, in order.

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::storage::autodoc::attribute::Attribute;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::autodoc::read_only_attribute_list::ReadOnlyAttributeList;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::autodoc::read_only_statement_list::ReadOnlyStatementList;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::storage::node;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::util::dataset_files;
use std::sync::{LazyLock, Mutex};

/// Java `INSTANCE`.
pub static INSTANCE: LazyLock<RemotePath> = LazyLock::new(RemotePath::new);
/// Java private static `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

/// Java package-private `MOUNT_RULE`.
pub(crate) const MOUNT_RULE: &str = "mountrule";
/// Java package-private `LOCAL`.
pub(crate) const LOCAL: &str = "local";
/// Java package-private `REMOTE`.
pub(crate) const REMOTE: &str = "remote";
/// Java package-private `AUTODOC`: `AutodocFactory.CPU`.
pub(crate) const AUTODOC: &str = autodoc_factory::CPU;
/// Java package-private `MOUNT_NAME`.
pub(crate) const MOUNT_NAME: &str = "mountname";
/// Java package-private `MOUNT_NAME_TAG`: `EtomoAutodoc.VAR_TAG + MOUNT_NAME`.
pub(crate) static MOUNT_NAME_TAG: LazyLock<String> =
    LazyLock::new(|| format!("{}{}", etomo_autodoc::VAR_TAG, MOUNT_NAME));

/// The mutable fields.
struct State {
    /// Java private field `mountRuleArray`, initialised to null.
    mount_rule_array: Option<Vec<MountRule>>,
    /// Java private field `mountRulesLoaded`, initialised to false.
    mount_rules_loaded: bool,
    /// Java private field `mountName`, initialised to null.
    mount_name: Option<String>,
    /// Java private field `hostName`, initialised to null.
    host_name: Option<String>,
    /// Java private field `localSection` (a `ReadOnlySection`), kept as its name.
    local_section: Option<String>,
}

/// Java `RemotePath`.
pub struct RemotePath {
    state: Mutex<State>,
}

impl RemotePath {
    /// Java private constructor `RemotePath()`.
    fn new() -> RemotePath {
        RemotePath {
            state: Mutex::new(State {
                mount_rule_array: None,
                mount_rules_loaded: false,
                mount_name: None,
                host_name: None,
                local_section: None,
            }),
        }
    }

    /// Java `getRemotePath(BaseManager, String, AxisID, Boolean)`.  Loads mount rules if
    /// necesasry.  Finds the first local rule which matches localPath.  Converts
    /// localPath to a remote path.  Returns remote path or null.
    ///
    /// A null `localPath` (the callers pass `manager.getPropertyUserDir()`) reaches
    /// `localPath.startsWith` in `getRule` and throws a NullPointerException whenever a
    /// mount rule exists (RemotePath.java:640).  Fixed in translation: a null path
    /// matches no rule, so the result is null.
    pub fn get_remote_path(
        &self,
        manager: &'static dyn BaseManager,
        local_path: Option<&str>,
        axis_id: AxisID,
        global_mount_rules: Option<bool>,
    ) -> Result<Option<String>, InvalidMountRuleException> {
        let mut state = self.state.lock().unwrap();
        self.load_mount_rules(&mut state, manager, axis_id);
        let local_path = match local_path {
            None => return Ok(None),
            Some(local_path) => local_path,
        };
        let mount_rule = self.get_rule(&state, local_path, global_mount_rules);
        if let Some(mount_rule) = mount_rule {
            return self.get_remote_path_for_rule(&state, Some(&mount_rule), local_path);
        }
        Ok(None)
    }

    /// Java `isLocalSection(String, BaseManager, AxisID)`.  Loads mount rules if
    /// necesasry.  Returns true if sectionName equals "localhost" or matches the
    /// section name of the local computer.
    pub fn is_local_section(
        &self,
        section_name: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> bool {
        let section_name = match section_name {
            None => return false,
            Some(section_name) => section_name,
        };
        if section_name == node::LOCAL_HOST_NAME {
            return true;
        }
        let mut state = self.state.lock().unwrap();
        self.load_mount_rules(&mut state, manager, axis_id);
        // check the local section
        match &state.local_section {
            None => false,
            Some(local_section) => section_name == local_section,
        }
    }

    /// Java private `getRemotePath(MountRule, String)`.  Given a local mount rule which
    /// matches localPath, converts localPath to a remote path.  Returns remote path or
    /// null.
    fn get_remote_path_for_rule(
        &self,
        state: &State,
        mount_rule: Option<&MountRule>,
        local_path: &str,
    ) -> Result<Option<String>, InvalidMountRuleException> {
        let mount_rule = match mount_rule {
            None => return Ok(None),
            Some(mount_rule) => mount_rule,
        };
        let mut remote_mount_rule = mount_rule.get_remote_rule().unwrap_or("null").to_string();
        // look for %mountname in remote mount rule
        let mount_name_index = remote_mount_rule.find(MOUNT_NAME_TAG.as_str());
        if let Some(mount_name_index) = mount_name_index {
            // (The source's "%mountname has no value" InvalidMountRuleException is
            // commented out.)
            // substitute mount name
            let mut buffer = String::new();
            if mount_name_index > 0 {
                buffer.push_str(&remote_mount_rule[..mount_name_index]);
            }
            buffer.push_str(state.mount_name.as_deref().unwrap_or("null"));
            if mount_name_index + MOUNT_NAME_TAG.len() < remote_mount_rule.len() {
                buffer.push_str(&remote_mount_rule[mount_name_index + MOUNT_NAME_TAG.len()..]);
            }
            remote_mount_rule = buffer;
        }

        // create remote path
        // WAS: return remoteMountRule + localPath
        // .substring(((String) localMountRules.get(ruleIndex)).length(), localPath.length());
        let local_rule_len = mount_rule.get_local_rule().map_or(4, str::len);
        Ok(Some(format!(
            "{}{}",
            remote_mount_rule,
            &local_path[local_rule_len..]
        )))
    }

    /// Java private synchronized `loadMountRules(BaseManager, AxisID)`.  Loads mount
    /// rules from cpu.adoc.  Tries to find a Computer section for the current host
    /// computer.  Loads section mount rules if a section has been found.  Loads global
    /// mount rules.
    fn load_mount_rules(
        &self,
        state: &mut State,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) {
        // only try to load mount rules once
        if state.mount_rules_loaded {
            return;
        }
        state.mount_rules_loaded = true;
        state.mount_rule_array = Some(Vec::new());

        let autodoc = match unsafe {
            autodoc_factory::get_instance(Some(manager), Some(AUTODOC), axis_id, false)
        } {
            Ok(autodoc) => autodoc,
            // `catch (final LockException e) { return; }`.
            Err(LogFileError::Lock(_)) => return,
            Err(e) => {
                // `e.printStackTrace()`.
                eprintln!("{}", e);
                return;
            }
        };
        let autodoc: &Autodoc = match unsafe { autodoc.as_ref() } {
            None => return,
            Some(autodoc) => autodoc,
        };
        // first load section-level mount rules
        // look for a section name that is the same as the output of hostname
        state.host_name = Network::get_local_host_name(
            manager,
            axis_id,
            manager.get_property_user_dir().as_deref(),
        );
        let section_type = cpu_adoc::INSTANCE.get_computer_section_type();

        if let Some(section_type) = &section_type {
            let host_name = state.host_name.clone();
            state.local_section = self.load_mount_rules_from_section(
                state,
                autodoc,
                section_type,
                host_name.as_deref(),
                true,
            );
            if state.local_section.is_none() {
                // try looking for a section name that is the same as the stripped
                // version of the hostname.
                if let Some(host_name) = &host_name {
                    let strip_index = host_name.find('.');
                    let stripped_found = match strip_index {
                        None => false,
                        Some(strip_index) => {
                            state.local_section = self.load_mount_rules_from_section(
                                state,
                                autodoc,
                                section_type,
                                Some(&host_name[..strip_index]),
                                true,
                            );
                            state.local_section.is_some()
                        }
                    };
                    if !stripped_found {
                        // look for a section name called "localhost"
                        state.local_section = self.load_mount_rules_from_section(
                            state,
                            autodoc,
                            section_type,
                            Some(node::LOCAL_HOST_NAME),
                            false,
                        );
                    }
                }
            }
        }
        // load global mount rules
        self.load_global_mount_rules(state, autodoc);
    }

    /// Java private `loadMountRules(ReadOnlyAutodoc, String, String, boolean)`.  Loads
    /// mount rules from a section if the section exists.  Returns the section (its
    /// name), or null if the section doesn't exist.
    fn load_mount_rules_from_section(
        &self,
        state: &mut State,
        autodoc: &Autodoc,
        section_type: &str,
        section_name: Option<&str>,
        section_name_can_be_mount_name: bool,
    ) -> Option<String> {
        let section =
            unsafe { ReadOnlySectionList::get_section(autodoc, Some(section_type), section_name) };
        let section = unsafe { section.as_ref() }?;
        // set mount name
        let mount_name: Option<&Attribute> =
            unsafe { ReadOnlySection::get_attribute(section, Some(MOUNT_NAME)).as_ref() };
        match mount_name {
            None => {
                if section_name_can_be_mount_name {
                    // use section name as mount name
                    state.mount_name = section_name.map(str::to_string);
                }
            }
            Some(mount_name) => {
                // load mount name from "mountname" attribute
                state.mount_name = mount_name.get_value();
                if state.mount_name.is_none() {
                    // allow empty mount name
                    state.mount_name = Some(String::new());
                }
            }
        }
        // load mount rules
        let mount_rules_attribute: Option<&Attribute> =
            unsafe { ReadOnlySection::get_attribute(section, Some(MOUNT_RULE)).as_ref() };
        self.load_mount_rules_from_attribute(
            state,
            mount_rules_attribute,
            Some(section_type),
            section_name,
        );
        Some(ReadOnlyStatementList::get_name(section).unwrap_or_default())
    }

    /// Java private `loadMountRules(ReadOnlyAutodoc)`.  Loads mount rules from the
    /// global attribute area of cpu.adoc.
    fn load_global_mount_rules(&self, state: &mut State, autodoc: &Autodoc) {
        let mount_rules_attribute: Option<&Attribute> =
            unsafe { ReadOnlyAutodoc::get_attribute(autodoc, Some(MOUNT_RULE)).as_ref() };
        self.load_mount_rules_from_attribute(state, mount_rules_attribute, None, None);
    }

    /// Java private `loadMountRules(ReadOnlyAttribute, String, String)`.  Loads mount
    /// rules from a mountrule attribute.
    ///
    /// A `mountrule` attribute with no children (`mountrule = x`) makes `getChildren()`
    /// null, and RemotePath.java:474-477 calls `iterator()` on it (NullPointerException).
    /// Fixed in translation: such an attribute holds no rules.
    fn load_mount_rules_from_attribute(
        &self,
        state: &mut State,
        mount_rules_attribute: Option<&Attribute>,
        section_type: Option<&str>,
        section_name: Option<&str>,
    ) {
        let mount_rules_attribute = match mount_rules_attribute {
            None => return,
            Some(mount_rules_attribute) => mount_rules_attribute,
        };
        let global_section = section_name.is_none();
        let location = if global_section {
            "the global section".to_string()
        } else {
            format!(
                "the {} {} section",
                section_name.unwrap(),
                section_type.unwrap_or("null")
            )
        };
        // load the rules in order of the mountrule number
        let number_attribute_list = match unsafe { mount_rules_attribute.get_children().as_ref() } {
            None => return,
            Some(number_attribute_list) => number_attribute_list,
        };
        // Size a temporary array based on the highest number mount rule in the current
        // section.
        let mut iterator = number_attribute_list.iterator();
        let mut max_index: i32 = -1;
        while iterator.has_next() {
            let number_attribute: &Attribute = unsafe { &**iterator.next().unwrap() };
            let name = number_attribute.get_name();
            // Mount rule number must be an integer.
            let rule_number = converter::to_integer(Some(&name));
            let rule_number = match rule_number {
                None => {
                    eprintln!(
                        "ERROR: Bad mountrule number (\"{}\") in {}.\n       In $IMOD_CALIB_DIR/cpu.adoc the mountrule keyword must be followed by a positive integer.\n       Type 'man cpuadoc' on the command line for more information.",
                        name, location
                    );
                    continue;
                }
                Some(rule_number) => rule_number,
            };
            // A negative attribute number may cause the mount rules to be saved in the
            // wrong order.
            if rule_number < 0 {
                eprintln!(
                    "ERROR: A mountrule with a negative number ({}) in {}.  Mountrule will be skipped.\n       Please correct in $IMOD_CALIB_DIR/cpu.adoc.",
                    rule_number, location
                );
                continue;
            }
            if rule_number > max_index {
                max_index = rule_number;
            }
        }
        if max_index < 0 {
            return;
        }
        // Fill a temporary array.
        let mut temp_mount_rule_array: Vec<Option<MountRule>> = vec![None; max_index as usize + 1];
        iterator = number_attribute_list.iterator();
        while iterator.has_next() {
            let number_attribute: &Attribute = unsafe { &**iterator.next().unwrap() };
            let name = number_attribute.get_name();
            // Mount rule number must be an integer.
            let rule_number = match converter::to_integer(Some(&name)) {
                None => continue,
                Some(rule_number) => rule_number,
            };
            // A negative attribute number may cause the mount rules to be saved in the
            // wrong order.
            if rule_number < 0 {
                continue;
            }
            let local_rule: Option<&Attribute> =
                unsafe { number_attribute.get_attribute_by_name(Some(LOCAL)).as_ref() };
            let remote_rule: Option<&Attribute> = unsafe {
                number_attribute
                    .get_attribute_by_name(Some(REMOTE))
                    .as_ref()
            };
            // run valid rule check
            if !self.is_valid_rule(
                state,
                local_rule,
                remote_rule,
                rule_number,
                section_type,
                section_name,
            ) {
                continue;
            }
            // Add the rule using the ruleNumber as the index. This fixes a bug where
            // mount rules were dropped (Bug# 2446). Very few mount rules will be used per
            // section, so use an incompletely populated array rather then rewriting it
            // with a sorted collection.
            temp_mount_rule_array[rule_number as usize] = Some(MountRule::new(
                Some(rule_number),
                local_rule.unwrap().get_value(),
                remote_rule.unwrap().get_value(),
                global_section,
            ));
        }
        //
        // Fill the mountRuleArray.
        let mount_rule_array = state.mount_rule_array.get_or_insert_with(Vec::new);
        for mount_rule in temp_mount_rule_array.into_iter().flatten() {
            mount_rule_array.push(mount_rule);
        }
    }

    /// Java private `isValidRule(ReadOnlyAttribute, ReadOnlyAttribute, int, String,
    /// String)`.  Checks the validity of a localRule and remoteRule.  Writes warning
    /// messages to etomo_err.log.  Returns true if the rule is valid, false if it is
    /// invalid.
    fn is_valid_rule(
        &self,
        state: &State,
        local_rule: Option<&Attribute>,
        remote_rule: Option<&Attribute>,
        mount_rule_number: i32,
        section_type: Option<&str>,
        section_name: Option<&str>,
    ) -> bool {
        // create the start of the error message
        let mut error_title = "Warning:  Problem".to_string();
        if let Some(host_name) = &state.host_name {
            error_title.push_str(&format!(" using {}", host_name));
        }
        error_title.push_str(&format!(
            " with {}",
            dataset_files::get_autodoc_name(Some(AUTODOC))
        ));
        let section_type = match section_type {
            Some(section_type) => Some(section_type.to_string()),
            None => cpu_adoc::INSTANCE.get_computer_section_type(),
        };
        let section_header = format!(
            "{}{} {} ",
            autodoc_tokenizer::OPEN_CHAR,
            section_type.as_deref().unwrap_or("null"),
            autodoc_tokenizer::DEFAULT_DELIMITER
        );
        if let Some(section_name) = section_name {
            error_title.push_str(&format!(
                ", section {}{}{}",
                section_header,
                section_name,
                autodoc_tokenizer::CLOSE_CHAR
            ));
        }
        error_title.push_str(" - ");
        // validate local rule
        if !self.is_valid_rule_single(local_rule, mount_rule_number, &error_title, "local") {
            return false;
        }
        // validate remote rule
        if !self.is_valid_rule_single(remote_rule, mount_rule_number, &error_title, "remote") {
            return false;
        }
        // can't use %mountname if there is no mount name
        // causes: either no section for this computer or the localhost section doesn't
        // have a mountname attribute
        let remote_value = remote_rule.unwrap().get_value();
        if remote_value
            .as_deref()
            .is_some_and(|remote_value| remote_value.contains(MOUNT_NAME_TAG.as_str()))
            && state.mount_name.is_none()
        {
            if *DEBUG {
                eprintln!(
                    "{}remote mount rule {}.  Cannot use {} because there is no mountname.\nEither there is no mountname entry under the {}{}{} section or there is no section for this computer.\n",
                    error_title,
                    mount_rule_number,
                    *MOUNT_NAME_TAG,
                    section_header,
                    node::LOCAL_HOST_NAME,
                    autodoc_tokenizer::CLOSE_CHAR
                );
            }
            // pass this problem so that it can be shown to the user
            return true;
        }
        true
    }

    /// Java private `isValidRule(ReadOnlyAttribute, int, String, String)`.  Checks the
    /// validity of a rule.  Writes warning messages to etomo_err.log.
    fn is_valid_rule_single(
        &self,
        rule: Option<&Attribute>,
        mount_rule_number: i32,
        error_title: &str,
        rule_type: &str,
    ) -> bool {
        // rule must exist
        let rule = match rule {
            None => {
                if *DEBUG {
                    eprintln!(
                        "{}{} mount rule {} is missing.",
                        error_title, rule_type, mount_rule_number
                    );
                }
                return false;
            }
            Some(rule) => rule,
        };
        let value = rule.get_value();
        // local rule must not have an empty value
        let value = match value {
            None => {
                if *DEBUG {
                    eprintln!(
                        "{}{} mount rule {} cannot be blank.",
                        error_title, rule_type, mount_rule_number
                    );
                }
                return false;
            }
            Some(value) => value,
        };
        if !std::path::Path::new(&value).is_absolute() {
            if *DEBUG {
                eprintln!(
                    "{}{} mount rule {} must be an absolute directory path.",
                    error_title, rule_type, mount_rule_number
                );
            }
            return false;
        }
        true
    }

    /// Java private `getRule(String, Boolean)`.  Tries to match localPath against a
    /// local rule.  Returns the matching rule.
    fn get_rule(
        &self,
        state: &State,
        local_path: &str,
        global_mount_rules: Option<bool>,
    ) -> Option<MountRule> {
        // check the rules in order
        let mut iterator = MountRuleIterator::new(global_mount_rules);
        while iterator.has_next(state) {
            // WAS if (localPath.startsWith((String) localMountRules.get(i))) {
            let mount_rule = iterator.next(state);
            if let Some(mount_rule) = mount_rule
                && local_path.starts_with(mount_rule.get_local_rule().unwrap_or("null"))
            {
                return Some(mount_rule);
            }
        }
        None
    }

    /// Java package-private `reset`.  Resets the instances so that mount rules can be
    /// reloaded.
    pub(crate) fn reset(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            panic!("java.lang.IllegalStateException");
        }
        let mut state = self.state.lock().unwrap();
        state.mount_rule_array = None;
        state.mount_rules_loaded = false;
        state.mount_name = None;
        state.host_name = None;
    }

    /// Java package-private `getHostName_test`.  For testing.  Calls the static
    /// getHostName() function.
    pub(crate) fn get_host_name_test(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Option<String> {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            panic!("java.lang.IllegalStateException");
        }
        Network::get_local_host_name(manager, axis_id, manager.get_property_user_dir().as_deref())
    }

    /// Java package-private `isMountRulesLoaded_test`.  For testing.
    pub(crate) fn is_mount_rules_loaded_test(&self) -> bool {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            panic!("java.lang.IllegalStateException");
        }
        self.state.lock().unwrap().mount_rules_loaded
    }

    /// Java package-private `mountRuleArrayIsNull_test`.  For testing.
    pub(crate) fn mount_rule_array_is_null_test(&self) -> bool {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            panic!("java.lang.IllegalStateException");
        }
        self.state.lock().unwrap().mount_rule_array.is_none()
    }

    /// Java package-private `getMountRuleArraySize_test`.  For testing.  (The source
    /// dereferences a null array; zero is returned for it here.)
    pub(crate) fn get_mount_rule_array_size_test(&self) -> i32 {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            panic!("java.lang.IllegalStateException");
        }
        self.state
            .lock()
            .unwrap()
            .mount_rule_array
            .as_ref()
            .map_or(0, |array| array.len() as i32)
    }

    /// Java `localGlobalRuleIterator(BaseManager, AxisID)`.  Returns the local rules in
    /// the global section in the order of the mount rule number.
    pub fn local_global_rule_iterator(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Vec<String> {
        let mut state = self.state.lock().unwrap();
        self.load_mount_rules(&mut state, manager, axis_id);
        MountRuleStringIterator::new(true, Some(true)).collect(&state)
    }

    /// Java `remoteGlobalRuleIterator(BaseManager, AxisID)`.  Returns the remote rules
    /// in the global section in the order of the mount rule number.
    pub fn remote_global_rule_iterator(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> Vec<String> {
        let mut state = self.state.lock().unwrap();
        self.load_mount_rules(&mut state, manager, axis_id);
        MountRuleStringIterator::new(false, Some(true)).collect(&state)
    }

    /// Java `numGlobalEntries(BaseManager, AxisID)`.
    pub fn num_global_entries(&self, manager: &'static dyn BaseManager, axis_id: AxisID) -> i32 {
        let mut state = self.state.lock().unwrap();
        self.load_mount_rules(&mut state, manager, axis_id);
        if state.mount_rule_array.is_none() {
            return 0;
        }
        let mut global_entry_count = 0;
        let mut iterator = MountRuleIterator::new(Some(true));
        while iterator.has_next(&state) {
            global_entry_count += 1;
            iterator.next(&state);
        }
        global_entry_count
    }

    /// Java package-private `iterator_test(BaseManager, AxisID, Boolean)`.  The rules
    /// the returned iterator would yield, in order.
    pub(crate) fn iterator_test(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        global_section: Option<bool>,
    ) -> Vec<Option<MountRule>> {
        let mut state = self.state.lock().unwrap();
        self.load_mount_rules(&mut state, manager, axis_id);
        let mut iterator = MountRuleIterator::new(global_section);
        let mut rules = Vec::new();
        while iterator.has_next(&state) {
            rules.push(iterator.next(&state));
        }
        rules
    }
}

/// Java public static final nested class `RemotePath.InvalidMountRuleException extends
/// Exception`.
#[derive(Clone, Debug)]
pub struct InvalidMountRuleException {
    /// `Throwable`'s detail message.
    message: Option<String>,
}

impl InvalidMountRuleException {
    /// Java package-private `InvalidMountRuleException(String)`.
    pub(crate) fn new(message: Option<&str>) -> InvalidMountRuleException {
        InvalidMountRuleException {
            message: message.map(str::to_string),
        }
    }

    /// `Throwable.getMessage()`; a null message is "null" when concatenated, which is
    /// how every caller uses it.
    pub fn get_message(&self) -> String {
        self.message.clone().unwrap_or_else(|| "null".to_string())
    }
}

impl std::fmt::Display for InvalidMountRuleException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.get_message())
    }
}

impl std::error::Error for InvalidMountRuleException {}

/// Java package-private static final nested class `MountRule`.
#[derive(Clone, Debug)]
pub(crate) struct MountRule {
    /// Java private final field `ruleNumber`, an `Integer`.
    rule_number: Option<i32>,
    /// Java private final field `localRule`.
    local_rule: Option<String>,
    /// Java private final field `remoteRule`.
    remote_rule: Option<String>,
    /// Java private final field `globalSection`.
    global_section: bool,
}

impl MountRule {
    /// Java private constructor `MountRule(Integer, String, String, boolean)`.
    fn new(
        rule_number: Option<i32>,
        local_rule: Option<String>,
        remote_rule: Option<String>,
        global_section: bool,
    ) -> MountRule {
        MountRule {
            rule_number,
            local_rule,
            remote_rule,
            global_section,
        }
    }

    /// Java private `matchesGlobalSection(Boolean)`.
    fn matches_global_section(&self, global_section: Option<bool>) -> bool {
        match global_section {
            // No global section requirement - use any this.globalSection.
            None => true,
            Some(global_section) => global_section == self.global_section,
        }
    }

    /// Java package-private `getRuleNumber`.
    pub(crate) fn get_rule_number(&self) -> Option<i32> {
        self.rule_number
    }

    /// Java private `getLocalRule`.
    fn get_local_rule(&self) -> Option<&str> {
        self.local_rule.as_deref()
    }

    /// Java private `getRemoteRule`.
    fn get_remote_rule(&self) -> Option<&str> {
        self.remote_rule.as_deref()
    }
}

/// Java `MountRule.toString`.
impl std::fmt::Display for MountRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[ruleNumber:{},globalSection:{},localRule:{},remoteRule:{}]",
            self.rule_number
                .map_or("null".to_string(), |rule_number| rule_number.to_string()),
            self.global_section,
            self.local_rule.as_deref().unwrap_or("null"),
            self.remote_rule.as_deref().unwrap_or("null")
        )
    }
}

/// Java private final inner class `MountRuleStringIterator implements
/// Iterator<String>`.  The inner class reads the outer instance's `mountRuleArray`,
/// which is passed in as the locked state.
struct MountRuleStringIterator {
    /// Java private final field `iterator`.
    iterator: MountRuleIterator,
    /// Java private final field `localRule`.
    local_rule: bool,
}

impl MountRuleStringIterator {
    /// Java private constructor.  `local_rule` - true:returns local path, false:returns
    /// remote path; `global_section` - true:global only, false:localhost only,
    /// null:both.
    fn new(local_rule: bool, global_section: Option<bool>) -> MountRuleStringIterator {
        MountRuleStringIterator {
            local_rule,
            iterator: MountRuleIterator::new(global_section),
        }
    }

    /// Java `hasNext`.
    fn has_next(&mut self, state: &State) -> bool {
        self.iterator.has_next(state)
    }

    /// Java `next`.
    fn next(&mut self, state: &State) -> Option<String> {
        let mount_rule = self.iterator.next(state)?;
        if self.local_rule {
            mount_rule.get_local_rule().map(str::to_string)
        } else {
            mount_rule.get_remote_rule().map(str::to_string)
        }
    }

    /// The Java caller's `while (iterator.hasNext()) iterator.next()` walk.  A null
    /// rule value (never stored: `isValidRule` rejects it) is skipped.
    fn collect(mut self, state: &State) -> Vec<String> {
        let mut strings = Vec::new();
        while self.has_next(state) {
            if let Some(string) = self.next(state) {
                strings.push(string);
            }
        }
        strings
    }
}

/// Java private final inner class `MountRuleIterator implements Iterator<MountRule>`.
/// Gets mount rules in order of mount rule number.  The inner class reads the outer
/// instance's `mountRuleArray`, which is passed in as the locked state.  (`remove` is
/// never called: `MountRuleStringIterator.remove` has no caller.)
struct MountRuleIterator {
    /// Java private final field `globalSection`.
    global_section: Option<bool>,
    /// Java private field `index`, initialised to 0.
    index: usize,
}

impl MountRuleIterator {
    /// Java private constructor.  `global_section` - true:global only, false:localhost
    /// only, null:both.
    fn new(global_section: Option<bool>) -> MountRuleIterator {
        MountRuleIterator {
            global_section,
            index: 0,
        }
    }

    /// Java `hasNext`.
    fn has_next(&mut self, state: &State) -> bool {
        let mount_rule_array = match &state.mount_rule_array {
            None => return false,
            Some(mount_rule_array) => mount_rule_array,
        };
        let size = mount_rule_array.len();
        if self.index >= size {
            return false;
        }
        if self.global_section.is_none() {
            return true;
        }
        // Look for the next element that matches globalSection.
        for i in self.index..size {
            let mount_rule = &mount_rule_array[i];
            if mount_rule.matches_global_section(self.global_section) {
                // Valid element.
                if i > self.index {
                    // Skip the elements that don't match globalSection.
                    self.index = i;
                }
                return true;
            }
        }
        false
    }

    /// Java `next`.  Returns the next mount rule that matches globalSection.  A null
    /// globalSection matches all mount rules.
    fn next(&mut self, state: &State) -> Option<MountRule> {
        if !self.has_next(state) {
            return None;
        }
        let mount_rule_array = state.mount_rule_array.as_ref().unwrap();
        if self.global_section.is_none() {
            let mount_rule = mount_rule_array[self.index].clone();
            self.index += 1;
            return Some(mount_rule);
        }
        // Skip elements that don't match globalSection.
        let size = mount_rule_array.len();
        while self.index < size {
            let mount_rule = &mount_rule_array[self.index];
            self.index += 1;
            if mount_rule.matches_global_section(self.global_section) {
                return Some(mount_rule.clone());
            }
        }
        None
    }
}
