//! `IMOD/Etomo/src/etomo/storage/IntermediateFileFilter.java`.
//!
//! `IntermediateFileFilter extends ExtensionFileFilter`.  The instance is built
//! and then changed (`setAcceptPretrimmedTomograms` calls the superclass's
//! `add`) through a shared `Rc` on the event dispatch thread, so the embedded
//! superclass is a `RefCell` and the `boolean` field a `Cell`.  The
//! `accept(File)` override is the inherent `accept`, and with
//! `getDescription` it is also the `jdk::FileFilter` implementation.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::util::utilities;

/// Java `String.matches(regex)`: the whole input must match.  A pattern Java
/// would reject with `PatternSyntaxException` (only possible when a dataset
/// name is spliced into one) matches nothing.
fn matches(name: &str, regex: &str) -> bool {
    match Regex::new(&format!("^(?:{regex})$")) {
        Ok(regex) => regex.is_match(name),
        Err(_) => false,
    }
}

/// Java's `\S`: anything but `[ \t\n\x0B\f\r]`.
const NON_SPACE: &str = "[^ \\t\\n\\x0B\\x0C\\r]";

/// Java `public class IntermediateFileFilter extends ExtensionFileFilter`.
pub struct IntermediateFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: RefCell<ExtensionFileFilter>,
    /// Java private final `datasetName`.
    dataset_name: Option<String>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private `acceptPretrimmedTomograms = false`.
    accept_pretrimmed_tomograms: Cell<bool>,
}

impl IntermediateFileFilter {
    /// Java private `IntermediateFileFilter(BaseManager, String)`.  Call init
    /// to complete.
    fn new(
        manager: &'static dyn BaseManager,
        dataset_name: Option<&str>,
    ) -> IntermediateFileFilter {
        IntermediateFileFilter {
            base: RefCell::new(ExtensionFileFilter::new(
                Some(manager),
                etomo_director::INSTANCE.is_unit_test(),
                false,
                false,
            )),
            dataset_name: dataset_name.map(str::to_owned),
            manager,
            accept_pretrimmed_tomograms: Cell::new(false),
        }
    }

    /// Java static `getInstance(BaseManager, String)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        dataset_name: Option<&str>,
    ) -> Rc<IntermediateFileFilter> {
        let instance = IntermediateFileFilter::new(manager, dataset_name);
        instance.setup();
        Rc::new(instance)
    }

    /// Java private `setup()`.  Call after construction.
    fn setup(&self) {
        // "matchcheck.rec", ".mat" have been removed. Matchcheck.rec and matchshift.mat have
        // been gone for many years.
        // processchunk rec files are standardized
        let rec_suffix = extension::CLASS.rec.get_suffix(
            // `manager.getBaseMetaData().getImageFilenameStyle()`: Java dereferences
            // the meta data without a null check; a missing one gives `getSuffix(null)`.
            self.manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_image_filename_style()),
        );
        let dataset_name = self
            .dataset_name
            .clone()
            .unwrap_or_else(|| "null".to_string());
        let suffixes = vec![
            "~".to_string(),
            "volcombine.log".to_string(),
            "bot".to_string() + &rec_suffix,
            "bot".to_string() + &AxisID::First.get_extension() + &rec_suffix,
            "bot".to_string() + &AxisID::Second.get_extension() + &rec_suffix,
            "mid".to_string() + &rec_suffix,
            "mid".to_string() + &AxisID::First.get_extension() + &rec_suffix,
            "mid".to_string() + &AxisID::Second.get_extension() + &rec_suffix,
            "top".to_string() + &rec_suffix,
            "top".to_string() + &AxisID::First.get_extension() + &rec_suffix,
            "top".to_string() + &AxisID::Second.get_extension() + &rec_suffix,
            "_3dfind".to_string() + &rec_suffix,
            dataset_name.clone() + &AxisID::First.get_extension() + &rec_suffix,
            dataset_name + &AxisID::Second.get_extension() + &rec_suffix,
        ];
        self.base.borrow_mut().setup(
            self.dataset_name.as_deref(),
            None,
            None,
            Some(&suffixes),
            Some(&[
                &extension::CLASS.ali,
                &extension::CLASS.preali,
                &extension::CLASS.bl,
                &extension::CLASS.dcst,
                &extension::CLASS.alilog10,
                &extension::CLASS.vsr,
                &extension::CLASS.mat,
            ]),
            Some(&[
                Some(&*file_type::CLASS.processchunks_rec),
                Some(&*file_type::CLASS.full_vsr),
                Some(&*file_type::CLASS.sub_vsr),
                Some(&*file_type::CLASS.fixed_xrays_stack),
            ]),
        );
    }

    /// Java `setAcceptPretrimmedTomograms()`.
    pub fn set_accept_pretrimmed_tomograms(&self) {
        self.accept_pretrimmed_tomograms.set(true);
        let rec_suffix = extension::CLASS.rec.get_suffix(
            // `manager.getBaseMetaData().getImageFilenameStyle()`: Java dereferences
            // the meta data without a null check; a missing one gives `getSuffix(null)`.
            self.manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_image_filename_style()),
        );
        self.base
            .borrow_mut()
            .add(Some(&("sum".to_string() + &rec_suffix)));
        self.base
            .borrow_mut()
            .add(Some(&("full".to_string() + &rec_suffix)));
    }

    /// Java `accept(File)`.
    // TODO 2206
    pub fn accept(&self, f: &Path) -> bool {
        if self.base.borrow().accept(Some(f)) {
            return true;
        }
        if f.is_file() || etomo_director::INSTANCE.is_unit_test() {
            // These temporary files are not standardized.
            // .rec.mat1659344 and .rec.wrp0905524
            let name = utilities::java_io_file_get_name(&f.to_string_lossy());
            if matches(&name, &format!("{NON_SPACE}+\\.rec\\.mat{NON_SPACE}+"))
                || matches(&name, &format!("{NON_SPACE}+\\.rec\\.wrp{NON_SPACE}+"))
            {
                return true;
            }
            // handle split... and processchunks files
            // gets standardized
            // last one should be \d+ (one or more) because it can go beyond three digits
            /*   if (name.matches(datasetName + "[ab]?-\\d\\d\\d+\\.rec")) {
              return true;
            }*/
            if matches(&name, "tilt[ab]?-[0-9][0-9][0-9]+\\.log") {
                return true;
            }
            if matches(&name, "tilt[ab]?-[0-9][0-9][0-9]+\\.com") {
                return true;
            }
            if matches(&name, "volcombine[ab]?-[0-9][0-9][0-9]+\\.log") {
                return true;
            }
            if matches(&name, "volcombine[ab]?-[0-9][0-9][0-9]+\\.com") {
                return true;
            }
        }
        false
    }

    /// Java inherited `ExtensionFileFilter.accept(File, String)`, whose virtual
    /// `accept(File)` call resolves to this class's override.  Its own
    /// directory test repeats the first test `super.accept(File)` makes, so the
    /// result is `accept(new File(dir, name))`.
    pub fn accept_in_dir(&self, dir: Option<&Path>, name: Option<&str>) -> bool {
        let Some(name) = name else {
            return false;
        };
        let file = utilities::java_io_file_new(
            &dir.map(|dir| dir.to_string_lossy().to_string())
                .unwrap_or("null".to_string()),
            name,
        );
        self.accept(Path::new(&file))
    }

    /// Java `acceptForRegressionTest(File)` (deprecated: only for testing).
    /// Old version.  For testing against the modified accept.
    pub fn accept_for_regression_test(&self, f: &Path) -> bool {
        let ends_with = [
            "~",
            "matchcheck.rec",
            ".mat",
            ".ali",
            ".preali",
            "bot.rec",
            "bota.rec",
            "botb.rec",
            "mid.rec",
            "mida.rec",
            "midb.rec",
            "top.rec",
            "topa.rec",
            "topb.rec",
            "volcombine.log",
            ".bl",
            ".dcst",
            ".alilog10",
            ".vsr",
            "_3dfind.rec",
        ];
        let pretrimmed_tomograms = ["sum.rec", "full.rec"];
        let dataset_name = self
            .dataset_name
            .clone()
            .unwrap_or_else(|| "null".to_string());
        if f.is_file() || etomo_director::INSTANCE.is_unit_test() {
            // .rec.mat1659344 and .rec.wrp0905524
            let name = utilities::java_io_file_get_name(&f.to_string_lossy());
            if matches(&name, &format!("{NON_SPACE}+\\.rec\\.mat{NON_SPACE}+"))
                || matches(&name, &format!("{NON_SPACE}+\\.rec\\.wrp{NON_SPACE}+"))
            {
                return true;
            }
            let path = utilities::java_io_file_get_absolute_path(&f.to_string_lossy());
            for i in 0..ends_with.len() {
                if path.ends_with(ends_with[i]) {
                    return true;
                }
            }
            if self.accept_pretrimmed_tomograms.get() {
                for i in 0..pretrimmed_tomograms.len() {
                    if path.ends_with(pretrimmed_tomograms[i]) {
                        return true;
                    }
                }
            }
            if path.ends_with(&(dataset_name.clone() + "a.rec")) {
                return true;
            }
            if path.ends_with(&(dataset_name.clone() + "b.rec")) {
                return true;
            }
            // handle split... and processchunks files
            if matches(
                &name,
                &(dataset_name.clone() + "[ab]?-[0-9][0-9][0-9]\\.rec"),
            ) {
                return true;
            }
            if matches(&name, "tilt[ab]?-[0-9][0-9][0-9]\\.log") {
                return true;
            }
            if matches(&name, "tilt[ab]?-[0-9][0-9][0-9]\\.com") {
                return true;
            }
            if matches(&name, "volcombine[ab]?-[0-9][0-9][0-9]\\.log") {
                return true;
            }
            if matches(&name, "volcombine[ab]?-[0-9][0-9][0-9]\\.com") {
                return true;
            }
            if self.accept_string_file_type(&name, &file_type::CLASS.erased_beads_stack_old) {
                return true;
            }
            // TODO(unit): needs etomo/logic/DatasetTool.java - Java
            // `if (accept(name, FileType.FIXED_XRAYS_STACK_OLD)) return true;`: that
            // deprecated FileType singleton is not declared yet (see file_type.rs).
            if self.accept_string_file_type(&name, &file_type::CLASS.ctf_corrected_stack_old) {
                return true;
            }
            if self.accept_string_file_type(&name, &file_type::CLASS.mtf_filtered_stack_old) {
                return true;
            }
            if matches(
                &name,
                &(dataset_name.clone() + "[ab]?_full\\.vsr[0-9][0-9]"),
            ) {
                return true;
            }
            if matches(&name, &(dataset_name + "[ab]?_sub\\.vsr[0-9][0-9]")) {
                return true;
            }
        }
        false
    }

    /// Java private `accept(String, FileType)` (deprecated).  Returns true if
    /// fileName matches file type.  Only works for file types that use the
    /// dataset and the axisID.
    fn accept_string_file_type(&self, file_name: &str, file_type: &FileType) -> bool {
        if !file_type.uses_dataset() || !file_type.uses_axis_id() {
            return false;
        }
        if file_name.ends_with(
            &(file_type.get_type_string().unwrap_or("null").to_string()
                + file_type.get_extension().unwrap_or("null")),
        ) {
            return true;
        }
        false
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "Intermediate files"
    }
}

impl FileFilter for IntermediateFileFilter {
    fn accept(&self, file: &Path) -> bool {
        IntermediateFileFilter::accept(self, file)
    }

    fn get_description(&self) -> Option<String> {
        Some(IntermediateFileFilter::get_description(self).to_string())
    }
}
