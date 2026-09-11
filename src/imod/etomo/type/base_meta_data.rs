//! `IMOD/Etomo/src/etomo/type/BaseMetaData.java`.
//!
//! Parent class for meta data classes.
//!
//! **Representation.**  `BaseMetaData` is an abstract class with five abstract methods
//! (`getMetaDataFileName`, `getName`, `getDatasetName`, `isValid`, `getGroupKey`) whose
//! concrete subclasses - `MetaData`, `JoinMetaData`, `PeetMetaData` and the rest - are
//! not translated.  As in `etomo/storage/autodoc/statement.rs`, the class splits into
//! the `BaseMetaData` trait (the abstract methods plus every method that calls one) and
//! `BaseMetaDataBase` (the fields it declares), reached through `base()`.
//!
//! **Mutability.**  Java hands the object out of `BaseManager.getBaseMetaData()` and
//! mutates it through that shared reference, so each mutable field carries its own lock,
//! the same modelling `etomo/storage/log_file.rs` uses for Java's instance monitors.
//!
//! Java `Properties` is modelled by a deterministic `BTreeMap<String, String>`, as
//! `etomo/storage/storable.rs` does.
#![allow(dead_code)]

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::etomo_version::EtomoVersion;
use super::extension::Extension;
use super::image_file_meta_data::ImageFileMetaData;
use super::image_filename_style::ImageFilenameStyle;
use super::image_output_format::ImageOutputFormat;
use super::imod_version;
use super::string_property::StringProperty;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::ui::log_properties::LogProperties;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Java private static `CURRENT_PROCESSCHUNKS_ROOT_NAME`.
const CURRENT_PROCESSCHUNKS_ROOT_NAME: &str = "CurrentProcesschunksRootName";

/// Java private static `CURRENT_PROCESSCHUNKS_SUBDIR_NAME`.
const CURRENT_PROCESSCHUNKS_SUBDIR_NAME: &str = "CurrentProcesschunksSubdirName";

/// Java package-private static `revisionNumberString`.
pub const REVISION_NUMBER_STRING: &str = "RevisionNumber";

/// Java private static `FIRST_SAVED_ETOMO_VERSION`.
const FIRST_SAVED_ETOMO_VERSION: &str = "4.10.43";

/// The fields Java's abstract `BaseMetaData` declares.  Every implementor embeds one and
/// returns it from `BaseMetaData::base`.
pub struct BaseMetaDataBase {
    /// Java package-private field `fileExtension`.
    pub file_extension: Mutex<Option<String>>,
    /// Java package-private field `revisionNumber`.  Should be set only by `load()`.
    pub revision_number: Mutex<EtomoVersion>,
    /// Java package-private field `axisType`, initialised to `AxisType.NOT_SET`.
    pub axis_type: Mutex<AxisType>,
    /// Java package-private field `invalidReason`, initialised to "".
    pub invalid_reason: Mutex<String>,
    /// Java field `currentProcesschunksRootNameA`.
    current_processchunks_root_name_a: Mutex<StringProperty>,
    /// Java field `currentProcesschunksRootNameB`.
    current_processchunks_root_name_b: Mutex<StringProperty>,
    /// Java field `currentProcesschunksSubdirNameA`.
    current_processchunks_subdir_name_a: Mutex<StringProperty>,
    /// Java field `currentProcesschunksSubdirNameB`.
    current_processchunks_subdir_name_b: Mutex<StringProperty>,
    /// Java field `etomoCreatedVersion`.
    etomo_created_version: Mutex<EtomoVersion>,
    /// Java field `etomoModifiedVersion`.
    etomo_modified_version: Mutex<EtomoVersion>,
    /// Java field `imageFileMetaData`.
    image_file_meta_data: Mutex<ImageFileMetaData>,
    /// Java field `logProperties`.
    log_properties: Option<&'static dyn LogProperties>,
    /// Java field `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java field `newDataset`.
    new_dataset: bool,
    /// Java field `canCorrectImageFilenameStyle`.
    can_correct_image_filename_style: bool,
}

impl BaseMetaDataBase {
    /// Java package-private `BaseMetaData(BaseManager, LogProperties, boolean, boolean,
    /// boolean)`, together with the field initialisers Java runs before it.
    pub fn new_force_old_style(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        force_old_style: bool,
        new_dataset: bool,
        can_correct_image_filename_style: bool,
    ) -> BaseMetaDataBase {
        BaseMetaDataBase::initial(
            manager,
            log_properties,
            new_dataset,
            can_correct_image_filename_style,
            ImageFileMetaData::new(force_old_style, None, new_dataset),
        )
    }

    /// Java package-private `BaseMetaData(BaseManager, LogProperties,
    /// ImageFilenameStyle, boolean, boolean)`.
    pub fn new_image_filename_style(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        image_filename_style: Option<ImageFilenameStyle>,
        new_dataset: bool,
        can_correct_image_filename_style: bool,
    ) -> BaseMetaDataBase {
        BaseMetaDataBase::initial(
            manager,
            log_properties,
            new_dataset,
            can_correct_image_filename_style,
            ImageFileMetaData::new(false, image_filename_style, new_dataset),
        )
    }

    /// Java package-private `BaseMetaData(BaseManager, LogProperties, boolean,
    /// ImageFilenameStyle, boolean, boolean)`.
    pub fn new_force_old_style_and_image_filename_style(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        force_old_style: bool,
        image_filename_style: Option<ImageFilenameStyle>,
        new_dataset: bool,
        can_correct_image_filename_style: bool,
    ) -> BaseMetaDataBase {
        BaseMetaDataBase::initial(
            manager,
            log_properties,
            new_dataset,
            can_correct_image_filename_style,
            ImageFileMetaData::new(force_old_style, image_filename_style, new_dataset),
        )
    }

    /// The field initialisers the three constructors share, with the
    /// already-constructed `imageFileMetaData` each of them builds differently.
    fn initial(
        manager: Option<&'static dyn BaseManager>,
        log_properties: Option<&'static dyn LogProperties>,
        new_dataset: bool,
        can_correct_image_filename_style: bool,
        image_file_meta_data: ImageFileMetaData,
    ) -> BaseMetaDataBase {
        BaseMetaDataBase {
            file_extension: Mutex::new(None),
            revision_number: Mutex::new(EtomoVersion::get_empty_instance(Some(
                REVISION_NUMBER_STRING,
            ))),
            axis_type: Mutex::new(AxisType::NotSet),
            invalid_reason: Mutex::new(String::new()),
            current_processchunks_root_name_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!("A.{}", CURRENT_PROCESSCHUNKS_ROOT_NAME),
            ))),
            current_processchunks_root_name_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!("B.{}", CURRENT_PROCESSCHUNKS_ROOT_NAME),
            ))),
            current_processchunks_subdir_name_a: Mutex::new(StringProperty::new_with_key(Some(
                &format!("A.{}", CURRENT_PROCESSCHUNKS_SUBDIR_NAME),
            ))),
            current_processchunks_subdir_name_b: Mutex::new(StringProperty::new_with_key(Some(
                &format!("B.{}", CURRENT_PROCESSCHUNKS_SUBDIR_NAME),
            ))),
            etomo_created_version: Mutex::new(EtomoVersion::get_empty_instance(Some(
                "Version.Etomo.Created",
            ))),
            etomo_modified_version: Mutex::new(EtomoVersion::get_empty_instance(Some(
                "Version.Etomo.Modified",
            ))),
            image_file_meta_data: Mutex::new(image_file_meta_data),
            log_properties,
            manager,
            new_dataset,
            can_correct_image_filename_style,
        }
    }

    /// Java `getImageFilenameStyle`.
    pub fn get_image_filename_style(&self) -> ImageFilenameStyle {
        self.image_file_meta_data
            .lock()
            .unwrap()
            .get_image_filename_style()
    }

    /// Java `getImageOutputFormat`.
    pub fn get_image_output_format(&self) -> ImageOutputFormat {
        self.image_file_meta_data
            .lock()
            .unwrap()
            .get_image_output_format()
    }

    /// Java package-private `etomoModifiedVersionLt`.
    pub fn etomo_modified_version_lt(&self, version: Option<&str>) -> bool {
        use crate::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
        self.etomo_modified_version
            .lock()
            .unwrap()
            .lt_string(version)
    }

    /// Java `getOrigRawImageStackExtension`.  Just returns the default.  MetaData that
    /// refers to a raw image input file should override this and return the extension of
    /// the file originally chosen by the user.  Should never return null.
    pub fn get_orig_raw_image_stack_extension(&self) -> &'static Extension {
        self.image_file_meta_data
            .lock()
            .unwrap()
            .get_default_raw_image_stack_extension()
    }

    /// Java `getRawImageStackExtension`.  Just returns the default.  Override this if
    /// there is an actual raw image stack involved.  Should never return null.
    pub fn get_raw_image_stack_extension(&self) -> &'static Extension {
        self.image_file_meta_data
            .lock()
            .unwrap()
            .get_default_raw_image_stack_extension()
    }

    /// Java `setRawImageStackExtension`.  No effect.  Override this for interfaces with
    /// an input image file.
    pub fn set_raw_image_stack_extension(&self, input: Option<&Extension>) {
        let _ = input;
    }

    /// Java `setOrigRawImageStackExtension`.  No effect.  Override this for interfaces
    /// with an input image file.
    pub fn set_orig_raw_image_stack_extension(&self, input: Option<&Extension>) {
        let _ = input;
    }

    /// Java `isOldImageFilenameStyle`.
    pub fn is_old_image_filename_style(&self) -> bool {
        self.image_file_meta_data
            .lock()
            .unwrap()
            .is_old_image_filename_style()
    }

    /// Java `getRevisionNumber`.
    pub fn get_revision_number(&self) -> EtomoVersion {
        self.revision_number.lock().unwrap().clone()
    }

    /// Java `getAxisType`.
    pub fn get_axis_type(&self) -> AxisType {
        *self.axis_type.lock().unwrap()
    }

    /// Java `getInvalidReason`.
    pub fn get_invalid_reason(&self) -> String {
        self.invalid_reason.lock().unwrap().clone()
    }

    /// Java `getFileExtension`.
    pub fn get_file_extension(&self) -> Option<String> {
        self.file_extension.lock().unwrap().clone()
    }

    /// Java package-private `isEtomoCreatedVersionSet`.  Note that the source returns
    /// `etomoCreatedVersion.isNull()`, which is the opposite of what the name says.
    pub fn is_etomo_created_version_set(&self) -> bool {
        self.etomo_created_version.lock().unwrap().is_null()
    }

    /// Java package-private `wasImageFilenameStyleLoaded`.
    pub fn was_image_filename_style_loaded(&self) -> bool {
        self.image_file_meta_data
            .lock()
            .unwrap()
            .was_image_filename_style_loaded()
    }

    /// Java `getCurrentProcesschunksRootName`.
    pub fn get_current_processchunks_root_name(&self, axis_id: Option<AxisID>) -> Option<String> {
        if axis_id == Some(AxisID::Second) {
            return self
                .current_processchunks_root_name_b
                .lock()
                .unwrap()
                .to_string_option();
        }
        self.current_processchunks_root_name_a
            .lock()
            .unwrap()
            .to_string_option()
    }

    /// Java `getCurrentProcesschunksSubdirName`.
    pub fn get_current_processchunks_subdir_name(&self, axis_id: Option<AxisID>) -> Option<String> {
        if axis_id == Some(AxisID::Second) {
            return self
                .current_processchunks_subdir_name_b
                .lock()
                .unwrap()
                .to_string_option();
        }
        self.current_processchunks_subdir_name_a
            .lock()
            .unwrap()
            .to_string_option()
    }

    /// Java `isCurrentProcesschunksSubdirNameSet`.
    pub fn is_current_processchunks_subdir_name_set(&self, axis_id: Option<AxisID>) -> bool {
        use super::const_string_property::ConstStringProperty;
        if axis_id == Some(AxisID::Second) {
            return !self
                .current_processchunks_subdir_name_b
                .lock()
                .unwrap()
                .is_empty();
        }
        !self
            .current_processchunks_subdir_name_a
            .lock()
            .unwrap()
            .is_empty()
    }

    /// Java `setCurrentProcesschunksRootName`.
    pub fn set_current_processchunks_root_name(
        &self,
        axis_id: Option<AxisID>,
        input: Option<&str>,
    ) {
        if axis_id == Some(AxisID::Second) {
            self.current_processchunks_root_name_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.current_processchunks_root_name_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `setCurrentProcesschunksSubdirName`.
    pub fn set_current_processchunks_subdir_name(
        &self,
        axis_id: Option<AxisID>,
        input: Option<&str>,
    ) {
        if axis_id == Some(AxisID::Second) {
            self.current_processchunks_subdir_name_b
                .lock()
                .unwrap()
                .set(input);
        } else {
            self.current_processchunks_subdir_name_a
                .lock()
                .unwrap()
                .set(input);
        }
    }

    /// Java `resetCurrentProcesschunksRootName`.
    pub fn reset_current_processchunks_root_name(&self, axis_id: Option<AxisID>) {
        if axis_id == Some(AxisID::Second) {
            self.current_processchunks_root_name_b
                .lock()
                .unwrap()
                .reset();
        } else {
            self.current_processchunks_root_name_a
                .lock()
                .unwrap()
                .reset();
        }
    }

    /// Java `resetCurrentProcesschunksSubdirName`.
    pub fn reset_current_processchunks_subdir_name(&self, axis_id: Option<AxisID>) {
        if axis_id == Some(AxisID::Second) {
            self.current_processchunks_subdir_name_b
                .lock()
                .unwrap()
                .reset();
        } else {
            self.current_processchunks_subdir_name_a
                .lock()
                .unwrap()
                .reset();
        }
    }

    /// Java `equals(BaseMetaData)`.
    pub fn equals(&self, input: &BaseMetaDataBase) -> bool {
        if !self
            .current_processchunks_root_name_a
            .lock()
            .unwrap()
            .equals_string_property(&input.current_processchunks_root_name_a.lock().unwrap())
        {
            return false;
        }
        if !self
            .current_processchunks_root_name_b
            .lock()
            .unwrap()
            .equals_string_property(&input.current_processchunks_root_name_b.lock().unwrap())
        {
            return false;
        }
        if *self.axis_type.lock().unwrap() != *input.axis_type.lock().unwrap() {
            return false;
        }
        true
    }

    /// Java `equals(Object)`.  The `instanceof BaseMetaData` test is the Rust parameter
    /// type; a caller that would pass anything else does not typecheck.
    pub fn equals_object(&self, object: Option<&BaseMetaDataBase>) -> bool {
        match object {
            None => false,
            Some(object) => self.equals(object),
        }
    }

    /// Java `getDatasetImageFilenameStyle`, package-private final.
    pub fn get_dataset_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        let mut image_filename_style: Option<ImageFilenameStyle> = None;
        let output = match self.manager {
            None => None,
            Some(manager) => manager.tomosetexts(),
        };
        if let Some(output) = output {
            image_filename_style = output.get_image_filename_style();
        }
        if image_filename_style.is_some() {
            return image_filename_style;
        }
        None
    }
}

/// Java `BaseMetaData`.
pub trait BaseMetaData: Storable {
    /// The fields Java's `BaseMetaData` declares.  Not a source member: it is how a Rust
    /// implementor exposes the superclass's field block, which Java reaches directly.
    fn base(&self) -> &BaseMetaDataBase;

    /// Java abstract `getMetaDataFileName`.
    fn get_meta_data_file_name(&self) -> Option<String>;

    /// Java abstract `getName`.
    fn get_name(&self) -> Option<String>;

    /// Java abstract `getDatasetName`.
    fn get_dataset_name(&self) -> Option<String>;

    /// Java abstract `isValid`.
    fn is_valid(&self) -> bool;

    /// Java abstract package-private `getGroupKey`.
    fn get_group_key(&self) -> Option<String>;

    /// Java package-private `createPrepend`.
    fn create_prepend(&self, prepend: &str) -> Option<String> {
        if prepend.is_empty() {
            return self.get_group_key();
        }
        Some(format!(
            "{}.{}",
            prepend,
            self.get_group_key().unwrap_or("null".to_string())
        ))
    }

    /// Java package-private `checkImageFilenameStyleLoaded`.
    fn check_image_filename_style_loaded(&self, parent_prepend: &str) {
        if !self.base().was_image_filename_style_loaded() {
            self.repair_image_filename_style(parent_prepend);
        }
    }

    /// Java package-private `getImageFilenameStyleKey`.
    fn get_image_filename_style_key(&self, parent_prepend: &str) -> Option<String> {
        let prepend = self.create_prepend(parent_prepend);
        self.base()
            .image_file_meta_data
            .lock()
            .unwrap()
            .get_image_filename_style_key(prepend.as_deref())
    }

    /// Java package-private `correctImageFilenameStyle`.
    fn correct_image_filename_style(
        &self,
        parent_prepend: &str,
        input: Option<ImageFilenameStyle>,
    ) -> bool {
        self.base()
            .image_file_meta_data
            .lock()
            .unwrap()
            .correct_image_filename_style(Some(parent_prepend), input)
    }

    /// Java package-private `repairImageFilenameStyle`.  Set an empty image filename
    /// style to old style.
    fn repair_image_filename_style(&self, parent_prepend: &str) {
        if self.base().was_image_filename_style_loaded() {
            return;
        }
        let repair_type = "old image file name style";
        let image_filename_style = ImageFilenameStyle::Old;
        let key = self
            .get_image_filename_style_key(parent_prepend)
            .unwrap_or("null".to_string());
        // The source's `if (imageFilenameStyle != null)` can never be false: the local
        // was just assigned `ImageFilenameStyle.OLD`.
        eprintln!(
            "\nINFO: Attempting to repair the {} property, which is missing from the\ndataset file.  Using the {} and\nsetting {} to {}.\n",
            key, repair_type, key, image_filename_style
        );
        if self.correct_image_filename_style(parent_prepend, Some(image_filename_style)) {
            return;
        }
        // The source's `repairType != null ? ... : null` is likewise always the first
        // arm.
        eprintln!(
            "\nERROR: Unable to repair the {} property, which is missing from the\ndataset file.  Unable to use the {}.",
            key, repair_type
        );
    }
}

/// Java `toString`.  The source appends `super.toString()`, which is
/// `Object.toString()` - `getClass().getName() + "@" +
/// Integer.toHexString(hashCode())` - whose identity hash is not reproducible; this
/// prints the four characters `null` in its place, as Java string concatenation does for
/// an absent value.
impl std::fmt::Display for BaseMetaDataBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[fileExtension:{},revisionNumber:{},\naxisType:{},invalidReason:{}super:{}]",
            self.file_extension
                .lock()
                .unwrap()
                .clone()
                .unwrap_or("null".to_string()),
            self.revision_number.lock().unwrap(),
            self.axis_type.lock().unwrap(),
            self.invalid_reason.lock().unwrap(),
            "null"
        )
    }
}

/// Java `store(Properties)`, `store(Properties, String)`, `load(Properties)` and
/// `load(Properties, String)`, declared by `Storable` and implemented by
/// `BaseMetaData`.  They live on `BaseMetaDataBase` because the trait's implementors
/// inherit them; `store`/`load` with a prepend need `getGroupKey`, which is abstract, so
/// those two take the prepend the trait's `create_prepend` produced.
impl BaseMetaDataBase {
    /// Java `store(Properties, String)`, with `prepend` already run through
    /// `createPrepend`.
    pub fn store_with_created_prepend(
        &self,
        props: &mut BTreeMap<String, String>,
        prepend: Option<&str>,
    ) {
        if self.new_dataset {
            self.etomo_created_version
                .lock()
                .unwrap()
                .set(Some(imod_version::CURRENT_VERSION));
        }
        self.etomo_modified_version
            .lock()
            .unwrap()
            .set(Some(imod_version::CURRENT_VERSION));
        let prepend = prepend.unwrap_or("");
        self.etomo_created_version
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend);
        self.etomo_modified_version
            .lock()
            .unwrap()
            .store_with_prepend(props, prepend);
        self.image_file_meta_data
            .lock()
            .unwrap()
            .store(props, Some(prepend));
        self.current_processchunks_root_name_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(prepend));
        self.current_processchunks_root_name_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(prepend));
        self.current_processchunks_subdir_name_a
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(prepend));
        self.current_processchunks_subdir_name_b
            .lock()
            .unwrap()
            .store_with_prepend(Some(props), Some(prepend));
        if let Some(log_properties) = self.log_properties {
            let _ = log_properties;
            // `logProperties.store(props, prepend)`; `LogProperties.store` takes
            // `&mut self` in the Rust trait because Java's implementor
            // (`etomo/ui/swing/LogWindow.java`) mutates while storing, and the field is
            // a shared reference, so this call is made by the implementor.
        }
    }

    /// Java `load(Properties, String)`, with `prepend` already run through
    /// `createPrepend`.  Returns whether the caller should run
    /// `checkImageFilenameStyleLoaded`, which is the source's trailing
    /// `canCorrectImageFilenameStyle` test.
    pub fn load_with_created_prepend(
        &self,
        props: &BTreeMap<String, String>,
        prepend: Option<&str>,
    ) -> bool {
        // reset
        self.etomo_modified_version.lock().unwrap().reset();
        self.current_processchunks_root_name_a
            .lock()
            .unwrap()
            .reset();
        self.current_processchunks_root_name_b
            .lock()
            .unwrap()
            .reset();
        self.current_processchunks_subdir_name_a
            .lock()
            .unwrap()
            .reset();
        self.current_processchunks_subdir_name_b
            .lock()
            .unwrap()
            .reset();
        // load
        let prepend = prepend.unwrap_or("");
        self.etomo_created_version
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.etomo_modified_version
            .lock()
            .unwrap()
            .load_with_prepend(props, prepend);
        self.image_file_meta_data
            .lock()
            .unwrap()
            .load(self.manager, props, Some(prepend));
        // `StringProperty.load` removes a backward-compatible key from `props`, so the
        // source's `Properties` is mutable here; this translation's `load` takes a
        // read-only map, and none of these four properties declares such a key.
        let mut props_copy = props.clone();
        self.current_processchunks_root_name_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(prepend));
        self.current_processchunks_root_name_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(prepend));
        self.current_processchunks_subdir_name_a
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(prepend));
        self.current_processchunks_subdir_name_b
            .lock()
            .unwrap()
            .load_with_prepend(Some(&mut props_copy), Some(prepend));
        if let Some(log_properties) = self.log_properties {
            let _ = log_properties;
            // `logProperties.load(props, prepend)`; see `store_with_created_prepend`.
        }
        // If canCorrectImageFilenameStyle is false, then correcting imageFilenameStyle
        // may require the child load function to be run beforehand.  It is the child
        // class's responsibility to correct imageFilenameStyle if
        // canCorrectImageFilenameStyle is false.
        self.can_correct_image_filename_style
    }
}
