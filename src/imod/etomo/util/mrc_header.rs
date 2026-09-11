//! `IMOD/Etomo/src/etomo/util/MRCHeader.java`.
//!
//! Reads and holds the header of an MRC file.
//!
//! **Frontier.**  `read(BaseManager)` runs `$IMOD_DIR/bin/header` through an
//! `etomo.process.SystemProgram` and reports failures through
//! `etomo.process.ProcessMessages` and `etomo.ui.swing.UIHarness`; none of those units
//! has a module, and neither does `etomo.ApplicationManager` (which supplies the bin
//! path) or `etomo.type.ImageOutputFormat` (the `imageOutputFormat` field's type).  The
//! members that need them carry `TODO(unit)` markers below.  Everything that does not -
//! the n'ton table, the constants, the fields, `makeKey`, `pixelEquals`, every getter
//! that does not return an `ImageOutputFormat`, `parseCommentData`, `paramString`,
//! `toString`, and the whole nested `CommentData` class - is translated.
//!
//! `parseCommentData` is the part of the unit that is pure string handling, and it is
//! what the comment-section fields (`binning`, `imageRotation`, `bidir`, `dosym`,
//! `feiPixelSize`) are actually read from; it is verified against a real JVM in
//! `tests/etomo_mrc_header_probe.rs`.
//!
//! The n'ton table `instances` maps an absolute path to one shared, mutable `MRCHeader`,
//! so its values are `Rc<RefCell<MRCHeader>>` - what a Java reference is here.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::util::file_modified_flag::FileModifiedFlag;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_string_split};
use regex::Regex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{LazyLock, Mutex};

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`, read once when the
/// class is initialised.
static DEBUG: LazyLock<bool> = LazyLock::new(|| {
    crate::imod::etomo::etomo_director::ARGUMENTS
        .lock()
        .unwrap()
        .is_debug()
});

// n'ton member variables

/// Java `instances`, a `Hashtable` keyed by absolute path.  `Hashtable` is synchronized,
/// which the `Mutex` supplies; the entries are shared references, which the `Rc` supplies.
/// The `Mutex` cannot hold a non-`Send` `Rc`, so the table is thread-local, which is
/// where the single-threaded source keeps it in practice.
type Instances = HashMap<String, Rc<RefCell<MRCHeader>>>;
thread_local! {
    static INSTANCES: RefCell<Instances> = RefCell::new(HashMap::new());
}
/// Java's `synchronized createInstance` lock, which is on the class, not on `instances`.
static CREATE_INSTANCE_LOCK: Mutex<()> = Mutex::new(());

/// Java `COMMENT_DIVIDER`.
const COMMENT_DIVIDER: &str = "=";

/// Java `SIZE_HEADER`.
pub const SIZE_HEADER: &str = "Number of columns, rows, sections";
/// Java `N_SECTIONS_INDEX`.
pub const N_SECTIONS_INDEX: i32 = 8;
/// Java `N_ROWS_INDEX`.
const N_ROWS_INDEX: i32 = N_SECTIONS_INDEX - 1;
/// Java `N_COLUMNS_INDEX`.
const N_COLUMNS_INDEX: i32 = N_SECTIONS_INDEX - 2;

// This is information about the header process. Will only be used by
// MrcHeader if it is run with the "-brief" option.
/// Java `SIZE_HEADER_BRIEF`.
pub const SIZE_HEADER_BRIEF: &str = "Dimensions:";
/// Java `N_SECTIONS_INDEX_BRIEF`.
pub const N_SECTIONS_INDEX_BRIEF: i32 = 3;
/// Java `FLOATING_POINT_MODE`.
pub const FLOATING_POINT_MODE: i32 = 2;

/// Java `MRCHeader`.
pub struct MRCHeader {
    // member variables to prevent unnecessary reads
    /// Java field `modifiedFlag`.
    modified_flag: FileModifiedFlag,

    // other member variables
    /// Java field `filename`.
    filename: Option<String>,
    /// Java field `nColumns`, initialised to -1.
    n_columns: i32,
    /// Java field `nRows`, initialised to -1.
    n_rows: i32,
    /// Java field `nSections`, initialised to -1.
    n_sections: i32,
    /// Java field `mode`, initialised to -1.
    mode: i32,
    /// Java field `xPixelSize`.
    x_pixel_size: EtomoNumber,
    /// Java field `yPixelSize`.
    y_pixel_size: EtomoNumber,
    /// Java field `zPixelSize`.
    z_pixel_size: EtomoNumber,
    // comment data
    /// Java field `commentDataVector`.  Each `CommentData` constructor adds itself to it,
    /// and `parseCommentData` walks it; the elements are the same objects as the five
    /// named fields below, so they are shared handles.
    comment_data_vector: Vec<Rc<RefCell<CommentData>>>,
    // When adding a new comment data tag, make sure there is no conflict. Use order of
    // construction to resolve conflicts.
    /// Java field `binning`.
    binning: Rc<RefCell<CommentData>>,
    /// Java field `imageRotation`.
    image_rotation: Rc<RefCell<CommentData>>,
    /// Java field `bidir`.
    bidir: Rc<RefCell<CommentData>>,
    /// Java field `doseSym`.
    dose_sym: Rc<RefCell<CommentData>>,
    /// Java field `feiPixelSize`.
    fei_pixel_size: Rc<RefCell<CommentData>>,

    /// Java field `xPixelSpacing`, initialised to `Double.NaN`.
    x_pixel_spacing: f64,
    /// Java field `yPixelSpacing`, initialised to `Double.NaN`.
    y_pixel_spacing: f64,
    /// Java field `zPixelSpacing`, initialised to `Double.NaN`.
    z_pixel_spacing: f64,
    /// Java field `axisID`.
    axis_id: Option<AxisID>,
    /// Java field `fileLocation`.
    file_location: Option<String>,

    /// Java field `debug`, initialised to false.
    debug: bool,
    /// Java field `imageOutputFormat`, initialised to `ImageOutputFormat.MRC`.  `read`
    /// sets it to `ImageOutputFormat.HDF` when the header says "This is an HDF file";
    /// `read` is the `SystemProgram` boundary and is marked below, so the field keeps
    /// its initial value here.
    image_output_format: ImageOutputFormat,
}

impl MRCHeader {
    /// Java `MRCHeader(String, File, AxisID)`, the private constructor.
    fn new(file_location: Option<&str>, file: &str, axis_id: Option<AxisID>) -> MRCHeader {
        let mut comment_data_vector: Vec<Rc<RefCell<CommentData>>> = Vec::new();
        let binning = CommentData::new_with_default(
            Some(&mut comment_data_vector),
            "binning",
            Type::Double,
            Some(1),
        );
        let image_rotation = CommentData::new_with_alt_tag(
            Some(&mut comment_data_vector),
            "Tilt axis angle",
            Some("Tilt axis rotation angle"),
            Type::Double,
        );
        let bidir = CommentData::new_with_name_ends_with_tag(
            Some(&mut comment_data_vector),
            "bidir",
            Type::Double,
            true,
        );
        let dose_sym = CommentData::new_with_name_ends_with_tag(
            Some(&mut comment_data_vector),
            "dosym",
            Type::Double,
            true,
        );
        let fei_pixel_size = CommentData::new_with_type(
            Some(&mut comment_data_vector),
            "Pixel size in nanometers",
            Type::Double,
        );
        MRCHeader {
            modified_flag: FileModifiedFlag::new(file),
            filename: Some(java_io_file_get_absolute_path(file)),
            n_columns: -1,
            n_rows: -1,
            n_sections: -1,
            mode: -1,
            x_pixel_size: EtomoNumber::new_with_type(Some(Type::Double)),
            y_pixel_size: EtomoNumber::new_with_type(Some(Type::Double)),
            z_pixel_size: EtomoNumber::new_with_type(Some(Type::Double)),
            comment_data_vector,
            binning,
            image_rotation,
            bidir,
            dose_sym,
            fei_pixel_size,
            x_pixel_spacing: f64::NAN,
            y_pixel_spacing: f64::NAN,
            z_pixel_spacing: f64::NAN,
            axis_id,
            file_location: file_location.map(|file_location| file_location.to_string()),
            debug: false,
            image_output_format: ImageOutputFormat::Mrc,
        }
    }

    /// Java `getInstance(BaseManager, AxisID, String)` (deprecated 5/9/2019).
    pub fn get_instance_from_manager(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_ext: Option<&str>,
    ) -> Option<Rc<RefCell<MRCHeader>>> {
        MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            Some(&java_io_file_get_absolute_path(
                &crate::imod::etomo::util::dataset_files::get_dataset_file(
                    manager, axis_id, file_ext,
                )
                .to_string_lossy(),
            )),
            axis_id,
        )
    }

    /// Java `getInstanceFromFileName`.
    pub fn get_instance_from_file_name(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_name: Option<&str>,
    ) -> Option<Rc<RefCell<MRCHeader>>> {
        MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            Some(&java_io_file_get_absolute_path(
                &crate::imod::etomo::util::dataset_files::get_dataset_file_from_file_name(
                    manager, axis_id, file_name,
                )
                .to_string_lossy(),
            )),
            axis_id,
        )
    }

    /// Java `getInstance(BaseManager, AxisID, FileType)`.
    pub fn get_instance_from_file_type(
        manager: &'static dyn BaseManager,
        axis_id: Option<AxisID>,
        file_type: &std::sync::Arc<crate::imod::etomo::r#type::file_type::FileType>,
    ) -> Option<Rc<RefCell<MRCHeader>>> {
        let key_file = crate::imod::etomo::util::utilities::get_file(
            manager.get_property_user_dir().as_deref().unwrap_or("null"),
            file_type.get_file_name(Some(manager), axis_id).as_deref(),
        );
        let key = MRCHeader::make_key(&key_file.to_string_lossy());
        let mrc_header = INSTANCES.with(|instances| instances.borrow().get(&key).cloned());
        if mrc_header.is_none() {
            return Some(MRCHeader::create_instance(
                manager.get_property_user_dir().as_deref(),
                &key,
                &key_file.to_string_lossy(),
                axis_id,
            ));
        }
        mrc_header
    }

    /// Java `getInstance(String, String, AxisID)`.  Function to get an instance of the
    /// class.
    pub fn get_instance_in_dir(
        file_location: Option<&str>,
        filename: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Option<Rc<RefCell<MRCHeader>>> {
        let key_file = crate::imod::etomo::util::utilities::get_file(
            file_location.unwrap_or("null"),
            filename,
        );
        let key_file = key_file.to_string_lossy().to_string();
        let key = MRCHeader::make_key(&key_file);
        let mrc_header = INSTANCES.with(|instances| instances.borrow().get(&key).cloned());
        if mrc_header.is_none() {
            return Some(MRCHeader::create_instance(
                file_location,
                &key,
                &key_file,
                axis_id,
            ));
        }
        mrc_header
    }

    /// Java `getInstance(String, AxisID)`.  Returns a new or saved instance.
    pub fn get_instance(
        file_path: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Option<Rc<RefCell<MRCHeader>>> {
        let file_path = match file_path {
            None => return None,
            Some(file_path) => file_path,
        };
        let key_file = file_path.to_string();
        let key = MRCHeader::make_key(&key_file);
        let mrc_header = INSTANCES.with(|instances| instances.borrow().get(&key).cloned());
        if mrc_header.is_none() {
            return Some(MRCHeader::create_instance(
                crate::imod::etomo::util::utilities::java_io_file_get_parent(&key_file).as_deref(),
                &key,
                &key_file,
                axis_id,
            ));
        }
        mrc_header
    }

    /// Java private `synchronized createInstance`.  Function to create and save an
    /// instance of the class.  Just returns the instance if it already exists.
    fn create_instance(
        file_location: Option<&str>,
        key: &str,
        file: &str,
        axis_id: Option<AxisID>,
    ) -> Rc<RefCell<MRCHeader>> {
        let _guard = CREATE_INSTANCE_LOCK.lock().unwrap();
        let mrc_header = INSTANCES.with(|instances| instances.borrow().get(key).cloned());
        if let Some(mrc_header) = mrc_header {
            return mrc_header;
        }
        let mrc_header = Rc::new(RefCell::new(MRCHeader::new(file_location, file, axis_id)));
        INSTANCES.with(|instances| {
            instances
                .borrow_mut()
                .insert(key.to_string(), Rc::clone(&mrc_header))
        });
        mrc_header
    }

    /// Java private `makeKey`.  Make a unique key from a file.
    fn make_key(file: &str) -> String {
        java_io_file_get_absolute_path(file)
    }

    // other functions

    // TODO(unit): needs etomo/process/SystemProgram.java, etomo/process/ProcessMessages.java,
    // etomo/ApplicationManager.java, etomo/BaseManager.java and
    // etomo/ui/swing/UIHarness.java - Java `synchronized read(BaseManager)` runs
    // `ApplicationManager.getIMODBinPath() + "header"` through a `SystemProgram`, reads
    // its exit value, its `ProcessMessages` and its stdout/stderr, and then parses that
    // output into the size, mode and pixel-spacing fields.  None of those five units has
    // a module.  `parseCommentData`, which `read` calls once per output line, is
    // translated below and is the part of the parse that does not need them.

    // TODO(unit): needs etomo/ui/swing/UIHarness.java and etomo/BaseManager.java - Java
    // private `parsePixelSpacing(BaseManager, EtomoNumber, String, boolean)` pops the
    // failure up through `UIHarness.INSTANCE.openMessageDialog`.

    /// Java private `pixelEquals`.  Note that the source tests `yPixelSize` twice and
    /// never tests `zPixelSize`.
    fn pixel_equals(&self, size: f64) -> bool {
        self.x_pixel_size.equals_double(size)
            && self.y_pixel_size.equals_double(size)
            && self.y_pixel_size.equals_double(size)
    }

    /// Java `getNColumns`.
    pub fn get_n_columns(&self) -> i32 {
        self.n_columns
    }

    /// Java `getNRows`.
    pub fn get_n_rows(&self) -> i32 {
        self.n_rows
    }

    /// Java `getTwodir`.
    pub fn get_twodir(&self) -> ConstEtomoNumber {
        self.bidir.borrow().get()
    }

    /// Java `getDoseSym`.
    pub fn get_dose_sym(&self) -> ConstEtomoNumber {
        self.dose_sym.borrow().get()
    }

    /// Java `getNSections`.
    pub fn get_n_sections(&self) -> i32 {
        self.n_sections
    }

    /// Java `getMode`.  Return the mode (type) of data in the file.
    pub fn get_mode(&self) -> i32 {
        self.mode
    }

    /// Java `getImageRotation`.
    ///
    /// Return the image rotation in degrees if present in the header.  If the
    /// header has not been read or the image rotation is not available
    /// imageRotation will be null.
    pub fn get_image_rotation(&self) -> ConstEtomoNumber {
        self.image_rotation.borrow().get()
    }

    /// Java `getXPixelSize`.
    pub fn get_x_pixel_size(&self) -> &ConstEtomoNumber {
        &self.x_pixel_size.base
    }

    /// Java `getYPixelSize`.
    pub fn get_y_pixel_size(&self) -> &ConstEtomoNumber {
        &self.y_pixel_size.base
    }

    /// Java `getZPixelSize`.
    pub fn get_z_pixel_size(&self) -> &ConstEtomoNumber {
        &self.z_pixel_size.base
    }

    /// Java `getImageOutputFormat`.
    pub fn get_image_output_format(&self) -> ImageOutputFormat {
        self.image_output_format
    }

    /// Java `getXPixelSpacing`.
    pub fn get_x_pixel_spacing(&self) -> f64 {
        self.x_pixel_spacing
    }

    /// Java `getBinning`.  Return the binning found in the header.
    pub fn get_binning(&self) -> String {
        self.binning.borrow().to_string()
    }

    /// Java private `parseCommentData`.  Parse all recognized comment data.
    ///
    /// Comment data varies in different headers. Comment format is variable and may
    /// change in the future. Some data has more then one identifying tag. Not all
    /// comment data needs to be saved.
    ///
    /// Assumptions:
    /// - An equals sign is always used between the name and the value.
    /// - The value is always present.
    /// - A comma and/or whitespace is used between the value and the next name/value
    ///   pair.
    /// - There are no whitespace or commas within comment data values.
    /// - Comment data is numeric.
    /// - The numeric type of comment data does not vary.
    ///
    /// Examples of comment data:
    /// `Tilt axis angle = -11.5, binning = 2 spot = 2 camera = 2`
    /// `Tilt axis rotation angle = -24.9 (Corrected sign)`
    /// `Pixel size in nanometers = 1.016`
    pub fn parse_comment_data(&mut self, line: &str) {
        if !line.contains(COMMENT_DIVIDER) {
            return;
        }
        let mut split_array: Vec<Option<String>> = java_lang_string_split(
            crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(line),
            &Regex::new(&format!("\\s*{}\\s*", COMMENT_DIVIDER)).unwrap(),
        )
        .into_iter()
        .map(Some)
        .collect();
        if split_array.is_empty() {
            return;
        }
        // A splitArray will look like this:
        // Tilt axis angle
        // -11.5, binning
        // 2 spot
        // 2 camera
        // 2
        //
        // Another splitArray:
        // Tilt axis rotation angle
        // -24.9 (Corrected sign)
        //
        // Look for known tags in the elements containing data names
        let num_tags = self.comment_data_vector.len();
        for i in 0..split_array.len().saturating_sub(1) {
            for comment_data_index in 0..num_tags {
                if split_array[i].is_none() {
                    continue;
                }
                if comment_data_index == 0 {
                    split_array[i] = Some(
                        crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim(
                            split_array[i].as_ref().unwrap(),
                        )
                        .to_string(),
                    );
                }
                if split_array[i].as_ref().unwrap().is_empty() {
                    continue;
                }
                let comment_data = Rc::clone(&self.comment_data_vector[comment_data_index]);
                // Look for comment data. Value is in the next element of splitArray.
                if !comment_data.borrow().find(split_array[i].as_deref())
                    || split_array[i + 1].is_none()
                {
                    continue;
                }
                let value_split = java_lang_string_split(
                    split_array[i + 1].as_ref().unwrap(),
                    &Regex::new("\\s*,\\s*|\\s+").unwrap(),
                );
                if value_split.is_empty() {
                    continue;
                }
                // The value is at the beginning of the next element.
                comment_data.borrow_mut().set(Some(&value_split[0]));
            }
        }
    }

    /// Java package-private `paramString`.
    pub fn param_string(&self) -> String {
        if self.get_dose_sym().is() {
            return format!(
                ",\nfilename={},nColumns={},nRows={},\nnSections={},mode={},\nxPixelSize={},yPixelSize={},\nzPixelSize={},xPixelSpacing={},\nyPixelSpacing={},zPixelSpacing={},\nimageRotation={},binning={},\naxisID={},dosesym ={},feiPixelSize={}",
                self.filename.as_deref().unwrap_or("null"),
                self.n_columns,
                self.n_rows,
                self.n_sections,
                self.mode,
                self.x_pixel_size,
                self.y_pixel_size,
                self.z_pixel_size,
                crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                    self.x_pixel_spacing
                ),
                crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                    self.y_pixel_spacing
                ),
                crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                    self.z_pixel_spacing
                ),
                self.image_rotation.borrow(),
                self.binning.borrow(),
                match self.axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                },
                self.dose_sym.borrow(),
                self.fei_pixel_size.borrow()
            );
        }
        format!(
            ",\nfilename={},nColumns={},nRows={},\nnSections={},mode={},\nxPixelSize={},yPixelSize={},\nzPixelSize={},xPixelSpacing={},\nyPixelSpacing={},zPixelSpacing={},\nimageRotation={},binning={},\naxisID={},bidir ={},feiPixelSize={}",
            self.filename.as_deref().unwrap_or("null"),
            self.n_columns,
            self.n_rows,
            self.n_sections,
            self.mode,
            self.x_pixel_size,
            self.y_pixel_size,
            self.z_pixel_size,
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                self.x_pixel_spacing
            ),
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                self.y_pixel_spacing
            ),
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                self.z_pixel_spacing
            ),
            self.image_rotation.borrow(),
            self.binning.borrow(),
            match self.axis_id {
                None => "null".to_string(),
                Some(axis_id) => axis_id.to_string(),
            },
            self.bidir.borrow(),
            self.fei_pixel_size.borrow()
        )
    }
}

/// Java `toString`.  `getClass().getName() + "[" + super.toString() + paramString() +
/// "]"`, where `Object.toString()` is `etomo.util.MRCHeader@<identityHashCode>` - a JVM
/// fact, not a program fact, so the identity hash is not reproducible and the address
/// part is written as the source's own class name with an empty hash.
impl std::fmt::Display for MRCHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "etomo.util.MRCHeader[etomo.util.MRCHeader@{}]",
            self.param_string()
        )
    }
}

/// Java's nested `private final class CommentData`.
///
/// Comment Data can recognize tags and store the value.  Each instance is added to a
/// vector, if available.
pub struct CommentData {
    /// Java field `tag` - required.
    tag: String,
    /// Java field `altTag` - optional.
    alt_tag: Option<String>,
    /// Java field `value`.
    value: EtomoNumber,
    /// Java field `nameEndsWithTag`.
    name_ends_with_tag: bool,
}

impl CommentData {
    /// Java `CommentData(Vector<CommentData>, String)`.  Creates an integer comment data
    /// instance.  `commentDataVector` is optional - the instance is added to this vector.
    pub fn new(
        comment_data_vector: Option<&mut Vec<Rc<RefCell<CommentData>>>>,
        tag: &str,
    ) -> Rc<RefCell<CommentData>> {
        let comment_data = Rc::new(RefCell::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value: EtomoNumber::new(),
            name_ends_with_tag: false,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Rc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, EtomoNumber.Type)`.
    pub fn new_with_type(
        comment_data_vector: Option<&mut Vec<Rc<RefCell<CommentData>>>>,
        tag: &str,
        r#type: Type,
    ) -> Rc<RefCell<CommentData>> {
        let comment_data = Rc::new(RefCell::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value: EtomoNumber::new_with_type(Some(r#type)),
            name_ends_with_tag: false,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Rc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, EtomoNumber.Type, Integer)`.
    pub fn new_with_default(
        comment_data_vector: Option<&mut Vec<Rc<RefCell<CommentData>>>>,
        tag: &str,
        r#type: Type,
        default_value: Option<i32>,
    ) -> Rc<RefCell<CommentData>> {
        let mut value = EtomoNumber::new_with_type(Some(r#type));
        if let Some(default_value) = default_value {
            value.set_display_value_int(default_value);
        }
        let comment_data = Rc::new(RefCell::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value,
            name_ends_with_tag: false,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Rc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, EtomoNumber.Type, boolean)`.
    pub fn new_with_name_ends_with_tag(
        comment_data_vector: Option<&mut Vec<Rc<RefCell<CommentData>>>>,
        tag: &str,
        r#type: Type,
        name_ends_with_tag: bool,
    ) -> Rc<RefCell<CommentData>> {
        let comment_data = Rc::new(RefCell::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value: EtomoNumber::new_with_type(Some(r#type)),
            name_ends_with_tag,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Rc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, String, EtomoNumber.Type)`.
    pub fn new_with_alt_tag(
        comment_data_vector: Option<&mut Vec<Rc<RefCell<CommentData>>>>,
        tag: &str,
        alt_tag: Option<&str>,
        r#type: Type,
    ) -> Rc<RefCell<CommentData>> {
        let comment_data = Rc::new(RefCell::new(CommentData {
            tag: tag.to_string(),
            alt_tag: alt_tag.map(|alt_tag| alt_tag.to_string()),
            name_ends_with_tag: false,
            value: EtomoNumber::new_with_type(Some(r#type)),
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Rc::clone(&comment_data));
        }
        comment_data
    }

    /// Java private `find`.  Returns true if tag or altTag is in string, avoiding the
    /// image file name.
    pub fn find(&self, string: Option<&str>) -> bool {
        let string = match string {
            None => return false,
            Some(string) => string,
        };
        if string.contains("RO image file") {
            return false;
        }
        if !self.name_ends_with_tag {
            if string.contains(&self.tag) {
                return true;
            }
            if let Some(alt_tag) = &self.alt_tag {
                if string.contains(alt_tag.as_str()) {
                    return true;
                }
            }
        }
        if string.ends_with(&self.tag) {
            return true;
        }
        if let Some(alt_tag) = &self.alt_tag {
            if string.ends_with(alt_tag.as_str()) {
                return true;
            }
        }
        false
    }

    /// Java private `set`.  Set comment data to value.
    pub fn set(&mut self, value: Option<&str>) {
        self.value.set_string(value);
    }

    /// Java private `isNull`.
    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    /// Java private `get`.
    pub fn get(&self) -> ConstEtomoNumber {
        self.value.base.clone()
    }
}

/// Java `CommentData.toString`.  Returns the string version of nValue.
impl std::fmt::Display for CommentData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
