//! `IMOD/Etomo/src/etomo/util/MRCHeader.java`.
//!
//! Reads and holds the header of an MRC file.
//!
//! `read(BaseManager)` ([`MRCHeader::read_with_manager`]) runs `$IMOD_DIR/bin/header`
//! through an `etomo.process.SystemProgram` and parses its output with
//! [`MRCHeader::read`].
//!
//! `parseCommentData` is the part of the unit that is pure string handling, and it is
//! what the comment-section fields (`binning`, `imageRotation`, `bidir`, `dosym`,
//! `feiPixelSize`) are actually read from; it is verified against a real JVM in
//! `tests/etomo_mrc_header_probe.rs`.
//!
//! The n'ton table `instances` maps an absolute path to one shared, mutable `MRCHeader`,
//! so its values are `Arc<SharedMRCHeader>` - what a Java reference is here.  The
//! table is `static` in Java and eTomo reads headers from the event thread, process
//! monitor threads and process-series threads alike, so it is process-global here;
//! a thread-local table re-ran `header` for every new thread (seen as extra `header`
//! launches in the whole-tomogram positioning series).
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::util::file_modified_flag::FileModifiedFlag;
use crate::imod::etomo::util::utilities::{java_io_file_get_absolute_path, java_lang_string_split};
use regex::Regex;

/// The exceptions `read(BaseManager)` throws: `IOException`,
/// `InvalidParameterException` (etomo.comscript), and the unchecked
/// `NumberFormatException` of the size and mode parse.  Each carries its message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReadError {
    Io(String),
    InvalidParameter(String),
    NumberFormat(String),
}

/// `Throwable.getMessage()`.
impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(message)
            | ReadError::InvalidParameter(message)
            | ReadError::NumberFormat(message) => f.write_str(message),
        }
    }
}

/// A caller that handles every exception alike keeps only the message.
impl From<ReadError> for String {
    fn from(e: ReadError) -> String {
        e.to_string()
    }
}
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`, read once when the
/// class is initialised.
static DEBUG: LazyLock<bool> = LazyLock::new(|| {
    crate::imod::etomo::etomo_director::ARGUMENTS
        .lock()
        .unwrap()
        .is_debug()
});

// n'ton member variables

/// Java `instances`, a static `Hashtable` keyed by absolute path.  `Hashtable` is
/// synchronized, which the `Mutex` supplies; the entries are shared references, which
/// the `Arc` supplies.
type Instances = HashMap<String, Arc<SharedMRCHeader>>;
static INSTANCES: LazyLock<Mutex<Instances>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// A Java reference to an `MRCHeader`.  The object is shared between threads, and its
/// `synchronized` methods serialise on it; `borrow` and `borrow_mut` take the lock.
pub struct SharedMRCHeader(RwLock<MRCHeader>);

impl SharedMRCHeader {
    pub fn borrow(&self) -> RwLockReadGuard<'_, MRCHeader> {
        self.0
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn borrow_mut(&self) -> RwLockWriteGuard<'_, MRCHeader> {
        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
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
    comment_data_vector: Vec<Arc<Mutex<CommentData>>>,
    // When adding a new comment data tag, make sure there is no conflict. Use order of
    // construction to resolve conflicts.
    /// Java field `binning`.
    binning: Arc<Mutex<CommentData>>,
    /// Java field `imageRotation`.
    image_rotation: Arc<Mutex<CommentData>>,
    /// Java field `bidir`.
    bidir: Arc<Mutex<CommentData>>,
    /// Java field `doseSym`.
    dose_sym: Arc<Mutex<CommentData>>,
    /// Java field `feiPixelSize`.
    fei_pixel_size: Arc<Mutex<CommentData>>,

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
        let mut comment_data_vector: Vec<Arc<Mutex<CommentData>>> = Vec::new();
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
    ) -> Option<Arc<SharedMRCHeader>> {
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
    ) -> Option<Arc<SharedMRCHeader>> {
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
    ) -> Option<Arc<SharedMRCHeader>> {
        let key_file = crate::imod::etomo::util::utilities::get_file(
            manager.get_property_user_dir().as_deref().unwrap_or("null"),
            file_type.get_file_name(Some(manager), axis_id).as_deref(),
        );
        let key = MRCHeader::make_key(&key_file.to_string_lossy());
        let mrc_header = INSTANCES.lock().unwrap().get(&key).cloned();
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
    ) -> Option<Arc<SharedMRCHeader>> {
        let key_file = crate::imod::etomo::util::utilities::get_file(
            file_location.unwrap_or("null"),
            filename,
        );
        let key_file = key_file.to_string_lossy().to_string();
        let key = MRCHeader::make_key(&key_file);
        let mrc_header = INSTANCES.lock().unwrap().get(&key).cloned();
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
    ) -> Option<Arc<SharedMRCHeader>> {
        let file_path = match file_path {
            None => return None,
            Some(file_path) => file_path,
        };
        let key_file = file_path.to_string();
        let key = MRCHeader::make_key(&key_file);
        let mrc_header = INSTANCES.lock().unwrap().get(&key).cloned();
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
    ) -> Arc<SharedMRCHeader> {
        let _guard = CREATE_INSTANCE_LOCK.lock().unwrap();
        let mrc_header = INSTANCES.lock().unwrap().get(key).cloned();
        if let Some(mrc_header) = mrc_header {
            return mrc_header;
        }
        let mrc_header = Arc::new(SharedMRCHeader(RwLock::new(MRCHeader::new(
            file_location,
            file,
            axis_id,
        ))));
        INSTANCES
            .lock()
            .unwrap()
            .insert(key.to_string(), Arc::clone(&mrc_header));
        mrc_header
    }

    /// Java private `makeKey`.  Make a unique key from a file.
    fn make_key(file: &str) -> String {
        java_io_file_get_absolute_path(file)
    }

    // other functions

    /// The output-parsing half of Java `synchronized read(BaseManager)`
    /// (`MRCHeader.java:458-548`), from "boolean pixelsParsed = false" to the
    /// end: parses the lines of `header`'s standard output and then handles
    /// the FEI pixel size.  [`MRCHeader::read_with_manager`] runs the program
    /// and calls this; the `Err` is the message of the `IOException` or
    /// `NumberFormatException` the source throws.
    pub fn parse_std_output(
        &mut self,
        manager: Option<&'static dyn BaseManager>,
        std_output: &[String],
    ) -> Result<bool, ReadError> {
        // java.util.regex `\s+`
        let whitespace = Regex::new(r"[ \t\n\x0B\x0C\r]+").unwrap();
        let filename = self.filename.clone();
        let failed = || {
            crate::imod::etomo::util::utilities::timestamp_full(
                Some("read"),
                Some("header"),
                filename.as_deref(),
                Some(crate::imod::etomo::util::utilities::FAILED_STATUS),
            );
        };
        let java_trim = |s: &str| s.trim_matches(|c: char| c <= ' ').to_owned();
        let parse_int = |token: &str| -> Result<i32, ReadError> {
            token
                .parse::<i32>()
                .map_err(|_| ReadError::NumberFormat(format!("For input string: \"{}\"", token)))
        };
        let mut pixels_parsed = false;
        for line in std_output {
            if line.contains("This is an HDF file") {
                self.image_output_format = ImageOutputFormat::Hdf;
                continue;
            }
            // Parse the size of the data
            // Note the initial space in the string below
            // Need to get brief header and regular header in the same way, so change
            // so that the output is trimmed for this parse.
            if java_trim(line).starts_with(SIZE_HEADER) {
                let tokens = java_lang_string_split(&java_trim(line), &whitespace);
                if self.debug {
                    print!("tokens=");
                    for token in &tokens {
                        print!("{},", token);
                    }
                }
                if tokens.len() < (N_SECTIONS_INDEX + 1) as usize {
                    failed();
                    return Err(ReadError::Io(
                        "Header returned less than three parameters for image size".into(),
                    ));
                }
                // Integer.parseInt throws NumberFormatException, uncaught here.
                self.n_columns = parse_int(&tokens[N_COLUMNS_INDEX as usize])?;
                match parse_int(&tokens[N_ROWS_INDEX as usize]) {
                    Ok(n_rows) => self.n_rows = n_rows,
                    Err(_) => {
                        self.n_rows = -1;
                        failed();
                        return Err(ReadError::NumberFormat(format!(
                            "nRows not set, token is {}",
                            tokens[N_ROWS_INDEX as usize]
                        )));
                    }
                }
                match parse_int(&tokens[N_SECTIONS_INDEX as usize]) {
                    Ok(n_sections) => self.n_sections = n_sections,
                    Err(e) => {
                        // e.printStackTrace()
                        eprintln!("java.lang.NumberFormatException: {}", e);
                        self.n_sections = -1;
                        failed();
                        return Err(ReadError::NumberFormat(format!(
                            "nSections not set, token is {}",
                            tokens[N_SECTIONS_INDEX as usize]
                        )));
                    }
                }
            }

            // Parse the mode
            if line.starts_with(" Map mode") {
                // String.split("\\s+"): the leading blank gives an empty first token.
                let tokens = java_lang_string_split(line, &whitespace);
                if tokens.len() < 5 {
                    failed();
                    return Err(ReadError::Io(
                        "Header returned less than one parameter for the mode".into(),
                    ));
                }
                self.mode = parse_int(&tokens[4])?;
            }

            // Parse the pixels size
            if line.starts_with(" Pixel spacing") {
                let tokens = java_lang_string_split(line, &whitespace);
                if tokens.len() < 7 {
                    failed();
                    return Err(ReadError::Io(
                        "Header returned less than three parameters for pixel size".into(),
                    ));
                }
                // PixelsParsed will be set to true if there are no errors parsing
                // "Pixel Spacing".
                pixels_parsed = Self::parse_pixel_spacing(
                    manager,
                    &mut self.x_pixel_size,
                    &tokens[4],
                    true,
                    self.filename.as_deref(),
                    self.axis_id,
                );
                pixels_parsed = pixels_parsed
                    && Self::parse_pixel_spacing(
                        manager,
                        &mut self.y_pixel_size,
                        &tokens[5],
                        !pixels_parsed,
                        self.filename.as_deref(),
                        self.axis_id,
                    );
                pixels_parsed = pixels_parsed
                    && Self::parse_pixel_spacing(
                        manager,
                        &mut self.z_pixel_size,
                        &tokens[6],
                        !pixels_parsed,
                        self.filename.as_deref(),
                        self.axis_id,
                    );

                self.x_pixel_spacing = self.x_pixel_size.get_double();
                self.y_pixel_spacing = self.y_pixel_size.get_double();
                self.z_pixel_spacing = self.z_pixel_size.get_double();
            }

            // Parse the rotation angle, binning, etc from the comment section
            self.parse_comment_data(line);
        }
        // Once the entire header is processed, handle pixel size issues.
        //
        // If the pixel sizes are default value scan, use the FEI pixel size in the
        // comment section (if available).
        if !self.fei_pixel_size.lock().unwrap().is_null()
            && (self.pixel_equals(1.0) || self.pixel_equals(2.0) || self.pixel_equals(4.0))
        {
            // This was function parseFEIPixelSize:
            let fei = self.fei_pixel_size.lock().unwrap().to_string();
            if Self::parse_pixel_spacing(
                manager,
                &mut self.x_pixel_size,
                &fei,
                !pixels_parsed,
                self.filename.as_deref(),
                self.axis_id,
            ) {
                let x = self.x_pixel_size.get_double();
                self.x_pixel_size.set_double(x * 10.0);
            }
            let x = self.x_pixel_size.base.clone();
            self.y_pixel_size.set_const_etomo_number(Some(&x));
            let y = self.y_pixel_size.base.clone();
            self.z_pixel_size.set_const_etomo_number(Some(&y));
        }

        crate::imod::etomo::util::utilities::timestamp_full(
            Some("read"),
            Some("header"),
            self.filename.as_deref(),
            Some(crate::imod::etomo::util::utilities::FINISHED_STATUS),
        );
        Ok(true)
    }

    /// Java private `parsePixelSpacing(BaseManager, EtomoNumber, String,
    /// boolean)`.  Parse pixel spacing (pixel size).  If there is an error, pop
    /// up an error message, if requested.  The fields the Java reads from
    /// `this` (`filename`, `axisID`) are passed in, since the pixel size being
    /// set is itself a field.
    fn parse_pixel_spacing(
        manager: Option<&'static dyn BaseManager>,
        pixel_spacing: &mut EtomoNumber,
        s_pixel_spacing: &str,
        popup_error_message: bool,
        filename: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> bool {
        pixel_spacing.set_string(Some(s_pixel_spacing));
        let d_pixel_spacing = pixel_spacing.get_double();
        if !pixel_spacing.is_valid() || d_pixel_spacing == -1.0 || d_pixel_spacing == 0.0 {
            if popup_error_message {
                let message = format!(
                    "Invalid pixel spacing:  {}.  Fix the mrc header in {} with alterheader.",
                    s_pixel_spacing,
                    filename.unwrap_or("null")
                );
                // UIHarness.INSTANCE.openMessageDialog: read() also runs on
                // process monitor threads, where the call is posted to the EDT.
                if crate::imod::etomo::util::event_queue::is_dispatch_thread() {
                    crate::imod::etomo::ui::swing::ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            manager,
                            &message,
                            "Header Error",
                            axis_id,
                        )
                    });
                } else {
                    crate::imod::etomo::ui::swing::ui_harness::post_message_dialog(
                        manager,
                        message,
                        "Header Error".to_owned(),
                        axis_id,
                    );
                }
            }
            return false;
        }
        true
    }

    /// Java `synchronized read(BaseManager)`: runs `ApplicationManager.getIMODBinPath()
    /// + "header"` on the file through a `SystemProgram` and parses its output with
    /// [`MRCHeader::read`].  `Err` carries the `IOException` or
    /// `InvalidParameterException` the source throws, or the `NumberFormatException`
    /// of the size parse; `Ok(false)` is the source's `return false` for a file that
    /// does not exist.
    pub fn read_with_manager(
        &mut self,
        manager: &'static dyn BaseManager,
    ) -> Result<bool, ReadError> {
        let file = crate::imod::etomo::util::utilities::get_file(
            self.file_location.as_deref().unwrap_or("null"),
            self.filename.as_deref(),
        );
        if self
            .filename
            .as_deref()
            // `filename.matches("\\s*")`: Java's `\s` is `[ \t\n\x0B\f\r]`
            .is_none_or(|filename| {
                filename
                    .chars()
                    .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
            })
            || file.is_dir()
        {
            return Err(ReadError::Io("No filename specified".to_owned()));
        }
        if !file.exists() {
            if *DEBUG {
                eprintln!(
                    "WARNING: attempting to read the header of {}, which doesn't exist.",
                    java_io_file_get_absolute_path(&file.to_string_lossy())
                );
            }
            return Ok(false);
        }
        // If the file hasn't changed, don't reread
        if !self.modified_flag.is_modified_since_last_read() {
            return Ok(true);
        }
        let filename = self.filename.clone();
        crate::imod::etomo::util::utilities::timestamp_full(
            Some("read"),
            Some("header"),
            filename.as_deref(),
            Some(crate::imod::etomo::util::utilities::STARTED_STATUS),
        );

        // Run the header command on the filename, need to use a String[] here to
        // prevent the Runtime from breaking up the command and arguments at spaces.
        let command_array = vec![
            format!(
                "{}header",
                crate::imod::etomo::base_manager::get_imod_bin_path()
                    .unwrap_or_else(|| "null".to_owned())
            ),
            filename.clone().unwrap_or_else(|| "null".to_owned()),
        ];
        let header = crate::imod::etomo::process::system_program::SystemProgram::new_array(
            Some(manager),
            self.file_location.clone(),
            Some(command_array),
            // `SystemProgram` takes a non-null `AxisID`; the headers read here always
            // carry one (`getInstance` is given the caller's axis).
            self.axis_id.unwrap_or(AxisID::Only),
        );
        self.modified_flag.set_reading_now();
        header.run();

        let failed = || {
            crate::imod::etomo::util::utilities::timestamp_full(
                Some("read"),
                Some("header"),
                filename.as_deref(),
                Some(crate::imod::etomo::util::utilities::FAILED_STATUS),
            );
        };
        if header.get_exit_value() != 0 {
            let messages = header.get_process_messages();
            let error = crate::imod::etomo::process::process_messages::MessageType::Error;
            if messages.size(error) > 0 {
                let mut message = "header returned an error:\n".to_owned();
                for i in 0..messages.size(error) {
                    message = message + messages.get(error, i).unwrap_or("null") + "\n";
                }
                failed();
                return Err(ReadError::InvalidParameter(format!(
                    "{}:{message}",
                    filename.as_deref().unwrap_or("null")
                )));
            }
        }
        // Throw an exception if the file can not be read
        let std_error = header.get_std_error().unwrap_or_default();
        if !std_error.is_empty() {
            let mut message = "header returned an error:\n".to_owned();
            for line in &std_error {
                message = message + line + "\n";
            }
            failed();
            return Err(ReadError::InvalidParameter(format!(
                "{}:{message}",
                filename.as_deref().unwrap_or("null")
            )));
        }

        // Parse the output
        let std_output = header.get_std_output().unwrap_or_default();
        if std_output.is_empty() {
            failed();
            return Err(ReadError::Io("header returned no data".to_owned()));
        }
        self.parse_std_output(Some(manager), &std_output)
    }

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
        self.bidir.lock().unwrap().get()
    }

    /// Java `getDoseSym`.
    pub fn get_dose_sym(&self) -> ConstEtomoNumber {
        self.dose_sym.lock().unwrap().get()
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
        self.image_rotation.lock().unwrap().get()
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
        self.binning.lock().unwrap().to_string()
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
                let comment_data = Arc::clone(&self.comment_data_vector[comment_data_index]);
                // Look for comment data. Value is in the next element of splitArray.
                if !comment_data.lock().unwrap().find(split_array[i].as_deref())
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
                comment_data.lock().unwrap().set(Some(&value_split[0]));
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
                self.image_rotation.lock().unwrap(),
                self.binning.lock().unwrap(),
                match self.axis_id {
                    None => "null".to_string(),
                    Some(axis_id) => axis_id.to_string(),
                },
                self.dose_sym.lock().unwrap(),
                self.fei_pixel_size.lock().unwrap()
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
            self.image_rotation.lock().unwrap(),
            self.binning.lock().unwrap(),
            match self.axis_id {
                None => "null".to_string(),
                Some(axis_id) => axis_id.to_string(),
            },
            self.bidir.lock().unwrap(),
            self.fei_pixel_size.lock().unwrap()
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
        comment_data_vector: Option<&mut Vec<Arc<Mutex<CommentData>>>>,
        tag: &str,
    ) -> Arc<Mutex<CommentData>> {
        let comment_data = Arc::new(Mutex::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value: EtomoNumber::new(),
            name_ends_with_tag: false,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Arc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, EtomoNumber.Type)`.
    pub fn new_with_type(
        comment_data_vector: Option<&mut Vec<Arc<Mutex<CommentData>>>>,
        tag: &str,
        r#type: Type,
    ) -> Arc<Mutex<CommentData>> {
        let comment_data = Arc::new(Mutex::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value: EtomoNumber::new_with_type(Some(r#type)),
            name_ends_with_tag: false,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Arc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, EtomoNumber.Type, Integer)`.
    pub fn new_with_default(
        comment_data_vector: Option<&mut Vec<Arc<Mutex<CommentData>>>>,
        tag: &str,
        r#type: Type,
        default_value: Option<i32>,
    ) -> Arc<Mutex<CommentData>> {
        let mut value = EtomoNumber::new_with_type(Some(r#type));
        if let Some(default_value) = default_value {
            value.set_display_value_int(default_value);
        }
        let comment_data = Arc::new(Mutex::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value,
            name_ends_with_tag: false,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Arc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, EtomoNumber.Type, boolean)`.
    pub fn new_with_name_ends_with_tag(
        comment_data_vector: Option<&mut Vec<Arc<Mutex<CommentData>>>>,
        tag: &str,
        r#type: Type,
        name_ends_with_tag: bool,
    ) -> Arc<Mutex<CommentData>> {
        let comment_data = Arc::new(Mutex::new(CommentData {
            tag: tag.to_string(),
            alt_tag: None,
            value: EtomoNumber::new_with_type(Some(r#type)),
            name_ends_with_tag,
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Arc::clone(&comment_data));
        }
        comment_data
    }

    /// Java `CommentData(Vector<CommentData>, String, String, EtomoNumber.Type)`.
    pub fn new_with_alt_tag(
        comment_data_vector: Option<&mut Vec<Arc<Mutex<CommentData>>>>,
        tag: &str,
        alt_tag: Option<&str>,
        r#type: Type,
    ) -> Arc<Mutex<CommentData>> {
        let comment_data = Arc::new(Mutex::new(CommentData {
            tag: tag.to_string(),
            alt_tag: alt_tag.map(|alt_tag| alt_tag.to_string()),
            name_ends_with_tag: false,
            value: EtomoNumber::new_with_type(Some(r#type)),
        }));
        if let Some(comment_data_vector) = comment_data_vector {
            comment_data_vector.push(Arc::clone(&comment_data));
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
