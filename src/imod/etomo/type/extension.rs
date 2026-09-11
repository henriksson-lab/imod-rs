//! `IMOD/Etomo/src/etomo/type/Extension.java`.
//!
//! Holds all file extensions, including customizable image file extensions.
//!
//! **Java class statics.**  `Extension`'s constructor has side effects: with `store`
//! set it registers the new instance in `INSTANCES`, appends to
//! `INPUT_IMAGE_FILE_DESCR` and `_imageInputRegex_`, and adds to
//! `STANDARDIZABLE_IMAGE_FILE_LIST`, `INPUT_IMAGE_FILE_LIST` and `VARIABLE_LIST`.  The
//! contents of all six therefore depend on the *declaration order* of the
//! `public static final Extension` fields (Extension.java:25-122).  `ClassStatics`
//! below is that class initialiser: one lazily-run block that constructs the singletons
//! in source order and keeps the six collections it fills.
//!
//! **Identity verses value.**  The source compares `Extension`s with `==`
//! (`extension == Extension.COM`, `ext1 == ext2`).  Every instance the lookups return
//! comes out of `INSTANCES`, which is keyed by the extension text, so no two stored
//! instances can carry the same text and value equality decides exactly what Java's
//! identity comparison decides.  The one instance that is *not* stored is the one
//! `getLiteralInstance` constructs for an unrecognised extension, and nothing in the
//! source compares that by identity.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::LazyLock;

use super::const_etomo_number::{ConstEtomoNumber, Type};
use super::etomo_number::EtomoNumber;
use super::extension_marker::ExtensionMarker;
use super::file_type;
use super::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::util::utilities;

/// Java `EXTENSION_DIVIDER`.
pub const EXTENSION_DIVIDER: &str = ".";
/// Java `STANDARDIZATION_DIVIDER`.
pub const STANDARDIZATION_DIVIDER: &str = "_";
/// Java `BACKUP_SUFFIX`.
pub const BACKUP_SUFFIX: &str = "~";

/// Java `Extension`.  Field order follows the source declaration
/// (Extension.java:124-127): `extension`, then the three booleans declared together,
/// then `inputImageFile`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extension {
    /// Java field `extension`.
    extension: String,
    /// Java field `etomoImageFile`.
    etomo_image_file: bool,
    /// Java field `standardImageFileExtension`.
    standard_image_file_extension: bool,
    /// Java field `variable`.
    variable: bool,
    /// Java field `inputImageFile`.  For file formats that can hold input image stacks
    /// (st,mrc,hdf,tif,tiff).
    input_image_file: bool,
}

/// The `Extension` class's static state and the class initialiser that fills it:
/// `_imageInputRegex_`, `INPUT_IMAGE_FILE_DESCR`, `INSTANCES`,
/// `STANDARDIZABLE_IMAGE_FILE_LIST`, `INPUT_IMAGE_FILE_LIST`, `VARIABLE_LIST` and every
/// `public static final Extension` singleton (Extension.java:25-122).
pub struct ClassStatics {
    /// Java `_imageInputRegex_`.
    image_input_regex: String,
    /// Java `INPUT_IMAGE_FILE_DESCR`.
    input_image_file_descr: String,
    /// Java `INSTANCES`.
    instances: HashMap<String, Extension>,
    /// Java `STANDARDIZABLE_IMAGE_FILE_LIST`.
    standardizable_image_file_list: Vec<Extension>,
    /// Java `INPUT_IMAGE_FILE_LIST`.
    input_image_file_list: Vec<Extension>,
    /// Java `VARIABLE_LIST`.
    variable_list: Vec<Extension>,
    /// Java `HDF`.
    pub hdf: Extension,
    /// Java `MRC`.
    pub mrc: Extension,
    /// Java `ST`.
    pub st: Extension,
    /// Java `TIF`.
    pub tif: Extension,
    /// Java `TIFF`.
    pub tiff: Extension,
    /// Java `ALI`.
    pub ali: Extension,
    /// Java `ALILOG10`.
    pub alilog10: Extension,
    /// Java `BL`.
    pub bl: Extension,
    /// Java `DCST`.
    pub dcst: Extension,
    /// Java `FLAT`.
    pub flat: Extension,
    /// Java `FLIP`.
    pub flip: Extension,
    /// Java `INPUT`.
    pub input: Extension,
    /// Java `JOIN`.
    pub join: Extension,
    /// Java `MAT`.
    pub mat: Extension,
    /// Java `NAD`.
    pub nad: Extension,
    /// Java `PREALI`.
    pub preali: Extension,
    /// Java `REC`.
    pub rec: Extension,
    /// Java `ROT`.
    pub rot: Extension,
    /// Java `SAMPAVG`.
    pub sampavg: Extension,
    /// Java `SAMPLE`.
    pub sample: Extension,
    /// Java `SQZ`.
    pub sqz: Extension,
    /// Java `VSR`.
    pub vsr: Extension,
    /// Java `SINT`.
    pub sint: Extension,
    /// Java `SREC`.
    pub srec: Extension,
    /// Java `COM`.
    pub com: Extension,
    /// Java `JPG`.
    pub jpg: Extension,
    /// Java `LOG`.
    pub log: Extension,
    /// Java `MDOC`.
    pub mdoc: Extension,
    /// Java `MOD`.
    pub r#mod: Extension,
    /// Java `OUT`.
    pub out: Extension,
    /// Java `PNG`.
    pub png: Extension,
    /// Java `PRM`.
    pub prm: Extension,
    /// Java `PCM`.
    pub pcm: Extension,
    /// Java `TXT`.
    pub txt: Extension,
    /// Java `TLT`.
    pub tlt: Extension,
    /// Java `PL`.
    pub pl: Extension,
    /// Java `WARPXG`.
    pub warpxg: Extension,
    /// Java `EER`.
    pub eer: Extension,
    /// Java `PT`.
    pub pt: Extension,
    /// Java `RAWTLT`.
    pub rawtlt: Extension,
    /// Java `CMDS`.
    pub cmds: Extension,
    /// Java `EBT`.
    pub ebt: Extension,
    /// Java `ACTIVE`.
    pub active: Extension,
}

/// The single instance of the class's static state.
pub static CLASS: LazyLock<ClassStatics> = LazyLock::new(|| {
    let mut image_input_regex = String::new();
    let mut input_image_file_descr = String::new();
    let mut instances: HashMap<String, Extension> = HashMap::new();
    let mut standardizable_image_file_list: Vec<Extension> = Vec::new();
    let mut input_image_file_list: Vec<Extension> = Vec::new();
    let mut variable_list: Vec<Extension> = Vec::new();
    // Java `Extension(String, boolean, boolean, boolean, boolean, boolean)`
    // (Extension.java:139).  The `store` branch is the class's registration side
    // effect; it is run here in source declaration order.
    let mut construct = |extension: &str,
                         etomo_image_file: bool,
                         input_image_file: bool,
                         standard_image_file_extension: bool,
                         variable: bool,
                         store: bool|
     -> Extension {
        let instance = Extension {
            extension: extension.to_string(),
            etomo_image_file,
            standard_image_file_extension,
            variable,
            input_image_file,
        };
        if store {
            instances.insert(extension.to_string(), instance.clone());
            if input_image_file {
                input_image_file_descr.push_str(
                    &(if !input_image_file_descr.is_empty() {
                        ", "
                    } else {
                        ""
                    })
                    .to_string(),
                );
                input_image_file_descr.push_str(extension);
            }
            if instance.is_standardizable_image_file() {
                standardizable_image_file_list.push(instance.clone());
            }
            if input_image_file {
                input_image_file_list.push(instance.clone());
                if !image_input_regex.is_empty() {
                    image_input_regex.push('|');
                }
                image_input_regex.push_str(&format!("\\Q{}\\E", extension));
            }
            if variable {
                variable_list.push(instance.clone());
            }
        }
        instance
    };
    let hdf = construct("hdf", true, true, true, false, true);
    let mrc = construct("mrc", true, true, true, false, true);
    //
    let st = construct("st", true, true, false, false, true);
    let tif = construct("tif", true, true, false, false, true);
    let tiff = construct("tiff", true, true, false, false, true);
    //
    let ali = construct("ali", true, false, false, false, true);
    let alilog10 = construct("alilog10", true, false, false, false, true);
    let bl = construct("bl", true, false, false, false, true);
    let dcst = construct("dcst", true, false, false, false, true);
    let flat = construct("flat", true, false, false, false, true);
    let flip = construct("flip", true, false, false, false, true);
    let input = construct("input", true, false, false, false, true);
    let join = construct("join", true, false, false, false, true);
    let mat = construct("mat", true, false, false, false, true);
    let nad = construct("nad", true, false, false, false, true);
    let preali = construct("preali", true, false, false, false, true);
    let rec = construct("rec", true, false, false, false, true);
    let rot = construct("rot", true, false, false, false, true);
    let sampavg = construct("sampavg", true, false, false, false, true);
    let sample = construct("sample", true, false, false, false, true);
    let sqz = construct("sqz", true, false, false, false, true);
    let vsr = construct("vsr", true, false, false, false, true);
    //
    let sint = construct("sint", true, false, false, true, true);
    let srec = construct("srec", true, false, false, true, true);
    //
    let com = construct("com", false, false, false, false, true);
    let jpg = construct("jpg", false, false, false, false, true);
    let log = construct("log", false, false, false, false, true);
    let mdoc = construct("mdoc", false, false, false, false, true);
    let r#mod = construct("mod", false, false, false, false, true);
    let out = construct("out", false, false, false, false, true);
    let png = construct("png", false, false, false, false, true);
    let prm = construct("prm", false, false, false, false, true);
    let pcm = construct("pcm", false, false, false, false, true);
    let txt = construct("txt", false, false, false, false, true);
    let tlt = construct("tlt", false, false, false, false, true);
    let pl = construct("pl", false, false, false, false, true);
    let warpxg = construct("warpxg", false, false, false, false, true);
    let eer = construct("eer", true, false, false, false, true);
    let pt = construct("pt", false, false, false, false, true);
    let rawtlt = construct("rawtlt", false, false, false, false, true);
    let cmds = construct("cmds", false, false, false, false, true);
    let ebt = construct("ebt", false, false, false, false, true);
    let active = construct("active", false, false, false, false, true);
    ClassStatics {
        image_input_regex,
        input_image_file_descr,
        instances,
        standardizable_image_file_list,
        input_image_file_list,
        variable_list,
        hdf,
        mrc,
        st,
        tif,
        tiff,
        ali,
        alilog10,
        bl,
        dcst,
        flat,
        flip,
        input,
        join,
        mat,
        nad,
        preali,
        rec,
        rot,
        sampavg,
        sample,
        sqz,
        vsr,
        sint,
        srec,
        com,
        jpg,
        log,
        mdoc,
        r#mod,
        out,
        png,
        prm,
        pcm,
        txt,
        tlt,
        pl,
        warpxg,
        eer,
        pt,
        rawtlt,
        cmds,
        ebt,
        active,
    }
});

impl Extension {
    /// Java `equals(String)`.
    pub fn equals_string(&self, string: Option<&str>) -> bool {
        let string = match string {
            None => return false,
            Some(string) => string,
        };
        // Java `String.equalsIgnoreCase`.
        if string.eq_ignore_ascii_case(&self.extension) {
            return true;
        }
        false
    }

    /// Java `getStandardizableImageFileIterator`.
    pub fn get_standardizable_image_file_iterator() -> std::slice::Iter<'static, Extension> {
        CLASS.standardizable_image_file_list.iter()
    }

    /// Java `getInputImageFileIterator`.
    pub fn get_input_image_file_iterator() -> std::slice::Iter<'static, Extension> {
        CLASS.input_image_file_list.iter()
    }

    /// Java `getInstance(String)`.  Returns the instance matching the regular extension
    /// of this file name.  Does not return the standardized extension.  Does handle
    /// variable extensions.  When the file name is dataset_rec.mrc, returns MRC.
    pub fn get_instance(file_name: &str) -> Option<&'static Extension> {
        let mut file_name = file_name.to_string();
        if utilities::is_empty(Some(&file_name)) {
            return None;
        }
        // Ignore the backup character(s).
        while file_name.ends_with(BACKUP_SUFFIX) {
            file_name = file_name[0..file_name.len() - 1].to_string();
        }
        if utilities::is_empty(Some(&file_name)) {
            return None;
        }
        let ext = match utilities::get_extension(Some(&file_name)) {
            // No "." so treat fileName as an extension string.
            None => file_name.clone(),
            Some(ext) => ext,
        };
        if ext.is_empty() {
            return None;
        }
        let mut extension = CLASS.instances.get(&ext);
        if extension.is_none() {
            // may be a variable extension (.srec86).
            extension = Extension::get_variable_instance(&ext);
        }
        if extension.is_none() {
            eprintln!("(1)Filename contains unknown extension:{}", file_name);
        }
        extension
    }

    /// Java `getLiteralInstance`.  Leave all character(s) in, including the backup
    /// suffix and the addons to a variable extension.  If the extension can't be found
    /// then build one, and do not add it to the collections.
    pub fn get_literal_instance(
        file_name: Option<&str>,
        extension_marker: Option<ExtensionMarker>,
        whole_string_can_be_an_extension: bool,
    ) -> Option<Extension> {
        let file_name = match file_name {
            None => return None,
            Some(file_name) if file_name.is_empty() => return None,
            Some(file_name) => file_name,
        };
        // Leave the backup character(s) in.
        let mut ext = utilities::get_extension(Some(file_name));
        if ext.is_none() && whole_string_can_be_an_extension {
            // No "." so treat fileName as an extension string.
            ext = Some(file_name.to_string());
        }
        if utilities::is_empty(ext.as_deref()) {
            return None;
        }
        let ext = ext.unwrap();
        let extension = CLASS.instances.get(&ext);
        if let Some(extension) = extension {
            return Some(extension.clone());
        }
        // Nothing matches exactly. Construct an extension but don't save it.
        Some(Extension {
            extension: ext.clone(),
            etomo_image_file: extension_marker == Some(ExtensionMarker::Image),
            standard_image_file_extension: false,
            variable: false,
            input_image_file: extension_marker == Some(ExtensionMarker::InputImage),
        })
    }

    /// Java `equals(File, File)`.  Returns true if the two files have the same
    /// extension, are both null, or both have no extension.  Doesn't handle
    /// standardized extensions.
    pub fn equals_files(file1: Option<&std::path::Path>, file2: Option<&std::path::Path>) -> bool {
        // Handle a missing file. Null files are considered equal.
        if file1.is_none() || file2.is_none() {
            if file1.is_none() && file2.is_none() {
                return true;
            }
            return false;
        }
        // Compare recognized extensions.
        let file_name1 = utilities::java_io_file_get_name(&file1.unwrap().to_string_lossy());
        let file_name2 = utilities::java_io_file_get_name(&file2.unwrap().to_string_lossy());
        let ext1 = Extension::get_instance(&file_name1);
        let ext2 = Extension::get_instance(&file_name2);
        if ext1.is_some() || ext2.is_some() {
            return ext1 == ext2;
        }
        // Handle a missing or unrecognized extension.
        let ext_str1 = utilities::get_extension(Some(&file_name1));
        let ext_str2 = utilities::get_extension(Some(&file_name2));
        // Handle a missing extension. Two missing extensions are considered equal.
        if ext_str1.is_none() || ext_str2.is_none() {
            if ext_str1.is_none() && ext_str2.is_none() {
                return true;
            }
            return false;
        }
        // Compare unrecognized extensions.
        ext_str1 == ext_str2
    }

    /// Java `getInstance(BaseManager, String, ImageFilenameStyle)`.  Returns the
    /// instance matching the regular extension or the standardized image extension.
    /// Handles variable extensions.  When the imageFilenameStyle is MRC and the file
    /// name is `dataset_rec.mrc`, returns REC.  When the imageFilenameStyle is not MRC
    /// and the file name is `dataset_rec.mrc`, returns MRC.
    pub fn get_instance_with_style(
        manager: Option<&'static dyn crate::imod::etomo::base_manager::BaseManager>,
        file_name: Option<&str>,
        image_filename_style: Option<ImageFilenameStyle>,
    ) -> Option<&'static Extension> {
        let file_name = match file_name {
            None => return None,
            Some(file_name) if file_name.is_empty() => return None,
            Some(file_name) => file_name,
        };
        let mut ext = utilities::get_extension(Some(file_name));
        let mut ext_only = false;
        if ext.is_none() {
            // No "." so treat fileName as an extension string.
            ext = Some(file_name.to_string());
            ext_only = true;
        }
        let ext = ext.unwrap();
        if ext.is_empty() {
            return None;
        }
        let mut extension = CLASS.instances.get(&ext);
        if extension.is_none() {
            // may be a variable extension (.srec86).
            extension = Extension::get_variable_instance(&ext);
        }
        // Unknown extension
        let extension = extension?;
        let mut image_filename_style = image_filename_style;
        if image_filename_style.is_none() {
            image_filename_style = manager
                .and_then(|manager| manager.get_base_meta_data())
                .map(|meta_data| meta_data.base().get_image_filename_style());
        }
        if ext_only
            || image_filename_style == Some(ImageFilenameStyle::Old)
            || image_filename_style
                .map(|style| {
                    !std::ptr::eq(style.get_default_raw_image_stack_extension(), extension)
                })
                .unwrap_or(true)
        {
            // This is not a standardized image file name. Return the regular extension.
            return Some(extension);
        }
        // The extension is MRC or HDF and the image filename style matches the extension.
        // Return the standardizable image file extension if possible.  For example for a
        // fileName that is dataset_rec.mrc, return REC.
        let standardized_extension = Extension::get_standardized_instance(Some(file_name));
        if standardized_extension.is_some() {
            return standardized_extension;
        }
        // Not a standardized image file name. Return MRC or HDF.
        Some(extension)
    }

    /// Java `getVariableInstance`.  Returns a variable extension if it matches ext - or
    /// null.  `ext` is an extension string (with no ".").
    fn get_variable_instance(ext: &str) -> Option<&'static Extension> {
        // may be a variable extension (.srec86).
        let mut iterator = CLASS.variable_list.iter();
        while let Some(variable_extension) = iterator.next() {
            if ext.starts_with(&variable_extension.extension)
                && variable_extension
                    .is_variable_piece(Some(&ext[variable_extension.extension.len()..]))
            {
                return Some(variable_extension);
            }
        }
        None
    }

    /// Java `getImageInputRegex`.
    pub fn get_image_input_regex() -> String {
        CLASS.image_input_regex.clone()
    }

    /// Java `getStandardizedInstance`.  Get the standardizeable extension from a
    /// standardized file name.  Returns null if fileName is not a standardized file
    /// name.
    fn get_standardized_instance(file_name: Option<&str>) -> Option<&'static Extension> {
        let file_name = match file_name {
            None => return None,
            Some(file_name) => file_name,
        };
        if file_name.is_empty()
            || !file_name.contains(STANDARDIZATION_DIVIDER)
            || !file_name.contains(EXTENSION_DIVIDER)
        {
            return None;
        }
        // Standardized file name always end in .mrc or .hdf.
        let reg_extension = Extension::get_instance(file_name);
        match reg_extension {
            None => return None,
            Some(reg_extension) if !reg_extension.is_standard_image_file_extension() => {
                return None;
            }
            Some(_) => {}
        }
        // Remove the regular extension. Get the poosible standardizable extension. Call
        // is the same as getSuffix but it removes the "_".
        let stripped = utilities::strip_string(
            true,
            false,
            utilities::remove_extension(Some(file_name)).as_deref(),
            true,
            false,
            Some(STANDARDIZATION_DIVIDER),
            true,
            true,
        );
        let extension = match stripped {
            // Java passes the null through to getInstance(String), whose
            // Utilities.isEmpty(null) guard returns null.
            None => None,
            Some(stripped) => Extension::get_instance(&stripped),
        };
        // Only return it if its standardizeable.
        if let Some(extension) = extension {
            if extension.is_standardizable_image_file() {
                return Some(extension);
            }
        }
        None
    }

    /// Java `substituteExtension`.  Returns fileName with newExtension substituted for
    /// its extension.  If there is no extension just tacks newExtension onto the end.
    /// Doesn't change a null or empty file name.  Doesn't change the file name if
    /// newExtension is null.
    pub fn substitute_extension(
        file_name: Option<&str>,
        new_extension: Option<&Extension>,
    ) -> Option<String> {
        let (file_name, new_extension) = match (file_name, new_extension) {
            (file_name, None) => return file_name.map(|file_name| file_name.to_string()),
            (None, _) => return None,
            (Some(file_name), _) if file_name.is_empty() => return Some(file_name.to_string()),
            (Some(file_name), Some(new_extension)) => (file_name, new_extension),
        };
        Some(
            utilities::remove_extension(Some(file_name)).unwrap_or_default()
                + EXTENSION_DIVIDER
                + &new_extension.to_string(),
        )
    }

    /// Java `substituteStandardizedExtension`.  Strictly substitutes the standardized
    /// extension (_ext), leaving the regular extension alone.  Returns null if the file
    /// does not have a standardized format, does not have a standard extension
    /// (mrc/hdf), if it doesn't contain a standardizeable extension, or if newExtension
    /// is not standardizable.
    pub fn substitute_standardized_extension(
        file_name: Option<&str>,
        new_extension: Option<&Extension>,
    ) -> Option<String> {
        let new_extension = match new_extension {
            None => return None,
            Some(new_extension) if !new_extension.is_standardizable_image_file() => return None,
            Some(new_extension) => new_extension,
        };
        let file_name = match file_name {
            None => return None,
            Some(file_name) if file_name.is_empty() => return None,
            Some(file_name) => file_name,
        };
        let reg_extension = match Extension::get_instance(file_name) {
            None => return None,
            Some(reg_extension) if !reg_extension.is_standard_image_file_extension() => {
                return None;
            }
            Some(reg_extension) => reg_extension,
        };
        if Extension::get_standardized_instance(Some(file_name)).is_none() {
            return None;
        }
        Some(
            utilities::remove_right_side(Some(file_name), Some(STANDARDIZATION_DIVIDER))
                .unwrap_or_default()
                + STANDARDIZATION_DIVIDER
                + &new_extension.to_string()
                + EXTENSION_DIVIDER
                + &reg_extension.to_string(),
        )
    }

    /// Java `getInputImageFileDescr`.
    pub fn get_input_image_file_descr() -> String {
        CLASS.input_image_file_descr.clone()
    }

    /// Java `isImageFile`.
    pub fn is_image_file(&self) -> bool {
        self.etomo_image_file || self.input_image_file || self.standard_image_file_extension
    }

    /// Java `isEtomoImageFile`.
    pub fn is_etomo_image_file(&self) -> bool {
        self.etomo_image_file
    }

    /// Java `isInputImageFile()`.
    pub fn is_input_image_file(&self) -> bool {
        self.input_image_file
    }

    /// Java `isInputImageFile(String)`.
    pub fn is_input_image_file_path(file_path: &str) -> bool {
        let extension = Extension::get_instance(file_path);
        let extension = match extension {
            None => return false,
            Some(extension) => extension,
        };
        extension.is_input_image_file()
    }

    /// Java `isComscript`.
    pub fn is_comscript(file_path: &str) -> bool {
        let extension = Extension::get_instance(file_path);
        let extension = match extension {
            None => return false,
            Some(extension) => extension,
        };
        *extension == CLASS.com || *extension == CLASS.pcm
    }

    /// Java `isVariable`.
    pub fn is_variable(&self) -> bool {
        self.variable
    }

    /// Java `isStandardizableImageFile`.
    pub fn is_standardizable_image_file(&self) -> bool {
        self.etomo_image_file && !self.input_image_file && !self.standard_image_file_extension
    }

    /// Java `fileNameEndsWith`.  Returns true if the fileName ends with this instance's
    /// suffix.  A suffix includes the "." or "_".  `fileName` may be a file name, path,
    /// or extension.
    pub fn file_name_ends_with(
        &self,
        image_filename_style: Option<ImageFilenameStyle>,
        file_name: Option<&str>,
    ) -> bool {
        let file_name = match file_name {
            None => return false,
            Some(file_name) => file_name,
        };
        let file_name = super::const_etomo_number::java_lang_string_trim(file_name);
        if file_name.is_empty() {
            return false;
        }
        let image_filename_style = match image_filename_style {
            None => ImageFilenameStyle::DEFAULT,
            Some(image_filename_style) => image_filename_style,
        };
        if !self.variable {
            // Non-variable extension. Test with endsWith.
            let suffix = self.get_suffix(Some(image_filename_style));
            if file_name.len() >= suffix.len() {
                return file_name.ends_with(&suffix);
            } else {
                // If fileName is the suffix without the divider character, return true;
                return file_name == &suffix[1..];
            }
        }
        // File has a variable extension. The numeric piece of the file name is required
        // so this file name matches only if a matching extension and valid file number
        // can be found.
        self.get_file_number(Some(file_name), Some(image_filename_style))
            .is_some()
    }

    /// Java `getChunkNumber`.  Returns the number from a NON-IMAGE chunk file name.
    /// Examples of chunk files: root-020.com, root-001.pcm, root-1000-sync.log.
    /// Works only for the above three extensions.
    pub fn get_chunk_number(file_name: Option<&str>) -> Option<ConstEtomoNumber> {
        let file_name = match file_name {
            None => return None,
            Some(file_name) => file_name,
        };
        // Strip off everything on the right side that is in the way of the chunk number.
        let extension = Extension::get_instance(file_name);
        if extension != Some(&CLASS.com)
            && extension != Some(&CLASS.pcm)
            && extension != Some(&CLASS.log)
        {
            return None;
        }
        let file_name = utilities::remove_right_side(
            utilities::remove_extension(Some(file_name)).as_deref(),
            Some("-sync"),
        );
        let file_name = match file_name {
            None => return None,
            Some(file_name) => file_name,
        };
        // Get the chunk number.
        let index = file_name.rfind(file_type::CHUNK_NUMBER_DIVIDER);
        let index = match index {
            None => return None,
            Some(index) if index + 1 == file_name.len() => return None,
            Some(index) => index,
        };
        let mut chunk_number = EtomoNumber::new();
        chunk_number.set_string(Some(&file_name[index + 1..]));
        if !chunk_number.is_null() && chunk_number.is_valid() {
            return Some((*chunk_number).clone());
        }
        None
    }

    /// Java `getFileNumber`.  The file number is an integer that follows the old style
    /// extension.  Only an instance with the member variable "variable" can have one and
    /// it is at least two digits long.
    pub fn get_file_number(
        &self,
        file_name: Option<&str>,
        image_filename_style: Option<ImageFilenameStyle>,
    ) -> Option<ConstEtomoNumber> {
        let file_name = match file_name {
            None => return None,
            Some(file_name) => file_name,
        };
        let file_name = super::const_etomo_number::java_lang_string_trim(file_name);
        if file_name.is_empty() {
            return None;
        }
        let image_filename_style = match image_filename_style {
            None => ImageFilenameStyle::DEFAULT,
            Some(image_filename_style) => image_filename_style,
        };
        if !self.variable {
            return None;
        }
        let mut file_number_string: Option<String>;
        // Old style file name
        if image_filename_style == ImageFilenameStyle::Old || !self.is_standardizable_image_file() {
            // Old style file name
            // Look for variable extension: .sintnn, .sintnnn, .srecnn, or .srecnnn.
            file_number_string = utilities::get_suffix(Some(file_name), Some(EXTENSION_DIVIDER));
            // Suffix is .sint or .srec.
            let suffix = EXTENSION_DIVIDER.to_string() + &self.extension;
            match &file_number_string {
                // name doesn't have an extension or doesn't match.
                None => return None,
                Some(value) if !value.starts_with(&suffix) => return None,
                Some(_) => {}
            }
            // Get nn or nnn.
            file_number_string =
                utilities::remove_left_side(file_number_string.as_deref(), Some(&suffix));
        } else {
            // Standardized file name
            // Look for variable extension: _sintnn.mrc, srecnnn.hdf, etc.
            file_number_string =
                utilities::get_suffix(Some(file_name), Some(STANDARDIZATION_DIVIDER));
            // midfix is _sint, _srec.
            let midfix = STANDARDIZATION_DIVIDER.to_string() + &self.extension;
            match &file_number_string {
                // name doesn't have an extension or doesn't match.
                None => return None,
                Some(value) if !value.starts_with(&midfix) => return None,
                Some(_) => {}
            }
            // reduce to nn.mrn, nn.hdf, nnn.mrc, or nnn.hdf
            file_number_string =
                utilities::remove_left_side(file_number_string.as_deref(), Some(&midfix));
            // Get nn or nnn.
            file_number_string = utilities::remove_right_side(
                file_number_string.as_deref(),
                Some(EXTENSION_DIVIDER),
            );
        }
        if self.is_variable_piece(file_number_string.as_deref()) {
            let mut file_number = EtomoNumber::new_with_type(Some(Type::Long));
            file_number.set_string(file_number_string.as_deref());
            if !file_number.is_null() && file_number.is_valid() {
                return Some((*file_number).clone());
            }
        }
        None
    }

    /// Java `isVariablePiece`.  Returns true if `number` is an integer at least two
    /// digits long (nn, nnn).
    fn is_variable_piece(&self, number: Option<&str>) -> bool {
        // More then three digits is fine. A dataset may need more then the predicted
        // maximum number of files.
        let number = match number {
            None => return false,
            Some(number) if number.len() < 2 => return false,
            Some(number) => number,
        };
        if super::const_etomo_number::java_lang_long_parse_long(number).is_ok() {
            return true;
        }
        // not an integer
        false
    }

    /// Java `length`.
    pub fn length(&self) -> i32 {
        self.extension.len() as i32
    }

    /// Java `getSuffix`.  Returns the current extension or standardized suffix: .ext,
    /// _ext.mrc, or _ext.hdf.  Includes the "." (and the "_" where used).
    pub fn get_suffix(&self, image_filename_style: Option<ImageFilenameStyle>) -> String {
        let image_filename_style = match image_filename_style {
            None => ImageFilenameStyle::DEFAULT,
            Some(image_filename_style) => image_filename_style,
        };
        if image_filename_style == ImageFilenameStyle::Old || !self.is_standardizable_image_file() {
            return EXTENSION_DIVIDER.to_string() + &self.extension;
        }
        STANDARDIZATION_DIVIDER.to_string()
            + &self.extension
            + EXTENSION_DIVIDER
            + &image_filename_style
                .get_default_raw_image_stack_extension()
                .extension
    }

    /// Java `isStandardImageFileExtension`.
    pub fn is_standard_image_file_extension(&self) -> bool {
        self.standard_image_file_extension
    }

    /// Java `isCompatible`.
    pub fn is_compatible(&self, extension_marker: Option<ExtensionMarker>) -> bool {
        if extension_marker == Some(ExtensionMarker::InputImage) {
            return self.input_image_file;
        }
        if extension_marker == Some(ExtensionMarker::Image) {
            return self.standard_image_file_extension;
        }
        true
    }
}

/// Java `toString`.
impl std::fmt::Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.extension)
    }
}
