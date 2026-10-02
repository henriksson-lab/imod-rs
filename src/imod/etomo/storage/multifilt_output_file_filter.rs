//! `IMOD/Etomo/src/etomo/storage/MultifiltOutputFileFilter.java`.
//!
//! `MultifiltOutputFileFilter extends ExtensionFileFilter`: the superclass is
//! embedded as `base` and reached through `Deref`.  The two `accept` overrides
//! are inherent methods (`accept_file`, `accept_file_string`), and with
//! `getDescription` they are also the `jdk::FileFilter` implementation, so the
//! filter can be handed to a file chooser as `Rc<dyn FileFilter>`.  Built on the
//! event dispatch thread, so instances are `Rc`.

use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use regex::Regex;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;

/// The strings `new java.math.BigDecimal(String)` accepts: an optional sign, a
/// significand with at least one (ASCII) digit and an optional point, and an
/// optional exponent.
static BIG_DECIMAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$").unwrap()
});

/// Java `public final class MultifiltOutputFileFilter extends ExtensionFileFilter`.
pub struct MultifiltOutputFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
    /// Java `private final BaseManager manager` (never null here).
    manager: &'static dyn BaseManager,
    /// Java `private final AxisID axisID`.
    axis_id: AxisID,
    /// Java `private final FileType fileType`.
    file_type: Option<Arc<FileType>>,
}

impl std::ops::Deref for MultifiltOutputFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl MultifiltOutputFileFilter {
    /// Java private `MultifiltOutputFileFilter(BaseManager, AxisID, FileType)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        file_type: Option<Arc<FileType>>,
    ) -> MultifiltOutputFileFilter {
        MultifiltOutputFileFilter {
            base: ExtensionFileFilter::new(Some(manager), false, false, false),
            manager,
            axis_id,
            file_type,
        }
    }

    /// Java static `getInstance(BaseManager, ImageFilenameStyle, AxisID,
    /// FileType)`.  The `imageFilenameStyle` parameter is not read.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        image_filename_style: Option<ImageFilenameStyle>,
        axis_id: AxisID,
        file_type: Option<Arc<FileType>>,
    ) -> Rc<MultifiltOutputFileFilter> {
        let _ = image_filename_style;
        let mut instance = MultifiltOutputFileFilter::new(manager, axis_id, file_type);
        instance.setup(Some(manager));
        Rc::new(instance)
    }

    /// Java private `setup(BaseManager)`.
    fn setup(&mut self, manager: Option<&'static dyn BaseManager>) {
        let file_type = self.file_type.clone();
        let file_types: Vec<Option<&FileType>> = if let Some(file_type) = &file_type {
            vec![Some(&**file_type)]
        } else {
            vec![
                Some(&*file_type::CLASS.mutlifilt_fake_sirt_iterations_output_template),
                Some(&*file_type::CLASS.mutlifilt_exact_object_sizes_output_template),
                Some(&*file_type::CLASS.mutlifilt_gaussian_output_template),
                Some(&*file_type::CLASS.mutlifilt_hamming_like_starts_output_template),
            ]
        };
        let mut dataset_name: Option<String> = None;
        let mut axis_type: Option<AxisType> = None;
        if let Some(manager) = manager {
            let meta_data = manager.get_base_meta_data();
            if let Some(meta_data) = meta_data {
                dataset_name = meta_data.get_name();
                axis_type = Some(meta_data.base().get_axis_type());
            }
        }
        self.base.setup(
            dataset_name.as_deref(),
            axis_type,
            Some(self.axis_id),
            None,
            None,
            Some(&file_types),
        );
    }

    /// Java `accept(File)`.  Return true if the file name contains a multifilt
    /// output template followed by valid number(s).
    pub fn accept_file(&self, file: Option<&Path>) -> bool {
        self.base.accept(file)
    }

    /// Java `accept(File, String)`.  Return true if the file name contains a
    /// multifilt output template followed by valid number(s).  If the fileType
    /// member variable is set, it only return true if the fileName type matches
    /// it.  `dir` is ignored; `file_name` must match a fileType for true to be
    /// returned.
    pub fn accept_file_string(&self, dir: &Path, file_name: &str) -> bool {
        self.base.accept_in_dir(Some(dir), Some(file_name))
    }

    /// Java `acceptForRegressionTest(File, String)` (deprecated: only for
    /// testing).  `dir` is ignored.
    pub fn accept_for_regression_test(&self, dir: Option<&Path>, file_name: Option<&str>) -> bool {
        let _ = dir;
        let Some(file_name) = file_name.filter(|file_name| !file_name.ends_with('~')) else {
            return false;
        };
        if let Some(file_type) = &self.file_type {
            return self.accept_template(file_name, file_type);
        }
        // dateset_slfinnnn.mrc
        if self.accept_template(
            file_name,
            &file_type::CLASS.mutlifilt_fake_sirt_iterations_output_template_old,
        ) {
            return true;
        }
        // dataset_efosnnnn.mrc
        if self.accept_template(
            file_name,
            &file_type::CLASS.mutlifilt_exact_object_sizes_output_template_old,
        ) {
            return true;
        }
        // dataset_gfc0.xxx-f0.xxx.mrc
        if self.accept_template(
            file_name,
            &file_type::CLASS.mutlifilt_gaussian_output_template_old,
        ) {
            return true;
        }
        // dataset_hlfs0.xxx.mrc
        if self.accept_template(
            file_name,
            &file_type::CLASS.mutlifilt_hamming_like_starts_output_template_old,
        ) {
            return true;
        }
        false
    }

    /// Java private `acceptTemplate(String, FileType)` (deprecated).  The file
    /// name should be longer then the template.  The part of the file name that
    /// extends beyond the template should be a valid number.
    fn accept_template(&self, file_name: &str, file_type: &Arc<FileType>) -> bool {
        // Upstream bug fixed in translation (MultifiltOutputFileFilter.java:150):
        // a null template makes `fileName.startsWith(template)` throw
        // NullPointerException; here the name does not match.
        let Some(template) = file_type.get_template(Some(self.manager), Some(self.axis_id)) else {
            return false;
        };
        if !file_name.starts_with(template.as_str()) {
            return false;
        }
        let template_length = template.len();
        if file_name.len() <= template_length {
            return false;
        }
        let middle_piece = file_type.get_middle_piece();
        let mut middle_piece_index: Option<usize> = None;
        let mut middle_piece_length = 0;
        if let Some(middle_piece) = middle_piece {
            middle_piece_length = middle_piece.len();
            middle_piece_index = file_name[template_length..]
                .find(middle_piece)
                .map(|index| index + template_length);
            if middle_piece_index.is_none() {
                return false;
            }
        }
        let Some(extension) = file_type.get_extension() else {
            return false;
        };
        // `fileName.indexOf(extension, templateLength + middlePieceLength)`: a start
        // past the end finds nothing.
        let ext_start = template_length + middle_piece_length;
        let ext_index = file_name
            .get(ext_start..)
            .and_then(|rest| rest.find(extension))
            .map(|index| index + ext_start);
        let Some(ext_index) = ext_index else {
            return false;
        };
        // Check the first number
        // `String.substring` throws StringIndexOutOfBoundsException when the end
        // precedes the start; that is not a valid number either.
        let substring = if let Some(middle_piece_index) = middle_piece_index {
            file_name.get(template_length..middle_piece_index)
        } else {
            file_name.get(template_length..ext_index)
        };
        // new BigDecimal(substring)
        if !substring.is_some_and(|substring| BIG_DECIMAL.is_match(substring)) {
            return false;
        }
        // Check the second number
        if let Some(middle_piece_index) = middle_piece_index {
            let substring = file_name.get(middle_piece_index + middle_piece_length..ext_index);
            if !substring.is_some_and(|substring| BIG_DECIMAL.is_match(substring)) {
                return false;
            }
        }
        true
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> Option<String> {
        if let Some(file_type) = &self.file_type {
            return file_type.get_description();
        }
        Some("All Filter Trials".to_owned())
    }
}

impl FileFilter for MultifiltOutputFileFilter {
    fn accept(&self, file: &Path) -> bool {
        MultifiltOutputFileFilter::accept_file(self, Some(file))
    }

    fn get_description(&self) -> Option<String> {
        MultifiltOutputFileFilter::get_description(self)
    }
}
