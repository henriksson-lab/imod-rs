//! `IMOD/Etomo/src/etomo/storage/SirtOutputFileFilter.java`.
//!
//! `SirtOutputFileFilter extends ExtensionFileFilter`: the superclass is embedded
//! as `base` and reached through `Deref`.  The two `accept` overrides are
//! inherent methods (`accept_file`, `accept_file_string`), and with
//! `getDescription` they are also the `jdk::FileFilter` implementation, so the
//! filter can be handed to a file chooser as `Rc<dyn FileFilter>`.  Built on the
//! event dispatch thread, so instances are `Rc`.

use std::path::Path;
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::FileFilter;
use crate::imod::etomo::storage::extension_file_filter::ExtensionFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class SirtOutputFileFilter extends ExtensionFileFilter`.
pub struct SirtOutputFileFilter {
    /// Java superclass `ExtensionFileFilter` state.
    pub base: ExtensionFileFilter,
    /// Java `private final BaseManager manager` (never null here).
    manager: &'static dyn BaseManager,
    /// Java `private final AxisID axisID`.
    axis_id: AxisID,
    /// Java `private final boolean includeScaledOutput`.
    include_scaled_output: bool,
    /// Java `private final boolean subarea`.
    subarea: bool,
    /// Java `private final boolean full`.
    full: bool,
}

impl std::ops::Deref for SirtOutputFileFilter {
    type Target = ExtensionFileFilter;

    fn deref(&self) -> &ExtensionFileFilter {
        &self.base
    }
}

impl SirtOutputFileFilter {
    /// Java private `SirtOutputFileFilter(BaseManager, ImageFilenameStyle, AxisID,
    /// boolean, boolean, boolean)`.  The `imageFilenameStyle` parameter is not
    /// read.
    fn new(
        manager: &'static dyn BaseManager,
        image_filename_style: Option<ImageFilenameStyle>,
        axis_id: AxisID,
        include_scaled_output: bool,
        subarea: bool,
        full: bool,
    ) -> SirtOutputFileFilter {
        let _ = image_filename_style;
        SirtOutputFileFilter {
            base: ExtensionFileFilter::new(Some(manager), false, false, false),
            manager,
            axis_id,
            include_scaled_output,
            subarea,
            full,
        }
    }

    /// Java static `getInstance(BaseManager, ImageFilenameStyle, AxisID, boolean,
    /// boolean, boolean)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        image_filename_style: Option<ImageFilenameStyle>,
        axis_id: AxisID,
        include_scaled_output: bool,
        subarea: bool,
        full: bool,
    ) -> Rc<SirtOutputFileFilter> {
        let mut instance = SirtOutputFileFilter::new(
            manager,
            image_filename_style,
            axis_id,
            include_scaled_output,
            subarea,
            full,
        );
        instance.setup(Some(manager));
        Rc::new(instance)
    }

    /// Java static `getSubareaInstance(BaseManager, ImageFilenameStyle, AxisID,
    /// boolean)`.
    pub fn get_subarea_instance(
        manager: &'static dyn BaseManager,
        image_filename_style: Option<ImageFilenameStyle>,
        axis_id: AxisID,
        include_scaled_output: bool,
    ) -> Rc<SirtOutputFileFilter> {
        let mut instance = SirtOutputFileFilter::new(
            manager,
            image_filename_style,
            axis_id,
            include_scaled_output,
            true,
            false,
        );
        instance.setup(Some(manager));
        Rc::new(instance)
    }

    /// Java static `getFullInstance(BaseManager, ImageFilenameStyle, AxisID,
    /// boolean)`.
    pub fn get_full_instance(
        manager: &'static dyn BaseManager,
        image_filename_style: Option<ImageFilenameStyle>,
        axis_id: AxisID,
        include_scaled_output: bool,
    ) -> Rc<SirtOutputFileFilter> {
        let mut instance = SirtOutputFileFilter::new(
            manager,
            image_filename_style,
            axis_id,
            include_scaled_output,
            false,
            true,
        );
        instance.setup(Some(manager));
        Rc::new(instance)
    }

    /// Java private `setup(BaseManager)`.
    fn setup(&mut self, manager: Option<&'static dyn BaseManager>) {
        let file_types: [Option<&FileType>; 4] = [
            if self.full {
                Some(&*file_type::CLASS.sirt_output_template)
            } else {
                None
            },
            if self.subarea {
                Some(&*file_type::CLASS.sirt_subarea_output_template)
            } else {
                None
            },
            if self.include_scaled_output && self.full {
                Some(&*file_type::CLASS.sirt_scaled_output_template)
            } else {
                None
            },
            if self.include_scaled_output && self.subarea {
                Some(&*file_type::CLASS.sirt_subarea_scaled_output_template)
            } else {
                None
            },
        ];
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

    /// Java `accept(File)`.  Return true if the file name contains a SIRT output
    /// template followed by a valid integer.
    pub fn accept_file(&self, file: Option<&Path>) -> bool {
        self.base.accept(file)
    }

    /// Java `accept(File, String)`.
    pub fn accept_file_string(&self, dir: &Path, file_name: &str) -> bool {
        self.base.accept_in_dir(Some(dir), Some(file_name))
    }

    /// Java `acceptForRegressionTest(File, String)` (deprecated: only for
    /// testing).  `dir` is not read.
    pub fn accept_for_regression_test(&self, dir: Option<&Path>, file_name: &str) -> bool {
        let _ = dir;
        if file_name.ends_with('~') {
            return false;
        }
        let manager = Some(self.manager);
        let axis_id = Some(self.axis_id);
        let mut template: Option<String>;
        if self.full {
            template = file_type::CLASS
                .sirt_output_template_old
                .get_template(manager, axis_id);
            if let Some(template) = &template
                && file_name.starts_with(template.as_str())
            {
                return self.accept_template(file_name, template);
            }
        }
        if self.subarea {
            template = file_type::CLASS
                .sirt_subarea_output_template_old
                .get_template(manager, axis_id);
            if let Some(template) = &template
                && file_name.starts_with(template.as_str())
            {
                return self.accept_template(file_name, template);
            }
        }
        if self.include_scaled_output {
            if self.full {
                template = file_type::CLASS
                    .sirt_scaled_output_template_old
                    .get_template(manager, axis_id);
                if let Some(template) = &template
                    && file_name.starts_with(template.as_str())
                {
                    return self.accept_template(file_name, template);
                }
            }
            if self.subarea {
                template = file_type::CLASS
                    .sirt_subarea_scaled_output_template_old
                    .get_template(manager, axis_id);
                if let Some(template) = &template
                    && file_name.starts_with(template.as_str())
                {
                    return self.accept_template(file_name, template);
                }
            }
        }
        false
    }

    /// Java private `acceptTemplate(String, String)` (deprecated).  The file name
    /// should be shorter then the template.  The part of the file name that
    /// extends beyond the template should be a valid integer.
    fn accept_template(&self, file_name: &str, template: &str) -> bool {
        let template_length = template.len();
        if file_name.len() <= template_length {
            return false;
        }
        // Upstream bug fixed in translation (SirtOutputFileFilter.java:171): Java
        // calls `Integer.getInteger(...)`, which looks up a system property and
        // never throws NumberFormatException, so every name longer than the
        // template was accepted.  The documented intent (a valid integer) is
        // `Integer.parseInt`, which is what is checked here.
        if java_lang_integer_parse_int(&file_name[template_length..]).is_err() {
            return false;
        }
        true
    }

    /// Java `getDescription()`.
    pub fn get_description(&self) -> &'static str {
        "SIRT Iteration Files"
    }
}

impl FileFilter for SirtOutputFileFilter {
    fn accept(&self, file: &Path) -> bool {
        SirtOutputFileFilter::accept_file(self, Some(file))
    }

    fn get_description(&self) -> Option<String> {
        Some(SirtOutputFileFilter::get_description(self).to_owned())
    }
}
