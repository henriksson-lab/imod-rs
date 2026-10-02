//! `IMOD/Etomo/src/etomo/storage/ExcludeViewsInfoFileNameFilter.java`.
//!
//! `implements java.io.FilenameFilter`: accepts the `<dataset><axis>_cutviews<N>.info`
//! files written by previous excludeviews runs (N is one or two digits).  Held by its
//! caller on the event dispatch thread; `setAxisID` takes `&self`.

use std::cell::Cell;
use std::path::Path;

use regex::Regex;

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;

/// Java `ExcludeViewsInfoFileNameFilter`.
pub struct ExcludeViewsInfoFileNameFilter {
    /// Java private final `datasetName`.
    dataset_name: Option<String>,
    /// Java private `axisID`, initialised to null.
    axis_id: Cell<Option<AxisID>>,
}

impl ExcludeViewsInfoFileNameFilter {
    /// Java `ExcludeViewsInfoFileNameFilter(String)`.
    pub fn new(dataset_name: Option<&str>) -> ExcludeViewsInfoFileNameFilter {
        ExcludeViewsInfoFileNameFilter {
            dataset_name: dataset_name.map(|dataset_name| dataset_name.to_string()),
            axis_id: Cell::new(None),
        }
    }

    /// Java `setAxisID(AxisID)`.
    pub fn set_axis_id(&self, axis_id: AxisID) {
        self.axis_id.set(Some(axis_id));
    }

    /// Java `accept(File, String)`.  Java's `\Q...\E` quoting becomes
    /// `regex::escape`, and `String.matches` is a whole-string match.
    pub fn accept(&self, _file: &Path, name: Option<&str>) -> bool {
        let Some(name) = name else {
            return false;
        };
        let file_type = &file_type::CLASS.exclude_views_info;
        let left_reg_exp = regex::escape(&format!(
            "{}{}{}",
            self.dataset_name.as_deref().unwrap_or("null"),
            match self.axis_id.get() {
                Some(axis_id) => axis_id.get_extension(),
                None => String::new(),
            },
            file_type.get_type_string().unwrap_or("null")
        ));
        let right_reg_exp = regex::escape(file_type.get_extension().unwrap_or("null"));
        // midzone2a_cutviews2.info
        if Regex::new(&format!("^(?:{}[0-9]{})$", left_reg_exp, right_reg_exp))
            .unwrap()
            .is_match(name)
        {
            return true;
        }
        // midzone2b_cutviews10.info
        if Regex::new(&format!(
            "^(?:{}[0-9][0-9]{})$",
            left_reg_exp, right_reg_exp
        ))
        .unwrap()
        .is_match(name)
        {
            return true;
        }
        false
    }
}
