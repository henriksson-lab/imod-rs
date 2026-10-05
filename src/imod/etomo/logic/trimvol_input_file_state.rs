//! `IMOD/Etomo/src/etomo/logic/TrimvolInputFileState.java`.
//!
//! Holds information about the input file of trimvol.  In the case of post processing,
//! it also compares the input file to the post processing trimvol result and sets the
//! changed variables.
//!
//! If this is used with a non-post-processing trimvol, and new getInstance function
//! should be created which doesn't call setChanged.
//!
//! The `MRCHeader` n'ton hands out shared `std::sync::Arc<crate::imod::etomo::util::mrc_header::SharedMRCHeader>` handles (see
//! `util/mrc_header.rs`), so this class, like its Java original, lives on the thread
//! that created it.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `TrimvolInputFileState`.
pub struct TrimvolInputFileState {
    /// Java field `manager`.
    manager: &'static dyn BaseManager,
    /// Java field `axisID`.
    axis_id: AxisID,
    /// Java field `volumeFlipped`.
    volume_flipped: bool,
    /// Java field `nColumnsChanged`.
    n_columns_changed: bool,
    /// Java field `nRowsChanged`.
    n_rows_changed: bool,
    /// Java field `nSectionsChanged`.
    n_sections_changed: bool,
    /// Java field `changed`.
    changed: bool,
    /// Java field `mrcHeader`.
    mrc_header: Option<std::sync::Arc<crate::imod::etomo::util::mrc_header::SharedMRCHeader>>,
    /// Java field `inputFileMissing`.
    input_file_missing: bool,
}

impl TrimvolInputFileState {
    /// Java private `TrimvolInputFileState(BaseManager, AxisID, boolean)`.
    fn new(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        volume_flipped: bool,
    ) -> TrimvolInputFileState {
        TrimvolInputFileState {
            manager,
            axis_id,
            volume_flipped,
            n_columns_changed: false,
            n_rows_changed: false,
            n_sections_changed: false,
            changed: false,
            mrc_header: None,
            input_file_missing: false,
        }
    }

    /// Java `getPostProcessingInstance`.  Gets an instance of this class for
    /// post-processing.  Assumes that the trimvol input file is flipped.  If the file
    /// exists, sets changed member variables.
    ///
    /// The `IOException` / `InvalidParameterException` that `MRCHeader.read` throws
    /// arrive as one message, as [`MRCHeader::read_with_manager`] reports them.
    pub fn get_post_processing_instance(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        input_file_name: Option<&str>,
        state: &TomogramState,
    ) -> Result<TrimvolInputFileState, String> {
        let mut instance = TrimvolInputFileState::new(manager, axis_id, true);
        if instance.init_mrc_header(input_file_name)? {
            instance.set_changed(state);
        } else {
            instance.input_file_missing = true;
            ui_harness::post_message_dialog(
                Some(manager),
                format!("{} does not exist.", input_file_name.unwrap_or("null")),
                "Missing Input File".to_string(),
                Some(axis_id),
            );
        }
        Ok(instance)
    }

    /// Java private `initMrcHeader(String)`.  Returns false when files is not found.
    ///
    /// `MRCHeader.getInstance(String, String, AxisID)` never returns null in Java; the
    /// Rust n'ton returns `None` only for a null file name, which is treated as a file
    /// that is not found.
    fn init_mrc_header(&mut self, input_file_name: Option<&str>) -> Result<bool, String> {
        self.mrc_header = MRCHeader::get_instance_in_dir(
            self.manager.get_property_user_dir().as_deref(),
            input_file_name,
            Some(AxisID::Only),
        );
        let mrc_header = match &self.mrc_header {
            None => return Ok(false),
            Some(mrc_header) => mrc_header.clone(),
        };
        if !mrc_header.borrow_mut().read_with_manager(self.manager)? {
            return Ok(false);
        }
        Ok(true)
    }

    /// Java private `setChanged(TomogramState)`.  Compares the trimvol input file to the
    /// last post-processing trimvol input file.  The source compares the flipped row
    /// count with the stored section count and vice versa; that is kept.
    fn set_changed(&mut self, state: &TomogramState) {
        if !state.is_post_proc_trim_vol_input_n_columns_null()
            && self.mrc_header.as_ref().unwrap().borrow().get_n_columns()
                != state.get_post_proc_trim_vol_input_n_columns()
        {
            self.changed = true;
            self.n_columns_changed = true;
        }
        if !state.is_post_proc_trim_vol_input_n_sections_null()
            && self.get_n_rows() != state.get_post_proc_trim_vol_input_n_sections()
        {
            self.changed = true;
            self.n_rows_changed = true;
        }
        if !state.is_post_proc_trim_vol_input_n_rows_null()
            && self.get_n_sections() != state.get_post_proc_trim_vol_input_n_rows()
        {
            self.changed = true;
            self.n_sections_changed = true;
        }
    }

    /// Java `isInputFileMissing`.
    pub fn is_input_file_missing(&self) -> bool {
        self.input_file_missing
    }

    /// Java `isChanged`.
    pub fn is_changed(&self) -> bool {
        self.changed
    }

    /// Java `isNColumnsChanged`.
    pub fn is_n_columns_changed(&self) -> bool {
        self.n_columns_changed
    }

    /// Java `isNRowsChanged`.
    pub fn is_n_rows_changed(&self) -> bool {
        self.n_rows_changed
    }

    /// Java `isNSectionsChanged`.
    pub fn is_n_sections_changed(&self) -> bool {
        self.n_sections_changed
    }

    /// Java `getNColumns`.
    pub fn get_n_columns(&self) -> i32 {
        if let Some(mrc_header) = &self.mrc_header {
            return mrc_header.borrow().get_n_columns();
        }
        -1
    }

    /// Java `getNRows`.  Returns the number of rows, taking whether the volume is
    /// flipped into account.
    pub fn get_n_rows(&self) -> i32 {
        if let Some(mrc_header) = &self.mrc_header {
            if self.volume_flipped {
                return mrc_header.borrow().get_n_sections();
            }
            return mrc_header.borrow().get_n_rows();
        }
        -1
    }

    /// Java `getNSections`.  Returns the number of sections, taking whether the volume
    /// is flipped into account.
    pub fn get_n_sections(&self) -> i32 {
        if let Some(mrc_header) = &self.mrc_header {
            if self.volume_flipped {
                return mrc_header.borrow().get_n_rows();
            }
            return mrc_header.borrow().get_n_sections();
        }
        -1
    }

    /// Java `isVolumeFlipped`.
    pub fn is_volume_flipped(&self) -> bool {
        self.volume_flipped
    }
}
