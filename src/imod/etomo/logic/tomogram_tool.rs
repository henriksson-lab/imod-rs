//! `IMOD/Etomo/src/etomo/logic/TomogramTool.java`.
//!
//! Description: For comparing tomogram size to TomogramState.tomogramSize fields.
//!
//! Copyright: Copyright 2011
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado
//!
//! `MRCHeader.read`'s `IOException` and `InvalidParameterException` arrive as one
//! `Err(String)` (see `util/mrc_header.rs`); every site here catches both alike.
//! `UIHarness.INSTANCE.openMessageDialog(manager, message, title, axisID)` is
//! `ui_harness::post_message_dialog`.

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Type, java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;
use std::sync::Arc;

/// Java `rcsid` (an instance field in the source).
pub const RCSID: &str = "$Id:$";

/// Java `TomogramTool`, a class of static methods.
pub struct TomogramTool;

impl TomogramTool {
    /// Java static `isTomogramSizeChanged(ApplicationManager, boolean, AxisID)`.
    /// Returns true if any of the tomogram's dimensions listed in the header have
    /// changed.
    pub fn is_tomogram_size_changed(
        manager: &'static ApplicationManager,
        axis_id_first: bool,
        popup_axis_id: AxisID,
    ) -> bool {
        let axis_id = if axis_id_first {
            AxisID::First
        } else {
            AxisID::Second
        };
        let state = manager.get_state();
        let saved_tomogram_size_columns = state.get_tomogram_size_columns(axis_id);
        if saved_tomogram_size_columns.is_null() {
            // Backward compatibility - check the size based only on deprecated
            // tomogramSize.  This functionality should be removed eventually.
            let saved_tomogram_size = manager.get_state().get_tomogram_size(axis_id);
            if saved_tomogram_size.is_null() {
                return false;
            }
            return !manager
                .get_state()
                .get_tomogram_size(axis_id)
                .equals_long(TomogramTool::get_tomogram_size(manager, axis_id));
        }
        let tomogram_name = dataset_files::get_tomogram(manager, Some(axis_id)).map(|file| {
            file.file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default()
        });
        let header = MRCHeader::get_instance_from_file_name(
            manager,
            Some(popup_axis_id),
            tomogram_name.as_deref(),
        );
        // `getInstanceFromFileName` always answers an instance in the source.
        let header = match header {
            None => return false,
            Some(header) => header,
        };
        let mut header = header.borrow_mut();
        // `catch (IOException e) {}` and `catch (InvalidParameterException e) {}`.
        if let Ok(true) = header.read_with_manager(manager) {
            return !saved_tomogram_size_columns.equals_int(header.get_n_columns())
                || !state
                    .get_tomogram_size_rows(axis_id)
                    .equals_int(header.get_n_rows())
                || !state
                    .get_tomogram_size_sections(axis_id)
                    .equals_int(header.get_n_sections());
        }
        false
    }

    /// Java private static `getTomogramSize(BaseManager, AxisID)`.  Returns the physical
    /// size of the tomogram.
    fn get_tomogram_size(manager: &'static dyn BaseManager, axis_id: AxisID) -> i64 {
        let mut size: i64 = 0;
        // `new FileInputStream(file).getChannel().size()`; `FileNotFoundException` and
        // `IOException` leave the size at 0.
        if let Some(tomogram) = dataset_files::get_tomogram(manager, Some(axis_id))
            && let Ok(stream) = std::fs::File::open(&tomogram)
            && let Ok(metadata) = stream.metadata()
        {
            size = metadata.len() as i64;
        }
        size
    }

    /// Java static `saveTomogramSize(ApplicationManager, AxisID, AxisID)`.  Save the
    /// dimensions of the tomogram's dimensions listed in the header.
    pub fn save_tomogram_size(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        popup_axis_id: AxisID,
    ) {
        let state = manager.get_state();
        let tomogram_name = dataset_files::get_tomogram(manager, Some(axis_id)).map(|file| {
            file.file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default()
        });
        let header = MRCHeader::get_instance_from_file_name(
            manager,
            Some(popup_axis_id),
            tomogram_name.as_deref(),
        );
        let header = match header {
            None => return,
            Some(header) => header,
        };
        let mut header = header.borrow_mut();
        if let Ok(true) = header.read_with_manager(manager) {
            state.set_tomogram_size_columns(axis_id, header.get_n_columns());
            state.set_tomogram_size_rows(axis_id, header.get_n_rows());
            state.set_tomogram_size_sections(axis_id, header.get_n_sections());
        }
    }

    /// Java static `getYStartingSlice`.  Calculates the starting slice for a subarea
    /// tomogram.  If height or yShift is invalid, or the one of the results of the
    /// calculations is invalid, popups up an errror message and returns null.
    /// `y_height` - unbinned subarea tomogram height in Y; `y_shift` - unbinned Y shift
    /// of the subarea tomogram.  Returns starting slice, empty etomoNumber (empty params
    /// or missing aligned stack), or null (invalid params).
    pub fn get_y_starting_slice(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        y_height: Option<&str>,
        y_shift: Option<&str>,
        y_height_label: Option<&str>,
        y_shift_label: Option<&str>,
    ) -> Option<ConstEtomoNumber> {
        let mut en_starting_slice = EtomoNumber::new();
        if y_height.is_none_or(java_lang_string_matches_whitespace) {
            return Some(en_starting_slice.base);
        }
        let y_shift = if y_shift.is_none_or(java_lang_string_matches_whitespace) {
            "0.0"
        } else {
            y_shift.unwrap()
        };
        let header = MRCHeader::get_instance_from_file_name(
            manager,
            Some(axis_id),
            file_type::CLASS
                .aligned_stack
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
        );
        let header = match header {
            None => return Some(en_starting_slice.base),
            Some(header) => header,
        };
        let n_rows = {
            let mut header = header.borrow_mut();
            match header.read_with_manager(manager) {
                Ok(true) => {}
                Ok(false) | Err(_) => return Some(en_starting_slice.base),
            }
            header.get_n_rows()
        };
        // Get the unbinned tomogram height in Y of the aligned stack
        let tomo_height = n_rows.wrapping_mul(utilities::get_stack_binning_for_file_type(
            manager,
            axis_id,
            &file_type::CLASS.aligned_stack,
        ));
        let mut en_y_height = EtomoNumber::new();
        en_y_height.set_string(y_height);
        if !en_y_height.is_valid() {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid {} - {}",
                    y_height_label.unwrap_or("null"),
                    en_y_height.get_invalid_reason()
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        let mut en_y_shift = EtomoNumber::new_with_type(Some(Type::Double));
        en_y_shift.set_string(Some(y_shift));
        if !en_y_shift.is_valid() {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid {} - {}",
                    y_shift_label.unwrap_or("null"),
                    en_y_shift.get_invalid_reason()
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        en_starting_slice.set_double(
            (tomo_height.wrapping_sub(en_y_height.get_int()) / 2) as f64 - en_y_shift.get_double(),
        );
        if !en_starting_slice.is_valid() {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid starting slice - {}",
                    en_starting_slice.get_invalid_reason()
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        if en_starting_slice.lt_int(0) {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid starting slice: {}.  {} and/or {} are incorrect.",
                    en_starting_slice,
                    y_height_label.unwrap_or("null"),
                    y_shift_label.unwrap_or("null")
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        Some(en_starting_slice.base)
    }

    /// Java static `getStartingAndEndingXAndY`.
    ///
    /// startingx = (nx - sizex) / 2 - shiftx
    /// endingx = (nx + sizex) / 2 - shiftx - 1
    /// startingy = (ny - sizey) / 2 - shifty
    /// endingy = (ny + sizey) / 2 - shifty - 1
    #[allow(clippy::too_many_arguments)]
    pub fn get_starting_and_ending_x_and_y(
        file_type: &Arc<FileType>,
        size_x: Option<&str>,
        shift_x: Option<&str>,
        size_y: Option<&str>,
        shift_y: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        size_x_descr: Option<&str>,
        shift_x_descr: Option<&str>,
        size_y_descr: Option<&str>,
        shift_y_descr: Option<&str>,
        error_msg_title: Option<&str>,
    ) -> Option<PairXAndY> {
        let header = MRCHeader::get_instance_from_file_type(manager, Some(axis_id), file_type)?;
        let mut header = header.borrow_mut();
        // The result is ignored; only the exceptions return null.
        if header.read_with_manager(manager).is_err() {
            return None;
        }
        let mut pair_x_and_y = PairXAndY::new();
        // X
        let mut n = header.get_n_columns();
        let mut pair = TomogramTool::get_starting_and_ending(
            n,
            size_x,
            shift_x,
            manager,
            axis_id,
            size_x_descr,
            shift_x_descr,
            error_msg_title,
        );
        pair_x_and_y.set_x(pair);
        // Y
        n = header.get_n_rows();
        pair = TomogramTool::get_starting_and_ending(
            n,
            size_y,
            shift_y,
            manager,
            axis_id,
            size_y_descr,
            shift_y_descr,
            error_msg_title,
        );
        pair_x_and_y.set_y(pair);
        Some(pair_x_and_y)
    }

    /// Java private static `getStartingAndEnding`.
    ///
    /// starting = (n - size) / 2 - shift
    /// ending = (n + size) / 2 - shift - 1
    #[allow(clippy::too_many_arguments)]
    fn get_starting_and_ending(
        file_n: i32,
        size_string: Option<&str>,
        shift_string: Option<&str>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        size_descr: Option<&str>,
        shift_descr: Option<&str>,
        error_msg_title: Option<&str>,
    ) -> Option<[i32; 2]> {
        let mut size = EtomoNumber::new();
        size.set_string(size_string);
        let mut error_msg = size.validate(size_descr);
        if let Some(error_msg) = error_msg {
            ui_harness::post_message_dialog(
                Some(manager),
                error_msg,
                error_msg_title.unwrap_or("null").to_string(),
                Some(axis_id),
            );
            return None;
        }
        if size.is_null() {
            size.set_int(file_n);
        }
        let mut shift = EtomoNumber::new_with_type(Some(Type::Double));
        shift.set_string(shift_string);
        error_msg = shift.validate(shift_descr);
        if let Some(error_msg) = error_msg {
            ui_harness::post_message_dialog(
                Some(manager),
                error_msg,
                error_msg_title.unwrap_or("null").to_string(),
                Some(axis_id),
            );
            return None;
        }
        if shift.is_null() {
            shift.set_int(0);
        }
        let starting = utilities::java_lang_math_round(
            (file_n.wrapping_sub(size.get_int()) / 2) as f64 - shift.get_double(),
        ) as i32;
        let ending = utilities::java_lang_math_round(
            (file_n.wrapping_add(size.get_int()) / 2) as f64 - shift.get_double() - 1.0,
        ) as i32;
        Some([starting, ending])
    }

    /// Java static `getYEndingSlice`.  Calculates the ending slice for a subarea
    /// tomogram.  If height is invalid, or the one of the results of the calculations
    /// is invalid, popups up an errror message and returns null.  `starting_slice` -
    /// assumes this parameter is a valid number; `y_height` - unbinned subarea tomogram
    /// height in Y.  Returns ending slice, empty etomoNumber (empty params or missing
    /// aligned stack), or null (invalid params).
    pub fn get_y_ending_slice(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        starting_slice: &ConstEtomoNumber,
        y_height: Option<&str>,
        y_height_label: Option<&str>,
    ) -> Option<ConstEtomoNumber> {
        let mut en_ending_slice = EtomoNumber::new();
        if starting_slice.is_null() || y_height.is_none_or(java_lang_string_matches_whitespace) {
            return Some(en_ending_slice.base);
        }
        let header = MRCHeader::get_instance_from_file_name(
            manager,
            Some(axis_id),
            file_type::CLASS
                .aligned_stack
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
        );
        let header = match header {
            None => return Some(en_ending_slice.base),
            Some(header) => header,
        };
        let n_rows = {
            let mut header = header.borrow_mut();
            match header.read_with_manager(manager) {
                Ok(true) => {}
                Ok(false) | Err(_) => return Some(en_ending_slice.base),
            }
            header.get_n_rows()
        };
        // Get the unbinned tomogram height in Y of the aligned stack
        let tomo_height = n_rows.wrapping_mul(utilities::get_stack_binning_for_file_type(
            manager,
            axis_id,
            &file_type::CLASS.aligned_stack,
        ));
        let mut en_y_height = EtomoNumber::new();
        en_y_height.set_string(y_height);
        if !en_y_height.is_valid() {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid {} - {}",
                    y_height_label.unwrap_or("null"),
                    en_y_height.get_invalid_reason()
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        en_ending_slice.set_int(
            starting_slice
                .get_int()
                .wrapping_add(en_y_height.get_int())
                .wrapping_sub(1),
        );
        if !en_ending_slice.is_valid() {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid starting slice - {}",
                    en_ending_slice.get_invalid_reason()
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        if en_ending_slice.ge_long(tomo_height as i64) {
            ui_harness::post_message_dialog(
                Some(manager),
                format!(
                    "Invalid starting slice: {}.  Check {}.",
                    en_ending_slice,
                    y_height_label.unwrap_or("null")
                ),
                "Entry Error".to_string(),
                Some(axis_id),
            );
            return None;
        }
        Some(en_ending_slice.base)
    }

    /// Java static `getYHeightAndShift`.  Calculated the subarea y height and slice
    /// from the starting and ending slices.  Returns pair of ints (height, shift) or
    /// null if aligned stack is missing.
    pub fn get_y_height_and_shift(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        y_starting_slice: i32,
        y_ending_slice: i32,
    ) -> Option<[i32; 2]> {
        let height = y_ending_slice
            .wrapping_add(1)
            .wrapping_sub(y_starting_slice);
        let header = MRCHeader::get_instance_from_file_name(
            manager,
            Some(axis_id),
            file_type::CLASS
                .aligned_stack
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
        )?;
        let n_rows = {
            let mut header = header.borrow_mut();
            match header.read_with_manager(manager) {
                Ok(true) => {}
                Ok(false) | Err(_) => return None,
            }
            header.get_n_rows()
        };
        // Get the unbinned tomogram height in Y of the aligned stack
        let tomo_height = n_rows.wrapping_mul(utilities::get_stack_binning_for_file_type(
            manager,
            axis_id,
            &file_type::CLASS.aligned_stack,
        ));
        let shift = (tomo_height.wrapping_sub(height) / 2).wrapping_sub(y_starting_slice);
        Some([height, shift])
    }

    /// Java static `convertShiftsToOffsets(String, String, boolean)`.  Negates shiftX
    /// and shiftY.  When colornewst is true and only one value is set, the other one is
    /// set to zero.
    pub fn convert_shifts_to_offsets(
        shift_x: Option<&str>,
        shift_y: Option<&str>,
        colornewst: bool,
    ) -> [ConstEtomoNumber; 2] {
        let mut offsets = [
            EtomoNumber::new_with_type(Some(Type::Double)),
            EtomoNumber::new_with_type(Some(Type::Double)),
        ];
        let mut index = 0;
        let mut x_empty = false;
        offsets[index].set_string(shift_x);
        if !offsets[index].is_null() {
            if offsets[index].is_valid() {
                let negated = offsets[index].get_double() * -1.0;
                offsets[index].set_double(negated);
            }
        } else {
            x_empty = true;
        }
        index = 1;
        let mut y_empty = false;
        offsets[index].set_string(shift_y);
        if !offsets[index].is_null() {
            if offsets[index].is_valid() {
                let negated = offsets[index].get_double() * -1.0;
                offsets[index].set_double(negated);
            }
        } else {
            y_empty = true;
        }
        if colornewst {
            if x_empty && !y_empty {
                offsets[0].set_int(0);
            } else if y_empty && !x_empty {
                offsets[1].set_int(0);
            }
        }
        let [offset_x, offset_y] = offsets;
        [offset_x.base, offset_y.base]
    }

    /// Java static `convertOffsetToShift(String)`.  Negates offset.
    pub fn convert_offset_to_shift(offset: Option<&str>) -> ConstEtomoNumber {
        let mut shift = EtomoNumber::new_with_type(Some(Type::Double));
        shift.set_string(offset);
        if !shift.is_null() && shift.is_valid() {
            let negated = shift.get_double() * -1.0;
            shift.set_double(negated);
        }
        shift.base
    }
}

/// Java public static final nested class `TomogramTool.PairXAndY`.  Handle a pair of X
/// and a pair of Y integer values.  Drops extra values when setting.  Each pair will
/// always contain either 2 null values or 2 non-null values.
#[derive(Clone, Debug)]
pub struct PairXAndY {
    /// Java private final field `pairX`.
    pair_x: [EtomoNumber; 2],
    /// Java private final field `pairY`.
    pair_y: [EtomoNumber; 2],
}

impl PairXAndY {
    /// Java private constructor `PairXAndY()`.
    fn new() -> PairXAndY {
        PairXAndY {
            pair_x: [EtomoNumber::new(), EtomoNumber::new()],
            pair_y: [EtomoNumber::new(), EtomoNumber::new()],
        }
    }

    /// Java private `setX(int[])`.  Calls set.
    fn set_x(&mut self, pair: Option<[i32; 2]>) {
        PairXAndY::set(pair, &mut self.pair_x);
    }

    /// Java private `setY(int[])`.  Calls set.
    fn set_y(&mut self, pair: Option<[i32; 2]>) {
        PairXAndY::set(pair, &mut self.pair_y);
    }

    /// Java private `set(int[], EtomoNumber[])`.  Sets toPair.  A null fromPair or a
    /// fromPair containing only one non-null number causes a reset of toPair.
    fn set(from_pair: Option<[i32; 2]>, to_pair: &mut [EtomoNumber; 2]) {
        let mut i = 0;
        if let Some(from_pair) = from_pair {
            to_pair[i].set_int(from_pair[i]);
            i += 1;
            to_pair[i].set_int(from_pair[i]);
            if to_pair[0].is_null() {
                to_pair[1].reset();
            } else if to_pair[1].is_null() {
                to_pair[0].reset();
            }
        } else {
            to_pair[i].reset();
            i += 1;
            to_pair[i].reset();
        }
    }

    /// Java `isXNull`.
    pub fn is_x_null(&self) -> bool {
        self.pair_x[0].is_null()
    }

    /// Java `isYNull`.
    pub fn is_y_null(&self) -> bool {
        self.pair_y[0].is_null()
    }

    /// Java `getFirstX`.  Returns pairX[0].  May return a null value.
    pub fn get_first_x(&self) -> i32 {
        self.pair_x[0].get_int()
    }

    /// Java `getSecondX`.  Returns pairX[1].  May return a null value.
    pub fn get_second_x(&self) -> i32 {
        self.pair_x[1].get_int()
    }

    /// Java `getFirstY`.  Returns pairY[0].  May return a null value.
    pub fn get_first_y(&self) -> i32 {
        self.pair_y[0].get_int()
    }

    /// Java `getSecondY`.  Returns pairY[1].  May return a null value.
    pub fn get_second_y(&self) -> i32 {
        self.pair_y[1].get_int()
    }
}
