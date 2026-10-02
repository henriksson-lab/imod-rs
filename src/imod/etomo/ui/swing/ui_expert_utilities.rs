//! `IMOD/Etomo/src/etomo/ui/swing/UIExpertUtilities.java`.
//!
//! Java `public final class UIExpertUtilities`, a stateless singleton
//! (`INSTANCE`).  Its methods are called on the event dispatch thread by the
//! experts and `ApplicationManager`, and from com-script parameter classes.
//!
//! Overloaded Java methods carry their parameter types as a suffix
//! (`getStackBinning(BaseManager, AxisID, FileType)` ->
//! `get_stack_binning_base_manager_axis_id_file_type`).

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    INTEGER_NULL_VALUE, Type, java_lang_double_to_string, java_lang_double_value_of,
    java_lang_string_matches_whitespace,
};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::ui::swing::fiducialess_params::FiducialessParams;
use crate::imod::etomo::ui::swing::tomogram_positioning_expert::TomogramPositioningExpert;
use crate::imod::etomo::ui::swing::ui_harness;
// TODO(unit): needs etomo/util/FidXyz.java - `FidXyz` has no module yet.
use crate::imod::etomo::util::fid_xyz::FidXyz;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities::java_lang_math_round;
use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;

/// Java `UIExpertUtilities`.
#[derive(Clone, Copy, Debug)]
pub struct UIExpertUtilities;

impl UIExpertUtilities {
    /// Java `public static final UIExpertUtilities INSTANCE`.
    pub const INSTANCE: UIExpertUtilities = UIExpertUtilities;

    /// Java `getStackBinning(BaseManager, AxisID, String)` (deprecated
    /// 5/9/2019).  Gets the binning that can be used to run tilt against a
    /// stack (.preali or .ali).  Function calculates the binning from the
    /// stack's pixel spacing and the raw stack's pixel spacing.
    pub fn get_stack_binning_base_manager_axis_id_string(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        stack_extension: Option<&str>,
    ) -> i32 {
        self.get_stack_binning_base_manager_axis_id_string_boolean(
            manager,
            axis_id,
            stack_extension,
            false,
        )
    }

    /// Java `getStackBinning(BaseManager, AxisID, String, boolean)` (deprecated
    /// 5/9/2019).  The Java passes `false` rather than `nullIfFailed` on to
    /// the header overload; kept as written.
    pub fn get_stack_binning_base_manager_axis_id_string_boolean(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        stack_extension: Option<&str>,
        _null_if_failed: bool,
    ) -> i32 {
        self.get_stack_binning_base_manager_axis_id_mrc_header_boolean(
            manager,
            axis_id,
            MRCHeader::get_instance_from_manager(manager, Some(axis_id), stack_extension),
            false,
        )
    }

    /// Java `getStackBinning(BaseManager, AxisID, FileType)`.
    pub fn get_stack_binning_base_manager_axis_id_file_type(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        stack_file_type: &Arc<FileType>,
    ) -> i32 {
        let property_user_dir = manager.get_property_user_dir();
        let file_name = stack_file_type.get_file_name(Some(manager), Some(axis_id));
        self.get_stack_binning_base_manager_axis_id_mrc_header_boolean(
            manager,
            axis_id,
            MRCHeader::get_instance_in_dir(
                property_user_dir.as_deref(),
                file_name.as_deref(),
                Some(axis_id),
            ),
            false,
        )
    }

    /// Java `getStackBinning(BaseManager, AxisID, FileType, boolean)`.
    pub fn get_stack_binning_base_manager_axis_id_file_type_boolean(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        stack_file_type: &Arc<FileType>,
        null_if_failed: bool,
    ) -> i32 {
        let property_user_dir = manager.get_property_user_dir();
        let file_name = stack_file_type.get_file_name(Some(manager), Some(axis_id));
        self.get_stack_binning_base_manager_axis_id_mrc_header_boolean(
            manager,
            axis_id,
            MRCHeader::get_instance_in_dir(
                property_user_dir.as_deref(),
                file_name.as_deref(),
                Some(axis_id),
            ),
            null_if_failed,
        )
    }

    /// Java `getStackBinningFromFileName(BaseManager, AxisID, String,
    /// boolean)`.  Gets the binning that can be used to run tilt against a
    /// stack (.preali, _3dfind.ali, or .ali).  Like the Java, `nullIfFailed`
    /// is not passed on.
    pub fn get_stack_binning_from_file_name(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        file_name: Option<&str>,
        _null_if_failed: bool,
    ) -> i32 {
        let Some(file_name) =
            file_name.filter(|file_name| !java_lang_string_matches_whitespace(file_name))
        else {
            return 1;
        };
        let property_user_dir = manager.get_property_user_dir();
        self.get_stack_binning_base_manager_axis_id_mrc_header_boolean(
            manager,
            axis_id,
            MRCHeader::get_instance_in_dir(
                property_user_dir.as_deref(),
                Some(file_name),
                Some(axis_id),
            ),
            false,
        )
    }

    /// Java `getStackBinning(BaseManager, AxisID, MRCHeader, boolean)`.  Gets
    /// the binning that can be used to run tilt against a stack (.preali or
    /// .ali).  When `nullIfFailed` is true it returns
    /// `EtomoNumber.INTEGER_NULL_VALUE` on failure, otherwise 1.
    ///
    /// `MRCHeader.getInstance` never returns null in the Java; the Rust
    /// constructors return `None` where the Java would build a header for a
    /// null path, and that is treated as the failed read it would become.
    pub fn get_stack_binning_base_manager_axis_id_mrc_header_boolean(
        &self,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        stack_header: Option<Rc<RefCell<MRCHeader>>>,
        null_if_failed: bool,
    ) -> i32 {
        let rawstack_header = MRCHeader::get_instance_from_file_type(
            manager,
            Some(axis_id),
            &file_type::CLASS.raw_stack,
        );
        let default_value = if null_if_failed {
            INTEGER_NULL_VALUE
        } else {
            1
        };
        let (Some(rawstack_header), Some(stack_header)) = (rawstack_header, stack_header) else {
            return default_value;
        };
        // `catch (InvalidParameterException e)` prints the stack trace (missing
        // file); `catch (IOException e)` does not.  `read_with_manager`
        // reports both as `Err`.
        let raw_read = rawstack_header.borrow_mut().read_with_manager(manager);
        match raw_read {
            Ok(false) => return default_value,
            Err(e) => {
                eprintln!("{e}");
                return default_value;
            }
            Ok(true) => {}
        }
        let stack_read = stack_header.borrow_mut().read_with_manager(manager);
        match stack_read {
            Ok(false) => return default_value,
            Err(e) => {
                eprintln!("{e}");
                return default_value;
            }
            Ok(true) => {}
        }
        let mut binning = default_value;
        let rawstack_x_pixel_spacing = rawstack_header.borrow().get_x_pixel_spacing();
        if rawstack_x_pixel_spacing > 0.0 {
            binning = java_lang_math_round(
                stack_header.borrow().get_x_pixel_spacing() / rawstack_x_pixel_spacing,
            ) as i32;
        }
        if binning != default_value && binning < 1 {
            return 1;
        }
        binning
    }

    /// Java `updateFiducialessParams(ApplicationManager, FiducialessParams,
    /// AxisID, boolean)`: get the fiducialess parameters from the specified
    /// dialog and set the metaData and rotationXF script.
    pub fn update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
        &self,
        manager: &'static ApplicationManager,
        dialog: &dyn FiducialessParams,
        axis_id: AxisID,
        do_validation: bool,
    ) -> bool {
        // The Java's inner `catch (NumberFormatException)` around this call
        // cannot be reached: the overload below catches its own parse error,
        // and `getImageRotation` declares only
        // `FieldValidationFailedException`.
        match dialog.get_image_rotation(do_validation) {
            Ok(image_rotation) => self
                .update_fiducialess_params_application_manager_string_boolean_axis_id(
                    manager,
                    &image_rotation,
                    dialog.is_fiducialess(),
                    axis_id,
                ),
            Err(_) => false,
        }
    }

    /// Java `updateFiducialessParams(ApplicationManager, String, boolean,
    /// AxisID)`: set the metaData and rotationXF script.
    pub fn update_fiducialess_params_application_manager_string_boolean_axis_id(
        &self,
        manager: &'static ApplicationManager,
        image_rotation: &str,
        fiducialess: bool,
        axis_id: AxisID,
    ) -> bool {
        if java_lang_string_matches_whitespace(image_rotation) {
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                "Missing tilt axis rotation value.  Make sure that the aligned stack has been created.",
                "Missing File",
                Some(axis_id),
            );
            return false;
        }
        let tilt_axis_angle = match java_lang_double_value_of(image_rotation) {
            Ok(tilt_axis_angle) => tilt_axis_angle,
            Err(message) => {
                let error_message = vec!["Tilt axis rotation format error".to_owned(), message];
                ui_harness::open_message_dialog_array_from_process(
                    Some(manager),
                    &error_message,
                    "Tilt axis rotation syntax error",
                    Some(axis_id),
                );
                return false;
            }
        };
        manager
            .get_meta_data()
            .set_fiducialess_alignment(axis_id, fiducialess);
        manager
            .get_meta_data()
            .set_image_rotation(Some(&java_lang_double_to_string(tilt_axis_angle)), axis_id);
        self.update_rotation_xf(
            manager,
            manager.get_property_user_dir().as_deref(),
            &java_lang_double_to_string(tilt_axis_angle),
            axis_id,
        );
        true
    }

    /// Java private `updateRotationXF(BaseManager, String, String, AxisID)`:
    /// write out the rotation transform for the specified axis.
    fn update_rotation_xf(
        &self,
        manager: &'static dyn BaseManager,
        property_user_dir: Option<&str>,
        angle: &str,
        axis_id: AxisID,
    ) {
        // Open the appropriate rotation file
        let fn_rotation_xf = format!(
            "{}{}rotation{}.xf",
            property_user_dir.unwrap_or("null"),
            std::path::MAIN_SEPARATOR,
            axis_id.get_extension()
        );
        let result = (|| -> std::io::Result<()> {
            let mut out = std::io::BufWriter::new(std::fs::File::create(&fn_rotation_xf)?);
            // Write out the transform to perform the rotation
            let mut n_angle = EtomoNumber::new_with_type(Some(Type::Double));
            n_angle.set_string(Some(angle));
            let rads = -1.0 * n_angle.get_double() * std::f64::consts::PI / 180.0;
            out.write_all(
                format!(
                    "{}   {}   {}   {}   0   0",
                    java_lang_double_to_string(rads.cos()),
                    java_lang_double_to_string((-rads).sin()),
                    java_lang_double_to_string(rads.sin()),
                    java_lang_double_to_string(rads.cos())
                )
                .as_bytes(),
            )?;
            out.write_all(b"\n")?;
            // Close the file
            out.flush()
        })();
        if let Err(except) = result {
            let error_message = vec![
                "Rotation Transform IO Exception".to_owned(),
                except.to_string(),
                fn_rotation_xf,
            ];
            ui_harness::open_message_dialog_array_from_process(
                Some(manager),
                &error_message,
                "Rotation Transform IO Exception",
                Some(axis_id),
            );
        }
    }

    /// Java `areScriptsCreated(BaseManager, ConstMetaData, AxisID)`.
    pub fn are_scripts_created(
        &self,
        manager: &'static dyn BaseManager,
        meta_data: &MetaData,
        axis_id: AxisID,
    ) -> bool {
        if meta_data.get_com_script_created() {
            return true;
        }
        let message = vec![
            "The setup process has not been completed".to_owned(),
            "Complete the Setup process before opening other process dialogs".to_owned(),
        ];
        ui_harness::open_message_dialog_array_from_process(
            Some(manager),
            &message,
            "Program Operation Error",
            Some(axis_id),
        );
        false
    }

    /// Java `upgradeOldAlignCom(ApplicationManager, AxisID, TiltalignParam)`:
    /// if tiltalignParam is an old version, upgrade it to the new version and
    /// save it to align.com.
    pub fn upgrade_old_align_com(
        &self,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        tiltalign_param: &mut TiltalignParam,
    ) {
        if !tiltalign_param.is_old_version() {
            return;
        }
        let correction_binning = self.get_backward_compatible_align_binning(manager, axis_id);
        let current_binning = self.get_stack_binning_base_manager_axis_id_file_type(
            manager,
            axis_id,
            &file_type::CLASS.prealigned_stack,
        );
        if tiltalign_param.upgrade_old_version(correction_binning, current_binning) {
            self.roll_align_com_angles(manager, axis_id);
            manager
                .get_com_script_manager()
                .save_align(tiltalign_param, axis_id);
        }
    }

    /// Java `upgradeOldTiltCom(ApplicationManager, AxisID, TiltParam)`: if
    /// tiltParam is an old version, upgrade it to the new version and save it
    /// to tilt.com.
    pub fn upgrade_old_tilt_com(
        &self,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        tilt_param: &mut TiltParam,
    ) {
        if !tilt_param.is_old_version() {
            return;
        }
        let correction_binning =
            self.get_backward_compatible_tilt_binning(manager, axis_id, tilt_param);
        let current_binning = self.get_stack_binning_base_manager_axis_id_file_type(
            manager,
            axis_id,
            &file_type::CLASS.aligned_stack,
        );
        if tilt_param.upgrade_old_version(correction_binning, current_binning) {
            self.roll_tilt_com_angles(manager, axis_id);
            manager
                .get_com_script_manager()
                .save_tilt(tilt_param, axis_id);
            manager
                .get_meta_data()
                .set_fiducialess(axis_id, tilt_param.is_fiducialess());
        }
    }

    /// Java private `getBackwardCompatibleAlignBinning(ApplicationManager,
    /// AxisID)`.  Gets the binning that can be used to repair a older
    /// align.com file, from the raw stack pixel spacing and the fid.xyz pixel
    /// spacing (if it exists) or the .preali pixel spacing.
    fn get_backward_compatible_align_binning(
        &self,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
    ) -> i32 {
        let Some(rawstack_header) = MRCHeader::get_instance_from_file_type(
            manager,
            Some(axis_id),
            &file_type::CLASS.raw_stack,
        ) else {
            return 1;
        };
        let raw_read = rawstack_header.borrow_mut().read_with_manager(manager);
        match raw_read {
            Ok(false) => return 1,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
            Ok(true) => {}
        }
        let mut fid_xyz = self.get_fid_xyz(manager, axis_id);
        let mut fid_xyz_failed = false;
        if let Err(e) = fid_xyz.read() {
            eprintln!("{e}");
            fid_xyz_failed = true;
        }
        if !fid_xyz_failed {
            // Another layer of backward compatibility. Handle the time before fid.xyz
            // files where created. This also handles the situation where align.com has
            // not been run and has the original values from copytomocoms.
            if !fid_xyz.exists() {
                return 1;
            }
            // Align.com must have failed. The fallback is to use pixel spacing from
            // .preali.
            if fid_xyz.is_empty() {
                fid_xyz_failed = true;
            }
            // Another layer of backward compatibility. There was a small period of
            // time when binning existed but the pixel spacing was not added to fid.xyz
            // (3.2.7 (3/16/04) - 3.2.20 (6/19/04). If this fid.xyz was created before
            // binning we could return 1, but we don't know that and we can't trust
            // change times on old files that could have been copied, so we need to use
            // the fallback in this case.
            else if !fid_xyz.is_pixel_size_set() {
                fid_xyz_failed = true;
            }
        }
        let rawstack_x_pixel_spacing = rawstack_header.borrow().get_x_pixel_spacing();
        // Unable to calculate binning
        if rawstack_x_pixel_spacing <= 0.0 {
            return 1;
        }
        // calculate binning from fid.xyz
        if !fid_xyz_failed {
            let binning =
                java_lang_math_round(fid_xyz.get_pixel_size() / rawstack_x_pixel_spacing) as i32;
            if binning < 1 {
                return 1;
            }
            return binning;
        }
        // fallback to .preali
        let Some(stack_header) = MRCHeader::get_instance_from_file_type(
            manager,
            Some(axis_id),
            &file_type::CLASS.prealigned_stack,
        ) else {
            return 1;
        };
        let stack_read = stack_header.borrow_mut().read_with_manager(manager);
        match stack_read {
            Ok(false) => return 1,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
            Ok(true) => {}
        }
        let binning = java_lang_math_round(
            stack_header.borrow().get_x_pixel_spacing() / rawstack_x_pixel_spacing,
        ) as i32;
        if binning < 1 {
            return 1;
        }
        binning
    }

    /// Java private `getBackwardCompatibleTiltBinning(ApplicationManager,
    /// AxisID, ConstTiltParam)`.  Calculates the binning from the raw stack
    /// full image size and the full image size in tilt.com.
    fn get_backward_compatible_tilt_binning(
        &self,
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        tilt_param: &TiltParam,
    ) -> i32 {
        let Some(rawstack_header) = MRCHeader::get_instance_from_file_type(
            manager,
            Some(axis_id),
            &file_type::CLASS.raw_stack,
        ) else {
            return 1;
        };
        let raw_read = rawstack_header.borrow_mut().read_with_manager(manager);
        match raw_read {
            Ok(false) => return 1,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
            Ok(true) => {}
        }
        let mut binning = 1;
        let tilt_full_image_x = tilt_param.get_full_image_x();
        // defaults to Integer.MIN_VALUE in ConstTiltParam
        if tilt_full_image_x > 0 {
            // `Math.round(int / int)`: an int quotient, rounded as a float, which
            // leaves it unchanged.
            binning = rawstack_header.borrow().get_n_columns() / tilt_full_image_x;
        }
        if binning < 1 {
            return 1;
        }
        binning
    }

    /// Java `getFidXyz(ApplicationManager, AxisID)`.
    pub fn get_fid_xyz(&self, manager: &'static ApplicationManager, axis_id: AxisID) -> FidXyz {
        FidXyz::new(
            manager.get_property_user_dir().as_deref(),
            &format!(
                "{}{}fid.xyz",
                manager.get_meta_data().get_dataset_name(),
                axis_id.get_extension()
            ),
        )
    }

    /// Java `rollAlignComAngles(ApplicationManager, AxisID)`.
    pub fn roll_align_com_angles(&self, manager: &'static ApplicationManager, axis_id: AxisID) {
        let expert = manager.get_ui_expert(Some(DialogType::TomogramPositioning), axis_id);
        if let Some(expert) = expert
            && let Some(expert) = expert.as_any().downcast_ref::<TomogramPositioningExpert>()
        {
            expert.roll_align_com_angles();
        }
    }

    /// Java `rollTiltComAngles(ApplicationManager, AxisID)`.
    pub fn roll_tilt_com_angles(&self, manager: &'static ApplicationManager, axis_id: AxisID) {
        let expert = manager.get_ui_expert(Some(DialogType::TomogramPositioning), axis_id);
        if let Some(expert) = expert
            && let Some(expert) = expert.as_any().downcast_ref::<TomogramPositioningExpert>()
        {
            expert.roll_tilt_com_angles();
        }
    }
}
