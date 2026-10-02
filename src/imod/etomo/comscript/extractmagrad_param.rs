//! `IMOD/Etomo/src/etomo/comscript/ExtractmagradParam.java`.
//!
//! Builds the `extractmagrad -rot R -grad G <stack> <maggrad>` command line.

use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "extractmagrad";

/// Java `ExtractmagradParam`.
pub struct ExtractmagradParam {
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
    rotation_angle: EtomoNumber,
    gradient_table: Option<String>,
    /// Built on first use by `getCommand`, which the process manager calls
    /// through a shared reference.
    command_array: std::sync::Mutex<Option<Vec<String>>>,
    pixel_size: Option<EtomoNumber>,
}

impl ExtractmagradParam {
    /// Java `ExtractmagradParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> ExtractmagradParam {
        ExtractmagradParam {
            axis_id,
            manager,
            rotation_angle: EtomoNumber::new_with_type(Some(Type::Double)),
            gradient_table: None,
            command_array: std::sync::Mutex::new(None),
            pixel_size: None,
        }
    }

    /// Java `getCommand()`.
    pub fn get_command(&self) -> Vec<String> {
        if self.command_array.lock().unwrap().is_none() {
            self.build_command();
        }
        self.command_array
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default()
    }

    /// Java private `buildCommand()`.
    fn build_command(&self) {
        let mut command: Vec<String> = Vec::new();
        command.push(format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_string()),
            COMMAND_NAME
        ));
        command.push("-rot".to_string());
        command.push(self.rotation_angle.to_string());
        command.push("-grad".to_string());
        // A null gradient table is a null element in Java's list; "null" here.
        command.push(
            self.gradient_table
                .clone()
                .unwrap_or_else(|| "null".to_string()),
        );
        // Java assigns `dataset = manager.getName()` and never reads it.
        let _dataset = self.manager.get_name();
        command.push(
            dataset_files::get_stack_name(self.manager, Some(self.axis_id))
                .unwrap_or_else(|| "null".to_string()),
        );
        command.push(dataset_files::get_mag_gradient_name(
            self.manager,
            Some(self.axis_id),
        ));
        let command_size = command.len();
        let mut command_array = vec![String::new(); command_size];
        for i in 0..command_size {
            command_array[i] = command[i].clone();
        }
        *self.command_array.lock().unwrap() = Some(command_array);
    }

    /// Java `setGradientTable(String)`.
    pub fn set_gradient_table(&mut self, gradient_table: Option<&str>) {
        self.gradient_table = gradient_table.map(|s| s.to_string());
    }

    /// Java `setRotationAngle(ConstEtomoNumber)`.
    pub fn set_rotation_angle(&mut self, rotation_angle: Option<&ConstEtomoNumber>) {
        self.rotation_angle.set_const_etomo_number(rotation_angle);
    }

    /// Java `setPixelSize(double)`.  Records the pixel size only when the stack's header
    /// pixel spacing is 1.  Note that the source never reads the header first, and
    /// never places `pixelSize` on the command line.
    pub fn set_pixel_size(&mut self, pixel_size: f64) {
        let header = MRCHeader::get_instance_in_dir(
            self.manager.get_property_user_dir().as_deref(),
            dataset_files::get_stack_name(self.manager, Some(self.axis_id)).as_deref(),
            Some(self.axis_id),
        );
        let x_pixel_spacing = match &header {
            Some(header) => header.borrow().get_x_pixel_spacing(),
            None => return,
        };
        if x_pixel_spacing == 1.0 {
            if self.pixel_size.is_none() {
                self.pixel_size = Some(EtomoNumber::new_with_type(Some(Type::Double)));
            }
            self.pixel_size.as_mut().unwrap().set_double(pixel_size);
        }
    }
}
