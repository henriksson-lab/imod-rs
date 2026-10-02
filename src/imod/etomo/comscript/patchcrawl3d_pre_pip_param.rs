//! `IMOD/Etomo/src/etomo/comscript/Patchcrawl3DPrePIPParam.java`.
//!
//! `Patchcrawl3DPrePIPParam extends ConstPatchcrawl3DPrePIPParam`: the superclass
//! state is the `base` field, reached through `Deref`/`DerefMut`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_patchcrawl3d_pre_pip_param::ConstPatchcrawl3DPrePIPParam;
use super::fortran_input_syntax_exception::FortranInputSyntaxException;
use super::matchorwarp_param::MatchorwarpParam;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_integer_parse_int;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:";
/// Java `COMMAND`.
pub const COMMAND: &str = "patchcrawl3d";

/// Java `Patchcrawl3DPrePIPParam`.
#[derive(Clone, Debug)]
pub struct Patchcrawl3DPrePIPParam {
    /// Java superclass `ConstPatchcrawl3DPrePIPParam` state.
    pub base: ConstPatchcrawl3DPrePIPParam,
}

impl std::ops::Deref for Patchcrawl3DPrePIPParam {
    type Target = ConstPatchcrawl3DPrePIPParam;

    fn deref(&self) -> &ConstPatchcrawl3DPrePIPParam {
        &self.base
    }
}

impl std::ops::DerefMut for Patchcrawl3DPrePIPParam {
    fn deref_mut(&mut self) -> &mut ConstPatchcrawl3DPrePIPParam {
        &mut self.base
    }
}

impl Default for Patchcrawl3DPrePIPParam {
    fn default() -> Patchcrawl3DPrePIPParam {
        Patchcrawl3DPrePIPParam::new()
    }
}

impl Patchcrawl3DPrePIPParam {
    /// Java's implicit `Patchcrawl3DPrePIPParam()`.
    pub fn new() -> Patchcrawl3DPrePIPParam {
        Patchcrawl3DPrePIPParam {
            base: ConstPatchcrawl3DPrePIPParam::new(),
        }
    }

    /// Java `setFileA`.
    pub fn set_file_a(&mut self, file_a: Option<&str>) {
        self.base.file_a = file_a.map(str::to_owned);
    }

    /// Java `setFileB`.
    pub fn set_file_b(&mut self, file_b: Option<&str>) {
        self.base.file_b = file_b.map(str::to_owned);
    }

    /// Java `setNX`.
    pub fn set_n_x(&mut self, n_x: i32) {
        self.base.n_x = n_x;
    }

    /// Java `setNY`.
    pub fn set_n_y(&mut self, n_y: i32) {
        self.base.n_y = n_y;
    }

    /// Java `setNZ`.
    pub fn set_n_z(&mut self, n_z: i32) {
        self.base.n_z = n_z;
    }

    /// Java `setOriginalFileB`.
    pub fn set_original_file_b(&mut self, original_file_b: Option<&str>) {
        self.base.original_file_b = original_file_b.map(str::to_owned);
    }

    /// Java `setTransformFile`.
    pub fn set_transform_file(&mut self, transform_file: Option<&str>) {
        self.base.transform_file = transform_file.map(str::to_owned);
    }

    /// Java `setBorders`.
    pub fn set_borders(
        &mut self,
        borders: Option<&str>,
    ) -> Result<(), FortranInputSyntaxException> {
        self.base.borders.validate_and_set(borders)
    }

    /// Java `setXHigh`.
    pub fn set_x_high(&mut self, x_high: i32) {
        self.base.x_high = x_high;
    }

    /// Java `setXLow`.
    pub fn set_x_low(&mut self, x_low: i32) {
        self.base.x_low = x_low;
    }

    /// Java `setXPatchSize`.
    pub fn set_x_patch_size(&mut self, x_patch_size: i32) {
        self.base.x_patch_size = x_patch_size;
    }

    /// Java `setYHigh`.
    pub fn set_y_high(&mut self, y_high: i32) {
        self.base.y_high = y_high;
    }

    /// Java `setYLow`.
    pub fn set_y_low(&mut self, y_low: i32) {
        self.base.y_low = y_low;
    }

    /// Java `setYPatchSize`.
    pub fn set_y_patch_size(&mut self, y_patch_size: i32) {
        self.base.y_patch_size = y_patch_size;
    }

    /// Java `setZHigh`.
    pub fn set_z_high(&mut self, z_high: i32) {
        self.base.z_high = z_high;
    }

    /// Java `setZLow`.
    pub fn set_z_low(&mut self, z_low: i32) {
        self.base.z_low = z_low;
    }

    /// Java `setZPatchSize`.
    pub fn set_z_patch_size(&mut self, z_patch_size: i32) {
        self.base.z_patch_size = z_patch_size;
    }

    /// Java `setMaxShift`.
    pub fn set_max_shift(&mut self, max_shift: i32) {
        self.base.max_shift = max_shift;
    }

    /// Java `setBoundaryModel`.
    pub fn set_boundary_model(&mut self, boundary_model: Option<&str>) {
        self.base.boundary_model = boundary_model.map(str::to_owned);
    }

    /// Java `setUseBoundaryModel`.
    pub fn set_use_boundary_model(&mut self, use_boundary_model: bool) {
        if use_boundary_model {
            self.base.boundary_model = Some(MatchorwarpParam::get_default_patch_region_model());
        } else {
            self.base.boundary_model = Some(String::new());
        }
    }

    /// Java `setOutputFile`.
    pub fn set_output_file(&mut self, output_file: Option<&str>) {
        self.base.output_file = output_file.map(str::to_owned);
    }
}

impl CommandParam for Patchcrawl3DPrePIPParam {
    /// Java `parseComScriptCommand`.
    ///
    /// A null argument array (Java NullPointerException) reads as empty here, so it
    /// takes the wrong-argument-count path.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        let cmd_line_args = script_command.get_command_line_args().unwrap_or_default();
        self.base.reset();
        if cmd_line_args.len() < 16 || cmd_line_args.len() > 20 {
            let message = format!(
                "Incorrect number of arguments, expected 16 - 20 found: {}",
                cmd_line_args.len()
            );
            return Err(BadComScriptException::new(&message).into());
        }
        let mut i: usize = 0;
        let mut parameter_id;
        // Java `Integer.parseInt(cmdLineArgs[i++])`; a NumberFormatException is caught
        // below the try block and reported with the already-incremented index.
        let parse_int = |index: usize| -> Result<i32, ()> {
            match cmd_line_args[index].as_deref() {
                None => Err(()),
                Some(arg) => java_lang_integer_parse_int(arg).map_err(|_| ()),
            }
        };
        let number_format = |i: usize, parameter_id: &str| -> ParseComScriptError {
            let message = format!(
                "NumberFormatException Argument #: {} value :{} for parameter: {}",
                i,
                cmd_line_args[i].as_deref().unwrap_or("null"),
                parameter_id
            );
            BadComScriptException::new(&message).into()
        };
        parameter_id = "xsize";
        i += 1;
        self.base.x_patch_size = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "ysize";
        i += 1;
        self.base.y_patch_size = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "zsize";
        i += 1;
        self.base.z_patch_size = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "nx";
        i += 1;
        self.base.n_x = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "ny";
        i += 1;
        self.base.n_y = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "nz";
        i += 1;
        self.base.n_z = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "xlo";
        i += 1;
        self.base.x_low = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "xhi";
        i += 1;
        self.base.x_high = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "ylo";
        i += 1;
        self.base.y_low = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "yhi";
        i += 1;
        self.base.y_high = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "zlo";
        i += 1;
        self.base.z_low = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "zhi";
        i += 1;
        self.base.z_high = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "matchshift";
        i += 1;
        self.base.max_shift = match parse_int(i - 1) {
            Ok(value) => value,
            Err(()) => return Err(number_format(i, parameter_id)),
        };
        parameter_id = "filea";
        self.base.file_a = cmd_line_args[i].clone();
        i += 1;
        parameter_id = "fileb";
        self.base.file_b = cmd_line_args[i].clone();
        i += 1;
        parameter_id = "output_file";
        self.base.output_file = cmd_line_args[i].clone();
        i += 1;
        let required_length = i;
        if cmd_line_args.len() == required_length + 1 {
            parameter_id = "boundary_model";
            self.base.boundary_model = cmd_line_args[i].clone();
        } else if cmd_line_args.len() == required_length + 3 {
            parameter_id = "transform_file";
            self.base.transform_file = cmd_line_args[i].clone();
            i += 1;
            parameter_id = "original_fileb";
            self.base.original_file_b = cmd_line_args[i].clone();
            i += 1;
            parameter_id = "borders";
            self.base
                .borders
                .validate_and_set(cmd_line_args[i].as_deref())?;
        } else if cmd_line_args.len() == required_length + 4 {
            parameter_id = "transform_file";
            self.base.transform_file = cmd_line_args[i].clone();
            i += 1;
            parameter_id = "original_fileb";
            self.base.original_file_b = cmd_line_args[i].clone();
            i += 1;
            parameter_id = "borders";
            self.base
                .borders
                .validate_and_set(cmd_line_args[i].as_deref())?;
            i += 1;
            parameter_id = "boundary_model";
            self.base.boundary_model = cmd_line_args[i].clone();
        }
        let _ = parameter_id;
        Ok(())
    }

    /// Java `updateComScriptCommand`.  A null `transformFile` or `boundaryModel` (a
    /// NullPointerException in Java's `equals`) counts as empty here.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let mut cmd_line_args: Vec<Option<String>> =
            Vec::with_capacity(script_command.get_command_line_length().max(0) as usize);
        let mut _bad_parameter;
        _bad_parameter = "xsize";
        cmd_line_args.push(Some(self.base.x_patch_size.to_string()));
        _bad_parameter = "ysize";
        cmd_line_args.push(Some(self.base.y_patch_size.to_string()));
        _bad_parameter = "zsize";
        cmd_line_args.push(Some(self.base.z_patch_size.to_string()));
        _bad_parameter = "nx";
        cmd_line_args.push(Some(self.base.n_x.to_string()));
        _bad_parameter = "ny";
        cmd_line_args.push(Some(self.base.n_y.to_string()));
        _bad_parameter = "nz";
        cmd_line_args.push(Some(self.base.n_z.to_string()));
        _bad_parameter = "xlo";
        cmd_line_args.push(Some(self.base.x_low.to_string()));
        _bad_parameter = "xhi";
        cmd_line_args.push(Some(self.base.x_high.to_string()));
        _bad_parameter = "ylo";
        cmd_line_args.push(Some(self.base.y_low.to_string()));
        _bad_parameter = "yhi";
        cmd_line_args.push(Some(self.base.y_high.to_string()));
        _bad_parameter = "Zlo";
        cmd_line_args.push(Some(self.base.z_low.to_string()));
        _bad_parameter = "Zhi";
        cmd_line_args.push(Some(self.base.z_high.to_string()));
        _bad_parameter = "max_shift";
        cmd_line_args.push(Some(self.base.max_shift.to_string()));
        _bad_parameter = "filea";
        cmd_line_args.push(self.base.file_a.clone());
        _bad_parameter = "fileb";
        cmd_line_args.push(self.base.file_b.clone());
        _bad_parameter = "output_file";
        cmd_line_args.push(self.base.output_file.clone());
        if self.base.transform_file.as_deref().unwrap_or("") != "" {
            _bad_parameter = "transform_file";
            cmd_line_args.push(self.base.transform_file.clone());
            _bad_parameter = "original_fileb";
            cmd_line_args.push(self.base.original_file_b.clone());
            _bad_parameter = "borders";
            cmd_line_args.push(Some(self.base.borders.to_string()));
        }
        if self.base.boundary_model.as_deref().unwrap_or("") != "" {
            _bad_parameter = "boundary_model";
            cmd_line_args.push(self.base.boundary_model.clone());
        }
        script_command.set_command_line_args(&cmd_line_args);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
