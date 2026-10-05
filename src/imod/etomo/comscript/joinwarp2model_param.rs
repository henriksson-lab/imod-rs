//! `IMOD/Etomo/src/etomo/comscript/Joinwarp2modelParam.java`.
//!
//! The `joinwarp2model` command of `joinwarp2model.com` (Refine Join), a keyword-value
//! command read from and written back to the com file by `JoinComscriptManager`.

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::r#type::string_parameter::StringParameter;

/// Java `public static final String COMMAND_NAME`.
pub const COMMAND_NAME: &str = "joinwarp2model";

/// Java `public final class Joinwarp2modelParam implements CommandParam`.
pub struct Joinwarp2modelParam {
    /// Java private final `joinedFile`.
    joined_file: StringParameter,
    /// Java private final `appliedTransformFile`.
    applied_transform_file: StringParameter,
    /// Java private final `offsetInXandY`.
    offset_in_xand_y: StringParameter,
    /// Java private final `chunkSizes`.
    chunk_sizes: StringParameter,
    /// Java private final `inputWarpFile`.
    input_warp_file: StringParameter,
    /// Java private final `ouputModelFile` (the source's spelling).
    ouput_model_file: StringParameter,
    /// Java private final `binningOfJoin`.
    binning_of_join: ScriptParameter,
    /// Java private final `manager` (stored, not read again).
    #[allow(dead_code)]
    manager: &'static JoinManager,
}

impl Joinwarp2modelParam {
    /// Java `Joinwarp2modelParam(JoinManager)`.
    pub fn new(manager: &'static JoinManager) -> Joinwarp2modelParam {
        Joinwarp2modelParam {
            joined_file: StringParameter::new("JoinedFile"),
            applied_transform_file: StringParameter::new("AppliedTransformFile"),
            offset_in_xand_y: StringParameter::new("OffsetInXandY"),
            chunk_sizes: StringParameter::new("ChunkSizes"),
            input_warp_file: StringParameter::new("InputWarpFile"),
            ouput_model_file: StringParameter::new("OutputModelFile"),
            binning_of_join: ScriptParameter::new_with_type_and_name(
                Type::Integer,
                "BinningOfJoin",
            ),
            manager,
        }
    }

    /// Java `setRefineModelFilename(String)`.
    pub fn set_refine_model_filename(&mut self, input: Option<&str>) {
        self.ouput_model_file.set(input);
    }

    /// Java `setModeledJoinFilename(String)`.
    pub fn set_modeled_join_filename(&mut self, input: Option<&str>) {
        self.joined_file.set(input);
    }

    /// Java `setInputWarpFilename(String)`.
    pub fn set_input_warp_filename(&mut self, input: Option<&str>) {
        self.input_warp_file.set(input);
    }

    /// Java `setAppliedTransformFilename(String)`.
    pub fn set_applied_transform_filename(&mut self, input: Option<&str>) {
        self.applied_transform_file.set(input);
    }

    /// Java `setBinningOfJoin(String)`.
    pub fn set_binning_of_join(&mut self, input: Option<&str>) {
        self.binning_of_join.set_string(input);
    }

    /// Java `setOffsetInXandY(String)`.
    pub fn set_offset_in_xand_y(&mut self, input: Option<&str>) {
        self.offset_in_xand_y.set(input);
    }

    /// Java `setChunkSizes(String)`.
    pub fn set_chunk_sizes(&mut self, input: Option<&str>) {
        self.chunk_sizes.set(input);
    }

    /// Java `getCommandName()`.
    pub fn get_command_name(&self) -> String {
        COMMAND_NAME.to_string()
    }
}

impl CommandParam for Joinwarp2modelParam {
    /// Java `parseComScriptCommand(ComScriptCommand)`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Java calls `scriptCommand.useKeywordValue()` on the command it parses,
        // converting an old-style command in place.  The trait lends the command
        // immutably, so the conversion is made on a copy, which is what is parsed.
        let mut script_command = ComScriptCommand::new_from(script_command);
        script_command.use_keyword_value();
        let script_command = &script_command;
        self.initialize_defaults();
        self.joined_file.parse(script_command)?;
        self.applied_transform_file.parse(script_command)?;
        self.offset_in_xand_y.parse(script_command)?;
        self.binning_of_join.parse(script_command)?;
        self.chunk_sizes.parse(script_command)?;
        self.input_warp_file.parse(script_command)?;
        self.ouput_model_file.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand(ComScriptCommand)`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        script_command.use_keyword_value();
        self.ouput_model_file.update_com_script(script_command);
        self.joined_file.update_com_script(script_command);
        self.applied_transform_file.update_com_script(script_command);
        self.offset_in_xand_y.update_com_script(script_command);
        self.binning_of_join.update_com_script(script_command);
        self.chunk_sizes.update_com_script(script_command);
        self.input_warp_file.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults()`.
    fn initialize_defaults(&mut self) {
        self.joined_file.reset();
        self.applied_transform_file.reset();
        self.offset_in_xand_y.reset();
        self.binning_of_join.reset();
        self.chunk_sizes.reset();
        self.input_warp_file.reset();
        self.ouput_model_file.reset();
    }
}
