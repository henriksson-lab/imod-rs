//! `IMOD/Etomo/src/etomo/comscript/ImodchopcontsParam.java`.

use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command_param::{CommandParam, ParseComScriptError};
use super::tiltxcorr_param::TiltxcorrParam;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `LENGTH_OF_PIECES_KEY`.
pub const LENGTH_OF_PIECES_KEY: &str = "LengthOfPieces";
/// Java `MINIMUM_OVERLAP_KEY`.
pub const MINIMUM_OVERLAP_KEY: &str = "MinimumOverlap";
/// Java private `LENGTH_OF_PIECES_DEFAULT`.
const LENGTH_OF_PIECES_DEFAULT: i32 = -1;
/// Java `GOTO_LABEL`.
pub const GOTO_LABEL: &str = "dochop";

/// Java final `ImodchopcontsParam implements CommandParam`.
#[derive(Clone, Debug)]
pub struct ImodchopcontsParam {
    length_of_pieces: ScriptParameter,
    minimum_overlap: ScriptParameter,
}

impl ImodchopcontsParam {
    /// Java `ImodchopcontsParam()`.
    pub fn new() -> ImodchopcontsParam {
        let mut minimum_overlap = ScriptParameter::new_with_name(MINIMUM_OVERLAP_KEY);
        minimum_overlap.set_display_value_int(4);
        ImodchopcontsParam {
            length_of_pieces: ScriptParameter::new_with_name(LENGTH_OF_PIECES_KEY),
            minimum_overlap,
        }
    }

    /// Java static `getBackwardCompatableInstance(TiltxcorrParam)` (deprecated).
    pub fn get_backward_compatable_instance(
        tiltxcorr_param: &TiltxcorrParam,
    ) -> ImodchopcontsParam {
        let mut imodchopconts_param = ImodchopcontsParam::new();
        imodchopconts_param
            .length_of_pieces
            .set_string(Some(&tiltxcorr_param.get_length_from_length_and_overlap()));
        imodchopconts_param
            .minimum_overlap
            .set_string(Some(&tiltxcorr_param.get_overlap_from_length_and_overlap()));
        imodchopconts_param
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.length_of_pieces.reset();
        self.minimum_overlap.reset();
    }

    /// Java `getLengthOfPieces`.
    pub fn get_length_of_pieces(&self) -> Option<String> {
        if !self.length_of_pieces.equals_int(LENGTH_OF_PIECES_DEFAULT) {
            return Some(self.length_of_pieces.to_string());
        }
        None
    }

    /// Java `isLengthOfPiecesDefault`.
    pub fn is_length_of_pieces_default(&self) -> bool {
        self.length_of_pieces.equals_int(LENGTH_OF_PIECES_DEFAULT)
    }

    /// Java `isLengthOfPiecesNull`.
    pub fn is_length_of_pieces_null(&self) -> bool {
        self.length_of_pieces.is_null()
    }

    /// Java `resetLengthOfPieces`.
    pub fn reset_length_of_pieces(&mut self) {
        self.length_of_pieces.reset();
    }

    /// Java `setLengthOfPiecesDefault`.
    pub fn set_length_of_pieces_default(&mut self) {
        self.length_of_pieces.set_int(LENGTH_OF_PIECES_DEFAULT);
    }

    /// Java `setLengthOfPieces(String)`.
    pub fn set_length_of_pieces(&mut self, input: Option<&str>) {
        self.length_of_pieces.set_string(input);
    }

    /// Java `getMinimumOverlap`.
    pub fn get_minimum_overlap(&self) -> String {
        self.minimum_overlap.to_string()
    }

    /// Java `setMinimumOverlap(String)`.
    pub fn set_minimum_overlap(&mut self, input: Option<&str>) {
        self.minimum_overlap.set_string(input);
    }

    /// Java static `getLengthOfPiecesDefault(BaseManager, AxisID, FileType)`.  For patch
    /// tracking of the prealigned stack.  Returns the default values of
    /// LengthAndOverlap.
    pub fn get_length_of_pieces_default(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        file_type: &Arc<FileType>,
    ) -> String {
        let mut length = EtomoNumber::new();
        let floor = 16;
        length.set_floor(floor);
        length.set_display_value_int(floor);
        let header = MRCHeader::get_instance_in_dir(
            manager.get_property_user_dir().as_deref(),
            file_type
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
            Some(axis_id),
        );
        if let Some(header) = header {
            // `catch (IOException | InvalidParameterException e) {e.printStackTrace();}`
            let read = header.borrow_mut().read_with_manager(manager);
            match read {
                Ok(_) => {
                    let z = header.borrow().get_n_sections();
                    if z != -1 {
                        // `Math.round((z / 5))`: an int quotient, rounded as a float.
                        length.set_int(z / 5);
                    }
                }
                Err(e) => eprintln!("{}", e),
            }
        }
        length.to_string() // + ",4";
    }
}

impl Default for ImodchopcontsParam {
    fn default() -> ImodchopcontsParam {
        ImodchopcontsParam::new()
    }
}

impl CommandParam for ImodchopcontsParam {
    /// Java `parseComScriptCommand`.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        self.reset();
        self.length_of_pieces.parse(script_command)?;
        self.minimum_overlap.parse(script_command)?;
        Ok(())
    }

    /// Java `updateComScriptCommand`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        self.length_of_pieces.update_com_script(script_command);
        self.minimum_overlap.update_com_script(script_command);
        Ok(())
    }

    /// Java `initializeDefaults`.
    fn initialize_defaults(&mut self) {}
}
