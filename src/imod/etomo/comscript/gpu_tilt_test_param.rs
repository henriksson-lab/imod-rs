//! `IMOD/Etomo/src/etomo/comscript/GpuTiltTestParam.java`.
//!
//! The parameters of the `gputilttest` Python script, run by the Tools
//! interface's "Test GPU" panel.

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::const_etomo_number::{Number, Type};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `public static final String OUTPUT_KEYWORD`.
pub const OUTPUT_KEYWORD: &str = "differed";

/// Java `public final class GpuTiltTestParam`.
pub struct GpuTiltTestParam {
    /// Java private final `nMinutes = new EtomoNumber(EtomoNumber.Type.DOUBLE)`.
    n_minutes: EtomoNumber,
    /// Java private final `gpuNumber = new EtomoNumber()`.
    gpu_number: EtomoNumber,
}

impl Default for GpuTiltTestParam {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuTiltTestParam {
    /// Java's implicit default constructor, with the field initialisers.
    pub fn new() -> GpuTiltTestParam {
        GpuTiltTestParam {
            n_minutes: EtomoNumber::new_with_type(Some(Type::Double)),
            gpu_number: EtomoNumber::new(),
        }
    }

    /// Java `setNMinutes(String)`.
    pub fn set_n_minutes(&mut self, input: Option<&str>) {
        self.n_minutes.set_string(input);
    }

    /// Java `setGpuNumber(Number)`.
    pub fn set_gpu_number(&mut self, input: Option<Number>) {
        self.gpu_number.set_number(input);
    }

    /// Java `getCommand()`.
    pub fn get_command(&self) -> Vec<String> {
        let mut command: Vec<String> = Vec::new();
        command.push("python".to_owned());
        command.push("-u".to_owned());
        command.push(format!(
            "{}{}",
            etomo_director::INSTANCE
                .get_python_script_path()
                .unwrap_or_else(|| "null".to_owned()),
            ProcessName::GPU_TILT_TEST
        ));
        command.push("-PID".to_owned());
        command.push(self.n_minutes.to_string());
        command.push(self.gpu_number.to_string());
        command
    }
}
