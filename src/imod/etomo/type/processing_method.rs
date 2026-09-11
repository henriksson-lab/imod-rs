//! `IMOD/Etomo/src/etomo/type/ProcessingMethod.java`.
#![allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessingMethod {
    LocalCpu,
    LocalGpu,
    PpCpu,
    PpGpu,
    Queue,
}
impl ProcessingMethod {
    pub const DEFAULT: Self = Self::LocalCpu;
    pub fn is_local(self) -> bool {
        matches!(self, Self::LocalCpu | Self::LocalGpu)
    }
    pub fn get_instance(value: Option<&str>) -> Option<Self> {
        match value? {
            "LOCAL_CPU" => Some(Self::LocalCpu),
            "LOCAL_GPU" => Some(Self::LocalGpu),
            "PP_CPU" => Some(Self::PpCpu),
            "PP_GPU" => Some(Self::PpGpu),
            "QUEUE" => Some(Self::Queue),
            _ => None,
        }
    }
    pub fn create_key(prepend: Option<&str>) -> String {
        if prepend.is_none_or(|v| v.trim().is_empty()) {
            "ProcessingMethod".into()
        } else {
            format!("{}.ProcessingMethod", prepend.unwrap())
        }
    }
}
impl std::fmt::Display for ProcessingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::LocalCpu => "LOCAL_CPU",
            Self::LocalGpu => "LOCAL_GPU",
            Self::PpCpu => "PP_CPU",
            Self::PpGpu => "PP_GPU",
            Self::Queue => "QUEUE",
        })
    }
}
