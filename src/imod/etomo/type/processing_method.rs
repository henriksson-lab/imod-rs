//! `IMOD/Etomo/src/etomo/type/ProcessingMethod.java`.
#![allow(dead_code)]

use std::collections::HashMap;
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

    /// Java `remove(Properties, String)`.
    pub fn remove(properties: Option<&mut HashMap<String, String>>, prepend: Option<&str>) {
        if let Some(properties) = properties {
            properties.remove(&Self::create_key(prepend));
        }
    }

    /// Java `store(Properties, String)`.
    pub fn store(self, properties: Option<&mut HashMap<String, String>>, prepend: Option<&str>) {
        if let Some(properties) = properties {
            properties.insert(Self::create_key(prepend), self.to_string());
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load(
        properties: Option<&HashMap<String, String>>,
        prepend: Option<&str>,
    ) -> Option<Self> {
        properties.and_then(|properties| {
            Self::get_instance(
                properties
                    .get(&Self::create_key(prepend))
                    .map(String::as_str),
            )
        })
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

#[cfg(test)]
mod tests {
    use super::ProcessingMethod;
    use std::collections::HashMap;

    #[test]
    fn properties_round_trip_with_java_key_and_null_boundary() {
        let mut properties = HashMap::new();
        ProcessingMethod::PpGpu.store(Some(&mut properties), Some("axisA"));
        assert_eq!(
            ProcessingMethod::load(Some(&properties), Some("axisA")),
            Some(ProcessingMethod::PpGpu)
        );
        ProcessingMethod::remove(Some(&mut properties), Some("axisA"));
        assert_eq!(
            ProcessingMethod::load(Some(&properties), Some("axisA")),
            None
        );
        ProcessingMethod::LocalCpu.store(None, None);
        ProcessingMethod::remove(None, None);
        assert_eq!(ProcessingMethod::load(None, None), None);
    }
}
