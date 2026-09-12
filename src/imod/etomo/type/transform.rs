//! `IMOD/Etomo/src/etomo/type/Transform.java`.
#![allow(dead_code)]

use std::collections::HashMap;

/// Java `Transform`: its five canonical singleton values are represented by
/// this closed Rust enum.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Transform {
    #[default]
    FullLinearTransformation,
    RotationTranslationMagnification,
    RotationTranslation,
    Translation,
    SkipSearch,
}

impl Transform {
    pub const DEFAULT: Self = Self::FullLinearTransformation;

    pub fn store(
        transform: Option<Self>,
        props: &mut HashMap<String, String>,
        prepend: &str,
        key: &str,
    ) {
        let prepend = Self::create_prepend(prepend);
        let property_key = format!("{prepend}.{key}");
        if transform.is_none() {
            props.remove(&property_key);
        }
        if let Some(transform) = transform {
            props.insert(property_key, transform.to_string());
        }
    }

    pub fn remove(props: &mut HashMap<String, String>, prepend: &str, key: &str) {
        let prepend = Self::create_prepend(prepend);
        props.remove(&format!("{prepend}.{key}"));
    }

    pub fn load(
        props: &HashMap<String, String>,
        prepend: &str,
        key: &str,
        default_transform: Self,
    ) -> Self {
        let prepend = Self::create_prepend(prepend);
        let Some(value) = props.get(&format!("{prepend}.{key}")) else {
            return default_transform;
        };
        Self::get_instance(value).unwrap_or(default_transform)
    }

    pub fn get_instance(name: &str) -> Option<Self> {
        [
            Self::FullLinearTransformation,
            Self::RotationTranslationMagnification,
            Self::RotationTranslation,
            Self::Translation,
            Self::SkipSearch,
        ]
        .into_iter()
        .find(|transform| transform.name() == name)
    }

    pub fn to_string(self) -> String {
        self.name().into()
    }

    pub fn get_value(self) -> &'static str {
        match self {
            Self::FullLinearTransformation => "0",
            Self::RotationTranslationMagnification => "4",
            Self::RotationTranslation => "3",
            Self::Translation => "2",
            Self::SkipSearch => "-1",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FullLinearTransformation => "FullLinearTransformation",
            Self::RotationTranslationMagnification => "RotationTranslationMagnification",
            Self::RotationTranslation => "RotationTranslation",
            Self::Translation => "Translation",
            Self::SkipSearch => "SkipSearch",
        }
    }

    fn create_prepend(prepend: &str) -> String {
        if prepend.is_empty() {
            "Transform".into()
        } else {
            format!("{prepend}.Transform")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Transform;
    use std::collections::HashMap;

    #[test]
    fn properties_follow_java_transform_protocol() {
        let mut props = HashMap::new();
        Transform::store(Some(Transform::Translation), &mut props, "join", "choice");
        assert_eq!(props["join.Transform.choice"], "Translation");
        assert_eq!(
            Transform::load(&props, "join", "choice", Transform::RotationTranslation),
            Transform::Translation
        );
        Transform::remove(&mut props, "join", "choice");
        assert_eq!(
            Transform::load(&props, "join", "choice", Transform::RotationTranslation),
            Transform::RotationTranslation
        );
    }
}
