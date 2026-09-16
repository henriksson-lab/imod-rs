//! Owned translation of `svlOptions.{h,cpp}`.
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SvlOptions {
    options: BTreeMap<String, String>,
}
impl SvlOptions {
    pub fn declare_option(
        &mut self,
        name: impl Into<String>,
        default: impl Into<String>,
    ) -> Result<(), String> {
        let name = name.into();
        if self.options.contains_key(&name) {
            return Err(format!("option {name} already declared"));
        }
        self.options.insert(name, default.into());
        Ok(())
    }
    pub fn undeclare_option(&mut self, name: &str) -> Result<(), String> {
        self.options
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| format!("option {name} not declared"))
    }
    pub fn set_option(&mut self, name: &str, value: impl ToString) -> Result<(), String> {
        let entry = self
            .options
            .get_mut(name)
            .ok_or_else(|| format!("option {name} not declared"))?;
        *entry = value.to_string();
        Ok(())
    }
    pub fn set_options(&mut self, options: &BTreeMap<String, String>) -> Result<(), String> {
        for (name, value) in options {
            self.set_option(name, value)?;
        }
        Ok(())
    }
    pub fn set_options_from_string(&mut self, options: &str) -> Result<(), String> {
        for item in options.split(',').filter(|s| !s.is_empty()) {
            let (name, value) = item.split_once('=').ok_or("bad option string")?;
            self.set_option(name, value)?;
        }
        Ok(())
    }
    pub fn get_option(&self, name: &str) -> Result<&str, String> {
        self.options
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("option {name} not declared"))
    }
    pub fn get_option_as_bool(&self, name: &str) -> Result<bool, String> {
        Ok(matches!(
            self.get_option(name)?.to_ascii_lowercase().as_str(),
            "true" | "yes" | "1"
        ))
    }
    pub fn get_option_as_int(&self, name: &str) -> Result<i32, String> {
        Ok(self.get_option(name)?.parse().unwrap_or(0))
    }
    pub fn get_option_as_double(&self, name: &str) -> Result<f64, String> {
        Ok(self.get_option(name)?.parse().unwrap_or(0.))
    }
    pub fn get_option_names(&self) -> Vec<String> {
        self.options.keys().cloned().collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_options_follow_source_coercions() {
        let mut o = SvlOptions::default();
        o.declare_option("a", "false").unwrap();
        o.set_option("a", "YES").unwrap();
        assert!(o.get_option_as_bool("a").unwrap());
        o.declare_option("n", "0").unwrap();
        o.set_option("n", 12).unwrap();
        assert_eq!(o.get_option_as_int("n").unwrap(), 12);
    }
}
