//! Owned translation of `svlConfigManager.{h,cpp}`.
//!
//! SVL used a process-global registry of raw module pointers.  The Rust
//! registry owns its modules, which makes registration and unregistration
//! explicit and avoids dangling configuration callbacks.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use super::svl_logger::{SvlLogLevel, log_message};

/// XML element needed by the SVL configuration grammar.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SvlConfigNode {
    pub name: String,
    pub attributes: BTreeMap<String, String>,
    pub children: Vec<Self>,
}

impl SvlConfigNode {
    /// Parses the element/attribute subset consumed by `svlConfigManager`.
    pub fn parse(xml: &str) -> Result<Self, String> {
        let mut roots = Vec::<Self>::new();
        let mut stack = Vec::<Self>::new();
        let mut cursor = 0;
        while let Some(relative_start) = xml[cursor..].find('<') {
            let start = cursor + relative_start;
            let end = xml[start..]
                .find('>')
                .map(|offset| start + offset)
                .ok_or_else(|| "unterminated XML tag".to_string())?;
            let tag = xml[start + 1..end].trim();
            cursor = end + 1;
            if tag.starts_with('?') || tag.starts_with('!') {
                continue;
            }
            if let Some(name) = tag.strip_prefix('/') {
                let node = stack
                    .pop()
                    .ok_or_else(|| "unmatched XML close tag".to_string())?;
                if node.name != name.trim() {
                    return Err(format!("XML close tag {name} does not match {}", node.name));
                }
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
                continue;
            }
            let self_closing = tag.ends_with('/');
            let tag = tag.trim_end_matches('/').trim();
            let mut words = tag.split_whitespace();
            let name = words
                .next()
                .ok_or_else(|| "empty XML tag".to_string())?
                .to_string();
            let mut node = Self {
                name,
                ..Self::default()
            };
            let mut rest = tag[node.name.len()..].trim();
            while !rest.is_empty() {
                let eq = rest
                    .find('=')
                    .ok_or_else(|| format!("malformed XML attribute in {tag}"))?;
                let key = rest[..eq].trim();
                let after_eq = rest[eq + 1..].trim_start();
                let quote = after_eq
                    .chars()
                    .next()
                    .ok_or_else(|| "missing XML attribute value".to_string())?;
                if quote != '\"' && quote != '\'' {
                    return Err("XML attribute values must be quoted".to_string());
                }
                let value_end = after_eq[1..]
                    .find(quote)
                    .map(|offset| offset + 1)
                    .ok_or_else(|| "unterminated XML attribute".to_string())?;
                node.attributes
                    .insert(key.to_string(), after_eq[1..value_end].to_string());
                rest = after_eq[value_end + 1..].trim_start();
            }
            if self_closing {
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            } else {
                stack.push(node);
            }
        }
        if !stack.is_empty() {
            return Err("unclosed XML tag".to_string());
        }
        if roots.len() != 1 {
            return Err("expected exactly one XML root element".to_string());
        }
        Ok(roots.pop().expect("checked XML root count"))
    }
}

/// Rust equivalent of the abstract `svlConfigurableModule` interface.
pub trait SvlConfigurableModule: Send {
    fn name(&self) -> &str;

    fn usage(&self) -> String {
        String::new()
    }

    fn set_configuration(&mut self, name: &str, value: &str) -> Result<(), String>;

    /// `svlConfigurableModule::readConfiguration(XMLNode&)`.
    fn read_configuration_node(&mut self, node: &SvlConfigNode) -> Result<(), String> {
        for (name, value) in &node.attributes {
            self.set_configuration(name, value)?;
        }
        for child in node.children.iter().filter(|child| child.name == "option") {
            let name = child
                .attributes
                .get("name")
                .ok_or_else(|| "option has no name".to_string())?;
            let value = child
                .attributes
                .get("value")
                .ok_or_else(|| "option has no value".to_string())?;
            self.set_configuration(name, value)?;
        }
        Ok(())
    }

    /// `svlConfigurableModule::readConfiguration(const char *)`.
    fn read_configuration_file(&mut self, filename: &Path) -> Result<(), String> {
        let root = SvlConfigNode::parse(
            &fs::read_to_string(filename).map_err(|error| error.to_string())?,
        )?;
        match root.children.iter().find(|node| node.name == self.name()) {
            Some(node) => self.read_configuration_node(node),
            None => {
                let message = format!(
                    "couldn't find configuration for {} in {}",
                    self.name(),
                    filename.display()
                );
                let _ = log_message(SvlLogLevel::Error, &message);
                Ok(())
            }
        }
    }
}

/// Owned configuration registry, replacing SVL's singleton of raw pointers.
#[derive(Default)]
pub struct SvlConfigurationManager {
    registry: BTreeMap<String, Box<dyn SvlConfigurableModule>>,
}

impl SvlConfigurationManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_module(&mut self, module: Box<dyn SvlConfigurableModule>) {
        let name = module.name().to_owned();
        let _ = log_message(SvlLogLevel::Debug, &format!("registering module {name}"));
        self.registry.insert(name, module);
    }

    pub fn unregister_module(&mut self, name: &str) -> Option<Box<dyn SvlConfigurableModule>> {
        let _ = log_message(SvlLogLevel::Debug, &format!("unregistering module {name}"));
        let module = self.registry.remove(name);
        if module.is_none() {
            let _ = log_message(
                SvlLogLevel::Warning,
                &format!("module with name \"{name}\" has already been unregistered"),
            );
        }
        module
    }

    /// `svlConfigurationManager::configure(const char *)`.
    pub fn configure_file(&mut self, filename: impl AsRef<Path>) -> Result<(), String> {
        let filename = filename.as_ref();
        let contents = match fs::read_to_string(filename) {
            Ok(contents) => contents,
            Err(_) => {
                let _ = log_message(
                    SvlLogLevel::Error,
                    &format!("couldn't find configuration in {}", filename.display()),
                );
                return Ok(());
            }
        };
        self.configure_node(&SvlConfigNode::parse(&contents)?)
    }

    /// `svlConfigurationManager::configure(XMLNode&)`.
    pub fn configure_node(&mut self, root: &SvlConfigNode) -> Result<(), String> {
        for node in &root.children {
            match self.registry.get_mut(&node.name) {
                None => {
                    let _ = log_message(
                        SvlLogLevel::Debug,
                        &format!("no module with name {} has been registered", node.name),
                    );
                }
                Some(module) => {
                    let _ = log_message(
                        SvlLogLevel::Debug,
                        &format!("configuring module {}", node.name),
                    );
                    module.read_configuration_node(node)?;
                }
            }
        }
        Ok(())
    }

    /// `svlConfigurationManager::configure(module, name, value)`.
    pub fn configure(&mut self, module: &str, name: &str, value: &str) -> Result<(), String> {
        let configurable = self
            .registry
            .get_mut(module)
            .ok_or_else(|| format!("no module with name \"{module}\" has been registered"))?;
        let _ = log_message(
            SvlLogLevel::Debug,
            &format!("setting {module}::{name} to {value}"),
        );
        configurable.set_configuration(name, value)
    }

    /// `svlConfigurationManager::showRegistry()` rendered as owned text.
    pub fn show_registry(&self) -> String {
        let mut result = String::from("--- svlConfigurationManager registry ---\n");
        for (name, module) in &self.registry {
            result.push_str(&format!("  * {name}\n{}", module.usage()));
        }
        result.push_str("--- -------------------------------- ---\n");
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Module {
        answer: String,
    }
    impl SvlConfigurableModule for Module {
        fn name(&self) -> &str {
            "unit"
        }
        fn usage(&self) -> String {
            "    answer\n".into()
        }
        fn set_configuration(&mut self, name: &str, value: &str) -> Result<(), String> {
            if name != "answer" {
                return Err(format!("unknown {name}"));
            }
            self.answer = value.into();
            Ok(())
        }
    }

    #[test]
    fn xml_attributes_and_option_children_configure_registered_modules() {
        let root = SvlConfigNode::parse(
            "<config><unit answer=\"one\"><option name=\"answer\" value=\"two\"/></unit></config>",
        )
        .unwrap();
        let mut manager = SvlConfigurationManager::new();
        manager.register_module(Box::<Module>::default());
        manager.configure_node(&root).unwrap();
        assert!(manager.show_registry().contains("  * unit\n    answer"));
        let module = manager.unregister_module("unit").unwrap();
        assert_eq!(module.name(), "unit");
    }
}
