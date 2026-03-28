use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// A parsed IMOD autodoc (.adoc) file.
///
/// Autodoc files define command-line parameter specifications using a simple
/// INI-like format with `[SectionHeader = Name]` sections and `key = value` fields.
#[derive(Debug, Clone)]
pub struct Autodoc {
    /// Global key-value pairs (before any section)
    pub globals: HashMap<String, String>,
    /// Sections in order of appearance
    pub sections: Vec<Section>,
}

/// A section in an autodoc file (e.g., `[Field = InputFile]`)
#[derive(Debug, Clone)]
pub struct Section {
    pub kind: String,
    pub name: String,
    pub values: HashMap<String, String>,
}

/// A parsed field definition (convenience view of a Field section)
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub short: String,
    pub field_type: String,
    pub usage: String,
    pub tooltip: String,
    pub manpage: String,
}

impl Autodoc {
    /// Parse an autodoc file from a path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let content = fs::read_to_string(path.as_ref())?;
        Ok(Self::parse(&content))
    }

    /// Parse autodoc content from a string.
    pub fn parse(input: &str) -> Self {
        let mut globals = HashMap::new();
        let mut sections = Vec::new();
        let mut current_section: Option<Section> = None;
        let mut current_key: Option<String> = None;

        for line in input.lines() {
            // Continuation line (starts with ^)
            if let Some(rest) = line.strip_prefix('^') {
                if let Some(ref key) = current_key {
                    let target = if let Some(ref mut sec) = current_section {
                        &mut sec.values
                    } else {
                        &mut globals
                    };
                    if let Some(val) = target.get_mut(key) {
                        val.push('\n');
                        val.push_str(rest);
                    }
                }
                continue;
            }

            // Section header: [SectionType = Name]
            if line.starts_with('[') {
                if let Some(sec) = current_section.take() {
                    sections.push(sec);
                }
                current_key = None;

                if let Some(content) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                    if let Some((kind, name)) = content.split_once('=') {
                        current_section = Some(Section {
                            kind: kind.trim().to_string(),
                            name: name.trim().to_string(),
                            values: HashMap::new(),
                        });
                    }
                }
                continue;
            }

            // Key = value
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim().to_string();
                let value = value.trim().to_string();
                current_key = Some(key.clone());

                if let Some(ref mut sec) = current_section {
                    sec.values.insert(key, value);
                } else {
                    globals.insert(key, value);
                }
                continue;
            }

            // Blank or comment lines — ignore
        }

        // Push final section
        if let Some(sec) = current_section {
            sections.push(sec);
        }

        Self { globals, sections }
    }

    /// Get all Field sections as structured Field objects.
    pub fn fields(&self) -> Vec<Field> {
        self.sections
            .iter()
            .filter(|s| s.kind == "Field")
            .map(|s| Field {
                name: s.name.clone(),
                short: s.values.get("short").cloned().unwrap_or_default(),
                field_type: s.values.get("type").cloned().unwrap_or_default(),
                usage: s.values.get("usage").cloned().unwrap_or_default(),
                tooltip: s.values.get("tooltip").cloned().unwrap_or_default(),
                manpage: s.values.get("manpage").cloned().unwrap_or_default(),
            })
            .collect()
    }

    /// Get the Pip version if set.
    pub fn pip_version(&self) -> Option<&str> {
        self.globals.get("Pip").map(|s| s.as_str())
    }

    /// Get the autodoc version.
    pub fn version(&self) -> Option<&str> {
        self.globals.get("Version").map(|s| s.as_str())
    }
}
