//! Parser for IMOD autodoc (.adoc) PIP parameter specification files.
//!
//! Autodoc files use an INI-like syntax with `[SectionType = Name]` headers
//! and `key = value` pairs to describe command-line parameters, their types,
//! usage strings, tooltips, and man-page entries. This crate parses that
//! format into an [`Autodoc`] struct containing global key-value pairs and
//! ordered [`Section`]s. A convenience [`Field`] type provides a structured
//! view of individual parameter definitions.

mod parser;

pub use parser::{Autodoc, Field, Section};

#[cfg(test)]
mod tests;
