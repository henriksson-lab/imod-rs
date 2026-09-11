//! `IMOD/Etomo/src/etomo/ui/StandardBarString.java`.
//!
//! Builds progress bar labels.  Java's `enum` is mirrored as a Rust enum with one
//! variant per constant; the five constructor arguments are returned by per-variant
//! field accessors because a Rust enum variant carries no per-instance storage.
#![allow(dead_code)]

use crate::imod::etomo::util::utilities;

/// Java `StandardBarString`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StandardBarString {
    /// Java `BACKING_UP("backup", "Backing up", null, 1, 3)`.
    BackingUp,
    /// Java `COPYING_FROM("copy from", "Copying from", null, 1, 2)`.
    CopyingFrom,
    /// Java `COPYING_TO("copy to", "Copying to", null, 1, 2)`.
    CopyingTo,
    /// Java `CREATING("create", "Creating", null, 1, 2)`.
    Creating,
    /// Java `DELETING("delete", "Deleting", null, 1, 1)`.
    Deleting,
    /// Java `ENDING("end", "Ending", null, 0, 2)`.
    Ending,
    /// Java `MOVING("move", "Moving", "to", 2, 2)`.
    Moving,
    /// Java `OVERWRITING("overwrite", "Overwriting", null, 1, 2)`.
    Overwriting,
    /// Java `READING("read", "Reading", null, 1, 2)`.
    Reading,
    /// Java `RECONNECTING("reconnect", "Reconnecting", null, 0, 2)`.
    Reconnecting,
    /// Java `RENAMING("rename", "Renaming", "to", 2, 2)`.
    Renaming,
    /// Java `RENAMING_TO("rename to", "Renaming to", null, 1, 2)`.
    RenamingTo,
    /// Java `STARTING("start", "Starting", null, 0, 2)`.
    Starting,
    /// Java `WRITING("write", "Writing", null, 1, 1)`.
    Writing,
}

impl StandardBarString {
    /// Java field `actionDescr`.
    fn action_descr_field(self) -> &'static str {
        match self {
            Self::BackingUp => "backup",
            Self::CopyingFrom => "copy from",
            Self::CopyingTo => "copy to",
            Self::Creating => "create",
            Self::Deleting => "delete",
            Self::Ending => "end",
            Self::Moving => "move",
            Self::Overwriting => "overwrite",
            Self::Reading => "read",
            Self::Reconnecting => "reconnect",
            Self::Renaming => "rename",
            Self::RenamingTo => "rename to",
            Self::Starting => "start",
            Self::Writing => "write",
        }
    }

    /// Java field `leftBarString`.
    fn left_bar_string_field(self) -> &'static str {
        match self {
            Self::BackingUp => "Backing up",
            Self::CopyingFrom => "Copying from",
            Self::CopyingTo => "Copying to",
            Self::Creating => "Creating",
            Self::Deleting => "Deleting",
            Self::Ending => "Ending",
            Self::Moving => "Moving",
            Self::Overwriting => "Overwriting",
            Self::Reading => "Reading",
            Self::Reconnecting => "Reconnecting",
            Self::Renaming => "Renaming",
            Self::RenamingTo => "Renaming to",
            Self::Starting => "Starting",
            Self::Writing => "Writing",
        }
    }

    /// Java field `joinBarString`, which is optional.
    fn join_bar_string_field(self) -> Option<&'static str> {
        match self {
            Self::Moving | Self::Renaming => Some("to"),
            _ => None,
        }
    }

    /// Java field `maxFiles`: maximum files that this instance refers to.
    fn max_files_field(self) -> i32 {
        match self {
            Self::Moving | Self::Renaming => 2,
            Self::Ending | Self::Reconnecting | Self::Starting => 0,
            _ => 1,
        }
    }

    /// Java field `precedence`.
    fn precedence_field(self) -> i32 {
        match self {
            Self::BackingUp => 3,
            Self::Deleting | Self::Writing => 1,
            _ => 2,
        }
    }

    /// Java `replaceWithPrecedence`.
    pub fn replace_with_precedence(self, new_bar_string: StandardBarString) -> StandardBarString {
        if self.precedence_field() > new_bar_string.precedence_field() {
            return self;
        }
        new_bar_string
    }

    /// Java `getActionDescr`.
    pub fn get_action_descr(self) -> &'static str {
        self.action_descr_field()
    }

    /// Java `buildBarString(StandardBarString, String, String, boolean, boolean)`.
    pub fn build_bar_string_static(
        standard_bar_string: Option<StandardBarString>,
        from_file_name: Option<&str>,
        to_file_name: Option<&str>,
        renamed: bool,
        failed: bool,
    ) -> String {
        // Make bar string.
        let bar_string: String;
        if let Some(standard_bar_string) = standard_bar_string {
            bar_string =
                standard_bar_string.build_bar_string(from_file_name, to_file_name, renamed, failed);
        } else {
            bar_string = StandardBarString::build_unknown_bar_string(
                0,
                from_file_name,
                to_file_name,
                renamed,
                failed,
            );
        }
        bar_string
    }

    /// Java `buildBarString(String, String, boolean, boolean)`.
    pub fn build_bar_string(
        self,
        from_file_name: Option<&str>,
        to_file_name: Option<&str>,
        renamed: bool,
        failed: bool,
    ) -> String {
        StandardBarString::build_bar_string_private(
            Some(self.left_bar_string_field()),
            self.join_bar_string_field(),
            self.max_files_field(),
            from_file_name,
            to_file_name,
            renamed,
            failed,
        )
    }

    /// Java `buildUnknownBarString`.  `max_files` must be greater than 0 for it to print
    /// anything.
    pub fn build_unknown_bar_string(
        max_files: i32,
        from_file_name: Option<&str>,
        to_file_name: Option<&str>,
        renamed: bool,
        failed: bool,
    ) -> String {
        StandardBarString::build_bar_string_private(
            None,
            None,
            max_files,
            from_file_name,
            to_file_name,
            renamed,
            failed,
        )
    }

    /// Java `buildBarString(String, String, int, String, String, boolean, boolean)`.
    /// All purpose bar string builder.  When maxFiles is 0, the file names may be set,
    /// but they are just leftovers from previous file manipulations, and they and the
    /// booleans should be ignored.
    fn build_bar_string_private(
        left_bar_string: Option<&str>,
        join_bar_string: Option<&str>,
        max_files: i32,
        from_file_name: Option<&str>,
        mut to_file_name: Option<&str>,
        renamed: bool,
        failed: bool,
    ) -> String {
        // Ignore parameters that are not related to this type of bar string.
        if max_files < 2 {
            to_file_name = None;
        }
        let mut empty = true;
        let mut bar_string = String::new();
        if let Some(left_bar_string) = left_bar_string {
            // example: Renaming
            empty = false;
            bar_string.push_str(left_bar_string);
        }
        if max_files > 0 {
            if let Some(from_file_name) = from_file_name {
                if !empty {
                    bar_string.push(' ');
                }
                // example: Renaming file.mrc
                empty = false;
                bar_string.push_str(from_file_name);
            } else {
                if !empty {
                    bar_string.push(' ');
                }
                // example: Renaming file
                empty = false;
                bar_string.push_str("file");
            }
        }
        if let Some(to_file_name) = to_file_name {
            if let Some(join_bar_string) = join_bar_string {
                if !empty {
                    bar_string.push(' ');
                }
                // example: Renaming file.mrc to
                empty = false;
                bar_string.push_str(join_bar_string);
            } else if left_bar_string.is_none() {
                // example: file.mrc,
                bar_string.push(',');
            }
            if !empty {
                bar_string.push(' ');
            }
            // example: Renaming file.mrc to file.mrc.orig
            // example: file.mrc, file.mrc.orig
            empty = false;
            bar_string.push_str(to_file_name);
        }
        // Don't include state if the instance isn't about file(s).
        if empty {
            if max_files == 0 {
                return " ".to_string();
            }
            if renamed && !failed {
                return "Done".to_string();
            } else if !renamed && failed {
                return "Failed".to_string();
            }
            return " ".to_string();
        }
        if max_files > 0 {
            if renamed && !failed {
                // example: Renaming file.mrc to file.mrc.orig done
                bar_string.push_str(" done");
            } else if !renamed && failed {
                // example: Renaming file.mrc to file.mrc.orig failed
                bar_string.push_str(" failed");
            }
        }
        bar_string
    }

    /// Java `matches`.
    pub fn matches(self, bar_string: Option<&str>) -> bool {
        !utilities::is_empty(bar_string)
            && bar_string
                .unwrap()
                .starts_with(self.left_bar_string_field())
    }
}
