//! `IMOD/Etomo/src/etomo/util/Redactor.java`.
//!
//! Removes private values (email addresses, user and host names, ids, addresses)
//! from the environment and property listings etomo writes to its logs.

use regex::Regex;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, Mutex};

/// Java `Utilities.isEmpty(String)`: null or empty.
fn is_empty(string: Option<&str>) -> bool {
    string.is_none_or(str::is_empty)
}

/// Java `private static final Map<String, String> ENV = System.getenv()`.
fn env_get(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Java `public static final Redactor INSTANCE`.
pub static INSTANCE: LazyLock<Redactor> = LazyLock::new(Redactor::new);

/// Java final `Redactor`.
pub struct Redactor {
    email_set: Category,
    user_set: Category,
    uid_set: Category,
    ip_set: Category,
    host_set: Category,
    unknown_set: Category,
}

impl Redactor {
    /// Java private `Redactor()`, with its field initialisers.
    fn new() -> Redactor {
        Redactor {
            email_set: Category::new(
                Some("EMAIL"),
                Some(&[
                    "EMAIL",
                    "EMAILADDR",
                    "DEBEMAIL",
                    "GIT_AUTHOR_EMAIL",
                    "GIT_AUTHOR_EMAIL",
                    "MAILTO",
                    "REPLYTO",
                    "mail.from",
                    "mail.smtp.user",
                    "mail.smtp.from",
                    "CI_COMMIT_AUTHOR_EMAIL",
                    "CHANGE_AUTHOR_EMAIL",
                    "BUILD_REQUESTEDFOREMAIL",
                ]),
                None,
                Some("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"),
                None,
                None,
            ),
            user_set: Category::new(
                Some("USER"),
                Some(&["user.name", "USER", "LOGNAME", "USERNAME", "NWUSERNAME"]),
                Some(&[&["whoami"], &["id", "-un"]]),
                None,
                Some(&[
                    "PATH",
                    "IMOD_DIR",
                    "LD_LIBRARY_PATH",
                    "PWD",
                    "PULSE_CLIENTCONFIG",
                    "XDG_CONFIG_DIRS",
                    "GTK_RC_FILES",
                    "PULSE_SCRIPT",
                    "_",
                    "GTK2_RC_FILES",
                    "XAUTHORITY",
                    "XDG_DATA_DIRS",
                    "NX_ROOT",
                    "TEST_IMOD_DIR",
                    "MAIL",
                    "PULSE_CONFIG",
                    "IMOD_QTLIBDIR",
                    "MANPATH",
                    "IMOD_PLUGIN_DIR",
                    "PULSE_SERVER",
                    "HOME",
                    "user.dir",
                    "java.library.path",
                    "user.home",
                    "java.class.path",
                    "HOMEPATH",
                    "USERPROFILE",
                ]),
                None,
            ),
            uid_set: Category::new(
                Some("UID"),
                Some(&["KDE_SESSION_UID"]),
                None,
                None,
                Some(&["PULSE_RUNTIME_PATH", "SSH_AUTH_SOCK", "XDG_RUNTIME_DIR"]),
                None,
            ),
            ip_set: Category::new(
                None,
                None,
                None,
                None,
                Some(&["SSH_CLIENT", "SSH_CONNECTION"]),
                None,
            ),
            host_set: Category::new(
                Some("HOST"),
                Some(&["HOSTNAME", "COMPUTERNAME", "HOST"]),
                None,
                None,
                None,
                None,
            ),
            unknown_set: Category::new(
                None,
                None,
                None,
                None,
                Some(&[
                    "LOGONSERVER",
                    "MACADDRESS",
                    "USERDNSDOMAIN",
                    "USERDOMAIN",
                    "USERDOMAIN_ROAMINGPROFILE",
                ]),
                Some(&["UNC"]),
            ),
        }
    }

    /// Java `redactName(String)`: `None` when the whole entry is to be removed.
    pub fn redact_name(&self, redaction_name: Option<&str>) -> Option<String> {
        if is_empty(redaction_name) {
            return redaction_name.map(str::to_owned);
        }
        let mut redaction_name = redaction_name.map(str::to_owned);
        redaction_name = self.email_set.redact_name(redaction_name.as_deref());
        redaction_name = self.ip_set.redact_name(redaction_name.as_deref());
        redaction_name = self.host_set.redact_name(redaction_name.as_deref());
        redaction_name = self.user_set.redact_name(redaction_name.as_deref());
        redaction_name = self.uid_set.redact_name(redaction_name.as_deref());
        redaction_name = self.unknown_set.redact_name(redaction_name.as_deref());
        redaction_name
    }

    /// Java `redactValue(String)`.
    pub fn redact_value(&self, redaction_value: Option<&str>) -> Option<String> {
        if is_empty(redaction_value) {
            return redaction_value.map(str::to_owned);
        }
        // When redacting with regular expressions, go from large with few false
        // positives to small.
        let mut redaction_value = redaction_value.map(str::to_owned);
        redaction_value = self.email_set.redact_value(redaction_value.as_deref());
        redaction_value = self.ip_set.redact_value(redaction_value.as_deref());
        redaction_value = self.host_set.redact_value(redaction_value.as_deref());
        redaction_value = self.user_set.redact_value(redaction_value.as_deref());
        redaction_value = self.uid_set.redact_value(redaction_value.as_deref());
        redaction_value = self.unknown_set.redact_value(redaction_value.as_deref());
        redaction_value
    }
}

/// Java `Category.PLACEHOLDER_START`.
const PLACEHOLDER_START: char = '[';
/// Java `Category.PLACEHOLDER_END`.
const PLACEHOLDER_END: char = ']';

/// Java private static class `Category`.
struct Category {
    placeholder_string: Option<String>,
    /// Environment variable names and Java property names.
    redaction_name_list: Option<Vec<&'static str>>,
    redaction_value_process_list: Option<Vec<&'static [&'static str]>>,
    regex: Option<&'static str>,
    fallback_redaction_name_list: Option<Vec<&'static str>>,
    fallback_redaction_prefix_list: Option<Vec<&'static str>>,
    /// Java `redactionValueList` and `redactionValuesLoaded`, guarded together
    /// (Java synchronizes the load on `INSTANCE`).
    loaded: Mutex<(bool, Option<Vec<String>>)>,
}

impl Category {
    fn new(
        placeholder_string: Option<&str>,
        redaction_name_list: Option<&[&'static str]>,
        redaction_value_process_list: Option<&[&'static [&'static str]]>,
        regex: Option<&'static str>,
        fallback_redaction_name_list: Option<&[&'static str]>,
        fallback_redaction_prefix_list: Option<&[&'static str]>,
    ) -> Category {
        Category {
            placeholder_string: placeholder_string.map(str::to_owned),
            redaction_name_list: redaction_name_list.map(<[_]>::to_vec),
            redaction_value_process_list: redaction_value_process_list.map(<[_]>::to_vec),
            regex,
            fallback_redaction_name_list: fallback_redaction_name_list.map(<[_]>::to_vec),
            fallback_redaction_prefix_list: fallback_redaction_prefix_list.map(<[_]>::to_vec),
            loaded: Mutex::new((false, None)),
        }
    }

    /// Java private `redactName(String)`.
    fn redact_name(&self, redaction_name: Option<&str>) -> Option<String> {
        let redaction_name = redaction_name.filter(|name| !name.is_empty())?;
        self.build_redaction_value_list();
        if let Some(redaction_name_list) = &self.redaction_name_list {
            for name in redaction_name_list {
                if redaction_name == *name {
                    // An environment variable or Java property that contains a
                    // redaction value.  Redact the entire entry.
                    return None;
                }
            }
        } else if self.regex.is_none()
            && self
                .loaded
                .lock()
                .unwrap()
                .1
                .as_ref()
                .is_none_or(Vec::is_empty)
        {
            // No other way to redact: remove every entry whose name is in the
            // fallback list of names.
            if let Some(fallback_redaction_name_list) = &self.fallback_redaction_name_list {
                for name in fallback_redaction_name_list {
                    if redaction_name == *name {
                        return None;
                    }
                }
            }
            if let Some(fallback_redaction_prefix_list) = &self.fallback_redaction_prefix_list {
                for prefix in fallback_redaction_prefix_list {
                    // Fixed in translation (BUGS.md): Java tests
                    // `startsWith(fallbackRedactionNameList[i])` here, indexing the
                    // name list with the prefix list's index.
                    if redaction_name.starts_with(prefix) {
                        return None;
                    }
                }
            }
        }
        Some(redaction_name.to_owned())
    }

    /// Java private `redactValue(String)`.
    fn redact_value(&self, redaction_value: Option<&str>) -> Option<String> {
        if is_empty(redaction_value) {
            return redaction_value.map(str::to_owned);
        }
        let mut redaction_value = redaction_value.unwrap().to_owned();
        self.build_redaction_value_list();
        let redaction_value_list = self.loaded.lock().unwrap().1.clone();
        if let Some(redaction_value_list) = redaction_value_list
            && !redaction_value_list.is_empty()
        {
            for (i, pattern) in redaction_value_list.iter().enumerate() {
                let regex = Regex::new(pattern).unwrap();
                let placeholder = self.build_placeholder(i as i32);
                redaction_value = regex
                    .replace_all(&redaction_value, regex::NoExpand(&placeholder))
                    .into_owned();
            }
        }
        if let Some(pattern) = self.regex {
            let regex = Regex::new(pattern).unwrap();
            let placeholder = self.build_placeholder(-1);
            redaction_value = regex
                .replace_all(&redaction_value, regex::NoExpand(&placeholder))
                .into_owned();
        }
        Some(redaction_value)
    }

    /// Java private `buildRedactionValueList()`.
    fn build_redaction_value_list(&self) {
        let mut loaded = self.loaded.lock().unwrap();
        if loaded.0 {
            return;
        }
        loaded.0 = true;
        // Load the values of the environment variables and/or Java properties.  The
        // entire value is a redaction value.
        if let Some(redaction_name_list) = &self.redaction_name_list {
            for name in redaction_name_list {
                let redaction_value = env_get(name);
                let Some(redaction_value) = redaction_value.filter(|value| !value.is_empty())
                else {
                    continue;
                };
                // Pattern.quote(value); Java's `contains` compares the raw value with
                // the quoted entries, so a value is never found and duplicates are
                // added, harmlessly.
                if loaded
                    .1
                    .as_ref()
                    .is_some_and(|list| list.contains(&redaction_value))
                {
                    continue;
                }
                loaded
                    .1
                    .get_or_insert_with(Vec::new)
                    .push(regex::escape(&redaction_value));
            }
        }
        // Load the results of shell commands that find redaction values.
        if let Some(redaction_value_process_list) = &self.redaction_value_process_list {
            for command in redaction_value_process_list {
                let Some((program, args)) = command.split_first() else {
                    continue;
                };
                // Combine the error stream with the output stream.
                let Ok(output) = Command::new(program)
                    .args(args)
                    .stdin(Stdio::null())
                    .stderr(Stdio::piped())
                    .stdout(Stdio::piped())
                    .spawn()
                else {
                    continue;
                };
                let Ok(output) = output.wait_with_output() else {
                    continue;
                };
                let mut combined = output.stdout;
                combined.extend_from_slice(&output.stderr);
                let text = String::from_utf8_lossy(&combined);
                // Java appends a line only while the builder is empty: the first
                // non-empty line.
                let first_line = text
                    .lines()
                    .find(|line| !line.is_empty())
                    .unwrap_or("")
                    .to_owned();
                if output.status.code() == Some(0) && !first_line.is_empty() {
                    loaded
                        .1
                        .get_or_insert_with(Vec::new)
                        .push(regex::escape(&first_line));
                }
            }
        }
    }

    /// Java private `buildPlaceholder(int)`.
    fn build_placeholder(&self, index: i32) -> String {
        let Some(placeholder_string) = &self.placeholder_string else {
            return format!("{PLACEHOLDER_START}REDACTED{PLACEHOLDER_END}");
        };
        format!(
            "{}{}{}{}",
            PLACEHOLDER_START,
            placeholder_string,
            if index > -1 {
                index.to_string()
            } else {
                String::new()
            },
            PLACEHOLDER_END
        )
    }
}
