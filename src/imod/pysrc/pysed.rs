//! Translation of `IMOD/pysrc/pysed.py`.
use regex::{Regex, RegexBuilder};

const ESC_SLASH: &str = "#escSlash^";

/// Matches `psReportErr` (`IMOD/pysrc/pysed.py:15`).
pub fn ps_report_err(error: String, return_on_error: bool) -> Result<(), String> {
    if return_on_error {
        Err(error)
    } else {
        panic!("{error}")
    }
}

/// Matches `swapParenCompile` (`IMOD/pysrc/pysed.py:23`).
pub fn swap_paren_compile(pattern: &str, nocase: bool) -> Result<Regex, String> {
    let mut output = String::new();
    let chars = pattern.chars().collect::<Vec<_>>();
    for (index, character) in chars.iter().enumerate() {
        let previous = index.checked_sub(1).and_then(|i| chars.get(i));
        let next = chars.get(index + 1);
        match character {
            '(' | ')' if previous != Some(&'\\') => {
                output.push('\\');
                output.push(*character);
            }
            '^' if index != 0 && previous != Some(&'\\') && previous != Some(&'[') => {
                output.push_str("\\^")
            }
            '$' if next.is_some() && previous != Some(&'\\') => output.push_str("\\$"),
            '+' if previous != Some(&'\\') => output.push_str("\\+"),
            _ => output.push(*character),
        }
    }
    // Python sed groups use \\( and \\), while Rust regex uses parentheses.
    output = output.replace("\\\\(", "(").replace("\\\\)", ")");
    RegexBuilder::new(&output)
        .case_insensitive(nocase)
        .build()
        .map_err(|e| e.to_string())
}

/// Matches `pysed` (`IMOD/pysrc/pysed.py:49`).
pub fn pysed(
    sedregs_in: &[String],
    src: &[String],
    nocase: bool,
    delim: char,
) -> Result<Vec<String>, String> {
    let mut output = src.to_vec();
    let mut print_mode = false;
    for sedreg in sedregs_in {
        let fields = if delim == '/' {
            sedreg
                .replace("\\/", ESC_SLASH)
                .split('/')
                .map(|s| s.replace(ESC_SLASH, "/"))
                .collect::<Vec<_>>()
        } else {
            sedreg.split(delim).map(str::to_owned).collect::<Vec<_>>()
        };
        if fields.len() < 3 || fields.len() > 6 {
            return Err(format!("Expression too short or too long: {sedreg}"));
        }
        let (address, action, replacement, modifiers) = if fields[0] == "s" {
            if fields.len() != 4 {
                return Err(format!("Incorrect s/// entry: {sedreg}"));
            }
            (None, "s", fields[2].clone(), fields[3].clone())
        } else {
            if fields[0].is_empty() {
                (
                    Some(fields[1].clone()),
                    fields[2].as_str(),
                    fields.get(4).cloned().unwrap_or_default(),
                    fields.get(5).cloned().unwrap_or_default(),
                )
            } else {
                return Err(format!("Only s can precede patterns: {sedreg}"));
            }
        };
        if modifiers.contains('p') {
            print_mode = true;
        }
        let address_re = address
            .as_deref()
            .map(|p| swap_paren_compile(p, nocase))
            .transpose()?;
        let search_pattern = if fields[0] == "s" {
            &fields[1]
        } else if fields.get(3).is_some_and(|v| !v.is_empty()) {
            &fields[3]
        } else {
            address.as_deref().unwrap_or("")
        };
        let search_re = swap_paren_compile(search_pattern, nocase)?;
        let mut next = Vec::new();
        for mut line in output {
            let matches = address_re.as_ref().is_none_or(|re| re.is_match(&line));
            let mut retain = !print_mode;
            if matches {
                match action {
                    "s" => {
                        line = if modifiers.contains('g') {
                            search_re
                                .replace_all(&line, replacement.as_str())
                                .into_owned()
                        } else {
                            search_re.replace(&line, replacement.as_str()).into_owned()
                        };
                        if modifiers.contains('p') {
                            retain = true;
                        }
                    }
                    "d" => retain = false,
                    "p" => retain = true,
                    "a" => {
                        if retain {
                            next.push(line.clone());
                        }
                        next.push(replacement.clone());
                        continue;
                    }
                    _ => return Err(format!("Only s, d, a, and p are allowed actions: {sedreg}")),
                }
            }
            if retain {
                next.push(line);
            }
        }
        output = next;
    }
    Ok(output)
}

/// Matches `sedDelAndAdd` (`IMOD/pysrc/pysed.py:233`).
pub fn sed_del_and_add(option: &str, value: &str, after_line: &str, delim: char) -> Vec<String> {
    vec![
        format!("{delim}^{option}{delim}d"),
        format!("{delim}^{after_line}{delim}a{delim}{option}\t{value}{delim}"),
    ]
}

/// Matches `sedModify` (`IMOD/pysrc/pysed.py:238`).
pub fn sed_modify(option: &str, value: &str, delim: char) -> String {
    format!("{delim}^{option}{delim}s{delim}[ \t].*{delim}\t{value}{delim}")
}
