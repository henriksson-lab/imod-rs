//! Translation of `IMOD/pysrc/pysed.py`.
use super::imodpy::{fmtstr, prnstr};
use regex::{Regex, RegexBuilder};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

/// Matches `escslash` (`IMOD/pysrc/pysed.py:11`).
const ESC_SLASH: &str = "#escSlash^";

/// Matches the module global `pysedReturnOnErr` (`IMOD/pysrc/pysed.py:12`).
///
/// Nothing in the source ever assigns it: `pysed` assigns `pysedReturnError`
/// (`pysed.py:75-76`), a different name, so `psReportErr` always reaches its
/// `sys.exit(1)` and the `retErr` argument has no effect.  Verified by running
/// the Python: `pysed(["s/a/"], ["x"], retErr=True)` prints
/// `ERROR: pysed - Incorrect s/// entry: s/a/` and exits 1.
static PYSED_RETURN_ON_ERR: AtomicBool = AtomicBool::new(false);
/// Matches the module global `pysedReturnError` that `pysed` assigns (`pysed.py:76`).
static PYSED_RETURN_ERROR: AtomicBool = AtomicBool::new(false);

/// Matches `psReportErr` (`IMOD/pysrc/pysed.py:14`).
pub fn ps_report_err(error: String) -> String {
    prnstr(&error, "\n", false);
    if PYSED_RETURN_ON_ERR.load(Ordering::SeqCst) {
        return error;
    }
    std::process::exit(1)
}

/// Matches `swapParenCompile` (`IMOD/pysrc/pysed.py:22`).
///
/// `caretMatch` and `dollarMatch` are module globals that `pysed` compiles
/// (`pysed.py:99-100`) before its first call here; they are the same two
/// constant patterns on every call, so they are compiled where they are used.
/// A pattern `re.compile` rejects raises `re.error` in the source, which is an
/// uncaught traceback on stderr and exit status 1.
pub fn swap_paren_compile(pattern: &str, flags: bool) -> Regex {
    let caret_match = Regex::new(r"([^\\\[])\^").unwrap();
    let dollar_match = Regex::new(r"\$(.)").unwrap();

    // Escape any ^ except one at the start or after [, and any $ except one at the end
    // Also escape any + because that is not supported by sed
    let mut pat = caret_match.replace_all(pattern, "${1}\\^").into_owned();
    // `\\$\1` in the source: a backslash, a `$`, then the escaped character.
    pat = dollar_match.replace_all(&pat, "\\$$${1}").into_owned();
    pat = pat.replace('+', "\\+");
    if pat.contains('(') {
        pat = pat.replace("\\(", ESC_SLASH);
        pat = pat.replace('(', "\\(");
        pat = pat.replace(ESC_SLASH, "(");
        // print 'replacement 1:', pattern, pat
    }

    if pat.contains(')') {
        pat = pat.replace("\\)", ESC_SLASH);
        pat = pat.replace(')', "\\)");
        pat = pat.replace(ESC_SLASH, ")");
        // print 'replacement 2:', pattern, pat
    }

    match RegexBuilder::new(&pat).case_insensitive(flags).build() {
        Ok(regex) => regex,
        Err(error) => {
            eprintln!("re.error: {error}");
            std::process::exit(1)
        }
    }
}

/// Reproduces the template parsing of `re.sub` (CPython
/// `Lib/re/_parser.py:parse_template`) for the replacement string the source
/// hands to `substr[i].sub(replace[i], line, numsub[i])` (`pysed.py:198,204`),
/// as the `regex` crate's own template syntax: `\1`, `\12` and `\g<n>` become
/// `${n}`, `\g<name>` becomes `${name}`, the C escapes and octal escapes become
/// their characters, and a literal `$` becomes `$$`.  A bad escape or a
/// reference to a group the pattern lacks is `re.error` in the source, an
/// uncaught traceback on stderr with exit status 1.
fn re_sub_template(template: &str, pattern: &Regex) -> String {
    let groups = pattern.captures_len() - 1;
    let chars = template.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut index = 0;
    fn invalid(message: String) -> ! {
        eprintln!("re.error: {message}");
        std::process::exit(1)
    }
    while index < chars.len() {
        let c = chars[index];
        index += 1;
        if c == '$' {
            out.push_str("$$");
            continue;
        }
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(&c) = chars.get(index) else {
            invalid("bad escape (end of pattern)".to_owned());
        };
        index += 1;
        match c {
            'g' => {
                if chars.get(index) != Some(&'<') {
                    invalid("missing <".to_owned());
                }
                index += 1;
                let start = index;
                while index < chars.len() && chars[index] != '>' {
                    index += 1;
                }
                if index >= chars.len() {
                    invalid("missing >, unterminated name".to_owned());
                }
                let name = chars[start..index].iter().collect::<String>();
                index += 1;
                if name.is_empty() {
                    invalid("missing group name".to_owned());
                }
                if name.chars().all(|d| d.is_ascii_digit()) {
                    let number: usize = name.parse().unwrap_or(usize::MAX);
                    if number > groups {
                        invalid(format!("invalid group reference {number}"));
                    }
                    out.push_str(&format!("${{{number}}}"));
                } else {
                    if pattern.capture_names().flatten().all(|n| n != name) {
                        invalid(format!("unknown group name '{name}'"));
                    }
                    out.push_str(&format!("${{{name}}}"));
                }
            }
            '0' => {
                let mut value = 0u32;
                let mut taken = 0;
                while taken < 2 && chars.get(index).is_some_and(|d| ('0'..='7').contains(d)) {
                    value = value * 8 + chars[index].to_digit(8).unwrap();
                    index += 1;
                    taken += 1;
                }
                match char::from_u32(value) {
                    Some(ch) if ch == '$' => out.push_str("$$"),
                    Some(ch) => out.push(ch),
                    None => invalid("bad escape".to_owned()),
                }
            }
            '1'..='9' => {
                let mut this = String::from(c);
                let mut isoctal = false;
                if chars.get(index).is_some_and(|d| d.is_ascii_digit()) {
                    this.push(chars[index]);
                    index += 1;
                    if ('0'..='7').contains(&c)
                        && ('0'..='7').contains(&this.chars().nth(1).unwrap())
                        && chars.get(index).is_some_and(|d| ('0'..='7').contains(d))
                    {
                        this.push(chars[index]);
                        index += 1;
                        isoctal = true;
                        let value = u32::from_str_radix(&this, 8).unwrap();
                        if value > 0o377 {
                            invalid(format!(
                                "octal escape value \\{this} outside of range 0-0o377"
                            ));
                        }
                        match char::from_u32(value) {
                            Some(ch) if ch == '$' => out.push_str("$$"),
                            Some(ch) => out.push(ch),
                            None => invalid("bad escape".to_owned()),
                        }
                    }
                }
                if !isoctal {
                    let number: usize = this.parse().unwrap();
                    if number > groups {
                        invalid(format!("invalid group reference {number}"));
                    }
                    out.push_str(&format!("${{{number}}}"));
                }
            }
            'a' => out.push('\x07'),
            'b' => out.push('\x08'),
            'f' => out.push('\x0c'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'v' => out.push('\x0b'),
            '\\' => out.push('\\'),
            other if other.is_ascii_alphabetic() => {
                invalid(format!("bad escape \\{other}"));
            }
            other => {
                out.push('\\');
                if other == '$' {
                    out.push_str("$$");
                } else {
                    out.push(other);
                }
            }
        }
    }
    out
}

/// The `src` argument of `pysed`: "the list of strings in src, or, if src is a
/// single string, ... the file whose name is given in src" (`pysed.py:47-49`).
pub enum PysedSrc<'a> {
    File(&'a str),
    Lines(&'a [String]),
}

/// Matches `pysed` (`IMOD/pysrc/pysed.py:46`).
///
/// Returns `Ok(Some(lines))` for the source's list return, `Ok(None)` for its
/// `return None` after writing `dstfile`, and `Err(message)` for the string
/// `psReportErr` returns — which, per [`PYSED_RETURN_ON_ERR`], it never does.
pub fn pysed(
    sedregs_in: &[String],
    src: PysedSrc,
    dstfile: Option<&str>,
    nocase: bool,
    delim: char,
    ret_err: bool,
) -> Result<Option<Vec<String>>, String> {
    PYSED_RETURN_ERROR.store(ret_err, Ordering::SeqCst);

    // Set error prefix and try to open input file
    // `progname` is not a name in this module (it is a global of the calling
    // script, which `import` does not share), so the `try` always takes its
    // `except` branch.
    let prefix = "ERROR: pysed -".to_owned();
    let srclines: Vec<String> = match src {
        PysedSrc::File(name) => match std::fs::read_to_string(name) {
            // Python opens the file with universal newlines, so `\r\n` and a
            // lone `\r` both end a line, and `readlines` keeps the `\n`.
            Ok(text) => text
                .replace("\r\n", "\n")
                .replace('\r', "\n")
                .split_inclusive('\n')
                .map(str::to_owned)
                .collect(),
            Err(error) => {
                let text = error.to_string();
                let exc_info = match error.raw_os_error() {
                    Some(errno) => format!(
                        "[Errno {errno}] {}: '{name}'",
                        text.strip_suffix(&format!(" (os error {errno})"))
                            .unwrap_or(&text)
                    ),
                    None => text.clone(),
                };
                return Err(ps_report_err(fmtstr(
                    "{} Opening or reading from {}: {}",
                    &[prefix, name.to_owned(), exc_info],
                )));
            }
        },
        PysedSrc::Lines(lines) => lines.to_vec(),
    };

    let sedregs = sedregs_in;

    // Initialize lists to be built up
    let mut action: Vec<String> = Vec::new();
    let mut numsub: Vec<usize> = Vec::new();
    let mut pattern: Vec<Option<Regex>> = Vec::new();
    let mut substr: Vec<Option<Regex>> = Vec::new();
    let mut replace: Vec<Option<String>> = Vec::new();
    let mut printind: Vec<usize> = Vec::new();
    let flags = nocase;

    // Loop on expressions, parsing them for action, pattern, etc
    for i in 0..sedregs.len() {
        let splt: Vec<String> = if delim == '/' {
            let line = sedregs[i].replace("\\/", ESC_SLASH);
            line.split(delim)
                .map(|field| field.replace(ESC_SLASH, "/"))
                .collect()
        } else {
            sedregs[i].split(delim).map(str::to_owned).collect()
        };
        if splt.len() < 3 || splt.len() > 6 {
            return Err(ps_report_err(fmtstr(
                "{} Expression too short or too long: {}",
                &[prefix, sedregs[i].clone()],
            )));
        }

        // Initialize entries for this expression
        let mut sea: Option<Regex> = None;
        let mut rep: Option<String> = None;
        let mut pat: Option<Regex> = None;
        let mut nsub = 1;
        let act: String;
        if splt[0] == "s" {
            if splt.len() != 4 || (!splt[3].is_empty() && splt[3] != "g") {
                return Err(ps_report_err(fmtstr(
                    "{} Incorrect s/// entry: {}",
                    &[prefix, sedregs[i].clone()],
                )));
            }

            sea = Some(swap_paren_compile(&splt[1], flags));
            rep = Some(splt[2].clone());
            act = "s".to_owned();
            if splt[3] == "g" {
                nsub = 0;
            }
        } else if !splt[0].is_empty() {
            return Err(ps_report_err(fmtstr(
                "{} Only s can preceed patterns: {}",
                &[prefix, sedregs[i].clone()],
            )));
        } else {
            act = splt[2].clone();
            if act.chars().count() > 1 {
                return Err(ps_report_err(fmtstr(
                    "{} Action must be a single letter: {}",
                    &[prefix, sedregs[i].clone()],
                )));
            }
            if act == "d" || act == "p" {
                if splt.len() > 3 {
                    return Err(ps_report_err(fmtstr(
                        "{} d or p must not be followed by pattern: {}",
                        &[prefix, sedregs[i].clone()],
                    )));
                }
                pat = Some(swap_paren_compile(&splt[1], flags));
                if act == "p" {
                    printind.push(i);
                }
            } else if act == "a" {
                if splt.len() != 5 || !splt[4].is_empty() {
                    return Err(ps_report_err(fmtstr(
                        "{} Incorrect 'a' entry: {}",
                        &[prefix, sedregs[i].clone()],
                    )));
                }
                pat = Some(swap_paren_compile(&splt[1], flags));
                rep = Some(splt[3].clone());
            } else if act == "s" {
                if splt.len() < 6 {
                    return Err(ps_report_err(fmtstr(
                        "{} Too few elements for s command: {}",
                        &[prefix, sedregs[i].clone()],
                    )));
                }
                rep = Some(splt[4].clone());
                for modifier in splt[5].chars() {
                    if modifier == 'g' {
                        nsub = 0;
                    } else if modifier == 'p' {
                        printind.push(i);
                    } else {
                        return Err(ps_report_err(fmtstr(
                            "{} Only p and g are allowed modifiers: {}",
                            &[prefix, sedregs[i].clone()],
                        )));
                    }
                }

                rep = Some(splt[4].clone());
                if splt[3].is_empty() {
                    sea = Some(swap_paren_compile(&splt[1], flags));
                    if printind.contains(&i) {
                        pat = sea.clone();
                    }
                } else {
                    sea = Some(swap_paren_compile(&splt[3], flags));
                    pat = Some(swap_paren_compile(&splt[1], flags));
                }
            } else {
                return Err(ps_report_err(fmtstr(
                    "{} Only s, d, a, and p are allowed actions: {}",
                    &[prefix, sedregs[i].clone()],
                )));
            }
        }

        pattern.push(pat);
        substr.push(sea);
        replace.push(rep);
        action.push(act);
        numsub.push(nsub);
    }

    let mut outlines: Vec<String> = Vec::new();
    for line in &srclines {
        let mut line = line.trim_end_matches(['\r', '\n']).to_owned();
        let mut delete = !printind.is_empty();
        let mut addlines: Vec<String> = Vec::new();
        for i in 0..sedregs.len() {
            if let Some(pat) = &pattern[i] {
                if pat.is_match(&line) {
                    if action[i] == "s" {
                        let sea = substr[i].as_ref().unwrap();
                        line = sea
                            .replacen(
                                &line,
                                numsub[i],
                                re_sub_template(replace[i].as_deref().unwrap(), sea).as_str(),
                            )
                            .into_owned();
                    } else if action[i] == "d" {
                        delete = true;
                    } else if action[i] == "a" {
                        addlines.push(replace[i].clone().unwrap());
                    }
                    if printind.contains(&i) {
                        delete = false;
                    }
                }
            } else {
                let sea = substr[i].as_ref().unwrap();
                line = sea
                    .replacen(
                        &line,
                        numsub[i],
                        re_sub_template(replace[i].as_deref().unwrap(), sea).as_str(),
                    )
                    .into_owned();
            }
        }

        if !delete {
            outlines.push(line);
        }
        for l in addlines {
            outlines.push(l);
        }
    }

    // `if dstfile:` — an empty string is false in Python.
    if let Some(dstfile) = dstfile.filter(|name| !name.is_empty()) {
        let mut sedout = match std::fs::File::create(dstfile) {
            Ok(file) => file,
            Err(error) => {
                let text = error.to_string();
                let exc_info = match error.raw_os_error() {
                    Some(errno) => format!(
                        "[Errno {errno}] {}: '{dstfile}'",
                        text.strip_suffix(&format!(" (os error {errno})"))
                            .unwrap_or(&text)
                    ),
                    None => text.clone(),
                };
                return Err(ps_report_err(fmtstr(
                    "{} Opening {}: {}",
                    &[prefix, dstfile.to_owned(), exc_info],
                )));
            }
        };
        for line in &outlines {
            if writeln!(sedout, "{line}").is_err() {
                return Err(ps_report_err(fmtstr(
                    "{} Writing to output file",
                    &[prefix],
                )));
            }
        }
        drop(sedout);
        return Ok(None);
    }

    Ok(Some(outlines))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Differential driver against the Python `pysed.pysed` run by
    /// `python3` over the same cases file: `IMOD_PYSED_CASES` names a file
    /// whose first line is the input lines joined by `\x1f` and whose later
    /// lines are `nocase<TAB>delim<TAB>sedreg\x1fsedreg...`; the report goes to
    /// `IMOD_PYSED_OUT` in the Python driver's format.
    #[test]
    fn pysed_differential() {
        let Ok(cases) = std::env::var("IMOD_PYSED_CASES") else {
            return;
        };
        let outpath = std::env::var("IMOD_PYSED_OUT").unwrap();
        let text = std::fs::read_to_string(&cases).unwrap();
        let mut it = text.lines();
        let lines = it
            .next()
            .unwrap()
            .split('\x1f')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut out = String::new();
        for (i, case) in it.enumerate() {
            let mut fields = case.splitn(3, '\t');
            let nocase = fields.next().unwrap() == "1";
            let delim = fields.next().unwrap().chars().next().unwrap();
            let sedregs = fields
                .next()
                .unwrap()
                .split('\x1f')
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let result = pysed(
                &sedregs,
                PysedSrc::Lines(&lines),
                None,
                nocase,
                delim,
                false,
            )
            .unwrap()
            .unwrap();
            let json = sedregs
                .iter()
                .map(|s| {
                    format!(
                        "\"{}\"",
                        s.replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\t', "\\t")
                            .replace('\n', "\\n")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("=== {i} [{json}]\n"));
            for l in result {
                out.push_str(&format!("[{l}]\n"));
            }
        }
        std::fs::write(outpath, out).unwrap();
    }
}
