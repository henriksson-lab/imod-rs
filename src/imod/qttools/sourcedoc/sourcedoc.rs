//! Rust translation of `IMOD/qttools/sourcedoc/sourcedoc.cpp`.
//!
//! `QString`, `QFile`, and `QTextStream` are ordinary Unicode strings and
//! text-file I/O here.  No Qt GUI or Qt-specific runtime facility is involved
//! in this program.

use regex::Regex;
use std::fs::File;
use std::io::{Read, Write};

const CAPTURE_NONBS: &str = "([^\\\\])";

/// Translation of `usage`.
pub fn usage(progname: &str) -> ! {
    println!("Usage: {progname} [options] input_doc output_doc");
    println!("  input_doc is the input html document to be scanned");
    println!("  output_doc is the output document");
    println!(" Options:");
    println!("    -f       Source files are Fortran (default is C/C++)");
    println!("    -d path  Set path to source files");
    println!("    -D       Debug mode");
    crate::imod::libcfshr::b3dutil::exit(1);
}

/// Translation of `main`.
///
/// The error paths `exit(1)` where the source does; the normal end returns
/// the source's `return 0` to the launcher.
pub fn sourcedoc(argv: &[String]) -> i32 {
    const LIST_FUNCTIONS_FROM: &str = "LIST FUNCTIONS FROM ";
    const DESCRIBE_FUNCTIONS_FROM: &str = "DESCRIBE FUNCTIONS FROM ";
    const LIST_CODE_FROM: &str = "LIST CODE FROM ";
    const DESCRIBE_CODE_FROM: &str = "DESCRIBE CODE FROM ";
    const DOC_SECTION: &str = "DOC_SECTION";
    const END_SECTION: &str = "END_SECTION";
    const DOC_CODE: &str = "DOC_CODE";
    const END_CODE: &str = "END_CODE";
    const NO_FUNCTION: &str = "0";
    const PROGNAME: &str = "sourcedoc";

    if argv.len() < 3 {
        usage(PROGNAME);
    }

    let mut fort77 = false;
    let mut doc_start = "/\\*!".to_owned();
    let mut doc_end = "\\*/".to_owned();
    let mut doc_continue = "\\*".to_owned();
    let mut non_doc_comment = "^[ \\t]*/[/\\*]".to_owned();
    let code_comment = "^[ \\t]*/\\*".to_owned();
    let mut debug = false;
    let mut path = String::new();
    let mut ind = 1usize;
    while argv.len() - ind > 2 {
        let argument = &argv[ind];
        if argument.starts_with('-') {
            match argument.as_bytes().get(1).copied() {
                Some(b'f') => {
                    fort77 = true;
                    doc_start = "^[Cc!][ \\t]*!".to_owned();
                    doc_end = doc_start.clone();
                    doc_continue = "[cC!]".to_owned();
                    non_doc_comment = "^[Cc!]".to_owned();
                }
                Some(b'D') => debug = true,
                Some(b'd') => {
                    ind += 1;
                    if ind >= argv.len() {
                        eprintln!("ERROR: {PROGNAME} - unknown argument {argument}");
                        usage(PROGNAME);
                    }
                    path = argv[ind].clone();
                    path.push(std::path::MAIN_SEPARATOR);
                }
                _ => {
                    eprintln!("ERROR: {PROGNAME} - unknown argument {argument}");
                    usage(PROGNAME);
                }
            }
        } else {
            eprintln!("ERROR: {PROGNAME} - too many arguments");
            usage(PROGNAME);
        }
        ind += 1;
    }

    let input_name = &argv[ind];
    // `QFile` opened `QIODevice::Text` read through a `QTextStream`: the
    // device drops every `\r` (measured: `a\rb\r\n` reads as `ab`), the
    // stream skips a leading UTF-8 byte-order mark, and the codec is UTF-8
    // under every locale tried (C, POSIX, C.UTF-8, en_US.ISO-8859-1 give
    // identical output), with each byte of an invalid sequence becoming one
    // U+FFFD (measured: a truncated 4-byte sequence gives three).
    // `IMOD/libwarp/nnpi.c` is ISO-8859.
    let qt_text_read = |bytes: &[u8]| -> String {
        let bytes: Vec<u8> = bytes
            .iter()
            .copied()
            .filter(|&byte| byte != b'\r')
            .collect();
        let mut rest: &[u8] = &bytes;
        if rest.starts_with(b"\xef\xbb\xbf") {
            rest = &rest[3..];
        }
        let mut text = String::with_capacity(rest.len());
        while !rest.is_empty() {
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    text.push_str(valid);
                    break;
                }
                Err(error) => {
                    let good = error.valid_up_to();
                    text.push_str(std::str::from_utf8(&rest[..good]).unwrap());
                    text.push('\u{FFFD}');
                    rest = &rest[good + 1..];
                }
            }
        }
        text
    };
    let mut input_bytes = Vec::new();
    if File::open(input_name)
        .and_then(|mut file| file.read_to_end(&mut input_bytes))
        .is_err()
    {
        eprintln!("ERROR: {PROGNAME} - cannot open input file {input_name}");
        crate::imod::libcfshr::b3dutil::exit(1);
    }
    let input = qt_text_read(&input_bytes);
    let output_name = &argv[ind + 1];
    let mut output = match File::create(output_name) {
        Ok(file) => file,
        Err(_) => {
            eprintln!("ERROR: {PROGNAME} - cannot open output file {output_name}");
            crate::imod::libcfshr::b3dutil::exit(1);
        }
    };

    let doc_start_re = Regex::new(&doc_start).expect("source documentation start regexp");
    let doc_end_re = Regex::new(&doc_end).expect("source documentation end regexp");
    let non_doc_comment_re =
        Regex::new(&non_doc_comment).expect("non-documentation comment regexp");
    let code_comment_re = Regex::new(&code_comment).expect("code comment regexp");
    let doc_continue_re = Regex::new(&doc_continue).expect("documentation continuation regexp");
    let function_open_re = Regex::new(r"\) *\{").unwrap();
    let function_semicolon_re = Regex::new(r"\) *;").unwrap();
    let function_name_open_re = Regex::new(r" *\(").unwrap();
    let function_name_prefix_re = Regex::new(r"[* :]").unwrap();

    let input_lines = input.lines().collect::<Vec<_>>();
    let mut descriptions = Vec::<String>::new();
    let mut file_names = Vec::<String>::new();
    let mut section_names = Vec::<String>::new();
    let mut file_start = vec![0usize];

    for input_line in input_lines {
        let mut line = input_line.to_owned();
        let mut do_code = false;
        let mut ind1 = line.find(LIST_FUNCTIONS_FROM);
        let mut ind2 = line.find(DESCRIBE_FUNCTIONS_FROM);
        if ind1.is_none() && ind2.is_none() {
            ind1 = line.find(LIST_CODE_FROM);
            ind2 = line.find(DESCRIBE_CODE_FROM);
            do_code = ind1.is_some() || ind2.is_some();
        }
        if let Some(list_index) = ind1 {
            let marker = if do_code {
                LIST_CODE_FROM
            } else {
                LIST_FUNCTIONS_FROM
            };
            let remainder = line[list_index + marker.len()..].trim().to_owned();
            let fields = if remainder.starts_with('"') {
                remainder
                    .split('"')
                    .filter(|field| !field.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            } else {
                remainder
                    .split(' ')
                    .filter(|field| !field.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            };
            // `tmpList[0]` of an empty list (no file name after the marker)
            // is undefined in the source; report it as the missing file.
            if fields.is_empty() {
                eprintln!("ERROR: {PROGNAME} - cannot open source file ");
                crate::imod::libcfshr::b3dutil::exit(1);
            }
            let file_name = fields[0].clone();
            let section_name = if fields.len() > 1 {
                if remainder.starts_with('"') {
                    fields[1].trim().to_owned()
                } else {
                    fields[1].clone()
                }
            } else {
                String::new()
            };
            if debug {
                println!("{file_name}");
                println!("{section_name}");
                println!("{}{}", path, file_name);
            }
            let mut source_bytes = Vec::new();
            if File::open(format!("{path}{file_name}"))
                .and_then(|mut file| file.read_to_end(&mut source_bytes))
                .is_err()
            {
                eprintln!("ERROR: {PROGNAME} - cannot open source file {file_name}");
                crate::imod::libcfshr::b3dutil::exit(1);
            }
            let source = qt_text_read(&source_bytes);
            let source_lines = source.lines().collect::<Vec<_>>();
            let mut source_index = 0usize;
            let mut in_section = false;
            while source_index < source_lines.len() {
                line = source_lines[source_index].to_owned();
                source_index += 1;
                if line.contains(DOC_SECTION) {
                    in_section = section_name.is_empty() || line.contains(&section_name);
                    continue;
                }
                if line.contains(END_SECTION) {
                    in_section = false;
                    continue;
                }
                if (!section_name.is_empty() && !in_section)
                    || (section_name.is_empty() && in_section)
                {
                    continue;
                }
                if !do_code && doc_start_re.is_match(&line) {
                    let mut documentation = Vec::<String>::new();
                    line = line.trim().to_owned();
                    // `str.right(str.length() - ind - 1)`: with no `!` (`ind` -1)
                    // `right` of the full length returns the whole string.
                    if let Some(bang) = line.find('!') {
                        line = line[bang + 1..].to_owned();
                    }
                    if debug {
                        println!("{line}");
                    }
                    loop {
                        if let Some(found) = doc_end_re.find(&line) {
                            if fort77 {
                                if let Some(bang) = line.find('!') {
                                    line = line[bang + 1..].to_owned();
                                }
                            } else {
                                line.truncate(found.start());
                            }
                            line = line.trim().to_owned();
                            if !line.is_empty() {
                                documentation.push(line);
                            }
                            break;
                        }
                        if !documentation.is_empty() || !line.is_empty() {
                            documentation.push(line);
                        }
                        if source_index >= source_lines.len() {
                            eprintln!(
                                "ERROR: {PROGNAME} - end of file {file_name} in middle of comment"
                            );
                            crate::imod::libcfshr::b3dutil::exit(1);
                        }
                        line = source_lines[source_index].to_owned();
                        source_index += 1;
                        if debug {
                            println!("{line}");
                        }
                    }

                    let mut function = Vec::<String>::new();
                    let mut in_comment = false;
                    loop {
                        if source_index >= source_lines.len() {
                            eprintln!(
                                "ERROR: {PROGNAME} - end of file {file_name} in middle of function"
                            );
                            crate::imod::libcfshr::b3dutil::exit(1);
                        }
                        line = source_lines[source_index].to_owned();
                        source_index += 1;
                        if debug {
                            println!("{line}");
                        }
                        if function.is_empty() && (in_comment || non_doc_comment_re.is_match(&line))
                        {
                            if !fort77 && (in_comment || line.contains("/*")) {
                                in_comment = !line.contains("*/");
                            }
                            line.clear();
                            continue;
                        }
                        line = line.trim().to_owned();
                        if let Some(found) = function_open_re.find(&line) {
                            if found.start() > 0 {
                                line.truncate(found.start() + 1);
                            }
                        }
                        if let Some(found) = function_semicolon_re.find(&line) {
                            if found.start() > 0 {
                                line.truncate(found.start() + 1);
                            }
                        }
                        // `str.right(str.length() - 1)` drops the first character;
                        // on an empty string `right(-1)` returns it unchanged.
                        if fort77 && !function.is_empty() {
                            let mut chars = line.chars();
                            chars.next();
                            line = chars.as_str().trim().to_owned();
                        }
                        if !line.is_empty() || !function.is_empty() {
                            function.push(line.clone());
                        }
                        if line.ends_with(')') {
                            break;
                        }
                    }
                    let first = &function[0];
                    if debug {
                        println!("{first}");
                    }
                    // `ind2 = str.indexOf(QREGEXP(" *\\("))`, -1 when absent.
                    let ind2: i64 = function_name_open_re
                        .find(first)
                        .map_or(-1, |value| value.start() as i64);
                    if debug {
                        println!("ind2 {ind2}");
                    }
                    // `ind1 = str.lastIndexOf(QREGEXP("[* :]"), ind2 - 1) + 1`:
                    // `lastIndexOf` takes the last match starting at or before
                    // `from`, and a negative `from` counts back from the end.
                    let mut from = ind2 - 1;
                    if from < 0 {
                        from += first.len() as i64;
                    }
                    let ind1: i64 = function_name_prefix_re
                        .find_iter(first)
                        .filter(|value| value.start() as i64 <= from)
                        .last()
                        .map_or(-1, |value| value.start() as i64)
                        + 1;
                    let prefix = ind1 as usize;
                    // `str.mid(ind1, ind2 - ind1)`: a negative length runs to the end.
                    let function_name = if ind2 - ind1 < 0 {
                        first[prefix..].to_owned()
                    } else {
                        first[prefix..ind2 as usize].to_owned()
                    };
                    // `str.right(str.length() - ind2)`: for `ind2` -1 the
                    // whole string.
                    let open = ind2.max(0) as usize;
                    if debug {
                        println!("{function_name}");
                    }
                    if function_name != NO_FUNCTION {
                        let mut entry = "<BR>".to_owned();
                        if prefix != 0 {
                            entry.push_str(&first[..prefix]);
                        }
                        entry
                            .push_str(&format!("<A HREF=\"#{function_name}\">{function_name}</A>"));
                        entry.push_str(first[open..].trim());
                        writeln!(output, "{entry}").unwrap();
                        for item in function.iter().skip(1) {
                            writeln!(output, "{item}").unwrap();
                        }
                        for (number, item) in function.iter().enumerate() {
                            let mut description = String::new();
                            if number == 0 {
                                description
                                    .push_str(&format!("<H3><A NAME=\"{function_name}\"></A>"));
                            }
                            description.push_str(item);
                            if number + 1 == function.len() {
                                description.push_str("</H3><P>");
                            }
                            descriptions.push(description);
                        }
                    } else {
                        descriptions.push("<P>\n".to_owned());
                    }
                    if debug {
                        println!("Adding docs to desc");
                    }
                    for mut item in documentation {
                        item = item.trim().to_owned();
                        while doc_continue_re
                            .find(&item)
                            .is_some_and(|found| found.start() == 0)
                        {
                            item = item[1..].trim().to_owned();
                        }
                        convert_special_codes(&mut item, PROGNAME, debug);
                        descriptions.push(item);
                    }
                    descriptions.push("</P>\n".to_owned());
                } else if do_code && line.contains(DOC_CODE) {
                    let code_index = line.find(DOC_CODE).unwrap();
                    let mut function_name = line[code_index + DOC_CODE.len()..].to_owned();
                    if let Some(found) = doc_end_re.find(&function_name) {
                        function_name.truncate(found.start());
                    }
                    function_name = function_name.trim().to_owned();
                    if debug {
                        println!("{function_name}");
                    }
                    writeln!(
                        output,
                        "<BR><A HREF=\"#{function_name}\">{function_name}</A>"
                    )
                    .unwrap();
                    descriptions.push(format!(
                        "<H3><A NAME=\"{function_name}\"></A>{function_name}</H3>"
                    ));
                    let mut in_comment = false;
                    let mut code_out = false;
                    loop {
                        if source_index >= source_lines.len() {
                            eprintln!(
                                "ERROR: {PROGNAME} - end of file {file_name} in middle of code"
                            );
                            crate::imod::libcfshr::b3dutil::exit(1);
                        }
                        line = source_lines[source_index].to_owned();
                        source_index += 1;
                        if debug {
                            println!("{line}");
                        }
                        if line.contains(END_CODE) {
                            break;
                        }
                        if !code_out && !in_comment && code_comment_re.is_match(&line) {
                            in_comment = true;
                        }
                        if in_comment {
                            if let Some(found) = doc_end_re.find(&line) {
                                in_comment = false;
                                line.truncate(found.start());
                            }
                            if let Some(star) = line.find('*') {
                                line = line[star + 1..].to_owned();
                            }
                            line = line.trim().to_owned();
                            if !line.is_empty() {
                                convert_special_codes(&mut line, PROGNAME, debug);
                                descriptions.push(line);
                            }
                        } else {
                            line = line
                                .replace('&', "&amp;")
                                .replace('<', "&lt;")
                                .replace('>', "&gt;");
                            descriptions.push(line);
                        }
                        if !code_out && !in_comment {
                            descriptions.push("<BR><PRE>".to_owned());
                            code_out = true;
                        }
                    }
                    if code_out {
                        descriptions.push("</PRE>".to_owned());
                    }
                }
            }
            file_names.push(file_name);
            section_names.push(section_name);
            file_start.push(descriptions.len());
        } else if let Some(description_index) = ind2 {
            let marker = if do_code {
                DESCRIBE_CODE_FROM
            } else {
                DESCRIBE_FUNCTIONS_FROM
            };
            let remainder = line[description_index + marker.len()..].trim().to_owned();
            let fields = remainder
                .split(' ')
                .filter(|field| !field.is_empty())
                .collect::<Vec<_>>();
            let file_name = fields.first().copied().unwrap_or("");
            let section_name = fields.get(1).copied().unwrap_or("");
            let mut match_index = None;
            for number in 0..file_names.len() {
                if file_name == file_names[number] && section_name == section_names[number] {
                    match_index = Some(number);
                    break;
                }
            }
            if let Some(number) = match_index {
                for description in &descriptions[file_start[number]..file_start[number + 1]] {
                    writeln!(output, "{description}").unwrap();
                }
            } else {
                eprintln!(
                    "ERROR: {PROGNAME} - function descriptions requested from {file_name} but no such\nfile processed for function list"
                );
                crate::imod::libcfshr::b3dutil::exit(1);
            }
        } else {
            writeln!(output, "{line}").unwrap();
        }
    }
    0
}

/// Translation of `convertSpecialCodes`.
pub fn convert_special_codes(string: &mut String, progname: &str, debug: bool) {
    *string = string
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    // `sourcedoc.cpp:430-438`: every character gets the not-after-a-backslash
    // rule, but only `[`, `{` and `^` also get a start-of-line rule.
    for (character, replacement, at_start) in [
        ('[', "<B>", true),
        (']', "</B>", false),
        ('{', "<I>", true),
        ('}', "</I>", false),
        ('^', "<BR>", true),
    ] {
        let expression = Regex::new(&format!(
            "{CAPTURE_NONBS}{}",
            regex::escape(&character.to_string())
        ))
        .unwrap();
        *string = expression
            .replace_all(string, format!("$1{replacement}"))
            .into_owned();
        if at_start && string.starts_with(character) {
            string.replace_range(..character.len_utf8(), replacement);
        }
    }
    if string.contains("  ") {
        let expression = Regex::new(r"([^ ]) ([^ ])").unwrap();
        *string = expression.replace_all(string, "$1%_%$2").into_owned();
        *string = expression.replace_all(string, "$1%_%$2").into_owned();
        *string = string.replace(' ', "&nbsp;").replace("%_%", " ");
    }
    if string.is_empty() {
        *string = "<P>".to_owned();
    }
    if debug {
        println!("{string}");
    }
    let unescaped_at = Regex::new(r"[^\\]@").unwrap();
    while string.starts_with('@') || unescaped_at.is_match(string) {
        let initial = if string.starts_with('@') {
            1
        } else {
            // `indexOf(QREGEXP("[^\\\\]@")) + 2`: the position after the `@`.
            unescaped_at.find(string).unwrap().end()
        };
        let mut start = initial;
        let bytes = string.as_bytes();
        if bytes.get(start) == Some(&b'@') {
            start += 1;
            while bytes.get(start) == Some(&b' ') {
                start += 1;
            }
            // `str.indexOf('@', ind1 + 1)` is -1 when `ind1 + 1` is past the end.
            let end = string
                .get(start + 1..)
                .and_then(|rest| rest.find('@'))
                .map(|value| start + 1 + value);
            let Some(end) = end else {
                eprintln!("ERROR: {progname} - Empty or unterminated @@ link in\n{string}");
                crate::imod::libcfshr::b3dutil::exit(1);
            };
            let function_name = string[start..end].to_owned();
            let href = if function_name.contains('#') {
                function_name.clone()
            } else {
                format!("#{function_name}")
            };
            let visible = if let Some(hash) = function_name.find('#') {
                function_name[hash + 1..].to_owned()
            } else {
                function_name
            };
            let mut changed = String::new();
            if initial > 0 {
                changed.push_str(&string[..initial - 1]);
            }
            changed.push_str(&format!("<A HREF=\"{href}\">{visible}</A>"));
            if end + 1 < string.len() {
                changed.push_str(&string[end + 1..]);
            }
            *string = changed;
        } else {
            while bytes.get(start) == Some(&b' ') {
                start += 1;
            }
            let end = string[start..]
                .find(|character: char| matches!(character, ' ' | ',' | ';' | '(' | ')'))
                .map_or(string.len(), |value| start + value);
            let function_name = string[start..end].to_owned();
            let href = if function_name.contains('#') {
                function_name.clone()
            } else {
                format!("#{function_name}")
            };
            let visible = if let Some(hash) = function_name.find('#') {
                function_name[hash + 1..].to_owned()
            } else {
                function_name
            };
            let mut changed = String::new();
            if initial > 0 {
                changed.push_str(&string[..initial - 1]);
            }
            changed.push_str(&format!("<A HREF=\"{href}\">{visible}</A>"));
            if end < string.len() {
                changed.push_str(&string[end..]);
            }
            *string = changed;
        }
    }
    let escaped = Regex::new(r"\\([\[\]\{\}\^@])").unwrap();
    *string = escaped.replace_all(string, "$1").into_owned();
    if debug {
        println!("{string}");
    }
}
