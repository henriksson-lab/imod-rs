//! Owned translation of `IMOD/raptor/lasik/svl/lib/base/svlStrUtils.{h,cpp}`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Display;
use std::str::FromStr;

/// Header template `toString(const T&)`.
pub fn svl_to_string<T: Display>(value: T) -> String {
    value.to_string()
}

/// Header `toString(const vector<T>&)`, retaining SVL's leading spaces.
pub fn svl_to_string_slice<T: Display>(values: &[T]) -> String {
    let mut output = String::new();
    for value in values {
        output.push(' ');
        output.push_str(&value.to_string());
    }
    output
}

/// Header `toString(const set<T>&)`, using Rust's ordered set equivalent.
pub fn svl_to_string_set<T: Display + Ord>(values: &BTreeSet<T>) -> String {
    let mut output = String::from("{");
    for value in values {
        output.push(' ');
        output.push_str(&value.to_string());
    }
    output.push_str(" }");
    output
}

/// Header `toString(const deque<T>&)`.
pub fn svl_to_string_deque<T: Display>(values: &VecDeque<T>) -> String {
    let mut output = String::new();
    for value in values {
        output.push(' ');
        output.push_str(&value.to_string());
    }
    output
}

/// Header `toString(const pair<T, U>&)`.
pub fn svl_to_string_pair<T: Display, U: Display>(value: &(T, U)) -> String {
    format!("({}, {})", value.0, value.1)
}

/// Source `toString(const map<string, string>&)`, using the ordered Rust map
/// that has the same key-order contract as `std::map`.
pub fn svl_to_string_map(values: &BTreeMap<String, String>) -> String {
    values
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Header `parseInfToken` specialization point for `parse_string`.
pub trait ParseInfToken: FromStr {
    fn parse_inf_token(token: &str) -> Option<Self>;
}

macro_rules! no_inf_token {
    ($($type:ty),+ $(,)?) => {
        $(
            impl ParseInfToken for $type {
                fn parse_inf_token(_: &str) -> Option<Self> { None }
            }
        )+
    };
}

no_inf_token!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, bool, String);

impl ParseInfToken for f32 {
    fn parse_inf_token(token: &str) -> Option<Self> {
        match token {
            "-inf" => Some(f32::NEG_INFINITY),
            "inf" => Some(f32::INFINITY),
            _ => None,
        }
    }
}

impl ParseInfToken for f64 {
    fn parse_inf_token(token: &str) -> Option<Self> {
        match token {
            "-inf" => Some(f64::NEG_INFINITY),
            "inf" => Some(f64::INFINITY),
            _ => None,
        }
    }
}

/// Header template `parseString`. Values parsed before the first invalid token
/// are appended to `values`, exactly as the C++ stream loop does.
pub fn parse_string<T: ParseInfToken>(input: &str, values: &mut Vec<T>) -> usize {
    let mut count = 0;
    for token in input.split_whitespace() {
        let value = match token.parse::<T>() {
            Ok(value) => Some(value),
            Err(_) => T::parse_inf_token(token),
        };
        let Some(value) = value else {
            break;
        };
        values.push(value);
        count += 1;
    }
    count
}

/// Source `strNoCaseCompare`, with ASCII case folding matching the source's
/// byte-oriented `toupper` calls for SVL's configuration strings.
pub fn str_no_case_compare(first: &str, second: &str) -> i32 {
    let first = first.as_bytes();
    let second = second.as_bytes();
    for (&left, &right) in first.iter().zip(second) {
        let left = left.to_ascii_uppercase();
        let right = right.to_ascii_uppercase();
        if left != right {
            return if left < right { -1 } else { 1 };
        }
    }
    if first.len() == second.len() {
        0
    } else if first.len() < second.len() {
        -1
    } else {
        1
    }
}

/// Source `parseNameValueString`.
pub fn parse_name_value_string(input: &str) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for token in input.split(|character| matches!(character, ',' | ';' | ' ')) {
        if token.is_empty() {
            continue;
        }
        if let Some((name, value)) = token.split_once('=') {
            result.insert(name.to_owned(), value.to_owned());
        } else {
            result.insert(token.to_owned(), "true".to_owned());
        }
    }
    result
}

/// Source `padString`, with a Unicode character rather than a C byte.
pub fn pad_string(input: &str, length: usize, pad_character: char) -> String {
    let missing = length.saturating_sub(input.chars().count());
    std::iter::repeat_n(pad_character, missing).collect::<String>() + input
}

/// Source `strReplaceSubstr`.
pub fn str_replace_substr(input: &str, substring: &str, replacement: &str) -> String {
    if substring.is_empty() {
        return input.to_owned();
    }
    input.replace(substring, replacement)
}

/// Source `strFilename`.
pub fn str_filename(full_path: &str) -> String {
    full_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// Source `strWithoutExt`.
pub fn str_without_ext(full_path: &str) -> String {
    full_path.rsplit_once('.').map_or_else(
        || full_path.to_owned(),
        |(without_extension, _)| without_extension.to_owned(),
    )
}

/// Source `strBaseName`.
pub fn str_base_name(full_path: &str) -> String {
    str_without_ext(&str_filename(full_path))
}

/// Source `strDirectory`.
pub fn str_directory(full_path: &str) -> String {
    full_path
        .rfind(['/', '\\'])
        .map_or_else(|| ".".to_owned(), |index| full_path[..index].to_owned())
}

/// Source `strExtension`.
pub fn str_extension(full_path: &str) -> String {
    str_filename(full_path)
        .rsplit_once('.')
        .map_or_else(String::new, |(_, extension)| extension.to_owned())
}

/// Source `strReplaceExt`. As in the C++ code, `extension` is appended as
/// supplied; callers include a dot when they require one.
pub fn str_replace_ext(full_path: &str, extension: &str) -> String {
    let old_extension = str_extension(full_path);
    let retained = if old_extension.is_empty() {
        full_path
    } else {
        &full_path[..full_path.len() - old_extension.len() - 1]
    };
    retained.to_owned() + extension
}

/// Source `strWithoutEndSlashes`; only forward slashes are stripped there.
pub fn str_without_end_slashes(full_path: &str) -> String {
    full_path.trim_end_matches('/').to_owned()
}

/// Source `strFileIndex`, including its `atoi` behaviour of accepting the
/// numeric prefix between the first and final digit.
pub fn str_file_index(full_path: &str) -> i32 {
    let base_name = str_base_name(full_path);
    let Some(first) = base_name.find(|character: char| character.is_ascii_digit()) else {
        return -1;
    };
    let Some(last) = base_name.rfind(|character: char| character.is_ascii_digit()) else {
        return -1;
    };
    base_name[first..=last]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_template_forms_preserve_their_text_layout() {
        assert_eq!(svl_to_string(12), "12");
        assert_eq!(svl_to_string_slice(&[1, 2]), " 1 2");
        assert_eq!(svl_to_string_pair(&(3, "x")), "(3, x)");
        let mut set = BTreeSet::new();
        set.extend([3, 1]);
        assert_eq!(svl_to_string_set(&set), "{ 1 3 }");
        let map = BTreeMap::from([
            ("b".to_owned(), "2".to_owned()),
            ("a".to_owned(), "1".to_owned()),
        ]);
        assert_eq!(svl_to_string_map(&map), "a=1, b=2");
    }

    #[test]
    fn source_parsers_stop_at_invalid_tokens_and_accept_inf() {
        let mut values = vec![1.0f64];
        assert_eq!(parse_string("2 -inf inf nope 4", &mut values), 3);
        assert_eq!(values.len(), 4);
        assert!(values[2].is_infinite() && values[2].is_sign_negative());
        assert!(values[3].is_infinite() && values[3].is_sign_positive());
        assert_eq!(
            parse_name_value_string("first=1, flag; second =split"),
            BTreeMap::from([
                ("".to_owned(), "split".to_owned()),
                ("first".to_owned(), "1".to_owned()),
                ("flag".to_owned(), "true".to_owned()),
                ("second".to_owned(), "true".to_owned()),
            ])
        );
    }

    #[test]
    fn path_and_replacement_rules_match_source_strings() {
        assert_eq!(str_no_case_compare("AbC", "aBc"), 0);
        assert_eq!(str_no_case_compare("ab", "AC"), -1);
        assert_eq!(pad_string("7", 3, '0'), "007");
        assert_eq!(str_replace_substr("a--a", "a", "xy"), "xy--xy");
        assert_eq!(str_base_name("dir.a/file12.png"), "file12");
        assert_eq!(str_filename("dir\\file.txt"), "file.txt");
        assert_eq!(str_directory("dir\\file.txt"), "dir");
        assert_eq!(str_extension("dir.a/file12.png"), "png");
        assert_eq!(str_replace_ext("file.txt", ".dat"), "file.dat");
        assert_eq!(str_without_ext("dir.a/file"), "dir");
        assert_eq!(str_without_end_slashes("abc///"), "abc");
        assert_eq!(str_file_index("dir/file12a3.txt"), 12);
    }
}
