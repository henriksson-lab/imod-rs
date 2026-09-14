//! `IMOD/Etomo/src/etomo/util/Utilities.java`.
//!
//! A class containing utility methods.  The class is `static`-only (its constructor is
//! private and empty), so every member is a free function here and the `private static`
//! fields are module statics.
//!
//! **JDK translations.**  The source leans on `java.io.File`, `java.util.Date`,
//! `java.text.DecimalFormat`/`SimpleDateFormat` and `java.util.regex`.  The
//! `java_io_*` / `java_util_*` / `java_lang_*` / `java_text_*` functions below are
//! translations of those JDK members, in the same shape as the `java_lang_*` functions
//! `etomo/type/ConstEtomoNumber.java`'s module already carries; they are not helpers
//! that decompose `Utilities` itself.
//!
//! **String indices.**  Java indexes `String` by UTF-16 code unit and this module
//! indexes by byte.  The two agree for the ASCII file names, labels and command lines
//! these functions are given; they diverge only for non-ASCII input, where Java's
//! `substring` can also split a surrogate pair.
//!
//! **Frontier.**  Members that reach Swing (`printComponents`,
//! `findMessageAndOpenDialog`), a manager (`BaseManager`), the process layer
//! (`SystemProgram`, `BaseProcessManager`, `MRCHeader`), the primative tokenizer, or a
//! type with no module carry a `// TODO(unit):` comment naming the blocking source.
#![allow(dead_code)]

use chrono::{Local, TimeZone};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer;
use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
use crate::imod::etomo::ui::swing::token::{Token, Type as TokenType};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use regex::Regex;

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    ConstEtomoNumber, Number, Type, java_lang_double_to_string, java_lang_string_trim,
};
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::environment_variable;

// ---------------------------------------------------------------------------
// JDK members the source uses.
// ---------------------------------------------------------------------------

/// `java.io.File`'s `UnixFileSystem.normalize(String)`: collapse runs of `/` into one
/// and drop a trailing `/` unless the whole path is `/`.  Every `new File(String)` in
/// the source runs this on its argument.
pub fn java_io_file_normalize(pathname: &str) -> String {
    let mut normalized = String::with_capacity(pathname.len());
    let mut previous_was_separator = false;
    for c in pathname.chars() {
        if c == '/' {
            if previous_was_separator {
                continue;
            }
            previous_was_separator = true;
        } else {
            previous_was_separator = false;
        }
        normalized.push(c);
    }
    if normalized.len() > 1 && normalized.ends_with('/') {
        normalized.pop();
    }
    normalized
}

/// `java.io.File(String, String)`: `UnixFileSystem.resolve(parent, child)`.  Note that
/// an absolute `child` is *not* treated specially - it is appended to the parent, which
/// is not what Rust's `Path::join` does.
pub fn java_io_file_new(parent: &str, child: &str) -> String {
    let parent = java_io_file_normalize(parent);
    let child = java_io_file_normalize(child);
    if child.is_empty() {
        return parent;
    }
    if child.starts_with('/') {
        if parent == "/" {
            return child;
        }
        return parent + &child;
    }
    if parent == "/" {
        return parent + &child;
    }
    parent + "/" + &child
}

/// `java.io.File.getName()`.
pub fn java_io_file_get_name(pathname: &str) -> String {
    let path = java_io_file_normalize(pathname);
    let prefix_length = if path.starts_with('/') { 1 } else { 0 };
    match path.rfind('/') {
        Some(index) if index >= prefix_length => path[index + 1..].to_string(),
        _ => path[prefix_length..].to_string(),
    }
}

/// `java.io.File.getParent()`.
pub fn java_io_file_get_parent(pathname: &str) -> Option<String> {
    let path = java_io_file_normalize(pathname);
    let prefix_length = if path.starts_with('/') { 1 } else { 0 };
    match path.rfind('/') {
        Some(index) if index >= prefix_length => Some(path[..index].to_string()),
        _ => {
            if prefix_length > 0 && path.len() > prefix_length {
                return Some(path[..prefix_length].to_string());
            }
            None
        }
    }
}

/// `java.io.File.getAbsolutePath()`.  A relative path is resolved against the process's
/// working directory, which is what the JVM's `user.dir` holds at startup.
pub fn java_io_file_get_absolute_path(pathname: &str) -> String {
    let path = java_io_file_normalize(pathname);
    if path.starts_with('/') {
        return path;
    }
    let user_dir = std::env::current_dir()
        .map(|dir| dir.to_string_lossy().to_string())
        .unwrap_or_default();
    java_io_file_new(&user_dir, &path)
}

/// `java.io.File.lastModified()`: milliseconds since the epoch, or 0 when the file does
/// not exist or cannot be read.
pub fn java_io_file_last_modified(pathname: &str) -> i64 {
    match std::fs::metadata(pathname).and_then(|metadata| metadata.modified()) {
        Ok(modified) => match modified.duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_millis() as i64,
            Err(_) => 0,
        },
        Err(_) => 0,
    }
}

/// `java.lang.System.currentTimeMillis()` / `new java.util.Date().getTime()`.
pub fn java_lang_system_current_time_millis() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i64,
        Err(_) => 0,
    }
}

/// `java.util.Date.toString()`: `EEE MMM dd HH:mm:ss zzz yyyy` with the English day and
/// month abbreviations, a zero-padded day of month, and the platform's short time zone
/// name.
pub fn java_util_date_to_string(millis: i64) -> String {
    Local
        .timestamp_millis_opt(millis)
        .single()
        .expect("Java Date milliseconds outside chrono's supported range")
        .format("%a %b %d %H:%M:%S %Z %Y")
        .to_string()
}

/// `new java.text.SimpleDateFormat("MMMdd-HHmmss", new Locale("en","US")).format(Date)`.
pub fn java_text_simple_date_format_mmmdd_hhmmss(millis: i64) -> String {
    Local
        .timestamp_millis_opt(millis)
        .single()
        .expect("Java Date milliseconds outside chrono's supported range")
        .format("%b%d-%H%M%S")
        .to_string()
}

/// `java.lang.FloatingDecimal.BinaryToASCIIConverter.digitsRoundedUp()` and
/// `decimalDigitsExact()` for a finite, non-negative `double`: whether the shortest
/// round-tripping decimal (`Double.toString`) is greater than the value's exact decimal
/// expansion, and whether it equals it.  `DigitList.set(boolean, double, int, boolean)`
/// hands both to `shouldRoundUp`, so they decide `DecimalFormat`'s HALF_EVEN ties.
fn java_lang_floating_decimal_conversion_flags(magnitude: f64) -> (bool, bool) {
    if magnitude == 0.0 {
        return (false, true);
    }
    // The exact decimal expansion of the double.  A `double` is m * 2^e, so the
    // expansion is finite; 1081 significant digits covers every finite value, including
    // the subnormals, whose expansions are the longest.
    let exact = format!("{:.*e}", 1080, magnitude);
    let (exact_mantissa, exact_exponent) = exact.split_once('e').unwrap();
    let exact_digits: Vec<u8> = exact_mantissa
        .bytes()
        .filter(|b| *b != b'.')
        .map(|b| b - b'0')
        .collect();
    let exact_exponent: i32 = exact_exponent.parse().unwrap();
    // The shortest round-tripping decimal, in the same normalised form.
    let shortest = java_lang_double_to_string(magnitude);
    let (shortest_mantissa, shortest_exponent) = match shortest.split_once('E') {
        Some((mantissa, exponent)) => (mantissa.to_string(), exponent.parse::<i32>().unwrap()),
        None => (shortest.clone(), 0),
    };
    let (shortest_integer, shortest_fraction) = shortest_mantissa
        .split_once('.')
        .unwrap_or((&shortest_mantissa, ""));
    let mut shortest_digits: Vec<u8> = Vec::new();
    shortest_digits.extend(shortest_integer.bytes().map(|b| b - b'0'));
    shortest_digits.extend(shortest_fraction.bytes().map(|b| b - b'0'));
    let mut shortest_exponent = shortest_exponent + shortest_integer.len() as i32 - 1;
    // Drop leading zeros so both are 0.d1d2... * 10^(exponent + 1) with d1 non-zero.
    while shortest_digits.first() == Some(&0) {
        shortest_digits.remove(0);
        shortest_exponent -= 1;
    }
    if shortest_exponent != exact_exponent {
        return (shortest_exponent > exact_exponent, false);
    }
    let length = shortest_digits.len().max(exact_digits.len());
    for index in 0..length {
        let shortest_digit = shortest_digits.get(index).copied().unwrap_or(0);
        let exact_digit = exact_digits.get(index).copied().unwrap_or(0);
        if shortest_digit != exact_digit {
            return (shortest_digit > exact_digit, false);
        }
    }
    (false, true)
}

/// `new java.text.DecimalFormat(".000").format(double)`: no required integer digit,
/// exactly three fraction digits, `RoundingMode.HALF_EVEN`, no grouping.  This is
/// `DigitList.set(boolean, double, 3, true)` followed by `DigitList.round` and
/// `DecimalFormat.subformat`, over the digits of `Double.toString` - `DigitList` rounds
/// the shortest round-tripping decimal, not the double's exact binary value.
pub fn java_text_decimal_format_three_fraction_digits(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "\u{221E}" } else { "-\u{221E}" }.to_string();
    }
    let maximum_fraction_digits = 3i32;
    let minimum_integer_digits = 0i32;
    let magnitude = value.abs();
    let (already_rounded, value_exact_as_decimal) =
        java_lang_floating_decimal_conversion_flags(magnitude);
    // `DigitList.set(boolean, String, boolean, boolean, int, boolean)`.
    let source = java_lang_double_to_string(magnitude);
    let bytes = source.as_bytes();
    let mut digits: Vec<u8> = Vec::new();
    let mut decimal_at: i32 = -1;
    let mut count: usize = 0;
    let mut exponent: i32 = 0;
    // Number of zeros between decimal point and first non-zero digit after decimal
    // point, for numbers < 1.
    let mut leading_zeros_after_decimal: i32 = 0;
    let mut non_zero_digit_seen = false;
    let mut index = 0usize;
    while index < bytes.len() {
        let c = bytes[index];
        index += 1;
        if c == b'.' {
            decimal_at = count as i32;
        } else if c == b'e' || c == b'E' {
            exponent = source[index..].parse().unwrap_or(0);
            break;
        } else {
            if !non_zero_digit_seen {
                non_zero_digit_seen = c != b'0';
                if !non_zero_digit_seen && decimal_at != -1 {
                    leading_zeros_after_decimal += 1;
                }
            }
            if non_zero_digit_seen {
                digits.push(c - b'0');
                count += 1;
            }
        }
    }
    if decimal_at == -1 {
        decimal_at = count as i32;
    }
    if non_zero_digit_seen {
        decimal_at += exponent - leading_zeros_after_decimal;
    }
    // `shouldRoundUp(int, boolean, boolean)` for RoundingMode.HALF_EVEN.
    let should_round_up = |maximum_digits: usize, digits: &Vec<u8>, count: usize| -> bool {
        if digits.get(maximum_digits).copied().unwrap_or(0) > 5 {
            return true;
        } else if digits.get(maximum_digits).copied().unwrap_or(0) == 5 {
            if maximum_digits == count - 1 {
                // The rounding position is exactly the last index.
                if already_rounded {
                    // FloatingDecimal rounded up (value was below the tie), so do not
                    // round up again.
                    return false;
                }
                if !value_exact_as_decimal {
                    // The digits do not represent the exact value: the value was above
                    // the tie and FloatingDecimal truncated to the tie.  Round up.
                    return true;
                }
                // An exact tie, with all digits provided: apply half-even.
                return maximum_digits > 0 && digits[maximum_digits - 1] % 2 != 0;
            } else {
                // The rounding position is not the last index.  If any further digit is
                // non-zero, round up.
                for i in maximum_digits + 1..count {
                    if digits[i] != 0 {
                        return true;
                    }
                }
            }
        }
        false
    };
    // The fixedPoint underflow branches, which run before trailing zeros are dropped.
    let mut underflowed = false;
    if -decimal_at > maximum_fraction_digits {
        // Underflow to zero, e.g. rounding 0.0009 to two fraction digits.
        count = 0;
        digits.clear();
        underflowed = true;
    } else if -decimal_at == maximum_fraction_digits {
        // Rounding 0.0009 to three fraction digits has to create a new digit in the
        // least significant location.
        if should_round_up(0, &digits, count) {
            count = 1;
            decimal_at += 1;
            digits.clear();
            digits.push(1);
        } else {
            count = 0;
            digits.clear();
        }
        underflowed = true;
    }
    if !underflowed {
        // Eliminate trailing zeros.
        while count > 1 && digits[count - 1] == 0 {
            count -= 1;
        }
        digits.truncate(count);
        // `DigitList.round(int, boolean, boolean)`.
        let mut maximum_digits = maximum_fraction_digits + decimal_at;
        if maximum_digits >= 0 && (maximum_digits as usize) < count {
            if should_round_up(maximum_digits as usize, &digits, count) {
                // Rounding up increments digits from LSD to MSD; in the worst case
                // (9999..99) the exponent has to move.
                loop {
                    maximum_digits -= 1;
                    if maximum_digits < 0 {
                        digits[0] = 1;
                        decimal_at += 1;
                        maximum_digits = 0;
                        break;
                    }
                    digits[maximum_digits as usize] += 1;
                    if digits[maximum_digits as usize] <= 9 {
                        break;
                    }
                }
                maximum_digits += 1;
            }
            count = maximum_digits as usize;
            digits.truncate(count);
        }
        // Eliminate trailing zeros.
        while count > 1 && digits[count - 1] == 0 {
            count -= 1;
        }
        digits.truncate(count);
    }
    // `DecimalFormat.subformat(..., isInteger = false, ...)`.
    let mut result = String::new();
    if value.is_sign_negative() {
        result.push('-');
    }
    let mut digit_index = 0usize;
    let mut integer_count = minimum_integer_digits;
    if decimal_at > 0 && integer_count < decimal_at {
        integer_count = decimal_at;
    }
    for i in (0..integer_count).rev() {
        if i < decimal_at && digit_index < count {
            result.push((b'0' + digits[digit_index]) as char);
            digit_index += 1;
        } else {
            result.push('0');
        }
    }
    result.push('.');
    for i in 0..maximum_fraction_digits {
        if -1 - i > decimal_at - 1 {
            result.push('0');
            continue;
        }
        if digit_index < count {
            result.push((b'0' + digits[digit_index]) as char);
            digit_index += 1;
        } else {
            result.push('0');
        }
    }
    result
}

/// `java.lang.String.indexOf(String, int)`.
pub fn java_lang_string_index_of_from(string: &str, needle: &str, from_index: i64) -> i64 {
    let from = from_index.max(0) as usize;
    if from > string.len() {
        return if needle.is_empty() {
            string.len() as i64
        } else {
            -1
        };
    }
    match string[from..].find(needle) {
        Some(index) => (from + index) as i64,
        None => -1,
    }
}

/// `java.lang.String.lastIndexOf(String, int)`: the largest index not greater than
/// `fromIndex` at which `needle` starts.
pub fn java_lang_string_last_index_of_from(string: &str, needle: &str, from_index: i64) -> i64 {
    if from_index < 0 {
        return -1;
    }
    let limit = (from_index as usize).min(string.len());
    let end = (limit + needle.len()).min(string.len());
    match string[..end].rfind(needle) {
        Some(index) if index <= limit => index as i64,
        _ => -1,
    }
}

/// `java.lang.String.split(String)` with the default limit of zero: trailing empty
/// strings are removed, and a leading empty string is kept only when the first match is
/// not at index zero with zero width.
pub fn java_lang_string_split(string: &str, pattern: &Regex) -> Vec<String> {
    let mut parts: Vec<String> = pattern.split(string).map(|part| part.to_string()).collect();
    while parts.len() > 1 && parts.last().map(|part| part.is_empty()).unwrap_or(false) {
        parts.pop();
    }
    if parts.len() == 1 && parts[0].is_empty() && !string.is_empty() {
        parts.clear();
    }
    parts
}

/// `java.lang.String.split(String, int)` with an explicit limit.  A positive limit
/// applies the pattern at most `limit - 1` times, keeps the whole remainder as the last
/// element, and keeps trailing empty strings; a zero limit behaves as
/// `java_lang_string_split`; a negative limit applies the pattern as often as possible
/// and keeps trailing empty strings.  As in Java 8 and later, a zero-width match at the
/// beginning of the input never produces a leading empty substring.
pub fn java_lang_string_split_limit(string: &str, pattern: &Regex, limit: i32) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut matches = 0i32;
    for found in pattern.find_iter(string) {
        if limit > 0 && matches == limit - 1 {
            break;
        }
        if found.start() == 0 && found.end() == 0 {
            // A zero-width match at the beginning never produces an empty leading
            // substring.
            continue;
        }
        parts.push(string[start..found.start()].to_string());
        start = found.end();
        matches += 1;
    }
    parts.push(string[start..].to_string());
    if limit == 0 {
        while parts.len() > 1 && parts.last().map(|part| part.is_empty()).unwrap_or(false) {
            parts.pop();
        }
        if parts.len() == 1 && parts[0].is_empty() && !string.is_empty() {
            parts.clear();
        }
    }
    parts
}

/// `java.lang.Math.round(double)`.
///
/// The javadoc's `(long) Math.floor(a + 0.5d)` wording has not described the
/// implementation since JDK-8010430, and the difference is observable: for
/// `0.49999999999999994` - the double just below one half - the sum `a + 0.5` rounds up
/// to exactly `1.0`, so the floor form answers 1 while a real JVM answers 0.  This is
/// the bit-manipulating body the runtime executes, verified against the reference JVM.
pub fn java_lang_math_round(value: f64) -> i64 {
    // DoubleConsts.SIGNIFICAND_WIDTH = 53, EXP_BIAS = 1023,
    // EXP_BIT_MASK = 0x7FF0000000000000L, SIGNIF_BIT_MASK = 0x000FFFFFFFFFFFFFL.
    let long_bits = value.to_bits() as i64;
    let biased_exp = (long_bits & 0x7FF0000000000000i64) >> (53 - 1);
    let shift = (53 - 2 + 1023) - biased_exp;
    if (shift & -64) == 0 {
        // shift >= 0 && shift < 64
        let mut r = (long_bits & 0x000FFFFFFFFFFFFFi64) | (0x000FFFFFFFFFFFFFi64 + 1);
        if long_bits < 0 {
            r = r.wrapping_neg();
        }
        ((r >> shift) + 1) >> 1
    } else {
        // a is a NaN, an infinity, or is already an integer of magnitude at least 2^52.
        // Rust's `as` cast for f64 to i64 saturates and maps NaN to 0, exactly as the
        // Java narrowing primitive conversion does.
        value as i64
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Java `ACTION_TAG`.
pub const ACTION_TAG: &str = "Etomo Action: ";
/// Java `ZERO_OR_MORE_WILDCARD`.
pub const ZERO_OR_MORE_WILDCARD: &str = "*";

/// Java `APRIL_FOOLS`.
pub static APRIL_FOOLS: LazyLock<bool> = LazyLock::new(|| {
    java_util_date_to_string(java_lang_system_current_time_millis()).contains("Apr 01 ")
});

/// Java `RETRIEVED_DEBUG`.
static RETRIEVED_DEBUG: AtomicBool = AtomicBool::new(false);
/// Java `debug`.
static DEBUG_FIELD: AtomicBool = AtomicBool::new(false);
/// Java `retrievedSelfTest`.
static RETRIEVED_SELF_TEST: AtomicBool = AtomicBool::new(false);
/// Java `selfTest`.
static SELF_TEST: AtomicBool = AtomicBool::new(false);
/// Java `timestamp`.
static TIMESTAMP: AtomicBool = AtomicBool::new(false);
/// Java `setWindowsOS`.
static SET_WINDOWS_OS: AtomicBool = AtomicBool::new(false);
/// Java `windowsOS`.
static WINDOWS_OS: AtomicBool = AtomicBool::new(false);
/// Java `setMacOS`.
static SET_MAC_OS: AtomicBool = AtomicBool::new(false);
/// Java `macOS`.
static MAC_OS: AtomicBool = AtomicBool::new(false);
/// Java `java1_5`.
static JAVA1_5: AtomicBool = AtomicBool::new(false);
/// Java `setJava1_5`.
static SET_JAVA1_5: AtomicBool = AtomicBool::new(false);
/// Java `startTime`.
static START_TIME: AtomicI64 = AtomicI64::new(0);
/// Java `python3`.
static PYTHON3: Mutex<Option<bool>> = Mutex::new(None);
/// Java `java7`.
static JAVA7: Mutex<Option<bool>> = Mutex::new(None);

// Java `timestampFormat` is `new DecimalFormat(".000")`; the format object is a JDK
// value with no state of its own here, so `getTimestamp` calls
// `java_text_decimal_format_three_fraction_digits` in its place.

/// Java `STARTED_STATUS`.
pub const STARTED_STATUS: &str = " started";
/// Java `FINISHED_STATUS`.
pub const FINISHED_STATUS: &str = "finished";
/// Java `FAILED_STATUS`.
pub const FAILED_STATUS: &str = "  failed";
/// Java `NOTHING_TO_DO_STATUS`.
pub const NOTHING_TO_DO_STATUS: &str = " nothing to do";

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`, read once when the
/// class is initialised.
static DEBUG: LazyLock<bool> =
    LazyLock::new(|| etomo_director::ARGUMENTS.lock().unwrap().is_debug());

/// Java private static field `redhat6`, an `EtomoBoolean2` that defaults to null and is
/// created once by `isRedhat6`.
static REDHAT6: Mutex<Option<EtomoBoolean2>> = Mutex::new(None);

/// Java `EMPTY_PATTERN`: `Pattern.compile("\\s+")`.  Java's `\s` is
/// `[ \t\n\x0B\f\r]`, not Unicode whitespace.
pub static EMPTY_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("\\A(?:[ \t\n\u{0B}\u{0C}\r]+)\\z").unwrap());

/// Java `OPEN_ANGLE_BRACKET`.
const OPEN_ANGLE_BRACKET: char = '<';
/// Java `CLOSE_ANGLE_BRACKET`.
const CLOSE_ANGLE_BRACKET: char = '>';
/// Java `OPEN_MESSAGE_ID_TAG`.
const OPEN_MESSAGE_ID_TAG: &str = "[";
/// Java `MESSAGE_ID_TAG`:
/// `Pattern.compile(".*(\\" + OPEN_MESSAGE_ID_TAG + "[a-zA-Z]{3}[0-9]+\\]).*")`.  Java's
/// `.` excludes `\n`, `\r`, `\u0085`, `\u2028` and `\u2029`; Rust's excludes only `\n`,
/// so the class is written out.
static MESSAGE_ID_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "\\A(?:[^\n\r\u{85}\u{2028}\u{2029}]*(\\[[a-zA-Z]{3}[0-9]+\\])[^\n\r\u{85}\u{2028}\u{2029}]*)\\z",
    )
    .unwrap()
});
/// Java `MESSAGE_ID_TAG_GROUP_INDEX`.
const MESSAGE_ID_TAG_GROUP_INDEX: usize = 1;
/// Java `LABEL_END`.
const LABEL_END: char = ':';
/// Java `OPEN_STACK_ID_TAG`.
const OPEN_STACK_ID_TAG: &str = "(";
/// Java `STACK_ID_TAG`:
/// `Pattern.compile(".*\\" + OPEN_STACK_ID_TAG + "([a-z]{3}[0-9]+)\\).*")`.
static STACK_ID_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "\\A(?:[^\n\r\u{85}\u{2028}\u{2029}]*\\(([a-z]{3}[0-9]+)\\)[^\n\r\u{85}\u{2028}\u{2029}]*)\\z",
    )
    .unwrap()
});
/// Java `STACK_ID_GROUP_INDEX`.
const STACK_ID_GROUP_INDEX: usize = 1;
/// Java `LIMITED_SEGMENT_MAX`.
pub const LIMITED_SEGMENT_MAX: i32 = 7;

/// Java `CLEAN_PRINT`:
/// `CleanPrint.getInstance(true, EtomoDirector.FILE_INFO_CLEAN_PRINT_LABEL)`.  It is
/// read only by `deleteFileOrDirectory` and the commented-out body of `copyFile`, both
/// of which are untranslated below.
static CLEAN_PRINT: LazyLock<crate::imod::etomo::util::clean_print::CleanPrint> =
    LazyLock::new(|| {
        crate::imod::etomo::util::clean_print::CleanPrint::get_instance_blockable(
            true,
            Some(etomo_director::FILE_INFO_CLEAN_PRINT_LABEL),
        )
    });

// TODO(unit): needs etomo/process/BaseProcessManager.java - Java `isPython3()` returns
// `BaseProcessManager.isPython3()`, which runs a python process; that unit has no
// module, so the `python3` field stays unread here.

/// Java `isJava7`.
///
/// Deviation: `System.getProperty("java.runtime.version")` names a property of the JVM
/// this program does not run in.  A missing property is `null` in Java, and the source
/// then takes the `isEmpty` branch and stores `false`; that is what this returns.
pub fn is_java7() -> bool {
    let mut java7 = JAVA7.lock().unwrap();
    if let Some(java7) = *java7 {
        return java7;
    }
    let java_runtime_version: Option<String> = None;
    if !is_empty(java_runtime_version.as_deref())
        && java_lang_string_trim(java_runtime_version.as_deref().unwrap()).starts_with("1.7")
    {
        *java7 = Some(true);
    } else {
        *java7 = Some(false);
    }
    java7.unwrap()
}

/// Java `convertWindowsAbsFilePathToCygdrivePath`.  Converts an absolute Windows file
/// path to a unix path, with a cygdrive if there is a drive letter.  Returns null if the
/// path can't be converted, or the converted path.
pub fn convert_windows_abs_file_path_to_cygdrive_path(file_path: Option<&str>) -> Option<String> {
    if !is_windows_os() {
        return None;
    }
    let file_path = match file_path {
        None => {
            eprintln!(
                "Warning:  {} is not an absolute Windows path and cannot be converted to a Cygwin path (msg 1).",
                "null"
            );
            return None;
        }
        Some(file_path) => file_path,
    };
    if file_path.len() < 3 || &file_path[1..3] != ":\\" {
        eprintln!(
            "Warning:  {} is not an absolute Windows path and cannot be converted to a Cygwin path (msg 1).",
            file_path
        );
        return None;
    }
    let file_path = java_lang_string_trim(file_path).to_string();
    let drive_letter = file_path.chars().next().unwrap();
    if !drive_letter.is_alphabetic() {
        eprintln!(
            "Warning:  {} is not an absolute Windows path and cannot be converted to a Cygwin path (msg 2).",
            file_path
        );
        return None;
    }
    // Replace "\" with "/", and replace the drive letter with a cygdrive:
    // Example: C:\ -> /cygdrive/c/
    Some(
        format!("/cygdrive/{}", drive_letter).to_lowercase()
            // File.separatorChar is '\\' on the Windows JVM this branch runs on.
            + &file_path[2..].replace('\\', "/"),
    )
}

/// Java `concatenate`.  Concatenate three strings inserting the insert parameter between
/// them.  Returns null if all three input parameters are null.
pub fn concatenate(
    input1: Option<&str>,
    input2: Option<&str>,
    input3: Option<&str>,
    insert: Option<&str>,
) -> Option<String> {
    if input1.is_none() && input2.is_none() && input3.is_none() {
        return None;
    }
    let mut builder = String::new();
    if let Some(input1) = input1 {
        builder.push_str(input1);
    }
    if let Some(input2) = input2 {
        if let Some(insert) = insert {
            if !builder.is_empty() {
                builder.push_str(insert);
            }
        }
        builder.push_str(input2);
    }
    if let Some(input3) = input3 {
        if let Some(insert) = insert {
            if !builder.is_empty() {
                builder.push_str(insert);
            }
        }
        builder.push_str(input3);
    }
    Some(builder)
}

/// Java `createPropertyKey`.
pub fn create_property_key(prepend: Option<&str>, key: Option<&str>) -> Option<String> {
    let prepend = match prepend {
        None => return key.map(|key| key.to_string()),
        Some(prepend) => prepend,
    };
    if crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(prepend)
    {
        return key.map(|key| key.to_string());
    }
    Some(prepend.to_string() + "." + key.unwrap_or("null"))
}

/// Java `isEmpty(String)`.  Returns true if string is null, empty, or contains nothing
/// but whitespace.
pub fn is_empty(string: Option<&str>) -> bool {
    match string {
        None => true,
        Some(string) => string.is_empty() || EMPTY_PATTERN.is_match(string),
    }
}

/// Java `isEmpty(String[])`.
///
/// Note: the loop body tests `stringArray[0]`, not `stringArray[i]`; that is the
/// source's expression and it is kept.
pub fn is_empty_array(string_array: Option<&[Option<String>]>) -> bool {
    let string_array = match string_array {
        None => return true,
        Some(string_array) => string_array,
    };
    for _i in 0..string_array.len() {
        if is_empty(string_array[0].as_deref()) {
            return true;
        }
    }
    false
}

/// Java `containsWildcard`.  Checks string to see if it contains "*", "?", or a list
/// ([...]).
pub fn contains_wildcard(string: Option<&str>) -> bool {
    static LIST_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            "\\A(?:[^\n\r\u{85}\u{2028}\u{2029}]*\\[[^ \t\n\u{0B}\u{0C}\r]+\\][^\n\r\u{85}\u{2028}\u{2029}]*)\\z",
        )
        .unwrap()
    });
    if is_empty(string) {
        return false;
    }
    let string = java_lang_string_trim(string.unwrap());
    if string.contains(ZERO_OR_MORE_WILDCARD)
        || string.contains('?')
        || LIST_PATTERN.is_match(string)
    {
        return true;
    }
    false
}

/// Java `getRegularExpressionClass`.
pub fn get_regular_expression_class(class_members: Option<&str>) -> Option<String> {
    if !is_empty(class_members) {
        return Some("[".to_string() + class_members.unwrap() + "]");
    }
    None
}

/// Java `getSuffix`.  Returns a suffix that starts with the last instance of tag.  If
/// tag is not found returns null.  Trims whitespace.
pub fn get_suffix(string: Option<&str>, tag: Option<&str>) -> Option<String> {
    strip_string(true, false, string, true, true, tag, true, true)
}

/// Java `removeLeftSide`.  Returns string with left side ending in tag removed.  The
/// first instance of tag is used.
pub fn remove_left_side(string: Option<&str>, tag: Option<&str>) -> Option<String> {
    strip_string(false, false, string, true, false, tag, false, false)
}

/// Java `removeRightSide`.  Returns string with right side starting with tag removed.
/// The last instance of tag is used.
pub fn remove_right_side(string: Option<&str>, tag: Option<&str>) -> Option<String> {
    strip_string(false, false, string, false, false, tag, true, false)
}

/// Java `removeExtension`.  Returns file name with the extension removed.  The "." is
/// removed.  Trims off leading and following spaces from the input string.
pub fn remove_extension(file_path: Option<&str>) -> Option<String> {
    strip_string(
        false,
        false,
        file_path,
        false,
        false,
        Some(extension::EXTENSION_DIVIDER),
        true,
        true,
    )
}

/// Java `extractDataset`.  Returns the dataset from a file.  Assumes the simplest file
/// name format: (DatasetAxis-in-lower-case.extension).  Only looks for axis when dual is
/// set, but doesn't fail if it's not there.
pub fn extract_dataset(dual: bool, file: Option<&std::path::Path>) -> Option<String> {
    let file = match file {
        None => return None,
        Some(file) => file,
    };
    let left_side = remove_extension(Some(&java_io_file_get_name(&file.to_string_lossy())));
    let left_side = match (dual, left_side) {
        (false, left_side) => return left_side,
        (_, None) => return None,
        (_, Some(left_side)) => left_side,
    };
    if left_side.ends_with(&AxisID::First.get_extension()) {
        return Some(left_side[0..left_side.len() - 1].to_string());
    }
    if left_side.ends_with(&AxisID::Second.get_extension()) {
        return Some(left_side[0..left_side.len() - 1].to_string());
    }
    Some(left_side)
}

/// Java `extractAxisID`.
pub fn extract_axis_id(dual: bool, file_name: Option<&str>) -> Option<AxisID> {
    if !dual {
        return Some(AxisID::Only);
    }
    let file_name = match file_name {
        None => return None,
        Some(file_name) => file_name,
    };
    let left_side = match remove_extension(Some(file_name)) {
        None => return None,
        Some(left_side) => left_side,
    };
    if left_side.ends_with(&AxisID::First.get_extension()) {
        return Some(AxisID::First);
    }
    if left_side.ends_with(&AxisID::Second.get_extension()) {
        return Some(AxisID::Second);
    }
    Some(AxisID::Only)
}

/// Java `equalsDataset`.  Returns true if both files contain the same dataset.  To be
/// the same dataset, it must be in the same directory, but the function also allows both
/// parents to be null.
pub fn equals_dataset(
    dual: bool,
    file1: Option<&std::path::Path>,
    file2: Option<&std::path::Path>,
) -> bool {
    let (file1, file2) = match (file1, file2) {
        (None, _) | (_, None) => return false,
        (Some(file1), Some(file2)) => (file1, file2),
    };
    // See if datasets are the same.
    let dataset1 = extract_dataset(dual, Some(file1));
    let dataset2 = extract_dataset(dual, Some(file2));
    match (&dataset1, &dataset2) {
        (None, _) | (_, None) => return false,
        (Some(dataset1), Some(dataset2)) if dataset1 != dataset2 => return false,
        _ => {}
    }
    // See if location is the same.
    let parent1 = java_io_file_get_parent(&file1.to_string_lossy());
    let parent2 = java_io_file_get_parent(&file2.to_string_lossy());
    match (&parent1, &parent2) {
        (None, _) | (_, None) => return parent1.is_none() && parent2.is_none(),
        (Some(parent1), Some(parent2)) => parent1 == parent2,
    }
}

// Deviation: Java `getClassString(Class)` is `getExtension(useClass.toString())` over a
// `java.lang.Class`.  Rust has no runtime class object whose `toString` produces
// `class etomo.type.EtomoNumber`, so there is nothing to translate the argument to and
// the member is left out.

/// Java `toStringIfSet(String, String)`.
pub fn to_string_if_set(label: Option<&str>, value: Option<&str>) -> String {
    if is_empty(value) {
        return "".to_string();
    }
    (if !is_empty(label) { label.unwrap() } else { "" }).to_string() + value.unwrap()
}

/// Java `toStringIfSet(String, AxisID)`.
pub fn to_string_if_set_axis_id(label: Option<&str>, value: Option<AxisID>) -> String {
    let value = match value {
        None => return "".to_string(),
        Some(value) => value,
    };
    (if !is_empty(label) { label.unwrap() } else { "" }).to_string() + &value.to_string()
}

/// Java `toStringIfSet(String, EtomoNumber)`.
pub fn to_string_if_set_etomo_number(label: Option<&str>, value: Option<&EtomoNumber>) -> String {
    let value = match value {
        None => return "".to_string(),
        Some(value) if value.is_null() => return "".to_string(),
        Some(value) => value,
    };
    (if !is_empty(label) { label.unwrap() } else { "" }).to_string() + &value.to_string()
}

/// Java `getExtension`.  Returns the extension of the file path.  The "." is not
/// included.  Trims off leading and following spaces from the input string and removes
/// the path.  Returns null if no extension is found.
pub fn get_extension(file_path: Option<&str>) -> Option<String> {
    let extension = strip_string(
        true,
        true,
        file_path,
        true,
        false,
        Some(extension::EXTENSION_DIVIDER),
        true,
        true,
    );
    match extension {
        None => None,
        Some(extension) if extension.is_empty() => None,
        Some(extension) => Some(extension),
    }
}

/// Java `stripIDs`.  Finds the IDs at the end of a message and returns everything to the
/// left of them.
pub fn strip_ids(message: Option<&str>) -> Option<String> {
    let message = match message {
        None => return None,
        Some(message) => message,
    };
    // Find the locations of the IDs. Use the left most ID.
    let stack_id_tag_index = match message.rfind(OPEN_STACK_ID_TAG) {
        Some(index) => index as i64,
        None => -1,
    };
    let message_id_tag_index = match message.rfind(OPEN_MESSAGE_ID_TAG) {
        Some(index) => index as i64,
        None => -1,
    };
    let mut id_index = stack_id_tag_index;
    let mut pattern: &Regex = &STACK_ID_TAG;
    if stack_id_tag_index == -1
        || (message_id_tag_index != -1 && message_id_tag_index < stack_id_tag_index)
    {
        id_index = message_id_tag_index;
        pattern = &MESSAGE_ID_TAG;
    }
    if id_index == -1 {
        return Some(message.to_string());
    }
    // Make sure this is really the ID.
    if !pattern.is_match(&message[id_index as usize..]) {
        return Some(message.to_string());
    }
    // Return text to the left of the ID.
    Some(message[0..id_index as usize].to_string())
}

/// Java `stripLabel`.  Strips off a label ending in a colon (assumes it's the first
/// colon).  Returns everything to the right of the colon.  Trims the result.  Returns
/// null if there is no colon.
pub fn strip_label(string: Option<&str>) -> Option<String> {
    let string = match string {
        None => return None,
        Some(string) => string,
    };
    let label_end_index = match string.find(LABEL_END) {
        None => return None,
        Some(index) => index,
    };
    let substring = &string[label_end_index + 1..];
    Some(java_lang_string_trim(substring).to_string())
}

/// Java `getMessageIDTag`.  Returns the message ID tag.
pub fn get_message_id_tag(input: Option<&str>, message_code: Option<&str>) -> Option<String> {
    get_end_tag(
        input,
        OPEN_MESSAGE_ID_TAG,
        &MESSAGE_ID_TAG,
        MESSAGE_ID_TAG_GROUP_INDEX,
        message_code,
    )
}

/// Java `getStackID`.  Returns the stack ID.
pub fn get_stack_id(input: Option<&str>, stack_code: Option<&str>) -> Option<String> {
    get_end_tag(
        input,
        OPEN_STACK_ID_TAG,
        &STACK_ID_TAG,
        STACK_ID_GROUP_INDEX,
        stack_code,
    )
}

/// Java `getEndTag`.  Returns one of the tags at the end of a message.
fn get_end_tag(
    input: Option<&str>,
    start_string: &str,
    tag_pattern: &Regex,
    tag_group_index: usize,
    code: Option<&str>,
) -> Option<String> {
    let input = match input {
        None => return None,
        Some(input) => input,
    };
    let tag_index = match input.rfind(start_string) {
        None => return None,
        Some(index) => index,
    };
    // The message ID must be after the last open square bracket.
    let suffix = &input[tag_index..];
    let captures = match tag_pattern.captures(suffix) {
        None => return None,
        Some(captures) => captures,
    };
    let tag = match captures.get(tag_group_index) {
        None => return None,
        Some(tag) => tag.as_str(),
    };
    // Check for matching code.
    if let Some(code) = code {
        if !code.is_empty() && !tag.contains(code) {
            return None;
        }
    }
    Some(tag.to_string())
}

/// Java `buildString`.  Builds a string from input, with the delimiter added between
/// input strings.  Ignores null or empty array elements.  Returns null if the inputArray
/// has nothing in it.
pub fn build_string(input: Option<&[Option<String>]>, delimeter: Option<&str>) -> Option<String> {
    let input = match input {
        None => return None,
        Some(input) if input.is_empty() => return None,
        Some(input) => input,
    };
    let mut builder = String::new();
    for element in input.iter() {
        if let Some(element) = element {
            if !element.is_empty() {
                if !builder.is_empty() {
                    builder.push_str(delimeter.unwrap_or("null"));
                }
                builder.push_str(element);
            }
        }
    }
    if !builder.is_empty() {
        return Some(builder);
    }
    None
}

/// Java `stripRepeatingString`.  Trim and remove all of the remove strings from the left
/// and right side of input.  Trim after each removal.  The remove string is also
/// trimmed.  Returns null if the input ends up null or empty.
/// Example: ".. .. Artie.Chuck .&. Bob." => "Artie.Chuck .&. Bob"
pub fn strip_repeating_string(input: Option<&str>, remove: Option<&str>) -> Option<String> {
    let (input, remove) = match (input, remove) {
        (None, _) => return None,
        (Some(input), remove) if is_empty(remove) => return Some(input.to_string()),
        (Some(input), Some(remove)) => (input, remove),
        (Some(_), None) => unreachable!(),
    };
    if is_empty(Some(input)) {
        return None;
    }
    let mut input = java_lang_string_trim(input).to_string();
    let remove = java_lang_string_trim(remove).to_string();
    let remove_len = remove.len();
    let mut ends_with = false;
    // Each iteration removes one remove string from the left and/or right sides of the
    // input.
    //
    // Note: Java writes the condition as
    // `(startsWith = input.startsWith(remove)) || (endsWith = input.endsWith(remove))`,
    // and `endsWith` is declared before the loop.  When the left operand is true the
    // `||` short-circuits, so `endsWith` keeps the value the *previous* iteration left
    // in it and the body reads that stale value.  That is reproduced here.
    loop {
        let starts_with = input.starts_with(&remove);
        if !starts_with {
            ends_with = input.ends_with(&remove);
        }
        if !(starts_with || ends_with) {
            break;
        }
        let start_index;
        if starts_with {
            start_index = remove_len;
        } else {
            start_index = 0;
        }
        let mut end_index = input.len();
        if ends_with {
            end_index -= remove_len;
        }
        if end_index < start_index {
            // Java's String.substring throws StringIndexOutOfBoundsException when
            // beginIndex > endIndex; the translation stops rather than panicking.
            return None;
        }
        input = input[start_index..end_index].to_string();
        input = java_lang_string_trim(&input).to_string();
        if is_empty(Some(&input)) {
            return None;
        }
    }
    Some(input)
}

/// Java `stripString`.  Returns a string with the left or right side stripped off.
///
/// * `return_null_if_fail` - returns null if tag is not found
/// * `strip_file_path` - ignores and strips the file path
/// * `remove_left_side` - remove a string on the left side ending with tag.  If false
///   removes a string on the right side starting with removeTag.
/// * `keep_tag` - keep the found tag in the result
/// * `tag` - remove a string starting or ending with tag
/// * `last_index_of` - use the last instance of tag rather then the first
/// * `trim` - remove whitespace from the start and end of the string before stripping
pub fn strip_string(
    return_null_if_fail: bool,
    strip_file_path: bool,
    string: Option<&str>,
    remove_left_side: bool,
    keep_tag: bool,
    tag: Option<&str>,
    last_index_of: bool,
    trim: bool,
) -> Option<String> {
    let mut string = match (string, tag) {
        (None, _) | (_, None) => {
            if return_null_if_fail {
                return None;
            }
            return string.map(|string| string.to_string());
        }
        (Some(string), Some(tag)) if string.is_empty() || tag.is_empty() => {
            if return_null_if_fail {
                return None;
            }
            return Some(string.to_string());
        }
        (Some(string), Some(_)) => string.to_string(),
    };
    let tag = tag.unwrap();
    if strip_file_path {
        string = java_io_file_get_name(&string);
    }
    if string.is_empty() {
        if return_null_if_fail {
            return None;
        }
        return Some(string);
    }
    if trim {
        string = java_lang_string_trim(&string).to_string();
    }
    if string.is_empty() {
        if return_null_if_fail {
            return None;
        }
        return Some(string);
    }
    let mut index: i64;
    // Find tag
    if last_index_of {
        index = match string.rfind(tag) {
            Some(index) => index as i64,
            None => -1,
        };
    } else {
        index = match string.find(tag) {
            Some(index) => index as i64,
            None => -1,
        };
    }
    if index == -1 {
        // Tag not found.
        if return_null_if_fail {
            return None;
        }
        return Some(string);
    }
    // Remove based on the location of tag.
    // Remove left side
    if remove_left_side {
        if !keep_tag {
            index += tag.len() as i64;
        }
        return Some(string[index as usize..].to_string());
    }
    // Remove right side
    if keep_tag {
        index += tag.len() as i64;
    }
    Some(string[0..index as usize].to_string())
}

/// Java `getNumberElements`.  Return the number of elements in a comma-divided array.
/// Goal is to match python's element count.
pub fn get_number_elements(text: Option<&str>) -> i32 {
    static COMMAS: LazyLock<Regex> = LazyLock::new(|| Regex::new(",+").unwrap());
    let mut text = match text {
        None => return 0,
        Some(text) if text.is_empty() => return 0,
        Some(text) => text.to_string(),
    };
    while text.starts_with(',') {
        text = text[1..].to_string();
    }
    let array = java_lang_string_split(&text, &COMMAS);
    array.len() as i32
}

// Boundary: Java `printComponents(StringBuilder, Container)` walks a Swing
// `java.awt.Container` and prints `AbstractButton`, `JComboBox`, `JFileChooser`,
// `JInternalFrame`, `JLabel`, `JProgressBar`, `JSpinner` and `JTextComponent` state.
// No etomo unit blocks it - the blocker is javax.swing itself, which the crate does not
// have and which CLAUDE.md puts out of scope, so the member is left untranslated.

/// Java `canWrap`.  Returns false if wrap would return the original string.  It is not
/// required to call this.
pub fn can_wrap(
    string: Option<&str>,
    divider: Option<&str>,
    _min_length: i32,
    wrap_length: i32,
    max_length: i32,
) -> bool {
    match string {
        None => return false,
        Some(string) if string.is_empty() => return false,
        Some(string) => {
            if (string.len() as i32) < max_length
                || string.contains('\n')
                || (max_length <= 0
                    && (divider.is_none()
                        || divider.map(|divider| divider.is_empty()).unwrap_or(false)
                        || wrap_length < 0))
            {
                return false;
            }
        }
    }
    true
}

/// Java `wrap`.  Wraps a string using \n.
pub fn wrap(
    string: Option<&str>,
    divider: Option<&str>,
    mut min_length: i32,
    mut wrap_length: i32,
    max_length: i32,
) -> Option<String> {
    if !can_wrap(string, divider, min_length, wrap_length, max_length) {
        return string.map(|string| string.to_string());
    }
    let string = string.unwrap();
    let mut divider_length = 0i32;
    if let Some(divider) = divider {
        divider_length = divider.len() as i32;
    }
    if wrap_length > 0 && wrap_length < divider_length {
        wrap_length = divider_length;
    }
    // MinLength needs to be 0 when not in use.
    if min_length < 0 {
        min_length = 0;
    }
    // Try to enable max length wrapping.
    let mut use_max_length = false;
    if wrap_length <= 0
        || divider.is_none()
        || divider.map(|divider| divider.is_empty()).unwrap_or(false)
        || !string.contains(divider.unwrap_or(""))
    {
        if max_length > 0 {
            use_max_length = true;
        } else {
            // No way to wrap
            return Some(string.to_string());
        }
    }
    let mut start_index = 0i32;
    let mut eol_index = -1i32;
    let mut lines: Option<String> = None;
    let length = string.len() as i32;
    // Loop until no more line breaks need to be added.
    while start_index + min_length < length {
        if !use_max_length {
            let divider = divider.unwrap();
            eol_index = java_lang_string_last_index_of_from(
                string,
                divider,
                (start_index + wrap_length - divider_length) as i64,
            ) as i32;
            if eol_index >= start_index {
                // Found the largest index up to the wrap lengthndex.
                eol_index += divider_length;
            } else {
                // Look for a longer wrap.
                eol_index = java_lang_string_index_of_from(
                    string,
                    divider,
                    (start_index + wrap_length - divider_length) as i64,
                ) as i32;
                if eol_index >= 0 && (max_length <= 0 || eol_index < start_index + max_length) {
                    // Found a longer wrap.
                    eol_index += divider_length;
                } else if max_length > 0 {
                    // Use maxLength
                    if eol_index < 0 {
                        // No more dividers.
                        use_max_length = true;
                    } else {
                        // Divider is beyond maxLength. Use maxLength just for this line
                        if start_index + max_length < length {
                            eol_index = start_index + max_length;
                        }
                    }
                }
            }
        }
        if use_max_length && start_index + max_length < length {
            eol_index = start_index + max_length;
        }
        // Get the current line
        let line: String;
        if eol_index < 0 {
            // Use the rest of string.
            if start_index == 0 {
                line = java_lang_string_trim(string).to_string();
            } else {
                line = java_lang_string_trim(&string[start_index as usize..]).to_string();
            }
        } else {
            line = java_lang_string_trim(&string[start_index as usize..eol_index as usize])
                .to_string();
        }
        // Add line to lines
        if !line.is_empty() {
            match &mut lines {
                None => {
                    lines = Some(line.clone());
                }
                Some(lines) => {
                    lines.push('\n');
                    lines.push_str(&line);
                }
            }
        }
        if eol_index == -1 {
            if let Some(lines) = lines {
                return Some(lines);
            }
            return Some(string.to_string());
        }
        start_index = eol_index;
        eol_index = -1;
    }
    let mut lines = match lines {
        // No dividers found and maxLength not set.
        None => return Some(string.to_string()),
        Some(lines) => lines,
    };
    // Add the last line.
    let line = java_lang_string_trim(&string[start_index as usize..]).to_string();
    if !line.is_empty() {
        lines.push('\n');
        lines.push_str(&line);
    }
    Some(lines)
}

/// Java `getElementFromList`.  Returns an element from a comma-divided list stored in a
/// string.  Returns null if the requested element does not exist.
pub fn get_element_from_list(list: Option<&str>, index: i32) -> Option<String> {
    static LIST_DIVIDER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]*,[ \t\n\u{0B}\u{0C}\r]*").unwrap());
    if let Some(list) = list {
        if index >= 0 {
            let array = java_lang_string_split(list, &LIST_DIVIDER);
            if (index as usize) < array.len() {
                return Some(array[index as usize].clone());
            }
        }
    }
    None
}

/// Java `millisToMinAndSecs`.  Convert milliseconds into a string of the format
/// Minutes:Seconds.
pub fn millis_to_min_and_secs(milliseconds: f64) -> String {
    let minutes = (milliseconds / 60000.0).floor() as i64;
    let seconds = ((milliseconds - (minutes as f64 * 60000.0)) / 1000.0).floor() as i64;
    let mut str_seconds = String::new();
    // Add a leading zero if less than 10 seconds
    if seconds < 10 {
        str_seconds = "0".to_string();
    }
    str_seconds += &seconds.to_string();
    minutes.to_string() + ":" + &str_seconds
}

/// Java `roundToMaxDecimalPlaces(Number, ConstEtomoNumber)`.
pub fn round_to_max_decimal_places_number(
    number: Option<Number>,
    max_decimal_places: Option<&ConstEtomoNumber>,
) -> Option<String> {
    let number = match number {
        None => return None,
        Some(number) => number,
    };
    if max_decimal_places.is_none() {
        return Some(number.to_string());
    }
    let mut en_double = EtomoNumber::new_with_type(Some(Type::Double));
    en_double.set_number(Some(number));
    round_to_max_decimal_places(Some(&en_double), max_decimal_places)
}

/// Java `roundToMaxDecimalPlaces(String, ConstEtomoNumber)`.
pub fn round_to_max_decimal_places_string(
    number: Option<&str>,
    max_decimal_places: Option<&ConstEtomoNumber>,
) -> Option<String> {
    let number = match number {
        None => return None,
        Some(number) => number,
    };
    if max_decimal_places.is_none() {
        return Some(number.to_string());
    }
    let mut en_double = EtomoNumber::new_with_type(Some(Type::Double));
    en_double.set_string(Some(number));
    round_to_max_decimal_places(Some(&en_double), max_decimal_places)
}

/// Java `roundToMaxDecimalPlaces(ConstEtomoNumber, ConstEtomoNumber)`.
/// `max_decimal_places` is the max digits allowed to the right of the decimal (may not
/// be negative).  Returns the rounded etomoNumber, or `etomoNumber.toString()` if unable
/// to round.
pub fn round_to_max_decimal_places(
    etomo_number: Option<&ConstEtomoNumber>,
    max_decimal_places: Option<&ConstEtomoNumber>,
) -> Option<String> {
    let decimal = '.';
    let etomo_number = match etomo_number {
        None => return None,
        Some(etomo_number) => etomo_number,
    };
    let s_number = etomo_number.to_string();
    match max_decimal_places {
        None => return Some(s_number),
        Some(max_decimal_places) => {
            if max_decimal_places.is_null()
                || max_decimal_places.is_negative()
                || etomo_number.is_null()
                || etomo_number.get_type() != Type::Double
            {
                return Some(s_number);
            }
        }
    }
    let max_decimal_places = max_decimal_places.unwrap();
    let decimal_index = match s_number.find(decimal) {
        // No decimal places
        None => return Some(s_number),
        Some(decimal_index) => decimal_index,
    };
    let i_max_decimal_places = max_decimal_places.get_int();
    if decimal_index as i64 + i_max_decimal_places as i64 + 1 > s_number.len() as i64 {
        // Does not have too many decimal places.
        return Some(s_number);
    }
    // Avoiding multiplying as a double because that can introduce rounding errors.
    // Move the decimal place by rebuilding the string with the decimal place moved to the
    // right by maxDecimalPlaces.
    if i_max_decimal_places > 0 {
        let i_max_decimal_places = i_max_decimal_places as usize;
        let s_num = (if decimal_index > 0 {
            &s_number[0..decimal_index]
        } else {
            ""
        })
        .to_string()
            + &s_number[decimal_index + 1..decimal_index + 1 + i_max_decimal_places]
            + &decimal.to_string()
            + &s_number[decimal_index + 1 + i_max_decimal_places..s_number.len()];
        let mut moved_decimal_place = EtomoNumber::new_with_type(Some(Type::Double));
        moved_decimal_place.set_string(Some(&s_num));
        // If this doesn't work send a warning and return the original number
        if !moved_decimal_place.is_valid() || moved_decimal_place.is_null() {
            eprintln!(
                "Warning:  unable to round {} to {}({}).  {}",
                s_number,
                i_max_decimal_places,
                s_num,
                moved_decimal_place.get_invalid_reason()
            );
            return Some(s_number);
        }
        // Round the number with the moved decimal place and put the decimal back in its
        // original location.
        let mut rounded = EtomoNumber::new_with_type(Some(Type::Long));
        rounded.set_long(java_lang_math_round(moved_decimal_place.get_double()));
        Some(java_lang_double_to_string(
            rounded.get_long() as f64 / 10f64.powi(i_max_decimal_places as i32),
        ))
    } else {
        // Remove all decimal places
        let mut rounded = EtomoNumber::new_with_type(Some(Type::Long));
        rounded.set_long(java_lang_math_round(etomo_number.get_double()));
        Some(rounded.to_string())
    }
}

/// Java `fileExists(BaseManager, String, AxisID)`.  Check to see if the particular
/// dataset file exists; returns true if the file exists.
pub fn file_exists(
    manager: &'static dyn BaseManager,
    extension: Option<&str>,
    axis_id: Option<AxisID>,
) -> bool {
    let file = std::path::PathBuf::from(java_io_file_new(
        &manager
            .get_property_user_dir()
            .unwrap_or("null".to_string()),
        &format!(
            "{}{}{}",
            manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.get_name())
                .unwrap_or("null".to_string()),
            axis_id
                .map(|axis_id| axis_id.get_extension())
                .unwrap_or_default(),
            extension.unwrap_or("null")
        ),
    ));
    if file.exists() {
        return true;
    }
    false
}

// TODO(unit): needs etomo/BaseManager.java and etomo/ui/swing/UIHarness.java - all three
// `getFile(BaseManager, ...)` overloads (Utilities.java:1061, 1074, 1086) read
// `manager.getPropertyUserDir()` / `manager.getName()` and the two with `mustExist` pop
// up `UIHarness.INSTANCE.openMessageDialog`.  The `FileType` overload additionally
// needs the untranslated instance half of etomo/type/FileType.java.

/// Java `getFile(String, String)`.
pub fn get_file(property_user_dir: &str, filename: Option<&str>) -> std::path::PathBuf {
    let filename = match filename {
        None => return std::path::PathBuf::from(java_io_file_normalize(property_user_dir)),
        Some(filename) => filename,
    };
    if crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(filename)
    {
        return std::path::PathBuf::from(java_io_file_normalize(property_user_dir));
    }
    let filename = java_lang_string_trim(filename);
    // File.separatorChar
    if filename.starts_with('/') {
        return std::path::PathBuf::from(java_io_file_normalize(filename));
    }
    if is_windows_os() {
        let drive_index = filename.find(':');
        if let Some(drive_index) = drive_index {
            if drive_index < filename.len() - 1
                && filename[drive_index + 1..].starts_with(std::path::MAIN_SEPARATOR)
            {
                return std::path::PathBuf::from(java_io_file_normalize(filename));
            }
        }
    }
    std::path::PathBuf::from(java_io_file_new(property_user_dir, filename))
}

// TODO(unit): needs etomo/BaseManager.java and etomo/process/SystemProgram.java - Java
// `isValidStack(File, BaseManager, AxisID)` reads the stack through
// `MRCHeader.getInstance(manager.getPropertyUserDir(), file.getName(), axisID)` and then
// `MRCHeader.read(manager)`.  `etomo/util/mrc_header.rs` now carries the class, but that
// `getInstance` overload and `read` are blocked on those two units.

/// Java `backupFile(File)`.
pub fn backup_file(source: Option<&std::path::Path>) -> Result<(), LogFileError> {
    let source = match source {
        None => return Ok(()),
        Some(source) => source,
    };
    rename_file(
        None,
        None,
        Some(source),
        Some(&std::path::PathBuf::from(format!(
            "{}~",
            java_io_file_get_absolute_path(&source.to_string_lossy())
        ))),
        false,
        false,
        false,
    )?;
    Ok(())
}

/// Java `renameFileSafely`.  Rename a file working around the Windows bug.  Never
/// delete a file in this function.  If the destination file exists, fail.
pub fn rename_file_safely(
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    source: Option<&std::path::Path>,
    destination: Option<&std::path::Path>,
) -> Result<bool, LogFileError> {
    if source.is_none() || destination.is_none() {
        return Ok(false);
    }
    let source_handle = LogFile::get_instance_file(
        source,
        match manager {
            None => None,
            Some(manager) => Some(manager.get_emergency_monitor(axis_id)),
        },
    )?;
    source_handle.rename_safely(destination)
}

/// Java `renameFile`.
pub fn rename_file(
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    source: Option<&std::path::Path>,
    destination: Option<&std::path::Path>,
    preserve_dest_file: bool,
    popup_err_msg: bool,
    primary_monitor: bool,
) -> Result<bool, LogFileError> {
    if source.is_none() || destination.is_none() {
        return Ok(false);
    }
    let source_handle = LogFile::get_instance_file(
        source,
        match manager {
            None => None,
            Some(manager) => Some(manager.get_emergency_monitor(axis_id)),
        },
    )?;
    source_handle.rename(
        manager,
        axis_id,
        destination,
        preserve_dest_file,
        popup_err_msg,
        primary_monitor,
    )
}

/// Java `prepareRenameActionMessage(File, File)`.
pub fn prepare_rename_action_message(
    from: Option<&std::path::Path>,
    to: Option<&std::path::Path>,
) -> Option<String> {
    prepare_rename_action_message_always(from, to, false)
}

/// Java `prepareRenameActionMessage(File, File, boolean)`.
pub fn prepare_rename_action_message_always(
    from: Option<&std::path::Path>,
    to: Option<&std::path::Path>,
    always_print_action: bool,
) -> Option<String> {
    if !etomo_director::ARGUMENTS.lock().unwrap().is_actions() && !always_print_action {
        return None;
    }
    Some(
        ACTION_TAG.to_string()
            + "Renamed "
            + &from
                .map(|from| java_io_file_get_name(&from.to_string_lossy()))
                .unwrap_or_default()
            + " to "
            + &to
                .map(|to| java_io_file_get_name(&to.to_string_lossy()))
                .unwrap_or_default(),
    )
}

/// Java `prepareCopyActionMessage`.
pub fn prepare_copy_action_message(
    from: Option<&std::path::Path>,
    to: Option<&std::path::Path>,
) -> Option<String> {
    if !etomo_director::ARGUMENTS.lock().unwrap().is_actions() {
        return None;
    }
    Some(
        ACTION_TAG.to_string()
            + "Copied "
            + &from
                .map(|from| java_io_file_get_name(&from.to_string_lossy()))
                .unwrap_or_default()
            + " to "
            + &to
                .map(|to| java_io_file_get_name(&to.to_string_lossy()))
                .unwrap_or_default(),
    )
}

/// Java `getCommandActionMessage`.  Returns the command action message.  Returns null if
/// the --actions parameter was not set, or the commandAction string is null.
pub fn get_command_action_message(command_action: Option<&str>) -> Option<String> {
    if !etomo_director::ARGUMENTS.lock().unwrap().is_actions() || command_action.is_none() {
        return None;
    }
    Some(ACTION_TAG.to_string() + "Ran " + command_action.unwrap())
}

/// Java `getCommandAction(String[], String[])`.
pub fn get_command_action_array(
    command_array: Option<&[String]>,
    std_input: Option<&[String]>,
) -> Option<String> {
    static NUMBER_LIST: LazyLock<Regex> =
        LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]*[,-][ \t\n\u{0B}\u{0C}\r]*").unwrap());
    let command_array = match command_array {
        None => return None,
        Some(command_array) if command_array.is_empty() => return None,
        Some(command_array) => command_array,
    };
    let command = java_lang_string_trim(&command_array[0]).to_string();
    if (command_array.len() == 1
        && (command == "env"
            || command == "hostname"
            || (command == "tcsh"
                && (std_input.is_none()
                    || std_input.map(|input| input.is_empty()).unwrap_or(false)))))
        || command.ends_with("ssh")
        || command == "ps"
        || command.ends_with("3dmod")
        || command.ends_with("imodsendevent")
    {
        return None;
    }
    let mut command_length = 1usize;

    let mut std_max = 0i32;
    if command.ends_with(&ProcessName::CLIP.to_string()) {
        command_length = 2;
    } else if command.ends_with("bash") {
        command_length = 4;
    } else if command.ends_with("python") {
        if command_array.len() < 3 {
            std_max = -1;
        } else {
            if command_array.len() >= 3 && command_array[2].ends_with("startprocess") {
                command_length = 6;
            } else {
                command_length = 4;
            }
        }
    } else if command.ends_with("tcsh") {
        command_length = 3;
        std_max = 2;
    } else if command.ends_with("cmd.exe") {
        command_length = 3;
    } else if command.starts_with("sh") {
        command_length = 2;
    }
    let mut buffer = String::new();
    let mut param: String;
    let mut ch_index: i64;
    let mut show_dash = false;
    for i in 0..command_array.len() {
        param = command_array[i].clone();
        if param == "None" {
            continue;
        }
        if param.ends_with("alignlog") {
            show_dash = true;
        }
        if !show_dash
            && (param.starts_with('-')
                || (is_windows_os() && param.len() == 2 && param.starts_with('/')))
        {
            continue;
        }
        if i >= command_length {
            // print file names after the command is printed
            // Eliminate strings that aren't file names.
            // Unable to print files that don't have an extension
            if !param.contains('.') {
                continue;
            } else {
                // Eliminate numbers
                if crate::imod::etomo::r#type::const_etomo_number::java_lang_double_value_of(&param)
                    .is_ok()
                {
                    continue;
                }
                // Eliminate lists of numbers
                if param.contains(',') || param.contains('-') {
                    let mut list_of_numbers = true;
                    let list = java_lang_string_split(&param, &NUMBER_LIST);
                    for element in list.iter() {
                        if !element.is_empty()
                            && crate::imod::etomo::r#type::const_etomo_number::java_lang_double_value_of(
                                element,
                            )
                            .is_err()
                        {
                            list_of_numbers = false;
                        }
                    }
                    if list_of_numbers {
                        continue;
                    }
                }
                // Probably a file - remove the file path
                // File.separator
                ch_index = match param.rfind(std::path::MAIN_SEPARATOR) {
                    Some(index) => index as i64,
                    None => -1,
                };
                if is_windows_os() {
                    // Windows paths are sometimes built with /.
                    ch_index = ch_index.max(match param.rfind('/') {
                        Some(index) => index as i64,
                        None => -1,
                    });
                }
                if ch_index != -1 {
                    param = param[ch_index as usize + 1..].to_string();
                }
            }
        } else {
            // File.separator
            ch_index = match param.rfind(std::path::MAIN_SEPARATOR) {
                Some(index) => index as i64,
                None => -1,
            };
            if is_windows_os() {
                // Windows paths are sometimes built with /.
                ch_index = ch_index.max(match param.rfind('/') {
                    Some(index) => index as i64,
                    None => -1,
                });
            }
            if ch_index != -1 {
                param = param[ch_index as usize + 1..].to_string();
            }
        }
        buffer.push_str(&(param + " "));
    }
    if let Some(std_input) = std_input {
        let length: usize;
        if std_max == -1 {
            // Unlimited search
            length = std_input.len();
        } else {
            length = (std_max as usize).min(std_input.len());
        }
        let mut done = false;
        for i in 0..length {
            param = java_lang_string_trim(&std_input[i]).to_string();
            if param.starts_with('#') || param.starts_with("nohup") {
                continue;
            }
            if command.ends_with("python") {
                if param.starts_with("makeBackupFile") {
                    let ch = match param.find(".log") {
                        Some(index) => index as i64,
                        None => -1,
                    };
                    let quote_index = match param.find('\'') {
                        Some(index) => index as i64,
                        None => -1,
                    };
                    if ch != -1 && quote_index != -1 {
                        param = param[quote_index as usize + 1..ch as usize].to_string();
                    }
                    done = true;
                } else {
                    continue;
                }
            } else if param.starts_with("if") {
                let ch = match param.find(".log") {
                    Some(index) => index as i64,
                    None => -1,
                };
                let quote_index = match param.find('"') {
                    Some(index) => index as i64,
                    None => -1,
                };
                if ch != -1 && quote_index != -1 {
                    param = param[quote_index as usize + 1..ch as usize].to_string();
                }
            }
            buffer.push_str(&param);
            if done {
                break;
            }
        }
    }
    Some(java_lang_string_trim(&buffer).to_string())
}

/// Java `getCommandAction(String)`.
pub fn get_command_action(command_line: Option<&str>) -> Option<String> {
    let command_line = match command_line {
        None => return None,
        Some(command_line) if command_line.is_empty() => return None,
        Some(command_line) => command_line,
    };
    let command_line = java_lang_string_trim(command_line).to_string();
    if command_line == "env"
        || command_line == "hostname"
        || command_line.contains("ssh")
        || command_line.contains("ps")
        || command_line.contains("3dmod")
        || command_line.contains("imodsendevent")
    {
        return None;
    }
    Some(command_line)
}

/// Java `prepareDialogActionMessage(DialogType, AxisID, DialogType)`.
pub fn prepare_dialog_action_message(
    dialog_type: Option<DialogType>,
    axis_id: AxisID,
    old_dialog_type: Option<DialogType>,
) -> Option<String> {
    if !etomo_director::ARGUMENTS.lock().unwrap().is_actions()
        || dialog_type.is_none()
        || dialog_type == old_dialog_type
    {
        return None;
    }
    let mut axis = String::new();
    if axis_id != AxisID::Only {
        axis = " in ".to_string() + &axis_id.to_string() + " axis";
    }
    Some(ACTION_TAG.to_string() + "Opened dialog " + &dialog_type.unwrap().to_string() + &axis)
}

/// Java `mostRecentFile(String, String, String, String, String)`.
pub fn most_recent_file(
    property_user_dir: &str,
    file1_name: Option<&str>,
    file2_name: Option<&str>,
    file3_name: Option<&str>,
    file4_name: Option<&str>,
) -> Option<std::path::PathBuf> {
    let mut file1: Option<String> = None;
    let mut file2: Option<String> = None;
    let mut file3: Option<String> = None;
    let mut file4: Option<String> = None;
    if let Some(file1_name) = file1_name {
        file1 = Some(java_io_file_new(property_user_dir, file1_name));
    }
    if let Some(file2_name) = file2_name {
        file2 = Some(java_io_file_new(property_user_dir, file2_name));
    }
    if let Some(file3_name) = file3_name {
        file3 = Some(java_io_file_new(property_user_dir, file3_name));
    }
    if let Some(file4_name) = file4_name {
        file4 = Some(java_io_file_new(property_user_dir, file4_name));
    }
    let mut file1_time = 0i64;
    let mut file2_time = 0i64;
    let mut file3_time = 0i64;
    let mut file4_time = 0i64;
    if let Some(file1) = &file1 {
        if std::path::Path::new(file1).exists() {
            file1_time = java_io_file_last_modified(file1);
        }
    }
    if let Some(file2) = &file2 {
        if std::path::Path::new(file2).exists() {
            file2_time = java_io_file_last_modified(file2);
        }
    }
    if let Some(file3) = &file3 {
        if std::path::Path::new(file3).exists() {
            file3_time = java_io_file_last_modified(file3);
        }
    }
    if let Some(file4) = &file4 {
        if std::path::Path::new(file4).exists() {
            file4_time = java_io_file_last_modified(file4);
        }
    }
    if file1_time >= file2_time && file1_time >= file3_time && file1_time >= file4_time {
        return file1.map(std::path::PathBuf::from);
    }
    if file2_time >= file3_time && file2_time >= file4_time {
        return file2.map(std::path::PathBuf::from);
    }
    if file3_time >= file4_time {
        return file3.map(std::path::PathBuf::from);
    }
    file4.map(std::path::PathBuf::from)
}

/// Java `mostRecentFile(BaseManager, AxisID, FileType[], int)`.
pub fn most_recent_file_type(
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    file_types: Option<&[Option<std::sync::Arc<FileType>>]>,
    default_index: i32,
) -> Option<std::sync::Arc<FileType>> {
    let file_types = match file_types {
        None => return None,
        Some(file_types) => file_types,
    };
    let mut most_recent_index: i32 = -1;
    let mut file: Option<std::path::PathBuf>;
    let mut most_recent_file_time: i64 = 0;
    if default_index >= 0
        && default_index < file_types.len() as i32
        && file_types[default_index as usize].is_some()
    {
        most_recent_index = default_index;
        file = file_types[most_recent_index as usize]
            .as_ref()
            .unwrap()
            .get_file(manager, axis_id);
        if let Some(file) = &file {
            if file.exists() {
                most_recent_file_time = java_io_file_last_modified(&file.to_string_lossy());
            }
        }
    }
    for i in 0..file_types.len() {
        if i as i32 == default_index {
            continue;
        }
        if file_types[i].is_some() {
            file = file_types[i].as_ref().unwrap().get_file(manager, axis_id);
            if let Some(file) = &file {
                if file.exists() {
                    let file_time = java_io_file_last_modified(&file.to_string_lossy());
                    if file_time > most_recent_file_time {
                        most_recent_file_time = file_time;
                        most_recent_index = i as i32;
                    }
                }
            }
        }
    }
    if most_recent_index == -1 {
        return None;
    }
    file_types[most_recent_index as usize].clone()
}

/// Java `copyFile(FileType, FileType, BaseManager, AxisID, boolean, boolean, boolean)`.
pub fn copy_file_file_types(
    source: &std::sync::Arc<FileType>,
    destination: &std::sync::Arc<FileType>,
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    preserve_copy_to_file: bool,
    popup_err_msg: bool,
    primary_monitor: bool,
) -> Result<(), LogFileError> {
    copy_file(
        manager,
        axis_id,
        source.get_file(manager, axis_id).as_deref(),
        destination.get_file(manager, axis_id).as_deref(),
        preserve_copy_to_file,
        popup_err_msg,
        primary_monitor,
    )
}

/// Java `copyFile(File, FileType, BaseManager, AxisID, boolean, boolean, boolean)`.
pub fn copy_file_to_file_type(
    source: Option<&std::path::Path>,
    destination: &std::sync::Arc<FileType>,
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    preserve_copy_to_file: bool,
    popup_err_msg: bool,
    primary_monitor: bool,
) -> Result<(), LogFileError> {
    copy_file(
        manager,
        axis_id,
        source,
        destination.get_file(manager, axis_id).as_deref(),
        preserve_copy_to_file,
        popup_err_msg,
        primary_monitor,
    )
}

/// Java `copyFile(BaseManager, AxisID, File, File, boolean, boolean, boolean)`.  Copy a
/// file using the fastest method available.
pub fn copy_file(
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    source: Option<&std::path::Path>,
    destination: Option<&std::path::Path>,
    preserve_copy_to_file: bool,
    popup_err_msg: bool,
    primary_monitor: bool,
) -> Result<(), LogFileError> {
    if source.is_none() || destination.is_none() {
        return Ok(());
    }
    let source_handle = LogFile::get_instance_file(
        source,
        match manager {
            None => None,
            Some(manager) => Some(manager.get_emergency_monitor(axis_id)),
        },
    )?;
    source_handle.copy(
        manager,
        axis_id,
        destination,
        preserve_copy_to_file,
        popup_err_msg,
        primary_monitor,
    )?;
    Ok(())
}

/// Java `debugPrint(String)`.  Print out the specified string to err if the debug flag
/// is set.
pub fn debug_print(string: &str) {
    debug_print_to_out(string, false);
}

/// Java `debugPrint(String, boolean)`.  Print out the specified string to err or out if
/// the debug flag is set.
pub fn debug_print_to_out(string: &str, to_out: bool) {
    if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
        if to_out {
            println!("{}", string);
        } else {
            eprintln!("{}", string);
        }
    }
}

// TODO(unit): needs etomo/BaseManager.java, etomo/ui/swing/UIHarness.java and
// etomo/process/SystemProgram.java - `deleteFileType(BaseManager, AxisID, FileType)` and
// both `deleteFileOrDirectory` overloads pop up a message dialog and run `b3dremove`
// through a SystemProgram.

/// Java `getStrippedFileName(File)`.  Returns the name of file, stripped of its
/// extension; a bad file causes null to be returned.
pub fn get_stripped_file_name_file(file: Option<&std::path::Path>) -> Option<String> {
    if let Some(file) = file {
        return get_stripped_file_name(Some(&java_io_file_get_name(&file.to_string_lossy())));
    }
    None
}

/// Java `getStrippedFileName(String)`.
pub fn get_stripped_file_name(file_name: Option<&str>) -> Option<String> {
    let file_name = match file_name {
        None => return None,
        Some(file_name) => file_name,
    };
    if crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace(
        file_name,
    ) {
        return None;
    }
    let ext_index = match file_name.rfind('.') {
        None => return Some(file_name.to_string()),
        Some(ext_index) => ext_index,
    };
    Some(file_name[0..ext_index].to_string())
}

/// Java `cleanUpLabel`.  Strips the last colon (and anything following it) and returns
/// it.
pub fn clean_up_label(label: &str) -> String {
    let colon_index = match label.rfind(':') {
        None => return label.to_string(),
        Some(colon_index) => colon_index,
    };
    label[0..colon_index].to_string()
}

/// Java `quoteLabel`.  Strips the last colon (and anything following it) and returns the
/// string in single quotes.
pub fn quote_label(label: Option<&str>) -> Option<String> {
    let label = match label {
        None => return None,
        Some(label) => label,
    };
    let quote = '\'';
    Some(quote.to_string() + &clean_up_label(java_lang_string_trim(label)) + &quote.to_string())
}

/// Java `escapeSpaces(String)`.
pub fn escape_spaces(string: &str) -> String {
    escape_spaces_double(string, false)
}

/// Java `escapeSpaces(String, boolean)`.
pub fn escape_spaces_double(string: &str, double_escape: bool) -> String {
    let mut index = match string.find(' ') {
        None => return string.to_string(),
        Some(index) => index as i64,
    };
    let mut buffer = String::new();
    let mut prev_index = 0i64;
    while index != -1 {
        buffer.push_str(&string[prev_index as usize..index as usize]);
        if !double_escape {
            buffer.push_str("\\ ");
        } else {
            buffer.push_str("\\\\ ");
        }
        index += 1;
        prev_index = index;
        index = java_lang_string_index_of_from(string, " ", prev_index);
    }
    buffer.push_str(&string[prev_index as usize..]);
    buffer
}

/// Java `writeFile(BaseManager, AxisID, File, String[], boolean, boolean)`.
pub fn write_file(
    manager: Option<&'static dyn BaseManager>,
    axis_id: Option<AxisID>,
    file: Option<&std::path::Path>,
    strings: Option<&[String]>,
    new_file: bool,
    backup: bool,
) -> std::io::Result<()> {
    let file = match file {
        None => return Err(std::io::Error::other("java.io.IOException")),
        Some(file) => file,
    };
    if new_file {
        let mut done = false;
        if backup {
            // Try to back up. If it fails, try a delete instead.
            match rename_file(
                manager,
                axis_id,
                Some(file),
                Some(&std::path::PathBuf::from(format!(
                    "{}~",
                    java_io_file_get_absolute_path(&file.to_string_lossy())
                ))),
                false,
                false,
                false,
            ) {
                Ok(renamed) => done = renamed,
                Err(e) => {
                    // `e.printStackTrace()`
                    eprintln!("{}", e);
                }
            }
        }
        if !done {
            // TODO(unit): needs etomo/ui/swing/UIHarness.java and
            // etomo/process/SystemProgram.java - `Utilities.deleteFileOrDirectory(file,
            // manager, axisID)`, which pops up a dialog on failure and removes a
            // directory by running `b3dremove`.
        }
    }
    let strings = match strings {
        None => return Ok(()),
        Some(strings) if strings.is_empty() => return Ok(()),
        Some(strings) => strings,
    };
    let mut buffered_writer = std::io::BufWriter::new(std::fs::File::create(file)?);
    for string in strings {
        std::io::Write::write_all(&mut buffered_writer, string.as_bytes())?;
        // `BufferedWriter.newLine()` writes the `line.separator` property.
        std::io::Write::write_all(&mut buffered_writer, b"\n")?;
    }
    std::io::Write::flush(&mut buffered_writer)?;
    Ok(())
}

/// Java `isValidFile`.  Validates a file and appends the failure reason to
/// invalidReason, which must not be null.
pub fn is_valid_file(
    file: Option<&std::path::Path>,
    file_description: Option<&str>,
    invalid_reason: &mut String,
    exists: bool,
    can_read: bool,
    can_write: bool,
    is_directory: bool,
) -> bool {
    let mut is_valid = true;
    let file = match file {
        None => {
            let non_whitespace = match file_description {
                None => false,
                Some(file_description) => !EMPTY_PATTERN.is_match(file_description),
            };
            if non_whitespace {
                invalid_reason
                    .push_str(&(file_description.unwrap().to_string() + " was not entered.\n"));
            } else if is_directory {
                invalid_reason.push_str("No directory name was entered.\n");
            } else {
                invalid_reason.push_str("No file name was entered.\n");
            }
            return false;
        }
        Some(file) => file,
    };
    let absolute_path = java_io_file_get_absolute_path(&file.to_string_lossy());
    if exists && !file.exists() {
        invalid_reason.push_str(&(absolute_path.clone() + " must exist.\n"));
        is_valid = false;
    }
    if can_read && std::fs::File::open(file).is_err() {
        invalid_reason.push_str(&(absolute_path.clone() + " must be readable.\n"));
        is_valid = false;
    }
    if can_write && std::fs::OpenOptions::new().write(true).open(file).is_err() {
        invalid_reason.push_str(&(absolute_path.clone() + " must be writable.\n"));
        is_valid = false;
    }
    if is_directory && !file.is_dir() {
        invalid_reason.push_str(&(absolute_path.clone() + " must be a directory.\n"));
        is_valid = false;
    }
    if !is_directory && file.is_dir() {
        invalid_reason.push_str(&(absolute_path + " must be a file.\n"));
        is_valid = false;
    }
    is_valid
}

/// Java `buttonTimestamp(String)`.  Print timestamp in error log.
pub fn button_timestamp(command: Option<&str>) {
    timestamp_full(Some("PRESSED"), command, None, None);
}

/// Java `buttonTimestamp(String, String)`.  Print timestamp in error log.
pub fn button_timestamp_container(command: Option<&str>, container: Option<&str>) {
    timestamp_full(Some("PRESSED"), command, container, None);
}

/// Java `timestamp(String, String, String)`.  Print timestamp in error log.
pub fn timestamp_process_container_status(
    process: Option<&str>,
    container: Option<&str>,
    status: Option<&str>,
) {
    timestamp_full(process, None, container, status);
}

/// Java `timestamp(String, ProcessName, String)`.
pub fn timestamp_process_name(
    process: Option<&str>,
    process_name: Option<ProcessName>,
    status: Option<&str>,
) {
    timestamp_full(
        process,
        process_name
            .map(|process_name| process_name.to_string())
            .as_deref(),
        None,
        status,
    );
}

/// Java `timestamp(String, ProcessName, String, String)`.
pub fn timestamp_process_name_subprocess(
    process: Option<&str>,
    process_name: Option<ProcessName>,
    subprocess_name: Option<&str>,
    status: Option<&str>,
) {
    timestamp_full(
        process,
        process_name
            .map(|process_name| process_name.to_string())
            .as_deref(),
        subprocess_name,
        status,
    );
}

/// Java `timestamp(String, String, File, String)`.  Print timestamp in error log.
pub fn timestamp_file(
    process: Option<&str>,
    command: Option<&str>,
    container: Option<&std::path::Path>,
    status: Option<&str>,
) {
    timestamp_full(
        process,
        command,
        container
            .map(|container| java_io_file_get_name(&container.to_string_lossy()))
            .as_deref(),
        status,
    );
}

// TODO(unit): needs etomo/comscript/ComScript.java - Java `timestamp(String, String,
// ComScript, String)` reads `container.getName()` off a `ComScript`, and that unit has
// no module.

/// Java `timestamp(String, String)`.
pub fn timestamp_command_status(command: Option<&str>, status: Option<&str>) {
    timestamp_full(None, command, None, status);
}

/// Java `timestamp(String, String, String, String)`.  Print timestamp in error log.
/// `status` is 0 = started, 1 = finished, -1 = failed, or -100 = null.
pub fn timestamp_full(
    process: Option<&str>,
    command: Option<&str>,
    container: Option<&str>,
    status: Option<&str>,
) {
    // Checking for problems.
    if process.map(|value| value.contains('@')).unwrap_or(false)
        || command.map(|value| value.contains('@')).unwrap_or(false)
        || container.map(|value| value.contains('@')).unwrap_or(false)
        || status.map(|value| value.contains('@')).unwrap_or(false)
    {
        // Deviation: `Thread.dumpStack()` prints the calling Java thread's stack trace
        // to stderr.  A Rust backtrace is neither the same frames nor the same text, so
        // nothing is printed in its place.
    }
    if !TIMESTAMP.load(Ordering::Relaxed) {
        return;
    }
    let mut buffer = "TIMESTAMP: ".to_string();
    if let Some(process) = process {
        buffer.push_str(&(process.to_string() + " "));
    }
    if let Some(command) = command {
        buffer.push_str(&(command.to_string() + " "));
    }
    if let Some(container) = container {
        let mut container = container.to_string();
        // File.separatorChar
        let separator_index = match container.rfind(std::path::MAIN_SEPARATOR) {
            Some(index) => index as i64,
            None => -1,
        };
        if separator_index != -1 && separator_index < container.len() as i64 - 1 {
            container = container[separator_index as usize + 1..].to_string();
        }
        buffer.push_str(&(container + " "));
    }
    if let Some(status) = status {
        buffer.push_str(&(status.to_string() + " "));
    }
    if !is_empty(Some(&buffer)) {
        buffer.push_str("at ");
    }
    buffer.push_str(&get_timestamp());
    eprintln!("{}", buffer);
}

/// Java `timestamp()`.
pub fn timestamp() {
    timestamp_full(None, None, None, None);
}

/// Java `timestamp(String)`.
pub fn timestamp_marker(marker: Option<&str>) {
    timestamp_full(marker, None, None, None);
}

/// Java `isDebug`.
pub fn is_debug() -> bool {
    if !RETRIEVED_DEBUG.load(Ordering::Relaxed) {
        DEBUG_FIELD.store(
            etomo_director::ARGUMENTS.lock().unwrap().is_debug(),
            Ordering::Relaxed,
        );
        RETRIEVED_DEBUG.store(true, Ordering::Relaxed);
    }
    DEBUG_FIELD.load(Ordering::Relaxed)
}

/// Java `setTimestamp`.
pub fn set_timestamp(timestamp: bool) {
    TIMESTAMP.store(timestamp, Ordering::Relaxed);
}

/// Java `isSelfTest`.
pub fn is_self_test() -> bool {
    if !RETRIEVED_SELF_TEST.load(Ordering::Relaxed) {
        SELF_TEST.store(
            etomo_director::ARGUMENTS.lock().unwrap().is_self_test(),
            Ordering::Relaxed,
        );
        RETRIEVED_SELF_TEST.store(true, Ordering::Relaxed);
    }
    SELF_TEST.load(Ordering::Relaxed)
}

/// Java `setStartTime`.
pub fn set_start_time() {
    START_TIME.store(java_lang_system_current_time_millis(), Ordering::Relaxed);
}

/// Java `getDateTimeStamp(boolean)`.
pub fn get_date_time_stamp_ms(include_ms: bool) -> String {
    let date = java_lang_system_current_time_millis();
    java_util_date_to_string(date)
        + &(if include_ms {
            format!(", {} ms", date.rem_euclid(1000))
        } else {
            "".to_string()
        })
}

/// Java `getDateTimeStampRootName`.
pub fn get_date_time_stamp_root_name() -> String {
    java_text_simple_date_format_mmmdd_hhmmss(java_lang_system_current_time_millis())
}

/// Java `getDateTimeStamp()`.
pub fn get_date_time_stamp() -> String {
    get_date_time_stamp_ms(false)
}

/// Java `dateTimeStamp`.
pub fn date_time_stamp() {
    eprintln!("{}", get_date_time_stamp_ms(false));
}

/// Java `managerStamp(String, String)`.
pub fn manager_stamp(property_user_dir: Option<&str>, dataset_name: Option<&str>) {
    manager_stamp_new_window(property_user_dir, dataset_name, false);
}

/// Java `managerStamp(String, String, boolean)`.  Print the date/time and manager info.
/// Use a different formula for switching tabs.
pub fn manager_stamp_new_window(
    property_user_dir: Option<&str>,
    dataset_name: Option<&str>,
    new_window: bool,
) {
    let border: &str = if new_window {
        "++++++++++++++++"
    } else {
        "----------------"
    };
    eprintln!("\n{}", border);
    date_time_stamp();
    // Display only the parent directory of the dataset directory unless verbose is
    // requested.
    if let Some(property_user_dir) = property_user_dir {
        if etomo_director::ARGUMENTS
            .lock()
            .unwrap()
            .get_debug_level()
            .is_verbose()
        {
            eprintln!("{}", property_user_dir);
        } else {
            let dir = java_io_file_normalize(property_user_dir);
            let parent_file = java_io_file_get_parent(&dir);
            if let Some(parent_file) = parent_file {
                eprint!("{}{}", java_io_file_get_name(&parent_file), "/");
            }
            eprintln!("{}", java_io_file_get_name(&dir));
        }
    }
    if let Some(dataset_name) = dataset_name {
        eprintln!("{}", dataset_name);
    }
    eprintln!("{}\n", border);
}

/// Java `getTimestamp`.
pub fn get_timestamp() -> String {
    java_text_decimal_format_three_fraction_digits(
        (java_lang_system_current_time_millis() - START_TIME.load(Ordering::Relaxed)) as f64
            / 1000.0,
    )
}

/// Java `isRedhat6`.
pub fn is_redhat6(manager: Option<&'static dyn BaseManager>, axis_id: Option<AxisID>) -> bool {
    let mut redhat6 = REDHAT6.lock().unwrap();
    if redhat6.is_none() {
        *redhat6 = Some(EtomoBoolean2::new());
        if !is_windows_os() && !is_mac_os() {
            // The Java `try` block; its `catch (final LogFileException | IOException |
            // LockException e) {}` arms do nothing, so every failure just leaves the
            // instance false.
            match LogFile::get_instance_file(
                Some(std::path::Path::new("/etc/redhat-release")),
                match manager {
                    None => None,
                    Some(manager) => Some(manager.get_emergency_monitor(axis_id)),
                },
            ) {
                Err(_) => {}
                Ok(log_file) => match log_file.open_reader() {
                    Err(_) => {}
                    Ok(None) => {}
                    Ok(Some(id)) => match log_file.read_line(&id) {
                        Err(_) => {}
                        Ok(line) => {
                            if let Some(line) = line {
                                if line.contains("release 6") {
                                    redhat6.as_mut().unwrap().set_boolean(true);
                                }
                            }
                        }
                    },
                },
            }
        }
    }
    redhat6.as_ref().unwrap().base.base.base.is()
}

/// Java `isWindowsOS`.
pub fn is_windows_os() -> bool {
    if !SET_WINDOWS_OS.load(Ordering::Relaxed) {
        // The JVM's `os.name`; `std::env::consts::OS` is the same fact for this build.
        let os_name = java_lang_system_get_property_os_name().to_lowercase();
        WINDOWS_OS.store(os_name.contains("windows"), Ordering::Relaxed);
        SET_WINDOWS_OS.store(true, Ordering::Relaxed);
    }
    WINDOWS_OS.load(Ordering::Relaxed)
}

/// Java `isMacOS`.
pub fn is_mac_os() -> bool {
    if !SET_MAC_OS.load(Ordering::Relaxed) {
        let os_name = java_lang_system_get_property_os_name().to_lowercase();
        MAC_OS.store(os_name.contains("mac"), Ordering::Relaxed);
        SET_MAC_OS.store(true, Ordering::Relaxed);
    }
    MAC_OS.load(Ordering::Relaxed)
}

/// `java.lang.System.getProperty("os.name")`.  The JVM fills this from `uname`; the
/// values `isWindowsOS` and `isMacOS` look for are "Windows *" and "Mac OS X".
fn java_lang_system_get_property_os_name() -> &'static str {
    match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "Mac OS X",
        "windows" => "Windows 10",
        "freebsd" => "FreeBSD",
        "netbsd" => "NetBSD",
        "openbsd" => "OpenBSD",
        "solaris" => "SunOS",
        other => other,
    }
}

// TODO(unit): needs etomo/ui/swing/UIHarness.java and etomo/BaseManager.java - Java
// `findMessageAndOpenDialog(BaseManager, AxisID, String[], String, String)` opens an
// info dialog through `UIHarness.INSTANCE`.

/// Java `getExistingDir(BaseManager, String, AxisID, String)`.
pub fn get_existing_dir_message(
    manager: Option<&'static dyn BaseManager>,
    env_variable: Option<&str>,
    axis_id: AxisID,
    not_found_message: Option<&str>,
) -> Option<std::path::PathBuf> {
    let env_variable = match env_variable {
        None => return None,
        // `String.matches("\\s*")`: the whole string is whitespace.
        Some(env_variable) if env_variable.chars().all(char::is_whitespace) => return None,
        Some(env_variable) => env_variable,
    };
    let dir_name =
        environment_variable::INSTANCE.get_value(manager, None, env_variable, Some(axis_id));
    if dir_name.chars().all(char::is_whitespace) {
        return None;
    }
    let dir = std::path::PathBuf::from(&dir_name);
    if !check_existing_dir_message(&dir, Some(env_variable), not_found_message) {
        return None;
    }
    Some(dir)
}

/// Java `getExistingDir(BaseManager, String, AxisID)`.
pub fn get_existing_dir(
    manager: Option<&'static dyn BaseManager>,
    env_variable: Option<&str>,
    axis_id: AxisID,
) -> Option<std::path::PathBuf> {
    let env_variable = match env_variable {
        None => return None,
        Some(env_variable) if env_variable.chars().all(char::is_whitespace) => return None,
        Some(env_variable) => env_variable,
    };
    let dir_name =
        environment_variable::INSTANCE.get_value(manager, None, env_variable, Some(axis_id));
    if dir_name.chars().all(char::is_whitespace) {
        return None;
    }
    let dir = std::path::PathBuf::from(&dir_name);
    if !check_existing_dir(&dir, Some(env_variable)) {
        return None;
    }
    Some(dir)
}

/// Java `checkExistingDir(File, String, String)`.
pub fn check_existing_dir_message(
    dir: &std::path::Path,
    _env_variable: Option<&str>,
    not_found_message: Option<&str>,
) -> bool {
    if !dir.exists() || !dir.is_dir() || std::fs::read_dir(dir).is_err() {
        if *DEBUG {
            eprintln!("{}", not_found_message.unwrap_or("null"));
        }
        return false;
    }
    true
}

/// Java `checkExistingDir(File, String)`.
pub fn check_existing_dir(dir: &std::path::Path, env_variable: Option<&str>) -> bool {
    let absolute_path = java_io_file_get_absolute_path(&dir.to_string_lossy());
    if !dir.exists() {
        if *DEBUG {
            eprintln!(
                "Warning:  {} does not exist.  See ${}.",
                absolute_path,
                env_variable.unwrap_or("null")
            );
        }
        return false;
    }
    if !dir.is_dir() {
        if *DEBUG {
            eprintln!(
                "Warning:  {} is not a directory.  See ${}.",
                absolute_path,
                env_variable.unwrap_or("null")
            );
        }
        return false;
    }
    if std::fs::read_dir(dir).is_err() {
        if *DEBUG {
            eprintln!(
                "Warning:  cannot read {}.  See ${}.",
                absolute_path,
                env_variable.unwrap_or("null")
            );
        }
        return false;
    }
    true
}

/// Java `convertLabelToName(String, String, String, boolean)`.
pub fn convert_label_to_name_three(
    label1: Option<&str>,
    label2: Option<&str>,
    label3: Option<&str>,
    unlimited_segments: bool,
) -> Option<String> {
    let mut buffer = String::new();
    if let Some(label1) = label1 {
        buffer.push_str(&format!("{} ", label1));
    }
    if let Some(label2) = label2 {
        buffer.push_str(&format!("{} ", label2));
    }
    if let Some(label3) = label3 {
        buffer.push_str(&format!("{} ", label3));
    }
    convert_label_to_name(Some(&buffer), unlimited_segments)
}

/// Java `convertLabelToName(String, boolean)`.
pub fn convert_label_to_name(label: Option<&str>, unlimited_segments: bool) -> Option<String> {
    let label = label?;
    // Place the label into a tokenizer
    let mut name = java_lang_string_trim(label).to_lowercase();
    let mut tokenizer = PrimativeTokenizer::get_numeric_string_instance(&name, is_debug());
    let mut buffer = String::new();
    let mut prev_token_type: Option<TokenType> = None;
    let mut token: *mut Token = std::ptr::null_mut();
    // `tokenizer.initialize(); token = tokenizer.next(token);` - the two calls the
    // source wraps in its `catch (LogFileException | IOException | LockException)` arms,
    // both of which return null for an empty label or the default delimiter and the
    // label otherwise.
    match tokenizer.initialize() {
        Err(e) => {
            // `e.printStackTrace()`
            eprintln!("{}", e);
            if label.is_empty() || label == autodoc_tokenizer::DEFAULT_DELIMITER {
                return None;
            }
            return Some(label.to_string());
        }
        Ok(()) => {}
    }
    token = unsafe { tokenizer.next(token) };
    // Remove unnecessary symbols and strings from the label.
    let mut peeked_token: *mut Token = std::ptr::null_mut();
    let mut ignore_paren = false;
    let mut ignore_bracket = false;
    let mut ignore_symbol;
    let mut replace_with_space;
    while !token.is_null()
        && !unsafe { (*token).is(TokenType::Eof) }
        && !unsafe { (*token).is(TokenType::Eol) }
    {
        ignore_symbol = false;
        replace_with_space = false;
        if unsafe { (*token).equals_type_and_char(TokenType::Symbol, NAME_SEPARATOR as u16) } {
            // Convert a dash to a space so that any mix of dashes and whitespace in the
            // original label gets converted to a single dash in the next loop, and the
            // number of elements in the name can be counted.  A dash that's probably part
            // of a number will be left in.  So "check-up" turns into "check up", then back
            // into "check-up".  And "equals -20%" stays the same, and then is changed to
            // "equals--20%".
            peeked_token = unsafe { tokenizer.peek(peeked_token) };
            if peeked_token.is_null() || unsafe { (*peeked_token).get_type() } != TokenType::Numeric
            {
                replace_with_space = true;
            }
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, '.' as u16) } {
            // "." cannot be used because a number that follows it may be mistaken for the
            // field index.  This solution turns "1.0" into "1-0", which seems somewhat
            // better than turning it into "10".
            replace_with_space = true;
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, '\'' as u16) } {
            // Skip "'" when it's used for quoting.  Keep it when it's probably an
            // apostrophe.  If it's at the beginning or the end, or next to whitespace,
            // assume it's being used for quoting.
            if prev_token_type.is_none() || prev_token_type == Some(TokenType::Whitespace) {
                ignore_symbol = true;
            } else {
                peeked_token = unsafe { tokenizer.peek(peeked_token) };
                if peeked_token.is_null() || !unsafe { (*peeked_token).is(TokenType::Alphabetic) } {
                    ignore_symbol = true;
                }
            }
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, ',' as u16) }
            || unsafe { (*token).equals_type_and_char(TokenType::Symbol, '"' as u16) }
        {
            // Skip some other types of punctuation.
            ignore_symbol = true;
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, '(' as u16) } {
            // ignore parenthesis and everything in them
            ignore_paren = true;
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, '<' as u16) } {
            // Replace html (angle brackets and contents) with a space.  The space is
            // necessary when a <br> is used.
            ignore_bracket = true;
            replace_with_space = true;
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, ':' as u16) }
            || unsafe { (*token).equals_type_and_char(TokenType::Symbol, ';' as u16) }
        {
            // ignore semicolons and everything after them
            break;
        }
        if replace_with_space {
            buffer.push(' ');
        }
        if !ignore_paren && !ignore_bracket && !ignore_symbol && !replace_with_space {
            buffer.push_str(unsafe { (*token).get_value() }.unwrap_or("null"));
        }
        if unsafe { (*token).equals_type_and_char(TokenType::Symbol, ')' as u16) } {
            ignore_paren = false;
        } else if unsafe { (*token).equals_type_and_char(TokenType::Symbol, '>' as u16) } {
            ignore_bracket = false;
        }
        prev_token_type = Some(unsafe { (*token).get_type() });
        token = unsafe { tokenizer.next(token) };
    }
    // Load the processed string into the tokenizer
    name = java_lang_string_trim(&buffer).to_string();
    // handle a string with nothing but strippable or illegal characters in it
    if name.is_empty() || name == autodoc_tokenizer::DEFAULT_DELIMITER {
        return None;
    }
    tokenizer = PrimativeTokenizer::get_string_instance(&name, false);
    buffer = String::new();
    match tokenizer.initialize() {
        Err(e) => {
            // `e.printStackTrace()`
            eprintln!("{}", e);
            if name.is_empty() || name == autodoc_tokenizer::DEFAULT_DELIMITER {
                return None;
            }
            return Some(name);
        }
        Ok(()) => {}
    }
    token = unsafe { tokenizer.next(token) };
    // Convert interior whitespace to a single dash
    let mut segment_count = 0;
    while !token.is_null()
        && !unsafe { (*token).is(TokenType::Eof) }
        && !unsafe { (*token).is(TokenType::Eol) }
        && (unlimited_segments || segment_count < LIMITED_SEGMENT_MAX)
    {
        if unsafe { (*token).is(TokenType::Whitespace) } {
            buffer.push(NAME_SEPARATOR);
        } else {
            let mut new_segment = false;
            let string = buffer.clone();
            if string.is_empty() || string.ends_with(NAME_SEPARATOR) {
                new_segment = true;
            }
            buffer.push_str(unsafe { (*token).get_value() }.unwrap_or("null"));
            if new_segment {
                segment_count += 1;
            }
        }
        token = unsafe { tokenizer.next(token) };
    }
    let retval = buffer;
    if retval.is_empty() || retval == autodoc_tokenizer::DEFAULT_DELIMITER {
        return None;
    }
    Some(retval)
}

/// Java `stripHTMLTags`.
///
/// Strips HTML tags from a string.  Does not strip tags with illegal syntax.  A tag that
/// contains a "<" or is empty has illegal syntax.  Examples of illegal syntax: "<>",
/// "<<html>".
///
/// The two `catch` blocks that return `input` unchanged belong to `tokenizer.initialize`
/// and are unreachable for a string instance: only the blocked `logFile` path opens a
/// file, so neither `LogFileException`, `IOException` nor `LockException` can be thrown.
pub fn strip_html_tags(input: Option<&str>) -> Option<String> {
    let input = input?;
    // Place the input into a tokenizer
    let mut tokenizer =
        crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer::get_string_instance(
            input, *DEBUG,
        );
    let mut output = String::new();
    let mut html_tag: Option<String> = None;
    // The `catch` blocks are unreachable for a string instance; see the doc comment.
    let _ = tokenizer.initialize();
    let mut token: Option<Box<crate::imod::etomo::ui::swing::token::Token>> =
        Some(unsafe { Box::from_raw(tokenizer.next(std::ptr::null_mut())) });
    let mut prev_token: Option<Box<crate::imod::etomo::ui::swing::token::Token>> = None;
    while token.is_some()
        && !token
            .as_ref()
            .unwrap()
            .is(crate::imod::etomo::ui::swing::token::Type::Eof)
        && !token
            .as_ref()
            .unwrap()
            .is(crate::imod::etomo::ui::swing::token::Type::Eol)
    {
        let current = token.as_ref().unwrap();
        // Searching for an HTML tag.
        if html_tag.is_none() {
            if current.equals_type_and_char(
                crate::imod::etomo::ui::swing::token::Type::Symbol,
                OPEN_ANGLE_BRACKET as u16,
            ) {
                // Found "<" - might be an HTML tag.
                let mut tag = String::new();
                tag.push_str(current.get_value().unwrap_or("null"));
                html_tag = Some(tag);
            } else {
                // Not part of an HTML tag.
                output.push_str(current.get_value().unwrap_or("null"));
            }
        }
        // Finding the extent of a possible HTML tag.
        else if current.equals_type_and_char(
            crate::imod::etomo::ui::swing::token::Type::Symbol,
            CLOSE_ANGLE_BRACKET as u16,
        ) {
            let prev_is_open = match &prev_token {
                // `prevToken.equals(...)` on a null prevToken throws; the loop always
                // sets it before a second iteration, and the first iteration cannot
                // reach here because `htmlTag` is still null.
                None => panic!("java.lang.NullPointerException"),
                Some(prev_token) => prev_token.equals_type_and_char(
                    crate::imod::etomo::ui::swing::token::Type::Symbol,
                    OPEN_ANGLE_BRACKET as u16,
                ),
            };
            if prev_is_open {
                // Found "<>". An empty tag is not valid HTML. Add it to the output.
                output.push_str(html_tag.as_ref().unwrap());
                html_tag = None;
            } else {
                // Found "<...>". It's a complete HTML tag. Throw it away.
                html_tag = None;
            }
        } else if current.equals_type_and_char(
            crate::imod::etomo::ui::swing::token::Type::Symbol,
            OPEN_ANGLE_BRACKET as u16,
        ) {
            // Found "<...<". A tag with "<" inside it is not valid HTML. Add it to the
            // output.
            output.push_str(html_tag.as_ref().unwrap());
            html_tag = None;
        } else {
            // Continue to build the possible HTML tag.
            html_tag
                .as_mut()
                .unwrap()
                .push_str(current.get_value().unwrap_or("null"));
        }
        // Get next token. Make sure the two tokens don't point to the same thing.
        prev_token = token.take();
        token = Some(unsafe { Box::from_raw(tokenizer.next(std::ptr::null_mut())) });
    }
    // End of the input. If it failed to find the end of an HTML tag, treat it as regular
    // text.
    if let Some(html_tag) = html_tag {
        output.push_str(&html_tag);
    }
    Some(output)
}

/// Java `convertNanometersToMicrons(ConstEtomoNumber)`.
pub fn convert_nanometers_to_microns(nm: Option<&ConstEtomoNumber>) -> ConstEtomoNumber {
    let mut microns = EtomoNumber::new_with_type(Some(Type::Double));
    microns.set_const_etomo_number(nm);
    microns.divide_by_int(1000);
    (*microns).clone()
}

/// Java `convertNanometersToMicrons(Double)`.
pub fn convert_nanometers_to_microns_double(nm: Option<f64>) -> ConstEtomoNumber {
    let mut microns = EtomoNumber::new_with_type(Some(Type::Double));
    if let Some(nm) = nm {
        microns.set_number(Some(Number::Double(nm)));
        microns.divide_by_int(1000);
    }
    (*microns).clone()
}

/// Java `convertMicronsToNanometers`.
pub fn convert_microns_to_nanometers(microns: Option<&str>) -> String {
    let mut nm = EtomoNumber::new_with_type(Some(Type::Double));
    nm.set_string(microns);
    nm.multiply_int(1000);
    nm.to_string()
}

/// Java `NAME_SEPARATOR`.
pub const NAME_SEPARATOR: char = '-';

/// Java `convertPathToUniversalWindows`.
pub fn convert_path_to_universal_windows(path: &str) -> String {
    let new_path = path.replace('\\', "/");
    "\"".to_string() + &new_path + "\""
}

// TODO(unit): needs etomo/BaseManager.java, etomo/process/SystemProgram.java and the
// untranslated instance half of etomo/type/FileType.java - both
// `getStackBinning(BaseManager, AxisID, FileType[, boolean])` overloads read the pixel
// spacing of `stackFileType` and `FileType.RAW_STACK` through `MRCHeader.read`, which is
// blocked on those units.

/// Java `setAlignFramesRootname`.
pub fn set_align_frames_rootname(file: Option<&str>) -> String {
    let file = match file {
        None => return "".to_string(),
        Some(file) => file,
    };
    let split_base_and_extension = java_lang_string_split_last_dot(file);
    if split_base_and_extension.len() > 1
        && split_base_and_extension[1] == extension::CLASS.mdoc.to_string()
    {
        let split_base_and_extension2 =
            java_lang_string_split_last_dot(&split_base_and_extension[0]);
        return split_base_and_extension2[0].clone();
    }
    split_base_and_extension[0].clone()
}

/// `String.split("\\.(?=[^\\.]+$)")`: split on the last "." that is followed by at least
/// one non-"." character and nothing else.
///
/// The `regex` crate has no lookahead, so the pattern is evaluated directly: the split
/// point is the last "." when the text after it is non-empty and contains no ".".
fn java_lang_string_split_last_dot(string: &str) -> Vec<String> {
    if let Some(index) = string.rfind('.') {
        let tail = &string[index + 1..];
        if !tail.is_empty() && !tail.contains('.') {
            // Java's split removes trailing empty strings; the tail is non-empty here.
            return vec![string[..index].to_string(), tail.to_string()];
        }
    }
    if string.is_empty() {
        return vec![String::new()];
    }
    vec![string.to_string()]
}

/// Java `setRootnameSelectedFiles`.
pub fn set_rootname_selected_files(files: &[String]) -> String {
    static DASH_OR_UNDERSCORE: LazyLock<Regex> = LazyLock::new(|| Regex::new("-|_").unwrap());
    if files.is_empty() {
        return "".to_string();
    }
    let rootname_for_selected_files = get_rootname_for_selected_files(files);
    let split_base_and_extension =
        java_lang_string_split(&rootname_for_selected_files, &DASH_OR_UNDERSCORE);
    for i in 0..split_base_and_extension.len() {
        let mut count_zeros = 0usize;
        for c in split_base_and_extension[i].chars() {
            if c == '0' {
                count_zeros += 1;
            }
        }
        if count_zeros == split_base_and_extension[i].len() {
            let mut calc_sub_str_length = 0i64;
            for k in 0..i {
                calc_sub_str_length += split_base_and_extension[k].len() as i64;
            }
            calc_sub_str_length += i as i64 - 1;
            if calc_sub_str_length < 0
                || calc_sub_str_length > rootname_for_selected_files.len() as i64
            {
                // Java's substring throws StringIndexOutOfBoundsException; the source has
                // no guard, and i == 0 with an all-zero first segment reaches -1.
                return rootname_for_selected_files;
            }
            return rootname_for_selected_files[0..calc_sub_str_length as usize].to_string();
        }
    }
    rootname_for_selected_files
}

/// Java `getRootnameForSelectedFiles`.
fn get_rootname_for_selected_files(files: &[String]) -> String {
    let filename1 = &files[0];
    let mut prev_sub_string = filename1[0..0].to_string();
    for i in 0..filename1.len() {
        if !filename1.is_char_boundary(i) {
            continue;
        }
        let curr_sub_string = filename1[0..i].to_string();
        for file in files.iter() {
            if !file.starts_with(&curr_sub_string) {
                return prev_sub_string;
            }
        }
        prev_sub_string = curr_sub_string;
    }
    prev_sub_string
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_date_to_string_formats_the_epoch_in_the_local_timezone() {
        let epoch = Local.timestamp_millis_opt(0).single().unwrap();

        assert_eq!(
            java_util_date_to_string(0),
            epoch.format("%a %b %d %H:%M:%S %Z %Y").to_string()
        );
    }

    #[test]
    fn simple_date_format_uses_english_fields_and_discards_milliseconds() {
        let millis = 1_704_067_245_987;
        let instant = Local.timestamp_millis_opt(millis).single().unwrap();

        assert_eq!(
            java_text_simple_date_format_mmmdd_hhmmss(millis),
            instant.format("%b%d-%H%M%S").to_string()
        );
        assert_eq!(
            java_text_simple_date_format_mmmdd_hhmmss(millis),
            java_text_simple_date_format_mmmdd_hhmmss(millis - 987)
        );
    }
}
