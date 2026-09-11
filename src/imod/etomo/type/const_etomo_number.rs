//! `IMOD/Etomo/src/etomo/type/ConstEtomoNumber.java`.
//!
//! DisplayValue verses defaultValue:
//!
//! The display value is the value returned when the instance is null.  Use the display
//! value to prevent an instance from returning the null value or a blank string.  The
//! display value does not affect the result of the `isNull()` function.
//!
//! The default value is just an extra value that ConstEtomoNumber can store.  "Get"
//! functions with the parameter "boolean defaultIfNull" are convenience functions which
//! return the default value when the instance is null.  Use the default value as a way
//! to store the default of the instance in one place.
//!
//! ScriptParameter uses the default value to decide whether an instance needs to be
//! placed in a script.
//!
//! **Java `Number`.**  The class carries its value in a `java.lang.Number` whose
//! *concrete* class is part of the behaviour: `isNull`, `isNumberNull`, `compare` and
//! `doMath` all branch on `instanceof`.  The `Number` enum below therefore names the
//! wrapper class, not just the width, and `newNumber` builds exactly the class the
//! source's `new Integer` / `Double.valueOf` / `new Long` calls build.
//!
//! **Boundaries.**  `internalTest` and `internalTestDeepCopy` are left untranslated;
//! see the two "Representation limit" comments below for why their bodies cannot be
//! written over a `Number` enum value.
#![allow(dead_code)]

use std::collections::BTreeMap;

/// Java `DOUBLE_NULL_VALUE`.
pub const DOUBLE_NULL_VALUE: f64 = f64::NAN;
/// Java `FLOAT_NULL_VALUE`.
pub const FLOAT_NULL_VALUE: f32 = f32::NAN;
/// Java `INTEGER_NULL_VALUE`.
pub const INTEGER_NULL_VALUE: i32 = i32::MIN;
/// Java `LONG_NULL_VALUE`.
pub const LONG_NULL_VALUE: i64 = i64::MIN;
/// Java `BOOLEAN_NULL_VALUE`.
pub const BOOLEAN_NULL_VALUE: i32 = INTEGER_NULL_VALUE;
/// Java `BOOLEAN_FALSE`.
const BOOLEAN_FALSE: i32 = 0;
/// Java `BOOLEAN_TRUE`.
const BOOLEAN_TRUE: i32 = 1;

/// Java `java.lang.Object.toString()`:
/// `getClass().getName() + "@" + Integer.toHexString(hashCode())`.
///
/// Used by `ConstEtomoNumber()`, `ConstEtomoNumber(Type)` and the null branch of
/// `ConstEtomoNumber(ConstEtomoNumber)` (ConstEtomoNumber.java:86, 105, 131) to default
/// the `name` field.
///
/// Deviation: the identity hash code is assigned by the JVM and differs between runs of
/// the same program, so the hexadecimal half is not reproducible in the sense CLAUDE.md
/// means (it belongs with the uninitialised-memory differentials).  A process-local
/// counter stands in for it so the *shape* of the default name matches
/// `etomo.type.EtomoNumber@1b6d3586`.  `ConstEtomoNumber` is abstract, so the class name
/// is the runtime class; `EtomoNumber` is the only subclass translated here.
fn java_lang_object_to_string() -> String {
    static IDENTITY_HASH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    format!(
        "etomo.type.EtomoNumber@{:x}",
        IDENTITY_HASH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

/// Java `java.lang.String.trim()`: strips every leading and trailing character whose
/// code point is less than or equal to `U+0020`.  This is not Rust's `str::trim`, which
/// strips Unicode whitespace.
pub fn java_lang_string_trim(value: &str) -> &str {
    value.trim_matches(|c: char| c <= ' ')
}

/// Java `java.lang.String.matches("\\s*")`.  Java's `\s` is `[ \t\n\x0B\f\r]`, and
/// `matches` anchors at both ends.
pub fn java_lang_string_matches_whitespace(value: &str) -> bool {
    value
        .chars()
        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
}

/// Java `java.lang.Integer.parseInt(String)` (radix 10), as reached through
/// `new Integer(String)`.  `Err` carries the `NumberFormatException` message, which
/// `newNumber(String, StringBuffer)` appends to its invalid buffer
/// (ConstEtomoNumber.java:1250).
pub fn java_lang_integer_parse_int(value: &str) -> Result<i32, String> {
    // Java accepts an optional leading '+' or '-' and then ASCII digits only; no
    // surrounding whitespace, no underscores, no radix prefix.
    let bytes = value.as_bytes();
    let mut index = 0;
    let negative = if !bytes.is_empty() && (bytes[0] == b'+' || bytes[0] == b'-') {
        index = 1;
        bytes[0] == b'-'
    } else {
        false
    };
    if index >= bytes.len() {
        return Err(format!("For input string: \"{}\"", value));
    }
    // Java accumulates negatively so that Integer.MIN_VALUE is representable.
    let mut result: i32 = 0;
    while index < bytes.len() {
        if !bytes[index].is_ascii_digit() {
            return Err(format!("For input string: \"{}\"", value));
        }
        let digit = (bytes[index] - b'0') as i32;
        result = match result.checked_mul(10).and_then(|r| r.checked_sub(digit)) {
            Some(result) => result,
            None => return Err(format!("For input string: \"{}\"", value)),
        };
        index += 1;
    }
    if negative {
        Ok(result)
    } else {
        match result.checked_neg() {
            Some(result) => Ok(result),
            None => Err(format!("For input string: \"{}\"", value)),
        }
    }
}

/// Java `java.lang.Long.parseLong(String)` (radix 10), as reached through
/// `new Long(String)`.
pub fn java_lang_long_parse_long(value: &str) -> Result<i64, String> {
    let bytes = value.as_bytes();
    let mut index = 0;
    let negative = if !bytes.is_empty() && (bytes[0] == b'+' || bytes[0] == b'-') {
        index = 1;
        bytes[0] == b'-'
    } else {
        false
    };
    if index >= bytes.len() {
        return Err(format!("For input string: \"{}\"", value));
    }
    let mut result: i64 = 0;
    while index < bytes.len() {
        if !bytes[index].is_ascii_digit() {
            return Err(format!("For input string: \"{}\"", value));
        }
        let digit = (bytes[index] - b'0') as i64;
        result = match result.checked_mul(10).and_then(|r| r.checked_sub(digit)) {
            Some(result) => result,
            None => return Err(format!("For input string: \"{}\"", value)),
        };
        index += 1;
    }
    if negative {
        Ok(result)
    } else {
        match result.checked_neg() {
            Some(result) => Ok(result),
            None => Err(format!("For input string: \"{}\"", value)),
        }
    }
}

/// Java `java.lang.Double.valueOf(String)`, which routes through
/// `FloatingDecimal.readJavaFormatString`: the string is trimmed first, an empty result
/// throws `NumberFormatException("empty String")`, and any other syntax error throws
/// `For input string: "<trimmed>"`.  A trailing `f`, `F`, `d` or `D` type suffix is
/// accepted and ignored.
///
/// Deviation: the hexadecimal-significand form the `Double.valueOf` javadoc also accepts
/// (`0x1.8p3`) is not implemented here.  Every caller in the translated units parses a
/// number written by a com script or a `.edf` file, which is always decimal; a hex
/// significand reaches this function as a syntax error rather than as `12.0`.
pub fn java_lang_double_value_of(value: &str) -> Result<f64, String> {
    let trimmed = java_lang_string_trim(value);
    if trimmed.is_empty() {
        return Err("empty String".to_string());
    }
    let error = || Err(format!("For input string: \"{}\"", trimmed));
    let (sign, unsigned) = match trimmed.as_bytes()[0] {
        b'+' => (1.0f64, &trimmed[1..]),
        b'-' => (-1.0f64, &trimmed[1..]),
        _ => (1.0f64, trimmed),
    };
    if unsigned == "NaN" {
        return Ok(f64::NAN);
    }
    if unsigned == "Infinity" {
        return Ok(sign * f64::INFINITY);
    }
    // Strip the optional type suffix.
    let body = match unsigned.as_bytes().last() {
        Some(b'f') | Some(b'F') | Some(b'd') | Some(b'D') => &unsigned[..unsigned.len() - 1],
        _ => unsigned,
    };
    // Digits [. Digits] [ExponentPart] with at least one digit in the significand.
    let bytes = body.as_bytes();
    let mut index = 0;
    let mut digits = 0;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
        digits += 1;
    }
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return error();
    }
    if index < bytes.len() && (bytes[index] == b'e' || bytes[index] == b'E') {
        index += 1;
        if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
            index += 1;
        }
        let mut exponent_digits = 0;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
            exponent_digits += 1;
        }
        if exponent_digits == 0 {
            return error();
        }
    }
    if index != bytes.len() {
        return error();
    }
    // Rust's f64 parser is correctly rounded, as Java's is.
    match body.parse::<f64>() {
        Ok(parsed) => Ok(sign * parsed),
        Err(_) => error(),
    }
}

/// Java `java.lang.Double.toString(double)`.  There is always at least one digit after
/// the decimal point; the decimal form is used when `1e-3 <= |d| < 1e7` and
/// computerized scientific notation (`d.dddEn`, with no `+` on the exponent) otherwise.
///
/// The digits themselves are the shortest decimal that round-trips, which is what Rust's
/// `{:e}` produces as well; only the layout differs, so the digits are taken from `{:e}`
/// and re-laid-out here rather than re-deriving them.
pub fn java_lang_double_to_string(d: f64) -> String {
    if d.is_nan() {
        return "NaN".to_string();
    }
    if d.is_infinite() {
        return if d > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let negative = d.is_sign_negative();
    let magnitude = d.abs();
    if magnitude == 0.0 {
        return if negative { "-0.0" } else { "0.0" }.to_string();
    }
    // The fewest significant digits whose correctly rounded decimal reads back as
    // `magnitude`.  Rust's `{:.*e}` rounds to nearest with ties to even, which is the
    // tie-break Java's specification names ("the one whose least significant digit is
    // even"); taking the shortest `{:e}` instead breaks those ties the other way and
    // prints 3605683.3 where Java prints 3605683.2.
    let mut digits = String::new();
    let mut exponent = 0i32;
    for precision in 0..=16usize {
        let candidate = format!("{:.*e}", precision, magnitude);
        if candidate.parse::<f64>() == Ok(magnitude) {
            let (mantissa, candidate_exponent) = candidate.split_once('e').unwrap();
            exponent = candidate_exponent.parse().unwrap();
            digits = mantissa.chars().filter(|c| *c != '.').collect();
            break;
        }
    }
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if exponent >= -3 && exponent < 7 {
        if exponent < 0 {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let whole = (exponent + 1) as usize;
            if digits.len() > whole {
                out.push_str(&digits[..whole]);
                out.push('.');
                out.push_str(&digits[whole..]);
            } else {
                out.push_str(&digits);
                for _ in 0..(whole - digits.len()) {
                    out.push('0');
                }
                out.push_str(".0");
            }
        }
    } else {
        out.push_str(&digits[..1]);
        out.push('.');
        if digits.len() > 1 {
            out.push_str(&digits[1..]);
        } else {
            out.push('0');
        }
        out.push('E');
        out.push_str(&exponent.to_string());
    }
    out
}

/// Java `java.lang.Float.toString(float)`.  Same layout rule as
/// `java_lang_double_to_string`, over the shortest decimal that round-trips as a
/// `float`.
pub fn java_lang_float_to_string(f: f32) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let negative = f.is_sign_negative();
    let magnitude = f.abs();
    if magnitude == 0.0 {
        return if negative { "-0.0" } else { "0.0" }.to_string();
    }
    // See `java_lang_double_to_string`: the shortest correctly rounded decimal, so that
    // a decimal lying exactly between two candidates ends on an even digit as Java's
    // specification requires.
    let mut digits = String::new();
    let mut exponent = 0i32;
    for precision in 0..=8usize {
        let candidate = format!("{:.*e}", precision, magnitude);
        if candidate.parse::<f32>() == Ok(magnitude) {
            let (mantissa, candidate_exponent) = candidate.split_once('e').unwrap();
            exponent = candidate_exponent.parse().unwrap();
            digits = mantissa.chars().filter(|c| *c != '.').collect();
            break;
        }
    }
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if exponent >= -3 && exponent < 7 {
        if exponent < 0 {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        } else {
            let whole = (exponent + 1) as usize;
            if digits.len() > whole {
                out.push_str(&digits[..whole]);
                out.push('.');
                out.push_str(&digits[whole..]);
            } else {
                out.push_str(&digits);
                for _ in 0..(whole - digits.len()) {
                    out.push('0');
                }
                out.push_str(".0");
            }
        }
    } else {
        out.push_str(&digits[..1]);
        out.push('.');
        if digits.len() > 1 {
            out.push_str(&digits[1..]);
        } else {
            out.push('0');
        }
        out.push('E');
        out.push_str(&exponent.to_string());
    }
    out
}

/// Java `java.lang.Number`, as this unit uses it.  The concrete wrapper class is part of
/// the behaviour - `isNull`, `isNumberNull`, `compare` and `doMath` all switch on
/// `instanceof` - so each `instanceof` the source tests for is a variant here.
///
/// Deviation: Java's `BigInteger` and `BigDecimal` are arbitrary precision.  They appear
/// in this unit only in the `instanceof` chains of `isNumberNull`
/// (ConstEtomoNumber.java:1391-1397) and `Type.getType` (ConstEtomoNumber.java:1960),
/// which read `intValue()` / `longValue()`; the variants below carry `i64` and `f64` so
/// that those two branches exist and behave correctly for in-range values.  Nothing in
/// the translated code constructs one.
#[derive(Clone, Copy, Debug)]
pub enum Number {
    /// `java.util.concurrent.atomic.AtomicInteger`.
    AtomicInteger(i32),
    /// `java.math.BigInteger`.
    BigInteger(i64),
    /// `java.lang.Integer`.
    Integer(i32),
    /// `java.util.concurrent.atomic.AtomicLong`.
    AtomicLong(i64),
    /// `java.math.BigDecimal`.
    BigDecimal(f64),
    /// `java.lang.Long`.
    Long(i64),
    /// `java.lang.Byte`.
    Byte(i8),
    /// `java.lang.Double`.
    Double(f64),
    /// `java.lang.Float`.
    Float(f32),
    /// `java.lang.Short`.
    Short(i16),
}

impl Number {
    /// Java `Number.intValue()`.  Java's narrowing of a floating point value saturates
    /// and maps NaN to zero, which is exactly what Rust's `as` cast does.
    pub fn int_value(self) -> i32 {
        match self {
            Number::AtomicInteger(value) => value,
            Number::BigInteger(value) => value as i32,
            Number::Integer(value) => value,
            Number::AtomicLong(value) => value as i32,
            Number::BigDecimal(value) => value as i32,
            Number::Long(value) => value as i32,
            Number::Byte(value) => value as i32,
            Number::Double(value) => value as i32,
            Number::Float(value) => value as i32,
            Number::Short(value) => value as i32,
        }
    }

    /// Java `Number.longValue()`.
    pub fn long_value(self) -> i64 {
        match self {
            Number::AtomicInteger(value) => value as i64,
            Number::BigInteger(value) => value,
            Number::Integer(value) => value as i64,
            Number::AtomicLong(value) => value,
            Number::BigDecimal(value) => value as i64,
            Number::Long(value) => value,
            Number::Byte(value) => value as i64,
            Number::Double(value) => value as i64,
            Number::Float(value) => value as i64,
            Number::Short(value) => value as i64,
        }
    }

    /// Java `Number.doubleValue()`.
    pub fn double_value(self) -> f64 {
        match self {
            Number::AtomicInteger(value) => value as f64,
            Number::BigInteger(value) => value as f64,
            Number::Integer(value) => value as f64,
            Number::AtomicLong(value) => value as f64,
            Number::BigDecimal(value) => value,
            Number::Long(value) => value as f64,
            Number::Byte(value) => value as f64,
            Number::Double(value) => value,
            Number::Float(value) => value as f64,
            Number::Short(value) => value as f64,
        }
    }

    /// Java `Number.floatValue()`.
    pub fn float_value(self) -> f32 {
        match self {
            Number::AtomicInteger(value) => value as f32,
            Number::BigInteger(value) => value as f32,
            Number::Integer(value) => value as f32,
            Number::AtomicLong(value) => value as f32,
            Number::BigDecimal(value) => value as f32,
            Number::Long(value) => value as f32,
            Number::Byte(value) => value as f32,
            Number::Double(value) => value as f32,
            Number::Float(value) => value,
            Number::Short(value) => value as f32,
        }
    }

    /// Java `Number.byteValue()`.
    pub fn byte_value(self) -> i8 {
        self.int_value() as i8
    }

    /// Java `Number.shortValue()`.
    pub fn short_value(self) -> i16 {
        self.int_value() as i16
    }
}

/// Java `Number.toString()`, dispatched on the concrete wrapper class.
impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Number::AtomicInteger(value) => write!(f, "{}", value),
            Number::BigInteger(value) => write!(f, "{}", value),
            Number::Integer(value) => write!(f, "{}", value),
            Number::AtomicLong(value) => write!(f, "{}", value),
            // Deviation: Java's BigDecimal.toString() is scale-driven; the f64 stand-in
            // prints through Double.toString.  See the `Number` doc comment.
            Number::BigDecimal(value) => f.write_str(&java_lang_double_to_string(value)),
            Number::Long(value) => write!(f, "{}", value),
            Number::Byte(value) => write!(f, "{}", value),
            Number::Double(value) => f.write_str(&java_lang_double_to_string(value)),
            Number::Float(value) => f.write_str(&java_lang_float_to_string(value)),
            Number::Short(value) => write!(f, "{}", value),
        }
    }
}

/// Java nested class `Type`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
    /// Java `Type.DOUBLE`.
    Double,
    /// Java `Type.INTEGER`.
    Integer,
    /// Java `Type.LONG`.
    Long,
    /// Java `Type.BOOLEAN`.  BOOLEAN is an integer where every non-null number other
    /// then 0 is turned into a 1.
    Boolean,
}

impl Type {
    /// Java `Type.getDefault`.
    pub fn get_default() -> Type {
        Type::Integer
    }

    /// Java `Type.getType`.
    pub fn get_type(number: Option<Number>) -> Option<Type> {
        let number = match number {
            None => return None,
            Some(number) => number,
        };
        if matches!(
            number,
            Number::AtomicInteger(_)
                | Number::BigInteger(_)
                | Number::Byte(_)
                | Number::Integer(_)
                | Number::Short(_)
        ) {
            return Some(Type::Integer);
        }
        if matches!(number, Number::AtomicLong(_) | Number::Long(_)) {
            return Some(Type::Long);
        }
        Some(Type::Double)
    }
}

/// Java `Type.toString`.  The source's trailing `return "Unknown Type";` is unreachable
/// for a Rust enum, which has no fifth instance.
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Type::Double => "Double",
            Type::Integer => "Integer",
            Type::Long => "Long",
            Type::Boolean => "Boolean",
        })
    }
}

/// Java private nested class `Comparison`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Comparison {
    /// Java `Comparison.GT`.
    Gt,
    /// Java `Comparison.GE`.
    Ge,
    /// Java `Comparison.LT`.
    Lt,
    /// Java `Comparison.LE`.
    Le,
    /// Java `Comparison.EQUALS`.
    Equals,
}

/// Java private nested class `Operator`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operator {
    /// Java `Operator.ADD`.
    Add,
    /// Java `Operator.MULTIPLY`.
    Multiply,
    /// Java `Operator.DIVIDE_BY`.
    DivideBy,
    /// Java `Operator.SUBTRACT`.
    Subtract,
}

/// Java `ConstEtomoNumber`.  Abstract in the source; `EtomoNumber` holds one of these as
/// its superclass state.
#[derive(Clone, Debug)]
pub struct ConstEtomoNumber {
    /// Java field `type`: defaults to integer, can't be changed once it is set.
    pub(crate) r#type: Type,
    /// Java field `name`: defaults to `Object.toString()`, can't be changed once it is
    /// set.
    pub(crate) name: String,
    /// Java field `description`: optional, defaults to name.
    pub(crate) description: String,
    /// Java field `currentValue`: optional, defaults to `newNumber()`.
    pub(crate) current_value: Number,
    /// Java field `displayValue`: value to display when `currentValue.isNull()`.
    pub(crate) display_value: Number,
    /// Java field `ceilingValue`: optional, defaults to `newNumber()`.
    pub(crate) ceiling_value: Number,
    /// Java field `floorValue`: optional, defaults to `newNumber()`.
    pub(crate) floor_value: Number,
    /// Java field `nullIsValid`: optional.  `validValues` overrides `validFloor`.
    pub(crate) null_is_valid: bool,
    /// Java field `validValues`: set of valid values, optional.
    pub(crate) valid_values: Option<Vec<Number>>,
    /// Java field `validFloor`: numbers below `validFloor` are invalid.
    pub(crate) valid_floor: Number,
    /// Java field `defaultValue`.
    pub(crate) default_value: Number,
    /// Java field `outputStringArray`.
    pub(crate) output_string_array: Option<Vec<String>>,
    /// Java field `inputStringArray`.
    input_string_array: Option<Vec<Vec<String>>>,
    /// Java field `invalidReason`: internal validation result.
    pub(crate) invalid_reason: Option<String>,
    /// Java field `debug`.
    debug: bool,
    /// Java field `valueAltered`.
    value_altered: bool,
    /// Java field `displayAsNumber`.
    display_as_number: bool,
}

impl ConstEtomoNumber {
    /// Java `ConstEtomoNumber()`.  Construct a ConstEtomoNumber with
    /// type = INTEGER_TYPE.
    pub(crate) fn new() -> ConstEtomoNumber {
        let name = java_lang_object_to_string();
        let mut instance = ConstEtomoNumber::allocate(Type::Integer, name.clone(), name);
        instance.initialize();
        instance
    }

    /// Java `ConstEtomoNumber(String)`.  Construct a ConstEtomoNumber with
    /// type = INTEGER_TYPE; the parameter is the name of the instance.
    pub(crate) fn new_with_name(name: &str) -> ConstEtomoNumber {
        let mut instance =
            ConstEtomoNumber::allocate(Type::Integer, name.to_string(), name.to_string());
        instance.initialize();
        instance
    }

    /// Java `ConstEtomoNumber(Type)`.
    pub(crate) fn new_with_type(r#type: Option<Type>) -> ConstEtomoNumber {
        let r#type = match r#type {
            None => Type::Integer,
            Some(r#type) => r#type,
        };
        let name = java_lang_object_to_string();
        let mut instance = ConstEtomoNumber::allocate(r#type, name.clone(), name);
        instance.initialize();
        instance
    }

    /// Java `ConstEtomoNumber(Type, String)`.
    pub(crate) fn new_with_type_and_name(r#type: Type, name: &str) -> ConstEtomoNumber {
        let mut instance = ConstEtomoNumber::allocate(r#type, name.to_string(), name.to_string());
        instance.initialize();
        instance
    }

    /// Java `ConstEtomoNumber(ConstEtomoNumber)`.  Makes a deep copy of instance.
    /// Returns empty instance when instance is null.
    pub(crate) fn new_from_instance(instance: Option<&ConstEtomoNumber>) -> ConstEtomoNumber {
        let instance = match instance {
            None => {
                let name = java_lang_object_to_string();
                let mut copy = ConstEtomoNumber::allocate(Type::Integer, name.clone(), name);
                copy.initialize();
                return copy;
            }
            Some(instance) => instance,
        };
        // OK to assign Strings because they are immutable.  OK to assign Numbers because
        // they are immutable.
        let mut copy = ConstEtomoNumber::allocate(
            instance.r#type,
            instance.name.clone(),
            instance.description.clone(),
        );
        copy.current_value = instance.current_value;
        copy.display_value = instance.display_value;
        copy.ceiling_value = instance.ceiling_value;
        copy.floor_value = instance.floor_value;
        copy.valid_floor = instance.valid_floor;
        copy.null_is_valid = instance.null_is_valid;
        copy.default_value = instance.default_value;
        if let Some(valid_values) = &instance.valid_values {
            if !valid_values.is_empty() {
                let mut copied = Vec::with_capacity(valid_values.len());
                for value in valid_values.iter() {
                    copied.push(copy.new_number_from_number(Some(*value)));
                }
                copy.valid_values = Some(copied);
            }
        }
        if let Some(output_string_array) = &instance.output_string_array {
            if !output_string_array.is_empty() {
                let mut copied = Vec::with_capacity(output_string_array.len());
                for value in output_string_array.iter() {
                    copied.push(value.clone());
                }
                copy.output_string_array = Some(copied);
            }
        }
        if let Some(input_string_array) = &instance.input_string_array {
            if !input_string_array.is_empty() {
                // ConstEtomoNumber.java:154-165 builds `stringArray` inside the loop and
                // never adds it to `inputStringArray`, so the copy's list stays empty
                // (`new ArrayList<String[]>()`).  The dropped copy is preserved here.
                copy.input_string_array = Some(Vec::new());
                let len = input_string_array.len();
                for i in 0..len {
                    let instance_string_array = &input_string_array[i];
                    if !instance_string_array.is_empty() {
                        let mut string_array = Vec::with_capacity(instance_string_array.len());
                        for j in 0..instance_string_array.len() {
                            string_array.push(instance_string_array[j].clone());
                        }
                        let _ = string_array;
                    }
                }
            }
        }
        if let Some(invalid_reason) = &instance.invalid_reason {
            copy.invalid_reason = Some(invalid_reason.clone());
        }
        copy
    }

    /// The Java object allocation the five constructors share: every field at its
    /// declared initial value, with `type`, `name` and `description` filled in.  The
    /// `Number` fields are the ones `initialize()` overwrites, and are set to the
    /// declaration-time Java `null` stand-in `Integer(INTEGER_NULL_VALUE)` here.
    fn allocate(r#type: Type, name: String, description: String) -> ConstEtomoNumber {
        ConstEtomoNumber {
            r#type,
            name,
            description,
            current_value: Number::Integer(INTEGER_NULL_VALUE),
            display_value: Number::Integer(INTEGER_NULL_VALUE),
            ceiling_value: Number::Integer(INTEGER_NULL_VALUE),
            floor_value: Number::Integer(INTEGER_NULL_VALUE),
            null_is_valid: true,
            valid_values: None,
            valid_floor: Number::Integer(INTEGER_NULL_VALUE),
            default_value: Number::Integer(INTEGER_NULL_VALUE),
            output_string_array: None,
            input_string_array: None,
            invalid_reason: None,
            debug: false,
            value_altered: false,
            display_as_number: false,
        }
    }

    /// Java `isNull(int)`.
    pub fn is_null_int(&self, value: i32) -> bool {
        self.is_null_number(Some(Number::Integer(value)))
    }

    /// Java `getDescription`.
    pub fn get_description(&self) -> &str {
        &self.description
    }

    /// Java `getName`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Java `isDebug`.
    pub(crate) fn is_debug(&self) -> bool {
        self.debug
    }

    /// Java `getDisplayInteger`.
    pub fn get_display_integer(&self) -> i32 {
        self.display_value.int_value()
    }

    /// Java `setInvalidReason`.  If validValues has been set, look for currentValue in
    /// validValues.  Set invalidReasion if currentValue is not found.  Null is ignored.
    /// Set invalidReason if currentValue is null and nullIsValid is false.
    pub(crate) fn set_invalid_reason(&mut self) {
        // Pass when there are no validation settings
        if self.null_is_valid
            && self.valid_values.is_none()
            && self.is_null_number(Some(self.valid_floor))
        {
            return;
        }
        // Catch illegal null values
        if self.is_null_number(Some(self.current_value)) {
            if self.null_is_valid {
                return;
            }
            self.add_invalid_reason(Some("This field cannot be empty."));
        }
        // Validate against validValues, overrides validFloor
        else if self.valid_values.is_some() {
            let valid_values = self.valid_values.clone().unwrap();
            for i in 0..valid_values.len() {
                if self.equals_numbers(Some(self.current_value), Some(valid_values[i])) {
                    return;
                }
            }
            let message = format!(
                "{} is not a valid value.",
                self.to_string_number(Some(self.current_value))
            );
            self.add_invalid_reason(Some(&message));
            let message = format!(
                "Valid values are {}.",
                self.to_string_number_vector(Some(&valid_values))
            );
            self.add_invalid_reason(Some(&message));
            return;
        }
        // If validValues is not set, validate against validFloor
        else if !self.is_null_number(Some(self.valid_floor)) {
            if self.ge_numbers(Some(self.current_value), Some(self.valid_floor)) {
                return;
            }
            let message = format!(
                "{} is not a valid value.",
                self.to_string_number(Some(self.current_value))
            );
            self.add_invalid_reason(Some(&message));
            let message = format!(
                "Valid values are greater or equal to {}.",
                self.to_string_number(Some(self.valid_floor))
            );
            self.add_invalid_reason(Some(&message));
        }
    }

    /// Java `applyCeilingValue`.  Returns ceilingValue if value > ceilingValue.
    /// Otherwise returns value.  Ignores null.
    pub(crate) fn apply_ceiling_value(&mut self, value: Option<Number>) -> Option<Number> {
        if value.is_some()
            && !self.is_null_number(Some(self.ceiling_value))
            && !self.is_null_number(value)
            && self.gt_numbers(value, Some(self.ceiling_value))
        {
            self.value_altered = true;
            return Some(self.ceiling_value);
        }
        value
    }

    /// Java `applyFloorValue`.  Returns floorValue if value < floorValue.  Otherwise
    /// returns values.  Ignores null.
    pub(crate) fn apply_floor_value(&mut self, value: Option<Number>) -> Option<Number> {
        if value.is_some()
            && !self.is_null_number(Some(self.floor_value))
            && !self.is_null_number(value)
            && self.lt_numbers(value, Some(self.floor_value))
        {
            self.value_altered = true;
            return Some(self.floor_value);
        }
        value
    }

    /// Java `isValid`.  If invalidReason is set, return true.
    pub fn is_valid(&self) -> bool {
        self.invalid_reason.is_none()
    }

    /// Java `isValueAltered`.  Returns true if the value that was set by another class
    /// has been altered internally by this class (or EtomoNumber).  This functionality
    /// currently ignores alteration because of an unparsable value.  When the ceiling
    /// and floor functionality cause the externally set value to change, this function
    /// will return true.
    pub fn is_value_altered(&self) -> bool {
        self.value_altered
    }

    /// Java `validate`.  If invalidReason is set, display an error message and return
    /// true.  Returns the error message if invalidReason is set, null if valid.
    pub fn validate(&self, description: Option<&str>) -> Option<String> {
        if let Some(invalid_reason) = &self.invalid_reason {
            let description = match description {
                None => return Some(format!("{}: {}", self.description, invalid_reason)),
                Some(description) => description,
            };
            let description = java_lang_string_trim(description);
            if description.ends_with(':') {
                return Some(format!("{}  {}", description, invalid_reason));
            }
            return Some(format!("{}:  {}", description, invalid_reason));
        }
        None
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `getInvalidReason`.
    pub fn get_invalid_reason(&self) -> String {
        match &self.invalid_reason {
            None => String::new(),
            Some(invalid_reason) => invalid_reason.clone(),
        }
    }

    /// Java `classInfoString`.  `getClass().getName()` is the runtime class; see
    /// `java_lang_object_to_string`.
    pub fn class_info_string(&self) -> String {
        format!("etomo.type.EtomoNumber[{}]", self.param_string())
    }

    /// Java `paramString`.  Java string concatenation renders a null reference as
    /// "null", which is what the `None` arms below produce.
    pub(crate) fn param_string(&self) -> String {
        let mut buffer = format!(
            ",\ntype={},\nname={},\ndescription={},\ninvalidReason={},\ncurrentValue={},\ndisplayValue={},\nceilingValue={},\nfloorValue={},\nnullIsValid={},\nvalidValues=",
            self.r#type,
            self.name,
            self.description,
            match &self.invalid_reason {
                None => "null".to_string(),
                Some(invalid_reason) => invalid_reason.clone(),
            },
            self.current_value,
            self.display_value,
            self.ceiling_value,
            self.floor_value,
            self.null_is_valid
        );
        match &self.valid_values {
            None => buffer.push_str("null"),
            Some(valid_values) => {
                // java.util.Vector.toString(): "[a, b, c]".
                let mut rendered = String::from("[");
                for (i, value) in valid_values.iter().enumerate() {
                    if i > 0 {
                        rendered.push_str(", ");
                    }
                    rendered.push_str(&value.to_string());
                }
                rendered.push(']');
                buffer.push_str(&rendered);
            }
        }
        buffer
    }

    /// Java `setCeiling`.  Sets ceiling value.  Ceiling value is applied when the value
    /// is set; if the value being set is more then the ceiling value, the value is set
    /// equals to the ceiling value.
    pub fn set_ceiling(&mut self, ceiling_value: i32) -> &mut ConstEtomoNumber {
        self.ceiling_value = self.new_number_from_int(ceiling_value);
        self.validate_floor_and_ceiling();
        let current_value = self.apply_ceiling_value(Some(self.current_value));
        self.current_value = self.new_number_from_number(current_value);
        self
    }

    /// Java `setFloor`.  Sets floor value.  Floor value is applied when the value is
    /// set; if the value being set is less then the floor value, the value is set equals
    /// to the floor value.
    pub fn set_floor(&mut self, floor_value: i32) -> &mut ConstEtomoNumber {
        self.floor_value = self.new_number_from_int(floor_value);
        self.validate_floor_and_ceiling();
        let current_value = self.apply_floor_value(Some(self.current_value));
        self.current_value = self.new_number_from_number(current_value);
        self
    }

    // Representation limit: Java `internalTest()` (ConstEtomoNumber.java:364-556) is
    // untranslated.  Its guard, `Utilities.isSelfTest()`, is now available
    // (`etomo/util/utilities.rs`), but its body is not expressible over this module's
    // representation: it is seven `x == null` checks and eighteen
    // `x instanceof Double/Integer/Long` checks over `java.lang.Number` *references*,
    // followed by ten `x == y` *reference-identity* comparisons that assert `newNumber()`
    // boxed a fresh object for each field.  The fields here are `Number` enum values, not
    // references: none of them can be null, the variant is the type, and two fields
    // holding equal values are not "the same object".  Every one of those checks is
    // therefore either vacuous or unaskable, so writing the body would be writing a
    // different test.  `validateFloorAndCeiling()`, the one behavioural call in it, is
    // translated below.

    /// Java `setDisplayValue(int)`.  Set the value will be returned if the user does not
    /// set a value or there is no value to load.
    pub fn set_display_value_int(&mut self, display_value: i32) -> &mut ConstEtomoNumber {
        self.display_value = self.new_number_from_int(display_value);
        self
    }

    /// Java `setDisplayValue(boolean)`.
    pub fn set_display_value_boolean(&mut self, display_value: bool) -> &mut ConstEtomoNumber {
        self.display_value = self.new_number_from_boolean(display_value);
        self
    }

    /// Java `setDisplayValue(Number)`.
    pub(crate) fn set_display_value_number(
        &mut self,
        display_value: Option<Number>,
    ) -> &mut ConstEtomoNumber {
        self.display_value = self.new_number_from_number(display_value);
        self
    }

    /// Java `setDisplayValue(double)`.  Set the value will be used if the user does not
    /// set a value or there is no value to load.  Also used in `reset()`.
    pub fn set_display_value_double(&mut self, display_value: f64) {
        self.display_value = self.new_number_from_double(display_value);
    }

    /// Java `setDisplayValue(long)`.
    pub fn set_display_value_long(&mut self, display_value: i64) {
        self.display_value = self.new_number_from_long(display_value);
    }

    /// Java `setDescription`.
    pub fn set_description(&mut self, description: Option<&str>) {
        match description {
            Some(description) => self.description = description.to_string(),
            None => self.description = self.name.clone(),
        }
    }

    /// Java `setNullIsValid`.
    pub fn set_null_is_valid(&mut self, null_is_valid: bool) -> &mut ConstEtomoNumber {
        self.reset_state();
        self.null_is_valid = null_is_valid;
        self.set_invalid_reason();
        self
    }

    /// Java `setValidValues`.  Set a list of non-null valid values.  A null param or an
    /// empty list causes this.validValues to be set to null.
    pub fn set_valid_values(&mut self, valid_values: Option<&[i32]>) -> &mut ConstEtomoNumber {
        self.reset_state();
        match valid_values {
            None => self.valid_values = None,
            Some(valid_values) if valid_values.is_empty() => self.valid_values = None,
            Some(valid_values) => {
                let mut list = Vec::with_capacity(valid_values.len());
                for i in 0..valid_values.len() {
                    let valid_value = valid_values[i];
                    if !self.is_null_int(valid_value) {
                        list.push(self.new_number_from_int(valid_values[i]));
                    }
                }
                self.valid_values = Some(list);
            }
        }
        self.set_invalid_reason();
        self
    }

    /// Java `setValidFloor`.  Sets valid floor.  Valid floor does not change a value
    /// being set.  It causes validation to fail if the current value is less then the
    /// valid floor.
    pub fn set_valid_floor(&mut self, valid_floor: i32) -> &mut ConstEtomoNumber {
        self.reset_state();
        self.valid_floor = self.new_number_from_int(valid_floor);
        self.set_invalid_reason();
        self
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        if self.is_null_number(Some(self.current_value)) {
            self.remove(props);
            return;
        }
        props.insert(
            self.name.clone(),
            self.to_string_number(Some(self.current_value)),
        );
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        if self.is_null_number(Some(self.current_value)) {
            self.remove_with_prepend(props, prepend);
            return;
        }
        match prepend {
            None => self.store(props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.store(props),
            Some(prepend) => {
                props.insert(
                    format!("{}.{}", prepend, self.name),
                    self.to_string_number(Some(self.current_value)),
                );
            }
        }
    }

    /// Java `store(EtomoNumber, String, Properties, String)`, the static overload.
    pub fn store_etomo_number(
        etomo_number: Option<&super::etomo_number::EtomoNumber>,
        name: &str,
        props: &mut BTreeMap<String, String>,
        prepend: Option<&str>,
    ) {
        let etomo_number = match etomo_number {
            None => {
                // Java concatenates a null prepend as the text "null".
                props.remove(&format!(
                    "{}.{}",
                    match prepend {
                        None => "null",
                        Some(prepend) => prepend,
                    },
                    name
                ));
                return;
            }
            Some(etomo_number) => etomo_number,
        };
        etomo_number.store_with_prepend(props, prepend);
    }

    /// Java `remove(Properties)`.
    pub fn remove(&self, props: &mut BTreeMap<String, String>) {
        props.remove(&self.name);
    }

    /// Java `remove(String, Properties)`, the static overload.
    pub fn remove_name(name: &str, props: &mut BTreeMap<String, String>) {
        props.remove(name);
    }

    /// Java `remove(Properties, String)`.
    pub fn remove_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: Option<&str>) {
        match prepend {
            None => self.remove(props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => self.remove(props),
            Some(prepend) => {
                props.remove(&format!("{}.{}", prepend, self.name));
            }
        }
    }

    /// Java `remove(String, Properties, String)`, the static overload.
    pub fn remove_name_with_prepend(
        name: &str,
        props: &mut BTreeMap<String, String>,
        prepend: Option<&str>,
    ) {
        match prepend {
            None => ConstEtomoNumber::remove_name(name, props),
            Some(prepend) if java_lang_string_matches_whitespace(prepend) => {
                ConstEtomoNumber::remove_name(name, props)
            }
            Some(prepend) => {
                props.remove(&format!("{}.{}", prepend, name));
            }
        }
    }

    /// Java `setOutputStrings`.  Sets an instance with a string array so numbers can be
    /// translated to and from strings.  No effect if the type isn't integer.  Use with
    /// `setValidValues` and `setDisplayValue` to create a boolean that can be null.
    pub fn set_output_strings(&mut self, input: Option<&[String]>) {
        if self.r#type != Type::Integer && self.r#type != Type::Boolean {
            return;
        }
        if input.is_none() || input.unwrap().is_empty() {
            self.output_string_array = None;
        }
        // ConstEtomoNumber.java:719 falls through to `new String[input.length]` even when
        // `input == null`, which throws NullPointerException.  The panic below is that
        // throw.
        let input = input.expect("java.lang.NullPointerException");
        let mut output_string_array = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            output_string_array.push(input[i].clone());
        }
        self.output_string_array = Some(output_string_array);
    }

    /// Java `addInputStrings`.  Sets an instance with a string array so numbers can be
    /// translated from strings.  No effect if the type isn't integer.  Use with
    /// `setValidValues` and `setDisplayValue` to create a boolean that can be null.
    pub fn add_input_strings(&mut self, input: Option<&[String]>) {
        if (self.r#type != Type::Integer && self.r#type != Type::Boolean)
            || input.is_none()
            || input.unwrap().is_empty()
        {
            return;
        }
        let input = input.unwrap();
        if self.input_string_array.is_none() {
            self.input_string_array = Some(Vec::new());
        }
        let mut array = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            array.push(input[i].clone());
        }
        self.input_string_array.as_mut().unwrap().push(array);
    }

    /// Java `toDefaultedString`.  If default is set and `isNull()` is true, defaultValue
    /// will be returned, even if displayValue is set.  If defaultValue is not set, or
    /// `isNull()` is false, then it works the same as `setValue()`.
    pub fn to_defaulted_string(&self) -> String {
        self.to_string_number(Some(self.get_defaulted_value()))
    }

    /// Java `getDefaultValue`.
    pub fn get_default_value(&self) -> Number {
        self.default_value
    }

    /// Java `getInt`.
    pub fn get_int(&self) -> i32 {
        self.get_value().int_value()
    }

    /// Java `is`.
    pub fn is(&self) -> bool {
        if self.is_null() || self.equals_int(0) {
            return false;
        }
        true
    }

    /// Java `isPositive`.
    pub fn is_positive(&self) -> bool {
        self.gt_numbers(Some(self.get_value()), Some(self.new_number_from_int(0)))
    }

    /// Java `isNegative`.
    pub fn is_negative(&self) -> bool {
        self.lt_numbers(Some(self.get_value()), Some(self.new_number_from_int(0)))
    }

    /// Java `getLong`.
    pub fn get_long(&self) -> i64 {
        self.get_value().long_value()
    }

    /// Java `getDouble`.
    pub fn get_double(&self) -> f64 {
        self.get_value().double_value()
    }

    /// Java `getDefaultedDouble`.  If default is set and `isNull()` is true,
    /// defaultValue will be returned, even if displayValue is set.
    pub fn get_defaulted_double(&self) -> f64 {
        self.get_defaulted_value().double_value()
    }

    /// Java `getDefaultedInt`.
    pub fn get_defaulted_int(&self) -> i32 {
        self.get_defaulted_value().int_value()
    }

    /// Java `setDefault(int)`.
    pub fn set_default_int(&mut self, input: i32) -> &mut ConstEtomoNumber {
        self.default_value = self.new_number_from_int(input);
        self
    }

    /// Java `setDefault(ConstEtomoNumber)`.
    pub fn set_default_const_etomo_number(&mut self, input: Option<&ConstEtomoNumber>) {
        match input {
            None => self.default_value = self.new_number(),
            Some(input) => {
                let number = input.get_number();
                self.default_value = self.new_number_from_number(Some(number));
            }
        }
    }

    /// Java `setDefault(boolean)`.
    pub fn set_default_boolean(&mut self, default_value: bool) -> &mut ConstEtomoNumber {
        self.default_value = self.new_number_from_boolean(default_value);
        self
    }

    /// Java `isDefault()`.  Returns true if currentValue is not null and is equal to
    /// defaultValue.  This function is not effected by displayValue.
    pub fn is_default(&self) -> bool {
        self.is_default_number(Some(self.current_value))
    }

    /// Java `isDefaultSet`.
    pub fn is_default_set(&self) -> bool {
        !self.is_null_number(Some(self.default_value))
    }

    /// Java `isDefault(Number)`.  Returns true if defaultValue is not null and value is
    /// equal to defaultValue.
    pub(crate) fn is_default_number(&self, value: Option<Number>) -> bool {
        if self.is_null_number(Some(self.default_value)) {
            return false;
        }
        self.equals_numbers(value, Some(self.default_value))
    }

    /// Java `useDefaultAsDisplayValue`.
    pub fn use_default_as_display_value(&mut self) -> &mut ConstEtomoNumber {
        let default_value = self.default_value;
        self.set_display_value_number(Some(default_value))
    }

    /// Java `getNumber`.
    pub fn get_number(&self) -> Number {
        self.new_number_from_number(Some(self.get_value()))
    }

    /// Java `equalsNameIgnoreCase`.
    pub fn equals_name_ignore_case(&self, input: &str) -> bool {
        // java.lang.String.compareToIgnoreCase(input) == 0
        self.name.to_uppercase().to_lowercase() == input.to_uppercase().to_lowercase()
    }

    /// Java `equals(ConstEtomoNumber)`.  Returns true if `getValue()` equals
    /// `that.getValue()`.
    pub fn equals_const_etomo_number(&self, that: Option<&ConstEtomoNumber>) -> bool {
        let that = match that {
            None => return false,
            Some(that) => that,
        };
        self.equals_numbers(Some(self.get_value()), Some(that.get_value()))
    }

    /// Java `gt(int)`.
    pub fn gt_int(&self, value: i32) -> bool {
        self.gt_numbers(Some(self.get_value()), Some(Number::Integer(value)))
    }

    /// Java `gt(String)`.
    pub fn gt_string(&self, value: Option<&str>) -> bool {
        let mut invalid_buffer = String::new();
        let number = self.new_number_from_string(value, &mut invalid_buffer);
        self.gt_numbers(Some(self.get_value()), Some(number))
    }

    /// Java `gt(Number)`.
    pub fn gt_number(&self, value: Option<Number>) -> bool {
        self.gt_numbers(Some(self.get_value()), value)
    }

    /// Java `ge(String)`.
    pub fn ge_string(&self, value: Option<&str>) -> bool {
        let mut invalid_buffer = String::new();
        let number = self.new_number_from_string(value, &mut invalid_buffer);
        self.ge_numbers(Some(self.get_value()), Some(number))
    }

    /// Java `lt(int)`.
    pub fn lt_int(&self, value: i32) -> bool {
        self.lt_numbers(Some(self.get_value()), Some(Number::Integer(value)))
    }

    /// Java `lt(String)`.
    pub fn lt_string(&self, value: Option<&str>) -> bool {
        let mut invalid_buffer = String::new();
        let number = self.new_number_from_string(value, &mut invalid_buffer);
        self.lt_numbers(Some(self.get_value()), Some(number))
    }

    /// Java `lt(Number)`.
    pub fn lt_number(&self, value: Option<Number>) -> bool {
        self.lt_numbers(Some(self.get_value()), value)
    }

    /// Java `le(long)`.
    pub fn le_long(&self, value: i64) -> bool {
        self.le_numbers(Some(self.get_value()), Some(Number::Long(value)))
    }

    /// Java `ge(long)`.
    pub fn ge_long(&self, value: i64) -> bool {
        self.ge_numbers(Some(self.get_value()), Some(Number::Long(value)))
    }

    /// Java `ge(double)`.
    pub fn ge_double(&self, value: f64) -> bool {
        self.ge_numbers(Some(self.get_value()), Some(Number::Double(value)))
    }

    /// Java `le(int)`.  Note that unlike `gt(int)` and `lt(int)` this one converts
    /// through `newNumber(int)` rather than autoboxing to an `Integer`.
    pub fn le_int(&self, value: i32) -> bool {
        let v = self.new_number_from_int(value);
        self.lt_numbers(Some(self.get_value()), Some(v))
            || self.equals_numbers(Some(self.get_value()), Some(v))
    }

    /// Java `le(ConstEtomoNumber)`.
    pub fn le_const_etomo_number(&self, value: &ConstEtomoNumber) -> bool {
        self.le_numbers(Some(self.get_value()), Some(value.get_number()))
    }

    /// Java `ge(ConstEtomoNumber)`.
    pub fn ge_const_etomo_number(&self, value: &ConstEtomoNumber) -> bool {
        self.ge_numbers(Some(self.get_value()), Some(value.get_number()))
    }

    /// Java `gt(ConstEtomoNumber)`.
    pub fn gt_const_etomo_number(&self, etomo_number: Option<&ConstEtomoNumber>) -> bool {
        let etomo_number = match etomo_number {
            None => return false,
            Some(etomo_number) => etomo_number,
        };
        self.gt_numbers(Some(self.get_value()), Some(etomo_number.get_value()))
    }

    /// Java `lt(ConstEtomoNumber)`.
    pub fn lt_const_etomo_number(&self, etomo_number: Option<&ConstEtomoNumber>) -> bool {
        let etomo_number = match etomo_number {
            None => return false,
            Some(etomo_number) => etomo_number,
        };
        self.lt_numbers(Some(self.get_value()), Some(etomo_number.get_value()))
    }

    /// Java `equals(int)`.
    pub fn equals_int(&self, value: i32) -> bool {
        self.equals_numbers(Some(self.get_value()), Some(Number::Integer(value)))
    }

    /// Java `equals(long)`.
    pub fn equals_long(&self, value: i64) -> bool {
        self.equals_numbers(Some(self.get_value()), Some(Number::Long(value)))
    }

    /// Java `equals(double)`.
    pub fn equals_double(&self, value: f64) -> bool {
        self.equals_numbers(Some(self.get_value()), Some(Number::Double(value)))
    }

    /// Java `isNull()`.  Returns true if currentValue is null.  `IsNull()` does not use
    /// `getValue()` and ignores displayValue, so it shows whether the instance has been
    /// explicitely set.
    pub fn is_null(&self) -> bool {
        self.is_null_number(Some(self.current_value))
    }

    /// Java `isSet`.  Returns true if currentValue is not null.  Ignores defaultValue
    /// and display value.
    pub fn is_set(&self) -> bool {
        !self.is_null_number(Some(self.current_value))
    }

    /// Java `equals(Number)`, the single-argument overload.
    pub fn equals_number(&self, value: Option<Number>) -> bool {
        self.equals_numbers(Some(self.get_value()), value)
    }

    /// Java `equals(String)`.
    pub fn equals_string(&self, value: Option<&str>) -> bool {
        let mut invalid_buffer = String::new();
        let number = self.new_number_from_string(value, &mut invalid_buffer);
        self.equals_numbers(Some(self.get_value()), Some(number))
    }

    /// Java `isNamed`.
    pub fn is_named(&self, name: &str) -> bool {
        self.name == name
    }

    /// Java `isInt`.
    pub fn is_int(&self) -> bool {
        self.r#type == Type::Integer
    }

    /// Java `initialize`.
    fn initialize(&mut self) {
        self.ceiling_value = self.new_number();
        self.floor_value = self.new_number();
        self.display_value = self.new_number();
        self.current_value = self.new_number();
        self.valid_floor = self.new_number();
        self.default_value = self.new_number();
        if self.r#type == Type::Boolean {
            self.set_output_strings(Some(&["false".to_string(), "true".to_string()][..]));
            self.add_input_strings(Some(&["no".to_string(), "yes".to_string()][..]));
            self.add_input_strings(Some(&["off".to_string(), "on".to_string()][..]));
            self.add_input_strings(Some(&["f".to_string(), "t".to_string()][..]));
            self.add_input_strings(Some(&["n".to_string(), "y".to_string()][..]));
        }
    }

    /// Java `setDisplayAsNumber`.  Values will always be displayed as numbers after this
    /// is run.  Removes the output string array.  The output string array is added to
    /// the input string array so its strings can be used for input.  Boolean numbers are
    /// 0 and 1.
    pub fn set_display_as_number(&mut self) {
        self.display_as_number = true;
        if self.output_string_array.is_some() {
            if self.input_string_array.is_none() {
                self.input_string_array = Some(Vec::new());
            }
            let output_string_array = self.output_string_array.take().unwrap();
            self.input_string_array
                .as_mut()
                .unwrap()
                .push(output_string_array);
            self.output_string_array = None;
        }
    }

    /// Java `isDisplayAsNumber`.
    pub(crate) fn is_display_as_number(&self) -> bool {
        self.display_as_number
    }

    /// Java `printValues`.
    pub(crate) fn print_values(&self) {
        eprintln!(
            "[currentValue:{},displayValue;{},defaultValue:{}]",
            self.current_value, self.display_value, self.default_value
        );
    }

    /// Java `getValue`.  If the currentValue is not null, returns it.  If the
    /// currentValue is null, returns the displayValue.  So, if the displayValue is null
    /// also, it returns null.
    pub(crate) fn get_value(&self) -> Number {
        if !self.is_null_number(Some(self.current_value)) {
            return self.current_value;
        }
        self.display_value
    }

    /// Java `getSetValue`.
    pub(crate) fn get_set_value(&self) -> Number {
        self.current_value
    }

    /// Java `isDefaultedNull`.
    pub fn is_defaulted_null(&self) -> bool {
        self.is_null_number(Some(self.get_defaulted_value()))
    }

    /// Java `getDefaultedValue`.  If default is set and `isNull()` is true, defaultValue
    /// will be returned, even if displayValue is set.
    pub(crate) fn get_defaulted_value(&self) -> Number {
        if self.is_default_set() && self.is_null() {
            return self.default_value;
        }
        self.get_value()
    }

    /// Java `getDefaultedNumber`.
    pub fn get_defaulted_number(&self) -> Number {
        self.new_number_from_number(Some(self.get_defaulted_value()))
    }

    /// Java `getNegatedDefaultedNumber`.
    pub fn get_negated_defaulted_number(&self) -> Number {
        self.negate(Some(self.get_defaulted_value()))
    }

    /// Java `getDefaultedBoolean`.
    pub fn get_defaulted_boolean(&self) -> bool {
        let value = self.get_defaulted_value();
        if self.is_null_number(Some(value))
            || self.equals_numbers(Some(value), Some(self.new_number_from_int(0)))
        {
            return false;
        }
        true
    }

    /// Java `toString(Number)`.
    pub(crate) fn to_string_number(&self, value: Option<Number>) -> String {
        if self.is_null_number(value) {
            return String::new();
        }
        let value = value.unwrap();
        if (self.r#type == Type::Integer || self.r#type == Type::Boolean)
            && self.output_string_array.is_some()
        {
            let output_string_array = self.output_string_array.as_ref().unwrap();
            let index = value.int_value();
            if index >= 0 && (index as usize) < output_string_array.len() {
                return output_string_array[index as usize].clone();
            }
        }
        value.to_string()
    }

    /// Java `toString(Vector)`.
    fn to_string_number_vector(&self, number_vector: Option<&Vec<Number>>) -> String {
        let number_vector = match number_vector {
            None => return String::new(),
            Some(number_vector) if number_vector.is_empty() => return String::new(),
            Some(number_vector) => number_vector,
        };
        let mut buffer = self.to_string_number(Some(number_vector[0]));
        for i in 1..number_vector.len() {
            buffer.push_str(&format!(
                ",{}",
                self.to_string_number(Some(number_vector[i]))
            ));
        }
        buffer
    }

    /// Java `resetState`.
    pub(crate) fn reset_state(&mut self) {
        self.invalid_reason = None;
        self.value_altered = false;
    }

    /// Java `addInvalidReason`.
    pub(crate) fn add_invalid_reason(&mut self, message: Option<&str>) {
        let message = match message {
            None => return,
            Some(message) => message,
        };
        match &mut self.invalid_reason {
            None => self.invalid_reason = Some(message.to_string()),
            Some(invalid_reason) => {
                invalid_reason.push('\n');
                invalid_reason.push_str(message);
            }
        }
    }

    /// Java `newNumber()`.
    pub(crate) fn new_number(&self) -> Number {
        if self.r#type == Type::Integer || self.r#type == Type::Boolean {
            return Number::Integer(INTEGER_NULL_VALUE);
        }
        if self.r#type == Type::Double {
            return Number::Double(DOUBLE_NULL_VALUE);
        }
        if self.r#type == Type::Long {
            return Number::Long(LONG_NULL_VALUE);
        }
        panic!("java.lang.IllegalStateException: type={}", self.r#type);
    }

    /// Java `newNumber(Number)`.  Creates a new number based on the type member
    /// variable.
    pub(crate) fn new_number_from_number(&self, value: Option<Number>) -> Number {
        if value.is_none() || self.is_null_number(value) {
            return self.new_number();
        }
        let value = value.unwrap();
        if self.r#type == Type::Boolean {
            if self.equals_numbers(Some(value), Some(Number::Integer(BOOLEAN_FALSE))) {
                return Number::Integer(BOOLEAN_FALSE);
            }
            return Number::Integer(BOOLEAN_TRUE);
        }
        if self.r#type == Type::Integer {
            return Number::Integer(value.int_value());
        }
        if self.r#type == Type::Double {
            return Number::Double(value.double_value());
        }
        if self.r#type == Type::Long {
            return Number::Long(value.long_value());
        }
        panic!("java.lang.IllegalStateException: type={}", self.r#type);
    }

    /// Java `newNumber(String, StringBuffer)`.  Override this class to display numbers
    /// as descriptive character strings.
    pub(crate) fn new_number_from_string(
        &self,
        value: Option<&str>,
        invalid_buffer: &mut String,
    ) -> Number {
        let value = match value {
            None => return self.new_number(),
            Some(value) => value,
        };
        let value = java_lang_string_trim(value);
        if value.is_empty() {
            return self.new_number();
        }
        // The Java `try` block.
        let error = 'parse: {
            if self.r#type == Type::Boolean {
                let integer = match java_lang_integer_parse_int(value) {
                    Ok(integer) => integer,
                    Err(message) => break 'parse message,
                };
                if integer == BOOLEAN_FALSE {
                    return Number::Integer(integer);
                }
                return Number::Integer(BOOLEAN_TRUE);
            }
            if self.r#type == Type::Integer {
                match java_lang_integer_parse_int(value) {
                    Ok(integer) => return Number::Integer(integer),
                    Err(message) => break 'parse message,
                }
            }
            if self.r#type == Type::Double {
                match java_lang_double_value_of(value) {
                    Ok(double) => return Number::Double(double),
                    Err(message) => break 'parse message,
                }
            }
            if self.r#type == Type::Long {
                match java_lang_long_parse_long(value) {
                    Ok(long) => return Number::Long(long),
                    Err(message) => break 'parse message,
                }
            }
            panic!("java.lang.IllegalStateException: type={}", self.r#type);
        };
        // The Java `catch (NumberFormatException e)` block.
        // Convert string to a specific integer value
        if self.r#type == Type::Integer || self.r#type == Type::Boolean {
            if let Some(output_string_array) = &self.output_string_array {
                for i in 0..output_string_array.len() {
                    if value.to_uppercase().to_lowercase()
                        == output_string_array[i].to_uppercase().to_lowercase()
                    {
                        return self.new_number_from_int(i as i32);
                    }
                }
            }
            if let Some(input_string_array) = &self.input_string_array {
                let len = input_string_array.len();
                for i in 0..len {
                    let string_array = &input_string_array[i];
                    for j in 0..string_array.len() {
                        if value.to_uppercase().to_lowercase()
                            == string_array[j].to_uppercase().to_lowercase()
                        {
                            return self.new_number_from_int(j as i32);
                        }
                    }
                }
            }
        }
        invalid_buffer.push_str(&format!(
            "{} is not a valid {}.  {}",
            value, self.r#type, error
        ));
        // TODO should the invalid string be stored somewhere?
        self.new_number()
    }

    /// Java `newNumber(int)`.
    pub(crate) fn new_number_from_int(&self, value: i32) -> Number {
        if self.is_null_int(value) {
            return self.new_number();
        }
        if self.r#type == Type::Boolean {
            if value == BOOLEAN_FALSE {
                return Number::Integer(BOOLEAN_FALSE);
            }
            return Number::Integer(BOOLEAN_TRUE);
        }
        if self.r#type == Type::Integer {
            return Number::Integer(value);
        }
        if self.r#type == Type::Double {
            return Number::Double(value as f64);
        }
        if self.r#type == Type::Long {
            return Number::Long(value as i64);
        }
        panic!("java.lang.IllegalStateException: type={}", self.r#type);
    }

    /// Java `newNumber(boolean)`.
    pub(crate) fn new_number_from_boolean(&self, value: bool) -> Number {
        if value {
            return self.new_number_from_int(BOOLEAN_TRUE);
        }
        self.new_number_from_int(BOOLEAN_FALSE)
    }

    /// Java `newNumber(double)`.  The `isNull(value)` call autoboxes to a `Double`, so
    /// an instance whose type is not DOUBLE does not see NaN as null and falls through
    /// to the narrowing conversion below.
    pub(crate) fn new_number_from_double(&self, value: f64) -> Number {
        if self.is_null_number(Some(Number::Double(value))) {
            return self.new_number();
        }
        if self.r#type == Type::Boolean {
            if value == BOOLEAN_FALSE as f64 {
                return Number::Integer(BOOLEAN_FALSE);
            }
            return Number::Integer(BOOLEAN_TRUE);
        }
        if self.r#type == Type::Integer {
            return Number::Integer(value as i32);
        }
        if self.r#type == Type::Double {
            return Number::Double(value);
        }
        if self.r#type == Type::Long {
            return Number::Long(value as i64);
        }
        panic!("java.lang.IllegalStateException: type={}", self.r#type);
    }

    /// Java `newNumber(long)`.  `isNull(value)` autoboxes to a `Long`.
    pub(crate) fn new_number_from_long(&self, value: i64) -> Number {
        if self.is_null_number(Some(Number::Long(value))) {
            return self.new_number();
        }
        if self.r#type == Type::Boolean {
            if value == BOOLEAN_FALSE as i64 {
                return Number::Integer(BOOLEAN_FALSE);
            }
            return Number::Integer(BOOLEAN_TRUE);
        }
        if self.r#type == Type::Integer {
            return Number::Integer(value as i32);
        }
        if self.r#type == Type::Double {
            return Number::Double(value as f64);
        }
        if self.r#type == Type::Long {
            return Number::Long(value);
        }
        panic!("java.lang.IllegalStateException: type={}", self.r#type);
    }

    /// Java `newNumber(float)`.  `isNull(value)` autoboxes to a `Float`, which no
    /// instance type ever recognises as its own null, so this overload never returns
    /// `newNumber()`.
    pub(crate) fn new_number_from_float(&self, value: f32) -> Number {
        if self.is_null_number(Some(Number::Float(value))) {
            return self.new_number();
        }
        if self.r#type == Type::Boolean {
            if value == BOOLEAN_FALSE as f32 {
                return Number::Integer(BOOLEAN_FALSE);
            }
            return Number::Integer(BOOLEAN_TRUE);
        }
        if self.r#type == Type::Integer {
            return Number::Integer(value as i32);
        }
        if self.r#type == Type::Double {
            return Number::Double(value as f64);
        }
        if self.r#type == Type::Long {
            return Number::Long(value as i64);
        }
        panic!("java.lang.IllegalStateException: type={}", self.r#type);
    }

    /// Java `isNull(Number)`.  Returns true if number is the null value for this type.
    pub fn is_null_number(&self, number: Option<Number>) -> bool {
        let number = match number {
            None => return true,
            Some(number) => number,
        };
        let null_value = ConstEtomoNumber::is_number_null(Some(number));
        if !null_value {
            return false;
        }
        if self.r#type == Type::Boolean || self.r#type == Type::Integer {
            return matches!(number, Number::Integer(_));
        }
        if self.r#type == Type::Double {
            return matches!(number, Number::Double(_));
        }
        if self.r#type == Type::Long {
            return matches!(number, Number::Long(_));
        }
        false
    }

    /// Java `isNumberNull`.  Return true if number is the null value of its instance.
    /// So for Integer, the Integer null value is null, but for Short only the Short null
    /// value is null.  Except for Double and Float, the minimum value is used for all
    /// null values.  NaN values are used for Double and Float.  The function will not
    /// return true for another type's null value.  The exception to this is Double and
    /// Float.  They recognize each other's NaN values with their isNaN functions.
    pub fn is_number_null(number: Option<Number>) -> bool {
        let number = match number {
            None => return true,
            Some(number) => number,
        };
        if matches!(
            number,
            Number::AtomicInteger(_) | Number::BigInteger(_) | Number::Integer(_)
        ) {
            return number.int_value() == i32::MIN;
        }
        if matches!(
            number,
            Number::AtomicLong(_) | Number::BigDecimal(_) | Number::Long(_)
        ) {
            return number.long_value() == i64::MIN;
        }
        if matches!(number, Number::Byte(_)) {
            return number.byte_value() == i8::MIN;
        }
        if matches!(number, Number::Double(_)) {
            return number.double_value().is_nan();
        }
        if matches!(number, Number::Float(_)) {
            return number.float_value().is_nan();
        }
        if matches!(number, Number::Short(_)) {
            return number.short_value() == i16::MIN;
        }
        // The source throws IllegalStateException for an unknown Number subclass; the
        // enum above has no other variant.
        unreachable!()
    }

    /// Java `gt(Number, Number)`.
    pub(crate) fn gt_numbers(&self, number1: Option<Number>, number2: Option<Number>) -> bool {
        self.compare_numbers(Comparison::Gt, number1, number2)
    }

    /// Java `ge(Number, Number)`.
    pub(crate) fn ge_numbers(&self, number1: Option<Number>, number2: Option<Number>) -> bool {
        self.compare_numbers(Comparison::Ge, number1, number2)
    }

    /// Java `lt(Number, Number)`.
    pub(crate) fn lt_numbers(&self, number1: Option<Number>, number2: Option<Number>) -> bool {
        self.compare_numbers(Comparison::Lt, number1, number2)
    }

    /// Java `le(Number, Number)`.
    pub(crate) fn le_numbers(&self, number1: Option<Number>, number2: Option<Number>) -> bool {
        self.compare_numbers(Comparison::Le, number1, number2)
    }

    /// Java `equals(Number, Number)`, the two-argument public overload.
    pub fn equals_numbers(&self, number1: Option<Number>, number2: Option<Number>) -> bool {
        self.compare_numbers(Comparison::Equals, number1, number2)
    }

    /// Java `add(Number, Number)`.
    pub(crate) fn add(&self, number1: Option<Number>, number2: Option<Number>) -> Number {
        self.do_math_numbers(Operator::Add, number1, number2)
    }

    /// Java `multiply(Number, Number)`.  If one of the numbers is null, the result is
    /// null.
    pub(crate) fn multiply(&self, number1: Option<Number>, number2: Option<Number>) -> Number {
        self.do_math_numbers(Operator::Multiply, number1, number2)
    }

    /// Java `divideBy(Number, Number)`.  If one of the numbers is null, the result is
    /// null.
    pub(crate) fn divide_by(&self, number1: Option<Number>, number2: Option<Number>) -> Number {
        self.do_math_numbers(Operator::DivideBy, number1, number2)
    }

    /// Java `subtract(Number, Number)`.
    pub(crate) fn subtract(&self, number1: Option<Number>, number2: Option<Number>) -> Number {
        self.do_math_numbers(Operator::Subtract, number1, number2)
    }

    /// Java `negate(Number)`.  The `-1` argument autoboxes to an `Integer`.
    pub(crate) fn negate(&self, number: Option<Number>) -> Number {
        self.do_math_numbers(Operator::Multiply, number, Some(Number::Integer(-1)))
    }

    /// Java `getType`.
    pub fn get_type(&self) -> Type {
        self.r#type
    }

    /// Java `validateFloorAndCeiling`.  Validation for floor and ceiling.
    fn validate_floor_and_ceiling(&self) {
        // if floorValue and ceilingValue are both used, then floorValue must be less
        // then or equal to ceilingValue.
        if !self.is_null_number(Some(self.ceiling_value))
            && !self.is_null_number(Some(self.floor_value))
            && self.gt_numbers(Some(self.floor_value), Some(self.ceiling_value))
        {
            panic!(
                "java.lang.IllegalStateException: FloorValue cannot be greater then ceilingValue.\nfloorValue={}, ceilingValue={}",
                self.floor_value, self.ceiling_value
            );
        }
    }

    // Representation limit: Java `internalTestDeepCopy(ConstEtomoNumber)`
    // (ConstEtomoNumber.java:1489-1522) is untranslated for the same reason as
    // `internalTest()` above - its checks are `this.x == that.x` reference-identity
    // comparisons asserting that a copy constructor boxed new objects, and this module's
    // fields are values, so the question does not arise.  Its guard,
    // `Utilities.isSelfTest()`, is available in `etomo/util/utilities.rs`.

    /// Java `compare(Comparison, Number, Number)`.
    fn compare_numbers(
        &self,
        comparison: Comparison,
        number1: Option<Number>,
        number2: Option<Number>,
    ) -> bool {
        if comparison == Comparison::Equals
            && self.is_null_number(number1)
            && self.is_null_number(number2)
        {
            return true;
        }
        if self.is_null_number(number1) || self.is_null_number(number2) {
            return false;
        }
        let number1 = number1.unwrap();
        let number2 = number2.unwrap();
        // Avoid casting double to/from float because it causes floating point errors
        if matches!(number1, Number::Double(_)) {
            if matches!(number2, Number::Double(_)) {
                return ConstEtomoNumber::compare_double_double(
                    comparison,
                    number1.double_value(),
                    number2.double_value(),
                );
            }
            if matches!(number2, Number::Float(_)) {
                return ConstEtomoNumber::compare_double_float(
                    comparison,
                    number1.double_value(),
                    number2.float_value(),
                );
            }
            return ConstEtomoNumber::compare_double_long(
                comparison,
                number1.double_value(),
                number2.long_value(),
            );
        }
        if matches!(number1, Number::Float(_)) {
            if matches!(number2, Number::Double(_)) {
                return ConstEtomoNumber::compare_float_double(
                    comparison,
                    number1.float_value(),
                    number2.double_value(),
                );
            }
            if matches!(number2, Number::Float(_)) {
                return ConstEtomoNumber::compare_float_float(
                    comparison,
                    number1.float_value(),
                    number2.float_value(),
                );
            }
            return ConstEtomoNumber::compare_float_long(
                comparison,
                number1.float_value(),
                number2.long_value(),
            );
        }
        if matches!(number2, Number::Double(_)) {
            return ConstEtomoNumber::compare_long_double(
                comparison,
                number1.long_value(),
                number2.double_value(),
            );
        }
        if matches!(number2, Number::Float(_)) {
            return ConstEtomoNumber::compare_long_float(
                comparison,
                number1.long_value(),
                number2.float_value(),
            );
        }
        ConstEtomoNumber::compare_long_long(comparison, number1.long_value(), number2.long_value())
    }

    /// Java `doMath(Operator, Number, Number)`.
    fn do_math_numbers(
        &self,
        operator: Operator,
        number1: Option<Number>,
        number2: Option<Number>,
    ) -> Number {
        let mut null1 = false;
        let mut null2 = false;
        null1 = self.is_null_number(number1);
        if null1 || {
            null2 = self.is_null_number(number2);
            null2
        } {
            if operator == Operator::Multiply || operator == Operator::DivideBy {
                return self.new_number();
            }
            if operator == Operator::Add {
                if null1 && null2 {
                    return self.new_number();
                }
                if null1 {
                    return self.new_number_from_number(number2);
                }
                if null2 {
                    return self.new_number_from_number(number1);
                }
            }
            if operator == Operator::Subtract {
                if null1 && null2 {
                    return self.new_number();
                }
                if null1 {
                    // The source calls doMath(MULTIPLY, -1, number2), whose first
                    // argument autoboxes to an Integer.
                    return self.do_math_numbers(
                        Operator::Multiply,
                        Some(Number::Integer(-1)),
                        number2,
                    );
                }
                if null2 {
                    return self.new_number_from_number(number1);
                }
            }
        }
        let number1 = match number1 {
            None => return self.new_number(),
            Some(number1) => number1,
        };
        let number2 = match number2 {
            None => return self.new_number(),
            Some(number2) => number2,
        };
        // Avoid casting double to/from float because it causes floating point errors
        if matches!(number1, Number::Double(_)) {
            if matches!(number2, Number::Double(_)) {
                return self.do_math_double_double(
                    operator,
                    number1.double_value(),
                    number2.double_value(),
                );
            }
            if matches!(number2, Number::Float(_)) {
                return self.do_math_double_float(
                    operator,
                    number1.double_value(),
                    number2.float_value(),
                );
            }
            return self.do_math_double_long(
                operator,
                number1.double_value(),
                number2.long_value(),
            );
        }
        if matches!(number1, Number::Float(_)) {
            if matches!(number2, Number::Double(_)) {
                return self.do_math_float_double(
                    operator,
                    number1.float_value(),
                    number2.double_value(),
                );
            }
            if matches!(number2, Number::Float(_)) {
                return self.do_math_float_float(
                    operator,
                    number1.float_value(),
                    number2.float_value(),
                );
            }
            return self.do_math_float_long(operator, number1.float_value(), number2.long_value());
        }
        if matches!(number2, Number::Double(_)) {
            return self.do_math_long_double(
                operator,
                number1.long_value(),
                number2.double_value(),
            );
        }
        if matches!(number2, Number::Float(_)) {
            return self.do_math_long_float(operator, number1.long_value(), number2.float_value());
        }
        self.do_math_long_long(operator, number1.long_value(), number2.long_value())
    }

    /// Java `doMath(Operator, double, double)`.
    fn do_math_double_double(&self, operator: Operator, number1: f64, number2: f64) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_double(number1 + number2);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_double(number1 * number2);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_double(number1 / number2);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, double, float)`.
    fn do_math_double_float(&self, operator: Operator, number1: f64, number2: f32) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_double(number1 + number2 as f64);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_double(number1 * number2 as f64);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_double(number1 / number2 as f64);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, double, long)`.
    fn do_math_double_long(&self, operator: Operator, number1: f64, number2: i64) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_double(number1 + number2 as f64);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_double(number1 * number2 as f64);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_double(number1 / number2 as f64);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, float, double)`.
    fn do_math_float_double(&self, operator: Operator, number1: f32, number2: f64) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_double(number1 as f64 + number2);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_double(number1 as f64 * number2);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_double(number1 as f64 / number2);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, float, float)`.  The sum, product and quotient are
    /// single-precision, and only then widen for `newNumber(float)`.
    fn do_math_float_float(&self, operator: Operator, number1: f32, number2: f32) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_float(number1 + number2);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_float(number1 * number2);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_float(number1 / number2);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, float, long)`.  Java promotes the `long` to `float` here,
    /// so the arithmetic is single-precision.
    fn do_math_float_long(&self, operator: Operator, number1: f32, number2: i64) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_float(number1 + number2 as f32);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_float(number1 * number2 as f32);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_float(number1 / number2 as f32);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, long, double)`.
    fn do_math_long_double(&self, operator: Operator, number1: i64, number2: f64) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_double(number1 as f64 + number2);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_double(number1 as f64 * number2);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_double(number1 as f64 / number2);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, long, float)`.
    fn do_math_long_float(&self, operator: Operator, number1: i64, number2: f32) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_float(number1 as f32 + number2);
        }
        if operator == Operator::Multiply {
            return self.new_number_from_float(number1 as f32 * number2);
        }
        if operator == Operator::DivideBy {
            return self.new_number_from_float(number1 as f32 / number2);
        }
        self.new_number()
    }

    /// Java `doMath(Operator, long, long)`.  Integer arithmetic wraps in Java, and
    /// division by zero throws ArithmeticException.
    fn do_math_long_long(&self, operator: Operator, number1: i64, number2: i64) -> Number {
        if operator == Operator::Add {
            return self.new_number_from_long(number1.wrapping_add(number2));
        }
        if operator == Operator::Multiply {
            return self.new_number_from_long(number1.wrapping_mul(number2));
        }
        if operator == Operator::DivideBy {
            if number2 == 0 {
                panic!("java.lang.ArithmeticException: / by zero");
            }
            return self.new_number_from_long(number1.wrapping_div(number2));
        }
        self.new_number()
    }

    /// Java `compare(Comparison, double, double)`.
    fn compare_double_double(comparison: Comparison, number1: f64, number2: f64) -> bool {
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, double, float)`.
    fn compare_double_float(comparison: Comparison, number1: f64, number2: f32) -> bool {
        let number2 = number2 as f64;
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, double, long)`.
    fn compare_double_long(comparison: Comparison, number1: f64, number2: i64) -> bool {
        let number2 = number2 as f64;
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, float, double)`.
    fn compare_float_double(comparison: Comparison, number1: f32, number2: f64) -> bool {
        let number1 = number1 as f64;
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, float, float)`.
    fn compare_float_float(comparison: Comparison, number1: f32, number2: f32) -> bool {
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, float, long)`.  Java promotes the `long` to `float`.
    fn compare_float_long(comparison: Comparison, number1: f32, number2: i64) -> bool {
        let number2 = number2 as f32;
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, long, double)`.
    fn compare_long_double(comparison: Comparison, number1: i64, number2: f64) -> bool {
        let number1 = number1 as f64;
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, long, float)`.
    fn compare_long_float(comparison: Comparison, number1: i64, number2: f32) -> bool {
        let number1 = number1 as f32;
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }

    /// Java `compare(Comparison, long, long)`.
    fn compare_long_long(comparison: Comparison, number1: i64, number2: i64) -> bool {
        if comparison == Comparison::Gt {
            return number1 > number2;
        }
        if comparison == Comparison::Ge {
            return number1 >= number2;
        }
        if comparison == Comparison::Lt {
            return number1 < number2;
        }
        if comparison == Comparison::Le {
            return number1 <= number2;
        }
        if comparison == Comparison::Equals {
            return number1 == number2;
        }
        false
    }
}

/// Java `toString()`.  Must show the string version of `getValue()` - not a description!
impl std::fmt::Display for ConstEtomoNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_number(Some(self.get_value())))
    }
}
