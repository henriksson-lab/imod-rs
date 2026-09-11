//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/logic/Converter.java` and the two `java.lang.Math.round`
//! overloads it leans on.
//!
//! Every expectation below was read out of a reference runtime built the way the header
//! of `tests/etomo_utilities_probe.rs` describes.  The `Math.round` rows are the reason
//! the overloads are modelled bit-for-bit rather than as `floor(a + 0.5)`: the javadoc's
//! floor wording has not described the implementation since JDK-8010430, and
//! `0.49999999999999994` / `0.49999997f` are where the two answers differ.
use imod_rs::imod::etomo::logic::converter;
use imod_rs::imod::etomo::util::utilities::java_lang_math_round;

fn s(value: Option<i64>) -> String {
    value.map(|v| v.to_string()).unwrap_or("null".to_string())
}
fn si(value: Option<i32>) -> String {
    value.map(|v| v.to_string()).unwrap_or("null".to_string())
}

#[test]
fn jvm_verified_converter_to_long() {
    let cases: Vec<(Option<&str>, &str)> = vec![
        (None, "null"),
        (Some(""), "null"),
        (Some("   "), "null"),
        (Some("12"), "12"),
        (Some("-12"), "-12"),
        (Some("+12"), "12"),
        (Some("12.5"), "13"),
        (Some("-12.5"), "-12"),
        (Some("12.4"), "12"),
        (Some("-12.4"), "-12"),
        (Some("0.5"), "1"),
        (Some("-0.5"), "0"),
        (Some("1.5"), "2"),
        (Some("2.5"), "3"),
        (Some("1e3"), "1000"),
        (Some("1E3"), "1000"),
        (Some("abc"), "null"),
        (Some("12abc"), "null"),
        (Some(" 12 "), "12"),
        (Some("2147483647"), "2147483647"),
        (Some("2147483648"), "2147483648"),
        (Some("-2147483648"), "-2147483648"),
        (Some("-2147483649"), "-2147483649"),
        (Some("9223372036854775807"), "9223372036854775807"),
        (Some("9223372036854775808"), "9223372036854775807"),
        (Some("1.0f"), "1"),
        (Some("1.0d"), "1"),
        (Some("NaN"), "0"),
        (Some("Infinity"), "9223372036854775807"),
        (Some("-Infinity"), "-9223372036854775808"),
        (Some("0x10"), "null"),
    ];
    for (text, expected) in cases.iter() {
        assert_eq!(
            s(converter::to_long(*text)),
            *expected,
            "toLong({:?})",
            text
        );
    }
}

#[test]
fn jvm_verified_converter_to_integer() {
    let cases: Vec<(Option<&str>, &str, &str)> = vec![
        (None, "null", "null"),
        (Some(""), "null", "null"),
        (Some("   "), "null", "null"),
        (Some("12"), "12", "12"),
        (Some("-12"), "-12", "-12"),
        (Some("+12"), "12", "12"),
        (Some("12.5"), "null", "13"),
        (Some("-12.5"), "null", "-12"),
        (Some("12.4"), "null", "12"),
        (Some("-12.4"), "null", "-12"),
        (Some("0.5"), "null", "1"),
        (Some("-0.5"), "null", "0"),
        (Some("1.5"), "null", "2"),
        (Some("2.5"), "null", "3"),
        (Some("1e3"), "null", "1000"),
        (Some("1E3"), "null", "1000"),
        (Some("abc"), "null", "null"),
        (Some("12abc"), "null", "null"),
        (Some(" 12 "), "null", "12"),
        (Some("2147483647"), "2147483647", "2147483647"),
        (Some("2147483648"), "null", "2147483647"),
        (Some("-2147483648"), "-2147483648", "-2147483648"),
        (Some("-2147483649"), "null", "-2147483648"),
        (Some("9223372036854775807"), "null", "2147483647"),
        (Some("9223372036854775808"), "null", "2147483647"),
        (Some("1.0f"), "null", "1"),
        (Some("1.0d"), "null", "1"),
        (Some("NaN"), "null", "0"),
        (Some("Infinity"), "null", "2147483647"),
        (Some("-Infinity"), "null", "-2147483648"),
        (Some("0x10"), "null", "null"),
    ];
    for (text, expected, expected_round) in cases.iter() {
        assert_eq!(
            si(converter::to_integer(*text)),
            *expected,
            "toInteger({:?})",
            text
        );
        assert_eq!(
            si(converter::to_integer_with_round(*text, true)),
            *expected_round,
            "toInteger({:?}, true)",
            text
        );
    }
}

#[test]
fn jvm_verified_converter_to_double_and_arrays() {
    assert_eq!(converter::to_double(None), None);
    assert_eq!(converter::to_double(Some("")), None);
    assert_eq!(converter::to_double(Some("   ")), None);
    assert_eq!(converter::to_double(Some("12")), Some(12.0));
    assert_eq!(converter::to_double(Some("12.4")), Some(12.4));
    assert_eq!(converter::to_double(Some(" 12 ")), Some(12.0));
    assert_eq!(converter::to_double(Some("1e3")), Some(1000.0));
    assert_eq!(converter::to_double(Some("abc")), None);
    assert_eq!(converter::to_double(Some("0x10")), None);
    assert_eq!(converter::to_double(Some("Infinity")), Some(f64::INFINITY));
    assert!(converter::to_double(Some("NaN")).unwrap().is_nan());

    assert_eq!(converter::to_integer_from_long(Some(5), false), Some(5));
    assert_eq!(converter::to_integer_from_long(None, false), None);
    assert_eq!(
        converter::to_integer_from_long(Some(2147483648), true),
        None
    );
    assert_eq!(
        converter::to_integer_from_long(Some(-2147483649), true),
        None
    );
    assert_eq!(
        converter::to_integer_from_long(Some(-2147483648), true),
        Some(-2147483648)
    );

    assert_eq!(converter::to_array(None), None);
    assert_eq!(converter::to_array(Some(&[])), None);
    assert_eq!(
        converter::to_array(Some(&[Some("one".to_string())])),
        Some(vec![Some("one".to_string())])
    );
    assert_eq!(
        converter::to_array(Some(&[Some("one".to_string()), Some("two".to_string())])),
        Some(vec![Some("one".to_string()), Some("two".to_string())])
    );
}

#[test]
fn jvm_verified_math_round_double() {
    // The last two rows are where `floor(a + 0.5)` disagrees with a real JVM.
    let cases: Vec<(f64, i64)> = vec![
        (0.5, 1),
        (-0.5, 0),
        (1.5, 2),
        (2.5, 3),
        (-1.5, -1),
        (-2.5, -2),
        (4503599627370496.0, 4503599627370496),
        (9.007199254740992E15, 9007199254740992),
        (f64::NAN, 0),
        (f64::INFINITY, i64::MAX),
        (f64::NEG_INFINITY, i64::MIN),
        (0.49999999999999994, 0),
        (-0.4999999999999999, 0),
    ];
    for (value, expected) in cases.iter() {
        assert_eq!(java_lang_math_round(*value), *expected, "round({})", value);
    }
}
