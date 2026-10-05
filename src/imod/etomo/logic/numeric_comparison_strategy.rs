//! `IMOD/Etomo/src/etomo/logic/NumericComparisonStrategy.java`.

use super::comparison_strategy::ComparisonStrategy;
use super::strategy_does_not_apply_exception::StrategyDoesNotApplyException;

/// Java `public final class NumericComparisonStrategy implements ComparisonStrategy`.
pub struct NumericComparisonStrategy;

/// Java `INSTANCE`.
pub static INSTANCE: NumericComparisonStrategy = NumericComparisonStrategy;

/// The value of `new java.math.BigDecimal(String)`, normalized so that two values are
/// numerically equal (`compareTo == 0`) exactly when the normal forms are equal: zero,
/// or a sign, the significant digits without leading or trailing zeros, and the power
/// of ten of the last digit.  `Err` carries the `NumberFormatException` message the
/// JDK gives for a string the constructor rejects.
fn big_decimal(value: &str) -> Result<Option<(bool, String, i64)>, String> {
    let bytes = value.as_bytes();
    let mut index = 0;
    let mut negative = false;
    if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
        negative = bytes[index] == b'-';
        index += 1;
    }
    let mut digits = String::new();
    let mut fraction_digits: i64 = 0;
    let mut seen_point = false;
    let mut seen_digit = false;
    while index < bytes.len() {
        let c = bytes[index];
        if c.is_ascii_digit() {
            seen_digit = true;
            digits.push(c as char);
            if seen_point {
                fraction_digits += 1;
            }
        } else if c == b'.' && !seen_point {
            seen_point = true;
        } else {
            break;
        }
        index += 1;
    }
    if !seen_digit {
        return Err("Character array is missing \"exponent\" mark 'e' or 'E'.".to_owned());
    }
    let mut exponent: i64 = 0;
    if index < bytes.len() {
        if bytes[index] != b'e' && bytes[index] != b'E' {
            return Err(format!(
                "Character {} is neither a decimal digit number, decimal point, nor \"e\" notation exponential mark.",
                bytes[index] as char
            ));
        }
        index += 1;
        let exponent_string = &value[index..];
        exponent = exponent_string
            .parse::<i32>()
            .map_err(|_| "Exponent overflow.".to_owned())? as i64;
    }
    let scale_exponent = exponent - fraction_digits;
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return Ok(None);
    }
    let without_trailing = trimmed.trim_end_matches('0');
    let removed = (trimmed.len() - without_trailing.len()) as i64;
    Ok(Some((
        negative,
        without_trailing.to_owned(),
        scale_exponent + removed,
    )))
}

impl ComparisonStrategy for NumericComparisonStrategy {
    /// Java `equals(String, String)`.  Attempts to do a numeric comparison on
    /// leftString and rightString.  Handles integers and floats of any size.  Does not
    /// have a problem with floating point errors.  Throws (returns `Err`) if either
    /// parameter is non-numeric or null; returns true if the parameters are numerically
    /// equal.
    fn equals(
        &self,
        left_string: Option<&str>,
        right_string: Option<&str>,
    ) -> Result<bool, StrategyDoesNotApplyException> {
        let (Some(left_string), Some(right_string)) = (left_string, right_string) else {
            return Err(StrategyDoesNotApplyException::new_with_message(Some("null")));
        };
        let left = big_decimal(left_string)
            .map_err(|message| StrategyDoesNotApplyException::new_with_message(Some(&message)))?;
        let right = big_decimal(right_string)
            .map_err(|message| StrategyDoesNotApplyException::new_with_message(Some(&message)))?;
        Ok(left == right)
    }
}
