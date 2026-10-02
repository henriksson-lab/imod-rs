//! `IMOD/Etomo/src/etomo/type/Time.java`.
//!
//! Description: Represents hours, minutes, and seconds.  Can pull hours, minutes, and
//! seconds out of a longer string.  Can compare times with almost equals, which allows
//! off-by-one errors in seconds.
//!
//! Formats parsed:
//! hours:minutes:seconds
//! hours:minutesAM/PM
//!
//! Copyright: Copyright 2006
//!
//! Organization: Boulder Laboratory for 3-Dimensional Electron Microscopy of Cells
//! (BL3DEMC), University of Colorado

use super::const_etomo_number::java_lang_integer_parse_int;
use crate::imod::etomo::util::utilities::java_lang_string_split;
use regex::Regex;
use std::sync::LazyLock;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private `DIVIDER`.
const DIVIDER: char = ':';

/// Java `"\\s+"`, with Java's `\s` class.
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("[ \t\n\u{0B}\u{0C}\r]+").unwrap());
/// Java `String.valueOf(DIVIDER)` used as a regex.
static DIVIDER_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(":").unwrap());

/// Java `Time`.
#[derive(Clone, Debug)]
pub struct Time {
    /// Java private field `hours`, initialised to 0.
    hours: i32,
    /// Java private field `minutes`, initialised to 0.
    minutes: i32,
    /// Java private field `seconds`, initialised to 0.
    seconds: i32,
}

impl Time {
    /// Java `Time(String)`.
    pub fn new(date: &str) -> Time {
        let mut time = Time {
            hours: 0,
            minutes: 0,
            seconds: 0,
        };
        time.parse(Some(date));
        time
    }

    /// Java private `reset`.
    fn reset(&mut self) {
        self.hours = 0;
        self.minutes = 0;
        self.seconds = 0;
    }

    /// Java private `parse(String)`.  Finds the first instance of a substring with the
    /// format hours:minutes:seconds or hours:minutesAM/PM in the date parameter and
    /// uses it to set the member variables.
    fn parse(&mut self, date: Option<&str>) {
        let date = match date {
            None => return,
            Some(date) => date,
        };
        let date_array = java_lang_string_split(date, &WHITESPACE);
        for i in 0..date_array.len() {
            if date_array[i].contains(DIVIDER) {
                let time_array = java_lang_string_split(&date_array[i], &DIVIDER_PATTERN);
                if time_array.len() == 3 {
                    if self.parse_hours_minutes_seconds(&time_array) {
                        break;
                    }
                } else if time_array.len() == 2 {
                    if self.parse_hours_minutes_ampm(&time_array) {
                        break;
                    }
                }
            }
        }
    }

    /// Java private `parseHoursMinutesSeconds(String[])`.
    fn parse_hours_minutes_seconds(&mut self, time_array: &[String]) -> bool {
        self.reset();
        // The Java `try` block; `catch (NumberFormatException e)` resets and fails.
        let parsed = (|| -> Result<(), String> {
            self.hours = java_lang_integer_parse_int(&time_array[0])?;
            self.minutes = java_lang_integer_parse_int(&time_array[1])?;
            self.seconds = java_lang_integer_parse_int(&time_array[2])?;
            Ok(())
        })();
        if parsed.is_err() {
            self.reset();
            return false;
        }
        true
    }

    /// Java private `parseHoursMinutesAMPM(String[])`.
    fn parse_hours_minutes_ampm(&mut self, time_array: &[String]) -> bool {
        self.reset();
        match java_lang_integer_parse_int(&time_array[0]) {
            Ok(hours) => self.hours = hours,
            Err(_) => {
                self.reset();
                return false;
            }
        }
        let minutes_am_pm: Vec<char> = time_array[1].chars().collect();
        let am_pm_start_index = minutes_am_pm.len() as i32 - 2;
        // Time.java:104 calls `substring(amPmStartIndex, length)` with a negative start
        // when the minutes field is shorter than two characters, which throws an
        // uncaught StringIndexOutOfBoundsException.  Fixed in translation: such a field
        // is not an AM/PM time, so the parse fails as it does for an unknown suffix.
        if am_pm_start_index < 0 {
            self.reset();
            return false;
        }
        let am_pm: String = minutes_am_pm[am_pm_start_index as usize..].iter().collect();
        if am_pm == "PM" {
            self.hours = self.hours.wrapping_add(12);
        } else if am_pm != "AM" {
            self.reset();
            return false;
        }
        let minutes: String = minutes_am_pm[..am_pm_start_index as usize].iter().collect();
        // Time.java:112 parses the minutes outside any `try`, so a non-numeric minutes
        // field throws an uncaught NumberFormatException.  Fixed in translation: the
        // parse fails, as it does for a bad hours field.
        match java_lang_integer_parse_int(&minutes) {
            Ok(minutes) => self.minutes = minutes,
            Err(_) => {
                self.reset();
                return false;
            }
        }
        true
    }

    /// Java `almostEquals(Time)`.  True if anotherTime equals the instance give or take
    /// one minute.
    pub fn almost_equals(&self, another_time: &Time) -> bool {
        (self.get_time() - another_time.get_time()).abs() <= 3600
    }

    /// Java private `getTime`.  Gets total time in milliseconds.  (The source multiplies
    /// every field by 60 in `int` arithmetic before widening to `long`; kept.)
    fn get_time(&self) -> i64 {
        self.hours
            .wrapping_mul(60)
            .wrapping_add(self.minutes.wrapping_mul(60))
            .wrapping_add(self.seconds.wrapping_mul(60)) as i64
    }
}

/// Java `toString`.
impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}{}{}{}",
            self.hours, DIVIDER, self.minutes, DIVIDER, self.seconds
        )
    }
}
