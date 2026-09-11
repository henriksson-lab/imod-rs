//! `IMOD/Etomo/src/etomo/logic/PopupTool.java`.
#![allow(dead_code)]

pub const MAX_LINES: usize = 20;
pub const WIDTH: usize = 60;

/// Java `linefeed(ArrayList<String>)`.
pub fn linefeed(message_array: Option<Vec<String>>) -> Vec<String> {
    let mut message_array = message_array.unwrap_or_default();
    message_array.push(" ".into());
    message_array
}

/// Java `wrapMessage(String, ArrayList<String>)`.
pub fn wrap_message(message: Option<&str>, message_array: Option<Vec<String>>) -> Vec<String> {
    let mut message_array = message_array.unwrap_or_default();
    let Some(message) = message else {
        if message_array.is_empty() {
            message_array.push(" ".into());
        }
        return message_array;
    };
    if message == "\n" {
        message_array.push(" ".into());
        return message_array;
    }
    // Java `String.split("\\n")` drops trailing empty fields.  Keep its one
    // special empty-input field, which is handled as a blank popup line below.
    let lines: Vec<&str> = if message.is_empty() {
        vec![""]
    } else {
        let trimmed = message.trim_end_matches('\n');
        if trimmed.is_empty() {
            vec![]
        } else {
            trimmed.split('\n').collect()
        }
    };
    for line in lines {
        if line.is_empty() {
            message_array.push(" ".into());
            continue;
        }
        let mut index = 0;
        while index < line.len() && message_array.len() < MAX_LINES {
            let mut end_index = (index + WIDTH).min(line.len());
            while !line.is_char_boundary(end_index) {
                end_index -= 1;
            }
            let mut new_end_index = end_index;
            let mut last_char = ' ';
            while new_end_index < line.len() {
                let c = line[new_end_index..].chars().next().unwrap();
                if c.is_whitespace() || last_char == ',' {
                    break;
                }
                last_char = c;
                new_end_index += c.len_utf8();
            }
            message_array.push(line[index..new_end_index].to_string());
            index = new_end_index;
        }
    }
    message_array
}

/// Java `adjustLocationY(int, int, int)`.
pub fn adjust_location_y(popup_y: i32, parent_height: i32, popup_height: i32) -> i32 {
    (popup_y - (parent_height / 2) - (popup_height / 2) - 20).max(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_wrap_and_adjust_rules_hold() {
        assert_eq!(wrap_message(None, None), vec![" "]);
        assert_eq!(wrap_message(Some("\n"), None), vec![" "]);
        assert_eq!(linefeed(None), vec![" "]);
        assert_eq!(adjust_location_y(10, 20, 20), 0);
    }
}
