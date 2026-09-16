//! Translation of `IMOD/imodutil/echo2.c`.

use std::io::Write;

/// C `main` in `echo2.c`: write its arguments to standard error.
pub fn echo2(arguments: &[String]) -> i32 {
    if arguments.len() <= 1 {
        return 0;
    }
    let (newline, values) = if arguments[1] == "-n" {
        (false, &arguments[2..])
    } else {
        (true, &arguments[1..])
    };
    let mut stderr = std::io::stderr().lock();
    let _ = stderr.write_all(values.join(" ").as_bytes());
    if newline {
        let _ = stderr.write_all(b"\n");
    }
    0
}

#[cfg(test)]
mod tests {
    use super::echo2;

    #[test]
    fn accepts_empty_and_no_newline_forms() {
        assert_eq!(echo2(&["echo2".into()]), 0);
        assert_eq!(echo2(&["echo2".into(), "-n".into(), "text".into()]), 0);
    }
}
