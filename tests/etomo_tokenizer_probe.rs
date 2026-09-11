//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/util/PrimativeTokenizer.java` and
//! `IMOD/Etomo/src/etomo/ui/swing/Token.java`.
//!
//! The expected strings below were produced by running a harness against a reference
//! runtime built the way the header of `tests/etomo_utilities_probe.rs` describes
//! (`javac -nowarn -encoding ISO-8859-1` over every `.java` under `IMOD/Etomo/src`
//! except the test classes, then `java -Djava.awt.headless=true`).  The harness printed
//! one line per input, each token rendered as `(TYPE,value)` with every character
//! outside `[32,127)` escaped as `\uXXXX`, which is what `render` below reproduces.
//!
//! `java.io.StreamTokenizer`'s behaviour under the configuration
//! `initializeStreamTokenizer` applies is the point of the exercise: `resetSyntax()`
//! leaves no whitespace, digit, quote or comment class at all, so `'\n'` comes back as
//! an ordinary character whose value is numerically `TT_EOL` (which is declared as
//! `'\n'`), `'\r'` comes back as the ordinary character 13, and every character at or
//! above 256 is treated as a word character.
use imod_rs::imod::etomo::ui::swing::token::{self, Token};
use imod_rs::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;

/// The harness's `esc`: `String.format("\\u%04x", (int) c)` for anything outside
/// `[32,127)`.
fn esc(value: Option<&str>) -> String {
    let value = match value {
        None => return "null".to_string(),
        Some(value) => value,
    };
    let mut buffer = String::new();
    for unit in value.encode_utf16() {
        if (32..127).contains(&unit) {
            buffer.push(unit as u8 as char);
        } else {
            buffer.push_str(&format!("\\u{:04x}", unit));
        }
    }
    buffer
}

/// The harness's `probe`: tokenize until EOF and render every token.
fn render(text: &str, numeric: bool) -> String {
    let mut tokenizer = if numeric {
        PrimativeTokenizer::get_numeric_string_instance(text, false)
    } else {
        PrimativeTokenizer::get_string_instance(text, false)
    };
    let _ = tokenizer.initialize();
    let mut buffer = String::new();
    let mut guard = 0;
    loop {
        let t = unsafe { Box::from_raw(tokenizer.next(std::ptr::null_mut())) };
        buffer.push_str(&format!(" ({},{})", t.get_type(), esc(t.get_value())));
        guard += 1;
        if guard > 60 || t.is(token::Type::Eof) {
            break;
        }
    }
    buffer
}

/// Builds a string from UTF-16 code units, as the harness's `ch(int)` does.
fn ch(units: &[u16]) -> String {
    String::from_utf16(units).unwrap()
}

#[test]
fn jvm_verified_primative_tokenizer_string_mode() {
    let cases: Vec<(&str, String)> = vec![
        ("cr", format!("a{}b", ch(&[13]))),
        ("crcrlf", format!("a{}{}b", ch(&[13, 13]), ch(&[10]))),
        ("crcr", format!("a{}b", ch(&[13, 13]))),
        ("crlf", format!("a{}b", ch(&[13, 10]))),
        ("lfcr", format!("a{}b", ch(&[10, 13]))),
        ("cr_eof", format!("a{}", ch(&[13]))),
        ("latin1_e9", format!("a{}b", ch(&[233]))),
        ("latin1_ff", format!("a{}b", ch(&[255]))),
        ("high_100", format!("a{}b", ch(&[256]))),
        ("high_20ac", format!("a{}b", ch(&[0x20AC]))),
        ("high_only", ch(&[0x20AC])),
        ("high_mix", format!("ab{}12", ch(&[256]))),
        ("surrogate", format!("a{}b", ch(&[0xD83D, 0xDE00]))),
        ("nul", format!("a{}b", ch(&[0]))),
        ("del", format!("a{}b", ch(&[127]))),
        ("ws_nl_ws", format!("a {} b", ch(&[10]))),
        ("formfeed", format!("a{}b", ch(&[12]))),
        ("vtab", format!("a{}b", ch(&[11]))),
        ("backslash", "a\\b".to_string()),
        ("dollar", "a$b".to_string()),
        ("plain", "abc def".to_string()),
        ("sym", "a.b[c]d".to_string()),
        ("quote", "\"hi there\"".to_string()),
        ("num", "12.5e3 007".to_string()),
        ("mixed", "abc123def".to_string()),
        ("empty", "".to_string()),
        ("onlynl", ch(&[10])),
        ("trailnl", format!("ab{}", ch(&[10]))),
        ("multinl", format!("a{}b", ch(&[10, 10]))),
        ("comment", format!("# hello{}x", ch(&[10]))),
        ("delim", "key = value".to_string()),
    ];
    // Captured from the reference JVM.
    let expected: Vec<(&str, &str)> = vec![
        (
            "cr",
            " (ALPHANUM,a) (WHITESPACE,\\u000d) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "crcrlf",
            " (ALPHANUM,a) (WHITESPACE,\\u000d) (EOL,null) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "crcr",
            " (ALPHANUM,a) (WHITESPACE,\\u000d\\u000d) (ALPHANUM,b) (EOF,null)",
        ),
        ("crlf", " (ALPHANUM,a) (EOL,null) (ALPHANUM,b) (EOF,null)"),
        (
            "lfcr",
            " (ALPHANUM,a) (EOL,null) (WHITESPACE,\\u000d) (ALPHANUM,b) (EOF,null)",
        ),
        ("cr_eof", " (ALPHANUM,a) (WHITESPACE,\\u000d) (EOF,null)"),
        (
            "latin1_e9",
            " (ALPHANUM,a) (WHITESPACE,\\u00e9) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "latin1_ff",
            " (ALPHANUM,a) (WHITESPACE,\\u00ff) (ALPHANUM,b) (EOF,null)",
        ),
        ("high_100", " (ALPHANUM,a\\u0100b) (EOF,null)"),
        ("high_20ac", " (ALPHANUM,a\\u20acb) (EOF,null)"),
        ("high_only", " (ALPHANUM,\\u20ac) (EOF,null)"),
        ("high_mix", " (ALPHANUM,ab\\u010012) (EOF,null)"),
        ("surrogate", " (ALPHANUM,a\\ud83d\\ude00b) (EOF,null)"),
        (
            "nul",
            " (ALPHANUM,a) (WHITESPACE,\\u0000) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "del",
            " (ALPHANUM,a) (WHITESPACE,\\u007f) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "ws_nl_ws",
            " (ALPHANUM,a) (WHITESPACE, ) (EOL,null) (WHITESPACE, ) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "formfeed",
            " (ALPHANUM,a) (WHITESPACE,\\u000c) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "vtab",
            " (ALPHANUM,a) (WHITESPACE,\\u000b) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "backslash",
            " (ALPHANUM,a) (SYMBOL,\\) (ALPHANUM,b) (EOF,null)",
        ),
        ("dollar", " (ALPHANUM,a) (SYMBOL,$) (ALPHANUM,b) (EOF,null)"),
        (
            "plain",
            " (ALPHANUM,abc) (WHITESPACE, ) (ALPHANUM,def) (EOF,null)",
        ),
        (
            "sym",
            " (ALPHANUM,a) (SYMBOL,.) (ALPHANUM,b) (SYMBOL,[) (ALPHANUM,c) (SYMBOL,]) (ALPHANUM,d) (EOF,null)",
        ),
        (
            "quote",
            " (SYMBOL,\") (ALPHANUM,hi) (WHITESPACE, ) (ALPHANUM,there) (SYMBOL,\") (EOF,null)",
        ),
        (
            "num",
            " (ALPHANUM,12) (SYMBOL,.) (ALPHANUM,5e3) (WHITESPACE, ) (ALPHANUM,007) (EOF,null)",
        ),
        ("mixed", " (ALPHANUM,abc123def) (EOF,null)"),
        ("empty", " (EOF,null)"),
        ("onlynl", " (EOL,null) (EOF,null)"),
        ("trailnl", " (ALPHANUM,ab) (EOL,null) (EOF,null)"),
        (
            "multinl",
            " (ALPHANUM,a) (EOL,null) (EOL,null) (ALPHANUM,b) (EOF,null)",
        ),
        (
            "comment",
            " (SYMBOL,#) (WHITESPACE, ) (ALPHANUM,hello) (EOL,null) (ALPHANUM,x) (EOF,null)",
        ),
        (
            "delim",
            " (ALPHANUM,key) (WHITESPACE, ) (SYMBOL,=) (WHITESPACE, ) (ALPHANUM,value) (EOF,null)",
        ),
    ];
    for (index, (label, text)) in cases.iter().enumerate() {
        assert_eq!(*label, expected[index].0);
        assert_eq!(render(text, false), expected[index].1, "case {}", label);
    }
}

#[test]
fn jvm_verified_primative_tokenizer_numeric_mode() {
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "cr",
            format!("a{}b", ch(&[13])),
            " (ALPHABETICAL,a) (WHITESPACE,\\u000d) (ALPHABETICAL,b) (EOF,null)",
        ),
        (
            "crcrlf",
            format!("a{}b", ch(&[13, 13, 10])),
            " (ALPHABETICAL,a) (WHITESPACE,\\u000d) (EOL,null) (ALPHABETICAL,b) (EOF,null)",
        ),
        (
            "crlf",
            format!("a{}b", ch(&[13, 10])),
            " (ALPHABETICAL,a) (EOL,null) (ALPHABETICAL,b) (EOF,null)",
        ),
        ("high_only", ch(&[0x20AC]), " (NULL,null) (EOF,null)"),
        (
            "high_mix",
            format!("ab{}12", ch(&[256])),
            " (ALPHABETICAL,ab\\u0100) (NUMERIC,12) (EOF,null)",
        ),
        (
            "num",
            "12.5e3 007".to_string(),
            " (NUMERIC,12) (SYMBOL,.) (NUMERIC,5) (ALPHABETICAL,e) (NUMERIC,3) (WHITESPACE, ) (NUMERIC,007) (EOF,null)",
        ),
        (
            "mixed",
            "abc123def".to_string(),
            " (ALPHABETICAL,abc) (NUMERIC,123) (ALPHABETICAL,def) (EOF,null)",
        ),
        (
            "num_hi",
            format!("12{}34", ch(&[256])),
            " (NUMERIC,12\\u010034) (EOF,null)",
        ),
        (
            "num_lead",
            "1a2b".to_string(),
            " (NUMERIC,1) (ALPHABETICAL,a) (NUMERIC,2) (ALPHABETICAL,b) (EOF,null)",
        ),
        (
            "num_all",
            "abc".to_string(),
            " (ALPHABETICAL,abc) (EOF,null)",
        ),
        ("num_digits", "999".to_string(), " (NUMERIC,999) (EOF,null)"),
        ("empty", "".to_string(), " (EOF,null)"),
        (
            "ws_nl_ws",
            format!("a {} b", ch(&[10])),
            " (ALPHABETICAL,a) (WHITESPACE, ) (EOL,null) (WHITESPACE, ) (ALPHABETICAL,b) (EOF,null)",
        ),
    ];
    for (label, text, expected) in cases.iter() {
        assert_eq!(render(text, true), *expected, "case {}", label);
    }
}

#[test]
fn jvm_verified_primative_tokenizer_peek() {
    let mut tokenizer = PrimativeTokenizer::get_string_instance("ab cd", false);
    let _ = tokenizer.initialize();
    let peek1 = unsafe { Box::from_raw(tokenizer.peek(std::ptr::null_mut())) };
    let peek2 = unsafe { Box::from_raw(tokenizer.peek(std::ptr::null_mut())) };
    let next1 = unsafe { Box::from_raw(tokenizer.next(std::ptr::null_mut())) };
    let next2 = unsafe { Box::from_raw(tokenizer.next(std::ptr::null_mut())) };
    assert_eq!(peek1.get_string(), "(ALPHANUM,ab)");
    assert_eq!(peek2.get_string(), "(ALPHANUM,ab)");
    assert_eq!(next1.get_string(), "(ALPHANUM,ab)");
    assert_eq!(next2.get_string(), "(WHITESPACE, )");
    assert_eq!(
        tokenizer.get_symbols(),
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    );
}

#[test]
fn jvm_verified_token_api() {
    let mut t = Token::new();
    t.set_type_and_string(token::Type::Alphanum, "MiXeD");
    assert_eq!(t.to_string(), "ALPHANUM MiXeD");
    assert_eq!(t.get_string(), "(ALPHANUM,MiXeD)");
    assert_eq!(unsafe { t.get_key() }, "mixed");
    assert_eq!(t.length(), 5);
    assert_eq!(t.get_char(), 77);
    assert!(t.equals_string(Some("mixed")));
    assert!(t.equals_string(Some("MIXED")));
    assert!(!t.equals_string(None));

    let mut e = Token::new();
    e.set_type(token::Type::Eol);
    assert_eq!(e.to_string(), "EOL null");
    assert_eq!(e.get_string(), "(EOL)");
    assert_eq!(e.length(), 0);
    assert_eq!(e.get_char(), 32);
    assert_eq!(unsafe { e.get_key() }, " ");
    // `set(String)` on an EOL token assigns its own parameter, not the field.
    e.set_string("hello");
    assert_eq!(e.get_value(), None);
    assert_eq!(e.get_string(), "(EOL)");
    assert_eq!(unsafe { e.get_key() }, " ");

    let mut d = Token::new();
    d.set_type_and_double(token::Type::Alphanum, 1.0);
    assert_eq!(d.get_value(), Some("1.0"));
    d.set_type_and_double(token::Type::Alphanum, 1.0e7);
    assert_eq!(d.get_value(), Some("1.0E7"));
    d.set_type_and_double(token::Type::Alphanum, 0.0001);
    assert_eq!(d.get_value(), Some("1.0E-4"));

    let mut n = Token::new();
    n.set_type_and_string(token::Type::Alphanum, "aaabbb");
    assert_eq!(n.number_of(b'a' as u16, 0), 3);
    assert_eq!(n.number_of(b'b' as u16, 0), 3);
    // fromIndex is declared and never read; the loop always starts at 0.
    assert_eq!(n.number_of(b'b' as u16, 3), 3);

    assert_eq!(token::Type::Separator.get_descr(), ".");
    assert_eq!(token::Type::Open.get_descr(), "[");
    assert_eq!(token::Type::Close.get_descr(), "]");
    assert_eq!(token::Type::Subopen.get_descr(), "[[");
    assert_eq!(token::Type::Subclose.get_descr(), "]]");
    assert_eq!(token::Type::Alphabetic.to_string(), "ALPHABETICAL");
    assert_eq!(token::Type::Word.get_descr(), "WORD");

    let mut s1 = Token::new();
    s1.set_type_and_string(token::Type::Alphanum, "abcdef");
    let s2 = unsafe { Box::from_raw(s1.split(token::Type::Word, 1, 2)) };
    assert_eq!(s2.get_string(), "(WORD,bc)");
    assert_eq!(s1.get_string(), "(ALPHANUM,cdef)");

    let mut c = Token::new();
    c.set_type_and_char(token::Type::Symbol, b'.' as u16);
    assert_eq!(c.get_string(), "(SYMBOL,.)");
    assert!(c.equals_type_and_char(token::Type::Symbol, b'.' as u16));
    assert!(c.equals_type_and_char_list(token::Type::Symbol, Some(&[b'a' as u16, b'.' as u16])));

    let panic = std::panic::catch_unwind(|| {
        let mut bad = Token::new();
        bad.set_type_and_string(token::Type::Alphanum, "ab");
        unsafe { bad.split(token::Type::Word, 1, 2) };
    });
    let panic = *panic.unwrap_err().downcast::<String>().unwrap();
    assert_eq!(
        panic,
        "java.lang.IndexOutOfBoundsException: startIndex + size, 12, must be less then value.length,2."
    );
}

#[test]
fn jvm_verified_token_link_list() {
    let mut h = Token::new();
    h.set_type_and_string(token::Type::Alphanum, "one");
    let mut h2 = Token::new();
    h2.set_type_and_string(token::Type::Alphanum, "two");
    let mut h3 = Token::new();
    h3.set_type(token::Type::Eol);
    unsafe {
        h.set_next(&mut h2);
        h2.set_next(&mut h3);
        assert_eq!(h.get_values(), "onetwo ");
        assert_eq!(h.get_multi_line_values(), "onetwo\n");
        assert_eq!(h.get_key(), "onetwo ");
        let dropped = h2.drop_from_list();
        assert_eq!((*dropped).get_string(), "(ALPHANUM,one)");
        assert_eq!(h.get_values(), "one ");
    }
}

/// `Utilities.stripHTMLTags` is the first caller in the crate that drives
/// `PrimativeTokenizer` end to end; it was blocked on this module until now.  Expected
/// values captured from the reference JVM.
#[test]
fn jvm_verified_strip_html_tags() {
    let cases: Vec<(Option<&str>, Option<&str>)> = vec![
        (None, None),
        (Some(""), Some("")),
        (Some("plain text"), Some("plain text")),
        (Some("<html>bold</html>"), Some("bold")),
        (Some("a<b>c"), Some("ac")),
        (Some("a<>b"), Some("a<b")),
        (Some("a<<html>b"), Some("a<html>b")),
        (Some("<html><b>x</b> y"), Some("x y")),
        (Some("no close <tag"), Some("no close <tag")),
        (Some("<a href=\"x\">link</a>"), Some("link")),
        (Some("a > b"), Some("a > b")),
        (Some("5 < 6 > 7"), Some("5  7")),
        (Some("<html>line1<br>line2"), Some("line1line2")),
        (Some("<<>>"), Some("<>>")),
        (Some("<>"), Some("<")),
        (Some("<"), Some("<")),
        (Some(">"), Some(">")),
    ];
    for (input, expected) in cases.iter() {
        assert_eq!(
            imod_rs::imod::etomo::util::utilities::strip_html_tags(*input).as_deref(),
            *expected,
            "stripHTMLTags({:?})",
            input
        );
    }
}
