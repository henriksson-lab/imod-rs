//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/comscript/ComScriptCommand.java` and
//! `IMOD/Etomo/src/etomo/comscript/ComScriptInputArg.java`.
//!
//! Every expectation below was printed by a harness compiled into `etomo.comscript`
//! (the class and most of its members are package-private) and run against a reference
//! runtime built the way the header of `tests/etomo_utilities_probe.rs` describes.
//!
//! Two source behaviours are pinned here on purpose: `containsOption` can never match a
//! command-line argument, because the source compares the `String[]` object rather than
//! its element against the option; and the copy constructor shares its
//! `ComScriptInputArg` objects with the original, so a later `setValue` on one is
//! visible through the other.
use imod_rs::imod::etomo::comscript::com_script_command::ComScriptCommand;
use imod_rs::imod::etomo::comscript::com_script_input_arg::ComScriptInputArg;

fn v(values: &[&str]) -> Vec<Option<String>> {
    values.iter().map(|s| Some(s.to_string())).collect()
}

fn arr(values: Option<Vec<Option<String>>>) -> String {
    match values {
        None => "null".to_string(),
        Some(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|value| value.as_deref().unwrap_or("null").to_string())
                .collect::<Vec<String>>()
                .join(",")
        ),
    }
}

#[test]
fn jvm_verified_com_script_command() {
    let mut c = ComScriptCommand::new(false, false);
    c.set_command(Some("tilt"));
    c.set_command_line_args(&v(&["-StandardInput"]));
    assert!(c.is_keyword_value_pairs());
    assert_eq!(arr(c.get_command_line_args()), "{-StandardInput}");
    assert_eq!(c.get_command_line_length(), 1);

    c.set_value(Some("InputProjections"), Some("BBa.ali"));
    c.set_value(Some("THICKNESS"), Some("100"));
    c.add_key(Some("SHIFT"), Some("0.0 -3.5"));
    assert_eq!(
        c.to_string(),
        "[tilt [[InputProjections\tBBa.ali], [THICKNESS\t100], [SHIFT\t0.0 -3.5]]]"
    );
    assert_eq!(
        c.get_value(Some("THICKNESS")).unwrap(),
        Some("100".to_string())
    );
    assert_eq!(
        c.get_value(Some("SHIFT")).unwrap(),
        Some("0.0 -3.5".to_string())
    );
    assert_eq!(c.get_value(Some("missing")).unwrap(), Some(String::new()));
    assert_eq!(c.get_value(None).unwrap(), Some(String::new()));
    assert!(c.has_keyword(Some("THICKNESS")).unwrap());
    assert!(!c.has_keyword(Some("thickness")).unwrap());
    assert_eq!(c.get_values(Some("THICKNESS")), v(&["100"]));
    assert!(c.contains_option(Some("THICKNESS")));
    // The command-line loop can never match; see the module header.
    assert!(!c.contains_option(Some("-StandardInput")));

    c.set_values(Some("MULTI"), &v(&["a", "b", "c"]));
    assert_eq!(
        c.to_string(),
        "[tilt [[InputProjections\tBBa.ali], [THICKNESS\t100], [SHIFT\t0.0 -3.5], \
         [MULTI\ta], [MULTI\tb], [MULTI\tc]]]"
    );
    assert_eq!(c.get_values(Some("MULTI")), v(&["a", "b", "c"]));
    assert!(c.delete_key(Some("MULTI")));
    assert_eq!(
        c.to_string(),
        "[tilt [[InputProjections\tBBa.ali], [THICKNESS\t100], [SHIFT\t0.0 -3.5], \
         [MULTI\tb], [MULTI\tc]]]"
    );
    c.delete_key_all(Some("MULTI"));
    assert_eq!(
        c.to_string(),
        "[tilt [[InputProjections\tBBa.ali], [THICKNESS\t100], [SHIFT\t0.0 -3.5]]]"
    );

    // The copy constructor shares the input args with the original.
    let copy = ComScriptCommand::new_from(&c);
    assert_eq!(
        copy.to_string(),
        "[tilt [[InputProjections\tBBa.ali], [THICKNESS\t100], [SHIFT\t0.0 -3.5]]]"
    );
    assert!(copy.is_keyword_value_pairs());
    c.set_value(Some("THICKNESS"), Some("999"));
    assert_eq!(
        copy.to_string(),
        "[tilt [[InputProjections\tBBa.ali], [THICKNESS\t999], [SHIFT\t0.0 -3.5]]]"
    );
}

#[test]
fn jvm_verified_com_script_command_case_insensitive() {
    let mut ci = ComScriptCommand::new(true, true);
    ci.set_command(Some("tilt"));
    ci.set_command_line_args(&v(&["-standardinput"]));
    assert!(ci.is_keyword_value_pairs());
    ci.set_value(Some("THICKNESS"), Some("100"));
    assert_eq!(ci.to_string(), "[tilt [[THICKNESS 100]]]");
    assert_eq!(
        ci.get_value(Some("thickness")).unwrap(),
        Some("100".to_string())
    );
    assert!(ci.has_keyword(Some("thickness")).unwrap());
}

#[test]
fn jvm_verified_com_script_command_non_keyword() {
    let mut nk = ComScriptCommand::new(false, false);
    nk.set_command(Some("copytomocoms"));
    nk.set_command_line_args(&v(&["-name", "BBa"]));
    let error = nk.has_keyword(Some("x")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "etomo.comscript.InvalidParameterException: Command copytomocoms does not use \
         keyword/value pairs"
    );
    assert!(!nk.contains_option(Some("-name")));

    nk.set_header_comments(&v(&["# one", "# two"]));
    nk.append_header_comments(&v(&["# three"]));
    assert_eq!(arr(nk.get_header_comments()), "{# one,# two,# three}");
    nk.append_command_line_args(&v(&["-extra"]));
    assert_eq!(arr(nk.get_command_line_args()), "{-name,BBa,-extra}");

    nk.use_keyword_value();
    assert_eq!(arr(nk.get_command_line_args()), "{-StandardInput}");
    assert!(nk.is_keyword_value_pairs());
    assert_eq!(nk.to_string(), "[copytomocoms []]");
}

#[test]
fn jvm_verified_com_script_command_set_values_interleaved() {
    let mut iv = ComScriptCommand::new(false, true);
    iv.set_command(Some("x"));
    iv.use_keyword_value();
    let a0 = Some(v(&["1", "2"]));
    let a1 = Some(v(&["p", "q", "r"]));
    iv.set_values_interleaved(Some(&v(&["K1", "K2"])), Some(&[a0, a1]));
    assert_eq!(
        iv.to_string(),
        "[x [[K1 1], [K2 p], [K1 2], [K2 q], [K2 r]]]"
    );
}

#[test]
fn jvm_verified_com_script_input_arg() {
    let mut ia = ComScriptInputArg::new();
    ia.set_argument_parse_comments(Some("KEY value  # trailing"), true);
    assert_eq!(ia.get_argument(), Some("KEY"));
    assert_eq!(ia.get_comments(), v(&["# value  # trailing"]));

    let mut ib = ComScriptInputArg::new();
    ib.set_argument_parse_comments(Some("KEY  no hash"), true);
    assert_eq!(ib.get_argument(), Some("KEY"));
    assert_eq!(ib.get_comments(), v(&["# no hash"]));

    let mut ic = ComScriptInputArg::new();
    ic.set_argument_double(1.0);
    assert_eq!(ic.get_argument(), Some("1.0"));
    ic.set_argument_int(12);
    assert_eq!(ic.get_argument(), Some("12"));
    ic.set_argument_boolean(true);
    assert_eq!(ic.get_argument(), Some("1"));
    ic.set_argument_double(0.0001);
    assert_eq!(ic.get_argument(), Some("1.0E-4"));
    assert_eq!(ic.to_string(), "[1.0E-4]");
}
