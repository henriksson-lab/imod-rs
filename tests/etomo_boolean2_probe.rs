//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/type/EtomoBoolean2.java` and
//! `IMOD/Etomo/src/etomo/type/ScriptParameter.java`.
//!
//! `jvm_verified_expectations` asserts values captured from a real JVM run; it always
//! runs.  `etomo_boolean2_probe` dumps the full comparison table used to capture them
//! and is skipped unless `IMOD_RS_ETOMO_PROBE` is set.  The Java harness is a class in
//! package `etomo.type` (both `paramString` and the constructors under test are
//! package-private), compiled against a reference runtime built the way the header of
//! `tests/etomo_utilities_probe.rs` describes, printing the same `label<TAB>value`
//! lines; an exception line is compared with the JDK class name removed, since a Rust
//! panic carries only the message.  `paramString` is package-private in Java and
//! `pub(crate)` here, so an integration test cannot reach it: its two captured lines are
//! asserted by the module's own `#[cfg(test)]` block in
//! `src/imod/etomo/type/script_parameter.rs`.  Two more harness artifacts are
//! normalized when diffing: the harness prints a null `toString()` - which this class
//! returns for an instance with no string equivalent - as `<null>`, where `Display`
//! renders the "null" `String.valueOf` produces; and `paramString`'s embedded newlines
//! make it span lines.
use imod_rs::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use imod_rs::imod::etomo::r#type::etomo_boolean2::{self, EtomoBoolean2};
use imod_rs::imod::etomo::r#type::etomo_number::EtomoNumber;
use imod_rs::imod::etomo::r#type::script_parameter::ScriptParameter;
use std::collections::{BTreeMap, HashMap};

fn p(label: &str, value: Option<String>) {
    println!(
        "{}\t{}",
        label,
        value.unwrap_or_else(|| "<null>".to_string())
    );
}

/// Java's `catch (RuntimeException e)`: run `body`, reporting a panic the way the Java
/// harness reports an unchecked exception.
fn caught<T>(body: impl FnOnce() -> T + std::panic::UnwindSafe) -> Result<T, String> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(body);
    std::panic::set_hook(previous);
    match result {
        Ok(value) => Ok(value),
        Err(payload) => Err(match payload.downcast_ref::<String>() {
            Some(message) => message.clone(),
            None => match payload.downcast_ref::<&str>() {
                Some(message) => message.to_string(),
                None => "<panic>".to_string(),
            },
        }),
    }
}

const VALUES: &[Option<&str>] = &[
    Some("true"),
    Some("false"),
    Some("TRUE"),
    Some("T"),
    Some("t"),
    Some("f"),
    Some("no"),
    Some("YES"),
    Some(" yes "),
    Some("1"),
    Some("0"),
    Some(""),
    Some("  "),
    None,
    Some("2"),
    Some("-1"),
    Some("abc"),
    Some("0.0"),
];

/// The Java harness's `setProbe(String, EtomoBoolean2, String)`.
fn set_probe(label: &str, b: EtomoBoolean2, value: Option<&str>) {
    let shown = value.unwrap_or("null");
    let outcome = caught(|| {
        let mut b = b;
        b.set_string(value);
        (
            b.to_string(),
            b.is(),
            b.is_null(),
            b.base.base.base.get_int(),
            b.base.base.base.get_invalid_reason(),
        )
    });
    match outcome {
        Err(message) => p(
            &format!("{}.set({})", label, shown),
            Some(format!("EXCEPTION {}", message)),
        ),
        Ok((string, is, is_null, get_int, invalid_reason)) => {
            p(&format!("{}.set({})", label, shown), Some(string));
            p(
                &format!("{}.set({}).is", label, shown),
                Some(is.to_string()),
            );
            p(
                &format!("{}.set({}).isNull", label, shown),
                Some(is_null.to_string()),
            );
            p(
                &format!("{}.set({}).getInt", label, shown),
                Some(get_int.to_string()),
            );
            p(
                &format!("{}.set({}).invalidReason", label, shown),
                Some(invalid_reason),
            );
        }
    }
}

#[test]
fn etomo_boolean2_probe() {
    if std::env::var("IMOD_RS_ETOMO_PROBE").is_err() {
        return;
    }
    p(
        "DEFAULT_FALSE_VALUE",
        Some(etomo_boolean2::DEFAULT_FALSE_VALUE.to_string()),
    );
    p(
        "DEFAULT_TRUE_VALUE",
        Some(etomo_boolean2::DEFAULT_TRUE_VALUE.to_string()),
    );
    let fresh = EtomoBoolean2::new();
    p("new.toString", Some(fresh.to_string()));
    p("new.is", Some(fresh.is().to_string()));
    p("new.isNull", Some(fresh.is_null().to_string()));
    p(
        "new.isUseInScript",
        Some(fresh.is_use_in_script().to_string()),
    );
    p("new.isActive", Some(fresh.base.is_active().to_string()));
    p(
        "new.getInt",
        Some(fresh.base.base.base.get_int().to_string()),
    );
    p(
        "new.isNotNullAndNotDefault",
        Some(fresh.base.is_not_null_and_not_default().to_string()),
    );
    let mut named = EtomoBoolean2::new_with_name("flag");
    p("named.toString", Some(named.to_string()));
    p(
        "named.equals(true)",
        Some(named.equals_boolean(true).to_string()),
    );
    p(
        "named.equals(false)",
        Some(named.equals_boolean(false).to_string()),
    );
    named.set_on();
    p("setOn.toString", Some(named.to_string()));
    p("setOn.is", Some(named.is().to_string()));
    p(
        "setOn.equals(true)",
        Some(named.equals_boolean(true).to_string()),
    );
    named.set_off();
    p("setOff.toString", Some(named.to_string()));
    p("setOff.is", Some(named.is().to_string()));
    for value in VALUES {
        set_probe("v", EtomoBoolean2::new_with_name("flag"), *value);
    }
    // displayAsInteger
    let mut dai = EtomoBoolean2::new_with_name("flag");
    dai.set_display_as_integer(true);
    dai.set_string(Some("true"));
    p("dai.toString", Some(dai.to_string()));
    dai.set_string(Some("false"));
    p("dai.false.toString", Some(dai.to_string()));
    // on/off values
    let onoff = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    p("onoff.toString", Some(onoff.to_string()));
    p("onoff.is", Some(onoff.is().to_string()));
    p("onoff.isNull", Some(onoff.is_null().to_string()));
    set_probe(
        "onoff",
        EtomoBoolean2::new_with_on_off_values("flag", 5, -1),
        Some("5"),
    );
    set_probe(
        "onoff",
        EtomoBoolean2::new_with_on_off_values("flag", 5, -1),
        Some("-1"),
    );
    set_probe(
        "onoff",
        EtomoBoolean2::new_with_on_off_values("flag", 5, -1),
        Some("true"),
    );
    set_probe(
        "onoff",
        EtomoBoolean2::new_with_on_off_values("flag", 5, -1),
        Some("0"),
    );
    let mut onoff2 = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    onoff2.set_on();
    p("onoff2.setOn.toString", Some(onoff2.to_string()));
    p(
        "onoff2.setOn.getInt",
        Some(onoff2.base.base.base.get_int().to_string()),
    );
    let mut onoff3 = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    onoff3.set_on();
    match caught(move || {
        let mut onoff3 = onoff3;
        onoff3.set_display_as_integer(false);
        onoff3
    }) {
        Err(message) => p(
            "onoff2.setDisplayAsInteger(false)",
            Some(format!("EXCEPTION {}", message)),
        ),
        Ok(_) => p("onoff2.setDisplayAsInteger(false)", Some("ok".to_string())),
    }
    onoff2.set_display_as_integer(true);
    p("onoff2.dai.toString", Some(onoff2.to_string()));
    // requiredMap
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert("flag".to_string(), "1".to_string());
    let map_for_ctor = map.clone();
    match caught(move || {
        let req = EtomoBoolean2::new_with_required_map("flag", Some(&map_for_ctor));
        (req.to_string(), req.is_null())
    }) {
        Err(message) => p("req.ctor", Some(format!("EXCEPTION {}", message))),
        Ok((string, is_null)) => {
            p("req.toString", Some(string));
            p("req.isNull", Some(is_null.to_string()));
        }
    }
    let mut map0: HashMap<String, String> = HashMap::new();
    map0.insert("flag".to_string(), "0".to_string());
    let map0_for_ctor = map0.clone();
    match caught(move || {
        let req0 = EtomoBoolean2::new_with_required_map("flag", Some(&map0_for_ctor));
        (req0.to_string(), req0.is_null())
    }) {
        Err(message) => p("req0.ctor", Some(format!("EXCEPTION {}", message))),
        Ok((string, is_null)) => {
            p("req0.toString", Some(string));
            p("req0.isNull", Some(is_null.to_string()));
            set_probe(
                "req0",
                EtomoBoolean2::new_with_required_map("flag", Some(&map0)),
                Some(""),
            );
        }
    }
    let mut map_other: HashMap<String, String> = HashMap::new();
    map_other.insert("other".to_string(), "1".to_string());
    match caught(move || EtomoBoolean2::new_with_required_map("flag", Some(&map_other)).to_string())
    {
        Err(message) => p("reqo.ctor", Some(format!("EXCEPTION {}", message))),
        Ok(string) => p("reqo.toString", Some(string)),
    }
    // store / load / remove
    let mut props: BTreeMap<String, String> = BTreeMap::new();
    let mut stored = EtomoBoolean2::new_with_name("flag");
    stored.set_boolean(true);
    EtomoBoolean2::store_instance(Some(&stored), &mut props, Some("a.b"), "flag");
    p("store.a.b.flag", props.get("a.b.flag").cloned());
    EtomoBoolean2::store_instance(None, &mut props, Some("a.b"), "flag");
    p("store(null).a.b.flag", props.get("a.b.flag").cloned());
    let mut props2: BTreeMap<String, String> = BTreeMap::new();
    props2.insert("a.b.flag".to_string(), "true".to_string());
    props2.insert("flag".to_string(), "false".to_string());
    props2.insert("c.flag".to_string(), "yes".to_string());
    p(
        "load(null,flag,props2,a.b)",
        EtomoBoolean2::load_instance(None, "flag", &props2, Some("a.b"))
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "load(null,flag,props2,null)",
        EtomoBoolean2::load_instance(None, "flag", &props2, None).and_then(|x| Some(x.to_string())),
    );
    p(
        "load(null,flag,props2,'')",
        EtomoBoolean2::load_instance(None, "flag", &props2, Some(""))
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "load(null,flag,props2,c.)",
        EtomoBoolean2::load_instance(None, "flag", &props2, Some("c."))
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "load(null,flag,props2,missing)",
        EtomoBoolean2::load_instance(None, "flag", &props2, Some("zz"))
            .and_then(|x| Some(x.to_string())),
    );
    let existing = EtomoBoolean2::new_with_name("flag");
    p(
        "load(existing,flag,props2,a.b)",
        EtomoBoolean2::load_instance(Some(existing), "flag", &props2, Some("a.b"))
            .and_then(|x| Some(x.to_string())),
    );
    let mut props3: BTreeMap<String, String> = BTreeMap::new();
    props3.insert("a.b.flag".to_string(), "true".to_string());
    EtomoBoolean2::remove("flag", &mut props3, Some("a.b"));
    p("remove.a.b.flag", props3.get("a.b.flag").cloned());
    props3.insert("flag".to_string(), "true".to_string());
    EtomoBoolean2::remove("flag", &mut props3, None);
    p("remove(null).flag", props3.get("flag").cloned());
    // getInstance
    p(
        "getInstance(null,flag,props2,a.b)",
        EtomoBoolean2::get_instance_from_props(None, "flag", &props2, Some("a.b"))
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "getInstance(null,flag,props2,zz)",
        EtomoBoolean2::get_instance_from_props(None, "flag", &props2, Some("zz"))
            .and_then(|x| Some(x.to_string())),
    );
    let gi = EtomoBoolean2::new_with_name("flag");
    p(
        "getInstance(gi,flag,props2,a.b)",
        EtomoBoolean2::get_instance_from_props(Some(gi), "flag", &props2, Some("a.b"))
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "getInstance(null,flag,true)",
        EtomoBoolean2::get_instance_from_boolean(None, "flag", true)
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "getInstance(null,flag,false)",
        EtomoBoolean2::get_instance_from_boolean(None, "flag", false)
            .and_then(|x| Some(x.to_string())),
    );
    // static equals / set
    let mut e1 = EtomoBoolean2::new_with_name("flag");
    let e2 = EtomoBoolean2::new_with_name("flag");
    p(
        "equals(e1,e1)",
        Some(EtomoBoolean2::equals_instances(Some(&e1), Some(&e1)).to_string()),
    );
    p(
        "equals(e1,e2)",
        Some(EtomoBoolean2::equals_instances(Some(&e1), Some(&e2)).to_string()),
    );
    p(
        "equals(null,null)",
        Some(EtomoBoolean2::equals_instances(None, None).to_string()),
    );
    p(
        "equals(null,e2)",
        Some(EtomoBoolean2::equals_instances(None, Some(&e2)).to_string()),
    );
    p(
        "equals(e1,null)",
        Some(EtomoBoolean2::equals_instances(Some(&e1), None).to_string()),
    );
    e1.set_boolean(true);
    p(
        "equals(e1set,e2)",
        Some(EtomoBoolean2::equals_instances(Some(&e1), Some(&e2)).to_string()),
    );
    p(
        "set(null,true,flag)",
        EtomoBoolean2::set_instance_boolean(None, true, "flag").and_then(|x| Some(x.to_string())),
    );
    p(
        "set(null,false,flag)",
        EtomoBoolean2::set_instance_boolean(None, false, "flag").and_then(|x| Some(x.to_string())),
    );
    let mut n = EtomoNumber::new_with_name("flag");
    n.set_int(1);
    p(
        "set(null,n,flag)",
        EtomoBoolean2::set_instance_const_etomo_number(None, Some(&n.base), "flag")
            .and_then(|x| Some(x.to_string())),
    );
    p(
        "set(null,(ConstEtomoNumber)null,flag)",
        EtomoBoolean2::set_instance_const_etomo_number(None, None, "flag")
            .and_then(|x| Some(x.to_string())),
    );
    // store to properties through the instance
    let mut props4: BTreeMap<String, String> = BTreeMap::new();
    let mut si = EtomoBoolean2::new_with_name("flag");
    si.set_boolean(true);
    si.store_with_prepend(&mut props4, Some("a.b"));
    p("si.store.a.b.flag", props4.get("a.b.flag").cloned());
    si.set_boolean(false);
    si.store_with_prepend(&mut props4, Some("a.b"));
    p("si.store.false.a.b.flag", props4.get("a.b.flag").cloned());
    // ScriptParameter
    let mut sp = ScriptParameter::new_with_type_and_name(Type::Integer, "thickness");
    p("sp.toString", Some(sp.to_string()));
    p("sp.isActive", Some(sp.is_active().to_string()));
    p(
        "sp.isNotNullAndNotDefault",
        Some(sp.is_not_null_and_not_default().to_string()),
    );
    sp.base.set_int(100);
    p(
        "sp.set100.isNotNullAndNotDefault",
        Some(sp.is_not_null_and_not_default().to_string()),
    );
    sp.set_active(false);
    p(
        "sp.setActive(false).isActive",
        Some(sp.is_active().to_string()),
    );
    sp.set_active(true);
    let mut map2: HashMap<String, String> = HashMap::new();
    map2.insert("thickness".to_string(), "1".to_string());
    let mut sp4 = ScriptParameter::new_with_required_map(Type::Integer, "thickness", Some(&map2));
    p(
        "sp4.invalidReason",
        Some(sp4.base.base.get_invalid_reason()),
    );
    sp4.base.set_string(Some(""));
    p(
        "sp4.set('').invalidReason",
        Some(sp4.base.base.get_invalid_reason()),
    );
    let x = EtomoNumber::new_with_name("x");
    let sp5 = ScriptParameter::new_from_instance(Some(&x.base));
    p("sp5.toString", Some(sp5.to_string()));
}

/// Values captured byte-for-byte from the reference JVM harness described above.  The
/// whole table matched; these are the lines that pin the source's behaviour, including
/// the four that only a real JVM would have shown.
#[test]
fn jvm_verified_expectations() {
    assert_eq!(etomo_boolean2::DEFAULT_FALSE_VALUE, 0);
    assert_eq!(etomo_boolean2::DEFAULT_TRUE_VALUE, 1);
    // A fresh instance is false, never null, and always in the script.
    let fresh = EtomoBoolean2::new();
    assert_eq!(fresh.to_string(), "false");
    assert!(!fresh.is());
    assert!(!fresh.is_null());
    assert!(fresh.is_use_in_script());
    assert!(fresh.base.is_active());
    // `isNotNullAndNotDefault` is true for a fresh instance: defaultValue is null, so
    // `isDefault(currentValue)` is false.
    assert!(fresh.base.is_not_null_and_not_default());
    // The string table: the two full words, the two single letters, "no"/"yes", and the
    // integers, all case-folded and trimmed.
    for (value, expected) in [
        ("true", "true"),
        ("false", "false"),
        ("TRUE", "true"),
        ("T", "true"),
        ("t", "true"),
        ("f", "false"),
        ("no", "false"),
        ("YES", "true"),
        (" yes ", "true"),
        ("1", "true"),
        ("0", "false"),
        ("", "false"),
        ("  ", "false"),
    ] {
        let mut b = EtomoBoolean2::new_with_name("flag");
        b.set_string(Some(value));
        assert_eq!(b.to_string(), expected, "set({:?})", value);
    }
    let mut null_set = EtomoBoolean2::new_with_name("flag");
    null_set.set_string(None);
    assert_eq!(null_set.to_string(), "false");
    // An unparseable string leaves the value null - which displays as the false display
    // value - and records the reason rather than throwing.
    let mut bad = EtomoBoolean2::new_with_name("flag");
    bad.set_string(Some("abc"));
    assert_eq!(bad.to_string(), "false");
    assert_eq!(
        bad.base.base.base.get_invalid_reason(),
        "abc is not a valid Integer.  For input string: \"abc\""
    );
    // A parseable value outside validValues throws, and the message prints the offending
    // value through this class's own toString(Number) - so 2 reads as "false".
    let out_of_range = std::panic::catch_unwind(|| {
        let mut b = EtomoBoolean2::new_with_name("flag");
        b.set_string(Some("2"));
    });
    let message = *out_of_range.unwrap_err().downcast::<String>().unwrap();
    assert_eq!(
        message,
        "false is not a valid value.\nValid values are false,true."
    );
    // displayAsInteger prints through the superclass toString.
    let mut dai = EtomoBoolean2::new_with_name("flag");
    dai.set_display_as_integer(true);
    dai.set_string(Some("true"));
    assert_eq!(dai.to_string(), "1");
    dai.set_string(Some("false"));
    assert_eq!(dai.to_string(), "0");
    // `EtomoBoolean2(String, int, int)` nulls all four string tables, so toString()
    // returns null - "null" through String.valueOf - and `is()` is true for a fresh
    // instance because the display value is the *off* value, -1, which is not zero.
    let onoff = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    assert_eq!(onoff.to_string(), "null");
    assert!(onoff.is());
    assert!(!onoff.is_null());
    let mut on = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    on.set_string(Some("5"));
    assert_eq!(on.base.base.base.get_int(), 5);
    let mut off = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    off.set_string(Some("-1"));
    assert_eq!(off.base.base.base.get_int(), -1);
    // "true" is not in that instance's tables and does not parse as an integer, so the
    // value stays null and getInt falls back to the display value.
    let mut word = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
    word.set_string(Some("true"));
    assert_eq!(word.base.base.base.get_int(), -1);
    assert_eq!(
        word.base.base.base.get_invalid_reason(),
        "true is not a valid Integer.  For input string: \"true\""
    );
    // Setting such an instance out of range reaches `toString(Vector)`, whose
    // `new StringBuffer(toString(validValues.get(0)))` is handed this class's null: the
    // reference throws a NullPointerException there, before the IllegalArgumentException
    // the override would have thrown.
    let npe = std::panic::catch_unwind(|| {
        let mut b = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
        b.set_string(Some("0"));
    });
    assert!(npe.is_err());
    // setDisplayAsInteger(false) on an instance with no string equivalent throws.
    let illegal_state = std::panic::catch_unwind(|| {
        let mut b = EtomoBoolean2::new_with_on_off_values("flag", 5, -1);
        b.set_display_as_integer(false);
    });
    let message = *illegal_state.unwrap_err().downcast::<String>().unwrap();
    assert_eq!(
        message,
        "Must display flagas an integer, since it has no string equivalent."
    );
    // A requiredMap that marks the field required makes `nullIsValid` false, and the
    // constructor's own `setValidValues` then throws through the override before the
    // instance exists.
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert("flag".to_string(), "1".to_string());
    let required = std::panic::catch_unwind(|| {
        let mut map: HashMap<String, String> = HashMap::new();
        map.insert("flag".to_string(), "1".to_string());
        EtomoBoolean2::new_with_required_map("flag", Some(&map));
    });
    let message = *required.unwrap_err().downcast::<String>().unwrap();
    assert_eq!(message, "This field cannot be empty.");
    // A map that does not name this field, or names it "0", constructs normally.
    let mut map0: HashMap<String, String> = HashMap::new();
    map0.insert("flag".to_string(), "0".to_string());
    assert_eq!(
        EtomoBoolean2::new_with_required_map("flag", Some(&map0)).to_string(),
        "false"
    );
    let mut other: HashMap<String, String> = HashMap::new();
    other.insert("other".to_string(), "1".to_string());
    assert_eq!(
        EtomoBoolean2::new_with_required_map("flag", Some(&other)).to_string(),
        "false"
    );
    // store/load/remove through Properties, including the null and dotted prepends.
    let mut props: BTreeMap<String, String> = BTreeMap::new();
    let mut stored = EtomoBoolean2::new_with_name("flag");
    stored.set_boolean(true);
    EtomoBoolean2::store_instance(Some(&stored), &mut props, Some("a.b"), "flag");
    assert_eq!(props.get("a.b.flag").map(|x| x.as_str()), Some("true"));
    EtomoBoolean2::store_instance(None, &mut props, Some("a.b"), "flag");
    assert!(props.get("a.b.flag").is_none());
    let mut props2: BTreeMap<String, String> = BTreeMap::new();
    props2.insert("a.b.flag".to_string(), "true".to_string());
    props2.insert("flag".to_string(), "false".to_string());
    props2.insert("c.flag".to_string(), "yes".to_string());
    assert_eq!(
        EtomoBoolean2::load_instance(None, "flag", &props2, Some("a.b"))
            .and_then(|x| Some(x.to_string())),
        Some("true".to_string())
    );
    assert_eq!(
        EtomoBoolean2::load_instance(None, "flag", &props2, None).and_then(|x| Some(x.to_string())),
        Some("false".to_string())
    );
    assert_eq!(
        EtomoBoolean2::load_instance(None, "flag", &props2, Some("c."))
            .and_then(|x| Some(x.to_string())),
        Some("true".to_string())
    );
    assert!(EtomoBoolean2::load_instance(None, "flag", &props2, Some("zz")).is_none());
    assert_eq!(
        EtomoBoolean2::get_instance_from_props(None, "flag", &props2, Some("a.b"))
            .and_then(|x| Some(x.to_string())),
        Some("true".to_string())
    );
    assert!(EtomoBoolean2::get_instance_from_props(None, "flag", &props2, Some("zz")).is_none());
    // The static equals: two fresh instances are equal, and reference identity settles
    // the two null cases.
    let mut e1 = EtomoBoolean2::new_with_name("flag");
    let e2 = EtomoBoolean2::new_with_name("flag");
    assert!(EtomoBoolean2::equals_instances(Some(&e1), Some(&e2)));
    assert!(EtomoBoolean2::equals_instances(None, None));
    assert!(!EtomoBoolean2::equals_instances(None, Some(&e2)));
    assert!(!EtomoBoolean2::equals_instances(Some(&e1), None));
    e1.set_boolean(true);
    assert!(!EtomoBoolean2::equals_instances(Some(&e1), Some(&e2)));
    // The static set() helpers, including the one that returns null.
    assert_eq!(
        EtomoBoolean2::set_instance_boolean(None, true, "flag").and_then(|x| Some(x.to_string())),
        Some("true".to_string())
    );
    let mut n = EtomoNumber::new_with_name("flag");
    n.set_int(1);
    assert_eq!(
        EtomoBoolean2::set_instance_const_etomo_number(None, Some(&n.base), "flag")
            .and_then(|x| Some(x.to_string())),
        Some("true".to_string())
    );
    assert!(
        EtomoBoolean2::set_instance_const_etomo_number(None, None::<&ConstEtomoNumber>, "flag")
            .is_none()
    );
    // ScriptParameter: a fresh instance is active, prints empty, and its paramString
    // reports a null `active`.
    let mut sp = ScriptParameter::new_with_type_and_name(Type::Integer, "thickness");
    assert_eq!(sp.to_string(), "");
    assert!(sp.is_active());
    assert!(!sp.is_not_null_and_not_default());
    sp.base.set_int(100);
    assert!(sp.is_not_null_and_not_default());
    sp.set_active(false);
    assert!(!sp.is_active());
    // `ScriptParameter(Type, String, HashMap)` only sets nullIsValid; unlike
    // EtomoBoolean2's constructor it calls no setValidValues, so it does not throw.
    let mut map2: HashMap<String, String> = HashMap::new();
    map2.insert("thickness".to_string(), "1".to_string());
    let mut sp4 = ScriptParameter::new_with_required_map(Type::Integer, "thickness", Some(&map2));
    assert_eq!(sp4.base.base.get_invalid_reason(), "");
    // `EtomoNumber.set(String)` validates only in its non-blank branch, so setting a
    // required field to "" records no reason at all.
    sp4.base.set_string(Some(""));
    assert_eq!(sp4.base.base.get_invalid_reason(), "");
}
