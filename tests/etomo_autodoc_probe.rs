//! Differential check against the reference eTomo JVM for the
//! `IMOD/Etomo/src/etomo/storage/autodoc` package.
//!
//! The expected text below is the byte-for-byte output of a Java harness run against a
//! reference runtime built the way the header of `tests/etomo_utilities_probe.rs`
//! describes (`javac -nowarn -encoding ISO-8859-1` over every `.java` under
//! `IMOD/Etomo/src` except the test classes, then `java -Djava.awt.headless=true`).
//! The harness lives in `etomo.storage.autodoc` itself, so it reaches the package's
//! package-private constructors and methods directly, and uses reflection only for the
//! two private statics `AutodocFactory.stripFileExtension` and
//! `AutodocFactory.setInstance`/`getExistingAutodoc`.
//!
//! The probe below issues the same calls in the same order and renders them the same
//! way.  It covers the whole programmatic half of the package: the `Autodoc`
//! constructor, statement building (`addComment`, `addEmptyLine`,
//! `addNameValuePairAttribute` at one and at three levels), section building
//! (`addSection`, subsections), the statement walk, the section walk **in file order
//! across collection types** and by type, the attribute tree, `setValue`, the removal
//! paths, `wrapAttributeValues`, `printStatementList`, `printStoredData` and the
//! `AutodocFactory` registry and file-name statics.
//!
//! What it cannot cover is the file half: `AutodocTokenizer`'s constructor needs a
//! `LogFile.Handle`, whose constructor takes an `EmergencyMonitor`, so `AutodocParser`
//! and `Autodoc`'s `initialize...` members are unreachable on both a translated and a
//! frontier basis.  See those modules' headers.
use imod_rs::imod::etomo::storage::autodoc::attribute::{self, Attribute};
use imod_rs::imod::etomo::storage::autodoc::attribute_list::AttributeList;
use imod_rs::imod::etomo::storage::autodoc::autodoc::Autodoc;
use imod_rs::imod::etomo::storage::autodoc::autodoc_factory::{self, extension};
use imod_rs::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use imod_rs::imod::etomo::storage::autodoc::read_only_attribute_list::ReadOnlyAttributeList;
use imod_rs::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use imod_rs::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use imod_rs::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use imod_rs::imod::etomo::storage::autodoc::read_only_statement::ReadOnlyStatement;
use imod_rs::imod::etomo::storage::autodoc::read_only_statement_list::ReadOnlyStatementList;
use imod_rs::imod::etomo::storage::autodoc::section::{self, Section};
use imod_rs::imod::etomo::storage::autodoc::statement::Statement;
use imod_rs::imod::etomo::storage::autodoc::statement_location::StatementLocation;
use imod_rs::imod::etomo::storage::autodoc::writable_attribute::WritableAttribute;
use imod_rs::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;
use imod_rs::imod::etomo::storage::autodoc::write_only_attribute_list::WriteOnlyAttributeList;
use imod_rs::imod::etomo::storage::autodoc::write_only_statement_list::WriteOnlyStatementList;
use imod_rs::imod::etomo::ui::swing::token::{self, Token};

/// The harness's `s`.
fn s(value: Option<String>) -> String {
    match value {
        None => "<null>".to_string(),
        Some(value) => value,
    }
}

/// The harness's `tok`.
fn tok(value: &str) -> *mut Token {
    let t = Box::into_raw(Box::new(Token::new()));
    unsafe { (*t).set_type_and_string(token::Type::Anything, value) };
    t
}

/// The harness's `dumpStatements`.
unsafe fn dump_statements(out: &mut String, list: &dyn ReadOnlyStatementList, prefix: &str) {
    let mut location: Option<StatementLocation> = list.get_statement_location();
    let mut statement = unsafe { list.next_statement(location.as_mut()) };
    let mut i = 0;
    while !statement.is_null() {
        let st: &dyn Statement = unsafe { &*statement };
        out.push_str(&format!(
            "{}stmt[{}] type={} line={} size={} left={} left0={} left1={} right={} string=[{}]\n",
            prefix,
            i,
            st.get_type(),
            st.get_line_num(),
            st.size_left_side(),
            s(st.get_left_side()),
            s(st.get_left_side_at(0)),
            s(st.get_left_side_at(1)),
            s(st.get_right_side()),
            st.get_string()
        ));
        let sub = st.get_subsection();
        if !sub.is_null() {
            out.push_str(&format!(
                "{}  subsection type={} name={}\n",
                prefix,
                unsafe { ReadOnlySection::get_type(&*sub) },
                s(unsafe { ReadOnlyStatementList::get_name(&*sub) })
            ));
            unsafe { dump_statements(out, &*sub, &format!("{}    ", prefix)) };
        }
        statement = unsafe { list.next_statement(location.as_mut()) };
        i += 1;
    }
    out.push_str(&format!("{}statement count={}\n", prefix, i));
}

/// The harness's `dumpAttributes`.
unsafe fn dump_attributes(out: &mut String, list: *mut AttributeList, prefix: &str) {
    if list.is_null() {
        out.push_str(&format!("{}attributes=<null>\n", prefix));
        return;
    }
    let mut iterator = unsafe { (*list).iterator() };
    while iterator.has_next() {
        let a: *mut Attribute = *iterator.next().unwrap();
        out.push_str(&format!(
            "{}attr name={} key={} line={} exists={} global={} base={} value={} multi={}\n",
            prefix,
            unsafe { ReadOnlyAttribute::get_name(&*a) },
            unsafe { (*a).get_key() },
            unsafe { ReadOnlyAttribute::get_line_num(&*a) },
            unsafe { (*a).exists() },
            unsafe { WriteOnlyAttributeList::is_global(&*a) },
            unsafe { (*a).is_base() },
            s(unsafe { ReadOnlyAttribute::get_value(&*a) }),
            s(unsafe { ReadOnlyAttribute::get_multi_line_value(&*a) })
        ));
        unsafe {
            dump_attributes(
                out,
                ReadOnlyAttribute::get_children(&*a),
                &format!("{}  ", prefix),
            )
        };
    }
}

/// The harness's `dumpSections`.
unsafe fn dump_sections(
    out: &mut String,
    list: &dyn ReadOnlySectionList,
    prefix: &str,
    r#type: Option<&str>,
) {
    let mut location = match r#type {
        None => list.get_section_location(),
        Some(r#type) => unsafe { list.get_section_location_by_type(Some(r#type)) },
    };
    out.push_str(&format!(
        "{}sectionLocation({})={}\n",
        prefix,
        r#type.unwrap_or("<null>"),
        match &location {
            None => "<null>".to_string(),
            Some(location) => location.to_string(),
        }
    ));
    let mut section = unsafe { list.next_section(location.as_mut()) };
    let mut i = 0;
    while !section.is_null() {
        out.push_str(&format!(
            "{}section[{}] type={} name={} string=[{}] loc={}\n",
            prefix,
            i,
            unsafe { ReadOnlySection::get_type(&*section) },
            s(unsafe { ReadOnlyStatementList::get_name(&*section) }),
            unsafe { ReadOnlyStatementList::get_string(&*section) },
            location.as_ref().unwrap().to_string()
        ));
        section = unsafe { list.next_section(location.as_mut()) };
        i += 1;
    }
    out.push_str(&format!("{}section count={}\n", prefix, i));
}

#[test]
fn jvm_verified_autodoc_package() {
    let mut out = String::new();
    unsafe {
        out.push_str("== statics ==\n");
        out.push_str(&format!(
            "Section.getKey(Token,Token)={}\n",
            s(section::get_key_of_tokens(
                tok("Field").as_ref(),
                tok("Name").as_ref(),
            ))
        ));
        out.push_str(&format!(
            "Section.getKey(Token,null)={}\n",
            s(section::get_key_of_tokens(tok("Field").as_ref(), None))
        ));
        out.push_str(&format!(
            "Section.getKey(null,Token)={}\n",
            s(section::get_key_of_tokens(None, tok("Name").as_ref()))
        ));
        out.push_str(&format!(
            "Section.getKey(null,null)={}\n",
            s(section::get_key_of_tokens(None, None))
        ));
        out.push_str(&format!(
            "Section.getKey(String,String)={}\n",
            s(section::get_key_of_strings(Some("FiElD"), Some("NaMe")))
        ));
        out.push_str(&format!(
            "Section.getKey(String,null)={}\n",
            s(section::get_key_of_strings(Some("FiElD"), None))
        ));
        out.push_str(&format!(
            "Section.getKey(null,String)={}\n",
            s(section::get_key_of_strings(None, Some("NaMe")))
        ));
        out.push_str(&format!(
            "Section.getKey(null,null)={}\n",
            s(section::get_key_of_strings(None, None))
        ));
        out.push_str(&format!(
            "Attribute.getKey(Token)={}\n",
            s(attribute::get_key_of_token(tok("AbC").as_ref()))
        ));
        out.push_str(&format!(
            "Attribute.getKey((Token)null)={}\n",
            s(attribute::get_key_of_token(None))
        ));
        out.push_str(&format!(
            "Attribute.getKey(String)={}\n",
            s(attribute::get_key_of_string(Some("AbC")))
        ));
        out.push_str(&format!(
            "Attribute.getKey((String)null)={}\n",
            s(attribute::get_key_of_string(None))
        ));
        for (label, input) in [
            ("etomo.adoc", "etomo.adoc"),
            ("etomo", "etomo"),
            ("a.b.c", "a.b.c"),
            (".adoc", ".adoc"),
            ("", ""),
        ] {
            out.push_str(&format!(
                "stripFileExtension({})={}\n",
                label,
                autodoc_factory::strip_file_extension(input)
            ));
        }
        for input in ["x.adoc", "x.prm", "x.txt"] {
            out.push_str(&format!(
                "endsWithAutodocExtension({})={}\n",
                input,
                autodoc_factory::ends_with_autodoc_extension(Some(input))
            ));
        }
        out.push_str(&format!(
            "endsWithAutodocExtension(null)={}\n",
            autodoc_factory::ends_with_autodoc_extension(None)
        ));
        out.push_str(&format!(
            "Extension.DEFAULT={} str={}\n",
            extension::DEFAULT,
            extension::DEFAULT.get_extension_string()
        ));
        out.push_str(&format!(
            "replacementDir={}\n",
            s(autodoc_factory::get_replacement_dir())
        ));
        autodoc_factory::set_replacement_dir(Some("/tmp/repl"));
        out.push_str(&format!(
            "replacementDir={}\n",
            s(autodoc_factory::get_replacement_dir())
        ));
        out.push_str(&format!(
            "isLoaded(etomo)={}\n",
            autodoc_factory::is_loaded(autodoc_factory::ETOMO)
        ));
        out.push_str(&format!(
            "isLoaded(tilt)={}\n",
            autodoc_factory::is_loaded(autodoc_factory::TILT)
        ));
        let reg = Autodoc::new(Some("reg"), std::ptr::null_mut());
        out.push_str(&format!(
            "setInstance(tilt)={}\n",
            autodoc_factory::set_instance(autodoc_factory::TILT, reg)
        ));
        out.push_str(&format!(
            "setInstance(bogus)={}\n",
            autodoc_factory::set_instance("bogus", reg)
        ));
        out.push_str(&format!(
            "isLoaded(tilt)={}\n",
            autodoc_factory::is_loaded(autodoc_factory::TILT)
        ));
        out.push_str(&format!(
            "getExistingAutodoc(tilt)==reg={}\n",
            std::ptr::eq(
                autodoc_factory::get_existing_autodoc(autodoc_factory::TILT),
                reg
            )
        ));
        out.push_str(&format!(
            "getExistingAutodoc(bogus)={}\n",
            autodoc_factory::get_existing_autodoc("bogus").is_null()
        ));
        autodoc_factory::reset_instance(autodoc_factory::TILT);
        out.push_str(&format!(
            "isLoaded(tilt)={}\n",
            autodoc_factory::is_loaded(autodoc_factory::TILT)
        ));
        let panicked = std::panic::catch_unwind(|| autodoc_factory::reset_instance("bogus"));
        out.push_str(&format!(
            "resetInstance(bogus)={}\n",
            match panicked {
                Ok(()) => "no throw".to_string(),
                Err(payload) => payload
                    .downcast_ref::<String>()
                    .unwrap()
                    .strip_prefix("java.lang.IllegalArgumentException: ")
                    .unwrap()
                    .to_string(),
            }
        ));

        out.push_str("== build ==\n");
        let a = Autodoc::new(Some("probe"), std::ptr::null_mut());
        out.push_str(&format!("autodocName={}\n", (*a).get_autodoc_name()));
        out.push_str(&format!(
            "getName={}\n",
            s(ReadOnlyStatementList::get_name(&*a))
        ));
        out.push_str(&format!(
            "getString=[{}]\n",
            ReadOnlyStatementList::get_string(&*a)
        ));
        out.push_str(&format!("toString=[{}]\n", (*a).to_string()));
        out.push_str(&format!("isError={}\n", (*a).is_error()));
        out.push_str(&format!("isDebug={}\n", (*a).is_debug()));
        out.push_str(&format!(
            "isGlobal={} isAttribute={}\n",
            WriteOnlyAttributeList::is_global(&*a),
            WriteOnlyAttributeList::is_attribute(&*a)
        ));
        out.push_str(&format!(
            "currentDelimiter={}\n",
            WriteOnlyStatementList::get_current_delimiter(&*a)
        ));
        (*a).add_comment_string(Some("first comment"), 1);
        WritableAutodoc::add_empty_line(&mut *a, 2);
        (*a).add_name_value_pair_attribute_with_line_num(Some("Version"), Some("1.2"), 3);
        (*a).add_name_value_pair_attribute_with_line_num(Some("Pip"), Some("1"), 4);
        (*a).add_name_value_pair_attribute_with_line_num(Some("  spaced  "), Some("trimmed"), 5);
        (*a).add_name_value_pair_attribute_with_line_num(Some("a.b.c"), Some("deep value"), 6);
        (*a).add_name_value_pair_attribute_with_line_num(Some("a.b.d"), Some("deep value 2"), 7);
        (*a).add_name_value_pair_attribute_with_line_num(Some("Version"), Some("1.3"), 8);
        WritableAutodoc::add_comment(&mut *a, tok("token comment"), 9);
        WritableAutodoc::add_comment(&mut *a, std::ptr::null_mut(), 10);
        let s1 = WriteOnlyStatementList::add_section(&mut *a, tok("Field"), tok("One"), 11);
        WriteOnlyStatementList::add_name_value_pair(&mut *s1, 12);
        let s2 = WriteOnlyStatementList::add_section(&mut *a, tok("Field"), tok("Two"), 13);
        let s3 = WriteOnlyStatementList::add_section(&mut *a, tok("Other"), tok("Three"), 14);
        let s1again = WriteOnlyStatementList::add_section(&mut *a, tok("field"), tok("one"), 15);
        out.push_str(&format!(
            "addSection returned existing={}\n",
            std::ptr::eq(s1again, s1)
        ));
        WriteOnlyStatementList::add_comment(&mut *s2, tok("in section"), 16);
        WriteOnlyStatementList::add_empty_line(&mut *s2, 17);
        let sub = WriteOnlyStatementList::add_section(&mut *s2, tok("Sub"), tok("Deep"), 18);
        out.push_str(&format!(
            "subsection string=[{}]\n",
            ReadOnlyStatementList::get_string(&*sub)
        ));
        out.push_str(&format!("s3 type={}\n", ReadOnlySection::get_type(&*s3)));

        out.push_str("== statements ==\n");
        dump_statements(&mut out, &*a, "");
        out.push_str("== sections in file order ==\n");
        dump_sections(&mut out, &*a, "", None);
        out.push_str("== sections of type Field ==\n");
        dump_sections(&mut out, &*a, "", Some("Field"));
        out.push_str("== sections of type field (case) ==\n");
        dump_sections(&mut out, &*a, "", Some("field"));
        out.push_str("== sections of type Other ==\n");
        dump_sections(&mut out, &*a, "", Some("Other"));
        out.push_str("== sections of type Missing ==\n");
        dump_sections(&mut out, &*a, "", Some("Missing"));
        out.push_str(&format!(
            "sectionExists(Field)={}\n",
            (*a).section_exists(Some("Field"))
        ));
        out.push_str(&format!(
            "sectionExists(Nope)={}\n",
            (*a).section_exists(Some("Nope"))
        ));
        out.push_str(&format!(
            "getSection(Field,One)={}\n",
            std::ptr::eq(
                ReadOnlySectionList::get_section(&*a, Some("Field"), Some("One")),
                s1
            )
        ));
        out.push_str(&format!(
            "getSection(field,ONE)={}\n",
            std::ptr::eq(
                ReadOnlySectionList::get_section(&*a, Some("field"), Some("ONE")),
                s1
            )
        ));
        out.push_str(&format!(
            "getSection(Field,Nope)={}\n",
            ReadOnlySectionList::get_section(&*a, Some("Field"), Some("Nope")).is_null()
        ));
        out.push_str(&format!(
            "s2.getSection(Sub,Deep)={}\n",
            std::ptr::eq(
                ReadOnlySectionList::get_section(&*s2, Some("Sub"), Some("Deep")),
                sub
            )
        ));

        out.push_str("== attributes ==\n");
        dump_attributes(&mut out, (*a).get_children(), "");
        out.push_str(&format!(
            "getAttribute(Version).getValue={}\n",
            s(ReadOnlyAttribute::get_value(
                &*(*a).get_attribute(Some("Version"))
            ))
        ));
        out.push_str(&format!(
            "getAttribute(version).getValue={}\n",
            s(ReadOnlyAttribute::get_value(
                &*(*a).get_attribute(Some("version"))
            ))
        ));
        let deep = ReadOnlyAttribute::get_attribute_by_name(
            &*ReadOnlyAttribute::get_attribute_by_name(&*(*a).get_attribute(Some("a")), Some("b")),
            Some("c"),
        );
        out.push_str(&format!(
            "getAttribute(a).getAttribute(b).getAttribute(c).getValue={}\n",
            s(ReadOnlyAttribute::get_value(&*deep))
        ));
        out.push_str(&format!(
            "getAttribute(nope)={}\n",
            (*a).get_attribute(Some("nope")).is_null()
        ));
        out.push_str("s1 statements:\n");
        dump_statements(&mut out, &*s1, "  ");
        out.push_str("s2 statements:\n");
        dump_statements(&mut out, &*s2, "  ");

        out.push_str("== section attributes ==\n");
        let np = WriteOnlyStatementList::add_name_value_pair(&mut *s2, 20);
        let at = WriteOnlyAttributeList::add_attribute(&mut *s2, tok("secattr"), 20);
        (*np).add_attribute(at);
        (*np).add_value(tok("secvalue"));
        let np2 = WriteOnlyStatementList::add_name_value_pair(&mut *s2, 21);
        let at2 = WriteOnlyAttributeList::add_attribute(&mut *s2, tok("secattr"), 21);
        (*np2).add_attribute(at2);
        (*np2).add_value(tok("secvalue2"));
        out.push_str(&format!(
            "same attribute instance={}\n",
            std::ptr::eq(at, at2)
        ));
        let ra = ReadOnlySection::get_attribute(&*s2, Some("SecAttr"));
        out.push_str(&format!(
            "s2 attr name={} value={} line={} global={} base={} exists={}\n",
            ReadOnlyAttribute::get_name(&*ra),
            s(ReadOnlyAttribute::get_value(&*ra)),
            ReadOnlyAttribute::get_line_num(&*ra),
            WriteOnlyAttributeList::is_global(&*ra),
            (*ra).is_base(),
            (*ra).exists()
        ));
        out.push_str(&format!(
            "np.getValue={} np2.getValue={}\n",
            s((*np).get_value()),
            s((*np2).get_value())
        ));
        out.push_str(&format!("np.toString=[{}]\n", (*np).to_string()));
        out.push_str(&format!(
            "s2 attr children={}\n",
            ReadOnlyAttribute::get_children(&*ra).is_null()
        ));
        out.push_str(&format!(
            "s2 getAttribute(nope)={}\n",
            ReadOnlySection::get_attribute(&*s2, Some("nope")).is_null()
        ));
        out.push_str(&format!(
            "firstAttribute={}\n",
            s(Some(ReadOnlyAttribute::get_name(
                &*(*(*a).get_children()).get_first_attribute()
            )))
        ));
        out.push_str(&format!(
            "getAttribute(a).getAttribute(9)={}\n",
            ReadOnlyAttribute::get_attribute_by_index(&*(*a).get_attribute(Some("a")), 9).is_null()
        ));
        out.push_str(&format!(
            "getAttribute(Version).getAttribute(9)={}\n",
            ReadOnlyAttribute::get_attribute_by_index(&*(*a).get_attribute(Some("Version")), 9)
                .is_null()
        ));

        out.push_str("== setValue ==\n");
        let wa = (*a).get_writable_attribute(Some("Pip"));
        WritableAttribute::set_value(&mut *wa, Some("changed"));
        out.push_str(&format!(
            "Pip={}\n",
            s(ReadOnlyAttribute::get_value(
                &*(*a).get_attribute(Some("Pip"))
            ))
        ));
        WritableAttribute::set_value(
            &mut *(*a).get_writable_attribute(Some("a")),
            Some("ignored"),
        );
        out.push_str(&format!(
            "a={}\n",
            s(ReadOnlyAttribute::get_value(
                &*(*a).get_attribute(Some("a"))
            ))
        ));

        out.push_str("== debug flags ==\n");
        let dbg = Autodoc::new(Some("dbg"), std::ptr::null_mut());
        out.push_str(&format!("isDebug={}\n", (*dbg).is_debug()));
        (*dbg).set_debug_to(true);
        out.push_str(&format!("isDebug={}\n", (*dbg).is_debug()));
        (*dbg).set_debug_to(false);
        out.push_str(&format!("isDebug={}\n", (*dbg).is_debug()));
        ReadOnlySectionList::set_debug(&mut *dbg);
        out.push_str(&format!("isDebug={}\n", (*dbg).is_debug()));

        out.push_str("== multi line values ==\n");
        let mv = (*a)
            .get_attribute_multi_line_values(Some("Field"), Some("secattr"))
            .unwrap();
        out.push_str(&format!(
            "multiValues size={} Two={}\n",
            mv.len(),
            s(mv.get("Two").cloned().flatten())
        ));
        let vv = (*a)
            .get_attribute_values(Some("Field"), Some("secattr"))
            .unwrap();
        out.push_str(&format!(
            "values size={} Two={}\n",
            vv.len(),
            s(vv.get("Two").cloned().flatten())
        ));

        out.push_str("== attribute values map ==\n");
        let m = (*a).get_attribute_values(Some("Field"), Some("nothing"));
        out.push_str(&format!(
            "values(Field,nothing)={}\n",
            match &m {
                None => "<null>".to_string(),
                Some(m) => m.len().to_string(),
            }
        ));
        out.push_str(&format!(
            "values(null,x)={}\n",
            (*a).get_attribute_values(None, Some("x")).is_none()
        ));
        out.push_str(&format!(
            "values(x,null)={}\n",
            (*a).get_attribute_values(Some("x"), None).is_none()
        ));

        out.push_str("== locations ==\n");
        let sl = ReadOnlyStatementList::get_statement_location(&*a).unwrap();
        out.push_str(&format!("sl={}\n", sl.to_string()));
        out.push_str(&format!(
            "nextStatement(null)={}\n",
            ReadOnlyStatementList::next_statement(&*a, None).is_null()
        ));
        out.push_str(&format!(
            "nextSection(null)={}\n",
            ReadOnlySectionList::next_section(&*a, None).is_null()
        ));

        out.push_str("== removal ==\n");
        let b = Autodoc::new(Some("removal"), std::ptr::null_mut());
        (*b).add_name_value_pair_attribute_with_line_num(Some("one"), Some("1"), 1);
        (*b).add_name_value_pair_attribute_with_line_num(Some("two"), Some("2"), 2);
        (*b).add_name_value_pair_attribute_with_line_num(Some("three"), Some("3"), 3);
        let prev = (*b).remove_name_value_pair(Some("two"));
        out.push_str(&format!(
            "removed prev=[{}]\n",
            if prev.is_null() {
                "<null>".to_string()
            } else {
                (*prev).get_string()
            }
        ));
        dump_statements(&mut out, &*b, "  ");
        out.push_str(&format!(
            "getAttribute(two) after remove={}\n",
            (*b).get_attribute(Some("two")).is_null()
        ));
        out.push_str(&format!(
            "removeNameValuePair(nope)={}\n",
            (*b).remove_name_value_pair(Some("nope")).is_null()
        ));
        let b2 = Autodoc::new(Some("removal2"), std::ptr::null_mut());
        (*b2).add_name_value_pair_attribute(Some("one"), Some("1"));
        (*b2).add_name_value_pair_attribute(Some("two"), Some("2"));
        (*b2).add_comment_string(Some("tail"), 3);
        let mut bl = ReadOnlyStatementList::get_statement_location(&*b2);
        let _first = ReadOnlyStatementList::next_statement(&*b2, bl.as_mut());
        let second = ReadOnlyStatementList::next_statement(&*b2, bl.as_mut());
        let prev2 = (*b2).remove_statement(second);
        out.push_str(&format!(
            "removeStatement prev=[{}]\n",
            if prev2.is_null() {
                "<null>".to_string()
            } else {
                (*prev2).get_string()
            }
        ));
        dump_statements(&mut out, &*b2, "  ");

        out.push_str("== odd names ==\n");
        let n = Autodoc::new(Some("names"), std::ptr::null_mut());
        (*n).add_name_value_pair_attribute_with_line_num(
            Some("\u{a0} pad \u{a0}"),
            Some("nbsp"),
            1,
        );
        (*n).add_name_value_pair_attribute_with_line_num(Some("a..b"), Some("doubledot"), 2);
        (*n).add_name_value_pair_attribute_with_line_num(Some("a."), Some("trailing"), 3);
        (*n).add_name_value_pair_attribute_with_line_num(Some(".a"), Some("leading"), 4);
        (*n).add_name_value_pair_attribute_with_line_num(Some("..."), Some("alldots"), 5);
        (*n).add_name_value_pair_attribute_with_line_num(None, Some("nullname"), 6);
        dump_statements(&mut out, &*n, "  ");
        dump_attributes(&mut out, (*n).get_children(), "  ");

        out.push_str("== wrap ==\n");
        let c = Autodoc::new(Some("wrap"), std::ptr::null_mut());
        (*c).add_name_value_pair_attribute_with_line_num(
            Some("w"),
            Some("aaaa bbbb cccc dddd eeee ffff gggg hhhh"),
            1,
        );
        (*c).wrap_attribute_values(Some("!"), Some("~"), Some("|"), Some(" "), 4, 10);
        out.push_str(&format!(
            "wrapped=[{}]\n",
            s(ReadOnlyAttribute::get_value(
                &*(*c).get_attribute(Some("w"))
            ))
        ));
    }
    assert_eq!(out, EXPECTED);
}

/// The `printStatementList` and `printStoredData` members of the same harness run,
/// which write through `System.out` rather than being rendered into the buffer.  The
/// libtest harness redirects Rust's `println!` at the library level rather than at file
/// descriptor 1, so this half is produced by re-running the test in a child process with
/// `--nocapture` and reading its real stdout.
#[test]
fn jvm_verified_autodoc_print_members() {
    if std::env::var("IMOD_RS_AUTODOC_PRINT_CHILD").is_err() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .env("IMOD_RS_AUTODOC_PRINT_CHILD", "1")
            .args([
                "--exact",
                "--nocapture",
                "--test-threads=1",
                "jvm_verified_autodoc_print_members",
            ])
            .output()
            .unwrap();
        let text = String::from_utf8(output.stdout).unwrap();
        let begin = text
            .find("<<<PROBE-BEGIN>>>\n")
            .expect("child begin marker")
            + "<<<PROBE-BEGIN>>>\n".len();
        let end = text.find("<<<PROBE-END>>>").expect("child end marker");
        assert_eq!(&text[begin..end], EXPECTED_PRINT);
        return;
    }
    unsafe {
        let c = Autodoc::new(Some("wrap"), std::ptr::null_mut());
        (*c).add_name_value_pair_attribute_with_line_num(
            Some("w"),
            Some("aaaa bbbb cccc dddd eeee ffff gggg hhhh"),
            1,
        );
        (*c).wrap_attribute_values(Some("!"), Some("~"), Some("|"), Some(" "), 4, 10);
        let d = Autodoc::new(Some("print"), std::ptr::null_mut());
        (*d).add_comment_string(Some("c"), 1);
        WritableAutodoc::add_empty_line(&mut *d, 2);
        (*d).add_name_value_pair_attribute_with_line_num(Some("only"), Some("value"), 3);
        let ds = WriteOnlyStatementList::add_section(&mut *d, tok("Sec"), tok("Nm"), 4);
        WriteOnlyStatementList::add_name_value_pair(&mut *ds, 5);
        println!("<<<PROBE-BEGIN>>>");
        (*c).print_statement_list();
        println!("== printStoredData ==");
        (*d).print_stored_data();
        println!("<<<PROBE-END>>>");
    }
}

/// Byte-for-byte output of the Java harness under the reference JVM.
const EXPECTED_PRINT: &str = r#"statementList=[w = aaaa bbbb
cccc dddd
eeee ffff
gggg
hhhh]
== printStoredData ==
Printing stored data:
LIST:
<comment> c
<empty-line>
value = value
Attributes:
only = value

[Sec = Nm]
Statements:
 = 
Attributes:
"#;

/// Byte-for-byte output of the buffered half of the Java harness under the reference JVM.
const EXPECTED: &str = r#"== statics ==
Section.getKey(Token,Token)=fieldname
Section.getKey(Token,null)=field
Section.getKey(null,Token)=name
Section.getKey(null,null)=<null>
Section.getKey(String,String)=fieldname
Section.getKey(String,null)=field
Section.getKey(null,String)=name
Section.getKey(null,null)=<null>
Attribute.getKey(Token)=abc
Attribute.getKey((Token)null)=<null>
Attribute.getKey(String)=abc
Attribute.getKey((String)null)=<null>
stripFileExtension(etomo.adoc)=etomo
stripFileExtension(etomo)=etomo
stripFileExtension(a.b.c)=a.b
stripFileExtension(.adoc)=
stripFileExtension()=
endsWithAutodocExtension(x.adoc)=true
endsWithAutodocExtension(x.prm)=true
endsWithAutodocExtension(x.txt)=false
endsWithAutodocExtension(null)=false
Extension.DEFAULT=.adoc str=adoc
replacementDir=<null>
replacementDir=/tmp/repl
isLoaded(etomo)=false
isLoaded(tilt)=false
setInstance(tilt)=true
setInstance(bogus)=false
isLoaded(tilt)=true
getExistingAutodoc(tilt)==reg=true
getExistingAutodoc(bogus)=true
isLoaded(tilt)=false
resetInstance(bogus)=Illegal autodoc name: bogus.
== build ==
autodocName=probe
getName=<null>
getString=[]
toString=[]
isError=true
isDebug=false
isGlobal=true isAttribute=false
currentDelimiter==
addSection returned existing=true
subsection string=[[[Sub = Deep]]]
s3 type=Other
== statements ==
stmt[0] type=COMMENT line=1 size=0 left=<null> left0=<null> left1=<null> right=first comment string=[# first comment]
stmt[1] type=EMPTY_LINE line=2 size=0 left=<null> left0=<null> left1=<null> right= string=[]
stmt[2] type=NAME_VALUE_PAIR line=3 size=1 left=Version left0=Version left1=<null> right=1.2 string=[Version = 1.2]
stmt[3] type=NAME_VALUE_PAIR line=4 size=1 left=Pip left0=Pip left1=<null> right=1 string=[Pip = 1]
stmt[4] type=NAME_VALUE_PAIR line=5 size=1 left=spaced left0=spaced left1=<null> right=trimmed string=[spaced = trimmed]
stmt[5] type=NAME_VALUE_PAIR line=6 size=3 left=a.b.c left0=a left1=b right=deep value string=[a.b.c = deep value]
stmt[6] type=NAME_VALUE_PAIR line=7 size=3 left=a.b.d left0=a left1=b right=deep value 2 string=[a.b.d = deep value 2]
stmt[7] type=NAME_VALUE_PAIR line=8 size=1 left=Version left0=Version left1=<null> right=1.3 string=[Version = 1.3]
stmt[8] type=COMMENT line=9 size=0 left=<null> left0=<null> left1=<null> right=token comment string=[# token comment]
stmt[9] type=COMMENT line=10 size=0 left=<null> left0=<null> left1=<null> right= string=[#]
statement count=10
== sections in file order ==
sectionLocation(<null>)=[type=null,index=0]
section[0] type=Field name=One string=[[Field = One]] loc=[type=null,index=1]
section[1] type=Field name=Two string=[[Field = Two]] loc=[type=null,index=2]
section[2] type=Other name=Three string=[[Other = Three]] loc=[type=null,index=3]
section count=3
== sections of type Field ==
sectionLocation(Field)=[type=Field,index=0]
section[0] type=Field name=One string=[[Field = One]] loc=[type=Field,index=1]
section[1] type=Field name=Two string=[[Field = Two]] loc=[type=Field,index=2]
section count=2
== sections of type field (case) ==
sectionLocation(field)=[type=field,index=0]
section[0] type=Field name=One string=[[Field = One]] loc=[type=field,index=1]
section[1] type=Field name=Two string=[[Field = Two]] loc=[type=field,index=2]
section count=2
== sections of type Other ==
sectionLocation(Other)=[type=Other,index=2]
section[0] type=Other name=Three string=[[Other = Three]] loc=[type=Other,index=3]
section count=1
== sections of type Missing ==
sectionLocation(Missing)=<null>
section count=0
sectionExists(Field)=true
sectionExists(Nope)=false
getSection(Field,One)=true
getSection(field,ONE)=true
getSection(Field,Nope)=true
s2.getSection(Sub,Deep)=true
== attributes ==
attr name=Version key=version line=3 exists=true global=true base=true value=1.3 multi=1.3
  attributes=<null>
attr name=Pip key=pip line=4 exists=true global=true base=true value=1 multi=1
  attributes=<null>
attr name=spaced key=spaced line=5 exists=true global=true base=true value=trimmed multi=trimmed
  attributes=<null>
attr name=a key=a line=6 exists=true global=true base=true value=<null> multi=<null>
  attr name=b key=b line=6 exists=true global=true base=false value=<null> multi=<null>
    attr name=c key=c line=6 exists=true global=true base=false value=deep value multi=deep value
      attributes=<null>
    attr name=d key=d line=7 exists=true global=true base=false value=deep value 2 multi=deep value 2
      attributes=<null>
getAttribute(Version).getValue=1.3
getAttribute(version).getValue=1.3
getAttribute(a).getAttribute(b).getAttribute(c).getValue=deep value
getAttribute(nope)=true
s1 statements:
  stmt[0] type=NAME_VALUE_PAIR line=12 size=0 left= left0=<null> left1=<null> right=<null> string=[ = ]
  statement count=1
s2 statements:
  stmt[0] type=COMMENT line=16 size=0 left=<null> left0=<null> left1=<null> right=in section string=[# in section]
  stmt[1] type=EMPTY_LINE line=17 size=0 left=<null> left0=<null> left1=<null> right= string=[]
  stmt[2] type=SUBSECTION line=18 size=1 left=Sub left0=Sub left1=<null> right=Deep string=[[[Sub = Deep]]]
    subsection type=Sub name=Deep
      statement count=0
  statement count=3
== section attributes ==
same attribute instance=true
s2 attr name=secattr value=secvalue2 line=20 global=false base=true exists=true
np.getValue=secvalue np2.getValue=secvalue2
np.toString=[secattr = secvalue]
s2 attr children=true
s2 getAttribute(nope)=true
firstAttribute=Version
getAttribute(a).getAttribute(9)=true
getAttribute(Version).getAttribute(9)=true
== setValue ==
Pip=changed
a=<null>
== debug flags ==
isDebug=false
isDebug=true
isDebug=false
isDebug=true
== multi line values ==
multiValues size=1 Two=secvalue2
values size=1 Two=secvalue2
== attribute values map ==
values(Field,nothing)=0
values(null,x)=true
values(x,null)=true
== locations ==
sl=index=0
nextStatement(null)=true
nextSection(null)=true
== removal ==
removed prev=[one = 1]
  stmt[0] type=NAME_VALUE_PAIR line=1 size=1 left=one left0=one left1=<null> right=1 string=[one = 1]
  stmt[1] type=NAME_VALUE_PAIR line=3 size=1 left=three left0=three left1=<null> right=3 string=[three = 3]
  statement count=2
getAttribute(two) after remove=true
removeNameValuePair(nope)=true
removeStatement prev=[one = 1]
  stmt[0] type=NAME_VALUE_PAIR line=0 size=1 left=one left0=one left1=<null> right=1 string=[one = 1]
  stmt[1] type=COMMENT line=3 size=0 left=<null> left0=<null> left1=<null> right=tail string=[# tail]
  statement count=2
== odd names ==
  stmt[0] type=NAME_VALUE_PAIR line=1 size=1 left=  pad   left0=  pad   left1=<null> right=nbsp string=[  pad   = nbsp]
  stmt[1] type=NAME_VALUE_PAIR line=2 size=2 left=a.b left0=a left1=b right=doubledot string=[a.b = doubledot]
  stmt[2] type=NAME_VALUE_PAIR line=3 size=1 left=a left0=a left1=<null> right=trailing string=[a = trailing]
  stmt[3] type=NAME_VALUE_PAIR line=4 size=1 left=a left0=a left1=<null> right=leading string=[a = leading]
  stmt[4] type=NAME_VALUE_PAIR line=5 size=0 left= left0=<null> left1=<null> right=<null> string=[ = ]
  statement count=5
  attr name=  pad   key=  pad   line=1 exists=true global=true base=true value=nbsp multi=nbsp
    attributes=<null>
  attr name=a key=a line=2 exists=true global=true base=true value=leading multi=leading
    attr name=b key=b line=2 exists=true global=true base=false value=doubledot multi=doubledot
      attributes=<null>
== wrap ==
wrapped=[aaaa bbbb
cccc dddd
eeee ffff
gggg
hhhh]
"#;
