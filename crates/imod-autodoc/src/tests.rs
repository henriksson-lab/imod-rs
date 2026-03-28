use crate::Autodoc;

const SAMPLE_ADOC: &str = r#"Version = 1.0
Pip = 1

[SectionHeader = IOOptions]
usage = INPUT AND OUTPUT FILE OPTIONS

[Field = InputFile]
short = input
type = FNM
usage = Input image file
tooltip =
manpage = Input image file.  Input files may also be entered after all
^arguments on the command line.

[Field = OutputFile]
short = output
type = FNM
usage = Output image file
tooltip =
manpage = Output image file.

[Field = ModeToOutput]
short = mode
type = I
usage = Output data mode (0=byte, 1=short, 2=float, 6=ushort)
tooltip =
manpage = Output file data mode.
"#;

#[test]
fn parse_globals() {
    let adoc = Autodoc::parse(SAMPLE_ADOC);
    assert_eq!(adoc.version(), Some("1.0"));
    assert_eq!(adoc.pip_version(), Some("1"));
}

#[test]
fn parse_sections() {
    let adoc = Autodoc::parse(SAMPLE_ADOC);
    // 1 SectionHeader + 3 Fields
    assert_eq!(adoc.sections.len(), 4);
    assert_eq!(adoc.sections[0].kind, "SectionHeader");
    assert_eq!(adoc.sections[0].name, "IOOptions");
}

#[test]
fn parse_fields() {
    let adoc = Autodoc::parse(SAMPLE_ADOC);
    let fields = adoc.fields();
    assert_eq!(fields.len(), 3);

    assert_eq!(fields[0].name, "InputFile");
    assert_eq!(fields[0].short, "input");
    assert_eq!(fields[0].field_type, "FNM");
    assert_eq!(fields[0].usage, "Input image file");

    assert_eq!(fields[1].name, "OutputFile");
    assert_eq!(fields[1].short, "output");

    assert_eq!(fields[2].name, "ModeToOutput");
    assert_eq!(fields[2].short, "mode");
    assert_eq!(fields[2].field_type, "I");
}

#[test]
fn continuation_lines() {
    let adoc = Autodoc::parse(SAMPLE_ADOC);
    let fields = adoc.fields();
    // The InputFile manpage should contain the continuation
    assert!(fields[0].manpage.contains("arguments on the command line"));
}

#[test]
fn parse_real_adoc_file() {
    let path = std::path::Path::new("../../IMOD/autodoc/newstack.adoc");
    if path.exists() {
        let adoc = Autodoc::from_file(path).unwrap();
        assert_eq!(adoc.version(), Some("1.0"));
        let fields = adoc.fields();
        assert!(!fields.is_empty());
        // newstack should have InputFile and OutputFile
        assert!(fields.iter().any(|f| f.name == "InputFile"));
        assert!(fields.iter().any(|f| f.name == "OutputFile"));
    }
}
