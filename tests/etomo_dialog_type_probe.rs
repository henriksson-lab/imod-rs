//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/type/DialogType.java`.
//!
//! `jvm_verified_expectations` asserts values captured from a real JVM run; it always
//! runs.  `etomo_dialog_type_probe` dumps the full comparison table used to capture
//! them and is skipped unless `IMOD_RS_ETOMO_PROBE` is set.  To regenerate the table,
//! build the reference classes outside `IMOD/` with
//! `javac -nowarn -d <out> -encoding ISO-8859-1` over every `.java` under
//! `IMOD/Etomo/src` except `*Test.java`, `*Tests.java`, `JUnit*`, `etomo/uitest/` and
//! `util/TestUtilites.java`, compile a Java harness that prints the same
//! `label<TAB>value` lines, and diff the two streams.
use imod_rs::imod::etomo::r#type::data_file_type::DataFileType;
use imod_rs::imod::etomo::r#type::dialog_type::{self, DialogType};
use std::collections::BTreeMap;

fn p(label: &str, value: Option<String>) {
    println!(
        "{}\t{}",
        label,
        value.unwrap_or_else(|| "<null>".to_string())
    );
}

const ALL: &[(&str, DialogType)] = &[
    ("SETUP_RECON", DialogType::SetupRecon),
    ("PRE_PROCESSING", DialogType::PreProcessing),
    ("COARSE_ALIGNMENT", DialogType::CoarseAlignment),
    ("FIDUCIAL_MODEL", DialogType::FiducialModel),
    ("FINE_ALIGNMENT", DialogType::FineAlignment),
    ("TOMOGRAM_POSITIONING", DialogType::TomogramPositioning),
    ("FINAL_ALIGNED_STACK", DialogType::FinalAlignedStack),
    ("TOMOGRAM_GENERATION", DialogType::TomogramGeneration),
    ("TOMOGRAM_COMBINATION", DialogType::TomogramCombination),
    ("POST_PROCESSING", DialogType::PostProcessing),
    ("CLEAN_UP", DialogType::CleanUp),
    ("JOIN", DialogType::Join),
    ("PARALLEL", DialogType::Parallel),
    ("ANISOTROPIC_DIFFUSION", DialogType::AnisotropicDiffusion),
    ("PEET_STARTUP", DialogType::PeetStartup),
    ("PEET", DialogType::Peet),
    ("SERIAL_SECTIONS_STARTUP", DialogType::SerialSectionsStartup),
    ("SERIAL_SECTIONS", DialogType::SerialSections),
    ("TOOLS", DialogType::Tools),
    ("DIRECTIVE_EDITOR", DialogType::DirectiveEditor),
    ("BATCH_RUN_TOMO", DialogType::BatchRunTomo),
];

const DFT: &[Option<DataFileType>] = &[
    Some(DataFileType::Recon),
    Some(DataFileType::Join),
    Some(DataFileType::Parallel),
    Some(DataFileType::BatchRunTomo),
    Some(DataFileType::Peet),
    Some(DataFileType::SerialSections),
    Some(DataFileType::Tools),
    Some(DataFileType::DirectiveEditor),
    None,
];

#[test]
fn etomo_dialog_type_probe() {
    if std::env::var("IMOD_RS_ETOMO_PROBE").is_err() {
        return;
    }
    p("TOTAL_RECON", Some(dialog_type::TOTAL_RECON.to_string()));
    p("TOTAL_JOIN", Some(dialog_type::TOTAL_JOIN.to_string()));
    p(
        "TOTAL_PARALLEL",
        Some(dialog_type::TOTAL_PARALLEL.to_string()),
    );
    p("TOTAL_PEET", Some(dialog_type::TOTAL_PEET.to_string()));
    p(
        "TOTAL_SERIAL_SECTIONS",
        Some(dialog_type::TOTAL_SERIAL_SECTIONS.to_string()),
    );
    p(
        "TOTAL_BATCH_RUN_TOMO",
        Some(dialog_type::TOTAL_BATCH_RUN_TOMO.to_string()),
    );
    for (n, d) in ALL {
        let d = *d;
        p(&format!("{}.toString", n), Some(d.to_string()));
        p(
            &format!("{}.getStorableName", n),
            Some(d.get_storable_name()),
        );
        p(
            &format!("{}.getCompactLabel", n),
            Some(d.get_compact_label()),
        );
        p(&format!("{}.getIndex", n), Some(d.get_index().to_string()));
        p(&format!("{}.toIndex", n), Some(d.to_index().to_string()));
        p(
            &format!("{}.getInterfaceType", n),
            d.get_interface_type().map(|x| x.to_string()),
        );
        p(
            &format!("{}.equals(null)", n),
            Some(d.equals(None).to_string()),
        );
        p(
            &format!("{}.equals(storable)", n),
            Some(d.equals(Some(&d.get_storable_name())).to_string()),
        );
        p(
            &format!("{}.equals(toString)", n),
            Some(d.equals(Some(&d.to_string())).to_string()),
        );
        p(
            &format!("{}.getInstance(storable)", n),
            DialogType::get_instance(Some(&d.get_storable_name())).map(|x| x.to_string()),
        );
        let mut props: BTreeMap<String, String> = BTreeMap::new();
        d.store(&mut props);
        p(&format!("{}.store", n), props.get("DialogType").cloned());
        let mut props2: BTreeMap<String, String> = BTreeMap::new();
        d.store_with_prepend(&mut props2, "a.b");
        p(
            &format!("{}.store(a.b)", n),
            props2.get("a.b.DialogType").cloned(),
        );
        let mut props3: BTreeMap<String, String> = BTreeMap::new();
        d.store_with_prepend(&mut props3, "a.");
        p(
            &format!("{}.store(a.)", n),
            props3.get("a.DialogType").cloned(),
        );
        p(
            &format!("{}.load(a.b)", n),
            DialogType::load(&props2, "a.b").map(|x| x.to_string()),
        );
        DialogType::remove(&mut props2, "a.b");
        p(
            &format!("{}.remove(a.b)", n),
            props2.get("a.b.DialogType").cloned(),
        );
    }
    p(
        "getInstance(null)",
        DialogType::get_instance(None).map(|x| x.to_string()),
    );
    p(
        "getInstance(empty)",
        DialogType::get_instance(Some("")).map(|x| x.to_string()),
    );
    p(
        "getInstance(Join)",
        DialogType::get_instance(Some("Join")).map(|x| x.to_string()),
    );
    p(
        "getInstance(SetupRecon)",
        DialogType::get_instance(Some("SetupRecon")).map(|x| x.to_string()),
    );
    p(
        "getInstance(unknown)",
        DialogType::get_instance(Some("nosuch")).map(|x| x.to_string()),
    );
    for t in DFT {
        let t = *t;
        let n = match t {
            None => "null".to_string(),
            Some(t) => t.to_string(),
        };
        p(
            &format!("getDefault({})", n),
            DialogType::get_default(t).map(|x| x.to_string()),
        );
        let empty: BTreeMap<String, String> = BTreeMap::new();
        p(
            &format!("load({}, empty)", n),
            DialogType::load_with_data_file_type(t, &empty).map(|x| x.to_string()),
        );
        let mut set: BTreeMap<String, String> = BTreeMap::new();
        set.insert("DialogType".to_string(), "TomoGen".to_string());
        p(
            &format!("load({}, TomoGen)", n),
            DialogType::load_with_data_file_type(t, &set).map(|x| x.to_string()),
        );
    }
}

/// Values captured byte-for-byte from the reference JVM harness described above.  The
/// full 353-line table matched; these are the lines that pin the source's quirks.
#[test]
fn jvm_verified_expectations() {
    // The five per-interface totals.
    assert_eq!(dialog_type::TOTAL_RECON, 11);
    assert_eq!(dialog_type::TOTAL_JOIN, 1);
    assert_eq!(dialog_type::TOTAL_PARALLEL, 2);
    assert_eq!(dialog_type::TOTAL_PEET, 2);
    assert_eq!(dialog_type::TOTAL_SERIAL_SECTIONS, 2);
    assert_eq!(dialog_type::TOTAL_BATCH_RUN_TOMO, 1);
    // The recon labels, including the one taken from SharedStrings.
    assert_eq!(DialogType::SetupRecon.to_string(), "Setup Tomogram");
    assert_eq!(DialogType::FiducialModel.to_string(), "Fiducial Model Gen.");
    assert_eq!(
        DialogType::FinalAlignedStack.to_string(),
        "Final Aligned Stack"
    );
    assert_eq!(DialogType::FinalAlignedStack.get_compact_label(), "Stack");
    assert_eq!(
        DialogType::FinalAlignedStack.get_storable_name(),
        "FinalStack"
    );
    assert_eq!(DialogType::CleanUp.get_index(), 10);
    // `getStorableName(DataFileType, int)` has no JOIN arm, so the JOIN singleton -
    // whose toString is "Join" and whose JOIN_NAME constant is "Join" - stores the
    // empty string and cannot be read back.
    assert_eq!(DialogType::Join.to_string(), "Join");
    assert_eq!(DialogType::Join.get_storable_name(), "");
    assert_eq!(DialogType::Join.get_compact_label(), "");
    assert!(DialogType::get_instance(Some("Join")).is_none());
    // Neither does it have a SERIAL_SECTIONS arm, and `toString` has none either, so
    // both serial-sections singletons are nameless.
    assert_eq!(DialogType::SerialSections.to_string(), "");
    assert_eq!(DialogType::SerialSectionsStartup.to_string(), "");
    assert_eq!(DialogType::SerialSections.get_storable_name(), "");
    assert!(DialogType::SerialSections.equals(Some("")));
    // PEET_STARTUP and PEET share DataFileType.PEET and differ only by index.
    assert_eq!(DialogType::PeetStartup.to_string(), "PEET Startup");
    assert_eq!(DialogType::PeetStartup.get_storable_name(), "PeetStart");
    assert_eq!(DialogType::PeetStartup.get_compact_label(), "PEET-Start");
    assert_eq!(DialogType::Peet.get_storable_name(), "Peet");
    // getInterfaceType delegates to DataFileType.
    assert_eq!(
        DialogType::SetupRecon
            .get_interface_type()
            .map(|x| x.to_string()),
        Some("recon".to_string())
    );
    assert_eq!(
        DialogType::SerialSections
            .get_interface_type()
            .map(|x| x.to_string()),
        Some("serialSections".to_string())
    );
    // store/load round trip through a prepend, with and without the trailing dot.
    let mut props: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    DialogType::TomogramGeneration.store_with_prepend(&mut props, "a.b");
    assert_eq!(
        props.get("a.b.DialogType").map(|x| x.as_str()),
        Some("TomoGen")
    );
    assert_eq!(
        DialogType::load(&props, "a.b"),
        Some(DialogType::TomogramGeneration)
    );
    DialogType::remove(&mut props, "a.b");
    assert!(props.get("a.b.DialogType").is_none());
    let mut dotted: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    DialogType::CleanUp.store_with_prepend(&mut dotted, "a.");
    assert_eq!(
        dotted.get("a.DialogType").map(|x| x.as_str()),
        Some("CleanUp")
    );
    // `load(DataFileType, Properties)` defaults with `defaultType.toString()` - the
    // display name - and then looks that up as a *storable* name, so only PARALLEL,
    // whose two names coincide, survives the round trip.
    let empty: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    assert_eq!(
        DialogType::load_with_data_file_type(Some(DataFileType::Parallel), &empty),
        Some(DialogType::Parallel)
    );
    assert!(DialogType::load_with_data_file_type(Some(DataFileType::Peet), &empty).is_none());
    assert!(
        DialogType::load_with_data_file_type(Some(DataFileType::BatchRunTomo), &empty).is_none()
    );
    assert!(
        DialogType::load_with_data_file_type(Some(DataFileType::DirectiveEditor), &empty).is_none()
    );
    assert!(DialogType::load_with_data_file_type(Some(DataFileType::Recon), &empty).is_none());
    // getDefault has four arms.
    assert_eq!(
        DialogType::get_default(Some(DataFileType::Parallel)),
        Some(DialogType::Parallel)
    );
    assert_eq!(
        DialogType::get_default(Some(DataFileType::BatchRunTomo)),
        Some(DialogType::BatchRunTomo)
    );
    assert_eq!(
        DialogType::get_default(Some(DataFileType::Peet)),
        Some(DialogType::Peet)
    );
    assert_eq!(
        DialogType::get_default(Some(DataFileType::DirectiveEditor)),
        Some(DialogType::DirectiveEditor)
    );
    assert!(DialogType::get_default(Some(DataFileType::Recon)).is_none());
    assert!(DialogType::get_default(None).is_none());
    // equals(null) is false for every singleton.
    assert!(!DialogType::SetupRecon.equals(None));
}
