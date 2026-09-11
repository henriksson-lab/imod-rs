//! Differential checks against the reference eTomo JVM for `etomo/BaseManager.java`'s
//! non-GUI half and the units it unblocked: `etomo/type/BaseMetaData.java`,
//! `etomo/type/EtomoVersion.java`, `etomo/type/StringProperty.java`,
//! `etomo/type/ImageOutputFormat.java`, `etomo/ui/FieldType.java`,
//! `etomo/logic/Converter.toDoubleArray` and `etomo/util/Utilities.convertLabelToName`.
//!
//! Every expectation below is the byte-for-byte output of a Java harness
//! (`etomo.type.Probe`) making the same calls in the same order against the reference
//! runtime, built the way the header of `tests/etomo_utilities_probe.rs` describes.  Set
//! `IMOD_RS_ETOMO_PROBE` to dump the table again for a fresh comparison.
use imod_rs::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use imod_rs::imod::etomo::logic::converter;
use imod_rs::imod::etomo::storage::storable::Storable;
use imod_rs::imod::etomo::r#type::axis_id::AxisID;
use imod_rs::imod::etomo::r#type::axis_type::AxisType;
use imod_rs::imod::etomo::r#type::base_meta_data::{BaseMetaData, BaseMetaDataBase};
use imod_rs::imod::etomo::r#type::const_etomo_version::ConstEtomoVersion;
use imod_rs::imod::etomo::r#type::etomo_version::EtomoVersion;
use imod_rs::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use imod_rs::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use imod_rs::imod::etomo::r#type::imod_output_format::ImodOutputFormat;
use imod_rs::imod::etomo::r#type::interface_type::InterfaceType;
use imod_rs::imod::etomo::r#type::string_property::StringProperty;
use imod_rs::imod::etomo::ui::field_type::FieldType;
use imod_rs::imod::etomo::util::utilities;
use std::collections::BTreeMap;
use std::convert::Infallible;

fn p(label: &str, value: &str) {
    if std::env::var("IMOD_RS_ETOMO_PROBE").is_ok() {
        println!("{}|{}", label, value);
    }
}

/// The harness's `TestMetaData`, a concrete subclass of the abstract `BaseMetaData`
/// built with `super(null, null, true, false, true)`.
struct TestMetaData {
    base: BaseMetaDataBase,
}

impl TestMetaData {
    fn new() -> TestMetaData {
        TestMetaData {
            base: BaseMetaDataBase::new_force_old_style(None, None, true, false, true),
        }
    }
}

impl BaseMetaData for TestMetaData {
    fn base(&self) -> &BaseMetaDataBase {
        &self.base
    }
    fn get_meta_data_file_name(&self) -> Option<String> {
        Some("test.edf".to_string())
    }
    fn get_name(&self) -> Option<String> {
        Some("dataset".to_string())
    }
    fn get_dataset_name(&self) -> Option<String> {
        Some("dataset".to_string())
    }
    fn is_valid(&self) -> bool {
        true
    }
    fn get_group_key(&self) -> Option<String> {
        Some("Setup".to_string())
    }
}

impl Storable for TestMetaData {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        self.store_with_prepend(properties, "");
    }
    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        let prepend = self.create_prepend(prepend);
        self.base()
            .store_with_created_prepend(properties, prepend.as_deref());
    }
    fn load(&mut self, properties: &BTreeMap<String, String>) {
        self.load_with_prepend(properties, "");
    }
    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, prepend: &str) {
        let created = self.create_prepend(prepend);
        if self
            .base()
            .load_with_created_prepend(properties, created.as_deref())
        {
            self.check_image_filename_style_loaded(prepend);
        }
    }
}

/// A concrete `BaseManager`, which the source's own managers all are.  It supplies only
/// the abstract members; every other member of the class has its base-class body.
struct TestManager {
    base: BaseManagerBase,
}

impl imod_rs::imod::etomo::ui::browsing_directory::BrowsingDirectory for TestManager {
    fn get_browsing_dir(&self) -> Option<std::path::PathBuf> {
        BaseManager::get_browsing_dir(self)
    }
    fn set_browsing_dir(&self, file: Option<&std::path::Path>) {
        BaseManager::set_browsing_dir(self, file)
    }
}

impl BaseManager for TestManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }
    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }
    fn get_interface_type(&self) -> Option<InterfaceType> {
        None
    }
    fn create_main_panel(&self) {}
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        None
    }
    fn get_main_panel(&self) -> Option<Infallible> {
        None
    }
    fn get_process_manager(&self) -> Option<Infallible> {
        None
    }
    fn get_storables_with_offset(
        &self,
        _offset: i32,
    ) -> Option<Vec<Box<dyn imod_rs::imod::etomo::storage::storable::Storable>>> {
        None
    }
    fn get_name(&self) -> Option<String> {
        Some("manager".to_string())
    }
}

#[test]
fn jvm_verified_etomo_version() {
    let v = EtomoVersion::get_default_instance_with_version(Some("4.10.43"));
    p("ev.toString", &v.to_string());
    assert_eq!(v.to_string(), "4.10.43");
    assert_eq!(v.get(0), Some("4".to_string()));
    assert_eq!(v.get(1), Some("10".to_string()));
    assert_eq!(v.get(5), None);
    assert!(v.is_numeric());
    assert!(v.lt_string(Some("4.11")));
    assert!(!v.lt_string(Some("4.10.42")));
    assert!(v.le_string(Some("4.10.43")));
    assert!(v.ge_string(Some("4.10.43")));
    assert!(v.gt(Some(&EtomoVersion::get_default_instance_with_version(
        Some("4.9")
    ))));
    assert!(
        v.equals(Some(&EtomoVersion::get_default_instance_with_version(
            Some("4.10.43")
        )))
    );

    // `extra` is `version.substring(array[0].length()).trim()`, printed after a space.
    let extra = EtomoVersion::get_default_instance_with_version(Some("1.2.3  extra bits"));
    p("ev.extra.toString", &extra.to_string());
    assert_eq!(extra.to_string(), "1.2.3 extra bits");

    // A version with no "." prints the WARNING and parses to nothing.
    let bad = EtomoVersion::get_default_instance_with_version(Some("nodots"));
    assert_eq!(bad.to_string(), "");
    assert!(bad.lt_string(Some("1.0")));

    let alpha = EtomoVersion::get_default_instance_with_version(Some("1.2.beta"));
    assert_eq!(alpha.to_string(), "1.2.beta");
    assert!(alpha.lt_string(Some("1.2.gamma")));
    assert!(!alpha.is_numeric());

    // `SectionList.add(SectionList)` copies `get(i)`, a String, so every section of the
    // copy is non-numeric - the JVM agrees: `isNumeric()` is false after `set`.
    let mut copy = EtomoVersion::get_empty_instance(Some("Version"));
    copy.set_etomo_version(Some(&v));
    p("ev.copy.isNumeric", &copy.is_numeric().to_string());
    assert_eq!(copy.to_string(), "4.10.43");
    assert!(!copy.is_numeric());

    let mut props: BTreeMap<String, String> = BTreeMap::new();
    let keyed = EtomoVersion::get_instance(Some("RevisionNumber"), Some("2.3.4"));
    keyed.store_with_prepend(&mut props, "Setup");
    assert_eq!(
        props.get("Setup.RevisionNumber"),
        Some(&"2.3.4".to_string())
    );
    keyed.store_with_prepend(&mut props, "Setup.");
    assert_eq!(
        props.get("Setup.RevisionNumber"),
        Some(&"2.3.4".to_string())
    );
    let mut loaded = EtomoVersion::get_empty_instance(Some("RevisionNumber"));
    loaded.load_with_prepend(&props, "Setup");
    assert_eq!(loaded.to_string(), "2.3.4");
}

#[test]
fn jvm_verified_string_property() {
    use imod_rs::imod::etomo::r#type::const_string_property::ConstStringProperty;
    let mut sp = StringProperty::new_with_key(Some("A.CurrentProcesschunksRootName"));
    assert_eq!(sp.to_string(), "");
    // `equals(null)` and `equals("*")` both fall through to `isEmpty(this.string)`.
    assert!(sp.equals(None));
    assert!(sp.equals(Some("*")));
    sp.set(Some("  "));
    assert_eq!(sp.to_string(), "");
    sp.set(Some("root"));
    p("sp.set.toString", &sp.to_string());
    assert_eq!(sp.to_string(), "root");
    assert_eq!(sp.length(), 4);
    assert!(sp.equals(Some("root")));
    assert!(!sp.equals(Some("other")));
    assert!(!sp.equals(Some("*")));

    let mut props: BTreeMap<String, String> = BTreeMap::new();
    sp.store_with_prepend(Some(&mut props), Some("Setup"));
    assert_eq!(
        props.get("Setup.A.CurrentProcesschunksRootName"),
        Some(&"root".to_string())
    );
    let mut sp2 = StringProperty::new_with_key(Some("A.CurrentProcesschunksRootName"));
    sp2.load_with_prepend(Some(&mut props), Some("Setup"));
    assert_eq!(sp2.to_string(), "root");
    sp2.reset();
    sp2.store_with_prepend(Some(&mut props), Some("Setup"));
    assert_eq!(props.get("Setup.A.CurrentProcesschunksRootName"), None);

    let sp_null = StringProperty::new_with_key_and_return_null_when_empty(Some("K"), true);
    assert_eq!(sp_null.to_string_option(), None);
    assert!(sp_null.is_empty());
    let mut sp_display = StringProperty::new_with_key_and_return_null_when_empty(Some("K"), true);
    sp_display.set_display_value(Some("dv"));
    assert_eq!(sp_display.to_string(), "dv");
}

#[test]
fn jvm_verified_image_output_format_and_field_type() {
    assert_eq!(ImageOutputFormat::DEFAULT.to_string(), "MRC");
    assert_eq!(
        ImageOutputFormat::get_instance(Some("TIFF")).to_string(),
        "TIFF"
    );
    assert_eq!(
        ImageOutputFormat::get_instance(Some("bogus")).to_string(),
        "MRC"
    );
    assert_eq!(
        ImageOutputFormat::get_instance_from_property_value(Some("bogus")),
        None
    );
    assert_eq!(
        ImageOutputFormat::get_instance_from_imod_output_format(Some(ImodOutputFormat::Tif))
            .to_string(),
        "TIFF"
    );
    assert_eq!(
        ImageOutputFormat::get_instance_from_image_filename_style(Some(ImageFilenameStyle::Hdf))
            .to_string(),
        "HDF"
    );
    assert!(ImageOutputFormat::Mrc.is_default());
    assert_eq!(ImageOutputFormat::Hdf.get_label(), "HDF");
    let mut props: BTreeMap<String, String> = BTreeMap::new();
    ImageOutputFormat::Hdf.store(&mut props, Some("Setup"), Some("ImageFile.Format"));
    assert_eq!(
        props.get("Setup.ImageFile.Format"),
        Some(&"HDF".to_string())
    );
    assert_eq!(
        ImageOutputFormat::load(&props, Some("Setup"), Some("ImageFile.Format")),
        Some(ImageOutputFormat::Hdf)
    );

    assert_eq!(FieldType::IntegerPair.required_size(), 2);
    assert!(FieldType::IntegerPair.has_required_size());
    assert!(!FieldType::String.has_required_size());
    assert_eq!(FieldType::StringArray.get_columns(), 15);
    assert_eq!(FieldType::IntegerTriple.get_columns(), 9);
    p(
        "ft.FLOATING_POINT_ARRAY.getSplitter",
        FieldType::FloatingPointArray.get_splitter(),
    );
    assert_eq!(
        FieldType::FloatingPointArray.get_splitter(),
        "\\s*,\\s*|\\s+"
    );
    assert_eq!(
        FieldType::IntegerList.get_splitter(),
        "\\s*,\\s*|\\s+|\\s*\\-\\s*"
    );
    assert_eq!(
        FieldType::MatlabIntegerArray.get_splitter(),
        "\\s*,\\s*|\\s+|\\s*\\:\\s*"
    );
    assert!(FieldType::Integer.get_numeric_type().is_some());
    assert!(FieldType::String.get_numeric_type().is_none());
    assert!(!FieldType::String.is_collection());
    // `toString`'s `validationType` half; the `collectionType` half is an identity hash
    // and is documented as unmatchable in `etomo/ui/field_type.rs`.
    assert!(
        FieldType::IntegerPair
            .to_string()
            .starts_with("[validationType:an integer,collectionType:")
    );
    assert_eq!(
        FieldType::String.to_string(),
        "[validationType:a string,collectionType:null,requiredSize:-1"
    );
}

#[test]
fn jvm_verified_converter_to_double_array() {
    let da = converter::to_double_array(Some("1.5, 2.5  3.5"), None).unwrap();
    assert_eq!(da, vec![Some(1.5), Some(2.5), Some(3.5)]);
    let db = converter::to_double_array(Some("1-3"), Some(FieldType::IntegerList)).unwrap();
    assert_eq!(db, vec![Some(1.0), Some(3.0)]);
    assert_eq!(converter::to_double_array(Some("  "), None), None);
}

#[test]
fn jvm_verified_base_meta_data() {
    let md = TestMetaData::new();
    assert_eq!(md.base().get_axis_type(), AxisType::NotSet);
    assert_eq!(md.base().get_axis_type().to_string(), "Not Set");
    assert_eq!(md.base().get_invalid_reason(), "");
    assert_eq!(md.base().get_revision_number().to_string(), "");
    assert_eq!(md.base().get_image_filename_style().to_string(), "0");
    assert!(md.base().is_old_image_filename_style());
    assert_eq!(md.base().get_image_output_format().to_string(), "MRC");
    assert_eq!(md.base().get_raw_image_stack_extension().to_string(), "st");
    assert_eq!(
        md.base().get_orig_raw_image_stack_extension().to_string(),
        "st"
    );
    assert_eq!(md.create_prepend(""), Some("Setup".to_string()));
    assert_eq!(md.create_prepend("A"), Some("A.Setup".to_string()));
    assert_eq!(
        md.get_image_filename_style_key(""),
        Some("Setup.ImageFile.ImageFilenameStyle".to_string())
    );
    assert_eq!(
        md.get_image_filename_style_key("A"),
        Some("A.Setup.ImageFile.ImageFilenameStyle".to_string())
    );
    assert_eq!(
        md.base()
            .get_current_processchunks_root_name(Some(AxisID::First)),
        Some("".to_string())
    );
    md.base()
        .set_current_processchunks_root_name(Some(AxisID::First), Some("rootA"));
    md.base()
        .set_current_processchunks_root_name(Some(AxisID::Second), Some("rootB"));
    md.base()
        .set_current_processchunks_subdir_name(Some(AxisID::Only), Some("subA"));
    assert_eq!(
        md.base()
            .get_current_processchunks_root_name(Some(AxisID::Only)),
        Some("rootA".to_string())
    );
    assert_eq!(
        md.base()
            .get_current_processchunks_root_name(Some(AxisID::Second)),
        Some("rootB".to_string())
    );
    assert!(
        md.base()
            .is_current_processchunks_subdir_name_set(Some(AxisID::Only))
    );
    assert!(
        !md.base()
            .is_current_processchunks_subdir_name_set(Some(AxisID::Second))
    );

    // The five properties the JVM's `store(props, "")` leaves behind, in the JVM's own
    // sorted order.  `Version.Etomo.Created` is absent because `newDataset` is false.
    let mut props: BTreeMap<String, String> = BTreeMap::new();
    md.store(&mut props);
    let dumped: Vec<String> = props
        .iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect();
    for entry in &dumped {
        p("bmd.prop", entry);
    }
    assert_eq!(
        dumped,
        vec![
            "Setup.A.CurrentProcesschunksRootName=rootA".to_string(),
            "Setup.A.CurrentProcesschunksSubdirName=subA".to_string(),
            "Setup.B.CurrentProcesschunksRootName=rootB".to_string(),
            "Setup.ImageFile.ImageFilenameStyle=OLD".to_string(),
            "Setup.Version.Etomo.Modified=5.2.17".to_string(),
        ]
    );

    let mut md2 = TestMetaData::new();
    md2.load(&props);
    assert_eq!(
        md2.base()
            .get_current_processchunks_root_name(Some(AxisID::Only)),
        Some("rootA".to_string())
    );
    assert_eq!(
        md2.base()
            .get_current_processchunks_root_name(Some(AxisID::Second)),
        Some("rootB".to_string())
    );
    assert!(md.base().equals(md2.base()));
    // `toString` up to the unmatchable `super.toString()`.
    let printed = md.base().to_string();
    assert!(
        printed
            .starts_with("[fileExtension:null,revisionNumber:,\naxisType:Not Set,invalidReason:")
    );
}

#[test]
fn jvm_verified_convert_label_to_name() {
    let labels = [
        "Fiducial Model",
        "check-up",
        "equals -20%",
        "1.0",
        "it's a test",
        "a, b \"c\"",
        "keep (drop this) rest",
        "<br>html<b>x</b>",
        "stop; after",
        "one two three four five six seven eight nine",
        "  Trim  Me  ",
        "",
        "=",
        "--",
        "Angle offset (degrees)",
    ];
    let limited = [
        Some("fiducial-model"),
        Some("check-up"),
        Some("equals--20%"),
        Some("1-0"),
        Some("it's-a-test"),
        Some("a-b-c"),
        Some("keep-rest"),
        Some("html-x"),
        Some("stop"),
        Some("one-two-three-four-five-six-seven"),
        Some("trim-me"),
        None,
        None,
        None,
        Some("angle-offset"),
    ];
    let unlimited = [
        Some("fiducial-model"),
        Some("check-up"),
        Some("equals--20%"),
        Some("1-0"),
        Some("it's-a-test"),
        Some("a-b-c"),
        Some("keep-rest"),
        Some("html-x"),
        Some("stop"),
        Some("one-two-three-four-five-six-seven-eight-nine"),
        Some("trim-me"),
        None,
        None,
        None,
        Some("angle-offset"),
    ];
    for index in 0..labels.len() {
        let got = utilities::convert_label_to_name(Some(labels[index]), false);
        p(
            &format!("label[{}]", index),
            got.clone().unwrap_or("null".to_string()).as_str(),
        );
        assert_eq!(
            got.as_deref(),
            limited[index],
            "convertLabelToName({:?}, false)",
            labels[index]
        );
        let got = utilities::convert_label_to_name(Some(labels[index]), true);
        assert_eq!(
            got.as_deref(),
            unlimited[index],
            "convertLabelToName({:?}, true)",
            labels[index]
        );
    }
    assert_eq!(
        utilities::convert_label_to_name_three(Some("First"), Some("Second"), Some("Third"), false)
            .as_deref(),
        Some("first-second-third")
    );
}

#[test]
fn base_manager_base_class_members() {
    // A leaked manager, which is what `EtomoDirector` does with every manager it builds.
    let manager: &'static TestManager = Box::leak(Box::new(TestManager {
        base: BaseManagerBase::initial(),
    }));
    // The base-class bodies that need nothing untranslated.
    assert!(manager.allow_process_watching());
    assert!(!manager.is_beadfixer_diameter_available());
    assert!(!manager.is_add_gpu_machine_to_process_chunks());
    assert!(!manager.is_dual_selection_queue_table());
    assert_eq!(manager.get_file_subdirectory_name(), None);
    assert_eq!(manager.get_parallel_processing_default_nice(), 15);
    assert!(!manager.can_change_param_file_name());
    assert!(!manager.can_save_directives());
    assert!(!manager.is_in_manager_frame());
    assert!(!manager.is_setup_done());
    assert!(!manager.is_loaded_param_file());
    assert!(manager.is_new_dataset());
    assert!(!manager.is_exiting());
    assert!(manager.is_valid());
    assert!(!manager.is_startup_popup_open());
    assert!(manager.is_popup_chunk_warnings());
    assert!(!manager.is_tomosnapshot_thumbnail());
    assert_eq!(manager.get_name().as_deref(), Some("manager"));
    assert_eq!(manager.param_string().as_deref(), Some("manager"));
    assert_eq!((manager as &dyn BaseManager).to_string(), "[manager]");
    // `getFileLockMessage` returns "" off Windows, which is where this test runs.
    assert_eq!(manager.get_file_lock_message(Some("\n\n")), "");

    // `setPropertyUserDir` returns the old value and blanks a whitespace-only argument.
    let old = manager.set_property_user_dir(Some("/tmp/etomo-probe"));
    assert_eq!(
        manager.get_property_user_dir().as_deref(),
        Some("/tmp/etomo-probe")
    );
    let old2 = manager.set_property_user_dir(Some("   "));
    assert_eq!(old2.as_deref(), Some("/tmp/etomo-probe"));
    assert_eq!(manager.get_property_user_dir(), None);
    let _ = old;

    // `setThreadName(null, ..)` restores NO_PROCESS_THREAD_NAME.
    manager.set_thread_name(Some("threadA"), Some(AxisID::First));
    manager.set_thread_name(Some("threadB"), Some(AxisID::Second));
    manager.set_thread_name(None, Some(AxisID::First));

    // `getEmergencyMonitor` is an n'ton on the instance and always uses AxisID.ONLY.
    let monitor = manager.get_emergency_monitor(Some(AxisID::Second));
    assert_eq!(monitor.get_axis_id(), Some(AxisID::Only));
    assert!(std::sync::Arc::ptr_eq(
        &monitor,
        &manager.get_emergency_monitor(None)
    ));
    assert!(monitor.get_manager().is_some());

    // `isReconnectRun` / `setReconnectRun` are per axis.
    assert!(!manager.is_reconnect_run(Some(AxisID::First)));
    manager.set_reconnect_run(Some(AxisID::Second));
    assert!(!manager.is_reconnect_run(Some(AxisID::First)));
    assert!(manager.is_reconnect_run(Some(AxisID::Second)));
}
