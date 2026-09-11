//! Differential check against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/storage/LogFile.java`'s locking layer and
//! `IMOD/Etomo/src/etomo/process/EmergencyMonitor.java`.
//!
//! Every expectation below was captured from a real JVM run of a Java harness that makes
//! the same calls in the same order.  Build the reference classes outside `IMOD/` with
//! `javac -nowarn -d <out> -encoding ISO-8859-1` over every `.java` under
//! `IMOD/Etomo/src` except `*Test.java`, `*Tests.java`, `JUnit*`, `etomo/uitest/` and
//! `util/TestUtilites.java`, then run the harness in an empty directory and diff its
//! `label=value` stream against this test's.
//!
//! The lock numbers in the `Id.toString` values are the sharp edge: `Lock.setLockNumber`
//! hands out one number per lock taken over the life of the `LogFile` instance, so
//! `writer=1`, `reader=2`, `bigBufferReader=5` and `outputStream=7` only line up if every
//! intervening `create`, `getLineContaining` and reader open takes exactly the lock the
//! source takes.
use imod_rs::imod::etomo::storage::log_file::LogFile;
use std::collections::BTreeMap;

/// `Thread.currentThread().getId()` differs between the JVM's main thread and a Rust test
/// thread, so the `handle.toString` suffix is normalised before comparison.
fn normalize_thread(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '(' {
            let mut digits = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    digits.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            if !digits.is_empty() && chars.peek() == Some(&')') {
                chars.next();
                out.push_str("(T)");
                continue;
            }
            out.push('(');
            out.push_str(&digits);
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn jvm_verified_log_file_locking() {
    let dir = std::env::temp_dir().join(format!("imod_rs_log_file_probe_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("probe.log");

    let handle = LogFile::get_instance_file(Some(&file), None).unwrap();
    assert_eq!(normalize_thread(&handle.to_string()), "probe.log(T)");
    assert_eq!(handle.get_name(), "probe.log");
    assert_eq!(
        handle
            .get_backup_file()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
        "probe.log~"
    );
    assert!(!handle.exists());
    assert!(handle.create().unwrap());
    assert!(handle.exists());
    assert!(!handle.is_locked());

    // Writer round trip.
    let writer_id = handle.open_writer_append(true).unwrap();
    assert_eq!(
        normalize_thread(&writer_id.to_string()),
        "[1,writer,probe.log(T)]"
    );
    assert!(!writer_id.is_empty());
    assert!(handle.is_locked_id(Some(&writer_id)));
    assert!(handle.is_locked());
    handle.write(Some("first line"), &writer_id).unwrap();
    handle.new_line(&writer_id).unwrap();
    handle.write(Some("second line"), &writer_id).unwrap();
    handle.new_line(&writer_id).unwrap();
    handle.write_character(Some('X'), &writer_id).unwrap();
    handle.new_line(&writer_id).unwrap();
    handle.flush(&writer_id).unwrap();
    handle.close_id(Some(&writer_id));
    assert!(!handle.is_locked());
    assert_eq!(
        normalize_thread(&writer_id.to_string()),
        "[null,writer,probe.log(T)]"
    );

    // Writing through a closed id is an UnlockedException whose message carries the id
    // and the LogFile.
    let e = handle.write(Some("nope"), &writer_id).unwrap_err();
    assert_eq!(
        normalize_thread(&e.get_message()),
        format!(
            "fileWriter is null.  Unable to write: nope\nid=[null,writer,probe.log(T)]logFile=[fileAbsolutePath={}]",
            file.display()
        )
    );

    // Reader round trip.
    let reader_id = handle.open_reader().unwrap().unwrap();
    assert_eq!(
        normalize_thread(&reader_id.to_string()),
        "[2,reader,probe.log(T)]"
    );
    let mut lines = Vec::new();
    while let Some(line) = handle.read_line(&reader_id).unwrap() {
        lines.push(line);
    }
    assert_eq!(lines, vec!["first line", "second line", "X"]);
    handle.close_id(Some(&reader_id));

    let e = handle.read_line(&reader_id).unwrap_err();
    assert_eq!(
        normalize_thread(&e.get_message()),
        format!(
            "Unlocked log file.  Cannot access.\nid=[null,reader,probe.log(T)]logFile=[fileAbsolutePath={}]",
            file.display()
        )
    );

    assert_eq!(
        handle.get_line_containing("second").unwrap(),
        Some("second line".to_string())
    );
    assert_eq!(handle.get_line_containing("absent").unwrap(), None);

    let big_id = handle.open_big_buffer_reader().unwrap().unwrap();
    assert_eq!(
        normalize_thread(&big_id.to_string()),
        "[5,high capacity reader,probe.log(T)]"
    );
    assert!(handle.search_for_last_line(&big_id, "X").unwrap());
    handle.close_id(Some(&big_id));
    let big_id2 = handle.open_big_buffer_reader().unwrap().unwrap();
    assert!(!handle.search_for_last_line(&big_id2, "first").unwrap());
    handle.close_id(Some(&big_id2));

    // Properties.
    let out_id = handle.open_output_stream().unwrap();
    assert_eq!(
        normalize_thread(&out_id.to_string()),
        "[7,output stream,probe.log(T)]"
    );
    let mut store: BTreeMap<String, String> = BTreeMap::new();
    store.insert("alpha".to_string(), "1".to_string());
    store.insert("beta".to_string(), "two".to_string());
    handle.store(&store, &out_id).unwrap();
    handle.close_id(Some(&out_id));
    // `Properties.store(OutputStream, null)` writes a `#` line carrying
    // `new Date().toString()` and then one `key=value` line per property; the reference
    // run left `#Fri Sep 11 06:51:34 CEST 2026`, `alpha=1`, `beta=two`.
    let stored = std::fs::read_to_string(&file).unwrap();
    let stored: Vec<&str> = stored.lines().collect();
    assert!(stored[0].starts_with('#'), "{}", stored[0]);
    assert_eq!(&stored[1..], ["alpha=1", "beta=two"]);
    let in_id = handle.open_input_stream().unwrap();
    let mut load: BTreeMap<String, String> = BTreeMap::new();
    handle.load(&mut load, &in_id).unwrap();
    handle.close_id(Some(&in_id));
    assert_eq!(load.get("alpha").map(String::as_str), Some("1"));
    assert_eq!(load.get("beta").map(String::as_str), Some("two"));

    // Backup.
    assert!(!handle.is_backedup());
    assert!(handle.backup_once().unwrap());
    assert!(handle.is_backedup());
    assert!(!handle.backup_once().unwrap());
    assert!(!handle.backup().unwrap());
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["probe.log~"]);
    assert!(!handle.exists());

    // Rename and delete.
    assert!(handle.create().unwrap());
    let dest = dir.join("renamed.log");
    assert!(handle.rename_safely(Some(&dest)).unwrap());
    assert!(!handle.exists());
    assert!(dest.exists());
    let dest_handle = LogFile::get_instance_file(Some(&dest), None).unwrap();
    assert!(dest_handle.equals_file(Some(&dest)));
    assert!(!dest_handle.equals_file(Some(&file)));
    assert!(dest_handle.equals_string(Some(dest.to_str().unwrap())));
    assert!(dest_handle.delete().unwrap());
    assert!(!dest.exists());

    // Copy.
    assert!(handle.create().unwrap());
    let w2 = handle.open_writer_append(false).unwrap();
    handle.write(Some("copy me"), &w2).unwrap();
    handle.new_line(&w2).unwrap();
    handle.close_id(Some(&w2));
    let copy_to = dir.join("copy.log");
    assert!(
        handle
            .copy(None, None, Some(&copy_to), false, false, false)
            .unwrap()
    );
    assert!(copy_to.exists());
    let copy_handle = LogFile::get_instance_file(Some(&copy_to), None).unwrap();
    let cr = copy_handle.open_reader().unwrap().unwrap();
    assert_eq!(
        copy_handle.read_line(&cr).unwrap(),
        Some("copy me".to_string())
    );
    copy_handle.close_id(Some(&cr));

    // Instance identity: one LogFile per physical file, one Handle per getInstance call.
    let handle2 = LogFile::get_instance_file(Some(&file), None).unwrap();
    assert!(!handle2.equals_handle(Some(&handle)));
    assert!(handle2.equals_file(Some(&file)));
    handle2.self_test_equals(&handle, true);
    handle2.self_test_equals(&copy_handle, false);

    // The test hook returns an id because the default test level is "Default test".
    assert_eq!(
        normalize_thread(&handle.get_writer_id_for_test().unwrap().to_string()),
        "[null,writer,probe.log(T)]"
    );

    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["copy.log", "probe.log", "probe.log~"]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A second handle on the same file cannot take a write lock while a file lock is held.
/// The source waits out `SleepTimer`'s full 15-second limit before giving up, so this
/// test takes that long on both sides; the reference JVM harness reported
/// `elapsedSeconds=15` for the same sequence.
///
/// What it pins: `Lock.checkBlockingId`'s message, `getExceptionMessage`'s extended-info
/// layout, the lock numbers the two ids carry when the collision is reported, and the
/// fact that `Lockable.retry` reaches `EmergencyMonitor.alert` - whose `updateProgressBar`
/// writes `"\n" + StandardBarString.buildBarString(...) + "\n"` to stderr, which the
/// reference run showed as `Writing blocked.log failed`.
#[test]
fn jvm_verified_lock_collision() {
    let dir = std::env::temp_dir().join(format!("imod_rs_log_lock_probe_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("blocked.log");

    let h1 = LogFile::get_instance_file(Some(&file), None).unwrap();
    assert!(h1.create().unwrap());
    let file_id = h1.open_file_lock(false).unwrap();
    assert_eq!(
        normalize_thread(&file_id.to_string()),
        "[1,file,blocked.log(T)]"
    );

    let h2 = LogFile::get_instance_file(Some(&file), None).unwrap();
    let start = std::time::Instant::now();
    let e = h2.open_writer().unwrap_err();
    let message = e.get_message();
    let (first, second) = message.split_once('\n').unwrap();
    assert_eq!(
        first,
        "Unable to lock blocked.log because of another etomo lock."
    );
    // The thread numbers are the JVM's; only their shape is comparable.
    assert!(
        second.starts_with("New Id thread: ") && second.contains(", Blocking Id thread: "),
        "{}",
        second
    );
    assert_eq!(start.elapsed().as_secs(), 15);
    assert!(h1.is_locked());

    // The bar string EmergencyMonitor.updateProgressBar printed on stderr.
    assert_eq!(
        imod_rs::imod::etomo::ui::standard_bar_string::StandardBarString::build_bar_string_static(
            Some(imod_rs::imod::etomo::ui::standard_bar_string::StandardBarString::Writing),
            Some("blocked.log"),
            None,
            false,
            true,
        ),
        "Writing blocked.log failed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
