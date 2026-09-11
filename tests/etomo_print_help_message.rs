//! Differential check against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/Arguments.java`'s `printHelpMessage`.
//!
//! The expected text below is the byte-for-byte tail of a real JVM run.  Build the
//! reference classes outside `IMOD/` with `javac -nowarn -d <out> -encoding ISO-8859-1`
//! over every `.java` under `IMOD/Etomo/src` except `*Test.java`, `*Tests.java`,
//! `JUnit*`, `etomo/uitest/` and `util/TestUtilites.java`, then run
//!
//! ```text
//! IMOD_DIR=<imod> java -Djava.awt.headless=true -cp <out> etomo.EtomoDirector --help
//! ```
//!
//! `EtomoDirector.setup()`'s unmatchable startup dump - `System.getenv()` in Java
//! `HashMap` order, every `System.getProperties()` entry, a listing of the JVM's own
//! `lib` directory and a `new Date()` (`EtomoDirector.java:254-340`) - goes to
//! **stderr**, so stdout alone is 90 lines: line 1 and the blank line 2 are
//! `EtomoDirector.printUsageMessage()`'s own
//! `System.out.println("Usage: etomo [options] [data files]\n")`
//! (`EtomoDirector.java:1303`), and lines 3-90 are `printHelpMessage`.  Those 88 lines
//! (3408 bytes) are what this test asserts.  Captured with the two streams merged
//! (`2>&1`) the same run is 295 lines and the tail starts at line 208.
//!
//! Two details of the expected bytes are load-bearing:
//!
//! * `usage` is preferred over `manpage` and is emitted **without**
//!   `stripManpageFormatting`, so `--namingstyle` keeps its literal `\fB0\fR...` roff
//!   escapes.  Only `format` and `manpage` are stripped.
//! * the walk is `getSectionLocation()` + `nextSection(sectionLocation)`, which is file
//!   order *across* collection types - each `SectionHeader` is followed by the `Field`
//!   sections that follow it in the file, not by every `SectionHeader` first.
//!
//! `printHelpMessage` writes to `System.out`, and a Rust `#[test]` runs under libtest's
//! output capture, so the test re-executes its own binary with `--nocapture` and reads
//! the child's stdout.
use imod_rs::imod::etomo::arguments::Arguments;
use std::io::Write;

/// Set in the child process only.
const CHILD_ENV: &str = "IMOD_RS_ETOMO_PRINT_HELP_CHILD";

/// Bracket markers the child prints around `printHelpMessage`'s own output, so the
/// libtest banner can be sliced off without touching the bytes under test.
const BEGIN: &str = "<<<BEGIN-PRINT-HELP-MESSAGE>>>";
const END: &str = "<<<END-PRINT-HELP-MESSAGE>>>";

/// `etomo.EtomoDirector --help` from line 208 on, captured from the reference JVM.
const EXPECTED: &str = "\nOPTIONS\n--help OR --h OR -h\n     Output usage message.\n--computersection   [Ialternative_computer_section]\n     Controls which type of section in cpu.adoc will be loaded into the CPU and GPU tables.  To prevent these tables from being filled from cpu.adoc, omit the value.\n--listen\n     Run 3dmod with -L (Windows only).\n--moveb\n     Automatically keeps the B axis window out of the way of the A axis window.\n--namingstyle   0|1|2\n     Sets the file name style for a new dataset.  \\fB0\\fRnon-standard,\\fB1\\fRmrc,\\fB2\\fRhdf.\n--queuesection   [Ialternative_queue_section]\n     Controls which type of section in cpu.adoc will be loaded into the Queue table.  To prevent this table from being fileed from cpu.adoc, omit the value.\n--timestamp\n     Timestamp processes.\n--userTemplateLoc   \"directory_path\"\n     Adds a second User Template directory.\n\nAUTOMATION OPTIONS - DIRECTIVE FILE\n--directive   \"directive_file.adoc\"\n     Do automation based on the directive_file.adoc.\n--fromBRT\n     Prevents Etomo from calling batchruntomo for validation.  Only used by batchruntomo when calling Etomo.\n\nAUTOMATION OPTIONS - COMMAND LINE\n--axis   single|dual\n     Sets the Axis Type in the Setup Tomogram dialog during automation.\n--cpus   [ignored]\n     Turns on the Parallel Processing checkbox in the Setup Tomogram dialog.\n--create\n     Runs Create Com Scripts in the Setup Tomogram dialog during automation.\n--dataset   tilt_series_file\n     Deprecated.  Same as rawimagestack.\n--rawimagestack   tilt_series_file\n     Sets Raw Image Stack in the Setup Tomogram dialog during automation.\n--dir   \"directory_path\"\n     The location of the dataset during automation.\n--exit\n     During automation, exit after the Setup Tomogram dialog has completed.\n--fg\n     Run Etomo in the foreground.\n--fiducial   Floating point\n     Sets the Fiducial Diameter in the Setup Tomogram dialog during automation.\n--frame   single|montage\n     Sets the Frame Type in the Setup Tomogram dialog during automation.\n--gpus   [ignored]\n     Selects the Graphics Card Processing checkbox in the Setup Tomogram dialog.\n--scan\n     Runs Scan Header in the Setup Tomogram dialog during automation.\n\nEXTENSIONS\n--plugin\n     For loading built-in plugins.  Will be expanded as necessary.  Currently causes the etomo.plugin.demo package to be loaded in preference to external plugins implementing the same interface.\n\nDIAGNOSTIC OPTIONS\n--actions\n     Send actions and file name to Etomo's _err.log.\n--debug   [-1|0|1|2|3|4]\n     Send extra information to Etomo's _err.log.\n--grabit   [c|p]\n     Causes the file parameter to be processed by the message processor.\n--ignoresettings\n     ETomo will not load from or save to the .etomo configuration file.\n--memory   [interval_in_minutes]\n     Log memory usage statements before and after processes are run.\n--selftest\n     Causes Etomo to do some internal testing.\n\nDEVELOPMENT AND TESTING OPTIONS\n--autoclose3dmod\n     ETomo automatically closes the 3dmod instance.\n--headless\n     No interface will come up.\n--ignoreloc\n     The interface will come up in the default location.\n--names\n     The names of screen elements are sent to Etomo's _out.log.\n--newstuff\n     May cause Etomo to run with unreleased functionality.\n--noload\n     Suppresses the computer and queue load updates.\n--test\n     Used for unit testing and automated user interface testing.\n\nDEPRECATED OPTIONS\n--demo\n     Deprecated\n";

#[test]
fn print_help_message_matches_the_jvm() {
    if std::env::var(CHILD_ENV).is_ok() {
        print!("{}", BEGIN);
        Arguments::print_help_message();
        print!("{}", END);
        std::io::stdout().flush().unwrap();
        std::process::exit(0);
    }
    // `IMOD/autodoc/etomo.adoc` is byte-identical to the reference build's copy, so the
    // reference run and this one read the same file.
    let imod_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    assert!(imod_dir.join("autodoc").join("etomo.adoc").is_file());
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("print_help_message_matches_the_jvm")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env("IMOD_DIR", &imod_dir)
        .env_remove("AUTODOC_DIR")
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let begin = stdout.find(BEGIN).unwrap_or_else(|| {
        panic!(
            "no output; stderr was:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    }) + BEGIN.len();
    let end = stdout.find(END).unwrap();
    let actual = &stdout[begin..end];
    if actual != EXPECTED {
        for (i, (a, e)) in actual.lines().zip(EXPECTED.lines()).enumerate() {
            if a != e {
                panic!("line {} differs:\n  rust: {:?}\n  java: {:?}", i + 1, a, e);
            }
        }
        panic!(
            "line counts differ: rust {} java {}",
            actual.lines().count(),
            EXPECTED.lines().count()
        );
    }
    assert_eq!(actual.lines().count(), 88);
}
