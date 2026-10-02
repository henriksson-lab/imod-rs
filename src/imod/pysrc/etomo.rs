//! Translation of `IMOD/pysrc/etomo`, the Python launcher that starts eTomo
//! and manages its log files.
//!
//! This launcher intentionally retains `java … etomo.EtomoDirector` as the
//! JVM/UI boundary.  It does not replace Java eTomo with a different Rust UI.
//!
//! The script's two functions are [`which`] and [`roll_logs`]; its top level
//! is [`etomo`], translated statement by statement.  Its platform tests map
//! as `'cygwin' in sys.platform` -> `target_os = "cygwin"`, `'win32'` ->
//! `windows`, `'darwin'` -> `target_os = "macos"` and `'linux'` ->
//! `target_os = "linux"`.  An exception the script does not catch ends it
//! with a traceback and exit status 1; the translation prints the
//! exception's last line on standard error and returns 1 at those points.

use super::imodpy::{
    bkgd_process, convert_to_integer, cygwin_path, glob_glob, make_backup_file, prnstr, run_cmd,
    set_lib_path,
};
use super::pip::set_exit_prefix;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Matches `which` (`IMOD/pysrc/etomo:11`).
pub fn which(prog: &str) -> Option<String> {
    let mut prog = prog.to_owned();
    if cfg!(target_os = "cygwin") || cfg!(windows) {
        prog.push_str(".exe");
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path) {
        let full = dir.join(&prog);
        // `os.path.exists(full) and os.access(full, os.X_OK)`
        let Ok(c_full) = std::ffi::CString::new(full.to_string_lossy().into_owned()) else {
            continue;
        };
        if full.exists() && unsafe { libc::access(c_full.as_ptr(), libc::X_OK) } == 0 {
            return Some(full.to_string_lossy().into_owned());
        }
    }
    None
}

/// Matches `rollLogs` (`IMOD/pysrc/etomo:21`).
pub fn roll_logs() {
    let mut lasterr = "etomo_err12.log".to_owned();
    for i in (0..=11).rev() {
        let mut thiserr = format!("etomo_err{i}.log");
        if i == 0 {
            thiserr = "etomo_err.log".to_owned();
        }
        if Path::new(&thiserr).exists() {
            let result = (|| {
                if lasterr == "etomo_err12.log" && Path::new(&lasterr).exists() {
                    fs::remove_file(&lasterr)?;
                }
                fs::rename(&thiserr, &lasterr)
            })();
            if let Err(error) = result {
                prnstr(
                    &format!(
                        "WARNING: an error occurred renaming {thiserr} to {lasterr} ({error})"
                    ),
                    "\n",
                    false,
                );
            }
        }
        lasterr = thiserr;
    }
}

/// Original Python top-level program (`IMOD/pysrc/etomo:1`).
pub fn etomo(arguments: &[OsString]) -> i32 {
    let progname = "etomo";
    let prefix = format!("ERROR: {progname} - ");
    let sys_argv = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let pathsep = if cfg!(windows) { ";" } else { ":" };

    //
    // Setup runtime environment - no need for nohup here
    if let Some(imod_dir) = std::env::var_os("IMOD_DIR") {
        let mut imod_dir = imod_dir.to_string_lossy().into_owned();
        if cfg!(target_os = "cygwin") {
            imod_dir = imod_dir.replace('\\', "/");
            let bytes = imod_dir.as_bytes();
            if bytes.len() < 3 {
                eprintln!("IndexError: string index out of range");
                return 1;
            }
            if bytes[1] == b':' && bytes[2] == b'/' {
                imod_dir = format!(
                    "/cygdrive/{}{}",
                    (bytes[0] as char).to_ascii_lowercase(),
                    &imod_dir[2..]
                );
            }
        }
        // `sys.path.insert(0, os.path.join(IMOD_DIR, 'pylib'))` and
        // `from imodpy import *` locate the Python library; here it is linked
        let mut path = OsString::from(Path::new(&cygwin_path(&imod_dir)).join("bin"));
        path.push(pathsep);
        path.push(std::env::var_os("PATH").unwrap_or_default());
        unsafe {
            std::env::set_var("PATH", path);
        }
    } else {
        print!(
            "The IMOD_DIR environment variable has not been set\nSet it to point to the directory where IMOD is installed\n"
        );
        return 1;
    }

    //
    // load IMOD Libraries
    set_exit_prefix(prefix);
    set_lib_path();
    let good_mac_javas = [(
        "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/",
        "/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/",
    )];
    let bad_mac_javas = [
        "/System/Library/Frameworks/JavaVM.framework/Versions/",
        "/usr/bin/",
    ];

    let _newstuff = sys_argv.iter().any(|arg| arg == "--newstuff");

    let mut etomo_mem_lim = "512m".to_owned();
    if let Some(value) = std::env::var_os("ETOMO_MEM_LIM") {
        etomo_mem_lim = value.to_string_lossy().into_owned();
    }

    // Set thread limit; default to minimum of 16 and number of cores including hyperthreading
    let etomo_thread_lim = if let Some(value) = std::env::var_os("ETOMO_THREAD_LIM") {
        value.to_string_lossy().into_owned()
    } else {
        let mut limit = 16;
        // `etomo:79-97` ran `imodqtassist -t` and took the integer after the
        // `=` on its `thread count` line.  Owner decision (2026-09-24): that
        // number is our `imodqtassist`'s "Qt ideal thread count"
        // (`imodqtassist.rs`, `-t`), computed here directly instead of
        // through `sh -c`/`PATH`.
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        limit = std::cmp::min(limit, cores);
        limit.to_string()
    };

    // Python `os.path.getmtime`: seconds as a double
    let getmtime = |path: &str| -> Option<f64> {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(path)
            .ok()
            .map(|meta| meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9)
    };

    let mut no_java = false;
    if let Some(java_dir) = std::env::var_os("IMOD_JAVADIR") {
        let java_dir = cygwin_path(&java_dir.to_string_lossy());
        let mut path = OsString::from(Path::new(&java_dir).join("bin"));
        path.push(pathsep);
        path.push(std::env::var_os("PATH").unwrap_or_default());
        unsafe {
            std::env::set_var("PATH", path);
        }
    } else if cfg!(target_os = "macos") {
        // On Mac, if there is no IMOD_JAVADIR, first see if it just runs, and if not look
        // for an openjdk install, add most recent one to path
        match run_cmd("java -version", None, None, Some("stdout"), &[]) {
            Ok(_) => no_java = false,
            Err(_) => {
                no_java = true;

                if Path::new("/Library/Java/JavaVirtualMachines").exists() {
                    let jvms = glob_glob("/Library/Java/JavaVirtualMachines/*");
                    if !jvms.is_empty() {
                        let mut good_path = String::new();
                        let mut good_time: Option<f64> = None;
                        for ind in 0..jvms.len() {
                            if Path::new(&format!("{}/Contents/Home/bin/java", jvms[ind])).exists()
                            {
                                let test_time = getmtime(&jvms[ind]).unwrap_or(0.0);
                                // Fixed in translation (BUGS.md): `if not ind or
                                // testTime > goodTime` (`etomo:120`) reads an unbound
                                // `goodTime` (NameError) when the first entry had no
                                // java; the first JVM that has one is taken instead.
                                if ind == 0
                                    || good_time.is_none_or(|good_time| test_time > good_time)
                                {
                                    good_time = Some(test_time);
                                    good_path = format!("{}/Contents/Home/bin", jvms[ind]);
                                }
                            }
                        }
                        if !good_path.is_empty() {
                            no_java = false;
                            let mut path = OsString::from(&good_path);
                            path.push(pathsep);
                            path.push(std::env::var_os("PATH").unwrap_or_default());
                            unsafe {
                                std::env::set_var("PATH", path);
                            }
                        }
                    }
                }
            }
        }

        // Otherwise see if there is a java in any good location from old Oracle installs
        for good_dir in good_mac_javas {
            if !no_java {
                continue;
            }

            let good_one = format!("{}/bin/java", good_dir.0);
            if Path::new(&good_one).exists() {
                // Get its time
                let good_time = getmtime(good_dir.1).unwrap_or(0.0);

                // Then look for java on the path
                let path_java = match run_cmd("which java", None, None, Some("stdout"), &[]) {
                    Ok(lines) => {
                        let lines = lines.unwrap_or_default();
                        no_java = lines.is_empty() || lines[0].contains("not found");
                        lines
                    }
                    Err(_) => {
                        no_java = true;
                        Vec::new()
                    }
                };

                // If there is a good java and none on path, put good one on path
                if no_java {
                    let mut path = OsString::from(Path::new(good_dir.0).join("bin"));
                    path.push(pathsep);
                    path.push(std::env::var_os("PATH").unwrap_or_default());
                    unsafe {
                        std::env::set_var("PATH", path);
                    }
                } else {
                    // If there is one on path, resolve the link if any and see if it is
                    // one of the bad places
                    let first = path_java[0].trim_end_matches(['\r', '\n']);
                    let real_path_java = fs::canonicalize(first)
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| first.to_owned());
                    let dirname = Path::new(&real_path_java)
                        .parent()
                        .map(|path| path.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let Some(path_time) = getmtime(&dirname) else {
                        eprintln!(
                            "FileNotFoundError: [Errno 2] No such file or directory: '{dirname}'"
                        );
                        return 1;
                    };
                    for bad_start in bad_mac_javas {
                        if real_path_java.starts_with(bad_start) {
                            // If java in a bad place is older than the good one, add to path
                            if good_time > path_time {
                                let mut path = OsString::from(Path::new(good_dir.0).join("bin"));
                                path.push(pathsep);
                                path.push(std::env::var_os("PATH").unwrap_or_default());
                                unsafe {
                                    std::env::set_var("PATH", path);
                                }
                                break;
                            }
                        }
                    }
                }
            }
            break;
        }
    }
    let _ = no_java;

    // Test for appropriate java run time
    let verslines = match run_cmd("java -version", None, None, Some("stdout"), &[]) {
        Ok(lines) => lines.unwrap_or_default(),
        Err(_) => {
            let mut no_java = true;
            let mut retry_lines: Vec<String> = Vec::new();
            let win_native = "C:/Windows/Sysnative";
            if (cfg!(target_os = "cygwin") || cfg!(windows)) && Path::new(win_native).exists() {
                let mut path = OsString::from(cygwin_path(win_native));
                path.push(pathsep);
                path.push(std::env::var_os("PATH").unwrap_or_default());
                unsafe {
                    std::env::set_var("PATH", path);
                }
                if let Ok(lines) = run_cmd("java -version", None, None, Some("stdout"), &[]) {
                    retry_lines = lines.unwrap_or_default();
                    no_java = false;
                }
            }

            if no_java {
                prnstr(
                    "ERROR: There is no java runtime in the current search path.  A Java
runtime environment needs to be installed and the command search path may need
to be defined or IMOD_JAVADIR set to locate the java command.",
                    "\n",
                    false,
                );
                return 1;
            }
            // Fixed in translation (BUGS.md): native's `sys.exit(1)`
            // (`etomo:185`) sits outside `if noJava:`, so it quits even after
            // the Sysnative java ran; this carries on with that java.
            retry_lines
        }
    };

    let mut major = 0i64;
    let mut build = 0i64;
    let version_re = regex::Regex::new(r"ersion.*1\.[45]").expect("version pattern");
    for line in &verslines {
        if line.contains("GNU") {
            let mut errstr = "ERROR: Etomo will not work with GNU java.  You should install an
OpenJDK version of the Java runtime environment and put it on
your command search path"
                .to_owned();
            if let Some(java_dir) = std::env::var_os("IMOD_JAVADIR") {
                errstr += &format!(" or make a link to it from {}", java_dir.to_string_lossy());
            }
            prnstr(&errstr, "\n", false);
            return 1;
        }

        if version_re.is_match(line) {
            let mut errstr = "ERROR: You are trying to run a version of Java before 1.6".to_owned();
            if let Some(fulljava) = which("java") {
                errstr += &format!(", located at {fulljava}");
            }
            prnstr(&errstr, "\n", false);
            let mut errstr = "Etomo will no longer work with java 1.4-1.5.  You should install an
Oracle or OpenJDK version of the Java runtime environment, version 1.6 or higher,
and put it on your command search path, or point IMOD_JAVADIR to it"
                .to_owned();
            if let Some(java_dir) = std::env::var_os("IMOD_JAVADIR") {
                errstr += &format!(" or make a link to it from {}", java_dir.to_string_lossy());
            }
            prnstr(&errstr, "\n", false);
            return 1;
        }

        // Try to get the precise version
        if line.contains("version") && line.contains('"') {
            let mut line = line.as_str();
            let ind = line.find('"').unwrap_or(0);
            line = &line[ind + 1..];
            if let Some(ind) = line.find('"').filter(|&ind| ind > 0) {
                line = &line[..ind];
                let replaced = line.replace('_', ".");
                let lsplit = replaced.split('.').collect::<Vec<_>>();
                if lsplit.len() > 2 {
                    // `int(token)` for every token, inside a `try` that
                    // abandons the whole assignment on the first failure;
                    // `int()` ignores surrounding white space
                    let vals = lsplit
                        .iter()
                        .map(|token| super::imodpy::py_int(token))
                        .collect::<Option<Vec<_>>>();
                    if let Some(vals) = vals {
                        let _minor;
                        if vals[0] > 1 {
                            major = vals[0];
                            _minor = vals[1];
                        } else {
                            major = vals[1];
                            _minor = vals[2];
                        }
                        build = vals[vals.len() - 1];
                    }
                }
            }
        }
    }

    // In cygwin, put bin on front of path and make sure python is installed
    // Class path separator has to be ; in both cygwin and Windows because java is using it
    let mut cygbin = String::new();
    let mut userhome = String::new();
    let mut userhome_quoted = String::new();
    let mut class_path_sep = ":";
    if cfg!(target_os = "cygwin") {
        let mut path = OsString::from("/bin");
        path.push(pathsep);
        path.push(std::env::var_os("PATH").unwrap_or_default());
        unsafe {
            std::env::set_var("PATH", path);
        }
        cygbin = "/bin".to_owned();
        class_path_sep = ";";
    }

    // But in Windows, we need to find cygwin in path unless the psutil module is present
    if cfg!(windows) {
        // `import psutil` has no counterpart in this program, so the
        // ImportError arm is the one taken
        let find_cyg = true;
        class_path_sep = ";";

        if find_cyg {
            let mut cygdrive = "C".to_owned();
            let path = std::env::var("PATH").unwrap_or_default();
            for dir in path.split(pathsep) {
                if dir.to_lowercase().contains("cygwin") {
                    cygdrive = dir.chars().next().map(String::from).unwrap_or_default();
                    break;
                }
            }
            let cygtry = Path::new(&format!("{cygdrive}:\\cygwin"))
                .join("bin")
                .to_string_lossy()
                .into_owned();
            if Path::new(&cygtry).join("python.exe").exists() {
                cygbin = cygtry;
                let mut path = OsString::from(&cygbin);
                path.push(pathsep);
                path.push(std::env::var_os("PATH").unwrap_or_default());
                unsafe {
                    std::env::set_var("PATH", path);
                }
            } else {
                prnstr(
                    "ERROR: You must have the psutil module installed to run Etomo with Windows Python",
                    "\n",
                    false,
                );
                return 1;
            }
        }
    }

    if !cygbin.is_empty() && !Path::new(&cygbin).join("python.exe").exists() {
        if Path::new(&cygbin).join("python").exists() {
            prnstr(
                "ERROR: There must be a python.exe in the Cygwin bin in order to use Etomo",
                "\n",
                false,
            );
            let pythlist = glob_glob(&Path::new(&cygbin).join("python?.?.exe").to_string_lossy());
            if !pythlist.is_empty() {
                prnstr(
                    "You should run this command in a Cygwin terminal:",
                    "\n",
                    false,
                );
                prnstr(
                    &format!("   cp {} {cygbin}/python.exe", pythlist[0]),
                    "\n",
                    false,
                );
            } else {
                prnstr(
                    "It does not work to have a Cygwin link from python to python2.x.exe",
                    "\n",
                    false,
                );
            }
        } else {
            prnstr(
                "ERROR: You must have python installed in Cygwin in order to use Etomo",
                "\n",
                false,
            );
        }
        return 1;
    }

    if !cygbin.is_empty() {
        // `getpass.getuser()`: the first of these variables that is set, a
        // KeyError (passed over) when none is.  `platform.release()` is never
        // 'XP' for this program, which no toolchain builds for XP.
        if let Some(username) = ["LOGNAME", "USER", "LNAME", "USERNAME"]
            .iter()
            .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
        {
            let home = format!("C:\\Users\\{username}");
            if Path::new(&home).exists() {
                userhome = format!("-Duser.home={home}");
                userhome_quoted = format!("-Duser.home=\"{home}\"");
            }
        }
    }

    // Make sure awk doesn't produce commas (probably not needed)
    unsafe {
        std::env::set_var("LC_NUMERIC", "C");
        std::env::set_var("PIP_PRINT_ENTRIES", "1");
    }

    // This variable will generate a lot of output and Etomo messes up after excludeviews
    if std::env::var_os("RUNCMD_VERBOSE").is_some() {
        unsafe {
            std::env::remove_var("RUNCMD_VERBOSE");
        }
    }

    // In linux, test for headless unless it is appropriate
    if cfg!(target_os = "linux")
        && !(sys_argv.iter().any(|arg| arg == "--directive")
            || sys_argv.iter().any(|arg| arg == "--headless"))
    {
        if let Ok(settings) = run_cmd("java -XshowSettings", None, None, Some("stdout"), &[1]) {
            for line in settings.unwrap_or_default() {
                if line.contains("java.awt.headless") && line.contains("true") {
                    prnstr(
                        "ERROR: The installed java is \"headless\"; to open the Etomo interface\n  you need to use a full installation of java that does not have\n  \"headless\" in its package name",
                        "\n",
                        false,
                    );
                    return 1;
                }
            }
        }
    }

    // Check for help option
    // Check for foreground option - needed to run multiple etomos with automation.
    let mut help = 0;
    let mut foreground = 0;
    let has = |name: &str| sys_argv.iter().any(|arg| arg == name);
    if has("-h") || has("--help") || has("--h") || has("--grabit") {
        help = 1;
    }
    if has("--fg") || has("--directive") || has("--grabit") {
        foreground = 1;
    }

    // add plugin locations to the classpath
    let imod_dir = std::env::var("IMOD_DIR").unwrap_or_default();
    let mut plugin_paths = String::new();
    let path = Path::new(&imod_dir)
        .join("Plugins")
        .to_string_lossy()
        .into_owned();
    if Path::new(&path).exists() {
        plugin_paths += &format!("{class_path_sep}{path}/*");
    }
    let path = Path::new(&imod_dir)
        .join("imodplug")
        .join("etomo")
        .to_string_lossy()
        .into_owned();
    if Path::new(&path).exists() {
        plugin_paths += &format!("{class_path_sep}{path}/*");
    }

    // Allow developer to run a specified jar
    let mut jar_dir = format!("{imod_dir}/bin/");
    if has("--jardir") {
        for ind in 1..sys_argv.len().saturating_sub(1) {
            if sys_argv[ind] == "--jardir" {
                jar_dir = sys_argv[ind + 1].clone();
                break;
            }
        }
    }

    // Build the common java command
    let mut javacom = format!("java -Xmx{etomo_mem_lim}");
    let mut com_array = vec!["java".to_owned(), format!("-Xmx{etomo_mem_lim}")];
    let mut opts = vec!["-XX:ConcGCThreads=", "-XX:ParallelGCThreads="];
    if major > 8 || (major == 8 && build >= 191) {
        opts.push("-XX:ActiveProcessorCount=");
    }
    for opt in opts {
        javacom += &format!(" {opt}{etomo_thread_lim}");
        com_array.push(format!("{opt}{etomo_thread_lim}"));
    }

    javacom += &format!(
        " {userhome_quoted} -cp \"{jar_dir}/etomo.jar{plugin_paths}\" etomo.EtomoDirector"
    );
    if !userhome.is_empty() {
        com_array.push(userhome);
    }
    com_array.extend([
        "-cp".to_owned(),
        format!("{jar_dir}/etomo.jar{plugin_paths}"),
        "etomo.EtomoDirector".to_owned(),
    ]);
    let mut skip_next = false;
    for ind in 1..sys_argv.len() {
        if skip_next {
            skip_next = false;
            continue;
        }
        let mut arg = sys_argv[ind].as_str();
        if arg == "-h" {
            arg = "--help";
        }
        if arg == "--jardir" {
            skip_next = true;
            continue;
        }
        javacom += &format!(" \"{arg}\"");
        com_array.push(arg.to_owned());
        if arg.starts_with('-') && !arg.starts_with("--") {
            prnstr(
                &format!("WARNING: YOU ENTERED AN ARGUMENT WITH A SINGLE DASH: {arg}"),
                "\n",
                false,
            );
        }
    }

    //prnstr(javacom)

    if help != 0 {
        match run_cmd(&javacom, None, None, Some("pipe"), &[]) {
            Ok(help_lines) => {
                // Fixed in translation (BUGS.md): `runcmd` returns None for
                // empty output and native's loop (`etomo:385`) raises an
                // uncaught TypeError, exit 1; here empty output prints nothing
                // and the help path exits 0.
                let help_lines = help_lines.unwrap_or_default();
                for l in help_lines {
                    prnstr(l.trim_end_matches(['\r', '\n']), "\n", false);
                }
            }
            Err(_) => prnstr(
                "An error occurred running etomo for help output",
                "\n",
                false,
            ),
        }
        return 0;
    }

    // If ETOMO_LOG_DIR is defined and writable, set up log files there with
    // date/time stamp; if not defined, put them in a hidden directory
    let outlog = "etomo_out.log";
    let mut errlog = "etomo_err.log".to_owned();

    let mut etomo_log_dir = String::new();
    if let Some(value) = std::env::var_os("ETOMO_LOG_DIR") {
        etomo_log_dir = value.to_string_lossy().into_owned();

    // Put logs in hidden directory if directory is not defined
    } else if let Some(home) = std::env::var_os("HOME") {
        etomo_log_dir = format!("{}/.etomologs", home.to_string_lossy());
        if cfg!(target_os = "cygwin") {
            etomo_log_dir = cygwin_path(&etomo_log_dir);
        }
        if !Path::new(&etomo_log_dir).exists() && fs::create_dir(&etomo_log_dir).is_err() {
            prnstr(
                &format!("WARNING: Failed to create logs directory {etomo_log_dir}"),
                "\n",
                false,
            );
        }
    }

    // `os.access(ETOMO_LOG_DIR, os.W_OK)`
    let writable = !etomo_log_dir.is_empty()
        && std::ffi::CString::new(etomo_log_dir.clone())
            .is_ok_and(|c_dir| unsafe { libc::access(c_dir.as_ptr(), libc::W_OK) } == 0);
    if writable {
        // purge the directory to 30 sessions or whatever user chooses
        let mut purgenum = 31;
        if let Some(value) = std::env::var_os("ETOMO_LOGS_TO_RETAIN") {
            purgenum = convert_to_integer(
                &value.to_string_lossy(),
                "environment variable ETOMO_LOGS_TO_RETAIN",
            );
        }
        let d = chrono::Local::now();
        let timestamp = d.format("%b-%d-%H%M%S").to_string();
        errlog = format!("{etomo_log_dir}/etomo_err_{timestamp}.log");

        // Get a sorted list by modification time (now from Python 2.3 docs!)
        let loglist = match fs::read_dir(&etomo_log_dir) {
            Ok(entries) => entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            Err(error) => {
                eprintln!("OSError: {error}: '{etomo_log_dir}'");
                return 1;
            }
        };
        let mut tmplist: Vec<(f64, String)> = Vec::new();
        for x in loglist {
            // `os.stat(...).st_mtime`, a double
            let full = Path::new(&etomo_log_dir).join(&x);
            match fs::metadata(&full) {
                Ok(meta) => {
                    use std::os::unix::fs::MetadataExt;
                    tmplist.push((meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9, x));
                }
                Err(error) => {
                    eprintln!("OSError: {error}: '{}'", full.display());
                    return 1;
                }
            }
        }
        // `tmplist.sort()` over `(st_mtime, name)` tuples: ties on the
        // time fall to the file name
        tmplist.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let loglist = tmplist.into_iter().map(|(_, x)| x).collect::<Vec<_>>();

        // Go through list from newest backwards, look for matches, and start removing
        // after the purge number is reached
        let mut num_match = 0;
        for ind in (0..loglist.len()).rev() {
            // `fnmatch.fnmatch(name, 'etomo_*.log')`
            if loglist[ind].starts_with("etomo_") && loglist[ind].ends_with(".log") {
                num_match += 1;
                if num_match > purgenum {
                    let fname = Path::new(&etomo_log_dir)
                        .join(&loglist[ind])
                        .to_string_lossy()
                        .into_owned();
                    if fs::remove_file(&fname).is_err() {
                        //prnstr('Purged ' + fname)
                        prnstr(
                            &format!("WARNING: failed to remove old log {fname}"),
                            "\n",
                            false,
                        );
                    }
                }
            }
        }

        // If there is an existing real log, roll it
        let mut errfile: Option<fs::File> = None;
        if Path::new("etomo_err.log").exists() {
            let result = (|| -> std::io::Result<()> {
                let mut file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open("etomo_err.log")?;
                // `errfile.readline()` in text mode: up to a universal line
                // ending, decoded as UTF-8
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;
                let end = bytes
                    .iter()
                    .position(|&byte| byte == b'\n' || byte == b'\r')
                    .unwrap_or(bytes.len());
                let line = String::from_utf8(bytes[..end].to_vec())
                    .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
                errfile = Some(file);
                if !line.contains("Error log") {
                    errfile = None;
                    roll_logs();
                }
                Ok(())
            })();
            if result.is_err() {
                prnstr(
                    "WARNING: Errors occurred managing an existing etomo_err.log",
                    "\n",
                    false,
                );
                errfile = None;
            }
        }

        // Append location of log to etomo_err.log here
        let result = (|| -> std::io::Result<()> {
            let mut file = match errfile.take() {
                None => fs::File::create("etomo_err.log")?,
                Some(mut file) => {
                    file.seek(SeekFrom::End(0))?;
                    file
                }
            };
            file.write_all(
                format!(
                    "Error log for {} is in {errlog}\n",
                    d.format("%a %b %d %H:%M:%S %Y")
                )
                .as_bytes(),
            )
        })();
        if result.is_err() {
            prnstr(
                "WARNING: An error occurred appending to the etomo_err.log",
                "\n",
                false,
            );
        }
    } else {
        // Otherwise roll numbered logs here
        roll_logs();
    }

    // Copy the previous out log file to backup
    make_backup_file(outlog);

    prnstr(&format!("Starting Etomo with log in {errlog}"), "\n", false);
    prnstr(
        "This log may contain personal information, such as your username",
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    if foreground == 0 {
        let com_array = com_array
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
        let _ = bkgd_process(&com_array, Some(outlog), Some(&errlog), false, false);
    } else {
        if fs::File::create(outlog).is_err() || fs::File::create(&errlog).is_err() {
            prnstr(
                "ERROR: An error occurred opening the standard output or error output log file",
                "\n",
                false,
            );
            return 1;
        }
        if run_cmd(&javacom, None, Some(outlog), Some(&errlog), &[]).is_err() {
            prnstr(
                &format!("ERROR: etomo exited with an error status, check: {errlog}"),
                "\n",
                false,
            );
            return 1;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::which;
    #[test]
    fn finds_a_path_command() {
        assert!(which("sh").is_some());
    }
}
