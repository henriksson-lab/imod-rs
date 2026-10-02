//! The eTomo generic process-launch layer (`etomo/process`), driven end to end.
//!
//! A test `BaseManager` owns a dataset directory; `BaseProcessManager` starts
//! a command file through `ComScriptProcess` (our `vmstopy`, then the script in
//! our command-file runner), a plain program through `BackgroundProcess`, and
//! kills a running command file through `imodkillgroup`, exactly as the Java
//! eTomo does.  Each test waits for the manager's `processDone`, which the
//! process thread posts to the event dispatch thread.
//!
//! The IMOD installation the processes see is built per test run: `bin/`
//! holds a link per command of our `imod` binary (the launcher dispatches on
//! the link name), plus the Python scripts eTomo runs that are not translated
//! (`imodkillgroup`), with `pylib` for them.
use imod_rs::imod::etomo::base_manager::{BaseManager, BaseManagerBase};
use imod_rs::imod::etomo::etomo_director;
use imod_rs::imod::etomo::process::base_process_manager::BaseProcessManager;
use imod_rs::imod::etomo::process::process_interface::{ProcessResultDisplayRef, ProcessSeriesRef};
use imod_rs::imod::etomo::process::system_program;
use imod_rs::imod::etomo::storage::storable::Storable;
use imod_rs::imod::etomo::r#type::axis_id::AxisID;
use imod_rs::imod::etomo::r#type::base_meta_data::BaseMetaData;
use imod_rs::imod::etomo::r#type::interface_type::InterfaceType;
use imod_rs::imod::etomo::r#type::process_end_state::ProcessEndState;
use imod_rs::imod::etomo::r#type::process_name::ProcessName;
use imod_rs::imod::etomo::util::event_queue::invoke_and_wait;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// What `processDone` was told.
#[derive(Debug, Clone)]
struct Done {
    thread_name: String,
    exit_value: i32,
    process_name: Option<String>,
    end_state: Option<ProcessEndState>,
    failed: bool,
}

struct TestManager {
    base: BaseManagerBase,
    process_manager: OnceLock<&'static BaseProcessManager>,
    done: Mutex<Sender<Done>>,
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
    fn get_main_panel(
        &self,
    ) -> Option<std::rc::Rc<dyn imod_rs::imod::etomo::ui::swing::main_panel::MainPanelVirtual>>
    {
        None
    }
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_manager.get().copied()
    }
    fn get_storables_with_offset(
        &self,
        _offset: i32,
    ) -> Option<Vec<Option<&'static dyn Storable>>> {
        None
    }
    fn get_name(&self) -> Option<String> {
        Some("test".to_owned())
    }
    #[allow(clippy::too_many_arguments)]
    fn process_done(
        &self,
        thread_name: Option<&str>,
        exit_value: i32,
        process_name: Option<ProcessName>,
        axis_id: Option<AxisID>,
        force_next_process: bool,
        end_state: Option<ProcessEndState>,
        status_string: Option<&str>,
        failed: bool,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        non_blocking: bool,
    ) {
        let _ = (
            force_next_process,
            status_string,
            process_result_display,
            process_series,
        );
        let _ = self.done.lock().unwrap().send(Done {
            thread_name: thread_name.unwrap_or("null").to_owned(),
            exit_value,
            process_name: process_name.map(|name| name.to_string()),
            end_state,
            failed,
        });
        // The base class's bookkeeping: clears the thread name and unblocks.
        if let Some(process_manager) = self.get_process_manager() {
            process_manager.unblock_axis(axis_id.unwrap_or(AxisID::Only));
        }
        let _ = non_blocking;
    }
}

/// An IMOD installation whose `bin` is our binary under each command's name.
fn imod_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let root =
            std::env::temp_dir().join(format!("imod-rs-etomo-launch-{}", std::process::id()));
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let imod = PathBuf::from(env!("CARGO_BIN_EXE_imod"));
        for command in imod_rs::imod::commands::COMMANDS {
            let _ = std::os::unix::fs::symlink(&imod, bin.join(command.name));
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD");
        let _ = std::os::unix::fs::symlink(
            source.join("pysrc/imodkillgroup"),
            bin.join("imodkillgroup"),
        );
        let _ = std::os::unix::fs::symlink(source.join("pysrc"), root.join("pylib"));
        let _ = std::os::unix::fs::symlink(source.join("com"), root.join("com"));
        let _ = std::os::unix::fs::symlink(source.join("autodoc"), root.join("autodoc"));
        // SAFETY: set once, before any thread of this test binary reads it.
        unsafe {
            std::env::set_var("IMOD_DIR", &root);
            std::env::set_var("AUTODOC_DIR", root.join("autodoc"));
        }
        system_program::set_imod_executable(imod);
        etomo_director::INSTANCE.init_imod_directory();
        root
    })
}

/// A manager over a fresh dataset directory, and the receiver of its
/// `processDone` calls.
fn manager(
    name: &str,
) -> (
    &'static TestManager,
    &'static BaseProcessManager,
    PathBuf,
    Receiver<Done>,
) {
    imod_dir();
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-etomo-launch-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (sender, receiver) = channel();
    // Java builds its managers on the event dispatch thread.
    let property_user_dir = dir.to_str().unwrap().to_owned();
    let (manager, process_manager) = invoke_and_wait(move || {
        let manager: &'static TestManager = Box::leak(Box::new(TestManager {
            base: BaseManagerBase::initial(),
            process_manager: OnceLock::new(),
            done: Mutex::new(sender),
        }));
        manager.base_manager();
        manager.set_property_user_dir(Some(&property_user_dir));
        let process_manager: &'static BaseProcessManager =
            Box::leak(Box::new(BaseProcessManager::new(manager)));
        let _ = manager.process_manager.set(process_manager);
        // The Java managers unblock the axis once they are ready to run.
        process_manager.unblock_axis(AxisID::Only);
        (manager, process_manager)
    });
    (manager, process_manager, dir, receiver)
}

fn wait(receiver: &Receiver<Done>) -> Done {
    receiver
        .recv_timeout(Duration::from_secs(60))
        .expect("processDone within a minute")
}

#[test]
fn com_script_runs_through_vmstopy_and_the_runner() {
    let (manager, process_manager, dir, done) = manager("com");
    std::fs::write(
        dir.join("sample.com"),
        "# a command file\n$echo2 hello from sample\n",
    )
    .unwrap();
    let process = process_manager
        .start_com_script_resumable("sample.com", None, AxisID::Only, None, None, false)
        .expect("axis is free");
    manager.set_thread_name(Some(&process.get_name()), Some(AxisID::Only));
    let result = wait(&done);
    assert_eq!(result.thread_name, process.get_name());
    assert_eq!(result.exit_value, 0, "{result:?}");
    assert!(!result.failed);
    let log = std::fs::read_to_string(dir.join("sample.log")).unwrap();
    assert!(log.contains("hello from sample"), "{log}");
    assert!(log.contains("SUCCESSFULLY COMPLETED"), "{log}");
    // The axis is free again.
    assert!(process_manager.is_axis_busy(AxisID::Only, None).is_ok());
}

#[test]
fn failing_com_script_reports_failure() {
    let (manager, process_manager, dir, done) = manager("fail");
    std::fs::write(
        dir.join("tomopitch.com"),
        "$header -size no-such-file.mrc\n",
    )
    .unwrap();
    let process = process_manager
        .start_com_script_resumable("tomopitch.com", None, AxisID::Only, None, None, false)
        .expect("axis is free");
    manager.set_thread_name(Some(&process.get_name()), Some(AxisID::Only));
    let result = wait(&done);
    assert_ne!(result.exit_value, 0, "{result:?}");
    assert!(result.failed);
    let log = std::fs::read_to_string(dir.join("tomopitch.log")).unwrap();
    assert!(log.contains("ERROR"), "{log}");
}

#[test]
fn busy_axis_refuses_a_second_process() {
    let (manager, process_manager, dir, done) = manager("busy");
    std::fs::write(dir.join("tilt.com"), "$sleep 2\n").unwrap();
    let process = process_manager
        .start_com_script_resumable("tilt.com", None, AxisID::Only, None, None, false)
        .expect("axis is free");
    manager.set_thread_name(Some(&process.get_name()), Some(AxisID::Only));
    let second = process_manager.start_com_script_resumable(
        "tilt.com",
        None,
        AxisID::Only,
        None,
        None,
        false,
    );
    assert!(second.is_err());
    assert!(
        second
            .err()
            .unwrap()
            .0
            .starts_with("A process is already executing")
    );
    let result = wait(&done);
    assert_eq!(result.exit_value, 0, "{result:?}");
}

#[test]
fn background_process_runs_our_program() {
    let (manager, process_manager, dir, done) = manager("background");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_imod"))
        .args(["raw2mrc", "-x", "4", "-y", "3", "-z", "2", "-t", "byte"])
        .arg("/dev/zero")
        .arg(dir.join("zero.mrc"))
        .output()
        .unwrap();
    assert!(status.status.success(), "{status:?}");
    let process = process_manager
        .start_background_process_array(
            vec![
                "header".to_owned(),
                "-size".to_owned(),
                "zero.mrc".to_owned(),
            ],
            AxisID::Only,
            None,
            None,
        )
        .expect("axis is free");
    manager.set_thread_name(Some(&process.get_name()), Some(AxisID::Only));
    let result = wait(&done);
    assert_eq!(result.exit_value, 0, "{result:?}");
    let output =
        imod_rs::imod::etomo::process::process_interface::SystemProcessInterface::get_std_output(
            &*process,
        )
        .unwrap();
    assert_eq!(output.len(), 1, "{output:?}");
    assert_eq!(
        output[0].split_whitespace().collect::<Vec<_>>(),
        ["4", "3", "2"]
    );
}

#[test]
fn killed_com_script_ends_killed() {
    let (manager, process_manager, dir, done) = manager("kill");
    std::fs::write(dir.join("align.com"), "$sleep 30\n$echo2 not reached\n").unwrap();
    let process = process_manager
        .start_com_script_resumable("align.com", None, AxisID::Only, None, None, false)
        .expect("axis is free");
    manager.set_thread_name(Some(&process.get_name()), Some(AxisID::Only));
    // Wait for ParsePID to see the runner's PID, as the Kill button would.
    let pid_seen = (0..100).any(|_| {
        std::thread::sleep(Duration::from_millis(100));
        !imod_rs::imod::etomo::process::process_interface::SystemProcessInterface::get_shell_process_id(&*process).is_empty()
    });
    assert!(pid_seen);
    process_manager.kill(AxisID::Only);
    let result = wait(&done);
    assert_eq!(
        result.end_state,
        Some(ProcessEndState::Killed),
        "{result:?}"
    );
    let log = std::fs::read_to_string(dir.join("align.log")).unwrap_or_default();
    assert!(!log.contains("not reached"), "{log}");
}
