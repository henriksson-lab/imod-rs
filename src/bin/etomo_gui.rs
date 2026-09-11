//! Slint launcher for `IMOD/Etomo/src/etomo/EtomoDirector.java`.
//!
//! Java `EtomoDirector.main` schedules `Setup`, which creates a `MainFrame`
//! through `UIHarness` before the manager-specific panels are opened.  This
//! optional launcher follows that director-to-`UIHarness` startup boundary
//! without changing the non-GUI `etomo` command's current behaviour.

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let mut director = imod_rs::imod::etomo::etomo_director::EtomoDirector::new();
    if let Err(error) = director.main_gui(&arguments) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
