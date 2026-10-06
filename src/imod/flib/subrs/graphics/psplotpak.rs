//! Translation of `IMOD/flib/subrs/graphics/psplotpak.f90`: the package
//! that draws with multiple line thicknesses through the PostScript
//! routines, with module `psparams` and `pltout`.
//!
//! Module `psparams`'s variables are thread-local cells (the Fortran
//! program runs on one thread).

use super::psf::{psclose, psfirstpoint, psframe, pslinewidth, psopen, pspoint, psvector};
use super::trnc::trnc;
use crate::imod::libcfshr::b3dutil::{exit, imod_backup_file};
use std::cell::{Cell, RefCell};
use std::io::Write as _;

thread_local! {
    /// Module `psparams` (`psplotpak.f90:9-14`).
    pub static N_THICK: Cell<i32> = const { Cell::new(1) };
    pub static IF_PS_OPEN: Cell<i32> = const { Cell::new(0) };
    pub static WIDTH: Cell<f32> = const { Cell::new(7.5) };
    pub static UNITS_PER_INCH: Cell<f32> = const { Cell::new(300.) };
    pub static SAFE: Cell<f32> = const { Cell::new(0.00022) };
    pub static XCUR: Cell<f32> = const { Cell::new(0.) };
    pub static YCUR: Cell<f32> = const { Cell::new(0.) };
    pub static UNIT_LEN: Cell<f32> = const { Cell::new(0.) };
    pub static EXTRA_LEN: Cell<f32> = const { Cell::new(0.) };
    pub static HALF_THICK: Cell<f32> = const { Cell::new(0.) };
    pub static SYM_SCALE: Cell<f32> = const { Cell::new(0.15) };
    /// `character*320 filename/'gmeta.ps'/`, trailing blanks dropped.
    pub static FILENAME: RefCell<String> = RefCell::new("gmeta.ps".to_owned());
}

/// Original `psSetup` (`psplotpak.f90:16`): opens the file on first use,
/// sets the line thickness, and either sets (`ifset != 0`) or returns the
/// page width, units per inch and safety offset.
pub fn ps_setup(
    ithick_set: i32,
    width_set: &mut f32,
    upi_set: &mut f32,
    safe_set: &mut f32,
    ifset: i32,
) {
    if IF_PS_OPEN.get() == 0 {
        let filename = FILENAME.with_borrow(|name| name.clone());
        let ierr = imod_backup_file(&filename);
        if ierr != 0 {
            let _ = write!(
                std::io::stdout(),
                "  Error attempting to rename existing file {filename}\n"
            );
        }
        psopen(filename.as_bytes(), 0.5, 1.75, UNITS_PER_INCH.get());
        IF_PS_OPEN.set(1);
    }
    N_THICK.set(ithick_set);
    let n_thick = N_THICK.get();
    let mut f_thick = n_thick as f32 - 1.;
    if n_thick == 2 {
        f_thick = 1.5;
    }
    if n_thick < 2 {
        f_thick = 1.;
    }
    pslinewidth(f_thick);
    if ifset != 0 {
        if *width_set > 0. {
            WIDTH.set(*width_set);
        }
        if *upi_set > 0. {
            UNITS_PER_INCH.set(*upi_set);
        }
        if *safe_set >= 0. {
            SAFE.set(*safe_set);
        }
    } else {
        *width_set = WIDTH.get();
        *upi_set = UNITS_PER_INCH.get();
        *safe_set = SAFE.get();
    }
    XCUR.set(0.);
    YCUR.set(0.);
    let unit_len = 1. / UNITS_PER_INCH.get();
    UNIT_LEN.set(unit_len);
    EXTRA_LEN.set(0f32.max(unit_len * (n_thick - 3) as f32));
    HALF_THICK.set(0.5 * unit_len * (n_thick - 1) as f32);
}

/// [`ps_setup`] with the three values only returned (`call psSetup(ithick,
/// c1, c2, c3, 0)`): returns the units per inch.
pub fn ps_setup_thick(ithick_set: i32) -> f32 {
    let (mut c1, mut c2, mut c3) = (0., 0., 0.);
    ps_setup(ithick_set, &mut c1, &mut c2, &mut c3, 0);
    c2
}

/// Original `psSetFilename` (`psplotpak.f90:57`).
pub fn ps_set_filename(new_name: &str) {
    ps_pak_off();
    FILENAME.with_borrow_mut(|name| *name = new_name.trim_end_matches(' ').to_owned());
}

/// Original `psPointAbs` (`psplotpak.f90:65`).
pub fn ps_point_abs(x: f32, y: f32) {
    XCUR.set(x);
    YCUR.set(y);
    pspoint(trnc(x), trnc(y));
}

/// Original `psMoveAbs` (`psplotpak.f90:75`).
pub fn ps_move_abs(xin: f32, yin: f32) {
    XCUR.set(xin);
    YCUR.set(yin);
    if N_THICK.get() == 1 {
        psfirstpoint(trnc(XCUR.get()), trnc(YCUR.get()));
    }
}

/// Original `psMoveInc` (`psplotpak.f90:85`).
///
/// Fixed in translation (BUGS.md, `psMoveInc`): the source moves to
/// `(xcur + dx, xcur + dy)`; here the Y position is `ycur + dy`.
pub fn ps_move_inc(dx: f32, dy: f32) {
    ps_move_abs(XCUR.get() + dx, YCUR.get() + dy);
}

/// Original `psVectInc` (`psplotpak.f90:92`).
pub fn ps_vect_inc(dx: f32, dy: f32) {
    ps_vect_abs(XCUR.get() + dx, YCUR.get() + dy);
}

/// Original `psVectAbs` (`psplotpak.f90:100`).
pub fn ps_vect_abs(x: f32, y: f32) {
    if N_THICK.get() <= 1 {
        psvector(trnc(x), trnc(y));
    } else {
        let (xcur, ycur) = (XCUR.get(), YCUR.get());
        let ddx = x - xcur;
        let ddy = y - ycur;
        if ddx != 0. || ddy != 0. {
            let extra_len = EXTRA_LEN.get();
            let ddlen = (ddx * ddx + ddy * ddy).sqrt();
            let xinc = 0.5 * ddx * extra_len / ddlen;
            let yinc = 0.5 * ddy * extra_len / ddlen;
            let xs = trnc(xcur - xinc);
            let xe = trnc(x + xinc);
            let ys = trnc(ycur - yinc);
            let ye = trnc(y + yinc);
            psfirstpoint(xs, ys);
            psvector(xe, ye);
        }
    }
    XCUR.set(x);
    YCUR.set(y);
}

/// Original `psExit` (`psplotpak.f90:126`).
pub fn ps_exit() -> ! {
    ps_pak_off();
    exit(0);
}

/// Original `psPakOff` (`psplotpak.f90:132`).
pub fn ps_pak_off() {
    if IF_PS_OPEN.get() == 0 {
        return;
    }
    psframe();
    psclose();
    IF_PS_OPEN.set(0);
}

/// Original `pltout` (`psplotpak.f90:143`): plots out the data to the screen
/// or the printer through IMOD's `imodpsview` script (an external program,
/// run as the source runs it).
///
/// Fixed in translation (BUGS.md, `pltout`): the source cuts `IMOD_DIR` to
/// 80 characters and dies with a runtime error when the command does not fit
/// in 120; here neither is cut.
pub fn pltout(meta_screen: i32) {
    ps_pak_off();
    //
    // 10/28/03: switch to calling imodpsview for printing too and run
    // tcsh explicitly
    //
    let outcom = if meta_screen == 0 {
        "imodpsview -p"
    } else {
        "imodpsview"
    };
    let Ok(imod_path) = std::env::var("IMOD_DIR") else {
        println!(" impak failed to get IMOD_DIR environment variable for running imodpsview");
        return;
    };
    let imodshell = std::env::var("IMOD_CSHELL").unwrap_or_else(|_| "tcsh".to_owned());
    let filename = FILENAME.with_borrow(|name| name.clone());
    let comstr = format!(
        "{} -f {}/bin/{} {}",
        imodshell.trim_end(),
        imod_path.trim_end(),
        outcom,
        filename.trim_end()
    );
    let _ = std::io::stdout().flush();
    // `call system(comstr)`
    let mut shell = std::process::Command::new("/bin/sh");
    std::os::unix::process::CommandExt::arg0(&mut shell, "sh");
    let _ = shell.arg("-c").arg(&comstr).status();
    if meta_screen != 0 {
        // `101 format(/,' WARNING: ...')`
        print!(
            "\n WARNING: If you start making more plots, a new plot file will be started,\n          the current file will become a backup ({}~),\n          and a previous backup will be deleted.\n",
            filename.trim_end()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BUGS.md` (`psMoveInc`), defined behaviour: the move is relative to
    /// the current Y for Y (the source adds `dy` to the current X).
    #[test]
    fn move_inc_moves_from_the_current_point() {
        XCUR.set(1.);
        YCUR.set(5.);
        ps_move_inc(0.5, 0.25);
        assert_eq!((XCUR.get(), YCUR.get()), (1.5, 5.25));
    }
}
