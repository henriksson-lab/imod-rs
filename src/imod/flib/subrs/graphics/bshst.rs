//! Translation of `IMOD/flib/subrs/graphics/bshst.f`: `bshst`, which plots
//! histograms of values in one or more groups on the screen, the terminal
//! and the PostScript plot, `pearev` and `gnhst`.
//!
//! `bshst`'s `SAVE`d locals and the common `/hstp/` are thread-local (the
//! Fortran program runs on one thread).  Fortran unit 8, the output file
//! that `bshst` and `bsplt` append typed-out values to, is [`UNIT8`].
//!
//! Fixed in translation (BUGS.md, `bshst`): `gnhst`'s fixed arrays
//! (`iz(100000)`, `jz(5000)`) and `bshst`'s `iginv(30)` and `xtick(310)` are
//! sized to what is entered instead of being written past; `gnhst` passes
//! `col(1)` without setting it, which the curve option then reads (here it
//! is 0, so there are no curves); a NaN value, which the source puts in bin
//! `INT_MIN`, goes in bin 1; the means-only path (`iskp < 0`) no longer
//! draws the saved kernel curve with this call's unset scaling; and a
//! symbol number past the 22-entry terminal symbol table prints a blank
//! instead of reading past it; a plot number past 16 reads past the six
//! plot positions (`xls`, `yls`), here the positions repeat; and writing the
//! typed-out values to a file fails in the source (it reads the file to its
//! end to append, and gfortran then refuses the write: "Sequential READ or
//! WRITE not allowed after EOF marker", exit 2), here they are appended.

use super::flnam::flnam;
use super::label_axis::label_axis;
use super::minmax::minmax;
use super::psdashline::ps_dash_interactive;
use super::psf::psframe;
use super::psgrid::{ps_grid_line, ps_log_grid};
use super::psmisc::ps_misc_items;
use super::psplotpak::{pltout, ps_exit, ps_move_abs, ps_setup, ps_setup_thick, ps_vect_abs};
use super::pssymbol::{ps_sym_size, ps_symbol};
use super::screenpak::{
    scrn_close, scrn_erase, scrn_grid_line, scrn_move_abs, scrn_symbol, scrn_update, scrn_vect_abs,
};
use crate::imod::flib::subrs::compat::gfortran_rt::{
    cvttss2si, format_f, format_i, ld_int, read_list_stdin,
};
use crate::imod::flib::subrs::hvem::b3dxor::b3dxor;
use crate::imod::flib::subrs::hvem::frefor::ListItem;
use crate::imod::flib::subrs::hvem::mypause::mypause;
use std::cell::{Cell, RefCell};
use std::io::{BufRead as _, Write as _};

/// `bshst`'s `save xloin,dxin,nbins,ifplt,ifnewp,fsc,iskp,binsin`.
#[derive(Clone, Copy, Default)]
struct BshstSave {
    xloin: f32,
    dxin: f32,
    nbins: i32,
    ifplt: i32,
    ifnewp: i32,
    fsc: f32,
    iskp: i32,
    binsin: f32,
}

thread_local! {
    static SAVED: Cell<BshstSave> = Cell::new(BshstSave::default());
    /// `common /hstp/ifgnh` with `data ifgnh/0/`.
    static IFGNH: Cell<i32> = const { Cell::new(0) };
    /// Fortran unit 8: the file typed-out values are appended to.
    pub static UNIT8: RefCell<Option<std::fs::File>> = const { RefCell::new(None) };
}

/// `data hisymb/'-','|','<','>','/','\\','X','+','O','@','v','^','3','4',
/// '5','6','7','8','9',' ',' ',' '/` (`-fbackslash`: one backslash).
const HISYMB: [u8; 22] = *b"-|<>/\\X+O@v^3456789   ";

/// `hisymb(max(1, isym))`, blank past the table (see the module notes).
fn hisymb(isym: i32) -> u8 {
    let k = 1.max(isym) as usize;
    if k <= HISYMB.len() {
        HISYMB[k - 1]
    } else {
        b' '
    }
}

/// `write(iout, ...)` to standard output (6) or unit 8.
pub fn write_unit(iout: i32, bytes: &[u8]) {
    if iout == 8 {
        UNIT8.with_borrow_mut(|file| {
            if let Some(file) = file.as_mut() {
                let _ = file.write_all(bytes);
            }
        });
    } else {
        let _ = std::io::stdout().write_all(bytes);
    }
}

/// `close(8)`, `open(8, file = name, err = ..., status = 'unknown')`, then
/// reading to the end so that writes append: `false` for the `ERR=` branch.
pub fn open_unit8_append(name: &[u8]) -> bool {
    let end = name.iter().rposition(|&c| c != b' ').map_or(0, |p| p + 1);
    let path = String::from_utf8_lossy(&name[..end]).into_owned();
    UNIT8.with_borrow_mut(|file| *file = None);
    match std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&path)
    {
        Ok(file) => {
            // `25 read(8, '(a4)', end = 26) name; go to 25`
            let mut reader = std::io::BufReader::new(&file);
            let mut line = Vec::new();
            while reader
                .read_until(b'\n', &mut line)
                .map(|n| n > 0)
                .unwrap_or(false)
            {
                line.clear();
            }
            UNIT8.with_borrow_mut(|slot| *slot = Some(file));
            true
        }
        Err(_) => false,
    }
}

/// Original `bshst` (`bshst.f:199`).  `iz` and `jz` are the caller's work
/// arrays (`iz` also holds curve starts); they are grown as needed.
#[allow(clippy::too_many_arguments)]
pub fn bshst(
    namx: &[i32],
    xx: &[f32],
    ngx: &[i32],
    nx: i32,
    nsymb: &[i32],
    ngrps: i32,
    iz: &mut Vec<i32>,
    jz: &mut Vec<i32>,
    irec: &[i32],
    col: &[f32],
    iflog: i32,
) {
    let xls: [f32; 6] = [0., 0., 0., 4., 4., 4.];
    let yls: [f32; 6] = [5.2, 2.6, 0., 5.2, 2.6, 0.];
    let mut sv = SAVED.get();
    let nxu = nx.max(0) as usize;
    if iz.len() < nxu.max(1) {
        iz.resize(nxu.max(1), 0);
    }
    let mut out = std::io::stdout();
    let nkpts: i32 = 200;
    let (mut xmin, mut xmax) = (0f32, 0f32);
    minmax(xx, nx, &mut xmin, &mut xmax);
    let xrange = xmax - xmin;
    let mut iftyp: i32 = 0;
    let _ =
        out.write_all(b" histograms (1), means only (-1), skip (0), exit (-123), plot out (209): ");
    read_list_stdin(&mut [ListItem::Integer(&mut sv.iskp)]);
    SAVED.set(sv);
    if sv.iskp == 209 || sv.iskp == 208 {
        pltout(209 - sv.iskp);
        return;
    }
    if sv.iskp == -123 {
        scrn_close();
        ps_exit();
    }
    // `if(iskp)26,82,11`
    if sv.iskp == 0 {
        return;
    }
    let mut start_at_26 = sv.iskp < 0;

    // Locals that stay set across the `go to 11` loop
    let mut xlo: f32 = 0.;
    let mut dx: f32 = 0.;
    let mut xhi: f32 = 0.;
    let mut ibmax: i32 = 0;
    let mut xscal: f32 = 0.;
    let mut yscal: f32 = 0.;
    let mut nlines: i32 = 0;
    let mut iout: i32 = 6;
    loop {
        if !start_at_26 {
            // 11
            let _ = write!(
                out,
                " min, max:{}{},  range:{}\n",
                format_f(xmin as f64, 10, 3),
                format_f(xmax as f64, 10, 3),
                format_f(xrange as f64, 10, 3)
            );
            if iflog != 0 {
                let xlnmn = 10f32.powf(xmin);
                let xlnmx = 10f32.powf(xmax);
                let _ = write!(
                    out,
                    " linear values:{}{}\n",
                    format_f(xlnmn as f64, 10, 3),
                    format_f(xlnmx as f64, 10, 3)
                );
            }
            // 14
            loop {
                let _ = out
                    .write_all(b" lowest value (linear even if log), bin width or -h, # of bins: ");
                read_list_stdin(&mut [
                    ListItem::Real(&mut sv.xloin),
                    ListItem::Real(&mut sv.dxin),
                    ListItem::Real(&mut sv.binsin),
                ]);
                SAVED.set(sv);
                sv.nbins = cvttss2si(sv.binsin);
                SAVED.set(sv);
                xlo = sv.xloin;
                dx = sv.dxin.abs();
                if dx == 0. {
                    if sv.nbins < 5 {
                        continue;
                    }
                    dx = xrange / (sv.nbins - 3) as f32;
                    xlo = xmin - 1.5 * dx;
                } else {
                    if iflog != 0 {
                        xlo = sv.xloin.log10();
                    }
                    if xlo > xmin || dx <= 0. || sv.nbins <= 0 {
                        continue;
                    }
                }
                break;
            }
            let nbins = sv.nbins;
            xhi = xlo + dx * nbins as f32;
            if sv.dxin < 0. {
                xhi = xlo + dx * sv.binsin;
            }
            if jz.len() < nbins as usize {
                jz.resize(nbins as usize, 0);
            }
            for i in 1..=nbins as usize {
                jz[i - 1] = 0;
            }
            //
            // DNM 5/11/01: add a bit to prevent horrible variability in rounding
            // on PC when data are an even multiple of bin size
            //
            ibmax = 0;
            for i in 1..=nxu {
                let mut ibin = cvttss2si((xx[i - 1] - xlo) / dx + 1.0001);
                if ibin > nbins {
                    ibin = nbins;
                }
                if ibin < 1 {
                    ibin = 1;
                }
                iz[i - 1] = ibin;
                jz[(ibin - 1) as usize] += 1;
                ibmax = ibmax.max(jz[(ibin - 1) as usize]);
            }
            let _ = write!(out, " highest bin is{}\n", ld_int(ibmax));
            let _ =
                out.write_all(b" full scale count (0 terminal plot, -1 labels also, -2 counts): ");
            read_list_stdin(&mut [ListItem::Real(&mut sv.fsc)]);
            SAVED.set(sv);
            scrn_erase(-1);
            if sv.fsc <= 0. {
                let mut line = vec![b' '; 240];
                for ibin in 1..=nbins {
                    let xlin = xlo + dx * (ibin - 1) as f32;
                    let mut linept: usize = 1;
                    if sv.fsc < 0. {
                        let text = format!("{}:   ", format_f(xlin as f64, 11, 4));
                        line[1..16].copy_from_slice(&text.as_bytes()[..15]);
                        linept = 16;
                    }
                    if sv.fsc == -2. {
                        let text = format_i(jz[(ibin - 1) as usize], 4);
                        line[16..20].copy_from_slice(text.as_bytes());
                        linept = 20;
                    } else {
                        for ng in 1..=ngrps {
                            for i in 1..=nxu {
                                if ngx[i - 1] == ng && iz[i - 1] == ibin && linept < 240 {
                                    linept += 1;
                                    line[linept - 1] = hisymb(nsymb[(ngx[i - 1] - 1) as usize]);
                                }
                            }
                        }
                    }
                    let _ = out.write_all(&line[..linept]);
                    let _ = out.write_all(b"\n");
                }
                mypause(&HISYMB);
            } else {
                scrn_grid_line(10, 10, 0, 100, 10);
                xscal = (1000 / nbins) as f32;
                yscal = 1000. / sv.fsc;
                let idx = cvttss2si(xscal);
                scrn_grid_line(10, 10, idx, 0, nbins);
            }
            for i in 1..=nbins as usize {
                jz[i - 1] = 0;
            }
            let _ = out
                .write_all(b" Draw n lines (n+1), type out (1), output to file (-1), or not (0): ");
            read_list_stdin(&mut [ListItem::Integer(&mut iftyp)]);
            nlines = 0;
            if iftyp > 1 {
                nlines = iftyp - 1;
                iftyp = 0;
            }
            iout = 6;
            if iftyp < 0 {
                loop {
                    // 24
                    iout = 8;
                    let _ = out.write_all(b" enter output file name\n");
                    let namef = flnam(80, 0, "0");
                    if open_unit8_append(&namef) {
                        break;
                    }
                }
            }
        }
        start_at_26 = false;

        // 26
        let iskp = sv.iskp;
        let fsc = sv.fsc;
        let dxin = sv.dxin;
        let nbins = sv.nbins;
        for ng in 1..=ngrps {
            let mut sx: f32 = 0.;
            let mut sxs: f32 = 0.;
            let mut np: i32 = 0;
            for i in 1..=nxu {
                if ngx[i - 1] == ng {
                    sx += xx[i - 1];
                    sxs += xx[i - 1] * xx[i - 1];
                    np += 1;
                    if iftyp != 0 {
                        if IFGNH.get() == 0 {
                            // `103 format(i3,2i7,2f10.3)`
                            let text = format!(
                                "{}{}{}{}{}\n",
                                format_i(ng, 3),
                                format_i(namx[i - 1], 7),
                                format_i(irec[i - 1], 7),
                                format_f(xx[i - 1] as f64, 10, 3),
                                format_f(col[i - 1] as f64, 10, 3)
                            );
                            write_unit(iout, text.as_bytes());
                        } else {
                            // `203 format(i3,f10.3)`
                            let text = format!(
                                "{}{}\n",
                                format_i(ng, 3),
                                format_f(xx[i - 1] as f64, 10, 3)
                            );
                            write_unit(iout, text.as_bytes());
                        }
                    }
                    if iskp >= 0 && fsc > 0. && dxin >= 0. {
                        let ibin = iz[i - 1];
                        jz[(ibin - 1) as usize] += 1;
                        let ix = cvttss2si(xscal * (ibin as f32 - 0.5) + 10.);
                        let iy = cvttss2si(yscal * (jz[(ibin - 1) as usize] as f32 - 0.5) + 10.);
                        let mut isymbt = nsymb[(ng - 1) as usize];
                        if isymbt < 0 {
                            isymbt = -(i as i32);
                        }
                        scrn_symbol(ix, iy, &mut isymbt);
                    }
                }
            }
            if iskp >= 0 && dxin < 0. && fsc > 0. {
                for ixk in 0..=nkpts {
                    let xk = xlo + (ixk as f32 / nkpts as f32) * (xhi - xlo);
                    let ix = cvttss2si(xscal * (ixk as f32 / nkpts as f32) * nbins as f32 + 10.);
                    let mut ysum: f32 = 0.;
                    for i in 1..=nxu {
                        if ngx[i - 1] <= ng && (xx[i - 1] - xk).abs() < dx {
                            let t = (xx[i - 1] - xk) / dx;
                            let u = 1. - t * t;
                            ysum += u * u * u;
                        }
                    }
                    ysum = ysum * 35. / 32.;
                    let iy = cvttss2si(yscal * ysum + 10.);
                    if ixk == 0 {
                        scrn_move_abs(ix, iy);
                    }
                    scrn_vect_abs(ix, iy);
                }
            }
            if np > 0 {
                let xav = sx / np as f32;
                let mut xsd: f32 = 0.;
                if np > 1 {
                    xsd = ((sxs - np as f32 * (xav * xav)) / (np - 1) as f32).sqrt();
                }
                let sem = xsd / (1. * np as f32).sqrt();
                // `105 format(' group',i3,'  avg =',f10.3,',  sd =',f10.3,',  sem=',f10.3,',  n=',i4)`
                let _ = write!(
                    out,
                    " group{}  avg ={},  sd ={},  sem={},  n={}\n",
                    format_i(ng, 3),
                    format_f(xav as f64, 10, 3),
                    format_f(xsd as f64, 10, 3),
                    format_f(sem as f64, 10, 3),
                    format_i(np, 4)
                );
                if iskp >= 0 && fsc > 0. && dxin >= 0. {
                    scrn_move_abs(10, 10);
                    let mut ix = 10;
                    let mut iy = 10;
                    for ib in 1..=nbins {
                        iy = cvttss2si(yscal * jz[(ib - 1) as usize] as f32 + 10.);
                        scrn_vect_abs(ix, iy);
                        ix = cvttss2si(xscal * ib as f32 + 10.);
                        scrn_vect_abs(ix, iy);
                    }
                    let _ = iy;
                    scrn_vect_abs(ix, 10);
                }
            }
        }
        scrn_update(1);
        if iskp < 0 {
            return;
        }
        // `109 format(' x from',f9.2,' to',f9.2)`
        let _ = write!(
            out,
            " x from{} to{}\n",
            format_f(xlo as f64, 9, 2),
            format_f(xhi as f64, 9, 2)
        );
        let _ = out
            .write_all(b" 4-7 or 11-16 for plot (-# for no points), return (0), redisplay (10): ");
        read_list_stdin(&mut [ListItem::Integer(&mut sv.ifplt)]);
        SAVED.set(sv);
        let mut ifcrv = 0;
        if sv.ifplt.abs() > 100 {
            ifcrv = 1;
            // `ifplt=ifplt-isign(100,ifplt)`
            sv.ifplt -= if sv.ifplt >= 0 { 100 } else { -100 };
            SAVED.set(sv);
        }
        if sv.ifplt.abs() <= 3 {
            return;
        }
        if sv.ifplt == 10 {
            continue;
        }
        let mut iaplt = sv.ifplt.abs();
        let mut ngbelow: i32 = 0;
        let mut iginv: Vec<i32> = Vec::new();
        let (mut fscup, mut fscdown) = (0f32, 0f32);
        if sv.fsc.abs() < ibmax as f32 && iaplt != 7 {
            let _ = write!(
                out,
                " highest bin is{}, enter full scale count: ",
                format_i(ibmax, 4)
            );
            read_list_stdin(&mut [ListItem::Real(&mut sv.fsc)]);
            SAVED.set(sv);
        } else if iaplt == 7 {
            let _ = out.write_all(b" # of groups to put below axis: ");
            read_list_stdin(&mut [ListItem::Integer(&mut ngbelow)]);
            iginv = vec![0; ngbelow.max(0) as usize];
            let _ = out.write_all(b" Group #'s: ");
            {
                let mut items: Vec<ListItem> = iginv.iter_mut().map(ListItem::Integer).collect();
                read_list_stdin(&mut items);
            }
            // `maxup=maxbin` reads `maxbin` before the first pass sets it (a
            // local); the value is overwritten on the second pass
            let mut maxup = 0;
            let mut maxbin = 0;
            for nupdown in 1..=2 {
                maxup = maxbin;
                maxbin = 0;
                for i in 1..=nbins as usize {
                    jz[i - 1] = 0;
                }
                for i in 1..=nxu {
                    let mut ifbelow = 0;
                    for jb in 0..iginv.len() {
                        if ngx[i - 1] == iginv[jb] {
                            ifbelow = 1;
                        }
                    }
                    if b3dxor(nupdown == 1, ifbelow == 1) {
                        let b = (iz[i - 1] - 1) as usize;
                        jz[b] += 1;
                        maxbin = maxbin.max(jz[b]);
                    }
                }
            }
            let _ = write!(
                out,
                " highest bins up & down are{}{}, enter full scale counts: ",
                format_i(maxup, 4),
                format_i(maxbin, 4)
            );
            read_list_stdin(&mut [ListItem::Real(&mut fscup), ListItem::Real(&mut fscdown)]);
        }
        let ifimg = iaplt - 3;
        let mut defscl: f32 = 1.;
        let mut wthinch: f32 = 0.;
        if ifimg > 0 {
            let (mut c2, mut c3) = (0f32, 0f32);
            ps_setup(1, &mut wthinch, &mut c2, &mut c3, 0);
            defscl = 0.74 * wthinch / 7.5;
            iaplt -= 3;
        }
        let mut ntx: i32 = 10;
        let mut nty: i32 = 10;
        let mut xran = 6. * defscl;
        let mut yran = 4. * defscl;
        let mut xl: f32 = 0.;
        let mut yl: f32 = 0.;
        let mut symwid: f32 = 0.08;
        let mut tiksiz: f32 = 0.05;
        let mut ithsym: i32 = 1;
        let mut ithgrd: i32 = 1;
        let mut ithhis: i32 = 1;
        let mut ifbox: i32 = 0;
        let mut xaxofs: f32 = 0.;
        let mut nhists = 1;
        if iaplt > 7 {
            xran = 3. * wthinch / 7.5;
            yran = 2. * wthinch / 7.5;
            // Fixed in translation (module notes): a plot number past 16
            // reads past the six positions; here the positions repeat, as
            // the automatic advance after 16 (label 81) does.
            let pos = ((iaplt - 8) % 6) as usize;
            xl = xls[pos] * wthinch / 7.5;
            yl = yls[pos] * wthinch / 7.5;
        } else if iaplt < 2 {
            yl = 5. * defscl;
        } else if iaplt > 2 {
            let _ = out.write_all(b" X and Y size, lower left X and Y, # ticks X and Y: ");
            read_list_stdin(&mut [
                ListItem::Real(&mut xran),
                ListItem::Real(&mut yran),
                ListItem::Real(&mut xl),
                ListItem::Real(&mut yl),
                ListItem::Integer(&mut ntx),
                ListItem::Integer(&mut nty),
            ]);
            if ifimg > 0 {
                let _ = out.write_all(
                    b" symbol and tick size, grid, histogram and symbol thickness, 1 for box: ",
                );
                read_list_stdin(&mut [
                    ListItem::Real(&mut symwid),
                    ListItem::Real(&mut tiksiz),
                    ListItem::Integer(&mut ithgrd),
                    ListItem::Integer(&mut ithhis),
                    ListItem::Integer(&mut ithsym),
                    ListItem::Integer(&mut ifbox),
                ]);
            }
            if yl < 0. {
                yl = -yl;
                let _ = out.write_all(b" Amount to offset x axis in y: ");
                read_list_stdin(&mut [ListItem::Real(&mut xaxofs)]);
            }
        }
        let mut xll: f32 = 0.;
        let mut yll: f32 = 0.;
        let mut xaxisy: f32 = 0.;
        let mut ylorig: f32 = 0.;
        let mut xtick: Vec<f32> = vec![0.; 310];
        if ifimg > 0 {
            let _ = write!(
                out,
                " new page (0 or 1: , gives{})?: ",
                format_i(sv.ifnewp, 2)
            );
            read_list_stdin(&mut [ListItem::Integer(&mut sv.ifnewp)]);
            SAVED.set(sv);
            if sv.ifnewp > 0 {
                psframe();
            }
            ps_sym_size(symwid);
            yscal = yran / sv.fsc.abs();
            xll = xl + 0.1;
            yll = yl + 0.1;
            let iabntx = ntx.abs();
            ps_setup_thick(ithgrd);
            xaxisy = yll - xaxofs;
            ylorig = yll;
            if iflog == 0 {
                ps_grid_line(xll, xaxisy, xran, 0., ntx, tiksiz);
                if ifbox != 0 {
                    ps_grid_line(xll, yll + yran + xaxofs, xran, 0., ntx, -tiksiz);
                }
            } else {
                let _ = write!(out, "{} tick values: ", format_i(iabntx, 3));
                if xtick.len() < iabntx as usize {
                    xtick.resize(iabntx as usize, 0.);
                }
                {
                    let mut items: Vec<ListItem> = xtick[..iabntx as usize]
                        .iter_mut()
                        .map(ListItem::Real)
                        .collect();
                    read_list_stdin(&mut items);
                }
                let xscl = xran / (xhi - xlo);
                ps_log_grid(xll, xaxisy, xscl, 0., &xtick, ntx, tiksiz);
                if ifbox != 0 {
                    ps_log_grid(xll, yll + yran + xaxofs, xscl, 0., &xtick, ntx, -tiksiz);
                }
            }
            xscal = xran / nbins as f32;
            ps_grid_line(xll, yll, 0., yran, nty, tiksiz);
            if ifbox != 0 {
                ps_grid_line(xll + xran, yll, 0., yran, nty, -tiksiz);
            }
            if iaplt == 4 {
                yscal = yran / (fscup + fscdown);
                yll += yscal * fscdown;
                ps_move_abs(xll, yll);
                ps_vect_abs(xll + xran, yll);
                nhists = 2;
            }
        }
        for nupdown in 1..=nhists {
            if nupdown == 2 {
                yscal = -yscal;
            }
            for i in 1..=nbins as usize {
                jz[i - 1] = 0;
            }
            for ng in 1..=ngrps {
                let mut ifbelow = 0;
                let mut np = 0;
                let upi = ps_setup_thick(ithsym);
                let adjthk = 0.5 * (ithhis - 1) as f32 / upi;
                if iaplt == 4 {
                    for &g in &iginv {
                        if ng == g {
                            ifbelow = 1;
                        }
                    }
                }
                if b3dxor(nupdown == 1, ifbelow == 1) {
                    for i in 1..=nxu {
                        if ngx[i - 1] == ng {
                            np += 1;
                            let ibin = iz[i - 1];
                            jz[(ibin - 1) as usize] += 1;
                            if sv.ifplt >= 0 {
                                let rx = xscal * (ibin as f32 - 0.5) + xll + adjthk;
                                let ry =
                                    yscal * (jz[(ibin - 1) as usize] as f32 - 0.5) + yll + adjthk;
                                let mut isymbt = nsymb[(ng - 1) as usize];
                                if ifimg > 0 && isymbt == -2 {
                                    let yfilhi = yscal * jz[(ibin - 1) as usize] as f32 + yll;
                                    let yfillo = yfilhi - yscal;
                                    let xfillo = xscal * (ibin - 1) as f32 + xll;
                                    let upi = ps_setup_thick(1);
                                    let nfill = cvttss2si(xscal * upi);
                                    for ifill in 0..=nfill {
                                        let xfil = xfillo + ifill as f32 / upi;
                                        ps_move_abs(xfil, yfillo);
                                        ps_vect_abs(xfil, yfilhi);
                                    }
                                } else if dxin >= 0. {
                                    if isymbt < 0 {
                                        isymbt = -(i as i32);
                                    }
                                    if ifimg > 0 {
                                        ps_symbol(rx, ry, isymbt);
                                    }
                                }
                            }
                        }
                    }
                }
                if np > 0 {
                    if dxin < 0. {
                        ps_setup_thick(ithhis);

                        for ixk in 0..=nkpts {
                            let xk = xlo + (ixk as f32 / nkpts as f32) * (xhi - xlo);
                            let rx = xscal * (ixk as f32 / nkpts as f32) * nbins as f32 + xll;
                            let mut ysum: f32 = 0.;
                            for i in 1..=nxu {
                                if ngx[i - 1] <= ng && (xx[i - 1] - xk).abs() < dx {
                                    let mut ifbelow = 0;
                                    if iaplt == 4 {
                                        for &g in &iginv {
                                            if ngx[i - 1] == g {
                                                ifbelow = 1;
                                            }
                                        }
                                    }
                                    if b3dxor(nupdown == 1, ifbelow == 1) {
                                        let t = (xx[i - 1] - xk) / dx;
                                        let u = 1. - t * t;
                                        ysum += u * u * u;
                                    }
                                }
                            }
                            ysum = ysum * 35. / 32.;
                            let ry = yscal * ysum + yll;
                            if ixk == 0 {
                                ps_move_abs(rx, ry);
                            }
                            ps_vect_abs(rx, ry);
                        }
                    } else if ifimg > 0 {
                        ps_setup_thick(ithhis);
                        ps_move_abs(xll, yll);
                        let mut rx = xll;
                        for ib in 1..=nbins {
                            let ry = yscal * jz[(ib - 1) as usize] as f32 + yll;
                            ps_vect_abs(rx, ry);
                            rx = xscal * ib as f32 + xll;
                            ps_vect_abs(rx, ry);
                        }
                        ps_vect_abs(rx, yll);
                    }
                }
            }
        }
        for _i in 1..=nlines {
            ps_dash_interactive(xscal / dx, xlo, xll, yscal.abs(), 0., yll);
        }
        if ifimg > 0 && iaplt > 2 && iaplt < 7 {
            label_axis(
                xll,
                xaxisy,
                xran,
                xran / (xhi - xlo),
                ntx.abs(),
                &xtick,
                iflog,
                0,
            );
            label_axis(xll, ylorig, yran, yscal.abs(), nty.abs(), &xtick, 0, 1);
            ps_misc_items(xscal / dx, xlo, xll, xran, yscal.abs(), 0., ylorig, yran);
        }
        //
        if ifcrv != 0 {
            'curves: {
                let colv =
                    |k: i32| -> f32 { col.get((k - 1).max(0) as usize).copied().unwrap_or(0.) };
                let mut ithcrv = ithhis;
                let mut ncrvs = 0;
                let mut inst = 1;
                // 95
                loop {
                    let itycrv = cvttss2si(colv(inst) + 0.0001);
                    if itycrv == 0 {
                        break;
                    }
                    ncrvs += 1;
                    iz[(ncrvs - 1) as usize] = inst;
                    inst += 5;
                    if itycrv > 1 {
                        inst += 4;
                    }
                }
                // 97
                if ncrvs == 0 {
                    break 'curves;
                }
                let mut out = std::io::stdout();
                loop {
                    // 87
                    let _ = out.write_all(
                        b" curve # to plot, 1 if sum (-1 this curve range only), thickness: ",
                    );
                    let (mut ncrplt, mut ifsum) = (0i32, 0i32);
                    read_list_stdin(&mut [
                        ListItem::Integer(&mut ncrplt),
                        ListItem::Integer(&mut ifsum),
                        ListItem::Integer(&mut ithcrv),
                    ]);
                    if ncrplt <= 0 || ncrplt > ncrvs {
                        break 'curves;
                    }
                    let mut ncrpar = 0i32;
                    if ifsum != 0 {
                        let _ = out.write_all(b" # of paired curve: ");
                        read_list_stdin(&mut [ListItem::Integer(&mut ncrpar)]);
                    }
                    let inst = iz[(ncrplt - 1) as usize];
                    ps_setup_thick(ithcrv);
                    // 90, 89: `go to (91,93),itycrv`, falling into 91 otherwise
                    let itycrv = cvttss2si(colv(inst) + 0.0001);
                    if itycrv != 2 {
                        // 91
                        let slope = colv(inst + 1);
                        let bint = colv(inst + 2);
                        let xcrvlo = xlo.max(colv(inst + 3));
                        let xcrvhi = xhi.min(colv(inst + 4));
                        let mut rx = xscal * (xcrvlo - xlo) / dx + xll;
                        let mut yact = slope * xcrvlo + bint;
                        let mut ry = dx * yscal * yact + yll;
                        ps_move_abs(rx, ry);
                        rx = xscal * (xcrvhi - xlo) / dx + xll;
                        yact = slope * xcrvhi + bint;
                        ry = dx * yscal * yact + yll;
                        ps_vect_abs(rx, ry);
                        continue;
                    }
                    // 93
                    let xcrvlo = xlo.max(colv(inst + 7));
                    let mut xcrvhi = xhi.min(colv(inst + 8));
                    if ifsum > 0 {
                        xcrvhi = xhi.min(colv(iz[(ncrpar - 1).max(0) as usize] + 8));
                    }
                    let dideal = 0.02 * dx / xscal;
                    let ndx = cvttss2si((xcrvhi - xcrvlo) / dideal);
                    let drx = (xcrvhi - xcrvlo) / ndx as f32;
                    for i in 0..=ndx {
                        let realx = xcrvlo + i as f32 * drx;
                        let mut yact = 0f32;
                        pearev(col, inst, realx, &mut yact);
                        if ifsum != 0 {
                            let mut yact2 = 0f32;
                            pearev(col, iz[(ncrpar - 1).max(0) as usize], realx, &mut yact2);
                            yact += yact2;
                        }
                        let rx = xscal * (realx - xlo) / dx + xll;
                        let ry = dx * yscal * yact + yll;
                        if i == 0 {
                            ps_move_abs(rx, ry);
                        }
                        if i != 0 {
                            ps_vect_abs(rx, ry);
                        }
                    }
                }
            }
        }
        // 81
        if sv.ifplt.abs() <= 10 {
            return;
        }
        sv.ifplt += if sv.ifplt >= 0 { 1 } else { -1 };
        sv.ifnewp = 0;
        SAVED.set(sv);
        if sv.ifplt.abs() <= 16 {
            return;
        }
        sv.ifnewp = 1;
        sv.ifplt = if sv.ifplt >= 0 { 11 } else { -11 };
        SAVED.set(sv);
        return;
    }
}

/// Original `pearev` (`bshst.f:705`): a Pearson or Gaussian curve's value.
pub fn pearev(col: &[f32], inst: i32, realx: f32, yact: &mut f32) {
    let colv = |k: i32| -> f32 { col.get((k - 1).max(0) as usize).copied().unwrap_or(0.) };
    let a1 = colv(inst + 1);
    let a2 = colv(inst + 2);
    let q1 = colv(inst + 3);
    let q2 = colv(inst + 4);
    let crvscl = colv(inst + 6);
    let ex = realx + colv(inst + 5);
    *yact = 0.;
    if ex <= -0.99999 * a1 || ex >= 0.99999 * a2 {
        return;
    }
    if q1 > 0. {
        *yact = crvscl * (1. + ex / a1).powf(q1) * (1. - ex / a2).powf(q2);
    }
    if q1 < 0. {
        *yact = crvscl * (q1 * ex * ex).exp();
    }
}

/// Original `gnhst` (`bshst.f:721`): `bshst` for general values, with no
/// names, records or curves.
pub fn gnhst(xx: &[f32], ngx: &[i32], nx: i32, nsymb: &[i32], ngrps: i32, iflog: i32) {
    let mut iz: Vec<i32> = vec![0; 100000.max(nx.max(0) as usize)];
    let mut jz: Vec<i32> = vec![0; 5000];
    let namx = [0i32; 1];
    let col = [0f32; 1];
    let irec = [0i32; 1];
    IFGNH.set(1);
    bshst(
        &namx, xx, ngx, nx, nsymb, ngrps, &mut iz, &mut jz, &irec, &col, iflog,
    );
}
