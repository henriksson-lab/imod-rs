//! Translation of `IMOD/flib/subrs/imsubs/wrap_iiunit.f90`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::extra_is_nbytes_and_flags;
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_convert, iiu_alt_print, iiu_file_info, iiu_open, iiu_read_lines, iiu_read_sec_part,
    iiu_read_section, iiu_ret_brief, iiu_ret_chunk_sizes, iiu_ret_print,
};
use crate::imod::libiimod::unit_header::iiu_ret_size;

/// Original `imopen` (`wrap_iiunit.f90:11`).
pub unsafe fn imopen(iunit: i32, name: &str, attribute: &str) {
    let _ = iiu_open_print(iunit, name, attribute);
}

/// Original `iiuOpenPrint` (`wrap_iiunit.f90:22`).
pub unsafe fn iiu_open_print(iunit: i32, name: &str, attribute: &str) -> i32 {
    let ierr = iiu_open(iunit, name, attribute);
    if ierr != 0 {
        return ierr;
    }
    let (mut num_kbytes, mut itype, mut iflags) = (0, 0, 0);
    let (mut nxyz, mut mxyz, mut nxyzst) = ([0; 3], [0; 3], [0; 3]);
    let (mut nx_tile, mut ny_tile, mut nz_chunk) = (0, 0, 0);
    iiu_file_info(iunit, &mut num_kbytes, &mut itype, &mut iflags);
    iiu_ret_size(
        iunit,
        nxyz.as_mut_ptr(),
        mxyz.as_mut_ptr(),
        nxyzst.as_mut_ptr(),
    );
    iiu_ret_chunk_sizes(iunit, &mut nx_tile, &mut ny_tile, &mut nz_chunk);
    if nx_tile > 0 || ny_tile > 0 {
        if itype == 5 && nx_tile == 0 {
            nx_tile = nxyz[0];
        }
        if ny_tile == 0 {
            ny_tile = nxyz[1];
        }
    }
    let do_print = iiu_ret_print() > 0;
    let do_extra = iiu_ret_brief() == 0 && do_print;
    let attrib = attribute.to_ascii_uppercase();
    if !attrib.starts_with('N') && do_print {
        if attrib.starts_with('S') || num_kbytes < 0 {
            println!("\n {} image file on unit{:4} : {}", attrib, iunit, name);
        } else {
            println!(
                "\n {} image file on unit{:4} : {}     Size= {:10} K",
                attrib, iunit, name, num_kbytes
            );
        }
    }
    if do_extra {
        match itype {
            1 if ny_tile > 0 && nx_tile == 0 => println!(
                "\n                    This is a TIFF file (in strips of{:7} x{:7}).",
                nxyz[0], ny_tile
            ),
            1 if ny_tile > 0 => println!(
                "\n                    This is a TIFF file (in tiles of{:7} x{:7}).",
                nx_tile, ny_tile
            ),
            1 => println!("\n                    This is a TIFF file."),
            5 if nx_tile > 0 => println!(
                "\n                    This is an HDF file (in chunks of{:7} x{:7} x{:5}).",
                nx_tile, ny_tile, nz_chunk
            ),
            5 => println!("\n                    This is an HDF file."),
            6 => println!("\n                    This is a JPEG file."),
            7 => println!("\n                    This is an image series file."),
            2 => (),
            _ => println!("\n                    This is a non-MRC file."),
        }
        if iflags & 1 != 0 {
            println!("\n                    This is a byte-swapped file.");
        }
    }
    0
}

/// Original `irdlin` (`wrap_iiunit.f90:90`).
pub unsafe fn irdlin(iunit: i32, array: &mut [f32]) -> Result<(), ()> {
    (iiu_read_lines(iunit, array.as_mut_ptr().cast(), 1) == 0)
        .then_some(())
        .ok_or(())
}

/// Original `irdsecl` (`wrap_iiunit.f90:102`).
pub unsafe fn irdsecl(iunit: i32, array: &mut [f32], num_lines: i32) -> Result<(), ()> {
    (iiu_read_lines(iunit, array.as_mut_ptr().cast(), num_lines) == 0)
        .then_some(())
        .ok_or(())
}

/// Original `irdsec` (`wrap_iiunit.f90:114`).
pub unsafe fn irdsec(iunit: i32, array: &mut [f32]) -> Result<(), ()> {
    (iiu_read_section(iunit, array.as_mut_ptr().cast()) == 0)
        .then_some(())
        .ok_or(())
}

/// Original `irdpas` (`wrap_iiunit.f90:127`).
pub unsafe fn irdpas(
    iunit: i32,
    array: &mut [f32],
    mx: i32,
    _my: i32,
    indx1: i32,
    indx2: i32,
    indy1: i32,
    indy2: i32,
) -> Result<(), ()> {
    (iiu_read_sec_part(
        iunit,
        array.as_mut_ptr().cast(),
        mx,
        indx1,
        indx2,
        indy1,
        indy2,
    ) == 0)
        .then_some(())
        .ok_or(())
}

/// Original `ialcon` (`wrap_iiunit.f90:139`).
pub unsafe fn ialcon(iunit: i32, convert: bool) {
    iiu_alt_convert(iunit, i32::from(convert));
}

/// Original `ialprt` (`wrap_iiunit.f90:149`).
pub unsafe fn ialprt(do_print: bool) {
    iiu_alt_print(i32::from(do_print));
}

/// Original `nbytes_and_flags` (`wrap_iiunit.f90:161`).
pub fn nbytes_and_flags(nint: i32, nreal: i32) -> bool {
    extra_is_nbytes_and_flags(nint, nreal) > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::iimage::{
        IIFILE_DEFAULT, ii_close, ii_open_new, ii_sync_from_mrc_header, ii_write_section_float,
    };
    use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_new, mrc_head_write};
    use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position};

    #[test]
    fn wrappers_read_a_real_mrc_section_and_preserve_extra_header_test() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-wrap-iiunit-{}.mrc", std::process::id()));
        let name = path.to_string_lossy().into_owned();
        unsafe {
            let file = ii_open_new(name.as_bytes(), "wb", IIFILE_DEFAULT);
            assert!(!file.is_null());
            let header = (*file).header.cast::<MrcHeader>();
            assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
            ii_sync_from_mrc_header(file, header);
            assert_eq!(
                mrc_head_write(&mut (*file).fp.clone().unwrap(), &mut *header),
                0
            );
            let mut first = [1.0_f32, 2.0, 3.0, 4.0];
            let mut second = [5.0_f32, 6.0, 7.0, 8.0];
            assert_eq!(
                ii_write_section_float(file, first.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(
                ii_write_section_float(file, second.as_mut_ptr().cast(), 1),
                0
            );
            ii_close(file);

            assert_eq!(iiu_open(97, &name, "RO"), 0);
            iiu_set_position(97, 1, 0);
            let mut read = [0.0_f32; 4];
            assert_eq!(irdsec(97, &mut read), Ok(()));
            assert_eq!(read, second);
            iiu_close(97);
        }
        assert!(nbytes_and_flags(8, 3));
        assert!(!nbytes_and_flags(7, 3));
        let _ = std::fs::remove_file(path);
    }
}
