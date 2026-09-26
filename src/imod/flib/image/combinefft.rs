//! Translation of `IMOD/flib/image/combinefft.f90`.
//!
//! COMBINEFFT combines the FFTs from the two tomograms of a double-axis tilt
//! series, taking into account the tilt range of each tilt series and the
//! transformation used to match one tomogram to the other.  For a location
//! in Fourier space where there is data from one tilt series but not the
//! other, it takes the Fourier value from just the one appropriate FFT;
//! everywhere else it averages the Fourier values from the two FFT files.
//!
//! The main program maps to [`combinefft`]; the two subroutines to
//! [`get_tilts`] and [`read_taper_transform`].  The Fortran `complex` arrays
//! `array`/`brray` are `f32` vectors holding the real and imaginary parts
//! interleaved, which is their storage layout; `ind` stays the source's
//! 1-based complex index and element `ind` is floats `2 * (ind - 1)` and
//! `2 * (ind - 1) + 1`.  `TAND` is the gfortran intrinsic the reference
//! object imports (`_gfortran_tand_r4`), [`gfortran_tand_r4`].

use crate::imod::flib::image::taperprep::taper_prep;
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::gfortran_tand_r4;
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::get_nxyz::line_is_filename;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::iclcdn::iclcdn;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas, irdsec};
use crate::imod::libcfshr::b3dutil::{
    b3d_lock_file, b3d_output_file_type, exit, override_output_type,
};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_float, pip_get_integer, pip_get_three_integers, pip_get_two_floats,
    pip_get_two_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::reduce_by_binning::irepak;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libcfshr::taperpad::{PadIn, sliceedgemean, taperoutpad};
use crate::imod::libfft::thrdfft;
use crate::imod::libiimod::iimage::ii_test_if_hdf;
use crate::imod::libiimod::parallelwrite::{
    iiu_par_wrt_flush_buffers, iiu_par_wrt_initialize, iiu_par_wrt_reclose_hdf,
    iiu_par_wrt_sec_part, iiu_write_dummy_sec_to_hdf, par_wrt_close, par_wrt_posn,
    par_wrt_properties,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_chunk_sizes, iiu_close, iiu_set_position, iiu_write_sec_part, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_mode, iiu_alt_origin, iiu_alt_sample, iiu_alt_size, iiu_trans_header,
    iiuwriteheaderstr,
};
use std::io::{BufRead, BufReader, Write};

/// `parameter (NUMLOOK = 16000, ...)` (`combinefft.f90:20`).
const NUMLOOK: i32 = 16000;
/// `LIMVIEW = 1440`.
const LIMVIEW: i32 = 1440;
/// `LIMRING = 100`.
const LIMRING: i32 = 100;
/// `LIMSLAB = 100`.
const LIMSLAB: i32 = 100;

/// `parameter (numOptions = 29)` (`combinefft.f90:65`), plus the
/// `LockFileForHDF` entry the table lacks.
///
/// Native's 29-entry table has no `LockFileForHDF`, which `:161` reads, so
/// without `combinefft.adoc` a chunked-HDF run stops with "Illegal option"
/// (BUGS.md).  Defined behaviour: the fallback table carries the autodoc's
/// 30 fields, so the program runs the same with or without the autodoc.
const COMBINEFFT_NUM_OPTIONS: i32 = 30;

/// Fallback PIP table, the `options(1)` string (`combinefft.f90:67-79`).
const COMBINEFFT_OPTIONS: &str = "ainput:AInputFFT:FN:@binput:BInputFFT:FN:@output:OutputFFT:FN:@\
xminmax:XMinAndMax:IP:@yminmax:YMinAndMax:IP:@zminmax:ZMinAndMax:IP:@\
taper:TaperPadsInXYZ:IT:@chunk:ChunkSizeForHDF:IT:@lock:LockFileForHDF:FN:@xsave:XSaveStartAndEnd:IP:@\
ysave:YSaveStartAndEnd:IP:@zsave:ZSaveStartAndEnd:IP:@\
place:PlaceChunkAtXYZ:IT:@atiltfile:ATiltFile:FN:@btiltfile:BTiltFile:FN:@\
ahighest:AHighestTilts:FP:@bhighest:BHighestTilts:FP:@\
inverse:InverseTransformFile:FN:@reduce:ReductionFraction:F:@\
separate:SeparateReduction:B:@joint:JointReduction:B:@ring:RingWidth:F:@\
nslabs:NumberOfSlabsInY:I:@radius:MinimumRadiusToReduce:F:@\
points:MinimumPointsInRing:I:@both:LowFromBothRadius:F:@\
verbose:VerboseOutput:B:@weight:WeightingPower:F:@param:ParameterFile:PF:@\
help:usage:B:";

/// Original program `combinefft` (`combinefft.f90:17`).
///
/// The `equivalence (nx, nxyz(1)), ...` scalars are `nxyz` read by index.
/// `numInRing(LIMRING,LIMSLAB,3)` and `ringSum` are flat column-major
/// arrays indexed `(ix - 1) + LIMRING * ((iy - 1) + LIMSLAB * (iz - 1))`;
/// `lookupAden(-NUMLOOK:NUMLOOK)` is indexed `i + NUMLOOK`.
pub fn combinefft() {
    unsafe {
        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut nxyzst = [0_i32; 3];
        let mut nxyz2 = [0_i32; 3];
        let mut mxyz2 = [0_i32; 3];
        let mut nxyz_in = [0_i32; 3];
        let mut work: Vec<f32> = Vec::new();
        let mut array: Vec<f32> = Vec::new();
        let mut brray: Vec<f32> = Vec::new();
        //
        let mut a_inv = [[0.0_f32; 3]; 3];
        let mut tilt_a = vec![0.0_f32; LIMVIEW as usize];
        let mut tilt_b = vec![0.0_f32; LIMVIEW as usize];
        let mut tilt_dens_a = vec![0.0_f32; LIMVIEW as usize];
        let mut tilt_dens_b = vec![0.0_f32; LIMVIEW as usize];
        let mut lookup_aden = vec![0_i32; (2 * NUMLOOK + 1) as usize];
        let mut lookup_bden = vec![0_i32; (2 * NUMLOOK + 1) as usize];
        let mut num_in_ring = vec![0_i32; (LIMRING * LIMSLAB * 3) as usize];
        let mut ring_sum = vec![0.0_f32; (LIMRING * LIMSLAB * 3) as usize];
        let rs = |ix: i32, iy: i32, iz: i32| -> usize {
            ((ix - 1) + LIMRING * ((iy - 1) + LIMSLAB * (iz - 1))) as usize
        };
        let mut cell2 = [0.0_f32; 6];
        let (mut origin_x, mut origin_y, mut origin_z) = (0.0_f32, 0.0_f32, 0.0_f32);
        //
        // `character*320 inputFile, fileOut, lockFile`
        let mut input_file = String::new();
        let mut file_out;
        let mut lock_file = [b' '; 320];
        //
        let mut dat = [b' '; 9];
        let mut tim = [b' '; 8];
        let mut title_ch = [b' '; 80];
        let (mut ina, mut inb): (bool, bool);
        let mut verbose;
        let mut inter_zone = false;
        let mut independent;
        let mut joint_zone;
        let mut real_input;
        let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
        let (mut a_crit_low, mut a_crit_high, mut b_crit_low, mut b_crit_high) =
            (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let (mut rad_a, mut rad_b): (f32, f32);
        let mut ierr: i32;
        let mut mode = 0_i32;
        let mut iunit_out;
        let (mut num_aviews, mut num_bviews) = (0_i32, 0_i32);
        let mut ind: i32;
        let mut ilook: i32;
        let (mut tsum, mut tmin, mut tmax);
        let (del_x, del_y, del_z);
        let (mut za, mut ya, mut ya_sq, mut xa, mut xa_sq, mut xp, mut yp, mut zp): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (mut ratio_a, mut ratio_b, mut wgt_a, mut wgt_b, mut tmean): (f32, f32, f32, f32, f32);
        let mut weight_power;
        let (mut fac_look_a, mut fac_look_b) = (0.0_f32, 0.0_f32);
        let mut ratio_mag_sq: f32;
        let (mut den_rad_a, mut den_rad_b, mut den_rad_sum): (f32, f32, f32);
        let (mut f_ring, mut f_slab, mut y_slab, mut this_mean);
        let mut sum_tmp = [0.0_f32; 3];
        let mut num_slabs;
        let mut num_rings = 0_i32;
        let (mut i_slab, mut i_ring, mut i_zone, mut min_in_ring, mut next_ring);
        let (mut radius_min, mut ring_width, mut reduce_frac);
        let (mut both_mean, mut one_mean, mut target, mut ring);
        let mut next_slab;
        let mut num_in_zero;
        let (mut nx_chunk, mut ny_chunk, mut nz_chunk) = (0_i32, 0_i32, 0_i32);
        let (mut ix_place, mut iy_place, mut iz_place) = (0_i32, 0_i32, 0_i32);
        let mut rad_sq: f32;
        let mut both_rad;
        let both_rad_sq;
        let mut za_sq;
        let mut dmean2 = 0.0_f32;
        let (mut ya_offset, mut za_offset);
        let (mut dmin2, mut dmax2) = (0.0_f32, 0.0_f32);
        let (mut ya_off_init, mut za_off_init);
        let dmean_in;
        let (mut ix_low, mut iy_low, mut iz_low, mut ixhi, mut iz_high, mut iy_high) =
            (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
        let (mut nx_box, mut ny_box, mut nz_box) = (0_i32, 0_i32, 0_i32);
        let idim: i32;
        let (mut nx3, mut ny3, mut nz3) = (0_i32, 0_i32, 0_i32);
        let mut mode2 = 0_i32;
        let (mut ix_save_start, mut ix_save_end, mut iy_save_start, mut iy_save_end) =
            (0_i32, 0_i32, 0_i32, 0_i32);
        let (mut iz_save_start, mut iz_save_end) = (0_i32, 0_i32);
        let initialize_hdf;
        let chunked_hdf;
        let mut parallel_hdf;
        //
        let pip_input;
        let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
        //
        weight_power = 0.0_f32;
        reduce_frac = 0.0_f32;
        num_slabs = 1_i32;
        ring_width = 0.01_f32;
        radius_min = 0.01_f32;
        min_in_ring = 30_i32;
        file_out = String::from(" ");
        verbose = false;
        joint_zone = false;
        independent = false;
        real_input = false;
        parallel_hdf = false;
        both_rad = 0.0_f32;
        num_in_zero = 0_i32;
        za_off_init = -0.5_f32;
        ya_off_init = -0.5_f32;
        let _ = num_in_zero;
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        pip_read_or_parse_options(
            &[COMBINEFFT_OPTIONS],
            COMBINEFFT_NUM_OPTIONS,
            "combinefft",
            "ERROR: COMBINEFFT - ",
            true,
            1,
            2,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        pip_input = num_opt_arg + num_non_opt_arg > 0;
        //
        // Get input and output files
        //
        if pip_get_in_out_file(
            "AInputFFT",
            1,
            "Name of first tomogram FFT file",
            &mut input_file,
            320,
        ) != 0
        {
            exit_error("No first input file specified");
        }
        imopen(1, &input_file, "ro");
        //
        if pip_get_in_out_file(
            "BInputFFT",
            2,
            "Name of second tomogram FFT file",
            &mut input_file,
            320,
        ) != 0
        {
            exit_error("No second input file specified");
        }
        imopen(2, &input_file, "old");
        //
        let _ = pip_get_in_out_file(
            "OutputFFT",
            3,
            "Name of output file, or Return to put in 2nd file",
            &mut file_out,
            320,
        );
        // `fileOut` is `character*320`: an empty entry is the blank record.
        let file_out_blank = file_out.trim_end_matches(' ').is_empty();
        //
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode,
            &raw mut dmin,
            &raw mut dmax,
            &raw mut dmean,
        );
        irdhdr(
            2,
            nxyz2.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode2,
            &raw mut dmin2,
            &raw mut dmax2,
            &raw mut dmean2,
        );
        if nxyz[0] != nxyz2[0] || nxyz[1] != nxyz2[1] || nxyz[2] != nxyz2[2] {
            exit_error("The two files are not the same size");
        }
        nxyz_in = nxyz;
        dmean_in = dmean;
        //
        if mode != 4 || mode2 != 4 {
            real_input = true;
            if mode == 4 || mode2 == 4 {
                exit_error("Both files must be either real data or FFTs");
            }
            if file_out_blank {
                exit_error("You must enter an output file for real input data");
            }
            if !pip_input {
                exit_error("You can use real data only with PIP input");
            }
            let nxyz_copy = nxyz;
            taper_prep(
                pip_input,
                false,
                &nxyz_copy,
                &mut ix_low,
                &mut ixhi,
                &mut iy_low,
                &mut iy_high,
                &mut iz_low,
                &mut iz_high,
                &mut nx_box,
                &mut ny_box,
                &mut nz_box,
                &mut nx3,
                &mut ny3,
                &mut nz3,
                &mut nxyz2,
                &mut mxyz2,
                &mut cell2,
                &mut origin_x,
                &mut origin_y,
                &mut origin_z,
            );
            //
            // these sizes below must be the same as if using FFT input
            nxyz[0] = nx3 / 2 + 1;
            nxyz[1] = ny3;
            nxyz[2] = nz3;
            za_off_init = 0.;
            ya_off_init = 0.;
            //
            // Promote the mode
            if mode == 2 || mode2 == 2 {
                mode = 2;
            } else if mode == 0 || mode2 == 0 {
                mode = mode.max(mode2);
            }
        }
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);

        initialize_hdf = pip_get_three_integers(
            b"ChunkSizeForHDF",
            &mut nx_chunk,
            &mut ny_chunk,
            &mut nz_chunk,
        ) == 0;
        ierr = pip_get_two_integers(b"XSaveStartAndEnd", &mut ix_save_start, &mut ix_save_end)
            + pip_get_two_integers(b"YSaveStartAndEnd", &mut iy_save_start, &mut iy_save_end)
            + pip_get_two_integers(b"ZSaveStartAndEnd", &mut iz_save_start, &mut iz_save_end)
            + pip_get_three_integers(
                b"PlaceChunkAtXYZ",
                &mut ix_place,
                &mut iy_place,
                &mut iz_place,
            );
        chunked_hdf = ierr == 0 && !initialize_hdf;
        if ierr > 0 && ierr != 4 && !initialize_hdf {
            exit_error(
                "For writing to a chunked HDF file, -xsave, -ysave, -zsave, and -place must all be entered",
            );
        }
        if chunked_hdf {
            parallel_hdf = pipgetstring_(b"LockFileForHDF", &mut lock_file) == 0;
        }
        if (initialize_hdf || chunked_hdf) && !real_input {
            exit_error("You must have real input data for writing to an HDF file");
        }

        if initialize_hdf {
            override_output_type(5);
            if b3d_output_file_type() != 5 {
                exit_error("Chunks cannot be used; this package was not built with HDF support");
            }
            idim = 0;
        } else {
            if (nx.wrapping_mul(ny)) as f32 * nz as f32 > 1.0e9 {
                exit_error("Volume to be combined is too large");
            }
            idim = nx.wrapping_mul(ny).wrapping_mul(nz);
            // `allocate(array(idim), brray(idim), work(2 * nx * nz), stat = ierr)`:
            // `complex` elements are two floats; the contents are undefined
            // until read, so zero fill stands in for them.
            ierr = 0;
            let n_complex = 2 * idim.max(0) as usize;
            let n_work = (2 * nx * nz).max(0) as usize;
            if array.try_reserve_exact(n_complex).is_err()
                || brray.try_reserve_exact(n_complex).is_err()
                || work.try_reserve_exact(n_work).is_err()
            {
                ierr = 1;
            } else {
                array.resize(n_complex, 0.0);
                brray.resize(n_complex, 0.0);
                work.resize(n_work, 0.0);
            }
            memory_error(ierr, "arrays for volumes of data");
            // `print *,'allocation in reals:', 2 * idim`
            println!(" allocation in reals:{}", ld_int(2 * idim));
        }
        //
        // Set up the output file
        iunit_out = 2;
        b3d_date(&mut dat);
        time(&mut tim);
        //
        // FORMAT 3000: `('COMBINEFFT: Combined FFT from two tomograms',t57,a9, 2x,a8)`
        let head = b"COMBINEFFT: Combined FFT from two tomograms";
        title_ch[..head.len()].copy_from_slice(head);
        title_ch[56..65].copy_from_slice(&dat);
        title_ch[67..75].copy_from_slice(&tim);
        //
        if !file_out_blank {
            iunit_out = 3;
            if chunked_hdf {
                if ii_test_if_hdf(file_out.trim_end_matches(' ').as_bytes()) <= 0 {
                    exit_error(
                        "Chunk writing options were entered but the output file is not an HDF file",
                    );
                }
                if parallel_hdf {
                    let lock_name = String::from_utf8_lossy(&lock_file)
                        .trim_end_matches(' ')
                        .to_string();
                    ierr = iiu_par_wrt_initialize(&lock_name, 5, -nx, ny, nz);
                    if ierr != 0 {
                        // `write(*,'(/,a,i3)')`
                        println!(
                            "\nERROR: COMBINEFFT - Initializing parallel write lock file, error{:>3}",
                            ierr
                        );
                        exit(1);
                    }
                    let (mut ix, mut iy, mut iz) = (0_i32, 0_i32, 0_i32);
                    par_wrt_properties(&mut ix, &mut iy, &mut iz);
                    if b3d_lock_file(ix) != 0 {
                        exit_error("Could not get lock for opening HDF file");
                    }
                }
                imopen(3, &file_out, "old");
                irdhdr(
                    3,
                    nxyz2.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &raw mut mode2,
                    &raw mut dmin,
                    &raw mut dmax,
                    &raw mut dmean2,
                );
                if nxyz_in[0] != nxyz2[0]
                    || nxyz_in[1] != nxyz2[1]
                    || nxyz_in[2] != nxyz2[2]
                    || mode2 != mode
                {
                    exit_error("The existing output file does not have right size or mode");
                }
                if parallel_hdf {
                    iiu_par_wrt_reclose_hdf(3, 0);
                }
            } else {
                imopen(3, &file_out, "new");
                iiu_trans_header(3, 1);
                if real_input {
                    //
                    iiu_alt_mode(3, mode);
                    if initialize_hdf {
                        //
                        // Initializing an HDF: set the chunk sizes, write the dummy section, and
                        // write header and exit
                        if iiu_alt_chunk_sizes(3, nx_chunk, ny_chunk, nz_chunk) != 0 {
                            exit_error("Setting up chunks in the HDF output file");
                        }
                        iiu_write_dummy_sec_to_hdf(3);
                        iiuwriteheaderstr(&3, &title_ch, &1, &(dmean + 1.), &(dmean - 1.), &dmean);
                        iiu_close(3);
                        exit(0);
                    } else {
                        nxyzst = [0; 3];
                        iiu_alt_size(3, &nxyz2, &nxyzst);
                        iiu_alt_sample(3, &mxyz2);
                        iiu_alt_cell(3, &cell2);
                        iiu_alt_origin(3, &[origin_x, origin_y, origin_z]);
                    }
                }
            }
        }

        input_file = String::from(" ");
        if pip_input {
            let mut record = [b' '; 320];
            let _ = pipgetstring_(b"InverseTransformFile", &mut record);
            input_file = String::from_utf8_lossy(&record).into_owned();
        } else {
            // `write(*,'(1x,a,$)')`
            print!(" File with inverse of matching transformation: ");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            // `read(5, '(a)') inputFile` has no `END=`.
            if matches!(std::io::stdin().read_line(&mut line), Ok(0) | Err(_)) {
                read_runtime_error(ListReadError::End);
            }
            input_file = line.trim_end_matches(['\r', '\n']).to_owned();
        }
        if input_file.trim_end_matches(' ').is_empty() {
            exit_error("No file specified with inverse transformation");
        }
        {
            let mut unit1 = BufReader::new(dopen(1, input_file.trim_end_matches(' '), "ro", "f"));
            for i in 0..3 {
                // `read(1,*) (aInv(i, j), j = 1, 3)` with no `END=`/`ERR=`.
                let [a0, a1, a2] = &mut a_inv[i];
                if let Err(err) = list_read(
                    &mut unit1,
                    &mut [ListItem::Real(a0), ListItem::Real(a1), ListItem::Real(a2)],
                ) {
                    read_runtime_error(err);
                }
            }
            // `close(1)`
        }
        //
        get_tilts(
            pip_input,
            "AHighestTilts",
            "ATiltFile",
            "first",
            &mut num_aviews,
            &mut tilt_a,
            LIMVIEW,
            &mut lookup_aden,
            NUMLOOK,
            &mut fac_look_a,
            &mut tilt_dens_a,
            &mut a_crit_low,
            &mut a_crit_high,
        );
        get_tilts(
            pip_input,
            "BHighestTilts",
            "BTiltFile",
            "second",
            &mut num_bviews,
            &mut tilt_b,
            LIMVIEW,
            &mut lookup_bden,
            NUMLOOK,
            &mut fac_look_b,
            &mut tilt_dens_b,
            &mut b_crit_low,
            &mut b_crit_high,
        );
        if pip_input {
            let _ = pip_get_float(b"WeightingPower", &mut weight_power);
            let _ = pip_get_float(b"ReductionFraction", &mut reduce_frac);
            if reduce_frac > 10. {
                exit_error("Reduction fraction must not be bigger than 10");
            }
            let _ = pip_get_logical("SeparateReduction", &mut independent);
            let _ = pip_get_logical("JointReduction", &mut joint_zone);
            inter_zone = !(joint_zone || independent);
            let _ = pip_get_logical("VerboseOutput", &mut verbose);
            let _ = pip_get_float(b"LowFromBothRadius", &mut both_rad);
        }
        //
        // set up reduction
        //
        if reduce_frac > 0. {
            let _ = pip_get_float(b"RingWidth", &mut ring_width);
            let _ = pip_get_float(b"MinimumRadiusToReduce", &mut radius_min);
            let _ = pip_get_integer(b"NumberOfSlabsInY", &mut num_slabs);
            let _ = pip_get_integer(b"MinimumPointsInRing", &mut min_in_ring);
            if num_slabs > LIMSLAB {
                exit_error("Too many slabs in Y for arrays");
            }
            // `combinefft.f90:275-278` computes `numRings` before testing
            // `ringWidth`; with `-ring 0` the quotient is infinite and native's
            // `cvttss2si` gives INT_MIN, which only happens to pass the
            // "Too many rings" test (and a tiny positive width gives INT_MIN
            // rings and carries on, BUGS.md).  Defined behaviour: validate the
            // width first, then compute the count (a huge count saturates and
            // is reported as too many rings).
            if ring_width <= 0. {
                exit_error("Illegal entry for ring width");
            }
            num_rings = ((0.9_f32 - radius_min) / ring_width + 2.) as i32;
            if num_rings > LIMRING {
                exit_error("Too many rings for arrays with this ring width");
            }
            if num_slabs <= 0 {
                exit_error("Illegal entry for number of slabs");
            }
            if min_in_ring <= 5 {
                exit_error("Minimum number of points in ring is too small to use");
            }
            if radius_min < 0. {
                exit_error("Minimum radius must be positive");
            }

            for iy in 1..=num_slabs {
                for ix in 1..=num_rings {
                    for iz in 1..=3 {
                        num_in_ring[rs(ix, iy, iz)] = 0;
                        ring_sum[rs(ix, iy, iz)] = 0.;
                    }
                }
            }
        }
        //
        pip_done();
        //
        tsum = 0.0_f32;
        tmin = 1.0e30_f32;
        tmax = -1.0e30_f32;
        del_x = 0.5_f32 / (nx as f32 - 1.);
        del_y = 1.0_f32 / ny as f32;
        del_z = 1.0_f32 / nz as f32;
        both_rad_sq = both_rad * both_rad;
        //
        // read real data (supply volume mean for testing to see if it comes out
        // the same as from taperoutvol)
        if real_input {
            read_taper_transform(
                1, &mut array, &mut work, ix_low, ixhi, iy_low, iy_high, iz_low, iz_high, nx_box,
                ny_box, nz_box, nx3, ny3, nz3, 0, &mut dmean,
            );
            read_taper_transform(
                2,
                &mut brray,
                &mut work,
                ix_low,
                ixhi,
                iy_low,
                iy_high,
                iz_low,
                iz_high,
                nx_box,
                ny_box,
                nz_box,
                nx3,
                ny3,
                nz3,
                0,
                &mut dmean2,
            );
        }
        //
        // Loop on Z, get index of start of plane in arrays
        za_offset = za_off_init;
        // Unchecked element access in this pass (TO_OPT.md, "combinefft
        // single-thread").  Soundness: `ind` runs from 1 to `nx * ny * nz`
        // (one step per voxel of the source's three loops), so `re = 2 *
        // (ind - 1)` and `im = re + 1` are below `2 * nx * ny * nz`, which
        // this assert bounds by both lengths.  The pointers are re-derived
        // after each plane's reads, which are the only other uses of the
        // arrays inside the loop.
        assert!(
            array.len() >= 2 * (nx as usize * ny as usize * nz as usize)
                && brray.len() >= 2 * (nx as usize * ny as usize * nz as usize)
        );
        for iz in 1..=nz {
            ind = 1 + nx * ny * (iz - 1);
            //
            // Still read FFT data plane by plane
            if !real_input {
                let off = 2 * (ind - 1) as usize;
                if irdsec(1, &mut array[off..]).is_err() {
                    exit_error("Reading FFT file");
                }
                iiu_set_position(2, iz - 1, 0);
                if irdsec(2, &mut brray[off..]).is_err() {
                    exit_error("Reading FFT file");
                }
            }
            let ap = array.as_mut_ptr();
            let bp = brray.as_mut_ptr();
            if real_input && iz > nz / 2 {
                za_offset = -1.;
            }
            // DO THIS IN MTFFLITER
            // if (iz > nz / 2) za = za - 1.
            za = del_z * (iz as f32 - 1.) + za_offset;
            za_sq = za * za;
            //
            // Loop on Y
            ya_offset = ya_off_init;
            for iy in 1..=ny {
                if real_input && iy > ny / 2 {
                    ya_offset = -1.;
                }
                ya = del_y * (iy as f32 - 1.) + ya_offset;
                ya_sq = ya * ya;
                //
                // Loop on X
                // The X loop runs out of line (TO_OPT.md, "combinefft
                // single-thread"; the `tilt` reprojection technique): inside
                // this 1500-line function every invariant was reloaded from
                // the stack on each voxel.  Same statements, same order.
                ind = combine_x_row(
                    ap,
                    bp,
                    ind,
                    nx,
                    del_x,
                    ya,
                    za,
                    ya_sq,
                    za_sq,
                    &a_inv,
                    a_crit_low,
                    a_crit_high,
                    b_crit_low,
                    b_crit_high,
                    both_rad_sq,
                    weight_power,
                    fac_look_a,
                    fac_look_b,
                    &lookup_aden,
                    &lookup_bden,
                    &tilt_dens_a,
                    &tilt_dens_b,
                    reduce_frac,
                    num_slabs,
                    radius_min,
                    ring_width,
                    inter_zone,
                    &mut num_in_ring,
                    &mut ring_sum,
                );
            }
        }
        //
        // If reducing, first get the scaling factors for the rings
        //
        if reduce_frac > 0. {
            if verbose {
                println!(" Ring  Slab  Zone  Mixed mean  Single mean    Target   Reduction factor");
            }
            for ix in 1..=num_rings {
                for iy in 1..=num_slabs {
                    if !inter_zone {
                        num_in_ring[rs(ix, iy, 1)] = num_in_ring[rs(ix, iy, 2)];
                        num_in_ring[rs(ix, iy, 3)] = num_in_ring[rs(ix, iy, 2)];
                    }
                    sum_tmp[0] = ring_sum[rs(ix, iy, 1)];
                    sum_tmp[2] = ring_sum[rs(ix, iy, 3)];
                    for iz in [1, 3] {
                        if num_in_ring[rs(ix, iy, iz)] >= min_in_ring
                            && num_in_ring[rs(ix, iy, 2)] >= min_in_ring
                        {
                            both_mean = ring_sum[rs(ix, iy, 2)] / num_in_ring[rs(ix, iy, 2)] as f32;
                            this_mean =
                                sum_tmp[(iz - 1) as usize] / num_in_ring[rs(ix, iy, iz)] as f32;
                            one_mean = this_mean;
                            if !independent && !inter_zone {
                                one_mean = 0.5 * (sum_tmp[0] + sum_tmp[2])
                                    / num_in_ring[rs(ix, iy, 2)] as f32;
                            }
                            // `combinefft.f90:481-483`: `maxss expr, 0.` and
                            // `minss target / oneMean, 1.` in the reference object.
                            target =
                                f_max((1. - reduce_frac) * one_mean + reduce_frac * both_mean, 0.);
                            ring_sum[rs(ix, iy, iz)] = f_min(target / one_mean, 1.);
                            if verbose {
                                // FORMAT 107: `(i4,2i6,1x,3f12.4,f8.4)`
                                println!(
                                    "{}{}{} {}{}{}{}",
                                    i_edit(ix, 4),
                                    i_edit(iy, 6),
                                    i_edit(iz, 6),
                                    f_edit(both_mean, 12, 4),
                                    f_edit(this_mean, 12, 4),
                                    f_edit(target, 12, 4),
                                    f_edit(ring_sum[rs(ix, iy, iz)], 8, 4)
                                );
                            }
                        } else {
                            ring_sum[rs(ix, iy, iz)] = 1.;
                        }
                    }
                }
            }
            //
            // Now run through the volume again
            //
            za_offset = za_off_init;
            for iz in 1..=nz {
                ind = 1 + nx * ny * (iz - 1);
                if real_input && iz > nz / 2 {
                    za_offset = -1.;
                }
                za = del_z * (iz as f32 - 1.) + za_offset;
                ya_offset = ya_off_init;
                for iy in 1..=ny {
                    if real_input && iy > ny / 2 {
                        ya_offset = -1.;
                    }
                    ya = del_y * (iy as f32 - 1.) + ya_offset;
                    ya_sq = ya * ya;
                    y_slab = (ya + 0.5) * num_slabs as f32;
                    i_slab = (y_slab + 0.5) as i32;
                    f_slab = y_slab + 0.5 - i_slab as f32;
                    next_slab = num_slabs.min(i_slab + 1);
                    i_slab = 1.max(i_slab);
                    for ix in 1..=nx {
                        //
                        // Find out where pixel is again
                        //
                        xa = del_x * (ix as f32 - 1.);
                        xa_sq = xa * xa;
                        xp = a_inv[0][0] * xa + a_inv[0][1] * ya + a_inv[0][2] * za;
                        yp = a_inv[1][0] * xa + a_inv[1][1] * ya + a_inv[1][2] * za;
                        if xp < 0. {
                            xp = -xp;
                            yp = -yp;
                        }
                        //
                        // `combinefft.f90:524-525`: here (unlike `:352-353`) the
                        // reference object emits `maxss 1.e-6, x`.
                        ratio_a = ya / f_max(1.0e-6, xa);
                        ratio_b = yp / f_max(1.0e-6, xp);
                        ina = ratio_a >= a_crit_low && ratio_a <= a_crit_high;
                        inb = ratio_b >= b_crit_low && ratio_b <= b_crit_high;
                        rad_a = (xa_sq + ya_sq + za * za).sqrt();
                        i_zone = 0;
                        if ina && !inb {
                            i_zone = 1;
                        }
                        if !ina && inb {
                            i_zone = 3;
                        }
                        if i_zone > 0 && rad_a >= radius_min {
                            //
                            // find ring and slab and adjust when in A or B only
                            //
                            ring = (rad_a - radius_min) / ring_width;
                            i_ring = (ring + 0.5) as i32;
                            f_ring = ring + 0.5 - i_ring as f32;
                            next_ring = num_rings.min(i_ring + 1);
                            i_ring = 1.max(i_ring);

                            let factor = (1. - f_ring)
                                * (1. - f_slab)
                                * ring_sum[rs(i_ring, i_slab, i_zone)]
                                + (1. - f_ring) * f_slab * ring_sum[rs(i_ring, next_slab, i_zone)]
                                + f_ring * (1. - f_slab) * ring_sum[rs(next_ring, i_slab, i_zone)]
                                + f_ring * f_slab * ring_sum[rs(next_ring, next_slab, i_zone)];
                            let re = 2 * (ind - 1) as usize;
                            // `array(ind) * (real)` is a full COMPLEX product with
                            // `(factor, 0.)` (reference object, `:546`).
                            let (ar, ai) = (array[re], array[re + 1]);
                            array[re] = ar * factor - ai * 0.0;
                            array[re + 1] = ar * 0.0 + ai * factor;
                        }
                        ind += 1;
                    }
                }
            }
        }
        //
        // Write the data out, taking inverse FFT and repacking real data first
        if real_input {
            thrdfft(&mut array, &mut work, nx3, ny3, nz3, -1);
        }

        for iz in 0..nz {
            if !chunked_hdf || (iz >= iz_save_start && iz <= iz_save_end) {
                ind = 1 + nx * ny * iz;
                let off = 2 * (ind - 1) as usize;
                if real_input {
                    {
                        // `irepak(array(ind), array(ind), ...)` passes the
                        // same storage as input and output; the input plane
                        // is read from a copy of its bytes.
                        let count = ((nx3 + 2) * ny) as usize;
                        let source: Vec<u8> = array[off..off + count]
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect();
                        let plane = &mut array[off..off + count];
                        let destination = core::slice::from_raw_parts_mut(
                            plane.as_mut_ptr().cast::<u8>(),
                            plane.len() * 4,
                        );
                        irepak(
                            destination,
                            &source,
                            &(nx3 + 2),
                            &ny,
                            &0,
                            &(nx3 - 1),
                            &0,
                            &(ny - 1),
                        );
                    }
                    array_min_max_mean_fortran(
                        &array[off..],
                        &nx3,
                        &ny,
                        &1,
                        &nx3,
                        &1,
                        &ny,
                        &mut dmin2,
                        &mut dmax2,
                        &mut dmean2,
                    );
                } else {
                    (dmin2, dmax2, dmean2) = iclcdn(&array[off..], nx, ny, 1, nx, 1, ny);
                }
                tmin = f_min(tmin, dmin2);
                tmax = f_max(tmax, dmax2);
                tsum += dmean2;
                if chunked_hdf {
                    if parallel_hdf {
                        par_wrt_posn(3, iz_place + iz - iz_save_start, iy_place);
                        ierr = iiu_par_wrt_sec_part(
                            3,
                            array[off..].as_mut_ptr().cast(),
                            nx3,
                            ix_save_start,
                            ix_place,
                            ix_place + ix_save_end - ix_save_start,
                            iy_save_start,
                            iy_save_end,
                        );
                    } else {
                        iiu_set_position(3, iz_place + iz - iz_save_start, iy_place);
                        ierr = iiu_write_sec_part(
                            3,
                            array[off..].as_mut_ptr().cast(),
                            nx3,
                            ix_save_start,
                            ix_place,
                            ix_place + ix_save_end - ix_save_start,
                            iy_save_start,
                            iy_save_end,
                        );
                    }
                    if ierr != 0 {
                        exit_error("Writing to chunk in HDF file");
                    }
                } else {
                    iiu_set_position(iunit_out, iz, 0);
                    iiu_write_section(iunit_out, array[off..].as_mut_ptr().cast());
                }
            }
        }

        tmean = tsum / nz as f32;
        if parallel_hdf {
            if iiu_par_wrt_flush_buffers(3) != 0 {
                exit_error("Finishing writing to output HDF file");
            }
            par_wrt_close();
            // `write(*,'(a,3g15.7,f15.0)')`
            println!(
                "Min, max, mean, # pixels={}{}{}{}",
                g_edit(tmin, 15, 7),
                g_edit(tmax, 15, 7),
                g_edit(tmean, 15, 7),
                f_edit(nx3 as f32 * ny3 as f32 * nz3 as f32, 15, 0)
            );
        } else {
            if chunked_hdf {
                if dmin < dmax {
                    tmin = f_min(dmin, tmin);
                    tmax = f_max(dmax, tmax);
                }
                tmean = dmean_in;
            }
            iiuwriteheaderstr(&iunit_out, &title_ch, &1, &tmin, &tmax, &tmean);
        }
        iiu_close(iunit_out);
        // print *,numInZero, ' points added to joint area near zero radius'
        let _ = (num_aviews, num_bviews, idim);
        exit(0);
    }
}

/// Original `getTilts` (`combinefft.f90:628`).
///
/// GETTILTS gets tilt angles and/or cutoffs for tangent of high angles.
/// PIPINPUT is true for PIP input; ANGLEOPT is the name of the option for
/// entering highest angles; FILEOPT is the name of the option for entering a
/// tilt angle file; WHICH has 'first' or 'second'; NVIEW is returned with the
/// number of angles read, or zero; TILT is returned with tilt angles; LIMVIEW
/// is the size of the angle arrays; LOOK is filled with a lookup table from
/// tangents to densities, dimensioned `-NUMLOOK:NUMLOOK` (indexed here
/// `i + NUMLOOK`); FACLOOK is returned with a factor to use for the lookup;
/// DENS is returned with relative tilt densities; CRITLO and CRITHI are
/// returned with criterion tangents of highest angles.
#[allow(clippy::too_many_arguments)]
pub fn get_tilts(
    pip_input: bool,
    angle_opt: &str,
    file_opt: &str,
    which: &str,
    nview: &mut i32,
    tilt: &mut [f32],
    lim_view: i32,
    look: &mut [i32],
    num_look: i32,
    fac_look: &mut f32,
    dens: &mut [f32],
    crit_low: &mut f32,
    crit_high: &mut f32,
) {
    let mut wgt_increm = [0.0_f32; 20];
    let (mut tilt_low, mut tilt_high) = (0.0_f32, 0.0_f32);
    // `character*120 line`
    let mut line = [b' '; 120];
    let (ierr, ierr2): (i32, i32);
    let num_weights: i32;
    let (mut min_look, mut max_look, mut ind_look, mut next_look, mut mid, mut ilook);
    let mut tmp: f32;
    let (mut sum_intrv, mut wsum, avg_intrv): (f32, f32, f32);
    let lk = |i: i32| -> usize { (i + num_look) as usize };
    //
    num_weights = 2;
    //
    *nview = 0;
    tilt_high = -9999.;
    let _ = &mut tilt_low;

    // `go to 20` skips the tilt-file block; `go to 10` enters it.
    let mut read_file = false;
    if pip_input {
        ierr = pip_get_two_floats(angle_opt.as_bytes(), &mut tilt_low, &mut tilt_high);
        ierr2 = pipgetstring_(file_opt.as_bytes(), &mut line);
        if ierr == 0 && ierr2 == 0 {
            exit_error("You cannot enter both highest angles and a tilt angle file");
        }
        if ierr != 0 && ierr2 != 0 {
            // `combinefft.f90:654-655` concatenates 'You must'//'enter' with no
            // blank (BUGS.md); the message is spelled correctly here.
            exit_error("You must enter either highest angles or a tilt angle file");
        }
        if ierr != 0 {
            read_file = true;
        }
    } else {
        // `print *,'For ', which, ' tomogram file, enter either the ', ...`
        println!(
            " For {} tomogram file, enter either the starting and ending tilt  angles, or the name of a file with tilt angles in it",
            which
        );
        let mut text = String::new();
        // `read(5, '(a)') line` with no `END=`.
        if matches!(std::io::stdin().read_line(&mut text), Ok(0) | Err(_)) {
            read_runtime_error(ListReadError::End);
        }
        let text = text.trim_end_matches(['\r', '\n']).as_bytes();
        let count = text.len().min(120);
        line[..count].copy_from_slice(&text[..count]);
        if line_is_filename(&line) {
            read_file = true;
        } else {
            // `read(line,*,err = 10, end = 10) tiltLow, tiltHigh`
            let mut cursor = std::io::Cursor::new(line.to_vec());
            if list_read(
                &mut cursor,
                &mut [
                    ListItem::Real(&mut tilt_low),
                    ListItem::Real(&mut tilt_high),
                ],
            )
            .is_err()
            {
                read_file = true;
            }
        }
    }
    if read_file {
        //
        // get here either way if line has a filename for tilt angle file
        //
        // 10
        if line.iter().all(|&c| c == b' ') {
            exit_error("No tilt angle filename entered");
        }
        let name = String::from_utf8_lossy(&line)
            .trim_end_matches(' ')
            .to_string();
        let mut unit1 = BufReader::new(dopen(1, &name, "ro", "f"));
        // 15
        loop {
            let mut value = tilt[*nview as usize];
            match list_read(&mut unit1, &mut [ListItem::Real(&mut value)]) {
                Err(ListReadError::End) => break,
                Err(ListReadError::Error) => exit_error("Reading tilt angle file"),
                Ok(()) => {}
            }
            tilt[*nview as usize] = value;
            *nview += 1;
            if *nview >= lim_view {
                exit_error("Too many tilt angles for arrays");
            }
        }
        // 18
        if *nview < 2 {
            exit_error("Too few tilt angles in file");
        }
        // `close(1)`
        let _ = unit1.fill_buf();
        drop(unit1);
        let nv = *nview as usize;
        tilt_low = tilt[0];
        tilt_high = tilt[nv - 1];
        //
        // INVERT TILT ANGLES BECAUSE "TILT" PROGRAM IS WEIRD
        //
        for value in &mut tilt[..nv] {
            *value = -*value;
        }
        //
        // order tilt angles to make it easier to get densities
        //
        for i in 0..nv - 1 {
            for j in i + 1..nv {
                if tilt[i] > tilt[j] {
                    tmp = tilt[i];
                    tilt[i] = tilt[j];
                    tilt[j] = tmp;
                }
            }
        }
        //
        // compute densities just as in Tilt program
        //
        for i in 1..=num_weights {
            wgt_increm[(i - 1) as usize] = 1. / (i as f32 - 0.5);
        }
        avg_intrv = (tilt[nv - 1] - tilt[0]) / (*nview - 1) as f32;
        // `tilt`/`dens` are 1-based in the source: `t(k)` is `tilt[k - 1]`.
        let t = |k: i32| -> usize { (k - 1) as usize };
        for iv in 1..=*nview {
            sum_intrv = 0.;
            wsum = 0.;
            for iw in 1..=num_weights {
                if iv - iw > 0 {
                    wsum += wgt_increm[(iw - 1) as usize];
                    sum_intrv +=
                        wgt_increm[(iw - 1) as usize] * (tilt[t(iv + 1 - iw)] - tilt[t(iv - iw)]);
                }
                if iv + iw <= *nview {
                    wsum += wgt_increm[(iw - 1) as usize];
                    sum_intrv +=
                        wgt_increm[(iw - 1) as usize] * (tilt[t(iv + iw)] - tilt[t(iv + iw - 1)]);
                }
            }
            dens[t(iv)] = avg_intrv / (sum_intrv / wsum);
        }
        //
        // build lookup table to tilt angles
        //
        *fac_look =
            (num_look as f32 - 10.) / gfortran_tand_r4(f_max(tilt[0].abs(), tilt[nv - 1].abs()));
        for i in -num_look..=num_look {
            look[lk(i)] = 0;
        }
        min_look = num_look;
        max_look = -num_look;
        //
        // fill in values at the angles
        //
        for i in 1..=*nview {
            ind_look = (*fac_look * gfortran_tand_r4(tilt[t(i)])).round() as i32;
            look[lk(ind_look)] = i;
            min_look = min_look.min(ind_look);
            max_look = max_look.max(ind_look);
        }
        //
        // extend endpoints
        //
        for i in -num_look..min_look {
            look[lk(i)] = look[lk(min_look)];
        }
        for i in max_look + 1..=num_look {
            look[lk(i)] = look[lk(max_look)];
        }
        ilook = min_look;
        //
        // fill in each interval split between neighbors
        //
        while ilook < max_look {
            next_look = ilook + 1;
            while look[lk(next_look)] == 0 {
                next_look += 1;
            }
            mid = (next_look + ilook) / 2;
            for i in ilook + 1..=mid {
                look[lk(i)] = look[lk(ilook)];
            }
            for i in mid + 1..next_look {
                look[lk(i)] = look[lk(next_look)];
            }
            ilook = next_look;
        }
    }
    //
    // 20
    if tilt_high == -9999. {
        exit_error("Highest tilt angle not entered");
    }
    //
    // INVERT ANGLES HERE TOO
    //
    // `min(-tiltLow, -tiltHigh)` / `max(...)` (`combinefft.f90:771-772`): the
    // reference object folds the negations out, as `-maxss(tiltHigh,
    // tiltLow)` and `-minss(tiltHigh, tiltLow)`.
    *crit_low = gfortran_tand_r4(-f_max(tilt_high, tilt_low));
    *crit_high = gfortran_tand_r4(-f_min(tilt_high, tilt_low));
    //
    // If there were no tilt angles, just set up for equal densities
    //
    if *nview == 0 {
        *fac_look = (num_look as f32 - 10.) / f_max(crit_low.abs(), crit_high.abs());
        for i in -num_look..=num_look {
            look[lk(i)] = 1;
        }
        dens[0] = 1.;
    }
}

/// Original `readTaperTransform` (`combinefft.f90:800`).
///
/// readTaperTransform reads a subvolume from a file, pads and tapers it
/// outside into a larger volume, and takes the FFT.  IUNIT is the unit
/// number; ARRAY is the array for the volume; WORK is a work array for 3D
/// FFTs; IXLO, IXHI, IYLO, IYHI, IZLO, IZHI are the index coordinates of the
/// subvolume in the file; NXBOX, NYBOX, NZBOX are the size of the box being
/// read in; NX3, NY3, NZ3 are the size of the padded volume; if IFMEAN is 1
/// it will taper to DMEANIN, otherwise it will use the mean of the faces of
/// the subvolume.  Planes of the FFT consist of (NX3 + 2) * NY floats
/// consecutively.
#[allow(clippy::too_many_arguments)]
pub fn read_taper_transform(
    iunit: i32,
    array: &mut [f32],
    work: &mut [f32],
    ix_low: i32,
    ixhi: i32,
    iy_low: i32,
    iy_high: i32,
    iz_low: i32,
    iz_high: i32,
    nx_box: i32,
    ny_box: i32,
    nz_box: i32,
    nx3: i32,
    ny3: i32,
    nz3: i32,
    if_mean: i32,
    dmean_in: &mut f32,
) {
    let (mut atten, mut dmin, mut dmax, mut dmean, mut base) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let (mut ibase, mut ix_base, mut iz_read, mut ioffset): (i32, i32, i32, i32);
    let mut num_pix: i32;
    let mut edge_mean: f64;
    let mut edge_sum: f64;
    let nx_dim = nx3 + 2;
    let iz_start = iz_low - ((nz3 - nz_box) / 2);
    let iz_end = iz_start + nz3 - 1;
    //
    // Read all the data first to get the mean
    edge_sum = 0.;
    num_pix = 0;
    for iz in iz_low..=iz_high {
        ibase = nx_dim * ny3 * (iz - iz_start);
        unsafe {
            iiu_set_position(iunit, iz, 0);
            if irdpas(
                iunit,
                &mut array[ibase as usize..],
                nx_box,
                ny_box,
                ix_low,
                ixhi,
                iy_low,
                iy_high,
            )
            .is_err()
            {
                exit_error("Reading image file");
            }
        }
        if iz == iz_low || iz == iz_high {
            array_min_max_mean_fortran(
                &array[ibase as usize..],
                &nx_box,
                &ny_box,
                &1,
                &nx_box,
                &1,
                &ny_box,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
            num_pix += nx_box * ny_box;
            // `nxBox * nyBox * dmean` is a real*4 product, widened for the sum.
            edge_sum += ((nx_box * ny_box) as f32 * dmean) as f64;
        } else {
            edge_mean = sliceedgemean(&array[ibase as usize..], &nx_box, &1, &nx_box, &1, &ny_box);
            num_pix += 2 * (nx_box + ny_box) - 4;
            edge_sum += edge_mean * (2 * (nx_box + ny_box) - 4) as f64;
        }
    }
    dmean = (edge_sum / num_pix as f64) as f32;
    // `print *,'file mean', dmeanIn, '    edge mean', dmean`
    println!(
        " file mean{}     edge mean{}",
        ld_real(*dmean_in),
        ld_real(dmean)
    );
    if if_mean == 0 {
        *dmean_in = dmean;
    }
    //
    // taper now that mean is known
    for iz in iz_low..=iz_high {
        ibase = nx_dim * ny3 * (iz - iz_start);
        taperoutpad(
            PadIn::InPlace,
            &nx_box,
            &ny_box,
            &mut array[ibase as usize..],
            &nx_dim,
            &nx3,
            &ny3,
            &1,
            dmean_in,
        );
    }
    for iz in iz_start..=iz_end {
        iz_read = iz_low.max(iz_high.min(iz));
        ibase = nx_dim * ny3 * (iz - iz_start);
        ioffset = nx_dim * ny3 * (iz_read - iz);
        if iz < iz_low || iz > iz_high {
            if iz < iz_low {
                atten = (iz - iz_start) as f32 / (iz_low - iz_start) as f32;
            } else {
                atten = (iz_end - iz) as f32 / (iz_end - iz_high) as f32;
            }
            base = (1. - atten) * *dmean_in;
            for iy in 1..=ny3 {
                ix_base = ibase + (iy - 1) * nx_dim;
                for i in ix_base + 1..=ix_base + nx3 {
                    array[(i - 1) as usize] = base + atten * array[(i + ioffset - 1) as usize];
                }
            }
        }
    }
    //
    // Take fft
    thrdfft(array, work, nx3, ny3, nz3, 0);
    let _ = (dmin, dmax, atten, base);
}

// ---------------------------------------------------------------------------
// gfortran runtime boundary: the `MIN`/`MAX` intrinsics on reals, formatted
// and list-directed output editing and the runtime error of a failed read.
// Not translations of source units: what gfortran and libgfortran do for the
// intrinsics, edit descriptors and `print *` items this program uses.
// ---------------------------------------------------------------------------

/// The X loop of `combinefft.f90:338-456` (first pass over the volume), run
/// out of line as a nested unit of [`combinefft`] would be: every statement is
/// the source's, in the source's order, on the same `real*4` values; only the
/// loop invariants arrive as arguments so they stay in registers (TO_OPT.md,
/// "combinefft single-thread").  Returns `ind` after the row.
///
/// # Safety
/// `ap` and `bp` must be valid for reads and writes of `2 * (ind - 1 + nx)`
/// floats (`combinefft` asserts both arrays hold `2 * nx * ny * nz`).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
unsafe fn combine_x_row(
    ap: *mut f32,
    bp: *mut f32,
    mut ind: i32,
    nx: i32,
    del_x: f32,
    ya: f32,
    za: f32,
    ya_sq: f32,
    za_sq: f32,
    a_inv: &[[f32; 3]; 3],
    a_crit_low: f32,
    a_crit_high: f32,
    b_crit_low: f32,
    b_crit_high: f32,
    both_rad_sq: f32,
    weight_power: f32,
    fac_look_a: f32,
    fac_look_b: f32,
    lookup_aden: &[i32],
    lookup_bden: &[i32],
    tilt_dens_a: &[f32],
    tilt_dens_b: &[f32],
    reduce_frac: f32,
    num_slabs: i32,
    radius_min: f32,
    ring_width: f32,
    inter_zone: bool,
    num_in_ring: &mut [i32],
    ring_sum: &mut [f32],
) -> i32 {
    let rs = |ix: i32, iy: i32, iz: i32| -> usize {
        ((ix - 1) + LIMRING * ((iy - 1) + LIMSLAB * (iz - 1))) as usize
    };
    let a_inv = *a_inv;
    let (mut xa, mut xa_sq, mut xp, mut yp, mut zp, mut rad_sq);
    let (mut ratio_a, mut ratio_b, mut ina, mut inb, mut wgt_a, mut wgt_b);
    let (mut ratio_mag_sq, mut rad_a, mut rad_b);
    let mut ilook: i32;
    let (mut den_rad_a, mut den_rad_b, mut den_rad_sum);
    let (mut i_slab, mut i_ring, mut i_zone);
    // `1..nx + 1`: the counts of `1..=nx` (`nx` is far below `i32::MAX`,
    // the volume having passed the 1e9 check), without `RangeInclusive`'s
    // extra flag.
    unsafe {
        for ix in 1..nx + 1 {
            xa = del_x * (ix as f32 - 1.);
            xa_sq = xa * xa;
            //
            // back transform this position to get vector in fft b
            //
            xp = a_inv[0][0] * xa + a_inv[0][1] * ya + a_inv[0][2] * za;
            yp = a_inv[1][0] * xa + a_inv[1][1] * ya + a_inv[1][2] * za;
            if xp < 0. {
                xp = -xp;
                yp = -yp;
            }
            rad_sq = za_sq + xa_sq + ya_sq;
            //
            ratio_a = ya / f_max(xa, 1.0e-6);
            ratio_b = yp / f_max(xp, 1.0e-6);
            ina = (ratio_a >= a_crit_low && ratio_a <= a_crit_high) || rad_sq < both_rad_sq;
            inb = (ratio_b >= b_crit_low && ratio_b <= b_crit_high) || rad_sq < both_rad_sq;
            //
            // For counting pixels
            // ina=(rata>=acritlo.and.rata<=acrithi)
            // inb=(ratb>=bcritlo.and.ratb<=bcrithi)
            // if (.not.(ina.and.inb) .and. radSq<bothRadSq) then
            // ina = .true.
            // inb = .true.
            // numInZero = numInZero + 1
            // endif
            let re = 2 * (ind - 1) as usize;
            let im = re + 1;
            let a_val = ((*ap.add(re)), (*ap.add(im)));
            // array(ind) =(2., 0.)
            if !ina && !inb {
                //
                // if in neither, take simple mean
                //
                // `0.5 * (array(ind) + brray(ind))` is a COMPLEX product:
                // gfortran promotes the real to `(0.5, 0.)` and, since
                // signed zeros are honoured, multiplies in full
                // (`rr = ar*br - ai*bi`, `ri = ar*bi + ai*br`), so a NaN in
                // one part reaches both (reference object, `:371`).
                let (sre, sim) = ((*ap.add(re)) + (*bp.add(re)), (*ap.add(im)) + (*bp.add(im)));
                (*ap.add(re)) = 0.5 * sre - 0.0 * sim;
                (*ap.add(im)) = 0.5 * sim + 0.0 * sre;
                // array(ind) =(2.5, 0.)
            } else if !ina && inb {
                //
                // if in B alone, take b's value
                //
                (*ap.add(re)) = (*bp.add(re));
                (*ap.add(im)) = (*bp.add(im));
                // array(ind) =(3., 0.)
            } else if ina && inb {
                //
                // in both: need to mix in selected way
                //
                wgt_a = 0.5_f32;
                wgt_b = 0.5_f32;
                if weight_power > 0.001 && rad_sq >= both_rad_sq {
                    //
                    // weighting by local density:  Find radius in A and B.
                    // get change in magnitude and make the magnitude change by
                    // the same amount in opposite direction, since stretching in
                    // real space is squeezing in Fourier space (the square is
                    // needed to get from stretch through null to squeeze) .
                    //
                    zp = a_inv[2][0] * xa + a_inv[2][1] * ya + a_inv[2][2] * za;
                    ratio_mag_sq = (xa_sq + ya_sq + za_sq) / (xp * xp + yp * yp + zp * zp);
                    rad_a = (xa_sq + ya_sq).sqrt();
                    rad_b = (xp * xp + yp * yp).sqrt() * ratio_mag_sq;
                    //
                    // find tilt density and
                    // effectively divide by radius (multiply by other radius)
                    //
                    ilook = (fac_look_a * ratio_a).round() as i32;
                    if ilook.abs() > NUMLOOK {
                        println!(
                            "{}{}{}{}{}{}",
                            ld_real(ratio_a),
                            ld_real(xa),
                            ld_real(ya),
                            ld_real(za),
                            ld_real(xp),
                            ld_real(yp)
                        );
                    }
                    den_rad_a = (rad_b
                        * tilt_dens_a[(lookup_aden[(ilook + NUMLOOK) as usize] - 1) as usize])
                        .powf(weight_power);
                    ilook = (fac_look_b * ratio_b).round() as i32;
                    den_rad_b = (rad_a
                        * tilt_dens_b[(lookup_bden[(ilook + NUMLOOK) as usize] - 1) as usize])
                        .powf(weight_power);
                    den_rad_sum = den_rad_a + den_rad_b;
                    if den_rad_sum > 1.0e-4 {
                        wgt_a = den_rad_a / den_rad_sum;
                        wgt_b = den_rad_b / den_rad_sum;
                    }
                }
                // array(ind) =(wa, 0.)
                // Two full COMPLEX products (see `:371` above), then the
                // sum (reference object, `:413`).
                let (are, aim) = ((*ap.add(re)), (*ap.add(im)));
                let (bre, bim) = ((*bp.add(re)), (*bp.add(im)));
                (*ap.add(re)) = (wgt_a * are - 0.0 * aim) + (wgt_b * bre - 0.0 * bim);
                (*ap.add(im)) = (wgt_a * aim + 0.0 * are) + (wgt_b * bim + 0.0 * bre);
                //
                // else if in A and not in B, leave A value as is
                //
            }
            //
            // if reducing, compute ring, slab, and zone and add abs value
            //
            if reduce_frac > 0. {
                i_slab = 1.max(num_slabs.min(((ya + 0.5) * num_slabs as f32) as i32 + 1));
                rad_a = (xa_sq + ya_sq + za_sq).sqrt();
                i_ring = ((rad_a - radius_min) / ring_width) as i32 + 1;
                if inter_zone {
                    //
                    // If doing comparisons between zones, determine zone and
                    // sum specifically for the zones
                    //
                    i_zone = 0;
                    if ina && !inb {
                        i_zone = 1;
                    }
                    if ina && inb {
                        i_zone = 2;
                    }
                    if !ina && inb {
                        i_zone = 3;
                    }
                    if i_ring > 0 && i_zone > 0 {
                        num_in_ring[rs(i_ring, i_slab, i_zone)] += 1;
                        ring_sum[rs(i_ring, i_slab, i_zone)] += (*ap.add(re)).hypot((*ap.add(im)));
                    }
                } else {
                    //
                    // Otherwise, just do joint area and sum A, B, and average
                    //
                    if i_ring > 0 && ina && inb {
                        num_in_ring[rs(i_ring, i_slab, 2)] += 1;
                        ring_sum[rs(i_ring, i_slab, 1)] += a_val.0.hypot(a_val.1);
                        ring_sum[rs(i_ring, i_slab, 2)] += (*ap.add(re)).hypot((*ap.add(im)));
                        ring_sum[rs(i_ring, i_slab, 3)] += (*bp.add(re)).hypot((*bp.add(im)));
                    }
                }
            }
            ind += 1;
        }
    }
    ind
}

/// gfortran `MIN(a, b)` on `real*4`, as `a < b ? a : b` (`minss`).
fn f_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// gfortran `MAX(a, b)` on `real*4`, as `a > b ? a : b` (`maxss`).
fn f_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `Iw` editing: right-justified in `w`; `w` asterisks if it does not fit.
fn i_edit(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// `Fw.d` editing (libgfortran `write_float`), as
/// [`crate::imod::flib::subrs::compat::gfortran_rt::format_f`] implements
/// it, with the trailing point of `d = 0`.
fn f_edit(value: f32, w: usize, d: usize) -> String {
    if d == 0 && value.is_finite() {
        let mut text = format!("{value:.0}");
        text.push('.');
        if text.len() > w {
            return "*".repeat(w);
        }
        return format!("{text:>w$}");
    }
    crate::imod::flib::subrs::compat::gfortran_rt::format_f(value as f64, w, d)
}

/// `Gw.d` editing with no exponent part: `F(w-4).(d-k)` plus four blanks when
/// the value rounded to `d` significant digits has `0 <= k <= d` integer
/// digits, else `Ew.d`.
fn g_edit(value: f32, w: usize, d: i32) -> String {
    if value.is_nan() {
        return format!("{:>w$}", "NaN");
    }
    if value.is_infinite() {
        let text = match (value < 0.0, w) {
            (false, 8..) => "Infinity",
            (false, _) => "Inf",
            (true, 9..) => "-Infinity",
            (true, _) => "-Inf",
        };
        return format!("{text:>w$}");
    }
    let magnitude = value.abs();
    let mut digits = "0".repeat(d as usize);
    let mut exponent = 0_i32;
    if magnitude != 0.0 {
        let scientific = format!("{:.*e}", (d - 1) as usize, magnitude);
        let (mantissa, power) = scientific.split_once('e').unwrap();
        digits = mantissa.replace('.', "");
        exponent = power.parse::<i32>().unwrap() + 1;
    }
    if magnitude == 0.0 || (0..=d).contains(&exponent) {
        let decimals = if magnitude == 0.0 {
            d - 1
        } else {
            d - exponent
        };
        let mut text = format!("{:.*}", decimals as usize, value);
        if decimals == 0 {
            text.push('.');
        }
        format!("{:>1$}    ", text, w - 4)
    } else {
        format!(
            "{:>1$}",
            format!(
                "{}0.{}E{}{:02}",
                if value < 0.0 { "-" } else { "" },
                digits,
                if exponent < 0 { '-' } else { '+' },
                exponent.abs()
            ),
            w
        )
    }
}

/// A list-directed (`print *`) `integer*4` item: a blank separator (the
/// record's leading blank when it is the first item) and `I11`.
fn ld_int(value: i32) -> String {
    format!("{value:>12}")
}

/// A list-directed `real*4` item (its blank separator included): `F` form
/// with nine significant digits and four trailing blanks for magnitudes in
/// [0.1, 1e9), `E` form with a two-digit exponent otherwise; `NaN` and
/// `Infinity` right-justified in the 17 columns.
fn ld_real(value: f32) -> String {
    if value.is_nan() {
        return format!("{:>17}", "NaN");
    }
    if value.is_infinite() {
        return format!("{:>17}", if value < 0. { "-Infinity" } else { "Infinity" });
    }
    if value == 0. {
        return format!("{:>13}    ", format!("{value:.8}"));
    }
    let scientific = format!("{:.8e}", value.abs());
    let (mantissa, power) = scientific.split_once('e').unwrap();
    let k = power.parse::<i32>().unwrap() + 1;
    if (0..=9).contains(&k) {
        let mut text = format!("{:.*}", (9 - k) as usize, value);
        if k == 9 {
            text.push('.');
        }
        format!("{text:>13}    ")
    } else {
        let e = k - 1;
        format!(
            "{:>17}",
            format!(
                "{}{}E{}{:02}",
                if value < 0. { "-" } else { "" },
                mantissa,
                if e < 0 { '-' } else { '+' },
                e.abs()
            )
        )
    }
}

/// A formatted or list-directed `read` with no `END=`/`ERR=` that fails:
/// libgfortran reports it and stops with status 2.
fn read_runtime_error(err: ListReadError) -> ! {
    let _ = std::io::stdout().flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}
