//! Translation of `IMOD/flib/image/newstack.f90`.
#![allow(dead_code)]

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::getbinnedsize::get_binned_size;
use crate::imod::flib::subrs::hvem::parse_input_params::set_current_adoc_or_exit;
use crate::imod::flib::subrs::hvem::parse_input_params::{exit_error, pip_read_or_parse_options};
use crate::imod::flib::subrs::hvem::rdlist::parselist2;
use crate::imod::flib::subrs::hvem::temp_filename::temp_filename;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{
    ialprt, iiu_open_print, imopen, irdsecl, nbytes_and_flags,
};
use crate::imod::flib::subrs::xfsubs::readdistortions::read_mag_gradients;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_add_section, adoc_clear, adoc_find_insert_index,
    adoc_get_collection_name, adoc_get_float, adoc_get_num_collections,
    adoc_get_number_of_sections, adoc_get_section_name, adoc_insert_section,
    adoc_lookup_by_name_value, adoc_open_image_metadata, adoc_set_current, adoc_set_float,
    adoc_set_integer, adoc_set_key_value, adoc_set_two_integers, adoc_transfer_section,
    adoc_transfer_to_new_type, adoc_write,
};
use crate::imod::libcfshr::b3dutil::{
    b3d_output_file_type, override_output_type, override_write_bytes, set_4_bit_output_mode,
    set_float_output_for_entered_mode, set_output_type_from_string, write_16_bit_mode_for_floats,
};
use crate::imod::libcfshr::cubinterp::cubinterp;
use crate::imod::libcfshr::extraheader::{
    copy_extra_header_section, get_extra_header_max_sec_size, get_extra_header_sec_offset,
    get_extra_header_tilts_fortran,
};
use crate::imod::libcfshr::filtxcorr::{
    fourier_crop_sizes, fourier_expand_image, fourier_reduce_image, fourier_shift_image, nice_frame,
};
use crate::imod::libcfshr::linearxforms::{xfmult, xfunit};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_float, pip_get_float_array, pip_get_integer, pip_get_integer_array,
    pip_get_non_option_arg, pip_get_string, pip_get_three_integers, pip_get_two_floats,
    pip_get_two_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::robuststat::rs_mad_median_outliers;
use crate::imod::libcfshr::taperpad::{
    slice_edge_median, slice_noise_taper_pad, slice_taper_out_pad,
};
use crate::imod::libfft::odfft::nice_fft_limit;
use crate::imod::libfft::todfft::todfft_c;
use crate::imod::libiimod::iimage::{
    IIFILE_HDF, IIFILE_MRC, ii_allow_multi_volume, ii_best_tile_size, ii_read_section_float,
    ii_sync_from_mrc_header, ii_write_header,
};
use crate::imod::libiimod::mrcfiles::{
    IIUNIT_4BIT_MODE, IIUNIT_HALF_XSIZE, MRC_LABEL_SIZE, MRC_NLABELS, MrcHeader, fix_title_padding,
    mrc_head_write, mrc_init_output_header,
};
use crate::imod::libiimod::unit_fileio::{
    IIFILE_SHR_MEM, iiu_alt_chunk_sizes, iiu_close, iiu_file_info, iiu_file_type,
    iiu_get_exit_on_error, iiu_get_ii_file, iiu_mrc_header, iiu_open, iiu_ret_adoc_index,
    iiu_ret_chunk_sizes, iiu_set_hdf_compression, iiu_set_position, iiu_trans_adoc_sections,
    iiu_volume_open, iiu_write_global_adoc, iiu_write_lines,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_extended_data, iiu_alt_extended_type, iiu_alt_num_extended, iiu_create_header,
    iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_num_extended, iiu_trans_extended_data,
    iiu_write_header,
};
use crate::imod::libiimod::unit_reduced::{SLICE_MODE_FLOAT, iiu_read_binned, iiu_read_reduced};
use crate::imod::libwarp::maggradfield::{add_mag_grad_field, make_mag_grad_field};
use crate::imod::libwarp::warpfiles::get_linear_transform;
use crate::imod::libwarp::warpinterp::warp_interp;
use crate::imod::libwarp::warputils::{find_max_grid_size, get_size_adjusted_grid};
use std::ffi::CString;
use std::io::BufReader;

/// Source fallback PIP table (`newstack.f90:151`), retained as 74 `@`-separated
/// entries rather than a hand-maintained Rust option list.
/// `LIMGRADSEC` (`newstack.f90:18`).
const LIM_GRAD_SEC: i32 = 10000;

/// `MAXTEMP` (`newstack.f90:19`).
const MAX_TEMP: i64 = 5_000_000;

const NEWSTACK_OPTIONS: &str = "input:InputFile:FNM:@output:OutputFile:FNM:@fileinlist:FileOfInputs:FN:@fileoutlist:FileOfOutputs:FN:@reverse:ReverseInputFileOrder:I:@split:SplitStartingNumber:I:@append:AppendExtension:CH:@format:FormatOfOutputFile:CH:@compression:HDFCompressionIndex:I:@volumes:VolumesToRead:LI:@3d:Store3DVolumes:I:@chunk:ChunkSizesInXYZ:IT:@mdoc:UseMdocFiles:B:@remove:RemoveForMdocName:CH:@addback:AddBackForMdocName:CH:@pixel:PixelSizeFromMdoc:B:@tilt:TiltAngleFile:FN:@reorder:ReorderByTiltAngle:I:@angle:AngleFileToReorder:FN:@newangle:NewAngleOutputFile:I:@secs:SectionsToRead:LIM:@samesec:SameSectionsToRead:B:@fromone:NumberedFromOne:B:@exclude:ExcludeSections:LI:@twodir:TwoDirectionTiltSeries:B:@skip:SkipSectionIncrement:I:@numout:NumberToOutput:IAM:@replace:ReplaceSections:LI:@blank:BlankOutput:B:@offset:OffsetsInXandY:FAM:@applyfirst:ApplyOffsetsFirst:B:@xform:TransformFile:FN:@uselines:UseTransformLines:LIM:@onexform:OneTransformPerFile:B:@phase:PhaseShiftFFT:B:@rotate:RotateByAngle:F:@expand:ExpandByFactor:F:@shrink:ShrinkByFactor:F:@antialias:AntialiasFilter:I:@bin:BinByFactor:I:@oddeven:AllowOddEvenChange:B:@ftreduce:FourierReduceByFactor:F:@ftexpand:FourierExpandByFactor:F:@noise:NoisePadForFFT:B:@distort:DistortionField:FN:@imagebinned:ImagesAreBinned:F:@fields:UseFields:LIM:@subarea:SubareaOffsetsXandY:FAM:@gradient:GradientFile:FN:@origin:AdjustOrigin:B:@linear:LinearInterpolation:B:@nearest:NearestNeighbor:B:@size:SizeToOutputInXandY:IP:@mode:ModeToOutput:I:@bytes:BytesSignedInOutput:I:@strip:StripExtraHeader:B:@float:FloatDensities:I:@meansd:MeanAndStandardDeviation:FP:@contrast:ContrastBlackWhite:IP:@scale:ScaleMinAndMax:FP:@map:MapFromRange:FP:@multadd:MultiplyAndAdd:FPM:@fixrange:FixRangeIfNeeded:FP:@rfparam:RangeFixingParams:FP:@fill:FillValue:F:@taper:TaperAtFill:IP:@memory:MemoryLimit:I:@test:TestLimits:IP:@megasec:MaxMegaSections:I:@quiet:QuietOutput:B:@print:PrintXYSizeAndExit:B:@verbose:VerboseOutput:I:@param:ParameterFile:PF:@help:usage:B:";

/// Original program `newstack` (`newstack.f90:15`).
///
/// Implements the native MRC stream path: section selection/reversal/blanking
/// and mode conversion use IMOD's own image dispatch, as in the source.
pub fn newstack() {
    //
    // Pip startup: set error, parse options, check help, set flag if used
    // (`newstack.f90:180-186`).
    //
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    pip_read_or_parse_options(
        &[NEWSTACK_OPTIONS],
        74,
        "newstack",
        "ERROR: NEWSTACK - ",
        true,
        2,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    // `pipinput = numOptArg + numNonOptArg > 0` (`newstack.f90:186`).  The
    // source's interactive branch (`read(5,*)` prompting from
    // `newstack.f90:341`) is not translated; without PIP entries this program
    // unit has nothing to do.
    let pipinput = num_opt_arg + num_non_opt_arg > 0;
    if !pipinput {
        exit_error("Interactive entry is not supported by this translation");
    }
    // `newstack.f90:302` suppresses the iiunit open/header banner while it
    // probes each input; command-owned diagnostics remain enabled below.
    unsafe { ialprt(false) };
    let mut input_entries = 0;
    let mut output_entries = 0;
    let mut section_list_entries = 0;
    // Keep these collections in the same input/output-file order as the
    // Fortran arrays INFILE, OUTFILE, NLIST, LISTIND, and NUMSECOUT.  This
    // deliberately does not collapse all inputs into one synthetic stream:
    // section provenance matters when repeated -input/-secs options are used.
    let (mut input_names, mut output_names, mut section_lists, mut num_output_sections) = (
        Vec::<String>::new(),
        Vec::<String>::new(),
        Vec::<Vec<i32>>::new(),
        Vec::<i32>::new(),
    );
    let (
        mut input_list_name,
        mut output_list_name,
        mut reverse_count,
        mut blank,
        mut mode,
        mut same_sections,
        mut numbered_from_one,
        mut list_increment,
        mut excluded_sections,
        mut xf_file,
        mut if_linear,
        mut size_to_output,
        mut tilt_angle_file,
        mut reorder_by_tilt,
        mut angle_file_to_reorder,
        mut new_angle_output_file,
    ) = (
        String::new(),
        String::new(),
        None::<i32>,
        false,
        None,
        false,
        false,
        1_i32,
        Vec::<i32>::new(),
        String::new(),
        0_i32,
        None::<[i32; 2]>,
        String::new(),
        0_i32,
        String::new(),
        String::new(),
    );
    let mut adjust_origin = false;
    let mut print_size_and_exit = false;
    let mut expand_factor = 0.0_f32;
    let mut rotate_angle = 0.0_f32;
    let mut linear_entered = false;
    let mut nearest_entered = false;
    let mut bin_factor = 1_i32;
    // Source `readReduction` (`newstack.f90:240`), set from the binning factor
    // at `newstack.f90:1047`.
    let mut read_reduction = 1.0_f32;
    // `newstack.f90:230-241`.
    let mut ind_filter = 6_i32;
    let mut shrink_factor = 1.0_f32;
    let mut read_shrunk = false;
    let mut lines_shrink = 0_i32;
    let mut odd_even_ok = 0_i32;
    let mut quiet = false;
    let mut bytes_signed = None::<i32>;
    let mut pixel_from_mdoc = false;
    let mut use_mdoc_files = false;
    // `newstack.f90:196-201, 207, 239`: the distortion-field and mag-gradient
    // state.  `LIMGRADSEC` is 10000 (`newstack.f90:18`) and `lmGrid` starts at
    // 200 (`newstack.f90:239`).
    // `xfText` (`newstack.f90:832-833, 1189, 1223`) is the 13-character
    // field of the output title.
    let mut xf_text = String::new();
    let mut if_warping = 0_i32;
    let mut warp_num_fields = 0_i32;
    let mut if_control = 0_i32;
    let mut warp_pixel_size = 0.0_f32;
    let mut idf_file = String::new();
    let mut mag_grad_file = String::new();
    let mut if_distort = 0_i32;
    let mut if_mag_grad = 0_i32;
    let mut binning_of_input = 1.0_f32;
    let mut warp_scale = 0.0_f32;
    let mut lm_grid = 200_i32;
    let mut idf_use = Vec::<i32>::new();
    let mut warp_x_offsets = Vec::<f32>::new();
    let mut warp_y_offsets = Vec::<f32>::new();
    let mut pixel_mag_grad = 0.0_f32;
    let mut axis_rot = 0.0_f32;
    let mut num_mag_grad = 0_i32;
    let mut tilt_angles = vec![0.0_f32; LIM_GRAD_SEC as usize];
    let mut dmag_per_micron = vec![0.0_f32; LIM_GRAD_SEC as usize];
    let mut rot_per_micron = vec![0.0_f32; LIM_GRAD_SEC as usize];
    let mut remove_from_name = String::new();
    let mut add_to_name = String::new();
    // `newstack.f90:234-235`.
    let mut series_base = -1_i32;
    let mut series_ext = String::new();
    // `newstack.f90:216`, `newstack.f90:222-223`.
    let mut strip_extra = false;
    let mut num_taper = 0_i32;
    let mut inside_taper = 0_i32;
    // `newstack.f90:243`, `newstack.f90:286`.
    let mut two_directions = false;
    let mut lim_sec = 1_000_000_i32;
    let mut non_options = Vec::<String>::new();
    let mut one_per_file = false;
    let mut fill_value = None::<f32>;
    // `phaseShift`, `ftReduceFac`, `ftExpandFac` and `fourierScaling`
    // (`newstack.f90:751-766`).
    let mut phase_shift = false;
    let mut ft_reduce_fac = 0.0_f32;
    let mut ft_expand_fac = 0.0_f32;
    let mut fourier_scaling = false;
    // `noisePad` (`newstack.f90:263`).
    let mut noise_pad = false;
    // `nxFSpad`, `nyFSpad`, `nxFCropPad`, `nyFCropPad` and `actualFac`
    // (`newstack.f90:1131-1143`); zero until `fourierCropSizes` sets them.
    let mut nx_fspad = 0_i32;
    let mut ny_fspad = 0_i32;
    let mut nx_fcrop_pad = 0_i32;
    let mut ny_fcrop_pad = 0_i32;
    let mut actual_fac = 0.0_f32;
    // `iVerbose` (`newstack.f90:95`), zero until `-verbose` is read at
    // `newstack.f90:308`.
    let mut i_verbose = 0_i32;
    // `loadTime`, `saveTime`, `rotTime` and `taperTime` (`newstack.f90:119-120,
    // 224-227`), the wall-clock accumulators the `newstack.f90:2788` report
    // prints.  Nothing else reads them.
    let (mut load_time, mut save_time, mut rot_time, mut taper_time) = (0.0_f64, 0., 0., 0.);
    // gfortran writes a list-directed `real*4` as `G16.9E2` with a scale
    // factor of 1 (`libgfortran/io/write.c`, `write_real` and
    // `set_fnode_default`): a value whose decimal exponent falls in 0..=9
    // prints in F editing with `9 - exponent` fraction digits, right
    // justified in 12 and followed by four blanks; anything else prints as
    // one digit, eight fraction digits and a signed two-digit exponent,
    // right justified in 16.  `{:16.9}` is plain F editing and matches
    // neither.  Integers are right justified in 11 (`integer*4`) or 20
    // (`integer(kind = 8)`), logicals in 1, and every item is preceded by a
    // one-blank separator, with the record itself starting with a blank.
    let list_real = |value: f32| -> String {
        let magnitude = value.abs();
        let mut exponent = 1_i32;
        if magnitude != 0.0 {
            let scientific = format!("{:.8e}", magnitude);
            exponent = scientific
                .split_once('e')
                .unwrap()
                .1
                .parse::<i32>()
                .unwrap()
                + 1;
        }
        if (0..=9).contains(&exponent) {
            let mut text = format!("{:.*}", (9 - exponent) as usize, value);
            if exponent == 9 {
                text.push('.');
            }
            format!("{text:>12}    ")
        } else {
            let scientific = format!("{:.8e}", value);
            let (mantissa, power) = scientific.split_once('e').unwrap();
            let power = power.parse::<i32>().unwrap();
            format!(
                "{:>16}",
                format!(
                    "{}E{}{:02}",
                    mantissa,
                    if power < 0 { '-' } else { '+' },
                    power.abs()
                )
            )
        }
    };
    // Source option retrieval (`newstack.f90:305-1235`).  Everything below is
    // read out of the PIP entry table that `PipReadOrParseOptions` filled, so
    // option abbreviation, `-param` files, and illegal-option exits all behave
    // as they do natively.
    unsafe {
        let mut string_value: *mut libc::c_char = core::ptr::null_mut();
        let mut integer_value = 0_i32;
        let mut float_value = 0.0_f32;
        if pip_get_string(c"FileOfInputs".as_ptr(), &raw mut string_value) == 0 {
            input_list_name = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        pip_number_of_entries(c"InputFile".as_ptr(), &raw mut input_entries);
        for _ in 0..input_entries {
            if pip_get_string(c"InputFile".as_ptr(), &raw mut string_value) == 0 {
                input_names.push(
                    std::ffi::CStr::from_ptr(string_value)
                        .to_string_lossy()
                        .into_owned(),
                );
                libc::free(string_value.cast());
            }
        }
        // `numInFiles = numInputFiles + max(0, numNonOptArg - 1)` and
        // `numOutFiles = numOutputFiles + min(1, numNonOptArg)`
        // (`newstack.f90:311, 558`): every non-option argument but the last is
        // an input, the last one is the output.
        for index in 0..num_non_opt_arg {
            if pip_get_non_option_arg(index, &raw mut string_value) == 0 {
                non_options.push(
                    std::ffi::CStr::from_ptr(string_value)
                        .to_string_lossy()
                        .into_owned(),
                );
                libc::free(string_value.cast());
            }
        }
        pip_number_of_entries(c"SectionsToRead".as_ptr(), &raw mut section_list_entries);
        if pip_get_integer(c"SkipSectionIncrement".as_ptr(), &raw mut integer_value) == 0 {
            list_increment = integer_value;
        }
        if pip_get_boolean(c"BlankOutput".as_ptr(), &raw mut integer_value) == 0 {
            blank = integer_value != 0;
        }
        // `newstack.f90:291`.
        if pip_get_integer(c"MaxMegaSections".as_ptr(), &raw mut integer_value) == 0 {
            lim_sec = 1_000_000 * 1.max(integer_value);
        }
        if pip_get_boolean(c"PrintXYSizeAndExit".as_ptr(), &raw mut integer_value) == 0 {
            print_size_and_exit = integer_value != 0;
        }
        quiet = print_size_and_exit;
        // `newstack.f90:318`.
        if pip_get_boolean(c"StripExtraHeader".as_ptr(), &raw mut integer_value) == 0 {
            strip_extra = integer_value != 0;
        }
        // `newstack.f90:322`.
        if pip_get_boolean(c"TwoDirectionTiltSeries".as_ptr(), &raw mut integer_value) == 0 {
            two_directions = integer_value != 0;
        }
        if pip_get_boolean(c"NumberedFromOne".as_ptr(), &raw mut integer_value) == 0 {
            numbered_from_one = integer_value != 0;
        }
        if pip_get_boolean(c"SameSectionsToRead".as_ptr(), &raw mut integer_value) == 0 {
            same_sections = integer_value != 0;
        }
        if same_sections && section_list_entries > 1 {
            exit_error("You cannot enter -samesec with multiple section list entries");
        }
        // `newstack.f90:327-328`.
        if same_sections && two_directions {
            exit_error("You cannot enter -samesec with -twodir");
        }
        if pip_get_integer(c"ReverseInputFileOrder".as_ptr(), &raw mut integer_value) == 0 {
            if !input_list_name.is_empty() {
                exit_error("You cannot enter -reverse with an input file list");
            }
            reverse_count = Some(integer_value);
        }
        if pip_get_integer(c"BytesSignedInOutput".as_ptr(), &raw mut integer_value) == 0 {
            bytes_signed = Some(integer_value);
        }
        if pip_get_string(c"ExcludeSections".as_ptr(), &raw mut string_value) == 0 {
            let list = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
            let mut parsed = vec![0_i32; 1_000_000];
            let (mut count, mut limit) = (0, 1_000_000);
            if parselist2(&list, &mut parsed, &mut count, &mut limit).is_err() {
                exit_error("Processing section list in list of input files");
            }
            excluded_sections.extend_from_slice(&parsed[..count as usize]);
        }
        // `newstack.f90:376-385`.
        if pip_get_string(c"FormatOfOutputFile".as_ptr(), &raw mut string_value) == 0 {
            let value = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
            let error = set_output_type_from_string(&value);
            if error == -5 {
                exit_error("HDF files are not supported by this IMOD package");
            }
            if error == -6 {
                exit_error("JPEG files are not supported by this IMOD package");
            }
            if error < 0 {
                exit_error("Unrecognized entry for output file format");
            }
        }
        // `newstack.f90:397-403`.  `-remove` implies `-mdoc`, and `-addback`
        // without `-remove` is an error.
        if pip_get_boolean(c"UseMdocFiles".as_ptr(), &raw mut integer_value) == 0 {
            use_mdoc_files = integer_value != 0;
        }
        if pip_get_boolean(c"PixelSizeFromMdoc".as_ptr(), &raw mut integer_value) == 0 {
            pixel_from_mdoc = integer_value != 0;
        }
        let remove_entered =
            pip_get_string(c"RemoveForMdocName".as_ptr(), &raw mut string_value) == 0;
        if remove_entered {
            remove_from_name = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
            use_mdoc_files = true;
        }
        if pip_get_string(c"AddBackForMdocName".as_ptr(), &raw mut string_value) == 0 {
            if !remove_entered {
                exit_error("You cannot enter -addback without -remove");
            }
            add_to_name = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        //
        // Output files (`newstack.f90:556-563`).
        if pip_get_string(c"FileOfOutputs".as_ptr(), &raw mut string_value) == 0 {
            output_list_name = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        pip_number_of_entries(c"OutputFile".as_ptr(), &raw mut output_entries);
        for _ in 0..output_entries {
            if pip_get_string(c"OutputFile".as_ptr(), &raw mut string_value) == 0 {
                output_names.push(
                    std::ffi::CStr::from_ptr(string_value)
                        .to_string_lossy()
                        .into_owned(),
                );
                libc::free(string_value.cast());
            }
        }
        // `newstack.f90:564-568`.
        if pip_get_integer(c"SplitStartingNumber".as_ptr(), &raw mut integer_value) == 0 {
            series_base = integer_value;
        }
        if pip_get_string(c"AppendExtension".as_ptr(), &raw mut string_value) == 0 {
            series_ext = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        //
        // Size, mode and transforms (`newstack.f90:685-1130`).
        let (mut size_x, mut size_y) = (-1_i32, -1_i32);
        if pip_get_two_integers(
            c"SizeToOutputInXandY".as_ptr(),
            &raw mut size_x,
            &raw mut size_y,
        ) == 0
        {
            size_to_output = Some([size_x, size_y]);
        }
        if pip_get_integer(c"ModeToOutput".as_ptr(), &raw mut integer_value) == 0 {
            mode = Some(set_float_output_for_entered_mode(integer_value));
        }
        if pip_get_boolean(c"LinearInterpolation".as_ptr(), &raw mut integer_value) == 0
            && integer_value != 0
        {
            linear_entered = true;
            if_linear = 1;
        }
        if pip_get_boolean(c"NearestNeighbor".as_ptr(), &raw mut integer_value) == 0
            && integer_value != 0
        {
            nearest_entered = true;
            if_linear = -1;
        }
        if pip_get_string(c"TransformFile".as_ptr(), &raw mut string_value) == 0 {
            xf_file = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        // `UseTransformLines` is read inside `getItemsToUse`
        // (`newstack.f90:3333-3344`), which loops over every entry; reading it
        // here would consume the first one.
        if pip_get_boolean(c"OneTransformPerFile".as_ptr(), &raw mut integer_value) == 0 {
            one_per_file = integer_value != 0;
        }
        if pip_get_boolean(c"PhaseShiftFFT".as_ptr(), &raw mut integer_value) == 0 {
            phase_shift = integer_value != 0;
        }
        if pip_get_float(c"FourierReduceByFactor".as_ptr(), &raw mut float_value) == 0 {
            ft_reduce_fac = float_value;
        }
        if pip_get_float(c"FourierExpandByFactor".as_ptr(), &raw mut float_value) == 0 {
            ft_expand_fac = float_value;
        }
        // `ierr = PipGetLogical('NoisePadForFFT', noisePad)` (`newstack.f90:754`).
        if pip_get_boolean(c"NoisePadForFFT".as_ptr(), &raw mut integer_value) == 0 {
            noise_pad = integer_value != 0;
        }
        fourier_scaling = ft_reduce_fac > 0. || ft_expand_fac > 0.;
        if pip_get_float(c"RotateByAngle".as_ptr(), &raw mut float_value) == 0 {
            rotate_angle = float_value;
        }
        if pip_get_float(c"ExpandByFactor".as_ptr(), &raw mut float_value) == 0 {
            expand_factor = float_value;
        }
        if pip_get_float(c"FillValue".as_ptr(), &raw mut float_value) == 0 {
            fill_value = Some(float_value);
        }
        // `newstack.f90:982`.
        pip_get_two_integers(
            c"TaperAtFill".as_ptr(),
            &raw mut num_taper,
            &raw mut inside_taper,
        );
        // `ierr = PipGetInteger('VerboseOutput', iVerbose)`
        // (`newstack.f90:308`).
        if pip_get_integer(c"VerboseOutput".as_ptr(), &raw mut integer_value) == 0 {
            i_verbose = integer_value;
        }
        // `newstack.f90:979-980`.
        if pip_get_string(c"DistortionField".as_ptr(), &raw mut string_value) == 0 {
            idf_file = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        if pip_get_string(c"GradientFile".as_ptr(), &raw mut string_value) == 0 {
            mag_grad_file = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        // `newstack.f90:1046`.
        if pip_get_integer(c"BinByFactor".as_ptr(), &raw mut integer_value) == 0 {
            bin_factor = integer_value;
            if bin_factor <= 0 {
                exit_error("Binning factor must be a positive number");
            }
        }
        if pip_get_boolean(c"AllowOddEvenChange".as_ptr(), &raw mut integer_value) == 0 {
            odd_even_ok = integer_value;
        }
        if pip_get_boolean(c"AdjustOrigin".as_ptr(), &raw mut integer_value) == 0 {
            adjust_origin = integer_value != 0;
        }
        if pip_get_boolean(c"QuietOutput".as_ptr(), &raw mut integer_value) == 0 {
            quiet = quiet || integer_value != 0;
        }
        //
        // Tilt angle handling (`newstack.f90:989-1020`).
        if pip_get_integer(c"ReorderByTiltAngle".as_ptr(), &raw mut integer_value) == 0 {
            reorder_by_tilt = integer_value;
        }
        if pip_get_string(c"AngleFileToReorder".as_ptr(), &raw mut string_value) == 0 {
            angle_file_to_reorder = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        if pip_get_string(c"NewAngleOutputFile".as_ptr(), &raw mut string_value) == 0 {
            new_angle_output_file = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
        if pip_get_string(c"TiltAngleFile".as_ptr(), &raw mut string_value) == 0 {
            // `newstack.f90:1008`.
            if strip_extra {
                exit_error("You cannot enter both -tilt and -strip");
            }
            tilt_angle_file = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
        }
    }
    if linear_entered && nearest_entered {
        exit_error("You cannot enter both -linear and -nearest");
    }
    // `newstack.f90:751-765`.  The Fourier resize itself is not translated on
    // this route; only its entry legality is enforced here, so that the
    // rejected entries exit exactly as native does.
    if (phase_shift || fourier_scaling) && if_linear != 0 {
        exit_error("You cannot enter -phase, -ftreduce, or -ftExpand with -linear or -nearest");
    }
    if ft_reduce_fac > 0. && ft_reduce_fac <= 1. {
        exit_error("Factor for reducing with FFT must be > 1");
    }
    if ft_expand_fac > 0. && ft_expand_fac <= 1. {
        exit_error("Factor for expanding with FFT must be > 1");
    }
    if ft_reduce_fac > 0. && ft_expand_fac > 0. {
        exit_error("You cannot enter both -ftreduce and -ftexpand");
    }
    if let Some(bytes_signed) = bytes_signed {
        override_write_bytes(bytes_signed);
    }
    // The source treats all non-option arguments except the last as inputs,
    // and the final one as an output.  Its list-file forms are mutually
    // exclusive with these direct forms.
    // `numInFiles = numInputFiles + max(0, numNonOptArg - 1)`
    // (`newstack.f90:311-313`): a lone non-option argument is the output file.
    if !input_list_name.is_empty() && (!input_names.is_empty() || non_options.len() > 1) {
        exit_error("You cannot enter both input files and an input list file");
    }
    if !input_list_name.is_empty() {
        // `call dopen(7, inFileList, 'ro', 'f')` (`newstack.f90:362`).
        drop(dopen(7, &input_list_name, "ro", "f"));
        let Ok(text) = std::fs::read_to_string(&input_list_name) else {
            exit_error("Opening input file list");
        };
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let Some(count) = lines.next().and_then(|value| value.parse::<usize>().ok()) else {
            exit_error("Input list file must start with a positive number of files");
        };
        for _ in 0..count {
            let Some(name) = lines.next() else {
                exit_error("Reading input file list");
            };
            input_names.push(name.to_owned());
            // The source list-file form also carries a section list after each
            // filename.  A bare slash represents every source section.
            let Some(list) = lines.next() else {
                exit_error(
                    "There must be a readable section list after each filename in list of input files",
                );
            };
            let mut parsed = vec![0_i32; 1_000_000];
            let (mut count, mut limit) = (0, 1_000_000);
            if parselist2(list, &mut parsed, &mut count, &mut limit).is_err() {
                exit_error("Invalid section list in input list");
            }
            parsed.truncate(count as usize);
            section_lists.push(parsed);
        }
    } else if !non_options.is_empty() {
        input_names.extend(non_options[..non_options.len() - 1].iter().cloned());
        output_names.push(non_options[non_options.len() - 1].clone());
    }
    // `newstack.f90:355`.
    if input_names.is_empty() {
        exit_error("No input file specified");
    }
    // `newstack.f90:560-562`.  This sits in the output-file block at
    // `newstack.f90:556-568`, after `No input file specified`.
    if !output_list_name.is_empty() && (!output_names.is_empty() || !non_options.is_empty()) {
        exit_error("You cannot enter both output files and an output list file");
    }
    // `newstack.f90:564-567`: the `-split` check reads `numOutFiles`, which is
    // the `-output` entries plus any non-option output argument and is -1 when
    // `-fileoutlist` was given.  It runs at `newstack.f90:566`, long after
    // `No input file specified` (`newstack.f90:355`), so `-split 0` with two
    // outputs and no input reports the missing input first.
    if series_base >= 0 && output_names.len() != 1 {
        exit_error("There must be only one output file name for series of numbered files");
    }
    if !output_list_name.is_empty() {
        // `call dopen(7, outFileList, 'ro', 'f')` (`newstack.f90:589`).
        drop(dopen(7, &output_list_name, "ro", "f"));
        let Ok(text) = std::fs::read_to_string(&output_list_name) else {
            exit_error("Opening output file list");
        };
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let Some(count) = lines
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|count| *count > 0)
        else {
            exit_error("The output list file must start with a positive number of files to output");
        };
        for _ in 0..count {
            let (Some(name), Some(count)) = (
                lines.next(),
                lines.next().and_then(|value| value.parse::<i32>().ok()),
            ) else {
                exit_error("Reading output file list");
            };
            output_names.push(name.to_owned());
            num_output_sections.push(count);
        }
    }
    //
    // Get HDF and volume related options (`newstack.f90:375-405`).  The
    // source runs these in a second `if (pipInput)` block here, after
    // `numInFiles` is final and `listVolumes(numInFiles)` is allocated,
    // because `-volumes` is capped by that count; `-format`, `-mdoc`,
    // `-pixelfrommdoc`, `-remove` and `-addback` are in the same source block
    // and are read with the other options above.
    //
    // `newstack.f90:247-248` and `newstack.f90:282` initialisers.
    let mut num_vol_read = 0_usize;
    // `listVolumes` is allocated but never initialised (`newstack.f90:370`);
    // `openInputFile` reads it only for `indInFile <= numVolRead`.
    let mut list_volumes = vec![0_i32; input_names.len()];
    let mut if3d_volumes = 0_i32;
    let mut nx_tile = 0_i32;
    let mut ny_tile = 0_i32;
    let mut nz_chunk = 1_i32;
    let mut i_hdf_compression = -1_i32;
    // `newstack.f90:268-269`.
    let mut need_close1 = 0_i32;
    let mut need_close2 = 0_i32;
    let if_chunk_in;
    unsafe {
        let mut string_value: *mut libc::c_char = core::ptr::null_mut();
        // `newstack.f90:386-390`.
        if pip_get_string(c"VolumesToRead".as_ptr(), &raw mut string_value) == 0 {
            let list = std::ffi::CStr::from_ptr(string_value)
                .to_string_lossy()
                .into_owned();
            libc::free(string_value.cast());
            // `call parseList2(listString, inList, numVolRead, limSec)`: with
            // a positive `limList` the routine prints its own diagnostic and
            // exits, so the source discards the outcome and so does this.
            let mut parsed = vec![0_i32; lim_sec as usize];
            let (mut count, mut limit) = (0_i32, lim_sec);
            drop(parselist2(&list, &mut parsed, &mut count, &mut limit));
            num_vol_read = (count as usize).min(input_names.len());
            list_volumes[..num_vol_read].copy_from_slice(&parsed[..num_vol_read]);
        }
        // `ifChunkIn = 1 - PipGetThreeIntegers('ChunkSizesInXYZ', nxTile,
        // nyTile, nzChunk)` (`newstack.f90:391-396`).  PIP leaves the three
        // targets alone when the option is absent, so they keep the source
        // defaults 0, 0 and 1.
        if_chunk_in = 1 - pip_get_three_integers(
            c"ChunkSizesInXYZ".as_ptr(),
            &raw mut nx_tile,
            &raw mut ny_tile,
            &raw mut nz_chunk,
        );
        if if_chunk_in > 0 {
            if3d_volumes = 1;
        }
        pip_get_integer(c"Store3DVolumes".as_ptr(), &raw mut if3d_volumes);
        if if_chunk_in > 0 && if3d_volumes < 0 {
            exit_error("You cannot enter chunk sizes and forbid volume output with -3d -1");
        }
        if if3d_volumes > 0 {
            override_output_type(5);
        }
        // `newstack.f90:399`.
        pip_get_integer(c"HDFCompressionIndex".as_ptr(), &raw mut i_hdf_compression);
    }
    if output_names.is_empty() {
        exit_error("No output file specified");
    }
    if list_increment < 1 {
        // This is source `ix1 = max(1, listIncrement)` at newstack.f90:519.
        list_increment = 1;
    }
    if numbered_from_one {
        for section in &mut excluded_sections {
            *section -= 1;
        }
    }
    // `newstack.f90:408-409`.
    if two_directions && input_names.len() != 2 {
        exit_error("There must be exactly two input files to use -twodir");
    }
    // `newstack.f90:470-535`: the section lists are read inside the source's
    // input-file scan, so `newstack.f90:493`'s `You cannot enter section lists
    // with -twodir` cannot be reached before `No input file specified`
    // (`newstack.f90:355`) or `There must be exactly two input files to use
    // -twodir` (`newstack.f90:409`).  Parsing them in the option block ahead of
    // both reported the wrong error for `-twodir 1 -secs ...` with no input.
    unsafe {
        let mut string_value: *mut std::os::raw::c_char = std::ptr::null_mut();
        for _ in 0..section_list_entries {
            if pip_get_string(c"SectionsToRead".as_ptr(), &raw mut string_value) == 0 {
                // `newstack.f90:491-493`.
                if two_directions {
                    exit_error("You cannot enter section lists with -twodir");
                }
                let list = std::ffi::CStr::from_ptr(string_value)
                    .to_string_lossy()
                    .into_owned();
                libc::free(string_value.cast());
                // `newstack.f90:286-291`: `limSec` bounds the section list and
                // `-megasec` raises it.
                let mut sections = vec![0_i32; lim_sec.max(1) as usize];
                let (mut count, mut limit) = (0, lim_sec);
                if parselist2(&list, &mut sections, &mut count, &mut limit).is_err() {
                    exit_error("Processing section list in list of input files");
                }
                sections.truncate(count as usize);
                section_lists.push(sections);
            }
        }
    }
    if input_names.len() == 1 && section_lists.len() > 1 {
        let mut combined = Vec::new();
        for list in section_lists.drain(..) {
            combined.extend(list);
        }
        section_lists.push(combined);
    }
    if section_lists.len() > input_names.len() {
        exit_error("Too many section lists");
    }
    if same_sections && section_lists.len() == 1 && input_names.len() > 1 {
        let list = section_lists[0].clone();
        section_lists.resize(input_names.len(), list);
    }
    while section_lists.len() < input_names.len() {
        section_lists.push(Vec::new());
    }
    if let Some(number) = reverse_count {
        if number.unsigned_abs() as usize > input_names.len() {
            exit_error("The entry to -reverse is bigger than the number of input files");
        }
        let count = if number == 0 {
            input_names.len()
        } else {
            number.unsigned_abs() as usize
        };
        if number >= 0 {
            input_names[..count].reverse();
            section_lists[..count].reverse();
        } else {
            let first = input_names.len() - count;
            input_names[first..].reverse();
            section_lists[first..].reverse();
        }
    }
    let mut transforms = Vec::<[f32; 6]>::new();
    if !xf_file.is_empty() {
        // `ierr = readCheckWarpFile(xfFile, 0, 1, ...)` and
        // `if (ierr < -1) call exitError(listString)` (`newstack.f90:781-783`).
        // The `ierr >= 0` warping-file branch is not translated on this route.
        let mut warp_name = CString::new(xf_file.as_bytes())
            .unwrap()
            .into_bytes_with_nul();
        let mut list_string = vec![0_i8; 1024];
        let (mut idf_nx, mut idf_ny, mut num_xforms, mut idf_binning, mut warp_flags) =
            (0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
        let mut pixel_size = 0.0_f32;
        let warp_check = unsafe {
            crate::imod::libwarp::warputils::read_check_warp_file(
                warp_name.as_mut_ptr().cast(),
                0,
                1,
                &raw mut idf_nx,
                &raw mut idf_ny,
                &raw mut num_xforms,
                &raw mut idf_binning,
                &raw mut pixel_size,
                &raw mut warp_flags,
                list_string.as_mut_ptr(),
                list_string.len() as i32,
            )
        };
        if warp_check < -1 {
            let message = unsafe { std::ffi::CStr::from_ptr(list_string.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            exit_error(message.trim_end());
        }
        if warp_check >= 0 {
            //
            // `newstack.f90:784-797`: the file is a warping file, so the
            // linear part of each warping is the transform and the grids are
            // fetched per section later.
            //
            println!("Warping file opened: {xf_file}");
            if_warping = 1;
            if (warp_flags / 2) % 2 != 0 {
                if_control = 1;
            }
            if num_xforms > 40_000 {
                exit_error("Too many sections in warping file for transform array");
            }
            warp_num_fields = num_xforms;
            warp_pixel_size = pixel_size;
            // `warpScale` and the transforms themselves need the first input
            // file's pixel spacing, which this route does not have until the
            // preliminary file pass below has run; both are finished there.
        } else {
            // `call dopen(3, xfFile, 'ro', 'f')` (`newstack.f90:799`).
            drop(dopen(3, &xf_file, "ro", "f"));
            let Ok(file) = std::fs::File::open(&xf_file) else {
                exit_error("Opening transform file");
            };
            match xfrdall2(&mut BufReader::new(file), &mut transforms, 40_000) {
                0 if !transforms.is_empty() => (),
                0 => {
                    exit_error("The transform file contains no transforms");
                }
                1 => {
                    exit_error("Too many transforms in file for transform array");
                }
                _ => {
                    exit_error("Reading transform file");
                }
            }
        }
        // `newstack.f90:832-833`.
        xf_text = if if_warping != 0 {
            ", warped".to_owned()
        } else {
            ", transformed".to_owned()
        };
    }
    unsafe {
        let mut routes = Vec::<(usize, i32)>::new();
        let mut mdoc_pixel_spacing = Vec::<Vec<Option<f32>>>::new();
        let mut first_header: Option<MrcHeader> = None;
        // `newstack.f90:427, 466-468`: differing X/Y sizes are not an error in
        // general; the source only records `sizesMatch`, which it rejects
        // later for `-phase`/`-ftreduce`.  `nxMax`/`nyMax` size the
        // distortion field grid.
        let mut sizes_match = true;
        let mut nx_max = 0_i32;
        let mut ny_max = 0_i32;
        let mut nx_tile_in = 0_i32;
        let mut ny_tile_in = 0_i32;
        let mut nz_chunk_in = 0_i32;
        for (file_index, name) in input_names.iter().enumerate() {
            let Ok(name) = CString::new(name.as_bytes()) else {
                exit_error("Invalid input file name");
            };
            // Source opens this preliminary pass through `openInputFile` and
            // `irdhdr`, before closing the iiunit and reopening it for the
            // streaming loop below (newstack.f90:444-465).
            open_input_file(
                file_index + 1,
                num_vol_read,
                &list_volumes,
                &input_names,
                &mut need_close1,
            );
            let (mut nxyz, mut mxyz, mut mode, mut dmin, mut dmax, mut dmean) =
                ([0_i32; 3], [0_i32; 3], 0, 0., 0., 0.);
            irdhdr(
                1,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
            let ii_file = iiu_get_ii_file(1);
            if ii_file.is_null() {
                exit_error("Opening input file");
            }
            // The unit's MRC header, not `ImodImageFile.header`.  For MRC,
            // RAW, HDF and shared-memory files `iiuOpen` copies the pointer
            // (`unit_fileio.c:269`) and the two are the same, but for every
            // other type it allocates its own `MrcHeader` and fills it through
            // `iiFillMrcHeader` (`unit_fileio.c:256-265`) -- a TIFF file's
            // `ImodImageFile.header` is the libtiff `TIFF *`
            // (`iitif.c:204, 687`), so reading it as an `MrcHeader` is garbage.
            let header = std::ptr::read(iiu_mrc_header(1, c"iiuRetBasicHead".as_ptr(), 1, 0));
            // `newstack.f90:449-464`: retain the first input file's volume
            // structure in the output if the output is already HDF, and adopt
            // its chunk sizes unless the user entered them.
            if file_index == 0 {
                iiu_ret_chunk_sizes(
                    1,
                    &raw mut nx_tile_in,
                    &raw mut ny_tile_in,
                    &raw mut nz_chunk_in,
                );
                if nz_chunk_in > 0 && b3d_output_file_type() == IIFILE_HDF {
                    if if3d_volumes == 0 {
                        if3d_volumes = 1;
                    }
                    if if_chunk_in == 0 && if3d_volumes > 0 {
                        nx_tile = nx_tile_in;
                        ny_tile = ny_tile_in;
                        nz_chunk = nz_chunk_in;
                    }
                }
            }
            iiu_close(1);
            // `newstack.f90:465`.
            if need_close1 > 0 {
                iiu_close(need_close1);
            }
            let mut spacing_for_section = vec![None; header.nz as usize];
            if pixel_from_mdoc {
                let mut montage = 0;
                let mut num_sections = 0;
                let mut section_type = 0;
                let adoc_index = adoc_open_image_metadata(
                    name.as_ptr(),
                    1,
                    &raw mut montage,
                    &raw mut num_sections,
                    &raw mut section_type,
                );
                if adoc_index >= 0 {
                    adoc_set_current(adoc_index);
                    for section in 0..header.nz {
                        let mut spacing = 0.0_f32;
                        if adoc_get_float(
                            ADOC_ZVALUE_NAME.as_ptr(),
                            section,
                            c"PixelSpacing".as_ptr(),
                            &raw mut spacing,
                        ) == 0
                        {
                            spacing_for_section[section as usize] = Some(spacing);
                        }
                    }
                    adoc_clear(adoc_index);
                }
            }
            mdoc_pixel_spacing.push(spacing_for_section);
            if let Some(ref first) = first_header {
                if header.nx != first.nx || header.ny != first.ny {
                    sizes_match = false;
                }
            } else {
                first_header = Some(std::ptr::read(&header));
            }
            nx_max = nx_max.max(header.nx);
            ny_max = ny_max.max(header.ny);
            // `newstack.f90:470-475`: the default list for a file is every
            // section, already carrying `numberOffset`.
            let number_offset = i32::from(numbered_from_one);
            if section_lists[file_index].is_empty() {
                // `newstack.f90:470-475`: with `-twodir` the first file's
                // sections are taken in reverse.
                section_lists[file_index].extend((1..=header.nz).map(|isec| {
                    let iy = if two_directions && file_index == 0 {
                        header.nz - isec
                    } else {
                        isec - 1
                    };
                    iy + number_offset
                }));
            }
            // `newstack.f90:521-535`: walk the list in steps of
            // `max(1, listIncrement)`, remove `numberOffset`, check legality,
            // and only then drop entries in `ExcludeSections`.
            let mut kept_sections = Vec::<i32>::new();
            for index in (0..section_lists[file_index].len()).step_by(list_increment as usize) {
                let section = section_lists[file_index][index] - number_offset;
                if !blank && (section < 0 || section >= header.nz) {
                    println!(
                        "\nERROR: NEWSTACK -{:9} is an illegal section number for {}",
                        section + number_offset,
                        input_names[file_index]
                    );
                    std::process::exit(1);
                }
                if !excluded_sections.contains(&section) {
                    kept_sections.push(section);
                }
            }
            section_lists[file_index] = kept_sections;
            routes.extend(
                section_lists[file_index]
                    .iter()
                    .copied()
                    .map(|section| (file_index, section)),
            );
        }
        //
        // if series, now take the one filename as root name and make filenames
        // (`newstack.f90:653-669`).  `ierr = alog10(10. * (base + total - 1))`
        // is a real*4 log truncated to an integer, and `convFormat` is
        // `(i<w>.<w>)` -- zero-padded to that width, then left-adjusted, so the
        // padding only matters when the number is shorter than the width.
        //
        if series_base >= 0 {
            let total = routes.len() as i32;
            let width = (10.0_f32 * (series_base + total - 1) as f32).log10() as i32;
            let root = output_names[0].clone();
            output_names.clear();
            num_output_sections = vec![1; total.max(0) as usize];
            for i in 1..=total {
                let number = format!(
                    "{:0width$}",
                    i + series_base - 1,
                    width = width.max(0) as usize
                );
                let number = number.trim_start().to_string();
                if series_ext.trim().is_empty() {
                    output_names.push(format!("{root}.{number}"));
                } else {
                    output_names.push(format!("{root}{number}.{}", series_ext.trim()));
                }
            }
        }
        //
        // or get all the output files and the number of sections
        // (`newstack.f90:606-635`)
        //
        // `numOutTot` is still zero here on every route this program unit
        // translates: it is only set before this point by the interactive
        // single-output arm at `newstack.f90:595-604`.  `inUnit == 7` is set at
        // `newstack.f90:584` exactly when `-fileoutlist` was entered, and that
        // branch has already filled both arrays from the list file, so
        // `pipinput .and. inUnit .ne. 7` reduces to the test below.
        if output_list_name.is_empty() && series_base < 0 {
            if output_names.len() == 1 {
                num_output_sections = vec![routes.len() as i32];
            } else if output_names.len() == routes.len() {
                num_output_sections = vec![1; output_names.len()];
            } else {
                let mut num_out_entries = 0_i32;
                pip_number_of_entries(c"NumberToOutput".as_ptr(), &raw mut num_out_entries);
                if num_out_entries == 0 {
                    exit_error("You must specify number of sections to write to each output file");
                }
                num_output_sections.clear();
                for _ in 0..num_out_entries {
                    let mut values = vec![0_i32; output_names.len()];
                    let mut num_to_get = 0_i32;
                    if pip_get_integer_array(
                        c"NumberToOutput".as_ptr(),
                        values.as_mut_ptr(),
                        &raw mut num_to_get,
                        output_names.len().saturating_sub(num_output_sections.len()) as i32,
                    ) == 0
                    {
                        num_output_sections.extend_from_slice(&values[..num_to_get as usize]);
                    }
                }
                if num_output_sections.len() != output_names.len() {
                    exit_error(
                        "The number of values for sections to output does not equal the number of output files",
                    );
                }
            }
        }
        // `newstack.f90:671-672`.
        if num_output_sections.iter().sum::<i32>() != routes.len() as i32 {
            exit_error("Number of input and output sections does not match");
        }
        // `newstack.f90:684-731` and `newstack.f90:975-1024` both run after the
        // `numOutTot .ne. listTotal` check at `newstack.f90:671-672`, so the
        // mode-16/101 block and the `-tilt`/`-reorder` block sit here and not
        // ahead of the output-section accounting.  With them earlier,
        // `-reorder 1` and two output files reported
        // `You cannot use -reorder with more than one output file` where the
        // reference reports the missing `-numout`.
        //
        // `newstack.f90:718-731`.  The source makes this check immediately
        // after reading `ModeToOutput` (`newstack.f90:684`), where `newMode`
        // still defaults to `modeOfFirst`.  This route does not have
        // `modeOfFirst` until the input scan above has run -- the source
        // establishes it at `newstack.f90:449`, inside its own input scan,
        // which it performs before reaching line 684 -- so the check lands
        // here, the earliest point at which `newMode` is known.  It is still
        // ahead of everything the source does after `newstack.f90:731`,
        // including the `-reorder`/`-tilt` handling immediately below.
        //
        let mut new_mode = mode.unwrap_or(first_header.as_ref().unwrap().mode);
        if new_mode == 16 {
            if input_names.len() > 1 {
                exit_error(
                    "Cannot output color data (mode 16); use clip for junk stacking multiple color files, use colornewst with the -cnfiles options, or set another output mode",
                );
            }
            exit_error(
                "Cannot output color data (mode 16); use colornewst instead or set another output mode",
            );
        }
        //
        // set up for making mode 101 4-bit output (`newstack.f90:726-731`)
        let mut pack_4bit_output = new_mode == 101;
        if pack_4bit_output {
            new_mode = 0;
            mode = Some(new_mode);
            set_4_bit_output_mode(1);
        }
        // Source `newstack.f90:991-1024`: `-reorder` without `-angle` takes the
        // angles from each input file's extended header (`needHeaderAngles`);
        // `extraTilts` is allocated for either angle route, and `saveTilts`
        // records whether those angles are to be written back out.
        // `newstack.f90:988-996`.  These three sit with `PipGetString(
        // 'TiltAngleFile', ...)` and inside the `ifReorderByTilt` block, well
        // after `No input file specified` (`newstack.f90:355`) and after the
        // output-file count is final, so `-tilt f -reorder n` with no input
        // file reports the missing input, not the option clash.
        if !tilt_angle_file.is_empty() && reorder_by_tilt != 0 {
            exit_error(
                "You cannot enter -tilt with angles to insert and -reorder to reorder by angle",
            );
        }
        let mut need_header_angles = false;
        if reorder_by_tilt != 0 {
            if output_names.len() > 1 {
                exit_error("You cannot use -reorder with more than one output file");
            }
            // `newstack.f90:994-995`.
            if blank || two_directions {
                exit_error("You cannot use -reorder with the -blank or -twodir option");
            }
            need_header_angles = angle_file_to_reorder.is_empty();
        }
        let mut save_tilts = false;
        let mut extra_tilts = Vec::<f32>::new();
        if !tilt_angle_file.is_empty() || reorder_by_tilt != 0 {
            // `allocate(extraTilts(listTotal))` (`newstack.f90:1000-1003`).
            extra_tilts = vec![0.0_f32; routes.len()];
        }
        //
        // Now read the angles from either kind of file (`newstack.f90:1006-1024`).
        // A leading period on `-tilt` is relative to the first input basename in
        // the source (e.g. input.mrc with -tilt .tlt reads input.tlt).
        //
        if !tilt_angle_file.is_empty() || (reorder_by_tilt != 0 && !need_header_angles) {
            let temp_name = if !tilt_angle_file.is_empty() {
                if tilt_angle_file.starts_with('.') {
                    let end = input_names[0].rfind('.').unwrap_or(input_names[0].len());
                    tilt_angle_file = format!("{}{}", &input_names[0][..end], tilt_angle_file);
                }
                save_tilts = true;
                tilt_angle_file.clone()
            } else {
                // `saveTilts = abs(ifReorderByTilt) > 1` (`newstack.f90:1016`).
                save_tilts = reorder_by_tilt.abs() > 1;
                angle_file_to_reorder.clone()
            };
            // `call dopen(3, tempName, 'ro', 'f')` (`newstack.f90:1018`).
            drop(dopen(3, &temp_name, "ro", "f"));
            let Ok(contents) = std::fs::read_to_string(&temp_name) else {
                exit_error(
                    "Reading tilt angle file: it must have as many lines as sections being written",
                );
            };
            let mut contents = contents.lines();
            for index in 0..routes.len() {
                let value = contents
                    .next()
                    .and_then(|line| line.split_whitespace().next())
                    .and_then(|word| word.parse::<f32>().ok());
                let Some(value) = value else {
                    exit_error(
                        "Reading tilt angle file: it must have as many lines as sections being written",
                    );
                };
                extra_tilts[index] = value;
            }
        }
        //
        // Reorder the section list and output angles for this file
        // (`newstack.f90:1668-1683`).  With an entered angle file the angles are
        // known before any input file is opened, so the source's per-file swaps
        // are done here; the `needHeaderAngles` route reorders in the output
        // loop below, where that file's own extended header has been read.  The
        // nested swaps (rather than a Rust sort) retain the source's
        // 0.01-degree comparison and tie behavior.
        //
        if reorder_by_tilt != 0 && !need_header_angles {
            let mut start = 0usize;
            for sections in &section_lists {
                let end = start + sections.len();
                for index in start..end.saturating_sub(1) {
                    for other in index + 1..end {
                        if reorder_by_tilt.signum() as f32
                            * (extra_tilts[index] - extra_tilts[other])
                            > 0.01
                        {
                            routes.swap(index, other);
                            extra_tilts.swap(index, other);
                        }
                    }
                }
                start = end;
            }
        }
        let mut header = first_header.unwrap();
        //
        // `newstack.f90:790-797`: `warpScale` is the warp file's pixel size
        // over the first input file's, and each warping's linear part becomes
        // the section's transform with its shifts scaled the same way.  The
        // source has the input files open by this point; this route reaches
        // that state only after the preliminary pass above.
        //
        if if_warping != 0 {
            warp_scale = warp_pixel_size / (header.xlen / header.mx as f32);
            transforms.clear();
            for index in 1..=warp_num_fields {
                let mut xform = [0.0_f32; 6];
                // `getlineartransform` (`warpwrapfort.c:128`) passes
                // `*iz - 1` and a row count of 2.
                if get_linear_transform(index - 1, xform.as_mut_ptr(), 2) != 0 {
                    exit_error("Getting linear transform from warp file");
                }
                xform[4] *= warp_scale;
                xform[5] *= warp_scale;
                transforms.push(xform);
            }
        }
        //
        // find out if float or other density modification
        // (`newstack.f90:858-936`).  The source reads and validates every
        // scaling option here, ahead of the memory limits at
        // `newstack.f90:1027-1038` and far ahead of the `ifMean` pre-scan at
        // `newstack.f90:1345-1367`.  With this block after the pre-scan,
        // `-verbose 1 -float 4` printed ` reallocating array to`, ` MB of
        // physical memory` and a ` scanning for mean/sd` line per section
        // before the error the reference prints on its own.
        //
        let mut contrast_limits = [0.0_f32, 255.0];
        // `newstack.f90:875`: `PipGetTwoFloats`.
        let contrast_entered = pip_get_two_floats(
            c"ContrastBlackWhite".as_ptr(),
            &raw mut contrast_limits[0],
            &raw mut contrast_limits[1],
        ) == 0;
        let mut scale_limits = [0.0_f32; 2];
        // `newstack.f90:876`: `PipGetTwoFloats`.
        let scale_entered = pip_get_two_floats(
            c"ScaleMinAndMax".as_ptr(),
            &raw mut scale_limits[0],
            &raw mut scale_limits[1],
        ) == 0;
        let mut float_densities = 0_i32;
        let float_entered = pip_get_integer(c"FloatDensities".as_ptr(), &mut float_densities) == 0;
        // `newstack.f90:899-900`'s `if (ifFloat < 0)` check is **not** made
        // here: it sits after the mutually-exclusive check at
        // `newstack.f90:896-898` and inside the `ifFloat < 4` arm, so it is
        // made with the rest of that block below.  Testing
        // `!(1..=4).contains(...)` here also rejected the legal `-float 0` and
        // `-float 5`.
        let mut mean_sd = [0.0_f32; 2];
        // `newstack.f90:878-879`: `PipGetTwoFloats`, not `PipGetFloatArray`.
        // The two are not interchangeable -- `PipGetTwoFloats` hands
        // `PipGetFloatArray` a `numToGet` of **2**, so PIP itself reports
        // `2 values expected but only 1 values found in value entry` for a
        // short entry (`parse_params.c:1953-1956`) and silently stops after
        // two for a long one, where a `numToGet` of zero reads the whole line
        // and reports `Too many values for input array`.
        let mean_sd_entered = pip_get_two_floats(
            c"MeanAndStandardDeviation".as_ptr(),
            &raw mut mean_sd[0],
            &raw mut mean_sd[1],
        ) == 0;
        let mut map_limits = [0.0_f32; 2];
        // `newstack.f90:880`: `PipGetTwoFloats`.
        let map_entered = pip_get_two_floats(
            c"MapFromRange".as_ptr(),
            &raw mut map_limits[0],
            &raw mut map_limits[1],
        ) == 0;
        // `newstack.f90:881`.
        let mut mult_add_count = 0_i32;
        pip_number_of_entries(c"MultiplyAndAdd".as_ptr(), &mut mult_add_count);
        // `newstack.f90:882-883`.
        if map_entered && !(contrast_entered || scale_entered) {
            exit_error("You can use -map only with -scale or -contrast");
        }
        // `newstack.f90:884-888`.  One check, as the source writes it: the
        // `-float` half and the `-contrast`/`-scale`/`-multadd` half are the
        // two arms of a single `.or.`, and `ifFloatDen = 1` below is what
        // makes `-meansd` count as a scaling option for the
        // mutually-exclusive test at `newstack.f90:896-898`.
        if mean_sd_entered {
            if (float_entered && float_densities != 2)
                || contrast_entered
                || scale_entered
                || mult_add_count > 0
            {
                exit_error("You cannot use -meansd with any scaling option except -float 2");
            }
            float_densities = 2;
        }
        let mut scale_factors = Vec::<[f32; 2]>::new();
        // `newstack.f90:890-916`.
        if float_densities >= 4 {
            if !scale_entered {
                exit_error("You must enter -scale with -float 4");
            }
            if map_entered {
                exit_error("You cannot use -map with -float 4");
            }
        } else {
            //
            // `newstack.f90:896-898` is **one** check over all four options --
            // `ifContrast + ifScaleMM + ifFloatDen + min(numScaleFacs, 1) > 1`
            // -- and its text ends in ` except with -float 4`.  Splitting it
            // into three pairwise checks dropped that tail from two of them
            // and missed `-float` with `-multadd` altogether, which then ran
            // and produced an output.  `ifFloatDen` is 1 for `-meansd` too
            // (`newstack.f90:887`).
            //
            if i32::from(contrast_entered)
                + i32::from(scale_entered)
                + i32::from(float_entered || mean_sd_entered)
                + mult_add_count.min(1)
                > 1
            {
                exit_error(
                    "The -scale, -contrast, -multadd, and -float options are mutually exclusive except with -float 4",
                );
            }
            // `newstack.f90:899-900`.
            if float_densities < 0 {
                exit_error("You must use -contrast or -scale instead of a negative -float entry");
            }
            //
            // get scale factors, make sure there are right number
            // (`newstack.f90:906-915`)
            //
            if mult_add_count > 0 {
                if mult_add_count != 1 && mult_add_count != input_names.len() as i32 {
                    exit_error("You must enter -multadd either once or once per input file");
                }
                for _ in 0..mult_add_count {
                    let mut factor_and_add = [0.0_f32; 2];
                    // `newstack.f90:911-912`: `ierr = PipGetTwoFloats(...)`,
                    // whose return value the source discards -- PIP has
                    // already reported and exited on a short entry.
                    drop(pip_get_two_floats(
                        c"MultiplyAndAdd".as_ptr(),
                        &raw mut factor_and_add[0],
                        &raw mut factor_and_add[1],
                    ));
                    scale_factors.push(factor_and_add);
                }
            }
        }
        let range_scale_entered = scale_entered || contrast_entered;
        // `newstack.f90:963-967`: the contrast entry becomes the scale range.
        if contrast_entered {
            contrast_limits[1] = contrast_limits[1].max(contrast_limits[0] + 1.0);
            scale_limits[0] =
                -contrast_limits[0] * 255.0 / (contrast_limits[1] - contrast_limits[0]);
            scale_limits[1] = scale_limits[0] + 65025.0 / (contrast_limits[1] - contrast_limits[0]);
        }
        // Source `FixRangeIfNeeded` acquisition and its legality checks
        // (`newstack.f90:918-936`).  Range correction itself is considered
        // only after the source has established whether interpolation is in
        // use; the source explicitly cancels it for a plain copy.
        let mut fix_range = [0.0_f32, 1.0_f32];
        // `newstack.f90:918`: `PipGetTwoFloats`.
        let fix_range_entered = pip_get_two_floats(
            c"FixRangeIfNeeded".as_ptr(),
            &raw mut fix_range[0],
            &raw mut fix_range[1],
        ) == 0;
        let mut output_mode = mode.unwrap_or(header.mode);
        if fix_range_entered {
            if float_densities != 0 || range_scale_entered || !scale_factors.is_empty() {
                exit_error("You cannot enter -fixrange with any scaling options");
            }
            if fix_range[0] != 0.0 && fix_range[0] < 2.0 {
                exit_error("The entry for -fixrange must be at least 2");
            }
            if input_names.len() > 1 {
                exit_error("You cannot use -fixrange with more than one input file");
            }
            if output_mode != header.mode {
                exit_error("You cannot use -fixrange if you enter an output mode");
            }
        }
        //
        // Memory limits (`newstack.f90:1027-1038`).  `limToAlloc` starts at
        // `4 * limSec` and `lenTemp` at `MAXTEMP`.
        //
        let mut lim_entered = 0_i32;
        let mut lim_to_alloc = 4_i64 * lim_sec as i64;
        let mut len_temp = 5_000_000_i64;
        let mut test_limits = [0_i32; 2];
        if pip_get_two_integers(
            c"TestLimits".as_ptr(),
            &raw mut test_limits[0],
            &raw mut test_limits[1],
        ) == 0
        {
            lim_entered = 1;
            lim_to_alloc = test_limits[0] as i64;
            len_temp = test_limits[1] as i64;
        }
        // `processInPlace` (`newstack.f90:274`) and `inPlaceFac`
        // (`newstack.f90:2801`, whose declaration initialiser makes it `SAVE`)
        // are the state `reallocateIfNeeded` carries from one call to the
        // next, along with `limToAlloc` and `lenTemp`.  This translation sizes
        // its buffers per stage instead of carving them out of one flat
        // `array`, so nothing but the `-verbose` reports consumes this
        // bookkeeping and every call to `reallocate_if_needed` below is made
        // under `iVerbose > 0`.
        let mut process_in_place = false;
        let mut in_place_fac = 1.0_f32;
        let mut alloc_lim_to_alloc = 0_usize;
        let mut alloc_len_temp = 0_usize;
        let mut alloc_idim_in_out = 0_usize;
        let physical_memory = crate::imod::libcfshr::b3dutil::b3d_physical_memory();
        let mut memory_limit_mb = 0_i32;
        let memory_limit_entered =
            pip_get_integer(c"MemoryLimit".as_ptr(), &mut memory_limit_mb) == 0;
        if memory_limit_entered {
            lim_entered = 2;
            lim_to_alloc = memory_limit_mb as i64 * 1024 * 256;
        }
        if lim_entered > 0 && (lim_to_alloc < 1000 || len_temp < 1 || len_temp > lim_to_alloc / 2) {
            exit_error("Inappropriate memory limits entered");
        }
        alloc_lim_to_alloc = lim_to_alloc as usize;
        alloc_len_temp = len_temp as usize;
        // `call reallocateArray()` (`newstack.f90:1038`), whose first
        // statement is this print (`newstack.f90:2887`).  The allocation it
        // wraps is deferred here to the working `Vec`, but the report it makes
        // of the limit that was entered is not.
        if lim_entered > 0 && i_verbose > 0 {
            print!(
                " reallocating array to {}  MB\n",
                list_real(lim_to_alloc as f32 / (1024 * 256) as f32)
            );
        }
        // `newstack.f90:1041-1045`.
        if if_warping != 0 && (!idf_file.is_empty() || !mag_grad_file.is_empty()) {
            exit_error("You cannot use distortion corrections with warping transforms");
        }
        if if_warping != 0 && (rotate_angle != 0. || expand_factor != 0.) {
            exit_error("You cannot use -expand or -rotate with warping transforms");
        }
        if expand_factor < 0.0 {
            exit_error("Expand factor must be positive");
        }
        if bin_factor <= 0 {
            exit_error("Binning factor must be a positive number");
        }
        read_reduction = bin_factor as f32;
        //
        // Get filter entry and if there is binning and no separate shrink entry, convert
        // the binning to a shrinkage.  Also allow a negative filter entry to set the default
        // (`newstack.f90:1049-1095`).
        //
        let mut ind_filt_temp = ind_filter;
        let if_filt_set = 1 - pip_get_integer(c"AntialiasFilter".as_ptr(), &raw mut ind_filt_temp);
        if ind_filt_temp < 0 {
            ind_filt_temp = ind_filter;
        }
        let if_shrink = 1 - pip_get_float(c"ShrinkByFactor".as_ptr(), &raw mut shrink_factor);
        if if_filt_set > 0 && if_shrink == 0 && bin_factor > 1 && ind_filt_temp > 0 {
            shrink_factor = bin_factor as f32;
            bin_factor = 1;
            // `newstack.f90:1058`.  This is a Fortran `print *`, so it shares
            // the record stream of every other write in this program unit;
            // routing it through `libc` stdout instead reorders it against
            // them when stdout is a pipe.
            print!(" Doing antialias-filtered image reduction instead of ordinary binning\n");
        }
        ind_filter = 0.max(ind_filt_temp - 1);
        //
        // Handle shrinkage
        //
        if if_shrink > 0 || shrink_factor > 1. {
            // Do shrinkage on input unless there is binning specified, since this will be
            // more memory-efficient by default and it will produce a correct origin by default
            // with no size change
            read_shrunk = bin_factor == 1;
            // `newstack.f90:1070-1077`.
            if bin_factor > 1
                && (!transforms.is_empty()
                    || rotate_angle != 0.0
                    || expand_factor != 0.0
                    || !idf_file.is_empty()
                    || !mag_grad_file.is_empty()
                    || if_warping != 0)
            {
                exit_error(
                    "You cannot use both -shrink and -bin with -xform, -rotate, -expand, -distort, or -gradient",
                );
            }
            if shrink_factor <= 1. {
                exit_error("Factor for -shrink must be greater than 1");
            }
            if if_warping != 0 && (shrink_factor.round() - shrink_factor).abs() > 1.0e-4 {
                exit_error("You cannot use -shrink with warping unless the factor is an integer");
            }
            let mut ierr = 1;
            let start_filter = ind_filter;
            while ierr == 1 {
                ierr = unsafe {
                    crate::imod::libcfshr::zoomdown::select_zoom_filter(
                        ind_filter,
                        // `newstack.f90:1080` divides in real*4 and the
                        // wrapper widens the result; dividing in f64 here
                        // gives a different filter scale by an ulp.
                        (1.0_f32 / shrink_factor) as f64,
                        &raw mut lines_shrink,
                    )
                };
                if ierr == 1 {
                    ind_filter -= 1;
                }
                if ierr > 1 {
                    exit_error("Selecting antialiasing filter");
                }
            }
            if ind_filter < start_filter {
                // `newstack.f90:1085`; list-directed, so the integer is right
                // justified in 11 after a one-blank separator.
                print!(
                    " Using the last antialiasing filter, # {:>11}\n",
                    ind_filter + 1
                );
            }
            if read_shrunk {
                read_reduction = shrink_factor;
                lines_shrink = 0;
            } else {
                // Post-read shrinkage: provide extra buffer of what needs reading in for a chunk
                // and set an expansion factor
                lines_shrink = lines_shrink / 2 + 2;
                expand_factor = 1.0 / shrink_factor;
            }
            // `newstack.f90:1095`.
            if i_verbose > 0 {
                print!(
                    " Shrinking; readShrunk  {}\n",
                    if read_shrunk { "T" } else { "F" }
                );
            }
        }
        //
        // Distortion field (`newstack.f90:1186-1215`)
        //
        // The source runs this between the `-phase` validation
        // (`newstack.f90:1098`) and the unit-transform setup below; this route
        // already had the two in the other order, and they are independent
        // except for which message a doubly-invalid entry produces.
        //
        let mut num_fields = 0_i32;
        if !idf_file.is_empty() {
            if_distort = 1;
            xf_text = ", undistorted".to_owned();
            let mut warp_name = CString::new(idf_file.as_bytes())
                .unwrap_or_default()
                .into_bytes_with_nul();
            let mut list_string = vec![0_i8; 1024];
            let (mut idf_nx, mut idf_ny, mut idf_binning, mut warp_flags) = (0, 0, 0, 0);
            let mut pixel_size = 0.0_f32;
            if crate::imod::libwarp::warputils::read_check_warp_file(
                warp_name.as_mut_ptr().cast(),
                1,
                1,
                &raw mut idf_nx,
                &raw mut idf_ny,
                &raw mut num_fields,
                &raw mut idf_binning,
                &raw mut pixel_size,
                &raw mut warp_flags,
                list_string.as_mut_ptr(),
                list_string.len() as i32,
            ) < 0
            {
                let message = std::ffi::CStr::from_ptr(list_string.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                exit_error(message.trim_end());
            }
            if pip_get_float(c"ImagesAreBinned".as_ptr(), &raw mut binning_of_input) != 0
                && header.nx <= idf_nx * idf_binning / 2
                && header.ny <= idf_ny * idf_binning / 2
            {
                exit_error(
                    "you must specify current binning of images with -imagebinned because they are not larger than half the camera size",
                );
            }
            if binning_of_input <= 0. {
                exit_error("Image binning must be a positive number");
            }
            warp_scale = idf_binning as f32 / binning_of_input;
            //
            // Set up default field numbers to use then process use list if any
            //
            let default_fields = routes.iter().map(|route| route.1).collect::<Vec<_>>();
            let mut fields = get_items_to_use(
                num_fields,
                &default_fields,
                c"UseFields",
                "FIELD",
                false,
                input_names.len() as i32,
                i32::from(numbered_from_one),
            );
            if fields.len() == 1 {
                fields.resize(routes.len(), fields[0]);
            }
            if fields.len() != routes.len() {
                exit_error("Specified # of fields does not match # of sections");
            }
            idf_use = fields;
        }
        //
        // get mag gradient information; multiply pixel size by binning
        // (`newstack.f90:1219-1226`)
        //
        if !mag_grad_file.is_empty() {
            if_mag_grad = 1;
            xf_text = ", undistorted".to_owned();
            read_mag_gradients(
                &mag_grad_file,
                LIM_GRAD_SEC,
                &mut pixel_mag_grad,
                &mut axis_rot,
                &mut tilt_angles,
                &mut dmag_per_micron,
                &mut rot_per_micron,
                &mut num_mag_grad,
            );
            pixel_mag_grad *= read_reduction;
        }
        //
        // Get offsets from center of image relative to warping or distortion
        // field (`newstack.f90:1228-1236`)
        //
        if if_distort > 0 {
            warp_x_offsets = vec![0.0_f32; routes.len().max(1)];
            warp_y_offsets = vec![0.0_f32; routes.len().max(1)];
            let mut subarea_entries = Vec::<Vec<f32>>::new();
            let mut subarea_count = 0_i32;
            pip_number_of_entries(c"SubareaOffsetsXandY".as_ptr(), &raw mut subarea_count);
            for _ in 0..subarea_count {
                let mut pair = [0.0_f32; 2];
                let mut number = 0_i32;
                if pip_get_float_array(
                    c"SubareaOffsetsXandY".as_ptr(),
                    pair.as_mut_ptr(),
                    &raw mut number,
                    2,
                ) != 0
                {
                    exit_error("Getting subarea offset");
                }
                subarea_entries.push(pair[..number.max(0) as usize].to_vec());
            }
            if get_offset_entries(
                &subarea_entries,
                routes.len(),
                &mut warp_x_offsets,
                &mut warp_y_offsets,
            )
            .is_err()
            {
                exit_error("There must be either one subarea offset or an offset for each section");
            }
        }
        //
        // if not transforming and distorting, rotating, or expanding, set up
        // a unit transform (`newstack.f90:1241-1256`).  `ifXform` is `.not.
        // transforms.is_empty()` on this route, and `lineUse(i) = 0` for every
        // section falls out of `getItemsToUse` for a single transform.
        //
        if transforms.is_empty()
            && (if_distort != 0 || if_mag_grad != 0 || rotate_angle != 0.0 || expand_factor != 0.0)
        {
            let mut unit = [0.0_f32; 6];
            xfunit(&mut unit, 1.0);
            transforms.push(unit);
            while rotate_angle > 180.01 || rotate_angle < -180.01 {
                rotate_angle -= 360.0_f32.copysign(rotate_angle);
            }
        }
        //
        // set up rotation and expansion transforms and multiply by transforms
        // (`newstack.f90:1258-1279`).
        //
        if rotate_angle != 0.0 || expand_factor != 0.0 {
            let mut frot = [0.0_f32; 6];
            xfunit(&mut frot, 1.0);
            if rotate_angle != 0.0 {
                // `cosd`/`sind` are the gfortran degree intrinsics, which
                // enter `_gfortran_cosd_r4`/`_gfortran_sind_r4`: the angle is
                // folded onto the nearest quadrant, the remaining degrees are
                // turned into radians through a double multiply that is
                // rounded back to single, and single-precision `cosf`/`sinf`
                // supply the value.  `to_radians().cos()` is a different
                // function: it disagrees with the reference by up to hundreds
                // of ulps and never returns the exact 0 and 1 that a multiple
                // of 90 degrees has to produce.
                let magnitude = rotate_angle.abs();
                let quadrant = (magnitude / 90.0).round_ties_even() as i32;
                let degrees = magnitude - quadrant as f32 * 90.0;
                let radians = (f64::from(degrees) * (std::f64::consts::PI / 180.0)) as f32;
                let (quadrant_cos, quadrant_sin) = match quadrant & 3 {
                    0 => (radians.cos(), radians.sin()),
                    1 => (-radians.sin(), radians.cos()),
                    2 => (-radians.cos(), -radians.sin()),
                    _ => (radians.sin(), -radians.cos()),
                };
                frot[0] = quadrant_cos;
                frot[2] = if rotate_angle < 0.0 {
                    quadrant_sin
                } else {
                    -quadrant_sin
                };
                frot[3] = frot[0];
                frot[1] = -frot[2];
            }
            if expand_factor == 0.0 {
                expand_factor = 1.0;
            }
            let mut fexp = [0.0_f32; 6];
            xfunit(&mut fexp, expand_factor);
            let mut fprod = [0.0_f32; 6];
            xfmult(&frot, &fexp, &mut fprod);
            for transform in &mut transforms {
                let mut product = [0.0_f32; 6];
                xfmult(transform, &fprod, &mut product);
                *transform = product;
            }
        }
        // Source `optimalMax` (`newstack.f90:38, 48`), indexed by mode + 1.
        let optimal_max: [f32; 17] = [
            255., 32767., 255., 32767., 255., 255., 65535., 255., 255., 511., 1023., 2047., 65504.,
            8191., 16383., 32767., 255.,
        ];
        // Source newstack.f90:1320-1515 pre-scans every selected section
        // before any output is opened when floating density policies need
        // section statistics.  Keep its section-indexed state in this program
        // unit; later float policy branches consume these arrays.
        let (mut sec_mean, mut sec_mins, mut sec_maxes, mut sec_sds) = (
            Vec::<f32>::new(),
            Vec::<f32>::new(),
            Vec::<f32>::new(),
            Vec::<f32>::new(),
        );
        // `if ((ifMean .ne. 0 .and. ifMeanSdEntered == 0) .or. fixRangeSDs > 0)`
        // (`newstack.f90:1345`): an entered mean and SD replaces the scan.
        // Source `mode`, the mode of the last header `irdhdr` read
        // (`newstack.f90:1349`).
        let mut scan_mode = header.mode;
        if float_densities > 1 && !mean_sd_entered {
            let mut scan_input = usize::MAX;
            // `newstack.f90:1348-1355`: this scan is also a loop over input
            // files, so each file's header is read and its own binned size is
            // computed before its sections are scanned.
            let (mut scan_nx, mut scan_ny) = (0_i32, 0_i32);
            let (mut scan_bin_nx, mut scan_bin_ny) = (0_i32, 0_i32);
            let (mut scan_rx_offset, mut scan_ry_offset) = (0.0_f32, 0.0_f32);
            for &(input_index, section) in &routes {
                if scan_input != input_index {
                    if scan_input != usize::MAX {
                        iiu_close(1);
                        // `newstack.f90:1428`.
                        if need_close1 > 0 {
                            iiu_close(need_close1);
                        }
                    }
                    // `call openInputFile(iFile)` (`newstack.f90:1348`).
                    open_input_file(
                        input_index + 1,
                        num_vol_read,
                        &list_volumes,
                        &input_names,
                        &mut need_close1,
                    );
                    scan_input = input_index;
                    let scan_file = iiu_get_ii_file(1);
                    if scan_file.is_null() {
                        exit_error("Opening input file");
                    }
                    // `unit_fileio.c:256-265`: the unit's own `MrcHeader`,
                    // which is the only one a non-MRC input has.
                    let scan_header =
                        std::ptr::read(iiu_mrc_header(1, c"iiuRetBasicHead".as_ptr(), 1, 0));
                    scan_nx = scan_header.nx;
                    scan_ny = scan_header.ny;
                    // Source `mode` after `call irdhdr(1, ...)`
                    // (`newstack.f90:1349`), which `newstack.f90:1439` then
                    // tests for the compression note.
                    scan_mode = scan_header.mode;
                    (scan_bin_nx, scan_rx_offset) =
                        get_reduced_size(scan_header.nx, read_reduction, read_shrunk, odd_even_ok);
                    (scan_bin_ny, scan_ry_offset) =
                        get_reduced_size(scan_header.ny, read_reduction, read_shrunk, odd_even_ok);
                    // `call reallocateIfNeeded()` (`newstack.f90:1357`) is
                    // made once per input file, with `nyNeeded = nyBin`
                    // (`newstack.f90:1356`) and `nxOut`/`nyOut` still at their
                    // `newstack.f90:676-677` initialiser of -1.  Only its two
                    // reports are consumed here; the buffer sizes below are
                    // this translation's own.
                    if i_verbose > 0 {
                        let mut allocation = ReallocateIfNeeded {
                            physical_memory,
                            process_in_place,
                            ft_reduce_fac,
                            phase_shift,
                            lim_entered,
                            nx: scan_nx,
                            ny: scan_ny,
                            nx_bin: scan_bin_nx,
                            ny_bin: scan_bin_ny,
                            ny_needed: scan_bin_ny,
                            nx_out: -1,
                            ny_out: -1,
                            read_shrunk,
                            read_reduction,
                            i_binning: bin_factor,
                            fourier_scaling,
                            nx_fspad,
                            ny_fspad,
                            nx_fcrop_pad,
                            ny_fcrop_pad,
                            ft_expand_fac,
                            noise_pad,
                            nx_bin_fft: scan_bin_nx,
                            ny_bin_fft: scan_bin_ny,
                            lim_to_alloc: alloc_lim_to_alloc,
                            len_temp: alloc_len_temp,
                            pre_set_scaling: false,
                            idim_in_out: alloc_idim_in_out,
                            in_place_fac,
                            i_verbose,
                        };
                        reallocate_if_needed(&mut allocation);
                        process_in_place = allocation.process_in_place;
                        in_place_fac = allocation.in_place_fac;
                        alloc_lim_to_alloc = allocation.lim_to_alloc;
                        alloc_len_temp = allocation.len_temp;
                        alloc_idim_in_out = allocation.idim_in_out;
                    }
                }
                //
                // `call reallocateIfNeeded()` (`newstack.f90:1357`) runs
                // before this file's sections are scanned, and `idimInOut` is
                // what `scanSection` divides by `nx` to get `maxLines` -- so
                // it decides how many loads the section is read in, and
                // therefore how the per-load sums round.  With `nxOut` and
                // `nyOut` still zero, `needDim` is just the binned input, so
                // for no entered limit `idimInOut` is that; an entered
                // `-test` pair leaves `limToAlloc - lenTemp` from
                // `newstack.f90:1037` alone; and an entered `-memory` gets
                // `limToAlloc` less the temporary size the reader needs.
                //
                let scan_idim_in_out = if lim_entered == 1 {
                    (lim_to_alloc - len_temp).max(1) as usize
                } else {
                    let mut need_temp = 1_i64;
                    if read_shrunk {
                        let min_chunk_lines = if read_reduction > 32. { 3.0_f32 } else { 10.0 };
                        need_temp = ((scan_nx as f32
                            * (((min_chunk_lines + 6.) * read_reduction).ceil() + 20.))
                            as i64)
                            .max((scan_nx as i64 * scan_ny as i64).min(5_000_000));
                    }
                    if bin_factor > 1 {
                        need_temp = scan_nx as i64 * bin_factor as i64;
                    }
                    if lim_entered == 2 {
                        (lim_to_alloc - need_temp).max(1) as usize
                    } else {
                        scan_bin_nx as usize * scan_bin_ny as usize
                    }
                };
                let mut scan_array = vec![
                    0.0_f32;
                    scan_idim_in_out
                        .min(scan_bin_nx as usize * scan_bin_ny as usize)
                        .max(scan_bin_nx as usize)
                ];
                // Same `needTemp` rule as `reallocateIfNeeded`
                // (`newstack.f90:2823-2831`), which sizes the temporary the
                // reader needs for this file.
                let mut scan_temp = vec![
                    0.0_f32;
                    // `reallocateIfNeeded` (`newstack.f90:2818-2842`) recomputes `lenTemp`
                    // for every limit except an entered `-test` pair, which keeps the size
                    // that was entered.
                    if lim_entered == 1 {
                        len_temp.max(1) as usize
                    } else {
                        // `newstack.f90:2822-2831`: `needTemp` is **one** unless
                        // the reader shrinks or bins; `call reallocateIfNeeded()`
                        // at `newstack.f90:1357` has already set `lenTemp` to
                        // that before this scan runs.
                        let mut need_temp = 1_i64;
                        if read_shrunk {
                            let min_chunk_lines = if read_reduction > 32. { 3.0_f32 } else { 10.0 };
                            need_temp = (scan_nx as i64
                                * ((((min_chunk_lines + 6.) * read_reduction).ceil() as i64) + 20))
                                .max((scan_nx as i64 * scan_ny as i64).min(5_000_000));
                        }
                        if bin_factor > 1 {
                            need_temp = scan_nx as i64 * bin_factor as i64;
                        }
                        need_temp.max(1) as usize
                    }
                ];
                // `newstack.f90:1367`.
                if i_verbose > 0 {
                    print!(" scanning for mean/sd {:>11}\n", section);
                }
                let Ok((dmin, dmax, dmean, sd, _load_start, _load_end)) = scan_section(
                    &mut scan_array,
                    scan_bin_nx,
                    scan_bin_ny,
                    0,
                    read_reduction,
                    scan_rx_offset,
                    scan_ry_offset,
                    float_densities,
                    0.0,
                    |data, lines, x_start, y_start| {
                        read_binned_or_reduced(
                            1,
                            section,
                            data,
                            scan_bin_nx,
                            lines,
                            x_start,
                            y_start,
                            read_reduction,
                            scan_bin_nx,
                            lines,
                            ind_filter,
                            read_shrunk,
                            &mut scan_temp,
                        )
                    },
                ) else {
                    exit_error("Reading image file");
                };
                sec_mean.push(dmean);
                sec_mins.push(dmin);
                sec_maxes.push(dmax);
                sec_sds.push(sd);
            }
            if scan_input != usize::MAX {
                iiu_close(1);
                // `newstack.f90:1428`.
                if need_close1 > 0 {
                    iiu_close(need_close1);
                }
            }
        }
        // Source `shiftMin`, `shiftMean` and `shiftMax` for shift-to-mean
        // floating (`newstack.f90:1434-1438`).
        let (mut shift_min, mut shift_mean, mut shift_max) = (0.0_f32, 0.0_f32, 0.0_f32);
        // Source `floatText = ', mean shift&scaled'` at `newstack.f90:1443`.
        let mut compress_to_fit_range = false;
        if float_densities > 2 && !sec_mean.is_empty() {
            let diff_min_mean = sec_mins
                .iter()
                .zip(&sec_mean)
                .map(|(&minimum, &mean)| minimum - mean)
                .fold(0.0_f32, f32::min);
            let diff_max_mean = sec_maxes
                .iter()
                .zip(&sec_mean)
                .map(|(&maximum, &mean)| maximum - mean)
                .fold(0.0_f32, f32::max);
            let grand_mean = sec_mean.iter().sum::<f32>() / sec_mean.len() as f32;
            shift_min = (grand_mean + diff_min_mean).max(0.0);
            shift_mean = shift_min - diff_min_mean;
            shift_max = shift_mean + diff_max_mean;
            //
            // `newstack.f90:1439-1444`: warn and change the title text when
            // the shifted range will not fit the input mode's range.
            //
            if float_densities == 3
                && scan_mode != 2
                && mode.unwrap_or(header.mode) != 2
                && optimal_max[scan_mode as usize] < shift_max
            {
                //
                // `print *` list-directed real*4 output: gfortran writes a
                // `G17.9E2` field and then the blank separator, so an F-form
                // value is right-justified in 13 and carries five trailing
                // blanks while an E-form value fills 17 and carries one.
                //
                let ratio = optimal_max[scan_mode as usize] / shift_max;
                let scientific = format!("{:.8e}", f64::from(ratio.abs()));
                let (mantissa, power) = scientific.split_once('e').unwrap();
                let exponent = power.parse::<i32>().unwrap() + 1;
                let field = if ratio == 0.0 || (0..=9).contains(&exponent) {
                    let decimals = if ratio == 0.0 { 9 } else { 9 - exponent };
                    let mut text = format!("{:.*}", decimals as usize, ratio);
                    if decimals == 0 {
                        text.push('.');
                    }
                    format!("{text:>13}     ")
                } else {
                    format!(
                        "{:>17} ",
                        format!(
                            "{}{}E{}{:02}",
                            if ratio < 0.0 { "-" } else { "" },
                            mantissa,
                            if exponent - 1 < 0 { '-' } else { '+' },
                            (exponent - 1).abs()
                        )
                    )
                };
                println!(" Densities will be compressed by{field} to fit in range");
                compress_to_fit_range = true;
            }
        }
        // `zmin = 1.e10`, `zmax = -1.e10` (`newstack.f90:1317-1318`) with the
        // per-section outlier flags and `numSecTrunc` kept for
        // `findScaleFactors` and the closing note (`newstack.f90:1453-1467`).
        let mut float2_zmin = 1.0e10_f32;
        let mut float2_zmax = -1.0e10_f32;
        let mut z_min_outlier = Vec::<f32>::new();
        let mut z_max_outlier = Vec::<f32>::new();
        let mut num_sec_trunc = 0_i32;
        if float_densities == 2 && !sec_mean.is_empty() {
            let mut zmins = sec_mins
                .iter()
                .zip(&sec_mean)
                .zip(&sec_sds)
                .map(|((&minimum, &mean), &sd)| if sd > 0. { (minimum - mean) / sd } else { 0. })
                .collect::<Vec<_>>();
            let mut zmaxs = sec_maxes
                .iter()
                .zip(&sec_mean)
                .zip(&sec_sds)
                .map(|((&maximum, &mean), &sd)| if sd > 0. { (maximum - mean) / sd } else { 0. })
                .collect::<Vec<_>>();
            let mut min_outliers = vec![0.0_f32; zmins.len()];
            let mut max_outliers = vec![0.0_f32; zmaxs.len()];
            rs_mad_median_outliers(
                zmins.as_mut_ptr(),
                zmins.len() as i32,
                8.0,
                min_outliers.as_mut_ptr(),
            );
            rs_mad_median_outliers(
                zmaxs.as_mut_ptr(),
                zmaxs.len() as i32,
                8.0,
                max_outliers.as_mut_ptr(),
            );
            for (&z, &outlier) in zmins.iter().zip(&min_outliers) {
                if outlier >= 0.0 {
                    float2_zmin = float2_zmin.min(z);
                } else {
                    num_sec_trunc += 1;
                }
            }
            for (&z, &outlier) in zmaxs.iter().zip(&max_outliers) {
                if outlier <= 0.0 {
                    float2_zmax = float2_zmax.max(z);
                } else {
                    num_sec_trunc += 1;
                }
            }
            z_min_outlier = min_outliers;
            z_max_outlier = max_outliers;
        }
        let transform_lines = if transforms.is_empty() {
            Vec::new()
        } else {
            let default_lines = routes.iter().map(|route| route.1).collect::<Vec<_>>();
            let mut lines = get_items_to_use(
                transforms.len() as i32,
                &default_lines,
                c"UseTransformLines",
                "TRANSFORM LINE",
                one_per_file,
                input_names.len() as i32,
                i32::from(numbered_from_one),
            );
            // `newstack.f90:812-828`: with one transform per file, expand the
            // per-file list into one line for each section of that file.
            if one_per_file {
                if lines.len() < input_names.len() {
                    exit_error("Not enough transforms specified for the input files");
                }
                let per_file = lines.clone();
                lines.clear();
                for (file_index, sections) in section_lists.iter().enumerate() {
                    for _ in 0..sections.len() {
                        lines.push(per_file[file_index]);
                    }
                }
            }
            if lines.len() == 1 {
                lines.resize(routes.len(), lines[0]);
            }
            if lines.len() != routes.len() {
                exit_error("Specified # of transform lines does not match # of sections");
            }
            lines
        };
        let mut offset_entries = Vec::<Vec<f32>>::new();
        let mut offset_count = 0_i32;
        pip_number_of_entries(c"OffsetsInXandY".as_ptr(), &mut offset_count);
        for _ in 0..offset_count {
            let mut values = vec![0.0_f32; 2 * routes.len()];
            let mut number = 0_i32;
            if pip_get_float_array(
                c"OffsetsInXandY".as_ptr(),
                values.as_mut_ptr(),
                &mut number,
                values.len() as i32,
            ) != 0
            {
                exit_error("Getting offset entries");
            }
            values.truncate(number as usize);
            offset_entries.push(values);
        }
        let mut x_offsets = vec![0.0_f32; routes.len()];
        let mut y_offsets = vec![0.0_f32; routes.len()];
        if get_offset_entries(
            &offset_entries,
            routes.len(),
            &mut x_offsets,
            &mut y_offsets,
        )
        .is_err()
        {
            exit_error("There must be either one offset or an offset for each section");
        }
        let mut apply_first = 0_i32;
        pip_get_boolean(c"ApplyOffsetsFirst".as_ptr(), &mut apply_first);
        //
        // Check validity of phase shifting now that all requested actions are
        // processed (`newstack.f90:1098-1150`).
        //
        if phase_shift || fourier_scaling {
            if !sizes_match {
                exit_error(
                    "All input files must have the same size in x and y with -phase or -ftreduce",
                );
            }
            if fourier_scaling
                && (rotate_angle != 0.
                    || expand_factor != 0.
                    || !idf_file.is_empty()
                    || !mag_grad_file.is_empty()
                    || if_warping != 0
                    || read_reduction > 1.)
            {
                exit_error(
                    "You cannot use -ftreduce with -rotate, -expand, -distort, -gradient, -shrink, -bin OR warping",
                );
            }
            if phase_shift
                && (rotate_angle != 0.
                    || expand_factor != 0.
                    || !idf_file.is_empty()
                    || !mag_grad_file.is_empty()
                    || if_warping != 0)
            {
                exit_error(
                    "You cannot use -phase with -rotate, -expand, -distort, -gradient, or warping, or with -shrink unless it is the only other operation",
                );
            }
            if apply_first != 0 && ft_expand_fac < 1. {
                exit_error("You cannot use -applyfirst with -phase or -ftreduce");
            }
            //
            // Check transforms and convert to xcen/ycen shifts
            //
            if !transforms.is_empty() {
                let tolerance = 0.01 / header.nx.max(header.ny) as f32;
                for (index, &line) in transform_lines.iter().enumerate() {
                    let transform = transforms[line as usize];
                    if (transform[0] - 1.).abs() > tolerance
                        || (transform[3] - 1.).abs() > tolerance
                        || transform[1].abs() > tolerance
                        || transform[2].abs() > tolerance
                    {
                        exit_error(
                            "With -phase or -reduce, transforms must contain only shifts (first four terms must be 1 0 0 1)",
                        );
                    }
                    x_offsets[index] -= transform[4];
                    y_offsets[index] -= transform[5];
                }
                transforms.clear();
            }
            //
            // Set up for fourier cropping
            //
            if fourier_scaling {
                let mut crop_factor = ft_reduce_fac;
                if ft_expand_fac > 0. {
                    crop_factor = 1. / ft_expand_fac;
                }
                if fourier_crop_sizes(
                    header.nx,
                    crop_factor,
                    0.01,
                    16,
                    nice_fft_limit(),
                    &raw mut nx_fspad,
                    &raw mut nx_fcrop_pad,
                    &raw mut actual_fac,
                ) > 0
                {
                    exit_error(
                        "Reduction or expansion factor must be an integer or integer divided by 2, 3, 4, 5, 6, 8, or 10 to 3 decimal places",
                    );
                }
                fourier_crop_sizes(
                    header.ny,
                    crop_factor,
                    0.01,
                    16,
                    nice_fft_limit(),
                    &raw mut ny_fspad,
                    &raw mut ny_fcrop_pad,
                    &raw mut actual_fac,
                );
                // `newstack.f90:1140`.
                if i_verbose > 0 {
                    print!(" Actual factor {}\n", list_real(actual_fac));
                }
                //
                // When cropping, reduce the shifts here for use when extracting
                // image area; the fractional part will be boosted back up for
                // the shift in the crop
                //
                if ft_reduce_fac > 1. || apply_first != 0 {
                    for index in 0..routes.len() {
                        x_offsets[index] /= actual_fac;
                        y_offsets[index] /= actual_fac;
                    }
                }
            }
        }
        //
        // Section replacement (`newstack.f90:1152-1184`)
        //
        let mut list_replace = Vec::<i32>::new();
        let mut replace_nxyz = [0_i32; 3];
        let (mut replace_dmin, mut replace_dmax, mut replace_dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
        let mut replace_mode_old = 0_i32;
        let mut replace_list_string: *mut std::ffi::c_char = std::ptr::null_mut();
        if pip_get_string(c"ReplaceSections".as_ptr(), &raw mut replace_list_string) == 0 {
            let list = std::ffi::CStr::from_ptr(replace_list_string)
                .to_string_lossy()
                .into_owned();
            let mut parsed = vec![0_i32; routes.len().max(1)];
            let (mut num_replace, mut limit) = (0_i32, parsed.len() as i32);
            if parselist2(&list, &mut parsed, &mut num_replace, &mut limit).is_err() {
                exit_error("Reading list of sections to replace");
            }
            parsed.truncate(num_replace.max(0) as usize);
            if num_replace > 0 {
                if output_names.len() > 1 {
                    exit_error("There must be only one output file to use -replace");
                }
                // `newstack.f90:1160`.  `-3d` and `-chunk` are refused by name
                // above, but a chunked HDF input still turns `if3dVolumes` on
                // at `newstack.f90:456`, so this exit is reachable.
                if if3d_volumes > 0 {
                    exit_error("You cannot use -3d or -chunk with -replace");
                }
                if !tilt_angle_file.is_empty() || reorder_by_tilt != 0 {
                    exit_error("You cannot use -tilt or -reorder with -replace");
                }
                if section_list_entries > 0 {
                    if section_lists[0].len() as i32 != num_replace {
                        exit_error(
                            "You must specify the same number of input sections as ones to replace",
                        );
                    }
                } else {
                    if (section_lists[0].len() as i32) < num_replace {
                        exit_error(
                            "There are not as many sections in the input file as ones to replace",
                        );
                    }
                    // `nlist(1) = numReplace` (`newstack.f90:1169`) truncates
                    // the first file's list, and `routes` is that list.
                    section_lists[0].truncate(num_replace as usize);
                    let kept = routes
                        .iter()
                        .filter(|(file, _)| *file != 0)
                        .copied()
                        .collect::<Vec<_>>();
                    routes.retain(|(file, _)| *file == 0);
                    routes.truncate(num_replace as usize);
                    routes.extend(kept);
                    num_output_sections = vec![routes.len() as i32];
                }
                if !quiet {
                    ialprt(true);
                }
                ii_allow_multi_volume(0);
                let Ok(output_name) = CString::new(output_names[0].as_bytes()) else {
                    exit_error("Invalid output file name");
                };
                imopen(2, output_name.to_str().unwrap_or_default(), "OLD");
                let mut mxyz2 = [0_i32; 3];
                let mut mode_old = 0_i32;
                irdhdr(
                    2,
                    replace_nxyz.as_mut_ptr(),
                    mxyz2.as_mut_ptr(),
                    &raw mut mode_old,
                    &raw mut replace_dmin,
                    &raw mut replace_dmax,
                    &raw mut replace_dmean,
                );
                ialprt(false);
                // `call iiuFileInfo(2, ix1, ix2, iiuFlags)` then
                // `pack4bitOutput = btest(iiuFlags, 5) .or. btest(iiuFlags, 6)`
                // (`newstack.f90:1176-1177`): replacing sections in an
                // existing 4-bit file packs the output regardless of `-mode`.
                let (mut ix1, mut ix2, mut iiu_flags) = (0_i32, 0_i32, 0_i32);
                iiu_file_info(2, &raw mut ix1, &raw mut ix2, &raw mut iiu_flags);
                pack_4bit_output = iiu_flags & (IIUNIT_4BIT_MODE | IIUNIT_HALF_XSIZE) != 0;
                replace_mode_old = mode_old;
                for section in &mut parsed {
                    *section -= i32::from(numbered_from_one);
                    if *section < 0 || *section >= replace_nxyz[2] {
                        exit_error("Replacement section number out of range");
                    }
                }
                list_replace = parsed;
            }
        }
        // `newstack.f90:1281-1282`.
        if fourier_scaling {
            expand_factor = 1. / actual_fac;
        }
        if expand_factor == 0.0 {
            expand_factor = 1.0;
        }
        //
        // adjust xcen, ycen and transforms if binning (`newstack.f90:1290-1300`)
        //
        // The source divides by `readReduction`, which is the binning factor
        // *or* the shrink factor on the read-shrunk route -- not by `iBinning`
        // alone -- and it does it once, here, for every offset and every
        // transform.
        //
        if read_reduction > 1. {
            for index in 0..routes.len() {
                x_offsets[index] /= read_reduction;
                y_offsets[index] /= read_reduction;
            }
            for transform in &mut transforms {
                transform[4] /= read_reduction;
                transform[5] /= read_reduction;
            }
        }
        // Source `SizeToOutputInXandY` defaulting (`newstack.f90:676-690,
        // 1693-1702`): absent or nonpositive axes retain the input extent.
        let size_x = size_to_output.map(|size| size[0]).filter(|size| *size > 0);
        let size_y = size_to_output.map(|size| size[1]).filter(|size| *size > 0);
        let factor = expand_factor;
        let transpose = (rotate_angle.abs() - 90.0).abs() < 40.0;
        // `newstack.f90:1570-1571` uses `getReducedSize`, whose X and Y
        // offsets are the starting coordinates handed to
        // `readBinnedOrReduced`; dropping them reads from the wrong corner
        // whenever the factor does not divide the size.
        // The source calls these inside its input-file loop
        // (`newstack.f90:1570-1571`), so they are refreshed for each input
        // file below; only the output size comes from the first one.
        let (mut bin_nx, mut rx_offset) =
            get_reduced_size(header.nx, read_reduction, read_shrunk, odd_even_ok);
        let (mut bin_ny, mut ry_offset) =
            get_reduced_size(header.ny, read_reduction, read_shrunk, odd_even_ok);
        // `newstack.f90:1694-1702`: the output size comes from the *reduced*
        // input size times `expandFactor`, which is 1 unless `-expand`,
        // `-shrink` on the post-read route or Fourier scaling changed it.
        let (output_nx, output_ny) = if size_x.is_none() && size_y.is_none() && transpose {
            (
                (bin_ny as f32 * factor).round() as i32,
                (bin_nx as f32 * factor).round() as i32,
            )
        } else {
            (
                size_x.unwrap_or_else(|| (bin_nx as f32 * factor).round() as i32),
                size_y.unwrap_or_else(|| (bin_ny as f32 * factor).round() as i32),
            )
        };
        if print_size_and_exit {
            println!(" Output size: {output_nx:12}{output_ny:12}");
            return;
        }
        //
        // if warping or distortions, figure out how big to allocate the arrays
        // (`newstack.f90:1707-1745`).  Only the `isec == 1` pass of the
        // source's section loop runs this, so it happens once here.
        //
        let (mut field_dx, mut field_dy) = (Vec::<f32>::new(), Vec::<f32>::new());
        let (mut tmp_dx, mut tmp_dy) = (Vec::<f32>::new(), Vec::<f32>::new());
        let mut n_control = Vec::<i32>::new();
        if if_distort != 0 || if_warping != 0 {
            n_control = vec![0_i32; num_fields.max(warp_num_fields).max(1) as usize];
            //
            // Expanded grid size is based on the input size for distortion and
            // the output size for warping (`newstack.f90:1715-1734`).
            //
            let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
            let (mut xn_big, mut yn_big) = (0.0_f32, 0.0_f32);
            if if_distort != 0 {
                xn_big = nx_max as f32 / warp_scale;
                yn_big = ny_max as f32 / warp_scale;
            } else {
                xn_big = read_reduction * output_nx as f32 / warp_scale;
                yn_big = read_reduction * output_ny as f32 / warp_scale;
                if apply_first == 0 {
                    dx = 1.0e20;
                    dy = 1.0e20;
                    xn_big = -dx;
                    // `newstack.f90:1727` sets `ynbig` from `-dx` too; the
                    // source writes `xnBig = -dx` twice and never gives
                    // `ynbig` its own initialiser, so it keeps the value the
                    // `else` branch above put there.
                    for index in 0..routes.len() {
                        xn_big = xn_big.max(
                            (output_nx as f32 + x_offsets[index]) * read_reduction / warp_scale,
                        );
                        dx = dx.min(x_offsets[index] * read_reduction / warp_scale);
                        yn_big = yn_big.max(
                            (output_ny as f32 + y_offsets[index]) * read_reduction / warp_scale,
                        );
                        dy = dy.min(y_offsets[index] * read_reduction / warp_scale);
                    }
                }
            }
            let (mut max_nx_grid, mut max_ny_grid) = (0_i32, 0_i32);
            let mut list_string = vec![0_i8; 1024];
            if find_max_grid_size(
                dx,
                xn_big,
                dy,
                yn_big,
                n_control.as_mut_ptr(),
                &raw mut max_nx_grid,
                &raw mut max_ny_grid,
                list_string.as_mut_ptr(),
                list_string.len() as i32,
            ) != 0
            {
                let message = std::ffi::CStr::from_ptr(list_string.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                exit_error(message.trim_end());
            }
            lm_grid = lm_grid.max(max_nx_grid).max(max_ny_grid);
        }
        if if_distort != 0 || if_mag_grad != 0 || if_warping != 0 {
            let size = lm_grid as usize * lm_grid as usize;
            field_dx = vec![0.0_f32; size];
            field_dy = vec![0.0_f32; size];
            tmp_dx = vec![0.0_f32; size];
            tmp_dy = vec![0.0_f32; size];
        }
        // `newstack.f90:1748-1751`.
        if num_taper == 1 {
            num_taper = 127.min(16.max(((output_nx + output_ny) as f32 / 400.).round() as i32));
            println!("\nTapering will be done over{num_taper:4} pixels");
        }
        // The remaining `FixRangeIfNeeded` execution block from
        // newstack.f90:1284-1510.  The source cancels it unless an operation
        // interpolates pixels; this direct route currently has affine
        // transforms as its interpolation operation.
        if fix_range_entered && fix_range[0] > 0.0 && !transforms.is_empty() {
            let mut range_params = [10.0_f32, 1.2_f32];
            // `newstack.f90:935`: `ierr = PipGetTwoFloats(...)`, return value
            // discarded.
            drop(pip_get_two_floats(
                c"RangeFixingParams".as_ptr(),
                &raw mut range_params[0],
                &raw mut range_params[1],
            ));
            let (mut range_fix_ok, mut scale_fix_alone_ok, mut scale_fix_shifted_ok) = (
                (header.mode == 1 && header.amean < 0.0)
                    || (header.mode == 6 && header.amean < 16000.0),
                fix_range[1] > 1.0 && header.amean >= 0.0,
                fix_range[1] > 1.0 && header.amean < 0.0,
            );
            let mut scale_range_fix_ok = fix_range[1] > 1.0;
            let mut need_range_fix = false;
            let mut need_scale_fix = false;
            let scan_stride = (routes.len() / routes.len().min(9)).max(1);
            let mut scan_array = vec![0.0_f32; header.nx as usize * header.ny as usize];
            let mut scan_temp = vec![0.0_f32; header.nx.max(1) as usize];
            let mut scan_input = usize::MAX;
            for (route, &(input_index, section)) in routes.iter().enumerate() {
                if route % scan_stride != 0 {
                    continue;
                }
                if scan_input != input_index {
                    if scan_input != usize::MAX {
                        iiu_close(1);
                        // `newstack.f90:1428`.
                        if need_close1 > 0 {
                            iiu_close(need_close1);
                        }
                    }
                    // `call openInputFile(iFile)` (`newstack.f90:1348`).
                    open_input_file(
                        input_index + 1,
                        num_vol_read,
                        &list_volumes,
                        &input_names,
                        &mut need_close1,
                    );
                    scan_input = input_index;
                    // `call reallocateIfNeeded()` (`newstack.f90:1357`) is
                    // made once per input file, with `nyNeeded = nyBin`
                    // (`newstack.f90:1356`) and `nxOut`/`nyOut` still at their
                    // `newstack.f90:676-677` initialiser of -1.  Only its two
                    // reports are consumed here; the buffer sizes below are
                    // this translation's own.
                    if i_verbose > 0 {
                        let mut allocation = ReallocateIfNeeded {
                            physical_memory,
                            process_in_place,
                            ft_reduce_fac,
                            phase_shift,
                            lim_entered,
                            nx: header.nx,
                            ny: header.ny,
                            nx_bin: bin_nx,
                            ny_bin: bin_ny,
                            ny_needed: bin_ny,
                            nx_out: -1,
                            ny_out: -1,
                            read_shrunk,
                            read_reduction,
                            i_binning: bin_factor,
                            fourier_scaling,
                            nx_fspad,
                            ny_fspad,
                            nx_fcrop_pad,
                            ny_fcrop_pad,
                            ft_expand_fac,
                            noise_pad,
                            nx_bin_fft: bin_nx,
                            ny_bin_fft: bin_ny,
                            lim_to_alloc: alloc_lim_to_alloc,
                            len_temp: alloc_len_temp,
                            pre_set_scaling: false,
                            idim_in_out: alloc_idim_in_out,
                            in_place_fac,
                            i_verbose,
                        };
                        reallocate_if_needed(&mut allocation);
                        process_in_place = allocation.process_in_place;
                        in_place_fac = allocation.in_place_fac;
                        alloc_lim_to_alloc = allocation.lim_to_alloc;
                        alloc_len_temp = allocation.len_temp;
                        alloc_idim_in_out = allocation.idim_in_out;
                    }
                }
                // `newstack.f90:1367`.
                if i_verbose > 0 {
                    print!(" scanning for mean/sd {:>11}\n", section);
                }
                let Ok((minimum, maximum, mean, sd, _, _)) = scan_section(
                    &mut scan_array,
                    header.nx,
                    header.ny,
                    0,
                    1.0,
                    0.0,
                    0.0,
                    0,
                    fix_range[0],
                    |data, lines, x_start, y_start| {
                        read_binned_or_reduced(
                            1,
                            section,
                            data,
                            header.nx,
                            lines,
                            x_start,
                            y_start,
                            1.0,
                            header.nx,
                            lines,
                            0,
                            false,
                            &mut scan_temp,
                        )
                    },
                ) else {
                    exit_error("Reading image file");
                };
                let range_add = (range_params[1] - 1.0) * (maximum - minimum);
                let limited_minimum = (mean - fix_range[0] * sd).max(minimum - range_add);
                let limited_maximum = (mean + fix_range[0] * sd).min(maximum + range_add);
                if sd < range_params[0] {
                    need_scale_fix = true;
                }
                let range_base = if header.mode == 1 { 32768.0 } else { 0.0 };
                if limited_minimum < -range_base {
                    need_range_fix = true;
                }
                if limited_maximum + range_base > 32767.0 {
                    range_fix_ok = false;
                }
                if (limited_maximum + range_base) * fix_range[1] > 32767.0 {
                    scale_range_fix_ok = false;
                }
                if mean < 0.0 {
                    if (limited_maximum + range_base) * fix_range[1] > 32767.0 {
                        scale_fix_shifted_ok = false;
                    }
                } else if limited_maximum * fix_range[1] > 65535.0 - range_base {
                    scale_fix_alone_ok = false;
                }
                // `newstack.f90:1395-1397`.
                if i_verbose > 0 {
                    print!(
                        " {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
                        list_real(minimum),
                        list_real(maximum),
                        list_real(mean),
                        list_real(sd),
                        list_real(limited_minimum),
                        list_real(limited_maximum),
                        list_real(range_base),
                        if need_scale_fix { "T" } else { "F" },
                        if need_range_fix { "T" } else { "F" },
                        if range_fix_ok { "T" } else { "F" },
                        if scale_range_fix_ok { "T" } else { "F" },
                        if scale_fix_alone_ok { "T" } else { "F" },
                        if scale_fix_shifted_ok { "T" } else { "F" }
                    );
                }
            }
            if scan_input != usize::MAX {
                iiu_close(1);
                // `newstack.f90:1428`.
                if need_close1 > 0 {
                    iiu_close(need_close1);
                }
            }
            let do_range_scale =
                need_range_fix && need_scale_fix && range_fix_ok && scale_range_fix_ok;
            let do_range_only = !do_range_scale && need_range_fix && range_fix_ok;
            let do_scale_only =
                !do_range_scale && need_scale_fix && (scale_fix_alone_ok || scale_fix_shifted_ok);
            if do_range_scale || do_range_only || do_scale_only {
                let mut factor = 1.0_f32;
                let mut constant = 0.0_f32;
                if do_range_scale || do_scale_only {
                    factor = fix_range[1];
                    println!(
                        "INFO: Newstack scaling values by {:6.1} to preserve intensity resolution",
                        factor
                    );
                    // `newstack.f90:1486-1488`: the literal ends at "below"
                    // with no separating blank, and the format's trailing "/"
                    // writes one more empty record.
                    println!("  because SD of values is below{:6.1}\n", range_params[0]);
                }
                if do_range_scale || do_range_only {
                    if output_mode == 6 {
                        output_mode = 1;
                        println!(
                            "INFO: Newstack changing mode to 1 (signed integer) to avoid truncation"
                        );
                    } else {
                        constant = 32768.0 * factor;
                        println!("INFO: Newstack shifting values up by 32768 to avoid truncation");
                        println!(
                            "  If taking logarithm in Tilt program, make the offset 0 instead 32768."
                        );
                    }
                } else if output_mode == 1 && scale_fix_shifted_ok {
                    constant = 32768.0 * (factor - 1.0);
                }
                scale_factors.push([factor, constant]);
            }
        }
        // Source `ifFloat` (`newstack.f90:868, 901-902, 1490`): the -float
        // entry, driven negative by -contrast, -scale, -multadd, or a range fix.
        let mut if_float = float_densities;
        if if_float < 4 {
            if contrast_entered {
                if_float = -2;
            }
            if scale_entered || !scale_factors.is_empty() {
                if_float = -1;
            }
        }
        // `ifMean` and `fracZero` (`newstack.f90:867, 866, 950`).
        let if_mean = i32::from(if_float > 1);
        let frac_zero = 0.0_f32;
        //
        // determine whether rescaling will be needed (`newstack.f90:1987-2021`)
        //
        // for no float: if either mode = 2, no rescale;
        // otherwise rescale from input range to output range only if
        // mode is changing
        //
        // `packed4bitInput = btest(iiuFlags, 5) .or. btest(iiuFlags, 6)`
        // (`newstack.f90:1530-1531`), taken from this input file's unit.
        let mut packed_4bit_input = header.iiu_flags & (IIUNIT_4BIT_MODE | IIUNIT_HALF_XSIZE) != 0;
        let mut rescale = false;
        if if_float == 0 && output_mode != 2 && header.mode != 2 {
            // `rescale = mode .ne. newMode .or. (pack4bitOutput .and.
            // .not.packed4bitInput)` (`newstack.f90:1989`).
            rescale = header.mode != output_mode || (pack_4bit_output && !packed_4bit_input);
        } else if if_float != 0 {
            rescale = true;
        }
        let pre_set_scaling = if_float <= 0;
        //
        // `optimalMax(mode + 1)` is Fortran's 1-based index for `mode`.
        let mut optimal_in = optimal_max[header.mode as usize];
        let mut optimal_out = optimal_max[output_mode as usize];
        // `newstack.f90:1997-1998`: packed 4-bit data has a range of 0-15 at
        // whichever end it is packed.
        if packed_4bit_input {
            optimal_in = 15.;
        }
        if pack_4bit_output {
            optimal_out = 15.;
        }
        //
        // set bottom of input range to 0 unless mode 1 or 2 and already
        // negative or not rescaling; set bottom of output range to 0 unless
        // not changing modes
        //
        let mut bottom_in = 0.0_f32;
        if header.amin < 0. || !rescale {
            if header.mode == 1 {
                bottom_in = -optimal_in - 1.;
            }
            if header.mode == 2 {
                bottom_in = -optimal_in;
            }
        }
        let mut bottom_out = 0.0_f32;
        if header.mode == output_mode {
            bottom_out = bottom_in;
        }
        // `floatInterpRangeFac` (`newstack.f90:283`).
        let float_interp_range_fac = 0.13_f32;
        let mut opt_float_range = 0.0_f32;
        let mut opt_float_min = 0.0_f32;
        let mut float_z_margin = 0.0_f32;
        if if_float == 2 {
            opt_float_range = optimal_out;
            opt_float_min = 0.;
            float_z_margin = 0.;
            // `if ((ifXform .ne. 0 .or. phaseShift .or. fourierScaling) .and.
            // (newMode == 1 .or. newMode == 6))` (`newstack.f90:2015-2016`).
            if (!transforms.is_empty() || phase_shift || fourier_scaling)
                && (output_mode == 1 || output_mode == 6)
            {
                opt_float_min += float_interp_range_fac * opt_float_range;
                opt_float_range = (1. - 2. * float_interp_range_fac) * opt_float_range;
                float_z_margin = float_interp_range_fac * (float2_zmax - float2_zmin);
            }
        }
        // Source `dmeanSec` for filling outside the input image
        // (`newstack.f90:2305-2312`).  The chunked-affine route below cannot
        // take the `needEdgeMean` branch (that one needs `numChunks == 1`), so
        // it keeps the entered fill value or the input file mean; the
        // single-chunk route computes its own `dmeanSec` per section.
        let mut dmean_sec = fill_value.unwrap_or(header.amean);
        let mut array = vec![0.0_f32; header.nx as usize * header.ny as usize];
        // `newstack.f90:1517`: the preliminary pass above ran with unit
        // printing off (`newstack.f90:302`); the processing loop turns it back
        // on so `imopen` and `irdhdr` report each file that is used.
        if !quiet {
            ialprt(true);
        }
        // Source `time`, `b3dDate`, and format 302 at newstack.f90:1518,
        // 1891-1898.  This direct stream path has no truncation title text.
        let mut title = [b' '; 80];
        title[..23].copy_from_slice(b"NEWSTACK: Images copied");
        // `newstack.f90:1891-1898` writes `xfText` into the `a13` field.
        if !xf_text.is_empty() {
            let bytes = xf_text.as_bytes();
            title[23..23 + bytes.len().min(13)].copy_from_slice(&bytes[..bytes.len().min(13)]);
        }
        // `floatText` is keyed on **`ifFloat`**, not on the `-float` entry.
        // `newstack.f90:951-952` gives `', densities scaled'` to every negative
        // `ifFloat`, and `newstack.f90:901-902` drives it negative for
        // `-contrast` (-2) and for `-scale` *or* `-multadd` (-1) -- so a
        // `-multadd` output carries the text too.  Keying it on the entered
        // range-scaling options alone left a `-multadd` title 18 characters
        // short, and put `', densities scaled'` on a `-float 4 -scale` output
        // whose `ifFloat` stays 4 (`newstack.f90:892-894` skips the negative
        // assignments for `ifFloat >= 4`).
        if if_float < 0 {
            title[36..54].copy_from_slice(b", densities scaled");
        } else if if_float > 0 {
            // `newstack.f90:1309-1331`.  The `a18` field clips the
            // 19-character texts.
            title[36..54].copy_from_slice(b", floated to range");
            if if_mean != 0 {
                if if_float == 2 {
                    title[36..54].copy_from_slice(b", floated to means");
                } else if if_float == 3 {
                    // `newstack.f90:1443` replaces the text when the shifted
                    // range had to be compressed.
                    if compress_to_fit_range {
                        title[36..54].copy_from_slice(b", mean shift&scale");
                    } else {
                        title[36..54].copy_from_slice(b",  shifted to mean");
                    }
                } else {
                    title[36..54].copy_from_slice(b", mean shift&scale");
                }
            }
        }
        let mut date = [b' '; 9];
        b3d_date(&mut date);
        title[56..65].copy_from_slice(&date);
        let mut now = 0_i64;
        let mut local = std::mem::zeroed::<libc::tm>();
        libc::time(&raw mut now);
        libc::localtime_r(&raw const now, &raw mut local);
        let mut time = [0_i8; 9];
        libc::strftime(
            time.as_mut_ptr(),
            time.len(),
            c"%H:%M:%S".as_ptr(),
            &raw const local,
        );
        // Source `timeStr` (`call time(timeStr)`, `newstack.f90:1518`), read
        // again for the scratch file's extension at `newstack.f90:2287-2290`.
        let time_str: [u8; 8] =
            unsafe { std::slice::from_raw_parts(time.as_ptr().cast::<u8>(), 8) }
                .try_into()
                .unwrap();
        title[67..75].copy_from_slice(&time_str);
        // Source `ifTempOpen` (`newstack.f90:1523`).
        let mut if_temp_open = 0_i32;
        let mut route_index = 0usize;
        let mut num_trunc_low = 0_i32;
        let mut num_trunc_high = 0_i32;
        // Source `ifHeaderOut` (`newstack.f90:1522`) is set once for the whole
        // run, so the column heading is printed before the first section only.
        let mut if_header_out = 0_i32;
        let mut active_input_index = usize::MAX;
        // `newstack.f90:1524, 109`: the input and output autodoc indices and
        // the FrameSet flag, all 1-based as the Fortran wrappers return them.
        let mut ind_adoc_in = 0_i32;
        let mut ind_adoc_out = 0_i32;
        let mut frame_set = false;
        let mut out_doc_changed = false;
        let mut active_input_file: *mut crate::imod::libiimod::iimage::ImodImageFile =
            std::ptr::null_mut();
        //
        // Extended-header state carried across the source's input-file loop
        // (`newstack.f90:23,45,90-91,111,118,228-229,267,272`).  `extraIn`
        // starts at `maxExtraIn = 4` bytes (`newstack.f90:228,373`) and
        // `extraOut` unallocated.
        //
        let mut max_extra_in = 4_i32;
        let mut extra_in = vec![0_u8; max_extra_in as usize];
        let mut max_extra_out = 0_i32;
        let mut extra_out = Vec::<u8>::new();
        let mut n_byte_sym_in = 0_i32;
        let mut num_int_or_bytes_in = 0_i32;
        let mut i_flag_extra_in = 0_i32;
        let mut n_byte_extra_out = 0_i32;
        let mut n_byte_sym_out = 0_i32;
        let mut ind_extra_out = 0_i32;
        let mut serial_em_type = false;
        let mut fei1_type = false;
        let mut iany_extra_type = 0_i32;
        let mut ifirst_extra_type = 0_i32;
        let mut max_in_file_angles = 0_i32;
        let mut all_file_tilts = Vec::<f32>::new();
        let mut iz_not_piece = Vec::<i32>::new();
        for (output_index, name) in output_names.iter().enumerate() {
            let Ok(name) = CString::new(name.as_bytes()) else {
                exit_error("Invalid output file name");
            };
            // `newstack.f90:1525-1527`: each input file is opened with
            // `openInputFile` and reported with `irdhdr` at the top of the
            // source's input-file loop, before the output file that its first
            // section goes into is created.  This stream path is output-file
            // major, so the file holding the first section of this output file
            // is opened here and the loop below only re-opens on a change.
            if let Some(&(first_input_index, _)) = routes.get(route_index)
                && active_input_index != first_input_index
            {
                if !active_input_file.is_null() {
                    iiu_close(1);
                    // `newstack.f90:2762`.
                    if need_close1 > 0 {
                        iiu_close(need_close1);
                    }
                    active_input_file = std::ptr::null_mut();
                }
                // `call openInputFile(iFile)` (`newstack.f90:1526`).
                open_input_file(
                    first_input_index + 1,
                    num_vol_read,
                    &list_volumes,
                    &input_names,
                    &mut need_close1,
                );
                let (mut nxyz, mut mxyz, mut mode, mut dmin_in, mut dmax_in, mut dmean_in) =
                    ([0_i32; 3], [0_i32; 3], 0, 0., 0., 0.);
                irdhdr(
                    1,
                    nxyz.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    &mut mode,
                    &mut dmin_in,
                    &mut dmax_in,
                    &mut dmean_in,
                );
                active_input_file = iiu_get_ii_file(1);
                if !active_input_file.is_null() {
                    active_input_index = first_input_index;
                    // The source's `nx`, `ny`, `nz`, `mode`, `dminIn` and
                    // `dmeanIn` are refreshed by that `irdhdr`, so everything
                    // below -- the output header it transfers, the rescaling
                    // decision and the section reads -- uses this file's
                    // header and not the first input file's.
                    // `unit_fileio.c:256-265`: the unit's own `MrcHeader`.
                    header = std::ptr::read(iiu_mrc_header(1, c"iiuRetBasicHead".as_ptr(), 1, 0));
                    // `newstack.f90:1570-1571`: the binned size to read is
                    // this file's, so inputs of different sizes each read
                    // their own extent into the common output size.
                    (bin_nx, rx_offset) =
                        get_reduced_size(header.nx, read_reduction, read_shrunk, odd_even_ok);
                    (bin_ny, ry_offset) =
                        get_reduced_size(header.ny, read_reduction, read_shrunk, odd_even_ok);
                    // `newstack.f90:1572-1573`.
                    if i_verbose > 0 {
                        print!(
                            " Size and offsets X: {:>11} {} , Y: {:>11} {}\n",
                            bin_nx,
                            list_real(rx_offset),
                            bin_ny,
                            list_real(ry_offset)
                        );
                    }
                    //
                    // get extra header information if any (`newstack.f90:1575-1683`).  The
                    // autodoc block below is the source's `newstack.f90:1535-1565,1663-1665`;
                    // nothing here touches an autodoc, so the two are independent.
                    //
                    iiu_ret_num_extended(1, &raw mut n_byte_sym_in);
                    let mut itype = 0_i32;
                    num_int_or_bytes_in = 0;
                    i_flag_extra_in = 0;
                    fei1_type = false;
                    if need_header_angles && n_byte_sym_in == 0 {
                        exit_error(
                            "There is no extended header; tilt angles for -reorder must be entered with the -angles option",
                        );
                    }
                    if n_byte_sym_in > 0 {
                        //
                        // Deallocate array if it was allocated and is not big enough
                        if max_extra_in > 0 && n_byte_sym_in > max_extra_in {
                            extra_in = Vec::new();
                            max_extra_in = 0;
                        }
                        //
                        // Allocate array if needed
                        if max_extra_in == 0 {
                            max_extra_in = n_byte_sym_in + 1024;
                            extra_in = vec![0_u8; max_extra_in as usize];
                        }
                        iiu_ret_extended_data(
                            1,
                            &raw mut n_byte_sym_in,
                            extra_in.as_mut_ptr().cast(),
                        );
                        iiu_ret_extended_type(
                            1,
                            &raw mut num_int_or_bytes_in,
                            &raw mut i_flag_extra_in,
                        );
                        //
                        // DNM 4/18/02: if these numbers do not represent bytes and
                        // flags, then number of bytes is 4 times nint + nreal
                        //
                        serial_em_type = nbytes_and_flags(num_int_or_bytes_in, i_flag_extra_in);
                        itype = 1;
                        if serial_em_type {
                            itype = -1;
                        }
                        //
                        // Mark new FEI type as 2 and no type for unknown
                        if num_int_or_bytes_in < 0 {
                            if num_int_or_bytes_in == -3 {
                                itype = 2;
                                fei1_type = true;
                            } else {
                                itype = 0;
                                n_byte_sym_in = 0;
                            }
                        }
                        if serial_em_type && save_tilts && i_flag_extra_in % 2 == 0 {
                            exit_error(
                                "You cannot save tilt angles into a SerialEM extended header that was not saved with tilt angles",
                            );
                        }
                        //
                        // Get tilt angles for reordering
                        if need_header_angles {
                            //
                            // Take care of tilt angle array
                            if header.nz > max_in_file_angles {
                                max_in_file_angles = header.nz + 8;
                                all_file_tilts = vec![0.0_f32; max_in_file_angles as usize];
                                iz_not_piece = vec![0_i32; max_in_file_angles as usize];
                            }
                            //
                            // Set up dummy piece lists and get the tilt angles
                            for ind in 0..header.nz as usize {
                                iz_not_piece[ind] = ind as i32;
                            }
                            let mut num_tilts = 0_i32;
                            let mut nz_in = header.nz;
                            get_extra_header_tilts_fortran(
                                extra_in.as_mut_ptr().cast(),
                                &raw mut n_byte_sym_in,
                                &raw mut num_int_or_bytes_in,
                                &raw mut i_flag_extra_in,
                                &raw mut nz_in,
                                all_file_tilts.as_mut_ptr(),
                                &raw mut num_tilts,
                                &raw mut max_in_file_angles,
                                iz_not_piece.as_mut_ptr(),
                            );
                            if num_tilts < header.nz {
                                exit_error(
                                    "There are either not enough or no tilt angles in extended header; tilt angles for -reorder must be entered with the -angles option",
                                );
                            }
                            //
                            // Fill the angle arrays from the section list
                            let mut start = 0usize;
                            for sections in &section_lists[..first_input_index] {
                                start += sections.len();
                            }
                            for ind in start..start + section_lists[first_input_index].len() {
                                extra_tilts[ind] = all_file_tilts[routes[ind].1 as usize];
                            }
                        }
                    }
                    if iany_extra_type == 0 {
                        iany_extra_type = itype;
                    }
                    if first_input_index == 0 {
                        ifirst_extra_type = itype;
                    }
                    if iany_extra_type != 0
                        && itype != 0
                        && iany_extra_type != itype
                        && !strip_extra
                    {
                        exit_error(
                            "You cannot include files with different types of extended headers in the same run; add the -strip option",
                        );
                    }
                    if save_tilts && itype != ifirst_extra_type {
                        exit_error(
                            "To store tilt angles, all input files must have the same type of extended header or no extended header",
                        );
                    }
                    if save_tilts && itype == 2 {
                        exit_error(
                            "You cannot store tilt angles back into a new FEI1-style extended header",
                        );
                    }
                    //
                    // Reorder the section list and output angles for this file
                    // (`newstack.f90:1667-1683`).  The entered-angle-file route was reordered
                    // before any input was opened; this is the `needHeaderAngles` route, whose
                    // angles only exist once the file's extended header has been read.
                    //
                    if reorder_by_tilt != 0 && need_header_angles {
                        let mut start = 0usize;
                        for sections in &section_lists[..first_input_index] {
                            start += sections.len();
                        }
                        let end = start + section_lists[first_input_index].len();
                        for ind in start..end.saturating_sub(1) {
                            for ilist in ind + 1..end {
                                if reorder_by_tilt.signum() as f32
                                    * (extra_tilts[ind] - extra_tilts[ilist])
                                    > 0.01
                                {
                                    routes.swap(ind, ilist);
                                    extra_tilts.swap(ind, ilist);
                                }
                            }
                        }
                    }
                    // `newstack.f90:1535-1565, 1663-1665`: the source obtains this
                    // file's autodoc index at the top of its input-file loop.  A
                    // single-image file or a `-remove` name substitution opens the
                    // metadata file by name; otherwise the file's own index is
                    // used, created when `-mdoc` or `-pixel` asked for it.
                    ind_adoc_in = 0;
                    if (*active_input_file).file != 5
                        && use_mdoc_files
                        && (header.nz == 1 || !remove_from_name.is_empty())
                    {
                        let mut list_string = input_names[first_input_index].clone();
                        if !remove_from_name.is_empty() {
                            let base = list_string
                                .strip_suffix(remove_from_name.as_str())
                                .filter(|base| !base.is_empty());
                            let Some(base) = base else {
                                exit_error(&format!(
                                    "{list_string} does not end with {remove_from_name}"
                                ));
                            };
                            list_string = format!("{base}{add_to_name}");
                        }
                        // `ierr = 1`, or 2 for a single-image file, which is what
                        // lets `AdocOpenImageMetadata` keep a FrameSet autodoc.
                        let add_mdoc = if header.nz == 1 { 2 } else { 1 };
                        let Ok(list_c) = CString::new(list_string.as_bytes()) else {
                            exit_error("Invalid input file name");
                        };
                        let (mut montage, mut num_sect, mut sect_type) = (0_i32, 0_i32, 0_i32);
                        let opened = adoc_open_image_metadata(
                            list_c.as_ptr(),
                            add_mdoc,
                            &raw mut montage,
                            &raw mut num_sect,
                            &raw mut sect_type,
                        );
                        ind_adoc_in = if opened >= 0 { opened + 1 } else { opened };
                        if ind_adoc_in <= 0 {
                            exit_error(&format!("No mdoc file found with name {list_string}.mdoc"));
                        }
                        frame_set = sect_type == 4;
                    }
                    let open_mdoc_or_new = i32::from(use_mdoc_files || pixel_from_mdoc);
                    if ind_adoc_in <= 0 {
                        let index = iiu_ret_adoc_index(1, 0, open_mdoc_or_new);
                        ind_adoc_in = if index >= 0 { index + 1 } else { index };
                    }
                }
            }
            //
            // `reallocateIfNeeded` (`newstack.f90:2818-2837`): `lenTemp` is
            // recomputed from what the reader actually needs unless the
            // `-test` pair set it, and `idimInOut` is `limToAlloc - lenTemp`.
            // Keeping the `MAXTEMP` default here instead would reserve five
            // million elements that nothing uses and shrink the working array
            // by that much, which moves the chunk boundary for `-memory`.
            //
            let effective_len_temp = if lim_entered != 1 {
                let mut need_temp = 1_i64;
                if read_shrunk {
                    let min_chunk_lines = if read_reduction > 32. { 3.0_f32 } else { 10.0 };
                    need_temp = (header.nx as i64
                        * ((((min_chunk_lines + 6.) * read_reduction).ceil() as i64) + 20))
                        .max(MAX_TEMP.min(header.nx as i64 * header.ny as i64));
                }
                if bin_factor > 1 {
                    need_temp = header.nx as i64 * bin_factor as i64;
                }
                if fourier_scaling && nx_fspad > 0 {
                    need_temp = need_temp.max((nx_fcrop_pad as i64 + 2) * ny_fcrop_pad as i64);
                }
                if (phase_shift || fourier_scaling) && nx_fspad > 0 && noise_pad {
                    need_temp = need_temp.max(
                        2 * i64::from(bin_nx.max(bin_ny))
                            + i64::from(nx_fspad - bin_nx)
                            + i64::from(ny_fspad - bin_ny),
                    );
                }
                need_temp
            } else {
                // `newstack.f90:2836-2838`'s companion exit is **not** raised
                // here.  This is not one of the source's `reallocateIfNeeded`
                // call sites -- those are `newstack.f90:1357`, inside the
                // conditional pre-scan, and `newstack.f90:2181`, inside the
                // section loop after the output file has been created -- so
                // raising it here reports
                // `Too small a temporary array size entered for Fourier
                // reduction` before the `NEW image file on unit 2` banner and
                // leaves no output file where the reference leaves one.
                // `reallocate_if_needed` carries the exit at the real call
                // site.
                len_temp
            };
            //
            // `newstack.f90:1290-1303` sets `idimInOut` before any of this:
            // `limToAlloc - lenTemp` only when there is a read reduction, the
            // Fourier crop size when that applies, and otherwise
            // **`limToAlloc - 1`**.  `reallocateIfNeeded` then overrides it for
            // an entered `-memory` (and computes it outright when no limit was
            // entered).  Using `limToAlloc - lenTemp` everywhere makes the
            // working array `lenTemp - 1` elements too small, which picks a
            // different chunk count: at `-test 2500,25` the source fits three
            // chunks in 2499 elements where 2475 forces four.
            //
            let chunk_limit = if lim_entered == 2 {
                (lim_to_alloc - effective_len_temp).max(0) as usize
            } else if read_reduction > 1. {
                (lim_to_alloc - len_temp).max(0) as usize
            } else if fourier_scaling && nx_fspad > 0 {
                (lim_to_alloc - (nx_fcrop_pad as i64 + 2) * ny_fcrop_pad as i64).max(0) as usize
            } else {
                (lim_to_alloc - 1).max(0) as usize
            };
            // `newstack.f90:1290-1305` is the last unconditional setting of
            // `idimInOut`; `reallocateIfNeeded` overrides it per section for
            // an entered `-memory` and for no entered limit, and leaves it
            // alone for an entered `-test` pair.
            alloc_idim_in_out = chunk_limit;
            //
            // The source's own chunk count (`newstack.f90:2189-2198`), which
            // is what its multi-chunk validations test.  It is **not** the
            // condition above: the source chunks when the input *fits* in
            // `idimInOut` but input plus output does not, and leaves
            // `numChunks` at zero when the input does not fit -- in which case
            // the search at `newstack.f90:2245-2270` takes over, *after* those
            // validations have already passed.  `processInPlace` is false here
            // because this route always has a transform
            // (`newstack.f90:2172`), and `nxDimNeed`/`nyDimNeed` are the
            // binned width and the lines the whole output needs.
            //
            let source_num_chunks = if route_index < routes.len() && !transforms.is_empty() {
                let mut fprod = transforms
                    .get(transform_lines.get(route_index).copied().unwrap_or(0) as usize)
                    .copied()
                    .unwrap_or([1., 0., 0., 1., 0., 0.]);
                if apply_first != 0 {
                    let frot = [
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        -x_offsets[route_index],
                        -y_offsets[route_index],
                    ];
                    let mut composed = [0.0_f32; 6];
                    xfmult(&frot, &fprod, &mut composed);
                    fprod = composed;
                } else {
                    fprod[4] -= x_offsets[route_index];
                    fprod[5] -= y_offsets[route_index];
                }
                let needed = lines_needed_for_output(
                    &LinesNeededForOutput {
                        fourier_scaling,
                        if_xform: 1,
                        nx_bin: bin_nx,
                        ny_bin: bin_ny,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        xcen: x_offsets[route_index],
                        ycen: y_offsets[route_index],
                        // `linesNeededForOutput` hands `fprod` on to
                        // `backXform` as its `amat(2,2)`, so the pairs here
                        // are Fortran `fprod(i, j)`: the flat transform is in
                        // column-major storage, which puts `a21` at index 1
                        // and `a12` at index 2.
                        fprod: [
                            [fprod[0], fprod[2], fprod[4]],
                            [fprod[1], fprod[3], fprod[5]],
                        ],
                        max_field_x: 0,
                        max_field_y: 0,
                        lines_shrink,
                    },
                    0,
                    output_ny - 1,
                );
                let nx_dim_need = bin_nx.max(1) as usize;
                let ny_dim_need = (needed.iy_in_2 - needed.iy_in_1 + 1).max(1) as usize;
                if chunk_limit / nx_dim_need > ny_dim_need {
                    let lines_left =
                        (chunk_limit - nx_dim_need * ny_dim_need) / output_nx.max(1) as usize;
                    if lines_left > 0 {
                        (output_ny as usize).div_ceil(lines_left)
                    } else {
                        0
                    }
                } else {
                    0
                }
            } else {
                0
            };
            // The output is written in pieces exactly when the source writes
            // it in pieces.  `numChunks == 0` means the input did not fit, and
            // the source's search then settles on a chunking whose pixels are
            // the same as the whole-section route's -- and which still tapers,
            // which this route's chunk loop cannot -- so that case takes the
            // whole-section route here.
            // This route writes the output in pieces whenever the input and
            // output together do not fit in the limit.  That is not the
            // source's condition -- see `source_num_chunks` above -- and where
            // the two disagree the exits below say so.
            let chunked_affine = lim_entered > 0
                && !transforms.is_empty()
                && chunk_limit > 0
                && 2 * header.nx as usize * header.ny as usize > chunk_limit;
            let out_file = if !list_replace.is_empty() {
                // `newstack.f90:1763`: with `-replace` no output file is
                // created here.  The one opened `OLD` during option processing
                // is still on unit 2, and its header is left alone until the
                // final `iiuWriteHeader(2, title, -1, ...)`.
                let out_file = iiu_get_ii_file(2);
                if out_file.is_null() {
                    exit_error("Opening output file");
                }
                // `newstack.f90:1959-1963`.
                if ((output_mode + 1) / 2 == 2 || (header.mode + 1) / 2 == 2)
                    && ((header.mode + 1) / 2 != 2 || (output_mode + 1) / 2 != 2)
                {
                    exit_error("All input files must be complex if any are");
                }
                // `newstack.f90:1754-1758`.
                if output_nx != replace_nxyz[0] || output_ny != replace_nxyz[1] {
                    exit_error("Existing output file does not have right size in X or Y");
                }
                if output_mode != replace_mode_old {
                    exit_error("Output mode does not match existing output file");
                }
                out_file
            } else if chunked_affine {
                if iiu_open_print(2, name.to_str().unwrap_or_default(), "NEW") != 0 {
                    exit_error("Opening output file");
                }
                //
                // These sit between the open and the header write because the
                // source reaches them inside its section loop: the
                // `NEW image file` banner is printed, but nothing is written.
                //
                // `newstack.f90:2200-2205`: a multi-chunk section cannot be
                // tapered, and no Fourier operation can run on one.  Both test
                // the source's own `numChunks`, computed above -- which is
                // zero, not one, when the input does not fit, so neither exit
                // fires in that case.
                if source_num_chunks > 1 && num_taper > 0 {
                    exit_error("Cannot taper output image - it does not fit completely in memory");
                }
                if source_num_chunks > 1 && (phase_shift || fourier_scaling) {
                    exit_error(
                        "Cannot apply Fourier operations - input and output images do not fit completely in memory",
                    );
                }
                // Below the limit that trips the exit above, the source
                // does not taper correctly -- it *corrupts memory*.  Its
                // `numChunks` was zero when that exit was tested, so the check
                // passes; the layout search at `newstack.f90:2245-2270` then
                // splits the output anyway, and `newstack.f90:2504` calls
                // `taperAtFill(array(iChunkBase), nxOut, nyOut, ...)` on a
                // buffer holding only `numLinesOut(iChunk)` lines.  The
                // reference reads and writes past it: at `-test 3000,30` it
                // aborts with a glibc heap error (rc 134) after writing an
                // output, and at `-test 2500,25` it survives with whatever the
                // adjacent array held.  There is nothing here to be faithful
                // to, so this is a refusal by name.
                if num_taper > 0 {
                    exit_error("-taper with -memory or -test is not supported by this translation");
                }
                // `newstack.f90:2591-2611`: with `preSetScaling` false the
                // source writes each chunk to temporary storage -- to the
                // scratch file opened below when `ifOutChunk` is positive,
                // otherwise to its own place in the array -- finds the scale
                // factors from the whole section, and rescales in a second,
                // backwards pass.  Both are translated in the chunk loop.
                let mut nxyz = [output_nx, output_ny, num_output_sections[output_index]];
                let mut mxyz = nxyz;
                iiu_create_header(
                    2,
                    nxyz.as_mut_ptr(),
                    mxyz.as_mut_ptr(),
                    output_mode,
                    std::ptr::null_mut(),
                    0,
                );
                let chunk_file = iiu_get_ii_file(2);
                if chunk_file.is_null() {
                    exit_error("Creating output header");
                }
                //
                //  Set HDF compression (`newstack.f90:1794-1796`).
                //
                if i_hdf_compression >= 0 && iiu_file_type(2) == IIFILE_HDF {
                    iiu_set_hdf_compression(2, i_hdf_compression);
                }
                //
                // Set up 3D volume (`newstack.f90:1798-1812`).
                //
                if if3d_volumes > 0 {
                    nx_tile_in = nx_tile;
                    ny_tile_in = ny_tile;
                    nz_chunk_in = nz_chunk;
                    let mut ierr = 0_i32;
                    ii_best_tile_size(output_nx, &mut nx_tile_in, &mut ierr, 1);
                    ii_best_tile_size(output_ny, &mut ny_tile_in, &mut ierr, 1);
                    ii_best_tile_size(
                        num_output_sections[output_index],
                        &mut nz_chunk_in,
                        &mut ierr,
                        1,
                    );
                    if iiu_alt_chunk_sizes(2, nx_tile_in, ny_tile_in, nz_chunk_in) != 0 {
                        exit_error("Setting chunk sizes in new volume");
                    }
                    if nx_tile_in != output_nx
                        || ny_tile_in != output_ny
                        || nz_chunk_in != num_output_sections[output_index]
                    {
                        // `write(*,'(a,i7,a,i7,a,i4)')` (`newstack.f90:1809`).
                        println!(
                            "Actual chunk size: {nx_tile_in:7} by{ny_tile_in:7} by{nz_chunk_in:4}"
                        );
                    }
                }
                // `unit_fileio.c:256-269`: the output unit's own `MrcHeader`,
                // which for a non-MRC output is not `ImodImageFile.header`.
                let chunk_header =
                    iiu_mrc_header(2, c"iiuTransHeader".as_ptr(), iiu_get_exit_on_error(), 2);
                // `iiuTransHeader` (`unit_header.c:381-385`) saves and restores
                // the destination `fp` around the whole-header copy, so the
                // output header keeps its own stream and not the input's.
                let fp_save = (*chunk_header).fp;
                std::ptr::copy_nonoverlapping(&header, chunk_header, 1);
                (*chunk_header).fp = fp_save;
                // `iiuTransHeader` runs `mrcInitOutputHeader`
                // (`unit_header.c:387`) over the copied header, never
                // `mrc_head_new`.  The two are not interchangeable:
                // `mrc_head_new` (`mrcfiles.c:689-766`) additionally resets
                // `amin` to `FLT_MAX`, `amax` to `-FLT_MAX`, `amean`, `ispg`,
                // `creatid`, `sub`, `zfac`, `min2`-`max3`, `idtype`, `lens`,
                // `nd1`-`vd2`, `blank`, `tiltangles` and `iiuFlags`, every one
                // of which the source's whole-header copy keeps from the
                // input.  Size and mode are what `iiuAltMode`
                // (`unit_header.c:245-263`) and `iiuAltSize`
                // (`unit_header.c:307-317`) set next
                // (`newstack.f90:1813-1821`).
                mrc_init_output_header(&mut *chunk_header);
                // `iiuTransHeader` ends in `iiuTransExtendedData`
                // (`unit_header.c:395`), after `mrcInitOutputHeader` has
                // cleared `next`, `nint`, `nreal`, `nversion` and `extType`
                // (`mrcfiles.c:797-801`).  That call restores `nint`, `nreal`
                // and a valid `extType` from the input and writes the input's
                // extended data to the output; the extra-header sizing below
                // then overrides `next` and `headerSize`.  The source reaches
                // `newstack.f90:1813` on every route, chunked or not, so this
                // call belongs here as much as on the whole-section route.
                iiu_trans_extended_data(2, 1);
                (*chunk_header).nx = output_nx;
                (*chunk_header).ny = output_ny;
                (*chunk_header).nz = num_output_sections[output_index];
                (*chunk_header).mode = output_mode;
                (*chunk_header).nxstart = header.nxstart;
                (*chunk_header).nystart = header.nystart;
                (*chunk_header).nzstart = header.nzstart;
                (*chunk_header).mapc = header.mapc;
                (*chunk_header).mapr = header.mapr;
                (*chunk_header).maps = header.maps;
                // `newstack.f90:1839-1840`: `iiuAltCell` writes `cell2(4:6)`,
                // which the source sets to 90 for every output file.
                (*chunk_header).alpha = 90.0;
                (*chunk_header).beta = 90.0;
                (*chunk_header).gamma = 90.0;
                (*chunk_header).xorg = header.xorg;
                (*chunk_header).yorg = header.yorg;
                (*chunk_header).zorg = header.zorg;
                if header.mx == header.nx && header.my == header.ny && header.mz == header.nz {
                    (*chunk_header).mx = output_nx;
                    (*chunk_header).my = output_ny;
                    (*chunk_header).mz = num_output_sections[output_index];
                } else {
                    (*chunk_header).mx = header.mx;
                    (*chunk_header).my = header.my;
                    (*chunk_header).mz = header.mz;
                }
                // `newstack.f90:1838-1843`: the X and Y cell sizes carry the
                // read reduction and expansion, and the Z size is then reset
                // without them.
                (*chunk_header).xlen =
                    (*chunk_header).mx as f32 * (header.xlen / header.mx as f32) * read_reduction
                        / expand_factor;
                (*chunk_header).ylen =
                    (*chunk_header).my as f32 * (header.ylen / header.my as f32) * read_reduction
                        / expand_factor;
                (*chunk_header).zlen = (*chunk_header).mz as f32 * (header.zlen / header.mz as f32);
                if pixel_from_mdoc {
                    let (input_index, input_section) = routes[route_index];
                    if let Some(spacing) = mdoc_pixel_spacing[input_index][input_section as usize] {
                        // `newstack.f90:1848-1849`: the mdoc spacing is
                        // scaled by `readReduction` just as the header
                        // spacing above it is, so `-pixel` with `-bin` or
                        // `-shrink` keeps the reduced pixel size.
                        (*chunk_header).xlen =
                            (*chunk_header).mx as f32 * spacing * read_reduction / expand_factor;
                        (*chunk_header).ylen =
                            (*chunk_header).my as f32 * spacing * read_reduction / expand_factor;
                        (*chunk_header).zlen = (*chunk_header).mz as f32 * spacing;
                    }
                }
                //
                // adjust extra header information if currently open file has it
                // (`newstack.f90:1904-1935`)
                //
                n_byte_sym_out = 0;
                // `newstack.f90:1906`.
                out_doc_changed = false;
                if (n_byte_sym_in > 0 || save_tilts)
                    && !strip_extra
                    && (*chunk_file).file == IIFILE_MRC
                {
                    if n_byte_sym_in == 0 {
                        n_byte_extra_out = 4;
                        serial_em_type = false;
                        // `iiuAltExtendedType(2, 0, 1)` (`unit_header.c:1099`)
                        // sets `nint` and `nreal` on unit 2's header.
                        (*chunk_header).nint = 0;
                        (*chunk_header).nreal = 1;
                    } else {
                        get_extra_header_max_sec_size(
                            extra_in.as_mut_ptr().cast(),
                            n_byte_sym_in,
                            num_int_or_bytes_in,
                            i_flag_extra_in,
                            header.nz,
                            &raw mut n_byte_extra_out,
                        );
                    }
                    n_byte_sym_out = num_output_sections[output_index] * n_byte_extra_out;
                    if max_extra_out > 0 && n_byte_sym_out > max_extra_out {
                        extra_out = Vec::new();
                        max_extra_out = 0;
                    }
                    //
                    // Allocate array if needed
                    if max_extra_out == 0 {
                        max_extra_out = n_byte_sym_out + 1024;
                        extra_out = vec![0_u8; max_extra_out as usize];
                    }
                    // `iiuAltNumExtended(2, nByteSymOut)` (`unit_header.c:1078`).
                    (*chunk_header).next = n_byte_sym_out;
                    (*chunk_header).header_size = 1024 + n_byte_sym_out;
                    iiu_set_position(2, 0, 0);
                    ind_extra_out = 0;
                } else {
                    // `iiuAltNumExtended(2, 0)`.  `iiuTransExtendedData` has
                    // already copied `nint`, `nreal` and `extType` from the
                    // input, so with `-strip` those survive with no data.
                    (*chunk_header).next = 0;
                    (*chunk_header).header_size = 1024;
                }
                // `newstack.f90:1855-1889`.  The origin shifts by the
                // fractional-pixel offset whenever the read reduced the image,
                // and `-origin` then re-centres with the *reduced* size and a
                // delta scaled by the reduction over the expansion.
                {
                    let mut delta = [
                        header.xlen / header.mx as f32,
                        header.ylen / header.my as f32,
                        header.zlen / header.mz as f32,
                    ];
                    let mut x_origin = (*chunk_header).xorg;
                    let mut y_origin = (*chunk_header).yorg;
                    let mut z_origin = (*chunk_header).zorg;
                    if read_reduction > 1. {
                        x_origin -= delta[0] * rx_offset;
                        y_origin -= delta[1] * ry_offset;
                    }
                    if adjust_origin {
                        let first_section = routes[route_index].1;
                        let x_center = x_offsets[route_index];
                        let y_center = y_offsets[route_index];
                        z_origin -= first_section as f32 * delta[2];
                        delta[0] = delta[0] * read_reduction / expand_factor;
                        delta[1] = delta[1] * read_reduction / expand_factor;
                        if transforms.is_empty() {
                            x_origin -= (bin_nx / 2 + x_center.round() as i32 - output_nx / 2)
                                as f32
                                * delta[0];
                            y_origin -= (bin_ny / 2 + y_center.round() as i32 - output_ny / 2)
                                as f32
                                * delta[1];
                        } else if apply_first != 0 {
                            x_origin -= (expand_factor * (bin_nx as f32 / 2. + x_center)
                                - output_nx as f32 / 2.)
                                * delta[0];
                            y_origin -= (expand_factor * (bin_ny as f32 / 2. + y_center)
                                - output_ny as f32 / 2.)
                                * delta[1];
                        } else {
                            x_origin -= (expand_factor * bin_nx as f32 / 2. + x_center
                                - output_nx as f32 / 2.)
                                * delta[0];
                            y_origin -= (expand_factor * bin_ny as f32 / 2. + y_center
                                - output_ny as f32 / 2.)
                                * delta[1];
                        }
                    }
                    if adjust_origin || read_reduction > 1. {
                        (*chunk_header).xorg = x_origin;
                        (*chunk_header).yorg = y_origin;
                        (*chunk_header).zorg = z_origin;
                    }
                }
                // `iiuTransHeader` retains the source titles, as the
                // whole-header copy above does; the final
                // `iiuWriteHeader(..., 1, ...)` at the end of the section loop
                // appends this run's title.
                (*chunk_header).labels = header.labels;
                (*chunk_header).nlabl = header.nlabl;
                ii_sync_from_mrc_header(chunk_file, chunk_header);
                std::ptr::null_mut()
            } else {
                // `newstack.f90:1766-1788`: with `-3d 2` or `-3d 3` the output
                // is a new volume inside an existing multi-volume HDF file, so
                // the file is opened `OLD` on unit 12 and unit 2 becomes the
                // new volume in it.
                if if3d_volumes > 1 {
                    ii_allow_multi_volume(1);
                    imopen(12, name.to_str().unwrap_or_default(), "OLD");
                    if iiu_volume_open(2, 12, -1) != 0 {
                        exit_error("Opening new volume in existing file");
                    }
                    need_close2 = 12;
                    ii_allow_multi_volume(0);
                    if if3d_volumes == 3 {
                        // `newstack.f90:1774-1787`.  The Fortran wrapper
                        // `iiuretadocindex` (`unit_fileio.c:401-405`) returns
                        // the index numbered from 1, and `adocsetcurrent`
                        // (`adoc_fwrap.c:159-162`) takes it back off, so the
                        // C entry points here see the 0-based index; likewise
                        // `adocsetinteger` (`adoc_fwrap.c:267-277`) turns the
                        // source's section index 1 into 0.
                        let mut ind_global_adoc = iiu_ret_adoc_index(12, 1, 0);
                        if ind_global_adoc >= 0 {
                            ind_global_adoc += 1;
                        }
                        if ind_global_adoc >= 0 {
                            if adoc_set_current(ind_global_adoc - 1) != 0 {
                                ind_global_adoc = -1;
                            } else if adoc_set_integer(
                                ADOC_GLOBAL_NAME.as_ptr(),
                                0,
                                c"image_pyramid".as_ptr(),
                                1,
                            ) != 0
                            {
                                ind_global_adoc = -1;
                            } else if iiu_write_global_adoc(12) != 0 {
                                ind_global_adoc = -1;
                            }
                        }
                        if ind_global_adoc < 0 {
                            exit_error(
                                "Setting image_pyramid attribute in global section of HDF file",
                            );
                        }
                    }
                } else {
                    // `newstack.f90:1790`: the output file is created through
                    // `imopen(2, ..., 'NEW')`, which is `iiuOpenPrint`
                    // (`wrap_iiunit.f90:11-17`) -- not `iiuOpen`.  The unit
                    // report *and* the file-type line ("This is an HDF file.",
                    // `wrap_iiunit.f90:67-69`) are both printed by that tail,
                    // so calling `iiuOpen` directly loses them for every
                    // non-MRC output.
                    if iiu_open_print(2, name.to_str().unwrap_or_default(), "NEW") != 0 {
                        exit_error("Opening output file");
                    }
                    need_close2 = 0;
                }
                let out_file = iiu_get_ii_file(2);
                if out_file.is_null() {
                    exit_error("Opening output file");
                }
                //
                //  Set HDF compression (`newstack.f90:1794-1796`).
                //
                if i_hdf_compression >= 0 && iiu_file_type(2) == IIFILE_HDF {
                    iiu_set_hdf_compression(2, i_hdf_compression);
                }
                //
                // Set up 3D volume (`newstack.f90:1798-1812`).  The chunk sizes
                // have to reach the file before `hdfWriteHeader` runs
                // `initNewHDFfile`, which is what decides between one 3-D
                // dataset and a stack of 2-D ones
                // (`hdf_imageio.c:510-526`).
                //
                if if3d_volumes > 0 {
                    nx_tile_in = nx_tile;
                    ny_tile_in = ny_tile;
                    nz_chunk_in = nz_chunk;
                    let mut ierr = 0_i32;
                    ii_best_tile_size(output_nx, &mut nx_tile_in, &mut ierr, 1);
                    ii_best_tile_size(output_ny, &mut ny_tile_in, &mut ierr, 1);
                    ii_best_tile_size(
                        num_output_sections[output_index],
                        &mut nz_chunk_in,
                        &mut ierr,
                        1,
                    );
                    if iiu_alt_chunk_sizes(2, nx_tile_in, ny_tile_in, nz_chunk_in) != 0 {
                        exit_error("Setting chunk sizes in new volume");
                    }
                    if nx_tile_in != output_nx
                        || ny_tile_in != output_ny
                        || nz_chunk_in != num_output_sections[output_index]
                    {
                        // `write(*,'(a,i7,a,i7,a,i4)')` (`newstack.f90:1809`).
                        println!(
                            "Actual chunk size: {nx_tile_in:7} by{ny_tile_in:7} by{nz_chunk_in:4}"
                        );
                    }
                }
                //
                // handle complex images here and skip out
                // (`newstack.f90:1959-1963`)
                //
                if ((output_mode + 1) / 2 == 2 || (header.mode + 1) / 2 == 2)
                    && ((header.mode + 1) / 2 != 2 || (output_mode + 1) / 2 != 2)
                {
                    exit_error("All input files must be complex if any are");
                }
                // `iiuTransHeader` (`unit_header.c:381-385`) works on the
                // *unit's* header, `iiuMrcHeader(2, ...)`, which is
                // `iiFile->header` for an MRC, RAW, HDF or shared-memory
                // output and a separately allocated header that
                // `iiFillMrcHeader` populated for a TIFF
                // (`unit_fileio.c:253-269`).  Everything the source does to
                // the output header between here and `iiuWriteHeader` --
                // `iiuAltMode`, `iiuAltSize`, `iiuAltSample`, `iiuAltCell`,
                // `iiuAltOrigin` -- goes through that same pointer, and
                // `iiuWriteLines` re-syncs a TIFF unit's `ImodImageFile` from
                // it on every write (`unit_fileio.c:715-716`).  Keeping this
                // header anywhere else leaves `u->header->ny` at zero for a
                // TIFF output and `setupCurrentLines` (`unit_fileio.c:744-750`)
                // then refuses every write with `ny 0`.
                let out_header =
                    iiu_mrc_header(2, c"iiuTransHeader".as_ptr(), iiu_get_exit_on_error(), 2);
                // `iiuTransHeader` (`unit_header.c:381-385`) saves and restores
                // the destination `fp` around the whole-header copy.
                let fp_save = (*out_header).fp;
                std::ptr::copy_nonoverlapping(&header, out_header, 1);
                (*out_header).fp = fp_save;
                // `iiuTransHeader` ends in `iiuTransExtendedData`
                // (`unit_header.c:395`), whose first act -- for every output
                // type but TIFF and shared memory -- is
                // `iiuTransAdocSections(intoUnit, iunit)`
                // (`unit_header.c:1191`).  That carries the input's global
                // (PreData) autodoc section to the output's
                // (`iimage.c:1047-1051`), which is what re-emits the input's
                // unconsumed attributes with the `IMOD.` prefix at
                // `iihdf.c:1364`.  This is the HDF->HDF transfer that
                // `newstack.f90:1939-1940` says "itrhdr takes care of".  The
                // source calls `iiuTransHeader` as a subroutine, so a failure
                // here is discarded exactly as it is there.
                // `iiuTransHeader` runs `mrcInitOutputHeader`
                // (`unit_header.c:387`) over the copied header, never
                // `mrc_head_new`.  The two are not interchangeable:
                // `mrc_head_new` (`mrcfiles.c:689-766`) additionally resets
                // `amin` to `FLT_MAX`, `amax` to `-FLT_MAX`, `amean`, `ispg`,
                // `creatid`, `sub`, `zfac`, `min2`-`max3`, `idtype`, `lens`,
                // `nd1`-`vd2`, `blank`, `tiltangles` and `iiuFlags`, every one
                // of which the source's whole-header copy keeps from the
                // input.  Two of those are visible in the output: the
                // `tiltangles` an MRC output carries over, and the `amin` and
                // `amax` a TIFF output puts in `SMinSampleValue` and
                // `SMaxSampleValue` on the first directory
                // (`iitif.c:2591-2593`, `iitif.c:3165-3166`) -- the sentinels
                // fail that `amax > amin` test and the tags go missing.
                // Size and mode are what `iiuAltMode`
                // (`unit_header.c:245-263`) and `iiuAltSize`
                // (`unit_header.c:307-317`) set next
                // (`newstack.f90:1813-1821`).
                mrc_init_output_header(&mut *out_header);
                // `iiuTransHeader` ends in `iiuTransExtendedData`
                // (`unit_header.c:395`), after `mrcInitOutputHeader` has
                // cleared `next`, `nint`, `nreal`, `nversion` and `extType`
                // (`mrcfiles.c:797-801`).  That call restores `nint`, `nreal`
                // and a valid `extType` from the input and writes the input's
                // extended data to the output; the extra-header sizing below
                // then overrides `next` and `headerSize`.  The source calls
                // `iiuTransHeader` as a subroutine, so a failure here is
                // discarded exactly as it is there.
                iiu_trans_extended_data(2, 1);
                (*out_header).nx = output_nx;
                (*out_header).ny = output_ny;
                (*out_header).nz = num_output_sections[output_index];
                (*out_header).mode = output_mode;
                // `iiuTransHeader`, `iiuAltSize`, `iiuAltSample`, and
                // `iiuAltCell` in newstack.f90:1814-1852 retain the input
                // geometry while changing the output sampling; the whole-header
                // copy above already carries it, and these repeat what
                // `iiuAltSize`'s `nxyzst` argument sets from the input.
                (*out_header).nxstart = header.nxstart;
                (*out_header).nystart = header.nystart;
                (*out_header).nzstart = header.nzstart;
                (*out_header).mapc = header.mapc;
                (*out_header).mapr = header.mapr;
                (*out_header).maps = header.maps;
                (*out_header).imod_stamp = header.imod_stamp;
                // `imodFlags` is deliberately not carried over: `iiuTransHeader`
                // (`unit_header.c:387`) runs `mrcInitOutputHeader` over the
                // copied header, which resets it to `MRC_FLAGS_BAD_RMS_NEG`
                // (`mrcfiles.c:790`).
                // `newstack.f90:1839-1840`: `iiuAltCell` writes `cell2(4:6)`,
                // which the source sets to 90 for every output file.
                (*out_header).alpha = 90.0;
                (*out_header).beta = 90.0;
                (*out_header).gamma = 90.0;
                (*out_header).xorg = header.xorg;
                (*out_header).yorg = header.yorg;
                (*out_header).zorg = header.zorg;
                if header.mx == header.nx && header.my == header.ny && header.mz == header.nz {
                    (*out_header).mx = output_nx;
                    (*out_header).my = output_ny;
                    (*out_header).mz = num_output_sections[output_index];
                } else {
                    (*out_header).mx = header.mx;
                    (*out_header).my = header.my;
                    (*out_header).mz = header.mz;
                }
                // `newstack.f90:1838-1843`: the X and Y cell sizes carry the
                // read reduction and expansion, and the Z size is then reset
                // without them.
                (*out_header).xlen =
                    (*out_header).mx as f32 * (header.xlen / header.mx as f32) * read_reduction
                        / expand_factor;
                (*out_header).ylen =
                    (*out_header).my as f32 * (header.ylen / header.my as f32) * read_reduction
                        / expand_factor;
                (*out_header).zlen = (*out_header).mz as f32 * (header.zlen / header.mz as f32);
                if pixel_from_mdoc {
                    let (input_index, input_section) = routes[route_index];
                    if let Some(spacing) = mdoc_pixel_spacing[input_index][input_section as usize] {
                        // `newstack.f90:1848-1849`: the mdoc spacing is
                        // scaled by `readReduction` just as the header
                        // spacing above it is, so `-pixel` with `-bin` or
                        // `-shrink` keeps the reduced pixel size.
                        (*out_header).xlen =
                            (*out_header).mx as f32 * spacing * read_reduction / expand_factor;
                        (*out_header).ylen =
                            (*out_header).my as f32 * spacing * read_reduction / expand_factor;
                        (*out_header).zlen = (*out_header).mz as f32 * spacing;
                    }
                }
                //
                // adjust extra header information if currently open file has it
                // (`newstack.f90:1904-1935`)
                //
                n_byte_sym_out = 0;
                if (n_byte_sym_in > 0 || save_tilts)
                    && !strip_extra
                    && (*out_file).file == IIFILE_MRC
                {
                    if n_byte_sym_in == 0 {
                        n_byte_extra_out = 4;
                        serial_em_type = false;
                        // `iiuAltExtendedType(2, 0, 1)` (`unit_header.c:1099`)
                        // sets `nint` and `nreal` on unit 2's header.
                        (*out_header).nint = 0;
                        (*out_header).nreal = 1;
                    } else {
                        get_extra_header_max_sec_size(
                            extra_in.as_mut_ptr().cast(),
                            n_byte_sym_in,
                            num_int_or_bytes_in,
                            i_flag_extra_in,
                            header.nz,
                            &raw mut n_byte_extra_out,
                        );
                    }
                    n_byte_sym_out = num_output_sections[output_index] * n_byte_extra_out;
                    if max_extra_out > 0 && n_byte_sym_out > max_extra_out {
                        extra_out = Vec::new();
                        max_extra_out = 0;
                    }
                    //
                    // Allocate array if needed
                    if max_extra_out == 0 {
                        max_extra_out = n_byte_sym_out + 1024;
                        extra_out = vec![0_u8; max_extra_out as usize];
                    }
                    // `iiuAltNumExtended(2, nByteSymOut)` (`unit_header.c:1078`).
                    (*out_header).next = n_byte_sym_out;
                    (*out_header).header_size = 1024 + n_byte_sym_out;
                    iiu_set_position(2, 0, 0);
                    ind_extra_out = 0;
                } else {
                    // `iiuAltNumExtended(2, 0)`.  `iiuTransExtendedData` has
                    // already copied `nint`, `nreal` and `extType` from the
                    // input, so with `-strip` those survive with no data.
                    (*out_header).next = 0;
                    (*out_header).header_size = 1024;
                }
                // `newstack.f90:1855-1889`.  The origin shifts by the
                // fractional-pixel offset whenever the read reduced the image,
                // and `-origin` then re-centres with the *reduced* size and a
                // delta scaled by the reduction over the expansion.
                {
                    let mut delta = [
                        header.xlen / header.mx as f32,
                        header.ylen / header.my as f32,
                        header.zlen / header.mz as f32,
                    ];
                    let mut x_origin = (*out_header).xorg;
                    let mut y_origin = (*out_header).yorg;
                    let mut z_origin = (*out_header).zorg;
                    if read_reduction > 1. {
                        x_origin -= delta[0] * rx_offset;
                        y_origin -= delta[1] * ry_offset;
                    }
                    if adjust_origin {
                        let first_section = routes[route_index].1;
                        let x_center = x_offsets[route_index];
                        let y_center = y_offsets[route_index];
                        z_origin -= first_section as f32 * delta[2];
                        delta[0] = delta[0] * read_reduction / expand_factor;
                        delta[1] = delta[1] * read_reduction / expand_factor;
                        if transforms.is_empty() {
                            x_origin -= (bin_nx / 2 + x_center.round() as i32 - output_nx / 2)
                                as f32
                                * delta[0];
                            y_origin -= (bin_ny / 2 + y_center.round() as i32 - output_ny / 2)
                                as f32
                                * delta[1];
                        } else if apply_first != 0 {
                            x_origin -= (expand_factor * (bin_nx as f32 / 2. + x_center)
                                - output_nx as f32 / 2.)
                                * delta[0];
                            y_origin -= (expand_factor * (bin_ny as f32 / 2. + y_center)
                                - output_ny as f32 / 2.)
                                * delta[1];
                        } else {
                            x_origin -= (expand_factor * bin_nx as f32 / 2. + x_center
                                - output_nx as f32 / 2.)
                                * delta[0];
                            y_origin -= (expand_factor * bin_ny as f32 / 2. + y_center
                                - output_ny as f32 / 2.)
                                * delta[1];
                        }
                    }
                    if adjust_origin || read_reduction > 1. {
                        (*out_header).xorg = x_origin;
                        (*out_header).yorg = y_origin;
                        (*out_header).zorg = z_origin;
                    }
                }
                // `iiuTransHeader` retains source titles, as the whole-header
                // copy above does.  The run's own title is
                // *not* applied at creation: the source writes this header in
                // exactly two places, `iiuWriteHeader(2, title, 1, ...)`
                // (`newstack.f90:2753`) and `iiuWriteHeader(2, title, -1, ...)`
                // (`newstack.f90:2765`), and appending the label is part of
                // what the first of those does (`unit_header.c:290-296`).
                // Nothing is written here -- an HDF output that is written at
                // creation lays its attribute block down before the raw data
                // and leaves free space behind it.
                (*out_header).labels = header.labels;
                (*out_header).nlabl = header.nlabl;
                ii_sync_from_mrc_header(out_file, out_header);
                (*out_file).llx = 0;
                (*out_file).lly = 0;
                (*out_file).llz = 0;
                (*out_file).urx = output_nx - 1;
                (*out_file).ury = output_ny - 1;
                (*out_file).urz = num_output_sections[output_index] - 1;
                // `newstack.f90:1906`.
                out_doc_changed = false;
                //
                // `newstack.f90:1936-1956`: attach or create the output
                // autodoc, copy the input's global section and every
                // collection other than `ZValue` and `T` into it, or compose a
                // global section when the input was a FrameSet.
                //
                let open_mdoc_or_new = if use_mdoc_files { -1 } else { 0 };
                let index = iiu_ret_adoc_index(2, 0, open_mdoc_or_new);
                ind_adoc_out = if index >= 0 { index + 1 } else { index };
                if ((*out_file).file != 5 || (*active_input_file).file != 5)
                    && ind_adoc_out > 0
                    && ind_adoc_in > 0
                    && !frame_set
                {
                    set_current_adoc_or_exit(ind_adoc_in, "input");
                    if adoc_transfer_section(
                        ADOC_GLOBAL_NAME.as_ptr(),
                        0,
                        ind_adoc_out - 1,
                        ADOC_GLOBAL_NAME.as_ptr(),
                        0,
                    ) != 0
                    {
                        exit_error("Transferring global data between autodocs");
                    }
                    if let Err(message) = transfer_collections(
                        ADOC_ZVALUE_NAME.to_str().unwrap_or_default(),
                        ind_adoc_out,
                    ) {
                        exit_error(&message);
                    }
                }
                if ind_adoc_out > 0 && ind_adoc_in > 0 && frame_set {
                    set_current_adoc_or_exit(ind_adoc_out, "output");
                    if adoc_set_key_value(
                        ADOC_GLOBAL_NAME.as_ptr(),
                        0,
                        c"ImageFile".as_ptr(),
                        name.as_ptr(),
                    ) != 0
                        || adoc_set_two_integers(
                            ADOC_GLOBAL_NAME.as_ptr(),
                            0,
                            c"ImageSize".as_ptr(),
                            output_nx,
                            output_ny,
                        ) != 0
                        || adoc_set_integer(
                            ADOC_GLOBAL_NAME.as_ptr(),
                            0,
                            c"DataMode".as_ptr(),
                            output_mode,
                        ) != 0
                    {
                        exit_error("Setting global section of output autodoc");
                    }
                }
                out_file
            };
            // `dmin`, `dmax`, and `dmean` are accumulated for this output
            // file exactly where the source accumulates section `dmin2`,
            // `dmax2`, and `dmean2` before its final `iiuWriteHeader`.
            // `newstack.f90:1899-1901` sets these only when a new output
            // file is created; with `-replace` they keep the values `irdhdr`
            // put there from the existing output file, so the header ends up
            // with the min and max over both old and replaced sections.
            let mut dmin = if list_replace.is_empty() {
                f32::INFINITY
            } else {
                replace_dmin
            };
            let mut dmax = if list_replace.is_empty() {
                f32::NEG_INFINITY
            } else {
                replace_dmax
            };
            let mut dsum = 0.0_f64;
            // Source `dmean` (`newstack.f90:92`) is a `real*4` sum of the
            // per-section `dmean2` values, divided by the section count at
            // `newstack.f90:2752`; it is not a sum over every pixel.
            let mut dmean = 0.0_f32;
            for out_section in 0..num_output_sections[output_index] as usize {
                let (input_index, in_section) = routes[route_index];
                route_index += 1;
                // `newstack.f90:1525-1527`: the source's outer loop is over
                // input files, so each file is opened with `openInputFile` and
                // read with `irdhdr` before any of its sections is processed.
                // This route is output-file major, so the change of input file
                // happens here instead, and it refreshes the same header
                // fields the source's `irdhdr` refreshes.
                if active_input_index != input_index {
                    if !active_input_file.is_null() {
                        iiu_close(1);
                        // `newstack.f90:2762`.
                        if need_close1 > 0 {
                            iiu_close(need_close1);
                        }
                        active_input_file = std::ptr::null_mut();
                    }
                    // `call openInputFile(iFile)` (`newstack.f90:1526`).
                    open_input_file(
                        input_index + 1,
                        num_vol_read,
                        &list_volumes,
                        &input_names,
                        &mut need_close1,
                    );
                    let (mut nxyz, mut mxyz, mut mode, mut dmin_in, mut dmax_in, mut dmean_in) =
                        ([0_i32; 3], [0_i32; 3], 0, 0., 0., 0.);
                    irdhdr(
                        1,
                        nxyz.as_mut_ptr(),
                        mxyz.as_mut_ptr(),
                        &mut mode,
                        &mut dmin_in,
                        &mut dmax_in,
                        &mut dmean_in,
                    );
                    active_input_file = iiu_get_ii_file(1);
                    if !active_input_file.is_null() {
                        active_input_index = input_index;
                        // `unit_fileio.c:256-265`: the unit's own `MrcHeader`.
                        header =
                            std::ptr::read(iiu_mrc_header(1, c"iiuRetBasicHead".as_ptr(), 1, 0));
                        // `newstack.f90:1570-1571`: the binned size to read is
                        // this file's, so inputs of different sizes each read
                        // their own extent into the common output size.
                        (bin_nx, rx_offset) =
                            get_reduced_size(header.nx, read_reduction, read_shrunk, odd_even_ok);
                        (bin_ny, ry_offset) =
                            get_reduced_size(header.ny, read_reduction, read_shrunk, odd_even_ok);
                        // `newstack.f90:1572-1573`.
                        if i_verbose > 0 {
                            print!(
                                " Size and offsets X: {:>11} {} , Y: {:>11} {}\n",
                                bin_nx,
                                list_real(rx_offset),
                                bin_ny,
                                list_real(ry_offset)
                            );
                        }
                        //
                        // get extra header information if any (`newstack.f90:1575-1683`).  The
                        // autodoc block below is the source's `newstack.f90:1535-1565,1663-1665`;
                        // nothing here touches an autodoc, so the two are independent.
                        //
                        iiu_ret_num_extended(1, &raw mut n_byte_sym_in);
                        let mut itype = 0_i32;
                        num_int_or_bytes_in = 0;
                        i_flag_extra_in = 0;
                        fei1_type = false;
                        if need_header_angles && n_byte_sym_in == 0 {
                            exit_error(
                                "There is no extended header; tilt angles for -reorder must be entered with the -angles option",
                            );
                        }
                        if n_byte_sym_in > 0 {
                            //
                            // Deallocate array if it was allocated and is not big enough
                            if max_extra_in > 0 && n_byte_sym_in > max_extra_in {
                                extra_in = Vec::new();
                                max_extra_in = 0;
                            }
                            //
                            // Allocate array if needed
                            if max_extra_in == 0 {
                                max_extra_in = n_byte_sym_in + 1024;
                                extra_in = vec![0_u8; max_extra_in as usize];
                            }
                            iiu_ret_extended_data(
                                1,
                                &raw mut n_byte_sym_in,
                                extra_in.as_mut_ptr().cast(),
                            );
                            iiu_ret_extended_type(
                                1,
                                &raw mut num_int_or_bytes_in,
                                &raw mut i_flag_extra_in,
                            );
                            //
                            // DNM 4/18/02: if these numbers do not represent bytes and
                            // flags, then number of bytes is 4 times nint + nreal
                            //
                            serial_em_type = nbytes_and_flags(num_int_or_bytes_in, i_flag_extra_in);
                            itype = 1;
                            if serial_em_type {
                                itype = -1;
                            }
                            //
                            // Mark new FEI type as 2 and no type for unknown
                            if num_int_or_bytes_in < 0 {
                                if num_int_or_bytes_in == -3 {
                                    itype = 2;
                                    fei1_type = true;
                                } else {
                                    itype = 0;
                                    n_byte_sym_in = 0;
                                }
                            }
                            if serial_em_type && save_tilts && i_flag_extra_in % 2 == 0 {
                                exit_error(
                                    "You cannot save tilt angles into a SerialEM extended header that was not saved with tilt angles",
                                );
                            }
                            //
                            // Get tilt angles for reordering
                            if need_header_angles {
                                //
                                // Take care of tilt angle array
                                if header.nz > max_in_file_angles {
                                    max_in_file_angles = header.nz + 8;
                                    all_file_tilts = vec![0.0_f32; max_in_file_angles as usize];
                                    iz_not_piece = vec![0_i32; max_in_file_angles as usize];
                                }
                                //
                                // Set up dummy piece lists and get the tilt angles
                                for ind in 0..header.nz as usize {
                                    iz_not_piece[ind] = ind as i32;
                                }
                                let mut num_tilts = 0_i32;
                                let mut nz_in = header.nz;
                                get_extra_header_tilts_fortran(
                                    extra_in.as_mut_ptr().cast(),
                                    &raw mut n_byte_sym_in,
                                    &raw mut num_int_or_bytes_in,
                                    &raw mut i_flag_extra_in,
                                    &raw mut nz_in,
                                    all_file_tilts.as_mut_ptr(),
                                    &raw mut num_tilts,
                                    &raw mut max_in_file_angles,
                                    iz_not_piece.as_mut_ptr(),
                                );
                                if num_tilts < header.nz {
                                    exit_error(
                                        "There are either not enough or no tilt angles in extended header; tilt angles for -reorder must be entered with the -angles option",
                                    );
                                }
                                //
                                // Fill the angle arrays from the section list
                                let mut start = 0usize;
                                for sections in &section_lists[..input_index] {
                                    start += sections.len();
                                }
                                for ind in start..start + section_lists[input_index].len() {
                                    extra_tilts[ind] = all_file_tilts[routes[ind].1 as usize];
                                }
                            }
                        }
                        if iany_extra_type == 0 {
                            iany_extra_type = itype;
                        }
                        if input_index == 0 {
                            ifirst_extra_type = itype;
                        }
                        if iany_extra_type != 0
                            && itype != 0
                            && iany_extra_type != itype
                            && !strip_extra
                        {
                            exit_error(
                                "You cannot include files with different types of extended headers in the same run; add the -strip option",
                            );
                        }
                        if save_tilts && itype != ifirst_extra_type {
                            exit_error(
                                "To store tilt angles, all input files must have the same type of extended header or no extended header",
                            );
                        }
                        if save_tilts && itype == 2 {
                            exit_error(
                                "You cannot store tilt angles back into a new FEI1-style extended header",
                            );
                        }
                        //
                        // Reorder the section list and output angles for this file
                        // (`newstack.f90:1667-1683`).  The entered-angle-file route was reordered
                        // before any input was opened; this is the `needHeaderAngles` route, whose
                        // angles only exist once the file's extended header has been read.
                        //
                        if reorder_by_tilt != 0 && need_header_angles {
                            let mut start = 0usize;
                            for sections in &section_lists[..input_index] {
                                start += sections.len();
                            }
                            let end = start + section_lists[input_index].len();
                            for ind in start..end.saturating_sub(1) {
                                for ilist in ind + 1..end {
                                    if reorder_by_tilt.signum() as f32
                                        * (extra_tilts[ind] - extra_tilts[ilist])
                                        > 0.01
                                    {
                                        routes.swap(ind, ilist);
                                        extra_tilts.swap(ind, ilist);
                                    }
                                }
                            }
                        }
                        // `newstack.f90:1535-1565, 1663-1665`: the source obtains this
                        // file's autodoc index at the top of its input-file loop.  A
                        // single-image file or a `-remove` name substitution opens the
                        // metadata file by name; otherwise the file's own index is
                        // used, created when `-mdoc` or `-pixel` asked for it.
                        ind_adoc_in = 0;
                        if (*active_input_file).file != 5
                            && use_mdoc_files
                            && (header.nz == 1 || !remove_from_name.is_empty())
                        {
                            let mut list_string = input_names[input_index].clone();
                            if !remove_from_name.is_empty() {
                                let base = list_string
                                    .strip_suffix(remove_from_name.as_str())
                                    .filter(|base| !base.is_empty());
                                let Some(base) = base else {
                                    exit_error(&format!(
                                        "{list_string} does not end with {remove_from_name}"
                                    ));
                                };
                                list_string = format!("{base}{add_to_name}");
                            }
                            // `ierr = 1`, or 2 for a single-image file, which is what
                            // lets `AdocOpenImageMetadata` keep a FrameSet autodoc.
                            let add_mdoc = if header.nz == 1 { 2 } else { 1 };
                            let Ok(list_c) = CString::new(list_string.as_bytes()) else {
                                exit_error("Invalid input file name");
                            };
                            let (mut montage, mut num_sect, mut sect_type) = (0_i32, 0_i32, 0_i32);
                            let opened = adoc_open_image_metadata(
                                list_c.as_ptr(),
                                add_mdoc,
                                &raw mut montage,
                                &raw mut num_sect,
                                &raw mut sect_type,
                            );
                            ind_adoc_in = if opened >= 0 { opened + 1 } else { opened };
                            if ind_adoc_in <= 0 {
                                exit_error(&format!(
                                    "No mdoc file found with name {list_string}.mdoc"
                                ));
                            }
                            frame_set = sect_type == 4;
                        }
                        let open_mdoc_or_new = i32::from(use_mdoc_files || pixel_from_mdoc);
                        if ind_adoc_in <= 0 {
                            let index = iiu_ret_adoc_index(1, 0, open_mdoc_or_new);
                            ind_adoc_in = if index >= 0 { index + 1 } else { index };
                        }
                    }
                }
                //
                // determine whether rescaling will be needed
                // (`newstack.f90:1985-2007`).  The source recomputes this for
                // every section from the open input file's `mode` and
                // `dminIn`, so with several input files of different modes
                // each one gets its own decision; taking it once from the
                // first file truncates the others instead of scaling them.
                //
                rescale = false;
                // `packed4bitInput` is recomputed from this file's unit flags
                // (`newstack.f90:1530-1531`), inside the same input-file loop.
                packed_4bit_input = header.iiu_flags & (IIUNIT_4BIT_MODE | IIUNIT_HALF_XSIZE) != 0;
                if if_float == 0 && output_mode != 2 && header.mode != 2 {
                    // `newstack.f90:1989`.
                    rescale =
                        header.mode != output_mode || (pack_4bit_output && !packed_4bit_input);
                } else if if_float != 0 {
                    rescale = true;
                }
                optimal_in = optimal_max[header.mode as usize];
                // `newstack.f90:1997`.
                if packed_4bit_input {
                    optimal_in = 15.;
                }
                //
                // set bottom of input range to 0 unless mode 1 or 2 and
                // already negative or not rescaling; set bottom of output
                // range to 0 unless not changing modes
                //
                bottom_in = 0.;
                if header.amin < 0. || !rescale {
                    if header.mode == 1 {
                        bottom_in = -optimal_in - 1.;
                    }
                    if header.mode == 2 {
                        bottom_in = -optimal_in;
                    }
                }
                bottom_out = 0.;
                if header.mode == output_mode {
                    bottom_out = bottom_in;
                }
                // Source `dmeanSec` for filling outside the input image
                // (`newstack.f90:2305-2312`).
                dmean_sec = fill_value.unwrap_or(header.amean);
                // `reallocateIfNeeded` (`newstack.f90:2181, 2846-2866`) grows
                // the working array to this file's binned input plus the
                // output section, so a later input file that is larger than
                // the first still has room to be read.
                let need_dim = (bin_nx as usize * bin_ny as usize)
                    .max(output_nx as usize * output_ny as usize);
                if array.len() < need_dim {
                    array.resize(need_dim, 0.0);
                }
                // `newstack.f90:2024`: a section outside the input file is the
                // blank one, whatever `-blank` was entered for.
                let blank_section = in_section < 0 || in_section >= header.nz;
                // `newstack.f90:2046`.  The blank-section branch above it
                // (`newstack.f90:2024-2044`) leaves the section loop before
                // this print, so a blank section never reports.
                if i_verbose > 0 && !blank_section {
                    print!(" rescale {}\n", if rescale { "T" } else { "F" });
                }
                if chunked_affine && !blank_section {
                    let Some(transform) = transforms.get(transform_lines[route_index - 1] as usize)
                    else {
                        exit_error(&format!(
                            "TRANSFORM LINE number out of bounds:{in_section:5}"
                        ));
                    };
                    let offset_index = route_index - 1;
                    //
                    // if doing distortions or warping, get the grid
                    // (`newstack.f90:2055-2110`)
                    //
                    let mut has_warp = false;
                    let (mut nx_grid, mut ny_grid) = (0_i32, 0_i32);
                    let (mut x_grid_start, mut y_grid_start) = (0.0_f32, 0.0_f32);
                    let (mut x_grid_intrv, mut y_grid_intrv) = (0.0_f32, 0.0_f32);
                    let (mut grid_iy, mut grid_dx, mut grid_dy) = (0_i32, 0.0_f32, 0.0_f32);
                    let (mut xn_big, mut yn_big) = (0.0_f32, 0.0_f32);
                    if if_distort > 0 {
                        grid_iy = idf_use[offset_index] + 1;
                        has_warp = true;
                        xn_big = header.nx as f32 / warp_scale;
                        yn_big = header.ny as f32 / warp_scale;
                    } else if if_warping != 0 {
                        grid_iy = transform_lines[offset_index] + 1;
                        has_warp = n_control[grid_iy as usize - 1] > 2;
                        xn_big = read_reduction * output_nx as f32 / warp_scale;
                        yn_big = read_reduction * output_ny as f32 / warp_scale;
                        if apply_first == 0 {
                            grid_dx = read_reduction * x_offsets[offset_index] / warp_scale;
                            grid_dy = read_reduction * y_offsets[offset_index] / warp_scale;
                        }
                    }
                    if has_warp {
                        let mut list_string = vec![0_i8; 1024];
                        if get_size_adjusted_grid(
                            grid_iy - 1,
                            xn_big,
                            yn_big,
                            grid_dx,
                            grid_dy,
                            1,
                            warp_scale,
                            read_reduction.round() as i32,
                            &raw mut nx_grid,
                            &raw mut ny_grid,
                            &raw mut x_grid_start,
                            &raw mut y_grid_start,
                            &raw mut x_grid_intrv,
                            &raw mut y_grid_intrv,
                            field_dx.as_mut_ptr(),
                            field_dy.as_mut_ptr(),
                            lm_grid,
                            lm_grid,
                            list_string.as_mut_ptr(),
                            list_string.len() as i32,
                        ) != 0
                        {
                            let message = std::ffi::CStr::from_ptr(list_string.as_ptr())
                                .to_string_lossy()
                                .into_owned();
                            exit_error(message.trim_end());
                        }
                        for iy in 0..ny_grid as usize {
                            for ix in 0..nx_grid as usize {
                                let index = ix + iy * lm_grid as usize;
                                tmp_dx[index] = field_dx[index];
                                tmp_dy[index] = field_dy[index];
                            }
                        }
                    }
                    if if_mag_grad != 0 {
                        let mag_use = (in_section + 1).min(num_mag_grad).max(1) as usize;
                        let (grad_nx, grad_ny) = (
                            if bin_factor > 1 { bin_nx } else { header.nx },
                            if bin_factor > 1 { bin_ny } else { header.ny },
                        );
                        if if_distort != 0 {
                            add_mag_grad_field(
                                tmp_dx.as_mut_ptr(),
                                tmp_dy.as_mut_ptr(),
                                field_dx.as_mut_ptr(),
                                field_dy.as_mut_ptr(),
                                lm_grid,
                                grad_nx,
                                grad_ny,
                                nx_grid,
                                ny_grid,
                                x_grid_start,
                                y_grid_start,
                                x_grid_intrv,
                                y_grid_intrv,
                                grad_nx as f32 / 2.,
                                grad_ny as f32 / 2.,
                                pixel_mag_grad,
                                axis_rot,
                                tilt_angles[mag_use - 1],
                                dmag_per_micron[mag_use - 1],
                                rot_per_micron[mag_use - 1],
                            );
                        } else {
                            make_mag_grad_field(
                                tmp_dx.as_mut_ptr(),
                                tmp_dy.as_mut_ptr(),
                                field_dx.as_mut_ptr(),
                                field_dy.as_mut_ptr(),
                                lm_grid,
                                grad_nx,
                                grad_ny,
                                &raw mut nx_grid,
                                &raw mut ny_grid,
                                &raw mut x_grid_start,
                                &raw mut y_grid_start,
                                &raw mut x_grid_intrv,
                                &raw mut y_grid_intrv,
                                grad_nx as f32 / 2.,
                                grad_ny as f32 / 2.,
                                pixel_mag_grad,
                                axis_rot,
                                tilt_angles[mag_use - 1],
                                dmag_per_micron[mag_use - 1],
                                rot_per_micron[mag_use - 1],
                            );
                        }
                    }
                    let (mut max_field_x, mut max_field_y) = (0_i32, 0_i32);
                    if if_mag_grad != 0 || has_warp {
                        let (mut field_max_x, mut field_max_y) = (0.0_f32, 0.0_f32);
                        for iy in 0..ny_grid as usize {
                            for ix in 0..nx_grid as usize {
                                let index = ix + iy * lm_grid as usize;
                                field_max_x = field_max_x.max(field_dx[index].abs());
                                field_max_y = field_max_y.max(field_dy[index].abs());
                            }
                        }
                        max_field_x = (field_max_x as f64 + 1.5) as i32;
                        max_field_y = (field_max_y as f64 + 1.5) as i32;
                    }
                    let mut fprod = *transform;
                    if apply_first != 0 {
                        let frot = [
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            -x_offsets[offset_index],
                            -y_offsets[offset_index],
                        ];
                        let mut composed = [0.0_f32; 6];
                        xfmult(&frot, &fprod, &mut composed);
                        fprod = composed;
                    } else if !(if_warping != 0 && has_warp) {
                        fprod[4] -= x_offsets[offset_index];
                        fprod[5] -= y_offsets[offset_index];
                    }
                    let affine = LinesNeededForOutput {
                        fourier_scaling: false,
                        if_xform: 1,
                        // Source `nxBin`/`nyBin` (`newstack.f90:2925-2940`),
                        // which `getReducedSize` gives for binning *and* for
                        // an antialiased shrink read.
                        nx_bin: bin_nx,
                        ny_bin: bin_ny,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        xcen: 0.0,
                        ycen: 0.0,
                        // `linesNeededForOutput` hands `fprod` on to
                        // `backXform` as its `amat(2,2)`, so the pairs here
                        // are Fortran `fprod(i, j)`: the flat transform is in
                        // column-major storage, which puts `a21` at index 1
                        // and `a12` at index 2.
                        fprod: [
                            [fprod[0], fprod[2], fprod[4]],
                            [fprod[1], fprod[3], fprod[5]],
                        ],
                        max_field_x,
                        max_field_y,
                        lines_shrink,
                    };
                    //
                    // get the mean of section for filling outside the image
                    // (`newstack.f90:2300-2321`), in the source's own order:
                    // an entered fill, then the edge mean when there is only
                    // one chunk, then the mean from the preliminary scan, then
                    // the input header's mean, and only failing all of those a
                    // scan of the section here.
                    //
                    let section_needed = lines_needed_for_output(&affine, 0, output_ny - 1);
                    // `newstack.f90:2158-2198`.  `nxDimNeed` and `nyDimNeed` are the
                    // row and line counts the input plus any FFT padding needs,
                    // `processInPlace` says the output can be built back into the
                    // input space, and `linesLeft`/`numChunks` follow from the room
                    // `reallocateIfNeeded` leaves in the flat `array`.
                    //
                    // The source computes every one of these **inside its section
                    // loop** (`newstack.f90:2046-2281`): `linesNeededForOutput`
                    // runs on this section's own `fprod`, so `nyNeeded` --
                    // and with it `nyDimNeed`, `linesLeft` and `numChunks` --
                    // changes from section to section, and `numChunks` selects the
                    // branch at `newstack.f90:2209` and sizes the chunk table
                    // built there.  Hoisted to one value for the whole run,
                    // `-xform xf2.txt -size 500,400 -test 400000,1` laid section 1
                    // out in the 2 chunks section 0 needed where the reference
                    // uses 3 -- a different `lineOutSt`/`numLinesOut` split, and
                    // so a different `ycenIn` and a different order for the
                    // per-chunk sums.
                    //
                    let ny_needed = section_needed.iy_in_2 - section_needed.iy_in_1 + 1;
                    let nx_dim_need = bin_nx.max(nx_fspad + 2);
                    let ny_dim_need = ny_needed.max(ny_fspad + 1);
                    process_in_place = transforms.is_empty()
                        && output_nx <= nx_dim_need
                        && output_ny <= ny_dim_need
                        && section_needed.in_place;
                    let mut allocation = ReallocateIfNeeded {
                        physical_memory,
                        process_in_place,
                        ft_reduce_fac,
                        phase_shift,
                        lim_entered,
                        nx: header.nx,
                        ny: header.ny,
                        nx_bin: bin_nx,
                        ny_bin: bin_ny,
                        ny_needed,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        read_shrunk,
                        read_reduction,
                        i_binning: bin_factor,
                        fourier_scaling,
                        nx_fspad,
                        ny_fspad,
                        nx_fcrop_pad,
                        ny_fcrop_pad,
                        ft_expand_fac,
                        noise_pad,
                        nx_bin_fft: bin_nx,
                        ny_bin_fft: bin_ny,
                        lim_to_alloc: alloc_lim_to_alloc,
                        len_temp: alloc_len_temp,
                        pre_set_scaling,
                        idim_in_out: alloc_idim_in_out,
                        in_place_fac,
                        i_verbose,
                    };
                    // `call reallocateIfNeeded()` (`newstack.f90:2181`), which
                    // makes the `newstack.f90:2815` and `newstack.f90:2865`
                    // reports and may make `newstack.f90:2887`.
                    reallocate_if_needed(&mut allocation);
                    process_in_place = allocation.process_in_place;
                    in_place_fac = allocation.in_place_fac;
                    alloc_lim_to_alloc = allocation.lim_to_alloc;
                    alloc_len_temp = allocation.len_temp;
                    alloc_idim_in_out = allocation.idim_in_out;
                    let idim_in_out = alloc_idim_in_out as i64;
                    // `newstack.f90:2182-2183`.
                    if idim_in_out / i64::from(nx_dim_need) <= i64::from(ny_dim_need)
                        && !pre_set_scaling
                    {
                        process_in_place = false;
                    }
                    // `newstack.f90:2184-2185`.
                    if i_verbose > 0 {
                        print!(
                            " preSetScaling  {}    processInPlace  {}\n",
                            if pre_set_scaling { "T" } else { "F" },
                            if process_in_place { "T" } else { "F" }
                        );
                    }
                    // `newstack.f90:2189-2198`.  `linesLeft` and `numChunks` are
                    // `integer*4` (`newstack.f90:95`), so the `integer(kind = 8)`
                    // quotient lands in 32 bits before it is edited.
                    let mut section_num_chunks = 0_i32;
                    if idim_in_out / i64::from(nx_dim_need) > i64::from(ny_dim_need) {
                        let mut lines_left =
                            ((idim_in_out - i64::from(nx_dim_need) * i64::from(ny_dim_need))
                                / i64::from(output_nx)) as i32;
                        if process_in_place {
                            lines_left = (idim_in_out / i64::from(nx_dim_need)) as i32;
                        }
                        // A zero `linesLeft` is an integer divide by zero in the
                        // source, which has no behaviour to reproduce.
                        if lines_left != 0 {
                            section_num_chunks = (output_ny + lines_left - 1) / lines_left;
                        }
                        if i_verbose > 0 {
                            print!(
                                " linesleft {:>11}   nchunk {:>11}\n",
                                lines_left, section_num_chunks
                            );
                        }
                    }
                    //
                    // `newstack.f90:2209-2232`: when the input fits and the
                    // scaling is pre-set, the source keeps the **whole** input
                    // for every chunk and splits only the output, into
                    // `numChunks` pieces of `nyOut / numChunks` lines with the
                    // remainder spread one line at a time over the first
                    // chunks.  Reading each chunk's own input window instead
                    // is a different `ycenIn` and a different grid start, and
                    // a mag-gradient run comes out tens of counts off from the
                    // second chunk on.
                    //
                    // The layout -- `lineOutSt`, `numLinesOut`, `lineInSt`,
                    // `numLinesIn` per chunk -- exactly as
                    // `newstack.f90:2209-2274` builds it.
                    // Source `nxBin` (`newstack.f90:2264-2266`).
                    let nx_dim = bin_nx as usize;
                    let split = |count: i32, index: i32| {
                        (output_ny / count) * index + index.min(output_ny % count)
                    };
                    let mut layout = Vec::<(i32, i32, i32, i32)>::new();
                    // Source `ifOutChunk` (`newstack.f90:2233, 2239, 2270`):
                    // zero when the whole output is held at once, one when the
                    // output itself had to be broken up and every chunk but
                    // the last goes to the scratch file.
                    let mut if_out_chunk = -1_i32;
                    // `newstack.f90:2209-2210`: the whole-input branch needs
                    // either a single chunk or pre-set scaling -- without it
                    // the min/max/mean of the *output* decide the scaling, so
                    // the source falls through to the search below even when
                    // its `numChunks` is a perfectly good count.
                    if section_num_chunks == 1
                        || (section_num_chunks > 0 && section_num_chunks <= 250 && pre_set_scaling)
                    {
                        let count = section_num_chunks;
                        let mut start = 0_i32;
                        for index in 1..=count {
                            let next = split(count, index);
                            layout.push((
                                start,
                                next - start,
                                section_needed.iy_in_1,
                                section_needed.iy_in_2 - section_needed.iy_in_1 + 1,
                            ));
                            start = next;
                        }
                        // `newstack.f90:2233`.
                        if_out_chunk = 1;
                    } else {
                        //
                        // `newstack.f90:2237-2273`: break the output into
                        // successively more chunks and see how much input each
                        // one needs.  Scan once trying to hold the whole
                        // output, then again allowing the output to be chunked
                        // too.
                        //
                        let mut found = false;
                        'scans: for scan in 1..=2 {
                            let mut count = 1_i32;
                            while count <= 250 {
                                layout.clear();
                                let mut start = 0_i32;
                                let mut max_in = 0_i32;
                                for index in 1..=count {
                                    let next = split(count, index);
                                    let needed = lines_needed_for_output(&affine, start, next - 1);
                                    let lines_in = needed.iy_in_2 + 1 - needed.iy_in_1;
                                    layout.push((start, next - start, needed.iy_in_1, lines_in));
                                    max_in = max_in.max(lines_in);
                                    start = next;
                                }
                                let iy_test = if scan == 2 { layout[0].1 } else { output_ny };
                                if max_in > 0
                                    && chunk_limit / max_in as usize > nx_dim
                                    && chunk_limit / iy_test.max(1) as usize > output_nx as usize
                                    && max_in as usize * nx_dim
                                        + iy_test as usize * output_nx as usize
                                        <= chunk_limit
                                {
                                    // `newstack.f90:2270`.
                                    if_out_chunk = scan - 1;
                                    found = true;
                                    break 'scans;
                                }
                                count += 1;
                            }
                        }
                        if !found {
                            exit_error(" Input image too large for array.");
                        }
                    }
                    // `newstack.f90:2276-2281`.
                    if i_verbose > 0 {
                        print!(
                            " number of chunks: {:>11} {:>11}\n",
                            layout.len(),
                            if_out_chunk
                        );
                        for (index, &(line_out_st, num_lines_out, line_in_st, num_lines_in)) in
                            layout.iter().enumerate()
                        {
                            print!(
                                " {:>11} {:>11} {:>11} {:>11} {:>11}\n",
                                index + 1,
                                line_in_st,
                                num_lines_in,
                                line_out_st,
                                num_lines_out
                            );
                        }
                    }
                    //
                    // open temp file if one is needed
                    // (`newstack.f90:2284-2296`).  It is opened at most once
                    // for the whole run, and its extension carries the digits
                    // of the run's time string.
                    //
                    if rescale
                        && !pre_set_scaling
                        && if_out_chunk > 0
                        && layout.len() > 1
                        && if_temp_open == 0
                    {
                        let mut temp_ext = *b"nws      ";
                        temp_ext[3..5].copy_from_slice(&time_str[0..2]);
                        temp_ext[5..7].copy_from_slice(&time_str[3..5]);
                        temp_ext[7..9].copy_from_slice(&time_str[6..8]);
                        let temp_name = temp_filename(
                            &output_names[output_index],
                            " ",
                            std::str::from_utf8(&temp_ext).unwrap_or_default(),
                        );
                        imopen(3, &temp_name, "scratch");
                        let mut nxyz3 = [output_nx, output_ny, num_output_sections[output_index]];
                        iiu_create_header(
                            3,
                            nxyz3.as_mut_ptr(),
                            nxyz3.as_mut_ptr(),
                            2,
                            title.as_mut_ptr().cast(),
                            0,
                        );
                        if_temp_open = 1;
                    }
                    //
                    // The source reads the input into `array(1 ...)`, whose
                    // room for the load is `maxin` lines of `nxDimNeed`
                    // (`newstack.f90:2298`), and it keeps that one buffer
                    // across the chunk loop so the move-down and move-up
                    // shortcuts have somewhere to move data to.
                    //
                    let nx_load = bin_nx;
                    let max_in = layout.iter().map(|entry| entry.3).max().unwrap_or(0);
                    let mut input = vec![0.0_f32; nx_load.max(1) as usize * max_in.max(1) as usize];
                    //
                    // The source keeps one load window across the chunk loop
                    // (`newstack.f90:2340-2382`): it only reads when the
                    // window does not already cover what the chunk needs, and
                    // `loadYstart`/`loadYend` are set to the *needed* range
                    // only then.  So a chunk whose need is already covered
                    // keeps the earlier, wider window -- and `ycenIn` and the
                    // grid start are measured from it, which changes the
                    // interpolated value by an ulp.  Its move-down/move-up
                    // optimisations only decide which lines are re-read, not
                    // the window, so this reads the whole needed region.
                    // `newstack.f90:2302-2303` initialises the window here,
                    // *before* the scan for the fill mean, because that scan
                    // reads into the same `array` and hands back the window
                    // its last load left there.
                    //
                    let (mut load_y_start, mut load_y_end) = (-1_i32, -1_i32);
                    let mut need_edge_mean = false;
                    let mut dmean_sec = if let Some(fill) = fill_value {
                        fill
                    } else if section_needed.need_fill && layout.len() == 1 {
                        // `newstack.f90:2307-2308`: with one chunk the mean is
                        // taken from the edges of the loaded data instead, in
                        // the chunk loop below.
                        need_edge_mean = true;
                        0.0
                    } else if if_mean != 0 && !mean_sd_entered && !sec_mean.is_empty() {
                        // `newstack.f90:2309-2310`: the mean from the scan that
                        // `-float 2`, `-float 3` and `-float 4` already made.
                        sec_mean[offset_index]
                    } else if !section_needed.need_fill {
                        header.amean
                    } else {
                        let scan_ny = section_needed.iy_in_2 - section_needed.iy_in_1 + 1;
                        let scan_nx = bin_nx;
                        // `scanSection` is handed `array` with `idimInOut`
                        // elements (`newstack.f90:2317`), and it splits the
                        // section into `array.len() / nx` line loads -- so the
                        // buffer size decides how the partial sums are
                        // grouped and therefore how the mean rounds.
                        let mut scan_array = vec![0.0_f32; chunk_limit.max(scan_nx as usize)];
                        let mut scan_temp = vec![
                            0.0_f32;
                            // `lenTemp` as `reallocateIfNeeded` left it
                            // (`newstack.f90:2818-2842`).  `needTemp` starts at
                            // **one** (`newstack.f90:2822`) and only `readShrunk`
                            // and `iBinning > 1` raise it, so reproducing it as
                            // `nx * readReduction` gave `nx` where the source has
                            // 1 -- and for an entered `-memory` that is
                            // `idimInOut = limToAlloc - lenTemp`, so the scan's
                            // load grouping and the fill mean moved with it.
                            effective_len_temp.max(1) as usize
                        ];
                        // `wallStart = wallTime()` (`newstack.f90:2316`).
                        let scan_wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        // `newstack.f90:2314-2315`.
                        if i_verbose > 0 {
                            print!(
                                " scanning for mean for fill {:>11} {:>11} {:>11} {:>11}\n",
                                in_section,
                                scan_ny,
                                section_needed.iy_in_1,
                                scan_temp.len()
                            );
                        }
                        let Ok((_, _, mean, _, scan_y_start, scan_y_end)) = scan_section(
                            &mut scan_array,
                            scan_nx,
                            scan_ny,
                            section_needed.iy_in_1,
                            read_reduction,
                            rx_offset,
                            ry_offset,
                            0,
                            0.0,
                            |data, lines, x_start, y_start| {
                                read_binned_or_reduced(
                                    1,
                                    in_section,
                                    data,
                                    scan_nx,
                                    lines,
                                    x_start,
                                    y_start,
                                    read_reduction,
                                    scan_nx,
                                    lines,
                                    ind_filter,
                                    read_shrunk,
                                    &mut scan_temp,
                                )
                            },
                        ) else {
                            exit_error("Reading image file");
                        };
                        //
                        // `scanSection` reads each of its loads into
                        // `array(1)` (`newstack.f90:3437`), so the section's
                        // *last* load is still sitting at the base of the
                        // working array when it returns, and it hands back
                        // that window in `loadYstart`/`loadYend`
                        // (`newstack.f90:3463-3464`).  The chunk loop below
                        // therefore skips its first read whenever the window
                        // already covers what chunk one needs -- and then uses
                        // the *scan's* wider window for `numYload`, `ycenIn`
                        // and the grid start.  `newstack.f90:2320` clamps it to
                        // the `maxin` lines that the chunk loop's own buffer
                        // holds, which is what makes the leftover usable.
                        // This route splits the source's one `array` into
                        // `input` and a per-chunk output buffer, so the lines
                        // that survive the clamp are copied across here; the
                        // scan itself keeps a buffer of `idimInOut` elements
                        // because its load grouping decides how the mean
                        // rounds.
                        //
                        load_y_start = scan_y_start;
                        load_y_end = scan_y_end.min(scan_y_start + max_in - 1);
                        let kept =
                            (load_y_end + 1 - load_y_start).max(0) as usize * nx_load as usize;
                        input[..kept].copy_from_slice(&scan_array[..kept]);
                        // `newstack.f90:2321`.
                        load_time += crate::imod::libcfshr::b3dutil::wall_time() - scan_wall_start;
                        mean
                    };
                    //
                    // `newstack.f90:2327-2331`: the per-section accumulators
                    // run across every chunk, and the report at
                    // `newstack.f90:2622-2628` is printed once from them.
                    //
                    let (mut tmp_min, mut tmp_max) = (1.0e30_f32, -1.0e30_f32);
                    let mut chunk_scaling = ScaleAndWriteChunk {
                        new_mode: output_mode,
                        write_16_bit_mode_for_floats: write_16_bit_mode_for_floats() != 0,
                        scale_factor: 1.0,
                        const_add: 0.0,
                        optimal_out,
                        dmin2: 1.0e30,
                        dmax2: -1.0e30,
                        dmean2: 0.0,
                        num_trunc_low: 0,
                        num_trunc_high: 0,
                    };
                    let mut chunk_scale_values = FindScaleFactors {
                        if_float,
                        rescale,
                        num_scale_facs: scale_factors.len() as i32,
                        bottom_in,
                        bottom_out,
                        optimal_in,
                        optimal_out,
                        dmin_specified: scale_limits[0],
                        dmax_specified: scale_limits[1],
                        dmin_in: header.amin,
                        dmax_in: header.amax,
                        if_map_range: i32::from(map_entered),
                        dmap_low: map_limits[0],
                        dmap_high: map_limits[1],
                        frac_zero,
                        if_mean,
                        if_mean_sd_entered: i32::from(mean_sd_entered),
                        entered_mean: mean_sd[0],
                        entered_sd: mean_sd[1],
                        shift_mean,
                        shift_min,
                        shift_max,
                        new_mode: output_mode,
                        dsum: 0.0,
                        dsum_sq: 0.0,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        scale_factor: 1.0,
                        const_add: 0.0,
                        scale_fac: scale_factors
                            .get(input_index.min(scale_factors.len().saturating_sub(1)))
                            .map_or(1.0, |entry| entry[0]),
                        scale_const: scale_factors
                            .get(input_index.min(scale_factors.len().saturating_sub(1)))
                            .map_or(0.0, |entry| entry[1]),
                        float_average: 0.0,
                        float_sd: 0.0,
                        zmin: float2_zmin,
                        zmax: float2_zmax,
                        float_z_margin,
                        opt_float_range,
                        opt_float_min,
                        section_min: 0.0,
                        section_max: 0.0,
                        section_intentionally_truncated: true,
                    };
                    //
                    // `newstack.f90:2514-2545`: the per-chunk statistics.  For
                    // `-float 2` they are `iclAvgSd`'s rebuilt sums, which
                    // `chunkSumsToAvgsd` turns into the section's mean and SD;
                    // otherwise `iclden`'s mean times the pixel count, or a
                    // plain scan when nothing needs a mean.
                    //
                    let mut chunk_dsum = 0.0_f64;
                    let mut chunk_dsum_sq = 0.0_f64;
                    // Source `tsum` (`newstack.f90:119`), the last chunk's own
                    // sum, which the truncation block at `newstack.f90:2563`
                    // subtracts back out of `dsum`.
                    let mut chunk_tsum = 0.0_f64;
                    let mut dsum_chunk = Vec::<f64>::new();
                    let mut sd_chunk = Vec::<f32>::new();
                    let mut pix_chunk = Vec::<f64>::new();
                    let mut deferred = Vec::<(i32, i32, Vec<f32>)>::new();
                    let mut chunk_index = 0_usize;
                    let mut line_out_start = 0_i32;
                    while line_out_start < output_ny {
                        let (start, lines_out, line_in_start, input_lines) = layout[chunk_index];
                        chunk_index += 1;
                        line_out_start = start;
                        let line_out_end = start + lines_out - 1;
                        let needed = LinesNeededResult {
                            iy_in_1: line_in_start,
                            iy_in_2: line_in_start + input_lines - 1,
                            need_fill: section_needed.need_fill,
                            in_place: section_needed.in_place,
                        };
                        // `wallStart = wallTime()` (`newstack.f90:2340`).
                        let mut wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        if needed.iy_in_1 < load_y_start || needed.iy_in_2 > load_y_end {
                            //
                            // first load data that is needed if not already
                            // loaded (`newstack.f90:2340-2382`).  The window
                            // that is already in the buffer is shifted down or
                            // up and only the lines past it are read, which is
                            // what decides how many lines any one
                            // `readBinnedOrReduced` call is asked for -- and
                            // therefore how big a scratch array `irdReduced`
                            // needs.
                            //
                            let mut load_y_offset = needed.iy_in_1;
                            let mut load_base_ind = 0_usize;
                            let num_lines_load;
                            if load_y_start <= needed.iy_in_1 && load_y_end >= needed.iy_in_1 {
                                //
                                // move data down if it will fill a bottom region
                                //
                                let num_move =
                                    (load_y_end + 1 - needed.iy_in_1) as usize * nx_load as usize;
                                let move_offset =
                                    (needed.iy_in_1 - load_y_start) as usize * nx_load as usize;
                                // `newstack.f90:2350`.  `numMove` and
                                // `moveOffset` are `integer(kind = 8)`
                                // (`newstack.f90:54`), so they are right
                                // justified in 20.
                                if i_verbose > 0 {
                                    print!(
                                        " moving data down {:>20} {:>20}\n",
                                        num_move, move_offset
                                    );
                                }
                                for i8 in 0..num_move {
                                    input[i8] = input[i8 + move_offset];
                                }
                                num_lines_load = needed.iy_in_2 - load_y_end;
                                load_y_offset = load_y_end + 1;
                                load_base_ind = num_move;
                            } else if needed.iy_in_1 <= load_y_start
                                && needed.iy_in_2 >= load_y_start
                            {
                                //
                                // move data up if it will fill top
                                //
                                let num_move =
                                    (needed.iy_in_2 + 1 - load_y_start) as usize * nx_load as usize;
                                let move_offset =
                                    (load_y_start - needed.iy_in_1) as usize * nx_load as usize;
                                // `newstack.f90:2363`.
                                if i_verbose > 0 {
                                    print!(
                                        " moving data up {:>20} {:>20}\n",
                                        num_move, move_offset
                                    );
                                }
                                for i8 in (0..num_move).rev() {
                                    input[i8 + move_offset] = input[i8];
                                }
                                num_lines_load = load_y_start - needed.iy_in_1;
                            } else {
                                //
                                // otherwise just get whole needed region
                                //
                                num_lines_load = needed.iy_in_2 + 1 - needed.iy_in_1;
                                // `newstack.f90:2373-2374`.
                                if i_verbose > 0 {
                                    print!(
                                        " loading whole region {:>11} {:>11} {:>11}\n",
                                        needed.iy_in_1, needed.iy_in_2, num_lines_load
                                    );
                                }
                            }
                            // `newstack.f90:2822-2830` sizes the scratch buffer for
                            // `irdReduced`; 6 is the biggest support width of any filter.
                            let mut temp = vec![
                                0.0_f32;
                                // `lenTemp` as `reallocateIfNeeded` left it
                                // (`newstack.f90:2818-2842`); see the note at the
                                // scan above -- `needTemp` is one unless the
                                // reader shrinks or bins.
                                effective_len_temp.max(1) as usize
                            ];
                            // `newstack.f90:2376-2379`: the start is the reduced
                            // offset plus the reduction times the load offset.
                            if read_binned_or_reduced(
                                1,
                                in_section,
                                &mut input[load_base_ind..],
                                nx_load,
                                num_lines_load,
                                rx_offset,
                                ry_offset + read_reduction * load_y_offset as f32,
                                read_reduction,
                                nx_load,
                                num_lines_load,
                                ind_filter,
                                read_shrunk,
                                &mut temp,
                            )
                            .is_err()
                            {
                                exit_error("Reading image file");
                            }
                            load_y_start = needed.iy_in_1;
                            load_y_end = needed.iy_in_2;
                        }
                        let num_y_load = load_y_end + 1 - load_y_start;
                        // `newstack.f90:2390`.
                        load_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        // `newstack.f90:2391-2392`.
                        if need_edge_mean {
                            dmean_sec = slice_edge_median(
                                input.as_mut_ptr(),
                                bin_nx,
                                0,
                                bin_nx - 1,
                                0,
                                num_y_load - 1,
                                1,
                            ) as f32;
                        }
                        let output_lines = line_out_end - line_out_start + 1;
                        let mut output = vec![0.0_f32; output_nx as usize * output_lines as usize];
                        let matrix = [[fprod[0], fprod[1]], [fprod[2], fprod[3]]];
                        let xcen_in = bin_nx as f32 / 2.0;
                        // `wallStart = wallTime()` (`newstack.f90:2398`).
                        wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        // `newstack.f90:2400`: measured from the load window,
                        // not from what this chunk happens to need.
                        let ycen_in = bin_ny as f32 / 2.0 - load_y_start as f32;
                        let dx = fprod[4];
                        let dy = (output_ny - output_lines) as f32 / 2.0 + fprod[5]
                            - line_out_start as f32;
                        if lines_shrink > 0 {
                            // `newstack.f90:2405-2412`: post-read shrinkage
                            // replaces the affine interpolation with the
                            // antialiasing filter's own resampler.
                            let ierr = unsafe {
                                crate::imod::libcfshr::zoomdown::zoom_filt_interp(
                                    input.as_mut_ptr(),
                                    output.as_mut_ptr(),
                                    bin_nx,
                                    num_y_load,
                                    output_nx,
                                    output_lines,
                                    xcen_in,
                                    ycen_in,
                                    dx,
                                    dy,
                                    dmean_sec,
                                )
                            };
                            if ierr != 0 {
                                exit_error(&format!(
                                    "Calling zoomFiltInterp for image reduction, error{ierr:3}"
                                ));
                            }
                        } else if !has_warp && if_mag_grad == 0 {
                            cubinterp(
                                input.as_mut_ptr(),
                                output.as_mut_ptr(),
                                bin_nx,
                                num_y_load,
                                output_nx,
                                output_lines,
                                &matrix,
                                xcen_in,
                                ycen_in,
                                dx,
                                dy,
                                1.0,
                                dmean_sec,
                                if_linear,
                            );
                        } else {
                            // `newstack.f90:2417-2431`, with the grid start
                            // taken down by the first loaded input line, or by
                            // the first output line of the chunk when warping.
                            let mut ystart = if if_warping != 0 {
                                y_grid_start - line_out_start as f32
                            } else {
                                y_grid_start - load_y_start as f32
                            };
                            let mut xstart = x_grid_start;
                            if if_distort > 0 {
                                ystart -= warp_y_offsets[offset_index] / read_reduction;
                                xstart -= warp_x_offsets[offset_index] / read_reduction;
                            }
                            warp_interp(
                                input.as_mut_ptr(),
                                output.as_mut_ptr(),
                                bin_nx,
                                num_y_load,
                                output_nx,
                                output_lines,
                                &matrix,
                                xcen_in,
                                ycen_in,
                                dx,
                                dy,
                                1.0,
                                dmean_sec,
                                if_linear,
                                if_warping,
                                field_dx.as_mut_ptr(),
                                field_dy.as_mut_ptr(),
                                lm_grid,
                                nx_grid,
                                ny_grid,
                                xstart,
                                ystart,
                                x_grid_intrv,
                                y_grid_intrv,
                            );
                        }
                        // `newstack.f90:2433`.
                        rot_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        // `newstack.f90:2570`.
                        wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        // `newstack.f90:2569-2576`: with `preSetScaling` the
                        // chunk is scaled and written straight away from the
                        // input file's own range.  The `.not. preSetScaling`
                        // two-pass rescale over temporary storage
                        // (`newstack.f90:2591-2611`) is not translated on this
                        // chunked route, so a positive `-float` entry leaves
                        // the chunk unscaled here.
                        if !rescale || if_mean != 0 {
                            let (mut tmin2, mut tmax2) = (0.0_f32, 0.0_f32);
                            if if_float == 2 {
                                let (mut tsum, mut tsum_sq) = (0.0_f64, 0.0_f64);
                                let (mut avg_sec, mut sd_this) = (0.0_f32, 0.0_f32);
                                crate::imod::libcfshr::simplestat::array_min_max_mean_sd(
                                    output.as_ptr(),
                                    output_nx,
                                    output_lines,
                                    0,
                                    output_nx - 1,
                                    0,
                                    output_lines - 1,
                                    &raw mut tmin2,
                                    &raw mut tmax2,
                                    &raw mut tsum,
                                    &raw mut tsum_sq,
                                    &raw mut avg_sec,
                                    &raw mut sd_this,
                                );
                                pix_chunk.push(f64::from(output_nx) * f64::from(output_lines));
                                dsum_chunk.push(tsum);
                                sd_chunk.push(sd_this);
                                // `newstack.f90:2520-2521`.
                                if i_verbose > 0 {
                                    print!(
                                        " chunk mean&sd min/max {:>11} {} {} {} {}\n",
                                        chunk_index,
                                        list_real(avg_sec),
                                        list_real(sd_this),
                                        list_real(tmin2),
                                        list_real(tmax2)
                                    );
                                }
                                chunk_dsum_sq += tsum_sq;
                                chunk_dsum += tsum;
                                chunk_tsum = tsum;
                            } else {
                                let mut tmean2 = 0.0_f32;
                                crate::imod::libcfshr::simplestat::array_min_max_mean(
                                    output.as_ptr(),
                                    output_nx,
                                    output_lines,
                                    0,
                                    output_nx - 1,
                                    0,
                                    output_lines - 1,
                                    &raw mut tmin2,
                                    &raw mut tmax2,
                                    &raw mut tmean2,
                                );
                                // `tsum = tmean2 * numPix` keeps a real*4 result.
                                let tsum =
                                    f64::from(tmean2 * (output_nx as f32 * output_lines as f32));
                                chunk_dsum += tsum;
                                chunk_tsum = tsum;
                            }
                            tmp_min = tmp_min.min(tmin2);
                            tmp_max = tmp_max.max(tmax2);
                            // `newstack.f90:2531`.
                            if i_verbose > 0 {
                                print!(
                                    " did iclden  {} {} {} {}\n",
                                    list_real(tmin2),
                                    list_real(tmax2),
                                    list_real(tmp_min),
                                    list_real(tmp_max)
                                );
                            }
                        } else {
                            //
                            // "otherwise get new min and max quickly"
                            // (`newstack.f90:2535-2545`).
                            //
                            // `tsum` is `real*8` (`newstack.f90:119`), so
                            // the running total is accumulated in double.
                            let mut tsum = 0.0_f64;
                            for value in &output {
                                tmp_min = tmp_min.min(*value);
                                tmp_max = tmp_max.max(*value);
                                tsum += f64::from(*value);
                            }
                            chunk_dsum += tsum;
                            chunk_tsum = tsum;
                        }
                        //
                        // 6/27/01: really want to truncate rather than
                        // rescale; so if the min or max is now out of range
                        // for the input mode, truncate the data and adjust the
                        // min and max (`newstack.f90:2545-2563`).
                        //
                        if if_float == 0
                            && output_mode != 2
                            && header.mode != 2
                            && (tmp_min < bottom_in || tmp_max > optimal_in)
                        {
                            let mut tsum2 = 0.0_f64;
                            for value in output.iter_mut() {
                                if *value < bottom_in {
                                    num_trunc_low += 1;
                                }
                                if *value > optimal_in {
                                    num_trunc_high += 1;
                                }
                                *value = bottom_in.max(optimal_in.min(*value));
                                tsum2 += f64::from(*value);
                            }
                            tmp_min = tmp_min.max(bottom_in);
                            tmp_max = tmp_max.min(optimal_in);
                            chunk_dsum = chunk_dsum + tsum2 - chunk_tsum;
                        }
                        if pre_set_scaling {
                            // `newstack.f90:2570-2576`: the pre-set factors do
                            // not depend on this chunk's min and max, so the
                            // chunk is scaled and written straight away.
                            let (save_min, save_max) = (tmp_min, tmp_max);
                            let factors =
                                find_scale_factors(&chunk_scale_values, header.amin, header.amax);
                            // `newstack.f90:2571-2574` saves and restores
                            // `tmpMin`/`tmpMax` around this call because
                            // `findScaleFactors` modifies them.
                            tmp_min = save_min;
                            tmp_max = save_max;
                            // `scaleAndWriteChunk` accumulates `dmin2`,
                            // `dmax2` and `dmean2` across every chunk of the
                            // section, so the state is carried here too.
                            chunk_scaling.scale_factor = factors.scale_factor;
                            chunk_scaling.const_add = factors.const_add;
                            if scale_and_write_chunk(
                                &mut output,
                                output_nx,
                                &mut chunk_scaling,
                                |_| Ok(()),
                            )
                            .is_err()
                            {
                                exit_error("Scaling output image");
                            }
                            // `scaleAndWriteChunk` does its own write
                            // (`newstack.f90:3255-3257`); here the write stayed
                            // with the caller, so the report sits in front of
                            // it.
                            if i_verbose > 0 {
                                print!(" writing {:>11}\n", chunk_index);
                            }
                            iiu_set_position(2, out_section as i32, line_out_start);
                            if iiu_write_lines(2, output.as_mut_ptr().cast(), output_lines) != 0 {
                                exit_error("Writing image file");
                            }
                        } else if chunk_index != layout.len() && if_out_chunk > 0 {
                            // `newstack.f90:2585-2588`: every chunk but the
                            // last goes out to the scratch file, and the
                            // buffer is reused for the next one.
                            // `newstack.f90:2586`.
                            if i_verbose > 0 {
                                print!(" writing to temp file {:>11}\n", chunk_index);
                            }
                            iiu_set_position(3, 0, line_out_start);
                            if iiu_write_lines(3, output.as_mut_ptr().cast(), output_lines) != 0 {
                                exit_error("Writing image file");
                            }
                            deferred.push((line_out_start, output_lines, Vec::new()));
                        } else {
                            // `newstack.f90:2580-2588`: without pre-set
                            // scaling the factors are not known until every
                            // chunk has been through, so the chunk waits for
                            // the second pass below.  With `ifOutChunk` zero
                            // the whole output is in the array at once, and
                            // the last chunk is always still in the buffer.
                            deferred.push((line_out_start, output_lines, output));
                        }
                        // `newstack.f90:2590`.
                        save_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        line_out_start = line_out_end + 1;
                    }
                    if !pre_set_scaling {
                        //
                        // `newstack.f90:2592-2611`: find the factors from the
                        // section's own min and max, then loop **backwards**
                        // over the chunks scaling and writing.  The direction
                        // is not cosmetic: `dmin2`, `dmax2` and `dmean2`
                        // accumulate in that order.
                        //
                        // `findScaleFactors` reads the accumulated sums for
                        // `-float 2`, `-float 3` and `-meansd`.
                        let (float_average, float_sd) = if dsum_chunk.is_empty() {
                            (0.0, 0.0)
                        } else {
                            chunk_sums_to_avgsd(
                                &dsum_chunk,
                                &sd_chunk,
                                &pix_chunk,
                                output_nx,
                                output_ny,
                            )
                        };
                        let section = route_index - 1;
                        chunk_scale_values.dsum = chunk_dsum;
                        chunk_scale_values.dsum_sq = chunk_dsum_sq;
                        chunk_scale_values.float_average = float_average;
                        chunk_scale_values.float_sd = float_sd;
                        chunk_scale_values.section_min =
                            sec_mins.get(section).copied().unwrap_or(0.0);
                        chunk_scale_values.section_max =
                            sec_maxes.get(section).copied().unwrap_or(0.0);
                        // The negated group of `newstack.f90:3057-3059`.
                        chunk_scale_values.section_intentionally_truncated = output_mode == 2
                            || (num_sec_trunc > 0
                                && (z_min_outlier.get(section).copied().unwrap_or(0.0) < 0.0
                                    || z_max_outlier.get(section).copied().unwrap_or(0.0) > 0.0));
                        let factors = find_scale_factors(&chunk_scale_values, tmp_min, tmp_max);
                        // `newstack.f90:2594` keeps what `findScaleFactors`
                        // does to `tmpMin` and `tmpMax` -- unlike the
                        // `preSetScaling` call above, which saves and restores
                        // them -- and the section report at
                        // `newstack.f90:2626` prints the truncated values.
                        tmp_min = factors.tmp_min;
                        tmp_max = factors.tmp_max;
                        chunk_scaling.scale_factor = factors.scale_factor;
                        chunk_scaling.const_add = factors.const_add;
                        chunk_scaling.dmean2 = 0.;
                        let num_chunks = deferred.len();
                        for (index, (start, lines, chunk)) in deferred.iter_mut().enumerate().rev()
                        {
                            // `newstack.f90:2605-2610`.
                            if index + 1 != num_chunks && if_out_chunk > 0 {
                                // `newstack.f90:2607`.
                                if i_verbose > 0 {
                                    print!(" reading {:>11}\n", index + 1);
                                }
                                chunk.resize(output_nx as usize * *lines as usize, 0.0);
                                iiu_set_position(3, 0, *start);
                                if irdsecl(3, chunk, *lines).is_err() {
                                    exit_error("Reading temporary file");
                                }
                            }
                            if scale_and_write_chunk(chunk, output_nx, &mut chunk_scaling, |_| {
                                Ok(())
                            })
                            .is_err()
                            {
                                exit_error("Scaling output image");
                            }
                            // `wallStart = wallTime()` (`newstack.f90:3254`).
                            let wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                            // `newstack.f90:3255`.
                            if i_verbose > 0 {
                                print!(" writing {:>11}\n", index + 1);
                            }
                            iiu_set_position(2, out_section as i32, *start);
                            if iiu_write_lines(2, chunk.as_mut_ptr().cast(), *lines) != 0 {
                                exit_error("Writing image file");
                            }
                            // `newstack.f90:3258`.
                            save_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        }
                    }
                    num_trunc_low += chunk_scaling.num_trunc_low;
                    num_trunc_high += chunk_scaling.num_trunc_high;
                    // `newstack.f90:2622-2628`.
                    let dmean2 = chunk_scaling.dmean2 / (output_nx as f32 * output_ny as f32);
                    if !quiet {
                        if if_header_out == 0 {
                            println!(" section   input min&max       output min&max  &  mean");
                        }
                        if_header_out = 1;
                        let (chunk_dmin2, chunk_dmax2) = (chunk_scaling.dmin2, chunk_scaling.dmax2);
                        // Fortran `f10.2` fills the field with asterisks when the
                        // value does not fit in ten characters (`newstack.f90:2627`
                        // is `write(*,'(i8,5f10.2)')`); Rust's `{:10.2}` widens the
                        // field instead and pushes the rest of the line right.
                        println!(
                            "{:8}{}{}{}{}{}",
                            route_index - 1,
                            if format!("{tmp_min:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{tmp_min:10.2}")
                            },
                            if format!("{tmp_max:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{tmp_max:10.2}")
                            },
                            if format!("{chunk_dmin2:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{chunk_dmin2:10.2}")
                            },
                            if format!("{chunk_dmax2:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{chunk_dmax2:10.2}")
                            },
                            if format!("{dmean2:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{dmean2:10.2}")
                            }
                        );
                    }
                    // `newstack.f90:2630-2635`.
                    dmin = dmin.min(chunk_scaling.dmin2);
                    dmax = dmax.max(chunk_scaling.dmax2);
                    // The source has one section loop, so its label 80 tail
                    // (`newstack.f90:2630-2683`) runs for a chunked section
                    // exactly as it does for a whole-section one.  This route
                    // leaves the loop here, so the tail is carried with it.
                    if list_replace.is_empty() {
                        dmean += dmean2;
                        //
                        // transfer extra header bytes if present
                        // (`newstack.f90:2636-2683`)
                        //
                        if n_byte_sym_out != 0 && ind_extra_out < n_byte_sym_out {
                            //
                            // get this section's size and offset, make sure its OK;
                            // only FEI gives error
                            let mut move_offset = 0_i32;
                            let mut n_byte_extra_in = 0_i32;
                            if get_extra_header_sec_offset(
                                extra_in.as_mut_ptr().cast(),
                                n_byte_sym_in,
                                num_int_or_bytes_in,
                                i_flag_extra_in,
                                in_section,
                                &raw mut move_offset,
                                &raw mut n_byte_extra_in,
                            ) != 0
                            {
                                exit_error(
                                    "FEI1 extended header does not contain data for all sections being read",
                                );
                            }
                            if fei1_type {
                                //
                                // FEI type, call the copy function
                                if copy_extra_header_section(
                                    extra_in.as_mut_ptr().cast(),
                                    n_byte_sym_in,
                                    extra_out.as_mut_ptr().cast(),
                                    n_byte_sym_out,
                                    num_int_or_bytes_in,
                                    i_flag_extra_in,
                                    in_section,
                                    &raw mut ind_extra_out,
                                ) != 0
                                {
                                    exit_error(
                                        "Space allowed for copying FEI1 extra header not big enough; this is due to either program error or mixing files with different sizes per section",
                                    );
                                }
                            } else {
                                let mut n_byte_copy =
                                    n_byte_extra_out.min(n_byte_extra_in).min(n_byte_sym_in);
                                let mut num_for_tilt = 0_i32;
                                if save_tilts {
                                    //
                                    // To save tilt angles, put angle in the integer or
                                    // real then copy the right number of bytes; adjust
                                    // the number to copy and number to clear.
                                    // `btiltTemp` is `equivalence`d onto both
                                    // (`newstack.f90:46-49`).
                                    let mut btilt_temp = [0_u8; 4];
                                    if serial_em_type {
                                        let itilt_temp =
                                            ((100. * extra_tilts[route_index - 1]).round() as i32)
                                                as i16;
                                        btilt_temp[..2].copy_from_slice(&itilt_temp.to_ne_bytes());
                                        num_for_tilt = 2;
                                    } else {
                                        btilt_temp.copy_from_slice(
                                            &extra_tilts[route_index - 1].to_ne_bytes(),
                                        );
                                        num_for_tilt = 4;
                                    }
                                    for i in 0..num_for_tilt as usize {
                                        ind_extra_out += 1;
                                        extra_out[ind_extra_out as usize - 1] = btilt_temp[i];
                                    }
                                    n_byte_copy = n_byte_copy.max(num_for_tilt) - num_for_tilt;
                                }
                                let n_byte_clear = n_byte_extra_out - num_for_tilt - n_byte_copy;
                                //
                                // Copy bytes, then clear out the rest if any
                                for i in 1..=n_byte_copy {
                                    ind_extra_out += 1;
                                    extra_out[ind_extra_out as usize - 1] =
                                        extra_in[(in_section * n_byte_extra_in + i + num_for_tilt)
                                            as usize
                                            - 1];
                                }
                                for _ in 0..n_byte_clear {
                                    ind_extra_out += 1;
                                    extra_out[ind_extra_out as usize - 1] = 0;
                                }
                            }
                        }
                    }
                    continue;
                }
                // The source reads every section into the one `array`
                // allocation that `reallocateIfNeeded` (`newstack.f90:2798`)
                // sizes to `idimInOut`, so the read buffer always holds a
                // full input section no matter what binning, transforming, or
                // repacking did with the previous section's data.  Restore
                // that invariant here, because this stream path replaces
                // `array` with the reduced-size result of each stage.
                if array.len() < header.nx as usize * header.ny as usize {
                    array.resize(header.nx as usize * header.ny as usize, 0.0);
                }
                // Source per-section accumulators (`newstack.f90:2327-2333`).
                let mut tmp_min = 1.0e30_f32;
                let mut tmp_max = -1.0e30_f32;
                let mut dmin2 = 1.0e30_f32;
                let mut dmax2 = -1.0e30_f32;
                let mut dmean2 = 0.0_f32;
                if blank_section {
                    //
                    // Handle blank images here and skip out
                    // (`newstack.f90:2024-2044`).
                    //
                    tmp_min = header.amean;
                    if let Some(fill) = fill_value {
                        tmp_min = fill;
                    }
                    tmp_max = tmp_min;
                    // `nyNeeded = 1` then `call reallocateIfNeeded()`
                    // (`newstack.f90:2031-2032`).  `processInPlace` is
                    // whatever the last section left it at -- the blank branch
                    // never recomputes it -- so the size it reports carries
                    // over too.  Only the reports are consumed here.
                    if i_verbose > 0 {
                        let mut allocation = ReallocateIfNeeded {
                            physical_memory,
                            process_in_place,
                            ft_reduce_fac,
                            phase_shift,
                            lim_entered,
                            nx: header.nx,
                            ny: header.ny,
                            nx_bin: bin_nx,
                            ny_bin: bin_ny,
                            ny_needed: 1,
                            nx_out: output_nx,
                            ny_out: output_ny,
                            read_shrunk,
                            read_reduction,
                            i_binning: bin_factor,
                            fourier_scaling,
                            nx_fspad,
                            ny_fspad,
                            nx_fcrop_pad,
                            ny_fcrop_pad,
                            ft_expand_fac,
                            noise_pad,
                            nx_bin_fft: bin_nx,
                            ny_bin_fft: bin_ny,
                            lim_to_alloc: alloc_lim_to_alloc,
                            len_temp: alloc_len_temp,
                            pre_set_scaling,
                            idim_in_out: alloc_idim_in_out,
                            in_place_fac,
                            i_verbose,
                        };
                        reallocate_if_needed(&mut allocation);
                        process_in_place = allocation.process_in_place;
                        in_place_fac = allocation.in_place_fac;
                        alloc_lim_to_alloc = allocation.lim_to_alloc;
                        alloc_len_temp = allocation.len_temp;
                        alloc_idim_in_out = allocation.idim_in_out;
                    }
                    let scaling_values = FindScaleFactors {
                        if_float,
                        rescale,
                        num_scale_facs: scale_factors.len() as i32,
                        bottom_in,
                        bottom_out,
                        optimal_in,
                        optimal_out,
                        dmin_specified: scale_limits[0],
                        dmax_specified: scale_limits[1],
                        dmin_in: header.amin,
                        dmax_in: header.amax,
                        if_map_range: i32::from(map_entered),
                        dmap_low: map_limits[0],
                        dmap_high: map_limits[1],
                        frac_zero,
                        if_mean,
                        if_mean_sd_entered: i32::from(mean_sd_entered),
                        entered_mean: mean_sd[0],
                        entered_sd: mean_sd[1],
                        shift_mean,
                        shift_min,
                        shift_max,
                        new_mode: output_mode,
                        // `dsumSq = 0.` and
                        // `dsum = tmpMin * (float(nxOut) * nyOut)`.
                        dsum: f64::from(tmp_min * (output_nx as f32 * output_ny as f32)),
                        dsum_sq: 0.0,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        scale_factor: 1.0,
                        const_add: 0.0,
                        scale_fac: scale_factors
                            .get(input_index.min(scale_factors.len().saturating_sub(1)))
                            .map_or(1.0, |entry| entry[0]),
                        scale_const: scale_factors
                            .get(input_index.min(scale_factors.len().saturating_sub(1)))
                            .map_or(0.0, |entry| entry[1]),
                        float_average: tmp_min,
                        float_sd: 0.0,
                        zmin: float2_zmin,
                        zmax: float2_zmax,
                        float_z_margin,
                        opt_float_range,
                        opt_float_min,
                        section_min: tmp_min,
                        section_max: tmp_max,
                        section_intentionally_truncated: true,
                    };
                    let factors = find_scale_factors(&scaling_values, tmp_min, tmp_max);
                    array = vec![
                        tmp_min * factors.scale_factor + factors.const_add;
                        output_nx as usize * output_ny as usize
                    ];
                } else {
                    if active_input_file.is_null() {
                        exit_error("End of image while reading");
                    }
                    // The input is loaded inside the chunk loop below, one load
                    // window at a time (`newstack.f90:2340-2384`), so nothing
                    // is read here.
                    let offset_index = route_index - 1;
                    let nx_bin_sec = bin_nx;
                    let ny_bin_sec = bin_ny;
                    //
                    // if doing distortions or warping, get the grid
                    // (`newstack.f90:2055-2110`)
                    //
                    let mut has_warp = false;
                    let (mut nx_grid, mut ny_grid) = (0_i32, 0_i32);
                    let (mut x_grid_start, mut y_grid_start) = (0.0_f32, 0.0_f32);
                    let (mut x_grid_intrv, mut y_grid_intrv) = (0.0_f32, 0.0_f32);
                    let (mut grid_iy, mut grid_dx, mut grid_dy) = (0_i32, 0.0_f32, 0.0_f32);
                    let (mut xn_big, mut yn_big) = (0.0_f32, 0.0_f32);
                    if if_distort > 0 {
                        grid_iy = idf_use[offset_index] + 1;
                        has_warp = true;
                        xn_big = header.nx as f32 / warp_scale;
                        yn_big = header.ny as f32 / warp_scale;
                    } else if if_warping != 0 {
                        grid_iy = transform_lines[offset_index] + 1;
                        has_warp = n_control[grid_iy as usize - 1] > 2;
                        xn_big = read_reduction * output_nx as f32 / warp_scale;
                        yn_big = read_reduction * output_ny as f32 / warp_scale;
                        //
                        // for warping with center offset applied after, it
                        // will subtract the offset from the grid start and add
                        // it to the grid displacements
                        //
                        if apply_first == 0 {
                            grid_dx = read_reduction * x_offsets[offset_index] / warp_scale;
                            grid_dy = read_reduction * y_offsets[offset_index] / warp_scale;
                        }
                    }
                    if has_warp {
                        let mut list_string = vec![0_i8; 1024];
                        if get_size_adjusted_grid(
                            // `getsizeadjustedgrid` (`warpwrapfort.c:326`)
                            // passes `*iz - 1`, so the source's 1-based index
                            // reaches the C entry point zero-based.
                            grid_iy - 1,
                            xn_big,
                            yn_big,
                            grid_dx,
                            grid_dy,
                            1,
                            warp_scale,
                            read_reduction.round() as i32,
                            &raw mut nx_grid,
                            &raw mut ny_grid,
                            &raw mut x_grid_start,
                            &raw mut y_grid_start,
                            &raw mut x_grid_intrv,
                            &raw mut y_grid_intrv,
                            field_dx.as_mut_ptr(),
                            field_dy.as_mut_ptr(),
                            lm_grid,
                            lm_grid,
                            list_string.as_mut_ptr(),
                            list_string.len() as i32,
                        ) != 0
                        {
                            let message = std::ffi::CStr::from_ptr(list_string.as_ptr())
                                .to_string_lossy()
                                .into_owned();
                            exit_error(message.trim_end());
                        }
                        // copy field to tmpDx, y in case there are mag grads
                        for iy in 0..ny_grid as usize {
                            for ix in 0..nx_grid as usize {
                                let index = ix + iy * lm_grid as usize;
                                tmp_dx[index] = field_dx[index];
                                tmp_dy[index] = field_dy[index];
                            }
                        }
                    }
                    //
                    // if doing mag gradients, set up or add to distortion field
                    // (`newstack.f90:2094-2110`)
                    //
                    if if_mag_grad != 0 {
                        let mag_use = (in_section + 1).min(num_mag_grad).max(1) as usize;
                        if if_distort != 0 {
                            add_mag_grad_field(
                                tmp_dx.as_mut_ptr(),
                                tmp_dy.as_mut_ptr(),
                                field_dx.as_mut_ptr(),
                                field_dy.as_mut_ptr(),
                                lm_grid,
                                nx_bin_sec,
                                ny_bin_sec,
                                nx_grid,
                                ny_grid,
                                x_grid_start,
                                y_grid_start,
                                x_grid_intrv,
                                y_grid_intrv,
                                nx_bin_sec as f32 / 2.,
                                ny_bin_sec as f32 / 2.,
                                pixel_mag_grad,
                                axis_rot,
                                tilt_angles[mag_use - 1],
                                dmag_per_micron[mag_use - 1],
                                rot_per_micron[mag_use - 1],
                            );
                        } else {
                            make_mag_grad_field(
                                tmp_dx.as_mut_ptr(),
                                tmp_dy.as_mut_ptr(),
                                field_dx.as_mut_ptr(),
                                field_dy.as_mut_ptr(),
                                lm_grid,
                                nx_bin_sec,
                                ny_bin_sec,
                                &raw mut nx_grid,
                                &raw mut ny_grid,
                                &raw mut x_grid_start,
                                &raw mut y_grid_start,
                                &raw mut x_grid_intrv,
                                &raw mut y_grid_intrv,
                                nx_bin_sec as f32 / 2.,
                                ny_bin_sec as f32 / 2.,
                                pixel_mag_grad,
                                axis_rot,
                                tilt_angles[mag_use - 1],
                                dmag_per_micron[mag_use - 1],
                                rot_per_micron[mag_use - 1],
                            );
                        }
                    }
                    //
                    // get maximum Y deviation with current field to adjust
                    // chunk limits with (`newstack.f90:2128-2141`)
                    //
                    let (mut max_field_x, mut max_field_y) = (0_i32, 0_i32);
                    if if_mag_grad != 0 || has_warp {
                        let (mut field_max_x, mut field_max_y) = (0.0_f32, 0.0_f32);
                        for iy in 0..ny_grid as usize {
                            for ix in 0..nx_grid as usize {
                                let index = ix + iy * lm_grid as usize;
                                field_max_x = field_max_x.max(field_dx[index].abs());
                                field_max_y = field_max_y.max(field_dy[index].abs());
                            }
                        }
                        max_field_x = (field_max_x as f64 + 1.5) as i32;
                        max_field_y = (field_max_y as f64 + 1.5) as i32;
                    }
                    //
                    // Get the index of the transform (`newstack.f90:2048-2053`)
                    // and fold the offsets into it (`newstack.f90:2112-2124`).
                    //
                    let mut fprod = [0.0_f32; 6];
                    if !transforms.is_empty() {
                        let Some(transform) =
                            transforms.get(transform_lines[offset_index] as usize)
                        else {
                            exit_error(&format!(
                                "TRANSFORM LINE number out of bounds:{in_section:5}"
                            ));
                        };
                        fprod = *transform;
                        if apply_first != 0 {
                            let frot = [
                                1.0,
                                0.0,
                                0.0,
                                1.0,
                                -x_offsets[offset_index],
                                -y_offsets[offset_index],
                            ];
                            let mut composed = [0.0_f32; 6];
                            xfmult(&frot, &fprod, &mut composed);
                            fprod = composed;
                        } else if !(if_warping != 0 && has_warp) {
                            // `newstack.f90:2122`: when a warping grid carries
                            // the offset, it is not folded into the transform.
                            fprod[4] -= x_offsets[offset_index];
                            fprod[5] -= y_offsets[offset_index];
                        }
                    }
                    //
                    // Determine starting and ending lines needed from the input,
                    // and whether any fill is needed (`newstack.f90:2144-2146`).
                    //
                    let section_affine = LinesNeededForOutput {
                        fourier_scaling,
                        if_xform: i32::from(!transforms.is_empty()),
                        nx_bin: nx_bin_sec,
                        ny_bin: ny_bin_sec,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        xcen: x_offsets[offset_index],
                        ycen: y_offsets[offset_index],
                        // Fortran `fprod(i, j)`, as `backXform` reads it.
                        fprod: [
                            [fprod[0], fprod[2], fprod[4]],
                            [fprod[1], fprod[3], fprod[5]],
                        ],
                        max_field_x,
                        max_field_y,
                        lines_shrink,
                    };
                    let needed_for_output =
                        lines_needed_for_output(&section_affine, 0, output_ny - 1);
                    let num_y_load = needed_for_output.iy_in_2 - needed_for_output.iy_in_1 + 1;
                    //
                    // Get padded input size for phase shifting
                    // (`newstack.f90:2149-2157`).  `maxFSpad = 100`,
                    // `minFSpad = 10` and `fsPadFrac = 0.1`
                    // (`newstack.f90:256-258`).  This sits ahead of
                    // `nxDimNeed`/`nyDimNeed` because they are `max(nxBin,
                    // nxFSpad + 2)` and `max(nyNeeded, nyFSpad + 1)`, and
                    // ahead of `reallocateIfNeeded`, whose `needDim` for a
                    // phase shift is `(nxFSpad + 2) * (nyFSpad + 1)`.
                    //
                    if phase_shift {
                        nx_fspad = nice_frame(
                            100.min(10.max((0.1 * nx_bin_sec as f32).round() as i32)) + nx_bin_sec,
                            2,
                            nice_fft_limit(),
                        );
                        ny_fspad = nice_frame(
                            100.min(10.max((0.1 * num_y_load as f32).round() as i32)) + num_y_load,
                            2,
                            nice_fft_limit(),
                        );
                        nx_fcrop_pad = nx_fspad;
                        ny_fcrop_pad = ny_fspad;
                    }
                    // `newstack.f90:2158-2281`.  `nxDimNeed` and `nyDimNeed` are the
                    // row and line counts the input plus any FFT padding needs,
                    // `processInPlace` says the output can be built back into the
                    // input space, `linesLeft`/`numChunks` follow from the room
                    // `reallocateIfNeeded` leaves in the flat `array`, and the chunk
                    // table follows from those.  `maxin` and `ifOutChunk` size the
                    // load buffer and place `iBufOutBase` for the chunk loop below.
                    // `newstack.f90:2158-2281` runs on every route, not just a
                    // verbose one: the layout search ends in
                    // `exitError(' Input image too large for array.')`
                    // (`newstack.f90:2274`) when no split of the output fits, so
                    // gating the whole block on `iVerbose` lost that exit.  Only
                    // the reports inside it are `iVerbose`-gated in the source.
                    let ny_needed = num_y_load;
                    let nx_dim_need = bin_nx.max(nx_fspad + 2);
                    let ny_dim_need = ny_needed.max(ny_fspad + 1);
                    process_in_place = transforms.is_empty()
                        && output_nx <= nx_dim_need
                        && output_ny <= ny_dim_need
                        && needed_for_output.in_place;
                    let mut allocation = ReallocateIfNeeded {
                        physical_memory,
                        process_in_place,
                        ft_reduce_fac,
                        phase_shift,
                        lim_entered,
                        nx: header.nx,
                        ny: header.ny,
                        nx_bin: bin_nx,
                        ny_bin: bin_ny,
                        ny_needed,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        read_shrunk,
                        read_reduction,
                        i_binning: bin_factor,
                        fourier_scaling,
                        nx_fspad,
                        ny_fspad,
                        nx_fcrop_pad,
                        ny_fcrop_pad,
                        ft_expand_fac,
                        noise_pad,
                        nx_bin_fft: bin_nx,
                        ny_bin_fft: bin_ny,
                        lim_to_alloc: alloc_lim_to_alloc,
                        len_temp: alloc_len_temp,
                        pre_set_scaling,
                        idim_in_out: alloc_idim_in_out,
                        in_place_fac,
                        i_verbose,
                    };
                    // `call reallocateIfNeeded()` (`newstack.f90:2181`), which
                    // makes the `newstack.f90:2815` and `newstack.f90:2865`
                    // reports and may make `newstack.f90:2887`.
                    reallocate_if_needed(&mut allocation);
                    process_in_place = allocation.process_in_place;
                    in_place_fac = allocation.in_place_fac;
                    alloc_lim_to_alloc = allocation.lim_to_alloc;
                    alloc_len_temp = allocation.len_temp;
                    alloc_idim_in_out = allocation.idim_in_out;
                    let idim_in_out = alloc_idim_in_out as i64;
                    // `newstack.f90:2182-2183`.
                    if idim_in_out / i64::from(nx_dim_need) <= i64::from(ny_dim_need)
                        && !pre_set_scaling
                    {
                        process_in_place = false;
                    }
                    // `newstack.f90:2184-2185`.
                    if i_verbose > 0 {
                        print!(
                            " preSetScaling  {}    processInPlace  {}\n",
                            if pre_set_scaling { "T" } else { "F" },
                            if process_in_place { "T" } else { "F" }
                        );
                    }
                    // `newstack.f90:2189-2198`.  `linesLeft` and `numChunks` are
                    // `integer*4` (`newstack.f90:95`), so the `integer(kind = 8)`
                    // quotient lands in 32 bits before it is edited.
                    let mut num_chunks = 0_i32;
                    if idim_in_out / i64::from(nx_dim_need) > i64::from(ny_dim_need) {
                        let mut lines_left =
                            ((idim_in_out - i64::from(nx_dim_need) * i64::from(ny_dim_need))
                                / i64::from(output_nx)) as i32;
                        if process_in_place {
                            lines_left = (idim_in_out / i64::from(nx_dim_need)) as i32;
                        }
                        // A zero `linesLeft` is an integer divide by zero in the
                        // source, which has no behaviour to reproduce.
                        if lines_left != 0 {
                            num_chunks = (output_ny + lines_left - 1) / lines_left;
                        }
                        if i_verbose > 0 {
                            print!(
                                " linesleft {:>11}   nchunk {:>11}\n",
                                lines_left, num_chunks
                            );
                        }
                    }
                    // `newstack.f90:2200-2207`: a multi-chunk section cannot be
                    // tapered, no Fourier operation can run on one, and a
                    // process-in-place run has to fit in `maxChunks`.  All three
                    // test the source's own `numChunks`, which is zero -- not one
                    // -- when the input does not fit, so none of them fires in
                    // that case and the search below takes over.
                    if num_chunks > 1 && num_taper > 0 {
                        exit_error(
                            "Cannot taper output image - it does not fit completely in memory",
                        );
                    }
                    if num_chunks > 1 && (phase_shift || fourier_scaling) {
                        exit_error(
                            "Cannot apply Fourier operations - input and output images do not fit completely in memory",
                        );
                    }
                    if num_chunks > 250 && process_in_place {
                        exit_error(
                            "The images are too large to process with the current memory limit",
                        );
                    }
                    // `newstack.f90:2209-2274`: either the whole input for every
                    // output chunk, or a search over successively more chunks for
                    // one whose input and output both fit.
                    let mut layout = Vec::<(i32, i32, i32, i32)>::new();
                    let mut if_out_chunk = -1_i32;
                    let mut max_in = 0_i32;
                    let split = |count: i32, index: i32| {
                        (output_ny / count) * index + index.min(output_ny % count)
                    };
                    if num_chunks == 1 || (num_chunks > 0 && num_chunks <= 250 && pre_set_scaling) {
                        let mut start = 0_i32;
                        for index in 1..=num_chunks {
                            let next = split(num_chunks, index);
                            if process_in_place && !phase_shift && !fourier_scaling {
                                layout.push((
                                    start,
                                    next - start,
                                    needed_for_output.iy_in_1 + start,
                                    next - start,
                                ));
                            } else {
                                layout.push((
                                    start,
                                    next - start,
                                    needed_for_output.iy_in_1,
                                    ny_needed,
                                ));
                            }
                            start = next;
                        }
                        max_in = ny_dim_need;
                        if_out_chunk = 1;
                    } else {
                        let mut found = false;
                        'scans: for scan in 1..=2 {
                            let mut count = 1_i32;
                            while count <= 250 {
                                layout.clear();
                                let mut start = 0_i32;
                                max_in = 0;
                                for index in 1..=count {
                                    let next = split(count, index);
                                    let needed =
                                        lines_needed_for_output(&section_affine, start, next - 1);
                                    let lines_in = needed.iy_in_2 + 1 - needed.iy_in_1;
                                    layout.push((start, next - start, needed.iy_in_1, lines_in));
                                    max_in = max_in.max(lines_in);
                                    start = next;
                                }
                                let iy_test = if scan == 2 { layout[0].1 } else { output_ny };
                                // `newstack.f90:2264-2266` tests the fit with
                                // **`nxBin`**, not `nxDimNeed`; the two differ
                                // whenever a Fourier pad widens `nxDimNeed`, and
                                // using the wider one here picks a different
                                // chunk count from the source's.
                                if max_in > 0
                                    && idim_in_out / i64::from(max_in) > i64::from(bin_nx)
                                    && idim_in_out / i64::from(iy_test.max(1))
                                        > i64::from(output_nx)
                                    && i64::from(max_in) * i64::from(bin_nx)
                                        + i64::from(iy_test) * i64::from(output_nx)
                                        <= idim_in_out
                                {
                                    // `newstack.f90:2270`.
                                    if_out_chunk = scan - 1;
                                    found = true;
                                    break 'scans;
                                }
                                count += 1;
                            }
                        }
                        if !found {
                            exit_error(" Input image too large for array.");
                        }
                    }
                    // Below the limit that trips the exits above, the source
                    // does not taper or Fourier-transform correctly -- it
                    // *corrupts memory*.  Its `numChunks` was zero when those
                    // were tested, so they passed; the search just above then
                    // splits the output anyway.  `newstack.f90:2504` tapers
                    // `nxOut * nyOut` elements of a buffer holding only
                    // `numLinesOut(iChunk)` lines, and `newstack.f90:2443-2450`
                    // pads `(nxFSpad + 2) * nyFSpad` elements plus a
                    // `(nxFSpad + 2)`-element working row into an `idimInOut`
                    // that `numChunks == 0` has already proved smaller than
                    // `nxDimNeed * (nyDimNeed + 1)`.  The reference reads and
                    // writes past its array -- `-phase -test 1000,1` segfaults
                    // (rc 139), `-size ... -test 1000,1` with a fill prints
                    // `NaN` from uninitialised memory -- so there is nothing
                    // here to be faithful to.  Both are refusals by name, as
                    // the chunked-affine route already refuses the taper.
                    if layout.len() > 1 && num_taper > 0 {
                        exit_error(
                            "-taper with -memory or -test is not supported by this translation",
                        );
                    }
                    if layout.len() > 1 && (phase_shift || fourier_scaling) {
                        exit_error(
                            "Fourier operations with -memory or -test are not supported by this translation",
                        );
                    }
                    // `newstack.f90:2276-2281`.
                    if i_verbose > 0 {
                        print!(
                            " number of chunks: {:>11} {:>11}\n",
                            layout.len(),
                            if_out_chunk
                        );
                        for (index, &(line_out_st, num_lines_out, line_in_st, num_lines_in)) in
                            layout.iter().enumerate()
                        {
                            print!(
                                " {:>11} {:>11} {:>11} {:>11} {:>11}\n",
                                index + 1,
                                line_in_st,
                                num_lines_in,
                                line_out_st,
                                num_lines_out
                            );
                        }
                    }
                    //
                    // open temp file if one is needed
                    // (`newstack.f90:2284-2296`).  It is opened at most once
                    // for the whole run, and its extension carries the digits
                    // of the run's time string.
                    //
                    if rescale
                        && !pre_set_scaling
                        && if_out_chunk > 0
                        && layout.len() > 1
                        && if_temp_open == 0
                    {
                        let mut temp_ext = *b"nws      ";
                        temp_ext[3..5].copy_from_slice(&time_str[0..2]);
                        temp_ext[5..7].copy_from_slice(&time_str[3..5]);
                        temp_ext[7..9].copy_from_slice(&time_str[6..8]);
                        let temp_name = temp_filename(
                            &output_names[output_index],
                            " ",
                            std::str::from_utf8(&temp_ext).unwrap_or_default(),
                        );
                        imopen(3, &temp_name, "scratch");
                        let mut nxyz3 = [output_nx, output_ny, num_output_sections[output_index]];
                        iiu_create_header(
                            3,
                            nxyz3.as_mut_ptr(),
                            nxyz3.as_mut_ptr(),
                            2,
                            title.as_mut_ptr().cast(),
                            0,
                        );
                        if_temp_open = 1;
                    }
                    // `newstack.f90:2298`.
                    let buf_out_base = i64::from(max_in) * i64::from(nx_dim_need) + 1;
                    //
                    // The source reads the input into `array(1 ...)`, whose
                    // room for the load is `maxin` lines of `nxDimNeed`
                    // (`newstack.f90:2298`), and it keeps that one buffer
                    // across the chunk loop so the move-down and move-up
                    // shortcuts have somewhere to move data to.  This
                    // translation splits the source's one flat `array` into
                    // that input buffer and the output space at
                    // `iBufOutBase`, which is `array` itself below.
                    //
                    let nx_load = bin_nx;
                    //
                    // The buffer also has to hold a whole input section.  With
                    // no transform `linesNeededForOutput` clamps both ends to
                    // the input (`newstack.f90:3322-3323`), so an output chunk
                    // that lies entirely in the padding above or below the
                    // image comes back with `iy2 < iy1` and
                    // `numLinesIn(iChunk)` *negative*.  The source then calls
                    // `readBinnedOrReduced` with that negative line count
                    // (`newstack.f90:2376-2379`), and `iiMRCsetLoadInfo`
                    // replaces the resulting negative `ury` with `ny - 1`
                    // (`iimrc.c:169-172`) -- so the read that was asked for no
                    // lines fetches the *whole* section, past the end of the
                    // source's own `array` when `idimInOut` is smaller than
                    // that.  The repack still sees `my <= 0` and fills the
                    // chunk, so the data read is never used; only the room it
                    // needs is real, and reserving it here is what keeps the
                    // source's own overrun from being one here.
                    //
                    let mut input = vec![
                        0.0_f32;
                        (nx_load.max(1) as usize * max_in.max(1) as usize).max(
                            header.nx.max(1) as usize * header.ny.max(1) as usize
                        )
                    ];
                    //
                    // `newstack.f90:2300-2302` initialises the load window
                    // here, *before* the scan for the fill mean, because that
                    // scan reads into the same `array` and hands back the
                    // window its last load left there.
                    //
                    let (mut load_y_start, mut load_y_end) = (-1_i32, -1_i32);
                    let mut need_edge_mean = false;
                    //
                    // Get the mean of section for filling outside the image
                    // (`newstack.f90:2303-2321`), in the source's own order: an
                    // entered fill, then the edge mean when there is only one
                    // chunk, then the mean from the preliminary scan, then the
                    // input header's mean, and only failing all of those a scan
                    // of the section here.
                    //
                    let mut dmean_sec = if let Some(fill) = fill_value {
                        fill
                    } else if needed_for_output.need_fill && layout.len() == 1 {
                        // `newstack.f90:2305-2306`: with one chunk the mean is
                        // taken from the edges of the loaded data instead, in
                        // the chunk loop below.
                        need_edge_mean = true;
                        0.0
                    } else if if_mean != 0 && !mean_sd_entered && !sec_mean.is_empty() {
                        // `newstack.f90:2309-2310`: the mean from the scan that
                        // `-float 2`, `-float 3` and `-float 4` already made.
                        sec_mean[offset_index]
                    } else if !needed_for_output.need_fill {
                        header.amean
                    } else {
                        //
                        // `newstack.f90:2313-2321`: with more than one chunk
                        // the edge median is not available, so the source
                        // scans the section for its mean.  `scanSection` is
                        // handed `array` with `idimInOut` elements
                        // (`newstack.f90:2317`), and it splits the section
                        // into `array.len() / nx` line loads -- so the buffer
                        // size decides how the partial sums are grouped and
                        // therefore how the mean rounds.
                        //
                        let scan_ny = needed_for_output.iy_in_2 - needed_for_output.iy_in_1 + 1;
                        let scan_nx = bin_nx;
                        let mut scan_array = vec![0.0_f32; chunk_limit.max(scan_nx as usize)];
                        let mut scan_temp = vec![0.0_f32; effective_len_temp.max(1) as usize];
                        // `wallStart = wallTime()` (`newstack.f90:2316`).
                        let scan_wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        // `newstack.f90:2314-2315`.
                        if i_verbose > 0 {
                            print!(
                                " scanning for mean for fill {:>11} {:>11} {:>11} {:>11}\n",
                                in_section,
                                scan_ny,
                                needed_for_output.iy_in_1,
                                scan_temp.len()
                            );
                        }
                        let Ok((_, _, mean, _, scan_y_start, scan_y_end)) = scan_section(
                            &mut scan_array,
                            scan_nx,
                            scan_ny,
                            needed_for_output.iy_in_1,
                            read_reduction,
                            rx_offset,
                            ry_offset,
                            0,
                            0.0,
                            |data, lines, x_start, y_start| {
                                read_binned_or_reduced(
                                    1,
                                    in_section,
                                    data,
                                    scan_nx,
                                    lines,
                                    x_start,
                                    y_start,
                                    read_reduction,
                                    scan_nx,
                                    lines,
                                    ind_filter,
                                    read_shrunk,
                                    &mut scan_temp,
                                )
                            },
                        ) else {
                            exit_error("Reading image file");
                        };
                        //
                        // `scanSection` reads each of its loads into
                        // `array(1)` (`newstack.f90:3437`), so the section's
                        // *last* load is still sitting at the base of the
                        // working array when it returns, and it hands back
                        // that window in `loadYstart`/`loadYend`
                        // (`newstack.f90:3463-3464`).  The chunk loop below
                        // therefore skips its first read whenever the window
                        // already covers what chunk one needs -- and then uses
                        // the *scan's* wider window for `numYload`, `ycenIn`
                        // and the repack start.  `newstack.f90:2319` clamps it
                        // to the `maxin` lines that the chunk loop's own
                        // buffer holds, which is what makes the leftover
                        // usable.  The scan keeps a buffer of `idimInOut`
                        // elements because its load grouping decides how the
                        // mean rounds, so the lines that survive the clamp are
                        // copied across to the load buffer here.
                        //
                        load_y_start = scan_y_start;
                        load_y_end = scan_y_end.min(scan_y_start + max_in - 1);
                        let kept =
                            (load_y_end + 1 - load_y_start).max(0) as usize * nx_load as usize;
                        input[..kept].copy_from_slice(&scan_array[..kept]);
                        // `newstack.f90:2321`.
                        load_time += crate::imod::libcfshr::b3dutil::wall_time() - scan_wall_start;
                        mean
                    };
                    //
                    // The source's output space is the same flat `array` from
                    // `iBufOutBase` on, with chunk `iChunk` at
                    // `lineOutSt(iChunk) * nxOut` when the whole output is held
                    // (`newstack.f90:2386-2389`) and at the base otherwise.
                    // Holding the whole output here is what the `ifOutChunk > 0`
                    // reload at `newstack.f90:2605-2609` reads back into, and a
                    // chunk never reads outside its own lines, so one
                    // whole-section buffer stands in for both.
                    //
                    array = vec![0.0_f32; output_nx as usize * output_ny as usize];
                    // `newstack.f90:1759, 2738-2739`: with `-replace` the output
                    // section is the next entry of the replacement list, not the
                    // running output index.
                    let write_section = if list_replace.is_empty() {
                        out_section as i32
                    } else {
                        list_replace[out_section.min(list_replace.len() - 1)]
                    };
                    //
                    // `newstack.f90:2334-2592` is a loop over the output chunks
                    // the layout above settled on, and `newstack.f90:2595-2611`
                    // then rescales and writes them **backwards**.  Both orders
                    // are observable: `dsum`, `sdChunk` and `pixChunk` are
                    // accumulated one chunk at a time and fed to
                    // `chunkSumsToAvgsd`, and `dmean2` -- which becomes the
                    // header mean -- is summed in the reverse order.  Each
                    // chunk loads its own input window, transforms or repacks
                    // it, and only then counts and writes it.
                    //
                    let mut dsum = 0.0_f64;
                    let mut dsum_sq = 0.0_f64;
                    let mut dsum_chunk = Vec::<f64>::new();
                    let mut sd_chunk = Vec::<f32>::new();
                    let mut pix_chunk = Vec::<f64>::new();
                    let section = route_index - 1;
                    let mut scaling_values = FindScaleFactors {
                        if_float,
                        rescale,
                        num_scale_facs: scale_factors.len() as i32,
                        bottom_in,
                        bottom_out,
                        optimal_in,
                        optimal_out,
                        dmin_specified: scale_limits[0],
                        dmax_specified: scale_limits[1],
                        dmin_in: header.amin,
                        dmax_in: header.amax,
                        if_map_range: i32::from(map_entered),
                        dmap_low: map_limits[0],
                        dmap_high: map_limits[1],
                        frac_zero,
                        if_mean,
                        if_mean_sd_entered: i32::from(mean_sd_entered),
                        entered_mean: mean_sd[0],
                        entered_sd: mean_sd[1],
                        shift_mean,
                        shift_min,
                        shift_max,
                        new_mode: output_mode,
                        dsum: 0.,
                        dsum_sq: 0.,
                        nx_out: output_nx,
                        ny_out: output_ny,
                        scale_factor: 1.0,
                        const_add: 0.0,
                        scale_fac: scale_factors
                            .get(input_index.min(scale_factors.len().saturating_sub(1)))
                            .map_or(1.0, |entry| entry[0]),
                        scale_const: scale_factors
                            .get(input_index.min(scale_factors.len().saturating_sub(1)))
                            .map_or(0.0, |entry| entry[1]),
                        float_average: 0.,
                        float_sd: 0.,
                        zmin: float2_zmin,
                        zmax: float2_zmax,
                        float_z_margin,
                        opt_float_range,
                        opt_float_min,
                        section_min: sec_mins.get(section).copied().unwrap_or(0.0),
                        section_max: sec_maxes.get(section).copied().unwrap_or(0.0),
                        // The negated group of `newstack.f90:3057-3059`.
                        section_intentionally_truncated: output_mode == 2
                            || (num_sec_trunc > 0
                                && (z_min_outlier.get(section).copied().unwrap_or(0.0) < 0.0
                                    || z_max_outlier.get(section).copied().unwrap_or(0.0) > 0.0)),
                    };
                    // `scaleAndWriteChunk` accumulates `dmin2`, `dmax2` and
                    // `dmean2` across every chunk of the section, so the state
                    // is carried across the loop.
                    let mut scaling = ScaleAndWriteChunk {
                        new_mode: output_mode,
                        write_16_bit_mode_for_floats: write_16_bit_mode_for_floats() != 0,
                        scale_factor: 1.0,
                        const_add: 0.0,
                        optimal_out,
                        dmin2,
                        dmax2,
                        dmean2,
                        num_trunc_low: 0,
                        num_trunc_high: 0,
                    };
                    let num_out_chunks = layout.len();
                    for (chunk_index, &(line_out_st, num_lines_out, line_in_st, num_lines_in)) in
                        layout.iter().enumerate()
                    {
                        let base = line_out_st as usize * output_nx as usize;
                        let end = base + num_lines_out as usize * output_nx as usize;
                        // `newstack.f90:2335-2336`.
                        let need_y_start = line_in_st;
                        let need_y_end = need_y_start + num_lines_in - 1;
                        // `wallStart = wallTime()` (`newstack.f90:2340`).
                        let mut wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        if need_y_start < load_y_start || need_y_end > load_y_end {
                            //
                            // first load data that is needed if not already
                            // loaded (`newstack.f90:2341-2382`).  The window
                            // that is already in the buffer is shifted down or
                            // up and only the lines past it are read, which is
                            // what decides how many lines any one
                            // `readBinnedOrReduced` call is asked for -- and
                            // therefore how big a scratch array `irdReduced`
                            // needs.
                            //
                            let mut load_y_offset = need_y_start;
                            let mut load_base_ind = 0_usize;
                            let num_lines_load;
                            if load_y_start <= need_y_start && load_y_end >= need_y_start {
                                //
                                // move data down if it will fill a bottom region
                                //
                                let num_move =
                                    (load_y_end + 1 - need_y_start) as usize * nx_load as usize;
                                let move_offset =
                                    (need_y_start - load_y_start) as usize * nx_load as usize;
                                // `newstack.f90:2350`.  `numMove` and
                                // `moveOffset` are `integer(kind = 8)`
                                // (`newstack.f90:54`), so they are right
                                // justified in 20.
                                if i_verbose > 0 {
                                    print!(
                                        " moving data down {:>20} {:>20}\n",
                                        num_move, move_offset
                                    );
                                }
                                for i8 in 0..num_move {
                                    input[i8] = input[i8 + move_offset];
                                }
                                num_lines_load = need_y_end - load_y_end;
                                load_y_offset = load_y_end + 1;
                                load_base_ind = num_move;
                            } else if need_y_start <= load_y_start && need_y_end >= load_y_start {
                                //
                                // move data up if it will fill top
                                //
                                let num_move =
                                    (need_y_end + 1 - load_y_start) as usize * nx_load as usize;
                                let move_offset =
                                    (load_y_start - need_y_start) as usize * nx_load as usize;
                                // `newstack.f90:2363`.
                                if i_verbose > 0 {
                                    print!(
                                        " moving data up {:>20} {:>20}\n",
                                        num_move, move_offset
                                    );
                                }
                                for i8 in (0..num_move).rev() {
                                    input[i8 + move_offset] = input[i8];
                                }
                                num_lines_load = load_y_start - need_y_start;
                            } else {
                                //
                                // otherwise just get whole needed region
                                //
                                num_lines_load = need_y_end + 1 - need_y_start;
                                // `newstack.f90:2373-2374`.
                                if i_verbose > 0 {
                                    print!(
                                        " loading whole region {:>11} {:>11} {:>11}\n",
                                        need_y_start, need_y_end, num_lines_load
                                    );
                                }
                            }
                            let mut temp = vec![
                                0.0_f32;
                                // `lenTemp` as `reallocateIfNeeded` left it
                                // (`newstack.f90:2818-2842`).  `needTemp` starts
                                // at **one** (`newstack.f90:2822`) and only
                                // `readShrunk` and `iBinning > 1` raise it, so
                                // reproducing it as `nx * readReduction` gave
                                // `nx` where the source has 1 -- and for an
                                // entered `-memory` that is
                                // `idimInOut = limToAlloc - lenTemp`, so the
                                // scan's load grouping and the fill mean moved
                                // with it.
                                effective_len_temp.max(1) as usize
                            ];
                            // `newstack.f90:2376-2379`: the start is the reduced
                            // offset plus the reduction times the load offset.
                            if read_binned_or_reduced(
                                1,
                                in_section,
                                &mut input[load_base_ind..],
                                nx_load,
                                num_lines_load,
                                rx_offset,
                                ry_offset + read_reduction * load_y_offset as f32,
                                read_reduction,
                                nx_load,
                                num_lines_load,
                                ind_filter,
                                read_shrunk,
                                &mut temp,
                            )
                            .is_err()
                            {
                                exit_error("End of image while reading");
                            }
                            load_y_start = need_y_start;
                            load_y_end = need_y_end;
                        }
                        // `newstack.f90:2384-2389`.
                        let num_y_load = load_y_end + 1 - load_y_start;
                        let num_y_chunk = num_lines_out;
                        let mut chunk_base = buf_out_base;
                        if if_out_chunk == 0 {
                            chunk_base =
                                buf_out_base + i64::from(line_out_st) * i64::from(output_nx);
                        }
                        if process_in_place {
                            chunk_base = 1;
                        }
                        // `newstack.f90:2390`.
                        load_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        // `newstack.f90:2391-2392`.  `slice_edge_median` now
                        // reproduces the source for a zero-sample side --
                        // `percentileFloat` returns 0 for a non-positive count
                        // (`percentile.c`), so a two-line load yields
                        // `(median1 + median2) / 4`.
                        if need_edge_mean {
                            dmean_sec = slice_edge_median(
                                input.as_mut_ptr(),
                                nx_bin_sec,
                                0,
                                nx_bin_sec - 1,
                                0,
                                num_y_load - 1,
                                1,
                            ) as f32;
                        }
                        if !transforms.is_empty() {
                            //
                            // do transform if called for (`newstack.f90:2394-2433`)
                            //
                            // `wallStart = wallTime()` (`newstack.f90:2398`).
                            wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                            let matrix = [[fprod[0], fprod[1]], [fprod[2], fprod[3]]];
                            // `newstack.f90:2399-2402`: `ycenIn` is measured
                            // from the load window, not from what this chunk
                            // happens to need, and `dy` carries the chunk's
                            // first output line.
                            let xcen_in = nx_bin_sec as f32 / 2.0;
                            let ycen_in = ny_bin_sec as f32 / 2.0 - load_y_start as f32;
                            let dx = fprod[4];
                            let dy = (output_ny - num_y_chunk) as f32 / 2.0 + fprod[5]
                                - line_out_st as f32;
                            if lines_shrink > 0 {
                                // `newstack.f90:2405-2412`.
                                let ierr = unsafe {
                                    crate::imod::libcfshr::zoomdown::zoom_filt_interp(
                                        input.as_mut_ptr(),
                                        array[base..end].as_mut_ptr(),
                                        nx_bin_sec,
                                        num_y_load,
                                        output_nx,
                                        num_y_chunk,
                                        xcen_in,
                                        ycen_in,
                                        dx,
                                        dy,
                                        dmean_sec,
                                    )
                                };
                                if ierr != 0 {
                                    exit_error(&format!(
                                        "Calling zoomFiltInterp for image reduction, error{ierr:3}"
                                    ));
                                }
                            } else if !has_warp && if_mag_grad == 0 {
                                cubinterp(
                                    input.as_mut_ptr(),
                                    array[base..end].as_mut_ptr(),
                                    nx_bin_sec,
                                    num_y_load,
                                    output_nx,
                                    num_y_chunk,
                                    &matrix,
                                    xcen_in,
                                    ycen_in,
                                    dx,
                                    dy,
                                    1.0,
                                    dmean_sec,
                                    if_linear,
                                );
                            } else {
                                //
                                // `newstack.f90:2417-2431`: if undistorting,
                                // adjust the grid start down by the first
                                // loaded input line and by the offset of the
                                // subarea; if warping, down by the chunk's
                                // first output line.
                                //
                                let mut ystart = if if_warping != 0 {
                                    y_grid_start - line_out_st as f32
                                } else {
                                    y_grid_start - load_y_start as f32
                                };
                                let mut xstart = x_grid_start;
                                if if_distort > 0 {
                                    ystart -= warp_y_offsets[offset_index] / read_reduction;
                                    xstart -= warp_x_offsets[offset_index] / read_reduction;
                                }
                                warp_interp(
                                    input.as_mut_ptr(),
                                    array[base..end].as_mut_ptr(),
                                    nx_bin_sec,
                                    num_y_load,
                                    output_nx,
                                    num_y_chunk,
                                    &matrix,
                                    xcen_in,
                                    ycen_in,
                                    dx,
                                    dy,
                                    1.0,
                                    dmean_sec,
                                    if_linear,
                                    if_warping,
                                    field_dx.as_mut_ptr(),
                                    field_dy.as_mut_ptr(),
                                    lm_grid,
                                    nx_grid,
                                    ny_grid,
                                    xstart,
                                    ystart,
                                    x_grid_intrv,
                                    y_grid_intrv,
                                );
                            }
                            // `newstack.f90:2433`.
                            rot_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        } else {
                            //
                            // otherwise repack array into output space nxOut by
                            // nyOut, with offset as specified, using the special
                            // repack routine -- but first apply phase shift or
                            // reduction in the FFT (`newstack.f90:2434-2502`).
                            //
                            // The phase-shift padding sizes were set at the
                            // source's own place for them, `newstack.f90:2149-2157`,
                            // which is above the `nxDimNeed` that depends on them.
                            //
                            // `ioutBase = 1` (`newstack.f90:2440`).
                            let mut out_base = 1_i64;
                            let mut repack_nx_dim = nx_bin_sec;
                            let mut repack_ny_dim = num_y_load.max(ny_fspad);
                            // The Fourier routes write into their own padded
                            // buffer, which stands in for `array(ioutBase)`.
                            let mut fourier_out = Vec::<f32>::new();
                            if phase_shift || fourier_scaling {
                                // `wallStart = wallTime()` (`newstack.f90:2442`).
                                // The Fourier block below accumulates `rotTime`
                                // off this same start, so its total includes the
                                // padding time counted into `taperTime`.
                                wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                                let mut fft =
                                    vec![0.0_f32; ((nx_fspad + 2) * ny_fspad.max(1)) as usize];
                                let mut temp = vec![
                                    0.0_f32;
                                    2 * (nx_fspad.max(nx_fcrop_pad) / 2 + 2)
                                        as usize
                                ];
                                if noise_pad {
                                    slice_noise_taper_pad(
                                        input.as_mut_ptr().cast(),
                                        SLICE_MODE_FLOAT,
                                        nx_bin_sec,
                                        num_y_load,
                                        fft.as_mut_ptr(),
                                        nx_fspad + 2,
                                        nx_fspad,
                                        ny_fspad,
                                        20.max(120.min(nx_bin_sec.max(num_y_load) / 50)),
                                        4,
                                        temp.as_mut_ptr(),
                                    );
                                } else {
                                    slice_taper_out_pad(
                                        input.as_mut_ptr().cast(),
                                        SLICE_MODE_FLOAT,
                                        nx_bin_sec,
                                        num_y_load,
                                        fft.as_mut_ptr(),
                                        nx_fspad + 2,
                                        nx_fspad,
                                        ny_fspad,
                                        1,
                                        dmean_sec,
                                    );
                                }
                                // `newstack.f90:2451`.
                                taper_time +=
                                    crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                                todfft_c(fft.as_mut_ptr(), nx_fspad, ny_fspad, 0);
                                let shift_x =
                                    x_offsets[offset_index].round() - x_offsets[offset_index];
                                let shift_y =
                                    y_offsets[offset_index].round() - y_offsets[offset_index];
                                let mut cropped =
                                    vec![0.0_f32; ((nx_fcrop_pad + 2) * ny_fcrop_pad) as usize];
                                if phase_shift {
                                    fourier_shift_image(
                                        fft.as_mut_ptr(),
                                        nx_fspad,
                                        ny_fspad,
                                        shift_x,
                                        shift_y,
                                        temp.as_mut_ptr(),
                                    );
                                } else {
                                    // `newstack.f90:2459`: a Fourier reduction
                                    // or expansion writes into the second half
                                    // of the array, as `reallocateIfNeeded`
                                    // has just left `idimInOut`.
                                    out_base = 1 + alloc_idim_in_out as i64;
                                    if ft_reduce_fac > 0. {
                                        fourier_reduce_image(
                                            fft.as_mut_ptr(),
                                            nx_fspad,
                                            ny_fspad,
                                            cropped.as_mut_ptr(),
                                            nx_fcrop_pad,
                                            ny_fcrop_pad,
                                            actual_fac * shift_x,
                                            actual_fac * shift_y,
                                            temp.as_mut_ptr(),
                                        );
                                    } else {
                                        fourier_expand_image(
                                            fft.as_mut_ptr(),
                                            nx_fspad,
                                            ny_fspad,
                                            cropped.as_mut_ptr(),
                                            nx_fcrop_pad,
                                            ny_fcrop_pad,
                                            shift_x,
                                            shift_y,
                                            temp.as_mut_ptr(),
                                        );
                                    }
                                }
                                //
                                // Phase shifting is done in place, so `ioutBase`
                                // stays at the start of the padded FFT array
                                // (`newstack.f90:2439, 2459`).
                                //
                                if phase_shift {
                                    cropped = fft;
                                }
                                todfft_c(cropped.as_mut_ptr(), nx_fcrop_pad, ny_fcrop_pad, 1);
                                //
                                // Replicate last real column of image into the extra
                                // two elements
                                //
                                for i in 1..=ny_fcrop_pad {
                                    let last = (i * (nx_fcrop_pad + 2) - 3) as usize;
                                    cropped[last + 1] = cropped[last];
                                    cropped[last + 2] = cropped[last];
                                }
                                fourier_out = cropped;
                                repack_nx_dim = nx_fcrop_pad + 2;
                                repack_ny_dim = ny_fcrop_pad;
                                // `newstack.f90:2478`.
                                rot_time +=
                                    crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                            }
                            //
                            // Then repack, adjusting starting coordinates for
                            // the padded array (`newstack.f90:2481-2500`).
                            //
                            let (ix1, iy1) = if fourier_scaling {
                                (
                                    nx_fcrop_pad / 2 - output_nx / 2
                                        + x_offsets[offset_index].round() as i32,
                                    ny_fcrop_pad / 2 - output_ny / 2
                                        + y_offsets[offset_index].round() as i32,
                                )
                            } else if phase_shift {
                                (
                                    nx_bin_sec / 2 - output_nx / 2
                                        + x_offsets[offset_index].round() as i32
                                        + (nx_fspad - nx_bin_sec) / 2,
                                    ny_bin_sec / 2 - output_ny / 2
                                        + y_offsets[offset_index].round() as i32
                                        + line_out_st
                                        - load_y_start
                                        + (ny_fspad - num_y_load) / 2,
                                )
                            } else {
                                (
                                    nx_bin_sec / 2 - output_nx / 2
                                        + x_offsets[offset_index].round() as i32,
                                    ny_bin_sec / 2 - output_ny / 2
                                        + y_offsets[offset_index].round() as i32
                                        + line_out_st
                                        - load_y_start,
                                )
                            };
                            irepak2(
                                &mut array[base..end],
                                if fourier_out.is_empty() {
                                    &input[..]
                                } else {
                                    &fourier_out[..]
                                },
                                repack_nx_dim,
                                repack_ny_dim,
                                ix1,
                                ix1 + output_nx - 1,
                                iy1,
                                iy1 + num_y_chunk - 1,
                                dmean_sec,
                            );
                            // `newstack.f90:2501-2502`.  `iChunkBase` is
                            // `integer(kind = 8)` (`newstack.f90:53`) and right
                            // justified in 20; `ioutBase` is `integer*4`
                            // (`newstack.f90:112`).
                            if i_verbose > 0 {
                                print!(
                                    " did repack {:>20} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}\n",
                                    chunk_base,
                                    out_base,
                                    repack_nx_dim,
                                    repack_ny_dim,
                                    ix1,
                                    ix1 + output_nx - 1,
                                    iy1,
                                    iy1 + num_y_chunk - 1
                                );
                            }
                        }
                        // `newstack.f90:2504-2509`.  The source tapers
                        // `nxOut * nyOut` elements whatever the chunk holds;
                        // a multi-chunk taper is refused above, so this is the
                        // one chunk that is the whole section.
                        if num_taper > 0 {
                            // `wallStart = wallTime()` (`newstack.f90:2505`).
                            let wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                            // SLICE_MODE_FLOAT (`mrcslice.h:19`).
                            if crate::imod::libcfshr::taperatfill::taper_at_fill(
                                array[base..end].as_mut_ptr().cast(),
                                2,
                                output_nx,
                                output_ny,
                                num_taper,
                                inside_taper,
                            ) != 0
                            {
                                exit_error("Memory allocation error tapering image");
                            }
                            // `newstack.f90:2508`.
                            taper_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        }
                        //
                        // `newstack.f90:2514-2543`: min, max, sum and sum of
                        // squares over this chunk of the output image, before
                        // any rescaling.
                        //
                        let mut chunk_tsum = 0.0_f64;
                        if if_float == 2 {
                            // `call iclAvgSd(...)` (`newstack.f90:2516`).
                            // `arrayMinMaxMeanSd` (`simplestat.c:270-320`) does not
                            // return the plain pixel sums: it accumulates about a
                            // subsampled `roughMean` in single precision and then
                            // rebuilds `tsum` as `nxArea * (nyArea * avg8)` and
                            // `tsumSq` from the resulting SD, so summing the pixels
                            // directly here gives a different `avgSec` and `sdSec`.
                            let (mut tmin2, mut tmax2) = (0.0_f32, 0.0_f32);
                            let (mut tsum, mut tsum_sq) = (0.0_f64, 0.0_f64);
                            let (mut avg_sec, mut sd_this) = (0.0_f32, 0.0_f32);
                            crate::imod::libcfshr::simplestat::array_min_max_mean_sd(
                                array[base..end].as_ptr(),
                                output_nx,
                                num_y_chunk,
                                0,
                                output_nx - 1,
                                0,
                                num_y_chunk - 1,
                                &raw mut tmin2,
                                &raw mut tmax2,
                                &raw mut tsum,
                                &raw mut tsum_sq,
                                &raw mut avg_sec,
                                &raw mut sd_this,
                            );
                            // `newstack.f90:2517-2519`.
                            pix_chunk.push(f64::from(output_nx) * f64::from(num_y_chunk));
                            dsum_chunk.push(tsum);
                            sd_chunk.push(sd_this);
                            // `newstack.f90:2520-2521`.  The source only reaches
                            // the statistics block when `.not. rescale .or.
                            // ifMean .ne. 0`; this route computes them either way,
                            // so the report carries the source's own condition.
                            if i_verbose > 0 && (!rescale || if_mean != 0) {
                                print!(
                                    " chunk mean&sd min/max {:>11} {} {} {} {}\n",
                                    chunk_index + 1,
                                    list_real(avg_sec),
                                    list_real(sd_this),
                                    list_real(tmin2),
                                    list_real(tmax2)
                                );
                            }
                            // `newstack.f90:2529-2531`.
                            tmp_min = tmp_min.min(tmin2);
                            tmp_max = tmp_max.max(tmax2);
                            chunk_tsum = tsum;
                            dsum += tsum;
                            // `dsumSq = dsumSq + tsumSq` (`newstack.f90:2522`);
                            // only the blank-section route reads it.
                            dsum_sq += tsum_sq;
                            // `newstack.f90:2531`, under the same condition.
                            if i_verbose > 0 && (!rescale || if_mean != 0) {
                                print!(
                                    " did iclden  {} {} {} {}\n",
                                    list_real(tmin2),
                                    list_real(tmax2),
                                    list_real(tmp_min),
                                    list_real(tmp_max)
                                );
                            }
                        } else {
                            // `call iclden(...)` then `tsum = tmean2 * numPix`
                            // (`newstack.f90:2523-2526`).  `arrayMinMaxMean`
                            // (`simplestat.c:177-199`) keeps its per-line running
                            // total in single precision, and the section sum is
                            // rebuilt from the rounded mean.
                            let (mut tmin2, mut tmax2, mut tmean2) = (0.0_f32, 0.0_f32, 0.0_f32);
                            crate::imod::libcfshr::simplestat::array_min_max_mean(
                                array[base..end].as_ptr(),
                                output_nx,
                                num_y_chunk,
                                0,
                                output_nx - 1,
                                0,
                                num_y_chunk - 1,
                                &raw mut tmin2,
                                &raw mut tmax2,
                                &raw mut tmean2,
                            );
                            // `newstack.f90:2529-2531`.
                            tmp_min = tmp_min.min(tmin2);
                            tmp_max = tmp_max.max(tmax2);
                            // `tsum = tmean2 * numPix` keeps a real*4 result.
                            chunk_tsum =
                                f64::from(tmean2 * (output_nx as f32 * num_y_chunk as f32));
                            dsum += chunk_tsum;
                            // `newstack.f90:2531`, under the same condition.
                            if i_verbose > 0 && (!rescale || if_mean != 0) {
                                print!(
                                    " did iclden  {} {} {} {}\n",
                                    list_real(tmin2),
                                    list_real(tmax2),
                                    list_real(tmp_min),
                                    list_real(tmp_max)
                                );
                            }
                        }
                        //
                        // 6/27/01: really want to truncate rather than rescale; so
                        // if the min or max is now out of range for the input
                        // mode, truncate the data and adjust the min and max
                        // (`newstack.f90:2544-2566`).
                        //
                        if if_float == 0
                            && output_mode != 2
                            && header.mode != 2
                            && (tmp_min < bottom_in || tmp_max > optimal_in)
                        {
                            let mut tsum2 = 0.0_f64;
                            for value in array[base..end].iter_mut() {
                                if *value < bottom_in {
                                    num_trunc_low += 1;
                                }
                                if *value > optimal_in {
                                    num_trunc_high += 1;
                                }
                                *value = bottom_in.max(optimal_in.min(*value));
                                tsum2 += f64::from(*value);
                            }
                            tmp_min = tmp_min.max(bottom_in);
                            tmp_max = tmp_max.min(optimal_in);
                            dsum = dsum + tsum2 - chunk_tsum;
                        }
                        // `wallStart = wallTime()` (`newstack.f90:2570`).
                        let wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                        if pre_set_scaling {
                            // `newstack.f90:2570-2576`: the pre-set factors do
                            // not depend on this chunk's min and max, so the
                            // chunk is scaled and written straight away.
                            // `findScaleFactors` modifies `tmpMin`/`tmpMax`, so
                            // the source saves and restores them around it.
                            let (save_min, save_max) = (tmp_min, tmp_max);
                            let factors =
                                find_scale_factors(&scaling_values, header.amin, header.amax);
                            tmp_min = save_min;
                            tmp_max = save_max;
                            scaling.scale_factor = factors.scale_factor;
                            scaling.const_add = factors.const_add;
                            if scale_and_write_chunk(
                                &mut array[base..end],
                                output_nx,
                                &mut scaling,
                                |_| Ok(()),
                            )
                            .is_err()
                            {
                                exit_error("Scaling output image");
                            }
                            // `scaleAndWriteChunk` does its own write
                            // (`newstack.f90:3255-3257`); here the write stayed
                            // with the caller, so the report sits in front of it.
                            if i_verbose > 0 {
                                print!(" writing {:>11}\n", chunk_index + 1);
                            }
                            iiu_set_position(2, write_section, line_out_st);
                            iiu_write_lines(2, array[base..end].as_mut_ptr().cast(), num_lines_out);
                        } else if !rescale {
                            // `newstack.f90:2581-2584`.
                            if i_verbose > 0 {
                                print!(" writing to real file {:>11}\n", chunk_index + 1);
                            }
                            iiu_set_position(2, write_section, line_out_st);
                            iiu_write_lines(2, array[base..end].as_mut_ptr().cast(), num_lines_out);
                        } else if chunk_index + 1 != num_out_chunks && if_out_chunk > 0 {
                            // `newstack.f90:2585-2588`: every chunk but the last
                            // goes out to the scratch file, and the buffer is
                            // reused for the next one.
                            if i_verbose > 0 {
                                print!(" writing to temp file {:>11}\n", chunk_index + 1);
                            }
                            iiu_set_position(3, 0, line_out_st);
                            if iiu_write_lines(
                                3,
                                array[base..end].as_mut_ptr().cast(),
                                num_lines_out,
                            ) != 0
                            {
                                exit_error("Writing image file");
                            }
                        }
                        // `newstack.f90:2590`.
                        save_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                    }
                    // `chunkSumsToAvgsd` (`newstack.f90:3556`) over the chunks
                    // as `iclAvgSd` supplied them.
                    let (float_average, float_sd) = if dsum_chunk.is_empty() {
                        (0.0, 0.0)
                    } else {
                        chunk_sums_to_avgsd(
                            &dsum_chunk,
                            &sd_chunk,
                            &pix_chunk,
                            output_nx,
                            output_ny,
                        )
                    };
                    scaling_values.dsum = dsum;
                    scaling_values.dsum_sq = dsum_sq;
                    scaling_values.float_average = float_average;
                    scaling_values.float_sd = float_sd;
                    if !pre_set_scaling {
                        //
                        // `newstack.f90:2592-2611`: find the factors from the
                        // section's own min and max, then loop **backwards**
                        // over the chunks scaling and writing.  The direction
                        // is not cosmetic: `dmin2`, `dmax2` and `dmean2`
                        // accumulate in that order.
                        //
                        let factors = find_scale_factors(&scaling_values, tmp_min, tmp_max);
                        tmp_min = factors.tmp_min;
                        tmp_max = factors.tmp_max;
                        scaling.scale_factor = factors.scale_factor;
                        scaling.const_add = factors.const_add;
                        scaling.dmean2 = 0.;
                        for (chunk_index, &(line_out_st, num_lines_out, _, _)) in
                            layout.iter().enumerate().rev()
                        {
                            let base = line_out_st as usize * output_nx as usize;
                            let end = base + num_lines_out as usize * output_nx as usize;
                            // `newstack.f90:2605-2610`.
                            if chunk_index + 1 != num_out_chunks && if_out_chunk > 0 {
                                // `newstack.f90:2607`.
                                if i_verbose > 0 {
                                    print!(" reading {:>11}\n", chunk_index + 1);
                                }
                                iiu_set_position(3, 0, line_out_st);
                                if irdsecl(3, &mut array[base..end], num_lines_out).is_err() {
                                    exit_error("Reading temporary file");
                                }
                            }
                            if scale_and_write_chunk(
                                &mut array[base..end],
                                output_nx,
                                &mut scaling,
                                |_| Ok(()),
                            )
                            .is_err()
                            {
                                exit_error("Scaling output image");
                            }
                            // `wallStart = wallTime()` (`newstack.f90:3254`).
                            let wall_start = crate::imod::libcfshr::b3dutil::wall_time();
                            // `newstack.f90:3255`.
                            if i_verbose > 0 {
                                print!(" writing {:>11}\n", chunk_index + 1);
                            }
                            iiu_set_position(2, write_section, line_out_st);
                            iiu_write_lines(2, array[base..end].as_mut_ptr().cast(), num_lines_out);
                            // `newstack.f90:3258`.
                            save_time += crate::imod::libcfshr::b3dutil::wall_time() - wall_start;
                        }
                    }
                    num_trunc_low += scaling.num_trunc_low;
                    num_trunc_high += scaling.num_trunc_high;
                    dmin2 = scaling.dmin2;
                    dmax2 = scaling.dmax2;
                    dmean2 = scaling.dmean2;
                }
                if blank_section {
                    // `newstack.f90:1759, 2738-2739`: with `-replace` the output
                    // section is the next entry of the replacement list, not the
                    // running output index.
                    let write_section = if list_replace.is_empty() {
                        out_section as i32
                    } else {
                        list_replace[out_section.min(list_replace.len() - 1)]
                    };
                    // `newstack.f90:2033-2039`: a blank section is written
                    // whole, outside the chunk loop, so its write is here.
                    // `iiuWriteLines` (`unit_fileio.c:724-727`) prints
                    // `ERROR: iiuWriteLines - writing lines to unit %d.` and exits
                    // on a failed write, so the source never sees a return value
                    // to test and nothing here does either.  Going through it is
                    // also what syncs a TIFF unit's `ImodImageFile` from the unit
                    // header first -- "Have to sync header for TIFF file because
                    // the write is the time when it matters"
                    // (`unit_fileio.c:715-716`) -- which reaches
                    // `tiffSyncFromMrcHeader`, whose last act is
                    // `tiffAddDescription(inFile->description)` (`iitif.c:809`),
                    // and `tiffWriteSetup` *frees* the static `sDescription` right
                    // after writing it into a directory (`iitif.c:2600-2603`), so
                    // the description is re-made per section by design.
                    iiu_set_position(2, write_section, 0);
                    iiu_write_lines(2, array.as_mut_ptr().cast(), output_ny);
                    // `newstack.f90:2041-2044`: a blank section jumps to the
                    // source's label 80, past both the division and the
                    // report, with the fill value as all three totals.
                    dmin2 = array[0];
                    dmax2 = dmin2;
                    dmean2 = dmin2;
                } else {
                    // `newstack.f90:2622-2628`.
                    dmean2 /= output_nx as f32 * output_ny as f32;
                    if !quiet {
                        if if_header_out == 0 {
                            println!(" section   input min&max       output min&max  &  mean");
                        }
                        if_header_out = 1;
                        // Fortran `f10.2` fills the field with asterisks when the
                        // value does not fit in ten characters (`newstack.f90:2627`
                        // is `write(*,'(i8,5f10.2)')`); Rust's `{:10.2}` widens the
                        // field instead and pushes the rest of the line right.
                        println!(
                            "{:8}{}{}{}{}{}",
                            route_index - 1,
                            if format!("{tmp_min:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{tmp_min:10.2}")
                            },
                            if format!("{tmp_max:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{tmp_max:10.2}")
                            },
                            if format!("{dmin2:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{dmin2:10.2}")
                            },
                            if format!("{dmax2:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{dmax2:10.2}")
                            },
                            if format!("{dmean2:.2}").len() > 10 {
                                "**********".to_owned()
                            } else {
                                format!("{dmean2:10.2}")
                            }
                        );
                    }
                }
                // `newstack.f90:2630-2635`.
                dmin = dmin.min(dmin2);
                dmax = dmax.max(dmax2);
                // The source only accumulates `dmean` when `numReplace == 0`,
                // so a replaced file keeps the mean its header already had.
                if list_replace.is_empty() {
                    dmean += dmean2;
                    //
                    // transfer extra header bytes if present
                    // (`newstack.f90:2636-2683`)
                    //
                    if n_byte_sym_out != 0 && ind_extra_out < n_byte_sym_out {
                        //
                        // get this section's size and offset, make sure its OK;
                        // only FEI gives error
                        let mut move_offset = 0_i32;
                        let mut n_byte_extra_in = 0_i32;
                        if get_extra_header_sec_offset(
                            extra_in.as_mut_ptr().cast(),
                            n_byte_sym_in,
                            num_int_or_bytes_in,
                            i_flag_extra_in,
                            in_section,
                            &raw mut move_offset,
                            &raw mut n_byte_extra_in,
                        ) != 0
                        {
                            exit_error(
                                "FEI1 extended header does not contain data for all sections being read",
                            );
                        }
                        if fei1_type {
                            //
                            // FEI type, call the copy function
                            if copy_extra_header_section(
                                extra_in.as_mut_ptr().cast(),
                                n_byte_sym_in,
                                extra_out.as_mut_ptr().cast(),
                                n_byte_sym_out,
                                num_int_or_bytes_in,
                                i_flag_extra_in,
                                in_section,
                                &raw mut ind_extra_out,
                            ) != 0
                            {
                                exit_error(
                                    "Space allowed for copying FEI1 extra header not big enough; this is due to either program error or mixing files with different sizes per section",
                                );
                            }
                        } else {
                            let mut n_byte_copy =
                                n_byte_extra_out.min(n_byte_extra_in).min(n_byte_sym_in);
                            let mut num_for_tilt = 0_i32;
                            if save_tilts {
                                //
                                // To save tilt angles, put angle in the integer or
                                // real then copy the right number of bytes; adjust
                                // the number to copy and number to clear.
                                // `btiltTemp` is `equivalence`d onto both
                                // (`newstack.f90:46-49`).
                                let mut btilt_temp = [0_u8; 4];
                                if serial_em_type {
                                    let itilt_temp = ((100. * extra_tilts[route_index - 1]).round()
                                        as i32)
                                        as i16;
                                    btilt_temp[..2].copy_from_slice(&itilt_temp.to_ne_bytes());
                                    num_for_tilt = 2;
                                } else {
                                    btilt_temp.copy_from_slice(
                                        &extra_tilts[route_index - 1].to_ne_bytes(),
                                    );
                                    num_for_tilt = 4;
                                }
                                for i in 0..num_for_tilt as usize {
                                    ind_extra_out += 1;
                                    extra_out[ind_extra_out as usize - 1] = btilt_temp[i];
                                }
                                n_byte_copy = n_byte_copy.max(num_for_tilt) - num_for_tilt;
                            }
                            let n_byte_clear = n_byte_extra_out - num_for_tilt - n_byte_copy;
                            //
                            // Copy bytes, then clear out the rest if any
                            for i in 1..=n_byte_copy {
                                ind_extra_out += 1;
                                extra_out[ind_extra_out as usize - 1] =
                                    extra_in[(in_section * n_byte_extra_in + i + num_for_tilt)
                                        as usize
                                        - 1];
                            }
                            for _ in 0..n_byte_clear {
                                ind_extra_out += 1;
                                extra_out[ind_extra_out as usize - 1] = 0;
                            }
                        }
                    }
                }
                //
                // Transfer an adoc section (`newstack.f90:2686-2716`).
                // `isecOut - 2` is this output section's zero-based number,
                // which is `out_section` here.
                //
                let list_string = format!("{out_section}");
                let Ok(list_c) = CString::new(list_string.as_bytes()) else {
                    exit_error("Invalid autodoc section name");
                };
                if ind_adoc_in > 0 && ind_adoc_out > 0 {
                    set_current_adoc_or_exit(ind_adoc_in, "input");
                    if frame_set {
                        // For a FrameSet, first try to get the angle for a
                        // title for the first section, then transfer it to a
                        // ZValue.
                        if out_section == 0 {
                            let mut value = 0.0_f32;
                            if adoc_get_float(
                                c"FrameSet".as_ptr(),
                                0,
                                c"RotationAngle".as_ptr(),
                                &raw mut value,
                            ) == 0
                            {
                                value -= 90.;
                                if value < -180. {
                                    value += 360.;
                                }
                                let title_text = format!("Tilt axis angle = {value:8.1}");
                                let Ok(title_c) = CString::new(title_text.as_bytes()) else {
                                    exit_error("Invalid autodoc title");
                                };
                                set_current_adoc_or_exit(ind_adoc_out, "output");
                                if adoc_add_section(c"T".as_ptr(), title_c.as_ptr()) <= 0 {
                                    exit_error("Adding title section to autodoc");
                                }
                                set_current_adoc_or_exit(ind_adoc_in, "input");
                            }
                        }
                        if adoc_transfer_to_new_type(
                            c"FrameSet".as_ptr(),
                            0,
                            ind_adoc_out - 1,
                            ADOC_ZVALUE_NAME.as_ptr(),
                            list_c.as_ptr(),
                            1,
                        ) != 0
                        {
                            exit_error("Transferring FrameSet data to ZValue in new autodoc");
                        }
                    } else {
                        // Otherwise just transfer the section.
                        let ind_sect_in =
                            adoc_lookup_by_name_value(ADOC_ZVALUE_NAME.as_ptr(), in_section);
                        if ind_sect_in >= 0
                            && adoc_transfer_section(
                                ADOC_ZVALUE_NAME.as_ptr(),
                                ind_sect_in,
                                ind_adoc_out - 1,
                                list_c.as_ptr(),
                                1,
                            ) != 0
                        {
                            exit_error("Transferring section data between autodocs");
                        }
                    }
                    out_doc_changed = true;
                }
                //
                // Save tilts in the adoc section: if the section does not
                // exist, create it at the right index; add the value to it
                // (`newstack.f90:2719-2735`).
                //
                if ind_adoc_out > 0 && save_tilts {
                    set_current_adoc_or_exit(ind_adoc_out, "output");
                    let mut ind_sect_in =
                        adoc_lookup_by_name_value(ADOC_ZVALUE_NAME.as_ptr(), out_section as i32);
                    if ind_sect_in < 0 {
                        ind_sect_in =
                            adoc_find_insert_index(ADOC_ZVALUE_NAME.as_ptr(), out_section as i32);
                        if ind_sect_in >= 0
                            && adoc_insert_section(
                                ADOC_ZVALUE_NAME.as_ptr(),
                                ind_sect_in,
                                list_c.as_ptr(),
                            ) < 0
                        {
                            ind_sect_in = -2;
                        }
                        if ind_sect_in < 0 {
                            exit_error("Adding an autodoc section for saving tilt angle");
                        }
                    }
                    if adoc_set_float(
                        ADOC_ZVALUE_NAME.as_ptr(),
                        ind_sect_in,
                        c"TiltAngle".as_ptr(),
                        extra_tilts[route_index - 1],
                    ) != 0
                    {
                        exit_error("Adding tilt angle to autodoc");
                    }
                    out_doc_changed = true;
                }
            }
            //
            // `newstack.f90:2745-2750`: the output autodoc is written beside
            // the output file and cleared before the final header write.
            //
            if list_replace.is_empty() && out_doc_changed && (*out_file).file != 5 {
                set_current_adoc_or_exit(ind_adoc_out, "output");
                let mdoc_name = format!("{}.mdoc", output_names[output_index]);
                let Ok(mdoc_c) = CString::new(mdoc_name.as_bytes()) else {
                    exit_error("Invalid output file name");
                };
                if adoc_write(mdoc_c.as_ptr()) != 0 {
                    exit_error("Writing mdoc file for output file");
                }
                adoc_clear(ind_adoc_out - 1);
            }
            if !list_replace.is_empty() {
                // `newstack.f90:2764-2767`: one write with `labFlag = -1`, so
                // the existing labels are untouched and only the min, max and
                // (unchanged) mean are stored.
                if iiu_write_header(2, title.as_mut_ptr().cast(), -1, dmin, dmax, replace_dmean)
                    != 0
                {
                    exit_error("Writing output header");
                }
                iiu_close(2);
            } else if chunked_affine {
                // `newstack.f90:2751`: `iiuAltExtendedData(2, nByteSymOut,
                // extraOut)` writes the assembled extended header, which is a
                // no-op when the output is not MRC.
                if n_byte_sym_out > 0 {
                    iiu_alt_extended_data(2, n_byte_sym_out, extra_out.as_mut_ptr().cast());
                }
                // `newstack.f90:2753`: `labFlag = 1` appends this run's title.
                if iiu_write_header(
                    2,
                    title.as_mut_ptr().cast(),
                    1,
                    dmin,
                    dmax,
                    // `newstack.f90:2752`: the header mean is the sum of the
                    // per-section means over the section count.
                    dmean / num_output_sections[output_index] as f32,
                ) != 0
                {
                    exit_error("Writing output header");
                }
                iiu_close(2);
            } else {
                let out_header =
                    iiu_mrc_header(2, c"iiuWriteHeader".as_ptr(), iiu_get_exit_on_error(), 2);
                // `newstack.f90:2753`: `iiuWriteHeader(2, title, 1, ...)`.
                // `labFlag == 1` puts the title in the next label slot, capped
                // at `MRC_NLABELS` (`unit_header.c:290-296`).
                (*out_header).nlabl = ((*out_header).nlabl + 1).min(MRC_NLABELS as i32);
                let title_index = ((*out_header).nlabl - 1) as usize;
                (&mut (*out_header).labels[title_index])[..MRC_LABEL_SIZE].copy_from_slice(&title);
                fix_title_padding(&mut (*out_header).labels[title_index]);
                (*out_header).amin = dmin;
                (*out_header).amax = dmax;
                // `newstack.f90:2752`.
                (*out_header).amean = dmean / num_output_sections[output_index] as f32;
                // `newstack.f90:2751`: `iiuAltExtendedData(2, nByteSymOut,
                // extraOut)` writes the assembled extended header, which is a
                // no-op when the output is not MRC.
                if n_byte_sym_out > 0 {
                    iiu_alt_extended_data(2, n_byte_sym_out, extra_out.as_mut_ptr().cast());
                }
                ii_sync_from_mrc_header(out_file, out_header);
                if ((*out_file).file == IIFILE_MRC
                    && mrc_head_write((*out_file).fp, out_header) != 0)
                    || ((*out_file).file != IIFILE_MRC && ii_write_header(out_file) != 0)
                {
                    exit_error("Writing output header");
                }
                iiu_close(2);
            }
            // `newstack.f90:2755`.
            if need_close2 > 0 {
                iiu_close(need_close2);
            }
        }
        if !active_input_file.is_null() {
            iiu_close(1);
            // `newstack.f90:2762`.
            if need_close1 > 0 {
                iiu_close(need_close1);
            }
        }
        // `newstack.f90:2769`.
        if if_temp_open != 0 {
            iiu_close(3);
        }
        if reorder_by_tilt != 0 && !new_angle_output_file.is_empty() {
            // `call dopen(3, newAngleFile, 'new', 'f')` (`newstack.f90:2771`).
            drop(dopen(3, &new_angle_output_file, "new", "f"));
            let Ok(mut file) = std::fs::File::create(&new_angle_output_file) else {
                exit_error("Opening new tilt angle output file");
            };
            use std::io::Write;
            for angle in &extra_tilts {
                if writeln!(file, "{angle:9.2}").is_err() {
                    exit_error("Writing new tilt angle output file");
                }
            }
        }
        // `newstack.f90:2776-2778`: format 103 is not guarded by `quiet` and
        // its second literal ends with " at high end of range".
        if num_trunc_low + num_trunc_high > 0 {
            println!(
                " TRUNCATIONS OCCURRED:{num_trunc_low:11} at low end,{num_trunc_high:11} at high end of range"
            );
        }
        // `newstack.f90:2779-2787`.
        if num_sec_trunc > 0
            && (num_trunc_low + num_trunc_high) as f32
                > num_sec_trunc as f32 * 4.0 * (1.0 + output_nx as f32 * (output_ny as f32 / 1.0e6))
        {
            println!(
                "\nWARNING: NEWSTACK - {num_sec_trunc:4} sections had extreme ranges and were truncated to preserve dynamic range (overall, {:11} pixels were truncated)",
                num_trunc_low + num_trunc_high
            );
        } else if num_sec_trunc > 0 {
            println!(
                "\nNOTE: {num_sec_trunc:4} sections had extreme ranges and were truncated to preserve dynamic range "
            );
        }
        // `newstack.f90:2788-2790`.  This one is a formatted `write`, not a
        // list-directed `print`, so there is no leading record blank and each
        // value is plain `f8.4`.  The times themselves are wall clock and
        // differ between two runs of the same command, native or not.
        if i_verbose > 0 {
            print!(
                "loadtime{load_time:8.4}  savetime{save_time:8.4}  sum{:8.4}  rottime{rot_time:8.4}  taper{taper_time:8.4}\n",
                load_time + save_time
            );
        }
    }
}

/// Original `irepak2` (`newstack.f90:3372`).
pub fn irepak2(
    brray: &mut [f32],
    array: &[f32],
    mx: i32,
    my: i32,
    nx1: i32,
    nx2: i32,
    ny1: i32,
    ny2: i32,
    dmean: f32,
) {
    let mut ind = 0usize;
    for iy in ny1..=ny2 {
        for ix in nx1..=nx2 {
            if ind == brray.len() {
                return;
            }
            brray[ind] = if ix >= 0 && ix < mx && iy >= 0 && iy < my {
                array[iy as usize * mx as usize + ix as usize]
            } else {
                dmean
            };
            ind += 1;
        }
    }
}

/// Original `backXform` (`newstack.f90:3476`).
pub fn back_xform(
    nxb: i32,
    nyb: i32,
    amat: [[f32; 2]; 2],
    xfcen: f32,
    yfcen: f32,
    xtrans: f32,
    ytrans: f32,
    ix: i32,
    iy: i32,
) -> (f32, f32) {
    //
    // Calc inverse transformation
    //
    // `newstack.f90:3487-3494` divides each matrix element by `denom` first
    // and adds `xcenOut`/`ycenOut` as a single precomputed term: dividing the
    // combined product instead, or adding `xfcen` and `0.5` separately, lands
    // an ulp away and moves a chunk's input window by a line.
    //
    let xcen = nxb as f32 / 2.0 + xtrans + 0.5;
    let ycen = nyb as f32 / 2.0 + ytrans + 0.5;
    let xcen_out = xfcen + 0.5;
    let ycen_out = yfcen + 0.5;
    let denom = amat[0][0] * amat[1][1] - amat[0][1] * amat[1][0];
    let a11 = amat[1][1] / denom;
    let a12 = -amat[0][1] / denom;
    let a21 = -amat[1][0] / denom;
    let a22 = amat[0][0] / denom;
    //
    // get coordinate transforming to ix, iy
    //
    let dyo = iy as f32 - ycen;
    let dxo = ix as f32 - xcen;
    (
        a11 * dxo + a12 * dyo + xcen_out,
        a21 * dxo + a22 * dyo + ycen_out,
    )
}

/// Original `getReducedSize` (`newstack.f90:3507`).
pub fn get_reduced_size(
    nx: i32,
    reduction: f32,
    do_shrink: bool,
    if_odd_even_ok: i32,
) -> (i32, f32) {
    if do_shrink && (reduction.round() - reduction).abs() > 1.0e-4 {
        let nx_bin = (nx as f32 / reduction) as i32;
        return (nx_bin, (nx as f32 - nx_bin as f32 * reduction) / 2.0);
    }
    let (nx_bin, ix_offset) = get_binned_size(nx, reduction.round() as i32, if_odd_even_ok);
    (nx_bin, ix_offset as f32)
}

/// Original `chunkSumsToAvgsd` (`newstack.f90:3556`).
pub fn chunk_sums_to_avgsd(
    dsum_chunk: &[f64],
    sd_chunk: &[f32],
    pix_chunk: &[f64],
    nx: i32,
    ny: i32,
) -> (f32, f32) {
    let pix_tot = f64::from(nx) * f64::from(ny);
    let dsum: f64 = dsum_chunk.iter().sum();
    let dmean = dsum / pix_tot;
    // `newstack.f90:3572-3574`: the two terms are added onto the running
    // `dsumSq` one after the other, and `sdChunk(ichunk)**2` squares a
    // **real*4**, so the square is rounded to single precision before it is
    // widened for the product.
    let mut dsum_sq = 0.0_f64;
    for index in 0..dsum_chunk.len() {
        let average = dsum_chunk[index] / pix_chunk[index];
        dsum_sq = dsum_sq
            + pix_chunk[index] * (average * average - dmean * dmean)
            + (pix_chunk[index] - 1.0) * f64::from(sd_chunk[index] * sd_chunk[index]);
    }
    (
        dmean as f32,
        (dsum_sq.max(0.0) / (pix_tot - 1.0).max(1.0)).sqrt() as f32,
    )
}

/// Original `getItemsToUse` (`newstack.f90:3296`).
pub fn get_items_to_use(
    nxforms: i32,
    in_list: &[i32],
    option: &std::ffi::CStr,
    error: &str,
    one_per_file: bool,
    num_in_files: i32,
    number_offset: i32,
) -> Vec<i32> {
    let mut line_use = if nxforms == 1 {
        vec![number_offset]
    } else if one_per_file {
        (0..num_in_files)
            .map(|index| index + number_offset)
            .collect()
    } else {
        in_list.iter().map(|&index| index + number_offset).collect()
    };
    //
    // `newstack.f90:3333-3344`: the option is read **here**, and every one of
    // its `PipNumberOfEntries` entries is parsed and appended.  Reading a
    // single entry outside made `-uselines 0,1 -uselines 2,3,4` use only the
    // first, which then failed the caller's section-count check.
    //
    let mut num_xf_lines = 0_i32;
    unsafe { pip_number_of_entries(option.as_ptr(), &raw mut num_xf_lines) };
    if num_xf_lines > 0 {
        let mut parsed = Vec::<i32>::new();
        for _ in 0..num_xf_lines {
            let mut string_value: *mut std::ffi::c_char = std::ptr::null_mut();
            let list = unsafe {
                if pip_get_string(option.as_ptr(), &raw mut string_value) != 0 {
                    continue;
                }
                let list = std::ffi::CStr::from_ptr(string_value)
                    .to_string_lossy()
                    .into_owned();
                libc::free(string_value.cast());
                list
            };
            let mut values = vec![0_i32; 1_000_000 - parsed.len()];
            let (mut count, mut limit) = (0, (1_000_000 - parsed.len()) as i32);
            // `parseList2` prints its own `ERROR: PARSELIST - ...` and exits
            // for a positive `limList` (`rdlist.f90:71-79`), which is what the
            // source passes, so there is no failure for this routine to
            // report.
            drop(parselist2(&list, &mut values, &mut count, &mut limit));
            values.truncate(count as usize);
            parsed.extend(values);
        }
        line_use = parsed;
    }
    for line in &mut line_use {
        *line -= number_offset;
        if *line < 0 || *line >= nxforms {
            //
            // `newstack.f90:3355-3358`: the subroutine builds
            // `error // ' number out of bounds:' // i5` and calls
            // `exitError(trim(errString))` itself.  Returning the failure to
            // the caller instead lost the offending index and the `i5` field,
            // and left each caller to invent its own wording.
            //
            exit_error(&format!(
                "{error} number out of bounds:{:5}",
                *line + number_offset
            ));
        }
    }
    line_use
}

/// Values from the `newstack` program unit used by original
/// `linesNeededForOutput` (`newstack.f90:2909`).
#[derive(Clone, Copy, Debug)]
pub struct LinesNeededForOutput {
    pub fourier_scaling: bool,
    pub if_xform: i32,
    pub nx_bin: i32,
    pub ny_bin: i32,
    pub nx_out: i32,
    pub ny_out: i32,
    pub xcen: f32,
    pub ycen: f32,
    pub fprod: [[f32; 3]; 2],
    pub max_field_x: i32,
    pub max_field_y: i32,
    pub lines_shrink: i32,
}

/// Return values of original `linesNeededForOutput`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinesNeededResult {
    pub iy_in_1: i32,
    pub iy_in_2: i32,
    pub need_fill: bool,
    pub in_place: bool,
}

/// Original `linesNeededForOutput` (`newstack.f90:2909`).
pub fn lines_needed_for_output(
    values: &LinesNeededForOutput,
    line_out_first: i32,
    line_out_last: i32,
) -> LinesNeededResult {
    if values.fourier_scaling {
        return LinesNeededResult {
            iy_in_1: 0,
            iy_in_2: values.ny_bin - 1,
            need_fill: values.xcen.abs() > 1.0 || values.ycen.abs() > 1.0,
            in_place: true,
        };
    }
    if values.if_xform == 0 {
        // `iyBase = nyBin / 2 + ycen(isec) - (nyOut / 2)`: the integer
        // divisions promote to real for the sum with `ycen`, and the whole
        // expression truncates on assignment to the integer.
        let iy_base =
            ((values.ny_bin / 2) as f32 + values.ycen - (values.ny_out / 2) as f32) as i32;
        let ix_base =
            ((values.nx_bin / 2) as f32 + values.xcen - (values.nx_out / 2) as f32) as i32;
        return LinesNeededResult {
            iy_in_1: 0.max(iy_base + line_out_first),
            iy_in_2: (values.ny_bin - 1).min(iy_base + line_out_last),
            need_fill: iy_base + line_out_first < 0
                || iy_base + line_out_last >= values.ny_bin
                || ix_base < 0
                || ix_base + values.nx_out > values.nx_bin,
            in_place: ix_base >= 0 && iy_base + line_out_first >= 0,
        };
    }
    let amat = [
        [values.fprod[0][0], values.fprod[0][1]],
        [values.fprod[1][0], values.fprod[1][1]],
    ];
    let xcen_in = values.nx_bin as f32 / 2.0;
    let ycen_in = values.ny_bin as f32 / 2.0;
    let dx = values.fprod[0][2];
    let dy = values.fprod[1][2];
    let points = [
        back_xform(
            values.nx_out,
            values.ny_out,
            amat,
            xcen_in,
            ycen_in,
            dx,
            dy,
            1,
            line_out_first + 1,
        ),
        back_xform(
            values.nx_out,
            values.ny_out,
            amat,
            xcen_in,
            ycen_in,
            dx,
            dy,
            values.nx_out,
            line_out_first + 1,
        ),
        back_xform(
            values.nx_out,
            values.ny_out,
            amat,
            xcen_in,
            ycen_in,
            dx,
            dy,
            1,
            line_out_last + 1,
        ),
        back_xform(
            values.nx_out,
            values.ny_out,
            amat,
            xcen_in,
            ycen_in,
            dx,
            dy,
            values.nx_out,
            line_out_last + 1,
        ),
    ];
    let ix1 = points
        .iter()
        .map(|point| point.0)
        .fold(f32::INFINITY, f32::min) as i32
        - 2
        - values.max_field_x
        - values.lines_shrink;
    let ix2 = points
        .iter()
        .map(|point| point.0)
        .fold(f32::NEG_INFINITY, f32::max) as i32
        + 1
        + values.max_field_x
        + values.lines_shrink;
    let iy1 = points
        .iter()
        .map(|point| point.1)
        .fold(f32::INFINITY, f32::min) as i32
        - 2
        - values.max_field_y
        - values.lines_shrink;
    let iy2 = points
        .iter()
        .map(|point| point.1)
        .fold(f32::NEG_INFINITY, f32::max) as i32
        + 1
        + values.max_field_y
        + values.lines_shrink;
    LinesNeededResult {
        iy_in_1: iy1.clamp(0, values.ny_bin - 1),
        iy_in_2: iy2.clamp(0, values.ny_bin - 1),
        need_fill: ix1 < 0
            || ix1 >= values.nx_bin
            || ix2 < 0
            || ix2 >= values.nx_bin
            || iy1 < 0
            || iy1 >= values.ny_bin
            || iy2 < 0
            || iy2 >= values.ny_bin,
        in_place: false,
    }
}

/// Program-unit values consumed by original `findScaleFactors`
/// (`newstack.f90:2975`).  Per-section values are supplied already selected
/// with the source expression `ilist + listInd(iFile) - 1`.
#[derive(Clone, Copy, Debug)]
pub struct FindScaleFactors {
    pub if_float: i32,
    pub rescale: bool,
    pub num_scale_facs: i32,
    pub bottom_in: f32,
    pub bottom_out: f32,
    pub optimal_in: f32,
    pub optimal_out: f32,
    pub dmin_specified: f32,
    pub dmax_specified: f32,
    pub dmin_in: f32,
    pub dmax_in: f32,
    pub if_map_range: i32,
    pub dmap_low: f32,
    pub dmap_high: f32,
    pub frac_zero: f32,
    pub if_mean: i32,
    pub if_mean_sd_entered: i32,
    pub entered_mean: f32,
    pub entered_sd: f32,
    pub shift_mean: f32,
    pub shift_min: f32,
    pub shift_max: f32,
    pub new_mode: i32,
    pub dsum: f64,
    pub dsum_sq: f64,
    pub nx_out: i32,
    pub ny_out: i32,
    pub scale_factor: f32,
    pub const_add: f32,
    pub scale_fac: f32,
    pub scale_const: f32,
    pub float_average: f32,
    pub float_sd: f32,
    pub zmin: f32,
    pub zmax: f32,
    pub float_z_margin: f32,
    pub opt_float_range: f32,
    pub opt_float_min: f32,
    pub section_min: f32,
    pub section_max: f32,
    pub section_intentionally_truncated: bool,
}

/// Result values (including source's mutable `tmpMin/tmpMax`) of
/// `findScaleFactors`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleFactors {
    pub tmp_min: f32,
    pub tmp_max: f32,
    pub scale_factor: f32,
    pub const_add: f32,
}

/// Original `findScaleFactors` (`newstack.f90:2975`), for all branches that
/// do not require the program's outlier-scan arrays.
pub fn find_scale_factors(
    values: &FindScaleFactors,
    tmp_min_in: f32,
    tmp_max_in: f32,
) -> ScaleFactors {
    let mut scale_factor = 1.;
    let mut const_add = 0.;
    let mut tmp_min = tmp_min_in;
    let mut tmp_max = tmp_max_in;
    let mut dmin_out = tmp_min;
    let mut dmax_out = tmp_max;
    if values.if_float == 0 && values.rescale {
        dmin_out = (tmp_min - values.bottom_in) * (values.optimal_out - values.bottom_out)
            / (values.optimal_in - values.bottom_in)
            + values.bottom_out;
        dmax_out = (tmp_max - values.bottom_in) * (values.optimal_out - values.bottom_out)
            / (values.optimal_in - values.bottom_in)
            + values.bottom_out;
    } else if values.if_float < 0 && values.num_scale_facs == 0 {
        let (dmin_new, dmax_new) = if values.dmin_specified == 0. && values.dmax_specified == 0. {
            (0., values.optimal_out)
        } else if values.dmin_specified == values.dmax_specified {
            (values.dmin_in, values.dmax_in)
        } else {
            (values.dmin_specified, values.dmax_specified)
        };
        let (use_min, use_max) = if values.if_map_range != 0 {
            (values.dmap_low, values.dmap_high)
        } else {
            (values.dmin_in, values.dmax_in)
        };
        dmin_out = (tmp_min - use_min) * (dmax_new - dmin_new) / (use_max - use_min) + dmin_new;
        dmax_out = (tmp_max - use_min) * (dmax_new - dmin_new) / (use_max - use_min) + dmin_new;
    } else if values.if_float > 0 && values.if_mean_sd_entered == 0 {
        dmin_out = -values.optimal_out * values.frac_zero / (1. - values.frac_zero);
        if values.if_mean == 0 {
            dmax_out = values.optimal_out;
        } else if values.if_float == 2 {
            let average = values.float_average;
            let mut sd = if tmp_min == tmp_max || values.float_sd == 0. {
                1.
            } else {
                values.float_sd
            };
            let zmin_now = (tmp_min - average) / sd;
            let zmax_now = (tmp_max - average) / sd;
            if !values.section_intentionally_truncated
                && (zmin_now < values.zmin - values.float_z_margin
                    || zmax_now > values.zmax + values.float_z_margin)
            {
                let mut boost_min = 1.;
                let mut boost_max = 1.;
                if average - values.section_min > 0.25 * (average - tmp_min)
                    && zmin_now < values.zmin - values.float_z_margin
                {
                    boost_min = zmin_now / (values.zmin - values.float_z_margin);
                }
                if values.section_max - average > 0.25 * (tmp_max - average)
                    && zmax_now > values.zmax + values.float_z_margin
                {
                    boost_max = zmax_now / (values.zmax + values.float_z_margin);
                }
                sd *= boost_min.max(boost_max);
            }
            tmp_min = tmp_min.max(values.zmin * sd + average);
            tmp_max = tmp_max.min(values.zmax * sd + average);
            let zmin_section = (tmp_min - average) / sd;
            let zmax_section = (tmp_max - average) / sd;
            dmin_out = ((zmin_section - values.zmin) * values.opt_float_range
                / (values.zmax - values.zmin)
                + values.opt_float_min)
                .max(0.);
            dmax_out = ((zmax_section - values.zmin) * values.opt_float_range
                / (values.zmax - values.zmin)
                + values.opt_float_min)
                .min(values.optimal_out);
        } else {
            // `tmpMean = dsum / (float(nxOut) * nyOut)`: a real*8 numerator
            // over a real*4 product, rounded once on assignment.
            let tmp_mean =
                (values.dsum / f64::from(values.nx_out as f32 * values.ny_out as f32)) as f32;
            let tmp_min_shift = tmp_min + values.shift_mean - tmp_mean;
            let tmp_max_shift = tmp_max + values.shift_mean - tmp_mean;
            if values.if_float == 3 {
                dmin_out = tmp_min_shift;
                dmax_out = tmp_max_shift;
                if values.new_mode != 2 {
                    let optimal_in = values.optimal_in.max(values.shift_max);
                    dmin_out = tmp_min_shift * values.optimal_out / optimal_in;
                    dmax_out = tmp_max_shift * values.optimal_out / optimal_in;
                }
            } else {
                let (dmin_new, dmax_new) = if values.dmin_specified == values.dmax_specified {
                    (0.5, values.optimal_out - 0.5)
                } else {
                    (values.dmin_specified, values.dmax_specified)
                };
                dmin_out = (tmp_min_shift - values.shift_min) * (dmax_new - dmin_new)
                    / (values.shift_max - values.shift_min)
                    + dmin_new;
                dmax_out = (tmp_max_shift - values.shift_min) * (dmax_new - dmin_new)
                    / (values.shift_max - values.shift_min)
                    + dmin_new;
            }
        }
    }
    if values.rescale {
        if values.num_scale_facs > 0 {
            scale_factor = values.scale_fac;
            const_add = values.scale_const;
        } else if values.if_mean_sd_entered != 0 {
            let count = (values.nx_out * values.ny_out) as f64;
            let average = values.dsum / count;
            let sd = ((values.dsum_sq - count * average * average) / (count - 1.).max(1.))
                .max(0.)
                .sqrt() as f32;
            scale_factor = values.entered_sd / if sd == 0. { 1. } else { sd };
            const_add = values.entered_mean - scale_factor * average as f32;
        } else {
            if dmax_out != dmin_out && tmp_max != tmp_min {
                scale_factor = (dmax_out - dmin_out) / (tmp_max - tmp_min);
            }
            const_add = dmin_out - scale_factor * tmp_min;
        }
    }
    ScaleFactors {
        tmp_min,
        tmp_max,
        scale_factor,
        const_add,
    }
}

/// Original `readBinnedOrReduced` (`newstack.f90:3528`).
pub fn read_binned_or_reduced(
    im_unit: i32,
    iz: i32,
    array: &mut [f32],
    nx_dim: i32,
    ny_dim: i32,
    x_ub_start: f32,
    y_ub_start: f32,
    red_fac: f32,
    nx_red: i32,
    ny_red: i32,
    ifilt_type: i32,
    do_shrink: bool,
    temp: &mut [f32],
) -> Result<(), i32> {
    let mut ierr = 0;
    unsafe {
        if do_shrink {
            iiu_read_reduced(
                im_unit,
                iz,
                array.as_mut_ptr(),
                nx_dim,
                x_ub_start,
                y_ub_start,
                red_fac,
                nx_red,
                ny_red,
                ifilt_type,
                temp.as_mut_ptr(),
                temp.len() as i32,
                &raw mut ierr,
            );
            // `newstack.f90:3539-3543`, whose `write(listString, '(a,i2,a)')`
            // right-justifies the code in two columns.
            if ierr > 0 {
                exit_error(&format!(
                    "Calling irdReduced to read image (error code{ierr:2})"
                ));
            }
        } else {
            iiu_read_binned(
                im_unit,
                iz,
                array.as_mut_ptr(),
                nx_dim,
                ny_dim,
                x_ub_start.round() as i32,
                y_ub_start.round() as i32,
                red_fac.round() as i32,
                nx_red,
                ny_red,
                temp.as_mut_ptr(),
                temp.len() as i32,
                &raw mut ierr,
            );
        }
    }
    // `newstack.f90:3548`: the subroutine itself exits on a read failure, so
    // the message is this routine's, not the caller's.
    if ierr != 0 {
        exit_error("Reading image file");
    }
    if ierr == 0 { Ok(()) } else { Err(ierr) }
}

/// Original `scanSection` (`newstack.f90:3408`) chunking and statistics.
/// The closure is the source's `readBinnedOrReduced` call retained at its
/// call site in the program unit.
pub fn scan_section<F>(
    array: &mut [f32],
    nx: i32,
    ny_needed: i32,
    need_y_first: i32,
    reduction: f32,
    rx_offset: f32,
    ry_offset: f32,
    if_float: i32,
    fix_range_sds: f32,
    mut read: F,
) -> Result<(f32, f32, f32, f32, i32, i32), i32>
where
    F: FnMut(&mut [f32], i32, f32, f32) -> Result<(), i32>,
{
    let max_lines = array.len() as i32 / nx;
    if max_lines <= 0 {
        return Err(1);
    }
    let num_loads = (ny_needed + max_lines - 1) / max_lines;
    let (mut line, mut dmin2, mut dmax2, mut dsum, mut last_lines) =
        (need_y_first, 1.0e30_f32, -1.0e30_f32, 0_f64, 0);
    let (mut sums, mut sds, mut pixels) = (Vec::new(), Vec::new(), Vec::new());
    for load in 1..=num_loads {
        let mut lines = ny_needed / num_loads;
        if load <= ny_needed % num_loads {
            lines += 1;
        }
        let count = nx as usize * lines as usize;
        read(
            &mut array[..count],
            lines,
            rx_offset,
            ry_offset + reduction * line as f32,
        )?;
        let values = &array[..count];
        dmin2 = dmin2.min(values.iter().copied().fold(f32::INFINITY, f32::min));
        dmax2 = dmax2.max(values.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        //
        // accumulate sums for mean and sd if float 2, otherwise just the mean
        //
        let sum = if if_float == 2 || fix_range_sds > 0. {
            // `call iclAvgSd(array, nx, numLines, 1, nx, 1, numLines, tmin2,
            // tmax2, sumLoad(iload), tsumSq, dmean2, sdLoad(iload))`
            // (`newstack.f90:3443-3445`).  `arrayMinMaxMeanSd`
            // (`simplestat.c:270-320`) does not return the plain pixel sum: it
            // accumulates in single precision about a subsampled rough mean
            // and then rebuilds the sum as `nxArea * (nyArea * avg8)`, so
            // summing the pixels here instead gives a different `sumLoad` and
            // a different `sdLoad`.
            let (mut tmin2, mut tmax2) = (0.0_f32, 0.0_f32);
            let (mut sum, mut tsum_sq) = (0.0_f64, 0.0_f64);
            let (mut avg_sec, mut sd_load) = (0.0_f32, 0.0_f32);
            unsafe {
                crate::imod::libcfshr::simplestat::array_min_max_mean_sd(
                    values.as_ptr(),
                    nx,
                    lines,
                    0,
                    nx - 1,
                    0,
                    lines - 1,
                    &raw mut tmin2,
                    &raw mut tmax2,
                    &raw mut sum,
                    &raw mut tsum_sq,
                    &raw mut avg_sec,
                    &raw mut sd_load,
                );
            }
            sds.push(sd_load);
            sum
        } else {
            // `call iclden(...)` then `sumLoad(iload) = (tmean2 * nx) * numLines`:
            // `arrayMinMaxMean` (`simplestat.c:177-199`) keeps the per-line
            // running total in single precision, and the Fortran product of a
            // real*4 mean with integer extents stays in single precision.
            let mut sum_dbl = 0.0_f64;
            for row in values.chunks_exact(nx as usize) {
                let mut sum_tmp = 0.0_f32;
                for value in row {
                    sum_tmp += *value;
                }
                sum_dbl += f64::from(sum_tmp);
            }
            let tmean2 = (sum_dbl / count as f64) as f32;
            sds.push(0.);
            f64::from((tmean2 * nx as f32) * lines as f32)
        };
        dsum += sum;
        sums.push(sum);
        pixels.push(count as f64);
        line += lines;
        last_lines = lines;
    }
    let dmean2 = (dsum / nx as f64 / ny_needed as f64) as f32;
    let sd_sec = if if_float == 2 || fix_range_sds > 0. {
        chunk_sums_to_avgsd(&sums, &sds, &pixels, nx, ny_needed).1
    } else {
        0.
    };
    Ok((dmin2, dmax2, dmean2, sd_sec, line - last_lines, line - 1))
}

/// Program-unit values used by `reallocateIfNeeded` (`newstack.f90:2798`).
#[derive(Clone, Copy, Debug)]
pub struct ReallocateIfNeeded {
    pub physical_memory: f64,
    pub process_in_place: bool,
    pub ft_reduce_fac: f32,
    pub phase_shift: bool,
    pub lim_entered: i32,
    pub nx: i32,
    pub ny: i32,
    pub nx_bin: i32,
    /// `nyBin` (`newstack.f90:2865`), reported alongside `nxBin`.
    pub ny_bin: i32,
    pub ny_needed: i32,
    pub nx_out: i32,
    pub ny_out: i32,
    pub read_shrunk: bool,
    pub read_reduction: f32,
    pub i_binning: i32,
    pub fourier_scaling: bool,
    pub nx_fspad: i32,
    pub ny_fspad: i32,
    pub nx_fcrop_pad: i32,
    pub ny_fcrop_pad: i32,
    pub ft_expand_fac: f32,
    pub noise_pad: bool,
    pub nx_bin_fft: i32,
    pub ny_bin_fft: i32,
    pub lim_to_alloc: usize,
    pub len_temp: usize,
    /// `preSetScaling` (`newstack.f90:78`), which decides at
    /// `newstack.f90:2853` whether a too-big in-place section gives up.
    pub pre_set_scaling: bool,
    /// `idimInOut` (`newstack.f90:53`).  The source leaves it alone for an
    /// entered `-test` pair, so it comes in as well as out.
    pub idim_in_out: usize,
    /// `inPlaceFac` (`newstack.f90:2801`).  Its declaration carries an
    /// initialiser, which makes it `SAVE`: once a section processed in place
    /// drops it to 0.25 it stays there for the rest of the run, and the
    /// reported limit halves with it.
    pub in_place_fac: f32,
    /// `iVerbose` (`newstack.f90:95`), for the reports this routine makes.
    pub i_verbose: i32,
}

/// Original `reallocateIfNeeded` memory arithmetic.  Allocation itself remains
/// `reallocate_array`, as in the two source procedures.
pub fn reallocate_if_needed(values: &mut ReallocateIfNeeded) -> (usize, usize) {
    // Same list-directed `G16.9E2` real editing as the program unit
    // (`libgfortran/io/write.c`, `write_real` with a scale factor of 1).
    let list_real = |value: f32| -> String {
        let magnitude = value.abs();
        let mut exponent = 1_i32;
        if magnitude != 0.0 {
            let scientific = format!("{:.8e}", magnitude);
            exponent = scientific
                .split_once('e')
                .unwrap()
                .1
                .parse::<i32>()
                .unwrap()
                + 1;
        }
        if (0..=9).contains(&exponent) {
            let mut text = format!("{:.*}", (9 - exponent) as usize, value);
            if exponent == 9 {
                text.push('.');
            }
            format!("{text:>12}    ")
        } else {
            let scientific = format!("{:.8e}", value);
            let (mantissa, power) = scientific.split_once('e').unwrap();
            let power = power.parse::<i32>().unwrap();
            format!(
                "{:>16}",
                format!(
                    "{}E{}{:02}",
                    mantissa,
                    if power < 0 { '-' } else { '+' },
                    power.abs()
                )
            )
        }
    };
    // `physicalMem`, `useLimit`, `inPlaceFac` and `defLimit` are all `real*4`
    // (`newstack.f90:117, 2801`), so every step of this rounds to single
    // precision; computing it in `f64` and casting at the end lands an ulp
    // away in the reported megabytes.
    let physical_mem = (values.physical_memory / 4.) as f32;
    let def_limit = 3.75e9_f32;
    let mut use_limit = def_limit;
    if values.process_in_place && values.ft_reduce_fac == 0. && !values.phase_shift {
        values.in_place_fac = 0.25;
    }
    if physical_mem > 0. {
        use_limit = (0.75 * values.in_place_fac * physical_mem)
            .min(physical_mem - 0.25e9)
            .max(0.1e9);
        if use_limit > def_limit {
            use_limit = def_limit.max(0.5 * values.in_place_fac * physical_mem);
        }
    }
    // `newstack.f90:2815-2816`.
    if values.i_verbose > 0 {
        print!(
            " MB of physical memory {}   limit {}\n",
            list_real(physical_mem / 250000.),
            list_real(use_limit / 250000.)
        );
    }
    if values.lim_entered != 1 {
        let mut need_temp = 1_i64;
        if values.read_shrunk {
            let minimum = if values.read_reduction > 32. { 3. } else { 10. };
            // `newstack.f90:2827` is `nx * (ceiling(...) + 20)`: the 20 is
            // inside the parentheses, multiplied by `nx`.
            need_temp = ((values.nx as f32
                * (((minimum + 6.) * values.read_reduction).ceil() + 20.))
                as i64)
                .max((values.nx as i64 * values.ny as i64).min(5_000_000));
        }
        if values.i_binning > 1 {
            need_temp = values.nx as i64 * values.i_binning as i64;
        }
        if values.fourier_scaling && values.nx_fspad > 0 {
            need_temp =
                need_temp.max((values.nx_fcrop_pad as i64 + 2) * values.ny_fcrop_pad as i64);
        }
        if (values.phase_shift || values.fourier_scaling) && values.nx_fspad > 0 && values.noise_pad
        {
            need_temp = need_temp.max(
                2 * i64::from(values.nx_bin_fft.max(values.ny_bin_fft))
                    + i64::from(values.nx_fspad - values.nx_bin_fft)
                    + i64::from(values.ny_fspad - values.ny_bin_fft),
            );
        }
        values.len_temp = need_temp as usize;
        if values.lim_entered == 0 {
            let mut needed = values.nx_bin as i64 * values.ny_needed as i64;
            if (values.phase_shift || values.fourier_scaling) && values.nx_fspad > 0 {
                needed = (values.nx_fspad as i64 + 2) * (values.ny_fspad as i64 + 1);
            }
            if values.nx_out > 0 && values.ny_out > 0 {
                if values.process_in_place
                    && !values.pre_set_scaling
                    && (needed + 2 * values.nx_bin as i64) as f32 > use_limit
                {
                    values.process_in_place = false;
                }
                if values.process_in_place {
                    needed += 2 * values.nx_bin as i64;
                } else if values.ft_expand_fac > 0. {
                    needed += (values.nx_out as i64 * values.ny_out as i64)
                        .max(values.nx_fcrop_pad as i64 * values.ny_fcrop_pad as i64);
                } else {
                    needed += values.nx_out as i64 * values.ny_out as i64;
                }
            }
            // `needDim > useLimit` compares an `integer(kind = 8)` against a
            // `real*4`, so the count goes through single precision and the
            // assignment back truncates toward zero.
            if needed as f32 > use_limit {
                needed = use_limit as i64;
            }
            // `newstack.f90:2865-2866`.  `needDim` is `integer(kind = 8)`
            // (`newstack.f90:2799`) and right justified in 20; `needTemp` is
            // `integer*4` (`newstack.f90:2800`).
            if values.i_verbose > 0 {
                print!(
                    " reallocate sizes: {:>11} {:>11} {:>11} {:>11} {:>20} {:>11}\n",
                    values.nx_out, values.ny_out, values.nx_bin, values.ny_bin, needed, need_temp
                );
            }
            if needed + need_temp > values.lim_to_alloc as i64 {
                values.lim_to_alloc = (needed + need_temp) as usize;
                // `call reallocateArray()` (`newstack.f90:2869`), whose first
                // statement is the report at `newstack.f90:2887`.  The
                // allocation it wraps is the working `Vec`s here, but the
                // report it makes is not.
                if values.i_verbose > 0 {
                    print!(
                        " reallocating array to {}  MB\n",
                        list_real(values.lim_to_alloc as f32 / (1024 * 256) as f32)
                    );
                }
                values.idim_in_out = values.lim_to_alloc - values.len_temp;
            } else {
                values.idim_in_out = needed as usize;
            }
        }
    } else if values.fourier_scaling
        && values.nx_fspad > 0
        && (values.len_temp as i64) < (values.nx_fcrop_pad as i64 + 2) * values.ny_fcrop_pad as i64
    {
        exit_error("Too small a temporary array size entered for Fourier reduction");
    }
    // `newstack.f90:2874-2878`: only an entered `-memory` re-derives this; an
    // entered `-test` pair leaves `idimInOut` at what `newstack.f90:1290-1305`
    // set.
    if values.lim_entered == 2 {
        values.idim_in_out = values.lim_to_alloc - values.len_temp;
    }
    (values.idim_in_out, values.len_temp)
}

/// Original `reallocateArray` (`newstack.f90:2886`).
pub fn reallocate_array(
    array: &mut Vec<f32>,
    lim_to_alloc: usize,
    len_temp: usize,
    lim_if_fail: usize,
    i_verbose: i32,
) -> Result<usize, String> {
    let mut limit = lim_to_alloc;
    // `newstack.f90:2887-2888`.  `limToAlloc / (1024 * 256.)` divides the
    // `integer(kind = 8)` by a `real*4`, so the quotient is `real*4` and is
    // edited as `G16.9E2`.
    if i_verbose > 0 {
        let value = limit as f32 / (1024 * 256) as f32;
        let magnitude = value.abs();
        let mut exponent = 1_i32;
        if magnitude != 0.0 {
            let scientific = format!("{:.8e}", magnitude);
            exponent = scientific
                .split_once('e')
                .unwrap()
                .1
                .parse::<i32>()
                .unwrap()
                + 1;
        }
        let field = if (0..=9).contains(&exponent) {
            let mut text = format!("{:.*}", (9 - exponent) as usize, value);
            if exponent == 9 {
                text.push('.');
            }
            format!("{text:>12}    ")
        } else {
            let scientific = format!("{:.8e}", value);
            let (mantissa, power) = scientific.split_once('e').unwrap();
            let power = power.parse::<i32>().unwrap();
            format!(
                "{:>16}",
                format!(
                    "{}E{}{:02}",
                    mantissa,
                    if power < 0 { '-' } else { '+' },
                    power.abs()
                )
            )
        };
        print!(" reallocating array to {field}  MB\n");
    }
    if limit.saturating_sub(len_temp) < 100 {
        return Err("With achievable memory allocation, the temporary array does not leave enough space for input/output".to_owned());
    }
    if array.try_reserve_exact(limit).is_err() && limit > lim_if_fail {
        limit = lim_if_fail;
        // `newstack.f90:2894-2895`.  `limToAlloc / (1024 * 256)` is an integer
        // division here, so this one is right justified in 20.
        if i_verbose > 0 {
            print!(
                " failed, dropping reallocation to {:>20}  MB\n",
                limit / (1024 * 256)
            );
        }
    }
    let mut replacement = Vec::new();
    replacement
        .try_reserve_exact(limit)
        .map_err(|_| "Reallocating memory for main array".to_owned())?;
    replacement.resize(limit, 0.);
    *array = replacement;
    if limit.saturating_sub(len_temp) < 100 {
        Err("With achievable memory allocation, the temporary array does not leave enough space for input/output".to_owned())
    } else {
        Ok(limit - len_temp)
    }
}

/// Inputs and running totals of original `scaleAndWriteChunk`
/// (`newstack.f90:3221`).
#[derive(Clone, Copy, Debug)]
pub struct ScaleAndWriteChunk {
    pub new_mode: i32,
    pub write_16_bit_mode_for_floats: bool,
    pub scale_factor: f32,
    pub const_add: f32,
    pub optimal_out: f32,
    pub dmin2: f32,
    pub dmax2: f32,
    /// `real*4` in the source (`newstack.f90:92`), so the running total is
    /// rounded back to single precision after every output line.
    pub dmean2: f32,
    pub num_trunc_low: i32,
    pub num_trunc_high: i32,
}

/// Original `scaleAndWriteChunk` scaling/truncation/accumulation.  The writer
/// is `iiuWriteLines` at the source call site.
pub fn scale_and_write_chunk<F>(
    array: &mut [f32],
    nx_out: i32,
    values: &mut ScaleAndWriteChunk,
    mut write: F,
) -> Result<(), i32>
where
    F: FnMut(&[f32]) -> Result<(), i32>,
{
    let (dens_out_min, optimal_out) = if values.new_mode == 1 {
        (-32768., values.optimal_out)
    } else if values.new_mode == 2 && values.write_16_bit_mode_for_floats {
        (-65504., 65504.)
    } else if values.new_mode == 2 {
        (-1.0e30_f32, 1.0e30_f32)
    } else {
        (0., values.optimal_out)
    };
    for line in array.chunks_exact_mut(nx_out as usize) {
        let mut sum = 0_f64;
        for density in line {
            let mut scaled = values.scale_factor * *density + values.const_add;
            if scaled < dens_out_min {
                values.num_trunc_low += 1;
                scaled = dens_out_min;
            } else if scaled > optimal_out {
                values.num_trunc_high += 1;
                scaled = optimal_out;
            }
            *density = scaled;
            sum += f64::from(scaled);
            values.dmin2 = values.dmin2.min(scaled);
            values.dmax2 = values.dmax2.max(scaled);
        }
        values.dmean2 = (f64::from(values.dmean2) + sum) as f32;
    }
    write(array)
}

/// Original `getOffsetEntries` (`newstack.f90:3179`) after PIP has returned
/// its repeated float-array entries.  The PIP adapter belongs in the program
/// unit; this procedure preserves the source's one-or-one-per-section rule.
pub fn get_offset_entries(
    entries: &[Vec<f32>],
    list_total: usize,
    x_offset: &mut [f32],
    y_offset: &mut [f32],
) -> Result<(f32, f32, i32), String> {
    if entries.is_empty() {
        return Ok((0., 0., 0));
    }
    let values: Vec<f32> = entries.iter().flatten().copied().collect();
    if values.len() != 2 && values.len() != 2 * list_total {
        return Err("There must be either one offset or an offset for each section".to_owned());
    }
    for (index, pair) in values.chunks_exact(2).enumerate() {
        x_offset[index] = pair[0];
        y_offset[index] = pair[1];
    }
    let one_or_many = if values.len() == 2 { -1 } else { 1 };
    let x_all = x_offset[0];
    let y_all = y_offset[0];
    if one_or_many <= 0 {
        for index in 0..list_total {
            x_offset[index] = x_all;
            y_offset[index] = y_all;
        }
    }
    Ok((x_all, y_all, one_or_many))
}

/// Original `transferCollections` (`newstack.f90:3265`).  `ind_adoc_out` and
/// the loop indices are the source's 1-based ones; the Fortran wrappers
/// (`adoc_fwrap.c:481, 397, 493`) subtract one before the C entry points, so
/// this does the same at each call.
pub unsafe fn transfer_collections(zvalue_name: &str, ind_adoc_out: i32) -> Result<(), String> {
    let count = adoc_get_num_collections();
    for collection in 1..=count {
        let mut name_ptr = core::ptr::null_mut();
        if adoc_get_collection_name(collection - 1, &raw mut name_ptr) != 0 || name_ptr.is_null() {
            return Err("Getting collection name for transferring other autodoc sections".into());
        }
        let name = std::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned();
        libc::free(name_ptr.cast());
        if name != zvalue_name && name != "T" {
            let name_c = CString::new(name.as_bytes()).map_err(|_| "Invalid collection name")?;
            let count = adoc_get_number_of_sections(name_c.as_ptr());
            for section in 1..=count {
                let mut section_ptr = core::ptr::null_mut();
                if adoc_get_section_name(name_c.as_ptr(), section - 1, &raw mut section_ptr) != 0
                    || section_ptr.is_null()
                {
                    return Err(
                        "Getting section name for transferring other autodoc sections".into(),
                    );
                }
                if adoc_transfer_section(
                    name_c.as_ptr(),
                    section - 1,
                    ind_adoc_out - 1,
                    section_ptr,
                    0,
                ) != 0
                {
                    libc::free(section_ptr.cast());
                    return Err("transferring other autodoc section".into());
                }
                libc::free(section_ptr.cast());
            }
        }
    }
    Ok(())
}

/// Original `openInputFile` (`newstack.f90:3157`).  `ind_in_file` is the
/// source's 1-based `indInFile`, and `need_close1` is the module variable
/// `needClose1`, which the last branch deliberately leaves alone.
pub unsafe fn open_input_file(
    ind_in_file: usize,
    num_vol_read: usize,
    list_volumes: &[i32],
    in_file: &[String],
    need_close1: &mut i32,
) {
    if ind_in_file <= num_vol_read {
        ii_allow_multi_volume(1);
        if list_volumes[ind_in_file - 1] > 1 {
            imopen(11, &in_file[ind_in_file - 1], "RO");
            // `iiuvolumeopen` (`unit_fileio.c:388`) only dereferences its
            // arguments, so the volume index reaches `iiuVolumeOpen`
            // unchanged and `listVolumes` stays 1-based here.
            if iiu_volume_open(1, 11, list_volumes[ind_in_file - 1] - 1) != 0 {
                exit_error("Opening volume in multi-volume file");
            }
            *need_close1 = 11;
        } else {
            imopen(1, &in_file[ind_in_file - 1], "RO");
            *need_close1 = 0;
        }
    } else {
        ii_allow_multi_volume(0);
        imopen(1, &in_file[ind_in_file - 1], "RO");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_math_matches_fortran_coordinates() {
        assert_eq!(
            back_xform(10, 10, [[1., 0.], [0., 1.]], 5., 5., 0., 0., 5, 5),
            (5.0, 5.0)
        );
        assert_eq!(get_reduced_size(100, 2., false, 1), (50, 0.));
        let (average, sd) = chunk_sums_to_avgsd(
            &[3., 7.],
            &[
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            &[2., 2.],
            2,
            2,
        );
        assert_eq!(average, 2.5);
        assert!((sd - 1.2909944).abs() < 1.0e-5);
    }
    #[test]
    fn repacking_and_list_syntax_match_source() {
        let mut packed = [0.; 9];
        irepak2(&mut packed, &[1., 2., 3., 4.], 2, 2, -1, 1, -1, 1, -1.);
        assert_eq!(packed, [-1., -1., -1., -1., 1., 2., -1., 3., 4.]);
        // `getItemsToUse` reads its own option through PIP
        // (`newstack.f90:3333-3344`), so the entry has to be in PIP's table
        // for this to exercise the parsed-list path.
        unsafe {
            crate::imod::libcfshr::parse_params::pip_initialize(1);
            crate::imod::libcfshr::parse_params::pip_add_option(
                c"uselines:UseTransformLines:LIM:".as_ptr(),
            );
            crate::imod::libcfshr::parse_params::pip_next_arg(c"-uselines".as_ptr());
            crate::imod::libcfshr::parse_params::pip_next_arg(c"1-3,0".as_ptr());
        }
        assert_eq!(
            get_items_to_use(
                5,
                &[0, 2],
                c"UseTransformLines",
                "TRANSFORM LINE",
                false,
                0,
                0
            ),
            [1, 2, 3, 0]
        );
        unsafe { crate::imod::libcfshr::parse_params::pip_done() };
    }
    #[test]
    fn lines_needed_keeps_source_centering_and_fill_test() {
        let values = LinesNeededForOutput {
            fourier_scaling: false,
            if_xform: 0,
            nx_bin: 10,
            ny_bin: 10,
            nx_out: 6,
            ny_out: 6,
            xcen: 0.,
            ycen: 0.,
            fprod: [[1., 0., 0.], [0., 1., 0.]],
            max_field_x: 0,
            max_field_y: 0,
            lines_shrink: 0,
        };
        assert_eq!(
            lines_needed_for_output(&values, 0, 5),
            LinesNeededResult {
                iy_in_1: 2,
                iy_in_2: 7,
                need_fill: false,
                in_place: true
            }
        );
    }
    #[test]
    fn scaling_uses_source_linear_mapping() {
        let values = FindScaleFactors {
            if_float: -1,
            rescale: true,
            num_scale_facs: 0,
            bottom_in: 0.,
            bottom_out: 0.,
            optimal_in: 255.,
            optimal_out: 100.,
            dmin_specified: 0.,
            dmax_specified: 0.,
            dmin_in: 0.,
            dmax_in: 10.,
            if_map_range: 0,
            dmap_low: 0.,
            dmap_high: 0.,
            frac_zero: 0.,
            if_mean: 0,
            if_mean_sd_entered: 0,
            entered_mean: 0.,
            entered_sd: 0.,
            shift_mean: 0.,
            shift_min: 0.,
            shift_max: 0.,
            new_mode: 2,
            dsum: 0.,
            dsum_sq: 0.,
            nx_out: 1,
            ny_out: 1,
            scale_factor: 0.,
            const_add: 0.,
            scale_fac: 0.,
            scale_const: 0.,
            float_average: 0.,
            float_sd: 0.,
            zmin: 0.,
            zmax: 0.,
            float_z_margin: 0.,
            opt_float_range: 0.,
            opt_float_min: 0.,
            section_min: 0.,
            section_max: 0.,
            section_intentionally_truncated: false,
        };
        assert_eq!(
            find_scale_factors(&values, 2., 8.),
            ScaleFactors {
                tmp_min: 2.,
                tmp_max: 8.,
                scale_factor: 10.,
                const_add: 0.
            }
        );
    }
    #[test]
    fn scaling_and_write_truncates_at_source_mode_limits() {
        let mut state = ScaleAndWriteChunk {
            new_mode: 0,
            write_16_bit_mode_for_floats: false,
            scale_factor: 2.,
            const_add: 0.,
            optimal_out: 255.,
            dmin2: f32::INFINITY,
            dmax2: f32::NEG_INFINITY,
            dmean2: 0.,
            num_trunc_low: 0,
            num_trunc_high: 0,
        };
        let mut pixels = [-2., 2., 200.];
        scale_and_write_chunk(&mut pixels, 3, &mut state, |_| Ok(())).unwrap();
        assert_eq!(pixels, [0., 4., 255.]);
        assert_eq!((state.num_trunc_low, state.num_trunc_high), (1, 1));
    }
    #[test]
    fn offset_entries_expand_one_entry_as_source_does() {
        let (mut x, mut y) = ([0.; 3], [0.; 3]);
        assert_eq!(
            get_offset_entries(&[vec![2., -3.]], 3, &mut x, &mut y).unwrap(),
            (2., -3., -1)
        );
        assert_eq!(x, [2.; 3]);
        assert_eq!(y, [-3.; 3]);
    }
}
