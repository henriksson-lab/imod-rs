//! Translation of `IMOD/flib/image/newstack.f90`.
#![allow(dead_code)]

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::getbinnedsize::get_binned_size;
use crate::imod::flib::subrs::hvem::rdlist::parselist2;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen};
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::libcfshr::autodoc::{
    ADOC_ZVALUE_NAME, adoc_clear, adoc_get_collection_name, adoc_get_float,
    adoc_get_num_collections, adoc_get_number_of_sections, adoc_get_section_name,
    adoc_open_image_metadata, adoc_set_current, adoc_transfer_section,
};
use crate::imod::libcfshr::b3dutil::{
    extra_is_nbytes_and_flags, override_write_bytes, set_output_type_from_string,
};
use crate::imod::libcfshr::cubinterp::cubinterp;
use crate::imod::libcfshr::linearxforms::xfmult;
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_error, pip_get_float_array, pip_get_integer, pip_get_string,
    pip_number_of_entries, pip_parse_input,
};
use crate::imod::libcfshr::robuststat::rs_mad_median_outliers;
use crate::imod::libiimod::iimage::{
    IIFILE_DEFAULT, IIFILE_MRC, ii_allow_multi_volume, ii_close, ii_open, ii_open_new,
    ii_read_section_float, ii_sync_from_mrc_header, ii_write_header, ii_write_section_float,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_NLABELS, MrcHeader, mrc_head_new, mrc_head_write, mrc_read_extra_header,
    mrc_write_extra_header,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_get_ii_file, iiu_open, iiu_set_position, iiu_volume_open, iiu_write_lines,
};
use crate::imod::libiimod::unit_header::{iiu_create_header, iiu_write_header};
use crate::imod::libiimod::unit_reduced::{iiu_read_binned, iiu_read_reduced};
use std::ffi::CString;
use std::io::BufReader;

/// Source fallback PIP table (`newstack.f90:151`), retained as 74 `@`-separated
/// entries rather than a hand-maintained Rust option list.
const NEWSTACK_OPTIONS: &str = "input:InputFile:FNM:@output:OutputFile:FNM:@fileinlist:FileOfInputs:FN:@fileoutlist:FileOfOutputs:FN:@reverse:ReverseInputFileOrder:I:@split:SplitStartingNumber:I:@append:AppendExtension:CH:@format:FormatOfOutputFile:CH:@compression:HDFCompressionIndex:I:@volumes:VolumesToRead:LI:@3d:Store3DVolumes:I:@chunk:ChunkSizesInXYZ:IT:@mdoc:UseMdocFiles:B:@remove:RemoveForMdocName:CH:@addback:AddBackForMdocName:CH:@pixel:PixelSizeFromMdoc:B:@tilt:TiltAngleFile:FN:@reorder:ReorderByTiltAngle:I:@angle:AngleFileToReorder:FN:@newangle:NewAngleOutputFile:I:@secs:SectionsToRead:LIM:@samesec:SameSectionsToRead:B:@fromone:NumberedFromOne:B:@exclude:ExcludeSections:LI:@twodir:TwoDirectionTiltSeries:B:@skip:SkipSectionIncrement:I:@numout:NumberToOutput:IAM:@replace:ReplaceSections:LI:@blank:BlankOutput:B:@offset:OffsetsInXandY:FAM:@applyfirst:ApplyOffsetsFirst:B:@xform:TransformFile:FN:@uselines:UseTransformLines:LIM:@onexform:OneTransformPerFile:B:@phase:PhaseShiftFFT:B:@rotate:RotateByAngle:F:@expand:ExpandByFactor:F:@shrink:ShrinkByFactor:F:@antialias:AntialiasFilter:I:@bin:BinByFactor:I:@oddeven:AllowOddEvenChange:B:@ftreduce:FourierReduceByFactor:F:@ftexpand:FourierExpandByFactor:F:@noise:NoisePadForFFT:B:@distort:DistortionField:FN:@imagebinned:ImagesAreBinned:F:@fields:UseFields:LIM:@subarea:SubareaOffsetsXandY:FAM:@gradient:GradientFile:FN:@origin:AdjustOrigin:B:@linear:LinearInterpolation:B:@nearest:NearestNeighbor:B:@size:SizeToOutputInXandY:IP:@mode:ModeToOutput:I:@bytes:BytesSignedInOutput:I:@strip:StripExtraHeader:B:@float:FloatDensities:I:@meansd:MeanAndStandardDeviation:FP:@contrast:ContrastBlackWhite:IP:@scale:ScaleMinAndMax:FP:@map:MapFromRange:FP:@multadd:MultiplyAndAdd:FPM:@fixrange:FixRangeIfNeeded:FP:@rfparam:RangeFixingParams:FP:@fill:FillValue:F:@taper:TaperAtFill:IP:@memory:MemoryLimit:I:@test:TestLimits:IP:@megasec:MaxMegaSections:I:@quiet:QuietOutput:B:@print:PrintXYSizeAndExit:B:@verbose:VerboseOutput:I:@param:ParameterFile:PF:@help:usage:B:";

/// Original program `newstack` (`newstack.f90:15`).
///
/// Implements the native MRC stream path: section selection/reversal/blanking
/// and mode conversion use IMOD's own image dispatch, as in the source.
pub fn newstack() {
    // `newstack.f90:303` suppresses the iiunit open/header banner while it
    // probes each input; command-owned diagnostics remain enabled below.
    unsafe { ialprt(false) };
    // `PipReadOrParseOptions` uses this exact fallback table after an autodoc
    // lookup.  The parser core is native; autodoc-option-file loading remains
    // its separate source unit, so command-line fallback starts here.
    let pip_options: Vec<CString> = NEWSTACK_OPTIONS
        .split('@')
        .map(|entry| CString::new(entry).unwrap())
        .collect();
    let mut pip_option_pointers: Vec<*const libc::c_char> =
        pip_options.iter().map(|entry| entry.as_ptr()).collect();
    let pip_arguments: Vec<CString> = std::env::args()
        .map(|argument| CString::new(argument).unwrap())
        .collect();
    let mut pip_argument_pointers: Vec<*mut libc::c_char> = pip_arguments
        .iter()
        .map(|argument| argument.as_ptr().cast_mut())
        .collect();
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    let pip_error = unsafe {
        pip_parse_input(
            pip_argument_pointers.len() as i32,
            pip_argument_pointers.as_mut_ptr(),
            pip_option_pointers.as_mut_ptr(),
            pip_option_pointers.len() as i32,
            &raw mut num_opt_arg,
            &raw mut num_non_opt_arg,
        )
    };
    let mut reverse_value = 0;
    let reverse_classification =
        unsafe { pip_get_boolean(c"reverse".as_ptr(), &raw mut reverse_value) };
    if pip_error != 0 {
        // The source falls back from an autodoc table to this table.  Keep the
        // pre-existing direct stream path live while the remaining PIP types
        // (`FNM`, `FAM`, and `FPM`) are completed in parse_params.
        let mut error = core::ptr::null_mut();
        unsafe { pip_get_error(&raw mut error) };
        if !error.is_null() {
            eprintln!(
                "PIP WARNING: {}",
                unsafe { std::ffi::CStr::from_ptr(error) }.to_string_lossy()
            );
            unsafe { libc::free(error.cast()) };
        }
        eprintln!(
            "PIP reverse classification: error={reverse_classification} value={reverse_value}"
        );
    }
    let mut input_entries = 0;
    let mut output_entries = 0;
    unsafe { pip_number_of_entries(c"InputFile".as_ptr(), &raw mut input_entries) };
    unsafe { pip_number_of_entries(c"OutputFile".as_ptr(), &raw mut output_entries) };
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
        mut transform_lines_option,
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
        None::<String>,
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
    let mut odd_even_ok = 0_i32;
    let mut quiet = false;
    let mut bytes_signed = None::<i32>;
    let mut pixel_from_mdoc = false;
    let mut non_options = Vec::<String>::new();
    let mut words = std::env::args().skip(1);
    while let Some(word) = words.next() {
        let option = word.trim_start_matches('-').to_ascii_lowercase();
        match option.as_str() {
            "input" | "inputfile" => input_names.push(words.next().unwrap_or_default()),
            "output" | "outputfile" => output_names.push(words.next().unwrap_or_default()),
            "format" | "formatofoutputfile" => {
                let Some(value) = words.next() else {
                    eprintln!("ERROR: NEWSTACK - Unrecognized entry for output file format");
                    return;
                };
                let error = set_output_type_from_string(&value);
                if error == -5 {
                    eprintln!("ERROR: NEWSTACK - HDF files are not supported by this IMOD package");
                    return;
                }
                if error == -6 {
                    eprintln!(
                        "ERROR: NEWSTACK - JPEG files are not supported by this IMOD package"
                    );
                    return;
                }
                if error < 0 {
                    eprintln!("ERROR: NEWSTACK - Unrecognized entry for output file format");
                    return;
                }
            }
            "fileinlist" | "fileofinputs" => input_list_name = words.next().unwrap_or_default(),
            "fileoutlist" | "fileofoutputs" => output_list_name = words.next().unwrap_or_default(),
            "reverse" | "reverseinputfileorder" => {
                let Some(value) = words.next() else {
                    eprintln!("ERROR: NEWSTACK - A value is required for -reverse");
                    return;
                };
                let Ok(value) = value.parse::<i32>() else {
                    eprintln!("ERROR: NEWSTACK - Reverse input order must be an integer");
                    return;
                };
                reverse_count = Some(value);
            }
            "blank" | "blankoutput" => blank = true,
            "xform" | "transformfile" => xf_file = words.next().unwrap_or_default(),
            "uselines" | "usetransformlines" => transform_lines_option = words.next(),
            // These values are acquired below through the source PIP FAM APIs.
            // Consume their command-line spelling here only so this direct program
            // path can coexist with the still-untranslated option dispatcher.
            "offset" | "offsetsinxandy" => {
                let _ = words.next();
            }
            "tilt" | "tiltanglefile" => tilt_angle_file = words.next().unwrap_or_default(),
            "pixel" | "pixelsizefrommdoc" => pixel_from_mdoc = true,
            "reorder" | "reorderbytiltangle" => {
                let Some(value) = words.next().and_then(|value| value.parse::<i32>().ok()) else {
                    eprintln!("ERROR: NEWSTACK - Reorder by tilt angle must be an integer");
                    return;
                };
                reorder_by_tilt = value;
            }
            "angle" | "anglefiletoreorder" => {
                angle_file_to_reorder = words.next().unwrap_or_default()
            }
            "newangle" | "newangleoutputfile" => {
                new_angle_output_file = words.next().unwrap_or_default()
            }
            "applyfirst" | "applyoffsetsfirst" => {}
            "origin" | "adjustorigin" => adjust_origin = true,
            "print" | "printxysizeandexit" => print_size_and_exit = true,
            "expand" | "expandbyfactor" => {
                let Some(value) = words.next().and_then(|value| value.parse::<f32>().ok()) else {
                    eprintln!("ERROR: NEWSTACK - Expand factor must be a number");
                    return;
                };
                expand_factor = value;
            }
            "rotate" | "rotatebyangle" => {
                let Some(value) = words.next().and_then(|value| value.parse::<f32>().ok()) else {
                    eprintln!("ERROR: NEWSTACK - Rotation angle must be a number");
                    return;
                };
                rotate_angle = value;
            }
            "bin" | "binbyfactor" => {
                let Some(value) = words.next().and_then(|value| value.parse::<i32>().ok()) else {
                    eprintln!("ERROR: NEWSTACK - Binning factor must be an integer");
                    return;
                };
                bin_factor = value;
            }
            "oddeven" | "allowoddevenchange" => odd_even_ok = 1,
            "quiet" | "quietoutput" => quiet = true,
            "bytes" | "bytessignedinoutput" => {
                bytes_signed = words.next().and_then(|value| value.parse().ok())
            }
            "memory" | "memorylimit" => {
                let _ = words.next();
            }
            "multadd" | "multiplyandadd" => {
                let _ = words.next();
            }
            "scale" | "scaleminandmax" => {
                let _ = words.next();
            }
            "map" | "mapfromrange" => {
                let _ = words.next();
            }
            "contrast" | "contrastblackwhite" => {
                let _ = words.next();
            }
            "float" | "floatdensities" => {
                let _ = words.next();
            }
            "meansd" | "meanandstandarddeviation" => {
                let _ = words.next();
            }
            // Values are acquired below with the matching PIP `FP` calls;
            // consume this direct pass exactly as the other PIP-owned options.
            "fixrange" | "fixrangeifneeded" | "rfparam" | "rangefixingparams" => {
                let _ = words.next();
            }
            "linear" | "linearinterpolation" => {
                linear_entered = true;
                if_linear = 1;
            }
            "nearest" | "nearestneighbor" => {
                nearest_entered = true;
                if_linear = -1;
            }
            "samesec" | "samesectionstoread" => same_sections = true,
            "fromone" | "numberedfromone" => numbered_from_one = true,
            "skip" | "skipsectionincrement" => {
                let Some(value) = words.next().and_then(|value| value.parse::<i32>().ok()) else {
                    eprintln!("ERROR: NEWSTACK - Skip section increment must be an integer");
                    return;
                };
                list_increment = value;
            }
            "exclude" | "excludesections" => {
                let Some(list) = words.next() else {
                    eprintln!("ERROR: NEWSTACK - Missing exclusion list");
                    return;
                };
                let mut parsed = vec![0_i32; 1_000_000];
                let (mut count, mut limit) = (0, 1_000_000);
                if parselist2(&list, &mut parsed, &mut count, &mut limit).is_err() {
                    eprintln!("ERROR: NEWSTACK - Invalid exclusion list: {list}");
                    return;
                }
                excluded_sections.extend_from_slice(&parsed[..count as usize]);
            }
            "mode" | "modetooutput" => mode = words.next().and_then(|value| value.parse().ok()),
            "size" | "sizetooutputinxandy" => {
                let Some(value) = words.next() else {
                    eprintln!("ERROR: NEWSTACK - SizeToOutputInXandY requires two integers");
                    return;
                };
                let values = value
                    .split([',', ' ', '\t'])
                    .filter(|value| !value.is_empty())
                    .filter_map(|value| value.parse::<i32>().ok())
                    .collect::<Vec<_>>();
                if values.len() != 2 {
                    eprintln!("ERROR: NEWSTACK - SizeToOutputInXandY requires two integers");
                    return;
                }
                size_to_output = Some([values[0], values[1]]);
            }
            "numout" | "numbertooutput" => {
                let Some(list) = words.next() else {
                    eprintln!("ERROR: NEWSTACK - Missing output section counts");
                    return;
                };
                for value in list
                    .split([',', ' ', '\t'])
                    .filter(|value| !value.is_empty())
                {
                    let Ok(value) = value.parse::<i32>() else {
                        eprintln!("ERROR: NEWSTACK - Invalid output section count: {list}");
                        return;
                    };
                    num_output_sections.push(value);
                }
            }
            "secs" | "sectionstoread" => {
                let Some(list) = words.next() else {
                    eprintln!("ERROR: NEWSTACK - Missing section list");
                    return;
                };
                let mut sections = vec![0_i32; 1_000_000];
                let (mut count, mut limit) = (0, 1_000_000);
                if parselist2(&list, &mut sections, &mut count, &mut limit).is_err() {
                    eprintln!("ERROR: NEWSTACK - Invalid section list: {list}");
                    return;
                }
                sections.truncate(count as usize);
                section_lists.push(sections);
            }
            "help" | "usage" => {
                println!(
                    "Usage: newstack -input INPUT -output OUTPUT [-secs LIST] [-reverse] [-blank] [-mode MODE]"
                );
                return;
            }
            _ if word.starts_with('-') => {
                eprintln!(
                    "ERROR: NEWSTACK - Option -{option} requires an untranslated IMOD program unit"
                );
                return;
            }
            _ => non_options.push(word),
        }
    }
    if linear_entered && nearest_entered {
        eprintln!("ERROR: NEWSTACK - You cannot enter both -linear and -nearest");
        return;
    }
    if !tilt_angle_file.is_empty() && reorder_by_tilt != 0 {
        eprintln!(
            "ERROR: NEWSTACK - You cannot enter -tilt with angles to insert and -reorder to reorder by angle"
        );
        return;
    }
    if reorder_by_tilt != 0 && output_names.len() > 1 {
        eprintln!("ERROR: NEWSTACK - You cannot use -reorder with more than one output file");
        return;
    }
    if let Some(bytes_signed) = bytes_signed {
        override_write_bytes(bytes_signed);
    }
    // The source treats all non-option arguments except the last as inputs,
    // and the final one as an output.  Its list-file forms are mutually
    // exclusive with these direct forms.
    if !input_list_name.is_empty() && (!input_names.is_empty() || !non_options.is_empty()) {
        eprintln!("ERROR: NEWSTACK - You cannot enter both input files and an input list file");
        return;
    }
    if !output_list_name.is_empty() && (!output_names.is_empty() || !non_options.is_empty()) {
        eprintln!("ERROR: NEWSTACK - You cannot enter both output files and an output list file");
        return;
    }
    if !input_list_name.is_empty() {
        let Ok(text) = std::fs::read_to_string(&input_list_name) else {
            eprintln!("ERROR: NEWSTACK - Opening input file list");
            return;
        };
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let Some(count) = lines.next().and_then(|value| value.parse::<usize>().ok()) else {
            eprintln!(
                "ERROR: NEWSTACK - Input list file must start with a positive number of files"
            );
            return;
        };
        for _ in 0..count {
            let Some(name) = lines.next() else {
                eprintln!("ERROR: NEWSTACK - Reading input file list");
                return;
            };
            input_names.push(name.to_owned());
            // The source list-file form also carries a section list after each
            // filename.  A bare slash represents every source section.
            let Some(list) = lines.next() else {
                eprintln!(
                    "ERROR: NEWSTACK - There must be a readable section list after each filename in list of input files"
                );
                return;
            };
            let mut parsed = vec![0_i32; 1_000_000];
            let (mut count, mut limit) = (0, 1_000_000);
            if parselist2(list, &mut parsed, &mut count, &mut limit).is_err() {
                eprintln!("ERROR: NEWSTACK - Invalid section list in input list");
                return;
            }
            parsed.truncate(count as usize);
            section_lists.push(parsed);
        }
    } else if !non_options.is_empty() {
        input_names.extend(non_options[..non_options.len() - 1].iter().cloned());
        output_names.push(non_options[non_options.len() - 1].clone());
    }
    if !output_list_name.is_empty() {
        let Ok(text) = std::fs::read_to_string(&output_list_name) else {
            eprintln!("ERROR: NEWSTACK - Opening output file list");
            return;
        };
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let Some(count) = lines
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|count| *count > 0)
        else {
            eprintln!(
                "ERROR: NEWSTACK - Output list file must start with a positive number of files"
            );
            return;
        };
        for _ in 0..count {
            let (Some(name), Some(count)) = (
                lines.next(),
                lines.next().and_then(|value| value.parse::<i32>().ok()),
            ) else {
                eprintln!("ERROR: NEWSTACK - Reading output file list");
                return;
            };
            output_names.push(name.to_owned());
            num_output_sections.push(count);
        }
    }
    if input_names.is_empty() || output_names.is_empty() {
        eprintln!("ERROR: NEWSTACK - Input and output files must be entered");
        return;
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
    if input_names.len() == 1 && section_lists.len() > 1 {
        let mut combined = Vec::new();
        for list in section_lists.drain(..) {
            combined.extend(list);
        }
        section_lists.push(combined);
    }
    if section_lists.len() > input_names.len() {
        eprintln!("ERROR: NEWSTACK - Too many section lists");
        return;
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
            eprintln!(
                "ERROR: NEWSTACK - The entry to -reverse is bigger than the number of input files"
            );
            return;
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
        let Ok(file) = std::fs::File::open(&xf_file) else {
            eprintln!("ERROR: NEWSTACK - Opening transform file");
            return;
        };
        match xfrdall2(&mut BufReader::new(file), &mut transforms, 40_000) {
            0 if !transforms.is_empty() => (),
            0 => {
                eprintln!("ERROR: NEWSTACK - The transform file contains no transforms");
                return;
            }
            1 => {
                eprintln!("ERROR: NEWSTACK - Too many transforms in file for transform array");
                return;
            }
            _ => {
                eprintln!("ERROR: NEWSTACK - Reading transform file");
                return;
            }
        }
    }
    unsafe {
        let mut routes = Vec::<(usize, i32)>::new();
        let mut mdoc_pixel_spacing = Vec::<Vec<Option<f32>>>::new();
        let mut first_header: Option<MrcHeader> = None;
        for (file_index, name) in input_names.iter().enumerate() {
            let Ok(name) = CString::new(name.as_bytes()) else {
                eprintln!("ERROR: NEWSTACK - Invalid input file name");
                return;
            };
            // Source opens this preliminary pass through `imopen` and
            // `irdhdr`, before closing the iiunit and reopening it for the
            // streaming loop below (newstack.f90:405-426).
            imopen(1, name.to_str().unwrap_or_default(), "RO");
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
                eprintln!("ERROR: NEWSTACK - Opening input file");
                return;
            }
            let header = std::ptr::read((*ii_file).header.cast::<MrcHeader>());
            iiu_close(1);
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
                    eprintln!("ERROR: NEWSTACK - Input image sizes differ");
                    return;
                }
            } else {
                first_header = Some(std::ptr::read(&header));
            }
            if section_lists[file_index].is_empty() {
                section_lists[file_index].extend(0..header.nz);
            } else if numbered_from_one {
                for section in &mut section_lists[file_index] {
                    *section -= 1;
                }
            }
            // Source walks each list in steps of `max(1, listIncrement)` and
            // only then removes entries in `ExcludeSections`.
            section_lists[file_index] = section_lists[file_index]
                .iter()
                .step_by(list_increment as usize)
                .copied()
                .filter(|section| !excluded_sections.contains(section))
                .collect();
            if !blank
                && section_lists[file_index]
                    .iter()
                    .any(|&section| section < 0 || section >= header.nz)
            {
                eprintln!("ERROR: NEWSTACK - Section number out of bounds");
                return;
            }
            routes.extend(
                section_lists[file_index]
                    .iter()
                    .copied()
                    .map(|section| (file_index, section)),
            );
        }
        // Source `newstack.f90:1006-1024,1668-1683`: with an explicitly
        // supplied angle file, reorder each input file's selected sections
        // in-place by tilt.  The nested swaps (rather than a Rust sort) retain
        // the source's 0.01-degree comparison and tie behavior.
        let mut reordered_angles = Vec::<f32>::new();
        if reorder_by_tilt != 0 {
            if angle_file_to_reorder.is_empty() {
                eprintln!(
                    "ERROR: NEWSTACK - There is no extended header; tilt angles for -reorder must be entered with the -angles option"
                );
                return;
            }
            let Ok(contents) = std::fs::read_to_string(&angle_file_to_reorder) else {
                eprintln!(
                    "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                );
                return;
            };
            let mut angles = Vec::new();
            for line in contents.lines().take(routes.len()) {
                let Ok(value) = line.trim().parse::<f32>() else {
                    eprintln!(
                        "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                    );
                    return;
                };
                angles.push(value);
            }
            if angles.len() < routes.len() {
                eprintln!(
                    "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                );
                return;
            }
            angles.truncate(routes.len());
            let mut start = 0usize;
            for sections in &section_lists {
                let end = start + sections.len();
                for index in start..end.saturating_sub(1) {
                    for other in index + 1..end {
                        if reorder_by_tilt.signum() as f32 * (angles[index] - angles[other]) > 0.01
                        {
                            routes.swap(index, other);
                            angles.swap(index, other);
                        }
                    }
                }
                start = end;
            }
            reordered_angles = angles;
        }
        // `newstack.f90:986-1024`: -tilt reads exactly one angle for every
        // section being written, then the output loop saves them as generic
        // MRC extended-header reals when the input has no extended header.
        // A leading period is relative to the first input basename in the
        // source (e.g. input.mrc with -tilt .tlt reads input.tlt).
        let mut inserted_tilts = Vec::<f32>::new();
        if !tilt_angle_file.is_empty() {
            if tilt_angle_file.starts_with('.') {
                let end = input_names[0].rfind('.').unwrap_or(input_names[0].len());
                tilt_angle_file = format!("{}{}", &input_names[0][..end], tilt_angle_file);
            }
            let Ok(contents) = std::fs::read_to_string(&tilt_angle_file) else {
                eprintln!(
                    "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                );
                return;
            };
            for line in contents.lines().take(routes.len()) {
                let Some(word) = line.split_whitespace().next() else {
                    eprintln!(
                        "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                    );
                    return;
                };
                let Ok(value) = word.parse::<f32>() else {
                    eprintln!(
                        "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                    );
                    return;
                };
                inserted_tilts.push(value);
            }
            if inserted_tilts.len() < routes.len() {
                eprintln!(
                    "ERROR: NEWSTACK - Reading tilt angle file: it must have as many lines as sections being written"
                );
                return;
            }
        }
        let mut float_densities = 0_i32;
        let float_entered = pip_get_integer(c"FloatDensities".as_ptr(), &mut float_densities) == 0;
        if float_entered && !(1..=4).contains(&float_densities) {
            eprintln!(
                "ERROR: NEWSTACK - You must use -contrast or -scale instead of a negative -float entry"
            );
            return;
        }
        let mut mean_sd = [0.0_f32; 2];
        let mut mean_sd_numbers = 0_i32;
        let mean_sd_entered = pip_get_float_array(
            c"MeanAndStandardDeviation".as_ptr(),
            mean_sd.as_mut_ptr(),
            &mut mean_sd_numbers,
            2,
        ) == 0;
        if mean_sd_entered {
            if mean_sd_numbers != 2 || (float_entered && float_densities != 2) {
                eprintln!(
                    "ERROR: NEWSTACK - You cannot use -meansd with any scaling option except -float 2"
                );
                return;
            }
            float_densities = 2;
        }
        let mut scale_limits = [0.0_f32; 2];
        let mut scale_numbers = 0_i32;
        let scale_entered = pip_get_float_array(
            c"ScaleMinAndMax".as_ptr(),
            scale_limits.as_mut_ptr(),
            &mut scale_numbers,
            2,
        ) == 0;
        let header = first_header.unwrap();
        if expand_factor < 0.0 {
            eprintln!("ERROR: NEWSTACK - Expand factor must be positive");
            return;
        }
        if bin_factor <= 0 {
            eprintln!("ERROR: NEWSTACK - Binning factor must be a positive number");
            return;
        }
        while rotate_angle > 180.01 || rotate_angle < -180.01 {
            rotate_angle -= 360.0_f32.copysign(rotate_angle);
        }
        if expand_factor > 0.0 || rotate_angle != 0.0 {
            let factor = expand_factor.max(1.0);
            let angle = rotate_angle.to_radians();
            let expansion = [
                angle.cos() * factor,
                -angle.sin() * factor,
                angle.sin() * factor,
                angle.cos() * factor,
                0.0,
                0.0,
            ];
            if transforms.is_empty() {
                transforms.push(expansion);
            } else {
                for transform in &mut transforms {
                    let mut product = [0.0_f32; 6];
                    xfmult(transform, &expansion, &mut product);
                    *transform = product;
                }
            }
        }
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
        if float_densities > 1 {
            let mut scan_input = usize::MAX;
            for &(input_index, section) in &routes {
                if scan_input != input_index {
                    if scan_input != usize::MAX {
                        iiu_close(1);
                    }
                    let Ok(scan_name) = CString::new(input_names[input_index].as_bytes()) else {
                        eprintln!("ERROR: NEWSTACK - Invalid input file name");
                        return;
                    };
                    if iiu_open(1, scan_name.as_ptr(), c"RO".as_ptr()) != 0 {
                        eprintln!("ERROR: NEWSTACK - Opening input file");
                        return;
                    }
                    scan_input = input_index;
                }
                let mut scan_array = vec![0.0_f32; header.nx as usize * header.ny as usize];
                let mut scan_temp = vec![0.0_f32; header.nx as usize];
                let Ok((dmin, dmax, dmean, sd, _load_start, _load_end)) = scan_section(
                    &mut scan_array,
                    header.nx,
                    header.ny,
                    0,
                    1.0,
                    0.0,
                    0.0,
                    float_densities,
                    0.0,
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
                    eprintln!("ERROR: NEWSTACK - Reading image file");
                    iiu_close(1);
                    return;
                };
                sec_mean.push(dmean);
                sec_mins.push(dmin);
                sec_maxes.push(dmax);
                sec_sds.push(sd);
            }
            if scan_input != usize::MAX {
                iiu_close(1);
            }
        }
        let mut float_shift_constants = Vec::<f32>::new();
        let mut float_shift_factor = 1.0_f32;
        if (float_densities == 3 || float_densities == 4) && !sec_mean.is_empty() {
            let diff_min_mean = sec_mins
                .iter()
                .zip(&sec_mean)
                .map(|(&minimum, &mean)| minimum - mean)
                .fold(0.0_f32, f32::min);
            let _diff_max_mean = sec_maxes
                .iter()
                .zip(&sec_mean)
                .map(|(&maximum, &mean)| maximum - mean)
                .fold(0.0_f32, f32::max);
            let grand_mean = sec_mean.iter().sum::<f32>() / sec_mean.len() as f32;
            let shift_min = (grand_mean + diff_min_mean).max(0.0);
            let shift_mean = shift_min - diff_min_mean;
            if float_densities == 4 {
                let shift_max = shift_mean + _diff_max_mean;
                if scale_limits[1] != scale_limits[0] && shift_max != shift_min {
                    float_shift_factor =
                        (scale_limits[1] - scale_limits[0]) / (shift_max - shift_min);
                }
                float_shift_constants.extend(sec_mean.iter().map(|mean| {
                    scale_limits[0] + float_shift_factor * (shift_mean - mean - shift_min)
                }));
            } else {
                float_shift_constants.extend(sec_mean.iter().map(|mean| shift_mean - mean));
            }
        }
        let mut float2_zmin = 0.0_f32;
        let mut float2_zmax = 0.0_f32;
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
            float2_zmin = zmins
                .iter()
                .zip(&min_outliers)
                .filter(|&(_, &outlier)| outlier >= 0.0)
                .map(|(&z, _)| z)
                .fold(f32::INFINITY, f32::min);
            float2_zmax = zmaxs
                .iter()
                .zip(&max_outliers)
                .filter(|&(_, &outlier)| outlier <= 0.0)
                .map(|(&z, _)| z)
                .fold(f32::NEG_INFINITY, f32::max);
        }
        let transform_lines = if transforms.is_empty() {
            Vec::new()
        } else {
            let default_lines = routes.iter().map(|route| route.1).collect::<Vec<_>>();
            let Ok(mut lines) = get_items_to_use(
                transforms.len() as i32,
                &default_lines,
                transform_lines_option.as_deref(),
                false,
                input_names.len() as i32,
                0,
            ) else {
                eprintln!("ERROR: NEWSTACK - TRANSFORM LINE number out of bounds");
                return;
            };
            if lines.len() == 1 {
                lines.resize(routes.len(), lines[0]);
            }
            if lines.len() != routes.len() {
                eprintln!(
                    "ERROR: NEWSTACK - Specified # of transform lines does not match # of sections"
                );
                return;
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
                eprintln!("ERROR: NEWSTACK - Getting offset entries");
                return;
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
            eprintln!(
                "ERROR: NEWSTACK - There must be either one offset or an offset for each section"
            );
            return;
        }
        let mut apply_first = 0_i32;
        pip_get_boolean(c"ApplyOffsetsFirst".as_ptr(), &mut apply_first);
        let mut mult_add_count = 0_i32;
        pip_number_of_entries(c"MultiplyAndAdd".as_ptr(), &mut mult_add_count);
        if mult_add_count != 0 && mult_add_count != 1 && mult_add_count != input_names.len() as i32
        {
            eprintln!(
                "ERROR: NEWSTACK - You must enter -multadd either once or once per input file"
            );
            return;
        }
        let mut scale_factors = Vec::<[f32; 2]>::new();
        for _ in 0..mult_add_count {
            let mut factor_and_add = [0.0_f32; 2];
            let mut number = 0_i32;
            if pip_get_float_array(
                c"MultiplyAndAdd".as_ptr(),
                factor_and_add.as_mut_ptr(),
                &mut number,
                2,
            ) != 0
                || number != 2
            {
                eprintln!("ERROR: NEWSTACK - Getting multiply and add entries");
                return;
            }
            scale_factors.push(factor_and_add);
        }
        if scale_entered && scale_numbers != 2 {
            eprintln!("ERROR: NEWSTACK - Getting scale min and max entries");
            return;
        }
        if scale_entered && !scale_factors.is_empty() {
            eprintln!("ERROR: NEWSTACK - The -scale and -multadd options are mutually exclusive");
            return;
        }
        let mut contrast_limits = [0.0_f32, 255.0];
        let mut contrast_numbers = 0_i32;
        let contrast_entered = pip_get_float_array(
            c"ContrastBlackWhite".as_ptr(),
            contrast_limits.as_mut_ptr(),
            &mut contrast_numbers,
            2,
        ) == 0;
        if contrast_entered && contrast_numbers != 2 {
            eprintln!("ERROR: NEWSTACK - Getting contrast black and white entries");
            return;
        }
        if contrast_entered && (scale_entered || !scale_factors.is_empty()) {
            eprintln!(
                "ERROR: NEWSTACK - The -scale, -contrast, -multadd, and -float options are mutually exclusive"
            );
            return;
        }
        if mean_sd_entered && (scale_entered || contrast_entered || !scale_factors.is_empty()) {
            eprintln!(
                "ERROR: NEWSTACK - You cannot use -meansd with any scaling option except -float 2"
            );
            return;
        }
        if contrast_entered {
            contrast_limits[1] = contrast_limits[1].max(contrast_limits[0] + 1.0);
            scale_limits[0] =
                -contrast_limits[0] * 255.0 / (contrast_limits[1] - contrast_limits[0]);
            scale_limits[1] = scale_limits[0] + 65025.0 / (contrast_limits[1] - contrast_limits[0]);
        }
        let mut map_limits = [0.0_f32; 2];
        let mut map_numbers = 0_i32;
        let map_entered = pip_get_float_array(
            c"MapFromRange".as_ptr(),
            map_limits.as_mut_ptr(),
            &mut map_numbers,
            2,
        ) == 0;
        if map_entered && map_numbers != 2 {
            eprintln!("ERROR: NEWSTACK - Getting map from range entries");
            return;
        }
        if map_entered && !(scale_entered || contrast_entered) {
            eprintln!("ERROR: NEWSTACK - You can use -map only with -scale or -contrast");
            return;
        }
        let range_scale_entered = scale_entered || contrast_entered;
        if float_densities == 4 && !scale_entered {
            eprintln!("ERROR: NEWSTACK - You must enter -scale with -float 4");
            return;
        }
        if float_densities == 4 && map_entered {
            eprintln!("ERROR: NEWSTACK - You cannot use -map with -float 4");
            return;
        }
        if float_entered && range_scale_entered && float_densities != 4 {
            eprintln!(
                "ERROR: NEWSTACK - The -scale, -contrast, -multadd, and -float options are mutually exclusive"
            );
            return;
        }
        // Source `FixRangeIfNeeded` acquisition and its legality checks
        // (`newstack.f90:918-936`).  Range correction itself is considered
        // only after the source has established whether interpolation is in
        // use; the source explicitly cancels it for a plain copy.
        let mut fix_range = [0.0_f32, 1.0_f32];
        let mut fix_range_numbers = 0_i32;
        let fix_range_entered = pip_get_float_array(
            c"FixRangeIfNeeded".as_ptr(),
            fix_range.as_mut_ptr(),
            &mut fix_range_numbers,
            2,
        ) == 0;
        if fix_range_entered && fix_range_numbers != 2 {
            eprintln!("ERROR: NEWSTACK - Getting fix range entries");
            std::process::exit(1);
        }
        let mut memory_limit_mb = 0_i32;
        let memory_limit_entered =
            pip_get_integer(c"MemoryLimit".as_ptr(), &mut memory_limit_mb) == 0;
        let mut output_mode = mode.unwrap_or(header.mode);
        if !matches!(output_mode, 0 | 1 | 2 | 6 | 12) {
            eprintln!("ERROR: NEWSTACK - Mode of output must be 0, 1, 2, 6, or 12");
            return;
        }
        if fix_range_entered {
            if float_densities != 0 || range_scale_entered || !scale_factors.is_empty() {
                eprintln!("ERROR: NEWSTACK - You cannot enter -fixrange with any scaling options");
                std::process::exit(1);
            }
            if fix_range[0] != 0.0 && fix_range[0] < 2.0 {
                eprintln!("ERROR: NEWSTACK - The entry for -fixrange must be at least 2");
                std::process::exit(1);
            }
            if input_names.len() > 1 {
                eprintln!(
                    "ERROR: NEWSTACK - You cannot use -fixrange with more than one input file"
                );
                std::process::exit(1);
            }
            if output_mode != header.mode {
                eprintln!("ERROR: NEWSTACK - You cannot use -fixrange if you enter an output mode");
                std::process::exit(1);
            }
        }
        if output_names.len() == 1 && num_output_sections.is_empty() {
            num_output_sections.push(routes.len() as i32);
        } else if output_names.len() == routes.len() && num_output_sections.is_empty() {
            num_output_sections.resize(output_names.len(), 1);
        }
        if num_output_sections.len() != output_names.len()
            || num_output_sections.iter().sum::<i32>() != routes.len() as i32
        {
            eprintln!("ERROR: NEWSTACK - Number of input and output sections does not match");
            return;
        }
        // Source `SizeToOutputInXandY` defaulting (`newstack.f90:676-690,
        // 1693-1702`): absent or nonpositive axes retain the input extent.
        let size_x = size_to_output.map(|size| size[0]).filter(|size| *size > 0);
        let size_y = size_to_output.map(|size| size[1]).filter(|size| *size > 0);
        let factor = expand_factor.max(1.0);
        let transpose = (rotate_angle.abs() - 90.0).abs() < 40.0;
        let (bin_nx, _) = get_binned_size(header.nx, bin_factor, odd_even_ok);
        let (bin_ny, _) = get_binned_size(header.ny, bin_factor, odd_even_ok);
        let (output_nx, output_ny) = if size_x.is_none() && size_y.is_none() && bin_factor > 1 {
            (bin_nx, bin_ny)
        } else if size_x.is_none() && size_y.is_none() && transpose {
            (
                (header.ny as f32 * factor).round() as i32,
                (header.nx as f32 * factor).round() as i32,
            )
        } else {
            (
                size_x.unwrap_or_else(|| (header.nx as f32 * factor).round() as i32),
                size_y.unwrap_or_else(|| (header.ny as f32 * factor).round() as i32),
            )
        };
        if print_size_and_exit {
            println!(" Output size: {output_nx:12}{output_ny:12}");
            return;
        }
        // The remaining `FixRangeIfNeeded` execution block from
        // newstack.f90:1284-1510.  The source cancels it unless an operation
        // interpolates pixels; this direct route currently has affine
        // transforms as its interpolation operation.
        if fix_range_entered && fix_range[0] > 0.0 && !transforms.is_empty() {
            let mut range_params = [10.0_f32, 1.2_f32];
            let mut range_param_numbers = 0_i32;
            if pip_get_float_array(
                c"RangeFixingParams".as_ptr(),
                range_params.as_mut_ptr(),
                &mut range_param_numbers,
                2,
            ) == 0
                && range_param_numbers != 2
            {
                eprintln!("ERROR: NEWSTACK - Getting range fixing parameters");
                std::process::exit(1);
            }
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
                    }
                    let Ok(scan_name) = CString::new(input_names[input_index].as_bytes()) else {
                        eprintln!("ERROR: NEWSTACK - Invalid input file name");
                        return;
                    };
                    if iiu_open(1, scan_name.as_ptr(), c"RO".as_ptr()) != 0 {
                        eprintln!("ERROR: NEWSTACK - Opening input file");
                        return;
                    }
                    scan_input = input_index;
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
                    eprintln!("ERROR: NEWSTACK - Reading image file");
                    if scan_input != usize::MAX {
                        iiu_close(1);
                    }
                    return;
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
            }
            if scan_input != usize::MAX {
                iiu_close(1);
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
                    println!("  because SD of values is below {:6.1}", range_params[0]);
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
        let mut array = vec![0.0_f32; header.nx as usize * header.ny as usize];
        // Source `time`, `b3dDate`, and format 302 at newstack.f90:1518,
        // 1891-1898.  This direct stream path has no truncation title text.
        let mut title = [b' '; 80];
        title[..23].copy_from_slice(b"NEWSTACK: Images copied");
        if !xf_file.is_empty() {
            title[23..36].copy_from_slice(b", transformed");
        }
        if range_scale_entered {
            title[36..54].copy_from_slice(b", densities scaled");
        } else if float_densities == 1 {
            title[36..54].copy_from_slice(b", floated to range");
        } else if float_densities == 2 {
            title[36..54].copy_from_slice(b", floated to means");
        } else if float_densities == 3 {
            title[36..54].copy_from_slice(b",  shifted to mean");
        } else if float_densities == 4 {
            title[36..54].copy_from_slice(b", mean shift&scale");
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
        title[67..75]
            .copy_from_slice(unsafe { std::slice::from_raw_parts(time.as_ptr().cast(), 8) });
        let mut route_index = 0usize;
        let mut num_trunc_low = 0_i32;
        let mut num_trunc_high = 0_i32;
        let mut active_input_index = usize::MAX;
        let mut active_input_file: *mut crate::imod::libiimod::iimage::ImodImageFile =
            std::ptr::null_mut();
        let mut active_chunk_input = usize::MAX;
        for (output_index, name) in output_names.iter().enumerate() {
            let output_tilt_start = route_index;
            let Ok(name) = CString::new(name.as_bytes()) else {
                eprintln!("ERROR: NEWSTACK - Invalid output file name");
                if !active_input_file.is_null() {
                    ii_close(active_input_file);
                }
                return;
            };
            let chunk_limit = memory_limit_mb.max(0) as usize * 1024 * 256;
            let chunked_affine = memory_limit_entered
                && !transforms.is_empty()
                && chunk_limit > 0
                && 2 * header.nx as usize * header.ny as usize > chunk_limit;
            let mut non_mrc_output_header: MrcHeader = std::mem::zeroed();
            let out_file = if chunked_affine {
                if iiu_open(2, name.as_ptr(), c"NEW".as_ptr()) != 0 {
                    eprintln!("ERROR: NEWSTACK - Opening output file");
                    return;
                }
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
                    eprintln!("ERROR: NEWSTACK - Creating output header");
                    iiu_close(2);
                    return;
                }
                let chunk_header = (*chunk_file).header.cast::<MrcHeader>();
                std::ptr::copy_nonoverlapping(&header, chunk_header, 1);
                mrc_head_new(
                    &mut *chunk_header,
                    output_nx,
                    output_ny,
                    num_output_sections[output_index],
                    output_mode,
                );
                (*chunk_header).nxstart = header.nxstart;
                (*chunk_header).nystart = header.nystart;
                (*chunk_header).nzstart = header.nzstart;
                (*chunk_header).mapc = header.mapc;
                (*chunk_header).mapr = header.mapr;
                (*chunk_header).maps = header.maps;
                (*chunk_header).alpha = header.alpha;
                (*chunk_header).beta = header.beta;
                (*chunk_header).gamma = header.gamma;
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
                (*chunk_header).xlen = (*chunk_header).mx as f32 * header.xlen / header.mx as f32;
                (*chunk_header).ylen = (*chunk_header).my as f32 * header.ylen / header.my as f32;
                (*chunk_header).zlen = (*chunk_header).mz as f32 * header.zlen / header.mz as f32;
                if pixel_from_mdoc {
                    let (input_index, input_section) = routes[route_index];
                    if let Some(spacing) = mdoc_pixel_spacing[input_index][input_section as usize] {
                        (*chunk_header).xlen =
                            (*chunk_header).mx as f32 * spacing / expand_factor.max(1.0);
                        (*chunk_header).ylen =
                            (*chunk_header).my as f32 * spacing / expand_factor.max(1.0);
                        (*chunk_header).zlen = (*chunk_header).mz as f32 * spacing;
                    }
                }
                if adjust_origin {
                    let first_section = routes[route_index].1;
                    let x_center = if transforms.is_empty() {
                        x_offsets[route_index].round()
                    } else {
                        x_offsets[route_index]
                    };
                    let y_center = if transforms.is_empty() {
                        y_offsets[route_index].round()
                    } else {
                        y_offsets[route_index]
                    };
                    let x_shift = if transforms.is_empty() {
                        (header.nx / 2 - output_nx / 2) as f32 + x_center
                    } else {
                        header.nx as f32 / 2.0 + x_center - output_nx as f32 / 2.0
                    };
                    let y_shift = if transforms.is_empty() {
                        (header.ny / 2 - output_ny / 2) as f32 + y_center
                    } else {
                        header.ny as f32 / 2.0 + y_center - output_ny as f32 / 2.0
                    };
                    (*chunk_header).xorg -= x_shift * header.xlen / header.mx as f32;
                    (*chunk_header).yorg -= y_shift * header.ylen / header.my as f32;
                    (*chunk_header).zorg -= first_section as f32 * header.zlen / header.mz as f32;
                }
                ii_sync_from_mrc_header(chunk_file, chunk_header);
                if iiu_write_header(
                    2,
                    title.as_mut_ptr().cast(),
                    0,
                    header.amin,
                    header.amax,
                    header.amean,
                ) != 0
                {
                    eprintln!("ERROR: NEWSTACK - Writing output header");
                    iiu_close(2);
                    return;
                }
                std::ptr::null_mut()
            } else {
                let out_file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
                if out_file.is_null() {
                    eprintln!("ERROR: NEWSTACK - Opening output file");
                    if !active_input_file.is_null() {
                        ii_close(active_input_file);
                    }
                    return;
                }
                let out_header = if (*out_file).file == IIFILE_MRC {
                    (*out_file).header.cast::<MrcHeader>()
                } else {
                    &mut non_mrc_output_header
                };
                std::ptr::copy_nonoverlapping(&header, out_header, 1);
                mrc_head_new(
                    &mut *out_header,
                    output_nx,
                    output_ny,
                    num_output_sections[output_index],
                    output_mode,
                );
                // `iiuTransHeader`, `iiuAltSize`, `iiuAltSample`, and
                // `iiuAltCell` in newstack.f90:1814-1852 retain the input
                // geometry while changing the output sampling.  `mrc_head_new`
                // establishes the output header's storage fields above, so
                // restore the source-retained geometry explicitly here.
                (*out_header).nxstart = header.nxstart;
                (*out_header).nystart = header.nystart;
                (*out_header).nzstart = header.nzstart;
                (*out_header).mapc = header.mapc;
                (*out_header).mapr = header.mapr;
                (*out_header).maps = header.maps;
                (*out_header).imod_stamp = header.imod_stamp;
                (*out_header).imod_flags = header.imod_flags;
                (*out_header).alpha = header.alpha;
                (*out_header).beta = header.beta;
                (*out_header).gamma = header.gamma;
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
                (*out_header).xlen = (*out_header).mx as f32 * header.xlen / header.mx as f32;
                (*out_header).ylen = (*out_header).my as f32 * header.ylen / header.my as f32;
                (*out_header).zlen = (*out_header).mz as f32 * header.zlen / header.mz as f32;
                if pixel_from_mdoc {
                    let (input_index, input_section) = routes[route_index];
                    if let Some(spacing) = mdoc_pixel_spacing[input_index][input_section as usize] {
                        (*out_header).xlen =
                            (*out_header).mx as f32 * spacing / expand_factor.max(1.0);
                        (*out_header).ylen =
                            (*out_header).my as f32 * spacing / expand_factor.max(1.0);
                        (*out_header).zlen = (*out_header).mz as f32 * spacing;
                    }
                }
                if !inserted_tilts.is_empty() && header.next == 0 && (*out_file).file == IIFILE_MRC
                {
                    (*out_header).nint = 0;
                    (*out_header).nreal = 1;
                    (*out_header).next = 4 * num_output_sections[output_index];
                    (*out_header).header_size = 1024 + (*out_header).next;
                } else if header.next > 0 && (*out_file).file == IIFILE_MRC {
                    // `newstack.f90:1907-1935`: retain a conventional MRC
                    // extended header, one source record for every selected
                    // output section.  The record bytes themselves are copied
                    // below in the section loop, where the selected input Z is
                    // known; the header must reserve their space before pixel
                    // writes begin.
                    let bytes_per_section = header.next / header.nz;
                    (*out_header).nint = header.nint;
                    (*out_header).nreal = header.nreal;
                    (*out_header).ext_type = header.ext_type;
                    (*out_header).next = bytes_per_section * num_output_sections[output_index];
                    (*out_header).header_size = 1024 + (*out_header).next;
                }
                if adjust_origin {
                    let first_section = routes[route_index].1;
                    let x_center = if transforms.is_empty() {
                        x_offsets[route_index].round()
                    } else {
                        x_offsets[route_index]
                    };
                    let y_center = if transforms.is_empty() {
                        y_offsets[route_index].round()
                    } else {
                        y_offsets[route_index]
                    };
                    let x_shift = if transforms.is_empty() {
                        (header.nx / 2 - output_nx / 2) as f32 + x_center
                    } else {
                        header.nx as f32 / 2.0 + x_center - output_nx as f32 / 2.0
                    };
                    let y_shift = if transforms.is_empty() {
                        (header.ny / 2 - output_ny / 2) as f32 + y_center
                    } else {
                        header.ny as f32 / 2.0 + y_center - output_ny as f32 / 2.0
                    };
                    (*out_header).xorg -= x_shift * header.xlen / header.mx as f32;
                    (*out_header).yorg -= y_shift * header.ylen / header.my as f32;
                    (*out_header).zorg -= first_section as f32 * header.zlen / header.mz as f32;
                }
                if blank {
                    // Source's blank-section branch has a constant zero
                    // output and propagates its zero min/max/mean to the
                    // final output header (`newstack.f90:2023, 3038`).
                    (*out_header).amin = 0.0;
                    (*out_header).amax = 0.0;
                    (*out_header).amean = 0.0;
                }
                // `iiuTransHeader` retains source titles, and the final
                // `iiuWriteHeader(..., 1, ...)` appends this title (or replaces
                // the final slot at the source limit).  `mrc_head_new` above
                // initializes storage fields and clears labels, so restore that
                // explicitly before applying the source title operation.
                (*out_header).labels = header.labels;
                (*out_header).nlabl = header.nlabl;
                (*out_header).nlabl = ((*out_header).nlabl + 1).min(MRC_NLABELS as i32);
                let title_index = ((*out_header).nlabl - 1) as usize;
                (&mut (*out_header).labels[title_index])[..80].copy_from_slice(&title);
                ii_sync_from_mrc_header(out_file, out_header);
                (*out_file).llx = 0;
                (*out_file).lly = 0;
                (*out_file).llz = 0;
                (*out_file).urx = output_nx - 1;
                (*out_file).ury = output_ny - 1;
                (*out_file).urz = num_output_sections[output_index] - 1;
                if ((*out_file).file == IIFILE_MRC
                    && mrc_head_write((*out_file).fp, out_header) != 0)
                    || ((*out_file).file != IIFILE_MRC && ii_write_header(out_file) != 0)
                {
                    eprintln!("ERROR: NEWSTACK - Writing output header");
                    ii_close(out_file);
                    if !active_input_file.is_null() {
                        ii_close(active_input_file);
                    }
                    return;
                }
                out_file
            };
            // `dmin`, `dmax`, and `dmean` are accumulated for this output
            // file exactly where the source accumulates section `dmin2`,
            // `dmax2`, and `dmean2` before its final `iiuWriteHeader`.
            let mut dmin = f32::INFINITY;
            let mut dmax = f32::NEG_INFINITY;
            let mut dsum = 0.0_f64;
            let mut output_extra_data = Vec::<u8>::new();
            for out_section in 0..num_output_sections[output_index] as usize {
                let (input_index, in_section) = routes[route_index];
                route_index += 1;
                if chunked_affine && !blank {
                    let Some(transform) = transforms.get(transform_lines[route_index - 1] as usize)
                    else {
                        eprintln!(
                            "ERROR: NEWSTACK - TRANSFORM LINE number out of bounds: {in_section}"
                        );
                        iiu_close(2);
                        return;
                    };
                    let offset_index = route_index - 1;
                    let mut fprod = *transform;
                    if bin_factor > 1 {
                        fprod[4] /= bin_factor as f32;
                        fprod[5] /= bin_factor as f32;
                    }
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
                    } else {
                        fprod[4] -= x_offsets[offset_index];
                        fprod[5] -= y_offsets[offset_index];
                    }
                    if active_chunk_input != input_index {
                        if active_chunk_input != usize::MAX {
                            iiu_close(1);
                        }
                        let Ok(input_name) = CString::new(input_names[input_index].as_bytes())
                        else {
                            eprintln!("ERROR: NEWSTACK - Invalid input file name");
                            iiu_close(2);
                            return;
                        };
                        if iiu_open(1, input_name.as_ptr(), c"RO".as_ptr()) != 0 {
                            eprintln!("ERROR: NEWSTACK - Opening input file");
                            iiu_close(2);
                            return;
                        }
                        active_chunk_input = input_index;
                    }
                    let affine = LinesNeededForOutput {
                        fourier_scaling: false,
                        if_xform: 1,
                        nx_bin: if bin_factor > 1 { bin_nx } else { header.nx },
                        ny_bin: if bin_factor > 1 { bin_ny } else { header.ny },
                        nx_out: output_nx,
                        ny_out: output_ny,
                        xcen: 0.0,
                        ycen: 0.0,
                        fprod: [
                            [fprod[0], fprod[1], fprod[4]],
                            [fprod[2], fprod[3], fprod[5]],
                        ],
                        max_field_x: 0,
                        max_field_y: 0,
                        lines_shrink: 0,
                    };
                    let mut line_out_start = 0_i32;
                    while line_out_start < output_ny {
                        let mut line_out_end = (line_out_start
                            + (chunk_limit
                                / (2 * if bin_factor > 1 { bin_nx } else { header.nx } as usize))
                                .max(1) as i32
                            - 1)
                        .min(output_ny - 1);
                        let (needed, input_lines) = loop {
                            let needed =
                                lines_needed_for_output(&affine, line_out_start, line_out_end);
                            let input_lines = needed.iy_in_2 - needed.iy_in_1 + 1;
                            let output_lines = line_out_end - line_out_start + 1;
                            if input_lines as usize
                                * if bin_factor > 1 { bin_nx } else { header.nx } as usize
                                + output_lines as usize * output_nx as usize
                                <= chunk_limit
                            {
                                break (needed, input_lines);
                            }
                            if line_out_end == line_out_start {
                                eprintln!("ERROR: NEWSTACK - Input image too large for array.");
                                iiu_close(2);
                                return;
                            }
                            line_out_end -= 1;
                        };
                        let mut input = vec![
                            0.0_f32;
                            (if bin_factor > 1 { bin_nx } else { header.nx })
                                as usize
                                * input_lines as usize
                        ];
                        let mut temp = vec![0.0_f32; (header.nx * bin_factor).max(1) as usize];
                        if read_binned_or_reduced(
                            1,
                            in_section,
                            &mut input,
                            if bin_factor > 1 { bin_nx } else { header.nx },
                            input_lines,
                            0.0,
                            needed.iy_in_1 as f32 * bin_factor as f32,
                            bin_factor as f32,
                            if bin_factor > 1 { bin_nx } else { header.nx },
                            input_lines,
                            0,
                            false,
                            &mut temp,
                        )
                        .is_err()
                        {
                            eprintln!("ERROR: NEWSTACK - Reading image file");
                            iiu_close(2);
                            return;
                        }
                        let output_lines = line_out_end - line_out_start + 1;
                        let mut output = vec![0.0_f32; output_nx as usize * output_lines as usize];
                        let matrix = [[fprod[0], fprod[1]], [fprod[2], fprod[3]]];
                        cubinterp(
                            input.as_mut_ptr(),
                            output.as_mut_ptr(),
                            if bin_factor > 1 { bin_nx } else { header.nx },
                            input_lines,
                            output_nx,
                            output_lines,
                            &matrix,
                            if bin_factor > 1 {
                                bin_nx as f32 / 2.0
                            } else {
                                header.nx as f32 / 2.0
                            },
                            if bin_factor > 1 {
                                bin_ny as f32 / 2.0
                            } else {
                                header.ny as f32 / 2.0
                            } - needed.iy_in_1 as f32,
                            fprod[4],
                            (output_ny - output_lines) as f32 / 2.0 + fprod[5]
                                - line_out_start as f32,
                            1.0,
                            header.amean,
                            if_linear,
                        );
                        if !scale_factors.is_empty()
                            || range_scale_entered
                            || !float_shift_constants.is_empty()
                            || float_densities == 2
                            || float_densities == 1
                        {
                            let [scale_factor, const_add] = if !scale_factors.is_empty() {
                                scale_factors[input_index.min(scale_factors.len() - 1)]
                            } else if !float_shift_constants.is_empty() {
                                [float_shift_factor, float_shift_constants[route_index - 1]]
                            } else if mean_sd_entered && sec_sds[route_index - 1] > 0.0 {
                                let section = route_index - 1;
                                let factor = mean_sd[1] / sec_sds[section];
                                [factor, mean_sd[0] - factor * sec_mean[section]]
                            } else if float_densities == 1 && header.amax != header.amin {
                                let factor = 1.0e30_f32 / (header.amax - header.amin);
                                [factor, -factor * header.amin]
                            } else if float2_zmax > float2_zmin && sec_sds[route_index - 1] > 0.0 {
                                let z_factor = 1.0e30_f32 / (float2_zmax - float2_zmin);
                                let section = route_index - 1;
                                [
                                    z_factor / sec_sds[section],
                                    -z_factor
                                        * (sec_mean[section] / sec_sds[section] + float2_zmin),
                                ]
                            } else {
                                // This is the `ifFloat < 0`, no-`multadd` branch
                                // of findScaleFactors: map the source header range
                                // to the requested ScaleMinAndMax range.
                                let (use_min, use_max) = if map_entered {
                                    (map_limits[0], map_limits[1])
                                } else {
                                    (header.amin, header.amax)
                                };
                                let factor =
                                    if scale_limits[1] != scale_limits[0] && use_max != use_min {
                                        (scale_limits[1] - scale_limits[0]) / (use_max - use_min)
                                    } else {
                                        1.0
                                    };
                                [factor, scale_limits[0] - factor * use_min]
                            };
                            let mut scaling = ScaleAndWriteChunk {
                                new_mode: output_mode,
                                write_16_bit_mode_for_floats: false,
                                scale_factor,
                                const_add,
                                optimal_out: match output_mode {
                                    0 => 255.0,
                                    1 => 32767.0,
                                    6 => 65535.0,
                                    _ => 1.0e30,
                                },
                                dmin2: f32::INFINITY,
                                dmax2: f32::NEG_INFINITY,
                                dmean2: 0.0,
                                num_trunc_low: 0,
                                num_trunc_high: 0,
                            };
                            if scale_and_write_chunk(&mut output, output_nx, &mut scaling, |_| {
                                Ok(())
                            })
                            .is_err()
                            {
                                eprintln!("ERROR: NEWSTACK - Scaling output image");
                                iiu_close(2);
                                return;
                            }
                            num_trunc_low += scaling.num_trunc_low;
                            num_trunc_high += scaling.num_trunc_high;
                        }
                        iiu_set_position(2, out_section as i32, line_out_start);
                        if iiu_write_lines(2, output.as_mut_ptr().cast(), output_lines) != 0 {
                            eprintln!("ERROR: NEWSTACK - Writing image file");
                            iiu_close(2);
                            return;
                        }
                        for value in &output {
                            dmin = dmin.min(*value);
                            dmax = dmax.max(*value);
                            dsum += f64::from(*value);
                        }
                        line_out_start = line_out_end + 1;
                    }
                    continue;
                }
                if blank {
                    array.fill(0.0);
                } else {
                    if active_input_index != input_index {
                        if !active_input_file.is_null() {
                            ii_close(active_input_file);
                        }
                        let Ok(name) = CString::new(input_names[input_index].as_bytes()) else {
                            eprintln!("ERROR: NEWSTACK - Invalid input file name");
                            ii_close(out_file);
                            return;
                        };
                        active_input_file = ii_open(name.as_ptr(), c"rb".as_ptr());
                        if !active_input_file.is_null() {
                            active_input_index = input_index;
                        }
                    }
                    if active_input_file.is_null()
                        || ii_read_section_float(
                            active_input_file,
                            array.as_mut_ptr().cast(),
                            in_section,
                        ) != 0
                    {
                        eprintln!("ERROR: NEWSTACK - End of image while reading");
                        if !active_input_file.is_null() {
                            ii_close(active_input_file);
                        }
                        ii_close(out_file);
                        return;
                    }
                    if (*out_file).file == IIFILE_MRC && header.next > 0 {
                        let input_header = (*active_input_file).header.cast::<MrcHeader>();
                        let bytes_per_section = (*input_header).next / (*input_header).nz;
                        let mut input_extra = std::ptr::null_mut();
                        if bytes_per_section <= 0
                            || mrc_read_extra_header(input_header, &mut input_extra) != 0
                        {
                            eprintln!("ERROR: NEWSTACK - Reading extended header");
                            ii_close(active_input_file);
                            ii_close(out_file);
                            return;
                        }
                        let offset = in_section as usize * bytes_per_section as usize;
                        output_extra_data.extend_from_slice(std::slice::from_raw_parts(
                            input_extra.add(offset),
                            bytes_per_section as usize,
                        ));
                        libc::free(input_extra.cast());
                        // `newstack.f90:2654-2670`: when -tilt is supplied
                        // for a conventional real-valued extended header, the
                        // inserted value replaces its first real.
                        if !inserted_tilts.is_empty() && (*input_header).nint == 0 {
                            let record_start = output_extra_data.len() - bytes_per_section as usize;
                            output_extra_data[record_start..record_start + 4]
                                .copy_from_slice(&inserted_tilts[route_index - 1].to_ne_bytes());
                        } else if !inserted_tilts.is_empty()
                            && extra_is_nbytes_and_flags(
                                (*input_header).nint as i32,
                                (*input_header).nreal as i32,
                            ) != 0
                        {
                            let record_start = output_extra_data.len() - bytes_per_section as usize;
                            output_extra_data[record_start..record_start + 2].copy_from_slice(
                                &((100.0 * inserted_tilts[route_index - 1]).round() as i16)
                                    .to_ne_bytes(),
                            );
                        }
                    }
                    if bin_factor > 1 {
                        let mut binned = vec![0.0_f32; bin_nx as usize * bin_ny as usize];
                        for iy in 0..bin_ny {
                            for ix in 0..bin_nx {
                                let mut sum = 0.0_f32;
                                let mut count = 0_i32;
                                for dy in 0..bin_factor {
                                    for dx in 0..bin_factor {
                                        let sx = ix * bin_factor + dx;
                                        let sy = iy * bin_factor + dy;
                                        if sx < header.nx && sy < header.ny {
                                            sum += array[(sy * header.nx + sx) as usize];
                                            count += 1;
                                        }
                                    }
                                }
                                binned[(iy * bin_nx + ix) as usize] = sum / count as f32;
                            }
                        }
                        array = binned;
                    }
                    if !transforms.is_empty() {
                        let Some(transform) =
                            transforms.get(transform_lines[route_index - 1] as usize)
                        else {
                            eprintln!(
                                "ERROR: NEWSTACK - TRANSFORM LINE number out of bounds: {in_section}"
                            );
                            ii_close(active_input_file);
                            ii_close(out_file);
                            return;
                        };
                        let offset_index = route_index - 1;
                        let mut fprod = *transform;
                        if bin_factor > 1 {
                            fprod[4] /= bin_factor as f32;
                            fprod[5] /= bin_factor as f32;
                        }
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
                        } else {
                            fprod[4] -= x_offsets[offset_index];
                            fprod[5] -= y_offsets[offset_index];
                        }
                        let mut transformed =
                            vec![0.0_f32; output_nx as usize * output_ny as usize];
                        let matrix = [[fprod[0], fprod[1]], [fprod[2], fprod[3]]];
                        cubinterp(
                            array.as_mut_ptr(),
                            transformed.as_mut_ptr(),
                            if bin_factor > 1 { bin_nx } else { header.nx },
                            if bin_factor > 1 { bin_ny } else { header.ny },
                            output_nx,
                            output_ny,
                            &matrix,
                            if bin_factor > 1 {
                                bin_nx as f32 / 2.0
                            } else {
                                header.nx as f32 / 2.0
                            },
                            if bin_factor > 1 {
                                bin_ny as f32 / 2.0
                            } else {
                                header.ny as f32 / 2.0
                            },
                            fprod[4],
                            fprod[5],
                            1.0,
                            header.amean,
                            if_linear,
                        );
                        array = transformed;
                    }
                    if transforms.is_empty()
                        && bin_factor == 1
                        && (output_nx != header.nx || output_ny != header.ny)
                        && array.len() >= (header.nx as usize * header.ny as usize)
                    {
                        let mut repacked = vec![0.0_f32; output_nx as usize * output_ny as usize];
                        let ix1 = header.nx / 2 - output_nx / 2;
                        let iy1 = header.ny / 2 - output_ny / 2;
                        irepak2(
                            &mut repacked,
                            &array,
                            header.nx,
                            header.ny,
                            ix1,
                            ix1 + output_nx - 1,
                            iy1,
                            iy1 + output_ny - 1,
                            header.amean,
                        );
                        array = repacked;
                    }
                    if !scale_factors.is_empty()
                        || range_scale_entered
                        || !float_shift_constants.is_empty()
                        || float_densities == 2
                        || float_densities == 1
                    {
                        let [scale_factor, const_add] = if !scale_factors.is_empty() {
                            scale_factors[input_index.min(scale_factors.len() - 1)]
                        } else if !float_shift_constants.is_empty() {
                            [float_shift_factor, float_shift_constants[route_index - 1]]
                        } else if mean_sd_entered && sec_sds[route_index - 1] > 0.0 {
                            let section = route_index - 1;
                            let factor = mean_sd[1] / sec_sds[section];
                            [factor, mean_sd[0] - factor * sec_mean[section]]
                        } else if float_densities == 1 && header.amax != header.amin {
                            let factor = 1.0e30_f32 / (header.amax - header.amin);
                            [factor, -factor * header.amin]
                        } else if float2_zmax > float2_zmin && sec_sds[route_index - 1] > 0.0 {
                            let z_factor = 1.0e30_f32 / (float2_zmax - float2_zmin);
                            let section = route_index - 1;
                            [
                                z_factor / sec_sds[section],
                                -z_factor * (sec_mean[section] / sec_sds[section] + float2_zmin),
                            ]
                        } else {
                            let (use_min, use_max) = if map_entered {
                                (map_limits[0], map_limits[1])
                            } else {
                                (header.amin, header.amax)
                            };
                            let factor = if scale_limits[1] != scale_limits[0] && use_max != use_min
                            {
                                (scale_limits[1] - scale_limits[0]) / (use_max - use_min)
                            } else {
                                1.0
                            };
                            [factor, scale_limits[0] - factor * use_min]
                        };
                        let mut scaling = ScaleAndWriteChunk {
                            new_mode: output_mode,
                            write_16_bit_mode_for_floats: false,
                            scale_factor,
                            const_add,
                            optimal_out: match output_mode {
                                0 => 255.0,
                                1 => 32767.0,
                                6 => 65535.0,
                                _ => 1.0e30,
                            },
                            dmin2: f32::INFINITY,
                            dmax2: f32::NEG_INFINITY,
                            dmean2: 0.0,
                            num_trunc_low: 0,
                            num_trunc_high: 0,
                        };
                        if scale_and_write_chunk(&mut array, output_nx, &mut scaling, |_| Ok(()))
                            .is_err()
                        {
                            eprintln!("ERROR: NEWSTACK - Scaling output image");
                            ii_close(active_input_file);
                            ii_close(out_file);
                            return;
                        }
                        num_trunc_low += scaling.num_trunc_low;
                        num_trunc_high += scaling.num_trunc_high;
                    }
                }
                if blank
                    && (output_nx != header.nx || output_ny != header.ny)
                    && array.len() >= (header.nx as usize * header.ny as usize)
                {
                    let mut repacked = vec![0.0_f32; output_nx as usize * output_ny as usize];
                    let ix1 = header.nx / 2 - output_nx / 2;
                    let iy1 = header.ny / 2 - output_ny / 2;
                    irepak2(
                        &mut repacked,
                        &array,
                        header.nx,
                        header.ny,
                        ix1,
                        ix1 + output_nx - 1,
                        iy1,
                        iy1 + output_ny - 1,
                        0.0,
                    );
                    array = repacked;
                }
                if ii_write_section_float(out_file, array.as_mut_ptr().cast(), out_section as i32)
                    != 0
                {
                    eprintln!("ERROR: NEWSTACK - Writing image file");
                    ii_close(out_file);
                    if !active_input_file.is_null() {
                        ii_close(active_input_file);
                    }
                    return;
                }
                for value in &array {
                    dmin = dmin.min(*value);
                    dmax = dmax.max(*value);
                    dsum += f64::from(*value);
                }
            }
            if chunked_affine {
                if iiu_write_header(
                    2,
                    title.as_mut_ptr().cast(),
                    0,
                    dmin,
                    dmax,
                    (dsum
                        / f64::from(output_nx)
                        / f64::from(output_ny)
                        / f64::from(num_output_sections[output_index])) as f32,
                ) != 0
                {
                    eprintln!("ERROR: NEWSTACK - Writing output header");
                    iiu_close(2);
                    return;
                }
                iiu_close(2);
            } else {
                let out_header = if (*out_file).file == IIFILE_MRC {
                    (*out_file).header.cast::<MrcHeader>()
                } else {
                    &mut non_mrc_output_header
                };
                (*out_header).amin = dmin;
                (*out_header).amax = dmax;
                (*out_header).amean = (dsum
                    / f64::from(output_nx)
                    / f64::from(output_ny)
                    / f64::from(num_output_sections[output_index]))
                    as f32;
                if ((!inserted_tilts.is_empty() && header.next == 0)
                    || !output_extra_data.is_empty())
                    && (*out_file).file == IIFILE_MRC
                    && mrc_write_extra_header(
                        out_header,
                        if output_extra_data.is_empty() {
                            inserted_tilts[output_tilt_start
                                ..output_tilt_start + num_output_sections[output_index] as usize]
                                .as_ptr()
                                .cast_mut()
                                .cast()
                        } else {
                            output_extra_data.as_mut_ptr()
                        },
                        if output_extra_data.is_empty() {
                            4 * num_output_sections[output_index]
                        } else {
                            output_extra_data.len() as i32
                        },
                    ) != 0
                {
                    eprintln!("ERROR: NEWSTACK - Writing output header");
                    ii_close(out_file);
                    if !active_input_file.is_null() {
                        ii_close(active_input_file);
                    }
                    return;
                }
                ii_sync_from_mrc_header(out_file, out_header);
                if ((*out_file).file == IIFILE_MRC
                    && mrc_head_write((*out_file).fp, out_header) != 0)
                    || ((*out_file).file != IIFILE_MRC && ii_write_header(out_file) != 0)
                {
                    eprintln!("ERROR: NEWSTACK - Writing output header");
                    ii_close(out_file);
                    if !active_input_file.is_null() {
                        ii_close(active_input_file);
                    }
                    return;
                }
                ii_close(out_file);
            }
        }
        if !active_input_file.is_null() {
            ii_close(active_input_file);
        }
        if reorder_by_tilt != 0 && !new_angle_output_file.is_empty() {
            let Ok(mut file) = std::fs::File::create(&new_angle_output_file) else {
                eprintln!("ERROR: NEWSTACK - Opening new tilt angle output file");
                return;
            };
            use std::io::Write;
            for angle in &reordered_angles {
                if writeln!(file, "{angle:9.2}").is_err() {
                    eprintln!("ERROR: NEWSTACK - Writing new tilt angle output file");
                    return;
                }
            }
        }
        if !quiet && num_trunc_low + num_trunc_high > 0 {
            println!(
                " TRUNCATIONS OCCURRED:{num_trunc_low:11} at low end,{num_trunc_high:11} at high end"
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
    let xcen = nxb as f32 / 2.0 + xtrans + 0.5;
    let ycen = nyb as f32 / 2.0 + ytrans + 0.5;
    let denom = amat[0][0] * amat[1][1] - amat[0][1] * amat[1][0];
    let dxo = ix as f32 - xcen;
    let dyo = iy as f32 - ycen;
    (
        (amat[1][1] * dxo - amat[0][1] * dyo) / denom + xfcen + 0.5,
        (-amat[1][0] * dxo + amat[0][0] * dyo) / denom + yfcen + 0.5,
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
    let dsum_sq: f64 = dsum_chunk
        .iter()
        .zip(sd_chunk)
        .zip(pix_chunk)
        .map(|((&sum, &sd), &pix)| {
            let average = sum / pix;
            pix * (average * average - dmean * dmean) + (pix - 1.0) * f64::from(sd) * f64::from(sd)
        })
        .sum();
    (
        dmean as f32,
        (dsum_sq.max(0.0) / (pix_tot - 1.0).max(1.0)).sqrt() as f32,
    )
}

/// Original `getItemsToUse` (`newstack.f90:3296`).
pub fn get_items_to_use(
    nxforms: i32,
    in_list: &[i32],
    requested: Option<&str>,
    one_per_file: bool,
    num_in_files: i32,
    number_offset: i32,
) -> Result<Vec<i32>, String> {
    let mut line_use = if nxforms == 1 {
        vec![number_offset]
    } else if one_per_file {
        (0..num_in_files)
            .map(|index| index + number_offset)
            .collect()
    } else {
        in_list.iter().map(|&index| index + number_offset).collect()
    };
    if let Some(list) = requested {
        let mut parsed = vec![0_i32; 1_000_000];
        let (mut count, mut limit) = (0, 1_000_000);
        parselist2(list, &mut parsed, &mut count, &mut limit)
            .map_err(|_| "invalid transform list".to_owned())?;
        parsed.truncate(count as usize);
        line_use = parsed;
    }
    for line in &mut line_use {
        *line -= number_offset;
        if *line < 0 || *line >= nxforms {
            return Err(format!(
                "transform number out of bounds: {}",
                *line + number_offset
            ));
        }
    }
    Ok(line_use)
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
        let iy_base = values.ny_bin / 2 + values.ycen as i32 - values.ny_out / 2;
        let ix_base = values.nx_bin / 2 + values.xcen as i32 - values.nx_out / 2;
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
            let tmp_mean = values.dsum as f32 / (values.nx_out * values.ny_out) as f32;
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
        let sum: f64 = values.iter().map(|&value| f64::from(value)).sum();
        let average = sum / count as f64;
        let sd = (values
            .iter()
            .map(|&value| {
                let delta = f64::from(value) - average;
                delta * delta
            })
            .sum::<f64>()
            / count.saturating_sub(1).max(1) as f64)
            .sqrt() as f32;
        dmin2 = dmin2.min(values.iter().copied().fold(f32::INFINITY, f32::min));
        dmax2 = dmax2.max(values.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        dsum += sum;
        sums.push(sum);
        sds.push(sd);
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
}

/// Original `reallocateIfNeeded` memory arithmetic.  Allocation itself remains
/// `reallocate_array`, as in the two source procedures.
pub fn reallocate_if_needed(values: &mut ReallocateIfNeeded) -> (usize, usize) {
    let in_place_fac =
        if values.process_in_place && values.ft_reduce_fac == 0. && !values.phase_shift {
            0.25
        } else {
            1.
        };
    let physical = values.physical_memory / 4.;
    let mut limit = 3.75e9;
    if physical > 0. {
        limit = (0.75 * in_place_fac * physical)
            .min(physical - 0.25e9)
            .max(0.1e9);
        if limit > 3.75e9 {
            limit = 3.75e9_f64.max(0.5 * in_place_fac * physical);
        }
    }
    if values.lim_entered != 1 {
        let mut temp = 1usize;
        if values.read_shrunk {
            let minimum = if values.read_reduction > 32. { 3. } else { 10. };
            temp = ((values.nx as f32 * ((minimum + 6.) * values.read_reduction).ceil() + 20.)
                as usize)
                .max((values.nx as usize * values.ny as usize).min(5_000_000));
        }
        if values.i_binning > 1 {
            temp = values.nx as usize * values.i_binning as usize;
        }
        if values.fourier_scaling && values.nx_fspad > 0 {
            temp = temp.max((values.nx_fcrop_pad as usize + 2) * values.ny_fcrop_pad as usize);
        }
        if (values.phase_shift || values.fourier_scaling) && values.nx_fspad > 0 && values.noise_pad
        {
            temp = temp.max(
                2 * values.nx_bin_fft.max(values.ny_bin_fft) as usize
                    + (values.nx_fspad - values.nx_bin_fft + values.ny_fspad - values.ny_bin_fft)
                        as usize,
            );
        }
        values.len_temp = temp;
    }
    if values.lim_entered == 0 {
        let mut needed = values.nx_bin as usize * values.ny_needed as usize;
        if (values.phase_shift || values.fourier_scaling) && values.nx_fspad > 0 {
            needed = (values.nx_fspad as usize + 2) * (values.ny_fspad as usize + 1);
        }
        if values.nx_out > 0 && values.ny_out > 0 {
            if values.process_in_place && needed + 2 * values.nx_bin as usize > limit as usize {
                values.process_in_place = false;
            }
            if values.process_in_place {
                needed += 2 * values.nx_bin as usize;
            } else if values.ft_expand_fac > 0. {
                needed += (values.nx_out as usize * values.ny_out as usize)
                    .max((values.nx_fcrop_pad as usize) * values.ny_fcrop_pad as usize);
            } else {
                needed += values.nx_out as usize * values.ny_out as usize;
            }
        }
        needed = needed.min(limit as usize);
        if needed + values.len_temp > values.lim_to_alloc {
            values.lim_to_alloc = needed + values.len_temp;
        }
        (values.lim_to_alloc - values.len_temp, values.len_temp)
    } else {
        (values.lim_to_alloc - values.len_temp, values.len_temp)
    }
}

/// Original `reallocateArray` (`newstack.f90:2886`).
pub fn reallocate_array(
    array: &mut Vec<f32>,
    lim_to_alloc: usize,
    len_temp: usize,
    lim_if_fail: usize,
) -> Result<usize, String> {
    let mut limit = lim_to_alloc;
    if limit.saturating_sub(len_temp) < 100 {
        return Err("With achievable memory allocation, the temporary array does not leave enough space for input/output".to_owned());
    }
    if array.try_reserve_exact(limit).is_err() && limit > lim_if_fail {
        limit = lim_if_fail;
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
    pub dmean2: f64,
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
        values.dmean2 += sum;
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

/// Original `transferCollections` (`newstack.f90:3265`).
pub unsafe fn transfer_collections(zvalue_name: &str, ind_adoc_out: i32) -> Result<(), String> {
    let count = adoc_get_num_collections();
    for collection in 1..=count {
        let mut name_ptr = core::ptr::null_mut();
        if adoc_get_collection_name(collection, &raw mut name_ptr) != 0 || name_ptr.is_null() {
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
                if adoc_transfer_section(name_c.as_ptr(), section - 1, ind_adoc_out, section_ptr, 0)
                    != 0
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

/// Original `openInputFile` (`newstack.f90:3157`).  Returns `needClose1`.
pub unsafe fn open_input_file(
    ind_in_file: usize,
    num_vol_read: usize,
    list_volumes: &[i32],
    in_file: &[String],
) -> Result<i32, String> {
    if ind_in_file <= num_vol_read {
        ii_allow_multi_volume(1);
        if list_volumes[ind_in_file - 1] > 1 {
            let name = CString::new(in_file[ind_in_file - 1].as_bytes())
                .map_err(|_| "Invalid input file name")?;
            if iiu_open(11, name.as_ptr(), c"RO".as_ptr()) != 0
                || iiu_volume_open(1, 11, list_volumes[ind_in_file - 1] - 1) != 0
            {
                return Err("Opening volume in multi-volume file".into());
            }
            Ok(11)
        } else {
            let name = CString::new(in_file[ind_in_file - 1].as_bytes())
                .map_err(|_| "Invalid input file name")?;
            if iiu_open(1, name.as_ptr(), c"RO".as_ptr()) != 0 {
                return Err("Opening image file".into());
            }
            Ok(0)
        }
    } else {
        ii_allow_multi_volume(0);
        let name = CString::new(in_file[ind_in_file - 1].as_bytes())
            .map_err(|_| "Invalid input file name")?;
        if iiu_open(1, name.as_ptr(), c"RO".as_ptr()) != 0 {
            return Err("Opening image file".into());
        }
        Ok(0)
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
        assert_eq!(
            get_items_to_use(5, &[0, 2], Some("1-3,0"), false, 0, 0).unwrap(),
            [1, 2, 3, 0]
        );
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
