//! Translation of `IMOD/pysrc/splitcorrection`: sets up command files for
//! parallel CTF correction.
//!
//! A Python command script with no functions; its top level is
//! [`splitcorrection`].  `getmrcsize` runs our own `header` in process.
//! Python semantics carried explicitly: `//` and `%` floor.

use super::imodpy::{
    INT_VALUE, OptionValue, STRING_VALUE, add_imod_bin_ignore_sighup, clean_chunk_files,
    complete_and_check_com_file, exit_from_imod_error, fmtstr, get_mrc_size, option_value,
    os_path_splitext, parallel_boundary_size, prnstr, py_int_floordiv, py_int_mod,
    read_text_file, standard_type_extensions, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_integer, pip_get_non_option_arg,
    pip_get_string, pip_get_three_integers, pip_read_or_parse_options,
};
use super::pysed::{PysedSrc, pysed};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`splitcorrection:1-177`).  Returns the status of
/// its `sys.exit`; error paths exit the process as `exitError` does.
pub fn splitcorrection(arguments: &[OsString]) -> i32 {
    let progname = "splitcorrection";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        let _ = std::io::stdout().flush();
        return 1;
    }

    let bound_ext = "cbound";
    let mut bound_pixels = parallel_boundary_size(2048);

    // Fallbacks from ../manpages/autodoc2man 3 1 splitcorrection
    let options: Vec<String> = [
        ":m:I:",
        ":b:I:",
        "i:InitialComNumber:I:",
        "o:OpenForMoreComs:B:",
        "r:RootNameOfOutput:FN:",
        "dir:DirectoryForOutput:FN:",
        "unique:UniqueInfoFile:B:",
        "size:InputDimensions:IT:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_num_opts, _num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 1);

    // Get the com file name, derive a root name and new com file name, check exists
    let comfile = pip_get_non_option_arg(0).unwrap_or_default();
    let (comfile, rootname) = complete_and_check_com_file(&comfile);
    let com_ext: String = comfile
        .chars()
        .skip(comfile.chars().count().saturating_sub(4))
        .collect();

    // Get options
    let max_slices = pip_get_integer("m", 5).unwrap_or(5) as i64;
    bound_pixels = pip_get_integer("b", bound_pixels).unwrap_or(bound_pixels);
    let start_num = pip_get_integer("InitialComNumber", 0).unwrap_or(0) as i64;
    let if_start_num = 1 - pip_get_err_no();
    let leave_open = pip_get_boolean("OpenForMoreComs", 0).unwrap_or(0);
    let rootname = pip_get_string("RootNameOfOutput", &rootname).unwrap_or(rootname);
    let unique_info = pip_get_boolean("UniqueInfoFile", 0).unwrap_or(0);
    let mut out_dir = pip_get_string("DirectoryForOutput", "").unwrap_or_default();
    let mut root_with_dir = rootname.clone();
    if !out_dir.is_empty() {
        root_with_dir = Path::new(&out_dir)
            .join(&rootname)
            .to_string_lossy()
            .into_owned();
    }

    let mut com_lines =
        read_text_file(&comfile, Some("ctfphaseflip command file"), false, None)
            .unwrap_or_default();
    let out_file = match option_value(&com_lines, "OutputFileName", STRING_VALUE, false, 0, None, None)
    {
        Some(OptionValue::String(value)) if !value.is_empty() => value,
        _ => exit_error(&format!("Cannot find name of output file in {comfile}")),
    };
    let mut dimensions: (i64, i64, i64) = match pip_get_three_integers("InputDimensions", (0, 0, 0))
    {
        Ok((x, y, z)) => (x as i64, y as i64, z as i64),
        Err(_) => (0, 0, 0),
    };
    if pip_get_err_no() != 0 {
        let in_stack = match option_value(&com_lines, "InputStack", STRING_VALUE, false, 0, None, None)
        {
            Some(OptionValue::String(value)) if !value.is_empty() => value,
            _ => exit_error(&format!("Cannot find name of input file in {comfile}")),
        };

        if !Path::new(&in_stack).exists() {
            exit_error(&format!("Input stack {in_stack} does not exist"));
        }

        match get_mrc_size(&in_stack) {
            Ok((x, y, z)) => dimensions = (x as i64, y as i64, z as i64),
            Err(_) => exit_from_imod_error(progname),
        }
    }

    let views: (i64, i64) =
        match option_value(&com_lines, "StartingEndingViews", INT_VALUE, false, 0, None, None) {
            Some(OptionValue::Integers(values)) if !values.is_empty() => {
                (values[0] as i64, values.get(1).copied().unwrap_or(0) as i64)
            }
            _ => (1, dimensions.2),
        };

    if if_start_num == 0 {
        clean_chunk_files(&root_with_dir, false);
    }

    let viewdel = "StartingEndingViews";
    let total = 1 + views.1 - views.0;

    let num_slabs = py_int_floordiv(total + max_slices - 1, max_slices);
    let slab_size = py_int_floordiv(total, num_slabs);
    let remainder = py_int_mod(total, num_slabs);

    let (out_root, _out_ext) = os_path_splitext(&out_file);
    let mut this_com = format!("{root_with_dir}-start{com_ext}");
    if if_start_num != 0 {
        this_com = format!(
            "{root_with_dir}{}",
            format!("-{start_num:03}-sync{com_ext}")
        );
    }

    // `for ... else`: insert the format line only when no line sets it
    if !com_lines
        .iter()
        .any(|line| line.starts_with("$setenv IMOD_OUTPUT_FORMAT"))
    {
        let mut out_format = std::env::var("IMOD_OUTPUT_FORMAT").unwrap_or_default();
        if out_format.is_empty() || !standard_type_extensions().contains(&out_format) {
            out_format = "MRC".to_owned();
        }
        com_lines.insert(0, format!("$setenv IMOD_OUTPUT_FORMAT {out_format}"));
    }

    let sedlist = vec![
        format!("/{viewdel}/d"),
        "/DefocusFile/a/StartingEndingViews -1  -1/".to_owned(),
        fmtstr(
            "/DefocusFile/a/TotalViews {} {}/",
            &[views.0.to_string(), views.1.to_string()],
        ),
    ];
    let mut sed_lines = match pysed(&sedlist, PysedSrc::Lines(&com_lines), None, false, '/', false)
    {
        Ok(lines) => lines.unwrap_or_default(),
        Err(message) => exit_error(&message),
    };
    sed_lines.push("$sync".to_owned());
    let _ = write_text_file(&this_com, &sed_lines, false);
    let mut total_coms = 1;

    let mut bound_file = format!("{rootname}-bound.info");
    if unique_info != 0 {
        bound_file = format!("{rootname}-bound-{start_num:03}.info");
    }
    let width = dimensions.0;
    let mut bound_lines = py_int_floordiv(bound_pixels as i64 + width - 1, width);
    let ny = dimensions.1;
    bound_lines = bound_lines.min(py_int_floordiv(ny, 2) + 1);

    let mut bound_out = vec![fmtstr(
        "1 0 {} {} {}",
        &[
            width.to_string(),
            bound_lines.to_string(),
            num_slabs.to_string(),
        ],
    )];

    let mut orig_offset = views.0;
    let firstview = views.0;
    for num in 1..=num_slabs {
        this_com = format!(
            "{root_with_dir}{}",
            format!("-{:03}{com_ext}", num + start_num)
        );
        let mut orig_end = orig_offset + slab_size - 1;
        if num <= remainder {
            orig_end += 1;
        }
        let sedlist = vec![
            format!("/{viewdel}/d"),
            fmtstr(
                "/DefocusFile/a/StartingEndingViews {}  {}/",
                &[orig_offset.to_string(), orig_end.to_string()],
            ),
            fmtstr(
                "/DefocusFile/a/TotalViews {} {}/",
                &[views.0.to_string(), views.1.to_string()],
            ),
            format!("/DefocusFile/a/BoundaryInfoFile {bound_file}/"),
        ];
        if let Err(message) = pysed(
            &sedlist,
            PysedSrc::Lines(&com_lines),
            Some(&this_com),
            false,
            '/',
            false,
        ) {
            exit_error(&message);
        }
        total_coms += 1;

        let mut bound_start = orig_offset - firstview;
        let mut bound_end = orig_end - firstview;
        if num == 1 {
            bound_start = -1;
        }
        if num == num_slabs {
            bound_end = -1;
        }
        bound_out.push(format!("{out_root}-{num:03}.{bound_ext}"));
        bound_out.push(fmtstr(
            "{} 0 {} -1",
            &[bound_start.to_string(), bound_end.to_string()],
        ));

        orig_offset = orig_end + 1;
    }

    let _ = write_text_file(&bound_file, &bound_out, false);

    let mut dir_mess = String::new();
    if !out_dir.is_empty() {
        out_dir = format!("\"{out_dir}\"");
        dir_mess = format!("in directory {out_dir}");
    }
    this_com = format!("{root_with_dir}-finish{com_ext}");
    if leave_open != 0 {
        this_com = format!(
            "{root_with_dir}{}",
            format!("-{:03}-sync{com_ext}", num_slabs + start_num + 1)
        );
    }

    let mut fin_lines = vec![
        fmtstr(
            "$fixboundaries \"{}\" \"{}\"",
            &[out_file.clone(), bound_file.clone()],
        ),
        fmtstr(
            "$collectmmm pixels= \"{}\" {} \"{}\" {} {}",
            &[
                rootname.clone(),
                num_slabs.to_string(),
                out_file.clone(),
                (start_num + 1).to_string(),
                out_dir.clone(),
            ],
        ),
        fmtstr(
            "$b3dremove -g \"{}-[0-9][0-9][0-9]*.{}\" \"{}\"",
            &[out_root.clone(), bound_ext.to_owned(), bound_file.clone()],
        ),
    ];
    if leave_open == 0 {
        fin_lines.push(fmtstr(
            &format!(
                "$b3dremove -g {{0}}-[0-9][0-9][0-9]*{com_ext}* {{0}}-[0-9][0-9][0-9]*.log* {{0}}-start*.* {{0}}-finish*{com_ext}*"
            ),
            &[root_with_dir.clone()],
        ));
    }
    let _ = write_text_file(&this_com, &fin_lines, false);
    total_coms += 1;

    prnstr(
        &fmtstr(
            "{} command files for {} chunks created {}",
            &[total_coms.to_string(), num_slabs.to_string(), dir_mess],
        ),
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    0
}
