//! Translation of `IMOD/pysrc/onegenplot`: runs `genhstplt` to make one
//! plot of columns of a data file.
//!
//! The script's top level is [`onegenplot`]; its one function is
//! [`cleanup_error`].  `genhstplt` is IMOD's graphics program
//! (`flib/graphics/genhstplt.f90`), translated in
//! `flib/graphics/genhstplt.rs` on the Rust-native plot layer
//! (`flib/subrs/graphics/qtplax.rs`).  It shows a window, so it does not run
//! in this process: `imodpy::run_cmd` runs it as a child of our own binary,
//! with the script's input, as the script runs it.

use super::imodpy::{
    add_imod_bin_ignore_sighup, cleanup_files, convert_to_integer, exit_from_imod_error, prnstr,
    py_float, py_int, py_str_float, run_cmd,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_err_no, pip_get_in_out_file, pip_get_integer,
    pip_get_integer_array, pip_get_string, pip_get_two_floats, pip_get_two_integers,
    pip_number_of_entries, pip_read_or_parse_options,
};
use std::ffi::OsString;
use std::io::{BufRead as _, Write as _};
use std::path::Path;

/// `NUM_KEY_LIMIT = 10` (`onegenplot:11`): `LIM_KEYS` in
/// `flib/subrs/graphics/plotvars.f90`.
const NUM_KEY_LIMIT: usize = 10;

/// `defSymbols` (`onegenplot:43`).
const DEF_SYMBOLS: [i64; 14] = [9, 7, 5, 8, 13, 14, 1, 11, 3, 10, 6, 2, 12, 4];

/// `stockColors` (`onegenplot:44-65`), in the dict's order.
const STOCK_COLORS: [(&str, [i64; 3]); 22] = [
    ("aqua", [0, 255, 255]),
    ("blue", [0, 0, 255]),
    ("cyan", [0, 255, 255]),
    ("darkblue", [0, 0, 139]),
    ("darkgreen", [0, 100, 0]),
    ("darkmagenta", [139, 0, 139]),
    ("darkorange", [255, 140, 0]),
    ("darkred", [139, 0, 0]),
    ("darkviolet", [148, 0, 211]),
    ("fuchsia", [255, 0, 255]),
    ("green", [0, 128, 0]),
    ("lime", [0, 255, 0]),
    ("magenta", [255, 0, 255]),
    ("maroon", [128, 0, 0]),
    ("navy", [0, 0, 128]),
    ("olive", [128, 128, 0]),
    ("orange", [255, 165, 0]),
    ("purple", [128, 0, 128]),
    ("red", [255, 0, 0]),
    ("teal", [0, 128, 128]),
    ("yellow", [255, 255, 0]),
    ("black", [0, 0, 0]),
];

/// `defColors` (`onegenplot:67`).
const DEF_COLORS: [&str; 12] = [
    "darkblue",
    "darkgreen",
    "darkred",
    "darkviolet",
    "darkorange",
    "teal",
    "olive",
    "black",
    "red",
    "green",
    "magenta",
    "cyan",
];

/// The RGB of a stock color, as `stockColors[name]`.
fn stock_color(name: &str) -> Option<[i64; 3]> {
    STOCK_COLORS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, rgb)| *rgb)
}

/// `def cleanupError(strn)` (`onegenplot:14`): exit and remove data file if
/// option given.
fn cleanup_error(cleanup: bool, data_name: &str, strn: &str) -> ! {
    if cleanup {
        cleanup_files(&[data_name.to_owned()]);
    }
    exit_error(strn)
}

/// The script's top level (`onegenplot:21-396`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn onegenplot(arguments: &[OsString]) -> i32 {
    let progname = "onegenplot";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 onegenplot
    let options: Vec<String> = [
        "input:InputDataFile:FN:",
        "ncol:NumberOfColumns:I:",
        "skip:SkipLinesAtStart:I:",
        "columns:ColumnsToPlot:IA:",
        "types:TypesToPlot:IA:",
        "symbols:SymbolsForTypes:IA:",
        "connect:ConnectWithLines:B:",
        "ordinal:OrdinalsForXvalues:B:",
        "xlog:XLogOrRootAndBase:FP:",
        "ylog:YLogOrRootAndBase:FP:",
        "hue:HueOfGroup:CHM:",
        "defhue:DefaultHues:B:",
        "stock:StockColorList:B:",
        "axis:XaxisLabel:CH:",
        "keys:KeyLabels:CHM:",
        "message:MessageBoxLine:CHM:",
        "tooltip:ToolTipLine:CHM:",
        "size:SizeOfPlot:IP:",
        "position:PositionOfPlot:IP:",
        "yminmax:YRangeMinAndMax:FP:",
        "png:SavePNGandExit:FN:",
        "remove:RemoveDataFile:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    let (_opts, _nonopts) = pip_read_or_parse_options(&argv, &options, progname, 1, 0, 0);

    if pip_get_boolean("StockColorList", 0).unwrap_or(0) != 0 {
        prnstr("Standard colors available by name:", "\n", false);
        for (col, rgb) in STOCK_COLORS {
            let longer = format!("{col}{}", " ".repeat(13 - col.len()));
            prnstr(
                &format!("  {longer} {:>3}  {:>3}  {:>3}", rgb[0], rgb[1], rgb[2]),
                "\n",
                false,
            );
        }
        return done(0);
    }

    let data_name = pip_get_in_out_file("InputDataFile", 0)
        .ok()
        .flatten()
        .unwrap_or_default();
    if data_name.is_empty() {
        exit_error("The data file name must be entered");
    }

    let cleanup = pip_get_boolean("RemoveDataFile", 0).unwrap_or(0) != 0;
    let mut skip_lines = pip_get_integer("SkipLinesAtStart", -1).unwrap_or(-1) as i64;
    let num_col = pip_get_integer("NumberOfColumns", -1).unwrap_or(-1) as i64;
    let mut column_list: Vec<i64> = pip_get_integer_array("ColumnsToPlot", 0).unwrap_or_default();
    let mut type_list: Vec<i64> = pip_get_integer_array("TypesToPlot", 0).unwrap_or_default();
    let mut if_types: i64 = 0;
    if !type_list.is_empty() {
        if_types = 1;
    }
    let mut symbols: Vec<i64> = pip_get_integer_array("SymbolsForTypes", 0).unwrap_or_default();
    let (xlog, xbase) = pip_get_two_floats("XLogOrRootAndBase", (0., 0.)).unwrap_or((0., 0.));
    let (ylog, ybase) = pip_get_two_floats("YLogOrRootAndBase", (0., 0.)).unwrap_or((0., 0.));
    let mut ordinals = pip_get_boolean("OrdinalsForXvalues", 0).unwrap_or(0) as i64;
    let connect = pip_get_boolean("ConnectWithLines", 0).unwrap_or(0);
    let mut axis_label = pip_get_string("XaxisLabel", "").unwrap_or_default();
    let num_keys = pip_number_of_entries("KeyLabels").unwrap_or(0) as usize;
    let mut keys: Vec<String> = Vec::new();
    for _ in 0..num_keys {
        keys.push(pip_get_string("KeyLabels", "").unwrap_or_default());
    }

    let save_png_name = pip_get_string("SavePNGandExit", "").unwrap_or_default();
    let (ymin, ymax) = pip_get_two_floats("YRangeMinAndMax", (0., 0.)).unwrap_or((0., 0.));
    if pip_get_err_no() == 0 && ymin >= ymax {
        exit_error("The minimum of the Y range must be less than the maximum");
    }
    let (xmin, xmax) = pip_get_two_floats("XRangeMinAndMax", (0., 0.)).unwrap_or((0., 0.));
    if pip_get_err_no() == 0 && xmin >= xmax {
        exit_error("The minimum of the X range must be less than the maximum");
    }
    let use_dflt_hues = pip_get_boolean("DefaultHues", 0).unwrap_or(0) != 0;
    let num_colors = pip_number_of_entries("HueOfGroup").unwrap_or(0);
    let mut colors: Vec<String> = Vec::new();
    let mut colored_groups: Vec<i32> = Vec::new();
    for _ in 0..num_colors {
        let color_str = pip_get_string("HueOfGroup", "").unwrap_or_default();
        let csplit: Vec<&str> = color_str.split(',').collect();
        if csplit.len() != 2 && csplit.len() != 4 {
            cleanup_error(
                cleanup,
                &data_name,
                &format!("The color entry {color_str} does not have 2 or 4 components"),
            );
        }
        let group = convert_to_integer(csplit[0], "group number of color entry");
        colored_groups.push(group);
        if csplit.len() == 2 {
            let col = csplit[1];
            let Some(rgb) = stock_color(col) else {
                cleanup_error(
                    cleanup,
                    &data_name,
                    &format!("{col} is not in the list of stock colors"),
                )
            };
            colors.push(format!("{group},{},{},{}", rgb[0], rgb[1], rgb[2]));
        } else {
            let red = convert_to_integer(csplit[1], "second component of color entry");
            let green = convert_to_integer(csplit[2], "third component of color entry");
            let blue = convert_to_integer(csplit[3], "fourth component of color entry");
            colors.push(format!("{group},{red},{green},{blue}"));
        }
    }

    let mut message = String::new();
    let mut tooltip = String::new();
    let mut pos_option = String::new();
    let mut size_option = String::new();
    let num = pip_number_of_entries("MessageBoxLine").unwrap_or(0);
    for ind in 0..num {
        let one_line = pip_get_string("MessageBoxLine", "").unwrap_or_default();
        if ind != 0 && !message.ends_with('\n') {
            message.push('\n');
        }
        message += &one_line;
    }

    let num = pip_number_of_entries("ToolTipLine").unwrap_or(0);
    for ind in 0..num {
        let one_line = pip_get_string("ToolTipLine", "").unwrap_or_default();
        if ind != 0 && !tooltip.ends_with('\n') {
            tooltip.push('\n');
        }
        tooltip += &one_line;
    }

    let (xsize, ysize) = pip_get_two_integers("SizeOfPlot", (0, 0)).unwrap_or((0, 0));
    if xsize > 0 && ysize > 0 {
        size_option = format!(" -s {xsize},{ysize}");
    }
    let (xpos, ypos) = pip_get_two_integers("PositionOfPlot", (0, 0)).unwrap_or((0, 0));
    if pip_get_err_no() == 0 {
        pos_option = format!(" -p {xpos},{ypos}");
    }

    if !Path::new(&data_name).exists() {
        exit_error(&format!("Data file {data_name} does not exist"));
    }

    let mut true_col = num_col;
    if skip_lines < 0 || num_col <= 0 {
        let mut err_string = "Opening";
        let read: Result<(), std::io::Error> = (|| {
            let data_file = std::fs::File::open(&data_name)?;
            let mut data_file = std::io::BufReader::new(data_file);
            err_string = "Reading";
            let mut line_num: i64 = 0;

            // set number of numeric fields needed if numCol is not zero
            let mut num_needed: i64 = 1;
            if num_col > 0 {
                num_needed = num_col;
            }
            if if_types != 0 {
                num_needed += 1;
            }
            let separators = regex::Regex::new("[, \t]+").expect("separator pattern");
            while skip_lines < 0 || true_col <= 0 {
                // Read a line, strip, and split on valid table separators
                let mut raw = Vec::new();
                data_file.read_until(b'\n', &mut raw)?;
                if raw.is_empty() {
                    cleanup_error(
                        cleanup,
                        &data_name,
                        &format!(
                            "Reached the end of the file before finding any data in {data_name}"
                        ),
                    );
                }
                let line_text = String::from_utf8_lossy(&raw).into_owned();
                line_num += 1;
                if skip_lines >= line_num {
                    continue;
                }
                let line = line_text.trim();
                if line.is_empty() {
                    continue;
                }
                let lsplit: Vec<&str> = separators.split(line).collect();

                // Determine if there is an integer at start, number of leading numeric fields,
                // and if there is any non-numeric stuff
                let first_int = py_int(lsplit[0]);
                let mut num_fields: i64 = 0;
                let mut non_numeric = false;
                for field in &lsplit {
                    if py_float(field).is_some() {
                        num_fields += 1;
                    } else {
                        non_numeric = true;
                        break;
                    }
                }

                // If trying to determine number of lines to skip, see if this line is a data line
                if skip_lines < 0 {
                    if (num_col == 0 && first_int.is_some())
                        || (num_col < 0 && num_fields >= num_needed)
                        || (num_col > 0 && (!non_numeric || num_fields >= num_needed))
                    {
                        skip_lines = line_num - 1;
                    } else {
                        continue;
                    }
                }

                // This is the column number or data line
                if num_fields == 0 {
                    cleanup_error(
                        cleanup,
                        &data_name,
                        "First line of data file does not have expected numeric text",
                    );
                }
                if num_col < 0 {
                    true_col = num_fields - if_types;
                    if true_col == 0 {
                        cleanup_error(
                            cleanup,
                            &data_name,
                            "There seem to be only types in the file, not data",
                        );
                    }
                } else if num_col == 0 {
                    match first_int {
                        Some(value) if value > 0 => true_col = value,
                        _ => cleanup_error(
                            cleanup,
                            &data_name,
                            "Entry on line just before data is not an integer > 1",
                        ),
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = read {
            let text = error.to_string();
            let exc = match error.raw_os_error() {
                Some(errno) => format!(
                    "[Errno {errno}] {}: '{data_name}'",
                    text.strip_suffix(&format!(" (os error {errno})"))
                        .unwrap_or(&text)
                ),
                None => text,
            };
            cleanup_error(
                cleanup,
                &data_name,
                &format!("{err_string} data file {data_name}: {exc}"),
            );
        }
    }

    // Leave numCol as original value, it is passed to genhstplt; but use trueCol instead
    // from here on for number of columns
    // Check the column list for validity
    if !column_list.is_empty() {
        for ind in 0..column_list.len() {
            let col = column_list[ind];
            if col < 1 || col > true_col {
                cleanup_error(
                    cleanup,
                    &data_name,
                    &format!("{col} is an invalid column number"),
                );
            }
            // `columnList[0:ind - 1]`: Python slice bounds, so empty for ind 0
            // and 1
            let before_end = (ind as i64 - 1).clamp(0, column_list.len() as i64) as usize;
            let before = if ind == 0 {
                // `[0:-1]`: all but the last element
                &column_list[..column_list.len() - 1]
            } else {
                &column_list[..before_end]
            };
            if (ind > 0 && before.contains(&col))
                || (ind < column_list.len() - 1 && column_list[ind + 1..].contains(&col))
            {
                cleanup_error(
                    cleanup,
                    &data_name,
                    &format!("Column {col} is in the column list more than once"),
                );
            }
        }
    } else {
        // Or take care of default column, 1 and 2 unless there is only 1 or doing ordinals
        column_list = vec![1];
        if true_col > 1 && ordinals == 0 {
            column_list.push(2);
        }
    }

    if column_list.len() == 1 {
        ordinals = 1;
    }

    // Check the column/type entry, set up type conversion if needed
    let def_key_base;
    let mut xcol: i64 = 0;
    if if_types != 0 {
        def_key_base = "Type ";
        if column_list.len() > 2 {
            cleanup_error(
                cleanup,
                &data_name,
                "When plotting by types, you cannot enter more than two columns",
            );
        }
    } else {
        def_key_base = "Column ";
        if column_list.len() as i64 > 2 - ordinals {
            if_types = -1;
            type_list = column_list[(1 - ordinals) as usize..].to_vec();
            xcol = column_list[0];
            if ordinals != 0 {
                column_list = vec![2];
            } else {
                column_list = vec![1, 2];
            }
        }
    }

    // Take care of axis labels, keys and symbols if they haven't been entered
    if axis_label.is_empty() {
        if ordinals != 0 {
            axis_label = "Data number".to_owned();
        } else {
            axis_label = format!("Column {}", column_list[0]);
        }
    }

    // Set default colors for ones that weren't specified
    if use_dflt_hues && !type_list.is_empty() {
        let mut ind_hue = 0;
        for ind in 0..type_list.len() {
            if !colored_groups.contains(&(ind as i32 + 1)) {
                let stock_hue = stock_color(DEF_COLORS[ind_hue]).unwrap_or_default();
                colors.push(format!(
                    "{},{},{},{}",
                    ind + 1,
                    stock_hue[0],
                    stock_hue[1],
                    stock_hue[2]
                ));
                ind_hue = (ind_hue + 1) % DEF_COLORS.len();
            }
        }
    }

    let mut num_curves = 1;
    if if_types != 0 {
        num_curves = type_list.len();
    }
    if num_curves > NUM_KEY_LIMIT {
        prnstr(
            &format!("You will only see keys for {NUM_KEY_LIMIT} curves"),
            "\n",
            false,
        );
    }

    let index_error =
        || -> ! { super::pip::python_uncaught("IndexError: list index out of range") };
    for ind in num_keys..num_curves.min(NUM_KEY_LIMIT) {
        if if_types != 0 {
            keys.push(format!("{def_key_base}{}", type_list[ind]));
        } else {
            let index = ind + 1 - ordinals as usize;
            match column_list.get(index) {
                Some(value) => keys.push(format!("{def_key_base}{value}")),
                None => index_error(),
            }
        }
    }

    if symbols.is_empty() {
        symbols = Vec::new();
        if num_curves == 1 && ordinals != 0 {
            symbols = vec![0];
        }
    }
    for _ in symbols.len()..num_curves {
        match DEF_SYMBOLS.iter().find(|sym| !symbols.contains(sym)) {
            Some(sym) => symbols.push(*sym),
            None => cleanup_error(
                cleanup,
                &data_name,
                "There are too many types or columns being plotted for the symbols available",
            ),
        }
    }

    // Ready to build the awful input list
    let mut inlist: Vec<String> = vec![
        "-1".to_owned(),
        if_types.to_string(),
        num_col.to_string(),
        skip_lines.to_string(),
        data_name.clone(),
    ];
    if if_types < 0 {
        inlist.push(xcol.to_string());
    }
    if if_types != 0 {
        inlist.push((-(type_list.len() as i64)).to_string());
        for ind in 0..type_list.len() {
            match symbols.get(ind) {
                Some(sym) => inlist.push(format!("{},{sym}", type_list[ind])),
                None => index_error(),
            }
        }
    } else {
        inlist.push(symbols[0].to_string());
    }

    // columns
    if true_col > 1 {
        inlist.push(column_list[0].to_string());
    }
    inlist.extend([
        format!("{},{}", xlog as i64, py_str_float(xbase)),
        "0".to_owned(),
    ]);
    if ordinals != 0 {
        inlist.push("16".to_owned());
    }
    inlist.push("1".to_owned());
    if true_col > 1 {
        match column_list.get((1 - ordinals) as usize) {
            Some(value) => inlist.push(value.to_string()),
            None => index_error(),
        }
    }
    inlist.extend([
        format!("{},{}", ylog as i64, py_str_float(ybase)),
        "0".to_owned(),
        "-2".to_owned(),
        axis_label.clone(),
        num_curves.to_string(),
    ]);
    inlist.extend(keys.iter().cloned());
    if !colors.is_empty() {
        inlist.extend(["-4".to_owned(), colors.len().to_string()]);
        inlist.extend(colors.iter().cloned());
    }

    if ymin != 0. || ymax != 0. {
        inlist.extend([
            "-9".to_owned(),
            format!("{},{}", py_str_float(ymin), py_str_float(ymax)),
        ]);
    }
    if xmin != 0. || xmax != 0. {
        inlist.extend([
            "-10".to_owned(),
            format!("{},{}", py_str_float(xmin), py_str_float(xmax)),
        ]);
    }

    if connect != 0 {
        inlist.push("17".to_owned());
    } else {
        inlist.push("2".to_owned());
    }
    inlist.extend(["0".to_owned(), "0".to_owned()]);
    if !save_png_name.is_empty() {
        inlist.extend(["-7".to_owned(), save_png_name.clone(), "8".to_owned()]);
    } else {
        inlist.push("-8".to_owned());
    }

    if save_png_name.is_empty() {
        prnstr("Close graphing window to exit", "\n", true);
    }
    let mut cmd = format!("genhstplt{pos_option}{size_option}");
    if !message.is_empty() {
        cmd += &format!(" -message \"\"\"{message}\"\"\"");
    }
    if !tooltip.is_empty() {
        cmd += &format!(" -tooltip \"\"\"{tooltip}\"\"\"");
    }
    if run_cmd(&cmd, Some(&inlist), None, None, &[]).is_err() {
        if cleanup {
            cleanup_files(&[data_name.clone()]);
        }
        exit_from_imod_error(progname);
    }

    if cleanup {
        cleanup_files(&[data_name]);
    }
    done(0)
}
