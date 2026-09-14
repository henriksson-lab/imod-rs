//! Translation of `IMOD/libcfshr/parse_params.c` — the PIP package for parsing
//! input parameters.
//!
//! One Rust function per source function, with the original identifier in the
//! doc comment.  The unit is converted to Rust types per `NATIVE.md`: no raw
//! pointers, no `unsafe`, no `libc::`, no NUL-terminated strings.
//!
//! **Strings are bytes.**  Every `char *` in the source is `Vec<u8>`/`&[u8]`,
//! because option names, values and help text arrive from `argv` and from
//! autodoc files and the source copies them through byte for byte.  A `String`
//! round trip would replace a byte that is not valid UTF-8; nothing here parses
//! text, so nothing here needs one.  The C sentinel `sNullString` — a pointer
//! to a shared empty string that means "this slot owns nothing" — is
//! `Option<Vec<u8>>::None`, which is also how a NULL `char *` is spelled; the
//! two are distinguishable where the source distinguishes them (`defaultVal`).
//!
//! **Output.**  The source writes with `fprintf`/`printf`.  Every format string
//! here that has a conversion other than `%s` goes through
//! [`crate::imod::libcfshr::b3dutil::c_format`] with the source's own format
//! string; the rest — which in this unit is most of them, since PIP's output is
//! literals and `%s` — is written as the bytes C's `%s` would copy, because
//! `c_format` returns a `String` and would put U+FFFD in place of a byte that
//! an option value legitimately carries.  The stream is
//! [`ImodFile::Stdout`]/[`ImodFile::Stderr`], which are the *C* streams: the
//! programs that call PIP still write their own output with `libc::printf`, and
//! C stdio is block-buffered under redirection while Rust's is not, so PIP's
//! error and usage text would move ahead of theirs in a captured file if it
//! went through `std::io::stdout()`.
//!
//! **`strtol` and `strtod`** are translated below as the C library functions
//! they are, beside the code that calls them, for the same reason `c_format` is
//! a translation of `printf`: `str::parse` rejects the partial parses that
//! `PipGetLineOfValues` depends on, and the position where the scan stops is
//! the value the source compares against `endPtr`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use std::cell::{Cell, RefCell};
use std::io::Write;

/* #define NON_OPTION_STRING "NonOptionArgument" */
const NON_OPTION_STRING: &[u8] = b"NonOptionArgument";
/* #define STANDARD_INPUT_STRING "StandardInput" */
const STANDARD_INPUT_STRING: &[u8] = b"StandardInput";
/* #define STANDARD_INPUT_END  "EndInput" */
const STANDARD_INPUT_END: &[u8] = b"EndInput";
/* #define PARAM_FILE_STRING  "PF" */
const PARAM_FILE_STRING: &[u8] = b"PF";
/* #define BOOLEAN_STRING    "B" */
const BOOLEAN_STRING: &[u8] = b"B";
/* #define LOOKUP_NOT_FOUND -1 */
const LOOKUP_NOT_FOUND: i32 = -1;
/* #define LOOKUP_AMBIGUOUS -2 */
const LOOKUP_AMBIGUOUS: i32 = -2;
/* #define TEMP_STR_SIZE  1024 */
const TEMP_STR_SIZE: i32 = 1024;
/* #define LINE_STR_SIZE  102400 */
const LINE_STR_SIZE: i32 = 102400;
/* #define ADOC_STR_SIZE  10240 */
const ADOC_STR_SIZE: i32 = 10240;
/* #define PREFIX_SIZE    64 */
const PREFIX_SIZE: usize = 64;
/* #define PATH_SEPARATOR '/' on everything but _WIN32 */
const PATH_SEPARATOR: u8 = b'/';
/* #define OPTFILE_DIR "autodoc" */
const OPTFILE_DIR: &[u8] = b"autodoc";
/* #define OPTFILE_EXT "adoc" */
const OPTFILE_EXT: &[u8] = b"adoc";
/* #define OPTDIR_VARIABLE "AUTODOC_DIR" */
const OPTDIR_VARIABLE: &[u8] = b"AUTODOC_DIR";
/* #define DEFAULTS_FILE "progDefaults.adoc" */
const DEFAULTS_FILE: &[u8] = b"progDefaults.adoc";
/* #define DEFAULTS_DIR "com" */
const DEFAULTS_DIR: &[u8] = b"com";
/* #define DEFAULT_SUB_STR "%{default}" */
const DEFAULT_SUB_STR: &[u8] = b"%{default}";
/* #define PRINTENTRY_VARIABLE  "PIP_PRINT_ENTRIES" */
const PRINTENTRY_VARIABLE: &[u8] = b"PIP_PRINT_ENTRIES";
/* #define OPEN_DELIM "[" */
const OPEN_DELIM: &[u8] = b"[";
/* #define CLOSE_DELIM "]" */
const CLOSE_DELIM: &[u8] = b"]";
/* #define VALUE_DELIM "=" */
const VALUE_DELIM: &[u8] = b"=";
/* `<limits.h>` PATH_MAX, which `PipReadOptionFile` uses under `#ifdef`. */
const PATH_MAX: i32 = 4096;

/// `PIP_INTEGER` (`parse_params.h`).
pub const PIP_INTEGER: i32 = 1;
/// `PIP_FLOAT` (`parse_params.h`).
pub const PIP_FLOAT: i32 = 2;
/// `PIP_DOUBLE` (`parse_params.h`).
pub const PIP_DOUBLE: i32 = 3;

/// The structure for storing the options and the arguments as they are parsed.
///
/// C `typedef struct pipOptions { ... } PipOptions` (`parse_params.c:44`).  The
/// `char *` members are `Option<Vec<u8>>`, where `None` is both the source's
/// NULL and its `sNullString` sentinel; `char **valuePtr` with its parallel
/// `count` is a `Vec<Vec<u8>>`, and `count` is kept because the source reads and
/// writes it in its own right.
#[derive(Default, Clone)]
pub struct PipOptions {
    /// `char *shortName` — short option name.
    pub short_name: Option<Vec<u8>>,
    /// `char *longName` — long option name.
    pub long_name: Option<Vec<u8>>,
    /// `char *type` — type string.  Named `type_0` because `type` is a keyword.
    pub type_0: Option<Vec<u8>>,
    /// `char *helpString` — help string.
    pub help_string: Option<Vec<u8>>,
    /// `char *format` — value format, to appear after option for usage output.
    pub format: Option<Vec<u8>>,
    /// `char *defaultVal` — default value when option not entered.
    pub default_val: Option<Vec<u8>>,
    /// `char **valuePtr` — array of string pointers with values.
    pub value_ptr: Vec<Vec<u8>>,
    /// `int multiple` — 0 if single value allowed, or number of next one being
    /// returned (numbered from 1).
    pub multiple: i32,
    /// `int count` — number of values accumulated.
    pub count: i32,
    /// `int lenShort` — length of short name.
    pub len_short: i32,
    /// `int lenLong` — length of long name.
    pub len_long: i32,
    /// `int *nextLinked` — array of indexes of next non-option argument or
    /// linked option, for associating an entry with another one.
    pub next_linked: Vec<i32>,
    /// `int linked` — flag that it is a linked option.
    pub linked: i32,
}

/// The array an option's values are read into, standing for the source's
/// `void *array` plus its `valType` selector in `PipGetLineOfValues`.
///
/// C reaches the same storage through `int *`, `float *` and `double *` aliases
/// of one `void *`; the arm the caller builds is the alias the source's
/// `valType` selects.
pub enum PipValueArray<'a> {
    /// `int *iarray` — `PIP_INTEGER`.
    Int(&'a mut [i32]),
    /// `float *farray` — `PIP_FLOAT`.
    Float(&'a mut [f32]),
    /// `double *darray` — `PIP_DOUBLE`.
    Double(&'a mut [f64]),
}

/// Which local variable `CheckKeyword`'s `char ***lastCopied` was made to point
/// at.
///
/// The source stores the *address of a variable* so the next line's
/// continuation text can be appended to whichever string was gotten last, and
/// compares it with `lastGottenStr == &usageStr`.  A pointer to a local cannot
/// be held safely, so the identity of the variable is carried instead and the
/// comparisons become a match on this value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PipKeywordSlot {
    /// `&sValueDelim`
    ValueDelim,
    /// `&shortName`
    ShortName,
    /// `&longName`
    LongName,
    /// `&type`
    Type,
    /// `&formatStr`
    FormatStr,
    /// `&defaultStr`
    DefaultStr,
    /// `&usageStr`
    UsageStr,
    /// `&tipStr`
    TipStr,
    /// `&manStr`
    ManStr,
}

/* static char *sTypes[] = {...} */
const S_TYPES: [&[u8]; 13] = [
    BOOLEAN_STRING,
    PARAM_FILE_STRING,
    b"LI",
    b"I",
    b"F",
    b"IP",
    b"FP",
    b"IT",
    b"FT",
    b"IA",
    b"FA",
    b"CH",
    b"FN",
];
/* static char *sTypeDescriptions[] = {...} */
const S_TYPE_DESCRIPTIONS: [&[u8]; 14] = [
    b"Boolean",
    b"Parameter file",
    b"List of integer ranges",
    b"Integer",
    b"Floating point",
    b"Two integers",
    b"Two floats",
    b"Three integers",
    b"Three floats",
    b"Multiple integers",
    b"Multiple floats",
    b"Text string",
    b"File name",
    b"Unknown argument type",
];
/* static char *sTypeForUsage[] = {...} */
const S_TYPE_FOR_USAGE: [&[u8]; 14] = [
    b"Boolean",
    b"File",
    b"List",
    b"Int",
    b"Float",
    b"2 ints",
    b"2 floats",
    b"3 ints",
    b"3 floats",
    b"Ints",
    b"Floats",
    b"String",
    b"File",
    b"Unknown argument type",
];
/* static char sNumTypes = 13; -- a char, promoted to int at its one use */
const S_NUM_TYPES: i8 = 13;
/* static char *sQuoteTypes = "\"'`"; */
const S_QUOTE_TYPES: &[u8] = b"\"'`";

thread_local! {
    /* static int sHighestNonOptGotten = -1; */
    static S_HIGHEST_NON_OPT_GOTTEN: Cell<i32> = const { Cell::new(-1) };
    /* static PipOptions *sOptTable = NULL; */
    static S_OPT_TABLE: RefCell<Vec<PipOptions>> = const { RefCell::new(Vec::new()) };
    /* static int sTableSize = 0; */
    static S_TABLE_SIZE: Cell<i32> = const { Cell::new(0) };
    /* static int sNumOptions = 0; */
    static S_NUM_OPTIONS: Cell<i32> = const { Cell::new(0) };
    /* static int sNonOptInd; */
    static S_NON_OPT_IND: Cell<i32> = const { Cell::new(0) };
    /* static char *sErrorString = NULL; */
    static S_ERROR_STRING: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
    /* static char *sUsageString = NULL; */
    static S_USAGE_STRING: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
    /* static char sExitPrefix[PREFIX_SIZE] = ""; -- fixed buffer, NUL-terminated */
    static S_EXIT_PREFIX: RefCell<[u8; PREFIX_SIZE]> = const { RefCell::new([0; PREFIX_SIZE]) };
    /* static int sErrorDest = 0; */
    static S_ERROR_DEST: Cell<i32> = const { Cell::new(0) };
    /* static int sNextOption = 0; */
    static S_NEXT_OPTION: Cell<i32> = const { Cell::new(0) };
    /* static int sNextArgBelongsTo = -1; */
    static S_NEXT_ARG_BELONGS_TO: Cell<i32> = const { Cell::new(-1) };
    /* static int sNumOptionArguments = 0; */
    static S_NUM_OPTION_ARGUMENTS: Cell<i32> = const { Cell::new(0) };
    /* static char *sTempStr = NULL; -- a TEMP_STR_SIZE buffer holding a string */
    static S_TEMP_STR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /* static char *sLineStr = NULL; -- a LINE_STR_SIZE buffer holding a line */
    static S_LINE_STR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /* static int sAllowDefaults = 0; */
    static S_ALLOW_DEFAULTS: Cell<i32> = const { Cell::new(0) };
    /* static int sOutputManpage = 0; */
    static S_OUTPUT_MANPAGE: Cell<i32> = const { Cell::new(0) };
    /* static int sPrintEntries = -1; */
    static S_PRINT_ENTRIES: Cell<i32> = const { Cell::new(-1) };
    /* static char sDefaultDelim[] = VALUE_DELIM; static char *sValueDelim = sDefaultDelim; */
    static S_VALUE_DELIM: RefCell<Option<Vec<u8>>> = RefCell::new(Some(VALUE_DELIM.to_vec()));
    /* static char *sProgramName = &sNullChar; */
    static S_PROGRAM_NAME: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
    /* static int sNoCase = 0; */
    static S_NO_CASE: Cell<i32> = const { Cell::new(0) };
    /* static int sDoneEnds = 0; */
    static S_DONE_ENDS: Cell<i32> = const { Cell::new(0) };
    /* static int sTakeStdIn = 0; */
    static S_TAKE_STD_IN: Cell<i32> = const { Cell::new(0) };
    /* static int sNonOptLines = 0; */
    static S_NON_OPT_LINES: Cell<i32> = const { Cell::new(0) };
    /* static int sNoAbbrevs = 0; */
    static S_NO_ABBREVS: Cell<i32> = const { Cell::new(0) };
    /* static int sNotFoundOK = 0; */
    static S_NOT_FOUND_OK: Cell<i32> = const { Cell::new(0) };
    /* static char *sLinkedOption = NULL; */
    static S_LINKED_OPTION: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
    /* static int sTestAbbrevForUsage = 0; */
    static S_TEST_ABBREV_FOR_USAGE: Cell<i32> = const { Cell::new(0) };
    /* static int sDoubleDashOptions = 0; */
    static S_DOUBLE_DASH_OPTIONS: Cell<i32> = const { Cell::new(0) };
    /* static int sNoHelpAbbrevs = 0; */
    static S_NO_HELP_ABBREVS: Cell<i32> = const { Cell::new(0) };
}

/// Original C `PipInitialize` (`parse_params.c:157`).
///
/// Initialize option tables for given number of options.
pub fn pip_initialize(num_opts: i32) -> i32 {
    /* if (!sTempStr) sTempStr = malloc(TEMP_STR_SIZE);  sLineStr = malloc(...)
    -- allocation cannot fail here, so the NULL tests that follow, and the
    PipMemoryError they guard, are unreachable. */
    S_TEMP_STR.with_borrow_mut(|s| s.clear());
    S_LINE_STR.with_borrow_mut(|s| s.clear());

    /* Make the table big enough for extra entries (NonOptionArgs) */
    S_NUM_OPTIONS.set(num_opts);
    S_TABLE_SIZE.set(num_opts + 2);
    S_NON_OPT_IND.set(S_NUM_OPTIONS.get());
    let table_size = S_TABLE_SIZE.get();
    let non_opt_ind = S_NON_OPT_IND.get();

    /* Initialize the table */
    S_OPT_TABLE.with_borrow_mut(|table| {
        table.clear();
        for _i in 0..table_size {
            table.push(PipOptions::default());
        }

        /* In the last slots, put non-option arguments, and also put the
        name for the standard input option for easy checking on duplication */
        table[non_opt_ind as usize].long_name = Some(NON_OPTION_STRING.to_vec());
        table[non_opt_ind as usize + 1].short_name = Some(STANDARD_INPUT_STRING.to_vec());
        table[non_opt_ind as usize + 1].long_name = Some(STANDARD_INPUT_END.to_vec());
        table[non_opt_ind as usize].multiple = 1;
    });

    0
}

/// Original C `PipDone` (`parse_params.c:206`).
///
/// Free all allocated memory and set state back to initial state.
pub fn pip_done() {
    pip_warn_unused_non_opt_args();
    S_OPT_TABLE.with_borrow_mut(|table| table.clear());
    S_TABLE_SIZE.set(0);
    S_NUM_OPTIONS.set(0);
    S_ERROR_STRING.with_borrow_mut(|s| *s = None);
    S_USAGE_STRING.with_borrow_mut(|s| *s = None);
    S_LINKED_OPTION.with_borrow_mut(|s| *s = None);
    S_NEXT_OPTION.set(0);
    S_NEXT_ARG_BELONGS_TO.set(-1);
    S_NUM_OPTION_ARGUMENTS.set(0);
    S_ALLOW_DEFAULTS.set(0);
    S_TEMP_STR.with_borrow_mut(|s| s.clear());
    S_LINE_STR.with_borrow_mut(|s| s.clear());
    S_PROGRAM_NAME.with_borrow_mut(|s| *s = None);
}

/// Original C `PipWarnUnusedNonOptArgs` (`parse_params.c:248`).
///
/// Warn if extra non-option args entered.
pub fn pip_warn_unused_non_opt_args() -> i32 {
    let non_opt_ind = S_NON_OPT_IND.get();
    /* The source indexes sOptTable unconditionally; with no table at all there
    is nothing entered, so the count is zero. */
    let (count, values) = S_OPT_TABLE.with_borrow(|table| match table.get(non_opt_ind as usize) {
        Some(opt) => (opt.count, opt.value_ptr.clone()),
        None => (0, Vec::new()),
    });
    let unused = (count - 1) - S_HIGHEST_NON_OPT_GOTTEN.get();
    if unused > 0 {
        let mut out = ImodFile::Stdout;
        let _ = out.write_all(b"\nWARNING: Extra non-option arguments not used by the program:");
        for ind in (S_HIGHEST_NON_OPT_GOTTEN.get() + 1)..count {
            /* printf("  %s", sOptTable[sNonOptInd].valuePtr[ind]); */
            let _ = out.write_all(b"  ");
            let _ = out.write_all(&values[ind as usize]);
        }
        let _ = out.write_all(b"\n\n");
    }
    unused
}

/// Original C `PipExitOnError` (`parse_params.c:265`).
///
/// Set up for Pip to handle exiting on error, with a prefix string.
pub fn pip_exit_on_error(use_std_err: i32, prefix: &[u8]) -> i32 {
    /* Get rid of existing string; and if called with null string,
    this cancels an existing exit on error */
    S_EXIT_PREFIX.with_borrow_mut(|p| p[0] = 0x00);

    if prefix.is_empty() {
        return 0;
    }

    S_ERROR_DEST.set(use_std_err);
    /* strncpy(sExitPrefix, prefix, PREFIX_SIZE - 1); sExitPrefix[PREFIX_SIZE - 1] = 0; */
    S_EXIT_PREFIX.with_borrow_mut(|p| {
        *p = [0; PREFIX_SIZE];
        let n = prefix.len().min(PREFIX_SIZE - 1);
        p[..n].copy_from_slice(&prefix[..n]);
        p[PREFIX_SIZE - 1] = 0x00;
    });
    0
}

/// Original C `setExitPrefix` (`parse_params.c:283`).
///
/// Function for compatibility with Fortran routines, so `setExitPrefix` and
/// `exitError` can be used without using PIP.
#[allow(non_snake_case)]
pub fn setExitPrefix(prefix: &[u8]) {
    pip_exit_on_error(0, prefix);
}

/// Original C `setStandardExitPrefix` (`parse_params.c:288`).
#[allow(non_snake_case)]
pub fn setStandardExitPrefix(prog_name: &[u8]) {
    /* sprintf(prefix, "\nERROR: %s - ", progName); */
    let mut prefix: Vec<u8> = Vec::with_capacity(prog_name.len() + 15);
    prefix.extend_from_slice(b"\nERROR: ");
    prefix.extend_from_slice(prog_name);
    prefix.extend_from_slice(b" - ");
    pip_exit_on_error(0, &prefix);
}

/// Original C `PipAllowCommaDefaults` (`parse_params.c:298`).
pub fn pip_allow_comma_defaults(val: i32) {
    S_ALLOW_DEFAULTS.set(val);
}

/// Original C `PipSetManpageOutput` (`parse_params.c:303`).
pub fn pip_set_manpage_output(val: i32) {
    S_OUTPUT_MANPAGE.set(val);
}

/// Original C `PipSetUsageString` (`parse_params.c:308`).
pub fn pip_set_usage_string(usage: &[u8]) -> i32 {
    S_USAGE_STRING.with_borrow_mut(|s| *s = Some(usage.to_vec()));
    0
}

/// Original C `PipEnableEntryOutput` (`parse_params.c:317`).
pub fn pip_enable_entry_output(val: i32) {
    S_PRINT_ENTRIES.set(val);
}

/// Original C `PipSetLinkedOption` (`parse_params.c:322`).
pub fn pip_set_linked_option(option: &[u8]) -> i32 {
    S_LINKED_OPTION.with_borrow_mut(|s| *s = Some(option.to_vec()));
    0
}

/// Original C `PipSetSpecialFlags` (`parse_params.c:334`).
///
/// Set noxious special flags for Tilt program.
pub fn pip_set_special_flags(
    in_case: i32,
    in_done: i32,
    in_std: i32,
    in_lines: i32,
    in_abbrevs: i32,
) {
    S_NO_CASE.set(in_case);
    S_DONE_ENDS.set(in_done);
    S_TAKE_STD_IN.set(in_std);
    S_NON_OPT_LINES.set(in_lines);
    S_NO_ABBREVS.set(in_abbrevs);
}

/// Original C `PipAddOption` (`parse_params.c:347`).
///
/// Add an option, with short and long name, type, and help string.
pub fn pip_add_option(option_string: &[u8]) -> i32 {
    let next_option = S_NEXT_OPTION.get();
    if next_option >= S_NUM_OPTIONS.get() {
        pip_set_error(b"Attempting to add more options than were originally specified");
        return -1;
    }

    /* In the following, if there is ever not another :, skip to error */
    /* get the short name */
    let mut sub_str: &[u8] = option_string;
    let mut new_short: Vec<u8> = Vec::new();
    let mut new_long: Vec<u8> = Vec::new();
    let mut new_slen = 0i32;
    let mut new_llen = 0i32;
    let mut ok = false;

    if let Some(colon) = sub_str.iter().position(|&c| c == b':') {
        let ind_end = colon as i32;
        if ind_end > 0 {
            new_short = pip_sub_str_dup(sub_str, 0, ind_end - 1);
            new_slen = ind_end;
        } else {
            new_short = Vec::new();
            new_slen = 0;
        }
        sub_str = &sub_str[(ind_end + 1) as usize..];

        /* Get the long name */
        if let Some(colon) = sub_str.iter().position(|&c| c == b':') {
            let ind_end = colon as i32;
            if ind_end > 0 {
                new_long = pip_sub_str_dup(sub_str, 0, ind_end - 1);
                new_llen = ind_end;
            } else {
                new_long = Vec::new();
                new_llen = 0;
            }
            sub_str = &sub_str[(ind_end + 1) as usize..];

            /* Get the type and if there is M at the end, trim it off and set
            multiple flag to 1 */
            if let Some(colon) = sub_str.iter().position(|&c| c == b':') {
                let ind_end = colon as i32;
                let mut multiple = 0i32;
                let mut linked = 0i32;
                let type_0: Vec<u8>;
                if ind_end > 0 {
                    let mut ind = ind_end - 1;
                    if sub_str[ind as usize] == b'M' || sub_str[ind as usize] == b'L' {
                        multiple = 1;
                        if sub_str[ind as usize] == b'L' {
                            linked = 1;
                        }
                        ind -= 1;
                    }
                    type_0 = pip_sub_str_dup(sub_str, 0, ind);
                } else {
                    type_0 = Vec::new();
                }
                sub_str = &sub_str[(ind_end + 1) as usize..];

                /* Now if there is anything left, it is the help string */
                let help_string = sub_str.to_vec();

                S_OPT_TABLE.with_borrow_mut(|table| {
                    let optp = &mut table[next_option as usize];
                    optp.short_name = Some(new_short.clone());
                    optp.len_short = new_slen;
                    optp.long_name = Some(new_long.clone());
                    optp.len_long = new_llen;
                    optp.multiple = multiple;
                    optp.linked = linked;
                    optp.type_0 = Some(type_0);
                    optp.help_string = Some(help_string);
                });
                ok = true;
            }
        }
    }

    if ok {
        /* Need to check the short and long names against all previous
        names and special names */
        let table_size = S_TABLE_SIZE.get();
        let non_opt_ind = S_NON_OPT_IND.get();
        let no_abbrevs = S_NO_ABBREVS.get();
        for ind in 0..table_size {
            /* after checking existing ones, skip to NonOptionArg and
            StandardInput entries */
            if ind >= next_option && ind < non_opt_ind {
                continue;
            }

            let (old_short, old_long, old_slen, old_llen) = S_OPT_TABLE.with_borrow(|table| {
                let o = &table[ind as usize];
                (
                    o.short_name.clone(),
                    o.long_name.clone(),
                    o.len_short,
                    o.len_long,
                )
            });
            let os: &[u8] = old_short.as_deref().unwrap_or(b"");
            let ol: &[u8] = old_long.as_deref().unwrap_or(b"");

            /* Allow ambiguous options if no abbrev */
            if ((pip_starts_with(&new_short, os) != 0 || pip_starts_with(os, &new_short) != 0)
                && ((new_slen > 1 && old_slen > 1) || (new_slen == 1 && old_slen == 1))
                && (no_abbrevs == 0 || new_slen == old_slen))
                || ((pip_starts_with(ol, &new_short) != 0 || pip_starts_with(&new_short, ol) != 0)
                    && (no_abbrevs == 0 || new_slen == old_llen))
                || ((pip_starts_with(os, &new_long) != 0 || pip_starts_with(&new_long, os) != 0)
                    && (no_abbrevs == 0 || old_slen == new_llen))
                || ((pip_starts_with(ol, &new_long) != 0 || pip_starts_with(&new_long, ol) != 0)
                    && (no_abbrevs == 0 || old_llen == new_slen))
            {
                /* sprintf(sTempStr, "Option %s  %s is ambiguous with option %s  %s", ...)
                -- glibc's %s prints "(null)" for a NULL pointer, and the two
                trailing table entries have NULL names. */
                let temp = S_TEMP_STR.with_borrow_mut(|t| {
                    t.clear();
                    t.extend_from_slice(b"Option ");
                    t.extend_from_slice(&new_short);
                    t.extend_from_slice(b"  ");
                    t.extend_from_slice(&new_long);
                    t.extend_from_slice(b" is ambiguous with option ");
                    t.extend_from_slice(old_short.as_deref().unwrap_or(b"(null)"));
                    t.extend_from_slice(b"  ");
                    t.extend_from_slice(old_long.as_deref().unwrap_or(b"(null)"));
                    t.clone()
                });
                pip_set_error(&temp);
                return -1;
            }
        }

        S_NEXT_OPTION.set(next_option + 1);
        return 0;
    }

    /* sprintf(sTempStr, "Option does not have three colons in it:  "); */
    S_TEMP_STR.with_borrow_mut(|t| {
        t.clear();
        t.extend_from_slice(b"Option does not have three colons in it:  ");
    });
    append_to_error_string(option_string);
    -1
}

/// Original C `PipNextArg` (`parse_params.c:479`).
///
/// Call this to process the next argument.
pub fn pip_next_arg(arg_string: &[u8]) -> i32 {
    /* If we are expecting a value for an option, duplicate string and add
    it to the option */
    let next_arg_belongs_to = S_NEXT_ARG_BELONGS_TO.get();
    if next_arg_belongs_to >= 0 {
        let arg_copy = arg_string.to_vec();
        let mut err = add_value_string(next_arg_belongs_to, &arg_copy);

        /* Check whether this option was for reading from parameter file */
        let is_param_file = S_OPT_TABLE.with_borrow(|table| {
            table[next_arg_belongs_to as usize].type_0.as_deref() == Some(PARAM_FILE_STRING)
        });
        if err == 0 && is_param_file {
            /* fopen(argCopy, "r").  `ImodFile::open` takes a `&str`, and this
            path comes from `argv` as bytes, so the file is opened from the
            bytes and wrapped in the same type. */
            match std::fs::File::open(std::path::Path::new(
                <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(&arg_copy),
            ))
            .ok()
            .map(|f| ImodFile::File(std::rc::Rc::new(f)))
            {
                Some(mut param_file) => {
                    err = read_param_file(&mut param_file);
                }
                None => {
                    /* sprintf(sTempStr, "Error opening parameter file %s", argCopy); */
                    let temp = S_TEMP_STR.with_borrow_mut(|t| {
                        t.clear();
                        t.extend_from_slice(b"Error opening parameter file ");
                        t.extend_from_slice(&arg_copy);
                        t.clone()
                    });
                    pip_set_error(&temp);
                    err = -1;
                }
            }
        }

        S_NEXT_ARG_BELONGS_TO.set(-1);
        return err;
    }

    /* Is it a legal option starting with - or -- ? */
    if !arg_string.is_empty() && arg_string[0] == b'-' {
        let mut ind_start = 1usize;
        let lenarg = arg_string.len();
        if lenarg > 1 && arg_string[1] == b'-' {
            ind_start = 2;
        }
        if lenarg == ind_start {
            pip_set_error(b"Illegal argument: - or --");
            return -1;
        }

        /* First check for StandardInput */
        if pip_starts_with(STANDARD_INPUT_STRING, &arg_string[ind_start..]) != 0 {
            let mut stdin_file = ImodFile::Stdin;
            return read_param_file(&mut stdin_file);
        }

        /* Next check if it is a potential numeric non-option arg */
        S_NOT_FOUND_OK.set(1);
        for i in ind_start..lenarg {
            let ch = arg_string[i];
            if ch != b'-' && ch != b',' && ch != b'.' && ch != b' ' && (ch < b'0' || ch > b'9') {
                S_NOT_FOUND_OK.set(0);
                break;
            }
        }

        /* Lookup the option among true defined options */
        let err = lookup_option(&arg_string[ind_start..], S_NEXT_OPTION.get());

        /* Process as an option unless it could be numeric and was not found */
        if !(S_NOT_FOUND_OK.get() != 0 && err == LOOKUP_NOT_FOUND) {
            S_NOT_FOUND_OK.set(0);
            if err < 0 {
                return err;
            }

            S_NUM_OPTION_ARGUMENTS.set(S_NUM_OPTION_ARGUMENTS.get() + 1);

            /* For an option with value, setup to get argument next time and
            return an indicator that there had better be another */
            let is_boolean = S_OPT_TABLE
                .with_borrow(|table| table[err as usize].type_0.as_deref() == Some(BOOLEAN_STRING));
            if !is_boolean {
                S_NEXT_ARG_BELONGS_TO.set(err);
                return 1;
            } else {
                /* for a boolean option, set the argument with a 1 */
                return add_value_string(err, b"1");
            }
        }
        S_NOT_FOUND_OK.set(0);
    }

    /* A non-option argument.
    `expandArgList` (`b3dutil.c:1702-1707`) is a wild-card expansion that
    exists only under `_WIN32`; everywhere else it is the `#else` arm, which
    sets `*ifAlloc = 0`, `*noMatchInd = -1`, `*newNum = numArg` and returns
    the vector it was given.  Calling the translated `expand_arg_list` would
    put a `*const *const c_char` back into a converted unit, so the `#else`
    arm is written out here instead. */
    let new_num = 1;
    for _i in 0..new_num {
        let arg_copy = arg_string.to_vec();
        let err = add_value_string(S_NON_OPT_IND.get(), &arg_copy);
        if err != 0 {
            return err;
        }
    }
    0
}

/// Original C `PipNumberOfArgs` (`parse_params.c:586`).
///
/// Return number of option arguments (approximate) and number of non-option
/// arguments.
pub fn pip_number_of_args(num_opt_args: &mut i32, num_non_opt_args: &mut i32) {
    *num_opt_args = S_NUM_OPTION_ARGUMENTS.get();
    *num_non_opt_args = S_OPT_TABLE.with_borrow(|table| table[S_NON_OPT_IND.get() as usize].count);
}

/// Original C `PipGetNonOptionArg` (`parse_params.c:595`).
///
/// Get a non-option argument, index numbered from 0 here.
pub fn pip_get_non_option_arg(arg_no: i32, arg: &mut Vec<u8>) -> i32 {
    let non_opt_ind = S_NON_OPT_IND.get();
    let count = S_OPT_TABLE.with_borrow(|table| table[non_opt_ind as usize].count);
    if arg_no >= count {
        pip_set_error(b"Requested a non-option argument beyond the number available");
        return -1;
    }
    /* ACCUM_MAX(sHighestNonOptGotten, argNo); */
    if arg_no > S_HIGHEST_NON_OPT_GOTTEN.get() {
        S_HIGHEST_NON_OPT_GOTTEN.set(arg_no);
    }
    *arg = S_OPT_TABLE
        .with_borrow(|table| table[non_opt_ind as usize].value_ptr[arg_no as usize].clone());
    0
}

/// Original C `PipGetString` (`parse_params.c:610`).
pub fn pip_get_string(option: &[u8], string: &mut Vec<u8>) -> i32 {
    let mut str_ptr: Vec<u8> = Vec::new();
    let val_err = get_next_value_string(option, &mut str_ptr);
    if val_err == 0 || val_err == 2 {
        *string = str_ptr;
    }
    val_err
}

/// Original C `PipGetBoolean` (`parse_params.c:627`).
///
/// Get a boolean (binary) option; make sure it has a legal specification.
pub fn pip_get_boolean(option: &[u8], val: &mut i32) -> i32 {
    let mut str_ptr: Vec<u8> = Vec::new();
    let err = get_next_value_string(option, &mut str_ptr);
    if err != 0 {
        return err;
    }
    let s: &[u8] = &str_ptr;
    if s == b"1"
        || s == b"T"
        || s == b"TRUE"
        || s == b"ON"
        || s == b"t"
        || s == b"true"
        || s == b"on"
    {
        *val = 1;
    } else if s == b"0"
        || s == b"F"
        || s == b"FALSE"
        || s == b"OFF"
        || s == b"f"
        || s == b"false"
        || s == b"off"
    {
        *val = 0;
    } else {
        /* sprintf(sTempStr, "Illegal entry for boolean option %s: %s", option, strPtr); */
        let temp = S_TEMP_STR.with_borrow_mut(|t| {
            t.clear();
            t.extend_from_slice(b"Illegal entry for boolean option ");
            t.extend_from_slice(option);
            t.extend_from_slice(b": ");
            t.extend_from_slice(s);
            t.clone()
        });
        pip_set_error(&temp);
        return -1;
    }
    0
}

/// Original C `PipGetInteger` (`parse_params.c:654`).
pub fn pip_get_integer(option: &[u8], val: &mut i32) -> i32 {
    let mut num = 1;
    let mut array = [*val];
    let err = pip_get_integer_array(option, &mut array, &mut num, 1);
    *val = array[0];
    err
}

/// Original C `PipGetFloat` (`parse_params.c:660`).
pub fn pip_get_float(option: &[u8], val: &mut f32) -> i32 {
    let mut num = 1;
    let mut array = [*val];
    let err = pip_get_float_array(option, &mut array, &mut num, 1);
    *val = array[0];
    err
}

/// Original C `PipGetTwoIntegers` (`parse_params.c:669`).
pub fn pip_get_two_integers(option: &[u8], val1: &mut i32, val2: &mut i32) -> i32 {
    let mut num = 2;
    let mut tmp = [*val1, *val2];
    let err = pip_get_integer_array(option, &mut tmp, &mut num, 2);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    0
}

/// Original C `PipGetTwoFloats` (`parse_params.c:683`).
pub fn pip_get_two_floats(option: &[u8], val1: &mut f32, val2: &mut f32) -> i32 {
    let mut num = 2;
    let mut tmp = [*val1, *val2];
    let err = pip_get_float_array(option, &mut tmp, &mut num, 2);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    0
}

/// Original C `PipGetThreeIntegers` (`parse_params.c:700`).
pub fn pip_get_three_integers(
    option: &[u8],
    val1: &mut i32,
    val2: &mut i32,
    val3: &mut i32,
) -> i32 {
    let mut num = 3;
    let mut tmp = [*val1, *val2, *val3];
    let err = pip_get_integer_array(option, &mut tmp, &mut num, 3);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    *val3 = tmp[2];
    0
}

/// Original C `PipGetThreeFloats` (`parse_params.c:716`).
pub fn pip_get_three_floats(option: &[u8], val1: &mut f32, val2: &mut f32, val3: &mut f32) -> i32 {
    let mut num = 3;
    let mut tmp = [*val1, *val2, *val3];
    let err = pip_get_float_array(option, &mut tmp, &mut num, 3);
    if err != 0 {
        return err;
    }
    *val1 = tmp[0];
    *val2 = tmp[1];
    *val3 = tmp[2];
    0
}

/// Original C `PipGetIntegerArray` (`parse_params.c:736`).
pub fn pip_get_integer_array(
    option: &[u8],
    array: &mut [i32],
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    option_line_of_values(
        option,
        PipValueArray::Int(array),
        PIP_INTEGER,
        num_to_get,
        array_size,
    )
}

/// Original C `PipGetFloatArray` (`parse_params.c:742`).
pub fn pip_get_float_array(
    option: &[u8],
    array: &mut [f32],
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    option_line_of_values(
        option,
        PipValueArray::Float(array),
        PIP_FLOAT,
        num_to_get,
        array_size,
    )
}

/// Original C `PipPrintHelp` (`parse_params.c:760`).
///
/// Print a complete usage statement, man page entry, or program fallback code.
pub fn pip_print_help(
    prog_name: &[u8],
    use_std_err: i32,
    input_files: i32,
    output_files: i32,
) -> i32 {
    let mut num_out = 0i32;
    let mut num_real = 0i32;
    let helplim: i32 = 74;
    let mut out = if use_std_err != 0 {
        ImodFile::Stderr
    } else {
        ImodFile::Stdout
    };
    let indent4: &[u8] = b"    ";
    let mut line_pos = 11i32;
    let output_manpage = S_OUTPUT_MANPAGE.get();
    let fort90 = if output_manpage == -3 { 1 } else { 0 };
    let fort77 = if output_manpage == -2 { 1 } else { 0 };
    let c_code = if output_manpage == 2 { 1 } else { 0 };
    let python = if output_manpage == 3 { 1 } else { 0 };
    let fort_cont: &[u8] = if fort90 != 0 {
        b" &\n      '"
    } else {
        b"\n     &    '"
    };
    let mut descriptions: &[&[u8]; 14] = &S_TYPE_DESCRIPTIONS;
    let num_options = S_NUM_OPTIONS.get();
    let double_dash: &[u8] = if S_DOUBLE_DASH_OPTIONS.get() != 0 {
        b"-"
    } else {
        b""
    };

    /* Get correct number of options for Fortran fallback */
    for i in 0..num_options {
        let (sname, lname) = S_OPT_TABLE.with_borrow(|t| {
            (
                t[i as usize].short_name.clone(),
                t[i as usize].long_name.clone(),
            )
        });
        if lname.as_deref().is_some_and(|l| !l.is_empty())
            || sname.as_deref().is_some_and(|s| !s.is_empty())
        {
            num_real += 1;
        }
    }

    if output_manpage == 0 {
        let usage = S_USAGE_STRING.with_borrow(|u| u.clone());
        if let Some(usage) = usage {
            let _ = out.write_all(&usage);
        } else {
            let _ = out.write_all(b"Usage: ");
            let _ = out.write_all(prog_name);
            let _ = out.write_all(b" ");
            if num_options != 0 {
                let _ = out.write_all(b"[Options]");
            }
            if input_files != 0 {
                let _ = out.write_all(b" input_file");
            }
            if input_files > 1 {
                let _ = out.write_all(b"s...");
            }
            if output_files != 0 {
                let _ = out.write_all(b" output_file");
            }
            if output_files > 1 {
                let _ = out.write_all(b"s...");
            }
        }
        let _ = out.write_all(b"\n");

        if num_real == 0 {
            return 0;
        }
        if S_NO_HELP_ABBREVS.get() == 0 {
            let _ = out.write_all(
                b"Options can be abbreviated, current short name abbreviations are in parentheses\n",
            );
        }
        let _ = out.write_all(b"Options:\n");
        descriptions = &S_TYPE_FOR_USAGE;
    }

    S_TEST_ABBREV_FOR_USAGE.set(1);
    for i in 0..num_options {
        let (sname_opt, lname_opt, type_opt, format_opt, default_opt, help_opt, multiple, linked) =
            S_OPT_TABLE.with_borrow(|t| {
                let o = &t[i as usize];
                (
                    o.short_name.clone(),
                    o.long_name.clone(),
                    o.type_0.clone(),
                    o.format.clone(),
                    o.default_val.clone(),
                    o.help_string.clone(),
                    o.multiple,
                    o.linked,
                )
            });
        let sname: &[u8] = sname_opt.as_deref().unwrap_or(b"");
        let lname: &[u8] = lname_opt.as_deref().unwrap_or(b"");
        let type_0: &[u8] = type_opt.as_deref().unwrap_or(b"");
        let mut indent_str: &[u8] = b"";

        /* Try to look up an abbreviation of the short name */
        let mut abbrev_ok = 0i32;
        if !sname.is_empty() && S_NO_HELP_ABBREVS.get() == 0 {
            let mut jlim = sname.len() as i32 - 1;
            if jlim > TEMP_STR_SIZE - 10 {
                jlim = TEMP_STR_SIZE - 10;
            }
            for j in 0..jlim {
                /* sTempStr[j] = sname[j]; sTempStr[j + 1] = 0x00; */
                let probe = S_TEMP_STR.with_borrow_mut(|t| {
                    t.truncate(j as usize);
                    t.push(sname[j as usize]);
                    t.clone()
                });
                if lookup_option(&probe, num_options) == i {
                    abbrev_ok = 1;
                    break;
                }
            }
        }

        if !lname.is_empty() || !sname.is_empty() {
            if output_manpage <= 0 && fort90 == 0 {
                indent_str = indent4;
            }

            /* Output Fortran fallback code (-2) */
            if fort77 != 0 || fort90 != 0 {
                let last_opt = i == num_options - 1;
                if num_out == 0 {
                    let _ = out.write_all(
                        &c_format_bytes(
                            "%s  integer numOptions\n%s  parameter (numOptions = %d)\n%s  character*(40 * numOptions) options(1)\n%s  options(1) =%s",
                            &[
                                CArg::Bytes(indent_str),
                                CArg::Bytes(indent_str),
                                CArg::Int(num_real as i64),
                                CArg::Bytes(indent_str),
                                CArg::Bytes(indent_str),
                                CArg::Bytes(fort_cont),
                            ],
                        ),
                    );
                }

                let opt_len =
                    sname.len() as i32 + lname.len() as i32 + type_0.len() as i32 + 4 + multiple;

                if line_pos
                    + opt_len
                    + (if last_opt {
                        0
                    } else if fort90 != 0 {
                        5
                    } else {
                        3
                    })
                    > 90
                {
                    let _ = out.write_all(b"'//");
                    let _ = out.write_all(fort_cont);
                    line_pos = if fort90 != 0 { 7 } else { 11 };
                }
                /* fprintf(out, "%s:%s:%s%s%s", ...) */
                let _ = out.write_all(sname);
                let _ = out.write_all(b":");
                let _ = out.write_all(lname);
                let _ = out.write_all(b":");
                let _ = out.write_all(type_0);
                let _ = out.write_all(if multiple != 0 {
                    if linked != 0 { &b"L:"[..] } else { &b"M:"[..] }
                } else {
                    &b":"[..]
                });
                let _ = out.write_all(if last_opt { &b"'\n"[..] } else { &b"@"[..] });
                line_pos += opt_len;
                num_out += 1;
                continue;
            }

            /* Fallback output for C code (2) or Python code (3) */
            if c_code != 0 || python != 0 {
                let last_opt = i == num_options - 1;
                if num_out == 0 {
                    if c_code != 0 {
                        let _ = out.write_all(
                            c_format(
                                "  int numOptions = %d;\n  const char *options[] = {\n    ",
                                &[CArg::Int(num_real as i64)],
                            )
                            .as_bytes(),
                        );
                        line_pos = 5;
                    } else {
                        let _ = out.write_all(b"options = [");
                        line_pos = 12;
                    }
                }
                let opt_len =
                    sname.len() as i32 + lname.len() as i32 + type_0.len() as i32 + 7 + multiple;
                if line_pos + opt_len > 90 {
                    if c_code != 0 {
                        let _ = out.write_all(b"\n    ");
                        line_pos = 5;
                    } else {
                        /* If Emacs indents as it is pasted (it used to), take out leading spaces */
                        let _ = out.write_all(b"\n           ");
                        line_pos = 12;
                    }
                }
                /* fprintf(out, "\"%s:%s:%s%s\"%s", ...) */
                let _ = out.write_all(b"\"");
                let _ = out.write_all(sname);
                let _ = out.write_all(b":");
                let _ = out.write_all(lname);
                let _ = out.write_all(b":");
                let _ = out.write_all(type_0);
                let _ = out.write_all(if multiple != 0 {
                    if linked != 0 { &b"L:"[..] } else { &b"M:"[..] }
                } else {
                    &b":"[..]
                });
                let _ = out.write_all(b"\"");
                let _ = out.write_all(if last_opt {
                    if c_code != 0 {
                        &b"};\n"[..]
                    } else {
                        &b"]\n"[..]
                    }
                } else {
                    &b", "[..]
                });
                line_pos += opt_len;
                num_out += 1;
                continue;
            }

            if i != 0 && output_manpage < 0 {
                let _ = out.write_all(b"\n");
            }
            if output_manpage > 0 {
                let _ = out.write_all(b".TP\n.B ");
            }
            let _ = out.write_all(b" ");
            if !sname.is_empty() {
                let _ = out.write_all(double_dash);
                let _ = out.write_all(b"-");
                let _ = out.write_all(sname);
            }
            if abbrev_ok != 0 {
                let temp = S_TEMP_STR.with_borrow(|t| t.clone());
                let _ = out.write_all(b" (");
                let _ = out.write_all(double_dash);
                let _ = out.write_all(b"-");
                let _ = out.write_all(&temp);
                let _ = out.write_all(b")");
            }
            if !sname.is_empty() && !lname.is_empty() {
                let _ = out.write_all(b"  ");
                let _ = out.write_all(if output_manpage > 0 {
                    &b"\\fR"[..]
                } else {
                    &b""[..]
                });
                let _ = out.write_all(b"OR");
                let _ = out.write_all(if output_manpage > 0 {
                    &b"\\fP"[..]
                } else {
                    &b""[..]
                });
                let _ = out.write_all(b"  ");
            }
            if !lname.is_empty() {
                let _ = out.write_all(double_dash);
                let _ = out.write_all(b"-");
                let _ = out.write_all(lname);
            }

            /* Get index for description string */
            let mut j = 0i32;
            while j < S_NUM_TYPES as i32 {
                if type_0 == S_TYPES[j as usize] {
                    break;
                }
                j += 1;
            }

            /* If there is a format, then for manpage output, wrap it in \fI-\fR unless it
            already starts with \f.  For usage output, strip \fx */
            if let Some(format) = format_opt.as_deref() {
                if output_manpage > 0 {
                    let hasbf = pip_starts_with(format, b"\\f");
                    let _ = out.write_all(b" \t ");
                    let _ = out.write_all(if hasbf == 0 { &b"\\fI"[..] } else { &b""[..] });
                    let _ = out.write_all(format);
                    let _ = out.write_all(if hasbf == 0 { &b"\\fR"[..] } else { &b""[..] });
                } else {
                    let opt_len = format.len() as i32;
                    let _ = out.write_all(b"   ");
                    let mut k = 0i32;
                    while k < opt_len {
                        if pip_starts_with(&format[k as usize..], b"\\f") != 0 {
                            k += 2;
                        } else {
                            let _ = out.write_all(&format[k as usize..k as usize + 1]);
                        }
                        k += 1;
                    }
                }
            /* Otherwise output the description string, inside \fI \fR for man output */
            } else if type_0 != BOOLEAN_STRING {
                let _ = out.write_all(if output_manpage > 0 {
                    &b" \t \\fI"[..]
                } else {
                    &b"   "[..]
                });
                let _ = out.write_all(descriptions[j as usize]);
                let _ = out.write_all(if output_manpage > 0 {
                    &b"\\fR"[..]
                } else {
                    &b""[..]
                });
            }
            let _ = out.write_all(b"\n");
        } else if fort77 != 0 || fort90 != 0 || c_code != 0 || python != 0 {
            continue;
        } else if output_manpage == 1 {
            let _ = out.write_all(b".SS ");
        } else {
            let _ = out.write_all(b"\n");
        }

        /* Print help string, breaking up line as needed */
        if help_opt.as_deref().is_some_and(|h| !h.is_empty()) {
            let mut help: Vec<u8> = help_opt.unwrap();

            /* First look for default variable string and replace it with default */
            if let Some(default_val) = default_opt.as_deref() {
                while (help.len() + default_val.len()) < (LINE_STR_SIZE - 10) as usize {
                    let def_pos = match help
                        .windows(DEFAULT_SUB_STR.len())
                        .position(|w| w == DEFAULT_SUB_STR)
                    {
                        Some(p) => p,
                        None => break,
                    };
                    /* strncpy(sLineStr, lname, defPtr - lname);
                    sprintf(&sLineStr[defPtr - lname], "%s%s", defaultVal,
                            defPtr + strlen(DEFAULT_SUB_STR));
                    free(lname); lname = strdup(sLineStr); */
                    let mut line: Vec<u8> = Vec::new();
                    line.extend_from_slice(&help[..def_pos]);
                    line.extend_from_slice(default_val);
                    line.extend_from_slice(&help[def_pos + DEFAULT_SUB_STR.len()..]);
                    S_LINE_STR.with_borrow_mut(|l| *l = line.clone());
                    help = line;
                }
            }

            /* sname = lname; -- a walking pointer into the duplicated string */
            let mut pos = 0usize;
            let mut opt_len = help.len() as i32;
            let mut new_line_pt = help.iter().position(|&c| c == b'\n');
            while opt_len > helplim || new_line_pt.is_some() {
                /* Break string at newline */
                let mut broke_at_new_line = 0i32;
                let broke_at_space;
                let mut j: i32;
                if new_line_pt.is_some_and(|p| p >= pos && (p - pos) as i32 <= helplim) {
                    j = (new_line_pt.unwrap() - pos) as i32;
                    new_line_pt = help[pos + j as usize + 1..]
                        .iter()
                        .position(|&c| c == b'\n')
                        .map(|p| p + pos + j as usize + 1);
                    broke_at_space = 0;
                    broke_at_new_line = 1;
                } else {
                    /* Or break string at last space before limit */
                    j = helplim;
                    while j >= 1 {
                        if help[pos + j as usize] == b' ' {
                            break;
                        }
                        j -= 1;
                    }
                    broke_at_space = 1;
                }

                /* For manpage output, insert zero-width character if line starts with . or ' */
                if output_manpage > 0 && (help[pos] == b'.' || help[pos] == b'\'') {
                    let _ = out.write_all(b"\\&");
                }

                /* Replace break point with null, print and reset pointer and count */
                let _ = out.write_all(indent_str);
                let _ = out.write_all(&help[pos..pos + j as usize]);
                let _ = out.write_all(b"\n");
                /* sname[j + 1] reads the byte after the break point, which at the
                end of the string is the C terminator. */
                let after_break = |k: usize| -> u8 {
                    if pos + k + 1 < help.len() {
                        help[pos + k + 1]
                    } else {
                        0
                    }
                };
                if output_manpage == 1 && broke_at_new_line != 0 && after_break(j as usize) != b' '
                {
                    let _ = out.write_all(b".br\n");
                }
                if broke_at_space != 0 && output_manpage > 0 {
                    while after_break(j as usize) == b' ' {
                        j += 1;
                    }
                }
                pos += j as usize + 1;
                opt_len -= j + 1;
            }
            let _ = out.write_all(indent_str);
            let _ = out.write_all(&help[pos..]);
            let _ = out.write_all(b"\n");
        }

        if linked != 0 {
            let _ = out.write_all(indent_str);
            let _ = out.write_all(b"(Multiple entries linked to a different option)\n");
        } else if multiple != 0 {
            let _ = out.write_all(indent_str);
            let _ = out.write_all(b"(Successive entries accumulate)\n");
        }
    }
    S_TEST_ABBREV_FOR_USAGE.set(0);
    let _ = out.flush();
    0
}

/// Original C `PipPrintEntries` (`parse_params.c:1006`).
///
/// Print all the option entries if enabled by program call and/or environment
/// variable.
pub fn pip_print_entries() {
    if S_PRINT_ENTRIES.get() < 0 {
        S_PRINT_ENTRIES.set(0);
        /* name = getenv(PRINTENTRY_VARIABLE); if (name) sPrintEntries = atoi(name); */
        if let Some(name) = std::env::var_os(std::ffi::OsStr::new(
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(PRINTENTRY_VARIABLE),
        )) {
            let bytes =
                <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::as_bytes(name.as_os_str());
            let mut end = 0usize;
            S_PRINT_ENTRIES.set(strtol(bytes, &mut end, 10) as i32);
        }
    }
    if S_PRINT_ENTRIES.get() == 0 {
        return;
    }

    let num_options = S_NUM_OPTIONS.get();
    let non_opt_ind = S_NON_OPT_IND.get();

    /* Count up entries */
    let mut j = 0i32;
    for i in 0..num_options {
        j += S_OPT_TABLE.with_borrow(|t| t[i as usize].count);
    }
    let non_opt_count = S_OPT_TABLE.with_borrow(|t| t[non_opt_ind as usize].count);
    if j + non_opt_count == 0 {
        return;
    }

    let mut out = ImodFile::Stdout;
    let program_name = S_PROGRAM_NAME.with_borrow(|p| p.clone());
    let _ = out.write_all(b"\n*** Entries to program ");
    let _ = out.write_all(program_name.as_deref().unwrap_or(b""));
    let _ = out.write_all(b" ***\n");
    for i in 0..num_options {
        let (sname, lname, values) = S_OPT_TABLE.with_borrow(|t| {
            let o = &t[i as usize];
            (
                o.short_name.clone(),
                o.long_name.clone(),
                o.value_ptr.clone(),
            )
        });
        let s: &[u8] = sname.as_deref().unwrap_or(b"");
        let l: &[u8] = lname.as_deref().unwrap_or(b"");
        if !l.is_empty() || !s.is_empty() {
            let name: &[u8] = if !l.is_empty() { l } else { s };
            let count = S_OPT_TABLE.with_borrow(|t| t[i as usize].count);
            for k in 0..count {
                let _ = out.write_all(b"  ");
                let _ = out.write_all(name);
                let _ = out.write_all(b" = ");
                let _ = out.write_all(&values[k as usize]);
                let _ = out.write_all(b"\n");
            }
        }
    }
    if non_opt_count != 0 {
        let values = S_OPT_TABLE.with_borrow(|t| t[non_opt_ind as usize].value_ptr.clone());
        let _ = out.write_all(b"  Non-option arguments:");
        for k in 0..non_opt_count {
            let v = &values[k as usize];
            if v.contains(&b' ') {
                let _ = out.write_all(b"   \"");
                let _ = out.write_all(v);
                let _ = out.write_all(b"\"");
            } else {
                let _ = out.write_all(b"   ");
                let _ = out.write_all(v);
            }
        }
        let _ = out.write_all(b"\n");
    }
    let _ = out.write_all(b"*** End of entries ***\n\n");

    /* Windows needed two flushes here */
    let _ = out.flush();
    let _ = out.flush();
}

/// Original C `PipGetError` (`parse_params.c:1059`).
///
/// Return the error string, or an empty string and an error if there is none.
pub fn pip_get_error(err_string: &mut Vec<u8>) -> i32 {
    match S_ERROR_STRING.with_borrow(|e| e.clone()) {
        None => {
            *err_string = Vec::new();
            -1
        }
        Some(s) => {
            *err_string = s;
            0
        }
    }
}

/// Original C `PipSetError` (`parse_params.c:1076`).
///
/// Set the error string.  If `sExitPrefix` is set, then output an error message
/// to stderr or stdout and exit.
pub fn pip_set_error(err_string: &[u8]) -> i32 {
    S_ERROR_STRING.with_borrow_mut(|e| *e = Some(err_string.to_vec()));

    let has_prefix = S_EXIT_PREFIX.with_borrow(|p| p[0] != 0);
    if has_prefix {
        /* `fprintf(outFile, "%s ", sExitPrefix)` -- the space is a second one
        after a prefix that already ends in one -- then `"%s\n"` with the
        message, on *stdout* unless sErrorDest was set. */
        let prefix = S_EXIT_PREFIX.with_borrow(|p| {
            let n = p.iter().position(|&c| c == 0).unwrap_or(PREFIX_SIZE);
            p[..n].to_vec()
        });
        let mut out = if S_ERROR_DEST.get() != 0 {
            ImodFile::Stderr
        } else {
            ImodFile::Stdout
        };
        let _ = out.write_all(&prefix);
        let _ = out.write_all(b" ");
        let _ = out.write_all(err_string);
        let _ = out.write_all(b"\n");
        let _ = out.flush();
        /* C `exit` flushes every stdio stream; standard output can be holding a
        partial line written by the program before the error. */
        let _ = ImodFile::Stdout.flush();
        std::process::exit(1);
    }
    0
}

/// Original C `exitError` (`parse_params.c:1093`).
///
/// The source is variadic and `vsprintf`s into a 512-byte buffer; the caller
/// formats the message here, as the translated callers already did.
pub fn exit_error(format: &[u8]) -> ! {
    pip_set_error(format);
    /* PipSetError already exited when an exit prefix is set. */
    let _ = ImodFile::Stdout.flush();
    std::process::exit(1);
}

/// Original C `PipNumberOfEntries` (`parse_params.c:1106`).
///
/// Return the number of entries for a particular option.
pub fn pip_number_of_entries(option: &[u8], num_entries: &mut i32) -> i32 {
    let err = lookup_option(option, S_NON_OPT_IND.get() + 1);
    if err < 0 {
        return err;
    }
    *num_entries = S_OPT_TABLE.with_borrow(|t| t[err as usize].count);
    0
}

/// Original C `PipLinkedIndex` (`parse_params.c:1118`).
///
/// Return the index of the next non-option arg or linked option that was
/// entered after this option.
pub fn pip_linked_index(option: &[u8], index: &mut i32) -> i32 {
    let mut which = 0i32;
    let err = lookup_option(option, S_NON_OPT_IND.get() + 1);
    if err < 0 {
        return err;
    }
    let (linked, multiple) =
        S_OPT_TABLE.with_borrow(|t| (t[err as usize].linked, t[err as usize].multiple));
    if linked == 0 {
        /* sprintf(sTempStr, "Trying to get a linked index for option %s, which is not "
        "identified as linked", option); */
        let temp = S_TEMP_STR.with_borrow_mut(|t| {
            t.clear();
            t.extend_from_slice(b"Trying to get a linked index for option ");
            t.extend_from_slice(option);
            t.extend_from_slice(b", which is not identified as linked");
            t.clone()
        });
        pip_set_error(&temp);
        return -1;
    }
    let mut ind = 0i32;
    if multiple != 0 {
        ind = multiple - 1;
    }

    /* Use count from non-option args, but use the count from the linked option
    instead if it was entered at all.  This allows other non-option args to be used */
    let linked_option = S_LINKED_OPTION.with_borrow(|l| l.clone());
    if let Some(linked_option) = linked_option {
        let ilink = lookup_option(&linked_option, S_NUM_OPTIONS.get());
        if ilink < 0 {
            return ilink;
        }
        if S_OPT_TABLE.with_borrow(|t| t[ilink as usize].count) != 0 {
            which = 1;
        }
    }
    *index = S_OPT_TABLE.with_borrow(|t| t[err as usize].next_linked[(2 * ind + which) as usize]);
    0
}

/// Original C `PipParseInput` (`parse_params.c:1160`).
///
/// Top level routine to be called to process options and arguments.
pub fn pip_parse_input(
    argc: i32,
    argv: &[Vec<u8>],
    options: &[&[u8]],
    num_opts: i32,
    num_opt_args: &mut i32,
    num_non_opt_args: &mut i32,
) -> i32 {
    /* Initialize */
    let err = pip_initialize(num_opts);
    if err != 0 {
        return err;
    }

    /* add the options */
    for i in 0..num_opts {
        let err = pip_add_option(options[i as usize]);
        if err != 0 {
            return err;
        }
    }

    pip_parse_entries(argc, argv, num_opt_args, num_non_opt_args)
}

/// Original C `PipReadOptionFile` (`parse_params.c:1182`).
///
/// Alternative routine to have options read from a file.
pub fn pip_read_option_file(prog_name: &[u8], help_level: i32, local_dir: i32) -> i32 {
    let mut is_section = 0i32;
    let mut opt_file: Option<ImodFile> = None;
    let mut num_opts = 0i32;
    let mut big_size = ADOC_STR_SIZE;
    let mut reading_opt = 0i32;
    let mut got_delim = 0i32;
    let mut in_quote_index = -1i32;
    let mut last_gotten_str: Option<PipKeywordSlot> = None;

    /* #ifdef PATH_MAX */
    if big_size < PATH_MAX {
        big_size = PATH_MAX;
    }

    /* Set up temp string for error processing and big string for lines */
    let mut big_str: Vec<u8> = Vec::new();

    /* Save the program name for entry output */
    S_PROGRAM_NAME.with_borrow_mut(|p| *p = Some(prog_name.to_vec()));

    /* If local directory not set, look for environment variable pointing
    directly to where the file should be */
    if local_dir == 0 {
        if let Some(pip_dir) = std::env::var_os(std::ffi::OsStr::new(
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(OPTDIR_VARIABLE),
        )) {
            let pip_dir =
                <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::as_bytes(pip_dir.as_os_str())
                    .to_vec();
            if pip_dir.len() as i32 > big_size - 100 {
                pip_set_error(b"AUTODOC_DIR is suspiciously long");
                return -1;
            }
            /* sprintf(bigStr, "%s%c%s.%s", pipDir, PATH_SEPARATOR, progName, OPTFILE_EXT); */
            big_str.clear();
            big_str.extend_from_slice(&pip_dir);
            big_str.push(PATH_SEPARATOR);
            big_str.extend_from_slice(prog_name);
            big_str.push(b'.');
            big_str.extend_from_slice(OPTFILE_EXT);
            opt_file = std::fs::File::open(std::path::Path::new(
                <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(&big_str),
            ))
            .ok()
            .map(|f| ImodFile::File(std::rc::Rc::new(f)));
        }

        if opt_file.is_none() {
            if let Some(pip_dir) = std::env::var_os("IMOD_DIR") {
                let pip_dir = <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::as_bytes(
                    pip_dir.as_os_str(),
                )
                .to_vec();
                if pip_dir.len() as i32 > big_size - 100 {
                    pip_set_error(b"IMOD_DIR is suspiciously long");
                    return -1;
                }
                /* sprintf(bigStr, "%s%c%s%c%s.%s", ...) */
                big_str.clear();
                big_str.extend_from_slice(&pip_dir);
                big_str.push(PATH_SEPARATOR);
                big_str.extend_from_slice(OPTFILE_DIR);
                big_str.push(PATH_SEPARATOR);
                big_str.extend_from_slice(prog_name);
                big_str.push(b'.');
                big_str.extend_from_slice(OPTFILE_EXT);
                opt_file = std::fs::File::open(std::path::Path::new(
                    <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(&big_str),
                ))
                .ok()
                .map(|f| ImodFile::File(std::rc::Rc::new(f)));
            }
        }
    }
    /* If local directory set, set up name with ../ as many times as specified
    and look for file there */
    else if local_dir > 0 {
        big_str.clear();
        let mut i = 0;
        while i < local_dir && i < 20 {
            big_str.push(b'.');
            big_str.push(b'.');
            big_str.push(PATH_SEPARATOR);
            i += 1;
        }
        big_str.extend_from_slice(OPTFILE_DIR);
        big_str.push(PATH_SEPARATOR);
        big_str.extend_from_slice(prog_name);
        big_str.push(b'.');
        big_str.extend_from_slice(OPTFILE_EXT);
        opt_file = std::fs::File::open(std::path::Path::new(
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(&big_str),
        ))
        .ok()
        .map(|f| ImodFile::File(std::rc::Rc::new(f)));
    }

    /* If there is still no file, look in current directory */
    if opt_file.is_none() {
        big_str.clear();
        big_str.extend_from_slice(prog_name);
        big_str.push(b'.');
        big_str.extend_from_slice(OPTFILE_EXT);
        opt_file = std::fs::File::open(std::path::Path::new(
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(&big_str),
        ))
        .ok()
        .map(|f| ImodFile::File(std::rc::Rc::new(f)));

        if opt_file.is_none() {
            /* sprintf(bigStr, "Autodoc file %s.%s was not found ...") */
            big_str.clear();
            big_str.extend_from_slice(b"Autodoc file ");
            big_str.extend_from_slice(prog_name);
            big_str.push(b'.');
            big_str.extend_from_slice(OPTFILE_EXT);
            big_str.extend_from_slice(
                b" was not found or not readable.\nCheck environment variable settings of AUTODOC_DIR and IMOD_DIR\nor place autodoc file in current directory",
            );
            pip_set_error(&big_str);
            return -1;
        }
    }
    let mut opt_file = opt_file.unwrap();

    /* Count up the options */
    let mut indst = 0i32;
    loop {
        let line_len = pip_read_next_line(
            &mut opt_file,
            &mut big_str,
            big_size,
            b'#',
            0,
            0,
            &mut indst,
        );
        if line_len == -3 {
            break;
        }
        if line_len == -2 {
            pip_set_error(b"Error reading option file");
            return -1;
        }

        /* If the string was not long enough, get a bigger string and start over */
        if line_len == -1 {
            big_size += ADOC_STR_SIZE;
            big_str.clear();
            num_opts = 0;
            let _ = std::io::Seek::seek(&mut opt_file, std::io::SeekFrom::Start(0));
            continue;
        }

        /* Look for new keyword-value delimiter before any options */
        if num_opts == 0 {
            let text: Vec<u8> = big_str[indst as usize..].to_vec();
            let mut delim = S_VALUE_DELIM.with_borrow(|d| d.clone());
            check_keyword(
                &text,
                b"KeyValueDelimiter",
                &mut delim,
                &mut got_delim,
                &mut last_gotten_str,
                PipKeywordSlot::ValueDelim,
                None,
            );
            S_VALUE_DELIM.with_borrow_mut(|d| *d = delim);
            if pip_starts_with(&text, b"DoubleDashOptions") != 0 {
                S_DOUBLE_DASH_OPTIONS.set(1);
            }
            if pip_starts_with(&text, b"NoHelpAbbreviations") != 0 {
                S_NO_HELP_ABBREVS.set(1);
            }
            if pip_starts_with(&text, b"NoAbbreviations") != 0 {
                S_NO_ABBREVS.set(1);
            }
        }

        /* Look for options */
        if line_is_option_token(&big_str[indst as usize..]) > 0 {
            num_opts += 1;
        }
    }

    /* Initialize */
    let err = pip_initialize(num_opts);
    if err != 0 {
        return err;
    }

    /* rewind file and process the options */
    let _ = std::io::Seek::seek(&mut opt_file, std::io::SeekFrom::Start(0));
    let mut long_name: Option<Vec<u8>> = None;
    let mut short_name: Option<Vec<u8>> = None;
    let mut type_0: Option<Vec<u8>> = None;
    let mut usage_str: Option<Vec<u8>> = None;
    let mut tip_str: Option<Vec<u8>> = None;
    let mut man_str: Option<Vec<u8>> = None;
    let mut format_str: Option<Vec<u8>> = None;
    let mut default_str: Option<Vec<u8>> = None;
    let (mut got_long, mut got_short, mut got_type, mut got_usage) = (0i32, 0i32, 0i32, 0i32);
    let (mut got_tip, mut got_man, mut got_format, mut got_default) = (0i32, 0i32, 0i32, 0i32);

    loop {
        let line_len = pip_read_next_line(
            &mut opt_file,
            &mut big_str,
            big_size,
            b'#',
            0,
            0,
            &mut indst,
        );
        if line_len == -2 {
            pip_set_error(b"Error reading autodoc file");
            return -1;
        }

        /* textStr = bigStr + indst; on end of file indst still names the start of
        the previous line, which is what the source reads too. */
        let mut text_str: Vec<u8> = if (indst as usize) <= big_str.len() {
            big_str[indst as usize..].to_vec()
        } else {
            Vec::new()
        };
        let is_option = line_is_option_token(&text_str);
        if reading_opt != 0 && (line_len == -3 || is_option != 0) {
            /* If we were reading options, it is time to add them if we are at
            end of file or if we have reached a new token of any kind

            Pick the closest help string that was read in if the given one
            does not match (there has got to be an easier way!) */
            let help_str: Vec<u8> = if help_level <= 1 {
                if got_usage != 0 {
                    usage_str.clone().unwrap_or_default()
                } else if got_tip != 0 {
                    tip_str.clone().unwrap_or_default()
                } else {
                    man_str.clone().unwrap_or_default()
                }
            } else if help_level == 2 {
                if got_tip != 0 {
                    tip_str.clone().unwrap_or_default()
                } else if got_usage != 0 {
                    usage_str.clone().unwrap_or_default()
                } else {
                    man_str.clone().unwrap_or_default()
                }
            } else if got_man != 0 {
                man_str.clone().unwrap_or_default()
            } else if got_tip != 0 {
                tip_str.clone().unwrap_or_default()
            } else {
                usage_str.clone().unwrap_or_default()
            };

            /* If it is a section header, get rid of the names */
            if is_section != 0 {
                short_name = None;
                long_name = None;
                got_long = 0;
                got_short = 0;
            }

            /* sprintf(optStr, "%s:%s:%s:%s", shortName, longName, type, helpStr); */
            let mut opt_str: Vec<u8> = Vec::new();
            opt_str.extend_from_slice(short_name.as_deref().unwrap_or(b""));
            opt_str.push(b':');
            opt_str.extend_from_slice(long_name.as_deref().unwrap_or(b""));
            opt_str.push(b':');
            opt_str.extend_from_slice(type_0.as_deref().unwrap_or(b""));
            opt_str.push(b':');
            opt_str.extend_from_slice(&help_str);

            let err = pip_add_option(&opt_str);
            if err != 0 {
                return err;
            }

            /* Assign format and default if got one */
            let next_option = S_NEXT_OPTION.get();
            if got_format != 0 {
                let f = format_str.take();
                S_OPT_TABLE.with_borrow_mut(|t| t[(next_option - 1) as usize].format = f);
            }
            if got_default != 0 {
                let d = default_str.take();
                S_OPT_TABLE.with_borrow_mut(|t| t[(next_option - 1) as usize].default_val = d);
            }

            /* Clean up memory and reset flags */
            long_name = None;
            short_name = None;
            type_0 = None;
            usage_str = None;
            tip_str = None;
            man_str = None;
            format_str = None;
            default_str = None;
            got_long = 0;
            got_short = 0;
            got_type = 0;
            got_usage = 0;
            got_tip = 0;
            got_man = 0;
            got_format = 0;
            got_default = 0;
            reading_opt = 0;
        }

        if line_len == -3 {
            break;
        }

        /* If reading options, look for the various keywords */
        if reading_opt != 0 {
            /* If the last string gotten was a help string and the line does not contain the
            value delimiter or we are in a quote, then append it to the last string */
            let value_delim = S_VALUE_DELIM.with_borrow(|d| d.clone()).unwrap_or_default();
            let has_delim = if value_delim.is_empty() {
                true
            } else {
                text_str
                    .windows(value_delim.len())
                    .any(|w| w == value_delim.as_slice())
            };
            let is_help_slot = matches!(
                last_gotten_str,
                Some(PipKeywordSlot::UsageStr)
                    | Some(PipKeywordSlot::TipStr)
                    | Some(PipKeywordSlot::ManStr)
            );
            if is_help_slot && (in_quote_index >= 0 || !has_delim) {
                let target: &mut Option<Vec<u8>> = match last_gotten_str {
                    Some(PipKeywordSlot::UsageStr) => &mut usage_str,
                    Some(PipKeywordSlot::TipStr) => &mut tip_str,
                    _ => &mut man_str,
                };
                let cur = target.get_or_insert_with(Vec::new);
                let ind = cur.len();
                let len = text_str.len();
                /* strcat(*lastGottenStr, (*lastGottenStr)[ind - 1] == '.' ? "  " : " ");
                -- with an empty string the source reads one byte before the
                allocation; the separator it would then append is unknowable, so
                a single space is used. */
                if ind >= 1 && cur[ind - 1] == b'.' {
                    cur.extend_from_slice(b"  ");
                } else {
                    cur.extend_from_slice(b" ");
                }

                /* Replace leading ^ with a newline */
                if !text_str.is_empty() && text_str[0] == b'^' {
                    text_str[0] = b'\n';
                }

                /* If inside quotes, look for quote at end and say it is the end of accepting
                continuation lines, and as a protection, also say a blank line ends it */
                if in_quote_index >= 0
                    && (len == 0 || text_str[len - 1] == S_QUOTE_TYPES[in_quote_index as usize])
                {
                    if len != 0 {
                        text_str.truncate(len - 1);
                        cur.extend_from_slice(&text_str);
                    }
                    in_quote_index = -1;
                    last_gotten_str = None;
                } else {
                    cur.extend_from_slice(&text_str);
                }
            }
            /* Otherwise look for each keyword of interest, but null out the pointer
            to last gotten one so that it will only be valid on the next line */
            else {
                last_gotten_str = None;
                in_quote_index = -1;
                let err = check_keyword(
                    &text_str,
                    b"short",
                    &mut short_name,
                    &mut got_short,
                    &mut last_gotten_str,
                    PipKeywordSlot::ShortName,
                    None,
                );
                if err != 0 {
                    return err;
                }
                let err = check_keyword(
                    &text_str,
                    b"long",
                    &mut long_name,
                    &mut got_long,
                    &mut last_gotten_str,
                    PipKeywordSlot::LongName,
                    None,
                );
                if err != 0 {
                    return err;
                }
                let err = check_keyword(
                    &text_str,
                    b"type",
                    &mut type_0,
                    &mut got_type,
                    &mut last_gotten_str,
                    PipKeywordSlot::Type,
                    None,
                );
                if err != 0 {
                    return err;
                }
                let err = check_keyword(
                    &text_str,
                    b"format",
                    &mut format_str,
                    &mut got_format,
                    &mut last_gotten_str,
                    PipKeywordSlot::FormatStr,
                    None,
                );
                if err != 0 {
                    return err;
                }
                let err = check_keyword(
                    &text_str,
                    b"default",
                    &mut default_str,
                    &mut got_default,
                    &mut last_gotten_str,
                    PipKeywordSlot::DefaultStr,
                    None,
                );
                if err != 0 {
                    return err;
                }

                /* Check for usage if at help level 1 or if we haven't got either of
                the other strings yet */
                if help_level <= 1 || !(got_tip != 0 || got_man != 0) {
                    let err = check_keyword(
                        &text_str,
                        b"usage",
                        &mut usage_str,
                        &mut got_usage,
                        &mut last_gotten_str,
                        PipKeywordSlot::UsageStr,
                        Some(&mut in_quote_index),
                    );
                    if err != 0 {
                        return err;
                    }
                }

                /* Check for tooltip if at level 2 or if at level 1 and haven't got
                usage, or at level 3 and haven't got manpage */
                if help_level == 2
                    || (help_level <= 1 && got_usage == 0)
                    || (help_level >= 3 && got_man == 0)
                {
                    let err = check_keyword(
                        &text_str,
                        b"tooltip",
                        &mut tip_str,
                        &mut got_tip,
                        &mut last_gotten_str,
                        PipKeywordSlot::TipStr,
                        Some(&mut in_quote_index),
                    );
                    if err != 0 {
                        return err;
                    }
                }

                /* Check for manpage if at level 3 or if at level 2 and haven't got
                tip, or at level 1 and haven't got tip or usage */
                if help_level >= 3
                    || (help_level == 2 && got_tip == 0)
                    || (help_level <= 1 && !(got_tip != 0 || got_usage != 0))
                {
                    let err = check_keyword(
                        &text_str,
                        b"manpage",
                        &mut man_str,
                        &mut got_man,
                        &mut last_gotten_str,
                        PipKeywordSlot::ManStr,
                        Some(&mut in_quote_index),
                    );
                    if err != 0 {
                        return err;
                    }
                }

                /* If that was a line with quoted string, check for quote at end of line and
                close out the help string if so */
                if in_quote_index >= 0 && last_gotten_str.is_some() {
                    let target: Option<&mut Option<Vec<u8>>> = match last_gotten_str {
                        Some(PipKeywordSlot::ShortName) => Some(&mut short_name),
                        Some(PipKeywordSlot::LongName) => Some(&mut long_name),
                        Some(PipKeywordSlot::Type) => Some(&mut type_0),
                        Some(PipKeywordSlot::FormatStr) => Some(&mut format_str),
                        Some(PipKeywordSlot::DefaultStr) => Some(&mut default_str),
                        Some(PipKeywordSlot::UsageStr) => Some(&mut usage_str),
                        Some(PipKeywordSlot::TipStr) => Some(&mut tip_str),
                        Some(PipKeywordSlot::ManStr) => Some(&mut man_str),
                        _ => None,
                    };
                    if let Some(target) = target {
                        let cur = target.get_or_insert_with(Vec::new);
                        let len = cur.len();
                        if len != 0 && cur[len - 1] == S_QUOTE_TYPES[in_quote_index as usize] {
                            cur.truncate(len - 1);
                            in_quote_index = -1;
                            last_gotten_str = None;
                        }
                    }
                }
            }
        }
        /* But if not reading options, check for a new option token and start
        reading if one is found.  But first take a Field value as default
        long option name */
        else if is_option > 0 {
            last_gotten_str = None;
            reading_opt = 1;
            is_section = is_option - 1;
            if is_section == 0 {
                let err = check_keyword(
                    &text_str[OPEN_DELIM.len()..],
                    b"Field",
                    &mut long_name,
                    &mut got_long,
                    &mut last_gotten_str,
                    PipKeywordSlot::LongName,
                    None,
                );
                if err != 0 {
                    return err;
                }
                if got_long != 0 {
                    /* longName[strlen(longName) - 1] = sNullChar; */
                    if let Some(l) = long_name.as_mut() {
                        let n = l.len();
                        if n > 0 {
                            l.truncate(n - 1);
                        }
                    }
                }
            }
        }
    }
    0
}

/// Original C `PipParseEntries` (`parse_params.c:1540`).
///
/// Routine to parse the entries in command line after options have been
/// defined one way or another.
pub fn pip_parse_entries(
    argc: i32,
    argv: &[Vec<u8>],
    num_opt_args: &mut i32,
    num_non_opt_args: &mut i32,
) -> i32 {
    /* Special case: no arguments and flag set to take stdin automatically */
    if argc < S_TAKE_STD_IN.get() {
        let mut stdin_file = ImodFile::Stdin;
        let err = read_param_file(&mut stdin_file);
        if err != 0 {
            return err;
        }
    } else {
        /* parse the arguments */
        for i in 1..argc {
            let err = pip_next_arg(&argv[i as usize]);
            if err < 0 {
                return err;
            }
            if err != 0 && i == argc - 1 {
                pip_set_error(
                    b"A value was expected but not found for the last option on the command line",
                );
                return -1;
            }
        }
    }
    pip_number_of_args(num_opt_args, num_non_opt_args);
    pip_print_entries();
    0
}

/// Original C `PipReadOrParseOptions` (`parse_params.c:1570`).
///
/// High-level routine to initialize from autodoc with optional fallback
/// options.  Set exit string and output to stdout, print usage if not enough
/// arguments.
#[allow(clippy::too_many_arguments)]
pub fn pip_read_or_parse_options(
    argc: i32,
    argv: &[Vec<u8>],
    options: &[&[u8]],
    num_opts: i32,
    prog_name: &[u8],
    min_args: i32,
    num_in_files: i32,
    num_out_files: i32,
    num_opt_args: &mut i32,
    num_non_opt_args: &mut i32,
    header_func: Option<fn(&[u8])>,
) {
    /* sprintf(prefix, "ERROR: %s -", progName); */
    let mut prefix: Vec<u8> = Vec::with_capacity(prog_name.len() + 12);
    prefix.extend_from_slice(b"ERROR: ");
    prefix.extend_from_slice(prog_name);
    prefix.extend_from_slice(b" -");

    /* Startup with fallback */
    let ierr = pip_read_option_file(prog_name, 0, 0);
    pip_exit_on_error(0, &prefix);
    if ierr == 0 {
        pip_parse_entries(argc, argv, num_opt_args, num_non_opt_args);
    } else {
        let mut err_string: Vec<u8> = Vec::new();
        pip_get_error(&mut err_string);
        if options.is_empty() || num_opts == 0 {
            pip_set_error(&err_string);
        }
        /* printf("PIP WARNING: %s\nUsing fallback options in main program\n", errString); */
        let mut out = ImodFile::Stdout;
        let _ = out.write_all(b"PIP WARNING: ");
        let _ = out.write_all(&err_string);
        let _ = out.write_all(b"\nUsing fallback options in main program\n");
        pip_parse_input(
            argc,
            argv,
            options,
            num_opts,
            num_opt_args,
            num_non_opt_args,
        );
        pip_read_prog_defaults(prog_name);
    }

    /* Output usage and exit if not enough arguments or help entered */
    let mut ierr = 0i32;
    if *num_opt_args + *num_non_opt_args < min_args
        || (pip_get_boolean(b"help", &mut ierr) == 0 && ierr != 0)
    {
        if let Some(header_func) = header_func {
            header_func(prog_name);
        }
        pip_print_help(prog_name, 0, num_in_files, num_out_files);
        let _ = ImodFile::Stdout.flush();
        std::process::exit(0);
    }
}

/// Original C `PipReadProgDefaults` (`parse_params.c:1616`).
///
/// If there was a failure to read autodoc, call this to check for defaults in
/// the master file.
pub fn pip_read_prog_defaults(prog_name: &[u8]) {
    let pip_dir = match std::env::var_os("IMOD_DIR") {
        Some(d) => {
            <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::as_bytes(d.as_os_str()).to_vec()
        }
        None => return,
    };
    if pip_dir.len() as i32 > TEMP_STR_SIZE - 100 {
        return;
    }

    /* Save and clear out the first character of exit prefix to prevent exit */
    let save_prefix = S_EXIT_PREFIX.with_borrow(|p| p[0]);
    S_EXIT_PREFIX.with_borrow_mut(|p| p[0] = 0x00);
    /* sprintf(sTempStr, "%s%c%s%c%s", pipDir, PATH_SEPARATOR, DEFAULTS_DIR, PATH_SEPARATOR,
    DEFAULTS_FILE); */
    let temp = S_TEMP_STR.with_borrow_mut(|t| {
        t.clear();
        t.extend_from_slice(&pip_dir);
        t.push(PATH_SEPARATOR);
        t.extend_from_slice(DEFAULTS_DIR);
        t.push(PATH_SEPARATOR);
        t.extend_from_slice(DEFAULTS_FILE);
        t.clone()
    });

    /* The autodoc unit is still C-shaped: this is the one foreign boundary
    left in this file, and it goes away when `libcfshr::autodoc` is converted. */
    let table_size = S_TABLE_SIZE.get();
    unsafe {
        let path = std::ffi::CString::new(temp).unwrap_or_default();
        let adoc_ind = crate::imod::libcfshr::autodoc::adoc_read(path.as_ptr());
        if adoc_ind >= 0 {
            /* Look up the program and then check for each option in its section */
            let type_name = std::ffi::CString::new("Program").unwrap();
            let name = std::ffi::CString::new(prog_name).unwrap_or_default();
            let sect_ind = crate::imod::libcfshr::autodoc::adoc_lookup_section(
                type_name.as_ptr(),
                name.as_ptr(),
            );
            if sect_ind >= 0 {
                for i in 0..table_size {
                    let long_name = S_OPT_TABLE.with_borrow(|t| t[i as usize].long_name.clone());
                    let key =
                        std::ffi::CString::new(long_name.unwrap_or_default()).unwrap_or_default();
                    let mut value: *mut std::ffi::c_char = std::ptr::null_mut();
                    if crate::imod::libcfshr::autodoc::adoc_get_string(
                        type_name.as_ptr(),
                        sect_ind,
                        key.as_ptr(),
                        &raw mut value,
                    ) == 0
                        && !value.is_null()
                    {
                        let bytes = std::ffi::CStr::from_ptr(value).to_bytes().to_vec();
                        S_OPT_TABLE.with_borrow_mut(|t| t[i as usize].default_val = Some(bytes));
                        libc::free(value.cast());
                    }
                }
            }
            crate::imod::libcfshr::autodoc::adoc_clear(adoc_ind);
        }
    }
    S_EXIT_PREFIX.with_borrow_mut(|p| p[0] = save_prefix);
}

/// Original C `PipGetInOutFile` (`parse_params.c:1650`).
///
/// Routine to get input/output file from parameter or non-option args.
pub fn pip_get_in_out_file(option: &[u8], non_opt_arg_no: i32, filename: &mut Vec<u8>) -> i32 {
    if pip_get_string(option, filename) != 0 {
        let count = S_OPT_TABLE.with_borrow(|t| t[S_NON_OPT_IND.get() as usize].count);
        if non_opt_arg_no >= count {
            return 1;
        }
        pip_get_non_option_arg(non_opt_arg_no, filename);
    }
    0
}

/// Original C `ReadParamFile` (`parse_params.c:1665`).
///
/// Read successive lines from a parameter file or standard input, and store as
/// options and values.
fn read_param_file(p_file: &mut ImodFile) -> i32 {
    loop {
        /* If non-option lines are allowed, set flag that it is OK for LookupOption
        to not find the option, but only for the given number of lines at the
        start of the input */
        let non_opt_count = S_OPT_TABLE.with_borrow(|t| t[S_NON_OPT_IND.get() as usize].count);
        S_NOT_FOUND_OK.set(
            if S_NUM_OPTION_ARGUMENTS.get() == 0 && non_opt_count < S_NON_OPT_LINES.get() {
                1
            } else {
                0
            },
        );
        let mut indst = 0i32;
        let mut line: Vec<u8> = S_LINE_STR.with_borrow(|l| l.clone());
        let line_len = pip_read_next_line(p_file, &mut line, LINE_STR_SIZE, b'#', 0, 1, &mut indst);
        S_LINE_STR.with_borrow_mut(|l| *l = line.clone());
        if line_len == -3 {
            break;
        }
        if line_len == -2 {
            pip_set_error(b"Error reading parameter file or StandardInput");
            return -1;
        }
        if line_len == -1 {
            pip_set_error(
                b"Line too long for buffer while reading parameter file or StandardInput",
            );
            return -1;
        }

        /* Find token and make a copy */
        let sep = line[indst as usize..]
            .iter()
            .position(|&c| c == b'=' || c == b' ' || c == b'\t')
            .map(|p| p + indst as usize);
        let mut indnd = match sep {
            Some(p) => p as i32 - 1,
            None => line_len - 1,
        };
        if indnd >= line_len {
            indnd = line_len - 1;
        }

        let token = pip_sub_str_dup(&line, indst, indnd);

        /* Done if it matches end of input string */
        if token == STANDARD_INPUT_END
            || (S_DONE_ENDS.get() != 0 && token.len() == 4 && pip_starts_with(b"DONE", &token) != 0)
        {
            break;
        }

        /* Look up option and free the token string */
        let opt_num = lookup_option(&token, S_NUM_OPTIONS.get());
        if opt_num < 0 {
            /* If no option, process special case if in-line non-options allowed,
            or error out */
            if S_NOT_FOUND_OK.get() != 0 {
                let token = pip_sub_str_dup(&line, indst, line_len - 1);
                let err = add_value_string(S_NON_OPT_IND.get(), &token);
                if err != 0 {
                    return err;
                }
                continue;
            } else {
                return opt_num;
            }
        }

        if S_OPT_TABLE
            .with_borrow(|t| t[opt_num as usize].type_0.as_deref() == Some(PARAM_FILE_STRING))
        {
            pip_set_error(
                b"Trying to open a parameter file while reading a parameter file or StandardInput",
            );
            return -1;
        }

        /* Find first non-white space, passing over at most one equals sign */
        let mut indst2 = indnd + 1;
        let mut got_equals = 0i32;
        while indst2 < line_len {
            if line[indst2 as usize] == b'=' {
                if got_equals != 0 {
                    S_TEMP_STR.with_borrow_mut(|t| {
                        t.clear();
                        t.extend_from_slice(b"Two = signs in input line:  ");
                    });
                    append_to_error_string(&line);
                    return -1;
                }
                got_equals = 1;
            } else if line[indst2 as usize] != b' ' && line[indst2 as usize] != b'\t' {
                break;
            }
            indst2 += 1;
        }

        /* If there is a string, get one; if not, get a "1" for boolean, otherwise
        it is an error */
        let token: Vec<u8> = if indst2 < line_len {
            pip_sub_str_dup(&line, indst2, line_len - 1)
        } else if S_OPT_TABLE
            .with_borrow(|t| t[opt_num as usize].type_0.as_deref() == Some(BOOLEAN_STRING))
        {
            b"1".to_vec()
        } else {
            S_TEMP_STR.with_borrow_mut(|t| {
                t.clear();
                t.extend_from_slice(b"Missing a value on the input line:  ");
            });
            append_to_error_string(&line);
            return -1;
        };

        /* Add the token as a value string and increment argument number */
        let err = add_value_string(opt_num, &token);
        if err != 0 {
            return err;
        }
        S_NUM_OPTION_ARGUMENTS.set(S_NUM_OPTION_ARGUMENTS.get() + 1);
    }
    S_NOT_FOUND_OK.set(0);
    0
}

/// Original C `PipReadStdinIfSet` (`parse_params.c:1758`).
///
/// Call from fortran to read stdin if the flag is set to take from stdin.
pub fn pip_read_stdin_if_set() -> i32 {
    if S_TAKE_STD_IN.get() != 0 {
        let mut stdin_file = ImodFile::Stdin;
        return read_param_file(&mut stdin_file);
    }
    0
}

/// Original C `PipReadNextLine` (`parse_params.c:1776`).
///
/// Reads a line from the file `pFile`, stripping white space at the end of the
/// line and in-line comments starting with `comment` if `inLineComments` is
/// non-zero.  Discards the line and reads another if it is blank or if the
/// first non-blank character is `comment`, unless `keepComments` is nonzero.
/// Returns the line in `line_str` and the index of the first non-white space
/// character in `first_non_white`.  The size of the source's buffer is provided
/// in `str_size`.  Returns the length of the line, or -3 for end of file, -1 if
/// the line is too long, or -2 for error reading file.
pub fn pip_read_next_line(
    p_file: &mut ImodFile,
    line_str: &mut Vec<u8>,
    str_size: i32,
    comment: u8,
    keep_comments: i32,
    in_line_comments: i32,
    first_non_white: &mut i32,
) -> i32 {
    use std::io::Read;
    let mut indst;
    let mut line_len: i32;

    loop {
        /* fgets(sLineStr, strSize, pFile): at most strSize - 1 bytes, stopping
        after a newline.  A seekable file is read in one block and repositioned
        to just past the newline, which is what stdio's own buffering does; a
        stream that cannot be repositioned is read a byte at a time, and Rust's
        standard input is itself buffered so that costs no extra system call. */
        let cap = (str_size - 1).max(0) as usize;
        let mut buf: Vec<u8> = vec![0; cap];
        let mut n = 0usize;
        let seekable = matches!(p_file, ImodFile::File(_));
        if seekable {
            while n < cap {
                let want = (cap - n).min(512);
                match p_file.read(&mut buf[n..n + want]) {
                    Ok(0) => break,
                    Ok(k) => {
                        n += k;
                        if buf[..n].contains(&b'\n') {
                            break;
                        }
                    }
                    Err(_) => return -2,
                }
            }
            let stop = match buf[..n].iter().position(|&c| c == b'\n') {
                Some(p) => p + 1,
                None => n,
            };
            if stop < n {
                if std::io::Seek::seek(p_file, std::io::SeekFrom::Current(-((n - stop) as i64)))
                    .is_err()
                {
                    return -2;
                }
            }
            n = stop;
        } else {
            while n < cap {
                let mut one = [0u8; 1];
                match p_file.read(&mut one) {
                    Ok(0) => break,
                    Ok(_) => {
                        buf[n] = one[0];
                        n += 1;
                        if one[0] == b'\n' {
                            break;
                        }
                    }
                    Err(_) => return -2,
                }
            }
        }

        /* If error, it's OK if it's an EOF, or an error otherwise */
        if n == 0 {
            return -3;
        }

        /* check for line too long */
        /* lineLen = strlen(sLineStr): an embedded NUL ends the string. */
        line_len = match buf[..n].iter().position(|&c| c == 0) {
            Some(p) => p as i32,
            None => n as i32,
        };
        line_str.clear();
        line_str.extend_from_slice(&buf[..line_len as usize]);
        if line_len == str_size - 1 {
            return -1;
        }

        /* Get first non-white space */
        indst = 0i32;
        while indst < line_len {
            if line_str[indst as usize] != b' ' && line_str[indst as usize] != b'\t' {
                break;
            }
            indst += 1;
        }

        /* If it is a comment, skip or strip line ending and return */
        let ch_at_indst = if indst < line_len {
            line_str[indst as usize]
        } else {
            0
        };
        if ch_at_indst == comment {
            if keep_comments != 0 {
                while line_len > 0
                    && (line_str[(line_len - 1) as usize] == b'\n'
                        || line_str[(line_len - 1) as usize] == b'\r')
                {
                    line_len -= 1;
                }
                line_str.truncate(line_len as usize);
                break;
            } else {
                continue;
            }
        }

        /* adjust line length to remove comment, if we have in-line comments */
        let com_pos = line_str.iter().position(|&c| c == comment);
        if let Some(p) = com_pos {
            if in_line_comments != 0 {
                line_len = p as i32;
            }
        }

        /* adjust line length back further to remove white space and newline */
        while line_len > 0 {
            let ch = line_str[(line_len - 1) as usize];
            if ch != b' ' && ch != b'\t' && ch != b'\n' && ch != b'\r' {
                break;
            }
            line_len -= 1;
        }
        line_str.truncate(line_len as usize);

        /* Return if something is on line or we are keeping comments */
        if indst < line_len || keep_comments != 0 {
            break;
        }
    }
    *first_non_white = indst;
    line_len
}

/// Original C `OptionLineOfValues` (`parse_params.c:1863`).
///
/// Parse a line of values for an option and return them into an array.
fn option_line_of_values(
    option: &[u8],
    array: PipValueArray<'_>,
    val_type: i32,
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    /* Get string and save pointer to it for error messages */
    let mut str_ptr: Vec<u8> = Vec::new();
    let val_err = get_next_value_string(option, &mut str_ptr);
    let mut err = 0i32;
    if val_err == 0 || val_err == 2 {
        err = pip_get_line_of_values(option, &str_ptr, array, val_type, num_to_get, array_size);
    }
    if err != 0 {
        return err;
    }
    val_err
}

/// Original C `PipGetLineOfValues` (`parse_params.c:1890`).
///
/// Parses a line of values from the string in `str_ptr` and returns them into
/// `array`, whose size is given by `array_size`.  The number of values to get is
/// set in `num_to_get`, where a value of zero indicates all values should be
/// returned, and a value of -1 indicates that up to `array_size` values should
/// be returned; in either case the number gotten is returned in `num_to_get`.
pub fn pip_get_line_of_values(
    option: &[u8],
    str_ptr: &[u8],
    mut array: PipValueArray<'_>,
    val_type: i32,
    num_to_get: &mut i32,
    array_size: i32,
) -> i32 {
    let mut num_got = 0i32;
    let mut got_comma = 1i32;
    /* char sepStr[] = ",\t /"; */
    let sep_str: &[u8] = b",\t /";

    let full_str: &[u8] = str_ptr;
    let mut pos = 0usize;
    while pos < full_str.len() {
        let sep_ptr = full_str[pos..]
            .iter()
            .position(|c| sep_str.contains(c))
            .map(|p| p + pos);
        let end_ptr: usize;
        match sep_ptr {
            None => {
                /* null pointer means read a number to end of string */
                end_ptr = full_str.len();
            }
            Some(p) if p == pos => {
                /* separator at start means advance by one byte and continue */
                /* if defaults allowed and a specific number are expected,
                / means stop processing and mark all values as received */
                if full_str[pos] == b'/' {
                    if S_ALLOW_DEFAULTS.get() != 0 && *num_to_get > 0 {
                        num_got = *num_to_get;
                        break;
                    }
                    line_of_values_error(
                        full_str,
                        "Default entry with a / is not allowed in value entry:  %s  ",
                        &[CArg::Bytes(option)],
                    );
                    return -1;
                }

                /* special handling of commas to allow default values */
                if full_str[pos] == b',' {
                    /* If already have a comma, skip an array value if defaults allowed */
                    if got_comma != 0 {
                        if S_ALLOW_DEFAULTS.get() != 0 && *num_to_get > 0 {
                            num_got += 1;
                            if num_got >= *num_to_get {
                                break;
                            }
                        } else {
                            line_of_values_error(
                                full_str,
                                "Default entries with commas are not allowed in value entry:  %s  ",
                                &[CArg::Bytes(option)],
                            );
                            return -1;
                        }
                    }
                    got_comma = 1;
                }
                pos += 1;
                continue;
            }
            Some(p) => {
                /* otherwise, this should be the end pointer in the strto[ld] call */
                end_ptr = p;
            }
        }

        /* If we are already full, then it is an error */
        if num_got >= array_size {
            line_of_values_error(
                full_str,
                "Too many values for input array in value entry:  %s  ",
                &[CArg::Bytes(option)],
            );
            return -1;
        }

        /* convert number, get pointer to first invalid char */
        let mut scanned = 0usize;
        if val_type == PIP_INTEGER {
            let v = strtol(&full_str[pos..], &mut scanned, 10);
            if let PipValueArray::Int(a) = &mut array {
                a[num_got as usize] = v as i32;
            }
            num_got += 1;
        } else if val_type == PIP_FLOAT {
            let v = strtod(&full_str[pos..], &mut scanned);
            if let PipValueArray::Float(a) = &mut array {
                a[num_got as usize] = v as f32;
            }
            num_got += 1;
        } else {
            let v = strtod(&full_str[pos..], &mut scanned);
            if let PipValueArray::Double(a) = &mut array {
                a[num_got as usize] = v;
            }
            num_got += 1;
        }
        let invalid = pos + scanned;

        /* If invalid character is before end character, it is an error */
        if invalid != end_ptr {
            line_of_values_error(
                full_str,
                "Illegal character in value entry:  %s  ",
                &[CArg::Bytes(option)],
            );
            return -1;
        }

        /* Mark that there is no comma after we have a number */
        got_comma = 0;

        /* Done if at end of line, or if count is fulfilled, or if buffer is full and
        numToGet < 0 */
        let end_char = if end_ptr < full_str.len() {
            full_str[end_ptr]
        } else {
            0
        };
        if end_char == 0
            || (*num_to_get > 0 && num_got >= *num_to_get)
            || (*num_to_get < 0 && num_got >= array_size)
        {
            break;
        }

        /* Otherwise advance to separator and continue */
        pos = end_ptr;
    }

    /* return number actually gotten if it was left open */
    if *num_to_get <= 0 {
        *num_to_get = num_got;
    }

    /* If not enough values found, return error */
    if num_got < *num_to_get {
        line_of_values_error(
            full_str,
            "%d values expected but only %d values found in value entry:  %s  ",
            &[
                CArg::Int(*num_to_get as i64),
                CArg::Int(num_got as i64),
                CArg::Bytes(option),
            ],
        );
        return -1;
    }

    0
}

/// Original C `LineOfValuesError` (`parse_params.c:1994`).
///
/// `PipGetLineOfValues` can be called without initializing PIP, so this
/// function is needed to allocate `sTempStr` if needed.  It handles appending
/// the input string to the error.  The source's `va_list` becomes the slice of
/// [`CArg`] that [`c_format`] takes.
fn line_of_values_error(full_str: &[u8], format: &str, args: &[CArg]) {
    /* vsprintf(sTempStr, format, args); */
    let formatted = c_format(format, args);
    S_TEMP_STR.with_borrow_mut(|t| {
        t.clear();
        t.extend_from_slice(formatted.as_bytes());
    });
    append_to_error_string(full_str);
}

/// Original C `GetNextValueString` (`parse_params.c:2015`).
///
/// Get the value string for the given option.  Return < 0 if the option is
/// invalid, 1 if the option was not entered, 2 if it was not entered and a
/// default string is being returned.  If the option allows multiple values,
/// advance the multiple counter.
pub fn get_next_value_string(option: &[u8], str_ptr: &mut Vec<u8>) -> i32 {
    let mut index = 0i32;

    let err = lookup_option(option, S_NON_OPT_IND.get() + 1);
    if err < 0 {
        return err;
    }
    let (count, multiple, default_val) = S_OPT_TABLE.with_borrow(|t| {
        let o = &t[err as usize];
        (o.count, o.multiple, o.default_val.clone())
    });
    if count == 0 {
        match default_val {
            None => return 1,
            Some(d) => {
                *str_ptr = d;
                return 2;
            }
        }
    }

    if multiple != 0 {
        index = multiple - 1;
        if multiple < count {
            S_OPT_TABLE.with_borrow_mut(|t| t[err as usize].multiple = multiple + 1);
        }
    }
    *str_ptr = S_OPT_TABLE.with_borrow(|t| t[err as usize].value_ptr[index as usize].clone());
    0
}

/// Original C `AddValueString` (`parse_params.c:2043`).
///
/// Add a string to the set of values for an option.
pub fn add_value_string(option: i32, str_ptr: &[u8]) -> i32 {
    /* If the count is zero or we accept multiple values, need to allocate
    array for address of string, and array for next non option index */
    let (count, multiple, linked) = S_OPT_TABLE.with_borrow(|t| {
        let o = &t[option as usize];
        (o.count, o.multiple, o.linked)
    });
    if count == 0 || multiple != 0 {
        if linked != 0 {
            S_OPT_TABLE.with_borrow_mut(|t| {
                t[option as usize]
                    .next_linked
                    .resize(((count + 1) * 2) as usize, 0)
            });
        }
    } else {
        /* otherwise, need to free existing value */
        S_OPT_TABLE.with_borrow_mut(|t| {
            t[option as usize].value_ptr.clear();
            t[option as usize].count = 0;
        });
    }
    /* save address that was passed in.  The caller had to make a duplicate */

    if linked != 0 {
        let count = S_OPT_TABLE.with_borrow(|t| t[option as usize].count);
        let non_opt_count = S_OPT_TABLE.with_borrow(|t| t[S_NON_OPT_IND.get() as usize].count);
        S_OPT_TABLE.with_borrow_mut(|t| {
            let o = &mut t[option as usize];
            o.next_linked[(2 * count) as usize] = non_opt_count;
            o.next_linked[(2 * count + 1) as usize] = 0;
        });
        let linked_option = S_LINKED_OPTION.with_borrow(|l| l.clone());
        if let Some(linked_option) = linked_option {
            let err = lookup_option(&linked_option, S_NUM_OPTIONS.get());
            if err < 0 {
                return err;
            }
            let linked_count = S_OPT_TABLE.with_borrow(|t| t[err as usize].count);
            S_OPT_TABLE.with_borrow_mut(|t| {
                t[option as usize].next_linked[(2 * count + 1) as usize] = linked_count
            });
        }
    }
    S_OPT_TABLE.with_borrow_mut(|t| {
        let o = &mut t[option as usize];
        o.value_ptr.push(str_ptr.to_vec());
        o.count += 1;
    });
    0
}

/// Original C `LookupOption` (`parse_params.c:2089`).
///
/// Look up an option in the table, issue an error message if the option does
/// not exist or is ambiguous; return index of option or an error code.  This is
/// the unique-prefix matcher: `PipStartsWith(sname, option)` accepts any
/// abbreviation that is not shared with another option.
pub fn lookup_option(option: &[u8], max_lookup: i32) -> i32 {
    let mut found = LOOKUP_NOT_FOUND;
    let lenopt = option.len() as i32;
    let no_abbrevs = S_NO_ABBREVS.get();

    /* Look at all of the options specified by maxLookup */
    for i in 0..max_lookup {
        let (sname_opt, lname_opt, len_short) = S_OPT_TABLE.with_borrow(|t| {
            let o = &t[i as usize];
            (o.short_name.clone(), o.long_name.clone(), o.len_short)
        });
        let sname: &[u8] = sname_opt.as_deref().unwrap_or(b"");
        let lname: &[u8] = lname_opt.as_deref().unwrap_or(b"");
        let starts = pip_starts_with(sname, option);

        /* First test for single letter short name match - if passes, skip ambiguity test */
        if lenopt == 1 && starts != 0 && len_short == 1 {
            found = i;
            break;
        }
        if (starts != 0 && (no_abbrevs == 0 || lenopt == len_short))
            || (pip_starts_with(lname, option) != 0
                && (no_abbrevs == 0 || lenopt == lname.len() as i32))
        {
            /* If it is found, it's an error if one has already been found */
            if found == LOOKUP_NOT_FOUND {
                found = i;
            } else {
                if S_TEST_ABBREV_FOR_USAGE.get() == 0 {
                    /* sprintf(sTempStr, "An option specified by \"%s\" is ambiguous between "
                    "option %s -  %s  and option %s -  %s", ...) */
                    let (found_short, found_long) = S_OPT_TABLE.with_borrow(|t| {
                        let o = &t[found as usize];
                        (o.short_name.clone(), o.long_name.clone())
                    });
                    let temp = S_TEMP_STR.with_borrow_mut(|t| {
                        t.clear();
                        t.extend_from_slice(b"An option specified by \"");
                        t.extend_from_slice(option);
                        t.extend_from_slice(b"\" is ambiguous between option ");
                        t.extend_from_slice(sname_opt.as_deref().unwrap_or(b"(null)"));
                        t.extend_from_slice(b" -  ");
                        t.extend_from_slice(lname_opt.as_deref().unwrap_or(b"(null)"));
                        t.extend_from_slice(b"  and option ");
                        t.extend_from_slice(found_short.as_deref().unwrap_or(b"(null)"));
                        t.extend_from_slice(b" -  ");
                        t.extend_from_slice(found_long.as_deref().unwrap_or(b"(null)"));
                        t.clone()
                    });
                    pip_set_error(&temp);
                }
                return LOOKUP_AMBIGUOUS;
            }
        }
    }

    /* Set error string unless flag set that non-options are OK */
    if found == LOOKUP_NOT_FOUND && S_NOT_FOUND_OK.get() == 0 {
        /* sprintf(sTempStr, "Illegal option: %s", option); */
        let temp = S_TEMP_STR.with_borrow_mut(|t| {
            t.clear();
            t.extend_from_slice(b"Illegal option: ");
            t.extend_from_slice(option);
            t.clone()
        });
        pip_set_error(&temp);
    }
    found
}

/// Original C `PipSubStrDup` (`parse_params.c:2128`).
///
/// Duplicate a substring into a new string.
fn pip_sub_str_dup(s1: &[u8], i1: i32, i2: i32) -> Vec<u8> {
    let mut s2: Vec<u8> = Vec::new();
    let mut i = i1;
    while i <= i2 {
        if i >= 0 && (i as usize) < s1.len() {
            s2.push(s1[i as usize]);
        } else {
            /* The source reads s1[i] regardless; past the terminator that is a
            NUL, which strdup-style copying would stop at. */
            break;
        }
        i += 1;
    }
    s2
}

/// Original C `PipMemoryError` (`parse_params.c:2145`).
///
/// Test for whether the pointer is valid and give memory error if not.  The
/// source's `void *ptr` is the result of an allocation; Rust allocation does
/// not return null, so callers inside this unit pass `true`.
pub fn pip_memory_error(ptr_non_null: bool, routine: &[u8]) -> i32 {
    if ptr_non_null {
        return 0;
    }
    /* sprintf(sTempStr, "Failed to get memory for string in %s", routine); */
    let temp = S_TEMP_STR.with_borrow_mut(|t| {
        t.clear();
        t.extend_from_slice(b"Failed to get memory for string in ");
        t.extend_from_slice(routine);
        t.clone()
    });
    pip_set_error(&temp);
    -1
}

/// Original C `AppendToErrorString` (`parse_params.c:2164`).
///
/// Add as much of a string as fits to the `sTempStr` and use to set error.
fn append_to_error_string(str_arg: &[u8]) {
    let temp = S_TEMP_STR.with_borrow_mut(|t| {
        let len = t.len();
        /* strncpy(&sTempStr[len], str, TEMP_STR_SIZE - len - 1) into a
        TEMP_STR_SIZE buffer whose last byte was just set to NUL. */
        let room = (TEMP_STR_SIZE as usize).saturating_sub(len + 1);
        let n = str_arg.len().min(room);
        t.extend_from_slice(&str_arg[..n]);
        t.clone()
    });
    pip_set_error(&temp);
}

/// Original C `PipStartsWith` (`parse_params.c:2176`).
///
/// Returns 1 if `full_str` starts with `sub_str`, where either string can be
/// empty (the source's NULL and empty-string cases both give 0).
pub fn pip_starts_with(full_str: &[u8], sub_str: &[u8]) -> i32 {
    if full_str.is_empty() || sub_str.is_empty() {
        return 0;
    }
    if S_NO_CASE.get() != 0 {
        let mut f = 0usize;
        let mut s = 0usize;
        while f < full_str.len() && s < sub_str.len() {
            if full_str[f].to_ascii_uppercase() != sub_str[s].to_ascii_uppercase() {
                return 0;
            }
            f += 1;
            s += 1;
        }
        if s >= sub_str.len() {
            return 1;
        }
    } else if full_str.starts_with(sub_str) {
        return 1;
    }
    0
}

/// Original C `LineIsOptionToken` (`parse_params.c:2197`).
///
/// Determines whether the line contains the token for an option inside the
/// opening and closing delimiters and returns 1 if it does, or -1 if it is
/// another token.
fn line_is_option_token(line: &[u8]) -> i32 {
    /* It is not a token unless it starts with open delim and contains close */
    if pip_starts_with(line, OPEN_DELIM) == 0
        || !line.windows(CLOSE_DELIM.len()).any(|w| w == CLOSE_DELIM)
    {
        return 0;
    }

    /* It must then contain "Field" right after delim to be an option */
    let token = &line[OPEN_DELIM.len()..];
    if pip_starts_with(token, b"Field") != 0 {
        return 1;
    }
    if pip_starts_with(token, b"SectionHeader") != 0 {
        return 2;
    }

    -1
}

/// Original C `CheckKeyword` (`parse_params.c:2222`).
///
/// Checks for whether a keyword occurs at the beginning of the line and is
/// followed by the keyword-value delimiter, and if so duplicates the value
/// string and sets the flag.  The source's `char ***lastCopied` — the address
/// of the variable holding the string — becomes `last_copied` plus the `slot`
/// naming which variable `copyto` is.
#[allow(clippy::too_many_arguments)]
fn check_keyword(
    line: &[u8],
    keyword: &[u8],
    copyto: &mut Option<Vec<u8>>,
    gotit: &mut i32,
    last_copied: &mut Option<PipKeywordSlot>,
    slot: PipKeywordSlot,
    quote_ind: Option<&mut i32>,
) -> i32 {
    /* First make sure line starts with it */
    if pip_starts_with(line, keyword) == 0 {
        return 0;
    }

    /* Now look for delimiter */
    let value_delim = S_VALUE_DELIM.with_borrow(|d| d.clone()).unwrap_or_default();
    let val_start = if value_delim.is_empty() {
        /* strstr(line, "") returns line */
        Some(0usize)
    } else {
        line.windows(value_delim.len())
            .position(|w| w == value_delim.as_slice())
    };
    let mut val_start = match val_start {
        Some(p) => p,
        None => return 0,
    };

    /* Free previous entry if there was one, and mark that it was not gotten,
    so that an empty entry can supercede a non-empty one */
    if *gotit != 0 && copyto.is_some() {
        *copyto = None;
        *gotit = 0;
    }

    /* Eat spaces after the delimiter and return if nothing left */
    /* In other words, a key with no value is the same as having no key at all */
    val_start += value_delim.len();
    while val_start < line.len() && (line[val_start] == b' ' || line[val_start] == b'\t') {
        val_start += 1;
    }
    if val_start >= line.len() {
        return 0;
    }

    /* Look for quote if directed to */
    if let Some(quote_ind) = quote_ind {
        match S_QUOTE_TYPES.iter().position(|&q| q == line[val_start]) {
            Some(p) => {
                *quote_ind = p as i32;
                val_start += 1;
            }
            None => *quote_ind = -1,
        }
    }

    /* Copy string and return address, set flag that it was gotten */
    let copy_str = line[val_start..].to_vec();

    *gotit = 1;
    *copyto = Some(copy_str);
    *last_copied = Some(slot);
    0
}

/// The C library's `strtol` with the source's base, as a Rust function.
///
/// This is a boundary translation, like [`c_format`]: `str::parse` rejects the
/// partial parses `PipGetLineOfValues` relies on, and the index where the scan
/// stopped is what the source compares against its own end pointer.  `*end` is
/// returned as an index into `s`, and is 0 when no conversion was performed, as
/// C leaves `endptr` at `nptr`.
fn strtol(s: &[u8], end: &mut usize, base: i32) -> i64 {
    let mut i = 0usize;
    while i < s.len()
        && (s[i] == b' '
            || s[i] == b'\t'
            || s[i] == b'\n'
            || s[i] == 0x0b
            || s[i] == 0x0c
            || s[i] == b'\r')
    {
        i += 1;
    }
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let mut base = base;
    if (base == 0 || base == 16)
        && i + 1 < s.len()
        && s[i] == b'0'
        && (s[i + 1] | 32) == b'x'
        && i + 2 < s.len()
        && (s[i + 2] as char).is_digit(16)
    {
        i += 2;
        base = 16;
    } else if base == 0 {
        base = if i < s.len() && s[i] == b'0' { 8 } else { 10 };
    }
    let digits_start = i;
    let mut value: i64 = 0;
    let mut overflow = false;
    while i < s.len() {
        let d = match (s[i] as char).to_digit(base as u32) {
            Some(d) => d as i64,
            None => break,
        };
        if !overflow {
            match value
                .checked_mul(base as i64)
                .and_then(|v| v.checked_add(d))
            {
                Some(v) => value = v,
                None => overflow = true,
            }
        }
        i += 1;
    }
    if i == digits_start {
        /* No conversion: endptr is left at the original nptr. */
        *end = 0;
        return 0;
    }
    *end = i;
    if overflow {
        return if negative { i64::MIN } else { i64::MAX };
    }
    if negative { -value } else { value }
}

/// The C library's `strtod`, as a Rust function.
///
/// Accepts what glibc accepts — leading white space, a sign, a decimal or C99
/// hexadecimal significand with an optional exponent, and `inf`/`infinity`/
/// `nan` — and reports in `*end` the index in `s` where the scan stopped, which
/// is 0 when no conversion was performed.
fn strtod(s: &[u8], end: &mut usize) -> f64 {
    let mut i = 0usize;
    while i < s.len()
        && (s[i] == b' '
            || s[i] == b'\t'
            || s[i] == b'\n'
            || s[i] == 0x0b
            || s[i] == 0x0c
            || s[i] == b'\r')
    {
        i += 1;
    }
    let sign_pos = i;
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let _ = sign_pos;

    /* infinity */
    let rest = &s[i..];
    let lower_starts = |p: &[u8]| -> bool {
        rest.len() >= p.len()
            && rest[..p.len()]
                .iter()
                .zip(p.iter())
                .all(|(a, b)| a.to_ascii_lowercase() == *b)
    };
    if lower_starts(b"infinity") {
        *end = i + 8;
        return if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if lower_starts(b"inf") {
        *end = i + 3;
        return if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if lower_starts(b"nan") {
        let mut j = i + 3;
        /* nan(n-char-sequence) */
        if j < s.len() && s[j] == b'(' {
            let mut k = j + 1;
            while k < s.len() && (s[k].is_ascii_alphanumeric() || s[k] == b'_') {
                k += 1;
            }
            if k < s.len() && s[k] == b')' {
                j = k + 1;
            }
        }
        *end = j;
        return if negative { -f64::NAN } else { f64::NAN };
    }

    /* C99 hexadecimal floating literal */
    if i + 1 < s.len() && s[i] == b'0' && (s[i + 1] | 32) == b'x' {
        let mut j = i + 2;
        let mut mantissa: f64 = 0.0;
        let mut any = false;
        while j < s.len() && (s[j] as char).is_digit(16) {
            mantissa = mantissa * 16.0 + (s[j] as char).to_digit(16).unwrap() as f64;
            j += 1;
            any = true;
        }
        let mut bin_exp: i32 = 0;
        if j < s.len() && s[j] == b'.' {
            j += 1;
            while j < s.len() && (s[j] as char).is_digit(16) {
                mantissa = mantissa * 16.0 + (s[j] as char).to_digit(16).unwrap() as f64;
                bin_exp -= 4;
                j += 1;
                any = true;
            }
        }
        if any {
            let mantissa_end = j;
            if j < s.len() && (s[j] | 32) == b'p' {
                let mut k = j + 1;
                let mut esign = 1i32;
                if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
                    if s[k] == b'-' {
                        esign = -1;
                    }
                    k += 1;
                }
                let estart = k;
                let mut ev: i32 = 0;
                while k < s.len() && s[k].is_ascii_digit() {
                    ev = ev.saturating_mul(10).saturating_add((s[k] - b'0') as i32);
                    k += 1;
                }
                if k > estart {
                    bin_exp = bin_exp.saturating_add(esign * ev);
                    j = k;
                } else {
                    j = mantissa_end;
                }
            }
            *end = j;
            let value = mantissa * (2.0f64).powi(bin_exp);
            return if negative { -value } else { value };
        }
    }

    /* Decimal */
    let num_start = i;
    let mut j = i;
    let digits_before = j;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
    }
    let mut any_digits = j > digits_before;
    if j < s.len() && s[j] == b'.' {
        j += 1;
        let digits_after = j;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
        any_digits = any_digits || j > digits_after;
    }
    if !any_digits {
        /* No conversion. */
        *end = 0;
        return 0.0;
    }
    let mantissa_end = j;
    if j < s.len() && (s[j] | 32) == b'e' {
        let mut k = j + 1;
        if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
            k += 1;
        }
        let estart = k;
        while k < s.len() && s[k].is_ascii_digit() {
            k += 1;
        }
        if k > estart {
            j = k;
        } else {
            j = mantissa_end;
        }
    }
    *end = j;
    let text = std::str::from_utf8(&s[num_start..j]).unwrap_or("0");
    let value: f64 = text.parse().unwrap_or(0.0);
    if negative { -value } else { value }
}
