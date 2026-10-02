//! Translation of `IMOD/raptor/opencv/cxerror.cpp` and `cxerror.h` (the
//! parts RAPTOR reaches): the error context, `cvSetErrMode`, `cvError`,
//! `cvStdErrReport` and `cvErrorStr`.
//!
//! The library is built as the static single-thread case, so `icvGetContext`
//! is one process-wide context created on first use with
//! `CV_ErrModeLeaf`; here it is a `thread_local`, so a program run in
//! process starts with a fresh one.  Only the error mode is kept: RAPTOR
//! never installs a callback (`cvRedirectError` is not reached) and never
//! reads the stored status, message or location (`cvGetErrStatus` is read
//! only on paths that exit first in leaf mode).
//!
//! RAPTOR runs in `CV_ErrModeLeaf` (the default, and what
//! `SFMestimationWithBA` sets), where `cvError` reports through
//! `cvStdErrReport` and terminates.  The C then does `assert(0)` "for
//! post-mortem analysis with GDB" before `exit(-abs(terminate))`; the
//! reference build keeps assertions, so native aborts with `SIGABRT` there.
//! The translation takes the `exit(-1)` the source writes next (status 255)
//! instead of crashing (BUGS.md, "RAPTOR OpenCV leaf-mode error").  The
//! other modes let the caller continue with an error status, which the
//! translated callers cannot express; RAPTOR never selects them.

use std::cell::Cell;

/// `CV_StsOk`.
pub const CV_STS_OK: i32 = 0;
/// `CV_StsBackTrace`.
pub const CV_STS_BACK_TRACE: i32 = -1;
/// `CV_StsError`.
pub const CV_STS_ERROR: i32 = -2;
/// `CV_StsInternal`.
pub const CV_STS_INTERNAL: i32 = -3;
/// `CV_StsNoMem`.
pub const CV_STS_NO_MEM: i32 = -4;
/// `CV_StsBadArg`.
pub const CV_STS_BAD_ARG: i32 = -5;
/// `CV_StsNoConv`.
pub const CV_STS_NO_CONV: i32 = -7;
/// `CV_StsAutoTrace`.
pub const CV_STS_AUTO_TRACE: i32 = -8;
/// `CV_BadStep`.
pub const CV_BAD_STEP: i32 = -13;
/// `CV_BadNumChannels`.
pub const CV_BAD_NUM_CHANNELS: i32 = -15;
/// `CV_BadDepth`.
pub const CV_BAD_DEPTH: i32 = -17;
/// `CV_BadCOI`.
pub const CV_BAD_COI: i32 = -24;
/// `CV_StsNullPtr`.
pub const CV_STS_NULL_PTR: i32 = -27;
/// `CV_StsBadSize`.
pub const CV_STS_BAD_SIZE: i32 = -201;
/// `CV_StsDivByZero`.
pub const CV_STS_DIV_BY_ZERO: i32 = -202;
/// `CV_StsInplaceNotSupported`.
pub const CV_STS_INPLACE_NOT_SUPPORTED: i32 = -203;
/// `CV_StsObjectNotFound`.
pub const CV_STS_OBJECT_NOT_FOUND: i32 = -204;
/// `CV_StsUnmatchedFormats`.
pub const CV_STS_UNMATCHED_FORMATS: i32 = -205;
/// `CV_StsBadFlag`.
pub const CV_STS_BAD_FLAG: i32 = -206;
/// `CV_StsBadPoint`.
pub const CV_STS_BAD_POINT: i32 = -207;
/// `CV_StsBadMask`.
pub const CV_STS_BAD_MASK: i32 = -208;
/// `CV_StsUnmatchedSizes`.
pub const CV_STS_UNMATCHED_SIZES: i32 = -209;
/// `CV_StsUnsupportedFormat`.
pub const CV_STS_UNSUPPORTED_FORMAT: i32 = -210;
/// `CV_StsOutOfRange`.
pub const CV_STS_OUT_OF_RANGE: i32 = -211;
/// `CV_StsParseError`.
pub const CV_STS_PARSE_ERROR: i32 = -212;
/// `CV_StsNotImplemented`.
pub const CV_STS_NOT_IMPLEMENTED: i32 = -213;
/// `CV_StsBadMemBlock`.
pub const CV_STS_BAD_MEM_BLOCK: i32 = -214;

/// `CV_ErrModeLeaf`: print error and exit program.
pub const CV_ERR_MODE_LEAF: i32 = 0;
/// `CV_ErrModeParent`: print error and continue.
pub const CV_ERR_MODE_PARENT: i32 = 1;
/// `CV_ErrModeSilent`: don't print and continue.
pub const CV_ERR_MODE_SILENT: i32 = 2;

thread_local! {
    /// `icvGetContext()->err_mode` (`icvCreateContext` starts it at
    /// `CV_ErrModeLeaf`).
    static ERR_MODE: Cell<i32> = const { Cell::new(CV_ERR_MODE_LEAF) };
}

/// `cvGetErrMode(void)` (`cxerror.cpp:299`).
pub fn cv_get_err_mode() -> i32 {
    ERR_MODE.with(|m| m.get())
}

/// `cvSetErrMode(mode)` (`cxerror.cpp:314`).
pub fn cv_set_err_mode(mode: i32) -> i32 {
    ERR_MODE.with(|m| m.replace(mode))
}

/// `cvErrorStr( int status )` (`cxerror.cpp:250`).
pub fn cv_error_str(status: i32) -> String {
    match status {
        CV_STS_OK => "No Error".into(),
        CV_STS_BACK_TRACE => "Backtrace".into(),
        CV_STS_ERROR => "Unspecified error".into(),
        CV_STS_INTERNAL => "Internal error".into(),
        CV_STS_NO_MEM => "Insufficient memory".into(),
        CV_STS_BAD_ARG => "Bad argument".into(),
        CV_STS_NO_CONV => "Iterations do not converge".into(),
        CV_STS_AUTO_TRACE => "Autotrace call".into(),
        CV_STS_BAD_SIZE => "Incorrect size of input array".into(),
        CV_STS_NULL_PTR => "Null pointer".into(),
        CV_STS_DIV_BY_ZERO => "Divizion by zero occured".into(),
        CV_BAD_STEP => "Image step is wrong".into(),
        CV_STS_INPLACE_NOT_SUPPORTED => "Inplace operation is not supported".into(),
        CV_STS_OBJECT_NOT_FOUND => "Requested object was not found".into(),
        CV_BAD_DEPTH => "Input image depth is not supported by function".into(),
        CV_STS_UNMATCHED_FORMATS => "Formats of input arguments do not match".into(),
        CV_STS_UNMATCHED_SIZES => "Sizes of input arguments do not match".into(),
        CV_STS_OUT_OF_RANGE => "One of arguments' values is out of range".into(),
        CV_STS_UNSUPPORTED_FORMAT => "Unsupported format or combination of formats".into(),
        CV_BAD_COI => "Input COI is not supported".into(),
        CV_BAD_NUM_CHANNELS => "Bad number of channels".into(),
        CV_STS_BAD_FLAG => "Bad flag (parameter or structure field)".into(),
        CV_STS_BAD_POINT => "Bad parameter of type CvPoint".into(),
        CV_STS_BAD_MASK => "Bad type of mask argument".into(),
        CV_STS_PARSE_ERROR => "Parsing error".into(),
        CV_STS_NOT_IMPLEMENTED => "The function/feature is not implemented".into(),
        CV_STS_BAD_MEM_BLOCK => "Memory block has been corrupted".into(),
        _ => format!(
            "Unknown {} code {}",
            if status >= 0 { "status" } else { "error" },
            status
        ),
    }
}

/// `cvStdErrReport( int code, const char* func_name, const char* err_msg,
/// const char* file, int line, void* )` (`cxerror.cpp:141`), the default
/// error callback on this platform.  Returns whether to terminate.
fn cv_std_err_report(code: i32, func_name: &str, err_msg: &str, file: &str, line: i32) -> i32 {
    use std::io::Write as _;
    let mut e = std::io::stderr();
    if code == CV_STS_BACK_TRACE || code == CV_STS_AUTO_TRACE {
        let _ = write!(e, "\tcalled from ");
    } else {
        let _ = write!(
            e,
            "OpenCV ERROR: {} ({})\n\tin function ",
            cv_error_str(code),
            err_msg
        );
    }

    let _ = writeln!(e, "{}, {}({})", func_name, file, line);

    if cv_get_err_mode() == CV_ERR_MODE_LEAF {
        let _ = writeln!(e, "Terminating the application...");
        1
    } else {
        0
    }
}

/// `cvError( int code, const char* func_name, const char* err_msg, const
/// char* file_name, int line )` (`cxerror.cpp:330`), as `CV_ERROR` reaches
/// it with a non-`CV_StsOk` code.  In leaf mode it reports and terminates
/// with status 255 (see the module documentation for why not `SIGABRT`).
pub fn cv_error(code: i32, func_name: &str, err_msg: &str, file_name: &str, line: i32) -> ! {
    let mode = cv_get_err_mode();
    let mut terminate = 0;
    if mode != CV_ERR_MODE_SILENT {
        terminate = cv_std_err_report(code, func_name, err_msg, file_name, line);
    }
    if terminate != 0 {
        crate::imod::libcfshr::b3dutil::exit(-terminate.abs());
    }
    panic!(
        "OpenCV error {code} in {func_name} with error mode {mode}: RAPTOR only runs in CV_ErrModeLeaf"
    );
}
