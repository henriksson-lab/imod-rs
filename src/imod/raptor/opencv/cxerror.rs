//! Owned Rust translation of `IMOD/raptor/opencv/cxerror.cpp` and its paired
//! declarations in `cxerror.h`, `cxcore.h`, and `cxmisc.h`.
//!
//! OpenCV 1.x stores mutable error state behind C global/TLS pointers and can
//! terminate the process from an error callback.  This version keeps one owned
//! context per Rust thread and reports termination as `CvRaisedError`, so a
//! Raptor caller can decide where failure crosses its process boundary.

use std::any::Any;
use std::cell::RefCell;

/// C `CVStatus`; retaining the integer representation also preserves unknown
/// status codes from third-party OpenCV-era callers.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct CvStatus(pub i32);

#[allow(non_upper_case_globals)]
impl CvStatus {
    pub const sts_ok: Self = Self(0);
    pub const sts_back_trace: Self = Self(-1);
    pub const sts_error: Self = Self(-2);
    pub const sts_internal: Self = Self(-3);
    pub const sts_no_mem: Self = Self(-4);
    pub const sts_bad_arg: Self = Self(-5);
    pub const sts_bad_func: Self = Self(-6);
    pub const sts_no_conv: Self = Self(-7);
    pub const sts_auto_trace: Self = Self(-8);
    pub const header_is_null: Self = Self(-9);
    pub const bad_image_size: Self = Self(-10);
    pub const bad_offset: Self = Self(-11);
    pub const bad_data_ptr: Self = Self(-12);
    pub const bad_step: Self = Self(-13);
    pub const bad_model_or_channel_sequence: Self = Self(-14);
    pub const bad_num_channels: Self = Self(-15);
    pub const bad_num_channel_1u: Self = Self(-16);
    pub const bad_depth: Self = Self(-17);
    pub const bad_alpha_channel: Self = Self(-18);
    pub const bad_order: Self = Self(-19);
    pub const bad_origin: Self = Self(-20);
    pub const bad_align: Self = Self(-21);
    pub const bad_callback: Self = Self(-22);
    pub const bad_tile_size: Self = Self(-23);
    pub const bad_coi: Self = Self(-24);
    pub const bad_roi_size: Self = Self(-25);
    pub const mask_is_tiled: Self = Self(-26);
    pub const sts_null_ptr: Self = Self(-27);
    pub const sts_vec_length_err: Self = Self(-28);
    pub const sts_filter_struct_content_err: Self = Self(-29);
    pub const sts_kernel_struct_content_err: Self = Self(-30);
    pub const sts_filter_offset_err: Self = Self(-31);
    pub const sts_bad_size: Self = Self(-201);
    pub const sts_div_by_zero: Self = Self(-202);
    pub const sts_inplace_not_supported: Self = Self(-203);
    pub const sts_object_not_found: Self = Self(-204);
    pub const sts_unmatched_formats: Self = Self(-205);
    pub const sts_bad_flag: Self = Self(-206);
    pub const sts_bad_point: Self = Self(-207);
    pub const sts_bad_mask: Self = Self(-208);
    pub const sts_unmatched_sizes: Self = Self(-209);
    pub const sts_unsupported_format: Self = Self(-210);
    pub const sts_out_of_range: Self = Self(-211);
    pub const sts_parse_error: Self = Self(-212);
    pub const sts_not_implemented: Self = Self(-213);
    pub const sts_bad_mem_block: Self = Self(-214);

    pub fn is_error(self) -> bool {
        self.0 < 0
    }
}

/// C `CV_ErrMode*` values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CvErrorMode {
    /// C `CV_ErrModeLeaf`: report and terminate.
    #[default]
    Leaf,
    /// C `CV_ErrModeParent`: report and continue.
    Parent,
    /// C `CV_ErrModeSilent`: retain status without reporting.
    Silent,
}

/// Detailed, owned equivalent of C `cvGetErrInfo` output parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CvErrorInfo {
    pub status: CvStatus,
    pub status_description: String,
    pub description: Option<String>,
    pub function: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
}

/// The callback result replacing C's integer termination flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvErrorAction {
    Continue,
    Terminate(i32),
}

/// Error propagated when the C callback would have called `exit`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CvRaisedError {
    pub info: CvErrorInfo,
    pub exit_code: i32,
}

/// Typed replacement for C `CvErrorCallback`.
pub trait CvErrorCallback: Send {
    fn report(&mut self, info: &CvErrorInfo, userdata: &mut dyn Any) -> CvErrorAction;
}

impl<F> CvErrorCallback for F
where
    F: FnMut(&CvErrorInfo, &mut dyn Any) -> CvErrorAction + Send,
{
    fn report(&mut self, info: &CvErrorInfo, userdata: &mut dyn Any) -> CvErrorAction {
        self(info, userdata)
    }
}

/// An error handler and the owned state corresponding to C callback userdata.
pub struct CvErrorHandler {
    callback: Box<dyn CvErrorCallback>,
    userdata: Box<dyn Any + Send>,
}

impl CvErrorHandler {
    pub fn new<C, U>(callback: C, userdata: U) -> Self
    where
        C: CvErrorCallback + 'static,
        U: Any + Send,
    {
        Self {
            callback: Box::new(callback),
            userdata: Box::new(userdata),
        }
    }
}

#[derive(Default)]
struct CvContext {
    status: CvStatus,
    mode: CvErrorMode,
    handler: Option<CvErrorHandler>,
    information: Option<CvErrorInfo>,
}

thread_local! {
    static CV_CONTEXT: RefCell<CvContext> = RefCell::new(CvContext::default());
}

/// C `icvCreateContext`: Rust constructs the owned TLS context lazily.
fn icv_create_context() -> CvContext {
    CvContext::default()
}

/// C `icvDestroyContext`: dropping the owned context performs destruction.
fn icv_destroy_context(context: CvContext) {
    drop(context);
}

/// C `icvGetContext`: execute a closure against the current thread's context.
fn icv_get_context<R>(operation: impl FnOnce(&mut CvContext) -> R) -> R {
    CV_CONTEXT.with(|context| operation(&mut context.borrow_mut()))
}

/// C `cvStdErrReport`.
pub fn cv_std_err_report(info: &CvErrorInfo, mode: CvErrorMode) -> CvErrorAction {
    if info.status == CvStatus::sts_back_trace || info.status == CvStatus::sts_auto_trace {
        eprintln!(
            "\tcalled from {}, {}({})",
            info.function.as_deref().unwrap_or("<unknown>"),
            info.file.as_deref().unwrap_or(""),
            info.line.unwrap_or(0)
        );
    } else {
        eprintln!(
            "OpenCV ERROR: {} ({})\n\tin function {}, {}({})",
            info.status_description,
            info.description.as_deref().unwrap_or("no description"),
            info.function.as_deref().unwrap_or("<unknown>"),
            info.file.as_deref().unwrap_or(""),
            info.line.unwrap_or(0)
        );
    }
    if mode == CvErrorMode::Leaf {
        eprintln!("Terminating the application...");
        CvErrorAction::Terminate(1)
    } else {
        CvErrorAction::Continue
    }
}

/// C `cvGuiBoxReport`; without a C GUI backend it has the POSIX source's
/// exact fallback behavior.
pub fn cv_gui_box_report(info: &CvErrorInfo, mode: CvErrorMode) -> CvErrorAction {
    cv_std_err_report(info, mode)
}

/// C `cvNulDevReport`.
pub fn cv_nul_dev_report(_: &CvErrorInfo, mode: CvErrorMode) -> CvErrorAction {
    if mode == CvErrorMode::Leaf {
        CvErrorAction::Terminate(1)
    } else {
        CvErrorAction::Continue
    }
}

/// C `cvRedirectError`. `None` restores the platform default handler.
pub fn cv_redirect_error(handler: Option<CvErrorHandler>) -> Option<CvErrorHandler> {
    icv_get_context(|context| std::mem::replace(&mut context.handler, handler))
}

/// C `cvGetErrInfo`.
pub fn cv_get_err_info() -> CvErrorInfo {
    icv_get_context(|context| {
        context.information.clone().unwrap_or_else(|| CvErrorInfo {
            status: context.status,
            status_description: cv_error_str(context.status),
            description: None,
            function: None,
            file: None,
            line: None,
        })
    })
}

/// C `cvErrorStr`, using owned text instead of its mutable shared C buffer.
pub fn cv_error_str(status: CvStatus) -> String {
    let description = match status {
        CvStatus::sts_ok => "No Error",
        CvStatus::sts_back_trace => "Backtrace",
        CvStatus::sts_error => "Unspecified error",
        CvStatus::sts_internal => "Internal error",
        CvStatus::sts_no_mem => "Insufficient memory",
        CvStatus::sts_bad_arg => "Bad argument",
        CvStatus::sts_no_conv => "Iterations do not converge",
        CvStatus::sts_auto_trace => "Autotrace call",
        CvStatus::sts_bad_size => "Incorrect size of input array",
        CvStatus::sts_null_ptr => "Null pointer",
        CvStatus::sts_div_by_zero => "Divizion by zero occured",
        CvStatus::bad_step => "Image step is wrong",
        CvStatus::sts_inplace_not_supported => "Inplace operation is not supported",
        CvStatus::sts_object_not_found => "Requested object was not found",
        CvStatus::bad_depth => "Input image depth is not supported by function",
        CvStatus::sts_unmatched_formats => "Formats of input arguments do not match",
        CvStatus::sts_unmatched_sizes => "Sizes of input arguments do not match",
        CvStatus::sts_out_of_range => "One of arguments' values is out of range",
        CvStatus::sts_unsupported_format => "Unsupported format or combination of formats",
        CvStatus::bad_coi => "Input COI is not supported",
        CvStatus::bad_num_channels => "Bad number of channels",
        CvStatus::sts_bad_flag => "Bad flag (parameter or structure field)",
        CvStatus::sts_bad_point => "Bad parameter of type CvPoint",
        CvStatus::sts_bad_mask => "Bad type of mask argument",
        CvStatus::sts_parse_error => "Parsing error",
        CvStatus::sts_not_implemented => "The function/feature is not implemented",
        CvStatus::sts_bad_mem_block => "Memory block has been corrupted",
        _ => {
            return format!(
                "Unknown {} code {}",
                if status.0 >= 0 { "status" } else { "error" },
                status.0
            );
        }
    };
    description.to_owned()
}

/// C `cvGetErrMode`.
pub fn cv_get_err_mode() -> CvErrorMode {
    icv_get_context(|context| context.mode)
}

/// C `cvSetErrMode`.
pub fn cv_set_err_mode(mode: CvErrorMode) -> CvErrorMode {
    icv_get_context(|context| std::mem::replace(&mut context.mode, mode))
}

/// C `cvGetErrStatus`.
pub fn cv_get_err_status() -> CvStatus {
    icv_get_context(|context| context.status)
}

/// C `cvSetErrStatus`.
pub fn cv_set_err_status(status: CvStatus) {
    icv_get_context(|context| {
        context.status = status;
        if !status.is_error() {
            context.information = None;
        }
    });
}

/// C `cvError`.
pub fn cv_error(
    status: CvStatus,
    function: impl Into<String>,
    description: impl Into<String>,
    file: impl Into<String>,
    line: u32,
) -> Result<(), CvRaisedError> {
    if status == CvStatus::sts_ok {
        cv_set_err_status(status);
        return Ok(());
    }

    let information = CvErrorInfo {
        status,
        status_description: cv_error_str(status),
        description: Some(description.into()),
        function: Some(function.into()),
        file: Some(file.into()),
        line: Some(line),
    };
    let action = icv_get_context(|context| {
        if status != CvStatus::sts_back_trace && status != CvStatus::sts_auto_trace {
            context.status = status;
            context.information = Some(information.clone());
        }
        match context.handler.as_mut() {
            Some(handler) => handler
                .callback
                .report(&information, handler.userdata.as_mut()),
            None => cv_std_err_report(&information, context.mode),
        }
    });
    match action {
        CvErrorAction::Continue => Ok(()),
        CvErrorAction::Terminate(exit_code) => Err(CvRaisedError {
            info: information,
            exit_code: -exit_code.abs(),
        }),
    }
}

/// C `icvPthreadDestructor`; TLS ownership makes it a drop operation.
pub fn icv_pthread_destructor() {
    icv_get_context(|context| {
        let previous = std::mem::take(context);
        icv_destroy_context(previous);
        *context = icv_create_context();
    });
}

/// C `cvErrorFromIppStatus`.
pub fn cv_error_from_ipp_status(status: i32) -> CvStatus {
    match status {
        -1 => CvStatus::sts_bad_size,
        -113 => CvStatus::sts_bad_mem_block,
        -2 => CvStatus::sts_null_ptr,
        -11 => CvStatus::sts_div_by_zero,
        -29 => CvStatus::bad_step,
        -3 => CvStatus::sts_no_mem,
        -49 => CvStatus::sts_bad_arg,
        -48 => CvStatus::sts_error,
        -112 => CvStatus::sts_inplace_not_supported,
        -110 => CvStatus::sts_object_not_found,
        -109 => CvStatus::sts_no_conv,
        -107 => CvStatus::bad_depth,
        -104 => CvStatus::sts_unmatched_formats,
        -103 => CvStatus::bad_coi,
        -102 => CvStatus::bad_num_channels,
        -12 => CvStatus::sts_bad_flag,
        -44 | -10 | -7 => CvStatus::sts_bad_arg,
        -6 => CvStatus::sts_bad_point,
        _ => CvStatus::sts_error,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CvErrorAction, CvErrorHandler, CvErrorMode, CvStatus, cv_error, cv_error_from_ipp_status,
        cv_error_str, cv_get_err_info, cv_get_err_status, cv_redirect_error, cv_set_err_mode,
        cv_set_err_status,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counting_callback(
        _: &super::CvErrorInfo,
        userdata: &mut dyn std::any::Any,
    ) -> CvErrorAction {
        let (calls, count) = userdata
            .downcast_mut::<(Arc<AtomicUsize>, usize)>()
            .unwrap();
        calls.fetch_add(1, Ordering::SeqCst);
        *count += 1;
        CvErrorAction::Continue
    }

    #[test]
    fn silent_error_retains_owned_context_details() {
        cv_redirect_error(None);
        cv_set_err_mode(CvErrorMode::Silent);
        cv_set_err_status(CvStatus::sts_ok);
        cv_error(CvStatus::sts_bad_arg, "cvUnit", "bad test", "unit.cpp", 17).unwrap();
        let info = cv_get_err_info();
        assert_eq!(info.status, CvStatus::sts_bad_arg);
        assert_eq!(info.description.as_deref(), Some("bad test"));
        assert_eq!(info.file.as_deref(), Some("unit.cpp"));
        assert_eq!(info.line, Some(17));
    }

    #[test]
    fn leaf_mode_propagates_the_former_exit_as_an_error() {
        cv_redirect_error(None);
        cv_set_err_mode(CvErrorMode::Leaf);
        let error =
            cv_error(CvStatus::sts_no_mem, "cvUnit", "no memory", "unit.cpp", 2).unwrap_err();
        assert_eq!(error.exit_code, -1);
        assert_eq!(error.info.status, CvStatus::sts_no_mem);
    }

    #[test]
    fn redirect_error_owns_callback_and_userdata() {
        let calls = Arc::new(AtomicUsize::new(0));
        cv_redirect_error(Some(CvErrorHandler::new(
            counting_callback,
            (Arc::clone(&calls), 0_usize),
        )));
        cv_error(
            CvStatus::sts_bad_point,
            "cvUnit",
            "bad point",
            "unit.cpp",
            4,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        cv_redirect_error(None);
    }

    #[test]
    fn status_text_and_ipp_mapping_follow_source_cases() {
        assert_eq!(cv_error_str(CvStatus::bad_step), "Image step is wrong");
        assert_eq!(cv_error_str(CvStatus(-999)), "Unknown error code -999");
        assert_eq!(cv_error_from_ipp_status(-113), CvStatus::sts_bad_mem_block);
        assert_eq!(cv_error_from_ipp_status(-44), CvStatus::sts_bad_arg);
        assert_eq!(cv_error_from_ipp_status(99), CvStatus::sts_error);
    }

    #[test]
    fn ok_status_clears_previous_error_details() {
        cv_set_err_status(CvStatus::sts_bad_arg);
        cv_error(CvStatus::sts_ok, "", "", "", 0).unwrap();
        assert_eq!(cv_get_err_status(), CvStatus::sts_ok);
        assert!(cv_get_err_info().description.is_none());
    }
}
