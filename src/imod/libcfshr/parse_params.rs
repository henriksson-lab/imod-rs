unsafe extern "C" {
    static mut stdin: *mut FILE;
    static mut stdout: *mut FILE;
    static mut stderr: *mut FILE;
    fn fclose(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn fflush(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn fopen(
        __filename: *const ::core::ffi::c_char,
        __modes: *const ::core::ffi::c_char,
    ) -> *mut FILE;
    fn fprintf(
        __stream: *mut FILE,
        __format: *const ::core::ffi::c_char,
        ...
    ) -> ::core::ffi::c_int;
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn sprintf(
        __s: *mut ::core::ffi::c_char,
        __format: *const ::core::ffi::c_char,
        ...
    ) -> ::core::ffi::c_int;
    fn fgets(
        __s: *mut ::core::ffi::c_char,
        __n: ::core::ffi::c_int,
        __stream: *mut FILE,
    ) -> *mut ::core::ffi::c_char;
    fn rewind(__stream: *mut FILE);
    fn feof(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn toupper(__c: ::core::ffi::c_int) -> ::core::ffi::c_int;
    fn atoi(__nptr: *const ::core::ffi::c_char) -> ::core::ffi::c_int;
    fn strtod(
        __nptr: *const ::core::ffi::c_char,
        __endptr: *mut *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_double;
    fn strtol(
        __nptr: *const ::core::ffi::c_char,
        __endptr: *mut *mut ::core::ffi::c_char,
        __base: ::core::ffi::c_int,
    ) -> ::core::ffi::c_long;
    fn malloc(__size: size_t) -> *mut ::core::ffi::c_void;
    fn realloc(__ptr: *mut ::core::ffi::c_void, __size: size_t) -> *mut ::core::ffi::c_void;
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn exit(__status: ::core::ffi::c_int) -> !;
    fn getenv(__name: *const ::core::ffi::c_char) -> *mut ::core::ffi::c_char;
    fn strncpy(
        __dest: *mut ::core::ffi::c_char,
        __src: *const ::core::ffi::c_char,
        __n: size_t,
    ) -> *mut ::core::ffi::c_char;
    fn strcat(
        __dest: *mut ::core::ffi::c_char,
        __src: *const ::core::ffi::c_char,
    ) -> *mut ::core::ffi::c_char;
    fn strcmp(
        __s1: *const ::core::ffi::c_char,
        __s2: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    fn strdup(__s: *const ::core::ffi::c_char) -> *mut ::core::ffi::c_char;
    fn strchr(__s: *const ::core::ffi::c_char, __c: ::core::ffi::c_int)
    -> *mut ::core::ffi::c_char;
    fn strpbrk(
        __s: *const ::core::ffi::c_char,
        __accept: *const ::core::ffi::c_char,
    ) -> *mut ::core::ffi::c_char;
    fn strstr(
        __haystack: *const ::core::ffi::c_char,
        __needle: *const ::core::ffi::c_char,
    ) -> *mut ::core::ffi::c_char;
    fn strlen(__s: *const ::core::ffi::c_char) -> size_t;
}
use crate::imod::libcfshr::b3dutil::expand_arg_list;
pub struct _IO_wide_data {
    _private: [u8; 0],
}
pub struct _IO_codecvt {
    _private: [u8; 0],
}
pub struct _IO_marker {
    _private: [u8; 0],
}
pub type __builtin_va_list = [__va_list_tag; 1];
#[derive(Copy, Clone)]
#[repr(C)]
pub struct __va_list_tag {
    pub gp_offset: ::core::ffi::c_uint,
    pub fp_offset: ::core::ffi::c_uint,
    pub overflow_arg_area: *mut ::core::ffi::c_void,
    pub reg_save_area: *mut ::core::ffi::c_void,
}
pub type size_t = usize;
pub type va_list = __builtin_va_list;
pub type __uint16_t = u16;
pub type __uint32_t = u32;
pub type __uint64_t = u64;
pub type __off_t = ::core::ffi::c_long;
pub type __off64_t = ::core::ffi::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: ::core::ffi::c_int,
    pub _IO_read_ptr: *mut ::core::ffi::c_char,
    pub _IO_read_end: *mut ::core::ffi::c_char,
    pub _IO_read_base: *mut ::core::ffi::c_char,
    pub _IO_write_base: *mut ::core::ffi::c_char,
    pub _IO_write_ptr: *mut ::core::ffi::c_char,
    pub _IO_write_end: *mut ::core::ffi::c_char,
    pub _IO_buf_base: *mut ::core::ffi::c_char,
    pub _IO_buf_end: *mut ::core::ffi::c_char,
    pub _IO_save_base: *mut ::core::ffi::c_char,
    pub _IO_backup_base: *mut ::core::ffi::c_char,
    pub _IO_save_end: *mut ::core::ffi::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1],
    pub _lock: *mut ::core::ffi::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: size_t,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
pub type PipOptions = pipOptions;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct pipOptions {
    pub shortName: *mut ::core::ffi::c_char,
    pub longName: *mut ::core::ffi::c_char,
    pub type_0: *mut ::core::ffi::c_char,
    pub helpString: *mut ::core::ffi::c_char,
    pub format: *mut ::core::ffi::c_char,
    pub defaultVal: *mut ::core::ffi::c_char,
    pub valuePtr: *mut *mut ::core::ffi::c_char,
    pub multiple: ::core::ffi::c_int,
    pub count: ::core::ffi::c_int,
    pub lenShort: ::core::ffi::c_int,
    pub lenLong: ::core::ffi::c_int,
    pub nextLinked: *mut ::core::ffi::c_int,
    pub linked: ::core::ffi::c_int,
}
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
pub const PIP_INTEGER: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
pub const PIP_FLOAT: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const PIP_DOUBLE: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
#[inline]
unsafe extern "C" fn __bswap_16(mut __bsx: __uint16_t) -> __uint16_t {
    return (__bsx as ::core::ffi::c_int >> 8 as ::core::ffi::c_int & 0xff as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_int & 0xff as ::core::ffi::c_int) << 8 as ::core::ffi::c_int)
        as __uint16_t;
}
#[inline]
unsafe extern "C" fn __bswap_32(mut __bsx: __uint32_t) -> __uint32_t {
    return (__bsx & 0xff000000 as __uint32_t) >> 24 as ::core::ffi::c_int
        | (__bsx & 0xff0000 as __uint32_t) >> 8 as ::core::ffi::c_int
        | (__bsx & 0xff00 as __uint32_t) << 8 as ::core::ffi::c_int
        | (__bsx & 0xff as __uint32_t) << 24 as ::core::ffi::c_int;
}
#[inline]
unsafe extern "C" fn __bswap_64(mut __bsx: __uint64_t) -> __uint64_t {
    return ((__bsx as ::core::ffi::c_ulonglong & 0xff00000000000000 as ::core::ffi::c_ulonglong)
        >> 56 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000000000 as ::core::ffi::c_ulonglong)
            >> 40 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000000000 as ::core::ffi::c_ulonglong)
            >> 24 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00000000 as ::core::ffi::c_ulonglong)
            >> 8 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff000000 as ::core::ffi::c_ulonglong)
            << 8 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff0000 as ::core::ffi::c_ulonglong)
            << 24 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff00 as ::core::ffi::c_ulonglong)
            << 40 as ::core::ffi::c_int
        | (__bsx as ::core::ffi::c_ulonglong & 0xff as ::core::ffi::c_ulonglong)
            << 56 as ::core::ffi::c_int) as __uint64_t;
}
#[inline]
unsafe extern "C" fn __uint16_identity(mut __x: __uint16_t) -> __uint16_t {
    return __x;
}
#[inline]
unsafe extern "C" fn __uint32_identity(mut __x: __uint32_t) -> __uint32_t {
    return __x;
}
#[inline]
unsafe extern "C" fn __uint64_identity(mut __x: __uint64_t) -> __uint64_t {
    return __x;
}
pub const PATH_MAX: ::core::ffi::c_int = 4096 as ::core::ffi::c_int;
pub const NON_OPTION_STRING: [::core::ffi::c_char; 18] = unsafe {
    ::core::mem::transmute::<[u8; 18], [::core::ffi::c_char; 18]>(*b"NonOptionArgument\0")
};
pub const STANDARD_INPUT_STRING: [::core::ffi::c_char; 14] =
    unsafe { ::core::mem::transmute::<[u8; 14], [::core::ffi::c_char; 14]>(*b"StandardInput\0") };
pub const STANDARD_INPUT_END: [::core::ffi::c_char; 9] =
    unsafe { ::core::mem::transmute::<[u8; 9], [::core::ffi::c_char; 9]>(*b"EndInput\0") };
pub const PARAM_FILE_STRING: [::core::ffi::c_char; 3] =
    unsafe { ::core::mem::transmute::<[u8; 3], [::core::ffi::c_char; 3]>(*b"PF\0") };
pub const BOOLEAN_STRING: [::core::ffi::c_char; 2] =
    unsafe { ::core::mem::transmute::<[u8; 2], [::core::ffi::c_char; 2]>(*b"B\0") };
pub const LOOKUP_NOT_FOUND: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
pub const LOOKUP_AMBIGUOUS: ::core::ffi::c_int = -(2 as ::core::ffi::c_int);
pub const TEMP_STR_SIZE: ::core::ffi::c_int = 1024 as ::core::ffi::c_int;
pub const LINE_STR_SIZE: ::core::ffi::c_int = 102400 as ::core::ffi::c_int;
pub const ADOC_STR_SIZE: ::core::ffi::c_int = 10240 as ::core::ffi::c_int;
pub const PREFIX_SIZE: ::core::ffi::c_int = 64 as ::core::ffi::c_int;
pub const PATH_SEPARATOR: ::core::ffi::c_int = '/' as i32;
pub const OPTFILE_DIR: [::core::ffi::c_char; 8] =
    unsafe { ::core::mem::transmute::<[u8; 8], [::core::ffi::c_char; 8]>(*b"autodoc\0") };
pub const OPTFILE_EXT: [::core::ffi::c_char; 5] =
    unsafe { ::core::mem::transmute::<[u8; 5], [::core::ffi::c_char; 5]>(*b"adoc\0") };
pub const OPTDIR_VARIABLE: [::core::ffi::c_char; 12] =
    unsafe { ::core::mem::transmute::<[u8; 12], [::core::ffi::c_char; 12]>(*b"AUTODOC_DIR\0") };
pub const DEFAULTS_FILE: [::core::ffi::c_char; 18] = unsafe {
    ::core::mem::transmute::<[u8; 18], [::core::ffi::c_char; 18]>(*b"progDefaults.adoc\0")
};
pub const DEFAULTS_DIR: [::core::ffi::c_char; 4] =
    unsafe { ::core::mem::transmute::<[u8; 4], [::core::ffi::c_char; 4]>(*b"com\0") };
pub const DEFAULT_SUB_STR: [::core::ffi::c_char; 11] =
    unsafe { ::core::mem::transmute::<[u8; 11], [::core::ffi::c_char; 11]>(*b"%{default}\0") };
pub const PRINTENTRY_VARIABLE: [::core::ffi::c_char; 18] = unsafe {
    ::core::mem::transmute::<[u8; 18], [::core::ffi::c_char; 18]>(*b"PIP_PRINT_ENTRIES\0")
};
pub const OPEN_DELIM: [::core::ffi::c_char; 2] =
    unsafe { ::core::mem::transmute::<[u8; 2], [::core::ffi::c_char; 2]>(*b"[\0") };
pub const CLOSE_DELIM: [::core::ffi::c_char; 2] =
    unsafe { ::core::mem::transmute::<[u8; 2], [::core::ffi::c_char; 2]>(*b"]\0") };
pub const VALUE_DELIM: [::core::ffi::c_char; 2] =
    unsafe { ::core::mem::transmute::<[u8; 2], [::core::ffi::c_char; 2]>(*b"=\0") };
static mut sTypes: [*mut ::core::ffi::c_char; 13] = [
    BOOLEAN_STRING.as_ptr() as *mut ::core::ffi::c_char,
    PARAM_FILE_STRING.as_ptr() as *mut ::core::ffi::c_char,
    b"LI\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"I\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"F\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"IP\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"FP\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"IT\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"FT\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"IA\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"FA\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"CH\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"FN\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
];
static mut sTypeDescriptions: [*mut ::core::ffi::c_char; 14] = [
    b"Boolean\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Parameter file\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"List of integer ranges\0" as *const u8 as *const ::core::ffi::c_char
        as *mut ::core::ffi::c_char,
    b"Integer\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Floating point\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Two integers\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Two floats\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Three integers\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Three floats\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Multiple integers\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Multiple floats\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Text string\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"File name\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Unknown argument type\0" as *const u8 as *const ::core::ffi::c_char
        as *mut ::core::ffi::c_char,
];
static mut sTypeForUsage: [*mut ::core::ffi::c_char; 14] = [
    b"Boolean\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"File\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"List\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Int\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Float\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"2 ints\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"2 floats\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"3 ints\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"3 floats\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Ints\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Floats\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"String\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"File\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char,
    b"Unknown argument type\0" as *const u8 as *const ::core::ffi::c_char
        as *mut ::core::ffi::c_char,
];
static mut sNumTypes: ::core::ffi::c_char = 13 as ::core::ffi::c_char;
static mut sNullChar: ::core::ffi::c_char = 0 as ::core::ffi::c_char;
static mut sNullString: *mut ::core::ffi::c_char =
    unsafe { &raw const sNullChar as *mut ::core::ffi::c_char };
static mut sQuoteTypes: *mut ::core::ffi::c_char =
    b"\"'`\0" as *const u8 as *const ::core::ffi::c_char as *mut ::core::ffi::c_char;
static mut sHighestNonOptGotten: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
static mut sOptTable: *mut PipOptions = ::core::ptr::null::<PipOptions>() as *mut PipOptions;
static mut sTableSize: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNumOptions: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNonOptInd: ::core::ffi::c_int = 0;
static mut sErrorString: *mut ::core::ffi::c_char =
    ::core::ptr::null::<::core::ffi::c_char>() as *mut ::core::ffi::c_char;
static mut sUsageString: *mut ::core::ffi::c_char =
    ::core::ptr::null::<::core::ffi::c_char>() as *mut ::core::ffi::c_char;
static mut sExitPrefix: [::core::ffi::c_char; 64] = unsafe {
    ::core::mem::transmute::<
        [u8; 64],
        [::core::ffi::c_char; 64],
    >(
        *b"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    )
};
static mut sErrorDest: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNextOption: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNextArgBelongsTo: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
static mut sNumOptionArguments: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sTempStr: *mut ::core::ffi::c_char =
    ::core::ptr::null::<::core::ffi::c_char>() as *mut ::core::ffi::c_char;
static mut sLineStr: *mut ::core::ffi::c_char =
    ::core::ptr::null::<::core::ffi::c_char>() as *mut ::core::ffi::c_char;
static mut sAllowDefaults: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sOutputManpage: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sPrintEntries: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
static mut sDefaultDelim: [::core::ffi::c_char; 2] = VALUE_DELIM;
static mut sValueDelim: *mut ::core::ffi::c_char =
    unsafe { &raw const sDefaultDelim as *mut ::core::ffi::c_char };
static mut sProgramName: *mut ::core::ffi::c_char =
    unsafe { &raw const sNullChar as *mut ::core::ffi::c_char };
static mut sNoCase: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sDoneEnds: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sTakeStdIn: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNonOptLines: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNoAbbrevs: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNotFoundOK: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sLinkedOption: *mut ::core::ffi::c_char =
    ::core::ptr::null::<::core::ffi::c_char>() as *mut ::core::ffi::c_char;
static mut sTestAbbrevForUsage: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sDoubleDashOptions: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNoHelpAbbrevs: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_initialize(mut numOpts: ::core::ffi::c_int) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    if sTempStr.is_null() {
        sTempStr = malloc(TEMP_STR_SIZE as size_t) as *mut ::core::ffi::c_char;
    }
    sLineStr = malloc(LINE_STR_SIZE as size_t) as *mut ::core::ffi::c_char;
    sNumOptions = numOpts;
    sTableSize = numOpts + 2 as ::core::ffi::c_int;
    sNonOptInd = sNumOptions;
    sOptTable =
        malloc((sTableSize as size_t).wrapping_mul(::core::mem::size_of::<PipOptions>() as size_t))
            as *mut PipOptions;
    if sTempStr.is_null() || sLineStr.is_null() || sOptTable.is_null() {
        pip_memory_error(
            NULL,
            b"pip_initialize\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    i = 0 as ::core::ffi::c_int;
    while i < sTableSize {
        let ref mut fresh0 = (*sOptTable.offset(i as isize)).shortName;
        *fresh0 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        let ref mut fresh1 = (*sOptTable.offset(i as isize)).longName;
        *fresh1 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        let ref mut fresh2 = (*sOptTable.offset(i as isize)).type_0;
        *fresh2 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        let ref mut fresh3 = (*sOptTable.offset(i as isize)).helpString;
        *fresh3 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        let ref mut fresh4 = (*sOptTable.offset(i as isize)).format;
        *fresh4 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        let ref mut fresh5 = (*sOptTable.offset(i as isize)).defaultVal;
        *fresh5 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        let ref mut fresh6 = (*sOptTable.offset(i as isize)).valuePtr;
        *fresh6 = ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
        (*sOptTable.offset(i as isize)).multiple = 0 as ::core::ffi::c_int;
        (*sOptTable.offset(i as isize)).count = 0 as ::core::ffi::c_int;
        let ref mut fresh7 = (*sOptTable.offset(i as isize)).nextLinked;
        *fresh7 = ::core::ptr::null_mut::<::core::ffi::c_int>();
        (*sOptTable.offset(i as isize)).linked = 0 as ::core::ffi::c_int;
        i += 1;
    }
    let ref mut fresh8 = (*sOptTable.offset(sNonOptInd as isize)).longName;
    *fresh8 = strdup(NON_OPTION_STRING.as_ptr());
    let ref mut fresh9 =
        (*sOptTable.offset((sNonOptInd + 1 as ::core::ffi::c_int) as isize)).shortName;
    *fresh9 = strdup(STANDARD_INPUT_STRING.as_ptr());
    let ref mut fresh10 =
        (*sOptTable.offset((sNonOptInd + 1 as ::core::ffi::c_int) as isize)).longName;
    *fresh10 = strdup(STANDARD_INPUT_END.as_ptr());
    if (*sOptTable.offset(sNonOptInd as isize)).longName.is_null()
        || (*sOptTable.offset((sNonOptInd + 1 as ::core::ffi::c_int) as isize))
            .shortName
            .is_null()
        || (*sOptTable.offset((sNonOptInd + 1 as ::core::ffi::c_int) as isize))
            .longName
            .is_null()
    {
        pip_memory_error(
            NULL,
            b"pip_initialize\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    (*sOptTable.offset(sNonOptInd as isize)).multiple = 1 as ::core::ffi::c_int;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_done() {
    let mut i: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut optp: *mut PipOptions = ::core::ptr::null_mut::<PipOptions>();
    pip_warn_unused_non_opt_args();
    i = 0 as ::core::ffi::c_int;
    while i < sTableSize {
        optp = sOptTable.offset(i as isize) as *mut PipOptions;
        free((*optp).shortName as *mut ::core::ffi::c_void);
        (*optp).shortName = ::core::ptr::null_mut::<::core::ffi::c_char>();
        free((*optp).longName as *mut ::core::ffi::c_void);
        (*optp).longName = ::core::ptr::null_mut::<::core::ffi::c_char>();
        free((*optp).type_0 as *mut ::core::ffi::c_void);
        (*optp).type_0 = ::core::ptr::null_mut::<::core::ffi::c_char>();
        free((*optp).helpString as *mut ::core::ffi::c_void);
        (*optp).helpString = ::core::ptr::null_mut::<::core::ffi::c_char>();
        free((*optp).format as *mut ::core::ffi::c_void);
        (*optp).format = ::core::ptr::null_mut::<::core::ffi::c_char>();
        free((*optp).defaultVal as *mut ::core::ffi::c_void);
        (*optp).defaultVal = ::core::ptr::null_mut::<::core::ffi::c_char>();
        if !(*optp).valuePtr.is_null() {
            j = 0 as ::core::ffi::c_int;
            while j < (*optp).count {
                free(*(*optp).valuePtr.offset(j as isize) as *mut ::core::ffi::c_void);
                let ref mut fresh19 = *(*optp).valuePtr.offset(j as isize);
                *fresh19 = ::core::ptr::null_mut::<::core::ffi::c_char>();
                j += 1;
            }
            free((*optp).valuePtr as *mut ::core::ffi::c_void);
        }
        free((*optp).nextLinked as *mut ::core::ffi::c_void);
        (*optp).nextLinked = ::core::ptr::null_mut::<::core::ffi::c_int>();
        i += 1;
    }
    free(sOptTable as *mut ::core::ffi::c_void);
    sOptTable = ::core::ptr::null_mut::<PipOptions>();
    sOptTable = ::core::ptr::null_mut::<PipOptions>();
    sTableSize = 0 as ::core::ffi::c_int;
    sNumOptions = 0 as ::core::ffi::c_int;
    free(sErrorString as *mut ::core::ffi::c_void);
    sErrorString = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sErrorString = ::core::ptr::null_mut::<::core::ffi::c_char>();
    free(sUsageString as *mut ::core::ffi::c_void);
    sUsageString = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sUsageString = ::core::ptr::null_mut::<::core::ffi::c_char>();
    free(sLinkedOption as *mut ::core::ffi::c_void);
    sLinkedOption = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sLinkedOption = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sNextOption = 0 as ::core::ffi::c_int;
    sNextArgBelongsTo = -(1 as ::core::ffi::c_int);
    sNumOptionArguments = 0 as ::core::ffi::c_int;
    sAllowDefaults = 0 as ::core::ffi::c_int;
    free(sTempStr as *mut ::core::ffi::c_void);
    sTempStr = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sTempStr = ::core::ptr::null_mut::<::core::ffi::c_char>();
    free(sLineStr as *mut ::core::ffi::c_void);
    sLineStr = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sLineStr = ::core::ptr::null_mut::<::core::ffi::c_char>();
    if sProgramName != sNullString {
        free(sProgramName as *mut ::core::ffi::c_void);
    }
    sProgramName = sNullString;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_warn_unused_non_opt_args() -> ::core::ffi::c_int {
    let mut ind: ::core::ffi::c_int = 0;
    let mut unused: ::core::ffi::c_int = (*sOptTable.offset(sNonOptInd as isize)).count
        - 1 as ::core::ffi::c_int
        - sHighestNonOptGotten;
    if unused > 0 as ::core::ffi::c_int {
        printf(
            b"\nWARNING: Extra non-option arguments not used by the program:\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        ind = sHighestNonOptGotten + 1 as ::core::ffi::c_int;
        while ind < (*sOptTable.offset(sNonOptInd as isize)).count {
            printf(
                b"  %s\0" as *const u8 as *const ::core::ffi::c_char,
                *(*sOptTable.offset(sNonOptInd as isize))
                    .valuePtr
                    .offset(ind as isize),
            );
            ind += 1;
        }
        printf(b"\n\n\0" as *const u8 as *const ::core::ffi::c_char);
    }
    return unused;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_exit_on_error(
    mut useStdErr: ::core::ffi::c_int,
    mut prefix: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    sExitPrefix[0 as ::core::ffi::c_int as usize] = 0 as ::core::ffi::c_char;
    if prefix.is_null() || *prefix == 0 {
        return 0 as ::core::ffi::c_int;
    }
    sErrorDest = useStdErr;
    strncpy(
        &raw mut sExitPrefix as *mut ::core::ffi::c_char,
        prefix,
        (PREFIX_SIZE - 1 as ::core::ffi::c_int) as size_t,
    );
    sExitPrefix[(PREFIX_SIZE - 1 as ::core::ffi::c_int) as usize] = 0 as ::core::ffi::c_char;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn setExitPrefix(mut prefix: *const ::core::ffi::c_char) {
    pip_exit_on_error(0 as ::core::ffi::c_int, prefix);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn setStandardExitPrefix(mut progName: *const ::core::ffi::c_char) {
    let mut prefix: *mut ::core::ffi::c_char =
        malloc(strlen(progName).wrapping_add(15 as size_t)) as *mut ::core::ffi::c_char;
    if pip_memory_error(
        prefix as *mut ::core::ffi::c_void,
        b"setStandardExitPrefix\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0
    {
        return;
    }
    sprintf(
        prefix,
        b"\nERROR: %s - \0" as *const u8 as *const ::core::ffi::c_char,
        progName,
    );
    pip_exit_on_error(0 as ::core::ffi::c_int, prefix);
    free(prefix as *mut ::core::ffi::c_void);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_allow_comma_defaults(mut val: ::core::ffi::c_int) {
    sAllowDefaults = val;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_set_manpage_output(mut val: ::core::ffi::c_int) {
    sOutputManpage = val;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_set_usage_string(
    mut usage: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    free(sUsageString as *mut ::core::ffi::c_void);
    sUsageString = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sUsageString = strdup(usage);
    if pip_memory_error(
        sUsageString as *mut ::core::ffi::c_void,
        b"pip_set_usage_string\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0
    {
        return -(1 as ::core::ffi::c_int);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_enable_entry_output(mut val: ::core::ffi::c_int) {
    sPrintEntries = val;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_set_linked_option(
    mut option: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    free(sLinkedOption as *mut ::core::ffi::c_void);
    sLinkedOption = ::core::ptr::null_mut::<::core::ffi::c_char>();
    sLinkedOption = strdup(option);
    if pip_memory_error(
        sLinkedOption as *mut ::core::ffi::c_void,
        b"pip_set_linked_option\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0
    {
        return -(1 as ::core::ffi::c_int);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_set_special_flags(
    mut inCase: ::core::ffi::c_int,
    mut inDone: ::core::ffi::c_int,
    mut inStd: ::core::ffi::c_int,
    mut inLines: ::core::ffi::c_int,
    mut inAbbrevs: ::core::ffi::c_int,
) {
    sNoCase = inCase;
    sDoneEnds = inDone;
    sTakeStdIn = inStd;
    sNonOptLines = inLines;
    sNoAbbrevs = inAbbrevs;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_add_option(
    mut optionString: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut ind: ::core::ffi::c_int = 0;
    let mut indEnd: ::core::ffi::c_int = 0;
    let mut oldSlen: ::core::ffi::c_int = 0;
    let mut newSlen: ::core::ffi::c_int = 0;
    let mut newLlen: ::core::ffi::c_int = 0;
    let mut oldLlen: ::core::ffi::c_int = 0;
    let mut optp: *mut PipOptions = sOptTable.offset(sNextOption as isize) as *mut PipOptions;
    let mut colonPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut oldShort: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut oldLong: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut newShort: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut newLong: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut subStr: *const ::core::ffi::c_char = optionString;
    if sNextOption >= sNumOptions {
        pip_set_error(
            b"Attempting to add more options than were originally specified\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    colonPtr = strchr(subStr, ':' as i32);
    if !colonPtr.is_null() {
        indEnd = colonPtr.offset_from(subStr) as ::core::ffi::c_long as ::core::ffi::c_int;
        if indEnd > 0 as ::core::ffi::c_int {
            (*optp).shortName = pip_sub_str_dup(
                subStr,
                0 as ::core::ffi::c_int,
                indEnd - 1 as ::core::ffi::c_int,
            );
            (*optp).lenShort = indEnd;
        } else {
            (*optp).shortName = strdup(sNullString);
            (*optp).lenShort = 0 as ::core::ffi::c_int;
        }
        if pip_memory_error(
            (*optp).shortName as *mut ::core::ffi::c_void,
            b"pip_add_option\0" as *const u8 as *const ::core::ffi::c_char,
        ) != 0
        {
            return -(1 as ::core::ffi::c_int);
        }
        subStr = subStr.offset((indEnd + 1 as ::core::ffi::c_int) as isize);
        colonPtr = strchr(subStr, ':' as i32);
        if !colonPtr.is_null() {
            if pip_memory_error(
                colonPtr as *mut ::core::ffi::c_void,
                b"pip_add_option\0" as *const u8 as *const ::core::ffi::c_char,
            ) != 0
            {
                return -(1 as ::core::ffi::c_int);
            }
            indEnd = colonPtr.offset_from(subStr) as ::core::ffi::c_long as ::core::ffi::c_int;
            if indEnd > 0 as ::core::ffi::c_int {
                (*optp).longName = pip_sub_str_dup(
                    subStr,
                    0 as ::core::ffi::c_int,
                    indEnd - 1 as ::core::ffi::c_int,
                );
                (*optp).lenLong = indEnd;
            } else {
                (*optp).longName = strdup(sNullString);
                (*optp).lenLong = 0 as ::core::ffi::c_int;
            }
            if pip_memory_error(
                (*optp).longName as *mut ::core::ffi::c_void,
                b"pip_add_option\0" as *const u8 as *const ::core::ffi::c_char,
            ) != 0
            {
                return -(1 as ::core::ffi::c_int);
            }
            subStr = subStr.offset((indEnd + 1 as ::core::ffi::c_int) as isize);
            colonPtr = strchr(subStr, ':' as i32);
            if !colonPtr.is_null() {
                if pip_memory_error(
                    colonPtr as *mut ::core::ffi::c_void,
                    b"pip_add_option\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    return -(1 as ::core::ffi::c_int);
                }
                indEnd = colonPtr.offset_from(subStr) as ::core::ffi::c_long as ::core::ffi::c_int;
                if indEnd > 0 as ::core::ffi::c_int {
                    ind = indEnd - 1 as ::core::ffi::c_int;
                    if *subStr.offset(ind as isize) as ::core::ffi::c_int == 'M' as i32
                        || *subStr.offset(ind as isize) as ::core::ffi::c_int == 'L' as i32
                    {
                        (*optp).multiple = 1 as ::core::ffi::c_int;
                        if *subStr.offset(ind as isize) as ::core::ffi::c_int == 'L' as i32 {
                            (*optp).linked = 1 as ::core::ffi::c_int;
                        }
                        ind -= 1;
                    }
                    (*optp).type_0 = pip_sub_str_dup(subStr, 0 as ::core::ffi::c_int, ind);
                } else {
                    (*optp).type_0 = strdup(sNullString);
                }
                if pip_memory_error(
                    (*optp).type_0 as *mut ::core::ffi::c_void,
                    b"pip_add_option\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    return -(1 as ::core::ffi::c_int);
                }
                subStr = subStr.offset((indEnd + 1 as ::core::ffi::c_int) as isize);
                (*optp).helpString = strdup(subStr);
                if pip_memory_error(
                    (*optp).helpString as *mut ::core::ffi::c_void,
                    b"pip_add_option\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    return -(1 as ::core::ffi::c_int);
                }
                newShort = (*optp).shortName;
                newLong = (*optp).longName;
                newSlen = (*optp).lenShort;
                newLlen = (*optp).lenLong;
                ind = 0 as ::core::ffi::c_int;
                while ind < sTableSize {
                    if !(ind >= sNextOption && ind < sNonOptInd) {
                        oldShort = (*sOptTable.offset(ind as isize)).shortName;
                        oldLong = (*sOptTable.offset(ind as isize)).longName;
                        oldSlen = (*sOptTable.offset(ind as isize)).lenShort;
                        oldLlen = (*sOptTable.offset(ind as isize)).lenLong;
                        if (pip_starts_with(newShort, oldShort) != 0
                            || pip_starts_with(oldShort, newShort) != 0)
                            && (newSlen > 1 as ::core::ffi::c_int
                                && oldSlen > 1 as ::core::ffi::c_int
                                || newSlen == 1 as ::core::ffi::c_int
                                    && oldSlen == 1 as ::core::ffi::c_int)
                            && (sNoAbbrevs == 0 || newSlen == oldSlen)
                            || (pip_starts_with(oldLong, newShort) != 0
                                || pip_starts_with(newShort, oldLong) != 0)
                                && (sNoAbbrevs == 0 || newSlen == oldLlen)
                            || (pip_starts_with(oldShort, newLong) != 0
                                || pip_starts_with(newLong, oldShort) != 0)
                                && (sNoAbbrevs == 0 || oldSlen == newLlen)
                            || (pip_starts_with(oldLong, newLong) != 0
                                || pip_starts_with(newLong, oldLong) != 0)
                                && (sNoAbbrevs == 0 || oldLlen == newSlen)
                        {
                            sprintf(
                                sTempStr,
                                b"Option %s  %s is ambiguous with option %s  %s\0" as *const u8
                                    as *const ::core::ffi::c_char,
                                newShort,
                                newLong,
                                oldShort,
                                oldLong,
                            );
                            pip_set_error(sTempStr);
                            return -(1 as ::core::ffi::c_int);
                        }
                    }
                    ind += 1;
                }
                sNextOption += 1;
                return 0 as ::core::ffi::c_int;
            }
        }
    }
    sprintf(
        sTempStr,
        b"Option does not have three colons in it:  \0" as *const u8 as *const ::core::ffi::c_char,
    );
    append_to_error_string(optionString);
    return -(1 as ::core::ffi::c_int);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_next_arg(
    mut argString: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut argCopy: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut lenarg: ::core::ffi::c_int = 0;
    let mut newNum: ::core::ffi::c_int = 0;
    let mut ifAlloc: ::core::ffi::c_int = 0;
    let mut noMatch: ::core::ffi::c_int = 0;
    let mut newArgs: *mut *mut ::core::ffi::c_char =
        ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
    let mut ch: ::core::ffi::c_char = 0;
    let mut paramFile: *mut FILE = ::core::ptr::null_mut::<FILE>();
    let mut indStart: ::core::ffi::c_int = 0;
    if sNextArgBelongsTo >= 0 as ::core::ffi::c_int {
        argCopy = strdup(argString);
        if pip_memory_error(
            argCopy as *mut ::core::ffi::c_void,
            b"pip_next_arg\0" as *const u8 as *const ::core::ffi::c_char,
        ) != 0
        {
            return -(1 as ::core::ffi::c_int);
        }
        err = add_value_string(sNextArgBelongsTo, argCopy);
        if err == 0
            && strcmp(
                (*sOptTable.offset(sNextArgBelongsTo as isize)).type_0,
                PARAM_FILE_STRING.as_ptr(),
            ) == 0
        {
            paramFile = fopen(argCopy, b"r\0" as *const u8 as *const ::core::ffi::c_char);
            if !paramFile.is_null() {
                err = read_param_file(paramFile);
                fclose(paramFile);
            } else {
                sprintf(
                    sTempStr,
                    b"Error opening parameter file %s\0" as *const u8 as *const ::core::ffi::c_char,
                    argCopy,
                );
                pip_set_error(sTempStr);
                err = -(1 as ::core::ffi::c_int);
            }
        }
        sNextArgBelongsTo = -(1 as ::core::ffi::c_int);
        return err;
    }
    if *argString.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == '-' as i32 {
        indStart = 1 as ::core::ffi::c_int;
        lenarg = strlen(argString) as ::core::ffi::c_int;
        if lenarg > 1 as ::core::ffi::c_int
            && *argString.offset(1 as ::core::ffi::c_int as isize) as ::core::ffi::c_int
                == '-' as i32
        {
            indStart = 2 as ::core::ffi::c_int;
        }
        if lenarg == indStart {
            pip_set_error(
                b"Illegal argument: - or --\0" as *const u8 as *const ::core::ffi::c_char,
            );
            return -(1 as ::core::ffi::c_int);
        }
        if pip_starts_with(
            STANDARD_INPUT_STRING.as_ptr(),
            argString.offset(indStart as isize),
        ) != 0
        {
            err = read_param_file(stdin);
            return err;
        }
        sNotFoundOK = 1 as ::core::ffi::c_int;
        i = indStart;
        while i < lenarg {
            ch = *argString.offset(i as isize);
            if ch as ::core::ffi::c_int != '-' as i32
                && ch as ::core::ffi::c_int != ',' as i32
                && ch as ::core::ffi::c_int != '.' as i32
                && ch as ::core::ffi::c_int != ' ' as i32
                && ((ch as ::core::ffi::c_int) < '0' as i32
                    || ch as ::core::ffi::c_int > '9' as i32)
            {
                sNotFoundOK = 0 as ::core::ffi::c_int;
                break;
            } else {
                i += 1;
            }
        }
        err = lookup_option(argString.offset(indStart as isize), sNextOption);
        if !(sNotFoundOK != 0 && err == LOOKUP_NOT_FOUND) {
            sNotFoundOK = 0 as ::core::ffi::c_int;
            if err < 0 as ::core::ffi::c_int {
                return err;
            }
            sNumOptionArguments += 1;
            if strcmp(
                BOOLEAN_STRING.as_ptr(),
                (*sOptTable.offset(err as isize)).type_0,
            ) != 0
            {
                sNextArgBelongsTo = err;
                return 1 as ::core::ffi::c_int;
            } else {
                argCopy = strdup(b"1\0" as *const u8 as *const ::core::ffi::c_char);
                if pip_memory_error(
                    argCopy as *mut ::core::ffi::c_void,
                    b"pip_next_arg\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    return -(1 as ::core::ffi::c_int);
                }
                return add_value_string(err, argCopy);
            }
        }
        sNotFoundOK = 0 as ::core::ffi::c_int;
    }
    newArgs = expand_arg_list(
        &raw mut argString,
        1 as ::core::ffi::c_int,
        &raw mut newNum,
        &raw mut ifAlloc,
        &raw mut noMatch,
    );
    if newArgs.is_null() {
        pip_set_error(
            b"Memory allocation failed when expanding non-option arguments\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    i = 0 as ::core::ffi::c_int;
    while i < newNum {
        if ifAlloc == 0 {
            argCopy = strdup(argString);
            if pip_memory_error(
                argCopy as *mut ::core::ffi::c_void,
                b"pip_next_arg\0" as *const u8 as *const ::core::ffi::c_char,
            ) != 0
            {
                return -(1 as ::core::ffi::c_int);
            }
        } else {
            argCopy = *newArgs.offset(i as isize);
        }
        err = add_value_string(sNonOptInd, argCopy);
        if err != 0 {
            return err;
        }
        i += 1;
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_number_of_args(
    mut numOptArgs: *mut ::core::ffi::c_int,
    mut numNonOptArgs: *mut ::core::ffi::c_int,
) {
    *numOptArgs = sNumOptionArguments;
    *numNonOptArgs = (*sOptTable.offset(sNonOptInd as isize)).count;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_non_option_arg(
    mut argNo: ::core::ffi::c_int,
    mut arg: *mut *mut ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    if argNo >= (*sOptTable.offset(sNonOptInd as isize)).count {
        pip_set_error(
            b"Requested a non-option argument beyond the number available\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    sHighestNonOptGotten = if sHighestNonOptGotten > argNo {
        sHighestNonOptGotten
    } else {
        argNo
    };
    *arg = strdup(
        *(*sOptTable.offset(sNonOptInd as isize))
            .valuePtr
            .offset(argNo as isize),
    );
    return pip_memory_error(
        *arg as *mut ::core::ffi::c_void,
        b"pip_get_non_option_arg\0" as *const u8 as *const ::core::ffi::c_char,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_string(
    mut option: *const ::core::ffi::c_char,
    mut string: *mut *mut ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    let mut valErr: ::core::ffi::c_int = 0;
    valErr = get_next_value_string(option, &raw mut strPtr);
    if valErr == 0 || valErr == 2 as ::core::ffi::c_int {
        *string = strdup(strPtr);
        err = pip_memory_error(
            *string as *mut ::core::ffi::c_void,
            b"pip_get_string\0" as *const u8 as *const ::core::ffi::c_char,
        );
        if err != 0 {
            return err;
        }
    }
    return valErr;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_boolean(
    mut option: *const ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0;
    err = get_next_value_string(option, &raw mut strPtr);
    if err != 0 {
        return err;
    }
    if strcmp(strPtr, b"1\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"T\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"TRUE\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"ON\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"t\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"true\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"on\0" as *const u8 as *const ::core::ffi::c_char) == 0
    {
        *val = 1 as ::core::ffi::c_int;
    } else if strcmp(strPtr, b"0\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"F\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(
            strPtr,
            b"FALSE\0" as *const u8 as *const ::core::ffi::c_char,
        ) == 0
        || strcmp(strPtr, b"OFF\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(strPtr, b"f\0" as *const u8 as *const ::core::ffi::c_char) == 0
        || strcmp(
            strPtr,
            b"false\0" as *const u8 as *const ::core::ffi::c_char,
        ) == 0
        || strcmp(strPtr, b"off\0" as *const u8 as *const ::core::ffi::c_char) == 0
    {
        *val = 0 as ::core::ffi::c_int;
    } else {
        sprintf(
            sTempStr,
            b"Illegal entry for boolean option %s: %s\0" as *const u8 as *const ::core::ffi::c_char,
            option,
            strPtr,
        );
        pip_set_error(sTempStr);
        return -(1 as ::core::ffi::c_int);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_integer(
    mut option: *const ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut num: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    return pip_get_integer_array(option, val, &raw mut num, 1 as ::core::ffi::c_int);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_float(
    mut option: *const ::core::ffi::c_char,
    mut val: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut num: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    return pip_get_float_array(option, val, &raw mut num, 1 as ::core::ffi::c_int);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_two_integers(
    mut option: *const ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_int,
    mut val2: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut num: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
    let mut tmp: [::core::ffi::c_int; 2] = [0; 2];
    tmp[0 as ::core::ffi::c_int as usize] = *val1;
    tmp[1 as ::core::ffi::c_int as usize] = *val2;
    err = pip_get_integer_array(
        option,
        &raw mut tmp as *mut ::core::ffi::c_int,
        &raw mut num,
        2 as ::core::ffi::c_int,
    );
    if err != 0 as ::core::ffi::c_int {
        return err;
    }
    *val1 = tmp[0 as ::core::ffi::c_int as usize];
    *val2 = tmp[1 as ::core::ffi::c_int as usize];
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_two_floats(
    mut option: *const ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_float,
    mut val2: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut num: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
    let mut tmp: [::core::ffi::c_float; 2] = [0.; 2];
    tmp[0 as ::core::ffi::c_int as usize] = *val1;
    tmp[1 as ::core::ffi::c_int as usize] = *val2;
    err = pip_get_float_array(
        option,
        &raw mut tmp as *mut ::core::ffi::c_float,
        &raw mut num,
        2 as ::core::ffi::c_int,
    );
    if err != 0 as ::core::ffi::c_int {
        return err;
    }
    *val1 = tmp[0 as ::core::ffi::c_int as usize];
    *val2 = tmp[1 as ::core::ffi::c_int as usize];
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_three_integers(
    mut option: *const ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_int,
    mut val2: *mut ::core::ffi::c_int,
    mut val3: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut num: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
    let mut tmp: [::core::ffi::c_int; 3] = [0; 3];
    tmp[0 as ::core::ffi::c_int as usize] = *val1;
    tmp[1 as ::core::ffi::c_int as usize] = *val2;
    tmp[2 as ::core::ffi::c_int as usize] = *val3;
    err = pip_get_integer_array(
        option,
        &raw mut tmp as *mut ::core::ffi::c_int,
        &raw mut num,
        3 as ::core::ffi::c_int,
    );
    if err != 0 as ::core::ffi::c_int {
        return err;
    }
    *val1 = tmp[0 as ::core::ffi::c_int as usize];
    *val2 = tmp[1 as ::core::ffi::c_int as usize];
    *val3 = tmp[2 as ::core::ffi::c_int as usize];
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_three_floats(
    mut option: *const ::core::ffi::c_char,
    mut val1: *mut ::core::ffi::c_float,
    mut val2: *mut ::core::ffi::c_float,
    mut val3: *mut ::core::ffi::c_float,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut num: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
    let mut tmp: [::core::ffi::c_float; 3] = [0.; 3];
    tmp[0 as ::core::ffi::c_int as usize] = *val1;
    tmp[1 as ::core::ffi::c_int as usize] = *val2;
    tmp[2 as ::core::ffi::c_int as usize] = *val3;
    err = pip_get_float_array(
        option,
        &raw mut tmp as *mut ::core::ffi::c_float,
        &raw mut num,
        3 as ::core::ffi::c_int,
    );
    if err != 0 as ::core::ffi::c_int {
        return err;
    }
    *val1 = tmp[0 as ::core::ffi::c_int as usize];
    *val2 = tmp[1 as ::core::ffi::c_int as usize];
    *val3 = tmp[2 as ::core::ffi::c_int as usize];
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_integer_array(
    mut option: *const ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_int,
    mut numToGet: *mut ::core::ffi::c_int,
    mut arraySize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return option_line_of_values(
        option,
        array as *mut ::core::ffi::c_void,
        PIP_INTEGER,
        numToGet,
        arraySize,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_float_array(
    mut option: *const ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_float,
    mut numToGet: *mut ::core::ffi::c_int,
    mut arraySize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return option_line_of_values(
        option,
        array as *mut ::core::ffi::c_void,
        PIP_FLOAT,
        numToGet,
        arraySize,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_print_help(
    mut progName: *const ::core::ffi::c_char,
    mut useStdErr: ::core::ffi::c_int,
    mut inputFiles: ::core::ffi::c_int,
    mut outputFiles: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    let mut abbrevOK: ::core::ffi::c_int = 0;
    let mut lastOpt: ::core::ffi::c_int = 0;
    let mut jlim: ::core::ffi::c_int = 0;
    let mut optLen: ::core::ffi::c_int = 0;
    let mut hasbf: ::core::ffi::c_int = 0;
    let mut numOut: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut numReal: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut brokeAtSpace: ::core::ffi::c_int = 0;
    let mut brokeAtNewLine: ::core::ffi::c_int = 0;
    let mut helplim: ::core::ffi::c_int = 74 as ::core::ffi::c_int;
    let mut sname: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut lname: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut newLinePt: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut defPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut out: *mut FILE = if useStdErr != 0 { stderr } else { stdout };
    let mut indent4: [::core::ffi::c_char; 5] =
        ::core::mem::transmute::<[u8; 5], [::core::ffi::c_char; 5]>(*b"    \0");
    let mut indentStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut linePos: ::core::ffi::c_int = 11 as ::core::ffi::c_int;
    let mut fort90: ::core::ffi::c_int = if sOutputManpage == -(3 as ::core::ffi::c_int) {
        1 as ::core::ffi::c_int
    } else {
        0 as ::core::ffi::c_int
    };
    let mut fort77: ::core::ffi::c_int = if sOutputManpage == -(2 as ::core::ffi::c_int) {
        1 as ::core::ffi::c_int
    } else {
        0 as ::core::ffi::c_int
    };
    let mut cCode: ::core::ffi::c_int = if sOutputManpage == 2 as ::core::ffi::c_int {
        1 as ::core::ffi::c_int
    } else {
        0 as ::core::ffi::c_int
    };
    let mut python: ::core::ffi::c_int = if sOutputManpage == 3 as ::core::ffi::c_int {
        1 as ::core::ffi::c_int
    } else {
        0 as ::core::ffi::c_int
    };
    let mut fortCont: *mut ::core::ffi::c_char = (if fort90 != 0 {
        b" &\n      '\0" as *const u8 as *const ::core::ffi::c_char
    } else {
        b"\n     &    '\0" as *const u8 as *const ::core::ffi::c_char
    }) as *mut ::core::ffi::c_char;
    let mut descriptions: *mut *mut ::core::ffi::c_char =
        (&raw mut sTypeDescriptions as *mut *mut ::core::ffi::c_char)
            .offset(0 as ::core::ffi::c_int as isize) as *mut *mut ::core::ffi::c_char;
    i = 0 as ::core::ffi::c_int;
    while i < sNumOptions {
        sname = (*sOptTable.offset(i as isize)).shortName;
        lname = (*sOptTable.offset(i as isize)).longName;
        if !lname.is_null() && *lname as ::core::ffi::c_int != 0
            || !sname.is_null() && *sname as ::core::ffi::c_int != 0
        {
            numReal += 1;
        }
        i += 1;
    }
    if sOutputManpage == 0 {
        if !sUsageString.is_null() {
            fprintf(
                out,
                b"%s\0" as *const u8 as *const ::core::ffi::c_char,
                sUsageString,
            );
        } else {
            fprintf(
                out,
                b"Usage: %s \0" as *const u8 as *const ::core::ffi::c_char,
                progName,
            );
            if sNumOptions != 0 {
                fprintf(
                    out,
                    b"[Options]\0" as *const u8 as *const ::core::ffi::c_char,
                );
            }
            if inputFiles != 0 {
                fprintf(
                    out,
                    b" input_file\0" as *const u8 as *const ::core::ffi::c_char,
                );
            }
            if inputFiles > 1 as ::core::ffi::c_int {
                fprintf(out, b"s...\0" as *const u8 as *const ::core::ffi::c_char);
            }
            if outputFiles != 0 {
                fprintf(
                    out,
                    b" output_file\0" as *const u8 as *const ::core::ffi::c_char,
                );
            }
            if outputFiles > 1 as ::core::ffi::c_int {
                fprintf(out, b"s...\0" as *const u8 as *const ::core::ffi::c_char);
            }
        }
        fprintf(out, b"\n\0" as *const u8 as *const ::core::ffi::c_char);
        if numReal == 0 {
            return 0 as ::core::ffi::c_int;
        }
        if sNoHelpAbbrevs == 0 {
            fprintf(
                out,
                b"Options can be abbreviated, current short name abbreviations are in parentheses\n\0"
                    as *const u8 as *const ::core::ffi::c_char,
            );
        }
        fprintf(
            out,
            b"Options:\n\0" as *const u8 as *const ::core::ffi::c_char,
        );
        descriptions = (&raw mut sTypeForUsage as *mut *mut ::core::ffi::c_char)
            .offset(0 as ::core::ffi::c_int as isize)
            as *mut *mut ::core::ffi::c_char;
    }
    sTestAbbrevForUsage = 1 as ::core::ffi::c_int;
    let mut current_block_166: u64;
    i = 0 as ::core::ffi::c_int;
    while i < sNumOptions {
        sname = (*sOptTable.offset(i as isize)).shortName;
        lname = (*sOptTable.offset(i as isize)).longName;
        indentStr = sNullString;
        abbrevOK = 0 as ::core::ffi::c_int;
        if !sname.is_null() && *sname as ::core::ffi::c_int != 0 && sNoHelpAbbrevs == 0 {
            jlim = strlen(sname).wrapping_sub(1 as size_t) as ::core::ffi::c_int;
            if jlim > TEMP_STR_SIZE - 10 as ::core::ffi::c_int {
                jlim = TEMP_STR_SIZE - 10 as ::core::ffi::c_int;
            }
            j = 0 as ::core::ffi::c_int;
            while j < jlim {
                *sTempStr.offset(j as isize) = *sname.offset(j as isize);
                *sTempStr.offset((j + 1 as ::core::ffi::c_int) as isize) = 0 as ::core::ffi::c_char;
                if lookup_option(sTempStr, sNumOptions) == i {
                    abbrevOK = 1 as ::core::ffi::c_int;
                    break;
                } else {
                    j += 1;
                }
            }
        }
        if !lname.is_null() && *lname as ::core::ffi::c_int != 0
            || !sname.is_null() && *sname as ::core::ffi::c_int != 0
        {
            if sOutputManpage <= 0 as ::core::ffi::c_int && fort90 == 0 {
                indentStr = &raw mut indent4 as *mut ::core::ffi::c_char;
            }
            if fort77 != 0 || fort90 != 0 {
                lastOpt = (i == sNumOptions - 1 as ::core::ffi::c_int) as ::core::ffi::c_int;
                if numOut == 0 {
                    fprintf(
                        out,
                        b"%s  integer numOptions\n%s  parameter (numOptions = %d)\n%s  character*(40 * numOptions) options(1)\n%s  options(1) =%s\0"
                            as *const u8 as *const ::core::ffi::c_char,
                        indentStr,
                        indentStr,
                        numReal,
                        indentStr,
                        indentStr,
                        fortCont,
                    );
                }
                optLen = strlen(sname)
                    .wrapping_add(strlen(lname))
                    .wrapping_add(strlen((*sOptTable.offset(i as isize)).type_0))
                    .wrapping_add(4 as size_t)
                    .wrapping_add((*sOptTable.offset(i as isize)).multiple as size_t)
                    as ::core::ffi::c_int;
                if linePos
                    + optLen
                    + (if lastOpt != 0 {
                        0 as ::core::ffi::c_int
                    } else {
                        (if fort90 != 0 {
                            5 as ::core::ffi::c_int
                        } else {
                            3 as ::core::ffi::c_int
                        })
                    })
                    > 90 as ::core::ffi::c_int
                {
                    fprintf(
                        out,
                        b"'//%s\0" as *const u8 as *const ::core::ffi::c_char,
                        fortCont,
                    );
                    linePos = if fort90 != 0 {
                        7 as ::core::ffi::c_int
                    } else {
                        11 as ::core::ffi::c_int
                    };
                }
                fprintf(
                    out,
                    b"%s:%s:%s%s%s\0" as *const u8 as *const ::core::ffi::c_char,
                    sname,
                    lname,
                    (*sOptTable.offset(i as isize)).type_0,
                    if (*sOptTable.offset(i as isize)).multiple != 0 {
                        if (*sOptTable.offset(i as isize)).linked != 0 {
                            b"L:\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"M:\0" as *const u8 as *const ::core::ffi::c_char
                        }
                    } else {
                        b":\0" as *const u8 as *const ::core::ffi::c_char
                    },
                    if lastOpt != 0 {
                        b"'\n\0" as *const u8 as *const ::core::ffi::c_char
                    } else {
                        b"@\0" as *const u8 as *const ::core::ffi::c_char
                    },
                );
                linePos += optLen;
                numOut += 1;
                current_block_166 = 2569451025026770673;
            } else if cCode != 0 || python != 0 {
                lastOpt = (i == sNumOptions - 1 as ::core::ffi::c_int) as ::core::ffi::c_int;
                if numOut == 0 {
                    if cCode != 0 {
                        fprintf(
                            out,
                            b"  int numOptions = %d;\n  const char *options[] = {\n    \0"
                                as *const u8
                                as *const ::core::ffi::c_char,
                            numReal,
                        );
                        linePos = 5 as ::core::ffi::c_int;
                    } else {
                        fprintf(
                            out,
                            b"options = [\0" as *const u8 as *const ::core::ffi::c_char,
                        );
                        linePos = 12 as ::core::ffi::c_int;
                    }
                }
                optLen = strlen(sname)
                    .wrapping_add(strlen(lname))
                    .wrapping_add(strlen((*sOptTable.offset(i as isize)).type_0))
                    .wrapping_add(7 as size_t)
                    .wrapping_add((*sOptTable.offset(i as isize)).multiple as size_t)
                    as ::core::ffi::c_int;
                if linePos + optLen > 90 as ::core::ffi::c_int {
                    if cCode != 0 {
                        fprintf(out, b"\n    \0" as *const u8 as *const ::core::ffi::c_char);
                        linePos = 5 as ::core::ffi::c_int;
                    } else {
                        fprintf(
                            out,
                            b"\n           \0" as *const u8 as *const ::core::ffi::c_char,
                        );
                        linePos = 12 as ::core::ffi::c_int;
                    }
                }
                fprintf(
                    out,
                    b"\"%s:%s:%s%s\"%s\0" as *const u8 as *const ::core::ffi::c_char,
                    sname,
                    lname,
                    (*sOptTable.offset(i as isize)).type_0,
                    if (*sOptTable.offset(i as isize)).multiple != 0 {
                        if (*sOptTable.offset(i as isize)).linked != 0 {
                            b"L:\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"M:\0" as *const u8 as *const ::core::ffi::c_char
                        }
                    } else {
                        b":\0" as *const u8 as *const ::core::ffi::c_char
                    },
                    if lastOpt != 0 {
                        if cCode != 0 {
                            b"};\n\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"]\n\0" as *const u8 as *const ::core::ffi::c_char
                        }
                    } else {
                        b", \0" as *const u8 as *const ::core::ffi::c_char
                    },
                );
                linePos += optLen;
                numOut += 1;
                current_block_166 = 2569451025026770673;
            } else {
                if i != 0 && sOutputManpage < 0 as ::core::ffi::c_int {
                    fprintf(out, b"\n\0" as *const u8 as *const ::core::ffi::c_char);
                }
                if sOutputManpage > 0 as ::core::ffi::c_int {
                    fprintf(
                        out,
                        b".TP\n.B \0" as *const u8 as *const ::core::ffi::c_char,
                    );
                }
                fprintf(out, b" \0" as *const u8 as *const ::core::ffi::c_char);
                if !sname.is_null() && *sname as ::core::ffi::c_int != 0 {
                    fprintf(
                        out,
                        b"%s-%s\0" as *const u8 as *const ::core::ffi::c_char,
                        if sDoubleDashOptions != 0 {
                            b"-\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"\0" as *const u8 as *const ::core::ffi::c_char
                        },
                        sname,
                    );
                }
                if abbrevOK != 0 {
                    fprintf(
                        out,
                        b" (%s-%s)\0" as *const u8 as *const ::core::ffi::c_char,
                        if sDoubleDashOptions != 0 {
                            b"-\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"\0" as *const u8 as *const ::core::ffi::c_char
                        },
                        sTempStr,
                    );
                }
                if !sname.is_null()
                    && *sname as ::core::ffi::c_int != 0
                    && !lname.is_null()
                    && *lname as ::core::ffi::c_int != 0
                {
                    fprintf(
                        out,
                        b"  %sOR%s  \0" as *const u8 as *const ::core::ffi::c_char,
                        if sOutputManpage > 0 as ::core::ffi::c_int {
                            b"\\fR\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"\0" as *const u8 as *const ::core::ffi::c_char
                        },
                        if sOutputManpage > 0 as ::core::ffi::c_int {
                            b"\\fP\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"\0" as *const u8 as *const ::core::ffi::c_char
                        },
                    );
                }
                if !lname.is_null() && *lname as ::core::ffi::c_int != 0 {
                    fprintf(
                        out,
                        b"%s-%s\0" as *const u8 as *const ::core::ffi::c_char,
                        if sDoubleDashOptions != 0 {
                            b"-\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"\0" as *const u8 as *const ::core::ffi::c_char
                        },
                        lname,
                    );
                }
                j = 0 as ::core::ffi::c_int;
                while j < sNumTypes as ::core::ffi::c_int {
                    if strcmp((*sOptTable.offset(i as isize)).type_0, sTypes[j as usize]) == 0 {
                        break;
                    }
                    j += 1;
                }
                if !(*sOptTable.offset(i as isize)).format.is_null() {
                    if sOutputManpage > 0 as ::core::ffi::c_int {
                        hasbf = pip_starts_with(
                            (*sOptTable.offset(i as isize)).format,
                            b"\\f\0" as *const u8 as *const ::core::ffi::c_char,
                        );
                        fprintf(
                            out,
                            b" \t %s%s%s\0" as *const u8 as *const ::core::ffi::c_char,
                            if hasbf == 0 as ::core::ffi::c_int {
                                b"\\fI\0" as *const u8 as *const ::core::ffi::c_char
                            } else {
                                b"\0" as *const u8 as *const ::core::ffi::c_char
                            },
                            (*sOptTable.offset(i as isize)).format,
                            if hasbf == 0 as ::core::ffi::c_int {
                                b"\\fR\0" as *const u8 as *const ::core::ffi::c_char
                            } else {
                                b"\0" as *const u8 as *const ::core::ffi::c_char
                            },
                        );
                    } else {
                        optLen =
                            strlen((*sOptTable.offset(i as isize)).format) as ::core::ffi::c_int;
                        fprintf(out, b"   \0" as *const u8 as *const ::core::ffi::c_char);
                        j = 0 as ::core::ffi::c_int;
                        while j < optLen {
                            if pip_starts_with(
                                (*sOptTable.offset(i as isize)).format.offset(j as isize),
                                b"\\f\0" as *const u8 as *const ::core::ffi::c_char,
                            ) != 0
                            {
                                j += 2 as ::core::ffi::c_int;
                            } else {
                                fprintf(
                                    out,
                                    b"%c\0" as *const u8 as *const ::core::ffi::c_char,
                                    *(*sOptTable.offset(i as isize)).format.offset(j as isize)
                                        as ::core::ffi::c_int,
                                );
                            }
                            j += 1;
                        }
                    }
                } else if strcmp(
                    (*sOptTable.offset(i as isize)).type_0,
                    BOOLEAN_STRING.as_ptr(),
                ) != 0
                {
                    fprintf(
                        out,
                        b"%s%s%s\0" as *const u8 as *const ::core::ffi::c_char,
                        if sOutputManpage > 0 as ::core::ffi::c_int {
                            b" \t \\fI\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"   \0" as *const u8 as *const ::core::ffi::c_char
                        },
                        *descriptions.offset(j as isize),
                        if sOutputManpage > 0 as ::core::ffi::c_int {
                            b"\\fR\0" as *const u8 as *const ::core::ffi::c_char
                        } else {
                            b"\0" as *const u8 as *const ::core::ffi::c_char
                        },
                    );
                }
                fprintf(out, b"\n\0" as *const u8 as *const ::core::ffi::c_char);
                current_block_166 = 11865390570819897086;
            }
        } else if fort77 != 0 || fort90 != 0 || cCode != 0 || python != 0 {
            current_block_166 = 2569451025026770673;
        } else {
            if sOutputManpage == 1 as ::core::ffi::c_int {
                fprintf(out, b".SS \0" as *const u8 as *const ::core::ffi::c_char);
            } else {
                fprintf(out, b"\n\0" as *const u8 as *const ::core::ffi::c_char);
            }
            current_block_166 = 11865390570819897086;
        }
        match current_block_166 {
            11865390570819897086 => {
                if !(*sOptTable.offset(i as isize)).helpString.is_null()
                    && *(*sOptTable.offset(i as isize)).helpString as ::core::ffi::c_int != 0
                {
                    lname = strdup((*sOptTable.offset(i as isize)).helpString);
                    if pip_memory_error(
                        lname as *mut ::core::ffi::c_void,
                        b"pip_print_help\0" as *const u8 as *const ::core::ffi::c_char,
                    ) != 0
                    {
                        return -(1 as ::core::ffi::c_int);
                    }
                    if !(*sOptTable.offset(i as isize)).defaultVal.is_null() {
                        while strlen(lname)
                            .wrapping_add(strlen((*sOptTable.offset(i as isize)).defaultVal))
                            < (LINE_STR_SIZE - 10 as ::core::ffi::c_int) as size_t
                        {
                            defPtr = strstr(lname, DEFAULT_SUB_STR.as_ptr());
                            if defPtr.is_null() {
                                break;
                            }
                            strncpy(
                                sLineStr,
                                lname,
                                defPtr.offset_from(lname) as ::core::ffi::c_long as size_t,
                            );
                            sprintf(
                                sLineStr.offset(
                                    defPtr.offset_from(lname) as ::core::ffi::c_long as isize
                                ) as *mut ::core::ffi::c_char,
                                b"%s%s\0" as *const u8 as *const ::core::ffi::c_char,
                                (*sOptTable.offset(i as isize)).defaultVal,
                                defPtr.offset(strlen(DEFAULT_SUB_STR.as_ptr()) as isize),
                            );
                            free(lname as *mut ::core::ffi::c_void);
                            lname = strdup(sLineStr);
                            if pip_memory_error(
                                lname as *mut ::core::ffi::c_void,
                                b"pip_print_help\0" as *const u8 as *const ::core::ffi::c_char,
                            ) != 0
                            {
                                return -(1 as ::core::ffi::c_int);
                            }
                        }
                    }
                    sname = lname;
                    optLen = strlen(sname) as ::core::ffi::c_int;
                    newLinePt = strchr(sname, '\n' as i32);
                    while optLen > helplim || !newLinePt.is_null() {
                        brokeAtNewLine = 0 as ::core::ffi::c_int;
                        if !newLinePt.is_null()
                            && newLinePt.offset_from(sname) as ::core::ffi::c_long
                                <= helplim as ::core::ffi::c_long
                        {
                            j = newLinePt.offset_from(sname) as ::core::ffi::c_long
                                as ::core::ffi::c_int;
                            newLinePt = strchr(
                                sname
                                    .offset(j as isize)
                                    .offset(1 as ::core::ffi::c_int as isize),
                                '\n' as i32,
                            );
                            brokeAtSpace = 0 as ::core::ffi::c_int;
                            brokeAtNewLine = 1 as ::core::ffi::c_int;
                        } else {
                            j = helplim;
                            while j >= 1 as ::core::ffi::c_int {
                                if *sname.offset(j as isize) as ::core::ffi::c_int == ' ' as i32 {
                                    break;
                                }
                                j -= 1;
                            }
                            brokeAtSpace = 1 as ::core::ffi::c_int;
                        }
                        if sOutputManpage > 0 as ::core::ffi::c_int
                            && (*sname.offset(0 as ::core::ffi::c_int as isize)
                                as ::core::ffi::c_int
                                == '.' as i32
                                || *sname.offset(0 as ::core::ffi::c_int as isize)
                                    as ::core::ffi::c_int
                                    == '\'' as i32)
                        {
                            fprintf(out, b"\\&\0" as *const u8 as *const ::core::ffi::c_char);
                        }
                        *sname.offset(j as isize) = 0 as ::core::ffi::c_char;
                        fprintf(
                            out,
                            b"%s%s\n\0" as *const u8 as *const ::core::ffi::c_char,
                            indentStr,
                            sname,
                        );
                        if sOutputManpage == 1 as ::core::ffi::c_int
                            && brokeAtNewLine != 0
                            && *sname.offset((j + 1 as ::core::ffi::c_int) as isize)
                                as ::core::ffi::c_int
                                != ' ' as i32
                        {
                            fprintf(out, b".br\n\0" as *const u8 as *const ::core::ffi::c_char);
                        }
                        if brokeAtSpace != 0 && sOutputManpage > 0 as ::core::ffi::c_int {
                            while *sname.offset((j + 1 as ::core::ffi::c_int) as isize)
                                as ::core::ffi::c_int
                                == ' ' as i32
                            {
                                j += 1;
                            }
                        }
                        sname = sname.offset((j + 1 as ::core::ffi::c_int) as isize);
                        optLen -= j + 1 as ::core::ffi::c_int;
                    }
                    fprintf(
                        out,
                        b"%s%s\n\0" as *const u8 as *const ::core::ffi::c_char,
                        indentStr,
                        sname,
                    );
                    free(lname as *mut ::core::ffi::c_void);
                }
                if (*sOptTable.offset(i as isize)).linked != 0 {
                    fprintf(
                        out,
                        b"%s(Multiple entries linked to a different option)\n\0" as *const u8
                            as *const ::core::ffi::c_char,
                        indentStr,
                    );
                } else if (*sOptTable.offset(i as isize)).multiple != 0 {
                    fprintf(
                        out,
                        b"%s(Successive entries accumulate)\n\0" as *const u8
                            as *const ::core::ffi::c_char,
                        indentStr,
                    );
                }
            }
            _ => {}
        }
        i += 1;
    }
    sTestAbbrevForUsage = 0 as ::core::ffi::c_int;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_print_entries() {
    let mut name: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut sname: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut lname: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut i: ::core::ffi::c_int = 0;
    let mut j: ::core::ffi::c_int = 0;
    if sPrintEntries < 0 as ::core::ffi::c_int {
        sPrintEntries = 0 as ::core::ffi::c_int;
        name = getenv(PRINTENTRY_VARIABLE.as_ptr());
        if !name.is_null() {
            sPrintEntries = atoi(name);
        }
    }
    if sPrintEntries == 0 {
        return;
    }
    j = 0 as ::core::ffi::c_int;
    i = 0 as ::core::ffi::c_int;
    while i < sNumOptions {
        j += (*sOptTable.offset(i as isize)).count;
        i += 1;
    }
    if j + (*sOptTable.offset(sNonOptInd as isize)).count == 0 {
        return;
    }
    printf(
        b"\n*** Entries to program %s ***\n\0" as *const u8 as *const ::core::ffi::c_char,
        sProgramName,
    );
    i = 0 as ::core::ffi::c_int;
    while i < sNumOptions {
        sname = (*sOptTable.offset(i as isize)).shortName;
        lname = (*sOptTable.offset(i as isize)).longName;
        if !lname.is_null() && *lname as ::core::ffi::c_int != 0
            || !sname.is_null() && *sname as ::core::ffi::c_int != 0
        {
            name = if !lname.is_null() && *lname as ::core::ffi::c_int != 0 {
                lname
            } else {
                sname
            };
            j = 0 as ::core::ffi::c_int;
            while j < (*sOptTable.offset(i as isize)).count {
                printf(
                    b"  %s = %s\n\0" as *const u8 as *const ::core::ffi::c_char,
                    name,
                    *(*sOptTable.offset(i as isize)).valuePtr.offset(j as isize),
                );
                j += 1;
            }
        }
        i += 1;
    }
    if (*sOptTable.offset(sNonOptInd as isize)).count != 0 {
        printf(b"  Non-option arguments:\0" as *const u8 as *const ::core::ffi::c_char);
        j = 0 as ::core::ffi::c_int;
        while j < (*sOptTable.offset(sNonOptInd as isize)).count {
            if !strchr(
                *(*sOptTable.offset(sNonOptInd as isize))
                    .valuePtr
                    .offset(j as isize),
                ' ' as i32,
            )
            .is_null()
            {
                printf(
                    b"   \"%s\"\0" as *const u8 as *const ::core::ffi::c_char,
                    *(*sOptTable.offset(sNonOptInd as isize))
                        .valuePtr
                        .offset(j as isize),
                );
            } else {
                printf(
                    b"   %s\0" as *const u8 as *const ::core::ffi::c_char,
                    *(*sOptTable.offset(sNonOptInd as isize))
                        .valuePtr
                        .offset(j as isize),
                );
            }
            j += 1;
        }
        printf(b"\n\0" as *const u8 as *const ::core::ffi::c_char);
    }
    printf(b"*** End of entries ***\n\n\0" as *const u8 as *const ::core::ffi::c_char);
    fflush(stdout);
    fflush(stdout);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_error(
    mut errString: *mut *mut ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    if sErrorString.is_null() {
        *errString = strdup(sNullString);
        pip_memory_error(
            *errString as *mut ::core::ffi::c_void,
            b"pip_get_error\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    *errString = strdup(sErrorString);
    return pip_memory_error(
        *errString as *mut ::core::ffi::c_void,
        b"pip_get_error\0" as *const u8 as *const ::core::ffi::c_char,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_set_error(
    mut errString: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut outFile: *mut FILE = if sErrorDest != 0 { stderr } else { stdout };
    if !sErrorString.is_null() {
        free(sErrorString as *mut ::core::ffi::c_void);
    }
    sErrorString = strdup(errString);
    if sErrorString.is_null() && sExitPrefix[0 as ::core::ffi::c_int as usize] == 0 {
        return -(1 as ::core::ffi::c_int);
    }
    if sExitPrefix[0 as ::core::ffi::c_int as usize] != 0 {
        fprintf(
            outFile,
            b"%s \0" as *const u8 as *const ::core::ffi::c_char,
            &raw mut sExitPrefix as *mut ::core::ffi::c_char,
        );
        fprintf(
            outFile,
            b"%s\n\0" as *const u8 as *const ::core::ffi::c_char,
            if !sErrorString.is_null() {
                sErrorString as *const ::core::ffi::c_char
            } else {
                b"Unspecified error\0" as *const u8 as *const ::core::ffi::c_char
            },
        );
        exit(1 as ::core::ffi::c_int);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn exit_error(format: *const ::core::ffi::c_char) {
    pip_set_error(format);
    exit(1 as ::core::ffi::c_int);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_number_of_entries(
    mut option: *const ::core::ffi::c_char,
    mut numEntries: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    err = lookup_option(option, sNonOptInd + 1 as ::core::ffi::c_int);
    if err < 0 as ::core::ffi::c_int {
        return err;
    }
    *numEntries = (*sOptTable.offset(err as isize)).count;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_linked_index(
    mut option: *const ::core::ffi::c_char,
    mut index: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut ilink: ::core::ffi::c_int = 0;
    let mut which: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    err = lookup_option(option, sNonOptInd + 1 as ::core::ffi::c_int);
    if err < 0 as ::core::ffi::c_int {
        return err;
    }
    if (*sOptTable.offset(err as isize)).linked == 0 {
        sprintf(
            sTempStr,
            b"Trying to get a linked index for option %s, which is not identified as linked\0"
                as *const u8 as *const ::core::ffi::c_char,
            option,
        );
        pip_set_error(sTempStr);
        return -(1 as ::core::ffi::c_int);
    }
    ind = 0 as ::core::ffi::c_int;
    if (*sOptTable.offset(err as isize)).multiple != 0 {
        ind = (*sOptTable.offset(err as isize)).multiple - 1 as ::core::ffi::c_int;
    }
    if !sLinkedOption.is_null() {
        ilink = lookup_option(sLinkedOption, sNumOptions);
        if ilink < 0 as ::core::ffi::c_int {
            return ilink;
        }
        if (*sOptTable.offset(ilink as isize)).count != 0 {
            which = 1 as ::core::ffi::c_int;
        }
    }
    *index = *(*sOptTable.offset(err as isize))
        .nextLinked
        .offset((2 as ::core::ffi::c_int * ind + which) as isize);
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_parse_input(
    mut argc: ::core::ffi::c_int,
    mut argv: *mut *mut ::core::ffi::c_char,
    mut options: *mut *const ::core::ffi::c_char,
    mut numOpts: ::core::ffi::c_int,
    mut numOptArgs: *mut ::core::ffi::c_int,
    mut numNonOptArgs: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut err: ::core::ffi::c_int = 0;
    err = pip_initialize(numOpts);
    if err != 0 {
        return err;
    }
    i = 0 as ::core::ffi::c_int;
    while i < numOpts {
        err = pip_add_option(*options.offset(i as isize));
        if err != 0 {
            return err;
        }
        i += 1;
    }
    return pip_parse_entries(argc, argv, numOptArgs, numNonOptArgs);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_read_option_file(
    mut progName: *const ::core::ffi::c_char,
    mut helpLevel: ::core::ffi::c_int,
    mut localDir: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut ind: ::core::ffi::c_int = 0;
    let mut len: ::core::ffi::c_int = 0;
    let mut indst: ::core::ffi::c_int = 0;
    let mut lineLen: ::core::ffi::c_int = 0;
    let mut err: ::core::ffi::c_int = 0;
    let mut needSize: ::core::ffi::c_int = 0;
    let mut isOption: ::core::ffi::c_int = 0;
    let mut isSection: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut optFile: *mut FILE = ::core::ptr::null_mut::<FILE>();
    let mut bigStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut pipDir: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut textStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut helpStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut formatStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut defaultStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut numOpts: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut bigSize: ::core::ffi::c_int = ADOC_STR_SIZE;
    let mut readingOpt: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut longName: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut shortName: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut type_0: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut usageStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut tipStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut manStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut gotLong: ::core::ffi::c_int = 0;
    let mut gotShort: ::core::ffi::c_int = 0;
    let mut gotType: ::core::ffi::c_int = 0;
    let mut gotUsage: ::core::ffi::c_int = 0;
    let mut gotTip: ::core::ffi::c_int = 0;
    let mut gotMan: ::core::ffi::c_int = 0;
    let mut gotFormat: ::core::ffi::c_int = 0;
    let mut gotDefault: ::core::ffi::c_int = 0;
    let mut gotDelim: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut optStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut optStrSize: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut lastGottenStr: *mut *mut ::core::ffi::c_char =
        ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
    let mut inQuoteIndex: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
    if bigSize < PATH_MAX {
        bigSize = PATH_MAX;
    }
    if sTempStr.is_null() {
        sTempStr = malloc(TEMP_STR_SIZE as size_t) as *mut ::core::ffi::c_char;
    }
    bigStr = malloc(bigSize as size_t) as *mut ::core::ffi::c_char;
    if bigStr.is_null() || sTempStr.is_null() {
        pip_memory_error(
            NULL,
            b"pip_read_option_file\0" as *const u8 as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    sProgramName = strdup(progName);
    if sProgramName.is_null() {
        sProgramName = sNullString;
    }
    if localDir == 0 {
        pipDir = getenv(OPTDIR_VARIABLE.as_ptr());
        if !pipDir.is_null() {
            if strlen(pipDir) > (bigSize - 100 as ::core::ffi::c_int) as size_t {
                pip_set_error(
                    b"AUTODOC_DIR is suspiciously long\0" as *const u8
                        as *const ::core::ffi::c_char,
                );
                return -(1 as ::core::ffi::c_int);
            }
            sprintf(
                bigStr,
                b"%s%c%s.%s\0" as *const u8 as *const ::core::ffi::c_char,
                pipDir,
                PATH_SEPARATOR,
                progName,
                OPTFILE_EXT.as_ptr(),
            );
            optFile = fopen(bigStr, b"r\0" as *const u8 as *const ::core::ffi::c_char);
        }
        if optFile.is_null() {
            pipDir = getenv(b"IMOD_DIR\0" as *const u8 as *const ::core::ffi::c_char);
            if !pipDir.is_null() {
                if strlen(pipDir) > (bigSize - 100 as ::core::ffi::c_int) as size_t {
                    pip_set_error(
                        b"IMOD_DIR is suspiciously long\0" as *const u8
                            as *const ::core::ffi::c_char,
                    );
                    return -(1 as ::core::ffi::c_int);
                }
                sprintf(
                    bigStr,
                    b"%s%c%s%c%s.%s\0" as *const u8 as *const ::core::ffi::c_char,
                    pipDir,
                    PATH_SEPARATOR,
                    OPTFILE_DIR.as_ptr(),
                    PATH_SEPARATOR,
                    progName,
                    OPTFILE_EXT.as_ptr(),
                );
                optFile = fopen(bigStr, b"r\0" as *const u8 as *const ::core::ffi::c_char);
            }
        }
    } else if localDir > 0 as ::core::ffi::c_int {
        ind = 0 as ::core::ffi::c_int;
        i = 0 as ::core::ffi::c_int;
        while i < localDir && i < 20 as ::core::ffi::c_int {
            let fresh20 = ind;
            ind = ind + 1;
            *bigStr.offset(fresh20 as isize) = '.' as i32 as ::core::ffi::c_char;
            let fresh21 = ind;
            ind = ind + 1;
            *bigStr.offset(fresh21 as isize) = '.' as i32 as ::core::ffi::c_char;
            let fresh22 = ind;
            ind = ind + 1;
            *bigStr.offset(fresh22 as isize) = PATH_SEPARATOR as ::core::ffi::c_char;
            i += 1;
        }
        sprintf(
            bigStr.offset(ind as isize),
            b"%s%c%s.%s\0" as *const u8 as *const ::core::ffi::c_char,
            OPTFILE_DIR.as_ptr(),
            PATH_SEPARATOR,
            progName,
            OPTFILE_EXT.as_ptr(),
        );
        optFile = fopen(bigStr, b"r\0" as *const u8 as *const ::core::ffi::c_char);
    }
    if optFile.is_null() {
        sprintf(
            bigStr,
            b"%s.%s\0" as *const u8 as *const ::core::ffi::c_char,
            progName,
            OPTFILE_EXT.as_ptr(),
        );
        optFile = fopen(bigStr, b"r\0" as *const u8 as *const ::core::ffi::c_char);
        if optFile.is_null() {
            sprintf(
                bigStr,
                b"Autodoc file %s.%s was not found or not readable.\nCheck environment variable settings of AUTODOC_DIR and IMOD_DIR\nor place autodoc file in current directory\0"
                    as *const u8 as *const ::core::ffi::c_char,
                progName,
                OPTFILE_EXT.as_ptr(),
            );
            pip_set_error(bigStr);
            return -(1 as ::core::ffi::c_int);
        }
    }
    loop {
        lineLen = pip_read_next_line(
            optFile,
            bigStr,
            bigSize,
            '#' as i32 as ::core::ffi::c_char,
            0 as ::core::ffi::c_int,
            0 as ::core::ffi::c_int,
            &raw mut indst,
        );
        if lineLen == -(3 as ::core::ffi::c_int) {
            break;
        }
        if lineLen == -(2 as ::core::ffi::c_int) {
            pip_set_error(
                b"Error reading option file\0" as *const u8 as *const ::core::ffi::c_char,
            );
            return -(1 as ::core::ffi::c_int);
        }
        if lineLen == -(1 as ::core::ffi::c_int) {
            bigSize += ADOC_STR_SIZE;
            free(bigStr as *mut ::core::ffi::c_void);
            bigStr = malloc(bigSize as size_t) as *mut ::core::ffi::c_char;
            if pip_memory_error(
                bigStr as *mut ::core::ffi::c_void,
                b"pip_read_option_file\0" as *const u8 as *const ::core::ffi::c_char,
            ) != 0
            {
                return -(1 as ::core::ffi::c_int);
            }
            numOpts = 0 as ::core::ffi::c_int;
            rewind(optFile);
        } else {
            if numOpts == 0 {
                check_keyword(
                    bigStr.offset(indst as isize),
                    b"KeyValueDelimiter\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut sValueDelim,
                    &raw mut gotDelim,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if pip_starts_with(
                    bigStr.offset(indst as isize),
                    b"DoubleDashOptions\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    sDoubleDashOptions = 1 as ::core::ffi::c_int;
                }
                if pip_starts_with(
                    bigStr.offset(indst as isize),
                    b"NoHelpAbbreviations\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    sNoHelpAbbrevs = 1 as ::core::ffi::c_int;
                }
                if pip_starts_with(
                    bigStr.offset(indst as isize),
                    b"NoAbbreviations\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    sNoAbbrevs = 1 as ::core::ffi::c_int;
                }
            }
            if line_is_option_token(bigStr.offset(indst as isize)) > 0 as ::core::ffi::c_int {
                numOpts += 1;
            }
        }
    }
    err = pip_initialize(numOpts);
    if err != 0 {
        return err;
    }
    rewind(optFile);
    formatStr = sNullString;
    manStr = formatStr;
    tipStr = manStr;
    usageStr = tipStr;
    type_0 = usageStr;
    shortName = type_0;
    longName = shortName;
    defaultStr = sNullString;
    gotDefault = 0 as ::core::ffi::c_int;
    gotFormat = gotDefault;
    gotMan = gotFormat;
    gotTip = gotMan;
    gotUsage = gotTip;
    gotType = gotUsage;
    gotShort = gotType;
    gotLong = gotShort;
    loop {
        lineLen = pip_read_next_line(
            optFile,
            bigStr,
            bigSize,
            '#' as i32 as ::core::ffi::c_char,
            0 as ::core::ffi::c_int,
            0 as ::core::ffi::c_int,
            &raw mut indst,
        );
        if lineLen == -(2 as ::core::ffi::c_int) {
            pip_set_error(
                b"Error reading autodoc file\0" as *const u8 as *const ::core::ffi::c_char,
            );
            return -(1 as ::core::ffi::c_int);
        }
        textStr = bigStr.offset(indst as isize);
        isOption = line_is_option_token(textStr);
        if readingOpt != 0 && (lineLen == -(3 as ::core::ffi::c_int) || isOption != 0) {
            if helpLevel <= 1 as ::core::ffi::c_int {
                if gotUsage != 0 {
                    helpStr = usageStr;
                } else if gotTip != 0 {
                    helpStr = tipStr;
                } else {
                    helpStr = manStr;
                }
            } else if helpLevel == 2 as ::core::ffi::c_int {
                if gotTip != 0 {
                    helpStr = tipStr;
                } else if gotUsage != 0 {
                    helpStr = usageStr;
                } else {
                    helpStr = manStr;
                }
            } else if gotMan != 0 {
                helpStr = manStr;
            } else if gotTip != 0 {
                helpStr = tipStr;
            } else {
                helpStr = usageStr;
            }
            if isSection != 0 {
                if gotShort != 0 {
                    free(shortName as *mut ::core::ffi::c_void);
                }
                if gotLong != 0 {
                    free(longName as *mut ::core::ffi::c_void);
                }
                shortName = sNullString;
                longName = shortName;
                gotShort = 0 as ::core::ffi::c_int;
                gotLong = gotShort;
            }
            needSize = strlen(shortName)
                .wrapping_add(strlen(longName))
                .wrapping_add(strlen(helpStr))
                .wrapping_add(15 as size_t) as ::core::ffi::c_int;
            if optStrSize < needSize {
                if optStrSize != 0 {
                    free(optStr as *mut ::core::ffi::c_void);
                }
                optStrSize = needSize;
                optStr = malloc(optStrSize as size_t) as *mut ::core::ffi::c_char;
                if pip_memory_error(
                    optStr as *mut ::core::ffi::c_void,
                    b"pip_read_option_file\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    return -(1 as ::core::ffi::c_int);
                }
            }
            sprintf(
                optStr,
                b"%s:%s:%s:%s\0" as *const u8 as *const ::core::ffi::c_char,
                shortName,
                longName,
                type_0,
                helpStr,
            );
            err = pip_add_option(optStr);
            if err != 0 {
                return err;
            }
            if gotFormat != 0 {
                let ref mut fresh23 =
                    (*sOptTable.offset((sNextOption - 1 as ::core::ffi::c_int) as isize)).format;
                *fresh23 = formatStr;
            }
            if gotDefault != 0 {
                let ref mut fresh24 = (*sOptTable
                    .offset((sNextOption - 1 as ::core::ffi::c_int) as isize))
                .defaultVal;
                *fresh24 = defaultStr;
            }
            if gotShort != 0 {
                free(shortName as *mut ::core::ffi::c_void);
            }
            if gotLong != 0 {
                free(longName as *mut ::core::ffi::c_void);
            }
            if gotType != 0 {
                free(type_0 as *mut ::core::ffi::c_void);
            }
            if gotUsage != 0 {
                free(usageStr as *mut ::core::ffi::c_void);
            }
            if gotTip != 0 {
                free(tipStr as *mut ::core::ffi::c_void);
            }
            if gotMan != 0 {
                free(manStr as *mut ::core::ffi::c_void);
            }
            formatStr = sNullString;
            manStr = formatStr;
            tipStr = manStr;
            usageStr = tipStr;
            type_0 = usageStr;
            shortName = type_0;
            longName = shortName;
            defaultStr = sNullString;
            gotFormat = 0 as ::core::ffi::c_int;
            gotMan = gotFormat;
            gotTip = gotMan;
            gotUsage = gotTip;
            gotType = gotUsage;
            gotShort = gotType;
            gotLong = gotShort;
            gotDefault = 0 as ::core::ffi::c_int;
            readingOpt = 0 as ::core::ffi::c_int;
        }
        if lineLen == -(3 as ::core::ffi::c_int) {
            break;
        }
        if readingOpt != 0 {
            if (lastGottenStr == &raw mut usageStr
                || lastGottenStr == &raw mut tipStr
                || lastGottenStr == &raw mut manStr)
                && (inQuoteIndex >= 0 as ::core::ffi::c_int
                    || strstr(textStr, sValueDelim).is_null())
            {
                ind = strlen(*lastGottenStr) as ::core::ffi::c_int;
                len = strlen(textStr) as ::core::ffi::c_int;
                needSize = ind + len + 3 as ::core::ffi::c_int;
                *lastGottenStr = realloc(
                    *lastGottenStr as *mut ::core::ffi::c_void,
                    needSize as size_t,
                ) as *mut ::core::ffi::c_char;
                strcat(
                    *lastGottenStr,
                    if *(*lastGottenStr).offset((ind - 1 as ::core::ffi::c_int) as isize)
                        as ::core::ffi::c_int
                        == '.' as i32
                    {
                        b"  \0" as *const u8 as *const ::core::ffi::c_char
                    } else {
                        b" \0" as *const u8 as *const ::core::ffi::c_char
                    },
                );
                if *textStr.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int
                    == '^' as i32
                {
                    *textStr.offset(0 as ::core::ffi::c_int as isize) =
                        '\n' as i32 as ::core::ffi::c_char;
                }
                if inQuoteIndex >= 0 as ::core::ffi::c_int
                    && (len == 0
                        || *textStr.offset((len - 1 as ::core::ffi::c_int) as isize)
                            as ::core::ffi::c_int
                            == *sQuoteTypes.offset(inQuoteIndex as isize) as ::core::ffi::c_int)
                {
                    if len != 0 {
                        *textStr.offset((len - 1 as ::core::ffi::c_int) as isize) =
                            0 as ::core::ffi::c_char;
                        strcat(*lastGottenStr, textStr);
                    }
                    inQuoteIndex = -(1 as ::core::ffi::c_int);
                    lastGottenStr = ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
                } else {
                    strcat(*lastGottenStr, textStr);
                }
            } else {
                lastGottenStr = ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
                inQuoteIndex = -(1 as ::core::ffi::c_int);
                err = check_keyword(
                    textStr,
                    b"short\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut shortName,
                    &raw mut gotShort,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if err != 0 {
                    return err;
                }
                err = check_keyword(
                    textStr,
                    b"long\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut longName,
                    &raw mut gotLong,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if err != 0 {
                    return err;
                }
                err = check_keyword(
                    textStr,
                    b"type\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut type_0,
                    &raw mut gotType,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if err != 0 {
                    return err;
                }
                err = check_keyword(
                    textStr,
                    b"format\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut formatStr,
                    &raw mut gotFormat,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if err != 0 {
                    return err;
                }
                err = check_keyword(
                    textStr,
                    b"default\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut defaultStr,
                    &raw mut gotDefault,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if err != 0 {
                    return err;
                }
                if helpLevel <= 1 as ::core::ffi::c_int || !(gotTip != 0 || gotMan != 0) {
                    err = check_keyword(
                        textStr,
                        b"usage\0" as *const u8 as *const ::core::ffi::c_char,
                        &raw mut usageStr,
                        &raw mut gotUsage,
                        &raw mut lastGottenStr,
                        &raw mut inQuoteIndex,
                    );
                    if err != 0 {
                        return err;
                    }
                }
                if helpLevel == 2 as ::core::ffi::c_int
                    || helpLevel <= 1 as ::core::ffi::c_int && gotUsage == 0
                    || helpLevel >= 3 as ::core::ffi::c_int && gotMan == 0
                {
                    err = check_keyword(
                        textStr,
                        b"tooltip\0" as *const u8 as *const ::core::ffi::c_char,
                        &raw mut tipStr,
                        &raw mut gotTip,
                        &raw mut lastGottenStr,
                        &raw mut inQuoteIndex,
                    );
                    if err != 0 {
                        return err;
                    }
                }
                if helpLevel >= 3 as ::core::ffi::c_int
                    || helpLevel == 2 as ::core::ffi::c_int && gotTip == 0
                    || helpLevel <= 1 as ::core::ffi::c_int && !(gotTip != 0 || gotUsage != 0)
                {
                    err = check_keyword(
                        textStr,
                        b"manpage\0" as *const u8 as *const ::core::ffi::c_char,
                        &raw mut manStr,
                        &raw mut gotMan,
                        &raw mut lastGottenStr,
                        &raw mut inQuoteIndex,
                    );
                    if err != 0 {
                        return err;
                    }
                }
                if inQuoteIndex >= 0 as ::core::ffi::c_int && !lastGottenStr.is_null() {
                    len = strlen(*lastGottenStr) as ::core::ffi::c_int;
                    if len != 0
                        && *(*lastGottenStr).offset((len - 1 as ::core::ffi::c_int) as isize)
                            as ::core::ffi::c_int
                            == *sQuoteTypes.offset(inQuoteIndex as isize) as ::core::ffi::c_int
                    {
                        *(*lastGottenStr).offset((len - 1 as ::core::ffi::c_int) as isize) =
                            0 as ::core::ffi::c_char;
                        inQuoteIndex = -(1 as ::core::ffi::c_int);
                        lastGottenStr = ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
                    }
                }
            }
        } else if isOption > 0 as ::core::ffi::c_int {
            lastGottenStr = ::core::ptr::null_mut::<*mut ::core::ffi::c_char>();
            readingOpt = 1 as ::core::ffi::c_int;
            isSection = isOption - 1 as ::core::ffi::c_int;
            if isSection == 0 {
                err = check_keyword(
                    textStr.offset(strlen(OPEN_DELIM.as_ptr()) as isize),
                    b"Field\0" as *const u8 as *const ::core::ffi::c_char,
                    &raw mut longName,
                    &raw mut gotLong,
                    &raw mut lastGottenStr,
                    ::core::ptr::null_mut::<::core::ffi::c_int>(),
                );
                if err != 0 {
                    return err;
                }
                if gotLong != 0 {
                    *longName.offset(strlen(longName).wrapping_sub(1 as size_t) as isize) =
                        sNullChar;
                }
            }
        }
    }
    free(bigStr as *mut ::core::ffi::c_void);
    if optStrSize != 0 {
        free(optStr as *mut ::core::ffi::c_void);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_parse_entries(
    mut argc: ::core::ffi::c_int,
    mut argv: *mut *mut ::core::ffi::c_char,
    mut numOptArgs: *mut ::core::ffi::c_int,
    mut numNonOptArgs: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut i: ::core::ffi::c_int = 0;
    let mut err: ::core::ffi::c_int = 0;
    if argc < sTakeStdIn {
        err = read_param_file(stdin);
        if err != 0 {
            return err;
        }
    } else {
        i = 1 as ::core::ffi::c_int;
        while i < argc {
            err = pip_next_arg(*argv.offset(i as isize));
            if err < 0 as ::core::ffi::c_int {
                return err;
            }
            if err != 0 && i == argc - 1 as ::core::ffi::c_int {
                pip_set_error(
                    b"A value was expected but not found for the last option on the command line\0"
                        as *const u8 as *const ::core::ffi::c_char,
                );
                return -(1 as ::core::ffi::c_int);
            }
            i += 1;
        }
    }
    pip_number_of_args(numOptArgs, numNonOptArgs);
    pip_print_entries();
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_read_or_parse_options(
    mut argc: ::core::ffi::c_int,
    mut argv: *mut *mut ::core::ffi::c_char,
    mut options: *mut *const ::core::ffi::c_char,
    mut numOpts: ::core::ffi::c_int,
    mut progName: *const ::core::ffi::c_char,
    mut minArgs: ::core::ffi::c_int,
    mut numInFiles: ::core::ffi::c_int,
    mut numOutFiles: ::core::ffi::c_int,
    mut numOptArgs: *mut ::core::ffi::c_int,
    mut numNonOptArgs: *mut ::core::ffi::c_int,
    mut headerFunc: Option<unsafe extern "C" fn(*const ::core::ffi::c_char) -> ()>,
) {
    let mut ierr: ::core::ffi::c_int = 0;
    let mut errString: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut prefix: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    prefix = malloc(strlen(progName).wrapping_add(12 as size_t)) as *mut ::core::ffi::c_char;
    sprintf(
        prefix,
        b"ERROR: %s -\0" as *const u8 as *const ::core::ffi::c_char,
        progName,
    );
    ierr = pip_read_option_file(progName, 0 as ::core::ffi::c_int, 0 as ::core::ffi::c_int);
    pip_exit_on_error(0 as ::core::ffi::c_int, prefix);
    free(prefix as *mut ::core::ffi::c_void);
    if ierr == 0 {
        pip_parse_entries(argc, argv, numOptArgs, numNonOptArgs);
    } else {
        pip_get_error(&raw mut errString);
        if options.is_null() || numOpts == 0 {
            pip_set_error(errString);
        }
        if !errString.is_null() {
            printf(
                b"PIP WARNING: %s\nUsing fallback options in main program\n\0" as *const u8
                    as *const ::core::ffi::c_char,
                errString,
            );
            free(errString as *mut ::core::ffi::c_void);
        }
        pip_parse_input(argc, argv, options, numOpts, numOptArgs, numNonOptArgs);
        pip_read_prog_defaults(progName);
    }
    if *numOptArgs + *numNonOptArgs < minArgs
        || pip_get_boolean(
            b"help\0" as *const u8 as *const ::core::ffi::c_char,
            &raw mut ierr,
        ) == 0 as ::core::ffi::c_int
            && ierr != 0
    {
        if headerFunc.is_some() {
            headerFunc.expect("non-null function pointer")(progName);
        }
        pip_print_help(progName, 0 as ::core::ffi::c_int, numInFiles, numOutFiles);
        exit(0 as ::core::ffi::c_int);
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_read_prog_defaults(mut progName: *const ::core::ffi::c_char) {
    let mut pipDir: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut savePrefix: ::core::ffi::c_char = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut sectInd: ::core::ffi::c_int = 0;
    let mut adocInd: ::core::ffi::c_int = 0;
    pipDir = getenv(b"IMOD_DIR\0" as *const u8 as *const ::core::ffi::c_char);
    if pipDir.is_null() || strlen(pipDir) > (TEMP_STR_SIZE - 100 as ::core::ffi::c_int) as size_t {
        return;
    }
    savePrefix = sExitPrefix[0 as ::core::ffi::c_int as usize];
    sExitPrefix[0 as ::core::ffi::c_int as usize] = 0 as ::core::ffi::c_char;
    sprintf(
        sTempStr,
        b"%s%c%s%c%s\0" as *const u8 as *const ::core::ffi::c_char,
        pipDir,
        PATH_SEPARATOR,
        DEFAULTS_DIR.as_ptr(),
        PATH_SEPARATOR,
        DEFAULTS_FILE.as_ptr(),
    );
    adocInd = crate::imod::libcfshr::autodoc::adoc_read(sTempStr);
    if adocInd >= 0 as ::core::ffi::c_int {
        sectInd = crate::imod::libcfshr::autodoc::adoc_lookup_section(
            b"Program\0" as *const u8 as *const ::core::ffi::c_char,
            progName,
        );
        if sectInd >= 0 as ::core::ffi::c_int {
            i = 0 as ::core::ffi::c_int;
            while i < sTableSize {
                crate::imod::libcfshr::autodoc::adoc_get_string(
                    b"Program\0" as *const u8 as *const ::core::ffi::c_char,
                    sectInd,
                    (*sOptTable.offset(i as isize)).longName,
                    &raw mut (*sOptTable.offset(i as isize)).defaultVal,
                );
                i += 1;
            }
        }
        crate::imod::libcfshr::autodoc::adoc_clear(adocInd);
    }
    sExitPrefix[0 as ::core::ffi::c_int as usize] = savePrefix;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_in_out_file(
    mut option: *const ::core::ffi::c_char,
    mut nonOptArgNo: ::core::ffi::c_int,
    mut filename: *mut *mut ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    if pip_get_string(option, filename) != 0 {
        if nonOptArgNo >= (*sOptTable.offset(sNonOptInd as isize)).count {
            return 1 as ::core::ffi::c_int;
        }
        pip_get_non_option_arg(nonOptArgNo, filename);
    }
    return 0 as ::core::ffi::c_int;
}
unsafe extern "C" fn read_param_file(mut pFile: *mut FILE) -> ::core::ffi::c_int {
    let mut lineLen: ::core::ffi::c_int = 0;
    let mut indst: ::core::ffi::c_int = 0;
    let mut indnd: ::core::ffi::c_int = 0;
    let mut optNum: ::core::ffi::c_int = 0;
    let mut gotEquals: ::core::ffi::c_int = 0;
    let mut err: ::core::ffi::c_int = 0;
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut token: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    loop {
        sNotFoundOK = if sNumOptionArguments == 0
            && (*sOptTable.offset(sNonOptInd as isize)).count < sNonOptLines
        {
            1 as ::core::ffi::c_int
        } else {
            0 as ::core::ffi::c_int
        };
        lineLen = pip_read_next_line(
            pFile,
            sLineStr,
            LINE_STR_SIZE,
            '#' as i32 as ::core::ffi::c_char,
            0 as ::core::ffi::c_int,
            1 as ::core::ffi::c_int,
            &raw mut indst,
        );
        if lineLen == -(3 as ::core::ffi::c_int) {
            break;
        }
        if lineLen == -(2 as ::core::ffi::c_int) {
            pip_set_error(
                b"Error reading parameter file or StandardInput\0" as *const u8
                    as *const ::core::ffi::c_char,
            );
            return -(1 as ::core::ffi::c_int);
        }
        if lineLen == -(1 as ::core::ffi::c_int) {
            pip_set_error(
                b"Line too long for buffer while reading parameter file or StandardInput\0"
                    as *const u8 as *const ::core::ffi::c_char,
            );
            return -(1 as ::core::ffi::c_int);
        }
        strPtr = strpbrk(
            sLineStr.offset(indst as isize),
            b"= \t\0" as *const u8 as *const ::core::ffi::c_char,
        );
        indnd = (if !strPtr.is_null() {
            strPtr.offset_from(sLineStr) as ::core::ffi::c_long - 1 as ::core::ffi::c_long
        } else {
            (lineLen - 1 as ::core::ffi::c_int) as ::core::ffi::c_long
        }) as ::core::ffi::c_int;
        if indnd >= lineLen {
            indnd = lineLen - 1 as ::core::ffi::c_int;
        }
        token = pip_sub_str_dup(sLineStr, indst, indnd);
        if pip_memory_error(
            token as *mut ::core::ffi::c_void,
            b"read_param_file\0" as *const u8 as *const ::core::ffi::c_char,
        ) != 0
        {
            return -(1 as ::core::ffi::c_int);
        }
        if strcmp(token, STANDARD_INPUT_END.as_ptr()) == 0
            || sDoneEnds != 0
                && strlen(token) == 4 as size_t
                && pip_starts_with(b"DONE\0" as *const u8 as *const ::core::ffi::c_char, token) != 0
        {
            break;
        }
        optNum = lookup_option(token, sNumOptions);
        free(token as *mut ::core::ffi::c_void);
        if optNum < 0 as ::core::ffi::c_int {
            if sNotFoundOK != 0 {
                token = pip_sub_str_dup(sLineStr, indst, lineLen - 1 as ::core::ffi::c_int);
                if pip_memory_error(
                    token as *mut ::core::ffi::c_void,
                    b"read_param_file\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
                {
                    return -(1 as ::core::ffi::c_int);
                }
                err = add_value_string(sNonOptInd, token);
                if err != 0 {
                    return err;
                }
            } else {
                return optNum;
            }
        } else {
            if strcmp(
                (*sOptTable.offset(optNum as isize)).type_0,
                PARAM_FILE_STRING.as_ptr(),
            ) == 0
            {
                pip_set_error(
                    b"Trying to open a parameter file while reading a parameter file or StandardInput\0"
                        as *const u8 as *const ::core::ffi::c_char,
                );
                return -(1 as ::core::ffi::c_int);
            }
            indst = indnd + 1 as ::core::ffi::c_int;
            gotEquals = 0 as ::core::ffi::c_int;
            while indst < lineLen {
                if *sLineStr.offset(indst as isize) as ::core::ffi::c_int == '=' as i32 {
                    if gotEquals != 0 {
                        sprintf(
                            sTempStr,
                            b"Two = signs in input line:  \0" as *const u8
                                as *const ::core::ffi::c_char,
                        );
                        append_to_error_string(sLineStr);
                        return -(1 as ::core::ffi::c_int);
                    }
                    gotEquals = 1 as ::core::ffi::c_int;
                } else if *sLineStr.offset(indst as isize) as ::core::ffi::c_int != ' ' as i32
                    && *sLineStr.offset(indst as isize) as ::core::ffi::c_int != '\t' as i32
                {
                    break;
                }
                indst += 1;
            }
            if indst < lineLen {
                token = pip_sub_str_dup(sLineStr, indst, lineLen - 1 as ::core::ffi::c_int);
            } else if strcmp(
                (*sOptTable.offset(optNum as isize)).type_0,
                BOOLEAN_STRING.as_ptr(),
            ) == 0
            {
                token = strdup(b"1\0" as *const u8 as *const ::core::ffi::c_char);
            } else {
                sprintf(
                    sTempStr,
                    b"Missing a value on the input line:  \0" as *const u8
                        as *const ::core::ffi::c_char,
                );
                append_to_error_string(sLineStr);
                return -(1 as ::core::ffi::c_int);
            }
            if pip_memory_error(
                token as *mut ::core::ffi::c_void,
                b"read_param_file\0" as *const u8 as *const ::core::ffi::c_char,
            ) != 0
            {
                return -(1 as ::core::ffi::c_int);
            }
            err = add_value_string(optNum, token);
            if err != 0 {
                return err;
            }
            sNumOptionArguments += 1;
        }
    }
    sNotFoundOK = 0 as ::core::ffi::c_int;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_read_stdin_if_set() -> ::core::ffi::c_int {
    if sTakeStdIn != 0 {
        return read_param_file(stdin);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_read_next_line(
    mut pFile: *mut FILE,
    mut sLineStr_0: *mut ::core::ffi::c_char,
    mut strSize: ::core::ffi::c_int,
    mut comment: ::core::ffi::c_char,
    mut keepComments: ::core::ffi::c_int,
    mut inLineComments: ::core::ffi::c_int,
    mut firstNonWhite: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut indst: ::core::ffi::c_int = 0;
    let mut lineLen: ::core::ffi::c_int = 0;
    let mut ch: ::core::ffi::c_char = 0;
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    loop {
        if fgets(sLineStr_0, strSize, pFile).is_null() {
            if feof(pFile) != 0 {
                return -(3 as ::core::ffi::c_int);
            }
            return -(2 as ::core::ffi::c_int);
        }
        lineLen = strlen(sLineStr_0) as ::core::ffi::c_int;
        if lineLen == strSize - 1 as ::core::ffi::c_int {
            return -(1 as ::core::ffi::c_int);
        }
        indst = 0 as ::core::ffi::c_int;
        while indst < lineLen {
            if *sLineStr_0.offset(indst as isize) as ::core::ffi::c_int != ' ' as i32
                && *sLineStr_0.offset(indst as isize) as ::core::ffi::c_int != '\t' as i32
            {
                break;
            }
            indst += 1;
        }
        if *sLineStr_0.offset(indst as isize) as ::core::ffi::c_int == comment as ::core::ffi::c_int
        {
            if !(keepComments != 0) {
                continue;
            }
            while *sLineStr_0.offset((lineLen - 1 as ::core::ffi::c_int) as isize)
                as ::core::ffi::c_int
                == '\n' as i32
                || *sLineStr_0.offset((lineLen - 1 as ::core::ffi::c_int) as isize)
                    as ::core::ffi::c_int
                    == '\r' as i32
            {
                lineLen -= 1;
            }
            *sLineStr_0.offset(lineLen as isize) = 0 as ::core::ffi::c_char;
            break;
        } else {
            strPtr = strchr(sLineStr_0, comment as ::core::ffi::c_int);
            if !strPtr.is_null() && inLineComments != 0 {
                lineLen =
                    strPtr.offset_from(sLineStr_0) as ::core::ffi::c_long as ::core::ffi::c_int;
            }
            while lineLen > 0 as ::core::ffi::c_int {
                ch = *sLineStr_0.offset((lineLen - 1 as ::core::ffi::c_int) as isize);
                if ch as ::core::ffi::c_int != ' ' as i32
                    && ch as ::core::ffi::c_int != '\t' as i32
                    && ch as ::core::ffi::c_int != '\n' as i32
                    && ch as ::core::ffi::c_int != '\r' as i32
                {
                    break;
                }
                lineLen -= 1;
            }
            *sLineStr_0.offset(lineLen as isize) = 0 as ::core::ffi::c_char;
            if indst < lineLen || keepComments != 0 {
                break;
            }
        }
    }
    *firstNonWhite = indst;
    return lineLen;
}
unsafe extern "C" fn option_line_of_values(
    mut option: *const ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_void,
    mut valType: ::core::ffi::c_int,
    mut numToGet: *mut ::core::ffi::c_int,
    mut arraySize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut strPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut err: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut valErr: ::core::ffi::c_int = 0;
    valErr = get_next_value_string(option, &raw mut strPtr);
    if valErr == 0 || valErr == 2 as ::core::ffi::c_int {
        err = pip_get_line_of_values(option, strPtr, array, valType, numToGet, arraySize);
    }
    if err != 0 {
        return err;
    }
    return valErr;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_get_line_of_values(
    mut option: *const ::core::ffi::c_char,
    mut strPtr: *const ::core::ffi::c_char,
    mut array: *mut ::core::ffi::c_void,
    mut valType: ::core::ffi::c_int,
    mut numToGet: *mut ::core::ffi::c_int,
    mut arraySize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut sepPtr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut endPtr: *const ::core::ffi::c_char = ::core::ptr::null::<::core::ffi::c_char>();
    let mut invalid: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut fullStr: *const ::core::ffi::c_char = ::core::ptr::null::<::core::ffi::c_char>();
    let mut iarray: *mut ::core::ffi::c_int = array as *mut ::core::ffi::c_int;
    let mut farray: *mut ::core::ffi::c_float = array as *mut ::core::ffi::c_float;
    let mut darray: *mut ::core::ffi::c_double = array as *mut ::core::ffi::c_double;
    let mut numGot: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut gotComma: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    let mut sepStr: [::core::ffi::c_char; 5] =
        ::core::mem::transmute::<[u8; 5], [::core::ffi::c_char; 5]>(*b",\t /\0");
    fullStr = strPtr;
    while strlen(strPtr) != 0 {
        sepPtr = strpbrk(strPtr, &raw mut sepStr as *mut ::core::ffi::c_char);
        if sepPtr.is_null() {
            endPtr = strPtr.offset(strlen(strPtr) as isize);
        } else if sepPtr == strPtr as *mut ::core::ffi::c_char {
            if *strPtr.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == '/' as i32
            {
                if sAllowDefaults != 0 && *numToGet > 0 as ::core::ffi::c_int {
                    numGot = *numToGet;
                    break;
                } else {
                    line_of_values_error(
                        fullStr,
                        format_args!(
                            "Default entry with a / is not allowed in value entry:  {}  ",
                            ::core::ffi::CStr::from_ptr(option).to_string_lossy()
                        ),
                    );
                    return -(1 as ::core::ffi::c_int);
                }
            } else {
                if *strPtr.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int
                    == ',' as i32
                {
                    if gotComma != 0 {
                        if sAllowDefaults != 0 && *numToGet > 0 as ::core::ffi::c_int {
                            numGot += 1;
                            if numGot >= *numToGet {
                                break;
                            }
                        } else {
                            line_of_values_error(
                                fullStr,
                                format_args!(
                                    "Default entries with commas are not allowed in value entry:  {}  ",
                                    ::core::ffi::CStr::from_ptr(option).to_string_lossy()
                                ),
                            );
                            return -(1 as ::core::ffi::c_int);
                        }
                    }
                    gotComma = 1 as ::core::ffi::c_int;
                }
                strPtr = strPtr.offset(1);
                continue;
            }
        } else {
            endPtr = sepPtr;
        }
        if numGot >= arraySize {
            line_of_values_error(
                fullStr,
                format_args!(
                    "Too many values for input array in value entry:  {}  ",
                    ::core::ffi::CStr::from_ptr(option).to_string_lossy()
                ),
            );
            return -(1 as ::core::ffi::c_int);
        }
        if valType == PIP_INTEGER {
            let fresh16 = numGot;
            numGot = numGot + 1;
            *iarray.offset(fresh16 as isize) =
                strtol(strPtr, &raw mut invalid, 10 as ::core::ffi::c_int) as ::core::ffi::c_int;
        } else if valType == PIP_FLOAT {
            let fresh17 = numGot;
            numGot = numGot + 1;
            *farray.offset(fresh17 as isize) =
                strtod(strPtr, &raw mut invalid) as ::core::ffi::c_float;
        } else {
            let fresh18 = numGot;
            numGot = numGot + 1;
            *darray.offset(fresh18 as isize) = strtod(strPtr, &raw mut invalid);
        }
        if invalid != endPtr as *mut ::core::ffi::c_char {
            line_of_values_error(
                fullStr,
                format_args!(
                    "Illegal character in value entry:  {}  ",
                    ::core::ffi::CStr::from_ptr(option).to_string_lossy()
                ),
            );
            return -(1 as ::core::ffi::c_int);
        }
        gotComma = 0 as ::core::ffi::c_int;
        if *endPtr == 0
            || *numToGet > 0 as ::core::ffi::c_int && numGot >= *numToGet
            || *numToGet < 0 as ::core::ffi::c_int && numGot >= arraySize
        {
            break;
        }
        strPtr = endPtr;
    }
    if *numToGet <= 0 as ::core::ffi::c_int {
        *numToGet = numGot;
    }
    if numGot < *numToGet {
        line_of_values_error(
            fullStr,
            format_args!(
                "{} values expected but only {} values found in value entry:  {}  ",
                *numToGet,
                numGot,
                ::core::ffi::CStr::from_ptr(option).to_string_lossy()
            ),
        );
        return -(1 as ::core::ffi::c_int);
    }
    return 0 as ::core::ffi::c_int;
}
/// Source `LineOfValuesError` (`parse_params.c:1999`).  The C `format`/varargs
/// pair maps to `core::fmt::Arguments`, the same boundary shape `b3dError`
/// uses, and `vsprintf` into `sTempStr` is reproduced byte for byte.
unsafe fn line_of_values_error(
    full_str: *const ::core::ffi::c_char,
    format: core::fmt::Arguments<'_>,
) {
    // `PipLineOfValues` is explicitly usable before `PipInitialize`; the C
    // `LineOfValuesError` therefore obtains this scratch buffer on demand.
    if sTempStr.is_null() {
        sTempStr = malloc(TEMP_STR_SIZE as size_t) as *mut ::core::ffi::c_char;
    }
    if sTempStr.is_null() {
        return;
    }
    let text = format.to_string();
    let bytes = text.as_bytes();
    let count = bytes.len().min(TEMP_STR_SIZE as usize - 1);
    ::core::ptr::copy_nonoverlapping(
        bytes.as_ptr().cast::<::core::ffi::c_char>(),
        sTempStr,
        count,
    );
    *sTempStr.add(count) = 0;
    append_to_error_string(full_str);
}
pub unsafe extern "C" fn get_next_value_string(
    mut option: *const ::core::ffi::c_char,
    mut strPtr: *mut *mut ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut index: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    err = lookup_option(option, sNonOptInd + 1 as ::core::ffi::c_int);
    if err < 0 as ::core::ffi::c_int {
        return err;
    }
    if (*sOptTable.offset(err as isize)).count == 0 {
        if (*sOptTable.offset(err as isize)).defaultVal.is_null() {
            return 1 as ::core::ffi::c_int;
        }
        *strPtr = (*sOptTable.offset(err as isize)).defaultVal;
        return 2 as ::core::ffi::c_int;
    }
    if (*sOptTable.offset(err as isize)).multiple != 0 {
        index = (*sOptTable.offset(err as isize)).multiple - 1 as ::core::ffi::c_int;
        if (*sOptTable.offset(err as isize)).multiple < (*sOptTable.offset(err as isize)).count {
            let ref mut fresh15 = (*sOptTable.offset(err as isize)).multiple;
            *fresh15 += 1;
        }
    }
    *strPtr = *(*sOptTable.offset(err as isize))
        .valuePtr
        .offset(index as isize);
    return 0 as ::core::ffi::c_int;
}
pub unsafe extern "C" fn add_value_string(
    mut option: ::core::ffi::c_int,
    mut strPtr: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut optp: *mut PipOptions = sOptTable.offset(option as isize) as *mut PipOptions;
    if (*optp).count == 0 || (*optp).multiple != 0 {
        if (*optp).count != 0 {
            (*optp).valuePtr = realloc(
                (*optp).valuePtr as *mut ::core::ffi::c_void,
                (((*optp).count + 1 as ::core::ffi::c_int) as size_t)
                    .wrapping_mul(::core::mem::size_of::<*mut ::core::ffi::c_char>() as size_t),
            ) as *mut *mut ::core::ffi::c_char;
            if (*optp).linked != 0 {
                (*optp).nextLinked = realloc(
                    (*optp).nextLinked as *mut ::core::ffi::c_void,
                    ((((*optp).count + 1 as ::core::ffi::c_int) * 2 as ::core::ffi::c_int)
                        as size_t)
                        .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
                ) as *mut ::core::ffi::c_int;
            }
        } else {
            (*optp).valuePtr = malloc(::core::mem::size_of::<*mut ::core::ffi::c_char>() as size_t)
                as *mut *mut ::core::ffi::c_char;
            if (*optp).linked != 0 {
                (*optp).nextLinked = malloc(
                    (2 as size_t)
                        .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
                ) as *mut ::core::ffi::c_int;
            }
        }
        if pip_memory_error(
            (*optp).valuePtr as *mut ::core::ffi::c_void,
            b"add_value_string\0" as *const u8 as *const ::core::ffi::c_char,
        ) != 0
            || (*optp).linked != 0
                && pip_memory_error(
                    (*optp).nextLinked as *mut ::core::ffi::c_void,
                    b"add_value_string\0" as *const u8 as *const ::core::ffi::c_char,
                ) != 0
        {
            return -(1 as ::core::ffi::c_int);
        }
    } else {
        if !(*(*optp).valuePtr).is_null() {
            free(*(*optp).valuePtr as *mut ::core::ffi::c_void);
        }
        (*optp).count = 0 as ::core::ffi::c_int;
    }
    if (*optp).linked != 0 {
        *(*optp)
            .nextLinked
            .offset((2 as ::core::ffi::c_int * (*optp).count) as isize) =
            (*sOptTable.offset(sNonOptInd as isize)).count;
        *(*optp)
            .nextLinked
            .offset((2 as ::core::ffi::c_int * (*optp).count + 1 as ::core::ffi::c_int) as isize) =
            0 as ::core::ffi::c_int;
        if !sLinkedOption.is_null() {
            err = lookup_option(sLinkedOption, sNumOptions);
            if err < 0 as ::core::ffi::c_int {
                return err;
            }
            *(*optp).nextLinked.offset(
                (2 as ::core::ffi::c_int * (*optp).count + 1 as ::core::ffi::c_int) as isize,
            ) = (*sOptTable.offset(err as isize)).count;
        }
    }
    let fresh13 = (*optp).count;
    (*optp).count = (*optp).count + 1;
    let ref mut fresh14 = *(*optp).valuePtr.offset(fresh13 as isize);
    *fresh14 = strPtr as *mut ::core::ffi::c_char;
    return 0 as ::core::ffi::c_int;
}
pub unsafe extern "C" fn lookup_option(
    mut option: *const ::core::ffi::c_char,
    mut maxLookup: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut starts: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut lenopt: ::core::ffi::c_int = 0;
    let mut lenShort: ::core::ffi::c_int = 0;
    let mut found: ::core::ffi::c_int = LOOKUP_NOT_FOUND;
    let mut sname: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut lname: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    lenopt = strlen(option) as ::core::ffi::c_int;
    i = 0 as ::core::ffi::c_int;
    while i < maxLookup {
        sname = (*sOptTable.offset(i as isize)).shortName;
        lname = (*sOptTable.offset(i as isize)).longName;
        lenShort = (*sOptTable.offset(i as isize)).lenShort;
        starts = pip_starts_with(sname, option);
        if lenopt == 1 as ::core::ffi::c_int && starts != 0 && lenShort == 1 as ::core::ffi::c_int {
            found = i;
            break;
        } else {
            if starts != 0 && (sNoAbbrevs == 0 || lenopt == lenShort)
                || pip_starts_with(lname, option) != 0
                    && (sNoAbbrevs == 0 || lenopt as size_t == strlen(lname))
            {
                if found == LOOKUP_NOT_FOUND {
                    found = i;
                } else {
                    if sTestAbbrevForUsage == 0 {
                        sprintf(
                            sTempStr,
                            b"An option specified by \"%s\" is ambiguous between option %s -  %s  and option %s -  %s\0"
                                as *const u8 as *const ::core::ffi::c_char,
                            option,
                            sname,
                            lname,
                            (*sOptTable.offset(found as isize)).shortName,
                            (*sOptTable.offset(found as isize)).longName,
                        );
                        pip_set_error(sTempStr);
                    }
                    return LOOKUP_AMBIGUOUS;
                }
            }
            i += 1;
        }
    }
    if found == LOOKUP_NOT_FOUND && sNotFoundOK == 0 {
        sprintf(
            sTempStr,
            b"Illegal option: %s\0" as *const u8 as *const ::core::ffi::c_char,
            option,
        );
        pip_set_error(sTempStr);
    }
    return found;
}
unsafe extern "C" fn pip_sub_str_dup(
    mut s1: *const ::core::ffi::c_char,
    mut i1: ::core::ffi::c_int,
    mut i2: ::core::ffi::c_int,
) -> *mut ::core::ffi::c_char {
    let mut i: ::core::ffi::c_int = 0;
    let mut size: ::core::ffi::c_int = i2 + 2 as ::core::ffi::c_int - i1;
    let mut s2: *mut ::core::ffi::c_char = malloc(size as size_t) as *mut ::core::ffi::c_char;
    if pip_memory_error(
        s2 as *mut ::core::ffi::c_void,
        b"pip_sub_str_dup\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0
    {
        return ::core::ptr::null_mut::<::core::ffi::c_char>();
    }
    i = i1;
    while i <= i2 {
        *s2.offset((i - i1) as isize) = *s1.offset(i as isize);
        i += 1;
    }
    *s2.offset((size - 1 as ::core::ffi::c_int) as isize) = 0 as ::core::ffi::c_char;
    return s2;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_memory_error(
    mut ptr: *mut ::core::ffi::c_void,
    mut routine: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    if !ptr.is_null() {
        return 0 as ::core::ffi::c_int;
    }
    if sTempStr.is_null() {
        sTempStr = malloc(TEMP_STR_SIZE as size_t) as *mut ::core::ffi::c_char;
    }
    if sTempStr.is_null() {
        pip_set_error(
            b"Failed to get memory for string in pip_memory_error\0" as *const u8
                as *const ::core::ffi::c_char,
        );
        return -(1 as ::core::ffi::c_int);
    }
    sprintf(
        sTempStr,
        b"Failed to get memory for string in %s\0" as *const u8 as *const ::core::ffi::c_char,
        routine,
    );
    pip_set_error(sTempStr);
    return -(1 as ::core::ffi::c_int);
}
unsafe extern "C" fn append_to_error_string(mut str: *const ::core::ffi::c_char) {
    let mut len: ::core::ffi::c_int = strlen(sTempStr) as ::core::ffi::c_int;
    *sTempStr.offset((TEMP_STR_SIZE - 1 as ::core::ffi::c_int) as isize) = 0 as ::core::ffi::c_char;
    strncpy(
        sTempStr.offset(len as isize) as *mut ::core::ffi::c_char,
        str,
        (TEMP_STR_SIZE - len - 1 as ::core::ffi::c_int) as size_t,
    );
    pip_set_error(sTempStr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pip_starts_with(
    mut fullStr: *const ::core::ffi::c_char,
    mut subStr: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    if fullStr.is_null() || subStr.is_null() {
        return 0 as ::core::ffi::c_int;
    }
    if *fullStr == 0 || *subStr == 0 {
        return 0 as ::core::ffi::c_int;
    }
    if sNoCase != 0 {
        while *fullStr as ::core::ffi::c_int != 0 && *subStr as ::core::ffi::c_int != 0 {
            let fresh11 = fullStr;
            fullStr = fullStr.offset(1);
            let fresh12 = subStr;
            subStr = subStr.offset(1);
            if toupper(*fresh11 as ::core::ffi::c_int) != toupper(*fresh12 as ::core::ffi::c_int) {
                return 0 as ::core::ffi::c_int;
            }
        }
        if *subStr == 0 {
            return 1 as ::core::ffi::c_int;
        }
    } else if strstr(fullStr, subStr) == fullStr as *mut ::core::ffi::c_char {
        return 1 as ::core::ffi::c_int;
    }
    return 0 as ::core::ffi::c_int;
}
unsafe extern "C" fn line_is_option_token(
    mut line: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut token: *const ::core::ffi::c_char = ::core::ptr::null::<::core::ffi::c_char>();
    if pip_starts_with(line, OPEN_DELIM.as_ptr()) == 0
        || strstr(line, CLOSE_DELIM.as_ptr()).is_null()
    {
        return 0 as ::core::ffi::c_int;
    }
    token = line.offset(strlen(OPEN_DELIM.as_ptr()) as isize);
    if pip_starts_with(token, b"Field\0" as *const u8 as *const ::core::ffi::c_char) != 0 {
        return 1 as ::core::ffi::c_int;
    }
    if pip_starts_with(
        token,
        b"SectionHeader\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0
    {
        return 2 as ::core::ffi::c_int;
    }
    return -(1 as ::core::ffi::c_int);
}
unsafe extern "C" fn check_keyword(
    mut line: *const ::core::ffi::c_char,
    mut keyword: *const ::core::ffi::c_char,
    mut copyto: *mut *mut ::core::ffi::c_char,
    mut gotit: *mut ::core::ffi::c_int,
    mut lastCopied: *mut *mut *mut ::core::ffi::c_char,
    mut quoteInd: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut valStart: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut quoteStart: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut copyStr: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    if pip_starts_with(line, keyword) == 0 {
        return 0 as ::core::ffi::c_int;
    }
    valStart = strstr(line, sValueDelim);
    if valStart.is_null() {
        return 0 as ::core::ffi::c_int;
    }
    if *gotit != 0 && *copyto != sNullString {
        free(*copyto as *mut ::core::ffi::c_void);
        *copyto = sNullString;
        *gotit = 0 as ::core::ffi::c_int;
    }
    valStart = valStart.offset(strlen(sValueDelim) as isize);
    while *valStart as ::core::ffi::c_int == ' ' as i32
        || *valStart as ::core::ffi::c_int == '\t' as i32
    {
        valStart = valStart.offset(1);
    }
    if *valStart == 0 {
        return 0 as ::core::ffi::c_int;
    }
    if !quoteInd.is_null() {
        quoteStart = strchr(sQuoteTypes, *valStart as ::core::ffi::c_int);
        if !quoteStart.is_null() {
            *quoteInd =
                quoteStart.offset_from(sQuoteTypes) as ::core::ffi::c_long as ::core::ffi::c_int;
            valStart = valStart.offset(1);
        } else {
            *quoteInd = -(1 as ::core::ffi::c_int);
        }
    }
    copyStr = strdup(valStart);
    if pip_memory_error(
        copyStr as *mut ::core::ffi::c_void,
        b"check_keyword\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0
    {
        return -(1 as ::core::ffi::c_int);
    }
    *gotit = 1 as ::core::ffi::c_int;
    *copyto = copyStr;
    *lastCopied = copyto;
    return 0 as ::core::ffi::c_int;
}
