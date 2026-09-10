fn main() {
    let arguments: Vec<std::ffi::CString> = std::env::args()
        .map(|argument| std::ffi::CString::new(argument).expect("argument contains NUL"))
        .collect();
    let mut pointers: Vec<*mut std::ffi::c_char> = arguments
        .iter()
        .map(|argument| argument.as_ptr().cast_mut())
        .collect();
    pointers.push(std::ptr::null_mut());
    std::process::exit(unsafe {
        imod_rs::imod::mrc::tif2mrc::tif2mrc(arguments.len() as i32, pointers.as_mut_ptr())
    });
}
