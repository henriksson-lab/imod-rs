//! Argument acquisition from `IMOD/flib/subrs/hvem/getinout.f`.

use std::io::{self, Write};

/// Original: `getinout` (`getinout.f:9`).
pub fn getinout(narg: i32) -> io::Result<(String, String)> {
    let arguments: Vec<String> = crate::imod::libcfshr::b3dutil::program_args()
        .into_iter()
        .skip(1)
        .collect();
    let input = match arguments.first() {
        Some(argument) => argument.clone(),
        None => {
            print!(" Name of input file: ");
            io::stdout().flush()?;
            let mut value = String::new();
            // `read(5,100)` has no `END=`: at end of input the gfortran
            // runtime reports the failure and stops with status 2.
            if io::stdin().read_line(&mut value)? == 0 {
                eprintln!("Fortran runtime error: End of file");
                crate::imod::libcfshr::b3dutil::exit(2);
            }
            value.trim_end_matches(['\r', '\n']).to_owned()
        }
    };
    let output = if narg > 1 {
        match arguments.get(1) {
            Some(argument) => argument.clone(),
            None => {
                print!(" Name of output file: ");
                io::stdout().flush()?;
                let mut value = String::new();
                if io::stdin().read_line(&mut value)? == 0 {
                    eprintln!("Fortran runtime error: End of file");
                    crate::imod::libcfshr::b3dutil::exit(2);
                }
                value.trim_end_matches(['\r', '\n']).to_owned()
            }
        }
    } else {
        String::new()
    };
    Ok((input, output))
}
