//! Conformance coverage for `IMOD/qttools/sourcedoc/sourcedoc.cpp`.

use std::fs;
use std::process::Command;

#[test]
fn sourcedoc_expands_c_documentation_and_special_codes() {
    let directory = std::env::temp_dir().join(format!(
        "imod-rs-sourcedoc-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("source.c"),
        "/*! First [bold] {italic} ^ @alpha @manual.html#external\n */\nint alpha(int value)\n{\n}\n",
    )
    .unwrap();
    fs::write(
        directory.join("input.html"),
        "Before\nLIST FUNCTIONS FROM source.c\nDESCRIBE FUNCTIONS FROM source.c\nAfter\n",
    )
    .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_sourcedoc"))
        .arg("-d")
        .arg(&directory)
        .arg(directory.join("input.html"))
        .arg(directory.join("output.html"))
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        fs::read_to_string(directory.join("output.html")).unwrap(),
        "Before\n<BR>int <A HREF=\"#alpha\">alpha</A>(int value)\n<H3><A NAME=\"alpha\"></A>int alpha(int value)</H3><P>\nFirst <B>bold</B> <I>italic</I> <BR> <A HREF=\"#alpha\">alpha</A> <A HREF=\"manual.html#external\">external</A>\n</P>\n\nAfter\n"
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sourcedoc_expands_documented_code_and_escapes_code_html() {
    let directory = std::env::temp_dir().join(format!(
        "imod-rs-sourcedoc-code-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("source.c"),
        "/*! DOC_CODE sample */\n/* Text [bold] */\nif (a < b && b > c)\n/*! END_CODE */\n",
    )
    .unwrap();
    fs::write(
        directory.join("input.html"),
        "LIST CODE FROM source.c\nDESCRIBE CODE FROM source.c\n",
    )
    .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_sourcedoc"))
        .arg("-d")
        .arg(&directory)
        .arg(directory.join("input.html"))
        .arg(directory.join("output.html"))
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        fs::read_to_string(directory.join("output.html")).unwrap(),
        "<BR><A HREF=\"#sample\">sample</A>\n<H3><A NAME=\"sample\"></A>sample</H3>\nText <B>bold</B>\n<BR><PRE>\nif (a &lt; b &amp;&amp; b &gt; c)\n</PRE>\n"
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sourcedoc_supports_fortran_documentation_continuations() {
    let directory = std::env::temp_dir().join(format!(
        "imod-rs-sourcedoc-fortran-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("source.f"),
        "c! A Fortran @beta\nc!\n      subroutine beta()\n      end\n",
    )
    .unwrap();
    fs::write(
        directory.join("input.html"),
        "LIST FUNCTIONS FROM source.f\nDESCRIBE FUNCTIONS FROM source.f\n",
    )
    .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_sourcedoc"))
        .arg("-f")
        .arg("-d")
        .arg(&directory)
        .arg(directory.join("input.html"))
        .arg(directory.join("output.html"))
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        fs::read_to_string(directory.join("output.html")).unwrap(),
        "<BR>subroutine <A HREF=\"#beta\">beta</A>()\n<H3><A NAME=\"beta\"></A>subroutine beta()</H3><P>\nA Fortran <A HREF=\"#beta\">beta</A>\n</P>\n\n"
    );
    fs::remove_dir_all(directory).unwrap();
}
