//! `IMOD/Etomo/src/etomo/util/PrimativeTokenizer.java`.
//!
//! Description:
//! Creates the following primative tokens: EOL, EOF, ALPHANUM (the largest
//! possible alphanumeric string), SYMBOL (a character matching one of the
//! following: `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).  Everything else is called
//! WHITESPACE and returned as the largest possible string.
//!
//! To Use:
//! construct with a file.
//! call initialize().
//! call next() to get the next token, until the end of file is reached.
//!
//! Testing:
//! Do not call initialize() when testing.
//! Call test() to test this class.
//! Call testStreamTokenizer() to test the StreamTokenizer.
//!
//! Copyright: Copyright 2002 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! The file factories resolve an autodoc directory through
//! `Utilities.getExistingDir(BaseManager, ...)` - which reads the environment of a
//! freshly forked `env`, see `etomo/util/environment_variable.rs` - and then open the
//! file as a `LogFile.Handle`.  `BaseManager` has no module; every caller that reaches
//! here passes a null one, so the parameter is typed `Option<Infallible>`.
//!
//! Java's `FileReader` is a lazy character stream and `closeFile` closes it partway
//! through when the tokenizer hits EOF.  `JavaIoStreamTokenizer` below owns the whole
//! character array instead, so the file is read once in `initializeStreamTokenizer` and
//! the `reader` field only records that a reader was opened - which is all `closeFile`
//! needs, because nothing reads the stream after it closes.
//!
//! **`java.io.StreamTokenizer` is the observable contract**, so it is modelled here
//! rather than approximated - see `JavaIoStreamTokenizer`.  A `java.io.Reader` yields
//! UTF-16 code units, so the reader is a `Vec<u16>`.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::log_file::{self, LogFileError};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::swing::token::{self, Token};
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

/// Java `RETURN`.
const RETURN: &str = "\r";
/// Java `AUTODOC_DIR_ENV_VAR`.
const AUTODOC_DIR_ENV_VAR: &str = "AUTODOC_DIR";
/// Java `DEFAULT_AUTODOC_DIR`.
const DEFAULT_AUTODOC_DIR: &str = "autodoc";

/// Java private static `autodoc_dir`, initialised to null.
static AUTODOC_DIR: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`, read once when the
/// class is initialised.
static DEBUG: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    crate::imod::etomo::etomo_director::ARGUMENTS
        .lock()
        .unwrap()
        .is_debug()
});

/// `java.io.StreamTokenizer`, modelled over the `Vec<u16>` its `java.io.Reader`
/// produces.
///
/// `initializeStreamTokenizer` applies exactly `resetSyntax()`, `wordChars('a','z')`,
/// `wordChars('A','Z')`, `wordChars('0','9')` and `eolIsSignificant(true)`.  That makes
/// four of `nextToken`'s branches provably dead, and they are written below as the
/// source's own guards over a panic rather than as unverified bodies:
///
/// * `CT_DIGIT` is set only by `parseNumbers()`, which `resetSyntax()` undoes and
///   nothing calls again - so the number branch is unreachable;
/// * `CT_QUOTE` is set only by `quoteChar()`, likewise - the quote branch is
///   unreachable;
/// * `CT_COMMENT` is set only by `commentChar()`, likewise - the comment branch is
///   unreachable;
/// * `slashSlashCommentsP`/`slashStarCommentsP` default to false and nothing enables
///   them - the slash branch is unreachable.
///
/// `CT_WHITESPACE` is likewise never set, so even the leading whitespace loop is dead;
/// it is translated in full anyway because it is short and it is what decides EOL.  The
/// consequence, verified against a real JVM, is that `'\n'` is returned as an *ordinary*
/// character whose value, 10, is numerically `TT_EOL` - `TT_EOL` is declared as `'\n'` -
/// while `'\r'` comes back as the ordinary character 13.  That is the bug
/// `saveNextToken`'s "\r" fixup works around.
struct JavaIoStreamTokenizer {
    /// The `java.io.Reader`'s characters, and the read position.
    chars: Vec<u16>,
    pos: usize,
    /// `private char buf[] = new char[20]`.
    buf: Vec<u16>,
    /// `private int peekc = NEED_CHAR`.
    peekc: i32,
    /// `private boolean pushedBack`.
    pushed_back: bool,
    /// `private int LINENO = 1`.
    lineno: i32,
    /// `private boolean forceLower`.
    force_lower: bool,
    /// `private boolean eolIsSignificantP = false`.
    eol_is_significant_p: bool,
    /// `private boolean slashSlashCommentsP = false`.
    slash_slash_comments_p: bool,
    /// `private boolean slashStarCommentsP = false`.
    slash_star_comments_p: bool,
    /// `private byte ctype[] = new byte[256]`.
    ctype: [u8; 256],
    /// `public int ttype = TT_NOTHING`.
    ttype: i32,
    /// `public String sval`.
    sval: Option<String>,
    /// `public double nval`.
    nval: f64,
}

impl JavaIoStreamTokenizer {
    /// `private static final int NEED_CHAR = Integer.MAX_VALUE`.
    const NEED_CHAR: i32 = i32::MAX;
    /// `private static final int SKIP_LF = Integer.MAX_VALUE - 1`.
    const SKIP_LF: i32 = i32::MAX - 1;
    /// `private static final byte CT_WHITESPACE = 1`.
    const CT_WHITESPACE: u8 = 1;
    /// `private static final byte CT_DIGIT = 2`.
    const CT_DIGIT: u8 = 2;
    /// `private static final byte CT_ALPHA = 4`.
    const CT_ALPHA: u8 = 4;
    /// `private static final byte CT_QUOTE = 8`.
    const CT_QUOTE: u8 = 8;
    /// `private static final byte CT_COMMENT = 16`.
    const CT_COMMENT: u8 = 16;
    /// `public static final int TT_EOF = -1`.
    const TT_EOF: i32 = -1;
    /// `public static final int TT_EOL = '\n'`.
    const TT_EOL: i32 = '\n' as i32;
    /// `public static final int TT_NUMBER = -2`.
    const TT_NUMBER: i32 = -2;
    /// `public static final int TT_WORD = -3`.
    const TT_WORD: i32 = -3;
    /// `private static final int TT_NOTHING = -4`.
    const TT_NOTHING: i32 = -4;

    /// `StreamTokenizer(Reader)`, which runs the private no-argument constructor's
    /// default syntax first.
    fn new(chars: Vec<u16>) -> JavaIoStreamTokenizer {
        let mut tokenizer = JavaIoStreamTokenizer {
            chars,
            pos: 0,
            buf: vec![0; 20],
            peekc: Self::NEED_CHAR,
            pushed_back: false,
            lineno: 1,
            force_lower: false,
            eol_is_significant_p: false,
            slash_slash_comments_p: false,
            slash_star_comments_p: false,
            ctype: [0; 256],
            ttype: Self::TT_NOTHING,
            sval: None,
            nval: 0.0,
        };
        tokenizer.word_chars('a' as i32, 'z' as i32);
        tokenizer.word_chars('A' as i32, 'Z' as i32);
        tokenizer.word_chars(128 + 32, 255);
        tokenizer.whitespace_chars(0, ' ' as i32);
        tokenizer.comment_char('/' as i32);
        tokenizer.quote_char('"' as i32);
        tokenizer.quote_char('\'' as i32);
        tokenizer.parse_numbers();
        tokenizer
    }

    /// `resetSyntax()`.
    fn reset_syntax(&mut self) {
        let mut i = self.ctype.len();
        while i > 0 {
            i -= 1;
            self.ctype[i] = 0;
        }
    }

    /// `wordChars(int, int)`.
    fn word_chars(&mut self, low: i32, hi: i32) {
        let mut low = low;
        let mut hi = hi;
        if low < 0 {
            low = 0;
        }
        if hi >= self.ctype.len() as i32 {
            hi = self.ctype.len() as i32 - 1;
        }
        while low <= hi {
            self.ctype[low as usize] |= Self::CT_ALPHA;
            low += 1;
        }
    }

    /// `whitespaceChars(int, int)`.
    fn whitespace_chars(&mut self, low: i32, hi: i32) {
        let mut low = low;
        let mut hi = hi;
        if low < 0 {
            low = 0;
        }
        if hi >= self.ctype.len() as i32 {
            hi = self.ctype.len() as i32 - 1;
        }
        while low <= hi {
            self.ctype[low as usize] = Self::CT_WHITESPACE;
            low += 1;
        }
    }

    /// `commentChar(int)`.
    fn comment_char(&mut self, ch: i32) {
        if (0..self.ctype.len() as i32).contains(&ch) {
            self.ctype[ch as usize] = Self::CT_COMMENT;
        }
    }

    /// `quoteChar(int)`.
    fn quote_char(&mut self, ch: i32) {
        if (0..self.ctype.len() as i32).contains(&ch) {
            self.ctype[ch as usize] = Self::CT_QUOTE;
        }
    }

    /// `parseNumbers()`.
    fn parse_numbers(&mut self) {
        for i in '0' as usize..='9' as usize {
            self.ctype[i] |= Self::CT_DIGIT;
        }
        self.ctype['.' as usize] |= Self::CT_DIGIT;
        self.ctype['-' as usize] |= Self::CT_DIGIT;
    }

    /// `eolIsSignificant(boolean)`.
    fn eol_is_significant(&mut self, flag: bool) {
        self.eol_is_significant_p = flag;
    }

    /// `lineno()`.
    fn lineno(&self) -> i32 {
        self.lineno
    }

    /// The `Reader.read()` the tokenizer calls: the next UTF-16 code unit, or -1.
    fn read(&mut self) -> i32 {
        if self.pos >= self.chars.len() {
            return -1;
        }
        let c = self.chars[self.pos] as i32;
        self.pos += 1;
        c
    }

    /// `nextToken()`.
    fn next_token(&mut self) -> i32 {
        if self.pushed_back {
            self.pushed_back = false;
            return self.ttype;
        }
        self.sval = None;

        let mut c = self.peekc;
        if c < 0 {
            c = Self::NEED_CHAR;
        }
        if c == Self::SKIP_LF {
            c = self.read();
            if c < 0 {
                self.ttype = Self::TT_EOF;
                return self.ttype;
            }
            if c == '\n' as i32 {
                c = Self::NEED_CHAR;
            }
        }
        if c == Self::NEED_CHAR {
            c = self.read();
            if c < 0 {
                self.ttype = Self::TT_EOF;
                return self.ttype;
            }
        }
        // Just to be safe
        self.ttype = c;

        // Set peekc so that the next invocation of nextToken will read another character
        // unless peekc is reset in this invocation
        self.peekc = Self::NEED_CHAR;

        let mut ctype = if c < 256 {
            self.ctype[c as usize]
        } else {
            Self::CT_ALPHA
        };
        while (ctype & Self::CT_WHITESPACE) != 0 {
            if c == '\r' as i32 {
                self.lineno += 1;
                if self.eol_is_significant_p {
                    self.peekc = Self::SKIP_LF;
                    self.ttype = Self::TT_EOL;
                    return self.ttype;
                }
                c = self.read();
                if c == '\n' as i32 {
                    c = self.read();
                }
            } else {
                if c == '\n' as i32 {
                    self.lineno += 1;
                    if self.eol_is_significant_p {
                        self.ttype = Self::TT_EOL;
                        return self.ttype;
                    }
                }
                c = self.read();
            }
            if c < 0 {
                self.ttype = Self::TT_EOF;
                return self.ttype;
            }
            ctype = if c < 256 {
                self.ctype[c as usize]
            } else {
                Self::CT_ALPHA
            };
        }

        if (ctype & Self::CT_DIGIT) != 0 {
            // Unreachable: CT_DIGIT is set only by parseNumbers(), which resetSyntax()
            // undid and which nothing calls again.  See the type's doc comment.
            unreachable!("StreamTokenizer number branch with parseNumbers() not in effect");
        }

        if (ctype & Self::CT_ALPHA) != 0 {
            let mut i = 0usize;
            loop {
                if i >= self.buf.len() {
                    self.buf.resize(self.buf.len() * 2, 0);
                }
                self.buf[i] = c as u16;
                i += 1;
                c = self.read();
                ctype = if c < 0 {
                    Self::CT_WHITESPACE
                } else if c < 256 {
                    self.ctype[c as usize]
                } else {
                    Self::CT_ALPHA
                };
                if (ctype & (Self::CT_ALPHA | Self::CT_DIGIT)) == 0 {
                    break;
                }
            }
            self.peekc = c;
            self.sval = Some(String::from_utf16_lossy(&self.buf[0..i]));
            if self.force_lower {
                self.sval = Some(token::convert_to_key(self.sval.as_ref().unwrap()));
            }
            self.ttype = Self::TT_WORD;
            return self.ttype;
        }

        if (ctype & Self::CT_QUOTE) != 0 {
            // Unreachable: CT_QUOTE is set only by quoteChar().  See the doc comment.
            unreachable!("StreamTokenizer quote branch with no quote character declared");
        }

        if ctype == Self::CT_COMMENT {
            // Unreachable: CT_COMMENT is set only by commentChar().  See the doc comment.
            unreachable!("StreamTokenizer comment branch with no comment character declared");
        }

        if self.slash_slash_comments_p || self.slash_star_comments_p {
            // Unreachable: both flags default to false and nothing enables them.
            unreachable!("StreamTokenizer slash-comment branch with slash comments off");
        }

        self.ttype = c;
        self.ttype
    }

    /// `toString()`.  Note that the `TT_EOL` case matches an ordinary `'\n'` too,
    /// because `TT_EOL` is declared as `'\n'`.
    fn to_string(&self) -> String {
        let ret: String = if self.ttype == Self::TT_EOF {
            "EOF".to_string()
        } else if self.ttype == Self::TT_EOL {
            "EOL".to_string()
        } else if self.ttype == Self::TT_WORD {
            match &self.sval {
                None => "null".to_string(),
                Some(sval) => sval.clone(),
            }
        } else if self.ttype == Self::TT_NUMBER {
            format!(
                "n={}",
                crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                    self.nval
                )
            )
        } else if self.ttype == Self::TT_NOTHING {
            "NOTHING".to_string()
        } else {
            if self.ttype < 256 && (self.ctype[self.ttype as usize] & Self::CT_QUOTE) != 0 {
                match &self.sval {
                    None => "null".to_string(),
                    Some(sval) => sval.clone(),
                }
            } else {
                let s: [u16; 3] = ['\'' as u16, self.ttype as u16, '\'' as u16];
                String::from_utf16_lossy(&s)
            }
        };
        format!("Token[{}], line {}", ret, self.lineno)
    }
}

/// Java `PrimativeTokenizer`.  The class is final.
pub struct PrimativeTokenizer {
    /// Java field `separateAlphabeticAndNumeric`.
    separate_alphabetic_and_numeric: bool,
    /// Java field `string`.
    string: Option<String>,
    /// Java field `logFile`, a `LogFile.Handle`.
    log_file: Option<std::sync::Arc<log_file::Handle>>,
    /// Java field `readingId`, a `LogFile.ReadingId`, initialised to null.
    reading_id: Option<log_file::ReadingId>,
    /// Java field `streamTokenizer`.
    stream_tokenizer: Option<JavaIoStreamTokenizer>,
    /// Java field `symbols`.
    symbols: String,
    /// Java field `digits`.
    digits: String,
    /// Java field `letters`.
    letters: String,
    /// Java field `token`.
    token: Token,
    /// Java field `nextStreamTokenFound`.
    next_stream_token_found: bool,
    /// Java field `savedToken`.
    saved_token: Token,
    /// Java field `reader`.  The character stream itself is owned by the
    /// `StreamTokenizer`, as the `Reader` is in Java; this records only that a reader
    /// was opened, which is all `closeFile` needs.
    reader: Option<()>,
    /// Java field `fileClosed`.
    file_closed: bool,
    /// Java field `streamTokenizerNothingValue`.
    stream_tokenizer_nothing_value: i32,
    /// Java field `debug`.
    debug: bool,
    /// Java field `valueBeingBrokenUp`.
    value_being_broken_up: Option<String>,
    /// Java field `valueIndex`.
    value_index: i32,
    /// Java field `peekedAtToken`.
    peeked_at_token: bool,
}

impl PrimativeTokenizer {
    /// Java `PrimativeTokenizer(LogFile.Handle, String, boolean, boolean)`, the private
    /// constructor.
    fn new(
        log_file: Option<std::sync::Arc<log_file::Handle>>,
        string: Option<&str>,
        separate_alphabetic_and_numeric: bool,
        debug: bool,
    ) -> PrimativeTokenizer {
        PrimativeTokenizer {
            log_file,
            string: string.map(|string| string.to_string()),
            separate_alphabetic_and_numeric,
            debug,
            reading_id: None,
            stream_tokenizer: None,
            symbols: "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".to_string(),
            digits: "0123456789".to_string(),
            letters: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".to_string(),
            token: Token::new(),
            next_stream_token_found: false,
            saved_token: Token::new(),
            reader: None,
            file_closed: false,
            stream_tokenizer_nothing_value: 0,
            value_being_broken_up: None,
            value_index: -1,
            peeked_at_token: false,
        }
    }

    /// Java `getAutodocInstance`.
    pub fn get_autodoc_instance(
        name: Option<&str>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        debug: bool,
    ) -> PrimativeTokenizer {
        let mut autodoc_dir = AUTODOC_DIR.lock().unwrap();
        if autodoc_dir.is_none() {
            *autodoc_dir = PrimativeTokenizer::get_file_location(
                None,
                None,
                manager,
                axis_id,
                not_found_message,
                name,
            );
        }
        let autodoc_dir = autodoc_dir.clone();
        PrimativeTokenizer::new(
            PrimativeTokenizer::get_log_file(
                manager,
                axis_id,
                autodoc_dir.as_deref(),
                name,
                not_found_message,
                false,
            ),
            None,
            false,
            debug,
        )
    }

    /// Java `getGenericInstance`.
    ///
    /// Gets an autodoc of an unknown type.  `AutodocFile` takes precedence over
    /// `envVar`, `subdirName`, and `name`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_generic_instance(
        env_var: Option<&str>,
        subdir_name: Option<&str>,
        name: Option<&str>,
        autodoc_file: Option<&std::path::Path>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        debug: bool,
        writable: bool,
    ) -> PrimativeTokenizer {
        let mut name = name.map(|name| name.to_string());
        let dir: Option<std::path::PathBuf>;
        if let Some(autodoc_file) = autodoc_file {
            // `java.io.File.getParentFile()` is null when the name has no directory part.
            dir = match utilities::java_io_file_get_parent(&autodoc_file.to_string_lossy()) {
                Some(parent) => Some(std::path::PathBuf::from(parent)),
                None => {
                    // The file did not include a directory path. Use absolute path
                    // instead of parent to set the directory.
                    let path =
                        utilities::java_io_file_get_absolute_path(&autodoc_file.to_string_lossy());
                    match path.rfind(std::path::MAIN_SEPARATOR) {
                        None => None,
                        Some(index) => Some(std::path::PathBuf::from(&path[0..index])),
                    }
                }
            };
            name = Some(utilities::java_io_file_get_name(
                &autodoc_file.to_string_lossy(),
            ));
        } else {
            dir = PrimativeTokenizer::get_file_location(
                env_var,
                subdir_name,
                manager,
                axis_id,
                not_found_message,
                name.as_deref(),
            );
        }
        PrimativeTokenizer::new(
            PrimativeTokenizer::get_log_file(
                manager,
                axis_id,
                dir.as_deref(),
                name.as_deref(),
                not_found_message,
                writable,
            ),
            None,
            false,
            debug,
        )
    }

    /// Java `getStringInstance`.
    pub fn get_string_instance(string: &str, debug: bool) -> PrimativeTokenizer {
        PrimativeTokenizer::new(None, Some(string), false, debug)
    }

    /// Java `getNumericStringInstance`.
    pub fn get_numeric_string_instance(string: &str, debug: bool) -> PrimativeTokenizer {
        PrimativeTokenizer::new(None, Some(string), true, debug)
    }

    /// Java private static `getFileLocation`.  Return the location of the autodoc.
    fn get_file_location(
        env_var: Option<&str>,
        subdir: Option<&str>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
        name: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        let find_autodoc_dir = env_var.is_none() && subdir.is_none();
        let mut dir;
        if find_autodoc_dir {
            dir = PrimativeTokenizer::get_directory(
                Some(AUTODOC_DIR_ENV_VAR),
                None,
                manager,
                axis_id,
                not_found_message,
            );
            if dir.is_none() {
                dir = PrimativeTokenizer::get_directory(
                    Some(etomo_director::IMOD_DIR_ENV_VAR),
                    Some(DEFAULT_AUTODOC_DIR),
                    manager,
                    axis_id,
                    not_found_message,
                );
            }
        } else {
            dir = PrimativeTokenizer::get_directory(
                env_var,
                subdir,
                manager,
                axis_id,
                not_found_message,
            );
        }
        if dir.is_none() && *DEBUG {
            // The source's conditional binds tighter than the concatenation, so the
            // whole message after "Info" is dropped when `notFoundMessage` is null.
            eprintln!(
                "{}",
                match not_found_message {
                    None => "Warning".to_string(),
                    Some(_) => format!(
                        "Info:  can't open the {} autodoc file.\nThis autodoc was not in in ${}{}{}.\n",
                        name.unwrap_or("null"),
                        env_var.unwrap_or("null"),
                        match subdir {
                            None => "".to_string(),
                            Some(subdir) => "/".to_string() + subdir,
                        },
                        if find_autodoc_dir {
                            " or $".to_string() + AUTODOC_DIR_ENV_VAR
                        } else {
                            "".to_string()
                        }
                    ),
                }
            );
        }
        dir
    }

    /// Java private static `getDirectory`.  Return the directory defined by `envVar` and
    /// `subdir`.
    fn get_directory(
        env_var: Option<&str>,
        subdir: Option<&str>,
        manager: Option<&'static dyn BaseManager>,
        axis_id: AxisID,
        not_found_message: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        let replacement_dir = autodoc_factory::get_replacement_dir();
        if let Some(replacement_dir) = replacement_dir {
            if subdir.is_none() {
                return Some(std::path::PathBuf::from(replacement_dir));
            }
            return Some(std::path::PathBuf::from(replacement_dir).join(subdir.unwrap()));
        }
        if env_var.is_some() {
            if subdir.is_none() {
                return utilities::get_existing_dir_message(
                    manager,
                    env_var,
                    axis_id,
                    not_found_message,
                );
            }
            return PrimativeTokenizer::get_dir(manager, env_var, subdir, axis_id);
        }
        None
    }

    /// Java private static `getLogFile`.  Gets the autodoc file as a `LogFile`.
    fn get_log_file(
        manager: Option<&'static dyn BaseManager>,
        _axis_id: AxisID,
        autodoc_dir: Option<&std::path::Path>,
        autodoc_name: Option<&str>,
        not_found_message: Option<&str>,
        writable: bool,
    ) -> Option<std::sync::Arc<log_file::Handle>> {
        let warn_if_fail = not_found_message.is_none();
        let autodoc_dir = autodoc_dir?;
        autodoc_name?;
        let file = dataset_files::get_autodoc(autodoc_dir, autodoc_name);
        let error_message_tag = if warn_if_fail { "Warning" } else { "Info" };
        let exists = file.exists();
        if !writable {
            if !exists {
                return None;
            }
            if std::fs::File::open(&file).is_err() {
                if *DEBUG {
                    eprintln!(
                        "{}:  Cannot read the autodoc file {}.",
                        error_message_tag,
                        utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                    );
                }
                return None;
            }
        } else if exists && std::fs::OpenOptions::new().write(true).open(&file).is_err() {
            if *DEBUG {
                eprintln!(
                    "{}:  Cannot write the autodoc file {}.",
                    error_message_tag,
                    utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                );
            }
            return None;
        }
        if file.is_dir() {
            if *DEBUG {
                eprintln!(
                    "{}:  The autodoc file {} is a directory.",
                    error_message_tag,
                    utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                );
            }
            return None;
        }

        // `manager != null ? manager.getEmergencyMonitor(axisID) : null`; `manager` is
        // always null here.
        match log_file::LogFile::get_instance_file(Some(&file), None) {
            Ok(handle) => Some(handle),
            Err(e) => {
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                if *DEBUG {
                    eprintln!(
                        "{}:  Cannot open the autodoc file {}.",
                        error_message_tag,
                        utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                    );
                }
                None
            }
        }
    }

    /// Java private static `getDir`.
    fn get_dir(
        manager: Option<&'static dyn BaseManager>,
        env_variable: Option<&str>,
        dir_name: Option<&str>,
        axis_id: AxisID,
    ) -> Option<std::path::PathBuf> {
        let parent_dir = utilities::get_existing_dir(manager, env_variable, axis_id)?;
        let dir = parent_dir.join(dir_name.unwrap_or("null"));
        if !utilities::check_existing_dir(&dir, env_variable) {
            return None;
        }
        Some(dir)
    }

    /// Java `getLogFile()`.
    pub fn get_log_file_handle(&self) -> Option<std::sync::Arc<log_file::Handle>> {
        self.log_file.clone()
    }

    /// Java `getToken`.  Returns the current token.
    pub fn get_token(&self) -> &Token {
        &self.token
    }

    /// Java `initialize`.  Must be called before the first call to next().
    pub fn initialize(&mut self) -> Result<(), LogFileError> {
        self.initialize_stream_tokenizer()?;
        self.next_stream_token();
        Ok(())
    }

    /// Java `initializeStreamTokenizer`.
    fn initialize_stream_tokenizer(&mut self) -> Result<(), LogFileError> {
        let chars: Vec<u16> = if let Some(log_file) = self.log_file.clone() {
            let reading_file = std::path::PathBuf::from(log_file.get_absolute_path());
            self.reading_id = match log_file.open_for_reading() {
                Ok(reading_id) => reading_id,
                Err(e) => {
                    // Java's `catch (FileNotFoundException)` closes the reading id and
                    // rethrows; `openForReading` has not returned one yet.
                    return Err(e);
                }
            };
            // `new FileReader(readingFile)`.  See the module header: the whole file is
            // read here instead of lazily.
            match std::fs::read(&reading_file) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).encode_utf16().collect(),
                Err(e) => {
                    if let Some(reading_id) = &self.reading_id {
                        if !reading_id.is_empty() {
                            log_file.close_id(Some(reading_id));
                        }
                    }
                    return Err(LogFileError::Io(e));
                }
            }
        } else if let Some(string) = &self.string {
            string.encode_utf16().collect()
        } else {
            "".encode_utf16().collect()
        };
        self.reader = Some(());
        let stream_tokenizer = JavaIoStreamTokenizer::new(chars);
        self.stream_tokenizer_nothing_value = stream_tokenizer.ttype;
        self.stream_tokenizer = Some(stream_tokenizer);
        let stream_tokenizer = self.stream_tokenizer.as_mut().unwrap();
        stream_tokenizer.reset_syntax();
        stream_tokenizer.word_chars('a' as i32, 'z' as i32);
        stream_tokenizer.word_chars('A' as i32, 'Z' as i32);
        stream_tokenizer.word_chars('0' as i32, '9' as i32);
        stream_tokenizer.eol_is_significant(true);
        Ok(())
    }

    /// Java `peek`.
    ///
    /// Get the next token without moving on.  Will return the same thing if called
    /// multiple times in a row.  Places a deep copy in the return token, and returns it.
    /// If it's null, constructs a deep copy and returns it.
    ///
    /// # Safety
    /// `return_token` must be null or point to a live `Token`.  When it is null the
    /// returned pointer owns a heap `Token` the caller must reclaim with
    /// `Box::from_raw`, where the source's owner is the GC.
    pub unsafe fn peek(&mut self, return_token: *mut Token) -> *mut Token {
        if self.peeked_at_token {
            // The token has been peeked at before. Just keep returning it.
            if return_token.is_null() {
                return Box::into_raw(Box::new(Token::new_from_token(&self.token)));
            }
            unsafe { (*return_token).copy(&self.token) };
            return return_token;
        }
        self.peeked_at_token = true;
        unsafe { self.save_next_token(return_token) }
    }

    /// Java `next`.
    ///
    /// Get the next token.  If the next token has been peeked at, it's already been
    /// saved.
    ///
    /// # Safety
    /// See `peek`.
    pub unsafe fn next(&mut self, return_token: *mut Token) -> *mut Token {
        if self.peeked_at_token {
            // The token has already been saved.
            self.peeked_at_token = false;
            if return_token.is_null() {
                return Box::into_raw(Box::new(Token::new_from_token(&self.token)));
            }
            unsafe { (*return_token).copy(&self.token) };
            return return_token;
        }
        unsafe { self.save_next_token(return_token) }
    }

    /// Java `saveNextToken`.
    ///
    /// Saves the next token.  Returns a deep copy of it.
    ///
    /// # Safety
    /// See `peek`.
    unsafe fn save_next_token(&mut self, return_token: *mut Token) -> *mut Token {
        if self.value_being_broken_up.is_none() {
            if self.file_closed {
                if return_token.is_null() {
                    return Box::into_raw(Box::new(Token::new_from_token(&self.token)));
                }
                unsafe { (*return_token).copy(&self.token) };
                return return_token;
            }
            let mut found = false;
            self.token.reset();
            let mut whitespace_found = false;
            let mut whitespace_buffer: Option<Vec<u16>> = None;
            let mut return_found = false;

            if self.next_stream_token_found {
                found = true;
                let saved_token = std::mem::replace(&mut self.saved_token, Token::new());
                self.token.copy(&saved_token);
                self.saved_token = saved_token;
                self.saved_token.reset();
                self.next_stream_token_found = false;
                self.next_stream_token();
            }
            while !found {
                let ttype = self.stream_tokenizer.as_ref().unwrap().ttype;
                if ttype == JavaIoStreamTokenizer::TT_EOF {
                    self.token.set_type(token::Type::Eof);
                    if self.debug {
                        println!("{}", self.token);
                    }
                    found = true;
                    return_found = false;
                    if self.log_file.is_some() {
                        self.close_file();
                    }
                } else if ttype == JavaIoStreamTokenizer::TT_EOL {
                    self.token.set_type(token::Type::Eol);
                    if self.debug {
                        println!("{}", self.token);
                    }
                    found = true;
                    // If "\r" was found before an EOL, roll it into the EOL. This is
                    // necessary because StreamTokenizer is not working according to its
                    // definition: It is supposed to return both "\n" and "\r\n" as EOL,
                    // but it does not do this for "\r\n". If this bug is fixed, then
                    // "\r\r\n" will appear as an EOL, but this is OK because this is
                    // character string that usually means that there was an error in
                    // transfering the file between Windows and Linux.
                    if return_found && whitespace_found {
                        return_found = false;
                        let buffer = whitespace_buffer.as_mut().unwrap();
                        buffer.remove(buffer.len() - 1);
                        if buffer.is_empty() {
                            whitespace_found = false;
                            whitespace_buffer = None;
                        }
                    }
                } else if ttype == JavaIoStreamTokenizer::TT_WORD {
                    let sval = self
                        .stream_tokenizer
                        .as_ref()
                        .unwrap()
                        .sval
                        .clone()
                        .unwrap_or_else(|| panic!("java.lang.NullPointerException"));
                    self.token.set_type_and_string(token::Type::Alphanum, &sval);
                    if self.debug && !self.separate_alphabetic_and_numeric {
                        println!("{}", self.token);
                    }
                    found = true;
                    return_found = false;
                } else if self.symbols.chars().any(|ch| ch as i32 == ttype) {
                    // `symbols.indexOf(streamTokenizer.ttype) != -1`; `String.indexOf(int
                    // ch)` searches for the code point `ch` and returns -1 for any value
                    // that is not one.

                    self.token
                        .set_type_and_char(token::Type::Symbol, ttype as u16);
                    if self.debug {
                        println!("{}", self.token);
                    }
                    found = true;
                    return_found = false;
                } else {
                    // `RETURN.indexOf(streamTokenizer.ttype) != -1`.
                    return_found = RETURN.chars().any(|ch| ch as i32 == ttype);
                    if !whitespace_found {
                        whitespace_found = true;
                        whitespace_buffer = Some(vec![ttype as u16]);
                    } else {
                        whitespace_buffer.as_mut().unwrap().push(ttype as u16);
                    }
                }
                if found && whitespace_found {
                    self.next_stream_token_found = true;
                    let token = std::mem::replace(&mut self.token, Token::new());
                    self.saved_token.copy(&token);
                    self.token = token;
                    self.token.set_type_and_string_buffer(
                        token::Type::Whitespace,
                        &String::from_utf16_lossy(whitespace_buffer.as_ref().unwrap()),
                    );
                    if self.debug {
                        println!("{}", self.token);
                    }
                    whitespace_buffer = None;
                    whitespace_found = false;
                }
                if !self.next_stream_token_found {
                    self.next_stream_token();
                }
            }
        }
        if self.separate_alphabetic_and_numeric {
            let token = std::mem::replace(&mut self.token, Token::new());
            self.token = self.separate_alphabetic_and_numeric(token);
        }
        if return_token.is_null() {
            return Box::into_raw(Box::new(Token::new_from_token(&self.token)));
        }
        unsafe { (*return_token).copy(&self.token) };
        return_token
    }

    /// Java `separateAlphabeticAndNumeric`.
    ///
    /// Take an ALPHANUM token and break it up.  Called multiple times.
    /// ValueBeingBrokenUp and valueIndex are preserve outside the function and
    /// managed inside the function.
    fn separate_alphabetic_and_numeric(&mut self, token: Token) -> Token {
        if self.value_being_broken_up.is_none() && !token.is(token::Type::Alphanum) {
            // Only ALPHANUM tokens are broken up.
            return token;
        }
        if self.value_being_broken_up.is_none() {
            // Start breaking up a token.
            self.value_being_broken_up = token.get_value().map(|value| value.to_string());
            self.value_index = 0;
        }
        // Grab an as large as possible NUMERIC or ALPHABETIC token starting at
        // valueIndex.
        let mut new_token = Token::new();
        let value_being_broken_up: Vec<u16> = match &self.value_being_broken_up {
            None => panic!("java.lang.NullPointerException"),
            Some(value) => value.encode_utf16().collect(),
        };
        let digits: Vec<u16> = self.digits.encode_utf16().collect();
        let letters: Vec<u16> = self.letters.encode_utf16().collect();
        let mut i = self.value_index as usize;
        while i < value_being_broken_up.len() {
            let ch = value_being_broken_up[i];
            if digits.contains(&ch) {
                // A NUMERIC token.
                if new_token.is(token::Type::Null) {
                    // Start the NUMERIC token.
                    new_token.set_type(token::Type::Numeric);
                } else if new_token.is(token::Type::Alphabetic) {
                    // Found the end of the NUMERIC token. Add the value to the new token
                    // and return.
                    new_token.set_string(&String::from_utf16_lossy(
                        &value_being_broken_up[self.value_index as usize..i],
                    ));
                    if self.debug {
                        println!("{}", new_token);
                    }
                    self.value_index = i as i32;
                    return new_token;
                }
            } else if letters.contains(&ch) {
                // An ALPHABETIC token
                if new_token.is(token::Type::Null) {
                    // Start the ALPHABETIC token.
                    new_token.set_type(token::Type::Alphabetic);
                } else if new_token.is(token::Type::Numeric) {
                    // Found the end of the ALPHABETIC token. Add the value to the new
                    // token and return.
                    new_token.set_string(&String::from_utf16_lossy(
                        &value_being_broken_up[self.value_index as usize..i],
                    ));
                    if self.debug {
                        println!("{}", new_token);
                    }
                    self.value_index = i as i32;
                    return new_token;
                }
            }
            i += 1;
        }
        // Found the end of the last token in valueBeingBrokenUp. Add the value to
        // the new token and reset valueBeingBrokenUp and valueIndex.
        new_token.set_string(&String::from_utf16_lossy(
            &value_being_broken_up[self.value_index as usize..value_being_broken_up.len()],
        ));
        if self.debug {
            println!("{}", new_token);
        }
        self.value_being_broken_up = None;
        self.value_index = -1;
        new_token
    }

    /// Java `getSymbols`.
    pub fn get_symbols(&self) -> &str {
        &self.symbols
    }

    /// Java `setDebug`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `test`.
    ///
    /// Tests this object.  Prints result to System.out.  `tokens`: if true, prints each
    /// token.  If false, prints the text.
    pub fn test(&mut self, tokens: bool) {
        self.initialize();
        loop {
            let token = unsafe { Box::from_raw(self.next(std::ptr::null_mut())) };
            if tokens {
                println!("{}", token);
            } else if token.is(token::Type::Eol) {
                println!();
            } else if !token.is(token::Type::Eof) {
                print!("{}", token.get_value().unwrap_or("null"));
            }
            if token.is(token::Type::Eof) {
                break;
            }
        }
    }

    /// Java `testStreamTokenizer`.
    ///
    /// Tests the StreamTokenizer.  Prints result to System.out.  `tokens`: if true,
    /// prints each token.  If false, prints the text.
    pub fn test_stream_tokenizer(&mut self, tokens: bool, details: bool) {
        self.initialize_stream_tokenizer();
        loop {
            self.next_stream_token();
            let stream_tokenizer = self.stream_tokenizer.as_ref().unwrap();
            if tokens {
                print!("{}", stream_tokenizer.to_string());
                if details {
                    let token_type = TokenType::get_instance(stream_tokenizer.ttype);
                    match token_type {
                        None => println!(
                            ", {},sval={},nval={}",
                            String::from_utf16_lossy(&[stream_tokenizer.ttype as u16]),
                            match &stream_tokenizer.sval {
                                None => "null".to_string(),
                                Some(sval) => sval.clone(),
                            },
                            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                                stream_tokenizer.nval
                            )
                        ),
                        Some(token_type) => println!(
                            ", {},sval={},nval={}",
                            token_type,
                            match &stream_tokenizer.sval {
                                None => "null".to_string(),
                                Some(sval) => sval.clone(),
                            },
                            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                                stream_tokenizer.nval
                            )
                        ),
                    }
                } else {
                    println!();
                }
            } else if stream_tokenizer.ttype == JavaIoStreamTokenizer::TT_EOL {
                println!();
            } else if stream_tokenizer.ttype != JavaIoStreamTokenizer::TT_EOF {
                if stream_tokenizer.ttype == JavaIoStreamTokenizer::TT_WORD {
                    print!(
                        "{}",
                        match &stream_tokenizer.sval {
                            None => "null".to_string(),
                            Some(sval) => sval.clone(),
                        }
                    );
                } else {
                    print!(
                        "{}",
                        String::from_utf16_lossy(&[stream_tokenizer.ttype as u16])
                    );
                }
            }
            let ttype = self.stream_tokenizer.as_ref().unwrap().ttype;
            if ttype == JavaIoStreamTokenizer::TT_EOF
                || ttype == self.stream_tokenizer_nothing_value
            {
                break;
            }
        }
        if self.log_file.is_some() {
            self.close_file();
        }
        println!();
    }

    /// Java `nextStreamToken`.
    fn next_stream_token(&mut self) {
        if self.log_file.is_some() && self.file_closed {
            return;
        }
        self.stream_tokenizer.as_mut().unwrap().next_token();
    }

    /// Java `closeFile`.
    fn close_file(&mut self) {
        if let Some(log_file) = self.log_file.clone() {
            if let Some(reading_id) = self.reading_id.clone() {
                if !reading_id.is_empty() {
                    self.file_closed = true;
                    // `reader.close()`; the character array is owned by the
                    // `StreamTokenizer`, see the module header.
                    self.reader = None;
                    log_file.close_id(Some(&reading_id));
                    self.reading_id = None;
                }
            }
        }
    }
}

/// Java's nested `private static final class TokenType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenType {
    /// Java `EOF`, name "TT_EOF".
    Eof,
    /// Java `EOL`, name "TT_EOL".
    Eol,
    /// Java `NUMBER`, name "TT_NUMBER".
    Number,
    /// Java `WORD`, name "TT_WORD".
    Word,
    /// Java `NOTHING`, name "TT_NOTHING".
    Nothing,
}

impl TokenType {
    /// Java `getInstance`.  Note that the `TT_EOL` case is `'\n'`, so an ordinary
    /// newline character maps to `EOL` here too.
    fn get_instance(ttype: i32) -> Option<TokenType> {
        match ttype {
            JavaIoStreamTokenizer::TT_EOF => Some(TokenType::Eof),
            JavaIoStreamTokenizer::TT_EOL => Some(TokenType::Eol),
            JavaIoStreamTokenizer::TT_NUMBER => Some(TokenType::Number),
            JavaIoStreamTokenizer::TT_WORD => Some(TokenType::Word),
            // StreamTokenizer.TT_NOTHING
            -4 => Some(TokenType::Nothing),
            _ => None,
        }
    }
}

/// Java `TokenType.toString`.  Returns `name`.
impl std::fmt::Display for TokenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Eof => "TT_EOF",
            Self::Eol => "TT_EOL",
            Self::Number => "TT_NUMBER",
            Self::Word => "TT_WORD",
            Self::Nothing => "TT_NOTHING",
        })
    }
}
