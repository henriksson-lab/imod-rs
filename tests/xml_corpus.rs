//! The mini-XML byte-parity corpus (`fixtures/xml`).
//!
//! `src/imod/libxml` is a translation of the vendored mini-XML, and what the
//! acceptance target cares about is the bytes it *writes*.  `fixtures/xml`
//! holds 139 XML autodocs that native IMOD produced from `IMOD/autodoc/*.adoc`,
//! three hand-written documents covering entities, quoting, CDATA and non-ASCII
//! text, and — the parity objects — the 133 files **native mini-XML** wrote
//! when it loaded each of them with `MXML_OPAQUE_CALLBACK` and saved them
//! through `mxmlSaveFile` with IMOD's whitespace callback and a wrap margin of
//! zero.  See `fixtures/xml/README.md`.
//!
//! This ran only as an env-gated report generator before 2026-09-20, against a
//! corpus that lived in a session scratchpad.  It is an ordinary gate test now:
//! the native side is checked in as bytes, so neither a reference build nor an
//! environment variable is needed.
//!
//! Ten of the inputs have no saved counterpart, and that is also the contract:
//! `AdocWrite` emits element names that are not XML names (`<^   critical_dose>`),
//! so native cannot reparse its own output for those ten.  The translation must
//! reject exactly the same ones.

mod common;

use imod_rs::imod::libcfshr::autodoc::{adoc_clear, adoc_read, adoc_set_write_as_xml, adoc_write};
use imod_rs::imod::libcfshr::b3dutil::ImodFile;
use imod_rs::imod::libcfshr::mxmlwrap::{ixml_reset_last_level, ixml_whitespace_cb};
use imod_rs::imod::libxml::MXML_NO_PARENT;
use imod_rs::imod::libxml::MxmlArena;
use imod_rs::imod::libxml::mxml_file::{mxml_load_file, mxml_save_file, mxml_set_wrap_margin};
use imod_rs::imod::libxml::mxml_node::mxml_delete;
use imod_rs::imod::libxml::mxml_private::mxml_opaque_cb;
use std::path::{Path, PathBuf};

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/xml")
}

/// Writes the corpus into `dir` the way `drivers/genxml.c` made native's:
/// `AdocRead` each `IMOD/autodoc/*.adoc` in byte order, `AdocSetWriteAsXML(1)`,
/// `AdocWrite("<dir>/<name>.adoc.xml")`, `AdocClear`.  Native's corpus is not
/// stored, only its digests (`fixtures/xml/golden.manifest`, keys
/// `corpus/<name>.adoc.xml`); every generated document must match its digest,
/// so the corpus the load/save parity below reads is native's, byte for byte
/// — and `AdocWrite`'s XML mode is checked on the way.
fn generate_corpus(dir: &Path) -> Vec<(PathBuf, String)> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut adocs: Vec<PathBuf> =
        std::fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/autodoc"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|e| e == "adoc"))
            .collect();
    adocs.sort();
    let mut made = Vec::new();
    let mut failures = Vec::new();
    for adoc in adocs {
        let base = format!("{}.xml", adoc.file_name().unwrap().to_string_lossy());
        let ind = adoc_read(adoc.to_string_lossy().as_bytes());
        let native = common::golden::load(&fixture_root().join("corpus").join(&base));
        if ind < 0 {
            if native.is_some() {
                failures.push(format!("{base}: AdocRead failed, native wrote it"));
            }
            continue;
        }
        adoc_set_write_as_xml(1);
        let out = dir.join(&base);
        adoc_write(out.to_string_lossy().as_bytes());
        adoc_clear(ind);
        adoc_set_write_as_xml(0);
        let written = std::fs::read(&out).unwrap();
        match native {
            None => failures.push(format!("{base}: written, native wrote nothing")),
            Some(native) => {
                if let Err(why) = native.compare(&written, common::golden::identity, false) {
                    failures.push(why);
                }
            }
        }
        made.push((out, base));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    made
}

/// Every input, as `(path, basename)`: the generated corpus first, then the
/// three hand-written documents.
fn inputs(corpus: &Path) -> Vec<(PathBuf, String)> {
    let root = fixture_root();
    let mut found = generate_corpus(corpus);
    for name in ["adoc1.xml", "adoc2.xml", "latin1.xml"] {
        found.push((root.join(name), name.to_string()));
    }
    // The 143rd input is the only `.xml` in the vendored tree outside
    // `raptor/`, and it is the one document here carrying a `<!DOCTYPE>`.
    // It is read in place — nothing under `IMOD/` is copied or modified.
    found.push((
        Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/build.xml"),
        "build.xml".to_string(),
    ));
    found
}

/// Load `path` the way IMOD loads an XML autodoc: `mxmlLoadFile` with
/// `MXML_OPAQUE_CALLBACK`, no parent.
fn load(arena: &mut MxmlArena, path: &Path) -> Option<usize> {
    let mut file = ImodFile::open(path, "r")?;
    let tree = mxml_load_file(arena, MXML_NO_PARENT, &mut file, Some(mxml_opaque_cb));
    drop(file);
    tree
}

/// Save `tree` the way `ixmlWriteFile` and `writeXmlFile` do: wrap margin 0,
/// the indentation state reset, IMOD's whitespace callback.
fn save(arena: &MxmlArena, tree: Option<usize>) -> Vec<u8> {
    let mut written = Vec::new();
    mxml_set_wrap_margin(0);
    ixml_reset_last_level();
    mxml_save_file(arena, tree, &mut written, Some(ixml_whitespace_cb));
    mxml_set_wrap_margin(72);
    written
}

/// Every document native could reparse is written back byte for byte, and
/// every document it could not is rejected here too.
#[test]
fn every_corpus_document_round_trips_to_native_bytes() {
    let root = fixture_root();
    let corpus = std::env::temp_dir().join(format!("imod-rs-xml-corpus-{}", std::process::id()));
    let all = inputs(&corpus);
    assert_eq!(all.len(), 143, "corpus size changed");

    let mut compared = 0;
    let mut rejected = Vec::new();
    for (path, base) in all {
        let expected =
            common::golden::load(&root.join("native-saves").join(format!("{base}.save")));
        let arena = &mut MxmlArena::new();
        let tree = load(arena, &path);

        let Some(expected) = expected else {
            // Native could not reparse its own output for this one.
            assert!(
                tree.is_none(),
                "{base}: native rejects this document, the translation loaded it"
            );
            rejected.push(base);
            continue;
        };

        let tree = tree.unwrap_or_else(|| panic!("{base}: failed to load"));
        let written = save(arena, Some(tree));
        if let Err(why) = expected.compare(&written, common::golden::identity, false) {
            panic!("{base}: {why}");
        }
        mxml_delete(arena, Some(tree));
        compared += 1;
    }

    assert_eq!(compared, 133, "expected 133 byte comparisons");
    let _ = std::fs::remove_dir_all(&corpus);
    rejected.sort();
    assert_eq!(
        rejected,
        vec![
            "alignframes.adoc.xml",
            "copytomocoms.adoc.xml",
            "extracttilts.adoc.xml",
            "justblend.adoc.xml",
            "makecomfile.adoc.xml",
            "mtffilter.adoc.xml",
            "peetprm.adoc.xml",
            "remapmodel.adoc.xml",
            "setupcombine.adoc.xml",
            "setupstitch.adoc.xml",
        ],
        "the set of documents native cannot reparse changed"
    );
}

// Re-saving a *reloaded* document is deliberately NOT tested for equality.
//
// A save is not a fixed point in mini-XML, and that is the library's own
// behaviour rather than a defect here: the newlines and indentation the
// writer emits come back as `MXML_OPAQUE` text nodes on the next load
// (whitespace does not end a value under `MXML_OPAQUE_CALLBACK`), and the
// writer then adds its own whitespace around those, so every round trip
// gains one `\n` per element.  Measured on `alignlog.adoc.xml`: one save
// gives `<autodoc>\n  \n  <PreData>`, a second gives
// `<autodoc>\n  \n  \n  <PreData>`.  It is also why `corpus/x.xml` and
// `native-saves/x.xml.save` differ for all 132 documents that have both.
// The contract the test above pins — one load of native's input, one save,
// compared with native's own save — is the one that matters.
