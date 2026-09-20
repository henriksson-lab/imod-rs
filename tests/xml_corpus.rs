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

/// Every input, as `(path, basename)`: the generated corpus first, then the
/// three hand-written documents.
fn inputs() -> Vec<(PathBuf, String)> {
    let root = fixture_root();
    let mut found = Vec::new();
    let mut corpus: Vec<PathBuf> = std::fs::read_dir(root.join("corpus"))
        .expect("fixtures/xml/corpus is missing")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "xml"))
        .collect();
    corpus.sort();
    for path in corpus {
        let base = path.file_name().unwrap().to_string_lossy().into_owned();
        found.push((path, base));
    }
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
    let all = inputs();
    assert_eq!(all.len(), 143, "corpus size changed");

    let mut compared = 0;
    let mut rejected = Vec::new();
    for (path, base) in all {
        let expected_path = root.join("native-saves").join(format!("{base}.save"));
        let arena = &mut MxmlArena::new();
        let tree = load(arena, &path);

        if !expected_path.exists() {
            // Native could not reparse its own output for this one.
            assert!(
                tree.is_none(),
                "{base}: native rejects this document, the translation loaded it"
            );
            rejected.push(base);
            continue;
        }

        let tree = tree.unwrap_or_else(|| panic!("{base}: failed to load"));
        let written = save(arena, Some(tree));
        let expected = std::fs::read(&expected_path).unwrap();
        assert_eq!(
            written.len(),
            expected.len(),
            "{base}: wrote {} bytes, native wrote {}",
            written.len(),
            expected.len()
        );
        if written != expected {
            let at = (0..written.len())
                .find(|&index| written[index] != expected[index])
                .unwrap();
            let from = at.saturating_sub(40);
            panic!(
                "{base}: first difference at byte {at}\n  native: {:?}\n  rust  : {:?}",
                String::from_utf8_lossy(&expected[from..(at + 40).min(expected.len())]),
                String::from_utf8_lossy(&written[from..(at + 40).min(written.len())]),
            );
        }
        mxml_delete(arena, Some(tree));
        compared += 1;
    }

    assert_eq!(compared, 133, "expected 133 byte comparisons");
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

/// A second load of what the translation just wrote produces the same bytes a
/// third time: the writer's output is a fixed point, as native's is.
#[test]
fn saving_a_reloaded_document_is_idempotent() {
    let root = fixture_root();
    for (path, base) in inputs() {
        if !root
            .join("native-saves")
            .join(format!("{base}.save"))
            .exists()
        {
            continue;
        }
        let arena = &mut MxmlArena::new();
        let tree = load(arena, &path).unwrap_or_else(|| panic!("{base}: failed to load"));
        let once = save(arena, Some(tree));
        mxml_delete(arena, Some(tree));

        let scratch = std::env::temp_dir().join(format!(
            "imod-rs-xml-{}-{}",
            std::process::id(),
            base.replace('/', "_")
        ));
        std::fs::write(&scratch, &once).unwrap();
        let arena = &mut MxmlArena::new();
        let again = load(arena, &scratch).unwrap_or_else(|| panic!("{base}: reload failed"));
        let twice = save(arena, Some(again));
        mxml_delete(arena, Some(again));
        let _ = std::fs::remove_file(&scratch);

        assert_eq!(
            String::from_utf8_lossy(&once),
            String::from_utf8_lossy(&twice),
            "{base}: re-saving a reloaded document changed it"
        );
    }
}
