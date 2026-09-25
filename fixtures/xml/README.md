# `fixtures/xml` — the mini-XML acceptance corpus

This is the verification the `src/imod/libxml` translation rests on, moved out
of a session scratchpad and into the tree on 2026-09-20. It had been sitting
in `/big/henriksson/temp/claude-tmp/.../scratchpad/xmldiff/`, which is exactly
how the native reference build was lost once before: **a perishable reference
is a silent one** — without it a failing parity test cannot be told apart from
a documented non-achievable.

`tests/xml_corpus.rs` runs against this directory in the ordinary gate. No
environment variable is needed and no reference build has to be present: the
native side is already here, as bytes.

## What is here

| path | what |
|---|---|
| `corpus/` | 139 XML autodocs, produced by **native** `AdocSetWriteAsXML(1); AdocWrite` from all 140 `IMOD/autodoc/*.adoc` (one input produced no output) |
| `adoc1.xml` | hand-written: the four escaped characters, `&#65;`, `&nbsp;`, an empty element, a comment, a `PreData` section |
| `adoc2.xml` | no whitespace between tags, single-quoted attributes, CDATA, a comment |
| `latin1.xml` | UTF-8 with a non-ASCII character (é) |
| `native-saves/` | 133 `<input>.save` files: what **native mini-XML** wrote when it loaded the corresponding input with `MXML_OPAQUE_CALLBACK` and saved it through `mxmlSaveFile` with IMOD's `ixmlWhitespace_cb` and `mxmlSetWrapMargin(0)`. These are the byte-parity objects. |
| `native-stderr.txt` | the 30 diagnostic lines native emitted across the whole corpus |
| `drivers/probe.c` | the C driver, linked against the reference `libimxml.so`: every node's type, links, refcount, value and getters, walk/find/findPath, the index API, retain/release, the 257-entity table, 15 pathological strings × 5 load callbacks, three save configurations |
| `drivers/adocrt.c` | the autodoc round-trip driver (adoc→XML→adoc and XML→adoc→XML) |
| `drivers/genxml.c` | what generated `corpus/` from `IMOD/autodoc/*.adoc` |

**10 of the 143 inputs have no `.save`.** That is not an omission: native
cannot reparse its own output for those files. `AdocWrite` emits element names
that are not XML names — `<^   critical_dose>` in `mtffilter.adoc.xml`,
`<3.6, 3.2, the program will map 0 to 1.4 (>` in another — and the loader then
rejects them with `mxml: Missing value for attribute 'critical_dose' in
element ^!`. The affected files are alignframes, copytomocoms, extracttilts,
justblend, makecomfile, mtffilter, peetprm, remapmodel, setupcombine and
setupstitch. The test asserts the translation rejects exactly those ten, which
is as much a part of the contract as the bytes of the other 133.

## A save is not a fixed point

Loading one of these documents and saving it does **not** reproduce the input,
and saving the result again does not reproduce that either. Under
`MXML_OPAQUE_CALLBACK` whitespace does not end a value, so the newlines and
indentation the writer emitted come back as `MXML_OPAQUE` text nodes, and the
writer then adds its own whitespace around them: every round trip gains one
`\n` per element. That is mini-XML's own behaviour, and it is why every
`corpus/x.xml` differs from its `native-saves/x.xml.save`. The contract is one
load and one save, compared against native's — not idempotence.

## Rebuilding the parts that are not stored

`native.txt`, the probe's full report, is **17 623 962 bytes** and is not in
the tree. Its SHA-256 is:

```
fb01da431b03512eba1c0ab6dcc60c6b8743386544ace2495a50f7fa42871493
```

To regenerate it, build `drivers/probe.c` against the reference build (see
CLAUDE.md § "Native reference differentials" for how to make that build) and
run it over `allfiles`:

```bash
cc -o probe drivers/probe.c -I/tmp/imod-reference-build/include \
   -L/tmp/imod-reference-build/buildlib -limxml -lm
LD_LIBRARY_PATH=/tmp/imod-reference-build/buildlib ./probe <files...> \
   > native.txt 2> native-stderr.txt
```

The Rust side of that report comes from the env-gated `xml_probe` test in
`src/imod/libxml/mxml_file.rs` (`IMOD_XML_PROBE_OUT`,
`IMOD_XML_PROBE_SAVEDIR`, `IMOD_XML_PROBE_FILES`), and the round trip from
`xml_autodoc_roundtrip` (`IMOD_XML_RT_REPORT`, `IMOD_XML_RT_OUTDIR`,
`IMOD_XML_RT_ASXML`, `IMOD_XML_RT_FILES`). Those two remain opt-in: they
compare 17.6 MB of report text and need the reference library, so they are a
deliberate extra pass, not gate material. What the gate now covers without
them is the part that actually matters for output: **the saved bytes**.

## If the loader is ever replaced

`XML.md` plans a possible swap of the character-level scanner for quick-xml.
This directory is the evidence that plan depends on: parity means every one of
the 133 saves still byte-identical and the same 10 rejections, with the same
stderr text. Do not regenerate `native-saves/` from a Rust build — they are
native's output and that is the whole point.
