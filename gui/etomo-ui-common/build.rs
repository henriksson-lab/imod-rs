// `ui/lib.slint` re-exports widgets.slint, cells.slint and fields.slint, so
// one compile covers the whole shared vocabulary.
fn main() {
    slint_build::compile("ui/lib.slint").expect("compiling ui/lib.slint");
}
