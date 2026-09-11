// `ui/lib.slint` re-exports both dialogs, so one compile covers them all.
fn main() {
    slint_build::compile("ui/lib.slint").expect("compiling ui/lib.slint");
}
