// `ui/lib.slint` re-exports every component in this crate, so one compile
// covers them all.
fn main() {
    slint_build::compile("ui/lib.slint").expect("compiling ui/lib.slint");
}
