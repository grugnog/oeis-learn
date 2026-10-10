use std::{collections::BTreeMap, env, fs, path::Path};
fn main() {
    let mut sources = BTreeMap::new();
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "src/sandbox.rs",
        "src/engine.rs",
    ] {
        if Path::new(name).exists() {
            sources.insert(name, fs::read(name).unwrap());
            println!("cargo:rerun-if-changed={name}");
        }
    }
    let bytes = sources
        .iter()
        .flat_map(|(n, b)| {
            n.as_bytes()
                .iter()
                .copied()
                .chain([0])
                .chain(b.iter().copied())
                .chain([0])
        })
        .collect::<Vec<_>>();
    // Embed source bytes; Python hashes and checks this against the checkout.
    fs::write(
        Path::new(&env::var("OUT_DIR").unwrap()).join("source_identity.bin"),
        bytes,
    )
    .unwrap();
}
