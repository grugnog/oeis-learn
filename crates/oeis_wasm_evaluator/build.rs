use std::{collections::BTreeMap, env, fs, path::Path};

fn collect(root: &Path, path: &Path, sources: &mut BTreeMap<String, Vec<u8>>) {
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if !["target", ".git"].contains(&path.file_name().unwrap().to_str().unwrap()) {
                collect(root, &path, sources);
            }
        } else if matches!(path.extension().and_then(|s| s.to_str()), Some("rs" | "toml" | "lock")) {
            let name = path.strip_prefix(root).unwrap().to_str().unwrap().to_owned();
            sources.insert(name, fs::read(&path).unwrap());
        }
    }
}

fn main() {
    let root = Path::new("../..");
    let mut sources = BTreeMap::new();
    for name in ["oeis_wasm_evaluator", "oeis_synthesis_core", "oeis_bigint_runtime"] {
        let path = root.join("crates").join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        collect(root, &path, &mut sources);
    }
    for name in [
        "Dockerfile",
        "docker/dev/build-native.sh",
        "docker/dev/build-synthesis.sh",
        "docker/dev/install.sh",
        "src/oeis_learn/synthesis/records.schema.json",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(name).display());
        sources.insert(name.to_owned(), fs::read(root.join(name)).unwrap());
    }
    let bytes = sources.iter().flat_map(|(name, bytes)| {
        name.as_bytes().iter().copied().chain([0]).chain(bytes.iter().copied()).chain([0])
    }).collect::<Vec<_>>();
    // Python verifies exactly the same deterministic supported source closure.
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("source_identity.bin"), bytes).unwrap();
}
