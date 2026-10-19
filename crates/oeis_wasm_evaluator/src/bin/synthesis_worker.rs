fn main() {
    if let Err(reason) = oeis_wasm_evaluator::serve_synthesis() {
        eprintln!("{reason}");
        std::process::exit(1);
    }
}
