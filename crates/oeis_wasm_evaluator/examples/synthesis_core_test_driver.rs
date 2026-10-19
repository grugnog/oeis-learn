//! Qualification-only bridge; not an installed production command.
use oeis_wasm_evaluator::synthesis_adapter::CheckedSigned256;

fn main() {
    if let Err(reason) = oeis_synthesis_core::diagnostic::run(&mut CheckedSigned256::default()) {
        eprintln!("{reason}");
        std::process::exit(1);
    }
}
