fn main() {
    println!("{}", oeis_synthesis_core::dependency_smoke());
    assert!(!oeis_synthesis_core::PRODUCTION_ENABLED);
}
