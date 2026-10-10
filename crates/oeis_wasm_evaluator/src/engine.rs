use std::sync::OnceLock;
use wasmtime::{Config, Engine, Result};

/// Creates a thread-safe shared Wasmtime engine configured with Cranelift JIT and fuel consumption.
pub fn create_fuel_engine() -> Result<Engine> {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    if let Some(engine) = ENGINE.get() {
        return Ok(engine.clone());
    }
    let mut config = Config::new();
    config.consume_fuel(true);
    config.parallel_compilation(false);
    config.wasm_threads(false);
    config.static_memory_maximum_size(0);
    config.static_memory_guard_size(0);
    config.dynamic_memory_guard_size(0);
    config.dynamic_memory_reserved_for_growth(0);
    config.cranelift_opt_level(wasmtime::OptLevel::Speed);
    // Disable multi-threading or signals that interfere with embeddings if needed
    let engine = Engine::new(&config)?;
    let _ = ENGINE.set(engine.clone());
    Ok(engine)
}
