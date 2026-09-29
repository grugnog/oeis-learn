use wasmtime::{Config, Engine, Result};

/// Creates a thread-safe shared Wasmtime engine configured with Cranelift JIT and fuel consumption.
pub fn create_fuel_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.consume_fuel(true);
    config.cranelift_opt_level(wasmtime::OptLevel::Speed);
    // Disable multi-threading or signals that interfere with embeddings if needed
    Engine::new(&config)
}
