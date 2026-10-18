//! Dependency smoke module only. No candidate ABI or arithmetic service yet.
//! Phase 2 replaces this diagnostic; production bigint execution stays disabled.
use dashu_int::IBig;

/// Forces allocation, parsing and full multiplication into standalone WASM.
/// 2^64 squared equals 2^128, independently fixed below.
#[no_mangle]
pub extern "C" fn dependency_smoke() -> i32 {
    let value: IBig = "18446744073709551616".parse().unwrap();
    i32::from((&value * &value).to_string() == "340282366920938463463374607431768211456")
}

#[no_mangle]
pub extern "C" fn production_enabled() -> i32 {
    0
}
