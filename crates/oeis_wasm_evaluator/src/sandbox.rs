use wasmtime::{Engine, Instance, Module, Store, StoreLimits, StoreLimitsBuilder, Val};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStatus {
    Success,
    OutOfFuel,
    ParseError,
    CompileError,
    ExecutionTrap,
    MissingEntrypoint,
    ConfigError,
}

impl ExecutionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionStatus::Success => "SUCCESS",
            ExecutionStatus::OutOfFuel => "OUT_OF_FUEL",
            ExecutionStatus::ParseError => "PARSE_ERROR",
            ExecutionStatus::CompileError => "COMPILE_ERROR",
            ExecutionStatus::ExecutionTrap => "EXECUTION_TRAP",
            ExecutionStatus::MissingEntrypoint => "MISSING_ENTRYPOINT",
            ExecutionStatus::ConfigError => "CONFIG_ERROR",
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluatorResult {
    pub status: ExecutionStatus,
    pub consumed_fuel: u64,
    pub max_fuel: u64,
    pub total_fuel: u64,
    pub output: Vec<i64>,
    pub wide_output: Vec<[i64; 4]>,
    pub error: Option<String>,
}

pub struct SandboxHostData {
    pub limits: StoreLimits,
}

pub fn evaluate_single_program(
    engine: &Engine,
    wat_code: &str,
    fuel_budget: u64,
    terms_to_generate: u32,
) -> EvaluatorResult {
    // 1. In-memory parse WAT text to WASM binary bytecode
    let wasm_bytes = match wat::parse_str(wat_code) {
        Ok(bytes) => bytes,
        Err(e) => {
            return EvaluatorResult {
                status: ExecutionStatus::ParseError,
                consumed_fuel: 0,
                max_fuel: 0,
                total_fuel: 0,
                output: Vec::new(),
                wide_output: Vec::new(),
                error: Some(format!("WAT parse error: {}", e)),
            };
        }
    };

    // 2. Compile WASM binary into Module
    let module = match Module::new(engine, &wasm_bytes) {
        Ok(m) => m,
        Err(e) => {
            return EvaluatorResult {
                status: ExecutionStatus::CompileError,
                consumed_fuel: 0,
                max_fuel: 0,
                total_fuel: 0,
                output: Vec::new(),
                wide_output: Vec::new(),
                error: Some(format!("WASM compilation error: {}", e)),
            };
        }
    };

    // 3. Configure store with fuel budget and 16 MiB linear memory limit
    let host_data = SandboxHostData {
        limits: StoreLimitsBuilder::new()
            .memory_size(16 * 1024 * 1024) // 16 MiB
            .instances(1)
            .tables(10)
            .memories(1)
            .build(),
    };

    let mut store = Store::new(engine, host_data);
    store.limiter(|data| &mut data.limits);

    if let Err(e) = store.set_fuel(fuel_budget) {
        return EvaluatorResult {
            status: ExecutionStatus::ConfigError,
            consumed_fuel: 0,
            max_fuel: 0,
            total_fuel: 0,
            output: Vec::new(),
            wide_output: Vec::new(),
            error: Some(format!("Failed to set fuel: {}", e)),
        };
    }

    // 4. Instantiate module
    let instance = match Instance::new(&mut store, &module, &[]) {
        Ok(inst) => inst,
        Err(e) => {
            let error_str = e.to_string();
            let fuel_rem = store.get_fuel().unwrap_or(0);
            let is_out_of_fuel = fuel_rem == 0
                || error_str.to_lowercase().contains("fuel")
                || error_str.to_lowercase().contains("all fuel consumed");
            let status = if is_out_of_fuel {
                ExecutionStatus::OutOfFuel
            } else {
                ExecutionStatus::ExecutionTrap
            };
            let consumed = fuel_budget.saturating_sub(fuel_rem);
            return EvaluatorResult {
                status,
                consumed_fuel: consumed,
                max_fuel: consumed,
                total_fuel: consumed,
                output: Vec::new(),
                wide_output: Vec::new(),
                error: Some(format!("Instantiation trap: {}", e)),
            };
        }
    };

    // 5. Locate entrypoint function: try "compute", "generate_term", "a", or the first exported function
    let func = instance
        .get_func(&mut store, "compute")
        .or_else(|| instance.get_func(&mut store, "generate_term"))
        .or_else(|| instance.get_func(&mut store, "a"))
        .or_else(|| {
            // Find any exported func
            module.exports().find_map(|exp| {
                if exp.ty().func().is_some() {
                    instance.get_func(&mut store, exp.name())
                } else {
                    None
                }
            })
        });

    let func = match func {
        Some(f) => f,
        None => {
            let fuel_rem = store.get_fuel().unwrap_or(fuel_budget);
            let consumed = fuel_budget.saturating_sub(fuel_rem);
            return EvaluatorResult {
                status: ExecutionStatus::MissingEntrypoint,
                consumed_fuel: consumed,
                max_fuel: consumed,
                total_fuel: consumed,
                output: Vec::new(),
                wide_output: Vec::new(),
                error: Some("No entrypoint function found (looked for 'compute', 'generate_term', 'a', or first export)".to_string()),
            };
        }
    };

    let param_types = func.ty(&store).params().collect::<Vec<_>>();
    let use_i64_param = match param_types.first() {
        Some(wasmtime::ValType::I64) => true,
        _ => false,
    };

    let result_types = func.ty(&store).results().collect::<Vec<_>>();
    let is_wide = result_types.len() == 4;

    // 6. Evaluate sequence terms for n = 0..terms_to_generate
    let mut outputs = Vec::with_capacity(terms_to_generate as usize);
    let mut wide_outputs = Vec::with_capacity(terms_to_generate as usize);
    let mut max_fuel_used = 0u64;
    let mut total_fuel_used = 0u64;

    for n in 0..terms_to_generate {
        // Reset fuel per invocation
        if let Err(e) = store.set_fuel(fuel_budget) {
            return EvaluatorResult {
                status: ExecutionStatus::ConfigError,
                consumed_fuel: max_fuel_used,
                max_fuel: max_fuel_used,
                total_fuel: total_fuel_used,
                output: outputs,
                wide_output: wide_outputs,
                error: Some(format!("Failed to set fuel: {}", e)),
            };
        }

        let arg = if use_i64_param {
            Val::I64(n as i64)
        } else {
            Val::I32(n as i32)
        };

        if is_wide {
            let mut results = [Val::I64(0), Val::I64(0), Val::I64(0), Val::I64(0)];
            match func.call(&mut store, &[arg], &mut results) {
                Ok(_) => {
                    let fuel_rem = store.get_fuel().unwrap_or(0);
                    let call_consumed = fuel_budget.saturating_sub(fuel_rem);
                    total_fuel_used += call_consumed;
                    if call_consumed > max_fuel_used {
                        max_fuel_used = call_consumed;
                    }
                    let l0 = match results[0] { Val::I64(v) => v, _ => 0 };
                    let l1 = match results[1] { Val::I64(v) => v, _ => 0 };
                    let l2 = match results[2] { Val::I64(v) => v, _ => 0 };
                    let l3 = match results[3] { Val::I64(v) => v, _ => 0 };
                    wide_outputs.push([l0, l1, l2, l3]);
                    outputs.push(l0);
                }
                Err(e) => {
                    let error_str = e.to_string();
                    let fuel_rem = store.get_fuel().unwrap_or(0);
                    let call_consumed = fuel_budget.saturating_sub(fuel_rem);
                    total_fuel_used += call_consumed;
                    if call_consumed > max_fuel_used {
                        max_fuel_used = call_consumed;
                    }
                    let is_out_of_fuel = fuel_rem == 0
                        || error_str.to_lowercase().contains("fuel")
                        || error_str.to_lowercase().contains("all fuel consumed");
                    let status = if is_out_of_fuel {
                        ExecutionStatus::OutOfFuel
                    } else {
                        ExecutionStatus::ExecutionTrap
                    };
                    return EvaluatorResult {
                        status,
                        consumed_fuel: max_fuel_used,
                        max_fuel: max_fuel_used,
                        total_fuel: total_fuel_used,
                        output: outputs,
                        wide_output: wide_outputs,
                        error: Some(error_str),
                    };
                }
            }
        } else {
            let mut results = [Val::I64(0)];
            match func.call(&mut store, &[arg], &mut results) {
                Ok(_) => {
                    let fuel_rem = store.get_fuel().unwrap_or(0);
                    let call_consumed = fuel_budget.saturating_sub(fuel_rem);
                    total_fuel_used += call_consumed;
                    if call_consumed > max_fuel_used {
                        max_fuel_used = call_consumed;
                    }
                    let term = match results[0] {
                        Val::I64(v) => v,
                        Val::I32(v) => v as i64,
                        _ => 0,
                    };
                    outputs.push(term);
                }
                Err(e) => {
                    let error_str = e.to_string();
                    let fuel_rem = store.get_fuel().unwrap_or(0);
                    let call_consumed = fuel_budget.saturating_sub(fuel_rem);
                    total_fuel_used += call_consumed;
                    if call_consumed > max_fuel_used {
                        max_fuel_used = call_consumed;
                    }
                    let is_out_of_fuel = fuel_rem == 0
                        || error_str.to_lowercase().contains("fuel")
                        || error_str.to_lowercase().contains("all fuel consumed");
                    let status = if is_out_of_fuel {
                        ExecutionStatus::OutOfFuel
                    } else {
                        ExecutionStatus::ExecutionTrap
                    };
                    return EvaluatorResult {
                        status,
                        consumed_fuel: max_fuel_used,
                        max_fuel: max_fuel_used,
                        total_fuel: total_fuel_used,
                        output: outputs,
                        wide_output: wide_outputs,
                        error: Some(error_str),
                    };
                }
            }
        }
    }

    EvaluatorResult {
        status: ExecutionStatus::Success,
        consumed_fuel: max_fuel_used,
        max_fuel: max_fuel_used,
        total_fuel: total_fuel_used,
        output: outputs,
        wide_output: wide_outputs,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::create_fuel_engine;

    #[test]
    fn test_valid_triangular_numbers() {
        let engine = create_fuel_engine().unwrap();
        // Triangular numbers: a(n) = n*(n+1)/2
        let wat = r#"
            (module
                (func (export "compute") (param $n i32) (result i64)
                    (local $n64 i64)
                    (local.set $n64 (i64.extend_i32_u (local.get $n)))
                    (i64.div_u
                        (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
                        (i64.const 2)
                    )
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 6);
        assert_eq!(res.status, ExecutionStatus::Success);
        assert_eq!(res.output, vec![0, 1, 3, 6, 10, 15]);
        assert!(res.consumed_fuel > 0 && res.consumed_fuel < 10000);
        assert!(res.total_fuel >= res.max_fuel);
    }

    #[test]
    fn test_per_invocation_fuel_reset() {
        let engine = create_fuel_engine().unwrap();
        // Constant function: consumes small fuel each term
        let wat = r#"
            (module
                (func (export "compute") (param $n i32) (result i64)
                    (i64.const 42)
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 10);
        assert_eq!(res.status, ExecutionStatus::Success);
        assert_eq!(res.output.len(), 10);
        // Total fuel should be ~10x max fuel per invocation
        assert!(res.total_fuel > res.max_fuel);
        assert!(res.max_fuel < 500);
    }

    #[test]
    fn test_memory_limit_enforcement() {
        let engine = create_fuel_engine().unwrap();
        // Module attempting to allocate 300 pages (19.2 MiB > 16 MiB ceiling)
        let wat = r#"
            (module
                (memory 300)
                (func (export "compute") (param $n i32) (result i64)
                    (i64.const 1)
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 1);
        assert!(res.status == ExecutionStatus::ExecutionTrap || res.status == ExecutionStatus::CompileError);
    }

    #[test]
    fn test_wide_result_4_limbs() {
        let engine = create_fuel_engine().unwrap();
        let wat = r#"
            (module
                (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
                    (i64.const 10)
                    (i64.const 20)
                    (i64.const 30)
                    (i64.const 40)
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 3);
        assert_eq!(res.status, ExecutionStatus::Success);
        assert_eq!(res.wide_output.len(), 3);
        assert_eq!(res.wide_output[0], [10, 20, 30, 40]);
    }

    #[test]
    fn test_infinite_loop_out_of_fuel() {
        let engine = create_fuel_engine().unwrap();
        // Infinite loop
        let wat = r#"
            (module
                (func (export "compute") (param $n i32) (result i64)
                    (loop $l
                        (br $l)
                    )
                    (i64.const 0)
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 5);
        assert_eq!(res.status, ExecutionStatus::OutOfFuel);
        assert_eq!(res.consumed_fuel, 10000);
    }

    #[test]
    fn test_division_by_zero_trap() {
        let engine = create_fuel_engine().unwrap();
        let wat = r#"
            (module
                (func (export "compute") (param $n i32) (result i64)
                    (i64.div_s (i64.const 100) (i64.const 0))
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 5);
        assert_eq!(res.status, ExecutionStatus::ExecutionTrap);
    }

    #[test]
    fn test_unreachable_trap() {
        let engine = create_fuel_engine().unwrap();
        let wat = r#"
            (module
                (func (export "compute") (param $n i32) (result i64)
                    unreachable
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 1);
        assert_eq!(res.status, ExecutionStatus::ExecutionTrap);
    }

    #[test]
    fn test_preamble_i256_add_and_sub() {
        let engine = create_fuel_engine().unwrap();
        let wat = r#"
            (module
                (func $i256_add
                    (param $a0 i64) (param $a1 i64) (param $a2 i64) (param $a3 i64)
                    (param $b0 i64) (param $b1 i64) (param $b2 i64) (param $b3 i64)
                    (result i64 i64 i64 i64)
                    (local $s0 i64) (local $c0 i64)
                    (local $s1 i64) (local $c1 i64)
                    (local $s2 i64) (local $c2 i64)
                    (local $s3 i64)
                    local.get $a0 local.get $b0 i64.add local.set $s0
                    local.get $s0 local.get $b0 i64.lt_u i64.extend_i32_u local.set $c0
                    local.get $a1 local.get $b1 i64.add local.get $c0 i64.add local.set $s1
                    local.get $a1 local.get $b1 i64.add local.get $b1 i64.lt_u i64.extend_i32_u
                    local.get $s1 local.get $c0 i64.lt_u i64.extend_i32_u i64.add local.set $c1
                    local.get $a2 local.get $b2 i64.add local.get $c1 i64.add local.set $s2
                    local.get $a2 local.get $b2 i64.add local.get $b2 i64.lt_u i64.extend_i32_u
                    local.get $s2 local.get $c1 i64.lt_u i64.extend_i32_u i64.add local.set $c2
                    local.get $a3 local.get $b3 i64.add local.get $c2 i64.add local.set $s3
                    local.get $s0 local.get $s1 local.get $s2 local.get $s3
                )
                (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
                    ;; (2^64 - 1) + 1 => should carry to limb 1 = 1, limb 0 = 0
                    i64.const -1 i64.const 0 i64.const 0 i64.const 0
                    i64.const 1 i64.const 0 i64.const 0 i64.const 0
                    call $i256_add
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 1);
        assert_eq!(res.status, ExecutionStatus::Success);
        assert_eq!(res.wide_output.len(), 1);
        assert_eq!(res.wide_output[0], [0, 1, 0, 0]);
        assert!(res.consumed_fuel < 500);
    }

    #[test]
    fn test_preamble_mul64_wide() {
        let engine = create_fuel_engine().unwrap();
        let wat = r#"
            (module
                (func $mul64_wide (param $u i64) (param $v i64) (result i64 i64)
                    (local $u0 i64) (local $u1 i64)
                    (local $v0 i64) (local $v1 i64)
                    (local $w0 i64) (local $k i64)
                    (local $w1 i64) (local $w2 i64)
                    (local $lo i64) (local $hi i64)
                    local.get $u i64.const 4294967295 i64.and local.set $u0
                    local.get $u i64.const 32 i64.shr_u local.set $u1
                    local.get $v i64.const 4294967295 i64.and local.set $v0
                    local.get $v i64.const 32 i64.shr_u local.set $v1
                    local.get $u0 local.get $v0 i64.mul local.set $w0
                    local.get $w0 i64.const 32 i64.shr_u local.set $k
                    local.get $u1 local.get $v0 i64.mul local.get $k i64.add local.set $w1
                    local.get $w1 i64.const 4294967295 i64.and local.set $k
                    local.get $w1 i64.const 32 i64.shr_u local.set $w1
                    local.get $u0 local.get $v1 i64.mul local.get $k i64.add local.set $w2
                    local.get $w2 i64.const 32 i64.shr_u local.set $k
                    local.get $u1 local.get $v1 i64.mul local.get $w1 i64.add local.get $k i64.add local.set $hi
                    local.get $w2 i64.const 32 i64.shl local.get $w0 i64.const 4294967295 i64.and i64.or local.set $lo
                    local.get $lo local.get $hi
                )
                (func (export "compute") (param $n i32) (result i64 i64 i64 i64)
                    ;; 2^32 * 2^32 = 2^64 => lo = 0, hi = 1
                    i64.const 4294967296 i64.const 4294967296
                    call $mul64_wide
                    i64.const 0 i64.const 0
                )
            )
        "#;
        let res = evaluate_single_program(&engine, wat, 10000, 1);
        assert_eq!(res.status, ExecutionStatus::Success);
        assert_eq!(res.wide_output.len(), 1);
        assert_eq!(res.wide_output[0], [0, 1, 0, 0]);
    }
}
