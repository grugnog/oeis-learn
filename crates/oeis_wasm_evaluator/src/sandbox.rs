use wasmtime::{
    Engine, Instance, Module, Store, StoreLimits, StoreLimitsBuilder, Trap, Val, ValType,
};

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
            Self::Success => "SUCCESS",
            Self::OutOfFuel => "OUT_OF_FUEL",
            Self::ParseError => "PARSE_ERROR",
            Self::CompileError => "COMPILE_ERROR",
            Self::ExecutionTrap => "EXECUTION_TRAP",
            Self::MissingEntrypoint => "MISSING_ENTRYPOINT",
            Self::ConfigError => "CONFIG_ERROR",
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

/// A compiled module is reusable; Store and Instance are never reused.
pub struct Compiled {
    pub engine: Engine,
    pub module: Module,
    pub strict: bool,
    pub wide: bool,
    pub input_i64: bool,
    pub entry: String,
}
pub struct Term {
    pub values: Option<Vec<i64>>,
    pub fuel: u64,
    pub outcome: Option<String>,
    pub reason: Option<String>,
}
impl Compiled {
    pub fn new(engine: &Engine, source: &str, profile: &str) -> Result<Self, String> {
        let bytes = wat::parse_str(source).map_err(|e| format!("PARSE_ERROR:{e}"))?;
        let module = Module::new(engine, bytes).map_err(|e| format!("COMPILE_ERROR:{e}"))?;
        Self::from_module(engine, module, profile)
    }
    pub fn from_module(engine: &Engine, module: Module, profile: &str) -> Result<Self, String> {
        let strict = profile == "wat_i256_checked_v1";
        let wide = strict || profile == "i256x4_v1";
        if !strict && !wide && profile != "i64_scalar_v1" {
            return Err("CONFIG_ERROR:explicit profile required".into());
        }
        if module.imports().next().is_some() {
            return Err("COMPILE_ERROR:imports forbidden".into());
        }
        let entry = if strict {
            "compute".to_string()
        } else {
            ["compute", "generate_term", "a"]
                .iter()
                .find(|name| {
                    module
                        .get_export(name)
                        .and_then(|t| t.func().cloned())
                        .is_some()
                })
                .ok_or("MISSING_ENTRYPOINT:no supported export")?
                .to_string()
        };
        let ty = module
            .get_export(&entry)
            .and_then(|t| t.func().cloned())
            .ok_or("MISSING_ENTRYPOINT:compute required")?;
        let params: Vec<_> = ty.params().collect();
        let results: Vec<_> = ty.results().collect();
        if params.len() != 1
            || !(matches!(params[0], ValType::I32)
                || (!strict && matches!(params[0], ValType::I64)))
            || results.len() != if wide { 4 } else { 1 }
            || !results.iter().all(|v| matches!(v, ValType::I64))
        {
            return Err("COMPILE_ERROR:exact signature mismatch".into());
        }
        if strict {
            let marker = module
                .get_export("__oeis_numeric_limit")
                .and_then(|t| t.global().cloned())
                .ok_or("COMPILE_ERROR:overflow marker required")?;
            if !matches!(marker.content(), ValType::I32) {
                return Err("COMPILE_ERROR:invalid overflow marker".into());
            }
            if module.exports().any(|e| {
                matches!(
                    e.ty(),
                    wasmtime::ExternType::Memory(_) | wasmtime::ExternType::Table(_)
                )
            }) {
                return Err("COMPILE_ERROR:guest memory/table forbidden".into());
            }
        }
        Ok(Self {
            engine: engine.clone(),
            module,
            strict,
            wide,
            input_i64: matches!(params[0], ValType::I64),
            entry,
        })
    }
    pub fn term(&self, index: i32, fuel: u64) -> Term {
        let limits = StoreLimitsBuilder::new()
            .memory_size(if self.strict { 0 } else { 16 * 1024 * 1024 })
            .instances(1)
            .tables(if self.strict { 0 } else { 10 })
            .memories(if self.strict { 0 } else { 1 })
            .build();
        let mut store = Store::new(&self.engine, SandboxHostData { limits });
        store.limiter(|d| &mut d.limits);
        if let Err(e) = store.set_fuel(fuel) {
            return Term {
                values: None,
                fuel: 0,
                outcome: Some("runtime_failure".into()),
                reason: Some(e.to_string()),
            };
        }
        let mut marker = None;
        let result = (|| -> wasmtime::Result<Vec<i64>> {
            let instance = Instance::new(&mut store, &self.module, &[])?;
            marker = instance.get_global(&mut store, "__oeis_numeric_limit");
            let func = instance
                .get_func(&mut store, &self.entry)
                .ok_or_else(|| wasmtime::Error::msg("missing entrypoint"))?;
            let input = if self.input_i64 {
                Val::I64(index as i64)
            } else {
                Val::I32(index)
            };
            let mut values = vec![Val::I64(0); if self.wide { 4 } else { 1 }];
            func.call(&mut store, &[input], &mut values)?;
            values
                .into_iter()
                .map(|v| match v {
                    Val::I64(x) => Ok(x),
                    _ => Err(wasmtime::Error::msg("non-i64 output")),
                })
                .collect()
        })();
        let used = fuel.saturating_sub(store.get_fuel().unwrap_or(0));
        match result {
            Ok(values) => Term {
                values: Some(values),
                fuel: used,
                outcome: None,
                reason: None,
            },
            Err(error) => {
                let overflow = self.strict
                    && marker
                        .map(|g| matches!(g.get(&mut store), Val::I32(1)))
                        .unwrap_or(false);
                let out_of_fuel = matches!(error.downcast_ref::<Trap>(), Some(Trap::OutOfFuel));
                let (outcome, reason) = if overflow {
                    ("numeric_limit".into(), "checked_signed_256_overflow".into())
                } else if out_of_fuel {
                    ("execution_limit".into(), "wasmtime_fuel".into())
                } else {
                    ("runtime_failure".into(), format!("wasmtime_trap:{error}"))
                };
                Term {
                    values: None,
                    fuel: used,
                    outcome: Some(outcome),
                    reason: Some(reason),
                }
            }
        }
    }
}

pub fn evaluate_profile(
    engine: &Engine,
    source: &str,
    fuel: u64,
    count: u32,
    profile: &str,
) -> EvaluatorResult {
    let mut res = EvaluatorResult {
        status: ExecutionStatus::Success,
        consumed_fuel: 0,
        max_fuel: 0,
        total_fuel: 0,
        output: vec![],
        wide_output: vec![],
        error: None,
    };
    if fuel == 0 || fuel > 1_000_000 || count == 0 || count > 120 || source.len() > 65536 {
        res.status = ExecutionStatus::ConfigError;
        res.error = Some("execution cap exceeded".into());
        return res;
    }
    let compiled = match Compiled::new(engine, source, profile) {
        Ok(c) => c,
        Err(e) => {
            res.status = if e.starts_with("PARSE_ERROR") {
                ExecutionStatus::ParseError
            } else if e.starts_with("MISSING_ENTRYPOINT") {
                ExecutionStatus::MissingEntrypoint
            } else {
                ExecutionStatus::CompileError
            };
            res.error = Some(e);
            return res;
        }
    };
    for n in 0..count {
        let allowance = fuel.min(50_000_000 - res.total_fuel);
        if allowance == 0 {
            res.status = ExecutionStatus::OutOfFuel;
            res.error = Some("aggregate_fuel".into());
            break;
        }
        let term = compiled.term(n as i32, allowance);
        res.total_fuel += term.fuel;
        res.max_fuel = res.max_fuel.max(term.fuel);
        res.consumed_fuel = res.max_fuel;
        if let Some(values) = term.values {
            if compiled.wide {
                res.wide_output
                    .push(values.try_into().expect("validated four results"));
            } else {
                res.output.push(values[0]);
            }
        } else {
            res.status = if term.outcome.as_deref() == Some("execution_limit") {
                ExecutionStatus::OutOfFuel
            } else {
                ExecutionStatus::ExecutionTrap
            };
            res.error = term.reason;
            break;
        }
    }
    res
}
/// Historical Rust diagnostic API: four results are explicit at the FFI boundary.
pub fn evaluate_single_program(
    engine: &Engine,
    source: &str,
    fuel: u64,
    count: u32,
) -> EvaluatorResult {
    let profile = wat::parse_str(source)
        .ok()
        .and_then(|b| Module::new(engine, b).ok())
        .and_then(|m| m.get_export("compute").and_then(|t| t.func().cloned()))
        .map(|f| {
            if f.results().len() == 4 {
                "i256x4_v1"
            } else {
                "i64_scalar_v1"
            }
        })
        .unwrap_or("i64_scalar_v1");
    evaluate_profile(engine, source, fuel, count, profile)
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
        assert!(
            res.status == ExecutionStatus::ExecutionTrap
                || res.status == ExecutionStatus::CompileError
        );
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
