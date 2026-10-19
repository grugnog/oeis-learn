pub mod engine;
pub mod sandbox;
pub mod synthesis_adapter;

use pyo3::prelude::*;

use sandbox::{evaluate_profile, Compiled, EvaluatorResult};

/// Python-visible execution result dictionary / struct
#[pyclass(dict)]
#[derive(Clone)]
pub struct ExecutionResult {
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub consumed_fuel: u64,
    #[pyo3(get)]
    pub max_fuel: u64,
    #[pyo3(get)]
    pub total_fuel: u64,
    #[pyo3(get)]
    pub output: Vec<i64>,
    #[pyo3(get)]
    pub wide_output: Vec<Vec<i64>>,
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl ExecutionResult {
    #[new]
    #[pyo3(signature = (status, consumed_fuel, output, error=None, max_fuel=None, total_fuel=None, wide_output=None))]
    fn new(
        status: String,
        consumed_fuel: u64,
        output: Vec<i64>,
        error: Option<String>,
        max_fuel: Option<u64>,
        total_fuel: Option<u64>,
        wide_output: Option<Vec<Vec<i64>>>,
    ) -> Self {
        ExecutionResult {
            status,
            consumed_fuel,
            max_fuel: max_fuel.unwrap_or(consumed_fuel),
            total_fuel: total_fuel.unwrap_or(consumed_fuel),
            output,
            wide_output: wide_output.unwrap_or_default(),
            error,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionResult(status='{}', consumed_fuel={}, max_fuel={}, total_fuel={}, output_len={}, error={:?})",
            self.status,
            self.consumed_fuel,
            self.max_fuel,
            self.total_fuel,
            self.output.len(),
            self.error
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<&'py pyo3::types::PyDict> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("status", &self.status)?;
        dict.set_item("consumed_fuel", self.consumed_fuel)?;
        dict.set_item("max_fuel", self.max_fuel)?;
        dict.set_item("total_fuel", self.total_fuel)?;
        dict.set_item("output", &self.output)?;
        dict.set_item("wide_output", &self.wide_output)?;
        dict.set_item("error", &self.error)?;
        Ok(dict)
    }
}

impl From<EvaluatorResult> for ExecutionResult {
    fn from(res: EvaluatorResult) -> Self {
        ExecutionResult {
            status: res.status.as_str().to_string(),
            consumed_fuel: res.consumed_fuel,
            max_fuel: res.max_fuel,
            total_fuel: res.total_fuel,
            output: res.output,
            wide_output: res
                .wide_output
                .into_iter()
                .map(|limbs| limbs.to_vec())
                .collect(),
            error: res.error,
        }
    }
}

/// Evaluates a single WAT program string in-memory with exact fuel metering and linear memory limits.
#[pyfunction]
#[pyo3(signature = (wat_code, fuel_budget=10000, terms_to_generate=20, result_profile="i64_scalar_v1"))]
pub fn evaluate_wat_single(
    py: Python<'_>,
    wat_code: &str,
    fuel_budget: u64,
    terms_to_generate: u32,
    result_profile: &str,
) -> PyResult<ExecutionResult> {
    let engine = engine::create_fuel_engine()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let res = py.allow_threads(|| {
        evaluate_profile(
            &engine,
            wat_code,
            fuel_budget,
            terms_to_generate,
            result_profile,
        )
    });
    Ok(res.into())
}

/// Evaluates a batch of WAT program strings concurrently across CPU worker threads using Rayon,
/// releasing the Python GIL.
#[pyfunction]
#[pyo3(signature = (wat_programs, fuel_budget=10000, terms_to_generate=20, result_profile="i64_scalar_v1"))]
pub fn evaluate_wat_batch(
    py: Python<'_>,
    wat_programs: Vec<String>,
    fuel_budget: u64,
    terms_to_generate: u32,
    result_profile: &str,
) -> PyResult<Vec<ExecutionResult>> {
    let engine = engine::create_fuel_engine()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    if wat_programs.len() > 32 {
        return Err(pyo3::exceptions::PyValueError::new_err("batch cap is 32"));
    }
    // The Python worker pool owns concurrency; native execution is single-threaded.
    let results: Vec<ExecutionResult> = py.allow_threads(|| {
        wat_programs
            .iter()
            .map(|wat| {
                let res =
                    evaluate_profile(&engine, wat, fuel_budget, terms_to_generate, result_profile);
                res.into()
            })
            .collect()
    });

    Ok(results)
}

#[pyclass]
pub struct NativeModule {
    compiled: Compiled,
}
#[pymethods]
impl NativeModule {
    #[new]
    fn new(source: &str) -> PyResult<Self> {
        let engine = engine::create_fuel_engine()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let compiled = Compiled::new(&engine, source, "wat_i256_checked_v1")
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { compiled })
    }
    fn term(
        &self,
        py: Python<'_>,
        index: i32,
        fuel: u64,
    ) -> PyResult<(Option<Vec<i64>>, u64, Option<String>, Option<String>)> {
        if !(0..100).contains(&index) || fuel > 1_000_000 {
            return Err(pyo3::exceptions::PyValueError::new_err("term bounds"));
        }
        let t = py.allow_threads(|| self.compiled.term(index, fuel));
        Ok((t.values, t.fuel, t.outcome, t.reason))
    }
    fn close(&self) {}
    fn serialize(&self) -> PyResult<Vec<u8>> {
        self.compiled
            .module
            .serialize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
#[pyfunction]
fn source_identity_bytes() -> Vec<u8> {
    include_bytes!(concat!(env!("OUT_DIR"), "/source_identity.bin")).to_vec()
}

/// A Python module implemented in Rust.
/// Dependency transport diagnostic: preserves exact numerals beyond machine width.
#[pyfunction]
fn synthesis_dependency_smoke() -> String {
    oeis_synthesis_core::dependency_smoke()
}

#[pymodule]
fn oeis_wasm_evaluator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synthesis_dependency_smoke, m)?)?;
    m.add_class::<ExecutionResult>()?;
    m.add_class::<NativeModule>()?;
    m.add_class::<LogicalModule>()?;
    m.add_function(wrap_pyfunction!(source_identity_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_wat_single, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_wat_batch, m)?)?;
    Ok(())
}

/// Explicit one-page logical profile. The Python typed lowering is the only
/// admission path; this low-level adapter never chooses a profile automatically.
#[pyclass]
pub struct LogicalModule {
    module: wasmtime::Module,
    engine: wasmtime::Engine,
    store: wasmtime::Store<sandbox::SandboxHostData>,
    instance: wasmtime::Instance,
    mode: String,
}

#[pymethods]
impl LogicalModule {
    #[new]
    fn new(source: &str, mode: &str) -> PyResult<Self> {
        use wasmtime::{ExternType, StoreLimitsBuilder, ValType};
        if !["pure", "stream"].contains(&mode)
            || source.len() > 524288
            || source.contains("memory.grow")
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "logical source/profile cap",
            ));
        }
        let engine = engine::create_fuel_engine()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let module = wasmtime::Module::new(&engine, source)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let memory = module
            .get_export("__memory")
            .and_then(|t| t.memory().cloned())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("one-page logical memory required")
            })?;
        if module.imports().next().is_some()
            || memory.minimum() != 1
            || memory.maximum() != Some(1)
            || memory.is_64()
            || memory.is_shared()
            || module
                .exports()
                .any(|e| matches!(e.ty(), ExternType::Table(_)))
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "logical memory/import escape",
            ));
        }
        let names = if mode == "pure" {
            vec![("compute", 1, 4)]
        } else {
            vec![("init", 1, 0), ("next", 0, 4)]
        };
        for (name, inputs, outputs) in names {
            let ty = module
                .get_export(name)
                .and_then(|t| t.func().cloned())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("logical entrypoint missing")
                })?;
            if ty.params().len() != inputs
                || !ty.params().all(|v| matches!(v, ValType::I32))
                || ty.results().len() != outputs
                || !ty.results().all(|v| matches!(v, ValType::I64))
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "logical exact signature mismatch",
                ));
            }
        }
        let limits = StoreLimitsBuilder::new()
            .memory_size(65536)
            .memories(1)
            .tables(0)
            .instances(1)
            .build();
        let mut store = wasmtime::Store::new(&engine, sandbox::SandboxHostData { limits });
        store.limiter(|d| &mut d.limits);
        store
            .set_fuel(1_000_000)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let instance = wasmtime::Instance::new(&mut store, &module, &[])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            module,
            engine,
            store,
            instance,
            mode: mode.into(),
        })
    }

    fn snapshot(&mut self) -> PyResult<(i64, Vec<i64>, Vec<u8>)> {
        use wasmtime::Val;
        let index = self
            .instance
            .get_global(&mut self.store, "__index")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("logical index missing"))?
            .get(&mut self.store);
        let index = match index {
            Val::I64(v) => v,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "logical index type",
                ))
            }
        };
        let mut slots = vec![];
        for s in 0..32 {
            for i in 0..4 {
                let g = self
                    .instance
                    .get_global(&mut self.store, &format!("__slot{s}_{i}"))
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("logical slot missing")
                    })?;
                match g.get(&mut self.store) {
                    Val::I64(v) => slots.push(v),
                    _ => return Err(pyo3::exceptions::PyValueError::new_err("logical slot type")),
                }
            }
        }
        let memory = self
            .instance
            .get_memory(&mut self.store, "__memory")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("logical memory missing"))?;
        Ok((index, slots, memory.data(&self.store).to_vec()))
    }

    fn restore(&mut self, index: i64, slots: Vec<i64>, memory: Vec<u8>) -> PyResult<()> {
        use wasmtime::Val;
        if slots.len() != 128 || memory.len() != 65536 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "logical snapshot dimensions",
            ));
        }
        let index_global = self
            .instance
            .get_global(&mut self.store, "__index")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("logical index missing"))?;
        index_global
            .set(&mut self.store, Val::I64(index))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        for s in 0..32 {
            for i in 0..4 {
                let g = self
                    .instance
                    .get_global(&mut self.store, &format!("__slot{s}_{i}"))
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("logical slot missing")
                    })?;
                g.set(&mut self.store, Val::I64(slots[s * 4 + i]))
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            }
        }
        let m = self
            .instance
            .get_memory(&mut self.store, "__memory")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("logical memory missing"))?;
        m.write(&mut self.store, 0, &memory)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let marker = self
            .instance
            .get_global(&mut self.store, "__oeis_numeric_limit")
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("logical overflow marker missing")
            })?;
        marker
            .set(&mut self.store, Val::I32(0))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[pyo3(signature = (name, index, fuel))]
    fn call(
        &mut self,
        name: &str,
        index: Option<i32>,
        fuel: u64,
    ) -> PyResult<(Option<Vec<i64>>, u64, Option<String>, Option<String>)> {
        use wasmtime::{StoreLimitsBuilder, Trap, Val};
        if fuel == 0
            || fuel > 1_000_000
            || !["init", "next", "compute"].contains(&name)
            || (self.mode == "pure") != (name == "compute")
            || (name == "next") != index.is_none()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "logical call/profile bounds",
            ));
        }
        if name == "compute" || name == "init" {
            let limits = StoreLimitsBuilder::new()
                .memory_size(65536)
                .memories(1)
                .tables(0)
                .instances(1)
                .build();
            self.store = wasmtime::Store::new(&self.engine, sandbox::SandboxHostData { limits });
            self.store.limiter(|d| &mut d.limits);
            self.store
                .set_fuel(fuel)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.instance = wasmtime::Instance::new(&mut self.store, &self.module, &[])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        let before = self.snapshot()?;
        self.store
            .set_fuel(fuel)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let func = self
            .instance
            .get_func(&mut self.store, name)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("logical entrypoint missing"))?;
        let args: Vec<Val> = index.into_iter().map(Val::I32).collect();
        let mut values = vec![Val::I64(0); if name == "init" { 0 } else { 4 }];
        let result = func.call(&mut self.store, &args, &mut values);
        let used = fuel.saturating_sub(self.store.get_fuel().unwrap_or(0));
        match result {
            Ok(()) => Ok((
                Some(
                    values
                        .into_iter()
                        .map(|v| v.i64().expect("validated i64 signature"))
                        .collect(),
                ),
                used,
                None,
                None,
            )),
            Err(error) => {
                let marker = self
                    .instance
                    .get_global(&mut self.store, "__oeis_numeric_limit")
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("logical overflow marker missing")
                    })?;
                let outcome = if matches!(marker.get(&mut self.store), Val::I32(1)) {
                    "numeric_limit"
                } else if matches!(error.downcast_ref::<Trap>(), Some(Trap::OutOfFuel)) {
                    "execution_limit"
                } else {
                    "runtime_failure"
                };
                self.restore(before.0, before.1, before.2)?;
                Ok((None, used, Some(outcome.into()), Some(error.to_string())))
            }
        }
    }
}

#[cfg(test)]
mod logical_tests {
    use super::LogicalModule;
    const SOURCE: &str = r#"(module
      (memory (export "__memory") 1 1)
      (global (export "__oeis_numeric_limit") (mut i32) (i32.const 0))
      (global $index (export "__index") (mut i64) (i64.const 0))
      SLOTS
      (func (export "init") (param $n i32) local.get $n i64.extend_i32_s global.set $index)
      (func (export "next") (result i64 i64 i64 i64)
        global.get $index i64.const 1 i64.add global.set $index
        global.get $index i64.const 0 i64.const 0 i64.const 0))"#;
    fn source() -> String {
        let mut slots = String::new();
        for s in 0..32 {
            for i in 0..4 {
                slots.push_str(&format!(
                    "(global (export \"__slot{s}_{i}\") (mut i64) (i64.const 0))"
                ));
            }
        }
        SOURCE.replace("SLOTS", &slots)
    }
    #[test]
    fn logical_stream_snapshot_reset_and_fuel_rollback() {
        let mut runtime = LogicalModule::new(&source(), "stream").unwrap();
        assert!(runtime.call("init", Some(4), 1000).unwrap().2.is_none());
        assert_eq!(runtime.call("next", None, 1000).unwrap().0.unwrap()[0], 5);
        let state = runtime.snapshot().unwrap();
        assert_eq!(
            runtime.call("next", None, 1).unwrap().2.as_deref(),
            Some("execution_limit")
        );
        assert_eq!(runtime.snapshot().unwrap().0, state.0);
        runtime.call("init", Some(0), 1000).unwrap();
        assert_eq!(runtime.call("next", None, 1000).unwrap().0.unwrap()[0], 1);
        runtime.restore(state.0, state.1, state.2).unwrap();
        assert_eq!(runtime.call("next", None, 1000).unwrap().0.unwrap()[0], 6);
    }
}

/// Framed native service. Core algorithms have no dependency on this binding.
pub fn serve_synthesis() -> Result<(), String> {
    oeis_synthesis_core::transport::serve(&mut synthesis_adapter::CheckedSigned256::default())
}
