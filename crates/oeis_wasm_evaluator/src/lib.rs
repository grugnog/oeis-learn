pub mod engine;
pub mod sandbox;

use pyo3::prelude::*;

use sandbox::{evaluate_profile, EvaluatorResult, Compiled};

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
    let res = py.allow_threads(|| evaluate_profile(&engine, wat_code, fuel_budget, terms_to_generate, result_profile));
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

    if wat_programs.len()>32 {return Err(pyo3::exceptions::PyValueError::new_err("batch cap is 32"));}
    // The Python worker pool owns concurrency; native execution is single-threaded.
    let results: Vec<ExecutionResult> = py.allow_threads(|| {
        wat_programs
            .iter()
            .map(|wat| {
                let res = evaluate_profile(&engine, wat, fuel_budget, terms_to_generate, result_profile);
                res.into()
            })
            .collect()
    });

    Ok(results)
}


#[pyclass]
pub struct NativeModule { compiled:Compiled }
#[pymethods]
impl NativeModule {
    #[new]
    fn new(source:&str)->PyResult<Self> {
        let engine=engine::create_fuel_engine().map_err(|e|pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let compiled=Compiled::new(&engine,source,"wat_i256_checked_v1").map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {compiled})
    }
    fn term(&self,py:Python<'_>,index:i32,fuel:u64)->PyResult<(Option<Vec<i64>>,u64,Option<String>,Option<String>)> {
        if !(0..100).contains(&index) || fuel>1_000_000 {return Err(pyo3::exceptions::PyValueError::new_err("term bounds"));}
        let t=py.allow_threads(||self.compiled.term(index,fuel));
        Ok((t.values,t.fuel,t.outcome,t.reason))
    }
    fn close(&self) {}
    fn serialize(&self)->PyResult<Vec<u8>> {
        self.compiled.module.serialize().map_err(|e|pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
#[pyfunction]
fn source_identity_bytes()->Vec<u8> {include_bytes!(concat!(env!("OUT_DIR"),"/source_identity.bin")).to_vec()}

/// A Python module implemented in Rust.
#[pymodule]
fn oeis_wasm_evaluator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionResult>()?;
    m.add_class::<NativeModule>()?;
    m.add_function(wrap_pyfunction!(source_identity_bytes,m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_wat_single, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_wat_batch, m)?)?;
    Ok(())
}
