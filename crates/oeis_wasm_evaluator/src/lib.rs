pub mod engine;
pub mod sandbox;

use pyo3::prelude::*;
use rayon::prelude::*;
use sandbox::{evaluate_single_program, EvaluatorResult};

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
#[pyo3(signature = (wat_code, fuel_budget=10000, terms_to_generate=20))]
pub fn evaluate_wat_single(
    wat_code: &str,
    fuel_budget: u64,
    terms_to_generate: u32,
) -> PyResult<ExecutionResult> {
    let engine = engine::create_fuel_engine()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let res = evaluate_single_program(&engine, wat_code, fuel_budget, terms_to_generate);
    Ok(res.into())
}

/// Evaluates a batch of WAT program strings concurrently across CPU worker threads using Rayon,
/// releasing the Python GIL.
#[pyfunction]
#[pyo3(signature = (wat_programs, fuel_budget=10000, terms_to_generate=20))]
pub fn evaluate_wat_batch(
    py: Python<'_>,
    wat_programs: Vec<String>,
    fuel_budget: u64,
    terms_to_generate: u32,
) -> PyResult<Vec<ExecutionResult>> {
    let engine = engine::create_fuel_engine()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Release the Python GIL and evaluate in parallel with Rayon
    let results: Vec<ExecutionResult> = py.allow_threads(|| {
        wat_programs
            .par_iter()
            .map(|wat| {
                let res = evaluate_single_program(&engine, wat, fuel_budget, terms_to_generate);
                res.into()
            })
            .collect()
    });

    Ok(results)
}

/// A Python module implemented in Rust.
#[pymodule]
fn oeis_wasm_evaluator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionResult>()?;
    m.add_function(wrap_pyfunction!(evaluate_wat_single, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_wat_batch, m)?)?;
    Ok(())
}
