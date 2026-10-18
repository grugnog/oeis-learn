//! Fail-closed Phase 1 transport shapes. No result here can grant admission.
use dashu_int::IBig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const VERSION: &str = "synthesis-worker/v1";
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
pub const RECORD_SCHEMAS: &str =
    include_str!("../../../src/oeis_learn/synthesis/records.schema.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub candidate_id: String,
    pub original_source: String,
    pub original_source_sha256: String,
    pub codec_profile: String,
    pub parameter_ids: Vec<String>,
    pub abstraction_policy: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub wall_time_ms: u32,
    pub attempts: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Fit,
    Execute,
    ReplayUnsat,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub protocol: String,
    pub request_id: String,
    pub operation: Operation,
    pub worker_generation: u32,
    pub contract_sha256: String,
    pub language_profile: String,
    pub resource_profile: String,
    pub solver_profile: String,
    pub candidates: Vec<Candidate>,
    pub visible_indices: Vec<String>,
    pub visible_values: Vec<String>,
    pub remaining_budget: Budget,
    pub seed: u32,
}

fn canonical_integer(s: &str) -> bool {
    if s.is_empty() || s.len() > 1235 {
        return false;
    }
    if s == "0" {
        return true;
    }
    let digits = s.strip_prefix('-').unwrap_or(s).as_bytes();
    !digits.is_empty() && digits[0] != b'0' && digits.iter().all(u8::is_ascii_digit)
}

fn hash(s: &str) -> bool {
    s.strip_prefix("sha256:").is_some_and(|v| {
        v.len() == 64
            && v.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

pub fn parse_request(bytes: &[u8]) -> Result<Request, String> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err("frame exceeds 4 MiB".into());
    }
    let request: Request = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    request.validate()?;
    Ok(request)
}

impl Request {
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol != VERSION
            || self.language_profile != "bigint-body/v1"
            || self.resource_profile != "bigint-resources/v1"
            || self.solver_profile != "native-fitting/v1"
            || !identifier(&self.request_id)
            || !hash(&self.contract_sha256)
        {
            return Err("unknown version/profile or invalid identity".into());
        }
        if self.candidates.is_empty()
            || self.candidates.len() > 32
            || self.visible_indices.is_empty()
            || self.visible_indices.len() > 20
            || self.visible_indices.len() != self.visible_values.len()
            || self.remaining_budget.attempts == 0
            || self.remaining_budget.attempts > 32
            || self.remaining_budget.wall_time_ms == 0
            || self.remaining_budget.wall_time_ms > 60000
        {
            return Err("invalid count/budget".into());
        }
        if !self
            .visible_indices
            .iter()
            .chain(&self.visible_values)
            .all(|s| canonical_integer(s))
        {
            return Err("noncanonical exact integer".into());
        }
        let indices: Vec<IBig> = self
            .visible_indices
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        if !indices.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err("visible indices must strictly increase".into());
        }
        let mut ids = BTreeSet::new();
        for c in &self.candidates {
            let params: BTreeSet<_> = c.parameter_ids.iter().collect();
            if !identifier(&c.candidate_id)
                || !ids.insert(&c.candidate_id)
                || c.original_source.is_empty()
                || c.original_source.len() > 65536
                || c.codec_profile != "bigint-codec/v1"
                || !["none", "eligible-literals/v1"].contains(&c.abstraction_policy.as_str())
                || c.parameter_ids.len() > 8
                || params.len() != c.parameter_ids.len()
                || !c.parameter_ids.iter().all(|p| {
                    matches!(
                        p.as_str(),
                        "p0" | "p1" | "p2" | "p3" | "p4" | "p5" | "p6" | "p7"
                    )
                })
                || format!("sha256:{:x}", Sha256::digest(c.original_source.as_bytes()))
                    != c.original_source_sha256
            {
                return Err("invalid candidate".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct UnsupportedResponse<'a> {
    protocol: &'static str,
    request_id: &'a str,
    worker_generation: u32,
    outcome: &'static str,
    reason: &'static str,
    admitted: bool,
    assignments: Vec<String>,
}

pub fn handle_request(request: &Request) -> Result<UnsupportedResponse<'_>, String> {
    request.validate()?;
    Ok(UnsupportedResponse {
        protocol: VERSION,
        request_id: &request.request_id,
        worker_generation: request.worker_generation,
        outcome: "unsupported",
        reason: "phase_2_not_implemented",
        admitted: false,
        assignments: vec![],
    })
}

/// Record envelopes are inert references, never semantic admission evidence.
pub fn validate_record(kind: &str, value: &serde_json::Value) -> Result<(), String> {
    let schemas: serde_json::Value = serde_json::from_str(RECORD_SCHEMAS).unwrap();
    let schema = schemas["$defs"]
        .get(kind)
        .filter(|_| kind != "Request")
        .ok_or("unknown record kind")?;
    let object = value.as_object().ok_or("record must be an object")?;
    let fields = schema["properties"].as_object().unwrap();
    if object.len() != fields.len() || !fields.keys().all(|k| object.contains_key(k)) {
        return Err("unknown or missing record field".into());
    }
    for (key, field) in fields {
        let v = &object[key];
        let valid = if let Some(options) = field.get("enum") {
            options.as_array().unwrap().contains(v)
        } else if let Some(constant) = field.get("const") {
            v == constant
        } else if field["type"] == "array" {
            v.as_array().is_some_and(|a| {
                !a.is_empty()
                    && a.len() <= 20
                    && a.iter().all(|v| v.as_str().is_some_and(canonical_integer))
            })
        } else {
            v.as_str().is_some_and(hash)
        };
        if !valid {
            return Err(format!("invalid record field: {key}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const REQUEST: &[u8] = include_bytes!("../../../tests/fixtures/bigint/request.json");

    #[test]
    fn stub_cannot_admit() {
        let request = parse_request(REQUEST).unwrap();
        let response = serde_json::to_value(handle_request(&request).unwrap()).unwrap();
        assert_eq!(response["outcome"], "unsupported");
        assert_eq!(response["admitted"], false);
        assert_eq!(response["assignments"], serde_json::json!([]));
    }

    #[test]
    fn malformed_and_unknown_fields_are_rejected() {
        for (field, bad) in [
            ("hidden_values", serde_json::json!(["3"])),
            ("protocol", serde_json::json!("unknown")),
            ("worker_generation", serde_json::json!(true)),
            ("visible_values", serde_json::json!([4])),
            ("visible_values", serde_json::json!(["-0"])),
            ("visible_values", serde_json::json!(["1\n"])),
            ("language_profile", serde_json::json!("unknown")),
        ] {
            let mut value: serde_json::Value = serde_json::from_slice(REQUEST).unwrap();
            value[field] = bad;
            assert!(parse_request(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        assert!(parse_request(&vec![b' '; MAX_FRAME_BYTES + 1]).is_err());
    }

    #[test]
    fn records_are_strict_and_disabled() {
        let records: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/bigint/record-stubs.json"
        ))
        .unwrap();
        for (kind, value) in records.as_object().unwrap() {
            validate_record(kind, value).unwrap();
            let mut invalid = value.clone();
            invalid["enabled"] = serde_json::json!(true);
            assert!(validate_record(kind, &invalid).is_err());
            invalid = value.clone();
            invalid["hidden_values"] = serde_json::json!([]);
            assert!(validate_record(kind, &invalid).is_err());
        }
    }
}
