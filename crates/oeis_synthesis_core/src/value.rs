//! Canonical exact values and content-addressed finite resource profiles.
use dashu_base::BitTest;
use dashu_int::IBig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub fn bits(n: &IBig) -> usize { n.bit_len() }
pub fn integer(text: &str, limit: usize) -> Result<IBig, String> {
    if limit == 0 || limit > 131072 || text.len() > limit * 30103 / 100000 + 2 {
        return Err("literal_limit".into());
    }
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit())
        || (digits.starts_with('0') && text != "0") {
        return Err("invalid_source".into());
    }
    let value: IBig = text.parse().map_err(|_| "invalid_source")?;
    if bits(&value) > limit { return Err("literal_limit".into()); }
    Ok(value)
}
pub fn digest(bytes: &[u8]) -> String { format!("sha256:{:x}", Sha256::digest(bytes)) }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Resources {
    pub input_bits: usize, pub literal_bits: usize, pub intermediate_bits: usize,
    pub output_bytes: usize, pub arena_entries: usize, pub arena_bytes: usize,
    pub memory_bytes: usize, pub term_fuel: u64, pub aggregate_fuel: u64,
    pub reference_steps: u64, pub aggregate_reference_steps: u64,
    pub control_max: u32, pub loop_iterations: u32, pub aggregate_loop_iterations: u32,
    pub ast_nodes: usize, pub source_bytes: usize, pub nesting: usize,
    pub bigint_locals: usize, pub control_locals: usize, pub bool_locals: usize,
    pub term_seconds: f64, pub evaluation_seconds: f64, pub cleanup_seconds: f64,
}
impl Default for Resources {
    fn default() -> Self { Self {
        input_bits:4096,literal_bits:4096,intermediate_bits:16384,output_bytes:2048,
        arena_entries:65536,arena_bytes:32*1024*1024,memory_bytes:64*1024*1024,
        term_fuel:10_000_000,aggregate_fuel:100_000_000,reference_steps:1_000_000,
        aggregate_reference_steps:50_000_000,control_max:65536,loop_iterations:1024,
        aggregate_loop_iterations:65536,ast_nodes:4096,source_bytes:65536,nesting:32,
        bigint_locals:16,control_locals:8,bool_locals:8,term_seconds:2.0,
        evaluation_seconds:30.0,cleanup_seconds:5.0,
    }}
}
impl Resources {
    pub fn validate(&self) -> Result<(),String> {
        let v=serde_json::to_value(self).map_err(|_| "unsupported_profile")?;
        for (key,value) in v.as_object().ok_or("unsupported_profile")? {
            let n=value.as_f64().ok_or("unsupported_profile")?;
            if !n.is_finite() || n <= 0.0 || n > 9007199254740991.0
                || (key.ends_with("seconds") && n > 3600.0) { return Err("unsupported_profile".into()); }
        }
        if self.input_bits.max(self.literal_bits)>self.intermediate_bits || self.intermediate_bits>131072
            || self.arena_bytes>self.memory_bytes || self.memory_bytes>1024*1024*1024
            || self.source_bytes>65536 || self.ast_nodes>4096 || self.nesting>32
            || self.bigint_locals>16 || self.control_locals>8 || self.bool_locals>8
            || self.aggregate_loop_iterations>i32::MAX as u32 || self.control_max>i32::MAX as u32 || self.loop_iterations>1024
            || self.arena_entries>i32::MAX as usize || self.output_bytes>16384 {
            return Err("unsupported_profile".into());
        }
        Ok(())
    }
    pub fn identity(&self)->String {
        let mut v=serde_json::to_value(self).unwrap();
        v.as_object_mut().unwrap().insert("version".into(),"bigint-resources/v1".into());
        digest(serde_json::to_string(&v).unwrap().as_bytes())
    }
}

/// Exact limb transport for the named retained checked profile, not bigint execution.
pub fn checked256_limbs(text:&str)->Result<[i64;4],String> {
    let n=integer(text,256)?;
    let half=IBig::ONE<<255usize;
    if n < -&half || n>=half {return Err("invalid_source".into());}
    let unsigned=if n<IBig::ZERO {n+(IBig::ONE<<256usize)} else {n};
    let mut raw=unsigned.into_parts().1.to_le_bytes().into_vec();raw.resize(32,0);
    Ok(std::array::from_fn(|i|i64::from_le_bytes(raw[i*8..i*8+8].try_into().unwrap())))
}
pub fn signed256_decimal(limbs:&[i64])->Result<String,String> {
    if limbs.len()!=4 {return Err("runtime_failure".into());}
    let mut n=IBig::ZERO;
    for (i,x) in limbs.iter().enumerate() {n+=IBig::from(*x as u64)<<(64*i);}
    if limbs[3]<0 {n-=IBig::ONE<<256usize;}
    Ok(n.to_string())
}
