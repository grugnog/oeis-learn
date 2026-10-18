//! Private, immutable append-only bigint arena. All candidate arithmetic is WASM.
//! Host transfer/status exports must NEVER be registered in the candidate linker.
use dashu_int::{IBig, UBig};
use std::cell::RefCell;

const INVALID_HANDLE:i32=1;
const INTEGER_LIMIT:i32=2;
const LITERAL_LIMIT:i32=3;
const OUTPUT_LIMIT:i32=4;
const ARENA_LIMIT:i32=5;
const CONTROL_RANGE:i32=6;
const RUNTIME_FAILURE:i32=8;
struct Arena {
    values:Vec<IBig>, literals:Vec<i32>, bytes:usize, status:i32,
    input_bits:usize,literal_bits:usize,bits:usize,entries:usize,byte_cap:usize,
    output_cap:usize,control:u32,sealed:bool,loading:Option<bool>,buffer:Vec<u8>,
    output:Vec<u8>,output_sign:i32,
}
thread_local! { static STATE:RefCell<Option<Arena>>=const { RefCell::new(None) }; }
impl Arena {
    fn fail(&mut self,status:i32)->i32 { if self.status==0 { self.status=status; } 0 }
    fn insert(&mut self,n:IBig)->i32 {
        if self.status!=0 { return 0; }
        let bits=n.clone().into_parts().1.bit_len();
        if bits>self.bits { return self.fail(INTEGER_LIMIT); }
        // Stable logical allocation accounting includes a 32-byte slot charge.
        // Linear-memory limiter additionally bounds Vec capacity/allocator slack.
        let cost=32+bits.div_ceil(8).max(1);
        if self.values.len()>=self.entries || cost>self.byte_cap.saturating_sub(self.bytes) { return self.fail(ARENA_LIMIT); }
        self.bytes+=cost; self.values.push(n); self.values.len() as i32
    }
    fn get(&mut self,h:i32)->Option<IBig> {
        if self.status!=0 { return None; }
        if h<=0 || h as usize>self.values.len() { self.fail(INVALID_HANDLE); return None; }
        Some(self.values[h as usize-1].clone())
    }
}
fn with(f:impl FnOnce(&mut Arena)->i32)->i32 { STATE.with(|s|match s.borrow_mut().as_mut() { Some(a)=>f(a),None=>0 }) }

#[no_mangle]
pub extern "C" fn host_init(input_bits:i32,literal_bits:i32,bits:i32,entries:i32,bytes:i32,output:i32,control:i32)->i32 {
    STATE.with(|s| {
        let mut state=s.borrow_mut();
        if let Some(a)=state.as_mut() { return a.fail(RUNTIME_FAILURE); }
        if input_bits<=0 || literal_bits<=0 || bits<input_bits.max(literal_bits) || bits>131072
            || entries<=0 || bytes<=0 || output<=0 || output>16384 || control<=0 { return 0; }
        *state=Some(Arena{values:Vec::new(),literals:Vec::new(),bytes:0,status:0,
            input_bits:input_bits as usize,literal_bits:literal_bits as usize,bits:bits as usize,
            entries:entries as usize,byte_cap:bytes as usize,output_cap:output as usize,control:control as u32,
            sealed:false,loading:None,buffer:Vec::new(),output:Vec::new(),output_sign:0});
        1
    })
}
#[no_mangle]
pub extern "C" fn host_begin_load(literal:i32)->i32 {
    with(|a| {
        if a.status!=0 { return 0; }
        if a.sealed || a.loading.is_some() || !(0..=1).contains(&literal) { return a.fail(RUNTIME_FAILURE); }
        a.loading=Some(literal==1); a.buffer.clear(); 1
    })
}
#[no_mangle]
pub extern "C" fn host_load_byte(byte:i32)->i32 {
    with(|a| {
        if a.status!=0 { return 0; }
        let Some(literal)=a.loading else { return a.fail(RUNTIME_FAILURE); };
        let bound=if literal {a.literal_bits} else {a.input_bits};
        if !(0..=255).contains(&byte) || a.buffer.len()>=bound.div_ceil(8) { return a.fail(LITERAL_LIMIT); }
        a.buffer.push(byte as u8); 1
    })
}
#[no_mangle]
pub extern "C" fn host_finish_load(sign:i32)->i32 {
    with(|a| {
        if a.status!=0 { return 0; }
        let Some(literal)=a.loading.take() else { return a.fail(RUNTIME_FAILURE); };
        if !(-1..=1).contains(&sign) || a.buffer.is_empty() || (a.buffer.len()>1 && a.buffer.last()==Some(&0)) { return a.fail(LITERAL_LIMIT); }
        let magnitude=UBig::from_le_bytes(&a.buffer);
        let limit=if literal {a.literal_bits} else {a.input_bits};
        if magnitude.bit_len()>limit || (magnitude==UBig::ZERO)!=(sign==0) { return a.fail(LITERAL_LIMIT); }
        let mut value=IBig::from(magnitude); if sign<0 { value=-value; }
        let h=a.insert(value);
        if literal && h!=0 { a.literals.push(h); }
        a.buffer.clear(); h
    })
}
#[no_mangle]
pub extern "C" fn host_seal()->i32 { with(|a| { if a.loading.is_some() {return a.fail(RUNTIME_FAILURE);} a.sealed=true; i32::from(a.status==0) }) }
#[no_mangle]
pub extern "C" fn host_status()->i32 { STATE.with(|s|s.borrow().as_ref().map_or(RUNTIME_FAILURE,|a|a.status)) }
#[no_mangle]
pub extern "C" fn host_entries()->i32 { with(|a|a.values.len() as i32) }
#[no_mangle]
pub extern "C" fn host_value_bytes()->i32 { with(|a|a.bytes as i32) }
#[no_mangle]
pub extern "C" fn from_i64(n:i64)->i32 { with(|a|a.insert(IBig::from(n))) }
#[no_mangle]
pub extern "C" fn constant(ordinal:i32)->i32 { with(|a| {
    if a.status!=0 {return 0;}
    if ordinal<0 || ordinal as usize>=a.literals.len() {return a.fail(INVALID_HANDLE);}
    a.literals[ordinal as usize]
}) }
fn binary(x:i32,y:i32,op:u8)->i32 { with(|a| {
    let Some(x)=a.get(x) else {return 0;}; let Some(y)=a.get(y) else {return 0;};
    // Operands already bounded. Temporaries need at most 2*bits plus allocator
    // overhead, bounded by the separate linear-memory limit. Never reject from
    // an upper-bound estimate: zero, cancellation and borderline products pass.
    let value=match op {0=>x+y,1=>x-y,_=>x*y}; a.insert(value)
}) }
#[no_mangle]
pub extern "C" fn add(x:i32,y:i32)->i32 {binary(x,y,0)}
#[no_mangle]
pub extern "C" fn sub(x:i32,y:i32)->i32 {binary(x,y,1)}
#[no_mangle]
pub extern "C" fn mul(x:i32,y:i32)->i32 {binary(x,y,2)}
#[no_mangle]
pub extern "C" fn cmp(x:i32,y:i32)->i32 {with(|a| {
    let Some(x)=a.get(x) else {return 0;}; let Some(y)=a.get(y) else {return 0;};
    match x.cmp(&y) {std::cmp::Ordering::Less=>-1,std::cmp::Ordering::Equal=>0,std::cmp::Ordering::Greater=>1}
})}
#[no_mangle]
pub extern "C" fn to_control(x:i32)->i32 {with(|a| {
    let Some(x)=a.get(x) else {return 0;};
    match u32::try_from(&x) {Ok(n) if n<=a.control=>n as i32,_=>a.fail(CONTROL_RANGE)}
})}
#[no_mangle]
pub extern "C" fn host_serialize(h:i32)->i32 {with(|a| {
    let Some(n)=a.get(h) else {return 0;};
    let magnitude=n.clone().into_parts().1;
    if magnitude.bit_len().div_ceil(8).max(1)>a.output_cap {return a.fail(OUTPUT_LIMIT);}
    a.output=magnitude.to_le_bytes(); if a.output.is_empty() {a.output.push(0);}
    a.output_sign=if n<IBig::ZERO {-1} else if n==IBig::ZERO {0} else {1};
    a.output.len() as i32
})}
#[no_mangle]
pub extern "C" fn host_output_sign()->i32 {with(|a|if a.status==0 {a.output_sign} else {0})}
#[no_mangle]
pub extern "C" fn host_output_byte(index:i32)->i32 {with(|a| {
    if a.status!=0 {return 0;}
    if index<0 || index as usize>=a.output.len() {return a.fail(OUTPUT_LIMIT);}
    a.output[index as usize] as i32
})}
#[no_mangle]
pub extern "C" fn dependency_smoke()->i32 {
    let n:IBig="18446744073709551616".parse().unwrap();
    i32::from((&n*&n).to_string()=="340282366920938463463374607431768211456")
}
#[no_mangle]
pub extern "C" fn production_enabled()->i32 {0}

#[cfg(test)]
mod tests {
    use super::*;
    // Fresh OS test thread provides a new thread-local arena, as each WASM term
    // does in production. No reset/free operation exists in the exported ABI.
    #[test] fn full_product_borderline_and_sticky_handles() {
        assert_eq!(host_init(8,8,8,20,10000,1,1024),1);
        let a=from_i64(128); let b=from_i64(1); let c=mul(a,b);
        assert_ne!(c,0); assert_eq!(host_serialize(c),1);assert_eq!(host_output_byte(0),128);
        assert_eq!(add(0,a),0); assert_eq!(host_status(),INVALID_HANDLE);
        assert_eq!(host_serialize(c),0); assert_eq!(from_i64(2),0);
    }
    #[test] fn large_operands_and_no_reset() {
        assert_eq!(host_init(4096,4096,16384,65536,32*1024*1024,2048,65536),1);
        let a=from_i64(i64::MAX); let b=mul(a,a); let c=mul(b,b);
        let d=mul(c,c); assert_ne!(d,0);assert!(host_serialize(d)>32);
        assert_eq!(host_init(1,1,1,1,1,1,1),0);assert_eq!(host_status(),RUNTIME_FAILURE);
    }
}
