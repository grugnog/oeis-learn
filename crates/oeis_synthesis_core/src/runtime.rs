//! Fresh two-instance Wasmtime stores. Arithmetic/transfer/serialization all fuel.
use std::{sync::mpsc,thread,time::{Duration,Instant}};
use dashu_int::{IBig,UBig};
use serde::Serialize;
use wasmtime::{Config,Engine,Instance,Linker,Module,Store,StoreLimits,StoreLimitsBuilder,ResourceLimiter,Trap,TypedFunc};
use crate::{language,lowering::{self,ABI},value::{integer,Resources}};

struct Limits { inner:StoreLimits, memory_denied:bool }
impl ResourceLimiter for Limits {
    fn memory_growing(&mut self,current:usize,desired:usize,maximum:Option<usize>)->wasmtime::Result<bool> {
        let result=self.inner.memory_growing(current,desired,maximum);
        if !matches!(&result,Ok(true)) {self.memory_denied=true;}
        result
    }
    fn table_growing(&mut self,current:u32,desired:u32,maximum:Option<u32>)->wasmtime::Result<bool> {
        self.inner.table_growing(current,desired,maximum)
    }
    fn memory_grow_failed(&mut self,error:wasmtime::Error)->wasmtime::Result<()> {
        self.memory_denied=true;Err(error)
    }
    fn instances(&self)->usize {2}
    fn memories(&self)->usize {1}
    fn tables(&self)->usize {1}
}
#[derive(Debug,Serialize)]
pub struct Evaluation {
    pub outcome:String,pub reason:Option<String>,pub values:Vec<String>,
    pub coverage:usize,pub fuel:u64,pub resource_identity:String,
}
/// Observation-only timings. They do not alter limits or public execution records.
#[derive(Default,Serialize)]
pub struct Measurements {
    pub prepare_seconds:f64,
    pub execution_seconds:f64,
    pub total_seconds:f64,
    #[serde(skip_serializing_if="Option::is_none")]
    pub stages:Option<StageMeasurements>,
}
pub const PROFILE_PHASES:[&str;10]=["literal_decode","store_setup","clock_start","guest_init","input_load","link_instantiate","compute","serialize","clock_stop","store_drop"];
#[derive(Default,Serialize)]
pub struct PhaseMeasurement {pub calls:usize,pub seconds:f64}
pub const INPUT_LOAD_PHASES:[&str;7]=["index_parse","begin_load","magnitude_encode","byte_transfer","finish_load","status_check","seal"];
#[derive(Serialize)]
pub struct InputLoadMeasurements {
    pub policy:&'static str,
    pub phases:std::collections::BTreeMap<&'static str,PhaseMeasurement>,
    pub literal_values_started:usize,pub index_values_started:usize,pub values_finished:usize,
    pub byte_calls_attempted:usize,pub byte_calls_returned:usize,pub unattributed_seconds:f64,
}
impl Default for InputLoadMeasurements {
    fn default()->Self {Self{policy:"bigint-input-load-profile/v1",phases:INPUT_LOAD_PHASES.into_iter().map(|name|(name,PhaseMeasurement::default())).collect(),literal_values_started:0,index_values_started:0,values_finished:0,byte_calls_attempted:0,byte_calls_returned:0,unattributed_seconds:0.0}}
}
struct InputObserver<'a> {detail:Option<&'a mut InputLoadMeasurements>}
impl InputObserver<'_> {
    fn phase<T>(&mut self,name:&'static str,work:impl FnOnce(&mut Self)->T)->T {
        let start=self.detail.as_ref().map(|_|Instant::now());
        let result=work(self);
        if let Some(start)=start {
            let seconds=start.elapsed().as_secs_f64();
            let phase=self.detail.as_deref_mut().unwrap().phases.get_mut(name).expect("fixed input observation phase");
            phase.calls+=1;phase.seconds+=seconds;
        }
        result
    }
}
#[derive(Serialize)]
pub struct StageMeasurements {
    pub policy:&'static str,
    pub phases:std::collections::BTreeMap<&'static str,PhaseMeasurement>,
    pub terms_started:usize,pub terms_finished:usize,pub unattributed_seconds:f64,
    #[serde(skip_serializing_if="Option::is_none")]
    pub input_load:Option<InputLoadMeasurements>,
}
impl Default for StageMeasurements {
    fn default()->Self {Self{policy:"bigint-runtime-profile/v1",phases:PROFILE_PHASES.into_iter().map(|name|(name,PhaseMeasurement::default())).collect(),terms_started:0,terms_finished:0,unattributed_seconds:0.0,input_load:None}}
}
struct Observer<'a> {stages:Option<&'a mut StageMeasurements>}
impl Observer<'_> {
    fn phase<T>(&mut self,name:&'static str,work:impl FnOnce()->T)->T {
        // Ordinary production/aggregate measurements do not read the phase clock.
        if let Some(stages)=self.stages.as_deref_mut() {
            let start=Instant::now();let result=work();let seconds=start.elapsed().as_secs_f64();
            let phase=stages.phases.get_mut(name).expect("fixed observation phase");
            phase.calls+=1;phase.seconds+=seconds;result
        } else {work()}
    }
    fn input_phase<T>(&mut self,work:impl FnOnce(&mut InputObserver<'_>)->T)->T {
        let start=self.stages.as_ref().map(|_|Instant::now());
        let result=work(&mut InputObserver{detail:self.stages.as_deref_mut().and_then(|s|s.input_load.as_mut())});
        if let Some(start)=start {
            let seconds=start.elapsed().as_secs_f64();
            let phase=self.stages.as_deref_mut().unwrap().phases.get_mut("input_load").unwrap();
            phase.calls+=1;phase.seconds+=seconds;
        }
        result
    }
    fn started(&mut self) {if let Some(s)=self.stages.as_deref_mut(){s.terms_started+=1;}}
    fn finished(&mut self) {if let Some(s)=self.stages.as_deref_mut(){s.terms_finished+=1;}}
}
struct Clock { stop:mpsc::Sender<()>,thread:Option<thread::JoinHandle<()>> }
impl Clock {
    fn start(engine:Engine,seconds:f64)->Self {
        let (stop,rx)=mpsc::channel();
        let handle=thread::spawn(move||if rx.recv_timeout(Duration::from_secs_f64(seconds.max(0.000001))).is_err() {engine.increment_epoch();});
        Self{stop,thread:Some(handle)}
    }
}
impl Drop for Clock {fn drop(&mut self) {let _=self.stop.send(());if let Some(h)=self.thread.take(){let _=h.join();}}}
fn trap(e:wasmtime::Error)->String {
    match e.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel)=>"term_fuel",Some(Trap::Interrupt)=>"deadline",_=>"runtime_failure",
    }.into()
}
fn status(code:i32)->Option<String> {
    match code {0=>None,1=>Some("invalid_handle"),2=>Some("integer_limit"),3=>Some("literal_limit"),
        4=>Some("output_limit"),5=>Some("arena_limit"),6=>Some("control_range"),7=>Some("loop_limit"),_=>Some("runtime_failure")}.map(str::to_owned)
}
/// Only host transfer/status functions; never added to the candidate linker.
/// A cache belongs to one fresh term and resolves each export at first use.
#[derive(Default)]
struct HostCalls {
    cached:bool,
    status:Option<TypedFunc<(),i32>>,seal:Option<TypedFunc<(),i32>>,sign:Option<TypedFunc<(),i32>>,
    begin:Option<TypedFunc<i32,i32>>,byte:Option<TypedFunc<i32,i32>>,finish:Option<TypedFunc<i32,i32>>,
    serialize:Option<TypedFunc<i32,i32>>,output:Option<TypedFunc<i32,i32>>,
}
impl HostCalls {
    fn new(cached:bool)->Self {Self{cached,..Self::default()}}
    fn call0(&mut self,i:&Instance,s:&mut Store<Limits>,name:&str)->Result<i32,String> {
        // The uncached branch is qualification-only and reproduces the old
        // get_typed_func/call ordering, without changing any guest instruction.
        if !self.cached {return i.get_typed_func::<(),i32>(&mut *s,name).map_err(trap)?.call(s,()).map_err(trap);}
        let slot=match name {"host_status"=>&mut self.status,"host_seal"=>&mut self.seal,
            "host_output_sign"=>&mut self.sign,_=>return Err("runtime_failure".into())};
        if slot.is_none() {*slot=Some(i.get_typed_func::<(),i32>(&mut *s,name).map_err(trap)?);}
        slot.as_ref().unwrap().call(s,()).map_err(trap)
    }
    fn call1(&mut self,i:&Instance,s:&mut Store<Limits>,name:&str,x:i32)->Result<i32,String> {
        if !self.cached {return i.get_typed_func::<i32,i32>(&mut *s,name).map_err(trap)?.call(s,x).map_err(trap);}
        let slot=match name {"host_begin_load"=>&mut self.begin,"host_load_byte"=>&mut self.byte,
            "host_finish_load"=>&mut self.finish,"host_serialize"=>&mut self.serialize,
            "host_output_byte"=>&mut self.output,_=>return Err("runtime_failure".into())};
        if slot.is_none() {*slot=Some(i.get_typed_func::<i32,i32>(&mut *s,name).map_err(trap)?);}
        slot.as_ref().unwrap().call(s,x).map_err(trap)
    }
}
fn check(calls:&mut HostCalls,i:&Instance,s:&mut Store<Limits>)->Result<(),String> {
    if let Some(e)=status(calls.call0(i,s,"host_status")?) {Err(e)} else {Ok(())}
}
fn load(calls:&mut HostCalls,i:&Instance,s:&mut Store<Limits>,n:&IBig,literal:bool,input:&mut InputObserver<'_>)->Result<i32,String> {
    if let Some(d)=input.detail.as_deref_mut() {if literal {d.literal_values_started+=1;}else{d.index_values_started+=1;}}
    input.phase("begin_load",|_|calls.call1(i,s,"host_begin_load",i32::from(literal)))?;
    let bytes=input.phase("magnitude_encode",|_| {
        let mut bytes=n.clone().into_parts().1.to_le_bytes().into_vec();if bytes.is_empty(){bytes.push(0);}bytes
    });
    // One clock around the loop; counters survive a partial/trapping transfer.
    input.phase("byte_transfer",|input|->Result<(),String>{
        for b in bytes {
            if let Some(d)=input.detail.as_deref_mut(){d.byte_calls_attempted+=1;}
            calls.call1(i,s,"host_load_byte",b as i32)?;
            if let Some(d)=input.detail.as_deref_mut(){d.byte_calls_returned+=1;}
        }
        Ok(())
    })?;
    let h=input.phase("finish_load",|_| {
        let sign=if n<&IBig::ZERO {-1} else if n==&IBig::ZERO {0} else {1};
        calls.call1(i,s,"host_finish_load",sign)
    })?;
    input.phase("status_check",|_|check(calls,i,s))?;
    if let Some(d)=input.detail.as_deref_mut(){d.values_finished+=1;}
    Ok(h)
}

pub struct BigintRuntime {engine:Engine,runtime:Module,compilation_seconds:f64,pub runtime_identity:String}
impl BigintRuntime {
    pub fn new(bytes:&[u8])->Result<Self,String> {
        if bytes.len()>16*1024*1024 {return Err("runtime_failure".into());}
        let compilation=Instant::now();
        let mut cfg=Config::new();cfg.consume_fuel(true).epoch_interruption(true).wasm_threads(false).parallel_compilation(false);
        cfg.static_memory_maximum_size(0).static_memory_guard_size(0).dynamic_memory_guard_size(0).dynamic_memory_reserved_for_growth(0);
        let engine=Engine::new(&cfg).map_err(trap)?;
        let runtime=Module::new(&engine,bytes).map_err(trap)?;
        if runtime.imports().next().is_some() {return Err("runtime_failure".into());}
        Ok(Self{engine,runtime,compilation_seconds:compilation.elapsed().as_secs_f64(),runtime_identity:crate::value::digest(bytes)})
    }
    pub fn evaluate(&mut self,source:&str,indices:&[String],r:&Resources)->Evaluation {
        let mut out=Evaluation{outcome:"failure".into(),reason:None,values:Vec::new(),coverage:0,fuel:0,resource_identity:r.identity()};
        let result=self.run(source,indices,r,&mut out,None,true);
        match result {Ok(())=>out.outcome="ok".into(),Err(e)=>{out.reason=Some(e);out.values.clear();}}
        out
    }
    /// Qualification-only observation of the same production run implementation.
    pub fn evaluate_measured(&mut self,source:&str,indices:&[String],r:&Resources)->(Evaluation,Measurements) {
        self.evaluate_observed(source,indices,r,false,true,false)
    }
    /// Opt-in qualification profiler; not reachable through candidate/worker requests.
    pub fn evaluate_profiled(&mut self,source:&str,indices:&[String],r:&Resources)->(Evaluation,Measurements) {
        self.evaluate_observed(source,indices,r,true,true,false)
    }
    /// Opt-in nested transfer observation; no worker request selects it.
    pub fn evaluate_input_profiled(&mut self,source:&str,indices:&[String],r:&Resources)->(Evaluation,Measurements) {
        self.evaluate_observed(source,indices,r,true,true,true)
    }
    /// Qualification-only lookup-per-call control. No worker request selects it.
    pub fn evaluate_lookup_baseline(&mut self,source:&str,indices:&[String],r:&Resources)->(Evaluation,Measurements) {
        self.evaluate_observed(source,indices,r,false,false,false)
    }
    fn evaluate_observed(&mut self,source:&str,indices:&[String],r:&Resources,profile:bool,cached:bool,input_detail:bool)->(Evaluation,Measurements) {
        let mut out=Evaluation{outcome:"failure".into(),reason:None,values:Vec::new(),coverage:0,fuel:0,resource_identity:r.identity()};
        let mut measured=Measurements{stages:if profile {Some(StageMeasurements::default())}else{None},..Measurements::default()};
        if input_detail {measured.stages.as_mut().unwrap().input_load=Some(InputLoadMeasurements::default());}
        let start=Instant::now();
        let result=self.run(source,indices,r,&mut out,Some(&mut measured),cached);
        measured.total_seconds=start.elapsed().as_secs_f64();
        measured.execution_seconds=(measured.total_seconds-measured.prepare_seconds).max(0.0);
        if let Some(stages)=measured.stages.as_mut() {
            let attributed:f64=stages.phases.values().map(|p|p.seconds).sum();
            stages.unattributed_seconds=(measured.execution_seconds-attributed).max(0.0);
            let input_seconds=stages.phases["input_load"].seconds;
            if let Some(input)=stages.input_load.as_mut() {
                let nested:f64=input.phases.values().map(|p|p.seconds).sum();
                input.unattributed_seconds=(input_seconds-nested).max(0.0);
            }
        }
        match result {Ok(())=>out.outcome="ok".into(),Err(e)=>{out.reason=Some(e);out.values.clear();}}
        (out,measured)
    }
    fn run(&self,source:&str,indices:&[String],r:&Resources,out:&mut Evaluation,mut measured:Option<&mut Measurements>,cached:bool)->Result<(),String> {
        r.validate()?;
        let start=Instant::now();
        if indices.is_empty()||indices.len()>100 {return Err("invalid_source".into());}
        let p=language::parse(source,r,false)?;
        let lowered=lowering::lower(&p,r)?;
        let candidate=Module::new(&self.engine,&lowered.wasm).map_err(trap)?;
        if let Some(m)=measured.as_deref_mut() {m.prepare_seconds=start.elapsed().as_secs_f64();}
        let mut observer=Observer{stages:measured.and_then(|m|m.stages.as_mut())};
        // Typed lookup below validates ABI signatures against the real guest.
        let literals=observer.phase("literal_decode",||lowered.literals.iter().map(|s|integer(s,r.literal_bits)).collect::<Result<Vec<_>,_>>())?;
        for index in indices {
            let remaining=r.evaluation_seconds-self.compilation_seconds-start.elapsed().as_secs_f64();
            if remaining<=0.0 {return Err("deadline".into());}
            let fuel=r.term_fuel.min(r.aggregate_fuel.saturating_sub(out.fuel));
            if fuel==0 {return Err("aggregate_fuel".into());}
            observer.started();
            let mut store=observer.phase("store_setup",||->Result<Store<Limits>,String>{
            let limit=StoreLimitsBuilder::new().instances(2).memories(1).tables(1).table_elements(1024)
                .memory_size(r.memory_bytes).trap_on_grow_failure(true).build();
            let mut store=Store::new(&self.engine,Limits{inner:limit,memory_denied:false});store.limiter(|x|x);store.set_fuel(fuel).map_err(trap)?;
            store.set_epoch_deadline(1);Ok(store)})?;
            let clock=observer.phase("clock_start",||Clock::start(self.engine.clone(),remaining.min(r.term_seconds)));
            let mut calls=HostCalls::new(cached);
            let term=(|| {
                let runtime=observer.phase("guest_init",||->Result<Instance,String>{
                let runtime=Instance::new(&mut store,&self.runtime,&[]).map_err(trap)?;
                let init=runtime.get_typed_func::<(i32,i32,i32,i32,i32,i32,i32),i32>(&mut store,"host_init").map_err(trap)?;
                let ok=init.call(&mut store,(r.input_bits as i32,r.literal_bits as i32,r.intermediate_bits as i32,
                    r.arena_entries as i32,r.arena_bytes as i32,r.output_bytes as i32,r.control_max as i32)).map_err(trap)?;
                if ok!=1 {return Err("runtime_failure".into());}Ok(runtime)})?;
                let h=observer.input_phase(|input|->Result<i32,String>{
                for literal in &literals {load(&mut calls,&runtime,&mut store,literal,true,input)?;}
                let n=input.phase("index_parse",|_|integer(index,r.input_bits).map_err(|_|"integer_limit".to_string()))?;
                let h=load(&mut calls,&runtime,&mut store,&n,false,input)?;
                input.phase("seal",|_|calls.call0(&runtime,&mut store,"host_seal"))?;Ok(h)})?;
                let (_linker,instance)=observer.phase("link_instantiate",||->Result<_,String>{
                let mut linker=Linker::new(&self.engine);
                // Explicit single-function registrations: NEVER linker.instance.
                for name in ABI {
                    let f=runtime.get_func(&mut store,name).ok_or("runtime_failure")?;
                    linker.define(&mut store,"oeis.big.v1",name,f).map_err(trap)?;
                }
                let instance=linker.instantiate(&mut store,&candidate).map_err(trap)?;Ok((linker,instance))})?;
                let answer=observer.phase("compute",||->Result<i32,String>{
                let answer=instance.get_typed_func::<i32,i32>(&mut store,"compute").map_err(trap)?.call(&mut store,h).map_err(trap)?;
                check(&mut calls,&runtime,&mut store)?;
                if answer<0 {return Err(status(-answer).unwrap_or("runtime_failure".into()));}Ok(answer)})?;
                observer.phase("serialize",||->Result<String,String>{
                let len=calls.call1(&runtime,&mut store,"host_serialize",answer)?;
                check(&mut calls,&runtime,&mut store)?;
                if len<1 || len as usize>r.output_bytes {return Err("output_limit".into());}
                let sign=calls.call0(&runtime,&mut store,"host_output_sign")?;
                let mut raw=Vec::with_capacity(len as usize);
                for at in 0..len {raw.push(calls.call1(&runtime,&mut store,"host_output_byte",at)? as u8);}
                check(&mut calls,&runtime,&mut store)?;
                let mut n=IBig::from(UBig::from_le_bytes(&raw));if sign<0{n=-n;}
                if start.elapsed().as_secs_f64()+self.compilation_seconds>=r.evaluation_seconds {return Err("deadline".into());}
                Ok(n.to_string())})
            })();
            let consumed=fuel-store.get_fuel().unwrap_or(0);out.fuel+=consumed;
            let disposition=if store.data().memory_denied {Err("memory_limit".into())} else {match term {
                Ok(n)=>{out.values.push(n);out.coverage+=1;Ok(())},
                Err(e) if e=="term_fuel" && fuel<r.term_fuel=>Err("aggregate_fuel".into()),
                Err(e)=>Err(e),
            }};
            // Same cleanup order on success/failure; observe cancellation/join and
            // store destruction instead of hiding teardown in the residual.
            observer.phase("clock_stop",||drop(clock));
            observer.phase("store_drop",||drop(store));
            observer.finished();
            disposition?;
        }
        Ok(())
    }
}

/// Explicit profile wrapper for the same native execution trait as retained
/// signed-256. A caller must choose resources; changing them changes identity.
pub struct BigintBackend { pub runtime:BigintRuntime, pub resources:Resources }
impl crate::ExecutionBackend for BigintBackend {
    fn profile(&self)->&str {"bigint-body/v1"}
    fn execute(&mut self,source:&str,indices:&[String])->Result<Vec<String>,String> {
        let result=self.runtime.evaluate(source,indices,&self.resources);
        if result.outcome=="ok" {Ok(result.values)} else {Err(result.reason.unwrap_or("runtime_failure".into()))}
    }
}

#[cfg(test)]
#[path="runtime/host_lookup_tests.rs"]
mod host_lookup_tests;
