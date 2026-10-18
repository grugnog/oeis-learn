//! Fresh two-instance Wasmtime stores. Arithmetic/transfer/serialization all fuel.
use std::{sync::mpsc,thread,time::{Duration,Instant}};
use dashu_int::{IBig,UBig};
use serde::Serialize;
use wasmtime::{Config,Engine,Instance,Linker,Module,Store,StoreLimits,StoreLimitsBuilder,ResourceLimiter,Trap};
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
fn call0(i:&Instance,s:&mut Store<Limits>,name:&str)->Result<i32,String> {
    i.get_typed_func::<(),i32>(&mut *s,name).map_err(trap)?.call(s,()).map_err(trap)
}
fn call1(i:&Instance,s:&mut Store<Limits>,name:&str,x:i32)->Result<i32,String> {
    i.get_typed_func::<i32,i32>(&mut *s,name).map_err(trap)?.call(s,x).map_err(trap)
}
fn check(i:&Instance,s:&mut Store<Limits>)->Result<(),String> {
    if let Some(e)=status(call0(i,s,"host_status")?) {Err(e)} else {Ok(())}
}
fn load(i:&Instance,s:&mut Store<Limits>,n:&IBig,literal:bool)->Result<i32,String> {
    call1(i,s,"host_begin_load",i32::from(literal))?;
    let mut bytes=n.clone().into_parts().1.to_le_bytes().into_vec();if bytes.is_empty(){bytes.push(0);}
    for b in bytes {call1(i,s,"host_load_byte",b as i32)?;}
    let sign=if n<&IBig::ZERO {-1} else if n==&IBig::ZERO {0} else {1};
    let h=call1(i,s,"host_finish_load",sign)?;check(i,s)?;Ok(h)
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
        let result=self.run(source,indices,r,&mut out);
        match result {Ok(())=>out.outcome="ok".into(),Err(e)=>{out.reason=Some(e);out.values.clear();}}
        out
    }
    fn run(&self,source:&str,indices:&[String],r:&Resources,out:&mut Evaluation)->Result<(),String> {
        r.validate()?;
        let start=Instant::now();
        if indices.is_empty()||indices.len()>100 {return Err("invalid_source".into());}
        let p=language::parse(source,r,false)?;
        let lowered=lowering::lower(&p,r)?;
        let candidate=Module::new(&self.engine,&lowered.wasm).map_err(trap)?;
        // Typed lookup below validates ABI signatures against the real guest.
        let literals=lowered.literals.iter().map(|s|integer(s,r.literal_bits)).collect::<Result<Vec<_>,_>>()?;
        for index in indices {
            let remaining=r.evaluation_seconds-self.compilation_seconds-start.elapsed().as_secs_f64();
            if remaining<=0.0 {return Err("deadline".into());}
            let fuel=r.term_fuel.min(r.aggregate_fuel.saturating_sub(out.fuel));
            if fuel==0 {return Err("aggregate_fuel".into());}
            let limit=StoreLimitsBuilder::new().instances(2).memories(1).tables(1).table_elements(1024)
                .memory_size(r.memory_bytes).trap_on_grow_failure(true).build();
            let mut store=Store::new(&self.engine,Limits{inner:limit,memory_denied:false});store.limiter(|x|x);store.set_fuel(fuel).map_err(trap)?;
            store.set_epoch_deadline(1);
            let _clock=Clock::start(self.engine.clone(),remaining.min(r.term_seconds));
            let term=(|| {
                let runtime=Instance::new(&mut store,&self.runtime,&[]).map_err(trap)?;
                let init=runtime.get_typed_func::<(i32,i32,i32,i32,i32,i32,i32),i32>(&mut store,"host_init").map_err(trap)?;
                let ok=init.call(&mut store,(r.input_bits as i32,r.literal_bits as i32,r.intermediate_bits as i32,
                    r.arena_entries as i32,r.arena_bytes as i32,r.output_bytes as i32,r.control_max as i32)).map_err(trap)?;
                if ok!=1 {return Err("runtime_failure".into());}
                for literal in &literals {load(&runtime,&mut store,literal,true)?;}
                let n=integer(index,r.input_bits).map_err(|_|"integer_limit".to_string())?;
                let h=load(&runtime,&mut store,&n,false)?;
                call0(&runtime,&mut store,"host_seal")?;
                let mut linker=Linker::new(&self.engine);
                // Explicit single-function registrations: NEVER linker.instance.
                for name in ABI {
                    let f=runtime.get_func(&mut store,name).ok_or("runtime_failure")?;
                    linker.define(&mut store,"oeis.big.v1",name,f).map_err(trap)?;
                }
                let instance=linker.instantiate(&mut store,&candidate).map_err(trap)?;
                let answer=instance.get_typed_func::<i32,i32>(&mut store,"compute").map_err(trap)?.call(&mut store,h).map_err(trap)?;
                check(&runtime,&mut store)?;
                if answer<0 {return Err(status(-answer).unwrap_or("runtime_failure".into()));}
                let len=call1(&runtime,&mut store,"host_serialize",answer)?;
                check(&runtime,&mut store)?;
                if len<1 || len as usize>r.output_bytes {return Err("output_limit".into());}
                let sign=call0(&runtime,&mut store,"host_output_sign")?;
                let mut raw=Vec::with_capacity(len as usize);
                for at in 0..len {raw.push(call1(&runtime,&mut store,"host_output_byte",at)? as u8);}
                check(&runtime,&mut store)?;
                let mut n=IBig::from(UBig::from_le_bytes(&raw));if sign<0{n=-n;}
                if start.elapsed().as_secs_f64()+self.compilation_seconds>=r.evaluation_seconds {return Err("deadline".into());}
                Ok(n.to_string())
            })();
            let consumed=fuel-store.get_fuel().unwrap_or(0);out.fuel+=consumed;
            if store.data().memory_denied {return Err("memory_limit".into());}
            match term {
                Ok(n)=>{out.values.push(n);out.coverage+=1;},
                Err(e) if e=="term_fuel" && fuel<r.term_fuel=>return Err("aggregate_fuel".into()),
                Err(e)=>return Err(e),
            }
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
