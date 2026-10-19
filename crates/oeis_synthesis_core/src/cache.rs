//! Pure aggregate cache accounting; disk/module ownership stays outside this policy.
use std::collections::BTreeMap;
#[derive(Default)]
pub struct Cache {pub max_entries:usize,pub max_bytes:usize,clock:u64,entries:BTreeMap<String,(usize,u64)>,reserved:BTreeMap<String,usize>}
impl Cache {
    pub fn new(max_entries:usize,max_bytes:usize)->Result<Self,String>{if (max_entries==0)!=(max_bytes==0){return Err("invalid_cache_limits".into());}Ok(Self{max_entries,max_bytes,..Self::default()})}
    pub fn bytes(&self)->usize{self.entries.values().map(|x|x.0).sum::<usize>()+self.reserved.values().sum::<usize>()}
    pub fn reserve(&mut self,key:&str,size:usize)->Result<bool,String>{
        if key.is_empty()||size==0||self.reserved.contains_key(key){return Err("invalid_reservation".into());}
        let reserved:usize=self.reserved.values().sum();
        if self.max_entries==0||size>self.max_bytes||self.reserved.len()>=self.max_entries||size>self.max_bytes-reserved{return Ok(false);}
        self.entries.remove(key);
        while self.bytes()>self.max_bytes-size||self.entries.len()+self.reserved.len()>=self.max_entries{
            let victim=self.entries.iter().min_by_key(|(k,(_,clock))|(*clock,*k)).map(|(k,_)|k.clone()).ok_or("reservation_invariant")?;self.entries.remove(&victim);
        }
        self.reserved.insert(key.into(),size);Ok(true)
    }
    pub fn publish(&mut self,key:&str)->Result<(),String>{let size=self.reserved.remove(key).ok_or("missing_reservation")?;self.clock=self.clock.checked_add(1).ok_or("access_counter_limit")?;self.entries.insert(key.into(),(size,self.clock));Ok(())}
    pub fn abort(&mut self,key:&str){self.reserved.remove(key);}
    pub fn touch(&mut self,key:&str)->bool{if let Some(entry)=self.entries.get_mut(key){self.clock=self.clock.saturating_add(1);entry.1=self.clock;true}else{false}}
}
#[cfg(test)]mod tests{use super::*;#[test]fn reserve_evict_abort_and_oversize(){let mut c=Cache::new(2,10).unwrap();assert!(c.reserve("a",4).unwrap());c.publish("a").unwrap();assert!(c.reserve("b",4).unwrap());c.publish("b").unwrap();c.touch("a");assert!(!c.reserve("huge",11).unwrap());assert!(c.touch("b"));c.touch("a");assert!(c.reserve("c",4).unwrap());assert!(!c.touch("b"));c.abort("c");assert_eq!(c.bytes(),4);assert!(c.reserve("d",6).unwrap());assert!(!c.reserve("e",6).unwrap());}}
