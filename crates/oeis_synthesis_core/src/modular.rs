//! Necessary modular constraints only: never constructs or admits a solution.
use crate::symbolic::{Node,Problem};
use dashu_int::IBig;
use std::time::Instant;

pub const PRIMES:[u32;5]=[2,3,5,7,11];
const MAX_ASSIGNMENTS:usize=4096;
const MAX_NODE_STEPS:usize=2_000_000;

// Replace each distinct affine parameter form at a nonlinear frontier by an
// independent residue variable. This enlarges the assignment space: only an
// exhaustive contradiction in that larger space proves the original UNSAT.
// Constants are separated so the same form across different indices shares a
// variable. All work is bounded and replay reconstructs the projection.
fn project(p:&Problem,prime:u32,deadline:Instant,left:&mut usize)->Result<Option<Problem>,String>{
    let cost=p.nodes.len().saturating_mul(p.parameters+4);
    if cost>*left{return Ok(None);}*left-=cost;
    let modulus=IBig::from(prime);
    let residue=|n:&IBig|->u32{let mut v=n%&modulus;if v<IBig::ZERO{v+=&modulus;}v.to_string().parse().expect("small residue")};
    let mut forms:Vec<Option<Vec<u32>>>=Vec::new();
    for (i,node) in p.nodes.iter().enumerate(){
        if Instant::now()>=deadline{return Err("deadline".into());}
        let form=match node {
            Node::Constant(n)=>{let mut v=vec![0;p.parameters+1];v[0]=residue(n);Some(v)},
            Node::Parameter(k) if (*k as usize)<p.parameters=>{let mut v=vec![0;p.parameters+1];v[*k as usize+1]=1;Some(v)},
            Node::Add(a,b)|Node::Sub(a,b)|Node::Mul(a,b) if *a<i&&*b<i=>match (&forms[*a],&forms[*b]){
                (Some(a),Some(b))=>match node {
                    Node::Add(_,_)=>Some(a.iter().zip(b).map(|(x,y)|(x+y)%prime).collect()),
                    Node::Sub(_,_)=>Some(a.iter().zip(b).map(|(x,y)|(x+prime-y)%prime).collect()),
                    _ if a[1..].iter().all(|x|*x==0)=>Some(b.iter().map(|x|x*a[0]%prime).collect()),
                    _ if b[1..].iter().all(|x|*x==0)=>Some(a.iter().map(|x|x*b[0]%prime).collect()),
                    _=>None,
                },_=>None,
            },
            _=>return Err("invalid_modular_problem".into()),
        };forms.push(form);
    }
    let mut needed=vec![false;p.nodes.len()];let mut pending=p.outputs.clone();
    while let Some(i)=pending.pop(){
        if Instant::now()>=deadline{return Err("deadline".into());}
        if i>=needed.len(){return Err("invalid_modular_problem".into());}
        if needed[i]{continue;}needed[i]=true;
        if forms[i].is_none(){match &p.nodes[i]{Node::Add(a,b)|Node::Sub(a,b)|Node::Mul(a,b)=>{pending.push(*a);pending.push(*b);},_=>return Err("invalid_modular_problem".into())}}
    }
    let mut nodes=Vec::new();let mut mapped=vec![0;p.nodes.len()];
    let mut variables=std::collections::BTreeMap::<Vec<u32>,usize>::new();
    for (i,node) in p.nodes.iter().enumerate(){
        if Instant::now()>=deadline{return Err("deadline".into());}
        if !needed[i]{continue;}
        let result=if let Some(form)=&forms[i]{
            if form[1..].iter().all(|v|*v==0){nodes.push(Node::Constant(IBig::from(form[0])));nodes.len()-1}
            else{
                let next=variables.len();let variable=*variables.entry(form[1..].to_vec()).or_insert(next);
                if variable>=8{return Ok(None);}
                nodes.push(Node::Parameter(variable as u8));let param=nodes.len()-1;
                if form[0]==0{param}else{nodes.push(Node::Constant(IBig::from(form[0])));let constant=nodes.len()-1;nodes.push(Node::Add(param,constant));nodes.len()-1}
            }
        }else{
            nodes.push(match node {Node::Add(a,b)=>Node::Add(mapped[*a],mapped[*b]),Node::Sub(a,b)=>Node::Sub(mapped[*a],mapped[*b]),Node::Mul(a,b)=>Node::Mul(mapped[*a],mapped[*b]),_=>return Err("invalid_modular_problem".into())});nodes.len()-1
        };mapped[i]=result;
        if nodes.len()>100_000{return Ok(None);}
    }
    Ok(Some(Problem{nodes,outputs:p.outputs.iter().map(|i|mapped[*i]).collect(),expected:p.expected.clone(),affine_rows:None,parameters:variables.len(),checked_signed256:p.checked_signed256}))
}

// Some(true): every residue assignment was refuted. Some(false): a modular
// witness exists (not an integer solution). None: bounded work was inconclusive.
fn check(p:&Problem,prime:u32,deadline:Instant,left:&mut usize)->Result<Option<bool>,String> {
    if Instant::now()>=deadline{return Err("deadline".into());}
    if !PRIMES.contains(&prime)||p.parameters>8||p.nodes.len()>100_000||p.outputs.len()!=p.expected.len()||p.outputs.is_empty(){return Err("invalid_modular_problem".into());}
    let projected=if (prime as usize).checked_pow(p.parameters as u32).is_some_and(|n|n<=MAX_ASSIGNMENTS){None}
        else{project(p,prime,deadline,left)?};
    let p=projected.as_ref().unwrap_or(p);
    let Some(count)=(prime as usize).checked_pow(p.parameters as u32).filter(|n|*n<=MAX_ASSIGNMENTS) else{return Ok(None);};
    let modulus=IBig::from(prime);
    let residue=|n:&IBig|->u32{let mut v=n%&modulus;if v<IBig::ZERO{v+=&modulus;}v.to_string().parse().expect("small canonical residue")};
    let expected:Vec<u32>=p.expected.iter().map(|n|residue(n)).collect();
    if p.nodes.len()>*left{return Ok(None);}*left-=p.nodes.len();
    let mut constants=Vec::with_capacity(p.nodes.len());
    for (i,node) in p.nodes.iter().enumerate(){
        if i%256==0&&Instant::now()>=deadline{return Err("deadline".into());}
        constants.push(if let Node::Constant(v)=node{residue(v)}else{0});
    }
    let mut values=vec![0u32;p.nodes.len()];
    for assignment in 0..count {
        if Instant::now()>=deadline{return Err("deadline".into());}
        if p.nodes.len()>*left{return Ok(None);}*left-=p.nodes.len();
        let mut digits=[0u32;8];let mut encoded=assignment;
        for d in digits.iter_mut().take(p.parameters){*d=(encoded%prime as usize) as u32;encoded/=prime as usize;}
        for (i,node) in p.nodes.iter().enumerate(){
            if i%256==0&&Instant::now()>=deadline{return Err("deadline".into());}
            values[i]=match node {
                Node::Constant(_)=>constants[i],
                Node::Parameter(k) if (*k as usize)<p.parameters=>digits[*k as usize],
                Node::Add(a,b)|Node::Sub(a,b)|Node::Mul(a,b) if *a<i&&*b<i=>match node {
                    Node::Add(_,_)=> (values[*a]+values[*b])%prime,
                    Node::Sub(_,_)=> (values[*a]+prime-values[*b])%prime,
                    _=> (values[*a]*values[*b])%prime,
                },
                _=>return Err("invalid_modular_problem".into()),
            };
        }
        if p.outputs.iter().any(|i|*i>=values.len()){return Err("invalid_modular_problem".into());}
        if p.outputs.iter().zip(&expected).all(|(i,e)|values[*i]==*e){return Ok(Some(false));}
    }
    if Instant::now()>=deadline{return Err("deadline".into());}Ok(Some(true))
}

pub fn contradiction(p:&Problem,deadline:Instant)->Result<Option<u32>,String>{
    let mut left=MAX_NODE_STEPS;
    for prime in PRIMES {if check(p,prime,deadline,&mut left)?==Some(true){return Ok(Some(prime));}}
    Ok(None)
}

pub fn verify(p:&Problem,prime:u32,deadline:Instant)->Result<bool,String>{
    let mut left=MAX_NODE_STEPS;Ok(check(p,prime,deadline,&mut left)?==Some(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{language,value::Resources};
    use std::time::Duration;
    fn problem(source:&str,values:&[&str])->Problem {
        let r=Resources::default();let program=language::parse(source,&r,true).unwrap();
        let indices:Vec<String>=(0..values.len()).map(|n|n.to_string()).collect();
        Problem::build(&program,&indices,&values.iter().map(|s|s.to_string()).collect::<Vec<_>>(),&r,Instant::now()+Duration::from_secs(2)).unwrap()
    }
    #[test]
    fn square_cannot_equal_negative_one_modulo_three(){
        let p=problem("(program (return (mul (param.get p0) (param.get p0))))",&["-1"]);
        assert_eq!(contradiction(&p,Instant::now()+Duration::from_secs(2)).unwrap(),Some(3));
        assert!(verify(&p,3,Instant::now()+Duration::from_secs(2)).unwrap());
        assert!(!verify(&p,2,Instant::now()+Duration::from_secs(2)).unwrap());
    }
    #[test]
    fn real_integer_solutions_and_modular_witnesses_are_never_admitted(){
        for expected in ["0","1","4","49","121"] {
            let p=problem("(program (return (mul (param.get p0) (param.get p0))))",&[expected]);
            assert_eq!(contradiction(&p,Instant::now()+Duration::from_secs(2)).unwrap(),None);
        }
        // 2311 is 1 modulo every selected prime but is not an integer square.
        // A modular witness is insufficient to decide the integer problem.
        let p=problem("(program (return (mul (param.get p0) (param.get p0))))",&["2311"]);
        assert_eq!(contradiction(&p,Instant::now()+Duration::from_secs(2)).unwrap(),None);
    }
    #[test]
    fn shifted_factorial_recurrence_has_a_sound_small_prime_contradiction(){
        let p=problem("(program (local $b0 bigint) (local $c0 control) (set $b0 (param.get p0)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (add (to_bigint (increment (get $c0))) (param.get p1))))) (return (add (get $b0) (add $index (param.get p2)))))",&["0","0","0"]);
        assert_eq!(contradiction(&p,Instant::now()+Duration::from_secs(2)).unwrap(),Some(2));
    }
    #[test]
    fn work_cap_deadline_and_bad_modulus_cannot_produce_a_proof(){
        let p=problem("(program (return (mul (param.get p0) (param.get p0))))",&["-1"]);
        assert_eq!(check(&p,3,Instant::now()+Duration::from_secs(2),&mut 0).unwrap(),None);
        assert!(contradiction(&p,Instant::now()-Duration::from_millis(1)).is_err());
        assert!(verify(&p,13,Instant::now()+Duration::from_secs(2)).is_err());
    }

    #[test]
    fn affine_frontier_projection_rejects_impossible_factors_but_not_real_solutions(){
        for source in [
            "(program (return (mul (add (add $index (param.get p0)) (param.get p1)) (sub (add (param.get p2) $index) (param.get p3)))))",
            "(program (return (mul (mul (add $index (param.get p0)) (param.get p1)) (sub (add (param.get p2) $index) (param.get p3)))))",
        ]{
            for offset in [-84,-85,-81]{
                let values:Vec<String>=(0..20).map(|n|(n*n+offset).to_string()).collect();
                let refs:Vec<&str>=values.iter().map(String::as_str).collect();let p=problem(source,&refs);
                let until=Instant::now()+Duration::from_secs(2);
                if offset==-81{assert!(!verify(&p,11,until).unwrap());}
                else{assert!(verify(&p,11,until).unwrap());assert_eq!(contradiction(&p,until).unwrap(),Some(11));}
            }
        }
    }
}
