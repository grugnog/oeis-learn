//! Deterministic fraction-free elimination, exact rational back substitution.
use dashu_int::IBig;
use crate::value::bits;
use std::time::Instant;
#[derive(Debug, PartialEq, Eq)]
pub enum Solution { Unique(Vec<IBig>), RankDeficient, NonIntegral, Inconsistent }
fn gcd(mut a: IBig, mut b: IBig) -> IBig { if a < IBig::ZERO { a = -a; } if b < IBig::ZERO { b = -b; } while b != IBig::ZERO { let c = a % &b; a = b; b = c; } a }
fn check(n: &IBig, deadline: Instant) -> Result<(), String> { if Instant::now() >= deadline { return Err("deadline".into()); } if bits(n)>131072 { return Err("symbolic_integer_limit".into()); } Ok(()) }
pub fn solve(rows: &[Vec<IBig>], parameters: usize, deadline: Instant) -> Result<Solution, String> {
    if parameters>8 || rows.len()>20 || rows.iter().any(|r|r.len()!=parameters+1) { return Err("invalid_system".into()); }
    let mut a=rows.to_vec(); let mut pivots=Vec::new(); let mut rank=0;
    for col in 0..parameters {
        let Some(pivot)=(rank..a.len()).find(|&i|a[i][col]!=IBig::ZERO) else {continue;};
        a.swap(rank,pivot);
        for i in rank+1..a.len() {
            if a[i][col]==IBig::ZERO {continue;}
            let left=a[rank][col].clone(); let right=a[i][col].clone();
            for j in col..=parameters { a[i][j]=&left*&a[i][j]-&right*&a[rank][j]; check(&a[i][j],deadline)?; }
            // Exact row gcd normalization bounds expression swell; no rational truncation.
            let divisor=a[i][col..].iter().cloned().fold(IBig::ZERO,gcd);
            if divisor!=IBig::ZERO { for j in col..=parameters { a[i][j]/=&divisor; } }
        }
        pivots.push(col); rank+=1;
    }
    if a.iter().any(|r|r[..parameters].iter().all(|x|*x==IBig::ZERO) && r[parameters]!=IBig::ZERO) {return Ok(Solution::Inconsistent);}
    if rank<parameters {return Ok(Solution::RankDeficient);}
    let mut x=vec![(IBig::ZERO,IBig::ONE);parameters];
    for i in (0..rank).rev() {
        let col=pivots[i]; let mut num=a[i][parameters].clone(); let mut den=IBig::ONE;
        for j in col+1..parameters { num=&num*&x[j].1-&a[i][j]*&x[j].0*&den; den*=&x[j].1; check(&num,deadline)?; check(&den,deadline)?; let d=gcd(num.clone(),den.clone()); num/=&d; den/=&d; }
        den*=&a[i][col]; if den<IBig::ZERO {num=-num;den=-den;}
        let d=gcd(num.clone(),den.clone()); x[col]=(num/&d,den/&d);
    }
    if x.iter().any(|(_,d)|*d!=IBig::ONE) {return Ok(Solution::NonIntegral);}
    let values:Vec<IBig>=x.into_iter().map(|(n,_)|n).collect();
    if rows.iter().any(|r|r[..parameters].iter().zip(&values).fold(IBig::ZERO,|sum,(a,x)|sum+a*x)!=r[parameters]) {return Err("elimination_disagreement".into());}
    Ok(Solution::Unique(values))
}
