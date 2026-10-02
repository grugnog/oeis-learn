"""Contained conditional formula identities; never unqualified OEIS proofs."""
from __future__ import annotations
import datetime
import json
from oeis_learn.discovery.formula_syntax import parse_formula, exact_value, sympy_expression, expression_hash
from oeis_learn.discovery.formula_checker import relation_degree, check_certificate
from oeis_learn.sandbox.contained import run_contained


def _proof(formulas,coefficients,transforms,domains,assumptions):
    import sympy as sp
    n=sp.Symbol('n',integer=True)
    evidence={'outcome':'UNKNOWN','proof_method':'rational-evaluation/v1','verifier_version':'fraction-checker/v1','backend_version':f'sympy-{sp.__version__}','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'assumptions':assumptions,'numeric_semantics':'mathematical_rational','independently_checked':False,'expression_hashes':[expression_hash(f) for f in formulas],'transforms':transforms}
    def finish(status,reason=None):
        evidence['outcome']=status; evidence['diagnostic']=reason
        return status,evidence
    try:
        if not 1<=len(formulas)<=16 or len(coefficients)!=len(formulas) or len(transforms)!=len(formulas) or len(domains)!=len(formulas): raise ValueError('operand arity')
        if any(type(c) is not int or c.bit_length()>256 for c in coefficients): raise ValueError('exact bounded coefficients required')
        trees=[parse_formula(f) for f in formulas]
        lower,upper=0,None; expressions=[]; poles=[]
        for tree,(a,b),domain in zip(trees,transforms,domains):
            if any(type(v) is not int or abs(v)>1000000 for v in (a,b)): raise ValueError('bounded exact index transforms required')
            if set(domain)!={'integer_only','lower_bound','upper_bound'} or domain['integer_only'] is not True: raise ValueError('integer interval required')
            lo,hi=domain['lower_bound'],domain['upper_bound']
            if type(lo) is not int or (hi is not None and type(hi) is not int): raise ValueError('exact domain bounds required')
            if hi is not None and lo>hi: return finish('EMPTY_DOMAIN')
            if a==0:
                if b<lo or (hi is not None and b>hi): return finish('EMPTY_DOMAIN')
            elif a>0:
                lower=max(lower,-((b-lo)//a))
                if hi is not None: upper=(hi-b)//a if upper is None else min(upper,(hi-b)//a)
            else:
                bound=(lo-b)//a
                upper=bound if upper is None else min(upper,bound)
                if hi is not None: lower=max(lower,-((b-hi)//a))
            original_poles=[]; expr=sympy_expression(tree,n,original_poles)
            expressions.append(expr.subs(n,a*n+b))
            poles.extend(p.subs(n,a*n+b) for p in original_poles)
        evidence['domain']={'integer_only':True,'lower_bound':lower,'upper_bound':upper,'excluded_poles':[str(p) for p in poles]}
        if (upper is not None and lower>upper) or any(sp.cancel(p)==0 for p in poles): return finish('EMPTY_DOMAIN')
        residual=sp.cancel(sum(c*e for c,e in zip(coefficients,expressions)))
        evidence['reduced_expression']=str(residual)
        defined=0
        for index in range(lower,min(lower+1024,upper+1 if upper is not None else lower+1024)):
            try: values=[exact_value(tree,a*index+b) for tree,(a,b) in zip(trees,transforms)]
            except ZeroDivisionError: continue
            defined+=1; value=sum(c*v for c,v in zip(coefficients,values))
            if value:
                evidence['witness']={'index':index,'values':[str(v) for v in values],'residual':str(value)}
                evidence['independently_checked']=True
                return finish('COUNTEREXAMPLE')
        if not defined: return finish('EMPTY_DOMAIN' if upper is not None and upper-lower<1024 else 'UNKNOWN','no defined witness point found')
        if residual!=0: return finish('UNKNOWN','no concrete counterexample found in bounded search')
        degree=relation_degree(trees)
        if degree>512: return finish('UNSUPPORTED','certificate degree cap')
        points=[]
        for index in range(2048):
            try:
                for tree,(a,b) in zip(trees,transforms): exact_value(tree,a*index+b)
            except ZeroDivisionError: continue
            points.append(index)
            if len(points)==degree+1: break
        certificate={'version':'rational-evaluation/v1','formulas':list(formulas),'expression_hashes':evidence['expression_hashes'],'coefficients':list(coefficients),'transforms':[list(t) for t in transforms],'degree_bound':degree,'points':points}
        if not check_certificate(certificate): return finish('UNKNOWN','independent certificate rejected')
        evidence['certificate']=certificate; evidence['independently_checked']=True
        return finish('FORMULA_IDENTITY')
    except (ValueError,TypeError,SyntaxError,OverflowError) as exc: return finish('UNSUPPORTED',str(exc))


class SymbolicProver:
    def __init__(self,timeout_ms=2000): self.timeout_ms=timeout_ms

    def _run(self,formulas,coefficients,transforms=None,domains=None,assumptions=None):
        count=len(formulas)
        status,value=run_contained(_proof,(list(formulas),list(coefficients),transforms if transforms is not None else [(1,0)]*count,domains if domains is not None else [{'integer_only':True,'lower_bound':0,'upper_bound':None} for _ in formulas],assumptions if assumptions is not None else ['Formula-to-OEIS correspondence is an external assumption.']),timeout_ms=self.timeout_ms)
        if status=='ok': return value
        outcome='TIMEOUT' if status=='timeout' else 'UNKNOWN'
        return outcome,{'outcome':outcome,'diagnostic':str(value),'independently_checked':False}

    def prove_relation(self,candidate,formulas=None):
        if not candidate.pslq_vector or len(candidate.pslq_vector)!=len(candidate.sequences):
            candidate.status='REJECTED'; return candidate
        if formulas is None or len(formulas)!=len(candidate.sequences):
            candidate.status='NUMERICALLY_VERIFIED_CONJECTURE'; candidate.symbolic_proof=None; return candidate
        status,evidence=self._run(formulas,candidate.pslq_vector)
        candidate.status=status
        candidate.symbolic_proof=json.dumps(evidence,sort_keys=True)
        return candidate

    def prove_canonical_relation(self,relation,registry):
        entries=[registry.get_definition(op.oeis_id) for op in relation.operands]
        if any(e is None for e in entries): return 'MISSING_DEFINITION',{'outcome':'MISSING_DEFINITION','independently_checked':False}
        from oeis_learn.data.symbolic_definitions import validate_definition
        try:
            for entry in entries: validate_definition(entry)
        except ValueError as exc: return 'UNSUPPORTED',{'outcome':'UNSUPPORTED','diagnostic':str(exc),'independently_checked':False}
        import re
        if any(not isinstance(c, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", c) or len(c) > 80 for c in relation.coefficients):
            return 'UNSUPPORTED', {'outcome':'UNSUPPORTED','diagnostic':'canonical exact coefficients required','independently_checked':False}
        status,evidence=self._run([e['expression'] for e in entries],[int(c) for c in relation.coefficients],[(op.index_scale,op.index_shift) for op in relation.operands],[e['domain'] for e in entries],[a for e in entries for a in e['assumptions']]+['Formula-to-OEIS correspondence is an external assumption.'])
        evidence.update(definition_ids=[e['definition_id'] for e in entries],definition_hashes=[e['expression_sha256'] for e in entries],normalized_identity=relation.canonical_expression)
        return status,evidence
