"""Independent rational-identity certificate checking with exact fractions.

After multiplying by all uncancelled denominators, a degree-D polynomial
vanishing at D+1 distinct defined points is zero. No CAS reduction is trusted.
"""
from oeis_learn.discovery.formula_syntax import parse_formula, degree_bound, exact_value, expression_hash


def relation_degree(trees):
    degrees=[degree_bound(t) for t in trees]
    return max(a+sum(d for j,(_,d) in enumerate(degrees) if j!=i) for i,(a,b) in enumerate(degrees))


def check_certificate(certificate):
    try:
        if not isinstance(certificate,dict) or set(certificate)!={'version','formulas','expression_hashes','coefficients','transforms','degree_bound','points'} or certificate['version']!='rational-evaluation/v1': return False
        formulas=certificate['formulas']; coeffs=certificate['coefficients']; transforms=certificate['transforms']
        if not isinstance(formulas,list) or not 1<=len(formulas)<=16 or len(coeffs)!=len(formulas) or len(transforms)!=len(formulas): return False
        if any(type(c) is not int or c.bit_length()>256 for c in coeffs): return False
        if any(not isinstance(t,(list,tuple)) or len(t)!=2 or any(type(v) is not int or abs(v)>1000000 for v in t) for t in transforms): return False
        if certificate['expression_hashes'] != [expression_hash(f) for f in formulas]: return False
        trees=[parse_formula(f) for f in formulas]; degree=relation_degree(trees)
        if degree>512 or type(certificate['degree_bound']) is not int or certificate['degree_bound']!=degree: return False
        points=certificate['points']
        if not isinstance(points,list) or len(points)!=degree+1 or any(type(n) is not int or abs(n)>1000000 for n in points) or len(set(points))!=len(points): return False
        return all(sum(c*exact_value(t,a*n+b) for c,t,(a,b) in zip(coeffs,trees,transforms))==0 for n in points)
    except (ValueError,TypeError,KeyError,ZeroDivisionError,SyntaxError,OverflowError): return False
