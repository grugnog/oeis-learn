"""Bounded arithmetic data parser: no eval, sympify or callable names."""
import ast
from fractions import Fraction
import hashlib


def expression_hash(source):
    return 'sha256:'+hashlib.sha256(source.encode()).hexdigest()


def parse_formula(source):
    if not isinstance(source,str) or len(source.encode())>8192: raise ValueError('formula size')
    clean=source.strip()
    if clean.startswith('a(n) ='): clean=clean[6:].strip()
    tree=ast.parse(clean.replace('^','**'),mode='eval').body
    if sum(1 for _ in ast.walk(tree))>512: raise ValueError('formula nodes')
    def visit(node,depth=0):
        if depth>32: raise ValueError('formula depth')
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value.bit_length()<=256: return
        if isinstance(node,ast.Name) and node.id=='n': return
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)): visit(node.operand,depth+1); return
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            visit(node.left,depth+1); visit(node.right,depth+1)
            if isinstance(node.op,ast.Pow) and not (isinstance(node.right,ast.Constant) and type(node.right.value) is int and 0<=node.right.value<=16): raise ValueError('bounded literal exponent required')
            return
        raise ValueError('unsupported formula syntax')
    visit(tree)
    if max(degree_bound(tree))>128: raise ValueError('formula degree')
    return tree


def degree_bound(node):
    if isinstance(node,ast.Constant): return 0,0
    if isinstance(node,ast.Name): return 1,0
    if isinstance(node,ast.UnaryOp): return degree_bound(node.operand)
    a,b=degree_bound(node.left);c,d=degree_bound(node.right)
    if isinstance(node.op,(ast.Add,ast.Sub)): return max(a+d,c+b),b+d
    if isinstance(node.op,ast.Mult): return a+c,b+d
    if isinstance(node.op,ast.Div): return a+d,b+c
    return a*node.right.value,b*node.right.value


def exact_value(node,n):
    if isinstance(node,ast.Constant): value=Fraction(node.value)
    elif isinstance(node,ast.Name): value=Fraction(n)
    elif isinstance(node,ast.UnaryOp): value=exact_value(node.operand,n)*(-1 if isinstance(node.op,ast.USub) else 1)
    else:
        a,b=exact_value(node.left,n),exact_value(node.right,n)
        if isinstance(node.op,ast.Add): value=a+b
        elif isinstance(node.op,ast.Sub): value=a-b
        elif isinstance(node.op,ast.Mult): value=a*b
        elif isinstance(node.op,ast.Div): value=a/b
        else: value=a**int(b)
    if max(value.numerator.bit_length(),value.denominator.bit_length())>32768: raise ValueError('evaluation bit limit')
    return value


def sympy_expression(node,n,poles):
    import sympy as sp
    if isinstance(node,ast.Constant): return sp.Integer(node.value)
    if isinstance(node,ast.Name): return n
    if isinstance(node,ast.UnaryOp): return sympy_expression(node.operand,n,poles)*(-1 if isinstance(node.op,ast.USub) else 1)
    a,b=sympy_expression(node.left,n,poles),sympy_expression(node.right,n,poles)
    if isinstance(node.op,ast.Add): return a+b
    if isinstance(node.op,ast.Sub): return a-b
    if isinstance(node.op,ast.Mult): return a*b
    if isinstance(node.op,ast.Div): poles.append(b); return a/b
    return a**b
