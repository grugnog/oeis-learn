"""Independent scalar executor using Python integer arithmetic, not the SMT expressions."""
from oeis_learn.decoder.grounding import instructions, Unsupported


def scalar_value(source, index):
    words, locals_, legacy = instructions(source)
    if not legacy:
        raise Unsupported("scalar reference requires legacy function")
    env = {key: index if key == "$n" else 0 for key in locals_}
    stack, i = [], 0
    def signed(value, width):
        value %= 1 << width
        return value - (1 << width) if value >= 1 << (width - 1) else value
    while i < len(words):
        op = words[i]; i += 1
        if op in ("i32.const", "i64.const"):
            width = int(op[1:3]); stack.append((width, signed(int(words[i]), width))); i += 1
        elif op.startswith("local."):
            key = words[i]; i += 1
            if op == "local.get":
                stack.append((locals_[key], env[key]))
            else:
                width, value = stack.pop()
                if width != locals_[key]: raise Unsupported("type mismatch")
                env[key] = value
                if op == "local.tee": stack.append((width,value))
        elif op.startswith("i64.extend_i32_"):
            width, value = stack.pop()
            if width != 32: raise Unsupported("type mismatch")
            stack.append((64, value if op.endswith("_s") else value % (1 << 32)))
        elif op == "drop": stack.pop()
        elif op == "nop": pass
        elif op in ("i32.add","i32.sub","i32.mul","i64.add","i64.sub","i64.mul"):
            width, right = stack.pop(); lw, left = stack.pop()
            if width != lw or width != int(op[1:3]): raise Unsupported("type mismatch")
            value = left + right if op.endswith("add") else left - right if op.endswith("sub") else left * right
            stack.append((width,signed(value,width)))
        else: raise Unsupported(op)
    if len(stack) != 1 or stack[0][0] != 64: raise Unsupported("final stack")
    return stack[0][1]
