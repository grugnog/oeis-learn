"""Dynamic Environment State Tracker for Lexical Scopes and Stack Constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from oeis_learn.decoder.wat_grammar import (
    BOS_ID,
    EOS_ID,
    ID_TO_TOKEN,
    IDENTIFIER_TOKENS,
    INSTRUCTION_TOKENS,
    LITERAL_TOKENS,
    OPCODE_SIGNATURES,
    SIG_I32_I32_TO_I32,
    SIG_I32_TO_I64,
    SIG_I64_I64_TO_I64,
    SIG_I64_TO_I32,
    SYNTAX_TOKENS,
    TOKEN_TO_ID,
    VOCAB_SIZE,
)


class StructuralPhase(str, Enum):
    """FSM enforcing mandatory structural sequence ordering across program regions."""

    MODULE_START = "MODULE_START"
    MODULE_HEADER = "MODULE_HEADER"
    FUNC_START = "FUNC_START"
    FUNC_HEADER = "FUNC_HEADER"
    FUNC_NAME_OR_EXPORT = "FUNC_NAME_OR_EXPORT"
    EXPORT_START = "EXPORT_START"
    EXPORT_NAME = "EXPORT_NAME"
    EXPORT_CLOSE = "EXPORT_CLOSE"
    PARAM_START = "PARAM_START"
    PARAM_KW = "PARAM_KW"
    PARAM_NAME = "PARAM_NAME"
    PARAM_TYPE = "PARAM_TYPE"
    PARAM_CLOSE = "PARAM_CLOSE"
    RESULT_START = "RESULT_START"
    RESULT_KW = "RESULT_KW"
    RESULT_TYPE = "RESULT_TYPE"
    RESULT_CLOSE = "RESULT_CLOSE"
    LOCAL_OR_BODY = "LOCAL_OR_BODY"
    LOCAL_START = "LOCAL_START"
    LOCAL_NAME = "LOCAL_NAME"
    LOCAL_TYPE = "LOCAL_TYPE"
    LOCAL_CLOSE = "LOCAL_CLOSE"
    BODY = "BODY"
    MODULE_END = "MODULE_END"


@dataclass
class ControlFrame:
    """Tracks structured control flow block baseline stack height and labels."""

    kind: str  # "block", "loop", "if"
    label: Optional[str] = None
    baseline_stack_depth: int = 0
    paren_depth_at_entry: int = 0


@dataclass
class EnvironmentTracker:
    """Maintains lexical scope, parenthesis nesting, control depth, and operand stack state

    during autoregressive WAT token generation to ensure 100% compilation soundness.
    """

    phase: StructuralPhase = StructuralPhase.MODULE_START
    declared_vars: Set[str] = field(default_factory=lambda: {"$n"})
    var_types: Dict[str, str] = field(default_factory=lambda: {"$n": "i32"})
    control_labels: Set[str] = field(default_factory=set)
    operand_stack: List[str] = field(default_factory=list)
    control_stack: List[ControlFrame] = field(default_factory=list)
    paren_depth: int = 0
    in_module: bool = False
    in_func: bool = False
    last_token: Optional[str] = None
    tokens_history: List[str] = field(default_factory=list)
    pending_local_name: Optional[str] = None
    pending_const_type: Optional[str] = None
    pending_var_op: Optional[str] = None
    pending_branch_op: Optional[str] = None
    min_locals: int = 0

    def reset(self) -> None:
        self.phase = StructuralPhase.MODULE_START
        self.declared_vars = {"$n"}
        self.var_types = {"$n": "i32"}
        self.control_labels = set()
        self.operand_stack.clear()
        self.control_stack.clear()
        self.paren_depth = 0
        self.in_module = False
        self.in_func = False
        self.last_token = None
        self.tokens_history.clear()
        self.pending_local_name = None
        self.pending_const_type = None
        self.pending_var_op = None
        self.pending_branch_op = None

    @property
    def stack_depth(self) -> int:
        return len(self.operand_stack)

    @property
    def control_depth(self) -> int:
        return len(self.control_stack)

    def update(self, token: str) -> None:
        """Update tracker state given newly emitted token."""
        self.tokens_history.append(token)

        # Track parenthesis depth
        if token == "(":
            self.paren_depth += 1
        elif token == ")":
            self.paren_depth = max(0, self.paren_depth - 1)
            if self.paren_depth == 0:
                self.in_module = False
                self.in_func = False
                self.phase = StructuralPhase.MODULE_END
            elif self.paren_depth == 1:
                self.in_func = False
            while self.control_stack and self.control_stack[-1].paren_depth_at_entry > self.paren_depth:
                popped = self.control_stack.pop()
                if popped.label:
                    self.control_labels.discard(popped.label)

        # Phase State Machine Transitions
        if self.phase == StructuralPhase.MODULE_START:
            if token == "(":
                self.phase = StructuralPhase.MODULE_HEADER
        elif self.phase == StructuralPhase.MODULE_HEADER:
            if token == "module":
                self.in_module = True
                self.phase = StructuralPhase.FUNC_START
        elif self.phase == StructuralPhase.FUNC_START:
            if token == "(":
                self.phase = StructuralPhase.FUNC_HEADER
        elif self.phase == StructuralPhase.FUNC_HEADER:
            if token == "func":
                self.in_func = True
                self.phase = StructuralPhase.FUNC_NAME_OR_EXPORT
        elif self.phase == StructuralPhase.FUNC_NAME_OR_EXPORT:
            if token == "(":
                self.phase = StructuralPhase.EXPORT_START
            elif token.startswith("$"):
                pass  # Optional function name
        elif self.phase == StructuralPhase.EXPORT_START:
            if token == "export":
                self.phase = StructuralPhase.EXPORT_NAME
        elif self.phase == StructuralPhase.EXPORT_NAME:
            if token in ('"compute"', '"generate_term"'):
                self.phase = StructuralPhase.EXPORT_CLOSE
        elif self.phase == StructuralPhase.EXPORT_CLOSE:
            if token == ")":
                self.phase = StructuralPhase.PARAM_START
        elif self.phase == StructuralPhase.PARAM_START:
            if token == "(":
                self.phase = StructuralPhase.PARAM_KW
        elif self.phase == StructuralPhase.PARAM_KW:
            if token == "param":
                self.phase = StructuralPhase.PARAM_NAME
        elif self.phase == StructuralPhase.PARAM_NAME:
            if token == "$n" or token.startswith("$"):
                self.declared_vars.add(token)
                self.var_types[token] = "i32"
                self.phase = StructuralPhase.PARAM_TYPE
        elif self.phase == StructuralPhase.PARAM_TYPE:
            if token in ("i32", "i64"):
                self.var_types["$n"] = token
                self.phase = StructuralPhase.PARAM_CLOSE
        elif self.phase == StructuralPhase.PARAM_CLOSE:
            if token == ")":
                self.phase = StructuralPhase.RESULT_START
        elif self.phase == StructuralPhase.RESULT_START:
            if token == "(":
                self.phase = StructuralPhase.RESULT_KW
        elif self.phase == StructuralPhase.RESULT_KW:
            if token == "result":
                self.phase = StructuralPhase.RESULT_TYPE
        elif self.phase == StructuralPhase.RESULT_TYPE:
            if token in ("i32", "i64"):
                self.phase = StructuralPhase.RESULT_CLOSE
        elif self.phase == StructuralPhase.RESULT_CLOSE:
            if token == ")":
                self.phase = StructuralPhase.LOCAL_OR_BODY
        elif self.phase == StructuralPhase.LOCAL_OR_BODY:
            if token == "local":
                self.phase = StructuralPhase.LOCAL_NAME
            elif token not in ("(", "$compute"):
                self.phase = StructuralPhase.BODY
        elif self.phase == StructuralPhase.LOCAL_START:
            if token == "local":
                self.phase = StructuralPhase.LOCAL_NAME
        elif self.phase == StructuralPhase.LOCAL_NAME:
            if token.startswith("$"):
                self.pending_local_name = token
                self.phase = StructuralPhase.LOCAL_TYPE
        elif self.phase == StructuralPhase.LOCAL_TYPE:
            if token in ("i32", "i64") and self.pending_local_name:
                self.declared_vars.add(self.pending_local_name)
                self.var_types[self.pending_local_name] = token
                self.pending_local_name = None
                self.phase = StructuralPhase.LOCAL_CLOSE
        elif self.phase == StructuralPhase.LOCAL_CLOSE:
            if token == ")":
                self.phase = StructuralPhase.LOCAL_OR_BODY

        # Operand Stack & Context Instruction State Transitions
        if token in ("i32.const", "i64.const"):
            self.pending_const_type = "i32" if token == "i32.const" else "i64"
        elif token == "i64.const_?":
            self.operand_stack.append("i64")
            self.pending_const_type = None
        elif self.pending_const_type is not None and (token.isdigit() or token.lstrip("-").isdigit()):
            self.operand_stack.append(self.pending_const_type)
            self.pending_const_type = None
        elif token in ("local.get", "local.set", "local.tee"):
            self.pending_var_op = token
        elif self.pending_var_op is not None and token.startswith("$"):
            var_type = self.var_types.get(token, "i64")
            if self.pending_var_op == "local.get":
                self.operand_stack.append(var_type)
            elif self.pending_var_op == "local.set":
                if self.operand_stack:
                    self.operand_stack.pop()
            elif self.pending_var_op == "local.tee":
                pass
            self.pending_var_op = None
        elif token in ("br", "br_if"):
            self.pending_branch_op = token
            if token == "br_if" and self.operand_stack:
                self.operand_stack.pop()
        elif self.pending_branch_op is not None and (token.startswith("$") or token.isdigit()):
            self.pending_branch_op = None
        elif token in ("block", "loop", "if"):
            frame = ControlFrame(
                kind=token,
                baseline_stack_depth=len(self.operand_stack),
                paren_depth_at_entry=self.paren_depth,
            )
            self.control_stack.append(frame)
        elif self.control_stack and self.control_stack[-1].label is None and token.startswith("$"):
            self.control_stack[-1].label = token
            self.control_labels.add(token)
        elif token in OPCODE_SIGNATURES:
            inputs, outputs = OPCODE_SIGNATURES[token]
            for _ in inputs:
                if self.operand_stack:
                    self.operand_stack.pop()
            for out in outputs:
                self.operand_stack.append(out)
        elif token == "drop":
            if self.operand_stack:
                self.operand_stack.pop()

        self.last_token = token

    def get_valid_next_tokens(self) -> Set[int]:
        """Returns the set of token IDs that are legally permitted next."""
        valid_ids: Set[int] = set()

        if not self.tokens_history:
            valid_ids.add(TOKEN_TO_ID["("])
            valid_ids.add(BOS_ID)
            return valid_ids

        if self.paren_depth == 0 and len(self.tokens_history) > 5:
            valid_ids.add(EOS_ID)
            return valid_ids

        if self.paren_depth == 1 and not self.in_func and len(self.tokens_history) > 10:
            valid_ids.add(TOKEN_TO_ID[")"])
            return valid_ids

        last = self.last_token

        if last == "<bos>":
            valid_ids.add(TOKEN_TO_ID["("])
            return valid_ids

        if self.phase == StructuralPhase.MODULE_START:
            valid_ids.add(TOKEN_TO_ID["("])
            return valid_ids

        if self.phase == StructuralPhase.MODULE_HEADER:
            valid_ids.add(TOKEN_TO_ID["module"])
            return valid_ids

        if self.phase == StructuralPhase.FUNC_START:
            valid_ids.add(TOKEN_TO_ID["("])
            return valid_ids

        if self.phase == StructuralPhase.FUNC_HEADER:
            valid_ids.add(TOKEN_TO_ID["func"])
            return valid_ids

        if self.phase == StructuralPhase.FUNC_NAME_OR_EXPORT:
            valid_ids.add(TOKEN_TO_ID["("])
            if "$compute" in TOKEN_TO_ID:
                valid_ids.add(TOKEN_TO_ID["$compute"])
            return valid_ids

        if self.phase == StructuralPhase.EXPORT_START:
            valid_ids.add(TOKEN_TO_ID["export"])
            return valid_ids

        if self.phase == StructuralPhase.EXPORT_NAME:
            valid_ids.add(TOKEN_TO_ID['"compute"'])
            return valid_ids

        if self.phase == StructuralPhase.EXPORT_CLOSE:
            valid_ids.add(TOKEN_TO_ID[")"])
            return valid_ids

        if self.phase == StructuralPhase.PARAM_START:
            valid_ids.add(TOKEN_TO_ID["("])
            return valid_ids

        if self.phase == StructuralPhase.PARAM_KW:
            valid_ids.add(TOKEN_TO_ID["param"])
            return valid_ids

        if self.phase == StructuralPhase.PARAM_NAME:
            valid_ids.add(TOKEN_TO_ID["$n"])
            return valid_ids

        if self.phase == StructuralPhase.PARAM_TYPE:
            valid_ids.add(TOKEN_TO_ID["i32"])
            return valid_ids

        if self.phase == StructuralPhase.PARAM_CLOSE:
            valid_ids.add(TOKEN_TO_ID[")"])
            return valid_ids

        if self.phase == StructuralPhase.RESULT_START:
            valid_ids.add(TOKEN_TO_ID["("])
            return valid_ids

        if self.phase == StructuralPhase.RESULT_KW:
            valid_ids.add(TOKEN_TO_ID["result"])
            return valid_ids

        if self.phase == StructuralPhase.RESULT_TYPE:
            valid_ids.add(TOKEN_TO_ID["i64"])
            return valid_ids

        if self.phase == StructuralPhase.RESULT_CLOSE:
            valid_ids.add(TOKEN_TO_ID[")"])
            return valid_ids

        if self.phase == StructuralPhase.LOCAL_NAME:
            for var in ["$n64", "$a", "$b", "$c", "$d", "$i", "$j", "$k", "$temp", "$val", "$res"]:
                if var not in self.declared_vars and var in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[var])
            if not valid_ids:
                # If all standard locals already declared, allow unused identifier
                for var in IDENTIFIER_TOKENS:
                    if var not in self.declared_vars and var in TOKEN_TO_ID and var not in ("$loop", "$exit", "$l", "$block"):
                        valid_ids.add(TOKEN_TO_ID[var])
            if not valid_ids:
                valid_ids.add(TOKEN_TO_ID["$temp"])
            return valid_ids

        if self.phase == StructuralPhase.LOCAL_TYPE:
            valid_ids.add(TOKEN_TO_ID["i32"])
            valid_ids.add(TOKEN_TO_ID["i64"])
            return valid_ids

        if self.phase == StructuralPhase.LOCAL_CLOSE:
            valid_ids.add(TOKEN_TO_ID[")"])
            return valid_ids

        # Instruction Arguments
        if last == "local.get":
            for var in self.declared_vars:
                if var in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[var])
            if not valid_ids:
                valid_ids.add(TOKEN_TO_ID["$n"])
            return valid_ids

        if last in ("local.set", "local.tee"):
            # STRICT TYPE SOUNDNESS: Only declared variables matching top-of-stack type allowed!
            top_type = self.operand_stack[-1] if self.operand_stack else "i64"
            for var, v_type in self.var_types.items():
                if v_type == top_type and var in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[var])
            if not valid_ids:
                for var in self.declared_vars:
                    if var in TOKEN_TO_ID:
                        valid_ids.add(TOKEN_TO_ID[var])
            return valid_ids

        if last in ("br", "br_if"):
            active_labels = [f.label for f in self.control_stack if f.label is not None]
            for lbl in active_labels:
                if lbl in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[lbl])
            for d in range(min(4, len(self.control_stack))):
                if str(d) in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[str(d)])
            if not valid_ids:
                if "0" in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID["0"])
            return valid_ids

        if last in ("i32.const", "i64.const"):
            for lit in LITERAL_TOKENS:
                valid_ids.add(TOKEN_TO_ID[lit])
            return valid_ids

        allow_body_start = (self.phase != StructuralPhase.LOCAL_OR_BODY) or (
            len(self.declared_vars) >= (1 + self.min_locals)
        )

        if last == "(":
            if self.phase in (StructuralPhase.LOCAL_OR_BODY, StructuralPhase.LOCAL_START):
                valid_ids.add(TOKEN_TO_ID["local"])
                if allow_body_start:
                    valid_ids.add(TOKEN_TO_ID["block"])
                    valid_ids.add(TOKEN_TO_ID["loop"])
            elif self.phase == StructuralPhase.BODY:
                valid_ids.add(TOKEN_TO_ID["block"])
                valid_ids.add(TOKEN_TO_ID["loop"])
            return valid_ids

        if last in ("block", "loop"):
            for lbl in ["$loop", "$exit", "$l", "$block"]:
                if lbl in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[lbl])
            for null_op in ["i64.const", "i64.const_?", "i32.const", "local.get", "nop"]:
                if null_op in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[null_op])
            return valid_ids

        # General Instruction & Body Context
        baseline = self.control_stack[-1].baseline_stack_depth if self.control_stack else 0
        depth = len(self.operand_stack)
        effective_depth = max(0, depth - baseline)
        top1 = self.operand_stack[-1] if effective_depth >= 1 else None
        top2 = self.operand_stack[-2] if effective_depth >= 2 else None

        # 1. Nullary operations (instructions that push constants or load variables)
        if allow_body_start:
            for null_op in ["i64.const", "i64.const_?", "i32.const", "local.get", "nop"]:
                if null_op in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[null_op])

        if not self.control_stack and depth == 1 and top1 == "i64":
            valid_ids.add(TOKEN_TO_ID["return"])

        if self.phase == StructuralPhase.LOCAL_OR_BODY:
            valid_ids.add(TOKEN_TO_ID["("])
        elif self.phase == StructuralPhase.BODY and len(self.control_stack) < 4:
            valid_ids.add(TOKEN_TO_ID["("])

        # 2. Variable assignments & drops
        if effective_depth >= 1:
            matching_vars = [v for v, vt in self.var_types.items() if vt == top1]
            if matching_vars:
                valid_ids.add(TOKEN_TO_ID["local.set"])
                valid_ids.add(TOKEN_TO_ID["local.tee"])
            valid_ids.add(TOKEN_TO_ID["drop"])

        # 3. Binary i64 operations
        if effective_depth >= 2 and top1 == "i64" and top2 == "i64":
            for op in SIG_I64_I64_TO_I64 + ["i64.eq", "i64.ne", "i64.lt_s", "i64.gt_s", "i64.le_s", "i64.ge_s"]:
                if op in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[op])

        # 4. Binary i32 operations
        if effective_depth >= 2 and top1 == "i32" and top2 == "i32":
            for op in SIG_I32_I32_TO_I32 + ["i32.eq", "i32.ne", "i32.lt_s", "i32.gt_s", "i32.le_s", "i32.ge_s"]:
                if op in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[op])

        # 5. Unary conversions & tests
        if effective_depth >= 1 and top1 == "i64":
            for op in ["i64.eqz", "i32.wrap_i64"]:
                if op in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[op])

        if effective_depth >= 1 and top1 == "i32":
            for op in ["i32.eqz", "i64.extend_i32_s", "i64.extend_i32_u"]:
                if op in TOKEN_TO_ID:
                    valid_ids.add(TOKEN_TO_ID[op])
            if len(self.control_stack) > 0:
                valid_ids.add(TOKEN_TO_ID["br_if"])

        if len(self.control_stack) > 0:
            valid_ids.add(TOKEN_TO_ID["br"])

        # Closing Parenthesis Soundness
        if self.paren_depth == 1:
            # Closing module is valid only if func has finished (i.e. not in_func or already closed)
            if not self.in_func or self.phase in (StructuralPhase.BODY, StructuralPhase.LOCAL_OR_BODY):
                valid_ids.add(TOKEN_TO_ID[")"])
        elif self.paren_depth == 2:
            # Closing func is valid ONLY if operand stack satisfies (result i64), i.e. operand_stack == ["i64"] and no open blocks
            if self.operand_stack == ["i64"] and not self.control_stack:
                valid_ids.add(TOKEN_TO_ID[")"])
        elif self.paren_depth > 2 and self.control_stack:
            # Inside a block/loop: closing ')' is valid ONLY if relative stack is empty (effective_depth == 0)
            if effective_depth == 0:
                valid_ids.add(TOKEN_TO_ID[")"])
        elif self.paren_depth > 2:
            # Closing subexpression / block / local declaration is valid
            valid_ids.add(TOKEN_TO_ID[")"])

        if not valid_ids:
            if self.paren_depth == 2 and self.operand_stack == ["i64"]:
                valid_ids.add(TOKEN_TO_ID[")"])
            elif self.paren_depth == 1:
                valid_ids.add(TOKEN_TO_ID[")"])
            elif self.paren_depth > 2:
                valid_ids.add(TOKEN_TO_ID[")"])
            elif self.paren_depth == 0:
                valid_ids.add(EOS_ID)
            else:
                # Force an i64 constant to satisfy result type if stuck
                valid_ids.add(TOKEN_TO_ID["i64.const"])

        return valid_ids


@dataclass
class RecurrenceFrameTracker:
    """Tracks prior-state reads, required next-state writes, temporary initialization,

    and loop counter progress to ensure sound state rotation before backedges.
    """

    state_locals: List[str]
    next_locals: List[str]
    progress_local: str
    required_commits: Set[str] = field(default_factory=set)
    completed_commits: Set[str] = field(default_factory=set)
    progress_advanced: bool = False
    phase: str = "GUARD"  # GUARD, COMPUTE_NEXT, COMMIT_ALL, ADVANCE, BACKEDGE_READY

    def __post_init__(self) -> None:
        if not self.required_commits:
            self.required_commits = set(self.state_locals)

    def transition_to_compute_next(self) -> None:
        self.phase = "COMPUTE_NEXT"

    def record_temp_assigned(self, temp_name: str) -> None:
        pass

    def transition_to_commit_all(self) -> None:
        self.phase = "COMMIT_ALL"

    def record_state_commit(self, local_name: str) -> None:
        self.completed_commits.add(local_name)

    def is_commit_complete(self) -> bool:
        return self.required_commits.issubset(self.completed_commits)

    def transition_to_advance(self) -> None:
        if self.is_commit_complete():
            self.phase = "ADVANCE"

    def record_progress_advanced(self) -> None:
        self.progress_advanced = True
        self.phase = "BACKEDGE_READY"

    def is_backedge_ready(self) -> bool:
        return self.phase == "BACKEDGE_READY" and self.progress_advanced

    def can_emit_backedge(self) -> bool:
        return self.is_backedge_ready()

