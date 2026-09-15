"""
rlang/parser.py
Recursive Descent Parser for the Recurrent Language (RLang).
Constructs an Abstract Syntax Tree (AST) representing model architecture,
pipeline stages, recurrent loops, dynamic schedules, and custom mathematical formulas.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from recurrent_engine.rlang.lexer import Lexer, Token, TokenType

@dataclass
class Range:
    start: int
    end: int

@dataclass
class ModelDecl:
    name: str
    architecture: str = "qwen2"
    total_layers: int = 28
    epsilon: float = 1e-6
    description: str = ""

@dataclass
class FeedforwardStage:
    name: str
    layers: Range
    type: str = "feedforward"

@dataclass
class LoopStage:
    name: str
    layers: Range
    iterations: int = 2
    params: Dict[str, float] = field(default_factory=dict)
    schedules: Dict[str, str] = field(default_factory=dict)
    registers: Dict[str, str] = field(default_factory=dict)
    anchors: Dict[str, str] = field(default_factory=dict)
    kv_cache: str = "step_overwrite"
    delta_mode: str = "pure_delta"
    break_if: Optional[str] = None
    formulas: List[str] = field(default_factory=list)
    system: str = "custom"
    merge: str = "last"
    merge_weights: List[float] = field(default_factory=list)
    branch: Optional[str] = None
    type: str = "loop"

@dataclass
class RunStmt:
    dest: str
    layers: Range
    input_var: str
    type: str = "run"

@dataclass
class AssignStmt:
    dest: str
    expr: str
    type: str = "assign"

@dataclass
class RepeatStmt:
    count: int
    stmts: List[Any]
    type: str = "repeat"

@dataclass
class PipelineBlock:
    stmts: List[Any] = field(default_factory=list)
    type: str = "pipeline"

@dataclass
class Program:
    model: Optional[ModelDecl] = None
    stages: List[Any] = field(default_factory=list)
    pipeline: Optional[PipelineBlock] = None

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        p = self.pos + offset
        if p >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[p]

    def advance(self) -> Token:
        tok = self.peek()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def match(self, *expected_types: TokenType) -> bool:
        if self.peek().type in expected_types:
            self.advance()
            return True
        return False

    def expect(self, expected_type: TokenType, msg: str = "") -> Token:
        tok = self.peek()
        if tok.type != expected_type:
            raise SyntaxError(f"Parse Error at line {tok.line}, col {tok.col}: expected {expected_type.name}, got {tok.type.name} ({tok.value}). {msg}")
        return self.advance()

    def parse(self) -> Program:
        prog = Program()
        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.MODEL:
                prog.model = self.parse_model()
            elif self.peek().type == TokenType.STAGE:
                prog.stages.append(self.parse_stage())
            elif self.peek().type == TokenType.PIPELINE:
                prog.pipeline = self.parse_pipeline()
            else:
                tok = self.peek()
                raise SyntaxError(f"Unexpected top-level token: {tok.type.name} ({tok.value}) at line {tok.line}")
        return prog

    def parse_pipeline(self) -> PipelineBlock:
        self.expect(TokenType.PIPELINE)
        self.expect(TokenType.LBRACE)
        stmts = []
        while not self.match(TokenType.RBRACE):
            if self.match(TokenType.REPEAT):
                has_paren = self.match(TokenType.LPAREN)
                count_tok = self.expect(TokenType.NUMBER)
                count = int(count_tok.value)
                if has_paren:
                    self.expect(TokenType.RPAREN)
                self.expect(TokenType.LBRACE)
                sub_stmts = []
                while not self.match(TokenType.RBRACE):
                    sub_stmts.append(self.parse_pipeline_stmt())
                stmts.append(RepeatStmt(count=count, stmts=sub_stmts))
            else:
                stmts.append(self.parse_pipeline_stmt())
        return PipelineBlock(stmts=stmts)

    def parse_pipeline_stmt(self) -> Any:
        dest_tok = self.expect(TokenType.IDENTIFIER)
        dest = dest_tok.value
        self.expect(TokenType.ASSIGN)
        if self.peek().type == TokenType.RUN:
            self.advance() # consume 'run'
            self.expect(TokenType.LPAREN)
            layers = self.parse_range()
            self.expect(TokenType.COMMA)
            input_tok = self.expect(TokenType.IDENTIFIER)
            input_var = input_tok.value
            self.expect(TokenType.RPAREN)
            self.match(TokenType.SEMICOLON)
            return RunStmt(dest=dest, layers=layers, input_var=input_var)
        else:
            expr_tokens = []
            while self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
                expr_tokens.append(str(self.advance().value))
            self.match(TokenType.SEMICOLON)
            return AssignStmt(dest=dest, expr=" ".join(expr_tokens).strip())

    def parse_model(self) -> ModelDecl:
        self.expect(TokenType.MODEL)
        name_tok = self.expect(TokenType.IDENTIFIER)
        m = ModelDecl(name=name_tok.value)

        self.expect(TokenType.LBRACE)
        while not self.match(TokenType.RBRACE):
            tok = self.advance()
            self.expect(TokenType.COLON)
            sign = 1.0
            if self.peek().type == TokenType.MINUS:
                self.advance()
                sign = -1.0
            val_tok = self.advance()
            self.match(TokenType.SEMICOLON) # optional semicolon

            attr = tok.value.lower()
            if attr == "architecture":
                m.architecture = str(val_tok.value).lower()
            elif attr == "total_layers":
                m.total_layers = int(sign * float(val_tok.value))
            elif attr in ("epsilon", "norm_eps"):
                m.epsilon = float(sign * float(val_tok.value))
            elif attr == "description":
                m.description = str(val_tok.value)
        return m

    def parse_range(self) -> Range:
        start_tok = self.expect(TokenType.NUMBER)
        if self.match(TokenType.DOT_DOT):
            end_tok = self.expect(TokenType.NUMBER)
            return Range(start=int(start_tok.value), end=int(end_tok.value))
        return Range(start=int(start_tok.value), end=int(start_tok.value))

    def parse_stage(self) -> Any:
        self.expect(TokenType.STAGE)
        name_tok = self.advance()
        stage_name = str(name_tok.value)

        tok_type = self.advance()
        if tok_type.type == TokenType.FEEDFORWARD:
            return self.parse_feedforward(stage_name)
        elif tok_type.type == TokenType.LOOP:
            return self.parse_loop(stage_name)
        else:
            raise SyntaxError(f"Unknown stage type '{tok_type.value}' at line {tok_type.line}")

    def parse_feedforward(self, name: str) -> FeedforwardStage:
        self.expect(TokenType.LBRACE)
        layers = None
        while not self.match(TokenType.RBRACE):
            tok = self.advance()
            if tok.type == TokenType.LAYERS:
                self.expect(TokenType.COLON)
                layers = self.parse_range()
                self.match(TokenType.SEMICOLON)
            else:
                # Skip unknown attributes
                self.advance()
                self.match(TokenType.SEMICOLON)

        if layers is None:
            raise SyntaxError(f"Feedforward stage '{name}' missing 'layers: start..end;' declaration")
        return FeedforwardStage(name=name, layers=layers)

    def parse_loop(self, name: str) -> LoopStage:
        loop = LoopStage(name=name, layers=Range(0, 0))
        self.expect(TokenType.LBRACE)

        while not self.match(TokenType.RBRACE):
            tok = self.advance()
            if tok.type == TokenType.LAYERS:
                self.expect(TokenType.COLON)
                loop.layers = self.parse_range()
                self.match(TokenType.SEMICOLON)

            elif tok.type == TokenType.ITERATIONS:
                self.expect(TokenType.COLON)
                it_tok = self.expect(TokenType.NUMBER)
                loop.iterations = int(it_tok.value)
                self.match(TokenType.SEMICOLON)

            elif tok.type == TokenType.SYSTEM:
                self.expect(TokenType.COLON)
                sys_tok = self.advance()
                loop.system = str(sys_tok.value).lower()
                self.match(TokenType.SEMICOLON)

            elif tok.type == TokenType.PARAMS:
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    p_name = self.expect(TokenType.IDENTIFIER).value
                    self.expect(TokenType.COLON)
                    sign = 1.0
                    if self.peek().type == TokenType.MINUS:
                        self.advance()
                        sign = -1.0
                    p_val = sign * float(self.advance().value)
                    self.match(TokenType.SEMICOLON)
                    loop.params[p_name] = p_val

            elif tok.type == TokenType.SCHEDULES:
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    s_name = self.expect(TokenType.IDENTIFIER).value
                    self.expect(TokenType.COLON)
                    s_val = str(self.advance().value)
                    self.match(TokenType.SEMICOLON)
                    loop.schedules[s_name] = s_val

            elif tok.type == TokenType.REGISTERS:
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    r_name = self.expect(TokenType.IDENTIFIER).value
                    self.expect(TokenType.ASSIGN)
                    r_init = ""
                    while self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE):
                        r_init += str(self.advance().value) + " "
                    self.match(TokenType.SEMICOLON)
                    loop.registers[r_name] = r_init.strip()

            elif tok.type == TokenType.ANCHORS:
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    a_name = self.expect(TokenType.IDENTIFIER).value
                    self.expect(TokenType.COLON)
                    a_val = str(self.advance().value)
                    self.match(TokenType.SEMICOLON)
                    loop.anchors[a_name] = a_val

            elif tok.type == TokenType.POLICY:
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    pol_k = self.advance().value
                    self.expect(TokenType.COLON)
                    pol_v = str(self.advance().value).lower()
                    self.match(TokenType.SEMICOLON)
                    if pol_k == "kv_cache":
                        loop.kv_cache = pol_v
                    elif pol_k == "delta_mode":
                        loop.delta_mode = pol_v

            elif tok.type == TokenType.BREAK_IF:
                self.expect(TokenType.COLON)
                cond = ""
                while self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE):
                    cond += str(self.advance().value) + " "
                self.match(TokenType.SEMICOLON)
                loop.break_if = cond.strip()

            elif tok.type == TokenType.MERGE:
                self.expect(TokenType.COLON)
                loop.merge = str(self.advance().value).lower()
                self.match(TokenType.SEMICOLON)

            elif tok.type == TokenType.WEIGHTS:
                self.expect(TokenType.COLON)
                if self.match(TokenType.LBRACKET):
                    while not self.match(TokenType.RBRACKET):
                        sign = 1.0
                        if self.peek().type == TokenType.MINUS:
                            self.advance()
                            sign = -1.0
                        val = sign * float(self.expect(TokenType.NUMBER).value)
                        loop.merge_weights.append(val)
                        self.match(TokenType.COMMA)
                self.match(TokenType.SEMICOLON)

            elif tok.type == TokenType.BRANCH:
                self.expect(TokenType.COLON)
                loop.branch = str(self.advance().value).lower()
                self.match(TokenType.SEMICOLON)

            elif tok.type == TokenType.FORMULAS:
                self.expect(TokenType.LBRACE)
                while not self.match(TokenType.RBRACE):
                    stmt = ""
                    while self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE):
                        stmt += str(self.advance().value) + " "
                    self.match(TokenType.SEMICOLON)
                    if stmt.strip():
                        loop.formulas.append(stmt.strip())
            else:
                # Unknown block/field: skip until semicolon
                while self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
                    self.advance()
                self.match(TokenType.SEMICOLON)

        return loop
