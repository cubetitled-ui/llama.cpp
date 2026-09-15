"""
rlang/lexer.py
Custom Lexer and Tokenizer for the Recurrent Language (RLang).
Zero external dependencies. Tokenizes keywords, mathematical expressions,
multi-line blocks, schedules, and registers.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Any

class TokenType(Enum):
    # Keywords
    MODEL = auto()
    ARCHITECTURE = auto()
    TOTAL_LAYERS = auto()
    STAGE = auto()
    FEEDFORWARD = auto()
    LOOP = auto()
    LAYERS = auto()
    ITERATIONS = auto()
    PARAMS = auto()
    SCHEDULES = auto()
    REGISTERS = auto()
    ANCHORS = auto()
    POLICY = auto()
    KV_CACHE = auto()
    FORMULAS = auto()
    BREAK_IF = auto()
    EPSILON = auto()
    SYSTEM = auto()
    MERGE = auto()
    WEIGHTS = auto()
    BRANCH = auto()
    PIPELINE = auto()
    RUN = auto()
    REPEAT = auto()
    
    # Literals and Identifiers
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    
    # Operators and Delimiters
    DOT_DOT = auto()      # ..
    ASSIGN = auto()       # =
    PLUS = auto()         # +
    MINUS = auto()        # -
    STAR = auto()         # *
    SLASH = auto()        # /
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    COLON = auto()        # :
    SEMICOLON = auto()    # ;
    COMMA = auto()        # ,
    LT = auto()           # <
    GT = auto()           # >
    LE = auto()           # <=
    GE = auto()           # >=
    EQ = auto()           # ==
    NE = auto()           # !=
    
    EOF = auto()

KEYWORDS = {
    "model": TokenType.MODEL,
    "architecture": TokenType.ARCHITECTURE,
    "total_layers": TokenType.TOTAL_LAYERS,
    "stage": TokenType.STAGE,
    "feedforward": TokenType.FEEDFORWARD,
    "loop": TokenType.LOOP,
    "layers": TokenType.LAYERS,
    "iterations": TokenType.ITERATIONS,
    "params": TokenType.PARAMS,
    "schedules": TokenType.SCHEDULES,
    "registers": TokenType.REGISTERS,
    "anchors": TokenType.ANCHORS,
    "policy": TokenType.POLICY,
    "kv_cache": TokenType.KV_CACHE,
    "formulas": TokenType.FORMULAS,
    "break_if": TokenType.BREAK_IF,
    "epsilon": TokenType.EPSILON,
    "system": TokenType.SYSTEM,
    "merge": TokenType.MERGE,
    "weights": TokenType.WEIGHTS,
    "branch": TokenType.BRANCH,
    "pipeline": TokenType.PIPELINE,
    "run": TokenType.RUN,
    "repeat": TokenType.REPEAT,
}

@dataclass
class Token:
    type: TokenType
    value: Any = None
    line: int = 1
    col: int = 1

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, line={self.line})"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.length = len(source)
        self.line = 1
        self.col = 1

    def peek(self, offset: int = 0) -> str:
        p = self.pos + offset
        if p >= self.length:
            return ""
        return self.source[p]

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_whitespace_and_comments(self):
        while self.pos < self.length:
            ch = self.peek()
            if ch in " \t\r\n":
                self.advance()
            elif ch == "#":
                while self.pos < self.length and self.peek() != "\n":
                    self.advance()
            elif ch == "/" and self.peek(1) == "/":
                while self.pos < self.length and self.peek() != "\n":
                    self.advance()
            elif ch == "/" and self.peek(1) == "*":
                self.advance()
                self.advance()
                while self.pos < self.length and not (self.peek() == "*" and self.peek(1) == "/"):
                    self.advance()
                if self.pos < self.length:
                    self.advance()
                    self.advance()
            else:
                break

    def tokenize_number(self) -> Token:
        start_line, start_col = self.line, self.col
        num_str = ""
        has_dot = False
        has_exp = False

        while self.pos < self.length:
            ch = self.peek()
            if ch.isdigit():
                num_str += self.advance()
            elif ch == "." and not has_dot and self.peek(1) != ".":
                has_dot = True
                num_str += self.advance()
            elif ch in "eE" and not has_exp:
                has_exp = True
                num_str += self.advance()
                if self.peek() in "+-":
                    num_str += self.advance()
            else:
                break

        val = float(num_str) if (has_dot or has_exp) else int(num_str)
        return Token(TokenType.NUMBER, val, start_line, start_col)

    def tokenize_identifier_or_keyword(self) -> Token:
        start_line, start_col = self.line, self.col
        ident = ""
        while self.pos < self.length:
            ch = self.peek()
            if ch.isalnum() or ch == "_":
                ident += self.advance()
            else:
                break

        tok_type = KEYWORDS.get(ident.lower(), TokenType.IDENTIFIER)
        return Token(tok_type, ident, start_line, start_col)

    def tokenize_string(self, quote_char: str) -> Token:
        start_line, start_col = self.line, self.col
        self.advance() # skip opening quote
        val = ""
        while self.pos < self.length and self.peek() != quote_char:
            ch = self.advance()
            if ch == "\\" and self.pos < self.length:
                escaped = self.advance()
                if escaped == "n": val += "\n"
                elif escaped == "t": val += "\t"
                elif escaped == "\\": val += "\\"
                elif escaped == quote_char: val += quote_char
                else: val += escaped
            else:
                val += ch

        if self.pos < self.length:
            self.advance() # skip closing quote

        return Token(TokenType.STRING, val, start_line, start_col)

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < self.length:
            self.skip_whitespace_and_comments()
            if self.pos >= self.length:
                break

            ch = self.peek()
            start_line, start_col = self.line, self.col

            if ch.isdigit() or (ch == "." and self.peek(1).isdigit()):
                tokens.append(self.tokenize_number())
            elif ch.isalpha() or ch == "_":
                tokens.append(self.tokenize_identifier_or_keyword())
            elif ch in ('"', "'"):
                tokens.append(self.tokenize_string(ch))
            elif ch == "." and self.peek(1) == ".":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.DOT_DOT, "..", start_line, start_col))
            elif ch == "=" and self.peek(1) == "=":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.EQ, "==", start_line, start_col))
            elif ch == "!" and self.peek(1) == "=":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.NE, "!=", start_line, start_col))
            elif ch == "<" and self.peek(1) == "=":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.LE, "<=", start_line, start_col))
            elif ch == ">" and self.peek(1) == "=":
                self.advance()
                self.advance()
                tokens.append(Token(TokenType.GE, ">=", start_line, start_col))
            elif ch == "=":
                self.advance()
                tokens.append(Token(TokenType.ASSIGN, "=", start_line, start_col))
            elif ch == "+":
                self.advance()
                tokens.append(Token(TokenType.PLUS, "+", start_line, start_col))
            elif ch == "-":
                self.advance()
                tokens.append(Token(TokenType.MINUS, "-", start_line, start_col))
            elif ch == "*":
                self.advance()
                tokens.append(Token(TokenType.STAR, "*", start_line, start_col))
            elif ch == "/":
                self.advance()
                tokens.append(Token(TokenType.SLASH, "/", start_line, start_col))
            elif ch == "(":
                self.advance()
                tokens.append(Token(TokenType.LPAREN, "(", start_line, start_col))
            elif ch == ")":
                self.advance()
                tokens.append(Token(TokenType.RPAREN, ")", start_line, start_col))
            elif ch == "{":
                self.advance()
                tokens.append(Token(TokenType.LBRACE, "{", start_line, start_col))
            elif ch == "}":
                self.advance()
                tokens.append(Token(TokenType.RBRACE, "}", start_line, start_col))
            elif ch == "[":
                self.advance()
                tokens.append(Token(TokenType.LBRACKET, "[", start_line, start_col))
            elif ch == "]":
                self.advance()
                tokens.append(Token(TokenType.RBRACKET, "]", start_line, start_col))
            elif ch == ":":
                self.advance()
                tokens.append(Token(TokenType.COLON, ":", start_line, start_col))
            elif ch == ";":
                self.advance()
                tokens.append(Token(TokenType.SEMICOLON, ";", start_line, start_col))
            elif ch == ",":
                self.advance()
                tokens.append(Token(TokenType.COMMA, ",", start_line, start_col))
            elif ch == "<":
                self.advance()
                tokens.append(Token(TokenType.LT, "<", start_line, start_col))
            elif ch == ">":
                self.advance()
                tokens.append(Token(TokenType.GT, ">", start_line, start_col))
            else:
                raise SyntaxError(f"Unexpected character '{ch}' at line {start_line}, col {start_col}")

        tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return tokens
