# Copyright 2025 Xdynix
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes portions of logic and structure adapted from the
# Kubernetes project (https://github.com/kubernetes/kubernetes), originally
# licensed under the Apache License, Version 2.0. Significant modifications
# have been made to port and restructure the code for use in Python.


"""Label selector utilities.

This module provides classes and utilities for validating, parsing, and matching labels
against complex selectors. It closely emulates the behavior of Kubernetes (K8s) label
selectors, including support for equality, set-based, and presence operators.

Example:
    >>> from pydantic import BaseModel
    >>> class Label(BaseModel):
    ...     key: LabelKey
    ...     value: LabelValue
    >>> label = Label.model_validate({"key": "example.com/env", "value": "prod"})
    >>> label.key.prefix, label.key.name
    ('example.com', 'env')

    >>> s = "env=prod, tier in (frontend,backend), !debug"
    >>> selector = LabelSelector.model_validate(s)
    >>> selector.matches({"env": "prod", "tier": "frontend"})
    True
    >>> selector.matches({"env": "staging", "tier": "frontend"})
    False
    >>> selector.matches({"env": "prod", "debug": "true", "tier": "backend"})
    False
"""

__all__ = (
    "LabelKey",
    "LabelSelector",
    "LabelValue",
    "Operator",
    "Requirement",
)

import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from enum import Enum, StrEnum, auto
from functools import cached_property, total_ordering
from operator import attrgetter
from typing import Annotated, Any, ClassVar, Literal, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    RootModel,
    TypeAdapter,
    field_serializer,
)
from pydantic_core import CoreSchema, core_schema

# ==== Label Key and Value ====

DNS_1123_LABEL = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"
DNS_1123_SUBDOMAIN = rf"{DNS_1123_LABEL}(\.{DNS_1123_LABEL})*"

QNAME_CHAR = "[A-Za-z0-9]"
QNAME_EXT_CHAR = "[-A-Za-z0-9_.]"
QNAME = f"({QNAME_CHAR}{QNAME_EXT_CHAR}*)?{QNAME_CHAR}"

DNS_1123_SUBDOMAIN_MAX_LEN = 253
QNAME_MAX_LEN = 63

DNS_1123_SUBDOMAIN_PATTERN = re.compile(DNS_1123_SUBDOMAIN)
QNAME_PATTERN = re.compile(QNAME)
QNAME_OPTIONAL_PATTERN = re.compile(f"({QNAME})?")


class LabelKey(str):
    """Validated label key.

    Valid label keys have two segments: an optional prefix and name, separated by a
    slash (`/`). The name segment is required and must be 63 characters or fewer,
    beginning and ending with an alphanumeric character (`[a-z0-9A-Z]`) with dashes
    (`-`), underscores (`_`), dots (`.`), and alphanumerics between. The prefix is
    optional. If specified, the prefix must be a DNS subdomain: a series of DNS labels
    separated by dots (.), not longer than 253 characters in total, followed by a slash
    (`/`).

    This class can also be used for type annotation in Pydantic models.

    Attributes:
        prefix (str): The optional prefix segment of the label key, without the slash.
        name (str): The name segment of the label key.
    """

    # This pattern is used solely for JSON Schema generation and is not involved in the
    # validation process within the Python code. It is designed to closely mirror the
    # actual validation logic, and currently, it replicates it exactly.
    PATTERN = re.compile(
        # Although the escape sequence `\/` is not required in Python,
        # it is retained to maximize the portability of the regular expression.
        "^"
        rf"((?=.{{1,{DNS_1123_SUBDOMAIN_MAX_LEN}}}\/){DNS_1123_SUBDOMAIN}\/)?"
        rf"((?=.{{1,{QNAME_MAX_LEN}}}$){QNAME})"
        "$"
    )
    MIN_LENGTH = 1
    MAX_LENGTH = (
        DNS_1123_SUBDOMAIN_MAX_LEN
        + 1  # `/` between the prefix and name
        + QNAME_MAX_LEN
    )

    __slots__ = ("_name", "_prefix")

    _prefix: str
    _name: str

    def __new__(cls, value: Any) -> Self:
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} requires a string, not {value!r}")

        errors: list[str] = []

        prefix: str
        name: str
        match value.split("/"):
            case [name]:
                prefix = ""
            case [prefix, name]:
                if not prefix:
                    errors.append("prefix part must be non-empty")
                else:
                    if len(prefix) > DNS_1123_SUBDOMAIN_MAX_LEN:
                        errors.append(
                            "prefix part must be no more than "
                            f"{DNS_1123_SUBDOMAIN_MAX_LEN} characters"
                        )
                    if not DNS_1123_SUBDOMAIN_PATTERN.fullmatch(prefix):
                        errors.append(
                            "prefix part a lowercase RFC 1123 subdomain must "
                            "consist of lower case alphanumeric characters, '-' "
                            "or '.', and must start and end with an alphanumeric "
                            "character (e.g. 'example.com', regex used for "
                            f"validation is '{DNS_1123_SUBDOMAIN_PATTERN.pattern}')"
                        )
            case _:
                errors.append(
                    "a qualified name must consist of alphanumeric characters, "
                    "'-', '_' or '.', and must start and end with an alphanumeric "
                    "character (e.g. 'MyName', 'my.name', '123-abc', regex used "
                    f"for validation is '{QNAME_PATTERN.pattern}') with an "
                    "optional DNS subdomain prefix and '/' (e.g. "
                    "'example.com/MyName')"
                )
                raise ValueError(errors)

        if not name:
            errors.append("name part must be non-empty")
        elif len(name) > QNAME_MAX_LEN:
            errors.append(f"name part must be no more than {QNAME_MAX_LEN} characters")
        if not QNAME_PATTERN.fullmatch(name):
            errors.append(
                "name part must consist of alphanumeric characters, '-', '_' or "
                "'.', and must start and end with an alphanumeric character (e.g. "
                "'MyName', 'my.name', '123-abc', regex used for validation is "
                f"'{QNAME_PATTERN.pattern}')"
            )

        if errors:
            raise ValueError("; ".join(errors))

        obj = super().__new__(cls, value)
        obj._prefix = prefix
        obj._name = name
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: "GetCoreSchemaHandler",
    ) -> "CoreSchema":
        return core_schema.no_info_plain_validator_function(
            cls,
            ref=cls.__name__,
            json_schema_input_schema=core_schema.str_schema(
                pattern=cls.PATTERN,
                min_length=cls.MIN_LENGTH,
                max_length=cls.MAX_LENGTH,
            ),
        )

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def name(self) -> str:
        return self._name


class LabelValue(str):
    """Validated label value.

    A valid label value:

    - Must be 63 characters or fewer (can be empty).
    - Unless empty, must begin and end with an alphanumeric character (`[a-z0-9A-Z]`).
    - Could contain dashes (`-`), underscores (`_`), dots (`.`), and alphanumerics
      between.

    This class can also be used for type annotation in Pydantic models.
    """

    # This pattern is used solely for JSON Schema generation and is not involved in the
    # validation process within the Python code. It is designed to closely mirror the
    # actual validation logic, and currently, it replicates it exactly.
    PATTERN = re.compile(
        "^"
        rf"((?=.{{1,{QNAME_MAX_LEN}}}){QNAME})?"
        "$"
    )
    MAX_LENGTH = QNAME_MAX_LEN

    def __new__(cls, value: Any) -> Self:
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} requires a string, not {value!r}.")

        errors: list[str] = []

        if len(value) > QNAME_MAX_LEN:
            errors.append(f"must be no more than {QNAME_MAX_LEN} characters")
        if not QNAME_OPTIONAL_PATTERN.fullmatch(value):
            errors.append(
                "a valid label must be an empty string or consist of alphanumeric "
                "characters, '-', '_' or '.', and must start and end with an "
                "alphanumeric character (e.g. 'MyValue', 'my_value', '12345', regex "
                f"used for validation is '{QNAME_OPTIONAL_PATTERN.pattern}')"
            )

        if errors:
            raise ValueError("; ".join(errors))

        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: "GetCoreSchemaHandler",
    ) -> "CoreSchema":
        return core_schema.no_info_plain_validator_function(
            cls,
            ref=cls.__name__,
            json_schema_input_schema=core_schema.str_schema(
                pattern=cls.PATTERN,
                max_length=cls.MAX_LENGTH,
            ),
        )


# ==== Label Selector ====


class Operator(StrEnum):
    """Enumeration of supported label selector operators."""

    # Equality Operator
    EQUALS = "Equals"
    DOUBLE_EQUALS = "DoubleEquals"
    NOT_EQUALS = "NotEquals"

    # Set Operator
    IN = "In"
    NOT_IN = "NotIn"

    # Presence Operator
    EXISTS = "Exists"
    DOES_NOT_EXIST = "DoesNotExist"


@total_ordering
class BaseRequirement(BaseModel):
    """Abstract representation of a single label selector requirement.

    Attributes:
        key (LabelKey): The label key the requirement applies to.
        operator (Operator): The operator defining how values are matched.
        values (frozenset[LabelValue]): A set of values relevant to the operator.
    """

    key: LabelKey
    operator: Operator
    values: frozenset[LabelValue]

    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        components = []

        if self.operator == Operator.DOES_NOT_EXIST:
            components.append("!")
        components.append(self.key)

        match self.operator:
            case Operator.EQUALS:
                op = "="
            case Operator.DOUBLE_EQUALS:
                op = "=="
            case Operator.NOT_EQUALS:
                op = "!="
            case Operator.IN:
                op = " in "
            case Operator.NOT_IN:
                op = " notin "
            case _:
                op = ""
        components.append(op)

        if self.operator in (Operator.IN, Operator.NOT_IN):
            components.append("(")
        components.append(",".join(self.sorted_values))
        if self.operator in (Operator.IN, Operator.NOT_IN):
            components.append(")")

        return "".join(components)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BaseRequirement):
            return NotImplemented
        to_tuple = attrgetter("key", "operator", "sorted_values")
        return to_tuple(self) < to_tuple(other)

    @cached_property
    def sorted_values(self) -> tuple[LabelValue, ...]:
        return tuple(sorted(self.values))

    def matches(self, labels: Mapping[str, str]) -> bool:
        match self.operator:
            case Operator.EQUALS | Operator.DOUBLE_EQUALS | Operator.IN:
                return labels.get(self.key) in self.values
            case Operator.NOT_EQUALS | Operator.NOT_IN:
                return labels.get(self.key) not in self.values
            case Operator.EXISTS:
                return self.key in labels
            case Operator.DOES_NOT_EXIST:
                return self.key not in labels
            case _:  # pragma: no cover (not reachable)
                return False

    @field_serializer("values")
    def _serialize_values(self, _: Any, __: Any) -> tuple[str, ...]:
        return self.sorted_values


class EqualityRequirement(BaseRequirement):
    """Represents a requirement for the equality of a label value."""

    operator: Literal[Operator.EQUALS, Operator.DOUBLE_EQUALS, Operator.NOT_EQUALS]
    values: Annotated[frozenset[LabelValue], Field(min_length=1, max_length=1)]


class SetRequirement(BaseRequirement):
    """Represents a requirement for the presence of a label in a set."""

    operator: Literal[Operator.IN, Operator.NOT_IN]
    values: Annotated[frozenset[LabelValue], Field(min_length=1)]


class PresenceRequirement(BaseRequirement):
    """Represents a requirement for the presence of a label."""

    operator: Literal[Operator.EXISTS, Operator.DOES_NOT_EXIST]
    values: Annotated[
        frozenset[LabelValue],
        Field(default_factory=frozenset, max_length=0),
    ]


Requirement = Annotated[
    EqualityRequirement | SetRequirement | PresenceRequirement,
    Field(discriminator="operator"),
]

_NOT_SET = object()
_requirement_adapter = TypeAdapter[Requirement](Requirement)


def build_requirement(
    key: str,
    operator: str,
    values: Any = _NOT_SET,
) -> Requirement:
    """Shortcut for creating a label selector requirement."""
    obj = {"key": key, "operator": operator}
    if values is not _NOT_SET:
        obj["values"] = values
    return _requirement_adapter.validate_python(obj)


@total_ordering
class LabelSelector(RootModel[frozenset[Requirement]]):
    """Container for a set of label selector requirements.

    Provides methods for matching a dictionary of labels against all contained
    requirements.

    Attributes:
        requirements (tuple[Requirement, ...]): The set of requirements.

    Example:
        >>> # The selector can be created using the Pydantic style.
        >>> selector = LabelSelector.model_validate([
        ...     {"key": "environment", "operator": "Equals", "values": ["production"]},
        ...     {"key": "tier", "operator": "In", "values": ["frontend", "backend"]},
        ...     {"key": "enabled", "operator": "Exists"},
        ... ])
        >>> selector.matches({
        ...     "environment": "production",
        ...     "tier": "frontend",
        ...     "enabled": "true",
        ... })
        True

        >>> # Or with a string representation.
        >>> s = "environment=production, tier in (frontend,backend), enabled"
        >>> selector = LabelSelector.model_validate(s)
        >>> selector.matches({
        ...     "environment": "production",
        ...     "tier": "frontend",
        ...     "enabled": "true",
        ... })
        True
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        json_schema_extra={"description": "A set of label selector requirements."},
    )

    def __str__(self) -> str:
        return ", ".join(str(req) for req in self.requirements)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, LabelSelector):
            return NotImplemented
        return self.requirements < other.requirements

    @cached_property
    def requirements(self) -> tuple[Requirement, ...]:
        return tuple(sorted(self.root))

    def matches(self, labels: Mapping[str, str]) -> bool:
        """Check if labels match all requirements in this selector.

        Args:
            labels: A mapping of label keys to their values.

        Returns:
            True if all requirements in this selector match the given labels,
            False otherwise.
        """

        return all(requirement.matches(labels) for requirement in self.requirements)

    @field_serializer("root")
    def _serialize_root(self, _: Any, __: Any) -> tuple[Requirement, ...]:
        return self.requirements

    @classmethod
    def from_str(cls, s: str) -> Self:
        """Create a LabelSelector from a string representation.

        The selector string uses a comma-separated list of requirements. Each
        requirement can use one of the following syntaxes:

        - Equality-based: `key=value`, `key==value`, or `key!=value`
        - Set-based: `key in (value1,value2,...)` or `key notin (value1,value2,...)`
        - Presence: `key` or `!key`

        Requirements can be combined with commas, and whitespace is ignored.
        Empty values are allowed in both equality-based and set-based requirements.

        Args:
            s: The selector string to parse.

        Returns:
            A new LabelSelector instance.

        Raises:
            ValueError: If the selector string is invalid.

        Example:
            >>> LabelSelector.from_str("env=prod, tier in (frontend,backend)")
            LabelSelector(...)
            >>> LabelSelector.from_str("!experimental, ready")
            LabelSelector(...)
            >>> LabelSelector.from_str("version!=2.0, environment=staging")
            LabelSelector(...)
        """
        requirements = tuple(parse(s))
        return cls.model_validate(requirements)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: "GetCoreSchemaHandler",
    ) -> "CoreSchema":
        default_schema = handler(source_type)
        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls.from_str),
                default_schema,
            ]
        )
        return core_schema.union_schema([default_schema, from_str_schema])


# ==== Lexer ====


class Token(Enum):
    # Misc
    ERROR = auto()
    END_OF_STRING = auto()
    IDENTIFIER = auto()
    # Delimiter
    OPEN_PAR = auto()
    CLOSED_PAR = auto()
    COMMA = auto()
    # Equality Operator
    EQUALS = auto()
    DOUBLE_EQUALS = auto()
    NOT_EQUALS = auto()
    # Set Operator
    IN = auto()
    NOT_IN = auto()
    # Presence Operator
    DOES_NOT_EXIST = auto()


STRING_TO_TOKEN = {
    "(": Token.OPEN_PAR,
    ")": Token.CLOSED_PAR,
    ",": Token.COMMA,
    "=": Token.EQUALS,
    "==": Token.DOUBLE_EQUALS,
    "!=": Token.NOT_EQUALS,
    "in": Token.IN,
    "notin": Token.NOT_IN,
    "!": Token.DOES_NOT_EXIST,
}

WHITESPACE = frozenset(" \t\r\n")
SPECIAL_SYMBOLS = frozenset("=!(),")
WHITESPACE_AND_SPECIAL_SYMBOLS = WHITESPACE | SPECIAL_SYMBOLS


def lex(s: str) -> Iterable[tuple[Token, str]]:
    """Tokenizes a label selector string into a sequence of tokens."""
    length = len(s)
    pos = 0
    while pos < length:
        # Whitespace
        if s[pos] in WHITESPACE:
            pos += 1
            continue

        # Special Symbol
        if s[pos] in SPECIAL_SYMBOLS:
            buffer = []
            last_item = None
            while pos < length and s[pos] in SPECIAL_SYMBOLS:
                buffer.append(s[pos])
                pos += 1
                if (buf := "".join(buffer)) in STRING_TO_TOKEN:
                    last_item = STRING_TO_TOKEN[buf], buf
                elif last_item:  # pragma: no cover (always true)
                    pos -= 1
                    break
            if not last_item:  # pragma: no cover (not reachable)
                yield (
                    Token.ERROR,
                    f"error expected: keyword found '{''.join(buffer)}'",
                )
            else:
                yield last_item

        # Identifier
        else:
            buffer = []
            while pos < length and s[pos] not in WHITESPACE_AND_SPECIAL_SYMBOLS:
                buffer.append(s[pos])
                pos += 1
            buf = "".join(buffer)
            if buf in STRING_TO_TOKEN:
                yield STRING_TO_TOKEN[buf], buf
            else:
                yield Token.IDENTIFIER, buf

    yield Token.END_OF_STRING, ""


# ==== Parser ====

T = TypeVar("T")


class Peekable(Iterator[T]):
    """Iterator that supports peeking at the n-th upcoming element."""

    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
        self.buffer = deque[T]()

    def __next__(self) -> T:
        if self.buffer:
            return self.buffer.popleft()
        return next(self.iterator)

    def peek(self, n: int = 1) -> T:
        """Returns the n-th upcoming element without advancing the iterator."""
        if n < 1:  # pragma: no cover (safeguard)
            raise ValueError("n must be at least 1")
        while len(self.buffer) < n:
            self.buffer.append(next(self.iterator))
        return self.buffer[n - 1]


TokenSequence = Peekable[tuple[Token, str]]

TOKEN_TO_OPERATOR = {
    # Only binary operators should be listed here,
    # unary operators are handled separately.
    Token.EQUALS: Operator.EQUALS,
    Token.DOUBLE_EQUALS: Operator.DOUBLE_EQUALS,
    Token.NOT_EQUALS: Operator.NOT_EQUALS,
    Token.IN: Operator.IN,
    Token.NOT_IN: Operator.NOT_IN,
}

# Tokens that can be treated as identifiers.
IDENTIFIER_TOKENS = frozenset(
    {
        Token.IDENTIFIER,
        Token.IN,  # Key/value named "in"
        Token.NOT_IN,  # Key/value named "notin"
    }
)

# Tokens that start a requirement.
START_OF_REQUIREMENT = frozenset({*IDENTIFIER_TOKENS, Token.DOES_NOT_EXIST})


# Label selector string syntax is defined as following:
#
# <selector-syntax>         ::= <requirement> | <requirement> "," <selector-syntax>
# <requirement>             ::= ["!"] KEY
#                               [ <set-based-restriction>
#                               | <exact-match-restriction> ]
# <set-based-restriction>   ::= <inclusion-exclusion> <value-set>
# <inclusion-exclusion>     ::= "in" | "notin"
# <value-set>               ::= "(" <values> ")"
# <values>                  ::= VALUE | VALUE "," <values>
# <exact-match-restriction> ::= "=" VALUE | "==" VALUE | "!=" VALUE
#
# Notes:
# - The formats of KEY and VALUE are validated by the LabelKey and LabelValue classes
#   above.
# - Whitespace characters (space, tab, carriage return, and newline) are treated as
#   delimiters.


def parse(s: str) -> Iterable[Requirement]:
    """Parses a label selector string into a sequence of requirements."""
    tokens = Peekable(iter(lex(s)))
    while True:
        next_token, next_value = tokens.peek()
        if next_token == Token.END_OF_STRING:
            break
        if next_token not in START_OF_REQUIREMENT:
            raise ValueError(
                f"found '{next_value}', expected: !, identifier, or 'end of string'"
            )

        try:
            requirement = parse_requirement(tokens)
        except ValueError as e:
            raise ValueError(f"unable to parse requirement: {e}") from e
        else:
            yield requirement

        token, value = next(tokens)
        if token == Token.END_OF_STRING:
            break
        if token == Token.COMMA:
            next_token, next_value = tokens.peek()
            if next_token not in START_OF_REQUIREMENT:
                raise ValueError(
                    f"found '{next_value}', expected: identifier after ','"
                )
            continue
        raise ValueError(f"found '{value}', expected: ',' or 'end of string'")


def parse_requirement(tokens: TokenSequence) -> Requirement:
    operator = None

    token, value = next(tokens)
    if token == Token.DOES_NOT_EXIST:
        operator = Operator.DOES_NOT_EXIST
        token, value = next(tokens)

    if token not in IDENTIFIER_TOKENS:
        raise ValueError(f"found '{value}', expected: identifier")

    label = LabelKey(value)

    next_token, next_value = tokens.peek()
    if next_token in (Token.END_OF_STRING, Token.COMMA) and operator is None:
        operator = Operator.EXISTS

    if operator is not None:
        return build_requirement(key=label, operator=operator)

    token, value = next(tokens)
    operator = TOKEN_TO_OPERATOR.get(token)
    if operator is None:
        raise ValueError(f"found '{value}', expected: in, notin, =, ==, !=")

    match operator:
        case Operator.EQUALS | Operator.DOUBLE_EQUALS | Operator.NOT_EQUALS:
            values = parse_exact_value(tokens)
        case Operator.IN | Operator.NOT_IN:
            values = parse_values(tokens)
        case _:  # pragma: no cover (not reachable)
            raise ValueError(f"unexpected operator: {operator}")
    return build_requirement(key=label, operator=operator, values=values)


def parse_exact_value(tokens: TokenSequence) -> set[LabelValue]:
    next_token, next_value = tokens.peek()
    if next_token in (Token.END_OF_STRING, Token.COMMA):  # Handle "= $"
        return {LabelValue("")}

    token, value = next(tokens)
    if token not in IDENTIFIER_TOKENS:
        raise ValueError(f"found '{value}', expected: identifier")
    return {LabelValue(value)}


def parse_values(tokens: TokenSequence) -> set[LabelValue]:
    token, value = next(tokens)
    if token != Token.OPEN_PAR:
        raise ValueError(f"found '{value}', expected: '('")

    next_token, next_value = tokens.peek()
    if next_token == Token.CLOSED_PAR:  # Handle "()"
        next(tokens)
        return {LabelValue("")}
    if next_token not in (*IDENTIFIER_TOKENS, Token.COMMA):
        raise ValueError(f"found '{next_value}', expected: ',', ')', or identifier")

    values = parse_identifiers_list(tokens)

    token, value = next(tokens)
    if token != Token.CLOSED_PAR:  # pragma: no cover (not reachable)
        raise ValueError(f"found '{value}', expected: ')'")

    return values


def parse_identifiers_list(tokens: TokenSequence) -> set[LabelValue]:
    identifiers = set[LabelValue]()
    while True:
        token, value = next(tokens)
        match token:
            case _ if token in IDENTIFIER_TOKENS:
                identifiers.add(LabelValue(value))

                next_token, next_value = tokens.peek()
                if next_token == Token.CLOSED_PAR:
                    break
                elif next_token != Token.COMMA:
                    raise ValueError(f"found '{next_value}', expected: ',' or ')'")

            case Token.COMMA:
                if not identifiers:  # Handle "(,"
                    identifiers.add(LabelValue(""))

                next_token, next_value = tokens.peek()
                if next_token == Token.CLOSED_PAR:  # Handle ",)"
                    identifiers.add(LabelValue(""))
                    break
                if next_token == Token.COMMA:  # Handle ",,"
                    identifiers.add(LabelValue(""))
                    # This line appears in the Kubernetes source code. However,
                    # retaining it will cause a failure when parsing strings like
                    # "key in (value,,)".
                    # Ref: https://github.com/kubernetes/kubernetes/blob/0b8133816b9e78f96042ae42150998299f134ea7/staging/src/k8s.io/apimachinery/pkg/labels/selector.go#L838
                    # next(tokens)

            case _:
                raise ValueError(f"found '{value}', expected: ',', or identifier")

    return identifiers
