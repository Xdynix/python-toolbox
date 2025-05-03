from typing import Any

import jsonschema
import label_selector as ls
import pydantic
import pytest
from label_selector import (
    LabelKey,
    LabelSelector,
    LabelValue,
    Operator,
    Requirement,
)
from pydantic import TypeAdapter


class TestLabelKey:
    VALID_DATA: tuple[tuple[str, tuple[str, str]], ...] = (
        ("simple", ("", "simple")),
        ("Upper.Case", ("", "Upper.Case")),
        ("prefix.com/name", ("prefix.com", "name")),
        # While this isn't strictly a DNS subdomain, Kubernetes permits it.
        ("prefix/name", ("prefix", "name")),
        (f"{'a' * 253}/{'b' * 63}", ("a" * 253, "b" * 63)),
    )
    INVALID_DATA: tuple[tuple[Any, type[Exception], str], ...] = (
        (None, TypeError, "LabelKey requires a string, not.+"),
        (1, TypeError, "LabelKey requires a string, not.+"),
        (
            "a/b/c",
            ValueError,
            "a qualified name must consist of alphanumeric characters.+",
        ),
        ("/name", ValueError, "prefix part must be non-empty"),
        (
            f"{'a' * 254}/name",
            ValueError,
            "prefix part must be no more than 253 characters",
        ),
        (
            "Upper.Case/name",
            ValueError,
            "prefix part a lowercase RFC 1123 subdomain must consist of lower case.+",
        ),
        (
            ".com/name",
            ValueError,
            "prefix part a lowercase RFC 1123 subdomain must consist of lower case.+",
        ),
        ("", ValueError, "name part must be non-empty"),
        ("a" * 64, ValueError, "name part must be no more than 63 characters"),
        (
            "invalid,key",
            ValueError,
            "name part must consist of alphanumeric characters.+",
        ),
        ("_a", ValueError, "name part must consist of alphanumeric characters.+"),
    )

    def test_repr(self) -> None:
        key = LabelKey("example.com/name")
        assert repr(key) == "LabelKey('example.com/name')"

    def test_schema(self) -> None:
        schema = TypeAdapter(LabelKey).json_schema()
        assert schema["type"] == "string"
        assert schema["pattern"] == LabelKey.PATTERN.pattern
        assert schema["minLength"] == LabelKey.MIN_LENGTH
        assert schema["maxLength"] == LabelKey.MAX_LENGTH

    @pytest.mark.parametrize("s, expected", VALID_DATA)
    def test_valid(self, s: str, expected: tuple[str, str]) -> None:
        key = LabelKey(s)
        assert key.prefix == expected[0]
        assert key.name == expected[1]
        assert key == s
        assert isinstance(key, LabelKey)

    @pytest.mark.parametrize("s", (s for s, _ in VALID_DATA))
    def test_valid_pydantic(self, s: str) -> None:
        key = TypeAdapter(LabelKey).validate_python(s)
        assert key == s
        assert isinstance(key, LabelKey)

    @pytest.mark.parametrize("s", (s for s, _ in VALID_DATA))
    def test_valid_jsonschema(self, s: str) -> None:
        schema = TypeAdapter(LabelKey).json_schema()
        jsonschema.validate(s, schema)

    @pytest.mark.parametrize("s, exc, message", INVALID_DATA)
    def test_invalid(self, s: Any, exc: type[Exception], message: str) -> None:
        with pytest.raises(exc, match=message):
            LabelKey(s)

    @pytest.mark.parametrize("s, exc", ((s, exc) for s, exc, _ in INVALID_DATA))
    def test_invalid_pydantic(self, s: Any, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            TypeAdapter(LabelKey).validate_python(s)

    @pytest.mark.parametrize("s", (s for s, _, _ in INVALID_DATA))
    def test_invalid_jsonschema(self, s: Any) -> None:
        schema = TypeAdapter(LabelKey).json_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(s, schema)


class TestLabelValue:
    VALID_DATA: tuple[str, ...] = (
        "simple",
        "Upper.Case",
        "",
    )
    INVALID_DATA: tuple[tuple[Any, type[Exception], str], ...] = (
        (None, TypeError, "LabelValue requires a string, not.+"),
        (1, TypeError, "LabelValue requires a string, not.+"),
        ("a" * 64, ValueError, "must be no more than 63 characters"),
        (
            "invalid,value",
            ValueError,
            "a valid label must be an empty string or consist of.+",
        ),
        (
            "value/slash",
            ValueError,
            "a valid label must be an empty string or consist of.+",
        ),
        (
            "_a",
            ValueError,
            "a valid label must be an empty string or consist of.+",
        ),
    )

    def test_repr(self) -> None:
        value = LabelValue("example")
        assert repr(value) == "LabelValue('example')"

    def test_schema(self) -> None:
        schema = TypeAdapter(LabelValue).json_schema()
        assert schema["type"] == "string"
        assert schema["pattern"] == LabelValue.PATTERN.pattern
        assert not schema.get("minLength")
        assert schema["maxLength"] == LabelValue.MAX_LENGTH

    @pytest.mark.parametrize("s", VALID_DATA)
    def test_valid(self, s: str) -> None:
        value = LabelValue(s)
        assert value == s
        assert isinstance(value, LabelValue)

    @pytest.mark.parametrize("s", VALID_DATA)
    def test_valid_pydantic(self, s: str) -> None:
        value = TypeAdapter(LabelValue).validate_python(s)
        assert value == s
        assert isinstance(value, LabelValue)

    @pytest.mark.parametrize("s", VALID_DATA)
    def test_valid_jsonschema(self, s: str) -> None:
        schema = TypeAdapter(LabelValue).json_schema()
        jsonschema.validate(s, schema)

    @pytest.mark.parametrize("s, exc, message", INVALID_DATA)
    def test_invalid(self, s: Any, exc: type[Exception], message: str) -> None:
        with pytest.raises(exc, match=message):
            LabelValue(s)

    @pytest.mark.parametrize("s, exc", ((s, exc) for s, exc, _ in INVALID_DATA))
    def test_invalid_pydantic(self, s: Any, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            TypeAdapter(LabelValue).validate_python(s)

    @pytest.mark.parametrize("s", (s for s, _, _ in INVALID_DATA))
    def test_invalid_jsonschema(self, s: Any) -> None:
        schema = TypeAdapter(LabelValue).json_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(s, schema)


_NOT_SET = object()


class TestRequirement:
    @pytest.mark.parametrize(
        "req, expected",
        (
            (ls.build_requirement("key", "Equals", ["value"]), "key=value"),
            (ls.build_requirement("key", "DoubleEquals", ["value"]), "key==value"),
            (ls.build_requirement("key", "NotEquals", ["value"]), "key!=value"),
            (ls.build_requirement("key", "In", ["b", "a"]), "key in (a,b)"),
            (ls.build_requirement("key", "NotIn", ["b", "a"]), "key notin (a,b)"),
            (ls.build_requirement("key", "Exists"), "key"),
            (ls.build_requirement("key", "DoesNotExist"), "!key"),
        ),
    )
    def test_str(self, req: Requirement, expected: str) -> None:
        assert str(req) == expected

    @pytest.mark.parametrize(
        "data, expected_type",
        (
            (
                {"key": "k", "operator": "Equals", "values": ["v"]},
                ls.EqualityRequirement,
            ),
            (
                {"key": "k", "operator": "DoubleEquals", "values": ["v"]},
                ls.EqualityRequirement,
            ),
            (
                {"key": "k", "operator": "NotEquals", "values": ["v"]},
                ls.EqualityRequirement,
            ),
            (
                {"key": "k", "operator": "In", "values": ["v"]},
                ls.SetRequirement,
            ),
            (
                {"key": "k", "operator": "NotIn", "values": ["v"]},
                ls.SetRequirement,
            ),
            (
                {"key": "k", "operator": "Exists"},
                ls.PresenceRequirement,
            ),
            (
                {"key": "k", "operator": "DoesNotExist"},
                ls.PresenceRequirement,
            ),
        ),
    )
    def test_discriminator(
        self,
        data: Any,
        expected_type: type[Requirement],
    ) -> None:
        requirement = ls.build_requirement(**data)
        assert isinstance(requirement, expected_type)

    @pytest.mark.parametrize("values", ([], ["value"], ["value1", "value2"]))
    def test_invalid_discriminator(self, values: Any) -> None:
        with pytest.raises(pydantic.ValidationError):
            ls.build_requirement("k", "invalid", values)

    def test_equality(self) -> None:
        r1 = ls.build_requirement("key", Operator.EQUALS, ["value"])
        r2 = ls.build_requirement("key", Operator.EQUALS, ["value"])
        assert r1 == r2
        assert r1 != "not a requirement"  # type: ignore[comparison-overlap]

        r3 = ls.build_requirement("key", Operator.EQUALS, ["other"])
        assert r1 != r3

    def test_ordering(self) -> None:
        r1 = ls.build_requirement("a", Operator.EQUALS, ["value"])
        r2 = ls.build_requirement("b", Operator.EQUALS, ["value"])
        assert r1 < r2
        with pytest.raises(TypeError):
            _ = r1 < "not a requirement"

    def test_hashable(self) -> None:
        r1 = ls.build_requirement("key", Operator.EQUALS, ["value"])
        r2 = ls.build_requirement("key", Operator.EQUALS, ["value"])
        assert r1 is not r2
        assert len({r1, r2}) == 1
        assert {r1: "value"}[r2] == "value"

    def test_values_serialization_deterministic(self) -> None:
        requirement = ls.build_requirement("key", Operator.IN, ["b", "a", "c"])
        assert requirement.model_dump(mode="json") == {
            "key": "key",
            "operator": "In",
            "values": ["a", "b", "c"],
        }


class TestEqualityRequirement:
    operators = (
        Operator.EQUALS,
        Operator.DOUBLE_EQUALS,
        Operator.NOT_EQUALS,
    )

    @pytest.mark.parametrize("op", operators)
    def test_valid(self, op: Any) -> None:
        ls.build_requirement("key", op, ["value"])

    @pytest.mark.parametrize("op", operators)
    @pytest.mark.parametrize("values", (None, [], ["value1", "value2"]))
    def test_invalid_values(self, op: Any, values: Any) -> None:
        with pytest.raises(pydantic.ValidationError):
            ls.build_requirement("key", op, values)

    @pytest.mark.parametrize(
        "op, labels, expected",
        (
            (Operator.EQUALS, {"key": "value"}, True),
            (Operator.EQUALS, {"key": "other"}, False),
            (Operator.EQUALS, {}, False),
            (Operator.DOUBLE_EQUALS, {"key": "value"}, True),
            (Operator.DOUBLE_EQUALS, {"key": "other"}, False),
            (Operator.DOUBLE_EQUALS, {}, False),
            (Operator.NOT_EQUALS, {"key": "value"}, False),
            (Operator.NOT_EQUALS, {"key": "other"}, True),
            (Operator.NOT_EQUALS, {}, True),
        ),
    )
    def test_matches(self, op: Any, labels: dict[str, str], expected: bool) -> None:
        requirement = ls.build_requirement("key", op, ["value"])
        assert requirement.matches(labels) == expected


class TestSetRequirement:
    operators = (
        Operator.IN,
        Operator.NOT_IN,
    )

    @pytest.mark.parametrize("op", operators)
    def test_valid(self, op: Any) -> None:
        ls.build_requirement("key", op, ["value1", "value2"])

    @pytest.mark.parametrize("op", operators)
    @pytest.mark.parametrize("values", (None, []))
    def test_invalid_values(self, op: Any, values: Any) -> None:
        with pytest.raises(pydantic.ValidationError):
            ls.build_requirement("key", op, values)

    @pytest.mark.parametrize(
        "op, labels, expected",
        (
            (Operator.IN, {"key": "value1"}, True),
            (Operator.IN, {"key": "value2"}, True),
            (Operator.IN, {"key": "other"}, False),
            (Operator.IN, {}, False),
            (Operator.NOT_IN, {"key": "value1"}, False),
            (Operator.NOT_IN, {"key": "value2"}, False),
            (Operator.NOT_IN, {"key": "other"}, True),
            (Operator.NOT_IN, {}, True),
        ),
    )
    def test_matches(self, op: Any, labels: dict[str, str], expected: bool) -> None:
        requirement = ls.build_requirement("key", op, ["value1", "value2"])
        assert requirement.matches(labels) == expected


class TestPresenceRequirement:
    operators = (
        Operator.EXISTS,
        Operator.DOES_NOT_EXIST,
    )

    @pytest.mark.parametrize("op", operators)
    def test_valid(self, op: Any) -> None:
        ls.build_requirement("key", op)

    @pytest.mark.parametrize("op", operators)
    @pytest.mark.parametrize("values", (None, ["value"]))
    def test_invalid_values(self, op: Any, values: Any) -> None:
        with pytest.raises(pydantic.ValidationError):
            ls.build_requirement("key", op, values)

    @pytest.mark.parametrize(
        "op, labels, expected",
        (
            (Operator.EXISTS, {"key": "value"}, True),
            (Operator.EXISTS, {"key": "other"}, True),
            (Operator.EXISTS, {}, False),
            (Operator.DOES_NOT_EXIST, {"key": "value"}, False),
            (Operator.DOES_NOT_EXIST, {"key": "other"}, False),
            (Operator.DOES_NOT_EXIST, {}, True),
        ),
    )
    def test_matches(self, op: Any, labels: dict[str, str], expected: bool) -> None:
        requirement = ls.build_requirement("key", op, ())
        assert requirement.matches(labels) == expected


class TestLabelSelector:
    def test_str(self) -> None:
        selector = TypeAdapter(LabelSelector).validate_python(
            [
                {"key": "tier", "operator": "In", "values": ["frontend", "backend"]},
                {"key": "env", "operator": "Equals", "values": ["production"]},
            ],
        )
        assert str(selector) == "env=production, tier in (backend,frontend)"

    def test_equality(self) -> None:
        s1 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "env", "operator": "Equals", "values": ["production"]}],
        )
        s2 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "env", "operator": "Equals", "values": ["production"]}],
        )
        assert s1 == s2
        assert s1 != "not a selector"

        s3 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "env", "operator": "Equals", "values": ["development"]}],
        )
        assert s1 != s3

    def test_ordering(self) -> None:
        s1 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "a", "operator": "Equals", "values": ["value"]}],
        )
        s2 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "b", "operator": "Equals", "values": ["value"]}],
        )
        assert s1 < s2
        with pytest.raises(TypeError):
            _ = s1 < "not a selector"

    def test_hashable(self) -> None:
        s1 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "env", "operator": "Equals", "values": ["production"]}],
        )
        s2 = TypeAdapter(LabelSelector).validate_python(
            [{"key": "env", "operator": "Equals", "values": ["production"]}],
        )
        assert s1 is not s2
        assert len({s1, s2}) == 1
        assert {s1: "value"}[s2] == "value"

    def test_matches(self) -> None:
        selector = TypeAdapter(LabelSelector).validate_python(
            [
                {"key": "env", "operator": "Equals", "values": ["production"]},
                {"key": "tier", "operator": "In", "values": ["frontend", "backend"]},
            ]
        )
        assert selector.matches({"env": "production", "tier": "frontend"})
        assert not selector.matches({"env": "staging", "tier": "frontend"})

    def test_serialization_deterministic(self) -> None:
        selector = TypeAdapter(LabelSelector).validate_python(
            [
                {"key": "tier", "operator": "In", "values": ["frontend", "backend"]},
                {"key": "env", "operator": "Equals", "values": ["production"]},
            ]
        )
        assert selector.model_dump(mode="json") == [
            {"key": "env", "operator": "Equals", "values": ["production"]},
            {"key": "tier", "operator": "In", "values": ["backend", "frontend"]},
        ]

    def test_empty_selector(self) -> None:
        selector = TypeAdapter(LabelSelector).validate_python([])
        assert selector.matches({})
        assert selector.matches({"key": "value"})

    @pytest.mark.parametrize(
        "s, expected",
        (
            # Single Requirement
            ("a=b", [{"key": "a", "operator": "Equals", "values": ["b"]}]),
            ("a==b", [{"key": "a", "operator": "DoubleEquals", "values": ["b"]}]),
            ("a!=b", [{"key": "a", "operator": "NotEquals", "values": ["b"]}]),
            ("a in (b)", [{"key": "a", "operator": "In", "values": ["b"]}]),
            ("a notin (b)", [{"key": "a", "operator": "NotIn", "values": ["b"]}]),
            ("a", [{"key": "a", "operator": "Exists"}]),
            ("!a", [{"key": "a", "operator": "DoesNotExist"}]),
            # Multiple Requirements
            (
                "a=b, !c",
                [
                    {"key": "a", "operator": "Equals", "values": ["b"]},
                    {"key": "c", "operator": "DoesNotExist"},
                ],
            ),
            (
                "a=b, c in (d,e)",
                [
                    {"key": "a", "operator": "Equals", "values": ["b"]},
                    {"key": "c", "operator": "In", "values": ["d", "e"]},
                ],
            ),
            (
                "example.com/release=stable, foo/bar in (baz)",
                [
                    {
                        "key": "example.com/release",
                        "operator": "Equals",
                        "values": ["stable"],
                    },
                    {"key": "foo/bar", "operator": "In", "values": ["baz"]},
                ],
            ),
            # Empty Selector
            ("", []),
            # Special Key/Value
            ("in=true", [{"key": "in", "operator": "Equals", "values": ["true"]}]),
            ("in notin (in)", [{"key": "in", "operator": "NotIn", "values": ["in"]}]),
            (
                "notin notin (in,notin)",
                [{"key": "notin", "operator": "NotIn", "values": ["in", "notin"]}],
            ),
            # Arbitrary Spaces
            ("  a = b", [{"key": "a", "operator": "Equals", "values": ["b"]}]),
            ("a   in(b)  ", [{"key": "a", "operator": "In", "values": ["b"]}]),
            (
                "a   in(b  , c , , )  ",
                [{"key": "a", "operator": "In", "values": ["", "b", "c"]}],
            ),
            (
                "a=b,c\tin\n(d,e)",
                [
                    {"key": "a", "operator": "Equals", "values": ["b"]},
                    {"key": "c", "operator": "In", "values": ["d", "e"]},
                ],
            ),
            # Empty Values
            ("a=", [{"key": "a", "operator": "Equals", "values": [""]}]),
            (
                "a=,b=",
                [
                    {"key": "a", "operator": "Equals", "values": [""]},
                    {"key": "b", "operator": "Equals", "values": [""]},
                ],
            ),
            ("a in ()", [{"key": "a", "operator": "In", "values": [""]}]),
            ("a in (,)", [{"key": "a", "operator": "In", "values": [""]}]),
            ("a in (,,)", [{"key": "a", "operator": "In", "values": [""]}]),
            ("a in (,,,)", [{"key": "a", "operator": "In", "values": [""]}]),
            ("a in (b,)", [{"key": "a", "operator": "In", "values": ["", "b"]}]),
            ("a in (b,,)", [{"key": "a", "operator": "In", "values": ["", "b"]}]),
            (
                "x in (foo,,baz),y,z notin ()",
                [
                    {"key": "x", "operator": "In", "values": ["", "foo", "baz"]},
                    {"key": "y", "operator": "Exists"},
                    {"key": "z", "operator": "NotIn", "values": [""]},
                ],
            ),
            # Duplicate Values
            ("a in (b,b)", [{"key": "a", "operator": "In", "values": ["b"]}]),
            # Duplicate Requirements
            ("a=b,a=b", [{"key": "a", "operator": "Equals", "values": ["b"]}]),
            (
                "a in (b,c), a in (c,b)",
                [{"key": "a", "operator": "In", "values": ["b", "c"]}],
            ),
        ),
    )
    def test_from_str(self, s: str, expected: Any) -> None:
        assert LabelSelector.from_str(s) == LabelSelector.model_validate(expected)

    @pytest.mark.parametrize(
        "s, match",
        (
            ("=a", "found '=', expected: !, identifier, or 'end of string'"),
            ("===", "found '==', expected: !, identifier, or 'end of string'"),
            ("a=b,", "found '', expected: identifier after ','"),
            ("a=b,=", "found '=', expected: identifier after ','"),
            ("a=b c", "found 'c', expected: ',' or 'end of string'"),
            ("!!", "unable to parse requirement: found '!', expected: identifier"),
            ("!,", "unable to parse requirement: found ',', expected: identifier"),
            ("!", "unable to parse requirement: found '', expected: identifier"),
            (
                "a b",
                (
                    "unable to parse requirement: found 'b', "
                    "expected: in, notin, =, ==, !="
                ),
            ),
            ("/a=b", "unable to parse requirement: prefix part must be"),
            ("@=b", "unable to parse requirement: name part must consist"),
            ("a=@", "unable to parse requirement: a valid label must be"),
            ("a=!", "unable to parse requirement: found '!', expected: identifier"),
            ("a=(", r"unable to parse requirement: found '\(', expected: identifier"),
            ("a in ", r"unable to parse requirement: found '', expected: '\('"),
            ("a in a", r"unable to parse requirement: found 'a', expected: '\('"),
            (
                "a in (",
                r"unable to parse requirement: found '', expected: ',', '\)', or id",
            ),
            (
                "a in (a a)",
                r"unable to parse requirement: found 'a', expected: ',' or '\)'",
            ),
            (
                "a in (a,=)",
                r"unable to parse requirement: found '=', expected: ',', or identifier",
            ),
        ),
    )
    def test_from_str_invalid(self, s: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LabelSelector.from_str(s)
