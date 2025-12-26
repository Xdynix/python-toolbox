import unicodedata
from typing import Any


def sanitize_text(value: Any) -> str:
    """Clean a text value by removing unprintable characters and normalizing whitespace.

    Converts all Unicode whitespace (including NBSP) to single ASCII spaces, removes
    control and formatting characters, and performs NFC normalization without altering
    meaningful symbols such as Greek letters, accents, or mathematical notation.

    This function is intentionally framework-agnostic. When used with Pydantic, apply
    it via ``BeforeValidator(sanitize_text)`` to chain with additional
    ``StringConstraints``.

    Examples:
        >>> sanitize_text("hello\\u00a0world")  # NBSP to space
        'hello world'
        >>> sanitize_text("line1\\nline2")  # Newline to space
        'line1 line2'
        >>> sanitize_text("tab\\there")  # Tab to space
        'tab here'
        >>> sanitize_text("text\\u200b\\u200c")  # Remove zero-width chars
        'text'
        >>> sanitize_text("  multiple   spaces  ")  # Collapse and trim
        'multiple spaces'
        >>> sanitize_text("α-helix β-sheet")  # Preserve Greek letters
        'α-helix β-sheet'
        >>> sanitize_text("H₂O CO₂")  # Preserve subscripts
        'H₂O CO₂'
        >>> sanitize_text("10² m²")  # Preserve superscripts
        '10² m²'
        >>> sanitize_text("∀x∈ℝ, x²≥0")  # Preserve math symbols
        '∀x∈ℝ, x²≥0'
        >>> sanitize_text("José García")  # Preserve accents
        'José García'
        >>> sanitize_text("\\x00control\\x01chars")  # Remove control chars
        'controlchars'
        >>> sanitize_text("\\u200b\\u200b\\u200b")  # All invisible becomes empty
        ''
        >>> sanitize_text("   ")  # All whitespace becomes empty
        ''
        >>> sanitize_text("test\\u2028line\\u2029break")  # Unicode line separators
        'test line break'
        >>> sanitize_text("cafe\\u0301")  # NFC normalization (decomposed -> composed)
        'café'
        >>> sanitize_text(42)
        Traceback (most recent call last):
          ...
        ValueError: Input should be a valid string.
    """  # noqa: RUF002
    if not isinstance(value, str):
        raise ValueError("Input should be a valid string.")

    normalized_value = unicodedata.normalize("NFC", value)
    cleaned_chars = []

    for char in normalized_value:
        if char.isspace():
            cleaned_chars.append(" ")
            continue

        category = unicodedata.category(char)
        if category.startswith("C"):
            continue

        cleaned_chars.append(char)

    return " ".join("".join(cleaned_chars).split())


def sanitize_formatted_text(value: Any) -> str:
    """Clean formatted text while preserving line structure and spacing.

    Similar to ``sanitize_text`` but preserves newlines, tabs, and all internal spacing
    to support formatted content like text-based tables. Each line is cleaned
    individually: control characters and zero-width characters are removed, Unicode is
    normalized (NFC), and non-standard whitespace (NBSP, etc.) is converted to regular
    spaces. Trailing whitespace is removed per line, and trailing empty lines are
    removed.

    Examples:
        >>> sanitize_formatted_text("Line 1\\nLine 2")
        'Line 1\\nLine 2'
        >>> sanitize_formatted_text("  Indented\\n    More indented")
        '  Indented\\n    More indented'
        >>> sanitize_formatted_text("Name\\tAge\\nAlice\\t25\\nBob\\t30")
        'Name\\tAge\\nAlice\\t25\\nBob\\t30'
        >>> sanitize_formatted_text("Multiple   spaces   preserved")
        'Multiple   spaces   preserved'
        >>> sanitize_formatted_text("Text\\u00a0NBSP")
        'Text NBSP'
        >>> sanitize_formatted_text("\\u200bInvisible removed\\nNext line")
        'Invisible removed\\nNext line'
        >>> sanitize_formatted_text("Trailing spaces  \\nGone   ")
        'Trailing spaces\\nGone'
        >>> sanitize_formatted_text("Empty\\n\\nLines preserved")
        'Empty\\n\\nLines preserved'
        >>> sanitize_formatted_text("Text\\n\\n")
        'Text'
        >>> sanitize_formatted_text("α-helix\\nβ-sheet")
        'α-helix\\nβ-sheet'
        >>> sanitize_formatted_text("\\x00control\\x01\\nremoved")
        'control\\nremoved'
        >>> sanitize_formatted_text("cafe\\u0301\\nline 2")
        'café\\nline 2'
        >>> sanitize_formatted_text(42)
        Traceback (most recent call last):
          ...
        ValueError: Input should be a valid string.
    """  # noqa: RUF002
    if not isinstance(value, str):
        raise ValueError("Input should be a valid string.")

    normalized_value = unicodedata.normalize("NFC", value)
    lines = normalized_value.splitlines()
    cleaned_lines = []

    for line in lines:
        cleaned_chars = []

        for char in line:
            if char in (" ", "\t"):
                cleaned_chars.append(char)
                continue

            if char.isspace():
                cleaned_chars.append(" ")
                continue

            category = unicodedata.category(char)
            if category.startswith("C"):
                continue

            cleaned_chars.append(char)

        cleaned_lines.append("".join(cleaned_chars).rstrip())

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)
