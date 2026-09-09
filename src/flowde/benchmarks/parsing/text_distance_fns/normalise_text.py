import re
import unicodedata

BULLET_CHARS = (
    "\N{BULLET}"  # U+2022 BULLET
    "\N{BLACK CIRCLE}"  # U+25CF BLACK CIRCLE
    "\N{WHITE CIRCLE}"  # U+25CB WHITE CIRCLE
    "\N{BLACK SMALL SQUARE}"  # U+25AA BLACK SMALL SQUARE
    "\N{WHITE SMALL SQUARE}"  # U+25AB WHITE SMALL SQUARE
    "\N{WHITE MEDIUM SQUARE}"
    "\N{BLACK SQUARE}"  # U+25A0 BLACK SQUARE
    "\N{WHITE SQUARE}"  # U+25A1 WHITE SQUARE
    "\N{WHITE BULLET}"  # U+25E6 WHITE BULLET
    "\N{TRIANGULAR BULLET}"  # U+2023 TRIANGULAR BULLET
    "\N{HYPHEN BULLET}"  # U+2043 HYPHEN BULLET
    "\N{BULLET OPERATOR}"  # U+2219 BULLET OPERATOR
    "\N{BLACK DIAMOND SUIT}"  # U+2666 BLACK DIAMOND SUIT
    "\N{BLACK DIAMOND}"  # U+25C6 BLACK DIAMOND
    "\N{BALLOT BOX}"  # U+2610 BALLOT BOX
)

DASH_CHARS = (
    "\N{HYPHEN}"  # U+2010 HYPHEN
    "-"  # U+002D HYPHEN-MINUS
    "\N{FIGURE DASH}"  # U+2012 FIGURE DASH
    "\N{EN DASH}"  # U+2013 EN DASH
    "\N{EM DASH}"  # U+2014 EM DASH
    "\N{HORIZONTAL BAR}"  # U+2015 HORIZONTAL BAR
    "\N{MINUS SIGN}"  # U+2212 MINUS SIGN
)

LEFT_QUOTES = (
    "\N{LEFT SINGLE QUOTATION MARK}"  # U+2018 LEFT SINGLE QUOTATION MARK
    "\N{RIGHT SINGLE QUOTATION MARK}"  # U+2019 RIGHT SINGLE QUOTATION MARK
    "\N{SINGLE LOW-9 QUOTATION MARK}"  # U+201A SINGLE LOW-9 QUOTATION MARK
    "\N{SINGLE HIGH-REVERSED-9 QUOTATION MARK}"  # U+201B QUOTATION MARK
)

RIGHT_QUOTES = (
    "\N{LEFT DOUBLE QUOTATION MARK}"  # U+201C LEFT DOUBLE QUOTATION MARK
    "\N{RIGHT DOUBLE QUOTATION MARK}"  # U+201D RIGHT DOUBLE QUOTATION MARK
    "\N{DOUBLE LOW-9 QUOTATION MARK}"  # U+201E DOUBLE LOW-9 QUOTATION MARK
    "\N{DOUBLE HIGH-REVERSED-9 QUOTATION MARK}"  # U+201F QUOTATION MARK
)

CARET_CHARS = (
    "^"  # U+005E CIRCUMFLEX ACCENT
    "\N{MODIFIER LETTER CIRCUMFLEX ACCENT}"  # U+02C6
    "\N{COMBINING CIRCUMFLEX ACCENT}"  # U+0302 ◌̂
)


def nfc_and_space_normalise(
    text: str,
    *,
    normalise_bullets: bool = True,
    normalise_dashes: bool = True,
    normalise_quotes: bool = True,
    normalise_punctuation_spacing: bool = True,
    normalise_hyphen_spacing: bool = True,
    normalise_carets: bool = True,
    normalise_plus_spacing: bool = True,
    normalise_html_breaks: bool = True,
    normalise_letter_o_bullets: bool = True,
    normalise_ordinal_superscripts: bool = True,
) -> str:
    text = unicodedata.normalize("NFC", text)

    if normalise_ordinal_superscripts:
        # Normalise ordinal suffixes written with superscripts:
        # "1ˢᵗ" -> "1st"
        # "2ⁿᵈ" -> "2nd"
        # "3ʳᵈ" -> "3rd"
        # "4ᵗʰ" -> "4th"
        # "23ᵗʰ" -> "23th"
        text = re.sub(r"(?<=\d)ˢᵗ", "st", text)
        text = re.sub(r"(?<=\d)ⁿᵈ", "nd", text)
        text = re.sub(r"(?<=\d)ʳᵈ", "rd", text)
        text = re.sub(r"(?<=\d)ᵗʰ", "th", text)

    # Normalise line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Treat tabs as ordinary spaces.
    text = text.replace("\t", " ")

    if normalise_html_breaks:
        # Collapse repeated HTML line breaks:
        # "<br><br>" -> "<br>"
        # "<br> <br>" -> "<br>"
        # "<br/>\n<br />" -> "<br>"
        text = re.sub(r"(?:<br\s*/?>\s*){2,}", "<br>", text, flags=re.IGNORECASE)

    if normalise_bullets:
        # Convert many bullet glyphs to one canonical bullet.
        text = re.sub(f"[{re.escape(BULLET_CHARS)}]", "•", text)

        if normalise_letter_o_bullets:
            # Convert OCR/transcription "o" bullets at the start of lines:
            # "o Non-smokers" -> "○ Non-smokers"
            # "    o   Non-smokers" -> "○ Non-smokers"
            # Only applies when o/O is the first non-space character on a line
            # and is followed by whitespace, so ordinary words are not changed.
            text = re.sub(r"(?m)^[ \t]*[oO]\s+", "• ", text)

    if normalise_dashes:
        # Convert en dash, em dash, minus sign, etc. to hyphen.
        text = re.sub(f"[{re.escape(DASH_CHARS)}]", "-", text)

    if normalise_quotes:
        # Convert curly single quotes to straight single quotes.
        text = re.sub(f"[{re.escape(LEFT_QUOTES)}]", "'", text)

        # Convert curly double quotes to straight double quotes.
        text = re.sub(f"[{re.escape(RIGHT_QUOTES)}]", '"', text)

    if normalise_carets:
        # Convert visually similar caret/circumflex markers to "^":
        # "219 subscript^" -> "219^"
        # "219̂" -> "219^"
        text = re.sub(f"[{re.escape(CARET_CHARS)}]", "^", text)

    # Collapse repeated spaces, but preserve single newlines.
    text = re.sub(r" +", " ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    if normalise_punctuation_spacing:
        # Normalise bullet markers at the start of lines, allowing indentation:
        # "•Withdrew" -> "• Withdrew"
        # "  •  Withdrew" -> "• Withdrew"
        # "-Withdrew" -> "- Withdrew"
        # "  -  Withdrew" -> "- Withdrew"
        text = re.sub(r"(?m)^[ \t]*•\s*", "• ", text)
        text = re.sub(r"(?m)^[ \t]*-\s*", "- ", text)

        # Remove spaces before common punctuation:
        # "participants ," -> "participants,"
        text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)

        # Remove spaces after opening brackets:
        # "( n = 5)" -> "(n = 5)"
        text = re.sub(r"([(\[])\s+", r"\1", text)

        # Normalise spaces around equals:
        # "n=5", "n =5", "n= 5" -> "n = 5"
        text = re.sub(r"\s*=\s*", " = ", text)

        if normalise_plus_spacing:
            # Remove spaces around plus signs:
            # "mean + SD" -> "mean+SD"
            # "n + 5" -> "n+5"
            # "++ +" -> "+++"
            # Does not remove newlines around plus signs.
            text = re.sub(r"[^\S\n]*\+[^\S\n]*", "+", text)

        if normalise_hyphen_spacing:
            # Remove spaces around hyphens only when the hyphen is inside a word:
            # "post - partum" -> "post-partum"
            # "6 - month" -> "6-month"
            # but "5 - 10" stays "5 - 10"
            text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)

        # Re-collapse spaces after punctuation spacing edits.
        text = re.sub(r" +", " ", text)

    # Collapse repeated newlines.
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def numbers_only_normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    # Normalise common thousands separators inside numbers:
    # "1,912" -> "1912"
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    # Extract integer or decimal numbers.
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    return " ".join(numbers)
