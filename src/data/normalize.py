"""Conservative, meaning-preserving normalization for the Myanmar/English corpus.

Design choice: we do NOT convert numerals or alter content — the gold references
must stay as the translators wrote them (Myanmar text legitimately mixes Myanmar
and Arabic digits). Numeral normalization is an EVAL-TIME operation inside IFS,
not a dataset mutation. Here we only: NFC-normalize, fix whitespace, and tidy
spacing around Myanmar section marks. Zawgyi is *detected and flagged* (not
auto-converted, since myanmar-tools is unavailable in this env) so a definitive
myanmar-tools pass can run in Colab.
"""
import re
import unicodedata

# U+1031 (vowel sign E) sits AFTER its consonant in Unicode but BEFORE it in
# Zawgyi storage. In valid Unicode every U+1031 is preceded by a base consonant
# (optionally + medials U+103B-103E). A U+1031 NOT so preceded => Zawgyi.
# (The naive "vowel before consonant" test is wrong: it fires at every normal
# syllable boundary, e.g. ကေ|က.)
_ZAWGYI = re.compile(r"(?<![က-အျ-ှ])ေ")
_BASE_CONSONANT = re.compile(r"[က-အ]")
_MY_PUNCT = re.compile(r"\s*([၊။])\s*")  # ၊ ။
_WS = re.compile(r"\s+")
_MY_DIGIT = re.compile(r"[၀-၉]")
_AR_DIGIT = re.compile(r"[0-9]")


def looks_zawgyi(text: str) -> bool:
    """Heuristic: returns True if the string shows the U+1031/U+103B-before-consonant
    pattern characteristic of Zawgyi-encoded Myanmar."""
    return bool(_ZAWGYI.search(str(text)))


def normalize_text(text: str) -> str:
    """NFC + whitespace + Myanmar-punctuation spacing. Content/numerals untouched."""
    s = unicodedata.normalize("NFC", str(text))
    s = _MY_PUNCT.sub(r"\1 ", s)   # no space before ၊/။, single space after
    s = _WS.sub(" ", s).strip()
    return s


def numeral_profile(text: str) -> str:
    """Tag a Myanmar string by which digit systems it uses (for stats)."""
    s = str(text)
    my = bool(_MY_DIGIT.search(s))
    ar = bool(_AR_DIGIT.search(s))
    if my and ar:
        return "mixed"
    if my:
        return "myanmar"
    if ar:
        return "arabic"
    return "none"
