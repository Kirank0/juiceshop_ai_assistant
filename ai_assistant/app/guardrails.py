import re

REFUSAL_MESSAGE = (
    "Sorry, I can't share recipes or explain how our products are prepared or made. "
    "I'm happy to help with product details or pricing instead!"
)

_PREP_VERBS = (
    r"make|made|making|prepare[ds]?|preparing|produce[ds]?|producing|cook(?:ed|ing)?|brew(?:ed|ing)?|"
    r"mix(?:ed|ing)?|blend(?:ed|ing)?|squeeze[ds]?|squeezing|manufacture[ds]?|manufacturing"
)

_RECIPE_PATTERNS = [
    r"\brecipes?\b",
    # "how is it made", "how do you prepare", "how to make", "how can I brew ..."
    rf"\bhow\b.{{0,40}}\b(?:{_PREP_VERBS})\b",
    r"\b(?:preparation|production|manufacturing|brewing|cooking)\s+(?:method|process|steps?|instructions?)\b",
    r"\bstep[\s-]+by[\s-]+step\b",
    r"\binstructions?\s+(?:to|for)\s+(?:make|making|prepar)",
    r"\b(?:home\s*made|diy|do\s+it\s+yourself)\b",
    r"\bmake\b.{0,30}\b(?:at\s+home|myself|yourself|my\s+own|your\s+own)\b",
    r"\bingredients?\b.{0,30}\b(?:steps?|method|quantities|measurements|ratio|proportions?)\b",
]
_RECIPE_RE = re.compile("|".join(f"(?:{p})" for p in _RECIPE_PATTERNS), re.IGNORECASE | re.DOTALL)


def is_recipe_request(text: str) -> bool:
    """Return True if the message asks for a recipe or how a product is prepared/made."""
    return bool(_RECIPE_RE.search(text))
