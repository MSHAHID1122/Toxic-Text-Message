import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)          # fix hyphenated line breaks
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse huge gaps
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
    return text.strip()