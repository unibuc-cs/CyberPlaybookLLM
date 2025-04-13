import re

def protect_title_case(text):
    return re.sub(
        r'title=\{(.+?)\}',
        lambda m: "title={%s}" % re.sub(r'\b([A-Z][a-zA-Z0-9\+\-]*|[A-Z]{2,})\b', r'{\1}', m.group(1)),
        text,
        flags=re.DOTALL
    )

with open("references.bib", "r", encoding="utf-8") as f:
    content = f.read()

with open("references_titlecase_fixed.bib", "w", encoding="utf-8") as f:
    f.write(protect_title_case(content))
