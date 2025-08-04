import re
from typing import List

def split_into_clauses(text: str) -> List[str]:
    # Simple splitter: split on periods followed by newline or just newlines
    clauses = re.split(r'\.\s*\n|\n+', text)
    clauses = [clause.strip() for clause in clauses if clause.strip()]
    return clauses
