#!/usr/bin/env python3
"""
Clean Sonnet datasets from wiki artifacts and vandalism.
"""

import re
import sys

def clean_line(line):
    """Remove wiki markup and check for vandalism."""

    # Check for vandalism keywords (Russian hate speech)
    vandalism_markers = [
        'уничтожить', 'выблядь', 'сжечь заживо', 'тварь',
        'косоглазую', 'сократить штаты на 95'
    ]
    for marker in vandalism_markers:
        if marker in line.lower():
            print(f"  REMOVED (vandalism): {line[:80]}...")
            return None

    # Remove wiki markup [[...]] - keep content inside if present
    # [[word]] -> word
    # [[link|text]] -> text
    line = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', line)
    line = re.sub(r'\[\[([^\]]+)\]\]', r'\1', line)

    # Remove unclosed wiki links [[...
    line = re.sub(r'\[\[[^\]]*$', '', line)

    # Remove stray ]] at start or anywhere
    line = re.sub(r'^\s*\]\]+\s*', '', line)
    line = re.sub(r'\]\]+', '', line)

    # Remove stray image captions like "|224x224px"
    line = re.sub(r'\|\d+x\d+px', '', line)

    # Remove category garbage at end of lines
    # Pattern: "word word word Category Category Category"
    category_patterns = [
        r'\s+(groups?|bands?|people|births?|deaths?|establishments?|disestablishments?|from|in|of)\s+(from|in|of|groups?|bands?|people|the|California|United|States|England|America)\s*(\s+\w+)*\s*$',
    ]
    for pattern in category_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            # Check if it looks like category spam (multiple short words at end)
            words = line.split()
            if len(words) > 10:
                # Check last 5+ words for category-like patterns
                last_words = ' '.join(words[-8:]).lower()
                if any(x in last_words for x in ['groups from', 'bands from', 'establishments in', 'disestablishments', 'births', 'deaths']):
                    # Trim to last sentence
                    match = re.search(r'^(.+[.!?])\s*\w+\s+\w+\s+\w+', line)
                    if match:
                        cleaned = match.group(1)
                        if len(cleaned) > 50:
                            print(f"  TRIMMED categories: ...{line[-60:]}")
                            line = cleaned

    # Remove image markup remnants
    line = re.sub(r'\|+\d*x?\d*px\]*', '', line)
    line = re.sub(r'Image:[^\]|]+\.(jpg|png|jpeg|gif)[^\]]*', '', line, flags=re.IGNORECASE)

    # Clean up double spaces
    line = re.sub(r'\s+', ' ', line)
    line = line.strip()

    # Skip empty lines or too short
    if len(line) < 20:
        return None

    return line


def clean_file(input_path, output_path):
    """Clean a dataset file."""
    print(f"\nCleaning: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  Input: {len(lines)} lines")

    cleaned = []
    removed = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        result = clean_line(line)
        if result:
            cleaned.append(result)
        else:
            removed += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + '\n')

    print(f"  Output: {len(cleaned)} lines")
    print(f"  Removed: {removed} lines")
    print(f"  Saved to: {output_path}")

    return len(cleaned)


if __name__ == '__main__':
    # Clean both files
    clean_file('arianna_facts.txt', 'arianna_facts_clean.txt')
    clean_file('arianna_asks.txt', 'arianna_asks_clean.txt')

    print("\nDone! Check *_clean.txt files")
