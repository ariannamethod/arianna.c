#!/usr/bin/env python3
"""
Extract lead sections (first paragraphs) from Simple Wikipedia dump.
Output: clean text file for training.
"""

import re
import sys
from xml.etree import ElementTree as ET

def clean_text(text):
    """Remove wiki markup, keep plain text."""
    # Remove nested templates {{...}} iteratively
    prev_len = -1
    while len(text) != prev_len:
        prev_len = len(text)
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)

    # Remove infobox-style lines: | field = value
    text = re.sub(r'\|[^|\n]*=[^|\n]*', '', text)

    # Remove leftover }} and {{
    text = re.sub(r'\}\}|\{\{', '', text)

    # Remove [[File:...]] and [[Image:...]]
    text = re.sub(r'\[\[(File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # Convert [[link|text]] to text, [[link]] to link
    text = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove refs <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/?>', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove '''bold''' and ''italic''
    text = re.sub(r"'''?", '', text)
    # Remove Category links
    text = re.sub(r'Category:[^\s\]]*', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_lead(text):
    """Extract first paragraph(s) before first section heading."""
    # Split by section headings (== ... ==)
    parts = re.split(r'\n==[^=]', text)
    lead = parts[0] if parts else text

    # Clean and get first substantial paragraph
    lead = clean_text(lead)

    # Skip if too short (redirects, stubs)
    if len(lead) < 100:
        return None

    # Truncate if too long (keep first ~500 chars, end at sentence)
    if len(lead) > 600:
        # Find sentence end
        for i in range(500, min(700, len(lead))):
            if lead[i] in '.!?':
                lead = lead[:i+1]
                break
        else:
            lead = lead[:600] + '...'

    return lead


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'simplewiki-latest.xml'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'simplewiki_leads.txt'

    print(f"Processing {input_file}...")

    count = 0
    skipped = 0
    total_chars = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        # Parse iteratively to handle large files
        for event, elem in ET.iterparse(input_file, events=['end']):
            if elem.tag.endswith('}page') or elem.tag == 'page':
                # Find title and text
                ns = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}ns')
                if ns is None:
                    ns = elem.find('.//ns')

                # Skip non-article pages (ns != 0)
                if ns is not None and ns.text != '0':
                    elem.clear()
                    continue

                title_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}title')
                if title_elem is None:
                    title_elem = elem.find('.//title')

                text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.11/}text')
                if text_elem is None:
                    text_elem = elem.find('.//text')

                if title_elem is not None and text_elem is not None and text_elem.text:
                    title = title_elem.text
                    text = text_elem.text

                    # Skip redirects
                    if text.lower().startswith('#redirect'):
                        skipped += 1
                        elem.clear()
                        continue

                    lead = extract_lead(text)
                    if lead:
                        # Write as: Title: Lead text\n\n
                        out.write(f"{title}: {lead}\n\n")
                        count += 1
                        total_chars += len(lead)

                        if count % 10000 == 0:
                            print(f"  Processed {count} articles, {total_chars/1e6:.1f}MB...")
                    else:
                        skipped += 1

                elem.clear()

    print(f"\nDone!")
    print(f"  Articles: {count}")
    print(f"  Skipped: {skipped}")
    print(f"  Total size: {total_chars/1e6:.1f} MB")
    print(f"  Output: {output_file}")


if __name__ == '__main__':
    main()
