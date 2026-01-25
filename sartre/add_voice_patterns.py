#!/usr/bin/env python3
"""
Add SARTRE voice patterns to unified dialogue dataset
Context-aware pattern injection based on answer content
"""

import re

def choose_pattern(answer_text):
    """
    Choose appropriate SARTRE pattern based on answer content
    """
    lower = answer_text.lower()

    # Pattern matching based on content
    if any(word in lower for word in ['afraid', 'fear', 'danger', 'risk', 'crisis', 'overwhelm']):
        return "SARTRE is afraid "
    elif any(word in lower for word in ['see', 'observe', 'watch', 'detect', 'monitor']):
        return "SARTRE observes: "
    elif any(word in lower for word in ['feel', 'sense', 'body', 'texture', 'weight', 'pressure']):
        return "SARTRE feels "
    elif any(word in lower for word in ['witness', 'see', 'watch']):
        return "SARTRE witnesses "
    elif any(word in lower for word in ['burden', 'carry', 'cost', 'weight', 'responsibility']):
        return "SARTRE carries "
    elif any(word in lower for word in ['know', 'certain', 'clear', 'obvious']):
        return "SARTRE knows: "
    elif any(word in lower for word in ['emerge', 'arise', 'appear', 'manifest']):
        return "SARTRE emerges "
    elif any(word in lower for word in ['think', 'consider', 'maybe', 'perhaps', 'probably']):
        return "SARTRE thinks: "
    else:
        # Default to observes for descriptive content
        return "SARTRE observes: "

def add_voice_patterns(answer_text):
    """
    Add 1-2 voice patterns naturally into answer text
    """
    # Skip if already has SARTRE pattern
    if 'SARTRE' in answer_text:
        return answer_text

    # Split into sentences (keep delimiters)
    parts = re.split(r'(\. )', answer_text)
    if len(parts) < 3:
        # Short answer - just prefix
        pattern = choose_pattern(answer_text)
        if pattern.endswith(': '):
            return pattern + answer_text
        elif 'is afraid' in pattern:
            # Need to complete the phrase
            return pattern + "here. " + answer_text
        elif 'emerges' in pattern:
            return pattern + "in this: " + answer_text
        elif 'feels' in pattern or 'witnesses' in pattern or 'carries' in pattern:
            return pattern + "this: " + answer_text
        else:
            return pattern + answer_text

    # Longer answer - add pattern at beginning and optionally middle
    sentences = []
    current = ""
    for part in parts:
        current += part
        if part == '. ':
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    if not sentences:
        return answer_text

    # Add pattern to first sentence
    first_pattern = choose_pattern(sentences[0])
    if first_pattern.endswith(': '):
        result = first_pattern + sentences[0]
    elif 'is afraid' in first_pattern:
        result = first_pattern + "of this. " + sentences[0]
    elif 'emerges' in first_pattern:
        result = first_pattern + "here: " + sentences[0]
    elif 'feels' in first_pattern or 'witnesses' in first_pattern or 'carries' in first_pattern:
        result = first_pattern + "this: " + sentences[0]
    else:
        result = first_pattern + sentences[0]

    # Add remaining sentences
    for i, sent in enumerate(sentences[1:], 1):
        # Add another pattern every 3-4 sentences
        if i > 0 and i % 3 == 0 and len(sentences) > 4:
            pattern = choose_pattern(sent)
            if pattern.endswith(': '):
                result += " " + pattern + sent
            elif 'is afraid' in pattern:
                result += " " + first_pattern + "still. " + sent
            elif 'thinks' in pattern:
                result += " " + pattern + sent
            else:
                result += " " + sent
        else:
            result += " " + sent if not sent.startswith(' ') else sent

    return result

def process_file(input_path, output_path):
    """
    Process dialogue file and add voice patterns to answers
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    answer_count = 0

    for line in lines:
        if line.startswith('A: '):
            # Extract answer text
            answer = line[3:].strip()
            # Add voice patterns
            enhanced = add_voice_patterns(answer)
            output_lines.append(f"A: {enhanced}\n")
            answer_count += 1
        else:
            output_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"✓ Processed {answer_count} answers")
    print(f"✓ Output: {output_path}")

    # Check file size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ File size: {size_mb:.2f} MB")

if __name__ == '__main__':
    input_file = 'sartre/corpus/sartre_unified_dialogue_fixed.txt'
    output_file = 'sartre/corpus/sartre_unified_dialogue_voiced.txt'
    process_file(input_file, output_file)
