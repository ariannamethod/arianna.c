#!/usr/bin/env python3
"""Compare the C encoder against the pinned base tiktoken encoder."""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import shlex
import shutil
import struct
import subprocess
import tempfile

from janus_tokenizer import JanusTokenizer, PATTERN, SPECIAL_TOKENS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-tokenizer", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--sanitize", action="store_true")
    args = ap.parse_args()
    tools = Path(__file__).resolve().parent
    tok = JanusTokenizer.from_header(tools / "janus_v4_bpe_merges.h")
    reference = tok.encoding
    if args.base_tokenizer:
        # The caller supplies the trusted, revision-pinned base tokenizer.
        with args.base_tokenizer.open("rb") as f:
            reference = pickle.load(f)
        assert reference._pat_str == PATTERN
        assert reference._special_tokens == SPECIAL_TOKENS
        assert reference._mergeable_ranks == {b: i for i, b in enumerate(tok.tokens)}

    cases = [
        "", "Hello, Arianna.", " Rain stopped.", "anna", "next",
        "line\nnext\tend\r", "I'm she's we'll I'M SHE'S WE'LL",
        "  leading and trailing  ", "<|assistant_end|>",
        "Я здесь. שלום! 中文 नमस्ते العربية Ελληνικά",
        "é e\u0301 😀 🔥", "\x00\x01A\x7f",
        "a" * 8192, " " * 8192,
    ]
    spaces = "\t\n\v\f\r \x1c\x1d\x1e\x1f\u0085\u00a0\u1680"
    spaces += "".join(chr(i) for i in range(0x2000, 0x200b))
    spaces += "\u2028\u2029\u202f\u205f\u3000"
    for space in spaces:
        cases += [space, space * 2, "a" + space + "b", space + "a",
                  "a" + space, space + "\n", "\r" + space + "word",
                  "a" + space + space + "b"]
    cases += [str(i) for i in range(100, 1000)]
    cases += ["foo123bar456", "...!???--", "function_name(arg)", "x\r\n  y\n"]
    if args.data:
        for line in args.data.read_text().splitlines():
            row = json.loads(line)
            cases += [row["prompt"], row["answer"]]
    rng = random.Random(1907)
    alphabet = "abxyABXYZ0123456789 '!\r\n\té\u0301אבЯ中😀\u0085\u00a0"
    cases += ["".join(rng.choices(alphabet, k=rng.randrange(1, 150))) for _ in range(300)]
    cases = list(dict.fromkeys(cases))

    config = shutil.which("pcre2-config")
    if not config and Path("/opt/homebrew/bin/pcre2-config").exists():
        config = "/opt/homebrew/bin/pcre2-config"
    cflags = os.environ.get("PCRE2_CFLAGS")
    libs = os.environ.get("PCRE2_LIBS")
    if cflags is None:
        cflags = subprocess.check_output([config, "--cflags"], text=True) if config else ""
    if libs is None:
        libs = subprocess.check_output([config, "--libs8"], text=True) if config else "-lpcre2-8"
    source = r'''
#include <stdio.h>
#include <stdint.h>
#include "janus_tokenizer.h"
#include "janus_v4_bpe_merges.h"
int main(void) {
    JanusTokenizer tok;
    if (janus_tokenizer_init(&tok, janus_v4_bpe_merges, JANUS_V4_BPE_MERGES)) return 2;
    uint32_t n, cap;
    while (fread(&n, sizeof(n), 1, stdin) == 1) {
        if (fread(&cap, sizeof(cap), 1, stdin) != 1) return 3;
        if (n > 2000000 || cap > 2000000) return 3;
        char *s = malloc(n ? n : 1);
        int *out = malloc((cap ? cap : 1) * sizeof(*out));
        if (!s || !out || fread(s, 1, n, stdin) != n) return 3;
        int count = janus_tokenizer_encode(&tok, s, (int)n, out, (int)cap);
        if (fwrite(&count, sizeof(count), 1, stdout) != 1) return 3;
        if (count > 0 && fwrite(out, sizeof(*out), count, stdout) != (size_t)count) return 3;
        fflush(stdout);
        free(s); free(out);
    }
    janus_tokenizer_free(&tok);
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="janus-tokenizer-test-") as scratch:
        scratch = Path(scratch)
        (scratch / "probe.c").write_text(source)
        flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]
        if args.sanitize:
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
        subprocess.run([os.environ.get("CC", "cc"), *flags, "-I" + str(tools),
                        *shlex.split(cflags), str(scratch / "probe.c"),
                        *shlex.split(libs), "-o", str(scratch / "probe")], check=True)
        proc = subprocess.Popen([str(scratch / "probe")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        def encode_c(raw, cap):
            proc.stdin.write(struct.pack("<II", len(raw), cap) + raw)
            proc.stdin.flush()
            size = proc.stdout.read(4)
            if len(size) != 4:
                raise RuntimeError("C tokenizer exited before replying")
            count = struct.unpack("<i", size)[0]
            return count if count < 0 else list(struct.unpack(
                "<" + "i" * count, proc.stdout.read(count * 4)))

        checks = 0
        try:
            for text in cases:
                expected = reference.encode_ordinary(text)
                assert tok.encode(text) == expected, ("Python", repr(text))
                assert tok.decode(expected) == text.encode(), ("decode", repr(text))
                actual = encode_c(text.encode(), max(1, len(text.encode())))
                assert actual == expected, ("C", repr(text), actual, expected)
                checks += 1
            for text in ["", "Hello, Arianna.", "Я здесь. שלום!", "a" * 8192, " " * 8192]:
                expected = reference.encode_ordinary(text)
                for cap in (0, 1, 2, 7, 4096):
                    assert encode_c(text.encode(), cap) == expected[:cap], ("capacity", cap)
                    checks += 1
            assert encode_c(b"\xff", 8) == -1
            assert encode_c(b"\xe2\x82", 8) == -1
            checks += 2
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)
            proc.stdout.close()
        assert proc.returncode == 0, proc.returncode
    print(json.dumps({"cases": len(cases), "checks": checks, "all_ids_equal": True,
                      "base_vocab": len(tok.tokens), "sanitizers": args.sanitize}))


if __name__ == "__main__":
    main()
