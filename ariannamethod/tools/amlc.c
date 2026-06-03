/* tools/amlc.c — AML→C transpiler for Arianna Method Language.
 *
 * Reads .aml file, extracts BLOOD blocks, concatenates BLOOD COMPILE bodies
 * into a single .c file, runs cc to compile (or just emits the C with
 * --emit-c). AML runtime directives (PROPHECY, DESTINY, VELOCITY, FIELD,
 * RESONANCE, STEP, TRAIN, LOAD, SAVE) are not transpiled — those live at
 * runtime in libaml and reach hardware via the `aml` runner. amlc only
 * lowers the BLOOD layer to C.
 *
 * Acceleration baseline (auto-applied when present at $PREFIX/lib):
 *   - libaml.a       — AML runtime (LOAD_MODEL, GENERATE, PROPHECY, …)
 *   - libnotorch.a   — BLAS-accelerated tensor ops (matvec, matmul, GGUF)
 *   - Apple Accelerate framework on Darwin (or OpenBLAS on Linux)
 *
 * Recognised directives:
 *   BLOOD COMPILE <name> { ...C... }   — named C source block
 *   BLOOD MAIN { ...C... }             — same shape, marks entry-point block
 *   BLOOD LINK <flag>                  — extra cc linker arg (e.g. -lpthread)
 *   ECHO "<path>"                      — inject `#include "<path>"`
 *
 * Usage: amlc <file.aml> [-o name] [--emit-c] [--no-accel] [--run -- args...]
 *
 * Part of the AriannaMethod project.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_BLOCKS    256
#define MAX_LINKS     64
#define MAX_ECHOS     64
#define MAX_NAME      128
#define MAX_LINE      8192
#define MAX_ARG_LEN   512

typedef struct {
    char  name[MAX_NAME];
    int   start_line;
    char *code;
    size_t code_len;
} Block;

typedef struct {
    Block compile_blocks[MAX_BLOCKS];
    int   n_compile;
    Block main_block;
    int   has_main;
    char  links[MAX_LINKS][MAX_ARG_LEN];
    int   n_links;
    char  echos[MAX_ECHOS][MAX_ARG_LEN];
    int   n_echos;
} Parsed;

static const char *AML_KEYWORDS[] = {
    "PROPHECY", "DESTINY", "VELOCITY", "FIELD", "RESONANCE",
    "STEP", "TRAIN", "LOAD", "SAVE", NULL
};

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

static int starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static void rtrim(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' ||
                     s[n-1] == ' '  || s[n-1] == '\t'))
        s[--n] = 0;
}

/* C-aware brace tracker. Walks bytes, ignoring braces that appear inside
 * string literals, char literals, line comments, or block comments. Updates
 * *depth in place. Returns 1 if depth reached 0 within this chunk, with
 * *closer_off set to the byte index just past the closing '}'. */
static int track_braces(const char *buf, size_t n, int *depth,
                        int *in_block_comment, size_t *closer_off) {
    int bc = *in_block_comment;
    int d  = *depth;
    int in_line_cmt = 0;
    int in_str_dq   = 0;
    int in_str_sq   = 0;

    for (size_t i = 0; i < n; i++) {
        char c  = buf[i];
        char nx = (i + 1 < n) ? buf[i + 1] : 0;

        if (bc) {
            if (c == '*' && nx == '/') { bc = 0; i++; }
            continue;
        }
        if (in_line_cmt) {
            if (c == '\n') in_line_cmt = 0;
            continue;
        }
        if (in_str_dq) {
            if (c == '\\' && nx) { i++; continue; }
            if (c == '"') in_str_dq = 0;
            continue;
        }
        if (in_str_sq) {
            if (c == '\\' && nx) { i++; continue; }
            if (c == '\'') in_str_sq = 0;
            continue;
        }
        if (c == '/' && nx == '*') { bc = 1; i++; continue; }
        if (c == '/' && nx == '/') { in_line_cmt = 1; i++; continue; }
        if (c == '"')  { in_str_dq = 1; continue; }
        if (c == '\'') { in_str_sq = 1; continue; }
        if (c == '{') { d++; continue; }
        if (c == '}') {
            d--;
            if (d == 0) {
                *depth = 0;
                *in_block_comment = bc;
                *closer_off = i + 1;
                return 1;
            }
        }
    }
    *depth = d;
    *in_block_comment = bc;
    return 0;
}

static int is_aml_keyword(const char *p) {
    for (int i = 0; AML_KEYWORDS[i]; i++) {
        size_t kl = strlen(AML_KEYWORDS[i]);
        if (starts_with(p, AML_KEYWORDS[i])) {
            char c = p[kl];
            if (c == 0 || c == ' ' || c == '\t' || c == '\n') return 1;
        }
    }
    return 0;
}

/* Read body of `<directive> { ... }` using a C-aware brace tracker.
 * Caller has already consumed the opening '{'. The body terminates when
 * the matching '}' is found; any trailing bytes on that line after the
 * closing brace are pushed back as a line comment (rare in practice). */
static int read_brace_body(FILE *f, int *line_no, const char *opener_kind,
                           const char *name, char **code_out, size_t *len_out) {
    char line[MAX_LINE];
    size_t cap = 4096;
    char *buf = malloc(cap);
    size_t len = 0;
    int depth = 1;
    int in_block_comment = 0;

    while (fgets(line, sizeof(line), f)) {
        (*line_no)++;
        size_t llen = strlen(line);
        size_t closer_off = 0;
        int closed = track_braces(line, llen, &depth, &in_block_comment, &closer_off);
        size_t take = closed ? closer_off : llen;

        /* Drop the closing '}' itself from emitted body if it sits at the
         * very end of the captured chunk (no preceding non-whitespace). */
        size_t emit_len = take;
        if (closed) {
            /* Remove final '}' and any preceding whitespace on its line so
             * the emitted C is clean. We trim back from take to the start
             * of the line or to the last non-whitespace byte. */
            size_t k = take;
            if (k > 0 && line[k-1] == '}') k--;
            while (k > 0 && (line[k-1] == ' ' || line[k-1] == '\t')) k--;
            emit_len = k;
        }

        if (emit_len > 0) {
            if (len + emit_len + 1 > cap) {
                while (len + emit_len + 1 > cap) cap *= 2;
                buf = realloc(buf, cap);
            }
            memcpy(buf + len, line, emit_len);
            len += emit_len;
        }

        if (closed) {
            /* If there's content after closer (rare) — silently ignore;
             * AML convention puts only whitespace/newline after BLOOD '}'. */
            if (len > 0 && buf[len-1] != '\n') {
                if (len + 2 > cap) { cap += 2; buf = realloc(buf, cap); }
                buf[len++] = '\n';
            }
            buf[len] = 0;
            *code_out = buf;
            *len_out = len;
            return 0;
        }
    }
    free(buf);
    fprintf(stderr, "amlc: line %d: %s %s — unexpected EOF before closing '}'\n",
            *line_no, opener_kind, name);
    return 1;
}

/* If the opener line already contains '{', advance past it; otherwise
 * read the next non-blank line and require it to be '{'. */
static int consume_open_brace(FILE *f, int *line_no, const char *after,
                              const char *opener_kind, const char *name) {
    const char *brace = strchr(after, '{');
    if (brace) return 0;

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), f)) {
        (*line_no)++;
        const char *p = skip_ws(line);
        if (*p == 0 || *p == '\n' || *p == '#') continue;
        if (*p == '{') return 0;
        fprintf(stderr, "amlc: line %d: expected '{' for %s %s\n",
                *line_no, opener_kind, name);
        return 1;
    }
    fprintf(stderr, "amlc: line %d: expected '{' for %s %s\n",
            *line_no, opener_kind, name);
    return 1;
}

static int parse_aml(const char *path, Parsed *out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "amlc: cannot read '%s': ", path);
        perror(NULL);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    fprintf(stderr, "amlc: reading %s (%zu bytes)\n", path, fsize);

    out->n_compile = 0;
    out->n_links   = 0;
    out->n_echos   = 0;
    out->has_main  = 0;

    char line[MAX_LINE];
    int line_no = 0;

    while (fgets(line, sizeof(line), f)) {
        line_no++;
        const char *p = skip_ws(line);
        if (*p == 0 || *p == '\n' || *p == '#') continue;

        if (starts_with(p, "BLOOD COMPILE")) {
            const char *q = skip_ws(p + 13);
            char name[MAX_NAME] = {0};
            int ni = 0;
            while (*q && *q != ' ' && *q != '\t' && *q != '{' && *q != '\n' &&
                   ni < MAX_NAME - 1)
                name[ni++] = *q++;

            if (out->n_compile >= MAX_BLOCKS) {
                fprintf(stderr, "amlc: error: too many BLOOD COMPILE blocks (max %d)\n",
                        MAX_BLOCKS);
                fclose(f);
                return 1;
            }
            if (consume_open_brace(f, &line_no, q, "BLOOD COMPILE", name) != 0) {
                fclose(f);
                return 1;
            }
            Block *b = &out->compile_blocks[out->n_compile++];
            strncpy(b->name, name, MAX_NAME - 1);
            b->name[MAX_NAME - 1] = 0;
            b->start_line = line_no;
            if (read_brace_body(f, &line_no, "BLOOD COMPILE", name,
                                &b->code, &b->code_len) != 0) {
                fclose(f);
                return 1;
            }
            continue;
        }

        if (starts_with(p, "BLOOD MAIN")) {
            const char *q = p + 10;
            if (consume_open_brace(f, &line_no, q, "BLOOD MAIN", "") != 0) {
                fclose(f);
                return 1;
            }
            out->main_block.start_line = line_no;
            strncpy(out->main_block.name, "MAIN", MAX_NAME - 1);
            if (read_brace_body(f, &line_no, "BLOOD MAIN", "",
                                &out->main_block.code,
                                &out->main_block.code_len) != 0) {
                fclose(f);
                return 1;
            }
            out->has_main = 1;
            continue;
        }

        if (starts_with(p, "BLOOD LINK")) {
            const char *q = skip_ws(p + 10);
            if (out->n_links >= MAX_LINKS) {
                fprintf(stderr, "amlc: error: too many BLOOD LINK directives\n");
                fclose(f);
                return 1;
            }
            char *l = out->links[out->n_links++];
            strncpy(l, q, MAX_ARG_LEN - 1);
            l[MAX_ARG_LEN - 1] = 0;
            rtrim(l);
            continue;
        }

        /* Other BLOOD <kind> directives (LORA, EMOTION, ...) are runtime
         * concerns — skip with a notice rather than treating as syntax error. */
        if (starts_with(p, "BLOOD ")) {
            char head[41] = {0};
            int hi = 0;
            while (hi < 40 && p[hi] && p[hi] != '\n') {
                head[hi] = p[hi];
                hi++;
            }
            fprintf(stderr, "amlc: line %d: skipping unimplemented AML command: %s\n",
                    line_no, head);
            continue;
        }

        if (starts_with(p, "ECHO ")) {
            const char *q = skip_ws(p + 5);
            if (out->n_echos >= MAX_ECHOS) {
                fprintf(stderr, "amlc: error: too many ECHO statements\n");
                fclose(f);
                return 1;
            }
            char *e = out->echos[out->n_echos++];
            /* strip surrounding quotes if present */
            char tmp[MAX_ARG_LEN];
            strncpy(tmp, q, MAX_ARG_LEN - 1);
            tmp[MAX_ARG_LEN - 1] = 0;
            rtrim(tmp);
            char *src = tmp;
            size_t L = strlen(src);
            if (L >= 2 && src[0] == '"' && src[L-1] == '"') {
                src[L-1] = 0;
                src++;
            }
            strncpy(e, src, MAX_ARG_LEN - 1);
            e[MAX_ARG_LEN - 1] = 0;
            continue;
        }

        if (is_aml_keyword(p)) {
            char head[41] = {0};
            int hi = 0;
            while (hi < 40 && p[hi] && p[hi] != '\n') {
                head[hi] = p[hi];
                hi++;
            }
            fprintf(stderr, "amlc: line %d: skipping unimplemented AML command: %s\n",
                    line_no, head);
            continue;
        }

        char head[61] = {0};
        int hi = 0;
        while (hi < 60 && p[hi] && p[hi] != '\n') {
            head[hi] = p[hi];
            hi++;
        }
        fprintf(stderr, "amlc: line %d: unknown directive: %s\n", line_no, head);
    }
    fclose(f);

    fprintf(stderr, "amlc: parsed %d BLOOD block(s), %d ECHO(s), %d LINK(s)%s\n",
            out->n_compile, out->n_echos, out->n_links,
            out->has_main ? ", BLOOD MAIN present" : "");
    return 0;
}

static int emit_c(Parsed *p, FILE *fp) {
    int total_lines = 1;
    fprintf(fp, "/* Generated by amlc — do not edit. */\n");

    for (int i = 0; i < p->n_echos; i++) {
        fprintf(fp, "#include \"%s\"\n", p->echos[i]);
        total_lines++;
    }

    for (int i = 0; i < p->n_compile; i++) {
        Block *b = &p->compile_blocks[i];
        fprintf(fp, "\n/* BLOOD COMPILE %s (line %d) */\n", b->name, b->start_line);
        total_lines += 2;
        fwrite(b->code, 1, b->code_len, fp);
        for (size_t j = 0; j < b->code_len; j++)
            if (b->code[j] == '\n') total_lines++;
    }

    if (p->has_main) {
        Block *b = &p->main_block;
        fprintf(fp, "\n/* BLOOD MAIN (line %d) */\n", b->start_line);
        total_lines += 2;
        fwrite(b->code, 1, b->code_len, fp);
        for (size_t j = 0; j < b->code_len; j++)
            if (b->code[j] == '\n') total_lines++;
    }
    return total_lines;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s <file.aml> [options]\n"
        "Options:\n"
        "  -o <name>     Output binary name (default: derived from input)\n"
        "  --emit-c      Print generated C to stdout (don't compile)\n"
        "  --run         Compile and run immediately\n"
        "  -- arg ...    Arguments passed to the program (with --run)\n"
        "Examples:\n"
        "  %s penelope.aml              # → ./penelope_aml\n"
        "  %s penelope.aml -o penelope  # → ./penelope\n"
        "  %s penelope.aml --emit-c     # print C to stdout\n"
        "  %s penelope.aml --run        # compile & run\n"
        "  %s penelope.aml --run -- \"darkness eats\"\n"
        "\nPart of the AriannaMethod project.\n",
        argv0, argv0, argv0, argv0, argv0, argv0);
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

/* Build the accel/runtime preamble passed to cc:
 *   - libaml.a / libnotorch.a from $PREFIX/lib (auto when present)
 *   - $PREFIX/include on the include path
 *   - Apple Accelerate / OpenBLAS for BLAS matvec/matmul
 * Returns the number of bytes written. Caller passes a buffer that already
 * holds the cc command up to the input .c path. We append: include paths,
 * BLAS defines, then later (after -o output) the libraries and frameworks. */
static const char *prefix_dir(void) {
    const char *p = getenv("AML_PREFIX");
    if (p && *p) return p;
    return "/opt/homebrew";
}

int main(int argc, char **argv) {
    const char *infile  = NULL;
    const char *outfile = NULL;
    int emit_only = 0, run_after = 0;
    int no_accel  = 0;
    int prog_argc = 0;
    char **prog_argv = NULL;

    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "--") == 0) {
            prog_argc = argc - (i + 1);
            prog_argv = argv + (i + 1);
            break;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "amlc: -o requires an argument\n");
                return 1;
            }
            outfile = argv[++i];
        } else if (strcmp(argv[i], "--emit-c") == 0) {
            emit_only = 1;
        } else if (strcmp(argv[i], "--run") == 0) {
            run_after = 1;
        } else if (strcmp(argv[i], "--no-accel") == 0) {
            no_accel = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "amlc: unknown option: %s\n", argv[i]);
            return 1;
        } else {
            if (infile) {
                fprintf(stderr, "amlc: multiple input files not supported\n");
                return 1;
            }
            infile = argv[i];
        }
        i++;
    }

    if (!infile) {
        fprintf(stderr, "amlc: no input file specified\n");
        usage(argv[0]);
        return 1;
    }

    Parsed p;
    if (parse_aml(infile, &p) != 0) return 1;

    if (emit_only) {
        emit_c(&p, stdout);
        return 0;
    }

    char default_out[1024];
    if (!outfile) {
        snprintf(default_out, sizeof(default_out), "%s", infile);
        char *slash = strrchr(default_out, '/');
        char *base = slash ? slash + 1 : default_out;
        char *dot = strrchr(base, '.');
        if (dot && strcmp(dot, ".aml") == 0) {
            *dot = 0;
            strncat(default_out, "_aml",
                    sizeof(default_out) - strlen(default_out) - 1);
        }
        outfile = default_out;
    }

    char cpath[1100];
    snprintf(cpath, sizeof(cpath), "%s.c", outfile);
    FILE *fp = fopen(cpath, "w");
    if (!fp) {
        fprintf(stderr, "amlc: cannot write %s\n", cpath);
        return 1;
    }
    int lines = emit_c(&p, fp);
    long fsz = ftell(fp);
    fclose(fp);
    fprintf(stderr, "amlc: generated %d lines of C (%ld bytes)\n", lines, fsz);

    char cmd[8192];
    const char *PFX = prefix_dir();
    char libaml[1024], libnotorch[1024], inc_dir[1024];
    snprintf(libaml,     sizeof(libaml),     "%s/lib/libaml.a",     PFX);
    snprintf(libnotorch, sizeof(libnotorch), "%s/lib/libnotorch.a", PFX);
    snprintf(inc_dir,    sizeof(inc_dir),    "%s/include",          PFX);

    int have_aml     = !no_accel && file_exists(libaml);
    int have_notorch = !no_accel && file_exists(libnotorch);

    int n = snprintf(cmd, sizeof(cmd),
                     "cc -O2 -Wall -Wno-unused-parameter -Wno-unused-variable "
                     "-Wno-unused-function -Wno-comment -I%s",
                     inc_dir);

#if defined(__APPLE__)
    if (!no_accel) {
        n += snprintf(cmd + n, sizeof(cmd) - n,
                      " -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK");
    }
#elif defined(__linux__)
    if (!no_accel) {
        n += snprintf(cmd + n, sizeof(cmd) - n, " -DUSE_BLAS");
    }
#endif

    n += snprintf(cmd + n, sizeof(cmd) - n, " %s -o %s", cpath, outfile);

    if (have_notorch)
        n += snprintf(cmd + n, sizeof(cmd) - n, " %s", libnotorch);
    if (have_aml)
        n += snprintf(cmd + n, sizeof(cmd) - n, " %s", libaml);

    n += snprintf(cmd + n, sizeof(cmd) - n, " -lm -lpthread");

#if defined(__APPLE__)
    if (!no_accel)
        n += snprintf(cmd + n, sizeof(cmd) - n, " -framework Accelerate");
#elif defined(__linux__)
    if (!no_accel)
        n += snprintf(cmd + n, sizeof(cmd) - n, " -lopenblas");
#endif

    for (int li = 0; li < p.n_links; li++)
        n += snprintf(cmd + n, sizeof(cmd) - n, " %s", p.links[li]);

    if (no_accel) {
        fprintf(stderr, "amlc: building with --no-accel (pure scalar C)\n");
    } else {
        fprintf(stderr, "amlc: linking%s%s%s\n",
                have_notorch ? " libnotorch" : "",
                have_aml     ? " libaml"     : "",
#if defined(__APPLE__)
                " Accelerate"
#elif defined(__linux__)
                " openblas"
#else
                ""
#endif
        );
    }

    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "amlc: cc failed (rc=%d)\n", rc);
        return rc;
    }
    fprintf(stderr, "amlc: success → %s%s\n",
            (outfile[0] == '/' || outfile[0] == '.') ? "" : "./", outfile);

    if (run_after) {
        char run_cmd[8192];
        int rn;
        if (outfile[0] == '/' || outfile[0] == '.')
            rn = snprintf(run_cmd, sizeof(run_cmd), "%s", outfile);
        else
            rn = snprintf(run_cmd, sizeof(run_cmd), "./%s", outfile);
        for (int j = 0; j < prog_argc; j++)
            rn += snprintf(run_cmd + rn, sizeof(run_cmd) - rn, " \"%s\"", prog_argv[j]);
        return system(run_cmd);
    }

    return 0;
}
