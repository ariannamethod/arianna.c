package main

// doe.go — the subconscious's notorch-native engine (the LoRA parliament). The
// nano (Arianna's 88M body) runs through the vendored doe.c (doe_field) so the
// living parliament can seat on it (#3). doe is a REPL: the seed is piped on stdin,
// the dream is read from stdout. doe.c is NOT a replacement for the nano — it is
// the engine + parliament over the same body. doeAlpha gates the parliament:
// "0" = dormant (plain notorch-native forward, the step-1 bridge), "0.1" = it seats.

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// the doe parliament persists its learned experts as mycelium spores (doe.c:2500 —
// doe_mycelium/spore_<fingerprint>_s<step>.bin). Cap the dir so it can't grow without
// bound across sessions; the parliament loads the HIGHEST-step spore, so the survivors
// always include the one it will pick.
const (
	myceliumDir  = "doe_mycelium"
	myceliumKeep = 8
)

// pruneMycelium keeps the myceliumKeep highest-step spores PER FINGERPRINT and deletes
// the rest. doe.c loads the highest-step spore for the current host fingerprint only
// (doe.c:2547), so the cap is per-fingerprint — a different host's spores can never
// crowd out THIS host's load target. Best-effort: a missing dir / a name that isn't a
// canonical spore_<16hex>_s<step>.bin (e.g. malformed, non-hex fingerprint, negative
// step) is left untouched. Safe to call before the daemon loads.
func pruneMycelium(dir string, keep int) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type spore struct {
		name string
		step int
	}
	byFP := map[string][]spore{}
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() || !strings.HasPrefix(n, "spore_") || !strings.HasSuffix(n, ".bin") {
			continue
		}
		mid := strings.TrimSuffix(strings.TrimPrefix(n, "spore_"), ".bin") // <16hex>_s<step>
		i := strings.LastIndex(mid, "_s")
		if i <= 0 { // need a non-empty fingerprint before "_s"
			continue
		}
		fp := mid[:i]
		if !isHex16(fp) {
			continue
		}
		step, err := strconv.Atoi(mid[i+2:])
		if err != nil || step < 0 {
			continue
		}
		byFP[fp] = append(byFP[fp], spore{n, step})
	}
	for _, spores := range byFP {
		if len(spores) <= keep {
			continue
		}
		sort.Slice(spores, func(a, b int) bool { return spores[a].step > spores[b].step }) // highest step first
		for _, s := range spores[keep:] {
			_ = os.Remove(filepath.Join(dir, s.name))
		}
	}
}

// isHex16 reports whether s is exactly 16 hex digits (doe's `%016llx` fingerprint).
func isHex16(s string) bool {
	if len(s) != 16 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// doe loads the model fresh + profiles + may generate up to its 200-token REPL cap,
// so it is slower than the nanollama one-shot — give it headroom.
const doeDreamTimeout = 45 * time.Second

// maxDoeSeedBytes keeps the piped seed under doe's REPL line buffer (input[1024])
// so a long KK fragment can't split into a second prompt.
const maxDoeSeedBytes = 1000

// doeStatusCmd is doe's read-only REPL command (doe.c:3470): it prints a `[field]
// step=…` line and continues WITHOUT generating, resetting the KV cache, or touching
// the experts. Sent after each seed, that reply is the end-of-generation sentinel
// (doe prints no <END> frame).
const doeStatusCmd = "status"

// doePrimeTimeout bounds the daemon's first load (the 169.8MB model + the sonar
// profile + the mycelium spore) — longer than a warm dream, which pays none of that.
const doePrimeTimeout = 90 * time.Second

// doeDaemon is a persistent doe_field REPL. The host model + the parliament's
// mycelium spore load ONCE; each dream is one prompt over the same loaded body, so
// the field, the experts, and the prophecy-debt evolve continuously across the
// session's dreams (doe's native REPL mode) instead of resetting per dream — and the
// 169.8MB model reload + sonar profile + spore save a one-shot spawn pays EVERY dream
// is gone. Talked to over stdin/stdout under the nano's mutex (one generation at a
// time, matching the single stream).
type doeDaemon struct {
	cmd    *exec.Cmd
	in     io.WriteCloser
	out    *bufio.Scanner
	dead   bool      // set when the REPL stops responding (stream ends before the sentinel)
	reaped sync.Once // cmd.Wait() runs exactly once — across a mid-session death and close()
}

// isDoeStatusSentinel matches doe's status reply (doe.c:3471 — `[field] step=%d
// debt=%.3f entropy=%.3f resonance=%.3f emergence=%.3f`). doe prints it right after
// its `> ` prompt (no newline, no echo), so the real line is `> [field] step=…`: strip
// the prompt + spaces and require the line to BEGIN with the status prefix AND carry
// the full signature — a dream line that merely mentions these words elsewhere (no
// leading prompt, or the prefix mid-sentence) is not mistaken for the end-of-generation
// frame.
func isDoeStatusSentinel(line string) bool {
	t := strings.TrimLeft(line, "> \t")
	return strings.HasPrefix(t, "[field] step=") &&
		strings.Contains(t, "debt=") &&
		strings.Contains(t, "entropy=") &&
		strings.Contains(t, "resonance=") &&
		strings.Contains(t, "emergence=")
}

// reap waits the process exactly once (idempotent across the death paths + close()).
func (d *doeDaemon) reap() { d.reaped.Do(func() { _ = d.cmd.Wait() }) }

// neutralizeDoeSeed defends a seed that is EXACTLY a doe REPL command (status/quit/
// exit): a leading space defeats doe's strcmp (doe.c:3469) so it is dreamt on, not
// executed (quit/exit would kill the REPL, status would desync the sentinel frame).
// Harmless inside the chat template.
func neutralizeDoeSeed(seed string) string {
	switch seed {
	case doeStatusCmd, "quit", "exit":
		return " " + seed
	}
	return seed
}

// startDoeDaemon launches doe_field as a persistent REPL and primes it (drains the
// load banner up to the first status sentinel, confirming the model is loaded and the
// REPL is ready for prompts). Returns nil on ANY failure — the caller then falls back
// to the one-shot spawn (graceful degradation, zero regression). train is fixed at
// launch (a process-level flag), mirroring the one-shot's --train.
func startDoeDaemon(bin, gguf, alpha, train string) *doeDaemon {
	if train == "" {
		train = "0" // proven default: static experts (no online weight drift)
	}
	cmd := exec.Command(bin, "--model", gguf, "--lora-alpha", alpha, "--train", train)
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil
	}
	cmd.Stderr = nil // discard [timing]/[profile] (they go to stderr)
	if err := cmd.Start(); err != nil {
		return nil
	}
	sc := bufio.NewScanner(outPipe)
	sc.Buffer(make([]byte, 1<<20), 1<<20) // tolerate the long banner / dream lines
	d := &doeDaemon{cmd: cmd, in: in, out: sc}
	// prime: send the sentinel alone (an empty seed line is skipped by doe.c:3468) and
	// drain the banner up to its reply, bounded so a model that never loads can't wedge
	// startup. On failure, reap the process and degrade to one-shot.
	ctx, cancel := context.WithTimeout(context.Background(), doePrimeTimeout)
	defer cancel()
	if _, ok := d.exchange(ctx, ""); !ok {
		d.kill()
		d.reap()
		return nil
	}
	return d
}

// generate sends ONE seed prompt and returns doe's raw stdout up to the status
// sentinel (banner/footer stripping is parseDoeDream's job). ok=false (and the daemon
// is marked dead) on timeout or if the stream ends before the sentinel — the caller
// then falls back to a one-shot spawn for this dream.
func (d *doeDaemon) generate(ctx context.Context, seed string) (string, bool) {
	if d == nil || d.dead {
		return "", false
	}
	return d.exchange(ctx, seed)
}

// exchange writes the seed followed by the read-only status sentinel, then reads
// stdout until the `[field] step=` reply. doe reads the two as REPL lines: it
// generates on the seed (an empty line is harmlessly skipped — the prime case), then
// status prints the sentinel. doe's `printf("> ")` carries no newline, so the
// sentinel line arrives as `> [field] step=…` — matched by Contains, not HasPrefix.
// The read runs in a goroutine under a deadline: a wedged generation would otherwise
// hold the nano's mutex forever, so on ctx-cancel the process is killed (unblocking
// the read with EOF) and marked dead.
func (d *doeDaemon) exchange(ctx context.Context, seed string) (string, bool) {
	if _, err := fmt.Fprintf(d.in, "%s\n%s\n", seed, doeStatusCmd); err != nil {
		d.dead = true
		d.reap() // the pipe broke — the process is gone; reap it now, not at shutdown
		return "", false
	}
	type reply struct {
		text string
		ok   bool
	}
	ch := make(chan reply, 1)
	go func() {
		var b strings.Builder
		ok := false
		for d.out.Scan() {
			line := d.out.Text()
			if isDoeStatusSentinel(line) { // the generation fully completed
				ok = true
				break
			}
			b.WriteString(line)
			b.WriteByte('\n')
		}
		ch <- reply{b.String(), ok}
	}()
	select {
	case r := <-ch:
		if !r.ok {
			d.dead = true // stream ended before the sentinel — the daemon is gone
			d.reap()      // it exited on its own (EOF); reap it now
			return "", false
		}
		return r.text, true
	case <-ctx.Done():
		d.dead = true
		d.kill() // closes the pipe → the read goroutine hits EOF and exits (no leak)
		<-ch
		d.reap() // reap the killed process now, not at shutdown
		return "", false
	}
}

// close stops the daemon: closing stdin makes doe's fgets return NULL → it breaks the
// REPL, saves its mycelium spore, and exits. Bounded so a wedged spore-save can't hang
// teardown. nil-safe (the daemon may never have started).
func (d *doeDaemon) close() {
	if d == nil {
		return
	}
	d.in.Close()
	done := make(chan struct{})
	go func() { d.reap(); close(done) }() // reap is a no-op if a death path already waited
	select {
	case <-done:
	case <-time.After(15 * time.Second):
		d.kill()
		<-done
	}
}

// kill terminates the process (Kill on an exited process errors — ignored).
func (d *doeDaemon) kill() {
	if d.cmd != nil && d.cmd.Process != nil {
		_ = d.cmd.Process.Kill()
	}
}

// doeDream runs the seed through doe_field (the nano body + the parliament at
// doeAlpha) and returns the dream murmur. "" on failure / empty seed. The seed is
// fed on stdin (doe's REPL has no --prompt); stderr (timing) is discarded.
func (n *nano) doeDream(parent context.Context, seed string) string {
	// collapse to ONE line — doe's REPL is line-oriented, so embedded newlines in a
	// KK fragment would become multiple prompts (extra generations, only the first
	// line dreamt on). One line = one dream.
	seed = strings.Join(strings.Fields(seed), " ")
	if seed == "" {
		return ""
	}
	// cap below doe's fgets line buffer (input[1024]) — a longer KK fragment would
	// split into a second REPL prompt. Trim at a word boundary so no UTF-8 rune is
	// cut (spaces are ASCII, safe to slice at).
	if len(seed) > maxDoeSeedBytes {
		cut := maxDoeSeedBytes
		if sp := strings.LastIndexByte(seed[:cut], ' '); sp > 0 {
			cut = sp
		}
		// ToValidUTF8 drops a multibyte rune split by the hard byte cut (when no
		// space was found before the cap — rare for word cues, but kept correct).
		seed = strings.ToValidUTF8(seed[:cut], "")
	}
	// a seed that is EXACTLY a doe REPL command (status/quit/exit) would be executed
	// instead of dreamt on — guards both the daemon and the one-shot path below.
	seed = neutralizeDoeSeed(seed)
	// ONE overall deadline for the whole dream (daemon + any one-shot fallback), so the
	// worst case stays a single doeDreamTimeout and stop()'s join budget provably holds.
	// The persistent daemon (the parliament stays awake: model + spore loaded once, the
	// field evolving across dreams) is tried first. On a FAST failure (down / EOF — the
	// budget is still left) ctx.Err() is nil, so the one-shot fallback runs on the
	// remaining time. On a daemon WEDGE the shared ctx is already fired, so the fallback
	// is correctly suppressed — a same-input one-shot won't unstick a wedge, and this
	// keeps the total bounded. /quit (parent cancel) suppresses it too. The daemon is
	// marked dead on any failure, so subsequent dreams go straight to one-shot.
	ctx, cancel := context.WithTimeout(parent, doeDreamTimeout)
	defer cancel()
	if n.doeD != nil && !n.doeD.dead {
		if raw, ok := n.doeD.generate(ctx, seed); ok {
			return parseDoeDream(raw)
		}
	}
	if ctx.Err() != nil { // budget spent (daemon wedge) or parent cancelled — don't spawn
		return ""
	}
	train := n.doeTrain
	if train == "" {
		train = "0" // proven default: static experts (no online weight drift)
	}
	cmd := exec.CommandContext(ctx, n.doeBin, "--model", n.gguf, "--lora-alpha", n.doeAlpha, "--train", train)
	cmd.Stdin = strings.NewReader(seed + "\n")
	out, err := cmd.Output() // stdout only; a hung doe is killed at the deadline
	if err != nil {
		return ""
	}
	return parseDoeDream(string(out))
}

// parseDoeDream extracts the dream from doe's REPL stdout. doe prints a banner +
// index logs (`[identity]`/`[host]`/`[sonar]`/`[mycelium]`/`[doe] …` and per-layer
// `  L#: health=…` lines), then the prompt `> ` immediately followed by the
// streamed dream on the same line, then a `  [life] …` footer. The dream is the
// first `>`-line that carries real text (not a `> [mycelium]`/`> [doe]` log line).
func parseDoeDream(out string) string {
	var b strings.Builder
	capturing := false
	for _, line := range strings.Split(out, "\n") {
		t := strings.TrimSpace(line)
		if !capturing {
			if !strings.HasPrefix(t, ">") { // not the generation prompt yet
				continue
			}
			body := strings.TrimSpace(strings.TrimPrefix(t, ">"))
			if body == "" || strings.HasPrefix(body, "[") { // empty prompt / "> [mycelium]"/"> [doe]" log
				continue
			}
			capturing = true // the dream starts here (may be just a label like "Arianna:" — continue below)
			b.WriteString(body)
			b.WriteByte(' ')
			continue
		}
		// capturing the dream: stop at a footer / log / the next prompt
		if t == "" || strings.HasPrefix(t, "[") || strings.HasPrefix(t, ">") {
			break
		}
		b.WriteString(t)
		b.WriteByte(' ')
	}
	dream := strings.TrimSpace(b.String())
	if i := strings.Index(dream, "[life]"); i >= 0 { // a footer that leaked onto the line
		dream = dream[:i]
	}
	dream = stripLabel(strings.TrimSpace(dream)) // drop a leading SFT label (A:/Q:/Arianna:)
	// drop any byte-fallback / invalid UTF-8 the doe model emitted (e.g. a raw 0xFF):
	// doe is a separate binary, so the C voices' output guard doesn't cover its stdout.
	dream = strings.ToValidUTF8(dream, "")
	return cutSentence(strings.Join(strings.Fields(dream), " "))
}
