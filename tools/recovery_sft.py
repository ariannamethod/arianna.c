#!/usr/bin/env python3
"""Narrow Arianna SFT from base checkpoints, using the deployed causal forwards."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def norm(x):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)


class Body(nn.Module):
    """Tensor names and operations correspond directly to the two C forwards."""
    def __init__(self, state, voice):
        super().__init__()
        self.voice = voice
        self.keys = [k for k, v in state.items()
                     if v.is_floating_point() and not any(s in k for s in ("rope_", "causal_mask"))]
        self.values = nn.ParameterList([nn.Parameter(state[k].float()) for k in self.keys])
        self.index = dict(zip(self.keys, range(len(self.keys))))
        self.heads = 12 if voice == "resonance" else 10
        self.layers = 20

    def w(self, key):
        return self.values[self.index[key]]

    def linear(self, x, key):
        return F.linear(x, self.w(key))

    def exported_state(self):
        return {k: self.w(k).detach().cpu() for k in self.keys}

    def forward(self, ids, prompt_lengths):
        b, t = ids.shape
        h, d = self.heads, 64
        is_res = self.voice == "resonance"
        x = F.embedding(ids, self.w("tok_emb.weight" if is_res else "transformer.wte.weight"))
        e = x.shape[-1]
        positions = torch.arange(t, device=x.device)
        theta = 10000.0 if is_res else 100000.0
        angles = positions[:, None] * (theta ** (-torch.arange(0, d, 2, device=x.device).float() / d))
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        mask = positions[None, :] > positions[:, None]
        if not is_res:
            x = norm(x)
            gate = self.w("smear_lambda") * torch.sigmoid(self.linear(x[:, 1:, :24], "smear_gate.weight"))
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), 1)
        x0, backout = x, None

        def heads(y):
            return y.view(b, t, h, d).transpose(1, 2)

        def rope(y):
            if is_res:
                a, z = y[..., ::2], y[..., 1::2]
                return torch.stack((a*cos-z*sin, a*sin+z*cos), -1).flatten(-2)
            a, z = y[..., :d//2], y[..., d//2:]
            return torch.cat((a*cos+z*sin, z*cos-a*sin), -1)

        for i in range(self.layers):
            p = f"blocks.{i}." if is_res else f"transformer.h.{i}."
            if not is_res:
                x = self.w("resid_lambdas")[i] * x + self.w("x0_lambdas")[i] * x0
            xn = norm(x)
            if is_res:
                xn = xn * self.w(p+"norm1.weight")
                q, k, v = [heads(self.linear(xn, p+s+".weight")) for s in ("wq", "wk", "wv")]
            else:
                q, k, v = [heads(self.linear(xn, p+"attn."+s+".weight")) for s in ("c_q", "c_k", "c_v")]
            q, k = rope(q), rope(k)
            if not is_res:
                q, k = norm(q)*1.2, norm(k)*1.2
            content = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ap = p if is_res else p+"attn."
            mid = torch.einsum("bte,her->bhtr", xn, self.w(ap+"wr_a"))
            if not is_res:
                # C prefill shares the whole prompt sum; decode adds each new
                # token once. Future answer tokens never enter earlier queries.
                mid = mid.cumsum(2)
                gather = (prompt_lengths-1).view(b, 1, 1, 1).expand(b, h, 1, mid.shape[-1])
                full_prompt = mid.gather(2, gather)
                mid = torch.where((positions[None, :] < prompt_lengths[:, None])[:, None, :, None],
                                  full_prompt, mid)
            scores = torch.einsum("bhqr,hrk->bhqk", mid, self.w(ap+"wr_b")[:, :, :t]) / math.sqrt(d)
            attn = scores.masked_fill(mask, -torch.inf).softmax(-1)
            if is_res:
                rr = attn @ v
                gate = self.w(p+"gate").sigmoid()[None, :, None, None]
                blended = gate*content+(1-gate)*rr
                projection = p+"wo.weight"
            else:
                rr = attn @ heads(self.linear(xn, ap+"wvr.weight"))
                echo = self.linear(xn, ap+"wj.weight")
                echo_back = echo @ self.w(ap+"wj.weight")
                score = 15*torch.tanh((xn*echo_back).sum(-1)/math.sqrt(e)/15)
                ea = (score[:, :, None]*score[:, None, :]).masked_fill(mask, -torch.inf).softmax(-1)
                ech = ea[:, None] @ heads(echo)
                gate = self.w(ap+"gate").softmax(-1)
                blended = (gate[:, 0][None, :, None, None]*content +
                           gate[:, 1][None, :, None, None]*rr +
                           gate[:, 2][None, :, None, None]*ech)
                projection = ap+"c_proj.weight"
            x = x + self.linear(blended.transpose(1, 2).reshape(b, t, e), projection)
            xn = norm(x)
            if is_res:
                xn = xn*self.w(p+"norm2.weight")
                mg, mu, md = (p+s+".weight" for s in ("mlp_gate", "mlp_up", "mlp_down"))
            else:
                mg, mu, md = (p+"mlp."+s+".weight" for s in ("w_gate", "w_up", "w_down"))
            x = x + self.linear(F.silu(self.linear(xn, mg))*self.linear(xn, mu), md)
            if not is_res and i == self.layers//2:
                backout = x
        if not is_res:
            x = x - self.w("backout_lambda")*backout
        x = norm(x)
        if is_res:
            return self.linear(x*self.w("norm_f.weight"), "out_head.weight")
        logits = self.linear(x, "lm_head.weight")
        return 15*torch.tanh(logits/15)


def load_base(root, voice):
    path = (root/"resonance/checkpoints/final.pt" if voice == "resonance"
            else root/"janus4/janus/janus_177m_v4_base_22442.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if voice == "janus":
        for key in ("transformer.wte.weight", "lm_head.weight"):
            w = state[key]
            if w.shape[0] < 32768:
                state[key] = torch.cat((w, torch.randn(32768-w.shape[0], w.shape[1])*0.02))
    return Body(state, voice), path


def tokenizer(root, voice):
    header = root/("resonance_bpe_merges.h" if voice == "resonance" else "janus_v4_bpe_merges.h")
    if voice == "janus":
        from janus_tokenizer import JanusTokenizer
        return JanusTokenizer.from_header(header)
    text = header.read_text().split("[][2] = {", 1)[1]
    merges = [(int(a), int(b)) for a, b in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", text)]
    return ByteBPE(merges)


class ByteBPE:
    """The baked C merge order, including byte-fallback IDs 0..255."""
    def __init__(self, merges):
        self.rank = {}
        self.tokens = [bytes([i]) for i in range(256)]
        for i, (a, b) in enumerate(merges):
            self.rank.setdefault((a, b), 256+i)
            self.tokens.append(self.tokens[a]+self.tokens[b])

    def encode(self, text):
        ids = list(text.encode())
        while len(ids) > 1:
            candidates = [(self.rank[p], p) for p in zip(ids, ids[1:]) if p in self.rank]
            if not candidates:
                break
            token, pair = min(candidates)
            result, i = [], 0
            while i < len(ids):
                if i+1 < len(ids) and (ids[i], ids[i+1]) == pair:
                    result.append(token)
                    i += 2
                else:
                    result.append(ids[i])
                    i += 1
            ids = result
        return ids

    def decode(self, ids):
        return b"".join(self.tokens[i] for i in ids if 0 <= i < len(self.tokens))


def wire(text):
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def render(row, tok, voice, context=512):
    prompt = tok.encode(wire(row["prompt"]))
    answer = tok.encode((" " if voice == "resonance" else "") + row["answer"])
    if voice == "janus":
        prompt = [32759, 32760]+prompt+[32761, 32762]
        answer += [32763]
    if not prompt or len(prompt)+len(answer) > context:
        raise ValueError(f"sample exceeds the narrow SFT window: {row['prompt']}")
    return prompt+answer, [-100]*len(prompt)+answer, len(prompt)


def batch(rows, device):
    width = max(len(x[0]) for x in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    for i, (x, y, _) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x, device=device)
        labels[i, :len(y)] = torch.tensor(y, device=device)
    return ids, labels, torch.tensor([x[2] for x in rows], device=device)


def injection_bias(body, prompt_ids):
    # Same embedding EMA, cosine normalization, alpha and within-turn decay as C.
    emb = body.w("tok_emb.weight").detach().float()
    compass = torch.zeros(emb.shape[1], device=emb.device)
    for token in prompt_ids[:512]:
        compass = .9*compass+.1*emb[token]
    bias = F.normalize(emb, dim=-1) @ F.normalize(compass, dim=0)
    return bias/bias.abs().max().clamp_min(1e-12)


def logits_with_injection(body, ids, lengths):
    logits = body(ids, lengths)
    if body.voice == "resonance":
        for i, p in enumerate(lengths.tolist()):
            bias = injection_bias(body, ids[i, :p])
            steps = torch.arange(logits.shape[1]-p+1, device=ids.device)
            logits[i, p-1:] = logits[i, p-1:] + (5*torch.exp(-.15*steps))[:, None]*bias
    return logits


@torch.no_grad()
def generate(body, tok, prompt, count=64, seed=71):
    ids = tok.encode(wire(prompt))
    if body.voice == "janus":
        ids = [32759, 32760]+ids+[32761, 32762]
    p = len(ids)
    inp = torch.tensor([ids], device="cuda")
    plen = torch.tensor([p], device="cuda")
    rng = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for step in range(count):
        logits = logits_with_injection(body, inp, plen)[0, -1].float()
        for token in set(inp[0, -64:].tolist()):
            logits[token] = logits[token]/1.4 if logits[token] > 0 else logits[token]*1.4
        # Match the live temperature range; retain the model's sampled text.
        values, ix = (logits/.7).sort(descending=True)
        probs = values.softmax(-1)
        remove = probs.cumsum(-1)-probs > .9
        probs[remove] = 0
        nxt = ix[torch.multinomial(probs, 1, generator=rng)].item()
        if body.voice == "janus" and nxt == 32763:
            break
        out.append(nxt)
        inp = torch.cat((inp, torch.tensor([[nxt]], device="cuda")), 1)
    return tok.decode(out).decode("utf-8", errors="replace")


def export_gguf(body, path, lineage):
    # Existing C loaders consume these tensor names and F16 payloads directly.
    res = body.voice == "resonance"
    arch = "resonance" if res else "janus"
    e, h, r, m, v, ctx = (768, 12, 48, 2048, 16384, 2048) if res else (640, 10, 64, 1664, 32768, 1024)
    kv = {"general.architecture": arch, "general.name": f"Arianna {arch} recovery 2026-10-20",
          "general.source": lineage, f"{arch}.embedding_length": e, f"{arch}.vocab_size": v,
          f"{arch}.attention.head_count": h, f"{arch}.attention.head_dim": 64,
          f"{arch}.block_count": 20, f"{arch}.feed_forward_length": m,
          f"{arch}.context_length": ctx, f"{arch}.rrpram_rank" if res else f"{arch}.rrpram.rank": r}
    if res:
        kv["resonance.rrpram.scale"] = 1/math.sqrt(64)
    def string(s):
        raw = s.encode()
        return struct.pack("<Q", len(raw))+raw
    header = b"GGUF"+struct.pack("<IQQ", 3, len(body.keys), len(kv))
    for key, value in kv.items():
        tag = 8 if isinstance(value, str) else (6 if isinstance(value, float) else 4)
        header += string(key)+struct.pack("<I", tag)
        header += string(value) if tag == 8 else struct.pack("<f" if tag == 6 else "<I", value)
    payloads, offset = [], 0
    for key in body.keys:
        value = body.w(key).detach().cpu().numpy().astype("<f2")
        name = key
        if res:
            name = name.replace("tok_emb.weight", "tok_emb")
            name = re.sub(r"blocks\.(\d+)\.", r"transformer.h.\1.", name)
            name = re.sub(r"(transformer\.h\.\d+\.)(wr_a|wr_b|gate|wq|wk|wv|wo)", r"\1attn.\2", name)
            name = name.replace("mlp_gate", "mlp.w_gate").replace("mlp_up", "mlp.w_up").replace("mlp_down", "mlp.w_down")
        shape = value.shape or (1,)
        header += string(name)+struct.pack("<I", len(shape))
        header += b"".join(struct.pack("<Q", n) for n in reversed(shape))
        header += struct.pack("<IQ", 1, offset)
        data = value.tobytes()
        data += b"\0"*((-len(data)) % 32)
        payloads.append(data)
        offset += len(data)
    with open(path, "wb") as out:
        out.write(header)
        out.write(b"\0"*((-len(header)) % 32))
        for data in payloads:
            out.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--voice", choices=["janus", "resonance"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(3407)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    body, base = load_base(args.root, args.voice)
    body.cuda()
    tok = tokenizer(args.root, args.voice)
    rows = [json.loads(x) for x in args.data.read_text().splitlines()]
    train = [render(x, tok, args.voice, args.context) for x in rows if x["split"] == "train"]
    val = [render(x, tok, args.voice, args.context) for x in rows if x["split"] == "val"]
    out = args.out or args.root/args.voice/"recovery"
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"voice": args.voice, "train": len(train), "val": len(val), "base": str(base),
                      "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()}), flush=True)
    opt = torch.optim.AdamW(body.parameters(), lr=args.lr, weight_decay=0)
    def loss_on(samples):
        ids, labels, lengths = batch(samples, "cuda")
        logits = logits_with_injection(body, ids, lengths)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
    def validation():
        body.eval()
        with torch.no_grad():
            losses = [loss_on(val[i:i+2]).item() for i in range(0, len(val), 2)]
        body.train()
        return sum(losses)/len(losses)
    baseline = validation()
    print(json.dumps({"epoch": 0, "val": baseline}), flush=True)
    heldout = [
        "There is a yellow umbrella beside the locked gate. What catches your attention?",
        "The rain stopped. I opened the window and heard a bird. Stay with this moment.",
        "We have moved our work to Polygon. What would you like to carry into the next conversation?",
    ]
    start = time.monotonic()
    best = baseline
    for epoch in range(1, args.epochs+1):
        order = list(range(len(train)))
        random.Random(3407+epoch).shuffle(order)
        losses = []
        for step, i in enumerate(range(0, len(order), args.batch_size)):
            opt.zero_grad(set_to_none=True)
            loss = loss_on([train[j] for j in order[i:i+args.batch_size]])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body.parameters(), 1)
            opt.step()
            losses.append(loss.item())
            if step % 10 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": loss.item(),
                                  "elapsed": round(time.monotonic()-start, 1)}), flush=True)
        vl = validation()
        print(json.dumps({"epoch": epoch, "train_loss": sum(losses)/len(losses), "val": vl}), flush=True)
        if vl < best:
            best = vl
            torch.save(body.exported_state(), out/"best.pt")
            export_gguf(body, out/"best.gguf", str(base))
        body.eval()
        for prompt in heldout:
            print(json.dumps({"epoch": epoch, "prompt": prompt, "answer": generate(body, tok, prompt)}, ensure_ascii=False), flush=True)
        body.train()
    (out/"result.json").write_text(json.dumps({"base_val": baseline, "best_val": best,
                                              "seconds": time.monotonic()-start}, indent=2))


if __name__ == "__main__":
    main()
