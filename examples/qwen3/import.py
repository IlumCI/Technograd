#!/usr/bin/env python3
# Import an open-weight Qwen3 checkpoint into Technograd's TGIR: the model
# becomes a plain .tg program (every operation explicit and editable) plus
# f32 weight files its file() params load. Dense transformer, expressed with
# existing ops only -- no batched attention op is needed because each of the
# 16 query heads is emitted as rank-2 matmuls and its o_proj contribution is
# summed (GQA: 8 KV heads, each shared by 2 query heads).
#
#   import.py MODEL.safetensors CONFIG.json OUTDIR [--seqlen T]
#
# Writes OUTDIR/qwen3.tg and OUTDIR/w/*.bin. Run with:
#   tgc run OUTDIR/qwen3.tg <comma-separated token ids>
import sys, os, json, struct, math
import numpy as np

def die(m): sys.stderr.write("import: %s\n" % m); sys.exit(1)

mp, cp, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
T = 16
for i, a in enumerate(sys.argv):
    if a == "--seqlen": T = int(sys.argv[i+1])
cfg = json.load(open(cp))
D   = cfg["hidden_size"]            # 1024
L   = cfg["num_hidden_layers"]      # 28
NQ  = cfg["num_attention_heads"]    # 16
NKV = cfg["num_key_value_heads"]    # 8
HD  = cfg["head_dim"]               # 128
FF  = cfg["intermediate_size"]      # 3072
V   = cfg["vocab_size"]             # 151936
EPS = cfg["rms_norm_eps"]
THETA = cfg["rope_theta"]           # 1e6
GRP = NQ // NKV
os.makedirs(os.path.join(outdir, "w"), exist_ok=True)

# ---- safetensors (bf16) ----
f = open(mp, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(hlen))
base = 8 + hlen
blob = np.memmap(mp, dtype=np.uint8, mode="r")
def tensor(name):
    e = hdr[name]; assert e["dtype"] == "BF16", e["dtype"]
    a, b = e["data_offsets"]
    raw = blob[base + a: base + b].view(np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(e["shape"])

def wbin(name, arr):
    arr.astype("<f4").tofile(os.path.join(outdir, "w", name + ".bin"))

# ---- weights: HF Linear is [out,in], y = x @ W^T, so store W^T = [in,out] ----
wbin("E", tensor("model.embed_tokens.weight"))                       # (V, D) for take()
wbin("LM", tensor("model.embed_tokens.weight").T.copy())             # (D, V) tied lm_head
wbin("lnf", tensor("model.norm.weight"))
for l in range(L):
    p = "model.layers.%d." % l
    wbin("ln1_%d" % l, tensor(p+"input_layernorm.weight"))
    wbin("ln2_%d" % l, tensor(p+"post_attention_layernorm.weight"))
    wbin("qn_%d" % l,  tensor(p+"self_attn.q_norm.weight"))          # (HD)
    wbin("kn_%d" % l,  tensor(p+"self_attn.k_norm.weight"))
    Wq = tensor(p+"self_attn.q_proj.weight")                         # (NQ*HD, D)
    Wk = tensor(p+"self_attn.k_proj.weight")                         # (NKV*HD, D)
    Wv = tensor(p+"self_attn.v_proj.weight")
    Wo = tensor(p+"self_attn.o_proj.weight")                         # (D, NQ*HD)
    for h in range(NQ):
        wbin("Wq_%d_%d" % (l, h), Wq[h*HD:(h+1)*HD, :].T.copy())     # (D, HD)
        wbin("Wo_%d_%d" % (l, h), Wo[:, h*HD:(h+1)*HD].T.copy())     # (HD, D)
    for j in range(NKV):
        wbin("Wk_%d_%d" % (l, j), Wk[j*HD:(j+1)*HD, :].T.copy())     # (D, HD)
        wbin("Wv_%d_%d" % (l, j), Wv[j*HD:(j+1)*HD, :].T.copy())
    wbin("Wg_%d" % l, tensor(p+"mlp.gate_proj.weight").T.copy())     # (D, FF)
    wbin("Wu_%d" % l, tensor(p+"mlp.up_proj.weight").T.copy())       # (D, FF)
    wbin("Wd_%d" % l, tensor(p+"mlp.down_proj.weight").T.copy())     # (FF, D)

# ---- RoPE tables (rotate-half / NeoX convention, exact to HF) ----
inv = THETA ** (-(np.arange(0, HD, 2) / HD))      # (HD/2)
ang = np.outer(np.arange(T), inv)                 # (T, HD/2)
cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)   # (T, HD)
sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=1)
wbin("cosT", cos); wbin("sinT", sin)
Rot = np.zeros((HD, HD), np.float32)              # (x @ Rot) = rotate_half(x)
for i in range(HD//2):
    Rot[i + HD//2, i] = -1.0
    Rot[i, i + HD//2] = 1.0
wbin("Rot", Rot)
mask = np.where(np.arange(T)[None, :] <= np.arange(T)[:, None], 0.0, -1e9).astype(np.float32)
wbin("mask", mask)
wbin("lastidx", np.array([T-1], np.float32))

# ---- emit the .tg program ----
o = []
def P(s): o.append(s)
P("# Qwen3-0.6B imported into TGIR by examples/qwen3/import.py. Do not edit by hand.")
P("# %d layers, %d query / %d KV heads x %d, SwiGLU %d, RoPE theta %g, vocab %d, seqlen %d."
  % (L, NQ, NKV, HD, FF, THETA, V, T))
P("model qwen3\n")
def par(name, shape):
    P('param %s : f32[%s] = file("w/%s.bin")' % (name, ", ".join(map(str, shape)), name))
par("E", (V, D)); par("LM", (D, V)); par("lnf", (D,))
par("cosT", (T, HD)); par("sinT", (T, HD)); par("Rot", (HD, HD))
par("mask", (T, T)); par("lastidx", (1,))
for l in range(L):
    par("ln1_%d" % l, (D,)); par("ln2_%d" % l, (D,))
    par("qn_%d" % l, (HD,)); par("kn_%d" % l, (HD,))
    for h in range(NQ): par("Wq_%d_%d" % (l, h), (D, HD)); par("Wo_%d_%d" % (l, h), (HD, D))
    for j in range(NKV): par("Wk_%d_%d" % (l, j), (D, HD)); par("Wv_%d_%d" % (l, j), (D, HD))
    par("Wg_%d" % l, (D, FF)); par("Wu_%d" % l, (D, FF)); par("Wd_%d" % l, (FF, D))
P("")
P("def rotary(x: f32[%d, %d]) -> f32[%d, %d]:" % (T, HD, T, HD))
P("    return x * cosT + (x @ Rot) * sinT\n")
scale = 1.0 / math.sqrt(HD)
P("def forward(ids: f32[%d]) -> f32[%d]:" % (T, V))
P("    h = take(E, ids)")
for l in range(L):
    P("    # layer %d" % l)
    P("    n = rmsnorm(h) * ln1_%d" % l)
    for j in range(NKV):
        P("    k%d = rotary(rmsnorm(n @ Wk_%d_%d) * kn_%d)" % (j, l, j, l))
        P("    v%d = n @ Wv_%d_%d" % (j, l, j))
    terms = []
    for hh in range(NQ):
        j = hh // GRP
        P("    q%d = rotary(rmsnorm(n @ Wq_%d_%d) * qn_%d)" % (hh, l, hh, l))
        P("    s%d = softmax((q%d @ transpose(k%d)) * %.9g + mask)" % (hh, hh, j, scale))
        P("    o%d = (s%d @ v%d) @ Wo_%d_%d" % (hh, hh, j, l, hh))
        terms.append("o%d" % hh)
    P("    h = h + (%s)" % " + ".join(terms))
    P("    n = rmsnorm(h) * ln2_%d" % l)
    P("    h = h + (silu(n @ Wg_%d) * (n @ Wu_%d)) @ Wd_%d" % (l, l, l))
P("    logits = (rmsnorm(h) * lnf) @ LM")
P("    return reshape(take(logits, lastidx), %d)" % V)
open(os.path.join(outdir, "qwen3.tg"), "w").write("\n".join(o) + "\n")
nf = len(os.listdir(os.path.join(outdir, "w")))
sys.stderr.write("wrote %s/qwen3.tg and %d weight files (seqlen %d)\n" % (outdir, nf, T))
