#!/usr/bin/env python3
# Check the TGIR import against Hugging Face: feed identical token ids to both
# and compare the final-position logits. Needs torch + transformers (float32).
#   verify.py TGC MODELDIR [ids.csv]
import sys, subprocess, numpy as np, torch
from transformers import AutoModelForCausalLM
tgc, d = sys.argv[1], sys.argv[2]
ids = sys.argv[3] if len(sys.argv) > 3 else "151643,785,3974,13876,11,323,279,1879,374,264,1602,2244,1992,311,3161,13"
idl = [int(x) for x in ids.split(",")]
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
with torch.no_grad():
    hf = m(torch.tensor([idl])).logits[0, -1].numpy().astype("float32")
out = subprocess.run([tgc, "run", d + "/qwen3.tg", ids, "-j", "4"], capture_output=True, text=True)
tg = np.fromstring(out.stdout.strip(), sep=" ", dtype=np.float32)
assert tg.size == hf.size, "tgc produced %d logits, expected %d\n%s" % (tg.size, hf.size, out.stderr[-500:])
dif = np.abs(tg - hf)
print("tokens            :", idl)
print("HF  argmax / top5 :", int(hf.argmax()), hf.argsort()[-5:][::-1].tolist())
print("TGIR argmax / top5:", int(tg.argmax()), tg.argsort()[-5:][::-1].tolist())
print("max abs diff %.3g   mean %.3g   relative %.3g" % (dif.max(), dif.mean(), dif.max() / np.abs(hf).max()))
print("ARGMAX MATCH:", int(tg.argmax()) == int(hf.argmax()))
sys.exit(0 if int(tg.argmax()) == int(hf.argmax()) and dif.max() < 1e-2 else 1)
