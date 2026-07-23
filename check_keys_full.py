"""
check_keys_full.py

Dumps a de-duplicated "skeleton" of key names from a safetensors file --
numeric block/layer indices are replaced with N so repeating patterns
collapse into one line each. Used to compare a base checkpoint's naming
convention against a LoRA's naming convention.

Usage:
    python check_keys_full.py "C:\\path\\to\\file.safetensors"
"""

import re
import sys
from safetensors import safe_open

if len(sys.argv) != 2:
    print("Usage: python check_keys_full.py \"C:\\path\\to\\file.safetensors\"")
    sys.exit(1)

path = sys.argv[1]


def skeleton(key: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", key)


with safe_open(path, framework="pt") as f:
    keys = list(f.keys())

skeletons = {}
for k in keys:
    s = skeleton(k)
    skeletons.setdefault(s, []).append(k)

print(f"File: {path}")
print(f"total keys: {len(keys)}")
print(f"unique skeleton patterns: {len(skeletons)}")
print()
for s in sorted(skeletons.keys()):
    example = skeletons[s][0]
    count = len(skeletons[s])
    print(f"  [{count:>4}x]  {s}    (e.g. {example})")
