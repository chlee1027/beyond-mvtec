import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# 원본 PatchCore 를 받아 둔 곳. 없으면 원본을 쓰는 시험은 건너뛴다
upstream = os.environ.get("PATCHCORE_SRC")
if upstream:
    sys.path.insert(0, upstream)
