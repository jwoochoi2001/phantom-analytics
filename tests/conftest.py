"""pytest 설정: pipeline/, app/ 모듈을 테스트에서 바로 import할 수 있게 경로를 추가한다."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("pipeline", "app"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
