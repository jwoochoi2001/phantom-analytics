"""캠페인 18의 매칭 후 결과에 대한 Rosenbaum 민감도 분석.

pre_days 민감도 분석(campaign18_pre_days_sensitivity.py)에서 pre_days=30은 매칭 후에도
유의한 양의 효과(target_sales)를 보였고, pre_days=90(기본값)은 유의하지 않았다.
두 경우 모두에 대해 "관찰되지 않은 교란이 얼마나 커야 결론이 뒤집히는지"를
Rosenbaum bounds로 정량화한다.

출력: 터미널 표만 (진단 목적, 파일 저장 없음)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from run_pipeline import run  # noqa: E402
from sensitivity import breakdown_gamma, rosenbaum_bounds  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
PRIMARY = ["target_purchase", "target_sales", "target_quantity"]


def analyze(pre_days: int) -> None:
    print(f"\n{'='*78}\n캠페인 18, pre_days={pre_days} — Rosenbaum 민감도 분석\n{'='*78}")
    res = run(campaign_id=18, pre_days=pre_days)
    if res["status"] != "ok":
        print(f"status={res['status']}로 효과 추정치가 없어 민감도 분석을 건너뜀: {res.get('reason')}")
        return

    matched = pd.read_csv(OUTPUTS / "campaign_18" / "matched_data.csv")
    t = matched.loc[matched["group"] == "처치"].set_index("pair_id")
    c = matched.loc[matched["group"] == "대조"].set_index("pair_id")
    common_pairs = t.index.intersection(c.index)
    print(f"매칭 쌍 수: {len(common_pairs)}")

    for col in PRIMARY:
        diffs = (t.loc[common_pairs, col] - c.loc[common_pairs, col]).to_numpy()
        mean_diff = diffs.mean()
        bounds = rosenbaum_bounds(diffs)
        bg = breakdown_gamma(diffs)
        bg_str = f"Gamma={bg}" if bg is not None else "Gamma=5까지 안 무너짐(강건)"
        print(f"\n[{col}] 매칭후 평균차이={mean_diff:.3f}, 붕괴점: {bg_str}")
        print(bounds.to_string(index=False))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    analyze(30)   # 유의했던 경우
    analyze(90)   # 기본값(비유의)

    print(f"\n{'='*78}\n해석 메모\n{'='*78}")
    print("- 붕괴점 Gamma가 클수록(예: 3~4 이상) '숨겨진 교란요인이 처치 배정 승산을 그만큼")
    print("  왜곡시켜야만 결론이 바뀐다'는 뜻이라 결과가 더 강건하다.")
    print("- Gamma=1~1.5 근처에서 이미 무너지면, 아주 약한 숨겨진 교란만으로도 결론이 뒤집힐")
    print("  수 있다는 뜻이라 해석에 신중해야 한다.")
    print("- pre_days=90(비유의)에서는 애초에 매칭후 CI가 0을 포함하므로 '유의함을 뒤집는다'는")
    print("  개념 자체가 성립하지 않는다 — 이 경우 Rosenbaum bounds는 반대로 '아주 작은 숨겨진")
    print("  편향만으로도 유의한 결과가 나올 수 있는지'를 보여주는 참고 지표로만 해석한다.")


if __name__ == "__main__":
    main()
