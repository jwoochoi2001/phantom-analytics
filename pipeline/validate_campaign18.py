"""파이프라인이 캠페인 18 분석(analysis/campaign18_*.py)에서 검증한 값을 그대로
재현하는지 확인하는 회귀 테스트.

전처리 규칙·매칭 기준(caliper, cap 등)을 바꾸는 코드 수정을 할 때마다 이 스크립트를
실행해 캠페인 18 결과가 바뀌지 않았는지 확인한다. campaign_id, pre_days 외의 값은
전부 pipeline/prepare_data.py, pipeline/estimate_effect.py의 상수(제약사항)로
고정되어 있어야 한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import run  # noqa: E402

# analysis/campaign18_*.py에서 확인한 기준값
EXPECTED = {
    "n_treatment": 510,
    "n_control": 335,
    "n_common_support_treatment": 507,
    "n_common_support_control": 313,
    "n_matched_pairs": 232,
    "max_abs_smd": 0.08,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res = run(campaign_id=18, pre_days=586)

    print(f"\n{'='*78}\n캠페인 18 기준값 대조\n{'='*78}")
    failures = []
    for key, expected in EXPECTED.items():
        actual = res.get(key)
        ok = actual == expected
        print(f"  {key}: 기대값={expected}, 실제값={actual} → {'통과' if ok else '불일치!!'}")
        if not ok:
            failures.append(key)

    if failures:
        print(f"\n불일치 항목: {failures} — 처리 규칙 또는 매칭 기준이 캠페인 18 검증 시점과 달라졌습니다.")
        sys.exit(1)
    print("\n모든 기준값 일치 — 파이프라인이 캠페인 18 검증 결과를 그대로 재현합니다.")


if __name__ == "__main__":
    main()
