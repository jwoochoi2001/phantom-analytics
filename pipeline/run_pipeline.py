"""캠페인 입력형 전체 파이프라인 실행기.

사용법:
    python pipeline/run_pipeline.py --campaign_id 18 --pre_days 586

순서: 캠페인 정보조회 → 집단 구성 → 사전 변수 계산 → 결측치 처리 → 인코딩
      → 성향점수 → 공통영역 확인 → 매칭 → 균형과 효과 추정

표본 부족(집단 구성 단계), 공통영역 부족, 잔여 불균형 기준을 통과하지 못하면
효과를 확정하지 않고 status·reason을 outputs/campaign_{id}/results.json에 남긴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import prepare_data  # noqa: E402
from estimate_effect import estimate_effect, result  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

MIN_GROUP_SIZE = 30  # 집단 구성 단계 표본 부족 기준 (campaign_overlap_check.py에서 확정한 값)


def run(campaign_id: int, pre_days: int, output_dir: Path | None = None) -> dict:
    """output_dir: 결과 저장 위치(기본값 outputs/). 테스트 등에서 실제 캐시를 건드리지
    않으려면 임시 디렉터리를 넘긴다 — 그러지 않으면 outputs/campaign_{id}/의 캐시가
    테스트가 쓰는 조건(예: 회귀 테스트의 pre_days=586)으로 덮어써진다."""
    output_dir = output_dir or OUTPUTS
    print(f"{'='*78}\n캠페인 {campaign_id} 파이프라인 실행 (pre_days={pre_days})\n{'='*78}")

    df, meta = prepare_data(campaign_id, pre_days, output_dir=output_dir)

    if meta.n_treatment < MIN_GROUP_SIZE or meta.n_control < MIN_GROUP_SIZE:
        res = result(
            "insufficient_sample",
            f"집단 구성 단계 표본 부족: 처치 {meta.n_treatment}명, 대조 {meta.n_control}명 "
            f"(기준 각 {MIN_GROUP_SIZE}명 이상)",
            campaign_id=campaign_id, description=meta.description, pre_days=pre_days,
            start_day=meta.start_day, end_day=meta.end_day, overlap_ids=meta.overlap_ids,
            n_treatment=meta.n_treatment, n_control=meta.n_control,
        )
        print(f"\n[중단] {res['reason']}")
    else:
        res = estimate_effect(df, meta, output_dir=output_dir)
        if res["status"] == "ok":
            print(f"\n[완료] 효과 추정 통과 — 매칭 {res['n_matched_pairs']}쌍, 최대|SMD|={res['max_abs_smd']}")
        else:
            print(f"\n[중단/경고] status={res['status']} — {res['reason']}")

    out_dir = output_dir / f"campaign_{campaign_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장: results.json → {results_path}")

    return res


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="캠페인 효과 분석 파이프라인 (준비 + 추정)")
    parser.add_argument("--campaign_id", type=int, required=True)
    parser.add_argument("--pre_days", type=int, required=True)
    args = parser.parse_args()
    run(args.campaign_id, args.pre_days)
