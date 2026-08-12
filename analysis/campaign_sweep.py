"""30개 캠페인 전체에 파이프라인(pre_days=90 고정)을 일괄 실행해 결과를 비교한다.

캠페인별로 status, 매칭 쌍 수, 매칭률, 매칭 후 최대|SMD|, 주요 결과(대상 상품 구매금액)의
보정전/매칭후 차이를 한 표로 정리한다. 개별 캠페인 스크립트를 하나씩 실행하지 않고도
전체 그림을 볼 수 있게 하기 위함이다.

출력: outputs/campaign_sweep_pre_days_90.csv, 터미널 요약표
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from run_pipeline import run  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_sweep_pre_days_90.csv"
PRE_DAYS = 90


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    desc = pd.read_csv(RAW / "campaign_desc.csv").sort_values("CAMPAIGN")

    rows = []
    for _, r in desc.iterrows():
        cid = int(r["CAMPAIGN"])
        print(f"\n{'#'*78}\n캠페인 {cid} 실행 (pre_days={PRE_DAYS})\n{'#'*78}")
        try:
            res = run(cid, PRE_DAYS)
        except Exception as exc:  # noqa: BLE001 - 스윕 도중 한 캠페인이 죽어도 나머지는 계속
            rows.append({"campaign_id": cid, "type": r["DESCRIPTION"], "status": f"error: {exc}"})
            print(f"[스윕 경고] 캠페인 {cid} 실행 중 예외 발생: {exc}")
            continue

        row = {
            "campaign_id": cid,
            "type": r["DESCRIPTION"],
            "period_days": int(r["END_DAY"]) - int(r["START_DAY"]) + 1,
            "status": res["status"],
            "n_treatment": res.get("n_treatment"),
            "n_control": res.get("n_control"),
            "n_common_support_t": res.get("n_common_support_treatment"),
            "n_common_support_c": res.get("n_common_support_control"),
            "n_matched_pairs": res.get("n_matched_pairs"),
            "match_rate": res.get("match_rate"),
            "max_abs_smd": res.get("max_abs_smd"),
        }
        if res["status"] == "ok":
            ts = res["outcomes"]["primary"]["target_sales"]
            row["raw_diff_target_sales"] = ts["raw"]["diff"]
            row["matched_diff_target_sales"] = ts["matched"]["diff"]
            row["matched_ci_lo"] = ts["matched"]["ci_lo"]
            row["matched_ci_hi"] = ts["matched"]["ci_hi"]
            row["matched_effect_significant"] = not (ts["matched"]["ci_lo"] <= 0 <= ts["matched"]["ci_hi"])
        rows.append(row)

    sweep = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*78}\n30개 캠페인 스윕 요약 (pre_days={PRE_DAYS})\n{'='*78}")
    print(sweep.to_string(index=False))

    print("\n[상태별 캠페인 수]")
    print(sweep["status"].value_counts().to_string())

    ok = sweep.loc[sweep["status"] == "ok"]
    if not ok.empty:
        n_sig = int(ok["matched_effect_significant"].sum())
        print(
            f"\nstatus=='ok'인 {len(ok)}개 캠페인 중, 매칭 후에도 대상 상품 매출 증분이 "
            f"통계적으로 유의(95% CI가 0 제외)한 캠페인: {n_sig}개"
        )
        print(ok[["campaign_id", "type", "n_matched_pairs", "raw_diff_target_sales",
                   "matched_diff_target_sales", "matched_effect_significant"]].to_string(index=False))

    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
