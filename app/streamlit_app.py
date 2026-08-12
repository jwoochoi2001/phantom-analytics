"""캠페인 효과 분석 Streamlit 앱.

캠페인과 발행 전 관찰 기간(pre_days)을 선택해 pipeline/(prepare_data → estimate_effect)을
실행한다. 효과 결과보다 먼저 성향점수 분포·공통영역·매칭률·매칭 전후 SMD를 보여주고,
품질 상태(status=="ok")를 통과한 경우에만 주요·보조 효과와 95% CI를 표시한다.
분석 조건(campaign_id, pre_days)이 이미 저장된 결과와 같으면 재계산하지 않고
outputs/campaign_{id}/results.json·analysis_data.csv·matched_data.csv를 재사용한다.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

def _force_utf8_stdio() -> None:
    """콘솔 코드페이지(cp949 등)에서 pipeline/*.py의 한글 print()가 깨지거나
    UnicodeEncodeError로 죽지 않도록 stdout/stderr를 강제로 UTF-8로 바꾼다."""
    import io

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                buffer = getattr(stream, "buffer", None)
                if buffer is not None:
                    setattr(sys, name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))
            except Exception:
                pass  # 그래도 안 되면 원래 스트림 유지(최후의 수단)


_force_utf8_stdio()

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

# data/raw/는 라이선스가 있는 Kaggle 데이터셋이라 저장소에 커밋하지 않는다(용량도
# transaction_data.csv 141MB로 GitHub 100MB 제한을 넘음). 클라우드 배포 환경처럼
# data/raw/가 없는 곳에서는 outputs/campaign_*/results.json에 이미 저장된 캐시
# 결과만 조회하는 모드로 우아하게 전환한다 — 원본이 없다고 앱이 죽지 않게 한다.
RAW_DATA_AVAILABLE = (RAW / "campaign_desc.csv").exists()
CACHE_ONLY_PRE_DAYS = 90  # 캐시 전용 모드에서 30개 캠페인 전체가 이 값으로 미리 계산되어 있음

sys.path.insert(0, str(ROOT / "pipeline"))
from run_pipeline import run as run_pipeline, MIN_GROUP_SIZE  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from profitability import (  # noqa: E402
    compute_candidates, empty_candidates, losing_candidates_example, recommend, validate_candidates,
)

OUTCOME_LABELS = {
    "target_purchase": "대상 상품 구매율",
    "target_sales": "대상 상품 구매금액",
    "target_quantity": "대상 상품 구매수량",
    "any_purchase": "전체 구매율",
    "total_sales": "전체 구매금액",
    "baskets": "장바구니 수",
}

STATUS_LABELS = {
    "ok": ("✅ 품질 기준 통과", "success"),
    "insufficient_sample": ("⛔ 중단: 표본 부족", "error"),
    "insufficient_overlap": ("⛔ 중단: 공통지지영역 부족", "error"),
    "residual_imbalance": ("⚠️ 경고: 매칭 후 잔여 불균형", "warning"),
    "no_raw_data": ("☁️ 캐시에 없는 조건", "warning"),
}


@st.cache_data(show_spinner=False)
def load_campaign_list() -> pd.DataFrame:
    if RAW_DATA_AVAILABLE:
        return pd.read_csv(RAW / "campaign_desc.csv").sort_values("CAMPAIGN")
    return _load_campaign_list_from_cache()


def _load_campaign_list_from_cache() -> pd.DataFrame:
    """원본 데이터가 없을 때: outputs/campaign_*/results.json을 스캔해 선택지를 만든다."""
    rows = []
    for results_path in sorted(OUTPUTS.glob("campaign_*/results.json")):
        try:
            r = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "start_day" not in r or "end_day" not in r:
            continue
        rows.append(
            {"CAMPAIGN": r["campaign_id"], "DESCRIPTION": r.get("description", ""),
             "START_DAY": r["start_day"], "END_DAY": r["end_day"]}
        )
    if not rows:
        return pd.DataFrame(columns=["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"])
    return pd.DataFrame(rows).sort_values("CAMPAIGN")


def get_or_run(campaign_id: int, pre_days: int, force: bool = False) -> tuple[dict, bool]:
    """캐시 재사용: outputs/campaign_{id}/results.json이 같은 조건이면 그대로 쓴다.

    원본 데이터(data/raw/)가 없는 환경(예: 클라우드 배포)에서 캐시에 없는 조건을
    요청하면, 계산을 시도하다 크래시하는 대신 안내 메시지가 담긴 status를 반환한다.
    """
    out_dir = OUTPUTS / f"campaign_{campaign_id}"
    results_path = out_dir / "results.json"

    if not force and results_path.exists():
        try:
            cached = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached and cached.get("campaign_id") == campaign_id and cached.get("pre_days") == pre_days:
            return cached, True

    if not RAW_DATA_AVAILABLE:
        return {
            "status": "no_raw_data",
            "reason": (
                "이 배포판에는 원본 데이터(data/raw/)가 없어 사전 계산된 결과만 조회할 수 있습니다. "
                f"campaign_id={campaign_id}, pre_days={pre_days} 조합은 캐시에 없습니다. "
                f"pre_days={CACHE_ONLY_PRE_DAYS}으로 다시 선택해 보세요."
            ),
            "campaign_id": campaign_id, "pre_days": pre_days,
        }, False

    res = run_pipeline(campaign_id, pre_days)
    return res, False


def load_analysis_data(campaign_id: int) -> pd.DataFrame | None:
    path = OUTPUTS / f"campaign_{campaign_id}" / "analysis_data.csv"
    return pd.read_csv(path) if path.exists() else None


def load_matched_data(campaign_id: int) -> pd.DataFrame | None:
    path = OUTPUTS / f"campaign_{campaign_id}" / "matched_data.csv"
    return pd.read_csv(path) if path.exists() else None


def render_pscore_plot(df: pd.DataFrame, lo: float | None, hi: float | None):
    t = df.loc[df["group"] == "처치", "p_score"]
    c = df.loc[df["group"] == "대조", "p_score"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(c, bins=bins, alpha=0.55, label=f"대조 (n={len(c)})", color="#1f77b4", density=True)
    ax.hist(t, bins=bins, alpha=0.55, label=f"처치 (n={len(t)})", color="#d62728", density=True)
    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, color="green", alpha=0.08, label=f"공통지지영역 [{lo:.3f}, {hi:.3f}]")
        ax.axvline(lo, color="green", linestyle="--", linewidth=1)
        ax.axvline(hi, color="green", linestyle="--", linewidth=1)
    ax.set_xlabel("p_score (캠페인 수신확률)")
    ax.set_ylabel("밀도")
    ax.set_title("처치/대조 p_score 분포")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def render_smd_love_plot(before_df: pd.DataFrame, after_df: pd.DataFrame | None):
    fig, ax = plt.subplots(figsize=(7, 4))
    vars_order = before_df["변수"].tolist()
    y = np.arange(len(vars_order))
    ax.scatter(before_df.set_index("변수").loc[vars_order, "SMD"], y, color="#888888", label="매칭 전", zorder=3)
    if after_df is not None:
        ax.scatter(after_df.set_index("변수").loc[vars_order, "SMD"], y, color="#d62728", label="매칭 후", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.1, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-0.1, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(vars_order, fontsize=8)
    ax.set_xlabel("표준화 평균차이 (SMD)")
    ax.set_title("매칭 전후 SMD 비교")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def render_outcome_table(outcomes: dict, cols: list[str], title: str):
    rows = []
    for col in cols:
        o = outcomes[col]
        raw, mat = o["raw"], o["matched"]
        rows.append(
            {
                "변수": f"{col} ({OUTCOME_LABELS[col]})",
                "보정전 차이": raw["diff"],
                "보정전 95%CI": f"[{raw['ci_lo']:.3f}, {raw['ci_hi']:.3f}]",
                "매칭후 차이": mat["diff"],
                "매칭후 95%CI": f"[{mat['ci_lo']:.3f}, {mat['ci_hi']:.3f}]",
            }
        )
    st.markdown(f"**{title}**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="캠페인 효과 분석", layout="wide")
    st.title("쿠폰 캠페인 효과 분석")
    st.caption("성향점수 매칭 기반 캠페인별 처치효과 추정 — outputs/campaign_{id}/ 결과를 캐시로 재사용합니다.")

    if not RAW_DATA_AVAILABLE:
        st.info(
            "☁️ 이 배포판에는 원본 데이터(data/raw/, 라이선스·용량 문제로 저장소에 포함하지 않음)가 "
            f"없어 사전 계산된 결과만 조회합니다. 발행 전 관찰 기간은 캐시된 값(pre_days="
            f"{CACHE_ONLY_PRE_DAYS})으로 고정됩니다. 전체 기능은 저장소를 내려받아 "
            "`data/raw/`를 채운 뒤 로컬에서 실행하세요."
        )

    desc = load_campaign_list()
    if desc.empty:
        st.error("조회 가능한 캠페인이 없습니다 (outputs/campaign_*/results.json 캐시가 비어 있음).")
        return

    options = [
        f"{int(r.CAMPAIGN)} - {r.DESCRIPTION} (DAY {int(r.START_DAY)}~{int(r.END_DAY)})"
        for _, r in desc.iterrows()
    ]
    id_by_option = {opt: int(opt.split(" - ")[0]) for opt in options}

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_option = st.selectbox("캠페인 선택", options, index=options.index(
            next(o for o in options if o.startswith("18 -"))
        ) if any(o.startswith("18 -") for o in options) else 0)
        campaign_id = id_by_option[selected_option]
    with col2:
        pre_days = st.number_input(
            "발행 전 관찰 기간(pre_days, 일)", min_value=1, max_value=600,
            value=CACHE_ONLY_PRE_DAYS, step=10, disabled=not RAW_DATA_AVAILABLE,
        )
    with col3:
        force = st.checkbox("캐시 무시하고 재실행", value=False, disabled=not RAW_DATA_AVAILABLE)

    run_clicked = st.button("분석 실행", type="primary")

    # 결과를 session_state에 보관한다 — 후보안 표(5번 섹션)의 버튼·입력 상호작용마다
    # Streamlit이 스크립트를 처음부터 다시 실행하는데, 그때마다 run_clicked는 다시
    # False가 되므로 매번 재계산하지 않고 마지막 결과를 그대로 이어서 보여준다.
    if run_clicked:
        with st.spinner("파이프라인 실행 중... (prepare_data → estimate_effect)"):
            res, from_cache = get_or_run(campaign_id, pre_days, force=force)
        st.session_state["last_res"] = res
        st.session_state["last_from_cache"] = from_cache
    elif (
        "last_res" in st.session_state
        and st.session_state["last_res"].get("campaign_id") == campaign_id
        and st.session_state["last_res"].get("pre_days") == pre_days
    ):
        res = st.session_state["last_res"]
        from_cache = st.session_state.get("last_from_cache", True)
    else:
        st.info("캠페인과 발행 전 관찰 기간을 선택하고 '분석 실행'을 누르세요.")
        return

    if res["status"] == "no_raw_data":
        st.warning(res["reason"])
        return

    if from_cache:
        st.success(f"저장된 결과를 재사용했습니다 (campaign_id={campaign_id}, pre_days={pre_days} 조건 동일).")
    else:
        st.info("조건이 새롭거나 재실행을 요청해 파이프라인을 새로 실행했습니다.")

    # -----------------------------------------------------------------
    # 캠페인/집단 정보
    # -----------------------------------------------------------------
    st.header(f"캠페인 {res.get('campaign_id')} 결과")
    info_cols = st.columns(4)
    info_cols[0].metric("캠페인 기간", f"DAY {res.get('start_day')}~{res.get('end_day')}")
    info_cols[1].metric("겹치는 캠페인", str(res.get("overlap_ids", [])))
    info_cols[2].metric("처치 가구", res.get("n_treatment"))
    info_cols[3].metric("대조 가구", res.get("n_control"))

    label, kind = STATUS_LABELS.get(res["status"], (res["status"], "info"))
    getattr(st, kind)(f"{label}" + (f" — {res['reason']}" if res.get("reason") else ""))

    if res.get("n_treatment", 0) < MIN_GROUP_SIZE or res.get("n_control", 0) < MIN_GROUP_SIZE:
        st.warning(
            f"집단 구성 단계에서 이미 표본 기준(각 {MIN_GROUP_SIZE}명 이상)을 충족하지 못해 "
            "이후 단계(성향점수·매칭)를 진행하지 않았습니다."
        )
        return

    # -----------------------------------------------------------------
    # 성향점수 분포 + 공통지지영역 (효과 결과보다 먼저)
    # -----------------------------------------------------------------
    st.subheader("1) 성향점수 분포 · 공통지지영역")
    df = load_analysis_data(campaign_id)
    if df is None or "p_score" not in df.columns:
        st.warning("p_score가 계산되지 않았습니다 (이전 단계에서 중단됨).")
        return

    lo, hi = res.get("common_support", [None, None])
    fig = render_pscore_plot(df, lo, hi)
    st.pyplot(fig, use_container_width=True)

    if "in_support" in df.columns:
        support_counts = df.groupby(["group", "in_support"]).size().unstack(fill_value=0)
        support_counts = support_counts.rename(columns={True: "영역 안", False: "영역 밖"})
        st.markdown("**공통지지영역 안/밖 가구 수**")
        st.dataframe(support_counts, use_container_width=True)

    if res["status"] == "insufficient_overlap":
        st.error("공통지지영역 표본이 부족해 매칭을 진행하지 않았습니다.")
        return

    # -----------------------------------------------------------------
    # 매칭률
    # -----------------------------------------------------------------
    st.subheader("2) 매칭 결과")
    m_cols = st.columns(4)
    m_cols[0].metric("caliper", f"{res.get('caliper_multiplier')} x SD = {res.get('caliper_logit')}")
    m_cols[1].metric("매칭 쌍 수", res.get("n_matched_pairs"))
    m_cols[2].metric("처치 매칭률", f"{res.get('match_rate', 0):.1%}" if res.get("match_rate") is not None else "-")
    m_cols[3].metric("공통영역 표본(처치/대조)",
                      f"{res.get('n_common_support_treatment')}/{res.get('n_common_support_control')}")

    if res["status"] == "insufficient_sample" and "n_matched_pairs" in res:
        st.error(f"매칭 쌍 수가 기준에 미달해 균형·효과 추정을 진행하지 않았습니다: {res.get('reason')}")
        return

    # -----------------------------------------------------------------
    # 매칭 전후 SMD (효과 결과보다 먼저)
    # -----------------------------------------------------------------
    st.subheader("3) 매칭 전후 균형 (SMD)")
    before_df = pd.DataFrame(res.get("balance_smd_before", []))
    after_df = pd.DataFrame(res.get("balance_smd_after", [])) if res.get("balance_smd_after") else None
    if not before_df.empty:
        fig2 = render_smd_love_plot(before_df, after_df)
        st.pyplot(fig2, use_container_width=True)
        smd_compare = before_df.rename(columns={"SMD": "매칭전 SMD"})
        if after_df is not None:
            smd_compare = smd_compare.merge(after_df.rename(columns={"SMD": "매칭후 SMD"}), on="변수")
        st.dataframe(smd_compare, hide_index=True, use_container_width=True)
        max_before = res.get("max_abs_smd_before")
        max_after = res.get("max_abs_smd")
        st.caption(f"매칭 전 최대|SMD|={max_before} → 매칭 후 최대|SMD|={max_after} (기준 0.1)")

    # -----------------------------------------------------------------
    # 효과 결과 — status == "ok"인 경우에만 표시
    # -----------------------------------------------------------------
    st.subheader("4) 캠페인 효과 (주요·보조 결과, 95% CI)")
    if res["status"] != "ok":
        st.warning(
            f"품질 상태가 '{res['status']}'로 기준을 통과하지 못해 효과 추정치를 표시하지 않습니다. "
            f"({res.get('reason', '')})"
        )
        return

    outcomes = res["outcomes"]
    render_outcome_table(outcomes["primary"], ["target_purchase", "target_sales", "target_quantity"], "주요 결과 (캠페인 대상 상품)")
    render_outcome_table(outcomes["secondary"], ["any_purchase", "total_sales", "baskets"], "보조 결과 (전체 상품)")

    st.caption(
        "매칭 후 차이는 관찰된 발행 전 특성만 통제한 추정치이며, 관찰되지 않은 교란요인은 통제되지 "
        "않았으므로 확정된 인과효과로 단정하지 않습니다."
    )

    render_profitability_section(res)


def render_profitability_section(res: dict):
    st.subheader("5) 쿠폰 후보안 수익성 비교")

    ate = res["outcomes"]["primary"]["target_sales"]["matched"]["diff"]
    ci_lo = res["outcomes"]["primary"]["target_sales"]["matched"]["ci_lo"]
    ci_hi = res["outcomes"]["primary"]["target_sales"]["matched"]["ci_hi"]

    st.markdown(
        f"**이 캠페인 분석에서 가져온 값 (모델 추정치, 사용자 입력 아님)**: "
        f"가구당 대상 상품 증분매출(매칭후) = **{ate:.2f}** (95% CI [{ci_lo:.2f}, {ci_hi:.2f}])"
    )
    if ate <= 0:
        st.warning(
            "이 캠페인의 매칭 후 증분매출 추정치가 0 이하입니다. 아래 계산은 그대로 진행되지만, "
            "모든 후보의 증분매출이 0 이하로 나와 손익분기를 충족하기 어렵습니다."
        )

    st.markdown("**후보 쿠폰안 입력 (사용자 입력 수익성 가정)** — 할인액/운영비는 원, 예상사용률은 0~1")
    key = f"candidates_{res.get('campaign_id')}"
    if key not in st.session_state:
        st.session_state[key] = empty_candidates()

    preset_cols = st.columns([1, 1, 4])
    if preset_cols[0].button("예시: 기본 후보로 초기화", key=f"reset_default_{res.get('campaign_id')}"):
        st.session_state[key] = empty_candidates()
        st.rerun()
    if preset_cols[1].button("예시: 손익분기 미충족 후보 채우기", key=f"reset_losing_{res.get('campaign_id')}"):
        st.session_state[key] = losing_candidates_example()
        st.rerun()

    edited = st.data_editor(
        st.session_state[key],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "예상사용률": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            "할인액": st.column_config.NumberColumn(min_value=0.0, step=0.5),
            "발행수": st.column_config.NumberColumn(min_value=0, step=100),
            "운영비": st.column_config.NumberColumn(min_value=0.0, step=100.0),
        },
        key=f"editor_{res.get('campaign_id')}",
    )
    st.session_state[key] = edited

    warnings = validate_candidates(edited)
    if warnings:
        for w in warnings:
            st.error(w)
        return

    result_df = compute_candidates(edited, ate_per_household=ate)
    display_cols = [
        "후보명", "할인액", "예상사용률", "발행수", "운영비",
        "예상사용건수", "예상증분매출", "총비용", "증분이익", "ROI", "손익분기충족",
    ]
    best_name, reason = recommend(result_df)

    def highlight_best(row):
        color = "background-color: #d4f4dd" if row["후보명"] == best_name else ""
        return [color] * len(row)

    styled = result_df[display_cols].style.apply(highlight_best, axis=1).format(
        {"할인액": "{:.2f}", "예상사용률": "{:.0%}", "예상사용건수": "{:.0f}", "예상증분매출": "{:.0f}",
         "총비용": "{:.0f}", "증분이익": "{:.0f}", "ROI": "{:.1%}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if best_name is not None:
        st.success(f"추천(입력한 후보군 내): **{best_name}** — {reason}")

    st.caption(
        "계산식: 예상사용건수=발행수×예상사용률 / 예상증분매출=예상사용건수×가구당증분매출 / "
        "총비용=예상사용건수×할인액+운영비 / 증분이익=예상증분매출-총비용 / ROI=증분이익/총비용. "
        "추천은 입력한 후보군 안에서만 판단하며 전역 최적 쿠폰을 의미하지 않습니다."
    )


if __name__ == "__main__":
    main()
