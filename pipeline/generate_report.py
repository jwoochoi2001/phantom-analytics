"""results.json으로부터 outputs/campaign_{id}/report.html을 생성한다.

캠페인별 최종 결과 4파일(analysis_data.csv, matched_data.csv, results.json, report.html)
중 마지막 파일을 채운다. 이미 계산된 results.json을 읽어 렌더링만 하므로 재계산은
하지 않는다(분석 재실행 규칙과 별개).
"""

import json
import sys
from pathlib import Path

from jinja2 import Template

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
TEMPLATE_PATH = Path(__file__).resolve().parent / "report_template.html.j2"


def generate_report(campaign_id: int) -> Path:
    results_path = OUTPUTS / f"campaign_{campaign_id}" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} 없음 — 먼저 pipeline/run_pipeline.py --campaign_id {campaign_id} 실행 필요"
        )
    res = json.loads(results_path.read_text(encoding="utf-8"))

    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    html = template.render(res=res)

    out_path = OUTPUTS / f"campaign_{campaign_id}" / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="캠페인 report.html 생성")
    parser.add_argument("--campaign_id", type=int, required=True)
    args = parser.parse_args()

    path = generate_report(args.campaign_id)
    print(f"저장 완료: {path}")
