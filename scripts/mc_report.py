"""Turn a montecarlo --json dump into a standalone HTML stability report.

    # 1) run a comparison, dumping raw data
    python -m scripts.montecarlo --config fishing.toml --equal-total 1200000 \
        --barges 4 6 8 --runs 100 --json mc.json

    # 2) render it to a self-contained chart you can open in any browser
    python -m scripts.mc_report mc.json --out mc_report.html

The report (survival bars + catch-distribution strip + table + takeaways) is generated
entirely from the JSON, so any comparison — any configs, any run count — produces the
matching chart with no hand editing. The output file embeds its data and works offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("mc_report_template.html")
TOKEN = "/*__MC_DATA__*/"


def build(mc_json_path, out_path):
    with open(mc_json_path, "r", encoding="utf-8") as fh:
        mc = json.load(fh)

    configs = []
    for c in mc["configs"]:
        label = c.get("label") or f"{c['n']} × {c['tank'] // 1000}k lb"
        entry = {
            "label": label,
            "survival": c["survival"], "mean": c["mean"], "std": c["std"],
            "min": c["min"], "p10": c["p10"], "median": c["median"], "cv": c["cv"],
            "catches": c["catches"], "deaths": c["deaths"],
            "loss_med_min": c.get("loss_med_min"), "loss_min_min": c.get("loss_min_min"),
            "sea_ref": c.get("sea_ref"), "dock_t_ref": c.get("dock_t_ref"),
            "dock_b_ref": c.get("dock_b_ref"),
        }
        if c.get("total"):
            entry["total"] = c["total"]
        configs.append(entry)
    data = {"runs": mc.get("runs"), "stage": mc.get("stage"),
            "title": mc.get("title"), "configs": configs}

    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace(TOKEN, json.dumps(data), 1)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="montecarlo --json output file")
    ap.add_argument("--out", default="mc_report.html", help="output HTML file")
    args = ap.parse_args()
    out = build(args.json, args.out)
    print(f"wrote {out}  ({len(json.load(open(args.json))['configs'])} configs)")
    print("open it in a browser, or share the file")


if __name__ == "__main__":
    main()
