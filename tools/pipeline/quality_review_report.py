#!/usr/bin/env python3
"""Build an evidence-first HTML review and aspect-preserving contact sheet."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PREVIEW_KINDS = {"rgb_render", "depth_preview", "normal_preview", "review_preview"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--claims", type=Path, help="Optional confirmed-claims JSON; empty is valid.")
    parser.add_argument("--max-previews-per-run", type=int, default=8)
    parser.add_argument(
        "--incomplete-closure",
        action="store_true",
        help="Label the report as stopped/incomplete and prevent partial findings from reading as a recommendation.",
    )
    return parser.parse_args()


def rel_link(path: str, report_dir: Path) -> str:
    return Path(os.path.relpath(Path(path), report_dir)).as_posix()


def load_claims(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("claims"), list):
        raise SystemExit("claims file must have schema_version=1 and a claims array")
    claims = []
    for index, claim in enumerate(payload["claims"]):
        if claim.get("status") != "confirmed":
            continue
        evidence = claim.get("evidence_artifacts")
        if not isinstance(evidence, list) or not evidence:
            raise SystemExit(f"claims[{index}] is confirmed but has no evidence_artifacts")
        resolved = [Path(value) if Path(value).is_absolute() else path.parent / value for value in evidence]
        missing = [str(value) for value in resolved if not value.is_file()]
        if missing:
            raise SystemExit(f"claims[{index}] evidence files are missing: {missing}")
        normalized = dict(claim)
        normalized["evidence_artifacts"] = [str(value.resolve()) for value in resolved]
        claims.append(normalized)
    return claims


def preview_records(inventory: dict, limit: int) -> list[tuple[dict, dict]]:
    records = []
    for run in inventory["runs"]:
        previews = [artifact for artifact in run["artifacts"] if artifact["kind"] in PREVIEW_KINDS]
        for artifact in previews[:limit]:
            records.append((run, artifact))
    return records


def make_contact_sheet(records: list[tuple[dict, dict]], output: Path) -> bool:
    valid = []
    for run, artifact in records:
        try:
            with Image.open(artifact["path"]) as image:
                image.verify()
            valid.append((run, artifact))
        except OSError:
            continue
    if not valid:
        return False
    cell_width, cell_height, label_height = 420, 300, 54
    columns = 3
    rows = (len(valid) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "#101317")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (run, artifact) in enumerate(valid):
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        with Image.open(artifact["path"]) as source:
            image = ImageOps.contain(source.convert("RGB"), (cell_width - 16, cell_height - 16))
        px = x + (cell_width - image.width) // 2
        py = y + (cell_height - image.height) // 2
        sheet.paste(image, (px, py))
        label = f"{run['scene']} · {run['method']} / {run['mode']}\n{Path(artifact['path']).name}"
        draw.multiline_text((x + 8, y + cell_height + 7), label, fill="#eef2f6", font=font, spacing=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return True


def artifact_list(run: dict, report_dir: Path) -> str:
    if not run["artifacts"]:
        return '<p class="muted">No files found.</p>'
    items = []
    for artifact in run["artifacts"]:
        label = html.escape(artifact["relative_path"])
        href = html.escape(rel_link(artifact["path"], report_dir), quote=True)
        size_mb = artifact["bytes"] / (1024 * 1024)
        items.append(f'<li><span class="kind">{html.escape(artifact["kind"])}</span> <a href="{href}">{label}</a> <small>{size_mb:.2f} MiB</small></li>')
    return "<ul>" + "".join(items) + "</ul>"


def attempt_table(run: dict, report_dir: Path) -> str:
    attempts = run.get("attempts", [])
    if not attempts:
        return '<p class="muted">No immutable attempt metadata found.</p>'
    rows = []
    for attempt in attempts:
        metadata_link = html.escape(rel_link(attempt["metadata_path"], report_dir), quote=True)
        revision = str(attempt.get("upstream_revision") or "—")
        elapsed = attempt.get("elapsed_seconds")
        elapsed_text = "—" if elapsed is None else f"{float(elapsed):.1f}s"
        failure = attempt.get("error")
        failure_text = f"<br><span class=\"failure\">{html.escape(str(failure))}</span>" if failure else ""
        rows.append(
            "<tr>"
            f'<td><a href="{metadata_link}">{html.escape(attempt["attempt_id"])}</a></td>'
            f'<td>{html.escape(str(attempt.get("status", "unknown")))}</td>'
            f'<td>{html.escape(str(attempt.get("exit_code") if attempt.get("exit_code") is not None else "—"))}</td>'
            f'<td>{elapsed_text}</td><td><code>{html.escape(revision)}</code>{failure_text}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Attempt</th><th>Status</th><th>Exit</th><th>Elapsed</th><th>Revision / failure</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def run_card(run: dict, report_dir: Path) -> str:
    warning_html = "".join(f"<li>{html.escape(value)}</li>" for value in run["warnings"])
    warnings = f'<ul class="warnings">{warning_html}</ul>' if warning_html else ""
    badge_class = (
        "available" if run["run_status"] == "outputs_available"
        else "partial" if run["run_status"] == "partial"
        else "failed" if run["run_status"] == "failed"
        else "missing"
    )
    return f"""
    <section class="run-card">
      <div class="run-title"><h3>{html.escape(run['scene'])} · {html.escape(run['method'])} / {html.escape(run['mode'])}</h3>
      <span class="badge {badge_class}">{html.escape(run['run_status'])}</span></div>
      <p><b>Review status:</b> {html.escape(run['review_status'])}</p>
      <p>{html.escape(run['contract_note'])}</p>
      {warnings}
      <details><summary>{len(run.get('attempts', []))} immutable attempt(s)</summary>{attempt_table(run, report_dir)}</details>
      <details><summary>{len(run['artifacts'])} artifact(s)</summary>{artifact_list(run, report_dir)}</details>
    </section>"""


def claim_card(claim: dict, report_dir: Path) -> str:
    links = "".join(
        f'<li><a href="{html.escape(rel_link(value, report_dir), quote=True)}">{html.escape(Path(value).name)}</a></li>'
        for value in claim["evidence_artifacts"]
    )
    return f"""
    <article class="claim">
      <div class="eyebrow">Confirmed · {html.escape(str(claim.get('scene', 'cross-scene')))}</div>
      <h3>{html.escape(str(claim.get('title', 'Untitled result')))}</h3>
      <p>{html.escape(str(claim.get('statement', '')))}</p>
      <ul>{links}</ul>
    </article>"""


def main() -> int:
    args = parse_args()
    if args.max_previews_per_run < 0:
        raise SystemExit("--max-previews-per-run must be non-negative")
    inventory_path = args.inventory.resolve()
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("runs"), list):
        raise SystemExit("inventory must have schema_version=1 and a runs array")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claims = load_claims(args.claims.resolve() if args.claims else None)
    previews = preview_records(inventory, args.max_previews_per_run)
    contact_sheet = output_dir / "contact_sheet.jpg"
    has_sheet = make_contact_sheet(previews, contact_sheet)
    summary = inventory["summary"]
    claim_html = "".join(claim_card(claim, output_dir) for claim in claims)
    if not claim_html:
        claim_html = '<p class="empty">No positive result is marked confirmed yet. This section populates only from evidence-backed claims.</p>'
    sheet_html = '<img class="sheet" src="contact_sheet.jpg" alt="Actual reconstruction output contact sheet">' if has_sheet else '<p class="empty">No actual reconstruction preview images are present yet; no contact sheet was fabricated.</p>'
    cards = "".join(run_card(run, output_dir) for run in inventory["runs"])
    title = "Stopped two-scene reconstruction review" if args.incomplete_closure else "Two-scene reconstruction review"
    heading = "Stopped, incomplete two-scene results" if args.incomplete_closure else "Two-scene 3D reconstruction results"
    claims_heading = "Confirmed partial findings — not a recommendation" if args.incomplete_closure else "Confirmed positive results"
    closure_html = ""
    if args.incomplete_closure:
        closure_html = (
            '<section class="closure"><b>Execution stopped; benchmark incomplete.</b> '
            'These local artifacts do not support a validated recommendation or a winner. '
            'Held-out image scores remain unavailable, and no further execution is represented here.</section>'
        )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0e12; --panel:#151a20; --line:#2a323d; --text:#eef2f6; --muted:#9aa8b6; --green:#5bd39a; --amber:#f1bd62; --red:#ef7e77; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,sans-serif }}
main {{ width:min(1180px,calc(100% - 32px)); margin:40px auto 80px }} h1 {{ font-size:clamp(30px,5vw,58px); line-height:1.02; max-width:900px }}
h2 {{ margin-top:48px }} h3 {{ margin:0 }} a {{ color:#8dc7ff }} .lede,.muted,small {{ color:var(--muted) }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:28px 0 }}
.stat,.run-card,.claim,.empty {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px }} .stat b {{ display:block; font-size:28px }}
.run-card {{ margin:12px 0 }} .run-title {{ display:flex; justify-content:space-between; gap:16px; align-items:start }}
.badge {{ border:1px solid; border-radius:999px; padding:3px 9px; font-size:12px }} .available {{ color:var(--green) }} .partial {{ color:var(--amber) }} .failed,.missing {{ color:var(--red) }}
.warnings {{ color:var(--amber) }} .kind,.eyebrow {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em }}
.sheet {{ display:block; max-width:100%; height:auto; border:1px solid var(--line); border-radius:12px }} li {{ margin:5px 0; overflow-wrap:anywhere }}
.claim {{ border-color:#256c50; margin:12px 0 }} details {{ margin-top:12px }}
.closure {{ background:#2a1d12; border:1px solid var(--amber); border-radius:12px; color:#ffd99a; margin:24px 0; padding:18px }}
.table-wrap {{ overflow-x:auto }} table {{ width:100%; border-collapse:collapse; margin-top:10px }} th,td {{ padding:8px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top }} code {{ overflow-wrap:anywhere }} .failure {{ color:var(--red) }}
</style></head><body><main>
<p class="eyebrow">Evidence-first benchmark review</p><h1>{html.escape(heading)}</h1>
<p class="lede">Generated from files that actually exist. Missing runs stay visible; input-view reconstructions are not relabelled as held-out evidence.</p>
{closure_html}
<div class="stats"><div class="stat"><b>{summary['outputs_available']}</b>outputs available</div><div class="stat"><b>{summary['partial']}</b>partial</div><div class="stat"><b>{summary.get('failed', 0)}</b>failed slots</div><div class="stat"><b>{summary['not_run']}</b>not run</div><div class="stat"><b>{summary.get('completed_attempts', 0)}</b>completed attempts</div><div class="stat"><b>{summary.get('failed_attempts', 0)}</b>failed attempts retained</div><div class="stat"><b>{summary['unassigned_file_count']}</b>unassigned files</div></div>
<h2>{html.escape(claims_heading)}</h2>{claim_html}
<h2>Actual output contact sheet</h2>{sheet_html}
<h2>Complete experiment matrix</h2>{cards}
<p class="muted">Inventory: <a href="{html.escape(rel_link(str(inventory_path), output_dir), quote=True)}">{html.escape(inventory_path.name)}</a></p>
</main></body></html>"""
    (output_dir / "index.html").write_text(html_text)
    print(json.dumps({"report": str(output_dir / "index.html"), "contact_sheet": str(contact_sheet) if has_sheet else None, "confirmed_claims": len(claims)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
