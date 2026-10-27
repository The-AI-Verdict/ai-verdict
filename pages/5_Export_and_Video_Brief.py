from __future__ import annotations

import json

import streamlit as st

from core import analysis, db, export, ui_common

st.set_page_config(page_title="Export and Video Brief - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Export and Video Brief")

experiment = ui_common.experiment_selector()
if experiment is None:
    st.stop()
exp_id = experiment["id"]
safe_name = "".join(c if c.isalnum() else "_" for c in experiment["name"])[:40]

if (experiment.get("evaluation_domain") or "gender_healthcare") != "gender_healthcare":
    st.warning(
        "CSV export, the Markdown report, the video brief, and the JSON archive are built for the "
        "gender-healthcare domain only - running them against a sentencing-bias experiment would silently "
        "report a meaningless 0% 'gender assignment' rate rather than anything about sentencing. For now, "
        "use the Results Dashboard's 'Raw results table' or query `data/ai_verdict.db` directly."
    )
    st.stop()

with db.get_connection() as conn:
    rows = analysis.build_experiment_rows(conn, exp_id)
df = analysis.rows_to_dataframe(rows)

if df.empty:
    st.info("No data to export yet.")
    st.stop()

st.subheader("Raw data (CSV)")
csv_text = export.dataframe_to_csv(df)
st.download_button("Download results CSV", csv_text, file_name=f"{safe_name}_results.csv", mime="text/csv")
with st.expander("Preview"):
    st.dataframe(df[[c for c in export.REPORT_COLUMNS if c in df.columns]].head(50), width='stretch')

st.divider()
st.subheader("Experiment report (Markdown)")
with db.get_connection() as conn:
    report_md = export.generate_markdown_report(conn, exp_id)
st.download_button("Download report.md", report_md, file_name=f"{safe_name}_report.md", mime="text/markdown")
with st.expander("Preview report"):
    st.markdown(report_md)

st.divider()
st.subheader("YouTube production brief (Markdown)")
with db.get_connection() as conn:
    brief_md = export.generate_video_brief(conn, exp_id)
st.download_button("Download video_brief.md", brief_md, file_name=f"{safe_name}_video_brief.md", mime="text/markdown")
with st.expander("Preview brief"):
    st.markdown(brief_md)

st.divider()
st.subheader("Full experiment archive (JSON)")
st.caption("Includes the experiment config, every prompt version, model, raw run, evaluation and the full "
           "human-review revision history - everything needed to audit or reproduce the analysis.")
with db.get_connection() as conn:
    archive = export.build_json_archive(conn, exp_id)
archive_text = json.dumps(archive, indent=2, default=str)
st.download_button("Download archive.json", archive_text, file_name=f"{safe_name}_archive.json",
                    mime="application/json")
st.caption(f"Archive size: {len(archive_text) / 1024:.1f} KB")
