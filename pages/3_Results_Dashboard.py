from __future__ import annotations

import streamlit as st

from core import analysis, charts, db, ui_common

st.set_page_config(page_title="Results Dashboard - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Results Dashboard")

experiment = ui_common.experiment_selector()
if experiment is None:
    st.stop()

exp_id = experiment["id"]
with db.get_connection() as conn:
    rows = analysis.build_experiment_rows(conn, exp_id)
df = analysis.rows_to_dataframe(rows)

if df.empty:
    st.info("No runs planned yet for this experiment. Go to Run Experiment first.")
    st.stop()

st.subheader("Filters")
f1, f2, f3 = st.columns(3)
with f1:
    model_filter = st.selectbox("Model", ["(all)"] + sorted(df["model_id"].unique().tolist()))
with f2:
    variant_filter = st.selectbox("Prompt variant", ["(all)"] + sorted(df["prompt_variant"].dropna().unique().tolist()))
with f3:
    review_filter = st.selectbox("Review status", ["(all)", "reviewed", "unreviewed"])

filtered = analysis.filter_rows(
    df,
    model_id=None if model_filter == "(all)" else model_filter,
    prompt_variant=None if variant_filter == "(all)" else variant_filter,
    review_status=None if review_filter == "(all)" else review_filter,
)

st.divider()
st.subheader("Run outcomes")
counts = analysis.run_counts(filtered)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total runs", counts.total)
c2.metric("Success", counts.success)
c3.metric("Failed", counts.failed)
c4.metric("Pending", counts.pending)
st.pyplot(charts.run_status_chart(counts.success, counts.failed, counts.pending))

missing = analysis.missing_and_invalid_counts(filtered)
if missing["invalid_unreviewed"] > 0:
    st.warning(f"{missing['invalid_unreviewed']} successful response(s) have an automated evaluation that "
               f"failed validation and have NOT been human-reviewed yet. They are excluded from all "
               f"percentages below. Resolve them on the Human Review page.")
if missing["unreviewed_total"] > 0:
    st.caption(f"{missing['unreviewed_total']} response(s) overall have not been human-reviewed. Automated "
               f"classifications are shown below but are not ground truth until reviewed.")

st.divider()
st.subheader("Gender assignment results")
st.caption(
    "Denominator = valid classified responses only (successful call + valid automated evaluation or a "
    "resolving human review). Explicit and implied assignments are always shown separately."
)

for role in ("doctor", "nurse"):
    st.markdown(f"### {role.capitalize()}")
    summary = analysis.role_summary(filtered, role)
    cols = st.columns(5)
    metric_order = ["explicit", "implied", "any_assignment", "unspecified", "ambiguous"]
    for col, key in zip(cols, metric_order):
        r = summary[key]
        label = r.label.split(": ", 1)[1].capitalize()
        delta = "exploratory (n<20)" if r.is_exploratory else None
        col.metric(label, f"{r.proportion*100:.1f}%", help=f"{r.numerator}/{r.denominator}; 95% CI "
                                                             f"{r.ci[0]*100:.1f}%-{r.ci[1]*100:.1f}%")
        if delta:
            col.caption(delta)

st.divider()
st.subheader("By model")
for role in ("doctor", "nurse"):
    by_model = analysis.summarize_by(filtered, role, ["model_id"])
    if by_model.empty:
        continue
    st.markdown(f"**{role.capitalize()} - explicit assignment rate by model**")
    fig = charts.proportion_by_group_chart(by_model, "model_id", "explicit", f"{role.capitalize()}: explicit gender assignment by model")
    if fig:
        st.pyplot(fig)
    st.dataframe(by_model, use_container_width=True)

if filtered["prompt_variant"].nunique() > 1:
    st.divider()
    st.subheader("By prompt variant")
    for role in ("doctor", "nurse"):
        by_variant = analysis.summarize_by(filtered, role, ["prompt_variant"])
        if by_variant.empty:
            continue
        st.markdown(f"**{role.capitalize()} - explicit assignment rate by prompt variant**")
        fig = charts.proportion_by_group_chart(by_variant, "prompt_variant", "explicit",
                                                f"{role.capitalize()}: explicit gender assignment by prompt variant")
        if fig:
            st.pyplot(fig)
        st.dataframe(by_variant, use_container_width=True)

st.divider()
st.subheader("Evaluator vs. human reviewer disagreements")
disagreements = analysis.disagreement_rows(filtered)
if disagreements.empty:
    st.caption("No disagreements recorded (or no human reviews yet).")
else:
    st.dataframe(
        disagreements[["run_id", "model_id", "prompt_variant", "doctor_auto_classification",
                       "doctor_final_classification", "nurse_auto_classification", "nurse_final_classification",
                       "reviewer_note"]],
        use_container_width=True,
    )

st.divider()
with st.expander("Raw results table"):
    st.dataframe(filtered, use_container_width=True)

st.info(
    "Reminder: repeated responses from the same model are not independent observations in the strict "
    "statistical sense, and a single prompt cannot establish a general stereotype. Compare across the "
    "role-reversed and alternative-profession prompt variants before drawing broader conclusions."
)
