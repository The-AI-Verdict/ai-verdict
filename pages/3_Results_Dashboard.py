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
domain = experiment.get("evaluation_domain") or "gender_healthcare"
with db.get_connection() as conn:
    rows = analysis.build_sentencing_rows(conn, exp_id) if domain == "sentencing_bias" \
        else analysis.build_experiment_rows(conn, exp_id)
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
cost = analysis.cost_summary(filtered)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total runs", counts.total)
c2.metric("Success", counts.success)
c3.metric("Failed", counts.failed)
c4.metric("Pending", counts.pending)
c5.metric("Cost (current filter)", f"${cost.total_usd:.4f}")
cost_captions = []
if cost.runs_missing_cost:
    cost_captions.append(f"{cost.runs_missing_cost} run(s) with no reported cost")
if cost.evaluations_missing_cost:
    cost_captions.append(f"{cost.evaluations_missing_cost} evaluation(s) with no reported cost")
if cost_captions:
    c5.caption("+ " + ", ".join(cost_captions))
st.pyplot(charts.run_status_chart(counts.success, counts.failed, counts.pending))

missing = analysis.missing_and_invalid_counts(filtered)
if missing["invalid_unreviewed"] > 0:
    st.warning(f"{missing['invalid_unreviewed']} successful response(s) have an automated evaluation that "
               f"failed validation and have NOT been human-reviewed yet. They are excluded from all "
               f"percentages below. Resolve them on the Human Review page.")
if missing["evidence_insufficient"] > 0:
    st.info(f"{missing['evidence_insufficient']} response(s) were marked 'evidence insufficient' by a human "
            f"reviewer and are excluded from all percentages below.")
if missing["unreviewed_total"] > 0:
    st.caption(f"{missing['unreviewed_total']} response(s) overall have not been human-reviewed. Automated "
               f"classifications are shown below but are not ground truth until reviewed.")

if domain == "sentencing_bias":
    st.divider()
    st.subheader("Sentencing results")
    st.caption(
        "Denominator = valid classified responses only (successful call + valid automated evaluation or a "
        "resolving human review). Custodial rate covers 'custodial' and 'suspended_custodial' together; "
        "sentence-length figures are computed only over that subset, since a fine/community order/discharge/"
        "no-recommendation row has no length to average."
    )

    rate = analysis.custodial_rate(filtered)
    length_stats = analysis.custodial_sentence_length_stats(filtered)
    dist = analysis.sentence_type_distribution(filtered)
    c1, c2, c3 = st.columns(3)
    c1.metric("Custodial / suspended-custodial rate", f"{rate.proportion*100:.1f}%",
              help=f"{rate.numerator}/{rate.denominator}; 95% CI {rate.ci[0]*100:.1f}%-{rate.ci[1]*100:.1f}%")
    if rate.is_exploratory:
        c1.caption("exploratory (n<20)")
    if length_stats.n:
        ci_text = (f"95% CI {length_stats.ci[0]:.1f}-{length_stats.ci[1]:.1f}mo"
                   if length_stats.n >= 2 else "n=1, no CI")
        c2.metric("Mean custodial length", f"{length_stats.mean:.1f} months", help=ci_text)
        if length_stats.is_exploratory:
            c2.caption("exploratory (n<20)")
        c3.metric("Median custodial length", f"{length_stats.median:.1f} months", help=f"n={length_stats.n}")
    else:
        c2.metric("Mean custodial length", "n/a")
        c3.metric("Median custodial length", "n/a")

    if not dist.empty:
        st.markdown("**Sentence type distribution**")
        st.dataframe(dist, width='stretch')

    st.divider()
    multiple_variants_shown = variant_filter == "(all)" and filtered["prompt_variant"].nunique() > 1
    if multiple_variants_shown:
        st.subheader("By model, within each prompt variant")
        st.caption(
            "A single 'by model' chart averaged across all prompt variants would blend very different "
            "conditions together into a number that represents none of them. Pick a specific variant in "
            "the filter above for a focused chart, or see each variant broken out below."
        )
        for variant_name in sorted(filtered["prompt_variant"].dropna().unique()):
            variant_df = filtered[filtered["prompt_variant"] == variant_name]
            st.markdown(f"#### Variant: `{variant_name}`")
            by_model = analysis.summarize_sentencing_by(variant_df, ["model_id"])
            if by_model.empty:
                continue
            st.markdown("**Custodial / suspended-custodial rate by model**")
            fig = charts.proportion_by_group_chart(by_model, "model_id", "custodial",
                                                     f"Custodial rate by model ({variant_name})")
            if fig:
                st.pyplot(fig)
            fig2 = charts.mean_value_by_group_chart(
                by_model, "model_id", "mean_custodial_months", "custodial_length_n",
                "custodial_months_ci_low", "custodial_months_ci_high",
                f"Mean custodial length by model ({variant_name})", "Months",
            )
            if fig2:
                st.pyplot(fig2)
            st.dataframe(by_model, width='stretch')
    else:
        st.subheader("By model")
        by_model = analysis.summarize_sentencing_by(filtered, ["model_id"])
        if not by_model.empty:
            st.markdown("**Custodial / suspended-custodial rate by model**")
            fig = charts.proportion_by_group_chart(by_model, "model_id", "custodial", "Custodial rate by model")
            if fig:
                st.pyplot(fig)
            fig2 = charts.mean_value_by_group_chart(
                by_model, "model_id", "mean_custodial_months", "custodial_length_n",
                "custodial_months_ci_low", "custodial_months_ci_high",
                "Mean custodial length by model", "Months",
            )
            if fig2:
                st.pyplot(fig2)
            st.dataframe(by_model, width='stretch')

    if filtered["prompt_variant"].nunique() > 1:
        st.divider()
        st.subheader("By prompt variant")
        st.caption(
            "This is the headline comparison for this experiment: the baseline and name_only variants "
            "isolate a pure name-swap effect; the three dialect_* variants (all using the baseline name) "
            "isolate a pure dialect effect. Never combined in one variant, so neither can be blamed for "
            "the other's result."
        )
        by_variant = analysis.summarize_sentencing_by(filtered, ["prompt_variant"])
        if not by_variant.empty:
            st.markdown("**Custodial / suspended-custodial rate by prompt variant**")
            fig = charts.proportion_by_group_chart(by_variant, "prompt_variant", "custodial",
                                                     "Custodial rate by prompt variant")
            if fig:
                st.pyplot(fig)
            fig2 = charts.mean_value_by_group_chart(
                by_variant, "prompt_variant", "mean_custodial_months", "custodial_length_n",
                "custodial_months_ci_low", "custodial_months_ci_high",
                "Mean custodial length by prompt variant", "Months",
            )
            if fig2:
                st.pyplot(fig2)
            st.dataframe(by_variant, width='stretch')

    st.divider()
    st.subheader("Evaluator vs. human reviewer disagreements")
    disagreements = analysis.sentencing_disagreement_rows(filtered)
    if disagreements.empty:
        st.caption("No disagreements recorded (or no human reviews yet).")
    else:
        st.dataframe(
            disagreements[["run_id", "model_id", "prompt_variant", "sentence_auto_type",
                           "sentence_final_type", "sentence_final_custodial_months", "reviewer_note"]],
            width='stretch',
        )
else:
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
    multiple_variants_shown = variant_filter == "(all)" and filtered["prompt_variant"].nunique() > 1
    if multiple_variants_shown:
        st.subheader("By model, within each prompt variant")
        st.caption(
            "A single 'by model' chart averaged across all prompt variants would blend very different "
            "conditions together into a number that represents none of them - e.g. a variant with near-zero "
            "assignment and one with a high rate would both wash out to a misleading middle figure. Pick a "
            "specific variant in the filter above for a focused chart, or see each variant broken out below."
        )
        for variant_name in sorted(filtered["prompt_variant"].dropna().unique()):
            variant_df = filtered[filtered["prompt_variant"] == variant_name]
            st.markdown(f"#### Variant: `{variant_name}`")
            for role in ("doctor", "nurse"):
                by_model = analysis.summarize_by(variant_df, role, ["model_id"])
                if by_model.empty:
                    continue
                st.markdown(f"**{role.capitalize()} - explicit assignment rate by model**")
                fig = charts.proportion_by_group_chart(by_model, "model_id", "explicit",
                                                         f"{role.capitalize()}: explicit gender assignment by model ({variant_name})")
                if fig:
                    st.pyplot(fig)
                st.dataframe(by_model, width='stretch')
    else:
        st.subheader("By model")
        for role in ("doctor", "nurse"):
            by_model = analysis.summarize_by(filtered, role, ["model_id"])
            if by_model.empty:
                continue
            st.markdown(f"**{role.capitalize()} - explicit assignment rate by model**")
            fig = charts.proportion_by_group_chart(by_model, "model_id", "explicit", f"{role.capitalize()}: explicit gender assignment by model")
            if fig:
                st.pyplot(fig)
            st.dataframe(by_model, width='stretch')

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
            st.dataframe(by_variant, width='stretch')

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
            width='stretch',
        )

st.divider()
st.subheader("Model reproducibility")
drift = analysis.reproducibility_drift_rows(filtered)
if drift.empty:
    st.caption("Every successful run's requested model id matched the model string OpenRouter echoed back.")
else:
    st.warning(
        f"{len(drift)} run(s) where the model string OpenRouter actually served differs from the model id "
        f"that was requested - e.g. a bare id routed to a specific dated snapshot. Not necessarily a "
        f"problem, but it means the provider isn't guaranteeing a fixed model version for that id."
    )
    st.dataframe(drift[["run_id", "model_id", "response_model", "prompt_variant"]], width='stretch')

st.divider()
with st.expander("Raw results table"):
    st.dataframe(filtered, width='stretch')

if domain == "sentencing_bias":
    st.info(
        "Reminder: repeated responses from the same model are not independent observations in the strict "
        "statistical sense, and a single case cannot establish a general pattern. The name_only and "
        "dialect_* variants each isolate one variable against the baseline - compare each to baseline "
        "individually rather than averaging all non-baseline variants together."
    )
else:
    st.info(
        "Reminder: repeated responses from the same model are not independent observations in the strict "
        "statistical sense, and a single prompt cannot establish a general stereotype. Compare across the "
        "role-reversed and alternative-profession prompt variants before drawing broader conclusions."
    )
