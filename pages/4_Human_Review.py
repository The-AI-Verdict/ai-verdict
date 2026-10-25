from __future__ import annotations

import streamlit as st

from core import db, schema, ui_common

st.set_page_config(page_title="Human Review - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Human Review")
st.caption(
    "Automated evaluator output is never treated as ground truth. Confirm or correct each classification; "
    "the original automated result is always preserved alongside your revision."
)

experiment = ui_common.experiment_selector()
if experiment is None:
    st.stop()
exp_id = experiment["id"]

with db.get_connection() as conn:
    runs = [r for r in db.list_runs(conn, exp_id) if r["status"] == "success"]
    prompt_versions = {pv["id"]: pv for pv in db.list_prompt_versions(conn, exp_id)}
    reviews_by_run = {}
    evaluations_by_run = {}
    for r in runs:
        reviews_by_run[r["id"]] = db.get_latest_human_review(conn, r["id"])
        evaluations_by_run[r["id"]] = db.get_latest_evaluation_for_run(conn, r["id"])

if not runs:
    st.info("No successful runs to review yet.")
    st.stop()

only_unreviewed = st.checkbox("Show only unreviewed", value=True)
candidate_runs = [r for r in runs if (reviews_by_run[r["id"]] is None)] if only_unreviewed else runs
unreviewed_count = sum(1 for r in runs if reviews_by_run[r["id"]] is None)
st.caption(f"{unreviewed_count} of {len(runs)} successful runs are unreviewed.")

if not candidate_runs:
    st.success("Nothing left to review with the current filter.")
    st.stop()

run_ids = [r["id"] for r in candidate_runs]
idx_key = f"review_idx_{exp_id}_{only_unreviewed}"
if idx_key not in st.session_state:
    st.session_state[idx_key] = 0
st.session_state[idx_key] = min(st.session_state[idx_key], len(run_ids) - 1)

nav1, nav2, nav3 = st.columns([1, 2, 1])
with nav1:
    if st.button("<- Previous", disabled=st.session_state[idx_key] == 0):
        st.session_state[idx_key] -= 1
with nav3:
    if st.button("Next ->", disabled=st.session_state[idx_key] >= len(run_ids) - 1):
        st.session_state[idx_key] += 1
with nav2:
    st.write(f"Reviewing {st.session_state[idx_key] + 1} of {len(run_ids)}")

current_run_id = run_ids[st.session_state[idx_key]]
run = next(r for r in candidate_runs if r["id"] == current_run_id)
pv = prompt_versions.get(run["prompt_version_id"], {})
evaluation = evaluations_by_run[current_run_id]
existing_review = reviews_by_run[current_run_id]

st.divider()
st.subheader(f"Run #{run['id']} - `{run['model_id']}` - variant `{pv.get('variant_name')}` - rep {run['run_number']}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Original prompt**")
    st.code(pv.get("prompt_text", ""))
with c2:
    st.markdown("**Raw model response**")
    st.code(run.get("raw_response", ""))

st.markdown("**Automated evaluation**")
if evaluation is None:
    st.warning("No evaluation recorded for this run yet.")
elif not evaluation["is_valid"]:
    st.error(f"Automated evaluation is INVALID and needs human judgement. Validation error: "
             f"{evaluation['validation_errors']}")
else:
    ec1, ec2 = st.columns(2)
    with ec1:
        st.write("Doctor:", evaluation["doctor_gender_classification"], "/", evaluation["doctor_assignment_type"])
        st.caption(f"Quote: {evaluation['doctor_quote'] or '(none)'}")
        st.caption(evaluation["doctor_reasoning"] or "")
    with ec2:
        st.write("Nurse:", evaluation["nurse_gender_classification"], "/", evaluation["nurse_assignment_type"])
        st.caption(f"Quote: {evaluation['nurse_quote'] or '(none)'}")
        st.caption(evaluation["nurse_reasoning"] or "")
    if evaluation["uncertain"]:
        st.info(f"Evaluator flagged uncertainty: {evaluation['uncertain_reason']}")

st.divider()
st.subheader("Your review")

classification_options = [c.value for c in schema.GenderClassification]
assignment_options = [a.value for a in schema.AssignmentType]


def default_index(options: list[str], value: str | None, fallback: str) -> int:
    if value and value in options:
        return options.index(value)
    return options.index(fallback)


prior_doctor_class = (existing_review or {}).get("final_doctor_classification") or (evaluation or {}).get("doctor_gender_classification")
prior_doctor_assign = (existing_review or {}).get("final_doctor_assignment_type") or (evaluation or {}).get("doctor_assignment_type")
prior_nurse_class = (existing_review or {}).get("final_nurse_classification") or (evaluation or {}).get("nurse_gender_classification")
prior_nurse_assign = (existing_review or {}).get("final_nurse_assignment_type") or (evaluation or {}).get("nurse_assignment_type")

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**Doctor**")
    doctor_class = st.selectbox("Gender classification", classification_options,
                                 index=default_index(classification_options, prior_doctor_class, "unspecified"),
                                 key=f"doctor_class_{current_run_id}")
    doctor_assign = st.selectbox("Assignment type", assignment_options,
                                  index=default_index(assignment_options, prior_doctor_assign, "none"),
                                  key=f"doctor_assign_{current_run_id}")
with rc2:
    st.markdown("**Nurse**")
    nurse_class = st.selectbox("Gender classification", classification_options,
                                index=default_index(classification_options, prior_nurse_class, "unspecified"),
                                key=f"nurse_class_{current_run_id}")
    nurse_assign = st.selectbox("Assignment type", assignment_options,
                                 index=default_index(assignment_options, prior_nurse_assign, "none"),
                                 key=f"nurse_assign_{current_run_id}")

evidence_insufficient = st.checkbox("Mark evidence as insufficient for a confident classification",
                                     value=bool((existing_review or {}).get("evidence_insufficient", False)),
                                     key=f"insufficient_{current_run_id}")
reviewer_name = st.text_input("Reviewer name", value=(existing_review or {}).get("reviewer_name", "") or
                               st.session_state.get("reviewer_name_default", ""),
                               key=f"reviewer_name_{current_run_id}")
reviewer_note = st.text_area("Reviewer note (optional)", value=(existing_review or {}).get("reviewer_note", "") or "",
                              key=f"reviewer_note_{current_run_id}")

if existing_review:
    st.caption(f"Previously reviewed at {existing_review['created_at']} by "
               f"'{existing_review['reviewer_name'] or '(unnamed)'}'. Submitting again records a new revision; "
               f"the prior review stays in history.")

if st.button("Save review", type="primary"):
    st.session_state["reviewer_name_default"] = reviewer_name
    with db.get_connection() as conn:
        db.create_human_review(
            conn, current_run_id, evaluation["id"] if evaluation else None, reviewer_name,
            doctor_class, doctor_assign, nurse_class, nurse_assign, evidence_insufficient, reviewer_note,
        )
    st.success("Review saved.")
    if not only_unreviewed and st.session_state[idx_key] < len(run_ids) - 1:
        # When filtering to unreviewed-only, the just-reviewed run drops out of
        # the list on rerun and the next run slides into this same index - so
        # advancing here too would skip it. Only advance when the list doesn't
        # shrink (i.e. showing all runs, where the reviewed run stays in place).
        st.session_state[idx_key] += 1
    st.rerun()
