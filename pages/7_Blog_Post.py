from __future__ import annotations

import streamlit as st

from core import db, export, sentencing_export, ui_common

st.set_page_config(page_title="Blog Post - The AI Verdict", layout="wide")
ui_common.bootstrap()

st.title("Blog Post")
st.caption(
    "A Medium-ready draft generated from this experiment's actually recorded data - headline options, a "
    "dek, and full body copy, not just an outline. Every number and quote in it is sourced and traceable. "
    "This is a strong first draft, not a finished article: read it out loud before publishing and cut "
    "anything that doesn't sound like your own voice - that's the one thing no generator can do for you."
)

experiment = ui_common.experiment_selector()
if experiment is None:
    st.stop()
exp_id = experiment["id"]
safe_name = "".join(c if c.isalnum() else "_" for c in experiment["name"])[:40]

domain = experiment.get("evaluation_domain") or "gender_healthcare"
generator = sentencing_export.generate_blog_post if domain == "sentencing_bias" else export.generate_blog_post

with db.get_connection() as conn:
    rows_check = db.list_runs(conn, exp_id)
has_success = any(r["status"] == "success" for r in rows_check)

if not has_success:
    st.info("No successful runs yet for this experiment - go to Run Experiment first, then come back here.")
    st.stop()

if st.button("Generate / regenerate post", type="primary"):
    st.session_state[f"blog_{exp_id}"] = None

cache_key = f"blog_{exp_id}"
if cache_key not in st.session_state or st.session_state.get(cache_key) is None:
    with db.get_connection() as conn:
        st.session_state[cache_key] = generator(conn, exp_id)

post_text = st.session_state[cache_key]
word_count = export.word_count(post_text)
read_minutes = max(1, round(word_count / 220))

col1, col2, col3 = st.columns(3)
col1.metric("Word count", word_count)
col2.metric("Est. read time", f"{read_minutes} min")
col3.download_button("Download post.md", post_text, file_name=f"{safe_name}_blog_post.md", mime="text/markdown")

if word_count < 600:
    st.caption(
        "On the short side for Medium's typical long-form range (roughly 1200-2000 words / 6-9 min read) - "
        "that usually means this experiment doesn't have much contrast or per-model spread to write about "
        "yet. More models or a second prompt variant will give the generator more to say."
    )
elif word_count > 2400:
    st.caption("Longer than Medium's typical sweet spot - consider trimming a section that doesn't earn its place.")

st.divider()

view_mode = st.radio("View as", ["Formatted", "Plain text (for editing/copying into Medium)"], horizontal=True)
if view_mode == "Formatted":
    st.markdown(post_text)
else:
    st.text_area("Post", post_text, height=700, label_visibility="collapsed")

st.divider()
if domain == "sentencing_bias":
    st.caption(
        "How this is structured: if only one model ever recommends a custodial sentence, the post is built "
        "around that model's own variant-by-variant breakdown and says explicitly that the finding belongs "
        "to one model, not a cross-model pattern - it won't claim a broader effect the data doesn't show. "
        "Data-quality issues (excluded or still-pending responses) get their own honest paragraph rather "
        "than being silently smoothed over."
    )
else:
    st.caption(
        "How this is structured: with a prompt variant where models stayed almost entirely neutral and "
        "another where they were forced to commit, the post builds a two-part 'round one / round two' "
        "narrative around that real contrast, plus a section naming which specific models landed on which "
        "side when the data shows a genuine, consistent split - not every experiment will have one, and the "
        "post doesn't invent one when it isn't there. Data-quality issues (excluded or failed responses) get "
        "their own honest paragraph rather than being silently smoothed over."
    )
