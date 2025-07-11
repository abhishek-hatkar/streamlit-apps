import streamlit as st
from utils import (
    clean_text_from_file,
    prepare_chunks,
    embed_and_store,
    retrieve_chunks_by_file,
    get_top_matches_for_rule,
    extract_topic_and_diff,
    prepare_chunks,
)
# === app.py ===
import re
import os, json
from uuid import uuid4
from config import COLLECTION_NAME
import pandas as pd
import difflib

def remove_mark_tags(text: str) -> str:
    return re.sub(r'</?mark.*?>', '', text)



st.set_page_config(layout="wide")
st.markdown("""
    <style>
    mark {
        background-color: #ffcdd2;  /* light red */
        padding: 0.2em 0.3em;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)
st.title("📜 Rule-Based Document Validator")

# --- File Uploads ---
st.sidebar.header("📄 Upload Files")
rules_file = st.sidebar.file_uploader("Upload Rules File (JSON or PDF)", type=["json", "pdf"])
doc_file = st.sidebar.file_uploader("Upload Document to Check", type=["pdf", "txt"])

if "rule_dict" not in st.session_state:
    st.session_state.rule_dict = {}

# --- Process Rules ---
if rules_file and st.sidebar.button("✅ Process Rules"):
    rules_text, rules_json = clean_text_from_file(rules_file)
    if rules_json:
        st.session_state.rule_dict = rules_json["Rules"][0]
    else:
        st.session_state.rule_dict = {f"Rule {i+1}": rule for i, rule in enumerate(prepare_chunks(rules_text))}

    file_id = f"rules_{uuid4().hex[:8]}"
    for rule_name, rule_text in st.session_state.rule_dict.items():
        chunks = prepare_chunks(rule_text)
        embed_and_store(chunks, file_id + "_" + rule_name)

    st.success(f"Rules embedded and stored for {len(st.session_state.rule_dict)} rule(s).")

# --- Process Document ---
if doc_file and st.sidebar.button("✅ Process Document"):
    doc_text, _ = clean_text_from_file(doc_file)
    file_id = f"doc_{uuid4().hex[:8]}"
    chunks = prepare_chunks(doc_text)
    embed_and_store(chunks, file_id)
    st.session_state.doc_file_id = file_id
    st.session_state.doc_text = doc_text
    st.success(f"Document '{doc_file.name}' processed and stored.")

# --- Rule Selection ---
if st.session_state.get("rule_dict"):
    st.subheader("🎯 Select Rule(s) to Apply")
    selected_rules = st.multiselect("Choose rule(s) to check:", list(st.session_state.rule_dict.keys()))

    if selected_rules and st.button("Check Rule Compliance 🔍"):
        if not st.session_state.get("doc_file_id"):
            st.warning("Please upload and process a document first.")
        else:
            st.info("Processing selected rule(s)... please wait.")
            with st.spinner("Matching rules against document..."):
                results = []
                for idx, rule_name in enumerate(selected_rules, 1):
                    rule_text = st.session_state.rule_dict[rule_name]
                    rule_chunks = prepare_chunks(rule_text)
                    for chunk in rule_chunks:
                        top_matches = get_top_matches_for_rule(chunk, st.session_state.doc_file_id, top_k=5, threshold=0.2)
                    top_texts = [m["text"] for m in top_matches]
                    match_result = extract_topic_and_diff(chunk, top_texts, rule_number=idx)
                    match_result.update({
                        "Rule Name": rule_name,
                        "Similarity": f"{top_matches[0]['score'] * 100:.2f}%",
                        "Rule Segment": chunk,
                        "Matched Segment": "\n\n---\n\n".join(
                            [f"[{m['score']*100:.2f}% match]\n{m['text']}" for m in top_matches]
                        ),
                    })
                    results.append(match_result)

            # --- Display Detailed Results ---
            st.subheader("🔍 Detailed Rule Analysis")
            for idx, r in enumerate(results, 1):
                st.markdown(f"### Rule Match #{idx}")
                with st.expander(f"📘 Rule Segment — {r.get('Rule Name', 'Unnamed Rule')}"):
                    st.write(r["Rule Segment"], score=r["Similarity"])
                with st.expander("📄 Matched Document Segment(s)"):
                    chunks = r["Matched Segment"].split("\n\n---\n\n")
                    violations = r.get("Violations", [])

                    highlighted_chunks = []
                    for chunk in chunks:
                        chunk_highlighted = remove_mark_tags(chunk)
                        sentences = re.split(r'(?<=[.!?])\s+', chunk_highlighted)
                        highlights = []

                        for v in violations:
                            statement = v.get("violating_statement", "").strip()
                            explanation = v.get("explanation", "").replace('"', "'")
                            if not statement:
                                continue

                            best_sentence = None
                            best_ratio = 0
                            best_index = 0
                            for i, s in enumerate(sentences):
                                ratio = difflib.SequenceMatcher(None, s.lower(), statement.lower()).ratio()
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_sentence = s
                                    best_index = i

                            if best_sentence and best_ratio > 0.5:
                                start_idx = max(0, best_index - 1)
                                end_idx = min(len(sentences), best_index + 2)
                                for j in range(start_idx, end_idx):
                                    match = re.search(re.escape(sentences[j]), chunk_highlighted, flags=re.IGNORECASE)
                                    if match:
                                        highlights.append((match.start(), match.end(), explanation))

                        # Remove overlaps by combining overlapping spans
                        merged = []
                        for start, end, explanation in sorted(highlights):
                            if not merged:
                                merged.append((start, end, explanation))
                            else:
                                last_start, last_end, last_exp = merged[-1]
                                if start <= last_end:
                                    merged[-1] = (last_start, max(end, last_end), last_exp)
                                else:
                                    merged.append((start, end, explanation))

                        # Apply highlights in reverse order
                        for start, end, explanation in sorted(merged, reverse=True):
                            chunk_highlighted = (
                                chunk_highlighted[:start]
                                + f'<mark title="{explanation}">' + chunk_highlighted[start:end] + "</mark>"
                                + chunk_highlighted[end:]
                            )

                        highlighted_chunks.append(chunk_highlighted)

                    st.markdown("\n\n---\n\n".join(highlighted_chunks), unsafe_allow_html=True)

                with st.expander("Explanation of Differences"):
                    violations = r.get("Violations", [])
                    summary = r.get("Key Issues Detected", "").split("📌")[-1].strip()

                    if violations:
                        for v in violations:
                            st.markdown(f"""
                **Violation:**  
                {v.get("violating_statement", "").strip()}

                **Location:**  
                [{v.get("start_index", "")}–{v.get("end_index", "")}]

                **Explanation:**  
                {v.get("explanation", "").strip()}

                ---
                """)
                        st.markdown(f"📌 **Summary:** {summary}")
                    else:
                        st.markdown(r["Key Issues Detected"])

            # --- Display Summary Table ---
            st.subheader("📊 Compliance Summary Table")
            summary_table = pd.DataFrame([
                {
                    "Rule": r["Rule"],
                    "Topic": r["Topic"],
                    "Compliant": r["Compliant"],
                    "Key Issues Detected": r["Key Issues Detected"]
                }
                for r in results
            ])
            st.dataframe(summary_table, use_container_width=True)
