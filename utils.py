# === utils.py ===
import os, re, json, ast, pdfplumber, stanza, tiktoken, difflib
from uuid import uuid4
from collections import Counter
from typing import List, Tuple
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import AzureOpenAI
import config

# === Setup ===
nlp = stanza.Pipeline('en', processors='tokenize', verbose=False)

embedder = AzureOpenAIEmbeddings(
    azure_deployment=config.DEPLOYMENT_NAME_EMBED,
    openai_api_key=config.OPENAI_API_KEY,
    azure_endpoint=config.AZURE_ENDPOINT,
    openai_api_version=config.OPENAI_API_VERSION
)

llm_client = AzureOpenAI(
    api_key=config.OPENAI_API_KEY,
    api_version=config.OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_ENDPOINT
)

reranker = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

# === Regex Helpers ===
def apply_pre_stanza_regex(text: str) -> str:
    # Replace a)/1)/A) → a---/1---/A---
    return re.sub(r'([a-zA-Z0-9])\)', r'\1---', text)

def restore_post_stanza_format(sentence: str) -> str:
    # Convert a---/1--- back to a)/1)
    return re.sub(r'([a-zA-Z0-9])---', r'\1)', sentence)

# === PDF/Text Cleaning ===
def clean_text_from_file(uploaded_file) -> Tuple[str, dict]:
    name = uploaded_file.name.lower()
    if name.endswith(".json"):
        data = json.load(uploaded_file)
        return json.dumps(data, indent=2), data
    elif name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            lines = []
            for page in pdf.pages:
                text = page.extract_text()
                if text: lines.extend(text.splitlines())
            raw_text = "\n".join(lines)
            return apply_pre_stanza_regex(raw_text), None
    elif name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
        return apply_pre_stanza_regex(text), None
    return "", None

# === Chunking ===
def semantic_chunk(text: str) -> List[str]:
    """Performs semantic chunking with sentence boundaries using LangChain."""
    doc = nlp(text)
    
    # Restore original bullet formatting after sentence splitting
    sentences = [restore_post_stanza_format(s.text.strip()) for s in doc.sentences if s.text.strip()]

    # 🔹 Print sentences before joining
    print("\n🔍 Sentences before boundary marker:\n")
    for i, s in enumerate(sentences):
        print(f"{i+1}. {s}\n")

    # Add sentence boundary markers
    joined = " @@@SENTENCE_BOUNDARY@@@ ".join(sentences)

    # 🔹 Print the joined text with sentence boundaries
    print("\n📌 Text after adding sentence boundary markers:\n")
    print(joined)
    print("\n" + "="*80 + "\n")

    chunker = SemanticChunker(
        embeddings=embedder,
        sentence_split_regex=r" @@@SENTENCE_BOUNDARY@@@ ",
        buffer_size=1,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=0.8,
    )
    chunks = [doc.page_content.strip() for doc in chunker.create_documents([joined])]
    return [chunk for chunk in chunks if chunk]

def prepare_chunks(text: str, min_chunk_chars: int = 200) -> List[str]:
    """Semantic chunking with better merge logic for tiny chunks."""
    print("📌 Semantic chunking...")
    sem_chunks = semantic_chunk(text)

    if not sem_chunks:
        return []

    merged_chunks = []
    buffer = ""

    for chunk in sem_chunks:
        chunk = chunk.strip()

        # If chunk is too short, buffer it
        if len(chunk) < min_chunk_chars:
            if merged_chunks:
                # Merge with the previous chunk
                merged_chunks[-1] += " " + chunk
            else:
                # No previous chunk: store it in buffer
                buffer = chunk
        else:
            if buffer:
                chunk = buffer + " " + chunk
                buffer = ""
            merged_chunks.append(chunk)

    # If any buffer remains at end, append it
    if buffer:
        if merged_chunks:
            merged_chunks[-1] += " " + buffer
        else:
            merged_chunks.append(buffer)

    # Output chunks for review
    for i, chunk in enumerate(merged_chunks):
        print("chunk number = ", i, "\n")
        print(chunk)
        print("\n\n")

    return merged_chunks


# === Qdrant I/O ===
def embed_and_store(chunks: List[str], file_id: str):
    if not qdrant.collection_exists(config.COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_VECTOR_SIZE, distance=Distance.COSINE)
        )
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=embedder.embed_query(chunk),
            payload={"text": chunk, "file_id": file_id, "chunk_index": idx}
        ) for idx, chunk in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=config.COLLECTION_NAME, points=points)

def retrieve_chunks_by_file(file_id: str) -> List[Tuple[str, int]]:
    results, _ = qdrant.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]),
        limit=10000
    )
    return [(pt.payload["text"], pt.payload["chunk_index"]) for pt in results]

# === Compare ===
def get_top_matches_for_rule(rule_text: str, doc_file_id: str, top_k: int = 5, threshold: float = 0.7) -> List[dict]:
    doc_chunks = retrieve_chunks_by_file(doc_file_id)
    doc_texts = [c[0] for c in doc_chunks]
    doc_vectors = reranker.encode(doc_texts)
    rule_vector = reranker.encode(rule_text)
    sims = [float(cos_sim(rule_vector, vec)) for vec in doc_vectors]

    # Filter by threshold
    filtered = [(i, sims[i]) for i in range(len(sims)) if sims[i] >= threshold]
    top_indices = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

    return [{"score": score, "text": doc_texts[i]} for i, score in top_indices]

# === LLM Comparison ===
def extract_topic_and_diff(rule_text: str, doc_chunks: List[str], rule_number: int = 0):
    doc_text = "".join(doc_chunks)

    prompt = f"""Role: You are an expert legal AI assistant specializing in reviewing Non-Disclosure Agreements (NDAs).
Your task is to review the provided NDA text and identify any statements that violate or contradict the "Rule".

Instructions:
Given this rule: {rule_text}
And this NDA clause: {doc_text}

For each identified violation, you must provide:
  1. The exact violating statement from the NDA.
  2. The start and end character indices of the violating statement within the provided NDA text.
  3. A concise explanation of why it violates the rule.

If no violations are found, state "No violations found."

Respond strictly in the following JSON format using double quotes only:
{{
  "violations": [
    {{
      "violating_statement": "The exact text of the statement that violates the rule.",
      "start_index": 123,
      "end_index": 145,
      "explanation": "Explanation of why this violates the rule."
    }}
  ],
  "summary": "Overall summary of findings: [e.g., 'One violation found regarding Rule 1.']"
}}"""

    try:
        content = llm_client.chat.completions.create(
            model=config.DEPLOYMENT_NAME_GPT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a compliance assistant. "
                        "Always return valid JSON using double quotes. "
                        "Only respond with JSON — no explanation outside of it."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        ).choices[0].message.content.strip()

        json_part = content[content.find("{"):content.rfind("}") + 1]
        result = json.loads(json_part)

        violations = result.get("violations", [])
        summary = result.get("summary", "")

        if not violations or (isinstance(violations, str) and "no violations" in violations.lower()):
            return {
                "Rule": f"Rule {rule_number}",
                "Topic": "NDA Review",
                "Compliant": "✅",
                "Key Issues Detected": "No violations found.",
            }

        formatted_violations = "\n".join([
            f"- **Violation**: \"{v['violating_statement']}\"\n  "
            f"**Location**: [{v['start_index']}–{v['end_index']}]\n  "
            f"**Explanation**: {v['explanation']}"
            for v in violations
        ])

        return {
            "Rule": f"Rule {rule_number}",
            "Topic": "NDA Review",
            "Compliant": "❌",
            "Key Issues Detected": formatted_violations + f"\n\n📌 {summary}",
            "Violations": violations,
            "Matched Text": doc_text
        }

    except Exception as e:
        print("LLM RAW RESPONSE (on error):", content if 'content' in locals() else 'N/A')
        return {
            "Rule": f"Rule {rule_number}",
            "Topic": "[Topic ERROR]",
            "Compliant": "❌",
            "Key Issues Detected": f"[LLM ERROR] {e}"
        }