"""Twin Mind — Learn Module

Extracts concepts from a session's transcript and visual observations,
generates structured wiki articles, and updates the knowledge index.

Embedding pipeline:
  - Each markdown file is chunked by ## section
  - Each section is embedded via cactus_embed and stored in cactus_index
  - Retrieval uses composite scoring:
      score = α * semantic_sim + γ * cognitive_engagement + δ * recency_decay
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

from cactus import (
    cactus_complete,
    cactus_embed,
    cactus_index_init,
    cactus_index_add,
    cactus_index_query,
    cactus_index_get,
    cactus_index_compact,
)

# Reference text for cognitive engagement scoring
_FOCUSED_REF = "focused engaged attentive concentrated deep work thinking"


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

def init_index(model, data_dir):
    """Initialise the vector index, inferring embedding dim from a test embed."""
    dim = len(cactus_embed(model, "init", True))
    index_dir = os.path.join(data_dir, "vector_index")
    os.makedirs(index_dir, exist_ok=True)
    index = cactus_index_init(index_dir, dim)
    return index


def chunk_markdown(text):
    """Split a markdown file into sections on ## headers."""
    sections = re.split(r'\n(?=## )', text)
    return [s.strip() for s in sections if s.strip()]


def _parse_timestamp(text):
    """Extract session timestamp from article Source section."""
    match = re.search(r'Session:\s*(\S+)', text)
    return match.group(1) if match else "unknown"


def build_wiki_index(model, index, data_dir):
    """Embed all existing wiki articles into the vector index. Returns next doc_id."""
    doc_id = 0
    for filepath in sorted(Path(data_dir).glob("*.md")):
        if filepath.name == "index.md":
            continue
        text = filepath.read_text()
        timestamp = _parse_timestamp(text)
        for section in chunk_markdown(text):
            emb = cactus_embed(model, section, True)
            cactus_index_add(
                index, [doc_id], [section], [emb],
                [f"{filepath.name}|{timestamp}"]
            )
            doc_id += 1
    cactus_index_compact(index)
    print(f"  Indexed {doc_id} sections from existing wiki articles")
    return doc_id


def add_article_to_index(model, index, doc_id, filepath, timestamp):
    """Embed a newly written article and add it to the index."""
    text = Path(filepath).read_text()
    for section in chunk_markdown(text):
        emb = cactus_embed(model, section, True)
        cactus_index_add(
            index, [doc_id], [section], [emb],
            [f"{os.path.basename(filepath)}|{timestamp}"]
        )
        doc_id += 1
    cactus_index_compact(index)
    return doc_id


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _cognitive_score(model, visual_summary):
    """
    Cosine similarity between the session's visual summary and a focused
    reference embedding. Both vectors are already unit-normalised by cactus_embed.
    """
    ref_emb = cactus_embed(model, _FOCUSED_REF, True)
    vis_emb = cactus_embed(model, visual_summary or "neutral", True)
    return float(sum(a * b for a, b in zip(ref_emb, vis_emb)))


def _recency_score(timestamp_str, decay=0.1):
    """Exponential decay: score = exp(-decay * days_since_session)."""
    try:
        session_date = datetime.strptime(timestamp_str[:10], "%Y-%m-%d")
        days = max(0, (datetime.now() - session_date).days)
        return math.exp(-decay * days)
    except Exception:
        return 0.5


def retrieve_and_score(model, index, query_text, visual_summary,
                       top_k=5, alpha=0.6, gamma=0.2, delta=0.2):
    """
    Retrieve relevant wiki sections and apply composite scoring:
      score = α * semantic_sim + γ * cognitive_engagement + δ * recency_decay
    Returns deduplicated list of top-k results sorted by score.
    """
    query_emb = cactus_embed(model, query_text, True)
    raw = cactus_index_query(index, query_emb, json.dumps({"top_k": top_k * 3}))
    results = json.loads(raw)["results"]

    cog = _cognitive_score(model, visual_summary)

    scored = []
    for r in results:
        doc_raw = json.loads(cactus_index_get(index, [r["id"]]))["results"][0]
        meta = doc_raw.get("metadata", "unknown|unknown")
        parts = (meta + "|unknown").split("|")
        filename, timestamp = parts[0], parts[1]
        rec = _recency_score(timestamp)
        final = alpha * r["score"] + gamma * cog + delta * rec
        scored.append({
            "filename": filename,
            "text": doc_raw["document"],
            "semantic_sim": r["score"],
            "cognitive": cog,
            "recency": rec,
            "score": final,
        })

    # deduplicate by filename, keep highest score per file
    seen = {}
    for s in sorted(scored, key=lambda x: x["score"], reverse=True):
        if s["filename"] not in seen:
            seen[s["filename"]] = s

    return list(seen.values())[:top_k]


# ---------------------------------------------------------------------------
# Knowledge extraction (unchanged logic, now index-aware)
# ---------------------------------------------------------------------------

def slugify(title):
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def read_existing_index(data_dir):
    index_path = os.path.join(data_dir, "index.md")
    if not os.path.exists(index_path):
        return "", []
    with open(index_path, "r") as f:
        content = f.read()
    articles = re.findall(r"\[([^\]]+)\]\(([^)]+\.md)\)", content)
    return content, articles


def extract_concepts(model, transcript, visual_summary, existing_articles):
    existing_names = ", ".join(name for name, _ in existing_articles) if existing_articles else "none yet"
    prompt = (
        f"You observed a session with this transcript and visual context.\n\n"
        f"Transcript: \"{transcript}\"\n\n"
        f"Visual observations: {visual_summary}\n\n"
        f"Existing wiki articles: {existing_names}\n\n"
        f"Identify the key concepts discussed or shown in this session.\n"
        f"For each concept, provide:\n"
        f"- title: a clear, concise concept name\n"
        f"- summary: one sentence explaining the concept\n"
        f"- related: list of existing article titles this concept relates to\n\n"
        f"Respond in JSON format: [{{\"title\": \"...\", \"summary\": \"...\", \"related\": [\"...\"]}}]\n"
        f"Only return the JSON array, nothing else."
    )
    messages = [{"role": "user", "content": prompt}]
    response = cactus_complete(model, json.dumps(messages), json.dumps({"max_tokens": 500}), None, None)
    raw = json.loads(response)["response"]
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        print(f"  [WARN] Could not parse concepts: {raw[:200]}")
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        print(f"  [WARN] Invalid JSON in concepts: {match.group()[:200]}")
        return []


def generate_article(model, concept, transcript, visual_summary, timestamp, existing_articles):
    related_names = concept.get("related", [])
    related_links = [
        f"- [{name}]({filename})"
        for name, filename in existing_articles
        if name in related_names
    ]
    related_section = "\n".join(related_links) if related_links else "- (No related articles yet)"

    prompt = (
        f"Write a structured wiki article about: {concept['title']}\n\n"
        f"Context — this was learned from a session:\n"
        f"Transcript: \"{transcript}\"\n"
        f"Visual context: {visual_summary}\n\n"
        f"Write the article in this exact markdown format:\n\n"
        f"# {concept['title']}\n\n"
        f"## Overview\n"
        f"(Explain the concept in plain language, accessible to a junior researcher)\n\n"
        f"## Key Ideas\n"
        f"(Main points and evidence from the session)\n\n"
        f"## Source\n"
        f"- Session: {timestamp}\n"
        f"- Transcript excerpt: (relevant quote)\n"
        f"- Visual context: (what was observed)\n\n"
        f"## Related Concepts\n"
        f"{related_section}\n\n"
        f"Only return the markdown article, nothing else."
    )
    messages = [{"role": "user", "content": prompt}]
    response = cactus_complete(model, json.dumps(messages), json.dumps({"max_tokens": 800}), None, None)
    return json.loads(response)["response"]


def save_article(data_dir, concept, article_content):
    slug = slugify(concept["title"])
    filename = f"{slug}.md"
    filepath = os.path.join(data_dir, filename)
    with open(filepath, "w") as f:
        f.write(article_content)
    print(f"  Saved article: {filename}")
    return filename, slug


def update_index(data_dir, new_entries):
    index_path = os.path.join(data_dir, "index.md")
    content = open(index_path).read() if os.path.exists(index_path) else "# Concept Wiki Index\n\n"
    divider_pos = content.rfind("---")
    section = "\n## Learned from Sessions\n\n" if "Learned from Sessions" not in content else ""
    entries_text = section + "\n".join(
        f"- [{title}]({filename}) — {summary}"
        for title, filename, summary in new_entries
    ) + "\n"
    if divider_pos != -1 and "Learned from Sessions" not in content:
        content = content[:divider_pos] + entries_text + "\n" + content[divider_pos:]
    else:
        content = content.rstrip() + "\n" + entries_text
    with open(index_path, "w") as f:
        f.write(content)
    print(f"  Updated index.md with {len(new_entries)} new entries")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def learn_from_session(model, transcript, visual_summary, timestamp, data_dir,
                       index=None, doc_id=0):
    """
    Full learning pipeline: retrieve related context, extract concepts,
    generate articles, update index and vector store.

    Returns (new_entries, updated_doc_id).
    """
    print("\n=== LEARNING FROM SESSION ===\n")

    # Use scored retrieval from vector index if available
    if index and transcript:
        print("Retrieving related wiki articles...")
        scored = retrieve_and_score(model, index, transcript, visual_summary)
        existing_articles = [(s["filename"].replace(".md", "").replace("-", " ").title(),
                              s["filename"]) for s in scored]
        for s in scored:
            print(f"  {s['filename']}  score={s['score']:.3f} "
                  f"(sem={s['semantic_sim']:.2f} cog={s['cognitive']:.2f} rec={s['recency']:.2f})")
        print()
    else:
        _, existing_articles = read_existing_index(data_dir)

    print("Extracting concepts...")
    concepts = extract_concepts(model, transcript, visual_summary, existing_articles)

    if not concepts:
        print("  No new concepts identified.")
        return [], doc_id

    print(f"  Found {len(concepts)} concepts: {', '.join(c['title'] for c in concepts)}\n")

    new_entries = []
    for concept in concepts:
        print(f"Generating article: {concept['title']}...")
        article = generate_article(model, concept, transcript, visual_summary, timestamp, existing_articles)
        filename, _ = save_article(data_dir, concept, article)
        new_entries.append((concept["title"], filename, concept.get("summary", "")))

        # add new article to vector index immediately
        if index:
            filepath = os.path.join(data_dir, filename)
            doc_id = add_article_to_index(model, index, doc_id, filepath, timestamp)

    if new_entries:
        update_index(data_dir, new_entries)

    print(f"\n=== LEARNED {len(new_entries)} NEW CONCEPTS ===\n")
    return new_entries, doc_id
