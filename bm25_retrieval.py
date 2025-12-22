import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import re

# -------------------------------
# CONFIG
# -------------------------------
METADATA_FILE = "datasets_landing_metadata.json"
QUERIES_BY_TOPIC_FILE = "queries_by_topic.json"
RESULTS_FILE = "bm25_retrieval_results_by_topic.json"
TOP_K = 5  # number of datasets to retrieve per query

# -------------------------------
# Helper functions
# -------------------------------
def tokenize(text):
    """Simple tokenizer: lowercase and split on non-word characters."""
    return re.findall(r"\w+", text.lower())

def save_results(results, filename=RESULTS_FILE):
    """Incremental save of results."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# -------------------------------
# Load metadata
# -------------------------------
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    all_metadata_by_topic = json.load(f)

all_datasets = []
for topic, datasets in all_metadata_by_topic.items():
    for ds in datasets:
        ds["topic"] = topic
        ds["text"] = f"{ds.get('title', '')} {ds.get('description', '')}"
        all_datasets.append(ds)

print(f"Total datasets: {len(all_datasets)}")

# -------------------------------
# Load queries by topic
# -------------------------------
with open(QUERIES_BY_TOPIC_FILE, "r", encoding="utf-8") as f:
    queries_by_topic = json.load(f)

# -------------------------------
# Check for already saved results
# -------------------------------
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        retrieved_results = json.load(f)
else:
    retrieved_results = {}

# -------------------------------
# Prepare BM25
# -------------------------------
corpus = [tokenize(ds["text"]) for ds in all_datasets]
bm25 = BM25Okapi(corpus)

# -------------------------------
# Retrieval loop
# -------------------------------
for topic, queries in tqdm(queries_by_topic.items(), desc="Processing topics"):
    if topic not in retrieved_results:
        retrieved_results[topic] = {}

    for query in tqdm(queries, desc=f"Processing queries in {topic}", leave=False):
        if query in retrieved_results[topic]:
            continue  # skip already processed query

        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-TOP_K:][::-1]

        retrieved = []
        for idx in top_indices:
            ds = all_datasets[idx]
            retrieved.append({
                "id": ds["id"],
                "title": ds.get("title"),
                "description": ds.get("description"),
                "topic": ds.get("topic"),
                "url": ds.get("url"),
                "score": float(scores[idx])
            })

        retrieved_results[topic][query] = retrieved

        # Immediate save after each query
        save_results(retrieved_results)

print(f"✅ Retrieval complete! Results saved to {RESULTS_FILE}")
