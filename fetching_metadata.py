import requests
import pandas as pd
import json
import time

# Base API
SEARCH_API = "https://data.europa.eu/api/hub/search/search"

# Storage
all_metadata = []

# We want ~1500 datasets -> 15 pages of 100 each
for page in range(1):
    print(f"Fetching page {page}...")
    url = f"{SEARCH_API}?page={page}&limit=10"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    
    results = data.get("result", {}).get("results", [])
    if not results:
        print(f"⚠️ No results found on page {page}, response keys: {list(data.keys())}")
        continue

    for item in results:
        # --- Handle multilingual fields safely ---
        def get_text(field):
            val = item.get(field)
            if isinstance(val, dict):
                return val.get("en", "")
            return val or ""

        # --- Handle publisher safely ---
        pub = item.get("publisher")
        if isinstance(pub, dict):
            if isinstance(pub.get("name"), dict):
                publisher = pub.get("name", {}).get("en", "")
            else:
                publisher = pub.get("name", "")
        else:
            publisher = pub or ""

        # --- Build metadata dict ---
        meta = {
            "id": item.get("id"),
            "title": get_text("title"),
            "description": get_text("description"),
            "publisher": publisher,
            "issued": item.get("issued"),
            "modified": item.get("modified"),
            "landingPage": item.get("landingPage", [None])[0] if item.get("landingPage") else None
        }
        all_metadata.append(meta)

    time.sleep(0.2)  # polite delay

# Save all metadata into a JSON file
with open("datasets_landing_metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, indent=2, ensure_ascii=False)

# Also save as CSV for quick exploration
#df = pd.DataFrame(all_metadata)
#df.to_csv("datasets_landing_metadata.csv", index=False, encoding="utf-8")

print("✅ Done! Saved to datasets_landing_metadata.json and .csv")
