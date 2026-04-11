#!/usr/bin/env python3
"""Generate fixture data files for self.curator pipeline tests.

Produces sample_data.jsonl and sample_data.parquet in the same directory.

Record design (30 total):
  - 20 normal varied records (ids 1-20)
  - 3 exact duplicates of record 1 (ids 21-23)
  - 3 near-duplicates of record 2 (ids 24-26)
  - 2 very short texts (ids 27-28, for word count filter testing)
  - 2 URL-bearing texts (ids 29-30)
"""

import json
import pathlib

NORMAL_TEXTS = [
    # 1 — exact-dup source
    "Quantum computing leverages superposition and entanglement to solve"
    " problems that classical computers find intractable. Researchers"
    " recently demonstrated a 1000-qubit processor running error-corrected"
    " circuits. The implications for cryptography are profound.",
    # 2 — near-dup source
    "Sourdough bread requires a living starter culture maintained with"
    " regular feedings of flour and water. The long fermentation process"
    " develops complex flavors and improves digestibility. Many bakers"
    " keep their starters alive for decades.",
    # 3
    "The Pacific Ocean covers more area than all of Earth's landmasses"
    " combined. Deep-sea trenches near the Mariana Islands reach depths"
    " exceeding 36,000 feet. Marine biologists continue to discover new"
    " species in these extreme environments.",
    # 4
    "Machine learning models trained on large text corpora can exhibit"
    " surprising emergent abilities. Fine-tuning on domain-specific data"
    " often yields better results than scaling alone.",
    # 5
    "Cast iron skillets distribute heat evenly and retain it for a long"
    " time, making them ideal for searing steaks. Proper seasoning builds"
    " a natural non-stick surface.",
    # 6
    "CRISPR-Cas9 gene editing technology has revolutionized molecular"
    " biology by allowing precise modifications to DNA sequences."
    " Clinical trials are underway for sickle cell disease treatments.",
    # 7
    "The Rust programming language guarantees memory safety without a"
    " garbage collector through its ownership and borrowing system."
    " Adoption has grown rapidly in systems programming.",
    # 8
    "Fermentation transforms simple ingredients into complex foods like"
    " kimchi, miso, and yogurt. Lactobacillus bacteria convert sugars"
    " into lactic acid, preserving the food.",
    # 9
    "Satellite imagery combined with deep learning enables real-time"
    " monitoring of deforestation in tropical rainforests. Brazil's INPE"
    " agency detected significant reduction in Amazon clearing.",
    # 10
    "The James Webb Space Telescope has captured images of galaxies"
    " formed just 300 million years after the Big Bang. These observations"
    " challenge existing models of early galaxy formation.",
    # 11
    "Electric vehicle battery technology has improved dramatically, with"
    " solid-state cells promising higher energy density and faster"
    " charging times than current lithium-ion designs.",
    # 12
    "Composting kitchen scraps and yard waste reduces landfill methane"
    " emissions while producing nutrient-rich soil amendments.",
    # 13
    "Neural network architectures based on the transformer design have"
    " become the dominant approach for natural language processing."
    " Attention mechanisms allow weighing input relevance.",
    # 14
    "The Mediterranean diet emphasizes olive oil, whole grains, legumes,"
    " and fresh vegetables. Large cohort studies associate this dietary"
    " pattern with reduced cardiovascular risk.",
    # 15
    "Containerized deployments using Docker and Kubernetes have become"
    " standard practice for microservice architectures. Orchestration"
    " tools handle scaling and self-healing automatically.",
    # 16
    "Coral reefs support roughly twenty-five percent of all marine"
    " species despite covering less than one percent of the ocean floor.",
    # 17
    "The art of espresso depends on grind size, water temperature, and"
    " pressure. A properly pulled shot takes between 25 and 30 seconds.",
    # 18
    "Differential privacy adds calibrated noise to query results,"
    " enabling statistical analysis of sensitive datasets without"
    " revealing individual records.",
    # 19
    "Honeybees communicate the location of food sources through a waggle"
    " dance that encodes both direction and distance relative to the sun.",
    # 20
    "Three-dimensional printing with biocompatible materials has enabled"
    " the fabrication of patient-specific implants and prosthetics.",
]

NEAR_DUP_TEXTS = [
    "Sourdough bread requires a living starter culture maintained with"
    " regular feedings of flour and water. The lengthy fermentation"
    " process develops complex flavors and improves digestibility. Many"
    " bakers keep their starters alive for decades.",
    "Sourdough bread needs a living starter culture maintained with"
    " regular feedings of flour and water. The long fermentation process"
    " develops rich flavors and improves digestibility. Many bakers keep"
    " their starters going for decades.",
    "Sourdough bread requires a living starter culture sustained with"
    " regular feedings of flour and water. The long fermentation process"
    " develops complex flavors and enhances digestibility. Many bakers"
    " keep their starters alive for decades.",
]

SHORT_TEXTS = ["Hello world", "Test data"]

URL_TEXTS = [
    "For more information on climate change data, visit"
    " https://climate.nasa.gov/evidence/ and review the latest global"
    " temperature anomaly charts. The data is updated monthly.",
    "The open-source project is hosted at"
    " https://github.com/example/project where contributors can submit"
    " pull requests. Documentation is at http://docs.example.com/guide.",
]


def build_records():
    records = []
    for i, text in enumerate(NORMAL_TEXTS, start=1):
        records.append({"id": i, "text": text})
    for i in range(21, 24):
        records.append({"id": i, "text": NORMAL_TEXTS[0]})  # exact dups of id 1
    for i, text in zip(range(24, 27), NEAR_DUP_TEXTS):
        records.append({"id": i, "text": text})
    for i, text in zip(range(27, 29), SHORT_TEXTS):
        records.append({"id": i, "text": text})
    for i, text in zip(range(29, 31), URL_TEXTS):
        records.append({"id": i, "text": text})
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_parquet(records, path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow not installed — skipping Parquet generation.")
        return
    table = pa.table({"id": [r["id"] for r in records], "text": [r["text"] for r in records]})
    pq.write_table(table, str(path))


def main():
    here = pathlib.Path(__file__).resolve().parent
    records = build_records()
    write_jsonl(records, here / "sample_data.jsonl")
    write_parquet(records, here / "sample_data.parquet")
    print(f"Wrote {len(records)} records to JSONL + Parquet")


if __name__ == "__main__":
    main()
