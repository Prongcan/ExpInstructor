# ExpInstructor

[![Python](https://img.shields.io/badge/python-3.x-blue)]()

## 🚀 Project Overview

**ExpInstructor** is the first **agentic AI instructor** designed to emulate how human mentors leverage past scholarly experience to guide research ideation. The system captures human academic knowledge by extracting **experience triples** from large-scale review corpora, where each triple consists of knowledge entities and their relationships. These triples are then used to construct an **Experience Graph**, which is utilized by an **agentic reasoning framework** to retrieve relevant experiences and provide experience-driven evaluation of research ideas.

We also design metrics to measure correlations between ExpInstructor and human evaluations in terms of **scores** and **concerns raised**.

To use this project, please download the `all_graph.json`, ICLR reviews, and Stanford raw dataset from [here](https://drive.google.com/drive/folders/1voE6Q9mwl1C-SDN6St3coECAbjyTAdNn?usp=sharing), unzip them, and organize the files according to the following project structure:

```
ExpInstructor/
├── data/
│   ├── ICLR/
│   └── Stanford/
├── Evaluation_feasibility/
├── Evaluation_feasibility_score/
├── Evaluation_novelty/
├── Evaluation_significance/
├── Evaluation_utils/
├── Graph_constract/
├── RAG_baseline_review_sentence/
├── result_v2/
│   └── all_graph.json
├── Retrive_Generate/
├── service/
├── .gitignore
├── README.md
├── clean_graph_data.py
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/Prongcan/ExpInstructor.git
cd ExpInstructor
pip install -r requirements.txt
```

Run the search service:

```bash
python3 Evaluation_feasibility/search_service.py
```

> **Note:** This repository currently loads the graph on the CPU, which is not optimized for large-scale or high-performance usage. A faster search implementation is under development. This version is intended for **lightweight experimentation and testing**.

After deploying the search service, you can test novelty evaluation using:

```bash
python3 Evaluation_novelty/ins_single.py
```