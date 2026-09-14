# Efficient Coordination Mechanisms for Multi-Agent Generative AI Systems in E-Commerce Platforms

This research investigates whether **explicit coordination between specialised Large Language Model (LLM) agents** improves the effectiveness, reliability, and computational efficiency of e-commerce product recommendation systems.

The study compares three architectures:

1. **Single-Agent System**
2. **Uncoordinated Multi-Agent System**
3. **Coordinated Multi-Agent System**

The central objective is to distinguish improvements resulting from **agent specialisation** from those specifically resulting from **explicit coordination**.

---

## Research Overview

E-commerce platforms contain increasingly large product catalogues, making product selection difficult when users have multiple requirements involving factors such as brand, budget, features, ratings, and customer feedback.

Large Language Models enable users to express these requirements through natural language. However, recommendation generation may involve several distinct tasks, including:

- Understanding user intent
- Retrieving relevant products
- Analysing customer reviews
- Ranking candidate products
- Verifying whether recommendations satisfy user requirements
- Avoiding unsupported recommendations

Multi-agent architectures allow these tasks to be distributed among specialised agents. However, specialisation alone does not guarantee reliable recommendations.

This research therefore examines whether adding **structured coordination and final verification** provides measurable benefits beyond simply using multiple specialised agents.

---

## Research Aim

To design, implement, and empirically evaluate a structured coordination framework for LLM-based multi-agent e-commerce recommendation and determine the contribution of explicit coordination by comparing recommendation effectiveness, reliability, computational efficiency, and bounded scalability against single-agent and uncoordinated multi-agent architectures.

---

## System Architectures

### 1. Single-Agent System

A single LLM agent performs the complete recommendation workflow, including:

**Query Understanding → Product Retrieval → Review Analysis → Ranking → Recommendation**

This architecture provides the baseline for evaluating the effect of agent specialisation.

### 2. Uncoordinated Multi-Agent System

Recommendation tasks are distributed across four specialised agents:

**Query Agent → Retrieval Agent → Review Agent → Ranking Agent**

Each agent performs a specialised function, but the architecture does not include an explicit final verification mechanism.

This architecture helps isolate the contribution of **agent specialisation**.

### 3. Coordinated Multi-Agent System

The coordinated architecture contains five specialised agents:

**Query Agent → Retrieval Agent → Review Agent → Ranking Agent → Verifier Agent**

The agents exchange structured outputs throughout the recommendation pipeline.

The **Verifier Agent** performs final validation of:

- Brand requirements
- Product type
- Requested features
- Budget constraints
- Supporting product evidence

The verifier can either approve the recommendation or **abstain** when sufficient evidence is unavailable.

This architecture is used to evaluate the additional contribution of **explicit coordination and verification**.

---

## Technology Stack

- **Python**
- **OpenAI GPT-4o-mini**
- **OpenAI text-embedding-3-small**
- **LangChain**
- **FAISS**
- **Pandas**
- **Pydantic**
- **Jupyter Notebook**

The LLM was configured with a temperature of `0` to improve experimental consistency.

---

## Dataset

The study uses **Amazon smartphone product and review data**.

Approximately **20,000 raw product records** were available before preprocessing.

The data contains information such as:

- Product titles and descriptions
- Product features
- Prices
- Average ratings
- Rating volumes
- Product categories
- Customer reviews

Large raw dataset files are not stored directly in this GitHub repository because of their size.

### Dataset Access

The complete dataset files (`.parquet`, `.csv`, and `.jsonl`) can be accessed here:

**[Download Dataset from Google Drive](PASTE_GOOGLE_DRIVE_LINK_HERE)**

A small sample of the processed data may be included in the repository to demonstrate the dataset structure.

> **Note:** The final experimental evaluation was conducted using the Amazon smartphone dataset. Cross-platform evaluation was not part of the final experiment.

---

## Retrieval Pipeline

Product information is transformed into textual representations and embedded using:

`text-embedding-3-small`

The embeddings are indexed using **FAISS** for semantic similarity search.

The retrieval pipeline identifies candidate products relevant to the user's natural-language requirements before further review analysis, ranking, and verification.

---

## Coordinated Ranking

The coordinated recommendation system combines several signals during ranking:

| Ranking Component | Weight |
|---|---:|
| Retrieval Relevance | 50% |
| Requested Feature Match | 20% |
| Customer Sentiment | 10% |
| Average Rating | 10% |
| Rating Volume / Popularity | 10% |

Budget violations are handled through a separate penalty mechanism.

---

## Experimental Design

A controlled benchmark of **60 natural-language recommendation queries** was developed across **9 query categories**.

The three architectures were evaluated using the same benchmark, resulting in:

**60 Queries × 3 Architectures = 180 Architecture-Level Evaluations**

The benchmark contained:

- **48 expected-match queries**
- **12 expected-no-match queries**

### Query Categories

- Brand + Feature
- Multiple Features
- Feature Only
- No Match
- Budget
- Unsupported
- Complex
- Brand Only
- Ambiguous

---

## Evaluation Framework

The systems were evaluated across four major dimensions.

### Recommendation Effectiveness

- Precision@5
- Recall@5
- Feature Satisfaction
- Task Completion

### Reliability and Abstention

- Grounded Recommendation Rate
- Partial Grounding Rate
- Hallucination (Unsupported Recommendation) Rate
- Correct Abstention Rate
- Recommendation Coverage

### Computational Efficiency

- Mean and median latency
- Token consumption
- Estimated API cost

### Bounded Scalability

Scalability was evaluated by increasing retrieval depth:

`k = 5, 10, 20, 30`

This represents **retrieval-workload scalability** rather than production-scale concurrency or distributed deployment.

---

## Key Results

| Metric | Single Agent | Uncoordinated MAS | Coordinated MAS |
|---|---:|---:|---:|
| Precision@5 | 0.3200 | **0.4800** | **0.4800** |
| Recall@5 | 0.1199 | **0.1967** | **0.1967** |
| Feature Satisfaction | 0.7789 | 0.7440 | **0.9109** |
| Grounded Recommendation Rate | 0.6122 | 0.6786 | **0.8372** |
| Partial Grounding Rate | 0.1020 | 0.1250 | **0.1395** |
| Hallucination / Unsupported Rate | 0.2857 | 0.1964 | **0.0233** |
| Correct Abstention Rate | 0.6667 | 0.2500 | **0.9167** |
| Recommendation Coverage | 0.8167 | **0.9333** | 0.7167 |
| Mean Latency (seconds) | 6.271 | 6.677 | **4.790** |
| Mean Tokens | 3374.6 | **1847.8** | 1851.4 |
| Estimated Cost / Query (USD) | $0.000646 | **$0.000365** | $0.000368 |

### Task Completion

| Architecture | Successful Queries | Task Completion |
|---|---:|---:|
| Single Agent | 42 / 60 | 70.00% |
| Uncoordinated MAS | 46 / 60 | 76.67% |
| Coordinated MAS | **52 / 60** | **86.67%** |

---

## Key Findings

The experiments produced three important findings.

**Agent specialisation improved retrieval effectiveness.** Both multi-agent architectures achieved higher Precision@5 and Recall@5 than the Single-Agent System.

However, **explicit coordination did not further improve top-k retrieval accuracy**. The Coordinated and Uncoordinated Multi-Agent Systems achieved identical aggregate Precision@5 and Recall@5.

The main benefit of explicit coordination was instead observed in **reliability and decision control**. The Coordinated MAS achieved the highest Feature Satisfaction and Grounded Recommendation Rate while reducing the Hallucination (Unsupported Recommendation) Rate to **2.33%**.

Compared with the Uncoordinated MAS, coordination also increased Correct Abstention Rate from **25.00% to 91.67%**, although this was accompanied by lower recommendation coverage.

Therefore, the results suggest that:

> **Agent specialisation primarily contributed to retrieval improvements, while explicit coordination primarily contributed to reliability, verification, and appropriate abstention.**

---

## Statistical Analysis

Paired statistical testing was used to determine whether observed differences were statistically significant.

### Wilcoxon Signed-Rank Test

The Wilcoxon Signed-Rank Test was applied to paired numerical outcomes.

Significant differences were observed for:

- Latency: Single Agent vs Coordinated MAS
- Latency: Uncoordinated MAS vs Coordinated MAS
- Token consumption: Single Agent vs Coordinated MAS

Feature Satisfaction differences were not statistically significant at `α = 0.05`.

Token consumption between the Uncoordinated and Coordinated MAS was also not significantly different.

### McNemar's Test

McNemar's test was used for paired categorical outcomes.

The Coordinated MAS produced a statistically significant reduction in **unsupported recommendations** compared with both alternative architectures.

Correct abstention also improved significantly between the Uncoordinated and Coordinated MAS.

Differences in Grounded Recommendation Rate were descriptive but were not statistically significant.

### Effect Sizes

Rank-biserial effect sizes were calculated to evaluate the magnitude of paired differences.

The results showed meaningful effects for Feature Satisfaction and latency, while the token difference between the Uncoordinated and Coordinated MAS was negligible.

---

## Coordination Efficiency

Compared with the Uncoordinated MAS, the Coordinated MAS achieved:

- **+22.43%** Feature Satisfaction
- **+23.37%** Grounded Recommendation Rate
- **−88.14%** Hallucination / Unsupported Recommendation Rate
- **+266.68%** Correct Abstention Rate
- **−28.26%** Mean Latency
- **+0.19%** Token consumption
- **+0.82%** Estimated API cost

This indicates that the reliability improvements associated with coordination were achieved with **minimal additional token and estimated API-cost overhead** relative to the Uncoordinated MAS.

---

## Repository Structure

```text
.
├── Agents/
│   ├── query_agent.py
│   ├── retrieval_agent.py
│   ├── review_agent.py
│   ├── ranking_agent.py
│   └── verifier_agent.py
│
├── Systems/
│   ├── single_agent.py
│   ├── uncoordinated_multi_agent.py
│   └── coordinated_multi_agent.py
│
├── Notebooks/
│   └── Experimental and evaluation notebooks
│
├── Results/
│   ├── Evaluation/
│   └── Final/
│
├── Frozen_Implementation/
│   └── Final experimental agent implementation
│
├── Data/
│   └── README.md
│
├── FINAL_EXPERIMENT_INFO.txt
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets, environment files, virtual environments, and vector-store files are excluded from version control.

---

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Create and activate a Python environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file locally and provide the required API credentials.

**API keys and `.env` files should never be committed to the repository.**

---

## Research Limitations

The study has several limitations:

- Evaluation was restricted to the Amazon smartphone domain.
- No real-user study was conducted.
- User trust, perceived usefulness, and recommendation satisfaction were therefore not directly evaluated.
- The recommendation system used textual evidence rather than multimodal product information.
- Scalability evaluation was limited to increasing retrieval depth.
- Concurrent users, distributed deployment, and sustained production workloads were not evaluated.

---

## Future Research

Future work could:

- Evaluate larger benchmarks and multiple product categories.
- Test the architectures across multiple e-commerce datasets.
- Conduct human-user studies evaluating relevance, usefulness, trust, recommendation satisfaction, and acceptance of abstention.
- Explore adaptive coordination based on query complexity and confidence.
- Compare alternative LLMs, embedding models, and retrieval methods.
- Extend the system to multimodal recommendations using product images and visual evidence.
- Evaluate larger catalogues, concurrent users, distributed deployment, and production-like workloads.

---

## Conclusion

This research demonstrates that the value of multi-agent recommendation systems depends not only on the presence of multiple specialised agents, but also on how their outputs are coordinated and verified.

While agent specialisation improved retrieval effectiveness, explicit coordination did not provide additional Precision@5 or Recall@5 improvements over the Uncoordinated MAS.

Instead, coordination primarily strengthened **requirement satisfaction, evidence grounding, unsupported-recommendation control, and appropriate abstention**, while introducing minimal additional token and estimated API-cost overhead compared with the Uncoordinated MAS.

The findings therefore position explicit coordination primarily as a mechanism for **reliability and decision control**, rather than simply a mechanism for improving retrieval accuracy.

---
