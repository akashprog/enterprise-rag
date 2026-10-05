"""Shared building blocks used by every phase of the Enterprise RAG series.

Modules:
    config      - one typed settings object (paths, models, prices, budgets).
    cache       - SQLite cache so no paid API call is ever made twice.
    llm         - cached, usage-tracked wrappers around chat and embedding models.
    evaluation  - local re-implementation of the EnterpriseRAG-Bench metrics.
"""
