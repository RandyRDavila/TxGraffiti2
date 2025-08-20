---
name: 🐞 Bug Report
about: Report unexpected behavior, incorrect output, or a crash in `txgraffiti`
title: "[Bug] "
labels: bug
assignees: randyrdavila, jeddyhub
---

## 🧩 What happened?

Describe the issue clearly. What function or feature of `txgraffiti` behaved incorrectly?

## 📋 Minimal Example

Please include a minimal, self-contained code snippet that reproduces the issue.

```python
import txgraffiti
from txgraffiti.example_data import graph_data

df = graph_data  # structured invariants for connected graphs

from txgraffiti.playground import ConjecturePlayground

ai = ConjecturePlayground(
    df,
    object_symbol="G",      # used in printed formulas (∀ G: …)
    base="connected",       # optional: global assumption
)

# Function call with unexpected result
print(ai.discover_equalities(min_fraction=0.2))  # Expected: ..., Got: ...
```

If the issue only occurs for specific inputs, please attach or describe them clearly.

## ✅ Expected Behavior

What result did you expect from this code or function?

## 🚨 Actual Behavior

What actually happened instead? If there was an error, paste the full traceback below.

## 🧪 Environment Info

Please complete the following so we can reproduce your environment:

- OS: (e.g., macOS 13.4, Ubuntu 22.04, Windows 11)
- Python version: (e.g., 3.10.6) – run `python --version`
- txgraffiti version: (e.g., 0.3.1) – run `pip show txgraffiti`
- Backend (if applicable): e.g., PuLP solver, NetworkX