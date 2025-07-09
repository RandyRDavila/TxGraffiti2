Statistical and Neural Approaches
=================================

Recent systems use statistical inference or neural networks to guide conjecturing—not by symbolic manipulation, but by detecting latent patterns in data.

**Notable Systems**:
- **The Ramanujan Machine**: Uses symbolic search and numerical matching to conjecture closed-form constants.
- **DeepMind's Neural Mathematician**: Learns predictive features and helps humans identify new theorems.

**Main Idea**:
- Replace hand-crafted heuristics with **learned patterns**.
- Use **gradient-based attribution** to surface conjecture-relevant features.
- Conjecture structure often emerges from data, not logic.

**TxGraffiti Connection**:
While TxGraffiti is not a deep learning system, it:
- Leverages tabular data and optimization as a statistical surrogate.
- Can be extended with learned priors (e.g., neural models ranking variable relevance).
- Supports self-training loops where significance improves over time.

TxGraffiti is compatible with these modern approaches and can serve as the "symbolic" front end to a learned backend.
