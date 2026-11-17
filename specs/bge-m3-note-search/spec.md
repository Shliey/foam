# BGE-M3 note content search

Readers can search the saved text of Foam notes by phrase, technical identifier,
or meaning across Chinese and English. Choosing a result opens its source passage.

- Given a prepared model and saved notes, a search recalls candidates by dense
  similarity and model-weighted token overlap. The final order uses token-level
  comparison of the fused candidates. _Verification: unit and extension host._
- Given a note edit, rename, or deletion, the next search reflects the change
  and changes only the necessary indexed representations. _Verification: unit
  and extension host._
- Given an unavailable model, the search explains how to prepare the local
  model or retry. A prepared model can search using local files.
  _Verification: extension host._
- Given a frozen set of labeled passages and queries, the evaluation compares
  each retrieval stage and the previous Foam baseline on the same corpus.
  _Verification: real-model evaluation report._
