---
type: feature
created: 2026-11-14
---

# Search note sections and check staged wikilinks

When a workspace has long notes, readers need to find the paragraph that
answers a search. Before committing changes, authors need to know which
wikilinks will become invalid in the resulting Git tree.

## Acceptance criteria

- **AC-1**: Given saved notes with headings and a distinctive term in a late
  section, when a reader searches for the term, then the results can open the
  matching excerpt in its source note. _verify: unit, human_
- **AC-2**: Given a previously searched workspace, when notes are changed,
  renamed or deleted, then the next search reflects those changes and can reuse
  embeddings for unchanged passages. _verify: unit, human_
- **AC-3**: Given a missing local embedding model, when a reader searches, then
  keyword search remains available. _verify: unit_
- **AC-4**: Given a staged rename or anchor removal, when an author checks
  staged wikilinks, then the report distinguishes newly broken links from
  existing broken links, including references in unchanged notes. _verify: unit_
- **AC-5**: Given a staged source with different unstaged edits, when an author
  opens a reported link, then the editor shows the corresponding staged text.
  _verify: human_

## Scope

Search operates on saved notes with local embeddings. The staged check covers
Foam wikilinks in a single Git workspace and reports file, heading and block
targets. It reads snapshots without altering the index. Model download is a
separate command. The initial package targets Windows x64.
