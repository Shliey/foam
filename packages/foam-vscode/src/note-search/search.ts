import { Resource, URI, hash } from '@foam/core';

export interface Passage {
  uri: URI;
  heading: string;
  start: number;
  end: number;
  text: string;
  input: string;
}

export function passages(note: Resource, markdown: string): Passage[] {
  const lines = markdown.split(/\r?\n/);
  const frontmatterEnd =
    lines[0]?.replace(/^\uFEFF/, '').trim() === '---'
      ? lines.findIndex(
          (line, index) => index > 0 && /^(---|\.\.\.)\s*$/.test(line)
        )
      : -1;
  const start = frontmatterEnd + 1;
  const headings = note.sections.filter(
    section => section.headingRange.start.line >= start
  );
  const results: Passage[] = [];
  const trail: typeof headings = [];
  let first = start;
  function append(end: number) {
    const heading = trail.map(section => section.label).join(' / ');
    const prefix = [note.title, heading]
      .filter(Boolean)
      .join('\n')
      .slice(0, 240);
    let cursor = first;
    while (cursor < end) {
      const from = cursor;
      let size = 0;
      let paragraphEnd = from;
      while (cursor < end && size + lines[cursor].length + 1 <= 650) {
        size += lines[cursor].length + 1;
        cursor++;
        if (!lines[cursor - 1].trim()) paragraphEnd = cursor;
      }
      if (cursor === from) {
        const line = lines[cursor++];
        for (const match of line.matchAll(/.{1,650}/gu)) {
          const text = match[0];
          if (text.trim())
            results.push({
              uri: note.uri,
              heading,
              start: from,
              end: cursor,
              text,
              input: [prefix, text].filter(Boolean).join('\n'),
            });
        }
        continue;
      }
      if (cursor < end && paragraphEnd > from) cursor = paragraphEnd;
      const text = lines.slice(from, cursor).join('\n').trim();
      if (text)
        results.push({
          uri: note.uri,
          heading,
          start: from,
          end: cursor,
          text,
          input: [prefix, text].filter(Boolean).join('\n'),
        });
    }
  }
  for (const section of headings) {
    const line = section.headingRange.start.line;
    append(line);
    while (trail.length && trail.at(-1)!.level >= section.level) trail.pop();
    trail.push(section);
    first = line;
  }
  append(lines.length);
  return results;
}

const segmenter = new Intl.Segmenter('zh', { granularity: 'word' });
function terms(value: string) {
  const text = value.normalize('NFKC').toLowerCase();
  const words = [...segmenter.segment(text)]
    .filter(part => part.isWordLike)
    .map(part => part.segment);
  const identifiers =
    text.match(/[\p{L}\p{N}]+(?:[-_./:][\p{L}\p{N}]+)+/gu) ?? [];
  return [...words, ...identifiers];
}

function frequencies(text: string) {
  const counts = new Map<string, number>();
  const tokens = terms(text);
  for (const term of tokens) counts.set(term, (counts.get(term) ?? 0) + 1);
  return { length: tokens.length, counts };
}

export type RankedPassage = Passage & { score: number; checksum: string };
export type EmbedBatch = (
  texts: string[],
  role: 'query' | 'passage'
) => Promise<number[][]>;

export class PassageIndex {
  private entries = new Map<
    string,
    {
      checksum: string;
      passages: Passage[];
      words: ReturnType<typeof frequencies>[];
      vectors?: number[][];
    }
  >();

  private reused = new Map<string, Map<string, number[]>>();

  set(note: Resource, source: string) {
    const checksum = hash(source);
    if (this.entries.get(note.uri.path)?.checksum === checksum) return;
    const chunks = passages(note, source);
    const cached =
      this.reused.get(note.uri.path) ?? new Map<string, number[]>();
    const previous = this.entries.get(note.uri.path);
    previous?.vectors?.forEach((vector, i) =>
      cached.set(previous.passages[i].input, vector)
    );
    this.reused.set(
      note.uri.path,
      new Map(
        chunks
          .map(chunk => [chunk.input, cached.get(chunk.input)])
          .filter((pair): pair is [string, number[]] => Array.isArray(pair[1]))
      )
    );
    this.entries.set(note.uri.path, {
      checksum,
      passages: chunks,
      words: chunks.map(chunk => frequencies(chunk.input)),
    });
  }

  delete(uri: URI, retainVectors = false) {
    const entry = this.entries.get(uri.path);
    if (retainVectors && entry?.vectors) {
      this.reused.set(
        uri.path,
        new Map(
          entry.passages.map((chunk, i) => [chunk.input, entry.vectors![i]])
        )
      );
    } else if (!retainVectors) this.reused.delete(uri.path);
    this.entries.delete(uri.path);
  }

  async rank(
    query: string,
    embed?: EmbedBatch,
    mode: 'hybrid' | 'dense' | 'lexical' = 'hybrid'
  ) {
    let semantic = false;
    let modelError: string | undefined;
    const entries = [...this.entries.values()];
    let vector: number[] | undefined;
    if (embed && mode !== 'lexical' && entries.length) {
      try {
        const existing = new Map<string, number[]>();
        for (const cache of this.reused.values()) {
          for (const [input, v] of cache) existing.set(input, v);
        }
        for (const entry of entries) {
          entry.vectors?.forEach((v, i) =>
            existing.set(entry.passages[i].input, v)
          );
        }
        const pending = [
          ...new Set(
            entries
              .filter(entry => !entry.vectors)
              .flatMap(entry => entry.passages.map(chunk => chunk.input))
              .filter(input => !existing.has(input))
          ),
        ];
        for (let i = 0; i < pending.length; i += 16) {
          const batch = pending.slice(i, i + 16);
          const vectors = await embed(batch, 'passage');
          batch.forEach((input, j) => existing.set(input, vectors[j]));
        }
        for (const entry of entries) {
          entry.vectors ??= entry.passages.map(
            chunk => existing.get(chunk.input)!
          );
        }
        [vector] = await embed([query], 'query');
        semantic = true;
      } catch (error) {
        modelError = error instanceof Error ? error.message : String(error);
      }
    }
    const rows = [...this.entries.values()].flatMap(entry =>
      entry.passages.map((chunk, index) => ({
        chunk,
        checksum: entry.checksum,
        words: entry.words[index],
        vector: entry.vectors?.[index],
      }))
    );
    const q = [...new Set(terms(query))];
    const avg =
      rows.reduce((sum, row) => sum + row.words.length, 0) / rows.length || 1;
    const weights = q.map(term => {
      const df = rows.filter(row => row.words.counts.has(term)).length;
      return [
        term,
        Math.log(1 + (rows.length - df + 0.5) / (df + 0.5)),
      ] as const;
    });
    function top(score: (row: (typeof rows)[number]) => number, floor: number) {
      return rows
        .map((row, i) => ({ i, score: score(row) }))
        .filter(item => item.score > floor)
        .sort((a, b) => b.score - a.score || a.i - b.i)
        .slice(0, 20);
    }
    const lexical = top(
      row =>
        weights.reduce((sum, [term, idf]) => {
          const tf = row.words.counts.get(term) ?? 0;
          return (
            sum +
            (idf * tf * 2.2) /
              (tf + 1.2 * (0.25 + (0.75 * row.words.length) / avg))
          );
        }, 0),
      0
    );
    const dense = vector
      ? top(
          row =>
            row.vector?.reduce((sum, n, i) => sum + n * vector![i], 0) ??
            -Infinity,
          -Infinity
        )
      : [];
    const combined = new Map<number, number>();
    const lists = mode === 'dense' && semantic ? [dense] : [lexical, dense];
    for (const list of lists) {
      list.forEach(({ i }, rank) =>
        combined.set(i, (combined.get(i) ?? 0) + 1 / (61 + rank))
      );
    }
    const results: RankedPassage[] = [];
    for (const [i, score] of [...combined].sort((a, b) => b[1] - a[1])) {
      const { chunk, checksum } = rows[i];
      if (
        results.some(
          item =>
            item.uri.path === chunk.uri.path &&
            item.start < chunk.end &&
            chunk.start < item.end
        )
      )
        continue;
      results.push({ ...chunk, checksum, score });
      if (results.length === 5) break;
    }
    return { results, semantic, modelError };
  }
}
