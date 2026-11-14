import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createMarkdownParser, URI } from '@foam/core';
import { checkStagedLinks } from './staged';
import { PassageIndex, passages } from './search';
import type { EmbedBatch } from './search';

const parser = createMarkdownParser();
const uri = URI.file('/note.md');

it('indexes the end of a long note, keeps source coordinates, and excludes frontmatter', async () => {
  const source =
    '---\ntitle: Orders\nprivate: hidden-only\n---\n## Overview\n' +
    'background information\n'.repeat(700) +
    '## Rules\nUse Idempotency-Key on retries.\n';
  const note = parser.parse(uri, source);
  const index = new PassageIndex();
  index.set(note, source);
  const { results } = await index.rank('Idempotency-Key', undefined, 'lexical');
  expect(results[0].heading).toBe('Rules');
  expect(
    source.split('\n').slice(results[0].start, results[0].end).join('\n')
  ).toContain('Idempotency-Key');
  expect(passages(note, source).every(part => part.start >= 4)).toBe(true);
  index.delete(uri);
  expect((await index.rank('Idempotency-Key')).results).toEqual([]);
});

it('reuses unchanged chunks, falls back after inference fails and removes old URIs', async () => {
  const old = '# Note\n\n## One\nalpha\n\n## Two\nbeta';
  const next = old.replace('beta', 'gamma');
  const index = new PassageIndex();
  const baked: string[] = [];
  const embed: EmbedBatch = async (inputs, role) => {
    if (role === 'passage') baked.push(...inputs);
    return inputs.map(() => [1]);
  };
  index.set(parser.parse(uri, old), old);
  await index.rank('alpha', embed);
  baked.length = 0;
  index.delete(uri, true);
  index.set(parser.parse(uri, next), next);
  await index.rank('gamma', embed);
  expect(baked).toHaveLength(1);
  expect(baked[0]).toContain('gamma');
  const fallback = await index.rank('gamma', async () => {
    throw new Error('Unavailable');
  });
  expect(fallback.semantic).toBe(false);
  expect(fallback.results[0].text).toContain('gamma');
  index.delete(uri);
  const renamed = URI.file('/new.md');
  index.set(parser.parse(renamed, next), next);
  expect(
    (await index.rank('gamma')).results.every(hit => hit.uri.isEqual(renamed))
  ).toBe(true);
});

it('never returns a deleted note when embedding was still running', async () => {
  const index = new PassageIndex();
  const source = '# Note\n\nobsolete';
  index.set(parser.parse(uri, source), source);
  let finish: (vectors: number[][]) => void;
  const block = new Promise<number[][]>(resolve => {
    finish = resolve;
  });
  const ranking = index.rank('obsolete', async (_texts, role) =>
    role === 'passage' ? block : [[1]]
  );
  index.delete(uri);
  finish([[1]]);
  expect((await ranking).results).toEqual([]);
});

it('compares complete Git trees, finds unchanged incoming links and ignores moved old failures', async () => {
  const root = mkdtempSync(join(tmpdir(), 'foam-staged-links-'));
  const git = (...args: string[]) =>
    execFileSync('git', args, { cwd: root, windowsHide: true });
  const write = (name: string, text: string) =>
    writeFileSync(join(root, name), text);
  const options = {
    matcher: { isMatch: () => true },
    noteExtensions: ['.md'],
    directoryMode: 'resolve' as const,
  };
  try {
    git('init', '-q');
    git('config', 'user.name', 'Foam tests');
    git('config', 'user.email', 'fixture@example.invalid');
    git('config', 'commit.gpgsign', 'false');
    mkdirSync(join(root, 'notes'));
    write('notes/target.md', '# Target\n\n## Old heading\n\nBody ^anchor\n');
    write(
      'notes/source.md',
      '[[target#Old heading]]\n[[target#^anchor]]\n[[placeholder]]\n'
    );
    git('add', '--', 'notes');
    expect(
      (await checkStagedLinks(root, options)).added.map(item => item.reason)
    ).toEqual(['missing-file']);
    git('-c', 'core.hooksPath=.git/empty-hooks', 'commit', '-qm', 'baseline');
    write('notes/target.md', '# Target\n\n## New heading\n\nBody\n');
    git('add', '--', 'notes/target.md');
    write('notes/target.md', '# Target\n\n## Old heading\n\nBody ^anchor\n');
    expect(
      (await checkStagedLinks(root, options)).added.map(item => item.reason)
    ).toEqual(['missing-section', 'missing-block']);
    write(
      'notes/source.md',
      '\n\n[[target#New heading]]\n[[placeholder]]\n[[new-missing]]\n'
    );
    git('add', '--', 'notes/source.md');
    write('notes/source.md', 'unstaged text\n');
    const checked = await checkStagedLinks(root, options);
    expect(checked.added.map(item => item.text)).toEqual(['[[new-missing]]']);
    expect(checked.added[0].source).toContain('[[new-missing]]');
    expect(checked.existing[0].line).toBe(3);
    write('notes/source.md', '[[target#New heading]]\n[[placeholder]]\n');
    git('add', '--', 'notes/source.md');
    expect((await checkStagedLinks(root, options)).added).toEqual([]);
    git('mv', '--', 'notes/target.md', 'notes/renamed.md');
    expect(
      (await checkStagedLinks(root, options)).added.map(item => item.reason)
    ).toEqual(['missing-file']);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}, 20000);
