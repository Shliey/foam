import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import {
  AttachmentResourceProvider,
  createMarkdownParser,
  FoamWorkspace,
  MarkdownResourceProvider,
  MarkdownLink,
  Resource,
  URI,
} from '@foam/core';
import type { IDataStore, IMatcher } from '@foam/core';

const run = promisify(execFile);

async function git(root: string, args: string[]) {
  return (
    await run('git', args, {
      cwd: root,
      windowsHide: true,
      maxBuffer: 32 * 1024 * 1024,
    })
  ).stdout;
}

function blobs(list: string, staged: boolean) {
  const result = new Map<string, string>();
  for (const row of list.split('\0').filter(Boolean)) {
    const tab = row.indexOf('\t');
    const meta = row.slice(0, tab).split(' ');
    if (staged && meta[2] !== '0')
      throw new Error('Resolve Git index conflicts first.');
    if (!/^100(644|755)$/.test(meta[0])) continue;
    result.set(row.slice(tab + 1), meta[staged ? 1 : 2]);
  }
  return result;
}

export interface BrokenLink {
  path: string;
  line: number;
  text: string;
  reason: 'missing-file' | 'missing-section' | 'missing-block';
  source: string;
}

export interface SnapshotOptions {
  matcher: Pick<IMatcher, 'isMatch'>;
  noteExtensions: string[];
  attachmentExtensions?: string[];
  defaultNoteExtension?: string;
  directoryMode: 'disabled' | 'resolve';
}

async function inspect(
  root: string,
  files: Map<string, string>,
  options: SnapshotOptions,
  cache: Map<string, Promise<string>>
) {
  const rootUri = URI.file(root);
  const names = new Map<string, string>();
  const uris: URI[] = [];
  for (const [name, oid] of files) {
    const uri = rootUri.joinPath(name);
    if (!options.matcher.isMatch(uri)) continue;
    names.set(uri.path, oid);
    uris.push(uri);
  }
  const parser = createMarkdownParser();
  const read = async (uri: URI) => {
    const oid = names.get(uri.path);
    if (!oid) return null;
    if (!cache.has(oid)) cache.set(oid, git(root, ['cat-file', 'blob', oid]));
    return cache.get(oid)!;
  };
  const rejectWrite = async () => {
    throw new Error('Snapshot is read-only');
  };
  const store: IDataStore = {
    list: async () => uris,
    read,
    write: rejectWrite,
    delete: rejectWrite,
    move: rejectWrite,
    exists: async uri => names.has(uri.path),
  };
  const provider = new MarkdownResourceProvider(
    store,
    parser,
    options.noteExtensions,
    options.directoryMode
  );
  const attachments = new AttachmentResourceProvider(
    options.attachmentExtensions
  );
  const workspace = await FoamWorkspace.fromProviders(
    [rootUri],
    [provider, attachments],
    store,
    options.defaultNoteExtension ?? '.md',
    12
  );
  try {
    const broken: BrokenLink[] = [];
    for (const note of workspace.resources()) {
      for (const link of note.links) {
        if (link.type !== 'wikilink') continue;
        const { section, blockId } = MarkdownLink.analyzeLink(link);
        const targetUri = workspace.resolveLink(note, link);
        const target = targetUri.isPlaceholder()
          ? null
          : workspace.find(targetUri.with({ fragment: '' }));
        const reason = !target
          ? 'missing-file'
          : blockId && !Resource.findBlock(target, blockId)
          ? 'missing-block'
          : section && !blockId && !Resource.findSection(target, section)
          ? 'missing-section'
          : undefined;
        if (reason)
          broken.push({
            path: workspace.relativePath(note.uri).slice(1),
            line: link.range.start.line,
            text: link.rawText,
            reason,
            source: (await read(note.uri))!,
          });
      }
    }
    return broken;
  } finally {
    workspace.dispose();
    provider.dispose();
    attachments.dispose();
  }
}

export async function checkStagedLinks(root: string, options: SnapshotOptions) {
  const prefix = (await git(root, ['rev-parse', '--show-prefix'])).trim();
  if (prefix) throw new Error('Open the Git repository root.');
  const readHead = async () => {
    const status = await git(root, [
      '--no-optional-locks',
      'status',
      '--porcelain=v2',
      '--branch',
      '--untracked-files=no',
    ]);
    const oid = /^# branch\.oid ([a-f0-9]+)\r?$/m.exec(status)?.[1];
    if (!oid && !status.includes('# branch.oid (initial)'))
      throw new Error('Cannot read the Git HEAD.');
    return oid;
  };
  const head = await readHead();
  const index = await git(root, ['ls-files', '--stage', '-z']);
  const staged = blobs(index, true);
  const base = head
    ? blobs(await git(root, ['ls-tree', '-r', '-z', head]), false)
    : new Map<string, string>();
  const changes = head
    ? (
        await git(root, ['diff', '--cached', '--name-status', '-z', '-M', head])
      ).split('\0')
    : [];
  const renames = new Map<string, string>();
  for (let i = 0; i < changes.length && changes[i]; ) {
    const status = changes[i++];
    const oldPath = changes[i++];
    if (status.startsWith('R')) renames.set(oldPath, changes[i++]);
  }
  const cache = new Map<string, Promise<string>>();
  const before = await inspect(root, base, options, cache);
  const after = await inspect(root, staged, options, cache);
  if (
    index !== (await git(root, ['ls-files', '--stage', '-z'])) ||
    head !== (await readHead())
  ) {
    throw new Error('The Git index or HEAD changed. Run the check again.');
  }
  const key = (path: string, raw: string) =>
    JSON.stringify([path, raw.replace(/\|[^\]]*]]$/, ']]')]);
  const counts = new Map<string, number>();
  for (const item of before) {
    const id = key(renames.get(item.path) ?? item.path, item.text);
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  const existing: BrokenLink[] = [];
  const added: BrokenLink[] = [];
  for (const item of after) {
    const id = key(item.path, item.text);
    const count = counts.get(id) ?? 0;
    if (count) {
      counts.set(id, count - 1);
      existing.push(item);
    } else added.push(item);
  }
  return { added, existing };
}
