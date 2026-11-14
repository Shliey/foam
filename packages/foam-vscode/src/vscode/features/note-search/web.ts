import type { LocalEmbedding } from '../../../note-search/local';
import type { Foam } from '../../../core/model/foam';
import type { ExtensionContext } from 'vscode';

export function createLocalEmbedding(
  _context: ExtensionContext
): LocalEmbedding | undefined {
  return undefined;
}

export function registerNoteSearch(
  _context: ExtensionContext,
  _foam: Promise<Foam>,
  _model: LocalEmbedding
) {}
