import type { BgeLocal } from '../../../note-search/bge-local';
import type { Foam } from '../../../core/model/foam';
import type { ExtensionContext } from 'vscode';

export function createLocalEmbedding(
  _context: ExtensionContext
): BgeLocal | undefined {
  return undefined;
}

export function registerNoteSearch(
  _context: ExtensionContext,
  _foam: Promise<Foam>,
  _model: BgeLocal
) {}
