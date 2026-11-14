import * as vscode from 'vscode';
import { toVsCodeUri } from '../../utils/vsc-utils';
import type { Foam } from '../../../core/model/foam';
import { LocalEmbedding } from '../../../note-search/local';
import { PassageIndex } from '../../../note-search/search';
import { checkStagedLinks } from '../../../note-search/staged';
import { Config, hash } from '@foam/core';
import type { Resource } from '@foam/core';

export function createLocalEmbedding(context: vscode.ExtensionContext) {
  const cache = vscode.Uri.joinPath(
    context.globalStorageUri,
    'embedding-models'
  ).fsPath;
  const worker = vscode.Uri.joinPath(
    context.extensionUri,
    'local-model',
    'worker.cjs'
  ).fsPath;
  const provider = new LocalEmbedding(worker, cache);
  context.subscriptions.push(provider);
  return provider;
}

export function registerNoteSearch(
  context: vscode.ExtensionContext,
  foamPromise: Promise<Foam>,
  model: LocalEmbedding
) {
  let stagedNext = 0;
  const stagedContent = new Map<string, string>();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider('foam-staged', {
      provideTextDocumentContent: uri =>
        stagedContent.get(uri.toString()) ?? '',
    })
  );
  const index = new PassageIndex();
  const dirty = new Map<string, Resource>();
  let initialized = false;
  let active = false;
  const message = (error: unknown) =>
    error instanceof Error ? error.message : String(error);
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'foam-vscode.download-local-embedding',
      async () => {
        try {
          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Downloading local embedding model',
            },
            () => model.prepare()
          );
          void vscode.window.showInformationMessage(
            'Local embedding model ready.'
          );
        } catch (error) {
          void vscode.window.showErrorMessage(message(error));
        }
      }
    ),
    vscode.commands.registerCommand(
      'foam-vscode.search-note-content',
      async () => {
        if (active) return;
        active = true;
        try {
          const foam = await foamPromise;
          const editor = vscode.window.activeTextEditor;
          const selection =
            editor?.document.languageId === 'markdown'
              ? editor.document.getText(editor.selection)
              : '';
          const query = (
            await vscode.window.showInputBox({
              prompt: 'Search saved note content',
              value: selection.slice(0, 500),
              validateInput: value =>
                value.length > 500
                  ? 'Enter at most 500 characters.'
                  : undefined,
            })
          )?.trim();
          if (!query) return;
          if (!initialized) {
            const queue = (note: Resource) => {
              if (note.type === 'note') dirty.set(note.uri.path, note);
            };
            for (const note of foam.workspace.resources()) queue(note);
            context.subscriptions.push(
              foam.workspace.onDidAdd(queue),
              foam.workspace.onDidUpdate(({ new: note }) => {
                index.delete(note.uri, true);
                queue(note);
              }),
              foam.workspace.onDidDelete(note => {
                dirty.delete(note.uri.path);
                index.delete(note.uri);
              })
            );
            initialized = true;
          }
          const search = () =>
            vscode.window.withProgress(
              {
                location: vscode.ProgressLocation.Notification,
                title: 'Indexing saved notes and ranking passages',
              },
              async () => {
                for (const [path, note] of Array.from(dirty.entries())) {
                  const content = await foam.workspace.readAsMarkdown(note.uri);
                  if (
                    dirty.get(path) !== note ||
                    foam.workspace.find(note.uri) !== note
                  )
                    continue;
                  if (content !== null) {
                    index.set(
                      foam.services.parser.parse(note.uri, content),
                      content
                    );
                    dirty.delete(path);
                  }
                }
                return index.rank(query, model.embedBatch);
              }
            );
          while (true) {
            const { results, semantic, modelError } = await search();
            if (modelError)
              void vscode.window.showWarningMessage(
                modelError + ' Showing keyword results.'
              );
            if (!results.length) {
              void vscode.window.showInformationMessage(
                'No matching indexed passages.'
              );
              break;
            }
            const chosen = await vscode.window.showQuickPick(
              results.map(item => ({
                label:
                  foam.workspace.find(item.uri)?.title ||
                  item.uri.getBasename(),
                description: [
                  foam.workspace.relativePath(item.uri),
                  item.heading,
                ]
                  .filter(Boolean)
                  .join(' · '),
                detail: item.text.replace(/\s+/g, ' ').slice(0, 250),
                item,
              })),
              {
                placeHolder: semantic
                  ? 'Hybrid search · select a passage'
                  : 'Keyword search · select a passage',
              }
            );
            if (!chosen) break;
            const { item } = chosen;
            let document: vscode.TextDocument;
            try {
              document = await vscode.workspace.openTextDocument(
                toVsCodeUri(item.uri)
              );
            } catch {
              index.delete(item.uri);
              void vscode.window.showWarningMessage(
                'The note is no longer available. Results were refreshed.'
              );
              continue;
            }
            const saved = await foam.workspace.readAsMarkdown(item.uri);
            if (saved === null || hash(saved) !== item.checksum) {
              index.delete(item.uri, true);
              const note = foam.workspace.find(item.uri);
              if (note && saved !== null) dirty.set(item.uri.path, note);
              void vscode.window.showWarningMessage(
                'The saved note changed. Select a refreshed result.'
              );
              continue;
            }
            if (
              document.isDirty ||
              saved.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n') !==
                document.getText().replace(/\r\n/g, '\n')
            ) {
              void vscode.window.showWarningMessage(
                'Save or revert the open note before jumping to a saved passage.'
              );
              break;
            }
            const start = new vscode.Position(item.start, 0);
            const end =
              item.end < document.lineCount
                ? new vscode.Position(item.end, 0)
                : document.lineAt(document.lineCount - 1).range.end;
            await vscode.window.showTextDocument(document, {
              selection: new vscode.Range(start, end),
            });
            break;
          }
        } catch (error) {
          void vscode.window.showErrorMessage(message(error));
        } finally {
          active = false;
        }
      }
    ),
    vscode.commands.registerCommand(
      'foam-vscode.check-staged-note-links',
      async () => {
        try {
          const folder = vscode.workspace.workspaceFolders?.[0];
          if (
            !vscode.workspace.isTrusted ||
            vscode.workspace.workspaceFolders?.length !== 1 ||
            folder?.uri.scheme !== 'file'
          ) {
            throw new Error('Open a single trusted local Git workspace.');
          }
          const foam = await foamPromise;
          const { added, existing } = await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'Checking staged wikilinks',
            },
            () =>
              checkStagedLinks(folder.uri.fsPath, {
                matcher: foam.services.matcher,
                noteExtensions: Config.getNotesExtensions(),
                attachmentExtensions: Config.getAttachmentExtensions(),
                defaultNoteExtension: Config.getDefaultNoteExtension(),
                directoryMode: Config.getLinksDirectoryMode(),
              })
          );
          if (!added.length && !existing.length) {
            void vscode.window.showInformationMessage(
              'No broken staged wikilinks.'
            );
            return;
          }
          const chosen = await vscode.window.showQuickPick(
            [...added, ...existing].map(item => ({
              label: item.text,
              description: item.path + ':' + (item.line + 1),
              detail:
                item.reason + (added.includes(item) ? ' · new' : ' · existing'),
              item,
            })),
            {
              placeHolder:
                added.length +
                ' new · ' +
                existing.length +
                ' existing broken links',
            }
          );
          if (!chosen) return;
          const { item } = chosen;
          const uri = vscode.Uri.joinPath(folder.uri, item.path);
          const file = await Promise.resolve(
            vscode.workspace.openTextDocument(uri)
          ).catch(() => undefined);
          const same = file?.getText() === item.source;
          const stagedUri = vscode.Uri.from({
            scheme: 'foam-staged',
            path: '/' + item.path,
            query: String(stagedNext++),
          });
          stagedContent.set(stagedUri.toString(), item.source);
          const document = await Promise.resolve(
            same ? file! : vscode.workspace.openTextDocument(stagedUri)
          ).finally(() => stagedContent.delete(stagedUri.toString()));
          await vscode.window.showTextDocument(document, {
            selection: new vscode.Range(item.line, 0, item.line, 0),
          });
        } catch (error) {
          void vscode.window.showErrorMessage(message(error));
        }
      }
    )
  );
}
