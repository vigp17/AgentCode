import * as vscode from "vscode";
import { AgentProcess } from "./agentProcess.js";

const MAX_CONTEXT_BYTES = 50_000;

export class ContextProvider implements vscode.Disposable {
  private disposables: vscode.Disposable[] = [];
  private debounceTimer: NodeJS.Timeout | null = null;

  constructor(private readonly agent: AgentProcess) {
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor((editor) => {
        this.scheduleUpdate(editor);
      })
    );
    // Send context for the currently open file on startup
    this.scheduleUpdate(vscode.window.activeTextEditor);
  }

  private scheduleUpdate(editor: vscode.TextEditor | undefined): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => this.sendContext(editor), 300);
  }

  private sendContext(editor: vscode.TextEditor | undefined): void {
    if (!editor || !this.agent.isReady()) return;

    const doc = editor.document;
    if (doc.uri.scheme !== "file") return;

    const content = doc.getText();
    if (!content.trim()) return;

    const truncated = content.length > MAX_CONTEXT_BYTES
      ? content.slice(0, MAX_CONTEXT_BYTES) + "\n... [truncated]"
      : content;

    this.agent.sendContext(doc.uri.fsPath, truncated);
  }

  dispose(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.disposables.forEach((d) => d.dispose());
  }
}
