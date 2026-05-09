import * as vscode from "vscode";
import { AgentProcess } from "./agentProcess.js";
import { AgentPanel } from "./agentPanel.js";
import { ContextProvider } from "./contextProvider.js";
import { InlineCompletionProvider } from "./inlineCompletion.js";

let agent: AgentProcess | undefined;
let contextProvider: ContextProvider | undefined;
let inlineProvider: InlineCompletionProvider | undefined;

export function activate(ctx: vscode.ExtensionContext): void {
  // Status bar item for inline completion fetch indicator
  const completionStatusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  completionStatusBar.tooltip = "AgentCode: fetching inline completion";
  ctx.subscriptions.push(completionStatusBar);

  // Register inline completion provider
  inlineProvider = new InlineCompletionProvider(completionStatusBar);
  ctx.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      { pattern: "**" },
      inlineProvider
    ),
    inlineProvider
  );

  function getOrStartAgent(): AgentProcess {
    if (!agent) {
      const workspaceDir =
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();
      agent = new AgentProcess();
      agent.start(workspaceDir);
      contextProvider = new ContextProvider(agent);
    }
    return agent;
  }

  function openPanel(): AgentPanel {
    const a = getOrStartAgent();
    return AgentPanel.create(ctx.extensionUri, a, contextProvider!);
  }

  ctx.subscriptions.push(
    // Open panel
    vscode.commands.registerCommand("agentcode.open", () => {
      openPanel();
    }),

    // Toggle inline completions on/off
    vscode.commands.registerCommand("agentcode.toggleInlineCompletions", async () => {
      const config = vscode.workspace.getConfiguration("agentcode");
      const current = config.get<boolean>("inlineCompletions.enabled", true);
      await config.update("inlineCompletions.enabled", !current, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(
        `AgentCode inline completions ${!current ? "enabled" : "disabled"}`
      );
    }),

    // Right-click: Ask about selection
    vscode.commands.registerCommand("agentcode.askAboutSelection", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;

      const selection = editor.document.getText(editor.selection);
      if (!selection.trim()) {
        vscode.window.showWarningMessage("AgentCode: No text selected.");
        return;
      }

      const fileName = editor.document.fileName.split("/").pop() ?? "file";
      const prompt = `In \`${fileName}\`, explain and suggest improvements for this code:\n\`\`\`\n${selection}\n\`\`\``;
      const panel = openPanel();
      panel.sendMessage(prompt);
    }),

    // Right-click: Explain this file
    vscode.commands.registerCommand("agentcode.explainFile", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;

      const fileName = editor.document.fileName.split("/").pop() ?? "this file";
      const panel = openPanel();
      panel.sendMessage(`Explain what \`${fileName}\` does and how it fits into the project.`);
    })
  );
}

export function deactivate(): void {
  agent?.dispose();
  contextProvider?.dispose();
  inlineProvider?.dispose();
}
