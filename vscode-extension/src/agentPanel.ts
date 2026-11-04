import * as vscode from "vscode";
import { AgentProcess } from "./agentProcess.js";
import { ContextProvider } from "./contextProvider.js";
import { showEditDiff, promptFallback } from "./diffManager.js";
import { WebviewOutMessage } from "./protocol.js";

export class AgentPanel implements vscode.Disposable {
  static currentPanel: AgentPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private readonly agent: AgentProcess;
  private readonly context: ContextProvider;
  private readonly disposables: vscode.Disposable[] = [];

  private constructor(
    panel: vscode.WebviewPanel,
    agent: AgentProcess,
    context: ContextProvider
  ) {
    this.panel = panel;
    this.agent = agent;
    this.context = context;

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      (msg: WebviewOutMessage) => this.onWebviewMessage(msg),
      null,
      this.disposables
    );

    this.agent.onMessage((msg) => {
      if (msg.type === "permission_request") {
        this.handlePermissionRequest(msg.tool, msg.args);
      } else {
        this.panel.webview.postMessage(msg);
      }
    });
  }

  static create(
    extensionUri: vscode.Uri,
    agent: AgentProcess,
    context: ContextProvider
  ): AgentPanel {
    if (AgentPanel.currentPanel) {
      AgentPanel.currentPanel.panel.reveal(vscode.ViewColumn.Two);
      return AgentPanel.currentPanel;
    }

    const panel = vscode.window.createWebviewPanel(
      "agentcode",
      "AgentCode",
      vscode.ViewColumn.Two,
      {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
        retainContextWhenHidden: true,
      }
    );

    panel.webview.html = AgentPanel.buildHtml(panel.webview, extensionUri);
    AgentPanel.currentPanel = new AgentPanel(panel, agent, context);
    return AgentPanel.currentPanel;
  }

  sendMessage(content: string): void {
    this.agent.sendMessage(content);
  }

  private onWebviewMessage(msg: WebviewOutMessage): void {
    if (msg.type === "send") {
      this.agent.sendMessage(msg.content);
    } else if (msg.type === "clear") {
      this.agent.sendClear();
    } else if (msg.type === "permission_response") {
      this.agent.sendPermissionResponse(msg.approved);
    } else if (msg.type === "refresh_models") {
      this.agent.requestModels(true);
    } else if (msg.type === "set_model") {
      this.agent.setModel(msg.model);
    } else if (msg.type === "set_effort") {
      this.agent.setEffort(msg.effort);
    }
  }

  private async handlePermissionRequest(
    tool: string,
    args: Record<string, unknown>
  ): Promise<void> {
    let approved: boolean;
    if (tool === "edit_file") {
      approved = await showEditDiff(args);
    } else {
      approved = await promptFallback(tool, args);
    }
    this.agent.sendPermissionResponse(approved);
    this.panel.webview.postMessage({
      type: "permission_response",
      approved,
    });
  }

  private static buildHtml(
    webview: vscode.Webview,
    extensionUri: vscode.Uri
  ): string {
    const cssUri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, "media", "chat.css")
    );
    const jsUri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, "media", "chat.js")
    );
    const nonce = getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src ${webview.cspSource} 'unsafe-inline';
             script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${cssUri}" rel="stylesheet">
  <title>AgentCode</title>
</head>
<body>
  <header id="header">
    <div id="model-picker">
      <button id="model-button" type="button" aria-haspopup="listbox" aria-expanded="false"
              title="Switch model">
        <span id="model-dot" class="dot"></span>
        <span id="model-name">claude-sonnet-5</span>
        <span class="chevron" aria-hidden="true">⌄</span>
      </button>
      <!-- Built by chat.js: the AgentCode lineup (axon-1, tandav-1, veda-1,
           chitra-1) plus every frontier model the backend discovered. -->
      <div id="model-dropdown" role="listbox"></div>
    </div>
    <button id="clear-btn" class="icon-btn" type="button" title="New chat">＋</button>
  </header>

  <main id="messages">
    <section id="empty-state">
      <h1>What are we building?</h1>
      <p>Ask about this workspace, or type <kbd>/</kbd> for commands.</p>
      <div class="suggestions">
        <button class="suggestion" data-prompt="Explain how this project is structured.">Explain this project</button>
        <button class="suggestion" data-prompt="Find and fix the failing tests.">Fix failing tests</button>
        <button class="suggestion" data-prompt="Review my uncommitted changes for bugs.">Review my changes</button>
      </div>
    </section>
  </main>

  <footer id="composer-wrap">
    <div id="composer">
      <div id="slash-menu" role="listbox" hidden></div>
      <textarea id="input" placeholder="Ask AgentCode…" rows="1"
                aria-autocomplete="list" aria-controls="slash-menu"></textarea>
      <div id="composer-bar">
        <div id="effort" role="radiogroup" aria-label="Reasoning effort" hidden>
          <button class="effort-pill" data-effort="low" role="radio"
                  title="Low — fast answers and autocomplete-style edits">Low</button>
          <button class="effort-pill" data-effort="medium" role="radio"
                  title="Medium — standard coding tasks">Medium</button>
          <button class="effort-pill" data-effort="high" role="radio"
                  title="High — deep multi-file fixes">High</button>
        </div>
        <span id="composer-hint"></span>
        <button id="send-btn" type="button" title="Send (Enter)" aria-label="Send">↑</button>
      </div>
    </div>
  </footer>
  <script nonce="${nonce}" src="${jsUri}"></script>
</body>
</html>`;
  }

  dispose(): void {
    AgentPanel.currentPanel = undefined;
    this.panel.dispose();
    this.context.dispose();
    this.agent.dispose();
    this.disposables.forEach((d) => d.dispose());
  }
}

function getNonce(): string {
  let text = "";
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}
