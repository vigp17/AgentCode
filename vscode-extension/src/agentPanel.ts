import * as vscode from "vscode";
import * as path from "path";
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
  <div id="header">
    <span id="model-label"></span>
    <button id="clear-btn" title="Clear conversation">↺</button>
  </div>
  <div id="messages"></div>
  <div id="input-area">
    <textarea id="input" placeholder="Ask AgentCode…" rows="1"></textarea>
    <button id="send-btn">Send</button>
  </div>
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
