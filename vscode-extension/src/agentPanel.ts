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
    } else if (msg.type === "set_model") {
      const workspaceDir =
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();
      this.agent.dispose();
      this.agent.start(workspaceDir, msg.model);
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
    <div id="model-picker">
      <span id="model-label">▾ claude-sonnet-4-6</span>
      <div id="model-dropdown">
        <div class="model-group">
          <div class="model-group-label">Claude</div>
          <div class="model-option" data-model="claude-sonnet-4-6">Sonnet 4.6</div>
          <div class="model-option" data-model="claude-opus-4-7">Opus 4.7</div>
          <div class="model-option" data-model="claude-haiku-4-5-20251001">Haiku 4.5</div>
        </div>
        <div class="model-group">
          <div class="model-group-label">GPT</div>
          <div class="model-option" data-model="gpt-5.5">GPT-5.5</div>
          <div class="model-option" data-model="gpt-4o">GPT-4o</div>
          <div class="model-option" data-model="gpt-4o-mini">GPT-4o Mini</div>
        </div>
        <div class="model-group">
          <div class="model-group-label">Gemini</div>
          <div class="model-option" data-model="gemini/gemini-2.5-pro">Gemini 2.5 Pro</div>
          <div class="model-option" data-model="gemini/gemini-2.5-flash">Gemini 2.5 Flash</div>
        </div>
        <div class="model-group">
          <div class="model-group-label">Azure OpenAI</div>
          <div class="model-option" data-model="azure/gpt-4o">Azure GPT-4o</div>
          <div class="model-option" data-model="azure/gpt-4o-mini">Azure GPT-4o Mini</div>
        </div>
      </div>
    </div>
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
