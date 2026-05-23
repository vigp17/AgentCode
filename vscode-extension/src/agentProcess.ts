import { spawn, ChildProcess } from "child_process";
import * as readline from "readline";
import * as vscode from "vscode";
import { ServerMessage, ClientMessage } from "./protocol.js";
import { loadWorkspaceEnv, resolveKey } from "./envLoader.js";

export type MessageHandler = (msg: ServerMessage) => void;

export class AgentProcess {
  private process: ChildProcess | null = null;
  private handlers: MessageHandler[] = [];
  private ready = false;
  private disposing = false;

  onMessage(handler: MessageHandler): void {
    this.handlers.push(handler);
  }

  private emit(msg: ServerMessage): void {
    this.handlers.forEach((h) => h(msg));
  }

  start(workspaceDir: string, modelOverride?: string): void {
    const cfg = vscode.workspace.getConfiguration("agentcode");
    const execPath = cfg.get<string>("executablePath", "agentcode");
    const model = modelOverride ?? cfg.get<string>("model", "claude-sonnet-4-6");

    const dotenvVars = loadWorkspaceEnv();
    // Only inject keys that are actually resolved — passing an empty string would
    // overwrite the env var and prevent Python's own load_dotenv() from reading .env.
    const resolved: Record<string, string> = {};
    for (const [setting, envVar] of [
      ["anthropicApiKey", "ANTHROPIC_API_KEY"],
      ["azureApiKey",     "AZURE_API_KEY"],
      ["azureEndpoint",   "AZURE_API_BASE"],
      ["openaiApiKey",    "OPENAI_API_KEY"],
      ["geminiApiKey",    "GEMINI_API_KEY"],
    ] as const) {
      const val = resolveKey(setting, envVar, dotenvVars);
      if (val) resolved[envVar] = val;
    }
    const env: NodeJS.ProcessEnv = { ...process.env, ...resolved };

    this.process = spawn(
      execPath,
      ["--server", "--model", model, "--dir", workspaceDir],
      { env, stdio: ["pipe", "pipe", "pipe"] }
    );

    const rl = readline.createInterface({ input: this.process.stdout! });
    rl.on("line", (line) => {
      if (!line.trim()) return;
      try {
        const msg = JSON.parse(line) as ServerMessage;
        if (msg.type === "ready") this.ready = true;
        this.emit(msg);
      } catch {
        // non-JSON output from Python (e.g. tracebacks) — ignore
      }
    });

    this.process.on("error", (err: NodeJS.ErrnoException) => {
      const hint = err.code === "ENOENT"
        ? `'${execPath}' not found. Set the correct path in AgentCode settings (agentcode.executablePath).`
        : err.message;
      this.emit({ type: "error", message: hint });
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      const text = data.toString().trim();
      if (text) {
        this.emit({ type: "error", message: text });
      }
    });

    this.process.on("exit", (code) => {
      this.ready = false;
      if (!this.disposing && code !== 0 && code !== null) {
        this.emit({ type: "error", message: `AgentCode process exited (code ${code}). Check your API keys and Python path.` });
      }
      this.disposing = false;
    });
  }

  send(msg: ClientMessage): void {
    if (!this.process?.stdin?.writable) {
      this.emit({ type: "error", message: "AgentCode process is not running. Check your executable path and API keys." });
      return;
    }
    this.process.stdin.write(JSON.stringify(msg) + "\n");
  }

  sendMessage(content: string): void {
    this.send({ type: "message", content });
  }

  sendContext(filePath: string, content: string): void {
    this.send({ type: "context", file_path: filePath, content });
  }

  sendPermissionResponse(approved: boolean): void {
    this.send({ type: "permission_response", approved });
  }

  sendClear(): void {
    this.send({ type: "clear" });
  }

  isReady(): boolean {
    return this.ready;
  }

  dispose(): void {
    this.disposing = true;
    this.process?.kill();
    this.process = null;
    this.ready = false;
  }
}
