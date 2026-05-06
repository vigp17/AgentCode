import { spawn, ChildProcess } from "child_process";
import * as readline from "readline";
import * as vscode from "vscode";
import { ServerMessage, ClientMessage } from "./protocol.js";

export type MessageHandler = (msg: ServerMessage) => void;

export class AgentProcess {
  private process: ChildProcess | null = null;
  private handlers: MessageHandler[] = [];
  private ready = false;

  onMessage(handler: MessageHandler): void {
    this.handlers.push(handler);
  }

  private emit(msg: ServerMessage): void {
    this.handlers.forEach((h) => h(msg));
  }

  start(workspaceDir: string): void {
    const cfg = vscode.workspace.getConfiguration("agentcode");
    const execPath = cfg.get<string>("executablePath", "agentcode");
    const model = cfg.get<string>("model", "claude-sonnet-4-6");

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      ANTHROPIC_API_KEY: cfg.get<string>("anthropicApiKey", "") || process.env.ANTHROPIC_API_KEY || "",
      OPENAI_API_KEY: cfg.get<string>("openaiApiKey", "") || process.env.OPENAI_API_KEY || "",
      GEMINI_API_KEY: cfg.get<string>("geminiApiKey", "") || process.env.GEMINI_API_KEY || "",
    };

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

    this.process.stderr?.on("data", (data: Buffer) => {
      const text = data.toString().trim();
      if (text) {
        this.emit({ type: "error", message: text });
      }
    });

    this.process.on("exit", (code) => {
      this.ready = false;
      if (code !== 0) {
        this.emit({ type: "error", message: `AgentCode process exited (code ${code}). Check your API keys and Python path.` });
      }
    });
  }

  send(msg: ClientMessage): void {
    if (!this.process?.stdin?.writable) return;
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
    this.process?.kill();
    this.process = null;
    this.ready = false;
  }
}
