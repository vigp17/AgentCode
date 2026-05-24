import * as vscode from "vscode";
import * as https from "https";
import * as http from "http";
import { URL } from "url";
import { loadWorkspaceEnv, resolveKey } from "./envLoader.js";

interface CacheEntry {
  completion: string;
}

export class InlineCompletionProvider
  implements vscode.InlineCompletionItemProvider
{
  private debounceTimer: ReturnType<typeof setTimeout> | undefined;
  private readonly cache = new Map<string, CacheEntry>();
  private readonly CACHE_SIZE = 20;
  private readonly DEBOUNCE_MS = 300;

  constructor(private readonly statusBar: vscode.StatusBarItem) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ): Promise<vscode.InlineCompletionList | undefined> {
    const config = vscode.workspace.getConfiguration("agentcode");
    if (!config.get<boolean>("inlineCompletions.enabled", true)) {
      return undefined;
    }

    const dotenvVars = loadWorkspaceEnv();
    const endpoint = config.get<string>("inlineCompletions.endpoint", "").trim();
    const model = config.get<string>("inlineCompletions.model", "").trim();
    const endpointKey = resolveKey("inlineCompletions.apiKey", "AGENTCODE_COMPLETION_API_KEY", dotenvVars);
    const anthropicKey = resolveKey("anthropicApiKey", "ANTHROPIC_API_KEY", dotenvVars);

    // Need either a custom endpoint (with model) OR an Anthropic key
    const useCustom = endpoint && model;
    if (!useCustom && !anthropicKey) return undefined;

    const text = document.getText();
    const offset = document.offsetAt(position);
    const prefix = text.slice(Math.max(0, offset - 2000), offset);
    const suffix = text.slice(offset, Math.min(text.length, offset + 500));

    // Skip if cursor is in the middle of a word (only complete at word boundaries)
    const charBefore = prefix.slice(-1);
    if (charBefore && /\w/.test(charBefore) && !prefix.endsWith(".")) {
      const linePrefix = document.lineAt(position.line).text.slice(0, position.character);
      if (linePrefix.trimEnd() !== linePrefix) return undefined;
    }

    const cacheKey = prefix.slice(-300) + "|||" + suffix.slice(0, 100);

    if (this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      if (!cached.completion) return undefined;
      return { items: [new vscode.InlineCompletionItem(cached.completion)] };
    }

    if (this.debounceTimer) clearTimeout(this.debounceTimer);

    return new Promise((resolve) => {
      this.debounceTimer = setTimeout(async () => {
        if (token.isCancellationRequested) {
          resolve(undefined);
          return;
        }

        this.statusBar.text = "$(loading~spin) AgentCode";
        this.statusBar.show();

        try {
          const completion = useCustom
            ? await this.fetchFromCustomEndpoint(
                prefix, suffix, endpoint, model, endpointKey,
                document.languageId, token,
              )
            : await this.fetchFromAnthropic(
                prefix, suffix, anthropicKey, document.languageId, token,
              );

          this.addToCache(cacheKey, completion);

          if (!completion || token.isCancellationRequested) {
            resolve(undefined);
          } else {
            resolve({ items: [new vscode.InlineCompletionItem(completion)] });
          }
        } catch {
          resolve(undefined);
        } finally {
          this.statusBar.hide();
        }
      }, this.DEBOUNCE_MS);
    });
  }

  private addToCache(key: string, completion: string): void {
    if (this.cache.size >= this.CACHE_SIZE) {
      const firstKey = this.cache.keys().next().value as string;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, { completion });
  }

  private fetchFromAnthropic(
    prefix: string,
    suffix: string,
    apiKey: string,
    languageId: string,
    token: vscode.CancellationToken
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const body = JSON.stringify({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 100,
        system:
          "You are a code completion engine. Output only the completion text with no explanation, no markdown fences, and no surrounding quotes. Single line only.",
        messages: [
          {
            role: "user",
            content: `Complete this ${languageId} code at the cursor position marked by <CURSOR>. Return only what should be inserted — a single line or empty string.

<prefix>${prefix.slice(-1500)}<CURSOR></prefix>
<suffix>${suffix.slice(0, 400)}</suffix>`,
          },
        ],
      });

      const req = https.request(
        {
          hostname: "api.anthropic.com",
          path: "/v1/messages",
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            "Content-Length": Buffer.byteLength(body),
          },
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            try {
              const parsed = JSON.parse(data);
              const raw: string = parsed?.content?.[0]?.text ?? "";
              const firstLine = raw.split("\n")[0].trimEnd();
              resolve(firstLine);
            } catch {
              resolve("");
            }
          });
        }
      );

      req.on("error", reject);
      token.onCancellationRequested(() => { req.destroy(); resolve(""); });
      req.write(body);
      req.end();
    });
  }

  // OpenAI-compatible endpoint: works with Ollama (localhost:11434/v1),
  // HF Inference Endpoints, vLLM, TGI, or any /v1/chat/completions server.
  // Designed to host Qwen 2.5-Coder fine-tunes (e.g. the AgentCode 3B).
  private fetchFromCustomEndpoint(
    prefix: string,
    suffix: string,
    endpoint: string,
    model: string,
    apiKey: string,
    languageId: string,
    token: vscode.CancellationToken
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      let url: URL;
      try { url = new URL(endpoint.replace(/\/+$/, "") + "/chat/completions"); }
      catch { resolve(""); return; }

      const body = JSON.stringify({
        model,
        max_tokens: 100,
        temperature: 0.2,
        messages: [
          {
            role: "system",
            content: "You are a code completion engine. Output only the completion text with no explanation, no markdown fences, and no surrounding quotes. Single line only.",
          },
          {
            role: "user",
            content: `Complete this ${languageId} code at the cursor position marked by <CURSOR>. Return only what should be inserted — a single line or empty string.

<prefix>${prefix.slice(-1500)}<CURSOR></prefix>
<suffix>${suffix.slice(0, 400)}</suffix>`,
          },
        ],
      });

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        "Content-Length": String(Buffer.byteLength(body)),
      };
      if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

      const transport = url.protocol === "https:" ? https : http;
      const req = transport.request(
        {
          hostname: url.hostname,
          port: url.port || (url.protocol === "https:" ? 443 : 80),
          path: url.pathname + url.search,
          method: "POST",
          headers,
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            try {
              const parsed = JSON.parse(data);
              const raw: string = parsed?.choices?.[0]?.message?.content ?? "";
              const firstLine = raw.split("\n")[0].trimEnd();
              resolve(firstLine);
            } catch {
              resolve("");
            }
          });
        }
      );

      req.on("error", reject);
      token.onCancellationRequested(() => { req.destroy(); resolve(""); });
      req.write(body);
      req.end();
    });
  }

  dispose(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
  }
}
