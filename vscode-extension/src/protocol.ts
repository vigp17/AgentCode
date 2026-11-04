/** A model the backend discovered (provider API, registry, or fine-tune). */
export interface ModelInfo {
  id: string;
  provider: "anthropic" | "openai" | "gemini" | "agentcode" | string;
  label: string;
  created?: number | null;
  source?: "api" | "registry" | "finetuned" | string;
  /** In-house ecosystem fields (axon-1, tandav-1, veda-1, chitra-1). */
  kind?: "chat" | "image" | string;
  available?: boolean;
  supports_effort?: boolean;
  description?: string;
}

export type Effort = "low" | "medium" | "high";

/** Messages the Python server sends to the extension. */
export type ServerMessage =
  | { type: "ready"; model: string; project_dir: string; effort?: Effort }
  | { type: "model_changed"; model: string; effort: Effort; routing: boolean }
  | { type: "text"; delta: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: string }
  | { type: "permission_request"; tool: string; args: Record<string, unknown> }
  | { type: "done"; cost: number }
  | { type: "error"; message: string }
  | { type: "cleared" }
  | { type: "models"; models: ModelInfo[]; current: string }
  | { type: "pong" };

/** Messages the extension sends to the Python server. */
export type ClientMessage =
  | { type: "message"; content: string }
  | { type: "context"; file_path: string; content: string }
  | { type: "permission_response"; approved: boolean }
  | { type: "clear" }
  | { type: "list_models"; refresh: boolean }
  | { type: "set_model"; model: string }
  | { type: "set_effort"; effort: Effort }
  | { type: "ping" };

/** Messages the extension sends to the webview. */
export type WebviewInMessage =
  | { type: "text"; delta: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: string }
  | { type: "permission_request"; tool: string; args: Record<string, unknown> }
  | { type: "done"; cost: number }
  | { type: "error"; message: string }
  | { type: "cleared" }
  | { type: "models"; models: ModelInfo[]; current: string }
  | { type: "model_changed"; model: string; effort: Effort; routing: boolean }
  | { type: "ready"; model: string; effort?: Effort };

/** Messages the webview sends to the extension. */
export type WebviewOutMessage =
  | { type: "send"; content: string }
  | { type: "permission_response"; approved: boolean }
  | { type: "clear" }
  | { type: "set_model"; model: string }
  | { type: "set_effort"; effort: Effort }
  | { type: "refresh_models" };
