/** Messages the Python server sends to the extension. */
export type ServerMessage =
  | { type: "ready"; model: string; project_dir: string }
  | { type: "text"; delta: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: string }
  | { type: "permission_request"; tool: string; args: Record<string, unknown> }
  | { type: "done"; cost: number }
  | { type: "error"; message: string }
  | { type: "cleared" }
  | { type: "pong" };

/** Messages the extension sends to the Python server. */
export type ClientMessage =
  | { type: "message"; content: string }
  | { type: "context"; file_path: string; content: string }
  | { type: "permission_response"; approved: boolean }
  | { type: "clear" }
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
  | { type: "ready"; model: string };

/** Messages the webview sends to the extension. */
export type WebviewOutMessage =
  | { type: "send"; content: string }
  | { type: "permission_response"; approved: boolean }
  | { type: "clear" }
  | { type: "set_model"; model: string };
