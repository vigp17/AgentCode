import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";

/**
 * Shows a VS Code diff view for a proposed edit_file change and asks the user
 * to accept or reject. Returns true if accepted.
 */
export async function showEditDiff(args: Record<string, unknown>): Promise<boolean> {
  const filePath = args["path"] as string | undefined;
  const oldString = args["old_string"] as string | undefined;
  const newString = args["new_string"] as string | undefined;

  if (!filePath || oldString === undefined || newString === undefined) {
    return promptFallback(`edit_file`, args);
  }

  // Read the current file content
  let original: string;
  try {
    original = fs.readFileSync(filePath, "utf8");
  } catch {
    return promptFallback(`edit_file`, args);
  }

  if (!original.includes(oldString)) {
    return promptFallback(`edit_file`, args);
  }

  const proposed = original.replace(oldString, newString);

  // Register a content provider for the proposed (right-hand) side
  const scheme = "agentcode-diff";
  const provider = new (class implements vscode.TextDocumentContentProvider {
    provideTextDocumentContent(): string {
      return proposed;
    }
  })();
  const reg = vscode.workspace.registerTextDocumentContentProvider(scheme, provider);

  const originalUri = vscode.Uri.file(filePath);
  const proposedUri = vscode.Uri.from({ scheme, path: filePath });
  const title = `AgentCode → ${path.basename(filePath)}`;

  try {
    await vscode.commands.executeCommand("vscode.diff", originalUri, proposedUri, title);

    const choice = await vscode.window.showInformationMessage(
      `AgentCode wants to edit ${path.basename(filePath)}`,
      { modal: false },
      "Accept",
      "Reject"
    );
    return choice === "Accept";
  } finally {
    reg.dispose();
    // Close the diff tab
    await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
  }
}

/** Fallback for non-edit_file write tools — simple approve/reject prompt. */
export async function promptFallback(
  toolName: string,
  args: Record<string, unknown>
): Promise<boolean> {
  const argsPreview = Object.entries(args)
    .map(([k, v]) => `${k}: ${String(v).slice(0, 80)}`)
    .join("\n");

  const choice = await vscode.window.showWarningMessage(
    `AgentCode wants to run: ${toolName}`,
    { modal: true, detail: argsPreview },
    "Allow",
    "Deny"
  );
  return choice === "Allow";
}
