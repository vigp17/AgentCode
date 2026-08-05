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
    await closeDiffTab(proposedUri);
  }
}

/**
 * Close only the diff tab we opened. closeActiveEditor would shut whatever the
 * user happened to switch to while the approval prompt was up.
 */
async function closeDiffTab(proposedUri: vscode.Uri): Promise<void> {
  for (const group of vscode.window.tabGroups.all) {
    for (const tab of group.tabs) {
      const input = tab.input;
      if (
        input instanceof vscode.TabInputTextDiff &&
        input.modified.toString() === proposedUri.toString()
      ) {
        await vscode.window.tabGroups.close(tab);
        return;
      }
    }
  }
}

/** Fallback for non-edit_file write tools — simple approve/reject prompt. */
export async function promptFallback(
  toolName: string,
  args: Record<string, unknown>
): Promise<boolean> {
  const argsPreview = Object.entries(args)
    .map(([k, v]) => {
      const s = String(v);
      // Never truncate the command — the user is approving this exact string,
      // and a trailing `; rm -rf ~` would be hidden by an ellipsis.
      if (k === "command" || s.length <= 200) return `${k}: ${s}`;
      return `${k}: ${s.slice(0, 200)}...`;
    })
    .join("\n");

  const choice = await vscode.window.showWarningMessage(
    `AgentCode wants to run: ${toolName}`,
    { modal: true, detail: argsPreview },
    "Allow",
    "Deny"
  );
  return choice === "Allow";
}
