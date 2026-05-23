import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

function parseEnvFile(filePath: string): Record<string, string> {
  let content: string;
  try {
    content = fs.readFileSync(filePath, "utf8");
  } catch {
    return {};
  }

  const result: Record<string, string> = {};
  for (const rawLine of content.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    result[key] = val;
  }
  return result;
}

export function loadWorkspaceEnv(): Record<string, string> {
  const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!workspaceRoot) return {};
  return parseEnvFile(path.join(workspaceRoot, ".env"));
}

export function resolveKey(
  settingName: string,
  envVarName: string,
  dotenvVars: Record<string, string>
): string {
  const cfg = vscode.workspace.getConfiguration("agentcode");
  return (
    cfg.get<string>(settingName, "") ||
    dotenvVars[envVarName] ||
    process.env[envVarName] ||
    ""
  );
}
