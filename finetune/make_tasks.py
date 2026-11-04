#!/usr/bin/env python3
"""
Generate the training task list for the AgentCode lineup.

    python finetune/make_tasks.py                    # writes finetune/tasks.jsonl
    python finetune/make_tasks.py --out my.jsonl --seed 3

Each task is {"repo", "task", "category", "effort", "language"}:

  category  coding    everyday work → axon-1 (effort low/medium/high varies)
            research  read-only, cited write-ups → veda-1
            long      multi-file, many-step work → tandav-1 (and axon-1 "high")

prepare_data.py decides which model a trajectory trains by what the agent
actually did, not by this label — the category only steers the task mix.

Tasks are templates that work on any repo (the agent explores to ground
them) plus repo-specific tasks for well-known projects. Add repos to REPOS,
or tasks to the template pools, and re-run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Repos ─────────────────────────────────────────────────────────────────────
# Small-to-medium, well-maintained projects with test suites that run without
# external services. (url, language)

REPOS: list[tuple[str, str]] = [
    # Python
    ("https://github.com/psf/requests", "python"),
    ("https://github.com/pallets/click", "python"),
    ("https://github.com/pallets/flask", "python"),
    ("https://github.com/pallets/itsdangerous", "python"),
    ("https://github.com/pallets/markupsafe", "python"),
    ("https://github.com/pallets/jinja", "python"),
    ("https://github.com/encode/httpx", "python"),
    ("https://github.com/Textualize/rich", "python"),
    ("https://github.com/fastapi/typer", "python"),
    ("https://github.com/fastapi/sqlmodel", "python"),
    ("https://github.com/python-attrs/attrs", "python"),
    ("https://github.com/marshmallow-code/marshmallow", "python"),
    ("https://github.com/jazzband/tablib", "python"),
    ("https://github.com/Delgan/loguru", "python"),
    ("https://github.com/tqdm/tqdm", "python"),
    ("https://github.com/more-itertools/more-itertools", "python"),
    ("https://github.com/hynek/structlog", "python"),
    ("https://github.com/pytest-dev/pluggy", "python"),
    ("https://github.com/python-poetry/tomlkit", "python"),
    ("https://github.com/dateutil/dateutil", "python"),
    ("https://github.com/arrow-py/arrow", "python"),
    ("https://github.com/mahmoud/boltons", "python"),
    ("https://github.com/jd/tenacity", "python"),
    # JavaScript / TypeScript
    ("https://github.com/expressjs/express", "javascript"),
    ("https://github.com/axios/axios", "javascript"),
    ("https://github.com/koajs/koa", "javascript"),
    ("https://github.com/tj/commander.js", "javascript"),
    ("https://github.com/yargs/yargs", "javascript"),
    ("https://github.com/mde/ejs", "javascript"),
    ("https://github.com/uuidjs/uuid", "typescript"),
    ("https://github.com/ai/nanoid", "javascript"),
    ("https://github.com/chalk/chalk", "javascript"),
    ("https://github.com/lukeed/clsx", "javascript"),
    ("https://github.com/sindresorhus/ky", "typescript"),
    ("https://github.com/sindresorhus/p-limit", "javascript"),
    ("https://github.com/colinhacks/zod", "typescript"),
    ("https://github.com/immerjs/immer", "typescript"),
    ("https://github.com/pmndrs/zustand", "typescript"),
    # Go
    ("https://github.com/spf13/cobra", "go"),
    ("https://github.com/go-chi/chi", "go"),
    ("https://github.com/gin-gonic/gin", "go"),
    ("https://github.com/sirupsen/logrus", "go"),
    ("https://github.com/urfave/cli", "go"),
    ("https://github.com/gorilla/mux", "go"),
    ("https://github.com/joho/godotenv", "go"),
    ("https://github.com/stretchr/testify", "go"),
    # Rust
    ("https://github.com/dtolnay/anyhow", "rust"),
    ("https://github.com/dtolnay/semver", "rust"),
    ("https://github.com/BurntSushi/walkdir", "rust"),
    ("https://github.com/rust-lang/log", "rust"),
]

TEST_CMD = {
    "python": "pytest (install the package and its test dependencies first if needed)",
    "javascript": "npm test (run npm install first if needed)",
    "typescript": "npm test (run npm install first if needed)",
    "go": "go test ./...",
    "rust": "cargo test",
}

TYPES_TASK = {
    "python": "Find a small module whose functions lack type hints and add accurate type "
              "hints to it without changing behavior. Run {test} afterwards.",
    "javascript": "Find a small module without JSDoc and add accurate JSDoc type annotations "
                  "to its exported functions. Don't change behavior. Run {test} afterwards.",
    "typescript": "Find a place that uses `any` (or an overly loose type) and replace it with "
                  "a precise type. Make sure the type checker and {test} still pass.",
    "go": "Find an exported function or type without a doc comment and add idiomatic Go doc "
          "comments to every exported identifier in that file. Run go vet and {test}.",
    "rust": "Find a public item without rustdoc and add doc comments (with a short example "
            "where it helps) to the public items in that module. Run {test}, including doc tests.",
}

# ── Template pools ────────────────────────────────────────────────────────────

CODING_LOW = [
    "Where is this project's version number defined? Answer with the file and line.",
    "Explain in a few sentences what this project does and what its main entry point is.",
    "List this project's runtime dependencies and say briefly what each is used for.",
    "Find a public function that has no docstring or doc comment and add a concise, "
    "accurate one. Don't change any behavior.",
    "How do I run this project's tests? Find out from the repo and give me the exact command.",
    "What license is this project under, and where is it declared?",
    "Find the function that is the main public entry point of this library and explain its "
    "parameters and return value.",
]

CODING_MEDIUM = [
    "Run the test suite with {test}. If anything fails, find the root cause and fix it. "
    "If everything passes, report the result and how many tests ran.",
    "Find a utility function that has no direct unit test, add a focused test for it "
    "(including one edge case), and run it with {test} to make sure it passes.",
    "{types}",
    "Find a public function that accepts input it doesn't validate. Add validation with a "
    "clear error message and a test for the invalid case. Run {test}.",
    "Find an unclear error message in this codebase and rewrite it so it says what went "
    "wrong and how to fix it. Update any tests that assert on the old message, then run {test}.",
    "Find two places with duplicated logic and extract it into one shared helper. Keep "
    "behavior identical and run {test} afterwards.",
    "Find a function longer than about 50 lines and split it into smaller, well-named "
    "functions without changing behavior. Verify with {test}.",
    "Add a test for an edge case that the existing tests miss (empty input, very large "
    "input, unicode, or None/null). If it exposes a bug, fix the bug. Run {test}.",
    "Find a TODO or FIXME comment that is small enough to resolve now, resolve it, and "
    "add or update a test. Run {test}.",
]

LONG = [
    "Pick one core module and bring its test coverage up: add tests until every public "
    "function and its main edge cases are exercised. Fix any bugs you find. Run the full "
    "suite with {test} at the end.",
    "Audit the codebase for unhandled error cases around I/O (files, network, parsing). Fix "
    "the three most important ones, add a test for each, and run {test}.",
    "Find every place that reads configuration or environment variables directly and route "
    "them through a single helper. Keep behavior identical and keep {test} passing.",
    "Refactor the largest function in the codebase into smaller units with clear names, "
    "updating call sites as needed. Behavior must not change; prove it with {test}.",
    "Find a public API that is inconsistent with its siblings (naming, argument order, or "
    "return type) and make it consistent across the codebase in a backward-compatible way "
    "(keep the old form working, with a deprecation warning if the language supports it). "
    "Add tests and run {test}.",
    "Add structured, consistent logging/debug output to the main code path in a way that is "
    "off by default, document how to enable it in the README, and add a test. Run {test}.",
]

RESEARCH = [
    "Write a short architecture overview of this project: the main modules, how they depend "
    "on each other, and how data flows through them. Cite file paths with line numbers "
    "(file:line) for every claim. This is read-only: do not modify any files.",
    "Trace step by step what happens from the public entry point to the core logic for the "
    "most common use case. Cite file:line for each step. Read-only: do not modify files.",
    "Document this project's error-handling strategy: which errors or exceptions exist, "
    "where they are raised, and where they are caught or converted. Cite file:line. "
    "Read-only.",
    "Write an onboarding guide for a new contributor: how the code is organized, how the "
    "tests are structured and run, and good first areas to work on. Cite the files you "
    "reference. Read-only.",
    "Identify the extension points (plugins, hooks, middleware, interfaces, callbacks) and "
    "explain how a user would extend this project, with file:line citations. Read-only.",
    "Write an engineering spec for a feature this project is missing but could reasonably "
    "add: motivation, proposed API, which files would change and why (cite file:line for "
    "the current code), and risks. Don't implement it; this is a read-only research task.",
    "Compare how this project handles configuration or options with the conventions of its "
    "ecosystem, citing the relevant code (file:line). Point out anything surprising. "
    "Read-only.",
]

# ── Repo-specific tasks ───────────────────────────────────────────────────────
# (repo url, category, effort, task)

SPECIFIC: list[tuple[str, str, str, str]] = [
    ("https://github.com/psf/requests", "research", "medium",
     "Explain how a redirect response is processed, including how the method and body "
     "change for 301/302/303/307/308. Cite file:line. Read-only."),
    ("https://github.com/psf/requests", "coding", "medium",
     "Add a test verifying that a POST answered with a 303 redirect is re-sent as a GET "
     "without a body (if that isn't already covered). Run the relevant tests."),
    ("https://github.com/psf/requests", "research", "medium",
     "Explain how authentication handlers are attached and applied to a request, with "
     "file:line citations. Read-only."),
    ("https://github.com/pallets/click", "research", "medium",
     "Explain how parameter parsing flows from the command line to the callback's "
     "arguments. Cite file:line. Read-only."),
    ("https://github.com/pallets/click", "coding", "medium",
     "Add a test for a command with an option that has multiple=True and a default, "
     "checking both the default and repeated values. Run the tests."),
    ("https://github.com/pallets/click", "research", "low",
     "How do command groups register subcommands? Point me to the code."),
    ("https://github.com/pallets/flask", "research", "medium",
     "Explain how the application and request contexts are pushed and popped during a "
     "request, citing file:line. Read-only."),
    ("https://github.com/pallets/flask", "coding", "medium",
     "Add a test that a blueprint's error handler takes precedence over the app's handler "
     "for errors raised inside that blueprint (if not already covered). Run the tests."),
    ("https://github.com/pallets/jinja", "research", "high",
     "Explain how a template goes from source text to executed Python: lexer, parser, "
     "compiler, runtime. Cite file:line for each stage. Read-only."),
    ("https://github.com/pallets/itsdangerous", "coding", "low",
     "Explain what happens when a timed signature has expired: which exception is raised "
     "and what information it carries."),
    ("https://github.com/encode/httpx", "research", "medium",
     "Research how connection pooling works and write a short spec of the design, citing "
     "file:line for each claim. Read-only."),
    ("https://github.com/encode/httpx", "coding", "medium",
     "Add a test for a timeout configured per request overriding the client's default "
     "timeout (if not already covered). Run the relevant tests."),
    ("https://github.com/Textualize/rich", "research", "medium",
     "Explain how console markup like [bold red]...[/] is parsed into styled segments. "
     "Cite file:line. Read-only."),
    ("https://github.com/Textualize/rich", "coding", "medium",
     "Add tests for markup edge cases: unclosed tags, escaped brackets and nested styles. "
     "Fix any bug you find. Run the relevant tests."),
    ("https://github.com/fastapi/typer", "research", "medium",
     "Explain how Python type hints on a function are turned into CLI options and "
     "arguments. Cite file:line. Read-only."),
    ("https://github.com/python-attrs/attrs", "research", "high",
     "Explain how a class decorated with @define gets its generated methods (__init__, "
     "__repr__, __eq__). Cite file:line. Read-only."),
    ("https://github.com/marshmallow-code/marshmallow", "coding", "medium",
     "Add a test for nested schema validation errors, checking that the error messages "
     "are keyed by the nested field path. Run the tests."),
    ("https://github.com/tqdm/tqdm", "research", "medium",
     "Explain how tqdm decides when to refresh the progress bar (miniters, mininterval). "
     "Cite file:line. Read-only."),
    ("https://github.com/more-itertools/more-itertools", "coding", "low",
     "Find a function whose docstring lacks an example and add a doctest example to it. "
     "Run the doctests."),
    ("https://github.com/jd/tenacity", "research", "medium",
     "Explain how stop, wait and retry strategies are combined when deciding whether to "
     "retry. Cite file:line. Read-only."),
    ("https://github.com/jd/tenacity", "coding", "medium",
     "Add a test that a retry with stop_after_attempt(3) calls the function exactly three "
     "times and then raises RetryError (if not already covered). Run the tests."),
    ("https://github.com/arrow-py/arrow", "coding", "medium",
     "Add tests for humanize() around boundary values (exactly 1 minute, 59 seconds, 1 "
     "day). Fix any off-by-one you find. Run the tests."),
    ("https://github.com/dateutil/dateutil", "research", "medium",
     "Explain how the fuzzy date parser tokenizes and interprets an ambiguous date like "
     "01/02/03. Cite file:line. Read-only."),
    ("https://github.com/expressjs/express", "research", "medium",
     "Document how the router matches a request path to middleware and route handlers, as "
     "an onboarding spec with file:line citations. Read-only."),
    ("https://github.com/expressjs/express", "coding", "medium",
     "Add a test that an error thrown in an async route handler reaches the error-handling "
     "middleware (if supported and not already covered). Run npm test."),
    ("https://github.com/expressjs/express", "long", "high",
     "Refactor the examples/auth example to use async/await instead of callbacks, keeping "
     "behavior identical, and make sure any related tests pass."),
    ("https://github.com/axios/axios", "research", "medium",
     "Explain how request and response interceptors are chained and executed. Cite "
     "file:line. Read-only."),
    ("https://github.com/axios/axios", "coding", "medium",
     "Add a test checking that a request interceptor can modify headers before the request "
     "is sent (if not already covered). Run npm test."),
    ("https://github.com/koajs/koa", "research", "medium",
     "Explain how Koa composes middleware into the 'onion' model and how errors propagate. "
     "Cite file:line. Read-only."),
    ("https://github.com/tj/commander.js", "coding", "medium",
     "Add a test for a variadic argument combined with an option, verifying both are "
     "parsed. Run npm test."),
    ("https://github.com/yargs/yargs", "research", "medium",
     "Explain how yargs builds and displays the help output for a command. Cite file:line. "
     "Read-only."),
    ("https://github.com/colinhacks/zod", "research", "high",
     "Explain how a schema like z.object({...}).parse(x) validates input and collects "
     "issues. Cite file:line. Read-only."),
    ("https://github.com/colinhacks/zod", "coding", "medium",
     "Add tests for error messages on nested object/array paths, checking the issue path "
     "is correct. Run the tests."),
    ("https://github.com/immerjs/immer", "research", "high",
     "Explain how produce() tracks changes to a draft and builds the next immutable state. "
     "Cite file:line. Read-only."),
    ("https://github.com/pmndrs/zustand", "research", "medium",
     "Explain how a store notifies subscribers and how selectors avoid unnecessary "
     "re-renders. Cite file:line. Read-only."),
    ("https://github.com/sindresorhus/ky", "coding", "medium",
     "Add a test that retries respect the configured limit and only retry on the documented "
     "status codes (if not already covered). Run the tests."),
    ("https://github.com/spf13/cobra", "research", "medium",
     "Explain how cobra finds and executes the right subcommand from os.Args, including "
     "persistent flags. Cite file:line. Read-only."),
    ("https://github.com/spf13/cobra", "coding", "medium",
     "Add a test that a persistent flag defined on the root command is available in a "
     "nested subcommand (if not already covered). Run go test ./..."),
    ("https://github.com/go-chi/chi", "research", "medium",
     "Explain how chi's radix-tree router matches URL patterns with parameters. Cite "
     "file:line. Read-only."),
    ("https://github.com/gin-gonic/gin", "research", "medium",
     "Explain how gin's context is pooled and reused across requests. Cite file:line. "
     "Read-only."),
    ("https://github.com/gin-gonic/gin", "coding", "medium",
     "Add a test for binding a JSON body with a missing required field, checking the "
     "validation error. Run go test ./..."),
    ("https://github.com/sirupsen/logrus", "coding", "medium",
     "Add a test for a custom hook that is only fired for Error level and above. Run "
     "go test ./..."),
    ("https://github.com/joho/godotenv", "coding", "medium",
     "Add tests for parsing quoted values with escaped quotes and inline comments. Fix any "
     "bug you find. Run go test ./..."),
    ("https://github.com/stretchr/testify", "research", "medium",
     "Explain how assert.Equal produces its diff output on failure. Cite file:line. "
     "Read-only."),
    ("https://github.com/dtolnay/anyhow", "research", "high",
     "Explain how anyhow::Error stores the underlying error and its context chain, and how "
     "downcasting works. Cite file:line. Read-only."),
    ("https://github.com/dtolnay/semver", "coding", "medium",
     "Add tests for version comparison with pre-release identifiers (1.0.0-alpha < "
     "1.0.0-alpha.1 < 1.0.0). Run cargo test."),
    ("https://github.com/BurntSushi/walkdir", "research", "medium",
     "Explain how walkdir handles symlinks and loop detection. Cite file:line. Read-only."),
    ("https://github.com/rust-lang/log", "research", "low",
     "How does the log crate let a library log without choosing a logger? Point me to the "
     "relevant code."),
]


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []

    def add(repo, lang, category, effort, text):
        test = TEST_CMD[lang]
        text = text.replace("{types}", TYPES_TASK[lang]).replace("{test}", test)
        tasks.append({"repo": repo, "task": text, "category": category,
                      "effort": effort, "language": lang})

    for i, (repo, lang) in enumerate(REPOS):
        # Rotate through the pools so each repo gets a different mix.
        add(repo, lang, "coding", "low", CODING_LOW[i % len(CODING_LOW)])
        for k in range(3):
            add(repo, lang, "coding", "medium", CODING_MEDIUM[(i * 3 + k) % len(CODING_MEDIUM)])
        add(repo, lang, "long", "high", LONG[i % len(LONG)])
        for k in range(2):
            add(repo, lang, "research", "medium", RESEARCH[(i * 2 + k) % len(RESEARCH)])

    lang_of = dict(REPOS)
    for repo, category, effort, text in SPECIFIC:
        add(repo, lang_of[repo], category, effort, text)

    rng.shuffle(tasks)  # mixed order: an interrupted run still covers all kinds
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "tasks.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    tasks = build(args.seed)
    with args.out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    by_cat: dict[str, int] = {}
    for t in tasks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    print(f"{len(tasks)} tasks across {len({t['repo'] for t in tasks})} repos → {args.out}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))


if __name__ == "__main__":
    main()
