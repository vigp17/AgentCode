#!/usr/bin/env python3
"""
Fine-tune one of AgentCode's three models.

Open-weight (default) — LoRA on Qwen2.5-Coder with Unsloth, on your own GPU:

    pip install -r finetune/requirements.txt
    python finetune/train.py light           # agentcode-light  (1.5B, 16GB GPU)
    python finetune/train.py medium          # agentcode-medium (7B,  24-48GB GPU)
    python finetune/train.py heavy           # agentcode-heavy  (32B, 80GB GPU)
    python finetune/train.py medium --gguf q4_k_m   # also export for Ollama

Hosted — OpenAI's fine-tuning API, no GPU needed:

    python finetune/train.py medium --backend openai --base <fine-tunable gpt model>

  When the job finishes, the resulting `ft:` model appears in AgentCode's
  model picker automatically (model discovery lists your fine-tunes).

Reads finetune/data/<model>/{train,eval}.jsonl from prepare_data.py and writes
finetune/out/<model>/{lora,merged}. Serve `merged` with serve.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_data import load_config  # noqa: E402

# Qwen2.5 chat template markers. Loss is computed only on assistant turns;
# system prompts, user messages and tool results are context, not targets.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def to_template_messages(messages: list[dict]) -> list[dict]:
    """OpenAI format stores tool-call arguments as a JSON string; HF chat
    templates (Qwen's included) expect a dict and re-serialize it."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {**tc, "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"].get("arguments") or "{}"),
                }}
                for tc in m["tool_calls"]
            ]
        out.append(m)
    return out


def render(rows: list[dict], tokenizer, max_len: int) -> tuple[list[dict], int]:
    texts, too_long = [], 0
    for r in rows:
        text = tokenizer.apply_chat_template(
            to_template_messages(r["messages"]), tools=r.get("tools") or None, tokenize=False,
        )
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_len:
            too_long += 1
            continue
        texts.append({"text": text})
    return texts, too_long


# ── Open-weight: Unsloth + TRL ────────────────────────────────────────────────

def train_unsloth(cfg: dict, data_dir: Path, out_dir: Path, gguf: str | None, max_steps: int) -> None:
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:
        raise SystemExit("Unsloth not installed: pip install -r finetune/requirements.txt (needs a CUDA GPU)")
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    t = cfg["train"]
    max_len = t["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"], max_seq_length=max_len,
        load_in_4bit=t.get("load_in_4bit", True), dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.0),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=17,
    )

    train_rows, skipped = render(read_jsonl(data_dir / "train.jsonl"), tokenizer, max_len)
    eval_rows, _ = render(read_jsonl(data_dir / "eval.jsonl"), tokenizer, max_len)
    if not train_rows:
        raise SystemExit(f"No training data in {data_dir}. Run finetune/prepare_data.py first.")
    print(f"train={len(train_rows)} eval={len(eval_rows)} (skipped {skipped} over {max_len} tokens)")

    sft_args = dict(
        output_dir=str(out_dir / "checkpoints"),
        dataset_text_field="text",
        num_train_epochs=t["epochs"],
        max_steps=max_steps,
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t.get("warmup_ratio", 0.03),
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_rows else "no",
        bf16=True,
        packing=False,
        report_to="none",
        seed=17,
    )
    try:  # TRL >= 0.20 renamed max_seq_length → max_length
        sft_config = SFTConfig(max_length=max_len, **sft_args)
    except TypeError:
        sft_config = SFTConfig(max_seq_length=max_len, **sft_args)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        args=sft_config,
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
    )
    trainer.train()

    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))
    # Merged 16-bit weights: what vLLM / SGLang serve (see serve.sh).
    model.save_pretrained_merged(str(out_dir / "merged"), tokenizer, save_method="merged_16bit")
    if gguf:
        model.save_pretrained_gguf(str(out_dir / "gguf"), tokenizer, quantization_method=gguf)
    (out_dir / "agentcode.json").write_text(json.dumps(
        {"name": cfg["name"], "tier": cfg["tier"], "base_model": cfg["base_model"]}, indent=2))
    print(f"\n✓ {cfg['name']} saved to {out_dir}. Serve it with: finetune/serve.sh {cfg['tier']}")


# ── Hosted: OpenAI fine-tuning API ────────────────────────────────────────────

def train_openai(cfg: dict, data_dir: Path, base: str | None) -> None:
    if not base:
        raise SystemExit("--backend openai needs --base <model> (a fine-tunable OpenAI model).")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    client = OpenAI()
    train = client.files.create(file=(data_dir / "train.jsonl").open("rb"), purpose="fine-tune")
    val = None
    if (data_dir / "eval.jsonl").exists() and (data_dir / "eval.jsonl").stat().st_size:
        val = client.files.create(file=(data_dir / "eval.jsonl").open("rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        model=base, training_file=train.id, validation_file=val.id if val else None,
        suffix=cfg["name"],
    )
    print(f"✓ Started OpenAI fine-tuning job {job.id} for {cfg['name']} on {base}.")
    print("  When it finishes, the ft:… model shows up in AgentCode's model list automatically")
    print("  (/models refresh in the CLI, or ↻ in the VS Code picker).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="light | medium | heavy (or agentcode-<tier>)")
    ap.add_argument("--backend", choices=["unsloth", "openai"], default="unsloth")
    ap.add_argument("--base", help="override the base model (required for --backend openai)")
    ap.add_argument("--data", type=Path, help="dir with train.jsonl/eval.jsonl")
    ap.add_argument("--out", type=Path, help="output dir (default finetune/out/<model>)")
    ap.add_argument("--gguf", help="also export GGUF for Ollama, e.g. q4_k_m")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.model.removeprefix("agentcode-"))
    data_dir = args.data or HERE / "data" / cfg["name"]
    if args.backend == "openai":
        train_openai(cfg, data_dir, args.base)
        return
    if args.base:
        cfg["base_model"] = args.base
    out_dir = args.out or HERE / "out" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_unsloth(cfg, data_dir, out_dir, args.gguf, args.max_steps)


if __name__ == "__main__":
    main()
