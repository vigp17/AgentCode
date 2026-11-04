#!/usr/bin/env python3
"""
Fine-tune a model in the AgentCode lineup.

    pip install -r finetune/requirements.txt
    python finetune/train.py axon-1                  # 14B, 1x 24GB GPU
    python finetune/train.py axon-1 --colab          # free Colab/Kaggle T4 (4k context)
    python finetune/train.py veda-1                  # 14B research model
    python finetune/train.py axon-1 --gguf q4_k_m    # also export GGUF for Ollama
    python finetune/train.py axon-1 --push-to-hub you/axon-1   # adapter (+GGUF) to HF

    # tandav-1 (80B MoE) across GPUs with DeepSpeed ZeRO-2:
    accelerate launch --num_processes 4 finetune/train.py tandav-1

  Uses Unsloth when it supports the model on one GPU, otherwise transformers +
  PEFT (always for multi-GPU / DeepSpeed). Force with --trainer unsloth|hf.

Hosted — OpenAI's fine-tuning API, no GPU needed:

    python finetune/train.py axon-1 --backend openai --base <fine-tunable gpt model>

Reads finetune/data/<model>/{train,eval}.jsonl from prepare_data.py and writes
finetune/out/<model>/{lora,merged[,gguf]}. Serve `merged` with serve.sh, or the
GGUF with Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_data import load_config  # noqa: E402

# Qwen chat template markers (ChatML, shared by the axon-1, tandav-1 and veda-1 bases). Loss is
# computed only on assistant turns: system prompts, user messages and tool
# results (which Qwen renders inside user turns) are context, not targets.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
RESPONSE_END = "<|im_end|>"

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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


def render(rows: list[dict], tokenizer, max_len: int, template_kwargs: dict) -> tuple[list[dict], int]:
    texts, too_long = [], 0
    for r in rows:
        text = tokenizer.apply_chat_template(
            to_template_messages(r["messages"]), tools=r.get("tools") or None,
            tokenize=False, **template_kwargs,
        )
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_len:
            too_long += 1
            continue
        texts.append({"text": text})
    return texts, too_long


def assistant_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of assistant replies (content after the assistant
    header, through its <|im_end|>). These are the only trained tokens."""
    spans, i = [], 0
    while (start := text.find(RESPONSE_PART, i)) != -1:
        body = start + len(RESPONSE_PART)
        end = text.find(RESPONSE_END, body)
        end = len(text) if end == -1 else end + len(RESPONSE_END)
        spans.append((body, end))
        i = end
    return spans


def mask_labels(input_ids: list[int], offsets: list[tuple[int, int]], spans) -> list[int]:
    """Labels = input_ids inside assistant spans, -100 everywhere else."""
    labels = []
    for tok, (a, b) in zip(input_ids, offsets):
        inside = any(s <= a and b <= e and b > a for s, e in spans)
        labels.append(tok if inside else -100)
    return labels


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _precision() -> dict:
    """bf16 where the GPU has it (Ampere+), fp16 otherwise (Colab T4, P100)."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return {"bf16": True}
    except ImportError:
        pass
    return {"fp16": True}


def _sft_config(cfg: dict, out_dir: Path, max_steps: int, has_eval: bool, **extra):
    from trl import SFTConfig
    t = cfg["train"]
    ds = t.get("deepspeed")
    if ds and world_size() > 1:
        extra["deepspeed"] = str(HERE / ds)
    args = dict(
        output_dir=str(out_dir / "checkpoints"),
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
        eval_strategy="epoch" if has_eval else "no",
        report_to="none",
        **_precision(),
        seed=17,
        **extra,
    )
    try:  # TRL >= 0.20 renamed max_seq_length → max_length
        return SFTConfig(max_length=t["max_seq_length"], **args)
    except TypeError:
        return SFTConfig(max_seq_length=t["max_seq_length"], **args)


def _load_rows(cfg, data_dir, tokenizer):
    max_len = cfg["train"]["max_seq_length"]
    kw = cfg.get("chat_template_kwargs", {})
    train_rows, skipped = render(read_jsonl(data_dir / "train.jsonl"), tokenizer, max_len, kw)
    eval_rows, _ = render(read_jsonl(data_dir / "eval.jsonl"), tokenizer, max_len, kw)
    if not train_rows:
        raise SystemExit(f"No training data in {data_dir}. Run finetune/prepare_data.py first.")
    print(f"train={len(train_rows)} eval={len(eval_rows)} (skipped {skipped} over {max_len} tokens)")
    return train_rows, eval_rows


def _push(cfg, model, tokenizer, repo: str | None, gguf_dir: Path | None = None) -> None:
    """Upload the LoRA adapter (and GGUF if exported) to the Hugging Face Hub."""
    if not repo:
        return
    token = os.environ.get("HF_TOKEN")
    model.push_to_hub(repo, token=token, private=True)
    tokenizer.push_to_hub(repo, token=token, private=True)
    if gguf_dir and gguf_dir.exists():
        from huggingface_hub import HfApi
        HfApi(token=token).upload_folder(folder_path=str(gguf_dir), repo_id=f"{repo}-GGUF",
                                        repo_type="model", create_pr=False)
    print(f"  pushed to https://huggingface.co/{repo} (private)")


def write_modelfile(gguf_dir: Path) -> Path | None:
    """Minimal Ollama Modelfile for the exported GGUF (Ollama reads the chat
    template from the GGUF metadata). Then: ollama create <name> -f Modelfile"""
    ggufs = sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        return None
    path = gguf_dir / "Modelfile"
    if not path.exists():
        path.write_text(f"FROM ./{ggufs[-1].name}\nPARAMETER num_ctx 16384\n")
    return path


def _write_meta(cfg, out_dir):
    (out_dir / "agentcode.json").write_text(json.dumps(
        {"name": cfg["name"], "tier": cfg["tier"], "base_model": cfg["base_model"],
         "params": cfg.get("params")}, indent=2))
    print(f"\n✓ {cfg['name']} saved to {out_dir}. Serve it with: finetune/serve.sh {cfg['tier']}")


# ── Open-weight: Unsloth + TRL ────────────────────────────────────────────────

class UnslothUnavailable(Exception):
    pass


def train_unsloth(cfg: dict, data_dir: Path, out_dir: Path, gguf: str | None, max_steps: int,
                  push: str | None = None) -> None:
    if world_size() > 1:
        raise UnslothUnavailable("multi-GPU run; Unsloth trains on one GPU")
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
    except ImportError as e:
        raise UnslothUnavailable(f"unsloth not installed ({e})")
    from datasets import Dataset
    from trl import SFTTrainer

    t = cfg["train"]
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["base_model"], max_seq_length=t["max_seq_length"],
            load_in_4bit=t.get("load_in_4bit", True), dtype=None,
        )
    except Exception as e:  # architecture Unsloth doesn't support (yet)
        raise UnslothUnavailable(f"{type(e).__name__}: {str(e)[:300]}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.0),
        target_modules=cfg["lora"].get("target_modules", DEFAULT_TARGETS),
        use_gradient_checkpointing="unsloth",
        random_state=17,
    )
    train_rows, eval_rows = _load_rows(cfg, data_dir, tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        args=_sft_config(cfg, out_dir, max_steps, bool(eval_rows),
                         dataset_text_field="text", packing=False),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
    )
    trainer.train()

    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))
    # Merged 16-bit weights: what vLLM / SGLang serve (see serve.sh). Skipped
    # with --colab: ~28GB for a 14B model, and Colab users take the GGUF.
    if not cfg.get("_skip_merged"):
        model.save_pretrained_merged(str(out_dir / "merged"), tokenizer, save_method="merged_16bit")
    if gguf:
        if push:
            # Creates <repo>-GGUF on the Hub with the quantized file.
            model.push_to_hub_gguf(f"{push}-GGUF", tokenizer, quantization_method=gguf,
                                   token=os.environ.get("HF_TOKEN"))
        model.save_pretrained_gguf(str(out_dir / "gguf"), tokenizer, quantization_method=gguf)
        write_modelfile(out_dir / "gguf")
    _push(cfg, model, tokenizer, push)
    _write_meta(cfg, out_dir)


# ── Open-weight: plain transformers + PEFT (fallback) ─────────────────────────

def train_hf(cfg: dict, data_dir: Path, out_dir: Path, gguf: str | None, max_steps: int,
             push: str | None = None) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq)
    from trl import SFTTrainer

    t = cfg["train"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if t.get("load_in_4bit", True) else None
    dtype = torch.bfloat16 if _precision().get("bf16") else torch.float16
    if world_size() > 1:
        # Data-parallel (DeepSpeed ZeRO-2): one full 4-bit copy per GPU.
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    else:
        device_map = "auto"  # single process: spread a big model across visible GPUs
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, torch_dtype=dtype, device_map=device_map,
    )
    if quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.0),
        target_modules=cfg["lora"].get("target_modules", DEFAULT_TARGETS),
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    def tokenize(rows):
        out = []
        for r in rows:
            enc = tokenizer(r["text"], add_special_tokens=False, return_offsets_mapping=True)
            labels = mask_labels(enc["input_ids"], enc["offset_mapping"], assistant_spans(r["text"]))
            out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                        "labels": labels})
        return Dataset.from_list(out)

    train_rows, eval_rows = _load_rows(cfg, data_dir, tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenize(train_rows),           # pre-tokenized with our own labels
        eval_dataset=tokenize(eval_rows) if eval_rows else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        args=_sft_config(cfg, out_dir, max_steps, bool(eval_rows),
                         gradient_checkpointing=True, remove_unused_columns=False,
                         dataset_kwargs={"skip_prepare_dataset": True}),
    )
    trainer.train()
    if not trainer.is_world_process_zero():
        return  # only rank 0 saves and merges

    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))
    _push(cfg, model, tokenizer, push)
    # Merge in bf16 (not on top of the 4-bit weights) for serving.
    del model, trainer
    torch.cuda.empty_cache()
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype,
                                                device_map="auto")
    merged = PeftModel.from_pretrained(base, str(out_dir / "lora")).merge_and_unload()
    merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir / "merged"))
    if gguf:
        print("  --gguf needs Unsloth or llama.cpp's convert_hf_to_gguf.py on the merged/ dir.")
    _write_meta(cfg, out_dir)


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
    ap.add_argument("model", help="axon-1 | tandav-1 | veda-1")
    ap.add_argument("--backend", choices=["local", "openai"], default="local")
    ap.add_argument("--trainer", choices=["auto", "unsloth", "hf"], default="auto",
                    help="local trainer: auto tries Unsloth, then transformers+PEFT")
    ap.add_argument("--base", help="override the base model (required for --backend openai)")
    ap.add_argument("--data", type=Path, help="dir with train.jsonl/eval.jsonl")
    ap.add_argument("--out", type=Path, help="output dir (default finetune/out/<model>)")
    ap.add_argument("--gguf", help="also export GGUF for Ollama, e.g. q4_k_m")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test)")
    ap.add_argument("--colab", action="store_true",
                    help="free Colab/Kaggle T4 preset: 4-bit, shorter context")
    ap.add_argument("--push-to-hub", metavar="REPO",
                    help="upload the adapter (and GGUF) to the HF Hub, e.g. you/axon-1 (needs HF_TOKEN)")
    args = ap.parse_args()

    cfg = load_config(args.model)
    if args.colab:
        cfg["train"]["load_in_4bit"] = True
        cfg["train"].update({k: v for k, v in cfg.get("colab", {}).items() if k != "note"})
        cfg["_skip_merged"] = True
    data_dir = args.data or HERE / "data" / cfg["name"]
    if args.backend == "openai":
        train_openai(cfg, data_dir, args.base)
        return
    if args.base:
        cfg["base_model"] = args.base
    out_dir = args.out or HERE / "out" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{cfg['name']}: {cfg['base_model']} ({cfg.get('params', '?')})")

    if args.trainer == "hf":
        train_hf(cfg, data_dir, out_dir, args.gguf, args.max_steps, args.push_to_hub)
        return
    try:
        train_unsloth(cfg, data_dir, out_dir, args.gguf, args.max_steps, args.push_to_hub)
    except UnslothUnavailable as e:
        if args.trainer == "unsloth":
            raise SystemExit(f"Unsloth can't train {cfg['base_model']}: {e}")
        print(f"Unsloth unavailable for {cfg['base_model']} ({e}); using transformers + PEFT.")
        train_hf(cfg, data_dir, out_dir, args.gguf, args.max_steps, args.push_to_hub)


if __name__ == "__main__":
    main()
