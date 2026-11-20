import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import agents
from metrics import EMBEDDING_MODEL, evaluate, load_encoder
from ollama_client import generate, inspect_server
from scenarios import SCENARIOS

FIELDS = [
    "scenario_id", "scenario_text", "method", "repeat", "coverage", "coverage_ratio",
    "duplication_score", "time_sec", "tokens", "prompt_tokens", "output_tokens",
    "token_source", "duplication_backend", "pair_count", "status", "error",
    "invalid_calls", "call_count", "metrics_time_sec", "order_index", "seed",
    "model", "min_chars", "config_hash", "raw_file",
    "agent_1_section", "agent_1_text", "agent_1_raw",
    "agent_2_section", "agent_2_text", "agent_2_raw",
    "agent_3_section", "agent_3_text", "agent_3_raw",
    "price_texts", "comfort_texts", "reliability_texts", "plans_json", "pairs_json",
]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_rows(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_row(scenario, result, measured, args, run_seed, order_index, config_hash, raw_file):
    row = {name: "" for name in FIELDS}
    row.update({
        "scenario_id": scenario["id"], "scenario_text": scenario["text"],
        "method": result["method"], "repeat": result["repeat"],
        "time_sec": round(result["time_sec"], 6), "tokens": result["tokens"],
        "prompt_tokens": result["prompt_tokens"], "output_tokens": result["output_tokens"],
        "token_source": result["token_source"], "duplication_backend": args.duplication,
        "status": result["status"], "error": "; ".join(result["errors"]),
        "invalid_calls": result["invalid_calls"], "call_count": result["call_count"],
        "order_index": order_index, "seed": run_seed, "model": args.model,
        "min_chars": args.min_chars, "config_hash": config_hash, "raw_file": raw_file,
        "plans_json": json.dumps(result["plans"], ensure_ascii=False),
    })
    if measured:
        row.update({name: measured[name] for name in ("coverage", "coverage_ratio", "pair_count")})
        row["duplication_score"] = measured["duplication_score"]
        row["pairs_json"] = json.dumps(measured["duplication_pairs"], ensure_ascii=False)
        row["metrics_time_sec"] = round(result.get("metrics_time_sec", 0), 6)
    for index, answer in enumerate(result["finals"], 1):
        row[f"agent_{index}_section"] = answer["section"]
        row[f"agent_{index}_text"] = answer.get("text", "")
        row[f"agent_{index}_raw"] = answer["raw"]
    for section in agents.SECTIONS:
        row[section + "_texts"] = json.dumps(result["sections"][section], ensure_ascii=False)
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Намерения и факты: семейные автомобили, три агента, локальная Ollama.")
    parser.add_argument("--model", default="mistral:7b")
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--limit", type=int, default=25, help="Первые N сценариев, 1..25.")
    parser.add_argument("--scenario", nargs="+", help="Конкретные ID, например S01 S21.")
    parser.add_argument("--methods", nargs="+", choices=["intentions", "facts"], default=["intentions", "facts"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-predict", type=int, default=900)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--duplication", choices=["semantic", "lexical"], default="semantic")
    parser.add_argument("--embedding-revision", default=None, help="Необязательный commit модели Hugging Face.")
    parser.add_argument("--resume", action="store_true", help="Продолжить тот же опыт, повторив только строки с ошибками.")
    parser.add_argument("--check-only", action="store_true", help="Проверить окружение и Ollama без генерации и эмбеддингов.")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 25 or min(args.repeats, args.min_chars, args.num_predict, args.num_ctx) < 1:
        parser.error("Нужны limit от 1 до 25 и положительные repeats/min-chars/num-predict/num-ctx.")
    if args.timeout <= 0 or not 0 <= args.temperature <= 2:
        parser.error("timeout должен быть >0, temperature — от 0 до 2.")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Способы не должны повторяться в --methods.")
    if args.scenario and set(args.scenario) - {s["id"] for s in SCENARIOS}:
        parser.error("Неизвестный scenario_id.")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    try:
        server = inspect_server(args.model, args.host)
        if args.check_only:
            print(f"Ок: Python {sys.executable}; Ollama {server['version']}; модель {args.model}.")
            return 0
        scenarios = ([s for s in SCENARIOS if s["id"] in args.scenario]
                     if args.scenario else SCENARIOS[:args.limit])
        source_files = ["agents.py", "metrics.py", "ollama_client.py", "run_experiment.py", "scenarios.py"]
        config = {
            "model": args.model, "host": args.host, "scenarios": scenarios,
            "methods": args.methods, "repeats": args.repeats, "seed": args.seed,
            "temperature": args.temperature, "num_predict": args.num_predict,
            "num_ctx": args.num_ctx, "timeout": args.timeout, "min_chars": args.min_chars,
            "duplication": args.duplication, "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": args.embedding_revision,
            "ollama_version": server["version"],
            "model_tags": [item for item in server["models"]
                           if item.get("name") in {args.model, args.model + ":latest"}],
            "ollama_num_parallel": os.getenv("OLLAMA_NUM_PARALLEL", "unknown"),
            "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in source_files},
        }
        config_hash = fingerprint(config)
        meta_path = output.with_suffix(".meta.json")
        rows = []
        if args.resume and output.exists():
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with output.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
        output.parent.mkdir(parents=True, exist_ok=True)
        encoder = load_encoder(revision=args.embedding_revision) if args.duplication == "semantic" else None
        options = {"host": args.host, "timeout": args.timeout, "temperature": args.temperature,
                   "num_predict": args.num_predict, "num_ctx": args.num_ctx}
        completed = {(row["scenario_id"], row["method"], int(row["repeat"])) for row in rows
                     if row["status"] in {"ok", "invalid_output"}}
        warmup = generate("Ответь одним словом по-русски: готов.", args.model,
                          **{**options, "num_predict": 8}, seed=args.seed)
        now = datetime.now(timezone.utc).isoformat()
        meta = old_meta if args.resume and output.exists() else {
            "created_at_utc": now, "config_hash": config_hash, "config": config,
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "server": server, "warmups": [],
            "environment": {key: os.getenv(key) for key in (
                "OLLAMA_MODELS", "OLLAMA_NUM_PARALLEL", "HF_HOME", "HF_HUB_CACHE",
                "TRANSFORMERS_CACHE", "PIP_CACHE_DIR", "TEMP", "TMP")},
        }
        meta["warmups"].append({"at_utc": now, "response": warmup})
        if encoder is not None:
            meta["embedding"] = {
                "name": EMBEDDING_MODEL, "requested_revision": args.embedding_revision,
                "resolved_revision": getattr(encoder[0].auto_model.config, "_commit_hash", None),
                "max_seq_length": encoder.max_seq_length, "device": "cpu",
            }
        save_json(meta_path, meta)
        total = len(scenarios) * len(args.methods) * args.repeats
        print(f"План: {total} строк. Уже завершено: {len(completed)}. Метрика: {args.duplication}.", flush=True)
        runners = {"intentions": agents.run_intentions, "facts": agents.run_facts}
        for repeat in range(1, args.repeats + 1):
            for scenario in scenarios:
                scenario_index = int(scenario["id"][1:])
                run_seed = args.seed + repeat * 100_000 + scenario_index * 1_000
                order = list(args.methods)
                if (scenario_index + repeat + args.seed) % 2:
                    order.reverse()
                for order_index, method in enumerate(order, 1):
                    key = (scenario["id"], method, repeat)
                    if key in completed:
                        continue
                    print(f"{scenario['id']} / {method} / повтор {repeat} ...", flush=True)
                    result = runners[method](scenario["text"], args.model, options=options, seed=run_seed)
                    result["repeat"] = repeat
                    measured = None
                    raw_path = output.parent / (output.stem + "_raw") / f"{scenario['id']}_{method}_{repeat}.json"
                    save_json(raw_path, {"scenario": scenario, "config_hash": config_hash, "result": result})
                    if result["status"] != "error":
                        try:
                            started = perf_counter()
                            measured = evaluate(result, args.min_chars, args.duplication, encoder)
                            result["metrics_time_sec"] = perf_counter() - started
                        except Exception as error:
                            result["status"] = "error_metrics"
                            result["errors"].append(str(error))
                    save_json(raw_path, {"scenario": scenario, "config_hash": config_hash,
                                         "result": result, "metrics": measured})
                    row = build_row(scenario, result, measured, args, run_seed, order_index,
                                    config_hash, str(raw_path.relative_to(output.parent)))
                    rows = [old for old in rows if (old["scenario_id"], old["method"], int(old["repeat"])) != key]
                    rows.append(row)
                    save_rows(output, rows)
                    print(f"  статус={row['status']}; полнота={row['coverage']}; сходство={row['duplication_score']}; {row['time_sec']:.1f} с", flush=True)
        print(f"Готово: {output}. Откройте analysis.ipynb, чтобы построить графики.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
