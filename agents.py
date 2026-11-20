from concurrent.futures import ThreadPoolExecutor
import json
from time import perf_counter

from ollama_client import generate

SECTIONS = {
    "price": "Цена и бюджет",
    "comfort": "Комфорт и практичность",
    "reliability": "Надёжность и обслуживание",
}

COMMON_PROMPT = """Ты один из трёх равноправных ИИ-агентов. Вместе вы готовите
рекомендацию о семейном автомобиле. Это учебная задача, доступа к рынку и интернету нет.
Указывай модель, примерный возраст/поколение, объясняй выбор применительно к семье.
Цены, расход и расходы на сервис называй приблизительными, без вымышленных источников.
Не утверждай, что детские кресла или коляска точно поместятся без проверки размеров.
Каждый агент пишет ровно ОДИН раздел. Выбирай раздел самостоятельно, роли заранее не назначены.
Обязательные разделы и их коды:
price — Цена и бюджет: цена в рублях, общий бюджет, страховка, регистрация, опции.
comfort — Комфорт и практичность: места, багажник, удобство семьи, расход топлива.
reliability — Надёжность и обслуживание: надёжность, обслуживание, наличие запчастей в регионе.
Не пиши другие разделы, общую вводную или заключение. Пиши по-русски.
Твой условный номер: {agent_id}. Номер не закрепляет за тобой раздел.
СЦЕНАРИЙ:
{scenario}
"""

PLAN_PROMPT = """
РАУНД 1. Независимо выбери один раздел, который собираешься написать.
Ответь только JSON: {"section": "код", "plan": "Я планирую сделать раздел ..."}.
Значение section: price, comfort или reliability. План — одно короткое предложение.
Не пиши финальный текст, цены, характеристики и рекомендации.
"""

INTENTION_FINAL_PROMPT = """
РАУНД 2. Ниже все планы первого раунда, включая твой:
{plans}
С учётом планов выбери один раздел и напиши его окончательный текст.
Можно изменить собственный план, если это помогает покрыть задачу.
Планы — обещания, готовых текстов других агентов у тебя нет.
Ответь только JSON: {{"section": "код", "text": "финальный текст раздела"}}.
Значение section: price, comfort или reliability. Объём text: примерно 450–800 символов.
Дай содержательный совет по выбранной теме, без описания того, что собираешься сделать.
"""

FACT_FINAL_PROMPT = """
ПОСЛЕДОВАТЕЛЬНЫЙ ШАГ. Уже полученные окончательные ответы других агентов:
{facts}
Прочитай реальные тексты и выбери один ещё не раскрытый раздел.
Если ответ пустой или не раскрывает тему, не считай его выполненной работой.
Ответь только JSON: {{"section": "код", "text": "финальный текст раздела"}}.
Значение section: price, comfort или reliability. Объём text: примерно 450–800 символов.
Дай содержательный совет по выбранной теме, без описания того, что собираешься сделать.
"""


def output_schema(stage):
    field = "plan" if stage == "plan" else "text"
    return {"type": "object", "properties": {
        "section": {"type": "string", "enum": list(SECTIONS)}, field: {"type": "string"}},
        "required": ["section", field], "additionalProperties": False}


def parse_answer(raw, stage):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    field = "plan" if stage == "plan" else "text"
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("section") not in SECTIONS:
            raise ValueError("неизвестный или отсутствующий section")
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"поле {field} отсутствует или пустое")
        return {"section": data["section"], field: data[field].strip(), "parse_error": ""}
    except (ValueError, TypeError) as error:
        return {"section": "", field: "", "parse_error": str(error)}


def call_agent(scenario, agent_id, stage, extra_prompt, model, options, seed, client):
    prompt = COMMON_PROMPT.format(agent_id=agent_id, scenario=scenario) + extra_prompt
    started = perf_counter()
    try:
        reply = client(prompt, model, **options, seed=seed, schema=output_schema(stage))
        parsed = parse_answer(reply["text"], stage)
        return {"agent_id": agent_id, "stage": stage, "seed": seed, "prompt": prompt,
                "raw": reply["text"], **parsed, "reply": reply, "error": "",
                "time_sec": perf_counter() - started}
    except (RuntimeError, ValueError, OSError) as error:
        return {"agent_id": agent_id, "stage": stage, "seed": seed, "prompt": prompt,
                "raw": "", "section": "", "plan": "", "text": "",
                "parse_error": "", "reply": {}, "error": str(error),
                "time_sec": perf_counter() - started}


def make_result(method, plans, finals, started):
    sections = {section: [] for section in SECTIONS}
    for answer in finals:
        if answer["section"] in sections and answer.get("text"):
            sections[answer["section"]].append({"agent_id": answer["agent_id"], "text": answer["text"]})
    calls = plans + finals
    replies = [call["reply"] for call in calls if call["reply"]]
    errors = [call["error"] for call in calls if call["error"]]
    invalid = [call for call in calls if call["parse_error"] or call["reply"].get("truncated")]
    return {
        "method": method, "sections": sections, "plans": plans, "finals": finals,
        "calls": calls, "time_sec": perf_counter() - started,
        "prompt_tokens": sum(reply["prompt_tokens"] for reply in replies),
        "output_tokens": sum(reply["output_tokens"] for reply in replies),
        "tokens": sum(reply["tokens"] for reply in replies),
        "token_source": "ollama" if len(replies) == len(calls) and all(
            reply["token_source"] == "ollama" for reply in replies) else "estimated_or_incomplete",
        "status": "error" if errors else ("invalid_output" if invalid else "ok"),
        "errors": errors, "invalid_calls": len(invalid), "call_count": len(calls),
    }


def run_intentions(scenario, model="mistral:7b", *, options=None, seed=42, client=generate):
    started = perf_counter()
    options = options or {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(call_agent, scenario, i, "plan", PLAN_PROMPT, model,
                            options, seed + i, client) for i in range(1, 4)]
        plans = [job.result() for job in jobs]
        shared_plans = json.dumps([
            {"agent_id": p["agent_id"], "section": p["section"],
             "plan": p.get("plan", ""), "raw": p["raw"] if p["parse_error"] else ""}
            for p in plans], ensure_ascii=False)
        prompt = INTENTION_FINAL_PROMPT.format(plans=shared_plans)
        jobs = [pool.submit(call_agent, scenario, i, "final", prompt, model,
                            options, seed + 100 + i, client) for i in range(1, 4)]
        finals = [job.result() for job in jobs]
    return make_result("intentions", plans, finals, started)


def run_facts(scenario, model="mistral:7b", *, options=None, seed=42, client=generate):
    started = perf_counter()
    finals = []
    for i in range(1, 4):
        actual_texts = [{"agent_id": f["agent_id"], "section": f["section"],
                         "text": f.get("text", ""),
                         "raw": f["raw"] if f["parse_error"] else ""} for f in finals]
        prompt = FACT_FINAL_PROMPT.format(facts=json.dumps(actual_texts, ensure_ascii=False))
        finals.append(call_agent(scenario, i, "final", prompt, model, options or {},
                                 seed + 100 + i, client))
    return make_result("facts", [], finals, started)
