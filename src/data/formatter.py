"""Format standardized QA dicts into instruction-tuning strings via chat template.

add_generation_prompt=False  → include assistant answer (training)
add_generation_prompt=True   → append generation prompt marker only (eval)

System prompt is intentionally empty — using the base chat template's default
framing avoids distribution mismatch between training and eval.
"""

_ANSWER_LETTERS = "ABCD"


def _build_messages(item: dict, add_generation_prompt: bool) -> list[dict]:
    task = item["task"]
    if task == "pubmedqa":
        user_content = (
            f"Context: {item['context']}\n\n"
            f"Question: {item['question']}\n\n"
            "Answer with exactly one word: yes, no, or maybe."
        )
        messages = [{"role": "user", "content": user_content}]
        if not add_generation_prompt:
            messages.append({"role": "assistant", "content": item["answer"]})

    elif task == "medmcqa":
        answer_letter = _ANSWER_LETTERS[item["answer_idx"]]
        user_content = (
            f"Question: {item['question']}\n"
            f"A) {item['opa']}\n"
            f"B) {item['opb']}\n"
            f"C) {item['opc']}\n"
            f"D) {item['opd']}\n\n"
            "Answer with exactly one letter: A, B, C, or D."
        )
        messages = [{"role": "user", "content": user_content}]
        if not add_generation_prompt:
            messages.append({"role": "assistant", "content": answer_letter})

    else:
        raise ValueError(f"Unknown task type: '{task}'")

    return messages


def format_example(
    item: dict,
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    messages = _build_messages(item, add_generation_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def format_dataset(
    items: list[dict],
    tokenizer,
    add_generation_prompt: bool = False,
) -> list[str]:
    return [format_example(item, tokenizer, add_generation_prompt) for item in items]
