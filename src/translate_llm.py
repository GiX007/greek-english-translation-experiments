# Translates a sentence or dataset index using an OpenAI LLM (Responses API)
# Requires OPENAI_API_KEY set in the environment.
import sys
import time

from datasets import load_dataset
from openai import OpenAI

from config import get_config

from dotenv import load_dotenv
load_dotenv("OPENAI_API_KEY.env")

def _openai_translate_one(client: OpenAI, model_name: str, source_text: str, max_output_tokens: int = 256) -> str:
    system_msg = (
        "You are a translation engine. Translate Greek to English.\n"
        "Return ONLY the English translation text. No explanations."
    )

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": source_text},
        ],
        temperature=0,
        max_output_tokens=max_output_tokens,
    )
    return (resp.output_text or "").strip()

def openai_translate_with_retries(client: OpenAI, model_name: str, source_text: str, max_output_tokens: int = 256, max_retries: int = 6, base_sleep: float = 1.0) -> str:
    for attempt in range(max_retries):
        try:
            return _openai_translate_one(client=client, model_name=model_name, source_text=source_text, max_output_tokens=max_output_tokens)
        except Exception:
            if attempt == max_retries - 1:
                raise time.sleep(base_sleep * (2 ** attempt))


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python src\\translate_llm.py "Greek sentence"')
        print("  python src\\translate_llm.py <dataset_index>")
        sys.exit(1)

    config = get_config()
    client = OpenAI()

    model_name = config["openai_model"]
    max_out = config.get("openai_max_output_tokens", 256)

    user_input = sys.argv[1]

    # Case 1: dataset index
    if user_input.isdigit():
        idx = int(user_input)

        dataset = load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

        src_text = dataset[idx]["translation"][config["lang_src"]]
        tgt_text = dataset[idx]["translation"][config["lang_tgt"]]

        pred = openai_translate_with_retries(client, model_name, src_text, max_output_tokens=max_out)

        print("-" * 40)
        print(f"INDEX: {idx}")
        print(f"SOURCE: {src_text}")
        print(f"TARGET: {tgt_text}")
        print(f"PREDICTED: {pred}")
        print("-" * 40)

    # Case 2: raw text
    else:
        pred = openai_translate_with_retries(client, model_name, user_input, max_output_tokens=max_out)
        print(pred)


if __name__ == "__main__":
    main()
