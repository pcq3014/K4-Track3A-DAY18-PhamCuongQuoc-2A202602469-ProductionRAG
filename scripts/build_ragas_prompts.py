"""Dịch prompt sinh câu hỏi của RAGAS answer_relevancy sang tiếng Việt và lưu cache vào repo.

Chạy 1 lần: python scripts/build_ragas_prompts.py
→ ragas_prompts/vietnamese/<prompt_name>.json. M4 load file này mỗi lần evaluate (không gọi LLM dịch lại),
nên baseline và production luôn được chấm bằng cùng một prompt.
"""

import json, os, sys, tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import OPENAI_API_KEY  # noqa: F401  (load .env)
from src.m4_eval import JUDGE_MODEL, PROMPT_CACHE_DIR, PROMPT_LANGUAGE

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy


def _strip_fence(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def main(max_attempts: int = 5):
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    for attempt in range(1, max_attempts + 1):
        try:
            with tempfile.TemporaryDirectory() as tmp:  # tmp cache → luôn dịch mới, không đọc cache cũ
                prompt = AnswerRelevancy().question_generation.adapt(PROMPT_LANGUAGE, llm, tmp)
            # Ví dụ dịch xong hay bị bọc ```json ... ``` → judge bắt chước → parse lỗi. Bỏ code fence.
            for example in prompt.examples:
                for key, value in example.items():
                    if isinstance(value, str):
                        example[key] = _strip_fence(value)
                json.loads(example[prompt.output_key])  # output mỗi ví dụ phải là JSON hợp lệ
            prompt.save(PROMPT_CACHE_DIR)
            path = os.path.join(PROMPT_CACHE_DIR, PROMPT_LANGUAGE, f"{prompt.name}.json")
            print(f"✓ Saved {path} (attempt {attempt})")
            print(json.dumps(prompt.examples[0], ensure_ascii=False, indent=2))
            return
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} failed: {e}")
    sys.exit("Không dịch được prompt sau nhiều lần thử")


if __name__ == "__main__":
    main()
