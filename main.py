import argparse
import sys
import torch
from transformers import pipeline

# Установите устройство (GPU или CPU)
DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_MODEL = "sberbank-ai/rugpt3medium_based_on_gpt2"


# Функция для очистки текста
def normalize(s: str) -> str:
    return " ".join(str(s).split())


# Функция для генерации текста (вступление, обзор, заключение)
def generate(gen, prompt: str, *, min_new_tokens: int, max_new_tokens: int,
             do_sample: bool, temperature: float, top_k: int, top_p: float,
             repetition_penalty: float, truncation: bool,
             no_repeat_ngram_size: int | None):
    kwargs = dict(
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=50256,
        truncation=truncation,
    )
    if no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    out = gen(prompt, **kwargs)
    return normalize(out[0]["generated_text"])


# Функция для создания генератора
def build_generator(model_name: str = DEFAULT_MODEL):
    try:
        return pipeline("text-generation", model=model_name, device=DEVICE)
    except Exception as e:
        print(f"[Ошибка] Не удалось загрузить модель {model_name}: {e}", file=sys.stderr)
        sys.exit(1)


# Функция для получения универсальных промптов на основе темы
def get_prompts_for_topic(topic):
    """Функция для формирования универсальных запросов на основе темы"""
    intro_prompt = f"Вступление: Краткое введение в тему '{topic}', описание её контекста и значимости."
    review_prompt = f"Обзор: Детальный обзор по теме '{topic}', включая ключевые аспекты, особенности и факты."
    conclusion_prompt = f"Заключение: Итоги обсуждения '{topic}', выводы и рекомендации."

    return intro_prompt, review_prompt, conclusion_prompt


# Функция для парсинга аргументов командной строки
def parse_args():
    p = argparse.ArgumentParser(description="Генерация сценария для видео на любую тему")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--min_new_tokens", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--do_sample", type=lambda s: str(s).lower() in {"1", "true", "t", "yes", "y"}, default=True)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.92)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--truncation", type=lambda s: str(s).lower() in {"1", "true", "t", "yes", "y"}, default=True)
    p.add_argument("--metric_n", type=int, default=3)
    return p.parse_args()


# Главная функция
def main():
    args = parse_args()

    # Запрос темы у пользователя
    prompt = input("Введите тему видео (например, 'Обзор смартфона'): ")

    # Получаем универсальные промпты для выбранной темы
    intro_prompt, review_prompt, conclusion_prompt = get_prompts_for_topic(prompt)

    # Инициализация генератора
    gen = build_generator(args.model)

    # Генерация вступления, обзора и заключения
    scenario = ""

    # Вступление
    section_text = generate(
        gen, intro_prompt,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        truncation=args.truncation,
        no_repeat_ngram_size=2
    )
    scenario += f"## Вступление\n{section_text}\n\n"

    # Обзор
    section_text = generate(
        gen, review_prompt,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        truncation=args.truncation,
        no_repeat_ngram_size=2
    )
    scenario += f"## Обзор\n{section_text}\n\n"

    # Заключение
    section_text = generate(
        gen, conclusion_prompt,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        truncation=args.truncation,
        no_repeat_ngram_size=2
    )
    scenario += f"## Заключение\n{section_text}\n\n"

    # Сохранение в формате markdown
    with open("generated_scenario.md", "w", encoding="utf-8") as f:
        f.write(scenario)

    print("Сценарий сохранён в файл generated_scenario.md.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Ошибка] {e}", file=sys.stderr)
        sys.exit(1)
