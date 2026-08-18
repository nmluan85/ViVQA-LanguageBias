import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from google import genai
from google.genai import types


# =========================================================
# CONFIG
# =========================================================

load_dotenv()


client = genai.Client(
    api_key="GEMINI_KEY"
)

MODEL_NAME = "gemini-3.1-flash-lite"

INPUT_FILES = [
    "data/ViVQA-csv/train.csv",
    "data/ViVQA-csv/val.csv",
    "data/ViVQA-csv/test.csv",
]


# =========================================================
# GEMINI
# =========================================================

SYSTEM_PROMPT = """
Xác định đối tượng chính được hỏi đến trong câu VQA.

Chỉ trả về danh từ hoặc cụm danh từ chỉ đối tượng đó.
Không giải thích.
Không thêm dấu câu.
Bỏ các từ chỉ loại như "chiếc", "cái", "con", "người" nếu chúng
không phải là một phần cần thiết của tên đối tượng.

Ví dụ:
"Màu của chiếc xe là gì?" -> xe
"Có bao nhiêu con chó trong ảnh?" -> chó
"Chiếc xe đạp đang nằm ở đâu?" -> xe đạp
"Màu của hàng rào là gì?" -> hàng rào
"""


def extract_core_noun(question: str, max_retries: int = 5) -> str:
    """
    Extract critical noun from one VQA question.
    """

    if pd.isna(question) or str(question).strip() == "":
        return ""

    question = str(question).strip()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=10,
                ),
            )

            result = response.text

            if result is None:
                raise ValueError("Gemini returned None")

            return result.strip()

        except Exception as e:
            print(
                f"\nError: {e}\n"
                f"Question: {question}\n"
                f"Retry {attempt + 1}/{max_retries}"
            )

            if attempt == max_retries - 1:
                return ""

            # exponential backoff
            time.sleep(2 ** attempt)

    return ""


# =========================================================
# PROCESS CSV
# =========================================================

def process_csv(input_file: str, cache: dict):
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"\nFile not found: {input_file}")
        return

    print("\n" + "=" * 70)
    print(f"Processing: {input_file}")
    print("=" * 70)

    df = pd.read_csv(input_path)

    if "question" not in df.columns:
        raise ValueError(
            f"{input_file} does not contain a 'question' column."
        )

    critical_nouns = []

    for question in tqdm(
        df["question"],
        desc=f"Extracting {input_path.stem}",
        total=len(df),
    ):
        question_key = str(question).strip()

        # Nếu câu hỏi đã gặp trước đó thì không gọi API lại
        if question_key in cache:
            noun = cache[question_key]

        else:
            noun = extract_core_noun(question)
            cache[question_key] = noun

        critical_nouns.append(noun)

    # Add column at the right-most position
    df["critical_noun"] = critical_nouns

    # Create NEW output file
    output_path = input_path.with_name(
        f"{input_path.stem}_with_critical_noun.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {output_path}")
    print(f"Rows : {len(df)}")

    print("\nPreview:")
    print(
        df[
            ["question", "critical_noun"]
        ].head(10).to_string(index=False)
    )


# =========================================================
# MAIN
# =========================================================

def main():
    # Cache dùng chung cho train / val / test.
    # Nếu có câu hỏi trùng nhau thì chỉ gọi Gemini một lần.
    cache = {}

    for input_file in INPUT_FILES:
        process_csv(
            input_file=input_file,
            cache=cache,
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Unique questions sent to Gemini: {len(cache)}")


if __name__ == "__main__":
    main()