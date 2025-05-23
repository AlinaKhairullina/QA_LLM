from argparse import ArgumentParser
from huggingface_hub import InferenceClient


SYSTEM_PROMPT = 'You are a professional Python software engineer performing code reviews. I give you a question about the function and the function itself in Python. The code is given in a special form without syntax errors: colons, brackets, commas, and parentheses are omitted. Your task is to understand the logic of the code despite the missing syntax. Respond to each code snippet or question concisely in no more than four-six words, one sentence or word.  Focus on correctness and efficiency.'


def parse_args() -> ArgumentParser:

    parser = ArgumentParser()
    parser.add_argument(
        'llm_endpoint_url',
        help='LLM endpoint url'
    )

    return parser.parse_args()


def generate_qc_pairs(f1, f2):
    qc_pairs = []
    with open(f1, "r") as file1, open(f2, "r") as file2:
        for q, c in zip(file1, file2):
            qc_pairs.append((q.strip(), c.strip()))
    return qc_pairs


if __name__ == '__main__':
    args = parse_args()
    llm_client = InferenceClient(base_url=args.llm_endpoint_url)
    qc_pairs = generate_qc_pairs(
        "/home/alina/hermes/db/test/test.question", "/home/alina/hermes/db/test/test.code")[6000:]
    docs = [f"Question: {q} Code: {c}" for q, c in qc_pairs]
    qwen_answers = []
    for i in range(len(docs)):
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': docs[i]}
        ]
        answer = llm_client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.0
        )
        print(answer.choices[0].message.content)
        print("---------------")
        qwen_answers.append(answer.choices[0].message.content)
    with open("qwen.txt", "a") as file:
        file.write("\n".join(qwen_answers) + "\n")
