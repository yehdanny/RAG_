import ollama
import os
import json
from numpy import linalg, dot

def parse_paragraph(filename):
    with open(filename,encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def calc_embedings(paragraphs):
    return [
        ollama.embeddings(model="mistral", prompt=data)["embedding"]
        for data in paragraphs
    ]

def calc_similar_vectors(v, vectors):
    v_norm = linalg.norm(v)
    scores = [dot(v, item) / (v_norm * linalg.norm(item)) for item in vectors]
    return sorted(enumerate(scores), reverse=True, key=lambda x: x[1])

def cache_embeddings(filename, paragraphs):
    embedding_file = f"cache/{filename}.json"

    if os.path.isfile(embedding_file):
        with open(embedding_file) as f:
            return json.load(f)

    os.makedirs("cache", exist_ok=True)

    embeddings = calc_embedings(paragraphs)

    with open(embedding_file, "w") as f:
        json.dump(embeddings, f)

    return embeddings

if __name__ == "__main__":
    doc = "data/data.txt"
    paragraphs = parse_paragraph(doc)
    embeddings = cache_embeddings(doc, paragraphs)

    prompt = input("請問你想問什麼問題？\n>>> ")

    while prompt.lower() != "bye":
        prompt_embedding = ollama.embeddings(model="mistral", prompt=prompt)[
            "embedding"
        ]
        similar_vectors = calc_similar_vectors(prompt_embedding, embeddings)[:3]

        system_prompt = (
            "現在開始使用我提供的情境來回答，只能使用繁體中文，不要有簡體中文字。如果你不確定答案，就說不知道。情境如下："
            + "\n".join(paragraphs[vector[0]] for vector in similar_vectors)
        )

        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        print(response["message"]["content"])
        prompt = input(">>> ")