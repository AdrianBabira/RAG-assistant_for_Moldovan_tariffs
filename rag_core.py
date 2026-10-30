"""Ядро RAG-системы: обогащение чанков, гибридный поиск, LLM, пайплайн."""
import json
import re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from llama_cpp import Llama


# ==========================================
# 1. Обогащение данных
# ==========================================
def create_enriched_chunks(df: pd.DataFrame) -> list[dict]:
    """Обогащённые чанки: явные 'ДА/НЕТ' вместо 0, 'безлимит' вместо 'unlimited'."""
    chunks = []
    for _, row in df.iterrows():
        intl = str(row["International minutes"])
        has_intl = "НЕТ" if intl in ["0", "0.0"] else f"ДА ({intl} мин)"

        internet = str(row["Internet (GB)"])
        is_unlimited = "ДА" if internet.lower() == "unlimited" else "НЕТ"
        internet_text = "безлимит" if is_unlimited == "ДА" else f"{internet} ГБ"

        text = (
            f"Оператор: {row['Operator']}. "
            f"Тариф: {row['Subscription']}. "
            f"Абонентская плата: {row['Price']} MDL. "
            f"Минуты и SMS по стране: {row['National minutes/SMS']}. "
            f"Международные минуты: {has_intl}. "
            f"Интернет: {internet_text} (Безлимитный интернет: {is_unlimited}). "
            f"Роуминг в ЕС: {row['Roaming EU (GB)']} ГБ."
        )
        chunks.append({
            "chunk_id": f"{row['Operator']}_{row['Subscription']}".replace(" ", "_"),
            "operator": row["Operator"],
            "subscription": row["Subscription"],
            "text": text,
        })
    return chunks


def create_summary_chunks(df: pd.DataFrame, base_chunks: list[dict]) -> list[dict]:
    """Добавляет сводки по операторам — для вопросов агрегации."""
    chunks = list(base_chunks)
    for op in df["Operator"].unique():
        op_df = df[df["Operator"] == op]
        cheapest = op_df.loc[op_df["Price"].idxmin()]
        text = (
            f"Оператор: {op}. "
            f"Всего тарифов у оператора: {len(op_df)}. "
            f"Список тарифов: {', '.join(op_df['Subscription'])}. "
            f"Самый дешевый тариф: {cheapest['Subscription']} ({cheapest['Price']} MDL)."
        )
        chunks.append({
            "chunk_id": f"Summary_{op}",
            "operator": op,
            "subscription": "Summary",
            "text": text,
        })
    return chunks


def create_markdown_table(df: pd.DataFrame) -> str:
    """Markdown-таблица всех тарифов — для prompt_style='table'."""
    md = "| Оператор | Тариф | Цена (MDL) | Минуты/SMS | Межд. минуты | Интернет | Роуминг ЕС |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for _, row in df.iterrows():
        intl = "НЕТ" if str(row["International minutes"]) in ["0", "0.0"] else row["International minutes"]
        net = row["Internet (GB)"]
        net_text = "безлимит" if str(net).lower() == "unlimited" else f"{net} ГБ"
        md += (
            f"| {row['Operator']} | {row['Subscription']} | {row['Price']} | "
            f"{row['National minutes/SMS']} | {intl} | {net_text} | {row['Roaming EU (GB)']} |\n"
        )
    return md


# ==========================================
# 2. Гибридный индекс (Dense + BM25)
# ==========================================
class HybridIndex:
    def __init__(self, embedding_model="BAAI/bge-m3"):
        self.embed_model = SentenceTransformer(embedding_model)
        self.chunks, self.index, self.bm25 = [], None, None

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embs = self.embed_model.encode(
            texts, normalize_embeddings=True, batch_size=32
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def search(self, query: str, k: int = 8) -> list[dict]:
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        _, dense_ids = self.index.search(q_emb, k * 2)

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][: k * 2]

        rrf = {}
        for rank, idx in enumerate(dense_ids[0]):
            if idx != -1:
                rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)
        for rank, idx in enumerate(bm25_top):
            rrf[idx] = rrf.get(idx, 0) + 1 / (60 + rank)

        sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]
        return [dict(self.chunks[i], score=float(rrf[i])) for i in sorted_ids]


# ==========================================
# 3. LLM-клиент (с генерацией и стримингом)
# ==========================================
class LocalLLMClient:
    def __init__(self, model_path: str, n_ctx: int = 4096):
        self.llm = Llama(
            model_path=model_path, n_ctx=n_ctx,
            n_threads=8, n_gpu_layers=-1, verbose=False,
        )

    def _messages(self, prompt: str, system: str):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str, max_tokens: int = 100,
                 system: str = "Ты — точный ассистент.") -> str:
        resp = self.llm.create_chat_completion(
            messages=self._messages(prompt, system),
            temperature=0.0, max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def stream_generate(self, prompt: str, max_tokens: int = 100,
                        system: str = "Ты — точный ассистент."):
        """Генератор токенов — для st.write_stream()."""
        stream = self.llm.create_chat_completion(
            messages=self._messages(prompt, system),
            temperature=0.0, max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]


# ==========================================
# 4. RAG-пайплайн (все стили промптов)
# ==========================================
class RAGPipeline:
    FEWSHOT_PREFIX = """Ты — аналитик тарифов. Отвечай только по контексту.

Пример 1 (Агрегация):
Контекст: [1] Оператор: A. Всего тарифов: 3.
Вопрос: Сколько тарифов у A?
Ответ: 3

Пример 2 (Сравнение):
Контекст: [1] A. Цена: 120 MDL. [2] B. Цена: 90 MDL.
Вопрос: Что дешевле: A или B?
Ответ: B (90 MDL) дешевле A (120 MDL).

Пример 3 (Да/Нет):
Контекст: [1] C. Межд. минуты: НЕТ.
Вопрос: Есть ли межд. минуты в C?
Ответ: Нет

"""

    def __init__(self, retriever: HybridIndex, llm: LocalLLMClient,
                 top_n: int = 8, prompt_style: str = "fewshot"):
        self.retriever = retriever
        self.llm = llm
        self.top_n = top_n
        self.prompt_style = prompt_style

    def _effective_k(self, question: str) -> int:
        if self.prompt_style == "table":
            return 1
        if self.prompt_style == "dynamic":
            q = question.lower()
            if any(w in q for w in ["сколько", "сравни", "дороже", "дешевле",
                                     "больше", "меньше", "перечисли"]):
                return 15
        return self.top_n

    def _build_prompt(self, question: str, chunks: list[dict],
                      table_text: str | None = None) -> str:
        if self.prompt_style == "table":
            return f"""Ниже таблица тарифов:

{table_text}

Вопрос: {question}
Найди в таблице нужные строки и ответь кратко. Ответ:"""

        context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))

        if self.prompt_style == "simple":
            return f"""Контекст:
{context}

Вопрос: {question}
Правила: "0" = "Нет". Если нет ответа — "Нет информации". Кратко.
Ответ:"""

        return self.FEWSHOT_PREFIX + f"""Теперь твой вопрос:
Контекст:
{context}

Вопрос: {question}
Ответ:"""

    def answer(self, question: str, table_text: str | None = None):
        k = self._effective_k(question)
        chunks = self.retriever.search(question, k=k)
        prompt = self._build_prompt(question, chunks, table_text)
        return self.llm.generate(prompt, max_tokens=100), chunks

    def answer_stream(self, question: str, table_text: str | None = None):
        k = self._effective_k(question)
        chunks = self.retriever.search(question, k=k)
        prompt = self._build_prompt(question, chunks, table_text)
        return self.llm.stream_generate(prompt, max_tokens=100), chunks