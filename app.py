"""Streamlit-демо RAG-ассистента по тарифам мобильных операторов Молдовы."""
from pathlib import Path
import pandas as pd
import streamlit as st

from rag_core import (
    HybridIndex, LocalLLMClient, RAGPipeline,
    create_enriched_chunks, create_summary_chunks, create_markdown_table,
)

# ==========================================
# Конфигурация
# ==========================================
st.set_page_config(
    page_title="RAG Тарифы Молдовы",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
DATA_CSV = "subscriptions.csv"
RESULTS_CSV = "final_results.csv"

EXAMPLE_QUESTIONS = [
    "Какова абонентская плата тарифа One 170 от Orange?",
    "Сколько тарифов у оператора Moldcell?",
    "Какой тариф дешевле: Start 5G или Junior 90?",
    "Включены ли международные минуты в тариф Lumo 170?",
    "У какого тарифа больше всего гигабайт в роуминге ЕС?",
    "Сколько тарифов стоят меньше 120 MDL?",
]


# ==========================================
# Кэшированная загрузка (один раз на сессию)
# ==========================================
@st.cache_resource(show_spinner="Загрузка моделей (первый раз ~1 минута)...")
def load_pipeline(top_n: int, prompt_style: str):
    df = pd.read_csv(DATA_CSV)
    chunks = create_enriched_chunks(df)
    chunks = create_summary_chunks(df, chunks)
    md_table = create_markdown_table(df)

    retriever = HybridIndex("BAAI/bge-m3")
    retriever.build(chunks)
    llm = LocalLLMClient(MODEL_PATH)
    pipeline = RAGPipeline(retriever, llm, top_n=top_n, prompt_style=prompt_style)
    return pipeline, chunks, md_table


@st.cache_data
def load_accuracy() -> float | None:
    if Path(RESULTS_CSV).exists():
        try:
            return float(pd.read_csv(RESULTS_CSV)["is_correct"].mean())
        except Exception:
            return None
    return None


# ==========================================
# Боковая панель
# ==========================================
with st.sidebar:
    st.title("⚙️ Настройки")

    top_n = st.slider(
        "Top-N чанков в контексте", 3, 15, 8,
        help="Сколько фрагментов базы знаний передавать модели. "
             "8 — оптимум для фактов, 15 — для сравнений и агрегации.",
    )

    prompt_style = st.selectbox(
        "Стиль промпта",
        options=["fewshot", "simple", "dynamic", "table"],
        format_func=lambda x: {
            "fewshot": "Few-Shot (рекомендуется, 85%)",
            "simple":  "Simple (быстро, 40%)",
            "dynamic": "Dynamic Top-N (для сложных, 85%)",
            "table":   "Markdown Table (быстро + точно, 82%)",
        }[x],
    )

    show_chunks = st.toggle("Показывать источники", value=True)
    stream_mode = st.toggle("Стриминг ответа", value=True,
                            help="Показывать текст по токенам, как в ChatGPT")

    st.divider()

    acc = load_accuracy()
    if acc is not None:
        st.metric("🏆 Accuracy на бенчмарке", f"{acc:.2%}",
                  help="Точность на 55 вопросах (final_results.csv)")
    else:
        st.info("Файл с результатами не найден")

    st.divider()

    st.markdown("**💡 Примеры вопросов**")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, use_container_width=True, key=f"ex_{i}"):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    if st.button("🗑 Очистить диалог", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# Основная область
# ==========================================
st.title("📱 RAG-ассистент по тарифам Молдовы")
st.caption(
    "Задайте вопрос о тарифах **Orange**, **Moldcell** или **Moldtelecom** — "
    "система найдёт ответ в базе знаний и покажет источники."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Загружаем пайплайн
try:
    pipeline, chunks, md_table = load_pipeline(top_n, prompt_style)
except FileNotFoundError as e:
    st.error(f"❌ Не найден файл: **{e.filename}**")
    st.info(f"Убедитесь, что {MODEL_PATH} и {DATA_CSV} лежат рядом с app.py")
    st.stop()

# Инфо о базе знаний
with st.expander(f"📚 База знаний ({len(chunks)} чанков)"):
    for c in chunks:
        st.markdown(f"**`{c['chunk_id']}`**")
        st.caption(c["text"])

# Рендер истории диалога
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_chunks and msg.get("chunks"):
            with st.expander(f"📎 Источники ({len(msg['chunks'])})"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(
                        f"**[{i}]** `{c['chunk_id']}` · score = `{c.get('score', 0):.3f}`"
                    )
                    st.caption(c["text"])

# Поле ввода
question = st.chat_input("Задайте вопрос...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

# Обработка вопроса
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Ищу в базе знаний..."):
            k = pipeline._effective_k(question)
            found_chunks = pipeline.retriever.search(question, k=k)
            prompt = pipeline._build_prompt(
                question, found_chunks,
                md_table if prompt_style == "table" else None,
            )

        if stream_mode:
            answer = st.write_stream(pipeline.llm.stream_generate(prompt, max_tokens=100))
        else:
            with st.spinner("Генерирую ответ..."):
                answer = pipeline.llm.generate(prompt, max_tokens=100)
            st.markdown(answer)

        if show_chunks:
            with st.expander(f"📎 Источники ({len(found_chunks)})", expanded=True):
                for i, c in enumerate(found_chunks, 1):
                    st.markdown(
                        f"**[{i}]** `{c['chunk_id']}` · score = `{c.get('score', 0):.3f}`"
                    )
                    st.caption(c["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": found_chunks,
    })