# Retrieval-augmented chatbot for university programme information

Research internship at the **International University of Rabat (UIR)**, summer 2024,
supervised by **Mounir Ghogho** and **Hakim Hafidi**.
Full report: [`report_UIR_chatbot_langchain.pdf`](report_UIR_chatbot_langchain.pdf).

The goal was a conversational assistant that answers student questions about UIR
programmes and admissions from the university's own documents, rather than from
an LLM's parametric memory, and that says "I don't know" when the documents don't
contain the answer.

## Architecture

```
PDF / text sources ──► chunking ──► embeddings ──► vector store
                                                        │ top-k similarity
user question ──► question condensation ──► retriever ──┘
     ▲             (uses chat history)          │
     │                                          ▼
     └──────── answer ◄── LLM (temperature 0) ◄── grounded prompt + k chunks
```

| Stage | Choice in `chatbotuir.ipynb` | Why |
|---|---|---|
| Sources | the UIR programme PDF, loaded with `PyPDFLoader`; the report also used two text extracts of the School of Engineering pages | web scraping of the site was attempted and abandoned (see report §2.2) |
| Chunking | `RecursiveCharacterTextSplitter`, 500 characters, 50 overlap | short chunks keep retrieved context specific; the overlap avoids cutting answers in half |
| Embeddings | OpenAI embeddings | |
| Vector store | `DocArrayInMemorySearch` (Chroma also set up) | corpus is small, so in-memory exact search is enough |
| Retrieval | cosine similarity, *k* = 4 | |
| Generation | `gpt-3.5-turbo`, temperature 0, via `ConversationalRetrievalChain` | follow-up questions are first rewritten into standalone queries using the chat history, then retrieval runs on the rewritten query |
| Grounding prompt | LangChain's default QA prompt (answer from the retrieved context, say so when it doesn't contain the answer). A stricter custom prompt, `QA_CHAIN_PROMPT`, is defined in the notebook but not yet passed to the chain | limits hallucinated programme details |
| Interface | Gradio front end over a session object that also exposes the rewritten query and the retrieved chunks (Panel views) | seeing what was retrieved makes wrong answers diagnosable |
| Tracing | LangSmith | per-call inspection of prompts and retrievals |

## Running it

Requires an `OPENAI_API_KEY` in a local `.env` file and the source PDF
(`documentati.pdf`), which is not in this repository. The notebook uses the 2024
LangChain API (`langchain.chains.ConversationalRetrievalChain`); later LangChain
releases moved or deprecated several of these classes.

## Limitations

- **No quantitative evaluation.** The project was evaluated qualitatively, on example
  questions. A proper evaluation would need a labelled question set to measure
  retrieval recall@k and answer faithfulness to the retrieved chunks, and to compare
  chunk sizes, *k* and embedding models.
- **Coverage.** The corpus covers the School of Engineering better than other schools,
  because scraping the rest of the site failed.
- **Single LLM provider.** The comparison of LLMs and vector stores planned in the
  report (§1.4) was not carried out.
