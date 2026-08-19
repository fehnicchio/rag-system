# evaluation/evaluate_rag.py
"""
Avaliação do RAG System usando dataset Amnesty QA (RAGAS).

Uso:
    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --samples 5
    python evaluation/evaluate_rag.py --no-ragas
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Carregar Dataset ─────────────────────────────────────────────────────────

def load_amnesty_dataset(n_samples: int = None) -> list[dict]:
    """
    Carrega o dataset Amnesty QA do HuggingFace.
    Colunas: user_input, reference, response, retrieved_contexts
    """
    logger.info("Carregando dataset Amnesty QA...")

    from datasets import load_dataset

    ds = load_dataset(
        'explodinggradients/amnesty_qa',
        'english_v3',
        split='eval'
    )

    samples = ds[:n_samples] if n_samples else ds[:]

    questions = []
    total = len(samples['user_input'])

    for i in range(total):
        questions.append({
            "question":    samples['user_input'][i],
            "ground_truth": samples['reference'][i],
            "reference_contexts": samples['retrieved_contexts'][i],
        })

    logger.info(f"✅ {len(questions)} exemplo(s) carregado(s) do Amnesty QA")

    # Mostrar exemplo
    logger.info(f"  Exemplo de pergunta: {questions[0]['question'][:80]}...")

    return questions


# ─── Preparar documentos do dataset ──────────────────────────────────────────

def prepare_dataset_documents(questions: list[dict], docs_dir: str = "data/documents") -> int:
    """
    Extrai os contextos do dataset e salva como documentos
    para o RAG poder indexar e usar como base de conhecimento.
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    # Arquivo com todos os contextos do dataset
    output_file = docs_path / "amnesty_qa_contexts.txt"

    logger.info(f"Salvando contextos do dataset em {output_file}...")

    seen = set()
    total_chunks = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Amnesty International — Human Rights Reports\n\n")

        for i, q in enumerate(questions):
            contexts = q.get("reference_contexts", [])
            for ctx in contexts:
                if ctx and ctx.strip() and ctx not in seen:
                    seen.add(ctx)
                    f.write(f"{ctx.strip()}\n\n")
                    total_chunks += 1

    logger.info(f"✅ {total_chunks} contexto(s) únicos salvos em {output_file}")
    return total_chunks


# ─── Rodar perguntas no RAG ───────────────────────────────────────────────────

def run_questions(rag, questions: list[dict]) -> list[dict]:
    """Executa cada pergunta no RAG e coleta resultados."""
    import time

    results = []

    for i, q in enumerate(questions, 1):
        logger.info(f"  [{i}/{len(questions)}] {q['question'][:70]}...")

        start = time.time()
        try:
            response      = rag.query(q["question"])
            response_time = time.time() - start

            result = {
                "question":           q["question"],
                "ground_truth":       q["ground_truth"],
                "answer":             response["answer"],
                "sources":            response["sources"],
                "reference_contexts": q.get("reference_contexts", []),
                "response_time":      response_time,
                "error":              False,
            }
            logger.info(f"    ✅ {response_time:.2f}s | {len(response['sources'])} fonte(s)")

        except Exception as e:
            response_time = time.time() - start
            logger.error(f"    ❌ {e}")
            result = {
                "question":           q["question"],
                "ground_truth":       q["ground_truth"],
                "answer":             "",
                "sources":            [],
                "reference_contexts": q.get("reference_contexts", []),
                "response_time":      response_time,
                "error":              True,
                "error_msg":          str(e),
            }

        results.append(result)

    return results


# ─── Métricas básicas ─────────────────────────────────────────────────────────

def compute_basic_metrics(results: list[dict]) -> dict:
    """Métricas simples sem LLM externo."""

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    similarity_scores = []
    has_sources_list  = []
    response_times    = []
    error_count       = 0

    for r in results:
        if r.get("error"):
            error_count += 1
            continue

        # Similaridade com ground truth
        sim = SequenceMatcher(
            None,
            r["answer"].lower(),
            r["ground_truth"].lower()
        ).ratio()
        similarity_scores.append(sim)

        # Tem fontes?
        has_sources_list.append(1.0 if r["sources"] else 0.0)

        # Tempo de resposta
        response_times.append(r["response_time"])

    return {
        "answer_similarity":   round(avg(similarity_scores), 4),
        "has_sources_rate":    round(avg(has_sources_list), 4),
        "avg_response_time_s": round(avg(response_times), 2),
        "total_questions":     len(results),
        "errors":              error_count,
        "success_rate":        round((len(results) - error_count) / len(results), 4) if results else 0,
    }


# ─── Métricas RAGAS ──────────────────────────────────────────────────────────

def compute_ragas_metrics(results: list[dict]) -> dict:
    """Avaliação avançada com RAGAS usando Ollama local."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        from datasets import Dataset

        # Filtrar resultados válidos
        valid = [
            r for r in results
            if not r.get("error")
            and r.get("answer")
            and r.get("sources")
        ]

        if not valid:
            logger.warning("Nenhum resultado válido para RAGAS")
            return {}

        logger.info(f"Preparando {len(valid)} resultado(s) para RAGAS...")

        # Montar dataset no formato RAGAS
        data = {
            "question":     [r["question"]     for r in valid],
            "answer":       [r["answer"]        for r in valid],
            "contexts":     [
                [s.get("content", "") for s in r["sources"]]
                for r in valid
            ],
            "ground_truth": [r["ground_truth"]  for r in valid],
        }

        dataset = Dataset.from_dict(data)

        # ── Configurar Ollama explicitamente ──────────────────────────────
        logger.info("Configurando Ollama para RAGAS (sem OpenAI)...")
        
        ollama_llm = OllamaLLM(
            model="mistral",
            temperature=0,
            base_url="http://localhost:11434"
        )
        ollama_embeddings = OllamaEmbeddings(
            model="mistral",
            base_url="http://localhost:11434"
        )

        llm        = LangchainLLMWrapper(ollama_llm)
        embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

        # ── Configurar métricas para usar Ollama ──────────────────────────
        faithfulness.llm        = llm
        faithfulness.embeddings = embeddings

        answer_relevancy.llm        = llm
        answer_relevancy.embeddings = embeddings

        context_precision.llm        = llm
        context_precision.embeddings = embeddings

        logger.info(f"Executando RAGAS em {len(valid)} resultado(s)...")
        logger.info("(Isso pode demorar 10-30 min dependendo da máquina)")

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
            ],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        output = {}
        
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
            if metric_name in result:
                val = result[metric_name]
                output[metric_name] = round(float(val), 4)

        if not output:
            logger.warning("RAGAS retornou métricas vazias mesmo com Ollama configurado")
            logger.warning("Verifique se Ollama está rodando: curl http://localhost:11434")
            return {}

        logger.info(f"✅ RAGAS concluído: {output}")
        return output

    except ImportError as e:
        logger.error(f"RAGAS não instalado: {e}")
        logger.error("Execute: pip install ragas")
        return {}
    except Exception as e:
        logger.error(f"Erro no RAGAS: {e}")
        logger.error("Verifique se Ollama está rodando e o modelo mistral está instalado")
        return {}


# ─── Relatório ────────────────────────────────────────────────────────────────

def print_report(report: dict):
    """Imprime relatório formatado no terminal."""
    meta = report["metadata"]
    b    = report["basic_metrics"]
    r    = report.get("ragas_metrics", {})

    print(f"\n{'='*60}")
    print(f"  RAG SYSTEM — RELATÓRIO DE AVALIAÇÃO")
    print(f"{'='*60}")
    print(f"  Dataset  : Amnesty QA (RAGAS benchmark)")
    print(f"  Data     : {meta['timestamp'][:19]}")
    print(f"  Modelo   : {meta['model']}")
    print(f"  Amostras : {meta['total_samples']}")
    print(f"{'='*60}")

    print(f"\n📊 MÉTRICAS BÁSICAS")
    print(f"  {'Similaridade com ground truth':<35} {b['answer_similarity']:.2%}")
    print(f"  {'Respostas com fontes':<35} {b['has_sources_rate']:.2%}")
    print(f"  {'Taxa de sucesso':<35} {b['success_rate']:.2%}")
    print(f"  {'Tempo médio de resposta':<35} {b['avg_response_time_s']:.2f}s")
    print(f"  {'Erros':<35} {b['errors']}")

    if r:
        print(f"\n🎯 MÉTRICAS RAGAS (Avançadas)")
        print(f"  {'Faithfulness (Fidelidade)':<35} {r.get('faithfulness', 0):.2%}")
        print(f"  {'Answer Relevancy (Relevância)':<35} {r.get('answer_relevancy', 0):.2%}")
        print(f"  {'Context Precision (Precisão)':<35} {r.get('context_precision', 0):.2%}")
    else:
        print(f"\n⚠️  RAGAS não executado (use sem --no-ragas para ativar)")

    # Score geral
    all_scores = [b["answer_similarity"], b["has_sources_rate"], b["success_rate"]]
    if r:
        all_scores += [
            r.get("faithfulness", 0),
            r.get("answer_relevancy", 0),
            r.get("context_precision", 0),
        ]

    overall = sum(all_scores) / len(all_scores)
    grade   = (
        "🏆 Excelente"         if overall >= 0.80 else
        "✅ Bom"               if overall >= 0.65 else
        "⚠️  Regular"          if overall >= 0.45 else
        "❌ Precisa melhorar"
    )

    print(f"\n{'─'*60}")
    print(f"  SCORE GERAL : {overall:.2%}  {grade}")
    print(f"{'='*60}")

    # Detalhes por pergunta
    print(f"\n📝 DETALHES POR PERGUNTA")
    print(f"{'─'*60}")
    for i, res in enumerate(report["results"], 1):
        status = "❌" if res.get("error") else "✅"
        print(f"\n  {status} [{i}] {res['question'][:75]}")
        if not res.get("error"):
            sim = SequenceMatcher(
                None,
                res["answer"].lower(),
                res["ground_truth"].lower()
            ).ratio()
            print(f"      Similaridade : {sim:.2%}")
            print(f"      Tempo        : {res['response_time']:.2f}s")
            print(f"      Fontes       : {len(res['sources'])}")
            print(f"      Resposta     : {res['answer'][:120]}...")
        else:
            print(f"      Erro: {res.get('error_msg', '?')[:80]}")

    print(f"\n{'='*60}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Avaliação do RAG com Amnesty QA")
    parser.add_argument("--samples",  type=int, default=None,
                        help="Número de amostras (padrão: todas as 20)")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Pular métricas RAGAS (mais rápido)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Pular construção da base (usar índice existente)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Arquivo de saída JSON")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  RAG SYSTEM — Iniciando Avaliação")
    logger.info("=" * 60)

    # 1. Carregar dataset
    questions = load_amnesty_dataset(n_samples=args.samples)

    # 2. Preparar documentos (contextos do dataset)
    if not args.skip_build:
        logger.info("\nPreparando base de conhecimento com contextos do dataset...")
        n_docs = prepare_dataset_documents(questions)
        logger.info(f"  {n_docs} contextos salvos como documentos")

    # 3. Inicializar RAG
    logger.info("\nInicializando RAG System...")
    from src.rag_system import RAGSystem

    try:
        rag = RAGSystem()
    except Exception as e:
        logger.error(f"Erro ao inicializar RAG: {e}")
        sys.exit(1)

    # 4. Construir índice com documentos do dataset
    if not args.skip_build:
        logger.info("Construindo índice FAISS com contextos do dataset...")
        success = rag.build_knowledge_base()
        if not success:
            logger.error("Falha ao construir base. Verifique os documentos.")
            sys.exit(1)
    elif not rag.get_status()["system_ready"]:
        logger.error("Sistema não está pronto. Remova --skip-build.")
        sys.exit(1)

    # 5. Rodar perguntas
    logger.info(f"\nProcessando {len(questions)} pergunta(s)...")
    results = run_questions(rag, questions)

    # 6. Calcular métricas
    logger.info("\nCalculando métricas...")
    basic_metrics = compute_basic_metrics(results)

    ragas_metrics = {}
    if not args.no_ragas:
        logger.info("Executando RAGAS (avaliação avançada)...")
        ragas_metrics = compute_ragas_metrics(results)

    # 7. Montar relatório
    report = {
        "metadata": {
            "timestamp":     datetime.now().isoformat(),
            "dataset":       "explodinggradients/amnesty_qa english_v3",
            "total_samples": len(questions),
            "model":         getattr(rag, "model_name", "unknown"),
            "ragas_used":    not args.no_ragas,
        },
        "basic_metrics":  basic_metrics,
        "ragas_metrics":  ragas_metrics,
        "results":        results,
    }

    # 8. Imprimir
    print_report(report)

    # 9. Salvar JSON
    out = args.output or f"evaluation/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"✅ Resultado salvo em: {out}")


if __name__ == "__main__":
    main()