# evaluation/evaluate_squad.py
"""
Avaliação do RAG System usando SQuAD 2.0.

Por que SQuAD?
- Padrão ouro da indústria para QA
- Perguntas baseadas em contextos reais da Wikipedia
- Inclui perguntas sem resposta (testa se RAG sabe dizer "não sei")
- Permite comparar com benchmarks públicos

Uso:
    python evaluation/evaluate_squad.py
    python evaluation/evaluate_squad.py --samples 20
    python evaluation/evaluate_squad.py --topic technology
    python evaluation/evaluate_squad.py --no-ragas
"""

import sys
import json
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Tópicos disponíveis no SQuAD ────────────────────────────────────────────
# Para filtrar perguntas por tema específico

TOPICS = {
    "technology":   ["computer", "software", "internet", "technology", "digital"],
    "science":      ["science", "research", "biology", "physics", "chemistry"],
    "history":      ["history", "war", "century", "ancient", "historical"],
    "geography":    ["country", "city", "river", "mountain", "located"],
    "general":      None,  # sem filtro
}


# ─── Carregar Dataset SQuAD ───────────────────────────────────────────────────

def load_squad_dataset(
    n_samples: int = 20,
    topic: str = "general",
    only_answerable: bool = True
) -> list[dict]:
    """
    Carrega amostras do SQuAD 2.0.

    Args:
        n_samples:        Quantas perguntas carregar
        topic:            Filtrar por tópico (technology, science, etc.)
        only_answerable:  Usar só perguntas com resposta definida
    """
    logger.info(f"Carregando SQuAD 2.0 (topic={topic}, n={n_samples})...")

    from datasets import load_dataset

    ds = load_dataset(
        'rajpurkar/squad_v2',
        split='validation',
        streaming=True
    )

    keywords = TOPICS.get(topic)
    questions = []

    for sample in ds:
        if len(questions) >= n_samples:
            break

        # Filtrar sem resposta se solicitado
        if only_answerable and not sample['answers']['text']:
            continue

        # Filtrar por tópico
        if keywords:
            context_lower = sample['context'].lower()
            if not any(kw in context_lower for kw in keywords):
                continue

        # Pegar primeira resposta como ground truth
        answers = sample['answers']['text']
        ground_truth = answers[0] if answers else ""

        questions.append({
            "question":     sample['question'],
            "ground_truth": ground_truth,
            "all_answers":  answers,
            "context":      sample['context'],
            "title":        sample.get('title', ''),
            "id":           sample.get('id', ''),
        })

    logger.info(f"✅ {len(questions)} pergunta(s) carregada(s) do SQuAD 2.0")

    if questions:
        logger.info(f"  Exemplo: {questions[0]['question']}")
        logger.info(f"  Resposta: {questions[0]['ground_truth']}")
        logger.info(f"  Tópico: {questions[0]['title']}")

    return questions


# ─── Preparar Documentos ──────────────────────────────────────────────────────

def prepare_squad_documents(
    questions: list[dict],
    docs_dir: str = "data/documents"
) -> int:
    """
    Salva os contextos do SQuAD como documentos para o RAG indexar.
    Cada contexto único vira um chunk separado para melhor retrieval.
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    output_file = docs_path / "squad_v2_contexts.txt"
    logger.info(f"Salvando contextos SQuAD em {output_file}...")

    # Agrupar por título (tópico)
    by_title = {}
    for q in questions:
        title = q.get('title', 'Unknown')
        if title not in by_title:
            by_title[title] = set()
        by_title[title].add(q['context'])

    total_chunks = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for title, contexts in by_title.items():
            f.write(f"# {title}\n\n")
            for ctx in contexts:
                f.write(f"{ctx.strip()}\n\n")
                total_chunks += 1

    logger.info(f"✅ {total_chunks} contexto(s) únicos salvos")
    return total_chunks


# ─── Métricas Específicas do SQuAD ───────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Normaliza resposta para comparação (remove artigos, pontuação, espaços)."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_exact_match(prediction: str, ground_truths: list[str]) -> float:
    """Exact Match: 1.0 se a resposta normalizada bate exatamente."""
    pred_normalized = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_normalized == normalize_answer(gt):
            return 1.0
    return 0.0


def compute_f1_score(prediction: str, ground_truths: list[str]) -> float:
    """
    F1 Score por tokens: métrica padrão do SQuAD.
    Mede sobreposição de palavras entre resposta e ground truth.
    """
    pred_tokens = normalize_answer(prediction).split()

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()

        if not pred_tokens or not gt_tokens:
            continue

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue

        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(gt_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)

    return best_f1


def compute_squad_metrics(results: list[dict]) -> dict:
    """Calcula métricas padrão do SQuAD (EM e F1)."""

    exact_matches = []
    f1_scores     = []
    has_sources   = []
    response_times = []
    errors = 0

    for r in results:
        if r.get("error"):
            errors += 1
            continue

        answer      = r.get("answer", "")
        all_answers = r.get("all_answers", [r.get("ground_truth", "")])

        # Exact Match
        em = compute_exact_match(answer, all_answers)
        exact_matches.append(em)

        # F1
        f1 = compute_f1_score(answer, all_answers)
        f1_scores.append(f1)

        # Has sources
        has_sources.append(1.0 if r.get("sources") else 0.0)

        # Tempo
        response_times.append(r.get("response_time", 0))

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    return {
        "exact_match":       round(avg(exact_matches), 4),
        "f1_score":          round(avg(f1_scores), 4),
        "has_sources_rate":  round(avg(has_sources), 4),
        "avg_response_time": round(avg(response_times), 2),
        "total_questions":   len(results),
        "errors":            errors,
        "success_rate":      round((len(results) - errors) / len(results), 4) if results else 0,
    }


# ─── Rodar Perguntas ─────────────────────────────────────────────────────────

def run_questions(rag, questions: list[dict]) -> list[dict]:
    """Executa cada pergunta no RAG."""
    import time
    results = []

    for i, q in enumerate(questions, 1):
        logger.info(f"  [{i}/{len(questions)}] {q['question'][:70]}...")

        start = time.time()
        try:
            response      = rag.query(q["question"])
            response_time = time.time() - start

            result = {
                "question":      q["question"],
                "ground_truth":  q["ground_truth"],
                "all_answers":   q.get("all_answers", []),
                "answer":        response["answer"],
                "sources":       response["sources"],
                "response_time": response_time,
                "title":         q.get("title", ""),
                "error":         False,
            }

            em = compute_exact_match(response["answer"], q.get("all_answers", [q["ground_truth"]]))
            f1 = compute_f1_score(response["answer"], q.get("all_answers", [q["ground_truth"]]))

            logger.info(f"    ✅ {response_time:.1f}s | EM={em:.2f} | F1={f1:.2f} | Fontes={len(response['sources'])}")

        except Exception as e:
            response_time = time.time() - start
            logger.error(f"    ❌ {e}")
            result = {
                "question":      q["question"],
                "ground_truth":  q["ground_truth"],
                "all_answers":   q.get("all_answers", []),
                "answer":        "",
                "sources":       [],
                "response_time": response_time,
                "title":         q.get("title", ""),
                "error":         True,
                "error_msg":     str(e),
            }

        results.append(result)

    return results


# ─── Relatório ────────────────────────────────────────────────────────────────

def print_report(report: dict):
    meta    = report["metadata"]
    m       = report["squad_metrics"]
    ragas   = report.get("ragas_metrics", {})

    print(f"\n{'='*65}")
    print(f"  RAG SYSTEM — RELATÓRIO SQuAD 2.0")
    print(f"{'='*65}")
    print(f"  Dataset  : SQuAD 2.0 (rajpurkar/squad_v2)")
    print(f"  Tópico   : {meta['topic']}")
    print(f"  Data     : {meta['timestamp'][:19]}")
    print(f"  Modelo   : {meta['model']}")
    print(f"  Amostras : {meta['total_samples']}")
    print(f"{'='*65}")

    print(f"\n📊 MÉTRICAS SQuAD (Padrão da Indústria)")
    print(f"  {'Exact Match (EM)':<40} {m['exact_match']:.2%}")
    print(f"  {'F1 Score (sobreposição de tokens)':<40} {m['f1_score']:.2%}")
    print(f"  {'Respostas com fontes':<40} {m['has_sources_rate']:.2%}")
    print(f"  {'Taxa de sucesso':<40} {m['success_rate']:.2%}")
    print(f"  {'Tempo médio por pergunta':<40} {m['avg_response_time']:.1f}s")
    print(f"  {'Erros':<40} {m['errors']}")

    if ragas:
        print(f"\n🎯 MÉTRICAS RAGAS")
        print(f"  {'Faithfulness':<40} {ragas.get('faithfulness', 0):.2%}")
        print(f"  {'Answer Relevancy':<40} {ragas.get('answer_relevancy', 0):.2%}")
        print(f"  {'Context Precision':<40} {ragas.get('context_precision', 0):.2%}")

    # Score geral
    scores = [m['f1_score'], m['has_sources_rate'], m['success_rate']]
    if ragas:
        scores += [ragas.get('faithfulness', 0), ragas.get('answer_relevancy', 0)]

    overall = sum(scores) / len(scores)
    grade   = (
        "🏆 Excelente"       if overall >= 0.80 else
        "✅ Bom"             if overall >= 0.65 else
        "⚠️  Regular"        if overall >= 0.45 else
        "❌ Precisa melhorar"
    )

    print(f"\n{'─'*65}")
    print(f"  SCORE GERAL : {overall:.2%}  {grade}")
    print(f"{'='*65}")

    # Benchmark de referência
    print(f"\n📌 REFERÊNCIA DO MERCADO (SQuAD 2.0)")
    print(f"  GPT-4 (OpenAI)     : EM ~85%  | F1 ~90%")
    print(f"  GPT-3.5            : EM ~75%  | F1 ~83%")
    print(f"  BERT Large         : EM ~80%  | F1 ~83%")
    print(f"  Mistral 7B (meta)  : EM ~55%  | F1 ~65%")
    print(f"  Seu Sistema        : EM {m['exact_match']:.0%}    | F1 {m['f1_score']:.0%}")
    print(f"{'='*65}")

    # Detalhes por pergunta
    print(f"\n📝 DETALHES")
    print(f"{'─'*65}")
    for i, r in enumerate(report["results"], 1):
        status = "❌" if r.get("error") else "✅"
        em = compute_exact_match(r["answer"], r.get("all_answers", [r["ground_truth"]]))
        f1 = compute_f1_score(r["answer"], r.get("all_answers", [r["ground_truth"]]))
        print(f"\n  {status} [{i}] [{r.get('title','?')[:30]}]")
        print(f"      Q: {r['question'][:75]}")
        print(f"      GT: {r['ground_truth'][:60]}")
        print(f"      RAG: {r['answer'][:80].strip()}...")
        print(f"      EM={em:.2f} | F1={f1:.2f} | {r['response_time']:.1f}s")

    print(f"\n{'='*65}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Avaliação RAG com SQuAD 2.0")
    parser.add_argument("--samples",     type=int,   default=20,
                        help="Número de amostras (padrão: 20)")
    parser.add_argument("--topic",       type=str,   default="general",
                        choices=list(TOPICS.keys()),
                        help="Filtrar por tópico")
    parser.add_argument("--no-ragas",    action="store_true",
                        help="Pular métricas RAGAS")
    parser.add_argument("--skip-build",  action="store_true",
                        help="Pular construção da base")
    parser.add_argument("--output",      type=str,   default=None,
                        help="Arquivo de saída JSON")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("  RAG SYSTEM — Avaliação SQuAD 2.0")
    logger.info("=" * 65)

    # 1. Carregar dataset
    questions = load_squad_dataset(
        n_samples=args.samples,
        topic=args.topic
    )

    if not questions:
        logger.error("Nenhuma pergunta carregada. Tente outro tópico.")
        sys.exit(1)

    # 2. Preparar documentos
    if not args.skip_build:
        logger.info("Preparando base de conhecimento com contextos SQuAD...")
        prepare_squad_documents(questions)

    # 3. Inicializar RAG
    logger.info("Inicializando RAG System...")
    from src.rag_system import RAGSystem

    try:
        rag = RAGSystem()
    except Exception as e:
        logger.error(f"Erro: {e}")
        sys.exit(1)

    # 4. Construir índice
    if not args.skip_build:
        logger.info("Construindo índice FAISS...")
        if not rag.build_knowledge_base():
            logger.error("Falha ao construir base.")
            sys.exit(1)
    elif not rag.get_status()["system_ready"]:
        logger.error("Sistema não pronto. Remova --skip-build.")
        sys.exit(1)

    # 5. Rodar perguntas
    logger.info(f"Processando {len(questions)} pergunta(s)...")
    results = run_questions(rag, questions)

    # 6. Calcular métricas
    squad_metrics = compute_squad_metrics(results)

    ragas_metrics = {}
    if not args.no_ragas:
        logger.info("Executando métricas RAGAS...")
        # Importar função do evaluate_rag.py
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from evaluate_rag import compute_ragas_metrics
            ragas_metrics = compute_ragas_metrics(results)
        except Exception as e:
            logger.error(f"RAGAS falhou: {e}")

    # 7. Montar relatório
    report = {
        "metadata": {
            "timestamp":     datetime.now().isoformat(),
            "dataset":       "rajpurkar/squad_v2",
            "topic":         args.topic,
            "total_samples": len(questions),
            "model":         getattr(rag, "model_name", "unknown"),
            "ragas_used":    not args.no_ragas,
        },
        "squad_metrics":  squad_metrics,
        "ragas_metrics":  ragas_metrics,
        "results":        results,
    }

    # 8. Imprimir
    print_report(report)

    # 9. Salvar
    out = args.output or f"evaluation/squad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"✅ Resultado salvo em: {out}")


if __name__ == "__main__":
    main()