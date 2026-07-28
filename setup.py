#!/usr/bin/env python3
"""
setup.py — Script de setup e health check do RAG System.

Uso:
    python setup.py           # Verifica + instala dependências + testa imports
    python setup.py --fix     # Força reinstalação de tudo
    python setup.py --run     # Verifica + sobe o servidor automaticamente
"""
import sys
import subprocess
import importlib
import argparse
from pathlib import Path

# ── Cores para terminal ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"{GREEN}✅ {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}⚠️  {msg}{RESET}")
def err(msg):   print(f"{RED}❌ {msg}{RESET}")
def info(msg):  print(f"{BLUE}ℹ️  {msg}{RESET}")
def title(msg): print(f"\n{BOLD}{msg}{RESET}")

# ── Imports críticos para testar ───────────────────────────────────────────────
CRITICAL_IMPORTS = [
    ("fastapi",                    "FastAPI"),
    ("uvicorn",                    "Uvicorn"),
    ("langchain_ollama",           "LangChain Ollama"),
    ("langchain_core",             "LangChain Core"),
    ("langchain_huggingface",      "LangChain HuggingFace"),
    ("langchain_community",        "LangChain Community"),
    ("sentence_transformers",      "Sentence Transformers"),
    ("huggingface_hub",            "HuggingFace Hub"),
    ("faiss",                      "FAISS"),
    ("pdfplumber",                 "PDFPlumber"),
    ("docx",                       "python-docx"),
    ("bs4",                        "BeautifulSoup4"),
    ("dotenv",                     "python-dotenv"),
    ("yaml",                       "PyYAML"),
    ("numpy",                      "NumPy"),
    ("pandas",                     "Pandas"),
]

# ── Imports do projeto ─────────────────────────────────────────────────────────
PROJECT_IMPORTS = [
    ("src.utils",                              "Utils"),
    ("src.embeddings.embedding_manager",       "EmbeddingManager"),
    ("src.retrieval.vector_store",             "VectorStore"),
    ("src.generation.llm_chain",               "LLMChain"),
    ("src.ranking.reranker",                   "Reranker"),
    ("src.cache.cache_manager",                "CacheManager"),
    ("src.database.models",                    "Database"),
    ("src.scraper.web_scraper",                "WebScraper"),
    ("src.learning.learning_store",            "LearningStore"),
    ("src.learning.feedback_learner",          "FeedbackLearner"),
    ("src.rag_system",                         "RAGSystem"),
]


def run_command(cmd: list, capture=True) -> tuple[int, str]:
    """Executa comando e retorna (returncode, output)."""
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True
    )
    
    if capture:
        output = (result.stdout or "") + (result.stderr or "")
    else:
        output = ""
    
    return result.returncode, output.strip()


def check_python_version():
    title("🐍 Verificando Python")
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    
    if major == 3 and minor >= 11:
        ok(f"Python {version_str}")
    elif major == 3 and minor >= 9:
        warn(f"Python {version_str} — recomendado 3.11+")
    else:
        err(f"Python {version_str} — mínimo 3.9 necessário")
        sys.exit(1)


def check_ollama():
    title("🤖 Verificando Ollama")
    
    code, _ = run_command(["curl", "-s", "http://localhost:11434"])
    if code == 0:
        ok("Ollama está rodando em localhost:11434")
    else:
        warn("Ollama não detectado em localhost:11434")
        info("Abra o Ollama antes de usar o sistema")
        info("Download: https://ollama.ai")
        return

    code, output = run_command(["ollama", "list"])
    if code == 0:
        lines = output.strip().split("\n")
        models = [l.split()[0] for l in lines[1:] if l.strip()]
        if models:
            ok(f"Modelos instalados: {', '.join(models)}")
            if "mistral" not in " ".join(models):
                warn("Mistral não encontrado — instale com: ollama pull mistral")
        else:
            warn("Nenhum modelo instalado")
            info("Instale com: ollama pull mistral")
    else:
        warn("Não foi possível listar modelos do Ollama")


def install_requirements(force=False):
    title("📦 Instalando/Verificando dependências")
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", "requirements.txt",
        "--prefer-binary",
    ]
    
    if not force:
        cmd.append("--quiet")
        info("Verificando dependências (use --fix para forçar reinstalação)...")
    else:
        warn("Forçando reinstalação de todas as dependências...")

    code, output = run_command(cmd, capture=False)
    
    if code == 0:
        ok("Dependências OK")
    else:
        err("Falha ao instalar dependências")
        err(output)
        sys.exit(1)


def test_imports():
    title("🔬 Testando imports de bibliotecas")
    
    failed = []
    
    for module, name in CRITICAL_IMPORTS:
        try:
            importlib.import_module(module)
            ok(name)
        except ImportError as e:
            err(f"{name}: {e}")
            failed.append((module, name, str(e)))
    
    return failed


def test_project_imports():
    title("🔬 Testando imports do projeto")
    
    # Adicionar raiz ao path
    sys.path.insert(0, str(Path(__file__).parent))
    
    failed = []
    
    for module, name in PROJECT_IMPORTS:
        try:
            importlib.import_module(module)
            ok(name)
        except Exception as e:
            err(f"{name}: {e}")
            failed.append((module, name, str(e)))
    
    return failed


def check_env():
    title("⚙️  Verificando configuração")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        ok(".env encontrado")
    elif env_example.exists():
        warn(".env não encontrado — copiando de .env.example")
        env_file.write_text(env_example.read_text())
        ok(".env criado a partir de .env.example")
    else:
        warn(".env não encontrado — crie o arquivo com suas configurações")
    
    config_file = Path("config/config.yaml")
    if config_file.exists():
        ok("config/config.yaml encontrado")
    else:
        err("config/config.yaml não encontrado!")
    
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs = list(docs_dir.glob("*.*"))
    
    if docs:
        ok(f"data/documents/ — {len(docs)} arquivo(s) encontrado(s)")
    else:
        warn("data/documents/ está vazio — adicione documentos para usar o sistema")


def try_fix_known_issues():
    """Tenta corrigir problemas conhecidos de compatibilidade."""
    title("🔧 Corrigindo problemas conhecidos de compatibilidade")
    
    fixes = [
        # sentence-transformers + huggingface_hub
        [sys.executable, "-m", "pip", "install",
         "sentence-transformers==3.0.1", "huggingface_hub==0.24.0",
         "--prefer-binary", "--quiet"],
        
        # langchain-core + langchain-ollama
        [sys.executable, "-m", "pip", "install",
         "langchain-core==0.3.65", "langchain-ollama==0.3.3",
         "--prefer-binary", "--quiet"],
    ]
    
    for cmd in fixes:
        packages = [p for p in cmd if not p.startswith("-") and p != sys.executable and p != "-m" and p != "pip" and p != "install"]
        info(f"Fixando: {' '.join(packages)}")
        run_command(cmd)
    
    ok("Correções aplicadas")


def print_summary(lib_failures, proj_failures):
    title("📋 Resumo")
    
    total_lib  = len(CRITICAL_IMPORTS)
    total_proj = len(PROJECT_IMPORTS)
    ok_lib     = total_lib - len(lib_failures)
    ok_proj    = total_proj - len(proj_failures)
    
    print(f"  Bibliotecas: {ok_lib}/{total_lib} OK")
    print(f"  Módulos do projeto: {ok_proj}/{total_proj} OK")
    
    if lib_failures or proj_failures:
        print()
        err("Problemas encontrados:")
        for module, name, error in (lib_failures + proj_failures):
            print(f"  • {name}: {error[:80]}")
        print()
        info("Tente: python setup.py --fix")
        return False
    
    print()
    ok("Tudo certo! Sistema pronto para uso.")
    info("Para subir o servidor: python server.py")
    return True


def start_server():
    title("🚀 Subindo o servidor")
    info("Acesse: http://localhost:8000")
    info("Pressione Ctrl+C para parar")
    print()
    
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Setup e health check do RAG System"
    )
    parser.add_argument("--fix",  action="store_true", help="Força reinstalação de todas as dependências")
    parser.add_argument("--run",  action="store_true", help="Sobe o servidor após verificação")
    parser.add_argument("--skip-install", action="store_true", help="Pula instalação de dependências")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}  RAG Q&A System — Setup & Health Check{RESET}")
    print(f"{BOLD}{'='*50}{RESET}")

    # 1. Verificar Python
    check_python_version()

    # 2. Verificar Ollama
    check_ollama()

    # 3. Instalar/verificar dependências
    if not args.skip_install:
        install_requirements(force=args.fix)
    
    # 4. Se --fix, corrigir problemas conhecidos
    if args.fix:
        try_fix_known_issues()

    # 5. Verificar configuração
    check_env()

    # 6. Testar imports
    lib_failures  = test_imports()
    proj_failures = test_project_imports()

    # 7. Se houver falhas e não foi --fix, tentar correção automática
    if (lib_failures or proj_failures) and not args.fix:
        print()
        warn("Problemas detectados — tentando correção automática...")
        try_fix_known_issues()
        
        # Re-testar após correção
        lib_failures  = test_imports()
        proj_failures = test_project_imports()

    # 8. Resumo
    all_ok = print_summary(lib_failures, proj_failures)

    # 9. Subir servidor se --run e tudo OK
    if args.run:
        if all_ok:
            start_server()
        else:
            err("Não foi possível subir o servidor — resolva os erros acima")
            sys.exit(1)


if __name__ == "__main__":
    main()