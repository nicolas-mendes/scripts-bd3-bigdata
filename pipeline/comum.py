from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import DIMENSAO, MODELO_EMBEDDING, Config

_MODELO = None


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def conectar_mysql(cfg: Config, com_banco: bool = True):
    import pymysql

    return pymysql.connect(
        host=cfg.mysql_host,
        port=cfg.mysql_porta,
        user=cfg.mysql_usuario,
        password=cfg.mysql_senha,
        database=cfg.mysql_banco if com_banco else None,
        charset="utf8mb4",
        autocommit=False,
    )


def conectar_milvus(cfg: Config):
    from pymilvus import MilvusClient

    return MilvusClient(uri=cfg.milvus_uri, token=cfg.milvus_token or "")


def carregar_modelo():
    global _MODELO
    if _MODELO is None:
        from sentence_transformers import SentenceTransformer

        log(f"Modelo: {MODELO_EMBEDDING}")
        _MODELO = SentenceTransformer(MODELO_EMBEDDING)
    return _MODELO


def vetorizar(textos: list[str], mostrar_progresso: bool = False, lote: int = 64) -> np.ndarray:
    vetores = carregar_modelo().encode(
        textos,
        batch_size=lote,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=mostrar_progresso,
    )
    vetores = vetores.astype(np.float32)
    assert vetores.shape[1] == DIMENSAO, vetores.shape
    return vetores


@contextmanager
def cronometro():
    inicio = time.perf_counter()
    fim: list[float] = []
    yield lambda: (fim[0] if fim else time.perf_counter()) - inicio
    fim.append(time.perf_counter())


def estatisticas(amostras_ms: list[float]) -> dict:
    return {
        "media_ms": statistics.fmean(amostras_ms),
        "desvio_ms": statistics.stdev(amostras_ms) if len(amostras_ms) > 1 else 0.0,
        "min_ms": min(amostras_ms),
        "max_ms": max(amostras_ms),
        "n": len(amostras_ms),
        "amostras_ms": [round(a, 3) for a in amostras_ms],
    }


def medir(
    funcao: Callable[[], Any],
    repeticoes: int,
    aquecimento: int,
    apos: Callable[[], Any] | None = None,
) -> tuple[dict, Any]:
    amostras: list[float] = []
    resultado = None
    for i in range(aquecimento + repeticoes):
        inicio = time.perf_counter()
        resultado = funcao()
        decorrido = (time.perf_counter() - inicio) * 1000
        if i >= aquecimento:
            amostras.append(decorrido)
        if apos is not None:
            apos()
    return estatisticas(amostras), resultado


def vetor_para_blob(vetor: np.ndarray) -> bytes:
    return np.asarray(vetor, dtype=np.float32).tobytes()


def blobs_para_matriz(blobs: list[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, DIMENSAO)


def truncar_bytes(texto: str, limite: int) -> str:
    codificado = texto.encode("utf-8")
    if len(codificado) <= limite:
        return texto
    return codificado[:limite].decode("utf-8", errors="ignore")


def _docker(*args: str) -> str | None:
    try:
        saida = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=60, check=True
        )
        return saida.stdout.strip()
    except Exception:
        return None


def memoria_container(nome: str) -> str | None:
    return _docker("stats", "--no-stream", "--format", "{{.MemUsage}}", nome)


def disco_container(nome: str, caminho: str) -> int | None:
    saida = _docker("exec", nome, "du", "-sb", caminho)
    try:
        return int(saida.split()[0]) if saida else None
    except ValueError:
        return None


def info_ambiente(cfg: Config, conexao_mysql=None, cliente_milvus=None) -> dict:
    import pymilvus

    info: dict[str, Any] = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "perfil": cfg.perfil.nome,
        "so": platform.platform(),
        "processador": platform.processor(),
        "python": platform.python_version(),
        "pymilvus": pymilvus.__version__,
        "milvus_modo": cfg.milvus_modo,
        "modelo_embedding": MODELO_EMBEDDING,
        "dimensao": DIMENSAO,
    }
    try:
        saida = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name;"
                "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.split("\n")
        info["processador"] = saida[0].strip() or info["processador"]
        info["ram_gb"] = saida[1].strip()
    except Exception:
        pass
    if conexao_mysql is not None:
        with conexao_mysql.cursor() as cur:
            cur.execute("SELECT VERSION()")
            info["mysql_versao"] = cur.fetchone()[0]
    if cliente_milvus is not None:
        try:
            info["milvus_versao"] = cliente_milvus.get_server_version()
        except Exception:
            info["milvus_versao"] = "indisponivel"
    return info


def salvar_json(caminho: Path, dados: Any) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def ler_json(caminho: Path) -> Any:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def carregar_dados_preparados(cfg: Config):
    import pandas as pd

    filmes = pd.read_parquet(cfg.dir_dados / "filmes.parquet")
    elenco = pd.read_parquet(cfg.dir_dados / "elenco.parquet")
    vet_filmes = np.load(cfg.dir_dados / "filmes_vetores.npy")
    vet_elenco = np.load(cfg.dir_dados / "elenco_vetores.npy")
    assert len(filmes) == len(vet_filmes) and len(elenco) == len(vet_elenco)
    return filmes, elenco, vet_filmes, vet_elenco
