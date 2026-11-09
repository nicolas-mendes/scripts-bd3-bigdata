from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DATASET_HF = "gaygaaa/TheMoviesDataset"
ARQUIVOS_BRUTOS = ("movies_metadata.csv", "credits.csv", "keywords.csv")
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
DIMENSAO = 384
SEMENTE = 42


@dataclass(frozen=True)
class Perfil:
    nome: str
    limite_filmes: int | None
    elenco_top_n: int
    repeticoes: int
    aquecimento: int
    lote_insercao: int
    min_filmes_ator: int


PERFIS: dict[str, Perfil] = {
    "dev": Perfil("dev", 2_000, 5, 3, 1, 500, 8),
    "final": Perfil("final", None, 5, 5, 1, 1_000, 10),
}


@dataclass(frozen=True)
class Config:
    perfil: Perfil

    @property
    def dir_brutos(self) -> Path:
        return RAIZ / "dados" / "brutos"

    @property
    def dir_dados(self) -> Path:
        return RAIZ / "dados" / self.perfil.nome

    @property
    def dir_resultados(self) -> Path:
        return RAIZ / "resultados" / self.perfil.nome

    @property
    def mysql_host(self) -> str:
        return os.getenv("MYSQL_HOST", "127.0.0.1")

    @property
    def mysql_porta(self) -> int:
        return int(os.getenv("MYSQL_PORT", "3306"))

    @property
    def mysql_usuario(self) -> str:
        return os.getenv("MYSQL_USER", "root")

    @property
    def mysql_senha(self) -> str:
        return os.getenv("MYSQL_PASSWORD", "bd3senha")

    @property
    def mysql_banco(self) -> str:
        return f"bd3_tmdb_{self.perfil.nome}"

    @property
    def mysql_container(self) -> str:
        return os.getenv("MYSQL_CONTAINER", "bd3-mysql")

    @property
    def milvus_modo(self) -> str:
        return os.getenv("MILVUS_MODO", "zilliz").lower()

    @property
    def milvus_uri(self) -> str:
        if self.milvus_modo == "local":
            return os.getenv("MILVUS_URI_LOCAL", "http://localhost:19530")
        return os.environ["URI_NUVEM"]

    @property
    def milvus_token(self) -> str | None:
        if self.milvus_modo == "local":
            return os.getenv("MILVUS_TOKEN_LOCAL") or None
        return os.environ["TOKEN"]

    @property
    def milvus_container(self) -> str:
        return os.getenv("MILVUS_CONTAINER", "milvus-standalone")

    @property
    def milvus_consistencia(self) -> str:
        return os.getenv("MILVUS_CONSISTENCIA", "Bounded")

    @property
    def colecao_filmes(self) -> str:
        return f"tmdb_filmes_{self.perfil.nome}"

    @property
    def colecao_elenco(self) -> str:
        return f"tmdb_elenco_{self.perfil.nome}"

    @property
    def indice_vetorial(self) -> dict:
        if self.milvus_modo == "local":
            return {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
        return {"index_type": "AUTOINDEX", "params": {}}

    @property
    def parametros_busca(self) -> dict:
        if self.milvus_modo == "local":
            return {"metric_type": "COSINE", "params": {"ef": 64}}
        return {"metric_type": "COSINE"}

    @property
    def tipo_indice_escalar(self) -> str:
        return "INVERTED" if self.milvus_modo == "local" else "AUTOINDEX"


def criar_parser(descricao: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument(
        "--modo",
        choices=sorted(PERFIS),
        default=os.getenv("MODO", "dev"),
    )
    return parser


def carregar_config(args: argparse.Namespace) -> Config:
    cfg = Config(PERFIS[args.modo])
    cfg.dir_dados.mkdir(parents=True, exist_ok=True)
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    return cfg
