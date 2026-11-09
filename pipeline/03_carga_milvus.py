from __future__ import annotations

import time

import numpy as np
from pymilvus import DataType, MilvusClient

from comum import (
    carregar_dados_preparados,
    conectar_milvus,
    cronometro,
    info_ambiente,
    log,
    salvar_json,
    truncar_bytes,
)
from config import DIMENSAO, Config, carregar_config, criar_parser

LIM = {
    "titulo": 1024,
    "sinopse": 8192,
    "idioma": 16,
    "diretor": 512,
    "item": 256,
    "ator": 512,
    "personagem": 1024,
}
INDICES_ESCALARES = {
    "filmes": ["ano", "nota", "votos", "diretor", "generos", "elenco_principal"],
    "elenco": ["filme_id", "ator"],
}
LIMITE_COLECOES_ZILLIZ = 5


def esquema_filmes():
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("vetor", DataType.FLOAT_VECTOR, dim=DIMENSAO)
    s.add_field("titulo", DataType.VARCHAR, max_length=LIM["titulo"])
    s.add_field("sinopse", DataType.VARCHAR, max_length=LIM["sinopse"])
    s.add_field("ano", DataType.INT32)
    s.add_field("nota", DataType.FLOAT)
    s.add_field("votos", DataType.INT32)
    s.add_field("popularidade", DataType.FLOAT)
    s.add_field("orcamento", DataType.INT64)
    s.add_field("receita", DataType.INT64)
    s.add_field("duracao", DataType.INT32)
    s.add_field("idioma", DataType.VARCHAR, max_length=LIM["idioma"])
    s.add_field("diretor", DataType.VARCHAR, max_length=LIM["diretor"])
    s.add_field(
        "generos",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=32,
        max_length=LIM["item"],
    )
    s.add_field(
        "palavras_chave",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=32,
        max_length=LIM["item"],
    )
    s.add_field(
        "elenco_principal",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=16,
        max_length=LIM["ator"],
    )
    return s


def esquema_elenco():
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("vetor", DataType.FLOAT_VECTOR, dim=DIMENSAO)
    s.add_field("filme_id", DataType.INT64)
    s.add_field("ordem", DataType.INT32)
    s.add_field("ator", DataType.VARCHAR, max_length=LIM["ator"])
    s.add_field("personagem", DataType.VARCHAR, max_length=LIM["personagem"])
    return s


def lista(valores, limite_itens: int, limite_bytes: int) -> list[str]:
    return [truncar_bytes(str(v), limite_bytes) for v in list(valores)[:limite_itens]]


def entidades_filmes(filmes, vetores: np.ndarray) -> list[dict]:
    return [
        {
            "id": int(f.id),
            "vetor": vetores[i].tolist(),
            "titulo": truncar_bytes(f.titulo, LIM["titulo"]),
            "sinopse": truncar_bytes(f.sinopse, LIM["sinopse"]),
            "ano": int(f.ano),
            "nota": float(round(float(f.nota), 1)),
            "votos": int(f.votos),
            "popularidade": float(f.popularidade),
            "orcamento": int(f.orcamento),
            "receita": int(f.receita),
            "duracao": int(f.duracao),
            "idioma": truncar_bytes(f.idioma, LIM["idioma"]),
            "diretor": truncar_bytes(f.diretor, LIM["diretor"]),
            "generos": lista(f.generos, 32, LIM["item"]),
            "palavras_chave": lista(f.palavras_chave, 32, LIM["item"]),
            "elenco_principal": lista(f.elenco_principal, 16, LIM["ator"]),
        }
        for i, f in enumerate(filmes.itertuples())
    ]


def entidades_elenco(elenco, vetores: np.ndarray) -> list[dict]:
    return [
        {
            "id": int(e.id),
            "vetor": vetores[i].tolist(),
            "filme_id": int(e.filme_id),
            "ordem": int(e.ordem),
            "ator": truncar_bytes(e.ator, LIM["ator"]),
            "personagem": truncar_bytes(e.personagem, LIM["personagem"]),
        }
        for i, e in enumerate(elenco.itertuples())
    ]


def remover_colecao_seguro(cliente, nome: str) -> None:
    if not cliente.has_collection(nome):
        return
    log(f"DROP {nome}")
    try:
        cliente.drop_collection(nome, timeout=15)
    except Exception:
        pass
    for _ in range(30):
        if not cliente.has_collection(nome):
            return
        time.sleep(1)


def limpar_colecoes_excedentes(cliente, necessarias: set[str]) -> None:
    existentes = set(cliente.list_collections())
    outros = sorted(existentes - necessarias)
    while len(outros) + len(necessarias) > LIMITE_COLECOES_ZILLIZ and outros:
        remover_colecao_seguro(cliente, outros.pop(0))


def recriar_colecao(cliente, cfg: Config, nome: str, esquema) -> None:
    if cliente.has_collection(nome):
        remover_colecao_seguro(cliente, nome)
    cliente.create_collection(
        collection_name=nome,
        schema=esquema,
        consistency_level=cfg.milvus_consistencia,
    )


def inserir_lotes(cliente, nome: str, entidades: list[dict], lote: int) -> float:
    with cronometro() as t:
        for i in range(0, len(entidades), lote):
            cliente.insert(collection_name=nome, data=entidades[i : i + lote])
        try:
            cliente.flush(nome)
        except Exception:
            pass
    return t()


def aguardar_indice(cliente, nome: str, indice: str, limite_s: int = 1800) -> None:
    inicio = time.time()
    while time.time() - inicio < limite_s:
        try:
            info = cliente.describe_index(nome, index_name=indice) or {}
        except Exception:
            return
        pendentes = info.get("pending_index_rows")
        estado = str(info.get("state", "")).lower()
        if pendentes in (0, None) and estado in ("finished", "", "none"):
            return
        time.sleep(1)


def indexar_e_carregar(cliente, cfg: Config, nome: str, escalares: list[str]) -> dict:
    tempos = {}
    with cronometro() as t:
        params = cliente.prepare_index_params()
        params.add_index(
            field_name="vetor",
            index_name="idx_vetor",
            metric_type="COSINE",
            **cfg.indice_vetorial,
        )
        cliente.create_index(nome, params)
        aguardar_indice(cliente, nome, "idx_vetor")
    tempos["indice_vetorial_s"] = round(t(), 3)

    with cronometro() as t:
        for campo in escalares:
            params = cliente.prepare_index_params()
            params.add_index(
                field_name=campo,
                index_name=f"idx_{campo}",
                index_type=cfg.tipo_indice_escalar,
            )
            try:
                cliente.create_index(nome, params)
            except Exception:
                pass
    tempos["indices_escalares_s"] = round(t(), 3)

    with cronometro() as t:
        cliente.load_collection(nome)
    tempos["load_s"] = round(t(), 3)
    return tempos


def main() -> None:
    parser = criar_parser()
    cfg = carregar_config(parser.parse_args())
    lote = cfg.perfil.lote_insercao

    filmes, elenco, vet_filmes, vet_elenco = carregar_dados_preparados(cfg)
    cliente = conectar_milvus(cfg)
    log(f"Milvus ({cfg.milvus_modo}): {cfg.colecao_filmes}, {cfg.colecao_elenco}")

    plano = [
        ("filmes", cfg.colecao_filmes, esquema_filmes(), entidades_filmes(filmes, vet_filmes)),
        ("elenco", cfg.colecao_elenco, esquema_elenco(), entidades_elenco(elenco, vet_elenco)),
    ]
    limpar_colecoes_excedentes(cliente, {cfg.colecao_filmes, cfg.colecao_elenco})

    resultado: dict = {
        "ambiente": info_ambiente(cfg, cliente_milvus=cliente),
        "linhas": {},
        "insercao_s": {},
        "indexacao": {},
        "estatisticas": {},
    }
    for chave, nome, esquema, entidades in plano:
        recriar_colecao(cliente, cfg, nome, esquema)
        tempo = round(inserir_lotes(cliente, nome, entidades, lote), 3)
        resultado["linhas"][chave] = len(entidades)
        resultado["insercao_s"][chave] = tempo
        log(f"INSERT {nome}: {len(entidades)} ({tempo:.2f}s)")

        resultado["indexacao"][chave] = indexar_e_carregar(
            cliente, cfg, nome, INDICES_ESCALARES[chave]
        )
        log(f"INDEX {nome}: {resultado['indexacao'][chave]}")
        resultado["estatisticas"][chave] = cliente.get_collection_stats(nome)

    resultado["insercao_total_s"] = round(sum(resultado["insercao_s"].values()), 3)
    resultado["indexacao_total_s"] = round(
        sum(v for d in resultado["indexacao"].values() for v in d.values()), 3
    )
    resultado["vetores_bytes_brutos"] = int(vet_filmes.nbytes + vet_elenco.nbytes)
    salvar_json(cfg.dir_resultados / "carga_milvus.json", resultado)
    log(
        f"Milvus concluido: insert={resultado['insercao_total_s']:.2f}s "
        f"index={resultado['indexacao_total_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
