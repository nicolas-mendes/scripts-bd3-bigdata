from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from comum import blobs_para_matriz, vetorizar
from config import SEMENTE, Config

LIMITE_QUERY = 16_384
LOTE_ITERADOR = 8_000
TEXTO_SEMANTICO = "Um herói mascarado combatendo o crime em uma cidade corrompida"
TEXTO_LEXICO_EN = "masked hero fighting crime in a corrupt city"
TEXTO_HIBRIDO = "Uma missão no espaço para salvar a humanidade da extinção"
CAMPOS_LEITURA = ["id", "titulo", "ano", "nota", "votos", "diretor"]


@dataclass
class Contexto:
    cfg: Config
    mysql: Any
    milvus: Any
    p: dict

    @property
    def cf(self) -> str:
        return self.cfg.colecao_filmes

    @property
    def ce(self) -> str:
        return self.cfg.colecao_elenco


def preparar_parametros(cfg: Config, conexao_mysql) -> dict:
    with conexao_mysql.cursor() as cur:
        cur.execute("SELECT id FROM filmes ORDER BY id")
        ids = [r[0] for r in cur.fetchall()]
    vetores = vetorizar([TEXTO_SEMANTICO, TEXTO_HIBRIDO])
    return {
        "ids_amostra": sorted(random.Random(SEMENTE).sample(ids, min(100, len(ids)))),
        "vetor_semantico": vetores[0],
        "vetor_hibrido": vetores[1],
        "texto_lexico": TEXTO_LEXICO_EN,
        "termo_titulo": "Love",
        "ator_parcial": "Hanks",
        "ator_exato": "Tom Hanks",
        "genero_hibrido": "Science Fiction",
        "min_filmes_ator": cfg.perfil.min_filmes_ator,
        "top_k": 10,
    }


def sql_todos(ctx: Contexto, sql: str, params: tuple | list = ()) -> list[tuple]:
    with ctx.mysql.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def milvus_iterar(ctx: Contexto, colecao: str, filtro: str, campos: list[str]) -> list[dict]:
    it = ctx.milvus.query_iterator(
        collection_name=colecao,
        batch_size=LOTE_ITERADOR,
        filter=filtro,
        output_fields=campos,
    )
    linhas: list[dict] = []
    try:
        while lote := it.next():
            linhas.extend(lote)
    finally:
        it.close()
    return linhas


def lista_ids(ids) -> str:
    return "[" + ",".join(str(int(i)) for i in ids) + "]"


def marcadores(n: int) -> str:
    return ",".join(["%s"] * n)


def top_k_exato(ids: np.ndarray, matriz: np.ndarray, consulta: np.ndarray, k: int):
    if len(ids) == 0:
        return []
    scores = matriz @ consulta
    k = min(k, len(ids))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(ids[i]), float(scores[i])) for i in idx]


def titulos_mysql(ctx: Contexto, ranking: list[tuple[int, float]]) -> list[tuple]:
    if not ranking:
        return []
    ids = [i for i, _ in ranking]
    nomes = dict(
        sql_todos(
            ctx,
            f"SELECT id, titulo FROM filmes WHERE id IN ({marcadores(len(ids))})",
            ids,
        )
    )
    return [(i, nomes[i], round(s, 4)) for i, s in ranking]


def c01_mysql(ctx: Contexto):
    ids = ctx.p["ids_amostra"]
    linhas = sql_todos(
        ctx,
        f"SELECT id, titulo, ano, nota, votos, diretor FROM filmes "
        f"WHERE id IN ({marcadores(len(ids))}) ORDER BY id",
        ids,
    )
    return [(r[0], r[1], r[2], float(r[3]), r[4], r[5]) for r in linhas]


def c01_milvus(ctx: Contexto):
    linhas = ctx.milvus.get(
        collection_name=ctx.cf,
        ids=ctx.p["ids_amostra"],
        output_fields=CAMPOS_LEITURA,
    )
    return sorted(
        (r["id"], r["titulo"], r["ano"], round(r["nota"], 1), r["votos"], r["diretor"])
        for r in linhas
    )


def _mysql_ajustar_votos(ctx: Contexto, delta: int):
    ids = ctx.p["ids_amostra"]
    with ctx.mysql.cursor() as cur:
        afetadas = cur.execute(
            f"UPDATE filmes SET votos = votos + %s WHERE id IN ({marcadores(len(ids))})",
            [delta, *ids],
        )
    ctx.mysql.commit()
    return afetadas


def _milvus_ajustar_votos(ctx: Contexto, delta: int):
    entidades = [
        dict(e)
        for e in ctx.milvus.query(
            collection_name=ctx.cf,
            ids=ctx.p["ids_amostra"],
            output_fields=["*"],
            consistency_level="Strong",
        )
    ]
    for e in entidades:
        e["votos"] = int(e["votos"]) + delta
        e["vetor"] = [float(x) for x in e["vetor"]]
    ctx.milvus.upsert(collection_name=ctx.cf, data=entidades)
    return len(entidades)


def c02_mysql(ctx: Contexto):
    return _mysql_ajustar_votos(ctx, +1)


def c02_milvus(ctx: Contexto):
    return _milvus_ajustar_votos(ctx, +1)


def c02_mysql_desfaz(ctx: Contexto):
    return _mysql_ajustar_votos(ctx, -1)


def c02_milvus_desfaz(ctx: Contexto):
    return _milvus_ajustar_votos(ctx, -1)


def c03_mysql(ctx: Contexto):
    linhas = sql_todos(
        ctx,
        "SELECT id, titulo, nota, votos FROM filmes "
        "WHERE nota >= 8 AND votos >= 500 AND ano BETWEEN 1990 AND 2010 "
        "ORDER BY nota DESC, votos DESC, id LIMIT 100",
    )
    return [(r[0], r[1], float(r[2]), r[3]) for r in linhas]


def c03_milvus(ctx: Contexto):
    linhas = ctx.milvus.query(
        collection_name=ctx.cf,
        filter="nota >= 8 and votos >= 500 and ano >= 1990 and ano <= 2010",
        output_fields=["id", "titulo", "nota", "votos"],
        limit=LIMITE_QUERY,
    )
    ordenado = sorted(linhas, key=lambda r: (-round(r["nota"], 1), -r["votos"], r["id"]))[:100]
    return [(r["id"], r["titulo"], round(r["nota"], 1), r["votos"]) for r in ordenado]


def c04_mysql(ctx: Contexto):
    return sql_todos(
        ctx,
        "SELECT id, titulo FROM filmes WHERE titulo LIKE %s ORDER BY id",
        (f"%{ctx.p['termo_titulo']}%",),
    )


def c04_milvus(ctx: Contexto):
    linhas = ctx.milvus.query(
        collection_name=ctx.cf,
        filter=f'titulo like "%{ctx.p["termo_titulo"]}%"',
        output_fields=["id", "titulo"],
        limit=LIMITE_QUERY,
    )
    return sorted((r["id"], r["titulo"]) for r in linhas)


def c05_mysql(ctx: Contexto):
    linhas = sql_todos(
        ctx,
        "SELECT ano, COUNT(*), AVG(nota), MAX(nota), MIN(nota) FROM filmes "
        "WHERE votos >= 10 AND ano BETWEEN 1980 AND 2017 "
        "GROUP BY ano ORDER BY ano",
    )
    return [(r[0], r[1], round(float(r[2]), 3), float(r[3]), float(r[4])) for r in linhas]


def c05_milvus(ctx: Contexto):
    linhas = milvus_iterar(
        ctx,
        ctx.cf,
        "votos >= 10 and ano >= 1980 and ano <= 2017",
        ["ano", "nota"],
    )
    if not linhas:
        return []
    df = pd.DataFrame(linhas)
    df["nota"] = df["nota"].astype("float64").round(1)
    g = df.groupby("ano")["nota"].agg(["count", "mean", "max", "min"]).sort_index()
    return [
        (int(a), int(r["count"]), round(float(r["mean"]), 3), float(r["max"]), float(r["min"]))
        for a, r in g.iterrows()
    ]


def c06_mysql(ctx: Contexto):
    return sql_todos(
        ctx,
        "SELECT f.id, f.titulo, f.ano, e.ator, e.personagem "
        "FROM elenco e JOIN filmes f ON f.id = e.filme_id "
        "WHERE e.ator LIKE %s ORDER BY f.ano, f.id, e.ator",
        (f"%{ctx.p['ator_parcial']}%",),
    )


def c06_milvus(ctx: Contexto):
    elenco = ctx.milvus.query(
        collection_name=ctx.ce,
        filter=f'ator like "%{ctx.p["ator_parcial"]}%"',
        output_fields=["filme_id", "ator", "personagem"],
        limit=LIMITE_QUERY,
    )
    if not elenco:
        return []
    ids = sorted({e["filme_id"] for e in elenco})
    filmes = {
        f["id"]: f
        for f in ctx.milvus.query(
            collection_name=ctx.cf,
            filter=f"id in {lista_ids(ids)}",
            output_fields=["id", "titulo", "ano"],
            limit=LIMITE_QUERY,
        )
    }
    linhas = [
        (f["id"], f["titulo"], f["ano"], e["ator"], e["personagem"])
        for e in elenco
        if (f := filmes.get(e["filme_id"]))
    ]
    return sorted(linhas, key=lambda r: (r[2], r[0], r[3]))


def c07_mysql(ctx: Contexto):
    return sql_todos(
        ctx,
        "SELECT f.id, f.titulo, f.ano FROM elenco e "
        "JOIN filmes f ON f.id = e.filme_id "
        "WHERE e.ator = %s ORDER BY f.id",
        (ctx.p["ator_exato"],),
    )


def c07_milvus(ctx: Contexto):
    linhas = ctx.milvus.query(
        collection_name=ctx.cf,
        filter=f'array_contains(elenco_principal, "{ctx.p["ator_exato"]}")',
        output_fields=["id", "titulo", "ano"],
        limit=LIMITE_QUERY,
    )
    return sorted((r["id"], r["titulo"], r["ano"]) for r in linhas)


def c08_mysql(ctx: Contexto):
    linhas = sql_todos(
        ctx,
        "SELECT e.ator, COUNT(*) AS filmes, AVG(f.nota) AS media, MAX(f.nota) "
        "FROM elenco e JOIN filmes f ON f.id = e.filme_id "
        "WHERE f.votos >= 50 GROUP BY e.ator HAVING COUNT(*) >= %s "
        "ORDER BY media DESC, e.ator LIMIT 10",
        (ctx.p["min_filmes_ator"],),
    )
    return [(r[0], r[1], round(float(r[2]), 3), float(r[3])) for r in linhas]


def c08_milvus(ctx: Contexto):
    filmes = milvus_iterar(ctx, ctx.cf, "votos >= 50", ["id", "nota"])
    elenco = milvus_iterar(ctx, ctx.ce, "id >= 0", ["filme_id", "ator"])
    if not filmes or not elenco:
        return []
    df = pd.DataFrame(elenco).merge(pd.DataFrame(filmes), left_on="filme_id", right_on="id")
    df["nota"] = df["nota"].astype("float64").round(1)
    g = df.groupby("ator")["nota"].agg(["count", "mean", "max"])
    g = g[g["count"] >= ctx.p["min_filmes_ator"]].reset_index()
    g["media_r"] = g["mean"].round(6)
    g = g.sort_values(["media_r", "ator"], ascending=[False, True]).head(10)
    return [
        (r.ator, int(r["count"]), round(float(r["mean"]), 3), float(r["max"]))
        for _, r in g.iterrows()
    ]


def c09_mysql(ctx: Contexto):
    linhas = sql_todos(
        ctx,
        "SELECT g.nome, COUNT(*), AVG(f.nota), MAX(f.nota), MIN(f.nota) "
        "FROM generos g JOIN filme_genero fg ON fg.genero_id = g.id "
        "JOIN filmes f ON f.id = fg.filme_id WHERE f.votos >= 10 "
        "GROUP BY g.nome ORDER BY g.nome",
    )
    return [(r[0], r[1], round(float(r[2]), 3), float(r[3]), float(r[4])) for r in linhas]


def c09_milvus(ctx: Contexto):
    linhas = milvus_iterar(ctx, ctx.cf, "votos >= 10", ["generos", "nota"])
    acc: dict[str, list[float]] = defaultdict(list)
    for r in linhas:
        for g in set(r["generos"]):
            acc[g].append(round(float(r["nota"]), 1))
    return [(g, len(v), round(sum(v) / len(v), 3), max(v), min(v)) for g, v in sorted(acc.items())]


def c10_mysql(ctx: Contexto):
    linhas = sql_todos(ctx, "SELECT filme_id, vetor FROM filmes_vetores")
    ids = np.fromiter((r[0] for r in linhas), dtype=np.int64, count=len(linhas))
    matriz = blobs_para_matriz([r[1] for r in linhas])
    return titulos_mysql(ctx, top_k_exato(ids, matriz, ctx.p["vetor_semantico"], ctx.p["top_k"]))


def c10_milvus(ctx: Contexto):
    res = ctx.milvus.search(
        collection_name=ctx.cf,
        data=[ctx.p["vetor_semantico"].tolist()],
        anns_field="vetor",
        limit=ctx.p["top_k"],
        output_fields=["titulo"],
        search_params=ctx.cfg.parametros_busca,
    )
    return [(h["id"], h["entity"]["titulo"], round(float(h["distance"]), 4)) for h in res[0]]


def c11_mysql(ctx: Contexto):
    linhas = sql_todos(
        ctx,
        "SELECT v.filme_id, v.vetor FROM filmes_vetores v "
        "JOIN filmes f ON f.id = v.filme_id "
        "JOIN filme_genero fg ON fg.filme_id = f.id "
        "JOIN generos g ON g.id = fg.genero_id "
        "WHERE f.nota >= 7 AND f.ano >= 2000 AND g.nome = %s",
        (ctx.p["genero_hibrido"],),
    )
    ids = np.fromiter((r[0] for r in linhas), dtype=np.int64, count=len(linhas))
    matriz = blobs_para_matriz([r[1] for r in linhas]) if linhas else np.empty((0, 0))
    return titulos_mysql(ctx, top_k_exato(ids, matriz, ctx.p["vetor_hibrido"], ctx.p["top_k"]))


def c11_milvus(ctx: Contexto):
    filtro = (
        f"nota >= 7 and ano >= 2000 and "
        f'array_contains(generos, "{ctx.p["genero_hibrido"]}")'
    )
    res = ctx.milvus.search(
        collection_name=ctx.cf,
        data=[ctx.p["vetor_hibrido"].tolist()],
        anns_field="vetor",
        filter=filtro,
        limit=ctx.p["top_k"],
        output_fields=["titulo"],
        search_params=ctx.cfg.parametros_busca,
    )
    return [(h["id"], h["entity"]["titulo"], round(float(h["distance"]), 4)) for h in res[0]]


def c12_mysql(ctx: Contexto):
    linhas = sql_todos(
        ctx,
        "SELECT id, titulo, MATCH(sinopse) AGAINST (%s IN NATURAL LANGUAGE MODE) AS s "
        "FROM filmes WHERE MATCH(sinopse) AGAINST (%s IN NATURAL LANGUAGE MODE) "
        "ORDER BY s DESC LIMIT %s",
        (ctx.p["texto_lexico"], ctx.p["texto_lexico"], ctx.p["top_k"]),
    )
    return [(r[0], r[1], round(float(r[2]), 4)) for r in linhas]


c12_milvus = c10_milvus


@dataclass
class Consulta:
    codigo: str
    titulo: str
    categoria: str
    sql: str
    milvus: str
    mysql_fn: Callable[[Contexto], Any]
    milvus_fn: Callable[[Contexto], Any]
    validacao: str = "igual"
    mysql_desfaz: Callable[[Contexto], Any] | None = None
    milvus_desfaz: Callable[[Contexto], Any] | None = None


CONSULTAS: list[Consulta] = [
    Consulta(
        "C01",
        "Leitura PK (100 registros)",
        "Leitura x Atualizacao",
        "SELECT id, titulo, ano, nota, votos, diretor FROM filmes WHERE id IN (...100 ids)",
        "client.get(collection, ids=[...100 ids], output_fields=[...])",
        c01_mysql,
        c01_milvus,
    ),
    Consulta(
        "C02",
        "Atualizacao em lote (100 registros)",
        "Leitura x Atualizacao",
        "UPDATE filmes SET votos = votos + 1 WHERE id IN (...100 ids)",
        "client.query(ids=[...], output_fields=['*']) -> client.upsert(data)",
        c02_mysql,
        c02_milvus,
        "nenhuma",
        c02_mysql_desfaz,
        c02_milvus_desfaz,
    ),
    Consulta(
        "C03",
        "Filtro condicional + ordenacao",
        "Uma tabela",
        "SELECT id, titulo, nota, votos FROM filmes WHERE nota >= 8 AND votos >= 500 "
        "AND ano BETWEEN 1990 AND 2010 ORDER BY nota DESC, votos DESC, id LIMIT 100",
        'client.query(filter="nota >= 8 and votos >= 500 and ano >= 1990 and ano <= 2010")',
        c03_mysql,
        c03_milvus,
    ),
    Consulta(
        "C04",
        "Busca textual no titulo (LIKE)",
        "Uma tabela",
        "SELECT id, titulo FROM filmes WHERE titulo LIKE '%Love%'",
        "client.query(filter='titulo like \"%Love%\"')",
        c04_mysql,
        c04_milvus,
    ),
    Consulta(
        "C05",
        "Agregacao por ano (COUNT/AVG/MAX/MIN)",
        "Uma tabela",
        "SELECT ano, COUNT(*), AVG(nota), MAX(nota), MIN(nota) FROM filmes WHERE votos >= 10 "
        "AND ano BETWEEN 1980 AND 2017 GROUP BY ano ORDER BY ano",
        "client.query_iterator(filter=...) + groupby",
        c05_mysql,
        c05_milvus,
    ),
    Consulta(
        "C06",
        "Busca de ator com JOIN",
        "Duas tabelas",
        "SELECT f.id, f.titulo, f.ano, e.ator, e.personagem FROM elenco e JOIN filmes f "
        "ON f.id = e.filme_id WHERE e.ator LIKE '%Hanks%' ORDER BY f.ano",
        "query(elenco, 'ator like \"%Hanks%\"') -> query(filmes, 'id in [...]')",
        c06_mysql,
        c06_milvus,
    ),
    Consulta(
        "C07",
        "Filmes por ator (JOIN x ARRAY)",
        "Duas tabelas",
        "SELECT f.id, f.titulo, f.ano FROM elenco e JOIN filmes f ON f.id = e.filme_id "
        "WHERE e.ator = 'Tom Hanks'",
        "client.query(filter='array_contains(elenco_principal, \"Tom Hanks\")')",
        c07_mysql,
        c07_milvus,
    ),
    Consulta(
        "C08",
        "Top-10 atores por media (JOIN + GROUP BY)",
        "Duas tabelas",
        "SELECT e.ator, COUNT(*), AVG(f.nota), MAX(f.nota) FROM elenco e JOIN filmes f "
        "ON f.id = e.filme_id WHERE f.votos >= 50 GROUP BY e.ator HAVING COUNT(*) >= N "
        "ORDER BY AVG(f.nota) DESC LIMIT 10",
        "query_iterator(filmes) + query_iterator(elenco) + merge/groupby",
        c08_mysql,
        c08_milvus,
    ),
    Consulta(
        "C09",
        "Agregacao por genero (N:N x ARRAY)",
        "Duas tabelas",
        "SELECT g.nome, COUNT(*), AVG(f.nota), MAX(f.nota), MIN(f.nota) FROM generos g "
        "JOIN filme_genero fg ON ... JOIN filmes f ON ... WHERE f.votos >= 10 GROUP BY g.nome",
        "query_iterator(filmes, output_fields=['generos', 'nota']) + groupby",
        c09_mysql,
        c09_milvus,
    ),
    Consulta(
        "C10",
        "Busca semantica kNN top-10",
        "Vetorial",
        "SELECT filme_id, vetor FROM filmes_vetores + cosseno",
        "client.search(data=[vetor], anns_field='vetor', limit=10)",
        c10_mysql,
        c10_milvus,
        "recall",
    ),
    Consulta(
        "C11",
        "Busca hibrida (vetor + nota + ano + genero)",
        "Vetorial",
        "SELECT v.vetor FROM filmes_vetores v JOIN filmes f ... WHERE f.nota >= 7 "
        "AND f.ano >= 2000 AND g.nome = 'Science Fiction' + cosseno",
        "client.search(data=[vetor], filter='nota >= 7 and ano >= 2000 and "
        "array_contains(generos, \"Science Fiction\")', limit=10)",
        c11_mysql,
        c11_milvus,
        "recall",
    ),
    Consulta(
        "C12",
        "Busca FULLTEXT x semantica",
        "Vetorial",
        "SELECT id, titulo FROM filmes WHERE MATCH(sinopse) AGAINST (...) LIMIT 10",
        "client.search(data=[vetor], limit=10)",
        c12_mysql,
        c12_milvus,
        "qualitativa",
    ),
]


def _normalizar(valor):
    if hasattr(valor, "item") and not isinstance(valor, (str, bytes)):
        try:
            valor = valor.item()
        except Exception:
            pass
    if isinstance(valor, (tuple, list)):
        return tuple(_normalizar(v) for v in valor)
    return valor


def _equivalentes(a, b, tolerancia: float = 2e-3) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_equivalentes(x, y, tolerancia) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tolerancia
    return a == b


def respostas_iguais(a, b) -> bool:
    return _equivalentes(_normalizar(list(a)), _normalizar(list(b)))


def recall(exato: list[tuple], aproximado: list[tuple]) -> float:
    referencia = {r[0] for r in exato}
    if not referencia:
        return 1.0
    return len(referencia & {r[0] for r in aproximado}) / len(referencia)
