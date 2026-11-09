from __future__ import annotations

import sys

import numpy as np

from comum import carregar_dados_preparados, conectar_milvus, conectar_mysql, log, salvar_json
from config import carregar_config, criar_parser
from consultas import CONSULTAS, Contexto, preparar_parametros, recall, respostas_iguais


def checar(nome: str, ok: bool, detalhe: str, relatorio: list) -> None:
    relatorio.append({"verificacao": nome, "ok": bool(ok), "detalhe": detalhe})
    log(f"[{'OK' if ok else 'FALHA'}] {nome}: {detalhe}")


def c01_votos(ctx: Contexto) -> dict:
    ids = ctx.p["ids_amostra"]
    with ctx.mysql.cursor() as cur:
        cur.execute(
            f"SELECT votos FROM filmes WHERE id IN ({','.join(['%s'] * len(ids))}) ORDER BY id",
            ids,
        )
        mysql = [r[0] for r in cur.fetchall()]
    milvus = [
        r["votos"]
        for r in sorted(
            ctx.milvus.query(
                collection_name=ctx.cf,
                ids=ids,
                output_fields=["votos"],
                consistency_level="Strong",
            ),
            key=lambda r: r["id"],
        )
    ]
    return {"mysql": mysql, "milvus": milvus}


def main() -> None:
    parser = criar_parser()
    cfg = carregar_config(parser.parse_args())
    filmes, elenco, vet_filmes, _ = carregar_dados_preparados(cfg)
    con = conectar_mysql(cfg)
    cli = conectar_milvus(cfg)
    ctx = Contexto(cfg, con, cli, preparar_parametros(cfg, con))
    rel: list[dict] = []
    forte = {"consistency_level": "Strong"}

    for tabela, colecao, esperado in [
        ("filmes", cfg.colecao_filmes, len(filmes)),
        ("elenco", cfg.colecao_elenco, len(elenco)),
    ]:
        with con.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {tabela}")
            n_mysql = cur.fetchone()[0]
        n_milvus = cli.query(
            collection_name=colecao,
            filter="",
            output_fields=["count(*)"],
            **forte,
        )[0]["count(*)"]
        checar(
            f"COUNT {tabela}",
            n_mysql == n_milvus == esperado,
            f"esperado={esperado} mysql={n_mysql} milvus={n_milvus}",
            rel,
        )

    amostra = ctx.p["ids_amostra"][:20]
    with con.cursor() as cur:
        cur.execute(
            f"SELECT id, titulo, ano, nota, diretor FROM filmes "
            f"WHERE id IN ({','.join(['%s'] * len(amostra))}) ORDER BY id",
            amostra,
        )
        lado_mysql = [(r[0], r[1], r[2], float(r[3]), r[4]) for r in cur.fetchall()]
    lado_milvus = sorted(
        (r["id"], r["titulo"], r["ano"], round(r["nota"], 1), r["diretor"])
        for r in cli.get(
            collection_name=cfg.colecao_filmes,
            ids=amostra,
            output_fields=["titulo", "ano", "nota", "diretor"],
        )
    )
    checar(
        "AMOSTRA 20 filmes",
        respostas_iguais(lado_mysql, lado_milvus),
        f"{len(lado_mysql)}=={len(lado_milvus)}",
        rel,
    )

    rng = np.random.default_rng(7)
    posicoes = rng.choice(len(filmes), size=min(10, len(filmes)), replace=False)
    res = cli.search(
        collection_name=cfg.colecao_filmes,
        data=[vet_filmes[i].tolist() for i in posicoes],
        anns_field="vetor",
        limit=1,
        search_params=cfg.parametros_busca,
    )
    acertos = sum(int(r[0]["id"]) == int(filmes.iloc[i]["id"]) for r, i in zip(res, posicoes))
    checar("TOP-1 auto-recuperacao", acertos >= len(posicoes) - 1, f"{acertos}/{len(posicoes)}", rel)

    exemplos: dict = {}
    for c in CONSULTAS:
        if c.validacao == "nenhuma":
            continue
        r_mysql, r_milvus = c.mysql_fn(ctx), c.milvus_fn(ctx)
        exemplos[c.codigo] = {
            "mysql": r_mysql[:5],
            "milvus": r_milvus[:5],
            "linhas_mysql": len(r_mysql),
            "linhas_milvus": len(r_milvus),
        }
        if c.validacao == "igual":
            ok = respostas_iguais(r_mysql, r_milvus) and len(r_mysql) > 0
            checar(f"{c.codigo}", ok, f"mysql={len(r_mysql)} milvus={len(r_milvus)}", rel)
        elif c.validacao == "recall":
            rc = recall(r_mysql, r_milvus)
            exemplos[c.codigo]["recall@10"] = rc
            checar(f"{c.codigo} recall@10", rc >= 0.8, f"{rc:.2f}", rel)
        else:
            log(f"[INFO] {c.codigo} MySQL={[t for _, t, _ in r_mysql[:3]]} Milvus={[t for _, t, _ in r_milvus[:3]]}")

    consulta = next(c for c in CONSULTAS if c.codigo == "C02")
    antes = c01_votos(ctx)
    consulta.mysql_fn(ctx)
    consulta.milvus_fn(ctx)
    depois = c01_votos(ctx)
    consulta.mysql_desfaz(ctx)
    consulta.milvus_desfaz(ctx)
    restaurado = c01_votos(ctx)
    ok = (
        all(d == a + 1 for a, d in zip(antes["mysql"], depois["mysql"]))
        and all(d == a + 1 for a, d in zip(antes["milvus"], depois["milvus"]))
        and restaurado == antes
    )
    checar("C02 update/rollback", ok, f"registros={len(antes['mysql'])}", rel)

    con.close()
    falhas = [r for r in rel if not r["ok"]]
    salvar_json(
        cfg.dir_resultados / "validacao.json",
        {"verificacoes": rel, "exemplos": exemplos, "falhas": len(falhas)},
    )
    log(f"Validacao: {len(rel) - len(falhas)}/{len(rel)} OK")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
