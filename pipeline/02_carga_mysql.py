from __future__ import annotations

from pathlib import Path

from comum import (
    carregar_dados_preparados,
    conectar_mysql,
    cronometro,
    info_ambiente,
    log,
    salvar_json,
    vetor_para_blob,
)
from config import Config, carregar_config, criar_parser

DIR_SQL = Path(__file__).resolve().parent / "sql"


def comandos_sql(arquivo: str) -> list[str]:
    texto = "\n".join(
        linha
        for linha in (DIR_SQL / arquivo).read_text(encoding="utf-8").splitlines()
        if not linha.strip().startswith("--")
    )
    return [c.strip() for c in texto.split(";") if c.strip()]


def recriar_banco(cfg: Config) -> None:
    con = conectar_mysql(cfg, com_banco=False)
    with con.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS `{cfg.mysql_banco}`")
        cur.execute(
            f"CREATE DATABASE `{cfg.mysql_banco}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
        )
    con.commit()
    con.close()


def inserir_lotes(con, sql: str, linhas: list[tuple], lote: int) -> float:
    with cronometro() as t, con.cursor() as cur:
        for i in range(0, len(linhas), lote):
            cur.executemany(sql, linhas[i : i + lote])
            con.commit()
    return t()


def espaco_tabelas(con, banco: str) -> dict:
    with con.cursor() as cur:
        for tabela in ("filmes", "generos", "filme_genero", "elenco", "filmes_vetores"):
            cur.execute(f"ANALYZE TABLE `{tabela}`")
            cur.fetchall()
        cur.execute(
            "SELECT table_name, table_rows, data_length, index_length "
            "FROM information_schema.tables WHERE table_schema = %s",
            (banco,),
        )
        return {
            nome: {"linhas_estimadas": linhas, "dados_bytes": dados, "indices_bytes": indices}
            for nome, linhas, dados, indices in cur.fetchall()
        }


def main() -> None:
    parser = criar_parser()
    cfg = carregar_config(parser.parse_args())
    lote = cfg.perfil.lote_insercao

    filmes, elenco, vet_filmes, _ = carregar_dados_preparados(cfg)
    log(f"MySQL: {cfg.mysql_host}:{cfg.mysql_porta}/{cfg.mysql_banco}")

    recriar_banco(cfg)
    con = conectar_mysql(cfg)
    with con.cursor() as cur:
        for comando in comandos_sql("schema_mysql.sql"):
            cur.execute(comando)
    con.commit()

    linhas_filmes = [
        (
            int(f.id),
            f.titulo,
            f.titulo_original,
            f.sinopse,
            int(f.ano),
            f.data_lancamento,
            round(float(f.nota), 1),
            int(f.votos),
            float(f.popularidade),
            int(f.orcamento),
            int(f.receita),
            int(f.duracao),
            f.idioma,
            f.diretor,
        )
        for f in filmes.itertuples()
    ]
    nomes_generos = sorted({g for lista in filmes["generos"] for g in lista})
    id_genero = {nome: i + 1 for i, nome in enumerate(nomes_generos)}
    linhas_generos = [(i, nome) for nome, i in id_genero.items()]
    linhas_fg = sorted(
        {(int(f.id), id_genero[g]) for f in filmes.itertuples() for g in f.generos}
    )
    linhas_elenco = [
        (int(e.id), int(e.filme_id), int(e.ordem), e.ator, e.personagem, int(e.genero_ator))
        for e in elenco.itertuples()
    ]
    linhas_vetores = [(int(i), vetor_para_blob(v)) for i, v in zip(filmes["id"], vet_filmes)]

    tempos: dict[str, float] = {}
    plano = [
        (
            "filmes",
            "INSERT INTO filmes (id, titulo, titulo_original, sinopse, ano, "
            "data_lancamento, nota, votos, popularidade, orcamento, receita, duracao, "
            "idioma, diretor) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            linhas_filmes,
        ),
        ("generos", "INSERT INTO generos (id, nome) VALUES (%s,%s)", linhas_generos),
        (
            "filme_genero",
            "INSERT INTO filme_genero (filme_id, genero_id) VALUES (%s,%s)",
            linhas_fg,
        ),
        (
            "elenco",
            "INSERT INTO elenco (id, filme_id, ordem, ator, personagem, genero_ator) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            linhas_elenco,
        ),
        (
            "filmes_vetores",
            "INSERT INTO filmes_vetores (filme_id, vetor) VALUES (%s,%s)",
            linhas_vetores,
        ),
    ]
    for tabela, sql, linhas in plano:
        tempos[tabela] = round(inserir_lotes(con, sql, linhas, lote), 3)
        log(f"INSERT {tabela}: {len(linhas)} ({tempos[tabela]:.2f}s)")

    with cronometro() as t, con.cursor() as cur:
        for comando in comandos_sql("indices_mysql.sql"):
            cur.execute(comando)
        con.commit()
    tempo_indices = round(t(), 3)
    log(f"Indices MySQL: {tempo_indices:.2f}s")

    resultado = {
        "ambiente": info_ambiente(cfg, conexao_mysql=con),
        "linhas": {t: len(l) for t, _, l in plano},
        "insercao_s": tempos,
        "insercao_total_s": round(sum(tempos.values()), 3),
        "indexacao_s": tempo_indices,
        "espaco": espaco_tabelas(con, cfg.mysql_banco),
    }
    con.close()
    salvar_json(cfg.dir_resultados / "carga_mysql.json", resultado)
    log(f"MySQL concluido: insert={resultado['insercao_total_s']:.2f}s index={tempo_indices:.2f}s")


if __name__ == "__main__":
    main()
