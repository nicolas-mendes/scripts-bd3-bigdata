from __future__ import annotations

from comum import (
    conectar_milvus,
    conectar_mysql,
    disco_container,
    info_ambiente,
    ler_json,
    log,
    medir,
    memoria_container,
    salvar_json,
)
from config import Config, carregar_config, criar_parser
from consultas import CONSULTAS, Contexto, preparar_parametros, recall


def latencia_base(ctx: Contexto, rep: int, aq: int) -> dict:
    def mysql_ping():
        with ctx.mysql.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchall()

    um_id = ctx.p["ids_amostra"][0]

    def milvus_ping():
        return ctx.milvus.get(collection_name=ctx.cf, ids=[um_id], output_fields=["id"])

    return {
        "mysql": medir(mysql_ping, rep * 4, aq)[0],
        "milvus": medir(milvus_ping, rep * 4, aq)[0],
    }


def recursos(cfg: Config, con) -> dict:
    with con.cursor() as cur:
        cur.execute(
            "SELECT SUM(data_length), SUM(index_length) FROM information_schema.tables "
            "WHERE table_schema = %s",
            (cfg.mysql_banco,),
        )
        dados, indices = cur.fetchone()
    info = {
        "mysql_dados_bytes": int(dados or 0),
        "mysql_indices_bytes": int(indices or 0),
        "mysql_memoria_container": memoria_container(cfg.mysql_container),
    }
    if cfg.milvus_modo == "local":
        info["milvus_memoria_container"] = memoria_container(cfg.milvus_container)
        info["milvus_disco_minio_bytes"] = disco_container("milvus-minio", "/minio_data")
    return info


def fmt(ms: float | None) -> str:
    if ms is None:
        return "--"
    return f"{ms:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def linhas_tabela(res: dict) -> list[tuple[str, str, float | None, float | None, str]]:
    linhas = []
    cm, cv = res.get("carga_mysql") or {}, res.get("carga_milvus") or {}
    if cm and cv:
        linhas.append(
            (
                "--",
                "Importacao (lotes)",
                cm["insercao_total_s"] * 1000,
                cv["insercao_total_s"] * 1000,
                "",
            )
        )
        linhas.append(
            (
                "--",
                "Indexacao (+ load)",
                cm["indexacao_s"] * 1000,
                cv["indexacao_total_s"] * 1000,
                "",
            )
        )
    for codigo, r in res["consultas"].items():
        obs = f"recall@10={r['recall']:.2f}" if "recall" in r else ""
        linhas.append((codigo, r["titulo"], r["mysql"]["media_ms"], r["milvus"]["media_ms"], obs))
    return linhas


def gerar_markdown(res: dict) -> str:
    amb = res["ambiente"]
    lb = res["latencia_base"]
    out = [
        f"# Benchmark MySQL x Milvus ({amb['perfil']})",
        "",
        f"- Filmes: {res['linhas_filmes']} | Repeticoes: {res['repeticoes']} | "
        f"Ping MySQL: {fmt(lb['mysql']['media_ms'])}ms | Ping Milvus: {fmt(lb['milvus']['media_ms'])}ms",
        "",
        "| Cod | Operacao | MySQL (ms) | Milvus (ms) | Obs |",
        "|---|---|---:|---:|---|",
    ]
    for codigo, nome, m, v, obs in linhas_tabela(res):
        out.append(f"| {codigo} | {nome} | {fmt(m)} | {fmt(v)} | {obs} |")
    return "\n".join(out) + "\n"


def escapar_latex(texto: str) -> str:
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        texto = texto.replace(a, b)
    return texto


def gerar_latex(res: dict) -> str:
    amb = res["ambiente"]
    corpo = "\n".join(
        f"{escapar_latex(c)} & {escapar_latex(n)} & {fmt(m)} & {fmt(v)} \\\\ \\hline"
        for c, n, m, v, _ in linhas_tabela(res)
    )
    return (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{Tempo médio de execução em ms (n = {res['repeticoes']}) -- "
        f"MySQL x Milvus ({escapar_latex(amb['milvus_modo'])}), base com "
        f"{res['linhas_filmes']} filmes}}\n"
        "\\label{tab:resultados}\n\\small\n"
        "\\begin{tabular}{|c|l|r|r|}\n\\hline\n"
        "\\textbf{Cód.} & \\textbf{Operação} & \\textbf{MySQL} & \\textbf{Milvus} \\\\ \\hline\n"
        f"{corpo}\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def main() -> None:
    parser = criar_parser()
    parser.add_argument("--consultas")
    parser.add_argument("--repeticoes", type=int)
    args = parser.parse_args()
    cfg = carregar_config(args)
    rep = args.repeticoes or cfg.perfil.repeticoes
    aq = cfg.perfil.aquecimento
    selecao = set(args.consultas.upper().split(",")) if args.consultas else None

    con = conectar_mysql(cfg)
    cli = conectar_milvus(cfg)
    ctx = Contexto(cfg, con, cli, preparar_parametros(cfg, con))

    with con.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM filmes")
        n_filmes = cur.fetchone()[0]

    log(f"Benchmark ({cfg.perfil.nome}): filmes={n_filmes} rep={rep}")
    res: dict = {
        "ambiente": info_ambiente(cfg, con, cli),
        "repeticoes": rep,
        "aquecimento": aq,
        "linhas_filmes": n_filmes,
        "latencia_base": latencia_base(ctx, rep, aq),
        "carga_mysql": ler_json(cfg.dir_resultados / "carga_mysql.json"),
        "carga_milvus": ler_json(cfg.dir_resultados / "carga_milvus.json"),
        "consultas": {},
    }

    for c in CONSULTAS:
        if selecao and c.codigo not in selecao:
            continue
        est_mysql, r_mysql = medir(
            lambda: c.mysql_fn(ctx),
            rep,
            aq,
            (lambda: c.mysql_desfaz(ctx)) if c.mysql_desfaz else None,
        )
        est_milvus, r_milvus = medir(
            lambda: c.milvus_fn(ctx),
            rep,
            aq,
            (lambda: c.milvus_desfaz(ctx)) if c.milvus_desfaz else None,
        )
        item = {
            "titulo": c.titulo,
            "categoria": c.categoria,
            "sql": c.sql,
            "milvus_op": c.milvus,
            "mysql": est_mysql,
            "milvus": est_milvus,
        }
        if c.validacao == "recall":
            item["recall"] = recall(r_mysql, r_milvus)
        res["consultas"][c.codigo] = item
        log(
            f"{c.codigo} | MySQL: {fmt(est_mysql['media_ms'])}ms | "
            f"Milvus: {fmt(est_milvus['media_ms'])}ms"
        )

    res["recursos"] = recursos(cfg, con)
    con.close()

    salvar_json(cfg.dir_resultados / "benchmark.json", res)
    (cfg.dir_resultados / "tabela.md").write_text(gerar_markdown(res), encoding="utf-8")
    (cfg.dir_resultados / "tabela.tex").write_text(gerar_latex(res), encoding="utf-8")
    print("\n" + gerar_markdown(res))
    log(f"Arquivos salvos em {cfg.dir_resultados}")


if __name__ == "__main__":
    main()
