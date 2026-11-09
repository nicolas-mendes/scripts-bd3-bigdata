from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from comum import cronometro, log, salvar_json, vetorizar
from config import ARQUIVOS_BRUTOS, DATASET_HF, Config, carregar_config, criar_parser

MAX_PALAVRAS_CHAVE = 30


def baixar_brutos(cfg: Config) -> dict[str, str]:
    caminhos = {}
    for arquivo in ARQUIVOS_BRUTOS:
        log(f"Download: {arquivo}")
        caminhos[arquivo] = hf_hub_download(
            repo_id=DATASET_HF,
            filename=arquivo,
            repo_type="dataset",
            local_dir=cfg.dir_brutos,
        )
    return caminhos


def literal(valor) -> list:
    if not isinstance(valor, str) or not valor.startswith("["):
        return []
    try:
        return ast.literal_eval(valor)
    except (ValueError, SyntaxError):
        return []


def nomes(valor) -> list[str]:
    return [d["name"].strip() for d in literal(valor) if isinstance(d, dict) and d.get("name")]


def ids_validos(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def preparar_filmes(caminho: str, limite: int | None) -> pd.DataFrame:
    df = pd.read_csv(caminho, dtype=str, low_memory=False)
    total_bruto = len(df)

    df["id"] = ids_validos(df["id"])
    df = df[df["id"].notna() & (df["adult"].str.lower() == "false")]
    df["id"] = df["id"].astype("int64")
    df = df.drop_duplicates("id")

    df["title"] = df["title"].str.strip()
    df["overview"] = df["overview"].str.strip()
    df = df[df["title"].str.len().gt(0) & df["overview"].str.len().ge(20)]

    data = pd.to_datetime(df["release_date"], errors="coerce")
    df = df[data.notna()].copy()
    df["data_lancamento"] = data[data.notna()].dt.date
    df["ano"] = data[data.notna()].dt.year.astype("int32")

    def num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)

    saida = pd.DataFrame(
        {
            "id": df["id"],
            "titulo": df["title"].str.slice(0, 300),
            "titulo_original": df["original_title"].fillna("").str.slice(0, 300),
            "sinopse": df["overview"],
            "ano": df["ano"],
            "data_lancamento": df["data_lancamento"],
            "nota": num("vote_average").round(1).astype("float32"),
            "votos": num("vote_count").astype("int32"),
            "popularidade": num("popularity").astype("float32"),
            "orcamento": num("budget").astype("int64"),
            "receita": num("revenue").astype("int64"),
            "duracao": num("runtime").astype("int32"),
            "idioma": df["original_language"].fillna("").str.slice(0, 10),
            "generos": df["genres"].map(nomes),
        }
    )

    saida = saida.sort_values(["votos", "id"], ascending=[False, True])
    if limite:
        saida = saida.head(limite)
    saida = saida.sort_values("id").reset_index(drop=True)
    log(f"Filmes filtrados: {len(saida)}/{total_bruto}")
    return saida


def anexar_palavras_chave(filmes: pd.DataFrame, caminho: str) -> pd.DataFrame:
    kw = pd.read_csv(caminho, dtype=str)
    kw["id"] = ids_validos(kw["id"])
    kw = kw.dropna(subset=["id"]).astype({"id": "int64"}).drop_duplicates("id")
    kw = kw[kw["id"].isin(filmes["id"])]
    mapa = dict(zip(kw["id"], kw["keywords"].map(lambda v: nomes(v)[:MAX_PALAVRAS_CHAVE])))
    filmes["palavras_chave"] = filmes["id"].map(lambda i: mapa.get(i, []))
    return filmes


def extrair_creditos(filmes: pd.DataFrame, caminho: str, top_n: int) -> pd.DataFrame:
    cred = pd.read_csv(caminho, dtype=str)
    cred["id"] = ids_validos(cred["id"])
    cred = cred.dropna(subset=["id"]).astype({"id": "int64"}).drop_duplicates("id")
    cred = cred[cred["id"].isin(filmes["id"])]

    diretores: dict[int, str] = {}
    linhas: list[dict] = []
    for filme_id, cast_txt, crew_txt in zip(cred["id"], cred["cast"], cred["crew"]):
        diretor = next(
            (p["name"] for p in literal(crew_txt) if isinstance(p, dict) and p.get("job") == "Director"),
            "",
        )
        diretores[filme_id] = diretor.strip()[:200]

        elenco = sorted(
            (p for p in literal(cast_txt) if isinstance(p, dict) and p.get("name")),
            key=lambda p: p.get("order", 999),
        )[:top_n]
        for posicao, p in enumerate(elenco):
            linhas.append(
                {
                    "id": filme_id * 100 + posicao,
                    "filme_id": filme_id,
                    "ordem": posicao,
                    "ator": p["name"].strip()[:200],
                    "personagem": (p.get("character") or "").strip()[:400],
                    "genero_ator": int(p.get("gender") or 0),
                }
            )

    filmes["diretor"] = filmes["id"].map(lambda i: diretores.get(i, ""))
    elenco_df = pd.DataFrame(linhas)
    principais = elenco_df.sort_values(["filme_id", "ordem"]).groupby("filme_id")["ator"].apply(list)
    filmes["elenco_principal"] = filmes["id"].map(lambda i: principais.get(i, []))
    log(f"Creditos extraidos: {len(elenco_df)} registros")
    return elenco_df.sort_values("id").reset_index(drop=True)


def texto_filme(linha) -> str:
    generos = ", ".join(linha.generos)
    return f"{linha.titulo}. {generos}. {linha.sinopse}"


def texto_elenco(linha) -> str:
    return f"{linha.ator} como {linha.personagem}" if linha.personagem else linha.ator


def resumo_base(filmes: pd.DataFrame, elenco: pd.DataFrame, tempos: dict) -> dict:
    return {
        "filmes": len(filmes),
        "elenco_linhas": len(elenco),
        "atores_distintos": int(elenco["ator"].nunique()),
        "generos_distintos": int(filmes["generos"].explode().nunique()),
        "ano_min": int(filmes["ano"].min()),
        "ano_max": int(filmes["ano"].max()),
        "nota_media": round(float(filmes["nota"].mean()), 3),
        "sinopse_media_caracteres": round(float(filmes["sinopse"].str.len().mean()), 1),
        "atributos_filme": list(filmes.columns),
        "atributos_elenco": list(elenco.columns),
        "tempos_s": tempos,
    }


def main() -> None:
    parser = criar_parser()
    parser.add_argument("--forcar", action="store_true")
    args = parser.parse_args()
    cfg = carregar_config(args)
    perfil = cfg.perfil

    if (cfg.dir_dados / "filmes_vetores.npy").exists() and not args.forcar:
        log(f"Cache encontrado em {cfg.dir_dados}")
        return

    tempos: dict[str, float] = {}
    caminhos = baixar_brutos(cfg)

    with cronometro() as t:
        filmes = preparar_filmes(caminhos["movies_metadata.csv"], perfil.limite_filmes)
        filmes = anexar_palavras_chave(filmes, caminhos["keywords.csv"])
        elenco = extrair_creditos(filmes, caminhos["credits.csv"], perfil.elenco_top_n)
    tempos["limpeza_integracao"] = round(t(), 2)

    log(f"Vetorizando filmes ({len(filmes)})...")
    with cronometro() as t:
        vet_filmes = vetorizar([texto_filme(l) for l in filmes.itertuples()], mostrar_progresso=True)
    tempos["vetorizacao_filmes"] = round(t(), 2)

    log(f"Vetorizando elenco ({len(elenco)})...")
    with cronometro() as t:
        vet_elenco = vetorizar(
            [texto_elenco(l) for l in elenco.itertuples()],
            mostrar_progresso=True,
            lote=256,
        )
    tempos["vetorizacao_elenco"] = round(t(), 2)

    filmes.to_parquet(cfg.dir_dados / "filmes.parquet", index=False)
    elenco.to_parquet(cfg.dir_dados / "elenco.parquet", index=False)
    np.save(cfg.dir_dados / "filmes_vetores.npy", vet_filmes)
    np.save(cfg.dir_dados / "elenco_vetores.npy", vet_elenco)

    resumo = resumo_base(filmes, elenco, tempos)
    salvar_json(cfg.dir_dados / "resumo.json", resumo)
    log(f"Preparacao concluida: filmes={resumo['filmes']} elenco={resumo['elenco_linhas']} tempos={tempos}")


if __name__ == "__main__":
    main()
