from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from comum import log
from config import criar_parser

DIR_PIPELINE = Path(__file__).resolve().parent

ETAPAS = [
    ("01_preparar_dados.py", "Preparacao"),
    ("02_carga_mysql.py", "Carga MySQL"),
    ("03_carga_milvus.py", "Carga Milvus"),
    ("04_validar.py", "Validacao"),
    ("05_benchmark.py", "Benchmark"),
]


def main() -> None:
    parser = criar_parser()
    parser.add_argument("--pular-preparacao", action="store_true")
    parser.add_argument("--pular-carga", action="store_true")
    args = parser.parse_args()

    for idx, (script, titulo) in enumerate(ETAPAS, start=1):
        if args.pular_preparacao and script.startswith("01_"):
            log(f"[{idx}/5] Pulando {titulo}")
            continue
        if args.pular_carga and (script.startswith("02_") or script.startswith("03_")):
            log(f"[{idx}/5] Pulando {titulo}")
            continue

        log(f"[{idx}/5] {titulo} ({script})")
        cmd = [sys.executable, str(DIR_PIPELINE / script), "--modo", args.modo]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            log(f"Erro em {script} (codigo {res.returncode})")
            sys.exit(res.returncode)

    log("Pipeline finalizado.")


if __name__ == "__main__":
    main()
