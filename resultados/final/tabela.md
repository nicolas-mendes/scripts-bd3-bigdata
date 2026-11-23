# Benchmark MySQL x Milvus (final)

- Filmes: 44213 | Repeticoes: 5 | Ping MySQL: 0,5ms | Ping Milvus: 249,5ms

| Cod | Operacao | MySQL (ms) | Milvus (ms) | Obs |
|---|---|---:|---:|---|
| -- | Importacao (lotes) | 22.465,0 | 148.569,0 |  |
| -- | Indexacao (+ load) | 5.560,0 | 61.528,0 |  |
| C01 | Leitura por chave primária (lote de 100) | 1,7 | 204,2 |  |
| C02 | Atualização em lote (100 registros) | 5,3 | 765,0 |  |
| C03 | Condicional simples + ordenação | 0,9 | 199,8 |  |
| C04 | Busca por parte do texto (título) | 14,8 | 219,9 |  |
| C05 | Agregação por ano (COUNT/AVG/MAX/MIN) | 44,8 | 1.505,2 |  |
| C06 | Busca por parte do nome do ator (JOIN) | 39,7 | 398,4 |  |
| C07 | Filmes de um ator: JOIN x documento desnormalizado | 0,8 | 198,1 |  |
| C08 | Top-10 atores por nota média (JOIN + GROUP BY + HAVING) | 94,7 | 9.661,9 |  |
| C09 | Busca semantica em lote (10 consultas x top-10) | 3.047,3 | 315,2 | recall@10=0.97 |
| C10 | Busca semântica kNN top-10 (consulta em português) | 245,2 | 203,3 | recall@10=0.90 |
| C11 | Busca híbrida: similaridade + filtros (nota, ano, gênero) | 10,7 | 203,6 | recall@10=0.90 |
| C12 | Linguagem natural: FULLTEXT (léxico) x semântica | 3,5 | 206,6 |  |
