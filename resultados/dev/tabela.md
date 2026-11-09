# Benchmark MySQL x Milvus -- perfil `dev`

- Data: 2026-11-08T22:12:29 | Milvus: zilliz (Zilliz Cloud Vector Database(Compatible with Milvus 3.0)) | MySQL 8.4.13
- Máquina: AMD Ryzen 5 5600 6-Core Processor | RAM 31,9 GB | Windows-11-10.0.26200-SP0
- Repetições medidas: 3 (+1 de aquecimento)
- Latência-base (ida e volta mínima): MySQL 0,5 ms | Milvus 299,1 ms

| Código | Operação | MySQL (ms) | Milvus (ms) | Obs. |
|---|---|---:|---:|---|
| -- | Importação (inserção em lotes) | 2.323,0 | 19.930,0 |  |
| -- | Indexação (+ load no Milvus) | 1.041,0 | 25.434,0 |  |
| C01 | Leitura por chave primária (lote de 100) | 1,7 | 206,8 |  |
| C02 | Atualização em lote (100 registros) | 4,1 | 991,3 |  |
| C03 | Condicional simples + ordenação | 1,0 | 198,5 |  |
| C04 | Busca por parte do texto (título) | 1,1 | 195,4 |  |
| C05 | Agregação por ano (COUNT/AVG/MAX/MIN) | 2,5 | 957,6 |  |
| C06 | Busca por parte do nome do ator (JOIN) | 3,2 | 424,1 |  |
| C07 | Filmes de um ator: JOIN x documento desnormalizado | 0,7 | 202,4 |  |
| C08 | Top-10 atores por nota média (JOIN + GROUP BY + HAVING) | 14,7 | 2.116,6 |  |
| C09 | Agregação por gênero (relação N:N x ARRAY) | 8,4 | 993,5 |  |
| C10 | Busca semântica kNN top-10 (consulta em português) | 12,4 | 202,9 | recall@10 = 1.00 |
| C11 | Busca híbrida: similaridade + filtros (nota, ano, gênero) | 2,6 | 206,1 | recall@10 = 1.00 |
| C12 | Linguagem natural: FULLTEXT (léxico) x semântica | 0,9 | 208,5 |  |
