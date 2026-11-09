CREATE INDEX idx_filmes_ano        ON filmes (ano);
CREATE INDEX idx_filmes_nota_votos ON filmes (nota, votos);
CREATE INDEX idx_filmes_votos      ON filmes (votos);
CREATE INDEX idx_filmes_diretor    ON filmes (diretor);
CREATE INDEX idx_elenco_ator       ON elenco (ator);
CREATE FULLTEXT INDEX ft_filmes_sinopse ON filmes (sinopse);
