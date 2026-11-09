CREATE TABLE filmes (
    id               INT           NOT NULL,
    titulo           VARCHAR(300)  COLLATE utf8mb4_bin NOT NULL,
    titulo_original  VARCHAR(300)  NULL,
    sinopse          TEXT          NOT NULL,
    ano              SMALLINT      NOT NULL,
    data_lancamento  DATE          NULL,
    nota             DECIMAL(3,1)  NOT NULL,
    votos            INT           NOT NULL,
    popularidade     FLOAT         NOT NULL,
    orcamento        BIGINT        NOT NULL,
    receita          BIGINT        NOT NULL,
    duracao          SMALLINT      NOT NULL,
    idioma           VARCHAR(10)   NOT NULL,
    diretor          VARCHAR(200)  COLLATE utf8mb4_bin NOT NULL,
    PRIMARY KEY (id)
) ENGINE = InnoDB;

CREATE TABLE generos (
    id    SMALLINT     NOT NULL AUTO_INCREMENT,
    nome  VARCHAR(50)  COLLATE utf8mb4_bin NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_generos_nome (nome)
) ENGINE = InnoDB;

CREATE TABLE filme_genero (
    filme_id   INT       NOT NULL,
    genero_id  SMALLINT  NOT NULL,
    PRIMARY KEY (filme_id, genero_id),
    CONSTRAINT fk_fg_filme  FOREIGN KEY (filme_id)  REFERENCES filmes (id),
    CONSTRAINT fk_fg_genero FOREIGN KEY (genero_id) REFERENCES generos (id)
) ENGINE = InnoDB;

CREATE TABLE elenco (
    id           BIGINT        NOT NULL,
    filme_id     INT           NOT NULL,
    ordem        TINYINT       NOT NULL,
    ator         VARCHAR(200)  COLLATE utf8mb4_bin NOT NULL,
    personagem   VARCHAR(400)  COLLATE utf8mb4_bin NOT NULL,
    genero_ator  TINYINT       NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT fk_elenco_filme FOREIGN KEY (filme_id) REFERENCES filmes (id)
) ENGINE = InnoDB;

CREATE TABLE filmes_vetores (
    filme_id  INT   NOT NULL,
    vetor     BLOB  NOT NULL,
    PRIMARY KEY (filme_id),
    CONSTRAINT fk_vetor_filme FOREIGN KEY (filme_id) REFERENCES filmes (id)
) ENGINE = InnoDB;
