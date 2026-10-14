import csv

import numpy as np


def _ler_declaracao(linha):
    resto = linha.strip()[len("@attribute"):].strip()
    if resto[0] in "\"'":
        aspas = resto[0]
        fim = resto.index(aspas, 1)
        return resto[1:fim], resto[fim + 1:].strip()
    nome, tipo = resto.split(None, 1)
    return nome, tipo.strip()


def _e_categorico(tipo):
    return tipo.startswith("{") or tipo.lower() == "string"


def _categorias_declaradas(tipo):
    return [item.strip().strip("'\"") for item in tipo.strip("{}").split(",")]


def carregar_arff(caminho):
    """Lê um ARFF e devolve (atributos, alvo, nomes_das_colunas, colunas_numericas).

    A última coluna é o alvo. Atributos categóricos viram colunas 0/1 (uma por
    categoria) e `colunas_numericas` marca quais colunas são contínuas.
    """
    declaracoes = []
    registros = []
    lendo_dados = False

    with open(caminho, encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha or linha.startswith("%"):
                continue

            if lendo_dados:
                registros.append(next(csv.reader([linha], quotechar="'", skipinitialspace=True)))
            elif linha.lower().startswith("@attribute"):
                declaracoes.append(_ler_declaracao(linha))
            elif linha.lower().startswith("@data"):
                lendo_dados = True

    colunas, nomes, numericas = [], [], []
    for posicao, (nome, tipo) in enumerate(declaracoes[:-1]):
        valores = [registro[posicao] for registro in registros]
        if _e_categorico(tipo):
            for categoria in sorted(set(valores)):
                colunas.append([1.0 if valor == categoria else 0.0 for valor in valores])
                nomes.append(f"{nome}={categoria}")
                numericas.append(False)
        else:
            colunas.append([float(valor) for valor in valores])
            nomes.append(nome)
            numericas.append(True)

    nome_alvo, tipo_alvo = declaracoes[-1]
    valores_alvo = [registro[-1] for registro in registros]
    if _e_categorico(tipo_alvo):
        categorias = _categorias_declaradas(tipo_alvo) if tipo_alvo.startswith("{") else sorted(set(valores_alvo))
        alvo = np.array([categorias.index(valor) for valor in valores_alvo], dtype=float)
    else:
        alvo = np.array([float(valor) for valor in valores_alvo])

    return np.array(colunas).T, alvo, nomes, np.array(numericas)


def padronizar(atributos_treino, atributos_teste, colunas_numericas=None):
    """Z-score com média e desvio calculados apenas no treino.

    Só as colunas marcadas em `colunas_numericas` são padronizadas; as colunas 0/1
    ficam como estão. Sem a máscara, todas as colunas são padronizadas.
    """
    if colunas_numericas is None:
        colunas_numericas = np.ones(atributos_treino.shape[1], dtype=bool)

    media = atributos_treino[:, colunas_numericas].mean(axis=0)
    desvio = atributos_treino[:, colunas_numericas].std(axis=0)
    desvio[desvio == 0] = 1.0

    treino, teste = atributos_treino.copy(), atributos_teste.copy()
    treino[:, colunas_numericas] = (treino[:, colunas_numericas] - media) / desvio
    teste[:, colunas_numericas] = (teste[:, colunas_numericas] - media) / desvio
    return treino, teste
