import numpy as np

from metricas import precisao_recall_f1_por_classe


def perfil_medio_por_classe(atributos, alvo, nomes_atributos, nomes_classes, colunas_numericas):
    """Média bruta de cada atributo contínuo dentro de cada classe."""
    perfil = {}
    for classe, nome_classe in nomes_classes.items():
        amostras_da_classe = atributos[alvo == classe]
        perfil[nome_classe] = {
            nome: float(amostras_da_classe[:, coluna].mean())
            for coluna, nome in enumerate(nomes_atributos)
            if colunas_numericas[coluna]
        }
    return perfil


def categorias_que_mais_separam(atributos, alvo, nomes_atributos, colunas_numericas, quantidade=6):
    """Categorias cuja frequência mais difere entre as classes 0 e 1 (em pontos percentuais)."""
    resultado = []
    for coluna, nome in enumerate(nomes_atributos):
        if colunas_numericas[coluna]:
            continue
        na_classe_0 = float(atributos[alvo == 0, coluna].mean())
        na_classe_1 = float(atributos[alvo == 1, coluna].mean())
        resultado.append({"categoria": nome, "classe_0": na_classe_0, "classe_1": na_classe_1})
    resultado.sort(key=lambda item: abs(item["classe_0"] - item["classe_1"]), reverse=True)
    return resultado[:quantidade]


def desempenho_por_classe(matriz, nomes_classes):
    precisao, recall, f1 = precisao_recall_f1_por_classe(matriz)
    return {
        nome: {
            "precisao": float(precisao[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "amostras": int(matriz[i].sum()),
        }
        for i, nome in enumerate(nomes_classes.values())
    }


def erro_medio_por_nota(alvo_real, alvo_previsto):
    resultado = {}
    for nota in np.unique(alvo_real):
        mascara = alvo_real == nota
        resultado[str(int(nota))] = {
            "mae": float(np.mean(np.abs(alvo_real[mascara] - alvo_previsto[mascara]))),
            "amostras": int(mascara.sum()),
        }
    return resultado
