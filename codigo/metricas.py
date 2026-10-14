import numpy as np


def matriz_confusao(alvo_real, alvo_previsto, classes):
    posicao = {classe: i for i, classe in enumerate(classes)}
    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for real, previsto in zip(alvo_real, alvo_previsto):
        matriz[posicao[real], posicao[previsto]] += 1
    return matriz


def precisao_recall_f1_por_classe(matriz):
    """Devolve três vetores (um valor por classe) a partir da matriz de confusão."""
    acertos = np.diag(matriz).astype(float)
    total_previsto = matriz.sum(axis=0)
    total_real = matriz.sum(axis=1)

    precisao = np.divide(acertos, total_previsto, out=np.zeros_like(acertos), where=total_previsto > 0)
    recall = np.divide(acertos, total_real, out=np.zeros_like(acertos), where=total_real > 0)
    soma = precisao + recall
    f1 = np.divide(2 * precisao * recall, soma, out=np.zeros_like(acertos), where=soma > 0)
    return precisao, recall, f1


def metricas_classificacao(alvo_real, alvo_previsto, classes):
    """Acurácia e média macro de precisão, recall e F1."""
    matriz = matriz_confusao(alvo_real, alvo_previsto, classes)
    precisao, recall, f1 = precisao_recall_f1_por_classe(matriz)
    return {
        "Acurácia": np.trace(matriz) / matriz.sum(),
        "Precisão": precisao.mean(),
        "Recall": recall.mean(),
        "F1-Score": f1.mean(),
    }


def metricas_regressao(alvo_real, alvo_previsto, quantidade_atributos):
    n = len(alvo_real)
    soma_quadrados_residuos = np.sum((alvo_real - alvo_previsto) ** 2)
    soma_quadrados_total = np.sum((alvo_real - alvo_real.mean()) ** 2)

    r2 = 1 - soma_quadrados_residuos / soma_quadrados_total
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - quantidade_atributos - 1)
    return {
        "R²": r2,
        "R² ajustado": r2_ajustado,
        "MAE": np.mean(np.abs(alvo_real - alvo_previsto)),
        "RMSE": np.sqrt(soma_quadrados_residuos / n),
    }


def taxa_de_acerto_com_tolerancia(alvo_real, alvo_previsto, tolerancia):
    return float(np.mean(np.abs(alvo_real - alvo_previsto) <= tolerancia))
