import json
import os
import random
import time

import numpy as np

import graficos
from analises import (categorias_que_mais_separam, desempenho_por_classe, erro_medio_por_nota,
                      perfil_medio_por_classe)
from dados import carregar_arff, padronizar
from metricas import (matriz_confusao, metricas_classificacao, metricas_regressao,
                      taxa_de_acerto_com_tolerancia)
from modelos import (BayesUnivariado, KNNClassificador, KNNRegressor,
                     RegressaoLinearMultipla)
from validacao import validacao_cruzada

PASTA_RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PASTA_RESULTADOS = os.path.join(PASTA_RAIZ, "resultados")
PASTA_FIGURAS = os.path.join(PASTA_RESULTADOS, "figuras")

QUANTIDADE_FOLDS = 5
K_PADRAO = 5
SEMENTE = 42
VALORES_DE_K = [1, 3, 5, 7, 9, 11, 15, 21, 31]
CLASSES_CREDITO = {0: "Bom pagador", 1: "Mau pagador"}
METRICAS_CLASSIFICACAO = ["Acurácia", "Precisão", "Recall", "F1-Score", "Tempo Treino (s)", "Tempo Teste (s)"]
METRICAS_REGRESSAO = ["R²", "R² ajustado", "MAE", "RMSE", "Tempo Treino (s)", "Tempo Teste (s)"]


def avaliar_modelos(modelos, atributos, alvo, tarefa, colunas_numericas=None):
    """Executa a validação cruzada de cada modelo.

    Devolve as métricas por fold e as previsões de cada amostra feitas quando ela
    estava no conjunto de teste.
    """
    classificacao = tarefa == "classificacao"
    classes = np.unique(alvo.astype(int)) if classificacao else None
    metricas_por_fold = {nome: {} for nome in modelos}
    previsoes = {nome: np.zeros(len(alvo)) for nome in modelos}

    folds = validacao_cruzada(alvo, QUANTIDADE_FOLDS, estratificada=classificacao, semente=SEMENTE)
    for indices_treino, indices_teste in folds:
        treino, teste = padronizar(atributos[indices_treino], atributos[indices_teste], colunas_numericas)
        alvo_treino, alvo_teste = alvo[indices_treino], alvo[indices_teste]

        for nome, criar_modelo in modelos.items():
            modelo = criar_modelo()

            inicio = time.perf_counter()
            modelo.fit(treino, alvo_treino)
            tempo_treino = time.perf_counter() - inicio

            inicio = time.perf_counter()
            previsto = modelo.predict(teste)
            tempo_teste = time.perf_counter() - inicio

            if classificacao:
                metricas = metricas_classificacao(alvo_teste.astype(int), previsto, classes)
            else:
                metricas = metricas_regressao(alvo_teste, previsto, atributos.shape[1])
            metricas["Tempo Treino (s)"] = tempo_treino
            metricas["Tempo Teste (s)"] = tempo_teste

            for metrica, valor in metricas.items():
                metricas_por_fold[nome].setdefault(metrica, []).append(valor)
            previsoes[nome][indices_teste] = previsto

    return metricas_por_fold, previsoes


def resumir(metricas_por_fold):
    """Média e desvio padrão amostral de cada métrica, por modelo."""
    return {
        nome: {
            metrica: {"media": float(np.mean(valores)), "desvio": float(np.std(valores, ddof=1))}
            for metrica, valores in metricas.items()
        }
        for nome, metricas in metricas_por_fold.items()
    }


def imprimir_tabela(resumo, metricas):
    print("| Modelo | " + " | ".join(metricas) + " |")
    print("|---|" + "---|" * len(metricas))
    for nome, valores in resumo.items():
        celulas = [f"{valores[m]['media']:.4f} ± {valores[m]['desvio']:.4f}" for m in metricas]
        print(f"| {nome} | " + " | ".join(celulas) + " |")


def salvar_csv(resumo, caminho):
    metricas = list(next(iter(resumo.values())))
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write("modelo," + ",".join(f"{m} media,{m} desvio" for m in metricas) + "\n")
        for nome, valores in resumo.items():
            colunas = [f"{valores[m]['media']:.6f},{valores[m]['desvio']:.6f}" for m in metricas]
            arquivo.write(nome + "," + ",".join(colunas) + "\n")


def varredura_de_k(atributos, alvo, tarefa, colunas_numericas=None):
    """Desempenho médio dos folds para cada k, nas duas distâncias."""
    classificacao = tarefa == "classificacao"
    classe_knn = KNNClassificador if classificacao else KNNRegressor
    resultado = {}

    for distancia in ("euclidiana", "manhattan"):
        valores = {k: [] for k in VALORES_DE_K}
        folds = validacao_cruzada(alvo, QUANTIDADE_FOLDS, estratificada=classificacao, semente=SEMENTE)
        for indices_treino, indices_teste in folds:
            treino, teste = padronizar(atributos[indices_treino], atributos[indices_teste], colunas_numericas)
            modelo = classe_knn(distancia=distancia).fit(treino, alvo[indices_treino])
            for k, previsto in modelo.prever_varios_k(teste, VALORES_DE_K).items():
                alvo_teste = alvo[indices_teste]
                if classificacao:
                    valores[k].append(np.mean(previsto == alvo_teste.astype(int)))
                else:
                    valores[k].append(metricas_regressao(alvo_teste, previsto, atributos.shape[1])["R²"])
        resultado[distancia.capitalize()] = [float(np.mean(valores[k])) for k in VALORES_DE_K]
    return resultado


def grafico_de_metricas(resumo, metricas, titulo, arquivo, escala_log=False, rotulo_y=""):
    series = {
        metrica: (
            [resumo[nome][metrica]["media"] for nome in resumo],
            [resumo[nome][metrica]["desvio"] for nome in resumo],
        )
        for metrica in metricas
    }
    graficos.barras_agrupadas(
        os.path.join(PASTA_FIGURAS, arquivo), titulo, list(resumo), series,
        rotulo_y=rotulo_y, escala_log=escala_log,
    )


def analisar_credito():
    atributos, alvo, nomes, colunas_numericas = carregar_arff(os.path.join(PASTA_RAIZ, "classificacao_credito.arff"))
    distribuicao = {CLASSES_CREDITO[int(c)]: int((alvo == c).sum()) for c in np.unique(alvo)}
    atributos_originais = len(nomes) - int((~colunas_numericas).sum()) + len({n.split("=")[0] for n in nomes if "=" in n})
    print(f"[Crédito] {atributos.shape[0]} amostras, {atributos_originais} atributos "
          f"({atributos.shape[1]} colunas após codificação), {distribuicao}")

    modelos = {
        "kNN (Euclidiana)": lambda: KNNClassificador(K_PADRAO, "euclidiana"),
        "kNN (Manhattan)": lambda: KNNClassificador(K_PADRAO, "manhattan"),
        "Bayesiano (Univariado)": lambda: BayesUnivariado(colunas_binarias=~colunas_numericas),
    }
    metricas_por_fold, previsoes = avaliar_modelos(modelos, atributos, alvo, "classificacao", colunas_numericas)
    resumo = resumir(metricas_por_fold)
    imprimir_tabela(resumo, METRICAS_CLASSIFICACAO)
    salvar_csv(resumo, os.path.join(PASTA_RESULTADOS, "classificacao_credito.csv"))

    melhor_modelo = max(resumo, key=lambda nome: resumo[nome]["F1-Score"]["media"])
    matrizes = {
        nome: matriz_confusao(alvo.astype(int), previsoes[nome].astype(int), list(CLASSES_CREDITO))
        for nome in modelos
    }
    matriz = matrizes[melhor_modelo]
    varredura = varredura_de_k(atributos, alvo, "classificacao", colunas_numericas)

    grafico_de_metricas(resumo, ["Acurácia", "Precisão", "Recall", "F1-Score"],
                        "Classificação (crédito): média ± desvio em 5 folds", "classificacao_metricas.svg")
    grafico_de_metricas(resumo, ["Tempo Treino (s)", "Tempo Teste (s)"],
                        "Classificação: tempos médios (escala log)", "classificacao_tempos.svg",
                        escala_log=True, rotulo_y="segundos")
    graficos.linhas(os.path.join(PASTA_FIGURAS, "classificacao_k.svg"), "kNN: acurácia vs k (crédito)",
                    VALORES_DE_K, varredura, "k (vizinhos)", "Acurácia média")
    graficos.mapa_de_calor(os.path.join(PASTA_FIGURAS, "classificacao_confusao.svg"),
                           f"Matriz de confusão: {melhor_modelo}", matriz.tolist(), list(CLASSES_CREDITO.values()))

    return {
        "amostras": int(atributos.shape[0]),
        "atributos": atributos_originais,
        "atributos_numericos": int(colunas_numericas.sum()),
        "colunas_codificadas": int(atributos.shape[1]),
        "distribuicao": distribuicao,
        "acuracia_classe_majoritaria": max(distribuicao.values()) / atributos.shape[0],
        "modelos": resumo,
        "melhor_modelo": melhor_modelo,
        "matriz_confusao": matriz.tolist(),
        "matrizes_de_confusao": {nome: matriz_modelo.tolist() for nome, matriz_modelo in matrizes.items()},
        "custo_5_para_1": {
            nome: int(5 * matriz_modelo[1, 0] + matriz_modelo[0, 1]) for nome, matriz_modelo in matrizes.items()
        },
        "desempenho_por_classe": desempenho_por_classe(matriz, CLASSES_CREDITO),
        "perfil_por_classe": perfil_medio_por_classe(atributos, alvo, nomes, CLASSES_CREDITO, colunas_numericas),
        "categorias_que_mais_separam": categorias_que_mais_separam(atributos, alvo, nomes, colunas_numericas),
        "valores_de_k": VALORES_DE_K,
        "varredura_de_k": varredura,
    }


def analisar_vinho():
    atributos, alvo, nomes, _ = carregar_arff(os.path.join(PASTA_RAIZ, "wine_quality.arff"))
    distribuicao = {str(int(n)): int((alvo == n).sum()) for n in np.unique(alvo)}
    print(f"\n[Vinho] {atributos.shape[0]} amostras, {atributos.shape[1]} atributos, notas {distribuicao}")

    modelos = {
        "kNN (Euclidiana)": lambda: KNNRegressor(K_PADRAO, "euclidiana"),
        "kNN (Manhattan)": lambda: KNNRegressor(K_PADRAO, "manhattan"),
        "Regressão Linear Múltipla": lambda: RegressaoLinearMultipla(),
    }
    metricas_por_fold, previsoes = avaliar_modelos(modelos, atributos, alvo, "regressao")
    resumo = resumir(metricas_por_fold)
    imprimir_tabela(resumo, METRICAS_REGRESSAO)
    salvar_csv(resumo, os.path.join(PASTA_RESULTADOS, "regressao_vinho.csv"))

    varredura = varredura_de_k(atributos, alvo, "regressao")
    atributos_padronizados, _ = padronizar(atributos, atributos)
    coeficientes = RegressaoLinearMultipla().fit(atributos_padronizados, alvo).coeficientes[1:]
    ordem = np.argsort(np.abs(coeficientes))[::-1]

    grafico_de_metricas(resumo, ["R²", "R² ajustado"], "Regressão (vinho): R² e R² ajustado", "regressao_r2.svg")
    grafico_de_metricas(resumo, ["MAE", "RMSE"], "Regressão (vinho): erros (menor é melhor)", "regressao_erros.svg")
    grafico_de_metricas(resumo, ["Tempo Treino (s)", "Tempo Teste (s)"],
                        "Regressão: tempos médios (escala log)", "regressao_tempos.svg",
                        escala_log=True, rotulo_y="segundos")
    graficos.linhas(os.path.join(PASTA_FIGURAS, "regressao_k.svg"), "kNN: R² vs k (vinho)",
                    VALORES_DE_K, varredura, "k (vizinhos)", "R² médio")
    graficos.barras_horizontais(
        os.path.join(PASTA_FIGURAS, "regressao_coeficientes.svg"),
        "Coeficientes da regressão (atributos padronizados)",
        [nomes[i] for i in ordem], [float(coeficientes[i]) for i in ordem],
    )

    deslocamento = random.Random(0)  # só para espalhar os pontos no gráfico (a nota é inteira)
    notas_com_ruido = [nota + deslocamento.uniform(-0.18, 0.18) for nota in alvo]
    graficos.dispersao_real_vs_previsto(
        os.path.join(PASTA_FIGURAS, "regressao_real_vs_previsto.svg"),
        "Qualidade real × prevista (previsões de cada fold de teste)",
        [(nome, notas_com_ruido, previsoes[nome].tolist())
         for nome in ("kNN (Euclidiana)", "Regressão Linear Múltipla")],
        minimo=alvo.min() - 1, maximo=alvo.max() + 1,
    )

    return {
        "amostras": int(atributos.shape[0]),
        "atributos": int(atributos.shape[1]),
        "distribuicao_das_notas": distribuicao,
        "mae_prevendo_a_media": float(np.mean(np.abs(alvo - alvo.mean()))),
        "modelos": resumo,
        "acerto_com_tolerancia": {
            nome: {
                "0.5": taxa_de_acerto_com_tolerancia(alvo, previsoes[nome], 0.5),
                "1.0": taxa_de_acerto_com_tolerancia(alvo, previsoes[nome], 1.0),
            }
            for nome in modelos
        },
        "erro_por_nota": {nome: erro_medio_por_nota(alvo, previsoes[nome]) for nome in modelos},
        "coeficientes": [[nomes[i], float(coeficientes[i])] for i in ordem],
        "valores_de_k": VALORES_DE_K,
        "varredura_de_k": varredura,
    }


def main():
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    resultados = {"credito": analisar_credito(), "vinho": analisar_vinho()}
    with open(os.path.join(PASTA_RESULTADOS, "resultados.json"), "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {os.path.abspath(PASTA_RESULTADOS)}")


if __name__ == "__main__":
    main()
