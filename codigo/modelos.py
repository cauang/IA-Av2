import numpy as np


def calcular_distancias(amostras, base, tipo):
    """Matriz (len(amostras) x len(base)) de distâncias entre cada par de pontos."""
    if tipo == "euclidiana":
        quadrados = (
            (amostras ** 2).sum(axis=1)[:, None]
            + (base ** 2).sum(axis=1)[None, :]
            - 2 * amostras @ base.T
        )
        return np.sqrt(np.maximum(quadrados, 0.0))
    if tipo == "manhattan":
        return np.array([np.abs(base - amostra).sum(axis=1) for amostra in amostras])
    raise ValueError(f"distância desconhecida: {tipo}")


class _KNN:
    def __init__(self, k=5, distancia="euclidiana"):
        self.k = k
        self.distancia = distancia

    def _indices_dos_vizinhos(self, atributos, quantidade):
        distancias = calcular_distancias(atributos, self.atributos_treino, self.distancia)
        return np.argsort(distancias, axis=1)[:, :quantidade]

    def predict(self, atributos):
        vizinhos = self._indices_dos_vizinhos(atributos, self.k)
        return self._agregar(vizinhos, self.k)

    def prever_varios_k(self, atributos, lista_k):
        """Calcula as distâncias uma vez e devolve {k: previsões}."""
        vizinhos = self._indices_dos_vizinhos(atributos, max(lista_k))
        return {k: self._agregar(vizinhos, k) for k in lista_k}


class KNNClassificador(_KNN):
    def fit(self, atributos, alvo):
        self.atributos_treino = atributos
        self.alvo_treino = alvo.astype(int)
        self.classes = np.unique(self.alvo_treino)
        return self

    def _agregar(self, vizinhos, k):
        rotulos = self.alvo_treino[vizinhos[:, :k]]
        votos = np.stack([(rotulos == classe).sum(axis=1) for classe in self.classes], axis=1)
        return self.classes[votos.argmax(axis=1)]


class KNNRegressor(_KNN):
    def fit(self, atributos, alvo):
        self.atributos_treino = atributos
        self.alvo_treino = alvo
        return self

    def _agregar(self, vizinhos, k):
        return self.alvo_treino[vizinhos[:, :k]].mean(axis=1)


class BayesUnivariado:
    """Bayes ingênuo com um modelo univariado por atributo e por classe.

    Colunas contínuas usam uma gaussiana; colunas 0/1 (categorias codificadas) usam
    a frequência da categoria na classe, com suavização de Laplace.
    """

    def __init__(self, colunas_binarias=None, variancia_minima=1e-9):
        self.colunas_binarias = colunas_binarias
        self.variancia_minima = variancia_minima

    def fit(self, atributos, alvo):
        alvo = alvo.astype(int)
        if self.colunas_binarias is None:
            self.colunas_binarias = np.zeros(atributos.shape[1], dtype=bool)
        self.classes = np.unique(alvo)
        self.log_prioris = np.log([np.mean(alvo == classe) for classe in self.classes])

        por_classe = [atributos[alvo == classe] for classe in self.classes]
        self.medias = np.array([amostras.mean(axis=0) for amostras in por_classe])
        self.variancias = np.array([amostras.var(axis=0) for amostras in por_classe]) + self.variancia_minima
        self.frequencias = np.array([
            (amostras.sum(axis=0) + 1) / (len(amostras) + 2) for amostras in por_classe
        ])
        return self

    def predict(self, atributos):
        continuas = ~self.colunas_binarias
        log_posterioris = []
        for i in range(len(self.classes)):
            log_gaussiana = -0.5 * (
                np.log(2 * np.pi * self.variancias[i, continuas])
                + (atributos[:, continuas] - self.medias[i, continuas]) ** 2 / self.variancias[i, continuas]
            )
            frequencia = self.frequencias[i, self.colunas_binarias]
            log_categorias = (
                atributos[:, self.colunas_binarias] * np.log(frequencia)
                + (1 - atributos[:, self.colunas_binarias]) * np.log(1 - frequencia)
            )
            log_posterioris.append(
                self.log_prioris[i] + log_gaussiana.sum(axis=1) + log_categorias.sum(axis=1)
            )
        return self.classes[np.argmax(np.stack(log_posterioris, axis=1), axis=1)]


class RegressaoLinearMultipla:
    """Mínimos quadrados pelas equações normais, com pseudo-inversa por estabilidade."""

    @staticmethod
    def _com_intercepto(atributos):
        return np.hstack([np.ones((len(atributos), 1)), atributos])

    def fit(self, atributos, alvo):
        matriz = self._com_intercepto(atributos)
        self.coeficientes = np.linalg.pinv(matriz.T @ matriz) @ matriz.T @ alvo
        return self

    def predict(self, atributos):
        return self._com_intercepto(atributos) @ self.coeficientes
