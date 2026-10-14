import random

import numpy as np


def validacao_cruzada(alvo, quantidade_folds=5, estratificada=False, semente=42):
    """Gera pares (indices_treino, indices_teste), um por fold."""
    sorteio = random.Random(semente)  # numpy.random está bloqueado nesta máquina

    if estratificada:
        folds = [[] for _ in range(quantidade_folds)]
        for classe in np.unique(alvo):
            indices_da_classe = np.where(alvo == classe)[0].tolist()
            sorteio.shuffle(indices_da_classe)
            for posicao, indice in enumerate(indices_da_classe):
                folds[posicao % quantidade_folds].append(indice)
        folds = [np.array(fold) for fold in folds]
    else:
        indices = list(range(len(alvo)))
        sorteio.shuffle(indices)
        folds = np.array_split(np.array(indices), quantidade_folds)

    for numero_do_fold in range(quantidade_folds):
        indices_teste = folds[numero_do_fold]
        indices_treino = np.concatenate(
            [fold for outro, fold in enumerate(folds) if outro != numero_do_fold]
        )
        yield indices_treino, indices_teste
