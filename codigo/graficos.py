"""Gráficos em SVG escritos à mão, já que matplotlib está bloqueado nesta máquina."""
import math
from xml.sax.saxutils import escape

AZUL, TINTA, AZUL_CLARO, CINZA = "#2563EB", "#0F172A", "#60A5FA", "#64748B"
GRADE = "#E2E8F0"
CORES = [AZUL, TINTA, AZUL_CLARO, CINZA]
FONTE = "Segoe UI, Arial, sans-serif"
LARGURA, ALTURA = 760, 420
MARGEM_ESQUERDA, MARGEM_DIREITA, MARGEM_TOPO, MARGEM_BASE = 70, 20, 50, 70


def _moldura_svg(titulo, conteudo, largura=LARGURA, altura=ALTURA):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {largura} {altura}" '
        f'width="{largura}" height="{altura}" font-family="{FONTE}" font-size="12" fill="{TINTA}">'
        f'<rect width="{largura}" height="{altura}" fill="white"/>'
        f'<text x="{largura / 2}" y="26" text-anchor="middle" font-size="15" '
        f'font-weight="600">{escape(titulo)}</text>{conteudo}</svg>'
    )


def _gravar(caminho, svg):
    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(svg)


def _legenda(nomes, x_inicial, y=40):
    partes = []
    x = x_inicial
    for indice, nome in enumerate(nomes):
        cor = CORES[indice % len(CORES)]
        partes.append(
            f'<rect x="{x}" y="{y - 9}" width="11" height="11" fill="{cor}"/>'
            f'<text x="{x + 15}" y="{y}">{escape(nome)}</text>'
        )
        x += 25 + 7 * len(nome)
    return "".join(partes)


def _area_do_grafico():
    return LARGURA - MARGEM_ESQUERDA - MARGEM_DIREITA, ALTURA - MARGEM_TOPO - MARGEM_BASE


def _passo_redondo(valor_aproximado):
    """Arredonda para 1, 2 ou 5 vezes uma potência de dez."""
    potencia = 10 ** math.floor(math.log10(valor_aproximado))
    for multiplo in (1, 2, 5, 10):
        if valor_aproximado <= multiplo * potencia:
            return multiplo * potencia


def barras_agrupadas(caminho, titulo, categorias, series, rotulo_y="", escala_log=False):
    """`series` é {nome: (medias, desvios)}, com um valor por categoria."""
    largura, altura = _area_do_grafico()
    medias_todas = [m for medias, _ in series.values() for m in medias]
    topos = [m + d for medias, desvios in series.values() for m, d in zip(medias, desvios)]

    if escala_log:
        minimo = 10 ** math.floor(math.log10(min(v for v in medias_todas if v > 0)))
        maximo = 10 ** math.ceil(math.log10(max(topos)))
        faixa = math.log10(maximo) - math.log10(minimo)

        def altura_da_barra(valor):
            return altura * (math.log10(max(valor, minimo)) - math.log10(minimo)) / faixa

        marcas = [10 ** e for e in range(round(math.log10(minimo)), round(math.log10(maximo)) + 1)]
    else:
        minimo = 0
        passo = _passo_redondo(max(topos) / 5)
        maximo = passo * math.ceil(max(topos) * 1.05 / passo)

        def altura_da_barra(valor):
            return altura * valor / maximo

        marcas = [passo * i for i in range(round(maximo / passo) + 1)]

    svg = []
    for marca in marcas:
        y = MARGEM_TOPO + altura - altura_da_barra(marca)
        svg.append(
            f'<line x1="{MARGEM_ESQUERDA}" x2="{LARGURA - MARGEM_DIREITA}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRADE}"/>'
            f'<text x="{MARGEM_ESQUERDA - 6}" y="{y + 4:.1f}" text-anchor="end">{marca:.3g}</text>'
        )

    largura_grupo = largura / len(categorias)
    largura_barra = largura_grupo * 0.8 / len(series)
    for i_categoria, categoria in enumerate(categorias):
        x_grupo = MARGEM_ESQUERDA + i_categoria * largura_grupo + largura_grupo * 0.1
        for i_serie, (_, (medias, desvios)) in enumerate(series.items()):
            x = x_grupo + i_serie * largura_barra
            altura_barra = altura_da_barra(medias[i_categoria])
            svg.append(
                f'<rect x="{x:.1f}" y="{MARGEM_TOPO + altura - altura_barra:.1f}" '
                f'width="{largura_barra - 2:.1f}" height="{altura_barra:.1f}" fill="{CORES[i_serie % len(CORES)]}"/>'
            )
            if desvios[i_categoria] > 0:
                x_centro = x + (largura_barra - 2) / 2
                y_alto = MARGEM_TOPO + altura - altura_da_barra(medias[i_categoria] + desvios[i_categoria])
                y_baixo = MARGEM_TOPO + altura - altura_da_barra(max(medias[i_categoria] - desvios[i_categoria], minimo))
                svg.append(
                    f'<line x1="{x_centro:.1f}" x2="{x_centro:.1f}" y1="{y_alto:.1f}" y2="{y_baixo:.1f}" stroke="{TINTA}"/>'
                    f'<line x1="{x_centro - 3:.1f}" x2="{x_centro + 3:.1f}" y1="{y_alto:.1f}" y2="{y_alto:.1f}" stroke="{TINTA}"/>'
                )
        x_texto = MARGEM_ESQUERDA + i_categoria * largura_grupo + largura_grupo / 2
        svg.append(f'<text x="{x_texto:.1f}" y="{ALTURA - MARGEM_BASE + 20}" text-anchor="middle">{escape(categoria)}</text>')

    svg.append(
        f'<line x1="{MARGEM_ESQUERDA}" x2="{MARGEM_ESQUERDA}" y1="{MARGEM_TOPO}" y2="{MARGEM_TOPO + altura}" stroke="{CINZA}"/>'
        f'<line x1="{MARGEM_ESQUERDA}" x2="{LARGURA - MARGEM_DIREITA}" y1="{MARGEM_TOPO + altura}" y2="{MARGEM_TOPO + altura}" stroke="{CINZA}"/>'
    )
    if rotulo_y:
        svg.append(
            f'<text transform="translate(16 {MARGEM_TOPO + altura / 2}) rotate(-90)" text-anchor="middle">{escape(rotulo_y)}</text>'
        )
    svg.append(_legenda(list(series), MARGEM_ESQUERDA))
    _gravar(caminho, _moldura_svg(titulo, "".join(svg)))


def linhas(caminho, titulo, valores_x, series, rotulo_x="", rotulo_y=""):
    largura, altura = _area_do_grafico()
    todos_os_valores = [v for valores in series.values() for v in valores]
    folga = (max(todos_os_valores) - min(todos_os_valores)) * 0.1 or 0.01
    minimo, maximo = min(todos_os_valores) - folga, max(todos_os_valores) + folga

    def coordenada_x(x):
        return MARGEM_ESQUERDA + largura * (x - min(valores_x)) / (max(valores_x) - min(valores_x))

    def coordenada_y(valor):
        return MARGEM_TOPO + altura - altura * (valor - minimo) / (maximo - minimo)

    svg = []
    for i in range(6):
        valor = minimo + (maximo - minimo) * i / 5
        y = coordenada_y(valor)
        svg.append(
            f'<line x1="{MARGEM_ESQUERDA}" x2="{LARGURA - MARGEM_DIREITA}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRADE}"/>'
            f'<text x="{MARGEM_ESQUERDA - 6}" y="{y + 4:.1f}" text-anchor="end">{valor:.3f}</text>'
        )
    for x in valores_x:
        svg.append(f'<text x="{coordenada_x(x):.1f}" y="{ALTURA - MARGEM_BASE + 18}" text-anchor="middle">{x}</text>')

    for i_serie, valores in enumerate(series.values()):
        cor = CORES[i_serie % len(CORES)]
        pontos = " ".join(f"{coordenada_x(x):.1f},{coordenada_y(v):.1f}" for x, v in zip(valores_x, valores))
        svg.append(f'<polyline points="{pontos}" fill="none" stroke="{cor}" stroke-width="2"/>')
        svg += [
            f'<circle cx="{coordenada_x(x):.1f}" cy="{coordenada_y(v):.1f}" r="3.5" fill="{cor}"/>'
            for x, v in zip(valores_x, valores)
        ]

    svg.append(
        f'<text x="{MARGEM_ESQUERDA + largura / 2}" y="{ALTURA - 20}" text-anchor="middle">{escape(rotulo_x)}</text>'
        f'<text transform="translate(16 {MARGEM_TOPO + altura / 2}) rotate(-90)" text-anchor="middle">{escape(rotulo_y)}</text>'
    )
    svg.append(_legenda(list(series), MARGEM_ESQUERDA))
    _gravar(caminho, _moldura_svg(titulo, "".join(svg)))


def _misturar_com_branco(cor_hex, intensidade):
    canais = [int(cor_hex[posicao:posicao + 2], 16) for posicao in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - canal) * intensidade):02X}" for canal in canais)


def mapa_de_calor(caminho, titulo, matriz, rotulos):
    tamanho_celula, origem_x, origem_y = 80, 130, 80
    quantidade = len(rotulos)

    svg = []
    for i in range(quantidade):
        for j in range(quantidade):
            valor = matriz[i][j]
            fracao_da_linha = valor / sum(matriz[i])
            if i == j:
                cor_celula, intensidade = AZUL, fracao_da_linha
            else:
                cor_celula, intensidade = CINZA, min(1, fracao_da_linha * 3)
            cor_texto = "white" if i == j and fracao_da_linha > 0.6 else TINTA
            x, y = origem_x + j * tamanho_celula, origem_y + i * tamanho_celula
            svg.append(
                f'<rect x="{x}" y="{y}" width="{tamanho_celula}" height="{tamanho_celula}" '
                f'fill="{_misturar_com_branco(cor_celula, intensidade)}" stroke="white"/>'
                f'<text x="{x + tamanho_celula / 2}" y="{y + tamanho_celula / 2 + 5}" text-anchor="middle" '
                f'fill="{cor_texto}" font-size="15">{valor}</text>'
            )
        svg.append(
            f'<text x="{origem_x - 8}" y="{origem_y + i * tamanho_celula + tamanho_celula / 2 + 4}" text-anchor="end">{escape(rotulos[i])}</text>'
            f'<text x="{origem_x + i * tamanho_celula + tamanho_celula / 2}" y="{origem_y - 8}" text-anchor="middle">{escape(rotulos[i])}</text>'
        )
    svg.append(
        f'<text x="{origem_x + quantidade * tamanho_celula / 2}" y="{origem_y + quantidade * tamanho_celula + 24}" '
        f'text-anchor="middle" fill="{CINZA}">Previsto (colunas) × Real (linhas)</text>'
    )
    _gravar(caminho, _moldura_svg(
        titulo, "".join(svg),
        largura=origem_x + quantidade * tamanho_celula + 30,
        altura=origem_y + quantidade * tamanho_celula + 40,
    ))


def barras_horizontais(caminho, titulo, rotulos, valores):
    altura_total, margem_rotulos = 28 * len(rotulos) + 80, 170
    largura_util = LARGURA - margem_rotulos - 40
    maior_modulo = max(abs(v) for v in valores) * 1.1
    x_zero = margem_rotulos + largura_util / 2

    svg = [f'<line x1="{x_zero}" x2="{x_zero}" y1="50" y2="{altura_total - 20}" stroke="{CINZA}"/>']
    for i, (rotulo, valor) in enumerate(zip(rotulos, valores)):
        y = 56 + i * 28
        largura_barra = abs(valor) / maior_modulo * largura_util / 2
        x = x_zero if valor >= 0 else x_zero - largura_barra
        cor = AZUL if valor >= 0 else TINTA
        x_valor = x + largura_barra + 4 if valor >= 0 else x - 4
        alinhamento = "start" if valor >= 0 else "end"
        svg.append(
            f'<rect x="{x:.1f}" y="{y}" width="{largura_barra:.1f}" height="20" fill="{cor}"/>'
            f'<text x="{margem_rotulos - 8}" y="{y + 14}" text-anchor="end">{escape(rotulo)}</text>'
            f'<text x="{x_valor:.1f}" y="{y + 14}" text-anchor="{alinhamento}">{valor:+.3f}</text>'
        )
    _gravar(caminho, _moldura_svg(titulo, "".join(svg), altura=altura_total))


def dispersao_real_vs_previsto(caminho, titulo, paineis, minimo, maximo):
    """`paineis` é uma lista de (nome, valores_reais, valores_previstos)."""
    largura_painel, altura_painel, espaco = 330, 300, 60
    largura_total = MARGEM_ESQUERDA + len(paineis) * (largura_painel + espaco)

    svg = []
    for i_painel, (nome, reais, previstos) in enumerate(paineis):
        origem_x, origem_y = MARGEM_ESQUERDA + i_painel * (largura_painel + espaco), 60

        def coordenada_x(valor):
            return origem_x + largura_painel * (valor - minimo) / (maximo - minimo)

        def coordenada_y(valor):
            return origem_y + altura_painel - altura_painel * (valor - minimo) / (maximo - minimo)

        svg.append(
            f'<rect x="{origem_x}" y="{origem_y}" width="{largura_painel}" height="{altura_painel}" fill="none" stroke="{GRADE}"/>'
            f'<line x1="{coordenada_x(minimo)}" y1="{coordenada_y(minimo)}" x2="{coordenada_x(maximo)}" y2="{coordenada_y(maximo)}" '
            f'stroke="{CINZA}" stroke-dasharray="5"/>'
            f'<text x="{origem_x + largura_painel / 2}" y="{origem_y - 8}" text-anchor="middle" font-weight="600">{escape(nome)}</text>'
            f'<text x="{origem_x + largura_painel / 2}" y="{origem_y + altura_painel + 32}" text-anchor="middle">Qualidade real</text>'
        )
        svg.append("".join(
            f'<circle cx="{coordenada_x(real):.1f}" cy="{coordenada_y(min(max(previsto, minimo), maximo)):.1f}" '
            f'r="1.6" fill="{AZUL}" fill-opacity="0.25"/>'
            for real, previsto in zip(reais, previstos)
        ))
        for nota in range(int(minimo), int(maximo) + 1):
            svg.append(
                f'<text x="{coordenada_x(nota):.1f}" y="{origem_y + altura_painel + 14}" text-anchor="middle">{nota}</text>'
                f'<text x="{origem_x - 6}" y="{coordenada_y(nota) + 4:.1f}" text-anchor="end">{nota}</text>'
            )
    svg.append(f'<text transform="translate(16 {60 + altura_painel / 2}) rotate(-90)" text-anchor="middle">Qualidade prevista</text>')
    _gravar(caminho, _moldura_svg(titulo, "".join(svg), largura=largura_total, altura=60 + altura_painel + 50))
