"""SalesInsight PY: analise e visualizacao de vendas de varejo.
Mini-projeto educacional, Modulo 1, Semana 07.
Execute: python salesinsight.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # permite salvar figuras mesmo sem interface grafica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
GRAFICOS_DIR = OUTPUT_DIR / "graficos"
MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}
CORES = {
    "azul": "#205D86", "azul_claro": "#57A5C5", "laranja": "#E49F46",
    "verde": "#329475", "roxo": "#7965A8", "rosa": "#C46D87",
    "cinza": "#61717D", "fundo": "#F7F9FB",
}
CATEGORIA_CORES = {
    "Computadores": CORES["azul"],
    "Celulares": CORES["laranja"],
    "Perifericos": CORES["verde"],
}


def gerar_dataset_vendas(n_registros: int = 200, seed: int = 42) -> pd.DataFrame:
    """Gera 200 vendas sinteticas (por padrao), com datas invalidas intencionais."""
    random.seed(seed)
    np.random.seed(seed)

    produtos = ["Notebook", "Smartphone", "Tablet", "Monitor", "Teclado", "Mouse", "Headset"]
    categorias = {
        "Notebook": "Computadores", "Smartphone": "Celulares", "Tablet": "Celulares",
        "Monitor": "Computadores", "Teclado": "Perifericos", "Mouse": "Perifericos",
        "Headset": "Perifericos",
    }
    precos = {
        "Notebook": 3500, "Smartphone": 2200, "Tablet": 1800, "Monitor": 1200,
        "Teclado": 250, "Mouse": 120, "Headset": 350,
    }
    regioes = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]

    data_inicio = datetime(2025, 1, 1)
    dados = []
    for i in range(n_registros):
        produto = random.choice(produtos)
        categoria = categorias[produto]
        quantidade = random.randint(1, 10)
        preco = round(precos[produto] * random.uniform(0.85, 1.15), 2)
        data = data_inicio + timedelta(days=random.randint(0, 364))
        data_txt = data.strftime("%Y-%m-%d")
        cliente = f"Cliente_{random.randint(1, 50):03d}"
        if random.random() < 0.03:
            data_txt = "DATA INVALIDA"
        dados.append({
            "id_venda": i + 1, "data_venda": data_txt, "cliente": cliente,
            "produto": produto, "categoria": categoria, "regiao": random.choice(regioes),
            "quantidade": quantidade, "preco_unitario": preco,
        })
    return pd.DataFrame(dados)


def inspecionar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Imprime estrutura, tipos, nulos e amostra inicial do dataset."""
    print("\n=== INSPECAO INICIAL DO DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    print(f"Tipos de dados:\n{df.dtypes}")
    print(f"Valores nulos por coluna:\n{df.isnull().sum()}")
    print(f"Primeiros registros:\n{df.head()}")
    return df


def tratar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normaliza textos/tipos e descarta datas invalidas; retorna relatorio."""
    resultado = df.copy()
    entrada = len(resultado)
    linhas_texto_ajustadas = pd.Series(False, index=resultado.index)
    for coluna in resultado.select_dtypes(include=["object", "string"]).columns:
        antes = resultado[coluna].astype("string")
        depois = antes.str.strip()
        linhas_texto_ajustadas |= antes.ne(depois).fillna(False)
        resultado[coluna] = depois

    # Registros com data invalida nao podem ser atribuidos a um mes.
    resultado["data_venda"] = pd.to_datetime(resultado["data_venda"], errors="coerce")
    datas_invalidas = int(resultado["data_venda"].isna().sum())
    resultado = resultado.dropna(subset=["data_venda"]).copy()

    quantidade_antes = resultado["quantidade"].dtype
    preco_antes = resultado["preco_unitario"].dtype
    resultado["quantidade"] = pd.to_numeric(resultado["quantidade"], errors="raise").astype(int)
    resultado["preco_unitario"] = pd.to_numeric(resultado["preco_unitario"], errors="raise").astype(float)
    qtde_convertidas = len(resultado) if quantidade_antes != resultado["quantidade"].dtype else 0
    preco_convertidos = len(resultado) if preco_antes != resultado["preco_unitario"].dtype else 0

    relatorio = {
        "registros_iniciais": entrada,
        "datas_invalidas_removidas": datas_invalidas,
        "linhas_com_espacos_ajustados": int(linhas_texto_ajustadas.sum()),
        "registros_com_quantidade_convertida": qtde_convertidas,
        "registros_com_preco_convertido": preco_convertidos,
        "datas_validas_convertidas": len(resultado),
        "registros_finais": len(resultado),
        "registros_sem_alteracoes_de_texto_ou_remocao": entrada - datas_invalidas - int(linhas_texto_ajustadas.loc[resultado.index].sum()),
    }
    print("\n=== RELATORIO DE LIMPEZA ===")
    for nome, valor in relatorio.items():
        print(f"{nome}: {valor}")
    return resultado, relatorio


def processar_coluna(
    df: pd.DataFrame, coluna: str, funcao_transformacao, nome_saida: str | None = None
) -> pd.DataFrame:
    """Recebe uma funcao como argumento e aplica-a a coluna do DataFrame."""
    destino = nome_saida or f"{coluna}_transformado"
    df[destino] = df[coluna].apply(funcao_transformacao)
    return df


def criar_colunas_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona receita, mes, trimestre, ano e classificacoes condicionais."""
    resultado = df.copy()
    resultado["receita_total"] = resultado["quantidade"] * resultado["preco_unitario"]
    resultado["mes"] = resultado["data_venda"].dt.month
    resultado["mes_nome"] = resultado["mes"].map(MESES_PT)
    resultado["trimestre"] = "Q" + resultado["data_venda"].dt.quarter.astype(str)
    resultado["ano"] = resultado["data_venda"].dt.year

    condicoes = [
        resultado["receita_total"] < 500,
        (resultado["receita_total"] >= 500) & (resultado["receita_total"] < 5000),
        resultado["receita_total"] >= 5000,
    ]
    faixas = ["Baixo Valor", "Medio Valor", "Alto Valor"]
    resultado["faixa_receita_item"] = "Nao Classificado"
    for condicao, faixa in zip(condicoes, faixas):
        resultado.loc[condicao, "faixa_receita_item"] = faixa

    # Dois contextos de uso de funcao lambda, com funcao de ordem superior.
    resultado = processar_coluna(
        resultado, "receita_total", lambda valor: round(valor / 1000, 2),
        nome_saida="receita_em_milhares",
    )
    resultado = processar_coluna(
        resultado, "quantidade", lambda valor: "Alto Volume" if valor > 5 else "Baixo Volume",
        nome_saida="perfil_volume",
    )
    return resultado


def calcular_metricas(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Consolida receita e volume por mes, produto, categoria e regiao."""
    por_mes = (
        df.groupby(["ano", "mes", "mes_nome"], as_index=False)
        .agg(receita_total=("receita_total", "sum"), quantidade=("quantidade", "sum"), n_vendas=("id_venda", "count"))
        .sort_values(["ano", "mes"])
    )
    top_produtos = (
        df.groupby("produto", as_index=False)["receita_total"]
        .sum().sort_values("receita_total", ascending=False).head(5)
    )
    por_categoria = (
        df.groupby("categoria", as_index=False)["receita_total"]
        .sum().sort_values("receita_total", ascending=False)
    )
    por_regiao = (
        df.groupby("regiao", as_index=False)
        .agg(receita_total=("receita_total", "sum"), ticket_medio=("receita_total", "mean"))
        .sort_values("receita_total", ascending=False)
    )
    por_trimestre = (
        df.groupby(["ano", "trimestre"], as_index=False)
        .agg(receita_total=("receita_total", "sum"), n_vendas=("id_venda", "count"))
        .sort_values(["ano", "trimestre"])
    )
    metricas = {
        "por_mes": por_mes, "top_produtos": top_produtos,
        "por_categoria": por_categoria, "por_regiao": por_regiao,
        "por_trimestre": por_trimestre,
    }
    for nome, tabela in metricas.items():
        print(f"\n=== {nome.upper().replace('_', ' ')} ===")
        print(tabela.to_string(index=False, float_format=lambda valor: f"{valor:,.2f}"))
    return metricas


def segmentar_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Soma os gastos por cliente e classifica em Bronze, Prata ou Ouro."""
    clientes = (
        df.groupby("cliente", as_index=False)["receita_total"]
        .sum().rename(columns={"receita_total": "total_gasto"})
    )
    clientes["segmento"] = clientes["total_gasto"].apply(
        lambda valor: "Bronze" if valor < 5000 else ("Prata" if valor <= 15000 else "Ouro")
    )
    clientes = clientes.sort_values("total_gasto", ascending=False).reset_index(drop=True)
    print("\n=== TOP 10 CLIENTES ===")
    print(clientes.head(10).to_string(index=False))
    print("\n=== DISTRIBUICAO POR SEGMENTO ===")
    print(clientes["segmento"].value_counts().to_string())
    return clientes


def _estilo_eixo(ax: plt.Axes, titulo: str, xlabel: str, ylabel: str) -> None:
    """Aplica um estilo consistente aos eixos das figuras."""
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.set_facecolor(CORES["fundo"])


def _grafico_meses(ax: plt.Axes, por_mes: pd.DataFrame) -> None:
    """Desenha a evolucao da receita mensal em um eixo fornecido."""
    ax.plot(por_mes["mes"], por_mes["receita_total"], marker="o", linewidth=2.5,
            color=CORES["azul"], label="Receita mensal")
    ax.set_xticks(range(1, 13))
    _estilo_eixo(ax, "Evolucao da receita mensal", "Mes de 2025", "Receita (R$)")
    ax.legend()


def _grafico_top(ax: plt.Axes, top: pd.DataFrame) -> None:
    """Desenha as barras horizontais dos cinco produtos com maior receita."""
    ordenados = top.sort_values("receita_total")
    ax.barh(ordenados["produto"], ordenados["receita_total"], color=CORES["azul_claro"])
    _estilo_eixo(ax, "Top 5 produtos por receita", "Receita (R$)", "Produto")


def _grafico_dispersao(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Desenha quantidade vs receita com cores distintas por categoria."""
    for categoria, grupo in df.groupby("categoria"):
        ax.scatter(grupo["quantidade"], grupo["receita_total"],
                   color=CATEGORIA_CORES.get(categoria, CORES["roxo"]),
                   label=categoria, alpha=0.72, s=32, edgecolor="white", linewidth=0.3)
    _estilo_eixo(ax, "Quantidade x receita", "Quantidade vendida", "Receita por venda (R$)")
    ax.legend(title="Categoria", fontsize=8)


def _grafico_regioes(ax: plt.Axes, por_regiao: pd.DataFrame) -> None:
    """Desenha a receita agregada pelas cinco regioes."""
    ax.bar(por_regiao["regiao"], por_regiao["receita_total"], color=CORES["verde"])
    _estilo_eixo(ax, "Receita por regiao", "Regiao", "Receita (R$)")
    ax.tick_params(axis="x", labelrotation=20)


def criar_visualizacoes(df: pd.DataFrame, metricas: dict[str, pd.DataFrame]) -> None:
    """Gera quatro figuras obrigatorias em PNG com resolucao de 150 DPI."""
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    _grafico_meses(ax, metricas["por_mes"])
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "receita_por_mes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    _grafico_top(ax, metricas["top_produtos"])
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "top_produtos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    _grafico_dispersao(ax, df)
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "quantidade_vs_receita.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _grafico_meses(axes[0, 0], metricas["por_mes"])
    _grafico_top(axes[0, 1], metricas["top_produtos"])
    _grafico_dispersao(axes[1, 0], df)
    _grafico_regioes(axes[1, 1], metricas["por_regiao"])
    fig.suptitle("SalesInsight PY | Painel Resumo 2025", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(GRAFICOS_DIR / "painel_resumo.png", dpi=150)
    plt.close(fig)
    print(f"\nGraficos salvos em: {GRAFICOS_DIR}")


def calcular_estatisticas(df: pd.DataFrame, relatorio: dict) -> dict:
    """Resume os indicadores gerais de vendas para exportacao em JSON."""
    return {
        "registros_brutos": int(relatorio["registros_iniciais"]),
        "registros_validos": len(df),
        "datas_invalidas_removidas": int(relatorio["datas_invalidas_removidas"]),
        "receita_total": round(float(df["receita_total"].sum()), 2),
        "total_itens_vendidos": int(df["quantidade"].sum()),
        "ticket_medio_por_venda": round(float(df["receita_total"].mean()), 2),
        "total_clientes": int(df["cliente"].nunique()),
        "total_produtos": int(df["produto"].nunique()),
        "total_regioes": int(df["regiao"].nunique()),
    }


def exportar_resultados(metricas: dict[str, pd.DataFrame], clientes: pd.DataFrame, estatisticas: dict) -> None:
    """Salva CSV e JSON e reabre o JSON para comprovar a gravacao."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metricas["por_mes"].to_csv(OUTPUT_DIR / "metricas_por_mes.csv", index=False, encoding="utf-8-sig")
    clientes.to_csv(OUTPUT_DIR / "segmentacao_clientes.csv", index=False, encoding="utf-8-sig")
    destino = OUTPUT_DIR / "estatisticas_gerais.json"
    with destino.open("w", encoding="utf-8") as f:
        json.dump(estatisticas, f, indent=4, ensure_ascii=False)
    with destino.open("r", encoding="utf-8") as f:
        conferencia = json.load(f)
    print(f"\nJSON gravado e lido novamente: {conferencia}")


class AnalisadorDeVendas:
    """Organiza as etapas do projeto em um fluxo reutilizavel."""

    def __init__(self) -> None:
        self.df_bruto: pd.DataFrame | None = None
        self.df_limpo: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.relatorio: dict = {}
        self.metricas: dict[str, pd.DataFrame] = {}
        self.clientes: pd.DataFrame | None = None
        self.estatisticas: dict = {}

    def carregar(self, caminho: Path) -> pd.DataFrame:
        """Le e guarda as vendas em CSV."""
        self.df_bruto = pd.read_csv(caminho)
        return self.df_bruto

    def transformar(self) -> pd.DataFrame:
        """Limpa datas/tipos e deriva colunas analiticas."""
        if self.df_bruto is None:
            raise ValueError("Carregue os dados antes de transformar.")
        self.df_limpo, self.relatorio = tratar_dados(self.df_bruto)
        self.df = criar_colunas_derivadas(self.df_limpo)
        return self.df

    def analisar(self) -> dict[str, pd.DataFrame]:
        """Calcula metricas, segmentos de clientes e estatisticas gerais."""
        if self.df is None:
            raise ValueError("Transforme os dados antes de analisar.")
        self.metricas = calcular_metricas(self.df)
        self.clientes = segmentar_clientes(self.df)
        self.estatisticas = calcular_estatisticas(self.df, self.relatorio)
        return self.metricas

    def visualizar(self) -> None:
        """Salva os graficos obrigatorios."""
        if self.df is None or not self.metricas:
            raise ValueError("Analise os dados antes de visualizar.")
        criar_visualizacoes(self.df, self.metricas)

    def resumo(self) -> dict:
        """Exporta resultados e exibe uma sintese do projeto."""
        if self.clientes is None:
            raise ValueError("Analise os clientes antes de exportar.")
        exportar_resultados(self.metricas, self.clientes, self.estatisticas)
        print("\n=== RESUMO GERAL ===")
        for nome, valor in self.estatisticas.items():
            print(f"{nome}: {valor}")
        return self.estatisticas


def main() -> None:
    """Executa o projeto completo do carregamento ate a exportacao."""
    print("=" * 60)
    print("  SALESINSIGHT PY - ANALISE DE DADOS DE VENDAS")
    print("=" * 60)
    caminho = BASE_DIR / "vendas.csv"
    if not caminho.exists():
        gerar_dataset_vendas().to_csv(caminho, index=False, encoding="utf-8-sig")
        print(f"Dataset sintetico gerado em {caminho}")
    analisador = AnalisadorDeVendas()
    analisador.carregar(caminho)
    inspecionar_dados(analisador.df_bruto)
    analisador.transformar()
    analisador.analisar()
    analisador.visualizar()
    analisador.resumo()
    print("\n[CONCLUIDO] Fluxo completo executado com sucesso.")


if __name__ == "__main__":
    main()
