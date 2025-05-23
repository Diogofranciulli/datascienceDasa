"""
DATA SCIENCE AND STATISTICAL COMPUTING - SPRINT 1
Projeto: Detecção de Anomalias em Registros de Consumo de Materiais
Objetivo: Identificar inconsistências que impactem a eficiência do estoque

Desenvolvido por: [

Arthur Cotrick Pagani - RM: 554510
Diogo Leles Franciulli - RM: 558487
Felipe Sousa De Oliveira - RM: 559085
Ryan Brito Pereira Ramos - RM: 554497
Vitor Chaves - RM: 557067
]
Data: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configurar backend não-interativo para evitar problemas com Tkinter
matplotlib.use('Agg')

# Configurações de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("🔬 DATA SCIENCE - DETECÇÃO DE ANOMALIAS EM CONSUMO DE MATERIAIS")
print("=" * 80)
print(f"Execução iniciada em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 80)

# =============================================================================
# 1. CRIAÇÃO DO CONJUNTO DE DADOS SIMULADOS
# =============================================================================
print("\n📊 ETAPA 1: CRIAÇÃO DO CONJUNTO DE DADOS SIMULADOS")
print("-" * 60)

# Definir seed para reprodutibilidade
np.random.seed(42)

# Parâmetros do projeto
TOTAL_DIAS = 200
DATA_INICIO = '2025-01-01'
CONSUMO_MEDIO = 50
CUSTO_UNITARIO = 10.00

# Gerar datas
dias = pd.date_range(start=DATA_INICIO, periods=TOTAL_DIAS, freq='D')

# Simular consumo normal (distribuição Poisson)
consumo_base = np.random.poisson(lam=CONSUMO_MEDIO, size=TOTAL_DIAS)

# Adicionar variações sazonais (maior consumo no meio do mês)
variacao_semanal = 5 * np.sin(2 * np.pi * np.arange(TOTAL_DIAS) / 7)
variacao_mensal = 10 * np.sin(2 * np.pi * np.arange(TOTAL_DIAS) / 30)

# Consumo com variações naturais
consumo_com_variacao = consumo_base + variacao_semanal + variacao_mensal

# Inserir anomalias propositais (outliers)
outliers_config = [
    {'posicao': 17, 'valor': 200, 'motivo': 'Demanda excepcional'},
    {'posicao': 45, 'valor': 180, 'motivo': 'Erro de registro'},
    {'posicao': 89, 'valor': 250, 'motivo': 'Pedido urgente'},
    {'posicao': 123, 'valor': 15, 'motivo': 'Falta de estoque'},
    {'posicao': 167, 'valor': 220, 'motivo': 'Compra emergencial'},
    {'posicao': 189, 'valor': 190, 'motivo': 'Reposição em lote'}
]

consumo_final = consumo_com_variacao.copy()
for outlier in outliers_config:
    if outlier['posicao'] < len(consumo_final):
        consumo_final[outlier['posicao']] = outlier['valor']

# Garantir valores não negativos
consumo_final = np.maximum(consumo_final, 0)

# Criar DataFrame principal
df = pd.DataFrame({
    'Data': dias,
    'Consumo': consumo_final.astype(int),
    'DiaSemana': dias.day_name(),
    'Mes': dias.month,
    'Semana': dias.isocalendar().week
})

print(f"✅ Dataset criado com sucesso:")
print(f"   • {len(df)} registros de consumo diário")
print(f"   • Período: {df['Data'].min().strftime('%d/%m/%Y')} a {df['Data'].max().strftime('%d/%m/%Y')}")
print(f"   • {len(outliers_config)} anomalias inseridas propositalmente")

# =============================================================================
# 2. ANÁLISES ESTATÍSTICAS PARA DETECÇÃO DE ANOMALIAS
# =============================================================================
print("\n📈 ETAPA 2: ANÁLISES ESTATÍSTICAS")
print("-" * 60)

# Estatísticas descritivas básicas
stats_basicas = df['Consumo'].describe()
print("📋 Estatísticas Descritivas Básicas:")
print(f"   • Média: {stats_basicas['mean']:.2f} unidades/dia")
print(f"   • Mediana: {stats_basicas['50%']:.2f} unidades/dia")
print(f"   • Desvio Padrão: {stats_basicas['std']:.2f}")
print(f"   • Amplitude: {stats_basicas['max'] - stats_basicas['min']:.0f} unidades")

# 2.1 MÉTODO Z-SCORE
print("\n🎯 Método 1: Z-Score")
df['Z_Score'] = (df['Consumo'] - df['Consumo'].mean()) / df['Consumo'].std()
df['Outlier_ZScore'] = df['Z_Score'].abs() > 3

outliers_zscore = df[df['Outlier_ZScore']]
print(f"   • Outliers detectados (|Z| > 3): {len(outliers_zscore)}")
print(f"   • Percentual: {len(outliers_zscore) / len(df) * 100:.1f}% do dataset")

# 2.2 MÉTODO IQR (INTERQUARTILE RANGE)
print("\n🎯 Método 2: IQR (Interquartile Range)")
Q1 = df['Consumo'].quantile(0.25)
Q3 = df['Consumo'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df['Outlier_IQR'] = (df['Consumo'] < limite_inferior) | (df['Consumo'] > limite_superior)
outliers_iqr = df[df['Outlier_IQR']]

print(f"   • Q1: {Q1:.2f} | Q3: {Q3:.2f} | IQR: {IQR:.2f}")
print(f"   • Limites: [{limite_inferior:.2f}, {limite_superior:.2f}]")
print(f"   • Outliers detectados: {len(outliers_iqr)}")
print(f"   • Percentual: {len(outliers_iqr) / len(df) * 100:.1f}% do dataset")

# 2.3 MÉTODO MODIFIED Z-SCORE (ROBUSTO)
print("\n🎯 Método 3: Modified Z-Score (Robusto)")
mediana = df['Consumo'].median()
mad = np.median(np.abs(df['Consumo'] - mediana))  # Median Absolute Deviation
df['Modified_ZScore'] = 0.6745 * (df['Consumo'] - mediana) / mad
df['Outlier_Modified'] = df['Modified_ZScore'].abs() > 3.5

outliers_modified = df[df['Outlier_Modified']]
print(f"   • Outliers detectados (|MZ| > 3.5): {len(outliers_modified)}")
print(f"   • Percentual: {len(outliers_modified) / len(df) * 100:.1f}% do dataset")

# Combinar métodos para consenso
df['Consenso_Outlier'] = df['Outlier_IQR'] | df['Outlier_ZScore']
outliers_consenso = df[df['Consenso_Outlier']]

print(f"\n🔍 Consenso entre Métodos:")
print(f"   • Outliers por consenso: {len(outliers_consenso)}")
print(f"   • Confiabilidade: {len(outliers_consenso) / len(df) * 100:.1f}% de anomalias detectadas")

# =============================================================================
# 3. PAINEL DE ALERTAS - CASOS FORA DO PADRÃO
# =============================================================================
print("\n🚨 ETAPA 3: PAINEL DE ALERTAS")
print("-" * 60)

print("⚠️  ALERTAS DE CONSUMO ANÔMALO (Método IQR):")
print("=" * 70)

if len(outliers_iqr) > 0:
    for idx, row in outliers_iqr.iterrows():
        desvio = row['Consumo'] - df['Consumo'].mean()
        impacto = abs(desvio) * CUSTO_UNITARIO

        print(f"📅 {row['Data'].strftime('%d/%m/%Y')} ({row['DiaSemana']})")
        print(f"   📊 Consumo: {row['Consumo']:.0f} unidades")
        print(f"   📈 Z-Score: {row['Z_Score']:.2f}")
        print(f"   💰 Impacto: R$ {impacto:.2f}")
        print(f"   📝 Desvio da média: {desvio:+.0f} unidades")
        print("-" * 40)
else:
    print("✅ Nenhuma anomalia detectada pelo método IQR")

# Análise por período
print("\n📊 ANÁLISE POR PERÍODOS:")
consumo_por_mes = df.groupby('Mes').agg({
    'Consumo': ['mean', 'std', 'min', 'max', 'sum'],
    'Outlier_IQR': 'sum'
}).round(2)

consumo_por_mes.columns = ['Média', 'Desvio', 'Mín', 'Máx', 'Total', 'Outliers']
print(consumo_por_mes)

# =============================================================================
# 4. SIMULAÇÃO DE IMPACTO FINANCEIRO
# =============================================================================
print(f"\n💰 ETAPA 4: SIMULAÇÃO DE IMPACTO FINANCEIRO")
print("-" * 60)

# Cálculos financeiros
consumo_total = df['Consumo'].sum()
custo_total = consumo_total * CUSTO_UNITARIO

consumo_outliers = outliers_iqr['Consumo'].sum()
custo_outliers = consumo_outliers * CUSTO_UNITARIO

consumo_esperado_outliers = len(outliers_iqr) * df['Consumo'].mean()
custo_esperado_outliers = consumo_esperado_outliers * CUSTO_UNITARIO

economia_potencial = custo_outliers - custo_esperado_outliers

print(f"💼 IMPACTO FINANCEIRO GERAL:")
print(f"   • Custo total do período: R$ {custo_total:,.2f}")
print(f"   • Custo médio diário: R$ {custo_total / len(df):,.2f}")
print(f"   • Custo unitário: R$ {CUSTO_UNITARIO:.2f}")

print(f"\n🚨 IMPACTO DAS ANOMALIAS:")
print(f"   • Consumo anômalo: {consumo_outliers:.0f} unidades")
print(f"   • Custo das anomalias: R$ {custo_outliers:,.2f}")
print(f"   • Percentual do custo total: {custo_outliers / custo_total * 100:.1f}%")

print(f"\n💡 ECONOMIA POTENCIAL:")
print(f"   • Consumo esperado: {consumo_esperado_outliers:.0f} unidades")
print(f"   • Custo esperado: R$ {custo_esperado_outliers:,.2f}")
print(f"   • Economia com controle: R$ {economia_potencial:,.2f}")

# Projeção anual
print(f"\n📈 PROJEÇÃO ANUAL:")
fator_anual = 365 / TOTAL_DIAS
custo_anual_projetado = custo_total * fator_anual
economia_anual = economia_potencial * fator_anual

print(f"   • Custo anual projetado: R$ {custo_anual_projetado:,.2f}")
print(f"   • Economia anual potencial: R$ {economia_anual:,.2f}")

# =============================================================================
# 5. VISUALIZAÇÕES AVANÇADAS
# =============================================================================
print(f"\n📊 ETAPA 5: GERAÇÃO DE VISUALIZAÇÕES")
print("-" * 60)

# Configurar o layout dos gráficos
fig = plt.figure(figsize=(20, 16))

# Gráfico 1: Série Temporal Principal
ax1 = plt.subplot(3, 3, (1, 3))
plt.plot(df['Data'], df['Consumo'], 'b-', alpha=0.7, linewidth=1.5, label='Consumo Diário')
plt.scatter(outliers_iqr['Data'], outliers_iqr['Consumo'],
            color='red', s=80, label=f'Outliers IQR ({len(outliers_iqr)})', zorder=5)
plt.axhline(y=df['Consumo'].mean(), color='green', linestyle='--',
            label=f'Média ({df["Consumo"].mean():.1f})', alpha=0.8)
plt.axhline(y=limite_superior, color='orange', linestyle=':',
            label=f'Limite Superior ({limite_superior:.1f})', alpha=0.8)
plt.axhline(y=limite_inferior, color='orange', linestyle=':',
            label=f'Limite Inferior ({limite_inferior:.1f})', alpha=0.8)
plt.title('🔍 Detecção de Anomalias no Consumo de Materiais', fontsize=14, fontweight='bold')
plt.xlabel('Data')
plt.ylabel('Consumo (unidades)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Gráfico 2: Boxplot Detalhado
ax2 = plt.subplot(3, 3, 4)
box_plot = plt.boxplot(df['Consumo'], patch_artist=True, notch=True)
box_plot['boxes'][0].set_facecolor('lightblue')
box_plot['boxes'][0].set_alpha(0.7)
plt.title('📦 Boxplot do Consumo')
plt.ylabel('Consumo (unidades)')
plt.grid(True, alpha=0.3)

# Gráfico 3: Histograma com Distribuição
ax3 = plt.subplot(3, 3, 5)
plt.hist(df['Consumo'], bins=25, alpha=0.7, color='skyblue',
         edgecolor='black', density=True, label='Distribuição Observada')
plt.axvline(df['Consumo'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Média ({df["Consumo"].mean():.1f})')
plt.axvline(df['Consumo'].median(), color='green', linestyle='--',
            linewidth=2, label=f'Mediana ({df["Consumo"].median():.1f})')
plt.title('📊 Distribuição do Consumo')
plt.xlabel('Consumo (unidades)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 4: Z-Score ao longo do tempo
ax4 = plt.subplot(3, 3, 6)
plt.plot(df['Data'], df['Z_Score'], 'purple', alpha=0.7, linewidth=1)
plt.axhline(y=3, color='red', linestyle='--', label='Limite Superior (+3)')
plt.axhline(y=-3, color='red', linestyle='--', label='Limite Inferior (-3)')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
plt.scatter(outliers_zscore['Data'], outliers_zscore['Z_Score'],
            color='red', s=50, zorder=5)
plt.title('📈 Z-Score ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Gráfico 5: Consumo por Dia da Semana
ax5 = plt.subplot(3, 3, 7)
consumo_dia_semana = df.groupby('DiaSemana')['Consumo'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
dias_pt = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
plt.bar(dias_pt, consumo_dia_semana.values, color='lightcoral', alpha=0.8)
plt.title('📅 Consumo Médio por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Consumo Médio')
plt.grid(True, alpha=0.3)

# Gráfico 6: Consumo Mensal
ax6 = plt.subplot(3, 3, 8)
consumo_mensal = df.groupby('Mes')['Consumo'].mean()
meses = ['Jan', 'Fev', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul']
plt.plot(meses[:len(consumo_mensal)], consumo_mensal.values,
         'o-', color='darkgreen', linewidth=2, markersize=8)
plt.title('📆 Evolução Mensal do Consumo')
plt.xlabel('Mês')
plt.ylabel('Consumo Médio')
plt.grid(True, alpha=0.3)

# Gráfico 7: Heatmap de Outliers
ax7 = plt.subplot(3, 3, 9)
df_heatmap = df.copy()
df_heatmap['Semana_Ano'] = df_heatmap['Data'].dt.isocalendar().week
df_heatmap['DiaSemana_Num'] = df_heatmap['Data'].dt.dayofweek

heatmap_data = df_heatmap.pivot_table(
    values='Outlier_IQR',
    index='Semana_Ano',
    columns='DiaSemana_Num',
    aggfunc='sum',
    fill_value=0
)

plt.imshow(heatmap_data.values, cmap='Reds', aspect='auto')
plt.colorbar(label='Outliers')
plt.title('🔥 Heatmap de Anomalias')
plt.xlabel('Dia da Semana')
plt.ylabel('Semana do Ano')

plt.tight_layout()

# Salvar visualizações
try:
    plt.savefig('analise_completa_anomalias.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Gráficos salvos: 'analise_completa_anomalias.png'")
except Exception as e:
    print(f"❌ Erro ao salvar gráficos: {e}")

plt.close()

# =============================================================================
# 6. EXPORTAÇÃO DE DADOS E RELATÓRIOS
# =============================================================================
print(f"\n💾 ETAPA 6: EXPORTAÇÃO DE DADOS")
print("-" * 60)

try:
    # Dataset completo
    df.to_csv('dataset_consumo_completo.csv', index=False, encoding='utf-8-sig')

    # Outliers detectados
    outliers_detalhado = outliers_iqr[['Data', 'Consumo', 'Z_Score', 'DiaSemana']].copy()
    outliers_detalhado['Desvio_Media'] = outliers_detalhado['Consumo'] - df['Consumo'].mean()
    outliers_detalhado['Impacto_Financeiro'] = outliers_detalhado['Desvio_Media'].abs() * CUSTO_UNITARIO
    outliers_detalhado.to_csv('outliers_detectados.csv', index=False, encoding='utf-8-sig')

    # Relatório estatístico
    relatorio_stats = pd.DataFrame({
        'Metrica': ['Total_Registros', 'Consumo_Medio', 'Desvio_Padrao', 'Outliers_IQR',
                    'Outliers_ZScore', 'Custo_Total', 'Custo_Outliers', 'Economia_Potencial'],
        'Valor': [len(df), df['Consumo'].mean(), df['Consumo'].std(), len(outliers_iqr),
                  len(outliers_zscore), custo_total, custo_outliers, economia_potencial]
    })
    relatorio_stats.to_csv('relatorio_estatistico.csv', index=False, encoding='utf-8-sig')

    print(f"✅ Arquivos exportados com sucesso:")
    print(f"   • dataset_consumo_completo.csv")
    print(f"   • outliers_detectados.csv")
    print(f"   • relatorio_estatistico.csv")

except Exception as e:
    print(f"❌ Erro na exportação: {e}")

# =============================================================================
# 7. RESUMO EXECUTIVO FINAL
# =============================================================================
print(f"\n📋 RESUMO EXECUTIVO")
print("=" * 80)

print(f"🎯 OBJETIVO: Detecção de anomalias em registros de consumo de materiais")
print(f"📊 DATASET: {len(df)} registros de {TOTAL_DIAS} dias ({DATA_INICIO})")
print(f"🔍 MÉTODOS: Z-Score, IQR e Modified Z-Score")

print(f"\n📈 RESULTADOS PRINCIPAIS:")
print(f"   • Consumo médio diário: {df['Consumo'].mean():.1f} ± {df['Consumo'].std():.1f} unidades")
print(f"   • Anomalias detectadas: {len(outliers_iqr)} casos ({len(outliers_iqr) / len(df) * 100:.1f}%)")
print(f"   • Maior anomalia: {outliers_iqr['Consumo'].max():.0f} unidades" if len(
    outliers_iqr) > 0 else "   • Maior anomalia: N/A")
print(f"   • Impacto financeiro: R$ {custo_outliers:,.2f}")

print(f"\n💰 IMPACTO ECONÔMICO:")
print(f"   • Custo total do período: R$ {custo_total:,.2f}")
print(f"   • Economia potencial com controle: R$ {economia_potencial:,.2f}")
print(f"   • Projeção economia anual: R$ {economia_anual:,.2f}")

print(f"\n🎯 RECOMENDAÇÕES:")
print(f"   • Investigar as {len(outliers_iqr)} anomalias identificadas")
print(f"   • Implementar sistema de monitoramento contínuo")
print(f"   • Estabelecer alertas automáticos para desvios > 3σ")
print(f"   • Revisar processos nos dias: {', '.join([d.strftime('%d/%m') for d in outliers_iqr['Data'].head(3)])}")

print(f"\n✅ CONCLUSÃO:")
print(f"   O projeto identificou com sucesso padrões anômalos no consumo,")
print(f"   demonstrando potencial de economia de R$ {economia_potencial:,.2f} no período")
print(f"   e R$ {economia_anual:,.2f} anualizados através do controle efetivo das anomalias.")

print("=" * 80)
print(f"🕐 Análise concluída em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 80)