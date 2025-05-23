import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use backend não-interativo para evitar problemas com Tkinter

# 1. Simulação de dados de consumo
np.random.seed(42)

# Gerar 200 registros de consumo (simulando retirada diária de materiais)
dias = pd.date_range(start='2024-01-01', periods=200, freq='D')
consumo_normal = np.random.poisson(lam=50, size=200)  # consumo normal

# Inserir alguns outliers propositalmente
outliers = np.random.choice([150, 200, 250], size=5)
pos_outliers = np.random.choice(range(200), size=5, replace=False)
consumo = consumo_normal.copy()
consumo[pos_outliers] = outliers

# Criar DataFrame
df = pd.DataFrame({
    'Data': dias,
    'Consumo': consumo
})

# 2. Análise Estatística: Z-Score e IQR
# ---- Z-Score
df['Z_Score'] = (df['Consumo'] - df['Consumo'].mean()) / df['Consumo'].std()
df['Outlier_Z'] = df['Z_Score'].abs() > 3  # limiar comum para outliers

# ---- IQR
Q1 = df['Consumo'].quantile(0.25)
Q3 = df['Consumo'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df['Outlier_IQR'] = (df['Consumo'] < limite_inferior) | (df['Consumo'] > limite_superior)

# 3. Estatísticas Descritivas
print("=== ESTATÍSTICAS DESCRITIVAS ===")
print(f"Média de consumo: {df['Consumo'].mean():.2f}")
print(f"Mediana de consumo: {df['Consumo'].median():.2f}")
print(f"Desvio padrão: {df['Consumo'].std():.2f}")
print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Limite inferior (IQR): {limite_inferior:.2f}")
print(f"Limite superior (IQR): {limite_superior:.2f}")

# 4. Painel de Alertas
print("\n=== ALERTAS DE CONSUMO FORA DO PADRÃO (IQR) ===")
outliers_iqr = df[df['Outlier_IQR']][['Data', 'Consumo', 'Z_Score']]
print(outliers_iqr)

print("\n=== ALERTAS DE CONSUMO FORA DO PADRÃO (Z-Score > 3) ===")
outliers_z = df[df['Outlier_Z']][['Data', 'Consumo', 'Z_Score']]
print(outliers_z)

# 5. Simulação do impacto financeiro
custo_unitario = 10

# Custo total dos outliers
custo_outliers_iqr = df[df['Outlier_IQR']]['Consumo'].sum() * custo_unitario
custo_outliers_z = df[df['Outlier_Z']]['Consumo'].sum() * custo_unitario
custo_total = df['Consumo'].sum() * custo_unitario

print(f"\n=== IMPACTO FINANCEIRO ===")
print(f"Custo total do período: R$ {custo_total:,.2f}")
print(f"Impacto financeiro dos outliers (IQR): R$ {custo_outliers_iqr:,.2f}")
print(f"Impacto financeiro dos outliers (Z-Score): R$ {custo_outliers_z:,.2f}")
print(f"Percentual de outliers IQR: {(len(outliers_iqr)/len(df))*100:.1f}%")
print(f"Percentual de outliers Z-Score: {(len(outliers_z)/len(df))*100:.1f}%")

# 6. Análise de tendências
print(f"\n=== ANÁLISE DE TENDÊNCIAS ===")
df['Mes'] = df['Data'].dt.month
consumo_mensal = df.groupby('Mes')['Consumo'].agg(['mean', 'std', 'sum']).round(2)
print("Consumo por mês:")
print(consumo_mensal)

# 7. Visualização dos dados (salvando em arquivo)
plt.style.use('default')  # Use estilo padrão
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Série temporal com outliers
axes[0, 0].plot(df['Data'], df['Consumo'], 'b-', alpha=0.7, label='Consumo diário')
axes[0, 0].scatter(df[df['Outlier_IQR']]['Data'],
                   df[df['Outlier_IQR']]['Consumo'],
                   color='red', s=50, label='Outliers (IQR)', zorder=5)
axes[0, 0].axhline(y=df['Consumo'].mean(), color='green', linestyle='--', label='Média')
axes[0, 0].set_title('Consumo de Materiais com Identificação de Outliers')
axes[0, 0].set_xlabel('Data')
axes[0, 0].set_ylabel('Quantidade Consumida')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Boxplot
axes[0, 1].boxplot(df['Consumo'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[0, 1].set_title('Boxplot do Consumo')
axes[0, 1].set_ylabel('Quantidade Consumida')
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Histograma
axes[1, 0].hist(df['Consumo'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 0].axvline(df['Consumo'].mean(), color='red', linestyle='--', label='Média')
axes[1, 0].axvline(df['Consumo'].median(), color='green', linestyle='--', label='Mediana')
axes[1, 0].set_title('Distribuição do Consumo')
axes[1, 0].set_xlabel('Quantidade Consumida')
axes[1, 0].set_ylabel('Frequência')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Consumo mensal
consumo_mensal_plot = df.groupby('Mes')['Consumo'].mean()
axes[1, 1].bar(consumo_mensal_plot.index, consumo_mensal_plot.values,
               color='lightcoral', alpha=0.7)
axes[1, 1].set_title('Consumo Médio por Mês')
axes[1, 1].set_xlabel('Mês')
axes[1, 1].set_ylabel('Consumo Médio')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Salvar o gráfico em arquivo
try:
    plt.savefig('analise_consumo_materiais.png', dpi=300, bbox_inches='tight')
    print(f"\n=== GRÁFICO SALVO ===")
    print("Gráfico salvo como 'analise_consumo_materiais.png'")
except Exception as e:
    print(f"Erro ao salvar gráfico: {e}")

# Fechar a figura para liberar memória
plt.close()
print("Gráfico gerado com sucesso! Verifique o arquivo 'analise_consumo_materiais.png'")

# 8. Salvar dados em CSV para análise posterior
try:
    df.to_csv('dados_consumo_com_outliers.csv', index=False)
    outliers_iqr.to_csv('outliers_detectados.csv', index=False)
    print(f"\n=== DADOS SALVOS ===")
    print("Dados salvos em:")
    print("- dados_consumo_com_outliers.csv")
    print("- outliers_detectados.csv")
except Exception as e:
    print(f"Erro ao salvar dados: {e}")

print(f"\n=== RESUMO EXECUTIVO ===")
print(f"• Total de registros analisados: {len(df)}")
print(f"• Outliers detectados (IQR): {len(outliers_iqr)} ({(len(outliers_iqr)/len(df))*100:.1f}%)")
print(f"• Consumo médio diário: {df['Consumo'].mean():.1f} unidades")
print(f"• Maior consumo registrado: {df['Consumo'].max()} unidades")
print(f"• Menor consumo registrado: {df['Consumo'].min()} unidades")
print(f"• Impacto financeiro dos outliers: R$ {custo_outliers_iqr:,.2f}")
print(f"• Economia potencial com controle: R$ {custo_outliers_iqr - (len(outliers_iqr) * df['Consumo'].mean() * custo_unitario):,.2f}")