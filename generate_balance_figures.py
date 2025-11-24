"""
Script per generare le figure della matrice categoria × difficoltà
e del dataset bilanciato per il Capitolo 3
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configura lo stile
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Dati per la matrice categoria × difficoltà
categories = ['Algoritmi', 'Matematica', 'Liste', 'Stringhe']
difficulties = ['Facile', 'Media', 'Difficile']

# Matrice 4×3 con 5 problemi per cella
matrix_data = np.ones((len(categories), len(difficulties))) * 5

# Figura 1: Heatmap della matrice categoria × difficoltà
fig1, ax1 = plt.subplots(figsize=(7, 5))

# Imposta i limiti e lo sfondo bianco
ax1.set_xlim(-0.5, len(difficulties)-0.5)
ax1.set_ylim(-0.5, len(categories)-0.5)
ax1.set_facecolor('white')

# Configura assi
ax1.set_xticks(np.arange(len(difficulties)))
ax1.set_yticks(np.arange(len(categories)))
ax1.set_xticklabels(difficulties, fontsize=12)
ax1.set_yticklabels(categories, fontsize=12)
ax1.set_xlabel('Difficoltà', fontsize=13, fontweight='bold')
ax1.set_ylabel('Categoria Funzionale', fontsize=13, fontweight='bold')
ax1.set_title('Matrice Categoria × Difficoltà\n(5 problemi per cella)', 
              fontsize=14, fontweight='bold', pad=20)

# Aggiungi valori nelle celle
for i in range(len(categories)):
    for j in range(len(difficulties)):
        text = ax1.text(j, i, int(matrix_data[i, j]),
                       ha="center", va="center", color="black", 
                       fontsize=16, fontweight='bold')

# Griglia principale nera
ax1.set_xticks(np.arange(len(difficulties)+1)-.5, minor=True)
ax1.set_yticks(np.arange(len(categories)+1)-.5, minor=True)
ax1.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
ax1.tick_params(which="minor", size=0)
ax1.invert_yaxis()  # Inverti asse y per avere la prima categoria in alto

plt.tight_layout()
plt.savefig('latex/figures/matrice_categoria_difficolta.pdf', dpi=300, bbox_inches='tight')
plt.savefig('latex/figures/matrice_categoria_difficolta.png', dpi=300, bbox_inches='tight')
print("✓ Salvata: matrice_categoria_difficolta.pdf/png")
plt.close()

# Figura 2: Distribuzione del dataset bilanciato
fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Distribuzione per difficoltà
difficulty_counts = [20, 20, 20]
colors_diff = ['#90EE90', '#FFD700', '#FF6B6B']

bars1 = ax2_1.bar(difficulties, difficulty_counts, color=colors_diff, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
ax2_1.set_ylabel('Numero di problemi', fontsize=14, fontweight='bold')
ax2_1.set_xlabel('Livello di difficoltà', fontsize=14, fontweight='bold')
ax2_1.set_title('Distribuzione per Difficoltà', fontsize=15, fontweight='bold')
ax2_1.set_ylim(0, 25)
ax2_1.grid(axis='y', alpha=0.3, linestyle='--')
ax2_1.tick_params(axis='both', which='major', labelsize=13)

# Aggiungi etichette sulle barre
for bar in bars1:
    height = bar.get_height()
    ax2_1.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=16, fontweight='bold')

# Subplot 2: Distribuzione per categoria
category_counts = [15, 15, 15, 15]
colors_cat = ['#8B5CF6', '#3B82F6', '#10B981', '#F59E0B']

bars2 = ax2_2.bar(categories, category_counts, color=colors_cat,
                   edgecolor='black', linewidth=1.5, alpha=0.8)
ax2_2.set_ylabel('Numero di problemi', fontsize=14, fontweight='bold')
ax2_2.set_xlabel('Categoria funzionale', fontsize=14, fontweight='bold')
ax2_2.set_title('Distribuzione per Categoria', fontsize=15, fontweight='bold')
ax2_2.set_ylim(0, 20)
ax2_2.grid(axis='y', alpha=0.3, linestyle='--')
ax2_2.tick_params(axis='both', which='major', labelsize=13)

# Aggiungi etichette sulle barre
for bar in bars2:
    height = bar.get_height()
    ax2_2.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.suptitle('Dataset Bilanciato: 60 Problemi Totali', 
             fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('latex/figures/dataset_bilanciato.pdf', dpi=300, bbox_inches='tight')
plt.savefig('latex/figures/dataset_bilanciato.png', dpi=300, bbox_inches='tight')
print("✓ Salvata: dataset_bilanciato.pdf/png")
plt.close()

# Figura 3: Heatmap completa con distribuzione per dataset di origine
fig3, ax3 = plt.subplots(figsize=(9, 5.5))

# Matrice dettagliata (esempio di distribuzione HumanEval vs MBPP per cella)
# Questi sono valori di esempio - potresti voler aggiornare con dati reali
distribution_data = np.array([
    [3, 3, 2],  # Algoritmi: (HE, MBPP, HE)
    [2, 3, 3],  # Matematica
    [3, 2, 3],  # Liste
    [2, 2, 2],  # Stringhe
])

# Imposta i limiti e lo sfondo bianco
ax3.set_xlim(-0.5, len(difficulties)-0.5)
ax3.set_ylim(-0.5, len(categories)-0.5)
ax3.set_facecolor('white')

ax3.set_xticks(np.arange(len(difficulties)))
ax3.set_yticks(np.arange(len(categories)))
ax3.set_xticklabels(difficulties, fontsize=12)
ax3.set_yticklabels(categories, fontsize=12)
ax3.set_xlabel('Difficoltà', fontsize=13, fontweight='bold')
ax3.set_ylabel('Categoria Funzionale', fontsize=13, fontweight='bold')
ax3.set_title('Strategia di Selezione Bilanciata\nDistribuzione uniforme 5×4×3', 
              fontsize=14, fontweight='bold', pad=20)

# Aggiungi valori e dettagli nelle celle
for i in range(len(categories)):
    for j in range(len(difficulties)):
        # Valore principale
        text1 = ax3.text(j, i-0.15, f'{int(matrix_data[i, j])} problemi',
                        ha="center", va="center", color="black", 
                        fontsize=13, fontweight='bold')
        # Distribuzione HE/MBPP (esempio)
        he_count = distribution_data[i, j]
        mbpp_count = int(matrix_data[i, j]) - he_count
        text2 = ax3.text(j, i+0.15, f'({he_count} HE, {mbpp_count} MBPP)',
                        ha="center", va="center", color="darkblue", 
                        fontsize=9, style='italic')

# Griglia principale nera
ax3.set_xticks(np.arange(len(difficulties)+1)-.5, minor=True)
ax3.set_yticks(np.arange(len(categories)+1)-.5, minor=True)
ax3.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
ax3.tick_params(which="minor", size=0)
ax3.invert_yaxis()  # Inverti asse y

# Legenda
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
               markersize=10, markeredgecolor='black', label='HumanEval (HE)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkgray', 
               markersize=10, markeredgecolor='black', label='MBPP')
]
ax3.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
          fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig('latex/figures/strategia_selezione_bilanciata.pdf', dpi=300, bbox_inches='tight')
plt.savefig('latex/figures/strategia_selezione_bilanciata.png', dpi=300, bbox_inches='tight')
print("✓ Salvata: strategia_selezione_bilanciata.pdf/png")
plt.close()

print("\n Tutte le figure generate con successo!")
print("   - matrice_categoria_difficolta.pdf/png")
print("   - dataset_bilanciato.pdf/png")
print("   - strategia_selezione_bilanciata.pdf/png")
