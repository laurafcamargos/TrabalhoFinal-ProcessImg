import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from src.descriptors import extract_all_features


def load_and_extract_features(split='train'):
    """Carrega imagens e extrai features."""
    features = []
    labels = []

    split_path = Path(f'dataset/{split}')
    total_files = len(list(split_path.rglob('*.jpg'))) + len(list(split_path.rglob('*.jpeg')))

    print(f"Processando {split}: {total_files} imagens")

    with tqdm(total=total_files, desc=f"Extraindo {split}") as pbar:
        for denom_folder in sorted(split_path.iterdir()):
            if not denom_folder.is_dir():
                continue

            denom = denom_folder.name.replace('nota-', '')

            for img_path in list(denom_folder.glob('*.jpg')) + list(denom_folder.glob('*.jpeg')):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (256, 256))
                    feat = extract_all_features(img)
                    features.append(feat)
                    labels.append(denom)

                pbar.update(1)

    return np.array(features), np.array(labels)


# ============ MAIN ============
print("\n" + "="*70)
print("CLASSIFICADOR DE CÉDULAS DO REAL")
print("="*70)

# Carrega e extrai
print("\nCarregando e extraindo features...")
X_train, y_train = load_and_extract_features('train')
X_test, y_test = load_and_extract_features('test')

print(f"\nTrain: {len(X_train)} amostras")
print(f"Test: {len(X_test)} amostras")
print(f"Classes: {sorted(np.unique(y_train))}")

# Normaliza
print("\nNormalizando features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Features: {X_train.shape[1]}")

# Treina Random Forest
print("\nTreinando Random Forest (300 árvores, max_depth=15)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
model.fit(X_train, y_train)
print("Modelo treinado!")

# Predição
y_pred = model.predict(X_test)

# ============ RESULTADOS ============
print("\n" + "="*70)
print("RESULTADOS")
print("="*70)

overall_acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia Geral: {overall_acc:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============ GRÁFICOS ============
print("\nGerando gráficos...")
Path('output/visualizations').mkdir(parents=True, exist_ok=True)

classes = sorted(np.unique(y_test))
labels_formatted = [f'R${c}' for c in classes]

# 1. MATRIZ DE CONFUSÃO
cm = confusion_matrix(y_test, y_pred, labels=classes)

plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_formatted,
            yticklabels=labels_formatted,
            cbar_kws={'label': 'Quantidade'},
            linewidths=0.5,
            linecolor='gray')
plt.ylabel('Verdadeiro (Real)', fontsize=12, fontweight='bold')
plt.xlabel('Predito (Modelo)', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/visualizations/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Matriz de confusão salva")
plt.close()

# 2. ACURÁCIA POR DENOMINAÇÃO
importances = model.feature_importances_
accuracies = []
counts = []

print("\nAcurácia por Denominação:")
for cls in classes:
    mask = y_test == cls
    if np.sum(mask) > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        count = np.sum(mask)
        accuracies.append(acc * 100)
        counts.append(count)
        print(f"   R${cls}: {acc*100:.1f}% ({count} amostras)")

plt.figure(figsize=(12, 7))
colors = ['#2ecc71' if acc >= 70 else '#e74c3c' if acc < 60 else '#f39c12' for acc in accuracies]
bars = plt.bar(labels_formatted, accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

for bar, acc, count in zip(bars, accuracies, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
plt.xlabel('Denominação', fontsize=12, fontweight='bold')
plt.title('Acurácia por Denominação - Conjunto de Teste', fontsize=14, fontweight='bold')
plt.ylim(0, 110)
plt.axhline(y=70, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Acurácia geral')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('output/visualizations/accuracy_per_denomination.png', dpi=150, bbox_inches='tight')
print("Acurácia por denominação salvo")
plt.close()

# 3. CONTRIBUIÇÃO POR DESCRITOR
hue_importance = np.sum(importances[:180])
sat_val_importance = np.sum(importances[180:244])
hough_importance = np.sum(importances[244:249])
lbp_importance = np.sum(importances[249:])

descriptors = ['HSV Hue\n(180 features)', 'HSV S/V\n(64 features)', 'Hough Lines\n(5 features)', 'LBP\n(59 features)']
contributions = [hue_importance, sat_val_importance, hough_importance, lbp_importance]
colors_desc = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

plt.figure(figsize=(10, 7))
bars = plt.bar(descriptors, contributions, color=colors_desc, edgecolor='black', linewidth=1.5, alpha=0.8)

for bar, contrib in zip(bars, contributions):
    height = bar.get_height()
    pct = 100 * contrib / np.sum(contributions)
    plt.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{pct:.1f}%\n({contrib:.4f})', ha='center', va='center', fontweight='bold', fontsize=11, color='white')

plt.ylabel('Importância Acumulada', fontsize=12, fontweight='bold')
plt.title('Contribuição de Cada Descritor - Random Forest', fontsize=14, fontweight='bold')
plt.ylim(0, max(contributions) * 1.15)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/visualizations/descriptor_contribution.png', dpi=150, bbox_inches='tight')
print("Contribuição por descritor salvo")
plt.close()

# 4. PRECISION/RECALL POR CLASSE
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=classes, average=None)

x_pos = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(13, 7))
bars1 = plt.bar(x_pos - width, precision * 100, width, label='Precisão', color='#3498db', edgecolor='black', linewidth=1)
bars2 = plt.bar(x_pos, recall * 100, width, label='Recall', color='#2ecc71', edgecolor='black', linewidth=1)
bars3 = plt.bar(x_pos + width, f1 * 100, width, label='F1-Score', color='#e74c3c', edgecolor='black', linewidth=1)

plt.xlabel('Denominação', fontsize=12, fontweight='bold')
plt.ylabel('Percentual (%)', fontsize=12, fontweight='bold')
plt.title('Precisão, Recall e F1-Score por Denominação', fontsize=14, fontweight='bold')
plt.xticks(x_pos, labels_formatted)
plt.ylim(0, 110)
plt.axhline(y=70, color='gray', linestyle='--', linewidth=2, alpha=0.5)
plt.legend(fontsize=11, loc='lower right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('output/visualizations/precision_recall_per_class.png', dpi=150, bbox_inches='tight')
print("Precisão/Recall por classe salvo")
plt.close()

# ============ RESUMO ============
print("\n" + "="*70)
print("ANÁLISE CONCLUÍDA")
print("="*70)

threshold = np.mean(importances[np.argsort(importances)[-20:]]) / 5
negligible_count = np.sum(importances < threshold)
negligible_pct = 100 * negligible_count / len(importances)
useful_features = 308 - negligible_count

# Calcula features negligenciáveis (apenas para informação)
threshold = np.mean(importances[np.argsort(importances)[-20:]]) / 5
negligible_count = np.sum(importances < threshold)
negligible_pct = 100 * negligible_count / len(importances)

print(f"\nResultados Finais:")
print(f"   Acurácia geral: {overall_acc*100:.1f}%")
print(f"   Melhor denominação: R${classes[np.argmax(accuracies)]} ({max(accuracies):.1f}%)")
print(f"   Pior denominação: R${classes[np.argmin(accuracies)]} ({min(accuracies):.1f}%)")
print(f"   Features: 308 (Hue 180 + S/V 64 + Hough 5 + LBP 59)")
print(f"   Features negligenciáveis: {negligible_count} ({negligible_pct:.1f}%)")
print(f"   Features úteis: {308 - negligible_count} ({100-negligible_pct:.1f}%)")
print(f"\nGraficos salvos em output/visualizations/")
print(f"   - confusion_matrix.png")
print(f"   - accuracy_per_denomination.png")
print(f"   - descriptor_contribution.png")
print(f"   - precision_recall_per_class.png")
