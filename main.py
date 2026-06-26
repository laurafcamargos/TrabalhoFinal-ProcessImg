"""
Pipeline BALANCEADO: HSV + Hough + LBP

Combina os 3 descritores mais importantes:
- HSV Hue: 180 features (COR - o que mais funciona!)
- Hough Lines: 5 features (PADRÃO DE LINHAS)
- LBP: 59 features (TEXTURA)

Total: 244 features (não é muito, não é pouco)

"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.descriptors import extract_all_features


# ============ CARREGA E EXTRAI ============
def load_and_extract_features(split='train'):
    """Carrega imagens E extrai features ao mesmo tempo."""
    features = []
    labels = []

    split_path = Path(f'dataset/{split}')

    # Conta total de imagens
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
                    # Extrai features
                    feat = extract_all_features(img)
                    features.append(feat)
                    labels.append(denom)

                pbar.update(1)

    return np.array(features), np.array(labels)


# ============ MAIN ============
print("\n" + "="*70)
print("CLASSIFICADOR BALANCEADO: HSV + Hough + LBP")
print("="*70)

# Carrega e extrai
print("\n⏳ Carregando e extraindo features...")
X_train, y_train = load_and_extract_features('train')
X_test, y_test = load_and_extract_features('test')

print(f"\n✓ Train: {len(X_train)} amostras")
print(f"✓ Test: {len(X_test)} amostras")
print(f"✓ Classes: {np.unique(y_train)}")

# Normaliza
print("\n⏳ Normalizando features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"✓ Shape: {X_train.shape}")
print(f"  180 Hue + 64 S/V + 5 Hough + 59 LBP = 308 features")

# Treina Random Forest
print("\n⏳ Treinando Random Forest (300 árvores, max_depth=15)...")
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
print("✓ Modelo treinado!")

# Avalia
print("\n" + "="*70)
print("RESULTADOS")
print("="*70)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Acurácia: {acc:.2%}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Gráficos
print("\n⏳ Gerando gráficos...")

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(np.unique(y_train)),
            yticklabels=sorted(np.unique(y_train)))
plt.title('Matriz de Confusão - HSV + Hough + LBP')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.tight_layout()
Path('output/visualizations').mkdir(parents=True, exist_ok=True)
plt.savefig('output/visualizations/confusion_matrix_balanced.png', dpi=150)
print("✓ Matriz de confusão salva")
plt.close()

# Feature importance
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
top_importances = importances[top_indices]

feature_names = [f'Hue_{i}' for i in range(180)]
feature_names += [f'Sat_{i}' for i in range(32)]
feature_names += [f'Val_{i}' for i in range(32)]
feature_names.extend(['Hough_Num', 'Hough_Ang', 'Hough_AngStd', 'Hough_Dist', 'Hough_DistStd'])
feature_names += [f'LBP_{i}' for i in range(59)]

top_feature_names = [feature_names[i] for i in top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_importances)
plt.xlabel('Importância')
plt.title('Top 10 Features - HSV + Hough + LBP')
plt.tight_layout()
plt.savefig('output/visualizations/feature_importance_balanced.png', dpi=150)
print("✓ Feature importance salvo")
plt.close()

# Resumo
print("\n" + "="*70)
print("✓ TESTE CONCLUÍDO")
print("="*70)
print(f"\n📊 Resultados (conjunto de teste — 273 imagens):")
print(f"   Acurácia: {acc:.2%}")
print(f"   Features: 308 (Hue 180 + S/V 64 + Hough 5 + LBP 59)")
print(f"   Modelo: Random Forest 300 árvores, class_weight=balanced")
print(f"\n📁 Gráficos salvos em output/visualizations/")
