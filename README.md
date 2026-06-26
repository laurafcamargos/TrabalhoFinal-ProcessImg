# Classificação de Cédulas do Real

Trabalho Final — SCC0251: Processamento de Imagens

## Alunos -  Grupo AE

- Gabriela dos Santos Amaral
- Laura Fernandes Camargos
- Vinicius Henrique Pereira Giroto

## Objetivo

Desenvolver um classificador supervisionado que identifica automaticamente a denominação de cédulas brasileiras (R$2, R$5, R$10, R$20, R$50, R$100 e R$200) a partir de fotografias reais adquiridas com câmera de celular.

O sistema utiliza descritores clássicos de Processamento de Imagens e um classificador Random Forest.

## Fonte dos dados

As imagens utilizadas neste projeto foram obtidas do dataset público no Kaggle:

**[Cedulas do Real — karenalmeida340](https://www.kaggle.com/datasets/karenalmeida340/cdulas-do-real)**

O dataset contém fotografias de cédulas brasileiras (R$2 a R$200) capturadas em condições reais, com variação de iluminação, ângulo e fundo. O diretório `dataset/` não é versionado neste repositório; é necessário baixar os arquivos manualmente pelo link acima.

## Abordagem

### Pré-processamento e Extração de Descritores

Cada imagem é redimensionada para 256×256 pixels e submetida a um pipeline de extração de 308 atributos numéricos. Os descritores foram selecionados para capturar características robustas das cédulas em diferentes condições de iluminação e ângulo de captura:

| Descritor | Features | Descrição |
|-----------|----------|-----------|
| HSV Hue | 180 | Histograma de matiz com máscara de pixels confiáveis (cores características da nota) |
| HSV S/V | 64 | Histogramas de saturação e valor (32 bins cada, captura luminosidade) |
| Hough Lines | 5 | Estatísticas de linhas detectadas (Canny + Hough; bordas e padrões geométricos) |
| LBP | 59 | Histograma de Local Binary Patterns uniforme (textura local) |

Os histogramas são normalizados (L1) para invariância de escala. As features passam por `StandardScaler` antes do treinamento, removendo vieses de escala entre descritores.

### Classificador

- `RandomForestClassifier` (scikit-learn)
- 300 árvores, `max_depth=15`, `min_samples_split=5`
- `class_weight='balanced'`
- `random_state=42`

### Resultados

- Conjunto de treino: 6.497 imagens (`dataset/train/`)
- Conjunto de teste: 273 imagens (`dataset/test/`)
- Acuracia no teste: aproximadamente **70%**

Durante o desenvolvimento foram testadas outras abordagens (multiplos descritores sem selecao, OCR, segmentacao com pre-processamento pesado) com desempenho inferior. Detalhes em `output/results/evolucao_versoes.txt`.

## Estrutura do projeto

```
trabalho_AEX/
├── main.py                      # Pipeline principal
├── requirements.txt
├── README.md
├── src/
│   ├── descriptors.py           # Extração de descritores
│   └── preprocessing.py         # Funções auxiliares (máscara HSV; segmentação experimental)
├── dataset/                     # Não versionado — ver seção Dataset
│   ├── train/
│   └── test/
└── output/
    ├── visualizations/          # Matriz de confusão e feature importance
    └── results/                 # Relatórios técnicos (.txt)
```

## Como executar

### 1. Ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

O diretório `dataset/` não está incluído no repositório devido ao tamanho (~8,6 GB).

Baixe o dataset em: https://www.kaggle.com/datasets/karenalmeida340/cdulas-do-real

Organize as pastas `train/` e `test/` com subpastas por denominação (`nota-2`, `nota-5`, ..., `nota-200`).

### 3. Executar o pipeline

```bash
python main.py
```

O script irá:

1. Carregar imagens de `dataset/train/` e `dataset/test/`
2. Extrair descritores de cada imagem
3. Treinar o Random Forest no conjunto de treino
4. Avaliar no conjunto de teste (acuracia, classification report)
5. Salvar graficos em `output/visualizations/`

Tempo estimado de execução: 7 a 8 minutos.

## Saídas geradas

```
output/visualizations/
├── confusion_matrix_balanced.png
└── feature_importance_balanced.png

output/results/
├── relatorio_pipeline.txt
└── evolucao_versoes.txt
```

## Métricas de avaliação

- **Acurácia**: proporção de classificações corretas no conjunto de teste
- **Precisão / Recall / F1**: por denominação (via `classification_report`)
- **Matriz de confusao**: erros entre classes
- **Feature importance**: relevancia de cada descritor no Random Forest

## Referências

- [HSV Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)
- [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Dataset Kaggle — Cedulas do Real](https://www.kaggle.com/datasets/karenalmeida340/cdulas-do-real)
