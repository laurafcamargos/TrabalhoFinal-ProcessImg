# Classificacao de Cedulas do Real

Trabalho Final — SCC0251: Processamento de Imagens

## Alunos -  Grupo AE

- Gabriela dos Santos Amaral
- Laura Fernandes Camargos
- Vinicius Henrique Pereira Giroto

## Objetivo

Desenvolver um classificador supervisionado que identifica automaticamente a denominacao de cedulas brasileiras (R$2, R$5, R$10, R$20, R$50, R$100 e R$200) a partir de fotografias reais adquiridas com camera de celular.

O sistema utiliza descritores classicos de Processamento de Imagens e um classificador Random Forest.

## Fonte dos dados

As imagens utilizadas neste projeto foram obtidas do dataset publico no Kaggle:

**[Cedulas do Real — karenalmeida340](https://www.kaggle.com/datasets/karenalmeida340/cdulas-do-real)**

O dataset contem fotografias de cedulas brasileiras (R$2 a R$200) capturadas em condicoes reais, com variacao de iluminacao, angulo e fundo. O diretorio `dataset/` nao e versionado neste repositorio; e necessario baixar os arquivos manualmente pelo link acima.

## Abordagem

Cada imagem e redimensionada para 256x256 pixels e convertida em um vetor de 308 atributos numericos:

| Descritor | Features | Descricao |
|-----------|----------|-----------|
| HSV Hue | 180 | Histograma de matiz com mascara de pixels confiaveis |
| HSV S/V | 64 | Histogramas de saturacao e valor (32 bins cada) |
| Hough Lines | 5 | Estatisticas de linhas detectadas (Canny + Hough) |
| LBP | 59 | Histograma de Local Binary Patterns uniforme |

Os histogramas sao normalizados (L1). As features passam por `StandardScaler` antes do treinamento.

### Classificador

- `RandomForestClassifier` (scikit-learn)
- 300 arvores, `max_depth=15`, `min_samples_split=5`
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
│   ├── descriptors.py           # Extracao de descritores
│   └── preprocessing.py         # Funcoes auxiliares (mascara HSV; segmentacao experimental)
├── dataset/                     # Nao versionado — ver secao Dataset
│   ├── train/
│   └── test/
└── output/
    ├── visualizations/          # Matriz de confusao e feature importance
    └── results/                 # Relatorios tecnicos (.txt)
```

## Como executar

### 1. Ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

O diretorio `dataset/` nao esta incluido no repositorio devido ao tamanho (~8,6 GB).

Baixe o dataset em: https://www.kaggle.com/datasets/karenalmeida340/cdulas-do-real

Organize as pastas `train/` e `test/` com subpastas por denominacao (`nota-2`, `nota-5`, ..., `nota-200`).

### 3. Executar o pipeline

```bash
python main.py
```

O script ira:

1. Carregar imagens de `dataset/train/` e `dataset/test/`
2. Extrair descritores de cada imagem
3. Treinar o Random Forest no conjunto de treino
4. Avaliar no conjunto de teste (acuracia, classification report)
5. Salvar graficos em `output/visualizations/`

Tempo estimado de execucao: 7 a 8 minutos.

## Saidas geradas

```
output/visualizations/
├── confusion_matrix_balanced.png
└── feature_importance_balanced.png

output/results/
├── relatorio_pipeline.txt
└── evolucao_versoes.txt
```

## Metricas de avaliacao

- **Acuracia**: proporcao de classificacoes corretas no conjunto de teste
- **Precision / Recall / F1**: por denominacao (via `classification_report`)
- **Matriz de confusao**: erros entre classes
- **Feature importance**: relevancia de cada descritor no Random Forest

## Referencias

- [HSV Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)
- [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Dataset Kaggle — Cedulas do Real](https://www.kaggle.com/datasets/karenalmeida340/cdulas-do-real)
