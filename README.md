# Detector de Cédulas Falsas - Processamento de Imagens

**Trabalho Final - SCC0251: Processamento de Imagens**

## 🎯 Objetivo

Desenvolver um classificador para detectar cédulas falsas do Real utilizando **descritores de imagem** e **Random Forest**.

Com dataset de **6.800+ imagens**, o projeto explora diferentes soluções e pode não chegar a 100% de acurácia (exploração também é válida).

## 📊 Descritores Utilizados

### 1. **HSV Hue Histogram** (Cor)
- Extrai o canal H (Hue) do espaço de cor HSV
- **Por quê HSV e não RGB?**
  - RGB muda completamente sob iluminação diferente (escuro, luz amarela)
  - HSV separa cor pura (H) do brilho (V)
- **Aplicação**: R$20 é laranja, R$100 é azul → diferenças de cor claras
- **Tamanho**: 180 features (bins do histograma)

### 2. **LBP Histogram** (Textura)
- Local Binary Patterns captura padrões de contraste local
- **Por quê LBP?**
  - Extremamente resistente a mudanças de iluminação
  - Foca apenas em contraste entre pixels vizinhos
  - Captura texturas: linhas de fundo, rosto, animais
- **Diferencia**: textura de cédula genuína vs falsificação
- **Tamanho**: 59 features (histograma LBP uniforme)

### 3. **Hu Moments** (Geometria)
- 7 momentos invariantes a rotação, escala e translação
- **Por quê Hu Moments?**
  - Se a cédula está de ponta-cabeça ou virada, os momentos permanecem iguais
  - Captura forma geral independente de como é fotografada
  - Log-normalizado para evitar overflow numérico
- **Tamanho**: 7 features

### 4. **Aspect Ratio** (Proporção)
- Largura / Altura da nota
- **Por quê?**
  - Cédulas de Real têm tamanhos diferentes por valor
  - Proporção não muda com rotação/translação
- **Tamanho**: 1 feature

**Total de features**: 180 + 59 + 7 + 1 = **247 features**

## 🤖 Classificador

**Random Forest com 200 árvores**
- Max depth: 20
- Min samples split: 5
- Min samples leaf: 2

Random Forest é escolhido porque:
- Não requer normalização de features individuais
- Robusto a outliers
- Fornece feature importance
- Eficiente com muitas features

## 📁 Estrutura

```
projeto_final/
├── main.py                # Orquestrador principal
├── requirements.txt       # Dependências
├── README.md             # Este arquivo
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Carregamento do Kaggle
│   ├── descriptors.py        # Extração de descritores
│   ├── classifier.py         # Random Forest + avaliação
│   └── visualization.py      # Gráficos
└── output/
    ├── models/               # Modelos .pkl salvos
    └── visualizations/       # PNG dos resultados
```

## 🚀 Como Usar

### 1. Preparar ambiente
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Kaggle (uma única vez)
```bash
# Crie conta em https://www.kaggle.com
# Vá em Account → API → Create New Token
# Isso baixa kaggle.json

# Coloque em:
# Linux/Mac: ~/.kaggle/kaggle.json
# Windows: C:\Users\<seu_usuario>\.kaggle\kaggle.json

# Dê permissão (Linux/Mac):
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Rodar pipeline
```bash
python3 main.py
```

Na primeira execução, isso irá:
1. Baixar dataset do Kaggle (6.8k imagens)
2. Dividir em train (70%), val (15%), test (15%)
3. Extrair descritores de todas as imagens
4. Treinar Random Forest
5. Avaliar na validação e teste
6. Gerar visualizações e salvar modelo

## 📈 Saídas Esperadas

```
output/
├── models/
│   └── banknote_detector.pkl      # Modelo treinado
└── visualizations/
    ├── sample_images.png           # Amostras do dataset
    ├── class_distribution_train.png
    ├── class_distribution_val.png
    ├── class_distribution_test.png
    ├── confusion_matrix_test.png   # Matriz de confusão
    └── feature_importance.png      # Top 30 features
```

## 🔍 Métricas de Avaliação

- **Acurácia**: Percentual total de predições corretas
- **Precisão**: De todas as preditas como falsas, quantas realmente são
- **Recall**: De todas as falsas, quantas foram detectadas
- **F1-Score**: Média harmônica (precisão vs recall)
- **Confusion Matrix**: Análise de erros por classe
- **Feature Importance**: Quais features o modelo mais usa

## 🛠️ Modificações e Experimentos

### Ajustar hiperparâmetros
Em `src/classifier.py`:
```python
self.model = RandomForestClassifier(
    n_estimators=300,   # ← aumentar/diminuir
    max_depth=25,       # ← profundidade
    # ...
)
```

### Adicionar novos descritores
Em `src/descriptors.py`, implemente em `BanknoteDescriptorExtractor`:
```python
def extract_meu_descritor(self, img):
    # Extrai sua feature
    return features
```

E adicione ao método `extract_all_descriptors()`.

### Usar classificador diferente
Crie nova classe em `src/classifier.py` herdando de BaseClassifier.

## 📝 Notas Importantes

- **Dataset**: Automaticamente baixado do Kaggle (6.8k imagens)
- **Tempo**: Primeira execução leva tempo (download + extração de descritores)
- **Memória**: Cuidado ao aumentar tamanho de imagem (padrão: 256x256)
- **Reprodutibilidade**: Use `random_state=42` para resultados consistentes

## 🚨 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'kagglehub'"
```bash
pip install kagglehub
```

### Erro: "PermissionError" ao acessar kaggle.json
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Dataset não baixa
1. Verifique conexão internet
2. Verifique que kaggle.json está no lugar certo
3. Tente em outro momento (servidor Kaggle pode estar lento)

## 📚 Referências

- [HSV Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)
- [Hu Moments](https://en.wikipedia.org/wiki/Image_moment)
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Kaggle API](https://github.com/Kaggle/kagglehub)

## 👥 Autores

Trabalho Final - Processamento de Imagens (SCC0251)

---

**Obs**: O projeto foca em exploração de soluções. Nem todas as ideias funcionam, e tudo bem. Isso também é uma entrega válida! 🎓
