# Classificação de Pneumonia Pediátrica em Raios-X com Transfer Learning

> **Aviso:** Este projeto é um exercício acadêmico. Não constitui dispositivo médico e não deve ser empregado em decisões diagnósticas sem supervisão profissional qualificada.

---

## Visão Geral

Este projeto implementa um classificador binário (**NORMAL vs. PNEUMONIA**) para radiografias de tórax pediátricas utilizando Transfer Learning. As arquiteturas EfficientNet-B3 e ResNet18, pré-treinadas no ImageNet, foram adaptadas ao domínio médico por meio de fine-tuning progressivo com tratamento explícito de desbalanceamento de classes.

O trabalho foi desenvolvido no contexto da disciplina **Projetos de IA (PPI SOFTEX / Residência TIC)**.

---

## Dataset

| Item | Detalhe |
|------|---------|
| **Fonte** | Chest X-Ray Images (Pneumonia) — Kaggle |
| **Repositório canônico** | Mendeley Data v2, DOI `10.17632/rscbjbr9sj.2` |
| **Licença** | CC BY 4.0 |
| **Artigo original** | Kermany et al., *Cell* 172(5):1122–1131, 2018 |
| **Volume** | 5.856 imagens JPEG |
| **Classes** | NORMAL (1.583) e PNEUMONIA (4.273) |
| **Desbalanceamento** | 2,7:1 a favor de PNEUMONIA |

**Distribuição dos splits:**

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Treino | 1.341 | 3.875 | 5.216 |
| Val oficial | 8 | 8 | 16 |
| Teste | 234 | 390 | 624 |

> O split de validação oficial (16 imagens) é insuficiente para monitorar overfitting. O pipeline cria internamente um split 85/15 a partir do treino (~782 imagens de validação), mantendo o teste oficial intacto.

---

## Metodologia

**Abordagem:** Classificação binária supervisionada com Transfer Learning.

**Pré-processamento:**
- Redimensionamento para 224×224 pixels
- Normalização com média/desvio do ImageNet
- Data augmentation no treino: flip horizontal, rotação ±15°, variação de brilho/contraste (ColorJitter)

**Tratamento de desbalanceamento:**
- Pesos de classe inversamente proporcionais à frequência na `CrossEntropyLoss`
- Seleção do melhor modelo por **AUC-ROC** (não acurácia)

**Protocolo de validação:**
- Semente fixa: `SEED = 42`
- Split estratificado 85/15 interno
- Conjunto de teste isolado — avaliado **uma única vez** por modelo

---

## Baselines

| Modelo | AUC-ROC | F1-macro | Acurácia |
|--------|---------|----------|----------|
| Trivial — sempre PNEUMONIA | — | 0,3846 | 0,6250 |
| Trivial — estratificado | — | 0,4716 | 0,5385 |
| CNN rasa (3 blocos conv, sem pré-treino) | 0,8618 | 0,7702 | 0,7885 |

---

## Métricas e Critérios de Sucesso

| Métrica | Meta | Justificativa |
|---------|------|---------------|
| AUC-ROC | ≥ 0,90 | Avalia discriminabilidade independentemente do limiar de decisão |
| F1-macro | ≥ 0,82 | Equilibra performance entre classes desbalanceadas |
| Recall (PNEUMONIA) | ≥ 0,90 | Minimiza falsos negativos — clinicamente mais grave |
| Acurácia | — | Referência secundária |

---

## Resultados

| Modelo | AUC-ROC | Acurácia | F1-macro | Recall (PNEUMONIA) |
|--------|---------|----------|----------|--------------------|
| EfficientNet-B3 | 0,951 | 0,867 | 0,865 | 0,985 |
| ResNet18 | 0,971 | 0,877 | 0,857 | 0,997 |

**Todas as metas foram superadas por ambos os modelos.**

**Resultados por classe:**

| Modelo | Classe | Precision | Recall | F1 |
|--------|--------|-----------|--------|----|
| EfficientNet-B3 | NORMAL | 0,97 | 0,66 | 0,79 |
| EfficientNet-B3 | PNEUMONIA | 0,83 | 0,99 | 0,90 |
| ResNet18 | NORMAL | 0,99 | 0,68 | 0,80 |
| ResNet18 | PNEUMONIA | 0,84 | 1,00 | 0,91 |

**Threshold ótimo por F2-score** (recall ponderado 2× sobre precisão):

| Modelo | Threshold | Recall | Precision |
|--------|-----------|--------|-----------|
| EfficientNet-B3 | 0,869 | 0,985 | — |
| ResNet18 | 0,612 | 0,997 | — |

---

## Explicabilidade

Mapas de ativação **Grad-CAM** foram gerados para imagens do conjunto de teste. O modelo foca em regiões pulmonares com infiltrado ou consolidação — padrões radiográficos característicos da pneumonia — evidenciando que o aprendizado é clinicamente coerente.

---

## Estrutura do Repositório

```
pneumonia-chest-xray/
├── pneumonia-chest-xray.ipynb   # Pipeline completo documentado
├── figures/
│   ├── training_curves.png      # Curvas de loss e AUC por época
│   ├── confusion_matrix_efficientnet_b3.png
│   ├── confusion_matrix_resnet18.png
│   └── gradcam_efficientnet_b3.png
├── models/                      # Checkpoints dos melhores modelos (.pt)
├── requirements.txt
└── README.md
```

---

## Como Reproduzir

### Requisitos

- Python 3.10+
- PyTorch 2.x (GPU recomendada)
- Dependências: `timm`, `pytorch-grad-cam`, `scikit-learn`, `matplotlib`, `seaborn`, `kagglehub`

### Instalação

```bash
git clone https://github.com/pauloeliezerg/pneumonia-chest-xray.git
cd pneumonia-chest-xray
pip install -r requirements.txt
```

### Obter os dados

O dataset é baixado automaticamente via `kagglehub` ao executar o notebook. É necessário um token de API Kaggle configurado no ambiente:

```bash
# Configure o token em ~/.kaggle/kaggle.json antes de executar
```

### Executar o pipeline

Abra e execute o notebook `pneumonia-chest-xray.ipynb` sequencialmente no Google Colab ou em ambiente local com GPU.

---

## Citação obrigatória do dataset

```
Kermany, D. S. et al. Identifying Medical Diagnoses and Treatable Diseases
by Image-Based Deep Learning. Cell, v. 172, n. 5, p. 1122–1131, 2018.
DOI: 10.1016/j.cell.2018.02.010
```

---

*Disciplina: Projetos de IA — Módulo 7 (PPI/SiDi) | Período: Abr–Jul/2026*