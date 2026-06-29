<div align="center">

# 🦕 Dino Fossil Classifier

### Classificador de espécies de dinossauros por imagem de fóssil

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316?logo=gradio&logoColor=white)](https://www.gradio.app)
[![HF Spaces](https://img.shields.io/badge/Deploy-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

**[🔴 Demo ao vivo](#) · [📊 Resultados](#resultados) · [⚡ Instalação](#instalação) · [📁 Estrutura](#estrutura-do-projeto)**

</div>

---

## Sobre o projeto

Modelo de **visão computacional** treinado com transfer learning sobre **ResNet50** para identificar espécies de dinossauros a partir de fotografias de fósseis. O modelo retorna as **3 espécies mais prováveis** com scores de confiança e inclui **Grad-CAM** para visualizar quais regiões da imagem influenciaram cada predição.

O dataset foi montado a partir de três fontes públicas: Wikimedia Commons, iNaturalist e Kaggle, totalizando aproximadamente 3 000 imagens distribuídas entre 7 espécies.

---

## Funcionalidades

- Upload de imagem → predição das 3 espécies mais prováveis com porcentagem de confiança
- Mapa de ativação Grad-CAM sobreposto à imagem original
- Exportação do modelo em ONNX para inferência sem dependência do PyTorch
- Demo interativa pública via Gradio hospedada no Hugging Face Spaces

---

## Espécies classificadas

| Espécie | Família | Período geológico |
|---------|---------|-------------------|
| *Tyrannosaurus rex* | Tyrannosauridae | Cretáceo Superior |
| *Triceratops horridus* | Ceratopsidae | Cretáceo Superior |
| *Stegosaurus stenops* | Stegosauridae | Jurássico Superior |
| *Diplodocus carnegii* | Diplodocidae | Jurássico Superior |
| *Velociraptor mongoliensis* | Dromaeosauridae | Cretáceo Superior |
| *Brachiosaurus altithorax* | Brachiosauridae | Jurássico Superior |
| *Ankylosaurus magniventris* | Ankylosauridae | Cretáceo Superior |

---

## Estrutura do projeto

```
dino-classifier/
├── data/
│   ├── raw/                    # imagens originais por fonte (não versionadas)
│   │   ├── wikimedia/
│   │   ├── inaturalist/
│   │   └── kaggle/
│   ├── processed/              # dataset limpo e dividido em splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset_stats.csv       # contagem por classe e split
├── docs/
│   ├── especies_alvo.md
│   └── analise_erros.md
├── notebooks/
│   ├── 01_coleta_dados.ipynb
│   ├── 02_eda_imagens.ipynb
│   ├── 03_treinamento.ipynb
│   └── 04_gradcam.ipynb
├── src/
│   ├── scraper_wikimedia.py    # coleta via Wikimedia Commons API
│   ├── scraper_inaturalist.py  # coleta via iNaturalist API
│   ├── dataset.py              # DinoDataset + pipeline de augmentação
│   ├── model.py                # arquitetura ResNet50 adaptada
│   ├── train.py                # loop de treinamento e avaliação
│   └── gradcam.py              # geração de mapas de ativação
├── checkpoints/                # pesos salvos durante treino (não versionados)
├── models/
│   └── dino_classifier.onnx   # artefato final exportado
├── app.py                      # interface Gradio
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalação

### Pré-requisitos

- Python 3.10 ou superior
- CUDA 11.8+ (opcional, mas recomendado para treinamento)

### Passos

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/dino-classifier.git
cd dino-classifier

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Verificar disponibilidade de GPU (opcional)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Como usar

### Demo local

```bash
python app.py
```

Acesse `http://localhost:7860` no navegador. Faça upload de uma foto de fóssil e o modelo retornará as 3 espécies mais prováveis com o mapa Grad-CAM.

### Inferência via código

```python
from PIL import Image
import onnxruntime as ort
import numpy as np

# Carregar modelo ONNX
session = ort.InferenceSession("models/dino_classifier.onnx")

# Pré-processar imagem
image = Image.open("fossil.jpg").resize((224, 224))
tensor = np.array(image).astype(np.float32) / 255.0
tensor = ((tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]

# Predição
logits = session.run(None, {"input": tensor})[0]
probabilidades = np.exp(logits) / np.exp(logits).sum()
top3 = np.argsort(probabilidades[0])[::-1][:3]

ESPECIES = [
    "Tyrannosaurus rex", "Triceratops horridus", "Stegosaurus stenops",
    "Diplodocus carnegii", "Velociraptor mongoliensis",
    "Brachiosaurus altithorax", "Ankylosaurus magniventris"
]

for idx in top3:
    print(f"{ESPECIES[idx]}: {probabilidades[0][idx]:.1%}")
```

### Treinar do zero

```bash
# Fase 1 — base congelada
python src/train.py --phase 1 --epochs 10 --lr 1e-3

# Fase 2 — fine-tuning
python src/train.py --phase 2 --epochs 20 --lr 1e-4 --checkpoint checkpoints/fase1_best.pth
```

---

## Dataset

| Fonte | Tipo | Volume estimado | Link |
|-------|------|-----------------|------|
| Wikimedia Commons | Imagens com licença aberta | ~800 imagens | [API](https://commons.wikimedia.org/w/api.php) |
| iNaturalist | Observações verificadas pela comunidade | ~600 imagens | [API](https://api.inaturalist.org/v1/docs/) |
| Kaggle | Dataset pré-rotulado por espécie | ~1 500 imagens | [Datasets](https://www.kaggle.com/datasets?search=dinosaur+fossil) |

**Split final:** 70% treino · 15% validação · 15% teste

---

## Arquitetura

```
Input (224 × 224 × 3)
        ↓
ResNet50 pré-treinada no ImageNet1K
        ↓
  [Dropout — p=0.3]
        ↓
  Linear (2048 → N classes)
        ↓
  Softmax → Top-3 predições
```

**Treinamento em dois estágios:**

| Estágio | Camadas treináveis | Epochs | Learning rate | Scheduler |
|---------|-------------------|--------|---------------|-----------|
| Fase 1 | Somente `fc` | 10 | 1e-3 | — |
| Fase 2 | `layer3`, `layer4`, `fc` | 15–20 | 1e-4 (base) / 1e-3 (fc) | CosineAnnealingLR |

---

## Resultados

> Métricas serão atualizadas após o treinamento completo.

| Métrica | Valor |
|---------|-------|
| Acurácia top-1 (teste) | — |
| Acurácia top-3 (teste) | — |
| F1-score macro | — |
| F1-score weighted | — |

---

## Stack

| Categoria | Tecnologia |
|-----------|-----------|
| Framework ML | [PyTorch](https://pytorch.org) + [torchvision](https://pytorch.org/vision) |
| Augmentação de dados | [Albumentations](https://albumentations.ai) |
| Rastreamento de experimentos | [Weights & Biases](https://wandb.ai) |
| Interpretabilidade | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| Interface de demo | [Gradio](https://www.gradio.app) |
| Hospedagem da demo | [Hugging Face Spaces](https://huggingface.co/spaces) |
| Exportação do modelo | [ONNX Runtime](https://onnxruntime.ai) |

---

## Licença

Distribuído sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais informações.

---

<div align="center">
Feito com Python · PyTorch · e fósseis reais 🦴
</div>
