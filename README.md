# Segmentação do dataset active_matter (The Well)

Projeto da disciplina de **Processamento de Imagens**. Compara três arquiteturas de segmentação (U-Net, SegFormer e Mask2Former) em um mesmo problema, avaliando o efeito de dez técnicas de pré-processamento e cinco de pós-processamento, todas sob o mesmo protocolo experimental e o mesmo Ground Truth.

## Sumário

1. [Objetivo](#objetivo)
2. [Dataset](#dataset)
3. [Preparação dos dados e Ground Truth](#preparação-dos-dados-e-ground-truth)
4. [Arquiteturas](#arquiteturas)
5. [Pré e pós-processamento](#pré-e-pós-processamento)
6. [Protocolo experimental](#protocolo-experimental)
7. [Estrutura do repositório](#estrutura-do-repositório)
8. [Guia de instalação e execução](#guia-de-instalação-e-execução)
9. [Módulos do projeto](#módulos-do-projeto)
10. [Uso em Python](#uso-em-python)
11. [Limitações e observações](#limitações-e-observações)
12. [Fluxo de trabalho em equipe (Git)](#fluxo-de-trabalho-em-equipe-git)
13. [Próximos passos](#próximos-passos)

## Objetivo

Verificar se é possível **identificar regiões de alta concentração de partículas ativas usando apenas os campos de velocidade, orientação e strain-rate**, sem fornecer a concentração à rede. A concentração é usada somente para construir e validar o Ground Truth. Em seguida, compara-se como cada arquitetura, cada pré-processamento e cada pós-processamento influenciam a qualidade da segmentação, o custo computacional e a estabilidade temporal das máscaras.

## Dataset

[active_matter](https://polymathic-ai.org/the_well/), do The Well (Polymathic AI): simulações 2D de matéria ativa em grade 256 × 256 com 81 instantes por trajetória. Cada arquivo HDF5 corresponde a uma combinação de parâmetros (`L`, `zeta`, `alpha`), com condições de contorno periódicas.

| Campo | Tipo | Componentes | Uso |
|---|---|---|---|
| `concentration` | escalar | 1 | apenas Ground Truth |
| `velocity` | vetorial | 2 | entrada |
| `D` (tensor de orientação) | tensorial | 4 | entrada |
| `E` (tensor strain-rate) | tensorial | 4 | entrada |

**Entrada das redes: 10 canais**, na ordem `vel_x, vel_y, D_xx, D_xy, D_yx, D_yy, E_xx, E_xy, E_yx, E_yy`.

Usam-se os splits oficiais do The Well, que separam os dados por arquivo e, portanto, por trajetória:

| Split | Arquivos | Trajetórias |
|---|---|---|
| `train` | 45 (uma combinação de zeta e alpha por arquivo) | 175 |
| `valid` | 16 | 24 |
| `test` | 21 | 26 |

## Preparação dos dados e Ground Truth

### Subconjunto e resolução

- **Redução espacial:** de 256 × 256 para **128 × 128** por média de blocos 2 × 2 (redução com antialiasing, sem aumentar a resolução). O erro relativo de reconstrução é de no máximo 0,14 nos canais e 0,12 na concentração, ou seja, mais de 98% da energia dos campos é preservada.
- **Seleção temporal:** todas as trajetórias, com **1 quadro a cada 3** (t = 0, 3, 6, ..., 78), 27 quadros por trajetória. O subconjunto equivale a cerca de **18,4 GB** dos dados originais, abaixo do limite de 25 GB.
- Os índices, o método e os tamanhos ficam registrados em `data/processed/manifest.json`.

| Split | Quadros | Quadros válidos |
|---|---|---|
| `train` | 4.725 | 4.583 |
| `valid` | 648 | 629 |
| `test` | 702 | 683 |

### Ground Truth

Máscara binária de regiões de alta concentração, calculada por quadro sobre a concentração já reduzida:

```
d(x,y,t) = c(x,y,t) - 1
M(x,y,t) = 1  se  d(x,y,t) > 1,0 · σ(t),  caso contrário 0
```

em que σ(t) é o desvio padrão de `d` no quadro `t`.

- **Por que um limiar relativo?** A concentração é uma pequena flutuação em torno de 1 (desvio de ~0,3%), e sua amplitude cai por ordens de grandeza ao longo do tempo e entre combinações de parâmetros. Limiares globais fixos produzem máscaras quase vazias em muitos quadros.
- **Quadros inválidos:** quadros com σ < 1e-5 têm flutuação próxima da resolução de float32 e são marcados como inválidos, sendo excluídos do treino e da avaliação (3,0% no `train`, 2,9% no `valid`, 2,7% no `test`).
- **Fração de pixels positivos:** 0,177 no `train`, 0,172 no `valid` e 0,173 no `test`.
- **Validação da resolução:** a máscara gerada em 128 px foi comparada, no `valid`, com a máscara de 256 px reduzida por nearest-neighbor: IoU médio de 0,955, concordância de pixels de 99,2% e razão entre os σ de 0,999.

### Normalização

Estatísticas por canal (média, desvio, RMS, quantis, extremos e percentis 1 e 99) calculadas **somente no `train`** e guardadas em `data/channel_stats_train.json`. A normalização por Z-score é o baseline (P1).

## Arquiteturas

| Arquitetura | Descrição | Inicialização |
|---|---|---|
| **U-Net** | CNN codificador-decodificador de 4 níveis (32 a 512 canais), com conexões de salto; da ordem de 8 milhões de parâmetros | do zero |
| **SegFormer** | Transformer hierárquico MiT-B0 (cerca de 3,7 milhões de parâmetros); logits em 1/4 da resolução ampliados até 128 px | codificador com pesos do ImageNet |
| **Mask2Former** | Transformer com previsão de máscaras, backbone Swin-T e 20 consultas; perda própria com casamento húngaro; probabilidade de primeiro plano pela inferência semântica padrão | backbone com pesos do ImageNet |

Nos transformers, a primeira camada é adaptada de 3 para 10 canais pela média dos pesos RGB replicada e reescalada. As três redes recebem a mesma entrada e produzem a mesma máscara binária. Os hiperparâmetros iniciais de cada uma ficam em `configs/<arquitetura>.yaml`.

## Pré e pós-processamento

### Pré-processamento (P1 a P10)

Aplicados individualmente. O **P1 (Z-score) é a condição de referência (baseline)**. Os filtros P6 a P10 atuam sobre os dados já normalizados por Z-score, canal a canal, e tratam as bordas de forma periódica, como o domínio da simulação.

| ID | Método | Definição |
|---|---|---|
| P1 | Z-score (baseline) | (x − média) / desvio |
| P2 | RMS | x / RMS |
| P3 | Robust scaling | (x − mediana) / (q75 − q25) |
| P4 | Min-Max | (x − mín) / (máx − mín) |
| P5 | Percentile clipping | corte em [p1, p99] e Z-score dos dados cortados |
| P6 | Gaussian smoothing | σ = 1,0 px |
| P7 | Median filtering | janela 3 × 3 |
| P8 | Bilateral filtering | janela 5 × 5, σ espacial 1,5 px, σ de cor 0,5 |
| P9 | Anisotropic diffusion | Perona-Malik, 10 iterações, κ = 0,5 |
| P10 | Wavelet denoising | db2, 2 níveis, limiar universal suave |

### Pós-processamento (Q1 a Q5)

Aplicados às probabilidades salvas de cada modelo, sem retreinar.

| ID | Método |
|---|---|
| Q1 | Threshold optimization (limiar escolhido na validação e congelado) |
| Q2 | Morphological opening |
| Q3 | Morphological closing |
| Q4 | Connected-component filtering |
| Q5 | Temporal consistency filtering (sobre as sequências ordenadas por trajetória) |

Os parâmetros de P6 a P10 e de Q2 a Q5 são escolhidos no conjunto de validação e registrados junto aos resultados.

## Protocolo experimental

As três arquiteturas passam pelos mesmos quatro experimentos, todos com o mesmo Ground Truth:

| Experimento | Conteúdo | Configurações por arquitetura |
|---|---|---|
| 1 | Baseline: P1, sem pós-processamento | 1 |
| 2 | P2 a P10, sem pós-processamento | 9 |
| 3 | Baseline + Q1 a Q5 | 5 |
| 4 | P2 a P10 × Q1 a Q5 | 45 |

São **60 configurações por arquitetura e 180 no total**, obtidas com **30 treinamentos** (3 arquiteturas × 10 condições de pré-processamento). Os pós-processamentos são aplicados às predições salvas.

- **Treino:** no máximo **20 épocas** por execução, com o mesmo limite para todas as arquiteturas. A cada época avalia-se o Dice no conjunto de validação e salva-se o melhor checkpoint (`best_model.pt`), registrando a época do pico. Early stopping é opcional.
- **Reprodutibilidade:** seeds, hiperparâmetros, estatísticas de normalização e parâmetros dos filtros são registrados em cada execução.
- **Isolamento do teste:** o split `test` só é usado na avaliação final, depois de congelados os modelos e os parâmetros.
- **Métricas:** IoU e Dice (principais); Precision, Recall e IoU do fundo; latência de inferência, memória máxima de GPU e estabilidade temporal; intervalos de confiança por bootstrap sobre trajetórias.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── exploration/                  # etapas exploratórias
│   ├── inspect_active_matter.py
│   └── analyze_concentration.py
├── prepare_dataset.py            # seleção de quadros, redução para 128 px e manifesto
├── ground_truth.py               # máscaras de alta concentração e validade dos quadros
├── data_loader.py                # Dataset PyTorch (10 canais + máscara)
├── preprocessing.py              # pré-processamentos P1 a P10
├── metrics.py                    # IoU, Dice, IC, estabilidade temporal, latência e memória
├── training.py                   # loop de treino (20 épocas, melhor checkpoint, predições)
├── test_models.py                # teste de aceite dos modelos
├── models/
│   ├── __init__.py               # build_model(arch) e load_arch_config
│   ├── common.py                 # adaptação da primeira camada para 10 canais
│   ├── unet.py
│   ├── segformer.py
│   └── mask2former.py
├── configs/
│   ├── unet.yaml
│   ├── segformer.yaml
│   └── mask2former.yaml
├── data/
│   ├── processed/                # dados reduzidos (fora do Git); manifest.json e manifest_files.csv versionados
│   ├── ground_truth/             # máscaras (fora do Git); manifesto, CSVs e figuras versionados
│   └── channel_stats_train.json  # estatísticas por canal do train
├── checkpoints/                  # pesos e histórico de treino (fora do Git)
├── predictions/                  # probabilidades salvas (fora do Git)
└── results/                      # resultados e figuras
```

Todos os comandos são executados **na raiz do repositório**.

## Guia de instalação e execução

Este guia leva de um computador sem nada instalado até a execução de todos os módulos atuais.

### 1. Pré-requisitos

| Item | Requisito | Como verificar |
|---|---|---|
| Python | versão **3.10 ou superior** | `python --version` |
| Git | qualquer versão recente | `git --version` |
| Espaço em disco | cerca de **5 GB** para os dados processados. O download passa por arquivos de até ~1,2 GB, que são descartados após o processamento | |
| GPU | opcional para os testes; **recomendada** (NVIDIA, por exemplo Colab T4) para o treinamento | `nvidia-smi` |

Downloads, se faltar algo:
- Python: https://www.python.org/downloads/ (no Windows, marque **"Add Python to PATH"** no instalador).
- Git: https://git-scm.com/downloads

### 2. Clonar o repositório

```bash
git clone <URL-DO-REPOSITORIO>
cd <NOME-DA-PASTA-DO-REPOSITORIO>
```

### 3. Criar e ativar o ambiente virtual

Crie **uma vez**:
```bash
python -m venv .venv
```

Ative **toda vez que abrir um terminal novo**:

| Terminal | Comando |
|---|---|
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| PowerShell (Windows) | `.venv\Scripts\Activate.ps1` |
| Prompt de Comando (Windows) | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

Com o ambiente ativo, o terminal mostra `(.venv)` no início da linha. No PowerShell, se aparecer erro de execução de scripts desabilitada, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Se `python` não for reconhecido, tente `py` (Windows) ou `python3` (Linux/macOS).

### 4. Instalar o PyTorch

O PyTorch tem versões para CPU e para GPU NVIDIA, por isso é instalado **antes** das demais dependências.

**Sem GPU (CPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Com GPU NVIDIA:** acesse https://pytorch.org/get-started/locally/, escolha seu sistema, `Pip` e a versão de CUDA, e copie o comando gerado. Ele tem este formato (o `cu121` é só um exemplo):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verifique:
```bash
python -c "import torch; print('torch', torch.__version__, '| GPU disponível:', torch.cuda.is_available())"
```

### 5. Instalar as demais dependências

```bash
pip install -r requirements.txt
```

Instala `numpy`, `h5py`, `matplotlib`, `scipy`, `PyWavelets`, `transformers`, `pyyaml`, `huggingface_hub` e `the_well`. Verifique:
```bash
python -c "import numpy, h5py, scipy, pywt, transformers, yaml; print('dependências ok')"
```

### 6. Baixar e preparar os dados

Os dados **não ficam no Git**. O `prepare_dataset.py` baixa cada arquivo do Hugging Face, seleciona os quadros, reduz para 128 px, salva o resultado compacto e descarta o original, de modo que o disco nunca guarda mais de um arquivo bruto por vez:

```bash
python prepare_dataset.py --splits valid test train --source hf --tmp ./data/tmp_download
```

O processo é demorado (são mais de 40 GB baixados no total). Se for interrompido, **repita o mesmo comando**: os arquivos já processados são pulados. Ao final, o bloco `=== Manifesto ===` resume os três splits. Depois, apague a pasta temporária:
```bash
rm -rf ./data/tmp_download
```

**Opcional (verificação da resolução):** para o Ground Truth poder ser comparado com a versão de 256 px no `valid`, baixe também os arquivos originais desse split:
```bash
the-well-download --base-path ./data/raw --dataset active_matter --split valid
```

### 7. Gerar o Ground Truth

```bash
python ground_truth.py --splits valid test train
```

Com os originais do `valid` baixados, acrescente `--compare-raw ./data/raw/datasets/active_matter/data` para incluir a comparação com 256 px. As máscaras ficam em `data/ground_truth/<split>/`, e o script grava também resumos em CSV, figuras de validação e `ground_truth_manifest.json`. As máscaras não ficam no Git, então cada integrante gera as suas (leva poucos minutos).

### 8. Estatísticas de normalização

```bash
python data_loader.py --split train --stats-out ./data/channel_stats_train.json
python preprocessing.py --add-clip-stats --stats data/channel_stats_train.json
```

O primeiro comando testa o pipeline de dados e calcula as estatísticas por canal; o segundo acrescenta as estatísticas dos dados cortados, usadas pelo P5. O arquivo `data/channel_stats_train.json` é versionado, então só precisa ser regerado se o subconjunto mudar.

### 9. Testes

```bash
python metrics.py                    # autoteste das métricas
python preprocessing.py              # autoteste de P1 a P10
python test_models.py                # forma, perda e gradiente dos três modelos
python test_models.py --pretrained   # idem, baixando os pesos do ImageNet (precisa de internet)
python training.py --smoke-test      # fluxo de treino completo com uma rede minúscula
```

### 10. Resultados de referência

Se tudo estiver correto, a equipe deve ver exatamente estes números.

| Etapa | O que conferir |
|---|---|
| `prepare_dataset.py` | `train: 45 arquivos, 175 trajetórias, 4725 quadros`; `valid: 16 arquivos, 24 trajetórias, 648 quadros`; `test: 21 arquivos, 26 trajetórias, 702 quadros`; subconjunto total de 18,37 GB, dentro do teto |
| `ground_truth.py` | `train` 142 quadros excluídos (3,0%), fração positiva 0,177; `valid` 19 excluídos (2,9%), 0,172; `test` 19 excluídos (2,7%), 0,173 |
| `data_loader.py --split train` | `4583 quadros válidos`, `x: (10, 128, 128)`, `y: (1, 128, 128)`, espaçamento temporal 3 |
| `metrics.py`, `preprocessing.py` | `todos os testes passaram` |

Se os números diferirem, veja os problemas comuns abaixo antes de continuar.

### 11. Alternativa: Google Colab

O Colab já traz o PyTorch instalado. Em uma célula:
```python
!git clone <URL-DO-REPOSITORIO>
%cd <NOME-DA-PASTA-DO-REPOSITORIO>
!pip install -r requirements.txt
!python prepare_dataset.py --splits valid test train --source hf --tmp ./data/tmp_download
```
O armazenamento do Colab é temporário. Mantenha os dados processados no disco local da sessão (mais rápido) e salve checkpoints e resultados no Google Drive montado. O `training.py` retoma automaticamente a partir do `last.pt` se a sessão for interrompida.

### 12. Problemas comuns

| Problema | Causa provável | Solução |
|---|---|---|
| `python` não é reconhecido | Python fora do PATH | Reinstale marcando "Add Python to PATH", ou use `py` / `python3` |
| `ModuleNotFoundError` (`torch`, `h5py`, `transformers`...) | Ambiente virtual inativo ou passos 4 e 5 não feitos | Ative o `.venv` e repita os passos 4 e 5 |
| `Nenhum arquivo processado em data/processed/...` | Dados ainda não preparados para esse split | Rode o `prepare_dataset.py` (passo 6) |
| `Ground Truth ausente` no `data_loader.py` | `ground_truth.py` não executado para o split | Rode o passo 7 |
| `P5 precisa das estatísticas dos dados cortados` | Falta o `--add-clip-stats` | Rode o passo 8 |
| Erro ao baixar pesos pré-treinados | Sem internet ou limite do Hugging Face | Rode `test_models.py` sem `--pretrained` para testar offline e tente de novo depois |
| `torch.cuda.is_available()` dá `False` com GPU NVIDIA | PyTorch instalado na versão CPU | Reinstale com a opção de GPU do passo 4 e atualize o driver |
| Erros de caminho muito longo no Windows | Nomes longos dos arquivos do dataset | Clone o repositório em uma pasta curta (ex.: `C:\proj`) e ative caminhos longos no Windows |
| Aviso `LF will be replaced by CRLF` no `git add` | Diferença de quebra de linha entre Windows e Linux | É só um aviso; pode ignorar |
| Download interrompido | Conexão ou espaço em disco | Confira o espaço livre e repita o mesmo comando |

## Módulos do projeto

| Módulo | Função |
|---|---|
| `exploration/inspect_active_matter.py` | Lê os HDF5, lista campos, dimensões e estatísticas, e gera histograma e quadros da concentração |
| `exploration/analyze_concentration.py` | Compara critérios de limiar em todos os arquivos de um split (fração de positivos, σ no tempo, prévia das máscaras) |
| `prepare_dataset.py` | Seleciona 1 quadro a cada 3, reduz para 128 px, salva em formato compacto e gera o manifesto com o erro de reconstrução. Funciona com arquivos locais ou direto do Hugging Face, com retomada |
| `ground_truth.py` | Gera as máscaras a partir da concentração reduzida, a flag de validade e os resumos; compara com a versão de 256 px |
| `data_loader.py` | `Dataset` PyTorch que lê um quadro por vez, exclui quadros inválidos, expõe sequências temporais por trajetória e calcula as estatísticas por canal |
| `preprocessing.py` | P1 a P10 como transformações para o `data_loader`, com `params` para registro |
| `metrics.py` | IoU, Dice, Precision, Recall, IoU do fundo, bootstrap por trajetória, estabilidade temporal, latência e memória |
| `training.py` | Treino genérico para as três arquiteturas: limite de 20 épocas, melhor checkpoint pelo Dice de validação, early stopping opcional, retomada e salvamento das probabilidades |
| `models/` e `configs/` | U-Net, SegFormer e Mask2Former, a fábrica `build_model` e os hiperparâmetros de cada uma |
| `test_models.py` | Verifica forma de saída, probabilidades em [0, 1], entrada de 10 canais, perda e gradiente de cada modelo |

Ordem de uso:
```
prepare_dataset → ground_truth → data_loader (+ preprocessing) → models → training → predictions/
```

## Uso em Python

```python
from data_loader import ActiveMatterSeg, make_loader
from preprocessing import load_stats, build_transform
from models import build_model, load_arch_config

stats = load_stats("data/channel_stats_train.json")
T = build_transform("P1", stats)                        # P1 = baseline; use "P2" ... "P10" para os demais

train_ds = ActiveMatterSeg("data/processed", "data/ground_truth", "train", transform=T)
x, y, meta = train_ds[0]                                # x: (10,128,128)  y: (1,128,128)

cfg = load_arch_config("unet")                          # configs/unet.yaml
model = build_model("unet", in_channels=10, **cfg["model"])
```

O treino é feito por `training.train(model, train_loader, val_loader, TrainConfig(...))`, que grava em `checkpoints/<arquitetura>/<P>/seed<k>/` o `best_model.pt`, o `last.pt`, o histórico por época e o resumo da execução.

## Limitações e observações

- **Ground Truth por limiar:** a máscara vem de um limiar sobre a concentração, então os resultados medem a capacidade de reproduzir um critério definido por essa variável a partir dos outros campos físicos, e não uma anotação independente.
- **Redundância dos canais:** `D` e `E` são tensores simétricos (`D` com traço 1 e `E` com traço nulo), então os 10 canais de entrada carregam 6 informações independentes.
- **Espaçamento temporal de 3 quadros:** o filtro de consistência temporal (Q5) e a estabilidade temporal operam sobre quadros com Δt = 3.
- **Inicialização:** U-Net treina do zero e os transformers partem de pesos do ImageNet adaptados para 10 canais. A diferença de partida e de tamanho entre as arquiteturas (da ordem de 3,7 milhões de parâmetros no SegFormer-B0 contra dezenas de milhões no Mask2Former) é considerada na comparação de custo.
- **Resolução:** as redes trabalham em 128 × 128; as máscaras de 128 px têm bordas em degraus de pixel.

## Fluxo de trabalho em equipe (Git)

- Antes de começar: `git pull`.
- Commits pequenos, um por módulo ou etapa concluída, com mensagem clara (ex.: `Etapa 6: data_loader v2`).
- **Nunca** versionar dados (`.hdf5`), checkpoints ou predições. O `.gitignore` já cobre essas pastas; confira o `git status` antes de cada commit.
- Alterações nos módulos compartilhados (`ground_truth.py`, `data_loader.py`, `preprocessing.py`, `metrics.py`, `training.py`) devem ser combinadas com a equipe, pois mudam os resultados de todos.
- Para versionar os manifestos de `data/processed/`, que ficam em pasta ignorada, use `git add -f data/processed/manifest.json data/processed/manifest_files.csv`.

## Próximos passos

- [x] Inspeção do dataset e critério do Ground Truth
- [x] Preparação dos dados (128 px, 1 quadro a cada 3) e manifesto
- [x] Ground Truth validado nos três splits
- [x] Pipeline de dados, métricas e treino (20 épocas, melhor checkpoint)
- [x] Pré-processamentos P1 a P10
- [x] Modelos U-Net, SegFormer e Mask2Former
- [ ] Script de experimento (dados + pré + modelo + treino + predições) e benchmark de tempo por época
- [ ] Experimento 1: baseline nas três arquiteturas
- [ ] Experimento 2: P2 a P10 nas três arquiteturas
- [ ] Pós-processamentos Q1 a Q5 e avaliação (CSV de resultados)
- [ ] Experimentos 3 e 4
- [ ] Análise comparativa (tabelas, heatmaps de Dice e IoU, intervalos de confiança) e relatório final
