# Segmentação do dataset active_matter (The Well)

Trabalho da disciplina de **Processamento de Imagens**. O projeto compara três arquiteturas de segmentação (U-Net, SegFormer e Mask2Former) em um mesmo problema, com e sem etapas de pré-processamento e pós-processamento.

> **Status:** em andamento. As etapas 1 a 3 (inspeção, Ground Truth e pipeline de dados) estão concluídas e validadas no split `valid`. O treinamento dos modelos ainda não começou: aguardamos a definição do professor sobre o uso do dataset (completo ou parcial) e a lista final de pré-processamentos.

## Sumário

1. [Objetivo](#objetivo)
2. [Dataset](#dataset)
3. [Ground Truth](#ground-truth)
4. [Arquivos do projeto](#arquivos-do-projeto)
5. [Guia de instalação e execução](#guia-de-instalação-e-execução)
6. [Estrutura de pastas](#estrutura-de-pastas)
7. [Fluxo de trabalho em equipe (Git)](#fluxo-de-trabalho-em-equipe-git)
8. [Plano experimental](#plano-experimental)
9. [Próximos passos](#próximos-passos)

## Objetivo

Verificar se é possível **identificar regiões de alta concentração de partículas ativas usando apenas os campos de velocidade, orientação e strain-rate**, sem fornecer a concentração à rede. A concentração é usada somente para construir e validar o Ground Truth.

## Dataset

[active_matter](https://polymathic-ai.org/the_well/) do The Well (Polymathic AI): simulações 2D de matéria ativa, grade 256 × 256, 81 instantes por trajetória. Cada arquivo HDF5 corresponde a uma combinação de parâmetros (`L`, `zeta`, `alpha`).

| Campo | Tipo | Componentes | Uso |
|---|---|---|---|
| `concentration` | escalar | 1 | apenas Ground Truth |
| `velocity` | vetorial | 2 | entrada |
| `D` (tensor de orientação) | tensorial | 4 | entrada |
| `E` (tensor strain-rate) | tensorial | 4 | entrada |

**Entrada das redes: 10 canais**, na ordem `vel_x, vel_y, D_xx, D_xy, D_yx, D_yy, E_xx, E_xy, E_yx, E_yy`.

Usamos os splits oficiais do The Well (`train`, `valid`, `test`), que separam os dados por arquivo e, portanto, por trajetória.

## Ground Truth

Máscara binária de regiões de alta concentração, calculada por quadro:

```
d(x,y,t) = c(x,y,t) - 1
M(x,y,t) = 1  se  d(x,y,t) > 1,0 · σ(t),  caso contrário 0
```

em que σ(t) é o desvio padrão de `d` no quadro `t`.

- **Por que relativo?** A concentração é uma pequena flutuação em torno de 1 (desvio de ~0,3%) e sua amplitude cai por ordens de grandeza ao longo do tempo e entre parâmetros. Limiares globais fixos produzem máscaras quase vazias em muitos quadros.
- **Quadros inválidos:** quadros com σ < 1e-5 são marcados como inválidos (a flutuação se aproxima da resolução de float32) e excluídos do treino e da avaliação. No split `valid`, são 58 de 1.944 quadros (3%).
- **Resultado no `valid`:** 24 trajetórias, 1.886 quadros válidos, cerca de 17% de pixels positivos.

## Arquivos do projeto

| Arquivo | Função |
|---|---|
| `inspect_active_matter.py` | Percorre os HDF5, lista campos, dimensões e estatísticas, e gera o histograma e quadros da concentração. |
| `analyze_concentration.py` | Compara critérios candidatos de limiar em todos os arquivos de um split (fração de positivos, desvio padrão ao longo do tempo, prévia das máscaras). |
| `ground_truth.py` | Gera as máscaras do Ground Truth (critério acima), a flag de validade e o resumo em CSV, com figura de validação. |
| `data_loader.py` | `Dataset` PyTorch que lê os 10 canais e a máscara, um quadro por vez, sem carregar tudo na memória. Aceita `transform` para os pré-processamentos e calcula estatísticas por canal. |
| `requirements.txt` | Dependências Python do projeto (exceto o PyTorch, que é instalado à parte). |

## Guia de instalação e execução

Este guia leva do zero (computador sem nada instalado) até rodar todos os scripts atuais. Todos os comandos devem ser executados **na pasta raiz do repositório**, a menos que indicado outro local.

### 1. Pré-requisitos

| Item | Requisito | Como verificar |
|---|---|---|
| Python | versão **3.10 ou superior** (exigência do pacote `the_well`) | `python --version` |
| Git | qualquer versão recente | `git --version` |
| Espaço em disco | o dataset completo tem cerca de 51 GB; o split `valid` é bem menor que o `train`. Comece só pelo `valid`. | |
| GPU | opcional nesta etapa. Placa NVIDIA recomendada para o treinamento futuro. | `nvidia-smi` |

Downloads oficiais, se faltar algo:
- Python: https://www.python.org/downloads/ (no Windows, marque a opção **"Add Python to PATH"** no instalador).
- Git: https://git-scm.com/downloads

### 2. Clonar o repositório

```bash
git clone <URL-DO-REPOSITORIO>
cd <NOME-DA-PASTA-DO-REPOSITORIO>
```

Substitua `<URL-DO-REPOSITORIO>` pelo endereço do repositório no GitHub (botão verde **Code**).

### 3. Criar e ativar o ambiente virtual

O ambiente virtual isola as bibliotecas do projeto e evita conflitos com outras instalações. Crie **uma vez**:

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

Quando estiver ativo, o terminal mostra `(.venv)` no início da linha.

> No PowerShell, se aparecer erro de "execução de scripts desabilitada", rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Se `python` não for reconhecido, tente `py` (Windows) ou `python3` (Linux/macOS).

### 4. Instalar o PyTorch

O PyTorch tem versões diferentes para CPU e para GPU NVIDIA, por isso é instalado **antes** das demais dependências. Escolha uma opção:

**Opção A: sem GPU (CPU apenas)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Opção B: com GPU NVIDIA**

Acesse https://pytorch.org/get-started/locally/, selecione seu sistema, `Pip` e a versão de CUDA, e copie o comando gerado. Ele tem este formato (o `cu121` é só um exemplo e muda conforme a versão de CUDA):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verifique a instalação:
```bash
python -c "import torch; print('torch', torch.__version__, '| GPU disponível:', torch.cuda.is_available())"
```
Sem GPU, `False` é esperado.

### 5. Instalar as demais dependências

```bash
pip install -r requirements.txt
```

Isso instala `numpy`, `h5py`, `matplotlib` e `the_well` (pacote usado para baixar o dataset). Verifique:
```bash
python -c "import numpy, h5py, matplotlib; print('dependências ok')"
the-well-download --help
```

Se o comando `the-well-download` não for encontrado, o ambiente virtual provavelmente não está ativo (passo 3).

### 6. Baixar os dados

Os dados **não ficam no Git** (são muito grandes). Cada integrante baixa os seus. Comece apenas pelo split `valid`:

```bash
the-well-download --base-path ./data/raw --dataset active_matter --split valid
```

Ao terminar, devem existir 16 arquivos `.hdf5` em:
```
data/raw/datasets/active_matter/data/valid/
```

Os splits `train` e `test` usam o mesmo comando, trocando `--split`. Eles ocupam quase todo o espaço do dataset, então **só baixe depois da definição do professor** sobre quanto do dataset usar.

### 7. Executar os scripts (nesta ordem)

**7.1 Inspeção dos dados**
```bash
python inspect_active_matter.py --root ./data/raw/datasets/active_matter
```
Lista a estrutura dos arquivos e salva `hist_concentration.png` e `concentration_frames.png` em `results/inspection/`.

**7.2 Análise dos critérios de limiar**
```bash
python analyze_concentration.py --root ./data/raw/datasets/active_matter/data/valid
```
Salva gráficos e tabelas em `results/inspection/`.

**7.3 Geração do Ground Truth**
```bash
python ground_truth.py --root ./data/raw/datasets/active_matter/data/valid --out ./data/ground_truth --k 1.0 --sigma-floor 1e-5
```
Cria as máscaras em `data/ground_truth/valid/`, o resumo `valid_gt_summary.csv` e a figura `valid_gt_validation.png`. As máscaras não ficam no Git, então cada integrante precisa rodar este passo (leva poucos minutos).

**7.4 Teste do pipeline de dados**
```bash
python data_loader.py --data-root ./data/raw/datasets/active_matter/data --gt-root ./data/ground_truth --split valid --stats-out ./data/channel_stats_valid.json
```
Requer que o passo 7.3 já tenha sido feito para o mesmo split.

**Resultado de referência (split `valid` completo):** se tudo estiver certo, a equipe deve ver exatamente estes números.

| Script | O que conferir |
|---|---|
| `inspect_active_matter.py` | `16 arquivo(s) HDF5 encontrados.` |
| `ground_truth.py` | `24 trajetórias, 1944 quadros`, `58 quadros excluídos (3.0%)`, fração positiva média `0.172` |
| `data_loader.py` | `16 arquivos, 1886 quadros válidos`, `x: (10, 256, 256)`, `y: (1, 256, 256)` |

Se os números diferirem, não siga adiante: veja os problemas comuns abaixo ou avise a equipe.

### 8. Alternativa: Google Colab (sem instalar nada)

Útil para quem não tem GPU. O Colab já traz o PyTorch instalado. Em uma célula:

```python
!git clone <URL-DO-REPOSITORIO>
%cd <NOME-DA-PASTA-DO-REPOSITORIO>
!pip install -r requirements.txt
!the-well-download --base-path ./data/raw --dataset active_matter --split valid
```

Depois, rode os scripts do passo 7 com `!python ...`. O armazenamento do Colab é temporário: para não perder os dados e as máscaras a cada sessão, monte o Google Drive e use `--base-path` e `--out` apontando para uma pasta do Drive.

### 9. Problemas comuns

| Problema | Causa provável | Solução |
|---|---|---|
| `python` não é reconhecido | Python fora do PATH | Reinstale marcando "Add Python to PATH", ou use `py` / `python3` |
| `ModuleNotFoundError: No module named 'torch'` (ou `h5py`) | Ambiente virtual não ativo, ou passo 4/5 não feito | Ative o `.venv` (passo 3) e repita os passos 4 e 5 |
| `the-well-download` não encontrado | Ambiente virtual não ativo | Ative o `.venv` |
| `Nenhum arquivo HDF5 encontrado` | Caminho de `--root` errado ou download incompleto | Confira se `data/raw/datasets/active_matter/data/valid/` existe e tem 16 arquivos |
| `Ground Truth ausente` no `data_loader.py` | Passo 7.3 não executado para esse split | Rode o `ground_truth.py` para o mesmo split |
| `torch.cuda.is_available()` dá `False` com GPU NVIDIA | PyTorch instalado na versão CPU | Reinstale com a opção B do passo 4 e atualize o driver da NVIDIA |
| Erros de caminho muito longo no Windows | Nomes de arquivo longos do dataset | Clone o repositório em uma pasta curta (ex.: `C:\proj`) e ative caminhos longos no Windows |
| Aviso `LF will be replaced by CRLF` ao usar `git add` | Diferença de quebra de linha entre Windows e Linux | É só um aviso; pode ignorar |
| Download interrompido | Conexão ou espaço em disco | Confira o espaço livre e rode o comando de download de novo |

## Estrutura de pastas

```
.
├── inspect_active_matter.py
├── analyze_concentration.py
├── ground_truth.py
├── data_loader.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/datasets/active_matter/data/{train,valid,test}/   # dados do The Well (fora do Git)
│   ├── ground_truth/{split}/                                 # máscaras .hdf5 (fora do Git)
│   ├── ground_truth/{split}_gt_summary.csv                   # resumo do Ground Truth
│   └── channel_stats_valid.json                              # estatísticas por canal (provisório)
└── results/inspection/                                       # figuras da inspeção
```

Pastas e arquivos que vão surgir nas próximas etapas: `models/`, `configs/`, `experiments/`, `checkpoints/`, `predictions/` e `results/*.csv`.

## Fluxo de trabalho em equipe (Git)

- Antes de começar a trabalhar: `git pull`.
- Faça commits pequenos, um por etapa ou módulo concluído, com mensagem clara (ex.: `Etapa 3: data_loader validado no split valid`).
- **Nunca** adicione ao Git dados (`.hdf5`), checkpoints ou predições. O `.gitignore` já cobre essas pastas; confira o `git status` antes de cada commit.
- Alterações em módulos compartilhados (`ground_truth.py`, `data_loader.py` e os próximos) devem ser combinadas com a equipe, pois mudam os resultados de todos.
- Depois de trabalhar: `git add`, `git commit`, `git push`.

## Plano experimental

As três arquiteturas (U-Net, SegFormer e Mask2Former) passam pelos mesmos quatro experimentos, todos com o mesmo Ground Truth:

1. Arquitetura original (baseline).
2. Pré-processamento + arquitetura.
3. Arquitetura + pós-processamento.
4. Pré-processamento + arquitetura + pós-processamento.

Métricas: IoU, Dice, Precision, Recall, F1, latência, memória e estabilidade temporal. Os pós-processamentos são aplicados às predições salvas, sem retreinar os modelos.

A lista final de pré e pós-processamentos e o tamanho do dataset usado dependem da definição do professor e serão atualizados aqui.

## Próximos passos

- [x] Inspeção do dataset
- [x] Ground Truth (versão 1)
- [x] Pipeline de dados (`data_loader.py`)
- [ ] Definições do professor (dataset e pré-processamentos)
- [ ] `metrics.py` e `training.py` (compartilhados pelas três arquiteturas)
- [ ] Implementar U-Net, SegFormer e Mask2Former
- [ ] Experimento 1 (baseline) nas três arquiteturas
- [ ] Pré e pós-processamentos (compartilhados)
- [ ] Experimentos 2 a 4 nas três arquiteturas
- [ ] Análise comparativa e relatório
