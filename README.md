# Segmentação do dataset active_matter (The Well)

Trabalho da disciplina de **Processamento de Imagens**. O projeto compara três arquiteturas de segmentação (U-Net, SegFormer e Mask2Former) em um mesmo problema, com e sem etapas de pré-processamento e pós-processamento.

> **Status:** em andamento. As etapas 1 a 3 (inspeção, Ground Truth e pipeline de dados) estão concluídas e validadas no split `valid`. O treinamento dos modelos ainda não começou: aguardamos a definição do professor sobre o uso do dataset (completo ou parcial) e a lista final de pré-processamentos.

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

Usamos os splits oficiais do The Well (`train`, `valid`, `test`), que separam os dados por arquivo, e portanto por trajetória.

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

## Arquivos

| Arquivo | Função |
|---|---|
| `inspect_active_matter.py` | Percorre os HDF5, lista campos, dimensões e estatísticas, e gera o histograma e quadros da concentração. |
| `analyze_concentration.py` | Compara critérios candidatos de limiar em todos os arquivos de um split (fração de positivos, desvio padrão ao longo do tempo, prévia das máscaras). |
| `ground_truth.py` | Gera as máscaras do Ground Truth (critério acima), a flag de validade e o resumo em CSV, com figura de validação. |
| `data_loader.py` | `Dataset` PyTorch que lê os 10 canais e a máscara, um quadro por vez, sem carregar tudo na memória. Aceita `transform` para os pré-processamentos e calcula estatísticas por canal. |

## Como executar

Dependências: `torch`, `h5py`, `numpy`, `matplotlib` e `the_well` (para baixar os dados).

```bash
# 1. Inspeção dos dados
python inspect_active_matter.py --root ./data/raw/datasets/active_matter

# 2. Comparação de critérios de limiar (split valid)
python analyze_concentration.py --root ./data/raw/datasets/active_matter/data/valid

# 3. Geração do Ground Truth
python ground_truth.py --root ./data/raw/datasets/active_matter/data/valid \
                       --out ./data/ground_truth --k 1.0 --sigma-floor 1e-5

# 4. Teste do pipeline de dados e estatísticas por canal
python data_loader.py --data-root ./data/raw/datasets/active_matter/data \
                      --gt-root ./data/ground_truth --split valid \
                      --stats-out ./data/channel_stats_valid.json
```

As estatísticas de canal geradas no `valid` são provisórias: para os pré-processamentos finais devem ser recalculadas no `train`.

## Estrutura de pastas

```
data/
  raw/datasets/active_matter/data/{train,valid,test}/   # dados do The Well (fora do Git)
  ground_truth/{split}/                                 # máscaras .hdf5 (fora do Git)
  ground_truth/{split}_gt_summary.csv                   # resumo do Ground Truth
  channel_stats_valid.json                              # estatísticas por canal (provisório)
results/inspection/                                     # figuras da inspeção
```

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
