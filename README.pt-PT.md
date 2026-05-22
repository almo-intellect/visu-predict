# visu-predict

[English](README.md) · **Português (Portugal)**

Conjunto de ferramentas para previsão de tráfego baseado em *Transformers*.
Faz previsões de curto horizonte de velocidade ou fluxo de tráfego em redes
de sensores (METR-LA, PEMS-BAY, PEMS-03/04/07/08 ou conjuntos de dados
personalizados) através de um *Transformer encoder–decoder* com
pré-processamento opcional por redes neuronais em grafo, atenção orientada a
características sobre dados temporais, meteorológicos, espaciais e desfasados,
e um caminho de aprendizagem por transferência para conjuntos de dados de
baixa cobertura.

## Funcionalidades

- **Transformer encoder–decoder** com descodificador linear, MLP ou
  *transformer* autoregressivo.
- **Atenção orientada a características**: incorporações e *transformer*
  separados para cada grupo de características (tráfego, hora do dia, dias
  feriados, meteorologia, valores desfasados, características espaciais),
  combinados com porta adaptativa ao contexto e atenção cruzada.
- **Pré-codificador GNN opcional** (GCN ou GAT, via `torch-geometric`).
- **Viés espacial** na auto-atenção do *encoder* a partir de coordenadas /
  matriz de adjacência dos sensores.
- **Integração meteorológica** com alinhamento temporal por vizinho mais
  próximo aos carimbos temporais do tráfego.
- **Aprendizagem por transferência**: carrega um *checkpoint* pré-treinado,
  congela as primeiras N camadas do *encoder* e faz ajuste fino com uma taxa
  de aprendizagem menor.
- Treino com **precisão mista** e acumulação de gradientes.

## Instalação

```bash
pip install -e .
# Extras opcionais
pip install -e ".[gnn,holidays,dev]"
```

Requer Python ≥ 3.10 e uma versão recente do PyTorch (2.x).

## Início rápido

1. Coloque um ficheiro CSV de tráfego (índice temporal, uma coluna por sensor)
   numa localização acessível.
2. Copie `configs/example.yaml` e ajuste, no mínimo:
   - `base_output_dir` — onde os *checkpoints* e gráficos serão guardados.
   - `dataset_name` — usado para localizar os ficheiros de adjacência e
     coordenadas em `<base_output_dir>/inputs/`.
3. Inicie o treino:

```bash
visu-predict train --config configs/example.yaml --data caminho/para/tráfego.csv
```

Ou de forma programática:

```python
from visu_predict import load_config
from visu_predict.runner import run_training

config = load_config("configs/example.yaml")
resultado = run_training(config, data_path="caminho/para/tráfego.csv")
# resultado: dicionário com 'model', 'train_losses', 'val_losses', 'predictions', 'actuals'
```

## Configuração

Todas as opções vivem em `TrainingConfig` (`src/visu_predict/config.py`).
Consulte `configs/example.yaml` para um ponto de partida com comentários.
Destaques:

| Chave | Predefinição | Notas |
|-------|--------------|-------|
| `dataset_name` | `METR-LA` | Usado por `paths.find_adjacency_matrix` / `find_coordinates` |
| `seq_length` / `pred_length` | `12` / `12` | Janela de entrada e horizonte de previsão (em passos) |
| `hidden_dim` / `num_heads` | `336` / `16` | Ajustado automaticamente para ser par e divisível |
| `decoder_type` | `linear` | `linear`, `mlp` ou `transformer` (autoregressivo) |
| `use_gnn_pre_transformer` | `false` | Requer `torch-geometric` |
| `use_weather_feature` | `false` | Necessita de `weather_data_file` |
| `enable_transfer_learning` | `false` | Necessita de `source_model_path` |

## Estrutura do repositório

```
src/visu_predict/
├── config.py              # Dataclass TrainingConfig + carregamento de YAML
├── paths.py               # Descoberta de ficheiros de adjacência / coordenadas
├── data.py                # TrafficDataset, prepare_data
├── features/
│   ├── weather.py         # WeatherIntegration
│   └── spatial.py         # SpatialIntegration, utilitários de adjacência
├── models/
│   ├── transformer.py     # TrafficTransformer
│   ├── attention.py       # FeatureAttention
│   ├── positional.py      # PositionalEncoding
│   ├── gnn.py             # GCNEncoder (opcional)
│   └── lr.py              # CosineWarmupLR
├── training/
│   ├── train.py           # train / evaluate / predict
│   ├── losses.py          # quantil, híbrida, MAPE robusto
│   └── transfer.py        # carregamento de checkpoints + congelamento
├── utils/                 # registo de eventos (logging), GPU, seeding
├── viz.py                 # gráficos de treino e de previsões
├── runner.py              # Pipeline ponta a ponta
└── cli.py                 # `visu-predict train ...`
```

## Dados

O repositório **não inclui** os ficheiros de dados de tráfego, adjacência,
coordenadas ou meteorologia. Consulte [DATA.md](DATA.md) para os links e o
*script* auxiliar de descarga.

## Desenvolvimento

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Licença

MIT, © Almo Intellect. Ver [LICENSE](LICENSE).

## Autores

- Lauro Mota — `lauro.mota@almo.co.mz` (autor principal)
