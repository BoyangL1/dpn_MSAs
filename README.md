# Distinct regimes of behavior-based dependency networks shape urban resilience

Analysis code and selected results for the accompanying manuscript. The study examines dependency networks across 376 U.S. metropolitan areas, using cross-city graph neural network transferability to identify recurring network regimes, relate them to metropolitan characteristics, and assess simulated shock amplification.

| Module | Analysis |
| --- | --- |
| [01_network_statistics](01_network_statistics/) | Network distributions and distance decay |
| [02_transferability](02_transferability/) | GNN training, cross-city transfer and regime identification |
| [03_regression](03_regression/) | Transferability regression and explanatory contributions |
| [04_disaster_shocks](04_disaster_shocks/) | Functional and natural hazard shock propagation |
| [05_economic_impacts](05_economic_impacts/) | Building-value-weighted impact estimates |

Use Python **3.12.7** and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
export DPN_DATA_DIR=/path/to/prepared_inputs
```