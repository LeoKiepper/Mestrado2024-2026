Desenvolvido para Python 3.13.11

Requisitos: `pip install -r requirements.txt`



OBS: O arquivo .bag usado encontra-se na pasta 'cache'



Para gerar as Figuras 3.7 -> Rodar DDSEspeleo.py com os flag PLOT_DATASET (linha 73) descomentada

Simulação Monte Carlo:
No terminal, rodar o comando
```python
python montecarlo.py --runs N <script.py> --noplot
```
Para executar a simulação do script python especificado N vezes. O script correspondente ao modelo proposto é DDSEspeleo.py, e os outros scripts de comparação são Comparison_RNN.py e Comparison_LSTM.py.
Serão gerados os outputs e KPIs dos scripts, organizados em arquivos dentro de uma pasta measure_outputs.

Para gerar as Figuras 4.3, 4.4 e 4.5, rodar o comando 
```python
python plot_mc.py <nome do script sem extensão> monte_carlo.yaml
```
informando os nomes DDSEspeleo, Comparison_RNN ou Comparison_LSTM, um de cada vez

Depois de ter gerado as Figuras 4.3, 4.4 e 4.5, os gráficos radar da Tabela 4.1 podem ser gerados com o comando
```python
python plot_radar DDSEspeleo,Comparison_RNN,Comparison_LSTM --plotstyle radar.yaml
```

Todas as Figuras são salvas na pasta Figures dentro do workspace.
