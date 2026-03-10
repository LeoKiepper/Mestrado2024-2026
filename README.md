> [!NOTE]
> Desenvolvido para Python 3.13.11
> 
> Para instalar bibliotecas utilizadas:
> 
> ```python 
> pip install -r requirements.txt
> ```

# Scripts

O script `DDSEspeleo.py` corresponde ao modelo híbrido e arquitetura propostas na dissertação: 

**MODELO HÍBRIDO EM DOIS ESTÁGIOS PARA PREDIÇÃO DA TEMPERATURA DA CPU EM UM ROBÔ DE SERVIÇO**
<p><br></p>

As definições de classes para o script `DDSEspeleo.py` encontram-se em `ddslib.py`.
Os scripts `Comparison_RNN.py`, `Comparison_LSTM.py` e `Comparison_Autoencoder.py` correspondem às técnicas contra as quais o modelo proposto é comparado. O script `plan_lag_feature_spacing.py` calcula a quantidade de passos de atraso, dado um *budget* de memória total para o DataFrame com *lag features* inclusos, e considerado o gasto de memória antes da adição desses *features*.

O script `montecarlo.py` executa a simulação Monte Carlo dos scripts dos algoritmos estudados, salvando os dados de saída que cada script salva separadamente usando funções fornecidas por `montecarlo_utils.py`.
`plot_mc.py` procura dados de simulação MC gerados por `montecarlo.py`, e plota a dispersão das séries temporais de previstas resultante de cada simulação, usando `plotstyle_configs/monte_carlo.yaml` para configurar plots.
O script `plot_radar.py` utiliza dados auxiliares das dispersões de séries temporais previstas calculados pelo script `plot_mc.py` para plotar gráficos radar dos KPIs de cada simulação, usando `plotstyle_configs/radar.yaml` como arquivo de configuração dos plots.

Os scripts `plotstyle.py`, `plotstyle_interface.py` e `plotstyle_validators.py` fornecem ferramentas para a geração de um objeto usável em runtime que encapsula opções e parâmetros para funções da bilbioteca matplotlib lidas de um arquivo .yaml. 

O arquivo .bag usado encontra-se na pasta `cache`.
O script `tellib.py` fornece uma camada de conveniência para a tradução do arquivo .bag para um objeto `DataFrame` acessível em runtime. O usuário deve fornecer uma bag e uma função de processamento da bag, que deve retornar um DataFrame. O objetivo principal desse módulo é a racionalização do tempo de tradução de uma bag para DataFrame, que não é sempre garantido ser um cálculo rápido, dependendo do tamanho da bag. Isso é feito salvando o DataFrame gerado em um arquivo .pkl com hash gerado a partir do conteúdo da bag e código fonte da função de tradução, que em chamadas posteriores é carregado desde haja match do hash entre o par bag/função numa nova execução com algum arquivo .pkl dentro da pasta `cache`. 

`VideoTelemetria.py` é uma ferramenta desenvolvida para produzir um vídeo composto de feed de camera e playback de telemetria, capturados juntos durante a mesma missão e sincronizados com base no relógio do computador embarcado.
`ResolveTel.py` é um script que utiliza tellib.py e VideoTelemetria.py para produzir o vídeo da missão específica na qual foi capturada a bag objeto de estudo da dissertação.



# Geração das Figuras:

Para gerar as Figuras 3.7 -> Rodar `DDSEspeleo.py` com os flag PLOT_DATASET (linha 73) descomentada;

Simulação Monte Carlo:
- No terminal, rodar o comando
  ```python
  python montecarlo.py --runs N <script.py> --noplot
  ```
  Para executar a simulação do script python especificado N vezes. O script correspondente ao modelo proposto é `DDSEspeleo.py` e os outros scripts de comparação são `Comparison_RNN.py` e `Comparison_LSTM.py`. Cada um desses scripts também pode ser rodado sem o intermédio do `montecarlo.py`, o que é útil para verificação manual de hiperparâmetros, por exemplo.
  Serão gerados os outputs e KPIs dos scripts, organizados em arquivos dentro de uma pasta measure_outputs.

- Para gerar as Figuras 4.3, 4.4 e 4.5, rodar o comando 
  ```python
  python plot_mc.py <nome do script sem extensão> monte_carlo.yaml
  ```
  informando os nomes `DDSEspeleo`, `Comparison_RNN` ou `Comparison_LSTM`, um a cada chamada do comando.

- Depois de ter gerado as Figuras 4.3, 4.4 e 4.5, os gráficos radar da Tabela 4.1 podem ser gerados com o comando
  ```python
  python plot_radar DDSEspeleo,Comparison_RNN,Comparison_LSTM --plotstyle radar.yaml
  ```

Todas as Figuras são salvas na pasta `Figures` dentro do workspace.
