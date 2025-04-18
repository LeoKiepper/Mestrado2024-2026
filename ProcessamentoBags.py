#%% Reset e imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import gc
from sklearn.model_selection import TimeSeriesSplit
import re
from telemetry import ResolveTel
from VideoTelemetria import Gera_VideoTelemetria


telemetria = ResolveTel('MAC_aguas_claras_2025-01-21-12-07-24_0.bag')

#%% Gera um vídeo de 10 segundos para propósito de desenvolvimento da função e ajuste de parâmetros
Gera_VideoTelemetria( "teste.mp4",
	("Project1.mp4", dict(
		# timestamp_video=1737471936.2761478
    )),
	( telemetria['T_CPU'], dict(
		plotkw=dict(labels=['Temperatura da CPU'])
	) ),
	( telemetria[['CPU_0','CPU_1','CPU_2','CPU_3','CPU_4','CPU_5','CPU_6','CPU_7']], dict(
		plotkw=dict(
			coltolabel_parser=lambda col: fr'$CPU_{{ {int(re.search(r"CPU_(\d+)", col).group(1)) + 1} }}$',
			linewidth=0.5, alpha=0.7
		),
		legendkw=dict(ncol=3, loc="upper left")
	) ),
	( telemetria[['i_M1','i_M2','i_M3','i_M4']], dict(
		plotkw=dict(
			coltolabel_parser=lambda col: fr'$i_{{m {re.search(r"i_M(\d+)", col).group(1)} }}$',
			linewidth=1
		),
		legendkw=dict(ncol=2, loc="upper left")
	) ),
	( telemetria[['cmd_vel','robot_vel']], dict(
		plotkw=dict(
			labels=['Vel. comandada','Vel. medida'],
			linewidth=1, alpha=0.7
		)
	) ),
	atraso=-108.361+5, 
	timestamp_tel=telemetria.timestamp_zero, casasdec_tempo=3, printyvals='legend',
	xlabel="Prévia [s]", facecolor="#fafafa", margem_ext=20, hspace=0.1,
	# temp_dir="D:\\", deleteaux=False, 
)

#%% Gera o vídeo da telemetria completa
# gerar_telemetria_video( "VideoTelemetria.mp4",
# 	( "itabira_galeria_oleo_frontal_jan-2025.mp4", dict(
# 		timestamp_video=1737471936.2761478
#     ) ),
# 	( telemetria['T_CPU'], dict(
# 		plotkw=dict(labels=['Temperatura da CPU'])
# 	) ),
# 	( telemetria[['CPU_0','CPU_1','CPU_2','CPU_3','CPU_4','CPU_5','CPU_6','CPU_7']], dict(
# 		plotkw=dict(
# 			coltolabel_parser=lambda col: fr'$CPU_{{ {int(re.search(r"CPU_(\d+)", col).group(1)) + 1} }}$',
# 			linewidth=0.5, alpha=0.7
# 		),
# 		legendkw=dict(ncol=3, loc="upper left")
# 	) ),
# 	( telemetria[['i_M1','i_M2','i_M3','i_M4']], dict(
# 		plotkw=dict(
# 			coltolabel_parser=lambda col: fr'$i_{{m {re.search(r"i_M(\d+)", col).group(1)} }}$',
# 			linewidth=1
# 		),
# 		legendkw=dict(ncol=2, loc="upper left")
# 	) ),
# 	( telemetria[['cmd_vel','robot_vel']], dict(
# 		plotkw=dict(
# 			labels=['Vel. comandada','Vel. medida'],
# 			linewidth=1, alpha=0.7
# 		)
# 	) ),	atraso=-108.361-96.192,
# 	timestamp_tel=b.start_time, casasdec_tempo=3, printyvals='legend',
# 	xlabel="Prévia [s]", facecolor="#fafafa", margem_ext=20, hspace=0.1,
# 	# temp_dir="D:\\", deleteaux=False,
# )
