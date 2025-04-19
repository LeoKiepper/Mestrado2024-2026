from bagpy import bagreader
import tellib
import re, pandas as pd, sys
from VideoTelemetria import Gera_VideoTelemetria

#%% Resolve a variável "telemetria"
def builder(self, bagfile):
	# Cria a variável "telemetria" a partir da bag informada
	b=bagreader(bagfile)

	# ===================== Desembaraça as mensagens salvas na bag =====================
	# Temperatura da CPU
	aux1 = b.message_by_topic('/cpu_temp')
	aux1 = pd.read_csv(aux1)
	aux1 = aux1.rename(columns={'data': 'T_CPU'})
	aux1.Time = aux1.Time - b.start_time

	# # Corrente instantânea da bateria 1
	# aux2 = b.message_by_topic('/espeleo_io/battery_1_signed_status')
	# aux2 = pd.read_csv(aux2)
	# aux2 = aux2.drop(['layout.dim','layout.data_offset','data_0','data_2'],axis=1)
	# aux2 = aux2.rename(columns={'data_1': 'i_B1'})
	# aux2.i_B1=-aux2.i_B1
	# aux2.Time = aux2.Time - b.start_time

	# # Corrente média da bateria 1
	# aux3 = b.message_by_topic('/espeleo_io/battery_1_signed_status')
	# aux3 = pd.read_csv(aux3)
	# aux3 = aux3.drop(['layout.dim','layout.data_offset','data_0','data_1'],axis=1)
	# aux3 = aux3.rename(columns={'data_2': 'i_B1_avg'})
	# aux3.i_B1_avg=-aux3.i_B1_avg
	# aux3.Time = aux3.Time - b.start_time

	# # Corrente instantânea da bateria 2
	# aux4 = b.message_by_topic('/espeleo_io/battery_2_signed_status')
	# aux4 = pd.read_csv(aux4)
	# aux4 = aux4.drop(['layout.dim','layout.data_offset','data_0','data_2'],axis=1)
	# aux4 = aux4.rename(columns={'data_1': 'i_B2'})
	# aux4.i_B2=-aux4.i_B2
	# aux4.Time = aux4.Time - b.start_time

	# # Corrente média da bateria 2
	# aux5 = b.message_by_topic('/espeleo_io/battery_2_signed_status')
	# aux5 = pd.read_csv(aux5)
	# aux5 = aux5.drop(['layout.dim','layout.data_offset','data_0','data_1'],axis=1)
	# aux5 = aux5.rename(columns={'data_2': 'i_B2_avg'})
	# aux5.i_B2_avg=-aux5.i_B2_avg
	# aux5.Time = aux5.Time - b.start_time

	# Corrente no motor 1
	aux6 = b.message_by_topic('/device1/get_current_actual_value')
	aux6 = pd.read_csv(aux6)
	aux6 = aux6.rename(columns={'data': 'i_M1'})
	aux6.Time = aux6.Time - b.start_time

	# Corrente no motor 2
	aux7 = b.message_by_topic('/device3/get_current_actual_value')
	aux7 = pd.read_csv(aux7)
	aux7 = aux7.rename(columns={'data': 'i_M2'})
	aux7.Time = aux7.Time - b.start_time

	# Corrente no motor 3
	aux8 = b.message_by_topic('/device4/get_current_actual_value')
	aux8 = pd.read_csv(aux8)
	aux8 = aux8.rename(columns={'data': 'i_M3'})
	aux8.Time = aux8.Time - b.start_time

	# Corrente no motor 4
	aux9 = b.message_by_topic('/device6/get_current_actual_value')
	aux9 = pd.read_csv(aux9)
	aux9 = aux9.rename(columns={'data': 'i_M4'})
	aux9.Time = aux9.Time - b.start_time

	# # Intensidade dos LEDs frontais
	# aux10 = b.message_by_topic('/espeleo_io/frontLight')
	# aux10 = pd.read_csv(aux10)
	# aux10 = aux10.drop(['chooseLight'],axis=1)
	# aux10 = aux10.rename(columns={'intensityLight': 'LED_F'})
	# aux10.Time = aux10.Time - b.start_time

	# # Intensidade dos LEDs traseiros
	# aux11 = b.message_by_topic('/espeleo_io/backLight')
	# aux11 = pd.read_csv(aux11)
	# aux11 = aux11.drop(['chooseLight'],axis=1)
	# aux11 = aux11.rename(columns={'intensityLight': 'LED_B'})
	# aux11.Time = aux11.Time - b.start_time

	# Porcentagem de utilizacao da CPU
	aux12=b.message_by_topic('/cpu_percent')
	aux12=pd.read_csv(aux12)
	aux12.rename(columns=lambda col: re.sub(r'^data_(\d+)$', r'CPU_\1', col), inplace=True)
	aux12.drop(['layout.dim','layout.data_offset'],axis=1,inplace=True)
	aux12.Time = aux12.Time - b.start_time

	# Resolve o número de CPUs
	NumCPUs=len(aux12.columns)-1    # Se for rodado depois de retirar colunas de layout


	aux13 = b.message_by_topic('/cmd_vel')
	aux13 = pd.read_csv(aux13)
	aux13.drop(['linear.y','linear.z','angular.x','angular.y','angular.z'],axis=1,inplace=True)
	aux13 = aux13.rename(columns={'linear.x': 'cmd_vel'})
	aux13.Time = aux13.Time - b.start_time

	aux14 = b.message_by_topic('/robot_vel')
	aux14 = pd.read_csv(aux14)
	aux14 = aux14.rename(columns={'linear.x': 'robot_vel'})
	aux14.drop(['linear.y','linear.z','angular.x','angular.y','angular.z'],axis=1,inplace=True)
	aux14.Time = aux14.Time - b.start_time

	# Monta um DataFrame consolidando todas as variáveis
	# telemetria=pd.merge_ordered(aux1,aux2,on='Time',how='outer')
	# telemetria=pd.merge_ordered(telemetria,aux3,on='Time',how='outer')
	# telemetria=pd.merge_ordered(telemetria,aux4,on='Time',how='outer')
	# telemetria=pd.merge_ordered(telemetria,aux5,on='Time',how='outer')
	telemetria=pd.merge_ordered(aux1,aux6,on='Time',how='outer')
	telemetria=pd.merge_ordered(telemetria,aux7,on='Time',how='outer')
	telemetria=pd.merge_ordered(telemetria,aux8,on='Time',how='outer')
	telemetria=pd.merge_ordered(telemetria,aux9,on='Time',how='outer')
	# telemetria=pd.merge_ordered(telemetria,aux10,on='Time',how='outer')
	# telemetria=pd.merge_ordered(telemetria,aux11,on='Time',how='outer')
	telemetria=pd.merge_ordered(telemetria,aux12,on='Time',how='outer')
	telemetria=pd.merge_ordered(telemetria,aux13,on='Time',how='outer')
	telemetria=pd.merge_ordered(telemetria,aux14,on='Time',how='outer')
	telemetria=telemetria.set_index('Time')


	# ========================================== Limpa o DataFrame ==========================================
	# Preenche os valores NaN com interpolação linear ou ZOH conforme o que faz mais sentido para cada variável.
	telemetria.T_CPU=telemetria.T_CPU.ffill()
	# telemetria.i_B1=telemetria.i_B1.interpolate(method='linear')
	# telemetria.i_B1_avg=telemetria.i_B1_avg.interpolate(method='linear')
	# telemetria.i_B2=telemetria.i_B2.interpolate(method='linear')
	# telemetria=telemetria.rename(columns={'i_B2_x': 'i_B2'})
	# telemetria.i_B2_avg=telemetria.i_B2_avg.interpolate(method='linear')
	telemetria.i_M1=telemetria.i_M1.interpolate(method='linear')
	telemetria=telemetria.rename(columns={'i_M1_x': 'i_M1'})
	telemetria.i_M2=telemetria.i_M2.interpolate(method='linear')
	telemetria=telemetria.rename(columns={'i_M2_x': 'i_M2'})
	telemetria.i_M3=telemetria.i_M3.interpolate(method='linear')
	telemetria=telemetria.rename(columns={'i_M3_x': 'i_M3'})
	telemetria.i_M4=telemetria.i_M4.interpolate(method='linear')
	telemetria=telemetria.rename(columns={'i_M4_x': 'i_M4'})
	# telemetria.LED_F=telemetria.LED_F.fillna(method='ffill').fillna(0)
	# telemetria.LED_F=telemetria.LED_F
	# telemetria.LED_B=telemetria.LED_B.fillna(method='ffill').fillna(0)
	# telemetria.LED_B=telemetria.LED_B.fillna(0)
	for CPU in range(NumCPUs):     telemetria[f'CPU_{CPU}']=telemetria[f'CPU_{CPU}'].interpolate(method='linear')
	telemetria.cmd_vel=telemetria.cmd_vel.interpolate(method='linear')
	telemetria.robot_vel=telemetria.robot_vel.interpolate(method='linear')
	# telemetria=telemetria.dropna()

	# Salva em arquivo do excel
	telemetria.to_excel('telemetria.xlsx')
	# Limpa variáveis desnecessárias
	# del aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12

	return tellib.telclass(
		tel=telemetria,
		timestamp_zero=b.start_time
	)
telobj = type("TP_type", (tellib.telprocessor,), {"builder": builder})('MAC_aguas_claras_2025-01-21-12-07-24_0.bag')
tel = telobj.get()
telemetria=tel.tel; timestamp_tel=tel.timestamp_zero

sys.exit()

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
	timestamp_tel=timestamp_tel, casasdec_tempo=3, printyvals='legend',
	xlabel="Prévia [s]", facecolor="#fafafa", margem_ext=20, hspace=0.1,
	# temp_dir="D:\\", deleteaux=False, 
)
