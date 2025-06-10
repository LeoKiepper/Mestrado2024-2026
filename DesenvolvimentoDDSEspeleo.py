from bagpy import bagreader
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import timedelta

#%% Resolve telemetry variable
import tellib
def resolvetel():
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
		telemetria.loc[:, 'T_CPU'] = telemetria.loc[:,'T_CPU'].ffill()
		# telemetria.i_B1=telemetria.i_B1.interpolate(method='linear')
		# telemetria.i_B1_avg=telemetria.i_B1_avg.interpolate(method='linear')
		# telemetria.i_B2=telemetria.i_B2.interpolate(method='linear')
		# telemetria=telemetria.rename(columns={'i_B2_x': 'i_B2'})
		# telemetria.i_B2_avg=telemetria.i_B2_avg.interpolate(method='linear')
		for m in ['i_M1','i_M2','i_M3','i_M4']: telemetria.loc[:,m] = telemetria.loc[:,m].interpolate(method='linear')
		# telemetria.i_M1=telemetria.i_M1.interpolate(method='linear')
		# telemetria=telemetria.rename(columns={'i_M1_x': 'i_M1'})
		# telemetria.i_M2=telemetria.i_M2.interpolate(method='linear')
		# telemetria=telemetria.rename(columns={'i_M2_x': 'i_M2'})
		# telemetria.i_M3=telemetria.i_M3.interpolate(method='linear')
		# telemetria=telemetria.rename(columns={'i_M3_x': 'i_M3'})
		# telemetria.i_M4=telemetria.i_M4.interpolate(method='linear')
		# telemetria=telemetria.rename(columns={'i_M4_x': 'i_M4'})
		# telemetria.LED_F=telemetria.LED_F.fillna(method='ffill').fillna(0)
		# telemetria.LED_F=telemetria.LED_F
		# telemetria.LED_B=telemetria.LED_B.fillna(method='ffill').fillna(0)
		# telemetria.LED_B=telemetria.LED_B.fillna(0)
		for CPU in range(NumCPUs):     telemetria.loc[:,f'CPU_{CPU}']=telemetria.loc[:,f'CPU_{CPU}'].interpolate(method='linear')
		telemetria.loc[:,'cmd_vel']=telemetria.loc[:,'cmd_vel'].interpolate(method='linear')
		telemetria.loc[:,'robot_vel']=telemetria.loc[:,'robot_vel'].interpolate(method='linear')
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
	return tel.tel
tel = resolvetel()

#%% Calculate and extract features from tel
def calcfeatures(tel: pd.DataFrame) -> pd.DataFrame:
	target=['T_CPU']
	features=[
				#'T_CPU',
				#'i_B1','i_B1_avg','i_B2','i_B2_avg',
				'i_M1','i_M2','i_M3','i_M4',
				# 'T_CPU_trend',# 'T_CPU_detrend',
				# 'T_CPU_trend_lag1','T_CPU_trend_lag2','T_CPU_trend_lag3','T_CPU_trend_lag4',
				# 'CPU_0','CPU_1','CPU_2','CPU_3','CPU_4','CPU_5','CPU_6','CPU_7',
				'CPU',
				# 'LED_F', 'LED_B',
				# 'Dissip_B1','Dissip_B2',
				# 'Dissip_M1','Dissip_M2','Dissip_M3','Dissip_M4',
				# 'Dissip_M1_lag1','Dissip_M1_lag2','Dissip_M1_lag3','Dissip_M1_lag4',
				# 'Dissip_M2_lag1','Dissip_M2_lag2','Dissip_M2_lag3','Dissip_M2_lag4',
				# 'Dissip_M3_lag1','Dissip_M3_lag2','Dissip_M3_lag3','Dissip_M3_lag4',
				# 'Dissip_M4_lag1','Dissip_M4_lag2','Dissip_M4_lag3','Dissip_M4_lag4',
			]
	df=tel.loc[:,target]
	for f in features:
		if f in tel.columns: 
			df=df.merge(tel.loc[:,f],left_index=True,right_index=True)

	regex = re.compile(r'^CPU_\d+$')
	cols = [col for col in tel.columns if regex.match(col)]
	if ('CPU' in features) and cols: # cols evaluates to false if it is empty
		df.loc[:,'CPU'] = tel.loc[:,cols].max(axis=1)
	
	return df.dropna()
df=calcfeatures(tel)
df=df[df.index<2790]			# cull dataset before temperature peak
df['CPU'] = df['CPU']/100		# in place of systematic normalization, just scale CPU usage to 0-1 range
def calc_temp_amp(df):
	return df.loc[df.index<=10,'T_CPU'].mean()
df2=df[['T_CPU','CPU']].copy(deep=True)

#%%  Define and instantiate DDS model components
from ddslib import M2, M2Kernel, M2Optimizer, M3, get_plotstlye
m2obj=M2(
	M2Kernel(lambda cpu: cpu**2, lambda temp_current, temp_ext: temp_current-temp_ext, TempAmb=calc_temp_amp(df2), Dt=(df2.index[-1]-df2.index[0])/len(df2),
		param_space=M2Kernel.ParamSpace(KCPU=(1e-5,10), KTemp=(1e-5,1), TauCPU=(1e-9,2), TauTemp=(1e-9,2)),
		# params=M2Kernel.Params(KCPU=0.8955001304094646, KTemp=0.0008084840447585498, TauCPU=0.7114574813747837, TauTemp=0.4034388338443532),		# RMSE = 1.249
		# params=M2Kernel.Params(KCPU=0.7994835811720532, KTemp=0.0012296959998389521, TauCPU=0.814607170204983, TauTemp=0.9513807657573216),		# RMSE = 1.398
		params=M2Kernel.Params(KCPU=0.9661344533627875, KTemp=0.0016280793961810907, TauCPU=0.8349590176487286, TauTemp=0.9907842974965935),		# RMSE = 1.171
	),
	M2Optimizer(training_duration=timedelta(seconds=10), composition='any', 
		training_stop_flags = M2Optimizer.StopConditions.GLOBAL_MIN_LOSS 
							| M2Optimizer.StopConditions.GLOBAL_MAX_DURATION
				)
)
m2target_col = 'T_CPU'
m2obj.fit(df2['CPU'].to_frame(), df2[m2target_col], plot=(plot:=True))
m2pred=m2obj.predict(df2['CPU'].to_frame())
m2pred=pd.Series(m2pred, index=df2.index)
if plot:
	fig, ax = plt.subplots(1,figsize=(10,6))
	ax.plot(df2['T_CPU'], label='Reference', lw=1)
	ax.plot(m2pred, label='Prediction', lw=2)
	ax.grid(which='major', color='#e0e0e0', linewidth=1.5)
	ax.grid(which='minor', color='#f0f0f0', linewidth=1)
	ax.set_ylabel('CPU temperature '+r'$[°C]$', fontsize=14)
	ax.set_xlabel('Time [s]', fontsize=14)
	ax.set_facecolor('#fafafa')
	ax.legend()
	plt.show()

m3_source_col = m2target_col
df3=df.loc[:,df.columns != m3_source_col].copy(deep=True)
target_col = 'Temp_residue'
df3.loc[:,target_col] = (df2.loc[:,m3_source_col].copy(deep=True)-m2pred).rename(target_col)

m3obj=M3(n_estimators=100, plotstyle=get_plotstlye('ICAR2025'))
test_idx = m3obj.cross_validation(
	df3.loc[:,df3.columns != target_col],	# X
	df3.loc[:,target_col],					# y
	n_splits=3, plot=True)					# options

