from bagpy import bagreader
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import timedelta
import ruptures as rpt
from math import floor
from ddslib import *
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from measure_utils import report_output, NOPLOT

# Plot flags
PLOT_DATASET = 1<<0
PLOT_CLIPPED_DATASET = 1<<1
PLOT_M2_PARTIAL_PREDICTION = 1<<2
PLOT_M3_PARTIAL_PREDICTION = 1<<3
PLOT_M2_TRAINING_HISTORY = 1<<4
PLOT_M3_TRAINING_HISTORY = 1<<5 # Not implemented
PLOT_COMPOSITE_PREDICTION = 1<<6

PLOT_FLAGS = 0 if NOPLOT else (	# Comment and uncomment to disable/enable plots
# PLOT_DATASET |
# PLOT_CLIPPED_DATASET | 
# PLOT_M2_TRAINING_HISTORY | 
# PLOT_M2_PARTIAL_PREDICTION | 
# PLOT_M3_TRAINING_HISTORY | 
# PLOT_M3_PARTIAL_PREDICTION | 
PLOT_COMPOSITE_PREDICTION | 
0 )

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
		df.loc[:,'CPU'] = tel.loc[:,cols].max(axis=1)/100
		df = df.join(tel[cols]/100)

	return df.dropna()
df=calcfeatures(tel)

#%% Define the plotter for the dataset
class DatasetPlotter:
	def __init__(self, plotstyle: PlotStyle):
		self.plotstyle = plotstyle
	def plot(self,df,segments=[],savefig_options: dict = {},set_title_options: dict = {}, savefig: bool = True, save_with_title: bool = True, plot_only_cpu_feature: bool = False, plot_only_raw_cpu: bool = False, show_segment: dict = {}, margin = 1):
		def sanitize_plotstyle(ps: PlotStyle):
			if ps is None: ps = get_plotstyle('')
			if not (isinstance(ps.M2_training_score_history_title, str)): 
				ps.M2_training_score_history_title = ''
			if not (isinstance(ps.M2_training_score_history_xlabel, str)): 
				ps.M2_training_score_history_xlabel = ''
			if not (isinstance(ps.score_label, str)): 
				ps.score_label = ''
			if not (isinstance(ps.M2_training_score_history_filename, str)): 
				ps.M2_training_score_history_filename = 'M2_training_score_history'
			if not (isinstance(ps.save_figure_extension, str)): 
				ps.save_figure_extension = 'svg'
			if not (isinstance(ps.savefig_bbox_inches, str)): 
				ps.savefig_bbox_inches = 'tight'
			if not (is_valid_font(ps.label_fontfamily)): 
				ps.label_fontfamily = None
			if not (isinstance(ps.label_fontsize, (int,float)) and ps.label_fontsize > 0): 
				ps.label_fontsize = None
			if not (isinstance(ps.title_fontsize, (int,float)) and ps.title_fontsize > 0): 
				ps.title_fontsize = None
			if not (isinstance(ps.tick_label_fontsize, (int,float)) and ps.tick_label_fontsize > 0): 
				ps.tick_label_fontsize = None
			if not (isinstance(ps.spine_linewidth, (int,float)) and ps.spine_linewidth > 0): 
				ps.spine_linewidth = None
			if not mcolors.is_color_like(ps.facecolor):
				ps.facecolor = None
			if not (isinstance(ps.grid_options, list)): 
				ps.grid_options = []
			if not isinstance(ps.single_figsize,tuple) or len(ps.single_figsize) != 2 or any([not isinstance(dim,(float,int)) or dim <=0 for dim in ps.single_figsize]):
				ps.single_figsize = None
			return ps
		PS = sanitize_plotstyle(self.plotstyle)
		fig, ax = plt.subplots(nrows=(N:=2), figsize=PS.multiple_figsize(N))
		# ============ Temperature
		aa=0
		ax[aa].plot('T_CPU', data=df, linewidth=PS.linewidth_thin)
		ax[aa].set_ylabel(PS.ylabel_temperature, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
		ax[aa].set_facecolor(PS.facecolor)
		for grid_option in PS.grid_options: ax[aa].grid(**grid_option)
		for spine in ax[aa].spines.values(): spine.set_linewidth(PS.spine_linewidth)
		
		# ============ CPU percent
		aa=1
		if not plot_only_raw_cpu:
			ax[aa].plot('CPU', data=df, label = 'CPU feature', linewidth=PS.linewidth_thin if plot_only_cpu_feature else PS.linewidth_thick+0.5, color='C8')
		if not plot_only_cpu_feature:
			num_cpus = len([col for col in df.columns if re.match(r'^CPU_\d+$', col)])
			for cpu in range(num_cpus):
				ax[aa].plot(f'CPU_{cpu}', data=df, label = fr'$CPU_{{{cpu+1}}}$', linewidth=PS.linewidth_thin)
			if plot_only_raw_cpu: ax[aa].legend(fontsize=PS.legend_fontsize, ncol=3)
		ax[aa].set_ylabel(PS.ylabel_cpu_load, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
		ax[aa].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y*100)}"))
		ax[aa].set_facecolor(PS.facecolor)
		for grid_option in PS.grid_options: ax[aa].grid(**grid_option)
		for spine in ax[aa].spines.values(): spine.set_linewidth(PS.spine_linewidth)

		# Set xlabel for the last plot
		ax[-1].set_xlabel(PS.xlabel_time, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)

		for seg in segments:
			if seg['state']=='high':
				start = df.index[seg['pos_low']]
				end = df.index[seg['pos_up']]
				for axis in ax: axis.axvspan(start,end,color = "#ff7676ff", alpha=0.4)

		if show_segment:
			start = max(df.index[show_segment['pos_low']]-margin,df.index[0])
			end = min(df.index[show_segment['pos_up']]+margin,df.index[-1])
			for axis in ax: axis.set_xlim(start,end)
			

		PlotStyle.settitle_and_savefig(fig, ax,
			savefig_options=savefig_options,
			set_title_options=set_title_options,
			savefig=savefig,
			save_with_title=save_with_title
		)
		plt.show(block=True)
PS=get_plotstyle('IEEE2025')
dsplotter=DatasetPlotter(PS)
#%% Segment time series
def HighCPUDetect(df, margin=0,threshold=0.8):
	"""
	Implements high cpu detection logic.
	
	:param cpu_percent: array containing CPU usage
	:param cp: array for change points detected in cpu_percent, listed as indexes for each position.
	Each position of cp is considered the last index of a segment (should not contain 0).
	:param margin: number of indexes to consider on each side of each segment investigated in the detection criterion implemented.
	:param threshold: consider high CPU usage if the average of the segment is greater than this value.

	:return segment: list of detected segments, formatted as dictionaries with the following keys:
	
	"""
	cpu_percent=df['CPU'].values
	model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
	algo = rpt.Window(model=model).fit(cpu_percent)
	ChangePoints = algo.predict(pen=5)
	Segments=[]
	for cc, pos_up in enumerate(ChangePoints):
		pos_up=min(pos_up+margin,len(cpu_percent))
		pos_low=max(ChangePoints[cc-1]+1-margin,0) if cc>0 else 0
		
		
		avg=float(np.average(cpu_percent[pos_low:pos_up]))
		if avg>threshold: 	Segments.append({'state':'high','pos_low':pos_low,'pos_up':pos_up,'avg':avg})
		else:				Segments.append({'state':'norm','pos_low':pos_low,'pos_up':pos_up,'avg':avg})
	return ChangePoints, Segments
bkpts, segments = HighCPUDetect(df, margin = 5,threshold=0.8)
#%% Dataset plots
if plotdataset:=bool(PLOT_FLAGS & PLOT_DATASET): dsplotter.plot(df,segments, plot_only_raw_cpu=True,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.full_dataset_filename, 
					format=PS.save_figure_extension, 
					bbox_inches=PS.savefig_bbox_inches
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.full_dataset_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.full_dataset_savefig,
				save_with_title=PS.save_with_title
)
if plotdataset: dsplotter.plot(df,segments=segments,show_segment=segments[1],margin=1,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.first_temp_peak_detail_filename, 
					format=PS.save_figure_extension, 
					bbox_inches=PS.savefig_bbox_inches
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.first_temp_peak_detail_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.first_temp_peak_detail_savefig,
				save_with_title=PS.save_with_title
)

idx = segments[1]['pos_low']
df=df[range(len(df))<idx]			# clip dataset before temperature peak
if plot_clipped_dataset := bool(PLOT_FLAGS & PLOT_CLIPPED_DATASET) : dsplotter.plot(df,plot_only_cpu_feature=True,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.clipped_dataset_filename, 
					format=PS.save_figure_extension, 
					bbox_inches=PS.savefig_bbox_inches
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.clipped_dataset_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.clipped_dataset_savefig,
				save_with_title=PS.save_with_title
)

#%% Define ambient temperature to be the temperature readout from the first few samples
def calc_temp_amp(df):
	return df.loc[df.index<=10,'T_CPU'].mean()
df2=df[['T_CPU','CPU']].copy(deep=True)



#%%  Define and instantiate DDS model components
from ddslib import M2, M2Kernel, M2Optimizer, get_plotstyle
m2obj=M2(
	M2Kernel(lambda cpu: cpu**2, lambda temp_current, temp_ext: temp_current-temp_ext, TempAmb=calc_temp_amp(df2), Dt=(df2.index[-1]-df2.index[0])/len(df2),
		param_space=M2Kernel.ParamSpace(KCPU=(8e-2,8), KTemp=(1e-3,0.01), TauCPU=(8e-2,8), TauTemp=(1e-1,10)),
		# params=M2Kernel.Params(KCPU=0.8955001304094646, KTemp=0.0008084840447585498, TauCPU=0.7114574813747837, TauTemp=0.4034388338443532),		# RMSE = 1.249
		# params=M2Kernel.Params(KCPU=0.7994835811720532, KTemp=0.0012296959998389521, TauCPU=0.814607170204983, TauTemp=0.9513807657573216),		# RMSE = 1.398
		# params=M2Kernel.Params(KCPU=0.9661344533627875, KTemp=0.0016280793961810907, TauCPU=0.8349590176487286, TauTemp=0.9907842974965935),		# RMSE = 1.171
		params=M2Kernel.Params(KCPU=1.9967875200537128, KTemp=0.001738236948110962, TauCPU=1.7285830177591441, TauTemp=1.0368206944316354), 		# RMSE = 1.104
	),
	M2Optimizer(training_duration=timedelta(seconds=1), composition='any', 
		training_stop_flags = M2Optimizer.StopConditions.GLOBAL_MIN_LOSS 
							| M2Optimizer.StopConditions.GLOBAL_MAX_DURATION
				),
	plotstyle=PS
)	
m2target_col = 'T_CPU'
m2obj.fit(df2['CPU'].to_frame(), df2[m2target_col], plot= bool(PLOT_FLAGS & PLOT_M2_TRAINING_HISTORY))
m2pred = m2obj.predict(df2['CPU'].to_frame(), 		plot= bool(PLOT_FLAGS & PLOT_M2_PARTIAL_PREDICTION), against = df2[m2target_col])

m3_source_col = m2target_col
df3=df.loc[:,df.columns != m3_source_col].copy(deep=True)
target_col = 'Temp_residue'
df3.loc[:,target_col] = (df2.loc[:,m3_source_col].copy(deep=True)-m2pred).rename(target_col)

m3obj=M3(
	XGBStrategy(n_estimators=1000),
	plotstyle=PS)
# m3obj.cross_validation(plot = bool(PLOT_FLAGS & PLOT_M3_TRAINING_HISTORY),
# 	X = df3.loc[:,df3.columns != target_col],	
# 	y = df3.loc[:,target_col],
# 	)
m3obj.fit(plot = bool(PLOT_FLAGS & PLOT_M3_TRAINING_HISTORY),
	X = df3.loc[:,df3.columns != target_col],	
	y = df3.loc[:,target_col],
	)
m3pred = m3obj.predict(plot = bool(PLOT_FLAGS & PLOT_M3_PARTIAL_PREDICTION),			against=df3[target_col],
	X = df3.loc[:,df3.columns != target_col]
	)


#%% Define the plotter for the composite prediction
class CompositePredictionPlotter:
	def __init__(self, plotstyle: PlotStyle):
		self.plotstyle = plotstyle
	def plot(self, reference, pred):
		def sanitize_plotstyle(ps: PlotStyle):
			if ps is None: ps = get_plotstyle('')
			if not (isinstance(ps.M2_training_score_history_title, str)): 
				ps.M2_training_score_history_title = ''
			if not (isinstance(ps.M2_training_score_history_xlabel, str)): 
				ps.M2_training_score_history_xlabel = ''
			if not (isinstance(ps.score_label, str)): 
				ps.score_label = ''
			if not (isinstance(ps.M2_training_score_history_filename, str)): 
				ps.M2_training_score_history_filename = 'M2_training_score_history'
			if not (isinstance(ps.save_figure_extension, str)): 
				ps.save_figure_extension = 'svg'
			if not (isinstance(ps.savefig_bbox_inches, str)): 
				ps.savefig_bbox_inches = 'tight'
			if not (is_valid_font(ps.label_fontfamily)): 
				ps.label_fontfamily = None
			if not (isinstance(ps.label_fontsize, (int,float)) and ps.label_fontsize > 0): 
				ps.label_fontsize = None
			if not (isinstance(ps.title_fontsize, (int,float)) and ps.title_fontsize > 0): 
				ps.title_fontsize = None
			if not (isinstance(ps.tick_label_fontsize, (int,float)) and ps.tick_label_fontsize > 0): 
				ps.tick_label_fontsize = None
			if not (isinstance(ps.spine_linewidth, (int,float)) and ps.spine_linewidth > 0): 
				ps.spine_linewidth = None
			if not mcolors.is_color_like(ps.facecolor):
				ps.facecolor = None
			if not (isinstance(ps.grid_options, list)): 
				ps.grid_options = []
			if not isinstance(ps.single_figsize,tuple) or len(ps.single_figsize) != 2 or any([not isinstance(dim,(float,int)) or dim <=0 for dim in ps.single_figsize]):
				ps.single_figsize = None
			return ps
		PS = sanitize_plotstyle(self.plotstyle)

		fig, ax = plt.subplots(1, figsize=PS.single_figsize)
		if isinstance(reference, pd.Series): 
			x = reference.index
			ref = reference.values
		else: 
			x = range(len(reference))
			ref = reference
		ax.plot(x, ref, **PS.reference_plot_options)	# Reference line
		ax.plot(x, pred, **PS.prediction_plot_options)	# Prediction line
		ax.set_facecolor(PS.facecolor)
		for grid_option in PS.grid_options: ax.grid(**grid_option)
		ax.set_xlabel(PS.xlabel_time, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
		ax.set_ylabel(PS.ylabel_temperature, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
		ax.legend(fontsize=PS.legend_fontsize)
		ax.tick_params(axis='both', labelsize=PS.tick_label_fontsize)
		if PS.spine_linewidth is not None:
			for spine in ax.spines.values(): spine.set_linewidth(PS.spine_linewidth)
		ax.annotate(PS.score_label+f' = {SCORE_FUNCTION(ref,pred):0.4f}', xy=(0.99,0.04), xycoords='axes fraction',
			fontsize=PS.annotate_fontsize, horizontalalignment='right', verticalalignment='bottom')
		PlotStyle.settitle_and_savefig(fig, ax,
			savefig_options=PlotStyle.compose_savefig_options(
				fname=PS.composite_prediction_filename, 
				format=PS.save_figure_extension, 
				bbox_inches=PS.savefig_bbox_inches
			),
			set_title_options=PlotStyle.compose_set_title_options(
				label=PS.composite_prediction_title, 
				fontsize=PS.title_fontsize,
				fontname=PS.label_fontfamily
			),
			savefig=PS.composite_prediction_savefig,
			save_with_title=PS.save_with_title
		)
		plt.show(block=True)
predplotter=CompositePredictionPlotter(PS)
if plotpred:=bool(PLOT_FLAGS & PLOT_COMPOSITE_PREDICTION): predplotter.plot(df2[m2target_col], m2pred+m3pred)

if NOPLOT: report_output(SCORE_FUNCTION(df2[m2target_col],m2pred+m3pred))