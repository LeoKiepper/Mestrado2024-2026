# import bagpy
from bagpy import bagreader
from dataclasses import dataclass, fields
import pandas as pd
import re, bagpy, os, pickle, hashlib, shutil, warnings, sys
from IPython import get_ipython
from warnings import warn
# Override the standard warnings formatter
warnings.formatwarning = lambda msg, cat, fn, ln, line=None: f"{cat.__name__}: {msg}\n"

# Get the current IPython instance and override its warning display
ip = get_ipython()
if ip is not None:
    ip.showwarning = lambda msg, cat, fn, ln, *args, **kwargs: sys.stderr.write(f"{cat.__name__}: {msg}\n")
TELEMS_DIR = os.path.abspath('telems')
BUILD_FLAG=0
BAG_FILENAME=None
if not os.path.exists(TELEMS_DIR):	os.makedirs(TELEMS_DIR)
pickle_filename = lambda bagfile_name: TELEMS_DIR + '\\' + bagfile_name + '.telemetry.pkl'

@dataclass
class telclass:
	tel: pd.DataFrame
	timestamp_zero: float
expected_fields = {f.name for f in fields(telclass)}

def builder(bagfile, PlotarGraficos=False):
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


	if PlotarGraficos:
		# ===================== Plota a telemetria extraída da bag e a média móvel da temperatura da CPU =====================
		fig, ax = bagpy.create_fig(1)
		ax=ax[0]
		ax.plot('T_CPU',     data = telemetria,       linewidth=0.8)
		ax.set_ylabel('Temperatura da CPU  '+r'$[°C]$')
		ax.set_xlabel('Tempo '+r'$[s]$')
		ax.set_facecolor('#fafafa')
		ax.grid(which='major', color='#e0e0e0', linewidth=1.5)
		ax.grid(which='minor', color='#f0f0f0', linewidth=1)
		fig.savefig("TelemetryTemperature.svg", format='svg')

		# ax[1].plot('i_B1',      data = telemetria,	  label = r'$i_{B1}$',                linewidth=0.8)
		# ax[1].plot('i_B2',      data = telemetria,      label = r'$i_{B2}$',                linewidth=0.8)
		# ax[1].plot('i_B1_avg',  data = telemetria,      label = r'$\overline{i}_{B1}$',     linewidth=2)
		# ax[1].plot('i_B2_avg',  data = telemetria,      label = r'$\overline{i}_{B2}$',     linewidth=2)
		# ax[1].set_ylabel('Corrente da bateria '+r'$[mA]$')
		fig, ax = bagpy.create_fig(1)
		ax=ax[0]
		ax.plot('i_M1',      data = telemetria,      label = r'$i_{M1}$',                linewidth=0.8)
		ax.plot('i_M2',      data = telemetria,      label = r'$i_{M2}$',                linewidth=0.8)
		ax.plot('i_M3',      data = telemetria,      label = r'$i_{M3}$',                linewidth=0.8)
		ax.plot('i_M4',      data = telemetria,      label = r'$i_{M4}$',                linewidth=0.8)
		ax.set_ylabel('Corrente do motor '+r'$[mA]$')
		ax.legend(fontsize=18)
		ax.set_xlabel('Tempo '+r'$[s]$')
		ax.set_facecolor('#fafafa')
		ax.grid(which='major', color='#e0e0e0', linewidth=1.5)
		ax.grid(which='minor', color='#f0f0f0', linewidth=1)
		fig.savefig("TelemetryMotors.svg", format='svg')

		# ax[3].plot('LED_F',     data = telemetria,      label = r'$LED_{F}$',               linewidth=2, alpha=0.7)
		# ax[3].plot('LED_B',     data = telemetria,      label = r'$LED_{B}$',               linewidth=2, alpha=0.7)
		# ax[3].set_ylabel('Carga '+r'$[\%]$')
		fig, ax = bagpy.create_fig(1)
		ax=ax[0]
		for CPU in range(NumCPUs):
			# kw = dict(linewidth=0.8) if CPU == 0 else dict(alpha=0.6, linewidth=0, linestyle=None, marker='o', markersize=1)
			kw=dict(linewidth=0.8)
			ax.plot(f'CPU_{CPU}', data = telemetria, label = fr'$CPU_{{{CPU+1}}}$', **kw)
		ax.set_ylabel('Utilização da CPU '+r'$[\%]$')
		ax.legend(fontsize=18,markerscale=5)
		ax.set_xlabel('Tempo '+r'$[s]$')
		ax.set_facecolor('#fafafa')
		ax.grid(which='major', color='#e0e0e0', linewidth=1.5)
		ax.grid(which='minor', color='#f0f0f0', linewidth=1)
		fig.savefig("TelemetryCPU.svg", format='svg')

		fig, ax = bagpy.create_fig(1)
		ax=ax[0]
		ax.plot('cmd_vel',      data = telemetria,      label = 'comandada',                linewidth=0.8)
		ax.plot('robot_vel',      data = telemetria,      label = 'medida',                linewidth=0.8)
		ax.set_ylabel('Velocidade')
		ax.legend(fontsize=18)
		ax.set_xlabel('Tempo '+r'$[s]$')
		ax.set_facecolor('#fafafa')
		ax.grid(which='major', color='#e0e0e0', linewidth=1.5)
		ax.grid(which='minor', color='#f0f0f0', linewidth=1)
		fig.savefig("TelemetryVel.svg", format='svg')
	
	return telclass(
		tel=telemetria,
		timestamp_zero=b.start_time
	)
def __CalcBagHash(bagfile):
    hash_md5 = hashlib.md5()  # No security concerns, choose md5
    with open(bagfile, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
	
	
    return hash_md5.hexdigest()
def __LoadTelemetry(bagfile):
	try:										
		with open(pickle_filename(BAG_FILENAME), 'rb') as f: var = pickle.load(f)
	except FileNotFoundError as e:						warn("Pickle file not found."); return BUILD_FLAG
	except pickle.UnpicklingError as e:					warn("Failed to unpickle the file."); return BUILD_FLAG
	except EOFError as e:								warn("Incomplete or corrupted pickle file."); return BUILD_FLAG
	except AttributeError as e:							warn("Error accessing saved object attributes."); return BUILD_FLAG
	except ImportError as e:							warn("Missing module required to load pickle."); return BUILD_FLAG
	except IndexError as e:								warn("Index error while loading pickle."); return BUILD_FLAG
	except TypeError as e:								warn("Type error while loading pickle."); return BUILD_FLAG
	if not isinstance(var, dict): 						warn("Object structure not recognized."); return BUILD_FLAG
	if not isinstance(baghash := var['baghash'], str):	warn("Loaded bag file hash in unexpected format."); return BUILD_FLAG
	if not isinstance(tel := var['tel'], telclass):			warn("tel in unexpected format."); return BUILD_FLAG
	if not set(vars(tel).keys()) == expected_fields: 	warn("tel fields do not match expected structure."); return BUILD_FLAG
	if not baghash == __CalcBagHash(bagfile):			warn("Hash mismatch detected."); return BUILD_FLAG
	
	
	return tel
def __ValidateBagfile(bagfile):
	if not isinstance(bagfile, str): 					raise ValueError('bagfile must be a file name')
	global BAG_FILENAME
	BAG_FILENAME = os.path.splitext(os.path.basename(bagfile))[0]
	bagfile_in_telems = os.path.join(TELEMS_DIR, BAG_FILENAME + '.bag')
	bagfile_dir = os.path.abspath(os.path.dirname(bagfile))
	if os.path.exists(bagfile):
		if os.path.exists(bagfile_in_telems):
			if bagfile_dir!=TELEMS_DIR:
				if __CalcBagHash(bagfile_in_telems) == __CalcBagHash(bagfile):
					warn(f"Specified bag file is duplicated in {TELEMS_DIR} and will be used instead.")
					bagfile=bagfile_in_telems
				else:
					print(f"A different bag file was found in {TELEMS_DIR} with the same name. Which bag should be used?")
					print(f"	1. {bagfile_in_telems}")
					print(f"	2. {bagfile}")
					whichbag = input()
					if whichbag!='2': bagfile = bagfile_in_telems
		else:
			print(f'Bag file is not in the expected directory. Move to {TELEMS_DIR} ?')
			move = input('y/[n]')
			if move == 'y': 	
				try: 
					shutil.move(bagfile, bagfile_in_telems)
					bagfile = bagfile_in_telems
				except: 	warn('Move failed')
	else:
		if os.path.exists(bagfile_in_telems): bagfile=bagfile_in_telems
		else: raise FileNotFoundError("Bag file not found.")
	return bagfile
def __SaveTelemetry(tel,bagfile):
	savedata = {
		"baghash": __CalcBagHash(bagfile),
		"tel": tel
	}

	with open(pickle_filename(BAG_FILENAME), "wb") as f:
		pickle.dump(savedata, f)
def ResolveTel(bagfile):
	"""
		Loads or creates a DataFrame variable converted from the specified bag file
	"""
	#%% ========================================================================  Input validation
	bagfile = __ValidateBagfile(bagfile)

	#%% ========================================================================  Attempts to load telemetry 
	tel = __LoadTelemetry(bagfile)
	if tel == BUILD_FLAG: 
		warn("Telemetry build required.")
		tel = builder(bagfile)
		__SaveTelemetry(tel,bagfile)
	else: print('Found valid .pkl file')

	#%% End of function
	return tel