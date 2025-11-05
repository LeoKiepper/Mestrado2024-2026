import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import tellib
from bagpy import bagreader
import numpy as np
import re
import ruptures as rpt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from measure_utils import report_output, NOPLOT

MODEL_FILENAME = os.path.join("models","lstm_model.pt")
NUM_EPOCHS = 8
SEQ_LENGTH = 50
LAG_WINDOW = 10  # seconds. Set to zero to not generate lag features
LAG_FEATURE_EXCLUDE_COLUMNS = ['T_CPU']  # Columns to exclude from lag feature generation
TARGET_MEMORY_FOR_LAG_FEATURES_BYTES = 20e6
HIDDEN_SIZE = 30
NUM_LAYERS = 2
FEATURE_SCALER = StandardScaler()
TARGET_SCALER = StandardScaler()
TRAINING_SPLIT = 0.8
LEARNING_RATE = 0.0005

#region Dataset loading and preprocessing
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
def calcfeatures(tel: pd.DataFrame) -> pd.DataFrame:
	target=['T_CPU']
	features=[
				#'T_CPU',
				#'i_B1','i_B1_avg','i_B2','i_B2_avg',
				'i_M1','i_M2','i_M3','i_M4',
				# 'T_CPU_trend',# 'T_CPU_detrend',
				# 'T_CPU_trend_lag1','T_CPU_trend_lag2','T_CPU_trend_lag3','T_CPU_trend_lag4',
				'CPU_0','CPU_1','CPU_2','CPU_3','CPU_4','CPU_5','CPU_6','CPU_7',
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
	CPU_cols = [col for col in tel.columns if regex.match(col)]
	if ('CPU' in features) and CPU_cols: # cols evaluates to false if it is empty
		df.loc[:,'CPU'] = tel.loc[:,CPU_cols].max(axis=1)/100
	for col in CPU_cols:
		df.loc[:,col] = tel.loc[:,col]/100

	return df.dropna()
df=calcfeatures(tel)
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
idx = segments[1]['pos_low']
df=df[range(len(df))<idx]			# clip dataset before temperature peak
def generate_lag_features(df: pd.DataFrame, lag_window: float, target_lag_memory_bytes: int, report: bool = False, exclude: list = []):
    from plan_lag_feature_spacing import plan_lag_feature_spacing

    # Columns to generate lag features for
    lag_spacing = plan_lag_feature_spacing(df, lag_window, target_lag_memory_bytes, report=report, exclude=exclude)
    if lag_spacing is None:
        raise ValueError("Insufficient memory for lag features.")

    n_lags = int(lag_window // lag_spacing)
    lagged_features = []
    lag_cols = [col for col in df.columns if col not in exclude]
    for lag in range(1, n_lags + 1):
        lagged = df[lag_cols].shift(lag).add_suffix(f"_lag{lag}")
        lagged_features.append(lagged)

    # Concatenate original columns and lagged features
    result = pd.concat([df] + lagged_features, axis=1)
    return result.dropna()
if LAG_WINDOW>0: df = generate_lag_features(df, lag_window=LAG_WINDOW, target_lag_memory_bytes=TARGET_MEMORY_FOR_LAG_FEATURES_BYTES, report=False, exclude=LAG_FEATURE_EXCLUDE_COLUMNS)

TARGET_COL = 'T_CPU'
FEATURE_COL = [col for col in df.columns if col != TARGET_COL]
X = df[FEATURE_COL].values
y = df[TARGET_COL].values
def create_sequences(X, y, seq_length):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        targets.append(y[i+seq_length])
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32)
X_scaled = FEATURE_SCALER.fit_transform(X)
y_scaled = TARGET_SCALER.fit_transform(y.reshape(-1, 1)).flatten()

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
split_idx = int(TRAINING_SPLIT * len(X_seq))
X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
y_seq_train, y_seq_val = y_seq[:split_idx], y_seq[split_idx:]
train_dataset = TensorDataset(X_seq_train, y_seq_train)
val_dataset = TensorDataset(X_seq_val, y_seq_val)
loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#endregion

#region Training and evaluation for LSTM
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()
def train_model():
    input_size = X_seq.shape[2]
    model = Model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    from tqdm import tqdm
    for epoch in tqdm(range(NUM_EPOCHS), desc="LSTM Training"):
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = 0
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
            val_loss /= len(val_dataset)
    model_dir = os.path.dirname(MODEL_FILENAME)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), MODEL_FILENAME)
    return model
def get_model(filename):
    # import os
    # input_size = X_seq.shape[2]
    # model = Model(input_size=input_size, hidden_size=32)
    # if os.path.exists(filename):
    #     model.load_state_dict(torch.load(filename))
    #     model.eval()
    #     return model
    # else:
    #     return train_model()
	return train_model()

model = get_model(MODEL_FILENAME)
model.eval()
with torch.no_grad():
    rnn_pred = model(X_seq).numpy()
pred_orig = TARGET_SCALER.inverse_transform(rnn_pred.reshape(-1, 1)).flatten()
#endregion

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y[SEQ_LENGTH:], pred_orig)

if not NOPLOT and (plot:=True):
	plt.figure()
	plt.plot(df.index[SEQ_LENGTH:], y[SEQ_LENGTH:], label='Reference')
	plt.plot(df.index[SEQ_LENGTH:], pred_orig, label='Prediction')
	plt.annotate(f'RMSE = {rmse:.4f}', xy=(0.99,0.04), xycoords='axes fraction', fontsize=12, ha='right', va='bottom')
	plt.legend()
	plt.xlabel('Time')
	plt.ylabel(TARGET_COL)
	plt.show(block=True)

report_output(rmse)