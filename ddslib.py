import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('QtAgg')
from collections.abc import Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.metrics import root_mean_squared_error
import inspect
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import math
from tqdm.auto import tqdm
import plotext as plt2
import sys
import threading
import time
from typing import Callable, List
class FunctionWrapper(ABC):
	def __init__(self):
		"""
		Instantiate a wrapper object to provide a common ground access to fields belonging to 
		a model and its associated objects to the method specified in subclass definition.
		The model field is initiated as None and should be set at model instantiation.
		"""
		self._model = None
	def __call__(self):
		return

class dds(BaseEstimator, RegressorMixin):
	def __init__(self, tel: pd.DataFrame, input_col: str, highcpudetector: callable, M1: callable, M2: callable, M3: callable):
		"""
			:param: tel: DataFrame containing the data to be analyzed.
			:param: col: Column name in the DataFrame that contains the CPU usage data.
		"""
		self.tel=tel
		if not input_col in tel.columns: raise ValueError(f"Column '{input_col}' not found in passed DataFrame.")
		self.input_col=input_col
		self.highcpudetector=highcpudetector
		self.M1=M1
		self.M2=M2
		self.M3=M3

class M2(BaseEstimator, RegressorMixin):
	def __init__(self, kernel: FunctionWrapper, optimizer: FunctionWrapper, random_state = None):
		"""
		:param source: Iterable containing the input data to be fed to the model.
		:param target: Iterable containing the output data to which the model must be fitted.
		:param kernel: Function, as a FunctionWrapper object, containing logic and calculations that will produce predictions.
		:param loss: Loss function, as a FunctionWrapper object, to be used in training the model.
		:param optimizer: Optimizer function, as a FunctionWrapper object, to be used in training the model.
		"""
		self._kernel = kernel;			self._kernel._model = self
		self._optimizer = optimizer;	self._optimizer._model = self
		
		# Hyperparamaters
		self.random_state = random_state
	def get_model_params(self):
		"""
		:return: The parameters of this model.
		"""		
		return self._kernel._params
	def fit(self, X, y):
		"""
		Orchestrates the training process by validating inputs and coordinating the kernel and optimizer.
		:param X: Input features.
		:param y: Target values.
		"""
		X, y = check_X_y(X, y)
		self._optimizer.fit(X, y)
	def predict(self, X):
		X = check_array(X, ensure_2d=False, dtype=float)
		return self._kernel.predict(X)

	def score(self, X, y):
		"""
		Evaluates the model by validating inputs and delegating to the optimizer's score method.
		:param X: Input features.
		:param y: Target values.
		:return: Model score.
		"""
		X, y = check_X_y(X, y)
		return self._optimizer.score(X, y)
	def get_params(self, deep=True):
		params = {
			'random_state': self.random_state,
		}
		# Kernel params
		kernel_params = self._kernel.get_params(deep=deep) if hasattr(self._kernel, "get_params") else {}
		for k, v in kernel_params.items():
			params[f"kernel__{k}"] = v
		# Optimizer params
		optimizer_params = self._optimizer.get_params(deep=deep) if hasattr(self._optimizer, "get_params") else {}
		for k, v in optimizer_params.items():
			params[f"optimizer__{k}"] = v
		return params
	def set_params(self, **params):
		kernel_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("kernel__")}
		optimizer_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("optimizer__")}
		if kernel_params and hasattr(self._kernel, "set_params"):
			self._kernel.set_params(**kernel_params)
		if optimizer_params and hasattr(self._optimizer, "set_params"):
			self._optimizer.set_params(**optimizer_params)
		return self
class M2Kernel(FunctionWrapper):
	@dataclass
	class Params:
		"""
		A dataclass to encapsulate the parameters for the M2 kernel model.

		Attributes:
			KCPU (float): CPU usage to heat transfer gain.
			KTemp (float): Temperature gradient usage to heat transfer gain.
			TauCPU (float): Time constant for the CPU usage effect on temperature.
			TauTemp (float): Time constant for the temperature gradient effect on temperature.
		"""
		KCPU: float
		KTemp: float
		TauCPU: float
		TauTemp: float

	@staticmethod
	def _params_domain_restrict(params: Params) -> Params:
		tau_tol=1e-9
		min_K=0.00001

		KCPU = math.sqrt(params.KCPU**2+min_K**2)
		KTemp = math.sqrt(params.KTemp**2+min_K**2)
		TauCPU = params.TauCPU if params.TauCPU > tau_tol else tau_tol
		TauTemp = params.TauTemp if params.TauTemp > tau_tol else tau_tol
		return M2Kernel.Params(KCPU=KCPU, KTemp=KTemp, TauCPU=TauCPU, TauTemp=TauTemp)
	def set_model_parameters(self, params: Params):
		"""
			Sets parameters of the M2 kernel model.

			:param params: An instance of M2KernelParams containing the new parameter values.
		"""
		# Calculates the parameters for the M2 kernel, according to the base closed-form analytical function:
		# 	TempNext = TempCurrent
		# 	+ KCPU * FCPU(CPUCurrent) * (1 - exp(- t / TauCPU))
		# 	- KTemp * FTEMP(TempCurrent, TempExt) * (1 - exp(- t / TauTemp))
		params = M2Kernel._params_domain_restrict(params)
		self._params = {
			'KCPU': params.KCPU,
			'KTemp': params.KTemp,
			'TauCPU': params.TauCPU,
			'TauTemp': params.TauTemp,
			# 'ALPHA_CPU': np.exp(-self.Dt/params.TauCPU),
			'BETA_CPU': 1 - np.exp(-self.Dt/params.TauCPU),
			# 'ALPHA_TEMP': np.exp(-self.Dt/params.TauTemp),
			'BETA_TEMP': 1 - np.exp(-self.Dt/params.TauTemp),
		}
	def get_model_params(self) -> Params:
		"""
		Returns current M2 kernel model parameters.
		:return: Parameter values, organized in a Params object.
		"""
		return self.Params(
			KCPU=self._params['KCPU'],
			KTemp=self._params['KTemp'],
			TauCPU=self._params['TauCPU'],
			TauTemp=self._params['TauTemp'])
	def guess_params(self) -> Params:
		return self.Params(
			KCPU = np.random.uniform(0,10),
			KTemp = np.random.uniform(0,1),
			TauCPU = np.random.uniform(0,1),
			TauTemp = np.random.uniform(0,0.5),
			)
	def _next_temp(self, cpu_current, temp_current, temp_ext):
		r"""
		Computes the next temperature incrementally based on the current CPU usage, 
		previous temperature, and external temperature.
			Implements incremental form for the closed form analytic model:
				TempNext = TempPrev
				+ KCPU * FCPU(CPUCurrent) * (1 - np.exp(-t / tauCPU))
				- KTEMP * FTEMP(TempPrev, TempExt) * (1 - np.exp(-t / tauTEMP))

		:param CPUCurrent: Current CPU usage.
		:param TempCurrent: Current temperature.
		:param TempExt: External temperature.
		:return: Next temperature.
		"""
		KCPU=self._params['KCPU']
		KTEMP=self._params['KTemp']
		BETA_CPU=self._params['BETA_CPU']
		BETA_TEMP=self._params['BETA_TEMP']
		FCPU = self.FCPU
		# FCPU = lambda cpu: cpu**2
		FTEMP = self.FTEMP
		# FTEMP = lambda TempPrev,TempExt: TempPrev - TempExt

		DeltaTempFromCPU  = KCPU * FCPU(cpu_current) * BETA_CPU
		DeltaTempFromTemp = KTEMP * FTEMP(temp_current, temp_ext) * BETA_TEMP

		return temp_current + DeltaTempFromCPU - DeltaTempFromTemp
	def predict(self, X):
		r"""
		Calculates a prediction series using the parrameters currently saved in the object.

		:param cpu_series: Pandas Series containing CPU usage values over time.
		:return: Pandas Series containing the simulated temperature values over time.
		"""			
		# X = check_array(X, ensure_2d=False, dtype=float)
		X = X.ravel()
		pred = [self._temp_amb] * len(X)
		for cc, cpu in enumerate(X[1:], start=1):
			pred[cc] = self._next_temp(cpu, pred[cc-1], self._temp_amb)
		return pred
	def __init__(self, FCPU: callable, FTEMP: callable, TempAmb: float, Dt:float,  params: Params=None, noise_level=0):	# Dt test value was Dt = 30/510.0
		r"""
		Instantiates the M2Kernel object

		:param FCPU: unit transform functions from cpu% to heat.
		:param FTEMP: unit transform functions from temperature gradient to heat.
		:param params: Dictionary of parameters for the M2 kernel model, according to the base closed-form analytical function:

			TempNext = TempCurrent
			+ KCPU * FCPU(CPUCurrent) * (1 - exp(- t / TauCPU))
			- KTemp * FTEMP(TempCurrent, TempExt) * (1 - exp(- t / TauTemp))

			It is strongly recommended to use the CalculateParams method provided by this class to generate this dictionary.
		:param TempAmb: Ambient temperature, used as the initial temperature.
		:param Dt: Time step size as used in the dataset this model will target.
		"""			
		super().__init__()
		self.FCPU = FCPU
		self.FTEMP = FTEMP
		self._temp_amb = TempAmb
		self.noise_level = noise_level

		self.Dt = Dt # set_params depends on Dt, so it must be called after setting it.
		if params is None: params = self.guess_params()
		self.set_model_parameters(params)	
	def get_params(self, deep=True):
		params = {
			'noise_level': self.noise_level,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self
class M2Optimizer(FunctionWrapper, BaseEstimator, RegressorMixin):
	class Plotter:
		from matplotlib.figure import Figure
		from matplotlib.axes import Axes
		def update(self):			
			data = self.data_source()
			self.line.set_xdata(list(range(len(data))))
			self.line.set_ydata(data)
			self.ax.relim()
			self.ax.autoscale_view(scalex=False, scaley=True)
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
			plt.pause(self.update_interval)
		def start(self):
			self.fig, self.ax = plt.subplots(1)
			# self._set_topmost(self.fig)
			data = self.data_source()
			self.line, = self.ax.plot(data)
			self.ax.set_xlim(0,len(data)-1)
			self.ax.relim()
			self.ax.autoscale_view(scalex=False, scaley=True)
			plt.ion()
		def stop(self):
			plt.ioff()

		def __init__(self, data_source: Callable[[], List[float]], update_interval: float = 0.01):
			self.data_source = data_source
			self.update_interval = update_interval
			self.fig = None
			self.ax = None
			self.line = None
	class StopConditions:
		MAX_ITERATIONS = 1 << 0
		MIN_LOSS      = 1 << 1
		AVG_GRADIENT  = 1 << 2
		
		@classmethod
		def _compute_code(cls, obj):
			code = 0
			if obj.current_iteration == obj.max_iter:		code |= cls.MAX_ITERATIONS
			if abs(obj._loss_history[-1]) <= obj.min_loss:	code |= cls.MIN_LOSS
			if abs(obj._loss_gradient()) <= obj.tol:		code |= cls.AVG_GRADIENT
			return code
		@classmethod
		def compose(cls, selected_flags, composition):
			if not cls.validate(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def stop_fn(obj):
				code = cls._compute_code(obj)
				if		composition == 'any':
					cond = bool(code & selected_flags)
				else: # composition == 'all'
					other = selected_flags & ~cls.MAX_ITERATIONS
					cond = bool((code & cls.MAX_ITERATIONS) or ((code & other) == other))
				if cond:
					obj._stop_reason = cls._describe_flags(code)
					return code
				return 0
			return stop_fn
		@classmethod
		def validate(cls, stop_condition: int):
			attributes = inspect.getmembers(cls, lambda a: isinstance(a, int))
			max_value = sum(v for _, v in attributes)
			if stop_condition < 0 or stop_condition == 0 or stop_condition > max_value:
				raise ValueError("Invalid stop flags.")
			return True
		@classmethod
		def _describe_flags(cls, code):
			flags = []
			if code & cls.MAX_ITERATIONS: flags.append("MAX_ITERATIONS")
			if code & cls.MIN_LOSS:       flags.append("MIN_LOSS")
			if code & cls.AVG_GRADIENT:   flags.append("AVG_GRADIENT")
			return ", ".join(flags)		
	def _reset_history(self):
		self._loss_history = [1.0 + i for i in reversed(range(self.gradient_window))]
		self._score_history = [math.nan]*self.max_iter	
	def fit(self, X, y):
		"""
		Fits the model by iterating over M2Kernel parameters.
		:param X: Input features.
		:param y: Target values.
		:return: self
		"""
		pbar = tqdm(total=self.max_iter, desc="M2 fitting in progress", leave=True)
		self._reset_history()
		self.current_iteration = 0
		best_score = self._loss_history[-1]
		best_params = self._model._kernel.get_model_params()
		self.plotter.start()
		while not self.stop_condition(self):
			try:
				with np.errstate(over='raise', invalid='raise'):
					self._update_loss(X, y)
			except (ValueError, FloatingPointError):
				raw = self._model._kernel.guess_params()
				self._model._kernel.set_model_parameters(raw)
				continue
			
			current_score = self._loss_history[-1]
			current_params = self._model._kernel.get_model_params()
			if current_score > best_score:
				best_score = current_score
				best_params = current_params

			code = self.StopConditions._compute_code(self)
			if code & self.StopConditions.AVG_GRADIENT:
				next_params = self._model._kernel.guess_params()
			else:
				next_params = self.optimize(current_params)

			self._model._kernel.set_model_parameters(next_params)
			self.current_iteration += 1
			pbar.update(1)
			self.plotter.update()
		pbar.close()
		self.plotter.stop()
		self._model._kernel.set_model_parameters(best_params)
		print("[INFO] Stop conditions met:", getattr(self, '_stop_reason', ''))
		return self
	# def fit(self, X, y):
	# 	"""
	# 	Fits the model by iterating over M2Kernel parameters.
	# 	:param X: Input features.
	# 	:param y: Target values.
	# 	:return: self
	# 	"""
	# 	pbar = tqdm(total=self.max_iter, desc="M2 fitting in progress", leave=True)
	# 	self.current_iteration = 0
	# 	while not self.stop_condition(self):
	# 		# _update_loss(X, y) calls the score method, which calls self._model._kernel.predict(X)
	# 		self._update_loss(X, y)

	# 		# Training step
	# 		self._model._kernel.set_model_parameters(
	# 			self.optimize(self._model._kernel.get_model_params())
	# 		)
	# 		self.current_iteration += 1
	# 		pbar.update(1)
	# 	pbar.close()
	# 	print("[INFO] Stop conditions met:", getattr(self, '_stop_reason', ''))
	# 	return self
	def score(self, X, y):
		"""
		Custom scoring method using RMSELoss.
		:param X: Input features.
		:param y: Target values.
		:return: Negative RMSE (to align with sklearn's maximization convention).
		"""
		predictions = self._model._kernel.predict(X)
		rmse = root_mean_squared_error(y, predictions)
		return -rmse
	def optimize(self, params: M2Kernel.Params) -> M2Kernel.Params:
			"""
			Implements a single step of Stochastic Gradient Descent (SGD) to calculate the next set of candidate parameters.
			:param params: Current parameters of the M2Kernel as a Params object.
			:return: Updated parameters as a Params object.
			"""
			KCPU, KTemp, TauCPU, TauTemp = params.KCPU, params.KTemp, params.TauCPU, params.TauTemp

			grad = self._loss_gradient()
			grad_KCPU = 2 * grad
			grad_KTemp = 2 * grad
			grad_TauCPU = 2 * grad
			grad_TauTemp = 2 * grad

			KCPU -= self.learning_rate * grad_KCPU
			KTemp -= self.learning_rate * grad_KTemp
			TauCPU -= self.learning_rate * grad_TauCPU
			TauTemp -= self.learning_rate * grad_TauTemp

			# Return updated parameters as a Params object
			return M2Kernel.Params(KCPU, KTemp, TauCPU, TauTemp)
	def _update_loss(self, X, y):
		current_loss = self.score(X, y)
		self._loss_history = (self._loss_history + [current_loss])[-self.gradient_window:]
		self._score_history[self.current_iteration]=-current_loss
	def _loss_gradient(self):
		return (self._loss_history[-1] - self._loss_history[0]) / self.gradient_window if self.gradient_window > 1 else self._loss_history[0]
	def get_params(self, deep=True):
		params = {
			'learning_rate': self.learning_rate,
			'max_iter': self.max_iter,
			'tol': self.tol,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self
	def get_score_history(self):
		
		return self._score_history
	def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, gradient_window=5, min_loss=0.01, \
				composition='any', stop_flags: int = StopConditions.MAX_ITERATIONS | StopConditions.AVG_GRADIENT):
		"""
		Initializes the Optimizer with a learning rate, maximum iterations, and tolerance.
		:param learning_rate: Step size for parameter updates.
		:param max_iter: Maximum number of iterations for training.
		:param tol: Tolerance for stopping criteria based on loss improvement.
		"""
		super().__init__()
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.tol = tol
		self.min_loss = min_loss
		self.gradient_window = gradient_window
		self._loss_history=[]
		self._score_history=[]
		self.current_iteration = 0

		if not self.StopConditions.validate(stop_flags): raise ValueError("Failed to validate stop flags.")
		self.stop_condition = self.StopConditions.compose(stop_flags,composition)

		self.plotter=self.Plotter(self.get_score_history)

if __name__ == "__main__":
	def AddNoise(df,Pt,col,scale,ceil=1,floor=0):
		Noise=np.random.normal(scale=scale, size=df[col].shape)
		col=df.columns.get_loc(col)
		for i in range(1, len(Pt)):  # Começamos de 1 para termos um intervalo de t[i-1] a t[i]
			start = df.index.get_loc(Pt[i-1])  # Índice do início do trecho
			stop = df.index.get_loc(Pt[i])-1  # Índice do final do trecho

			c = df.iloc[start:stop, col]
			r = Noise[start:stop]
			val = c + r
			
			val_max=max(val)
			if ceil is not None and val_max > ceil:
				idmax = val.tolist().index(val_max)
				CorrectionFactor = (ceil - c.iloc[idmax]) / r[idmax]
			else: CorrectionFactor = 1
				# Aplicar o fator de correção no ruído do trecho
			df.iloc[start:stop, col] = df.iloc[start:stop, col] + r * CorrectionFactor
			if floor is not None: df.iloc[start:stop, col] = df.iloc[start:stop, col].clip(lower=0)
	def SynthCPUpercent(Pt, Pcpu, NoiseScale=0.075, TimeArrayParams=dict(tol=0.0001,MinStep=0.01,MinPoints=501,MaxPoints=3001)):
		def _frange(start,stop,**kwargs):
			opcoes={'step','num'}
			arg = opcoes.intersection(kwargs.keys())
			if len(arg) != 1:
				raise ValueError("Exatamente um dos argumentos 'step' ou 'num' deve ser passado.")
				
			from math import floor, log10
			if stop<start:
				aux=stop
				stop=start
				start=aux

			arg=arg.pop()
			match arg:
				case 'step':
					step = kwargs['step']

					# Calcula a ordem de grandeza para os cálculos
					gr=10**floor(log10(step))
					# Converte os argumentos para números inteiros
					iStart=start/gr
					iStep=step/gr
					iStop=stop/gr

					# Constrói o vetor
					NumElem=floor((iStop-iStart)/iStep)+1
					array=np.zeros(NumElem)
					num=iStart
					for i in range(NumElem):
						array[i]=num*gr
						num+=iStep

					return array
				case 'num':
					num = kwargs['num'] - 1
					delta = (stop-start)/num
					
					array = start+np.array(range(num+1))*delta
					return array
		def _GenerateTimeArray(t, tol, MinStep, MinPoints, MaxPoints):
			"""
			Para o intervalo [min(t), max(t)], tenta encontrar um vetor de tempos com número de pontos
			variando de min_points até max_points que satisfaça:
				1. Um passo constante, com valor >= min_step.
				2. Cada valor em t apareça no vetor (dentro de tol * max(1, |ti|)).
			
			Se encontrado, 'snap' os valores para que sejam exatamente t; caso contrário, levanta um erro.
			
			Parâmetros:
				- t: lista ou array de números (em ordem crescente) que devem aparecer no vetor final.
				- min_step: valor mínimo permitido para o passo entre os pontos.
				- tol: tolerância para considerar que um valor gerado é igual a um valor em t.
				- min_points: número mínimo de pontos a tentar.
				- max_points: número máximo de pontos permitidos no vetor.
			"""
			# Bloco de validação dos argumentos
			if not isinstance(t, (list, tuple, np.ndarray)):
				raise ValueError("t deve ser uma lista, tupla ou numpy array.")
			if len(t) == 0:
				raise ValueError("t não pode ser vazio.")
			# Verifica se todos os elementos de t são numéricos
			for ti in t:
				if not isinstance(ti, (int, float)):
					raise ValueError("Todos os elementos em t devem ser números.")
			# Verifica se os valores de t estão em ordem crescente
			if any(t[i] > t[i+1] for i in range(len(t)-1)):
				raise ValueError("Os valores em t devem estar em ordem crescente.")
			
			if not (isinstance(MinStep, (int, float)) and MinStep > 0):
				raise ValueError("min_step deve ser um número positivo.")
			if not (isinstance(tol, (int, float)) and tol >= 0):
				raise ValueError("tol deve ser um número não negativo.")
			if not (isinstance(MinPoints, int) and MinPoints > 0):
				raise ValueError("min_points deve ser um inteiro positivo.")
			if not (isinstance(MaxPoints, int) and MaxPoints >= MinPoints):
				raise ValueError("max_points deve ser um inteiro maior ou igual a min_points.")
			if MinPoints < len(t):
				raise ValueError("min_points deve ser pelo menos o número de valores em t.")
			
			start = min(t)
			stop = max(t)
			if (stop - start) / (MinPoints - 1) < MinStep:
				raise ValueError("Parâmetros conflitantes: usando min_points pontos, o step resultante é menor que min_step.")    # Fim da validação dos argumentos

			found = False
			candidate_time_vec = None

			for n in range(MinPoints, MaxPoints + 1):
				# O passo candidato, se usarmos n pontos, seria:
				candidate_step = (stop - start) / (n - 1)
				if candidate_step < MinStep:
					# Se o step resultante for menor que o mínimo, esse candidato é inválido
					continue
				# Gera o vetor usando a opção 'num' de frange (garante exatamente n pontos)
				time_vec = _frange(start, stop, num=n)
				# Verifica se cada valor em t está presente (dentro da tolerância)
				all_found = True
				for ti in t:
					if np.abs(time_vec - ti).min() > tol * max(1, abs(ti)):
						all_found = False
						break
				if all_found:
					# "Snap": força os valores que estão dentro da tolerância a serem exatamente t
					candidate_time_vec = time_vec.copy()
					for ti in t:
						idx = np.argmin(np.abs(candidate_time_vec - ti))
						if np.abs(candidate_time_vec[idx] - ti) <= tol * max(1, abs(ti)):
							candidate_time_vec[idx] = ti
					found = True
					break

			if not found:
				raise ValueError(
					f"Não foi possível encontrar um vetor de tempos que inclua todos os valores de t (dentro da tolerância {tol}) "
					f"usando entre {MinPoints} e {MaxPoints} pontos e com step >= {MinStep}."
				)
			return candidate_time_vec
		def GenerateDF(t,CPU,TimeArrayParams):
			# Gera um DataFrame que contém os dados listados
			if TimeArrayParams is None:
				TimeArrayParams = {'min_step': 0.1, 'tol': 1e-8, 'min_points': 50, 'max_points': 200}
			
			aux=_GenerateTimeArray(t, **TimeArrayParams)
			df=pd.DataFrame(data={'t':aux,
								'CPU':np.zeros(len(aux)),
							})
			# Alimenta valores do vetor CPU em df, nas posições especificadas em t
			ind = df.loc[df['t'].isin(t)].index[:len(CPU)]
			df.loc[ind, 'CPU'] = CPU[:len(ind)]

			# Preenche valores de df
			t_stop = t[-2]  # Penúltimo valor de t

			last_value = None
			for i in reversed(df.index):  # Percorre de trás para frente
				if i in ind:
					last_value = df.at[i, 'CPU']  # Atualiza o último valor encontrado
				elif last_value is not None and df.at[i, 't'] > t_stop:
					df.at[i, 'CPU'] = last_value  # Replica o valor
			last_value = None
			for i in df.index:  # Percorre do início ao fim
				if i in ind:
					last_value = df.at[i, 'CPU']  # Atualiza o último valor encontrado
				elif last_value is not None and df.at[i, 't'] <= t_stop:
					df.at[i, 'CPU'] = last_value  # Replica o valor

			df['t_mod'] = df['t']  # Cria uma nova coluna para armazenar o t ajustado
			pt_idx = 0  # Índice para percorrer o vetor Pt
			for i in range(len(df)):
				while pt_idx < len(Pt) - 1 and df.loc[i, 't'] >= Pt[pt_idx + 1]:  
					pt_idx += 1  # Atualiza o índice para o próximo intervalo
				df.loc[i, 't_mod'] = df.loc[i, 't'] - Pt[pt_idx]  # Ajusta t conforme Pt
			df.iloc[-1,-1]=df.iloc[-2,-1]+(df.iloc[-2,-1]-df.iloc[-3,-1])
			df=df.rename(columns={'t':'Timestamp'})
			df=df.set_index('Timestamp')
			df=df.rename(columns={'t_mod':'t'})
			return df
		

		df = GenerateDF(Pt,Pcpu,TimeArrayParams)
		AddNoise(df,Pt,'CPU',NoiseScale)

		return df
	def PlotSim(df, offset_sec_axis=30, fontsize_sec_axis=12, hide_sec_axis_line=False, offset_xlabel=50, fontsize_axis_labels=20, fontsize_ticklabels=16, save=None):
		import matplotlib.pyplot as plt
		import matplotlib.ticker as mticker
		from mpl_toolkits.axes_grid1 import host_subplot
		import mpl_toolkits.axisartist as AA

		t=df['t'].values
		# Cria a figura e os eixos host e parasite
		plt.figure(figsize=(10, 5))
		host = host_subplot(111, axes_class=AA.Axes)
		parasite = host.twinx()
		parasite.axis["right"].toggle(all=True)		# Força a exibição do eixo direito do parasite (necessário com axisartist)
		host.set_zorder(2)							# Host (temperatura) acima
		parasite.set_zorder(1)						# Parasite (CPU) abaixo
		parasite.patch.set_visible(False)


		# Plots
		host.plot('Temp', data=df, linewidth=2, label='Resposta da temperatura', zorder=4)
		parasite.plot('CPU', data=df, linewidth=0.9, color='orange', label='Degraus da %CPU', zorder=3)


		# Eixo %CPU
		parasite.set_ylim(0, 1)
		parasite.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*100:.0f}'))


		# Eixo X secundário para os valores de t
		secax = host.secondary_xaxis('bottom')
		secax.set_xticks(t)
		secax.spines['bottom'].set_position(('outward', offset_sec_axis))
		secax.spines['bottom'].set_visible(not hide_sec_axis_line)
		sec_tick_labels = [rf'$t_{{{i}}} = {t_val:.1f}$' for i, t_val in enumerate(t)]
		secax.set_xticklabels(sec_tick_labels, fontsize=fontsize_sec_axis)
		secax.tick_params(axis='x', length=5, width=1, labelsize=fontsize_sec_axis)
		# for t_val in t:     parasite.axvline(x=t_val, color='#a0a0a0', linestyle='--', linewidth=1, zorder=2)   # Linhas verticais para cada valor de t


		# Axis labels
		host.axis["bottom"].label.set_text(r'Tempo $[s]$')
		host.axis["bottom"].label.set_fontsize(fontsize_axis_labels)
		host.axis["bottom"].label.set_pad(offset_xlabel)
		host.axis["left"].label.set_text(r'Temperatura $[°C]$')
		host.axis["left"].label.set_fontsize(fontsize_axis_labels)
		parasite.axis["right"].label.set_text(r'Utilização da CPU $[\%]$')
		parasite.axis["right"].label.set_fontsize(fontsize_axis_labels)


		# Tick labels
		host.axis["left"].major_ticklabels.set_fontsize(fontsize_ticklabels)
		host.axis["bottom"].major_ticklabels.set_fontsize(fontsize_ticklabels)
		parasite.axis["right"].major_ticklabels.set_fontsize(fontsize_ticklabels)


		# Fundo e grid
		host.set_facecolor('#fafafa')
		host.set_axisbelow(True)
		parasite.set_axisbelow(True)
		host.grid(which='major', color='#e0e0e0', linewidth=1.5)					# Major grid para o eixo X e Y do host
		host.xaxis.grid(True, which='major', color='#e0e0e0', linewidth=1.5)		# Força o grid major no eixo X do host


		# Legenda
		handles_host, labels_host = host.get_legend_handles_labels()
		handles_par, labels_par = parasite.get_legend_handles_labels()
		legend_dict = {}
		for h, l in zip(handles_host + handles_par, labels_host + labels_par):	legend_dict[l] = h
		host.legend(list(legend_dict.values()), list(legend_dict.keys()), loc='upper left', fontsize=fontsize_axis_labels)


		if save is not None: plt.savefig(save, bbox_inches='tight')
		plt.show(block=False)
		return
	def SimulateTmep(df: pd.DataFrame,KCPU=4,KTemp=0.1,TauCPU=0.1,TauTemp=0.05, temp_ext: float=None):
		from math import exp

		if temp_ext is None: temp_ext = 0
		if 'Temp' not in df.columns: df['Temp'] = float(temp_ext)

		idx=df.columns.get_loc('CPU')
		idy=df.columns.get_loc('Temp')	
		idt=df.columns.get_loc('t')

		temp_ext=df.iloc[0,idy]
		Dt=df.index[1]-df.index[0]
		for i in range(1,len(df)):
			temp_current=df.iloc[i-1,idy]
			t=df.iloc[i,idt]
			cpu_current=df.iloc[i,idx]
			BETA_CPU = 1 - np.exp(-Dt/TauCPU)
			BETA_TEMP = 1 - np.exp(-Dt/TauTemp)

			FCPU = lambda cpu: cpu**2
			FTEMP = lambda temp_current, temp_ext: temp_current-temp_ext
			DeltaTempFromCPU  = KCPU * FCPU(cpu_current) * BETA_CPU
			DeltaTempFromTemp = KTemp * FTEMP(temp_current, temp_ext) * BETA_TEMP

			TempNext = temp_current + DeltaTempFromCPU - DeltaTempFromTemp
				
			df.iloc[i,idy]=float(TempNext)

		return df

	# ==================  Synthesize a dataset
	# Define times and values for CPU usage
	Pt   = [0  ,    10 ,    12  ,   15  ,   21  ,   30  ]       # Times
	Pcpu = [0.4,    0.3,    0.98,   0.21,   0.61,   0.61]       # Values

	df=SynthCPUpercent(Pt,Pcpu)									# Create a %CPU dataframe
	df=SimulateTmep(df,temp_ext=40)								# Calculate temperature values	
	AddNoise(df,Pt,'Temp',scale=1,ceil=None, floor=None)

	# PlotSim(df)

	def HighCPUDetect(df, margin=0,threshold=.8):
		"""
		Implements high cpu detection logic.
		
		:param cpu_percent: array containing CPU usage
		:param cp: array for change points detected in cpu_percent, listed as indexes for each position.
		Each position of cp is considered the last index of a segment (should not contain 0).
		:param margin: number of indexes to consider on each side of each segment investigated in the detection criterion implemented.
		:param threshold: consider high CPU usage if the average of the segment is greater than this value.

		:return segment: list of detected segments, formatted as dictionaries with the following keys:
		
		"""
		import ruptures as rpt
		cpu_percent=df['CPU'].values
		algo = rpt.Pelt(model="rbf").fit(cpu_percent)
		ChangePoints = algo.predict(pen=5)
		Segments=[]
		for cc, pos_up in enumerate(ChangePoints):
			pos_up=min(pos_up+margin,len(cpu_percent))
			pos_low=max(ChangePoints[cc-1]+1-margin,0) if cc>0 else 0
			
			
			avg=float(np.average(cpu_percent[pos_low:pos_up]))
			if avg>threshold: 	Segments.append({'state':'high','pos_low':pos_low,'pos_up':pos_up,'avg':avg})
			else:				Segments.append({'state':'norm','pos_low':pos_low,'pos_up':pos_up,'avg':avg})
		return ChangePoints, Segments

	# ==================  Define and instantiate model components
	m2obj=M2(
		M2Kernel(lambda cpu: cpu**2, lambda temp_current, temp_ext: temp_current-temp_ext, 
			# params=M2Kernel.Params(KCPU=4, KTemp= 0.1, TauCPU=0.1, TauTemp=0.05),
			# params=M2Kernel.Params(KCPU=5.354987897910978, KTemp= 2.8002979490801128, TauCPU=0.18629771646496662, TauTemp=0.9528960069986708),
			TempAmb=40, Dt=30/510),
		M2Optimizer(max_iter=1000, composition='all', stop_flags=
			M2Optimizer.StopConditions.MIN_LOSS | 
			M2Optimizer.StopConditions.AVG_GRADIENT | 
			M2Optimizer.StopConditions.MAX_ITERATIONS)
	)
	m2obj.fit(df['CPU'].to_frame(), df['Temp'])
	print(m2obj.get_model_params())
	pred=m2obj.predict(df['CPU'].to_frame())
	pred=pd.Series(pred, index=df.index)
	fig, ax = plt.subplots(1)
	ax.plot(df['Temp'], label='Reference' , lw=1)
	ax.plot(pred, label='Prediction', lw=2)
	ax.legend()
	plt.show(block=True)

