from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
import os
from matplotlib.font_manager import FontProperties, findfont
from sklearn.metrics import root_mean_squared_error
import inspect, ast, textwrap, warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import math, time
from datetime import timedelta
from tqdm.auto import tqdm
from typing import IO, Callable, List
import xgboost as xgb
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.container import StemContainer
import matplotlib.colors as mcolors
import matplotlib; matplotlib.use('QtAgg')
_STARTING_SCORE = float('-inf')
SCORE_FUNCTION = root_mean_squared_error

def compose_dataset_splitter(split_units, test_size, gap_size):
	if split_units not in ['%','index','positions']:
		raise ValueError("Unrecognized key for split_units. Must be either '%' for a percent (0 to 100) of the total length of dataset, 'index' for the same units used for values in the dataset's DataFrame.index, or 'positions' for basic integer indexation from 0 to len(dataset).")
	if split_units == '%':
		def split_dataset(X,y, n_splits):
			TEST_SIZE = round(len(X) * test_size / 100.0)
			GAP_SIZE = round(len(X) * gap_size / 100.0)
			if n_splits > 1:
				tss = TimeSeriesSplit(n_splits=n_splits, test_size=TEST_SIZE, gap=GAP_SIZE)
				return tss.split(X,y)
			else:
				test_idx = range(len(X) - TEST_SIZE, len(X))
				train_idx = range(len(X) - TEST_SIZE - GAP_SIZE)
				return train_idx, test_idx
	if split_units == 'positions':
		def split_dataset(X,y, n_splits):
			TEST_SIZE = test_size
			GAP_SIZE = gap_size
			if n_splits > 1:
				tss = TimeSeriesSplit(n_splits=n_splits, test_size=TEST_SIZE, gap_size=GAP_SIZE)
				return tss.split(X,y)
			else:
				test_idx = range(len(X) - TEST_SIZE, len(X))
				train_idx = range(len(X) - TEST_SIZE - GAP_SIZE)
				return train_idx, test_idx
	if split_units == 'index':
		def split_dataset(X,y, n_splits):
			if not isinstance(X,pd.DataFrame):
				raise TypeError("If split_units = 'positions', X must be a DataFrame")
			t_end = X.index[-1]
			t_test_start = t_end - test_size
			t_gap_start = t_test_start - gap_size
			pos_test_start = X.index.searchsorted(t_test_start, side='right')
			pos_gap_start = X.index.searchsorted(t_gap_start, side='left')

			TEST_SIZE = len(X)-pos_test_start
			GAP_SIZE = pos_test_start-pos_gap_start
			if n_splits > 1:
				tss = TimeSeriesSplit(n_splits=n_splits, test_size=TEST_SIZE, gap_size=GAP_SIZE)
				return tss.split(X,y)
			else:
				test_idx = range(pos_test_start, len(X))
				train_idx = range(pos_test_start - pos_gap_start)
				return train_idx, test_idx
	return split_dataset

@dataclass
class PlotStyle:
	"""
	This class is meant to be used indirectly through the get_plotstlye function
	"""
	@staticmethod
	def compose_savefig_options(fname: str | PathLike | IO, format: str = '', **kwargs) -> dict:
		fname = str(fname)
		_, ext = os.path.splitext(fname)
		if ext: format = ext
		format = format.lstrip('.').lower()	# lower converts to lowercase
		fname = f"{fname}.{format}"
		return {'fname': fname, 'format': format, **kwargs}
	@staticmethod
	def compose_set_title_options(label: str, **kwargs):
		return {'label': label} | kwargs
	def settitle_and_savefig(fig: plt.Figure, ax: plt.Axes, savefig_options: dict = {}, set_title_options: dict = {}, savefig: bool = True, save_with_title: bool = False):
			if isinstance(ax, (list, np.ndarray)): ax = ax[0]
			set_title = lambda: ax.set_title(**set_title_options)
			if save_with_title: set_title()
			if savefig: fig.savefig(**savefig_options)
			if not save_with_title: set_title()
	layout: str = 'two-column'
	facecolor: str = '#fbfbfb'
	label_fontfamily = 'Times New Roman'
	time_unit_annotation: str = r' $[s]$'
	temperature_unit_annotation: str = r' $[°C]$'
	percent_unit_annotation: str = r' $[\%]$'
	time_unit_annotation: str = ' [s]'
	temperature_unit_annotation: str = ' [°C]'
	percent_unit_annotation: str = ' [%]'
	savefig_bbox_inches: str = 'tight'
	save_figure_extension: str = 'pdf' # This is used as the default extension, for when the filename does not provide one.
	save_with_title = False	# if True, the figure title is drawn before the figure is saved and will show on the saved file.

	# =================== Figure specific parameters ===================
	# figure specific strings that are initialized as empty
	# strings are to be set by the get_plotstlye function.
	M2_training_score_history_title: str = ''
	M2_training_score_history_xlabel: str = ''
	score_label: str = ''
	M2_training_score_history_filename: str = 'M2_training_score_history'
	M2_training_score_history_savefig: bool = False

	M2_partial_prediction_title: str = ''
	M2_partial_prediction_filename: str = 'M2_partial_prediction'
	M2_partial_prediction_savefig: bool = True

	M3_crossvalidation_title: str = ''
	M3_crossvalidation_filename: str = 'M3_crossvalidation'
	M3_crossvalidation_savefig: bool = False

	ylabel_temperature_diff: str = ''
	M3_partial_prediction_title: str = ''
	M3_partial_prediction_filename: str = 'M3_partial_prediction'
	M3_partial_prediction_savefig: bool = True

	full_dataset_title: str = ''
	full_dataset_filename: str = 'Dataset_full'
	full_dataset_savefig: bool = True

	clipped_dataset_title: str = ''
	clipped_dataset_filename: str = 'Dataset_clipped'
	clipped_dataset_savefig: bool = False

	CPU_load_feature_in_legend: str = ''
	first_temp_peak_detail_title: str = ''
	first_temp_peak_detail_filename: str = 'First_temp_peak_detail'
	first_temp_peak_detail_savefig: bool = True

	composite_prediction_title: str = ''
	composite_prediction_filename: str = 'composite_prediction'
	composite_prediction_savefig: bool = True

	def __post_init__(self):		# Layout specific parameters
		if self.layout == 'two-column':
			self.single_figsize = (3.6,2.7)	# matplotlib default is (6.4,4.8) inches
			self.multiple_figsize: callable = lambda n: (3.6, 2.6 * n)
			# fontsizes are given in pt. 	1 in = 72 pts
			self.label_fontsize: float = 10
			self.title_fontsize: float = 12
			self.annotate_fontsize: float = 8
			self.legend_fontsize: float = 7
			self.tick_label_fontsize: float = 7

			self.linewidth_thin: float = 0.5
			self.linewidth_medium: float = 1
			self.linewidth_thick: float = 1.5
			self.spine_linewidth: float = self.linewidth_thin
			self.prediction_plot_options: dict = {'lw':self.linewidth_thick, 'label':'Prediction', 'color':'C1'}
			self.reference_plot_options: dict = {'lw':self.linewidth_thin, 'label':'Reference', 'color':'C0'}
			self.grid_options: list = [
				{'which': 'major', 'color': '#e0e0e0', 'linewidth': 0.6},
				{'which': 'minor', 'color': '#f0f0f0', 'linewidth': 0.3, 'ls': '--'}
			]
		else:			# 'one-column'
			self.multiple_figsize: callable = lambda n: (6, 2.5 * n)
			self.label_fontsize: float = 12
			self.title_fontsize: float = 16
			self.annotate_fontsize: float = 10
			self.legend_fontsize: float = 10
			self.tick_label_fontsize: float = 10

			self.linewidth_thin: float = 0.5
			self.linewidth_medium: float = 1
			self.linewidth_thick: float = 1.5
			self.spine_linewidth: float = self.linewidth_thin
			self.prediction_plot_options: dict = {'lw':self.linewidth_thick, 'label':'Prediction', 'color':'C1'}
			self.reference_plot_options: dict = {'lw':self.linewidth_thin, 'label':'Reference', 'color':'C0'}
			self.grid_options: list = [
				{'which': 'major', 'color': '#e0e0e0', 'linewidth': 0.6},
				{'which': 'minor', 'color': '#f0f0f0', 'linewidth': 0.3, 'ls': '--'}
			]
	def localize(self, lang='en'):	# Localization for various strings
		self.prediction_plot_options['label'] = {'en': 'Prediction', 'pt': 'Predição'}[lang]
		self.reference_plot_options['label'] = {'en': 'Reference', 'pt': 'Referência'}[lang]
		self.M2_training_score_history_title = {'en': 'Score History', 'pt': 'Histórico do score'}[lang]
		self.M2_training_score_history_xlabel = {'en': 'Iteration', 'pt': 'Iteração'}[lang]
		self.xlabel_time = {'en': 'Time', 'pt': 'Tempo'}[lang] + self.time_unit_annotation
		self.ylabel_temperature = {'en': 'CPU Temperature', 'pt': 'Temperatura da CPU'}[lang] + self.temperature_unit_annotation 
		self.ylabel_temperature_diff = {'en': 'Temperature difference', 'pt': 'Diferença de temperatura'}[lang] + self.temperature_unit_annotation
		self.ylabel_cpu_load = {'en': 'CPU load', 'pt': 'Utilização da CPU'}[lang] + self.percent_unit_annotation
		self.M2_partial_prediction_title = {'en': 'M2 prediction', 'pt': 'Predição M2'}[lang]
		self.M3_partial_prediction_title = {'en': 'M3 prediction', 'pt': 'Predição M3'}[lang]
		self.M3_crossvalidation_title = {'en': 'M3 Cross-validation', 'pt': 'Validação cruzada M3'}[lang]
		self.full_dataset_title = {'en': 'Full length dataset','pt':'Dataset em toda duração'}[lang]
		self.clipped_dataset_title = {'en': 'Clipped length dataset','pt':'Dataset até o primeiro superaquecimento'}[lang]
		self.first_temp_peak_detail_title = {'en': 'Detailed view for detected high CPU segment','pt':'Detalhe do segmento de alta utilização de CPU'}[lang]
		self.CPU_load_feature_in_legend = {'en':'CPU load feature','pt':'Atributo da utilização da CPU'}[lang]
		self.composite_prediction_title = {'en': 'Composite prediction', 'pt': 'Predição composta'}[lang]
def get_plotstyle(publication):

	# Set publications specific parameter tweaks 
	match publication:
		case 'IEEE2025': 	
			ps = PlotStyle(layout='two-column')
			ps.localize(lang='en')
		case _:	# None of the above		
			ps = PlotStyle(layout='two-column')
			ps.localize(lang='en')

	ps.score_label = infer_score_label(SCORE_FUNCTION)

	return ps
def infer_score_label(score_fn):
	SCORING_LABELS = {
		'root_mean_squared_error': 'RMSE',
		'mean_absolute_error': 'MAE',
		'mean_squared_error': 'MSE',
		'mean_absolute_percentage_error': 'MAPE',
	}
	# check function name first
	func_name = getattr(score_fn, '__name__', None)
	if func_name:
		for key, label in sorted(SCORING_LABELS.items(), key=lambda x: -len(x[0])):
			if key.lower() == func_name.lower():
				return label
			if key.lower() in func_name.lower():
				return label
	source = inspect.getsource(score_fn)
	source_dedented = textwrap.dedent(source)
	tree = ast.parse(source_dedented)

	# check the functions source code
	for node in ast.walk(tree):
		if isinstance(node, ast.Call):
			if isinstance(node.func, ast.Name):
				name = node.func.id
			elif isinstance(node.func, ast.Attribute):
				name = node.func.attr
			else:
				continue
			for key, label in sorted(SCORING_LABELS.items(), key=lambda x: -len(x[0])):
				if key.lower() == name.lower():
					return label
				if key.lower() in name.lower():
					return label
	return "Score"
def is_valid_font(fontname):
    try:
        prop = FontProperties(family=fontname)
        fontpath = findfont(prop, fallback_to_default=False)
        return True
    except Exception:
        return False


class FunctionWrapper(ABC):
	_model: 'M1 | M2 | M3_XGBoost'		# type: ignore
	def __init__(self):
		"""
		Instantiate a wrapper object to provide a common ground access to fields belonging to 
		a model and its associated objects to the method specified in subclass definition.
		The model field is initiated as None and should be set at model instantiation.
		"""
		self._model = None
	def __call__(self):
		return


class DDS(BaseEstimator, RegressorMixin):
	def __init__(self, tel: pd.DataFrame, input_col: str, highcpudetector, M1, M2, M3):
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
	def get_model_params(self):
		"""
		:return: The parameters of this model.
		"""		
		return self._kernel._params
	def fit(self, X, y, plot = True):
		"""
		Orchestrates the training process by validating inputs and coordinating the kernel and optimizer.
		:param X: Input features.
		:param y: Target values.
		"""
		X, y = check_X_y(X, y)
		self._optimizer.fit(X, y, plot)
	def predict(self, X, plot = True, against=[]):
		"""
		Predicts the target values using the trained model.
		:param X: Input features.
		:param plot: Whether to plot the predictions against the true values.
		:param against: True values to plot against the predictions. Only used if plot is True.
		"""
		X = check_array(X, ensure_2d=False, dtype=float)
		if self._optimizer._training_finished:
			pred = self._kernel.predict(X)
			if plot: self._optimizer.plotter.plot_prediction(y_true=against, y_pred=pred)
			return pred
		else:
			warnings.warn("Model is not trained yet. Call fit() before predict().", UserWarning)
		return 
	def score(self, X, y):
		"""
		Evaluates the model by validating inputs and delegating to the optimizer's score method.
		:param X: Input features.
		:param y: Target values.
		:return: Model score.
		"""
		X, y = check_X_y(X, y)
		return self._optimizer.score(X, y)

	def __init__(self, kernel: 'M2Kernel', optimizer: 'M2Optimizer', random_state = None, plotstyle: PlotStyle = None):
		"""
		:param source: Iterable containing the input data to be fed to the model.
		:param target: Iterable containing the output data to which the model must be fitted.
		:param kernel: Function, as a FunctionWrapper object, containing logic and calculations that will produce predictions.
		:param loss: Loss function, as a FunctionWrapper object, to be used in training the model.
		:param optimizer: Optimizer function, as a FunctionWrapper object, to be used in training the model.
		"""
		self._kernel = kernel;			self._kernel._model = self
		self._optimizer = optimizer;	self._optimizer._model = self
		
		self.plotstyle = plotstyle
		# Hyperparamaters
		self.random_state = random_state
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
	@dataclass
	class ParamSpace:
		KCPU: tuple
		KTemp: tuple
		TauCPU: tuple
		TauTemp: tuple
	def _params_domain_restrict(self,params: Params) -> Params:
		KCPU = math.sqrt(params.KCPU**2+self._param_space.KCPU[0]**2)
		KTemp = math.sqrt(params.KTemp**2+self._param_space.KTemp[0]**2)
		TauCPU = np.clip(params.TauCPU, *self._param_space.TauCPU)
		TauTemp = np.clip(params.TauTemp, *self._param_space.TauTemp)
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
		params = self._params_domain_restrict(params)
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
			KCPU = np.random.uniform(*self._param_space.KCPU),
			KTemp = np.random.uniform(*self._param_space.KTemp),
			TauCPU = np.random.uniform(*self._param_space.TauCPU),
			TauTemp = np.random.uniform(*self._param_space.TauTemp),
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
	def predict(self, X) -> List:
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

	def __init__(self, FCPU: callable, FTEMP: callable, TempAmb: float, Dt:float,  params: Params=None, param_space: ParamSpace = None, noise_level=0):	# Dt test value was Dt = 30/510.0
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
		if param_space is None: self._param_space = self.ParamSpace(KCPU = (0.00001, 1), KTemp=(0.00001, 0.01), TauCPU=(1e-9,2), TauTemp=(1e-9,2))
		else: self._param_space = param_space
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
		def __init__(self, data_source: Callable[[], List[float]], best_model_getter: callable = None, plotstyle: PlotStyle = None):
			self.data_source = data_source
			self.best_model_getter = best_model_getter
			self.plotstyle = plotstyle

		def plot_training(self):
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
			data = self.data_source()
			fig, ax = plt.subplots(1, figsize = PS.single_figsize)
			x = range(len(data))
			ax.plot(x, data)
			ax.set_xlabel(PS.M2_training_score_history_xlabel, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			ax.set_ylabel(PS.score_label, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			ax.tick_params(axis='both', labelsize=PS.tick_label_fontsize)
			for grid_option in PS.grid_options: ax.grid(**grid_option)
			if PS.spine_linewidth is not None:
				for spine in ax.spines.values(): spine.set_linewidth(PS.spine_linewidth)
			ax.set_facecolor(PS.facecolor)
			PlotStyle.settitle_and_savefig(fig, ax,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.M2_training_score_history_filename, 
					format=PS.save_figure_extension, 
					bbox_inches='tight'
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.M2_training_score_history_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.M2_training_score_history_savefig,
				save_with_title=PS.save_with_title
			)
			plt.show(block=True)
		def plot_prediction(self, y_true, y_pred, **kwargs):
			"""
			Plot predictions vs reference for a single fit (full dataset).
			"""
			if len(y_true) != len(y_pred):
				warnings.warn("y_true and y_pred must have the same length.")
				return
			import matplotlib.pyplot as plt
			def sanitize_plotstyle(ps: PlotStyle) -> PlotStyle:
				if ps is None: ps = get_plotstyle('')
				if not isinstance(ps.reference_plot_options, dict):
					ps.reference_plot_options = {}
				if not isinstance(ps.prediction_plot_options, dict):
					ps.prediction_plot_options = {}
				if not mcolors.is_color_like(ps.facecolor):
					ps.facecolor = None
				if not isinstance(ps.grid_options, list):
					ps.grid_options = []
				if not (isinstance(ps.legend_fontsize, (int,float)) and ps.legend_fontsize > 0):
					ps.legend_fontsize = None
				if not (isinstance(ps.tick_label_fontsize, (int,float)) and ps.tick_label_fontsize > 0): 
					ps.tick_label_fontsize = None
				if not (isinstance(ps.xlabel_time, str)):
					ps.xlabel_time = ''
				if not (isinstance(ps.ylabel_temperature, str)):
					ps.ylabel_temperature = ''
				if not (isinstance(ps.M2_partial_prediction_title, str)):
					ps.M2_partial_prediction_title = ''
				if not (isinstance(ps.M2_partial_prediction_filename, str)): 
					ps.M2_partial_prediction_filename = 'M2_partial_prediction'
				if not (isinstance(ps.save_figure_extension, str)): 
					ps.save_figure_extension = 'svg'
				if not (isinstance(ps.savefig_bbox_inches, str)): 
					ps.savefig_bbox_inches = 'tight'
				if not (is_valid_font(ps.label_fontfamily)): 
					ps.label_fontfamily = None
				if not isinstance(ps.single_figsize,tuple) or len(ps.single_figsize) != 2 or any([not isinstance(dim,(float,int)) or dim <=0 for dim in ps.single_figsize]):
					ps.single_figsize = None
				if not (isinstance(ps.spine_linewidth, (int,float)) and ps.spine_linewidth > 0): 
					ps.spine_linewidth = None
				return ps
			PS = sanitize_plotstyle(self.plotstyle)

			fig, ax = plt.subplots(1, figsize=PS.single_figsize)
			if isinstance(y_true, pd.Series): x = y_true.index
			else: x = range(len(y_true))
			ax.plot(x, y_true, **PS.reference_plot_options)		# Reference line
			ax.plot(x, y_pred, **PS.prediction_plot_options)	# Prediction line
			ax.set_facecolor(PS.facecolor)
			for grid_option in PS.grid_options: ax.grid(**grid_option)
			ax.set_xlabel(PS.xlabel_time, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			ax.set_ylabel(PS.ylabel_temperature, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			ax.legend(fontsize=PS.legend_fontsize)
			ax.tick_params(axis='both', labelsize=PS.tick_label_fontsize)
			ax.annotate(PS.score_label+f' = {SCORE_FUNCTION(y_true,y_pred):0.4f}', xy=(0.99,0.04), xycoords='axes fraction',
				fontsize=PS.annotate_fontsize, horizontalalignment='right', verticalalignment='bottom')
			if PS.spine_linewidth is not None:
				for spine in ax.spines.values(): spine.set_linewidth(PS.spine_linewidth)
			PlotStyle.settitle_and_savefig(fig, ax,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.M2_partial_prediction_filename, 
					format=PS.save_figure_extension, 
					bbox_inches=PS.savefig_bbox_inches
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.M2_partial_prediction_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.M2_partial_prediction_savefig,
				save_with_title=PS.save_with_title
			)
			plt.show(block=True)

	class StopConditions:
		GLOBAL_MAX_ITERATIONS 		= 1 << 0
		GLOBAL_MIN_LOSS      		= 1 << 1
		GLOBAL_MAX_DURATION 		= 1 << 2
		STALE_PATH_AVG_GRADIENT  	= 1 << 3
		STALE_PATH_MAX_ITER			= 1 << 4

		@classmethod
		def compute_flags(cls, obj: 'M2Optimizer'):
			code = 0
			if obj.current_iteration == obj.max_iter:												code |= cls.GLOBAL_MAX_ITERATIONS
			if abs(obj._gradient_window[-1]) <= abs(obj.global_min_loss):								code |= cls.GLOBAL_MIN_LOSS
			if abs(obj._loss_gradient()) <= obj.avg_gradient_tol:									code |= cls.STALE_PATH_AVG_GRADIENT
			if obj.iterations_on_this_path == obj.max_iter_on_one_path:							code |= cls.STALE_PATH_MAX_ITER
			if (time.time() - obj._training_start_time) >= obj.training_duration.total_seconds():	code |= cls.GLOBAL_MAX_DURATION
			return code
		@classmethod
		def compose_training_stop_function(cls, selected_flags, composition) -> callable:
			if not cls.validate_flags(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def training_stop(obj: 'M2Optimizer'):
				raised_flags = cls.compute_flags(obj)
				watched_flags = selected_flags & (cls.GLOBAL_MAX_ITERATIONS | cls.GLOBAL_MIN_LOSS | cls.GLOBAL_MAX_DURATION)
				if		composition == 'any':
					cond = bool(raised_flags & watched_flags)
				else: # composition == 'all'
					other = watched_flags & ~cls.GLOBAL_MAX_ITERATIONS
					cond = bool((raised_flags & cls.GLOBAL_MAX_ITERATIONS) or ((raised_flags & other) == other))
				if cond:
					obj._training_stop_reason = cls._describe_flags(raised_flags)
					return raised_flags
				return 0
			return training_stop
		@classmethod
		def compose_stale_path_function(cls, selected_flags, composition) -> callable:
			if not cls.validate_flags(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def stale_path(obj: 'M2Optimizer'):
				raised_flags = cls.compute_flags(obj)
				watched_flags = selected_flags & (cls.STALE_PATH_AVG_GRADIENT | cls.STALE_PATH_MAX_ITER)
				if		composition == 'any':
					cond = bool(raised_flags & watched_flags)
				else: # composition == 'all'
					cond = ((raised_flags & watched_flags) == watched_flags)
				if cond:
					obj._stale_path_reason = cls._describe_flags(raised_flags)
					return raised_flags
				return 0
			return stale_path
		@classmethod
		def validate_flags(cls, stop_flags: int):
			attributes = inspect.getmembers(cls, lambda a: isinstance(a, int))
			max_value = sum(v for _, v in attributes)
			if stop_flags < 0 or stop_flags == 0 or stop_flags > max_value:
				raise ValueError("Invalid stop flags.")
			return True
		@classmethod
		def _describe_flags(cls, flags):
			flag_description = []
			if flags & cls.GLOBAL_MAX_ITERATIONS: 	flag_description.append("GLOBAL_MAX_ITERATIONS")
			if flags & cls.GLOBAL_MIN_LOSS:       	flag_description.append("GLOBAL_MIN_LOSS")
			if flags & cls.GLOBAL_MAX_DURATION:     flag_description.append("GLOBAL_MAX_DURATION")
			return ", ".join(flag_description)		
	def _reset_score_buffer(self):
		self._score_buffer = [_STARTING_SCORE] * self.max_iter
	def _reset_history(self):
		self._gradient_window = [(1.0 + i)*self.global_min_loss*10 for i in reversed(range(self.gradient_window_size))]
		self._reset_score_buffer()
		self.score_history = []
	def _set_best_model(self):
		self._best_model_params = self._model._kernel.get_model_params()
		self._best_score = self.current_score
		self._best_training_iter = self.current_iteration
	def get_best_model(self):
		return self._best_model_params, self._best_score, self._best_training_iter
	def fit(self, X, y, plot = True):
		"""
		Fits the model by iterating over M2Kernel parameters.
		:param X: Input features.
		:param y: Target values.
		:return: self
		"""
		self.current_iteration = 0
		self.buffer_index = 0
		self.iterations_on_this_path = 0
		self._reset_history()
		self._set_best_model()

		# Initialize progress bar and progress functions. Feed training time across iterations to the progress function
		pbar, progress = self._start_pbar();			last = time.time(); self._training_start_time = last

		while not self.training_stop_condition(self):
			now = time.time()
			
			if self.buffer_index == self.max_iter: self._flush_score_buffer() 
			try:		# calculate a prediction, compute score and update loss and score history
				with np.errstate(over='raise', invalid='raise'): 	self._update_score(X, y)
			except (ValueError, FloatingPointError):  # default to a new guessed model parameters
				self._model._kernel.set_model_parameters(self._model._kernel.guess_params())
				self.iterations_on_this_path = 0; continue
			
			# keep track of the best model found
			if self.current_score > self._best_score: 	self._set_best_model()

			# calculate params for next iteration
			if self.stale_path_condition(self): 		next_params = self._model._kernel.guess_params(); self.iterations_on_this_path = 0
			else: next_params = self.optimize(self._model._kernel.get_model_params())

			self._model._kernel.set_model_parameters(next_params)
			self.current_iteration += 1
			self.buffer_index += 1
			self.iterations_on_this_path += 1
			self._update_pbar(pbar, progress(now,last)); last = now
		self._flush_score_buffer()
		pbar.close()
		if plot: self.plotter.plot_training()
		self._model._kernel.set_model_parameters(self.get_best_model()[0])
		print("[INFO] Stop conditions met: ", self._training_stop_reason)
		print("[INFO] At iteration {iter:d}, got the best score of {score:.3f} with model params\n{model}".format(
			aux := self.get_best_model(),
			model = aux[0],
			score = aux[1],
			iter = aux[2],
		))
		self._training_finished = True
		return self
	def score(self, X, y):
		"""
		Custom scoring method using RMSELoss.
		:param X: Input features.
		:param y: Target values.
		:return: Negative RMSE (to align with sklearn's maximization convention).
		"""
		self._current_prediction = self._model._kernel.predict(X)
		rmse = SCORE_FUNCTION(y, self._current_prediction)
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
	def _update_score(self, X, y):
		self.current_score = self.score(X, y)
		self._gradient_window = (self._gradient_window + [self.current_score])[-self.gradient_window_size:]
		self._score_buffer[self.buffer_index] = self.current_score
	def _loss_gradient(self):
		return (self._gradient_window[-1] - self._gradient_window[0]) / self.gradient_window_size if self.gradient_window_size > 1 else self._gradient_window[0]
	def get_score_history(self):
		return np.concatenate(self.score_history + [self._score_buffer[:self.buffer_index]]).tolist()
	def _flush_score_buffer(self):		
		self.buffer_index = 0
		self.score_history.append(self._score_buffer.copy())
		self._reset_score_buffer()
		return

	def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, gradient_window=5, global_min_loss=1, score_tol=100, training_duration: timedelta = timedelta(minutes=1), \
				composition='any',
				training_stop_flags: int = StopConditions.GLOBAL_MAX_ITERATIONS | StopConditions.GLOBAL_MIN_LOSS,
				stale_path_flags: int = StopConditions.STALE_PATH_MAX_ITER | StopConditions.STALE_PATH_AVG_GRADIENT,
				plotstyle: PlotStyle = None
				):
		"""
		Initializes the Optimizer with a learning rate, maximum iterations, and tolerance.
		:param learning_rate: Step size for parameter updates.
		:param max_iter: Maximum number of iterations for training.
		:param tol: Tolerance for stopping criteria based on loss improvement.
		"""
		super().__init__()
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.global_min_loss = global_min_loss
		self.training_duration = training_duration
		self._training_start_time = None
		self.score_tol = score_tol
		self.current_iteration = 0
		self.current_score = _STARTING_SCORE
		self.score_history = []
		self._gradient_window=[]
		self._score_buffer=[]
		self._current_prediction=[]
		self._best_prediction=[]
		

		SC = self.StopConditions
		if not SC.validate_flags(training_stop_flags | stale_path_flags): raise ValueError("Failed to validate StopConditions flags.")
		if not (training_stop_flags & (SC.GLOBAL_MAX_ITERATIONS | SC.GLOBAL_MAX_DURATION)): training_stop_flags |= SC.GLOBAL_MAX_ITERATIONS
		self.training_stop_condition = SC.compose_training_stop_function(training_stop_flags,composition)
		self.stale_path_condition = SC.compose_stale_path_function(stale_path_flags,'any')
		self._training_stop_reason = None
		self._stale_path_reason = None

		# Set up progress bar
		pbar_opts = {'leave': True, 'desc': 'M2 fitting in progress'}
		self._max_duration_flag_set = bool(training_stop_flags & SC.GLOBAL_MAX_DURATION)
		self._pbar_iters = 0
		if self._max_duration_flag_set:
			total_sec = self.training_duration.total_seconds()
			if total_sec >= (denominator:=3600): unit = 'h'
			elif total_sec >= 2*(denominator:=60): unit = 'min'
			else: unit, denominator = 's', 1
			def _start_progress_bar():
				self._pbar_iters = 0
				progress_func = (lambda now, last: now - last) 
				return tqdm(**(pbar_opts | {'total': total_sec/denominator, 'unit': unit, 'bar_format': '{l_bar}{bar}| {n:.2f}/{total:.2f} {unit} | {postfix}'})), progress_func
			def _update_progress_bar(pbar: tqdm, progress: float):
				self._pbar_iters = getattr(self, '_pbar_iters', 0) + 1
				elapsed = (time.time() - self._training_start_time)
				it_per_s = self._pbar_iters / elapsed if elapsed > 0 else 0
				pbar.update(progress/denominator)
				pbar.set_postfix({'it': self._pbar_iters, 'it/s': f'{it_per_s:.2f}'})
				# pbar.update(progress/denominator)
				# pbar.set_postfix({'it/s': f'{1/progress:.2f}'})
		else:
			def _start_progress_bar():
				progress_func = (lambda now, last: 1)
				return tqdm(**(pbar_opts | {'total': self.max_iter})), progress_func
			def _update_progress_bar(pbar: tqdm, progress: float):
				pbar.update(progress)
		self._start_pbar, self._update_pbar = _start_progress_bar, _update_progress_bar
		self.iterations_on_this_path = 0
		self.max_iter_on_one_path = 100
		self.gradient_window_size = gradient_window
		self.avg_gradient_tol = tol

		self._best_model_params = None
		self._best_training_iter = 0
		self._best_score = _STARTING_SCORE

		self._training_finished = False
		self.plotter=self.Plotter(
			data_source=self.get_score_history,
			plotstyle=plotstyle
			)
	def get_params(self, deep=True):
		params = {
			'learning_rate': self.learning_rate,
			'max_iter': self.max_iter,
			'tol': self.avg_gradient_tol,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self

class M3Strategy(ABC, BaseEstimator, RegressorMixin):
	@abstractmethod
	def prefit(self, X, y, **kwargs): pass
	@abstractmethod
	def fit(self, X, y, **kwargs): pass
	@abstractmethod
	def predict(self, X, **kwargs): pass
	@abstractmethod
	def get_params(self, deep=True): pass
	@abstractmethod
	def set_params(self, **params): pass
	def __init__(self, **kwargs):
		self.prefit_context: dict = {'from_xval': False}
class XGBStrategy(M3Strategy):
	def prefit(self, X, y, **kwargs):
		train_idx = self.prefit_context.get('train_idx', None)
		test_idx = self.prefit_context.get('test_idx', None)
		X = self.prefit_context.get('X', [])
		y = self.prefit_context.get('y', [])
		if train_idx is None or test_idx is None:
			raise ValueError("train_idx and test_idx cannot be None. XGBoost requires defining an eval_set.")
		if len(X)==0 or len(y)==0:
			raise ValueError("X and y cannot be empty.")
		if len(X) != len(y):
			raise ValueError("X and y must have the same length.")
		if not isinstance(X, (pd.DataFrame, pd.Series)):
			X_train, X_test  = X[train_idx], X[test_idx]
		else:
			X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		if not isinstance(y, (pd.DataFrame, pd.Series)):
			y_train, y_test = y[train_idx], y[test_idx]
		else:
			y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
		return {
			'eval_set':	[(X_train, y_train), (X_test, y_test)],
			}
	def fit(self, X, y, **kwargs):
		eval_set = kwargs.get('eval_set', None)
		if eval_set is None: raise ValueError("eval_set cannot be None. XGBoost requires defining an eval_set.")
		verbose = bool(kwargs.get('verbose', False))
		self.reg.fit(X, y, eval_set=eval_set, verbose=verbose)
		return self
	def predict(self, X, **kwargs):
		return self.reg.predict(X, **kwargs)

	def __init__(self, n_estimators=1000, early_stopping_rounds=20, learning_rate=0.001, **kwargs):
		super().__init__(**kwargs)
		self.n_estimators = n_estimators
		self.early_stopping_rounds = early_stopping_rounds
		self.learning_rate = learning_rate
		self.reg = xgb.XGBRegressor(
			n_estimators=self.n_estimators,
			early_stopping_rounds=self.early_stopping_rounds,
			learning_rate=self.learning_rate,
			**kwargs
		)
	def get_params(self, deep=True):
		return self.reg.get_params(deep=deep)
	def set_params(self, **params):
		self.reg.set_params(**params)
		return self
class M3(BaseEstimator, RegressorMixin):
	class Plotter:
		def plot_crossvalidation(self, y_tests, y_preds, scores, fold_indices):
			def sanitize_plotstyle(ps: PlotStyle) -> PlotStyle:
				if ps is None: ps = get_plotstyle('')
				def _is_valid_figsize_callable(obj):
					try:
						if not callable(obj): return False
						result = obj(1)
						return (
							isinstance(result, tuple)
							and len(result) == 2
							and all(isinstance(x, (int, float)) for x in result)
						)
					except Exception:
						return False
				if not _is_valid_figsize_callable(ps.multiple_figsize):
					ps.multiple_figsize = None
				if not isinstance(ps.reference_plot_options, dict):
					ps.reference_plot_options = {}
				if not isinstance(ps.prediction_plot_options, dict):
					ps.prediction_plot_options = {}
				if not mcolors.is_color_like(ps.facecolor):
					ps.facecolor = None
				if not isinstance(ps.grid_options, list):
					ps.grid_options = []
				if not (isinstance(ps.annotate_fontsize, (int,float)) and ps.annotate_fontsize > 0):
					ps.annotate_fontsize = None
				if not (isinstance(ps.legend_fontsize, (int,float)) and ps.legend_fontsize > 0):
					ps.legend_fontsize = None
				if not (isinstance(ps.tick_label_fontsize, (int,float)) and ps.tick_label_fontsize > 0): 
					ps.tick_label_fontsize = None
				if not (isinstance(ps.M3_crossvalidation_filename, str)): 
					ps.M3_crossvalidation_filename = 'M3_crossvalidation'
				if not (isinstance(ps.M3_crossvalidation_title, str)): 
					ps.M3_crossvalidation_title = ''
				if not (isinstance(ps.save_figure_extension, str)): 
					ps.save_figure_extension = 'svg'
				if not (isinstance(ps.savefig_bbox_inches, str)): 
					ps.savefig_bbox_inches = 'tight'
				if not (is_valid_font(ps.label_fontfamily)): 
					ps.label_fontfamily = None
				if not isinstance(ps.single_figsize,tuple) or len(ps.single_figsize) != 2 or any([not isinstance(dim,(float,int)) or dim <=0 for dim in ps.single_figsize]):
					ps.single_figsize = None
				if not (isinstance(ps.spine_linewidth, (int,float)) and ps.spine_linewidth > 0): 
					ps.spine_linewidth = None
				return ps
			PS = sanitize_plotstyle(self.plotstyle)
			n_folds = len(fold_indices)
			fig, ax = plt.subplots(n_folds, figsize=PS.multiple_figsize(n_folds) )
			if n_folds == 1: ax = [ax]
			for ff in fold_indices:
				ax[ff].plot(y_tests[ff], **PS.reference_plot_options)	# Reference line
				ax[ff].plot(y_preds[ff], **PS.prediction_plot_options)	# Prediction line
				ax[ff].set_facecolor(PS.facecolor)
				for grid_option in PS.grid_options: ax[ff].grid(**grid_option)
				ax[ff].annotate(f'Fold {ff+1}', xy=(0.01,0.04), xycoords='axes fraction',
					fontsize=PS.annotate_fontsize, horizontalalignment='left', verticalalignment='bottom')
				ax[ff].annotate(PS.score_label+f' = {scores[ff]:0.4f}', xy=(0.99,0.04), xycoords='axes fraction',
					fontsize=PS.annotate_fontsize, horizontalalignment='right', verticalalignment='bottom')
				ax[ff].legend(loc='lower center', fontsize=PS.legend_fontsize)
				ax[ff].tick_params(axis='both', labelsize=PS.tick_label_fontsize)
				if PS.spine_linewidth is not None:
					for spine in ax[ff].spines.values(): spine.set_linewidth(PS.spine_linewidth)
			# prints a shared y axis label
			fig.text(0.05, 0.5, PS.ylabel_temperature_diff, va='center', rotation='vertical', fontsize = PS.label_fontsize, fontname=PS.label_fontfamily)
			ax[-1].set_xlabel(PS.xlabel_time, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			PlotStyle.settitle_and_savefig(fig, ax,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.M3_crossvalidation_filename, 
					format=PS.save_figure_extension, 
					bbox_inches=PS.savefig_bbox_inches
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.M3_crossvalidation_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.M3_crossvalidation_savefig,
				save_with_title=PS.save_with_title
			)
			plt.show(block=True)
		def plot_prediction(self, y_true, y_pred, **kwargs):
			"""
			Plot predictions vs reference for a single fit (full dataset).
			"""
			if y_true is not None and len(y_true) != len(y_pred):
				warnings.warn("y_true and y_pred must have the same length.")
				return
			def sanitize_plotstyle(ps: PlotStyle) -> PlotStyle:
				if ps is None: ps = get_plotstyle('')
				if not isinstance(ps.reference_plot_options, dict):
					ps.reference_plot_options = {}
				if not isinstance(ps.prediction_plot_options, dict):
					ps.prediction_plot_options = {}
				if not mcolors.is_color_like(ps.facecolor):
					ps.facecolor = None
				if not isinstance(ps.grid_options, list):
					ps.grid_options = []
				if not (isinstance(ps.legend_fontsize, (int,float)) or ps.legend_fontsize <= 0):
					ps.legend_fontsize = None
				if not (isinstance(ps.tick_label_fontsize, (int,float)) and ps.tick_label_fontsize > 0): 
					ps.tick_label_fontsize = None
				if not (isinstance(ps.xlabel_time, str)):
					ps.xlabel_time = ''
				if not (isinstance(ps.ylabel_temperature, str)):
					ps.ylabel_temperature = ''
				if not (isinstance(ps.M3_partial_prediction_title, str)):
					ps.M3_partial_prediction_title = ''
				if not (isinstance(ps.ylabel_temperature_diff, str)):
					ps.M3_partial_prediction_title = ''
				if not (isinstance(ps.M3_partial_prediction_filename, str)): 
					ps.M3_partial_prediction_filename = 'M3_partial_prediction'
				if not (isinstance(ps.save_figure_extension, str)): 
					ps.save_figure_extension = 'svg'
				if not (isinstance(ps.savefig_bbox_inches, str)): 
					ps.savefig_bbox_inches = 'tight'
				if not (is_valid_font(ps.label_fontfamily)): 
					ps.label_fontfamily = None
				if not isinstance(ps.single_figsize,tuple) or len(ps.single_figsize) != 2 or any([not isinstance(dim,(float,int)) or dim <=0 for dim in ps.single_figsize]):
					ps.single_figsize = None
				if not (isinstance(ps.spine_linewidth, (int,float)) and ps.spine_linewidth > 0): 
					ps.spine_linewidth = None
				return ps
			PS = sanitize_plotstyle(self.plotstyle)

			fig, ax = plt.subplots(1, figsize=PS.single_figsize)
			if isinstance(y_true, pd.Series): x = y_true.index
			else: x = range(len(y_true))
			if y_true is not None: ax.plot(x, y_true, **PS.reference_plot_options)		# Reference line
			ax.plot(x, y_pred, **PS.prediction_plot_options)	# Prediction line
			ax.set_facecolor(PS.facecolor)
			for grid_option in PS.grid_options: ax.grid(**grid_option)
			ax.set_xlabel(PS.xlabel_time, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			ylabel = PS.ylabel_temperature_diff or PS.ylabel_temperature
			ax.set_ylabel(ylabel, fontsize=PS.label_fontsize, fontname=PS.label_fontfamily)
			ax.legend(fontsize=PS.legend_fontsize)
			ax.tick_params(axis='both', labelsize=PS.tick_label_fontsize)
			ax.annotate(PS.score_label+f' = {SCORE_FUNCTION(y_true,y_pred):0.4f}', xy=(0.99,0.04), xycoords='axes fraction',
				fontsize=PS.annotate_fontsize, horizontalalignment='right', verticalalignment='bottom')
			if PS.spine_linewidth is not None:
				for spine in ax.spines.values(): spine.set_linewidth(PS.spine_linewidth)
			PlotStyle.settitle_and_savefig(fig, ax,
				savefig_options=PlotStyle.compose_savefig_options(
					fname=PS.M3_partial_prediction_filename, 
					format=PS.save_figure_extension, 
					bbox_inches=PS.savefig_bbox_inches
				),
				set_title_options=PlotStyle.compose_set_title_options(
					label=PS.M3_partial_prediction_title, 
					fontsize=PS.title_fontsize,
					fontname=PS.label_fontfamily
				),
				savefig=PS.M3_partial_prediction_savefig,
				save_with_title=PS.save_with_title
			)
			plt.show(block=True)

		def __init__(self, plotstyle: PlotStyle = None):
			self.plotstyle = plotstyle

	def fit(self, X, y, plot=True, split_units='%', test_size=20, gap_size=0, **kwargs):
		# Plotting functionality is still not implemented
		if not self.strategy.prefit_context['from_xval']:
			train_idx, test_idx = compose_dataset_splitter(split_units,test_size,gap_size)(X, y, 1)
			self.strategy.prefit_context.update({
				'train_idx': train_idx,
				'test_idx': test_idx,
				'X': X,
				'y': y,
				})
		prefit_dict = self.strategy.prefit(X, y, **kwargs)
		self.strategy.fit(X, y, **(kwargs|prefit_dict))
		return self
	def predict(self, X, plot=True, against=None, **kwargs):
		y_pred = self.strategy.predict(X, **kwargs)
		if plot: self.plotter.plot_prediction(against, y_pred)
		return y_pred
	def cross_validation(self, X, y, plot=True, n_splits = 5, split_units = '%', test_size = 20, gap_size = 0, **kwargs):
		"""
		Perform cross-validation on the dataset using TimeSeriesSplit.
		:param X: Input features.
		:param y: Target values.
		:param plot: Whether to plot the results.
		:param n_splits: Number of splits for cross-validation.
		:param split_units: Units for split size, can be '%', 'index', or 'positions'.
		:param test_size: Size of the test set, interpreted according to split_units.
		:param gap_size: Size of the gap between train and test sets, interpreted according to split_units.
		:return: Tuple of test indices, predicted values, and scores for each fold.
		"""

		X, y = check_X_y(X, y)
		y = y.ravel()
		y_tests, y_preds, scores, test_pos = [], [], [], []
		self.strategy.prefit_context.update({
			'from_xval': True,
			'X': X,
			'y': y,
			})
		for train_idx, test_idx in compose_dataset_splitter(split_units,test_size,gap_size)(X, y, n_splits):
			self.strategy.prefit_context.update({
				'train_idx': train_idx,
				'test_idx': test_idx,
				})
			X_train, y_train = X[train_idx], y[train_idx]
			X_test, y_test = X[test_idx], y[test_idx]
			self.fit(X_train, y_train, plot=False, **kwargs)
			pred = self.predict(X_test, plot=False)
			score = SCORE_FUNCTION(pred, y_test)
			y_tests.append(y_test)
			y_preds.append(pred)
			scores.append(score)
			test_pos.append(test_idx)
		self.strategy.prefit_context.update({
			'from_xval': False,
			})
		if plot:
			self.plotter.plot_crossvalidation(y_tests, y_preds, scores, list(range(n_splits)))
		return test_idx, y_preds, scores

	def __init__(self, strategy: M3Strategy, plotstyle=None):
		self.strategy: M3Strategy = strategy
		self.plotter = self.Plotter(plotstyle=plotstyle)
	def get_params(self, deep=True):
		return self.strategy.get_params(deep=deep)
	def set_params(self, **params):
		self.strategy.set_params(**params)
		return self


if __name__ == "__main__":
	def SythesizeDataset():
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
				# df=df.rename(columns={'t_mod':'t'})
				return df.drop(columns=['t_mod'])
			

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

			temp_ext=df.iloc[0,idy]
			Dt=df.index[1]-df.index[0]
			for i in range(1,len(df)):
				temp_current=df.iloc[i-1,idy]
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
		return df
	df2 = SythesizeDataset()	# Synthesize a dataset


	# ==================  Define and instantiate model components
	m2obj=M2(
		M2Kernel(lambda cpu: cpu**2, lambda temp_current, temp_ext: temp_current-temp_ext, TempAmb=40, Dt=30/510,
			param_space=M2Kernel.ParamSpace(KCPU=(1e-5,10), KTemp=(1e-5,1), TauCPU=(1e-9,1), TauTemp=(1e-9,0.5)),
			# params=M2Kernel.Params(KCPU=4, KTemp= 0.1, TauCPU=0.1, TauTemp=0.05),
			# params=M2Kernel.Params(KCPU=5.354987897910978, KTemp= 2.8002979490801128, TauCPU=0.18629771646496662, TauTemp=0.9528960069986708),
			# params=M2Kernel.Params(KCPU=8.328657109382513, KTemp=0.5493883896628988, TauCPU=np.float64(0.19189461891598183), TauTemp=np.float64(0.33700592667911833))		# RMSE = 1.150
			# params=M2Kernel.Params(KCPU=3.2672845157080244, KTemp=0.27811782995435014, TauCPU=np.float64(0.06603239807557451), TauTemp=np.float64(0.18592294666152198))	# RMSE = 0.985
			params=M2Kernel.Params(KCPU=4.838110172690417, KTemp=0.6594080619235382, TauCPU=np.float64(0.12214871259156063), TauTemp=np.float64(0.4990802997780703))		# RMSE = 0.977
			),
		M2Optimizer(global_min_loss = 0.5, training_duration = timedelta(seconds=1), composition='any', 
			training_stop_flags = M2Optimizer.StopConditions.GLOBAL_MIN_LOSS 
								| M2Optimizer.StopConditions.GLOBAL_MAX_DURATION 
			),
		plotstyle=get_plotstyle('IEEE2025')
	)
	m2obj.fit(plot = 							(PP := True),
		X = df2['CPU'].to_frame(),
		y = df2['Temp'],
		)						# options
	m2pred=m2obj.predict(df2['CPU'].to_frame(),  plot = PP, against=df2['Temp'])

	m3_source_col = 'Temp'
	df3=df2.loc[:,df2.columns != m3_source_col].copy(deep=True)
	target_col = 'Temp_residue'
	df3.loc[:,target_col] = (df2.loc[:,m3_source_col].copy(deep=True)-m2pred).rename(target_col)

	m3obj = M3(
		XGBStrategy(n_estimators=1000, early_stopping_rounds=20, learning_rate=0.001),
		plotstyle=get_plotstyle('IEEE2025')
	)
	m3obj.cross_validation(plot = 	(PP := True),
		X = df3.loc[:,df3.columns != target_col],	
		y = df3.loc[:,target_col],					
		n_splits=3
		)					# options
	m3obj.fit(plot = 				(PP := True),
		X = df3.loc[:,df3.columns != target_col],	
		y = df3.loc[:,target_col],
		)
	m3obj.predict(df3.loc[:,df3.columns != target_col], plot = PP, against=df3[target_col])

