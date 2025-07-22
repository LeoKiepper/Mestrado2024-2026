from dataclasses import dataclass, field
from matplotlib.font_manager import FontProperties, findfont
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
from os import PathLike
from typing import IO
import numpy as np
import yaml
import warnings
import ast

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

	def sanitize(self):
		"""
			Validates paramaters for self and attributes default values when not valid (generally None).

			:raises ValueError: If filename is set to an empty string and it cannot be inferred from other parameters.
		"""
			# :param extraparams: links each new parameter with a type.
			# :type extraparams: dict
		if not isinstance(self.score_history_title, str): 
			self.score_history_title = ''
		if not isinstance(self.score_history_xaxis_label, str): 
			self.score_history_xaxis_label = ''
		if not isinstance(self.yaxis_label, str): 
			self.yaxis_label = ''
		if not isinstance(self.M2_training_score_history_filename, str): 
			self.M2_training_score_history_filename = self.yaxis_label
			if self.M2_training_score_history_filename == '': raise ValueError('PlotStyle.filename must be set to a non-empty string.')
		if not isinstance(self.file_format, str): 
			self.file_format = 'pdf'
		if not isinstance(self.savefig_bbox_inches, str): 
			self.savefig_bbox_inches = 'tight'
		if not validate_fontfamily(self.label_fontfamily): 
			self.label_fontfamily = None
		if not validate_fontsize(self.label_fontsize): 
			self.label_fontsize = None
		if not validate_fontsize(self.title_fontsize): 
			self.title_fontsize = None
		if not validate_fontsize(self.tick_label_fontsize):
			self.tick_label_fontsize = None
		if not validate_linewidth(self.spine_linewidth): 
			self.spine_linewidth = None
		if not mcolors.is_color_like(self.facecolor):
			self.facecolor = None
		if not _validate_gridoptions(self.grid_options): 
			self.grid_options = []
		if not _validate_figsize(self.single_figsize):
			self.single_figsize = None
	# =============== Object control fields ===============
	field_types: dict = field(default_factory=dict)	# contains a dict with the type of each field for validation purposes.
	master: 'PlotStyle' = field(default = None, repr=False) # contains a reference to the master PlotStyle object, if any.

	# =============== General parameters ================
	layout: str = field(default = 'two-column', repr=False)

	plotarea_facecolor: str = '#fbfbfb'
	fontfamily: str = ''
	label_fontfamily: str = ''
	title_fontfamily: str = ''
	score_history_title: str = ''
	score_history_xaxis_label: str = ''
	yaxis_label: str = ''

	filename: str = ''
	file_format: str = 'pdf'		# This is a fallback extension, for when the filename does not provide one.
	save_with_title = False	# if True, the figure title is drawn before the figure is saved and will show on the saved file.
	savefig_bbox_inches: str = 'tight'

	time_unit_annotation: str = r' $[s]$'
	temperature_unit_annotation: str = r' $[°C]$'
	percent_unit_annotation: str = r' $[\%]$'
	
	# =================== Figure specific parameters ===================
	# figure specific strings that are initialized as empty
	# strings are to be set by the get_plotstlye function.
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
		self.prediction_plot_options['label'] = {
			'en': 'Prediction', 
			'pt': 'Predição'
			}[lang]
		self.reference_plot_options['label'] = {
			'en': 'Reference',
			'pt': 'Referência'
			}[lang]
		self.score_history_title = {
			'en': 'Score History',
			'pt': 'Histórico do score'
			}[lang]
		self.score_history_xaxis_label = {
			'en': 'Iteration',
			'pt': 'Iteração'
			}[lang]
		self.xlabel_time = {
			'en': 'Time',
			'pt': 'Tempo'
			}[lang] + self.time_unit_annotation
		self.ylabel_temperature = {
			'en': 'CPU Temperature',
			'pt': 'Temperatura da CPU'
			}[lang] + self.temperature_unit_annotation 
		self.ylabel_temperature_diff = {
			'en': 'Temperature difference',
			'pt': 'Diferença de temperatura'
			}[lang] + self.temperature_unit_annotation
		self.ylabel_cpu_load = {
			'en': 'CPU load',
			'pt': 'Utilização da CPU'
			}[lang] + self.percent_unit_annotation
		self.M2_partial_prediction_title = {
			'en': 'M2 prediction',
			'pt': 'Predição M2'
			}[lang]
		self.M3_partial_prediction_title = {
			'en': 'M3 prediction',
			'pt': 'Predição M3'
			}[lang]
		self.M3_crossvalidation_title = {
			'en': 'M3 Cross-validation',
			'pt': 'Validação cruzada M3'
			}[lang]
		self.full_dataset_title = {
			'en': 'Full length dataset',
			'pt': 'Dataset em duração cheia'
			}[lang]
		self.clipped_dataset_title = {
			'en': 'Clipped length dataset',
			'pt': 'Dataset até o primeiro superaquecimento'
			}[lang]
		self.first_temp_peak_detail_title = {
			'en': 'Detailed view for detected high CPU segment',
			'pt': 'Detalhe do segmento de alta utilização de CPU'
			}[lang]
		self.CPU_load_feature_in_legend = {
			'en': 'CPU load feature',
			'pt': 'Atributo da utilização da CPU'
			}[lang]
		self.composite_prediction_title = {
			'en': 'Composite prediction',
			'pt': 'Predição composta'
			}[lang]
def get_plotstyle(publication: str=None, master: PlotStyle=None, params: str='') -> PlotStyle:
	if isinstance(publication, str):
		# Set publications specific parameter tweaks 
		match publication:
			case 'IEEE2025': 	
				ps = PlotStyle(layout='two-column')
				ps.localize(lang='en')
			case _:	
				ps = PlotStyle(layout='two-column')
				ps.localize(lang='en')
		return ps
	
	if isinstance(master, PlotStyle):
		ps = PlotStyle(layout=master.layout)
		ps.master=master
		WARN_STRING = lambda key: f"get_plotstyle: Ignoring invalid entry for key '{key}'"
		with open(params) as f:
			params: dict = yaml.safe_load(f)
		for key, entry in params.items():
			if not isinstance(entry, dict) or not "type" in entry or not "value" in entry:
				warnings.warn(WARN_STRING(key))
				continue
			if entry["type"] == "figsize":
				tt = _infer_figsize_type(entry["value"])
				if  tt == 'callable':
					val = _safe_eval_lambda(entry["value"])
				elif tt == 'tuple':
					val = tuple(eval(entry["value"]))
				elif tt == '': 
					warnings.warn(WARN_STRING(key))
					continue
				setattr(ps, key, val)
			else:
				setattr(ps, key, entry["value"])
			ps.field_types[key] = entry["type"]
		return ps
def validate_fontfamily(fontname):
    try:
        prop = FontProperties(family=fontname)
        fontpath = findfont(prop, fallback_to_default=False)
        return True
    except Exception:
        return False
def validate_fontsize(fontsize):
	if isinstance(fontsize, (int, float)):
		return fontsize > 0
	if isinstance(fontsize, str):
		return fontsize in ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
	return fontsize == None
def validate_linewidth(linewidth):
	if isinstance(linewidth, (int, float)):
		return linewidth > 0
	return linewidth == None
def _validate_gridoptions(grid_options):
	if not isinstance(grid_options, list): return False
	for opt in grid_options:
		if not isinstance(opt, dict): return False
		if ('visible' in opt) 	and (not isinstance(opt['visible'], bool)): 		return False
		if ('which' in opt) 	and (not opt['which'] in ['major','minor','both']): return False
		if ('axis' in opt) 		and (not opt['axis'] in ['x','y','both']): 			return False
		if ('color'in opt) 		and (not mcolors.is_color_like(opt['color'])): 		return False
		if ('linewidth' in opt) and (not validate_linewidth(opt['linewidth'])): 	return False
	return grid_options == []
def _validate_figsize(figsize):
	if figsize is callable:
		try: figsize = figsize(1)
		except: return False
	if not isinstance(figsize, tuple): 			return False
	if len(figsize) != 2: 						return False
	for dim in figsize:
		if not isinstance(dim, (int, float)): 	return False
		if dim <= 0:							return False
	return figsize == None

def _infer_figsize_type(figsize):
	if isinstance(figsize, tuple) and len(figsize) == 2:
		if all(isinstance(dim, (int, float)) for dim in figsize):
			return 'tuple'
	else:
		try:
			_safe_eval_lambda(figsize)
			return 'callable'
		except ValueError:
			return ''
def _is_safe_math_expr(expr, allowed_names):
	try:
		node = ast.parse(expr, mode='eval')
		for n in ast.walk(node):
			if isinstance(n, ast.Name):
				if n.id not in allowed_names:
					return False
			elif isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Constant, ast.Expression, ast.Load)):
				continue
			elif isinstance(n, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub)):
				continue
			else:
				# Disallow function calls, attribute access, etc.
				return False
		return True
	except Exception:
		return False
def _safe_eval_lambda(expr: str):
	if not expr.strip().startswith("lambda"):
		raise ValueError("Only lambda expressions are allowed.")
	if expr.count(":") != 1:
		raise ValueError("The expression argument must contain exactly one ':' to separate parameters from the return value.")
	params = expr[len('lambda '):expr.index(":")].strip()  # removes "lambda " and gets the part before ":"
	params = [p.strip() for p in params.split(",")]
	dims = expr.split(":", 1)[1].strip()
	if not dims[0] == '(' and dims[-1] == ')':
		raise ValueError("The expression must return a tuple.")
	dims = [dim.strip() for dim in dims[1:-1].split(",")]
	if len(dims) != 2:
		raise ValueError("The expression must return a tuple of length 2.")
	for dim in dims:
		if not _is_safe_math_expr(dim, allowed_names=params):
			raise ValueError(f"Unsafe expression in dimension: {dim.strip()}")
	return eval(expr, {"__builtins__": {}}, {})

def __getattr__(self, name):
	if self.master is not None:
		return getattr(self.master, name)
	raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

if __name__ == "__main__":
	PS = get_plotstyle(publication='IEEE2025')
	PS1 = get_plotstyle(master=PS, params='plot1.yaml')
	PS2 = get_plotstyle(master=PS1, params='plot2.yaml')
	def plot1(ps: PlotStyle=PS1):
		fig, ax = plt.subplots(N:=1,figsize=ps.figsize(N))
		ax.plot([0, 1], [0, 1], label=ps.linelabel)
		ax.set_title(ps.score_history_title)
		ax.set_xlabel(ps.yaxis_label)
		ax.set_ylabel(ps.xlabel)
		ax.legend()
		plt.show()
	plot1(ps=PS2)