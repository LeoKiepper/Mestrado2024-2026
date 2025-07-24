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

# Standardized strings used for yaml files and properties
BASE_ROLE = 'base'
TEMPLATE_ROLE = 'template'
LAYOUT_ROLE = 'layout'

PROP_STRING_SOURCE = 'source'
PROP_STRING_SOURCE_VALUE_FROM_CONSTANT = 'constant'
PROP_STRING_SOURCE_VALUE_FROM_FIELD = 'field'

PROP_STRING_VALIDATION = 'validation'
PROP_STRING_VALIDATION_YAML = 'yaml'
PROP_STRING_VALIDATION_FIGSIZE = 'figsize'
PROP_STRING_VALUE = 'value'


class NotLambdaFigsize(Exception):
	pass
class NotTupleFigsize(Exception):
	pass
class RejectedExpression(Exception):
	pass


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
		if not _validate_tuple_figsize(self.single_figsize):
			self.single_figsize = None
	# =============== Object control fields (mutable) ===============
	field_types: dict = field(default_factory=dict)	# contains a dict with the type of each field for validation purposes.
	layout: str = field(default_factory=str) # contains the filename to the layout yaml.
	template: str = field(default_factory=str) # contains the filename to the template yaml.
	base: str = field(default_factory=str) # contains the filename to the base yaml.
	_CONFIGS_FOLDER: str = field(default_factory=str) # contains the folder name for the config files.
	_YAML_PATH: str = field(default_factory=str) # contains the full path to the yaml file.

	def _yaml_parse(self, entry_file: str) -> dict:
		def _load_yaml(path):
			with open(path, 'r') as f:
				return yaml.safe_load(f)
		def _dump_to_dict(handle: dict, params: dict = {}):
			def _fetch_value(prop):
				if prop[PROP_STRING_SOURCE] == PROP_STRING_SOURCE_VALUE_FROM_CONSTANT:
					return prop[PROP_STRING_VALUE]
				elif prop[PROP_STRING_SOURCE] == PROP_STRING_SOURCE_VALUE_FROM_FIELD:
					return params[prop[PROP_STRING_VALUE]]
				else:
					raise ValueError(f"Invalid source type: {prop[PROP_STRING_SOURCE]}")
			for key, prop, in handle.items():
				next_value = _fetch_value(prop)
				if prop[PROP_STRING_VALIDATION] == PROP_STRING_VALIDATION_YAML:
					next_value = self.resolve_yaml_path(next_value) if not os.path.isabs(next_value) else next_value
					if not validate_yaml(next_value): raise ValueError(f"Invalid YAML file: {next_value}")
					params.update(_dump_to_dict(_load_yaml(next_value), params))
				elif prop[PROP_STRING_VALIDATION] == PROP_STRING_VALIDATION_FIGSIZE:
					params.update({key: parse_figsize(next_value)})
				else:
					params.update({key: next_value})
			return params
		return _dump_to_dict(_load_yaml(entry_file))
	def resolve_yaml_path(self, filename: str) -> str:
		for root, _, files in os.walk(self._CONFIGS_FOLDER):
			if filename in files:
				return os.path.normpath(os.path.join(root, filename))
		raise FileNotFoundError(f"YAML file '{filename}' not found in '{self._CONFIGS_FOLDER}' or its subdirectories.")

	def __init__(self, *, yaml_file: str=None, yaml_list: list=None, master=None, configs_folder: str='plotstyle_configs', base_folder: str=''):
		if not validate_foldername(configs_folder): raise TypeError("configs_folder is an invalid folder name.")
		self._CONFIGS_FOLDER = configs_folder
		if not validate_foldername(base_folder): raise ValueError("base_folder is an invalid folder name.")
		self._BASE_FOLDER = base_folder

		if yaml_file is not None:
			yaml_file = self.resolve_yaml_path(yaml_file) if not os.path.isabs(yaml_file) else yaml_file
			if not validate_yaml(yaml_file): raise ValueError(f"Invalid YAML file: {yaml_file}")
			
			params = self._yaml_parse(yaml_file)

		for key, value in params.items():
			setattr(self, key, value)
		# elif yaml_list is not None:
		# elif master is not None and isinstance(master, PlotStyle):
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
def validate_yaml(yaml_file):
	if not isinstance(yaml_file, (str, PathLike)): return False
	if not os.path.isfile(yaml_file): return False
	try:
		with open(yaml_file, 'r') as f:
			yaml.safe_load(f)
		return True
	except yaml.YAMLError:
		return False
def validate_foldername(name: str) -> bool:
	if not isinstance(name, str) or not name.strip(): 	return False
	
	# Disallowed characters on Windows
	reserved_names = {
		"CON", "PRN", "AUX", "NUL",
		*(f"COM{i}" for i in range(1, 10)),
		*(f"LPT{i}" for i in range(1, 10))
	}
	if name.upper().split('.')[0] in reserved_names: 	return False
	if any(c in name for c in r'<>:"/\\|?*'): 			return False
	if any(ord(c) < 32 for c in name): 					return False
	if name.endswith(' ') or name.endswith('.'): 		return False
	
	# Disallowed characters on POSIX
	# POSIX systems: '/' is invalid in any path segment
	if '/' in name: return False
	return True
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

def _validate_tuple_figsize(figsize):
	if not isinstance(figsize, tuple): 			return False
	if len(figsize) != 2: 						return False
	for dim in figsize:
		if not isinstance(dim, (int, float)): 	return False
		if dim <= 0:							return False
	return True
def parse_figsize(expr):
	try:
		figsize = _safe_eval_lambda(expr)
		if _validate_tuple_figsize(figsize(1)): return figsize
	except NotLambdaFigsize: pass
	try:
		try: figsize = eval(expr) if isinstance(expr, str) else expr
		except Exception: raise NotTupleFigsize
		if not _validate_tuple_figsize(figsize): raise NotTupleFigsize
		return figsize
	except NotTupleFigsize: raise ValueError(f"Invalid figsize expression: {expr}")
def _is_math_expr_safe(expr, allowed_names):
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
	if isinstance(expr, str) and expr.strip().startswith("lambda"): 
		if 'lambda' in expr[len("lambda"):-1].lower(): raise RejectedExpression('Lambda expressions are accepted if "lambda" appears only in the beginning of the string.')
	else:
		raise NotLambdaFigsize
	if expr.count(":") != 1: raise ValueError("Lambda expressions must contain exactly one ':' to separate arguments from return.")
	params = expr[len('lambda '):expr.index(":")].strip()  # removes "lambda " and gets the part before ":"
	params = [p.strip() for p in params.split(",")]
	dims = expr.split(":", 1)[1].strip()
	if not dims[0] == '(' and dims[-1] == ')': raise ValueError(invalid_tuple_error_string:="The expression must return a tuple of length 2.")
	dims = [dim.strip() for dim in dims[1:-1].split(",")]
	if len(dims) != 2: raise ValueError(invalid_tuple_error_string)

	for dim in dims:
		if not _is_math_expr_safe(dim, allowed_names=params): raise ValueError(f"Unsafe expression in dimension: {dim.strip()}")
	return eval(expr, {"__builtins__": {}}, {})

if __name__ == "__main__":
	EXAMPLE_CONFIGS_FOLDER = 'example_plotstyle_configs'
	EXAMPLE_BASE_FOLDER = 'figs'
	def generate_yaml_files():
		os.makedirs(os.path.join(EXAMPLE_CONFIGS_FOLDER, EXAMPLE_BASE_FOLDER), exist_ok=True)
		layout_filename = LAYOUT_ROLE + '.yaml'
		layout_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, layout_filename)
		if not os.path.exists(layout_path):
			with open(layout_path, 'w') as f:
				yaml.dump({
					'single_figsize': {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_FIGSIZE,
						PROP_STRING_SOURCE:PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: '(3.6, 2.7)'
						},
				}, f, default_flow_style=False)
		template_filename = TEMPLATE_ROLE + '.yaml'
		template_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, template_filename)
		if not os.path.exists(template_path):
			with open(template_path, 'w') as f:
				yaml.dump({
					LAYOUT_ROLE: {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_YAML,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: layout_filename
						},
					'label_fontsize': {
						PROP_STRING_VALIDATION: 'fontsize',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 10
						},
					'title_fontsize': {
						PROP_STRING_VALIDATION: 'fontsize',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 12
						}
				}, f, default_flow_style=False)
		base_filename = BASE_ROLE + '.yaml'
		base_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, base_filename)
		if not os.path.exists(base_path):
			with open(base_path, 'w') as f:
				yaml.dump({
					TEMPLATE_ROLE: {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_YAML,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: template_filename
						},
					'savefig_bbox_inches': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'tight'
						},
					'file_format': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'pdf'
						}
				}, f, default_flow_style=False)
		plot1_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, EXAMPLE_BASE_FOLDER, 'plot1.yaml')
		if not os.path.exists(plot1_path):
			with open(plot1_path, 'w') as f:
				yaml.dump({
					BASE_ROLE: {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_YAML,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: base_filename
						},
					'line_label': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'Foo'
						},
					'xlabel': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'My X Axis Label'
						},
					'title': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'This uses PS1'
						},
					'figsize': {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_FIGSIZE,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_FIELD,
						PROP_STRING_VALUE: 'single_figsize'
						}
				}, f, default_flow_style=False)
		plot2_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, EXAMPLE_BASE_FOLDER, 'plot2.yaml')
		if not os.path.exists(plot2_path):
			with open(plot2_path, 'w') as f:
				yaml.dump({
					BASE_ROLE: {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_YAML,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: base_filename
						},
					'line_label': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'Foobar'
						},
					'xlabel': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'My X Axis Label'
						},
					'title': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'This uses PS2'
						},
					'figsize': {
						PROP_STRING_VALIDATION: 'figsize',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: "lambda n: (6, 2.5 * n)"
						}
				}, f, default_flow_style=False)
	generate_yaml_files()

	# using chain-referencing
	PS1 = PlotStyle(yaml_file='plot1.yaml', configs_folder=EXAMPLE_CONFIGS_FOLDER, base_folder=EXAMPLE_BASE_FOLDER)
	PS2 = PlotStyle(yaml_file='plot2.yaml', configs_folder=EXAMPLE_CONFIGS_FOLDER, base_folder=EXAMPLE_BASE_FOLDER)
	def plot1(ps: PlotStyle=PS1):
		# fig, ax = plt.subplots(N:=1,figsize=ps.figsize)			# Use with PS1
		fig, ax = plt.subplots(N:=1,figsize=ps.figsize(1))		# Use with PS2
		ax.plot([0, 1], [0, 1], label=ps.line_label)
		ax.set_title(ps.title, fontsize=ps.title_fontsize)
		ax.set_xlabel(ps.xlabel)
		ax.legend()
		plt.show()
	plot1(ps=PS2)