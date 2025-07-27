from dataclasses import dataclass, field
from matplotlib.font_manager import FontProperties, findfont
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import os
from os import PathLike
from typing import IO
import numpy as np
import yaml
import warnings
import ast
import plotstyle
from parse import parse

# Standardized strings used for yaml files and properties
ROLE_BASE = 'base'
ROLE_TEMPLATE = 'template'
ROLE_LAYOUT = 'layout'


PROP_STRING_SOURCE = 'source'
PROP_STRING_SOURCE_VALUE_FROM_CONSTANT = 'constant'
PROP_STRING_SOURCE_VALUE_FROM_FIELD = 'field'


PROP_STRING_VALIDATION = 'validation'
PROP_STRING_VALIDATION_YAML = 'yaml'
PROP_STRING_VALIDATION_STR = 'str'
PROP_STRING_VALIDATION_BOOL = 'bool'
PROP_STRING_VALIDATION_COLOR = 'color'
PROP_STRING_VALIDATION_FIGSIZE = 'figsize'
PROP_STRING_VALIDATION_FONTSIZE = 'fontsize'
PROP_STRING_VALIDATION_LINEWIDTH = 'linewidth'
PROP_STRING_VALIDATION_FONTFAMILY = 'fontfamily'
PROP_STRING_VALIDATION_FILEFORMAT = 'fileformat'
PROP_STRING_VALIDATION_PATHSTR = 'pathstr'
PROP_STRING_VALIDATION_FILENAME = 'filename'
PROP_STRING_VALIDATION_GRIDOPTIONS = 'gridoptions'

PROP_STRING_VALUE = 'value'
PROP_STRING_SUFFIX = 'suffix'
PROP_STRING_PREFIX = 'prefix'
AFFIX_KEY_FORMAT = "{key}__{affix}__"
PROP_STRING_LOCALIZABLE = 'localizable'
PROP_STRING_LOCALIZATION_ENGLISH = 'en'
PROP_STRING_LOCALIZATION_PROTUGUESE = 'pt'

def unpack_affix_key(key: str, pattern: str = AFFIX_KEY_FORMAT) -> tuple[str, str]:
	result = parse(pattern, key)
	if result:
		return result['key'], result['affix']
	raise ValueError(f"Key '{key}' does not match format '{pattern}'")

class NotLambdaFigsize(Exception):
	pass
class NotTupleFigsize(Exception):
	pass
class RejectedExpression(Exception):
	pass
class InvalidSourceType(Exception):
	pass
class SourceFieldMissing(Exception):
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
		if not validate_gridoptions(self.grid_options): 
			self.grid_options = []
		if not _validate_tuple_figsize(self.single_figsize):
			self.single_figsize = None
	# =============== Object control fields (mutable) ===============
	field_validation: dict = field(default_factory=dict)	# contains a dict with the type of each field for validation purposes.
	localization: dict = field(default_factory=dict)	# contains a dict pointing to fields that are localizable.
	affix: dict = field(default_factory=dict)			# contains a dict with affixes to apply to specified fields.
	_CONFIGS_FOLDER: str = field(default_factory=str) # contains the folder name for the config files.
	_YAML_PATH: str = field(default_factory=str) # contains the full path to the yaml file.

	def _yaml_parse(self, entry_file: str) -> dict:
		def _load_yaml(path):
			with open(path, 'r') as f:
				return yaml.safe_load(f)
		def _dump_to_dict(handle: dict, params: dict = {}):
			def _fetch_value(prop: dict):
				if not PROP_STRING_SOURCE in prop.keys(): raise SourceFieldMissing
				if prop[PROP_STRING_SOURCE] == PROP_STRING_SOURCE_VALUE_FROM_CONSTANT: return prop[PROP_STRING_VALUE]
				elif prop[PROP_STRING_SOURCE] == PROP_STRING_SOURCE_VALUE_FROM_FIELD: return params[prop[PROP_STRING_VALUE]]
				else: raise InvalidSourceType
			for key, prop, in handle.items():
				try: next_value = _fetch_value(prop)
				except SourceFieldMissing:
					warnings.warn(f"[INFO]: Missing source field for key '{key}'. Ignoring.")
					continue
				except InvalidSourceType:
					warnings.warn(f"[INFO]: Unrecognized source type '{prop[PROP_STRING_SOURCE]}' for key '{key}'. Ignoring.")
					continue

				match prop[PROP_STRING_VALIDATION]:
					case plotstyle.PROP_STRING_VALIDATION_YAML:
						next_value = self.resolve_yaml_path(next_value) if not os.path.isabs(next_value) else next_value
						if not validate_yaml(next_value): raise ValueError(f"Invalid YAML file: {next_value}")
						dump = lambda: params.update(_dump_to_dict(_load_yaml(next_value), params))
					case plotstyle.PROP_STRING_VALIDATION_FIGSIZE:
						dump = lambda: params.update({key: parse_figsize(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_FONTSIZE:
						dump = lambda: params.update({key: parse_fontsize(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_LINEWIDTH:
						dump = lambda: params.update({key: parse_linewidth(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_FONTFAMILY:
						dump = lambda: params.update({key: parse_fontfamily(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_STR:
						dump = lambda: params.update({key: parse_str(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_BOOL:
						dump = lambda: params.update({key: parse_bool(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_FILEFORMAT:
						dump = lambda: params.update({key: parse_fileformat(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_PATHSTR | plotstyle.PROP_STRING_VALIDATION_FILENAME:
						dump = lambda: params.update({key: parse_pathstr(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_COLOR:
						dump = lambda: params.update({key: parse_color(next_value)})
					case plotstyle.PROP_STRING_VALIDATION_GRIDOPTIONS:
						dump = lambda: params.update({key: parse_gridoptions(next_value)})
					case _:
						warnings.warn(f"[INFO]: Unrecognized validation type '{prop[PROP_STRING_VALIDATION]}' for key '{key}'. Ignoring.")
						continue

				def _register_localizable(prop: dict, keyname: str):
					if PROP_STRING_LOCALIZABLE in prop.keys() and prop[PROP_STRING_LOCALIZABLE]:
						self.localization[keyname] = _fetch_value(prop)
				def _register_affixable(prop: dict, keyname: str, affix_type: str):					
					if affix_type in prop and prop[affix_type] != {}:
						new_key = AFFIX_KEY_FORMAT.format(key=keyname, affix=affix_type)
						self.affix.update({new_key: _fetch_value(prop[affix_type])})
						_register_localizable(prop[affix_type], new_key)
				_register_localizable(prop, key)
				_register_affixable(prop, key, PROP_STRING_SUFFIX)
				_register_affixable(prop, key, PROP_STRING_PREFIX)

				try:
					dump()
					self.field_validation[key] = prop[PROP_STRING_VALIDATION]
				except Exception as e:
					warnings.warn(f"[INFO]: Error {e} raised when parsing value for key '{key}'. Ignoring this key.")
					continue
				
			return params
		return _dump_to_dict(_load_yaml(entry_file))
	def resolve_yaml_path(self, filename: str) -> str:
		for root, _, files in os.walk(self._CONFIGS_FOLDER):
			if filename in files:
				return os.path.normpath(os.path.join(root, filename))
		raise FileNotFoundError(f"YAML file '{filename}' not found in '{self._CONFIGS_FOLDER}' or its subdirectories.")

	def apply_localization(self):
		for key, loc in self.localization.items():
			if key in self.affix.keys(): self.affix[key] = loc[self.language]
			else: self.param[key] = loc[self.language]
	def apply_affixes(self):
		for field, value in self.affix.items():
			key, affix = unpack_affix_key(field)
			if affix == PROP_STRING_SUFFIX: self.param.update({key: self.param[key] + value})
			elif affix == PROP_STRING_PREFIX: self.param.update({key: value + self.param[key]})
			else: raise ValueError(f"Unknown affix type for key {key}")

			if field in self.localization.keys(): self.localization.pop(field)
	def __init__(self, *, yaml_file: str=None, yaml_list: list=None, master=None, language: str = PROP_STRING_LOCALIZATION_ENGLISH, configs_folder: str='plotstyle_configs', base_folder: str=''):
		if not validate_pathstr(configs_folder): raise TypeError("configs_folder is an invalid folder name.")
		self._CONFIGS_FOLDER = configs_folder
		if not validate_pathstr(base_folder): raise ValueError("base_folder is an invalid folder name.")
		self._BASE_FOLDER = base_folder
		self.language = language
		self.field_validation = {}
		self.localization = {}
		self.affix = {}

		if yaml_file is not None:
			yaml_file = self.resolve_yaml_path(yaml_file) if not os.path.isabs(yaml_file) else yaml_file
			if not validate_yaml(yaml_file): raise ValueError(f"Invalid YAML file: {yaml_file}")
			
			self.param = self._yaml_parse(yaml_file)

		# elif yaml_list is not None:
		# elif master is not None and isinstance(master, PlotStyle):

		self.apply_localization()
		self.apply_affixes()
		# Load attributes from yaml file onto the object
		for key, value in self.param.items():
			setattr(self, key, value)

def validate_gridoptions(options):
	if isinstance(options, list):
		if not all(isinstance(opt, dict) for opt in options): return False
		if not all('which' in opt for opt in options): return False
	return True
def validate_color(color):
	return mcolors.is_color_like(color)
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
def validate_fileformat(format):
    canvas = FigureCanvasAgg(Figure())
    return format in canvas.get_supported_filetypes()
def validate_yaml(yaml_file):
	if not isinstance(yaml_file, (str, PathLike)): return False
	if not os.path.isfile(yaml_file): return False
	try:
		with open(yaml_file, 'r') as f:
			yaml.safe_load(f)
		return True
	except yaml.YAMLError:
		return False
def validate_pathstr(path_str: str) -> bool:
	if not isinstance(path_str, str) or not path_str.strip():
		return False
	
	norm = os.path.normpath(path_str)
	drive, tail = os.path.splitdrive(norm)
	segments = tail.split(os.sep)
	for seg in segments:
		if not seg: continue
		if seg in (os.curdir, os.pardir): continue

		_RESERVED_NAMES = {
			"CON", "PRN", "AUX", "NUL",
			*(f"COM{i}" for i in range(1, 10)),
			*(f"LPT{i}" for i in range(1, 10))
		}
		if seg.upper().split('.')[0] in _RESERVED_NAMES: 	return False
		if any(c in set(r'<>:"/\\|?*') for c in seg): 		return False
		if any(ord(c) < 32 for c in seg): 					return False
		if seg.endswith(' ') or seg.endswith('.'): 			return False
	return True
def validate_str(expr):
	if isinstance(expr, str): return True
	if isinstance(expr, dict):
		if all(isinstance(loc, str) for loc in expr.values()):
			return True
	return False
def validate_gridoptions(grid_options):
	if not isinstance(grid_options, list): return False
	for opt in grid_options:
		if not isinstance(opt, dict): return False
		if ('visible' in opt) 	and (not isinstance(opt['visible'], bool)): 		return False
		if ('which' in opt) 	and (not opt['which'] in ['major','minor','both']): return False
		if ('axis' in opt) 		and (not opt['axis'] in ['x','y','both']): 			return False
		if ('color'in opt) 		and (not mcolors.is_color_like(opt['color'])): 		return False
		if ('linewidth' in opt) and (not validate_linewidth(opt['linewidth'])): 	return False
	return grid_options == []
def validate_bool(expr):
	return isinstance(expr, bool)
def _validate_tuple_figsize(figsize):
	if not isinstance(figsize, tuple): 			return False
	if len(figsize) != 2: 						return False
	for dim in figsize:
		if not isinstance(dim, (int, float)): 	return False
		if dim <= 0:							return False
	return True
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
def parse_figsize(expr):
	try:
		figsize_candidate = _safe_eval_lambda(expr)
		if _validate_tuple_figsize(figsize_candidate(1)): return figsize_candidate
	except NotLambdaFigsize: pass
	try:
		try: figsize_candidate = eval(expr) if isinstance(expr, str) else expr
		except Exception: raise ValueError(f"Invalid eval expression: {expr}")
		if not _validate_tuple_figsize(figsize_candidate): raise NotTupleFigsize
		return figsize_candidate
	except NotTupleFigsize: raise ValueError(f"Invalid figsize: {expr}")
def parse_fontsize(expr):
	if validate_fontsize(expr): return expr
	else: raise ValueError(f"Invalid fontsize: {expr}")
def parse_linewidth(expr):
	if validate_linewidth(expr): return expr
	else: raise ValueError(f"Invalid linewidth: {expr}")
def parse_fontfamily(expr):
	if validate_fontfamily(expr): return expr
	else: raise ValueError(f"Invalid fontfamily: {expr}")
def parse_fileformat(expr):
	if validate_fileformat(expr): return expr
	else: raise ValueError(f"Unsopported file format: {expr}")
def parse_str(expr):
	if validate_str(expr): return expr
	else: raise ValueError(f"Not a string: {expr}")
def parse_bool(expr):
	if validate_bool(expr): return expr
	raise ValueError(f"Invalid boolean value: {expr}")
def parse_pathstr(expr):
	if validate_pathstr(expr): return expr
	else: raise ValueError(f"Invalid path string: {expr}")
def parse_color(expr):
	if validate_color(expr): return expr
	else: raise ValueError(f"Invalid color: {expr}")
def parse_gridoptions(expr):
	if validate_gridoptions(expr): return expr
	else: raise ValueError(f"Invalid grid options: {expr}")




if __name__ == "__main__":
	EXAMPLE_CONFIGS_FOLDER = 'example_plotstyle_configs'
	EXAMPLE_BASE_FOLDER = 'figs'
	def generate_yaml_files():
		os.makedirs(os.path.join(EXAMPLE_CONFIGS_FOLDER, EXAMPLE_BASE_FOLDER), exist_ok=True)
		layout_filename = ROLE_LAYOUT + '.yaml'
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
		template_filename = ROLE_TEMPLATE + '.yaml'
		template_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, template_filename)
		if not os.path.exists(template_path):
			with open(template_path, 'w') as f:
				yaml.dump({
					ROLE_LAYOUT: {
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
		base_filename = ROLE_BASE + '.yaml'
		base_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, base_filename)
		if not os.path.exists(base_path):
			with open(base_path, 'w') as f:
				yaml.dump({
					ROLE_TEMPLATE: {
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
					ROLE_BASE: {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_YAML,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: base_filename
						},
					'line_label': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'Foo'
						},
					'ylabel': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_VALUE: 'My Y Axis Label'
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
						},
					'xlabel':{
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_STR,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
						PROP_STRING_LOCALIZABLE: True,
						PROP_STRING_VALUE: {
							PROP_STRING_LOCALIZATION_ENGLISH: 'My X Axis Label',
							PROP_STRING_LOCALIZATION_PROTUGUESE: 'Meu Label Eixo X',
						},
						PROP_STRING_SUFFIX: {
							PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_STR,
							PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
							PROP_STRING_VALUE: '[unlocalized suffix]'
						},
						PROP_STRING_PREFIX: {
							PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_STR,
							PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_CONSTANT,
							PROP_STRING_LOCALIZABLE: True,
							PROP_STRING_VALUE: {
								PROP_STRING_LOCALIZATION_ENGLISH: '[localized prefix]',
								PROP_STRING_LOCALIZATION_PROTUGUESE: '[prefixo localizado]'
								},
							}
						}
				}, f, default_flow_style=False)
		plot2_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, EXAMPLE_BASE_FOLDER, 'plot2.yaml')
		if not os.path.exists(plot2_path):
			with open(plot2_path, 'w') as f:
				yaml.dump({
					ROLE_BASE: {
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
	PS1 = PlotStyle(yaml_file='plot1.yaml', configs_folder=EXAMPLE_CONFIGS_FOLDER, base_folder=EXAMPLE_BASE_FOLDER, language=PROP_STRING_LOCALIZATION_ENGLISH)
	PS2 = PlotStyle(yaml_file='plot2.yaml', configs_folder=EXAMPLE_CONFIGS_FOLDER, base_folder=EXAMPLE_BASE_FOLDER)
	def plot1(ps: PlotStyle=PS1):
		fig, ax = plt.subplots(N:=1,figsize=ps.figsize)			# Use with PS1
		# fig, ax = plt.subplots(N:=1,figsize=ps.figsize(1))		# Use with PS2
		ax.plot([0, 1], [0, 1], label=ps.line_label)
		ax.set_title(ps.title, fontsize=ps.title_fontsize)
		ax.set_ylabel(ps.ylabel)
		ax.set_xlabel(ps.xlabel)
		ax.legend()
		plt.show()
	plot1(ps=PS1)