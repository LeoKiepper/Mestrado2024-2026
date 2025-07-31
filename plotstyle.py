import matplotlib.pyplot as plt
import os
from os import PathLike
from typing import IO
import numpy as np
import yaml
from parse import parse
from plotstyle_validators import *
from plotstyle_validators import UnkownValidator,ParseError


import warnings	# leave this import last for neat folding of the import block
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
	print(f"{message}")
warnings.showwarning = custom_showwarning

# Standardized strings used for yaml files and properties
ROLE_BASE = 'base'
ROLE_TEMPLATE = 'template'
ROLE_LAYOUT = 'layout'


PROP_STRING_SOURCE = 'source'
PROP_STRING_SOURCE_VALUE_FROM_LITERAL = 'literal'
PROP_STRING_SOURCE_VALUE_FROM_FIELD = 'field'

PROP_STRING_VALIDATION = 'validation'
PROP_STRING_VALIDATION_GRIDOPTIONS = 'gridoptions'

PROP_STRING_VALUE = 'value'
PROP_STRING_SUFFIX = 'suffix'
PROP_STRING_PREFIX = 'prefix'
AFFIX_TYPES = [PROP_STRING_SUFFIX, PROP_STRING_PREFIX]
AFFIX_KEY_FORMAT = "{key}__{affix}__"
PROP_STRING_LOCALIZABLE = 'localizable'
LOCALIZATION_ENGLISH = 'en'
LOCALIZATION_PROTUGUESE = 'pt'

def unpack_affix_key(key: str, pattern: str = AFFIX_KEY_FORMAT) -> tuple[str, str]:
	result = parse(pattern, key)
	if result:
		return result['key'], result['affix']
	raise ValueError(f"Key '{key}' does not match format '{pattern}'")

class InvalidSourceType(Exception):
	pass
class SourceFieldMissing(Exception):
	pass


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
	def _cull_redundant_fields(self):
		for key in self.__marked_for_delete__:
			if key in self.__params__: self.__params__.pop(key)
	def _yaml_parse(self, entry_file: str) -> dict:
		YAML_LOADING_ERROR_MSG = "Error when loading yaml file {path}"
		PARSE_ERROR_MSG = "Failed to parse key '{key}' with validator '{validator}': {error}"
		UNKNOWN_VALIDATOR_MSG = "Unknown validator"
		IGNORE_FIELD_MSG = "[INFO]: Error raised when parsing field '{key}'. Field was ignored and will not load onto the object.\n\t{error}."
		def _load_yaml(path):
			try:
				with open(path, 'r') as f:
					return yaml.safe_load(f)
			except yaml.YAMLError:
				raise yaml.YAMLError(YAML_LOADING_ERROR_MSG.format(path=path))
		def _dump_to_dict(handle: dict, dump_to: dict = {}, read_from: dict ={}):
			def _fetch_value(key: str, prop: dict):
				if not PROP_STRING_SOURCE in prop.keys(): raise SourceFieldMissing
				if prop[PROP_STRING_SOURCE] == PROP_STRING_SOURCE_VALUE_FROM_LITERAL: return prop[PROP_STRING_VALUE]
				elif prop[PROP_STRING_SOURCE] == PROP_STRING_SOURCE_VALUE_FROM_FIELD:
					if not prop[PROP_STRING_VALUE] in self.__marked_for_delete__:
						self.__marked_for_delete__.append(prop[PROP_STRING_VALUE])
					return (dump_to | read_from)[prop[PROP_STRING_VALUE]]
				else: raise InvalidSourceType
			def _parse_prop(key: str, prop: dict):
				try:
					validator = VALIDATORS.get(prop[PROP_STRING_VALIDATION])
					if not validator: raise UnkownValidator(UNKNOWN_VALIDATOR_MSG.format(validator=prop[PROP_STRING_VALIDATION], key=key))
					return validator.parse(_fetch_value(key, prop), **{
						CONTEXT_FIELD_PARSER_FUNC: lambda handle: _dump_to_dict(handle, dump_to={}, read_from=dump_to)
						})
				except Exception as e:
					raise ParseError(PARSE_ERROR_MSG.format(key=key, validator=prop[PROP_STRING_VALIDATION], error=e))
			def _register_localizable(key: str, prop: dict):
				if PROP_STRING_LOCALIZABLE in prop.keys() and prop[PROP_STRING_LOCALIZABLE]:
					self.__localization__[key] = _fetch_value(key,prop)
			def _register_affixable(key: str, prop: dict, affix_type: str):
				if affix_type in prop and prop[affix_type] != {}:
					new_key = AFFIX_KEY_FORMAT.format(key=key, affix=affix_type)
					self.__affix__.update({new_key: _parse_prop(key, prop[affix_type])})
					_register_localizable(new_key, prop[affix_type])				
			for key, prop, in handle.items():				
				# Treat recursive yaml parsing separately
				if prop[PROP_STRING_VALIDATION] == PROP_STRING_VALIDATION_YAML:
					try:
						path = _fetch_value(key, prop)	# First call self.resolve_yaml_path, then validate
						self.__file_stack__.append(path)						# for debugging
						path = self._resolve_yaml_path(path)
						dump_to.update(_dump_to_dict(_load_yaml(VALIDATORS[PROP_STRING_VALIDATION_YAML].parse(path)), dump_to=dump_to))
						self.__file_stack__ = self.__file_stack__[:-1]	# for debugging
					except Exception as e:
						warnings.warn(IGNORE_FIELD_MSG.format(error=e, key=key))
						continue
				else: # Treat all other validation types
					try:
						parsed_value = _parse_prop(key, prop)
						dump_to.update({key:parsed_value})
						self.__field_validation__.update({key:prop[PROP_STRING_VALIDATION]})
					except Exception as e:
						warnings.warn(IGNORE_FIELD_MSG.format(error=e, key=key))
						continue
					_register_localizable(key, prop)
					# check affixes
					for affix_type in AFFIX_TYPES:
						try:
							_register_affixable(key, prop, affix_type)
						except Exception as e:
							warnings.warn(IGNORE_FIELD_MSG.format(error=e, key=key+'.'+affix_type))
							continue
			return dump_to
		return _dump_to_dict(_load_yaml(entry_file))
	def _resolve_yaml_path(self, filename: str) -> str:
		for root, _, files in os.walk(self.CONFIGS_FOLDER):
			if filename in files:
				return os.path.normpath(os.path.join(root, filename))
		raise FileNotFoundError(f"YAML file '{filename}' not found in '{self.CONFIGS_FOLDER}' or its subdirectories.")

	def _apply_localization(self):
		for key, loc in self.__localization__.items():
			if key in self.__affix__.keys(): self.__affix__[key] = loc[self.language]
			else: self.__params__[key] = loc[self.language]
	def _apply_affixes(self):
		for field, value in self.__affix__.items():
			key, affix = unpack_affix_key(field)
			if affix == PROP_STRING_SUFFIX: self.__params__.update({key: self.__params__[key] + value})
			elif affix == PROP_STRING_PREFIX: self.__params__.update({key: value + self.__params__[key]})
			else: raise ValueError(f"Unknown affix type for key {key}")
	def __init__(self, *, yaml_file: str=None, yaml_list: list=None, master=None, language: str = LOCALIZATION_ENGLISH, configs_folder: str='plotstyle_configs', base_folder: str='', cull_redundant_fields: bool = True):
		if not VALIDATORS[PROP_STRING_VALIDATION_PATHSTR].validate(configs_folder): 
			raise TypeError("configs_folder is an invalid folder name.")
		self.CONFIGS_FOLDER = configs_folder
		if not VALIDATORS[PROP_STRING_VALIDATION_PATHSTR].validate(base_folder): 
			raise ValueError("base_folder is an invalid folder name.")
		self.BASE_FOLDER = base_folder
		self.language = language
		self.__field_validation__ = {}
		self.__localization__ = {}
		self.__affix__ = {}
		self.__marked_for_delete__ = []
		# Reporting variables for debugging purposes
		self.__file_stack__ = []


		if yaml_file is not None:
			self.__file_stack__.append(yaml_file) 
			yaml_file = self._resolve_yaml_path(yaml_file) if not os.path.isabs(yaml_file) else yaml_file
			if not VALIDATORS[PROP_STRING_VALIDATION_YAML].validate(yaml_file):
				raise ValueError(f"Invalid YAML file: {yaml_file}")
			
			self.__params__ = self._yaml_parse(yaml_file)

		# elif yaml_list is not None:
		# elif master is not None and isinstance(master, PlotStyle):

		self._apply_localization()
		self._apply_affixes()
		if cull_redundant_fields: self._cull_redundant_fields()
		# Load attributes from yaml file onto the object
		for key, value in self.__params__.items():
			setattr(self, key, value)



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
						PROP_STRING_SOURCE:PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
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
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: layout_filename
						},
					'label_fontsize': {
						PROP_STRING_VALIDATION: 'fontsize',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 10
						},
					'title_fontsize': {
						PROP_STRING_VALIDATION: 'fontsize',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
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
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: template_filename
						},
					'savefig_bbox_inches': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'tight'
						},
					'file_format': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'pdf'
						}
				}, f, default_flow_style=False)

		plot1_path = os.path.join(EXAMPLE_CONFIGS_FOLDER, EXAMPLE_BASE_FOLDER, 'plot1.yaml')
		if not os.path.exists(plot1_path):
			with open(plot1_path, 'w') as f:
				yaml.dump({
					ROLE_BASE: {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_YAML,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: base_filename
						},
					'line_label': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'Foo'
						},
					'ylabel': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'My Y Axis Label'
						},
					'title': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'This uses PS1'
						},
					'figsize': {
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_FIGSIZE,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_FIELD,
						PROP_STRING_VALUE: 'single_figsize'
						},
					'xlabel':{
						PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_STR,
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_LOCALIZABLE: True,
						PROP_STRING_VALUE: {
							LOCALIZATION_ENGLISH: 'My X Axis Label',
							LOCALIZATION_PROTUGUESE: 'Meu Label Eixo X',
						},
						PROP_STRING_SUFFIX: {
							PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_STR,
							PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
							PROP_STRING_VALUE: '[unlocalized suffix]'
						},
						PROP_STRING_PREFIX: {
							PROP_STRING_VALIDATION: PROP_STRING_VALIDATION_STR,
							PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
							PROP_STRING_LOCALIZABLE: True,
							PROP_STRING_VALUE: {
								LOCALIZATION_ENGLISH: '[localized prefix]',
								LOCALIZATION_PROTUGUESE: '[prefixo localizado]'
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
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: base_filename
						},
					'line_label': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'Foobar'
						},
					'xlabel': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'My X Axis Label'
						},
					'title': {
						PROP_STRING_VALIDATION: 'str',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: 'This uses PS2'
						},
					'figsize': {
						PROP_STRING_VALIDATION: 'figsize',
						PROP_STRING_SOURCE: PROP_STRING_SOURCE_VALUE_FROM_LITERAL,
						PROP_STRING_VALUE: "lambda n: (6, 2.5 * n)"
						}
				}, f, default_flow_style=False)
	generate_yaml_files()

	# using chain-referencing
	PS1 = PlotStyle(yaml_file='plot1.yaml', configs_folder=EXAMPLE_CONFIGS_FOLDER, base_folder=EXAMPLE_BASE_FOLDER, language=LOCALIZATION_ENGLISH)
	# PS2 = PlotStyle(yaml_file='plot2.yaml', configs_folder=EXAMPLE_CONFIGS_FOLDER, base_folder=EXAMPLE_BASE_FOLDER)
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