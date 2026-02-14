import matplotlib.pyplot as plt
import os
from os import PathLike
from typing import IO, Any, List
import numpy as np
import yaml
from plotstyle_validators import *
from dataclasses import dataclass
import plotstyle_interface as PSIF
import warnings	# leave this import last for neat folding of the import block
PSIF.configure_tags()
warnings.showwarning = PSIF.custom_showwarning

YAML_FILE_NOT_FOUND_MSG = "YAML file '{filename}' not found in '{configs_folder}' or its subdirectories."
TYPE_ERROR_MSG = "'configs' must either be a list of dicts or a dict"
@dataclass
class _ParseContext:
	explicitly_kept: set
	explicitly_delete: set
	marked_for_delete: set
	field_props: dict
	affix: dict
	file_stack: list 	# for debugging
	def __init__(self):
		self.explicitly_kept = set()
		self.explicitly_deleted = set()
		self.marked_for_delete = set()
		self.field_props = dict()
		self.affix = dict()
		self.file_stack = []

# Helper functions
def _load_yaml(path):
	try:
		with open(path, 'r', encoding='utf-8') as f:
			data = yaml.safe_load(f) or {}
	except yaml.YAMLError: raise yaml.YAMLError(f"Error when loading yaml file {path}")
	if not isinstance(data, dict): raise TypeError("Top-level YAML must be a mapping")
	return data
def _resolve_yaml_path(filename: str, configs_folder) -> str:
	basename = os.path.basename(filename)
	if not os.path.isabs(filename):
		os.makedirs(configs_folder, exist_ok=True)
		found = False
		for root, _, files in os.walk(configs_folder):
			if basename in files:
				filename = os.path.normpath(os.path.join(root, basename))
				found = True
				break
		if not found: raise FileNotFoundError(YAML_FILE_NOT_FOUND_MSG.format(filename=filename,configs_folder=configs_folder))
	if not os.path.isfile(filename): raise FileNotFoundError(YAML_FILE_NOT_FOUND_MSG.format(filename=filename,configs_folder=configs_folder))
	_, extension = os.path.splitext(filename)
	if not extension == ".yaml": raise ValueError(f"file {basename} is not yaml")
	return filename
def _yaml_parse_from_dict(handle: dict, ctx: _ParseContext, configs_folder: str, dump_to=None, read_from=None):
	def _dump_to_dict(local_handle: dict, dump_to: dict = {}, read_from: dict = {}):
		local_handle = dict(local_handle)
		local_handle = PSIF.ignore_configs(local_handle)

		for key, raw_prop in local_handle.items():
			prop = PSIF.normalize_prop(key, raw_prop)
			ctx.field_props[key] = prop
			if prop[PSIF.PropKeys.VALIDATOR] == PROP_STRING_VALIDATION_YAML:
				try:
					value = PSIF.fetch_value(key, prop, dump_to, read_from)[1]
					paths = value if isinstance(value, list) else [value]

					for path in paths:
						ctx.file_stack.append(path)
						resolved = _resolve_yaml_path(path, configs_folder)
						included = _load_yaml(VALIDATORS[PROP_STRING_VALIDATION_YAML].parse(resolved))
						dump_to.update(_dump_to_dict(included, dump_to=dump_to, read_from=dump_to))
						ctx.file_stack.pop()
				except Exception as e:
					warnings.warn(PSIF.IGNORE_FIELD_MSG.format(error=e, key=key))
					continue
			else:
				try:
					parsed_value = PSIF.parse_prop(key, prop, dumper=_dump_to_dict, dump_to=dump_to, read_from=read_from)
					dump_to.update({key: parsed_value})
				except Exception as e:
					warnings.warn(PSIF.IGNORE_FIELD_MSG.format(error=e, key=key))
					continue
				PSIF.register_affixable(affix_dict=ctx.affix,key=key, prop=prop, dumper=_dump_to_dict,dump_to=dump_to, read_from=read_from)
			if PSIF.PropKeys.KEEP in prop:
				try:
					keep_val = prop[PSIF.PropKeys.KEEP]
					bv = VALIDATORS[PROP_STRING_VALIDATION_BOOL]
					if bv.validate(keep_val):
						parsed_keep = bv.parse(keep_val)
						if parsed_keep: ctx.explicitly_kept.add(key)
						else: ctx.explicitly_deleted.add(key)
					# malformed keep => ignore
				except Exception:
					pass
		return dump_to

	if dump_to is None: dump_to = {}
	if read_from is None: read_from = dump_to
	return _dump_to_dict(handle, dump_to=dump_to, read_from=read_from)
def _yaml_parse_single_file(entry_file: str, ctx: _ParseContext, configs_folder: str):
	entry_file = _resolve_yaml_path(filename = entry_file, configs_folder=configs_folder)
	handle = _load_yaml(entry_file)
	if not isinstance(handle, dict): raise TypeError("Top-level YAML must be a mapping")
	return _yaml_parse_from_dict(handle=handle, ctx=ctx, configs_folder=configs_folder)
def _yaml_parse_list(entry_list: List[str], ctx: _ParseContext, configs_folder: str):
	if not isinstance(entry_list, list): raise TypeError("Received non-list argument")
	merged = {}
	for file in entry_list:
		entry_file = _resolve_yaml_path(file, configs_folder)
		handle = _load_yaml(entry_file)
		merged = _yaml_parse_from_dict(
			handle=handle,
			ctx=ctx,
			configs_folder=configs_folder,
			dump_to=merged,
			read_from=merged
		)
	return merged

def _resolve_marked_for_delete(ctx: _ParseContext):
	marked = set()
	# Derive referenced fields
	referenced = set()
	for key, prop in ctx.field_props.items():
		if prop.get(PSIF.PropKeys.SOURCE) == PSIF.PropKeys.SOURCE_OPTIONS.VALUE_FROM_FIELD:
			referenced.add(prop.get(PSIF.PropKeys.VALUE))
	# Infer keep using schema defaults
	for field in referenced:
		prop = ctx.field_props.get(field)
		if prop is None:
			if not PSIF.PropSchema.default_keep(): marked.add(field)
			continue
		source = prop.get(PSIF.PropKeys.SOURCE)
		inferred_keep = PSIF.PropSchema.default_keep_for_source(source)
		if not inferred_keep: marked.add(field)


	marked |= ctx.explicitly_deleted
	marked -= ctx.explicitly_kept
	ctx.marked_for_delete = marked

class PSTemplate:
	def __init__(self, base_yaml: dict, configs: list, **ps_kwargs):
		if not isinstance(configs, list):
			if not isinstance(configs, dict):
				raise TypeError(TYPE_ERROR_MSG)
			configs = [configs]
		if not all([isinstance(item,dict) for item in configs]): raise TypeError(TYPE_ERROR_MSG)
		if not isinstance(base_yaml, dict): raise TypeError("'base_yaml' must be a mapping (dict)")
		self.base_yaml = base_yaml
		self.configs = configs
		# kwargs to forward when constructing PlotStyle from a yaml dict
		self._ps_kwargs = ps_kwargs

	def expand(self):
		CONFIG_KEY='config'
		CTX_KEY='ctx'
		payloads = [{} for _ in self.configs]
		for cc, cfg in enumerate(self.configs):
			cfg_norm = {k: PSIF.normalize_prop(k, v) for k, v in cfg.items()}
			ctx = _ParseContext()
			merged = {}
			for key, prop in cfg_norm.items():
				if key == PSIF.PropKeys.YAML:
					#region TODO: Test if it is really necessary to reject source: field 
					# if not cfg_norm[PSIF.PropKeys.YAML][PSIF.PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL]: 
					# 	warnings.warn(PSIF.IGNORE_FIELD_MSG.format(key=k,error=f"'{PSIF.PropKeys.CONFIGS}' field only accepts '{PSIF.PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL}' source")) 
					# else: 
					#endregion
					files = PSIF.fetch_value(
						PSIF.PropKeys.YAML,
						prop,
						read_from=self.base_yaml
					)[1]
					files = files if isinstance(files, list) else [files]
					parsed = _yaml_parse_list(
						entry_list=files,
						ctx=ctx,
						configs_folder=self._ps_kwargs.get(
							'configs_folder',
							PSIF.DEFAULT_CONFIGS_FOLDER
						)
					)
					merged.update(parsed)
				else:
					merged[key] = prop
			payloads[cc][CTX_KEY] = ctx
			payloads[cc][CONFIG_KEY] = merged
		for pp, payload in enumerate(payloads):
			virtual_yaml_dict = {}
			virtual_yaml_dict.update(payload.get(CONFIG_KEY,{}))
			virtual_yaml_dict.update(self.base_yaml)
			if PSIF.PropKeys.LANGUAGE in virtual_yaml_dict:
				language = PSIF.fetch_value(PSIF.PropKeys.LANGUAGE,virtual_yaml_dict[PSIF.PropKeys.LANGUAGE])[1]
			else: language =  PSIF.PropKeys.LANGUAGE_OPTIONS.DEFAULT
			kwargs = {k:v for k,v in self._ps_kwargs.items() if k!=PSIF.PropKeys.LANGUAGE}
			yield PlotStyle(yaml_dict=virtual_yaml_dict, ctx = payload.get(CTX_KEY,_ParseContext()), language=language, **kwargs)

# Entry points
def load_plotstyle(yaml_dict: dict=None, file: str=None, yaml_list: list=None, master=None, 
		language: str = PSIF.PropKeys.LANGUAGE_OPTIONS.DEFAULT, 
		configs_folder: str=PSIF.DEFAULT_CONFIGS_FOLDER, 
		base_folder: str=None, 
		keep_all_fields: bool = PSIF.PropSchema.default_keep(),
	):
	PlotStyle._input_validate(yaml_dict=yaml_dict,yaml_file=file, yaml_list=yaml_list, master=master, language=language, configs_folder=configs_folder, base_folder=base_folder, keep_all_fields=keep_all_fields, ctx=None)

	if file is not None:
		file = _resolve_yaml_path(file, configs_folder=configs_folder)
		yaml_dict = _load_yaml(file)
		yaml_dict, configs = PSIF.ignore_configs(yaml_dict), PSIF.pop_configs(yaml_dict)
		if configs is not None:
			return PSTemplate(yaml_dict, configs, language=language, configs_folder=configs_folder, base_folder=base_folder, keep_all_fields=keep_all_fields)
		return PlotStyle(yaml_dict=yaml_dict, language=language, configs_folder=configs_folder, base_folder=base_folder, keep_all_fields=keep_all_fields)

	# elif yaml_list is not None:
	# elif master is not None and isinstance(master, PlotStyle):
class PlotStyle:
	@staticmethod
	def compose_savefig_options(fname: str | PathLike | IO, format: str = '', **kwargs) -> dict:
		fname = str(fname)
		_, ext = os.path.splitext(fname)
		if ext: format = ext
		format = format.lstrip('.').lower()	# lower converts to lowercase
		fname = f"{fname}.{format}"
		return {'fname': fname, 'format': format, **kwargs}
	@staticmethod
	def compose_set_title_options(label: str, **kwargs): return {'label': label} | kwargs
	@staticmethod
	def settitle_and_savefig(fig: plt.Figure, ax: plt.Axes, savefig_options: dict = {}, set_title_options: dict = {}, savefig: bool = True, save_with_title: bool = False):
		if isinstance(ax, (list, np.ndarray)): ax = ax[0]
		set_title = lambda: ax.set_title(**set_title_options) if set_title_options else lambda: None
		if save_with_title: set_title()
		if savefig: fig.savefig(**savefig_options)
		if not save_with_title: set_title()
	@staticmethod
	def _input_validate(yaml_dict: dict, yaml_file: str, yaml_list: list, master, language: str, configs_folder: str, base_folder: str, keep_all_fields: bool, ctx: _ParseContext):
		if not VALIDATORS[PROP_STRING_VALIDATION_PATHSTR].validate(configs_folder): raise TypeError("configs_folder is an invalid folder name.")
		if bool(base_folder) and not VALIDATORS[PROP_STRING_VALIDATION_PATHSTR].validate(base_folder): raise ValueError("base_folder is an invalid folder name.")
		if yaml_dict is not None:
			if not isinstance(yaml_dict, dict): raise TypeError("yaml_dict must be a dict.")
			return
		if yaml_file is not None:
			yaml_file = _resolve_yaml_path(yaml_file, configs_folder)
			if not VALIDATORS[PROP_STRING_VALIDATION_YAML].validate(yaml_file): raise ValueError(f"Invalid YAML file: {yaml_file}")
		if ctx is not None and not isinstance(ctx,_ParseContext): raise ValueError(f"Invalid context: {ctx}")
		# elif yaml_list is not None:
		# elif master is not None and isinstance(master, PlotStyle):



	def _delete_marked_fields(self, ctx: _ParseContext):
		for key in ctx.marked_for_delete: self.__params__.pop(key, None)
	def __init__(self, *, yaml_dict: dict=None, yaml_file: str=None, yaml_list: list=None, master=None,
			language: str = PSIF.PropKeys.LANGUAGE_OPTIONS.DEFAULT, 
			configs_folder: str=PSIF.DEFAULT_CONFIGS_FOLDER, 
			base_folder: str=None, 
			keep_all_fields: bool = PSIF.PropSchema.default_keep(),
			ctx: _ParseContext=_ParseContext(),
		):
		PlotStyle._input_validate(yaml_dict=yaml_dict, yaml_file=yaml_file, yaml_list=yaml_list, master=master, language=language, configs_folder=configs_folder, base_folder=base_folder, keep_all_fields=keep_all_fields, ctx=ctx)

		if yaml_dict is not None:
			yaml_dict = dict(yaml_dict)
			self.__params__ = _yaml_parse_from_dict(handle = yaml_dict, ctx=ctx, configs_folder=configs_folder)
		elif yaml_file is not None:
			ctx.file_stack.append(yaml_file)
			yaml_file = _resolve_yaml_path(yaml_file, configs_folder) if not os.path.isabs(yaml_file) else yaml_file
			self.__params__ = _yaml_parse_single_file(entry_file=yaml_file, ctx=ctx, configs_folder=configs_folder)
		# elif yaml_list is not None:
		# elif master is not None and isinstance(master, PlotStyle):

		_resolve_marked_for_delete(ctx)

		self.__params__, ctx.affix = PSIF.apply_localization(params_dict=self.__params__, affix_dict=ctx.affix, language=language)
		self.__params__ = PSIF.apply_affixes(params_dict=self.__params__, affix_dict=ctx.affix)
		if not keep_all_fields: self._delete_marked_fields()
		# Load attributes from yaml file onto the object
		for key, value in self.__params__.items(): setattr(self, key, value)
