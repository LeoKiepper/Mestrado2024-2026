from enum import Enum, auto
from plotstyle_validators import *
from dataclasses import dataclass
from parse import parse
from typing import Dict
import warnings
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
	print(f"{lineno}: {message}")
warnings.showwarning = custom_showwarning


#region SCHEMA
DEFAULT_CONFIGS_FOLDER = 'plotstyle_configs'
class InvalidSourceType(Exception): pass
class SourceFieldMissing(Exception): pass
class FieldIntent(Enum):
	IMPLICIT = 'implicit'
	EXPLICIT = 'explicit'
class PropKeys:
	VALUE = "value"
	VALIDATOR = "validator"
	SOURCE = "source"
	@dataclass
	class SOURCE_OPTIONS:
		VALUE_FROM_LITERAL = 'literal'
		VALUE_FROM_FIELD = 'field'
	KEEP = "keep"
	SUFFIX = "suffix"
	PREFIX = "prefix"
	LOCALIZABLE = "localizable"
	LANGUAGE = 'language'
	@dataclass
	class LANGUAGE_OPTIONS:
		ENGLISH = 'en'
		PROTUGUESE = 'pt'
		DEFAULT = ENGLISH
	CONFIGS = "configs"
	SAVE_FOLDER = "save_folder"
	YAML = "yaml"
	LANGUAGE = "language"
class PropSchema:
	@dataclass
	class TAGS:
		IMPLICIT = FieldIntent.IMPLICIT
		EXPLICIT = FieldIntent.EXPLICIT
	TAG_TO_INTENT = {
		f"!{TAGS.IMPLICIT}": FieldIntent.IMPLICIT,
		f"!{TAGS.EXPLICIT}": FieldIntent.EXPLICIT,
	}
	INTENT_DEFAULTS = {
		FieldIntent.IMPLICIT: {
			PropKeys.VALIDATOR: PROP_STRING_VALIDATION_UNDETERMINED,
			PropKeys.SOURCE: PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL,
		},
		FieldIntent.EXPLICIT: {
			PropKeys.VALIDATOR: PROP_STRING_VALIDATION_UNDETERMINED,
			PropKeys.SOURCE: PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL,
		},
	}

	
	REQUIRED_EXPLICIT_PROPS = {
		PropKeys.VALUE,
		PropKeys.SOURCE,
	}
	@classmethod
	def looks_explicit(cls, mapping: dict) -> bool:
		return bool(set(mapping).intersection(cls.REQUIRED_EXPLICIT_PROPS))

	AFFIX_KEYS = (
		PropKeys.PREFIX,
		PropKeys.SUFFIX,
	)
	AFFIX_KEY_FORMAT = "{key}__{affix}__"
	@classmethod
	def iter_affixes(cls):
		return cls.AFFIX_KEYS
	@classmethod
	def is_affix(cls, key: str) -> bool:
		return key in cls.AFFIX_KEYS
	@classmethod
	def format_key(cls, key: str, affix: str) -> str:
		return cls.AFFIX_KEY_FORMAT.format(key=key, affix=affix)
	@classmethod
	def unpack_affix_key(cls, key: str) -> tuple[str, str]:
		result = parse(cls.AFFIX_KEY_FORMAT, key)
		if result:
			return result['key'], result['affix']
		raise ValueError(f"Key '{key}' does not match format '{cls.AFFIX_KEY_FORMAT}'")

	SOURCE_DEFAULT_KEEP = {
		PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL: True,
		PropKeys.SOURCE_OPTIONS.VALUE_FROM_FIELD: False,
	}
	_KEEP_DEFAULT_CLASS_LEVEL = True
	@classmethod
	def default_keep_for_source(cls, source: str):
		return cls.SOURCE_DEFAULT_KEEP.get(source, cls._KEEP_DEFAULT_CLASS_LEVEL)
	@classmethod
	def default_keep(cls): return cls._KEEP_DEFAULT_CLASS_LEVEL

class _TaggedValue:
	def __init__(self, value, intent: FieldIntent):
		self.value = value
		self.intent = intent
def _yaml_intent_constructor(intent: FieldIntent):
	def ctor(loader, node):
		if isinstance(node, yaml.MappingNode):
			value = loader.construct_mapping(node)
		elif isinstance(node, yaml.SequenceNode):
			value = loader.construct_sequence(node)
		else:
			value = loader.construct_scalar(node)
		return _TaggedValue(value, intent)
	return ctor
def configure_tags():
	for tag, intent in PropSchema.TAG_TO_INTENT.items():
		yaml.SafeLoader.add_constructor(tag, _yaml_intent_constructor(intent))
#endregion


#region INTERFACE
IGNORE_FIELD_MSG = "[INFO]: Error raised when parsing field '{key}'. Field was ignored and will not load onto the object.\n\t{error}."
PARSE_ERROR_MSG = "Failed to parse key '{key}' with validator '{validator}': {error}"
def normalize_prop(key: str, raw_prop: Any) -> dict:
	def _as_implicit(value: Any) -> dict:
		return {
			PropKeys.VALUE: value,
			**PropSchema.INTENT_DEFAULTS[FieldIntent.IMPLICIT],
		}

	# --- Tagged nodes: follow explicit intent ---
	if isinstance(raw_prop, _TaggedValue):
		intent = raw_prop.intent
		value = raw_prop.value

		if intent is FieldIntent.IMPLICIT:
			return _as_implicit(value)

		if intent is FieldIntent.EXPLICIT:
			if not isinstance(value, dict) or PropKeys.VALUE not in value:
				raise ValueError(f"Field '{key}' tagged {PropSchema.TAGS.EXPLICIT} but missing payload key '{PropKeys.VALUE}'")
			prop = value.copy()
			for k, v in PropSchema.INTENT_DEFAULTS[FieldIntent.EXPLICIT].items():
				prop.setdefault(k, v)
			return prop

		raise AssertionError("Unhandled FieldIntent")

	# --- Untagged nodes: decide by shape and presence of payload key ---
	if isinstance(raw_prop, dict):
		# canonical when payload key present
		if PropKeys.VALUE in raw_prop:
			prop = raw_prop.copy()
			prop.setdefault(PropKeys.VALIDATOR, PROP_STRING_VALIDATION_UNDETERMINED)
			prop.setdefault(PropKeys.SOURCE, PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL)
			return prop

		# implicit dict (but warn if it contains canonical-like keys)
		suspicious = set(raw_prop.keys()).intersection(PropSchema.REQUIRED_EXPLICIT_PROPS)
		if PropSchema.looks_explicit(raw_prop):			warnings.warn(
				f"Field '{key}' contains keys {sorted(suspicious)} but no "
				f"'{PropKeys.VALUE}' entry. Interpreted as implicit field.",
				UserWarning,
			)
		return _as_implicit(raw_prop)

	# non-dict -> implicit
	return _as_implicit(raw_prop)
def fetch_value(key: str, prop: dict, dump_to: dict={}, read_from: dict={}):
	if PropKeys.SOURCE not in prop.keys(): raise SourceFieldMissing
	if prop[PropKeys.SOURCE] == PropKeys.SOURCE_OPTIONS.VALUE_FROM_LITERAL:
		return (key, prop[PropKeys.VALUE])
	elif prop[PropKeys.SOURCE] == PropKeys.SOURCE_OPTIONS.VALUE_FROM_FIELD:
		return (prop[PropKeys.VALUE], (dump_to | read_from)[prop[PropKeys.VALUE]])
	else:
		raise InvalidSourceType
def parse_prop(key: str, prop: dict, dumper: callable, dump_to: dict, read_from: dict):
	try:
		validator = VALIDATORS.get(prop[PropKeys.VALIDATOR])
		if not validator:
			raise UnkownValidator(UNKNOWN_VALIDATOR_MSG.format(validator=prop[PropKeys.VALIDATOR], key=key))
		return validator.parse(fetch_value(key, prop, dump_to, read_from)[1], **{
			CONTEXT_FIELD_PARSER_FUNC: lambda nested_handle: dumper(nested_handle, dump_to={}, read_from=dump_to)
		})
	except Exception as e:
		raise ParseError(PARSE_ERROR_MSG.format(key=key, validator=prop[PropKeys.VALIDATOR], error=e))
def apply_localization(params_dict: dict, affix_dict: dict, language: str):
	def resolve(value):
		if isinstance(value, dict):
			# localization dict
			if language in value:
				return value[language]
			# generic dict
			return {k: resolve(v) for k, v in value.items()}
		if isinstance(value, list): return [resolve(v) for v in value]
		return value

	params_dict = {k: resolve(v) for k, v in params_dict.items()}
	affix_dict  = {k: resolve(v) for k, v in affix_dict.items()}
	return params_dict, affix_dict
def register_affixable(affix_dict: dict, key: str, prop: dict, dumper: callable, dump_to: dict, read_from: dict):
	for affix_type in PropSchema.iter_affixes():
		try:
			if affix_type in prop and prop[affix_type] != {}:
				new_key = PropSchema.format_key(key, affix_type)
				affix_dict.update({new_key: parse_prop(key, prop[affix_type], dumper=dumper, dump_to=dump_to, read_from=read_from)})
		except Exception as e:
			warnings.warn(IGNORE_FIELD_MSG.format(error=e, key=key + '.' + affix_type))
			continue
def apply_affixes(params_dict: dict, affix_dict: dict):
	for field, affix_value in affix_dict.items():
		key, affix = PropSchema.unpack_affix_key(field)
		if affix == PropKeys.SUFFIX: params_dict.update({key: params_dict[key] + affix_value})
		elif affix == PropKeys.PREFIX: params_dict.update({key: affix_value + params_dict[key]})
		else: raise ValueError(f"Unknown affix type for key {key}")
	
	return params_dict
def ignore_configs(handle:dict):
	handle = dict(handle)
	if PropKeys.CONFIGS in handle: del handle[PropKeys.CONFIGS]
	return handle
def pop_configs(handle:dict):
	return handle[PropKeys.CONFIGS]
#endregion