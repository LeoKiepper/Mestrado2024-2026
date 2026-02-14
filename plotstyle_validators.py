from __future__ import annotations
from typing import Any, Protocol, TypeVar, Callable
from abc import ABC, abstractmethod
import os
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
import matplotlib.colors as mcolors
import ast
UNKNOWN_VALIDATOR_MSG = "Unknown validator"


class NotLambdaFigsize(Exception):
	pass
class NotTupleFigsize(Exception):
	pass
class RejectedExpression(Exception):
	pass
class UnkownValidator(Exception):
	pass
class ParseError(Exception):
	pass

VALIDATORS: dict[str, Validator] = {}
CONTEXT_FIELD_PARSER_FUNC = 'field_parser'
PROP_STRING_VALIDATION = 'validation'
class Validator(ABC):
	@abstractmethod
	def validate(self, value: Any, **context) -> bool: ...
	@abstractmethod
	def parse(self, value: Any, owner: Any = None, **context) -> Any: ...
	@abstractmethod
	def sanitize(self, value: Any, **context) -> Any: ...
def register_validator(name: str):
	def decorator(cls: type[Validator]):
		VALIDATORS[name] = cls()
		return cls
	return decorator

#%% Helper functions

#%% Validator registries
#region pathstr validator
PROP_STRING_VALIDATION_PATHSTR = "pathstr"
@register_validator(PROP_STRING_VALIDATION_PATHSTR)
class PathstrValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if not isinstance(value, str) or not value.strip():
			return False
		
		norm = os.path.normpath(value)
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
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Not a valid pathstr: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return False
		else: return value
#endregion
#region filename validator
PROP_STRING_VALIDATION_FILENAME = "filename"
@register_validator(PROP_STRING_VALIDATION_FILENAME)
class FilenameValidator(Validator):
	def validate(self, value, **context) -> bool:
		return VALIDATORS[PROP_STRING_VALIDATION_PATHSTR].validate(value)
	def parse(self, value, **context) -> str:
		if not self.validate(value): raise ValueError(f"Not a valid filename: {value}")
		return value
	def sanitize(self, value, **context) -> str:
		if not self.validate(value): return ''
		else: return value
#endregion
#region yaml validator
PROP_STRING_VALIDATION_YAML = "yaml"
@register_validator(PROP_STRING_VALIDATION_YAML)
class YamlValidator(Validator):
	def sanitize(self, value: Any, **context) -> Any:
		if not self.validate(value): return ""
		else: return value
	def validate(self, value: Any, **context) -> bool:
		if not VALIDATORS[PROP_STRING_VALIDATION_PATHSTR].validate(value): return False
		if not os.path.isfile(value): return False
		try:
			with open(value, 'r') as f: yaml.safe_load(f)
			return True
		except yaml.YAMLError:
			return False
	def parse(self, value: str, **context) -> str:
		if not self.validate(value): raise ValueError(f"Invalid YAML path: {value}")
		return os.path.normpath(value)
#endregion

#region fileformat validator
PROP_STRING_VALIDATION_FILEFORMAT = "fileformat"
@register_validator(PROP_STRING_VALIDATION_FILEFORMAT)
class FileformatValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		return value in FigureCanvasAgg(Figure()).get_supported_filetypes()
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Unsopported file format: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return None
		else: return value
#endregion
#region str validator
PROP_STRING_VALIDATION_STR = "str"
@register_validator(PROP_STRING_VALIDATION_STR)
class StrValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if isinstance(value,dict):
			return all(isinstance(v,str) for v in value.values())
		return isinstance(value, str)
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Expected string, got {type(value).__name__}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return ""
		else: return value
#endregion
#region bool validator
PROP_STRING_VALIDATION_BOOL = "bool"
@register_validator(PROP_STRING_VALIDATION_BOOL)
class BoolValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		return isinstance(value, bool)
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Not a bool: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return False
		else: return value
#endregion
#region color validator
PROP_STRING_VALIDATION_COLOR = "color"
@register_validator(PROP_STRING_VALIDATION_COLOR)
class ColorValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		return mcolors.is_color_like(value)
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Not a valid color: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return None
		else: return value
#endregion
#region float validator
PROP_STRING_VALIDATION_FLOAT = "float"

@register_validator(PROP_STRING_VALIDATION_FLOAT)
class FloatValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if isinstance(value, bool):
			return False
		try:
			float(value)
			return True
		except (TypeError, ValueError):
			return False

	def parse(self, value: Any, **context):
		if not self.validate(value):
			raise ValueError(f"Expected real-valued numeric, got {type(value).__name__}")
		return float(value)

	def sanitize(self, value: Any, **context):
		if not self.validate(value):
			return 0.0
		return float(value)
#endregion
#region int validator
PROP_STRING_VALIDATION_INT = "int"

@register_validator(PROP_STRING_VALIDATION_INT)
class FloatValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if isinstance(value, bool):
			return False
		try:
			int(value)
			return True
		except (TypeError, ValueError):
			return False

	def parse(self, value: Any, **context):
		if not self.validate(value):
			raise ValueError(f"Expected real-valued numeric, got {type(value).__name__}")
		return int(value)

	def sanitize(self, value: Any, **context):
		if not self.validate(value):
			return 0
		return int(value)
#endregion
#region undetermined validator
PROP_STRING_VALIDATION_UNDETERMINED = "undetermined"
@register_validator(PROP_STRING_VALIDATION_UNDETERMINED)
class UndeterminedValidator(Validator):
    def validate(self, value, **context) -> bool:
        return True
    def parse(self, value, **context):
        # identity parse — but respect composite parsing via field_parser if given
        # if a composite value (dict/list) includes nested field dicts, the caller will
        # have to have used the field_parser context to resolve them; here we just return
        return value
    def sanitize(self, value, **context):
        return value
#endregion
#region fontfamily validator
PROP_STRING_VALIDATION_FONTFAMILY = "fontfamily"
@register_validator(PROP_STRING_VALIDATION_FONTFAMILY)
class FontfamilyValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		try:
			prop = FontProperties(family=value)
			fontpath = findfont(prop, fallback_to_default=False)
			return True
		except Exception:
			return False
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Font family not found: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return None
		else: return value
#endregion
#region fontsize validator
PROP_STRING_VALIDATION_FONTSIZE = "fontsize"
@register_validator(PROP_STRING_VALIDATION_FONTSIZE)
class FontsizeValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if isinstance(value, (int, float)):
			return value > 0
		if isinstance(value, str):
			return value in ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
		return value == None
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Not a valid fontsize: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return None
		else: return value
#endregion
#region linewidth validator
PROP_STRING_VALIDATION_LINEWIDTH = "linewidth"
@register_validator(PROP_STRING_VALIDATION_LINEWIDTH)
class LinewidthValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if isinstance(value, (int, float)): return value >= 0
		return value == None
	def parse(self, value: Any, **context) -> str:
		if not self.validate(value): raise ValueError(f"Not a valid linewidth: {value}")
		return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return None
		else: return value
#endregion

#region dict validator
PROP_STRING_VALIDATION_DICT = "dict"
@register_validator(PROP_STRING_VALIDATION_DICT)
class DictValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if not isinstance(value, dict): return False
		field_parser = context.get(CONTEXT_FIELD_PARSER_FUNC)
		if not callable(field_parser): return False
		try:
			for key, subprop in value.items():
				if not isinstance(key, str): return False
				# field_parser expects a mapping {key: subprop} and will raise on failure
				_ = field_parser({key: subprop})
		except Exception: return False
		return True

	def parse(self, value: dict, **context) -> dict:
		if not self.validate(value, **context): raise ValueError("Invalid dict structure for 'dict' validator")
		field_parser = context.get(CONTEXT_FIELD_PARSER_FUNC)
		if not callable(field_parser): raise ValueError("Missing field_parser context for dict parsing")
		parsed = {}
		for key, subprop in value.items():
			sub_parsed = field_parser({key: subprop})
			# field_parser returns a mapping; extract the parsed value for this key
			parsed[key] = sub_parsed[key]
		return parsed

	def sanitize(self, value: Any, **context) -> dict:
		if not isinstance(value, dict): return {}
		field_parser = context.get(CONTEXT_FIELD_PARSER_FUNC)
		if not callable(field_parser): return {}
		try: return self.parse(value, **context)
		except Exception: return {}
#endregion


#region figsize validator
PROP_STRING_VALIDATION_FIGSIZE = "figsize"
@register_validator(PROP_STRING_VALIDATION_FIGSIZE)
class FigsizeValidator(Validator):
	INVALID_TUPLE_MSG="The expression must return a tuple of length 2 containing ints or floats."
	def _safe_eval(self,value: str):
		return eval(value, {"__builtins__": {}}, {})
	def _is_math_expr_safe(self, expr, allowed_names):
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
	def _unpack_tuple_str(self, value: str):
		if not isinstance(value, str): raise ValueError('Cannot parse non-string value')
		if not (value[0] == '(' and value[-1] == ')'): raise NotTupleFigsize(self.INVALID_TUPLE_MSG)
		value=[v.strip() for v in value[1:-1].split(",")]
		if len(value) != 2: raise ValueError(self.INVALID_TUPLE_MSG)
		return value
	def _check_lambda_str(self, value: str):
		"""Raises error unless it receives an expression strictly matching the expected format"""
		if isinstance(value, str) and value.strip().startswith("lambda"): 
			if 'lambda' in value[len("lambda"):-1].lower(): raise RejectedExpression('Lambda expressions are accepted if "lambda" appears only in the beginning of the string.')
		else: raise NotLambdaFigsize(f'Expression {value} does not start with "lambda".')
		if value.count(":") != 1: raise ValueError("Lambda expressions must contain exactly one ':' to separate arguments from return.")
		params = value[len('lambda '):value.index(":")].strip()
		params = [p.strip() for p in params.split(",")]
		dims = self._unpack_tuple_str(value.split(":", 1)[1].strip())
		for dim in dims:
			if not self._is_math_expr_safe(dim, allowed_names=params): raise ValueError(f"Unsafe expression in dimension: {dim.strip()}")
	def _check_numeric_tuple_from_str(self, value: str):
		"""Raises error unless it receives an expression strictly matching the expected format"""
		try:
			for v in self._unpack_tuple_str(value): float(v)
		except:
			raise ValueError(self.INVALID_TUPLE_MSG)
	def _check_literal_tuple(self, value: tuple):
		if not isinstance(value, tuple): raise ValueError("Not a tuple")
		if len(value) != 2: raise ValueError("Tuple must have length 2")
		for dim in value:
			if not isinstance(dim, (int, float)): raise ValueError('Tuple must have a number on all positions')
			if dim <= 0: raise ValueError('All dimensions must be greater than zero')
	def validate(self, value: str, **context) -> bool:
		try:
			self._check_lambda_str(value)
			return True
		except: pass
		try:
			self._check_numeric_tuple_from_str(value)
			return True
		except: pass
		try:
			self._check_literal_tuple(value)
			return True
		except: pass
		return False
	def parse(self, value: Any, **context) -> tuple | callable:
		if not self.validate(value): raise ValueError(f"figsize {value} in unexpected format")
		if isinstance(value, str):
			return self._safe_eval(value)
		if isinstance(value, tuple):
			return value
	def sanitize(self, value: Any, **context) -> str:
		if not self.validate(value): return None
		else: return self._safe_eval(value)
#endregion

#region gridoptions validator
PROP_STRING_VALIDATION_GRIDOPTIONS = "gridoptions"
@register_validator(PROP_STRING_VALIDATION_GRIDOPTIONS)
class GridoptionsValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if not isinstance(value, list): return False
		field_parser = context.get(CONTEXT_FIELD_PARSER_FUNC)
		if not callable(field_parser): return False
		for item in value:
			if not isinstance(item, dict): return False
			try:
				for subkey, subprop in item.items():
					if not isinstance(subprop, dict): return False
					field_parser({subkey: subprop})
			except Exception as e: 
				return False
		return True
	def parse(self, value: Any, **context) -> list[dict]:
		if not self.validate(value, **context):
			raise ValueError("Invalid gridoptions structure")
		field_parser = context["field_parser"]
		parsed_list = []
		for item in value:
			parsed_dict = {}
			for subkey, subprop in item.items():
				parsed = field_parser({subkey: subprop})
				parsed_dict[subkey] = parsed[subkey]
			parsed_list.append(parsed_dict)
		return parsed_list
	def sanitize(self, value, **context) -> list:
		if not self.validate(value, **context): return []
		return value
#endregion
#region plotstyle validator
PROP_STRING_VALIDATION_PLOTOPTIONS = "plotoptions"
@register_validator(PROP_STRING_VALIDATION_PLOTOPTIONS)
class PlotoptionsValidator(Validator):
	def validate(self, value: Any, **context) -> bool:
		if not isinstance(value, dict):
			return False
		field_parser = context.get(CONTEXT_FIELD_PARSER_FUNC)
		if not callable(field_parser):
			return False
		try:
			field_parser(value)
			return True
		except Exception:
			return False
	def parse(self, value: dict, **context) -> dict:
		field_parser = context.get(CONTEXT_FIELD_PARSER_FUNC)
		if not callable(field_parser):
			raise ValueError("plotstyle requires field_parser context")
		parsed = field_parser(value)
		if not isinstance(parsed, dict):
			raise ValueError("plotstyle did not produce a dict")
		return parsed
	def sanitize(self, value: Any, **context) -> dict:
		if not self.validate(value, **context):
			return {}
		return self.parse(value, **context)
#endregion