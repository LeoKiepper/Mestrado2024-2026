import os
import sys
import tempfile
import numpy as np
import json
# default local dir (used when env var not present)
_DEFAULT_MEASURE_OUTPUT_DIR = "measure_outputs"
NOPLOT = any(arg == "--noplot" for arg in sys.argv)

def get_output_dir():
	"""
	Resolve e garante existência do diretório de saída.
	Prioriza a variável de ambiente MEASURE_OUTPUT_DIR (exportada por measure.py).
	"""
	base = os.environ.get("MEASURE_OUTPUT_DIR")
	if not base:
		base = os.path.join(os.getcwd(), _DEFAULT_MEASURE_OUTPUT_DIR)
	os.makedirs(base, exist_ok=True)
	return base
def resolve_base_name():
	"""
	Resolve nome base para arquivos de predição:
	1) MEASURE_PREDICTION_NAME env var (exportada por measure.py), else
	2) nome do script em execução (sys.argv[0], sem extensão).
	"""
	env_name = os.environ.get("MEASURE_PREDICTION_NAME")
	if env_name:
		return env_name
	script_path = sys.argv[0]
	if not script_path:
		raise RuntimeError("Cannot resolve prediction output name: sys.argv[0] is empty")
	base = os.path.basename(script_path)
	name, _ = os.path.splitext(base)
	if not name:
		raise RuntimeError(f"Cannot resolve prediction output name from script path: {script_path}")
	return name
def resolve_scalar_output_path(name: str = None) -> str:
	"""
	Return path for scalar output file for given name.
	If name is None, resolve using resolve_prediction_output_name().
	"""
	if name is None:
		name = resolve_base_name()
	base = get_output_dir()
	return os.path.join(base, f"{name}.out")
def report(output, prediction=None, reference=None, x=None, save_metadata: bool = False) -> None:
	"""
	Write scalar output, optionally persist prediction arrays and metadata.
	Atomic for scalar output; best-effort for auxiliary artifacts.
	"""
	# ---- scalar output ----
	try:
		s = f"{float(output):.12g}"
	except Exception:
		s = "nan"

	try:
		path = resolve_scalar_output_path()
		d = os.path.dirname(path) or "."
		fd, tmp = tempfile.mkstemp(prefix=".tmp_measure_", dir=d)
		try:
			with os.fdopen(fd, "w", encoding="utf-8") as tf:
				tf.write(s + "\n")
				tf.flush()
				os.fsync(tf.fileno())
			os.replace(tmp, path)
		except Exception:
			try:
				if os.path.exists(tmp):
					os.remove(tmp)
			except Exception:
				pass
			sys.stdout.write(s + "\n")
			sys.stdout.flush()
	except Exception:
		sys.stdout.write(s + "\n")
		sys.stdout.flush()

	# ---- prediction + reference ----
	if prediction is not None:
		try:
			base_name = resolve_base_name()
			save_to_npy(prediction, base_name)

			import re
			m = re.match(r"^(.*)_\d+$", base_name)
			stem = m.group(1) if m else base_name

			out_dir = os.environ.get("MEASURE_OUTPUT_DIR")
			if not out_dir:
				out_dir = os.path.join(os.getcwd(), _DEFAULT_MEASURE_OUTPUT_DIR)

			ref_path = os.path.join(out_dir, f"{stem}_ref.npy")
			if not os.path.exists(ref_path):
				# caller is responsible for providing reference-compatible prediction
				save_to_npy(reference, f"{stem}_ref")

			if x is not None:
				time_path = os.path.join(out_dir, f"{stem}_timestamps.npy")
				if not os.path.exists(time_path):
					save_to_npy(x, f"{stem}_timestamps")
		except Exception as _e:
			print(f"Warning: failed to save prediction/reference for {base_name}: {_e}")

	# ---- optional metadata ----
	if not save_metadata:
		return
	if os.environ.get("MEASURE_ACTIVE") == "1":
		return

	try:
		stem = os.environ.get("MEASURE_OUTPUT_STEM")
		if not stem:
			import __main__, pathlib
			stem = pathlib.Path(__main__.__file__).stem
		dump_metadata(stem, suffix="_standalone")
	except Exception:
		pass
def read_prediction_output(name: str = None) -> float:
	"""
	Read the scalar output saved by report_output(name).
	Returns float value or float('nan') on failure.
	"""
	path = resolve_scalar_output_path(name)
	try:
		if not os.path.exists(path):
			return float("nan")
		with open(path, "r", encoding="utf-8", errors="replace") as f:
			txt = f.read().strip()
			if not txt:
				return float("nan")
			toks = txt.split()
			try:
				return float(toks[-1])
			except Exception:
				return float("nan")
	except Exception:
		return float("nan")
def resolve_prediction_npy_path(name: str = None) -> str:
	"""
	Return full path for prediction .npy file.
	"""
	if name is None:
		name = resolve_base_name()
	base = get_output_dir()
	return os.path.join(base, f"{name}.npy")
def save_to_npy(obj, name: str) -> str:
	"""
	Save array-like `obj` as .npy under the resolved output dir,
	using the provided `name` (no extension required). Returns full path.
	"""
	if not name or not isinstance(name, str):
		raise TypeError("Output name must be a non-empty string")

	try:
		arr = np.asarray(obj)
	except Exception as e:
		raise TypeError("Object cannot be converted to a NumPy array") from e

	path = resolve_prediction_npy_path(name)
	try:
		np.save(path, arr)
	except Exception as e:
		raise RuntimeError(f"Failed to save prediction .npy to {path}: {e}")
	return path
def load_prediction_npy(name: str):
	"""
	Load a prediction .npy by base name (no extension) from the outputs dir.
	Returns a numpy array. Raises FileNotFoundError if missing.
	"""
	if not name or not isinstance(name, str): raise TypeError("Name must be a non-empty string")
	path = resolve_prediction_npy_path(name)
	if not os.path.exists(path): raise FileNotFoundError(path)
	try:
		return np.load(path, allow_pickle=True)
	except Exception as e:
		raise RuntimeError(f"Failed loading {path}: {e}") from e
def remove_prediction_npy(name: str) -> None:
	"""Remove the prediction .npy file identified by base name (no extension)."""
	path = resolve_prediction_npy_path(name)
	try:
		if os.path.exists(path):
			os.remove(path)
	except Exception as e:
		raise RuntimeError(f"Failed removing {path}: {e}") from e
def aggregate_predictions_to_matrix(names: list, out_name: str) -> str:
	"""
	Load a list of prediction .npy files (base names without extension),
	stack them into a 2-D matrix with shape (len(names), L) where L is the
	length of each flattened prediction, and save the result using
	save_output_to_npy(out_matrix, out_name).
	Returns the saved path.
	Raises ValueError if input shapes mismatch.
	"""
	if not names or not isinstance(names, (list, tuple)): raise TypeError("names must be a non-empty list or tuple of base names")
	arrs = []
	for n in names:
		a = load_prediction_npy(n)
		a = np.asarray(a)
		if a.ndim != 1:
			a = a.ravel()
		arrs.append(a)
	lengths = [a.shape[0] for a in arrs]
	if len(set(lengths)) != 1: raise ValueError("Predictions have different lengths; cannot stack into matrix")
	mat = np.vstack(arrs)
	return save_to_npy(mat, out_name)

_metadata_registry = {}
import inspect, types
import json, pathlib, enum, dataclasses, numpy as np
def _normalize_metadata_value(v):
	if v is None:
		return None
	if isinstance(v, (str, int, float, bool)):
		return v
	if isinstance(v, (list, tuple, set)):
		return [_normalize_metadata_value(x) for x in v]
	if isinstance(v, dict):
		return {str(k): _normalize_metadata_value(val) for k, val in v.items()}
	if isinstance(v, type):
		return v.__name__
	if isinstance(v, enum.Enum):
		return v.name
	if dataclasses.is_dataclass(v):
		return {k: _normalize_metadata_value(val) for k, val in dataclasses.asdict(v).items()}
	if isinstance(v, pathlib.Path):
		return str(v)
	if isinstance(v, np.ndarray):
		return {
			"type": "ndarray",
			"shape": v.shape,
			"dtype": str(v.dtype)
		}
	return repr(v)
def register_metadata(metadata: dict, *, scope: str = None):
	if scope is None:
		scope = _get_active_metadata_scope()

	normalized = {k: _normalize_metadata_value(v) for k, v in metadata.items()}

	if scope:
		_metadata_registry.setdefault(scope, {}).update(normalized)
	else:
		_metadata_registry.update(normalized)

def collect_module_variables():
	frame = inspect.currentframe()
	try:
		caller_globals = frame.f_back.f_globals
		return {
			k: v
			for k, v in caller_globals.items()
			if not k.startswith("_")
			and not callable(v)
			and not isinstance(v, types.ModuleType)
		}
	finally:
		del frame
def dump_metadata(stem, suffix=""):
	path = os.path.join(
		get_output_dir(),
		f"{stem}_metadata{'_'+suffix if suffix else ''}.json"
	)

	if os.path.exists(path):
		with open(path, "r", encoding="utf-8") as f:
			try:
				existing = json.load(f)
			except json.JSONDecodeError:
				existing = {}
	else:
		existing = {}

	# overwritten = False
	# touched_existing_scope = False

	# for scope, new_data in _metadata_registry.items():
	# 	if scope not in existing:
	# 		continue

	# 	touched_existing_scope = True
	# 	old_data = existing.get(scope, {})

	# 	if not isinstance(old_data, dict):
	# 		overwritten = True
	# 		continue

	# 	if set(new_data.keys()) & set(old_data.keys()):
	# 		overwritten = True

	# merge
	merged = existing.copy()
	for scope, data in _metadata_registry.items():
		if scope not in merged or not isinstance(merged.get(scope), dict):
			merged[scope] = {}
		merged[scope].update(data)

	with open(path, "w", encoding="utf-8") as f:
		json.dump(merged, f, indent=4)

	# if not existing:
	# 	message = "[INFO] Metadata saved successfully"
	# elif overwritten:
	# 	message = "[WARNING] Metadata updated successfully. Older metadata was overwritten."
	# else:
	# 	message = "[INFO] Metadata updated successfully"

	# print(message)

_ACTIVE_METADATA_SCOPE = ""
def _set_active_metadata_scope(scope: str):
	global _ACTIVE_METADATA_SCOPE
	_ACTIVE_METADATA_SCOPE = scope
def _get_active_metadata_scope():
	return _ACTIVE_METADATA_SCOPE

def lambda_to_source(fn):
	import inspect, ast
	try:
		src = inspect.getsource(fn).strip()
	except Exception:
		return repr(fn)

	tree = ast.parse(src)

	for node in ast.walk(tree):
		if isinstance(node, ast.Lambda):
			return src[node.col_offset:node.end_col_offset]

	return src

def get_figure_run_context() -> tuple[str,str]:
	prefix=os.environ.get("MEASURE_FIGURE_FILENAME_PREFIX","")
	folder=os.environ.get("MEASURE_FIGURE_FOLDER_SUFFX","")
	return prefix,folder
