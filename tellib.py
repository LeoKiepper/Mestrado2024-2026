import hashlib, pickle, pandas as pd, ast, textwrap
import inspect, shutil, os, platform, warnings, sys
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from IPython import get_ipython
from warnings import warn
# Override the standard warnings formatter
warnings.formatwarning = lambda msg, cat, fn, ln, line=None: f"{cat.__name__}: {msg}\n"
ip = get_ipython()
if ip is not None: ip.showwarning = lambda msg, cat, fn, ln, *args, **kwargs: sys.stderr.write(f"{cat.__name__}: {msg}\n")

@dataclass
class telclass:
	tel: pd.DataFrame
	timestamp_zero: float
expected_fields = {f.name for f in fields(telclass)}

class telprocessor(ABC):
	def __init__(self, bagfile: str, cache_dir: str = "cache"):
		# ====================================  Treats bagfile related inputs and attributes
		self._bagfile_input = bagfile;	simple_filename = os.path.basename(bagfile) == bagfile
		if not telprocessor._is_valid_dirname(self._bagfile_input, as_abs = not simple_filename):	raise ValueError(f"Falied to validate '{self._bagfile_input}'.")
		if cache_dir:
			cache_dir = os.path.normpath(cache_dir)
			self._cache_dir = os.path.abspath(cache_dir)
		else:
			self._cache_dir = ""
		if not os.path.exists(self._cache_dir):	os.makedirs(self._cache_dir)
		self._bagfile_input_dir = os.path.dirname(bagfile)
		self._bagfile_basename = os.path.splitext(os.path.basename(bagfile))[0]
		self._bagfile_in_cache = os.path.join(self._cache_dir, self._bagfile_basename + '.bag')
		if not telprocessor._is_valid_dirname(self._bagfile_in_cache): 								raise ValueError(f"Falied to validate '{self._bagfile_in_cache}'.")
		if not (bagfile_ext := os.path.splitext(os.path.basename(bagfile))[1]) == '.bag': 			raise ValueError(f"Bag file in {bagfile_ext} format, expected .bag")
		self.bagfile, self.baghash = telprocessor._ValidateBagfile(self)
		self._picklefile = os.path.join(self._cache_dir, self._bagfile_basename + '.telemetry.pkl')

		# ====================================  Treats builder related inputs and attributes
		source = textwrap.dedent(inspect.getsource(self.builder))
		tree = ast.parse(source)
		normalized = ast.dump(tree, annotate_fields=False, include_attributes=False)
		self._builderhash = self._calc_hash(normalized.encode("utf-8"))


	@staticmethod
	def _calc_hash(content) -> str:
		return hashlib.md5(content).hexdigest()
	@staticmethod
	def _get_num_serial_ports() -> int:
		system = platform.system().lower()
		valid_ports=[]
		if system == 'windows':
			ports = [f"COM{i}" for i in range(1, 256)]  # Geração até COM255, mas pode ser ajustado
			valid_ports = []
			for port in ports:
				if os.path.exists(f"\\\\.\\{port}"):
					valid_ports.append(port)
		elif system == 'linux' or system == 'darwin':  # Linux/Mac
			dev_dir = "/dev/"
			valid_ports = [f"{dev_dir}{entry}" for entry in os.listdir(dev_dir) if entry.startswith("ttyS") or entry.startswith("ttyUSB")]
		else: return None
		return len(valid_ports)
	@staticmethod
	def _is_valid_dirname(name: str, as_abs: bool=True) -> bool:
		"""
			Validate that a path-like directory name is safe to use.

			- 	On Windows, rejects reserved device names (CON, PRN, AUX, NUL, COM, LPT)
				and illegal characters: <>:\"/\\|?*\0
			- 	On POSIX, rejects only the null character.
			- 	Ensures no single path component exceeds NAME_MAX and full path does not exceed PATH_MAX.
			:param name: Single or multi-level directory path to validate. Allows "." for indicating local directory
			:return: True if name is acceptable, False otherwise.
		"""
		system = platform.system()

		# determine limits via pathconf or fallbacks
		try:
			name_max = os.pathconf('.', 'PC_NAME_MAX')
		except (AttributeError, ValueError, OSError):
			name_max = 255
		try:
			path_max = os.pathconf('.', 'PC_PATH_MAX')
		except (AttributeError, ValueError, OSError):
			path_max = 4096

		# check full path length
		if as_abs:
			full_path = os.path.abspath(name)
			if len(full_path) > path_max:
				warn(f"Full path too long (>{path_max} chars)")
				return False

		NumPorts = telprocessor._get_num_serial_ports()
		if system == "Windows":
			reserved_names = {
				"CON", "PRN", "AUX", "NUL",
				*(f"COM{i}" for i in range(1, NumPorts)),
				*(f"LPT{i}" for i in range(1, NumPorts)),
			}
			# extract only the final component to check reserved names
			base = os.path.splitext(os.path.basename(name))[0].upper()
			if base in reserved_names:
				warn(f"'{base}' is a reserved Windows name")
				return False

			illegal_chars = {'<', '>', ':', '"', '/', '\\', '|', '?', '*', '\0'}
			_, path = os.path.splitdrive(name)

			for part in path.split(os.sep):
				if any(c in illegal_chars for c in part):
					warn(f"Illegal character in '{part}'")
					return False
				if len(part) > name_max:
					warn(f"Component '{part}' exceeds max length ({name_max})")
					return False

		else:
			for part in name.split('/'):
				if '\0' in part:
					warn("Null character not allowed")
					return False
				if len(part) > name_max:
					warn(f"Component '{part}' exceeds max length ({name_max})")
					return False

		return True

	def _ValidateBagfile(self):
		"""
		Validates bagfiles for processing, selecting between the one informed explicitly in the constructor and the one in cache_dir, if it exists.
		Intended use case is for the bagfile specified on the constructor to just be a filename with no directory, and for it to be present in cache_dir.
		Steers and prompts the user toward the intended use case

		Prompts the user in these cases: 
			-	If bagfiles with the provided name exists in both directories, but are different, ask which bag to use,
				but take no umprompted action to correct the situation toward intended use case, so as to nag the user to resolve the situation;
			-	If the bagfile specified in the constructor exists in cache_dir but could not be opened in base directory, 
				nags the user to consent to move the bagfile by making "don't move" the default option.
		In all other cases, warns the user to hint toward the intended use case

		Returns the selected bagfile and its hash

		:return: Tuple (selected_bagfile, bagfile_hash)
		:raises Exception: If neither file can be read successfully;
		"""
		hash_input = ''; hash_cache = ''; e_input = None; e_cache = None
		try:
			with open(self._bagfile_input, "rb") as f1:
				hash_input = telprocessor._calc_hash(f1.read())
		except FileNotFoundError as e_input:							pass
		except PermissionError as e_input:								pass
		except OSError as e_input:										pass
		try:
			with open(self._bagfile_in_cache, "rb") as f2:
				hash_cache = telprocessor._calc_hash(f2.read())
		except FileNotFoundError as e_cache:							pass
		except PermissionError as e_cache:								pass
		except OSError as e_cache:										pass

		if not hash_input and not hash_cache:							raise Exception(f"Failed to read bag file: {e_input,e_cache}")
		samehash = (hash_input == hash_cache)		
		if self._bagfile_input_dir != '' and self._cache_dir != '':
			if samehash:
				warn(f"Specified bag file is duplicated in '{self._cache_dir}' and will be used instead.")
				b = self._bagfile_in_cache
				h = hash_cache
			elif not hash_input:
				warn(f"Bag file in '{os.abspath(self._bagfile_input_dir)}' failed to open. Using from '{self._cache_dir}'")
				b = self._bagfile_in_cache
				h = hash_cache
			elif not hash_cache:
				print(f"Bag file is not in '{self._cache_dir}'. Move?")
				move = input('y/[n]')
				if move == 'y': 	
					try: 
						shutil.move(self._bagfile_input, self._bagfile_in_cache)
						b = self._bagfile_in_cache
						h = hash_cache
					except: 	
						warn('Move failed')
						b = self._bagfile_input
						h = hash_input
			else:
				print(f"A different bag file was found in '{self._cache_dir}' with the same name. Which bag should be used?")
				print(f"	1. {self._bagfile_in_cache}")
				print(f"	2. {self._bagfile_input}")
				whichbag = input()
				if whichbag!='2':
					b = self._bagfile_in_cache
					h = hash_cache
				else:
					b = self._bagfile_input
					h = hash_input
		else:
			# Expected case: bagfile is in cache_dir; 
			# 	-	bagfile_input == bagfile_in_cache. This equivalence arises from the way  
			# 		bagfile_input and bagfile_in_cache are calculated composing user inputs.
			#	-	hash1 == hash2, because the OS makes sure file names are unique in a directory, so a hash can only ever be unique in this case
			# Return whichever
			b = self._bagfile_in_cache
			h = hash_cache
		return b, h
	def _SaveTelemetry(self):
		savedata = {
			"baghash": self.baghash,
			"builderhash": self._builderhash,
			"tel": self.tel
		}

		with open(self._picklefile, "wb") as f:
			pickle.dump(savedata, f)
	def _LoadTelemetry(self):
		try: var = pickle.load(open(self._picklefile, 'rb'))
		except FileNotFoundError as e:										print("[INFO]  Pickle file not found."); return False
		except pickle.UnpicklingError as e:									print("[INFO]  Failed to unpickle the file."); return False
		except EOFError as e:												print("[INFO]  Incomplete or corrupted pickle file."); return False
		except AttributeError as e:											print("[INFO]  Error accessing saved object attributes."); return False
		except ImportError as e:											print("[INFO]  Missing module required to load pickle."); return False
		except IndexError as e:												print("[INFO]  Index error while loading pickle."); return False
		except TypeError as e:												print("[INFO]  Type error while loading pickle."); return False
		if not isinstance(var, dict): 										print("[INFO]  Object structure not recognized."); return False
		if 'baghash' not in var:                 							print("[INFO]  Missing bag hash in pickle file."); return False
		if not isinstance(loadedbaghash := var['baghash'], str):			print("[INFO]  Loaded bag file hash in unexpected format."); return False
		if 'builderhash' not in var:                 						print("[INFO]  Missing builder hash in pickle file."); return False
		if not isinstance(loadedbuilderhash := var['builderhash'], str):	print("[INFO]  Loaded builder hash in unexpected format."); return False
		if 'tel' not in var:                 								print("[INFO]  Missing telemetry in pickle file."); return False
		if not isinstance(loadedtel := var['tel'], telclass):				print("[INFO]  tel in unexpected format."); return False
		if not set(vars(loadedtel).keys()) == expected_fields: 				print("[INFO]  tel fields do not match expected structure."); return False
		if not loadedbaghash == self.baghash:								print("[INFO]  Mismatch detected in bag file hash."); return False
		if not loadedbuilderhash == self._builderhash:						print("[INFO]  Mismatch detected in builder hash."); return False
		
		print('[INFO]  Found a valid .pkl file. Telemetry loaded successfully.')
		return loadedtel
	def get(self):
		"""
			Checks if a pre-built .telemetry.pkl file exists for the specified .bag file and builder method, and loads it. 
			Otherwise, creates a DataFrame variable converted from the .bag file using the specified builder and saves it.
		"""

		tel = self._LoadTelemetry()
		if not tel: 
			print("[INFO]  Telemetry build required.")
			self.tel = tel = self.builder(self.bagfile)
			print('[INFO]  Telemetry build finished.')
			self._SaveTelemetry()
		return tel

	@abstractmethod
	def builder(self, bagfile: str) -> telclass:
		...