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
	def __init__(self, bagfile: str, telems_dir: str = "telems"):
		if not telprocessor.__is_valid_dirname(telems_dir): 							raise ValueError(f"Invalid directory: {telems_dir}")
		self.telems_dir = telems_dir
		if not os.path.exists(self.telems_dir):	os.makedirs(self.telems_dir)
		if not (bagfile_ext := os.path.splitext(os.path.basename(bagfile))[1]) == '.bag': 	raise ValueError(f"Bag file in {bagfile_ext}, expected .bag")
		self.bagfile_input = os.path.abspath(bagfile)
		self.bagfile_basename = os.path.splitext(os.path.basename(bagfile))[0]
		self.bagfile_in_telems = os.path.join(telems_dir, self.bagfile_basename + '.bag')
		self.bagfile_input_dir = os.path.abspath(os.path.dirname(bagfile))
		self.bagfile, self.baghash = telprocessor.__ValidateBagfile(self)
		self.picklefile = self.telems_dir + '\\' + self.bagfile_basename + '.telemetry.pkl'


		source = textwrap.dedent(inspect.getsource(self.builder))
		tree = ast.parse(source)
		normalized = ast.dump(tree, annotate_fields=False, include_attributes=False)
		self.builderhash = self.__calc_hash(normalized.encode("utf-8"))
		# self.builderhash = self.__calc_hash(inspect.getsource(self.builder).encode("utf-8"))

		self.cache_dir = telems_dir

	@staticmethod
	def __calc_hash(content) -> str:
		return hashlib.md5(content).hexdigest()
	@staticmethod
	def __get_num_serial_ports() -> int:
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
	def __is_valid_dirname(name: str) -> bool:
		system = platform.system()

		NumPorts=telprocessor.__get_num_serial_ports()
		if system == "Windows":
			reserved_names = {
				"CON", "PRN", "AUX", "NUL",
				*(f"COM{i}" for i in range(1, NumPorts)),
				*(f"LPT{i}" for i in range(1, NumPorts)),
			}
			# extrai só o último componente para checar nome reservado
			base = os.path.splitext(os.path.basename(name))[0].upper()
			if base in reserved_names:									return False

			illegal_chars = r'<>:"/\\|?*\0'
			for part in name.split(os.sep):
				if any(c in illegal_chars for c in part):				return False

		else:
			for part in name.split('/'):
				if '\0' in part:										return False

		return True

	def __ValidateBagfile(self):
		errorcode = 0;  hash1 = ''; hash2 = ''
		try:
			with open(self.bagfile_input, "rb") as f1:
				hash1 = telprocessor.__calc_hash(f1.read())
		except FileNotFoundError as e:											errorcode |= 0b000001
		except PermissionError as e:											errorcode |= 0b000010
		except OSError as e:													errorcode |= 0b000100
		try:
			with open(self.bagfile_in_telems, "rb") as f2:
				hash2 = telprocessor.__calc_hash(f2.read())
		except FileNotFoundError as e:											errorcode |= 0b001000
		except PermissionError as e:											errorcode |= 0b010000
		except OSError as e:													errorcode |= 0b100000

		if not hash1 and not hash2:												raise Exception(f"Failed to read bag file: {e}")
		samehash = (hash1 == hash2)		
		if self.bagfile_input_dir != self.telems_dir:
			if samehash:
				warn(f"Specified bag file is duplicated in '{self.telems_dir}' and will be used instead.")
				b = self.bagfile_in_telems
				h = hash2
			elif not hash1:
				b = self.bagfile_in_telems
				h = hash2
			elif not hash1:
				b = self.bagfile_input
				h = hash1
			else:
				print(f"A different bag file was found in {self.telems_dir} with the same name. Which bag should be used?")
				print(f"	1. {self.bagfile_in_telems}")
				print(f"	2. {self.bagfile_input}")
				whichbag = input()
				if whichbag!='2':
					b = self.bagfile_in_telems 
					h = hash2
				else:
					b = self.bagfile_input
					h = hash1
		else:
			if samehash:
				print(f'Bag file is not in the specified directory. Move to {self.telems_dir} ?')
				move = input('y/[n]')
				if move == 'y': 	
					try: 
						shutil.move(self.bagfile_input, self.bagfile_in_telems)
						b = self.bagfile_in_telems
						h = hash2
					except: 	
						warn('Move failed')
						b = self.bagfile_input
						h = hash1
		return b, h
	def __SaveTelemetry(self):
		savedata = {
			"baghash": self.baghash,
			"builderhash": self.builderhash,
			"tel": self.tel
		}

		with open(self.picklefile, "wb") as f:
			pickle.dump(savedata, f)
	def __LoadTelemetry(self):
		try: var = pickle.load(open(self.picklefile, 'rb'))
		except FileNotFoundError as e:										warn("Pickle file not found."); return False
		except pickle.UnpicklingError as e:									warn("Failed to unpickle the file."); return False
		except EOFError as e:												warn("Incomplete or corrupted pickle file."); return False
		except AttributeError as e:											warn("Error accessing saved object attributes."); return False
		except ImportError as e:											warn("Missing module required to load pickle."); return False
		except IndexError as e:												warn("Index error while loading pickle."); return False
		except TypeError as e:												warn("Type error while loading pickle."); return False
		if not isinstance(var, dict): 										warn("Object structure not recognized."); return False
		if 'baghash' not in var:                 							warn("Missing bag hash in pickle file."); return False
		if not isinstance(loadedbaghash := var['baghash'], str):			warn("Loaded bag file hash in unexpected format."); return False
		if 'builderhash' not in var:                 						warn("Missing builder hash in pickle file."); return False
		if not isinstance(loadedbuilderhash := var['builderhash'], str):	warn("Loaded builder hash in unexpected format."); return False
		if 'tel' not in var:                 								warn("Missing telemetry in pickle file."); return False
		if not isinstance(loadedtel := var['tel'], telclass):				warn("tel in unexpected format."); return False
		if not set(vars(loadedtel).keys()) == expected_fields: 				warn("tel fields do not match expected structure."); return False
		if not loadedbaghash == self.baghash:								warn("Mismatch detected in bag file hash."); return False
		if not loadedbuilderhash == self.builderhash:						warn("Mismatch detected in builder hash."); return False
		
		print('Found a valid .pkl file. Telemetry loaded successfully.')
		return loadedtel
	def get(self):
		"""
			Loads or creates a DataFrame variable converted from the specified bag file
		"""

		tel = self.__LoadTelemetry()
		if not tel: 
			warn("Telemetry build required.")
			self.tel = tel = self.builder(self.bagfile)
			print('Telemetry build finished.')
			self.__SaveTelemetry()
		return tel

	@abstractmethod
	def builder(self, bagfile: str) -> telclass:
		...