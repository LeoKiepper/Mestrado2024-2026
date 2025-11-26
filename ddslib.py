#region Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
import os
from matplotlib.font_manager import FontProperties, findfont
from sklearn.metrics import root_mean_squared_error
import inspect, ast, textwrap, warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import time
from datetime import timedelta
from tqdm.auto import tqdm
from typing import IO, Callable, List
import xgboost as xgb
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import matplotlib; matplotlib.use('QtAgg')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections.abc import Iterable, Hashable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np, pandas as pd
import torch
import torch.nn as nn
try: _has_torch = True
except Exception: _has_torch = False
import weakref
_param_domain_map = weakref.WeakKeyDictionary()
#endregion

#region Helper variables and functions
_STARTING_SCORE = float('-inf')
SCORE_FUNCTION = root_mean_squared_error
def compose_dataset_splitter(split_units: str, test_size: int|float, gap_size: int|float):
	"""
	Fabricate a dataset splitter function based on the specified units for
	``test_size`` and ``gap_size``.

	The split is done along the first dimension of ``X`` and ``y`` (i.e., indexes
	for 1-D arrays, rows for 2-D arrays, etc.). This makes the function suitable
	for sample-level splitting as well as sequence- or group-level splitting.

	:param split_units: Units for ``test_size`` and ``gap_size``. One of:

	    - ``'%'`` — Percent (0 to 100) of the total length of the dataset.
	    - ``'index'`` — Same units used by the dataset's ``DataFrame.`` or ``Series.index``.
	    - ``'positions'`` — Integer positions from `0` to ``len(dataset)``.

	:param test_size: Size of the test set, in the specified units.
	:param gap_size: Size of the gap between training and test sets, in the specified units.

	:return: A callable to perform the split, with the following signature:

	        ``train_idx, test_idx = split(X, y, n_splits)``

	        **Arguments**

	            • **X** — Features array-like.
	            • **y** — Labels/targets array-like. Must have the same length along the first dimension as **X**.
	            • **n_splits** — Number of splits to create. If *n_splits > 1*, the callable yields a generator of
	            • *(train_idx, test_idx)* tuples for cross-validation. If *n_splits == 1*, it returns a single tuple.

	        **Returns**

	            • **train_idx** — Integer indexes for the training set.  
	            • **test_idx** — Integer indexes for the test set.

	        **Raises**

	            • **ValueError** — If ``n_splits`` is not a positive integer.
	            • **TypeError** — If ``X`` is not a DataFrame when ``split_units`` is ``'index'``.  

	:raises ValueError: If ``split_units`` is not one of the recognized options.

	
	## Example usage
	
	.. code-block:: python
	    # For one-time fit:
		split = compose_dataset_splitter(split_units='%', test_size=20, gap_size=0)
	    X_train_idx, X_test_idx = split(X, y, n_splits=1)

	    # For cross-validation with multiple splits:
		split = compose_dataset_splitter(split_units='%', test_size=20, gap_size=0)
	    for train_idx, test_idx in split(X, y, n_splits=5):
	        print(train_idx, test_idx)

	"""

	N_SPLITS_VALUE_ERROR_MESSAGE = "n_splits must be a positive integer."
	from sklearn.model_selection import TimeSeriesSplit
	if split_units not in ['%','index','positions']:
		raise ValueError("Unrecognized key for split_units. Must be either '%' for a percent (0 to 100) of the total length of dataset, 'index' for the same units used for values in the dataset's DataFrame.index, or 'positions' for basic integer indexation from 0 to len(dataset).")
	if split_units == '%':
		def split_dataset(X: Iterable, y: Iterable, n_splits: int):
			if not isinstance(n_splits,int) or n_splits < 1: raise ValueError(N_SPLITS_VALUE_ERROR_MESSAGE)
			TEST_SIZE = round(len(X) * test_size / 100.0)
			GAP_SIZE = round(len(X) * gap_size / 100.0)
			if n_splits > 1:
				tss = TimeSeriesSplit(n_splits=n_splits, test_size=TEST_SIZE, gap=GAP_SIZE)
				return tss.split(X,y)
			else:
				test_idx = range(len(X) - TEST_SIZE, len(X))
				train_idx = range(len(X) - TEST_SIZE - GAP_SIZE)
				return train_idx, test_idx
	if split_units == 'positions':
		def split_dataset(X: Iterable, y: Iterable, n_splits:int):
			if not isinstance(n_splits,int) or n_splits < 1: raise ValueError(N_SPLITS_VALUE_ERROR_MESSAGE)
			TEST_SIZE = test_size
			GAP_SIZE = gap_size
			if n_splits > 1:
				tss = TimeSeriesSplit(n_splits=n_splits, test_size=TEST_SIZE, gap_size=GAP_SIZE)
				return tss.split(X,y)
			else:
				test_idx = range(len(X) - TEST_SIZE, len(X))
				train_idx = range(len(X) - TEST_SIZE - GAP_SIZE)
				return train_idx, test_idx
	if split_units == 'index':
		def split_dataset(X: Iterable, y: Iterable, n_splits:int):
			if not isinstance(X,pd.DataFrame):
				raise TypeError("If split_units = 'index', X must be a DataFrame")
			if not isinstance(n_splits,int) or n_splits < 1: raise ValueError(N_SPLITS_VALUE_ERROR_MESSAGE)
			t_end = X.index[-1]
			t_test_start = t_end - test_size
			t_gap_start = t_test_start - gap_size
			pos_test_start = X.index.searchsorted(t_test_start, side='right')
			pos_gap_start = X.index.searchsorted(t_gap_start, side='left')

			TEST_SIZE = len(X)-pos_test_start
			GAP_SIZE = pos_test_start-pos_gap_start
			if n_splits > 1:
				tss = TimeSeriesSplit(n_splits=n_splits, test_size=TEST_SIZE, gap_size=GAP_SIZE)
				return tss.split(X,y)
			else:
				test_idx = range(pos_test_start, len(X))
				train_idx = range(pos_test_start - pos_gap_start)
				return train_idx, test_idx
	return split_dataset
def apply_slice(iter, *slicer, along_dim: int=0, return_format: str='2d', flat_order: str ='row-major'):
	"""
	Apply one or more slicing or indexing operations to one or more array-like objects along a specified dimension.

	:param iter:  
		Single array-like object or an iterable of array-like objects to be sliced.  
		Supports ``numpy.ndarray``, ``pandas.DataFrame``, ``pandas.Series``, ``list``, and ``tuple``.

	:param slicer:  
		One or more 1d slicing/indexing objects. If multiple slicers are provided, each is applied independently along the specified dimension. Alternatively, a single container can be passed.

	:param along_dim:  
		Integer specifying the dimension (axis) along which the slicing will occur, counted from zero (e.g. ``along_dim=0`` for rows, ``along_dim=1`` for columns).

	:param return_format:  
		Specifies the structure of the returned result. One of:
		
			- ``'2d_explicit'`` — Always returns a list of lists.  
			- ``'2d'`` or ``'2d_reduced'`` — Returns a reduced-dimension result when possible: If ``iter`` is singleton, returns a list of objects sweeping along ``slicer``, If ``slicer`` is also a singleton, return the sliced object.  
			- ``'flat'`` — Returns a flat list of all results in row-major order.
	:param flat_order: Defines how elements are flattened when ``return_format='flat'`` is specified and return would otherwise be 2d.

			- ``'row'`` or ``'row-major'`` (default): groups by iterable — [A_s1, A_s2, B_s1, B_s2].
			- ``'col'`` or ``'column-major'``: groups by slice — [A_s1, B_s1, A_s2, B_s2].

	:return:  
		Sliced and indexed objects. If the returned object is 2d, iter sweeps row-wise and slicer sweeps column-wise.

	:raises AssertionError:  
		If ``along_dim`` exceeds the number of dimensions of any array-like in ``iter``.  

	:raises ValueError:  
		If ``return_format`` is not one of the recognized options.  

	:raises IndexError:  
		If an index, position, or mask is out of bounds for the given ``along_dim``.  

	:raises TypeError:  
		If any input type is not sliceable or the slicing operation is not supported.

	
	## Example usage

	.. code-block:: python

	    arr = np.arange(12).reshape(3, 4)

	    # Slice rows 1–2
	    res = apply_slice(arr, slice(1,3), along_dim=0, return_format='2d')
	    print(res)

	    # Slice columns 1–3
	    res = apply_slice(arr, slice(1,3), along_dim=1, return_format='2d')
	    print(res)

	    # Apply multiple slicers at once
	    res = apply_slice(arr, [slice(0,2), [2]], along_dim=0, return_format='2d_explicit')
	    print(res)

	    # With pandas
	    df = pd.DataFrame(np.arange(9).reshape(3,3), columns=list('ABC'))
	    res = apply_slice(df, slice(1,3), along_dim=0, return_format='2d_reduced')
	    print(res)

	    # Return flat list
	    res = apply_slice(arr, [slice(0,2), 2], along_dim=0, return_format='flat')
	    print(res)

	"""
	if return_format not in ('2d', '2d_reduced', '2d_explicit', 'flat'):
		raise ValueError("return_format must be '2d', '2d_reduced', '2d_explicit', or 'flat'")
	if return_format == 'flat' and flat_order not in ("row","col","row-major", "column-major"):
		raise ValueError("Invalid flat_order: expected 'row', 'row-major', 'col' or 'column-major'")

	import numpy as np, pandas as pd, collections.abc, builtins as _builtins
	from itertools import chain

	def _is_array_like(x):
		if _has_torch and isinstance(x, torch.Tensor): return True
		return isinstance(x, (pd.DataFrame, pd.Series, np.ndarray, list, tuple))
	def _cast_back(orig, res):
		if isinstance(orig, list):
			if isinstance(res, list): return res
			if isinstance(res, np.ndarray): return res.tolist()
			if isinstance(res, tuple): return list(res)
			return res
		if isinstance(orig, tuple):
			if isinstance(res, tuple): return res
			if isinstance(res, np.ndarray): return tuple(res.tolist())
			if isinstance(res, list): return tuple(res)
			return res
		if isinstance(orig, np.ndarray):
			return np.asarray(res)
		return res

	#region Normalize iter input and build metadata (single conversion per sequence) ---
	iterables = iter
	atomic_types = (pd.DataFrame, pd.Series, np.ndarray)
	if _has_torch:
		atomic_types = atomic_types + (torch.Tensor,)
	if not isinstance(iterables, collections.abc.Iterable) or isinstance(iterables, atomic_types):
		iterables = (iterables,)
	else:
		sample = None
		try:
			it = _builtins.iter(iterables)
			sample = next(it)
		except StopIteration:
			iterables = ()
		except TypeError:
			iterables = (iterables,)
		else:
			if not _is_array_like(sample):
				iterables = (iterables,)
	iter_meta = []
	for a in iterables:
		if isinstance(a, pd.DataFrame):
			iter_meta.append({'orig': a, 'kind': 'dataframe', 'shape': a.shape, 'ndim': 2})
		elif isinstance(a, pd.Series):
			iter_meta.append({'orig': a, 'kind': 'series', 'shape': (a.shape[0],), 'ndim': 1})
		elif _has_torch and isinstance(a, torch.Tensor):
			iter_meta.append({'orig': a, 'kind': 'torch', 'shape': tuple(a.size()), 'ndim': a.dim(), 'view': None})
		elif isinstance(a, np.ndarray):
			iter_meta.append({'orig': a, 'kind': 'ndarray', 'shape': a.shape, 'ndim': a.ndim, 'view': a})
		elif isinstance(a, (list, tuple)):
			# try to obtain a numpy view; allow copy if necessary
			try:
				aa = np.asarray(a, copy=False)
			except Exception:
				aa = np.asarray(a)
			iter_meta.append({'orig': a, 'kind': 'ndarray', 'shape': aa.shape, 'ndim': aa.ndim, 'view': aa})
		else:
			# last-resort attempt: try to convert to ndarray (allows copy)
			try:
				aa = np.asarray(a)
				iter_meta.append({'orig': a, 'kind': 'ndarray', 'shape': aa.shape, 'ndim': aa.ndim, 'view': aa})
			except Exception:
				raise TypeError(f"Unsupported iterable type: {type(a).__name__}")
	#endregion

	#region Validate along_dim
	if not isinstance(along_dim, int) or along_dim < 0:
		raise ValueError("along_dim must be a non-negative integer")
	for m in iter_meta:
		if along_dim >= m['ndim']:
			raise AssertionError(f"Invalid along={along_dim} for object with ndim={m['ndim']}")
	#endregion

	#region Normalize slicers (expand single container)
	if len(slicer) == 0:
		slicers = []
	elif len(slicer) == 1:
		s0 = slicer[0]
		if isinstance(s0, collections.abc.Iterable) and not isinstance(s0, (str, bytes, pd.Series, pd.DataFrame, np.ndarray)):
			# treat as container-of-slicers if elements look like indexers (include range)
			allowed = (slice, list, tuple, np.ndarray, int, np.integer, bool, range)
			if all(isinstance(el, allowed) for el in s0):
				slicers = list(s0)
			else:
				slicers = [s0]
		else:
			slicers = [s0]
	else:
		slicers = list(slicer)
	#endregion

	#region Convert range -> slice (no large allocations) and prelim normalize
	_norm_slicers = []
	for s in slicers:
		if isinstance(s, range):
			_norm_slicers.append(slice(s.start, s.stop, s.step))
			continue
		if isinstance(s, (list, tuple)):
			# convert to ndarray of objects first to detect heterogeneity, but avoid coercing numbers unnecessarily
			arr_obj = np.asarray(s, dtype=object)
			if arr_obj.ndim != 1:
				raise AssertionError(f"Only 1D slicers are supported, got slicer with ndim={arr_obj.ndim}")
			# check if all ints -> convert to numeric index array for faster bounds checks later
			if all(isinstance(x, (int, np.integer)) for x in arr_obj):
				_norm_slicers.append(np.asarray(arr_obj, dtype=np.intp))
			elif all(isinstance(x, (bool, np.bool_)) for x in arr_obj):
				_norm_slicers.append(np.asarray(arr_obj, dtype=bool))
			else:
				_norm_slicers.append(arr_obj)  # keep object array for label-style or mixed
			continue
		if isinstance(s, np.ndarray):
			if s.ndim != 1:
				raise AssertionError(f"Only 1D slicers are supported, got slicer with ndim={s.ndim}")
			if s.dtype == bool or s.dtype == np.bool_:
				_norm_slicers.append(s.astype(bool, copy=False))
			elif np.issubdtype(s.dtype, np.integer):
				_norm_slicers.append(s.astype(np.intp, copy=False))
			else:
				_norm_slicers.append(s)
			continue
		_norm_slicers.append(s)  # int, slice, label, bool scalar, etc.
	slicers = _norm_slicers
	#endregion

	#region Full negative-space combinatorial validation (no side-effects) --
	def _validate_indexer_for_shape(shape, indexer):
		axis_len = shape[along_dim]
		if isinstance(indexer, _builtins.slice):
			# slice always allowed (NumPy/pandas tolerate out-of-bounds for slices)
			return
		if isinstance(indexer, (int, np.integer)):
			ix = int(indexer)
			if ix < -axis_len or ix >= axis_len:
				raise IndexError(f"index {ix} out of bounds for axis {along_dim} with size {axis_len}")
			return
		if isinstance(indexer, np.ndarray):
			if indexer.dtype == bool:
				if indexer.ndim != 1:
					raise IndexError(f"boolean index must be 1-D, got ndim={indexer.ndim}")
				if indexer.shape[0] != axis_len:
					raise IndexError(f"boolean indexer length {indexer.shape[0]} does not match axis {along_dim} size {axis_len}")
				return
			if np.issubdtype(indexer.dtype, np.integer):
				if indexer.ndim != 1:
					raise IndexError(f"integer indexer must be 1-D, got ndim={indexer.ndim}")
				if indexer.size > 0:
					pos = indexer >= 0
					if pos.any() and int(indexer[pos].max()) >= axis_len:
						raise IndexError(f"integer index out of bounds for axis {along_dim} with size {axis_len}")
					if (~pos).any() and int(indexer[~pos].min()) < -axis_len:
						raise IndexError(f"integer index out of bounds for axis {along_dim} with size {axis_len}")
				return
		if isinstance(indexer, (list, tuple)):
			# convert to len-only check (no large allocations)
			try:
				idx_len = len(indexer)
			except Exception:
				pass
			else:
				# if boolean-like list, ensure lengths match
				if all(isinstance(x, (bool, np.bool_)) for x in indexer):
					if idx_len != axis_len:
						raise IndexError(f"boolean indexer length {idx_len} does not match axis {along_dim} size {axis_len}")
					return
				# if integer-like list, check bounds via min/max without converting to huge np.array
				if all(isinstance(x, (int, np.integer)) for x in indexer) and idx_len > 0:
					maxv = max(indexer)
					minv = min(indexer)
					if maxv >= axis_len or minv < -axis_len:
						raise IndexError(f"integer index out of bounds for axis {along_dim} with size {axis_len}")
					return
		# fallback: label-like (pandas .loc) allowed; no further validation here
		return
	for meta in iter_meta:
		for s in slicers:
			_validate_indexer_for_shape(meta['shape'], s)
	#endregion

	#region Central helper: build index tuple for numpy-like objects ---
	def _make_idx_tuple(shape, indexer):
		nd = len(shape)
		if along_dim >= nd:
			raise IndexError(f"along_dim={along_dim} out of bounds for ndim={nd}")
		idx = [slice(None)] * nd
		if isinstance(indexer, _builtins.slice):
			idx[along_dim] = indexer
			return tuple(idx)
		if isinstance(indexer, (int, np.integer)):
			ix = int(indexer)
			idx[along_dim] = ix
			return tuple(idx)
		if isinstance(indexer, np.ndarray):
			idx[along_dim] = indexer
			return tuple(idx)
		if isinstance(indexer, (list, tuple)):
			idx[along_dim] = np.asarray(indexer, dtype=np.intp)
			return tuple(idx)
		# boolean sequence stored as object-array maybe; try to coerce to bool array
		if isinstance(indexer, np.ndarray) and indexer.dtype == object:
			bo = np.asarray(indexer, dtype=bool)
			if bo.shape == (shape[along_dim],):
				idx[along_dim] = bo
				return tuple(idx)
		# fallback: treat as label-ish and return slice(None) placeholder (caller will use .loc/.iloc where appropriate)
		idx[along_dim] = indexer
		return tuple(idx)

	#region Single-slice apply helper (hot path)
	def _slice_single(meta, indexer):
		a = meta['orig']
		kind = meta['kind']
		if kind == 'dataframe':
			if isinstance(indexer, _builtins.slice):
				return a.iloc[indexer] if along_dim == 0 else a.iloc[:, indexer]
			if isinstance(indexer, (int, np.integer)):
				return a.iloc[[int(indexer)]] if along_dim == 0 else a.iloc[:, [int(indexer)]]
			# numpy-like or sequence indexers
			ia = indexer
			if isinstance(ia, (np.ndarray, list, tuple)):
				# boolean or integer arrays
				if isinstance(ia, np.ndarray) and ia.dtype == bool:
					return a.iloc[ia] if along_dim == 0 else a.iloc[:, ia]
				if np.issubdtype(np.asarray(ia).dtype, np.integer):
					return a.iloc[ia] if along_dim == 0 else a.iloc[:, ia]
			# fallback to label-based
			return a.loc[indexer] if along_dim == 0 else a.loc[:, indexer]

		if kind == 'series':
			if along_dim != 0:
				raise AssertionError("Series is 1D; along_dim must be 0")
			if isinstance(indexer, _builtins.slice):
				return a.iloc[indexer]
			if isinstance(indexer, (int, np.integer)):
				return a.iloc[[int(indexer)]]
			if isinstance(indexer, (np.ndarray, list, tuple)):
				if isinstance(indexer, np.ndarray) and indexer.dtype == bool:
					return a.iloc[indexer]
				if np.issubdtype(np.asarray(indexer).dtype, np.integer):
					return a.iloc[indexer]
			return a.loc[indexer]

		if kind == 'torch':
			# keep tensor on its native type, avoid .numpy() unless explicitly required later
			t = meta['orig']
			# simple slice along first axis
			if isinstance(indexer, _builtins.slice):
				return t[indexer]
			if isinstance(indexer, (int, np.integer)):
				return t[int(indexer)]
			# boolean mask
			if isinstance(indexer, np.ndarray) and indexer.dtype == bool:
				return t[torch.as_tensor(indexer, dtype=torch.bool, device=t.device)]
			# integer index array/list/tuple -> index_select along the axis
			if isinstance(indexer, (list, tuple, np.ndarray)):
				idx = torch.as_tensor(np.asarray(indexer, dtype=np.int64), dtype=torch.long, device=t.device)
				return torch.index_select(t, along_dim, idx)
			# fallback: raise to signal unsupported label-based indexing for torch
			raise TypeError("Label-based indexing not supported for torch.Tensor in apply_slice")

		# ndarray / sequence path
		arr = meta.get('view')
		if arr is None:
			arr = np.asarray(meta['orig'])
			meta['view'] = arr
		idx_tuple = _make_idx_tuple(arr.shape, indexer)
		res = arr[idx_tuple]
		return _cast_back(meta['orig'], res)

	# --- Build result quickly ---
	result = [ [ _slice_single(m, s) for s in slicers ] for m in iter_meta ]

	# --- Return formatting optimized ---
	if return_format == '2d_explicit':
		return [list(r) for r in result]
	if return_format in ('2d', '2d_reduced'):
		if len(iter_meta) == 1:
			if len(slicers) == 1:
				return result[0][0]
			# return list over slicers (single iterable)
			return [c for c in result[0]]
		else:
			return [list(r) for r in result]
	if return_format == "flat":
		if flat_order in ("row","row-major"):
			return [x for row in result for x in row]
		else:  # "col", "colum-major"
			return [result[r][c] for c in range(len(result[0])) for r in range(len(result))]
def sythesize_dataset(NoiseScale=0.075):
	def AddNoise(df,Pt,col,scale,ceil=1,floor=0):
		Noise=np.random.normal(scale=scale, size=df[col].shape)
		col=df.columns.get_loc(col)
		for i in range(1, len(Pt)):  # Começamos de 1 para termos um intervalo de t[i-1] a t[i]
			start = df.index.get_loc(Pt[i-1])  # Índice do início do trecho
			stop = df.index.get_loc(Pt[i])-1  # Índice do final do trecho

			c = df.iloc[start:stop, col]
			r = Noise[start:stop]
			val = c + r
			
			val_max=max(val)
			if ceil is not None and val_max > ceil:
				idmax = val.tolist().index(val_max)
				CorrectionFactor = (ceil - c.iloc[idmax]) / r[idmax]
			else: CorrectionFactor = 1
				# Aplicar o fator de correção no ruído do trecho
			df.iloc[start:stop, col] = df.iloc[start:stop, col] + r * CorrectionFactor
			if floor is not None: df.iloc[start:stop, col] = df.iloc[start:stop, col].clip(lower=0)
	def SynthCPUpercent(Pt, Pcpu, NoiseScale=0.075, TimeArrayParams=dict(tol=0.0001,MinStep=0.01,MinPoints=501,MaxPoints=3001)):
		def _frange(start,stop,**kwargs):
			opcoes={'step','num'}
			arg = opcoes.intersection(kwargs.keys())
			if len(arg) != 1:
				raise ValueError("Exatamente um dos argumentos 'step' ou 'num' deve ser passado.")
				
			from math import floor, log10
			if stop<start:
				aux=stop
				stop=start
				start=aux

			arg=arg.pop()
			match arg:
				case 'step':
					step = kwargs['step']

					# Calcula a ordem de grandeza para os cálculos
					gr=10**floor(log10(step))
					# Converte os argumentos para números inteiros
					iStart=start/gr
					iStep=step/gr
					iStop=stop/gr

					# Constrói o vetor
					NumElem=floor((iStop-iStart)/iStep)+1
					array=np.zeros(NumElem)
					num=iStart
					for i in range(NumElem):
						array[i]=num*gr
						num+=iStep

					return array
				case 'num':
					num = kwargs['num'] - 1
					delta = (stop-start)/num
					
					array = start+np.array(range(num+1))*delta
					return array
		def _GenerateTimeArray(t, tol, MinStep, MinPoints, MaxPoints):
			"""
			Para o intervalo [min(t), max(t)], tenta encontrar um vetor de tempos com número de pontos
			variando de min_points até max_points que satisfaça:
				1. Um passo constante, com valor >= min_step.
				2. Cada valor em t apareça no vetor (dentro de tol * max(1, |ti|)).
			
			Se encontrado, 'snap' os valores para que sejam exatamente t; caso contrário, levanta um erro.
			
			Parâmetros:
				- t: lista ou array de números (em ordem crescente) que devem aparecer no vetor final.
				- min_step: valor mínimo permitido para o passo entre os pontos.
				- tol: tolerância para considerar que um valor gerado é igual a um valor em t.
				- min_points: número mínimo de pontos a tentar.
				- max_points: número máximo de pontos permitidos no vetor.
			"""
			# Bloco de validação dos argumentos
			if not isinstance(t, (list, tuple, np.ndarray)):
				raise ValueError("t deve ser uma lista, tupla ou numpy array.")
			if len(t) == 0:
				raise ValueError("t não pode ser vazio.")
			# Verifica se todos os elementos de t são numéricos
			for ti in t:
				if not isinstance(ti, (int, float)):
					raise ValueError("Todos os elementos em t devem ser números.")
			# Verifica se os valores de t estão em ordem crescente
			if any(t[i] > t[i+1] for i in range(len(t)-1)):
				raise ValueError("Os valores em t devem estar em ordem crescente.")
			
			if not (isinstance(MinStep, (int, float)) and MinStep > 0):
				raise ValueError("min_step deve ser um número positivo.")
			if not (isinstance(tol, (int, float)) and tol >= 0):
				raise ValueError("tol deve ser um número não negativo.")
			if not (isinstance(MinPoints, int) and MinPoints > 0):
				raise ValueError("min_points deve ser um inteiro positivo.")
			if not (isinstance(MaxPoints, int) and MaxPoints >= MinPoints):
				raise ValueError("max_points deve ser um inteiro maior ou igual a min_points.")
			if MinPoints < len(t):
				raise ValueError("min_points deve ser pelo menos o número de valores em t.")
			
			start = min(t)
			stop = max(t)
			if (stop - start) / (MinPoints - 1) < MinStep:
				raise ValueError("Parâmetros conflitantes: usando min_points pontos, o step resultante é menor que min_step.")    # Fim da validação dos argumentos

			found = False
			candidate_time_vec = None

			for n in range(MinPoints, MaxPoints + 1):
				# O passo candidato, se usarmos n pontos, seria:
				candidate_step = (stop - start) / (n - 1)
				if candidate_step < MinStep:
					# Se o step resultante for menor que o mínimo, esse candidato é inválido
					continue
				# Gera o vetor usando a opção 'num' de frange (garante exatamente n pontos)
				time_vec = _frange(start, stop, num=n)
				# Verifica se cada valor em t está presente (dentro da tolerância)
				all_found = True
				for ti in t:
					if np.abs(time_vec - ti).min() > tol * max(1, abs(ti)):
						all_found = False
						break
				if all_found:
					# "Snap": força os valores que estão dentro da tolerância a serem exatamente t
					candidate_time_vec = time_vec.copy()
					for ti in t:
						idx = np.argmin(np.abs(candidate_time_vec - ti))
						if np.abs(candidate_time_vec[idx] - ti) <= tol * max(1, abs(ti)):
							candidate_time_vec[idx] = ti
					found = True
					break

			if not found:
				raise ValueError(
					f"Não foi possível encontrar um vetor de tempos que inclua todos os valores de t (dentro da tolerância {tol}) "
					f"usando entre {MinPoints} e {MaxPoints} pontos e com step >= {MinStep}."
				)
			return candidate_time_vec
		def GenerateDF(t,CPU,TimeArrayParams):
			# Gera um DataFrame que contém os dados listados
			if TimeArrayParams is None:
				TimeArrayParams = {'min_step': 0.1, 'tol': 1e-8, 'min_points': 50, 'max_points': 200}
			
			aux=_GenerateTimeArray(t, **TimeArrayParams)
			df=pd.DataFrame(data={'t':aux,
								'CPU':np.zeros(len(aux)),
							})
			# Alimenta valores do vetor CPU em df, nas posições especificadas em t
			ind = df.loc[df['t'].isin(t)].index[:len(CPU)]
			df.loc[ind, 'CPU'] = CPU[:len(ind)]

			# Preenche valores de df
			t_stop = t[-2]  # Penúltimo valor de t

			last_value = None
			for i in reversed(df.index):  # Percorre de trás para frente
				if i in ind:
					last_value = df.at[i, 'CPU']  # Atualiza o último valor encontrado
				elif last_value is not None and df.at[i, 't'] > t_stop:
					df.at[i, 'CPU'] = last_value  # Replica o valor
			last_value = None
			for i in df.index:  # Percorre do início ao fim
				if i in ind:
					last_value = df.at[i, 'CPU']  # Atualiza o último valor encontrado
				elif last_value is not None and df.at[i, 't'] <= t_stop:
					df.at[i, 'CPU'] = last_value  # Replica o valor

			df['t_mod'] = df['t']  # Cria uma nova coluna para armazenar o t ajustado
			pt_idx = 0  # Índice para percorrer o vetor Pt
			for i in range(len(df)):
				while pt_idx < len(Pt) - 1 and df.loc[i, 't'] >= Pt[pt_idx + 1]:  
					pt_idx += 1  # Atualiza o índice para o próximo intervalo
				df.loc[i, 't_mod'] = df.loc[i, 't'] - Pt[pt_idx]  # Ajusta t conforme Pt
			df.iloc[-1,-1]=df.iloc[-2,-1]+(df.iloc[-2,-1]-df.iloc[-3,-1])
			df=df.rename(columns={'t':'Timestamp'})
			df=df.set_index('Timestamp')
			# df=df.rename(columns={'t_mod':'t'})
			return df.drop(columns=['t_mod'])
		

		df = GenerateDF(Pt,Pcpu,TimeArrayParams)
		AddNoise(df,Pt,'CPU',NoiseScale)

		return df
	def SimulateTmep(df: pd.DataFrame,KCPU=4,KTemp=0.1,TauCPU=0.1,TauTemp=0.05, temp_ext: float=None):
		from math import exp

		if temp_ext is None: temp_ext = 0
		if 'Temp' not in df.columns: df['Temp'] = float(temp_ext)

		idx=df.columns.get_loc('CPU')
		idy=df.columns.get_loc('Temp')	

		temp_ext=df.iloc[0,idy]
		Dt=df.index[1]-df.index[0]
		for i in range(1,len(df)):
			temp_current=df.iloc[i-1,idy]
			cpu_current=df.iloc[i,idx]
			BETA_CPU = 1 - np.exp(-Dt/TauCPU)
			BETA_TEMP = 1 - np.exp(-Dt/TauTemp)

			FCPU = lambda cpu: cpu**2
			FTEMP = lambda temp_current, temp_ext: temp_current-temp_ext
			DeltaTempFromCPU  = KCPU * FCPU(cpu_current) * BETA_CPU
			DeltaTempFromTemp = KTemp * FTEMP(temp_current, temp_ext) * BETA_TEMP

			TempNext = temp_current + DeltaTempFromCPU - DeltaTempFromTemp
				
			df.iloc[i,idy]=float(TempNext)

		return df

	Pt   = [0  ,    10 ,    12  ,   15  ,   21  ,   30  ]       # Times
	Pcpu = [0.4,    0.3,    0.98,   0.21,   0.61,   0.61]       # Values
	df=SynthCPUpercent(Pt,Pcpu, NoiseScale=NoiseScale)									# Create a %CPU dataframe
	df=SimulateTmep(df,temp_ext=40)								# Calculate temperature values	
	AddNoise(df,Pt,'Temp',scale=1,ceil=None, floor=None)

	return df
def PlotSim(df, offset_sec_axis=30, fontsize_sec_axis=12, hide_sec_axis_line=False, offset_xlabel=50, fontsize_axis_labels=20, fontsize_ticklabels=16, save=None):
	import matplotlib.pyplot as plt
	import matplotlib.ticker as mticker
	from mpl_toolkits.axes_grid1 import host_subplot
	import mpl_toolkits.axisartist as AA

	t=df.index
	# Cria a figura e os eixos host e parasite
	plt.figure(figsize=(10, 5))
	host = host_subplot(111, axes_class=AA.Axes)
	parasite = host.twinx()
	parasite.axis["right"].toggle(all=True)		# Força a exibição do eixo direito do parasite (necessário com axisartist)
	host.set_zorder(2)							# Host (temperatura) acima
	parasite.set_zorder(1)						# Parasite (CPU) abaixo
	parasite.patch.set_visible(False)


	# Plots
	host.plot('Temp', data=df, linewidth=2, label='Resposta da temperatura', zorder=4)
	parasite.plot('CPU', data=df, linewidth=0.9, color='orange', label='Degraus da %CPU', zorder=3)


	# Eixo %CPU
	parasite.set_ylim(0, 1)
	parasite.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*100:.0f}'))


	# Eixo X secundário para os valores de t
	secax = host.secondary_xaxis('bottom')
	secax.set_xticks(t)
	secax.spines['bottom'].set_position(('outward', offset_sec_axis))
	secax.spines['bottom'].set_visible(not hide_sec_axis_line)
	sec_tick_labels = [rf'$t_{{{i}}} = {t_val:.1f}$' for i, t_val in enumerate(t)]
	secax.set_xticklabels(sec_tick_labels, fontsize=fontsize_sec_axis)
	secax.tick_params(axis='x', length=5, width=1, labelsize=fontsize_sec_axis)
	# for t_val in t:     parasite.axvline(x=t_val, color='#a0a0a0', linestyle='--', linewidth=1, zorder=2)   # Linhas verticais para cada valor de t


	# Axis labels
	host.axis["bottom"].label.set_text(r'Tempo $[s]$')
	host.axis["bottom"].label.set_fontsize(fontsize_axis_labels)
	host.axis["bottom"].label.set_pad(offset_xlabel)
	host.axis["left"].label.set_text(r'Temperatura $[°C]$')
	host.axis["left"].label.set_fontsize(fontsize_axis_labels)
	parasite.axis["right"].label.set_text(r'Utilização da CPU $[\%]$')
	parasite.axis["right"].label.set_fontsize(fontsize_axis_labels)


	# Tick labels
	host.axis["left"].major_ticklabels.set_fontsize(fontsize_ticklabels)
	host.axis["bottom"].major_ticklabels.set_fontsize(fontsize_ticklabels)
	parasite.axis["right"].major_ticklabels.set_fontsize(fontsize_ticklabels)


	# Fundo e grid
	host.set_facecolor('#fafafa')
	host.set_axisbelow(True)
	parasite.set_axisbelow(True)
	host.grid(which='major', color='#e0e0e0', linewidth=1.5)					# Major grid para o eixo X e Y do host
	host.xaxis.grid(True, which='major', color='#e0e0e0', linewidth=1.5)		# Força o grid major no eixo X do host


	# Legenda
	handles_host, labels_host = host.get_legend_handles_labels()
	handles_par, labels_par = parasite.get_legend_handles_labels()
	legend_dict = {}
	for h, l in zip(handles_host + handles_par, labels_host + labels_par):	legend_dict[l] = h
	host.legend(list(legend_dict.values()), list(legend_dict.keys()), loc='upper left', fontsize=fontsize_axis_labels)


	if save is not None: plt.savefig(save, bbox_inches='tight')
	plt.show(block=False)
	return



class Param:
	from numbers import Number
	import weakref
	from collections import Counter
	_domains = weakref.WeakKeyDictionary()
	_behaviors = weakref.WeakKeyDictionary()
	_derive_inputs = weakref.WeakKeyDictionary()
	_derive_fn = weakref.WeakKeyDictionary()
	_types = weakref.WeakKeyDictionary()

	@property
	def domain(self) -> 'Param.Domain': return Param._domains.get(self)
	@property
	def behaviors(self) -> dict: return Param._behaviors.get(self)
	@behaviors.setter
	def behaviors(self, val: dict): Param._behaviors[self] = val
	@property
	def types(self) -> dict: return Param._types.get(self)
	@types.setter
	def types(self, val: dict): Param._types[self] = val
	@property
	def derive_inputs(self) -> list: return Param._derive_inputs.get(self)
	@derive_inputs.setter
	def derive_inputs(self, val: list): Param._derive_inputs[self] = val
	@property
	def derive_fn(self) -> callable: return Param._derive_fn.get(self)
	@derive_fn.setter
	def derive_fn(self, val: callable): Param._derive_fn[self] = val

	def _validate(self, params: dict, partial=False, reject_derived=False):
		"""
		Validate a mapping of parameter values against this Param instance.

		:param values:
			Dictionary mapping parameter names to candidate values. When the
			constructor was called with a names-list, these names come from that list.
		:param partial:
			If True, allow a subset of parameters (useful for updates/guesses).
		:param reject_derived:
			If True, raise if any provided names are flagged as derived.

		:raises TypeError:
			If ``values`` is not a dict.
		:raises AssertionError:
			On unrecognized parameter names, derived-parameter violations, or type mismatches.
		"""
		names = set(self.names())
		# prescribed = {n for n, v in self.behaviors.items() if v == self.Consts.FLAG_PRESCRIBED}
		derived = self.names(self.Utils.FLAG_DERIVED)

		input_names = set(params.keys())
		if reject_derived: assert input_names.isdisjoint(derived), f'Derived parameters not allowed here: {input_names & derived}'
		if partial: assert input_names.issubset(names), f'Unrecognized parameters: {input_names - names}'
		else: assert input_names ^ names == set(), f'Unrecognized parameters: {input_names ^ names}'

		if (bad := [k for k, v in params.items() if not Param.Utils.is_numeric(v)]) != []: raise TypeError(f"Non-numeric parameters: {bad}")
	def names(self, flag=None):
		"""
		Return parameter names known to this Param instance, optionally filtered by behavior.

		:param flag:
			One of ``None`` (return all names), ``Param.Consts.FLAG_PRESCRIBED``, or
			``Param.Consts.FLAG_DERIVED`` to return only those names with the given behavior.

		:return:
			A list of parameter names (strings).

		:notes:
			Used throughout the class to centralize the "masking" logic for prescribed vs derived names.
		"""
		names = [k for k in self.__dict__.keys() if not k.startswith('_')]
		if flag in (self.Utils.FLAG_DERIVED, self.Utils.FLAG_PRESCRIBED):
			names = [n for n in names if self.behaviors.get(n) == flag]
		return names
	def to_dict(self, flag=None): 
		"""
		Return parameter name→value mapping, optionally limited to a behavior flag.

		:param flag:
			See :meth:`names` for accepted values.

		:return:
			A shallow copy of the dict mapping parameter name -> current value for the selected names, as specified by **flag**.
		"""
		return {name: getattr(self, name) for name in self.names(flag)}.copy()
	def to_list(self, flag=None): 
		"""
		Same as :meth:`to_dict`, but returns a list of values.

		:param flag:
			See :meth:`names` for accepted values.

		:return:
			List of parameter values corresponding to the selected names, as specified by **flag**.
		"""		
		return [getattr(self, name) for name in self.names(flag)]
	def update(self, params, restrict=True, derive = True, hold_types=True):	
		"""
		Update one or more prescribed parameters onto instance parameters, run domain restriction and re-derive.

		:param params:
			Dict with the parameter names and new values. Derived parameters are rejected.

		:raises AssertionError:
			If provided keys are unknown or violate derived/prescribed rules or types.
		"""
		new_types = {name: type(value) for name, value in params.items()}
		self._validate(params, partial=True, reject_derived=True)
		for name, value in params.items(): setattr(self, name, value)
		if not hold_types: self.types.update(new_types)
		if restrict: self.domain.restrict()
		if derive: self._derive()
		return self
	def guess(self, names: list | None = None, derive = True, inplace = True):
		"""
		Assign guessed values to prescribed parameters using the active domain's guesser.

		:param names:
			Optional list of parameter names to guess. If None, all prescribed parameters are guessed.
			When provided, only names that are prescribed and recognized are used.

		:raises AssertionError:
			If the domain is not bound, or if ``names`` contains unknown entries.
		:post:
			After successful call, ``self._derive()`` is invoked so derived parameters remain consistent.
		"""
		assert self.domain is not None, 'Domain not bound to parent.'
		if names is None:
			names = self.names(self.Utils.FLAG_PRESCRIBED)
		else:
			if not isinstance(names, list): raise TypeError('names is not a list')
			if any(not isinstance(n, str) for n in names): raise TypeError('names contain non-str entries')
			names = [n for n in names if n in self.names(self.Utils.FLAG_PRESCRIBED)]
		limits = self.domain.limits
		new = {n: self.types[n](self.domain._guesser(*limits[n])) for n in names}
		self._validate(new, partial=True, reject_derived=True)
		if inplace:
			for n, v in new.items(): setattr(self, n, v)

		derive_out = {}
		if derive: derive_out = self._derive(new,inplace)

		if inplace: return self
		else: return new | derive_out
	def _derive(self, new={}, inplace=True):
		"""
		Run the configured derive function to compute derived parameters.

		Behavior:
			Attempts three call styles in order: ``derive_fn(**kwargs)``, ``derive_fn(*args)``,
			and finally ``derive_fn(self)``. The derive function must return a dict mapping
			derived parameter names to values.

		:raises AssertionError:
			If the derive result is not a dict, contains unknown names, or returns values
			for parameters not flagged as derived.
		"""
		if inplace: input_kwargs = {n: new.get(n, getattr(self, n)) for n in self.derive_inputs}
		else: input_kwargs = {n: getattr(self, n) for n in self.derive_inputs}
		
		out = self.derive_fn(**input_kwargs)

		if inplace: 
			for k, v in out.items(): setattr(self, k, v)
		return out

	def __init__(self, params: dict|list, domain: 'Param.Domain' = None, derive_fn=None, derive_inputs=None, behaviors: dict|None=None, types: dict|None=None, derive_after_init: bool = False):
		"""
		Construct a Param object encapsulating Numeric-valued parameters.

		Two construction forms are supported:

		1. ``params`` is a **dict**: caller provides explicit initial values for each parameter.
		In this mode ``domain.limits``, ``domain.restricts`` and ``behaviors`` must
		be dicts whose keys are a subset of (or equal to) the parameter names.

		2. ``params`` is a **list** of names: caller provides parameter names only and the
		Domain must be provided with ``limits``/``restricts`` as lists (parallel to the
		prescribed subset of names). In this mode initial parameter values are drawn by
		calling the domain guesser for each prescribed name and all auxiliary structures
		are normalized to dicts internally, as though instantiation followed form 1.

		:param params:
			Dict mapping name->value or list of parameter names.
		:param domain:
			A Param.Domain instance describing limits/restrictions and providing a guesser.
		:param derive_fn:
			Optional callable used to compute derived parameters. See :meth:`_derive`.
		:param derive_inputs:
			Optional list of parameter names (subset of params) used as inputs to ``derive_fn``.

				- In Param instantiation form 1, domain.limits must be a dict of {str:tuple(Numeric,Numeric)}, with all keys flagged as 'prescribed'
				- In Param instantiation form 2, domain.limits must be a list of [tuple(Numeric,Numeric)], with all keys flagged as 'prescribed'
		:param behaviors:
			Optional mapping name -> ``FLAG_PRESCRIBED``|``FLAG_DERIVED``. When omitted, all
			parameters are treated as prescribed.

		:raises TypeError:
			If ``params`` has an unexpected type.
		:raises AssertionError:
			If provided dict/list shapes or behavior mappings are inconsistent with names.
		:post:
			After construction, internal maps (domain.limits, restricts, behaviors)
			are consistent dicts keyed by parameter name and the domain is bound to this Param.
		"""

		# Resolve params as instance attributes. If params is a list, no starting values are passed. Defer value generation for later.
		if isinstance(params, dict):
			params_was_passed_as_list = False
			names = params.keys()
		elif isinstance(params, list):
			params_was_passed_as_list = True
			names = params
			params = {name: None for name in names}
		else: raise TypeError('params is not a dict or list.')
		if any(not isinstance(name, str) for name in names): raise TypeError('params contains non-string names.')
		for name, val in params.items(): setattr(self, name, val)		# from this point self.names() will return correctly when no flags are passed


		# Resolve behaviors
		DEFAULT_BEHAVIORS = {name:self.Utils.FLAG_PRESCRIBED for name in names}
		if behaviors is None: behaviors = DEFAULT_BEHAVIORS
		else:
			if not isinstance(behaviors, dict): raise TypeError('When informed, behaviors must be a dict')
			assert (unrec := set(behaviors.keys()) - set(names))==set(), f"behaviors contains keys not present in names: {unrec}" 
			supported = (self.Utils.FLAG_PRESCRIBED,self.Utils.FLAG_DERIVED)
			if any(unsupported := set([v for v in behaviors.values() if v not in supported])):
				raise ValueError(f"Unsupported flag found in behaviors.values: {unsupported}")
			behaviors = DEFAULT_BEHAVIORS | behaviors
		Param._behaviors[self] = behaviors					# from this point self.names() is able to apply flags to filter names

		if domain is None: domain = Param.Domain(
			limits={name:Param.Utils.UNDETERMINED_LIMIT for name in names}, 
			restricts={name:Param.Utils.IDENTITY_RESTRICT for name in names})
		domain._delayed_domain_init_routines(self)


		# Resolve derive_inputs
		if derive_inputs is None: derive_inputs = []
		else:
			if not isinstance(derive_inputs, list): raise TypeError('derive_inputs must be a list')
			if any(not isinstance(n, str) for n in derive_inputs): raise TypeError('derive_inputs contains non-str entries')
			dup = [x for x in derive_inputs if derive_inputs.count(x) > 1]
			assert dup == [], f'Found duplicated derive_inputs: {dup}'
			assert (diff := set(derive_inputs) - set(self.names()))==set(), f"derive_inputs contains entries not present in names: {diff}"
		Param._derive_inputs[self] = derive_inputs

		# Resolve initial types
		if types is None:
			# domain constructor validates _guesser signature, no need to check.
			guesser_type = type(domain._guesser(0,1))
			if params_was_passed_as_list: types = {name: guesser_type for name in names}
			else:
				types = {}
				for name in names:
					if params[name] is not None: types[name] = type(params[name]); continue
					if domain.limits[name] != self.Utils.UNDETERMINED_LIMIT: types[name] = type(domain.limits[name][0]); continue
					types[name] = guesser_type
		elif isinstance(types, type):
			assert Param.Utils.is_numeric(types), "types informed as single type must be numeric"
			types = {name: types for name in names}
		else:
			if not isinstance(types, dict): raise TypeError('When informed, types must be a dict or single type')
			assert (unrec := set(types.keys()) - set(names))==set(), f"types contains keys not present in names: {unrec}" 
			if (bad := [k for k, v in types.items() if not Param.Utils.is_numeric(v)])!=[]: raise TypeError(f"types contains non-numeric parameters: {bad}")
		Param._types[self] = types


		# Dry run to validate derive_fn signature
		if derive_fn is None: derive_fn = lambda **kwargs: {}
		else:
			test_kwargs = {name: value for name, value in self.guess(inplace=False,derive=False).items()}
			out = derive_fn(**test_kwargs)
			assert isinstance(out, dict), "derive_fn must return dict"
			assert (unrec := set(out.keys()) - set(self.names())) == set(), f"derive_fn returned unrecognized names: {unrec}"
			derived_set = self.names(self.Utils.FLAG_DERIVED)
			assert (der := set(out.keys())-set(derived_set))==set(), f"derive_fn returned values for names not marked as derived: {der}"
		Param._derive_fn[self] = derive_fn
		

		if params_was_passed_as_list: self.guess()
		if derive_after_init: self._derive()
	def __getitem__(self, key):
		if isinstance(key, str): return getattr(self, key)
		if isinstance(key, list): return {k: getattr(self, k) for k in key}
		raise TypeError('key must be str or list[str]')
	def __setitem__(self, key, value): self.update({key: value})
	def __iter__(self): return iter(self.names())
	def __len__(self): return len(self.__dict__)
	def __or__(self, other):
		if isinstance(other, dict): return self.__dict__ | other
		elif isinstance(other, Param): 
			new_values = self.__dict__.copy() | other.to_dict()
			seen=set()
			unique=[]
			for x in self.derive_inputs+other.derive_inputs:
				if x in seen and x in self.derive_inputs: continue
				seen.add(x)
				unique.append(x)
			new_derive_inputs = unique
			new_param_behaviors = self.behaviors | other.behaviors
			def new_derive_fn(**kwargs):
				out_1 = self.derive_fn(**{name: kwargs[name] for name in self.derive_inputs})
				out_2 = other.derive_fn(**{name: kwargs[name] for name in other.derive_inputs})
				return out_1 | out_2
			new_domain = self.domain | other.domain
			return Param(new_values, domain=new_domain, derive_fn=new_derive_fn, derive_inputs=new_derive_inputs, behaviors=new_param_behaviors)
		else: raise TypeError(f'Unsupported operand type(s) for |: {type(other).__name__}. Only dict and Param are supported.')


	class Utils:
		FLAG_DERIVED = 'derived'
		FLAG_PRESCRIBED = 'prescribed'
		UNDETERMINED_LIMIT = (0,1)
		IDENTITY_RESTRICT = lambda value, limits: value
		def is_numeric(val):
			from numbers import Number
			return isinstance(val, Number) or (isinstance(val, type) and issubclass(val, Number))
	@dataclass
	class Domain:
		import numpy as np
		_parent: 'Param'
		_guesser: callable

		def restrict(self):
			"""
			Apply domain-specific restriction functions to the parent Param's prescribed values.

			Behavior:
				For every key in ``self.limits``, obtains the parent's current value and passes it
				and the corresponding limit tuple to the restrict function registered in
				``self.restricts`` (or the identity function if none). The resulting values are
				written back to the parent.

			:raises TypeError:
				If no parent has been set.
			"""
			if self._parent is None: return
			new = {}
			for name in self.limits.keys():
				val = getattr(self._parent, name)
				fn = self.restricts.get(name, Param.Utils.IDENTITY_RESTRICT)
				new[name] = fn(val, self.limits[name])
			for name, value in new.items(): setattr(self._parent, name, self._parent.types[name](value))

		#region Clarification for Param.__init__, Domain.__init__, _delayed_init_routines responsibility split
		# Param and Domain variables are resolved whenever it is first possible to do so in the
		# callstack without thematic overreach. See docstrings for specific information.
		# Construction callstack should look like (oldest first):
		# 	Domain.__init__() ->
		# 	Param.__init__() ->
		# 		...
		# 		_delayed_domain_init_routines()
		# 		...
		#endregion
		def _delayed_domain_init_routines(self, parent: 'Param'):
			"""
			Provides logic thematically belonging to Domain for delayed execution, which would normally
			belong in the constructor, but requires data that comes from a Param instance, created after
			Domain instantiation. Calling this from within Param constructor completes paired instantiation
			orchestration while neatly maintaining separation of concerns.
			
			Dependencies from Param constructor:
				- Fully functioning `parent.names()`

			Responsibilities:
				- Resolve `self.limits` and `self.restricts` to dicts containing {*limits*, *restricts*} for all parent.names. For *names* not informed in *limits* during instantiation, generate (-inf, inf) *limits* and identity function *restricts*
				- Assert coherence between (self.limits, self.restricts) and (parent.names, parent.behaviors) 


			:param parent:
				The Param instance to bind to.

			:raises TypeError:
				If ``parent`` is not a Param.
			:raises AssertionError:
				If list lengths or key sets do not match the parent's names / prescribed subset.
			"""
			if not isinstance(parent, Param): raise TypeError('parent is not Param.')
			names = parent.names()

			# Coherence between (self.limits or self._limits_list) and (self.restric_map or self._restrict_list) is asserted on instantiation
			# In both (self.limits, self._limits_list) and (self.restric_map, self._restrict_list), Domain constructor asserts that only one in each pair can be not None

			# resolve determined_limits as dict from either self.limits or self._limits_list
			determined_limits = {}
			if self.limits is not None: determined_limits.update(self.limits)
			elif self._limits_list is not None: determined_limits.update({n: t for n, t in zip(names, self._limits_list)})
			for name in set(names) - set(determined_limits.keys()): determined_limits.update({name:Param.Utils.UNDETERMINED_LIMIT})
			self.limits=determined_limits

			# Assert coherence between self.limits and parent.param_behavior. Determined limits can only be specified for params flagged as prescribed
			assert (failed := [n for n in parent.names(Param.Utils.FLAG_DERIVED) if self.limits[n] != Param.Utils.UNDETERMINED_LIMIT]) == [], \
				f"limits contains keys not flagged as 'prescribed': {failed}. Do not specify limits for parameters flagged '{Param.Utils.FLAG_DERIVED}'."

			# resolve determined_restricts as dict from either self.restricts or self._restrict_list
			determined_restricts = {}
			if self.restricts is not None: determined_restricts.update(self.restricts)
			elif self._restrict_list is not None: determined_restricts.update({n: t for n, t in zip(names, self._restrict_list)})
			for name in set(names) - set(determined_restricts.keys()): determined_restricts.update({name:Param.Utils.IDENTITY_RESTRICT})
			self.restricts=determined_restricts

			# Assert coherence between (self.limits, self.restricts) and parent.names
			if self.limits is not None: assert (unrec := set(self.limits.keys()) - set(names))==set(), f"limits contains keys not present in parent.names: {unrec}" 
			if self.restricts is not None: assert (unrec := set(self.restricts.keys()) - set(names))==set(), f"restricts contains keys not present in parent.names: {unrec}" 

			self._parent = parent
			Param._domains[parent] = self
		def __init__(self, limits: dict[str, tuple]|list[tuple]|None=None, restricts: dict[str, callable]|list[callable]|None=None, guesser = np.random.uniform):
			"""
			Construct a Domain describing parameter limits, per-parameter restrictors and a guesser.
			Arguments must agree with Param instantiation form (see :meth:`Param.__init__`)

			:param limits:
				Either a dict or a list:

					- In Param instantiation form 1, must be a dict of `{ name: tuple(Numeric,Numeric) }`
					- In Param instantiation form 2, domain.limits must be a list of `[ tuple(Numeric,Numeric) ]`. Matching with parameters is position-based

				In either form, corresponding parameters must be flagged as ``FLAG_PRESCRIBED``
			:param restricts:
				Either a dict or a list:

					- In Param instantiation form 1, must be a dict of `{ name: callable(value, domain_limits) }`
					- In Param instantiation form 2, domain.limits must be a list of `[ callable(value, domain_limits) ]`. Matching with parameters is position-based

				In either form, corresponding parameters must be flagged as ``FLAG_PRESCRIBED``
			:param guesser:
				Callable used to sample initial or guessed values; default uses ``np.random.uniform``. **limits** are passed to guesser after unpacking (i.e. val = guesser(*limits))

			:raises AssertionError:
				If provided limits or restricts entries do not have the expected shapes/signatures.
			:post:
				List-style inputs are preserved temporarily as ``_limits_list`` / ``_restrict_list``
				and converted into dicts in :meth:`_set_parent`.
			"""
			# At the end of contructor execution, either self.limits or self._limits_list, and either self.restricts or self._restrict_list must be set, even if with all None


			if limits is not None: # Params is to be used with domain restriction
				if isinstance(limits, limits_type:=dict):
					if len(limits)>0: 
						if any(not isinstance(t, tuple) for t in limits.values()): raise TypeError('limits contains non-tuple entry.')
						if any(len(t) != 2 for t in limits.values()): raise ValueError('limits contains non 2-length tuple.')
					self.limits = limits
					self._limits_list = None
				elif isinstance(limits, limits_type:=list):
					if len(limits)>0: 
						if any(not isinstance(t, tuple) for t in limits): raise TypeError('limits list contains non-tuple entry.')
						if any(len(t) != 2 for t in limits): raise ValueError('limits list contains non 2-length tuple.')
					self._limits_list = limits
					self.limits = None
				else: raise TypeError('When informed, limits must be a dict or list.')

				# limits was passed and is validated. Need restricts
				if restricts is not None:
					def _validate_restricts_signature(fn):
						if callable(fn):
							y=fn(0.5,(0,1))
							if not Param.Utils.is_numeric(y): raise TypeError('restricts must return single numeric value')
							if not (0 <= y and y <= 1): raise ValueError("Unexpected signature in restricts")
						else: raise TypeError('restricts is not a callable')
						# if not callable(fn) or len(inspect.signature(fn).parameters) != 2: raise ValueError("Unexpected signature in restricts")
					if not callable(restricts) and not type(restricts) == limits_type : raise TypeError('When limits is informed, restricts must either be of same type or a singleton callable')
					if callable(restricts):
						_validate_restricts_signature(restricts)
						# replicate singleton callable into dict or list depending on limits type
						if limits_type == dict:
							self.restricts = {k: restricts for k in limits.keys()}
							self._restrict_list = None
						else: # limits_type == list
							self._restrict_list = [restricts for _ in limits]
							self.restricts = None
					elif isinstance(restricts, dict):
						assert (mismatch := set(restricts.keys()) ^ set(limits.keys())) == set(), f"keys from restricts and from limits don't agree. Mismatch: {mismatch}"
						for fn in restricts.values(): _validate_restricts_signature(fn)
						self.restricts = restricts
						self._restrict_list = None
					elif isinstance(restricts, list):
						for fn in restricts: _validate_restricts_signature(fn)
						self._restrict_list = restricts
						self.restricts = None
					else: raise TypeError('When informed, restirct_map must be a dict or list.')
				else: raise AttributeError('When limits is informed, restricts must also be informed')
			else: # limits is None: Param is to be used without domain restriction. Set all to None
				self.limits = None
				self._limits_list = None
				self.restricts = None
				self._restrict_list = None
				if restricts is not None: UserWarning('restricts was passed, but will be ignored because limits was not passed.')

			# dry run guesser to validate signature
			try: 
				if not Param.Utils.is_numeric(guesser(0,1)): raise ValueError('guesser must return numeric value')
			except Exception as e: raise ValueError('guesser has unexpected signature: must accept two numeric arguments (min, max)') from e
			self._guesser = guesser
		def __or__(self, other):
			if not isinstance(other, Param.Domain): raise TypeError(f'{other} is not a Param.Domain')
			# Merge limits
			new_limits = self.limits.copy()
			new_limits.update(other.limits)
			# Merge restricts
			new_restricts = self.restricts.copy()
			new_restricts.update(other.restricts)
			# Create new Domain
			return Param.Domain(limits=new_limits, restricts=new_restricts, guesser=other._guesser)


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
	layout: str = 'two-column'
	facecolor: str = '#fbfbfb'
	label_fontfamily = 'Times New Roman'
	time_unit_annotation: str = r' $[s]$'
	temperature_unit_annotation: str = r' $[°C]$'
	percent_unit_annotation: str = r' $[\%]$'
	time_unit_annotation: str = ' [s]'
	temperature_unit_annotation: str = ' [°C]'
	percent_unit_annotation: str = ' [%]'
	savefig_bbox_inches: str = 'tight'
	save_figure_extension: str = 'pdf' # This is used as the default extension, for when the filename does not provide one.
	save_with_title = False	# if True, the figure title is drawn before the figure is saved and will show on the saved file.

	# =================== Figure specific parameters ===================
	# figure specific strings that are initialized as empty
	# strings are to be set by the get_plotstlye function.
	training_score_history_xlabel: str = ''
	score_label: str = ''

	M1_training_score_history_title: str = ''
	M1_training_score_history_filename: str = 'M1_training_score_history'
	M1_training_score_history_savefig: bool = False

	M2_training_score_history_title: str = ''
	M2_training_score_history_filename: str = 'M2_training_score_history'
	M2_training_score_history_savefig: bool = False

	M1_prediction_title: str = ''
	M1_prediction_filename = 'M1_prediction'
	M1_prediction_savefig: bool = True

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

	full_prediction_title: str = ''
	full_prediction_filename: str = 'full_prediction'
	full_prediction_savefig: bool = True

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
		self.prediction_plot_options['label'] = {'en': 'Prediction', 'pt': 'Predição'}[lang]
		self.reference_plot_options['label'] = {'en': 'Reference', 'pt': 'Referência'}[lang]
		self.M1_training_score_history_title = {'en': 'M1 Score History', 'pt': 'Histórico do score (M1)'}[lang]
		self.M2_training_score_history_title = {'en': 'M2 Score History', 'pt': 'Histórico do score (M2)'}[lang]
		self.training_score_history_xlabel = {'en': 'Iteration', 'pt': 'Iteração'}[lang]
		self.xlabel_time = {'en': 'Time', 'pt': 'Tempo'}[lang] + self.time_unit_annotation
		self.ylabel_temperature = {'en': 'CPU Temperature', 'pt': 'Temperatura da CPU'}[lang] + self.temperature_unit_annotation 
		self.ylabel_temperature_diff = {'en': 'Temperature difference', 'pt': 'Diferença de temperatura'}[lang] + self.temperature_unit_annotation
		self.ylabel_cpu_load = {'en': 'CPU load', 'pt': 'Utilização da CPU'}[lang] + self.percent_unit_annotation
		self.M1_prediction_title = {'en': 'M1 prediction', 'pt': 'Predição M1'}[lang]
		self.M2_partial_prediction_title = {'en': 'M2 prediction', 'pt': 'Predição M2'}[lang]
		self.M3_partial_prediction_title = {'en': 'M3 prediction', 'pt': 'Predição M3'}[lang]
		self.M3_crossvalidation_title = {'en': 'M3 Cross-validation', 'pt': 'Validação cruzada M3'}[lang]
		self.full_dataset_title = {'en': 'Full length dataset','pt':'Dataset em toda duração'}[lang]
		self.clipped_dataset_title = {'en': 'Clipped length dataset','pt':'Dataset até o primeiro superaquecimento'}[lang]
		self.first_temp_peak_detail_title = {'en': 'Detailed view for detected high CPU segment','pt':'Detalhe do segmento de alta utilização de CPU'}[lang]
		self.CPU_load_feature_in_legend = {'en':'CPU load feature','pt':'Atributo da utilização da CPU'}[lang]
		self.full_prediction_title = {'en': 'Full length prediction', 'pt': 'Predição em todo o dataset'}[lang]
def get_plotstyle(publication):

	# Set publications specific parameter tweaks 
	match publication:
		case 'IEEE2025': 	
			ps = PlotStyle(layout='two-column')
			ps.localize(lang='en')
		case _:	# None of the above		
			ps = PlotStyle(layout='two-column')
			ps.localize(lang='en')

	ps.score_label = infer_score_label(SCORE_FUNCTION)

	return ps
def infer_score_label(score_fn):
	SCORING_LABELS = {
		'root_mean_squared_error': 'RMSE',
		'mean_absolute_error': 'MAE',
		'mean_squared_error': 'MSE',
		'mean_absolute_percentage_error': 'MAPE',
	}
	# check function name first
	func_name = getattr(score_fn, '__name__', None)
	if func_name:
		for key, label in sorted(SCORING_LABELS.items(), key=lambda x: -len(x[0])):
			if key.lower() == func_name.lower():
				return label
			if key.lower() in func_name.lower():
				return label
	source = inspect.getsource(score_fn)
	source_dedented = textwrap.dedent(source)
	tree = ast.parse(source_dedented)

	# check the functions source code
	for node in ast.walk(tree):
		if isinstance(node, ast.Call):
			if isinstance(node.func, ast.Name):
				name = node.func.id
			elif isinstance(node.func, ast.Attribute):
				name = node.func.attr
			else:
				continue
			for key, label in sorted(SCORING_LABELS.items(), key=lambda x: -len(x[0])):
				if key.lower() == name.lower():
					return label
				if key.lower() in name.lower():
					return label
	return "Score"
def is_valid_font(fontname):
    try:
        prop = FontProperties(family=fontname)
        fontpath = findfont(prop, fallback_to_default=False)
        return True
    except Exception:
        return False
class FunctionWrapper(ABC):
	_model: 'M1 | M2 | M3'		# type: ignore
	def __init__(self):
		"""
		Instantiate a wrapper object to provide a common ground access to fields belonging to 
		a model and its associated objects to the method specified in subclass definition.
		The model field is initiated as None and should be set at model instantiation.
		"""
		self._model = None
	def __call__(self):
		return

#endregion


class DDS(BaseEstimator, RegressorMixin):
	def _merged_output(self, seg: Iterable=None):
		input_seg = seg
		if input_seg==None: segs = range(len(self.segmentation_summary))
		elif not isinstance(input_seg, Iterable): segs =[input_seg]
		else: segs = input_seg
		ret = []
		for seg in segs:
			merge = {}
			for out in self.step_outputs[seg]:
				merge |= out
			ret.append(merge)
		return ret if len(segs)>1 else ret[0]
	def fit(self, df: pd.DataFrame, **kwargs):
		if not isinstance(df, pd.DataFrame): raise TypeError("df must be a DataFrame")
		# Segment dataset
		_, self.segmentation_summary = self.segmenter(df, **kwargs)

		self.models_in_pipe = set()
		# TODO: for now, train models with data from only the first segment where they are required. In general, the user may want to fit fusing data from multiple segments; research possible strategies
		trained_models = []
		self.step_outputs=[]
		for seg, segment in enumerate(self.segmentation_summary):
			segment_df = df.iloc[segment['pos_first']:segment['pos_last']]
			segment_dict={col: segment_df[col].to_numpy() for col in segment_df.columns}
			self.step_outputs.append([])
			for step, step_dict in enumerate(self.state_pipe_mapping[segment['state']]):
				X = {key:value for key,value in segment_dict.items() if key in step_dict['X']}				
				X_mat = np.column_stack([(X|self._merged_output(seg))[col] for col in step_dict['X']])
				if not callable(step_dict['op']):
					self.models_in_pipe.add(step_dict['op']) # set.add automatically rejects repeats
					if step_dict['op'] not in trained_models:
						y = (segment_dict|self._merged_output(seg))[step_dict['y']]
						step_dict['op'].fit(X_mat,y,y0=y[0])
						trained_models.append(step_dict['op'])
					fn_ret = step_dict['op'].predict(X_mat,y0=y[0])
					self.step_outputs[seg].append({step_dict['ret']:fn_ret})

				else:
					args = [(X|self._merged_output(seg))[col] for col in step_dict['X']]
					out = step_dict['op'](*args)
					self.step_outputs[seg].append({step_dict['ret']: np.asarray(out)})

		self._trained = True
	def predict(self, df: pd.DataFrame, **kwargs):
		if not isinstance(df, pd.DataFrame): raise TypeError("df must be a DataFrame")
		assert self._trained, "predict must be called after fitting"

		# Segment dataset
		_, self.segmentation_summary = self.segmenter(df, **kwargs)

		# TODO: for now, train models with data from only the first segment where they are required. In general, the user may want to fit fusing data from multiple segments; research possible strategies
		self.step_outputs=[]
		all_cols = []
		model_dispatch = []
		partials = []
		for seg, segment in enumerate(self.segmentation_summary):
			model_dispatch.append({'models':[]})
			partials.append({})
			segment_df = df.iloc[segment['pos_first']:segment['pos_last']]
			segment_dict={col: segment_df[col].to_numpy() for col in segment_df.columns}
			self.step_outputs.append([])
			for step, step_dict in enumerate(self.state_pipe_mapping[segment['state']]):
				X = {key:value for key,value in segment_dict.items() if key in step_dict['X']}
				X_mat = np.column_stack([(X|self._merged_output(seg))[col] for col in step_dict['X']])
				if not callable(step_dict['op']):
					model_name = step_dict['op'].__class__.__name__
					model_dispatch[seg]['models'].append(model_name)
					if seg==0: y_hat0 = None 
					else:
						y_hat0 = self._merged_output(seg-1)[self.predict_col][0]
					fn_ret = step_dict['op'].predict(X_mat,y0=y_hat0)
					self.step_outputs[seg].append({step_dict['ret']:fn_ret})
					partials[seg][model_name] = fn_ret
				else:
					args = [(X|self._merged_output(seg))[col] for col in step_dict['X']]
					out = step_dict['op'](*args)
					self.step_outputs[seg].append({step_dict['ret']: np.asarray(out)})
				if step_dict['ret'] not in all_cols: all_cols.append(step_dict['ret'])

		if self.predict_col in all_cols: all_cols.remove(self.predict_col)
		all_cols.append(self.predict_col)
	
		out_df = pd.DataFrame(np.nan, index=df.index, columns=all_cols)
		for seg,segment in enumerate(self.segmentation_summary):
			idx = df.index[segment['pos_first']:segment['pos_last']]
			for col,val in self._merged_output(seg).items():
				arr = np.asarray(val).ravel()
				if arr.shape[0] != len(idx): raise ValueError(f"length mismatch for col {col}: {arr.shape[0]} vs {len(idx)}")
				out_df.loc[idx, col] = arr
		df_out_segments = [
			out_df.iloc[seg['pos_first']:seg['pos_last']]
			for seg in self.segmentation_summary
		]
		df_in_segments=[]
		for ss, seg in enumerate(self.segmentation_summary):
			df_in_segments.append(df.iloc[seg['pos_first']:seg['pos_last'],:])
		
		return out_df, self.segmentation_summary, df_in_segments, df_out_segments, model_dispatch, partials

	@staticmethod
	def _validate_pipeline(pipe):
		if isinstance(pipe, dict): pipe = [pipe]
		if not isinstance(pipe, list): raise TypeError('pipe must be a list')
		if not len(pipe) > 0: raise ValueError("pipe must have at least one element")
		valid = {X:='X',y:='y',op:='op',ret:='ret'}
		for ss, step in enumerate(pipe):
			if not isinstance(step, dict): raise TypeError(f"Found non-dict element in step {ss}: {step}")
			if not (unrec:=set(step.keys())-valid)==set(): raise KeyError(f"Found unrecognized keys in step {ss}: {unrec}")
			INVALID_COLUMN_NAME_MSG = "Found invalid column name for key '{key}' in step {ss}: {value}"
			op_is_callable = callable(step[op])
			op_has_fit_and_predict = hasattr(step[op], 'fit') and callable(getattr(step[op], 'fit')) and hasattr(step[op], 'predict') and callable(getattr(step[op], 'predict'))
			if op_is_callable:
				if step[y] is not None: raise ValueError(f"Found not None item for key '{y}' in step {ss} with callable operator. Must be None in this case")
				if not isinstance(step[X], Iterable) or isinstance(step[X], (str,bytes)): step[X]=[step[X]]
				step[X] = tuple(step[X])
			elif op_has_fit_and_predict:
				if step[y] is None: raise ValueError(f"Found None item for key '{y}' in step {ss} with operator with fit and predict methods. Must not be a valid column name in this case")
				if not isinstance(step[y], Hashable): raise ValueError(INVALID_COLUMN_NAME_MSG.format(key=y, ss=ss, value=step[y]))
				if not isinstance(step[X], Iterable) or isinstance(step[X], (str,bytes)): step[X]=[step[X]]
			else: raise TypeError(f"Operator in step {ss} is neither callable nor has fit and predict methods")
			for name in step[X]:
				if not isinstance(name, Hashable): raise ValueError(INVALID_COLUMN_NAME_MSG.format(key=X, ss=ss, value=name))
			if not isinstance(step[ret], Hashable): raise ValueError(INVALID_COLUMN_NAME_MSG.format(key=ret, ss=ss, value=step[ret]))
		return pipe
	def __init__(self, target_col: Hashable, predict_col: Hashable, temp_amb: float, segmenter: tuple, state_pipe_mapping: dict):
		# Prediction column name validation
		if not isinstance(predict_col, Hashable): raise TypeError("target_col must be hashable")
		self.predict_col = predict_col
		# Target column name validation
		if not isinstance(predict_col, Hashable): raise TypeError("target_col must be hashable")
		self.target_col = target_col

		# Overheating detector validation
		if not isinstance(segmenter, tuple): raise TypeError(f"segmenter must be a tuple")
		if not len(segmenter) == 2: raise ValueError(f"segmenter must be a 2-tuple with a callable an a column name for df")
		if not isinstance(col:=segmenter[0], Hashable): raise TypeError(f"The first element in segmenter must be a single column name. {col} is not a valid column name")
		if not callable(fn:=segmenter[1]): raise TypeError(f"The second element in segmenter must be a callable")
		test_df = pd.DataFrame(data={'x':[0]*10})
		try:
			if not isinstance(ret:=fn(test_df,'x'), tuple): raise ValueError(f"segmenter function must return a 2-tuple")
		except TypeError as e:
			raise TypeError(f"segmenter function must take a DataFrame and a str as arguments. Test run raised: \n{e}")
		if not isinstance(ret[0],list): raise ValueError("segmenter function's return 1 must be a list containing integer indexes pointing to change points")
		UNEXPECTED_SEGMENTATION_FORMAT_MSG = "segmenter function's return 2 must be a list of dicts summarizing segmentation. See documentation for expected format"
		if not isinstance(ret[1],list): raise ValueError(UNEXPECTED_SEGMENTATION_FORMAT_MSG)
		if not isinstance(ret[1][0],dict): raise ValueError(UNEXPECTED_SEGMENTATION_FORMAT_MSG)
		if not (unrec:=set(ret[1][0].keys())-{'state','pos_first','pos_last','avg'})==set(): raise KeyError(f"Found unrecognized keys in segment dict: {unrec}"+UNEXPECTED_SEGMENTATION_FORMAT_MSG)
		self.segmenter = lambda df, **kwargs: fn(df, col, **kwargs)

		# Validate pipelines
		if any(invalid:=[not isinstance(state, str) for state in state_pipe_mapping.keys()]): raise TypeError(f"Found non-str state names: {invalid}")
		for state, pipe in state_pipe_mapping.items():
			try: state_pipe_mapping[state]=DDS._validate_pipeline(pipe)
			except (ValueError, TypeError, AssertionError) as e: raise type(e)(f"Error validating normal_pipe: {e}") from e
		self.state_pipe_mapping = state_pipe_mapping


		self._trained = False


class M1(BaseEstimator, RegressorMixin):
	def get_model_params(self):
		"""
		:return: The parameters of this model.
		"""		
		return self._kernel._params
	def fit(self, X, y, y0: float=None):
		"""
		Orchestrates the training process by validating inputs and coordinating the kernel and optimizer.
		:param X: Input features.
		:param y: Target values.
		"""
		X, y = check_X_y(X, y)
		from numbers import Real
		if y0 is not None and not isinstance(y0, Real): raise TypeError("temp0 must be a float-like")
		self._optimizer.fit(X, y, temp0=y0)
	def predict(self, X, y0: float=None):
		"""
		Predicts the target values using the trained model.
		:param X: Input features.
		:param plot: Whether to plot the predictions against the true values.
		:param against: True values to plot against the predictions. Only used if plot is True.
		"""
		X = check_array(X, ensure_2d=False, dtype=float)
		if self._optimizer._training_finished:
			from numbers import Real
			if y0 is not None and not isinstance(y0, Real): raise TypeError("temp0 must be a float-like")
			pred = self._kernel.predict(X, y0)
			return pred
		else:
			warnings.warn("Model is not trained yet. Call fit() before predict().", UserWarning)
		return 
	def score(self, X, y):
		"""
		Evaluates the model by validating inputs and delegating to the optimizer's score method.
		:param X: Input features.
		:param y: Target values.
		:return: Model score.
		"""
		X, y = check_X_y(X, y)
		return self._optimizer.score(X, y)

	@property
	def score_history(self):
		return self._optimizer.get_score_history()

	def __init__(self, kernel: 'DelayRegressionStrategy', optimizer: 'FirstOrderOptimizer', random_state = None, plotstyle: PlotStyle = None):
		"""
		:param kernel: Function, as a FunctionWrapper object, containing logic and calculations that will produce predictions.
		:param loss: Loss function, as a FunctionWrapper object, to be used in training the model.
		:param optimizer: Optimizer function, as a FunctionWrapper object, to be used in training the model.
		"""
		self._kernel = kernel;			self._kernel._model = self
		self._optimizer = optimizer;	self._optimizer._model = self
		
		self.plotstyle = plotstyle
		# Hyperparamaters
		self.random_state = random_state
	def get_params(self, deep=True):
		params = {
			'random_state': self.random_state,
		}
		# Kernel params
		kernel_params = self._kernel.get_params(deep=deep) if hasattr(self._kernel, "get_params") else {}
		for k, v in kernel_params.items():
			params[f"kernel__{k}"] = v
		# Optimizer params
		optimizer_params = self._optimizer.get_params(deep=deep) if hasattr(self._optimizer, "get_params") else {}
		for k, v in optimizer_params.items():
			params[f"optimizer__{k}"] = v
		return params
	def set_params(self, **params):
		kernel_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("kernel__")}
		optimizer_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("optimizer__")}
		if kernel_params and hasattr(self._kernel, "set_params"):
			self._kernel.set_params(**kernel_params)
		if optimizer_params and hasattr(self._optimizer, "set_params"):
			self._optimizer.set_params(**optimizer_params)
		return self
class DelayRegressionStrategy(FunctionWrapper):		
	def set_model_parameters(self, params: dict):
		self._params.update(params)
	def get_model_params(self) -> dict:
		"""
		Returns current M1 kernel model parameters.
		:return: Parameter values, organized in a dict.
		"""
		return self._params.to_dict(Param.Utils.FLAG_PRESCRIBED)
	def guess_params(self) -> dict:
		return self._params.guess(inplace=False)
	def predict(self, X, y0: float=None) -> List:
		r"""
		Calculates a prediction series using the parrameters currently saved in the object.

		:param cpu_series: Pandas Series containing CPU usage values over time.
		:return: Pandas Series containing the simulated temperature values over time.
		"""			
		X = check_array(X, dtype=float)
		params = self.get_model_params()
		delay = params['delay']
		wi = np.array([value for key, value in params.items() if key != 'delay'])

		X_proc = np.vectorize(self.FCPU)(X)  # transforma X inteiro
		# if y0 is None: y0 = self._temp0
		y_hat = np.full(len(X), y0)
		y_hat[delay+1:] = X_proc[delay+1:,:] @ wi
		return y_hat
	def __init__(self, FCPU: callable, params: Param):
		r"""
		Instantiates the M2Kernel object

		:param FCPU: unit transform functions from cpu% to heat.
		:param FTEMP: unit transform functions from temperature gradient to heat.
		:param params: Dictionary of parameters for the M2 kernel model, according to the base closed-form analytical function:

			TempNext = TempCurrent
			+ KCPU * FCPU(CPUCurrent) * (1 - exp(- t / TauCPU))
			- KTemp * FTEMP(TempCurrent, TempExt) * (1 - exp(- t / TauTemp))

			It is strongly recommended to use the CalculateParams method provided by this class to generate this dictionary.
		:param TempAmb: Ambient temperature, used as the initial temperature.
		:param Dt: Time step size as used in the dataset this model will target.
		"""			
		super().__init__()
		if not callable(FCPU): raise TypeError('FCPU must be a callable')
		self.FCPU = FCPU

		# self._temp0 = temp0

		if not isinstance(params, Param): raise TypeError('params must be a Param object')
		self._params = params

	def get_params(self, deep=True):
		params = {
			'noise_level': self.noise_level,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self
class DelayRegressionOptimizer(FunctionWrapper, BaseEstimator, RegressorMixin):
	class StopConditions:
		GLOBAL_MAX_ITERATIONS 		= 1 << 0
		GLOBAL_MIN_LOSS      		= 1 << 1
		GLOBAL_MAX_DURATION 		= 1 << 2
		STALE_PATH_AVG_GRADIENT  	= 1 << 3
		STALE_PATH_MAX_ITER			= 1 << 4

		@classmethod
		def compute_flags(cls, obj: 'FirstOrderOptimizer'):
			code = 0
			if obj.current_iteration == obj.max_iter:												code |= cls.GLOBAL_MAX_ITERATIONS
			if abs(obj._gradient_window[-1]) <= abs(obj.global_min_loss):								code |= cls.GLOBAL_MIN_LOSS
			if abs(obj._loss_gradient()) <= obj.avg_gradient_tol:									code |= cls.STALE_PATH_AVG_GRADIENT
			if obj.iterations_on_this_path == obj.max_iter_on_one_path:							code |= cls.STALE_PATH_MAX_ITER
			if (time.time() - obj._training_start_time) >= obj.training_duration.total_seconds():	code |= cls.GLOBAL_MAX_DURATION
			return code
		@classmethod
		def compose_training_stop_function(cls, selected_flags, composition) -> callable:
			if not cls.validate_flags(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def training_stop(obj: 'FirstOrderOptimizer'):
				raised_flags = cls.compute_flags(obj)
				watched_flags = selected_flags & (cls.GLOBAL_MAX_ITERATIONS | cls.GLOBAL_MIN_LOSS | cls.GLOBAL_MAX_DURATION)
				if		composition == 'any':
					cond = bool(raised_flags & watched_flags)
				else: # composition == 'all'
					other = watched_flags & ~cls.GLOBAL_MAX_ITERATIONS
					cond = bool((raised_flags & cls.GLOBAL_MAX_ITERATIONS) or ((raised_flags & other) == other))
				if cond:
					obj._training_stop_reason = cls._describe_flags(raised_flags)
					return raised_flags
				return 0
			return training_stop
		@classmethod
		def compose_stale_path_function(cls, selected_flags, composition) -> callable:
			if not cls.validate_flags(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def stale_path(obj: 'FirstOrderOptimizer'):
				raised_flags = cls.compute_flags(obj)
				watched_flags = selected_flags & (cls.STALE_PATH_AVG_GRADIENT | cls.STALE_PATH_MAX_ITER)
				if		composition == 'any':
					cond = bool(raised_flags & watched_flags)
				else: # composition == 'all'
					cond = ((raised_flags & watched_flags) == watched_flags)
				if cond:
					obj._stale_path_reason = cls._describe_flags(raised_flags)
					return raised_flags
				return 0
			return stale_path
		@classmethod
		def validate_flags(cls, stop_flags: int):
			attributes = inspect.getmembers(cls, lambda a: isinstance(a, int))
			max_value = sum(v for _, v in attributes)
			if stop_flags < 0 or stop_flags == 0 or stop_flags > max_value:
				raise ValueError("Invalid stop flags.")
			return True
		@classmethod
		def _describe_flags(cls, flags):
			flag_description = []
			if flags & cls.GLOBAL_MAX_ITERATIONS: 	flag_description.append("GLOBAL_MAX_ITERATIONS")
			if flags & cls.GLOBAL_MIN_LOSS:       	flag_description.append("GLOBAL_MIN_LOSS")
			if flags & cls.GLOBAL_MAX_DURATION:     flag_description.append("GLOBAL_MAX_DURATION")
			return ", ".join(flag_description)		
	def _reset_score_buffer(self):
		self._score_buffer = [_STARTING_SCORE] * self.max_iter
	def _reset_history(self):
		self._gradient_window = [(1.0 + i)*self.global_min_loss*10 for i in reversed(range(self.gradient_window_size))]
		self._reset_score_buffer()
		self.score_history = []
	def _set_best_model(self):
		self._best_model_params = self._model._kernel.get_model_params()
		self._best_score = self.current_score
		self._best_training_iter = self.current_iteration
	def get_best_model(self):
		return self._best_model_params, self._best_score, self._best_training_iter
	def fit(self, X, y, temp0: float=None):
		"""
		Fits the model by iterating over M2Kernel parameters.
		:param X: Input features.
		:param y: Target values.
		:return: self
		"""
		self.current_iteration = 0
		self.buffer_index = 0
		self.iterations_on_this_path = 0
		self._reset_history()
		self._set_best_model()

		max_delay = self._model._kernel._params.domain.limits['delay'][1]
		delays = range(max_delay+1)
		self.max_iter = max_delay
		self._temp0 = temp0
		# Initialize progress bar and progress functions. Feed training time across iterations to the progress function
		pbar, progress = self._start_pbar();	self._training_start_time = (last := time.time())
		while not self.training_stop_condition(self):
			now = time.time()
			params = self.optimize(delays[self.current_iteration], X, y, temp0)

			self._model._kernel.set_model_parameters(params)

			self._update_score(X, y)
			
			# keep track of the best model found
			if self.current_score > self._best_score: 	self._set_best_model()

			self.current_iteration += 1
			self.buffer_index += 1
			self._update_pbar(pbar, progress(now,last)); last = now
		self._flush_score_buffer()
		pbar.close()
		self._model._kernel.set_model_parameters(self.get_best_model()[0])
		print("[INFO] Stop conditions met: ", self._training_stop_reason)
		print("[INFO] At iteration {iter:d}, got the best score of {score:.3f} with model params\n{model}".format(
			aux := self.get_best_model(),
			model = aux[0],
			score = aux[1],
			iter = aux[2],
		))
		self._training_finished = True
		return self
	def score(self, X, y):
		"""
		Custom scoring method using RMSELoss.
		:param X: Input features.
		:param y: Target values.
		:return: Negative RMSE (to align with sklearn's maximization convention).
		"""
		self._current_prediction = self._model._kernel.predict(X, y0 = self._temp0)
		rmse = SCORE_FUNCTION(y, self._current_prediction)
		return -rmse
	def optimize(self, delay: int, X, y, temp0: float=None) -> dict:
			"""
			Implements a single step of Stochastic Gradient Descent (SGD) to calculate the next set of candidate parameters.
			:param dict: Current parameters of the M2Kernel as a dict.
			:return: Updated parameters as a dict.
			"""

			x0 = X[0, :]
			if temp0 is None: temp0 = self._model._kernel._temp0
			y0 = temp0
			X = np.vectorize(self._model._kernel.FCPU)(X[delay+1:,:])
			y = y[:-delay-1]
			w0 = np.linalg.lstsq(X, y, rcond=None)[0]
			A = np.linalg.pinv(np.transpose(X) @ X)
			num = y0 - x0 @ w0
			den = x0 @ (A @ x0)
			w = w0 + (num / den) * (A @ x0)

			keys = ['delay'] + [f'w{i}' for i in range(len(w))]
			values = [delay] + list(w)
			return dict(zip(keys, values))
	def _update_score(self, X, y):
		self.current_score = self.score(X, y)
		self._gradient_window = (self._gradient_window + [self.current_score])[-self.gradient_window_size:]
		self._score_buffer[self.buffer_index] = self.current_score
	def _loss_gradient(self):
		return (self._gradient_window[-1] - self._gradient_window[0]) / self.gradient_window_size if self.gradient_window_size > 1 else self._gradient_window[0]
	def get_score_history(self):
		return np.concatenate(self.score_history + [self._score_buffer[:self.buffer_index]]).tolist()
	def _flush_score_buffer(self):		
		self.buffer_index = 0
		self.score_history.append(self._score_buffer.copy())
		self._reset_score_buffer()
		return

	def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, gradient_window=5, global_min_loss=1, score_tol=100, training_duration: timedelta = timedelta(minutes=1), \
				composition='any',
				training_stop_flags: int = StopConditions.GLOBAL_MAX_ITERATIONS | StopConditions.GLOBAL_MIN_LOSS,
				stale_path_flags: int = StopConditions.STALE_PATH_MAX_ITER | StopConditions.STALE_PATH_AVG_GRADIENT,
				plotstyle: PlotStyle = None
				):
		"""
		Initializes the Optimizer with a learning rate, maximum iterations, and tolerance.
		:param learning_rate: Step size for parameter updates.
		:param max_iter: Maximum number of iterations for training.
		:param tol: Tolerance for stopping criteria based on loss improvement.
		"""
		super().__init__()
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.global_min_loss = global_min_loss
		self.training_duration = training_duration
		self._training_start_time = None
		self.score_tol = score_tol
		self.current_iteration = 0
		self.current_score = _STARTING_SCORE
		self.score_history = []
		self._gradient_window=[]
		self._score_buffer=[]
		self._current_prediction=[]
		self._best_prediction=[]
		

		SC = self.StopConditions
		if not SC.validate_flags(training_stop_flags | stale_path_flags): raise ValueError("Failed to validate StopConditions flags.")
		if not (training_stop_flags & (SC.GLOBAL_MAX_ITERATIONS | SC.GLOBAL_MAX_DURATION)): training_stop_flags |= SC.GLOBAL_MAX_ITERATIONS
		self.training_stop_condition = SC.compose_training_stop_function(training_stop_flags,composition)
		self.stale_path_condition = SC.compose_stale_path_function(stale_path_flags,'any')
		self._training_stop_reason = None
		self._stale_path_reason = None

		# Set up progress bar
		pbar_opts = {'leave': True, 'desc': 'M1 fitting in progress'}
		self._max_duration_flag_set = bool(training_stop_flags & SC.GLOBAL_MAX_DURATION)
		self._pbar_iters = 0
		if self._max_duration_flag_set:
			total_sec = self.training_duration.total_seconds()
			if total_sec >= (denominator:=3600): unit = 'h'
			elif total_sec >= 2*(denominator:=60): unit = 'min'
			else: unit, denominator = 's', 1
			def _start_progress_bar():
				self._pbar_iters = 0
				progress_func = (lambda now, last: now - last) 
				return tqdm(**(pbar_opts | {'total': total_sec/denominator, 'unit': unit, 'bar_format': '{l_bar}{bar}| {n:.2f}/{total:.2f} {unit} | {postfix}'})), progress_func
			def _update_progress_bar(pbar: tqdm, progress: float):
				self._pbar_iters = getattr(self, '_pbar_iters', 0) + 1
				elapsed = (time.time() - self._training_start_time)
				it_per_s = self._pbar_iters / elapsed if elapsed > 0 else 0
				pbar.update(progress/denominator)
				pbar.set_postfix({'it': self._pbar_iters, 'it/s': f'{it_per_s:.2f}'})
				# pbar.update(progress/denominator)
				# pbar.set_postfix({'it/s': f'{1/progress:.2f}'})
		else:
			def _start_progress_bar():
				progress_func = (lambda now, last: 1)
				return tqdm(**(pbar_opts | {'total': self.max_iter})), progress_func
			def _update_progress_bar(pbar: tqdm, progress: float):
				pbar.update(progress)
		self._start_pbar, self._update_pbar = _start_progress_bar, _update_progress_bar
		self.iterations_on_this_path = 0
		self.max_iter_on_one_path = 100
		self.gradient_window_size = gradient_window
		self.avg_gradient_tol = tol

		self._best_model_params = None
		self._best_training_iter = 0
		self._best_score = _STARTING_SCORE

		self._training_finished = False
	def get_params(self, deep=True):
		params = {
			'learning_rate': self.learning_rate,
			'max_iter': self.max_iter,
			'tol': self.avg_gradient_tol,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self

class M2(BaseEstimator, RegressorMixin):
	def get_model_params(self):
		"""
		:return: The parameters of this model.
		"""		
		return self._kernel._params
	def fit(self, X, y, y0: float=None):
		"""
		Orchestrates the training process by validating inputs and coordinating the kernel and optimizer.
		:param X: Input features.
		:param y: Target values.
		"""
		X, y = check_X_y(X, y)
		from numbers import Real
		if y0 is not None and not isinstance(y0, Real): raise TypeError("temp0 must be a float-like")
		self._optimizer.fit(X, y, y0)
	def predict(self, X, y0: float=None):
		"""
		Predicts the target values using the trained model.
		:param X: Input features.
		:param plot: Whether to plot the predictions against the true values.
		:param against: True values to plot against the predictions. Only used if plot is True.
		"""
		X = check_array(X, ensure_2d=False, dtype=float)
		if self._optimizer._training_finished:
			from numbers import Real
			if y0 is not None and not isinstance(y0, Real): raise TypeError("temp0 must be a float-like")
			pred = self._kernel.predict(X, y0)
			return pred
		else:
			warnings.warn("Model is not trained yet. Call fit() before predict().", UserWarning)
		return 
	def score(self, X, y):
		"""
		Evaluates the model by validating inputs and delegating to the optimizer's score method.
		:param X: Input features.
		:param y: Target values.
		:return: Model score.
		"""
		X, y = check_X_y(X, y)
		return self._optimizer.score(X, y)

	@property
	def score_history(self):
		return self._optimizer.get_score_history()

	def __init__(self, kernel: 'FirstOrderStrategy', optimizer: 'FirstOrderOptimizer', random_state = None, plotstyle: PlotStyle = None):
		"""
		:param kernel: Function, as a FunctionWrapper object, containing logic and calculations that will produce predictions.
		:param loss: Loss function, as a FunctionWrapper object, to be used in training the model.
		:param optimizer: Optimizer function, as a FunctionWrapper object, to be used in training the model.
		"""
		self._kernel = kernel;			self._kernel._model = self
		self._optimizer = optimizer;	self._optimizer._model = self
		
		self.plotstyle = plotstyle
		# Hyperparamaters
		self.random_state = random_state
	def get_params(self, deep=True):
		params = {
			'random_state': self.random_state,
		}
		# Kernel params
		kernel_params = self._kernel.get_params(deep=deep) if hasattr(self._kernel, "get_params") else {}
		for k, v in kernel_params.items():
			params[f"kernel__{k}"] = v
		# Optimizer params
		optimizer_params = self._optimizer.get_params(deep=deep) if hasattr(self._optimizer, "get_params") else {}
		for k, v in optimizer_params.items():
			params[f"optimizer__{k}"] = v
		return params
	def set_params(self, **params):
		kernel_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("kernel__")}
		optimizer_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("optimizer__")}
		if kernel_params and hasattr(self._kernel, "set_params"):
			self._kernel.set_params(**kernel_params)
		if optimizer_params and hasattr(self._optimizer, "set_params"):
			self._optimizer.set_params(**optimizer_params)
		return self
class FirstOrderStrategy(FunctionWrapper):
	def set_model_parameters(self, params: dict):
		"""
			Sets parameters of the M2 kernel model.

			:param params: An instance of M2KernelParams containing the new parameter values.
		"""
		# Calculates the parameters for the M2 kernel, according to the base closed-form analytical function:
		# 	TempNext = TempCurrent
		# 	+ KCPU * FCPU(CPUCurrent) * (1 - exp(- t / TauCPU))
		# 	- KTemp * FTEMP(TempCurrent, TempExt) * (1 - exp(- t / TauTemp))
		self._params.update(params)
	def get_model_params(self) -> dict:
		"""
		Returns current M2 kernel model parameters.
		:return: Parameter values, organized in a Params object.
		"""
		return self._params.to_dict(Param.Utils.FLAG_PRESCRIBED)
	def guess_params(self) -> dict:
		return self._params.guess(inplace=False)
	def _next_temp(self, cpu_current, temp_current, temp_ext):
		r"""
		Computes the next temperature incrementally based on the current CPU usage, 
		previous temperature, and external temperature.
			Implements incremental form for the closed form analytic model:
				TempNext = TempPrev
				+ KCPU * FCPU(CPUCurrent) * (1 - np.exp(-t / tauCPU))
				- KTEMP * FTEMP(TempPrev, TempExt) * (1 - np.exp(-t / tauTEMP))

		:param CPUCurrent: Current CPU usage.
		:param TempCurrent: Current temperature.
		:param TempExt: External temperature.
		:return: Next temperature.
		"""
		KCPU, KTEMP, BETA_CPU, BETA_TEMP = tuple(self._params[['KCPU','KTemp','BetaCPU','BetaTemp']].values())
		FCPU = self.FCPU
		# FCPU = lambda cpu: cpu**2
		FTEMP = self.FTEMP
		# FTEMP = lambda TempPrev,TempExt: TempPrev - TempExt

		DeltaTempFromCPU  = KCPU * FCPU(cpu_current) * BETA_CPU
		DeltaTempFromTemp = KTEMP * FTEMP(temp_current, temp_ext) * BETA_TEMP

		return temp_current + DeltaTempFromCPU - DeltaTempFromTemp
	def predict(self, X, temp0: float=None) -> List:
		r"""
		Calculates a prediction series using the parrameters currently saved in the object.

		:param cpu_series: Pandas Series containing CPU usage values over time.
		:return: Pandas Series containing the simulated temperature values over time.
		"""			
		# X = check_array(X, ensure_2d=False, dtype=float)
		X = X.ravel()
		if temp0 is None: temp0 = self._temp0
		pred = [temp0] * len(X)
		for cc, cpu in enumerate(X[1:], start=1):
			pred[cc] = self._next_temp(cpu, pred[cc-1], self._temp_amb)
		return pred

	def __init__(self, FCPU: callable, FTEMP: callable, temp0: float, temp_amb: float, params: Param, noise_level=0):	# Dt test value was Dt = 30/510.0
		r"""
		Instantiates the M2Kernel object

		:param FCPU: unit transform functions from cpu% to heat.
		:param FTEMP: unit transform functions from temperature gradient to heat.
		:param params: Dictionary of parameters for the M2 kernel model, according to the base closed-form analytical function:

			TempNext = TempCurrent
			+ KCPU * FCPU(CPUCurrent) * (1 - exp(- t / TauCPU))
			- KTemp * FTEMP(TempCurrent, TempExt) * (1 - exp(- t / TauTemp))

			It is strongly recommended to use the CalculateParams method provided by this class to generate this dictionary.
		:param TempAmb: Ambient temperature, used as the initial temperature.
		:param Dt: Time step size as used in the dataset this model will target.
		"""			
		super().__init__()
		self.FCPU = FCPU
		self.FTEMP = FTEMP
		self._temp0 = temp0
		self._temp_amb = temp_amb
		self.noise_level = noise_level

		if not isinstance(params, Param): raise TypeError('params must be a Param object')
		self._params = params

	def get_params(self, deep=True):
		params = {
			'noise_level': self.noise_level,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self
class FirstOrderOptimizer(FunctionWrapper, BaseEstimator, RegressorMixin):
	class StopConditions:
		GLOBAL_MAX_ITERATIONS 		= 1 << 0
		GLOBAL_MIN_LOSS      		= 1 << 1
		GLOBAL_MAX_DURATION 		= 1 << 2
		STALE_PATH_AVG_GRADIENT  	= 1 << 3
		STALE_PATH_MAX_ITER			= 1 << 4

		@classmethod
		def compute_flags(cls, obj: 'FirstOrderOptimizer'):
			code = 0
			if obj.current_iteration == obj.max_iter:												code |= cls.GLOBAL_MAX_ITERATIONS
			if abs(obj._gradient_window[-1]) <= abs(obj.global_min_loss):								code |= cls.GLOBAL_MIN_LOSS
			if abs(obj._loss_gradient()) <= obj.avg_gradient_tol:									code |= cls.STALE_PATH_AVG_GRADIENT
			if obj.iterations_on_this_path == obj.max_iter_on_one_path:							code |= cls.STALE_PATH_MAX_ITER
			if (time.time() - obj._training_start_time) >= obj.training_duration.total_seconds():	code |= cls.GLOBAL_MAX_DURATION
			return code
		@classmethod
		def compose_training_stop_function(cls, selected_flags, composition) -> callable:
			if not cls.validate_flags(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def training_stop(obj: 'FirstOrderOptimizer'):
				raised_flags = cls.compute_flags(obj)
				watched_flags = selected_flags & (cls.GLOBAL_MAX_ITERATIONS | cls.GLOBAL_MIN_LOSS | cls.GLOBAL_MAX_DURATION)
				if		composition == 'any':
					cond = bool(raised_flags & watched_flags)
				else: # composition == 'all'
					other = watched_flags & ~cls.GLOBAL_MAX_ITERATIONS
					cond = bool((raised_flags & cls.GLOBAL_MAX_ITERATIONS) or ((raised_flags & other) == other))
				if cond:
					obj._training_stop_reason = cls._describe_flags(raised_flags)
					return raised_flags
				return 0
			return training_stop
		@classmethod
		def compose_stale_path_function(cls, selected_flags, composition) -> callable:
			if not cls.validate_flags(selected_flags) or composition not in ('any', 'all'):
				raise ValueError("Invalid compose arguments.")
			def stale_path(obj: 'FirstOrderOptimizer'):
				raised_flags = cls.compute_flags(obj)
				watched_flags = selected_flags & (cls.STALE_PATH_AVG_GRADIENT | cls.STALE_PATH_MAX_ITER)
				if		composition == 'any':
					cond = bool(raised_flags & watched_flags)
				else: # composition == 'all'
					cond = ((raised_flags & watched_flags) == watched_flags)
				if cond:
					obj._stale_path_reason = cls._describe_flags(raised_flags)
					return raised_flags
				return 0
			return stale_path
		@classmethod
		def validate_flags(cls, stop_flags: int):
			attributes = inspect.getmembers(cls, lambda a: isinstance(a, int))
			max_value = sum(v for _, v in attributes)
			if stop_flags < 0 or stop_flags == 0 or stop_flags > max_value:
				raise ValueError("Invalid stop flags.")
			return True
		@classmethod
		def _describe_flags(cls, flags):
			flag_description = []
			if flags & cls.GLOBAL_MAX_ITERATIONS: 	flag_description.append("GLOBAL_MAX_ITERATIONS")
			if flags & cls.GLOBAL_MIN_LOSS:       	flag_description.append("GLOBAL_MIN_LOSS")
			if flags & cls.GLOBAL_MAX_DURATION:     flag_description.append("GLOBAL_MAX_DURATION")
			return ", ".join(flag_description)		
	def _reset_score_buffer(self):
		self._score_buffer = [_STARTING_SCORE] * self.max_iter
	def _reset_history(self):
		self._gradient_window = [(1.0 + i)*self.global_min_loss*10 for i in reversed(range(self.gradient_window_size))]
		self._reset_score_buffer()
		self.score_history = []
	def _set_best_model(self):
		self._best_model_params = self._model._kernel.get_model_params()
		self._best_score = self.current_score
		self._best_training_iter = self.current_iteration
	def get_best_model(self):
		return self._best_model_params, self._best_score, self._best_training_iter
	def fit(self, X, y, temp0: float=None):
		"""
		Fits the model by iterating over M2Kernel parameters.
		:param X: Input features.
		:param y: Target values.
		:return: self
		"""
		self.current_iteration = 0
		self.buffer_index = 0
		self.iterations_on_this_path = 0
		self._reset_history()
		self._set_best_model()
		self._temp0 = temp0
		# Initialize progress bar and progress functions. Feed training time across iterations to the progress function
		pbar, progress = self._start_pbar();			last = time.time(); self._training_start_time = last

		while not self.training_stop_condition(self):
			now = time.time()
			
			if self.buffer_index == self.max_iter: self._flush_score_buffer() 
			try:		# calculate a prediction, compute score and update loss and score history
				with np.errstate(over='raise', invalid='raise'): 	self._update_score(X, y)
			except (ValueError, FloatingPointError):  # default to a new guessed model parameters
				self._model._kernel.set_model_parameters(self._model._kernel.guess_params())
				self.iterations_on_this_path = 0; continue
			
			# keep track of the best model found
			if self.current_score > self._best_score: 	self._set_best_model()

			# calculate params for next iteration
			if self.stale_path_condition(self): 		next_params = self._model._kernel.guess_params(); self.iterations_on_this_path = 0
			else: next_params = self.optimize(self._model._kernel.get_model_params())

			self._model._kernel.set_model_parameters(next_params)
			self.current_iteration += 1
			self.buffer_index += 1
			self.iterations_on_this_path += 1
			self._update_pbar(pbar, progress(now,last)); last = now
		self._flush_score_buffer()
		pbar.close()
		self._model._kernel.set_model_parameters(self.get_best_model()[0])
		print("[INFO] Stop conditions met: ", self._training_stop_reason)
		print("[INFO] At iteration {iter:d}, got the best score of {score:.3f} with model params\n{model}".format(
			aux := self.get_best_model(),
			model = aux[0],
			score = aux[1],
			iter = aux[2],
		))
		self._training_finished = True
		return self
	def score(self, X, y):
		"""
		Custom scoring method using RMSELoss.
		:param X: Input features.
		:param y: Target values.
		:return: Negative RMSE (to align with sklearn's maximization convention).
		"""
		self._current_prediction = self._model._kernel.predict(X, self._temp0)
		rmse = SCORE_FUNCTION(y, self._current_prediction)
		return -rmse
	def optimize(self, params: dict) -> dict:
			"""
			Implements a single step of Stochastic Gradient Descent (SGD) to calculate the next set of candidate parameters.
			:param params: Current parameters of the M2Kernel as a Params object.
			:return: Updated parameters as a Params object.
			"""
			KCPU, KTemp, TauCPU, TauTemp = tuple(params.values())

			grad = self._loss_gradient()
			grad_KCPU = 2 * grad
			grad_KTemp = 2 * grad
			grad_TauCPU = 2 * grad
			grad_TauTemp = 2 * grad

			KCPU -= self.learning_rate * grad_KCPU
			KTemp -= self.learning_rate * grad_KTemp
			TauCPU -= self.learning_rate * grad_TauCPU
			TauTemp -= self.learning_rate * grad_TauTemp

			# Return updated parameters as a Params object
			return dict(zip(params.keys(),[KCPU, KTemp, TauCPU, TauTemp]))
	def _update_score(self, X, y):
		self.current_score = self.score(X, y)
		self._gradient_window = (self._gradient_window + [self.current_score])[-self.gradient_window_size:]
		self._score_buffer[self.buffer_index] = self.current_score
	def _loss_gradient(self):
		return (self._gradient_window[-1] - self._gradient_window[0]) / self.gradient_window_size if self.gradient_window_size > 1 else self._gradient_window[0]
	def get_score_history(self):
		return np.concatenate(self.score_history + [self._score_buffer[:self.buffer_index]]).tolist()
	def _flush_score_buffer(self):		
		self.buffer_index = 0
		self.score_history.append(self._score_buffer.copy())
		self._reset_score_buffer()
		return

	def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, gradient_window=5, global_min_loss=1, score_tol=100, training_duration: timedelta = timedelta(minutes=1), \
				composition='any',
				training_stop_flags: int = StopConditions.GLOBAL_MAX_ITERATIONS | StopConditions.GLOBAL_MIN_LOSS,
				stale_path_flags: int = StopConditions.STALE_PATH_MAX_ITER | StopConditions.STALE_PATH_AVG_GRADIENT,
				plotstyle: PlotStyle = None
				):
		"""
		Initializes the Optimizer with a learning rate, maximum iterations, and tolerance.
		:param learning_rate: Step size for parameter updates.
		:param max_iter: Maximum number of iterations for training.
		:param tol: Tolerance for stopping criteria based on loss improvement.
		"""
		super().__init__()
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.global_min_loss = global_min_loss
		self.training_duration = training_duration
		self._training_start_time = None
		self.score_tol = score_tol
		self.current_iteration = 0
		self.current_score = _STARTING_SCORE
		self.score_history = []
		self._gradient_window=[]
		self._score_buffer=[]
		self._current_prediction=[]
		self._best_prediction=[]
		

		SC = self.StopConditions
		if not SC.validate_flags(training_stop_flags | stale_path_flags): raise ValueError("Failed to validate StopConditions flags.")
		if not (training_stop_flags & (SC.GLOBAL_MAX_ITERATIONS | SC.GLOBAL_MAX_DURATION)): training_stop_flags |= SC.GLOBAL_MAX_ITERATIONS
		self.training_stop_condition = SC.compose_training_stop_function(training_stop_flags,composition)
		self.stale_path_condition = SC.compose_stale_path_function(stale_path_flags,'any')
		self._training_stop_reason = None
		self._stale_path_reason = None

		# Set up progress bar
		pbar_opts = {'leave': True, 'desc': 'M2 fitting in progress'}
		self._max_duration_flag_set = bool(training_stop_flags & SC.GLOBAL_MAX_DURATION)
		self._pbar_iters = 0
		if self._max_duration_flag_set:
			total_sec = self.training_duration.total_seconds()
			if total_sec >= (denominator:=3600): unit = 'h'
			elif total_sec >= 2*(denominator:=60): unit = 'min'
			else: unit, denominator = 's', 1
			def _start_progress_bar():
				self._pbar_iters = 0
				progress_func = (lambda now, last: now - last) 
				return tqdm(**(pbar_opts | {'total': total_sec/denominator, 'unit': unit, 'bar_format': '{l_bar}{bar}| {n:.2f}/{total:.2f} {unit} | {postfix}'})), progress_func
			def _update_progress_bar(pbar: tqdm, progress: float):
				self._pbar_iters = getattr(self, '_pbar_iters', 0) + 1
				elapsed = (time.time() - self._training_start_time)
				it_per_s = self._pbar_iters / elapsed if elapsed > 0 else 0
				pbar.update(progress/denominator)
				pbar.set_postfix({'it': self._pbar_iters, 'it/s': f'{it_per_s:.2f}'})
				# pbar.update(progress/denominator)
				# pbar.set_postfix({'it/s': f'{1/progress:.2f}'})
		else:
			def _start_progress_bar():
				progress_func = (lambda now, last: 1)
				return tqdm(**(pbar_opts | {'total': self.max_iter})), progress_func
			def _update_progress_bar(pbar: tqdm, progress: float):
				pbar.update(progress)
		self._start_pbar, self._update_pbar = _start_progress_bar, _update_progress_bar
		self.iterations_on_this_path = 0
		self.max_iter_on_one_path = 100
		self.gradient_window_size = gradient_window
		self.avg_gradient_tol = tol

		self._best_model_params = None
		self._best_training_iter = 0
		self._best_score = _STARTING_SCORE

		self._training_finished = False
	def get_params(self, deep=True):
		params = {
			'learning_rate': self.learning_rate,
			'max_iter': self.max_iter,
			'tol': self.avg_gradient_tol,
		}
		return params
	def set_params(self, **params):
		for key, value in params.items():
			if hasattr(self, key): setattr(self, key, value)
		return self

class M3(BaseEstimator, RegressorMixin):
	def fit(self, X, y, split_units='%', test_size=20, gap_size=0, **kwargs):
		prefit_dict = {}
		if not self.strategy.defer_split:
			if not self.strategy.prefit_context['from_xval']:
				train_idx, test_idx = compose_dataset_splitter(split_units,test_size,gap_size)(X, y, 1)
				self.strategy.prefit_context.update({
					'train_idx': train_idx,
					'test_idx': test_idx,
					'X': X,
					'y': y,
					})
			prefit_dict = self.strategy.prefit(X, y, **kwargs)
		self.strategy.fit(X, y, **(({k: v for k, v in kwargs.items() if k != 'y0'}) | prefit_dict
))
		return self
	def predict(self, X, **kwargs):
		y_pred = self.strategy.predict(X, **{k: v for k, v in kwargs.items() if k != 'y0'})
		return y_pred
	def cross_validation(self, X, y, n_splits = 5, split_units = '%', test_size = 20, gap_size = 0, **kwargs):
		X, y = check_X_y(X, y)
		y = y.ravel()
		y_tests, y_preds, scores, test_pos = [], [], [], []
		if not self.strategy.defer_split:	# XGBStrategy does not defer split
			self.strategy.prefit_context.update({
				'from_xval': True,
				'X': X,
				'y': y,
				})
			for train_idx, test_idx in compose_dataset_splitter(split_units,test_size,gap_size)(X, y, n_splits):
				self.strategy.prefit_context.update({
					'train_idx': train_idx,
					'test_idx': test_idx,
					})
				X_train, y_train = X[train_idx], y[train_idx]
				X_test, y_test = X[test_idx], y[test_idx]
				self.fit(X_train, y_train, plot=False, **kwargs)
				pred = self.predict(X_test, plot=False)
				score = SCORE_FUNCTION(pred, y_test)
				y_tests.append(y_test)
				y_preds.append(pred)
				scores.append(score)
				test_pos.append(test_idx)
			self.strategy.prefit_context.update({
				'from_xval': False,
				})
		# else:	# Implement for RNNStrategy and others that defer split
		return test_idx, y_preds, scores

	def __init__(self, strategy: 'M3Strategy', plotstyle=None):
		self.strategy: M3Strategy = strategy
	def get_params(self, deep=True):
		return self.strategy.get_params(deep=deep)
	def set_params(self, **params):
		self.strategy.set_params(**params)
		return self
class M3Strategy(ABC, BaseEstimator, RegressorMixin):
	@abstractmethod
	def prefit(self, X, y, **kwargs): return {}
	@abstractmethod
	def fit(self, X, y, **kwargs): pass
	@abstractmethod
	def predict(self, X, **kwargs): pass
	@abstractmethod
	def get_params(self, deep=True): pass
	@abstractmethod
	def set_params(self, **params): pass
	def __init__(self, **kwargs):
		self.prefit_context: dict = {'from_xval': False}
		self.defer_split: bool = False
class XGBStrategy(M3Strategy):
	def prefit(self, X, y, **kwargs):
		train_idx = self.prefit_context.get('train_idx', None)
		test_idx = self.prefit_context.get('test_idx', None)
		X = self.prefit_context.get('X', [])
		y = self.prefit_context.get('y', [])
		if train_idx is None or test_idx is None:
			raise ValueError("train_idx and test_idx cannot be None. XGBoost requires defining an eval_set.")
		if len(X)==0 or len(y)==0:
			raise ValueError("X and y cannot be empty.")
		if len(X) != len(y):
			raise ValueError("X and y must have the same length.")
		X_train, y_train, X_test, y_test = apply_slice([X, y], [train_idx, test_idx], return_format='flat', flat_order='col')
		return {
			'eval_set':	[(X_train, y_train), (X_test, y_test)],
			}
	def fit(self, X, y, **kwargs):
		eval_set = kwargs.get('eval_set', None)
		if eval_set is None: raise ValueError("eval_set cannot be None. XGBoost requires defining an eval_set.")
		try: 
			verbose = int(kwargs.get('verbose', 0))
			if verbose <= 0 : verbose = False
		except (TypeError, ValueError): verbose = False
		self.reg.fit(X, y, eval_set=eval_set, verbose=verbose)
		return self
	def predict(self, X, **kwargs):
		return self.reg.predict(X, **kwargs)

	def __init__(self, n_estimators=1000, early_stopping_rounds=20, learning_rate=0.001, **kwargs):
		super().__init__(**kwargs)
		self.n_estimators = n_estimators
		self.early_stopping_rounds = early_stopping_rounds
		self.learning_rate = learning_rate
		self.reg = xgb.XGBRegressor(
			n_estimators=self.n_estimators,
			early_stopping_rounds=self.early_stopping_rounds,
			learning_rate=self.learning_rate,
			**kwargs
		)
		self.defer_split = False
	def get_params(self, deep=True):
		return self.reg.get_params(deep=deep)
	def set_params(self, **params):
		self.reg.set_params(**params)
		return self
class RNNStrategy(nn.Module, M3Strategy):
	def _create_sequences(self, X, y=None):
		sequences = []
		if y is not None: 
			targets = []
			append_y = lambda idx: targets.append(y[idx])
		else: append_y = lambda idx: None
		for i in range(len(X) - self.seq_length):
			sequences.append(X[i:i + self.seq_length])
			append_y(i + self.seq_length)
		ret = torch.tensor(np.array(sequences), dtype=torch.float32)
		if y is not None: ret = (ret, torch.tensor(np.array(targets), dtype=torch.float32))
		return ret
	def _resolve_batch_size(self, X_seq, y_seq=None, **kwargs):
		batch_size = kwargs.get('batch_size', None)
		if batch_size is not None: assert isinstance(batch_size, int) and batch_size > 0, 'When set, batch_size must be a positive integer.'
		mem_budget = kwargs.get('batch_memory_budget_MB', None)
		if mem_budget is not None: assert isinstance(mem_budget, (int, float)) and mem_budget > 0, 'When set, batch_memory_budget_MB must be a positive number.'

		if batch_size is not None: pass			
		elif mem_budget is not None:
			x_bytes = X_seq[0].nbytes
			y_bytes = y_seq[0].nbytes if y_seq is not None else 0
			bytes_hidden = self.num_layers * self.hidden_size * X_seq.shape[1] * X_seq.dtype.itemsize
			sample_bytes = x_bytes + y_bytes+ bytes_hidden
			batch_size = max(1, int(mem_budget)<<11 // sample_bytes) # int(mem_budget)<<11 converts MB to bytes, same as mem_budget * 1024**2
		
		self.batch_first = batch_size is not None
		return batch_size
	def _prepare_dataloader(self, X, y=None, **kwargs):
		batch_size = kwargs.get('batch_size', None)
		if batch_size is not None: assert isinstance(batch_size, int) and batch_size > 0, 'When set, batch_size must be a positive integer.'
		if y is not None: dataset = TensorDataset(X, y)
		else: dataset = TensorDataset(X)
		return DataLoader(dataset, batch_size=batch_size, shuffle=(y is not None))
	@staticmethod
	def _compose_last_sequence_extractor(batching: bool):
		"""
			Select slicing strategy to extract the last sequence from RNN output tensor. Must be called after _resolve_batch_size.
		"""
		assert isinstance(batching, bool), "batching must be a boolean value."
		if batching:
			def last(a):
				if isinstance(a, (tuple, list)): a = a[0]
				if a.ndim == 3: return a[:, -1, :]
				if a.ndim == 2: return a
				raise RuntimeError(f"unexpected network output ndim={a.ndim}")
		else:
			def last(a):
				if isinstance(a, (tuple, list)): a = a[0]
				if a.ndim == 3: return a[-1, :, :]
				if a.ndim == 2: return a[-1, :]
				raise RuntimeError(f"unexpected network output ndim={a.ndim}")
		return last

	def prefit(self, X, y, **kwargs): return {}
	def fit(self, X, y, **kwargs):
		X, y = check_X_y(X, y)
		#region data pre-processing
		self.feature_scaler=self.feature_scaler()
		self.target_scaler=self.target_scaler()
		X_scaled = self.feature_scaler.fit_transform(X)
		y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
		X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
		
		train_idx = self.prefit_context.get('train_idx', None)
		test_idx = self.prefit_context.get('test_idx', None)
		assert train_idx is not None and test_idx is not None, "Aborting. train_idx or test_idx are None."
		X_seq_train, y_seq_train, X_seq_test, y_seq_test = apply_slice([X_seq, y_seq], [train_idx, test_idx], along_dim=0, return_format='flat', flat_order='col')

		kwargs.update({'batch_size': self._resolve_batch_size(X_seq, y_seq, **kwargs)})
	
		train_loader = self._prepare_dataloader(X_seq_train, y_seq_train, **kwargs)
		test_loader = self._prepare_dataloader(X_seq_test, y_seq_test, **kwargs)
		#endregion

		#region model training
		sample_batch = next(iter(train_loader))
		xb = sample_batch[0] if isinstance(sample_batch, (tuple, list)) else sample_batch
		num_features = X.shape[1]
		self.network = self.network(
			input_size=num_features, 
			num_layers=self.num_layers,
			hidden_size=self.hidden_size, 
			batch_first=self.batch_first)
		self.connection = self.connection(self.hidden_size, 1)
		LEARNING_RATE = kwargs.get('learning_rate', None) # let optimizer treat errors from learning rate
		self.optimizer = self.optimizer(self.network.parameters(), lr=LEARNING_RATE)
		self.loss = self.loss()

		# training loop
		device = next(self.network.parameters()).device if any(True for _ in self.network.parameters()) else torch.device('cpu')
		self.network.to(device); self.connection.to(device)

		last = self._compose_last_sequence_extractor(self.batch_first)
		NUM_EPOCHS = kwargs.get('num_epochs', None)
		assert isinstance(NUM_EPOCHS, int) and NUM_EPOCHS > 0, "NUM_EPOCHS must be a positive integer."
		for epoch in tqdm(range(NUM_EPOCHS), desc=f"{self.network.__class__.__name__} training in progress"):
			train_loss_accum = 0
			for xb, yb in train_loader:
				pred, _ = self.network(xb)
				out = self.connection(last(pred)).squeeze(-1) 
				yb = yb.view(-1)
				loss = self.loss(out, yb)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				train_loss_accum += loss.item() * xb.size(0)

			test_loss_accum = 0
			with torch.no_grad():
				for xb, yb in test_loader:
					pred, _ = self.network(xb)
					out = self.connection(last(pred)).squeeze(-1)
					yb = yb.view(-1)
					loss = self.loss(out, yb)
					test_loss_accum += loss.item() * xb.size(0)

			# For future use, add logging of train and test loss per epoch
			# train_loss = train_loss_accum / len(train_loader.dataset)
			# test_loss = test_loss_accum / len(test_loader.dataset)
		#endregion
		self._trained = True
		return self
	def predict(self, X, **kwargs):
		X = check_array(X)
		X_scaled = self.feature_scaler.transform(X)
		X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
		kwargs.update({'batch_size': self._resolve_batch_size(X_seq, None, **kwargs)})
		
		pred_loader = self._prepare_dataloader(X_seq, None, **kwargs)

		device = next(self.network.parameters()).device if any(True for _ in self.network.parameters()) else torch.device('cpu')
		self.network.to(device); self.connection.to(device)
		self.network.eval(); self.connection.eval()

		last = self._compose_last_sequence_extractor(self.batch_first)
		preds = []
		with torch.no_grad():
			for batch in pred_loader:
				xb = batch[0] if isinstance(batch, (tuple, list)) else batch
				if not torch.is_tensor(xb): xb = torch.tensor(np.asarray(xb), dtype=torch.float32)
				xb = xb.to(device)
				out_net = self.network(xb)
				last_out = last(out_net)
				pred_batch = self.connection(last_out).squeeze(-1)
				if pred_batch.ndim == 2 and pred_batch.shape[1] == 1: pred_batch = pred_batch.squeeze(1)
				preds.append(pred_batch.cpu().numpy())

		if len(preds) == 0:
			return np.full(len(X), np.nan)  # fallback: no sequences -> return same-length array

		pred_seq = np.concatenate(preds, axis=0)  # length = len(X) - seq_length

		# produce output same length as input by repeating first prediction for initial positions
		L = X.shape[0]
		seq_len = self.seq_length
		if pred_seq.shape[0] != max(0, L - seq_len):
			# defensive: truncate or pad to expected sequence count
			expected = max(0, L - seq_len)
			if pred_seq.shape[0] > expected:
				pred_seq = pred_seq[:expected]
			else:
				pad_val = pred_seq[0] if pred_seq.size else 0.0
				pred_seq = np.concatenate([np.full(expected - pred_seq.size, pad_val), pred_seq], axis=0)

		full = np.empty(L, dtype=pred_seq.dtype)
		if seq_len > 0:
			full[:seq_len] = pred_seq[0]
		full[seq_len:] = pred_seq

		pred = self.target_scaler.inverse_transform(full.reshape(-1, 1)).flatten()
		return pred

	def forward(self, x):
		out, _ = self.network(x)
		out = out[:, -1, :]
		return self.connection(out).squeeze()

	def get_params(self, deep=True):
		return {
			'seq_length': self.seq_length,
			'hidden_size': self.hidden_size,
			'num_layers': self.num_layers,
			'epochs': self.epochs,
			'batch_size': self.batch_size,
			'lr': self.learning_rate,
			'model_filename': self.model_filename,
		}
	def set_params(self, **params):
		for k, v in params.items():
			if hasattr(self, k):
				setattr(self, k, v)
		return self
	def __init__(self, network=nn.RNN, connection = nn.Linear, optimizer = torch.optim.Adam, loss = nn.MSELoss, feature_scaler = StandardScaler, target_scaler = StandardScaler, seq_length=50, hidden_size=32, num_layers=1, epochs=20, batch_size=32, learning_rate=1e-3, **kwargs):
		super().__init__(**kwargs)
		M3Strategy.__init__(self, **kwargs)
		self.seq_length = int(seq_length)
		self.hidden_size = int(hidden_size)
		self.num_layers = int(num_layers)
		self.feature_scaler: StandardScaler | MinMaxScaler = feature_scaler
		self.target_scaler: StandardScaler | MinMaxScaler = target_scaler
		self.defer_split = False
		self._trained = False
		self.input_size: int = None
		self.network = network
		self.connection = connection
		self.optimizer = optimizer
		self.loss = loss
		self.batch_first = None


if __name__ == "__main__":
	def sythesize_dataset():
		def AddNoise(df,Pt,col,scale,ceil=1,floor=0):
			Noise=np.random.normal(scale=scale, size=df[col].shape)
			col=df.columns.get_loc(col)
			for i in range(1, len(Pt)):  # Começamos de 1 para termos um intervalo de t[i-1] a t[i]
				start = df.index.get_loc(Pt[i-1])  # Índice do início do trecho
				stop = df.index.get_loc(Pt[i])-1  # Índice do final do trecho

				c = df.iloc[start:stop, col]
				r = Noise[start:stop]
				val = c + r
				
				val_max=max(val)
				if ceil is not None and val_max > ceil:
					idmax = val.tolist().index(val_max)
					CorrectionFactor = (ceil - c.iloc[idmax]) / r[idmax]
				else: CorrectionFactor = 1
					# Aplicar o fator de correção no ruído do trecho
				df.iloc[start:stop, col] = df.iloc[start:stop, col] + r * CorrectionFactor
				if floor is not None: df.iloc[start:stop, col] = df.iloc[start:stop, col].clip(lower=0)
		def SynthCPUpercent(Pt, Pcpu, NoiseScale=0.075, TimeArrayParams=dict(tol=0.0001,MinStep=0.01,MinPoints=501,MaxPoints=3001)):
			def _frange(start,stop,**kwargs):
				opcoes={'step','num'}
				arg = opcoes.intersection(kwargs.keys())
				if len(arg) != 1:
					raise ValueError("Exatamente um dos argumentos 'step' ou 'num' deve ser passado.")
					
				from math import floor, log10
				if stop<start:
					aux=stop
					stop=start
					start=aux

				arg=arg.pop()
				match arg:
					case 'step':
						step = kwargs['step']

						# Calcula a ordem de grandeza para os cálculos
						gr=10**floor(log10(step))
						# Converte os argumentos para números inteiros
						iStart=start/gr
						iStep=step/gr
						iStop=stop/gr

						# Constrói o vetor
						NumElem=floor((iStop-iStart)/iStep)+1
						array=np.zeros(NumElem)
						num=iStart
						for i in range(NumElem):
							array[i]=num*gr
							num+=iStep

						return array
					case 'num':
						num = kwargs['num'] - 1
						delta = (stop-start)/num
						
						array = start+np.array(range(num+1))*delta
						return array
			def _GenerateTimeArray(t, tol, MinStep, MinPoints, MaxPoints):
				"""
				Para o intervalo [min(t), max(t)], tenta encontrar um vetor de tempos com número de pontos
				variando de min_points até max_points que satisfaça:
					1. Um passo constante, com valor >= min_step.
					2. Cada valor em t apareça no vetor (dentro de tol * max(1, |ti|)).
				
				Se encontrado, 'snap' os valores para que sejam exatamente t; caso contrário, levanta um erro.
				
				Parâmetros:
					- t: lista ou array de números (em ordem crescente) que devem aparecer no vetor final.
					- min_step: valor mínimo permitido para o passo entre os pontos.
					- tol: tolerância para considerar que um valor gerado é igual a um valor em t.
					- min_points: número mínimo de pontos a tentar.
					- max_points: número máximo de pontos permitidos no vetor.
				"""
				# Bloco de validação dos argumentos
				if not isinstance(t, (list, tuple, np.ndarray)):
					raise ValueError("t deve ser uma lista, tupla ou numpy array.")
				if len(t) == 0:
					raise ValueError("t não pode ser vazio.")
				# Verifica se todos os elementos de t são numéricos
				for ti in t:
					if not isinstance(ti, (int, float)):
						raise ValueError("Todos os elementos em t devem ser números.")
				# Verifica se os valores de t estão em ordem crescente
				if any(t[i] > t[i+1] for i in range(len(t)-1)):
					raise ValueError("Os valores em t devem estar em ordem crescente.")
				
				if not (isinstance(MinStep, (int, float)) and MinStep > 0):
					raise ValueError("min_step deve ser um número positivo.")
				if not (isinstance(tol, (int, float)) and tol >= 0):
					raise ValueError("tol deve ser um número não negativo.")
				if not (isinstance(MinPoints, int) and MinPoints > 0):
					raise ValueError("min_points deve ser um inteiro positivo.")
				if not (isinstance(MaxPoints, int) and MaxPoints >= MinPoints):
					raise ValueError("max_points deve ser um inteiro maior ou igual a min_points.")
				if MinPoints < len(t):
					raise ValueError("min_points deve ser pelo menos o número de valores em t.")
				
				start = min(t)
				stop = max(t)
				if (stop - start) / (MinPoints - 1) < MinStep:
					raise ValueError("Parâmetros conflitantes: usando min_points pontos, o step resultante é menor que min_step.")    # Fim da validação dos argumentos

				found = False
				candidate_time_vec = None

				for n in range(MinPoints, MaxPoints + 1):
					# O passo candidato, se usarmos n pontos, seria:
					candidate_step = (stop - start) / (n - 1)
					if candidate_step < MinStep:
						# Se o step resultante for menor que o mínimo, esse candidato é inválido
						continue
					# Gera o vetor usando a opção 'num' de frange (garante exatamente n pontos)
					time_vec = _frange(start, stop, num=n)
					# Verifica se cada valor em t está presente (dentro da tolerância)
					all_found = True
					for ti in t:
						if np.abs(time_vec - ti).min() > tol * max(1, abs(ti)):
							all_found = False
							break
					if all_found:
						# "Snap": força os valores que estão dentro da tolerância a serem exatamente t
						candidate_time_vec = time_vec.copy()
						for ti in t:
							idx = np.argmin(np.abs(candidate_time_vec - ti))
							if np.abs(candidate_time_vec[idx] - ti) <= tol * max(1, abs(ti)):
								candidate_time_vec[idx] = ti
						found = True
						break

				if not found:
					raise ValueError(
						f"Não foi possível encontrar um vetor de tempos que inclua todos os valores de t (dentro da tolerância {tol}) "
						f"usando entre {MinPoints} e {MaxPoints} pontos e com step >= {MinStep}."
					)
				return candidate_time_vec
			def GenerateDF(t,CPU,TimeArrayParams):
				# Gera um DataFrame que contém os dados listados
				if TimeArrayParams is None:
					TimeArrayParams = {'min_step': 0.1, 'tol': 1e-8, 'min_points': 50, 'max_points': 200}
				
				aux=_GenerateTimeArray(t, **TimeArrayParams)
				df=pd.DataFrame(data={'t':aux,
									'CPU':np.zeros(len(aux)),
								})
				# Alimenta valores do vetor CPU em df, nas posições especificadas em t
				ind = df.loc[df['t'].isin(t)].index[:len(CPU)]
				df.loc[ind, 'CPU'] = CPU[:len(ind)]

				# Preenche valores de df
				t_stop = t[-2]  # Penúltimo valor de t

				last_value = None
				for i in reversed(df.index):  # Percorre de trás para frente
					if i in ind:
						last_value = df.at[i, 'CPU']  # Atualiza o último valor encontrado
					elif last_value is not None and df.at[i, 't'] > t_stop:
						df.at[i, 'CPU'] = last_value  # Replica o valor
				last_value = None
				for i in df.index:  # Percorre do início ao fim
					if i in ind:
						last_value = df.at[i, 'CPU']  # Atualiza o último valor encontrado
					elif last_value is not None and df.at[i, 't'] <= t_stop:
						df.at[i, 'CPU'] = last_value  # Replica o valor

				df['t_mod'] = df['t']  # Cria uma nova coluna para armazenar o t ajustado
				pt_idx = 0  # Índice para percorrer o vetor Pt
				for i in range(len(df)):
					while pt_idx < len(Pt) - 1 and df.loc[i, 't'] >= Pt[pt_idx + 1]:  
						pt_idx += 1  # Atualiza o índice para o próximo intervalo
					df.loc[i, 't_mod'] = df.loc[i, 't'] - Pt[pt_idx]  # Ajusta t conforme Pt
				df.iloc[-1,-1]=df.iloc[-2,-1]+(df.iloc[-2,-1]-df.iloc[-3,-1])
				df=df.rename(columns={'t':'Timestamp'})
				df=df.set_index('Timestamp')
				# df=df.rename(columns={'t_mod':'t'})
				return df.drop(columns=['t_mod'])
			

			df = GenerateDF(Pt,Pcpu,TimeArrayParams)
			AddNoise(df,Pt,'CPU',NoiseScale)

			return df
		def PlotSim(df, offset_sec_axis=30, fontsize_sec_axis=12, hide_sec_axis_line=False, offset_xlabel=50, fontsize_axis_labels=20, fontsize_ticklabels=16, save=None):
			import matplotlib.pyplot as plt
			import matplotlib.ticker as mticker
			from mpl_toolkits.axes_grid1 import host_subplot
			import mpl_toolkits.axisartist as AA

			t=df['t'].values
			# Cria a figura e os eixos host e parasite
			plt.figure(figsize=(10, 5))
			host = host_subplot(111, axes_class=AA.Axes)
			parasite = host.twinx()
			parasite.axis["right"].toggle(all=True)		# Força a exibição do eixo direito do parasite (necessário com axisartist)
			host.set_zorder(2)							# Host (temperatura) acima
			parasite.set_zorder(1)						# Parasite (CPU) abaixo
			parasite.patch.set_visible(False)


			# Plots
			host.plot('Temp', data=df, linewidth=2, label='Resposta da temperatura', zorder=4)
			parasite.plot('CPU', data=df, linewidth=0.9, color='orange', label='Degraus da %CPU', zorder=3)


			# Eixo %CPU
			parasite.set_ylim(0, 1)
			parasite.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x*100:.0f}'))


			# Eixo X secundário para os valores de t
			secax = host.secondary_xaxis('bottom')
			secax.set_xticks(t)
			secax.spines['bottom'].set_position(('outward', offset_sec_axis))
			secax.spines['bottom'].set_visible(not hide_sec_axis_line)
			sec_tick_labels = [rf'$t_{{{i}}} = {t_val:.1f}$' for i, t_val in enumerate(t)]
			secax.set_xticklabels(sec_tick_labels, fontsize=fontsize_sec_axis)
			secax.tick_params(axis='x', length=5, width=1, labelsize=fontsize_sec_axis)
			# for t_val in t:     parasite.axvline(x=t_val, color='#a0a0a0', linestyle='--', linewidth=1, zorder=2)   # Linhas verticais para cada valor de t


			# Axis labels
			host.axis["bottom"].label.set_text(r'Tempo $[s]$')
			host.axis["bottom"].label.set_fontsize(fontsize_axis_labels)
			host.axis["bottom"].label.set_pad(offset_xlabel)
			host.axis["left"].label.set_text(r'Temperatura $[°C]$')
			host.axis["left"].label.set_fontsize(fontsize_axis_labels)
			parasite.axis["right"].label.set_text(r'Utilização da CPU $[\%]$')
			parasite.axis["right"].label.set_fontsize(fontsize_axis_labels)


			# Tick labels
			host.axis["left"].major_ticklabels.set_fontsize(fontsize_ticklabels)
			host.axis["bottom"].major_ticklabels.set_fontsize(fontsize_ticklabels)
			parasite.axis["right"].major_ticklabels.set_fontsize(fontsize_ticklabels)


			# Fundo e grid
			host.set_facecolor('#fafafa')
			host.set_axisbelow(True)
			parasite.set_axisbelow(True)
			host.grid(which='major', color='#e0e0e0', linewidth=1.5)					# Major grid para o eixo X e Y do host
			host.xaxis.grid(True, which='major', color='#e0e0e0', linewidth=1.5)		# Força o grid major no eixo X do host


			# Legenda
			handles_host, labels_host = host.get_legend_handles_labels()
			handles_par, labels_par = parasite.get_legend_handles_labels()
			legend_dict = {}
			for h, l in zip(handles_host + handles_par, labels_host + labels_par):	legend_dict[l] = h
			host.legend(list(legend_dict.values()), list(legend_dict.keys()), loc='upper left', fontsize=fontsize_axis_labels)


			if save is not None: plt.savefig(save, bbox_inches='tight')
			plt.show(block=False)
			return
		def SimulateTmep(df: pd.DataFrame,KCPU=4,KTemp=0.1,TauCPU=0.1,TauTemp=0.05, temp_ext: float=None):
			from math import exp

			if temp_ext is None: temp_ext = 0
			if 'Temp' not in df.columns: df['Temp'] = float(temp_ext)

			idx=df.columns.get_loc('CPU')
			idy=df.columns.get_loc('Temp')	

			temp_ext=df.iloc[0,idy]
			Dt=df.index[1]-df.index[0]
			for i in range(1,len(df)):
				temp_current=df.iloc[i-1,idy]
				cpu_current=df.iloc[i,idx]
				BETA_CPU = 1 - np.exp(-Dt/TauCPU)
				BETA_TEMP = 1 - np.exp(-Dt/TauTemp)

				FCPU = lambda cpu: cpu**2
				FTEMP = lambda temp_current, temp_ext: temp_current-temp_ext
				DeltaTempFromCPU  = KCPU * FCPU(cpu_current) * BETA_CPU
				DeltaTempFromTemp = KTemp * FTEMP(temp_current, temp_ext) * BETA_TEMP

				TempNext = temp_current + DeltaTempFromCPU - DeltaTempFromTemp
					
				df.iloc[i,idy]=float(TempNext)

			return df

		Pt   = [0  ,    10 ,    12  ,   15  ,   21  ,   30  ]       # Times
		Pcpu = [0.4,    0.3,    0.98,   0.21,   0.61,   0.61]       # Values
		df=SynthCPUpercent(Pt,Pcpu)									# Create a %CPU dataframe
		df=SimulateTmep(df,temp_ext=40)								# Calculate temperature values	
		AddNoise(df,Pt,'Temp',scale=1,ceil=None, floor=None)

		# PlotSim(df)

		def HighCPUDetect(df, margin=0,threshold=.8):
			"""
			Implements high cpu detection logic.
			
			:param cpu_percent: array containing CPU usage
			:param cp: array for change points detected in cpu_percent, listed as indexes for each position.
			Each position of cp is considered the last index of a segment (should not contain 0).
			:param margin: number of indexes to consider on each side of each segment investigated in the detection criterion implemented.
			:param threshold: consider high CPU usage if the average of the segment is greater than this value.

			:return segment: list of detected segments, formatted as dictionaries with the following keys:
			
			"""
			import ruptures as rpt
			cpu_percent=df['CPU'].values
			algo = rpt.Pelt(model="rbf").fit(cpu_percent)
			ChangePoints = algo.predict(pen=5)
			Segments=[]
			for cc, pos_up in enumerate(ChangePoints):
				pos_up=min(pos_up+margin,len(cpu_percent))
				pos_low=max(ChangePoints[cc-1]+1-margin,0) if cc>0 else 0
				
				
				avg=float(np.average(cpu_percent[pos_low:pos_up]))
				if avg>threshold: 	Segments.append({'state':'high','pos_low':pos_low,'pos_up':pos_up,'avg':avg})
				else:				Segments.append({'state':'norm','pos_low':pos_low,'pos_up':pos_up,'avg':avg})
			return ChangePoints, Segments
		return df
	df2 = sythesize_dataset()	# Synthesize a dataset


	# ==================  Define and instantiate model components
	m2obj=M2(
		FirstOrderStrategy(lambda cpu: cpu**2, lambda temp_current, temp_ext: temp_current-temp_ext, temp0=40, Dt=30/510,
			param_space=FirstOrderStrategy.ParamSpace(KCPU=(1e-5,10), KTemp=(1e-5,1), TauCPU=(1e-9,1), TauTemp=(1e-9,0.5)),
			# params=M2Kernel.Params(KCPU=4, KTemp= 0.1, TauCPU=0.1, TauTemp=0.05),
			# params=M2Kernel.Params(KCPU=5.354987897910978, KTemp= 2.8002979490801128, TauCPU=0.18629771646496662, TauTemp=0.9528960069986708),
			# params=M2Kernel.Params(KCPU=8.328657109382513, KTemp=0.5493883896628988, TauCPU=np.float64(0.19189461891598183), TauTemp=np.float64(0.33700592667911833))		# RMSE = 1.150
			# params=M2Kernel.Params(KCPU=3.2672845157080244, KTemp=0.27811782995435014, TauCPU=np.float64(0.06603239807557451), TauTemp=np.float64(0.18592294666152198))	# RMSE = 0.985
			start_params=FirstOrderStrategy.Params(KCPU=4.838110172690417, KTemp=0.6594080619235382, TauCPU=np.float64(0.12214871259156063), TauTemp=np.float64(0.4990802997780703))		# RMSE = 0.977
			),
		FirstOrderOptimizer(global_min_loss = 0.5, training_duration = timedelta(seconds=1), composition='any', 
			training_stop_flags = FirstOrderOptimizer.StopConditions.GLOBAL_MIN_LOSS 
								| FirstOrderOptimizer.StopConditions.GLOBAL_MAX_DURATION 
			),
		plotstyle=get_plotstyle('IEEE2025')
	)
	m2obj.fit(plot = 							(PP := True),
		X = df2['CPU'].to_frame(),
		y = df2['Temp'],
		)						# options
	m2pred=m2obj.predict(df2['CPU'].to_frame(),  plot = PP, against=df2['Temp'])

	m3_source_col = 'Temp'
	df3=df2.loc[:,df2.columns != m3_source_col].copy(deep=True)
	TEMPERATURE_RESIDUE = 'Temp_residue'
	df3.loc[:,TEMPERATURE_RESIDUE] = (df2.loc[:,m3_source_col].copy(deep=True)-m2pred).rename(TEMPERATURE_RESIDUE)

	m3obj = M3(
		XGBStrategy(n_estimators=1000, early_stopping_rounds=20, learning_rate=0.001),
		plotstyle=get_plotstyle('IEEE2025')
	)
	m3obj.cross_validation(plot = 	(PP := True),
		X = df3.loc[:,df3.columns != TEMPERATURE_RESIDUE],	
		y = df3.loc[:,TEMPERATURE_RESIDUE],					
		n_splits=3
		)					# options
	m3obj.fit(plot = 				(PP := True),
		X = df3.loc[:,df3.columns != TEMPERATURE_RESIDUE],	
		y = df3.loc[:,TEMPERATURE_RESIDUE],
		)
	m3obj.predict(df3.loc[:,df3.columns != TEMPERATURE_RESIDUE], plot = PP, against=df3[TEMPERATURE_RESIDUE])

