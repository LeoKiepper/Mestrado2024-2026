import re
import math
from typing import List
from montecarlo_utils import get_output_dir
import sys, os
import numpy as np
from plotstyle import PlotStyle, load_plotstyle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple, Any, Optional
import numpy as np
import pandas as pd
import math


def extract_report_block(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8", errors="replace") as f:
		lines = f.read().splitlines()

	idx = -1
	for i, ln in enumerate(lines):
		if '--- Report ---' in ln:
			idx = i
	if idx == -1:
		return []

	return [ln.rstrip() for ln in lines[idx + 1 :]]
_NUM_PM_RE = re.compile(r'([-+]?\d+(?:\.\d+)?)\s*±\s*([-+]?\d+(?:\.\d+)?)')
def parse_report_metrics_df(report_lines: List[str], queries: List[str]) -> pd.DataFrame:
	mu_vals = []
	sigma_vals = []
	unit_vals = []

	for q in queries:
		q = q.strip()
		if not q:
			mu_vals.append(math.nan)
			sigma_vals.append(math.nan)
			unit_vals.append("")
			continue

		if ':' in q:
			base, sub = [s.strip() for s in q.split(':', 1)]
			sub = sub.lower()
		else:
			base, sub = q, None
		base_l = base.lower()

		target_line = None
		for ln in report_lines:
			if base_l in ln.lower():
				target_line = ln
				break

		if target_line is None:
			mu_vals.append(math.nan)
			sigma_vals.append(math.nan)
			unit_vals.append("")
			continue

		parts = [target_line]
		if sub is not None:
			split_parts = [p.strip() for p in target_line.split('|')]
			for p in split_parts:
				if sub in p.lower():
					parts = [p]
					break
			else:
				parts = split_parts

		found_mu = math.nan
		found_sigma = math.nan
		found_unit = ""

		for txt in parts:
			m = _NUM_PM_RE.search(txt)
			if not m:
				continue

			found_mu = float(m.group(1))
			found_sigma = float(m.group(2))

			tail = txt[m.end():].strip()
			if tail and re.match(r'[A-Za-z%]', tail):
				found_unit = tail
			else:
				found_unit = ""
			break

		mu_vals.append(found_mu)
		sigma_vals.append(found_sigma)
		unit_vals.append(found_unit)

	return pd.DataFrame(
		{"mu": mu_vals, "sigma": sigma_vals, "unit": unit_vals},
		index=queries,
	)
def parse_report_file_df(path: str, queries: List[str]) -> pd.DataFrame:
	block = extract_report_block(path)
	return parse_report_metrics_df(block, queries)
@dataclass(frozen=True)
class RadarAxisModel:
	queries: List[str]
	N: int
	Q: int
	S: Dict[str, Tuple[float, float]]						# scale in original data space (a_q, b_q)
	T: Callable[[str, np.ndarray], np.ndarray]				# transformation provided by caller
	M: Dict[str, Callable[[np.ndarray], np.ndarray]]		# mapping q -> (x_dt -> r_gr)
	M_inv: Dict[str, Callable[[np.ndarray], np.ndarray]]	# mapping (x_dt -> r_gr) -> q
	R: np.ndarray                                       	# shape (N, Q) radial values in [0,1]
	R_inf: np.ndarray										# shape (N, Q) radial values in [0,1]
	R_sup: np.ndarray										# shape (N, Q) radial values in [0,1]
	C: np.ndarray											# circles definition in graph space (values in [0,1])
	ticks_dt: Dict[str, np.ndarray]							# ticks expressed in data space for each axis
	ticks_r: Dict[str, np.ndarray]							# ticks expressed in graph space for each axis
	X: Dict[str, np.ndarray]								# raw values per axis (shape (N,))
	X_inf: Dict[str, np.ndarray]							# lower envelope values per axis (shape (N,))
	X_sup: Dict[str, np.ndarray]							# upper envelope values per axis (shape (N,))
	angles: np.ndarray										# angular positions (Q+1, closed)
	labels: List[str]										# same as queries (semantic alias)
	main_axis: str
	main_axis_direction: float
def build_radar_axis_model( 
		df_list: List[pd.DataFrame], 
		T: Optional[Callable[[str, np.ndarray], np.ndarray]] = None, 
		T_inv: Optional[Callable[[str, np.ndarray], np.ndarray]] = None, 
		PS: Optional[PlotStyle] = None 
) -> RadarAxisModel:
	#region Argument validation
	if len(df_list) == 0: raise ValueError("df_list must be non-empty")
	# derive parameters from PS when available
	if PS is not None:
		queries = list(PS.queries)
		grid_levels = PS.radial_grid_levels
		scale_margin_gamma = PS.axis_scale_margin_gamma
		try: main_axis = PS.main_axis
		except Exception: main_axis = queries[0]
		try: main_axis_direction = float(PS.main_axis_direction)
		except Exception: main_axis_direction = 0.0
		if PS.axis_sequence not in ('cw','ccw'): raise ValueError("axis_sequence must be either 'cw' or 'ccw'.")
		dir_sign = -1 if PS.axis_sequence == 'cw' else 1
	else:	# fallbacks when PS not provided
		# infer queries from first dataframe index if possible
		queries = list(df_list[0].index)
		grid_levels = 5
		scale_margin_gamma = 0.1
		main_axis = queries[0]
		main_axis_direction = 0.0
	if main_axis not in queries: raise ValueError(f"main_axis must be one of queries: {queries}")
	def _identity_transform(q: str, x: np.ndarray) -> np.ndarray: return np.asarray(x, dtype=float)
	if T is None and T_inv is None: 
		T = _identity_transform
		T_inv = _identity_transform
	elif callable(T) and callable(T_inv):pass
	else: raise ValueError("When either T or T_inv are defined, both must be defined")
	#endregion
	#region initialize model parameters
	Q = len(queries)
	C = np.linspace(0.0, 1.0, grid_levels+1)
	N = len(df_list)
	X: Dict[str, np.ndarray] = {}
	X_inf: Dict[str, np.ndarray] = {}
	X_sup: Dict[str, np.ndarray] = {}
	S: Dict[str, Tuple[float, float]] = {}
	M_funcs: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
	M_inv_funcs: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
	ticks_dt: Dict[str, np.ndarray] = {}
	ticks_r: Dict[str, np.ndarray] = {}
	r_matrix = np.empty((N, Q), dtype=float)
	r_matrix_inf = np.empty((N, Q), dtype=float)
	r_matrix_sup = np.empty((N, Q), dtype=float)
	#endregion

	#region main calculations
	# build angular layout
	base_angles = dir_sign * np.linspace(0, 2 * np.pi, Q, endpoint=False)
	main_idx = queries.index(main_axis)
	theta_main = math.radians(main_axis_direction)
	theta_offset = theta_main - base_angles[main_idx]
	angles = base_angles + theta_offset
	angles = (angles + 2*np.pi) % (2*np.pi)
	angles = np.concatenate([angles, [angles[0]]])  # closes polygon
	# Calculate scales, mappings and axis definitions for each axis
	def _extract_mu(df: pd.DataFrame, q: str) -> float:
		try: return float(df.at[q, 'mu'])
		except Exception: pass
		try: return float(df.loc[q, 'mu'])
		except Exception: pass
		if q in df.columns and df.shape[0] == 1: return float(df[q].iat[0])
		raise KeyError(f"Cannot find mu for query '{q}' in dataframe")
	def _extract_sigma(df: pd.DataFrame, q: str) -> float:
		try: return float(df.at[q, 'sigma'])
		except Exception: pass
		try: return float(df.loc[q, 'sigma'])
		except Exception: pass
		if q in df.columns and df.shape[0] == 2: return float(df[q].iat[1])
		raise KeyError(f"Cannot find mu for query '{q}' in dataframe")
	for qq, query in enumerate(queries):
		means = []
		sigmas = []
		for df in df_list:
			means.append(_extract_mu(df, query))
			sigmas.append(_extract_sigma(df, query))
		means = np.asarray(means, dtype=float)
		sigmas = np.asarray(sigmas, dtype=float)
		if not np.all(np.isfinite(means)): raise ValueError(f"Non-finite raw values for axis '{query}'")
		x_min, x_max = float(np.nanmin(means-sigmas)), float(np.nanmax(means+sigmas))
		if not (math.isfinite(x_min) and math.isfinite(x_max)): raise ValueError(f"Non-finite extrema for axis '{query}'")
		X[query] = means
		X_inf[query] = means - sigmas
		X_sup[query] = means + sigmas

		span = x_max - x_min
		if span == 0.0:
			abs_eps = max(abs(x_min) * 1e-6, 1e-8)
			x_min = x_min - abs_eps
			x_max = x_max + abs_eps
			span = x_max - x_min
		margin = scale_margin_gamma * span
		a_dt = max(x_min,1e-9)
		b_dt = x_max + margin
		S[query] = (a_dt, b_dt)
		# transformed endpoints (in D') for constructing affine M
		try: t_endpoints = np.asarray(T(query, np.array([a_dt, b_dt])), dtype=float)
		except Exception as e: raise RuntimeError(f"Transformation T failed for axis '{query}': {e}") from e
		if t_endpoints.shape != (2,): t_endpoints = t_endpoints.reshape(-1)[:2]
		if not np.all(np.isfinite(t_endpoints)): raise ValueError(f"Transformed endpoints not finite for axis '{query}'")
		a_pr, b_pr = float(t_endpoints[0]), float(t_endpoints[1])
		if a_pr == b_pr:	# attempt perturbation
			abs_eps = max(abs(a_dt), abs(b_dt)) * 1e-6 or 1e-8
			a_dt = a_dt - abs_eps
			b_dt = b_dt + abs_eps
			S[query] = (a_dt, b_dt)
			t_endpoints = np.asarray(T(query, np.array([a_dt, b_dt])), dtype=float)
			a_pr, b_pr = float(t_endpoints[0]), float(t_endpoints[1])
			if a_pr == b_pr: raise ValueError(f"Zero span in transformed domain for axis '{query}' after perturbation")
		a_pr, b_pr = (min(a_pr, b_pr), max(a_pr, b_pr))
		delta_pr = b_pr - a_pr
		if not math.isfinite(delta_pr) or delta_pr == 0.0: raise ValueError(f"Degenerate transformed span for axis '{query}'")



		def _make_M(q_local: str, a_pr_local: float, delta_pr_local: float):
			def M_axis(x_dt: np.ndarray) -> np.ndarray:
				arr = np.asarray(T(q_local, np.asarray(x_dt)), dtype=float)
				if arr.shape == (): arr = arr.reshape(1)
				if not np.all(np.isfinite(arr)): raise ValueError(f"Non-finite values after T for axis '{q_local}'")
				return (arr - a_pr_local) / delta_pr_local
			return M_axis
		def _make_M_inv(q_local: str, a_pr_local: float, delta_pr_local: float):
			def M_axis_inv(r_gr: np.ndarray) -> np.ndarray:
				arr = np.asarray(r_gr, dtype=float)
				if arr.shape == (): arr = arr.reshape(1)
				if not np.all(np.isfinite(arr)): raise ValueError(f"Non-finite radial values before T_inv for axis '{q_local}'")
				return np.asarray(
					T_inv(q_local, a_pr_local + arr * delta_pr_local),
					dtype=float
				)
			return M_axis_inv
		M_q = _make_M(query, a_pr, delta_pr)
		M_q_inv = _make_M_inv(query, a_pr, delta_pr)
		M_funcs[query] = M_q
		M_inv_funcs[query] = M_q_inv

		r_vals = M_q(X[query])
		r_infs = M_q(X_inf[query])
		r_sups = M_q(X_sup[query])
		if np.any(np.isnan(r_vals)) or np.any(np.isinf(r_vals)): raise ValueError(f"Non-finite radial values for axis '{query}' after mapping")
		if np.any(np.isnan(r_infs)) or np.any(np.isinf(r_infs)): raise ValueError(f"Non-finite radial values for axis '{query}' after mapping")
		if np.any(np.isnan(r_sups)) or np.any(np.isinf(r_sups)): raise ValueError(f"Non-finite radial values for axis '{query}' after mapping")
		r_matrix[:, qq] = r_vals
		r_matrix_inf[:, qq] = r_infs
		r_matrix_sup[:, qq] = r_sups

		# ticks in data space from C
		t_dt = M_q_inv(C)
		ticks_dt[query] = t_dt
		ticks_r[query] = M_q(t_dt)
	#endregion

	return RadarAxisModel(
		queries = list(queries),
		N = N,
		Q = Q,
		S = S,
		T = T,
		M = M_funcs,
		M_inv = M_inv_funcs,
		R = r_matrix,
		R_inf = r_matrix_inf,
		R_sup = r_matrix_sup,
		C = C, 
		ticks_dt = ticks_dt, 
		ticks_r = ticks_r, 
		X = X, 
		X_inf = X_inf,
		X_sup = X_sup,
		angles = angles, 
		labels = list(queries), 
		main_axis = main_axis, 
		main_axis_direction = main_axis_direction,
	)
def _render_and_save_mini_radar_from_model(model: RadarAxisModel, plot_idx: int, plot_name: str, PS: PlotStyle):
	#region Validation
	if plot_idx < 0 or plot_idx >= model.N: raise IndexError("plot_idx out of range")
	if (r_vals:= model.R[plot_idx, :]).shape[0] != len(model.labels): raise ValueError("Mismatch between r values and labels length")
	if (r_inf:= model.R_inf[plot_idx, :]).shape[0] != len(model.labels): raise ValueError("Mismatch between r values and labels length")
	if (r_sup:= model.R_sup[plot_idx, :]).shape[0] != len(model.labels): raise ValueError("Mismatch between r values and labels length")
	if not np.all(np.isfinite(r_plot := np.concatenate([r_vals, [r_vals[0]]]))): raise ValueError(f"Non-finite radial values for plot {plot_name}")
	if not np.all(np.isfinite(r_plot_inf := np.concatenate([r_inf, [r_inf[0]]]))): raise ValueError(f"Non-finite radial values for plot {plot_name}")
	if not np.all(np.isfinite(r_plot_sup := np.concatenate([r_sup, [r_sup[0]]]))): raise ValueError(f"Non-finite radial values for plot {plot_name}")
	if (C := np.asarray(model.C, dtype=float)).ndim != 1 or C.size == 0: raise ValueError("model.C must be a 1-D non-empty array")
	#endregion
	#region Compose figure
	fig = plt.figure(figsize=PS.figsize)
	ax = plt.subplot(111, polar=True)
	ax.plot(model.angles, r_plot, linewidth=PS.linewidth, color=PS.line_color, zorder=4)
	ax.fill_between(
		model.angles,
		r_plot_inf,
		r_plot_sup,
		color=PS.line_color,
		alpha=PS.region_alpha,
		linewidth=0,
		zorder=3
	)
	ax.set_xticks(model.angles[:-1])
	ax.set_xticklabels([])
	ax.set_yticks(C)
	ax.set_yticklabels([]) 
	ax.spines['polar'].set_visible(False)
	for grid_option in PS.grid_options: ax.yaxis.grid(grid_option)
	global_max = float(np.nanmax(model.R_sup))  # já em espaço gráfico
	margin = 0.05 * (global_max - 0.0) if (global_max - 0.0) != 0 else 0.05
	ymax = max(1.0, global_max) + margin
	ax.set_ylim(0.0, ymax)
	for angle, key in zip(model.angles[:-1], model.labels):		# Set axis labels
		label = PS.query_axis_label_mapping[key]
		if PS.axis_label_direction == "upright":
			specific_opts = {}
		elif PS.axis_label_direction == "radial":
			rotation = np.degrees(angle) + 90
			rot_norm = ((rotation + 180) % 360) - 180
			if abs(rot_norm) > PS.upside_down_text_threshold_angle_from_horizontal: rotation += 180
			specific_opts = {'rotation': rotation, 'rotation_mode': 'anchor'}
		else: specific_opts = {}

		ax.annotate(label, xy=(angle, ax.get_ylim()[1]),
			xytext=np.array([np.cos(angle), np.sin(angle)]) * PS.axis_label_offset,
			**(specific_opts | {
				'xycoords': 'data',
				'textcoords': 'offset points',
				'ha': 'center',
				'va': 'center',
				'clip_on': False, 
				'fontsize': PS.axis_label_fontsize, 
				'fontname': PS.fontfamily, 
				'color': PS.label_color,
			})
		)
	for angle, key in zip(model.angles[:-1], model.labels):		# Per-axis tick labels drawn in data space
		ticks_for_axis_dt = np.asarray(model.ticks_dt[key], dtype=float)
		if ticks_for_axis_dt.shape[0] != C.shape[0]:
			a_q, b_q = model.S[key]
			ticks_for_axis_dt = a_q + C * (b_q - a_q)
		rotation = np.degrees(angle) + 90
		rot_norm = ((rotation + 180) % 360) - 180
		if abs(rot_norm) > PS.upside_down_text_threshold_angle_from_horizontal: rotation += 180
		tick_text_opts = {'rotation': rotation, 'rotation_mode': 'anchor'}
		ticks_for_axis_r = np.asarray(model.ticks_r[key], dtype=float)
		for ci, (r_pos, tick_val) in enumerate(zip(ticks_for_axis_r, ticks_for_axis_dt)): 	# draw labels
			if r_pos < PS.min_tick_label_radius: continue
			offset_points = 4 if ci == 0 else 2
			xytext = np.array([np.cos(angle), np.sin(angle)]) * offset_points
			reduction=60 if key=='Elapsed' else 1
			ax.annotate(f"{tick_val/reduction:.1f}",
				xy=(angle, r_pos),
				xytext=xytext,
				xycoords='data',
				textcoords='offset points',
				ha='center',
				va='center_baseline',
				clip_on=False,
				fontsize=PS.tick_label_fontsize,
				color=PS.label_color,
				**tick_text_opts
			)

	#endregion
	#region Set title and save figure
	os.makedirs(PS.save_folder, exist_ok=True)
	PlotStyle.settitle_and_savefig(fig, ax,
		savefig_options=PlotStyle.compose_savefig_options(**{
			'fname': os.path.join(PS.save_folder, PS.filename_prefix + f"_{plot_name}"),
			'format': PS.file_format,
			'bbox_inches': 'tight',
			'pad_inches': 0.0,
		}),
		set_title_options=None,
		savefig=PS.save,
		save_with_title=PS.save_with_title
	)
	#endregion

	plt.close(fig)	# Prevent piling figures

if __name__ == "__main__":
	#region Argument parsing
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--plot",
		required=True,
		type=str,
		help="Comma-separated list of plot base names (at least one required)"
	)
	parser.add_argument(
		"--plotstyle",
		required=True,
		type=str,
		help="PlotStyle YAML configuration file"
	)
	args = parser.parse_args()
	plot_names = [p.strip() for p in args.plot.split(",") if p.strip()]
	if not plot_names: raise ValueError("--plot must contain at least one plot name")
	output_dir = get_output_dir()
	plot_paths = [
		os.path.join(output_dir, f"{name}_measure_summary.txt")
		for name in plot_names
	]

	for p in plot_paths:
		if not os.path.isfile(p): raise FileNotFoundError(p)


	# PS=PlotStyle(
	# 	yaml_file=args.plotstyle,
	# 	configs_folder="plotstyle_configs",
	# 	base_folder="IEEE-TIM-2026",
	# )
	PST=load_plotstyle(
		file=args.plotstyle,
		configs_folder="plotstyle_configs",
	)
	#endregion
	for pp, PS in enumerate(PST.expand()):
		dfs = []
		for name in plot_names:
			dfs.append(parse_report_file_df(
				os.path.join(output_dir, f"{name}_measure_summary.txt"),
				PS.queries
			))
		def T(q: str, val: float) -> float:
			import numpy as np
			return np.emath.logn(PS.bases_for_logscale_transformation[q], val / PS.normalizing_values[q])
		def T_inv(q: str, val: float) -> float:
			return (PS.bases_for_logscale_transformation[q] ** val) * PS.normalizing_values[q]

		model = build_radar_axis_model(dfs, T=T, T_inv=T_inv, PS=PS)
		for nn, name in enumerate(plot_names):
			_render_and_save_mini_radar_from_model(model, nn, name, PS=PS)