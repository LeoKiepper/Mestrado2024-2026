import sys, subprocess, time, shutil, psutil, argparse, tempfile, os, numpy as np, math
from montecarlo_utils import get_output_dir, read_prediction_output, save_to_npy, resolve_scalar_output_path, aggregate_predictions_to_matrix, remove_prediction_npy, dump_metadata, register_metadata, _set_active_metadata_scope

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

import importlib.util, uuid
def _query_script_metadata(path):
	try:
		name = f"__measure_probe_{uuid.uuid4().hex}"
		spec = importlib.util.spec_from_file_location(name, path)
		if spec is None:
			return False
		mod = importlib.util.module_from_spec(spec)
		sys.modules[name] = mod
		try:
			spec.loader.exec_module(mod)
		finally:
			try: del sys.modules[name]
			except KeyError: pass
		return True
	except Exception as e:
		print(f"Warning: metadata query failed: {e}")
		return False
def make_printer(fp):
	def p(*args, **kwargs):
		print(*args, **kwargs)
		print(*args, **kwargs, file=fp)
	return p
def draw_bar(label, value, total, width):
	ratio = min(value / total, 1.0)
	filled = int(ratio * width)
	bar = '█' * filled + ' ' * (width - filled)
	return f"{label}: [{bar}]{value:6.2f}{'%  ' if total==100 else 'GB'}"
def mu_sigma(values):
	arr = np.array(values, dtype=float)
	if arr.size == 0:
		return 0.0, 0.0
	return float(np.mean(arr)), float(np.std(arr))
def clear_block(n):
	if n <= 0: return
	sys.stdout.write(f"\033[{n}A")
	for _ in range(n):
		sys.stdout.write("\033[K\n")
	sys.stdout.write(f"\033[{n}A")
	sys.stdout.flush()
def _quote_arg(a):
	if not a: return '""'
	if any(c.isspace() for c in a) or '"' in a:
		return f'"{a.replace("\"","\\\"")}"'
	return a
def draw_progress(run_idx, runs, width):
	ratio = (run_idx + 1) / runs
	filled = int(ratio * width)
	bar = '█' * filled + ' ' * (width - filled)
	return f"RUN: [{bar}] {run_idx+1}/{runs}\n"
def run_script(script, script_args, debug, prediction_name, output_dir, run_idx):
	script_abs = os.path.abspath(script)
	script_dir = os.path.dirname(script_abs) or os.getcwd()
	python_exe = os.path.abspath(sys.executable)
	args_quoted = " ".join(_quote_arg(a) for a in script_args)

	cpu_count = psutil.cpu_count(logical=True)
	cpu_global_samples = []
	mem_samples = []

	with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmpfile:
		log_path = tmpfile.name

	ps1_path = tempfile.mktemp(suffix=".ps1")
	with open(ps1_path, "w", encoding="utf-8") as f:
		# export MEASURE_OUTPUT_DIR and MEASURE_PREDICTION_NAME for child
		f.write("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n")
		f.write(f"$env:MEASURE_OUTPUT_DIR = '{output_dir}'\n")
		f.write(f"$env:MEASURE_PREDICTION_NAME = '{prediction_name}'\n")
		f.write(f"$env:MEASURE_FIGURE_FILENAME_PREFIX = 'run{run_idx+1}_'\n")
		f.write(f"$env:MEASURE_FIGURE_FOLDER_SUFFX = '{'monte_carlo'}'\n")
		f.write("$env:MEASURE_ACTIVE = '1'\n")
		f.write(f"$env:MEASURE_OUTPUT_STEM = '{prediction_name}'\n")
		f.write("$env:PYTHONUNBUFFERED = '1'\n")
		f.write(f"Set-Location -Path '{script_dir}'\n")
		f.write("$oldEAP = $ErrorActionPreference\n")
		f.write("$ErrorActionPreference = 'SilentlyContinue'\n")
		f.write(f"& '{python_exe}' '{script_abs}' {args_quoted} 2>&1 | Tee-Object -FilePath '{log_path}'\n")
		f.write("$ErrorActionPreference = $oldEAP\n")
		f.write("$ec = $LASTEXITCODE\n")
		f.write("Write-Host ''\n")
		f.write("Write-Host ('PROCESS EXITED WITH CODE: {0}' -f $ec) -ForegroundColor Yellow\n")
		if debug:
			f.write("Write-Host ''\n")
			f.write("Write-Host 'Press Enter to close this window...'\n")
			f.write("Read-Host\n")

	cmd = f'start "" /wait powershell -NoProfile -ExecutionPolicy Bypass -File "{ps1_path}"'
	proc_term = subprocess.Popen(cmd, shell=True)

	start_time = time.time()
	term_w = shutil.get_terminal_size().columns
	first_draw = True
	lines_drawn = 0

	try:
		while proc_term.poll() is None:
			try:
				procs = [p for p in psutil.process_iter(['pid','name']) if p.info.get('name') and 'python' in p.info['name'].lower()]
				if not procs:
					time.sleep(0.5)
					continue

				mem = sum(p.memory_info().rss for p in procs)/1024**3
				cpu_list = [p.cpu_percent(interval=0.5)/cpu_count for p in procs]
				per_core = [0.0]*cpu_count
				for i, cpu in enumerate(cpu_list):
					per_core[i % cpu_count] += cpu

				mem_samples.append(mem)
				cpu_global_samples.append(sum(per_core))

			except psutil.NoSuchProcess:
				continue

			bar_w = max(10, term_w-40)
			mem_bar = draw_bar("MEM", mem, psutil.virtual_memory().total/1024**3, width=bar_w)
			lines_count = min(4, cpu_count)
			chunk_size = (cpu_count + lines_count - 1)//lines_count
			cpu_lines = []
			for start in range(0, cpu_count, chunk_size):
				line_vals = per_core[start:start+chunk_size]
				line = " ".join(draw_bar(f"CPU{i+start}", v, 100, width=(bar_w)//chunk_size) for i,v in enumerate(line_vals))
				cpu_lines.append(line)

			if first_draw:
				print(mem_bar)
				for line in cpu_lines: print(line)
				first_draw = False
				lines_drawn = 1 + len(cpu_lines)
			else:
				sys.stdout.write("\033[F"*(1+len(cpu_lines)))
				print(mem_bar)
				for line in cpu_lines: print(line)
			sys.stdout.flush()
	except KeyboardInterrupt:
		try: proc_term.terminate()
		except Exception: pass

	proc_term.wait()
	end_time = time.time()

	# read scalar output via measure_utils read function (delegated)
	output_value = read_prediction_output(prediction_name)
	try:
		os.remove(resolve_scalar_output_path(prediction_name))
	except Exception:
		pass

	elapsed = end_time - start_time
	cpu_avg_global = np.mean(cpu_global_samples) if cpu_global_samples else 0.0
	cpu_max_global = np.max(cpu_global_samples) if cpu_global_samples else 0.0
	mem_avg = np.mean(mem_samples) if mem_samples else 0.0
	mem_max = np.max(mem_samples) if mem_samples else 0.0

	return cpu_avg_global, cpu_max_global, mem_avg, mem_max, elapsed, output_value, lines_drawn, log_path, ps1_path


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("script", help="Script to measure")
	parser.add_argument("script_args", nargs=argparse.REMAINDER)
	parser.add_argument("--runs", type=int, default=1)
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args()
	# compute output dir absolute path once and pass to child
	output_dir = os.path.abspath(get_output_dir())

	all_cpu_avg, all_cpu_max, all_mem_avg, all_mem_max, all_elapsed, all_output = [], [], [], [], [], []

	script_stem = os.path.splitext(os.path.basename(args.script))[0]
	summary_path = os.path.join(output_dir, f"{script_stem}_measure_summary.txt")
	summary_fp = open(summary_path, "w", encoding="utf-8")
	p = make_printer(summary_fp)

	# Get metadata from script
	try:
		_set_active_metadata_scope("script")
		_query_script_metadata(os.path.abspath(args.script))
	except Exception as _e:
		print(f"Warning: failed to import target for metadata: {_e}")
	finally:
		_set_active_metadata_scope("")

	# Save run metadata
	register_metadata({
		'runs': args.runs,
		'start_time': time.strftime(TIME_FORMAT, time.localtime()),
		}, scope="measure")
	dump_metadata(script_stem)

	
	run_pad = len(str(args.runs))
	try:
		progress_drawn = False
		term_w = shutil.get_terminal_size().columns
		for run_idx in range(args.runs):
			#region Print runs progress bar
			# print(f"=== Run {run_idx+1}/{args.runs} ===")
			bar_w = max(10, term_w - 20)
			progress = draw_progress(run_idx, args.runs, bar_w)

			if not progress_drawn:
				p(progress)
				progress_drawn = True
			else:
				sys.stdout.write("\033[F")
				p(progress)
				sys.stdout.flush()
			#endregion
			script_stem = os.path.splitext(os.path.basename(args.script))[0]
			run_id = f"{run_idx+1:0{run_pad}d}"
			prediction_name = f"{script_stem}_{run_id}"

			(cpu_avg, cpu_max, mem_avg, mem_max,
			elapsed, output_value, lines_drawn,
			log_path, ps1_path) = run_script(args.script,args.script_args,args.debug,prediction_name, output_dir, run_idx)

			total_to_clear = 1 + (lines_drawn if lines_drawn else 0)
			clear_block(total_to_clear)

			with open(log_path,'r',encoding='utf-8',errors='replace') as f: text=f.read()
			for path in [log_path, ps1_path]:
				try: os.remove(path)
				except: pass

			print(f"\n--- Script output for run {run_idx+1}---\n")
			print(text)
			p(f"\nRun {run_idx+1} summary:")
			out_disp = "NaN" if math.isnan(output_value) else f"{output_value:.6g}"
			p(f"  Output: {out_disp}")
			p(f"  CPU total: avg {cpu_avg:5.2f}% | max {cpu_max:5.2f}%")
			p(f"  MEM avg: {mem_avg:6.3f} GB | MEM max: {mem_max:6.3f} GB")
			p(f"  Elapsed: {elapsed:.2f}s\n\n")

			all_cpu_avg.append(cpu_avg)
			all_cpu_max.append(cpu_max)
			all_mem_avg.append(mem_avg)
			all_mem_max.append(mem_max)
			all_elapsed.append(elapsed)
			all_output.append(output_value)

	except KeyboardInterrupt:
		try: clear_block(50)
		except: pass
		print("\nInterrupted by user.")

	finally:
		p("\n--- Report ---")
		mem_avg_mu, mem_avg_sigma = mu_sigma(all_mem_avg)
		mem_max_mu, mem_max_sigma = mu_sigma(all_mem_max)
		elapsed_mu, elapsed_sigma = mu_sigma(all_elapsed)
		output_mu, output_sigma = mu_sigma(all_output)

		p(f"MEM avg: {mem_avg_mu:.3f} ± {mem_avg_sigma:.3f} GB")
		p(f"MEM max: {mem_max_mu:.3f} ± {mem_max_sigma:.3f} GB")
		p(f"Elapsed: {elapsed_mu:.2f} ± {elapsed_sigma:.2f} s")
		p(f"Output: {output_mu:.5f} ± {output_sigma:.5f}")

		cpu_avg_mu, cpu_avg_sigma = mu_sigma(all_cpu_avg)
		cpu_max_mu, cpu_max_sigma = mu_sigma(all_cpu_max)
		p(f"CPU total: avg {cpu_avg_mu:5.1f} ± {cpu_avg_sigma:5.1f}% | max {cpu_max_mu:5.1f} ± {cpu_max_sigma:5.1f}%")
		summary_fp.close()

		script_stem = os.path.splitext(os.path.basename(args.script))[0]
		save_to_npy(np.asarray(all_output), f"{script_stem}_outputs")

		names = [f"{script_stem}_{i:0{run_pad}d}" for i in range(1, args.runs+1)]
		try:
			aggregate_predictions_to_matrix(names, f"{script_stem}_predictions")
			for n in names:
				try: remove_prediction_npy(n)
				except Exception: pass
		except Exception as e:
			# do not fail the whole run on aggregation errors; emit a concise warning
			print(f"Warning: aggregation of predictions failed: {e}")
		
	

	# Save run metadata
	register_metadata({
		'end_time': time.strftime(TIME_FORMAT, time.localtime()),
		}, scope="measure")
	dump_metadata(script_stem)

if __name__=="__main__":
	main()
