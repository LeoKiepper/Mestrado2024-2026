import sys, subprocess, time, shutil, psutil, argparse, tempfile, os, numpy as np, math

def draw_bar(label, value, total, width):
	ratio = min(value / total, 1.0)
	filled = int(ratio * width)
	bar = '█' * filled + ' ' * (width - filled)
	return f"{label}: [{bar}]{value:6.2f}{'%  ' if total==100 else 'GB'}"

def mu_sigma(values):
	# aceita listas ou numpy arrays; retorna (mu, sigma) como floats
	arr = np.array(values, dtype=float)
	if arr.size == 0:
		return 0.0, 0.0
	return float(np.mean(arr)), float(np.std(arr))

def clear_block(n):
	"""Apaga n linhas imediatamente acima do cursor e posiciona o cursor no início do bloco apagado."""
	if n <= 0:
		return
	# sobe n linhas
	sys.stdout.write(f"\033[{n}A")
	# para cada linha, limpa e avança uma linha
	for _ in range(n):
		sys.stdout.write("\033[K")
		sys.stdout.write("\n")
	# volta para o topo do bloco apagado
	sys.stdout.write(f"\033[{n}A")
	sys.stdout.flush()

def _quote_arg(a):
	# coloca entre aspas se contém espaços ou caracteres especiais
	if not a:
		return '""'
	if any(c.isspace() for c in a) or '"' in a:
		# escape internal quotes by doubling
		return f'"{a.replace("\"","\\\"")}"'
	return a

def run_script(script, script_args, debug):
	# normalize paths and executable
	script_abs = os.path.abspath(script)
	script_dir = os.path.dirname(script_abs) or os.getcwd()
	python_exe = os.path.abspath(sys.executable)
	args_quoted = " ".join(_quote_arg(a) for a in script_args)

	cpu_count = psutil.cpu_count(logical=True)
	cpu_samples = [[] for _ in range(cpu_count)]
	mem_samples = []

	# temp log and ps1
	with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmpfile:
		log_path = tmpfile.name

	# create out_path for numeric output
	out_fd, out_path = tempfile.mkstemp(suffix=".out")
	os.close(out_fd)  # measured script will write to it if measure_utils is used

	ps1_path = tempfile.mktemp(suffix=".ps1")
	with open(ps1_path, "w", encoding="utf-8") as f:
		# robust powershell script: set encoding, env var, location, silence error record shown by PS
		f.write("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n")
		f.write(f"$env:MEASURE_OUTPUT_PATH = '{out_path}'\n")
		f.write(f"$env:PYTHONUNBUFFERED = '1'\n")
		f.write(f"Set-Location -Path '{script_dir}'\n")
		f.write("$oldEAP = $ErrorActionPreference\n")
		f.write("$ErrorActionPreference = 'SilentlyContinue'\n")
		# call Python absolute path; keep stderr merged to stdout; tee writes the console log into log_path
		f.write(f"& '{python_exe}' '{script_abs}' {args_quoted} 2>&1 | Tee-Object -FilePath '{log_path}'\n")
		f.write("$ErrorActionPreference = $oldEAP\n")
		f.write("$ec = $LASTEXITCODE\n")
		# show exit code and optionally pause if debug is requested by measure.py (Read-Host)
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
				# procura processos python (a janela externa criará um python child)
				procs = [p for p in psutil.process_iter(['pid', 'name']) if p.info.get('name') and 'python' in p.info['name'].lower()]
				if not procs:
					time.sleep(0.5)
					continue
				# memória somada, em GB
				mem = sum(p.memory_info().rss for p in procs) / 1024**3
				# cpu por processo (interval controla tempo entre amostras)
				cpu_list = [p.cpu_percent(interval=0.5)/cpu_count for p in procs]
				per_core = [0.0]*cpu_count
				for i, cpu in enumerate(cpu_list):
					per_core[i % cpu_count] += cpu

				mem_samples.append(mem)
				for i in range(cpu_count):
					cpu_samples[i].append(per_core[i])
			except psutil.NoSuchProcess:
				continue

			# desenha barras
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
				# cabeçalho da run é impresso por main; aqui apenas desenhamos as barras
				print(mem_bar)
				for line in cpu_lines:
					print(line)
				first_draw = False
				lines_drawn = 1 + len(cpu_lines)
			else:
				sys.stdout.write("\033[F" * (1 + len(cpu_lines)))
				print(mem_bar)
				for line in cpu_lines:
					print(line)
			sys.stdout.flush()
	except KeyboardInterrupt:
		# tenta terminar processo secundário e sinaliza retorno
		try:
			proc_term.terminate()
		except Exception:
			pass
		# não apagar aqui; main fará a limpeza
		pass

	# espera terminar
	proc_term.wait()
	end_time = time.time()

	# FIRST: try to read numeric output from out_path (written by measure_utils.report_output)
	output_value = float('nan')
	try:
		if os.path.exists(out_path):
			with open(out_path, "r", encoding="utf-8", errors="replace") as outf:
				txt = outf.read().strip()
				if txt:
					# take last whitespace-separated token
					toks = txt.split()
					try:
						output_value = float(toks[-1])
					except Exception:
						output_value = float('nan')
	except Exception:
		output_value = float('nan')

	# FALLBACK: if no out_path value, parse last line from the log (legacy behavior)
	if math.isnan(output_value):
		try:
			with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
				output_lines = f.read().strip().splitlines()
				last_line = output_lines[-1] if output_lines else ""
			try:
				output_value = float(last_line)
			except Exception:
				output_value = float('nan')
		except Exception:
			output_value = float('nan')

	elapsed = end_time - start_time
	cpu_avg = [np.mean(c) if c else 0.0 for c in cpu_samples]
	cpu_max = [np.max(c) if c else 0.0 for c in cpu_samples]
	mem_avg = np.mean(mem_samples) if mem_samples else 0.0
	mem_max = np.max(mem_samples) if mem_samples else 0.0

	# NOTE: do not remove temp files here; main will handle cleanup after clearing the UI
	return cpu_avg, cpu_max, mem_avg, mem_max, elapsed, output_value, lines_drawn, log_path, ps1_path, out_path

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("script", help="Script to measure")
	parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments for the script")
	parser.add_argument("--runs", type=int, default=1, help="Number of runs")
	parser.add_argument("--debug", action="store_true", help="Pause secondary terminal after execution")
	args = parser.parse_args()

	all_cpu_avg, all_cpu_max, all_mem_avg, all_mem_max, all_elapsed, all_output = [], [], [], [], [], []

	try:
		for run_idx in range(args.runs):
			# print header (single line, no leading blank)
			print(f"=== Run {run_idx+1}/{args.runs} ===")
			(cpu_avg, cpu_max, mem_avg, mem_max,
			 elapsed, output_value, lines_drawn,
			 log_path, ps1_path, out_path) = run_script(args.script, args.script_args, args.debug)

			# agora apagamos header + barras (header = 1 linha)
			total_to_clear = 1 + (lines_drawn if lines_drawn else 0)
			clear_block(total_to_clear)

			# ler log (já disponível) e mostrar o conteúdo copiado
			with open(log_path, "r", encoding="utf-8", errors="replace") as f:
				text = f.read()
			# cleanup temporários
			try:
				os.remove(log_path)
			except Exception:
				pass
			try:
				os.remove(ps1_path)
			except Exception:
				pass
			try:
				# remove out_path after reading (if exists)
				if os.path.exists(out_path):
					os.remove(out_path)
			except Exception:
				pass

			# imprimir log e resumo da run
			print("\n--- Script output (copied from secondary terminal) ---\n")
			print(text)
			print(f"\nRun {run_idx+1} summary:")
			out_disp = "NaN" if math.isnan(output_value) else f"{output_value:.6g}"
			print(f"  Output: {out_disp}")
			print(f"  CPU avg (per-core): {', '.join(f'{v:5.2f}%' for v in cpu_avg)}")
			print(f"  CPU max (per-core): {', '.join(f'{v:5.2f}%' for v in cpu_max)}")
			print(f"  MEM avg: {mem_avg:6.3f} GB | MEM max: {mem_max:6.3f} GB")
			print(f"  Elapsed: {elapsed:.2f}s\n")

			# store for aggregated report
			all_cpu_avg.append(cpu_avg)
			all_cpu_max.append(cpu_max)
			all_mem_avg.append(mem_avg)
			all_mem_max.append(mem_max)
			all_elapsed.append(elapsed)
			all_output.append(output_value)

	except KeyboardInterrupt:
		# if interrupted in between runs, try to clear any partial UI
		try:
			# best-effort: attempt to clear last header+bars if present
			clear_block(50)
		except Exception:
			pass
		print("\nInterrupted by user.")
		# still print aggregated results of completed runs
	finally:
		# final aggregated report (only μ ± σ as requested)
		print("\n--- Report ---")
		mem_avg_mu, mem_avg_sigma = mu_sigma(all_mem_avg)
		mem_max_mu, mem_max_sigma = mu_sigma(all_mem_max)
		elapsed_mu, elapsed_sigma = mu_sigma(all_elapsed)
		output_mu, output_sigma = mu_sigma(all_output)

		print(f"MEM avg: {mem_avg_mu:.3f} ± {mem_avg_sigma:.3f} GB")
		print(f"MEM max: {mem_max_mu:.3f} ± {mem_max_sigma:.3f} GB")
		print(f"Elapsed: {elapsed_mu:.2f} ± {elapsed_sigma:.2f} s")
		print(f"Output: {output_mu:.5f} ± {output_sigma:.5f}")

		all_cpu_avg_np = np.array(all_cpu_avg) if all_cpu_avg else np.zeros((0, psutil.cpu_count(logical=True)))
		all_cpu_max_np = np.array(all_cpu_max) if all_cpu_max else np.zeros((0, psutil.cpu_count(logical=True)))
		if all_cpu_avg_np.size:
			for i in range(all_cpu_avg_np.shape[1]):
				cpu_avg_mu, cpu_avg_sigma = mu_sigma(all_cpu_avg_np[:, i])
				cpu_max_mu, cpu_max_sigma = mu_sigma(all_cpu_max_np[:, i])
				print(f"CPU{i}: avg {cpu_avg_mu:5.1f} ± {cpu_avg_sigma:5.1f}% | max {cpu_max_mu:5.1f} ± {cpu_max_sigma:5.1f}%")

if __name__ == "__main__":
	main()
