import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from montecarlo_utils import get_output_dir
from plotstyle import PlotStyle, load_plotstyle
def main():
	#region argument parsing
	if len(sys.argv) != 3:
		raise SystemExit(f"Usage: python {os.path.basename(__file__)} <scriptname> <plotstyle_config.yaml>")

	scriptname = sys.argv[1]
	output_dir = get_output_dir()
	pred_path = os.path.join(output_dir, f"{scriptname}_predictions.npy")
	out_path = os.path.join(output_dir, f"{scriptname}_outputs.npy")

	if not os.path.isfile(pred_path):
		raise FileNotFoundError(pred_path)
	if not os.path.isfile(out_path):
		raise FileNotFoundError(out_path)
	pred = np.load(pred_path, allow_pickle=True)
	outputs = np.load(out_path, allow_pickle=True)
	if pred.ndim != 2:
		raise ValueError("Predictions must be 2-D (runs, time)")
	if outputs.ndim != 1:
		raise ValueError("Outputs must be 1-D (runs,)")
	ref = np.load(os.path.join(output_dir, f"{scriptname}_ref.npy"), allow_pickle=True)
	timestamps = np.load(os.path.join(output_dir, f"{scriptname}_timestamps.npy"), allow_pickle=True)
	assert ref.shape == (pred.shape[1],), "Reference shape mismatch"
	assert timestamps.shape == (pred.shape[1],), "Timestamps shape mismatch"

	# PS = PlotStyle(yaml_file=sys.argv[2])
	PST = load_plotstyle(file=sys.argv[2])
	#endregion

	#region compute MC statistics
	lower_envelope = np.min(pred, axis=0)
	mean = np.mean(pred, axis=0)
	upper_envelope = np.max(pred, axis=0)
	acc_mu = float(np.mean(outputs))
	acc_sigma = float(np.std(outputs))
	#endregion
	#region persist MC results
	mc_matrix = np.vstack([lower_envelope, mean, upper_envelope])
	mc_path = os.path.join(output_dir, f"{scriptname}_MC.npy")
	np.save(mc_path, mc_matrix)

	acc_path = os.path.join(output_dir, f"{scriptname}_acc.txt")
	with open(acc_path, "w", encoding="utf-8") as f:
		f.write(f"mu = {acc_mu}\n")
		f.write(f"sigma = {acc_sigma}\n")
	#endregion

	for pp, PS in enumerate(PST.expand()):
		fig, ax = plt.subplots(1, figsize=PS.figsize_single)
		ax.plot(timestamps, ref, **(PS.reference_plot_options|{'zorder':2}))
		ax.plot(timestamps, mean, **(PS.prediction_plot_options|{'zorder':4}))
		pred_line_color = ax.lines[-1].get_color()
		ax.fill_between(
			timestamps,
			lower_envelope,
			upper_envelope,
			color=pred_line_color,
			alpha=PS.region_alpha,
			linewidth=0,
			zorder=3
		)
		ax.annotate(rf"RMSE = {acc_mu:.4f} $\pm$ {acc_sigma:.3f}", xy=(0.99,0.04), xycoords='axes fraction',
			fontsize=PS.annotate_fontsize, horizontalalignment='right', verticalalignment='bottom', zorder=5)



		ax.set_facecolor(PS.plotarea_facecolor)
		for grid_option in PS.grid_options: ax.grid(**grid_option)
		ax.set_xlabel(PS.xlabel_timeseries, fontsize=PS.label_fontsize, fontname=PS.fontfamily)
		ax.set_ylabel(PS.ylabel_temperature, fontsize=PS.label_fontsize, fontname=PS.fontfamily)
		ax.legend(fontsize=PS.legend_fontsize)
		ax.tick_params(axis='both', labelsize=PS.tick_label_fontsize)
		ax.set_ylim(PS.y_limits)
		if hasattr(PS,'spine_linewidth'):
			for spine in ax.spines.values(): spine.set_linewidth(PS.spine_linewidth)


		#region Save figure and set title
		# diagnostic prints for one run
		os.makedirs(PS.save_folder, exist_ok=True)
		savefig_options_dict = {
			'fname': os.path.join(PS.save_folder, PS.filename_prefix + f"{scriptname}")
			}
		set_title_options_dict = {
			'label':PS.title_prefix+'\n'+scriptname,
			}
		savefig=PS.save
		savefig_options_dict |= {
			'format':PS.file_format,
			'bbox_inches':'tight',
			}
		PlotStyle.settitle_and_savefig(fig, ax,
			savefig_options=PlotStyle.compose_savefig_options(
				**savefig_options_dict
			),
			set_title_options=PlotStyle.compose_set_title_options(
				**set_title_options_dict
			),
			savefig=savefig,
			save_with_title=PS.save_with_title
		)
		#endregion

		if pp == 1: plt.show(block=True)
		else: plt.close(fig)

if __name__ == "__main__":
	main()
