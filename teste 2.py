import matplotlib; matplotlib.use('QtAgg')  
import matplotlib.pyplot as plt
import numpy as np

def random_point(n=1):
	return np.random.uniform(0, 1, (2, n)).tolist()
def append_point(point, line):
	x = list(line.get_xdata())
	y = list(line.get_ydata())
	x.append(point[0][0])
	y.append(point[1][0])
	line.set_xdata(x)
	line.set_ydata(y)
	return line

def set_topmost(fig):
    """
    Tenta tornar a janela do matplotlib 'always on top'.
    Suporte para backends TkAgg e QtAgg (qualquer binding Qt).
    """
    import matplotlib
    import warnings

    backend = matplotlib.get_backend().lower()
    window = getattr(fig.canvas.manager, "window", None)
    if window is None:
        warnings.warn("set_topmost: não encontrou janela associada à figura.")
        return

    # Tk backend
    if "tkagg" in backend:
        try:
            window.wm_attributes("-topmost", 1)
        except Exception as e:
            warnings.warn(f"set_topmost (Tk): erro ao aplicar 'topmost': {e}")
        return

    # Qt backend (detecta dinamicamente)
    if "qt" in backend:
        try:
            # Tenta importar o Qt que o matplotlib usou internamente
            qt_mod = type(window).__module__.split('.')[0]  # ex: 'PySide6'
            Qt = __import__(f"{qt_mod}.QtCore", fromlist=["Qt"]).Qt
            window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            window.show()
            return
        except Exception as e:
            warnings.warn(f"set_topmost (Qt): falhou ao aplicar WindowStaysOnTopHint: {e}")
            return

    warnings.warn(f"set_topmost: backend '{backend}' não suportado.")

fig, ax = plt.subplots(1)
set_topmost(fig)
line, = ax.plot(*random_point(1))
ax.set_xlim(0, 1)  
ax.set_ylim(0, 1)
plt.ion()
for i in range(10):
	append_point(random_point(1), line)
	fig.canvas.draw()
	fig.canvas.flush_events()
	plt.pause(0.5)

plt.ioff()
plt.show(block=True)
