import matplotlib; matplotlib.use('QtAgg')  
import matplotlib.pyplot as plt
import numpy as np
import time

def random_point(n=1):
	return np.random.uniform(0, 1, (2, n)).tolist()

def append_and_dump_oldest(point, line, size_limit=10):
	x = list(line.get_xdata()); x.append(point[0][0]); x = x[-size_limit:]; line.set_xdata(x)
	y = list(line.get_ydata());	y.append(point[1][0]); y = y[-size_limit:]; line.set_ydata(y)	
	return line

fig, ax = plt.subplots(1)
line, = ax.plot(*random_point(1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.ion()

start_time = time.time()
while time.time() - start_time < 10:
	append_and_dump_oldest(random_point(1), line)
	ax.relim()
	ax.autoscale_view()
	fig.canvas.draw()
	fig.canvas.flush_events()
	plt.pause(0.001)  # Responsivo, não bloqueia

plt.ioff()
plt.show()
