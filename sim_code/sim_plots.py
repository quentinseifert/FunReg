
import os
if not os.path.exists("paper_plots"):
    os.makedirs("paper_plots")

from bricks import *

x = np.linspace(-3, 3, 100)
t = np.linspace(0, 1, 100)
func_int = stats.beta(2, 7).pdf(t) + 1
func_t = stats.norm.pdf(4 * (t - 0.2))
func_t = (func_t - np.mean(func_t)) / np.std(func_t)
func_x = x * func_t


fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t, func_int, 'k')
ax[1].plot(x, func_t, 'k')
sample = sim_function_lin(100)
s = 5
y_sample = np.reshape(sample.y.iloc[0:s*40],(s, 40))
t_discrete= np.tile(np.linspace(0, 1, 40), (s, 1))
ax[2].plot(t_discrete.T, y_sample.T)

#ax[0].set_title("Functional Intercept")
#ax[1].set_title("Functional Covariate Effect")
#ax[2].set_title("Samples")


ax[0].set_xlabel("t")
ax[1].set_xlabel("t")
ax[2].set_xlabel("t")
ax[0].set_ylabel(r"$\beta_0(t)$")
ax[1].set_ylabel(r"$\beta_1(t)$")
ax[2].set_ylabel(r"$y(t)$")


plt.tight_layout()
plt.savefig("paper_plots/sim_lin.pdf")



x = np.linspace(-5, 5, 100)
t = np.linspace(0, 1, 100)
mesh = np.meshgrid(x, t)
y_plot = interaction_effect(mesh[1], mesh[0])

fig = plt.figure(figsize=(6, 4))
#ax.pcolormesh(x, t, y_plot, cmap='autumn', shading='auto')
## Define contour levels
#levels = np.linspace(y_plot.min(), y_plot.max(), 10)
#
## Make linewidths scale with the level
#linewidths = np.linspace(0.2, 1, len(levels))



#ax.1contour(x,t, y_plot, colors="black", levels=levels, linewidths=linewidths, linestyles="solid")
ax1 = fig.add_subplot(111, projection="3d")
ax1.plot_surface(mesh[0], mesh[1], y_plot, cmap="autumn")
ax1.contourf(mesh[0], mesh[1], y_plot, zdir='z', offset=y_plot.min()-1, cmap='autumn', alpha=0.7)
ax1.set_zlim(y_plot.min()-1, y_plot.max())
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel(r"$f(x, t)$")
plt.tight_layout()
plt.savefig("paper_plots/sim_smoo.pdf")



############


fig, ax = plt.subplots(2,2, figsize=(14,10))
def plot_lines(i, ax):
    with open(f"sim_results/lin_100_{i}.pkl", "rb") as f:
        weights = pickle.load(f)
    t = np.linspace(0, 1)
    model_dummy, spline, spline_x = fit_lin(0, 100, True) # not actually fitted
    model_dummy.set_weights(weights)
    t_plot = spline.transform_new(t)
    t_plot_x = spline_x.transform_new(t, center=False)
    line_t = t_plot @ model_dummy.weights[0][:19]
    line_x = t_plot_x @ model_dummy.weights[0][19:]
    lines_me = pd.DataFrame(np.column_stack([t, line_t, line_x]))
    lines_refund = pd.read_csv(f"sim_results/lines/line{i}.csv", index_col=0)
    ax[0, 0].plot(lines_me.iloc[:, 0], lines_me.iloc[:, 1], color="grey", alpha=0.1)
    ax[0, 1].plot(lines_me.iloc[:, 0], lines_me.iloc[:, 2], color="grey", alpha=0.1)
    ax[1, 0].plot(lines_refund.iloc[:, 0], lines_refund.iloc[:, 1], color="grey", alpha=0.1)
    ax[1, 1].plot(lines_refund.iloc[:, 0], lines_refund.iloc[:, 2], color="grey", alpha=0.1)
    del model_dummy, spline, spline_x
    gc.collect()

for i in range(100):
    plot_lines(i, ax)


t = np.linspace(0, 1, 40)
func_int = (stats.beta(2, 7).pdf(t) + 1) - np.mean(stats.beta(2, 7).pdf(t) + 1)
func_t = (stats.norm.pdf(4 * (t - 0.2)) - np.mean(stats.norm.pdf(4 * (t - 0.2)))) / np.std(stats.norm.pdf(4 * (t - 0.2)))



ax[0, 0].plot(t, func_int, "r")
ax[1, 0].plot(t, func_int, "r")
ax[0, 1].plot(t, func_t, "r")
ax[1, 1].plot(t, func_t, "r")

ax[0, 0].set_xlabel(r"$t$")
ax[1, 0].set_xlabel(r"$t$")
ax[0, 1].set_xlabel(r"$t$")
ax[1, 1].set_xlabel(r"$t$")

ax[0, 0].set_title(r"GDFR")
ax[0, 1].set_title(r"GDFR")
ax[1, 0].set_title(r"refund")
ax[1, 1].set_title(r"refund")

ax[0, 0].set_ylabel(r"$\beta_0(t)$")
ax[1, 0].set_ylabel(r"$\beta_0(t)$")
ax[0, 1].set_ylabel(r"$\beta_1(t)$")
ax[1, 1].set_ylabel(r"$\beta_1(t)$")
plt.tight_layout()
plt.savefig("paper_plots/sim-lin-results.pdf")


















