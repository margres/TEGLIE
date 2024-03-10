import matplotlib.pyplot as plt


def plot_rcParams():
    #to have nice plots
    params = {'font.family': 'DejaVu Sans',
              'font.serif': 'Computer Modern Raman',
              'axes.labelsize': 22,
              'axes.titlesize': 22,
              'xtick.labelsize' : 22,
              'ytick.labelsize' : 22,
              'font.size':18,
              'text.usetex': True,
              'savefig.dpi' : 100,
              'legend.fontsize':18,
              'lines.markersize':5
              #'figure.figsize': fig_size
             }
    plt.rcParams.update(params)