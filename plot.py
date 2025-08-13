import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from scikits.bootstrap import ci

COLORS = {
    'black': '#000000',
    'mustard': '#E69F00',
    'light_blue': '#56B4E9', # copilot guessed this value by its name, wow
    'green': '#009E73',
    'yellow': '#F0E442',
    'orange': '#D55E00',# guessed this one too
    'pink': '#CC79A7',# guessed this one too 
    'dark_pink': '#D81B60',
    'dark_green': '#004D40', 
    'pair1': '#D41159',
    'pair2': '#1A85FF'
}
TOL_COLORS = {
    '1': '#332288',
    '2': '#117733',
    '3': '#44AA99',
    '4': '#88CCEE',
    '5': '#DDCC77',
    '6': '#CC6677',
    '7': '#AA4499',
    '8': '#882255'
}

def box_violin_plot(data, colors, xtick_labels, saving_path, horizontal_lines=None, title=None, 
                    calculate_p_values=False, x_label=None, y_label=None, legends=None, grid=False,
                    figsize=None, plot_violin=True, plot_box=True, plot_jitter=True, 
                    xtick_labels_size=None, xtick_labels_rotation=None,
                    jitter_size=None, jitter_alpha=None):
    """ plots violin and box plot for the given data and using the parameters.

    Parameters
    ----------
    data : list of lists
        Each element of the list is a list of the values that will be plotted.
        Each element of the list will be plotted as a violin and a box plot.
    colors : list of strings
        Each element of the list is a string representing the color of the
        corresponding element of the list of lists 'data'.
    xtick_labels : list of strings
        Each element of the list is a string representing the label of the
        corresponding element of the list of lists 'data'.
    saving_path : string
        The path where the plot will be saved.
    horizontal_lines : list of floats, optional
        If given, horizontal lines will be plotted at the given values.
    title : string, optional
        If given, the title of the plot will be set to this value.
    calculate_p_values : bool, optional
        If True, the p values will be calculated and plotted.
    x_label : string, optional
        If given, the x label of the plot will be set to this value.
    y_label : string, optional
        If given, the y label of the plot will be set to this value.
    legends : list of tuples, optional
        If given, the legends will be added to the plot. Each tuple should
        contain the label and the color of the legend.
    grid : bool, optional
        If True, a grid will be plotted.
    figsize : tuple of ints, optional
        If given, the size of the figure will be set to this value.
    plot_violin : bool, optional
        If True, the violin plot will be plotted.
    plot_box : bool, optional
        If True, the box plot will be plotted.
    plot_jitter : bool, optional
        If True, the jittered dots will be plotted.
    xtick_labels_size : int, optional
        If given, the size of the xtick labels will be set to this value.
    xtick_labels_rotation : int, optional
        If given, the rotation of the xtick labels will be set to this value.
    jitter_size : int, optional
        If given, the size of the jittered dots will be set to this value.
    jitter_alpha : float, optional
        If given, the alpha of the jittered dots will be set to this value.
    """
    # Colors
    BG_WHITE = "#fbf9f4"
    GREY_LIGHT = "#b4aea9"
    GREY50 = "#7F7F7F"
    BLUE_DARK = "#1B2838"
    BLUE = "#2a475e"
    BLACK = "#282724"
    GREY_DARK = "#747473"
    RED_DARK = "#850e00"
    # Horizontal positions for the violins. 
    # They are arbitrary numbers. They could have been [-1, 0, 1] for example.
    positions = np.arange(len(data))

    if figsize is None:
        fig, ax = plt.subplots(figsize=(30, 10))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Horizontal lines that are used as scale reference
    if horizontal_lines is not None:
        for h in horizontal_lines:
            ax.axhline(h, color='black', ls=(0, (5, 5)), alpha=0.8, zorder=0)

    # Add violins ----------------------------------------------------
    # bw_method="silverman" means the bandwidth of the kernel density
    # estimator is computed via Silverman's rule of thumb. 
    # More on this in the bonus track ;)

    # violins
    if plot_violin:
        violins = ax.violinplot(
            data,
            positions=positions,
            widths=0.45,
            bw_method="silverman",
            showmeans=False, 
            showmedians=False,
            showextrema=False
        )

        # Customize violins (remove fill, customize line, etc.)
        for pc in violins["bodies"]:
            pc.set_facecolor("none")
            pc.set_edgecolor(BLACK)
            pc.set_linewidth(1.4)
            pc.set_alpha(1)

    # Add boxplots ---------------------------------------------------
    # Note that properties about the median and the box are passed
    # as dictionaries.
    if plot_box:

        medianprops = dict(
            linewidth=4, 
            color=GREY_DARK,
            solid_capstyle="butt"
        )
        boxprops = dict(
            linewidth=2, 
            color=GREY_DARK
        )
        ax.boxplot(
            data,
            positions=positions,
            showfliers = False, # Do not show the outliers beyond the caps.
            showcaps = False,   # Do not show the caps
            medianprops = medianprops,
            whiskerprops = boxprops,
            boxprops = boxprops
        )

    # Add jittered dots ----------------------------------------------
    if plot_jitter:
        for i, (y, color) in enumerate(zip(data, colors)):
            # get x values
            x = np.ones(len(y)) * positions[i]
            # add noise to x values
            x += np.random.normal(0, 0.05, len(x))
            if jitter_size is None:
                jitter_size = 100
            if jitter_alpha is None:
                jitter_alpha = 0.8
            ax.scatter(x, y, s = jitter_size, color=color, alpha=jitter_alpha, marker='.')


    # Add mean value -----------------------------------------
    means = [np.mean(y) for y in data]
    for i, mean in enumerate(means):
        # Add dot representing the mean
        ax.scatter(i, mean, s=200, color=RED_DARK, zorder=3, marker="x", linewidth=3)
        
    if title:
        ax.set_title(
            title,
            color=BLUE_DARK,
            weight="bold"
        )

    # calculate the p values
    y_upper_lim = ax.get_ylim()[1]
    vertical_move = 0.05 * y_upper_lim
    vertical_offset = 0.02 * y_upper_lim
    tick_length = 0.002 * y_upper_lim
    if calculate_p_values:
        p_values = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    p_values[i,j] = ranksums(data[i], data[j])[1]
        print("p_values: ", p_values)
        # plot the p values, don't use y_lims
        significant_counter = 0
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                if i != j:
                    if p_values[i,j] < 0.05:
                        if p_values[i,j] < 0.001:
                            txt = '***'
                        elif p_values[i,j] < 0.01:
                            txt = '**'
                        else:
                            txt = '*'
                        ax.text((i+j)/2, 
                                y_upper_lim-vertical_offset+vertical_move*significant_counter, 
                                txt, horizontalalignment='center', verticalalignment='center')
                        ax.plot([i, i, j, j], 
                                [y_upper_lim-vertical_offset+vertical_move*significant_counter - tick_length, y_upper_lim-vertical_offset+vertical_move*significant_counter, y_upper_lim-vertical_offset+vertical_move*significant_counter, y_upper_lim-vertical_offset+vertical_move*significant_counter - tick_length], c="black", lw=3)
                        significant_counter += 1
    # extra legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    if legends:
        legends_handles = []
        for l in legends:
            legends_handles.append(mpatches.Patch(color=l[0], label=l[1]))
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.extend(legends_handles)
        plt.legend(handles=handles, loc='lower right')
    # Set grid to use minor tick locations. 
    if grid:
        ax.grid(which = 'major', color='gray', linestyle='dashed')
    if x_label:
        ax.set_xlabel(x_label, weight="bold") 
    if y_label:
        ax.set_ylabel(y_label, weight="bold")
    if xtick_labels:
        ax.set_xticks(np.arange(len(data)))
        if xtick_labels_size is not None and xtick_labels_rotation is not None:
            ax.set_xticklabels(xtick_labels, rotation=xtick_labels_rotation, fontsize=xtick_labels_size)
        elif xtick_labels_size is not None:
            ax.set_xticklabels(xtick_labels, fontsize=xtick_labels_size)
        elif xtick_labels_rotation is not None:
            ax.set_xticklabels(xtick_labels, rotation=xtick_labels_rotation)
        else:
            ax.set_xticklabels(xtick_labels)
    else:
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(saving_path)
    plt.close()

def plot_fots(data, colors, labels, saving_path, legend,
              y_limits=None, x_label=None, y_label=None, title=None,
              figsize=None,
              linestyles=None):
    """ plots the fitness over time for given experiments.

    Parameters
    ----------
    data : list of numpy arrays
        Each element of the list is a numpy array of
        shape (2, generations) where the first row is the mean and the second row is the std
        or 
        shape (generations,) where the values are the mean (or a single run).
    colors : list of strings
        Each element of the list is a string representing the color of the
        corresponding element of the list of numpy arrays 'data'.
    labels : list of strings
        Each element of the list is a string representing the label of the
        corresponding element of the list of numpy arrays 'data'.
    saving_path : string
        The path where the plot will be saved.
    legend : bool
        If True, the legend will be plotted (outside the plot).
    y_limits : tuple of floats, optional
        If given, the y limits of the plot will be set to this value.
    x_label : string, optional
        If given, the x label of the plot will be set to this value.
    y_label : string, optional
        If given, the y label of the plot will be set to this value.
    title : string, optional
        If given, the title of the plot will be set to this value.
    figsize : tuple of ints, optional
        If given, the size of the figure will be set to this value.
    linestyles : list of strings, optional
        Each element of the list is a string representing the linestyle of the
        corresponding element of the list of numpy arrays 'data'.
    """
    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # calculate cis for data
    datas_ci = []
    datas_mean = []
    for d in data:
        dci = []
        dmean = []
        for i in range(len(d)):
            r = ci(d[i], np.mean, n_samples=100)
            dci.append(r)
            dmean.append(np.mean(d[i]))
        datas_ci.append(np.transpose(np.asarray(dci)))
        datas_mean.append(np.asarray(dmean))

    # plot each experiment
    for i, (dci, dm, c, l) in enumerate(zip(datas_ci, datas_mean, colors, labels)):
        if linestyles is not None:
            ax.plot(dm, color=c, linestyle=linestyles[i], label=l)
        else:
            ax.plot(dm, color=c, label=l)
        ax.fill_between(np.arange(len(dm)), 
                        dci[0],
                        dci[1],
                        color=c, 
                        alpha=0.2)

    ## plot each experiment
    #for i, (d, c, l) in enumerate(zip(data, colors, labels)):
    #    if d.shape[0] == 2:
    #        if linestyles is not None:
    #            ax.plot(d[0], color=c, linestyle=linestyles[i], label=l)
    #        else:
    #            ax.plot(d[0], color=c, label=l)
    #        ax.fill_between(np.arange(len(d[0])), d[0]-d[1], d[0]+d[1], color=c, alpha=0.2)
    #    else:
    #        if linestyles is not None:
    #            ax.plot(d, color=c, linestyle=linestyles[i], label=l)
    #        else:
    #            ax.plot(d, color=c, label=l)

    # set the labels
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    # set the limits
    if y_limits:
        plt.ylim(y_limits)
    # set the legend (outside the plot)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # save the figure
    plt.tight_layout()
    plt.savefig(saving_path)
    plt.close()


