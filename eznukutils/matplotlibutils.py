import numpy as np

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    taken from https://stackoverflow.com/questions/10481990/
    matplotlib-axis-with-two-scales-shared-origin
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
    
def multiple_axis_legend(ax, *labels):
    """
    Do the plotting like this:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    l1, = ax1.plot(x, y, color="C0")
    l2, = ax2.plot(x, z, color="C1")
    hlp.multiple_axis_legend(ax1, l1, l2)
    """
    labs = [l.get_label() for l in labels]
    ax.legend(labels, labs)
    
def correct_log_scaling(ax, *ls, which="x", margin=0.2):
    xdat = np.array([])
    ydat = np.array([])
    for l in ls:
        try:
            # for scatter data
            xdat = np.append(xdat, np.array(l.get_offsets()[:,0]))
            ydat = np.append(ydat, np.array(l.get_offsets()[:,1]))
        except AttributeError:
            # for line data
            xdat = np.append(xdat, np.array(l[0].get_xdata()))
            ydat = np.append(ydat, np.array(l[0].get_ydata()))
    if which == "x" or which == "both":
        x_min = np.min(xdat)*(1-margin)
        x_max = np.max(xdat)*(1+margin)
        ax.set_xlim(left=x_min, right=x_max)
    if which == "y" or which == "both":
        y_min = np.min(ydat)*(1-margin)
        y_max = np.max(ydat)*(1+margin)
        ax.set_ylim(bottom=y_min, top=y_max)
    if which != "x" and which != "y" and which != "both":
        print(which)
        raise ValueError('"which" must be either "x" or "y" or "both"')
        
def correct_lin_scaling(ax, *ls, which="x", margin=0.2):
    xdat = np.array([])
    ydat = np.array([])
    for l in ls:
        try:
            # for scatter data
            xdat = np.append(xdat, np.array(l.get_offsets()[:,0]))
            ydat = np.append(ydat, np.array(l.get_offsets()[:,1]))
        except AttributeError:
            # for line data
            xdat = np.append(xdat, np.array(l[0].get_xdata()))
            ydat = np.append(ydat, np.array(l[0].get_ydata()))
    if which == "x" or which == "both":
        x_min = np.min(xdat)-margin
        x_max = np.max(xdat)+margin
        ax.set_xlim(left=x_min, right=x_max)
    if which == "y" or which == "both":
        y_min = np.min(ydat)-margin
        y_max = np.max(ydat)+margin
        ax.set_ylim(bottom=y_min, top=y_max)
    if which != "x" and which != "y" and which != "both":
        print(which)
        raise ValueError('"which" must be either "x" or "y" or "both"')