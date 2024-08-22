import segyio
import matplotlib.pyplot as plt
import numpy as np


def plot_sgy(path,
             output=None,
             start=0,
             end=-1,
             norm=None,
             clip=None,
             rate=4,
             crossline_w=3449,
             title=None):
    if output is None:
        output = "algumacoisa.png"

    with segyio.open(path, ignore_geometry=True) as f:
        # figsize = (20, 20)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=1,
            # figsize=figsize,
            facecolor='w',
            edgecolor='k',
            squeeze=False,
            sharex=True)
        axs = axs.ravel()

        if clip is None:
            im = axs[0].imshow(f.trace.raw[start:end].T, cmap=plt.cm.seismic)
        else:
            vmin, vmax = -clip, clip
            im = axs[0].imshow(f.trace.raw[start:end].T,
                               cmap=plt.cm.seismic,
                               vmin=vmin,
                               vmax=vmax)

        plt.title(f"Inline {start//crossline_w}. Normalized [{vmin}, {vmax}]")

        trace_size = len(f.trace.raw)
        yticks = list(range(0, len(f.trace.raw[0]) + 1, 400))
        plt.yticks(yticks, [rate * y for y in yticks])
        plt.xlabel("Crossline")
        plt.ylabel("ms")
        fig.colorbar(im, shrink=0.5)
        plt.tight_layout()
        im.figure.savefig(output)


def plot_well(well_inline,
              well_crossline,
              seismic_data=None,
              path=None,
              min_time=None,
              max_time=None,
              well_name=None,
              output=None,
              title=None):
    if output is None:
        output = f"well_at_{well_inline}_{well_crossline}.png"

    if path is not None:
        seismic_data = np.load(path)
    else:
        assert seismic_data is not None

    print(f"Seismic data shape: {seismic_data.shape}")

    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        # figsize=figsize,
        facecolor='w',
        edgecolor='k',
        squeeze=False,
        sharex=True)
    axs = axs.ravel()

    # Each inline has crossline_w traces.
    print()
    print(f"Well {well_name}")
    print("min_time", min_time)
    print("max_time", max_time)
    time_rate = 4
    ymin = min_time / time_rate if min_time else 0
    ymax = max_time / time_rate if max_time else seismic_data.shape[2]
    print("ymin", ymin)
    print("ymax", ymax)
    

    clip = 1
    vmin, vmax = -clip, clip
    crossline_offset = 100
    z_offset_in_ms = 400
    print("Well crossline", well_crossline)
    im = axs[0].imshow(seismic_data[well_inline,:,:].T,
                       cmap=plt.cm.seismic,
                       vmin=vmin,
                       vmax=vmax,
                       origin='lower')

    axs[0].vlines(x=well_crossline, ymin=ymin, ymax=ymax)
    hline_xline_offset=20
    axs[0].hlines(y=ymin, xmin=well_crossline-hline_xline_offset, 
                  xmax=well_crossline+hline_xline_offset)
    axs[0].hlines(y=ymax, xmin=well_crossline-hline_xline_offset, 
                  xmax=well_crossline+hline_xline_offset)

    if title is None:
        title = "Well "
        if well_name:
            title = f"{well_name} "

        title += f"at (Inline, Crossline) ({well_inline}, {well_crossline})\n"
        title += f"Normalized [{vmin}, {vmax}]"

    plt.title(title)
    plt.xlabel("Crossline")
    plt.ylabel("ms")
    
    crossline_start = int(max([well_crossline - crossline_offset, 0]))
    crossline_end = int(min(
        [well_crossline + crossline_offset, seismic_data.shape[1]]))
    print("Crossline start", crossline_start)
    print("Crossline end", crossline_end)
    z_start = int(max([ymin - z_offset_in_ms/time_rate, 0]))
    z_end = int(min([ymax + z_offset_in_ms/time_rate, seismic_data.shape[2]]))
    print("Z start", z_start)
    print("Z end", z_end)
    plt.xlim([crossline_start, crossline_end])
    plt.ylim([z_start, z_end])

    yticks = list(range(z_start, z_end + 1, (z_end-z_start)//10))
    print("yticks", yticks)
    plt.yticks(yticks, [time_rate * y for y in yticks])

    plt.gca().invert_yaxis()
    fig.colorbar(im, shrink=0.5)
    plt.tight_layout()
    im.figure.savefig(output)
