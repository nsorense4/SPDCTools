"""
 @author  Nicholas J. Sorensen
 @date    2022-02-09
"""
import time as time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_custom(xsize, ysize, x_label, y_label, equal=False, ncol=1, nrow=1, 
                labHor = False, hSpaceSetting = 0.4, 
                wSpaceSetting = 0.4, labelPad = 10, labelPadX = 10, labelPadY = 10, axefont = 12, numsize = 10,
                legfont = 12, 
                commonX = False, commonY = False,
                spineColor = 'black', tickColor = 'black', textColor = 'black', axew = 1, axel = 2, widthBool = False,
                widthRatios = 1, heightRatios = 1, radialBool = False, fontType = 'sansSerif',
                linewidth = 2, invisibleTopRightSpine = False):
    """Initiates and plots a figure and a set of axes.

    Parameters
    ----------
    xsize : float
        size of the figure's x-dimension
    ysize : float
        size of the figure's y-dimension
    xlabel : string
        label for the x-axis
    ylabel : string
        label for the y-axis
    equal : boolean
        (optional) true if the axis relative dimensions are to be the same

    Returns
    -------
    fig
        the figure variable
    ax
        the axis variable
    """

    plt.rc("text", usetex=True)
    if fontType == 'garamond':
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, xcolor, siunitx, ebgaramond, ebgaramond-maths}"
        plt.rc("font", family="serif")
    elif fontType == 'BeraSans':
        matplotlib.rcParams["text.latex.preamble"] = (r"""\usepackage{sansmathfonts,amsmath, xcolor, siunitx, berasans}
                                                            \renewcommand*\familydefault{\sfdefault}  %% Only if the base font of the document is to be sans serif
                                                            \usepackage[T1]{fontenc}""")
        plt.rcParams.update({
                  "text.usetex": True,
                  "font.family": "sans-serif",  # Use a sans serif font (change as needed)
                  # 'font.sans-serif': ['ppl']
              })

    else: 
        
        matplotlib.rcParams["text.latex.preamble"] = (r"""\usepackage{sansmathfonts, amsmath, siunitx, xcolor} 
                                                          \usepackage[OT1]{fontenc}
                                                          \renewcommand*\familydefault{\sfdefault}""") #sansmathfonts
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",  # Use a sans serif font (change as needed)
            # 'font.sans-serif': ['ppl']
        })

    # set global plotting parameters
      # line width
    # msize = 14 #marker size

    # set the distance to offset the numbers from the ticks
    numpad = 5
    # set global tick parameters
    majw = axew  # major tick width
    majl = axel  # major tick length
    minw = axew  # minor tick width
    minl = axel  # minor tick length

    # set global font sizes
      # axis number font size
      # legend font size
    
    # set label rotations
    if labHor is True:
        ylabelRot = 0    
    else:
        ylabelRot = 90
        
    if radialBool is True:
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize,ysize),subplot_kw=dict(projection='polar'))
        
    else:
        if commonX is True and commonY is False:
            if widthBool == False:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), sharex = 'col')
            else:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), sharex = 'col', 
                                       gridspec_kw={'width_ratios': widthRatios, 'height_ratios': heightRatios})
        elif commonY is True and commonX is False:
            if widthBool == False:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), sharey = 'row')
            else:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), sharey = 'row', 
                                       gridspec_kw={'width_ratios': widthRatios, 'height_ratios': heightRatios})
        elif commonY is True and commonX is True:
            if widthBool == False:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), sharey = 'row',
                                       sharex = 'col')
            else:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), sharey = 'row',
                                       sharex = 'col', gridspec_kw={'width_ratios': widthRatios, 'height_ratios': heightRatios})
        else:
            if widthBool == False:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize))
            else:
                fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(xsize, ysize), 
                                       gridspec_kw={'width_ratios': widthRatios, 'height_ratios': heightRatios})
    
    if ncol == 1 and nrow == 1:
        ax.tick_params(
            axis="x",
            which="major",
            width=majw,
            length=majl,
            labelsize=numsize,
            zorder=1,
            direction="in",
            pad=numpad,
            top="off",
            colors=tickColor
        )
        ax.tick_params(
            axis="x",
            which="minor",
            width=minw,
            length=minl,
            labelsize=numsize,
            zorder=1,
            direction="in",
            pad=numpad,
            top="off",
            colors=tickColor
        )
        ax.tick_params(
            axis="y",
            which="major",
            width=majw,
            length=majl,
            labelsize=numsize,
            zorder=1,
            direction="in",
            pad=numpad,
            right="off",
            colors=tickColor
        )
        ax.tick_params(
            axis="y",
            which="minor",
            width=minw,
            length=minl,
            labelsize=numsize,
            zorder=1,
            direction="in",
            pad=numpad,
            right="off",
            colors=tickColor
        )
        if radialBool == False:
            ax.spines['bottom'].set_color(spineColor)
            ax.spines['top'].set_color(spineColor) 
            ax.spines['right'].set_color(spineColor)
            ax.spines['left'].set_color(spineColor)
        ax.set_ylabel(y_label, fontsize=axefont, color=textColor, rotation = ylabelRot, 
                      ha = 'center', va = 'center', labelpad = labelPadY)
        ax.set_xlabel(x_label, fontsize=axefont, color=textColor, labelpad=labelPadX)         
        if equal == True:
            ax.axis("equal")
        plt.subplots_adjust(hspace = hSpaceSetting, wspace = wSpaceSetting)

    elif ncol == 1 or nrow == 1:
        fig.subplots_adjust(right=0.75)
        for i in range(max([ncol, nrow])):
            ax[i].tick_params(
                axis="x",
                which="major",
                width=majw,
                length=majl,
                labelsize=numsize,
                zorder=1,
                direction="in",
                pad=numpad,
                top="off",
                colors=tickColor
            )
            ax[i].tick_params(
                axis="x",
                which="minor",
                width=minw,
                length=minl,
                labelsize=numsize,
                zorder=1,
                direction="in",
                pad=numpad,
                top="off",
                colors=tickColor
            )
            ax[i].tick_params(
                axis="y",
                which="major",
                width=majw,
                length=majl,
                labelsize=numsize,
                zorder=1,
                direction="in",
                pad=numpad,
                right="off",
                colors=tickColor
            )
            ax[i].tick_params(
                axis="y",
                which="minor",
                width=minw,
                length=minl,
                labelsize=numsize,
                zorder=1,
                direction="in",
                pad=numpad,
                right="off",
                colors=tickColor
            )
            ax[i].spines['bottom'].set_color(spineColor)
            ax[i].spines['top'].set_color(spineColor) 
            ax[i].spines['right'].set_color(spineColor)
            ax[i].spines['left'].set_color(spineColor)
        
        if commonX is False and commonY is False:
            for i in range(max([ncol, nrow])):
                ax[i].set_xlabel(x_label[i], fontsize=axefont, color=textColor, labelpad=labelPadX)
                ax[i].set_ylabel(y_label[i], fontsize=axefont, color=textColor, labelpad=labelPadY)
                if equal is True:
                    ax[i].axis("equal")
        elif commonX is True:
            for i in range(max([ncol, nrow])):
                plt.subplots_adjust(hspace = 0, wspace = 0)
                ax[i].set_ylabel(y_label[i], fontsize=axefont, color=textColor, rotation = ylabelRot,  
                                 ha = 'center', va = 'center',labelpad=labelPadY)
            ax[-1].set_xlabel(x_label, fontsize=axefont, color=textColor, labelpad=labelPadX)
        elif commonY is True:
            for i in range(max([ncol, nrow])):
                plt.subplots_adjust(hspace = 0, wspace = 0)
                ax[i].set_xlabel(x_label[i], fontsize=axefont, color=textColor,  
                                 ha = 'center', va = 'center',labelpad=labelPadX)
            ax[0].set_ylabel(y_label, fontsize=axefont, color=textColor, labelpad=labelPadY)
        

    else:
        fig.subplots_adjust(right=0.75)
        for i in range(nrow):
            for j in range(ncol):
                ax[i, j].tick_params(
                    axis="x",
                    which="major",
                    width=majw,
                    length=majl,
                    labelsize=numsize,
                    zorder=1,
                    direction="in",
                    pad=numpad,
                    top="off",
                    colors=tickColor
                )
                ax[i, j].tick_params(
                    axis="x",
                    which="minor",
                    width=minw,
                    length=minl,
                    labelsize=numsize,
                    zorder=1,
                    direction="in",
                    pad=numpad,
                    top="off",
                    colors=tickColor
                )
                ax[i, j].tick_params(
                    axis="y",
                    which="major",
                    width=majw,
                    length=majl,
                    labelsize=numsize,
                    zorder=1,
                    direction="in",
                    pad=numpad,
                    right="off",
                    colors=tickColor
                )
                ax[i, j].tick_params(
                    axis="y",
                    which="minor",
                    width=minw,
                    length=minl,
                    labelsize=numsize,
                    zorder=1,
                    direction="in",
                    pad=numpad,
                    right="off",
                    colors=tickColor
                )
                ax[i, j].spines['bottom'].set_color(spineColor)
                ax[i, j].spines['top'].set_color(spineColor) 
                ax[i, j].spines['right'].set_color(spineColor)
                ax[i, j].spines['left'].set_color(spineColor)
        if commonX is False and commonY is False:
            for i in range(nrow):
                for j in range(ncol):
                    ax[i,j].set_xlabel(x_label[nrow*i+j], fontsize=axefont, color=textColor, labelpad=labelPadX)
                    ax[i,j].set_ylabel(y_label[nrow*i+j], fontsize=axefont, color=textColor, labelpad=labelPadY)
                    if equal is True:
                        ax[i,j].axis("equal")
        elif commonX is True and commonY is False:
            for i in range(nrow):
                for j in range(ncol):
                    ax[i,j].set_ylabel(y_label[nrow*i+j], fontsize=axefont, color=textColor, labelpad=labelPadY)
                    ax[-1, j].set_xlabel(x_label[j], fontsize=axefont, color=textColor, labelpad=labelPadX)
                    if equal is True:
                        ax[i,j].axis("equal")
                    
        elif commonY is True and commonX is False:
            for i in range(nrow):
                for j in range(ncol):
                    ax[i,j].set_xlabel(x_label[nrow*i+j], fontsize=axefont, color=textColor, labelpad=labelPadX)
                    if equal is True:
                        ax[i,j].axis("equal")
                ax[0,j].set_ylabel(y_label[i], fontsize=axefont, color=textColor, labelpad=labelPadY)

        elif commonY is True and commonX is True:
            for i in range(ncol):
                ax[-1,i].set_xlabel(x_label[i], fontsize=axefont, color=textColor, labelpad=labelPadX)
            for i in range(nrow):
                ax[i,0].set_ylabel(y_label[i], fontsize=axefont, color=textColor, labelpad=labelPadY)
            if equal is True:
                for i in range(nrow):
                    for j in range(ncol):
                        ax[i,j].axis("equal")
            
    plt.subplots_adjust(hspace = hSpaceSetting, wspace = wSpaceSetting)

    plt.tight_layout()

    return fig, ax

def thesisColorPalette(n_colors,paletteType = "smooth", darkness = 1.3):
    if paletteType == "smooth":
        colorArray = sns.color_palette(palette='hot', n_colors=int(np.ceil(1.4*n_colors)))[0:-1]
    elif paletteType == "contrast":
        nSubColours = (int(np.floor(n_colors/2))+1)
        colorArrayR = sns.color_palette(palette='hot', n_colors=nSubColours + 1)[0:-1]
        colorArrayB = sns.color_palette(palette='Blues_r', n_colors=nSubColours)[0:-1]
        colorArray= [(0,0,0)]*n_colors
        for i in range(n_colors):
            try:
                colorArray[2*i] = colorArrayR[i]
            except:
                return colorArray
            try:
                colorArray[2*i+1] = colorArrayB[i]
            except:
                return colorArray
            
    elif paletteType == "dark":
        colorArray = sns.color_palette(palette='hot', n_colors=int(np.ceil(1.4*n_colors)))[0:-1]
        colorArrayDark = tuple(tuple(i / darkness for i in color) for color in colorArray)
        return colorArrayDark
    elif paletteType == "darkContrast":
        colorArray = sns.color_palette(palette='hot', n_colors=int(np.ceil(1.4*n_colors)))[0:-1]
        nSubColours = (int(np.floor(n_colors/2))+1)
        colorArrayR = sns.color_palette(palette='hot', n_colors=nSubColours + 1)[0:-1]
        colorArrayB = sns.color_palette(palette='Blues_r', n_colors=nSubColours)[0:-1]
        colorArray= [(0,0,0)]*n_colors
        for i in range(n_colors):
            try:
                colorArray[2*i] = colorArrayR[i]
            except:
                return colorArray
            try:
                colorArray[2*i+1] = colorArrayB[i]
            except:
                return colorArray
        return colorArrayDark
    elif paletteType == 'blues':
        colorArray = sns.color_palette(palette='Blues_r', n_colors=int(np.ceil(1.4*n_colors)))[0:-1]
    return colorArray
    

def plotPalette(pal):
    n = len(pal)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n):
        plt.bar(i, 1, color=pal[i])
        
def makeTopLeftSpinesInvisible(ax, logScale = False, spinePositionY = 'Left', spinePositionX = 'Bottom'):
    # Get all lines from the current axes
    lines = ax.get_lines()
    
    # Initialize max_y_value with a small number
    max_y_value = float('-inf')
    min_y_value = float('inf')
    max_x_value = float('-inf')
    min_x_value = float('inf')
    
    # Iterate over each line to find the maximum y-value
    for line in lines:
        # Get the data points for the line
        x_data, y_data = line.get_xdata(), line.get_ydata()
        # Find the maximum y-value for this line
        max_y_value = max(max_y_value, max(y_data))
        min_y_value = min(min_y_value, min(y_data))
        max_x_value = max(max_x_value, max(x_data))
        min_x_value = min(min_x_value, min(x_data))
        
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.15
    if logScale == False:
        limScaley = 0.09
        limScaleMin = rangey*limScaley
        limScaleMax = rangey*limScaley      
    else:
        limScaley = 0.1
        logRange = (np.log10(max_y_value) - np.log10(min_y_value))
        limScaleMin = 10**(np.log10(min_y_value)-limScaley*logRange)
        limScaleMax = 10**(np.log10(max_y_value)+limScaley*logRange)
    
    if spinePositionX == 'Top':
        ax.spines["bottom"].set_visible(False)
        ax.spines['top'].set_bounds(min_x_value,max_x_value)
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position('top')
    elif spinePositionX == 'Bottom':
        ax.spines["top"].set_visible(False)
        ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position('bottom')
        
    if spinePositionY == 'Left':
        ax.spines["right"].set_visible(False)
        ax.spines['left'].set_bounds(min_y_value,max_y_value)
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position('left')
    elif spinePositionY == 'Right':
        ax.spines["left"].set_visible(False)
        ax.spines['right'].set_bounds(min_y_value,max_y_value)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position('right')
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    # Set the filtered ticks
    ax.set_xticks(filtered_xticks)
    ax.set_yticks(filtered_yticks)
    
    ax.set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    ax.set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
    
    
    return ax
