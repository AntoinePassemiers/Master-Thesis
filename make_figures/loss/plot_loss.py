import json
import numpy as np
from bokeh.io import output_file, show, export_png
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d, FixedTicker


with open('../hpo/hyperopt/29.result', 'r') as f:
    data = json.load(f)
    loss = data['training_loss']
    ppv = data['training_ppv']


def moving_average(signal, wsize, wmin=10, alpha=0.9):
    signal = np.asarray(signal)
    means = np.cumsum(signal, dtype=np.float)
    offset = wsize // 2
    diff = means[wsize:] - means[:-wsize]
    means[offset:offset+len(diff)] = diff[:] / wsize

    means[:offset] = np.nan
    means[offset+len(diff):] = np.nan
    return means


title_text_font_size = '14pt'
axis_label_text_font_size = '15pt'
axis_label_text_font_style = 'normal'
legend_label_text_font_size = '20px'
window_size = 80

p = figure(plot_width=1400, plot_height=700, y_range=(-0.01, np.max(loss)),
           title='Moving average of training loss and best-L accuracy',
           toolbar_location=None)
p.xaxis.axis_label = 'Iterations'
p.xaxis.axis_label_text_font_size = axis_label_text_font_size
p.xaxis.axis_label_text_font_style = axis_label_text_font_style
p.yaxis.axis_label = 'BCE loss'
p.yaxis.axis_label_text_font_size = axis_label_text_font_size
p.yaxis.axis_label_text_font_style = axis_label_text_font_style
p.title.text_font_size = title_text_font_size
p.xgrid[0].ticker = FixedTicker(ticks=np.arange(0, 10000, 50))
p.ygrid[0].ticker = FixedTicker(ticks=np.linspace(0, 0.2, 10))

p.extra_y_ranges = { 'ppv': Range1d(start=0.1, end=np.max(ppv)) }
p.add_layout(LinearAxis(y_range_name='ppv', axis_label='Best-L PPV',
                        axis_label_text_font_size=axis_label_text_font_size,
                        axis_label_text_font_style=axis_label_text_font_style), 'right')
  
xs = np.arange(len(loss))
p.line(xs, loss, line_width=2, color='#AD2831', line_alpha=0.6)
p.line(xs, moving_average(loss, window_size), line_width=4, color='#640D14', legend='Averaged best-L PPV')

p.line(xs, ppv, line_width=2, color='#DBAD6A', y_range_name='ppv', line_alpha=0.6)
p.line(xs, moving_average(ppv, window_size), line_width=4, color='#CF995F', y_range_name='ppv', legend='Averaged BCE loss')

p.legend.label_text_font_size = legend_label_text_font_size

# export_png(p, filename='../imgs/loss.png')
show(p)
    