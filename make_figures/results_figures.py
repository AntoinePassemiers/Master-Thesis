import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt


colors = ['#073B3A', '#0B6E4F', '#08A045', '#6BBF59', '#DDB771', '#DE6B48', '#C65B7C']


def load_results(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    col_names = lines[0].replace('\n', '').replace(' ', '').split(',')
    data = { col_name: list() for col_name in col_names }
    for line in lines[1:]:
        el = line.replace('\n', '').replace(' ', '').split(',')
        for i in range(len(el)):
            if i > 0:
                el[i] = float(el[i])
            data[col_names[i]].append(el[i])
    return data


casp11 = load_results('casp11.full.txt')
cameo = load_results('cameo.full.txt')
casp11_plmdca = load_results('casp11.plmdca.txt')


for name, results in zip(['CASP11', 'CAMEO'], [casp11, cameo]):
    print('%s PPV: %f | short-range: %f | Medium-range: %f | Long-range: %f' % (name, np.mean(results['PPV']), np.mean(results['PPV-short']), np.mean(results['PPV-medium']), np.mean(results['PPV-long'])))
    print('%s PPV/2: %f | short-range: %f | Medium-range: %f | Long-range: %f' % (name, np.mean(results['PPV/2']), np.mean(results['PPV/2-short']), np.mean(results['PPV/2-medium']), np.mean(results['PPV/2-long'])))
    print('%s PPV/5: %f | short-range: %f | Medium-range: %f | Long-range: %f' % (name, np.mean(results['PPV/5']), np.mean(results['PPV/5-short']), np.mean(results['PPV/5-medium']), np.mean(results['PPV/5-long'])))
    print('%s PPV/10: %f | short-range: %f | Medium-range: %f | Long-range: %f' % (name, np.mean(results['PPV/10']), np.mean(results['PPV/10-short']), np.mean(results['PPV/10-medium']), np.mean(results['PPV/10-long'])))
    print('')


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax2.set_axis_off()
options = { 'marker': 'x', 's': 25 }
for ax, metric, name in zip(
        [ax1,ax3,ax4],
        ['PPV/5','PPV/5-medium','PPV/5-long'],
        ['All contacts','Medium-range contacts','Long-range contacts']):
    xs, ys = np.log(casp11_plmdca['Meff']), casp11_plmdca[metric]
    b, m = polyfit(xs, ys, 1)
    ax.scatter(xs, ys, label='plmDCA', color=colors[0], **options)
    xs = np.asarray([np.min(xs), np.max(xs)])
    ax.plot(xs, b + m * xs, '--', color=colors[0])
    xs, ys = np.log(casp11['Meff']), casp11[metric]
    b, m = polyfit(xs, ys, 1)
    ax.scatter(xs, ys, label='Wynona', color=colors[5], **options)
    xs = np.asarray([np.min(xs), np.max(xs)])
    ax.plot(xs, b + m * xs, '--', color=colors[5])
    ax.set_ylabel('Best-L/5 PPV', fontsize=12)
    if ax in [ax3, ax4]:
        ax.set_xlabel('ln(Meff)', fontsize=12)
    ax.set_ylim(-.05, np.max(ys))
    ax.set_title(name, fontsize=12)
    ax.legend(prop={ 'size': 12 })
plt.show()

# import sys; sys.exit(0)

options = { 'marker': 'o', 's': 18 }
scatter_axes = plt.subplot2grid(
        (3, 3), (1, 0), rowspan=2, colspan=2)
x_hist_axes = plt.subplot2grid(
        (3, 3), (0, 0), colspan=2, sharex=scatter_axes)
y_hist_axes = plt.subplot2grid(
        (3, 3), (1, 2), rowspan=2, sharey=scatter_axes)
scatter_axes.scatter(casp11['RMSD'], casp11['TM-score'], label='CASP11', color=colors[0], **options)
scatter_axes.scatter(cameo['RMSD'], cameo['TM-score'], label='CAMEO', color=colors[2], **options)
scatter_axes.set_ylabel('Template modeling score', fontsize=17)
scatter_axes.set_xlabel('Root mean square deviation', fontsize=17)
scatter_axes.axhline(y=.20, linestyle='--', color='grey')
scatter_axes.legend(prop={ 'size': 15 })
x_hist_axes.hist(casp11['RMSD'], color=colors[0], bins=20, orientation='vertical')
y_hist_axes.hist(casp11['TM-score'], color=colors[0], bins=20, orientation='horizontal')
x_hist_axes.hist(cameo['RMSD'], color=colors[2], bins=20, orientation='vertical')
y_hist_axes.hist(cameo['TM-score'], color=colors[2], bins=20, orientation='horizontal')
y_hist_axes.axhline(y=.20, linestyle='--', color='grey')
plt.show()
