import numpy as np
import matplotlib.pyplot as plt


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

print(casp11['RMSD'])


options = {
    'marker': 'o',
    's': 6
}
colors = ['#007379', '#009360', '#86a824', '#ffa600']


scatter_axes = plt.subplot2grid(
        (3, 3), (1, 0), rowspan=2, colspan=2)
x_hist_axes = plt.subplot2grid(
        (3, 3), (0, 0), colspan=2, sharex=scatter_axes)
y_hist_axes = plt.subplot2grid(
        (3, 3), (1, 2), rowspan=2, sharey=scatter_axes)
scatter_axes.scatter(casp11['RMSD'], casp11['TM-score'], label='CASP11', color=colors[0], **options)
#scatter_axes.scatter(cameo_rmsd, cameo_tm, label='CAMEO', color=colors[1], **options)

scatter_axes.set_ylabel('Template modeling score')
scatter_axes.set_xlabel('Root mean square deviation')
scatter_axes.legend()

x_hist_axes.hist(casp11['RMSD'], color=colors[0], bins=20, orientation='vertical')
y_hist_axes.hist(casp11['TM-score'], color=colors[0], bins=20, orientation='horizontal')

#x_hist_axes.hist(cameo_rmsd, color=colors[1], bins=20, orientation='vertical')
#y_hist_axes.hist(cameo_tm, color=colors[1], bins=20, orientation='horizontal')

plt.show()