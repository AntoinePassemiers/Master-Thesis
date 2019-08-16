import os
import json
import numpy as np
import matplotlib.pyplot as plt


colors = ['#073B3A', '#0B6E4F', '#08A045', '#6BBF59', '#DDB771', '#DE6B48', '#C65B7C']


hpo = list()
for i in range(32):
    with open(os.path.join('hyperopt', '%i.result' % i), 'r') as f:
        hpo.append(json.load(f))

def get(name):
    r = list()
    for x in hpo:
        if name not in x.keys():
            r.append(x['params'][name])
        else:
            r.append(x[name])
    return r


def moving_average(signal, wsize):
    means = np.cumsum(signal, dtype=np.float)
    offset = wsize // 2
    diff = means[wsize:] - means[:-wsize]
    means[offset:offset+len(diff)] = diff[:] / wsize
    means[:offset] = signal[:offset]
    means[offset+len(diff):] = signal[offset+len(diff):]
    return means


best_it = np.argmin(get('loss'))


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Validation loss
ys = -np.asarray(get('loss'))
ax1.plot(ys, label='Validation loss', color='#F17F29')
ax1.plot(moving_average(ys, 8), label='Moving average', color='#800020')
# ax1.set_title('Learning rate', fontsize=12)
ax1.axhline(y=np.max(ys), linestyle='--', color='grey', label='Optimum')
ax1.axvline(x=best_it, linestyle='--', color='grey')
ax1.set_ylabel('Validation loss')
ax1.set_xlabel('HPO steps')
ax1.legend(prop={ 'size': 12 })
# Learning rate
ys = np.asarray(get('learning_rate'))
ax2.semilogy(ys, label='Learning rate', color='#FF8966')
ax2.semilogy(moving_average(ys, 8), label='Moving average', color='#E5446D')
# ax2.set_title('Learning rate', fontsize=12)
ax2.set_ylabel('Learning rate')
ax2.set_xlabel('HPO steps')
ax2.axhline(y=ys[best_it], linestyle='--', color='grey', label='Optimum')
ax2.axvline(x=best_it, linestyle='--', color='grey')
ax2.legend(prop={ 'size': 12 })
# L2 penalty
ys = np.asarray(get('l2_penalty'))
ax3.semilogy(ys, label='L2 penalty', color='#2EC4B6')
ax3.semilogy(moving_average(ys, 8), label='Moving average', color='#011627')
# ax3.set_title('Learning rate', fontsize=12)
ax3.set_ylabel('L2 penalty')
ax3.axhline(y=ys[best_it], linestyle='--', color='grey', label='Optimum')
ax3.axvline(x=best_it, linestyle='--', color='grey')
ax3.legend(prop={ 'size': 12 })
ax3.set_xlabel('HPO steps')
# Learning rate
ys = np.asarray(get('batch_size'))
ax4.plot(ys, label='Batch size')
ax4.plot(moving_average(ys, 8), label='Moving average', color='#073B3A')
# ax4.set_title('Learning rate', fontsize=12)
ax4.set_ylabel('Batch size')
ax4.set_xlabel('HPO steps')
ax4.axhline(y=ys[best_it], linestyle='--', color='grey', label='Optimum')
ax4.axvline(x=best_it, linestyle='--', color='grey')
ax4.legend(prop={ 'size': 12 })
plt.show()




f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Validation loss
ys = -np.asarray(get('loss'))
ax1.plot(ys, label='Validation loss', color='#F17F29')
ax1.plot(moving_average(ys, 8), label='Moving average', color='#800020')
# ax1.set_title('Learning rate', fontsize=12)
ax1.axhline(y=np.max(ys), linestyle='--', color='grey', label='Optimum')
ax1.axvline(x=best_it, linestyle='--', color='grey')
ax1.set_ylabel('Validation loss')
ax1.set_xlabel('HPO steps')
ax1.legend(prop={ 'size': 12 })
# Learning rate
ys = np.asarray(get('kernel_size'), dtype=np.int)
ax2.plot(ys, label='Kernel size', color='#A7A284')
ax2.plot(moving_average(ys, 8), label='Moving average', color='#433E0E')
# ax2.set_title('Learning rate', fontsize=12)
ax2.set_ylabel('Kernel size')
ax2.set_xlabel('HPO steps')
ax2.axhline(y=ys[best_it], linestyle='--', color='grey', label='Optimum')
ax2.axvline(x=best_it, linestyle='--', color='grey')
ax2.legend(prop={ 'size': 12 })
# L2 penalty
ys = np.asarray(get('num_kernels'), dtype=np.int)
ax3.plot(ys, label='Number of kernels', color='#A27035')
ax3.plot(moving_average(ys, 8), label='Moving average', color='#242331')
# ax3.set_title('Learning rate', fontsize=12)
ax3.set_ylabel('Number of kernels per layer')
ax3.axhline(y=ys[best_it], linestyle='--', color='grey', label='Optimum')
ax3.axvline(x=best_it, linestyle='--', color='grey')
ax3.legend(prop={ 'size': 12 })
ax3.set_xlabel('HPO steps')
# Learning rate
ys = get('activation')
ax4.scatter(np.arange(len(ys)), ys, marker='x', label='Activation function')
# ax4.set_title('Learning rate', fontsize=12)
#ax4.set_ylabel('Activation function')
ax4.set_xlabel('HPO steps')
ax4.axhline(y=ys[best_it], linestyle='--', color='grey', label='Optimum')
ax4.axvline(x=best_it, linestyle='--', color='grey')
ax4.legend(prop={ 'size': 12 })
plt.show()