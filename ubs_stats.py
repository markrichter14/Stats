"""
Code for Understanding Basic Statistics, 8th edition
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats


def str_to_arr(s):
    '''
    converts a str of space separeted values to an array
    '''
    res = map(int, s.split())
    res = np.array(list(res))
    return res


def class_width(min_val, max_val, num_classes):
    '''
    returns class width, p. 45
    '''
    return int(((max_val - min_val) / num_classes) + 1)


def class_limits(min_val, classes, cls_width):
    '''
    returns class limits (p. 45) as a list of tuples
    '''
    lower_limits = np.array([min_val + n * cls_width for n in range(classes)])
    upper_limits = lower_limits + cls_width - 1
    
    res = [(a, b) for a, b in (zip(lower_limits, upper_limits))]
    
    return res


def freq_dist(data, classes, titles=('','')):
    '''
    produces histogram and relative frequency graphs

    '''
    if isinstance(titles, str):
        titles = (titles, titles)
    
    print(f'min: {data.min()}  max: {data.max()}  size: {data.size}')

    cls_width = class_width(data.min(), data.max(), classes)
    print(f'class width: {cls_width}')

    freq, edges = np.histogram(data, range=(data.min(), data.min() + cls_width*classes), bins=classes)
    print(f'freq: {freq}')

    cls_limits = class_limits(data.min(), classes, cls_width)
    print(f'cls_limits: {cls_limits}')

    bounds = np.array([data.min() - 0.5 + n * cls_width for n in range(classes + 1)])
    print(f'boundaries: {bounds}')

    midpts = [sum(e) / 2 for e in cls_limits]
    print(f'midpts: {midpts}')

    rel_freq = freq / data.size
    print(f'rel_freq: {rel_freq}')
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    ax[0].hist(bounds[:-1], bounds, weights=freq)
    ax[0].set_title(titles[0])
    ax[0].set_ylabel(r'Frequency, $f$')
    ax[0].grid(axis='x', color='0.85')
    ax[0].set_xticks(bounds)

    ax[1].hist(bounds[:-1], bounds, weights=rel_freq)
    ax[1].set_title(titles[0])
    ax[1].set_ylabel(r'Relative frequency, $f/n$')
    ax[1].grid(axis='x', color='0.85')
    ax[1].set_xticks(bounds)

    plt.show()


def ogive(data, classes, title=''):
    '''
    '''
    cls_width = class_width(data.min(), data.max(), classes)
    print(f'class width: {cls_width}')

    freq, edges = np.histogram(data, range=(data.min(), data.min() + cls_width*classes), bins=classes)
    print(f'freq: {freq}')
    cumulative = np.cumsum(np.hstack(([0],freq)))
    print(f'cumulative: {cumulative}')

    bounds = np.array([data.min() - 0.5 + n * cls_width for n in range(classes + 1)])
    print(f'boundaries: {bounds}')

    fig, ax0 = plt.subplots(1, 1, figsize=(6, 4))

    ax0.plot(bounds, cumulative, 'ro-')
    ax0.set_title(title)
    ax0.set_ylabel(r'Cumulative frequency, $\Sigma f$')
    ax0.grid(axis='x', color='0.85')
    ax0.set_xticks(bounds)

    plt.show()

    
def dotplot(data, title=''):
    '''
    produces dot plot graph
    '''
    x = []
    y = []
    unique, counts = np.unique(data, return_counts=True)
    for v, n in zip(unique, counts):
        for i in range(1, n + 1):
            x.append(v)
            y.append(i)

    w = 2
    fig, ax0 = plt.subplots(1, 1, figsize=(15, w))

    ax0.scatter(x, y, c='k')
    ax0.set_title(title)
    ax0.grid(axis='x', color='0.85')
    ax0.set_ylim((0, 6 * w))

    plt.show()

    
def stem_leave(df, sort=True, fill=True, split=False, show_table=False):
    '''
    produces stem and leaf display
    '''
    table = {}
    for n in df:
        stem = n // 10
        leaf = n % 10
        leaves = table.get(stem, [])
        leaves.append(leaf)
        table[stem] = leaves

    if show_table:
        print(f'\ntable:\n{table}')

    print('Stem | Leaves')
    if sort:
        if fill:
            for stem in range(df.min() // 10, 
                              df.max() // 10 + 1):
                leaves = table.get(stem, [])
                if split:
                    for half in [filter(lambda n: n<5, leaves), filter(lambda n: n>=5, leaves)]:
                       print(f'{stem:4d} | {" ".join(map(str, sorted(half)))}')
                else:
                    print(f'{stem:4d} | {" ".join(map(str, sorted(leaves)))}')
        else:
            for stem, leaves in sorted(table.items()):
                print(f'{stem:4d} | {" ".join(map(str, sorted(leaves)))}')
    else:
        if fill:
            for stem in range(df.min() // 10, 
                              df.max() // 10 + 1):
                leaves = table.get(stem, [])
                print(f'{stem:4d} | {" ".join(map(str, leaves))}')
        else:
            for stem, leaves in sorted(table.items()):
                print(f'{stem:4d} | {" ".join(map(str, leaves))}')


def pareto_plot(df, data_col, label_col, title=''):
    '''
    plots a pareto chart
    
    df: dataframe
    data_col: str, data column name
    label_col: str, label column name
    title: str
    '''
    df = df.sort_values(by=[data_col], ascending=False)
    df = df.set_index(label_col)

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

    ax0.bar(df.index, df[data_col])
    ax0.set_title(title)
    ax0.grid(axis='y', color='0.95')
    ax0.set_xticks(df.index)
    ax0.set_xticklabels(labels=df.index, rotation=45)

    plt.show()

    
def pie_plot(df, data_col, label_col, title=''):
    '''
    plots a pie chart
    
    df: dataframe
    data_col: str, data column name
    label_col: str, label column name
    title: str
    '''
    df = df.sort_values(by=[data_col], ascending=False)
    df = df.set_index(label_col)

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

    ax0.pie(df[data_col], labels=df.index, autopct='%1.0f%%', 
            shadow=True, explode=[.1]*df.size, normalize=True)
    ax0.set_title(title)
    ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

    
def mode(data, show=True):
    '''
    calculates the frequency of values
    returns the mode (p. 92)
    '''
    vals, cnts = np.unique(data, return_counts=True)
    vc = {'values': vals, 'counts': cnts}
    df = pd.DataFrame(data=vc)
    df = df.sort_values(by=['counts'], ascending=False)
    df = df.set_index('values')
    if show: display(df.head())
    
    if df['counts'].iat[0] > df['counts'].iat[1]:
        res = df.index[0]
    else:
        res = None
    
    if show: print(f'mode: {res}')
    return res


def chebyshev(k):
    return 1 - 1/(k ** 2)


def chebyshev_int(percent, m, s, show=True):
    k = np.sqrt(1 / (1 - percent))
    res = (m - k * s, m + k * s)
    if show:
        print(f'interval(k={k}): {res[0]:.3f}, {res[1]:.3f}')
    return k, res

def quartiles(data):
    '''
    returns quartiles as a np.array
    '''
    q = [0.25, 0.50, 0.75]
    return np.quantile(data, q)

def five_num(data):
    '''
    returns five-number summary
    '''
    q = [0, 0.25, 0.50, 0.75, 1]
    return np.quantile(data, q)

def interquartile(data):
    '''
    returns interquartile range
    '''
#     return np.subtract(*np.percentile(data, [75, 25]))
    q = [0.25, 0.75]
    q1, q3 = np.quantile(data, q)
    return q3 - q1


def all_stats(data):
    '''
    calculates many stats and returns a dictionary
    '''
    d = {}
    q = [0, 0.25, 0.50, 0.75, 1]
    d['low'], d['q1'], d['q2'], d['q3'], d['high'] = np.quantile(data, q)
    d['median'] = np.median(data)
    d['iqr'] = d['q3'] - d['q1']
    d['mean'] = data.mean()
    d['s'] = np.std(data, ddof=1)
    d['sigma'] = np.std(data, ddof=0)
    d['s2'] = np.var(data, ddof=1)
    d['sigma2'] = np.var(data, ddof=0)
    d['range'] = data.max() - data.min()
    d['cv'] = d['s'] / d['mean']
    d['cv_pop'] = d['sigma'] / d['mean']
    d['five_num'] = (
        f"low: {d['low']:.3f}, Q_1: {d['q1']:.3f}, " + \
        f"median: {d['median']:.3f}, Q_3: {d['q3']:.3f}, " + \
        f"high: {d['high']:.3f}")

    vals, cnts = np.unique(data, return_counts=True)
    vc = {'values': vals, 'counts': cnts}
    df = pd.DataFrame(data=vc)
    df = df.sort_values(by=['counts'], ascending=False)
    df = df.set_index('values')
    d['mode_df'] = df.head()
    if df['counts'].iat[0] > df['counts'].iat[1]:
        d['mode'] = df.index[0]
    else:
        d['mode'] = None

    return d

'''
d = all_stats(data)
print(f"mean: {d['mean']:.3f}")
print(f"median: {d['median']:.3f}")
print(f"mode: {d['mode']:.3f}")
print(f"range: {d['range']:.3f}")
print(f"s: {d['s']:.3f}")
print(f"var: {d['s2']:.3f}")
print(d['five_num'])
print(f"IQR: {d['iqr']:.3f}")
print(f"CV: {d['cv']:.3f}")
'''
