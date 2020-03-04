import os
from contextlib import ExitStack
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse


# threshholds = [490, 495]#[475 ,480, 485, 490, 495 ]
# threshholds = [670, 675, 680]#[ 650, 665, 675, 680 ]
# threshholds = [ 630, 640, 650, 660, 665 ]
# threshholds = [ 500, 515, 525, 530, 545  ]
# threshholds = [ 820, 830, 840, 845 ]


def generate_ert_simple(threshhold, N, g_algo, max_iter = 500 ):
    filenames = ["{}{}.csv".format(g_algo, i) for i in range(N)]
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname)) for fname in filenames]
        readCSVs = [csv.reader(files[i], delimiter=';') for i in range(N)]
        data = []
        ii = None
        key = 0
        for CSVs in readCSVs:
            ii = 0
            for row in CSVs:
                tmp  = 1 if float(row[0]) > threshhold else 0
                if key == 0:
                    if ii == 0:
                        data.append(tmp)
                    elif ii > 0 and ii < max_iter:
                        data.append(max(tmp, data[-1]))
                    else:
                        break
                else:
                    if ii == 0:
                        data[ii] += tmp
                    elif ii > 0 and ii < max_iter:
                        data[ii] = max(data[ii] + tmp, data[ii-1])
                    else:
                        break
                ii += 1

            key += 1
    return np.array(data)/N



def exe (N, ns, iters):
    for r in range(N):
        os.system("python snp.py -n {n} -m genetic --n_run {r} --iters {it} --steady-delta 2500 --sensor-range 0.3".format(n = ns, r=r, it = iters))
        os.system("python snp.py -n {n} -m num_greedy --n_run {r} --iters {it} --steady-delta 2500 --sensor-range 0.3".format(n = ns, r=r, it = iters))
        os.system("python snp.py -n {n} -m recuitSimule --n_run {r} --iters {it} --steady-delta 2500 --sensor-range 0.3".format(n = ns, r=r, it = iters))



if __name__=="__main__":

    can = argparse.ArgumentParser()

    can.add_argument("-t","--threshhold", default=490, type=int,
            help="Threshhold")

    can.add_argument("-ns", "--nb-sensors", metavar="NB", default=3, type=int,
            help="Number of sensors")

    can.add_argument("-n", "--nb_run", default=10, type=int,
            help="Number of runs")

    can.add_argument("-i", "--iter", default=1500, type=int,
            help="Joint min number of iterations in all files")

    can.add_argument("-e", "--exec", default=0, type=int,
            help="Run all algorithms. 1 to execute")



    the = can.parse_args()

    threshholds = [the.threshhold]

    g_algo =  'genetic'
    r_algo =  'num_greedy'
    rs_algo = 'recuitSimule'


    if the.exec == 1:
        exe (the.nb_run, the.nb_sensors, the.iter)



    tests_g = [generate_ert_simple(threshholds[i], the.nb_run, g_algo, the.iter) for i in range(len(threshholds))]
    tests_r = [generate_ert_simple(threshholds[i], the.nb_run, r_algo, the.iter) for i in range(len(threshholds))]
    tests_rs = [generate_ert_simple(threshholds[i], the.nb_run, rs_algo, the.iter) for i in range(len(threshholds))]

    # multiple line plot
    num=0
    x = range(the.iter)
    palette_g = plt.get_cmap('Set1')
    palette_r = plt.get_cmap('flag')
    palette_rs = plt.get_cmap('prism')

    for test in tests_g:
        num+=1
        plt.plot(x, test, marker='_', color=palette_g(num), linewidth=5, alpha=0.9, label= g_algo+" : threshhold - "+ str(threshholds[num-1]))
    num=0
    for test in tests_r:
        num+=1
        plt.plot(x, test, marker='_', color=palette_r(num*50), linewidth=2, alpha=0.9, label= r_algo + " : threshhold - "+ str(threshholds[num-1]))

    num=0
    for test in tests_rs:
        num+=1
        plt.plot(x, test, marker='_', color=palette_rs(num*50), linewidth=3, alpha=0.9, label= rs_algo + " : threshhold - "+ str(threshholds[num-1]))


    # Add legend
    plt.legend(loc=0, ncol=2)
    plt.grid(True)
    plt.show()
