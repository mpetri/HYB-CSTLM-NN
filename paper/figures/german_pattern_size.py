# coding: utf-8
# run using ipython --pylab; then %run file.py

hits = [266815, 266815, 233793, 164871, 87869, 37653, 15282, 6466, 2964, 1525, 863, 542, 355, 250, 188, 145, 112, 83, 63, 47, 38, 30, 23, 17, 14, 11, 9, 7, 6, 5, 4, 3, 2, 1]
len(hits)
m = range(1,len(hits)+1)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)
bar(m, hits)
yscale('log')
xlabel('$m$gram size')
ylabel('number of successful queries')

