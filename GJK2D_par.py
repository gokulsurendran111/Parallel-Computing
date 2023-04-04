import numpy as np
import matplotlib.pyplot as plt
from compyle.api import annotate, wrap, Elementwise, get_config, declare
from compyle.types import floatp, doublep, float_
from numpy import cos, sin
from compyle.api import Reduction



def diff(j, i, ai, b, mdiff, blen):
    mdiff[i*blen + j] = ai -b[j]


@annotate
def pminkowskiDiff(i, a, b, mdiff, blen):
    ai = a[i]
    j = declare('int')
    #dif = Elementwise(diff, backend='cython')
    #dif(i, ai, b, mdiff, blen) 
    for j in range(blen):
        mdiff[i*blen + j] = ai - b[j]


@annotate (dot = 'doublep', mlen='int', return_='double')
def findmax(dot, mlen):
    max = dot[0]
    argmax = 0
    for ind in range(mlen):
        if dot[ind] > max:
            max = dot[ind]
            argmax = ind
    return argmax


@annotate(i='int', doublep = 'rad, mdiffx, mdiffy, dot, res', mlen = 'int')
def dotp(i, rad, mdiffx, mdiffy, mlen, dot, res):
    j = declare('int')
    radi = rad[i]
    dirx = cos(radi)
    diry = sin(radi)
    for j in range(mlen):
        dot[j] = dirx*mdiffx[j] + diry*mdiffy[j]
    
    res[i] = findmax(dot, mlen)
        
    
    

def run(a, b, backend):
    alen = len(a)
    blen = len(b)
    mlen = alen*blen
    mdiff = np.zeros(mlen)
    a, b, mdiff= wrap(a, b, mdiff, backend = backend)
    e = Elementwise(pminkowskiDiff, backend=backend)
    e(a, b, mdiff, blen)

    return mdiff


cfg = get_config()
cfg.suppress_warnings = True


sq1 = [[0,0], [0,1], [1,1], [1,0], [0.5, -0.5]]
sq2 = [[2,2], [2,3], [3,3], [3,2]]
sq3 = [[1,1], [2,1], [2,2], [1,2]]
tr1 = [[0, 0.5], [-1, -2], [-1,2]]

ax, ay = np.array(sq1).T
bx, by = np.array(sq2).T
ax, ay, bx, by = np.array(ax), np.array(ay), np.array(bx), np.array(by)

backend = 'cython'
mdiffx = run(ax, bx, backend)
mdiffy = run(ay, by, backend)
mdiffx = np.array(mdiffx, dtype=float)
mdiffy = np.array(mdiffy, dtype=float)


mlen = len(mdiffx)

rad = np.arange(1, 361) * np.pi/180
#print(mdiffx)
dot = np.zeros(mlen)

res = np.zeros(360)

rad, mdiffx, mdiffy, dot, res = wrap(rad, mdiffx, mdiffy,dot, res ,backend=backend)
dot_arr = Elementwise(dotp, backend=backend)
dot_arr(rad, mdiffx, mdiffy, mlen, dot, res)
#r = Reduction('max(a, b)', backend='cython')
#maxi=r(np.abs(dot))
#print(maxi)
print(mdiffx)
print('dot',dot)

res = list(set(res))
res.append(res[0])
res = np.array(res, dtype=int)

print('res', res)

plt.scatter(mdiffx, mdiffy)
plt.plot(mdiffx[res], mdiffy[res])
plt.show()




