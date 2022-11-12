import numpy as np
import hw19data

#------------------------------------------------------------
x_data = hw19data.chardata.astype('float32')
target = hw19data.targetdata.T
CharStr = 'ABCDEJK'

#------------------------------------------------------------


def target2c(t):
    id = list(np.where(t == 1))[0][0]
#    printff(t, id)
    return CharStr[id]


#------------------------------------------------------------
W = np.random.rand(25, x_data.shape[1])

#------------------------------------------------------------


def WTA2(x, w):
    """ Win-Take-All
    In: x-sample(x1,x2)
           w-net argument
    Ret: id-Win ID of w
    """
    dist = np.array([(x-ww).dot(x-ww) for ww in w])

    return list(np.where(dist == np.min(dist)))[0][0]


#------------------------------------------------------------
SHOW_SEGMENT_LEN = 10


def shownet0(w):                    # Show net result: 0-dimension
    strdim = [''] * 25

    for id, x in enumerate(x_data):
        iidd = WTA2(x, w)
        c = target2c(target[id])
        strdim[iidd] += c

    for i in range(5):
        showstr = ''
        for j in range(5):
            strid = i * 5 + j
            outstr = '%2d.%s' % (strid+1, strdim[strid])

            if len(outstr) < SHOW_SEGMENT_LEN:
                outstr += " " * (SHOW_SEGMENT_LEN - len(outstr))
            showstr += outstr

        print(showstr)


shownet0(W)

#------------------------------------------------------------


def compete0(x, w, eta):
    for xx in x:
        id = WTA2(xx, w)
        w[id] = w[id] + eta * (xx - w[id])

    return w

#------------------------------------------------------------


def compete1(x, w, eta):
    for xx in x:
        id = WTA2(xx, w)
        w[id] = w[id] + eta * (xx - w[id])

        if id > 0:
            w[id-1] = w[id-1] + eta*(xx-w[id-1])
        if id+1 < w.shape[0]:
            w[id+1] = w[id+1] + eta*(xx-w[id+1])

    return w

#------------------------------------------------------------


def neighborid2(id, row, col):
    rown = id // col
    coln = id % col

    iddim = [id]

    if coln > 0:
        iddim.append(id-1)
    if coln < col-1:
        iddim.append(id+1)
    if rown > 0:
        iddim.append(id-col)
    if rown < row-1:
        iddim.append(id+col)

    return iddim


def compete2(x, w, eta):
    for xx in x:
        id = WTA2(xx, w)

        iddim = neighborid2(id, 5, 5)

        for iidd in iddim:
            w[iidd] = w[iidd] + eta * (xx - w[iidd])

    return w


#------------------------------------------------------------
STEPS = 1000
for i in range(STEPS):
    eta = 0.6 - (0.59 * i/STEPS)
    x = x_data.copy()
    np.random.shuffle(x)
    W = compete2(x, W, eta)

shownet0(W)

#------------------------------------------------------------
#        END OF FILE : HW23.PY
#============================================================
