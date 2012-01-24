from pylab import *

def graph_side_to_side(sound, predict):
    y = array(sound)
    plt.plot(y)

    x = array(xrange(len(predict))) * FFT_STEP
    plt.plot(x, array(predict) * 5000 - 40000)
    plt.show()

def graph_one(data):
    y = array(data)
    plt.plot(y)
    plt.show()
