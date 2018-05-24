import time
import ctypes
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def generateNodes(N):
    """ Generate random 3D nodes
    """
    
    return np.random.rand(N, 3)

def spCalcDistance(nodes):
    """ Single process calculation of the distance function.
    """
    
    p = nodes
    q = nodes.T
    
    # components of the distance vector        
    Rx = p[:, 0:1] - q[0:1]
    Ry = p[:, 1:2] - q[1:2]
    Rz = p[:, 2:3] - q[2:3]
    
    # calculate function of the distance
    L = np.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
    D = L * L * L / 12 + L * L / 6
    
    return D
    
def mpCalcDistance_Worker(nodes, queue, arrD):
    """ Worker process for the multiprocessing calculations
    """

    nP = nodes.shape[0]
    nQ = nodes.shape[0]

    D = np.reshape(np.frombuffer(arrD), (nP, nQ))

    while True:
        job = queue.get()
        if job == None:
            break

        start = job[0]
        stop = job[0] + job[1]
          
        # components of the distance vector
        p = nodes[start:stop]
        q = nodes.T
        
        Rx = p[:, 0:1] - q[0:1]
        Ry = p[:, 1:2] - q[1:2]
        Rz = p[:, 2:3] - q[2:3]

        # calculate function of the distance
        L = np.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
        D[start:stop, :] = L * L * L / 12 + L * L / 6
        
        queue.task_done()
    queue.task_done()

def mpCalcDistance(nodes):
    """ Multiple processes calculation of the distance function.
    """

    # allocate shared array
    nP = nodes.shape[0]    
    nQ = nodes.shape[0]

    arrD = mp.RawArray(ctypes.c_double, nP * nQ)
   
    # setup jobs
    #nCPU = mp.cpu_count()
    nCPU = 2
    nJobs = nCPU * 36
   
    q = nP // nJobs
    r = nP % nJobs
 
    jobs = []
    firstRow = 0
    for i in range(nJobs):
        rowsInJob = q
        if (r > 0):
            rowsInJob += 1
            r -= 1
        jobs.append((firstRow, rowsInJob))
        firstRow += rowsInJob

    queue = mp.JoinableQueue()
    for job in jobs:
        queue.put(job)
    for i in range(nCPU):
        queue.put(None)

    # run workers
    workers = []
    for i in range(nCPU):
        worker = mp.Process(target = mpCalcDistance_Worker,
                            args = (nodes, queue, arrD))
        workers.append(worker)
        worker.start()

    queue.join()
   
    # make array from shared memory    
    D = np.reshape(np.frombuffer(arrD), (nP, nQ))
    return D

def compareTimes(N = 3000):
    """ Compare execution time single processing versus multiple processing.
    """
    nodes = generateNodes(N)
    
    t0 = time.time()
    spD = spCalcDistance(nodes)
    t1 = time.time()
    print("single process time: {:.3f} s.".format(t1 - t0))

    t0 = time.time()
    mpD = mpCalcDistance(nodes)
    t1 = time.time()
    print("multiple processes time: {:.3f} s.".format(t1 - t0))
    
    err = np.linalg.norm(mpD - spD)
    print("calculate error: {:.2e}".format(err))
    
def showTimePlot(N_start = 100, N_stop = 4000, step = 4):
    """ Generate execution time plot single processing versus multiple processing.
    """
    
    N = range(N_start, N_stop, step)
    spTimes = []
    mpTimes = []
    rates = []
    for i in N:
        print(i)
        nodes = generateNodes(i)
        
        t0 = time.time()
        spD = spCalcDistance(nodes)
        t1 = time.time()
        sp_tt = t1 - t0
        spTimes.append(sp_tt)
        
        t0 = time.time()
        mpD = mpCalcDistance(nodes)
        t1 = time.time()
        mp_tt = t1 - t0
        mpTimes.append(mp_tt)
        
        rates.append(sp_tt / mp_tt)
                
    plt.figure()
    plt.plot(N, spTimes)
    plt.plot(N, mpTimes)
    plt.xlabel("N")
    plt.ylabel("Execution time")
    
    plt.figure()
    plt.plot(N, rates)
    plt.xlabel("N")
    plt.ylabel("Rate")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compare', type = int, help = 'If you want to compare results of single and multy processes with N points.' )
    parser.add_argument('-p','--plot', help = 'Shows plots to single and multy process times to computing Radial matrix and dencity plot.', action = 'store_true')
    parser.add_argument('-st','--start', type = int)
    parser.add_argument('-sp','--stop', type = int)
    parser.add_argument('-s','--step',type = int)
    args = parser.parse_args()
    print(args.__dict__)

    #compareTimes()
    #showTimePlot()

if __name__ == '__main__':
    main()
