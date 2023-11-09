import torch
import hdxor

import time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host = "cpu"
    print(device)
    matsize = [512*(i+1) for i in range(10)]

    print("\nbinxor\n")

    for n in matsize:
        totaltime = 0
        #print(n)
        A = torch.randn(n,int(n/32), dtype = torch.float32)
        B = torch.randn(int(n/32),n, dtype = torch.float32)
        C = torch.zeros(n,n, dtype = torch.float32)
        
        gpu_A = A.to(device)
        gpu_B = B.to(device)
        gpu_C = C.to(device)
        start = time.time()
        for i in range(10000):    
            gpu_C = hdxor.binxor(gpu_A, gpu_B)
            torch.cuda.synchronize(device)
        print("%f" % ((time.time()-start)*1000))
        C = gpu_C.to(host)
    
    print(" \npytorch\n")
    for n in matsize:
        totaltime = 0
        #print(n)
        A = torch.randint(0,2, size = (n,n), dtype=float)
        A = 2*A - 1
        B = torch.randint(0,2, size = (n,n), dtype=float)
        B = 2*B - 1
        C = torch.zeros(n,n, dtype=float)
        
        gpu_A = A.to(device)
        gpu_B = B.to(device)
        gpu_C = C.to(device)
        start = time.time()
        for i in range(10000):    
            gpu_C = torch.matmul(gpu_A, gpu_B)
            torch.cuda.synchronize(device)
        print("%f" % ((time.time()-start)*1000))
        C = gpu_C.to(host)
