import torch
from hdxor import binxor

import time


def set_bits_randomly(tensor):
    numel = tensor.numel()

    flat_tensor = tensor.view(-1)

    random_bits = torch.randint(-2**32, 2**32, (numel,), dtype=torch.int64)

    random_ints = random_bits.to(torch.int32)


    flat_tensor.copy_(random_ints)

def binary_matmul(A, B):

    C = torch.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = torch.sum(A[i, :] ^ B[:, j])

    return C

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host = "cpu"
    print(device)
    matsize = [32*(i+1) for i in range(15)]

    print("\nbinxor\n")

    for n in matsize:
        totaltime = 0
        #print(n)
        #iA = torch.randint(high=2**32-2, size=(n,n), dtype=torch.int32)
        #iB = torch.randint(high=2**32-2, size=(n,n), dtype=torch.int32)
        A = torch.randint(0,2, size = (n,n*32), dtype=torch.int32)
        B = torch.randint(0,2, size = (n*32,n), dtype=torch.int32)
        binxor_C = torch.zeros(size=(n,n), dtype = torch.int32)
        
        A=A.to(device)
        B=B.to(device)
        binxor_C=binxor_C.to(device)
        start = time.time()
        for i in range(100000):    
            binxor(A, B, binxor_C)
            torch.cuda.synchronize(device)
        print("%f" % ((time.time()-start)*1000))
        #print(C)


    print(" \npytorch\n")

    for n in matsize:
        totaltime = 0
        #print(n)
        A = torch.randint(0,2, size = (n,n*32), dtype=float)
        A = 2*A - 1
        B = torch.randint(0,2, size = (n*32,n), dtype=float)
        B = 2*B - 1
        C = torch.zeros(n,n, dtype=float)
        
        gpu_A = A.to(device)
        gpu_B = B.to(device)
        gpu_C = C.to(device)
        start = time.time()
        for i in range(100000):    
            gpu_C = torch.matmul(gpu_A, gpu_B)
            torch.cuda.synchronize(device)
        print("%f" % ((time.time()-start)*1000))
        C = gpu_C.to(host)
