import torch
from hdxor import binxor

import time


def set_bits_randomly(tensor):
    numel = tensor.numel()
    # Convert the tensor to a 1D view
    flat_tensor = tensor.view(-1)

    # Generate random integers representing bit patterns
    random_bits = torch.randint(-2**32, 2**32, (numel,), dtype=torch.int64)

    # Convert random bits to float32
    random_floats = random_bits.to(torch.float32)

    # Set the bits in the tensor
    flat_tensor.copy_(random_floats)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host = "cpu"
    print(device)
    matsize = [32*(i+1) for i in range(100)]

    print("\nbinxor\n")

    for n in matsize:
        totaltime = 0

        #print(n)
        #iA = torch.randint(high=2**32-2, size=(n,n), dtype=torch.int32)
        #iB = torch.randint(high=2**32-2, size=(n,n), dtype=torch.int32)
        A =torch.empty((n,n),dtype=torch.float32)
        B =torch.empty((n,n),dtype=torch.float32)
        set_bits_randomly(A)
        set_bits_randomly(B)
        
        C = torch.zeros(size=(n,n), dtype = torch.float32)
        
        A=A.to(device)
        B=B.to(device)
        C=C.to(device)
        for i in range(10):

            print("%x" % int(A[0][i].item()))
        start = time.time()
        for i in range(10000):    
            binxor(A, B, C)
        torch.cuda.synchronize(device)
        print("%f" % ((time.time()-start)*1000))
        #print(A)
        #print(B)
        print(C[0][0].item())
        print(C.size(1))
    
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
