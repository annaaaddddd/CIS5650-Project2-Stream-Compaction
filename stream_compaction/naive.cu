#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static const int blockSize = 128;

        // TODO: __global__
        __global__ void kernNaiveScanStep(int n, int offset, int* odata, const int* idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            // Two device buffers for ping-pong
            int* dev_A = nullptr;
            int* dev_B = nullptr;
            cudaMalloc((void**)&dev_A, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_A failed");
            cudaMalloc((void**)&dev_B, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_B failed");

            // Shift the input right by one (A[0] = 0, A[i] = idata[i-1]) so that
            // running the inclusive scan yields an exclusive result.
            cudaMemset(dev_A, 0, sizeof(int));
            cudaMemcpy(dev_A + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_A failed");

            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            int numPasses = ilog2ceil(n);
            for (int d = 1; d <= numPasses; d++) {
                int offset = 1 << (d - 1);
                kernNaiveScanStep<<<blocksPerGrid, blockSize>>>(n, offset, dev_B, dev_A);
                checkCUDAError("kernNaiveScanStep failed");
                std::swap(dev_A, dev_B);   // latest result is now in dev_A
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_A, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed");

            cudaFree(dev_A);
            cudaFree(dev_B);
        }
    }
}
