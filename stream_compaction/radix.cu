#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "efficient.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static const int blockSize = 256;

        /**
         * e[i] = 1 if bit `bit` of in[i] is 0 (a "false" key), else 0.
         */
        __global__ void kernComputeE(int n, int bit, int *e, const int *in) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            e[i] = ((in[i] >> bit) & 1) ? 0 : 1;
        }

        /**
         * Split scatter. f is the exclusive scan of e. False keys go to f[i],
         * true keys go to t[i] = i - f[i] + totalFalses.
         */
        __global__ void kernScatterRadix(int n, int bit, int totalFalses,
                int *out, const int *in, const int *f) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            int b = (in[i] >> bit) & 1;
            int d = b ? (i - f[i] + totalFalses) : f[i];
            out[d] = in[i];
        }

        /**
         * LSB-first radix sort of non-negative ints, one split pass per bit.
         * Uses the work-efficient scan for the f array.
         */
        void sort(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            // Only sort as many bits as the largest key needs (keys are non-negative).
            int maxVal = *std::max_element(idata, idata + n);
            int numBits = 0;
            while (numBits < 31 && (maxVal >> numBits) != 0) {
                numBits++;
            }

            int paddedN = 1 << ilog2ceil(n);

            int *dev_in = nullptr;
            int *dev_out = nullptr;
            int *dev_e = nullptr;   // e before the scan, f after it
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_in failed");
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_out failed");
            cudaMalloc((void**)&dev_e, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_e failed");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_in failed");

            dim3 blocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int bit = 0; bit < numBits; bit++) {
                // The scan is in place and writes the padded tail, so re-zero it every pass.
                cudaMemset(dev_e, 0, paddedN * sizeof(int));
                checkCUDAError("cudaMemset dev_e failed");

                kernComputeE<<<blocks, blockSize>>>(n, bit, dev_e, dev_in);
                checkCUDAError("kernComputeE failed");

                // totalFalses = e[n-1] + f[n-1]; grab e[n-1] before the scan overwrites it.
                int eLast = 0;
                cudaMemcpy(&eLast, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy eLast failed");

                Efficient::scanDevice(paddedN, dev_e);

                int fLast = 0;
                cudaMemcpy(&fLast, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("cudaMemcpy fLast failed");
                int totalFalses = eLast + fLast;

                kernScatterRadix<<<blocks, blockSize>>>(n, bit, totalFalses, dev_out, dev_in, dev_e);
                checkCUDAError("kernScatterRadix failed");

                std::swap(dev_in, dev_out);   // sorted-so-far is now in dev_in
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed");

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_e);
        }
    }
}
