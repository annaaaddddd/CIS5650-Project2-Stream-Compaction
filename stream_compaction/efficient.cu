#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static const int blockSize = 128;

        __global__ void kernUpSweep(int n, int stride, int* data) {
            int t = blockIdx.x * blockDim.x + threadIdx.x;
            // Only n / stride threads have work this level. Checking t first keeps
            // t * stride from overflowing int for large arrays / block sizes.
            if (t >= n / stride) return;
            int k = t * stride;

            data[k + stride - 1] += data[k + stride / 2 - 1];
        }

        __global__ void kernDownSweep(int n, int stride, int* data) {
            int t = blockIdx.x * blockDim.x + threadIdx.x;
            // Only n / stride threads have work this level. Checking t first keeps
            // t * stride from overflowing int for large arrays / block sizes.
            if (t >= n / stride) return;
            int k = t * stride;

            int left = k + stride / 2 - 1;
            int right = k + stride - 1;

            t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        /**
         * Runs the work-efficient exclusive scan in place on a device array.
         * paddedN must be a power of two; the array must already be zero-padded.
         * No timing and no memory allocation here, so compact() can reuse it.
         */
        static void scanDevice(int paddedN, int* dev_data) {
            int numLevels = ilog2(paddedN);

            // Up-sweep (parallel reduction): stride = 2, 4, 8, ..., paddedN
            for (int d = 0; d < numLevels; d++) {
                int stride = 1 << (d + 1);
                int threads = paddedN / stride;
                dim3 blocks((threads + blockSize - 1) / blockSize);
                kernUpSweep<<<blocks, blockSize>>>(paddedN, stride, dev_data);
                checkCUDAError("kernUpSweep failed");
            }

            // Set root to zero
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset root failed");

            // Down-sweep: stride = paddedN, ..., 8, 4, 2
            for (int d = numLevels - 1; d >= 0; d--) {
                int stride = 1 << (d + 1);
                int threads = paddedN / stride;
                dim3 blocks((threads + blockSize - 1) / blockSize);
                kernDownSweep<<<blocks, blockSize>>>(paddedN, stride, dev_data);
                checkCUDAError("kernDownSweep failed");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            // Pad up to a power of two
            int paddedN = 1 << ilog2ceil(n);

            int* dev_data = nullptr;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed");
            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            checkCUDAError("cudaMemset dev_data failed");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed");

            timer().startGpuTimer();
            scanDevice(paddedN, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);

            int* dev_idata = nullptr;
            int* dev_bools = nullptr;
            int* dev_indices = nullptr;
            int* dev_odata = nullptr;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed");
            cudaMalloc((void**)&dev_bools, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed");
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed");

    
            cudaMemset(dev_bools, 0, paddedN * sizeof(int)); // The padded tail of bools must be 0 so it doesn't affect the scan.
            checkCUDAError("cudaMemset dev_bools failed");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed");


            dim3 blocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            // 1. map: bools[i] = (idata[i] != 0)
            Common::kernMapToBoolean<<<blocks, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed");

            // 2. scan: scanDevice works in place, so scan a copy and keep dev_bools for scatter
            cudaMemcpy(dev_indices, dev_bools, paddedN * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy bools to indices failed");
            scanDevice(paddedN, dev_indices);

            // 3. scatter: odata[indices[i]] = idata[i] where bools[i] == 1
            Common::kernScatter<<<blocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed");

            timer().endGpuTimer();

            // count = indices[n-1] + bools[n-1]. indices lives on the device, so copy
            // that one int back; bools[n-1] is just whether idata[n-1] is nonzero.
            int lastIndex = 0;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy lastIndex failed");
            int count = lastIndex + (idata[n - 1] != 0 ? 1 : 0);

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed");

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return count;
        }
    }
}
