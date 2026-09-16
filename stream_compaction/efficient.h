#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        // In-place work-efficient exclusive scan on a device array of power-of-two
        // length that is already zero-padded. No timing, no allocation.
        void scanDevice(int paddedN, int *dev_data);

        int compact(int n, int *odata, const int *idata);
    }
}
