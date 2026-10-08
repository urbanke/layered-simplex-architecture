/* Exact physical-node, centered Neumaier interpolation for sealed ALT kernels.
 * Build only through sealed_native.py: floating-point contraction and fast math
 * are disabled. The caller retains ownership of every buffer during the call.
 */
#include <stdint.h>
#include <math.h>
#include <float.h>

#ifndef LSA_SEALED_SOURCE_SHA256
#error "The immutable builder must supply the exact source SHA256"
#endif

int lsa_sealed_interp_abi(void) { return 1; }
const char *lsa_sealed_interp_source_sha256(void) {
    return LSA_SEALED_SOURCE_SHA256;
}
const char *lsa_sealed_interp_compiler(void) { return __VERSION__; }

static double compensated(const double terms[8]) {
    double total = 0.0, correction = 0.0;
    for (int i = 0; i < 8; i++) {
        const double term = terms[i];
        const double updated = total + term;
        correction += fabs(total) >= fabs(term)
            ? (total - updated) + term : (term - updated) + total;
        total = updated;
    }
    return total + correction;
}

/* Inputs have already passed the reader's declared support/tail bounds.
 * This routine only interpolates queries at or to the right of u_min. Tiny
 * final-node extrapolation caused by rounded physical coordinates is retained
 * exactly as in the Python reader; the caller enforces its declared u_max.
 */
int lsa_sealed_interp_column(const double *values, int64_t length,
                            const double *queries, int64_t count,
                            double u_min, double step, double upper_guard,
                            double *output, int64_t *failed_index) {
    if (length < 8 || count < 0 || !(step > 0.0) || !isfinite(step)
        || !isfinite(u_min) || !isfinite(upper_guard)) return 4;
    for (int64_t k = 0; k < count; k++) {
        const double query = queries[k];
        if (!isfinite(query) || query < u_min || query > upper_guard) {
            *failed_index = k; return 1;
        }
        const double position = (query - u_min) / step;
        if (!isfinite(position) || position < 0.0 || position >= (double)INT64_MAX) {
            *failed_index = k; return 4;
        }
        int64_t start = (int64_t)floor(position) - 3;
        if (start < 0) start = 0;
        if (start > length - 8) start = length - 8;
        double nodes[8], xs[8], weights[8], terms[8];
        int exact = -1;
        for (int i = 0; i < 8; i++) {
            nodes[i] = u_min + step * (double)(start + i);
            if (query == nodes[i] && exact < 0) exact = i;
            if (!isfinite(nodes[i]) || (i && nodes[i] <= nodes[i-1])) {
                *failed_index = k; return 4;
            }
        }
        if (exact >= 0) {
            output[k] = values[start + exact];
        } else {
            const double node_origin = nodes[3];
            for (int i = 0; i < 8; i++) xs[i] = (nodes[i] - node_origin) / step;
            const double xq = (query - node_origin) / step;
            const double value_origin = values[start + 3];
            for (int i = 0; i < 8; i++) {
                double weight = 1.0;
                for (int j = 0; j < 8; j++)
                    if (i != j) weight /= xs[i] - xs[j];
                weights[i] = weight / (xq - xs[i]);
                terms[i] = weights[i] * (values[start + i] - value_origin);
            }
            output[k] = value_origin + compensated(terms) / compensated(weights);
        }
        if (!isfinite(output[k])) { *failed_index = k; return 3; }
    }
    return 0;
}
