/* Diagnostic only: exact current physical-node, centered compensated stencil.
 * Compile with contraction and fast math disabled; no contour/approximation.
 */
#include <stdint.h>
#include <math.h>

static double compensated(const double terms[8]) {
    double total=0.0, correction=0.0;
    for (int i=0;i<8;i++) {
        const double term=terms[i];
        const double updated=total+term;
        correction += fabs(total)>=fabs(term) ? (total-updated)+term : (term-updated)+total;
        total=updated;
    }
    return total+correction;
}

int sealed_interp_column(const double *values, int64_t length,
                         const double *queries, int64_t count,
                         double u_min, double step, double u_max,
                         double tail_shift, double tail_limit,
                         double *output, int64_t *failed_index) {
    if (length<8 || !(step>0.0) || !isfinite(u_min) || !isfinite(u_max)) return 4;
    for (int64_t k=0;k<count;k++) {
        const double query=queries[k];
        if (!isfinite(query) || query>u_max) {*failed_index=k;return 1;}
        if (query<u_min) {
            if (query+tail_shift>-60.0) {*failed_index=k;return 2;}
            output[k]=tail_limit;continue;
        }
        const double position=(query-u_min)/step;
        int64_t start=(int64_t)floor(position)-3;
        if (start<0) start=0;
        if (start>length-8) start=length-8;
        double nodes[8], xs[8], weights[8], terms[8];
        int exact=-1;
        for (int i=0;i<8;i++) {
            nodes[i]=u_min+step*(double)(start+i);
            if (query==nodes[i] && exact<0) exact=i;
        }
        if (exact>=0) {output[k]=values[start+exact];continue;}
        const double node_origin=nodes[3];
        for (int i=0;i<8;i++) xs[i]=(nodes[i]-node_origin)/step;
        const double xq=(query-node_origin)/step;
        const double value_origin=values[start+3];
        for (int i=0;i<8;i++) {
            double weight=1.0;
            for (int j=0;j<8;j++) if (i!=j) weight/=xs[i]-xs[j];
            weights[i]=weight/(xq-xs[i]);
            terms[i]=weights[i]*(values[start+i]-value_origin);
        }
        output[k]=value_origin+compensated(terms)/compensated(weights);
        if (!isfinite(output[k])) {*failed_index=k;return 3;}
    }
    return 0;
}

/* Optional common-grid row batch. Python may retain ownership of all buffers;
 * this routine borrows them only during the call and never stores pointers. */
int sealed_interp_rows(const double *data, const int64_t *offsets,
                       const int64_t *lengths, const double *u_min,
                       const double *tail_shift, const double *tail_limit,
                       int64_t rows, const double *queries, int64_t count,
                       double step, double u_max, double *output,
                       int64_t *failed_row, int64_t *failed_index) {
    for (int64_t row=0;row<rows;row++) {
        int result=sealed_interp_column(data+offsets[row],lengths[row],queries,count,
             u_min[row],step,u_max,tail_shift[row],tail_limit[row],output+row*count,
             failed_index);
        if (result) {*failed_row=row;return result;}
    }
    return 0;
}
