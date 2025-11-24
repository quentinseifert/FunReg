/* splines_standalone_full.c - B-splines, full n x k basis matrix */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- B-spline structure --- */
typedef struct {
    int order, ordm1, nknots, curs, boundary;
    double *ldel, *rdel, *knots, *a;
} spl_struct;

/* --- cursor --- */
static int set_cursor(spl_struct *sp, double x) {
    sp->curs = -1;
    sp->boundary = 0;
    for (int i = 0; i < sp->nknots; i++) {
        if (sp->knots[i] >= x) sp->curs = i;
        if (sp->knots[i] > x) break;
    }
    if (sp->curs > sp->nknots - sp->order) {
        int lastLegit = sp->nknots - sp->order;
        if (x == sp->knots[lastLegit]) { sp->boundary = 1; sp->curs = lastLegit; }
    }
    return sp->curs;
}

/* --- difference table --- */
static void diff_table(spl_struct *sp, double x, int ndiff) {
    for (int i = 0; i < ndiff; i++) {
        sp->rdel[i] = sp->knots[sp->curs + i] - x;
        sp->ldel[i] = x - sp->knots[sp->curs - (i + 1)];
    }
}

/* --- fast basis functions --- */
static void basis_funcs(spl_struct *sp, double x, double *b) {
    diff_table(sp, x, sp->ordm1);
    b[0] = 1.;
    for (int j = 1; j <= sp->ordm1; j++) {
        double saved = 0.;
        for (int r = 0; r < j; r++) {
            double den = sp->rdel[r] + sp->ldel[j - 1 - r];
            if (den != 0.) {
                double term = b[r]/den;
                b[r] = saved + sp->rdel[r] * term;
                saved = sp->ldel[j - 1 - r] * term;
            } else {
                if (r != 0 || sp->rdel[r] != 0.) b[r] = saved;
                saved = 0.;
            }
        }
        b[j] = saved;
    }
}

/* --- compute full n x k basis matrix --- */
void spline_basis_matrix_full(const double *knots, int nknots, int order,
                              const double *xvals, int nx, int k,
                              double *out) {
    spl_struct sp;
    sp.order = order;
    sp.ordm1 = order-1;
    sp.nknots = nknots;
    sp.knots = (double*)knots;
    double ldel[order], rdel[order], a[order];
    sp.ldel = ldel; sp.rdel = rdel; sp.a = a;

    double temp[order];

    for (int i = 0; i < nx; i++) {
        int curs = set_cursor(&sp, xvals[i]);
        basis_funcs(&sp, xvals[i], temp);

        // zero the full row first
        for (int j = 0; j < k; j++) out[i*k + j] = 0.0;

        // compute first column and handle boundaries
        int first_col = curs - order; // FIXED: removed +1
        int start = 0;
        if (first_col < 0) {
            start = -first_col;
            first_col = 0;
        }

        for (int j = start; j < order; j++) {
            int col = first_col + j; // FIXED: no -start
            if (col < k) out[i*k + col] = temp[j];
        }
    }
}

/* --- wrapper for Python ctypes --- */
void spline_basis_matrix_full_c(const double *knots, int nknots, int order,
                                const double *xvals, int nx, int k,
                                double *out) {
    spline_basis_matrix_full(knots, nknots, order, xvals, nx, k, out);
}

/* --- optional standalone test --- */
#ifdef TEST_SPLINE
int main() {
    int k = 6;
    int order = 4;
    double knots[10] = {0,0,0,0,1,2,3,3,3,3};
    int nknots = 10;
    double xvals[] = {0.5, 1.5, 2.5};
    int nx = 3;
    double mat[nx*k];

    spline_basis_matrix_full(knots, nknots, order, xvals, nx, k, mat);

    for(int i=0;i<nx;i++){
        for(int j=0;j<k;j++) printf("%f ", mat[i*k+j]);
        printf("\n");
    }
    return 0;
}
#endif

