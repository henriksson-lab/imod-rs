/* LD_PRELOAD recorder for the qtplax entry points of the reference
   libdnmncar.so, used by make-pyscript-goldens.py for the native genhstplt:
   appends one line per plax_* call to $PLAX_CALL_LOG (the format of the
   Rust qtplax.rs call log), then forwards to the real routine. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static FILE *logfp(void) {
  static FILE *fp = NULL; static int tried = 0;
  if (!tried) { const char *n = getenv("PLAX_CALL_LOG"); tried = 1; if (n) fp = fopen(n, "w"); }
  return fp;
}
#define REAL(name, ret, args) static ret (*real_##name) args = NULL; if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)
#define LOG(...) do { FILE *f = logfp(); if (f) { fprintf(f, __VA_ARGS__); fflush(f);} } while (0)
int plax_open_(void) { REAL(plax_open_, int, (void)); LOG("open\n"); return real_plax_open_(); }
void plax_close_(void) { REAL(plax_close_, void, (void)); LOG("close\n"); real_plax_close_(); }
void plax_flush_(void) { REAL(plax_flush_, void, (void)); LOG("flush\n"); real_plax_flush_(); }
void plax_erase_(void) { REAL(plax_erase_, void, (void)); LOG("erase\n"); real_plax_erase_(); }
void plax_wait_for_close_(void) { REAL(plax_wait_for_close_, void, (void)); LOG("wait\n"); real_plax_wait_for_close_(); }
void plax_mapcolor_(int *c, int *r, int *g, int *b) { REAL(plax_mapcolor_, void, (int*,int*,int*,int*)); LOG("mapcolor %d %d %d %d\n", *c, *r, *g, *b); real_plax_mapcolor_(c, r, g, b); }
#define FIVE(nm, tag) void nm(int *c, int *a, int *b, int *d, int *e) { REAL(nm, void, (int*,int*,int*,int*,int*)); LOG(tag " %d %d %d %d %d\n", *c, *a, *b, *d, *e); real_##nm(c, a, b, d, e); }
FIVE(plax_box_, "box")
FIVE(plax_boxo_, "boxo")
FIVE(plax_vect_, "vect")
void plax_vectw_(int *w, int *c, int *a, int *b, int *d, int *e) { REAL(plax_vectw_, void, (int*,int*,int*,int*,int*,int*)); LOG("vectw %d %d %d %d %d %d\n", *w, *c, *a, *b, *d, *e); real_plax_vectw_(w, c, a, b, d, e); }
void plax_circ_(int *c, int *r, int *x, int *y) { REAL(plax_circ_, void, (int*,int*,int*,int*)); LOG("circ %d %d %d %d\n", *c, *r, *x, *y); real_plax_circ_(c, r, x, y); }
void plax_circo_(int *c, int *r, int *x, int *y) { REAL(plax_circo_, void, (int*,int*,int*,int*)); LOG("circo %d %d %d %d\n", *c, *r, *x, *y); real_plax_circo_(c, r, x, y); }
static void logpoly(const char *tag, int c, int n, short *v) { FILE *f = logfp(); int i; if (!f) return; fprintf(f, "%s %d %d", tag, c, n); for (i = 0; i < 2 * n; i++) fprintf(f, " %d", v[i]); fprintf(f, "\n"); fflush(f); }
void plax_poly_(int *c, int *n, short *v) { REAL(plax_poly_, void, (int*,int*,short*)); logpoly("poly", *c, *n, v); real_plax_poly_(c, n, v); }
void plax_polyo_(int *c, int *n, short *v) { REAL(plax_polyo_, void, (int*,int*,short*)); logpoly("polyo", *c, *n, v); real_plax_polyo_(c, n, v); }
void plax_sctext_(int *t, int *xs, int *ys, int *c, int *x, int *y, char *s, size_t len) {
  REAL(plax_sctext_, void, (int*,int*,int*,int*,int*,int*,char*,size_t));
  FILE *f = logfp(); if (f) { fprintf(f, "sctext %d %d %d %d %d %d %d [", *t, *xs, *ys, *c, *x, *y, (int)len); fwrite(s, 1, len, f); fprintf(f, "]\n"); fflush(f); }
  real_plax_sctext_(t, xs, ys, c, x, y, s, len);
}
void plax_next_text_align_(int *t) { REAL(plax_next_text_align_, void, (int*)); LOG("align %d\n", *t); real_plax_next_text_align_(t); }
void plax_drawing_scale_(float *a, float *b, float *c, float *d) { REAL(plax_drawing_scale_, void, (float*,float*,float*,float*)); LOG("scale %g %g %g %g\n", *a, *b, *c, *d); real_plax_drawing_scale_(a, b, c, d); }
int plax_save_png_(const char *n, size_t len) { REAL(plax_save_png_, int, (const char*,size_t)); FILE *f = logfp(); if (f) { fprintf(f, "savepng %d [", (int)len); fwrite(n, 1, len, f); fprintf(f, "]\n"); fflush(f);} return real_plax_save_png_(n, len); }
int plax_save_tiff_(const char *n, size_t len) { REAL(plax_save_tiff_, int, (const char*,size_t)); FILE *f = logfp(); if (f) { fprintf(f, "savetiff %d [", (int)len); fwrite(n, 1, len, f); fprintf(f, "]\n"); fflush(f);} return real_plax_save_tiff_(n, len); }
