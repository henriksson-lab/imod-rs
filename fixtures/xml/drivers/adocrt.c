/* autodoc round-trip: adocrt report.txt outdir asxml file... */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "autodoc.h"
int main(int argc, char **argv)
{
  FILE *rep = fopen(argv[1], "w");
  char *outdir = argv[2];
  int asxml = atoi(argv[3]);
  char out[2048];
  int i, ind, err;
  const char *base;
  char *root;
  for (i = 4; i < argc; i++) {
    base = strrchr(argv[i], '/');
    base = base ? base + 1 : argv[i];
    ind = AdocRead(argv[i]);
    fprintf(rep, "%s read=%d", base, ind);
    if (ind < 0) { fprintf(rep, "\n"); continue; }
    root = NULL;
    err = AdocGetXmlRootElement(&root);
    fprintf(rep, " rootErr=%d root=%s", err, root ? root : "(nil)");
    if (root) free(root);
    fprintf(rep, " xmlRead=%d", AdocGetWriteAsXML());
    AdocSetWriteAsXML(asxml);
    snprintf(out, sizeof(out), "%s/%s.out", outdir, base);
    fprintf(rep, " write=%d\n", AdocWrite(out));
    AdocClear(ind);
    AdocSetWriteAsXML(0);
  }
  fclose(rep);
  return 0;
}
