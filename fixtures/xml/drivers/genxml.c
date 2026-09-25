/* Generate a native XML corpus from IMOD .adoc files. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "autodoc.h"

int main(int argc, char **argv)
{
  int i, ind;
  char out[1024];
  for (i = 1; i < argc; i++) {
    ind = AdocRead(argv[i]);
    if (ind < 0) { printf("SKIP %s %d\n", argv[i], ind); continue; }
    AdocSetWriteAsXML(1);
    const char *base = strrchr(argv[i], '/');
    base = base ? base + 1 : argv[i];
    snprintf(out, sizeof(out), "%s/%s.xml", argv[argc-1], base);
    if (i == argc - 1) break;
    printf("OK %s %d %d\n", base, ind, AdocWrite(out));
    AdocClear(ind);
    AdocSetWriteAsXML(0);
  }
  return 0;
}
