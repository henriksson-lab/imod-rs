#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mrcfiles.h"
#include "iimage.h"

static void dump(const char *tag, void *buf, int n, int esize)
{
  unsigned char *p = (unsigned char *)buf;
  unsigned long long sum = 0;
  int i;
  for (i = 0; i < n * esize; i++) sum = sum * 131 + p[i];
  printf("%s n=%d sum=%llu first=", tag, n, sum);
  for (i = 0; i < 8 && i < n * esize; i++) printf("%02x", p[i]);
  printf("\n");
}

int main(int argc, char **argv)
{
  MrcHeader h;
  IloadInfo li;
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { printf("open fail\n"); return 1; }
  if (mrc_head_read(fp, &h)) { printf("head fail\n"); return 1; }
  h.fp = fp;
  int nx = h.nx, ny = h.ny, nz = h.nz, z = nz / 2;
  unsigned char *b = malloc((size_t)nx * (ny > nz ? ny : nz) * 16 + 4096);
  float *f = (float *)b;

  mrc_init_li(&li, NULL);
  mrc_init_li(&li, &h);
  printf("li xmin %d xmax %d ymin %d ymax %d mode %d\n", li.xmin, li.xmax, li.ymin, li.ymax, h.mode);

  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadSection(&h, &li, b, z));       dump("sec", b, nx*ny, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadSectionByte(&h, &li, b, z));   dump("secB", b, nx*ny, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadSectionUShort(&h, &li, b, z)); dump("secU", b, nx*ny, 2);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadSectionFloat(&h, &li, f, z));  dump("secF", b, nx*ny, 4);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadZ(&h, &li, b, z));             dump("z", b, nx*ny, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadZByte(&h, &li, b, z));         dump("zB", b, nx*ny, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadZUShort(&h, &li, b, z));       dump("zU", b, nx*ny, 2);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadZFloat(&h, &li, f, z));        dump("zF", b, nx*ny, 4);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadY(&h, &li, b, 1));             dump("y", b, nx*nz, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadYByte(&h, &li, b, 1));         dump("yB", b, nx*nz, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadYUShort(&h, &li, b, 1));       dump("yU", b, nx*nz, 2);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadYFloat(&h, &li, f, 1));        dump("yF", b, nx*nz, 4);

  /* Sub-rectangle and scaling */
  li.xmin = 3; li.xmax = nx - 4; li.ymin = 2; li.ymax = ny - 3;
  li.slope = 1.5f; li.offset = -3.f; li.smin = 0.f; li.smax = 0.f;
  int rw = li.xmax - li.xmin + 1, rh = li.ymax - li.ymin + 1;
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadSectionByte(&h, &li, b, z));   dump("subB", b, rw*rh, 1);
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadSectionFloat(&h, &li, f, z));  dump("subF", b, rw*rh, 4);
  li.axis = 2;
  memset(b, 0, (size_t)nx*(ny>nz?ny:nz)*16); printf("rc=%d ", mrcReadYFloat(&h, &li, f, 2));        dump("subYF", b, rw*nz, 4);
  return 0;
}
