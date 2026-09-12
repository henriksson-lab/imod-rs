// Direct Qt ABI boundary for IMOD/3dmod/iiqimage.cpp.  Rust retains the
// source's ImodImageFile state and conversion loops; this file owns only the
// C++ QImage object and QImage member calls that cannot cross the Rust ABI.
#include <QImage>
#include <QString>

extern "C" void *iiqimage_open(const char *filename)
{
  return new QImage(QString(filename));
}

extern "C" void iiqimage_delete(void *image)
{
  delete static_cast<QImage *>(image);
}

extern "C" int iiqimage_is_null(void *image)
{
  return !image || static_cast<QImage *>(image)->isNull();
}

extern "C" int iiqimage_width(void *image)
{
  return static_cast<QImage *>(image)->width();
}

extern "C" int iiqimage_height(void *image)
{
  return static_cast<QImage *>(image)->height();
}

extern "C" int iiqimage_depth(void *image)
{
  return static_cast<QImage *>(image)->depth();
}

extern "C" int iiqimage_is_grayscale(void *image)
{
  return static_cast<QImage *>(image)->isGrayscale();
}

extern "C" int iiqimage_color_count(void *image)
{
  return static_cast<QImage *>(image)->colorTable().size();
}

extern "C" unsigned int iiqimage_color(void *image, int index)
{
  return static_cast<QImage *>(image)->colorTable()[index];
}

// Copies one QImage scan line as its source index bytes.  This is called only
// for the source's depth-8 paths.
extern "C" void iiqimage_index_row(void *image, int y, unsigned char *out, int width)
{
  const uchar *line = static_cast<QImage *>(image)->constScanLine(y);
  for (int x = 0; x < width; x++)
    out[x] = line[x];
}

// Equivalent to the source's QRgb scan-line access, including the 16-bit
// QImage::convertToFormat(Format_RGB32) case: QImage::pixel supplies qRed,
// qGreen, and qBlue in all non-indexed formats.
extern "C" void iiqimage_rgb_row(void *image, int y, unsigned char *out, int width)
{
  QImage *qimage = static_cast<QImage *>(image);
  for (int x = 0; x < width; x++) {
    QRgb pixel = qimage->pixel(x, y);
    out[3 * x] = qRed(pixel);
    out[3 * x + 1] = qGreen(pixel);
    out[3 * x + 2] = qBlue(pixel);
  }
}
