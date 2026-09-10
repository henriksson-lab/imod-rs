// Direct Qt ABI boundary for the QImage block in mrc2tif.cpp:260-280,583-596.
// Qt retains responsibility for JPEG/PNG encoding and plugin discovery.
#include <QCoreApplication>
#include <QDir>
#include <QImage>
#include <QStringList>

#include <cstdlib>

extern "C" int mrc2tif_qimage_save(unsigned char *data, int width, int height,
                                    int bytes_per_line, int rgb, int resolution,
                                    const char *filename, const char *format, int quality)
{
  QStringList paths;
  const char *plugin_dir = std::getenv("IMOD_PLUGIN_DIR");
  if (plugin_dir)
    paths << plugin_dir;
  const char *imod_dir = std::getenv("IMOD_DIR");
  if (imod_dir)
    paths << QString(imod_dir) + QString("/lib/imodplug");
  for (int i = 0; i < paths.count(); i++)
    if (QDir(paths[i] + "/imageformats").exists()) {
      QCoreApplication::setLibraryPaths(paths);
      break;
    }

  QImage image(data, width, height, bytes_per_line,
               rgb ? QImage::Format_RGB888 : QImage::Format_Indexed8);
  if (resolution) {
    image.setDotsPerMeterX(resolution / (resolution > 0 ? 0.01 : -0.0254));
    image.setDotsPerMeterY(resolution / (resolution > 0 ? 0.01 : -0.0254));
  }
  if (!rgb)
    for (int i = 0; i < 256; i++)
      image.setColor(i, qRgb(i, i, i));
  return image.save(filename, format, quality) ? 0 : 1;
}
