use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use clap::Parser;
use imod_mrc::MrcReader;
use tiff::encoder::colortype::Gray32Float;
use tiff::encoder::TiffEncoder;

/// Convert an MRC file to a multi-page TIFF file.
#[derive(Parser)]
#[command(name = "mrc2tif")]
struct Args {
    /// Input MRC file
    input: PathBuf,

    /// Output TIFF file
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut reader = MrcReader::open(&args.input)?;
    let header = reader.header().clone();
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;

    eprintln!(
        "Converting {} ({nx}x{ny}x{nz}) to {}",
        args.input.display(),
        args.output.display()
    );

    let file = File::create(&args.output)?;
    let buf = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(buf)?;

    for z in 0..nz {
        let data = reader.read_slice_f32(z)?;
        encoder.write_image::<Gray32Float>(nx as u32, ny as u32, &data)?;
    }

    eprintln!("Wrote {nz} pages.");
    Ok(())
}
