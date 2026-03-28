use clap::Parser;
use imod_autodoc::Autodoc;
use std::fs;

/// Convert between IMOD autodoc and XML file formats.
///
/// Reads an autodoc (.adoc) file and writes XML, or reads a conforming XML file
/// and writes an autodoc.
#[derive(Parser)]
#[command(name = "adocxmlconv", about = "Convert autodoc to/from XML")]
struct Args {
    /// Root element name for XML output
    #[arg(short = 'r', long)]
    root_element: Option<String>,

    /// Input file (autodoc or XML)
    input: String,

    /// Output file (XML or autodoc)
    output: String,
}

fn main() {
    let args = Args::parse();

    // Try to detect if input is XML by extension or content
    let content = fs::read_to_string(&args.input).unwrap_or_else(|e| {
        eprintln!("Error reading input file {}: {}", args.input, e);
        std::process::exit(1);
    });

    let is_xml = args.input.ends_with(".xml")
        || content.trim_start().starts_with("<?xml")
        || content.trim_start().starts_with('<');

    if is_xml {
        // Convert XML to autodoc
        let adoc_text = xml_to_autodoc(&content);
        fs::write(&args.output, &adoc_text).unwrap_or_else(|e| {
            eprintln!("Error writing output file {}: {}", args.output, e);
            std::process::exit(1);
        });
        println!("Converted XML to autodoc");
    } else {
        // Parse as autodoc, write as XML
        let adoc = Autodoc::parse(&content);
        let root = args
            .root_element
            .as_deref()
            .unwrap_or("Autodoc");
        let xml_text = autodoc_to_xml(&adoc, root);
        fs::write(&args.output, &xml_text).unwrap_or_else(|e| {
            eprintln!("Error writing output file {}: {}", args.output, e);
            std::process::exit(1);
        });
        println!("Converted autodoc to XML file");
    }
}

/// Convert an Autodoc structure to an XML string.
fn autodoc_to_xml(adoc: &Autodoc, root_element: &str) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!("<{}>\n", xml_escape(root_element)));

    // Write global key-value pairs
    for (key, value) in &adoc.globals {
        out.push_str(&format!(
            "  <{0}>{1}</{0}>\n",
            xml_escape(key),
            xml_escape(value)
        ));
    }

    // Write sections
    for sec in &adoc.sections {
        out.push_str(&format!(
            "  <{} name=\"{}\">\n",
            xml_escape(&sec.kind),
            xml_escape(&sec.name)
        ));
        for (key, value) in &sec.values {
            out.push_str(&format!(
                "    <{0}>{1}</{0}>\n",
                xml_escape(key),
                xml_escape(value)
            ));
        }
        out.push_str(&format!("  </{}>\n", xml_escape(&sec.kind)));
    }

    out.push_str(&format!("</{}>\n", xml_escape(root_element)));
    out
}

/// Convert a simple XML string back to autodoc format.
///
/// This is a basic parser for the subset of XML that autodoc can represent.
fn xml_to_autodoc(xml: &str) -> String {
    let mut out = String::new();
    let mut in_section = false;
    let mut section_kind = String::new();

    for line in xml.lines() {
        let trimmed = line.trim();

        // Skip XML declaration and root element
        if trimmed.starts_with("<?xml") || trimmed.is_empty() {
            continue;
        }

        // Detect section open: <Kind name="Name">
        if trimmed.starts_with('<')
            && !trimmed.starts_with("</")
            && trimmed.contains(" name=\"")
            && trimmed.ends_with('>')
        {
            let tag_end = trimmed.find(' ').unwrap_or(1);
            let kind = &trimmed[1..tag_end];
            if let Some(name_start) = trimmed.find("name=\"") {
                let name_rest = &trimmed[name_start + 6..];
                if let Some(name_end) = name_rest.find('"') {
                    let name = &name_rest[..name_end];
                    out.push_str(&format!("\n[{} = {}]\n", kind, name));
                    section_kind = kind.to_string();
                    in_section = true;
                }
            }
            continue;
        }

        // Detect section close
        if in_section && trimmed.starts_with("</") && trimmed.contains(&section_kind) {
            in_section = false;
            continue;
        }

        // Detect simple element: <key>value</key>
        if trimmed.starts_with('<') && !trimmed.starts_with("</") && trimmed.contains("</") {
            let tag_end = trimmed.find('>').unwrap_or(1);
            let key = &trimmed[1..tag_end];
            let value_start = tag_end + 1;
            if let Some(close_start) = trimmed[value_start..].find("</") {
                let value = &trimmed[value_start..value_start + close_start];
                out.push_str(&format!("{} = {}\n", key, xml_unescape(value)));
            }
            continue;
        }
    }

    out
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
}
