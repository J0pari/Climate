use std::env;
use std::path::PathBuf;

use commons_kernel::storage::FileCasStore;

fn usage() -> &'static str {
    "usage: climate-commons-store put-file <store-root> <input-file>"
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| usage().to_string())?;
    if command != "put-file" {
        return Err(usage().into());
    }
    let root = PathBuf::from(args.next().ok_or_else(|| usage().to_string())?);
    let input = PathBuf::from(args.next().ok_or_else(|| usage().to_string())?);
    if args.next().is_some() {
        return Err(usage().into());
    }

    let store = FileCasStore::open(root)?;
    let (digest, byte_count) = store.put_file(&input)?;
    println!(
        "{{\"schema\":\"climate.commons-store-put/v1\",\"digest\":\"sha256:{}\",\"byte_count\":{}}}",
        digest, byte_count
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
