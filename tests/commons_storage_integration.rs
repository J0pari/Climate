use std::fs;
use std::process::Command;

#[test]
fn pinned_commons_store_streams_and_reuses_content_identity() {
    let root = tempfile::tempdir().expect("store root");
    let source_dir = tempfile::tempdir().expect("source root");
    let source = source_dir.path().join("partition.ndjson");
    let payload = vec![0x31u8; 4 * 1024 * 1024 + 19];
    fs::write(&source, &payload).expect("source write");

    let bin = env!("CARGO_BIN_EXE_climate-commons-store");
    let first = Command::new(bin)
        .arg("put-file")
        .arg(root.path())
        .arg(&source)
        .output()
        .expect("first put-file invocation");
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let first_json: serde_json::Value =
        serde_json::from_slice(&first.stdout).expect("first receipt json");

    let second = Command::new(bin)
        .arg("put-file")
        .arg(root.path())
        .arg(&source)
        .output()
        .expect("second put-file invocation");
    assert!(second.status.success(), "{}", String::from_utf8_lossy(&second.stderr));
    let second_json: serde_json::Value =
        serde_json::from_slice(&second.stdout).expect("second receipt json");

    assert_eq!(first_json["schema"], "climate.commons-store-put/v1");
    assert_eq!(first_json["digest"], second_json["digest"]);
    assert_eq!(first_json["byte_count"], payload.len() as u64);

    let digest = first_json["digest"]
        .as_str()
        .expect("digest string")
        .strip_prefix("sha256:")
        .expect("sha256 prefix");
    let object = root.path().join("objects").join(&digest[..2]).join(digest);
    assert_eq!(fs::read(object).expect("stored bytes"), payload);
}

#[test]
fn adapter_refuses_missing_source_without_success_receipt() {
    let root = tempfile::tempdir().expect("store root");
    let missing = root.path().join("missing.bin");
    let bin = env!("CARGO_BIN_EXE_climate-commons-store");
    let output = Command::new(bin)
        .arg("put-file")
        .arg(root.path())
        .arg(&missing)
        .output()
        .expect("put-file invocation");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}
