//! Spike (ENG-52692): can a DV be resolved from nothing but the descriptor JSON that Spark/Delta
//! hands the scan (`row_index_filter_id_encoded` decoded to JSON), without a table snapshot?
//!
//! Builds descriptors of every storage type from a real fixture and checks each one decodes to
//! the same deleted row positions.
//! Usage: cargo run -p deltalake-core --example dv_descriptor_spike -- <table-with-dv dir>
use std::sync::Arc;

use delta_kernel::Engine;
use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel_default_engine::DefaultEngineBuilder;
use object_store::local::LocalFileSystem;
use serde_json::{Value, json};

const Z85: &[u8; 85] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

/// Z85 encode; input is zero-padded to a multiple of 4 bytes (as Delta's Base85Codec does).
fn z85_encode(bytes: &[u8]) -> String {
    let mut padded = bytes.to_vec();
    while padded.len() % 4 != 0 {
        padded.push(0);
    }
    let mut out = String::new();
    for chunk in padded.chunks(4) {
        let mut v = u32::from_be_bytes(chunk.try_into().unwrap()) as u64;
        let mut c = [0u8; 5];
        for i in (0..5).rev() {
            c[i] = Z85[(v % 85) as usize];
            v /= 85;
        }
        out.push_str(std::str::from_utf8(&c).unwrap());
    }
    out
}

fn positions(
    engine: &impl Engine,
    root: &url::Url,
    json: &Value,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let d: DeletionVectorDescriptor = serde_json::from_value(json.clone())?;
    Ok(d.row_indexes(engine.storage_handler(), root)?)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::fs::canonicalize(std::env::args().nth(1).expect("table dir"))?;
    let root = url::Url::from_directory_path(&dir).unwrap();
    let engine = DefaultEngineBuilder::new(Arc::new(LocalFileSystem::new())).build();

    // 1. The descriptor exactly as it sits in the log (storageType "u").
    let mut original: Option<Value> = None;
    let mut log: Vec<_> = std::fs::read_dir(dir.join("_delta_log"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    log.sort();
    for p in &log {
        for line in std::fs::read_to_string(p)?.lines() {
            let v: Value = serde_json::from_str(line)?;
            if let Some(dv) = v.get("add").and_then(|a| a.get("deletionVector")) {
                original = Some(dv.clone());
            }
        }
    }
    let mut u = original.expect("no add with deletionVector in the log");
    // Delta 3.3 serializes an extra `maxRowIndex`; make sure it is tolerated.
    u["maxRowIndex"] = json!(9);
    println!("descriptor json (u): {u}");
    let want = positions(&engine, &root, &u)?;
    println!("u  (UUID file)         -> {want:?}");

    // 2. Same DV as storageType "p", for the URL spellings a JVM could send.
    let desc: DeletionVectorDescriptor = serde_json::from_value(u.clone())?;
    let abs = desc.absolute_path(&root)?.expect("on disk");
    let spellings = [
        ("p  file:///abs", abs.to_string()),
        ("p  file:/abs (Hadoop)", format!("file:{}", abs.path())),
    ];
    for (label, path) in spellings {
        let mut p = u.clone();
        p["storageType"] = json!("p");
        p["pathOrInlineDv"] = json!(path);
        match positions(&engine, &root, &p) {
            Ok(got) => println!(
                "{label:<22} -> {got:?} {}",
                if got == want { "OK" } else { "MISMATCH" }
            ),
            Err(e) => println!("{label:<22} -> ERROR {e}"),
        }
    }
    // A bare local path has no scheme and must be rejected, not guessed.
    let mut bare = u.clone();
    bare["storageType"] = json!("p");
    bare["pathOrInlineDv"] = json!(abs.path());
    println!(
        "p  bare /abs           -> {}",
        match positions(&engine, &root, &bare) {
            Ok(g) => format!("accepted {g:?}"),
            Err(e) => format!("rejected: {e}"),
        }
    );

    // 3. Same DV inline: bytes [magic + bitmap] cut out of the .bin file, z85 encoded.
    let file = std::fs::read(abs.to_file_path().unwrap())?;
    let offset = u["offset"].as_u64().unwrap_or(1) as usize;
    let size = u["sizeInBytes"].as_u64().unwrap() as usize;
    let payload = &file[offset + 4..offset + 4 + size]; // skip the 4-byte big-endian length
    let mut inline = json!({
        "storageType": "i",
        "pathOrInlineDv": z85_encode(payload),
        "sizeInBytes": size,
        "cardinality": u["cardinality"],
    });
    match positions(&engine, &root, &inline) {
        Ok(got) => println!(
            "i  (inline, {} chars)  -> {got:?} {}",
            inline["pathOrInlineDv"].as_str().unwrap().len(),
            if got == want { "OK" } else { "MISMATCH" }
        ),
        Err(e) => println!("i  (inline)            -> ERROR {e}"),
    }
    // Inline needs no table root at all.
    inline["cardinality"] = json!(want.len());
    let nowhere = url::Url::parse("s3://does-not-exist/table/")?;
    println!(
        "i  with unrelated root -> {:?}",
        positions(&engine, &nowhere, &inline).ok()
    );
    Ok(())
}
