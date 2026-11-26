use arrow_array::Int32Array;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use serde_json::{Value, json};

use crate::ffi::FfiDeltaScanContext;
use crate::reader::selection_from_deleted;
use crate::{delta_free_arrow_stream, new_delta_file_reader};

const DATA_FILE: &str = "part-00000-fae5310a-a37d-4e51-827b-c3d5516560ca-c000.snappy.parquet";

fn table_dir() -> std::path::PathBuf {
    std::fs::canonicalize(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test/tests/data/table-with-dv-small"
    ))
    .unwrap()
}

fn root_url() -> url::Url {
    url::Url::from_directory_path(table_dir()).unwrap()
}

/// The descriptor exactly as it sits in the log (storageType `u`). The table holds the values
/// 0..=9 and the DV deletes 0 and 9.
fn logged_descriptor() -> Value {
    let log =
        std::fs::read_to_string(table_dir().join("_delta_log/00000000000000000001.json")).unwrap();
    log.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find_map(|v| v.get("add").and_then(|a| a.get("deletionVector")).cloned())
        .expect("fixture has a DV")
}

fn ctx(root: &str, dv: &str) -> FfiDeltaScanContext {
    FfiDeltaScanContext {
        table_root: root.to_string(),
        file_path: table_dir().join(DATA_FILE).to_string_lossy().into_owned(),
        file_size: 635,
        dv_descriptor_json: dv.to_string(),
        projected_columns: vec![],
        batch_size: 0,
        props_keys: vec![],
        props_values: vec![],
    }
}

/// Goes through the same calls and Arrow C stream hand-off that Velox makes.
fn read_values(ctx: FfiDeltaScanContext) -> Result<Vec<i32>, String> {
    let reader = new_delta_file_reader(ctx)?;
    let ptr = reader.read()?;
    let stream = unsafe { ArrowArrayStreamReader::from_raw(ptr as *mut FFI_ArrowArrayStream) }
        .map_err(|e| e.to_string())?;
    let mut out = vec![];
    for batch in stream {
        let batch = batch.map_err(|e| e.to_string())?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        out.extend(col.values().iter().copied());
    }
    unsafe { delta_free_arrow_stream(ptr) };
    Ok(out)
}

const LIVE: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[test]
fn no_dv_reads_every_row() {
    assert_eq!(
        read_values(ctx("", "")).unwrap(),
        (0..10).collect::<Vec<_>>()
    );
}

#[test]
fn table_relative_dv_drops_deleted_rows() {
    let dv = logged_descriptor().to_string();
    assert_eq!(read_values(ctx(root_url().as_str(), &dv)).unwrap(), LIVE);
}

#[test]
fn table_root_without_trailing_slash_is_accepted() {
    let dv = logged_descriptor().to_string();
    let root = root_url().as_str().trim_end_matches('/').to_string();
    assert_eq!(read_values(ctx(&root, &dv)).unwrap(), LIVE);
}

#[test]
fn table_relative_dv_without_root_is_an_error_not_a_guess() {
    let dv = logged_descriptor().to_string();
    let err = read_values(ctx("", &dv)).unwrap_err();
    assert!(err.contains("no table root"), "{err}");
}

#[test]
fn extra_descriptor_fields_are_tolerated() {
    let mut dv = logged_descriptor();
    dv["maxRowIndex"] = json!(9);
    assert_eq!(
        read_values(ctx(root_url().as_str(), &dv.to_string())).unwrap(),
        LIVE
    );
}

#[test]
fn absolute_path_dv_needs_no_table_root() {
    let mut dv = logged_descriptor();
    let desc: DeletionVectorDescriptor = serde_json::from_value(dv.clone()).unwrap();
    let abs = desc.absolute_path(&root_url()).unwrap().expect("on disk");
    dv["storageType"] = json!("p");
    dv["pathOrInlineDv"] = json!(abs.to_string());
    assert_eq!(read_values(ctx("", &dv.to_string())).unwrap(), LIVE);
}

#[test]
fn bare_path_dv_is_rejected() {
    let mut dv = logged_descriptor();
    let desc: DeletionVectorDescriptor = serde_json::from_value(dv.clone()).unwrap();
    let abs = desc.absolute_path(&root_url()).unwrap().expect("on disk");
    dv["storageType"] = json!("p");
    dv["pathOrInlineDv"] = json!(abs.path());
    assert!(read_values(ctx("", &dv.to_string())).is_err());
}

#[test]
fn inline_dv_needs_no_table_root() {
    let u = logged_descriptor();
    let desc: DeletionVectorDescriptor = serde_json::from_value(u.clone()).unwrap();
    let abs = desc.absolute_path(&root_url()).unwrap().expect("on disk");
    let file = std::fs::read(abs.to_file_path().unwrap()).unwrap();
    let offset = u["offset"].as_u64().unwrap_or(1) as usize;
    let size = u["sizeInBytes"].as_u64().unwrap() as usize;
    // Skip the 4-byte big-endian length that precedes each DV in the file.
    let payload = &file[offset + 4..offset + 4 + size];
    let inline = json!({
        "storageType": "i",
        "pathOrInlineDv": z85_encode(payload),
        "sizeInBytes": size,
        "cardinality": u["cardinality"],
    });
    assert_eq!(read_values(ctx("", &inline.to_string())).unwrap(), LIVE);
}

#[test]
fn cardinality_mismatch_is_an_error() {
    let mut dv = logged_descriptor();
    dv["cardinality"] = json!(3);
    let err = read_values(ctx(root_url().as_str(), &dv.to_string())).unwrap_err();
    assert!(err.contains("descriptor says 3"), "{err}");
}

#[test]
fn malformed_descriptor_is_an_error() {
    assert!(read_values(ctx(root_url().as_str(), "{not json")).is_err());
}

#[test]
fn projection_reads_named_columns_and_missing_column_is_an_error() {
    let dv = logged_descriptor().to_string();
    let mut c = ctx(root_url().as_str(), &dv);
    c.projected_columns = vec!["value".into()];
    assert_eq!(read_values(c).unwrap(), LIVE);
    let mut c = ctx(root_url().as_str(), &dv);
    c.projected_columns = vec!["nope".into()];
    assert!(read_values(c).unwrap_err().contains("not found"));
}

#[test]
fn small_batches_do_not_change_the_result() {
    let dv = logged_descriptor().to_string();
    let mut c = ctx(root_url().as_str(), &dv);
    c.batch_size = 3;
    assert_eq!(read_values(c).unwrap(), LIVE);
}

#[test]
fn unsupported_locations_are_rejected() {
    let mut c = ctx("", "");
    c.file_path = "s3://bucket/key.parquet".into();
    assert!(new_delta_file_reader(c).is_err());
}

/// `RowSelection` into (selected, skipped) flags per row, for readable assertions.
fn keep_mask(deleted: &[u64], num_rows: u64) -> Vec<bool> {
    let mut mask = vec![];
    for s in selection_from_deleted(deleted, num_rows).unwrap().iter() {
        mask.extend(std::iter::repeat_n(!s.skip, s.row_count));
    }
    mask
}

#[test]
fn selection_covers_runs_edges_and_everything() {
    assert_eq!(keep_mask(&[], 3), [true, true, true]);
    assert_eq!(keep_mask(&[0], 3), [false, true, true]);
    assert_eq!(keep_mask(&[2], 3), [true, true, false]);
    assert_eq!(
        keep_mask(&[1, 2, 3, 6], 8),
        [true, false, false, false, true, true, false, true]
    );
    assert_eq!(keep_mask(&[0, 1, 2], 3), [false, false, false]);
    assert!(keep_mask(&[], 0).is_empty());
}

#[test]
fn selection_rejects_rows_past_the_end() {
    assert!(selection_from_deleted(&[3], 3).is_err());
}

const Z85: &[u8; 85] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

/// Z85 as Delta's Base85Codec does it: input is zero-padded to a multiple of 4 bytes.
fn z85_encode(bytes: &[u8]) -> String {
    let mut padded = bytes.to_vec();
    while !padded.len().is_multiple_of(4) {
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
