use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use serde_json::{Value, json};

use crate::ffi::FfiDeltaScanContext;
use crate::reader::drop_deleted;
use crate::{delta_free_arrow_stream, new_delta_file_reader};

const VALUE_INT: &str = r#"{"type":"struct","fields":[{"name":"value","type":"integer","nullable":true,"metadata":{}}]}"#;
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
        physical_schema_json: VALUE_INT.to_string(),
        props_keys: vec![],
        props_values: vec![],
    }
}

/// Goes through the same calls and Arrow C stream hand-off that Velox makes.
fn read_batches(ctx: FfiDeltaScanContext) -> Result<Vec<RecordBatch>, String> {
    let reader = new_delta_file_reader(ctx)?;
    let ptr = reader.read()?;
    let stream = unsafe { ArrowArrayStreamReader::from_raw(ptr as *mut FFI_ArrowArrayStream) }
        .map_err(|e| e.to_string());
    let out = stream.and_then(|s| s.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string()));
    unsafe { delta_free_arrow_stream(ptr) };
    out
}

fn read_values(ctx: FfiDeltaScanContext) -> Result<Vec<i32>, String> {
    let mut out = vec![];
    for batch in read_batches(ctx)? {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        out.extend(col.values().iter().copied());
    }
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
fn unsupported_locations_are_rejected() {
    let mut c = ctx("", "");
    c.file_path = "s3://bucket/key.parquet".into();
    assert!(new_delta_file_reader(c).is_err());
}

#[test]
fn physical_schema_is_required() {
    let mut c = ctx("", "");
    c.physical_schema_json = String::new();
    assert!(new_delta_file_reader(c).err().unwrap().contains("required"));
    let mut c = ctx("", "");
    c.physical_schema_json = "{nope".into();
    assert!(new_delta_file_reader(c).is_err());
}

#[test]
fn kernel_widens_integer_to_long() {
    let dv = logged_descriptor().to_string();
    let mut c = ctx(root_url().as_str(), &dv);
    c.physical_schema_json = VALUE_INT.replace("integer", "long");
    let batches = read_batches(c).unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        col.values().iter().copied().collect::<Vec<_>>(),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
}

#[test]
fn column_absent_from_the_file() {
    // What the kernel does with a column the file does not have (e.g. added by schema evolution).
    let dv = logged_descriptor().to_string();
    let mut c = ctx(root_url().as_str(), &dv);
    c.physical_schema_json = r#"{"type":"struct","fields":[
        {"name":"value","type":"integer","nullable":true,"metadata":{}},
        {"name":"added","type":"string","nullable":true,"metadata":{}}]}"#
        .into();
    let batches = read_batches(c).unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 8);
    let added = batches[0].column(1);
    assert_eq!(
        added.null_count(),
        added.len(),
        "absent nullable column should read as null"
    );
}

#[test]
fn batch_schema_matches_stream_schema() {
    let dv = logged_descriptor().to_string();
    let batches = read_batches(ctx(root_url().as_str(), &dv)).unwrap();
    assert_eq!(batches[0].schema().field(0).name(), "value");
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &arrow_schema::DataType::Int32
    );
}

/// Builds `n` rows with values `start..start+n`.
fn batch(start: i32, n: i32) -> RecordBatch {
    let col = Int32Array::from((start..start + n).collect::<Vec<_>>());
    RecordBatch::try_from_iter([("v", std::sync::Arc::new(col) as _)]).unwrap()
}

fn live_values(batches: &[(i32, i32)], deleted: &[u64]) -> Vec<i32> {
    let mut cursor = 0;
    let mut offset = 0u64;
    let mut out = vec![];
    for &(start, n) in batches {
        let b = drop_deleted(batch(start, n), offset, deleted, &mut cursor).unwrap();
        out.extend(
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
        );
        offset += n as u64;
    }
    out
}

#[test]
fn deleted_rows_are_dropped_across_batch_boundaries() {
    // Two batches of 4: rows 0..4 and 4..8. Delete one at each edge of the boundary, plus a run.
    assert_eq!(live_values(&[(0, 4), (4, 4)], &[3, 4]), [0, 1, 2, 5, 6, 7]);
    assert_eq!(
        live_values(&[(0, 4), (4, 4)], &[]),
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(live_values(&[(0, 4), (4, 4)], &[0, 1, 2, 3]), [4, 5, 6, 7]);
    assert_eq!(live_values(&[(0, 4), (4, 4)], &[7]), [0, 1, 2, 3, 4, 5, 6]);
    // A batch with nothing deleted in it passes through untouched.
    assert_eq!(
        live_values(&[(0, 2), (2, 2), (4, 2)], &[4]),
        [0, 1, 2, 3, 5]
    );
    // Everything deleted leaves an empty batch, not an error.
    assert!(live_values(&[(0, 3)], &[0, 1, 2]).is_empty());
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
