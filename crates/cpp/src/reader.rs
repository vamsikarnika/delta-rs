//! Reads one Delta data file with its deletion vector applied.
//!
//! A split carries the file, the DV descriptor JSON that Spark hands the scan, and the physical
//! schema to read. No table snapshot is needed (see `crates/core/examples/dv_descriptor_spike.rs`
//! on the spike branch). Parquet reading, column matching (field id, then name) and type widening
//! are the kernel's; this module adds the DV and the Arrow stream.
use std::sync::{Arc, OnceLock};

use arrow_array::{BooleanArray, RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use arrow_select::filter::filter_record_batch;
use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::schema::{SchemaRef, StructType};
use delta_kernel::{Engine, FileMeta};
use delta_kernel_default_engine::executor::tokio::TokioBackgroundExecutor;
use delta_kernel_default_engine::{DefaultEngine, DefaultEngineBuilder};
use object_store::local::LocalFileSystem;

type SharedEngine = DefaultEngine<TokioBackgroundExecutor>;

/// One engine for every split: it owns a background executor thread, so building one per split
/// would spawn a thread per file.
pub fn shared_engine() -> &'static SharedEngine {
    static ENGINE: OnceLock<SharedEngine> = OnceLock::new();
    ENGINE.get_or_init(|| DefaultEngineBuilder::new(Arc::new(LocalFileSystem::new())).build())
}

/// A data file as the kernel wants it.
pub struct DataFile {
    pub meta: FileMeta,
}

/// Accepts an absolute local path or a `file://` URL. Other schemes are not supported yet.
/// `file_size <= 0` means unknown; it is looked up from the filesystem.
pub fn resolve_file(file_path: &str, file_size: i64) -> Result<DataFile, String> {
    let local: std::path::PathBuf = if file_path.starts_with("file:") {
        let url =
            url::Url::parse(file_path).map_err(|e| format!("bad file url {file_path}: {e}"))?;
        url.to_file_path()
            .map_err(|_| format!("file url is not a local path: {file_path}"))?
    } else if file_path.starts_with('/') {
        file_path.into()
    } else {
        return Err(format!(
            "unsupported data file location (local paths only for now): {file_path}"
        ));
    };
    let size = if file_size > 0 {
        file_size as u64
    } else {
        std::fs::metadata(&local)
            .map_err(|e| format!("cannot stat {file_path}: {e}"))?
            .len()
    };
    let location = url::Url::from_file_path(&local)
        .map_err(|_| format!("not an absolute path: {file_path}"))?;
    Ok(DataFile {
        meta: FileMeta {
            location,
            last_modified: 0,
            size,
        },
    })
}

/// Parses the physical schema: Delta schema JSON, as in the table's `schemaString` but with
/// physical column names (column mapping) and field-id metadata where the table has them.
pub fn parse_schema(schema_json: &str) -> Result<SchemaRef, String> {
    if schema_json.is_empty() {
        return Err("physical_schema_json is required".into());
    }
    let schema: StructType =
        serde_json::from_str(schema_json).map_err(|e| format!("bad physical schema: {e}"))?;
    Ok(Arc::new(schema))
}

/// Resolves a DV descriptor JSON to the sorted positions of deleted rows. `None` means no DV.
///
/// `table_root` is a directory URL, needed only for storage type `u` (a UUID file relative to the
/// table). Inline (`i`) and absolute-path (`p`) DVs resolve without it.
pub fn resolve_deleted_rows(
    descriptor_json: &str,
    table_root: &str,
    engine: &impl Engine,
) -> Result<Option<Vec<u64>>, String> {
    if descriptor_json.is_empty() {
        return Ok(None);
    }
    let descriptor: DeletionVectorDescriptor = serde_json::from_str(descriptor_json)
        .map_err(|e| format!("bad deletion vector descriptor: {e}"))?;
    let root = if table_root.is_empty() {
        let is_table_relative = serde_json::from_str::<serde_json::Value>(descriptor_json)
            .ok()
            .and_then(|v| {
                v.get("storageType")
                    .and_then(|s| s.as_str())
                    .map(|s| s == "u")
            })
            .unwrap_or(false);
        if is_table_relative {
            return Err(
                "deletion vector is table-relative (storageType u) but no table root was given"
                    .into(),
            );
        }
        // Inline and absolute-path DVs never look at the root.
        url::Url::parse("file:///").expect("valid url")
    } else {
        let mut root =
            url::Url::parse(table_root).map_err(|e| format!("bad table root {table_root}: {e}"))?;
        if !root.path().ends_with('/') {
            root.set_path(&format!("{}/", root.path()));
        }
        root
    };
    let rows = descriptor
        .row_indexes(engine.storage_handler(), &root)
        .map_err(|e| format!("failed to read deletion vector: {e}"))?;
    // The kernel does not compare what it decoded with what the log says. A mismatch means a
    // truncated or wrong file, and reading on would return rows that should be deleted.
    if rows.len() as i64 != descriptor.cardinality {
        return Err(format!(
            "deletion vector has {} rows but its descriptor says {}",
            rows.len(),
            descriptor.cardinality
        ));
    }
    if !rows.windows(2).all(|w| w[0] < w[1]) {
        return Err("deletion vector positions are not sorted and unique".into());
    }
    Ok(Some(rows))
}

/// Drops the deleted rows from `batch`, which starts at file row `row_offset`. `cursor` indexes the
/// first entry of `deleted` that is not yet behind the batch; it advances as batches go by.
pub fn drop_deleted(
    batch: RecordBatch,
    row_offset: u64,
    deleted: &[u64],
    cursor: &mut usize,
) -> Result<RecordBatch, ArrowError> {
    let end = row_offset + batch.num_rows() as u64;
    let start_cursor = *cursor;
    while *cursor < deleted.len() && deleted[*cursor] < end {
        *cursor += 1;
    }
    let in_batch = &deleted[start_cursor..*cursor];
    if in_batch.is_empty() {
        return Ok(batch);
    }
    let mut keep = vec![true; batch.num_rows()];
    for d in in_batch {
        keep[(*d - row_offset) as usize] = false;
    }
    filter_record_batch(&batch, &BooleanArray::from(keep))
}

type KernelBatches = delta_kernel::FileDataReadResultIterator;

fn external<E: std::error::Error + Send + Sync + 'static>(e: E) -> ArrowError {
    ArrowError::ExternalError(Box::new(e))
}

/// Streams the live rows of one file, one kernel batch at a time.
struct LiveRows {
    batches: KernelBatches,
    schema: ArrowSchemaRef,
    deleted: Vec<u64>,
    cursor: usize,
    row_offset: u64,
    done: bool,
}

impl Iterator for LiveRows {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.batches.next() {
            Some(Err(e)) => {
                self.done = true;
                Some(Err(external(e)))
            }
            Some(Ok(data)) => {
                let result = ArrowEngineData::try_from_engine_data(data)
                    .map_err(external)
                    .map(RecordBatch::from)
                    .and_then(|batch| {
                        let n = batch.num_rows() as u64;
                        let out =
                            drop_deleted(batch, self.row_offset, &self.deleted, &mut self.cursor)?;
                        self.row_offset += n;
                        Ok(out)
                    });
                if result.is_err() {
                    self.done = true;
                }
                Some(result)
            }
            None => {
                self.done = true;
                // A DV that points past the end of the file means it belongs to a different
                // version of the file, so the result cannot be trusted.
                match self.deleted.get(self.cursor) {
                    Some(d) => Some(Err(ArrowError::ExternalError(
                        format!(
                            "deletion vector references row {d} but the file has {} rows",
                            self.row_offset
                        )
                        .into(),
                    ))),
                    None => None,
                }
            }
        }
    }
}

impl RecordBatchReader for LiveRows {
    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }
}

/// Opens the file and returns a reader over its live rows.
pub fn open(
    file: &DataFile,
    physical_schema: SchemaRef,
    deleted: Option<&[u64]>,
) -> Result<impl RecordBatchReader + Send + 'static, String> {
    let arrow_schema: ArrowSchema = physical_schema
        .as_ref()
        .try_into_arrow()
        .map_err(|e: ArrowError| format!("schema to arrow: {e}"))?;
    let batches = shared_engine()
        .parquet_handler()
        .read_parquet_files(std::slice::from_ref(&file.meta), physical_schema, None)
        .map_err(|e| format!("open parquet: {e}"))?;
    Ok(LiveRows {
        batches,
        schema: Arc::new(arrow_schema),
        deleted: deleted.map(<[u64]>::to_vec).unwrap_or_default(),
        cursor: 0,
        row_offset: 0,
        done: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use delta_kernel::EngineData;

    fn live_rows(batches: Vec<(i32, i32)>, deleted: Vec<u64>) -> LiveRows {
        let data: Vec<delta_kernel::KernelResult<Box<dyn EngineData>>> = batches
            .into_iter()
            .map(|(start, n)| {
                let col = Int32Array::from((start..start + n).collect::<Vec<_>>());
                let b = RecordBatch::try_from_iter([("v", Arc::new(col) as _)]).unwrap();
                Ok(Box::new(ArrowEngineData::new(b)) as Box<dyn EngineData>)
            })
            .collect();
        LiveRows {
            batches: Box::new(data.into_iter()),
            schema: Arc::new(ArrowSchema::empty()),
            deleted,
            cursor: 0,
            row_offset: 0,
            done: false,
        }
    }

    #[test]
    fn dv_past_the_end_of_the_file_is_an_error() {
        let results: Vec<_> = live_rows(vec![(0, 4)], vec![1, 9]).collect();
        assert!(results[0].is_ok());
        let err = results[1].as_ref().unwrap_err().to_string();
        assert!(
            err.contains("references row 9 but the file has 4 rows"),
            "{err}"
        );
        assert_eq!(results.len(), 2, "stream ends after the error");
    }

    #[test]
    fn dv_inside_the_file_ends_cleanly() {
        let results: Vec<_> = live_rows(vec![(0, 4), (4, 4)], vec![3, 4]).collect();
        assert!(results.iter().all(|r| r.is_ok()));
        assert_eq!(results.len(), 2);
    }
}
