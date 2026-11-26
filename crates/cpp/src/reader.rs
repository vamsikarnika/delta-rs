//! Reads one Delta data file with its deletion vector applied.
//!
//! A split carries the file and the DV descriptor JSON that Spark hands the scan; no table snapshot
//! is needed (see `crates/core/examples/dv_descriptor_spike.rs`).
use std::sync::{Arc, OnceLock};

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use delta_kernel::Engine;
use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel_default_engine::DefaultEngineBuilder;
use futures::{Stream, StreamExt};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
// `ParquetObjectReader` is deprecated upstream in favour of implementing `AsyncFileReader`. It still
// does exactly what we need on our own runtime; swap it when the object-store work lands.
use parquet::arrow::ProjectionMask;
#[allow(deprecated)]
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use tokio::runtime::Runtime;

/// One runtime for every read: avoids per-split setup. Reads `block_on` it.
pub fn shared_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| Runtime::new().expect("failed to build tokio runtime"))
}

/// A data file location resolved to a store and a path inside it.
pub struct FileLocation {
    pub store: Arc<dyn ObjectStore>,
    pub path: StorePath,
}

/// Accepts an absolute local path or a `file://` URL. Other schemes are not supported yet.
pub fn resolve_file(file_path: &str) -> Result<FileLocation, String> {
    let local = if file_path.starts_with("file:") {
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
    let path = StorePath::from_filesystem_path(&local).map_err(|e| e.to_string())?;
    Ok(FileLocation {
        store: Arc::new(LocalFileSystem::new()),
        path,
    })
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

pub fn default_engine() -> impl Engine {
    DefaultEngineBuilder::new(Arc::new(LocalFileSystem::new())).build()
}

/// Turns sorted deleted positions into a selection that skips them over `num_rows` rows.
pub fn selection_from_deleted(deleted: &[u64], num_rows: u64) -> Result<RowSelection, String> {
    if let Some(last) = deleted.last()
        && *last >= num_rows
    {
        return Err(format!(
            "deletion vector references row {last} but the file has {num_rows} rows"
        ));
    }
    let mut selectors = Vec::new();
    let mut next_live = 0u64;
    let mut i = 0;
    while i < deleted.len() {
        let start = deleted[i];
        let mut end = start + 1;
        while i + 1 < deleted.len() && deleted[i + 1] == end {
            i += 1;
            end += 1;
        }
        if start > next_live {
            selectors.push(RowSelector::select((start - next_live) as usize));
        }
        selectors.push(RowSelector::skip((end - start) as usize));
        next_live = end;
        i += 1;
    }
    if next_live < num_rows {
        selectors.push(RowSelector::select((num_rows - next_live) as usize));
    }
    Ok(selectors.into())
}

/// Pulls batches from an async stream one at a time, on the shared runtime.
struct BlockingReader<S> {
    stream: S,
    schema: SchemaRef,
}

impl<S> Iterator for BlockingReader<S>
where
    S: Stream<Item = Result<RecordBatch, parquet::errors::ParquetError>> + Unpin,
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        shared_runtime()
            .block_on(self.stream.next())
            .map(|r| r.map_err(|e| ArrowError::ExternalError(Box::new(e))))
    }
}

impl<S> RecordBatchReader for BlockingReader<S>
where
    S: Stream<Item = Result<RecordBatch, parquet::errors::ParquetError>> + Unpin,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Opens the file and returns a reader over its live rows.
pub fn open(
    location: &FileLocation,
    file_size: i64,
    deleted: Option<&[u64]>,
    projected_columns: &[String],
    batch_size: i64,
) -> Result<impl RecordBatchReader + Send + 'static, String> {
    #[allow(deprecated)]
    shared_runtime().block_on(async {
        let mut reader = ParquetObjectReader::new(location.store.clone(), location.path.clone());
        if file_size > 0 {
            reader = reader.with_file_size(file_size as u64);
        }
        let mut builder = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(|e| format!("open parquet: {e}"))?;
        if batch_size > 0 {
            builder = builder.with_batch_size(batch_size as usize);
        }
        if !projected_columns.is_empty() {
            // `ProjectionMask::columns` silently drops names it cannot find; a missing column would
            // then shift every column after it, so check up front.
            let file_schema = builder.schema().clone();
            for c in projected_columns {
                if file_schema.field_with_name(c).is_err() {
                    return Err(format!("column {c} not found in {}", location.path));
                }
            }
            let mask = ProjectionMask::columns(
                builder.parquet_schema(),
                projected_columns.iter().map(String::as_str),
            );
            builder = builder.with_projection(mask);
        }
        if let Some(deleted) = deleted {
            let num_rows = builder.metadata().file_metadata().num_rows() as u64;
            builder = builder.with_row_selection(selection_from_deleted(deleted, num_rows)?);
        }
        let schema = builder.schema().clone();
        let stream = builder.build().map_err(|e| format!("build reader: {e}"))?;
        Ok(BlockingReader { stream, schema })
    })
}
