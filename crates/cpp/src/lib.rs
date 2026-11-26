//! cxx bridge that reads one Delta data file, deletion vector applied, as an Arrow C stream.

mod reader;
mod util;

// The cxx::bridge expansion uses fully-qualified paths that trip `unused_qualifications`.
#[allow(unused_qualifications)]
#[cxx::bridge]
pub mod ffi {
    /// One split: a data file plus the DV descriptor Spark gave the scan.
    struct FfiDeltaScanContext {
        /// Table directory as a URL (`file:///...`). Needed only for table-relative (`u`) DVs; "" if
        /// unknown. Never guessed from the data file path.
        table_root: String,
        /// Data file: absolute local path or `file://` URL. Local only for now.
        file_path: String,
        /// File size in bytes; <= 0 if unknown (one extra request to find the footer).
        file_size: i64,
        /// Deletion vector descriptor as JSON (the decoded `row_index_filter_id_encoded`); ""
        /// means the file has no DV.
        dv_descriptor_json: String,
        /// Physical column names to read; empty reads every column. A name missing from the file
        /// is an error, not a silent drop.
        projected_columns: Vec<String>,
        /// Rows per batch; <= 0 uses the parquet default.
        batch_size: i64,
        /// Reserved for object-store options.
        props_keys: Vec<String>,
        props_values: Vec<String>,
    }

    unsafe extern "C++" {
        include!("arrow/c/abi.h");
        type ArrowArrayStream;
    }

    extern "Rust" {
        type DeltaFileReader;

        /// Validates the context and resolves the DV. Fails on a bad descriptor, a DV whose count
        /// disagrees with its descriptor, or an unsupported file location.
        fn new_delta_file_reader(ctx: FfiDeltaScanContext) -> Result<Box<DeltaFileReader>>;

        /// Opens the file and returns a stream of its live rows. The caller owns the pointer and
        /// must pass it to `delta_free_arrow_stream` exactly once.
        fn read(self: &DeltaFileReader) -> Result<*mut ArrowArrayStream>;

        /// Frees a stream returned by `read`. Null is a no-op.
        unsafe fn delta_free_arrow_stream(ptr: *mut ArrowArrayStream);
    }
}

/// A validated split: location, resolved deleted rows and read options.
pub struct DeltaFileReader {
    location: reader::FileLocation,
    file_size: i64,
    deleted: Option<Vec<u64>>,
    projected_columns: Vec<String>,
    batch_size: i64,
}

pub fn new_delta_file_reader(
    ctx: ffi::FfiDeltaScanContext,
) -> Result<Box<DeltaFileReader>, String> {
    guard_no_runtime("new_delta_file_reader")?;
    let location = reader::resolve_file(&ctx.file_path)?;
    let engine = reader::default_engine();
    let deleted = reader::resolve_deleted_rows(&ctx.dv_descriptor_json, &ctx.table_root, &engine)?;
    Ok(Box::new(DeltaFileReader {
        location,
        file_size: ctx.file_size,
        deleted,
        projected_columns: ctx.projected_columns,
        batch_size: ctx.batch_size,
    }))
}

impl DeltaFileReader {
    pub fn read(&self) -> Result<*mut ffi::ArrowArrayStream, String> {
        guard_no_runtime("read")?;
        let reader = reader::open(
            &self.location,
            self.file_size,
            self.deleted.as_deref(),
            &self.projected_columns,
            self.batch_size,
        )?;
        Ok(util::create_raw_pointer_for_record_batch_reader(reader))
    }
}

/// # Safety
/// `ptr` must come from `read` and be freed exactly once, or be null.
pub unsafe fn delta_free_arrow_stream(ptr: *mut ffi::ArrowArrayStream) {
    unsafe { util::free_arrow_stream(ptr) };
}

/// `block_on` panics inside a tokio runtime; return an error instead of aborting the process.
fn guard_no_runtime(what: &str) -> Result<(), String> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(format!(
            "delta-cpp {what}() called from within a tokio runtime; call from a plain thread"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
