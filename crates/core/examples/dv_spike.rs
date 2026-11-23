//! Spike (ENG-52692): read a Delta table file by file with deletion vectors applied, straight
//! from delta-kernel, no DataFusion. This is the per-split read a native (Velox) reader needs:
//! given one data file and its DV descriptor, return the live rows only.
//!
//! Usage: cargo run -p deltalake-core --example dv_spike -- <table_dir>
use std::sync::Arc;

use arrow::array::BooleanArray;
use arrow::compute::filter_record_batch;
use arrow::record_batch::RecordBatch;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::scan::state::{ScanFile, transform_to_logical};
use delta_kernel::{Engine, FileMeta, Snapshot};
use delta_kernel_default_engine::DefaultEngineBuilder;
use object_store::local::LocalFileSystem;

fn collect(files: &mut Vec<ScanFile>, f: ScanFile) {
    files.push(f);
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::fs::canonicalize(std::env::args().nth(1).expect("table dir"))?;
    let root = url::Url::from_directory_path(&dir).unwrap();
    let engine = DefaultEngineBuilder::new(Arc::new(LocalFileSystem::new())).build();

    let snapshot = Snapshot::builder_for(root.clone()).build(&engine)?;
    let scan = snapshot.clone().scan_builder().build()?;
    let logical_schema = scan.logical_schema().clone();
    let physical_schema = scan.physical_schema().clone();

    let mut files = vec![];
    for sm in scan.scan_metadata(&engine)? {
        files = sm?.visit_scan_files(files, collect)?;
    }
    println!("version={} files={}", snapshot.version(), files.len());

    let mut total_live = 0usize;
    for f in &files {
        // 1. Resolve the DV descriptor to sorted, deleted row positions (None = no DV).
        let deleted = f.dv_info.get_row_indexes(&engine, &root)?;
        let cardinality = f.dv_info.cardinality()?;
        if let (Some(d), Some(c)) = (&deleted, cardinality) {
            assert_eq!(
                d.len() as u64,
                c,
                "decoded DV disagrees with descriptor cardinality"
            );
            assert!(
                d.windows(2).all(|w| w[0] < w[1]),
                "DV positions not sorted/unique"
            );
        }

        // 2. Read the physical Parquet file, one file per call (what a split would do).
        let meta = FileMeta {
            location: root.join(&f.path)?,
            last_modified: f.modification_time,
            size: f.size as u64,
        };
        let batches =
            engine
                .parquet_handler()
                .read_parquet_files(&[meta], physical_schema.clone(), None)?;

        // 3. Physical -> logical (column mapping, partition columns), then drop deleted rows.
        let mut row_offset = 0u64;
        let mut deleted_iter = deleted.as_deref().unwrap_or(&[]).iter().peekable();
        let mut live = 0usize;
        for data in batches {
            let data = transform_to_logical(
                &engine,
                data?,
                &physical_schema,
                &logical_schema,
                f.transform.clone(),
            )?;
            let batch: RecordBatch = ArrowEngineData::try_from_engine_data(data)?.into();
            let n = batch.num_rows() as u64;
            let keep: BooleanArray = (0..n)
                .map(|i| {
                    let pos = row_offset + i;
                    while deleted_iter.next_if(|d| **d < pos).is_some() {}
                    Some(deleted_iter.peek().map_or(true, |d| **d != pos))
                })
                .collect();
            live += filter_record_batch(&batch, &keep)?.num_rows();
            row_offset += n;
        }
        if let Some(d) = &deleted {
            assert!(
                d.last().is_none_or(|last| *last < row_offset),
                "DV references rows past the end of the file"
            );
        }
        println!(
            "{}: physical_rows={row_offset} deleted={} live_rows={live}",
            f.path,
            deleted.as_ref().map_or(0, |d| d.len())
        );
        assert_eq!(
            live as u64,
            row_offset - deleted.as_ref().map_or(0, |d| d.len() as u64)
        );
        total_live += live;
    }
    println!("total_live_rows={total_live}");
    Ok(())
}
