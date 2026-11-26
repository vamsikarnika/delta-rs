//! Arrow C stream plumbing shared by the bridge.
use crate::ffi;
use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;

/// Wrap a [`RecordBatchReader`] in a heap-allocated `FFI_ArrowArrayStream` and hand it out as an
/// opaque pointer. The reader is pulled lazily by the C++ side; nothing is materialised up front.
/// Reclaim it only with [`free_arrow_stream`].
pub fn create_raw_pointer_for_record_batch_reader<R>(reader: R) -> *mut ffi::ArrowArrayStream
where
    R: RecordBatchReader + Send + 'static,
{
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    Box::into_raw(Box::new(stream)) as *mut ffi::ArrowArrayStream
}

/// Free a stream produced by [`create_raw_pointer_for_record_batch_reader`]. Null is a no-op.
///
/// # Safety
/// `ptr` must come from [`create_raw_pointer_for_record_batch_reader`] and be freed exactly once.
pub unsafe fn free_arrow_stream(ptr: *mut ffi::ArrowArrayStream) {
    if !ptr.is_null() {
        // Same type `Box::into_raw` produced above; dropping runs the Arrow release callback.
        unsafe { drop(Box::from_raw(ptr as *mut FFI_ArrowArrayStream)) };
    }
}
