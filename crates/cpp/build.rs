fn main() {
    // The include prefix must carry the crate name: hudi, iceberg and delta each generate a
    // `lib.rs.h`, and Velox puts all of their include dirs on one path.
    cxx_build::CFG.include_prefix = "delta";
    cxx_build::bridge("src/lib.rs")
        .include("include")
        .include("include/arrow/c")
        .flag_if_supported("-std=c++17")
        .compile("delta");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/util.rs");
    println!("cargo:rerun-if-changed=include/arrow/c/abi.h");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:root={manifest}");
}
