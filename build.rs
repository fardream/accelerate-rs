use std::{env, path::PathBuf, process::Command, str::from_utf8};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rerun-if-changed=wrapper.h");
    let sdkpath = from_utf8(
        &Command::new("xcrun")
            .arg("--show-sdk-path")
            .output()?
            .stdout,
    )?
    .trim_end()
    .to_owned();
    let bindings = bindgen::Builder::default()
        .clang_args(&["-isysroot", &sdkpath])
        .clang_arg("-DACCELERATE_NEW_LAPACK")
        .opaque_type("^HFS.*$")
        .opaque_type("FndrOpaqueInfo")
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))?;

    Ok(())
}
