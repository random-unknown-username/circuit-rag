use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

const DEFAULT_CONFIG: &str = "\
# CIRCUIT Mem config
format_version = 1
";

fn main() -> ExitCode {
    let mut args = env::args().skip(1);

    let Some(command) = args.next() else {
        print_help();
        return ExitCode::SUCCESS;
    }

    if args.next().is_some() {
        eprintln!("Error: Too many args");
        print_help();
        return ExitCode::FAILURE;
    }

    if command == "init" {
        match initialize(Path::new(".circuit")) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Error initializing circuit dir: {error}");
                ExitCode::FAILURE
            }
        }
    ) else if command == "--help" || command == "-h" {
        print_help();
        ExitCode::SUCCESS
    } else {
        eprintln!("Error: Unknown command '{command}'");
        print_help();
        ExitCode::FAILURE
    }
}

fn print_help() {
    println!("CIRCUIT CLI: memory for your terminal and agents");
    println!();
    println!("Commands:");
    println!("  init      Initialize a new circuit directory");
    println!("  --help    Show this help message");
}

fn initialize(root: &Path) -> io::Result<()> {
      let created_directory = ensure_directory(root)?;
      let config_path = root.join("config.toml");

      let mut config_file = match OpenOptions::new()
          .write(true)
          .create_new(true)
          .open(&config_path)
      {
          Ok(file) => file,

          Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
              let metadata = fs::symlink_metadata(&config_path)?;

              if !metadata.file_type().is_file() {
                  return Err(io::Error::new(
                      io::ErrorKind::InvalidInput,
                      format!(
                          "{} exists but is not a regular file",
                          config_path.display()
                      ),
                  ));
              }

              println!(
                  "Existing configuration preserved: {}",
                  config_path.display()
              );
              println!("Configuration validation is WIP right now.");

              return Ok(());
          }

          Err(error) => {
              if created_directory {
                  cleanup_empty_directory(root);
              }

              return Err(error);
          }
      };

      if let Err(error) = config_file.write_all(DEFAULT_CONFIG.as_bytes()) {
        drop(config_file);

        if let Err(cleanup_error) = fs::remove_file(&config_path) {
            eprintln!(
                "warn: couldnt remove incomplete config {}: {}" ,
                config_path.display(),
                cleanup_error
            );
        }
        if created_directory {
            cleanup_empty_directory(root);
        }
        return Err(error);
      } 

      println!("Created config  {}", config_path.display());
      println!("Circuit memory init at  {}", root.display());

      Ok(())
}

fn ensure_directory(path: &Path) -> io::Result<bool> {
      match fs::symlink_metadata(path) {
          Ok(metadata) => {
              if metadata.file_type().is_symlink() {
                  return Err(io::Error::new(
                      io::ErrorKind::InvalidInput,
                      format!(
                          "{} is a symlink; choose a regular directory",
                          path.display()
                      ),
                  ));
              }
              if !metadata.is_dir() {
                  return Err(io::Error::new(
                      io::ErrorKind::AlreadyExists,
                      format!(
                          "{} exists but is not a directory",
                          path.display()
                      ),
                  ));
              }

              Ok(false)
          }

          Err(error) if error.kind() == io::ErrorKind::NotFound => {
              fs::create_dir(path)?;
              Ok(true)
          }

          Err(error) => Err(error),
      }
  }

fn cleanup_empty_directory(path: &Path) {
      if let Err(error) = fs::remove_dir(path) {
          eprintln!(
              "warning: could not remove empty directory {}: {}",
              path.display(),
              error
          );
      }
  }
