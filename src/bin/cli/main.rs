mod detect;
mod enroll;
mod inference;
mod utils;

use clap::{Parser, Subcommand};
use pam_tirface_pam::config::Config;

#[derive(Parser)]
#[command(author, version, about = "Tirface PAM - Face Authentication CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Starts the enrollment mode to register a new face
    Enroll {
        /// Name of the user to register. Defaults to the current user (or sudo user).
        username: Option<String>,
        /// Runs the registration process without a graphical interface (debug/headless mode)
        #[arg(long)]
        headless: bool,
    },
    /// Starts the inference mode to test the model in real time
    Test,
    /// Runs a hardware self-diagnosis and benchmark of the configured AI model
    Detect,
}

fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    let config = Config::load();

    match cli.command {
        Commands::Enroll { username, headless } => enroll::run_enroll(&config, username, headless),
        Commands::Test => inference::run_test(&config),
        Commands::Detect => detect::run_detect(&config),
    }
}
