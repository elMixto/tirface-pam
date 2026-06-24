use thiserror::Error;

#[derive(Error, Debug)]
pub enum FacePamError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Camera error: {0}")]
    Camera(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("Authentication timeout")]
    Timeout,

    #[error("Authentication rejected")]
    Rejected,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}
