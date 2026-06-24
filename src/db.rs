use crate::config::Config;
use crate::crypto::FaceCrypto;
use crate::error::FacePamError;
use log::{debug, warn};
use std::path::Path;
use rusqlite::{params, Connection};
use uuid::Uuid;

pub fn get_db_connection() -> Result<Connection, FacePamError> {
    let db_path = crate::paths::SYSTEM_DB_PATH;
    if let Some(parent) = Path::new(db_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let conn = Connection::open(db_path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS face_signatures (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            label TEXT NOT NULL,
            model_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_model ON face_signatures(username, model_name)",
        [],
    )?;
    
    Ok(conn)
}

pub fn save_embedding(
    username: &str,
    label: &str,
    model_name: &str,
    embedding: &[f32],
    crypto: &FaceCrypto,
) -> Result<(), FacePamError> {
    let conn = get_db_connection()?;
    
    let encrypted_blob = crypto.encrypt_vector(embedding)
        .map_err(|e| FacePamError::Crypto(format!("{}", e)))?;

    let id = Uuid::new_v4().to_string();

    conn.execute(
        "INSERT INTO face_signatures (id, username, label, model_name, embedding)
         VALUES (?, ?, ?, ?, ?)",
        params![id, username, label, model_name, encrypted_blob],
    )?;

    Ok(())
}

pub fn remove_embedding(username: &str, id_or_label: &str) -> Result<bool, FacePamError> {
    let conn = get_db_connection()?;
    let rows_affected = conn.execute(
        "DELETE FROM face_signatures WHERE username = ? AND (id = ? OR label = ?)",
        params![username, id_or_label, id_or_label],
    )?;
    Ok(rows_affected > 0)
}

#[derive(serde::Serialize)]
pub struct FaceSignatureRecord {
    pub id: String,
    pub username: String,
    pub label: String,
    pub model_name: String,
    pub created_at: String,
}

pub fn list_user_embeddings(username: &str) -> Result<Vec<FaceSignatureRecord>, FacePamError> {
    let conn = get_db_connection()?;
    let mut stmt = conn.prepare(
        "SELECT id, username, label, model_name, datetime(created_at, 'localtime') FROM face_signatures WHERE username = ?"
    )?;
    
    let rows = stmt.query_map(params![username], |row| {
        Ok(FaceSignatureRecord {
            id: row.get(0)?,
            username: row.get(1)?,
            label: row.get(2)?,
            model_name: row.get(3)?,
            created_at: row.get(4)?,
        })
    })?;

    let mut result = Vec::new();
    for row in rows {
        if let Ok(record) = row {
            result.push(record);
        }
    }
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct EmbeddingDatabase {
    pub records: Vec<(String, Vec<f32>)>,
}

impl EmbeddingDatabase {
    pub fn load_all(crypto: &FaceCrypto, config: &Config) -> Self {
        let mut records = Vec::new();
        let conn = match get_db_connection() {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to open DB in load_all: {}", e);
                return Self { records };
            }
        };

        let active_model = config.models.model_name();

        let mut stmt = match conn.prepare(
            "SELECT username, embedding FROM face_signatures WHERE model_name = ?"
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to prepare select in load_all: {}", e);
                return Self { records };
            }
        };

        let rows = stmt.query_map(params![active_model], |row| {
            let username: String = row.get(0)?;
            let embedding_bytes: Vec<u8> = row.get(1)?;
            Ok((username, embedding_bytes))
        });

        if let Ok(rows) = rows {
            for row in rows.flatten() {
                let (username, bytes) = row;
                match crypto.decrypt_vector(&bytes) {
                    Ok(emb) => records.push((username, emb)),
                    Err(e) => warn!("Failed to decrypt embedding in DB for user {}: {}", username, e),
                }
            }
        }

        Self { records }
    }

    pub fn load_for_user(username: &str, crypto: &FaceCrypto, config: &Config) -> Vec<Vec<f32>> {
        let mut user_embeddings = Vec::new();
        let conn = match get_db_connection() {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to open DB in load_for_user: {}", e);
                return user_embeddings;
            }
        };

        let active_model = config.models.model_name();

        let mut stmt = match conn.prepare(
            "SELECT embedding FROM face_signatures WHERE username = ? AND model_name = ?"
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to prepare select in load_for_user: {}", e);
                return user_embeddings;
            }
        };

        let rows = stmt.query_map(params![username, active_model], |row| {
            let embedding_bytes: Vec<u8> = row.get(0)?;
            Ok(embedding_bytes)
        });

        if let Ok(rows) = rows {
            for row in rows.flatten() {
                let bytes = row;
                match crypto.decrypt_vector(&bytes) {
                    Ok(emb) => user_embeddings.push(emb),
                    Err(e) => warn!("Failed to decrypt embedding in DB for user {}: {}", username, e),
                }
            }
        }

        debug!("Loaded {} embeddings from SQLite for user {}", user_embeddings.len(), username);
        user_embeddings
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
