use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid key length (expected 32, got {0})")]
    InvalidKeyLength(usize),
    #[error("Encryption failed: {0}")]
    Encryption(String),
    #[error("Encrypted data too short (< 12 bytes)")]
    DataTooShort,
    #[error("Decryption failed: {0}")]
    Decryption(String),
    #[error("Decrypted data is not a multiple of 4 (f32 size), got {0} bytes")]
    InvalidDecryptedSize(usize),
}

#[derive(Clone, Debug)]
pub struct FaceCrypto {
    key: [u8; 32],
}

impl FaceCrypto {
    pub fn load_or_create() -> Result<Self, CryptoError> {
        // Opción A: Systemd Credentials (prioritario)
        if let Ok(creds_dir) = std::env::var("CREDENTIALS_DIRECTORY") {
            let cred_path = Path::new(&creds_dir).join("master_key");
            if cred_path.exists() {
                let key_bytes = fs::read(&cred_path)?;
                if key_bytes.len() == 32 {
                    let mut key_arr = [0u8; 32];
                    key_arr.copy_from_slice(&key_bytes);
                    log::info!("Master key loaded from Systemd Credentials");
                    return Ok(Self { key: key_arr });
                } else {
                    log::warn!(
                        "Systemd credential master_key has invalid length (got {}), falling back to local file",
                        key_bytes.len()
                    );
                }
            }
        }

        // Opción C: Archivo Local (fallback o por defecto si Systemd no inyectó nada)
        let key_file = crate::paths::SYSTEM_KEY_FILE;

        if Path::new(key_file).exists() {
            let key_bytes = fs::read(key_file)?;
            if key_bytes.len() != 32 {
                return Err(CryptoError::InvalidKeyLength(key_bytes.len()));
            }
            let mut key_arr = [0u8; 32];
            key_arr.copy_from_slice(&key_bytes);
            Ok(Self { key: key_arr })
        } else {
            let mut key_arr = [0u8; 32];
            rand::fill(&mut key_arr);
            
            if let Some(parent) = Path::new(key_file).parent() {
                if parent.to_str() != Some("") {
                    fs::create_dir_all(parent)?;
                }
            }
            
            fs::write(key_file, key_arr)?;
            
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let metadata = fs::metadata(key_file)?;
                let mut perms = metadata.permissions();
                perms.set_mode(0o400); // Solo owner puede leer
                fs::set_permissions(key_file, perms)?;
            }
            Ok(Self { key: key_arr })
        }
    }

    pub fn encrypt_vector(&self, data: &[f32]) -> Result<Vec<u8>, CryptoError> {
        let cipher = Aes256Gcm::new((&self.key).into());
        let mut nonce_bytes = [0u8; 12];
        rand::fill(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let byte_data: &[u8] = bytemuck::cast_slice(data);

        let ciphertext = cipher
            .encrypt(nonce, byte_data)
            .map_err(|e| CryptoError::Encryption(format!("{:?}", e)))?;

        let mut final_data = nonce_bytes.to_vec();
        final_data.extend(ciphertext);
        Ok(final_data)
    }

    pub fn decrypt_vector(&self, encrypted_data: &[u8]) -> Result<Vec<f32>, CryptoError> {
        if encrypted_data.len() < 12 {
            return Err(CryptoError::DataTooShort);
        }

        let cipher = Aes256Gcm::new((&self.key).into());
        let nonce = Nonce::from_slice(&encrypted_data[0..12]);
        let ciphertext = &encrypted_data[12..];

        let plaintext_bytes = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| CryptoError::Decryption(format!("{:?}", e)))?;

        if plaintext_bytes.len() % 4 != 0 {
            return Err(CryptoError::InvalidDecryptedSize(plaintext_bytes.len()));
        }

        let float_data: &[f32] = bytemuck::cast_slice(&plaintext_bytes);
        Ok(float_data.to_vec())
    }
}