use std::thread::sleep;
use std::time::Duration;
use pam_tirface_pam::camera::{Camera, CameraType, CameraManager};
use pam_tirface_pam::config::CameraPath;

#[test]
fn test_ir_emitter_3_seconds() {
    println!("Starting IR emitter test for 3 seconds...");
    
    // 1. Instanciar la cámara IR de Chicony en /dev/video2
    let camera = Camera::new(
        &CameraPath::Custom("/dev/video2".to_string()),
        CameraType::Ir
    ).expect("No se pudo instanciar la cámara");

    println!("Starting camera manager and video stream (Warmup)...");
    // 2. Start the CameraManager (this starts the video stream and activates the IR emitter natively)
    let mut manager = CameraManager::start(&camera).expect("Failed to start CameraManager");

    println!("Stream active! The IR emitter should be physically on now.");
    println!("Keeping on for 3 seconds...");
    
    // Esperar 3 segundos con el emisor encendido
    sleep(Duration::from_secs(3));

    println!("Stopping camera manager (this will deactivate the IR emitter)...");
    // 3. Stop the camera (this calls deactivate() on the IR emitter and closes the stream)
    manager.stop();

    println!("✅ Test completado con éxito.");
}
