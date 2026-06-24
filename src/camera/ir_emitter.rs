use crate::error::FacePamError;
use log::debug;
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

const UVCIOC_CTRL_QUERY: libc::c_ulong = 0xC010_7521;
const UVC_SET_CUR: u8 = 0x01;

#[repr(C)]
struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    _pad0: u8,
    size: u16,
    _pad1: u16,
    data: *mut u8,
}

pub struct IRMetadata {
    pub vendor_id: u16,
    pub product_id: u16,
    pub name: &'static str,
    pub unit: u8,
    pub selector: u8,
    pub control_bytes: &'static [u8],
}

// No camera metadata configured by default for now.
static IR_METADATA_DB: &[IRMetadata] = &[
    IRMetadata {
        vendor_id: 0x04f2,
        product_id: 0xb805,
        name: "Chicony Integrated IR Camera",
        unit: 0x0e,
        selector: 0x0e,
        control_bytes: &[0x02, 0x19],
    },
];

pub struct IrEmitter {
    device_path: String,
    metadata: &'static IRMetadata,
}

impl IrEmitter {
    pub fn for_device(device_path: &str) -> Option<Self> {
        let (vid, pid) = get_usb_ids(device_path)?;
        let metadata = IR_METADATA_DB.iter().find(|m| m.vendor_id == vid && m.product_id == pid)?;
        Some(Self {
            device_path: device_path.to_string(),
            metadata,
        })
    }

    pub fn activate(&self) -> Result<(), FacePamError> {
        debug!("Activating native IR emitter for camera: {}", self.metadata.name);
        let mut payload = self.metadata.control_bytes.to_vec();
        self.send_uvc_control(&mut payload)
    }

    pub fn deactivate(&self) -> Result<(), FacePamError> {
        debug!("Deactivating native IR emitter for camera: {}", self.metadata.name);
        let mut payload = vec![0u8; self.metadata.control_bytes.len()];
        self.send_uvc_control(&mut payload)
    }

    fn send_uvc_control(&self, payload: &mut [u8]) -> Result<(), FacePamError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.device_path)
            .map_err(|e| FacePamError::Camera(format!("Failed to open device for UVC control: {}", e)))?;

        let mut query = UvcXuControlQuery {
            unit: self.metadata.unit,
            selector: self.metadata.selector,
            query: UVC_SET_CUR,
            _pad0: 0,
            size: payload.len() as u16,
            _pad1: 0,
            data: payload.as_mut_ptr(),
        };

        let ret = unsafe {
            libc::ioctl(
                file.as_raw_fd(),
                UVCIOC_CTRL_QUERY,
                &mut query as *mut UvcXuControlQuery,
            )
        };

        if ret < 0 {
            Err(FacePamError::Camera(format!("UVC ioctl failed: {}", std::io::Error::last_os_error())))
        } else {
            Ok(())
        }
    }
}

pub fn get_usb_ids(device_path: &str) -> Option<(u16, u16)> {
    let dev_name = std::path::Path::new(device_path).file_name()?.to_str()?;
    let device_link = format!("/sys/class/video4linux/{}/device", dev_name);
    let interface_dir = std::fs::canonicalize(&device_link).ok()?;
    let usb_device_dir = interface_dir.parent()?;

    let vid_str = std::fs::read_to_string(usb_device_dir.join("idVendor")).ok()?;
    let pid_str = std::fs::read_to_string(usb_device_dir.join("idProduct")).ok()?;

    let vid = u16::from_str_radix(vid_str.trim(), 16).ok()?;
    let pid = u16::from_str_radix(pid_str.trim(), 16).ok()?;
    Some((vid, pid))
}
