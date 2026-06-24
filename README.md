# About this thing

I have a ThinkPad T14 Gen 5 and i wanted to use the Windows Hello capable IR camera. I made a simple pam module that runs a face authentication pipeline with ArcFace ResNet100 on the integrated NPU of the Intel 155H via OpenVino.

* Installing python based projects from the AUR gives me the ick, so i wont use howdy ;-;.

* Inference takes about 22ms per frame so with a nice enroll process, it should authenticate in under a second. (After the latency of initializing the camera)

* It falls back to CPU in case the laptop does not have a NPU or openvino, for wich i recomend using facenet, which takes me to the models i've bundled.

-   `MobileFaceNet` (13MB)
-   `ArcFace ResNet100` (249MB)


* This specific camera has a stroboscopic IR emitter (I dont kwon if this is common for all hardware) and the software filters frames based on that when processing the frames.

The project consists in tree binaries:

1.  The daemon: Handles the camera, data encryption and decryption, and inference. 

2.  pam_tirface_pam.so:  The pam module that triggers the daemon authentication logic.

3.  cli: For enrolling, and testing the process. It runs fully on the terminal.


## Performance
I use RustfaceDetector to detect the face rounding box for the enrollment process. That runs on the CPU, and i havent really measured how fast it runs. A breakdown of how fast the other models run it's shown below.

| Model             | Framework    | Hardware |  Inference   |
| :---              | :---         | :---     | :---         |
| MobileFaceNet     | ONNX Runtime | CPU      | `~ 51.26 ms` |
| MobileFaceNet     | OpenVINO     | NPU      | `~ 21.38 ms` |
| ArcFace ResNet100 | ONNX Runtime | CPU      | `~ 321.41 ms`|
| ArcFace ResNet100 | OpenVINO     | NPU      | `~ 30.65 ms` |


## Clone and Install

This repository uses **Git LFS** to store the heavy ONNX model files.

1. Install `git-lfs` and clone the repository:
```bash
git clone https://github.com/elMixto/tirface-pam.git
cd tirface-pam
```

*Note: If you already cloned the repository without `git-lfs` installed, the ONNX files will be 3-line pointer files. You can fetch the actual binary models by installing `git-lfs` and running:*
```bash
git lfs pull
```

Use the included PKGBUILD:
```bash
makepkg -si
```

Then, enable and start the daemon service:
```bash
sudo systemctl enable --now tirface-pam.service
```

### Enable for Sudo
To authenticate `sudo` with your face, edit `/etc/pam.d/sudo` and add the module at the top as `sufficient`:

```pam
#%PAM-1.0
auth      sufficient pam_tirface_pam.so
auth      include    system-auth
...
```

*Using `sufficient` ensures that if facial recognition fails, is canceled (Ctrl+C), or the daemon is stopped, PAM will fall back to password authentication.*

## Enrollment Guide

To register your face for authentication, simply follow these steps:

1. **Run the enroll command** (specify a username if enrolling someone else, otherwise it defaults to the current user):
   ```bash
   sudo tirface-pam-cli enroll [username]
   ```

2. **Follow the on-screen poses**:
   * Center your face in the camera box until it turns green.
   * Press `[Space]` to start the capture.
   * Look in the requested directions when prompted (Frontal, Left, Right, Up, Down).

3. **Test authentication**:
   ```bash
   sudo tirface-pam-cli test
   ```