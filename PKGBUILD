# Maintainer: mixto
pkgname=tirface-pam
pkgver=1.0.0
pkgrel=1
pkgdesc="A fast and pythonless face authentication module for the IR camera on some Thinkpads, that runs on CPU and NPUs with OpenVINO with a simple installation."
arch=('x86_64')
url="https://github.com/elMixto/tirface-pam"
license=('MIT')
depends=('v4l-utils' 'pam' 'openvino' 'sqlite')
makedepends=('cargo')
install=install.install

prepare() {
  export CARGO_HOME="${srcdir}/cargo-home"
  cargo fetch --manifest-path "${startdir}/Cargo.toml" --target "$CARCH-unknown-linux-gnu"
}

build() {
  export CARGO_HOME="${srcdir}/cargo-home"
  cargo build --manifest-path "${startdir}/Cargo.toml" --frozen --release --all-targets
}

package() {
  install -d "${pkgdir}/usr/bin"
  install -d "${pkgdir}/usr/lib/security"
  install -d "${pkgdir}/usr/lib/systemd/system"
  install -d "${pkgdir}/var/lib/tirface-pam/models"
  install -d "${pkgdir}/etc/tirface-pam"
  install -d "${pkgdir}/usr/share/dbus-1/system.d"

  install -m755 "${startdir}/target/release/tirface-pam-cli" "${pkgdir}/usr/bin/tirface-pam-cli"
  install -m755 "${startdir}/target/release/tirface-pam-daemon" "${pkgdir}/usr/bin/tirface-pam-daemon"
  install -m755 "${startdir}/target/release/libpam_tirface_pam.so" "${pkgdir}/usr/lib/security/pam_tirface_pam.so"

  install -m644 "${startdir}/src/models/rustface/seeta_fd_frontal_v1.0.bin" "${pkgdir}/var/lib/tirface-pam/models/"
  install -m644 "${startdir}/src/models/mobilefacenet/mobilefacenet.onnx" "${pkgdir}/var/lib/tirface-pam/models/"
  install -m644 "${startdir}/src/models/arcface/arcface.onnx" "${pkgdir}/var/lib/tirface-pam/models/"

  install -m644 "${startdir}/config/tirface-pam.service" "${pkgdir}/usr/lib/systemd/system/tirface-pam.service"
  install -m644 "${startdir}/config/tirface-pam.conf" "${pkgdir}/etc/tirface-pam/config.toml"
  install -m644 "${startdir}/config/org.freedesktop.TirfacePam1.conf" "${pkgdir}/usr/share/dbus-1/system.d/org.freedesktop.TirfacePam1.conf"
}
