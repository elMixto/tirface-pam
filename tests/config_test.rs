use pam_tirface_pam::config::Config;
use std::fs;

#[test]
fn test_config() {
    let content = fs::read_to_string("config/tirface-pam.conf").unwrap();
    let config: Config = toml::from_str(&content).unwrap();
    println!("{:#?}", config);
}
