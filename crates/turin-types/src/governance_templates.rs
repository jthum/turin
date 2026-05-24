pub const OPEN: &str = include_str!("../../../templates/governance/open.toml");
pub const BALANCED: &str = include_str!("../../../templates/governance/balanced.toml");
pub const GOVERNED: &str = include_str!("../../../templates/governance/governed.toml");

pub fn by_name(name: &str) -> Option<&'static str> {
    match name {
        "open" => Some(OPEN),
        "balanced" => Some(BALANCED),
        "governed" => Some(GOVERNED),
        _ => None,
    }
}
