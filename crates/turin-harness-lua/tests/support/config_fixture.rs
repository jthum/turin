macro_rules! config_fixture {
    ($($field:ident $(: $value:expr)?),* $(,)?) => {{
        let mut config = TurinConfig::default();
        $(config_fixture!(@set config, $field $(: $value)?);)*
        config
    }};
    (@set $config:ident, $field:ident: $value:expr) => { $config.$field = $value };
    (@set $config:ident, $field:ident) => { $config.$field = $field };
}
