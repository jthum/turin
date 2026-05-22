mod auth;
mod manifest;
mod messages;
mod routing;
mod settings;

pub use auth::{
    ChannelAuthFlow, ChannelAuthFlowDisplay, ChannelAuthFlowKind, ChannelAuthFlowPollRequest,
    ChannelAuthFlowPollResponse, ChannelAuthFlowResolvedValue, ChannelAuthFlowStartRequest,
    ChannelAuthFlowStartResponse,
};
pub use manifest::{
    ChannelAdapterManifest, ChannelCapabilities, ChannelConfigField, ChannelConfigFieldOption,
    ChannelConfigTarget, ChannelConfigTargetKind, ChannelEnumSetting, ChannelFieldVisibilityRule,
    ChannelIdentitySelectors, ChannelInstallManifest, ChannelRuntimeCapabilities,
    ChannelRuntimeManifest, ChannelSecretRequirement, ChannelSetupManifest, ChannelValidationCheck,
    channel_enum_setting, channel_setting_target, channel_setting_target_opt, config_field_option,
    config_field_options, max_inbound_text_chars_field, validate_adapter_manifest,
};
pub use messages::{
    ChannelAttachment, ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope,
    ChannelUser, InboundEvent, MessageBlock, OutboundMessage, bound_inbound_text,
    render_plain_text_blocks, split_text_lines_to_char_limit,
};
pub use routing::{ConversationBinding, RoutingDecision, decide_routing};
pub use settings::{
    ChannelConfigError, optional_bool_setting, optional_non_empty_setting,
    optional_session_scope_setting, positive_usize_setting, required_non_empty_setting,
    session_scope_setting, string_enum_setting, u64_setting_with_min,
};

pub const CHANNEL_ADAPTER_PROTOCOL_VERSION: u32 = 2;
pub const DEFAULT_MAX_INBOUND_TEXT_CHARS: usize = 16_000;

#[cfg(test)]
mod tests;
