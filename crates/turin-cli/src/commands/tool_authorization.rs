use std::io::{self, BufRead, Write};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use turin::display;
use turin::kernel::builder::RuntimeBuilder;
use turin::kernel::tool_authorization::{
    ToolAuthorizationDecision, ToolAuthorizationFuture, ToolAuthorizationRequest, ToolAuthorizer,
};

#[derive(Debug)]
struct InteractiveToolAuthorizer;

impl ToolAuthorizer for InteractiveToolAuthorizer {
    fn authorize(
        &self,
        request: ToolAuthorizationRequest,
        _cancellation: CancellationToken,
    ) -> ToolAuthorizationFuture {
        Box::pin(async move {
            tokio::task::block_in_place(|| {
                let ansi = display::stderr_ansi();
                eprint!(
                    "{} {} Allow '{}' with arguments {}? (y/n): ",
                    display::approval_prompt_prefix(ansi),
                    request.reason,
                    request.tool_name,
                    request.arguments,
                );
                io::stderr().flush().ok();

                let mut input = String::new();
                if io::stdin().lock().read_line(&mut input).is_ok()
                    && input.trim().eq_ignore_ascii_case("y")
                {
                    ToolAuthorizationDecision::Approve
                } else {
                    ToolAuthorizationDecision::deny(None)
                }
            })
        })
    }
}

pub(crate) fn with_interactive_authorization(builder: RuntimeBuilder) -> RuntimeBuilder {
    builder.with_tool_authorizer(Arc::new(InteractiveToolAuthorizer))
}
