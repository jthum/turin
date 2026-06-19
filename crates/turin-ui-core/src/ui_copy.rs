pub fn unsupported_ui_source_message(surface: &str, source: &str, client: &str) -> String {
    let surface = surface.trim();
    let surface = if surface.is_empty() {
        "surface"
    } else {
        surface
    };
    let source = source.trim();
    if source.is_empty() {
        return format!(
            "This {surface} is declared and visible, but no source was provided. Add a worklists.<name> source or a deliberate adapter for this client."
        );
    }

    let client = client.trim();
    let client = if client.is_empty() {
        "this client"
    } else {
        client
    };
    format!(
        "This {surface} is declared and visible, but source '{source}' cannot load in {client} yet. Only named worklists.<name> sources load today; model this data as a worklist or add a deliberate adapter for this client."
    )
}

pub fn ui_data_not_loaded_message(surface: &str) -> String {
    let surface = surface.trim();
    let surface = if surface.is_empty() {
        "surface"
    } else {
        surface
    };
    format!(
        "This {surface} is visible, but its backing data has not loaded yet. It will appear after the client requests and receives the current data."
    )
}

pub fn ui_data_load_failed_message(surface: &str, error: &str) -> String {
    let surface = surface.trim();
    let surface = if surface.is_empty() {
        "surface"
    } else {
        surface
    };
    let error = error.trim();
    if error.is_empty() {
        return format!(
            "This {surface} is visible, but its backing data failed to load. Refresh the view to retry."
        );
    }
    format!(
        "This {surface} is visible, but its backing data failed to load: {error}. Refresh the view to retry."
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ui_data_load_failed_message, ui_data_not_loaded_message, unsupported_ui_source_message,
    };

    #[test]
    fn unsupported_source_message_names_surface_source_and_client() {
        let message = unsupported_ui_source_message("list", "tables.release", "the terminal");

        assert!(message.contains("This list is declared and visible"));
        assert!(message.contains("source 'tables.release'"));
        assert!(message.contains("cannot load in the terminal yet"));
        assert!(message.contains("Only named worklists.<name> sources load today"));
        assert!(message.contains("deliberate adapter for this client"));
    }

    #[test]
    fn unsupported_source_message_handles_missing_source() {
        let message = unsupported_ui_source_message("detail", "  ", "the desktop app");

        assert!(message.contains("This detail is declared and visible"));
        assert!(message.contains("no source was provided"));
        assert!(message.contains("worklists.<name> source"));
        assert!(!message.contains("the desktop app"));
    }

    #[test]
    fn data_not_loaded_message_names_surface_and_client_loading_flow() {
        let message = ui_data_not_loaded_message("report");

        assert!(message.contains("This report is visible"));
        assert!(message.contains("backing data has not loaded yet"));
        assert!(message.contains("client requests and receives"));
    }

    #[test]
    fn data_not_loaded_message_handles_missing_surface() {
        let message = ui_data_not_loaded_message(" ");

        assert!(message.contains("This surface is visible"));
    }

    #[test]
    fn data_load_failed_message_names_surface_error_and_retry() {
        let message = ui_data_load_failed_message("list", "worklist 'release' was not found");

        assert!(message.contains("This list is visible"));
        assert!(message.contains("backing data failed to load"));
        assert!(message.contains("worklist 'release' was not found"));
        assert!(message.contains("Refresh the view to retry"));
    }

    #[test]
    fn data_load_failed_message_handles_missing_surface_and_error() {
        let message = ui_data_load_failed_message(" ", " ");

        assert!(message.contains("This surface is visible"));
        assert!(message.contains("failed to load"));
        assert!(message.contains("Refresh the view to retry"));
    }
}
