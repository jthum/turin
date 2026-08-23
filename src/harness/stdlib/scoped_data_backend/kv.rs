use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StorePathScope, StoreSelector};

use super::{open_state_store, selector_scope_ref};

pub async fn kv_get_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
    store_selector: Option<&StoreSelector>,
    path_scope: StorePathScope,
) -> anyhow::Result<Option<String>> {
    let store = open_state_store(manager, store_selector, path_scope).await?;
    let scope = selector_scope_ref(selector)?;
    store.kv_get(&scope.scope_kind, &scope.scope_key, key).await
}

pub async fn kv_set_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
    value: &str,
    store_selector: Option<&StoreSelector>,
    path_scope: StorePathScope,
) -> anyhow::Result<()> {
    let store = open_state_store(manager, store_selector, path_scope).await?;
    let scope = selector_scope_ref(selector)?;
    store
        .kv_set(&scope.scope_kind, &scope.scope_key, key, value)
        .await
}

pub async fn kv_delete_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
    store_selector: Option<&StoreSelector>,
    path_scope: StorePathScope,
) -> anyhow::Result<()> {
    let store = open_state_store(manager, store_selector, path_scope).await?;
    let scope = selector_scope_ref(selector)?;
    store
        .kv_delete(&scope.scope_kind, &scope.scope_key, key)
        .await
}
