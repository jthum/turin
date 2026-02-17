use anyhow::Result;
use turin::tools::{is_safe_path, ToolError};
use proptest::prelude::*;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

proptest! {
    #[test]
    fn test_is_safe_path_properties(path_str in ".*") {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        let path = Path::new(&path_str);

        let result = is_safe_path(root, path);

        if result.is_ok() {
            let res_path = result.unwrap();
            // Property 1: The resolved path must actually start with the root (canonicalized)
            let canonical_root = root.canonicalize().unwrap();
            
            // If it exists, we can canonicalize it and check
            if res_path.exists() {
                let canonical_res = res_path.canonicalize().unwrap();
                prop_assert!(canonical_res.starts_with(&canonical_root));
            } else {
                // If it doesn't exist, check against existing ancestors
                let mut current = res_path.clone();
                while !current.exists() {
                    if let Some(parent) = current.parent() {
                        current = parent.to_path_buf();
                    } else {
                        break;
                    }
                }
                let canonical_current = current.canonicalize().unwrap();
                prop_assert!(canonical_current.starts_with(&canonical_root));
            }

            // Property 2: We can verify it's not a traversal by checking components
            // if we really want to, but starts_with covers the security property.
        } else {
            // If it's blocked, it should be a PermissionDenied error if it was a traversal attempt
            if let Err(ToolError::PermissionDenied(_)) = result {
                // Good
            }
        }
    }
}

#[test]
fn test_is_safe_path_known_attacks() {
    let tmp = tempdir().unwrap();
    let root = tmp.path();
    
    // Explicit traversal
    assert!(is_safe_path(root, Path::new("../etc/passwd")).is_err());
    assert!(is_safe_path(root, Path::new("foo/../../etc/passwd")).is_err());
    
    // Absolute path outside root
    assert!(is_safe_path(root, Path::new("/etc/passwd")).is_err());
    
    // Valid path
    assert!(is_safe_path(root, Path::new("docs/manual.md")).is_ok());
    
    // Path with dots but not traversal
    assert!(is_safe_path(root, Path::new(".hidden")).is_ok());
}
