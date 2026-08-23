use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Default)]
pub struct HarnessSourceOverlay {
    changes: BTreeMap<PathBuf, Option<String>>,
}

impl HarnessSourceOverlay {
    pub fn insert(&mut self, path: PathBuf, source: Option<String>) {
        self.changes.insert(path, source);
    }

    pub fn root_lua_paths(&self) -> impl Iterator<Item = (&Path, bool)> {
        self.changes.iter().filter_map(|(path, source)| {
            let is_root_lua = path
                .parent()
                .is_none_or(|parent| parent.as_os_str().is_empty())
                && path.extension().is_some_and(|extension| extension == "lua");
            is_root_lua.then_some((path.as_path(), source.is_some()))
        })
    }

    /// Iterates all candidate source changes supplied to an adapter validation pass.
    #[doc(hidden)]
    pub fn entries(&self) -> impl Iterator<Item = (&Path, Option<&str>)> {
        self.changes
            .iter()
            .map(|(path, source)| (path.as_path(), source.as_deref()))
    }

    pub fn path_exists(&self, root: &Path, path: &Path) -> bool {
        match self.lookup(root, path) {
            Some(source) => source.is_some(),
            None => path.is_file(),
        }
    }

    pub fn read_to_string(&self, root: &Path, path: &Path) -> io::Result<String> {
        match self.lookup(root, path) {
            Some(Some(source)) => Ok(source.clone()),
            Some(None) => Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("candidate source '{}' was deleted", path.display()),
            )),
            None => std::fs::read_to_string(path),
        }
    }

    fn lookup(&self, root: &Path, path: &Path) -> Option<&Option<String>> {
        let relative = path.strip_prefix(root).ok()?;
        self.changes.get(relative)
    }
}
