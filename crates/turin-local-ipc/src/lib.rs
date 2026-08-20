use std::io;
use std::path::{Path, PathBuf};

use tokio::io::{AsyncRead, AsyncWrite};

#[cfg(windows)]
use sha2::{Digest, Sha256};

pub const TRANSPORT_UNIX: &str = "unix";
pub const TRANSPORT_NAMED_PIPE: &str = "named_pipe";

pub trait LocalIpcStream: AsyncRead + AsyncWrite + Send + Unpin + 'static {}

impl<T> LocalIpcStream for T where T: AsyncRead + AsyncWrite + Send + Unpin + 'static {}

pub type BoxedLocalIpcStream = Box<dyn LocalIpcStream>;
pub type LocalIpcReadHalf = tokio::io::ReadHalf<BoxedLocalIpcStream>;
pub type LocalIpcWriteHalf = tokio::io::WriteHalf<BoxedLocalIpcStream>;

pub fn current_transport_name() -> &'static str {
    #[cfg(windows)]
    {
        TRANSPORT_NAMED_PIPE
    }

    #[cfg(not(windows))]
    {
        TRANSPORT_UNIX
    }
}

pub fn split(stream: BoxedLocalIpcStream) -> (LocalIpcReadHalf, LocalIpcWriteHalf) {
    tokio::io::split(stream)
}

pub fn resolve_endpoint(base: &Path, workspace_root: &str, value: &str) -> PathBuf {
    let workspace_root = Path::new(workspace_root);
    let workspace = if workspace_root.is_absolute() {
        workspace_root.to_path_buf()
    } else {
        base.join(workspace_root)
    };

    let path = Path::new(value);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };

    #[cfg(windows)]
    {
        if is_windows_named_pipe_path(&resolved) {
            return resolved;
        }
        return windows_named_pipe_path_for_seed(&resolved);
    }

    #[cfg(not(windows))]
    {
        resolved
    }
}

pub async fn connect(endpoint: &Path) -> io::Result<BoxedLocalIpcStream> {
    #[cfg(unix)]
    {
        let stream = tokio::net::UnixStream::connect(endpoint).await?;
        Ok(Box::new(stream))
    }

    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ClientOptions;

        let stream = ClientOptions::new().open(endpoint.as_os_str())?;
        Ok(Box::new(stream))
    }
}

pub async fn cleanup_stale_endpoint(endpoint: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        if !ensure_endpoint_is_socket_or_missing(endpoint).await? {
            return Ok(());
        }

        match tokio::net::UnixStream::connect(endpoint).await {
            Ok(_) => Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                format!(
                    "Local IPC endpoint '{}' is already in use",
                    endpoint.display()
                ),
            )),
            Err(_) => tokio::fs::remove_file(endpoint).await,
        }
    }

    #[cfg(not(unix))]
    {
        let _ = endpoint;
        Ok(())
    }
}

pub async fn remove_endpoint(endpoint: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        match ensure_endpoint_is_socket_or_missing(endpoint).await? {
            true => tokio::fs::remove_file(endpoint).await?,
            false => return Ok(()),
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        let _ = endpoint;
        Ok(())
    }
}

#[cfg(unix)]
async fn ensure_endpoint_is_socket_or_missing(endpoint: &Path) -> io::Result<bool> {
    use std::os::unix::fs::FileTypeExt;

    match tokio::fs::symlink_metadata(endpoint).await {
        Ok(metadata) if metadata.file_type().is_socket() => Ok(true),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Local IPC endpoint '{}' exists but is not a Unix socket",
                endpoint.display()
            ),
        )),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

pub struct LocalIpcListener {
    #[cfg(unix)]
    listener: tokio::net::UnixListener,
    #[cfg(windows)]
    endpoint: std::path::PathBuf,
    #[cfg(windows)]
    next_server: tokio::net::windows::named_pipe::NamedPipeServer,
}

#[cfg(windows)]
fn is_windows_named_pipe_path(path: &Path) -> bool {
    let raw = path.to_string_lossy();
    raw.starts_with(r"\\.\pipe\") || raw.starts_with("//./pipe/")
}

#[cfg(windows)]
fn windows_named_pipe_path_for_seed(seed: &Path) -> PathBuf {
    let raw = seed.to_string_lossy().replace('\\', "/");
    let leaf = seed
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("daemon");
    let slug: String = leaf
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() { "daemon" } else { slug };

    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    let digest = hasher.finalize();
    let mut short = String::new();
    for byte in &digest[..8] {
        use std::fmt::Write as _;
        let _ = write!(&mut short, "{:02x}", byte);
    }

    PathBuf::from(format!(r"\\.\pipe\turin-{}-{}", slug, short))
}

impl LocalIpcListener {
    pub fn bind(endpoint: &Path) -> io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let listener = tokio::net::UnixListener::bind(endpoint)?;
            if let Err(err) =
                std::fs::set_permissions(endpoint, std::fs::Permissions::from_mode(0o600))
            {
                drop(listener);
                let _ = std::fs::remove_file(endpoint);
                return Err(err);
            }
            Ok(Self { listener })
        }

        #[cfg(windows)]
        {
            let next_server = create_named_pipe_server(endpoint, true)?;
            return Ok(Self {
                endpoint: endpoint.to_path_buf(),
                next_server,
            });
        }
    }

    pub async fn accept(&mut self) -> io::Result<BoxedLocalIpcStream> {
        #[cfg(unix)]
        {
            let (stream, _) = self.listener.accept().await?;
            Ok(Box::new(stream))
        }

        #[cfg(windows)]
        {
            let server = std::mem::replace(
                &mut self.next_server,
                create_named_pipe_server(&self.endpoint, false)?,
            );
            server.connect().await?;
            return Ok(Box::new(server));
        }
    }
}

#[cfg(windows)]
fn create_named_pipe_server(
    endpoint: &Path,
    first_pipe_instance: bool,
) -> io::Result<tokio::net::windows::named_pipe::NamedPipeServer> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let mut options = ServerOptions::new();
    options.first_pipe_instance(first_pipe_instance);
    options.create(endpoint.as_os_str())
}

#[cfg(test)]
mod tests {
    use super::resolve_endpoint;
    use std::path::Path;

    #[cfg(unix)]
    use super::{LocalIpcListener, cleanup_stale_endpoint, remove_endpoint};
    #[cfg(not(windows))]
    use super::{TRANSPORT_UNIX, current_transport_name};
    #[cfg(unix)]
    use std::os::unix::fs::{PermissionsExt, symlink};
    #[cfg(not(windows))]
    use std::path::PathBuf;

    #[test]
    fn unix_resolution_uses_workspace_relative_seed() {
        let endpoint =
            resolve_endpoint(Path::new("/tmp/project"), "workspace", ".turin/daemon.sock");

        #[cfg(not(windows))]
        assert_eq!(
            endpoint,
            PathBuf::from("/tmp/project/workspace/.turin/daemon.sock")
        );
    }

    #[test]
    fn current_transport_matches_platform() {
        #[cfg(not(windows))]
        assert_eq!(current_transport_name(), TRANSPORT_UNIX);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_listener_endpoint_is_owner_only() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = dir.path().join("daemon.sock");
        let _listener = LocalIpcListener::bind(&endpoint).unwrap();

        let mode = std::fs::metadata(&endpoint).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stale_socket_cleanup_removes_only_socket_entries() {
        let dir = tempfile::tempdir().unwrap();
        let endpoint = dir.path().join("daemon.sock");
        let listener = LocalIpcListener::bind(&endpoint).unwrap();
        drop(listener);

        cleanup_stale_endpoint(&endpoint).await.unwrap();
        assert!(!endpoint.exists());

        std::fs::write(&endpoint, "keep").unwrap();
        let error = cleanup_stale_endpoint(&endpoint).await.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(std::fs::read_to_string(&endpoint).unwrap(), "keep");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn endpoint_removal_rejects_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let endpoint = dir.path().join("daemon.sock");
        std::fs::write(&target, "keep").unwrap();
        symlink(&target, &endpoint).unwrap();

        let error = remove_endpoint(&endpoint).await.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep");
        assert!(std::fs::symlink_metadata(&endpoint).is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn windows_resolution_derives_stable_named_pipe_endpoint() {
        let endpoint = resolve_endpoint(
            Path::new("C:\\projects\\turin"),
            "workspace",
            ".turin/daemon.sock",
        );

        assert!(
            endpoint
                .to_string_lossy()
                .starts_with(r"\\.\pipe\turin-daemon-")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_resolution_preserves_explicit_named_pipe_path() {
        let endpoint = resolve_endpoint(
            Path::new("C:\\projects\\turin"),
            "workspace",
            r"\\.\pipe\custom-turin",
        );

        assert_eq!(endpoint, PathBuf::from(r"\\.\pipe\custom-turin"));
    }
}
