//! Remote editing over a slop2 daemon.
//!
//! The project host is the same `remote_server proxy` every other transport
//! runs; only the pipe to it differs. Instead of `ssh <host> …` this spawns
//! `slop2-zed-proxy`, which opens a tunnel on a slop2 daemon (a local unix
//! socket, or TCP to a daemon in tailnet mode) and bridges its own stdio. The
//! daemon starts `remote_server proxy` on ITS machine and pipes it back, so from
//! here the remote server looks like a local child process — which is exactly
//! what `handle_rpc_messages_over_child_process_stdio` wants.
//!
//! Why this exists rather than reusing the SSH transport: slop2 already holds an
//! authenticated connection to that machine (peer creds locally, `tailscale
//! whois` over the tailnet), so there is no second credential to manage, and the
//! daemon owns the server binary's lifetime the way it owns PTYs and
//! compositors. The Zed UI still runs locally, because rendering needs a local
//! dma-buf.
//!
//! Deliberately narrower than SSH, and the trait methods say so rather than
//! pretending: the daemon supplies the remote server binary (so there is nothing
//! to upload), and running arbitrary remote commands — Zed's remote terminals,
//! task spawning, port forwarding — has no route through the tunnel yet.

use crate::{
    RemoteArch, RemoteClientDelegate, RemoteOs, RemotePlatform,
    remote_client::{CommandTemplate, Interactive, RemoteConnection, RemoteConnectionOptions},
};
use anyhow::{Result, bail};
use async_trait::async_trait;
use collections::HashMap;
use futures::channel::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use gpui::{App, AsyncApp, Task};
use parking_lot::Mutex;
use rpc::proto::Envelope;
use std::path::PathBuf;
use std::sync::Arc;
use util::command::Stdio;
use util::paths::{PathStyle, RemotePathBuf};

/// How to reach the slop2 daemon that hosts the project.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct Slop2ConnectionOptions {
    /// The remote's name in slop2's client config — also what the UI shows.
    pub name: String,
    /// Hostname or tailnet address of the daemon. `None` means the local
    /// daemon (its unix socket).
    pub host: Option<String>,
    /// Port for `host`. `None` uses the daemon's default (7767).
    pub port: Option<u16>,
    /// An explicit unix socket path, for a daemon that isn't at the default
    /// location. Ignored when `host` is set.
    pub socket: Option<String>,
}

impl Slop2ConnectionOptions {
    /// The label used in the UI and in log lines.
    pub fn display_name(&self) -> String {
        if self.name.is_empty() {
            self.target_description()
        } else {
            self.name.clone()
        }
    }

    fn target_description(&self) -> String {
        match (&self.host, self.port, &self.socket) {
            (Some(host), Some(port), _) => format!("{host}:{port}"),
            (Some(host), None, _) => host.clone(),
            (None, _, Some(socket)) => socket.clone(),
            (None, _, None) => "local daemon".to_string(),
        }
    }

    /// Arguments that point `slop2-zed-proxy` at this daemon.
    fn proxy_target_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(host) = &self.host {
            args.push("--host".into());
            args.push(host.clone());
            if let Some(port) = self.port {
                args.push("--port".into());
                args.push(port.to_string());
            }
        } else if let Some(socket) = &self.socket {
            args.push("--socket".into());
            args.push(socket.clone());
        }
        args
    }
}

/// The bridge binary. `$SLOP2_ZED_PROXY_BIN` wins so a dev build can point at
/// `slop2/target/debug/slop2-zed-proxy`, which is on nobody's PATH.
fn proxy_binary() -> PathBuf {
    if let Some(path) = std::env::var_os("SLOP2_ZED_PROXY_BIN").filter(|v| !v.is_empty()) {
        return PathBuf::from(path);
    }
    PathBuf::from("slop2-zed-proxy")
}

pub(crate) struct Slop2RemoteConnection {
    connection_options: Slop2ConnectionOptions,
    /// PID of the running bridge, so `kill` can end the session.
    proxy_process: Mutex<Option<u32>>,
    killed: Mutex<bool>,
}

impl Slop2RemoteConnection {
    pub(crate) async fn new(
        connection_options: Slop2ConnectionOptions,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Result<Self> {
        log::info!(
            "connecting to slop2 daemon {}",
            connection_options.target_description()
        );
        delegate.set_status(Some("Checking the slop2 bridge"), cx);

        // Fail here, with a sentence the user can act on, rather than after Zed
        // has built a session around a bridge that was never going to run. The
        // bridge has no --version flag to probe (it would need a daemon to talk
        // to), so this is a plain "can we execute it" check.
        let bin = proxy_binary();
        let resolvable = if bin.components().count() > 1 {
            bin.exists()
        } else {
            // Bare name: leave PATH resolution to the spawn, which reports it.
            true
        };
        if !resolvable {
            bail!(
                "{} does not exist. Build slop2 and put slop2-zed-proxy on PATH, \
                 or point $SLOP2_ZED_PROXY_BIN at it.",
                bin.display()
            );
        }

        Ok(Self {
            connection_options,
            proxy_process: Mutex::new(None),
            killed: Mutex::new(false),
        })
    }

    fn kill_inner(&self) -> Result<()> {
        *self.killed.lock() = true;
        if let Some(pid) = self.proxy_process.lock().take() {
            // Killing the bridge drops its tunnel, and that is what makes the
            // daemon stop the remote server (see slop2's `ZedProxies`). Same
            // `kill(1)` the docker transport uses, so no new dependency.
            if util::command::new_command("kill")
                .arg(pid.to_string())
                .spawn()
                .is_err()
            {
                anyhow::bail!("failed to kill the slop2 bridge (pid {pid})");
            }
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl RemoteConnection for Slop2RemoteConnection {
    fn start_proxy(
        &self,
        unique_identifier: String,
        reconnect: bool,
        incoming_tx: UnboundedSender<Envelope>,
        outgoing_rx: UnboundedReceiver<Envelope>,
        connection_activity_tx: Sender<()>,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Task<Result<i32>> {
        delegate.set_status(Some("Starting proxy"), cx);

        let mut args = vec!["--identifier".to_string(), unique_identifier];
        if reconnect {
            args.push("--reconnect".to_string());
        }
        args.extend(self.connection_options.proxy_target_args());

        let mut command = util::command::new_command(proxy_binary());
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .args(args);

        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return Task::ready(Err(anyhow::anyhow!(
                    "failed to start the slop2 bridge: {error}"
                )));
            }
        };
        *self.proxy_process.lock() = Some(child.id());

        cx.spawn(async move |cx| {
            super::handle_rpc_messages_over_child_process_stdio(
                child,
                incoming_tx,
                outgoing_rx,
                connection_activity_tx,
                cx,
            )
            .await
            .and_then(|status| {
                // The bridge exits with the REMOTE server's status, so a nonzero
                // code here means the project host died, not the transport.
                if status != 0 {
                    anyhow::bail!("remote server exited with status {status}");
                }
                Ok(0)
            })
        })
    }

    fn upload_directory(
        &self,
        _src_path: PathBuf,
        _dest_path: RemotePathBuf,
        _cx: &App,
    ) -> Task<Result<()>> {
        // Nothing to upload: the daemon resolves and runs its own
        // `remote_server`, and the two halves version-check each other over the
        // tunnel. Dev extensions and binary upload would need a slop2 file
        // transfer RPC.
        Task::ready(Err(anyhow::anyhow!(
            "uploading directories is not supported over a slop2 connection: \
             the daemon provides its own remote server binary"
        )))
    }

    async fn kill(&self) -> Result<()> {
        self.kill_inner()
    }

    fn has_been_killed(&self) -> bool {
        *self.killed.lock()
    }

    fn build_command(
        &self,
        _program: Option<String>,
        _args: &[String],
        _env: &HashMap<String, String>,
        _working_dir: Option<String>,
        _port_forward: Option<(u16, String, u16)>,
        _interactive: Interactive,
    ) -> Result<CommandTemplate> {
        // Zed builds these for remote terminals and tasks. slop2 has its own
        // remote terminals (that is the whole app around this pane), and the
        // tunnel carries exactly one thing: the remote server's protocol. An
        // explicit error beats a command that silently runs on the wrong machine.
        bail!(
            "running commands on the remote host is not supported over a slop2 \
             connection; use a slop2 terminal tab on that daemon instead"
        )
    }

    fn build_forward_ports_command(
        &self,
        _forwards: Vec<(u16, String, u16)>,
    ) -> Result<CommandTemplate> {
        bail!("port forwarding is not supported over a slop2 connection")
    }

    fn connection_options(&self) -> RemoteConnectionOptions {
        RemoteConnectionOptions::Slop2(self.connection_options.clone())
    }

    fn path_style(&self) -> PathStyle {
        // The daemon reports its platform in its handshake, but that happens
        // inside the bridge process; until that is surfaced, slop2 daemons are
        // POSIX (the Windows port has no remote server yet).
        PathStyle::Unix
    }

    fn remote_platform(&self) -> RemotePlatform {
        RemotePlatform {
            os: RemoteOs::Linux,
            arch: if cfg!(target_arch = "aarch64") {
                RemoteArch::Aarch64
            } else {
                RemoteArch::X86_64
            },
        }
    }

    fn remote_os_version(&self) -> Option<String> {
        None
    }

    fn shell(&self) -> String {
        "/bin/sh".to_string()
    }

    fn default_system_shell(&self) -> String {
        "/bin/sh".to_string()
    }

    fn has_wsl_interop(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_args_prefer_host_then_socket() {
        let tailnet = Slop2ConnectionOptions {
            name: "desktop".into(),
            host: Some("100.64.0.24".into()),
            port: Some(7767),
            socket: Some("/ignored".into()),
        };
        assert_eq!(
            tailnet.proxy_target_args(),
            vec!["--host", "100.64.0.24", "--port", "7767"]
        );

        let socket = Slop2ConnectionOptions {
            name: "local".into(),
            host: None,
            port: None,
            socket: Some("/run/user/1000/slop2/daemon.sock".into()),
        };
        assert_eq!(
            socket.proxy_target_args(),
            vec!["--socket", "/run/user/1000/slop2/daemon.sock"]
        );

        // Nothing configured: the bridge falls back to the local daemon itself.
        let default = Slop2ConnectionOptions {
            name: "local".into(),
            host: None,
            port: None,
            socket: None,
        };
        assert!(default.proxy_target_args().is_empty());
    }

    #[test]
    fn display_name_falls_back_to_the_target() {
        let unnamed = Slop2ConnectionOptions {
            name: String::new(),
            host: Some("host".into()),
            port: Some(1234),
            socket: None,
        };
        assert_eq!(unnamed.display_name(), "host:1234");
        let named = Slop2ConnectionOptions {
            name: "desktop".into(),
            ..unnamed
        };
        assert_eq!(named.display_name(), "desktop");
    }
}
