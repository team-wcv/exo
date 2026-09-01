use std::sync::Arc;

use tokio::task::JoinHandle;
use zenoh::{Result, Session as ZSession, config::Locator};
use zenoh_plugin_storage_manager::StoragesPlugin;
use zenoh_plugin_trait::PluginsManager;

pub use zenoh::{Config, config::ZenohId};

use crate::discovery::Discovery;

pub mod discovery;
pub mod swarm;

pub fn is_valid_zid(identity: &str) -> bool {
    let mut iter = identity.chars();
    iter.next()
        .is_some_and(|c| ('1'..='9').contains(&c) || ('a'..='f').contains(&c))
        && iter.all(|c| ('0'..='9').contains(&c) || ('a'..='f').contains(&c))
        && identity.len() <= 32
}

fn namespace_prefix(namespace: &str) -> String {
    blake3::hash(namespace.as_bytes()).as_bytes()[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn cfg(
    identity: &str,
    listen_port: u16,
    connect_endpoints: &[String],
) -> Result<zenoh::Config> {
    assert!(is_valid_zid(identity));
    assert!(identity.len() <= 32);
    assert!(listen_port != 0, "must used defined listen port");
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity}\""))?;
    cfg.insert_json5("mode", "\"router\"")?;
    cfg.insert_json5("listen/endpoints", &format!("[\"tcp/[::]:{listen_port}\"]"))?;
    if !connect_endpoints.is_empty() {
        cfg.insert_json5(
            "connect/endpoints",
            &serde_json::to_string(connect_endpoints)?,
        )?;
    }
    cfg.insert_json5("scouting/multicast/enabled", "false")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    cfg.insert_json5("adminspace/enabled", "true")?;
    //cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
    cfg.insert_json5("transport/link/rx/buffer_size", "16777216")?;
    //cfg.insert_json5("timestamping/enabled", "true")?;
    cfg.insert_json5("plugins/storage_manager/__required__", "true")?;
    cfg.insert_json5(
        "plugins/storage_manager/storages/mem1",
        r#"{
            key_expr: "storage/mem1/**",
            strip_prefix: "storage/mem1",
            volume: "memory",
            replication: {
                interval: 2,
            }
        }"#,
    )?;
    Ok(cfg)
}

pub async fn open(
    cfg: zenoh::Config,
    namespace: &str,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Session> {
    open_with_discovery(cfg, namespace, listen_port, discovery_service_port, true).await
}

pub async fn open_with_discovery(
    cfg: zenoh::Config,
    namespace: &str,
    listen_port: u16,
    discovery_service_port: u16,
    enable_discovery: bool,
) -> Result<Session> {
    assert!(listen_port != 0, "must used defined listen port");
    let namespace_hash: [u8; 8] = {
        blake3::hash(namespace.as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("8 is equal to 8")
    };
    let namespace_prefix = namespace_prefix(namespace);
    let mut plugins = PluginsManager::static_plugins_only();
    plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .plugins_manager(plugins)
        .build()
        .await?;
    let z = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    let discovery_task = if enable_discovery {
        let mut discovery =
            Discovery::new(z.zid(), namespace_hash, listen_port, discovery_service_port).await?;
        Some(Arc::new(AbortOnDrop(tokio::task::spawn(async move {
            loop {
                let Ok(discovered) = discovery.next().await.inspect_err(|e| {
                    log::warn!("discovery error {e}");
                }) else {
                    continue;
                };

                if discovered.zid > runtime.zid() {
                    log::debug!("not connecting to peer with greater zid");
                    continue;
                }

                let Ok(locator) =
                    Locator::new("tcp", discovered.addr.to_string(), "").inspect_err(|e| {
                        log::warn!("failed to parse locator from addr: {e}");
                    })
                else {
                    continue;
                };

                runtime
                    .connect_peer(&discovered.zid.into(), &[locator])
                    .await;
            }
        }))))
    } else {
        log::info!("IPv6 discovery disabled because explicit bootstrap endpoints are configured");
        None
    };
    Ok(Session {
        z,
        namespace_prefix,
        _discovery_task: discovery_task,
    })
}

struct AbortOnDrop(JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Clone)]
pub struct Session {
    pub z: ZSession,
    pub namespace_prefix: String,
    _discovery_task: Option<Arc<AbortOnDrop>>,
}

#[cfg(test)]
mod tests {
    use super::namespace_prefix;

    #[test]
    fn namespace_prefix_is_stable_and_isolated() {
        let first = namespace_prefix("private-cluster");
        assert_eq!(first, namespace_prefix("private-cluster"));
        assert_ne!(first, namespace_prefix("other-cluster"));
        assert_eq!(first.len(), 16);
        assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
    }
}
