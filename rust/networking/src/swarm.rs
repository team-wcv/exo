//! Compat shim for the old libp2p code

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use futures_lite::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use zenoh::Result;
use zenoh::Session;
use zenoh::handlers::FifoChannelHandler;
use zenoh::liveliness::LivelinessToken;
use zenoh::pubsub::Publisher;
use zenoh::pubsub::Subscriber;
use zenoh::qos::CongestionControl;
use zenoh::sample::Sample;
use zenoh::sample::SampleKind;

#[derive(Debug)]
pub enum ToSwarm {
    Unsubscribe {
        topic: String,
        result_sender: oneshot::Sender<bool>,
    },
    Subscribe {
        topic: String,
        result_sender: oneshot::Sender<Result<bool>>,
    },
    Publish {
        topic: String,
        data: Vec<u8>,
        result_sender: oneshot::Sender<Result<()>>,
    },
}
#[derive(Debug)]
pub enum FromSwarm {
    Message { topic: String, data: Vec<u8> },
    Discovered { peer_id: String },
    Expired { peer_id: String },
}

pub type Topics = HashMap<
    String,
    (
        Subscriber<FifoChannelHandler<Sample>>,
        Publisher<'static>,
        JoinHandle<()>,
    ),
>;
pub struct Swarm {
    pub session: crate::Session,
    pub from_client: mpsc::Receiver<ToSwarm>,
}

fn live_prefix(namespace_prefix: &str) -> String {
    format!("clusters/{namespace_prefix}/live/")
}

fn topic_key(namespace_prefix: &str, topic: &str) -> String {
    format!("clusters/{namespace_prefix}/topics/{topic}")
}

impl Swarm {
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = FromSwarm> + Send>> {
        let Swarm {
            session,
            mut from_client,
        } = self;
        let stream = async_stream::stream! {
            let mut session = session;
            let (mut to_topics, from_topics) = mpsc::channel(1024);
            let mut topics = Topics::new();
            let namespace_prefix = session.namespace_prefix.clone();
            let Ok((_token, discovery)) = register_liveness(&mut session.z, &namespace_prefix).await else { return; };
            let mut incoming = Box::pin(inbound_events(from_topics, discovery, namespace_prefix));
            loop {
                tokio::select! {
                    msg = from_client.recv() => {
                        let Some(msg) = msg else { break };
                        let operation = on_message(
                            &mut session.z,
                            &session.namespace_prefix,
                            &mut topics,
                            &mut to_topics,
                            msg,
                        );
                        // CongestionControl::Block can suspend a publish. Keep
                        // receiving acknowledgements and liveness while the
                        // operation waits, without reordering client commands.
                        let mut events = Box::pin(forward_events_during(operation, &mut incoming));
                        while let Some(event) = events.next().await {
                            yield event;
                        }
                    }
                    event = incoming.next() => {
                        let Some(event) = event else { break };
                        yield event;
                    }
                }
            }
        };
        Box::pin(stream)
    }
}

fn inbound_events(
    mut from_topics: mpsc::Receiver<FromSwarm>,
    discovery: Subscriber<FifoChannelHandler<Sample>>,
    namespace_prefix: String,
) -> impl Stream<Item = FromSwarm> + Send {
    async_stream::stream! {
        let prefix = live_prefix(&namespace_prefix);
        loop {
            tokio::select! {
                event = from_topics.recv() => {
                    let Some(event) = event else { break };
                    yield event;
                }
                token = discovery.recv_async() => {
                    let Ok(token) = token else { break };
                    if let Some(peer_id) = token.key_expr().as_str().strip_prefix(&prefix) {
                        let peer_id = peer_id.to_owned();
                        yield match token.kind() {
                            SampleKind::Put => {
                                log::info!("discovered: {peer_id}");
                                FromSwarm::Discovered { peer_id }
                            }
                            SampleKind::Delete => {
                                log::info!("expired: {peer_id}");
                                FromSwarm::Expired { peer_id }
                            }
                        };
                    }
                }
            }
        }
    }
}

fn forward_events_during<'a, F, S>(
    operation: F,
    incoming: &'a mut S,
) -> impl Stream<Item = FromSwarm> + Send + 'a
where
    F: Future<Output = ()> + Send + 'a,
    S: Stream<Item = FromSwarm> + Unpin + Send + 'a,
{
    async_stream::stream! {
        tokio::pin!(operation);
        loop {
            tokio::select! {
                () = &mut operation => break,
                event = incoming.next() => match event {
                    Some(event) => yield event,
                    None => {
                        // The transport cannot deliver further control traffic.
                        // Dropping the operation also closes its result sender,
                        // so the waiting Python caller observes cancellation.
                        break;
                    }
                }
            }
        }
    }
}

async fn register_liveness(
    session: &mut Session,
    namespace_prefix: &str,
) -> Result<(LivelinessToken, Subscriber<FifoChannelHandler<Sample>>)> {
    let token = session
        .liveliness()
        .declare_token(format!(
            "{}{}",
            live_prefix(namespace_prefix),
            session.zid()
        ))
        .await?;
    let sub = session
        .liveliness()
        .declare_subscriber(format!("{}*", live_prefix(namespace_prefix)))
        .history(true)
        .await?;
    Ok((token, sub))
}

async fn on_message(
    session: &mut Session,
    namespace_prefix: &str,
    topics: &mut Topics,
    to_topics: &mut mpsc::Sender<FromSwarm>,
    msg: ToSwarm,
) {
    match msg {
        ToSwarm::Publish {
            topic,
            data,
            result_sender,
        } => {
            let res = match topics.get(&topic) {
                Some(topic) => topic.1.put(data).await,
                None => {
                    // TODO: this should be an error but the python FromSwarm is somewhat nondeterministic
                    Ok(()) //Err("not subscribed to topic!".into()),
                }
            };
            _ = result_sender.send(res);
        }
        ToSwarm::Unsubscribe {
            topic,
            result_sender,
        } => {
            let Some((_, (subscriber, publisher, forwarder))) = topics.remove_entry(&topic) else {
                _ = result_sender.send(false);
                return;
            };
            forwarder.abort();
            _ = publisher.undeclare().await;
            _ = subscriber.undeclare().await;
            _ = result_sender.send(true);
        }
        ToSwarm::Subscribe {
            topic,
            result_sender,
        } => {
            assert!(topic.is_ascii());
            if topics.contains_key(&topic) {
                _ = result_sender.send(Ok(false));
                return;
            }

            let publisher_res = session
                .declare_publisher(topic_key(namespace_prefix, &topic))
                .congestion_control(CongestionControl::Block)
                .await;
            let publisher = match publisher_res {
                Ok(p) => p,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };

            let subscriber_res = session
                .declare_subscriber(topic_key(namespace_prefix, &topic))
                .allowed_origin(zenoh::sample::Locality::Remote)
                .await;
            let subscriber = match subscriber_res {
                Ok(s) => s,
                Err(e) => {
                    _ = result_sender.send(Err(e));
                    return;
                }
            };

            let handler = subscriber.handler().clone();
            let sender = to_topics.clone();
            let forward_topic = topic.clone();
            let forwarder = tokio::spawn(async move {
                while let Ok(sample) = handler.recv_async().await {
                    if sample.kind() != SampleKind::Put {
                        continue;
                    }
                    if sender
                        .send(FromSwarm::Message {
                            topic: forward_topic.clone(),
                            data: sample.payload().to_bytes().to_vec(),
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });

            assert!(
                topics
                    .insert(topic, (subscriber, publisher, forwarder))
                    .is_none()
            );
            _ = result_sender.send(Ok(true));
        }
    }
}

pub async fn create_swarm(
    identity: &str,
    namespace: &str,
    from_client: mpsc::Receiver<ToSwarm>,
    listen_port: u16,
    discovery_service_port: u16,
    connect_endpoints: Vec<String>,
) -> Result<Swarm> {
    let enable_discovery = should_enable_discovery(&connect_endpoints);
    let cfg = crate::cfg(identity, listen_port, &connect_endpoints)?;
    let session = crate::open_with_discovery(
        cfg,
        namespace,
        listen_port,
        discovery_service_port,
        enable_discovery,
    )
    .await?;
    Ok(Swarm {
        session,
        from_client,
    })
}

fn should_enable_discovery(connect_endpoints: &[String]) -> bool {
    connect_endpoints.is_empty()
}

#[cfg(test)]
mod tests {
    use super::{
        FromSwarm, forward_events_during, live_prefix, should_enable_discovery, topic_key,
    };
    use futures_lite::{StreamExt, stream};
    use std::time::Duration;
    use tokio::sync::oneshot;
    use tokio::time::timeout;

    #[tokio::test]
    async fn blocked_publish_still_delivers_incoming_acknowledgement() {
        let (release, wait) = oneshot::channel::<()>();
        let operation = async {
            wait.await.expect("publish released");
        };
        let mut incoming = stream::iter([FromSwarm::Message {
            topic: "global_events".to_owned(),
            data: b"ack".to_vec(),
        }]);
        let mut events = Box::pin(forward_events_during(operation, &mut incoming));
        let event = timeout(Duration::from_secs(1), events.next())
            .await
            .expect("incoming acknowledgement must not wait for publish");
        assert!(matches!(event, Some(FromSwarm::Message { data, .. }) if data == b"ack"));
        release.send(()).expect("publish still pending");
        assert!(
            timeout(Duration::from_secs(1), events.next())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn completed_publish_does_not_wait_for_incoming_traffic() {
        let mut incoming = stream::pending::<FromSwarm>();
        let mut events = Box::pin(forward_events_during(async {}, &mut incoming));
        assert!(
            timeout(Duration::from_secs(1), events.next())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn closed_inbound_cancels_pending_network_operation() {
        let (result_sender, result_receiver) = oneshot::channel::<()>();
        let operation = async move {
            let _result_sender = result_sender;
            std::future::pending::<()>().await;
        };
        let mut incoming = stream::empty::<FromSwarm>();
        let mut events = Box::pin(forward_events_during(operation, &mut incoming));
        assert!(
            timeout(Duration::from_secs(1), events.next())
                .await
                .expect("closed ingress must not wait for a blocked publish")
                .is_none()
        );
        drop(events);
        assert!(
            result_receiver.await.is_err(),
            "caller must observe cancellation"
        );
    }

    #[test]
    fn zenoh_keys_are_scoped_by_namespace() {
        assert_eq!(live_prefix("abc"), "clusters/abc/live/");
        assert_eq!(topic_key("abc", "events"), "clusters/abc/topics/events");
        assert_ne!(topic_key("abc", "events"), topic_key("def", "events"));
    }

    #[test]
    fn explicit_bootstrap_disables_ipv6_discovery() {
        assert!(should_enable_discovery(&[]));
        assert!(!should_enable_discovery(&[
            "tcp/192.0.2.10:52414".to_owned()
        ]));
    }
}
