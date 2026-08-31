use crate::ext::MultiaddrExt;
use delegate::delegate;
use either::Either;
use futures_lite::FutureExt;
use futures_timer::Delay;
use libp2p::core::transport::PortUse;
use libp2p::core::{ConnectedPoint, Endpoint};
use libp2p::multiaddr::Protocol;
use libp2p::swarm::behaviour::ConnectionEstablished;
use libp2p::swarm::dial_opts::DialOpts;
use libp2p::swarm::{
    ConnectionClosed, ConnectionDenied, ConnectionHandler, ConnectionHandlerSelect, ConnectionId,
    FromSwarm, NetworkBehaviour, THandler, THandlerInEvent, THandlerOutEvent, ToSwarm, dummy,
};
use libp2p::{Multiaddr, PeerId, identity, mdns};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::convert::Infallible;
use std::io;
use std::net::IpAddr;
use std::task::{Context, Poll};
use std::time::Duration;
use util::wakerdeque::WakerDeque;

use bytemuck::{Pod, Zeroable};
use log::{debug, trace, warn};
use netwatcher::WatchHandle;
use parking_lot::Mutex;
use tokio::{
    net::UdpSocket,
    time::{Interval, interval},
};
use zenoh::config::ZenohId;

mod managed {
    use libp2p::swarm::NetworkBehaviour;
    use libp2p::{identity, mdns, ping};
    use std::env;
    use std::io;
    use std::time::Duration;

    const MDNS_RECORD_TTL: Duration = Duration::from_secs(2_500);
    const MDNS_QUERY_INTERVAL: Duration = Duration::from_secs(1_500);
    const DEFAULT_PING_TIMEOUT_MS: u64 = 15_000;
    const DEFAULT_PING_INTERVAL_MS: u64 = 5_000;
    const PING_TIMEOUT_MS_ENV: &str = "EXO_LIBP2P_PING_TIMEOUT_MS";
    const PING_INTERVAL_MS_ENV: &str = "EXO_LIBP2P_PING_INTERVAL_MS";

    #[derive(NetworkBehaviour)]
    pub struct Behaviour {
        mdns: mdns::tokio::Behaviour,
        ping: ping::Behaviour,
    }

    impl Behaviour {
        pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
            Ok(Self {
                mdns: mdns_behaviour(keypair)?,
                ping: ping_behaviour(),
            })
        }
    }

    fn mdns_behaviour(keypair: &identity::Keypair) -> io::Result<mdns::tokio::Behaviour> {
        use mdns::{Config, tokio};

        // mDNS config => enable IPv6
        let mdns_config = Config {
            ttl: MDNS_RECORD_TTL,
            query_interval: MDNS_QUERY_INTERVAL,

            // enable_ipv6: true, // TODO: for some reason, TCP+mDNS don't work well with ipv6?? figure out how to make work
            ..Default::default()
        };

        let mdns_behaviour = tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id());
        Ok(mdns_behaviour?)
    }

    fn ping_behaviour() -> ping::Behaviour {
        let timeout = Duration::from_millis(duration_millis_env(
            PING_TIMEOUT_MS_ENV,
            DEFAULT_PING_TIMEOUT_MS,
        ));
        let interval = Duration::from_millis(duration_millis_env(
            PING_INTERVAL_MS_ENV,
            DEFAULT_PING_INTERVAL_MS,
        ));
        ping::Behaviour::new(
            ping::Config::new()
                .with_timeout(timeout)
                .with_interval(interval),
        )
    }

    fn duration_millis_env(name: &str, default: u64) -> u64 {
        env::var(name)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Discovered {
    pub zid: ZenohId,
    pub addr: SocketAddrV6,
}

/// Discovery behavior that wraps mDNS to produce truly discovered durable peer-connections.
///
/// The behaviour operates as such:
///  1) All true (listening) connections/disconnections are tracked, emitting corresponding events
///     to the swarm.
///  1) mDNS discovered/expired peers are tracked; discovered but not connected peers are dialed
///     immediately, and expired but connected peers are disconnected from immediately.
///  2) Every fixed interval: discovered but not connected peers are dialed, and expired but
///     connected peers are disconnected from.
pub struct Behaviour {
    // state-tracking for managed behaviors & mDNS-discovered peers
    managed: managed::Behaviour,
    mdns_discovered: HashMap<PeerId, BTreeSet<Multiaddr>>,
    bootstrap_peers: Vec<Multiaddr>,
    connected_peers: HashMap<PeerId, HashSet<ConnectionId>>,

    retry_delay: Delay, // retry interval

    // pending events to emmit => waker-backed Deque to control polling
    pending_events: WakerDeque<ToSwarm<Event, Infallible>>,
}

impl Behaviour {
    pub fn new(keypair: &identity::Keypair, bootstrap_peers: Vec<Multiaddr>) -> io::Result<Self> {
        let mut behaviour = Self {
            managed: managed::Behaviour::new(keypair)?,
            mdns_discovered: HashMap::new(),
            bootstrap_peers,
            connected_peers: HashMap::new(),
            retry_delay: Delay::new(RETRY_CONNECT_INTERVAL),
            pending_events: WakerDeque::new(),
        };
        behaviour.dial_bootstrap_peers();
        Ok(behaviour)
    }

    fn dial(&mut self, peer_id: PeerId, addr: Multiaddr) {
        if self.is_connected(&peer_id) {
            return;
        }
        let addr = addr_without_peer_id(&addr);
        self.pending_events.push_back(ToSwarm::Dial {
            opts: DialOpts::peer_id(peer_id).addresses(vec![addr]).build(),
        })
    }

    fn dial_bootstrap_peers(&mut self) {
        for addr in self.bootstrap_peers.clone() {
            if peer_id_from_addr(&addr).is_some_and(|peer_id| self.is_connected(&peer_id)) {
                continue;
            }
            self.pending_events.push_back(ToSwarm::Dial {
                opts: DialOpts::unknown_peer_id().address(addr).build(),
            })
        }
    }

    fn is_connected(&self, peer_id: &PeerId) -> bool {
        self.connected_peers
            .get(peer_id)
            .is_some_and(|connection_ids| !connection_ids.is_empty())
    }

    fn redial_peer(&mut self, peer_id: PeerId) {
        let mut dialed = false;

        if let Some(addrs) = self.mdns_discovered.get(&peer_id).cloned() {
            for addr in addrs {
                self.dial(peer_id, addr);
                dialed = true;
            }
        }

        for addr in self.bootstrap_peers.clone() {
            if peer_id_from_addr(&addr).is_some_and(|candidate| candidate == peer_id) {
                self.dial(peer_id, addr);
                dialed = true;
            }
        }

        if !dialed {
            self.dial_bootstrap_peers();
        }
    }

    fn handle_mdns_discovered(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            self.mdns_discovered.entry(p).or_default().insert(ma.clone());
            self.dial(p, ma);
        }
    }

    fn handle_mdns_expired(&mut self, peers: Vec<(PeerId, Multiaddr)>) {
        for (p, ma) in peers {
            let should_remove = if let Some(mas) = self.mdns_discovered.get_mut(&p) {
                mas.remove(&ma);
                mas.is_empty()
            } else {
                false
            };
            if should_remove {
                self.mdns_discovered.remove(&p);
            }
        }
    }

    fn on_connection_established(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        let connection_ids = self.connected_peers.entry(peer_id).or_default();
        let was_disconnected = connection_ids.is_empty();
        connection_ids.insert(connection_id);
        if !was_disconnected {
            return;
        }

        // send out connected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionEstablished {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
    }

    fn on_connection_closed(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        remote_ip: IpAddr,
        remote_tcp_port: u16,
    ) {
        let Some(connection_ids) = self.connected_peers.get_mut(&peer_id) else {
            return;
        };
        connection_ids.remove(&connection_id);
        if !connection_ids.is_empty() {
            return;
        }
        self.connected_peers.remove(&peer_id);

        // send out disconnected event
        self.pending_events
            .push_back(ToSwarm::GenerateEvent(Event::ConnectionClosed {
                peer_id,
                connection_id,
                remote_ip,
                remote_tcp_port,
            }));
        self.redial_peer(peer_id);
    }
}

fn peer_id_from_addr(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|protocol| match protocol {
        Protocol::P2p(peer_id) => Some(peer_id),
        _ => None,
    })
}

fn addr_without_peer_id(addr: &Multiaddr) -> Multiaddr {
    addr.iter()
        .filter(|protocol| !matches!(protocol, Protocol::P2p(_)))
        .collect()
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler =
        ConnectionHandlerSelect<dummy::ConnectionHandler, THandler<managed::Behaviour>>;
    type ToSwarm = Event;

    // simply delegate to underlying mDNS behaviour

    delegate! {
        to self.managed {
            fn handle_pending_inbound_connection(&mut self, connection_id: ConnectionId, local_addr: &Multiaddr, remote_addr: &Multiaddr) -> Result<(), ConnectionDenied>;
            fn handle_pending_outbound_connection(&mut self, connection_id: ConnectionId, maybe_peer: Option<PeerId>, addresses: &[Multiaddr], effective_role: Endpoint) -> Result<Vec<Multiaddr>, ConnectionDenied>;
        }
    }

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_inbound_connection(
                connection_id,
                peer,
                local_addr,
                remote_addr,
            )?,
        ))
    }

    #[allow(clippy::needless_question_mark)]
    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        addr: &Multiaddr,
        role_override: Endpoint,
        port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(ConnectionHandler::select(
            dummy::ConnectionHandler,
            self.managed.handle_established_outbound_connection(
                connection_id,
                peer,
                addr,
                role_override,
                port_use,
            )?,
        ))
    }

    fn on_connection_handler_event(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        match event {
            Either::Left(ev) => libp2p::core::util::unreachable(ev),
            Either::Right(ev) => {
                self.managed
                    .on_connection_handler_event(peer_id, connection_id, ev)
            }
        }
    }

    // hook into these methods to drive behavior

    fn on_swarm_event(&mut self, event: FromSwarm) {
        self.managed.on_swarm_event(event); // let mDNS handle swarm events

        // handle swarm events to update internal state:
        match event {
            FromSwarm::ConnectionEstablished(ConnectionEstablished {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection established event which is filtered correctly
                    self.on_connection_established(peer_id, connection_id, ip, port)
                }
            }
            FromSwarm::ConnectionClosed(ConnectionClosed {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => {
                let remote_address = match endpoint {
                    ConnectedPoint::Dialer { address, .. } => address,
                    ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr,
                };

                if let Some((ip, port)) = remote_address.try_to_tcp_addr() {
                    // handle connection closed event which is filtered correctly
                    self.on_connection_closed(peer_id, connection_id, ip, port)
                }
            }

            // since we are running TCP/IP transport layer, we are assuming that
            // no address changes can occur, hence encountering one is a fatal error
            FromSwarm::AddressChange(a) => {
                unreachable!("unhandlable: address change encountered: {:?}", a)
            }
            _ => {}
        }
    }

    fn poll(&mut self, cx: &mut Context) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        // delegate to managed behaviors for any behaviors they need to perform
        match self.managed.poll(cx) {
            Poll::Ready(ToSwarm::GenerateEvent(e)) => {
                match e {
                    // handle discovered and expired events from mDNS
                    managed::BehaviourEvent::Mdns(e) => match e.clone() {
                        mdns::Event::Discovered(peers) => {
                            self.handle_mdns_discovered(peers);
                        }

                    // Let libp2p manage connection lifecycle. Treating a
                    // ping behaviour error as an immediate peer disconnect
                    // can tear down gossipsub during transient control-plane
                    // stalls while the peer is still reachable for exo/RDMA.
                    managed::BehaviourEvent::Ping(_) => {}
                }
            })
            // todo: better error handling here
            .expect("failed to bind discovery watcher"),
        );
        Ok(Self {
            sock,
            namespace,
            ifaces,
            last_nonce: Mutex::new(rand::random()),
            listen_port,
            zid,
            tick: interval(Duration::from_secs(1)),
            _sync,
        })
    }

    pub async fn next(&mut self) -> io::Result<Discovered> {
        let mut buf = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        loop {
            tokio::select! {
                _ = self.tick.tick() => {
                    self.announce().await?;
                }
                res = self.sock.recv_from(&mut buf) => {
                    let Ok((bytes_read, addr)) = res else { continue; };
                    if let Some(discovered) = self.respond(bytes_read, addr, &buf).await? {
                        return Ok(discovered)
                    }
                }
            }
            // dial bootstrap peers (for environments where mDNS is unavailable)
            self.dial_bootstrap_peers();
            self.retry_delay.reset(RETRY_CONNECT_INTERVAL) // reset timeout
        }
    }

    async fn announce(&self) -> io::Result<()> {
        let nonce = rand::random();
        *self.last_nonce.lock() = nonce;
        let buf = Hello {
            nonce,
            namespace: self.namespace,
        }
        .alloc();

        let addrs = self.ifaces.lock().clone();
        debug!("announcing Hello({nonce:?}) to {addrs:?}");
        // rev so .remove() doesn't break things
        for (i, addr) in addrs.into_iter().enumerate().rev() {
            match self.sock.send_to(&buf, addr).await {
                Ok(bytes) => trace!("sent {bytes} to {addr}"),
                Err(e) if e.kind() == io::ErrorKind::HostUnreachable => {
                    debug!("disabling discovery address {addr}: {e}");
                    _ = self.ifaces.lock().swap_remove(i);
                }
                Err(e) => debug!("failed to reach {addr}: {e}"),
            }
        }
        Ok(())
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
// packet & version
pub enum Kind {
    Hello = 0,
    WhatsUp = 1,
}

pub struct UnknownKind;
impl TryFrom<u8> for Kind {
    type Error = UnknownKind;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Hello),
            1 => Ok(Self::WhatsUp),
            _ => Err(UnknownKind),
        }
    }
}

pub trait Message: Pod {
    const KIND: Kind;
}
// should be part of the Message trait, but const in traits isnt stabilized. this lets alloc :: Self -> [u8; Self::buf_size()]
macro_rules! impl_alloc {
    ($a:ident) => {
        impl $a {
            const fn buf_size() -> usize {
                size_of::<Header>() + size_of::<Self>()
            }
            pub fn alloc(self) -> [u8; Self::buf_size()] {
                let mut buf = [0u8; Self::buf_size()];
                buf[0..size_of::<Header>()].copy_from_slice(bytemuck::bytes_of(&Header {
                    magic: MAGIC,
                    kind: Self::KIND as u8,
                }));
                buf[size_of::<Header>()..Self::buf_size()]
                    .copy_from_slice(bytemuck::bytes_of(&self));
                buf
            }
        }
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Header {
    magic: [u8; 3],
    kind: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Hello {
    pub nonce: [u8; 8],
    pub namespace: [u8; 8],
}
impl Message for Hello {
    const KIND: Kind = Kind::Hello;
}
impl_alloc!(Hello);

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WhatsUp {
    pub nonce: [u8; 8],
    pub zid: [u8; 16],
    pub port_le: [u8; 2],
}
impl Message for WhatsUp {
    const KIND: Kind = Kind::WhatsUp;
}
impl_alloc!(WhatsUp);
